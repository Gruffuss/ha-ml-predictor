#!/bin/bash
# Automated Environment Deployment Script
# Deploys HA ML Predictor to different environments with proper validation

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENVIRONMENT="production"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Show usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS] [ENVIRONMENT]

Deploy HA ML Predictor to specified environment.

ENVIRONMENT:
    development     Deploy to development environment
    testing         Deploy to testing environment  
    staging         Deploy to staging environment
    production      Deploy to production environment (default)

OPTIONS:
    -h, --help              Show this help message
    -v, --validate-only     Only validate configuration, don't deploy
    -b, --build-only        Build images only, don't start services
    -t, --test-connections  Test external connections during validation
    -s, --skip-backup       Skip creating backup before deployment
    -f, --force            Force deployment even if validation fails
    --dry-run              Show what would be deployed without executing

Examples:
    $0 production                    Deploy to production
    $0 staging --validate-only       Validate staging configuration
    $0 development --test-connections Deploy dev with connection tests
    $0 production --skip-backup      Deploy production without backup

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="$DEFAULT_ENVIRONMENT"
    VALIDATE_ONLY=false
    BUILD_ONLY=false
    TEST_CONNECTIONS=false
    SKIP_BACKUP=false
    FORCE=false
    DRY_RUN=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -h|--help)
                usage
                exit 0
                ;;
            -v|--validate-only)
                VALIDATE_ONLY=true
                shift
                ;;
            -b|--build-only)
                BUILD_ONLY=true
                shift
                ;;
            -t|--test-connections)
                TEST_CONNECTIONS=true
                shift
                ;;
            -s|--skip-backup)
                SKIP_BACKUP=true
                shift
                ;;
            -f|--force)
                FORCE=true
                shift
                ;;
            --dry-run)
                DRY_RUN=true
                shift
                ;;
            development|testing|staging|production)
                ENVIRONMENT="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                usage
                exit 1
                ;;
        esac
    done
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."

    # Check if Docker is installed and running
    if ! command -v docker &> /dev/null; then
        log_error "Docker is not installed"
        exit 1
    fi

    if ! docker info &> /dev/null; then
        log_error "Docker is not running"
        exit 1
    fi

    # Check if Docker Compose is available
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is not installed"
        exit 1
    fi

    # Check if Python is available for configuration management
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi

    # Check if we're in the right directory
    if [[ ! -f "$PROJECT_ROOT/docker/docker-compose.yml" ]]; then
        log_error "docker-compose.yml not found. Are you in the right directory?"
        exit 1
    fi

    log_success "Prerequisites check completed"
}

# Validate environment configuration
validate_configuration() {
    log_info "Validating $ENVIRONMENT environment configuration..."

    cd "$PROJECT_ROOT"
    
    local python_cmd="python3 scripts/environment-manager.py validate $ENVIRONMENT"
    if [[ "$TEST_CONNECTIONS" == true ]]; then
        python_cmd="$python_cmd --test-connections"
    fi

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would execute: $python_cmd"
        return 0
    fi

    if $python_cmd; then
        log_success "Configuration validation passed"
        return 0
    else
        log_error "Configuration validation failed"
        if [[ "$FORCE" == true ]]; then
            log_warning "Continuing deployment due to --force flag"
            return 0
        else
            log_error "Use --force to deploy anyway, or fix configuration issues"
            exit 1
        fi
    fi
}

# Create backup before deployment
create_backup() {
    if [[ "$SKIP_BACKUP" == true ]] || [[ "$BUILD_ONLY" == true ]] || [[ "$VALIDATE_ONLY" == true ]]; then
        return 0
    fi

    log_info "Creating backup before deployment..."

    cd "$PROJECT_ROOT"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would create backup for $ENVIRONMENT"
        return 0
    fi

    if python3 scripts/environment-manager.py backup "$ENVIRONMENT" --type all; then
        log_success "Backup created successfully"
    else
        log_warning "Backup creation failed, continuing deployment"
    fi
}

# Stop running services
stop_services() {
    log_info "Stopping existing services..."

    cd "$PROJECT_ROOT/docker"
    
    local compose_files="-f docker-compose.yml"
    
    case "$ENVIRONMENT" in
        production)
            compose_files="$compose_files -f docker-compose.prod.yml"
            ;;
        staging)
            compose_files="$compose_files -f docker-compose.staging.yml"
            ;;
        development)
            compose_files="$compose_files -f docker-compose.development.yml"
            ;;
    esac

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would stop services with: docker-compose $compose_files down"
        return 0
    fi

    # Stop services gracefully
    if docker-compose $compose_files down --timeout 30; then
        log_success "Services stopped successfully"
    else
        log_warning "Some services may not have stopped gracefully"
    fi
}

# Deploy environment
deploy_environment() {
    if [[ "$VALIDATE_ONLY" == true ]]; then
        log_success "Validation completed successfully"
        return 0
    fi

    log_info "Deploying $ENVIRONMENT environment..."

    cd "$PROJECT_ROOT/docker"

    local compose_files="-f docker-compose.yml"
    local deploy_command="up -d --build"
    
    if [[ "$BUILD_ONLY" == true ]]; then
        deploy_command="build"
    fi

    case "$ENVIRONMENT" in
        production)
            compose_files="$compose_files -f docker-compose.prod.yml"
            ;;
        staging)
            compose_files="$compose_files -f docker-compose.staging.yml"
            ;;
        development)
            compose_files="$compose_files -f docker-compose.development.yml"
            ;;
    esac

    if [[ "$DRY_RUN" == true ]]; then
        log_info "DRY RUN: Would deploy with: docker-compose $compose_files $deploy_command"
        return 0
    fi

    log_info "Executing: docker-compose $compose_files $deploy_command"
    
    if docker-compose $compose_files $deploy_command; then
        if [[ "$BUILD_ONLY" == true ]]; then
            log_success "Images built successfully for $ENVIRONMENT environment"
        else
            log_success "Deployment completed successfully for $ENVIRONMENT environment"
            
            # Show service status
            log_info "Service status:"
            docker-compose $compose_files ps
        fi
    else
        log_error "Deployment failed"
        exit 1
    fi
}

# Wait for services to be healthy
wait_for_health() {
    if [[ "$VALIDATE_ONLY" == true ]] || [[ "$BUILD_ONLY" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    log_info "Waiting for services to become healthy..."

    local max_wait=300  # 5 minutes
    local wait_time=0
    local check_interval=10

    cd "$PROJECT_ROOT/docker"

    local compose_files="-f docker-compose.yml"
    
    case "$ENVIRONMENT" in
        production)
            compose_files="$compose_files -f docker-compose.prod.yml"
            ;;
        staging)
            compose_files="$compose_files -f docker-compose.staging.yml"
            ;;
        development)
            compose_files="$compose_files -f docker-compose.development.yml"
            ;;
    esac

    while [[ $wait_time -lt $max_wait ]]; do
        local unhealthy_services=$(docker-compose $compose_files ps --filter health=unhealthy --quiet | wc -l)
        local starting_services=$(docker-compose $compose_files ps --filter health=starting --quiet | wc -l)
        
        if [[ $unhealthy_services -eq 0 ]] && [[ $starting_services -eq 0 ]]; then
            log_success "All services are healthy"
            return 0
        fi

        log_info "Waiting for services to be healthy... ($wait_time/$max_wait seconds)"
        sleep $check_interval
        wait_time=$((wait_time + check_interval))
    done

    log_warning "Some services may not be healthy after $max_wait seconds"
    docker-compose $compose_files ps
}

# Validate deployment
validate_deployment() {
    if [[ "$VALIDATE_ONLY" == true ]] || [[ "$BUILD_ONLY" == true ]] || [[ "$DRY_RUN" == true ]]; then
        return 0
    fi

    log_info "Validating deployment..."

    # Test API endpoint if available
    local api_port
    case "$ENVIRONMENT" in
        development)
            api_port="8000"
            ;;
        staging)
            api_port="8000"
            ;;
        production)
            api_port="8000"
            ;;
        *)
            api_port="8000"
            ;;
    esac

    # Wait a bit for services to fully start
    sleep 10

    if curl -f -s "http://localhost:$api_port/health" > /dev/null; then
        log_success "API health check passed"
    else
        log_warning "API health check failed - service may still be starting"
    fi

    # Show final service status
    cd "$PROJECT_ROOT/docker"
    
    local compose_files="-f docker-compose.yml"
    
    case "$ENVIRONMENT" in
        production)
            compose_files="$compose_files -f docker-compose.prod.yml"
            ;;
        staging)
            compose_files="$compose_files -f docker-compose.staging.yml"
            ;;
        development)
            compose_files="$compose_files -f docker-compose.development.yml"
            ;;
    esac

    log_info "Final service status:"
    docker-compose $compose_files ps
}

# Print deployment summary
print_summary() {
    log_info "Deployment Summary"
    echo "=================="
    echo "Environment: $ENVIRONMENT"
    echo "Validate Only: $VALIDATE_ONLY"
    echo "Build Only: $BUILD_ONLY"
    echo "Test Connections: $TEST_CONNECTIONS"
    echo "Skip Backup: $SKIP_BACKUP"
    echo "Force: $FORCE"
    echo "Dry Run: $DRY_RUN"
    echo ""

    if [[ "$DRY_RUN" == true ]]; then
        log_info "This was a dry run. No actual changes were made."
        return 0
    fi

    if [[ "$VALIDATE_ONLY" == true ]]; then
        log_success "Configuration validation completed successfully"
        return 0
    fi

    if [[ "$BUILD_ONLY" == true ]]; then
        log_success "Docker images built successfully"
        return 0
    fi

    log_success "$ENVIRONMENT environment deployed successfully!"
    
    # Show access information
    case "$ENVIRONMENT" in
        development)
            echo "Access URLs:"
            echo "  API: http://localhost:8000"
            echo "  API Docs: http://localhost:8000/docs"
            echo "  PgAdmin: http://localhost:8080 (admin@example.com/admin)"
            echo "  Redis Commander: http://localhost:8081"
            ;;
        staging)
            echo "Access URLs:"
            echo "  API: http://localhost:8000"
            echo "  Grafana: http://localhost:3000"
            echo "  Prometheus: http://localhost:9090"
            ;;
        production)
            echo "Services are running in production mode"
            echo "Check service status with: docker-compose -f docker-compose.yml -f docker-compose.prod.yml ps"
            ;;
    esac
}

# Main execution
main() {
    log_info "HA ML Predictor Environment Deployment Script"
    log_info "=============================================="

    parse_args "$@"
    check_prerequisites
    validate_configuration
    create_backup
    
    if [[ "$VALIDATE_ONLY" != true ]]; then
        stop_services
        deploy_environment
        wait_for_health
        validate_deployment
    fi

    print_summary
}

# Handle script interruption
cleanup() {
    log_warning "Deployment interrupted"
    exit 1
}

trap cleanup INT TERM

# Execute main function
main "$@"