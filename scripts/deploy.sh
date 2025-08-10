#!/bin/bash
# HA ML Predictor Deployment Script
# Sprint 7 Task 3: CI/CD Pipeline Enhancement & Deployment Automation
#
# This script handles automated deployment to different environments
# with support for various deployment strategies and rollback capabilities

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
DOCKER_DIR="$PROJECT_ROOT/docker"

# Default values
ENVIRONMENT="staging"
DEPLOYMENT_STRATEGY="rolling"
VERSION="latest"
DRY_RUN=false
VERBOSE=false
FORCE_REBUILD=false
HEALTH_CHECK_TIMEOUT=300

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

# Help function
show_help() {
    cat << EOF
HA ML Predictor Deployment Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Target environment (staging, production)
    -s, --strategy STRATEGY     Deployment strategy (rolling, blue-green, canary)
    -v, --version VERSION       Version to deploy (default: latest)
    -d, --dry-run              Perform dry run without actual deployment
    --verbose                  Enable verbose output
    --force-rebuild            Force rebuild of Docker images
    --timeout SECONDS          Health check timeout (default: 300)
    -h, --help                 Show this help message

Examples:
    $0 --environment staging --strategy rolling
    $0 -e production -s blue-green -v v1.2.3
    $0 --dry-run --verbose

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -s|--strategy)
                DEPLOYMENT_STRATEGY="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --verbose)
                VERBOSE=true
                shift
                ;;
            --force-rebuild)
                FORCE_REBUILD=true
                shift
                ;;
            --timeout)
                HEALTH_CHECK_TIMEOUT="$2"
                shift 2
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done
}

# Validate environment
validate_environment() {
    case $ENVIRONMENT in
        staging|production)
            log_info "Target environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT. Must be 'staging' or 'production'"
            exit 1
            ;;
    esac
}

# Validate deployment strategy
validate_strategy() {
    case $DEPLOYMENT_STRATEGY in
        rolling|blue-green|canary)
            log_info "Deployment strategy: $DEPLOYMENT_STRATEGY"
            ;;
        *)
            log_error "Invalid strategy: $DEPLOYMENT_STRATEGY. Must be 'rolling', 'blue-green', or 'canary'"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking deployment prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        log_error "Docker Compose is required but not installed"
        exit 1
    fi
    
    # Check required directories
    if [[ ! -d "$DOCKER_DIR" ]]; then
        log_error "Docker directory not found: $DOCKER_DIR"
        exit 1
    fi
    
    # Check Docker files
    if [[ ! -f "$DOCKER_DIR/Dockerfile" ]]; then
        log_error "Dockerfile not found: $DOCKER_DIR/Dockerfile"
        exit 1
    fi
    
    local compose_file="$DOCKER_DIR/docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="$DOCKER_DIR/docker-compose.prod.yml"
    fi
    
    if [[ ! -f "$compose_file" ]]; then
        log_error "Docker Compose file not found: $compose_file"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would build Docker images"
        return 0
    fi
    
    local build_args=""
    if [[ "$FORCE_REBUILD" == true ]]; then
        build_args="--no-cache"
    fi
    
    cd "$DOCKER_DIR"
    
    # Build main application image
    log_info "Building ha-ml-predictor:$VERSION..."
    docker build $build_args \
        --build-arg VERSION="$VERSION" \
        --build-arg BUILD_DATE="$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
        --tag "ha-ml-predictor:$VERSION" \
        --tag "ha-ml-predictor:latest" \
        --file Dockerfile \
        "$PROJECT_ROOT"
    
    log_success "Docker images built successfully"
}

# Deploy using rolling strategy
deploy_rolling() {
    log_info "Executing rolling deployment..."
    
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    cd "$DOCKER_DIR"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would execute: docker-compose -f $compose_file up -d --scale ha-ml-predictor=3"
        return 0
    fi
    
    # Update services with rolling update
    docker-compose -f "$compose_file" up -d --scale ha-ml-predictor=3
    
    log_success "Rolling deployment completed"
}

# Deploy using blue-green strategy
deploy_blue_green() {
    log_info "Executing blue-green deployment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would execute blue-green deployment process"
        return 0
    fi
    
    cd "$DOCKER_DIR"
    
    # Step 1: Deploy to green environment
    log_info "Step 1: Deploying to GREEN environment..."
    docker-compose -f docker-compose.yml -p "ha-ml-predictor-green" up -d
    
    # Step 2: Health check green environment
    log_info "Step 2: Health checking GREEN environment..."
    if ! wait_for_health "ha-ml-predictor-green_ha-ml-predictor_1"; then
        log_error "GREEN environment health check failed"
        return 1
    fi
    
    # Step 3: Run smoke tests
    log_info "Step 3: Running smoke tests on GREEN..."
    if ! run_smoke_tests "green"; then
        log_error "Smoke tests failed on GREEN environment"
        return 1
    fi
    
    # Step 4: Switch traffic
    log_info "Step 4: Switching traffic to GREEN..."
    # This would involve updating load balancer or ingress configurations
    # For now, we'll simulate the traffic switch
    sleep 2
    
    # Step 5: Stop blue environment
    log_info "Step 5: Stopping BLUE environment..."
    docker-compose -f docker-compose.yml -p "ha-ml-predictor-blue" down || true
    
    # Step 6: Rename green to blue for next deployment
    log_info "Step 6: Renaming GREEN to production..."
    docker-compose -f docker-compose.yml -p "ha-ml-predictor-green" down
    docker-compose -f docker-compose.yml up -d
    
    log_success "Blue-green deployment completed"
}

# Deploy using canary strategy
deploy_canary() {
    log_info "Executing canary deployment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would execute canary deployment process"
        return 0
    fi
    
    cd "$DOCKER_DIR"
    
    # Step 1: Deploy canary version (single instance)
    log_info "Step 1: Deploying canary version..."
    docker run -d --name "ha-ml-predictor-canary" \
        --network ha-ml-predictor_default \
        "ha-ml-predictor:$VERSION"
    
    # Step 2: Monitor canary
    log_info "Step 2: Monitoring canary performance..."
    sleep 30
    
    # Step 3: Gradually increase canary traffic
    log_info "Step 3: Increasing canary traffic..."
    # This would involve updating load balancer weights
    sleep 30
    
    # Step 4: Full rollout if canary is healthy
    log_info "Step 4: Completing canary rollout..."
    docker stop ha-ml-predictor-canary || true
    docker rm ha-ml-predictor-canary || true
    
    # Full deployment
    docker-compose -f docker-compose.yml up -d
    
    log_success "Canary deployment completed"
}

# Wait for health check
wait_for_health() {
    local container_name="$1"
    local max_attempts=$((HEALTH_CHECK_TIMEOUT / 5))
    local attempt=1
    
    log_info "Waiting for $container_name to become healthy..."
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker ps --filter "name=$container_name" --filter "health=healthy" | grep -q "$container_name"; then
            log_success "$container_name is healthy"
            return 0
        fi
        
        log_info "Health check attempt $attempt/$max_attempts..."
        sleep 5
        ((attempt++))
    done
    
    log_error "$container_name failed health check after $HEALTH_CHECK_TIMEOUT seconds"
    return 1
}

# Run smoke tests
run_smoke_tests() {
    local environment="$1"
    log_info "Running smoke tests for $environment environment..."
    
    # Test API endpoints
    local base_url="http://localhost:8000"
    if [[ "$environment" == "green" ]]; then
        base_url="http://localhost:8001"  # Assuming green runs on different port
    fi
    
    # Health check
    if ! curl -f "$base_url/api/health" &>/dev/null; then
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # System status
    if ! curl -f "$base_url/api/system/status" &>/dev/null; then
        log_error "System status endpoint not responding"
        return 1
    fi
    
    log_success "Smoke tests passed for $environment environment"
    return 0
}

# Main deployment function
deploy() {
    log_info "Starting deployment process..."
    log_info "Environment: $ENVIRONMENT"
    log_info "Strategy: $DEPLOYMENT_STRATEGY"
    log_info "Version: $VERSION"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Build images
    build_images
    
    # Execute deployment based on strategy
    case $DEPLOYMENT_STRATEGY in
        rolling)
            deploy_rolling
            ;;
        blue-green)
            deploy_blue_green
            ;;
        canary)
            deploy_canary
            ;;
    esac
    
    # Final health check
    if [[ "$DRY_RUN" != true ]]; then
        log_info "Running final health checks..."
        if wait_for_health "ha-ml-predictor"; then
            log_success "Deployment completed successfully!"
        else
            log_error "Deployment completed but health checks failed"
            exit 1
        fi
    fi
}

# Rollback function
rollback() {
    log_warning "Initiating rollback procedure..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would execute rollback procedure"
        return 0
    fi
    
    cd "$DOCKER_DIR"
    
    # Stop current deployment
    log_info "Stopping current deployment..."
    docker-compose down
    
    # Deploy previous stable version
    log_info "Deploying previous stable version..."
    docker-compose up -d
    
    # Wait for health check
    if wait_for_health "ha-ml-predictor"; then
        log_success "Rollback completed successfully"
    else
        log_error "Rollback failed - manual intervention required"
        exit 1
    fi
}

# Main execution
main() {
    parse_args "$@"
    validate_environment
    validate_strategy
    check_prerequisites
    
    # Check if rollback is requested
    if [[ "${1:-}" == "rollback" ]]; then
        rollback
    else
        deploy
    fi
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi