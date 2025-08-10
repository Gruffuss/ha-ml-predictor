#!/bin/bash
# HA ML Predictor Rollback Script
# Sprint 7 Task 3: CI/CD Pipeline Enhancement & Deployment Automation
#
# This script provides comprehensive rollback capabilities with safety checks
# and automated recovery procedures for failed deployments

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DOCKER_DIR="$PROJECT_ROOT/docker"
BACKUP_DIR="$PROJECT_ROOT/backups"

# Default values
ENVIRONMENT="production"
ROLLBACK_VERSION=""
DRY_RUN=false
VERBOSE=false
FORCE_ROLLBACK=false
AUTO_APPROVE=false
BACKUP_CURRENT=true

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
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

log_debug() {
    if [[ "$VERBOSE" == true ]]; then
        echo -e "${PURPLE}[DEBUG]${NC} $1"
    fi
}

# Help function
show_help() {
    cat << EOF
HA ML Predictor Rollback Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Target environment (staging, production)
    -v, --version VERSION       Specific version to rollback to
    -d, --dry-run              Perform dry run without actual rollback
    --verbose                  Enable verbose output
    --force                    Force rollback without safety checks
    --auto-approve             Skip interactive confirmations
    --no-backup                Skip backing up current state
    -h, --help                 Show this help message

Rollback Strategies:
    automatic                  Rollback to last known good version
    specific                   Rollback to specific version (use -v)
    emergency                  Emergency rollback with minimal checks

Examples:
    $0 --environment production
    $0 -v v1.2.3 --environment staging
    $0 --force --auto-approve --environment production
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
            -v|--version)
                ROLLBACK_VERSION="$2"
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
            --force)
                FORCE_ROLLBACK=true
                shift
                ;;
            --auto-approve)
                AUTO_APPROVE=true
                shift
                ;;
            --no-backup)
                BACKUP_CURRENT=false
                shift
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

# Get current deployment info
get_current_deployment_info() {
    log_info "Gathering current deployment information..."
    
    cd "$DOCKER_DIR"
    
    # Get current container info
    CURRENT_CONTAINER_ID=$(docker ps -f "name=ha-ml-predictor" --format "{{.ID}}" | head -1)
    if [[ -n "$CURRENT_CONTAINER_ID" ]]; then
        CURRENT_IMAGE=$(docker inspect "$CURRENT_CONTAINER_ID" --format "{{.Config.Image}}")
        CURRENT_VERSION=$(echo "$CURRENT_IMAGE" | cut -d: -f2)
        CURRENT_CREATED=$(docker inspect "$CURRENT_CONTAINER_ID" --format "{{.Created}}")
        
        log_info "Current deployment:"
        log_info "  Container ID: $CURRENT_CONTAINER_ID"
        log_info "  Image: $CURRENT_IMAGE"
        log_info "  Version: $CURRENT_VERSION"
        log_info "  Created: $CURRENT_CREATED"
    else
        log_warning "No running ha-ml-predictor container found"
        CURRENT_CONTAINER_ID=""
        CURRENT_IMAGE=""
        CURRENT_VERSION=""
    fi
}

# Determine rollback target
determine_rollback_target() {
    log_info "Determining rollback target..."
    
    if [[ -n "$ROLLBACK_VERSION" ]]; then
        log_info "Using specified rollback version: $ROLLBACK_VERSION"
        TARGET_VERSION="$ROLLBACK_VERSION"
    else
        # Find last known good version
        TARGET_VERSION=$(get_last_known_good_version)
        if [[ -z "$TARGET_VERSION" ]]; then
            log_error "Unable to determine rollback target version"
            exit 1
        fi
        log_info "Using last known good version: $TARGET_VERSION"
    fi
    
    TARGET_IMAGE="ha-ml-predictor:$TARGET_VERSION"
}

# Get last known good version
get_last_known_good_version() {
    # Check deployment history file
    local history_file="$BACKUP_DIR/deployment-history.log"
    if [[ -f "$history_file" ]]; then
        # Get the second-to-last successful deployment
        local last_good=$(grep "SUCCESS" "$history_file" | tail -2 | head -1 | awk '{print $3}')
        if [[ -n "$last_good" ]]; then
            echo "$last_good"
            return 0
        fi
    fi
    
    # Fallback: look for available Docker images
    local available_versions
    available_versions=$(docker images "ha-ml-predictor" --format "{{.Tag}}" | grep -v latest | head -5)
    
    if [[ -n "$available_versions" ]]; then
        # Return the first non-current version
        for version in $available_versions; do
            if [[ "$version" != "$CURRENT_VERSION" ]]; then
                echo "$version"
                return 0
            fi
        done
    fi
    
    # Ultimate fallback
    echo "latest"
}

# Pre-rollback safety checks
pre_rollback_safety_checks() {
    log_info "Running pre-rollback safety checks..."
    
    if [[ "$FORCE_ROLLBACK" == true ]]; then
        log_warning "Force rollback enabled - skipping safety checks"
        return 0
    fi
    
    # Check 1: Verify target image exists
    log_debug "Checking if target image exists..."
    if ! docker images "$TARGET_IMAGE" | grep -q "$TARGET_VERSION"; then
        log_error "Target image not found: $TARGET_IMAGE"
        log_info "Available images:"
        docker images "ha-ml-predictor" --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}"
        exit 1
    fi
    
    # Check 2: Database connectivity
    log_debug "Checking database connectivity..."
    if [[ -n "$CURRENT_CONTAINER_ID" ]]; then
        if ! docker exec "$CURRENT_CONTAINER_ID" python -c "
import asyncio
from src.data.storage.database import get_database_manager
async def check():
    db = await get_database_manager()
    health = await db.health_check()
    print('Database health:', health)
asyncio.run(check())
" 2>/dev/null; then
            log_warning "Database connectivity check failed"
            if [[ "$AUTO_APPROVE" != true ]]; then
                read -p "Continue with rollback? (y/N): " -n 1 -r
                echo
                if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                    exit 1
                fi
            fi
        else
            log_success "Database connectivity verified"
        fi
    fi
    
    # Check 3: System resource availability
    log_debug "Checking system resources..."
    local available_memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $7}')
    if [[ $available_memory -lt 512 ]]; then
        log_warning "Low available memory: ${available_memory}MB"
    fi
    
    # Check 4: Disk space
    local available_disk
    available_disk=$(df "$PROJECT_ROOT" | awk 'NR==2{print $4}')
    if [[ $available_disk -lt 1048576 ]]; then  # 1GB in KB
        log_warning "Low disk space: $(($available_disk / 1024))MB"
    fi
    
    log_success "Pre-rollback safety checks completed"
}

# Backup current state
backup_current_state() {
    if [[ "$BACKUP_CURRENT" != true ]]; then
        log_info "Skipping current state backup"
        return 0
    fi
    
    log_info "Backing up current state..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would backup current state"
        return 0
    fi
    
    # Create backup directory
    local backup_timestamp
    backup_timestamp=$(date +%Y%m%d_%H%M%S)
    local backup_path="$BACKUP_DIR/rollback_backup_$backup_timestamp"
    mkdir -p "$backup_path"
    
    # Backup container configuration
    if [[ -n "$CURRENT_CONTAINER_ID" ]]; then
        log_debug "Backing up container configuration..."
        docker inspect "$CURRENT_CONTAINER_ID" > "$backup_path/container_config.json"
    fi
    
    # Backup Docker Compose configuration
    log_debug "Backing up Docker Compose configuration..."
    cp "$DOCKER_DIR/docker-compose.yml" "$backup_path/" || true
    cp "$DOCKER_DIR/docker-compose.prod.yml" "$backup_path/" || true
    
    # Backup environment variables
    log_debug "Backing up environment configuration..."
    docker exec "$CURRENT_CONTAINER_ID" env > "$backup_path/environment.txt" 2>/dev/null || true
    
    # Backup application logs (last 1000 lines)
    log_debug "Backing up recent application logs..."
    docker logs --tail=1000 "$CURRENT_CONTAINER_ID" > "$backup_path/application.log" 2>/dev/null || true
    
    # Create rollback metadata
    cat > "$backup_path/rollback_metadata.json" << EOF
{
    "backup_timestamp": "$backup_timestamp",
    "current_version": "$CURRENT_VERSION",
    "current_image": "$CURRENT_IMAGE",
    "current_container_id": "$CURRENT_CONTAINER_ID",
    "target_version": "$TARGET_VERSION",
    "environment": "$ENVIRONMENT",
    "rollback_reason": "manual_rollback"
}
EOF
    
    log_success "Current state backed up to: $backup_path"
    echo "$backup_path" > "$BACKUP_DIR/latest_rollback_backup.txt"
}

# Execute rollback
execute_rollback() {
    log_info "Executing rollback to version $TARGET_VERSION..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would execute rollback procedure"
        log_info "[DRY RUN] Target image: $TARGET_IMAGE"
        return 0
    fi
    
    cd "$DOCKER_DIR"
    
    # Step 1: Prepare rollback environment
    log_info "Step 1: Preparing rollback environment..."
    
    # Set target version in environment
    export IMAGE_TAG="$TARGET_VERSION"
    
    # Step 2: Stop current services gracefully
    log_info "Step 2: Stopping current services..."
    docker-compose down --timeout 30 || true
    
    # Step 3: Deploy rollback version
    log_info "Step 3: Deploying rollback version..."
    
    # Modify compose file to use target version
    local compose_file="docker-compose.yml"
    if [[ "$ENVIRONMENT" == "production" ]]; then
        compose_file="docker-compose.prod.yml"
    fi
    
    # Update image tag in compose file temporarily
    sed -i.rollback_backup "s/ha-ml-predictor:latest/ha-ml-predictor:$TARGET_VERSION/g" "$compose_file"
    
    # Start services with rollback version
    docker-compose -f "$compose_file" up -d
    
    # Restore original compose file
    mv "$compose_file.rollback_backup" "$compose_file"
    
    log_success "Rollback deployment initiated"
}

# Post-rollback validation
post_rollback_validation() {
    log_info "Running post-rollback validation..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would run post-rollback validation"
        return 0
    fi
    
    # Wait for container to start
    log_info "Waiting for rolled-back container to start..."
    sleep 10
    
    # Get new container ID
    local rollback_container_id
    rollback_container_id=$(docker ps -f "name=ha-ml-predictor" --format "{{.ID}}" | head -1)
    
    if [[ -z "$rollback_container_id" ]]; then
        log_error "Rollback container not found - rollback may have failed"
        return 1
    fi
    
    # Wait for health check
    log_info "Waiting for health checks..."
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if docker ps --filter "id=$rollback_container_id" --filter "health=healthy" | grep -q "$rollback_container_id"; then
            log_success "Rollback container is healthy"
            break
        fi
        
        log_debug "Health check attempt $attempt/$max_attempts..."
        sleep 5
        ((attempt++))
        
        if [[ $attempt -gt $max_attempts ]]; then
            log_error "Rollback container failed health check"
            return 1
        fi
    done
    
    # API endpoint tests
    log_info "Testing API endpoints..."
    
    # Health endpoint
    if curl -f "http://localhost:8000/api/health" &>/dev/null; then
        log_success "Health endpoint responding"
    else
        log_error "Health endpoint not responding"
        return 1
    fi
    
    # System status endpoint
    if curl -f "http://localhost:8000/api/system/status" &>/dev/null; then
        log_success "System status endpoint responding"
    else
        log_warning "System status endpoint not responding"
    fi
    
    # Database connectivity test
    log_info "Testing database connectivity..."
    if docker exec "$rollback_container_id" python -c "
import asyncio
from src.data.storage.database import get_database_manager
async def check():
    try:
        db = await get_database_manager()
        health = await db.health_check()
        print('✅ Database connectivity verified')
        return True
    except Exception as e:
        print(f'❌ Database connectivity failed: {e}')
        return False
asyncio.run(check())
" 2>/dev/null; then
        log_success "Database connectivity verified"
    else
        log_error "Database connectivity test failed"
        return 1
    fi
    
    log_success "Post-rollback validation completed successfully"
    return 0
}

# Update deployment history
update_deployment_history() {
    local status="$1"
    local history_file="$BACKUP_DIR/deployment-history.log"
    
    mkdir -p "$BACKUP_DIR"
    
    local timestamp
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    
    echo "$timestamp ROLLBACK $TARGET_VERSION $status $ENVIRONMENT" >> "$history_file"
    
    # Keep only last 100 entries
    tail -100 "$history_file" > "$history_file.tmp" && mv "$history_file.tmp" "$history_file"
}

# Rollback confirmation
confirm_rollback() {
    if [[ "$AUTO_APPROVE" == true ]]; then
        return 0
    fi
    
    echo
    log_warning "ROLLBACK CONFIRMATION REQUIRED"
    echo
    echo "Current deployment:"
    echo "  Version: $CURRENT_VERSION"
    echo "  Image: $CURRENT_IMAGE"
    echo
    echo "Rollback target:"
    echo "  Version: $TARGET_VERSION"
    echo "  Image: $TARGET_IMAGE"
    echo "  Environment: $ENVIRONMENT"
    echo
    echo "This operation will:"
    echo "  1. Stop the current deployment"
    echo "  2. Deploy the rollback version"
    echo "  3. Run validation tests"
    echo
    
    if [[ "$BACKUP_CURRENT" == true ]]; then
        echo "Current state will be backed up before rollback."
    else
        echo "⚠️  Current state will NOT be backed up."
    fi
    
    echo
    read -p "Are you sure you want to proceed with the rollback? (y/N): " -n 1 -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        log_info "Rollback cancelled by user"
        exit 0
    fi
    
    log_info "Rollback confirmed by user"
}

# Main rollback function
main_rollback() {
    log_info "=== HA ML Predictor Rollback Procedure ==="
    log_info "Environment: $ENVIRONMENT"
    
    if [[ "$DRY_RUN" == true ]]; then
        log_warning "DRY RUN MODE - No actual changes will be made"
    fi
    
    # Gather information
    get_current_deployment_info
    determine_rollback_target
    
    # Safety checks
    pre_rollback_safety_checks
    
    # Confirmation
    confirm_rollback
    
    # Backup current state
    backup_current_state
    
    # Execute rollback
    execute_rollback
    
    # Validation
    if post_rollback_validation; then
        log_success "🎉 Rollback completed successfully!"
        update_deployment_history "SUCCESS"
        
        echo
        log_success "Rollback Summary:"
        log_success "  From: $CURRENT_VERSION → To: $TARGET_VERSION"
        log_success "  Environment: $ENVIRONMENT"
        log_success "  Timestamp: $(date)"
        
        if [[ "$BACKUP_CURRENT" == true ]]; then
            local backup_location
            backup_location=$(cat "$BACKUP_DIR/latest_rollback_backup.txt" 2>/dev/null || echo "unknown")
            log_success "  Backup location: $backup_location"
        fi
        
    else
        log_error "❌ Rollback validation failed!"
        update_deployment_history "FAILED"
        
        log_error "Rollback completed but validation failed."
        log_error "Manual intervention may be required."
        
        # Offer emergency recovery
        if [[ "$AUTO_APPROVE" != true ]]; then
            echo
            read -p "Would you like to attempt emergency recovery? (y/N): " -n 1 -r
            echo
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                emergency_recovery
            fi
        fi
        
        exit 1
    fi
}

# Emergency recovery procedure
emergency_recovery() {
    log_warning "Initiating emergency recovery procedure..."
    
    # Stop all containers
    log_info "Stopping all containers..."
    docker stop $(docker ps -q) || true
    
    # Start with minimal configuration
    log_info "Starting with minimal configuration..."
    cd "$DOCKER_DIR"
    
    # Start only essential services
    docker-compose up -d postgres redis
    sleep 10
    
    # Start application with latest stable image
    docker run -d --name ha-ml-predictor-emergency \
        --network ha-ml-predictor_default \
        ha-ml-predictor:latest
    
    log_info "Emergency recovery initiated - manual verification required"
}

# Main execution
main() {
    parse_args "$@"
    validate_environment
    main_rollback
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi