#!/bin/bash
# HA ML Predictor Environment Provisioning Script  
# Sprint 7 Task 3: CI/CD Pipeline Enhancement & Deployment Automation
#
# This script handles automated environment provisioning for different
# deployment targets with infrastructure as code principles

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
CONFIG_DIR="$PROJECT_ROOT/config"
TERRAFORM_DIR="$PROJECT_ROOT/infrastructure"

# Default values
ENVIRONMENT="staging"
CLOUD_PROVIDER="local"
DRY_RUN=false
VERBOSE=false
DESTROY=false
AUTO_APPROVE=false

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
HA ML Predictor Environment Provisioning Script

Usage: $0 [OPTIONS]

Options:
    -e, --environment ENV       Target environment (staging, production, development)
    -p, --provider PROVIDER     Cloud provider (local, aws, gcp, azure)
    -d, --dry-run              Plan only without applying changes
    --verbose                  Enable verbose output
    --destroy                  Destroy existing infrastructure
    --auto-approve            Skip interactive confirmations
    -h, --help                Show this help message

Supported Environments:
    development               Local development with minimal resources
    staging                   Staging environment with production-like setup
    production               Full production environment with HA

Supported Providers:
    local                     Local Docker/Docker Compose setup
    aws                       AWS ECS/RDS deployment
    gcp                       Google Cloud Run/CloudSQL deployment
    azure                     Azure Container Instances/Database

Examples:
    $0 --environment staging --provider local
    $0 -e production -p aws --verbose
    $0 --destroy --environment staging

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
            -p|--provider)
                CLOUD_PROVIDER="$2"
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
            --destroy)
                DESTROY=true
                shift
                ;;
            --auto-approve)
                AUTO_APPROVE=true
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
        development|staging|production)
            log_info "Target environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Validate cloud provider
validate_provider() {
    case $CLOUD_PROVIDER in
        local|aws|gcp|azure)
            log_info "Cloud provider: $CLOUD_PROVIDER"
            ;;
        *)
            log_error "Invalid provider: $CLOUD_PROVIDER"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites for $CLOUD_PROVIDER deployment..."
    
    case $CLOUD_PROVIDER in
        local)
            check_local_prerequisites
            ;;
        aws)
            check_aws_prerequisites
            ;;
        gcp)
            check_gcp_prerequisites
            ;;
        azure)
            check_azure_prerequisites
            ;;
    esac
}

# Check local prerequisites
check_local_prerequisites() {
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
    
    # Check available resources
    local available_memory
    available_memory=$(free -m | awk 'NR==2{printf "%.0f", $2}')
    
    local required_memory
    case $ENVIRONMENT in
        development) required_memory=2048 ;;
        staging) required_memory=4096 ;;
        production) required_memory=8192 ;;
    esac
    
    if [[ $available_memory -lt $required_memory ]]; then
        log_warning "Available memory (${available_memory}MB) is less than recommended (${required_memory}MB)"
    fi
    
    log_success "Local prerequisites check passed"
}

# Check AWS prerequisites
check_aws_prerequisites() {
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI is required but not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured or invalid"
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        log_error "Terraform is required for AWS deployment"
        exit 1
    fi
    
    log_success "AWS prerequisites check passed"
}

# Check GCP prerequisites  
check_gcp_prerequisites() {
    # Check gcloud CLI
    if ! command -v gcloud &> /dev/null; then
        log_error "Google Cloud CLI is required but not installed"
        exit 1
    fi
    
    # Check authentication
    if ! gcloud auth application-default print-access-token &> /dev/null; then
        log_error "GCP authentication not configured"
        exit 1
    fi
    
    log_success "GCP prerequisites check passed"
}

# Check Azure prerequisites
check_azure_prerequisites() {
    # Check Azure CLI
    if ! command -v az &> /dev/null; then
        log_error "Azure CLI is required but not installed"
        exit 1
    fi
    
    # Check authentication
    if ! az account show &> /dev/null; then
        log_error "Azure authentication not configured"
        exit 1
    fi
    
    log_success "Azure prerequisites check passed"
}

# Generate environment configuration
generate_environment_config() {
    log_info "Generating environment configuration..."
    
    local env_config_dir="$CONFIG_DIR/environments/$ENVIRONMENT"
    mkdir -p "$env_config_dir"
    
    # Base configuration template
    cat > "$env_config_dir/config.yaml" << EOF
# Environment configuration for $ENVIRONMENT
environment: $ENVIRONMENT
cloud_provider: $CLOUD_PROVIDER

# Application configuration
application:
  name: ha-ml-predictor
  version: latest
  replicas: $(get_replica_count)
  
# Resource configuration
resources:
  cpu_limit: $(get_cpu_limit)
  memory_limit: $(get_memory_limit)
  cpu_request: $(get_cpu_request)
  memory_request: $(get_memory_request)
  
# Database configuration
database:
  type: postgresql
  version: "15"
  storage_size: $(get_db_storage_size)
  backup_retention: $(get_backup_retention)
  
# Cache configuration
cache:
  type: redis
  version: "7"
  memory_size: $(get_cache_memory_size)
  
# Monitoring configuration
monitoring:
  enabled: $(get_monitoring_enabled)
  prometheus_retention: $(get_prometheus_retention)
  grafana_enabled: $(get_grafana_enabled)
  
# Security configuration
security:
  network_policies: $(get_network_policies_enabled)
  pod_security_standards: $(get_pod_security_standard)
  secrets_encryption: $(get_secrets_encryption)
  
# Backup configuration
backup:
  enabled: $(get_backup_enabled)
  schedule: "$(get_backup_schedule)"
  retention_days: $(get_backup_retention_days)
EOF
    
    log_success "Environment configuration generated: $env_config_dir/config.yaml"
}

# Helper functions for environment-specific values
get_replica_count() {
    case $ENVIRONMENT in
        development) echo "1" ;;
        staging) echo "2" ;;
        production) echo "5" ;;
    esac
}

get_cpu_limit() {
    case $ENVIRONMENT in
        development) echo "500m" ;;
        staging) echo "1000m" ;;
        production) echo "2000m" ;;
    esac
}

get_memory_limit() {
    case $ENVIRONMENT in
        development) echo "1Gi" ;;
        staging) echo "2Gi" ;;
        production) echo "4Gi" ;;
    esac
}

get_cpu_request() {
    case $ENVIRONMENT in
        development) echo "200m" ;;
        staging) echo "500m" ;;
        production) echo "1000m" ;;
    esac
}

get_memory_request() {
    case $ENVIRONMENT in
        development) echo "512Mi" ;;
        staging) echo "1Gi" ;;
        production) echo "2Gi" ;;
    esac
}

get_db_storage_size() {
    case $ENVIRONMENT in
        development) echo "10Gi" ;;
        staging) echo "50Gi" ;;
        production) echo "200Gi" ;;
    esac
}

get_cache_memory_size() {
    case $ENVIRONMENT in
        development) echo "256Mi" ;;
        staging) echo "512Mi" ;;
        production) echo "1Gi" ;;
    esac
}

get_monitoring_enabled() {
    case $ENVIRONMENT in
        development) echo "false" ;;
        staging) echo "true" ;;
        production) echo "true" ;;
    esac
}

get_prometheus_retention() {
    case $ENVIRONMENT in
        development) echo "7d" ;;
        staging) echo "15d" ;;
        production) echo "30d" ;;
    esac
}

get_grafana_enabled() {
    case $ENVIRONMENT in
        development) echo "false" ;;
        staging) echo "true" ;;
        production) echo "true" ;;
    esac
}

get_network_policies_enabled() {
    case $ENVIRONMENT in
        development) echo "false" ;;
        staging) echo "true" ;;
        production) echo "true" ;;
    esac
}

get_pod_security_standard() {
    case $ENVIRONMENT in
        development) echo "baseline" ;;
        staging) echo "restricted" ;;
        production) echo "restricted" ;;
    esac
}

get_secrets_encryption() {
    case $ENVIRONMENT in
        development) echo "false" ;;
        staging) echo "true" ;;
        production) echo "true" ;;
    esac
}

get_backup_enabled() {
    case $ENVIRONMENT in
        development) echo "false" ;;
        staging) echo "true" ;;
        production) echo "true" ;;
    esac
}

get_backup_schedule() {
    case $ENVIRONMENT in
        development) echo "0 6 * * 0" ;;  # Weekly
        staging) echo "0 2 * * *" ;;      # Daily
        production) echo "0 1 * * *" ;;   # Daily
    esac
}

get_backup_retention() {
    case $ENVIRONMENT in
        development) echo "7" ;;
        staging) echo "15" ;;
        production) echo "30" ;;
    esac
}

get_backup_retention_days() {
    case $ENVIRONMENT in
        development) echo "7" ;;
        staging) echo "30" ;;
        production) echo "90" ;;
    esac
}

# Provision local environment
provision_local() {
    log_info "Provisioning local $ENVIRONMENT environment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would provision local environment"
        return 0
    fi
    
    cd "$PROJECT_ROOT/docker"
    
    # Create environment-specific compose override
    local compose_override="docker-compose.$ENVIRONMENT.yml"
    
    cat > "$compose_override" << EOF
# Environment-specific overrides for $ENVIRONMENT
version: '3.8'

services:
  ha-ml-predictor:
    deploy:
      replicas: $(get_replica_count)
      resources:
        limits:
          cpus: '$(echo $(get_cpu_limit) | sed 's/m$//' | awk '{print $1/1000}')'
          memory: $(get_memory_limit)
        reservations:
          cpus: '$(echo $(get_cpu_request) | sed 's/m$//' | awk '{print $1/1000}')'
          memory: $(get_memory_request)
    environment:
      - ENVIRONMENT=$ENVIRONMENT
      - LOG_LEVEL=$(case $ENVIRONMENT in development) echo "DEBUG" ;; *) echo "INFO" ;; esac)
      
  postgres:
    volumes:
      - postgres_data_$ENVIRONMENT:/var/lib/postgresql/data
    environment:
      - POSTGRES_DB=ha_ml_predictor_$ENVIRONMENT
      
  redis:
    command: redis-server --maxmemory $(get_cache_memory_size | sed 's/i$/b/')
    
volumes:
  postgres_data_$ENVIRONMENT:
    external: false
EOF
    
    # Start services
    log_info "Starting services with environment-specific configuration..."
    docker-compose -f docker-compose.yml -f "$compose_override" up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Run health checks
    if health_check_services; then
        log_success "Local $ENVIRONMENT environment provisioned successfully"
    else
        log_error "Local $ENVIRONMENT environment provisioning failed"
        return 1
    fi
}

# Provision AWS environment
provision_aws() {
    log_info "Provisioning AWS $ENVIRONMENT environment..."
    
    if [[ "$DRY_RUN" == true ]]; then
        log_info "[DRY RUN] Would provision AWS environment using Terraform"
        return 0
    fi
    
    # Create Terraform configuration directory
    local tf_dir="$TERRAFORM_DIR/aws/$ENVIRONMENT"
    mkdir -p "$tf_dir"
    
    # Generate Terraform configuration
    generate_aws_terraform_config "$tf_dir"
    
    cd "$tf_dir"
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan -out=tfplan
    
    # Apply if not dry run
    if [[ "$AUTO_APPROVE" == true ]]; then
        terraform apply -auto-approve tfplan
    else
        terraform apply tfplan
    fi
    
    log_success "AWS $ENVIRONMENT environment provisioned successfully"
}

# Generate AWS Terraform configuration
generate_aws_terraform_config() {
    local tf_dir="$1"
    
    cat > "$tf_dir/main.tf" << EOF
# AWS Infrastructure for HA ML Predictor - $ENVIRONMENT
terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
  }
  
  backend "s3" {
    bucket = "ha-ml-predictor-terraform-state"
    key    = "$ENVIRONMENT/terraform.tfstate"
    region = var.aws_region
  }
}

provider "aws" {
  region = var.aws_region
  
  default_tags {
    tags = {
      Environment = "$ENVIRONMENT"
      Project     = "ha-ml-predictor"
      ManagedBy   = "terraform"
    }
  }
}

# Variables
variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "app_name" {
  description = "Application name"
  type        = string
  default     = "ha-ml-predictor"
}

# VPC and Networking
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "\${var.app_name}-\${var.environment}"
  cidr = "10.0.0.0/16"
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = ["10.0.1.0/24", "10.0.2.0/24"]
  public_subnets  = ["10.0.101.0/24", "10.0.102.0/24"]
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  
  tags = {
    Environment = "$ENVIRONMENT"
  }
}

# ECS Cluster
resource "aws_ecs_cluster" "main" {
  name = "\${var.app_name}-\${var.environment}"
  
  capacity_providers = ["FARGATE", "FARGATE_SPOT"]
  
  default_capacity_provider_strategy {
    capacity_provider = "FARGATE"
    weight           = 1
  }
}

# RDS Instance
resource "aws_db_instance" "postgres" {
  identifier = "\${var.app_name}-\${var.environment}-db"
  
  engine         = "postgres"
  engine_version = "15"
  instance_class = "$(case $ENVIRONMENT in development) echo "db.t3.micro" ;; staging) echo "db.t3.small" ;; production) echo "db.t3.medium" ;; esac)"
  
  allocated_storage     = $(echo $(get_db_storage_size) | sed 's/Gi$//')
  max_allocated_storage = $(($(echo $(get_db_storage_size) | sed 's/Gi$//') * 2))
  
  db_name  = "ha_ml_predictor"
  username = "ha_ml_predictor"
  password = random_password.db_password.result
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = $(get_backup_retention)
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  skip_final_snapshot = $(case $ENVIRONMENT in production) echo "false" ;; *) echo "true" ;; esac)
  deletion_protection = $(case $ENVIRONMENT in production) echo "true" ;; *) echo "false" ;; esac)
}

# ElastiCache Redis
resource "aws_elasticache_subnet_group" "redis" {
  name       = "\${var.app_name}-\${var.environment}-redis"
  subnet_ids = module.vpc.private_subnets
}

resource "aws_elasticache_cluster" "redis" {
  cluster_id           = "\${var.app_name}-\${var.environment}-redis"
  engine               = "redis"
  node_type           = "$(case $ENVIRONMENT in development) echo "cache.t3.micro" ;; staging) echo "cache.t3.small" ;; production) echo "cache.t3.medium" ;; esac)"
  num_cache_nodes     = 1
  parameter_group_name = "default.redis7"
  port                = 6379
  subnet_group_name   = aws_elasticache_subnet_group.redis.name
  security_group_ids  = [aws_security_group.redis.id]
}

# Security Groups
resource "aws_security_group" "rds" {
  name_prefix = "\${var.app_name}-\${var.environment}-rds"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 5432
    to_port     = 5432
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_security_group" "redis" {
  name_prefix = "\${var.app_name}-\${var.environment}-redis"
  vpc_id      = module.vpc.vpc_id
  
  ingress {
    from_port   = 6379
    to_port     = 6379
    protocol    = "tcp"
    cidr_blocks = [module.vpc.vpc_cidr_block]
  }
}

# Random password for database
resource "random_password" "db_password" {
  length  = 16
  special = true
}

# Outputs
output "rds_endpoint" {
  value = aws_db_instance.postgres.endpoint
}

output "redis_endpoint" {
  value = aws_elasticache_cluster.redis.cache_nodes[0].address
}
EOF
    
    log_info "AWS Terraform configuration generated in $tf_dir"
}

# Health check services
health_check_services() {
    log_info "Running health checks..."
    
    # Check if containers are running
    if ! docker ps | grep -q "ha-ml-predictor"; then
        log_error "HA ML Predictor container not running"
        return 1
    fi
    
    if ! docker ps | grep -q "postgres"; then
        log_error "PostgreSQL container not running"
        return 1
    fi
    
    if ! docker ps | grep -q "redis"; then
        log_error "Redis container not running"
        return 1
    fi
    
    # Check API endpoints
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f "http://localhost:8000/api/health" &>/dev/null; then
            log_success "API health check passed"
            break
        fi
        
        log_info "Health check attempt $attempt/$max_attempts..."
        sleep 5
        ((attempt++))
        
        if [[ $attempt -gt $max_attempts ]]; then
            log_error "API health check failed after $((max_attempts * 5)) seconds"
            return 1
        fi
    done
    
    return 0
}

# Destroy environment
destroy_environment() {
    log_warning "Destroying $ENVIRONMENT environment..."
    
    if [[ "$AUTO_APPROVE" != true ]]; then
        echo
        read -p "Are you sure you want to destroy the $ENVIRONMENT environment? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Destroy cancelled by user"
            exit 0
        fi
    fi
    
    case $CLOUD_PROVIDER in
        local)
            destroy_local
            ;;
        aws)
            destroy_aws
            ;;
        *)
            log_error "Destroy not implemented for $CLOUD_PROVIDER"
            exit 1
            ;;
    esac
}

# Destroy local environment
destroy_local() {
    log_info "Destroying local $ENVIRONMENT environment..."
    
    cd "$PROJECT_ROOT/docker"
    
    # Stop and remove containers
    docker-compose -f docker-compose.yml -f "docker-compose.$ENVIRONMENT.yml" down -v
    
    # Remove environment-specific volumes
    docker volume rm "docker_postgres_data_$ENVIRONMENT" 2>/dev/null || true
    
    # Remove environment-specific compose file
    rm -f "docker-compose.$ENVIRONMENT.yml"
    
    log_success "Local $ENVIRONMENT environment destroyed"
}

# Destroy AWS environment
destroy_aws() {
    log_info "Destroying AWS $ENVIRONMENT environment..."
    
    local tf_dir="$TERRAFORM_DIR/aws/$ENVIRONMENT"
    
    if [[ -d "$tf_dir" ]]; then
        cd "$tf_dir"
        
        if [[ "$AUTO_APPROVE" == true ]]; then
            terraform destroy -auto-approve
        else
            terraform destroy
        fi
        
        log_success "AWS $ENVIRONMENT environment destroyed"
    else
        log_warning "No Terraform configuration found for AWS $ENVIRONMENT"
    fi
}

# Main provisioning function
main_provision() {
    log_info "=== HA ML Predictor Environment Provisioning ==="
    
    generate_environment_config
    
    if [[ "$DESTROY" == true ]]; then
        destroy_environment
        return 0
    fi
    
    case $CLOUD_PROVIDER in
        local)
            provision_local
            ;;
        aws)
            provision_aws
            ;;
        gcp)
            log_error "GCP provisioning not yet implemented"
            exit 1
            ;;
        azure)
            log_error "Azure provisioning not yet implemented"
            exit 1
            ;;
    esac
    
    log_success "Environment provisioning completed successfully!"
    
    # Display connection information
    display_connection_info
}

# Display connection information
display_connection_info() {
    echo
    log_success "=== Environment Connection Information ==="
    log_success "Environment: $ENVIRONMENT"
    log_success "Provider: $CLOUD_PROVIDER"
    
    case $CLOUD_PROVIDER in
        local)
            log_success "API Endpoint: http://localhost:8000"
            log_success "Health Check: http://localhost:8000/api/health"
            log_success "System Status: http://localhost:8000/api/system/status"
            log_success "Metrics: http://localhost:8000/metrics"
            
            if [[ "$(get_monitoring_enabled)" == "true" ]]; then
                log_success "Grafana: http://localhost:3000 (admin/admin)"
                log_success "Prometheus: http://localhost:9090"
            fi
            ;;
        aws)
            log_success "Check Terraform outputs for connection details"
            ;;
    esac
    
    echo
}

# Main execution
main() {
    parse_args "$@"
    validate_environment
    validate_provider
    check_prerequisites
    main_provision
}

# Execute main function if script is run directly
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi