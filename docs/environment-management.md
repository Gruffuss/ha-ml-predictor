# Environment Management Guide

This guide covers the comprehensive environment management system for the HA ML Predictor project, including configuration management, secrets handling, multi-environment deployment, and disaster recovery.

## Overview

The environment management system provides:
- **Environment-based configuration** with automatic detection
- **Secure secrets management** with encryption
- **Configuration validation** and testing framework
- **Multi-environment deployment** (development, testing, staging, production)
- **Backup and disaster recovery** procedures
- **Automated deployment** scripts

## Supported Environments

### Development
- **Purpose**: Local development and testing
- **Configuration**: `config/config.development.yaml`
- **Features**: Debug mode, verbose logging, no authentication, development tools
- **Services**: All services on different ports to avoid conflicts

### Testing
- **Purpose**: Automated testing and CI/CD
- **Configuration**: Uses base `config/config.yaml` with testing overrides
- **Features**: Minimal resources, fast startup, test-specific settings

### Staging
- **Purpose**: Pre-production testing and validation
- **Configuration**: `config/config.staging.yaml`
- **Features**: Production-like setup, monitoring enabled, secure but accessible

### Production
- **Purpose**: Live production deployment
- **Configuration**: `config/config.production.yaml`
- **Features**: Maximum security, monitoring, backups, optimized performance

## Environment Management Tools

### Environment Manager Script

The `scripts/environment-manager.py` script provides comprehensive environment management:

```bash
# List all available environments
python scripts/environment-manager.py list

# Set up a new environment (interactive)
python scripts/environment-manager.py setup production

# Validate environment configuration
python scripts/environment-manager.py validate staging --test-connections

# Manage secrets
python scripts/environment-manager.py secrets production list
python scripts/environment-manager.py secrets production set ha_token your_token_here

# Create backups
python scripts/environment-manager.py backup production --type database

# Deploy environment
python scripts/environment-manager.py deploy staging
```

### Deployment Script

The `scripts/deploy-environment.sh` script provides automated deployment:

```bash
# Deploy to production
./scripts/deploy-environment.sh production

# Validate configuration only
./scripts/deploy-environment.sh staging --validate-only

# Deploy with connection testing
./scripts/deploy-environment.sh development --test-connections

# Build images only
./scripts/deploy-environment.sh production --build-only

# Force deployment even if validation fails
./scripts/deploy-environment.sh staging --force
```

## Configuration Management

### Environment Detection

The system automatically detects the current environment using:
1. `ENVIRONMENT` or `ENV` environment variable (highest priority)
2. Environment-specific config files (e.g., `config.production.yaml`)
3. Docker environment detection
4. Defaults to development

### Configuration Files

Each environment can have its own configuration file:
- `config/config.yaml` - Base configuration
- `config/config.development.yaml` - Development overrides
- `config/config.staging.yaml` - Staging overrides
- `config/config.production.yaml` - Production overrides

### Configuration Validation

The validation framework checks:
- **Home Assistant connectivity** and API access
- **Database connectivity** and TimescaleDB extension
- **MQTT broker** connectivity and topic structure
- **Room and sensor configurations**
- **System requirements** (Python version, disk space, memory)
- **Environment-specific rules** (security, performance)

Example validation:
```python
from src.core.config_validator import ConfigurationValidator

validator = ConfigurationValidator()
result = validator.validate_config_files(
    environment="production",
    test_connections=True
)
print(result)
```

## Secrets Management

### Encryption

Secrets are encrypted using Fernet (AES 128) encryption:
- Master key stored in `secrets/master.key`
- Environment-specific secrets in `secrets/{environment}.json`
- Automatic encryption/decryption based on environment settings

### Required Secrets by Environment

#### Development
- `ha_token` (optional, default: "dev_token")
- `database_password` (optional, default: "dev_pass")

#### Testing
- `ha_token` (optional, default: "test_token")
- `database_password` (optional, default: "test_pass")

#### Staging
- `ha_token` (required)
- `database_password` (required)
- `redis_password` (optional)

#### Production
- `ha_token` (required)
- `database_password` (required)
- `redis_password` (required)
- `api_secret_key` (required)
- `grafana_password` (optional)

### Setting Secrets

Interactive setup:
```bash
python scripts/environment-manager.py secrets production setup
```

Direct secret setting:
```bash
python scripts/environment-manager.py secrets production set ha_token "your_ha_token_here"
```

Using environment variables (highest priority):
```bash
export HA_TOKEN="your_token"
export DATABASE_PASSWORD="secure_password"
```

### Key Rotation

Rotate encryption keys for all environments:
```python
from src.core.environment import get_environment_manager

env_manager = get_environment_manager()
env_manager.secrets_manager.rotate_encryption_key()
```

## Multi-Environment Deployment

### Docker Compose Configuration

Each environment uses a combination of Docker Compose files:

#### Development
```bash
docker-compose -f docker-compose.yml -f docker-compose.development.yml up -d
```
- Source code mounting for live development
- Development tools (PgAdmin, Redis Commander, MQTT Explorer)
- Different ports to avoid conflicts
- Relaxed security settings

#### Staging
```bash
docker-compose -f docker-compose.yml -f docker-compose.staging.yml up -d
```
- Production-like configuration
- Monitoring enabled (Prometheus, Grafana)
- Secure but accessible for testing
- Staging-specific volumes and networks

#### Production
```bash
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d
```
- Maximum security and performance
- Resource limits and restart policies
- Structured logging with rotation
- Production backup procedures

### Environment Variables

Each environment supports environment-specific variables:

```bash
# Development
export ENVIRONMENT=development
export LOG_LEVEL=DEBUG
export DATABASE_PASSWORD=dev_pass

# Staging
export ENVIRONMENT=staging
export LOG_LEVEL=INFO
export STAGING_DB_PASSWORD=secure_staging_pass
export STAGING_API_SECRET=staging_secret_key

# Production
export ENVIRONMENT=production
export LOG_LEVEL=WARNING
export DATABASE_PASSWORD=ultra_secure_prod_pass
export API_SECRET_KEY=production_secret_key
export REDIS_PASSWORD=redis_prod_pass
```

## Backup and Disaster Recovery

### Backup Types

1. **Database Backup**: PostgreSQL/TimescaleDB dump
2. **Models Backup**: Trained ML models and artifacts
3. **Configuration Backup**: All configuration files
4. **Disaster Recovery Package**: Complete system backup

### Automated Backups

Production environments automatically create backups:
- **Database**: Every 6 hours
- **Models**: Every 24 hours
- **Configuration**: Daily at 2 AM
- **Retention**: 30 days for production, 14 days for staging

### Manual Backup Creation

```bash
# Create full backup
python scripts/environment-manager.py backup production --type all

# Database only
python scripts/environment-manager.py backup production --type database

# Disaster recovery package
python scripts/environment-manager.py backup production --type disaster-recovery
```

### Backup Restoration

```python
from src.core.backup_manager import BackupManager

# Initialize backup manager
backup_manager = BackupManager("backups", config)

# List available backups
backups = backup_manager.list_backups("database")

# Restore database
backup_manager.restore_database_backup("db_20241210_143000")

# Restore models
backup_manager.restore_models_backup("models_20241210_143000")
```

### Disaster Recovery Procedure

1. **Assess the situation** and determine recovery scope
2. **Prepare new environment** (if needed)
3. **Restore from disaster recovery package**:
   ```bash
   # Find latest disaster recovery package
   ls backups/disaster_recovery_*/
   
   # Restore database
   python scripts/environment-manager.py backup production restore db_backup_id
   
   # Restore models
   python scripts/environment-manager.py backup production restore models_backup_id
   
   # Verify configuration
   python scripts/environment-manager.py validate production --test-connections
   ```

4. **Start services** and verify functionality
5. **Update DNS/routing** if environment changed
6. **Monitor** system health and performance

## Best Practices

### Security

1. **Never commit secrets** to version control
2. **Use environment variables** for sensitive data in CI/CD
3. **Rotate secrets regularly** (quarterly recommended)
4. **Limit secret access** to necessary personnel only
5. **Use encrypted secrets** in staging and production
6. **Audit secret access** and modifications

### Configuration Management

1. **Validate before deployment** with connection tests
2. **Use environment-specific overrides** instead of duplicating entire configs
3. **Document configuration changes** in commit messages
4. **Test configuration changes** in staging first
5. **Keep base configuration minimal** and environment-specific configs focused

### Deployment

1. **Always validate** configuration before deployment
2. **Create backups** before production deployments
3. **Deploy to staging first** for validation
4. **Monitor services** after deployment
5. **Have rollback procedures** ready
6. **Use health checks** to ensure service readiness

### Backup and Recovery

1. **Test backup restoration** regularly
2. **Store backups** in multiple locations
3. **Monitor backup completion** and failures
4. **Document recovery procedures** and keep them updated
5. **Practice disaster recovery** scenarios
6. **Encrypt sensitive backups**

## Troubleshooting

### Common Issues

#### Configuration Validation Failures
- Check required secrets are set
- Verify Home Assistant connectivity
- Ensure database is accessible
- Check MQTT broker availability

#### Deployment Failures
- Review Docker logs: `docker-compose logs haml-predictor`
- Check service health: `docker-compose ps`
- Verify environment variables are set
- Ensure ports are not in use

#### Secret Management Issues
- Check file permissions on `secrets/` directory
- Verify encryption key exists and is readable
- Ensure environment is correctly detected
- Check environment variable precedence

#### Backup Failures
- Verify disk space for backup storage
- Check database connectivity and permissions
- Ensure pg_dump/psql tools are available
- Verify backup directory permissions

### Diagnostic Commands

```bash
# Check environment detection
python -c "from src.core.environment import get_environment_manager; print(get_environment_manager().current_environment.value)"

# Test configuration loading
python -c "from src.core.environment import get_environment_manager; config = get_environment_manager().load_environment_config(); print('Config loaded successfully')"

# Validate all environments
for env in development staging production; do
  echo "Validating $env..."
  python scripts/environment-manager.py validate $env
done

# Check Docker services
docker-compose ps
docker-compose logs --tail=50 haml-predictor

# Check system resources
df -h  # Disk space
free -h  # Memory
docker system df  # Docker space usage
```

## Migration Between Environments

### Promoting from Staging to Production

1. **Validate staging deployment**:
   ```bash
   ./scripts/deploy-environment.sh staging --validate-only --test-connections
   ```

2. **Create production secrets** (copy from staging and update):
   ```bash
   python scripts/environment-manager.py secrets production setup
   ```

3. **Backup current production** (if exists):
   ```bash
   python scripts/environment-manager.py backup production --type disaster-recovery
   ```

4. **Deploy to production**:
   ```bash
   ./scripts/deploy-environment.sh production
   ```

5. **Verify production deployment**:
   ```bash
   python scripts/environment-manager.py validate production --test-connections
   ```

### Environment Cloning

To clone an environment's configuration:

1. **Export source environment**:
   ```bash
   python scripts/environment-manager.py export staging staging_template.yaml
   ```

2. **Create target environment configuration**:
   ```bash
   # Edit template and save as config/config.target.yaml
   ```

3. **Set up secrets for target environment**:
   ```bash
   python scripts/environment-manager.py secrets target setup
   ```

4. **Validate and deploy**:
   ```bash
   python scripts/environment-manager.py validate target
   ./scripts/deploy-environment.sh target
   ```

This comprehensive environment management system ensures secure, reliable, and consistent deployments across all environments while providing robust backup and recovery capabilities.