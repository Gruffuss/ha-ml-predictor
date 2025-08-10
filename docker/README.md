# Home Assistant ML Predictor - Docker Deployment

This directory contains the complete Docker containerization setup for the Home Assistant ML Predictor system, providing production-ready deployment with monitoring, security hardening, and automated management.

## 🏗️ Architecture Overview

The Docker deployment includes the following services:

- **haml-predictor**: Main ML prediction application
- **timescaledb**: PostgreSQL with TimescaleDB for time-series data
- **mosquitto**: MQTT broker for Home Assistant integration
- **redis**: Caching and session management
- **prometheus**: Metrics collection (optional)
- **grafana**: Visualization and monitoring (optional)

## 🚀 Quick Start

### Prerequisites

- Docker 20.10+ and Docker Compose 2.0+
- 4GB+ RAM available for containers
- Home Assistant instance with API access token

### 1. Initial Setup

```bash
# Clone the repository
cd docker/

# Copy environment template
cp .env.template .env

# Edit configuration with your values
nano .env  # or use your preferred editor
```

### 2. Configure Environment

Edit `.env` file with your settings:

```bash
# Home Assistant Configuration
HA_URL=http://your-ha-instance:8123
HA_TOKEN=your_long_lived_access_token

# Database Configuration (defaults are fine)
DATABASE_PASSWORD=your_secure_password

# Optional: Change default ports if needed
API_PORT=8000
```

### 3. Start the System

**Development Mode:**
```bash
./start.sh
```

**Production Mode:**
```bash
./start.sh prod
```

**With Monitoring:**
```bash
./start.sh prod monitoring
```

### 4. Verify Deployment

```bash
# Check all services
./health-check.sh

# View logs
docker-compose logs -f haml-predictor
```

## 📋 Service URLs

Once started, access the services at:

- **🤖 ML Predictor API**: http://localhost:8000
- **📊 API Documentation**: http://localhost:8000/docs
- **🗄️ Database**: localhost:5432
- **📡 MQTT Broker**: localhost:1883
- **🔴 Redis**: localhost:6379
- **📈 Grafana**: http://localhost:3000 (if monitoring enabled)
- **📊 Prometheus**: http://localhost:9090 (if monitoring enabled)

## 🔧 Configuration Files

### Core Configuration

| File | Purpose |
|------|---------|
| `docker-compose.yml` | Main service definitions |
| `docker-compose.prod.yml` | Production overrides |
| `Dockerfile` | Multi-stage application build |
| `.env` | Environment variables |

### Service Configuration

| File | Purpose |
|------|---------|
| `init-scripts/01-init-timescale.sql` | Database initialization |
| `mosquitto/mosquitto.conf` | MQTT broker configuration |
| `monitoring/prometheus.yml` | Metrics collection setup |
| `monitoring/grafana/provisioning/` | Dashboard and datasource setup |

## 🛠️ Management Scripts

### Start/Stop Operations

```bash
# Start development environment
./start.sh

# Start production environment
./start.sh prod

# Start with monitoring
./start.sh prod monitoring

# Stop environment (preserve data)
./stop.sh

# Stop and clean all data
./stop.sh dev clean
```

### Health Monitoring

```bash
# Comprehensive health check
./health-check.sh

# View service status
docker-compose ps

# View resource usage
docker stats
```

### Backup & Restore

```bash
# Create backup
./backup.sh

# List available backups
./restore.sh

# Restore specific backup
./restore.sh haml_backup_20240115_143022
```

## 🔒 Security Features

### Application Security

- **Non-root user**: Application runs as `haml` user
- **Minimal attack surface**: Multi-stage build with only runtime dependencies
- **Resource limits**: Memory and CPU constraints prevent resource exhaustion
- **Health checks**: Automatic container restart on failure

### Network Security

- **Isolated network**: All services run in dedicated bridge network
- **Internal communication**: Services communicate via internal network names
- **Exposed ports**: Only necessary ports are exposed to host

### Data Security

- **Volume persistence**: Data persists across container restarts
- **Encrypted connections**: Database connections use secure authentication
- **Secret management**: Sensitive data via environment variables

## 📊 Monitoring & Logging

### Application Logging

```bash
# View live logs
docker-compose logs -f haml-predictor

# View logs for all services
docker-compose logs -f

# View logs with timestamps
docker-compose logs -f --timestamps
```

### Health Monitoring

```bash
# Check application health
curl http://localhost:8000/health

# Check database health
docker-compose exec timescaledb pg_isready -U occupancy_user

# Check MQTT broker
docker-compose exec mosquitto mosquitto_pub -h localhost -t test -m "ping"
```

### Performance Monitoring

If monitoring is enabled:

1. **Grafana Dashboard**: http://localhost:3000
   - Default login: admin/admin
   - Pre-configured dashboards for system metrics

2. **Prometheus Metrics**: http://localhost:9090
   - Application and system metrics
   - Custom queries and alerts

## 🔧 Customization

### Resource Allocation

Edit `docker-compose.prod.yml` to adjust resource limits:

```yaml
services:
  haml-predictor:
    deploy:
      resources:
        limits:
          memory: 4G    # Adjust based on your needs
          cpus: '2.0'   # Adjust based on your needs
```

### Port Configuration

Change ports in `.env` file:

```bash
API_PORT=8001           # Change API port
GRAFANA_PORT=3001      # Change Grafana port
PROMETHEUS_PORT=9091   # Change Prometheus port
```

### External Services

To use external database or MQTT broker:

```bash
# In .env file
DATABASE_HOST=your.external.db.com
MQTT_BROKER=your.external.mqtt.broker.com

# Then disable internal services
docker-compose up -d haml-predictor  # Only start the app
```

## 🚨 Troubleshooting

### Common Issues

**Application won't start:**
```bash
# Check logs
docker-compose logs haml-predictor

# Verify configuration
./health-check.sh

# Restart services
docker-compose restart
```

**Database connection issues:**
```bash
# Check database status
docker-compose exec timescaledb pg_isready

# View database logs
docker-compose logs timescaledb

# Reset database
docker-compose down
docker volume rm haml-predictor_timescale_data
./start.sh
```

**MQTT connection issues:**
```bash
# Test MQTT connectivity
docker-compose exec mosquitto mosquitto_pub -h localhost -t test -m "hello"

# View MQTT logs
docker-compose logs mosquitto
```

### Performance Issues

**High memory usage:**
```bash
# Check resource usage
docker stats

# Adjust limits in docker-compose.prod.yml
# Restart with new limits
./stop.sh prod
./start.sh prod
```

**Slow predictions:**
```bash
# Check application logs for performance metrics
docker-compose logs haml-predictor | grep -i "prediction\|latency"

# Monitor system resources
docker stats --no-stream
```

## 🔄 Updates and Maintenance

### Updating the Application

```bash
# Pull latest code
git pull

# Rebuild containers
docker-compose build --no-cache

# Restart with new version
./stop.sh
./start.sh
```

### Database Maintenance

```bash
# Create database backup
./backup.sh

# Access database for maintenance
docker-compose exec timescaledb psql -U occupancy_user -d occupancy_prediction

# View database size
docker-compose exec timescaledb du -sh /var/lib/postgresql/data
```

### Log Rotation

Logs are automatically rotated in production mode. Manual cleanup:

```bash
# Clear application logs
docker-compose exec haml-predictor find /app/logs -name "*.log" -mtime +7 -delete

# Clear container logs
docker system prune -f
```

## 📈 Production Deployment

### Recommended Hardware

- **Minimum**: 2 CPU cores, 4GB RAM, 20GB storage
- **Recommended**: 4 CPU cores, 8GB RAM, 50GB SSD storage
- **Network**: Stable connection to Home Assistant instance

### Production Checklist

- [ ] Configure secure passwords in `.env`
- [ ] Set up external backup strategy
- [ ] Configure log monitoring and alerting
- [ ] Set up reverse proxy with SSL (nginx/traefik)
- [ ] Configure firewall rules
- [ ] Set up automated health monitoring
- [ ] Plan disaster recovery procedures

### High Availability Setup

For production environments requiring high availability:

1. **Load Balancer**: Use nginx or traefik for load balancing
2. **Database Replication**: Configure PostgreSQL streaming replication
3. **Container Orchestration**: Consider Kubernetes for larger deployments
4. **Monitoring**: Set up comprehensive monitoring with alerting

## 📞 Support

For issues and questions:

1. Check the troubleshooting section above
2. Review application logs: `docker-compose logs haml-predictor`
3. Run health check: `./health-check.sh`
4. Check GitHub issues and documentation

## 📝 License

This Docker configuration is part of the Home Assistant ML Predictor project and follows the same license terms.