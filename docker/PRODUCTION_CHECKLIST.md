# Production Deployment Checklist

This checklist ensures a secure and reliable production deployment of the Home Assistant ML Predictor system.

## 📋 Pre-Deployment Checklist

### System Requirements
- [ ] **Hardware Requirements Met**
  - [ ] Minimum: 2 CPU cores, 4GB RAM, 20GB storage
  - [ ] Recommended: 4 CPU cores, 8GB RAM, 50GB SSD storage
  - [ ] Network connectivity to Home Assistant instance

- [ ] **Software Requirements**
  - [ ] Docker 20.10+ installed
  - [ ] Docker Compose 2.0+ installed
  - [ ] Git for code updates
  - [ ] Sufficient disk space for data and logs

### Configuration Security
- [ ] **Environment Configuration**
  - [ ] `.env` file created from template
  - [ ] Secure database passwords set (not default values)
  - [ ] Home Assistant URL and token configured
  - [ ] API secrets changed from defaults
  - [ ] MQTT credentials configured (if using authentication)

- [ ] **File Permissions**
  - [ ] Ensure `.env` file is not readable by other users
  - [ ] Management scripts are executable
  - [ ] Configuration files have appropriate permissions

### Network Security
- [ ] **Firewall Configuration**
  - [ ] Only necessary ports exposed (8000, 5432, 1883, etc.)
  - [ ] Consider using reverse proxy with SSL
  - [ ] Internal container communication secured
  - [ ] Home Assistant network connectivity verified

- [ ] **SSL/TLS** (Recommended)
  - [ ] Reverse proxy configured (nginx/traefik)
  - [ ] SSL certificates obtained and installed
  - [ ] HTTP redirects to HTTPS configured
  - [ ] Security headers configured

## 🚀 Deployment Steps

### 1. Initial Setup
```bash
# Clone repository
git clone <repository-url>
cd ha-ml-predictor/docker

# Validate setup
./validate-docker-setup.sh --test-build

# Configure environment
cp .env.template .env
# Edit .env with production values
```

### 2. First-Time Deployment
- [ ] **Database Initialization**
  ```bash
  # Start database only first
  docker-compose up -d timescaledb
  
  # Verify database health
  ./health-check.sh
  ```

- [ ] **Full System Start**
  ```bash
  # Start production environment
  ./start.sh prod
  
  # Verify all services
  ./health-check.sh
  ```

- [ ] **Initial Data Import** (Optional)
  ```bash
  # Import historical data if needed
  docker-compose exec haml-predictor python scripts/setup_database.py
  ```

### 3. Verification
- [ ] **Service Health Checks**
  - [ ] All containers running: `docker-compose ps`
  - [ ] API server responding: `curl http://localhost:8000/health`
  - [ ] Database connected: `./health-check.sh`
  - [ ] MQTT broker active: Test with MQTT client

- [ ] **Integration Testing**
  - [ ] Home Assistant connection working
  - [ ] Predictions being generated
  - [ ] MQTT messages being published
  - [ ] API endpoints responding correctly

## 📊 Monitoring Setup

### Essential Monitoring
- [ ] **Log Monitoring**
  - [ ] Application logs accessible
  - [ ] Log rotation configured
  - [ ] Error alerting set up (optional)

- [ ] **Health Monitoring**
  - [ ] Regular health checks scheduled
  - [ ] Database performance monitoring
  - [ ] Resource usage monitoring

### Optional Advanced Monitoring
- [ ] **Prometheus + Grafana** (if enabled)
  ```bash
  ./start.sh prod monitoring
  ```
  - [ ] Grafana dashboards configured
  - [ ] Alert rules configured
  - [ ] Data source connections verified

## 🔒 Security Hardening

### Application Security
- [ ] **Container Security**
  - [ ] Non-root user running application
  - [ ] Resource limits configured
  - [ ] No unnecessary privileges
  - [ ] Regular security updates scheduled

- [ ] **Data Security**
  - [ ] Database passwords secured
  - [ ] API tokens and secrets secured
  - [ ] Sensitive data not in logs
  - [ ] Regular backups configured

### Network Security
- [ ] **Access Control**
  - [ ] API access restricted (if needed)
  - [ ] Database not externally accessible
  - [ ] MQTT authentication configured
  - [ ] Rate limiting enabled

## 💾 Backup & Recovery

### Backup Strategy
- [ ] **Automated Backups**
  - [ ] Database backup schedule
  - [ ] Application data backup
  - [ ] Configuration backup
  
- [ ] **Manual Backup Test**
  ```bash
  ./backup.sh
  # Verify backup files created
  ```

- [ ] **Recovery Testing**
  ```bash
  # Test restore process
  ./restore.sh <backup_name>
  ```

### Disaster Recovery Plan
- [ ] **Documentation**
  - [ ] Recovery procedures documented
  - [ ] Contact information available
  - [ ] Escalation procedures defined

- [ ] **Recovery Testing**
  - [ ] Full system restore tested
  - [ ] Recovery time objectives met
  - [ ] Data integrity verified

## 🔄 Maintenance Procedures

### Regular Maintenance
- [ ] **Weekly Tasks**
  - [ ] Check system health
  - [ ] Review error logs
  - [ ] Verify backup completion
  - [ ] Monitor resource usage

- [ ] **Monthly Tasks**
  - [ ] Update system packages
  - [ ] Review security logs
  - [ ] Test disaster recovery
  - [ ] Performance optimization review

### Update Procedures
- [ ] **Application Updates**
  ```bash
  # Update procedure
  git pull
  docker-compose build --no-cache
  ./stop.sh prod
  ./start.sh prod
  ./health-check.sh
  ```

- [ ] **Database Updates**
  - [ ] Backup before updates
  - [ ] Test migrations in staging
  - [ ] Verify data integrity after updates

## 🚨 Troubleshooting

### Common Issues
- [ ] **Application Won't Start**
  - [ ] Check logs: `docker-compose logs haml-predictor`
  - [ ] Verify configuration: `.env` file
  - [ ] Check dependencies: Database, MQTT
  - [ ] Verify disk space and permissions

- [ ] **Database Connection Issues**
  - [ ] Check database status: `./health-check.sh`
  - [ ] Verify credentials in `.env`
  - [ ] Check network connectivity
  - [ ] Review database logs

- [ ] **MQTT Integration Issues**
  - [ ] Test MQTT connectivity
  - [ ] Verify Home Assistant integration
  - [ ] Check MQTT broker logs
  - [ ] Validate topic subscriptions

### Performance Issues
- [ ] **High Resource Usage**
  - [ ] Monitor with: `docker stats`
  - [ ] Check application logs for errors
  - [ ] Review resource limits
  - [ ] Consider scaling options

- [ ] **Slow Response Times**
  - [ ] Check API latency: `/health` endpoint
  - [ ] Monitor database performance
  - [ ] Review prediction generation times
  - [ ] Check network connectivity

## 📞 Support & Documentation

### Documentation
- [ ] **Available Resources**
  - [ ] README.md - Complete setup guide
  - [ ] API documentation at `/docs`
  - [ ] Configuration examples
  - [ ] Troubleshooting guides

### Getting Help
- [ ] **Support Channels**
  - [ ] GitHub issues for bug reports
  - [ ] Documentation for common questions
  - [ ] Log analysis for troubleshooting

---

## ✅ Final Production Readiness Verification

Before declaring the system production-ready:

1. **All checklist items completed** ✅
2. **Full system test performed** ✅
3. **Monitoring and alerting active** ✅
4. **Backup and recovery verified** ✅
5. **Security hardening applied** ✅
6. **Documentation updated** ✅

**Production Deployment Date:** ___________

**Deployed By:** ___________

**System Version:** ___________

---

*This checklist should be completed for every production deployment and kept as a record of the deployment process.*