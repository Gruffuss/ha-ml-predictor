#!/bin/bash
# Docker setup validation script for Home Assistant ML Predictor

set -e

echo "🔍 Docker Setup Validation"
echo "=========================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[1;34m'
NC='\033[0m' # No Color

# Validation functions
check_docker() {
    echo -e "${BLUE}📦 Checking Docker installation...${NC}"
    if command -v docker &> /dev/null; then
        echo -e "   ✅ ${GREEN}Docker found:${NC} $(docker --version)"
    else
        echo -e "   ❌ ${RED}Docker not found${NC}"
        exit 1
    fi
}

check_docker_compose() {
    echo -e "${BLUE}🐙 Checking Docker Compose...${NC}"
    if command -v docker-compose &> /dev/null; then
        echo -e "   ✅ ${GREEN}Docker Compose found:${NC} $(docker-compose --version)"
    else
        echo -e "   ❌ ${RED}Docker Compose not found${NC}"
        exit 1
    fi
}

validate_dockerfile() {
    echo -e "${BLUE}🏗️ Validating Dockerfile...${NC}"
    
    if [ ! -f "Dockerfile" ]; then
        echo -e "   ❌ ${RED}Dockerfile not found${NC}"
        return 1
    fi
    
    # Check for multi-stage build
    if grep -q "FROM.*as.*" Dockerfile; then
        echo -e "   ✅ ${GREEN}Multi-stage build detected${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}No multi-stage build found${NC}"
    fi
    
    # Check for non-root user
    if grep -q "USER.*haml" Dockerfile; then
        echo -e "   ✅ ${GREEN}Non-root user configured${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}No non-root user found${NC}"
    fi
    
    # Check for health check
    if grep -q "HEALTHCHECK" Dockerfile; then
        echo -e "   ✅ ${GREEN}Health check configured${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}No health check found${NC}"
    fi
}

validate_compose_file() {
    echo -e "${BLUE}📋 Validating docker-compose.yml...${NC}"
    
    if [ ! -f "docker-compose.yml" ]; then
        echo -e "   ❌ ${RED}docker-compose.yml not found${NC}"
        return 1
    fi
    
    # Validate compose file syntax
    if docker-compose config &> /dev/null; then
        echo -e "   ✅ ${GREEN}Compose file syntax valid${NC}"
    else
        echo -e "   ❌ ${RED}Compose file syntax invalid${NC}"
        return 1
    fi
    
    # Check for required services
    services=("haml-predictor" "timescaledb" "mosquitto" "redis")
    for service in "${services[@]}"; do
        if docker-compose config --services | grep -q "$service"; then
            echo -e "   ✅ ${GREEN}Service '$service' configured${NC}"
        else
            echo -e "   ⚠️  ${YELLOW}Service '$service' missing${NC}"
        fi
    done
    
    # Check for networks
    if docker-compose config | grep -q "networks:"; then
        echo -e "   ✅ ${GREEN}Custom networks configured${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}No custom networks found${NC}"
    fi
    
    # Check for volumes
    if docker-compose config | grep -q "volumes:"; then
        echo -e "   ✅ ${GREEN}Persistent volumes configured${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}No persistent volumes found${NC}"
    fi
}

validate_environment() {
    echo -e "${BLUE}⚙️ Validating environment configuration...${NC}"
    
    if [ ! -f ".env.template" ]; then
        echo -e "   ❌ ${RED}.env.template not found${NC}"
        return 1
    fi
    
    # Check for required environment variables
    required_vars=("HA_URL" "HA_TOKEN" "DATABASE_PASSWORD")
    for var in "${required_vars[@]}"; do
        if grep -q "$var=" .env.template; then
            echo -e "   ✅ ${GREEN}Variable '$var' in template${NC}"
        else
            echo -e "   ⚠️  ${YELLOW}Variable '$var' missing from template${NC}"
        fi
    done
    
    if [ -f ".env" ]; then
        echo -e "   ℹ️  ${BLUE}Custom .env file exists${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}No .env file (will use template)${NC}"
    fi
}

validate_supporting_files() {
    echo -e "${BLUE}📁 Validating supporting files...${NC}"
    
    # Check for init scripts
    if [ -d "init-scripts" ]; then
        echo -e "   ✅ ${GREEN}Database init scripts directory exists${NC}"
        if [ -f "init-scripts/01-init-timescale.sql" ]; then
            echo -e "   ✅ ${GREEN}TimescaleDB init script found${NC}"
        else
            echo -e "   ⚠️  ${YELLOW}TimescaleDB init script missing${NC}"
        fi
    else
        echo -e "   ⚠️  ${YELLOW}No init-scripts directory${NC}"
    fi
    
    # Check for MQTT config
    if [ -f "mosquitto/mosquitto.conf" ]; then
        echo -e "   ✅ ${GREEN}Mosquitto configuration found${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}Mosquitto configuration missing${NC}"
    fi
    
    # Check for monitoring configs
    if [ -f "monitoring/prometheus.yml" ]; then
        echo -e "   ✅ ${GREEN}Prometheus configuration found${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}Prometheus configuration missing${NC}"
    fi
    
    # Check for management scripts
    scripts=("start.sh" "stop.sh" "health-check.sh" "backup.sh" "restore.sh")
    for script in "${scripts[@]}"; do
        if [ -f "$script" ]; then
            echo -e "   ✅ ${GREEN}Script '$script' exists${NC}"
        else
            echo -e "   ⚠️  ${YELLOW}Script '$script' missing${NC}"
        fi
    done
    
    # Check for .dockerignore
    if [ -f ".dockerignore" ]; then
        echo -e "   ✅ ${GREEN}.dockerignore exists${NC}"
    else
        echo -e "   ⚠️  ${YELLOW}.dockerignore missing${NC}"
    fi
}

test_build() {
    echo -e "${BLUE}🔨 Testing Docker build...${NC}"
    
    if docker build -t haml-predictor:test -f Dockerfile .. &> /dev/null; then
        echo -e "   ✅ ${GREEN}Docker build successful${NC}"
        
        # Clean up test image
        docker rmi haml-predictor:test &> /dev/null
    else
        echo -e "   ❌ ${RED}Docker build failed${NC}"
        echo "   Run 'docker build -t haml-predictor:test -f Dockerfile ..' for details"
        return 1
    fi
}

# Run validations
echo ""
check_docker
echo ""
check_docker_compose
echo ""
validate_dockerfile
echo ""
validate_compose_file
echo ""
validate_environment
echo ""
validate_supporting_files
echo ""

# Optional build test (can be slow)
if [ "$1" = "--test-build" ]; then
    test_build
    echo ""
fi

echo -e "${GREEN}🎉 Docker setup validation complete!${NC}"
echo ""
echo -e "${BLUE}📋 Next Steps:${NC}"
echo "   1. Copy .env.template to .env and configure your settings"
echo "   2. Run './start.sh' to start the development environment"
echo "   3. Run './start.sh prod' to start the production environment"
echo "   4. Run './health-check.sh' to verify all services are running"
echo ""
echo -e "${BLUE}📖 Documentation:${NC}"
echo "   Full instructions available in README.md"