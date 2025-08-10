#!/bin/bash
# Health check script for Home Assistant ML Predictor Docker environment

set -e

echo "🔍 Health Check - Home Assistant ML Predictor Environment"
echo "================================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if Docker Compose is running
echo "📋 Checking Docker Compose services..."

services=("haml-predictor" "timescaledb" "mosquitto" "redis")
all_healthy=true

for service in "${services[@]}"; do
    status=$(docker-compose ps --format "table" | grep "$service" | awk '{print $3}' || echo "Down")
    
    if [ "$status" = "Up" ]; then
        echo -e "   ✅ ${GREEN}$service${NC}: Running"
    else
        echo -e "   ❌ ${RED}$service${NC}: $status"
        all_healthy=false
    fi
done

echo ""

# Check application health endpoint
echo "🤖 Checking application health endpoint..."
if curl -f -s http://localhost:8000/health > /dev/null 2>&1; then
    echo -e "   ✅ ${GREEN}API Health Check${NC}: OK"
    
    # Get detailed health information
    health_data=$(curl -s http://localhost:8000/health)
    echo "   📊 Health Details: $health_data"
else
    echo -e "   ❌ ${RED}API Health Check${NC}: Failed"
    all_healthy=false
fi

echo ""

# Check database connectivity
echo "🗄️ Checking database connectivity..."
if docker-compose exec -T timescaledb pg_isready -U occupancy_user -d occupancy_prediction > /dev/null 2>&1; then
    echo -e "   ✅ ${GREEN}Database${NC}: Ready"
    
    # Check TimescaleDB extension
    extensions=$(docker-compose exec -T timescaledb psql -U occupancy_user -d occupancy_prediction -c "SELECT extname FROM pg_extension WHERE extname = 'timescaledb';" -t 2>/dev/null || echo "")
    if [[ "$extensions" =~ "timescaledb" ]]; then
        echo -e "   ✅ ${GREEN}TimescaleDB Extension${NC}: Loaded"
    else
        echo -e "   ⚠️  ${YELLOW}TimescaleDB Extension${NC}: Not loaded"
    fi
else
    echo -e "   ❌ ${RED}Database${NC}: Not ready"
    all_healthy=false
fi

echo ""

# Check MQTT broker
echo "📡 Checking MQTT broker..."
if docker-compose exec -T mosquitto mosquitto_pub -h localhost -t "health/check" -m "test" > /dev/null 2>&1; then
    echo -e "   ✅ ${GREEN}MQTT Broker${NC}: Ready"
else
    echo -e "   ❌ ${RED}MQTT Broker${NC}: Not ready"
    all_healthy=false
fi

echo ""

# Check Redis
echo "🔴 Checking Redis cache..."
if docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; then
    echo -e "   ✅ ${GREEN}Redis${NC}: Ready"
else
    echo -e "   ❌ ${RED}Redis${NC}: Not ready"
    all_healthy=false
fi

echo ""

# Check resource usage
echo "💻 Resource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -n 10

echo ""

# Final status
if [ "$all_healthy" = true ]; then
    echo -e "🎉 ${GREEN}Overall Status: HEALTHY${NC}"
    echo "   All services are running and responding correctly"
    exit 0
else
    echo -e "⚠️  ${YELLOW}Overall Status: DEGRADED${NC}"
    echo "   Some services are not running correctly"
    echo "   Check the output above for details"
    exit 1
fi