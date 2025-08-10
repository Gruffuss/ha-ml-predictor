#!/bin/bash
# Stop script for Home Assistant ML Predictor Docker environment

set -e

echo "🛑 Stopping Home Assistant ML Predictor Docker Environment"

# Determine compose files
if [ "$1" = "prod" ]; then
    echo "🏭 Stopping PRODUCTION environment"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
else
    echo "🔧 Stopping DEVELOPMENT environment"
    COMPOSE_FILES="-f docker-compose.yml"
fi

# Stop all services
echo "🔄 Stopping all services..."
docker-compose $COMPOSE_FILES --profile monitoring down

# Option to remove volumes
if [ "$2" = "clean" ]; then
    echo "🧹 Removing volumes and cleaning up..."
    docker-compose $COMPOSE_FILES down -v --remove-orphans
    docker system prune -f
    echo "✅ Environment cleaned"
else
    echo "💾 Data volumes preserved"
    echo "   Use 'stop.sh [mode] clean' to remove all data"
fi

echo ""
echo "✅ Environment stopped successfully"
echo ""
echo "🔄 To start again: ./start.sh [prod] [monitoring]"