#!/bin/bash
# Start script for Home Assistant ML Predictor Docker environment

set -e

echo "🚀 Starting Home Assistant ML Predictor Docker Environment"

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  .env file not found. Creating from template..."
    cp .env.template .env
    echo "✅ Please edit .env file with your configuration before continuing"
    exit 1
fi

# Check if we're in production mode
if [ "$1" = "prod" ]; then
    echo "🏭 Starting in PRODUCTION mode"
    COMPOSE_FILES="-f docker-compose.yml -f docker-compose.prod.yml"
else
    echo "🔧 Starting in DEVELOPMENT mode"
    COMPOSE_FILES="-f docker-compose.yml"
fi

# Build images if needed
echo "🔨 Building Docker images..."
docker-compose $COMPOSE_FILES build --no-cache haml-predictor

# Start core services first
echo "📊 Starting database and MQTT services..."
docker-compose $COMPOSE_FILES up -d timescaledb mosquitto redis

# Wait for database to be ready
echo "⏳ Waiting for database to be ready..."
until docker-compose $COMPOSE_FILES exec timescaledb pg_isready -U occupancy_user -d occupancy_prediction; do
    echo "Database not ready, waiting..."
    sleep 2
done

echo "✅ Database is ready"

# Run database migrations if needed
echo "🔄 Running database setup..."
docker-compose $COMPOSE_FILES exec haml-predictor python scripts/setup_database.py || true

# Start the main application
echo "🤖 Starting ML Predictor application..."
docker-compose $COMPOSE_FILES up -d haml-predictor

# Start monitoring services if requested
if [ "$2" = "monitoring" ]; then
    echo "📈 Starting monitoring services..."
    docker-compose $COMPOSE_FILES --profile monitoring up -d
fi

echo ""
echo "🎉 Startup complete!"
echo ""
echo "📋 Service URLs:"
echo "   🤖 ML Predictor API: http://localhost:8000"
echo "   📊 API Documentation: http://localhost:8000/docs"
echo "   🗄️  Database: localhost:5432"
echo "   📡 MQTT Broker: localhost:1883"
echo "   🔴 Redis: localhost:6379"

if [ "$2" = "monitoring" ]; then
    echo "   📈 Grafana: http://localhost:3000"
    echo "   📊 Prometheus: http://localhost:9090"
fi

echo ""
echo "🔍 To view logs: docker-compose logs -f haml-predictor"
echo "🛑 To stop: docker-compose down"