#!/bin/bash
# Restore script for Home Assistant ML Predictor Docker environment

set -e

BACKUP_DIR="./backups"
BACKUP_NAME="$1"

if [ -z "$BACKUP_NAME" ]; then
    echo "❌ Usage: $0 <backup_name>"
    echo ""
    echo "📁 Available backups:"
    ls -1 "$BACKUP_DIR" | grep "_metadata.txt" | sed 's/_metadata.txt//' | sed 's/^/   /'
    exit 1
fi

if [ ! -f "$BACKUP_DIR/${BACKUP_NAME}_metadata.txt" ]; then
    echo "❌ Backup not found: $BACKUP_NAME"
    exit 1
fi

echo "🔄 Starting restore process for backup: $BACKUP_NAME"

# Show backup metadata
echo "📋 Backup Information:"
cat "$BACKUP_DIR/${BACKUP_NAME}_metadata.txt" | sed 's/^/   /'
echo ""

# Confirm restore
read -p "⚠️  This will replace current data. Continue? [y/N]: " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Restore cancelled"
    exit 1
fi

# Stop services
echo "🛑 Stopping services..."
docker-compose down

# Restore database
if [ -f "$BACKUP_DIR/${BACKUP_NAME}_database.sql" ]; then
    echo "🗄️ Restoring database..."
    
    # Start database only
    docker-compose up -d timescaledb
    
    # Wait for database
    echo "⏳ Waiting for database..."
    until docker-compose exec timescaledb pg_isready -U occupancy_user -d occupancy_prediction; do
        sleep 2
    done
    
    # Restore database
    docker-compose exec -T timescaledb psql -U occupancy_user -d occupancy_prediction < "$BACKUP_DIR/${BACKUP_NAME}_database.sql"
    echo "✅ Database restored"
else
    echo "⚠️ No database backup found, skipping..."
fi

# Restore application data
if [ -f "$BACKUP_DIR/${BACKUP_NAME}_app_data.tar.gz" ]; then
    echo "📁 Restoring application data..."
    
    # Remove existing volumes
    docker volume rm -f \
        haml-predictor_haml_models \
        haml-predictor_haml_data \
        haml-predictor_haml_logs 2>/dev/null || true
    
    # Restore data
    docker run --rm \
        -v haml-predictor_haml_models:/target/models \
        -v haml-predictor_haml_data:/target/data \
        -v haml-predictor_haml_logs:/target/logs \
        -v "$(pwd)/$BACKUP_DIR":/backup:ro \
        alpine:latest \
        sh -c "cd /target && tar xzf /backup/${BACKUP_NAME}_app_data.tar.gz"
    
    echo "✅ Application data restored"
else
    echo "⚠️ No application data backup found, skipping..."
fi

# Restore configuration
if [ -f "$BACKUP_DIR/${BACKUP_NAME}_config.tar.gz" ]; then
    echo "⚙️ Restoring configuration..."
    
    # Backup current config
    if [ -d "config" ] || [ -f ".env" ]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        mkdir -p "./config_backup"
        [ -d "config" ] && cp -r config "./config_backup/config_$timestamp" 2>/dev/null || true
        [ -f ".env" ] && cp .env "./config_backup/env_$timestamp" 2>/dev/null || true
        echo "   💾 Current config backed up to config_backup/"
    fi
    
    # Extract backup
    tar xzf "$BACKUP_DIR/${BACKUP_NAME}_config.tar.gz"
    echo "✅ Configuration restored"
else
    echo "⚠️ No configuration backup found, skipping..."
fi

echo ""
echo "🎉 Restore completed successfully!"
echo ""
echo "🚀 To start the restored environment:"
echo "   ./start.sh [prod] [monitoring]"