#!/bin/bash
# Backup script for Home Assistant ML Predictor Docker environment

set -e

BACKUP_DIR="./backups"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="haml_backup_$TIMESTAMP"

echo "💾 Starting backup process..."

# Create backup directory
mkdir -p "$BACKUP_DIR"

# Backup database
echo "🗄️ Backing up database..."
docker-compose exec -T timescaledb pg_dump -U occupancy_user -d occupancy_prediction --clean --if-exists > "$BACKUP_DIR/${BACKUP_NAME}_database.sql"

# Backup application data volumes
echo "📁 Backing up application data..."
docker run --rm \
    -v haml-predictor_haml_models:/source/models:ro \
    -v haml-predictor_haml_data:/source/data:ro \
    -v haml-predictor_haml_logs:/source/logs:ro \
    -v "$(pwd)/$BACKUP_DIR":/backup \
    alpine:latest \
    sh -c "cd /source && tar czf /backup/${BACKUP_NAME}_app_data.tar.gz ."

# Backup configuration
echo "⚙️ Backing up configuration..."
tar czf "$BACKUP_DIR/${BACKUP_NAME}_config.tar.gz" \
    config/ \
    docker/ \
    .env 2>/dev/null || true

# Create backup metadata
cat > "$BACKUP_DIR/${BACKUP_NAME}_metadata.txt" << EOF
Backup Created: $(date)
Backup Name: $BACKUP_NAME
Environment: $(if [ -f .env ]; then echo "Custom"; else echo "Template"; fi)
Services Running: $(docker-compose ps --services --filter "status=running" | tr '\n' ' ')
Docker Compose Version: $(docker-compose version --short)
System: $(uname -a)
EOF

echo ""
echo "✅ Backup completed successfully!"
echo "📁 Backup location: $BACKUP_DIR/$BACKUP_NAME*"
echo "📊 Backup files:"
ls -lh "$BACKUP_DIR/${BACKUP_NAME}"*

echo ""
echo "🔄 To restore this backup, use: ./restore.sh $BACKUP_NAME"