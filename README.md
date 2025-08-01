# Home Assistant ML Predictor

An intelligent room occupancy prediction system that integrates with Home Assistant to predict when rooms will be occupied or vacant using machine learning.

## Project Status

**Sprint 1 Complete ✅** - Foundation & Data Infrastructure
- ✅ Core configuration system with YAML loading
- ✅ TimescaleDB database schema and models
- ✅ Home Assistant WebSocket/REST API integration
- ✅ Event processing pipeline with human/cat movement detection
- ✅ Bulk historical data import (6 months capability)
- ✅ Database connection management and health monitoring

**Next: Sprint 2** - Feature Engineering Pipeline

## Architecture

The system follows a 7-sprint implementation plan with:

1. **Data Ingestion Layer** - Home Assistant API integration
2. **Feature Engineering Engine** - Temporal, sequential, and contextual features
3. **Prediction Engine** - Ensemble ML models (XGBoost, MLPRegressor, HMM, GP)
4. **Self-Adaptation System** - Real-time accuracy monitoring and model updates
5. **Integration Layer** - MQTT publisher and REST API

## Hardware Requirements

- **Target**: LXC container with 2 cores, 6GB RAM
- **Database**: PostgreSQL with TimescaleDB extension
- **ML Libraries**: scikit-learn (lightweight, no TensorFlow)

## Home Assistant Integration

**Data Sources:**
- Motion/presence sensors from all rooms
- Door sensors
- Environmental sensors (temperature, humidity, light)
- 6 months of historical data for training

**Data Outputs:**
- MQTT topics: `occupancy/predictions/{room}/next_occupied_time`
- Real-time occupancy predictions with confidence intervals

## Quick Start

### Setup
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup TimescaleDB database
python scripts/setup_database.py

# Import historical data
python -c "
from src.data.ingestion.bulk_importer import BulkImporter, ImportConfig
import asyncio

async def import_data():
    config = ImportConfig(months_to_import=6, batch_size=1000)
    importer = BulkImporter(import_config=config)
    progress = await importer.import_historical_data()
    print(f'Imported {progress.valid_events} events')

asyncio.run(import_data())
"
```

### Configuration

Edit `config/config.yaml` with your Home Assistant details:
```yaml
home_assistant:
  url: "http://your-ha-ip:8123"
  token: "your-long-lived-access-token"
```

Edit `config/rooms.yaml` with your sensor mappings.

### Development

See `CLAUDE.md` for detailed development guidelines and `TODO.md` for current progress.

## Project Structure

```
ha-ml-predictor/
├── config/                    # YAML configuration files
├── src/
│   ├── core/                  # Configuration, constants, exceptions ✅
│   ├── data/                  # Database models & HA integration ✅
│   ├── features/              # Feature engineering (Sprint 2)
│   ├── models/                # ML models (Sprint 3)
│   ├── adaptation/            # Self-adaptation (Sprint 4)
│   ├── integration/           # MQTT/API (Sprint 5)
│   └── utils/                 # Logging, metrics
├── scripts/                   # Setup and maintenance scripts ✅
├── tests/                     # Unit and integration tests
└── logs/                      # Application logs
```

## License

This project is developed for personal Home Assistant integration.