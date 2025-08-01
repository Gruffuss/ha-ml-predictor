# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@implementation-plan.md
@occupancy-architecture.md

## Project Overview

This is a **Home Assistant ML Predictor** project for room occupancy prediction. Sprint 1 (Foundation & Data Infrastructure) is complete with full implementation.

**Status**: Sprint 1 Complete ✅ - Foundation and data infrastructure implemented, ready for Sprint 2

## Project Architecture

### Core System Components

The system is designed as a machine learning pipeline with the following key components:

1. **Data Ingestion Layer**
   - Home Assistant API integration (WebSocket + REST)
   - Event processing pipeline with human/cat movement detection
   - TimescaleDB event storage

2. **Feature Engineering Engine**
   - Temporal features (time since last occupancy, cyclical encodings)
   - Sequential features (room transitions, movement patterns)
   - Contextual features (environmental data, cross-room correlations)

3. **Prediction Engine**
   - Ensemble model architecture with multiple base models:
     - LSTM Networks for sequence patterns
     - XGBoost for temporal features
     - Hidden Markov Models for state transitions
     - Gaussian Process for uncertainty quantification
   - Meta-learner with stacking ensemble
   - Per-room model specialization

4. **Self-Adaptation System**
   - Real-time prediction validation
   - Concept drift detection
   - Continuous learning with online updates

5. **Integration Layer**
   - MQTT publisher for Home Assistant integration
   - REST API for manual control and monitoring

### Directory Structure (Implemented ✅)

```
ha-ml-predictor/
├── config/                    # Configuration files ✅
│   ├── config.yaml           # Main config (HA, DB, MQTT)
│   ├── rooms.yaml            # Room and sensor mappings
│   └── logging.yaml          # Logging configuration
├── src/
│   ├── core/                 # Core system ✅
│   │   ├── config.py         # Configuration management
│   │   ├── constants.py      # System constants and enums
│   │   └── exceptions.py     # Custom exception classes
│   ├── data/                 # Data layer ✅
│   │   ├── ingestion/        # HA integration
│   │   │   ├── ha_client.py  # HA WebSocket/REST client
│   │   │   ├── event_processor.py # Event validation/processing
│   │   │   └── bulk_importer.py   # Historical data import
│   │   └── storage/          # Database layer
│   │       ├── models.py     # SQLAlchemy models
│   │       └── database.py   # Connection management
│   ├── features/             # Feature engineering (Sprint 2)
│   ├── models/               # ML models (Sprint 3)
│   ├── adaptation/           # Self-adaptation (Sprint 4)
│   ├── integration/          # MQTT/API (Sprint 5)
│   └── utils/                # Logging, metrics
├── scripts/                  # Setup scripts ✅
│   └── setup_database.py     # Database initialization
├── tests/                    # Testing framework
├── logs/                     # Application logs
├── requirements.txt          # Python dependencies ✅
└── TODO.md                   # Progress tracking ✅
```

## Development Commands

### Initial Setup ✅
```bash
# Setup Python environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Setup TimescaleDB database
python scripts/setup_database.py

# Import 6 months of historical HA data
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

### Development Workflow
```bash
# Run tests (Sprint 6)
pytest tests/

# Run linting
black src/ tests/
flake8 src/ tests/

# Start real-time event processing
python -c "
from src.data.ingestion.ha_client import HomeAssistantClient
from src.data.ingestion.event_processor import EventProcessor
import asyncio

async def start_processing():
    async with HomeAssistantClient() as client:
        processor = EventProcessor()
        # Process real-time events
        pass

asyncio.run(start_processing())
"

# Database health check
python -c "
from src.data.storage.database import get_database_manager
import asyncio

async def check_health():
    db_manager = await get_database_manager()
    health = await db_manager.health_check()
    print(f'Database health: {health}')

asyncio.run(check_health())
"
```

### Docker Deployment (Sprint 7)
```bash
docker-compose up -d
```

## Key Implementation Notes

### Technology Stack ✅
- **Language**: Python 3.11+
- **Web Framework**: AsyncIO, aiohttp, websockets
- **Database**: PostgreSQL with TimescaleDB extension, AsyncPG, SQLAlchemy 2.0
- **ML Libraries**: scikit-learn (including MLPRegressor for LSTM), XGBoost (Sprint 3)
- **Integration**: Home Assistant API, MQTT (paho-mqtt), FastAPI (Sprint 5)
- **Configuration**: YAML, Pydantic dataclasses
- **Development**: Black, Flake8, pytest, mypy

### Code Standards (From Planning Documents)
- Follow PEP 8 with 88-character line limit (Black formatter)
- Use type hints for all function signatures
- Docstrings for all public methods (Google style)
- Async/await for all I/O operations
- One class per file for major components
- Custom exceptions for domain-specific errors

### Performance Requirements (Planned)
- Prediction generation < 100ms
- Feature computation < 500ms
- Model update < 5 minutes
- Average prediction error < 15 minutes

## Home Assistant Integration

The system will integrate with Home Assistant via:
- **Data Input**: WebSocket connection for real-time sensor events
- **Data Output**: MQTT topics for predictions
  - `occupancy/predictions/{room}/next_occupied_time`
  - `occupancy/predictions/{room}/next_vacant_time`
  - `occupancy/predictions/{room}/confidence`

## Current State ✅

**Sprint 1 Complete**: Foundation & Data Infrastructure implemented and committed to git
- ✅ Complete project structure with all core modules
- ✅ Configuration system with YAML loading (improved nested structure)
- ✅ TimescaleDB database schema with hypertables and indexes
- ✅ Home Assistant WebSocket/REST API integration
- ✅ Event processing pipeline with human/cat movement detection
- ✅ Bulk historical data import (6 months capability)
- ✅ Database connection management with health monitoring
- ✅ Comprehensive exception handling and logging setup

**Git Repository**: ✅ Published to GitHub with comprehensive test suite
- **GitHub URL**: https://github.com/Gruffuss/ha-ml-predictor
- **3 commits** covering complete Sprint 1 implementation
- **6,000+ lines of test code** with 200+ test methods
- **Complete test infrastructure** for unit, integration, and validation testing

**Test Suite**: ✅ Comprehensive Sprint 1 validation completed
- Unit tests for all core components (config, database, HA integration)
- Integration tests for database operations and HA API
- Mock strategies for external dependencies
- Automated test runner with coverage reporting
- Test fixtures for realistic data scenarios

**LXC Container**: 🔄 Environment setup in progress
- **IP**: 192.168.51.10 (Ubuntu 24.04, 2 cores, 6GB RAM)
- **Password**: ha-ml-predictor
- **Status**: Ready for deployment and testing

**Next Steps**: Complete LXC setup and validate Sprint 1 before Sprint 2

## Implementation Methods Tracker

### Core System (`src/core/`) ✅
| Method | Purpose | Status |
|--------|---------|--------|
| `ConfigLoader.load_config()` | Load YAML configuration | ✅ |
| `get_config()` | Global config instance | ✅ |
| `SystemConfig.get_all_entity_ids()` | Extract all HA entity IDs | ✅ |
| `RoomConfig.get_sensors_by_type()` | Filter sensors by type | ✅ |

### Database (`src/data/storage/`) ✅
| Method | Purpose | Status |
|--------|---------|--------|
| `DatabaseManager.get_engine()` | SQLAlchemy async engine | ✅ |
| `get_db_session()` | Session context manager | ✅ |
| `SensorEvent.bulk_create()` | Bulk insert events | ✅ |
| `SensorEvent.get_recent_events()` | Query recent events | ✅ |
| `RoomState.get_current_state()` | Current room occupancy | ✅ |

### Home Assistant Integration (`src/data/ingestion/`) ✅
| Method | Purpose | Status |
|--------|---------|--------|
| `HomeAssistantClient.connect()` | WebSocket connection | ✅ |
| `HomeAssistantClient.subscribe_to_events()` | Real-time events | ✅ |
| `HomeAssistantClient.get_entity_history()` | Historical data | ✅ |
| `EventProcessor.process_event()` | Event validation/processing | ✅ |
| `BulkImporter.import_historical_data()` | Import 6 months data | ✅ |
| `MovementPatternClassifier.classify()` | Human vs cat detection | ✅ |