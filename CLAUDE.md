# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

@implementation-plan.md
@occupancy-architecture.md

## ⚠️ CRITICAL: Error Handling Philosophy

**NEVER downplay errors, warnings, or test failures. This pattern has caused significant wasted time and broken systems.**

### Absolute Rules:
1. **FAILING TESTS = BROKEN SYSTEM** - Never declare success with failing tests
2. **ALL WARNINGS MATTER** - Investigate every warning, don't dismiss as "minor"  
3. **FIX COMPLETELY** - Don't move forward with "mostly working" systems
4. **NO PREMATURE SUCCESS** - Only claim completion when ALL validation passes

### Red Flags - Never Say:
- "Just minor test issues"
- "Mostly working, small cleanup needed"
- "Core functionality works despite failing tests"
- "We can fix these later"

### Green Flags - Only Proceed When:
- ✅ All tests pass completely
- ✅ All warnings addressed or explicitly justified
- ✅ All error messages resolved
- ✅ System fully validated and proven functional

**Remember: Broken tests mean broken system. No exceptions.**

## 📋 Function Implementation Tracking - MANDATORY

**⚠️ CRITICAL: ALL AGENTS MUST UPDATE TODO.md FUNCTION TRACKER**

### When Creating New Functions:
1. **BEFORE implementing**: Check TODO.md function tracker for existing implementations
2. **DURING implementation**: Add function to appropriate Sprint section in TODO.md
3. **AFTER implementation**: Mark function as ✅ completed with description

### Function Tracker Format:
```
- ✅ `ClassName.method_name()` - Brief description of what it does
```

### Example:
```
#### Sprint 4 Functions 🔄 (IN PROGRESS)
- ✅ `PredictionValidator.__init__()` - Initialize validator with accuracy thresholds
- ✅ `PredictionValidator.record_prediction()` - Store prediction for validation
```

**This prevents duplicate function creation and maintains clear implementation tracking!**

## 🔧 Component Integration - MANDATORY

**⚠️ CRITICAL: ALL AGENTS MUST INTEGRATE COMPONENTS INTO MAIN SYSTEM**

### Integration Requirements:
1. **NEVER create standalone components** - Always integrate into main system workflow
2. **NO example-only implementations** - Components must work automatically in production
3. **ALWAYS modify existing system files** to use new components
4. **ENSURE automatic operation** - No manual setup required for core functionality

### Integration Checklist for All Agents:
- [ ] Component is used by main system automatically (not just in examples)
- [ ] Modified existing files to integrate the new component
- [ ] Component works without manual setup or configuration
- [ ] Integration follows existing system patterns and architecture
- [ ] Background tasks or event triggers are properly integrated

### Bad Examples (DON'T DO):
```python
# ❌ BAD: Standalone component with only example usage
# example_component_usage.py
from new_component import NewComponent
component = NewComponent()
component.do_something()  # Manual setup required
```

### Good Examples (DO THIS):
```python
# ✅ GOOD: Integrated into main system
# src/main_system.py
class MainSystem:
    def __init__(self):
        self.new_component = NewComponent()  # Automatic integration
        
    async def process(self):
        # Component used automatically in main workflow
        await self.new_component.do_something()
```

**Remember: If users need to manually run example scripts, the integration is incomplete!**

## 🤖 Use Specialized Agents Proactively

**You have access to specialized agents with deep expertise. Use them whenever their skills match the task - don't try to do everything manually.**

### Available Specialized Agents:
- **test-automator**: Create/fix test suites, CI pipelines, test automation
- **database-optimizer**: Fix SQL queries, optimize schemas, database performance  
- **debugger**: Debug errors, test failures, unexpected behavior
- **code-reviewer**: Review code quality, security, maintainability
- **security-auditor**: Security reviews, vulnerability fixes, auth flows
- **typescript-expert**: TypeScript development, type system design
- **python-pro**: Python refactoring, optimization, advanced features
- **backend-architect**: API design, system architecture, scalability
- **frontend-developer**: Next.js, React, UI components, modern patterns
- **deployment-engineer**: CI/CD, Docker, cloud deployments, infrastructure
- **performance-engineer**: Profiling, optimization, caching strategies

### When to Use Agents:
- **Complex technical problems** that match their expertise
- **Multi-step tasks** requiring specialized knowledge  
- **Code quality issues** that need systematic fixes
- **Architecture decisions** requiring domain expertise
- **ANY TIME** their description matches what you need to do

### Don't Try to Do Everything Yourself:
- Agents have specialized knowledge and tools
- They can solve problems more efficiently and thoroughly
- Use them proactively, not just when explicitly asked

## Project Overview

This is a **Home Assistant ML Predictor** project for room occupancy prediction. Sprint 1 (Foundation & Data Infrastructure) implementation is nearly complete but requires test validation.

**Status**: Sprint 1 IN PROGRESS ⚠️ - Core infrastructure implemented but 4 test failures must be fixed before completion

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

**LXC Container**: ✅ Environment setup in progress
- **Hostname**: ha-ml-predictor
- **IP**: 192.168.51.112 (Ubuntu 24.04, 2 cores, 6GB RAM)
- **SSH Key**: ~/.ssh/ha-ml-predictor (passwordless access configured)
- **Python**: 3.12.3 with venv support installed
- **Project Path**: `/opt/ha-ml-predictor/` with virtual environment
- **Important**: Always activate venv before running: `source /opt/ha-ml-predictor/venv/bin/activate`

**SSH Aliases** (added to ~/.bashrc):
- `hamp-ssh` - SSH into the container
- `hamp-logs` - View application logs
- `hamp-status` - Check database health status
- `hamp-test` - Run unit tests remotely

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