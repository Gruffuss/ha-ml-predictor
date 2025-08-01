# Test Suite for Occupancy Prediction System

This directory contains comprehensive unit and integration tests for all Sprint 1 components of the occupancy prediction system.

## Test Structure

```
tests/
├── conftest.py              # Pytest configuration and shared fixtures
├── fixtures/                # Test configuration files and data
│   ├── test_config.yaml    # Test system configuration
│   └── test_rooms.yaml     # Test room configurations
├── unit/                   # Unit tests (fast, isolated)
│   ├── test_core/         # Core system tests
│   │   ├── test_config.py        # Configuration management tests
│   │   ├── test_constants.py     # Constants and enums tests
│   │   └── test_exceptions.py    # Exception handling tests
│   ├── test_data/         # Data layer tests
│   │   ├── test_models.py        # Database models tests
│   │   └── test_database.py      # Database manager tests
│   └── test_ingestion/    # Data ingestion tests
│       ├── test_ha_client.py     # Home Assistant client tests
│       ├── test_event_processor.py    # Event processing tests
│       └── test_bulk_importer.py      # Bulk import tests
├── integration/           # Integration tests (slower, with services)
│   ├── test_database_integration.py  # Full database operations
│   ├── test_ha_integration.py        # HA API integration
│   └── test_end_to_end.py            # Complete data flow
├── run_tests.py          # Test runner script
└── README.md            # This file
```

## Test Categories

### Unit Tests (`pytest -m unit`)
- **Fast execution** (< 1 second per test)
- **Isolated** - no external dependencies
- **Mocked** - external services are mocked
- **Comprehensive** - test individual functions/classes

### Integration Tests (`pytest -m integration`)
- **Slower execution** (1-30 seconds per test)
- **Real dependencies** - actual database, services
- **End-to-end workflows** - test component interactions
- **System validation** - verify complete functionality

### Specialized Markers
- `database` - Tests requiring database access
- `ha_client` - Tests requiring Home Assistant client
- `slow` - Long-running tests (> 30 seconds)
- `smoke` - Quick validation tests

## Running Tests

### Quick Start
```bash
# Run all unit tests (fast)
python tests/run_tests.py unit

# Run all tests with coverage
python tests/run_tests.py all --coverage

# Run specific component tests
pytest tests/unit/test_core/ -v
```

### Using the Test Runner
The `run_tests.py` script provides easy test execution:

```bash
# Run different test categories
python tests/run_tests.py unit              # Unit tests only
python tests/run_tests.py integration       # Integration tests only
python tests/run_tests.py database         # Database-related tests
python tests/run_tests.py ha_client        # HA client tests
python tests/run_tests.py all              # All tests

# With additional options
python tests/run_tests.py unit --coverage   # With coverage report
python tests/run_tests.py all --parallel 4  # Parallel execution
python tests/run_tests.py unit -v           # Verbose output
python tests/run_tests.py all --html-report # Generate HTML report
```

### Direct Pytest Usage
```bash
# Run specific test files
pytest tests/unit/test_core/test_config.py -v

# Run tests matching pattern
pytest -k "test_config" -v

# Run with coverage
pytest --cov=src --cov-report=html tests/unit/

# Run specific markers
pytest -m "unit and not slow" -v

# Stop on first failure
pytest -x tests/unit/
```

## Test Configuration

### Environment Setup
Tests use isolated configuration and in-memory databases:
- **Database**: SQLite in-memory for fast, isolated tests
- **Home Assistant**: Mocked API responses
- **Configuration**: Test-specific config files in `fixtures/`

### Fixtures Available
- `test_system_config` - Complete system configuration for testing
- `test_room_config` - Individual room configuration
- `test_db_session` - Database session with test data
- `test_db_manager` - Database manager instance
- `sample_sensor_events` - Pre-created sensor events
- `sample_ha_events` - Pre-created HA events
- `mock_ha_client` - Mocked Home Assistant client
- `populated_test_db` - Database with sample data

## Test Requirements

### Sprint 1 Testing Requirements ✅
- [x] **Core System Tests** - Configuration, constants, exceptions
- [x] **Database Tests** - Models, relationships, queries, connection management
- [x] **HA Integration Tests** - WebSocket/REST client, authentication, rate limiting
- [x] **Event Processing Tests** - Validation, human/cat detection, deduplication
- [x] **Bulk Import Tests** - Historical data import, batching, error handling
- [x] **Integration Tests** - End-to-end workflows, database operations
- [x] **Error Handling** - All error conditions and edge cases
- [x] **Configuration Testing** - All config loading and validation scenarios

### Coverage Goals
- **Unit Tests**: > 90% line coverage
- **Integration Tests**: Complete workflow coverage
- **Error Paths**: All exception scenarios tested
- **Edge Cases**: Boundary conditions and invalid inputs

## Test Data

### Realistic Test Scenarios
Tests use realistic data based on the actual room configuration:
- **Rooms**: living_kitchen, bedroom, office, bathroom, etc.
- **Sensors**: Actual entity IDs from `config/rooms.yaml`
- **Events**: Realistic sensor state changes and timings
- **Predictions**: Plausible occupancy predictions with accuracy tracking

### Mock Data Generation
Helper functions create consistent test data:
- `create_test_ha_event()` - Generate HA events
- `create_test_sensor_event()` - Generate sensor events
- `assert_sensor_event_equal()` - Compare sensor events

## Continuous Integration

### Pre-commit Testing
Before committing code, run:
```bash
# Quick validation
python tests/run_tests.py unit

# Full validation (for Sprint completion)
python tests/run_tests.py all --coverage
```

### Sprint Validation
To validate Sprint 1 completion:
```bash
# Run all Sprint 1 tests
python tests/run_tests.py all --coverage --html-report

# Verify coverage meets requirements
# Check that all core functionality is tested
# Ensure integration tests pass
```

## Test Development Guidelines

### Writing Unit Tests
- **Fast**: Each test < 1 second
- **Isolated**: No external dependencies
- **Deterministic**: Same result every time
- **Focused**: One behavior per test
- **Clear names**: Describe what is being tested

### Writing Integration Tests
- **Realistic**: Use actual services when possible
- **Complete workflows**: Test end-to-end scenarios  
- **Error recovery**: Test failure and retry scenarios
- **Performance**: Validate reasonable execution times

### Test Organization
- **Group related tests** in classes
- **Use descriptive test names** that explain the scenario
- **Include docstrings** for complex test scenarios
- **Use appropriate markers** for test categorization

## Troubleshooting

### Common Issues
1. **Database connection errors**: Ensure test database is properly configured
2. **Async test failures**: Use `@pytest.mark.asyncio` decorator
3. **Import errors**: Ensure `src/` is in Python path
4. **Fixture errors**: Check fixture dependencies and scopes

### Debugging Tests
```bash
# Run single test with debugging
pytest tests/unit/test_core/test_config.py::TestConfigLoader::test_load_config_success -v -s

# Show test output
pytest -s tests/unit/test_core/test_config.py

# Drop into debugger on failure
pytest --pdb tests/unit/test_core/test_config.py
```

### Performance Monitoring
Monitor test execution times:
```bash
# Show slowest tests
pytest --durations=10 tests/

# Profile test execution
pytest --profile tests/unit/
```

This comprehensive test suite ensures that all Sprint 1 components are thoroughly validated before proceeding to Sprint 2 development.