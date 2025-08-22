# Integration Test Suite

This directory contains comprehensive integration tests for the HA ML Predictor system, designed to achieve 85% coverage across all integration modules with realistic testing scenarios.

## Test Structure

### Test Categories

- **MQTT Integration Tests** (`test_mqtt_integration_comprehensive.py`)
  - Real MQTT broker testing with embedded mosquitto
  - Connection failure and recovery scenarios
  - High-throughput message publishing
  - Network partition and reconnection handling
  - Message queuing and delivery guarantees

- **API Server Load Testing** (`test_api_server_load_testing.py`)
  - High concurrent request handling
  - Authentication system stress testing
  - Rate limiting effectiveness
  - Database connection pooling under stress
  - WebSocket connection handling

- **HA Entity Integration** (`test_ha_entities_integration.py`)
  - Entity discovery message publishing
  - State publishing and subscription verification
  - Entity lifecycle management
  - Service integration and command handling
  - Performance under high entity counts

- **Cross-Component Integration** (`test_cross_component_integration.py`)
  - End-to-end data flow testing
  - MQTT + API + HA entities working together
  - Service command handling across components
  - Real-time state updates
  - Data consistency verification

- **Performance & Reliability** (`test_performance_reliability.py`)
  - High-throughput scenario testing
  - Resource usage under load
  - Memory and CPU usage patterns
  - Long-running system stability
  - Error recovery and graceful degradation

- **Container Framework** (`test_container_framework.py`)
  - Real infrastructure testing with Docker containers
  - Multi-service integration testing
  - Network condition simulation
  - End-to-end workflow validation

## Requirements

### Basic Requirements
- Python 3.11+
- pytest
- pytest-asyncio
- All project dependencies from `requirements.txt`

### Container Testing Requirements
- Docker installed and running
- Docker Python SDK: `pip install docker`
- Network access for pulling container images

### Optional Performance Testing Requirements
- psutil: `pip install psutil`
- Additional system monitoring tools

## Running the Tests

### Quick Start
```bash
# Run all integration tests
pytest tests/integration/

# Run with verbose output
pytest tests/integration/ -v

# Run specific test categories
pytest tests/integration/ -m "integration"
pytest tests/integration/ -m "performance"
pytest tests/integration/ -m "container"
```

### Test Categories

#### Basic Integration Tests
```bash
# MQTT integration tests (no containers required)
pytest tests/integration/test_mqtt_integration_comprehensive.py -v

# API server load testing
pytest tests/integration/test_api_server_load_testing.py -v

# HA entity integration tests
pytest tests/integration/test_ha_entities_integration.py -v

# Cross-component integration tests
pytest tests/integration/test_cross_component_integration.py -v
```

#### Performance Tests
```bash
# Performance and reliability tests
pytest tests/integration/test_performance_reliability.py -v -m "performance"

# Skip slow tests
pytest tests/integration/ -v -m "not slow"

# Skip performance tests in CI
SKIP_PERFORMANCE_TESTS=1 pytest tests/integration/ -v
```

#### Container Tests (Requires Docker)
```bash
# Container framework tests
pytest tests/integration/test_container_framework.py -v -m "container"

# Skip container tests if Docker not available
pytest tests/integration/ -v -m "not container"
```

#### Selective Test Execution
```bash
# Run only fast tests
pytest tests/integration/ -v -m "not slow and not container"

# Run only MQTT tests
pytest tests/integration/ -v -k "mqtt"

# Run only API tests
pytest tests/integration/ -v -k "api"

# Run only entity tests
pytest tests/integration/ -v -k "entity"
```

## Test Configuration

### Environment Variables

Set these environment variables to customize test behavior:

```bash
# Test environment setup
export ENVIRONMENT=test
export JWT_SECRET_KEY=test_secret_key_for_integration_testing
export API_KEY=test_api_key_for_integration_testing

# Optional: Skip certain test types
export SKIP_PERFORMANCE_TESTS=1
export SKIP_SLOW_TESTS=1
export SKIP_CONTAINER_TESTS=1
```

### Docker Configuration

For container tests, ensure Docker is running:

```bash
# Check Docker status
docker info

# Pull required images (optional, done automatically)
docker pull eclipse-mosquitto:2.0
docker pull timescale/timescaledb:latest-pg14
docker pull redis:7-alpine
```

## Test Markers

Tests are categorized using pytest markers:

- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.performance` - Performance tests  
- `@pytest.mark.container` - Tests requiring Docker containers
- `@pytest.mark.slow` - Long-running tests (>30 seconds)
- `@pytest.mark.network` - Tests requiring network access

### Running Tests by Marker
```bash
# Integration tests only
pytest tests/integration/ -m "integration"

# Performance tests only  
pytest tests/integration/ -m "performance"

# Container tests only
pytest tests/integration/ -m "container"

# Exclude slow tests
pytest tests/integration/ -m "not slow"

# Multiple markers
pytest tests/integration/ -m "integration and not slow"
```

## Coverage Analysis

### Running with Coverage
```bash
# Install coverage tools
pip install coverage pytest-cov

# Run with coverage reporting
pytest tests/integration/ --cov=src/integration --cov-report=html --cov-report=term-missing

# View coverage report
open htmlcov/index.html  # Opens coverage report in browser
```

### Coverage Targets

The integration test suite targets 85% coverage for:

- `src/integration/mqtt_publisher.py`
- `src/integration/api_server.py` 
- `src/integration/ha_entity_definitions.py`
- `src/integration/discovery_publisher.py`

### Coverage Analysis Commands
```bash
# Generate detailed coverage report
coverage run -m pytest tests/integration/
coverage report --show-missing
coverage html

# Check specific module coverage
coverage report --include="src/integration/*"
```

## Test Data and Fixtures

### Shared Fixtures

The `conftest.py` file provides shared fixtures:

- `integration_test_config` - Test configuration
- `test_data_generator` - Realistic test data generation
- `performance_monitor` - Performance monitoring utilities
- `network_condition_simulator` - Network condition simulation
- `cleanup_tasks` - Test cleanup management

### Test Data Generation

Tests use realistic data generators:

```python
# Generate room configurations
room_config = test_data_generator.generate_room_config("living_room", num_sensors=5)

# Generate prediction data
prediction = test_data_generator.generate_prediction_data("bedroom")

# Generate sensor events
events = test_data_generator.generate_sensor_events("kitchen", count=100)

# Generate MQTT messages
messages = test_data_generator.generate_mqtt_messages("test_topic", count=50)
```

## Performance Monitoring

### Performance Test Configuration

Performance tests use configurable thresholds:

```python
PERFORMANCE_CONFIG = {
    "high_load_message_count": 10000,
    "concurrent_clients": 50,
    "sustained_duration_minutes": 5,
    "memory_limit_mb": 500,
    "cpu_limit_percent": 80,
    "latency_threshold_ms": 1000,
    "throughput_threshold_msg_per_sec": 100
}
```

### Monitoring Performance

```bash
# Run performance tests with monitoring
pytest tests/integration/test_performance_reliability.py -v -s

# Monitor resource usage during tests
top -p $(pgrep -f pytest)
```

## Troubleshooting

### Common Issues

#### Docker Not Available
```
Error: Docker not available for container tests
Solution: Install Docker and ensure it's running, or skip container tests:
pytest tests/integration/ -m "not container"
```

#### Port Conflicts
```
Error: Port already in use
Solution: Tests automatically find available ports, but ensure no conflicting services
```

#### Memory Issues
```
Error: Test fails due to memory constraints
Solution: Close other applications or reduce test load:
SKIP_PERFORMANCE_TESTS=1 pytest tests/integration/
```

#### Network Issues
```
Error: Network timeouts during container tests
Solution: Check internet connection and Docker network configuration
```

### Debug Mode

Run tests in debug mode for detailed output:

```bash
# Enable debug logging
pytest tests/integration/ -v -s --log-cli-level=DEBUG

# Run single test with full output
pytest tests/integration/test_mqtt_integration_comprehensive.py::TestMQTTRealBrokerIntegration::test_real_broker_connection -v -s
```

### Test Isolation

Each test is isolated and includes cleanup:

```bash
# Run tests in parallel (if pytest-xdist installed)
pip install pytest-xdist
pytest tests/integration/ -n auto

# Force garbage collection between tests
pytest tests/integration/ --forked
```

## CI/CD Integration

### GitHub Actions Configuration

```yaml
name: Integration Tests

on: [push, pull_request]

jobs:
  integration-tests:
    runs-on: ubuntu-latest
    
    services:
      mosquitto:
        image: eclipse-mosquitto:2.0
        ports:
          - 1883:1883
      
      postgres:
        image: timescale/timescaledb:latest-pg14
        env:
          POSTGRES_DB: test_ha_ml
          POSTGRES_USER: test_user
          POSTGRES_PASSWORD: test_password
        ports:
          - 5432:5432

    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest-cov
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ --cov=src/integration --cov-report=xml
      env:
        SKIP_SLOW_TESTS: 1
    
    - name: Upload coverage
      uses: codecov/codecov-action@v3
      with:
        file: coverage.xml
```

### Local CI Simulation

```bash
# Simulate CI environment
export SKIP_SLOW_TESTS=1
export SKIP_PERFORMANCE_TESTS=1
export CI=true

# Run tests as in CI
pytest tests/integration/ --cov=src/integration --cov-report=xml --tb=short
```

## Contributing

### Adding New Integration Tests

1. Follow the existing test structure
2. Use appropriate markers (`@pytest.mark.integration`, etc.)
3. Include realistic test scenarios
4. Add proper cleanup in fixtures
5. Document test purpose and expected outcomes

### Test Naming Convention

- `test_<component>_<scenario>` - Basic test format
- `test_<component>_<condition>_<expected_behavior>` - Detailed format
- Use descriptive names that explain what is being tested

### Example Test Structure

```python
@pytest.mark.integration
@pytest.mark.asyncio
async def test_mqtt_publisher_high_throughput_resilience(mock_mqtt_broker):
    """Test MQTT publisher resilience under high-throughput conditions."""
    # Arrange
    publisher = MQTTPublisher(test_config)
    
    # Act
    results = await run_high_throughput_test(publisher)
    
    # Assert
    assert_performance_requirements_met(results)
```

## Support

For issues with integration tests:

1. Check the troubleshooting section above
2. Verify all requirements are installed
3. Check Docker is running (for container tests)
4. Run tests with debug output: `-v -s --log-cli-level=DEBUG`
5. Review test logs in `tests/integration/logs/`

## Test Coverage Goals

The integration test suite aims for:

- **85% overall coverage** of integration modules
- **90% coverage** of critical paths (MQTT publishing, API endpoints)
- **100% coverage** of error handling scenarios
- **Comprehensive testing** of real-world failure scenarios

Current coverage can be viewed with:
```bash
pytest tests/integration/ --cov=src/integration --cov-report=term-missing
```