"""
pytest configuration and fixtures for the occupancy prediction system.

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
import pytest
import pytest_asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch
import tempfile
import os
from pathlib import Path

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy import text

from src.core.config import SystemConfig, DatabaseConfig, HomeAssistantConfig, RoomConfig
from src.core.constants import SensorType, SensorState
from src.data.storage.models import Base, SensorEvent, RoomState, Prediction
from src.data.storage.database import DatabaseManager
from src.data.ingestion.ha_client import HAEvent, HomeAssistantClient
from src.data.ingestion.event_processor import EventProcessor


# Test database configuration
# Use actual TimescaleDB container for testing
# Environment variable TEST_DB_URL can override this for CI/CD
TEST_DB_URL = os.getenv(
    "TEST_DB_URL", 
    "postgresql+asyncpg://occupancy_user:occupancy_pass@localhost:5432/occupancy_prediction_test"
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def test_config_dir():
    """Create a temporary directory with test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)
        
        # Create main config file
        main_config = {
            'home_assistant': {
                'url': 'http://test-ha:8123',
                'token': 'test_token_12345',
                'websocket_timeout': 30,
                'api_timeout': 10
            },
            'database': {
                'connection_string': TEST_DB_URL,
                'pool_size': 5,
                'max_overflow': 10
            },
            'mqtt': {
                'broker': 'test-mqtt',
                'port': 1883,
                'username': 'test_user',
                'password': 'test_pass',
                'topic_prefix': 'test/occupancy'
            },
            'prediction': {
                'interval_seconds': 300,
                'accuracy_threshold_minutes': 15,
                'confidence_threshold': 0.7
            },
            'features': {
                'lookback_hours': 24,
                'sequence_length': 50,
                'temporal_features': True,
                'sequential_features': True,
                'contextual_features': True
            },
            'logging': {
                'level': 'DEBUG',
                'format': 'structured'
            }
        }
        
        # Create rooms config file
        rooms_config = {
            'rooms': {
                'test_room': {
                    'name': 'Test Room',
                    'sensors': {
                        'presence': {
                            'main': 'binary_sensor.test_room_presence',
                            'secondary': 'binary_sensor.test_room_motion'
                        },
                        'door': 'binary_sensor.test_room_door',
                        'climate': {
                            'temperature': 'sensor.test_room_temperature',
                            'humidity': 'sensor.test_room_humidity'
                        },
                        'light': 'sensor.test_room_light'
                    }
                },
                'living_room': {
                    'name': 'Living Room',
                    'sensors': {
                        'presence': {
                            'main': 'binary_sensor.living_room_presence',
                            'couch': 'binary_sensor.living_room_couch'
                        },
                        'climate': {
                            'temperature': 'sensor.living_room_temperature'
                        }
                    }
                }
            }
        }
        
        # Write config files
        import yaml
        with open(config_dir / 'config.yaml', 'w') as f:
            yaml.dump(main_config, f)
        
        with open(config_dir / 'rooms.yaml', 'w') as f:
            yaml.dump(rooms_config, f)
        
        yield str(config_dir)


@pytest.fixture
def test_system_config(test_config_dir):
    """Create a SystemConfig instance for testing."""
    from src.core.config import ConfigLoader
    loader = ConfigLoader(test_config_dir)
    return loader.load_config()


@pytest.fixture
def test_room_config():
    """Create a test room configuration."""
    return RoomConfig(
        room_id='test_room',
        name='Test Room',
        sensors={
            'presence': {
                'main': 'binary_sensor.test_room_presence',
                'secondary': 'binary_sensor.test_room_motion'
            },
            'door': 'binary_sensor.test_room_door',
            'climate': {
                'temperature': 'sensor.test_room_temperature',
                'humidity': 'sensor.test_room_humidity'
            },
            'light': 'sensor.test_room_light'
        }
    )


@pytest.fixture
async def test_db_engine():
    """Create a test database engine with TimescaleDB."""
    # Use the real TimescaleDB container with a test database
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
        future=True,
        pool_size=1,
        max_overflow=0
    )
    
    try:
        # First, check if the test database exists, create it if not
        main_db_url = TEST_DB_URL.replace('/occupancy_prediction_test', '/occupancy_prediction')
        main_engine = create_async_engine(main_db_url, isolation_level="AUTOCOMMIT")
        
        try:
            async with main_engine.connect() as conn:
                # Check if test database exists
                result = await conn.execute(text(
                    "SELECT 1 FROM pg_database WHERE datname = 'occupancy_prediction_test'"
                ))
                if not result.scalar():
                    # Create test database
                    await conn.execute(text("CREATE DATABASE occupancy_prediction_test"))
        except Exception:
            # Database might already exist or we don't have permissions
            pass
        finally:
            await main_engine.dispose()
        
        # Now create tables in test database
        async with engine.begin() as conn:
            # Import TimescaleDB extension if available
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE"))
            except Exception:
                pass  # TimescaleDB might not be available in test environment
            
            # Drop and recreate all tables for clean test state
            await conn.run_sync(Base.metadata.drop_all)
            await conn.run_sync(Base.metadata.create_all)
            
            # Create hypertables if TimescaleDB is available
            try:
                await conn.execute(text(
                    "SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE)"
                ))
                await conn.execute(text(
                    "SELECT create_hypertable('room_states', 'timestamp', if_not_exists => TRUE)"
                ))
                await conn.execute(text(
                    "SELECT create_hypertable('predictions', 'prediction_time', if_not_exists => TRUE)"
                ))
            except Exception:
                # Not a problem if TimescaleDB functions aren't available
                pass
        
        yield engine
        
    finally:
        # Clean up tables and dispose engine
        try:
            async with engine.begin() as conn:
                await conn.run_sync(Base.metadata.drop_all)
        except Exception:
            pass  # Ignore cleanup errors
        await engine.dispose()


@pytest.fixture
async def test_db_session(test_db_engine):
    """Create a test database session."""
    async_session = async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )
    
    async with async_session() as session:
        # Start a transaction that we can rollback
        # No need to manually begin transaction in SQLAlchemy 2.0
        try:
            yield session
        finally:
            # Always rollback to ensure clean state
            await session.rollback()


@pytest.fixture
async def test_db_manager(test_system_config):
    """Create a test database manager."""
    # Override config to use test database
    test_db_config = DatabaseConfig(
        connection_string=TEST_DB_URL,
        pool_size=1,
        max_overflow=0
    )
    
    manager = DatabaseManager(test_db_config)
    await manager.initialize()
    
    yield manager
    
    await manager.close()


@pytest.fixture
def sample_sensor_events():
    """Create sample sensor events for testing."""
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    events = []
    for i in range(10):
        event = SensorEvent(
            room_id='test_room',
            sensor_id=f'binary_sensor.test_sensor_{i % 3}',
            sensor_type=SensorType.PRESENCE.value,
            state=SensorState.ON.value if i % 2 == 0 else SensorState.OFF.value,
            previous_state=SensorState.OFF.value if i % 2 == 0 else SensorState.ON.value,
            timestamp=base_time + timedelta(minutes=i * 5),
            attributes={'test': True, 'sequence': i},
            is_human_triggered=True,
            confidence_score=0.8 + (i * 0.02),
            created_at=datetime.utcnow()
        )
        events.append(event)
    
    return events


@pytest.fixture
def sample_ha_events():
    """Create sample Home Assistant events for testing."""
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    events = []
    for i in range(5):
        event = HAEvent(
            entity_id=f'binary_sensor.test_sensor_{i}',
            state=SensorState.ON.value if i % 2 == 0 else SensorState.OFF.value,
            previous_state=SensorState.OFF.value if i % 2 == 0 else SensorState.ON.value,
            timestamp=base_time + timedelta(minutes=i * 10),
            attributes={'device_class': 'motion', 'friendly_name': f'Test Sensor {i}'}
        )
        events.append(event)
    
    return events


@pytest.fixture
def mock_ha_client():
    """Create a mock Home Assistant client."""
    client = AsyncMock(spec=HomeAssistantClient)
    client.is_connected = True
    client.connect = AsyncMock()
    client.disconnect = AsyncMock()
    client.get_entity_state = AsyncMock()
    client.get_entity_history = AsyncMock()
    client.validate_entities = AsyncMock()
    client.subscribe_to_events = AsyncMock()
    return client


@pytest.fixture
def mock_event_processor():
    """Create a mock event processor."""
    processor = AsyncMock(spec=EventProcessor)
    processor.process_event = AsyncMock()
    processor.process_event_batch = AsyncMock()
    processor.get_processing_stats = MagicMock(return_value={
        'total_processed': 100,
        'valid_events': 95,
        'invalid_events': 5,
        'human_classified': 80,
        'cat_classified': 15,
        'duplicates_filtered': 2
    })
    return processor


@pytest.fixture
async def populated_test_db(test_db_session, sample_sensor_events):
    """Create a test database with sample data."""
    # Add sample events
    for event in sample_sensor_events:
        test_db_session.add(event)
    
    # Add sample room states
    for i in range(3):
        room_state = RoomState(
            room_id='test_room',
            timestamp=datetime.utcnow() - timedelta(hours=i),
            is_occupied=i % 2 == 0,
            occupancy_confidence=0.8,
            occupant_type='human',
            state_duration=300 + i * 60,
            transition_trigger=f'binary_sensor.test_sensor_{i}',
            created_at=datetime.utcnow()
        )
        test_db_session.add(room_state)
    
    # Add sample predictions
    for i in range(3):
        prediction = Prediction(
            room_id='test_room',
            prediction_time=datetime.utcnow() - timedelta(hours=i),
            predicted_transition_time=datetime.utcnow() + timedelta(minutes=15 + i * 5),
            transition_type='occupied_to_vacant',
            confidence_score=0.75 + i * 0.05,
            model_type='lstm',
            model_version='v1.0',
            created_at=datetime.utcnow()
        )
        test_db_session.add(prediction)
    
    await test_db_session.commit()
    yield test_db_session


@pytest.fixture
def mock_websocket():
    """Create a mock WebSocket connection."""
    ws = AsyncMock()
    ws.recv = AsyncMock()
    ws.send = AsyncMock()
    ws.close = AsyncMock()
    ws.closed = False
    return ws


@pytest.fixture  
def mock_aiohttp_session():
    """Create a mock aiohttp session."""
    session = AsyncMock()
    response = AsyncMock()
    response.status = 200
    response.json = AsyncMock()
    response.text = AsyncMock(return_value="OK")
    session.get = AsyncMock(return_value=response)
    session.post = AsyncMock(return_value=response)
    session.close = AsyncMock()
    return session, response


@pytest.fixture
def test_environment_variables():
    """Set up test environment variables."""
    original_env = dict(os.environ)
    
    # Set test environment variables
    test_vars = {
        'HA_URL': 'http://test-ha:8123',
        'HA_TOKEN': 'test_token_12345',
        'DB_URL': TEST_DB_URL,
        'TESTING': '1'
    }
    
    os.environ.update(test_vars)
    
    yield test_vars
    
    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "unit: mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "database: mark test as requiring database"
    )
    config.addinivalue_line(
        "markers", "ha_client: mark test as requiring Home Assistant client"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )


# Helper functions for tests
def assert_sensor_event_equal(event1: SensorEvent, event2: SensorEvent, ignore_timestamps=True):
    """Assert that two sensor events are equal, optionally ignoring timestamps."""
    assert event1.room_id == event2.room_id
    assert event1.sensor_id == event2.sensor_id
    assert event1.sensor_type == event2.sensor_type
    assert event1.state == event2.state
    assert event1.previous_state == event2.previous_state
    assert event1.is_human_triggered == event2.is_human_triggered
    
    if not ignore_timestamps:
        assert event1.timestamp == event2.timestamp
        assert event1.created_at == event2.created_at


def create_test_ha_event(
    entity_id: str = "binary_sensor.test_sensor",
    state: str = "on",
    previous_state: Optional[str] = "off",
    timestamp: Optional[datetime] = None,
    attributes: Optional[Dict[str, Any]] = None
) -> HAEvent:
    """Create a test Home Assistant event."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    if attributes is None:
        attributes = {'device_class': 'motion'}
    
    return HAEvent(
        entity_id=entity_id,
        state=state,
        previous_state=previous_state,
        timestamp=timestamp,
        attributes=attributes
    )


def create_test_sensor_event(
    room_id: str = "test_room",
    sensor_id: str = "binary_sensor.test_sensor",
    sensor_type: str = "presence",
    state: str = "on",
    timestamp: Optional[datetime] = None
) -> SensorEvent:
    """Create a test sensor event."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    
    return SensorEvent(
        room_id=room_id,
        sensor_id=sensor_id,
        sensor_type=sensor_type,
        state=state,
        previous_state="off",
        timestamp=timestamp,
        attributes={'test': True},
        is_human_triggered=True,
        confidence_score=0.8,
        created_at=datetime.utcnow()
    )