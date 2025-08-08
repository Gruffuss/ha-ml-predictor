"""
pytest configuration and fixtures for the occupancy prediction system.

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
import os
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.ext.asyncio import async_sessionmaker
from sqlalchemy.ext.asyncio import create_async_engine

from src.core.config import DatabaseConfig
from src.core.config import HomeAssistantConfig
from src.core.config import RoomConfig
from src.core.config import SystemConfig
from src.core.constants import SensorState
from src.core.constants import SensorType
from src.data.ingestion.event_processor import EventProcessor
from src.data.ingestion.ha_client import HAEvent
from src.data.ingestion.ha_client import HomeAssistantClient
from src.data.storage.database import DatabaseManager
from src.data.storage.models import Base
from src.data.storage.models import Prediction
from src.data.storage.models import RoomState
from src.data.storage.models import SensorEvent


def _patch_models_for_sqlite(engine):
    """Patch models to be SQLite-compatible by removing composite primary keys."""
    from sqlalchemy import BigInteger
    from sqlalchemy import Column
    from sqlalchemy import DateTime
    from sqlalchemy.sql import func

    from src.data.storage.models import Prediction
    from src.data.storage.models import RoomState
    from src.data.storage.models import SensorEvent

    # Remove timestamp from primary key for SensorEvent
    # This allows SQLite to work with autoincrement on id only
    if hasattr(SensorEvent, "__table__"):
        # Remove the composite primary key constraint
        table = SensorEvent.__table__
        # Find the timestamp column and remove it from primary key
        for col in table.columns:
            if col.name == "timestamp" and col.primary_key:
                col.primary_key = False

    # Similar patching for other models if needed
    # RoomState and Prediction should be fine as they use simple primary keys


# Test database configuration
# Use SQLite in-memory database for testing by default, but mock the tests
# Environment variable TEST_DB_URL can override this for CI/CD with real PostgreSQL
TEST_DB_URL = os.getenv("TEST_DB_URL", "sqlite+aiosqlite:///:memory:")

# Flag to determine if we should use mock databases for tests
USE_MOCK_DB = True


# Remove custom event_loop fixture - use pytest-asyncio default
# @pytest.fixture(scope="session")
# def event_loop():
#     """Create an instance of the default event loop for the test session."""
#     loop = asyncio.get_event_loop_policy().new_event_loop()
#     yield loop
#     loop.close()


@pytest.fixture
def test_config_dir():
    """Create a temporary directory with test configuration files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_dir = Path(temp_dir)

        # Create main config file
        main_config = {
            "home_assistant": {
                "url": "http://test-ha:8123",
                "token": "test_token_12345",
                "websocket_timeout": 30,
                "api_timeout": 10,
            },
            "database": {
                "connection_string": TEST_DB_URL,
                "pool_size": 5,
                "max_overflow": 10,
            },
            "mqtt": {
                "broker": "test-mqtt",
                "port": 1883,
                "username": "test_user",
                "password": "test_pass",
                "topic_prefix": "test/occupancy",
            },
            "prediction": {
                "interval_seconds": 300,
                "accuracy_threshold_minutes": 15,
                "confidence_threshold": 0.7,
            },
            "features": {
                "lookback_hours": 24,
                "sequence_length": 50,
                "temporal_features": True,
                "sequential_features": True,
                "contextual_features": True,
            },
            "logging": {"level": "DEBUG", "format": "structured"},
        }

        # Create rooms config file
        rooms_config = {
            "rooms": {
                "test_room": {
                    "name": "Test Room",
                    "sensors": {
                        "presence": {
                            "main": "binary_sensor.test_room_presence",
                            "secondary": "binary_sensor.test_room_motion",
                        },
                        "door": "binary_sensor.test_room_door",
                        "climate": {
                            "temperature": "sensor.test_room_temperature",
                            "humidity": "sensor.test_room_humidity",
                        },
                        "light": "sensor.test_room_light",
                    },
                },
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "presence": {
                            "main": "binary_sensor.living_room_presence",
                            "couch": "binary_sensor.living_room_couch",
                        },
                        "climate": {"temperature": "sensor.living_room_temperature"},
                    },
                },
            }
        }

        # Write config files
        import yaml

        with open(config_dir / "config.yaml", "w") as f:
            yaml.dump(main_config, f)

        with open(config_dir / "rooms.yaml", "w") as f:
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
        room_id="test_room",
        name="Test Room",
        sensors={
            "presence": {
                "main": "binary_sensor.test_room_presence",
                "secondary": "binary_sensor.test_room_motion",
            },
            "door": "binary_sensor.test_room_door",
            "climate": {
                "temperature": "sensor.test_room_temperature",
                "humidity": "sensor.test_room_humidity",
            },
            "light": "sensor.test_room_light",
        },
    )


@pytest_asyncio.fixture
async def test_db_engine():
    """Create a test database engine with TimescaleDB or SQLite."""
    # SQLite doesn't support pool_size and max_overflow parameters
    if "sqlite" in TEST_DB_URL:
        engine = create_async_engine(TEST_DB_URL, echo=False, future=True)
    else:
        # PostgreSQL/TimescaleDB
        engine = create_async_engine(
            TEST_DB_URL, echo=False, future=True, pool_size=1, max_overflow=0
        )

    try:
        # Create tables in test database (SQLite or PostgreSQL)
        async with engine.begin() as conn:
            # Drop and recreate all tables for clean test state
            await conn.run_sync(Base.metadata.drop_all)

            # For SQLite, temporarily patch models to avoid composite PK issues
            if "sqlite" in TEST_DB_URL:
                # Temporarily patch the models for SQLite compatibility
                await conn.run_sync(_patch_models_for_sqlite)
                await conn.run_sync(Base.metadata.create_all)
            else:
                await conn.run_sync(Base.metadata.create_all)

            # For PostgreSQL, try to create hypertables if TimescaleDB is available
            if "postgresql" in TEST_DB_URL:
                try:
                    await conn.execute(
                        text("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE")
                    )
                    await conn.execute(
                        text(
                            "SELECT create_hypertable('sensor_events', 'timestamp', if_not_exists => TRUE)"
                        )
                    )
                    await conn.execute(
                        text(
                            "SELECT create_hypertable('room_states', 'timestamp', if_not_exists => TRUE)"
                        )
                    )
                    await conn.execute(
                        text(
                            "SELECT create_hypertable('predictions', 'prediction_time', if_not_exists => TRUE)"
                        )
                    )
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


@pytest_asyncio.fixture
async def test_db_session(test_db_engine):
    """Create a test database session."""
    async_session = async_sessionmaker(
        bind=test_db_engine, class_=AsyncSession, expire_on_commit=False
    )

    session = async_session()
    try:
        yield session
    finally:
        # Always rollback and close to ensure clean state
        try:
            await session.rollback()
        except Exception:
            pass  # Ignore rollback errors during cleanup
        await session.close()


@pytest_asyncio.fixture
async def test_db_manager(test_db_engine):
    """Create a test database manager."""
    # Override config to use test database
    test_db_config = DatabaseConfig(
        connection_string=TEST_DB_URL, pool_size=1, max_overflow=0
    )

    manager = DatabaseManager(test_db_config)
    # Use test engine directly to avoid reinitializing
    manager.engine = test_db_engine
    manager.session_factory = async_sessionmaker(
        bind=test_db_engine,
        class_=AsyncSession,
        expire_on_commit=False,
        autoflush=True,
        autocommit=False,
    )

    yield manager

    # Clean up without disposing engine (fixture handles that)
    if manager._health_check_task and not manager._health_check_task.done():
        manager._health_check_task.cancel()
        try:
            await manager._health_check_task
        except asyncio.CancelledError:
            pass
    manager.session_factory = None
    manager.engine = None


@pytest.fixture
def sample_sensor_events():
    """Create sample sensor events for testing."""
    base_time = datetime.utcnow() - timedelta(hours=1)

    events = []
    for i in range(10):
        event = SensorEvent(
            room_id="test_room",
            sensor_id=f"binary_sensor.test_sensor_{i % 3}",
            sensor_type=SensorType.PRESENCE.value,
            state=SensorState.ON.value if i % 2 == 0 else SensorState.OFF.value,
            previous_state=(
                SensorState.OFF.value if i % 2 == 0 else SensorState.ON.value
            ),
            timestamp=base_time + timedelta(minutes=i * 5),
            attributes={"test": True, "sequence": i},
            is_human_triggered=True,
            confidence_score=0.8 + (i * 0.02),
            created_at=datetime.utcnow(),
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
            entity_id=f"binary_sensor.test_sensor_{i}",
            state=SensorState.ON.value if i % 2 == 0 else SensorState.OFF.value,
            previous_state=(
                SensorState.OFF.value if i % 2 == 0 else SensorState.ON.value
            ),
            timestamp=base_time + timedelta(minutes=i * 10),
            attributes={"device_class": "motion", "friendly_name": f"Test Sensor {i}"},
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
    processor.get_processing_stats = MagicMock(
        return_value={
            "total_processed": 100,
            "valid_events": 95,
            "invalid_events": 5,
            "human_classified": 80,
            "cat_classified": 15,
            "duplicates_filtered": 2,
        }
    )
    return processor


@pytest_asyncio.fixture
async def populated_test_db(test_db_session, sample_sensor_events):
    """Create a test database with sample data."""
    session = test_db_session

    # Add sample events
    for event in sample_sensor_events:
        session.add(event)

    # Add sample room states
    for i in range(3):
        room_state = RoomState(
            room_id="test_room",
            timestamp=datetime.utcnow() - timedelta(hours=i),
            is_occupied=i % 2 == 0,
            occupancy_confidence=0.8,
            occupant_type="human",
            state_duration=300 + i * 60,
            transition_trigger=f"binary_sensor.test_sensor_{i}",
            created_at=datetime.utcnow(),
        )
        session.add(room_state)

    # Add sample predictions
    for i in range(3):
        prediction = Prediction(
            room_id="test_room",
            prediction_time=datetime.utcnow() - timedelta(hours=i),
            predicted_transition_time=datetime.utcnow() + timedelta(minutes=15 + i * 5),
            transition_type="occupied_to_vacant",
            confidence_score=0.75 + i * 0.05,
            model_type="lstm",
            model_version="v1.0",
            created_at=datetime.utcnow(),
        )
        session.add(prediction)

    await session.commit()
    yield session


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
        "HA_URL": "http://test-ha:8123",
        "HA_TOKEN": "test_token_12345",
        "DB_URL": TEST_DB_URL,
        "TESTING": "1",
    }

    os.environ.update(test_vars)

    yield test_vars

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: mark test as a unit test")
    config.addinivalue_line("markers", "integration: mark test as an integration test")
    config.addinivalue_line("markers", "database: mark test as requiring database")
    config.addinivalue_line(
        "markers", "ha_client: mark test as requiring Home Assistant client"
    )
    config.addinivalue_line("markers", "slow: mark test as slow running")


# Helper functions for tests
def assert_sensor_event_equal(
    event1: SensorEvent, event2: SensorEvent, ignore_timestamps=True
):
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
    attributes: Optional[Dict[str, Any]] = None,
) -> HAEvent:
    """Create a test Home Assistant event."""
    if timestamp is None:
        timestamp = datetime.utcnow()
    if attributes is None:
        attributes = {"device_class": "motion"}

    return HAEvent(
        entity_id=entity_id,
        state=state,
        previous_state=previous_state,
        timestamp=timestamp,
        attributes=attributes,
    )


def create_test_sensor_event(
    room_id: str = "test_room",
    sensor_id: str = "binary_sensor.test_sensor",
    sensor_type: str = "presence",
    state: str = "on",
    timestamp: Optional[datetime] = None,
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
        attributes={"test": True},
        is_human_triggered=True,
        confidence_score=0.8,
        created_at=datetime.utcnow(),
    )
