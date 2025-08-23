"""
pytest configuration and fixtures for the occupancy prediction system.

This module provides shared fixtures and configuration for all tests.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio


# Set up JWT test environment variables before any imports
# This must happen before modules try to import api_server at collection time
def _setup_jwt_test_env():
    """Set up JWT test environment variables for test collection."""
    if not os.environ.get("JWT_SECRET_KEY"):
        os.environ["JWT_SECRET_KEY"] = (
            "test_jwt_secret_key_for_security_validation_testing_at_least_32_characters_long"
        )
        os.environ["JWT_ALGORITHM"] = "HS256"
        os.environ["JWT_ACCESS_TOKEN_EXPIRE_MINUTES"] = "60"
        os.environ["JWT_REFRESH_TOKEN_EXPIRE_DAYS"] = "30"
        os.environ["JWT_ISSUER"] = "ha-ml-predictor-test"
        os.environ["JWT_AUDIENCE"] = "ha-ml-predictor-api-test"
        os.environ["JWT_REQUIRE_HTTPS"] = "false"
        os.environ["JWT_BLACKLIST_ENABLED"] = "true"
        os.environ["API_KEY_ENABLED"] = "true"
        os.environ["API_KEY"] = "test_api_key_for_security_validation_testing"
        os.environ["ENVIRONMENT"] = "test"
        os.environ["DEBUG"] = "true"

        # Disable background tasks in test environment
        os.environ["DISABLE_BACKGROUND_TASKS"] = "true"
        os.environ["TESTING"] = "true"


# Set up environment variables immediately when conftest is imported
_setup_jwt_test_env()
from sqlalchemy import text
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from src.core.config import (
    DatabaseConfig,
    HomeAssistantConfig,
    RoomConfig,
    SystemConfig,
)
from src.core.constants import SensorState, SensorType
from src.data.ingestion.event_processor import EventProcessor
from src.data.ingestion.ha_client import HAEvent, HomeAssistantClient
from src.data.storage.database import DatabaseManager
from src.data.storage.models import Base, Prediction, RoomState, SensorEvent


def _patch_models_for_sqlite():
    """
    Patch models to be SQLite-compatible by removing composite primary keys.

    This function modifies the SensorEvent model to use a single primary key (id)
    instead of the composite primary key (id, timestamp) that SQLite doesn't support
    with autoincrement.
    """
    from src.data.storage.models import SensorEvent

    # Check if the model has already been patched
    if hasattr(SensorEvent, "_sqlite_patched"):
        return

    # Mark as patched to avoid double-patching
    SensorEvent._sqlite_patched = True

    # For SQLite, we need to ensure only 'id' is the primary key
    # The timestamp column should remain indexed but not part of primary key
    if hasattr(SensorEvent, "__table__"):
        table = SensorEvent.__table__

        # Remove timestamp from primary key if it exists
        timestamp_col = None
        for col in table.columns:
            if col.name == "timestamp":
                timestamp_col = col
                break

        if timestamp_col is not None and hasattr(timestamp_col, "primary_key"):
            try:
                if timestamp_col.primary_key:
                    timestamp_col.primary_key = False
            except (TypeError, AttributeError):
                # Skip if primary_key property evaluation fails
                pass

        # Ensure id column is the only primary key with autoincrement
        id_col = None
        for col in table.columns:
            if col.name == "id":
                id_col = col
                break

        if id_col is not None:
            id_col.primary_key = True
            id_col.autoincrement = True


# Test database configuration
# Use SQLite in-memory database for testing by default, but mock the tests
# Environment variable TEST_DB_URL can override this for CI/CD with real PostgreSQL
TEST_DB_URL = os.getenv("TEST_DB_URL", "sqlite+aiosqlite:///:memory:")

# Flag to determine if we should use mock databases for tests
USE_MOCK_DB = True


# Use pytest-asyncio event loop with proper cleanup via autouse fixture
# The cleanup_background_tasks fixture handles task cleanup automatically


@pytest.fixture(autouse=True)
async def cleanup_background_tasks():
    """Automatically clean up background tasks after each test."""
    # Store initial tasks
    initial_tasks = set(asyncio.all_tasks())

    yield

    # Find and cancel any new tasks created during the test
    current_tasks = set(asyncio.all_tasks())
    new_tasks = current_tasks - initial_tasks

    if new_tasks:
        for task in new_tasks:
            if not task.done():
                task.cancel()

        # Wait for cancellation to complete
        if new_tasks:
            await asyncio.gather(*new_tasks, return_exceptions=True)


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
            "tracking": {
                "enabled": True,
                "monitoring_interval_seconds": 60,
                "auto_validation_enabled": True,
                "validation_window_minutes": 30,
                "drift_detection_enabled": True,
                "drift_threshold": 0.1,
                "auto_retraining_enabled": True,
                "adaptive_retraining_enabled": True,
            },
            "api": {
                "enabled": False,  # Disable API for tests to avoid port conflicts
                "host": "127.0.0.1",
                "port": 8000,
                "debug": True,
                "background_tasks_enabled": False,  # Disable background tasks in tests
            },
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

            # For SQLite, patch models to avoid composite PK issues
            if "sqlite" in TEST_DB_URL:
                # Patch the models for SQLite compatibility before creating tables
                _patch_models_for_sqlite()
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
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)

    events = []
    for i in range(10):
        event = SensorEvent(
            # Explicitly set id=None to allow autoincrement
            id=None,
            room_id="test_room",
            sensor_id=f"binary_sensor.test_sensor_{i % 3}",
            sensor_type=SensorType.PRESENCE.value,
            state=(SensorState.ON.value if i % 2 == 0 else SensorState.OFF.value),
            previous_state=(
                SensorState.OFF.value if i % 2 == 0 else SensorState.ON.value
            ),
            timestamp=base_time + timedelta(minutes=i * 5),
            attributes={"test": True, "sequence": i},
            is_human_triggered=True,
            confidence_score=0.8 + (i * 0.02),
            # Let created_at use its default=func.now() instead of explicitly setting it
            # created_at=datetime.now(datetime.UTC),
        )
        events.append(event)

    return events


@pytest.fixture
def sample_ha_events():
    """Create sample Home Assistant events for testing."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)

    events = []
    for i in range(5):
        event = HAEvent(
            entity_id=f"binary_sensor.test_sensor_{i}",
            state=(SensorState.ON.value if i % 2 == 0 else SensorState.OFF.value),
            previous_state=(
                SensorState.OFF.value if i % 2 == 0 else SensorState.ON.value
            ),
            timestamp=base_time + timedelta(minutes=i * 10),
            attributes={
                "device_class": "motion",
                "friendly_name": f"Test Sensor {i}",
            },
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
            # Explicitly set id=None to allow autoincrement
            id=None,
            room_id="test_room",
            timestamp=datetime.now(timezone.utc) - timedelta(hours=i),
            is_occupied=i % 2 == 0,
            occupancy_confidence=0.8,
            occupant_type="human",
            state_duration=300 + i * 60,
            transition_trigger=f"binary_sensor.test_sensor_{i}",
            # Let created_at use its default instead of explicitly setting it
            # created_at=datetime.now(datetime.UTC),
        )
        session.add(room_state)

    # Add sample predictions
    for i in range(3):
        prediction = Prediction(
            # Explicitly set id=None to allow autoincrement
            id=None,
            room_id="test_room",
            prediction_time=datetime.now(timezone.utc) - timedelta(hours=i),
            predicted_transition_time=datetime.now(timezone.utc)
            + timedelta(minutes=15 + i * 5),
            transition_type="occupied_to_vacant",
            confidence_score=0.75 + i * 0.05,
            model_type="lstm",
            model_version="v1.0",
            # Let created_at use its default instead of explicitly setting it
            # created_at=datetime.now(datetime.UTC),
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
    previous_state: Optional[str] = "of",
    timestamp: Optional[datetime] = None,
    attributes: Optional[Dict[str, Any]] = None,
) -> HAEvent:
    """Create a test Home Assistant event."""
    if timestamp is None:
        timestamp = datetime.now(timezone.utc)
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
        timestamp = datetime.now(timezone.utc)

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
        created_at=datetime.now(timezone.utc),
    )


# Missing fixtures for production-grade test infrastructure


@pytest.fixture
def mock_validator():
    """Create mock prediction validator for AccuracyTracker tests."""
    from src.adaptation.validator import PredictionValidator

    validator = AsyncMock(spec=PredictionValidator)
    validator._lock = AsyncMock()
    validator._validation_records = {}
    validator._pending_predictions = {}
    validator.record_prediction = AsyncMock()
    validator.validate_prediction = AsyncMock()

    # Mock get_accuracy_metrics with realistic data
    validator.get_accuracy_metrics = AsyncMock(
        return_value={
            "mean_error_minutes": 12.5,
            "median_error_minutes": 8.0,
            "accuracy_within_threshold": 0.82,
            "total_predictions": 150,
            "validated_predictions": 128,
            "confidence_score": 0.75,
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "time_range": {
                "start": (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat(),
                "end": datetime.now(timezone.utc).isoformat(),
            },
        }
    )

    return validator


@pytest.fixture
def sample_training_data():
    """Create realistic sample training data for ML model tests."""
    np.random.seed(42)  # Ensure reproducible test data

    # Generate 1000 samples with realistic occupancy prediction features
    n_samples = 1000

    # Temporal features
    features = pd.DataFrame(
        {
            # Time-based features
            "time_since_last_change": np.random.exponential(300, n_samples),  # seconds
            "current_state_duration": np.random.exponential(600, n_samples),  # seconds
            "hour_sin": np.sin(2 * np.pi * np.random.rand(n_samples)),
            "hour_cos": np.cos(2 * np.pi * np.random.rand(n_samples)),
            "day_sin": np.sin(2 * np.pi * np.random.rand(n_samples) / 7),
            "day_cos": np.cos(2 * np.pi * np.random.rand(n_samples) / 7),
            "is_weekend": np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
            # Sequential features
            "movement_velocity": np.random.gamma(2, 0.5, n_samples),
            "room_transition_pattern": np.random.randint(0, 10, n_samples),
            "trigger_sequence_score": np.random.beta(2, 3, n_samples),
            # Contextual features
            "temperature": np.random.normal(22, 3, n_samples),
            "humidity": np.random.normal(45, 10, n_samples),
            "light_level": np.random.exponential(100, n_samples),
            "other_rooms_occupied": np.random.randint(0, 4, n_samples),
            # Historical patterns
            "historical_pattern_similarity": np.random.beta(3, 2, n_samples),
            "occupancy_trend": np.random.normal(0, 1, n_samples),
        }
    )

    # Target variable: time until next transition (in seconds)
    # Use realistic distribution - exponential with some correlation to features
    base_time = np.random.exponential(1800, n_samples)  # 30 min average

    # Add feature-based variation
    time_adjustment = (
        features["time_since_last_change"] * 0.1
        + features["current_state_duration"] * 0.05
        + features["is_weekend"] * 300  # Weekend effect
        + features["historical_pattern_similarity"] * 600  # Pattern effect
    )

    targets = pd.DataFrame(
        {
            "time_until_transition_seconds": np.maximum(
                60, base_time + time_adjustment
            )  # Min 1 minute
        }
    )

    return features, targets


@pytest.fixture
def large_sample_training_data():
    """Create larger realistic training dataset for comprehensive tests."""
    np.random.seed(42)
    n_samples = 5000  # Large dataset for robust testing

    # Generate time series with realistic patterns
    time_index = pd.date_range(
        start=datetime.now(timezone.utc) - timedelta(days=30),
        periods=n_samples,
        freq="5min",
    )

    features = pd.DataFrame(
        {
            "timestamp": time_index,
            "time_since_last_change": np.random.exponential(450, n_samples),
            "current_state_duration": np.random.exponential(900, n_samples),
            "hour_sin": np.sin(2 * np.pi * time_index.hour / 24),
            "hour_cos": np.cos(2 * np.pi * time_index.hour / 24),
            "day_sin": np.sin(2 * np.pi * time_index.dayofweek / 7),
            "day_cos": np.cos(2 * np.pi * time_index.dayofweek / 7),
            "is_weekend": (time_index.dayofweek >= 5).astype(int),
            "movement_velocity": np.random.gamma(2.5, 0.6, n_samples),
            "room_transition_pattern": np.random.randint(0, 12, n_samples),
            "temperature": 20
            + 5 * np.sin(2 * np.pi * time_index.hour / 24)
            + np.random.normal(0, 1, n_samples),
            "humidity": 50 + np.random.normal(0, 8, n_samples),
            "light_level": np.maximum(
                0,
                80
                + 70 * np.sin(2 * np.pi * time_index.hour / 24)
                + np.random.normal(0, 15, n_samples),
            ),
            "other_rooms_occupied": np.random.poisson(1.2, n_samples),
            "historical_pattern_similarity": np.random.beta(2.5, 2, n_samples),
            "occupancy_trend": np.random.normal(0, 0.8, n_samples),
        }
    )

    # More sophisticated target generation with daily patterns
    hour_effect = 300 * np.sin(2 * np.pi * (time_index.hour - 6) / 24)  # Peak at noon
    weekend_effect = features["is_weekend"] * 600  # Longer weekend transitions
    pattern_effect = features["historical_pattern_similarity"] * 800

    base_transition_time = np.random.exponential(1200, n_samples)  # 20 min average
    targets = pd.DataFrame(
        {
            "time_until_transition_seconds": np.maximum(
                30, base_transition_time + hour_effect + weekend_effect + pattern_effect
            )
        }
    )

    return features, targets


@pytest.fixture
def sample_room_states_data():
    """Create sample room states data for database tests."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=2)

    room_states = []
    for i in range(800):  # Generate 800 records to match test expectations
        room_state = RoomState(
            id=None,  # Let autoincrement handle this
            room_id=f"room_{i % 4}",  # 4 different rooms
            timestamp=base_time + timedelta(minutes=i * 2),
            is_occupied=(i % 3 != 0),  # 2/3 occupied, 1/3 vacant
            occupancy_confidence=0.6 + (i % 10) * 0.04,  # 0.6 to 0.96
            occupant_type="human" if i % 10 < 8 else "cat",
            state_duration=180 + (i % 20) * 30,  # 3-13 minutes
            transition_trigger=f"binary_sensor.room_{i % 4}_motion_{i % 3}",
        )
        room_states.append(room_state)

    return room_states


@pytest.fixture
def sample_prediction_validation_data():
    """Create sample prediction validation data for accuracy tests."""
    base_time = datetime.now(timezone.utc) - timedelta(hours=1)

    validation_data = []
    for i in range(44):  # Generate 44 records to match test expectations
        # Create realistic prediction validation records
        predicted_time = base_time + timedelta(minutes=i * 5)
        actual_time = predicted_time + timedelta(
            minutes=np.random.normal(0, 8)
        )  # Some error

        validation_record = {
            "room_id": f"test_room_{i % 3}",
            "predicted_time": predicted_time,
            "actual_time": actual_time,
            "confidence": 0.5 + (i % 10) * 0.05,  # 0.5 to 0.95
            "error_minutes": abs((actual_time - predicted_time).total_seconds() / 60),
            "within_threshold": abs((actual_time - predicted_time).total_seconds() / 60)
            <= 15,
            "prediction_id": f"pred_{i:04d}",
            "model_type": ["lstm", "xgboost", "hmm", "gp"][i % 4],
        }
        validation_data.append(validation_record)

    return validation_data
