"""
Sprint 1 Validation Tests

This module contains comprehensive validation tests to ensure all Sprint 1
components are working correctly before proceeding to Sprint 2.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import pytest_asyncio


# Test that all core imports work
def test_sprint1_imports():
    """Test that all Sprint 1 components can be imported successfully."""

    # Core components
    from src.core.config import ConfigLoader, SystemConfig, get_config
    from src.core.constants import EventType, SensorState, SensorType
    from src.core.exceptions import (
        DatabaseConnectionError,
        OccupancyPredictionError,
    )
    from src.data.ingestion.bulk_importer import BulkImporter, ImportProgress
    from src.data.ingestion.event_processor import (
        EventProcessor,
        MovementPatternClassifier,
    )

    # Data ingestion components
    from src.data.ingestion.ha_client import HAEvent, HomeAssistantClient
    from src.data.storage.database import DatabaseManager, get_database_manager

    # Data storage components
    from src.data.storage.models import Prediction, RoomState, SensorEvent

    # All imports successful
    assert True


def test_sprint1_config_system(test_config_dir):
    """Test that the configuration system works end-to-end."""
    from src.core.config import ConfigLoader

    # Load configuration
    loader = ConfigLoader(test_config_dir)
    config = loader.load_config()

    # Verify configuration structure
    assert config.home_assistant.url == "http://test-ha:8123"
    assert (
        config.database.connection_string
    )  # Just verify it exists and is not empty
    assert len(config.rooms) >= 2

    # Verify room configuration
    assert "test_room" in config.rooms
    test_room = config.rooms["test_room"]
    assert test_room.name == "Test Room"

    # Verify entity extraction
    entity_ids = config.get_all_entity_ids()
    assert len(entity_ids) > 0

    # Verify room lookup
    if entity_ids:
        room = config.get_room_by_entity_id(entity_ids[0])
        assert room is not None


def test_sprint1_database_system():
    """Test that the database system components are properly structured."""
    from unittest.mock import AsyncMock, MagicMock

    from src.core.config import DatabaseConfig
    from src.data.storage.database import DatabaseManager
    from src.data.storage.models import RoomState, SensorEvent

    # Test that models can be instantiated
    event = SensorEvent(
        room_id="sprint1_test_room",
        sensor_id="binary_sensor.sprint1_test",
        sensor_type="motion",
        state="on",
        timestamp=datetime.utcnow(),
    )
    assert event.room_id == "sprint1_test_room"
    assert event.sensor_type == "motion"
    assert event.state == "on"

    room_state = RoomState(
        room_id="sprint1_test_room",
        timestamp=datetime.utcnow(),
        is_occupied=True,
        occupancy_confidence=0.9,
    )
    assert room_state.room_id == "sprint1_test_room"
    assert room_state.is_occupied is True
    assert room_state.occupancy_confidence == 0.9

    # Test DatabaseManager can be instantiated
    db_config = DatabaseConfig(
        connection_string="postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db",
        pool_size=5,
        max_overflow=10,
    )
    manager = DatabaseManager(db_config)
    assert manager.config == db_config

    # Test that health check method exists
    assert hasattr(manager, "health_check")
    assert hasattr(manager, "get_session")


def test_sprint1_ha_client_structure(test_system_config):
    """Test that HA client components are properly structured."""
    from src.data.ingestion.ha_client import (
        HAEvent,
        HomeAssistantClient,
        RateLimiter,
    )

    # Test client initialization
    client = HomeAssistantClient(test_system_config)
    assert client.config == test_system_config
    assert client.ha_config == test_system_config.home_assistant
    assert isinstance(client.rate_limiter, RateLimiter)

    # Test HAEvent creation
    event = HAEvent(
        entity_id="binary_sensor.test",
        state="on",
        previous_state="off",
        timestamp=datetime.utcnow(),
        attributes={"device_class": "motion"},
    )
    assert event.is_valid()

    # Test conversion to SensorEvent
    sensor_event = client.convert_ha_event_to_sensor_event(
        event, "test_room", "motion"
    )
    assert sensor_event.room_id == "test_room"
    assert sensor_event.sensor_type == "motion"


def test_sprint1_event_processing():
    """Test that event processing components work correctly."""
    from src.data.ingestion.event_processor import (
        EventProcessor,
        EventValidator,
    )
    from src.data.ingestion.ha_client import HAEvent

    # Create processor with test config
    processor = EventProcessor()

    # Test validator
    validator = EventValidator(processor.config)

    # Create test HAEvent
    ha_event = HAEvent(
        entity_id="binary_sensor.test_motion",
        state="on",
        previous_state="off",
        timestamp=datetime.utcnow(),
        attributes={"device_class": "motion"},
    )

    # Test that event can be processed (structure-wise)
    assert ha_event.is_valid()

    # Test processing stats
    stats = processor.get_processing_stats()
    assert isinstance(stats, dict)
    assert "total_processed" in stats


def test_sprint1_bulk_importer_structure():
    """Test that bulk importer components are properly structured."""
    from src.data.ingestion.bulk_importer import (
        BulkImporter,
        ImportConfig,
        ImportProgress,
    )

    # Test ImportProgress
    progress = ImportProgress()
    assert progress.total_entities == 0
    assert progress.processed_entities == 0
    assert progress.duration_seconds >= 0

    # Test ImportConfig
    config = ImportConfig(months_to_import=3, batch_size=500)
    assert config.months_to_import == 3
    assert config.batch_size == 500

    # Test BulkImporter initialization
    importer = BulkImporter()
    assert isinstance(importer.import_config, ImportConfig)
    assert isinstance(importer.progress, ImportProgress)


def test_sprint1_exception_handling():
    """Test that exception handling works correctly."""
    from src.core.exceptions import (
        ConfigurationError,
        DatabaseConnectionError,
        HomeAssistantConnectionError,
        ModelTrainingError,
        OccupancyPredictionError,
    )

    # Test base exception
    base_error = OccupancyPredictionError(
        "Test error", error_code="TEST_001", context={"test": True}
    )
    assert "Test error" in str(base_error)
    assert "TEST_001" in str(base_error)

    # Test exception hierarchy
    config_error = ConfigurationError("Config error")
    assert isinstance(config_error, OccupancyPredictionError)

    db_error = DatabaseConnectionError("postgresql://test")
    assert isinstance(db_error, OccupancyPredictionError)

    # Test that exceptions can be caught at different levels
    try:
        raise ConfigurationError("Test config error")
    except OccupancyPredictionError:
        # Should catch at base level
        pass
    except Exception:
        pytest.fail("Should have caught as OccupancyPredictionError")


def test_sprint1_constants_and_enums():
    """Test that constants and enums are properly defined."""
    from src.core.constants import (
        ABSENCE_STATES,
        DB_TABLES,
        INVALID_STATES,
        MAX_SEQUENCE_GAP,
        MIN_EVENT_SEPARATION,
        MQTT_TOPICS,
        PRESENCE_STATES,
        TEMPORAL_FEATURE_NAMES,
        EventType,
        ModelType,
        SensorState,
        SensorType,
    )

    # Test enums have expected values
    assert SensorType.MOTION.value == "motion"
    assert SensorState.ON.value == "on"
    assert EventType.STATE_CHANGE.value == "state_change"

    # Test constants are reasonable
    assert MIN_EVENT_SEPARATION > 0
    assert MAX_SEQUENCE_GAP > MIN_EVENT_SEPARATION

    # Test feature names are defined
    assert len(TEMPORAL_FEATURE_NAMES) > 0
    assert "hour_sin" in TEMPORAL_FEATURE_NAMES

    # Test MQTT topics and DB tables are defined
    assert len(MQTT_TOPICS) > 0
    assert len(DB_TABLES) > 0
    assert "predictions" in MQTT_TOPICS
    assert "sensor_events" in DB_TABLES


def test_sprint1_model_relationships():
    """Test that database model relationships are properly structured."""
    from src.data.storage.models import Prediction, RoomState, SensorEvent

    # Test that models can be instantiated with relationships
    sensor_event = SensorEvent(
        room_id="sprint1_validation_room",
        sensor_id="binary_sensor.validation_test",
        sensor_type="motion",
        state="on",
        timestamp=datetime.utcnow(),
    )
    assert sensor_event.room_id == "sprint1_validation_room"

    room_state = RoomState(
        room_id="sprint1_validation_room",
        timestamp=datetime.utcnow(),
        is_occupied=True,
        occupancy_confidence=0.9,
    )
    assert room_state.room_id == "sprint1_validation_room"

    prediction = Prediction(
        room_id="sprint1_validation_room",
        prediction_time=datetime.utcnow(),
        predicted_transition_time=datetime.utcnow() + timedelta(minutes=15),
        transition_type="occupied_to_vacant",
        confidence_score=0.8,
        model_type="lstm",
        model_version="v1.0",
        triggering_event_id=1,  # Mock ID
        room_state_id=2,  # Mock ID
    )
    assert prediction.room_id == "sprint1_validation_room"
    assert prediction.transition_type == "occupied_to_vacant"
    assert prediction.model_type == "lstm"

    # Test that relationship methods exist
    assert hasattr(prediction, "get_triggering_event")
    assert hasattr(prediction, "get_room_state")
    assert hasattr(sensor_event, "get_predictions")


def test_sprint1_file_structure():
    """Test that all expected Sprint 1 files exist."""
    base_path = Path(__file__).parent.parent

    # Core files
    assert (base_path / "src" / "core" / "__init__.py").exists()
    assert (base_path / "src" / "core" / "config.py").exists()
    assert (base_path / "src" / "core" / "constants.py").exists()
    assert (base_path / "src" / "core" / "exceptions.py").exists()

    # Data storage files
    assert (base_path / "src" / "data" / "storage" / "models.py").exists()
    assert (base_path / "src" / "data" / "storage" / "database.py").exists()

    # Data ingestion files
    assert (base_path / "src" / "data" / "ingestion" / "ha_client.py").exists()
    assert (
        base_path / "src" / "data" / "ingestion" / "event_processor.py"
    ).exists()
    assert (
        base_path / "src" / "data" / "ingestion" / "bulk_importer.py"
    ).exists()

    # Configuration files
    assert (base_path / "config" / "config.yaml").exists()
    assert (base_path / "config" / "rooms.yaml").exists()

    # Test files
    assert (base_path / "tests" / "conftest.py").exists()
    assert (base_path / "tests" / "unit" / "test_core").exists()
    assert (base_path / "tests" / "unit" / "test_data").exists()
    assert (base_path / "tests" / "integration").exists()


@pytest.mark.integration
def test_sprint1_end_to_end_workflow():
    """Test a complete end-to-end workflow for Sprint 1 (structure validation)."""
    from src.data.ingestion.event_processor import EventProcessor
    from src.data.ingestion.ha_client import HAEvent
    from src.data.storage.models import RoomState, SensorEvent

    # 1. Event processing workflow validation
    processor = EventProcessor()
    assert hasattr(processor, "process_event")
    assert hasattr(processor, "get_processing_stats")

    # Create HA event
    ha_event = HAEvent(
        entity_id="binary_sensor.e2e_test",
        state="on",
        previous_state="off",
        timestamp=datetime.utcnow(),
        attributes={"device_class": "motion"},
    )

    # Validate HA event structure
    assert ha_event.is_valid()
    assert ha_event.entity_id == "binary_sensor.e2e_test"
    assert ha_event.state == "on"
    assert ha_event.previous_state == "of"

    # 2. Model instantiation validation
    sensor_event = SensorEvent(
        room_id="e2e_test_room",
        sensor_id="binary_sensor.e2e_test",
        sensor_type="motion",
        state="on",
        timestamp=datetime.utcnow(),
    )
    assert sensor_event.room_id == "e2e_test_room"
    assert sensor_event.sensor_type == "motion"

    room_state = RoomState(
        room_id="e2e_test_room",
        timestamp=datetime.utcnow(),
        is_occupied=True,
        occupancy_confidence=0.9,
    )
    assert room_state.room_id == "e2e_test_room"
    assert room_state.is_occupied is True

    # 3. Test that all Sprint 1 components are importable and structured
    from src.core.config import DatabaseConfig
    from src.data.storage.database import DatabaseManager

    # Test database manager instantiation
    db_config = DatabaseConfig(
        connection_string="postgresql+asyncpg://test:test@localhost:5432/test",
        pool_size=5,
        max_overflow=10,
    )
    db_manager = DatabaseManager(db_config)
    assert hasattr(db_manager, "health_check")
    assert hasattr(db_manager, "get_session")


@pytest.mark.smoke
def test_sprint1_smoke_test():
    """Smoke test to verify basic Sprint 1 functionality."""
    # This test should run very quickly and catch major issues

    # Test imports work
    from src.core.config import SystemConfig
    from src.core.constants import SensorType
    from src.core.exceptions import OccupancyPredictionError
    from src.data.ingestion.ha_client import HomeAssistantClient
    from src.data.storage.models import SensorEvent

    # Test basic object creation
    sensor_type = SensorType.MOTION
    assert sensor_type.value == "motion"

    exception = OccupancyPredictionError("Test")
    assert str(exception) == "Test"

    sensor_event = SensorEvent(
        room_id="test",
        sensor_id="test",
        sensor_type="motion",
        state="on",
        timestamp=datetime.utcnow(),
    )
    assert sensor_event.room_id == "test"

    # Test configuration structure exists
    from src.core.config import DatabaseConfig, HomeAssistantConfig

    ha_config = HomeAssistantConfig(url="http://test", token="test")
    db_config = DatabaseConfig(
        connection_string="postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"
    )

    assert ha_config.url == "http://test"
    assert (
        db_config.connection_string
        == "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"
    )


if __name__ == "__main__":
    """
    Run Sprint 1 validation tests directly.

    Usage: python tests/test_sprint1_validation.py
    """
    pytest.main([__file__, "-v", "--tb=short"])
