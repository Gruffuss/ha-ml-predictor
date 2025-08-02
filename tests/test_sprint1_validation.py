"""
Sprint 1 Validation Tests

This module contains comprehensive validation tests to ensure all Sprint 1 
components are working correctly before proceeding to Sprint 2.
"""

import pytest
import pytest_asyncio
import asyncio
from datetime import datetime, timedelta
from pathlib import Path

# Test that all core imports work
def test_sprint1_imports():
    """Test that all Sprint 1 components can be imported successfully."""
    
    # Core components
    from src.core.config import SystemConfig, ConfigLoader, get_config
    from src.core.constants import SensorType, SensorState, EventType
    from src.core.exceptions import OccupancyPredictionError, DatabaseConnectionError
    
    # Data storage components  
    from src.data.storage.models import SensorEvent, RoomState, Prediction
    from src.data.storage.database import DatabaseManager, get_database_manager
    
    # Data ingestion components
    from src.data.ingestion.ha_client import HomeAssistantClient, HAEvent
    from src.data.ingestion.event_processor import EventProcessor, MovementPatternClassifier
    from src.data.ingestion.bulk_importer import BulkImporter, ImportProgress
    
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
    assert config.database.connection_string.startswith("postgresql")
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


@pytest.mark.asyncio
async def test_sprint1_database_system(test_system_config):
    """Test that the database system works end-to-end."""
    from src.data.storage.database import DatabaseManager
    from src.data.storage.models import SensorEvent, RoomState
    
    # Override with test database
    test_system_config.database.connection_string = "postgresql+asyncpg://localhost/testdb"
    
    # Initialize database manager
    manager = DatabaseManager(test_system_config.database)
    await manager.initialize()
    
    try:
        # Test session creation
        async with manager.get_session() as session:
            # Create and save a sensor event
            event = SensorEvent(
                room_id="sprint1_test_room",
                sensor_id="binary_sensor.sprint1_test",
                sensor_type="motion",
                state="on",
                timestamp=datetime.utcnow()
            )
            session.add(event)
            await session.commit()
            
            # Create and save a room state
            room_state = RoomState(
                room_id="sprint1_test_room",
                timestamp=datetime.utcnow(),
                is_occupied=True,
                occupancy_confidence=0.9
            )
            session.add(room_state)
            await session.commit()
        
        # Test health check
        health = await manager.health_check()
        assert health['status'] == 'healthy'
        
    finally:
        await manager.close()


def test_sprint1_ha_client_structure(test_system_config):
    """Test that HA client components are properly structured."""
    from src.data.ingestion.ha_client import HomeAssistantClient, HAEvent, RateLimiter
    
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
        attributes={"device_class": "motion"}
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
    from src.data.ingestion.event_processor import EventProcessor, EventValidator
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
        attributes={"device_class": "motion"}
    )
    
    # Test that event can be processed (structure-wise)
    assert ha_event.is_valid()
    
    # Test processing stats
    stats = processor.get_processing_stats()
    assert isinstance(stats, dict)
    assert 'total_processed' in stats


def test_sprint1_bulk_importer_structure():
    """Test that bulk importer components are properly structured."""
    from src.data.ingestion.bulk_importer import BulkImporter, ImportProgress, ImportConfig
    
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
        OccupancyPredictionError, ConfigurationError, DatabaseConnectionError,
        HomeAssistantConnectionError, ModelTrainingError
    )
    
    # Test base exception
    base_error = OccupancyPredictionError(
        "Test error",
        error_code="TEST_001",
        context={"test": True}
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
        SensorType, SensorState, EventType, ModelType,
        PRESENCE_STATES, ABSENCE_STATES, INVALID_STATES,
        MIN_EVENT_SEPARATION, MAX_SEQUENCE_GAP,
        TEMPORAL_FEATURE_NAMES, MQTT_TOPICS, DB_TABLES
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


@pytest.mark.asyncio
async def test_sprint1_model_relationships(test_db_session):
    """Test that database model relationships work correctly."""
    from src.data.storage.models import SensorEvent, RoomState, Prediction
    
    # Create related models
    sensor_event = SensorEvent(
        room_id="sprint1_validation_room",
        sensor_id="binary_sensor.validation_test",
        sensor_type="motion",
        state="on",
        timestamp=datetime.utcnow()
    )
    test_db_session.add(sensor_event)
    await test_db_session.flush()
    
    room_state = RoomState(
        room_id="sprint1_validation_room",
        timestamp=datetime.utcnow(),
        is_occupied=True,
        occupancy_confidence=0.9
    )
    test_db_session.add(room_state)
    await test_db_session.flush()
    
    prediction = Prediction(
        room_id="sprint1_validation_room",
        prediction_time=datetime.utcnow(),
        predicted_transition_time=datetime.utcnow() + timedelta(minutes=15),
        transition_type="occupied_to_vacant",
        confidence_score=0.8,
        model_type="lstm",
        model_version="v1.0",
        triggering_event_id=sensor_event.id,
        room_state_id=room_state.id
    )
    test_db_session.add(prediction)
    await test_db_session.commit()
    
    # Test relationships
    await test_db_session.refresh(prediction)
    assert prediction.triggering_event is not None
    assert prediction.room_state is not None
    assert prediction.triggering_event.id == sensor_event.id
    assert prediction.room_state.id == room_state.id


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
    assert (base_path / "src" / "data" / "ingestion" / "event_processor.py").exists()
    assert (base_path / "src" / "data" / "ingestion" / "bulk_importer.py").exists()
    
    # Configuration files
    assert (base_path / "config" / "config.yaml").exists()
    assert (base_path / "config" / "rooms.yaml").exists()
    
    # Test files
    assert (base_path / "tests" / "conftest.py").exists()
    assert (base_path / "tests" / "unit" / "test_core").exists()
    assert (base_path / "tests" / "unit" / "test_data").exists()
    assert (base_path / "tests" / "integration").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sprint1_end_to_end_workflow(test_system_config):
    """Test a complete end-to-end workflow for Sprint 1."""
    from src.core.config import ConfigLoader
    from src.data.storage.database import DatabaseManager
    from src.data.storage.models import SensorEvent, RoomState
    from src.data.ingestion.ha_client import HAEvent
    from src.data.ingestion.event_processor import EventProcessor
    
    # 1. Configuration loading
    test_system_config.database.connection_string = "postgresql+asyncpg://localhost/testdb"
    
    # 2. Database initialization
    db_manager = DatabaseManager(test_system_config.database)
    await db_manager.initialize()
    
    try:
        # 3. Event processing workflow
        processor = EventProcessor(test_system_config)
        
        # Create HA event
        ha_event = HAEvent(
            entity_id="binary_sensor.e2e_test",
            state="on",
            previous_state="off",
            timestamp=datetime.utcnow(),
            attributes={"device_class": "motion"}
        )
        
        # Process event (this will be limited without full room config)
        # But we can test the structure
        assert ha_event.is_valid()
        
        # 4. Database operations
        async with db_manager.get_session() as session:
            # Save sensor event
            sensor_event = SensorEvent(
                room_id="e2e_test_room",
                sensor_id="binary_sensor.e2e_test",
                sensor_type="motion",
                state="on",
                timestamp=datetime.utcnow()
            )
            session.add(sensor_event)
            
            # Save room state
            room_state = RoomState(
                room_id="e2e_test_room",  
                timestamp=datetime.utcnow(),
                is_occupied=True,
                occupancy_confidence=0.9
            )
            session.add(room_state)
            await session.commit()
        
        # 5. Verify data persistence
        async with db_manager.get_session() as session:
            from sqlalchemy import select
            
            # Check sensor event was saved
            result = await session.execute(
                select(SensorEvent).where(SensorEvent.sensor_id == "binary_sensor.e2e_test")
            )
            saved_event = result.scalar_one_or_none()
            assert saved_event is not None
            assert saved_event.state == "on"
            
            # Check room state was saved  
            result = await session.execute(
                select(RoomState).where(RoomState.room_id == "e2e_test_room")
            )
            saved_state = result.scalar_one_or_none()
            assert saved_state is not None
            assert saved_state.is_occupied is True
        
        # 6. Health check
        health = await db_manager.health_check()
        assert health['status'] == 'healthy'
        
    finally:
        await db_manager.close()


@pytest.mark.smoke
def test_sprint1_smoke_test():
    """Smoke test to verify basic Sprint 1 functionality."""
    # This test should run very quickly and catch major issues
    
    # Test imports work
    from src.core.config import SystemConfig
    from src.core.constants import SensorType
    from src.core.exceptions import OccupancyPredictionError
    from src.data.storage.models import SensorEvent
    from src.data.ingestion.ha_client import HomeAssistantClient
    
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
        timestamp=datetime.utcnow()
    )
    assert sensor_event.room_id == "test"
    
    # Test configuration structure exists
    from src.core.config import HomeAssistantConfig, DatabaseConfig
    ha_config = HomeAssistantConfig(url="http://test", token="test")
    db_config = DatabaseConfig(connection_string="postgresql://localhost/testdb")
    
    assert ha_config.url == "http://test"
    assert db_config.connection_string == "postgresql://localhost/testdb"


if __name__ == "__main__":
    """
    Run Sprint 1 validation tests directly.
    
    Usage: python tests/test_sprint1_validation.py
    """
    pytest.main([__file__, "-v", "--tb=short"])