"""
Comprehensive unit tests for BulkImporter module.

Tests bulk historical data import functionality including:
- Import configuration and progress tracking
- Data validation and processing
- Error handling and recovery mechanisms
- Performance optimization and integrity checks
- Resume capability and checkpointing
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import pickle
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest
import pytest_asyncio
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import SystemConfig
from src.core.exceptions import (
    DatabaseError,
    DataValidationError,
    HomeAssistantError,
    InsufficientTrainingDataError,
)
from src.data.ingestion.bulk_importer import (
    BulkImporter,
    ImportConfig,
    ImportProgress,
)
from src.data.ingestion.event_processor import EventProcessor
from src.data.ingestion.ha_client import HAEvent, HomeAssistantClient
from src.data.storage.models import SensorEvent


class TestImportProgress:
    """Test ImportProgress dataclass functionality."""

    def test_import_progress_initialization(self):
        """Test ImportProgress initialization with defaults."""
        progress = ImportProgress()

        assert progress.total_entities == 0
        assert progress.processed_entities == 0
        assert progress.total_events == 0
        assert progress.processed_events == 0
        assert progress.valid_events == 0
        assert isinstance(progress.start_time, datetime)
        assert isinstance(progress.last_update, datetime)
        assert progress.current_entity == ""
        assert progress.current_date_range == ""
        assert progress.errors == []

    def test_import_progress_properties(self):
        """Test ImportProgress calculated properties."""
        progress = ImportProgress()
        progress.total_entities = 10
        progress.processed_entities = 5
        progress.total_events = 1000
        progress.processed_events = 250
        progress.start_time = datetime.utcnow() - timedelta(seconds=60)

        # Test percentage calculations
        assert progress.entity_progress_percent == 50.0
        assert progress.event_progress_percent == 25.0

        # Test duration
        assert progress.duration_seconds > 55  # Should be around 60 seconds

        # Test events per second
        assert progress.events_per_second > 4  # Around 4.16 events/second

    def test_import_progress_edge_cases(self):
        """Test ImportProgress edge cases with zero values."""
        progress = ImportProgress()

        # Zero totals should return 0% progress
        assert progress.entity_progress_percent == 0.0
        assert progress.event_progress_percent == 0.0

        # Zero duration should return 0 events per second
        progress.start_time = datetime.utcnow()
        assert progress.events_per_second == 0.0

    def test_import_progress_to_dict(self):
        """Test ImportProgress serialization to dictionary."""
        progress = ImportProgress()
        progress.total_entities = 5
        progress.processed_entities = 2
        progress.errors = ["Error 1", "Error 2"]
        progress.current_entity = "binary_sensor.test"

        data = progress.to_dict()

        assert data["total_entities"] == 5
        assert data["processed_entities"] == 2
        assert data["current_entity"] == "binary_sensor.test"
        assert "start_time" in data
        assert "last_update" in data
        assert "duration_seconds" in data
        assert "entity_progress_percent" in data
        assert "event_progress_percent" in data
        assert "events_per_second" in data
        assert len(data["errors"]) == 2

    def test_import_progress_error_truncation(self):
        """Test that to_dict() truncates errors to last 10."""
        progress = ImportProgress()
        # Add 15 errors
        for i in range(15):
            progress.errors.append(f"Error {i}")

        data = progress.to_dict()

        # Should only have last 10 errors
        assert len(data["errors"]) == 10
        assert data["errors"][0] == "Error 5"  # Should start from 5th error
        assert data["errors"][-1] == "Error 14"  # Should end with 14th error


class TestImportConfig:
    """Test ImportConfig dataclass functionality."""

    def test_import_config_defaults(self):
        """Test ImportConfig default values."""
        config = ImportConfig()

        assert config.months_to_import == 6
        assert config.batch_size == 1000
        assert config.entity_batch_size == 10
        assert config.max_concurrent_entities == 3
        assert config.chunk_days == 7
        assert config.resume_file is None
        assert config.skip_existing is True
        assert config.validate_events is True
        assert config.store_raw_data is False
        assert config.progress_callback is None

    def test_import_config_custom_values(self):
        """Test ImportConfig with custom values."""
        callback = Mock()
        config = ImportConfig(
            months_to_import=12,
            batch_size=2000,
            entity_batch_size=20,
            max_concurrent_entities=5,
            chunk_days=14,
            resume_file="/path/to/resume.pkl",
            skip_existing=False,
            validate_events=False,
            store_raw_data=True,
            progress_callback=callback,
        )

        assert config.months_to_import == 12
        assert config.batch_size == 2000
        assert config.entity_batch_size == 20
        assert config.max_concurrent_entities == 5
        assert config.chunk_days == 14
        assert config.resume_file == "/path/to/resume.pkl"
        assert config.skip_existing is False
        assert config.validate_events is False
        assert config.store_raw_data is True
        assert config.progress_callback == callback


class TestBulkImporter:
    """Test BulkImporter class functionality."""

    def test_bulk_importer_initialization(self, test_system_config):
        """Test BulkImporter initialization."""
        import_config = ImportConfig(batch_size=500)
        importer = BulkImporter(test_system_config, import_config)

        assert importer.config == test_system_config
        assert importer.import_config == import_config
        assert importer.ha_client is None
        assert importer.event_processor is None
        assert isinstance(importer.progress, ImportProgress)
        assert importer._resume_data == {}
        assert importer._completed_entities == set()

        # Check statistics initialization
        expected_stats = {
            "entities_processed": 0,
            "events_imported": 0,
            "events_skipped": 0,
            "validation_errors": 0,
            "database_errors": 0,
            "api_errors": 0,
        }
        assert importer.stats == expected_stats

    def test_bulk_importer_default_config(self):
        """Test BulkImporter with default configuration."""
        with patch("src.data.ingestion.bulk_importer.get_config") as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            importer = BulkImporter()

            assert importer.config == mock_config
            assert isinstance(importer.import_config, ImportConfig)
            mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_components(self, test_system_config):
        """Test component initialization."""
        importer = BulkImporter(test_system_config)

        with (
            patch(
                "src.data.ingestion.bulk_importer.HomeAssistantClient"
            ) as mock_client_class,
            patch(
                "src.data.ingestion.bulk_importer.EventProcessor"
            ) as mock_processor_class,
        ):
            mock_ha_client = AsyncMock()
            mock_client_class.return_value = mock_ha_client
            mock_processor = Mock()
            mock_processor_class.return_value = mock_processor

            await importer._initialize_components()

            # Should create and connect HA client
            mock_client_class.assert_called_once_with(test_system_config)
            mock_ha_client.connect.assert_called_once()
            assert importer.ha_client == mock_ha_client

            # Should create event processor
            mock_processor_class.assert_called_once_with(test_system_config)
            assert importer.event_processor == mock_processor

    @pytest.mark.asyncio
    async def test_cleanup_components(self, test_system_config):
        """Test component cleanup."""
        importer = BulkImporter(test_system_config)

        # Mock components
        mock_ha_client = AsyncMock()
        importer.ha_client = mock_ha_client

        await importer._cleanup_components()

        # Should disconnect HA client
        mock_ha_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_resume_data_file_exists(self, test_system_config):
        """Test loading resume data from existing file."""
        importer = BulkImporter(test_system_config)

        # Create temporary resume file
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            resume_data = {
                "completed_entities": ["entity1", "entity2"],
                "progress": {"total_entities": 10},
                "stats": {"entities_processed": 2},
            }
            pickle.dump(resume_data, temp_file)
            temp_file_path = temp_file.name

        try:
            importer.import_config.resume_file = temp_file_path
            await importer._load_resume_data()

            assert importer._resume_data == resume_data
            assert importer._completed_entities == {"entity1", "entity2"}
        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.asyncio
    async def test_load_resume_data_file_not_exists(self, test_system_config):
        """Test loading resume data when file doesn't exist."""
        importer = BulkImporter(test_system_config)
        importer.import_config.resume_file = "/nonexistent/path.pkl"

        await importer._load_resume_data()

        # Should not modify resume data
        assert importer._resume_data == {}
        assert importer._completed_entities == set()

    @pytest.mark.asyncio
    async def test_load_resume_data_invalid_file(self, test_system_config):
        """Test loading resume data from invalid file."""
        importer = BulkImporter(test_system_config)

        # Create temporary file with invalid data
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as temp_file:
            temp_file.write("invalid pickle data")
            temp_file_path = temp_file.name

        try:
            importer.import_config.resume_file = temp_file_path
            await importer._load_resume_data()

            # Should not crash, just log warning
            assert importer._resume_data == {}
            assert importer._completed_entities == set()
        finally:
            Path(temp_file_path).unlink()

    @pytest.mark.asyncio
    async def test_save_resume_data(self, test_system_config):
        """Test saving resume data to file."""
        importer = BulkImporter(test_system_config)

        # Set up some data
        importer._completed_entities = {"entity1", "entity2"}
        importer.progress.total_entities = 10
        importer.stats["entities_processed"] = 2

        with tempfile.TemporaryDirectory() as temp_dir:
            resume_file = Path(temp_dir) / "resume.pkl"
            importer.import_config.resume_file = str(resume_file)

            await importer._save_resume_data()

            # Verify file was created and data is correct
            assert resume_file.exists()

            with open(resume_file, "rb") as f:
                saved_data = pickle.load(f)

            assert set(saved_data["completed_entities"]) == {"entity1", "entity2"}
            assert saved_data["stats"]["entities_processed"] == 2
            assert "timestamp" in saved_data

    @pytest.mark.asyncio
    async def test_estimate_total_events(self, test_system_config):
        """Test event estimation for progress tracking."""
        importer = BulkImporter(test_system_config)

        # Mock HA client
        mock_ha_client = AsyncMock()
        mock_ha_client.get_entity_history.return_value = [
            {"entity_id": "test", "state": "on", "timestamp": "2023-01-01T00:00:00Z"},
            {"entity_id": "test", "state": "off", "timestamp": "2023-01-01T00:05:00Z"},
        ]
        importer.ha_client = mock_ha_client

        entity_ids = ["entity1", "entity2", "entity3", "entity4", "entity5"]
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 8, tzinfo=timezone.utc)  # 7 days

        await importer._estimate_total_events(entity_ids, start_date, end_date)

        # Should have sampled first 5 entities
        assert mock_ha_client.get_entity_history.call_count == 5

        # Should have estimated total events
        # 2 events per entity per day * 5 entities * 7 days = 70 events
        assert importer.progress.total_events == 70

    @pytest.mark.asyncio
    async def test_estimate_total_events_with_errors(self, test_system_config):
        """Test event estimation with API errors."""
        importer = BulkImporter(test_system_config)

        # Mock HA client with errors - the method converts generic exceptions to HomeAssistantError
        async def mock_get_history(entity_id, start, end):
            if entity_id == "error_entity":
                raise Exception(
                    "Generic API error"
                )  # This will be converted to HomeAssistantError
            return [{"entity_id": entity_id, "state": "on"}]

        mock_ha_client = AsyncMock()
        mock_ha_client.get_entity_history.side_effect = mock_get_history
        importer.ha_client = mock_ha_client

        entity_ids = ["good_entity", "error_entity"]
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

        # Should raise HomeAssistantError when converted from generic exception
        with pytest.raises(HomeAssistantError):
            await importer._estimate_total_events(entity_ids, start_date, end_date)

    @pytest.mark.asyncio
    async def test_process_entities_batch(self, test_system_config):
        """Test processing entities in batches."""
        import_config = ImportConfig(entity_batch_size=2, max_concurrent_entities=1)
        importer = BulkImporter(test_system_config, import_config)

        # Mock _process_entity_with_semaphore
        with patch.object(importer, "_process_entity_with_semaphore") as mock_process:
            mock_process.return_value = None  # Simulate successful processing

            entity_ids = ["entity1", "entity2", "entity3", "entity4", "entity5"]
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

            await importer._process_entities_batch(entity_ids, start_date, end_date)

            # Should have processed all entities
            assert mock_process.call_count == 5

            # Should have updated progress
            assert importer.progress.processed_entities == 5

    @pytest.mark.asyncio
    async def test_process_entities_batch_with_completed(self, test_system_config):
        """Test processing entities batch with some already completed."""
        importer = BulkImporter(test_system_config)

        # Mark some entities as completed
        importer._completed_entities = {"entity1", "entity3"}

        with patch.object(importer, "_process_entity_with_semaphore") as mock_process:
            entity_ids = ["entity1", "entity2", "entity3", "entity4"]
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

            await importer._process_entities_batch(entity_ids, start_date, end_date)

            # Should only process entities not already completed
            assert mock_process.call_count == 2  # entity2 and entity4

            # Verify correct entities were processed
            processed_entities = set()
            for call_args in mock_process.call_args_list:
                processed_entities.add(call_args[0][1])  # Second argument is entity_id

            assert processed_entities == {"entity2", "entity4"}

    @pytest.mark.asyncio
    async def test_process_single_entity(self, test_system_config):
        """Test processing a single entity."""
        import_config = ImportConfig(chunk_days=1)  # Small chunks for testing
        importer = BulkImporter(test_system_config, import_config)

        # Mock HA client and history processing
        mock_ha_client = AsyncMock()
        mock_ha_client.get_entity_history.return_value = [
            {"entity_id": "test_entity", "state": "on"},
            {"entity_id": "test_entity", "state": "off"},
        ]
        importer.ha_client = mock_ha_client

        with patch.object(importer, "_process_history_chunk") as mock_process_chunk:
            mock_process_chunk.return_value = 2  # Processed 2 events

            entity_id = "test_entity"
            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 3, tzinfo=timezone.utc)  # 2 days

            await importer._process_single_entity(entity_id, start_date, end_date)

            # Should have processed 2 chunks (1 day each)
            assert mock_ha_client.get_entity_history.call_count == 2
            assert mock_process_chunk.call_count == 2

            # Should mark entity as completed
            assert entity_id in importer._completed_entities
            assert importer.stats["entities_processed"] == 1

    @pytest.mark.asyncio
    async def test_process_single_entity_with_errors(self, test_system_config):
        """Test processing a single entity with errors."""
        importer = BulkImporter(test_system_config)

        # Mock HA client with error
        mock_ha_client = AsyncMock()
        mock_ha_client.get_entity_history.side_effect = Exception("API error")
        importer.ha_client = mock_ha_client

        entity_id = "error_entity"
        start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end_date = datetime(2023, 1, 2, tzinfo=timezone.utc)

        await importer._process_single_entity(entity_id, start_date, end_date)

        # Should handle error gracefully
        assert len(importer.progress.errors) > 0
        assert importer.stats["api_errors"] == 1

        # Should still mark entity as completed
        assert entity_id in importer._completed_entities

    def test_convert_history_record_to_ha_event(self, test_system_config):
        """Test converting history record to HAEvent."""
        importer = BulkImporter(test_system_config)

        record = {
            "entity_id": "binary_sensor.test",
            "state": "on",
            "last_changed": "2023-01-01T12:00:00Z",
            "attributes": {"device_class": "motion"},
        }

        ha_event = importer._convert_history_record_to_ha_event(record)

        assert ha_event is not None
        assert ha_event.entity_id == "binary_sensor.test"
        assert ha_event.state == "on"
        assert ha_event.previous_state is None
        assert ha_event.timestamp == datetime(2023, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        assert ha_event.attributes == {"device_class": "motion"}

    def test_convert_history_record_invalid_timestamp(self, test_system_config):
        """Test converting history record with invalid timestamp."""
        importer = BulkImporter(test_system_config)

        record = {
            "entity_id": "binary_sensor.test",
            "state": "on",
            "last_changed": "invalid_timestamp",
            "attributes": {},
        }

        ha_event = importer._convert_history_record_to_ha_event(record)

        # Should return None for invalid timestamp
        assert ha_event is None

    def test_convert_history_record_missing_timestamp(self, test_system_config):
        """Test converting history record with missing timestamp."""
        importer = BulkImporter(test_system_config)

        record = {
            "entity_id": "binary_sensor.test",
            "state": "on",
            "attributes": {},
        }

        ha_event = importer._convert_history_record_to_ha_event(record)

        # Should return None for missing timestamp
        assert ha_event is None

    def test_determine_sensor_type(self, test_system_config):
        """Test sensor type determination."""
        importer = BulkImporter(test_system_config)

        # Create mock room config
        mock_room_config = Mock()
        mock_room_config.sensors = {
            "presence": {
                "main": "binary_sensor.test_presence",
                "secondary": "binary_sensor.test_motion",
            },
            "door": "binary_sensor.test_door",
        }

        # Test exact match in config
        sensor_type = importer._determine_sensor_type(
            "binary_sensor.test_presence", mock_room_config
        )
        assert sensor_type == "presence"

        # Test string sensor match
        sensor_type = importer._determine_sensor_type(
            "binary_sensor.test_door", mock_room_config
        )
        assert sensor_type == "door"

        # Test fallback analysis - the method returns "presence" for motion/presence
        sensor_type = importer._determine_sensor_type(
            "binary_sensor.unknown_motion", mock_room_config
        )
        assert sensor_type == "presence"  # Should detect "motion" and return "presence"

        sensor_type = importer._determine_sensor_type(
            "sensor.unknown_temperature", mock_room_config
        )
        assert sensor_type == "climate"  # Should detect "temperature" in name

        # Test default fallback
        sensor_type = importer._determine_sensor_type(
            "binary_sensor.unknown_something", mock_room_config
        )
        assert sensor_type == "motion"  # Default fallback

    @pytest.mark.asyncio
    async def test_convert_ha_events_to_sensor_events(self, test_system_config):
        """Test converting HAEvents to SensorEvents."""
        importer = BulkImporter(test_system_config)

        # Mock config
        mock_room_config = Mock()
        mock_room_config.room_id = "test_room"
        importer.config.get_room_by_entity_id = Mock(return_value=mock_room_config)

        with patch.object(importer, "_determine_sensor_type") as mock_determine_type:
            mock_determine_type.return_value = "presence"

            ha_events = [
                HAEvent(
                    entity_id="binary_sensor.test",
                    state="on",
                    previous_state="off",
                    timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    attributes={"device_class": "motion"},
                ),
                HAEvent(
                    entity_id="binary_sensor.unknown",  # Will be filtered out
                    state="on",
                    previous_state="off",
                    timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                    attributes={},
                ),
            ]

            # Mock room config for second entity to return None
            def mock_get_room(entity_id):
                if entity_id == "binary_sensor.test":
                    return mock_room_config
                return None

            importer.config.get_room_by_entity_id.side_effect = mock_get_room

            sensor_events = await importer._convert_ha_events_to_sensor_events(
                ha_events
            )

            # Should only convert events with valid room config
            assert len(sensor_events) == 1
            assert sensor_events[0].room_id == "test_room"
            assert sensor_events[0].sensor_id == "binary_sensor.test"
            assert sensor_events[0].sensor_type == "presence"
            assert sensor_events[0].state == "on"
            assert sensor_events[0].is_human_triggered is True

    @pytest.mark.asyncio
    async def test_bulk_insert_events(self, test_system_config):
        """Test bulk inserting events to database."""
        importer = BulkImporter(test_system_config)

        # Create mock session and result
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.rowcount = 3
        mock_session.execute.return_value = mock_result

        # Create test events
        events = [
            SensorEvent(
                room_id="test_room",
                sensor_id="binary_sensor.test_1",
                sensor_type="presence",
                state="on",
                timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                attributes={},
                is_human_triggered=True,
                created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ),
            SensorEvent(
                room_id="test_room",
                sensor_id="binary_sensor.test_2",
                sensor_type="presence",
                state="off",
                timestamp=datetime(2023, 1, 1, 1, tzinfo=timezone.utc),
                attributes={},
                is_human_triggered=True,
                created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
            ),
        ]

        with patch(
            "src.data.ingestion.bulk_importer.get_db_session"
        ) as mock_get_session:
            # Create async context manager mock
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_get_session.return_value = mock_context_manager

            result_count = await importer._bulk_insert_events(events)

            assert result_count == 3
            mock_session.execute.assert_called_once()
            mock_session.commit.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_insert_events_empty_list(self, test_system_config):
        """Test bulk inserting empty event list."""
        importer = BulkImporter(test_system_config)

        result_count = await importer._bulk_insert_events([])

        assert result_count == 0

    @pytest.mark.asyncio
    async def test_bulk_insert_events_database_error(self, test_system_config):
        """Test bulk insert with database error."""
        importer = BulkImporter(test_system_config)

        mock_session = AsyncMock(spec=AsyncSession)
        mock_session.execute.side_effect = Exception("Database error")

        events = [
            SensorEvent(
                room_id="test_room",
                sensor_id="binary_sensor.test",
                sensor_type="presence",
                state="on",
                timestamp=datetime(2023, 1, 1, tzinfo=timezone.utc),
                attributes={},
                is_human_triggered=True,
                created_at=datetime(2023, 1, 1, tzinfo=timezone.utc),
            )
        ]

        with patch(
            "src.data.ingestion.bulk_importer.get_db_session"
        ) as mock_get_session:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_get_session.return_value = mock_context_manager

            with pytest.raises(DatabaseError):
                await importer._bulk_insert_events(events)

            assert importer.stats["database_errors"] == 1

    @pytest.mark.asyncio
    async def test_process_history_chunk(self, test_system_config):
        """Test processing a chunk of historical data."""
        importer = BulkImporter(test_system_config)

        # Mock event processor
        mock_processor = AsyncMock()
        mock_sensor_events = [Mock(), Mock()]  # Mock processed events
        mock_processor.process_event_batch.return_value = mock_sensor_events
        importer.event_processor = mock_processor

        with (
            patch.object(
                importer, "_convert_history_record_to_ha_event"
            ) as mock_convert,
            patch.object(importer, "_bulk_insert_events") as mock_insert,
        ):

            # Mock conversion
            mock_ha_event = Mock()
            mock_convert.return_value = mock_ha_event
            mock_insert.return_value = 2

            history_data = [
                {"entity_id": "test", "state": "on"},
                {"entity_id": "test", "state": "off"},
            ]

            result = await importer._process_history_chunk("test_entity", history_data)

            assert result == 2
            assert mock_convert.call_count == 2
            mock_processor.process_event_batch.assert_called_once()
            mock_insert.assert_called_once_with(mock_sensor_events)

    @pytest.mark.asyncio
    async def test_process_history_chunk_validation_disabled(self, test_system_config):
        """Test processing history chunk with validation disabled."""
        import_config = ImportConfig(validate_events=False)
        importer = BulkImporter(test_system_config, import_config)

        with (
            patch.object(
                importer, "_convert_history_record_to_ha_event"
            ) as mock_convert,
            patch.object(
                importer, "_convert_ha_events_to_sensor_events"
            ) as mock_convert_sensor,
            patch.object(importer, "_bulk_insert_events") as mock_insert,
        ):

            mock_ha_event = Mock()
            mock_convert.return_value = mock_ha_event
            mock_sensor_events = [Mock(), Mock()]
            mock_convert_sensor.return_value = mock_sensor_events
            mock_insert.return_value = 2

            history_data = [{"entity_id": "test", "state": "on"}]

            result = await importer._process_history_chunk("test_entity", history_data)

            assert result == 2
            # Should use direct conversion instead of event processor
            mock_convert_sensor.assert_called_once_with([mock_ha_event])
            mock_insert.assert_called_once_with(mock_sensor_events)

    @pytest.mark.asyncio
    async def test_process_history_chunk_conversion_errors(self, test_system_config):
        """Test processing history chunk with conversion errors."""
        importer = BulkImporter(test_system_config)

        with patch.object(
            importer, "_convert_history_record_to_ha_event"
        ) as mock_convert:
            # First record succeeds, second raises DataValidationError, third returns None
            mock_convert.side_effect = [
                Mock(),  # Success
                DataValidationError("validation", ["error"], {}),  # Validation error
                None,  # Invalid record
            ]

            history_data = [
                {"entity_id": "test1", "state": "on"},
                {"entity_id": "test2", "state": "invalid"},
                {"entity_id": "test3", "state": "on"},
            ]

            result = await importer._process_history_chunk("test_entity", history_data)

            # Should handle errors and continue processing
            assert result == 0  # No events successfully processed
            assert importer.stats["validation_errors"] == 1

    @pytest.mark.asyncio
    async def test_update_progress(self, test_system_config):
        """Test progress update functionality."""
        callback_calls = []

        def progress_callback(progress):
            callback_calls.append(progress)

        import_config = ImportConfig(progress_callback=progress_callback)
        importer = BulkImporter(test_system_config, import_config)

        # Set some progress
        importer.progress.processed_entities = 10

        await importer._update_progress()

        # Should have called callback
        assert len(callback_calls) == 1
        assert callback_calls[0] == importer.progress

    @pytest.mark.asyncio
    async def test_update_progress_async_callback(self, test_system_config):
        """Test progress update with async callback."""
        callback_calls = []

        async def async_progress_callback(progress):
            callback_calls.append(progress)

        import_config = ImportConfig(progress_callback=async_progress_callback)
        importer = BulkImporter(test_system_config, import_config)

        await importer._update_progress()

        # Should have called async callback
        assert len(callback_calls) == 1

    @pytest.mark.asyncio
    async def test_update_progress_callback_error(self, test_system_config):
        """Test progress update with callback error."""

        def error_callback(progress):
            raise Exception("Callback error")

        import_config = ImportConfig(progress_callback=error_callback)
        importer = BulkImporter(test_system_config, import_config)

        # Should not raise exception
        await importer._update_progress()

    @pytest.mark.asyncio
    async def test_generate_import_report(self, test_system_config):
        """Test import report generation."""
        importer = BulkImporter(test_system_config)

        # Set up some statistics
        importer.progress.total_entities = 10
        importer.progress.processed_events = 1000
        importer.progress.valid_events = 950
        importer.stats.update(
            {
                "entities_processed": 10,
                "events_imported": 900,
                "events_skipped": 50,
                "validation_errors": 30,
                "database_errors": 10,
                "api_errors": 10,
            }
        )
        importer.progress.errors = ["Error 1", "Error 2"]

        await importer._generate_import_report()

        # Should complete without error (report is logged)

    def test_get_import_stats(self, test_system_config):
        """Test getting import statistics."""
        importer = BulkImporter(test_system_config)

        # Set up some data
        importer.progress.total_entities = 5
        importer.stats["events_imported"] = 100

        stats = importer.get_import_stats()

        assert "progress" in stats
        assert "stats" in stats
        assert stats["progress"]["total_entities"] == 5
        assert stats["stats"]["events_imported"] == 100

    @pytest.mark.asyncio
    async def test_validate_data_sufficiency_sufficient(self, test_system_config):
        """Test data sufficiency validation with sufficient data."""
        importer = BulkImporter(test_system_config)

        # Mock database query results
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(event_date="2023-01-01", event_count=50),
            Mock(event_date="2023-01-02", event_count=45),
            Mock(event_date="2023-01-03", event_count=60),
        ] * 15  # 45 days worth of data
        mock_session.execute.return_value = mock_result

        with patch(
            "src.data.ingestion.bulk_importer.get_db_session"
        ) as mock_get_session:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_get_session.return_value = mock_context_manager

            result = await importer.validate_data_sufficiency("test_room")

            assert result["sufficient"] is True
            assert result["total_days"] == 45
            assert result["meets_day_requirement"] is True
            assert result["meets_event_requirement"] is True
            assert "sufficient for model training" in result["recommendation"]

    @pytest.mark.asyncio
    async def test_validate_data_sufficiency_insufficient(self, test_system_config):
        """Test data sufficiency validation with insufficient data."""
        importer = BulkImporter(test_system_config)

        # Mock database query with insufficient data (only 10 days with low event count)
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.fetchall.return_value = [
            Mock(event_date="2023-01-01", event_count=5),
            Mock(event_date="2023-01-02", event_count=3),
            Mock(event_date="2023-01-03", event_count=4),
            Mock(event_date="2023-01-04", event_count=6),
            Mock(event_date="2023-01-05", event_count=2),
            Mock(event_date="2023-01-06", event_count=7),
            Mock(event_date="2023-01-07", event_count=3),
            Mock(event_date="2023-01-08", event_count=5),
            Mock(event_date="2023-01-09", event_count=4),
            Mock(event_date="2023-01-10", event_count=1),
        ]  # 10 days with avg 4 events/day (insufficient for 10 events/day requirement)
        mock_session.execute.return_value = mock_result

        with patch(
            "src.data.ingestion.bulk_importer.get_db_session"
        ) as mock_get_session:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_get_session.return_value = mock_context_manager

            with pytest.raises(InsufficientTrainingDataError) as exc_info:
                await importer.validate_data_sufficiency("test_room")

            assert exc_info.value.room_id == "test_room"
            assert exc_info.value.data_points == 40  # 10 days * 4 events/day avg
            assert exc_info.value.time_span_days == 10

    @pytest.mark.asyncio
    async def test_validate_data_sufficiency_no_data(self, test_system_config):
        """Test data sufficiency validation with no data."""
        importer = BulkImporter(test_system_config)

        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.fetchall.return_value = []  # No data
        mock_session.execute.return_value = mock_result

        with patch(
            "src.data.ingestion.bulk_importer.get_db_session"
        ) as mock_get_session:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_get_session.return_value = mock_context_manager

            with pytest.raises(InsufficientTrainingDataError) as exc_info:
                await importer.validate_data_sufficiency("test_room")

            assert exc_info.value.room_id == "test_room"
            assert exc_info.value.data_points == 0
            assert exc_info.value.time_span_days == 0

    @pytest.mark.asyncio
    async def test_optimize_import_performance(self, test_system_config):
        """Test import performance optimization analysis."""
        importer = BulkImporter(test_system_config)

        # Set up performance metrics
        importer.progress.processed_events = 1000
        importer.progress.start_time = datetime.utcnow() - timedelta(seconds=100)

        with patch("psutil.Process") as mock_process_class:
            mock_process = Mock()
            mock_process.memory_info.return_value.rss = 600 * 1024 * 1024  # 600MB
            mock_process_class.return_value = mock_process

            result = await importer.optimize_import_performance()

            assert "current_settings" in result
            assert "performance_metrics" in result
            assert "optimization_suggestions" in result

            # Should suggest memory optimization for high usage
            suggestions = result["optimization_suggestions"]
            assert any("memory usage" in suggestion for suggestion in suggestions)

    @pytest.mark.asyncio
    async def test_verify_import_integrity(self, test_system_config):
        """Test import integrity verification."""
        importer = BulkImporter(test_system_config)

        # Mock database query for integrity check
        mock_session = AsyncMock(spec=AsyncSession)
        mock_result = Mock()
        mock_result.__iter__ = Mock(
            return_value=iter(
                [
                    Mock(
                        room_id="test_room",
                        total_events=1000,
                        future_timestamps=10,
                        missing_states=5,
                    )
                ]
            )
        )
        mock_session.execute.return_value = mock_result

        with patch(
            "src.data.ingestion.bulk_importer.get_db_session"
        ) as mock_get_session:
            mock_context_manager = AsyncMock()
            mock_context_manager.__aenter__.return_value = mock_session
            mock_context_manager.__aexit__.return_value = None
            mock_get_session.return_value = mock_context_manager

            result = await importer.verify_import_integrity(sample_percentage=1.0)

            assert "verification_timestamp" in result
            assert "sample_percentage" in result
            assert "checks_performed" in result
            assert "issues_found" in result
            assert "overall_integrity_score" in result

            # Should have found issues
            assert len(result["issues_found"]) > 0
            assert result["overall_integrity_score"] == 0.985  # (1000-15)/1000

    @pytest.mark.asyncio
    async def test_create_import_checkpoint(self, test_system_config):
        """Test creating import checkpoint."""
        importer = BulkImporter(test_system_config)

        # Set up some state
        importer.progress.total_entities = 10
        importer._completed_entities = {"entity1", "entity2"}
        importer.stats["events_imported"] = 100

        with tempfile.TemporaryDirectory() as temp_dir:
            original_cwd = Path.cwd()
            try:
                # Change to temp directory for checkpoint creation
                import os

                os.chdir(temp_dir)

                result = await importer.create_import_checkpoint("test_checkpoint")

                assert result is True

                # Verify checkpoint file was created
                checkpoint_files = list(
                    Path(temp_dir).glob("checkpoint_test_checkpoint_*.json")
                )
                assert len(checkpoint_files) == 1

                # Verify checkpoint content
                with open(checkpoint_files[0]) as f:
                    checkpoint_data = json.load(f)

                assert checkpoint_data["checkpoint_name"] == "test_checkpoint"
                assert checkpoint_data["progress"]["total_entities"] == 10
                assert set(checkpoint_data["completed_entities"]) == {
                    "entity1",
                    "entity2",
                }
                assert checkpoint_data["stats"]["events_imported"] == 100
            finally:
                os.chdir(original_cwd)

    @pytest.mark.asyncio
    async def test_import_historical_data_complete_flow(self, test_system_config):
        """Test complete import flow with mocked components."""
        import_config = ImportConfig(months_to_import=1, batch_size=10)
        importer = BulkImporter(test_system_config, import_config)

        # Mock all dependencies
        mock_ha_client = AsyncMock()
        mock_processor = AsyncMock()

        # Mock configuration
        test_system_config.get_all_entity_ids = Mock(
            return_value=["entity1", "entity2"]
        )

        with (
            patch.object(importer, "_initialize_components") as mock_init,
            patch.object(importer, "_cleanup_components") as mock_cleanup,
            patch.object(importer, "_load_resume_data") as mock_load_resume,
            patch.object(importer, "_save_resume_data") as mock_save_resume,
            patch.object(importer, "_estimate_total_events") as mock_estimate,
            patch.object(importer, "_process_entities_batch") as mock_process,
            patch.object(importer, "_generate_import_report") as mock_report,
        ):

            # Mock successful initialization
            mock_init.side_effect = lambda: setattr(
                importer, "ha_client", mock_ha_client
            ) or setattr(importer, "event_processor", mock_processor)

            start_date = datetime(2023, 1, 1, tzinfo=timezone.utc)
            end_date = datetime(2023, 1, 31, tzinfo=timezone.utc)

            result = await importer.import_historical_data(start_date, end_date)

            # Verify complete flow execution
            mock_init.assert_called_once()
            mock_cleanup.assert_called_once()
            mock_load_resume.assert_called_once()
            mock_save_resume.assert_called_once()
            mock_estimate.assert_called_once()
            mock_process.assert_called_once()
            mock_report.assert_called_once()

            # Verify result is ImportProgress
            assert isinstance(result, ImportProgress)
            assert result.total_entities == 2

    @pytest.mark.asyncio
    async def test_import_historical_data_with_error(self, test_system_config):
        """Test import flow with error handling."""
        importer = BulkImporter(test_system_config)

        with (
            patch.object(importer, "_initialize_components"),
            patch.object(importer, "_cleanup_components") as mock_cleanup,
            patch.object(importer, "_save_resume_data") as mock_save_resume,
            patch.object(importer, "_load_resume_data"),
            patch.object(
                importer, "_estimate_total_events", side_effect=Exception("Test error")
            ),
        ):

            with pytest.raises(Exception, match="Test error"):
                await importer.import_historical_data()

            # Should still call cleanup and save resume data
            mock_cleanup.assert_called_once()
            mock_save_resume.assert_called_once()

            # Should have logged error
            assert len(importer.progress.errors) > 0

    def test_generate_sufficiency_recommendation(self, test_system_config):
        """Test sufficiency recommendation generation."""
        importer = BulkImporter(test_system_config)

        # Test sufficient data
        recommendation = importer._generate_sufficiency_recommendation(
            sufficient_days=True,
            sufficient_events=True,
            total_days=45,
            avg_events_per_day=25.0,
        )
        assert "sufficient for model training" in recommendation

        # Test insufficient days
        recommendation = importer._generate_sufficiency_recommendation(
            sufficient_days=False,
            sufficient_events=True,
            total_days=15,
            avg_events_per_day=25.0,
        )
        assert "Need more historical data" in recommendation
        assert "15 days" in recommendation

        # Test insufficient events
        recommendation = importer._generate_sufficiency_recommendation(
            sufficient_days=True,
            sufficient_events=False,
            total_days=45,
            avg_events_per_day=5.0,
        )
        assert "Low event frequency" in recommendation
        assert "5.0 events/day" in recommendation


@pytest.mark.integration
class TestBulkImporterIntegration:
    """Integration tests for BulkImporter with real components."""

    @pytest.mark.asyncio
    async def test_integration_with_real_event_processor(self, test_system_config):
        """Test BulkImporter integration with real EventProcessor."""
        import_config = ImportConfig(validate_events=True, batch_size=5)
        importer = BulkImporter(test_system_config, import_config)

        # Create real event processor
        from src.data.ingestion.event_processor import EventProcessor

        importer.event_processor = EventProcessor(test_system_config)

        # Create test HA events
        ha_events = [
            HAEvent(
                entity_id="binary_sensor.test_room_presence",
                state="on",
                previous_state="off",
                timestamp=datetime(2023, 1, 1, 12, 0, tzinfo=timezone.utc),
                attributes={"device_class": "motion"},
            ),
            HAEvent(
                entity_id="binary_sensor.test_room_presence",
                state="off",
                previous_state="on",
                timestamp=datetime(2023, 1, 1, 12, 5, tzinfo=timezone.utc),
                attributes={"device_class": "motion"},
            ),
        ]

        # Test conversion and processing
        with patch.object(importer, "_bulk_insert_events") as mock_insert:
            mock_insert.return_value = 2

            # Simulate processing history chunk
            history_data = [
                {
                    "entity_id": "binary_sensor.test_room_presence",
                    "state": "on",
                    "last_changed": "2023-01-01T12:00:00Z",
                    "attributes": {"device_class": "motion"},
                },
                {
                    "entity_id": "binary_sensor.test_room_presence",
                    "state": "off",
                    "last_changed": "2023-01-01T12:05:00Z",
                    "attributes": {"device_class": "motion"},
                },
            ]

            result = await importer._process_history_chunk("test_entity", history_data)

            # Should have processed events successfully
            assert result == 2
            mock_insert.assert_called_once()

            # Verify sensor events were created correctly
            inserted_events = mock_insert.call_args[0][0]
            assert len(inserted_events) == 2
            assert all(isinstance(event, SensorEvent) for event in inserted_events)
            assert inserted_events[0].room_id == "test_room"
            assert inserted_events[0].sensor_type == "presence"
