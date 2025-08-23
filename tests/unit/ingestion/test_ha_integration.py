"""Unit tests for Home Assistant integration and data ingestion.

Covers:
- src/data/ingestion/ha_client.py (Home Assistant Client)
- src/data/ingestion/event_processor.py (Event Processing Pipeline)  
- src/data/ingestion/bulk_importer.py (Historical Data Import)

This test file provides comprehensive testing for all Home Assistant data ingestion functionality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest
import websockets
from websockets.exceptions import ConnectionClosed, InvalidStatusCode, InvalidURI

from src.core.config import HomeAssistantConfig, RoomConfig, SystemConfig
from src.core.constants import SensorState, SensorType
from src.core.exceptions import (
    HomeAssistantAPIError,
    HomeAssistantAuthenticationError,
    HomeAssistantConnectionError,
    RateLimitExceededError,
    WebSocketError,
)
from src.data.ingestion.bulk_importer import BulkImporter, ImportConfig, ImportProgress
from src.data.ingestion.event_processor import (
    ClassificationResult,
    EventProcessor,
    EventValidator,
    MovementPatternClassifier,
    MovementSequence,
    ValidationResult,
)
from src.data.ingestion.ha_client import HAEvent, HomeAssistantClient, RateLimiter
from src.data.storage.models import SensorEvent


@pytest.fixture
def mock_ha_config():
    """Create mock Home Assistant configuration."""
    return HomeAssistantConfig(
        url="http://homeassistant.local:8123",
        token="test_token_12345",
        websocket_timeout=30,
        api_timeout=10,
    )


@pytest.fixture
def mock_room_config():
    """Create mock room configuration."""
    return RoomConfig(
        room_id="living_room",
        name="Living Room",
        sensors={
            "motion": {"main": "binary_sensor.living_room_motion"},
            "door": {"main": "binary_sensor.living_room_door"},
            "presence": {"main": "binary_sensor.living_room_presence"},
        },
    )


@pytest.fixture
def mock_system_config(mock_ha_config, mock_room_config):
    """Create mock system configuration."""
    config = MagicMock(spec=SystemConfig)
    config.home_assistant = mock_ha_config
    config.rooms = {"living_room": mock_room_config}
    config.get_room_by_entity_id.return_value = mock_room_config
    config.get_all_entity_ids.return_value = [
        "binary_sensor.living_room_motion",
        "binary_sensor.living_room_door",
        "binary_sensor.living_room_presence",
    ]
    return config


@pytest.fixture
def sample_ha_event():
    """Create sample Home Assistant event."""
    return HAEvent(
        entity_id="binary_sensor.living_room_motion",
        state="on",
        previous_state="off",
        timestamp=datetime.now(timezone.utc),
        attributes={"device_class": "motion"},
    )


@pytest.fixture
def sample_sensor_event():
    """Create sample sensor event."""
    return SensorEvent(
        room_id="living_room",
        sensor_id="binary_sensor.living_room_motion",
        sensor_type="motion",
        state="on",
        previous_state="off",
        timestamp=datetime.now(timezone.utc),
        attributes={"device_class": "motion"},
        is_human_triggered=True,
        created_at=datetime.now(timezone.utc),
    )


class AsyncContextManagerMock:
    """Mock async context manager for aiohttp responses."""

    def __init__(self, return_value):
        self.return_value = return_value

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class TestHAEvent:
    """Test HAEvent data class functionality."""

    def test_ha_event_creation(self):
        """Test HAEvent creation with all fields."""
        timestamp = datetime.now(timezone.utc)
        event = HAEvent(
            entity_id="sensor.test",
            state="on",
            previous_state="off",
            timestamp=timestamp,
            attributes={"test": "value"},
            event_type="state_changed",
        )

        assert event.entity_id == "sensor.test"
        assert event.state == "on"
        assert event.previous_state == "off"
        assert event.timestamp == timestamp
        assert event.attributes == {"test": "value"}
        assert event.event_type == "state_changed"

    def test_ha_event_is_valid(self):
        """Test HAEvent validation logic."""
        # Valid event
        valid_event = HAEvent(
            entity_id="sensor.test",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )
        assert valid_event.is_valid() is True

        # Invalid state
        invalid_event = HAEvent(
            entity_id="sensor.test",
            state="unknown",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )
        assert invalid_event.is_valid() is False

        # Missing entity_id
        missing_entity = HAEvent(
            entity_id="",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )
        assert missing_entity.is_valid() is False


class TestRateLimiter:
    """Test rate limiting functionality."""

    @pytest.mark.asyncio
    async def test_rate_limiter_initialization(self):
        """Test rate limiter initialization."""
        limiter = RateLimiter(max_requests=100, window_seconds=60)
        assert limiter.max_requests == 100
        assert limiter.window_seconds == 60
        assert len(limiter.requests) == 0

    @pytest.mark.asyncio
    async def test_rate_limiter_allow_requests(self):
        """Test rate limiter allows requests within limits."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Should allow first 5 requests
        for i in range(5):
            await limiter.acquire()
            assert len(limiter.requests) == i + 1

    @pytest.mark.asyncio
    async def test_rate_limiter_blocks_excess_requests(self):
        """Test rate limiter blocks requests exceeding limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)

        # Fill up the rate limiter
        await limiter.acquire()
        await limiter.acquire()

        # Mock sleep to speed up the test
        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            # This should trigger waiting
            await limiter.acquire()
            mock_sleep.assert_called_once()


class TestHomeAssistantClient:
    """Test Home Assistant client functionality."""

    @pytest.mark.asyncio
    async def test_client_initialization(self, mock_system_config):
        """Test Home Assistant client initialization."""
        client = HomeAssistantClient(mock_system_config)
        assert client.config == mock_system_config
        assert client.ha_config == mock_system_config.home_assistant
        assert client.session is None
        assert client.websocket is None
        assert client._connected is False

    @pytest.mark.asyncio
    async def test_client_connect_success(self, mock_system_config):
        """Test successful Home Assistant client connection."""
        with patch("aiohttp.ClientSession") as mock_session_class, patch(
            "websockets.connect"
        ) as mock_ws_connect:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_ws = AsyncMock()
            
            # Make websockets.connect awaitable
            async def mock_connect(*args, **kwargs):
                return mock_ws
            mock_ws_connect.side_effect = mock_connect

            # Mock authentication test
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value = AsyncContextManagerMock(mock_response)

            # Mock WebSocket authentication
            mock_ws.recv.side_effect = [
                json.dumps({"type": "auth_required"}),
                json.dumps({"type": "auth_ok"}),
            ]

            client = HomeAssistantClient(mock_system_config)

            with patch.object(
                client, "_handle_websocket_messages", new_callable=AsyncMock
            ):
                await client.connect()

            assert client._connected is True
            assert client.session == mock_session
            assert client.websocket == mock_ws

    @pytest.mark.asyncio
    async def test_client_connect_auth_failure(self, mock_system_config):
        """Test Home Assistant client connection with authentication failure."""
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            # Mock authentication failure
            mock_response = AsyncMock()
            mock_response.status = 401
            mock_session.get.return_value.__aenter__.return_value = mock_response

            client = HomeAssistantClient(mock_system_config)

            with pytest.raises(HomeAssistantConnectionError):
                await client.connect()

    @pytest.mark.asyncio
    async def test_websocket_authentication(self, mock_system_config):
        """Test WebSocket authentication process."""
        client = HomeAssistantClient(mock_system_config)
        mock_ws = AsyncMock()
        client.websocket = mock_ws

        # Mock successful authentication flow
        mock_ws.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_ok"}),
        ]

        await client._authenticate_websocket()

        # Verify auth message was sent
        mock_ws.send.assert_called_once()
        auth_call = mock_ws.send.call_args[0][0]
        auth_data = json.loads(auth_call)
        assert auth_data["type"] == "auth"
        assert auth_data["access_token"] == "test_token_12345"

    @pytest.mark.asyncio
    async def test_websocket_authentication_failure(self, mock_system_config):
        """Test WebSocket authentication failure."""
        client = HomeAssistantClient(mock_system_config)
        mock_ws = AsyncMock()
        client.websocket = mock_ws

        # Mock authentication failure
        mock_ws.recv.side_effect = [
            json.dumps({"type": "auth_required"}),
            json.dumps({"type": "auth_invalid"}),
        ]

        with pytest.raises(HomeAssistantAuthenticationError):
            await client._authenticate_websocket()

    @pytest.mark.asyncio
    async def test_get_entity_state_success(self, mock_system_config):
        """Test successful entity state retrieval."""
        client = HomeAssistantClient(mock_system_config)
        
        # Mock successful API response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={
            "state": "on",
            "attributes": {"device_class": "motion"},
        })
        
        # Mock session with proper context manager
        mock_session = Mock()
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)
        client.session = mock_session

        result = await client.get_entity_state("sensor.test")

        assert result["state"] == "on"
        assert result["attributes"]["device_class"] == "motion"

    @pytest.mark.asyncio
    async def test_get_entity_state_not_found(self, mock_system_config):
        """Test entity state retrieval with 404 response."""
        client = HomeAssistantClient(mock_system_config)
        
        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        
        # Mock session with proper context manager
        mock_session = Mock()
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)
        client.session = mock_session

        result = await client.get_entity_state("sensor.nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_state_rate_limit(self, mock_system_config):
        """Test entity state retrieval with rate limiting."""
        client = HomeAssistantClient(mock_system_config)
        
        # Mock rate limit response
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}
        
        # Mock session with proper context manager
        mock_session = Mock()
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)
        client.session = mock_session

        with pytest.raises(RateLimitExceededError) as exc_info:
            await client.get_entity_state("sensor.test")

        assert exc_info.value.service == "home_assistant_api"
        assert exc_info.value.reset_time == 60

    @pytest.mark.asyncio
    async def test_get_entity_history_success(self, mock_system_config):
        """Test successful entity history retrieval."""
        client = HomeAssistantClient(mock_system_config)
        
        # Mock successful history response
        mock_response = AsyncMock()
        mock_response.status = 200
        history_data = [
            [
                {
                    "entity_id": "sensor.test",
                    "state": "on",
                    "last_changed": "2024-01-01T12:00:00Z",
                    "attributes": {},
                }
            ]
        ]
        mock_response.json = AsyncMock(return_value=history_data)
        
        # Mock session with proper context manager
        mock_session = Mock()
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)
        client.session = mock_session

        start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
        result = await client.get_entity_history("sensor.test", start_time)

        assert len(result) == 1
        assert result[0]["entity_id"] == "sensor.test"
        assert result[0]["state"] == "on"

    @pytest.mark.asyncio
    async def test_subscribe_to_events_success(self, mock_system_config):
        """Test successful event subscription."""
        client = HomeAssistantClient(mock_system_config)
        mock_ws = AsyncMock()
        client.websocket = mock_ws
        client._connected = True
        client._ws_message_id = 1

        # Mock the response by patching the websocket wait mechanism
        with patch('asyncio.wait_for') as mock_wait_for:
            mock_wait_for.return_value = {"success": True}
            
            await client.subscribe_to_events(["sensor.test"])

        # Verify subscription command was sent
        mock_ws.send.assert_called_once()
        command_data = json.loads(mock_ws.send.call_args[0][0])
        assert command_data["type"] == "subscribe_events"
        assert command_data["event_type"] == "state_changed"

    @pytest.mark.asyncio
    async def test_validate_entities(self, mock_system_config):
        """Test entity validation functionality."""
        client = HomeAssistantClient(mock_system_config)

        # Mock get_entity_state responses
        with patch.object(client, "get_entity_state") as mock_get_state:
            mock_get_state.side_effect = [
                {"state": "on"},  # First entity exists
                None,  # Second entity doesn't exist
                {"state": "off"},  # Third entity exists
            ]

            result = await client.validate_entities(
                ["sensor.existing1", "sensor.nonexistent", "sensor.existing2"]
            )

            assert result["sensor.existing1"] is True
            assert result["sensor.nonexistent"] is False
            assert result["sensor.existing2"] is True

    @pytest.mark.asyncio
    async def test_event_handling(self, mock_system_config):
        """Test WebSocket event handling."""
        client = HomeAssistantClient(mock_system_config)
        event_handler = AsyncMock()
        client.add_event_handler(event_handler)
        client._subscribed_entities = {"sensor.test"}

        # Mock event data
        event_data = {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "sensor.test",
                    "new_state": {
                        "state": "on",
                        "last_changed": "2024-01-01T12:00:00Z",
                        "attributes": {},
                    },
                    "old_state": {
                        "state": "off",
                        "attributes": {},
                    },
                },
            },
        }

        await client._process_websocket_message(event_data)

        # Verify event handler was called
        event_handler.assert_called_once()
        ha_event = event_handler.call_args[0][0]
        assert ha_event.entity_id == "sensor.test"
        assert ha_event.state == "on"
        assert ha_event.previous_state == "off"

    @pytest.mark.asyncio
    async def test_state_validation_and_normalization(self, mock_system_config):
        """Test state validation and normalization."""
        client = HomeAssistantClient(mock_system_config)

        # Test valid states
        assert client._validate_and_normalize_state("on") == "on"
        assert client._validate_and_normalize_state("off") == "off"
        assert client._validate_and_normalize_state("open") == "open"
        assert client._validate_and_normalize_state("closed") == "closed"

        # Test case normalization
        assert client._validate_and_normalize_state("ON") == "on"
        assert client._validate_and_normalize_state("Off") == "off"

        # Test partial matches
        assert client._validate_and_normalize_state("detected") == "on"
        assert client._validate_and_normalize_state("clear") == "off"
        assert client._validate_and_normalize_state("motion detected") == "on"

    def test_convert_ha_event_to_sensor_event(
        self, mock_system_config, sample_ha_event
    ):
        """Test conversion from HAEvent to SensorEvent."""
        client = HomeAssistantClient(mock_system_config)
        sensor_event = client.convert_ha_event_to_sensor_event(
            sample_ha_event, "living_room", "motion"
        )

        assert sensor_event.room_id == "living_room"
        assert sensor_event.sensor_id == sample_ha_event.entity_id
        assert sensor_event.sensor_type == "motion"
        assert sensor_event.state == sample_ha_event.state
        assert sensor_event.previous_state == sample_ha_event.previous_state
        assert sensor_event.timestamp == sample_ha_event.timestamp
        assert sensor_event.attributes == sample_ha_event.attributes

    def test_convert_history_to_sensor_events(self, mock_system_config):
        """Test conversion from history data to SensorEvent list."""
        client = HomeAssistantClient(mock_system_config)
        history_data = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "last_changed": "2024-01-01T12:00:00Z",
                "attributes": {"device_class": "motion"},
            },
            {
                "entity_id": "sensor.test",
                "state": "off",
                "last_changed": "2024-01-01T12:05:00Z",
                "attributes": {"device_class": "motion"},
            },
        ]

        events = client.convert_history_to_sensor_events(
            history_data, "living_room", "motion"
        )

        assert len(events) == 2
        assert events[0].state == "on"
        assert events[1].state == "off"
        assert events[1].previous_state == "on"


class TestEventValidator:
    """Test event validation functionality."""

    @pytest.fixture
    def validator(self, mock_system_config):
        """Create event validator instance."""
        return EventValidator(mock_system_config)

    def test_validator_initialization(self, validator, mock_system_config):
        """Test validator initialization."""
        assert validator.config == mock_system_config

    def test_validate_valid_event(self, validator, sample_sensor_event):
        """Test validation of a valid event."""
        result = validator.validate_event(sample_sensor_event)
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert result.confidence_score > 0.8

    def test_validate_missing_required_fields(self, validator):
        """Test validation with missing required fields."""
        invalid_event = SensorEvent(
            room_id="",  # Missing
            sensor_id="",  # Missing
            sensor_type="motion",
            state="",  # Missing
            timestamp=None,  # Missing
        )

        result = validator.validate_event(invalid_event)
        assert result.is_valid is False
        assert "Missing room_id" in result.errors
        assert "Missing sensor_id" in result.errors
        assert "Missing state" in result.errors
        assert "Missing timestamp" in result.errors

    def test_validate_invalid_state(self, validator, sample_sensor_event):
        """Test validation with invalid state."""
        sample_sensor_event.state = "unknown"
        result = validator.validate_event(sample_sensor_event)
        assert result.is_valid is False
        assert "Invalid state: unknown" in result.errors

    def test_validate_future_timestamp(self, validator, sample_sensor_event):
        """Test validation with future timestamp."""
        sample_sensor_event.timestamp = datetime.now(timezone.utc) + timedelta(
            minutes=10
        )
        result = validator.validate_event(sample_sensor_event)
        assert result.is_valid is True  # Only a warning, not an error
        assert "Event timestamp is in the future" in result.warnings
        assert result.confidence_score < 1.0

    def test_validate_old_timestamp(self, validator, sample_sensor_event):
        """Test validation with old timestamp."""
        sample_sensor_event.timestamp = datetime.now(timezone.utc) - timedelta(days=2)
        result = validator.validate_event(sample_sensor_event)
        assert result.is_valid is True  # Only a warning, not an error
        assert "Event timestamp is more than 24 hours old" in result.warnings
        assert result.confidence_score < 1.0

    def test_validate_no_state_change(self, validator, sample_sensor_event):
        """Test validation when state doesn't change."""
        sample_sensor_event.state = "on"
        sample_sensor_event.previous_state = "on"
        result = validator.validate_event(sample_sensor_event)
        assert result.is_valid is True  # Only a warning, not an error
        assert "State did not change from previous state" in result.warnings
        assert result.confidence_score < 1.0


class TestMovementPatternClassifier:
    """Test movement pattern classification."""

    @pytest.fixture
    def classifier(self, mock_system_config):
        """Create movement pattern classifier."""
        return MovementPatternClassifier(mock_system_config)

    @pytest.fixture
    def movement_sequence(self, sample_sensor_event):
        """Create sample movement sequence."""
        events = [sample_sensor_event]
        return MovementSequence(
            events=events,
            start_time=sample_sensor_event.timestamp,
            end_time=sample_sensor_event.timestamp + timedelta(seconds=30),
            duration_seconds=30.0,
            rooms_visited={"living_room"},
            sensors_triggered={sample_sensor_event.sensor_id},
        )

    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier.config is not None
        assert classifier.human_patterns is not None
        assert classifier.cat_patterns is not None

    def test_classify_movement(self, classifier, movement_sequence, mock_room_config):
        """Test movement classification."""
        result = classifier.classify_movement(movement_sequence, mock_room_config)
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.is_human_triggered, bool)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.classification_reason is not None
        assert result.movement_metrics is not None

    def test_movement_sequence_properties(self, movement_sequence):
        """Test movement sequence property calculations."""
        assert movement_sequence.average_velocity >= 0.0
        assert movement_sequence.trigger_pattern is not None
        assert "living_room_motion" in movement_sequence.trigger_pattern

    def test_calculate_movement_metrics(
        self, classifier, movement_sequence, mock_room_config
    ):
        """Test movement metrics calculation."""
        metrics = classifier._calculate_movement_metrics(
            movement_sequence, mock_room_config
        )

        assert "duration_seconds" in metrics
        assert "event_count" in metrics
        assert "rooms_visited" in metrics
        assert "sensors_triggered" in metrics
        assert "average_velocity" in metrics
        assert "movement_entropy" in metrics
        assert "spatial_dispersion" in metrics

        # Verify metric types and ranges
        assert metrics["duration_seconds"] == 30.0
        assert metrics["event_count"] == 1
        assert metrics["rooms_visited"] == 1
        assert metrics["sensors_triggered"] == 1

    def test_score_human_pattern(self, classifier):
        """Test human pattern scoring."""
        metrics = {
            "duration_seconds": 45.0,
            "max_velocity": 2.0,
            "door_interactions": 1,
            "event_count": 5,
            "revisit_count": 0,
            "sensors_triggered": 4,
        }
        score = classifier._score_human_pattern(metrics)
        assert 0.0 <= score <= 1.0

    def test_score_cat_pattern(self, classifier):
        """Test cat pattern scoring."""
        metrics = {
            "duration_seconds": 10.0,
            "max_velocity": 8.0,
            "door_interactions": 0,
            "event_count": 8,
            "revisit_count": 3,
            "sensors_triggered": 6,
        }
        score = classifier._score_cat_pattern(metrics)
        assert 0.0 <= score <= 1.0

    def test_generate_classification_reason(self, classifier):
        """Test classification reason generation."""
        metrics = {
            "duration_seconds": 45.0,
            "max_velocity": 2.0,
            "door_interactions": 1,
            "revisit_count": 0,
        }
        reason = classifier._generate_classification_reason(metrics, 0.8, 0.3, True)
        assert "Human pattern" in reason
        assert "door interactions observed" in reason

    def test_analyze_sequence_patterns(
        self, classifier, movement_sequence, mock_room_config
    ):
        """Test comprehensive sequence pattern analysis."""
        classification, confidence, metrics = classifier.analyze_sequence_patterns(
            movement_sequence, mock_room_config
        )

        assert classification in ["human", "cat"]
        assert 0.0 <= confidence <= 1.0
        assert "statistical_confidence" in metrics
        assert "pattern_consistency" in metrics
        assert "anomaly_score" in metrics


class TestEventProcessor:
    """Test event processing functionality."""

    @pytest.fixture
    def processor(self, mock_system_config):
        """Create event processor."""
        return EventProcessor(mock_system_config)

    def test_processor_initialization(self, processor):
        """Test processor initialization."""
        assert processor.config is not None
        assert processor.validator is not None
        assert processor.classifier is not None
        assert processor.stats["total_processed"] == 0

    @pytest.mark.asyncio
    async def test_process_event_success(self, processor, sample_ha_event):
        """Test successful event processing."""
        result = await processor.process_event(sample_ha_event)
        assert result is not None
        assert isinstance(result, SensorEvent)
        assert result.room_id == "living_room"
        assert result.sensor_id == sample_ha_event.entity_id
        assert processor.stats["valid_events"] == 1

    @pytest.mark.asyncio
    async def test_process_event_unknown_entity(self, processor, mock_system_config):
        """Test processing event for unknown entity."""
        # Mock get_room_by_entity_id to return None
        mock_system_config.get_room_by_entity_id.return_value = None

        unknown_event = HAEvent(
            entity_id="sensor.unknown",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )

        result = await processor.process_event(unknown_event)
        assert result is None
        assert processor.stats["total_processed"] == 1
        assert processor.stats["valid_events"] == 0

    @pytest.mark.asyncio
    async def test_process_event_batch(self, processor, sample_ha_event):
        """Test batch event processing."""
        events = [sample_ha_event] * 5
        results = await processor.process_event_batch(events)

        assert len(results) <= 5  # Some might be filtered as duplicates
        assert all(isinstance(event, SensorEvent) for event in results)
        assert processor.stats["total_processed"] == 5

    def test_determine_sensor_type(self, processor, mock_room_config):
        """Test sensor type determination."""
        # Test configured sensor
        sensor_type = processor._determine_sensor_type(
            "binary_sensor.living_room_motion", mock_room_config
        )
        assert sensor_type == "motion"

        # Test fallback for unknown sensor
        sensor_type = processor._determine_sensor_type(
            "sensor.unknown_motion", mock_room_config
        )
        assert sensor_type == SensorType.PRESENCE.value

    def test_duplicate_detection(self, processor, sample_sensor_event):
        """Test duplicate event detection."""
        # First event should not be duplicate
        is_duplicate = processor._is_duplicate_event(sample_sensor_event)
        assert is_duplicate is False

        # Update tracking
        processor._update_event_tracking(sample_sensor_event)

        # Immediate duplicate should be detected
        duplicate_event = SensorEvent(
            room_id=sample_sensor_event.room_id,
            sensor_id=sample_sensor_event.sensor_id,
            sensor_type=sample_sensor_event.sensor_type,
            state="off",
            timestamp=sample_sensor_event.timestamp
            + timedelta(seconds=1),  # Very close
        )
        is_duplicate = processor._is_duplicate_event(duplicate_event)
        assert is_duplicate is True

    @pytest.mark.asyncio
    async def test_validate_event_sequence_integrity(
        self, processor, sample_sensor_event
    ):
        """Test event sequence integrity validation."""
        events = [sample_sensor_event] * 3
        # Modify timestamps to create sequence
        for i, event in enumerate(events):
            event.timestamp = sample_sensor_event.timestamp + timedelta(seconds=i * 10)

        result = await processor.validate_event_sequence_integrity(events)
        assert result["valid"] is True
        assert "analysis" in result
        assert result["confidence"] > 0.0

    @pytest.mark.asyncio
    async def test_validate_room_configuration(self, processor):
        """Test room configuration validation."""
        result = await processor.validate_room_configuration("living_room")
        assert result["valid"] is True
        assert result["entity_count"] > 0
        assert "motion" in result["sensor_types"]

        # Test unknown room
        result = await processor.validate_room_configuration("unknown_room")
        assert result["valid"] is False
        assert "not found in configuration" in result["error"]

    def test_get_processing_stats(self, processor):
        """Test processing statistics retrieval."""
        stats = processor.get_processing_stats()
        assert "total_processed" in stats
        assert "valid_events" in stats
        assert "invalid_events" in stats
        assert "human_classified" in stats
        assert "cat_classified" in stats
        assert "duplicates_filtered" in stats

    def test_reset_stats(self, processor):
        """Test statistics reset."""
        processor.stats["total_processed"] = 100
        processor.reset_stats()
        assert processor.stats["total_processed"] == 0


class TestBulkImporter:
    """Test bulk historical data import functionality."""

    @pytest.fixture
    def import_config(self):
        """Create import configuration."""
        return ImportConfig(
            months_to_import=1,
            batch_size=100,
            entity_batch_size=5,
            max_concurrent_entities=2,
            chunk_days=7,
            skip_existing=True,
            validate_events=True,
        )

    @pytest.fixture
    def importer(self, mock_system_config, import_config):
        """Create bulk importer."""
        return BulkImporter(mock_system_config, import_config)

    def test_importer_initialization(self, importer, mock_system_config, import_config):
        """Test bulk importer initialization."""
        assert importer.config == mock_system_config
        assert importer.import_config == import_config
        assert importer.progress.total_entities == 0
        assert importer.stats["entities_processed"] == 0

    def test_import_progress(self):
        """Test import progress tracking."""
        progress = ImportProgress(
            total_entities=100,
            processed_entities=50,
            total_events=1000,
            processed_events=600,
            valid_events=580,
        )

        assert progress.entity_progress_percent == 50.0
        assert progress.event_progress_percent == 60.0
        assert progress.events_per_second >= 0.0

        # Test dictionary conversion
        progress_dict = progress.to_dict()
        assert progress_dict["total_entities"] == 100
        assert progress_dict["entity_progress_percent"] == 50.0

    @pytest.mark.asyncio
    async def test_initialize_components(self, mock_system_config, import_config):
        """Test component initialization."""
        with patch(
            "src.data.ingestion.bulk_importer.HomeAssistantClient"
        ) as mock_client_class, patch(
            "src.data.ingestion.bulk_importer.EventProcessor"
        ) as mock_processor_class:
            mock_client = AsyncMock()
            mock_client_class.return_value = mock_client
            mock_processor_class.return_value = Mock()
            
            # Create importer inside the patch context
            importer = BulkImporter(mock_system_config, import_config)
            await importer._initialize_components()

            assert importer.ha_client == mock_client
            assert importer.event_processor is not None
            mock_client.connect.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_components(self, importer):
        """Test component cleanup."""
        mock_client = AsyncMock()
        importer.ha_client = mock_client

        await importer._cleanup_components()
        mock_client.disconnect.assert_called_once()

    def test_convert_history_record_to_ha_event(self, importer):
        """Test history record conversion."""
        record = {
            "entity_id": "sensor.test",
            "state": "on",
            "last_changed": "2024-01-01T12:00:00Z",
            "attributes": {"device_class": "motion"},
        }

        ha_event = importer._convert_history_record_to_ha_event(record)
        assert ha_event is not None
        assert ha_event.entity_id == "sensor.test"
        assert ha_event.state == "on"
        assert ha_event.previous_state is None
        assert ha_event.attributes == {"device_class": "motion"}

    def test_convert_history_record_invalid(self, importer):
        """Test conversion of invalid history record."""
        invalid_record = {"entity_id": "sensor.test"}
        ha_event = importer._convert_history_record_to_ha_event(invalid_record)
        assert ha_event is None

    def test_determine_sensor_type(self, importer, mock_room_config):
        """Test sensor type determination from entity ID."""
        # Test configured sensor
        sensor_type = importer._determine_sensor_type(
            "binary_sensor.living_room_motion", mock_room_config
        )
        assert sensor_type == "motion"

        # Test fallback logic
        sensor_type = importer._determine_sensor_type(
            "sensor.presence_test", mock_room_config
        )
        assert sensor_type == "presence"

        sensor_type = importer._determine_sensor_type(
            "sensor.door_test", mock_room_config
        )
        assert sensor_type == "door"

        sensor_type = importer._determine_sensor_type(
            "sensor.temperature_test", mock_room_config
        )
        assert sensor_type == "climate"

    def test_get_import_stats(self, importer):
        """Test import statistics retrieval."""
        stats = importer.get_import_stats()
        assert "progress" in stats
        assert "stats" in stats
        assert stats["stats"]["entities_processed"] == 0

    @pytest.mark.asyncio
    async def test_optimize_import_performance(self, importer):
        """Test import performance optimization."""
        # Set some progress to test optimization logic
        importer.progress.processed_events = 1000
        importer.progress.start_time = datetime.utcnow() - timedelta(seconds=100)

        optimization_report = await importer.optimize_import_performance()
        assert "current_settings" in optimization_report
        assert "performance_metrics" in optimization_report
        assert "optimization_suggestions" in optimization_report

        # Check that suggestions are provided for low performance
        if importer.progress.events_per_second < 50:
            suggestions = optimization_report["optimization_suggestions"]
            assert any("batch_size" in suggestion for suggestion in suggestions)

    @pytest.mark.asyncio
    async def test_create_import_checkpoint(self, importer):
        """Test checkpoint creation."""
        with patch("builtins.open", create=True) as mock_open:
            mock_file = MagicMock()
            mock_open.return_value.__enter__.return_value = mock_file

            result = await importer.create_import_checkpoint("test_checkpoint")
            assert result is True
            mock_open.assert_called_once()
            mock_file.write.assert_called()


class TestHomeAssistantIntegration:
    """Test end-to-end Home Assistant integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_event_processing_pipeline(self, mock_system_config):
        """Test complete event processing from HA event to database."""
        # Create components
        ha_client = HomeAssistantClient(mock_system_config)
        processor = EventProcessor(mock_system_config)

        # Mock HA event
        ha_event = HAEvent(
            entity_id="binary_sensor.living_room_motion",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={"device_class": "motion"},
        )

        # Process event
        sensor_event = await processor.process_event(ha_event)

        assert sensor_event is not None
        assert sensor_event.room_id == "living_room"
        assert sensor_event.sensor_type == "motion"
        assert isinstance(sensor_event.is_human_triggered, bool)

    @pytest.mark.asyncio
    async def test_websocket_connection_lifecycle(self, mock_system_config):
        """Test WebSocket connection lifecycle management."""
        with patch("aiohttp.ClientSession") as mock_session_class, patch(
            "websockets.connect"
        ) as mock_ws_connect:
            mock_session = Mock()
            mock_session_class.return_value = mock_session
            mock_ws = AsyncMock()
            
            # Make websockets.connect awaitable
            async def mock_connect(*args, **kwargs):
                return mock_ws
            mock_ws_connect.side_effect = mock_connect

            # Mock successful authentication
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_session.get.return_value = AsyncContextManagerMock(mock_response)

            mock_ws.recv.side_effect = [
                json.dumps({"type": "auth_required"}),
                json.dumps({"type": "auth_ok"}),
            ]

            client = HomeAssistantClient(mock_system_config)

            # Test connection
            with patch.object(
                client, "_handle_websocket_messages", new_callable=AsyncMock
            ):
                await client.connect()
                assert client.is_connected is True

                # Test disconnection
                await client.disconnect()
                assert client._connected is False

    @pytest.mark.asyncio
    async def test_bulk_import_workflow(self, mock_system_config):
        """Test bulk import workflow with mocked components."""
        import_config = ImportConfig(
            months_to_import=1,
            batch_size=50,
            entity_batch_size=2,
            validate_events=False,  # Skip validation for faster testing
        )
        importer = BulkImporter(mock_system_config, import_config)

        # Mock history data
        mock_history_data = [
            {
                "entity_id": "sensor.test",
                "state": "on",
                "last_changed": "2024-01-01T12:00:00Z",
                "attributes": {},
            }
        ]

        with patch.object(
            importer, "_initialize_components", new_callable=AsyncMock
        ), patch.object(
            importer, "_cleanup_components", new_callable=AsyncMock
        ), patch.object(
            importer, "_estimate_total_events", new_callable=AsyncMock
        ), patch.object(
            importer, "_process_entities_batch", new_callable=AsyncMock
        ) as mock_process, patch.object(
            importer, "_generate_import_report", new_callable=AsyncMock
        ), patch.object(
            importer, "_load_resume_data", new_callable=AsyncMock
        ), patch.object(
            importer, "_save_resume_data", new_callable=AsyncMock
        ):
            progress = await importer.import_historical_data(
                entity_ids=["sensor.test1", "sensor.test2"]
            )

            assert progress is not None
            assert progress.total_entities == 2
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, mock_system_config):
        """Test error handling and recovery mechanisms."""
        client = HomeAssistantClient(mock_system_config)

        # Test connection error handling
        with patch("aiohttp.ClientSession") as mock_session_class:
            mock_session_class.side_effect = aiohttp.ClientError("Connection failed")

            with pytest.raises(HomeAssistantConnectionError):
                await client.connect()

        # Test rate limit handling
        client = HomeAssistantClient(mock_system_config)
        mock_response = AsyncMock()
        mock_response.status = 429
        mock_response.headers = {"Retry-After": "60"}
        
        mock_session = Mock()
        mock_session.get.return_value = AsyncContextManagerMock(mock_response)
        client.session = mock_session

        with pytest.raises(RateLimitExceededError) as exc_info:
            await client.get_entity_state("sensor.test")

        assert exc_info.value.reset_time == 60

    def test_data_validation_and_classification(
        self, mock_system_config, mock_room_config
    ):
        """Test comprehensive data validation and movement classification."""
        processor = EventProcessor(mock_system_config)
        classifier = MovementPatternClassifier(mock_system_config)

        # Create test events for sequence
        base_time = datetime.now(timezone.utc)
        events = []
        for i in range(5):
            event = SensorEvent(
                room_id="living_room",
                sensor_id=f"sensor.test_{i}",
                sensor_type="motion",
                state="on",
                timestamp=base_time + timedelta(seconds=i * 5),
            )
            events.append(event)

        # Create movement sequence
        sequence = MovementSequence(
            events=events,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            duration_seconds=20.0,
            rooms_visited={"living_room"},
            sensors_triggered={e.sensor_id for e in events},
        )

        # Test classification
        result = classifier.classify_movement(sequence, mock_room_config)
        assert isinstance(result, ClassificationResult)
        assert 0.0 <= result.confidence_score <= 1.0
        assert result.classification_reason is not None
