"""
Unit tests for Home Assistant client.

Tests WebSocket and REST API integration, authentication, event handling,
rate limiting, and error handling for the HomeAssistantClient class.
"""

import asyncio
from datetime import datetime, timedelta
import json
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from aiohttp import ClientError, ClientTimeout
import pytest
import pytest_asyncio
from websockets.exceptions import ConnectionClosed, InvalidStatusCode

from src.core.config import HomeAssistantConfig, SystemConfig
from src.core.exceptions import (
    EntityNotFoundError,
    HomeAssistantAPIError,
    HomeAssistantAuthenticationError,
    HomeAssistantConnectionError,
    WebSocketError,
)
from src.data.ingestion.ha_client import HAEvent, HomeAssistantClient, RateLimiter
from src.data.storage.models import SensorEvent


class TestHAEvent:
    """Test HAEvent dataclass."""

    def test_ha_event_creation(self):
        """Test creating HAEvent with all fields."""
        timestamp = datetime.utcnow()
        attributes = {"device_class": "motion", "friendly_name": "Test Sensor"}

        event = HAEvent(
            entity_id="binary_sensor.test_motion",
            state="on",
            previous_state="off",
            timestamp=timestamp,
            attributes=attributes,
            event_type="state_changed",
        )

        assert event.entity_id == "binary_sensor.test_motion"
        assert event.state == "on"
        assert event.previous_state == "off"
        assert event.timestamp == timestamp
        assert event.attributes == attributes
        assert event.event_type == "state_changed"

    def test_ha_event_minimal(self):
        """Test creating HAEvent with minimal fields."""
        event = HAEvent(
            entity_id="sensor.temperature",
            state="22.5",
            previous_state=None,
            timestamp=datetime.utcnow(),
            attributes={},
        )

        assert event.entity_id == "sensor.temperature"
        assert event.state == "22.5"
        assert event.previous_state is None
        assert event.event_type == "state_changed"  # Default value

    def test_ha_event_is_valid_true(self):
        """Test HAEvent validation for valid events."""
        event = HAEvent(
            entity_id="binary_sensor.valid",
            state="on",
            previous_state="off",
            timestamp=datetime.utcnow(),
            attributes={},
        )

        assert event.is_valid() is True

    def test_ha_event_is_valid_false_invalid_state(self):
        """Test HAEvent validation for invalid states."""
        event = HAEvent(
            entity_id="binary_sensor.test",
            state="unavailable",  # Invalid state
            previous_state="on",
            timestamp=datetime.utcnow(),
            attributes={},
        )

        assert event.is_valid() is False

    def test_ha_event_is_valid_false_missing_entity_id(self):
        """Test HAEvent validation for missing entity ID."""
        event = HAEvent(
            entity_id="",  # Empty entity ID
            state="on",
            previous_state="off",
            timestamp=datetime.utcnow(),
            attributes={},
        )

        assert event.is_valid() is False

    def test_ha_event_is_valid_false_missing_timestamp(self):
        """Test HAEvent validation for missing timestamp."""
        event = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=None,  # Missing timestamp
            attributes={},
        )

        assert event.is_valid() is False


class TestRateLimiter:
    """Test RateLimiter class."""

    def test_rate_limiter_init(self):
        """Test RateLimiter initialization."""
        limiter = RateLimiter(max_requests=100, window_seconds=30)

        assert limiter.max_requests == 100
        assert limiter.window_seconds == 30
        assert limiter.requests == []

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_under_limit(self):
        """Test rate limiter when under the limit."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)

        # Should not block when under limit
        start_time = datetime.utcnow()
        await limiter.acquire()
        end_time = datetime.utcnow()

        # Should be very fast (no waiting)
        assert (end_time - start_time).total_seconds() < 0.1
        assert len(limiter.requests) == 1

    @pytest.mark.asyncio
    async def test_rate_limiter_acquire_at_limit(self):
        """Test rate limiter when at the limit."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        # Fill up the limit
        await limiter.acquire()
        await limiter.acquire()

        # This should trigger rate limiting
        with patch("asyncio.sleep") as mock_sleep:
            await limiter.acquire()
            mock_sleep.assert_called_once()
            # Should have waited some amount of time
            assert mock_sleep.call_args[0][0] > 0

    @pytest.mark.asyncio
    async def test_rate_limiter_cleanup_old_requests(self):
        """Test that old requests are cleaned up."""
        limiter = RateLimiter(max_requests=3, window_seconds=1)

        # Add old request manually
        old_time = datetime.utcnow() - timedelta(seconds=2)
        limiter.requests.append(old_time)

        # Add new request
        await limiter.acquire()

        # Old request should be cleaned up
        assert len(limiter.requests) == 1
        assert limiter.requests[0] > old_time


class TestHomeAssistantClient:
    """Test HomeAssistantClient class."""

    def test_ha_client_init(self, test_system_config):
        """Test HomeAssistantClient initialization."""
        client = HomeAssistantClient(test_system_config)

        assert client.config == test_system_config
        assert client.ha_config == test_system_config.home_assistant
        assert client.session is None
        assert client.websocket is None
        assert not client._connected
        assert client._reconnect_attempts == 0
        assert client._max_reconnect_attempts == 10
        assert client._base_reconnect_delay == 5

        # Test default rate limiter
        assert isinstance(client.rate_limiter, RateLimiter)
        assert client.rate_limiter.max_requests == 300
        assert client.rate_limiter.window_seconds == 60

    def test_ha_client_init_no_config(self):
        """Test HomeAssistantClient initialization without config."""
        with patch("src.data.ingestion.ha_client.get_config") as mock_get_config:
            mock_config = Mock()
            mock_get_config.return_value = mock_config

            client = HomeAssistantClient()

            assert client.config == mock_config
            mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_success(self, test_system_config):
        """Test successful connection to Home Assistant."""
        client = HomeAssistantClient(test_system_config)

        with patch("aiohttp.ClientSession") as mock_session_class, patch.object(
            client, "_test_authentication"
        ) as mock_auth, patch.object(client, "_connect_websocket") as mock_ws:

            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            await client.connect()

            assert client._connected is True
            assert client._reconnect_attempts == 0
            assert client.session == mock_session

            # Should set up session with correct headers and timeout
            mock_session_class.assert_called_once()
            args, kwargs = mock_session_class.call_args
            assert "timeout" in kwargs
            assert "headers" in kwargs
            assert (
                f"Bearer {test_system_config.home_assistant.token}"
                in kwargs["headers"]["Authorization"]
            )

            mock_auth.assert_called_once()
            mock_ws.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_failure_cleanup(self, test_system_config):
        """Test connection failure triggers cleanup."""
        client = HomeAssistantClient(test_system_config)

        with patch("aiohttp.ClientSession") as mock_session_class, patch.object(
            client, "_test_authentication", side_effect=Exception("Auth failed")
        ), patch.object(client, "_cleanup_connections") as mock_cleanup:

            mock_session = AsyncMock()
            mock_session_class.return_value = mock_session

            with pytest.raises(HomeAssistantConnectionError):
                await client.connect()

            mock_cleanup.assert_called_once()
            assert not client._connected

    @pytest.mark.asyncio
    async def test_connect_already_connected(self, test_system_config):
        """Test connecting when already connected."""
        client = HomeAssistantClient(test_system_config)
        client._connected = True

        with patch.object(client, "_test_authentication") as mock_auth:
            await client.connect()

            # Should return early without doing anything
            mock_auth.assert_not_called()

    @pytest.mark.asyncio
    async def test_disconnect(self, test_system_config):
        """Test disconnection from Home Assistant."""
        client = HomeAssistantClient(test_system_config)
        client._connected = True

        with patch.object(client, "_cleanup_connections") as mock_cleanup:
            await client.disconnect()

            assert not client._connected
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_connections(self, test_system_config):
        """Test connection cleanup."""
        client = HomeAssistantClient(test_system_config)

        # Mock connections
        mock_websocket = AsyncMock()
        mock_session = AsyncMock()
        client.websocket = mock_websocket
        client.session = mock_session

        await client._cleanup_connections()

        mock_websocket.close.assert_called_once()
        mock_session.close.assert_called_once()
        assert client.websocket is None
        assert client.session is None

    @pytest.mark.asyncio
    async def test_test_authentication_success(self, test_system_config):
        """Test successful authentication test."""
        client = HomeAssistantClient(test_system_config)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        # Should not raise exception
        await client._test_authentication()

        mock_session.get.assert_called_once_with(
            f"{test_system_config.home_assistant.url}/api/"
        )

    @pytest.mark.asyncio
    async def test_test_authentication_401(self, test_system_config):
        """Test authentication failure (401)."""
        client = HomeAssistantClient(test_system_config)

        mock_response = AsyncMock()
        mock_response.status = 401
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        with pytest.raises(HomeAssistantAuthenticationError):
            await client._test_authentication()

    @pytest.mark.asyncio
    async def test_test_authentication_other_error(self, test_system_config):
        """Test authentication with other HTTP error."""
        client = HomeAssistantClient(test_system_config)

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        with pytest.raises(HomeAssistantAPIError) as exc_info:
            await client._test_authentication()

        assert exc_info.value.context["status_code"] == 500

    @pytest.mark.asyncio
    async def test_test_authentication_connection_error(self, test_system_config):
        """Test authentication with connection error."""
        client = HomeAssistantClient(test_system_config)

        mock_session = AsyncMock()
        mock_session.get.side_effect = ClientError("Connection failed")
        client.session = mock_session

        with pytest.raises(HomeAssistantConnectionError):
            await client._test_authentication()

    @pytest.mark.asyncio
    async def test_connect_websocket_success(self, test_system_config):
        """Test successful WebSocket connection."""
        client = HomeAssistantClient(test_system_config)

        with patch("websockets.connect") as mock_connect, patch.object(
            client, "_authenticate_websocket"
        ) as mock_auth, patch("asyncio.create_task") as mock_create_task:

            mock_websocket = AsyncMock()
            mock_connect.return_value = mock_websocket

            await client._connect_websocket()

            # Should create WebSocket connection
            expected_url = (
                test_system_config.home_assistant.url.replace("http://", "ws://")
                + "/api/websocket"
            )
            mock_connect.assert_called_once_with(
                expected_url,
                timeout=test_system_config.home_assistant.websocket_timeout,
                ping_interval=20,
                ping_timeout=10,
            )

            assert client.websocket == mock_websocket
            mock_auth.assert_called_once()
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_websocket_failure(self, test_system_config):
        """Test WebSocket connection failure."""
        client = HomeAssistantClient(test_system_config)

        with patch("websockets.connect", side_effect=ConnectionClosed(None, None)):
            with pytest.raises(WebSocketError):
                await client._connect_websocket()

    @pytest.mark.asyncio
    async def test_authenticate_websocket_success(self, test_system_config):
        """Test successful WebSocket authentication."""
        client = HomeAssistantClient(test_system_config)

        mock_websocket = AsyncMock()

        # Mock authentication flow
        auth_required_msg = json.dumps({"type": "auth_required"})
        auth_ok_msg = json.dumps({"type": "auth_ok"})
        mock_websocket.recv.side_effect = [auth_required_msg, auth_ok_msg]

        client.websocket = mock_websocket

        await client._authenticate_websocket()

        # Should send auth message
        assert mock_websocket.send.call_count == 1
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "auth"
        assert sent_data["access_token"] == test_system_config.home_assistant.token

    @pytest.mark.asyncio
    async def test_authenticate_websocket_wrong_initial_message(
        self, test_system_config
    ):
        """Test WebSocket authentication with wrong initial message."""
        client = HomeAssistantClient(test_system_config)

        mock_websocket = AsyncMock()
        wrong_msg = json.dumps({"type": "something_else"})
        mock_websocket.recv.return_value = wrong_msg

        client.websocket = mock_websocket

        with pytest.raises(WebSocketError):
            await client._authenticate_websocket()

    @pytest.mark.asyncio
    async def test_authenticate_websocket_auth_failed(self, test_system_config):
        """Test WebSocket authentication failure."""
        client = HomeAssistantClient(test_system_config)

        mock_websocket = AsyncMock()
        auth_required_msg = json.dumps({"type": "auth_required"})
        auth_failed_msg = json.dumps({"type": "auth_invalid"})
        mock_websocket.recv.side_effect = [auth_required_msg, auth_failed_msg]

        client.websocket = mock_websocket

        with pytest.raises(HomeAssistantAuthenticationError):
            await client._authenticate_websocket()

    @pytest.mark.asyncio
    async def test_get_entity_state_success(self, test_system_config):
        """Test successful entity state retrieval."""
        client = HomeAssistantClient(test_system_config)
        client.rate_limiter = AsyncMock()

        expected_state = {
            "entity_id": "binary_sensor.test",
            "state": "on",
            "attributes": {"device_class": "motion"},
        }

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = expected_state
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        result = await client.get_entity_state("binary_sensor.test")

        assert result == expected_state
        client.rate_limiter.acquire.assert_called_once()
        expected_url = (
            f"{test_system_config.home_assistant.url}/api/states/binary_sensor.test"
        )
        mock_session.get.assert_called_once_with(expected_url)

    @pytest.mark.asyncio
    async def test_get_entity_state_not_found(self, test_system_config):
        """Test entity state retrieval for non-existent entity."""
        client = HomeAssistantClient(test_system_config)
        client.rate_limiter = AsyncMock()

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        result = await client.get_entity_state("binary_sensor.nonexistent")

        assert result is None

    @pytest.mark.asyncio
    async def test_get_entity_state_api_error(self, test_system_config):
        """Test entity state retrieval with API error."""
        client = HomeAssistantClient(test_system_config)
        client.rate_limiter = AsyncMock()

        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text.return_value = "Internal Server Error"
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        with pytest.raises(HomeAssistantAPIError):
            await client.get_entity_state("binary_sensor.test")

    @pytest.mark.asyncio
    async def test_get_entity_history_success(self, test_system_config):
        """Test successful entity history retrieval."""
        client = HomeAssistantClient(test_system_config)
        client.rate_limiter = AsyncMock()

        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        expected_history = [
            {
                "entity_id": "binary_sensor.test",
                "state": "on",
                "last_changed": start_time.isoformat() + "Z",
                "attributes": {},
            }
        ]

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [expected_history]  # HA returns list of lists
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        result = await client.get_entity_history(
            "binary_sensor.test", start_time, end_time
        )

        assert result == expected_history
        client.rate_limiter.acquire.assert_called_once()

        # Check URL and parameters
        mock_session.get.assert_called_once()
        args, kwargs = mock_session.get.call_args
        expected_url = f"{test_system_config.home_assistant.url}/api/history/period/{start_time.isoformat()}Z"
        assert args[0] == expected_url
        assert kwargs["params"]["filter_entity_id"] == "binary_sensor.test"
        assert kwargs["params"]["end_time"] == end_time.isoformat() + "Z"

    @pytest.mark.asyncio
    async def test_get_entity_history_default_end_time(self, test_system_config):
        """Test entity history retrieval with default end time."""
        client = HomeAssistantClient(test_system_config)
        client.rate_limiter = AsyncMock()

        start_time = datetime.utcnow() - timedelta(hours=1)

        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json.return_value = [[]]
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        await client.get_entity_history("binary_sensor.test", start_time)

        # Should use current time as end_time
        args, kwargs = mock_session.get.call_args
        assert "end_time" in kwargs["params"]

    @pytest.mark.asyncio
    async def test_get_entity_history_not_found(self, test_system_config):
        """Test entity history retrieval for non-existent entity."""
        client = HomeAssistantClient(test_system_config)
        client.rate_limiter = AsyncMock()

        start_time = datetime.utcnow() - timedelta(hours=1)

        mock_response = AsyncMock()
        mock_response.status = 404
        mock_session = AsyncMock()
        mock_session.get.return_value.__aenter__.return_value = mock_response
        client.session = mock_session

        with pytest.raises(EntityNotFoundError):
            await client.get_entity_history("binary_sensor.nonexistent", start_time)

    @pytest.mark.asyncio
    async def test_validate_entities_success(self, test_system_config):
        """Test successful entity validation."""
        client = HomeAssistantClient(test_system_config)

        # Mock get_entity_state to return states for some entities
        async def mock_get_state(entity_id):
            if entity_id in ["binary_sensor.exists1", "binary_sensor.exists2"]:
                return {"entity_id": entity_id, "state": "on"}
            return None

        client.get_entity_state = AsyncMock(side_effect=mock_get_state)

        entities = [
            "binary_sensor.exists1",
            "binary_sensor.exists2",
            "binary_sensor.missing",
        ]
        result = await client.validate_entities(entities)

        expected = {
            "binary_sensor.exists1": True,
            "binary_sensor.exists2": True,
            "binary_sensor.missing": False,
        }

        assert result == expected
        assert client.get_entity_state.call_count == 3

    @pytest.mark.asyncio
    async def test_validate_entities_with_errors(self, test_system_config):
        """Test entity validation with errors."""
        client = HomeAssistantClient(test_system_config)

        # Mock get_entity_state to raise errors for some entities
        async def mock_get_state(entity_id):
            if entity_id == "binary_sensor.error":
                raise Exception("API Error")
            return {"entity_id": entity_id, "state": "on"}

        client.get_entity_state = AsyncMock(side_effect=mock_get_state)

        entities = ["binary_sensor.good", "binary_sensor.error"]
        result = await client.validate_entities(entities)

        expected = {"binary_sensor.good": True, "binary_sensor.error": False}

        assert result == expected

    def test_convert_ha_event_to_sensor_event(self, test_system_config):
        """Test converting HAEvent to SensorEvent."""
        client = HomeAssistantClient(test_system_config)

        timestamp = datetime.utcnow()
        ha_event = HAEvent(
            entity_id="binary_sensor.test_motion",
            state="on",
            previous_state="off",
            timestamp=timestamp,
            attributes={"device_class": "motion", "friendly_name": "Test Motion"},
        )

        sensor_event = client.convert_ha_event_to_sensor_event(
            ha_event, room_id="living_room", sensor_type="presence"
        )

        assert isinstance(sensor_event, SensorEvent)
        assert sensor_event.room_id == "living_room"
        assert sensor_event.sensor_id == "binary_sensor.test_motion"
        assert sensor_event.sensor_type == "presence"
        assert sensor_event.state == "on"
        assert sensor_event.previous_state == "off"
        assert sensor_event.timestamp == timestamp
        assert sensor_event.attributes == ha_event.attributes
        assert sensor_event.is_human_triggered is True  # Default

    def test_convert_history_to_sensor_events(self, test_system_config):
        """Test converting history data to SensorEvent list."""
        client = HomeAssistantClient(test_system_config)

        base_time = datetime.utcnow()
        history_data = [
            {
                "entity_id": "binary_sensor.test",
                "state": "off",
                "last_changed": base_time.isoformat() + "Z",
                "attributes": {"device_class": "motion"},
            },
            {
                "entity_id": "binary_sensor.test",
                "state": "on",
                "last_changed": (base_time + timedelta(minutes=5)).isoformat() + "Z",
                "attributes": {"device_class": "motion"},
            },
            {
                "entity_id": "binary_sensor.test",
                "state": "off",
                "last_changed": (base_time + timedelta(minutes=10)).isoformat() + "Z",
                "attributes": {"device_class": "motion"},
            },
        ]

        events = client.convert_history_to_sensor_events(
            history_data, room_id="bedroom", sensor_type="motion"
        )

        assert len(events) == 3

        # Check first event
        assert events[0].room_id == "bedroom"
        assert events[0].sensor_id == "binary_sensor.test"
        assert events[0].sensor_type == "motion"
        assert events[0].state == "off"
        assert events[0].previous_state is None  # First event has no previous

        # Check second event
        assert events[1].state == "on"
        assert events[1].previous_state == "off"  # Previous state from last event

        # Check third event
        assert events[2].state == "off"
        assert events[2].previous_state == "on"

    def test_convert_history_invalid_timestamps(self, test_system_config):
        """Test converting history data with invalid timestamps."""
        client = HomeAssistantClient(test_system_config)

        history_data = [
            {
                "entity_id": "binary_sensor.test",
                "state": "on",
                "last_changed": "invalid_timestamp",
                "attributes": {},
            },
            {
                "entity_id": "binary_sensor.test",
                "state": "off",
                # Missing timestamp
                "attributes": {},
            },
        ]

        events = client.convert_history_to_sensor_events(
            history_data, room_id="test_room", sensor_type="motion"
        )

        # Should skip events with invalid/missing timestamps
        assert len(events) == 0

    def test_is_connected_property(self, test_system_config):
        """Test is_connected property."""
        client = HomeAssistantClient(test_system_config)

        # Initially not connected
        assert not client.is_connected

        # Mock connected state
        client._connected = True
        mock_websocket = Mock()
        mock_websocket.closed = False
        client.websocket = mock_websocket

        assert client.is_connected

        # Test with closed websocket
        mock_websocket.closed = True
        assert not client.is_connected

        # Test without websocket
        client.websocket = None
        assert not client.is_connected


class TestHomeAssistantClientWebSocketHandling:
    """Test WebSocket event handling functionality."""

    @pytest.mark.asyncio
    async def test_subscribe_to_events_success(self, test_system_config):
        """Test successful event subscription."""
        client = HomeAssistantClient(test_system_config)
        client._connected = True

        mock_websocket = AsyncMock()
        client.websocket = mock_websocket
        client._ws_message_id = 1

        # Mock successful response
        future = asyncio.Future()
        future.set_result({"success": True})
        client._pending_responses[1] = future

        entity_ids = ["binary_sensor.test1", "binary_sensor.test2"]

        with patch("asyncio.wait_for", return_value={"success": True}):
            await client.subscribe_to_events(entity_ids)

        # Should send subscription message
        mock_websocket.send.assert_called_once()
        sent_data = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_data["type"] == "subscribe_events"
        assert sent_data["event_type"] == "state_changed"
        assert sent_data["id"] == 1

        # Should update subscribed entities
        assert client._subscribed_entities == set(entity_ids)

    @pytest.mark.asyncio
    async def test_subscribe_to_events_not_connected(self, test_system_config):
        """Test event subscription when not connected."""
        client = HomeAssistantClient(test_system_config)
        client._connected = False

        with pytest.raises(HomeAssistantConnectionError):
            await client.subscribe_to_events(["binary_sensor.test"])

    @pytest.mark.asyncio
    async def test_subscribe_to_events_timeout(self, test_system_config):
        """Test event subscription timeout."""
        client = HomeAssistantClient(test_system_config)
        client._connected = True
        client.websocket = AsyncMock()

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            with pytest.raises(HomeAssistantAPIError):
                await client.subscribe_to_events(["binary_sensor.test"])

    def test_add_remove_event_handler(self, test_system_config):
        """Test adding and removing event handlers."""
        client = HomeAssistantClient(test_system_config)

        def handler1(event):
            pass

        def handler2(event):
            pass

        # Add handlers
        client.add_event_handler(handler1)
        client.add_event_handler(handler2)

        assert len(client._event_handlers) == 2
        assert handler1 in client._event_handlers
        assert handler2 in client._event_handlers

        # Remove handler
        client.remove_event_handler(handler1)

        assert len(client._event_handlers) == 1
        assert handler1 not in client._event_handlers
        assert handler2 in client._event_handlers

        # Remove non-existent handler (should not error)
        client.remove_event_handler(handler1)
        assert len(client._event_handlers) == 1

    @pytest.mark.asyncio
    async def test_handle_event_processing(self, test_system_config):
        """Test WebSocket event processing."""
        client = HomeAssistantClient(test_system_config)
        client._subscribed_entities = {"binary_sensor.test"}

        # Mock event handler
        handler_calls = []

        async def mock_handler(event):
            handler_calls.append(event)

        client.add_event_handler(mock_handler)

        # Create WebSocket event data
        event_data = {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "binary_sensor.test",
                    "new_state": {
                        "state": "on",
                        "last_changed": datetime.utcnow().isoformat() + "Z",
                        "attributes": {"device_class": "motion"},
                    },
                    "old_state": {"state": "off"},
                },
            },
        }

        await client._handle_event(event_data)

        # Should have called handler with HAEvent
        assert len(handler_calls) == 1
        ha_event = handler_calls[0]
        assert isinstance(ha_event, HAEvent)
        assert ha_event.entity_id == "binary_sensor.test"
        assert ha_event.state == "on"
        assert ha_event.previous_state == "off"

    @pytest.mark.asyncio
    async def test_handle_event_not_subscribed(self, test_system_config):
        """Test WebSocket event for non-subscribed entity."""
        client = HomeAssistantClient(test_system_config)
        client._subscribed_entities = {"binary_sensor.other"}  # Different entity

        handler_calls = []

        def mock_handler(event):
            handler_calls.append(event)

        client.add_event_handler(mock_handler)

        event_data = {
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "binary_sensor.test",  # Not subscribed
                    "new_state": {"state": "on"},
                    "old_state": {"state": "off"},
                },
            },
        }

        await client._handle_event(event_data)

        # Should not call handler for unsubscribed entity
        assert len(handler_calls) == 0

    @pytest.mark.asyncio
    async def test_should_process_event_deduplication(self, test_system_config):
        """Test event deduplication logic."""
        client = HomeAssistantClient(test_system_config)

        base_time = datetime.utcnow()

        # First event should be processed
        event1 = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=base_time,
            attributes={},
        )

        assert client._should_process_event(event1) is True

        # Update last event time
        client._last_event_times["binary_sensor.test"] = base_time

        # Second event too soon should be filtered
        event2 = HAEvent(
            entity_id="binary_sensor.test",
            state="off",
            previous_state="on",
            timestamp=base_time
            + timedelta(seconds=2),  # Less than MIN_EVENT_SEPARATION
            attributes={},
        )

        assert client._should_process_event(event2) is False

        # Third event after sufficient time should be processed
        event3 = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=base_time
            + timedelta(seconds=10),  # More than MIN_EVENT_SEPARATION
            attributes={},
        )

        assert client._should_process_event(event3) is True

    @pytest.mark.asyncio
    async def test_should_process_event_invalid(self, test_system_config):
        """Test filtering invalid events."""
        client = HomeAssistantClient(test_system_config)

        # Invalid event (unavailable state)
        invalid_event = HAEvent(
            entity_id="binary_sensor.test",
            state="unavailable",  # Invalid state
            previous_state="on",
            timestamp=datetime.utcnow(),
            attributes={},
        )

        assert client._should_process_event(invalid_event) is False

        # Valid event
        valid_event = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=datetime.utcnow(),
            attributes={},
        )

        assert client._should_process_event(valid_event) is True


@pytest.mark.unit
@pytest.mark.ha_client
class TestHomeAssistantClientIntegration:
    """Integration tests for HomeAssistantClient functionality."""

    @pytest.mark.asyncio
    async def test_async_context_manager(self, test_system_config):
        """Test using client as async context manager."""
        with patch.object(HomeAssistantClient, "connect") as mock_connect, patch.object(
            HomeAssistantClient, "disconnect"
        ) as mock_disconnect:

            async with HomeAssistantClient(test_system_config) as client:
                assert isinstance(client, HomeAssistantClient)

            mock_connect.assert_called_once()
            mock_disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_reconnection_logic(self, test_system_config):
        """Test automatic reconnection on connection loss."""
        client = HomeAssistantClient(test_system_config)
        client._connected = True
        client._max_reconnect_attempts = 2
        client._base_reconnect_delay = 0.01  # Fast for testing

        # Mock successful reconnection
        with patch.object(client, "_cleanup_connections") as mock_cleanup, patch.object(
            client, "connect"
        ) as mock_connect, patch.object(
            client, "subscribe_to_events"
        ) as mock_subscribe, patch(
            "asyncio.sleep"
        ) as mock_sleep:

            # Set up subscribed entities
            client._subscribed_entities = {"binary_sensor.test"}

            await client._reconnect()

            mock_cleanup.assert_called_once()
            mock_connect.assert_called_once()
            mock_subscribe.assert_called_once_with(["binary_sensor.test"])
            mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_bulk_history_processing(self, test_system_config):
        """Test bulk history data processing with batching."""
        client = HomeAssistantClient(test_system_config)

        entity_ids = [f"binary_sensor.test_{i}" for i in range(5)]
        start_time = datetime.utcnow() - timedelta(hours=1)
        end_time = datetime.utcnow()

        # Mock get_entity_history to return different data for each entity
        async def mock_get_history(entity_id, start, end):
            return [
                {
                    "entity_id": entity_id,
                    "state": "on",
                    "last_changed": start.isoformat() + "Z",
                }
            ]

        client.get_entity_history = AsyncMock(side_effect=mock_get_history)

        # Process in batches of 2
        results = []
        async for batch in client.get_bulk_history(
            entity_ids, start_time, end_time, batch_size=2
        ):
            results.extend(batch)

        # Should get data for all entities
        assert len(results) == 5

        # Should have called get_entity_history for each entity
        assert client.get_entity_history.call_count == 5

        # Check that each entity was processed
        processed_entities = set()
        for call in client.get_entity_history.call_args_list:
            processed_entities.add(call[0][0])  # First argument is entity_id

        assert processed_entities == set(entity_ids)
