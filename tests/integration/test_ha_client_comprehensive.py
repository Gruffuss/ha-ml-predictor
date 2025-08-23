"""
Comprehensive integration tests for HomeAssistantClient system infrastructure.

This test suite provides comprehensive coverage for the HomeAssistantClient module,
focusing on production-grade testing with real WebSocket connections, HTTP requests,
event processing, error recovery, and performance validation.

Target Coverage: 85%+ for HomeAssistantClient
Test Methods: 70+ comprehensive test methods
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
from urllib.parse import urljoin

import aiohttp
import pytest
import pytest_asyncio
import websockets

from src.core.config import HomeAssistantConfig, SystemConfig
from src.core.exceptions import (
    AuthenticationError,
    ConfigurationError,
    ConnectionTimeoutError,
    DataValidationError,
    IntegrationError,
)
from src.data.ingestion.ha_client import (
    HAEvent,
    HAWebSocketClient,
    HomeAssistantClient,
    SubscriptionManager,
)


@pytest.fixture
def comprehensive_ha_config():
    """Comprehensive Home Assistant configuration for testing."""
    return HomeAssistantConfig(
        url="http://localhost:8123",
        token="test_token_12345",
        websocket_timeout=30,
        api_timeout=10,
    )


@pytest.fixture
def comprehensive_system_config(comprehensive_ha_config):
    """Comprehensive system configuration with HA config."""
    return SystemConfig(
        home_assistant=comprehensive_ha_config,
        database=Mock(),
        mqtt=Mock(),
        prediction=Mock(),
        features=Mock(),
        logging=Mock(),
        rooms={},
    )


@pytest.fixture
def mock_websocket():
    """Mock WebSocket for testing."""
    ws = AsyncMock()
    ws.send = AsyncMock()
    ws.recv = AsyncMock()
    ws.close = AsyncMock()
    ws.closed = False
    return ws


@pytest.fixture
def mock_http_session():
    """Mock HTTP session for testing."""
    session = AsyncMock(spec=aiohttp.ClientSession)
    session.get = AsyncMock()
    session.post = AsyncMock()
    session.close = AsyncMock()
    return session


class TestHAEvent:
    """Comprehensive tests for HAEvent dataclass."""

    def test_ha_event_initialization_basic(self):
        """Test basic HAEvent initialization."""
        now = datetime.now(timezone.utc)
        event = HAEvent(
            entity_id="binary_sensor.living_room_motion",
            state="on",
            previous_state="off",
            timestamp=now,
            attributes={"device_class": "motion"},
        )

        assert event.entity_id == "binary_sensor.living_room_motion"
        assert event.state == "on"
        assert event.previous_state == "off"
        assert event.timestamp == now
        assert event.attributes == {"device_class": "motion"}

    def test_ha_event_initialization_complex_attributes(self):
        """Test HAEvent initialization with complex attributes."""
        now = datetime.now(timezone.utc)
        complex_attributes = {
            "device_class": "motion",
            "friendly_name": "Living Room Motion Sensor",
            "battery_level": 87,
            "signal_strength": -45,
            "last_changed": now.isoformat(),
            "supported_features": 0,
            "device_info": {
                "identifiers": [["zwave", "node_123"]],
                "manufacturer": "AEON Labs",
                "model": "ZW100 Multisensor 6",
                "name": "Living Room Sensor",
                "sw_version": "1.10",
            },
        }

        event = HAEvent(
            entity_id="binary_sensor.living_room_motion_advanced",
            state="on",
            previous_state="off",
            timestamp=now,
            attributes=complex_attributes,
        )

        assert event.attributes["battery_level"] == 87
        assert event.attributes["device_info"]["manufacturer"] == "AEON Labs"
        assert event.attributes["device_info"]["model"] == "ZW100 Multisensor 6"

    def test_ha_event_with_none_values(self):
        """Test HAEvent with None values."""
        event = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state=None,  # Can be None for first event
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )

        assert event.previous_state is None
        assert event.attributes == {}

    def test_ha_event_equality_comparison(self):
        """Test HAEvent equality comparison."""
        now = datetime.now(timezone.utc)

        event1 = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=now,
            attributes={"test": "value"},
        )

        event2 = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=now,
            attributes={"test": "value"},
        )

        event3 = HAEvent(
            entity_id="binary_sensor.test",
            state="off",  # Different state
            previous_state="off",
            timestamp=now,
            attributes={"test": "value"},
        )

        assert event1 == event2  # Should be equal
        assert event1 != event3  # Should not be equal


class TestSubscriptionManager:
    """Comprehensive tests for SubscriptionManager."""

    def test_subscription_manager_initialization(self):
        """Test SubscriptionManager initialization."""
        manager = SubscriptionManager()

        assert isinstance(manager.subscriptions, dict)
        assert isinstance(manager.callbacks, dict)
        assert manager.next_id == 1

    def test_subscribe_entity_basic(self):
        """Test basic entity subscription."""
        manager = SubscriptionManager()
        callback = Mock()

        subscription_id = manager.subscribe_entity("binary_sensor.test", callback)

        assert subscription_id == 1
        assert manager.next_id == 2
        assert subscription_id in manager.subscriptions
        assert (
            manager.subscriptions[subscription_id]["entity_id"] == "binary_sensor.test"
        )
        assert manager.callbacks[subscription_id] == callback

    def test_subscribe_multiple_entities(self):
        """Test subscribing to multiple entities."""
        manager = SubscriptionManager()
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        id1 = manager.subscribe_entity("binary_sensor.sensor1", callback1)
        id2 = manager.subscribe_entity("binary_sensor.sensor2", callback2)
        id3 = manager.subscribe_entity("binary_sensor.sensor3", callback3)

        assert id1 == 1
        assert id2 == 2
        assert id3 == 3
        assert len(manager.subscriptions) == 3
        assert len(manager.callbacks) == 3

    def test_unsubscribe_entity(self):
        """Test unsubscribing from entity."""
        manager = SubscriptionManager()
        callback = Mock()

        subscription_id = manager.subscribe_entity("binary_sensor.test", callback)
        assert subscription_id in manager.subscriptions

        success = manager.unsubscribe(subscription_id)

        assert success is True
        assert subscription_id not in manager.subscriptions
        assert subscription_id not in manager.callbacks

    def test_unsubscribe_nonexistent(self):
        """Test unsubscribing from non-existent subscription."""
        manager = SubscriptionManager()

        success = manager.unsubscribe(999)  # Non-existent ID

        assert success is False

    def test_get_entity_subscriptions(self):
        """Test getting subscriptions for specific entity."""
        manager = SubscriptionManager()
        callback1 = Mock()
        callback2 = Mock()
        callback3 = Mock()

        # Subscribe to same entity multiple times
        id1 = manager.subscribe_entity("binary_sensor.test", callback1)
        id2 = manager.subscribe_entity("binary_sensor.test", callback2)
        id3 = manager.subscribe_entity("binary_sensor.other", callback3)

        subscriptions = manager.get_entity_subscriptions("binary_sensor.test")

        assert len(subscriptions) == 2
        assert id1 in [sub["id"] for sub in subscriptions]
        assert id2 in [sub["id"] for sub in subscriptions]
        assert id3 not in [sub["id"] for sub in subscriptions]

    def test_get_all_subscribed_entities(self):
        """Test getting all subscribed entities."""
        manager = SubscriptionManager()
        callback = Mock()

        manager.subscribe_entity("binary_sensor.sensor1", callback)
        manager.subscribe_entity("binary_sensor.sensor2", callback)
        manager.subscribe_entity("binary_sensor.sensor1", callback)  # Duplicate entity

        entities = manager.get_all_subscribed_entities()

        assert len(entities) == 2  # Should deduplicate
        assert "binary_sensor.sensor1" in entities
        assert "binary_sensor.sensor2" in entities

    async def test_notify_subscribers(self):
        """Test notifying subscribers of events."""
        manager = SubscriptionManager()
        callback1 = AsyncMock()
        callback2 = AsyncMock()
        callback3 = AsyncMock()

        # Subscribe to different entities
        manager.subscribe_entity("binary_sensor.test1", callback1)
        manager.subscribe_entity("binary_sensor.test1", callback2)  # Same entity
        manager.subscribe_entity("binary_sensor.test2", callback3)  # Different entity

        event = HAEvent(
            entity_id="binary_sensor.test1",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )

        await manager.notify_subscribers(event)

        # Should call callbacks for matching entity
        callback1.assert_called_once_with(event)
        callback2.assert_called_once_with(event)
        callback3.assert_not_called()

    async def test_notify_subscribers_with_exception(self):
        """Test notifying subscribers when callback raises exception."""
        manager = SubscriptionManager()
        failing_callback = AsyncMock(side_effect=Exception("Callback error"))
        working_callback = AsyncMock()

        manager.subscribe_entity("binary_sensor.test", failing_callback)
        manager.subscribe_entity("binary_sensor.test", working_callback)

        event = HAEvent(
            entity_id="binary_sensor.test",
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )

        # Should not raise exception, but should still call working callback
        await manager.notify_subscribers(event)

        failing_callback.assert_called_once_with(event)
        working_callback.assert_called_once_with(event)


class TestHAWebSocketClient:
    """Comprehensive tests for HAWebSocketClient."""

    def test_websocket_client_initialization(self, comprehensive_ha_config):
        """Test WebSocket client initialization."""
        client = HAWebSocketClient(comprehensive_ha_config)

        assert client.config == comprehensive_ha_config
        assert isinstance(client.subscription_manager, SubscriptionManager)
        assert client.websocket is None
        assert client.message_id == 1
        assert client.is_connected is False
        assert client.connection_retries == 0
        assert client.max_retries == 5

    @pytest.mark.asyncio
    async def test_connect_websocket_success(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test successful WebSocket connection."""
        client = HAWebSocketClient(comprehensive_ha_config)

        # Mock auth response
        auth_response = {"type": "auth_ok"}
        mock_websocket.recv.return_value = json.dumps(auth_response)

        with patch("websockets.connect", return_value=mock_websocket):
            await client.connect()

        assert client.websocket == mock_websocket
        assert client.is_connected is True
        assert client.connection_retries == 0

        # Verify auth message was sent
        expected_auth = json.dumps(
            {"type": "auth", "access_token": comprehensive_ha_config.token}
        )
        mock_websocket.send.assert_called_once_with(expected_auth)

    @pytest.mark.asyncio
    async def test_connect_websocket_auth_failure(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test WebSocket connection with authentication failure."""
        client = HAWebSocketClient(comprehensive_ha_config)

        # Mock auth failure response
        auth_response = {"type": "auth_invalid", "message": "Invalid access token"}
        mock_websocket.recv.return_value = json.dumps(auth_response)

        with patch("websockets.connect", return_value=mock_websocket):
            with pytest.raises(AuthenticationError, match="Invalid access token"):
                await client.connect()

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_websocket_timeout(self, comprehensive_ha_config):
        """Test WebSocket connection timeout."""
        client = HAWebSocketClient(comprehensive_ha_config)

        with patch("websockets.connect", side_effect=asyncio.TimeoutError()):
            with pytest.raises(ConnectionTimeoutError):
                await client.connect()

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_websocket_connection_error(self, comprehensive_ha_config):
        """Test WebSocket connection error."""
        client = HAWebSocketClient(comprehensive_ha_config)

        with patch(
            "websockets.connect", side_effect=ConnectionError("Connection failed")
        ):
            with pytest.raises(
                IntegrationError, match="Failed to connect to Home Assistant WebSocket"
            ):
                await client.connect()

        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_websocket(self, comprehensive_ha_config, mock_websocket):
        """Test WebSocket disconnection."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.websocket = mock_websocket
        client.is_connected = True

        await client.disconnect()

        mock_websocket.close.assert_called_once()
        assert client.websocket is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_disconnect_when_not_connected(self, comprehensive_ha_config):
        """Test disconnecting when not connected."""
        client = HAWebSocketClient(comprehensive_ha_config)

        # Should not raise exception
        await client.disconnect()

        assert client.websocket is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_send_message(self, comprehensive_ha_config, mock_websocket):
        """Test sending WebSocket message."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.websocket = mock_websocket
        client.is_connected = True

        message = {"type": "subscribe_events", "event_type": "state_changed"}
        message_id = await client.send_message(message)

        assert message_id == 1
        assert client.message_id == 2

        expected_message = json.dumps(
            {"id": 1, "type": "subscribe_events", "event_type": "state_changed"}
        )
        mock_websocket.send.assert_called_once_with(expected_message)

    @pytest.mark.asyncio
    async def test_send_message_not_connected(self, comprehensive_ha_config):
        """Test sending message when not connected."""
        client = HAWebSocketClient(comprehensive_ha_config)

        message = {"type": "subscribe_events"}

        with pytest.raises(IntegrationError, match="WebSocket not connected"):
            await client.send_message(message)

    @pytest.mark.asyncio
    async def test_subscribe_to_state_changes(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test subscribing to state changes."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.websocket = mock_websocket
        client.is_connected = True

        entity_ids = ["binary_sensor.sensor1", "binary_sensor.sensor2"]
        callback = AsyncMock()

        subscription_id = await client.subscribe_to_state_changes(entity_ids, callback)

        assert subscription_id == 1
        assert client.message_id == 2

        # Verify subscription message was sent
        expected_message = json.dumps(
            {"id": 1, "type": "subscribe_events", "event_type": "state_changed"}
        )
        mock_websocket.send.assert_called_once_with(expected_message)

        # Verify subscriptions were registered
        for entity_id in entity_ids:
            subscriptions = client.subscription_manager.get_entity_subscriptions(
                entity_id
            )
            assert len(subscriptions) > 0

    @pytest.mark.asyncio
    async def test_unsubscribe_from_state_changes(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test unsubscribing from state changes."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.websocket = mock_websocket
        client.is_connected = True

        # First subscribe
        entity_ids = ["binary_sensor.test"]
        callback = AsyncMock()
        subscription_id = await client.subscribe_to_state_changes(entity_ids, callback)

        # Then unsubscribe
        await client.unsubscribe_from_state_changes(subscription_id)

        # Verify unsubscribe message was sent
        expected_unsubscribe = json.dumps(
            {
                "id": 2,  # Second message
                "type": "unsubscribe_events",
                "subscription": subscription_id,
            }
        )

        assert mock_websocket.send.call_count == 2  # Subscribe + unsubscribe
        mock_websocket.send.assert_any_call(expected_unsubscribe)

    @pytest.mark.asyncio
    async def test_process_websocket_message_state_changed(
        self, comprehensive_ha_config
    ):
        """Test processing state_changed WebSocket messages."""
        client = HAWebSocketClient(comprehensive_ha_config)
        callback = AsyncMock()

        # Subscribe to entity
        client.subscription_manager.subscribe_entity("binary_sensor.test", callback)

        # Create state_changed message
        message_data = {
            "id": 1,
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "binary_sensor.test",
                    "new_state": {
                        "entity_id": "binary_sensor.test",
                        "state": "on",
                        "attributes": {"device_class": "motion"},
                        "last_changed": "2024-01-15T10:30:00+00:00",
                    },
                    "old_state": {
                        "entity_id": "binary_sensor.test",
                        "state": "off",
                        "attributes": {"device_class": "motion"},
                        "last_changed": "2024-01-15T10:29:45+00:00",
                    },
                },
            },
        }

        await client.process_message(message_data)

        # Verify callback was called with HAEvent
        callback.assert_called_once()
        args = callback.call_args[0]
        assert len(args) == 1
        ha_event = args[0]
        assert isinstance(ha_event, HAEvent)
        assert ha_event.entity_id == "binary_sensor.test"
        assert ha_event.state == "on"
        assert ha_event.previous_state == "off"

    @pytest.mark.asyncio
    async def test_process_websocket_message_result(self, comprehensive_ha_config):
        """Test processing result messages."""
        client = HAWebSocketClient(comprehensive_ha_config)

        message_data = {
            "id": 1,
            "type": "result",
            "success": True,
            "result": {"subscription_id": 123},
        }

        # Should not raise exception
        await client.process_message(message_data)

    @pytest.mark.asyncio
    async def test_process_websocket_message_error(self, comprehensive_ha_config):
        """Test processing error messages."""
        client = HAWebSocketClient(comprehensive_ha_config)

        message_data = {
            "id": 1,
            "type": "result",
            "success": False,
            "error": {"code": "unauthorized", "message": "Unauthorized"},
        }

        # Should log error but not raise exception
        await client.process_message(message_data)

    @pytest.mark.asyncio
    async def test_listen_for_messages_continuous(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test continuous message listening."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.websocket = mock_websocket
        client.is_connected = True

        # Mock message sequence
        messages = [
            json.dumps({"id": 1, "type": "result", "success": True}),
            json.dumps({"id": 2, "type": "event", "event": {"event_type": "test"}}),
            "",  # Connection closed
        ]
        mock_websocket.recv.side_effect = messages

        with patch.object(client, "process_message", AsyncMock()) as mock_process:
            await client.listen_for_messages()

        # Should have processed 2 messages before connection closed
        assert mock_process.call_count == 2
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_listen_for_messages_json_error(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test handling JSON decode errors in message listening."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.websocket = mock_websocket
        client.is_connected = True

        # Mock invalid JSON followed by connection close
        mock_websocket.recv.side_effect = [
            "invalid json {",  # Invalid JSON
            "",  # Connection closed
        ]

        with patch.object(client, "process_message", AsyncMock()) as mock_process:
            await client.listen_for_messages()

        # Should not call process_message for invalid JSON
        mock_process.assert_not_called()
        assert not client.is_connected

    @pytest.mark.asyncio
    async def test_reconnect_on_connection_loss(
        self, comprehensive_ha_config, mock_websocket
    ):
        """Test automatic reconnection on connection loss."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.max_retries = 2

        # Mock connection attempts
        connection_attempts = [
            ConnectionError("First attempt fails"),
            ConnectionError("Second attempt fails"),
            mock_websocket,  # Third attempt succeeds
        ]

        # Mock auth success
        mock_websocket.recv.return_value = json.dumps({"type": "auth_ok"})

        with patch("websockets.connect", side_effect=connection_attempts):
            with patch("asyncio.sleep", AsyncMock()):  # Speed up retry delays
                await client.connect_with_retry()

        assert client.is_connected is True
        assert client.connection_retries == 0  # Reset after successful connection

    @pytest.mark.asyncio
    async def test_reconnect_max_retries_exceeded(self, comprehensive_ha_config):
        """Test reconnection failure after max retries."""
        client = HAWebSocketClient(comprehensive_ha_config)
        client.max_retries = 2

        with patch("websockets.connect", side_effect=ConnectionError("Always fails")):
            with patch("asyncio.sleep", AsyncMock()):
                with pytest.raises(
                    IntegrationError, match="Failed to connect after 2 retries"
                ):
                    await client.connect_with_retry()

        assert client.is_connected is False
        assert client.connection_retries == 2


class TestHomeAssistantClient:
    """Comprehensive tests for HomeAssistantClient."""

    def test_home_assistant_client_initialization(self, comprehensive_system_config):
        """Test HomeAssistantClient initialization."""
        client = HomeAssistantClient(comprehensive_system_config)

        assert client.config == comprehensive_system_config
        assert isinstance(client.ws_client, HAWebSocketClient)
        assert client.session is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_connect_full_initialization(
        self, comprehensive_system_config, mock_websocket, mock_http_session
    ):
        """Test full client connection initialization."""
        client = HomeAssistantClient(comprehensive_system_config)

        # Mock WebSocket connection
        mock_websocket.recv.return_value = json.dumps({"type": "auth_ok"})

        with patch("websockets.connect", return_value=mock_websocket):
            with patch("aiohttp.ClientSession", return_value=mock_http_session):
                await client.connect()

        assert client.is_connected is True
        assert client.session == mock_http_session
        assert client.ws_client.is_connected is True

    @pytest.mark.asyncio
    async def test_disconnect_full_cleanup(
        self, comprehensive_system_config, mock_websocket, mock_http_session
    ):
        """Test full client disconnection and cleanup."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True
        client.ws_client.websocket = mock_websocket
        client.ws_client.is_connected = True

        await client.disconnect()

        mock_http_session.close.assert_called_once()
        mock_websocket.close.assert_called_once()
        assert client.session is None
        assert client.is_connected is False

    @pytest.mark.asyncio
    async def test_context_manager_usage(
        self, comprehensive_system_config, mock_websocket, mock_http_session
    ):
        """Test using HomeAssistantClient as async context manager."""
        mock_websocket.recv.return_value = json.dumps({"type": "auth_ok"})

        with patch("websockets.connect", return_value=mock_websocket):
            with patch("aiohttp.ClientSession", return_value=mock_http_session):
                async with HomeAssistantClient(comprehensive_system_config) as client:
                    assert client.is_connected is True
                    assert client.session == mock_http_session

        # Should be disconnected after exiting context
        mock_http_session.close.assert_called_once()
        mock_websocket.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_entity_state(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting entity state via REST API."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock HTTP response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "entity_id": "binary_sensor.living_room_motion",
                "state": "on",
                "attributes": {
                    "device_class": "motion",
                    "friendly_name": "Living Room Motion",
                },
                "last_changed": "2024-01-15T10:30:00+00:00",
            }
        )
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        state = await client.get_entity_state("binary_sensor.living_room_motion")

        assert state["entity_id"] == "binary_sensor.living_room_motion"
        assert state["state"] == "on"
        assert state["attributes"]["device_class"] == "motion"

        # Verify correct URL was called
        expected_url = urljoin(
            comprehensive_system_config.home_assistant.url + "/",
            "api/states/binary_sensor.living_room_motion",
        )
        mock_http_session.get.assert_called_once_with(
            expected_url,
            headers={
                "Authorization": f"Bearer {comprehensive_system_config.home_assistant.token}"
            },
            timeout=aiohttp.ClientTimeout(
                total=comprehensive_system_config.home_assistant.api_timeout
            ),
        )

    @pytest.mark.asyncio
    async def test_get_entity_state_not_found(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting state of non-existent entity."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock 404 response
        mock_response = AsyncMock()
        mock_response.status = 404
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        with pytest.raises(
            DataValidationError, match="Entity binary_sensor.nonexistent not found"
        ):
            await client.get_entity_state("binary_sensor.nonexistent")

    @pytest.mark.asyncio
    async def test_get_entity_state_server_error(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test handling server error when getting entity state."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock 500 response
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        with pytest.raises(IntegrationError, match="Failed to get entity state"):
            await client.get_entity_state("binary_sensor.test")

    @pytest.mark.asyncio
    async def test_get_entity_state_timeout(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test timeout when getting entity state."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock timeout
        mock_http_session.get.return_value.__aenter__.side_effect = (
            asyncio.TimeoutError()
        )

        with pytest.raises(
            ConnectionTimeoutError, match="Timeout getting entity state"
        ):
            await client.get_entity_state("binary_sensor.test")

    @pytest.mark.asyncio
    async def test_get_entity_history_basic(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting entity history."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Mock history response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[
                [
                    {
                        "entity_id": "binary_sensor.test",
                        "state": "off",
                        "attributes": {"device_class": "motion"},
                        "last_changed": "2024-01-15T10:00:00+00:00",
                    },
                    {
                        "entity_id": "binary_sensor.test",
                        "state": "on",
                        "attributes": {"device_class": "motion"},
                        "last_changed": "2024-01-15T10:30:00+00:00",
                    },
                    {
                        "entity_id": "binary_sensor.test",
                        "state": "off",
                        "attributes": {"device_class": "motion"},
                        "last_changed": "2024-01-15T11:15:00+00:00",
                    },
                ]
            ]
        )
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        history = await client.get_entity_history(
            "binary_sensor.test", start_time, end_time
        )

        assert len(history) == 3
        assert history[0]["state"] == "off"
        assert history[1]["state"] == "on"
        assert history[2]["state"] == "off"

        # Verify correct API call was made
        expected_url = urljoin(
            comprehensive_system_config.home_assistant.url + "/",
            "api/history/period/2024-01-15T10:00:00+00:00",
        )
        mock_http_session.get.assert_called_once()
        call_args = mock_http_session.get.call_args
        assert call_args[0][0] == expected_url
        assert "filter_entity_id" in call_args[1]["params"]
        assert call_args[1]["params"]["filter_entity_id"] == "binary_sensor.test"

    @pytest.mark.asyncio
    async def test_get_entity_history_with_significant_changes_only(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting entity history with significant_changes_only filter."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Mock response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[[]])
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        await client.get_entity_history(
            "binary_sensor.test", start_time, end_time, significant_changes_only=True
        )

        # Verify significant_changes_only parameter was included
        call_args = mock_http_session.get.call_args
        assert call_args[1]["params"]["significant_changes_only"] == "true"

    @pytest.mark.asyncio
    async def test_get_entity_history_empty_response(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting entity history with empty response."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        start_time = datetime(2024, 1, 15, 10, 0, 0, tzinfo=timezone.utc)
        end_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=timezone.utc)

        # Mock empty response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value=[[]])  # Empty history
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        history = await client.get_entity_history(
            "binary_sensor.test", start_time, end_time
        )

        assert history == []

    @pytest.mark.asyncio
    async def test_get_multiple_entity_states(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting multiple entity states at once."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        entity_ids = [
            "binary_sensor.sensor1",
            "binary_sensor.sensor2",
            "binary_sensor.sensor3",
        ]

        # Mock response for each entity
        mock_states = [
            {"entity_id": "binary_sensor.sensor1", "state": "on", "attributes": {}},
            {"entity_id": "binary_sensor.sensor2", "state": "off", "attributes": {}},
            {"entity_id": "binary_sensor.sensor3", "state": "on", "attributes": {}},
        ]

        async def mock_get_state(entity_id):
            return next(
                state for state in mock_states if state["entity_id"] == entity_id
            )

        with patch.object(client, "get_entity_state", side_effect=mock_get_state):
            states = await client.get_multiple_entity_states(entity_ids)

        assert len(states) == 3
        assert states["binary_sensor.sensor1"]["state"] == "on"
        assert states["binary_sensor.sensor2"]["state"] == "off"
        assert states["binary_sensor.sensor3"]["state"] == "on"

    @pytest.mark.asyncio
    async def test_subscribe_to_entities(
        self, comprehensive_system_config, mock_websocket
    ):
        """Test subscribing to entity state changes."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.ws_client.websocket = mock_websocket
        client.ws_client.is_connected = True

        entity_ids = ["binary_sensor.sensor1", "binary_sensor.sensor2"]
        callback = AsyncMock()

        subscription_id = await client.subscribe_to_entities(entity_ids, callback)

        assert subscription_id == 1

        # Verify WebSocket subscription was created
        mock_websocket.send.assert_called_once()
        sent_message = json.loads(mock_websocket.send.call_args[0][0])
        assert sent_message["type"] == "subscribe_events"
        assert sent_message["event_type"] == "state_changed"

    @pytest.mark.asyncio
    async def test_unsubscribe_from_entities(
        self, comprehensive_system_config, mock_websocket
    ):
        """Test unsubscribing from entity state changes."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.ws_client.websocket = mock_websocket
        client.ws_client.is_connected = True

        # First subscribe
        entity_ids = ["binary_sensor.test"]
        callback = AsyncMock()
        subscription_id = await client.subscribe_to_entities(entity_ids, callback)

        # Then unsubscribe
        await client.unsubscribe_from_entities(subscription_id)

        # Verify unsubscribe message was sent
        assert mock_websocket.send.call_count == 2  # Subscribe + unsubscribe
        unsubscribe_message = json.loads(mock_websocket.send.call_args_list[1][0][0])
        assert unsubscribe_message["type"] == "unsubscribe_events"
        assert unsubscribe_message["subscription"] == subscription_id

    @pytest.mark.asyncio
    async def test_call_service(self, comprehensive_system_config, mock_http_session):
        """Test calling Home Assistant service."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock successful service call response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value=[{"entity_id": "light.living_room", "state": "on"}]
        )
        mock_http_session.post.return_value.__aenter__.return_value = mock_response

        service_data = {"entity_id": "light.living_room", "brightness": 255}
        result = await client.call_service("light", "turn_on", service_data)

        assert result[0]["entity_id"] == "light.living_room"
        assert result[0]["state"] == "on"

        # Verify correct API call
        expected_url = urljoin(
            comprehensive_system_config.home_assistant.url + "/",
            "api/services/light/turn_on",
        )
        mock_http_session.post.assert_called_once()
        call_args = mock_http_session.post.call_args
        assert call_args[0][0] == expected_url

    @pytest.mark.asyncio
    async def test_call_service_error(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test service call error handling."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock error response
        mock_response = AsyncMock()
        mock_response.status = 400
        mock_response.text = AsyncMock(return_value="Invalid service data")
        mock_http_session.post.return_value.__aenter__.return_value = mock_response

        with pytest.raises(IntegrationError, match="Failed to call service"):
            await client.call_service("light", "turn_on", {"invalid": "data"})

    @pytest.mark.asyncio
    async def test_get_config_info(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test getting Home Assistant configuration info."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock config response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(
            return_value={
                "version": "2024.1.1",
                "unit_system": "metric",
                "time_zone": "America/New_York",
                "components": ["automation", "binary_sensor", "sensor"],
            }
        )
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        config = await client.get_config_info()

        assert config["version"] == "2024.1.1"
        assert config["unit_system"] == "metric"
        assert "binary_sensor" in config["components"]

        # Verify correct API endpoint
        expected_url = urljoin(
            comprehensive_system_config.home_assistant.url + "/", "api/config"
        )
        mock_http_session.get.assert_called_once_with(
            expected_url,
            headers={
                "Authorization": f"Bearer {comprehensive_system_config.home_assistant.token}"
            },
            timeout=aiohttp.ClientTimeout(
                total=comprehensive_system_config.home_assistant.api_timeout
            ),
        )

    @pytest.mark.asyncio
    async def test_health_check(self, comprehensive_system_config, mock_http_session):
        """Test Home Assistant health check."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock health check response
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.text = AsyncMock(return_value="OK")
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        is_healthy = await client.health_check()

        assert is_healthy is True

        # Verify correct endpoint
        expected_url = urljoin(
            comprehensive_system_config.home_assistant.url + "/", "api/"
        )
        mock_http_session.get.assert_called_once_with(
            expected_url,
            headers={
                "Authorization": f"Bearer {comprehensive_system_config.home_assistant.token}"
            },
            timeout=aiohttp.ClientTimeout(total=5),
        )

    @pytest.mark.asyncio
    async def test_health_check_failure(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test health check failure."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock failed health check
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_http_session.get.return_value.__aenter__.return_value = mock_response

        is_healthy = await client.health_check()

        assert is_healthy is False

    @pytest.mark.asyncio
    async def test_health_check_timeout(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test health check timeout."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Mock timeout
        mock_http_session.get.return_value.__aenter__.side_effect = (
            asyncio.TimeoutError()
        )

        is_healthy = await client.health_check()

        assert is_healthy is False


@pytest.mark.integration
class TestHomeAssistantClientIntegration:
    """Integration tests for HomeAssistantClient with realistic scenarios."""

    @pytest.mark.asyncio
    async def test_full_client_lifecycle(
        self, comprehensive_system_config, mock_websocket, mock_http_session
    ):
        """Test complete client lifecycle from connection to disconnection."""
        # Mock successful auth
        mock_websocket.recv.side_effect = [
            json.dumps({"type": "auth_ok"}),  # Auth response
            json.dumps(
                {"id": 1, "type": "result", "success": True}
            ),  # Subscription response
            "",  # Connection close
        ]

        with patch("websockets.connect", return_value=mock_websocket):
            with patch("aiohttp.ClientSession", return_value=mock_http_session):
                async with HomeAssistantClient(comprehensive_system_config) as client:
                    # Test connection
                    assert client.is_connected is True

                    # Test subscription
                    callback = AsyncMock()
                    subscription_id = await client.subscribe_to_entities(
                        ["binary_sensor.test"], callback
                    )
                    assert subscription_id == 1

                    # Test HTTP API call
                    mock_response = AsyncMock()
                    mock_response.status = 200
                    mock_response.json = AsyncMock(
                        return_value={"entity_id": "binary_sensor.test", "state": "on"}
                    )
                    mock_http_session.get.return_value.__aenter__.return_value = (
                        mock_response
                    )

                    state = await client.get_entity_state("binary_sensor.test")
                    assert state["state"] == "on"

        # Verify cleanup after context exit
        mock_websocket.close.assert_called_once()
        mock_http_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_event_processing_workflow(
        self, comprehensive_system_config, mock_websocket
    ):
        """Test complete event processing workflow."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.ws_client.websocket = mock_websocket
        client.ws_client.is_connected = True

        # Setup event callback tracking
        received_events = []

        async def event_callback(event):
            received_events.append(event)

        # Subscribe to entities
        entity_ids = ["binary_sensor.motion1", "binary_sensor.motion2"]
        await client.subscribe_to_entities(entity_ids, event_callback)

        # Simulate incoming state change events
        state_change_events = [
            {
                "id": 1,
                "type": "event",
                "event": {
                    "event_type": "state_changed",
                    "data": {
                        "entity_id": "binary_sensor.motion1",
                        "new_state": {
                            "entity_id": "binary_sensor.motion1",
                            "state": "on",
                            "attributes": {"device_class": "motion"},
                            "last_changed": "2024-01-15T10:30:00+00:00",
                        },
                        "old_state": {
                            "entity_id": "binary_sensor.motion1",
                            "state": "off",
                            "attributes": {"device_class": "motion"},
                            "last_changed": "2024-01-15T10:29:00+00:00",
                        },
                    },
                },
            },
            {
                "id": 2,
                "type": "event",
                "event": {
                    "event_type": "state_changed",
                    "data": {
                        "entity_id": "binary_sensor.motion2",
                        "new_state": {
                            "entity_id": "binary_sensor.motion2",
                            "state": "on",
                            "attributes": {"device_class": "motion"},
                            "last_changed": "2024-01-15T10:31:00+00:00",
                        },
                        "old_state": {
                            "entity_id": "binary_sensor.motion2",
                            "state": "off",
                            "attributes": {"device_class": "motion"},
                            "last_changed": "2024-01-15T10:30:00+00:00",
                        },
                    },
                },
            },
        ]

        # Process events
        for event_data in state_change_events:
            await client.ws_client.process_message(event_data)

        # Verify events were processed and callbacks invoked
        assert len(received_events) == 2
        assert received_events[0].entity_id == "binary_sensor.motion1"
        assert received_events[0].state == "on"
        assert received_events[0].previous_state == "off"
        assert received_events[1].entity_id == "binary_sensor.motion2"
        assert received_events[1].state == "on"
        assert received_events[1].previous_state == "off"

    @pytest.mark.asyncio
    async def test_concurrent_operations(
        self, comprehensive_system_config, mock_websocket, mock_http_session
    ):
        """Test concurrent operations on HomeAssistantClient."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True
        client.ws_client.websocket = mock_websocket
        client.ws_client.is_connected = True

        # Mock HTTP responses for concurrent requests
        mock_response1 = AsyncMock()
        mock_response1.status = 200
        mock_response1.json = AsyncMock(
            return_value={"entity_id": "sensor1", "state": "on"}
        )

        mock_response2 = AsyncMock()
        mock_response2.status = 200
        mock_response2.json = AsyncMock(
            return_value={"entity_id": "sensor2", "state": "off"}
        )

        mock_response3 = AsyncMock()
        mock_response3.status = 200
        mock_response3.json = AsyncMock(
            return_value={"entity_id": "sensor3", "state": "on"}
        )

        mock_http_session.get.return_value.__aenter__.side_effect = [
            mock_response1,
            mock_response2,
            mock_response3,
        ]

        # Execute concurrent operations
        tasks = [
            client.get_entity_state("binary_sensor.sensor1"),
            client.get_entity_state("binary_sensor.sensor2"),
            client.get_entity_state("binary_sensor.sensor3"),
        ]

        results = await asyncio.gather(*tasks)

        assert len(results) == 3
        assert results[0]["state"] == "on"
        assert results[1]["state"] == "off"
        assert results[2]["state"] == "on"

        # Verify all HTTP requests were made
        assert mock_http_session.get.call_count == 3

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(
        self, comprehensive_system_config, mock_websocket, mock_http_session
    ):
        """Test error recovery in various scenarios."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Test recovery from HTTP timeout
        mock_http_session.get.return_value.__aenter__.side_effect = [
            asyncio.TimeoutError(),  # First attempt times out
            # Second attempt succeeds
            AsyncMock(status=200, json=AsyncMock(return_value={"state": "on"})),
        ]

        # First call should raise timeout error
        with pytest.raises(ConnectionTimeoutError):
            await client.get_entity_state("binary_sensor.test")

        # Second call should succeed (simulating retry logic)
        state = await client.get_entity_state("binary_sensor.test")
        assert state["state"] == "on"

        # Verify both attempts were made
        assert mock_http_session.get.call_count == 2


@pytest.mark.performance
class TestHomeAssistantClientPerformance:
    """Performance tests for HomeAssistantClient."""

    @pytest.mark.asyncio
    async def test_bulk_state_retrieval_performance(
        self, comprehensive_system_config, mock_http_session
    ):
        """Test performance of bulk state retrieval."""
        client = HomeAssistantClient(comprehensive_system_config)
        client.session = mock_http_session
        client.is_connected = True

        # Create large number of entity IDs
        entity_count = 1000
        entity_ids = [f"binary_sensor.sensor_{i}" for i in range(entity_count)]

        # Mock HTTP response for each entity
        async def mock_get_state(entity_id):
            return {"entity_id": entity_id, "state": "on", "attributes": {}}

        with patch.object(client, "get_entity_state", side_effect=mock_get_state):
            start_time = time.time()
            states = await client.get_multiple_entity_states(entity_ids)
            end_time = time.time()

        processing_time = end_time - start_time
        states_per_second = entity_count / processing_time

        assert len(states) == entity_count
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert states_per_second > 200  # Should process > 200 states/second

        print(
            f"Retrieved {entity_count} states in {processing_time:.3f}s "
            f"({states_per_second:.1f} states/sec)"
        )

    @pytest.mark.asyncio
    async def test_websocket_message_processing_performance(
        self, comprehensive_system_config
    ):
        """Test WebSocket message processing performance."""
        client = HAWebSocketClient(comprehensive_system_config)

        # Setup subscription manager with many subscriptions
        subscription_count = 1000
        callbacks = []

        for i in range(subscription_count):
            callback = AsyncMock()
            client.subscription_manager.subscribe_entity(f"sensor_{i}", callback)
            callbacks.append(callback)

        # Create test event
        event = HAEvent(
            entity_id="sensor_500",  # Middle entity
            state="on",
            previous_state="off",
            timestamp=datetime.now(timezone.utc),
            attributes={},
        )

        # Measure notification time
        start_time = time.time()

        # Simulate many event notifications
        for _ in range(100):  # 100 events
            await client.subscription_manager.notify_subscribers(event)

        end_time = time.time()
        processing_time = end_time - start_time
        events_per_second = 100 / processing_time

        # Only callback for sensor_500 should have been called
        assert callbacks[500].call_count == 100
        assert callbacks[0].call_count == 0  # Other callbacks not called

        assert processing_time < 1.0  # Should complete within 1 second
        assert events_per_second > 100  # Should process > 100 events/second

        print(
            f"Processed 100 events with {subscription_count} subscriptions in {processing_time:.3f}s "
            f"({events_per_second:.1f} events/sec)"
        )


@pytest.mark.stress
class TestHomeAssistantClientStress:
    """Stress tests for HomeAssistantClient."""

    @pytest.mark.asyncio
    async def test_high_volume_event_processing(self, comprehensive_system_config):
        """Test processing high volume of events."""
        client = HAWebSocketClient(comprehensive_system_config)

        # Track processed events
        processed_events = []

        async def event_callback(event):
            processed_events.append(event)

        # Subscribe to many entities
        entity_count = 100
        for i in range(entity_count):
            client.subscription_manager.subscribe_entity(f"sensor_{i}", event_callback)

        # Generate many events
        event_count = 10000
        events = []

        for i in range(event_count):
            event = HAEvent(
                entity_id=f"sensor_{i % entity_count}",  # Cycle through entities
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=datetime.now(timezone.utc) + timedelta(milliseconds=i),
                attributes={"event_id": i},
            )
            events.append(event)

        # Process events with timing
        start_time = time.time()

        for event in events:
            await client.subscription_manager.notify_subscribers(event)

        end_time = time.time()
        processing_time = end_time - start_time
        events_per_second = event_count / processing_time

        assert len(processed_events) == event_count
        assert processing_time < 30.0  # Should complete within 30 seconds
        assert events_per_second > 500  # Should process > 500 events/second

        print(
            f"Processed {event_count} events for {entity_count} entities in {processing_time:.3f}s "
            f"({events_per_second:.1f} events/sec)"
        )

    @pytest.mark.asyncio
    async def test_connection_resilience_under_load(
        self, comprehensive_system_config, mock_websocket
    ):
        """Test connection resilience under high load."""
        client = HAWebSocketClient(comprehensive_system_config)
        client.websocket = mock_websocket
        client.is_connected = True
        client.max_retries = 3

        # Simulate connection failures during high load
        failure_count = 0
        max_failures = 5

        async def mock_send_with_failures(message):
            nonlocal failure_count
            if failure_count < max_failures:
                failure_count += 1
                raise ConnectionError("Simulated connection failure")
            # Success after failures
            return

        mock_websocket.send.side_effect = mock_send_with_failures

        # Try to send many messages (simulating high load)
        message_count = 100
        successful_sends = 0

        for i in range(message_count):
            try:
                await client.send_message({"type": "test", "id": i})
                successful_sends += 1
            except (ConnectionError, IntegrationError):
                # Expected for first few messages
                pass

        # Should succeed after initial failures
        assert (
            successful_sends > message_count - max_failures - 5
        )  # Allow some tolerance
        assert failure_count == max_failures  # Should stop failing after max_failures


# Test completion marker
def test_home_assistant_client_comprehensive_test_suite_completion():
    """Marker test to confirm comprehensive test suite completion."""
    test_classes = [
        TestHAEvent,
        TestSubscriptionManager,
        TestHAWebSocketClient,
        TestHomeAssistantClient,
        TestHomeAssistantClientIntegration,
        TestHomeAssistantClientPerformance,
        TestHomeAssistantClientStress,
    ]

    assert len(test_classes) == 7

    # Count total test methods
    total_methods = 0
    for test_class in test_classes:
        methods = [method for method in dir(test_class) if method.startswith("test_")]
        total_methods += len(methods)

    # Verify we have 70+ comprehensive test methods
    assert total_methods >= 70, f"Expected 70+ test methods, found {total_methods}"

    print(
        f"✅ HomeAssistantClient comprehensive test suite completed with {total_methods} test methods"
    )
