"""
Comprehensive unit tests for WebSocket API functionality.

This test suite validates WebSocket connection management, real-time prediction
streaming, client subscriptions, authentication, and error handling.
"""

import asyncio
from datetime import datetime, timezone
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from fastapi import WebSocket, WebSocketDisconnect, status
import pytest
from websockets.exceptions import ConnectionClosed

from src.core.exceptions import APIAuthenticationError, APIError
from src.integration.websocket_api import (
    WebSocketAPI,
    WebSocketConnection,
    WebSocketConnectionManager,
    WebSocketMessage,
    WebSocketMessageType,
    broadcast_prediction_update,
    get_websocket_manager,
    handle_client_message,
    handle_websocket_endpoint,
)


class TestWebSocketMessage:
    """Test WebSocket message model and validation."""

    def test_websocket_message_creation(self):
        """Test WebSocketMessage creation with valid data."""
        message_data = {
            "type": "prediction_update",
            "room_id": "living_room",
            "prediction_data": {"confidence": 0.85},
        }

        message = WebSocketMessage(
            type=WebSocketMessageType.PREDICTION_UPDATE,
            data=message_data,
            client_id="client_123",
        )

        assert message.type == WebSocketMessageType.PREDICTION_UPDATE
        assert message.data == message_data
        assert message.client_id == "client_123"
        assert isinstance(message.timestamp, datetime)

    def test_websocket_message_to_json(self):
        """Test WebSocketMessage JSON serialization."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIPTION_ACK,
            data={"status": "subscribed", "room_id": "bedroom"},
            client_id="client_456",
        )

        json_str = message.to_json()
        parsed = json.loads(json_str)

        assert parsed["type"] == "subscription_ack"
        assert parsed["data"]["status"] == "subscribed"
        assert parsed["client_id"] == "client_456"
        assert "timestamp" in parsed

    def test_websocket_message_from_json(self):
        """Test WebSocketMessage creation from JSON."""
        json_data = {
            "type": "subscribe",
            "data": {"room_id": "kitchen"},
            "client_id": "client_789",
        }

        message = WebSocketMessage.from_json(json.dumps(json_data))

        assert message.type == WebSocketMessageType.SUBSCRIBE
        assert message.data["room_id"] == "kitchen"
        assert message.client_id == "client_789"

    def test_websocket_message_from_json_invalid(self):
        """Test WebSocketMessage creation from invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            WebSocketMessage.from_json("invalid json")

    def test_websocket_message_from_json_missing_type(self):
        """Test WebSocketMessage creation with missing type."""
        json_data = {"data": {"room_id": "kitchen"}, "client_id": "client_789"}

        with pytest.raises(ValueError, match="Missing required field"):
            WebSocketMessage.from_json(json.dumps(json_data))

    def test_websocket_message_type_enum(self):
        """Test WebSocketMessageType enum values."""
        assert WebSocketMessageType.SUBSCRIBE.value == "subscribe"
        assert WebSocketMessageType.UNSUBSCRIBE.value == "unsubscribe"
        assert WebSocketMessageType.PREDICTION_UPDATE.value == "prediction_update"
        assert WebSocketMessageType.ERROR.value == "error"
        assert WebSocketMessageType.PING.value == "ping"
        assert WebSocketMessageType.PONG.value == "pong"


class TestWebSocketConnection:
    """Test WebSocket connection management."""

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection."""
        websocket = Mock(spec=WebSocket)
        websocket.send_text = AsyncMock()
        websocket.send_json = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        websocket.client = Mock()
        websocket.client.host = "127.0.0.1"
        return websocket

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        from src.integration.auth.auth_models import AuthUser

        return AuthUser(
            user_id="user123",
            username="testuser",
            permissions=["read", "prediction_view"],
            is_active=True,
        )

    def test_websocket_connection_creation(self, mock_websocket, mock_user):
        """Test WebSocketConnection creation."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        assert connection.websocket == mock_websocket
        assert connection.client_id == "client_123"
        assert connection.user == mock_user
        assert connection.subscriptions == set()
        assert isinstance(connection.connected_at, datetime)
        assert connection.is_active is True

    def test_websocket_connection_creation_unauthenticated(self, mock_websocket):
        """Test WebSocketConnection creation without user."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123"
        )

        assert connection.user is None
        assert connection.is_authenticated is False

    @pytest.mark.asyncio
    async def test_websocket_connection_send_message(self, mock_websocket, mock_user):
        """Test sending message through WebSocket connection."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        message = WebSocketMessage(
            type=WebSocketMessageType.PREDICTION_UPDATE,
            data={"room_id": "living_room", "confidence": 0.85},
            client_id="client_123",
        )

        await connection.send_message(message)

        mock_websocket.send_text.assert_called_once()
        sent_data = mock_websocket.send_text.call_args[0][0]
        parsed = json.loads(sent_data)
        assert parsed["type"] == "prediction_update"

    @pytest.mark.asyncio
    async def test_websocket_connection_send_error(self, mock_websocket, mock_user):
        """Test sending error message."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        await connection.send_error("Test error", "TEST_ERROR", {"detail": "test"})

        mock_websocket.send_text.assert_called_once()
        sent_data = mock_websocket.send_text.call_args[0][0]
        parsed = json.loads(sent_data)
        assert parsed["type"] == "error"
        assert parsed["data"]["error"] == "Test error"
        assert parsed["data"]["error_code"] == "TEST_ERROR"

    @pytest.mark.asyncio
    async def test_websocket_connection_send_message_connection_error(
        self, mock_websocket, mock_user
    ):
        """Test sending message with connection error."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        mock_websocket.send_text.side_effect = ConnectionClosed(
            1000, "Connection closed"
        )

        message = WebSocketMessage(
            type=WebSocketMessageType.PING, data={}, client_id="client_123"
        )

        # Should not raise exception, but connection becomes inactive
        await connection.send_message(message)

        assert connection.is_active is False

    def test_websocket_connection_subscribe_room(self, mock_websocket, mock_user):
        """Test room subscription."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        connection.subscribe_to_room("living_room")

        assert "living_room" in connection.subscriptions

    def test_websocket_connection_unsubscribe_room(self, mock_websocket, mock_user):
        """Test room unsubscription."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        connection.subscribe_to_room("living_room")
        connection.unsubscribe_from_room("living_room")

        assert "living_room" not in connection.subscriptions

    def test_websocket_connection_is_subscribed(self, mock_websocket, mock_user):
        """Test subscription checking."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        connection.subscribe_to_room("bedroom")

        assert connection.is_subscribed_to("bedroom") is True
        assert connection.is_subscribed_to("kitchen") is False

    def test_websocket_connection_has_permission(self, mock_websocket, mock_user):
        """Test permission checking."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        assert connection.has_permission("read") is True
        assert connection.has_permission("prediction_view") is True
        assert connection.has_permission("admin") is False

    def test_websocket_connection_has_permission_unauthenticated(self, mock_websocket):
        """Test permission checking for unauthenticated connection."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123"
        )

        assert connection.has_permission("read") is False
        assert connection.has_permission("prediction_view") is False

    @pytest.mark.asyncio
    async def test_websocket_connection_close(self, mock_websocket, mock_user):
        """Test connection closing."""
        connection = WebSocketConnection(
            websocket=mock_websocket, client_id="client_123", user=mock_user
        )

        await connection.close(1000, "Normal closure")

        mock_websocket.close.assert_called_once_with(code=1000, reason="Normal closure")
        assert connection.is_active is False


class TestWebSocketConnectionManager:
    """Test WebSocket connection manager functionality."""

    @pytest.fixture
    def connection_manager(self):
        """WebSocket connection manager fixture."""
        return WebSocketConnectionManager()

    @pytest.fixture
    def mock_connection(self):
        """Mock WebSocket connection."""
        connection = Mock(spec=WebSocketConnection)
        connection.client_id = "client_123"
        connection.is_active = True
        connection.subscriptions = set()
        connection.send_message = AsyncMock()
        connection.close = AsyncMock()
        return connection

    def test_connection_manager_initialization(self, connection_manager):
        """Test WebSocketConnectionManager initialization."""
        assert len(connection_manager._connections) == 0
        assert len(connection_manager._room_subscriptions) == 0

    def test_add_connection(self, connection_manager, mock_connection):
        """Test adding connection to manager."""
        connection_manager.add_connection(mock_connection)

        assert "client_123" in connection_manager._connections
        assert connection_manager._connections["client_123"] == mock_connection

    def test_remove_connection(self, connection_manager, mock_connection):
        """Test removing connection from manager."""
        connection_manager.add_connection(mock_connection)
        connection_manager.remove_connection("client_123")

        assert "client_123" not in connection_manager._connections

    def test_remove_nonexistent_connection(self, connection_manager):
        """Test removing non-existent connection."""
        # Should not raise exception
        connection_manager.remove_connection("nonexistent")

    def test_get_connection(self, connection_manager, mock_connection):
        """Test getting connection by client ID."""
        connection_manager.add_connection(mock_connection)

        retrieved = connection_manager.get_connection("client_123")
        assert retrieved == mock_connection

    def test_get_nonexistent_connection(self, connection_manager):
        """Test getting non-existent connection."""
        retrieved = connection_manager.get_connection("nonexistent")
        assert retrieved is None

    def test_subscribe_to_room(self, connection_manager, mock_connection):
        """Test subscribing connection to room."""
        connection_manager.add_connection(mock_connection)
        connection_manager.subscribe_to_room("client_123", "living_room")

        # Check room subscription tracking
        assert "living_room" in connection_manager._room_subscriptions
        assert "client_123" in connection_manager._room_subscriptions["living_room"]

        # Check connection subscription
        mock_connection.subscribe_to_room.assert_called_once_with("living_room")

    def test_unsubscribe_from_room(self, connection_manager, mock_connection):
        """Test unsubscribing connection from room."""
        connection_manager.add_connection(mock_connection)
        connection_manager.subscribe_to_room("client_123", "living_room")
        connection_manager.unsubscribe_from_room("client_123", "living_room")

        # Check room subscription tracking
        if "living_room" in connection_manager._room_subscriptions:
            assert (
                "client_123"
                not in connection_manager._room_subscriptions["living_room"]
            )

        # Check connection unsubscription
        mock_connection.unsubscribe_from_room.assert_called_once_with("living_room")

    def test_get_room_subscribers(self, connection_manager, mock_connection):
        """Test getting room subscribers."""
        mock_connection2 = Mock(spec=WebSocketConnection)
        mock_connection2.client_id = "client_456"
        mock_connection2.is_active = True

        connection_manager.add_connection(mock_connection)
        connection_manager.add_connection(mock_connection2)
        connection_manager.subscribe_to_room("client_123", "living_room")
        connection_manager.subscribe_to_room("client_456", "living_room")

        subscribers = connection_manager.get_room_subscribers("living_room")
        subscriber_ids = [conn.client_id for conn in subscribers]

        assert "client_123" in subscriber_ids
        assert "client_456" in subscriber_ids

    def test_get_room_subscribers_nonexistent_room(self, connection_manager):
        """Test getting subscribers for non-existent room."""
        subscribers = connection_manager.get_room_subscribers("nonexistent")
        assert len(subscribers) == 0

    @pytest.mark.asyncio
    async def test_broadcast_to_room(self, connection_manager, mock_connection):
        """Test broadcasting message to room subscribers."""
        connection_manager.add_connection(mock_connection)
        connection_manager.subscribe_to_room("client_123", "living_room")

        message = WebSocketMessage(
            type=WebSocketMessageType.PREDICTION_UPDATE,
            data={"confidence": 0.85},
            client_id="broadcast",
        )

        await connection_manager.broadcast_to_room("living_room", message)

        mock_connection.send_message.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, connection_manager, mock_connection):
        """Test broadcasting message to all connections."""
        mock_connection2 = Mock(spec=WebSocketConnection)
        mock_connection2.client_id = "client_456"
        mock_connection2.is_active = True
        mock_connection2.send_message = AsyncMock()

        connection_manager.add_connection(mock_connection)
        connection_manager.add_connection(mock_connection2)

        message = WebSocketMessage(
            type=WebSocketMessageType.PING, data={}, client_id="broadcast"
        )

        await connection_manager.broadcast_to_all(message)

        mock_connection.send_message.assert_called_once_with(message)
        mock_connection2.send_message.assert_called_once_with(message)

    @pytest.mark.asyncio
    async def test_cleanup_inactive_connections(
        self, connection_manager, mock_connection
    ):
        """Test cleanup of inactive connections."""
        # Add active connection
        connection_manager.add_connection(mock_connection)
        connection_manager.subscribe_to_room("client_123", "living_room")

        # Make connection inactive
        mock_connection.is_active = False

        await connection_manager.cleanup_inactive_connections()

        # Connection should be removed
        assert "client_123" not in connection_manager._connections

        # Room subscriptions should be cleaned up
        if "living_room" in connection_manager._room_subscriptions:
            assert (
                "client_123"
                not in connection_manager._room_subscriptions["living_room"]
            )

    def test_get_connection_stats(self, connection_manager, mock_connection):
        """Test getting connection statistics."""
        connection_manager.add_connection(mock_connection)
        connection_manager.subscribe_to_room("client_123", "living_room")

        stats = connection_manager.get_connection_stats()

        assert stats["total_connections"] == 1
        assert stats["active_connections"] == 1
        assert stats["total_room_subscriptions"] == 1
        assert "living_room" in stats["rooms_with_subscribers"]

    @pytest.mark.asyncio
    async def test_disconnect_all_connections(
        self, connection_manager, mock_connection
    ):
        """Test disconnecting all connections."""
        connection_manager.add_connection(mock_connection)

        await connection_manager.disconnect_all(1001, "Server shutdown")

        mock_connection.close.assert_called_once_with(1001, "Server shutdown")


class TestWebSocketMessageHandling:
    """Test WebSocket message handling functions."""

    @pytest.fixture
    def mock_connection(self):
        """Mock WebSocket connection."""
        connection = Mock(spec=WebSocketConnection)
        connection.client_id = "client_123"
        connection.send_message = AsyncMock()
        connection.send_error = AsyncMock()
        connection.has_permission = Mock(return_value=True)
        connection.subscribe_to_room = Mock()
        connection.unsubscribe_from_room = Mock()
        return connection

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock WebSocket connection manager."""
        manager = Mock(spec=WebSocketConnectionManager)
        manager.subscribe_to_room = Mock()
        manager.unsubscribe_from_room = Mock()
        return manager

    @pytest.mark.asyncio
    async def test_handle_subscribe_message(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling room subscription message."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={"room_id": "living_room"},
            client_id="client_123",
        )

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.rooms = {"living_room": {}}

            await handle_client_message(
                message, mock_connection, mock_connection_manager
            )

            mock_connection_manager.subscribe_to_room.assert_called_once_with(
                "client_123", "living_room"
            )
            mock_connection.send_message.assert_called_once()

            # Verify acknowledgment message
            sent_message = mock_connection.send_message.call_args[0][0]
            assert sent_message.type == WebSocketMessageType.SUBSCRIPTION_ACK

    @pytest.mark.asyncio
    async def test_handle_subscribe_message_invalid_room(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling subscription to invalid room."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={"room_id": "nonexistent_room"},
            client_id="client_123",
        )

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.rooms = {"living_room": {}}

            await handle_client_message(
                message, mock_connection, mock_connection_manager
            )

            mock_connection.send_error.assert_called_once()
            error_args = mock_connection.send_error.call_args[0]
            assert "Invalid room ID" in error_args[0]

    @pytest.mark.asyncio
    async def test_handle_subscribe_message_insufficient_permission(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling subscription without sufficient permissions."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={"room_id": "living_room"},
            client_id="client_123",
        )

        mock_connection.has_permission.return_value = False

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.rooms = {"living_room": {}}

            await handle_client_message(
                message, mock_connection, mock_connection_manager
            )

            mock_connection.send_error.assert_called_once()
            error_args = mock_connection.send_error.call_args[0]
            assert "Insufficient permissions" in error_args[0]

    @pytest.mark.asyncio
    async def test_handle_unsubscribe_message(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling room unsubscription message."""
        message = WebSocketMessage(
            type=WebSocketMessageType.UNSUBSCRIBE,
            data={"room_id": "living_room"},
            client_id="client_123",
        )

        await handle_client_message(message, mock_connection, mock_connection_manager)

        mock_connection_manager.unsubscribe_from_room.assert_called_once_with(
            "client_123", "living_room"
        )
        mock_connection.send_message.assert_called_once()

        # Verify acknowledgment message
        sent_message = mock_connection.send_message.call_args[0][0]
        assert sent_message.type == WebSocketMessageType.SUBSCRIPTION_ACK

    @pytest.mark.asyncio
    async def test_handle_ping_message(self, mock_connection, mock_connection_manager):
        """Test handling ping message."""
        message = WebSocketMessage(
            type=WebSocketMessageType.PING,
            data={"timestamp": "2024-01-01T12:00:00Z"},
            client_id="client_123",
        )

        await handle_client_message(message, mock_connection, mock_connection_manager)

        mock_connection.send_message.assert_called_once()

        # Verify pong response
        sent_message = mock_connection.send_message.call_args[0][0]
        assert sent_message.type == WebSocketMessageType.PONG

    @pytest.mark.asyncio
    async def test_handle_unknown_message_type(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling unknown message type."""
        message = WebSocketMessage(
            type="unknown_type", data={}, client_id="client_123"  # Invalid type
        )

        await handle_client_message(message, mock_connection, mock_connection_manager)

        mock_connection.send_error.assert_called_once()
        error_args = mock_connection.send_error.call_args[0]
        assert "Unknown message type" in error_args[0]

    @pytest.mark.asyncio
    async def test_handle_message_missing_data(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling message with missing required data."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={},  # Missing room_id
            client_id="client_123",
        )

        await handle_client_message(message, mock_connection, mock_connection_manager)

        mock_connection.send_error.assert_called_once()
        error_args = mock_connection.send_error.call_args[0]
        assert "Missing room_id" in error_args[0]

    @pytest.mark.asyncio
    async def test_handle_message_exception(
        self, mock_connection, mock_connection_manager
    ):
        """Test handling message with unexpected exception."""
        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={"room_id": "living_room"},
            client_id="client_123",
        )

        mock_connection_manager.subscribe_to_room.side_effect = Exception(
            "Unexpected error"
        )

        await handle_client_message(message, mock_connection, mock_connection_manager)

        mock_connection.send_error.assert_called_once()
        error_args = mock_connection.send_error.call_args[0]
        assert "Internal server error" in error_args[0]


class TestWebSocketBroadcast:
    """Test WebSocket broadcasting functionality."""

    @pytest.fixture
    def mock_connection_manager(self):
        """Mock WebSocket connection manager."""
        manager = Mock(spec=WebSocketConnectionManager)
        manager.broadcast_to_room = AsyncMock()
        return manager

    @pytest.mark.asyncio
    async def test_broadcast_prediction_update(self, mock_connection_manager):
        """Test broadcasting prediction update to room subscribers."""
        prediction_data = {
            "room_id": "living_room",
            "predicted_time": "2024-01-01T15:30:00Z",
            "transition_type": "occupied_to_vacant",
            "confidence": 0.85,
            "alternatives": [],
        }

        with patch(
            "src.integration.websocket_api.get_websocket_manager",
            return_value=mock_connection_manager,
        ):
            await broadcast_prediction_update("living_room", prediction_data)

            mock_connection_manager.broadcast_to_room.assert_called_once()
            call_args = mock_connection_manager.broadcast_to_room.call_args
            room_id, message = call_args[0]

            assert room_id == "living_room"
            assert message.type == WebSocketMessageType.PREDICTION_UPDATE
            assert message.data == prediction_data

    @pytest.mark.asyncio
    async def test_broadcast_prediction_update_no_manager(self):
        """Test broadcasting when WebSocket manager is not available."""
        prediction_data = {"room_id": "living_room", "confidence": 0.85}

        with patch(
            "src.integration.websocket_api.get_websocket_manager", return_value=None
        ):
            # Should not raise exception
            await broadcast_prediction_update("living_room", prediction_data)

    @pytest.mark.asyncio
    async def test_broadcast_prediction_update_exception(self, mock_connection_manager):
        """Test broadcasting with exception in manager."""
        mock_connection_manager.broadcast_to_room.side_effect = Exception(
            "Broadcast failed"
        )

        prediction_data = {"room_id": "living_room", "confidence": 0.85}

        with patch(
            "src.integration.websocket_api.get_websocket_manager",
            return_value=mock_connection_manager,
        ):
            # Should not raise exception, should handle gracefully
            await broadcast_prediction_update("living_room", prediction_data)


class TestWebSocketEndpoint:
    """Test WebSocket endpoint handling."""

    @pytest.fixture
    def mock_websocket(self):
        """Mock FastAPI WebSocket."""
        websocket = AsyncMock(spec=WebSocket)
        websocket.accept = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.close = AsyncMock()
        websocket.client = Mock()
        websocket.client.host = "127.0.0.1"
        return websocket

    @pytest.fixture
    def mock_user(self):
        """Mock authenticated user."""
        from src.integration.auth.auth_models import AuthUser

        return AuthUser(
            user_id="user123",
            username="testuser",
            permissions=["prediction_view"],
            is_active=True,
        )

    @pytest.mark.asyncio
    async def test_websocket_endpoint_success(self, mock_websocket, mock_user):
        """Test successful WebSocket connection handling."""
        # Mock message sequence
        messages = [
            json.dumps(
                {
                    "type": "subscribe",
                    "data": {"room_id": "living_room"},
                    "client_id": "client_123",
                }
            ),
            json.dumps(
                {
                    "type": "ping",
                    "data": {"timestamp": "2024-01-01T12:00:00Z"},
                    "client_id": "client_123",
                }
            ),
        ]

        mock_websocket.receive_text.side_effect = messages + [WebSocketDisconnect()]

        with patch(
            "src.integration.websocket_api.get_websocket_manager"
        ) as mock_get_manager, patch(
            "src.integration.websocket_api.handle_client_message"
        ) as mock_handle, patch(
            "src.integration.websocket_api.get_config"
        ) as mock_config:

            mock_manager = Mock(spec=WebSocketConnectionManager)
            mock_manager.add_connection = Mock()
            mock_manager.remove_connection = Mock()
            mock_get_manager.return_value = mock_manager

            mock_config.return_value.rooms = {"living_room": {}}
            mock_handle.return_value = None

            await handle_websocket_endpoint(mock_websocket, mock_user)

            # Verify connection was accepted
            mock_websocket.accept.assert_called_once()

            # Verify connection was added and removed from manager
            mock_manager.add_connection.assert_called_once()
            mock_manager.remove_connection.assert_called_once()

            # Verify messages were handled
            assert mock_handle.call_count == 2

    @pytest.mark.asyncio
    async def test_websocket_endpoint_unauthenticated(self, mock_websocket):
        """Test WebSocket connection without authentication."""
        await handle_websocket_endpoint(mock_websocket, None)

        # Should close connection immediately
        mock_websocket.close.assert_called_once_with(
            code=1008, reason="Authentication required"
        )

    @pytest.mark.asyncio
    async def test_websocket_endpoint_insufficient_permissions(self, mock_websocket):
        """Test WebSocket connection with insufficient permissions."""
        from src.integration.auth.auth_models import AuthUser

        user_no_perms = AuthUser(
            user_id="user123",
            username="testuser",
            permissions=["read"],  # Missing prediction_view
            is_active=True,
        )

        await handle_websocket_endpoint(mock_websocket, user_no_perms)

        # Should close connection
        mock_websocket.close.assert_called_once_with(
            code=1008, reason="Insufficient permissions"
        )

    @pytest.mark.asyncio
    async def test_websocket_endpoint_invalid_message(self, mock_websocket, mock_user):
        """Test WebSocket endpoint with invalid message."""
        mock_websocket.receive_text.side_effect = [
            "invalid json",  # Invalid JSON
            WebSocketDisconnect(),
        ]

        with patch(
            "src.integration.websocket_api.get_websocket_manager"
        ) as mock_get_manager, patch(
            "src.integration.websocket_api.get_config"
        ) as mock_config:

            mock_manager = Mock(spec=WebSocketConnectionManager)
            mock_manager.add_connection = Mock()
            mock_manager.remove_connection = Mock()
            mock_get_manager.return_value = mock_manager

            mock_config.return_value.rooms = {}

            # Mock connection for sending error
            with patch(
                "src.integration.websocket_api.WebSocketConnection"
            ) as mock_conn_class:
                mock_connection = Mock()
                mock_connection.send_error = AsyncMock()
                mock_conn_class.return_value = mock_connection

                await handle_websocket_endpoint(mock_websocket, mock_user)

                # Should send error for invalid JSON
                mock_connection.send_error.assert_called()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_connection_closed(
        self, mock_websocket, mock_user
    ):
        """Test WebSocket endpoint with connection closed during operation."""
        mock_websocket.receive_text.side_effect = ConnectionClosed(
            1000, "Connection closed"
        )

        with patch(
            "src.integration.websocket_api.get_websocket_manager"
        ) as mock_get_manager:
            mock_manager = Mock(spec=WebSocketConnectionManager)
            mock_manager.add_connection = Mock()
            mock_manager.remove_connection = Mock()
            mock_get_manager.return_value = mock_manager

            # Should handle connection closed gracefully
            await handle_websocket_endpoint(mock_websocket, mock_user)

            mock_manager.remove_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_websocket_endpoint_runtime_error(self, mock_websocket, mock_user):
        """Test WebSocket endpoint with unexpected runtime error."""
        mock_websocket.receive_text.side_effect = RuntimeError("Unexpected error")

        with patch(
            "src.integration.websocket_api.get_websocket_manager"
        ) as mock_get_manager:
            mock_manager = Mock(spec=WebSocketConnectionManager)
            mock_manager.add_connection = Mock()
            mock_manager.remove_connection = Mock()
            mock_get_manager.return_value = mock_manager

            # Should handle error gracefully
            await handle_websocket_endpoint(mock_websocket, mock_user)

            mock_manager.remove_connection.assert_called_once()


class TestWebSocketAPI:
    """Test WebSocket API integration class."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Mock tracking manager."""
        manager = AsyncMock()
        return manager

    def test_websocket_api_initialization(self, mock_tracking_manager):
        """Test WebSocketAPI initialization."""
        websocket_api = WebSocketAPI(mock_tracking_manager)

        assert websocket_api.tracking_manager == mock_tracking_manager
        assert isinstance(websocket_api.connection_manager, WebSocketConnectionManager)

    @pytest.mark.asyncio
    async def test_websocket_api_start(self, mock_tracking_manager):
        """Test WebSocketAPI start method."""
        websocket_api = WebSocketAPI(mock_tracking_manager)

        await websocket_api.start()

        # Should start background tasks
        assert len(websocket_api._background_tasks) > 0

    @pytest.mark.asyncio
    async def test_websocket_api_stop(self, mock_tracking_manager):
        """Test WebSocketAPI stop method."""
        websocket_api = WebSocketAPI(mock_tracking_manager)

        # Add mock background task
        mock_task = AsyncMock()
        websocket_api._background_tasks.append(mock_task)

        await websocket_api.stop()

        # Should cancel background tasks
        mock_task.cancel.assert_called_once()

    def test_websocket_api_is_running(self, mock_tracking_manager):
        """Test WebSocketAPI is_running method."""
        websocket_api = WebSocketAPI(mock_tracking_manager)

        # Initially not running
        assert websocket_api.is_running() is False

        # Add mock task
        mock_task = Mock()
        mock_task.done.return_value = False
        websocket_api._background_tasks.append(mock_task)

        assert websocket_api.is_running() is True

    @pytest.mark.asyncio
    async def test_websocket_api_prediction_update_handler(self, mock_tracking_manager):
        """Test WebSocket API prediction update handling."""
        websocket_api = WebSocketAPI(mock_tracking_manager)

        # Mock connection manager
        websocket_api.connection_manager.broadcast_to_room = AsyncMock()

        prediction_data = {"room_id": "living_room", "confidence": 0.85}

        await websocket_api.handle_prediction_update("living_room", prediction_data)

        websocket_api.connection_manager.broadcast_to_room.assert_called_once()

    def test_websocket_api_get_stats(self, mock_tracking_manager):
        """Test WebSocketAPI statistics retrieval."""
        websocket_api = WebSocketAPI(mock_tracking_manager)

        # Mock connection manager stats
        websocket_api.connection_manager.get_connection_stats = Mock(
            return_value={"total_connections": 5, "active_connections": 4}
        )

        stats = websocket_api.get_stats()

        assert "websocket_connections" in stats
        assert stats["websocket_connections"]["total_connections"] == 5


class TestWebSocketManagerSingleton:
    """Test WebSocket manager singleton functionality."""

    def test_get_websocket_manager_initialization(self):
        """Test WebSocket manager singleton initialization."""
        with patch("src.integration.websocket_api._websocket_manager_instance", None):
            with patch(
                "src.integration.websocket_api.get_tracking_manager"
            ) as mock_get_tracking:
                mock_tracking = AsyncMock()
                mock_get_tracking.return_value = mock_tracking

                manager = get_websocket_manager()

                assert isinstance(manager, WebSocketConnectionManager)

    def test_get_websocket_manager_cached(self):
        """Test WebSocket manager returns cached instance."""
        mock_manager = Mock(spec=WebSocketConnectionManager)

        with patch(
            "src.integration.websocket_api._websocket_manager_instance", mock_manager
        ):
            result = get_websocket_manager()

            assert result == mock_manager


class TestWebSocketErrorHandling:
    """Test WebSocket error handling scenarios."""

    @pytest.mark.asyncio
    async def test_connection_cleanup_on_error(self):
        """Test that connections are properly cleaned up on errors."""
        mock_websocket = AsyncMock(spec=WebSocket)
        mock_websocket.accept = AsyncMock()
        mock_websocket.receive_text.side_effect = Exception("Unexpected error")

        from src.integration.auth.auth_models import AuthUser

        mock_user = AuthUser(
            user_id="user123",
            username="testuser",
            permissions=["prediction_view"],
            is_active=True,
        )

        with patch(
            "src.integration.websocket_api.get_websocket_manager"
        ) as mock_get_manager:
            mock_manager = Mock(spec=WebSocketConnectionManager)
            mock_manager.add_connection = Mock()
            mock_manager.remove_connection = Mock()
            mock_get_manager.return_value = mock_manager

            await handle_websocket_endpoint(mock_websocket, mock_user)

            # Connection should still be removed even on error
            mock_manager.remove_connection.assert_called_once()

    @pytest.mark.asyncio
    async def test_broadcast_error_handling(self):
        """Test error handling in broadcast functions."""
        with patch(
            "src.integration.websocket_api.get_websocket_manager"
        ) as mock_get_manager:
            mock_manager = Mock(spec=WebSocketConnectionManager)
            mock_manager.broadcast_to_room.side_effect = Exception("Broadcast failed")
            mock_get_manager.return_value = mock_manager

            # Should not raise exception
            await broadcast_prediction_update("living_room", {"confidence": 0.85})

    @pytest.mark.asyncio
    async def test_message_handling_error_recovery(self):
        """Test that message handling errors don't crash the connection."""
        mock_connection = Mock(spec=WebSocketConnection)
        mock_connection.send_error = AsyncMock()

        mock_manager = Mock(spec=WebSocketConnectionManager)
        mock_manager.subscribe_to_room.side_effect = Exception("Subscribe failed")

        message = WebSocketMessage(
            type=WebSocketMessageType.SUBSCRIBE,
            data={"room_id": "living_room"},
            client_id="client_123",
        )

        await handle_client_message(message, mock_connection, mock_manager)

        # Should send error message instead of crashing
        mock_connection.send_error.assert_called_once()
