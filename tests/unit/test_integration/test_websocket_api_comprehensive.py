"""
Comprehensive unit tests for WebSocket API module.

This test suite validates WebSocket API server functionality including
connection management, authentication, message handling, and real-time updates.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, Mock, patch
import uuid

import pytest
from starlette.testclient import TestClient
from starlette.websockets import WebSocket

from src.core.exceptions import (
    WebSocketAuthenticationError,
    WebSocketConnectionError,
    WebSocketValidationError,
)
from src.integration.websocket_api import (
    ClientAuthRequest,
    ClientConnection,
    ClientSubscription,
    MessageType,
    WebSocketAPIServer,
    WebSocketConnectionManager,
    WebSocketEndpoint,
    WebSocketMessage,
    WebSocketStats,
    create_websocket_api_server,
    create_websocket_app,
    websocket_api_context,
)
from src.models.base.predictor import PredictionResult


class TestWebSocketMessage:
    """Test WebSocket message functionality."""

    def test_websocket_message_creation(self):
        """Test WebSocket message creation."""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        message = WebSocketMessage(
            message_id=message_id,
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=timestamp,
            endpoint="/ws/predictions",
            data={"room_id": "living_room", "confidence": 0.85},
            requires_ack=True,
            room_id="living_room",
        )

        assert message.message_id == message_id
        assert message.message_type == MessageType.PREDICTION_UPDATE
        assert message.timestamp == timestamp
        assert message.endpoint == "/ws/predictions"
        assert message.data["room_id"] == "living_room"
        assert message.requires_ack is True
        assert message.room_id == "living_room"

    def test_websocket_message_to_json(self):
        """Test WebSocket message JSON serialization."""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        message = WebSocketMessage(
            message_id=message_id,
            message_type=MessageType.HEARTBEAT,
            timestamp=timestamp,
            endpoint="/ws/system-status",
            data={"server_time": timestamp.isoformat()},
        )

        json_str = message.to_json()
        parsed = json.loads(json_str)

        assert parsed["message_id"] == message_id
        assert parsed["message_type"] == MessageType.HEARTBEAT.value
        assert parsed["timestamp"] == timestamp.isoformat()
        assert parsed["endpoint"] == "/ws/system-status"
        assert parsed["requires_ack"] is False  # Default value

    def test_websocket_message_from_json(self):
        """Test WebSocket message JSON deserialization."""
        message_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)

        json_data = {
            "message_id": message_id,
            "message_type": MessageType.SUBSCRIBE.value,
            "timestamp": timestamp.isoformat(),
            "endpoint": "/ws/alerts",
            "data": {"room_filter": "kitchen"},
            "requires_ack": True,
            "room_id": "kitchen",
        }

        message = WebSocketMessage.from_json(json.dumps(json_data))

        assert message.message_id == message_id
        assert message.message_type == MessageType.SUBSCRIBE
        assert message.timestamp == timestamp
        assert message.endpoint == "/ws/alerts"
        assert message.data["room_filter"] == "kitchen"
        assert message.requires_ack is True
        assert message.room_id == "kitchen"

    def test_websocket_message_round_trip(self):
        """Test WebSocket message serialization round trip."""
        original_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"error_code": "VALIDATION_FAILED", "message": "Invalid request"},
            requires_ack=False,
        )

        json_str = original_message.to_json()
        restored_message = WebSocketMessage.from_json(json_str)

        assert restored_message.message_id == original_message.message_id
        assert restored_message.message_type == original_message.message_type
        assert restored_message.endpoint == original_message.endpoint
        assert restored_message.data == original_message.data


class TestClientAuthRequest:
    """Test client authentication request validation."""

    def test_client_auth_request_valid(self):
        """Test valid client authentication request."""
        auth_request = ClientAuthRequest(
            api_key="test-api-key-12345",
            client_name="TestClient",
            capabilities=["predictions", "alerts"],
            room_filters=["living_room", "bedroom"],
        )

        assert auth_request.api_key == "test-api-key-12345"
        assert auth_request.client_name == "TestClient"
        assert auth_request.capabilities == ["predictions", "alerts"]
        assert auth_request.room_filters == ["living_room", "bedroom"]

    def test_client_auth_request_minimal(self):
        """Test client authentication request with minimal data."""
        auth_request = ClientAuthRequest(api_key="minimal-key")

        assert auth_request.api_key == "minimal-key"
        assert auth_request.client_name is None
        assert auth_request.capabilities == []
        assert auth_request.room_filters == []

    def test_client_auth_request_validation_errors(self):
        """Test client authentication request validation."""
        # Too short API key
        with pytest.raises(ValueError, match="API key must be at least 10 characters"):
            ClientAuthRequest(api_key="short")

        # Empty API key
        with pytest.raises(ValueError, match="API key must be at least 10 characters"):
            ClientAuthRequest(api_key="")


class TestClientSubscription:
    """Test client subscription request validation."""

    def test_client_subscription_valid_endpoints(self):
        """Test valid subscription endpoints."""
        valid_endpoints = [
            "/ws/predictions",
            "/ws/system-status",
            "/ws/alerts",
            "/ws/room/living_room",
            "/ws/room/bedroom_1",
        ]

        for endpoint in valid_endpoints:
            subscription = ClientSubscription(endpoint=endpoint)
            assert subscription.endpoint == endpoint

    def test_client_subscription_with_room_id(self):
        """Test subscription with room ID."""
        subscription = ClientSubscription(
            endpoint="/ws/room/kitchen",
            room_id="kitchen",
            filters={"confidence_threshold": 0.8},
        )

        assert subscription.endpoint == "/ws/room/kitchen"
        assert subscription.room_id == "kitchen"
        assert subscription.filters["confidence_threshold"] == 0.8

    def test_client_subscription_invalid_endpoint(self):
        """Test subscription with invalid endpoint."""
        invalid_endpoints = [
            "/invalid/endpoint",
            "/ws/invalid",
            "/ws/room/",
            "/ws/room/invalid-room-name!",
        ]

        for endpoint in invalid_endpoints:
            with pytest.raises(ValueError):
                ClientSubscription(endpoint=endpoint)


class TestClientConnection:
    """Test client connection data class."""

    def test_client_connection_initialization(self):
        """Test client connection initialization."""
        connection_id = str(uuid.uuid4())
        mock_websocket = Mock()

        connection = ClientConnection(
            connection_id=connection_id,
            websocket=mock_websocket,
            endpoint="/ws/predictions",
        )

        assert connection.connection_id == connection_id
        assert connection.websocket == mock_websocket
        assert connection.endpoint == "/ws/predictions"
        assert connection.authenticated is False
        assert isinstance(connection.connected_at, datetime)
        assert isinstance(connection.last_activity, datetime)
        assert len(connection.capabilities) == 0
        assert len(connection.subscriptions) == 0

    def test_client_connection_activity_tracking(self):
        """Test client connection activity tracking."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        original_activity = connection.last_activity

        # Wait a moment and update activity
        import time

        time.sleep(0.01)
        connection.update_activity()

        assert connection.last_activity > original_activity

    def test_client_connection_heartbeat_tracking(self):
        """Test client connection heartbeat tracking."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        original_heartbeat = connection.last_heartbeat
        original_activity = connection.last_activity

        import time

        time.sleep(0.01)
        connection.update_heartbeat()

        assert connection.last_heartbeat > original_heartbeat
        assert connection.last_activity > original_activity

    def test_client_connection_rate_limiting(self):
        """Test client connection rate limiting."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        # Should not be rate limited initially
        assert connection.is_rate_limited(10) is False

        # Add messages up to limit
        for _ in range(10):
            connection.increment_message_count()

        # Should now be rate limited
        assert connection.is_rate_limited(10) is True

    def test_client_connection_rate_limit_window_reset(self):
        """Test rate limit window reset."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        # Fill rate limit
        for _ in range(10):
            connection.increment_message_count()

        assert connection.is_rate_limited(10) is True

        # Mock time passage
        connection.last_rate_limit_reset = datetime.now(timezone.utc) - timedelta(
            minutes=2
        )

        # Should reset and allow messages
        assert connection.is_rate_limited(10) is False

    def test_client_connection_room_access(self):
        """Test client connection room access control."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        # No filters means access to all rooms
        assert connection.can_access_room("any_room") is True

        # Add room filters
        connection.room_filters = {"living_room", "bedroom"}

        assert connection.can_access_room("living_room") is True
        assert connection.can_access_room("bedroom") is True
        assert connection.can_access_room("kitchen") is False

    def test_client_connection_capabilities(self):
        """Test client connection capability checking."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        assert connection.has_capability("predictions") is False

        connection.capabilities.add("predictions")
        connection.capabilities.add("alerts")

        assert connection.has_capability("predictions") is True
        assert connection.has_capability("alerts") is True
        assert connection.has_capability("admin") is False


class TestWebSocketStats:
    """Test WebSocket statistics tracking."""

    def test_websocket_stats_initialization(self):
        """Test WebSocket stats initialization."""
        stats = WebSocketStats()

        assert stats.total_connections == 0
        assert stats.active_connections == 0
        assert stats.authenticated_connections == 0
        assert stats.total_messages_sent == 0
        assert stats.authentication_failures == 0

    def test_websocket_stats_updates(self):
        """Test WebSocket stats updates."""
        stats = WebSocketStats()

        # Update various metrics
        stats.total_connections = 100
        stats.active_connections = 25
        stats.total_messages_sent = 5000
        stats.rate_limited_clients = 3

        assert stats.total_connections == 100
        assert stats.active_connections == 25
        assert stats.total_messages_sent == 5000
        assert stats.rate_limited_clients == 3


class TestWebSocketConnectionManager:
    """Test WebSocket connection manager functionality."""

    @pytest.fixture
    def connection_manager(self):
        """Create WebSocket connection manager."""
        config = {
            "max_connections": 100,
            "max_messages_per_minute": 60,
            "heartbeat_interval_seconds": 30,
        }
        return WebSocketConnectionManager(config)

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket."""
        websocket = Mock()
        websocket.send = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_connection_manager_connect(self, connection_manager, mock_websocket):
        """Test connecting to WebSocket manager."""
        endpoint = "/ws/predictions"

        connection_id = await connection_manager.connect(mock_websocket, endpoint)

        assert connection_id in connection_manager.connections
        assert connection_manager.connections[connection_id].endpoint == endpoint
        assert connection_manager.stats.total_connections == 1
        assert connection_manager.stats.active_connections == 1

    @pytest.mark.asyncio
    async def test_connection_manager_disconnect(
        self, connection_manager, mock_websocket
    ):
        """Test disconnecting from WebSocket manager."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        await connection_manager.disconnect(connection_id)

        assert connection_id not in connection_manager.connections
        assert connection_manager.stats.active_connections == 0

    @pytest.mark.asyncio
    async def test_connection_manager_max_connections(self, mock_websocket):
        """Test connection manager enforces max connections."""
        connection_manager = WebSocketConnectionManager({"max_connections": 2})

        # Should allow up to max connections
        conn1 = await connection_manager.connect(mock_websocket, "/ws/test1")
        conn2 = await connection_manager.connect(mock_websocket, "/ws/test2")

        assert len(connection_manager.connections) == 2

        # Should reject additional connections
        with pytest.raises(
            WebSocketConnectionError, match="Maximum connections exceeded"
        ):
            await connection_manager.connect(mock_websocket, "/ws/test3")

    @pytest.mark.asyncio
    async def test_connection_manager_authenticate_client(
        self, connection_manager, mock_websocket
    ):
        """Test client authentication."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        auth_request = ClientAuthRequest(
            api_key="test-api-key-12345",
            client_name="TestClient",
            capabilities=["predictions"],
            room_filters=["living_room"],
        )

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = True
            mock_config.return_value.api.api_key = "test-api-key-12345"

            result = await connection_manager.authenticate_client(
                connection_id, auth_request
            )

        assert result is True
        connection = connection_manager.connections[connection_id]
        assert connection.authenticated is True
        assert connection.client_name == "TestClient"
        assert "predictions" in connection.capabilities
        assert "living_room" in connection.room_filters

    @pytest.mark.asyncio
    async def test_connection_manager_authenticate_invalid_key(
        self, connection_manager, mock_websocket
    ):
        """Test client authentication with invalid key."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        auth_request = ClientAuthRequest(api_key="invalid-key-12345")

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = True
            mock_config.return_value.api.api_key = "correct-key-12345"

            with pytest.raises(WebSocketAuthenticationError, match="Invalid API key"):
                await connection_manager.authenticate_client(
                    connection_id, auth_request
                )

    @pytest.mark.asyncio
    async def test_connection_manager_subscribe_client(
        self, connection_manager, mock_websocket
    ):
        """Test client subscription."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        # Authenticate first
        auth_request = ClientAuthRequest(api_key="test-key-12345")
        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False
            await connection_manager.authenticate_client(connection_id, auth_request)

        subscription = ClientSubscription(
            endpoint="/ws/predictions", room_id="living_room"
        )

        result = await connection_manager.subscribe_client(connection_id, subscription)

        assert result is True
        connection = connection_manager.connections[connection_id]
        assert "/ws/predictions" in connection.subscriptions
        assert "living_room" in connection.room_subscriptions

    @pytest.mark.asyncio
    async def test_connection_manager_subscribe_unauthenticated(
        self, connection_manager, mock_websocket
    ):
        """Test subscription without authentication."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        subscription = ClientSubscription(endpoint="/ws/predictions")

        with pytest.raises(
            WebSocketAuthenticationError, match="Authentication required"
        ):
            await connection_manager.subscribe_client(connection_id, subscription)

    @pytest.mark.asyncio
    async def test_connection_manager_subscribe_no_room_access(
        self, connection_manager, mock_websocket
    ):
        """Test subscription to room without access."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        # Authenticate with room filter
        auth_request = ClientAuthRequest(
            api_key="test-key-12345", room_filters=["bedroom"]  # Only bedroom access
        )

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False
            await connection_manager.authenticate_client(connection_id, auth_request)

        # Try to subscribe to different room
        subscription = ClientSubscription(
            endpoint="/ws/room/kitchen", room_id="kitchen"
        )

        with pytest.raises(WebSocketValidationError, match="Access denied for room"):
            await connection_manager.subscribe_client(connection_id, subscription)

    @pytest.mark.asyncio
    async def test_connection_manager_send_message(
        self, connection_manager, mock_websocket
    ):
        """Test sending message to client."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"test": "data"},
        )

        result = await connection_manager.send_message(connection_id, message)

        assert result is True
        mock_websocket.send.assert_called_once()
        connection_manager.stats.total_messages_sent == 1

    @pytest.mark.asyncio
    async def test_connection_manager_send_message_rate_limited(
        self, connection_manager, mock_websocket
    ):
        """Test sending message when rate limited."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")
        connection = connection_manager.connections[connection_id]

        # Fill rate limit
        for _ in range(60):
            connection.increment_message_count()

        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"test": "data"},
        )

        result = await connection_manager.send_message(connection_id, message)

        assert result is False  # Should fail due to rate limit
        assert connection_manager.stats.total_rate_limit_violations == 1

    @pytest.mark.asyncio
    async def test_connection_manager_broadcast_to_endpoint(self, connection_manager):
        """Test broadcasting to endpoint."""
        # Create multiple connections
        mock_ws1 = Mock()
        mock_ws1.send = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send = AsyncMock()

        conn1 = await connection_manager.connect(mock_ws1, "/ws/predictions")
        conn2 = await connection_manager.connect(mock_ws2, "/ws/predictions")

        # Authenticate and subscribe both
        auth_request = ClientAuthRequest(api_key="test-key-12345")
        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False

            await connection_manager.authenticate_client(conn1, auth_request)
            await connection_manager.authenticate_client(conn2, auth_request)

        subscription = ClientSubscription(endpoint="/ws/predictions")
        await connection_manager.subscribe_client(conn1, subscription)
        await connection_manager.subscribe_client(conn2, subscription)

        # Broadcast message
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"broadcast": "test"},
        )

        sent_count = await connection_manager.broadcast_to_endpoint(
            "/ws/predictions", message
        )

        assert sent_count == 2
        mock_ws1.send.assert_called_once()
        mock_ws2.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_connection_manager_heartbeat(
        self, connection_manager, mock_websocket
    ):
        """Test sending heartbeat to client."""
        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        result = await connection_manager.send_heartbeat(connection_id)

        assert result is True
        mock_websocket.send.assert_called_once()

        # Verify heartbeat was updated
        connection = connection_manager.connections[connection_id]
        assert connection.last_heartbeat is not None
        assert connection_manager.stats.total_heartbeats_sent == 1

    def test_connection_manager_stats(self, connection_manager):
        """Test getting connection statistics."""
        stats = connection_manager.get_connection_stats()

        assert isinstance(stats, dict)
        assert "total_connections" in stats
        assert "active_connections" in stats
        assert "total_messages_sent" in stats
        assert "authentication_failures" in stats


class TestWebSocketAPIServer:
    """Test WebSocket API server functionality."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Create mock tracking manager."""
        manager = Mock()
        manager.add_notification_callback = Mock()
        return manager

    @pytest.fixture
    def websocket_server(self, mock_tracking_manager):
        """Create WebSocket API server."""
        config = {
            "host": "localhost",
            "port": 8765,
            "enabled": True,
            "max_connections": 100,
        }
        return WebSocketAPIServer(mock_tracking_manager, config)

    @pytest.mark.asyncio
    async def test_websocket_server_initialization(self, websocket_server):
        """Test WebSocket server initialization."""
        await websocket_server.initialize()

        assert websocket_server.enabled is True
        assert websocket_server.host == "localhost"
        assert websocket_server.port == 8765

    @pytest.mark.asyncio
    async def test_websocket_server_disabled(self, mock_tracking_manager):
        """Test WebSocket server when disabled."""
        config = {"enabled": False}
        server = WebSocketAPIServer(mock_tracking_manager, config)

        await server.initialize()
        # Should not raise error even when disabled

    @pytest.mark.asyncio
    async def test_publish_prediction_update(self, websocket_server):
        """Test publishing prediction update."""
        # Mock connection manager
        websocket_server.connection_manager.broadcast_to_endpoint = AsyncMock(
            return_value=2
        )

        # Create mock prediction result
        prediction_result = Mock(spec=PredictionResult)
        prediction_result.predicted_time = datetime.now(timezone.utc) + timedelta(
            minutes=30
        )
        prediction_result.transition_type = "occupied_to_vacant"
        prediction_result.confidence_score = 0.85
        prediction_result.model_type = "lstm"
        prediction_result.model_version = "1.0"
        prediction_result.alternatives = []
        prediction_result.features_used = ["time_since_last", "day_of_week"]
        prediction_result.prediction_metadata = {}

        result = await websocket_server.publish_prediction_update(
            prediction_result, "living_room", "occupied"
        )

        assert result["success"] is True
        assert result["clients_notified"] == 4  # 2 + 2 from both endpoints
        assert websocket_server.connection_manager.broadcast_to_endpoint.call_count == 2

    @pytest.mark.asyncio
    async def test_publish_system_status_update(self, websocket_server):
        """Test publishing system status update."""
        websocket_server.connection_manager.broadcast_to_endpoint = AsyncMock(
            return_value=3
        )

        status_data = {
            "status": "healthy",
            "components": {"database": "healthy", "tracking": "healthy"},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await websocket_server.publish_system_status_update(status_data)

        assert result["success"] is True
        assert result["clients_notified"] == 3

    @pytest.mark.asyncio
    async def test_publish_alert_notification(self, websocket_server):
        """Test publishing alert notification."""
        websocket_server.connection_manager.broadcast_to_endpoint = AsyncMock(
            return_value=1
        )

        alert_data = {
            "alert_type": "system_error",
            "severity": "high",
            "message": "Database connection lost",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        result = await websocket_server.publish_alert_notification(
            alert_data, "kitchen"
        )

        assert result["success"] is True
        assert result["clients_notified"] == 1

    def test_get_server_stats(self, websocket_server):
        """Test getting server statistics."""
        stats = websocket_server.get_server_stats()

        assert isinstance(stats, dict)
        assert "server_running" in stats
        assert "host" in stats
        assert "port" in stats
        assert "enabled" in stats
        assert "connection_stats" in stats

    def test_format_prediction_data(self, websocket_server):
        """Test prediction data formatting."""
        # Mock prediction result
        prediction_result = Mock(spec=PredictionResult)
        prediction_result.predicted_time = datetime.now(timezone.utc) + timedelta(
            minutes=45
        )
        prediction_result.transition_type = "vacant_to_occupied"
        prediction_result.confidence_score = 0.92
        prediction_result.model_type = "xgboost"
        prediction_result.model_version = "2.1"
        prediction_result.alternatives = [
            (datetime.now(timezone.utc) + timedelta(minutes=50), 0.75),
            (datetime.now(timezone.utc) + timedelta(minutes=40), 0.65),
        ]
        prediction_result.features_used = ["feature1", "feature2"]
        prediction_result.prediction_metadata = {"training_date": "2024-01-01"}

        # Mock config
        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.rooms = {"bedroom": {"name": "Master Bedroom"}}

            formatted_data = websocket_server._format_prediction_data(
                prediction_result, "bedroom", "vacant"
            )

        assert formatted_data["room_id"] == "bedroom"
        assert formatted_data["room_name"] == "Master Bedroom"
        assert formatted_data["transition_type"] == "vacant_to_occupied"
        assert formatted_data["confidence_score"] == 0.92
        assert formatted_data["current_state"] == "vacant"
        assert formatted_data["model_type"] == "xgboost"
        assert len(formatted_data["alternatives"]) == 2
        assert formatted_data["features_used"] == 2

    def test_format_time_until(self, websocket_server):
        """Test time until formatting."""
        test_cases = [
            (30, "30 seconds"),
            (60, "1 minute"),
            (90, "1 minute"),
            (3600, "1 hour"),
            (3900, "1h 5m"),
            (86400, "1 day"),
            (90000, "1d 1h"),
        ]

        for seconds, expected in test_cases:
            result = websocket_server._format_time_until(seconds)
            assert result == expected

    @pytest.mark.asyncio
    async def test_websocket_server_start_stop(self, websocket_server):
        """Test WebSocket server start and stop."""
        with patch("websockets.serve") as mock_serve:
            mock_server = Mock()
            mock_serve.return_value = mock_server

            # Start server
            await websocket_server.start()
            assert websocket_server._server_running is True
            assert websocket_server._websocket_server == mock_server

            # Stop server
            mock_server.close = Mock()
            mock_server.wait_closed = AsyncMock()

            await websocket_server.stop()
            assert websocket_server._server_running is False
            mock_server.close.assert_called_once()


class TestWebSocketAPIHelpers:
    """Test WebSocket API helper functions."""

    def test_create_websocket_api_server(self):
        """Test creating WebSocket API server."""
        mock_tracking_manager = Mock()

        server = create_websocket_api_server(
            config={"host": "0.0.0.0", "port": 9000},
            tracking_manager=mock_tracking_manager,
        )

        assert isinstance(server, WebSocketAPIServer)
        assert server.tracking_manager == mock_tracking_manager
        assert server.host == "0.0.0.0"
        assert server.port == 9000

    def test_create_websocket_api_server_default_config(self):
        """Test creating WebSocket API server with default config."""
        with patch("src.integration.websocket_api.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.websocket_host = "127.0.0.1"
            mock_config.websocket_port = 8888
            mock_get_config.return_value = mock_config

            server = create_websocket_api_server()

            assert server.host == "127.0.0.1"
            assert server.port == 8888

    def test_create_websocket_app(self):
        """Test creating Starlette WebSocket app."""
        from starlette.applications import Starlette

        app = create_websocket_app()

        assert isinstance(app, Starlette)
        assert app.debug is False

    @pytest.mark.asyncio
    async def test_websocket_api_context(self):
        """Test WebSocket API context manager."""
        mock_tracking_manager = Mock()
        config = {"enabled": False}  # Disabled to avoid actual server start

        async with websocket_api_context(config, mock_tracking_manager) as server:
            assert isinstance(server, WebSocketAPIServer)
            assert server.tracking_manager == mock_tracking_manager


class TestWebSocketAPIStarletteIntegration:
    """Test WebSocket API integration with Starlette."""

    def test_starlette_app_routes(self):
        """Test that Starlette app has correct routes."""
        app = create_websocket_app()

        route_paths = [route.path for route in app.routes]

        assert "/health" in route_paths
        assert "/health/simple" in route_paths
        assert "/ws/predictions" in route_paths
        assert "/ws/system-status" in route_paths
        assert "/ws/alerts" in route_paths

    def test_starlette_health_endpoint(self):
        """Test Starlette health endpoint."""
        app = create_websocket_app()

        with TestClient(app) as client:
            response = client.get("/health")
            assert response.status_code == 200

            data = response.json()
            assert data["status"] == "healthy"
            assert data["service"] == "websocket_api"

    def test_starlette_simple_health_endpoint(self):
        """Test simple health endpoint."""
        app = create_websocket_app()

        with TestClient(app) as client:
            response = client.get("/health/simple")
            assert response.status_code == 200
            assert "healthy" in response.text


class TestWebSocketAPIExceptionHandling:
    """Test WebSocket API exception handling."""

    @pytest.fixture
    def websocket_server(self):
        """Create WebSocket server for exception testing."""
        config = {"enabled": True, "host": "localhost", "port": 8765}
        return WebSocketAPIServer(None, config)

    @pytest.mark.asyncio
    async def test_publish_prediction_update_error(self, websocket_server):
        """Test prediction update error handling."""
        # Mock connection manager to raise exception
        websocket_server.connection_manager.broadcast_to_endpoint = AsyncMock(
            side_effect=Exception("Broadcast failed")
        )

        prediction_result = Mock(spec=PredictionResult)
        prediction_result.predicted_time = datetime.now(timezone.utc)
        prediction_result.transition_type = "test"
        prediction_result.confidence_score = 0.5
        prediction_result.model_type = "test"
        prediction_result.model_version = "1.0"
        prediction_result.alternatives = []
        prediction_result.features_used = []
        prediction_result.prediction_metadata = {}

        result = await websocket_server.publish_prediction_update(
            prediction_result, "test_room"
        )

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_publish_system_status_update_error(self, websocket_server):
        """Test system status update error handling."""
        websocket_server.connection_manager.broadcast_to_endpoint = AsyncMock(
            side_effect=Exception("Status update failed")
        )

        result = await websocket_server.publish_system_status_update({})

        assert result["success"] is False
        assert "error" in result

    @pytest.mark.asyncio
    async def test_publish_alert_notification_error(self, websocket_server):
        """Test alert notification error handling."""
        websocket_server.connection_manager.broadcast_to_endpoint = AsyncMock(
            side_effect=Exception("Alert failed")
        )

        result = await websocket_server.publish_alert_notification({})

        assert result["success"] is False
        assert "error" in result


class TestWebSocketAPIPerformance:
    """Test WebSocket API performance aspects."""

    @pytest.mark.asyncio
    async def test_connection_manager_scalability(self):
        """Test connection manager with many connections."""
        config = {"max_connections": 1000}
        connection_manager = WebSocketConnectionManager(config)

        # Create many connections
        connections = []
        for i in range(100):
            mock_ws = Mock()
            mock_ws.send = AsyncMock()
            conn_id = await connection_manager.connect(mock_ws, f"/ws/test_{i}")
            connections.append(conn_id)

        assert len(connection_manager.connections) == 100

        # Test broadcast performance
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/test_1",
            data={},
        )

        # Authenticate and subscribe one connection
        auth_request = ClientAuthRequest(api_key="test-key-12345")
        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False
            await connection_manager.authenticate_client(connections[0], auth_request)

        subscription = ClientSubscription(endpoint="/ws/test_1")
        await connection_manager.subscribe_client(connections[0], subscription)

        # Broadcast should be efficient
        sent_count = await connection_manager.broadcast_to_endpoint(
            "/ws/test_1", message
        )
        assert sent_count == 1

    def test_message_serialization_performance(self):
        """Test message serialization performance."""
        # Create large message
        large_data = {"data": "x" * 10000, "items": list(range(1000))}

        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data=large_data,
        )

        # Serialization should handle large data
        json_str = message.to_json()
        assert len(json_str) > 10000

        # Deserialization should work
        restored_message = WebSocketMessage.from_json(json_str)
        assert restored_message.data == large_data


class TestWebSocketAPISecurityFeatures:
    """Test WebSocket API security features."""

    @pytest.mark.asyncio
    async def test_api_key_authentication(self):
        """Test API key authentication security."""
        connection_manager = WebSocketConnectionManager()
        mock_websocket = Mock()

        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        # Test with correct API key
        auth_request = ClientAuthRequest(api_key="correct-api-key-12345")

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = True
            mock_config.return_value.api.api_key = "correct-api-key-12345"

            result = await connection_manager.authenticate_client(
                connection_id, auth_request
            )
            assert result is True

        # Test with incorrect API key
        auth_request = ClientAuthRequest(api_key="wrong-api-key-12345")

        connection_id2 = await connection_manager.connect(Mock(), "/ws/test2")

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = True
            mock_config.return_value.api.api_key = "correct-api-key-12345"

            with pytest.raises(WebSocketAuthenticationError):
                await connection_manager.authenticate_client(
                    connection_id2, auth_request
                )

    @pytest.mark.asyncio
    async def test_room_access_control(self):
        """Test room-based access control."""
        connection_manager = WebSocketConnectionManager()
        mock_websocket = Mock()

        connection_id = await connection_manager.connect(mock_websocket, "/ws/test")

        # Authenticate with limited room access
        auth_request = ClientAuthRequest(
            api_key="test-key-12345", room_filters=["living_room", "bedroom"]
        )

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False
            await connection_manager.authenticate_client(connection_id, auth_request)

        # Should allow access to permitted rooms
        subscription1 = ClientSubscription(
            endpoint="/ws/room/living_room", room_id="living_room"
        )
        result = await connection_manager.subscribe_client(connection_id, subscription1)
        assert result is True

        # Should deny access to non-permitted rooms
        subscription2 = ClientSubscription(
            endpoint="/ws/room/kitchen", room_id="kitchen"
        )
        with pytest.raises(WebSocketValidationError):
            await connection_manager.subscribe_client(connection_id, subscription2)

    def test_rate_limiting_security(self):
        """Test rate limiting as security feature."""
        connection = ClientConnection(
            connection_id="test", websocket=Mock(), endpoint="/ws/test"
        )

        # Should handle rapid message attempts
        for i in range(100):
            connection.increment_message_count()

        # Should be rate limited
        assert connection.is_rate_limited(50) is True

        # Should apply rate limit timeout
        connection.apply_rate_limit(60)
        assert connection.rate_limited_until is not None

        # Should still be rate limited even after message count reset
        connection.message_count = 0
        assert connection.is_rate_limited(50) is True

    def test_input_validation_security(self):
        """Test input validation for security."""
        # Test endpoint validation
        invalid_endpoints = [
            "/ws/../admin",
            "/ws/../../etc/passwd",
            "/ws/room/../../secrets",
            "javascript:alert('xss')",
        ]

        for endpoint in invalid_endpoints:
            with pytest.raises(ValueError):
                ClientSubscription(endpoint=endpoint)

        # Test API key validation
        invalid_keys = [
            "",  # Empty
            "short",  # Too short
            "\x00\x01\x02",  # Binary data
        ]

        for key in invalid_keys:
            with pytest.raises(ValueError):
                ClientAuthRequest(api_key=key)
