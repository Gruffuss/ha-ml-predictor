"""
Integration tests for the WebSocket API system.

These tests validate the WebSocket API's integration with the TrackingManager
and ensure proper functionality of authentication, subscriptions, and real-time updates.
"""

import asyncio
from datetime import datetime, timedelta
import json
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest
import websockets
from websockets.server import WebSocketServerProtocol

from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.config import get_config
from src.integration.websocket_api import (
    ClientConnection,
    MessageType,
    WebSocketAPIServer,
    WebSocketConnectionManager,
    WebSocketMessage,
)
from src.models.base.predictor import PredictionResult


class TestWebSocketAPIIntegration:
    """Test WebSocket API integration with TrackingManager."""

    @pytest.fixture
    async def tracking_manager(self):
        """Create a TrackingManager instance for testing."""
        config = TrackingConfig(
            enabled=True,
            websocket_api_enabled=True,
            websocket_api_host="127.0.0.1",
            websocket_api_port=8767,  # Different port for testing
            websocket_api_max_connections=10,
            websocket_api_max_messages_per_minute=100,
        )

        # Mock dependencies
        tracking_manager = TrackingManager(
            config=config,
            database_manager=MagicMock(),
            model_registry={},
            feature_engineering_engine=MagicMock(),
            notification_callbacks=[],
            mqtt_integration_manager=MagicMock(),
        )

        return tracking_manager

    @pytest.fixture
    def websocket_config(self):
        """WebSocket API configuration for testing."""
        return {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8767,
            "max_connections": 10,
            "max_messages_per_minute": 100,
            "heartbeat_interval_seconds": 5,
            "connection_timeout_seconds": 30,
            "acknowledgment_timeout_seconds": 10,
        }

    @pytest.fixture
    async def websocket_server(self, tracking_manager, websocket_config):
        """Create WebSocket API server for testing."""
        server = WebSocketAPIServer(
            tracking_manager=tracking_manager, config=websocket_config
        )
        await server.initialize()
        await server.start()

        yield server

        await server.stop()

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection for testing."""
        websocket = MagicMock(spec=WebSocketServerProtocol)
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_websocket_api_initialization(
        self, tracking_manager, websocket_config
    ):
        """Test WebSocket API server initialization."""
        server = WebSocketAPIServer(
            tracking_manager=tracking_manager, config=websocket_config
        )

        assert server.tracking_manager is tracking_manager
        assert server.enabled is True
        assert server.host == "127.0.0.1"
        assert server.port == 8767

        await server.initialize()
        assert len(server._background_tasks) > 0

        await server.stop()

    @pytest.mark.asyncio
    async def test_connection_manager_basic_operations(self, websocket_config):
        """Test basic connection management operations."""
        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        # Test connection
        connection_id = await manager.connect(
            mock_websocket, "/ws/predictions"
        )
        assert connection_id in manager.connections
        assert manager.stats.active_connections == 1
        assert manager.stats.total_connections == 1

        # Test disconnection
        await manager.disconnect(connection_id)
        assert connection_id not in manager.connections
        assert manager.stats.active_connections == 0

    @pytest.mark.asyncio
    async def test_client_authentication(self, websocket_config):
        """Test client authentication process."""
        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        # Connect client
        connection_id = await manager.connect(
            mock_websocket, "/ws/predictions"
        )

        # Mock authentication request
        from src.integration.websocket_api import ClientAuthRequest

        with patch("src.core.config.get_config") as mock_get_config:
            mock_config = MagicMock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "test-api-key"
            mock_get_config.return_value = mock_config

            auth_request = ClientAuthRequest(
                api_key="test-api-key",
                client_name="TestClient",
                capabilities=["predictions", "alerts"],
                room_filters=["living_room"],
            )

            # Test successful authentication
            success = await manager.authenticate_client(
                connection_id, auth_request
            )
            assert success is True

            connection = manager.connections[connection_id]
            assert connection.authenticated is True
            assert connection.client_name == "TestClient"
            assert "predictions" in connection.capabilities
            assert "living_room" in connection.room_filters

    @pytest.mark.asyncio
    async def test_subscription_management(self, websocket_config):
        """Test client subscription management."""
        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        # Connect and authenticate client
        connection_id = await manager.connect(
            mock_websocket, "/ws/predictions"
        )
        connection = manager.connections[connection_id]
        connection.authenticated = True
        connection.room_filters.add("living_room")

        # Test subscription
        from src.integration.websocket_api import ClientSubscription

        subscription = ClientSubscription(
            endpoint="/ws/predictions", room_id="living_room", filters={}
        )

        success = await manager.subscribe_client(connection_id, subscription)
        assert success is True
        assert "/ws/predictions" in connection.subscriptions
        assert "living_room" in connection.room_subscriptions

    @pytest.mark.asyncio
    async def test_message_broadcasting(self, websocket_config):
        """Test message broadcasting to subscribed clients."""
        manager = WebSocketConnectionManager(websocket_config)

        # Create multiple mock clients
        clients = []
        for i in range(3):
            mock_websocket = MagicMock(spec=WebSocketServerProtocol)
            mock_websocket.send = AsyncMock()
            connection_id = await manager.connect(
                mock_websocket, "/ws/predictions"
            )

            # Authenticate and subscribe
            connection = manager.connections[connection_id]
            connection.authenticated = True
            connection.subscriptions.add("/ws/predictions")
            connection.room_subscriptions.add("living_room")

            clients.append((connection_id, mock_websocket))

        # Create test message
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.utcnow(),
            endpoint="/ws/predictions",
            room_id="living_room",
            data={"test": "data"},
        )

        # Broadcast message
        sent_count = await manager.broadcast_to_endpoint(
            "/ws/predictions", message, "living_room"
        )

        # Verify all clients received the message
        assert sent_count == 3
        for _, mock_websocket in clients:
            mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_rate_limiting(self, websocket_config):
        """Test rate limiting functionality."""
        # Set low rate limit for testing
        websocket_config["max_messages_per_minute"] = 2

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock()

        connection_id = await manager.connect(
            mock_websocket, "/ws/predictions"
        )
        connection = manager.connections[connection_id]
        connection.authenticated = True

        # Create test message
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.utcnow(),
            endpoint="/ws/predictions",
            data={"test": "data"},
        )

        # Send messages up to limit
        assert await manager.send_message(connection_id, message) is True
        assert await manager.send_message(connection_id, message) is True

        # Next message should be rate limited
        assert await manager.send_message(connection_id, message) is False
        assert connection.is_rate_limited(2) is True

    @pytest.mark.asyncio
    async def test_prediction_publishing(
        self, websocket_server, websocket_config
    ):
        """Test publishing prediction updates via WebSocket API."""
        # Create mock prediction result
        prediction_result = PredictionResult(
            predicted_time=datetime.utcnow() + timedelta(minutes=30),
            confidence_score=0.85,
            transition_type="occupied",
            model_type="ensemble",
            model_version="1.0.0",
            features_used=["time_since_last", "hour_of_day"],
            prediction_metadata={"room_id": "living_room"},
        )

        # Mock some connected clients
        mock_clients = []
        for i in range(2):
            mock_websocket = MagicMock(spec=WebSocketServerProtocol)
            mock_websocket.send = AsyncMock()
            connection_id = await websocket_server.connection_manager.connect(
                mock_websocket, "/ws/predictions"
            )

            connection = websocket_server.connection_manager.connections[
                connection_id
            ]
            connection.authenticated = True
            connection.subscriptions.add("/ws/predictions")

            mock_clients.append((connection_id, mock_websocket))

        # Publish prediction update
        result = await websocket_server.publish_prediction_update(
            prediction_result, "living_room", "vacant"
        )

        # Verify successful publishing
        assert result["success"] is True
        assert result["clients_notified"] > 0

        # Verify clients received messages
        for _, mock_websocket in mock_clients:
            mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_system_status_publishing(self, websocket_server):
        """Test publishing system status updates."""
        # Mock connected client
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock()
        connection_id = await websocket_server.connection_manager.connect(
            mock_websocket, "/ws/system-status"
        )

        connection = websocket_server.connection_manager.connections[
            connection_id
        ]
        connection.authenticated = True
        connection.subscriptions.add("/ws/system-status")

        # Publish system status
        status_data = {
            "system_health": "healthy",
            "uptime_seconds": 3600,
            "active_connections": 5,
            "predictions_per_minute": 12,
        }

        result = await websocket_server.publish_system_status_update(
            status_data
        )

        # Verify successful publishing
        assert result["success"] is True
        assert result["clients_notified"] == 1

        # Verify client received message
        mock_websocket.send.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_publishing_with_acknowledgment(
        self, websocket_server
    ):
        """Test publishing alert notifications that require acknowledgment."""
        # Mock connected client
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock()
        connection_id = await websocket_server.connection_manager.connect(
            mock_websocket, "/ws/alerts"
        )

        connection = websocket_server.connection_manager.connections[
            connection_id
        ]
        connection.authenticated = True
        connection.subscriptions.add("/ws/alerts")

        # Publish alert
        alert_data = {
            "alert_type": "accuracy_degradation",
            "severity": "warning",
            "message": "Model accuracy below threshold",
            "room_id": "bedroom",
            "current_accuracy": 0.65,
            "threshold": 0.70,
        }

        result = await websocket_server.publish_alert_notification(
            alert_data, "bedroom"
        )

        # Verify successful publishing
        assert result["success"] is True
        assert result["clients_notified"] == 1

        # Verify client received message
        mock_websocket.send.assert_called_once()

        # Verify message requires acknowledgment
        sent_args = mock_websocket.send.call_args[0]
        sent_message = json.loads(sent_args[0])
        assert sent_message["requires_ack"] is True

    @pytest.mark.asyncio
    async def test_heartbeat_mechanism(self, websocket_server):
        """Test heartbeat functionality."""
        # Mock connected client
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock()
        connection_id = await websocket_server.connection_manager.connect(
            mock_websocket, "/ws/predictions"
        )

        connection = websocket_server.connection_manager.connections[
            connection_id
        ]
        connection.authenticated = True

        # Send heartbeat
        success = await websocket_server.connection_manager.send_heartbeat(
            connection_id
        )

        # Verify heartbeat sent
        assert success is True
        mock_websocket.send.assert_called_once()

        # Verify heartbeat message format
        sent_args = mock_websocket.send.call_args[0]
        sent_message = json.loads(sent_args[0])
        assert sent_message["message_type"] == "heartbeat"
        assert "server_time" in sent_message["data"]
        assert "connection_uptime" in sent_message["data"]

    @pytest.mark.asyncio
    async def test_tracking_manager_integration(self, tracking_manager):
        """Test WebSocket API integration with TrackingManager."""
        # Initialize tracking manager (which should start WebSocket API)
        with patch(
            "src.integration.websocket_api.WebSocketAPIServer"
        ) as mock_server_class:
            mock_server = AsyncMock()
            mock_server_class.return_value = mock_server

            await tracking_manager.initialize()

            # Verify WebSocket API server was created and started
            mock_server_class.assert_called_once()
            mock_server.initialize.assert_called_once()
            mock_server.start.assert_called_once()

            # Verify tracking manager has reference to WebSocket server
            assert tracking_manager.websocket_api_server is mock_server

    @pytest.mark.asyncio
    async def test_prediction_integration_flow(self, tracking_manager):
        """Test complete prediction flow through WebSocket API."""
        # Mock WebSocket API server
        mock_websocket_server = AsyncMock()
        mock_websocket_server.publish_prediction_update = AsyncMock(
            return_value={"success": True, "clients_notified": 2}
        )
        tracking_manager.websocket_api_server = mock_websocket_server
        tracking_manager.config.websocket_api_enabled = True

        # Create mock prediction result
        prediction_result = PredictionResult(
            predicted_time=datetime.utcnow() + timedelta(minutes=45),
            confidence_score=0.92,
            transition_type="vacant",
            model_type="ensemble",
            model_version="1.0.0",
            features_used=["temporal", "sequential"],
            prediction_metadata={"room_id": "kitchen"},
        )

        # Record prediction (should trigger WebSocket publishing)
        await tracking_manager.record_prediction(prediction_result)

        # Verify WebSocket API was called
        mock_websocket_server.publish_prediction_update.assert_called_once()
        call_args = mock_websocket_server.publish_prediction_update.call_args
        assert call_args[1]["room_id"] == "kitchen"
        assert call_args[0][0] == prediction_result

    def test_websocket_api_status_reporting(self, tracking_manager):
        """Test WebSocket API status reporting."""
        # Mock WebSocket API server with stats
        mock_websocket_server = MagicMock()
        mock_websocket_server.get_server_stats.return_value = {
            "server_running": True,
            "host": "127.0.0.1",
            "port": 8766,
            "connection_stats": {
                "active_connections": 15,
                "authenticated_connections": 12,
                "total_connections": 150,
                "predictions_connections": 8,
                "system_status_connections": 3,
                "alerts_connections": 4,
                "room_specific_connections": 2,
                "total_messages_sent": 5000,
                "total_messages_received": 800,
                "rate_limited_clients": 0,
                "authentication_failures": 2,
            },
            "tracking_manager_integrated": True,
        }

        tracking_manager.websocket_api_server = mock_websocket_server
        tracking_manager.config.websocket_api_enabled = True

        # Get status
        status = tracking_manager.get_websocket_api_status()

        # Verify status fields
        assert status["enabled"] is True
        assert status["running"] is True
        assert status["host"] == "127.0.0.1"
        assert status["port"] == 8766
        assert status["active_connections"] == 15
        assert status["authenticated_connections"] == 12
        assert status["total_messages_sent"] == 5000
        assert status["tracking_manager_integrated"] is True

    @pytest.mark.asyncio
    async def test_error_handling_and_recovery(self, websocket_config):
        """Test error handling and recovery mechanisms."""
        manager = WebSocketConnectionManager(websocket_config)

        # Mock websocket that fails on send
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock(
            side_effect=Exception("Connection lost")
        )

        connection_id = await manager.connect(
            mock_websocket, "/ws/predictions"
        )
        connection = manager.connections[connection_id]
        connection.authenticated = True

        # Create test message
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.utcnow(),
            endpoint="/ws/predictions",
            data={"test": "data"},
        )

        # Attempt to send message (should fail gracefully)
        success = await manager.send_message(connection_id, message)

        # Verify failure was handled gracefully
        assert success is False
        assert manager.stats.failed_message_deliveries > 0

    @pytest.mark.asyncio
    async def test_connection_cleanup(self, websocket_config):
        """Test automatic connection cleanup."""
        # Set short timeout for testing
        websocket_config["connection_timeout_seconds"] = 1

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        connection_id = await manager.connect(
            mock_websocket, "/ws/predictions"
        )
        connection = manager.connections[connection_id]

        # Simulate old last activity
        connection.last_activity = datetime.utcnow() - timedelta(seconds=2)

        # Simulate cleanup process
        current_time = datetime.utcnow()
        stale_connections = [
            conn_id
            for conn_id, conn in manager.connections.items()
            if (current_time - conn.last_activity).total_seconds()
            > manager.connection_timeout
        ]

        # Verify connection identified as stale
        assert connection_id in stale_connections

    def test_message_serialization(self):
        """Test WebSocket message serialization and deserialization."""
        original_message = WebSocketMessage(
            message_id="test-123",
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.utcnow(),
            endpoint="/ws/predictions",
            room_id="living_room",
            data={"room_name": "Living Room", "confidence": 0.85},
            requires_ack=True,
        )

        # Serialize to JSON
        json_str = original_message.to_json()

        # Deserialize from JSON
        deserialized_message = WebSocketMessage.from_json(json_str)

        # Verify all fields match
        assert deserialized_message.message_id == original_message.message_id
        assert (
            deserialized_message.message_type == original_message.message_type
        )
        assert deserialized_message.endpoint == original_message.endpoint
        assert deserialized_message.room_id == original_message.room_id
        assert deserialized_message.data == original_message.data
        assert (
            deserialized_message.requires_ack == original_message.requires_ack
        )
