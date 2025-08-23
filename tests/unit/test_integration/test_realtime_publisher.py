"""
Unit tests for real-time prediction publishing system.

Tests RealtimePublishingSystem, WebSocketConnectionManager, SSEConnectionManager,
and related components for broadcasting predictions to multiple channels.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

import pytest

from src.core.config import MQTTConfig, RoomConfig
from src.integration.mqtt_publisher import MQTTPublishResult
from src.integration.realtime_publisher import (
    ClientConnection,
    PublishingChannel,
    PublishingMetrics,
    RealtimePredictionEvent,
    RealtimePublishingSystem,
    SSEConnectionManager,
    WebSocketConnectionManager,
)
from src.models.base.predictor import PredictionResult


class TestClientConnection:
    """Test ClientConnection dataclass."""

    def test_client_connection_initialization(self):
        """Test basic initialization of client connection."""
        now = datetime.now(timezone.utc)
        connection = ClientConnection(
            connection_id="test_client_123",
            client_type="websocket",
            connected_at=now,
            last_activity=now,
            room_subscriptions={"room1", "room2"},
            metadata={"user_agent": "test_browser"},
        )

        assert connection.connection_id == "test_client_123"
        assert connection.client_type == "websocket"
        assert connection.connected_at == now
        assert connection.last_activity == now
        assert connection.room_subscriptions == {"room1", "room2"}
        assert connection.metadata["user_agent"] == "test_browser"

    def test_update_activity(self):
        """Test updating last activity timestamp."""
        now = datetime.now(timezone.utc)
        connection = ClientConnection(
            connection_id="test_client",
            client_type="sse",
            connected_at=now,
            last_activity=now,
            room_subscriptions=set(),
            metadata={},
        )

        original_activity = connection.last_activity

        # Small delay to ensure time difference
        import time

        time.sleep(0.01)

        connection.update_activity()

        assert connection.last_activity > original_activity


class TestPublishingMetrics:
    """Test PublishingMetrics dataclass."""

    def test_metrics_initialization(self):
        """Test metrics initialization with defaults."""
        metrics = PublishingMetrics()

        assert metrics.total_predictions_published == 0
        assert metrics.mqtt_publishes == 0
        assert metrics.websocket_publishes == 0
        assert metrics.sse_publishes == 0
        assert metrics.active_websocket_connections == 0
        assert metrics.active_sse_connections == 0
        assert metrics.broadcast_errors == 0
        assert len(metrics.channel_errors) == len(PublishingChannel)

    def test_metrics_post_init(self):
        """Test post-init processing of channel errors."""
        metrics = PublishingMetrics(
            total_predictions_published=10,
            mqtt_publishes=5,
        )

        # Check that channel_errors was initialized
        assert "mqtt" in metrics.channel_errors
        assert "websocket" in metrics.channel_errors
        assert "sse" in metrics.channel_errors
        assert all(count == 0 for count in metrics.channel_errors.values())


class TestRealtimePredictionEvent:
    """Test RealtimePredictionEvent dataclass and serialization."""

    def test_event_initialization(self):
        """Test event initialization."""
        now = datetime.now(timezone.utc)
        event_id = str(uuid.uuid4())

        event = RealtimePredictionEvent(
            event_id=event_id,
            event_type="prediction",
            timestamp=now,
            room_id="living_room",
            data={"prediction": "occupied", "confidence": 0.85},
        )

        assert event.event_id == event_id
        assert event.event_type == "prediction"
        assert event.timestamp == now
        assert event.room_id == "living_room"
        assert event.data["prediction"] == "occupied"
        assert event.data["confidence"] == 0.85

    def test_to_websocket_message(self):
        """Test WebSocket message formatting."""
        now = datetime.now(timezone.utc)
        event_id = "test_event_123"

        event = RealtimePredictionEvent(
            event_id=event_id,
            event_type="prediction",
            timestamp=now,
            room_id="bedroom",
            data={"next_occupied_time": "2024-01-15T14:30:00Z"},
        )

        message = event.to_websocket_message()
        parsed = json.loads(message)

        assert parsed["event_id"] == event_id
        assert parsed["event_type"] == "prediction"
        assert parsed["timestamp"] == now.isoformat()
        assert parsed["room_id"] == "bedroom"
        assert parsed["data"]["next_occupied_time"] == "2024-01-15T14:30:00Z"

    def test_to_sse_message(self):
        """Test SSE message formatting."""
        event = RealtimePredictionEvent(
            event_id="sse_event_456",
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            room_id=None,
            data={"status": "healthy", "uptime": 3600},
        )

        message = event.to_sse_message()

        assert "id: sse_event_456\n" in message
        assert "event: system_status\n" in message
        assert '"status": "healthy"' in message
        assert message.endswith("\n\n")


class TestWebSocketConnectionManager:
    """Test WebSocket connection management."""

    @pytest.fixture
    def ws_manager(self):
        """Create WebSocketConnectionManager instance."""
        return WebSocketConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = AsyncMock()
        websocket.send = AsyncMock()
        return websocket

    async def test_connect_websocket(self, ws_manager, mock_websocket):
        """Test WebSocket connection registration."""
        client_id = await ws_manager.connect(mock_websocket)

        assert client_id in ws_manager.connections
        assert ws_manager.connections[client_id] == mock_websocket
        assert client_id in ws_manager.client_metadata
        assert ws_manager.client_metadata[client_id].client_type == "websocket"

    async def test_connect_websocket_with_custom_id(self, ws_manager, mock_websocket):
        """Test WebSocket connection with custom client ID."""
        custom_id = "custom_client_123"

        client_id = await ws_manager.connect(mock_websocket, custom_id)

        assert client_id == custom_id
        assert custom_id in ws_manager.connections

    async def test_disconnect_websocket(self, ws_manager, mock_websocket):
        """Test WebSocket disconnection."""
        client_id = await ws_manager.connect(mock_websocket)

        assert client_id in ws_manager.connections

        await ws_manager.disconnect(client_id)

        assert client_id not in ws_manager.connections
        assert client_id not in ws_manager.client_metadata

    async def test_subscribe_to_room(self, ws_manager, mock_websocket):
        """Test room subscription."""
        client_id = await ws_manager.connect(mock_websocket)

        await ws_manager.subscribe_to_room(client_id, "kitchen")

        metadata = ws_manager.client_metadata[client_id]
        assert "kitchen" in metadata.room_subscriptions

    async def test_unsubscribe_from_room(self, ws_manager, mock_websocket):
        """Test room unsubscription."""
        client_id = await ws_manager.connect(mock_websocket)
        await ws_manager.subscribe_to_room(client_id, "kitchen")
        await ws_manager.subscribe_to_room(client_id, "bedroom")

        await ws_manager.unsubscribe_from_room(client_id, "kitchen")

        metadata = ws_manager.client_metadata[client_id]
        assert "kitchen" not in metadata.room_subscriptions
        assert "bedroom" in metadata.room_subscriptions

    async def test_broadcast_to_room(self, ws_manager):
        """Test broadcasting to room subscribers."""
        # Create mock websockets
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        ws3 = AsyncMock()

        # Connect clients
        client1 = await ws_manager.connect(ws1)
        client2 = await ws_manager.connect(ws2)
        client3 = await ws_manager.connect(ws3)

        # Subscribe to rooms
        await ws_manager.subscribe_to_room(client1, "living_room")
        await ws_manager.subscribe_to_room(client2, "living_room")
        await ws_manager.subscribe_to_room(client3, "kitchen")  # Different room

        # Create event
        event = RealtimePredictionEvent(
            event_id="test_broadcast",
            event_type="prediction",
            timestamp=datetime.now(timezone.utc),
            room_id="living_room",
            data={"test": "data"},
        )

        # Broadcast
        sent_count = await ws_manager.broadcast_to_room("living_room", event)

        # Check results
        assert sent_count == 2  # Only clients subscribed to living_room
        ws1.send.assert_called_once()
        ws2.send.assert_called_once()
        ws3.send.assert_not_called()  # Not subscribed to living_room

    async def test_broadcast_to_room_with_failed_connection(self, ws_manager):
        """Test broadcasting with a failed connection."""
        # Create websockets with one that will fail
        ws_good = AsyncMock()
        ws_bad = AsyncMock()
        ws_bad.send.side_effect = Exception("Connection failed")

        # Connect clients
        client_good = await ws_manager.connect(ws_good)
        client_bad = await ws_manager.connect(ws_bad)

        # Subscribe both to same room
        await ws_manager.subscribe_to_room(client_good, "test_room")
        await ws_manager.subscribe_to_room(client_bad, "test_room")

        # Create event
        event = RealtimePredictionEvent(
            event_id="test_fail",
            event_type="test",
            timestamp=datetime.now(timezone.utc),
            room_id="test_room",
            data={},
        )

        # Broadcast
        sent_count = await ws_manager.broadcast_to_room("test_room", event)

        # Should succeed for good connection, fail for bad one
        assert sent_count == 1
        assert client_bad not in ws_manager.connections  # Should be disconnected

    async def test_broadcast_to_all(self, ws_manager):
        """Test broadcasting to all connected clients."""
        # Connect multiple clients
        clients = []
        websockets = []
        for i in range(3):
            ws = AsyncMock()
            client_id = await ws_manager.connect(ws)
            clients.append(client_id)
            websockets.append(ws)

        # Create event
        event = RealtimePredictionEvent(
            event_id="broadcast_all",
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            room_id=None,
            data={"status": "healthy"},
        )

        # Broadcast to all
        sent_count = await ws_manager.broadcast_to_all(event)

        # Check all clients received message
        assert sent_count == 3
        for ws in websockets:
            ws.send.assert_called_once()

    def test_get_connection_stats(self, ws_manager):
        """Test connection statistics."""
        stats = ws_manager.get_connection_stats()

        assert "total_active_connections" in stats
        assert "connections_by_room" in stats
        assert "oldest_connection" in stats
        assert "most_recent_activity" in stats

        # With no connections, should be empty
        assert stats["total_active_connections"] == 0
        assert stats["oldest_connection"] is None
        assert stats["most_recent_activity"] is None


class TestSSEConnectionManager:
    """Test Server-Sent Events connection management."""

    @pytest.fixture
    def sse_manager(self):
        """Create SSEConnectionManager instance."""
        return SSEConnectionManager()

    async def test_connect_sse(self, sse_manager):
        """Test SSE connection registration."""
        client_id, queue = await sse_manager.connect()

        assert client_id in sse_manager.connections
        assert sse_manager.connections[client_id] == queue
        assert isinstance(queue, asyncio.Queue)
        assert client_id in sse_manager.client_metadata
        assert sse_manager.client_metadata[client_id].client_type == "sse"

    async def test_connect_sse_with_custom_id(self, sse_manager):
        """Test SSE connection with custom client ID."""
        custom_id = "sse_client_789"

        client_id, queue = await sse_manager.connect(custom_id)

        assert client_id == custom_id
        assert custom_id in sse_manager.connections

    async def test_disconnect_sse(self, sse_manager):
        """Test SSE disconnection."""
        client_id, _ = await sse_manager.connect()

        assert client_id in sse_manager.connections

        await sse_manager.disconnect(client_id)

        assert client_id not in sse_manager.connections
        assert client_id not in sse_manager.client_metadata

    async def test_sse_subscribe_to_room(self, sse_manager):
        """Test SSE room subscription."""
        client_id, _ = await sse_manager.connect()

        await sse_manager.subscribe_to_room(client_id, "bathroom")

        metadata = sse_manager.client_metadata[client_id]
        assert "bathroom" in metadata.room_subscriptions

    async def test_broadcast_to_room_sse(self, sse_manager):
        """Test SSE broadcasting to room subscribers."""
        # Connect multiple SSE clients
        client1, queue1 = await sse_manager.connect()
        client2, queue2 = await sse_manager.connect()
        client3, queue3 = await sse_manager.connect()

        # Subscribe to different rooms
        await sse_manager.subscribe_to_room(client1, "office")
        await sse_manager.subscribe_to_room(client2, "office")
        await sse_manager.subscribe_to_room(client3, "garage")

        # Create event
        event = RealtimePredictionEvent(
            event_id="sse_broadcast",
            event_type="prediction",
            timestamp=datetime.now(timezone.utc),
            room_id="office",
            data={"prediction": "vacant"},
        )

        # Broadcast to office
        sent_count = await sse_manager.broadcast_to_room("office", event)

        # Check results
        assert sent_count == 2

        # Check that messages were queued for office subscribers
        assert not queue1.empty()
        assert not queue2.empty()
        assert queue3.empty()  # Not subscribed to office

    async def test_broadcast_to_all_sse(self, sse_manager):
        """Test SSE broadcasting to all clients."""
        # Connect multiple SSE clients
        clients_queues = []
        for i in range(3):
            client_id, queue = await sse_manager.connect()
            clients_queues.append((client_id, queue))

        # Create event
        event = RealtimePredictionEvent(
            event_id="sse_all",
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            room_id=None,
            data={"system": "online"},
        )

        # Broadcast to all
        sent_count = await sse_manager.broadcast_to_all(event)

        # Check all clients received message
        assert sent_count == 3
        for _, queue in clients_queues:
            assert not queue.empty()

    def test_sse_get_connection_stats(self, sse_manager):
        """Test SSE connection statistics."""
        stats = sse_manager.get_connection_stats()

        assert "total_active_connections" in stats
        assert "connections_by_room" in stats
        assert "oldest_connection" in stats
        assert "most_recent_activity" in stats

        # With no connections
        assert stats["total_active_connections"] == 0


class TestRealtimePublishingSystem:
    """Test main RealtimePublishingSystem class."""

    @pytest.fixture
    def mock_mqtt_config(self):
        """Create mock MQTT config."""
        return MQTTConfig(
            broker="test-mqtt",
            port=1883,
            topic_prefix="test/predictions",
        )

    @pytest.fixture
    def mock_rooms(self):
        """Create mock room configurations."""
        return {
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={"presence": {"main": "binary_sensor.living_room_presence"}},
            ),
            "kitchen": RoomConfig(
                room_id="kitchen",
                name="Kitchen",
                sensors={"motion": "binary_sensor.kitchen_motion"},
            ),
        }

    @pytest.fixture
    def mock_prediction_publisher(self):
        """Create mock prediction publisher."""
        publisher = AsyncMock()
        publisher.publish_prediction = AsyncMock(
            return_value=MQTTPublishResult(
                success=True,
                topic="test/topic",
                payload_size=100,
                publish_time=datetime.now(timezone.utc),
                error_message=None,
            )
        )
        return publisher

    @pytest.fixture
    def realtime_system(self, mock_mqtt_config, mock_rooms, mock_prediction_publisher):
        """Create RealtimePublishingSystem instance."""
        return RealtimePublishingSystem(
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
            prediction_publisher=mock_prediction_publisher,
            enabled_channels=[
                PublishingChannel.MQTT,
                PublishingChannel.WEBSOCKET,
                PublishingChannel.SSE,
            ],
        )

    def test_system_initialization(self, realtime_system, mock_mqtt_config, mock_rooms):
        """Test system initialization."""
        assert realtime_system.config == mock_mqtt_config
        assert realtime_system.rooms == mock_rooms
        assert len(realtime_system.enabled_channels) == 3
        assert isinstance(realtime_system.websocket_manager, WebSocketConnectionManager)
        assert isinstance(realtime_system.sse_manager, SSEConnectionManager)
        assert isinstance(realtime_system.metrics, PublishingMetrics)
        assert not realtime_system._publishing_active

    @patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"})
    async def test_initialize_system(self, realtime_system):
        """Test system initialization."""
        await realtime_system.initialize()

        assert realtime_system._publishing_active
        # No background tasks in test environment
        assert len(realtime_system._background_tasks) == 0

    async def test_shutdown_system(self, realtime_system):
        """Test system shutdown."""
        # Initialize first
        await realtime_system.initialize()
        assert realtime_system._publishing_active

        # Then shutdown
        await realtime_system.shutdown()

        assert not realtime_system._publishing_active
        assert realtime_system._shutdown_event.is_set()

    async def test_publish_prediction_all_channels(
        self, realtime_system, mock_prediction_publisher
    ):
        """Test publishing prediction across all channels."""
        # Create mock prediction result
        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            confidence=0.85,
            model_type="ensemble",
            features_used=["temporal", "sequential"],
            prediction_horizon_minutes=30,
            raw_predictions={"lstm": 0.82, "xgboost": 0.88},
        )

        # Mock connections for WebSocket and SSE
        ws_mock = AsyncMock()
        await realtime_system.websocket_manager.connect(ws_mock)
        await realtime_system.websocket_manager.subscribe_to_room(
            list(realtime_system.websocket_manager.connections.keys())[0], "living_room"
        )

        sse_client, _ = await realtime_system.sse_manager.connect()
        await realtime_system.sse_manager.subscribe_to_room(sse_client, "living_room")

        # Publish prediction
        results = await realtime_system.publish_prediction(
            prediction_result, "living_room", "vacant"
        )

        # Check results for all channels
        assert "mqtt" in results
        assert results["mqtt"]["success"] is True
        assert "websocket" in results
        assert results["websocket"]["success"] is True
        assert results["websocket"]["clients_notified"] == 1
        assert "sse" in results
        assert results["sse"]["success"] is True
        assert results["sse"]["clients_notified"] == 1

        # Check metrics updated
        assert realtime_system.metrics.total_predictions_published == 1
        assert realtime_system.metrics.mqtt_publishes == 1
        assert realtime_system.metrics.websocket_publishes == 1
        assert realtime_system.metrics.sse_publishes == 1

    async def test_publish_prediction_mqtt_only(
        self, mock_mqtt_config, mock_rooms, mock_prediction_publisher
    ):
        """Test publishing with only MQTT enabled."""
        system = RealtimePublishingSystem(
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
            prediction_publisher=mock_prediction_publisher,
            enabled_channels=[PublishingChannel.MQTT],
        )

        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=15),
            confidence=0.75,
            model_type="lstm",
            features_used=["temporal"],
            prediction_horizon_minutes=15,
            raw_predictions={"lstm": 0.75},
        )

        results = await system.publish_prediction(
            prediction_result, "kitchen", "occupied"
        )

        # Should only have MQTT result
        assert len(results) == 1
        assert "mqtt" in results
        assert results["mqtt"]["success"] is True

        # Check metrics
        assert system.metrics.mqtt_publishes == 1
        assert system.metrics.websocket_publishes == 0
        assert system.metrics.sse_publishes == 0

    async def test_publish_prediction_mqtt_failure(
        self, realtime_system, mock_prediction_publisher
    ):
        """Test handling MQTT publish failure."""
        # Make MQTT publisher fail
        mock_prediction_publisher.publish_prediction = AsyncMock(
            return_value=MQTTPublishResult(
                success=False,
                topic="test/topic",
                payload_size=100,
                publish_time=datetime.now(timezone.utc),
                error_message="MQTT connection failed",
            )
        )

        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=20),
            confidence=0.90,
            model_type="xgboost",
            features_used=["temporal", "contextual"],
            prediction_horizon_minutes=20,
            raw_predictions={"xgboost": 0.90},
        )

        results = await realtime_system.publish_prediction(
            prediction_result, "living_room"
        )

        # MQTT should show failure
        assert "mqtt" in results
        assert results["mqtt"]["success"] is False
        assert results["mqtt"]["error"] == "MQTT connection failed"

        # Error counter should be incremented
        assert realtime_system.metrics.channel_errors["mqtt"] == 1

    async def test_publish_system_status(self, realtime_system):
        """Test publishing system status."""
        # Add some connections
        ws_mock = AsyncMock()
        await realtime_system.websocket_manager.connect(ws_mock)

        sse_client, _ = await realtime_system.sse_manager.connect()

        status_data = {
            "system_health": "good",
            "active_predictions": 5,
            "uptime_seconds": 3600,
            "connected_clients": 2,
        }

        results = await realtime_system.publish_system_status(status_data)

        # Should broadcast to WebSocket and SSE
        assert "websocket" in results
        assert results["websocket"]["success"] is True
        assert results["websocket"]["clients_notified"] == 1

        assert "sse" in results
        assert results["sse"]["success"] is True
        assert results["sse"]["clients_notified"] == 1

    async def test_handle_websocket_connection(self, realtime_system):
        """Test WebSocket connection handling."""
        mock_websocket = AsyncMock()
        mock_websocket.__aiter__ = AsyncMock(return_value=iter([]))
        mock_websocket.send = AsyncMock()

        # Handle connection (will exit immediately due to empty message iterator)
        await realtime_system.handle_websocket_connection(mock_websocket, "/ws")

        # Should have sent welcome message
        mock_websocket.send.assert_called_once()

        # Check welcome message content
        call_args = mock_websocket.send.call_args[0][0]
        message_data = json.loads(call_args)

        assert message_data["event_type"] == "connection"
        assert (
            message_data["data"]["message"]
            == "Connected to real-time prediction system"
        )
        assert "client_id" in message_data["data"]
        assert "available_rooms" in message_data["data"]

    def test_format_prediction_data(self, realtime_system):
        """Test prediction data formatting."""
        prediction_result = PredictionResult(
            predicted_time=datetime(2024, 1, 15, 14, 30, tzinfo=timezone.utc),
            confidence=0.85,
            model_type="ensemble",
            features_used=["temporal", "sequential"],
            prediction_horizon_minutes=30,
            raw_predictions={"lstm": 0.82, "xgboost": 0.88},
        )

        formatted = realtime_system._format_prediction_data(
            prediction_result, "bedroom", "vacant"
        )

        assert formatted["room_id"] == "bedroom"
        assert formatted["current_state"] == "vacant"
        assert formatted["predicted_time"] == "2024-01-15T14:30:00+00:00"
        assert formatted["confidence"] == 0.85
        assert formatted["model_type"] == "ensemble"
        assert formatted["prediction_horizon_minutes"] == 30
        assert "features_used" in formatted
        assert "raw_predictions" in formatted

    def test_get_system_stats(self, realtime_system):
        """Test getting system statistics."""
        stats = realtime_system.get_system_stats()

        assert "system_uptime_seconds" in stats
        assert "publishing_metrics" in stats
        assert "websocket_connections" in stats
        assert "sse_connections" in stats
        assert "enabled_channels" in stats
        assert "rooms_configured" in stats

        # Check basic values
        assert stats["publishing_active"] == realtime_system._publishing_active
        assert stats["enabled_channels"] == [
            channel.value for channel in realtime_system.enabled_channels
        ]
        assert stats["rooms_configured"] == len(realtime_system.rooms)


class TestBroadcastCallbacks:
    """Test broadcast callback functionality."""

    @pytest.fixture
    def system_with_callbacks(self, mock_mqtt_config, mock_rooms):
        """Create system for callback testing."""
        return RealtimePublishingSystem(
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
            enabled_channels=[PublishingChannel.WEBSOCKET],
        )

    async def test_add_broadcast_callback(self, system_with_callbacks):
        """Test adding broadcast callbacks."""
        callback_called = False
        received_event = None
        received_results = None

        def test_callback(event, results):
            nonlocal callback_called, received_event, received_results
            callback_called = True
            received_event = event
            received_results = results

        # Add callback
        system_with_callbacks.add_broadcast_callback(test_callback)

        # Create prediction
        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=10),
            confidence=0.80,
            model_type="test",
            features_used=[],
            prediction_horizon_minutes=10,
            raw_predictions={},
        )

        # Publish prediction
        await system_with_callbacks.publish_prediction(prediction_result, "living_room")

        # Check callback was called
        assert callback_called
        assert received_event is not None
        assert received_event.event_type == "prediction"
        assert received_event.room_id == "living_room"
        assert received_results is not None

    async def test_async_broadcast_callback(self, system_with_callbacks):
        """Test async broadcast callbacks."""
        callback_called = False

        async def async_callback(event, results):
            nonlocal callback_called
            await asyncio.sleep(0.01)  # Simulate async work
            callback_called = True

        # Add async callback
        system_with_callbacks.add_broadcast_callback(async_callback)

        # Create and publish prediction
        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=5),
            confidence=0.70,
            model_type="test",
            features_used=[],
            prediction_horizon_minutes=5,
            raw_predictions={},
        )

        await system_with_callbacks.publish_prediction(prediction_result, "kitchen")

        # Check callback was called
        assert callback_called

    async def test_callback_error_handling(self, system_with_callbacks):
        """Test error handling in callbacks."""

        def failing_callback(event, results):
            raise Exception("Callback failed")

        def working_callback(event, results):
            working_callback.called = True

        working_callback.called = False

        # Add both callbacks
        system_with_callbacks.add_broadcast_callback(failing_callback)
        system_with_callbacks.add_broadcast_callback(working_callback)

        # Create prediction
        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=8),
            confidence=0.65,
            model_type="test",
            features_used=[],
            prediction_horizon_minutes=8,
            raw_predictions={},
        )

        # Should not raise exception despite failing callback
        await system_with_callbacks.publish_prediction(prediction_result, "living_room")

        # Working callback should still be called
        assert working_callback.called


class TestErrorHandling:
    """Test error handling scenarios."""

    async def test_broadcast_error_metrics(self, mock_mqtt_config, mock_rooms):
        """Test error metrics tracking."""
        system = RealtimePublishingSystem(
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
            enabled_channels=[PublishingChannel.WEBSOCKET],
        )

        # Mock websocket manager to raise exception
        system.websocket_manager.broadcast_to_room = AsyncMock(
            side_effect=Exception("Broadcast failed")
        )

        prediction_result = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=12),
            confidence=0.60,
            model_type="test",
            features_used=[],
            prediction_horizon_minutes=12,
            raw_predictions={},
        )

        # Should handle error gracefully
        results = await system.publish_prediction(prediction_result, "living_room")

        # Check error handling
        assert "websocket" in results
        assert results["websocket"]["success"] is False
        assert system.metrics.channel_errors["websocket"] == 1

    async def test_connection_cleanup_on_error(self):
        """Test connection cleanup when broadcasts fail."""
        ws_manager = WebSocketConnectionManager()

        # Create failing websocket
        failing_ws = AsyncMock()
        failing_ws.send.side_effect = Exception("Send failed")

        client_id = await ws_manager.connect(failing_ws)
        await ws_manager.subscribe_to_room(client_id, "test_room")

        # Create event
        event = RealtimePredictionEvent(
            event_id="cleanup_test",
            event_type="test",
            timestamp=datetime.now(timezone.utc),
            room_id="test_room",
            data={},
        )

        # Broadcast should fail and cleanup connection
        sent_count = await ws_manager.broadcast_to_room("test_room", event)

        assert sent_count == 0
        assert client_id not in ws_manager.connections


class TestIntegrationScenarios:
    """Integration tests for complete scenarios."""

    async def test_multiple_clients_different_rooms(self):
        """Test multiple clients subscribing to different rooms."""
        system = RealtimePublishingSystem(
            mqtt_config=MQTTConfig(broker="test", port=1883, topic_prefix="test"),
            rooms={
                "room1": RoomConfig(room_id="room1", name="Room 1", sensors={}),
                "room2": RoomConfig(room_id="room2", name="Room 2", sensors={}),
            },
            enabled_channels=[PublishingChannel.WEBSOCKET, PublishingChannel.SSE],
        )

        # Connect WebSocket clients
        ws1 = AsyncMock()
        ws2 = AsyncMock()
        client_ws1 = await system.websocket_manager.connect(ws1)
        client_ws2 = await system.websocket_manager.connect(ws2)

        await system.websocket_manager.subscribe_to_room(client_ws1, "room1")
        await system.websocket_manager.subscribe_to_room(client_ws2, "room2")

        # Connect SSE clients
        client_sse1, _ = await system.sse_manager.connect()
        client_sse2, _ = await system.sse_manager.connect()

        await system.sse_manager.subscribe_to_room(client_sse1, "room1")
        await system.sse_manager.subscribe_to_room(
            client_sse2, "room1"
        )  # Both on room1

        # Publish prediction for room1
        prediction = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=25),
            confidence=0.95,
            model_type="test",
            features_used=[],
            prediction_horizon_minutes=25,
            raw_predictions={},
        )

        results = await system.publish_prediction(prediction, "room1")

        # Check targeting
        assert (
            results["websocket"]["clients_notified"] == 1
        )  # Only ws1 subscribed to room1
        assert (
            results["sse"]["clients_notified"] == 2
        )  # Both SSE clients subscribed to room1

        # Verify correct clients were called
        ws1.send.assert_called_once()
        ws2.send.assert_not_called()

    async def test_client_connection_lifecycle(self):
        """Test complete client connection lifecycle."""
        system = RealtimePublishingSystem(
            mqtt_config=MQTTConfig(broker="test", port=1883, topic_prefix="test"),
            rooms={
                "test_room": RoomConfig(
                    room_id="test_room", name="Test Room", sensors={}
                )
            },
            enabled_channels=[PublishingChannel.WEBSOCKET],
        )

        # Connect client
        ws = AsyncMock()
        client_id = await system.websocket_manager.connect(ws)

        # Subscribe to room
        await system.websocket_manager.subscribe_to_room(client_id, "test_room")

        # Publish prediction - should reach client
        prediction = PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=15),
            confidence=0.80,
            model_type="test",
            features_used=[],
            prediction_horizon_minutes=15,
            raw_predictions={},
        )

        results1 = await system.publish_prediction(prediction, "test_room")
        assert results1["websocket"]["clients_notified"] == 1

        # Disconnect client
        await system.websocket_manager.disconnect(client_id)

        # Publish again - should reach no clients
        results2 = await system.publish_prediction(prediction, "test_room")
        assert results2["websocket"]["clients_notified"] == 0
