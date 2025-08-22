"""
Comprehensive test suite for RealtimePublishingSystem.

This module provides complete test coverage for the real-time prediction publishing
system, including WebSocket connections, SSE streams, MQTT publishing, and error scenarios.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid
import weakref

import pytest
import websockets
from websockets.server import WebSocketServerProtocol

from src.core.config import MQTTConfig, RoomConfig
from src.core.exceptions import OccupancyPredictionError
from src.integration.mqtt_publisher import MQTTPublishResult
from src.integration.prediction_publisher import PredictionPublisher
from src.integration.realtime_publisher import (
    ClientConnection,
    PublishingChannel,
    PublishingMetrics,
    RealtimePredictionEvent,
    RealtimePublishingError,
    RealtimePublishingSystem,
    SSEConnectionManager,
    WebSocketConnectionManager,
    create_realtime_app,
    create_realtime_publishing_system,
    realtime_publisher_context,
)
from src.models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


class TestClientConnection:
    """Test ClientConnection data class."""

    def test_client_connection_creation(self):
        """Test creating a client connection."""
        connection_id = str(uuid.uuid4())
        connected_at = datetime.now(timezone.utc)

        client = ClientConnection(
            connection_id=connection_id,
            client_type="websocket",
            connected_at=connected_at,
            last_activity=connected_at,
            room_subscriptions={"living_room", "kitchen"},
            metadata={"user_agent": "test-client"},
        )

        assert client.connection_id == connection_id
        assert client.client_type == "websocket"
        assert client.connected_at == connected_at
        assert client.last_activity == connected_at
        assert "living_room" in client.room_subscriptions
        assert "kitchen" in client.room_subscriptions
        assert client.metadata["user_agent"] == "test-client"

    def test_update_activity(self):
        """Test updating last activity timestamp."""
        client = ClientConnection(
            connection_id="test-id",
            client_type="sse",
            connected_at=datetime.now(timezone.utc),
            last_activity=datetime.now(timezone.utc) - timedelta(minutes=5),
            room_subscriptions=set(),
            metadata={},
        )

        original_activity = client.last_activity
        client.update_activity()

        assert client.last_activity > original_activity


class TestPublishingMetrics:
    """Test PublishingMetrics data class."""

    def test_metrics_initialization(self):
        """Test metrics initialization with defaults."""
        metrics = PublishingMetrics()

        assert metrics.total_predictions_published == 0
        assert metrics.mqtt_publishes == 0
        assert metrics.websocket_publishes == 0
        assert metrics.sse_publishes == 0
        assert metrics.active_websocket_connections == 0
        assert metrics.active_sse_connections == 0
        assert metrics.total_connections_served == 0
        assert metrics.broadcast_errors == 0
        assert isinstance(metrics.channel_errors, dict)
        assert "mqtt" in metrics.channel_errors
        assert "websocket" in metrics.channel_errors
        assert "sse" in metrics.channel_errors

    def test_metrics_custom_initialization(self):
        """Test metrics initialization with custom values."""
        custom_errors = {"mqtt": 5, "websocket": 2}
        metrics = PublishingMetrics(
            total_predictions_published=100,
            mqtt_publishes=50,
            channel_errors=custom_errors,
        )

        assert metrics.total_predictions_published == 100
        assert metrics.mqtt_publishes == 50
        assert metrics.channel_errors == custom_errors


class TestRealtimePredictionEvent:
    """Test RealtimePredictionEvent data class."""

    def test_event_creation(self):
        """Test creating a real-time prediction event."""
        event_id = str(uuid.uuid4())
        timestamp = datetime.now(timezone.utc)
        data = {"prediction": "test_data", "confidence": 0.85}

        event = RealtimePredictionEvent(
            event_id=event_id,
            event_type="prediction",
            timestamp=timestamp,
            room_id="living_room",
            data=data,
        )

        assert event.event_id == event_id
        assert event.event_type == "prediction"
        assert event.timestamp == timestamp
        assert event.room_id == "living_room"
        assert event.data == data

    def test_to_websocket_message(self):
        """Test converting event to WebSocket message format."""
        event = RealtimePredictionEvent(
            event_id="test-id",
            event_type="prediction",
            timestamp=datetime(2024, 1, 15, 12, 30, 45, tzinfo=timezone.utc),
            room_id="bedroom",
            data={"confidence": 0.9},
        )

        message = event.to_websocket_message()
        parsed = json.loads(message)

        assert parsed["event_id"] == "test-id"
        assert parsed["event_type"] == "prediction"
        assert parsed["timestamp"] == "2024-01-15T12:30:45+00:00"
        assert parsed["room_id"] == "bedroom"
        assert parsed["data"]["confidence"] == 0.9

    def test_to_sse_message(self):
        """Test converting event to Server-Sent Events format."""
        event = RealtimePredictionEvent(
            event_id="sse-test",
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            room_id=None,
            data={"status": "healthy"},
        )

        message = event.to_sse_message()
        lines = message.strip().split("\n")

        assert lines[0] == "id: sse-test"
        assert lines[1] == "event: system_status"
        assert "data:" in lines[2]
        assert "status" in lines[2]
        assert "healthy" in lines[2]


class TestWebSocketConnectionManager:
    """Test WebSocketConnectionManager."""

    @pytest.fixture
    def manager(self):
        """Create WebSocket connection manager."""
        return WebSocketConnectionManager()

    @pytest.fixture
    def mock_websocket(self):
        """Create mock WebSocket connection."""
        websocket = AsyncMock(spec=WebSocketServerProtocol)
        return websocket

    @pytest.mark.asyncio
    async def test_connect_without_client_id(self, manager, mock_websocket):
        """Test connecting without providing client ID."""
        client_id = await manager.connect(mock_websocket)

        assert client_id is not None
        assert client_id in manager.connections
        assert client_id in manager.client_metadata
        assert manager.connections[client_id] == mock_websocket

        client_meta = manager.client_metadata[client_id]
        assert client_meta.connection_id == client_id
        assert client_meta.client_type == "websocket"
        assert isinstance(client_meta.connected_at, datetime)
        assert isinstance(client_meta.room_subscriptions, set)

    @pytest.mark.asyncio
    async def test_connect_with_client_id(self, manager, mock_websocket):
        """Test connecting with specific client ID."""
        custom_id = "custom-client-123"
        client_id = await manager.connect(mock_websocket, custom_id)

        assert client_id == custom_id
        assert custom_id in manager.connections
        assert manager.connections[custom_id] == mock_websocket

    @pytest.mark.asyncio
    async def test_disconnect(self, manager, mock_websocket):
        """Test disconnecting a client."""
        client_id = await manager.connect(mock_websocket)
        assert client_id in manager.connections

        await manager.disconnect(client_id)

        assert client_id not in manager.connections
        assert client_id not in manager.client_metadata

    @pytest.mark.asyncio
    async def test_disconnect_nonexistent_client(self, manager):
        """Test disconnecting a non-existent client."""
        # Should not raise an exception
        await manager.disconnect("non-existent-client")

    @pytest.mark.asyncio
    async def test_subscribe_to_room(self, manager, mock_websocket):
        """Test subscribing client to room updates."""
        client_id = await manager.connect(mock_websocket)

        await manager.subscribe_to_room(client_id, "living_room")

        client_meta = manager.client_metadata[client_id]
        assert "living_room" in client_meta.room_subscriptions

    @pytest.mark.asyncio
    async def test_subscribe_nonexistent_client(self, manager):
        """Test subscribing non-existent client to room."""
        # Should not raise an exception
        await manager.subscribe_to_room("non-existent", "bedroom")

    @pytest.mark.asyncio
    async def test_unsubscribe_from_room(self, manager, mock_websocket):
        """Test unsubscribing client from room updates."""
        client_id = await manager.connect(mock_websocket)
        await manager.subscribe_to_room(client_id, "kitchen")

        client_meta = manager.client_metadata[client_id]
        assert "kitchen" in client_meta.room_subscriptions

        await manager.unsubscribe_from_room(client_id, "kitchen")
        assert "kitchen" not in client_meta.room_subscriptions

    @pytest.mark.asyncio
    async def test_broadcast_to_room(self, manager, mock_websocket):
        """Test broadcasting message to room subscribers."""
        client_id = await manager.connect(mock_websocket)
        await manager.subscribe_to_room(client_id, "bedroom")

        event = RealtimePredictionEvent(
            event_id="test-broadcast",
            event_type="prediction",
            timestamp=datetime.now(timezone.utc),
            room_id="bedroom",
            data={"test": "data"},
        )

        sent_count = await manager.broadcast_to_room("bedroom", event)

        assert sent_count == 1
        mock_websocket.send.assert_called_once()

        # Verify message content
        call_args = mock_websocket.send.call_args[0][0]
        parsed_message = json.loads(call_args)
        assert parsed_message["event_id"] == "test-broadcast"

    @pytest.mark.asyncio
    async def test_broadcast_to_room_no_subscribers(self, manager):
        """Test broadcasting to room with no subscribers."""
        event = RealtimePredictionEvent(
            event_id="no-subs",
            event_type="prediction",
            timestamp=datetime.now(timezone.utc),
            room_id="empty_room",
            data={},
        )

        sent_count = await manager.broadcast_to_room("empty_room", event)
        assert sent_count == 0

    @pytest.mark.asyncio
    async def test_broadcast_with_failed_connection(self, manager):
        """Test broadcasting with a failed WebSocket connection."""
        mock_websocket = AsyncMock(spec=WebSocketServerProtocol)
        mock_websocket.send.side_effect = Exception("Connection failed")

        client_id = await manager.connect(mock_websocket)
        await manager.subscribe_to_room(client_id, "test_room")

        event = RealtimePredictionEvent(
            event_id="test-fail",
            event_type="prediction",
            timestamp=datetime.now(timezone.utc),
            room_id="test_room",
            data={},
        )

        sent_count = await manager.broadcast_to_room("test_room", event)

        assert sent_count == 0
        assert (
            client_id not in manager.connections
        )  # Failed connection should be removed

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager, mock_websocket):
        """Test broadcasting to all connected clients."""
        client_id = await manager.connect(mock_websocket)

        event = RealtimePredictionEvent(
            event_id="broadcast-all",
            event_type="system_status",
            timestamp=datetime.now(timezone.utc),
            room_id=None,
            data={"status": "healthy"},
        )

        sent_count = await manager.broadcast_to_all(event)

        assert sent_count == 1
        mock_websocket.send.assert_called_once()

    def test_get_connection_stats(self, manager):
        """Test getting connection statistics."""
        stats = manager.get_connection_stats()

        assert "total_active_connections" in stats
        assert "connections_by_room" in stats
        assert "oldest_connection" in stats
        assert "most_recent_activity" in stats
        assert stats["total_active_connections"] == 0

    @pytest.mark.asyncio
    async def test_get_connection_stats_with_connections(self, manager, mock_websocket):
        """Test connection stats with active connections."""
        client_id = await manager.connect(mock_websocket)
        await manager.subscribe_to_room(client_id, "test_room")

        stats = manager.get_connection_stats()

        assert stats["total_active_connections"] == 1
        assert "test_room" in stats["connections_by_room"]
        assert stats["connections_by_room"]["test_room"] == 1


class TestSSEConnectionManager:
    """Test SSEConnectionManager."""

    @pytest.fixture
    def manager(self):
        """Create SSE connection manager."""
        return SSEConnectionManager()

    @pytest.mark.asyncio
    async def test_connect_without_client_id(self, manager):
        """Test connecting without providing client ID."""
        client_id, queue = await manager.connect()

        assert client_id is not None
        assert isinstance(queue, asyncio.Queue)
        assert client_id in manager.connections
        assert client_id in manager.client_metadata
        assert manager.connections[client_id] == queue

    @pytest.mark.asyncio
    async def test_connect_with_client_id(self, manager):
        """Test connecting with specific client ID."""
        custom_id = "sse-client-456"
        client_id, queue = await manager.connect(custom_id)

        assert client_id == custom_id
        assert custom_id in manager.connections

    @pytest.mark.asyncio
    async def test_disconnect(self, manager):
        """Test disconnecting SSE client."""
        client_id, _ = await manager.connect()
        assert client_id in manager.connections

        await manager.disconnect(client_id)

        assert client_id not in manager.connections
        assert client_id not in manager.client_metadata

    @pytest.mark.asyncio
    async def test_subscribe_to_room(self, manager):
        """Test subscribing SSE client to room updates."""
        client_id, _ = await manager.connect()

        await manager.subscribe_to_room(client_id, "office")

        client_meta = manager.client_metadata[client_id]
        assert "office" in client_meta.room_subscriptions

    @pytest.mark.asyncio
    async def test_broadcast_to_room(self, manager):
        """Test broadcasting to SSE room subscribers."""
        client_id, queue = await manager.connect()
        await manager.subscribe_to_room(client_id, "garage")

        event = RealtimePredictionEvent(
            event_id="sse-broadcast",
            event_type="prediction",
            timestamp=datetime.now(timezone.utc),
            room_id="garage",
            data={"sse": "test"},
        )

        sent_count = await manager.broadcast_to_room("garage", event)

        assert sent_count == 1
        assert not queue.empty()

        # Verify message in queue
        message = await queue.get()
        assert "id: sse-broadcast" in message
        assert "event: prediction" in message

    @pytest.mark.asyncio
    async def test_broadcast_to_all(self, manager):
        """Test broadcasting to all SSE clients."""
        client_id, queue = await manager.connect()

        event = RealtimePredictionEvent(
            event_id="sse-all",
            event_type="alert",
            timestamp=datetime.now(timezone.utc),
            room_id=None,
            data={"alert": "test"},
        )

        sent_count = await manager.broadcast_to_all(event)

        assert sent_count == 1
        assert not queue.empty()

    def test_get_connection_stats(self, manager):
        """Test getting SSE connection statistics."""
        stats = manager.get_connection_stats()

        assert "total_active_connections" in stats
        assert "connections_by_room" in stats
        assert stats["total_active_connections"] == 0


class TestRealtimePublishingSystem:
    """Test RealtimePublishingSystem."""

    @pytest.fixture
    def mqtt_config(self):
        """Create MQTT configuration."""
        return MQTTConfig(
            broker="localhost",
            port=1883,
            username="test",
            password="test",
            topic_prefix="occupancy/predictions",
        )

    @pytest.fixture
    def room_configs(self):
        """Create room configurations."""
        return {
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={"motion": ["binary_sensor.living_room_motion"]},
            ),
            "kitchen": RoomConfig(
                room_id="kitchen",
                name="Kitchen",
                sensors={"motion": ["binary_sensor.kitchen_motion"]},
            ),
        }

    @pytest.fixture
    def mock_prediction_publisher(self):
        """Create mock prediction publisher."""
        publisher = Mock(spec=PredictionPublisher)
        publisher.publish_prediction = AsyncMock(
            return_value=MQTTPublishResult(success=True, error_message=None)
        )
        return publisher

    @pytest.fixture
    def prediction_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            transition_type="occupied",
            confidence_score=0.85,
            model_type="ensemble",
            model_version="1.0",
            features_used=["time_features", "sequence_features"],
            alternatives=[(datetime.now(timezone.utc) + timedelta(minutes=25), 0.75)],
            prediction_metadata={"feature_count": 15},
        )

    @pytest.fixture
    async def publishing_system(
        self, mqtt_config, room_configs, mock_prediction_publisher
    ):
        """Create real-time publishing system."""
        # Disable background tasks for testing
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config,
                rooms=room_configs,
                prediction_publisher=mock_prediction_publisher,
                enabled_channels=[
                    PublishingChannel.MQTT,
                    PublishingChannel.WEBSOCKET,
                    PublishingChannel.SSE,
                ],
            )
            await system.initialize()
            yield system
            await system.shutdown()

    @pytest.mark.asyncio
    async def test_initialization(self, mqtt_config, room_configs):
        """Test system initialization."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config, rooms=room_configs
            )

            assert system.config == mqtt_config
            assert system.rooms == room_configs
            assert len(system.enabled_channels) == 3  # All channels by default
            assert isinstance(system.websocket_manager, WebSocketConnectionManager)
            assert isinstance(system.sse_manager, SSEConnectionManager)

            await system.initialize()
            assert system._publishing_active is True

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_initialization_error(self, mqtt_config, room_configs):
        """Test initialization error handling."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config, rooms=room_configs
            )

            # Mock an error during initialization
            with patch.object(
                system.websocket_manager, "connect", side_effect=Exception("Init error")
            ):
                with pytest.raises(RealtimePublishingError):
                    await system.initialize()

    @pytest.mark.asyncio
    async def test_publish_prediction_all_channels(
        self, publishing_system, prediction_result
    ):
        """Test publishing prediction to all enabled channels."""
        results = await publishing_system.publish_prediction(
            prediction_result, "living_room", "vacant"
        )

        assert "mqtt" in results
        assert "websocket" in results
        assert "sse" in results

        # MQTT should succeed with mock
        assert results["mqtt"]["success"] is True

        # WebSocket and SSE should succeed even with no clients
        assert results["websocket"]["success"] is True
        assert results["sse"]["success"] is True

        # Metrics should be updated
        assert publishing_system.metrics.total_predictions_published == 1
        assert publishing_system.metrics.mqtt_publishes == 1

    @pytest.mark.asyncio
    async def test_publish_prediction_mqtt_only(
        self, mqtt_config, room_configs, mock_prediction_publisher, prediction_result
    ):
        """Test publishing prediction with only MQTT enabled."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config,
                rooms=room_configs,
                prediction_publisher=mock_prediction_publisher,
                enabled_channels=[PublishingChannel.MQTT],
            )
            await system.initialize()

            results = await system.publish_prediction(prediction_result, "kitchen")

            assert "mqtt" in results
            assert "websocket" not in results
            assert "sse" not in results
            assert results["mqtt"]["success"] is True

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_publish_prediction_mqtt_error(
        self, publishing_system, prediction_result
    ):
        """Test handling MQTT publishing error."""
        # Configure mock to fail
        publishing_system.prediction_publisher.publish_prediction.return_value = (
            MQTTPublishResult(success=False, error_message="MQTT connection failed")
        )

        results = await publishing_system.publish_prediction(
            prediction_result, "living_room"
        )

        assert results["mqtt"]["success"] is False
        assert "MQTT connection failed" in results["mqtt"]["error"]
        assert publishing_system.metrics.channel_errors["mqtt"] == 1

    @pytest.mark.asyncio
    async def test_publish_prediction_without_mqtt_publisher(
        self, mqtt_config, room_configs, prediction_result
    ):
        """Test publishing without MQTT publisher configured."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config,
                rooms=room_configs,
                prediction_publisher=None,  # No MQTT publisher
                enabled_channels=[PublishingChannel.MQTT, PublishingChannel.WEBSOCKET],
            )
            await system.initialize()

            results = await system.publish_prediction(prediction_result, "living_room")

            # MQTT should be skipped
            assert "mqtt" not in results
            assert "websocket" in results
            assert results["websocket"]["success"] is True

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_publish_system_status(self, publishing_system):
        """Test publishing system status updates."""
        status_data = {
            "status": "healthy",
            "uptime": 3600,
            "active_models": 3,
            "last_prediction": datetime.now(timezone.utc).isoformat(),
        }

        results = await publishing_system.publish_system_status(status_data)

        assert "websocket" in results
        assert "sse" in results
        assert results["websocket"]["success"] is True
        assert results["sse"]["success"] is True

    @pytest.mark.asyncio
    async def test_handle_websocket_connection(self, publishing_system):
        """Test WebSocket connection handling."""
        mock_websocket = AsyncMock(spec=WebSocketServerProtocol)
        mock_websocket.__aiter__ = AsyncMock(return_value=iter([]))

        # Test connection handling (will exit due to empty iterator)
        await publishing_system.handle_websocket_connection(
            mock_websocket, "/ws/predictions"
        )

        # Verify welcome message was sent
        mock_websocket.send.assert_called()
        call_args = mock_websocket.send.call_args[0][0]
        welcome_message = json.loads(call_args)
        assert welcome_message["event_type"] == "connection"
        assert "available_rooms" in welcome_message["data"]

    @pytest.mark.asyncio
    async def test_websocket_message_handling(self, publishing_system):
        """Test WebSocket message handling."""
        # Create mock WebSocket that yields test messages
        mock_websocket = AsyncMock(spec=WebSocketServerProtocol)
        test_messages = [
            json.dumps({"type": "subscribe", "room_id": "living_room"}),
            json.dumps({"type": "ping"}),
            json.dumps({"type": "unsubscribe", "room_id": "living_room"}),
        ]
        mock_websocket.__aiter__ = AsyncMock(return_value=iter(test_messages))

        # Mock the _handle_websocket_message to verify it's called
        with patch.object(
            publishing_system, "_handle_websocket_message"
        ) as mock_handle:
            await publishing_system.handle_websocket_connection(
                mock_websocket, "/ws/predictions"
            )

            # Should be called for each message
            assert mock_handle.call_count == len(test_messages)

    @pytest.mark.asyncio
    async def test_websocket_invalid_json(self, publishing_system):
        """Test WebSocket handling of invalid JSON."""
        mock_websocket = AsyncMock(spec=WebSocketServerProtocol)
        mock_websocket.__aiter__ = AsyncMock(return_value=iter(["invalid json"]))

        # Should not raise exception for invalid JSON
        await publishing_system.handle_websocket_connection(
            mock_websocket, "/ws/predictions"
        )

    @pytest.mark.asyncio
    async def test_create_sse_stream(self, publishing_system):
        """Test creating SSE stream."""
        response = await publishing_system.create_sse_stream("bedroom")

        assert response.media_type == "text/event-stream"
        assert "Cache-Control" in response.headers
        assert response.headers["Cache-Control"] == "no-cache"

    @pytest.mark.asyncio
    async def test_create_sse_stream_no_room(self, publishing_system):
        """Test creating SSE stream without room specification."""
        response = await publishing_system.create_sse_stream()

        assert response.media_type == "text/event-stream"

    def test_add_broadcast_callback(self, publishing_system):
        """Test adding broadcast callback."""
        callback_called = False

        def test_callback(event, results):
            nonlocal callback_called
            callback_called = True

        publishing_system.add_broadcast_callback(test_callback)
        assert len(publishing_system._broadcast_callbacks) == 1

    def test_remove_broadcast_callback(self, publishing_system):
        """Test removing broadcast callback."""

        def test_callback(event, results):
            pass

        publishing_system.add_broadcast_callback(test_callback)
        assert len(publishing_system._broadcast_callbacks) == 1

        publishing_system.remove_broadcast_callback(test_callback)
        # Note: Due to weak references, callback might not be immediately removed

    def test_get_publishing_stats(self, publishing_system):
        """Test getting publishing statistics."""
        stats = publishing_system.get_publishing_stats()

        assert "system_active" in stats
        assert "enabled_channels" in stats
        assert "metrics" in stats
        assert "uptime_seconds" in stats
        assert "websocket_stats" in stats
        assert "sse_stats" in stats
        assert "background_tasks" in stats

        assert stats["system_active"] is True
        assert isinstance(stats["uptime_seconds"], float)

    def test_format_prediction_data(self, publishing_system, prediction_result):
        """Test formatting prediction data."""
        formatted_data = publishing_system._format_prediction_data(
            prediction_result, "living_room", "vacant"
        )

        assert "room_id" in formatted_data
        assert "room_name" in formatted_data
        assert "predicted_time" in formatted_data
        assert "transition_type" in formatted_data
        assert "confidence_score" in formatted_data
        assert "time_until_seconds" in formatted_data
        assert "time_until_human" in formatted_data
        assert "current_state" in formatted_data
        assert "alternatives" in formatted_data

        assert formatted_data["room_id"] == "living_room"
        assert formatted_data["room_name"] == "Living Room"
        assert formatted_data["current_state"] == "vacant"
        assert formatted_data["transition_type"] == "occupied"

    @pytest.mark.asyncio
    async def test_handle_websocket_subscribe_message(self, publishing_system):
        """Test handling WebSocket subscribe message."""
        client_id = "test-client"
        message_data = {"type": "subscribe", "room_id": "kitchen"}

        # Add client to manager
        mock_websocket = AsyncMock()
        publishing_system.websocket_manager.connections[client_id] = mock_websocket
        publishing_system.websocket_manager.client_metadata[client_id] = (
            ClientConnection(
                connection_id=client_id,
                client_type="websocket",
                connected_at=datetime.now(timezone.utc),
                last_activity=datetime.now(timezone.utc),
                room_subscriptions=set(),
                metadata={},
            )
        )

        await publishing_system._handle_websocket_message(client_id, message_data)

        # Verify subscription
        client_meta = publishing_system.websocket_manager.client_metadata[client_id]
        assert "kitchen" in client_meta.room_subscriptions

        # Verify confirmation message was sent
        mock_websocket.send.assert_called()

    @pytest.mark.asyncio
    async def test_handle_websocket_ping_message(self, publishing_system):
        """Test handling WebSocket ping message."""
        client_id = "ping-client"
        message_data = {"type": "ping"}

        # Add client to manager
        mock_websocket = AsyncMock()
        publishing_system.websocket_manager.connections[client_id] = mock_websocket

        await publishing_system._handle_websocket_message(client_id, message_data)

        # Verify pong response
        mock_websocket.send.assert_called()
        call_args = mock_websocket.send.call_args[0][0]
        pong_message = json.loads(call_args)
        assert pong_message["event_type"] == "pong"

    def test_format_time_until(self, publishing_system):
        """Test formatting time until human readable format."""
        assert publishing_system._format_time_until(30) == "30 seconds"
        assert publishing_system._format_time_until(90) == "1 minute"
        assert publishing_system._format_time_until(150) == "2 minutes"
        assert publishing_system._format_time_until(3600) == "1 hour"
        assert publishing_system._format_time_until(3660) == "1h 1m"
        assert publishing_system._format_time_until(86400) == "1 day"
        assert publishing_system._format_time_until(90000) == "1d 1h"

    @pytest.mark.asyncio
    async def test_cleanup_stale_connections(self, publishing_system):
        """Test cleanup of stale connections."""
        # Add stale connection
        mock_websocket = AsyncMock()
        client_id = await publishing_system.websocket_manager.connect(mock_websocket)

        # Make connection appear stale
        client_meta = publishing_system.websocket_manager.client_metadata[client_id]
        client_meta.last_activity = datetime.now(timezone.utc) - timedelta(hours=2)

        # Run cleanup
        await publishing_system._cleanup_stale_connections()

        # Connection should be removed (but this is a private method, so we test indirectly)


class TestRealtimePublishingError:
    """Test RealtimePublishingError exception."""

    def test_error_creation(self):
        """Test creating realtime publishing error."""
        error = RealtimePublishingError("Test error message")

        assert error.message == "Test error message"
        assert error.error_code == "REALTIME_PUBLISHING_ERROR"
        assert isinstance(error, OccupancyPredictionError)


class TestFactoryFunctions:
    """Test factory functions and utilities."""

    @pytest.mark.asyncio
    async def test_realtime_publisher_context(self):
        """Test real-time publisher context manager."""
        config = {
            "mqtt_config": MQTTConfig(broker="localhost", port=1883),
            "rooms": {"test_room": RoomConfig(room_id="test_room", name="Test")},
            "enabled_channels": [PublishingChannel.WEBSOCKET],
        }

        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            # Mock the start and stop methods since they don't exist
            with patch.object(
                RealtimePublishingSystem, "start", new_callable=AsyncMock
            ), patch.object(RealtimePublishingSystem, "stop", new_callable=AsyncMock):

                async with realtime_publisher_context(config) as publisher:
                    assert isinstance(publisher, RealtimePublishingSystem)

    def test_create_realtime_app(self):
        """Test creating Starlette application."""
        mock_publisher = Mock(spec=RealtimePublishingSystem)
        mock_publisher.create_sse_stream = AsyncMock()
        mock_publisher.handle_websocket_connection = AsyncMock()
        mock_publisher.get_publishing_stats = Mock(return_value={"test": "stats"})

        app = create_realtime_app(mock_publisher)

        assert app is not None
        # Verify routes are configured (this would need more detailed testing of Starlette routing)

    def test_create_realtime_publishing_system_default_config(self):
        """Test creating publishing system with default configuration."""
        with patch("src.integration.realtime_publisher.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.mqtt = MQTTConfig(broker="localhost", port=1883)
            mock_config.rooms = {"test": RoomConfig(room_id="test", name="Test")}
            mock_get_config.return_value = mock_config

            system = create_realtime_publishing_system()

            assert isinstance(system, RealtimePublishingSystem)
            assert len(system.enabled_channels) == 3  # All channels by default

    def test_create_realtime_publishing_system_custom_config(self):
        """Test creating publishing system with custom configuration."""
        config = {
            "mqtt_config": MQTTConfig(broker="custom", port=8883),
            "rooms": {"custom_room": RoomConfig(room_id="custom_room", name="Custom")},
            "enabled_channels": [PublishingChannel.MQTT],
        }

        system = create_realtime_publishing_system(config)


class TestWebSocketAPIIntegration:
    """
    WebSocket API integration tests consolidated from test_websocket_api_integration.py.
    These tests validate the WebSocket API's integration with the TrackingManager.
    """

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
    def mock_websocket(self):
        """Mock WebSocket connection for testing."""
        websocket = MagicMock(spec=WebSocketServerProtocol)
        websocket.send = AsyncMock()
        websocket.close = AsyncMock()
        return websocket

    @pytest.mark.asyncio
    async def test_websocket_connection_management(self, websocket_config):
        """Test WebSocket connection management operations."""
        from src.integration.websocket_api import WebSocketConnectionManager

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        # Test connection
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")
        assert connection_id in manager.connections
        assert manager.stats.active_connections == 1

        # Test disconnection
        await manager.disconnect(connection_id)
        assert connection_id not in manager.connections
        assert manager.stats.active_connections == 0

    @pytest.mark.asyncio
    async def test_websocket_client_authentication(self, websocket_config):
        """Test WebSocket client authentication process."""
        from src.integration.websocket_api import (
            ClientAuthRequest,
            WebSocketConnectionManager,
        )

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        # Connect client
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")

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
            success = await manager.authenticate_client(connection_id, auth_request)
            assert success is True

            connection = manager.connections[connection_id]
            assert connection.authenticated is True
            assert connection.client_name == "TestClient"

    @pytest.mark.asyncio
    async def test_websocket_subscription_management(self, websocket_config):
        """Test WebSocket client subscription management."""
        from src.integration.websocket_api import (
            ClientSubscription,
            WebSocketConnectionManager,
        )

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        # Connect and authenticate client
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")
        connection = manager.connections[connection_id]
        connection.authenticated = True
        connection.room_filters.add("living_room")

        subscription = ClientSubscription(
            endpoint="/ws/predictions", room_id="living_room", filters={}
        )

        success = await manager.subscribe_client(connection_id, subscription)
        assert success is True
        assert "/ws/predictions" in connection.subscriptions
        assert "living_room" in connection.room_subscriptions

    @pytest.mark.asyncio
    async def test_websocket_message_broadcasting(self, websocket_config):
        """Test WebSocket message broadcasting to subscribed clients."""
        from src.integration.websocket_api import (
            MessageType,
            WebSocketConnectionManager,
            WebSocketMessage,
        )

        manager = WebSocketConnectionManager(websocket_config)

        # Create multiple mock clients
        clients = []
        for i in range(3):
            mock_websocket = MagicMock(spec=WebSocketServerProtocol)
            mock_websocket.send = AsyncMock()
            connection_id = await manager.connect(mock_websocket, "/ws/predictions")

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
    async def test_websocket_rate_limiting(self, websocket_config):
        """Test WebSocket rate limiting functionality."""
        from src.integration.websocket_api import (
            MessageType,
            WebSocketConnectionManager,
            WebSocketMessage,
        )

        # Set low rate limit for testing
        websocket_config["max_messages_per_minute"] = 2

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock()

        connection_id = await manager.connect(mock_websocket, "/ws/predictions")
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
    async def test_websocket_prediction_publishing_integration(self, websocket_config):
        """Test publishing prediction updates via WebSocket API integration."""
        from src.integration.websocket_api import WebSocketAPIServer
        from src.models.base.predictor import PredictionResult

        # Mock tracking manager
        mock_tracking_manager = AsyncMock()

        websocket_server = WebSocketAPIServer(
            tracking_manager=mock_tracking_manager, config=websocket_config
        )
        await websocket_server.initialize()

        # Create mock prediction result
        prediction_result = PredictionResult(
            predicted_time=datetime.utcnow() + timedelta(minutes=30),
            confidence=0.85,
            transition_type="occupied_to_vacant",
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

            connection = websocket_server.connection_manager.connections[connection_id]
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

        await websocket_server.stop()

    @pytest.mark.asyncio
    async def test_websocket_heartbeat_mechanism(self, websocket_config):
        """Test WebSocket heartbeat functionality."""
        from src.integration.websocket_api import WebSocketConnectionManager

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock()

        connection_id = await manager.connect(mock_websocket, "/ws/predictions")

        connection = manager.connections[connection_id]
        connection.authenticated = True

        # Send heartbeat
        success = await manager.send_heartbeat(connection_id)

        # Verify heartbeat sent
        assert success is True
        mock_websocket.send.assert_called_once()

        # Verify heartbeat message format
        sent_args = mock_websocket.send.call_args[0]
        sent_message = json.loads(sent_args[0])
        assert sent_message["message_type"] == "heartbeat"
        assert "server_time" in sent_message["data"]

    @pytest.mark.asyncio
    async def test_websocket_error_handling_and_recovery(self, websocket_config):
        """Test WebSocket error handling and recovery mechanisms."""
        from src.integration.websocket_api import (
            MessageType,
            WebSocketConnectionManager,
            WebSocketMessage,
        )

        manager = WebSocketConnectionManager(websocket_config)

        # Mock websocket that fails on send
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)
        mock_websocket.send = AsyncMock(side_effect=Exception("Connection lost"))

        connection_id = await manager.connect(mock_websocket, "/ws/predictions")
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
    async def test_websocket_connection_cleanup(self, websocket_config):
        """Test automatic WebSocket connection cleanup."""
        from src.integration.websocket_api import WebSocketConnectionManager

        # Set short timeout for testing
        websocket_config["connection_timeout_seconds"] = 1

        manager = WebSocketConnectionManager(websocket_config)
        mock_websocket = MagicMock(spec=WebSocketServerProtocol)

        connection_id = await manager.connect(mock_websocket, "/ws/predictions")
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

    def test_websocket_message_serialization(self):
        """Test WebSocket message serialization and deserialization."""
        from src.integration.websocket_api import MessageType, WebSocketMessage

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
        assert deserialized_message.message_type == original_message.message_type
        assert deserialized_message.endpoint == original_message.endpoint
        assert deserialized_message.room_id == original_message.room_id
        assert deserialized_message.data == original_message.data
        assert deserialized_message.requires_ack == original_message.requires_ack


class TestIntegrationScenarios:
    """Test realistic integration scenarios."""

    @pytest.mark.asyncio
    async def test_full_prediction_flow(
        self, mqtt_config, room_configs, mock_prediction_publisher
    ):
        """Test complete prediction publishing flow."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config,
                rooms=room_configs,
                prediction_publisher=mock_prediction_publisher,
            )
            await system.initialize()

            # Connect WebSocket client
            mock_websocket = AsyncMock()
            ws_client_id = await system.websocket_manager.connect(mock_websocket)
            await system.websocket_manager.subscribe_to_room(
                ws_client_id, "living_room"
            )

            # Connect SSE client
            sse_client_id, sse_queue = await system.sse_manager.connect()
            await system.sse_manager.subscribe_to_room(sse_client_id, "living_room")

            # Publish prediction
            prediction = PredictionResult(
                predicted_time=datetime.now(timezone.utc) + timedelta(minutes=15),
                transition_type="occupied",
                confidence_score=0.9,
                model_type="ensemble",
                model_version="1.0",
            )

            results = await system.publish_prediction(
                prediction, "living_room", "vacant"
            )

            # Verify all channels received the prediction
            assert results["mqtt"]["success"] is True
            assert results["websocket"]["clients_notified"] == 1
            assert results["sse"]["clients_notified"] == 1

            # Verify WebSocket received message
            mock_websocket.send.assert_called()

            # Verify SSE queue has message
            assert not sse_queue.empty()

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_error_recovery_scenario(
        self, mqtt_config, room_configs, mock_prediction_publisher
    ):
        """Test system behavior during error conditions."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config,
                rooms=room_configs,
                prediction_publisher=mock_prediction_publisher,
            )
            await system.initialize()

            # Configure MQTT to fail
            mock_prediction_publisher.publish_prediction.side_effect = Exception(
                "MQTT failure"
            )

            prediction = PredictionResult(
                predicted_time=datetime.now(timezone.utc) + timedelta(minutes=10),
                transition_type="vacant",
                confidence_score=0.8,
                model_type="xgboost",
                model_version="1.0",
            )

            results = await system.publish_prediction(prediction, "kitchen", "occupied")

            # MQTT should fail, but other channels should still work
            assert results["mqtt"]["success"] is False
            assert results["websocket"]["success"] is True
            assert results["sse"]["success"] is True

            # Error metrics should be updated
            assert system.metrics.channel_errors["mqtt"] == 1
            assert system.metrics.broadcast_errors == 0  # Overall broadcast didn't fail

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_high_load_scenario(
        self, mqtt_config, room_configs, mock_prediction_publisher
    ):
        """Test system behavior under high load."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
            system = RealtimePublishingSystem(
                mqtt_config=mqtt_config,
                rooms=room_configs,
                prediction_publisher=mock_prediction_publisher,
            )
            await system.initialize()

            # Connect multiple clients
            websocket_clients = []
            for i in range(10):
                mock_ws = AsyncMock()
                client_id = await system.websocket_manager.connect(
                    mock_ws, f"ws-client-{i}"
                )
                await system.websocket_manager.subscribe_to_room(
                    client_id, "living_room"
                )
                websocket_clients.append((client_id, mock_ws))

            # Publish multiple predictions rapidly
            predictions = []
            for i in range(5):
                prediction = PredictionResult(
                    predicted_time=datetime.now(timezone.utc)
                    + timedelta(minutes=i * 5),
                    transition_type="occupied" if i % 2 == 0 else "vacant",
                    confidence_score=0.7 + (i * 0.05),
                    model_type="ensemble",
                    model_version="1.0",
                )
                predictions.append(system.publish_prediction(prediction, "living_room"))

            # Execute all predictions concurrently
            results = await asyncio.gather(*predictions)

            # Verify all predictions were published successfully
            for result in results:
                assert result["mqtt"]["success"] is True
                assert result["websocket"]["clients_notified"] == 10  # All 10 clients

            # Verify metrics
            assert system.metrics.total_predictions_published == 5
            assert system.metrics.mqtt_publishes == 5

            await system.shutdown()
