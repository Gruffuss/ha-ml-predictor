"""
Real-time Prediction Publishing System for Home Assistant Integration.

This module provides comprehensive real-time prediction publishing across multiple
channels (MQTT, WebSocket, Server-Sent Events) with full integration into the
main system workflow through TrackingManager.

Features:
- Multiple publishing channels (MQTT, WebSocket, SSE)
- Real-time prediction streaming to connected clients
- Automatic integration with existing prediction workflow
- Channel-specific formatting and optimization
- Client connection management and broadcasting
- Error handling and retry mechanisms
- Performance monitoring and metrics
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set
import uuid
import weakref

from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route, WebSocketRoute
import websockets
from websockets.server import WebSocketServerProtocol

from ..core.config import MQTTConfig, RoomConfig, get_config
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..models.base.predictor import PredictionResult
from .prediction_publisher import PredictionPublisher

logger = logging.getLogger(__name__)


class PublishingChannel(Enum):
    """Available publishing channels."""

    MQTT = "mqtt"
    WEBSOCKET = "websocket"
    SSE = "sse"


@dataclass
class ClientConnection:
    """Represents a real-time client connection."""

    connection_id: str
    client_type: str  # 'websocket' or 'sse'
    connected_at: datetime
    last_activity: datetime
    room_subscriptions: Set[str]
    metadata: Dict[str, Any]

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()


@dataclass
class PublishingMetrics:
    """Metrics for real-time publishing system."""

    total_predictions_published: int = 0
    mqtt_publishes: int = 0
    websocket_publishes: int = 0
    sse_publishes: int = 0

    active_websocket_connections: int = 0
    active_sse_connections: int = 0
    total_connections_served: int = 0

    broadcast_errors: int = 0
    channel_errors: Dict[str, int] = None

    def __post_init__(self):
        if self.channel_errors is None:
            self.channel_errors = {channel.value: 0 for channel in PublishingChannel}


@dataclass
class RealtimePredictionEvent:
    """Real-time prediction event for broadcasting."""

    event_id: str
    event_type: str  # 'prediction', 'system_status', 'alert'
    timestamp: datetime
    room_id: Optional[str]
    data: Dict[str, Any]

    def to_websocket_message(self) -> str:
        """Convert to WebSocket message format."""
        return json.dumps(
            {
                "event_id": self.event_id,
                "event_type": self.event_type,
                "timestamp": self.timestamp.isoformat(),
                "room_id": self.room_id,
                "data": self.data,
            }
        )

    def to_sse_message(self) -> str:
        """Convert to Server-Sent Events format."""
        return (
            f"id: {self.event_id}\n"
            f"event: {self.event_type}\n"
            f"data: {json.dumps(self.data)}\n\n"
        )


class WebSocketConnectionManager:
    """Manages WebSocket connections and broadcasting."""

    def __init__(self):
        self.connections: Dict[str, WebSocketServerProtocol] = {}
        self.client_metadata: Dict[str, ClientConnection] = {}
        self._lock = asyncio.Lock()

    async def connect(
        self, websocket: WebSocketServerProtocol, client_id: str = None
    ) -> str:
        """Register a new WebSocket connection."""
        if client_id is None:
            client_id = str(uuid.uuid4())

        async with self._lock:
            self.connections[client_id] = websocket
            self.client_metadata[client_id] = ClientConnection(
                connection_id=client_id,
                client_type="websocket",
                connected_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                room_subscriptions=set(),
                metadata={},
            )

        logger.info(f"WebSocket client connected: {client_id}")
        return client_id

    async def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if client_id in self.connections:
                del self.connections[client_id]
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]

        logger.info(f"WebSocket client disconnected: {client_id}")

    async def subscribe_to_room(self, client_id: str, room_id: str):
        """Subscribe client to room updates."""
        if client_id in self.client_metadata:
            self.client_metadata[client_id].room_subscriptions.add(room_id)
            self.client_metadata[client_id].update_activity()
            logger.debug(f"Client {client_id} subscribed to room {room_id}")

    async def unsubscribe_from_room(self, client_id: str, room_id: str):
        """Unsubscribe client from room updates."""
        if client_id in self.client_metadata:
            self.client_metadata[client_id].room_subscriptions.discard(room_id)
            self.client_metadata[client_id].update_activity()
            logger.debug(f"Client {client_id} unsubscribed from room {room_id}")

    async def broadcast_to_room(
        self, room_id: str, event: RealtimePredictionEvent
    ) -> int:
        """Broadcast event to all clients subscribed to a room."""
        message = event.to_websocket_message()
        successful_sends = 0
        failed_connections = []

        async with self._lock:
            for client_id, websocket in self.connections.items():
                client_meta = self.client_metadata.get(client_id)
                if client_meta and room_id in client_meta.room_subscriptions:
                    try:
                        await websocket.send(message)
                        client_meta.update_activity()
                        successful_sends += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to send to WebSocket client {client_id}: {e}"
                        )
                        failed_connections.append(client_id)

        # Clean up failed connections
        for client_id in failed_connections:
            await self.disconnect(client_id)

        return successful_sends

    async def broadcast_to_all(self, event: RealtimePredictionEvent) -> int:
        """Broadcast event to all connected clients."""
        message = event.to_websocket_message()
        successful_sends = 0
        failed_connections = []

        async with self._lock:
            for client_id, websocket in self.connections.items():
                try:
                    await websocket.send(message)
                    if client_id in self.client_metadata:
                        self.client_metadata[client_id].update_activity()
                    successful_sends += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to send to WebSocket client {client_id}: {e}"
                    )
                    failed_connections.append(client_id)

        # Clean up failed connections
        for client_id in failed_connections:
            await self.disconnect(client_id)

        return successful_sends

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "total_active_connections": len(self.connections),
            "connections_by_room": {
                room_id: sum(
                    1
                    for client in self.client_metadata.values()
                    if room_id in client.room_subscriptions
                )
                for room_id in set().union(
                    *(
                        client.room_subscriptions
                        for client in self.client_metadata.values()
                    )
                )
            },
            "oldest_connection": min(
                (client.connected_at for client in self.client_metadata.values()),
                default=None,
            ),
            "most_recent_activity": max(
                (client.last_activity for client in self.client_metadata.values()),
                default=None,
            ),
        }


class SSEConnectionManager:
    """Manages Server-Sent Events connections."""

    def __init__(self):
        self.connections: Dict[str, asyncio.Queue] = {}
        self.client_metadata: Dict[str, ClientConnection] = {}
        self._lock = asyncio.Lock()

    async def connect(self, client_id: str = None) -> tuple[str, asyncio.Queue]:
        """Register a new SSE connection."""
        if client_id is None:
            client_id = str(uuid.uuid4())

        queue = asyncio.Queue()

        async with self._lock:
            self.connections[client_id] = queue
            self.client_metadata[client_id] = ClientConnection(
                connection_id=client_id,
                client_type="sse",
                connected_at=datetime.utcnow(),
                last_activity=datetime.utcnow(),
                room_subscriptions=set(),
                metadata={},
            )

        logger.info(f"SSE client connected: {client_id}")
        return client_id, queue

    async def disconnect(self, client_id: str):
        """Remove an SSE connection."""
        async with self._lock:
            if client_id in self.connections:
                del self.connections[client_id]
            if client_id in self.client_metadata:
                del self.client_metadata[client_id]

        logger.info(f"SSE client disconnected: {client_id}")

    async def subscribe_to_room(self, client_id: str, room_id: str):
        """Subscribe client to room updates."""
        if client_id in self.client_metadata:
            self.client_metadata[client_id].room_subscriptions.add(room_id)
            self.client_metadata[client_id].update_activity()
            logger.debug(f"SSE client {client_id} subscribed to room {room_id}")

    async def broadcast_to_room(
        self, room_id: str, event: RealtimePredictionEvent
    ) -> int:
        """Broadcast event to all SSE clients subscribed to a room."""
        message = event.to_sse_message()
        successful_sends = 0
        failed_connections = []

        async with self._lock:
            for client_id, queue in self.connections.items():
                client_meta = self.client_metadata.get(client_id)
                if client_meta and room_id in client_meta.room_subscriptions:
                    try:
                        await queue.put(message)
                        client_meta.update_activity()
                        successful_sends += 1
                    except Exception as e:
                        logger.warning(
                            f"Failed to queue message for SSE client {client_id}: {e}"
                        )
                        failed_connections.append(client_id)

        # Clean up failed connections
        for client_id in failed_connections:
            await self.disconnect(client_id)

        return successful_sends

    async def broadcast_to_all(self, event: RealtimePredictionEvent) -> int:
        """Broadcast event to all SSE clients."""
        message = event.to_sse_message()
        successful_sends = 0
        failed_connections = []

        async with self._lock:
            for client_id, queue in self.connections.items():
                try:
                    await queue.put(message)
                    if client_id in self.client_metadata:
                        self.client_metadata[client_id].update_activity()
                    successful_sends += 1
                except Exception as e:
                    logger.warning(
                        f"Failed to queue message for SSE client {client_id}: {e}"
                    )
                    failed_connections.append(client_id)

        # Clean up failed connections
        for client_id in failed_connections:
            await self.disconnect(client_id)

        return successful_sends

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get SSE connection statistics."""
        return {
            "total_active_connections": len(self.connections),
            "connections_by_room": {
                room_id: sum(
                    1
                    for client in self.client_metadata.values()
                    if room_id in client.room_subscriptions
                )
                for room_id in set().union(
                    *(
                        client.room_subscriptions
                        for client in self.client_metadata.values()
                    )
                )
            },
            "oldest_connection": min(
                (client.connected_at for client in self.client_metadata.values()),
                default=None,
            ),
            "most_recent_activity": max(
                (client.last_activity for client in self.client_metadata.values()),
                default=None,
            ),
        }


class RealtimePublishingSystem:
    """
    Comprehensive real-time prediction publishing system.

    Integrates with TrackingManager to provide automatic real-time publishing
    across multiple channels without requiring manual setup or intervention.
    """

    def __init__(
        self,
        mqtt_config: Optional[MQTTConfig] = None,
        rooms: Optional[Dict[str, RoomConfig]] = None,
        prediction_publisher: Optional[PredictionPublisher] = None,
        enabled_channels: Optional[List[PublishingChannel]] = None,
    ):
        """
        Initialize real-time publishing system.

        Args:
            mqtt_config: MQTT configuration
            rooms: Room configurations
            prediction_publisher: Existing prediction publisher for MQTT
            enabled_channels: List of channels to enable
        """
        # Load configuration if not provided
        if mqtt_config is None or rooms is None:
            system_config = get_config()
            mqtt_config = mqtt_config or system_config.mqtt
            rooms = rooms or system_config.rooms

        self.config = mqtt_config
        self.rooms = rooms
        self.prediction_publisher = prediction_publisher

        # Default to all channels if not specified
        self.enabled_channels = enabled_channels or [
            PublishingChannel.MQTT,
            PublishingChannel.WEBSOCKET,
            PublishingChannel.SSE,
        ]

        # Connection managers
        self.websocket_manager = WebSocketConnectionManager()
        self.sse_manager = SSEConnectionManager()

        # Metrics and state
        self.metrics = PublishingMetrics()
        self.system_start_time = datetime.utcnow()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._publishing_active = False

        # Event broadcasting
        self._broadcast_callbacks: List[Callable] = []

        logger.info(
            "Initialized RealtimePublishingSystem with channels: "
            f"{[channel.value for channel in self.enabled_channels]}"
        )

    async def initialize(self) -> None:
        """Initialize the real-time publishing system."""
        try:
            # Skip background tasks in test environment
            import os

            if not os.getenv("DISABLE_BACKGROUND_TASKS"):
                # Start background tasks
                cleanup_task = asyncio.create_task(self._cleanup_stale_connections())
                self._background_tasks.append(cleanup_task)

                metrics_task = asyncio.create_task(self._update_metrics_loop())
                self._background_tasks.append(metrics_task)

            self._publishing_active = True
            logger.info("Real-time publishing system initialized")

        except Exception as e:
            logger.error(f"Failed to initialize real-time publishing system: {e}")
            raise RealtimePublishingError("Failed to initialize system", cause=e)

    async def shutdown(self) -> None:
        """Shutdown the real-time publishing system."""
        try:
            self._shutdown_event.set()
            self._publishing_active = False

            # Cancel background tasks
            if self._background_tasks:
                for task in self._background_tasks:
                    task.cancel()
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            # Close all connections
            await self._close_all_connections()

            logger.info("Real-time publishing system shutdown complete")

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    async def publish_prediction(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publish prediction across all enabled channels.

        This method is called automatically by TrackingManager when predictions are made.

        Args:
            prediction_result: The prediction to publish
            room_id: Room the prediction is for
            current_state: Current occupancy state if known

        Returns:
            Dictionary with publish results for each channel
        """
        results = {}

        try:
            # Create real-time event
            event = RealtimePredictionEvent(
                event_id=str(uuid.uuid4()),
                event_type="prediction",
                timestamp=datetime.utcnow(),
                room_id=room_id,
                data=self._format_prediction_data(
                    prediction_result, room_id, current_state
                ),
            )

            # Publish to MQTT if enabled and available
            if (
                PublishingChannel.MQTT in self.enabled_channels
                and self.prediction_publisher is not None
            ):
                try:
                    # TODO: Create standardized prediction payload for future use
                    # prediction_payload = PredictionPayload(...)

                    # TODO: Use dedicated MQTT publisher for enhanced functionality
                    # mqtt_publisher = MQTTPublisher(...)

                    mqtt_result = await self.prediction_publisher.publish_prediction(
                        prediction_result, room_id, current_state
                    )
                    results["mqtt"] = {
                        "success": mqtt_result.success,
                        "error": mqtt_result.error_message,
                    }
                    if mqtt_result.success:
                        self.metrics.mqtt_publishes += 1
                    else:
                        self.metrics.channel_errors["mqtt"] += 1
                except Exception as e:
                    logger.error(f"MQTT publish failed: {e}")
                    results["mqtt"] = {"success": False, "error": str(e)}
                    self.metrics.channel_errors["mqtt"] += 1

            # Publish to WebSocket if enabled
            if PublishingChannel.WEBSOCKET in self.enabled_channels:
                try:
                    ws_sent = await self.websocket_manager.broadcast_to_room(
                        room_id, event
                    )
                    results["websocket"] = {
                        "success": True,
                        "clients_notified": ws_sent,
                    }
                    self.metrics.websocket_publishes += ws_sent
                except Exception as e:
                    logger.error(f"WebSocket publish failed: {e}")
                    results["websocket"] = {"success": False, "error": str(e)}
                    self.metrics.channel_errors["websocket"] += 1

            # Publish to SSE if enabled
            if PublishingChannel.SSE in self.enabled_channels:
                try:
                    sse_sent = await self.sse_manager.broadcast_to_room(room_id, event)
                    results["sse"] = {
                        "success": True,
                        "clients_notified": sse_sent,
                    }
                    self.metrics.sse_publishes += sse_sent
                except Exception as e:
                    logger.error(f"SSE publish failed: {e}")
                    results["sse"] = {"success": False, "error": str(e)}
                    self.metrics.channel_errors["sse"] += 1

            # Update metrics
            self.metrics.total_predictions_published += 1

            # Call broadcast callbacks
            valid_callbacks = []
            for weak_callback in self._broadcast_callbacks:
                callback = (
                    weak_callback()
                    if hasattr(weak_callback, "__call__")
                    else weak_callback
                )
                if callback is not None:
                    valid_callbacks.append(callback)
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(event, results)
                        else:
                            callback(event, results)
                    except Exception as e:
                        logger.error(f"Error in broadcast callback: {e}")

            # Update callback list to remove stale weak references
            self._broadcast_callbacks = [weakref.ref(cb) for cb in valid_callbacks]

            logger.debug(
                f"Published prediction for {room_id} across {len(results)} channels"
            )

        except Exception as e:
            logger.error(f"Error publishing prediction for {room_id}: {e}")
            self.metrics.broadcast_errors += 1
            results["error"] = str(e)

        return results

    async def publish_system_status(
        self, status_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Publish system status across real-time channels.

        Args:
            status_data: System status information

        Returns:
            Dictionary with publish results for each channel
        """
        results = {}

        try:
            event = RealtimePredictionEvent(
                event_id=str(uuid.uuid4()),
                event_type="system_status",
                timestamp=datetime.utcnow(),
                room_id=None,
                data=status_data,
            )

            # Broadcast to all WebSocket clients
            if PublishingChannel.WEBSOCKET in self.enabled_channels:
                try:
                    ws_sent = await self.websocket_manager.broadcast_to_all(event)
                    results["websocket"] = {
                        "success": True,
                        "clients_notified": ws_sent,
                    }
                except Exception as e:
                    logger.error(f"WebSocket status broadcast failed: {e}")
                    results["websocket"] = {"success": False, "error": str(e)}

            # Broadcast to all SSE clients
            if PublishingChannel.SSE in self.enabled_channels:
                try:
                    sse_sent = await self.sse_manager.broadcast_to_all(event)
                    results["sse"] = {
                        "success": True,
                        "clients_notified": sse_sent,
                    }
                except Exception as e:
                    logger.error(f"SSE status broadcast failed: {e}")
                    results["sse"] = {"success": False, "error": str(e)}

            logger.debug(f"Published system status across {len(results)} channels")

        except Exception as e:
            logger.error(f"Error publishing system status: {e}")
            results["error"] = str(e)

        return results

    async def handle_websocket_connection(
        self, websocket: WebSocketServerProtocol, path: str
    ):
        """Handle WebSocket connection and message routing."""
        client_id = None
        try:
            client_id = await self.websocket_manager.connect(websocket)
            self.metrics.total_connections_served += 1

            # Send welcome message
            welcome_event = RealtimePredictionEvent(
                event_id=str(uuid.uuid4()),
                event_type="connection",
                timestamp=datetime.utcnow(),
                room_id=None,
                data={
                    "message": "Connected to real-time prediction system",
                    "client_id": client_id,
                    "available_rooms": list(self.rooms.keys()),
                },
            )

            await websocket.send(welcome_event.to_websocket_message())

            # Handle incoming messages
            async for message in websocket:
                try:
                    data = json.loads(message)
                    await self._handle_websocket_message(client_id, data)
                except json.JSONDecodeError:
                    logger.warning(f"Invalid JSON from WebSocket client {client_id}")
                except Exception as e:
                    logger.error(
                        f"Error handling WebSocket message from {client_id}: {e}"
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"WebSocket client {client_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            if client_id:
                await self.websocket_manager.disconnect(client_id)

    async def create_sse_stream(
        self, room_id: Optional[str] = None
    ) -> StreamingResponse:
        """Create Server-Sent Events stream."""

        async def event_stream():
            client_id, queue = await self.sse_manager.connect()
            self.metrics.total_connections_served += 1

            try:
                # Subscribe to room if specified
                if room_id:
                    await self.sse_manager.subscribe_to_room(client_id, room_id)

                # Send initial connection message
                yield (
                    f"id: {uuid.uuid4()}\n"
                    "event: connection\n"
                    f"data: {json.dumps({'message': 'Connected to real-time predictions', 'client_id': client_id})}\n\n"
                )

                # Stream events
                while True:
                    try:
                        # Wait for event with timeout to allow periodic keepalives
                        message = await asyncio.wait_for(queue.get(), timeout=30.0)
                        yield message
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield f"data: {json.dumps({'type': 'keepalive', 'timestamp': datetime.utcnow().isoformat()})}\n\n"

            except asyncio.CancelledError:
                logger.debug(f"SSE stream cancelled for client {client_id}")
            except Exception as e:
                logger.error(f"Error in SSE stream for client {client_id}: {e}")
            finally:
                await self.sse_manager.disconnect(client_id)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control",
            },
        )

    def add_broadcast_callback(self, callback: Callable) -> None:
        """Add callback to be called when events are broadcast."""
        if callback not in self._broadcast_callbacks:
            # Store callback as weak reference to prevent memory leaks
            weak_callback = weakref.ref(callback)
            self._broadcast_callbacks.append(weak_callback)
            logger.debug("Added broadcast callback")

    def remove_broadcast_callback(self, callback: Callable) -> None:
        """Remove broadcast callback."""
        if callback in self._broadcast_callbacks:
            self._broadcast_callbacks.remove(callback)
            logger.debug("Removed broadcast callback")

    def get_publishing_stats(self) -> Dict[str, Any]:
        """Get comprehensive publishing statistics."""
        # Update connection counts
        self.metrics.active_websocket_connections = len(
            self.websocket_manager.connections
        )
        self.metrics.active_sse_connections = len(self.sse_manager.connections)

        return {
            "system_active": self._publishing_active,
            "enabled_channels": [channel.value for channel in self.enabled_channels],
            "metrics": asdict(self.metrics),
            "uptime_seconds": (
                datetime.utcnow() - self.system_start_time
            ).total_seconds(),
            "websocket_stats": self.websocket_manager.get_connection_stats(),
            "sse_stats": self.sse_manager.get_connection_stats(),
            "background_tasks": len(self._background_tasks),
        }

    # Private methods

    def _format_prediction_data(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str],
    ) -> Dict[str, Any]:
        """Format prediction result for real-time broadcasting."""
        # Calculate time until transition
        time_until = prediction_result.predicted_time - datetime.utcnow()
        time_until_seconds = max(0, int(time_until.total_seconds()))

        # Get room name
        room_name = self.rooms.get(room_id, {}).get(
            "name", room_id.replace("_", " ").title()
        )

        return {
            "room_id": room_id,
            "room_name": room_name,
            "predicted_time": prediction_result.predicted_time.isoformat(),
            "transition_type": prediction_result.transition_type,
            "confidence_score": float(prediction_result.confidence_score),
            "time_until_seconds": time_until_seconds,
            "time_until_human": self._format_time_until(time_until_seconds),
            "current_state": current_state,
            "model_type": prediction_result.model_type,
            "model_version": prediction_result.model_version,
            "alternatives": [
                {
                    "predicted_time": alt_time.isoformat(),
                    "confidence": float(alt_confidence),
                }
                for alt_time, alt_confidence in (prediction_result.alternatives or [])[
                    :3
                ]
            ],
            "features_used": len(prediction_result.features_used or []),
            "prediction_metadata": prediction_result.prediction_metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }

    async def _handle_websocket_message(
        self, client_id: str, message_data: Dict[str, Any]
    ):
        """Handle incoming WebSocket message from client."""
        try:
            message_type = message_data.get("type")

            if message_type == "subscribe":
                room_id = message_data.get("room_id")
                if room_id and room_id in self.rooms:
                    await self.websocket_manager.subscribe_to_room(client_id, room_id)

                    # Send subscription confirmation
                    response = RealtimePredictionEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="subscription",
                        timestamp=datetime.utcnow(),
                        room_id=room_id,
                        data={
                            "message": f"Subscribed to {room_id}",
                            "status": "success",
                        },
                    )

                    websocket = self.websocket_manager.connections.get(client_id)
                    if websocket:
                        await websocket.send(response.to_websocket_message())

            elif message_type == "unsubscribe":
                room_id = message_data.get("room_id")
                if room_id:
                    await self.websocket_manager.unsubscribe_from_room(
                        client_id, room_id
                    )

                    # Send unsubscription confirmation
                    response = RealtimePredictionEvent(
                        event_id=str(uuid.uuid4()),
                        event_type="subscription",
                        timestamp=datetime.utcnow(),
                        room_id=room_id,
                        data={
                            "message": f"Unsubscribed from {room_id}",
                            "status": "success",
                        },
                    )

                    websocket = self.websocket_manager.connections.get(client_id)
                    if websocket:
                        await websocket.send(response.to_websocket_message())

            elif message_type == "ping":
                # Respond to ping with pong
                response = RealtimePredictionEvent(
                    event_id=str(uuid.uuid4()),
                    event_type="pong",
                    timestamp=datetime.utcnow(),
                    room_id=None,
                    data={"timestamp": datetime.utcnow().isoformat()},
                )

                websocket = self.websocket_manager.connections.get(client_id)
                if websocket:
                    await websocket.send(response.to_websocket_message())

        except Exception as e:
            logger.error(f"Error handling WebSocket message from {client_id}: {e}")

    def _format_time_until(self, seconds: int) -> str:
        """Format seconds into human readable time."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if minutes == 0:
                return f"{hours} hour{'s' if hours != 1 else ''}"
            else:
                return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            if hours == 0:
                return f"{days} day{'s' if days != 1 else ''}"
            else:
                return f"{days}d {hours}h"

    async def _cleanup_stale_connections(self):
        """Background task to clean up stale connections."""
        while not self._shutdown_event.is_set():
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=1)

                # Clean up stale WebSocket connections
                stale_ws_connections = [
                    client_id
                    for client_id, client_meta in self.websocket_manager.client_metadata.items()
                    if client_meta.last_activity < cutoff_time
                ]

                for client_id in stale_ws_connections:
                    await self.websocket_manager.disconnect(client_id)
                    logger.debug(f"Cleaned up stale WebSocket connection: {client_id}")

                # Clean up stale SSE connections
                stale_sse_connections = [
                    client_id
                    for client_id, client_meta in self.sse_manager.client_metadata.items()
                    if client_meta.last_activity < cutoff_time
                ]

                for client_id in stale_sse_connections:
                    await self.sse_manager.disconnect(client_id)
                    logger.debug(f"Cleaned up stale SSE connection: {client_id}")

                # Wait before next cleanup
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=300  # 5 minutes
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in connection cleanup: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _update_metrics_loop(self):
        """Background task to update metrics."""
        while not self._shutdown_event.is_set():
            try:
                # Update connection counts
                self.metrics.active_websocket_connections = len(
                    self.websocket_manager.connections
                )
                self.metrics.active_sse_connections = len(self.sse_manager.connections)

                # Wait before next update
                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=30  # 30 seconds
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in metrics update: {e}")
                await asyncio.sleep(30)

    async def _close_all_connections(self):
        """Close all active connections."""
        try:
            # Close all WebSocket connections
            for client_id in list(self.websocket_manager.connections.keys()):
                await self.websocket_manager.disconnect(client_id)

            # Close all SSE connections
            for client_id in list(self.sse_manager.connections.keys()):
                await self.sse_manager.disconnect(client_id)

            logger.info("All real-time connections closed")

        except Exception as e:
            logger.error(f"Error closing connections: {e}")


class RealtimePublishingError(OccupancyPredictionError):
    """Raised when real-time publishing operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="REALTIME_PUBLISHING_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )


# Factory functions and utilities


@asynccontextmanager
async def realtime_publisher_context(
    config: Optional[Dict[str, Any]] = None,
    tracking_manager=None,
) -> RealtimePublishingSystem:
    """Context manager for real-time publishing system."""
    publisher = RealtimePublishingSystem(
        config=config, tracking_manager=tracking_manager
    )

    try:
        await publisher.start()
        yield publisher
    finally:
        await publisher.stop()


def create_realtime_app(publisher: RealtimePublishingSystem) -> Starlette:
    """Create Starlette application with real-time endpoints."""

    async def sse_predictions(request):
        """Server-Sent Events endpoint for predictions."""
        return await publisher.create_sse_stream()

    async def sse_room_predictions(request):
        """Server-Sent Events endpoint for specific room predictions."""
        room_id = request.path_params.get("room_id")
        return await publisher.create_sse_stream(room_id=room_id)

    async def sse_health(request):
        """Health check for SSE endpoints."""
        stats = publisher.get_publishing_stats()
        return JSONResponse(
            {
                "status": "healthy",
                "active_connections": stats.get("metrics", {}).get(
                    "active_sse_connections", 0
                ),
                "uptime_seconds": stats.get("uptime_seconds", 0),
            }
        )

    async def websocket_predictions(websocket):
        """WebSocket endpoint for real-time predictions."""
        await publisher.handle_websocket_connection(websocket, "/ws/predictions")

    async def websocket_room_predictions(websocket):
        """WebSocket endpoint for room-specific predictions."""
        # Extract room_id from path
        room_id = websocket.path_params.get("room_id", "unknown")
        await publisher.handle_websocket_connection(websocket, f"/ws/room/{room_id}")

    # Define routes including WebSocket routes
    routes = [
        Route("/sse/predictions", sse_predictions, methods=["GET"]),
        Route("/sse/room/{room_id}", sse_room_predictions, methods=["GET"]),
        Route("/sse/health", sse_health, methods=["GET"]),
        WebSocketRoute("/ws/predictions", websocket_predictions),
        WebSocketRoute("/ws/room/{room_id}", websocket_room_predictions),
    ]

    # Create application
    app = Starlette(routes=routes)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    return app


def create_realtime_publishing_system(
    config: Optional[Dict[str, Any]] = None, tracking_manager=None
) -> RealtimePublishingSystem:
    """Create and configure real-time publishing system."""

    # Get default config if none provided
    if config is None:
        system_config = get_config()
        config = {
            "mqtt_config": (
                system_config.mqtt if hasattr(system_config, "mqtt") else None
            ),
            "rooms": system_config.rooms if hasattr(system_config, "rooms") else {},
            "enabled_channels": [
                PublishingChannel.MQTT,
                PublishingChannel.WEBSOCKET,
                PublishingChannel.SSE,
            ],
        }

    # Create publisher instance
    return RealtimePublishingSystem(config=config, tracking_manager=tracking_manager)
