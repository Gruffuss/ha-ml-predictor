"""
Comprehensive WebSocket API for Real-time Updates in the Home Assistant Occupancy Prediction System.

This module provides a production-ready WebSocket API server that integrates seamlessly with
the existing system architecture, providing real-time updates for predictions, system status,
and alerts across multiple WebSocket endpoints with authentication, rate limiting, and
automatic integration through TrackingManager.

Features:
- Multiple WebSocket endpoints (/ws/predictions, /ws/system-status, /ws/alerts, /ws/room/{room_id})
- API key authentication with secure token validation
- Rate limiting per connection and per message type
- Connection management with automatic cleanup
- Message queuing with acknowledgments and heartbeat
- Integration with existing TrackingManager and real-time publishing system
- Graceful error handling and reconnection support
- Comprehensive message schema validation
- Performance monitoring and connection statistics
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any, Dict, List, Optional, Set
import uuid
import weakref

from pydantic import BaseModel, Field
try:
    from pydantic import validator
except ImportError:
    # For Pydantic v2 compatibility
    from pydantic import field_validator as validator
from starlette.applications import Starlette
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse, PlainTextResponse
from starlette.routing import Route, WebSocketRoute
import websockets
from websockets.server import WebSocketServerProtocol

from ..core.config import get_config
from ..core.exceptions import (
    ErrorSeverity,
    OccupancyPredictionError,
    WebSocketAuthenticationError,
    WebSocketConnectionError,
    WebSocketRateLimitError,
    WebSocketValidationError,
)
from ..models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


class WebSocketEndpoint(Enum):
    """Available WebSocket endpoints."""

    PREDICTIONS = "/ws/predictions"
    SYSTEM_STATUS = "/ws/system-status"
    ALERTS = "/ws/alerts"
    ROOM_SPECIFIC = "/ws/room/{room_id}"


class MessageType(Enum):
    """WebSocket message types."""

    # Connection lifecycle
    CONNECTION = "connection"
    AUTHENTICATION = "authentication"
    HEARTBEAT = "heartbeat"

    # Subscription management
    SUBSCRIBE = "subscribe"
    UNSUBSCRIBE = "unsubscribe"
    SUBSCRIPTION_STATUS = "subscription_status"

    # Real-time updates
    PREDICTION_UPDATE = "prediction_update"
    SYSTEM_STATUS_UPDATE = "system_status_update"
    ALERT_NOTIFICATION = "alert_notification"
    DRIFT_NOTIFICATION = "drift_notification"

    # Control messages
    ACKNOWLEDGE = "acknowledge"
    ERROR = "error"
    RATE_LIMIT_WARNING = "rate_limit_warning"


@dataclass
class WebSocketMessage:
    """Standard WebSocket message format."""

    message_id: str
    message_type: MessageType
    timestamp: datetime
    endpoint: str
    data: Dict[str, Any]
    requires_ack: bool = False
    room_id: Optional[str] = None

    def to_json(self) -> str:
        """Convert message to JSON format."""
        return json.dumps(
            {
                "message_id": self.message_id,
                "message_type": self.message_type.value,
                "timestamp": self.timestamp.isoformat(),
                "endpoint": self.endpoint,
                "data": self.data,
                "requires_ack": self.requires_ack,
                "room_id": self.room_id,
            }
        )

    @classmethod
    def from_json(cls, json_data: str) -> "WebSocketMessage":
        """Create message from JSON data."""
        data = json.loads(json_data)
        return cls(
            message_id=data["message_id"],
            message_type=MessageType(data["message_type"]),
            timestamp=datetime.fromisoformat(data["timestamp"]),
            endpoint=data["endpoint"],
            data=data["data"],
            requires_ack=data.get("requires_ack", False),
            room_id=data.get("room_id"),
        )


class ClientAuthRequest(BaseModel):
    """Client authentication request model."""

    api_key: str = Field(..., min_length=1)
    client_name: Optional[str] = Field(None, max_length=100)
    capabilities: List[str] = Field(default_factory=list)
    room_filters: List[str] = Field(default_factory=list)
    
    @validator('api_key')
    def validate_api_key(cls, v):
        """Validate API key format."""
        if v and len(v) < 10:
            raise ValueError('API key must be at least 10 characters long')
        return v


class ClientSubscription(BaseModel):
    """Client subscription request model."""

    endpoint: str = Field(
        ...,
        pattern=r"^/ws/(predictions|system-status|alerts|room/[a-zA-Z0-9_]+)$",
    )
    room_id: Optional[str] = None
    filters: Dict[str, Any] = Field(default_factory=dict)


@dataclass
class ClientConnection:
    """Represents an authenticated WebSocket client connection."""

    connection_id: str
    websocket: WebSocketServerProtocol
    endpoint: str
    authenticated: bool = False
    connected_at: datetime = None
    last_activity: datetime = None
    last_heartbeat: datetime = None

    # Authentication details
    api_key: Optional[str] = None
    client_name: Optional[str] = None
    capabilities: Set[str] = None
    room_filters: Set[str] = None

    # Subscription management
    subscriptions: Set[str] = None
    room_subscriptions: Set[str] = None

    # Rate limiting
    message_count: int = 0
    last_rate_limit_reset: datetime = None
    rate_limited_until: Optional[datetime] = None

    # Message queue for reliable delivery
    pending_messages: List[WebSocketMessage] = None
    unacknowledged_messages: Dict[str, WebSocketMessage] = None

    def __post_init__(self):
        """Initialize default values."""
        if self.connected_at is None:
            self.connected_at = datetime.utcnow()
        if self.last_activity is None:
            self.last_activity = datetime.utcnow()
        if self.last_heartbeat is None:
            self.last_heartbeat = datetime.utcnow()
        if self.capabilities is None:
            self.capabilities = set()
        if self.room_filters is None:
            self.room_filters = set()
        if self.subscriptions is None:
            self.subscriptions = set()
        if self.room_subscriptions is None:
            self.room_subscriptions = set()
        if self.pending_messages is None:
            self.pending_messages = []
        if self.unacknowledged_messages is None:
            self.unacknowledged_messages = {}
        if self.last_rate_limit_reset is None:
            self.last_rate_limit_reset = datetime.utcnow()

    def update_activity(self):
        """Update last activity timestamp."""
        self.last_activity = datetime.utcnow()

    def update_heartbeat(self):
        """Update last heartbeat timestamp."""
        self.last_heartbeat = datetime.utcnow()
        self.update_activity()

    def is_rate_limited(self, max_messages_per_minute: int = 60) -> bool:
        """Check if client is currently rate limited."""
        if (
            self.rate_limited_until
            and datetime.utcnow() < self.rate_limited_until
        ):
            return True

        now = datetime.utcnow()
        if now - self.last_rate_limit_reset >= timedelta(minutes=1):
            self.message_count = 0
            self.last_rate_limit_reset = now

        return self.message_count >= max_messages_per_minute

    def increment_message_count(self):
        """Increment message count for rate limiting."""
        self.message_count += 1
        self.update_activity()

    def apply_rate_limit(self, duration_seconds: int = 60):
        """Apply rate limiting to client."""
        self.rate_limited_until = datetime.utcnow() + timedelta(
            seconds=duration_seconds
        )

    def can_access_room(self, room_id: str) -> bool:
        """Check if client can access specific room."""
        if not self.room_filters:
            return True  # No restrictions
        return room_id in self.room_filters

    def has_capability(self, capability: str) -> bool:
        """Check if client has specific capability."""
        return capability in self.capabilities


@dataclass
class WebSocketStats:
    """WebSocket API statistics."""

    total_connections: int = 0
    active_connections: int = 0
    authenticated_connections: int = 0

    # Connection counts by endpoint
    predictions_connections: int = 0
    system_status_connections: int = 0
    alerts_connections: int = 0
    room_specific_connections: int = 0

    # Message statistics
    total_messages_sent: int = 0
    total_messages_received: int = 0
    total_heartbeats_sent: int = 0
    total_authentications: int = 0

    # Performance metrics
    average_response_time_ms: float = 0.0
    message_queue_size: int = 0
    failed_message_deliveries: int = 0

    # Rate limiting statistics
    rate_limited_clients: int = 0
    total_rate_limit_violations: int = 0

    # Error statistics
    authentication_failures: int = 0
    connection_errors: int = 0
    message_errors: int = 0


class WebSocketConnectionManager:
    """Manages WebSocket connections with authentication and rate limiting."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize connection manager."""
        self.config = config or {}
        self.connections: Dict[str, ClientConnection] = {}
        self.stats = WebSocketStats()
        self._lock = asyncio.Lock()
        
        # Use weakref to track connection managers to avoid circular references
        self._manager_registry = weakref.WeakSet()

        # Configuration
        self.max_connections = self.config.get("max_connections", 1000)
        self.max_messages_per_minute = self.config.get(
            "max_messages_per_minute", 60
        )
        self.heartbeat_interval = self.config.get(
            "heartbeat_interval_seconds", 30
        )
        self.connection_timeout = self.config.get(
            "connection_timeout_seconds", 300
        )
        self.message_acknowledgment_timeout = self.config.get(
            "acknowledgment_timeout_seconds", 30
        )

    async def connect(
        self,
        websocket: WebSocketServerProtocol,
        endpoint: str,
        connection_id: Optional[str] = None,
    ) -> str:
        """Register a new WebSocket connection."""
        if connection_id is None:
            connection_id = str(uuid.uuid4())

        async with self._lock:
            if len(self.connections) >= self.max_connections:
                raise WebSocketConnectionError("Maximum connections exceeded")

            connection = ClientConnection(
                connection_id=connection_id,
                websocket=websocket,
                endpoint=endpoint,
            )

            self.connections[connection_id] = connection
            self.stats.total_connections += 1
            self.stats.active_connections = len(self.connections)

            # Update endpoint-specific statistics
            self._update_endpoint_stats(endpoint, 1)

        logger.info(
            f"WebSocket client connected: {connection_id} to {endpoint}"
        )
        return connection_id

    async def disconnect(self, connection_id: str):
        """Remove a WebSocket connection."""
        async with self._lock:
            if connection_id in self.connections:
                connection = self.connections[connection_id]

                # Update endpoint-specific statistics
                self._update_endpoint_stats(connection.endpoint, -1)

                # Update authentication statistics
                if connection.authenticated:
                    self.stats.authenticated_connections -= 1

                del self.connections[connection_id]
                self.stats.active_connections = len(self.connections)

        logger.info(f"WebSocket client disconnected: {connection_id}")

    async def authenticate_client(
        self, connection_id: str, auth_request: ClientAuthRequest
    ) -> bool:
        """Authenticate a client connection."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        # Verify API key
        config = get_config()
        if config.api.api_key_enabled:
            if auth_request.api_key != config.api.api_key:
                self.stats.authentication_failures += 1
                raise WebSocketAuthenticationError("Invalid API key")

        # Update connection with authentication details
        connection.authenticated = True
        connection.api_key = auth_request.api_key
        connection.client_name = auth_request.client_name
        connection.capabilities.update(auth_request.capabilities)
        connection.room_filters.update(auth_request.room_filters)
        connection.update_activity()

        self.stats.authenticated_connections += 1
        self.stats.total_authentications += 1

        logger.info(
            f"Client authenticated: {connection_id} ({auth_request.client_name})"
        )
        return True

    async def subscribe_client(
        self, connection_id: str, subscription: ClientSubscription
    ) -> bool:
        """Subscribe client to specific updates."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        if not connection.authenticated:
            raise WebSocketAuthenticationError(
                "Authentication required for subscriptions"
            )

        # Validate room access if room-specific subscription
        if subscription.room_id and not connection.can_access_room(
            subscription.room_id
        ):
            raise WebSocketValidationError(
                f"Access denied for room: {subscription.room_id}"
            )

        # Add subscription
        connection.subscriptions.add(subscription.endpoint)
        if subscription.room_id:
            connection.room_subscriptions.add(subscription.room_id)

        connection.update_activity()

        logger.debug(
            f"Client subscribed: {connection_id} to {subscription.endpoint}"
        )
        return True

    async def unsubscribe_client(
        self, connection_id: str, endpoint: str, room_id: Optional[str] = None
    ) -> bool:
        """Unsubscribe client from updates."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        connection.subscriptions.discard(endpoint)

        if room_id:
            connection.room_subscriptions.discard(room_id)

        connection.update_activity()

        logger.debug(f"Client unsubscribed: {connection_id} from {endpoint}")
        return True

    async def send_message(
        self, connection_id: str, message: WebSocketMessage
    ) -> bool:
        """Send message to specific client."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]

        # Check rate limiting
        if connection.is_rate_limited(self.max_messages_per_minute):
            self.stats.total_rate_limit_violations += 1
            await self._send_rate_limit_warning(connection)
            return False

        try:
            await connection.websocket.send(message.to_json())
            connection.increment_message_count()

            # Handle message acknowledgment if required
            if message.requires_ack:
                connection.unacknowledged_messages[message.message_id] = (
                    message
                )

            self.stats.total_messages_sent += 1
            return True

        except Exception as e:
            logger.error(f"Failed to send message to {connection_id}: {e}")
            self.stats.failed_message_deliveries += 1
            return False

    async def broadcast_to_endpoint(
        self,
        endpoint: str,
        message: WebSocketMessage,
        room_filter: Optional[str] = None,
    ) -> int:
        """Broadcast message to all clients subscribed to endpoint."""
        successful_sends = 0
        failed_connections = []

        async with self._lock:
            target_connections = [
                conn
                for conn in self.connections.values()
                if (
                    endpoint in conn.subscriptions
                    and conn.authenticated
                    and (
                        not room_filter
                        or room_filter in conn.room_subscriptions
                    )
                )
            ]

        for connection in target_connections:
            try:
                success = await self.send_message(
                    connection.connection_id, message
                )
                if success:
                    successful_sends += 1
                else:
                    failed_connections.append(connection.connection_id)
            except Exception as e:
                logger.warning(
                    f"Failed to send to connection {connection.connection_id}: {e}"
                )
                failed_connections.append(connection.connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        return successful_sends

    async def process_acknowledgment(
        self, connection_id: str, message_id: str
    ) -> bool:
        """Process message acknowledgment from client."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        if message_id in connection.unacknowledged_messages:
            del connection.unacknowledged_messages[message_id]
            connection.update_activity()
            return True

        return False

    async def send_heartbeat(self, connection_id: str) -> bool:
        """Send heartbeat to specific client."""
        if connection_id not in self.connections:
            return False

        connection = self.connections[connection_id]
        heartbeat_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            timestamp=datetime.utcnow(),
            endpoint=connection.endpoint,
            data={
                "server_time": datetime.utcnow().isoformat(),
                "connection_uptime": (
                    datetime.utcnow() - connection.connected_at
                ).total_seconds(),
            },
        )

        success = await self.send_message(connection_id, heartbeat_message)
        if success:
            connection.update_heartbeat()
            self.stats.total_heartbeats_sent += 1

        return success

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get comprehensive connection statistics."""
        # Update dynamic statistics
        self.stats.active_connections = len(self.connections)
        self.stats.authenticated_connections = sum(
            1 for conn in self.connections.values() if conn.authenticated
        )
        self.stats.rate_limited_clients = sum(
            1
            for conn in self.connections.values()
            if conn.rate_limited_until
            and datetime.utcnow() < conn.rate_limited_until
        )
        self.stats.message_queue_size = sum(
            len(conn.pending_messages) + len(conn.unacknowledged_messages)
            for conn in self.connections.values()
        )

        return asdict(self.stats)

    def _update_endpoint_stats(self, endpoint: str, delta: int):
        """Update endpoint-specific connection statistics."""
        if "/ws/predictions" in endpoint:
            self.stats.predictions_connections += delta
        elif "/ws/system-status" in endpoint:
            self.stats.system_status_connections += delta
        elif "/ws/alerts" in endpoint:
            self.stats.alerts_connections += delta
        elif "/ws/room/" in endpoint:
            self.stats.room_specific_connections += delta

    async def _send_rate_limit_warning(self, connection: ClientConnection):
        """Send rate limit warning to client."""
        warning_message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.RATE_LIMIT_WARNING,
            timestamp=datetime.utcnow(),
            endpoint=connection.endpoint,
            data={
                "message": "Rate limit exceeded",
                "max_messages_per_minute": self.max_messages_per_minute,
                "current_count": connection.message_count,
                "reset_time": connection.last_rate_limit_reset.isoformat(),
            },
        )

        try:
            await connection.websocket.send(warning_message.to_json())
        except Exception as e:
            logger.error(f"Failed to send rate limit warning: {e}")


class WebSocketAPIServer:
    """
    Comprehensive WebSocket API Server for real-time updates.

    Provides multiple WebSocket endpoints with authentication, rate limiting,
    and seamless integration with the existing TrackingManager system.
    """

    def __init__(
        self, tracking_manager=None, config: Optional[Dict[str, Any]] = None
    ):
        """Initialize WebSocket API server."""
        self.tracking_manager = tracking_manager
        self.config = config or {}
        self.connection_manager = WebSocketConnectionManager(self.config)

        # Server configuration
        self.host = self.config.get("host", "0.0.0.0")
        self.port = self.config.get("port", 8765)
        self.enabled = self.config.get("enabled", True)

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._server_running = False

        # WebSocket server
        self._websocket_server = None

        logger.info(
            f"WebSocket API Server initialized on {self.host}:{self.port}"
        )

    async def initialize(self) -> None:
        """Initialize the WebSocket API server."""
        try:
            if not self.enabled:
                logger.info("WebSocket API server disabled in configuration")
                return

            # Start background tasks
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            acknowledgment_task = asyncio.create_task(
                self._acknowledgment_timeout_loop()
            )

            self._background_tasks.extend(
                [heartbeat_task, cleanup_task, acknowledgment_task]
            )

            logger.info("WebSocket API server initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket API server: {e}")
            raise WebSocketConnectionError(
                "Failed to initialize WebSocket server", cause=e
            )

    async def start(self) -> None:
        """Start the WebSocket API server."""
        try:
            if not self.enabled:
                return

            # Start WebSocket server
            self._websocket_server = await websockets.serve(
                self._handle_websocket_connection,
                self.host,
                self.port,
                ping_interval=30,
                ping_timeout=10,
                max_size=1024 * 1024,  # 1MB max message size
                max_queue=32,
            )

            self._server_running = True
            logger.info(
                f"WebSocket API server started on ws://{self.host}:{self.port}"
            )

            # Register with tracking manager if available
            if self.tracking_manager:
                await self._register_with_tracking_manager()

        except Exception as e:
            logger.error(f"Failed to start WebSocket API server: {e}")
            raise WebSocketConnectionError(
                "Failed to start WebSocket server", cause=e
            )

    async def stop(self) -> None:
        """Stop the WebSocket API server."""
        try:
            logger.info("Stopping WebSocket API server...")

            # Signal shutdown
            self._shutdown_event.set()
            self._server_running = False

            # Close WebSocket server
            if self._websocket_server:
                self._websocket_server.close()
                await self._websocket_server.wait_closed()

            # Stop background tasks
            for task in self._background_tasks:
                task.cancel()

            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks, return_exceptions=True
                )

            # Close all connections
            await self._close_all_connections()

            logger.info("WebSocket API server stopped successfully")

        except Exception as e:
            logger.error(f"Error stopping WebSocket API server: {e}")

    async def publish_prediction_update(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Publish prediction update to WebSocket clients."""
        try:
            # Create prediction update message
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PREDICTION_UPDATE,
                timestamp=datetime.utcnow(),
                endpoint="/ws/predictions",
                room_id=room_id,
                data=self._format_prediction_data(
                    prediction_result, room_id, current_state
                ),
            )

            # Broadcast to all prediction subscribers
            predictions_sent = (
                await self.connection_manager.broadcast_to_endpoint(
                    "/ws/predictions", message
                )
            )

            # Broadcast to room-specific subscribers
            room_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.PREDICTION_UPDATE,
                timestamp=datetime.utcnow(),
                endpoint=f"/ws/room/{room_id}",
                room_id=room_id,
                data=message.data,
            )

            room_sent = await self.connection_manager.broadcast_to_endpoint(
                f"/ws/room/{room_id}", room_message, room_filter=room_id
            )

            logger.debug(
                f"Published prediction update for {room_id} to {predictions_sent + room_sent} clients"
            )

            return {
                "success": True,
                "clients_notified": predictions_sent + room_sent,
                "predictions_endpoint": predictions_sent,
                "room_endpoint": room_sent,
            }

        except Exception as e:
            logger.error(f"Failed to publish prediction update: {e}")
            return {"success": False, "error": str(e)}

    async def publish_system_status_update(
        self, status_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish system status update to WebSocket clients."""
        try:
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SYSTEM_STATUS_UPDATE,
                timestamp=datetime.utcnow(),
                endpoint="/ws/system-status",
                data=status_data,
            )

            sent = await self.connection_manager.broadcast_to_endpoint(
                "/ws/system-status", message
            )

            logger.debug(f"Published system status update to {sent} clients")

            return {"success": True, "clients_notified": sent}

        except Exception as e:
            logger.error(f"Failed to publish system status update: {e}")
            return {"success": False, "error": str(e)}

    async def publish_alert_notification(
        self, alert_data: Dict[str, Any], room_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Publish alert notification to WebSocket clients."""
        try:
            message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.ALERT_NOTIFICATION,
                timestamp=datetime.utcnow(),
                endpoint="/ws/alerts",
                room_id=room_id,
                data=alert_data,
                requires_ack=True,  # Alerts require acknowledgment
            )

            sent = await self.connection_manager.broadcast_to_endpoint(
                "/ws/alerts", message
            )

            logger.info(f"Published alert notification to {sent} clients")

            return {"success": True, "clients_notified": sent}

        except Exception as e:
            logger.error(f"Failed to publish alert notification: {e}")
            return {"success": False, "error": str(e)}

    def get_server_stats(self) -> Dict[str, Any]:
        """Get comprehensive server statistics."""
        connection_stats = self.connection_manager.get_connection_stats()

        return {
            "server_running": self._server_running,
            "host": self.host,
            "port": self.port,
            "enabled": self.enabled,
            "connection_stats": connection_stats,
            "background_tasks": len(self._background_tasks),
            "tracking_manager_integrated": self.tracking_manager is not None,
        }

    # Private methods

    async def _handle_websocket_connection(
        self, websocket: WebSocketServerProtocol, path: str
    ):
        """Handle incoming WebSocket connection."""
        connection_id = None
        try:
            # Determine endpoint from path
            endpoint = self._normalize_endpoint(path)

            # Register connection
            connection_id = await self.connection_manager.connect(
                websocket, endpoint
            )

            # Send welcome message
            welcome_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.CONNECTION,
                timestamp=datetime.utcnow(),
                endpoint=endpoint,
                data={
                    "message": f"Connected to {endpoint}",
                    "connection_id": connection_id,
                    "server_time": datetime.utcnow().isoformat(),
                    "authentication_required": True,
                    "supported_message_types": [
                        mt.value for mt in MessageType
                    ],
                },
            )

            await websocket.send(welcome_message.to_json())

            # Handle incoming messages
            async for raw_message in websocket:
                try:
                    await self._process_client_message(
                        connection_id, raw_message
                    )
                except json.JSONDecodeError:
                    await self._send_error_message(
                        connection_id,
                        "Invalid JSON format",
                        "JSON_DECODE_ERROR",
                    )
                except Exception as e:
                    logger.error(
                        f"Error processing message from {connection_id}: {e}"
                    )
                    await self._send_error_message(
                        connection_id,
                        f"Message processing error: {str(e)}",
                        "MESSAGE_ERROR",
                    )

        except websockets.exceptions.ConnectionClosed:
            logger.debug(f"WebSocket client {connection_id} disconnected")
        except Exception as e:
            logger.error(f"WebSocket connection error: {e}")
        finally:
            if connection_id:
                await self.connection_manager.disconnect(connection_id)

    async def _process_client_message(
        self, connection_id: str, raw_message: str
    ):
        """Process incoming message from client."""
        try:
            message_data = json.loads(raw_message)
            message_type = MessageType(message_data.get("type"))

            if message_type == MessageType.AUTHENTICATION:
                await self._handle_authentication(
                    connection_id, message_data["data"]
                )
            elif message_type == MessageType.SUBSCRIBE:
                await self._handle_subscription(
                    connection_id, message_data["data"]
                )
            elif message_type == MessageType.UNSUBSCRIBE:
                await self._handle_unsubscription(
                    connection_id, message_data["data"]
                )
            elif message_type == MessageType.HEARTBEAT:
                await self._handle_heartbeat_response(
                    connection_id, message_data
                )
            elif message_type == MessageType.ACKNOWLEDGE:
                await self._handle_acknowledgment(connection_id, message_data)
            else:
                await self._send_error_message(
                    connection_id,
                    f"Unsupported message type: {message_type.value}",
                    "UNSUPPORTED_MESSAGE_TYPE",
                )

        except ValueError as e:
            await self._send_error_message(
                connection_id,
                f"Invalid message type: {str(e)}",
                "INVALID_MESSAGE_TYPE",
            )

    async def _handle_authentication(
        self, connection_id: str, auth_data: Dict[str, Any]
    ):
        """Handle client authentication request."""
        try:
            auth_request = ClientAuthRequest(**auth_data)
            success = await self.connection_manager.authenticate_client(
                connection_id, auth_request
            )

            response_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.AUTHENTICATION,
                timestamp=datetime.utcnow(),
                endpoint=self.connection_manager.connections[
                    connection_id
                ].endpoint,
                data={
                    "success": success,
                    "message": (
                        "Authentication successful"
                        if success
                        else "Authentication failed"
                    ),
                    "capabilities": list(auth_request.capabilities),
                    "room_filters": list(auth_request.room_filters),
                },
            )

            await self.connection_manager.send_message(
                connection_id, response_message
            )

        except WebSocketAuthenticationError as e:
            await self._send_error_message(
                connection_id, str(e), "AUTHENTICATION_ERROR"
            )
        except Exception as e:
            logger.error(f"Authentication error for {connection_id}: {e}")
            await self._send_error_message(
                connection_id,
                "Authentication processing error",
                "AUTH_PROCESSING_ERROR",
            )

    async def _handle_subscription(
        self, connection_id: str, subscription_data: Dict[str, Any]
    ):
        """Handle client subscription request."""
        try:
            subscription = ClientSubscription(**subscription_data)
            success = await self.connection_manager.subscribe_client(
                connection_id, subscription
            )

            response_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SUBSCRIPTION_STATUS,
                timestamp=datetime.utcnow(),
                endpoint=subscription.endpoint,
                room_id=subscription.room_id,
                data={
                    "action": "subscribe",
                    "success": success,
                    "endpoint": subscription.endpoint,
                    "room_id": subscription.room_id,
                    "message": (
                        f"Subscribed to {subscription.endpoint}"
                        if success
                        else "Subscription failed"
                    ),
                },
            )

            await self.connection_manager.send_message(
                connection_id, response_message
            )

        except (WebSocketAuthenticationError, WebSocketValidationError) as e:
            await self._send_error_message(
                connection_id, str(e), "SUBSCRIPTION_ERROR"
            )
        except Exception as e:
            logger.error(f"Subscription error for {connection_id}: {e}")
            await self._send_error_message(
                connection_id,
                "Subscription processing error",
                "SUBSCRIPTION_PROCESSING_ERROR",
            )

    async def _handle_unsubscription(
        self, connection_id: str, unsubscription_data: Dict[str, Any]
    ):
        """Handle client unsubscription request."""
        try:
            endpoint = unsubscription_data["endpoint"]
            room_id = unsubscription_data.get("room_id")

            success = await self.connection_manager.unsubscribe_client(
                connection_id, endpoint, room_id
            )

            response_message = WebSocketMessage(
                message_id=str(uuid.uuid4()),
                message_type=MessageType.SUBSCRIPTION_STATUS,
                timestamp=datetime.utcnow(),
                endpoint=endpoint,
                room_id=room_id,
                data={
                    "action": "unsubscribe",
                    "success": success,
                    "endpoint": endpoint,
                    "room_id": room_id,
                    "message": (
                        f"Unsubscribed from {endpoint}"
                        if success
                        else "Unsubscription failed"
                    ),
                },
            )

            await self.connection_manager.send_message(
                connection_id, response_message
            )

        except Exception as e:
            logger.error(f"Unsubscription error for {connection_id}: {e}")
            await self._send_error_message(
                connection_id,
                "Unsubscription processing error",
                "UNSUBSCRIPTION_PROCESSING_ERROR",
            )

    async def _handle_heartbeat_response(
        self, connection_id: str, heartbeat_data: Dict[str, Any]
    ):
        """Handle heartbeat response from client."""
        if connection_id in self.connection_manager.connections:
            self.connection_manager.connections[
                connection_id
            ].update_heartbeat()

    async def _handle_acknowledgment(
        self, connection_id: str, ack_data: Dict[str, Any]
    ):
        """Handle message acknowledgment from client."""
        message_id = ack_data.get("message_id")
        if message_id:
            await self.connection_manager.process_acknowledgment(
                connection_id, message_id
            )

    async def _send_error_message(
        self, connection_id: str, error_message: str, error_code: str
    ):
        """Send error message to client."""
        error_msg = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            timestamp=datetime.utcnow(),
            endpoint=self.connection_manager.connections.get(
                connection_id, {}
            ).endpoint
            or "unknown",
            data={
                "error": error_message,
                "error_code": error_code,
                "timestamp": datetime.utcnow().isoformat(),
            },
        )

        await self.connection_manager.send_message(connection_id, error_msg)

    def _normalize_endpoint(self, path: str) -> str:
        """Normalize WebSocket path to standard endpoint."""
        if path.startswith("/ws/room/"):
            return path  # Keep room-specific paths as-is
        elif path.startswith("/ws/"):
            return path
        else:
            return "/ws/predictions"  # Default endpoint

    def _format_prediction_data(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str],
    ) -> Dict[str, Any]:
        """Format prediction result for WebSocket transmission."""
        config = get_config()
        room_name = config.rooms.get(room_id, {}).get(
            "name", room_id.replace("_", " ").title()
        )

        time_until = prediction_result.predicted_time - datetime.utcnow()
        time_until_seconds = max(0, int(time_until.total_seconds()))

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
                for alt_time, alt_confidence in (
                    prediction_result.alternatives or []
                )[:3]
            ],
            "features_used": len(prediction_result.features_used or []),
            "prediction_metadata": prediction_result.prediction_metadata,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def _format_time_until(self, seconds: int) -> str:
        """Format seconds into human-readable time."""
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

    async def _heartbeat_loop(self):
        """Background task for sending heartbeats."""
        while not self._shutdown_event.is_set():
            try:
                # Send heartbeats to all authenticated connections
                for connection_id, connection in list(
                    self.connection_manager.connections.items()
                ):
                    if connection.authenticated:
                        try:
                            await self.connection_manager.send_heartbeat(
                                connection_id
                            )
                        except Exception as e:
                            logger.warning(
                                f"Failed to send heartbeat to {connection_id}: {e}"
                            )

                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self.connection_manager.heartbeat_interval,
                )

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in heartbeat loop: {e}")
                await asyncio.sleep(30)

    async def _cleanup_loop(self):
        """Background task for connection cleanup."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()
                stale_connections = []

                # Find stale connections
                for (
                    connection_id,
                    connection,
                ) in self.connection_manager.connections.items():
                    # Check for timeout
                    if (
                        current_time - connection.last_activity
                    ).total_seconds() > self.connection_manager.connection_timeout:
                        stale_connections.append(connection_id)
                    # Check for missed heartbeats
                    elif connection.authenticated and (
                        current_time - connection.last_heartbeat
                    ).total_seconds() > (
                        self.connection_manager.heartbeat_interval * 3
                    ):
                        stale_connections.append(connection_id)

                # Clean up stale connections
                for connection_id in stale_connections:
                    await self.connection_manager.disconnect(connection_id)
                    logger.info(
                        f"Cleaned up stale connection: {connection_id}"
                    )

                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=60
                )  # Run every minute

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)

    async def _acknowledgment_timeout_loop(self):
        """Background task for handling message acknowledgment timeouts."""
        while not self._shutdown_event.is_set():
            try:
                current_time = datetime.utcnow()

                for connection in self.connection_manager.connections.values():
                    expired_messages = []

                    for (
                        msg_id,
                        message,
                    ) in connection.unacknowledged_messages.items():
                        # Check if message has expired
                        if (
                            current_time - message.timestamp
                        ).total_seconds() > self.connection_manager.message_acknowledgment_timeout:
                            expired_messages.append(msg_id)

                    # Remove expired messages
                    for msg_id in expired_messages:
                        del connection.unacknowledged_messages[msg_id]
                        logger.warning(
                            f"Message {msg_id} acknowledgment timeout for connection {connection.connection_id}"
                        )

                await asyncio.wait_for(
                    self._shutdown_event.wait(), timeout=30
                )  # Check every 30 seconds

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in acknowledgment timeout loop: {e}")
                await asyncio.sleep(30)

    async def _close_all_connections(self):
        """Close all active WebSocket connections."""
        try:
            connection_ids = list(self.connection_manager.connections.keys())
            for connection_id in connection_ids:
                await self.connection_manager.disconnect(connection_id)

            logger.info("All WebSocket connections closed")

        except Exception as e:
            logger.error(f"Error closing WebSocket connections: {e}")

    async def _register_with_tracking_manager(self):
        """Register WebSocket API with tracking manager for automatic updates."""
        try:
            if self.tracking_manager:
                # Add callback for prediction updates
                self.tracking_manager.add_notification_callback(
                    self._handle_tracking_manager_callback
                )

                logger.info("WebSocket API registered with TrackingManager")

        except Exception as e:
            logger.error(f"Failed to register with TrackingManager: {e}")

    async def _handle_tracking_manager_callback(self, callback_data: Any):
        """Handle callbacks from TrackingManager."""
        try:
            # This would be called when the tracking manager has updates
            # Implementation depends on the specific callback data format
            logger.debug(
                f"Received TrackingManager callback: {type(callback_data)}"
            )

        except Exception as e:
            logger.error(f"Error handling TrackingManager callback: {e}")


# Exception classes for WebSocket API


class WebSocketAPIError(OccupancyPredictionError):
    """Base exception for WebSocket API errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_API_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )


class WebSocketAuthenticationError(WebSocketAPIError):
    """Raised when WebSocket authentication fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_AUTH_ERROR",
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class WebSocketConnectionError(WebSocketAPIError):
    """Raised when WebSocket connection operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_CONNECTION_ERROR",
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class WebSocketRateLimitError(WebSocketAPIError):
    """Raised when WebSocket rate limits are exceeded."""

    def __init__(self, client_id: str, limit: int, **kwargs):
        message = f"Rate limit exceeded for client {client_id}: {limit} messages/minute"
        super().__init__(
            message=message,
            error_code="WEBSOCKET_RATE_LIMIT_ERROR",
            severity=ErrorSeverity.LOW,
            context={"client_id": client_id, "limit": limit},
            **kwargs,
        )


class WebSocketValidationError(WebSocketAPIError):
    """Raised when WebSocket message validation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_VALIDATION_ERROR",
            severity=ErrorSeverity.LOW,
            **kwargs,
        )


# Starlette Application Setup for WebSocket API

async def websocket_endpoint(websocket):
    """Handle WebSocket connections for all endpoints."""
    from starlette.websockets import WebSocket
    
    # Validate websocket is proper WebSocket instance
    if not isinstance(websocket, WebSocket):
        logger.error("Invalid WebSocket connection provided")
        return
    
    ws_api = None
    connection_id = None
    
    try:
        # Accept connection
        await websocket.accept()
        
        # Get the WebSocket API instance (this would typically be passed as a dependency)
        ws_api = WebSocketAPIServer()
        
        # Determine endpoint from path
        endpoint_path = websocket.url.path
        
        # Register connection
        connection_id = await ws_api.connection_manager.connect(
            websocket=websocket, 
            endpoint=endpoint_path
        )
        
        logger.info(f"WebSocket client {connection_id} connected to {endpoint_path}")
        
        # Handle messages
        while True:
            try:
                # Receive message
                raw_message = await websocket.receive_text()
                
                # Process message through WebSocket API
                response = await ws_api._handle_websocket_message(
                    connection_id, raw_message
                )
                
                # Send response if any
                if response:
                    await websocket.send_text(response)
                    
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {e}")
                await websocket.send_text(json.dumps({
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }))
                break
                
    except Exception as e:
        logger.error(f"WebSocket connection error: {e}")
    finally:
        # Clean up connection
        if ws_api and connection_id:
            await ws_api.connection_manager.disconnect(connection_id)


async def health_endpoint(request):
    """Health check endpoint for WebSocket API."""
    # Use JSONResponse for structured health information
    return JSONResponse({
        "status": "healthy",
        "service": "websocket_api",
        "timestamp": datetime.utcnow().isoformat(),
        "connections": "API not initialized"
    })


def create_websocket_app() -> Starlette:
    """Create Starlette application with WebSocket support."""
    
    async def simple_health_endpoint(request):
        """Simple health check endpoint returning plain text."""
        return PlainTextResponse("WebSocket API is healthy")

    # Define routes
    routes = [
        Route("/health", health_endpoint, methods=["GET"]),
        Route("/health/simple", simple_health_endpoint, methods=["GET"]),
        WebSocketRoute("/ws/predictions", websocket_endpoint),
        WebSocketRoute("/ws/system-status", websocket_endpoint),
        WebSocketRoute("/ws/alerts", websocket_endpoint),
        WebSocketRoute("/ws/room/{room_id}", websocket_endpoint),
    ]
    
    # Create application
    app = Starlette(routes=routes, debug=False)
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )
    
    return app


# Factory function to create configured WebSocket API server
def create_websocket_api_server(
    config: Optional[Dict[str, Any]] = None,
    tracking_manager=None
) -> WebSocketAPIServer:
    """Create and configure WebSocket API server."""
    
    # Get default config if none provided
    if config is None:
        system_config = get_config()
        config = {
            "host": getattr(system_config, "websocket_host", "0.0.0.0"),
            "port": getattr(system_config, "websocket_port", 8765),
            "max_connections": 1000,
            "max_messages_per_minute": 60,
            "heartbeat_interval_seconds": 30,
            "connection_timeout_seconds": 300,
        }
    
    # Create server instance
    server = WebSocketAPIServer(config=config, tracking_manager=tracking_manager)
    
    return server


@asynccontextmanager
async def websocket_api_context(
    config: Optional[Dict[str, Any]] = None,
    tracking_manager=None
):
    """Context manager for WebSocket API server lifecycle."""
    server = create_websocket_api_server(config, tracking_manager)
    
    try:
        await server.start()
        yield server
    finally:
        await server.stop()
