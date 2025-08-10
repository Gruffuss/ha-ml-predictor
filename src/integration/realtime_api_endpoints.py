"""
Real-time API Endpoints for FastAPI Integration.

This module provides FastAPI endpoints for real-time prediction streaming
via WebSocket and Server-Sent Events, integrating seamlessly with the
existing API server and TrackingManager.

Features:
- WebSocket endpoints for real-time predictions
- Server-Sent Events (SSE) endpoints
- Room-specific subscription management
- Integration with existing API authentication
- Real-time connection monitoring
- Performance metrics for real-time endpoints
"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
from typing import Any, Dict, List, Optional, Set

from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from ..core.exceptions import APIAuthenticationError, APIError
from .realtime_publisher import RealtimePredictionEvent
from .tracking_integration import TrackingIntegrationManager

logger = logging.getLogger(__name__)


# Global integration manager (set by main system)
_integration_manager: Optional[TrackingIntegrationManager] = None


def set_integration_manager(manager: TrackingIntegrationManager):
    """Set the global integration manager."""
    global _integration_manager
    _integration_manager = manager


def get_integration_manager() -> TrackingIntegrationManager:
    """Get the integration manager."""
    if _integration_manager is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Real-time publishing not available",
        )
    return _integration_manager


# Request/Response Models


class WebSocketSubscription(BaseModel):
    """WebSocket subscription request model."""

    action: str  # 'subscribe' or 'unsubscribe'
    room_id: str
    client_metadata: Optional[Dict[str, Any]] = None


class RealtimeStatsResponse(BaseModel):
    """Response model for real-time statistics."""

    websocket_connections: int
    sse_connections: int
    total_active_connections: int
    predictions_published_last_minute: int
    channels_active: List[str]
    uptime_seconds: float


# Create router
realtime_router = APIRouter(prefix="/realtime", tags=["real-time"])


@realtime_router.websocket("/predictions")
async def websocket_predictions_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time prediction streaming.

    Clients can connect to this endpoint to receive real-time predictions
    for all rooms or subscribe to specific rooms.
    """
    try:
        integration_manager = get_integration_manager()

        # Get WebSocket handler from integration manager
        websocket_handler = integration_manager.get_websocket_handler()
        if not websocket_handler:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Accept connection
        await websocket.accept()

        # Delegate to integration manager's WebSocket handler
        await websocket_handler(websocket, "/predictions")

    except Exception as e:
        logger.error(f"Error in WebSocket predictions endpoint: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass


@realtime_router.websocket("/predictions/{room_id}")
async def websocket_room_predictions_endpoint(websocket: WebSocket, room_id: str):
    """
    WebSocket endpoint for room-specific prediction streaming.

    Clients connect to receive predictions for a specific room only.
    """
    client_id = None
    try:
        integration_manager = get_integration_manager()

        # Get WebSocket handler
        websocket_handler = integration_manager.get_websocket_handler()
        if not websocket_handler:
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
            return

        # Accept connection
        await websocket.accept()

        # Generate unique client ID for tracking
        client_id = f"room_{room_id}_{datetime.utcnow().timestamp()}"

        # Send initial subscription message
        subscription_message = {
            "type": "subscription_confirmation",
            "room_id": room_id,
            "client_id": client_id,
            "auto_subscribed": True,
            "message": f"Successfully subscribed to {room_id} predictions",
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Send subscription confirmation to client
        await websocket.send_text(json.dumps(subscription_message))
        logger.info(
            f"Sent subscription confirmation to client {client_id} for room {room_id}"
        )

        try:
            # Get WebSocket manager for proper connection handling
            if hasattr(
                integration_manager.enhanced_mqtt_manager, "realtime_publisher"
            ) and hasattr(
                integration_manager.enhanced_mqtt_manager.realtime_publisher,
                "websocket_manager",
            ):
                ws_manager = (
                    integration_manager.enhanced_mqtt_manager.realtime_publisher.websocket_manager
                )

                # Register connection with WebSocket manager using FastAPI websocket
                await api_websocket_handler.connect(websocket, client_id)

                # Subscribe client to room updates through the manager
                await ws_manager.subscribe_to_room(client_id, room_id)

                logger.info(
                    f"Client {client_id} connected and subscribed to room {room_id}"
                )

                # Handle incoming messages from client
                try:
                    while True:
                        try:
                            # Wait for messages from client
                            data = await websocket.receive_text()
                            message = json.loads(data)

                            # Handle client subscription management
                            await _handle_client_websocket_message(
                                client_id, message, ws_manager, websocket
                            )

                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from client {client_id}")
                            error_msg = {
                                "type": "error",
                                "message": "Invalid JSON format",
                                "timestamp": datetime.utcnow().isoformat(),
                            }
                            await websocket.send_text(json.dumps(error_msg))

                except Exception as e:
                    if "websocket.disconnect" not in str(e).lower():
                        logger.error(
                            f"Error handling client messages for {client_id}: {e}"
                        )

            else:
                # Fallback to direct websocket handler
                await websocket_handler(websocket, f"/predictions/{room_id}")

        except Exception as e:
            logger.error(f"Error setting up room-specific WebSocket for {room_id}: {e}")
            await websocket.close(code=status.WS_1011_INTERNAL_ERROR)

    except WebSocketDisconnect:
        logger.info(f"Client {client_id} disconnected from room {room_id}")
    except Exception as e:
        logger.error(f"Error in WebSocket room predictions endpoint for {room_id}: {e}")
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.close(code=status.WS_1011_INTERNAL_ERROR)
        except Exception:
            pass
    finally:
        # Cleanup: disconnect client from API handler
        if client_id:
            await api_websocket_handler.disconnect(client_id)
            logger.debug(f"Cleaned up connection for client {client_id}")


@realtime_router.get("/events")
async def sse_predictions_endpoint(request: Request):
    """
    Server-Sent Events endpoint for real-time prediction streaming.

    Returns a streaming response with real-time predictions for all rooms.
    """
    try:
        integration_manager = get_integration_manager()

        # Get SSE handler from integration manager
        sse_handler = integration_manager.get_sse_handler()
        if not sse_handler:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SSE streaming not available",
            )

        # Create SSE stream
        return await sse_handler()

    except Exception as e:
        logger.error(f"Error in SSE predictions endpoint: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create SSE stream",
        )


@realtime_router.get("/events/{room_id}")
async def sse_room_predictions_endpoint(request: Request, room_id: str):
    """
    Server-Sent Events endpoint for room-specific prediction streaming.

    Returns a streaming response with real-time predictions for a specific room.
    """
    try:
        integration_manager = get_integration_manager()

        # Get SSE handler
        sse_handler = integration_manager.get_sse_handler()
        if not sse_handler:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="SSE streaming not available",
            )

        # Create room-specific SSE stream with StreamingResponse wrapper
        sse_stream = await sse_handler(room_id=room_id)
        if not isinstance(sse_stream, StreamingResponse):
            # If not already a streaming response, wrap it
            return StreamingResponse(
                sse_stream,
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
            )
        return sse_stream

    except Exception as e:
        logger.error(f"Error in SSE room predictions endpoint for {room_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create room-specific SSE stream",
        )


@realtime_router.get("/stats", response_model=RealtimeStatsResponse)
async def get_realtime_stats(request: Request, api_key: Optional[str] = Depends(None)):
    """
    Get real-time publishing statistics.

    Returns current statistics for WebSocket and SSE connections.
    """
    # Basic API key validation for stats endpoint
    try:
        if api_key and len(api_key) < 10:
            raise APIAuthenticationError(
                endpoint="/realtime/stats", reason="Invalid API key format"
            )

        integration_manager = get_integration_manager()
        if not integration_manager:
            raise APIError("Integration manager not available")

        realtime_publisher = integration_manager.get_realtime_publisher()
        if not realtime_publisher:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Real-time publishing not available",
            )

    except (APIAuthenticationError, APIError) as e:
        logger.warning(f"API error in realtime stats: {e}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    try:
        integration_manager = get_integration_manager()

        # Get integration stats
        stats = integration_manager.get_integration_stats()

        # Extract relevant metrics
        enhanced_mqtt_stats = stats.get("enhanced_mqtt_stats", {})
        connection_info = enhanced_mqtt_stats.get("connections", {})
        realtime_stats = enhanced_mqtt_stats.get("realtime_publishing", {})

        return RealtimeStatsResponse(
            websocket_connections=connection_info.get("websocket_clients", 0),
            sse_connections=connection_info.get("sse_clients", 0),
            total_active_connections=connection_info.get("total_clients", 0),
            predictions_published_last_minute=int(
                enhanced_mqtt_stats.get("performance", {}).get(
                    "predictions_per_minute", 0
                )
            ),
            channels_active=realtime_stats.get("enabled_channels", []),
            uptime_seconds=realtime_stats.get("uptime_seconds", 0),
        )

    except Exception as e:
        logger.error(f"Error getting real-time stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get real-time statistics",
        )


@realtime_router.get("/connections")
async def get_realtime_connections():
    """
    Get detailed information about active real-time connections.

    Returns detailed connection information for monitoring and debugging.
    """
    try:
        integration_manager = get_integration_manager()

        # Get connection information
        stats = integration_manager.get_integration_stats()
        connection_info = stats.get("enhanced_mqtt_stats", {}).get("connections", {})

        # Get API WebSocket handler session information
        api_sessions = api_websocket_handler.get_all_sessions()
        api_connection_count = api_websocket_handler.get_connection_count()

        return {
            "websocket_connections": connection_info.get("websocket_connections", {}),
            "sse_connections": connection_info.get("sse_connections", {}),
            "api_websocket_sessions": {
                "count": api_connection_count,
                "sessions": {
                    client_id: {
                        **session,
                        "connected_at": session["connected_at"].isoformat(),
                        "last_activity": session["last_activity"].isoformat(),
                        "subscriptions": list(session["subscriptions"]),
                    }
                    for client_id, session in api_sessions.items()
                },
            },
            "summary": {
                "total_active": connection_info.get("total_active_connections", 0)
                + api_connection_count,
                "websocket_count": connection_info.get("websocket_clients", 0),
                "sse_count": connection_info.get("sse_clients", 0),
                "api_websocket_count": api_connection_count,
            },
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting connection information: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get connection information",
        )


@realtime_router.post("/connections/cleanup")
async def cleanup_stale_connections():
    """
    Manually trigger cleanup of stale WebSocket connections.

    Removes connections that have been idle for more than 30 minutes.
    """
    try:
        cleaned_count = await api_websocket_handler.cleanup_stale_connections()

        return {
            "success": True,
            "message": f"Cleaned up {cleaned_count} stale connections",
            "connections_cleaned": cleaned_count,
            "active_connections_remaining": api_websocket_handler.get_connection_count(),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error during connection cleanup: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to cleanup connections",
        )


@realtime_router.post("/broadcast/test")
async def test_realtime_broadcast():
    """
    Test endpoint for broadcasting a test message to all real-time clients.

    Useful for testing real-time connectivity and debugging.
    """
    try:
        integration_manager = get_integration_manager()

        # Create test event using RealtimePredictionEvent
        test_event = RealtimePredictionEvent(
            event_id=f"test-broadcast-{asyncio.get_event_loop().time()}",
            event_type="api_test_broadcast",
            timestamp=datetime.utcnow(),
            room_id=None,
            data={
                "message": "Test broadcast from API",
                "timestamp": datetime.utcnow().isoformat(),
                "type": "test",
                "source": "api_endpoint",
            },
        )

        # Create test data with room filtering using Set type
        target_rooms: Set[str] = {"living_room", "kitchen", "bedroom"}

        test_data = test_event.data

        # Broadcast via enhanced MQTT manager
        if integration_manager.enhanced_mqtt_manager:
            results = (
                await integration_manager.enhanced_mqtt_manager.publish_system_status(
                    {"test_broadcast": test_data}
                )
            )

            return {
                "success": True,
                "message": "Test broadcast sent",
                "results": results,
                "timestamp": datetime.utcnow().isoformat(),
            }
        else:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Real-time broadcasting not available",
            )

    except Exception as e:
        logger.error(f"Error in test broadcast: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send test broadcast",
        )


@realtime_router.get("/health")
async def realtime_health_check():
    """
    Health check endpoint for real-time publishing system.

    Returns the status of real-time publishing components.
    """
    try:
        integration_manager = get_integration_manager()

        # Get integration stats for health check
        stats = integration_manager.get_integration_stats()

        # Determine health status
        integration_active = stats.get("integration_active", False)
        enhanced_mqtt_available = "enhanced_mqtt_stats" in stats

        health_status = (
            "healthy" if integration_active and enhanced_mqtt_available else "degraded"
        )

        return {
            "status": health_status,
            "integration_active": integration_active,
            "enhanced_mqtt_available": enhanced_mqtt_available,
            "realtime_publishing_enabled": stats.get("integration_config", {}).get(
                "realtime_publishing_enabled", False
            ),
            "websocket_enabled": stats.get("integration_config", {}).get(
                "websocket_enabled", False
            ),
            "sse_enabled": stats.get("integration_config", {}).get(
                "sse_enabled", False
            ),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error in real-time health check: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
        }


async def _handle_client_websocket_message(
    client_id: str, message: Dict[str, Any], ws_manager, websocket: WebSocket
):
    """
    Handle WebSocket messages from clients for subscription management.

    Args:
        client_id: Unique client identifier
        message: Message from client
        ws_manager: WebSocket connection manager
        websocket: FastAPI WebSocket connection
    """
    try:
        message_type = message.get("type")
        room_id = message.get("room_id")

        if message_type == "subscribe" and room_id:
            # Subscribe to additional room
            await ws_manager.subscribe_to_room(client_id, room_id)
            response = {
                "type": "subscription_success",
                "room_id": room_id,
                "message": f"Subscribed to {room_id}",
                "timestamp": datetime.utcnow().isoformat(),
            }
            await websocket.send_text(json.dumps(response))
            logger.debug(f"Client {client_id} subscribed to additional room {room_id}")

        elif message_type == "unsubscribe" and room_id:
            # Unsubscribe from room
            await ws_manager.unsubscribe_from_room(client_id, room_id)
            response = {
                "type": "unsubscription_success",
                "room_id": room_id,
                "message": f"Unsubscribed from {room_id}",
                "timestamp": datetime.utcnow().isoformat(),
            }
            await websocket.send_text(json.dumps(response))
            logger.debug(f"Client {client_id} unsubscribed from room {room_id}")

        elif message_type == "ping":
            # Handle ping/pong for connection health
            response = {"type": "pong", "timestamp": datetime.utcnow().isoformat()}
            await websocket.send_text(json.dumps(response))

        elif message_type == "get_subscriptions":
            # Return current subscriptions
            client_meta = ws_manager.client_metadata.get(client_id)
            subscriptions = list(client_meta.room_subscriptions) if client_meta else []
            response = {
                "type": "subscription_list",
                "subscriptions": subscriptions,
                "timestamp": datetime.utcnow().isoformat(),
            }
            await websocket.send_text(json.dumps(response))

        else:
            # Unknown message type
            response = {
                "type": "error",
                "message": f"Unknown message type: {message_type}",
                "timestamp": datetime.utcnow().isoformat(),
            }
            await websocket.send_text(json.dumps(response))

    except Exception as e:
        logger.error(f"Error handling client message from {client_id}: {e}")
        error_response = {
            "type": "error",
            "message": "Error processing message",
            "timestamp": datetime.utcnow().isoformat(),
        }
        try:
            await websocket.send_text(json.dumps(error_response))
        except Exception:
            pass  # Connection may be closed


# WebSocket connection manager for endpoint-specific handling
class WebSocketConnectionHandler:
    """
    Handles WebSocket connections at the endpoint level.

    This provides additional connection management specifically for
    API endpoints while delegating to the main real-time system.
    """

    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.client_sessions: Dict[str, Dict[str, Any]] = (
            {}
        )  # Track client session data

    async def connect(self, websocket: WebSocket, connection_id: str = None) -> str:
        """Connect a WebSocket client."""
        if connection_id is None:
            connection_id = f"api_{datetime.utcnow().timestamp()}"

        # WebSocket is already accepted in the endpoint, don't accept again
        self.active_connections[connection_id] = websocket

        # Initialize client session data
        self.client_sessions[connection_id] = {
            "connected_at": datetime.utcnow(),
            "last_activity": datetime.utcnow(),
            "subscriptions": set(),
            "message_count": 0,
        }

        logger.debug(f"API WebSocket client connected: {connection_id}")
        return connection_id

    async def disconnect(self, connection_id: str):
        """Disconnect a WebSocket client."""
        if connection_id in self.active_connections:
            del self.active_connections[connection_id]
        if connection_id in self.client_sessions:
            del self.client_sessions[connection_id]
        logger.debug(f"API WebSocket client disconnected: {connection_id}")

    async def send_message(self, connection_id: str, message: Dict[str, Any]) -> bool:
        """Send message to a specific client."""
        if connection_id in self.active_connections:
            try:
                websocket = self.active_connections[connection_id]
                await websocket.send_text(json.dumps(message))

                # Update client session data
                if connection_id in self.client_sessions:
                    self.client_sessions[connection_id][
                        "last_activity"
                    ] = datetime.utcnow()
                    self.client_sessions[connection_id]["message_count"] += 1

                return True
            except Exception as e:
                logger.error(f"Error sending message to {connection_id}: {e}")
                await self.disconnect(connection_id)
        return False

    async def broadcast_message(self, message: Dict[str, Any]) -> int:
        """Broadcast message to all connected clients."""
        successful_sends = 0
        failed_connections = []

        for connection_id, websocket in self.active_connections.items():
            try:
                await websocket.send_text(json.dumps(message))
                successful_sends += 1
            except Exception as e:
                logger.error(f"Error broadcasting to {connection_id}: {e}")
                failed_connections.append(connection_id)

        # Clean up failed connections
        for connection_id in failed_connections:
            await self.disconnect(connection_id)

        return successful_sends

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_client_session_info(self, connection_id: str) -> Optional[Dict[str, Any]]:
        """Get session information for a specific client."""
        return self.client_sessions.get(connection_id)

    def get_all_sessions(self) -> Dict[str, Dict[str, Any]]:
        """Get session information for all clients."""
        return self.client_sessions.copy()

    async def cleanup_stale_connections(self, max_idle_minutes: int = 30):
        """Clean up connections that have been idle too long."""
        cutoff_time = datetime.utcnow() - timedelta(minutes=max_idle_minutes)
        stale_connections = []

        for connection_id, session in self.client_sessions.items():
            if session["last_activity"] < cutoff_time:
                stale_connections.append(connection_id)

        for connection_id in stale_connections:
            await self.disconnect(connection_id)
            logger.debug(f"Cleaned up stale API WebSocket connection: {connection_id}")

        return len(stale_connections)


# Global connection handler for API endpoints
api_websocket_handler = WebSocketConnectionHandler()


# Additional utility endpoints


@realtime_router.get("/channels")
async def get_available_channels():
    """
    Get information about available real-time channels.

    Returns configuration and status of different real-time channels.
    """
    try:
        integration_manager = get_integration_manager()
        stats = integration_manager.get_integration_stats()

        return {
            "available_channels": [
                {
                    "name": "mqtt",
                    "description": "MQTT publishing to Home Assistant",
                    "enabled": True,
                },
                {
                    "name": "websocket",
                    "description": "WebSocket real-time streaming",
                    "enabled": stats.get("integration_config", {}).get(
                        "websocket_enabled", False
                    ),
                },
                {
                    "name": "sse",
                    "description": "Server-Sent Events streaming",
                    "enabled": stats.get("integration_config", {}).get(
                        "sse_enabled", False
                    ),
                },
            ],
            "endpoints": {
                "websocket_predictions": "/realtime/predictions",
                "websocket_room_predictions": "/realtime/predictions/{room_id}",
                "sse_predictions": "/realtime/events",
                "sse_room_predictions": "/realtime/events/{room_id}",
            },
            "configuration": stats.get("integration_config", {}),
            "timestamp": datetime.utcnow().isoformat(),
        }

    except Exception as e:
        logger.error(f"Error getting channel information: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get channel information",
        )
