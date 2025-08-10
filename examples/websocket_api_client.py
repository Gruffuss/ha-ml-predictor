"""
WebSocket API Client Example for Home Assistant Occupancy Prediction System.

This example demonstrates how to connect to and interact with the comprehensive
WebSocket API for real-time predictions, system status, and alerts.

Usage:
    python websocket_api_client.py --api-key YOUR_API_KEY --host localhost --port 8766
    
Features demonstrated:
- Authentication with API key
- Subscription to different endpoints
- Real-time prediction updates
- System status monitoring
- Alert notifications
- Room-specific subscriptions
- Heartbeat handling
- Message acknowledgments
- Error handling and reconnection
"""

import argparse
import asyncio
from datetime import datetime
import json
import logging
from typing import Dict, List, Optional, Set

import websockets
from websockets.client import WebSocketClientProtocol

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class WebSocketAPIClient:
    """
    Example WebSocket API client for the Occupancy Prediction System.

    Demonstrates how to connect, authenticate, subscribe to updates,
    and handle real-time messages from the WebSocket API.
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 8766,
        api_key: str = "",
        client_name: str = "ExampleClient",
    ):
        """Initialize WebSocket API client."""
        self.host = host
        self.port = port
        self.api_key = api_key
        self.client_name = client_name

        # Connection management
        self.websocket: Optional[WebSocketClientProtocol] = None
        self.connection_id: Optional[str] = None
        self.authenticated = False
        self.subscriptions: Set[str] = set()

        # Message handling
        self.message_handlers = {
            "connection": self._handle_connection,
            "authentication": self._handle_authentication,
            "subscription_status": self._handle_subscription_status,
            "prediction_update": self._handle_prediction_update,
            "system_status_update": self._handle_system_status_update,
            "alert_notification": self._handle_alert_notification,
            "drift_notification": self._handle_drift_notification,
            "heartbeat": self._handle_heartbeat,
            "error": self._handle_error,
            "rate_limit_warning": self._handle_rate_limit_warning,
        }

        # Configuration
        self.auto_reconnect = True
        self.reconnect_delay = 5  # seconds
        self.max_reconnect_attempts = 10

        logger.info(f"WebSocket API Client initialized for {host}:{port}")

    async def connect(self) -> bool:
        """Connect to the WebSocket API server."""
        try:
            uri = f"ws://{self.host}:{self.port}/ws/predictions"
            logger.info(f"Connecting to {uri}")

            self.websocket = await websockets.connect(
                uri, ping_interval=30, ping_timeout=10, close_timeout=10
            )

            logger.info("WebSocket connection established")
            return True

        except Exception as e:
            logger.error(f"Failed to connect: {e}")
            return False

    async def disconnect(self):
        """Disconnect from the WebSocket API server."""
        if self.websocket:
            try:
                await self.websocket.close()
                logger.info("WebSocket connection closed")
            except Exception as e:
                logger.error(f"Error closing WebSocket connection: {e}")
            finally:
                self.websocket = None
                self.authenticated = False
                self.connection_id = None
                self.subscriptions.clear()

    async def authenticate(self) -> bool:
        """Authenticate with the WebSocket API server."""
        if not self.websocket:
            logger.error("Not connected to WebSocket server")
            return False

        auth_message = {
            "type": "authentication",
            "data": {
                "api_key": self.api_key,
                "client_name": self.client_name,
                "capabilities": ["predictions", "system_status", "alerts"],
                "room_filters": [],  # Empty means access to all rooms
            },
        }

        try:
            await self.websocket.send(json.dumps(auth_message))
            logger.info("Authentication request sent")
            return True
        except Exception as e:
            logger.error(f"Failed to send authentication: {e}")
            return False

    async def subscribe_to_endpoint(
        self, endpoint: str, room_id: Optional[str] = None
    ) -> bool:
        """Subscribe to a specific endpoint."""
        if not self.authenticated:
            logger.error("Must authenticate before subscribing")
            return False

        subscription_message = {
            "type": "subscribe",
            "data": {"endpoint": endpoint, "room_id": room_id, "filters": {}},
        }

        try:
            await self.websocket.send(json.dumps(subscription_message))
            logger.info(
                f"Subscription request sent for {endpoint}"
                + (f" (room: {room_id})" if room_id else "")
            )
            return True
        except Exception as e:
            logger.error(f"Failed to subscribe to {endpoint}: {e}")
            return False

    async def unsubscribe_from_endpoint(
        self, endpoint: str, room_id: Optional[str] = None
    ) -> bool:
        """Unsubscribe from a specific endpoint."""
        if not self.websocket:
            return False

        unsubscription_message = {
            "type": "unsubscribe",
            "data": {"endpoint": endpoint, "room_id": room_id},
        }

        try:
            await self.websocket.send(json.dumps(unsubscription_message))
            logger.info(f"Unsubscription request sent for {endpoint}")
            return True
        except Exception as e:
            logger.error(f"Failed to unsubscribe from {endpoint}: {e}")
            return False

    async def send_heartbeat(self):
        """Send heartbeat to server."""
        if not self.websocket:
            return

        heartbeat_message = {
            "type": "heartbeat",
            "data": {"client_time": datetime.utcnow().isoformat()},
        }

        try:
            await self.websocket.send(json.dumps(heartbeat_message))
            logger.debug("Heartbeat sent")
        except Exception as e:
            logger.error(f"Failed to send heartbeat: {e}")

    async def acknowledge_message(self, message_id: str):
        """Acknowledge a message that requires acknowledgment."""
        if not self.websocket:
            return

        ack_message = {"type": "acknowledge", "data": {"message_id": message_id}}

        try:
            await self.websocket.send(json.dumps(ack_message))
            logger.debug(f"Acknowledged message: {message_id}")
        except Exception as e:
            logger.error(f"Failed to acknowledge message {message_id}: {e}")

    async def listen(self):
        """Listen for messages from the WebSocket server."""
        if not self.websocket:
            logger.error("Not connected to WebSocket server")
            return

        try:
            async for raw_message in self.websocket:
                try:
                    message = json.loads(raw_message)
                    await self._process_message(message)
                except json.JSONDecodeError:
                    logger.error(f"Received invalid JSON: {raw_message}")
                except Exception as e:
                    logger.error(f"Error processing message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info("WebSocket connection closed by server")
        except Exception as e:
            logger.error(f"Error in message listening loop: {e}")

    async def run_with_reconnect(self):
        """Run the client with automatic reconnection."""
        reconnect_attempts = 0

        while reconnect_attempts < self.max_reconnect_attempts:
            try:
                # Connect
                if await self.connect():
                    reconnect_attempts = 0  # Reset on successful connection

                    # Authenticate
                    if await self.authenticate():
                        # Start listening for messages
                        await self.listen()
                    else:
                        logger.error("Authentication failed")

                await self.disconnect()

                if self.auto_reconnect:
                    reconnect_attempts += 1
                    logger.info(
                        f"Reconnecting in {self.reconnect_delay} seconds (attempt {reconnect_attempts})"
                    )
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break

            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                if self.auto_reconnect:
                    reconnect_attempts += 1
                    await asyncio.sleep(self.reconnect_delay)
                else:
                    break

        logger.info("WebSocket client stopped")

    # Message handlers

    async def _process_message(self, message: Dict):
        """Process incoming message from server."""
        message_type = message.get("message_type", "unknown")
        message_id = message.get("message_id")
        requires_ack = message.get("requires_ack", False)

        logger.debug(f"Received message: {message_type} (ID: {message_id})")

        # Handle message
        handler = self.message_handlers.get(message_type)
        if handler:
            await handler(message)
        else:
            logger.warning(f"No handler for message type: {message_type}")

        # Send acknowledgment if required
        if requires_ack and message_id:
            await self.acknowledge_message(message_id)

    async def _handle_connection(self, message: Dict):
        """Handle connection welcome message."""
        data = message.get("data", {})
        self.connection_id = data.get("connection_id")

        logger.info(f"Connection established with ID: {self.connection_id}")
        logger.info(f"Server time: {data.get('server_time')}")
        logger.info(
            f"Supported message types: {data.get('supported_message_types', [])}"
        )

    async def _handle_authentication(self, message: Dict):
        """Handle authentication response."""
        data = message.get("data", {})
        success = data.get("success", False)

        if success:
            self.authenticated = True
            logger.info("Authentication successful")
            logger.info(f"Capabilities: {data.get('capabilities', [])}")
            logger.info(f"Room filters: {data.get('room_filters', [])}")

            # Subscribe to desired endpoints after authentication
            await self._setup_subscriptions()
        else:
            logger.error(
                f"Authentication failed: {data.get('message', 'Unknown error')}"
            )

    async def _handle_subscription_status(self, message: Dict):
        """Handle subscription status updates."""
        data = message.get("data", {})
        action = data.get("action")
        success = data.get("success", False)
        endpoint = data.get("endpoint")
        room_id = data.get("room_id")

        if success:
            if action == "subscribe":
                self.subscriptions.add(endpoint)
                logger.info(
                    f"Successfully subscribed to {endpoint}"
                    + (f" (room: {room_id})" if room_id else "")
                )
            elif action == "unsubscribe":
                self.subscriptions.discard(endpoint)
                logger.info(f"Successfully unsubscribed from {endpoint}")
        else:
            logger.error(
                f"Subscription {action} failed for {endpoint}: {data.get('message')}"
            )

    async def _handle_prediction_update(self, message: Dict):
        """Handle real-time prediction updates."""
        data = message.get("data", {})
        room_id = data.get("room_id")
        room_name = data.get("room_name")
        predicted_time = data.get("predicted_time")
        transition_type = data.get("transition_type")
        confidence = data.get("confidence_score", 0)
        time_until = data.get("time_until_human")

        logger.info(f"🔮 Prediction Update - {room_name or room_id}:")
        logger.info(f"   Next {transition_type} at {predicted_time}")
        logger.info(f"   Time until: {time_until}")
        logger.info(f"   Confidence: {confidence:.2%}")

        # Handle alternatives if present
        alternatives = data.get("alternatives", [])
        if alternatives:
            logger.info(f"   Alternatives:")
            for i, alt in enumerate(alternatives[:3], 1):
                logger.info(
                    f"     {i}. {alt['predicted_time']} (confidence: {alt['confidence']:.2%})"
                )

    async def _handle_system_status_update(self, message: Dict):
        """Handle system status updates."""
        data = message.get("data", {})
        logger.info(f"📊 System Status Update:")

        # Log key system metrics
        for key, value in data.items():
            if isinstance(value, dict):
                logger.info(f"   {key}:")
                for sub_key, sub_value in value.items():
                    logger.info(f"     {sub_key}: {sub_value}")
            else:
                logger.info(f"   {key}: {value}")

    async def _handle_alert_notification(self, message: Dict):
        """Handle alert notifications."""
        data = message.get("data", {})
        room_id = message.get("room_id")

        logger.warning(
            f"🚨 Alert Notification" + (f" - Room: {room_id}" if room_id else "")
        )
        logger.warning(f"   Type: {data.get('alert_type', 'Unknown')}")
        logger.warning(f"   Severity: {data.get('severity', 'Unknown')}")
        logger.warning(f"   Message: {data.get('message', 'No message')}")

        # Handle specific alert types
        alert_type = data.get("alert_type")
        if alert_type == "accuracy_degradation":
            logger.warning(
                f"   Current accuracy: {data.get('current_accuracy', 'Unknown')}"
            )
            logger.warning(f"   Threshold: {data.get('threshold', 'Unknown')}")
        elif alert_type == "drift_detected":
            logger.warning(f"   Drift score: {data.get('drift_score', 'Unknown')}")
            logger.warning(
                f"   Retraining recommended: {data.get('retraining_recommended', False)}"
            )

    async def _handle_drift_notification(self, message: Dict):
        """Handle concept drift notifications."""
        data = message.get("data", {})
        room_id = message.get("room_id")

        logger.warning(f"📈 Drift Notification - Room: {room_id}")
        logger.warning(f"   Severity: {data.get('drift_severity', 'Unknown')}")
        logger.warning(f"   Score: {data.get('drift_score', 'Unknown')}")
        logger.warning(f"   Types: {data.get('drift_types', [])}")
        logger.warning(
            f"   Retraining recommended: {data.get('retraining_recommended', False)}"
        )

    async def _handle_heartbeat(self, message: Dict):
        """Handle heartbeat from server."""
        data = message.get("data", {})
        server_time = data.get("server_time")
        uptime = data.get("connection_uptime", 0)

        logger.debug(
            f"💓 Heartbeat - Server time: {server_time}, Uptime: {uptime:.1f}s"
        )

    async def _handle_error(self, message: Dict):
        """Handle error messages from server."""
        data = message.get("data", {})
        error_message = data.get("error", "Unknown error")
        error_code = data.get("error_code", "UNKNOWN")

        logger.error(f"❌ Server Error [{error_code}]: {error_message}")

    async def _handle_rate_limit_warning(self, message: Dict):
        """Handle rate limit warnings."""
        data = message.get("data", {})
        logger.warning(f"⚠️  Rate Limit Warning:")
        logger.warning(
            f"   Max messages per minute: {data.get('max_messages_per_minute')}"
        )
        logger.warning(f"   Current count: {data.get('current_count')}")
        logger.warning(f"   Reset time: {data.get('reset_time')}")

    async def _setup_subscriptions(self):
        """Set up initial subscriptions after authentication."""
        # Subscribe to prediction updates
        await self.subscribe_to_endpoint("/ws/predictions")

        # Subscribe to system status updates
        await self.subscribe_to_endpoint("/ws/system-status")

        # Subscribe to alerts
        await self.subscribe_to_endpoint("/ws/alerts")

        # Example: Subscribe to specific room updates
        # await self.subscribe_to_endpoint("/ws/room/living_room", room_id="living_room")


async def main():
    """Main function to run the WebSocket API client example."""
    parser = argparse.ArgumentParser(description="WebSocket API Client Example")
    parser.add_argument("--host", default="localhost", help="WebSocket server host")
    parser.add_argument("--port", type=int, default=8766, help="WebSocket server port")
    parser.add_argument("--api-key", required=True, help="API key for authentication")
    parser.add_argument(
        "--client-name", default="ExampleClient", help="Client name for identification"
    )
    parser.add_argument(
        "--no-reconnect", action="store_true", help="Disable automatic reconnection"
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")

    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Create client
    client = WebSocketAPIClient(
        host=args.host,
        port=args.port,
        api_key=args.api_key,
        client_name=args.client_name,
    )

    if args.no_reconnect:
        client.auto_reconnect = False

    logger.info("Starting WebSocket API client...")
    logger.info(f"Connecting to ws://{args.host}:{args.port}")
    logger.info("Press Ctrl+C to stop")

    try:
        await client.run_with_reconnect()
    except KeyboardInterrupt:
        logger.info("Received interrupt signal, shutting down...")
        client.auto_reconnect = False
        await client.disconnect()


if __name__ == "__main__":
    asyncio.run(main())
