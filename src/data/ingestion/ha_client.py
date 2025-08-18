"""
Home Assistant API client for real-time event streaming and historical data fetching.

This module provides the main interface for connecting to Home Assistant via both
WebSocket (for real-time events) and REST API (for historical data bulk import).
Includes automatic reconnection, rate limiting, and event processing.
"""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
import json
import logging
from typing import Any, AsyncGenerator, Callable, Dict, List, Optional
from urllib.parse import urljoin

import aiohttp
import websockets
from websockets.exceptions import (
    ConnectionClosed,
    InvalidStatusCode,
    InvalidURI,
)

from ...core.config import SystemConfig, get_config
from ...core.constants import INVALID_STATES, MIN_EVENT_SEPARATION, SensorState
from ...core.exceptions import (
    EntityNotFoundError,
    HomeAssistantAPIError,
    HomeAssistantAuthenticationError,
    HomeAssistantConnectionError,
    RateLimitExceededError,
    WebSocketError,
)
from ..storage.models import SensorEvent

logger = logging.getLogger(__name__)


@dataclass
class HAEvent:
    """Represents a Home Assistant event."""

    entity_id: str
    state: str
    previous_state: Optional[str]
    timestamp: datetime
    attributes: Dict[str, Any]
    event_type: str = "state_changed"

    def is_valid(self) -> bool:
        """Check if the event contains valid data."""
        return (
            self.state not in INVALID_STATES
            and self.entity_id
            and self.timestamp is not None
        )


class RateLimiter:
    """Simple rate limiter for API requests."""

    def __init__(self, max_requests: int = 300, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.window = timedelta(seconds=window_seconds)
        self.requests = []
        self._lock = asyncio.Lock()

    async def acquire(self):
        """Wait if necessary to respect rate limits."""
        async with self._lock:
            now = datetime.now(UTC)
            cutoff_time = now - self.window
            # Remove old requests outside the window
            self.requests = [
                req_time for req_time in self.requests if req_time >= cutoff_time
            ]

            if len(self.requests) >= self.max_requests:
                # Calculate wait time
                oldest_request = min(self.requests)
                wait_time_seconds = (
                    self.window.total_seconds() - (now - oldest_request).total_seconds()
                )
                if wait_time_seconds > 0:
                    logger.warning(
                        f"Rate limit reached, waiting {wait_time_seconds:.2f} seconds"
                    )
                    # Actually wait instead of raising exception for internal rate limiting
                    await asyncio.sleep(wait_time_seconds)
                    # After waiting, proceed to add the request

            self.requests.append(now)


class HomeAssistantClient:
    """
    Home Assistant API client with WebSocket and REST support.

    Provides:
    - Real-time event streaming via WebSocket
    - Historical data fetching via REST API
    - Automatic reconnection and error handling
    - Rate limiting and connection management
    - Event processing and filtering
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        self.ha_config = self.config.home_assistant
        self.session: Optional[aiohttp.ClientSession] = None
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.rate_limiter = RateLimiter()

        # Connection state
        self._connected = False
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 10
        self._base_reconnect_delay = 5  # seconds

        # Event handling
        self._event_handlers: List[Callable[[HAEvent], None]] = []
        self._subscribed_entities: set = set()
        self._last_event_times: Dict[str, datetime] = {}

        # WebSocket message tracking
        self._ws_message_id = 1
        self._pending_responses: Dict[int, asyncio.Future] = {}

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self):
        """Establish connection to Home Assistant."""
        if self._connected:
            return

        try:
            # Create HTTP session
            timeout = aiohttp.ClientTimeout(total=self.ha_config.api_timeout)
            self.session = aiohttp.ClientSession(
                timeout=timeout,
                headers={
                    "Authorization": f"Bearer {self.ha_config.token}",
                    "Content-Type": "application/json",
                },
            )

            # Test authentication
            await self._test_authentication()

            # Connect WebSocket
            await self._connect_websocket()

            self._connected = True
            self._reconnect_attempts = 0
            logger.info("Successfully connected to Home Assistant")

        except Exception as e:
            await self._cleanup_connections()
            raise HomeAssistantConnectionError(self.ha_config.url, cause=e)

    async def disconnect(self):
        """Disconnect from Home Assistant."""
        self._connected = False
        await self._cleanup_connections()
        logger.info("Disconnected from Home Assistant")

    async def _cleanup_connections(self):
        """Clean up all connections."""
        if self.websocket:
            try:
                await self.websocket.close()
            except Exception:
                pass
            self.websocket = None

        if self.session:
            try:
                await self.session.close()
            except Exception:
                pass
            self.session = None

    async def _test_authentication(self):
        """Test if authentication is working using proper URL construction."""
        try:
            # Use urljoin for proper URL construction
            api_url = urljoin(self.ha_config.url, "/api/")
            async with self.session.get(api_url) as response:
                if response.status == 401:
                    raise HomeAssistantAuthenticationError(
                        self.ha_config.url, len(self.ha_config.token)
                    )
                elif response.status != 200:
                    raise HomeAssistantAPIError(
                        "/api/", response.status, await response.text(), "GET"
                    )
        except aiohttp.ClientError as e:
            raise HomeAssistantConnectionError(self.ha_config.url, cause=e)

    async def _connect_websocket(self):
        """Connect to Home Assistant WebSocket API."""
        ws_url = self.ha_config.url.replace("http://", "ws://").replace(
            "https://", "wss://"
        )
        ws_url = f"{ws_url}/api/websocket"

        try:
            self.websocket = await websockets.connect(
                ws_url,
                timeout=self.ha_config.websocket_timeout,
                ping_interval=20,
                ping_timeout=10,
            )

            # Handle authentication
            await self._authenticate_websocket()

            # Start message handling task
            asyncio.create_task(self._handle_websocket_messages())

        except (ConnectionClosed, InvalidStatusCode, InvalidURI) as e:
            raise WebSocketError(str(e), ws_url)

    async def _authenticate_websocket(self):
        """Authenticate WebSocket connection."""
        # Wait for auth_required message
        auth_msg = await self.websocket.recv()
        auth_data = json.loads(auth_msg)

        if auth_data.get("type") != "auth_required":
            raise WebSocketError(
                f"Expected auth_required, got {auth_data.get('type')}", ""
            )

        # Send authentication
        auth_response = {"type": "auth", "access_token": self.ha_config.token}
        await self.websocket.send(json.dumps(auth_response))

        # Wait for auth result
        result_msg = await self.websocket.recv()
        result_data = json.loads(result_msg)

        if result_data.get("type") != "auth_ok":
            raise HomeAssistantAuthenticationError(
                self.ha_config.url, len(self.ha_config.token)
            )

    async def _handle_websocket_messages(self):
        """Handle incoming WebSocket messages."""
        try:
            async for message in self.websocket:
                try:
                    data = json.loads(message)
                    await self._process_websocket_message(data)
                except json.JSONDecodeError:
                    logger.warning(f"Received invalid JSON: {message}")
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
        except ConnectionClosed:
            logger.warning("WebSocket connection closed")
            if self._connected:
                asyncio.create_task(self._reconnect())
        except Exception as e:
            logger.error(f"WebSocket error: {e}")
            if self._connected:
                asyncio.create_task(self._reconnect())

    async def _process_websocket_message(self, data: Dict[str, Any]):
        """Process a WebSocket message."""
        msg_type = data.get("type")

        if msg_type == "event":
            await self._handle_event(data)
        elif msg_type == "result":
            # Handle command responses
            msg_id = data.get("id")
            if msg_id in self._pending_responses:
                future = self._pending_responses.pop(msg_id)
                if not future.done():
                    future.set_result(data)
        elif msg_type == "pong":
            # Handle ping/pong
            pass
        else:
            logger.debug(f"Unhandled message type: {msg_type}")

    async def _handle_event(self, event_data: Dict[str, Any]):
        """Handle state change events."""
        try:
            event_info = event_data.get("event", {})
            if event_info.get("event_type") != "state_changed":
                return

            data = event_info.get("data", {})
            entity_id = data.get("entity_id")
            new_state = data.get("new_state", {})
            old_state = data.get("old_state", {})

            if not entity_id or not new_state:
                return

            # Filter subscribed entities
            if self._subscribed_entities and entity_id not in self._subscribed_entities:
                return

            # Extract and validate state using SensorState constants
            current_state = new_state.get("state", "")
            previous_state = old_state.get("state") if old_state else None

            # Validate states using SensorState enum
            validated_current_state = self._validate_and_normalize_state(current_state)
            validated_previous_state = (
                self._validate_and_normalize_state(previous_state)
                if previous_state
                else None
            )

            # Handle various timestamp formats from Home Assistant
            timestamp_str = new_state.get("last_changed", "")
            timestamp_clean = timestamp_str.replace("Z", "+00:00")
            # Handle double timezone suffixes that can occur in tests
            if timestamp_clean.count("+00:00") > 1:
                timestamp_clean = timestamp_clean.replace("+00:00+00:00", "+00:00")

            # Create HAEvent with validated states
            ha_event = HAEvent(
                entity_id=entity_id,
                state=validated_current_state,
                previous_state=validated_previous_state,
                timestamp=datetime.fromisoformat(timestamp_clean),
                attributes=new_state.get("attributes", {}),
            )

            # Apply deduplication
            if self._should_process_event(ha_event):
                await self._notify_event_handlers(ha_event)
                self._last_event_times[entity_id] = ha_event.timestamp

        except Exception as e:
            logger.error(f"Error handling event: {e}")

    def _validate_and_normalize_state(self, state: str) -> str:
        """
        Validate and normalize sensor state using SensorState constants.

        Args:
            state: Raw state from Home Assistant

        Returns:
            Validated and normalized state string
        """
        if not state:
            return ""

        # Normalize state to lowercase for consistent comparison
        normalized_state = state.lower().strip()

        # Map common Home Assistant states to SensorState values
        state_mapping = {
            "on": SensorState.ON.value,
            "off": SensorState.OFF.value,
            "open": SensorState.OPEN.value,
            "closed": SensorState.CLOSED.value,
            "unavailable": SensorState.UNAVAILABLE.value,
            "unknown": SensorState.UNKNOWN.value,
        }

        # Try exact match first
        if normalized_state in state_mapping:
            return state_mapping[normalized_state]

        # Try partial matches for common variations
        if "on" in normalized_state or "active" in normalized_state:
            return SensorState.ON.value
        elif "off" in normalized_state or "inactive" in normalized_state:
            return SensorState.OFF.value
        elif "open" in normalized_state:
            return SensorState.OPEN.value
        elif "closed" in normalized_state or "close" in normalized_state:
            return SensorState.CLOSED.value
        elif "detect" in normalized_state or "motion" in normalized_state:
            return SensorState.ON.value  # Map detected/motion to ON
        elif "clear" in normalized_state or "no" in normalized_state:
            return SensorState.OFF.value  # Map clear/no to OFF
        elif "unavailable" in normalized_state:
            return SensorState.UNAVAILABLE.value

        # If no match found, log warning and return original normalized state
        logger.debug(f"Unknown sensor state '{state}', using as-is")
        return normalized_state

    def _should_process_event(self, event: HAEvent) -> bool:
        """Check if event should be processed (deduplication)."""
        if not event.is_valid():
            return False

        # Check minimum time separation
        last_time = self._last_event_times.get(event.entity_id)
        if last_time:
            time_diff = (event.timestamp - last_time).total_seconds()
            if time_diff < MIN_EVENT_SEPARATION:
                return False

        return True

    async def _notify_event_handlers(self, event: HAEvent):
        """Notify all registered event handlers."""
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"Error in event handler: {e}")

    async def _reconnect(self):
        """Attempt to reconnect to Home Assistant."""
        if self._reconnect_attempts >= self._max_reconnect_attempts:
            logger.error("Max reconnection attempts reached")
            return

        self._reconnect_attempts += 1
        delay = self._base_reconnect_delay * (
            2 ** (self._reconnect_attempts - 1)
        )  # Exponential backoff
        delay = min(delay, 300)  # Cap at 5 minutes

        logger.info(
            f"Reconnecting to Home Assistant (attempt {self._reconnect_attempts}/{self._max_reconnect_attempts}) in {delay}s"
        )
        await asyncio.sleep(delay)

        try:
            await self._cleanup_connections()
            await self.connect()

            # Re-subscribe to entities
            if self._subscribed_entities:
                await self.subscribe_to_events(list(self._subscribed_entities))

        except Exception as e:
            logger.error(f"Reconnection failed: {e}")
            asyncio.create_task(self._reconnect())

    async def subscribe_to_events(self, entity_ids: List[str]):
        """
        Subscribe to state change events for specific entities.

        Args:
            entity_ids: List of entity IDs to subscribe to
        """
        if not self._connected or not self.websocket:
            raise HomeAssistantConnectionError(self.ha_config.url)

        self._subscribed_entities.update(entity_ids)

        # Subscribe to state change events
        command = {
            "id": self._ws_message_id,
            "type": "subscribe_events",
            "event_type": "state_changed",
        }

        future = asyncio.Future()
        self._pending_responses[self._ws_message_id] = future
        self._ws_message_id += 1

        await self.websocket.send(json.dumps(command))

        try:
            result = await asyncio.wait_for(future, timeout=10)
            if not result.get("success"):
                raise HomeAssistantAPIError("subscribe_events", 0, str(result))
        except asyncio.TimeoutError:
            raise HomeAssistantAPIError(
                "subscribe_events", 0, "Timeout waiting for response"
            )

        logger.info(f"Subscribed to events for {len(entity_ids)} entities")

    def add_event_handler(self, handler: Callable[[HAEvent], None]):
        """Add an event handler for state changes."""
        self._event_handlers.append(handler)

    def remove_event_handler(self, handler: Callable[[HAEvent], None]):
        """Remove an event handler."""
        if handler in self._event_handlers:
            self._event_handlers.remove(handler)

    async def get_entity_state(self, entity_id: str) -> Optional[Dict[str, Any]]:
        """Get current state of an entity with rate limiting and proper URL construction."""
        try:
            await self.rate_limiter.acquire()
        except RateLimitExceededError:
            # If rate limited, wait and retry once
            logger.warning("Rate limited, retrying entity state request")
            await asyncio.sleep(1)
            await self.rate_limiter.acquire()

        try:
            # Use urljoin for proper URL construction
            url = urljoin(self.ha_config.url, f"/api/states/{entity_id}")
            async with self.session.get(url) as response:
                if response.status == 404:
                    return None
                elif response.status == 429:
                    # Handle rate limiting from HA server
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitExceededError(
                        service="home_assistant_api",
                        limit=self.rate_limiter.max_requests,
                        window_seconds=int(self.rate_limiter.window.total_seconds()),
                        reset_time=retry_after,
                    )
                elif response.status != 200:
                    raise HomeAssistantAPIError(
                        f"/api/states/{entity_id}",
                        response.status,
                        await response.text(),
                        "GET",
                    )

                # Validate response data
                data = await response.json()
                if data and "state" in data:
                    # Normalize the state in the response
                    data["state"] = self._validate_and_normalize_state(data["state"])

                return data
        except RateLimitExceededError:
            raise  # Re-raise rate limit errors
        except aiohttp.ClientError as e:
            raise HomeAssistantConnectionError(self.ha_config.url, cause=e)

    async def get_entity_history(
        self,
        entity_id: str,
        start_time: datetime,
        end_time: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get historical state changes for an entity.

        Args:
            entity_id: Entity ID to get history for
            start_time: Start of time range
            end_time: End of time range (defaults to now)

        Returns:
            List of state change records
        """
        await self.rate_limiter.acquire()

        if end_time is None:
            end_time = datetime.now(UTC)

        params = {
            "filter_entity_id": entity_id,
            "end_time": end_time.isoformat() + "Z",
        }

        try:
            # Use urljoin for proper URL construction
            history_path = f"/api/history/period/{start_time.isoformat()}Z"
            url = urljoin(self.ha_config.url, history_path)

            async with self.session.get(url, params=params) as response:
                if response.status == 404:
                    raise EntityNotFoundError(entity_id)
                elif response.status == 429:
                    # Handle rate limiting from HA server
                    retry_after = int(response.headers.get("Retry-After", 60))
                    raise RateLimitExceededError(
                        service="home_assistant_history_api",
                        limit=self.rate_limiter.max_requests,
                        window_seconds=int(self.rate_limiter.window.total_seconds()),
                        reset_time=retry_after,
                    )
                elif response.status != 200:
                    raise HomeAssistantAPIError(
                        history_path,
                        response.status,
                        await response.text(),
                        "GET",
                    )

                data = await response.json()

                # Validate and normalize states in historical data
                if data and isinstance(data, list) and len(data) > 0:
                    historical_records = data[0]
                    for record in historical_records:
                        if "state" in record:
                            record["state"] = self._validate_and_normalize_state(
                                record["state"]
                            )
                    return historical_records

                return []

        except RateLimitExceededError:
            raise  # Re-raise rate limit errors
        except aiohttp.ClientError as e:
            raise HomeAssistantConnectionError(self.ha_config.url, cause=e)

    async def get_bulk_history(
        self,
        entity_ids: List[str],
        start_time: datetime,
        end_time: Optional[datetime] = None,
        batch_size: int = 50,
    ) -> AsyncGenerator[List[Dict[str, Any]], None]:
        """
        Get historical data for multiple entities in batches.

        Args:
            entity_ids: List of entity IDs
            start_time: Start of time range
            end_time: End of time range
            batch_size: Number of entities to fetch per batch

        Yields:
            Batches of historical records
        """
        if end_time is None:
            end_time = datetime.now(UTC)

        # Process entities in batches to avoid overwhelming the API
        for i in range(0, len(entity_ids), batch_size):
            batch_entities = entity_ids[i : i + batch_size]
            batch_results = []

            for entity_id in batch_entities:
                try:
                    history = await self.get_entity_history(
                        entity_id, start_time, end_time
                    )
                    batch_results.extend(history)

                    # Small delay between requests to be nice to HA
                    await asyncio.sleep(0.1)

                except RateLimitExceededError as e:
                    # Handle rate limiting gracefully in bulk operations
                    logger.warning(
                        f"Rate limited during bulk history fetch, waiting {e.reset_time}s"
                    )
                    await asyncio.sleep(e.reset_time or 60)
                    # Retry the request once
                    try:
                        history = await self.get_entity_history(
                            entity_id, start_time, end_time
                        )
                        batch_results.extend(history)
                    except Exception as retry_e:
                        logger.error(f"Retry failed for {entity_id}: {retry_e}")
                        continue
                except EntityNotFoundError:
                    logger.warning(f"Entity {entity_id} not found, skipping")
                    continue
                except Exception as e:
                    logger.error(f"Error fetching history for {entity_id}: {e}")
                    continue

            if batch_results:
                yield batch_results

    async def validate_entities(self, entity_ids: List[str]) -> Dict[str, bool]:
        """
        Validate that entities exist in Home Assistant.

        Args:
            entity_ids: List of entity IDs to validate

        Returns:
            Dictionary mapping entity_id to existence boolean
        """
        results = {}

        for entity_id in entity_ids:
            try:
                state = await self.get_entity_state(entity_id)
                results[entity_id] = state is not None
            except Exception as e:
                logger.warning(f"Error validating entity {entity_id}: {e}")
                results[entity_id] = False

        return results

    def convert_ha_event_to_sensor_event(
        self, ha_event: HAEvent, room_id: str, sensor_type: str
    ) -> SensorEvent:
        """Convert HA event to internal SensorEvent model."""
        return SensorEvent(
            room_id=room_id,
            sensor_id=ha_event.entity_id,
            sensor_type=sensor_type,
            state=ha_event.state,
            previous_state=ha_event.previous_state,
            timestamp=ha_event.timestamp,
            attributes=ha_event.attributes,
            is_human_triggered=True,  # Will be updated by event processor
            created_at=datetime.now(UTC),
        )

    def convert_history_to_sensor_events(
        self,
        history_data: List[Dict[str, Any]],
        room_id: str,
        sensor_type: str,
    ) -> List[SensorEvent]:
        """Convert Home Assistant history data to SensorEvent models."""
        events = []
        previous_state = None

        for record in history_data:
            timestamp_str = record.get("last_changed", record.get("last_updated", ""))
            if timestamp_str:
                try:
                    # Handle various timestamp formats from Home Assistant
                    timestamp_clean = timestamp_str.replace("Z", "+00:00")
                    # Handle double timezone suffixes that can occur in tests
                    if timestamp_clean.count("+00:00") > 1:
                        timestamp_clean = timestamp_clean.replace(
                            "+00:00+00:00", "+00:00"
                        )
                    timestamp = datetime.fromisoformat(timestamp_clean)
                except ValueError:
                    logger.warning(f"Invalid timestamp format: {timestamp_str}")
                    continue
            else:
                continue

            event = SensorEvent(
                room_id=room_id,
                sensor_id=record.get("entity_id", ""),
                sensor_type=sensor_type,
                state=record.get("state", ""),
                previous_state=previous_state,
                timestamp=timestamp,
                attributes=record.get("attributes", {}),
                is_human_triggered=True,  # Will be updated by event processor
                created_at=datetime.now(UTC),
            )

            events.append(event)
            previous_state = event.state

        return events

    @property
    def is_connected(self) -> bool:
        """Check if client is connected to Home Assistant."""
        return self._connected and self.websocket and not self.websocket.closed
