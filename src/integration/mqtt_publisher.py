"""
MQTT Publisher Infrastructure for Home Assistant Integration.

This module provides the core MQTT publishing infrastructure for publishing
occupancy predictions and system status to Home Assistant via MQTT.

Features:
- Automatic connection management with reconnection
- Home Assistant MQTT discovery support
- Prediction publishing with proper topic structure
- System status monitoring and publishing
- Integration with TrackingManager for automatic operation
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
import threading
import ssl

from paho.mqtt import client as mqtt_client
import paho.mqtt.enums as mqtt_enums

from ..core.config import MQTTConfig
from ..core.exceptions import OccupancyPredictionError, ErrorSeverity
from ..models.base.predictor import PredictionResult


logger = logging.getLogger(__name__)


@dataclass
class MQTTConnectionStatus:
    """MQTT connection status information."""
    connected: bool = False
    last_connected: Optional[datetime] = None
    last_disconnected: Optional[datetime] = None
    connection_attempts: int = 0
    last_error: Optional[str] = None
    reconnect_count: int = 0
    uptime_seconds: float = 0.0


@dataclass
class MQTTPublishResult:
    """Result of MQTT publish operation."""
    success: bool
    topic: str
    payload_size: int
    publish_time: datetime
    error_message: Optional[str] = None
    message_id: Optional[int] = None


class MQTTPublisher:
    """
    Core MQTT publisher with automatic connection management.
    
    Provides reliable MQTT connectivity with automatic reconnection,
    message queuing during disconnections, and comprehensive error handling.
    """
    
    def __init__(
        self,
        config: MQTTConfig,
        client_id: Optional[str] = None,
        on_connect_callback: Optional[Callable] = None,
        on_disconnect_callback: Optional[Callable] = None,
        on_message_callback: Optional[Callable] = None
    ):
        """
        Initialize MQTT publisher.
        
        Args:
            config: MQTT configuration
            client_id: Optional MQTT client ID (auto-generated if None)
            on_connect_callback: Callback for connection events
            on_disconnect_callback: Callback for disconnection events
            on_message_callback: Callback for received messages
        """
        self.config = config
        self.client_id = client_id or f"ha_ml_predictor_{int(datetime.utcnow().timestamp())}"
        
        # Callbacks
        self.on_connect_callback = on_connect_callback
        self.on_disconnect_callback = on_disconnect_callback
        self.on_message_callback = on_message_callback
        
        # MQTT client
        self.client: Optional[mqtt_client.Client] = None
        self.connection_status = MQTTConnectionStatus()
        
        # Message queuing for offline publishing
        self.message_queue: List[Dict[str, Any]] = []
        self.max_queue_size = 1000
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._publisher_active = False
        
        # Statistics
        self.total_messages_published = 0
        self.total_messages_failed = 0
        self.total_bytes_published = 0
        self.last_publish_time: Optional[datetime] = None
        
        logger.info(f"Initialized MQTTPublisher with client_id: {self.client_id}")
    
    async def initialize(self) -> None:
        """Initialize MQTT client and establish connection."""
        try:
            if not self.config.publishing_enabled:
                logger.info("MQTT publishing is disabled in configuration")
                return
            
            # Create MQTT client
            self.client = mqtt_client.Client(
                callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
                client_id=self.client_id,
                protocol=mqtt_client.MQTTv311,
                clean_session=True
            )
            
            # Set up callbacks
            self.client.on_connect = self._on_connect
            self.client.on_disconnect = self._on_disconnect
            self.client.on_publish = self._on_publish
            self.client.on_message = self._on_message
            self.client.on_log = self._on_log
            
            # Configure authentication if provided
            if self.config.username and self.config.password:
                self.client.username_pw_set(self.config.username, self.config.password)
                logger.info("MQTT authentication configured")
            
            # Configure TLS if needed (for secure connections)
            if self.config.port == 8883:
                context = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
                self.client.tls_set_context(context)
                logger.info("MQTT TLS encryption enabled")
            
            # Set keepalive and connection timeout
            self.client.keepalive = self.config.keepalive
            
            # Connect to broker
            await self._connect_to_broker()
            
            # Start background tasks
            await self.start_publisher()
            
            logger.info("MQTT publisher initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize MQTT publisher: {e}")
            raise MQTTPublisherError("Failed to initialize MQTT publisher", cause=e)
    
    async def start_publisher(self) -> None:
        """Start background publisher tasks."""
        try:
            if self._publisher_active:
                logger.warning("MQTT publisher already active")
                return
            
            if not self.config.publishing_enabled:
                logger.info("MQTT publishing is disabled, not starting publisher")
                return
            
            # Start connection monitoring task
            connection_task = asyncio.create_task(self._connection_monitoring_loop())
            self._background_tasks.append(connection_task)
            
            # Start message queue processing task
            queue_task = asyncio.create_task(self._message_queue_processing_loop())
            self._background_tasks.append(queue_task)
            
            self._publisher_active = True
            logger.info("Started MQTT publisher background tasks")
            
        except Exception as e:
            logger.error(f"Failed to start MQTT publisher: {e}")
            raise MQTTPublisherError("Failed to start MQTT publisher", cause=e)
    
    async def stop_publisher(self) -> None:
        """Stop MQTT publisher gracefully."""
        try:
            if not self._publisher_active:
                return
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Process remaining queued messages (with timeout)
            if self.message_queue:
                logger.info(f"Processing {len(self.message_queue)} remaining queued messages")
                try:
                    await asyncio.wait_for(self._process_message_queue(), timeout=10.0)
                except asyncio.TimeoutError:
                    logger.warning("Timeout processing remaining messages during shutdown")
            
            # Disconnect MQTT client
            if self.client and self.connection_status.connected:
                await self._disconnect_from_broker()
            
            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            self._background_tasks.clear()
            self._publisher_active = False
            
            logger.info("Stopped MQTT publisher")
            
        except Exception as e:
            logger.error(f"Error stopping MQTT publisher: {e}")
    
    async def publish(
        self,
        topic: str,
        payload: Union[str, bytes, Dict[str, Any]],
        qos: int = 1,
        retain: bool = False
    ) -> MQTTPublishResult:
        """
        Publish a message to MQTT broker.
        
        Args:
            topic: MQTT topic to publish to
            payload: Message payload (string, bytes, or dict for JSON)
            qos: Quality of Service level (0, 1, or 2)
            retain: Whether to retain the message
            
        Returns:
            MQTTPublishResult with publish details
        """
        try:
            # Convert payload to string if necessary
            if isinstance(payload, dict):
                payload_str = json.dumps(payload, default=str)
            elif isinstance(payload, bytes):
                payload_str = payload.decode('utf-8')
            else:
                payload_str = str(payload)
            
            payload_size = len(payload_str.encode('utf-8'))
            
            # Check if client is connected
            if not self.client or not self.connection_status.connected:
                # Queue message for later delivery
                queued_message = {
                    'topic': topic,
                    'payload': payload_str,
                    'qos': qos,
                    'retain': retain,
                    'queued_at': datetime.utcnow()
                }
                
                with self._lock:
                    if len(self.message_queue) < self.max_queue_size:
                        self.message_queue.append(queued_message)
                        logger.debug(f"Queued message for topic {topic} (queue size: {len(self.message_queue)})")
                    else:
                        # Remove oldest message to make room
                        self.message_queue.pop(0)
                        self.message_queue.append(queued_message)
                        logger.warning(f"Message queue full, removed oldest message for topic {topic}")
                
                return MQTTPublishResult(
                    success=False,
                    topic=topic,
                    payload_size=payload_size,
                    publish_time=datetime.utcnow(),
                    error_message="Client not connected - message queued"
                )
            
            # Publish message
            try:
                info = self.client.publish(topic, payload_str, qos=qos, retain=retain)
                
                if info.rc == mqtt_client.MQTT_ERR_SUCCESS:
                    self.total_messages_published += 1
                    self.total_bytes_published += payload_size
                    self.last_publish_time = datetime.utcnow()
                    
                    logger.debug(f"Published message to {topic} (size: {payload_size} bytes)")
                    
                    return MQTTPublishResult(
                        success=True,
                        topic=topic,
                        payload_size=payload_size,
                        publish_time=datetime.utcnow(),
                        message_id=info.mid
                    )
                else:
                    error_msg = f"MQTT publish failed with return code: {info.rc}"
                    self.total_messages_failed += 1
                    logger.error(error_msg)
                    
                    return MQTTPublishResult(
                        success=False,
                        topic=topic,
                        payload_size=payload_size,
                        publish_time=datetime.utcnow(),
                        error_message=error_msg
                    )
                    
            except Exception as e:
                error_msg = f"Exception during MQTT publish: {e}"
                self.total_messages_failed += 1
                logger.error(error_msg)
                
                return MQTTPublishResult(
                    success=False,
                    topic=topic,
                    payload_size=payload_size,
                    publish_time=datetime.utcnow(),
                    error_message=error_msg
                )
                
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            return MQTTPublishResult(
                success=False,
                topic=topic,
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e)
            )
    
    async def publish_json(
        self,
        topic: str,
        data: Dict[str, Any],
        qos: int = 1,
        retain: bool = False
    ) -> MQTTPublishResult:
        """
        Publish JSON data to MQTT broker.
        
        Args:
            topic: MQTT topic to publish to
            data: Dictionary to convert to JSON
            qos: Quality of Service level
            retain: Whether to retain the message
            
        Returns:
            MQTTPublishResult with publish details
        """
        return await self.publish(topic, data, qos=qos, retain=retain)
    
    def get_connection_status(self) -> MQTTConnectionStatus:
        """Get current MQTT connection status."""
        if self.connection_status.connected and self.connection_status.last_connected:
            self.connection_status.uptime_seconds = (
                datetime.utcnow() - self.connection_status.last_connected
            ).total_seconds()
        
        return self.connection_status
    
    def get_publisher_stats(self) -> Dict[str, Any]:
        """Get MQTT publisher statistics."""
        return {
            'client_id': self.client_id,
            'connection_status': asdict(self.get_connection_status()),
            'messages_published': self.total_messages_published,
            'messages_failed': self.total_messages_failed,
            'bytes_published': self.total_bytes_published,
            'last_publish_time': self.last_publish_time.isoformat() if self.last_publish_time else None,
            'queued_messages': len(self.message_queue),
            'max_queue_size': self.max_queue_size,
            'publisher_active': self._publisher_active,
            'config': {
                'broker': self.config.broker,
                'port': self.config.port,
                'publishing_enabled': self.config.publishing_enabled,
                'discovery_enabled': self.config.discovery_enabled,
                'keepalive': self.config.keepalive
            }
        }
    
    # Private methods
    
    async def _connect_to_broker(self) -> None:
        """Connect to MQTT broker with retry logic."""
        max_attempts = self.config.max_reconnect_attempts
        attempt = 0
        
        while (max_attempts == -1 or attempt < max_attempts) and not self._shutdown_event.is_set():
            try:
                logger.info(f"Connecting to MQTT broker {self.config.broker}:{self.config.port} (attempt {attempt + 1})")
                
                self.client.connect(
                    self.config.broker,
                    self.config.port,
                    self.config.keepalive
                )
                
                # Start network loop
                self.client.loop_start()
                
                # Wait for connection confirmation (with timeout)
                wait_start = datetime.utcnow()
                while not self.connection_status.connected and not self._shutdown_event.is_set():
                    if (datetime.utcnow() - wait_start).total_seconds() > self.config.connection_timeout:
                        raise TimeoutError("Connection timeout")
                    await asyncio.sleep(0.1)
                
                if self.connection_status.connected:
                    logger.info("Successfully connected to MQTT broker")
                    return
                    
            except Exception as e:
                attempt += 1
                self.connection_status.connection_attempts = attempt
                self.connection_status.last_error = str(e)
                
                logger.error(f"Failed to connect to MQTT broker (attempt {attempt}): {e}")
                
                if max_attempts != -1 and attempt >= max_attempts:
                    raise MQTTPublisherError(f"Failed to connect after {attempt} attempts")
                
                # Wait before retry
                await asyncio.sleep(self.config.reconnect_delay_seconds)
    
    async def _disconnect_from_broker(self) -> None:
        """Disconnect from MQTT broker."""
        try:
            if self.client:
                self.client.loop_stop()
                self.client.disconnect()
                logger.info("Disconnected from MQTT broker")
        except Exception as e:
            logger.error(f"Error disconnecting from MQTT broker: {e}")
    
    async def _connection_monitoring_loop(self) -> None:
        """Background loop for monitoring MQTT connection."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check connection status
                    if not self.connection_status.connected and self.client:
                        logger.warning("MQTT connection lost, attempting to reconnect")
                        await self._connect_to_broker()
                    
                    # Wait for next monitoring cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=10.0  # Check every 10 seconds
                    )
                    
                except asyncio.TimeoutError:
                    # Expected timeout for monitoring interval
                    continue
                except Exception as e:
                    logger.error(f"Error in connection monitoring loop: {e}")
                    await asyncio.sleep(5.0)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Connection monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Connection monitoring loop failed: {e}")
    
    async def _message_queue_processing_loop(self) -> None:
        """Background loop for processing queued messages."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    if self.connection_status.connected and self.message_queue:
                        await self._process_message_queue()
                    
                    # Wait for next processing cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=5.0  # Check every 5 seconds
                    )
                    
                except asyncio.TimeoutError:
                    # Expected timeout for processing interval
                    continue
                except Exception as e:
                    logger.error(f"Error in message queue processing loop: {e}")
                    await asyncio.sleep(2.0)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Message queue processing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Message queue processing loop failed: {e}")
    
    async def _process_message_queue(self) -> None:
        """Process queued messages."""
        try:
            with self._lock:
                messages_to_process = self.message_queue.copy()
                self.message_queue.clear()
            
            if not messages_to_process:
                return
            
            logger.info(f"Processing {len(messages_to_process)} queued messages")
            
            processed = 0
            failed = 0
            
            for message in messages_to_process:
                try:
                    result = await self.publish(
                        topic=message['topic'],
                        payload=message['payload'],
                        qos=message['qos'],
                        retain=message['retain']
                    )
                    
                    if result.success:
                        processed += 1
                    else:
                        failed += 1
                        logger.warning(f"Failed to publish queued message to {message['topic']}: {result.error_message}")
                        
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing queued message: {e}")
            
            logger.info(f"Processed queued messages: {processed} successful, {failed} failed")
            
        except Exception as e:
            logger.error(f"Error processing message queue: {e}")
    
    # MQTT client callbacks
    
    def _on_connect(self, client, userdata, flags, reason_code, properties):
        """Callback for MQTT connection events."""
        try:
            if reason_code == 0:
                self.connection_status.connected = True
                self.connection_status.last_connected = datetime.utcnow()
                self.connection_status.reconnect_count += 1
                logger.info("MQTT client connected successfully")
                
                if self.on_connect_callback:
                    try:
                        if asyncio.iscoroutinefunction(self.on_connect_callback):
                            asyncio.create_task(self.on_connect_callback(client, userdata, flags, reason_code))
                        else:
                            self.on_connect_callback(client, userdata, flags, reason_code)
                    except Exception as e:
                        logger.error(f"Error in connect callback: {e}")
            else:
                self.connection_status.connected = False
                self.connection_status.last_error = f"Connection failed with reason code: {reason_code}"
                logger.error(f"MQTT connection failed with reason code: {reason_code}")
                
        except Exception as e:
            logger.error(f"Error in MQTT connect callback: {e}")
    
    def _on_disconnect(self, client, userdata, flags, reason_code, properties):
        """Callback for MQTT disconnection events."""
        try:
            self.connection_status.connected = False
            self.connection_status.last_disconnected = datetime.utcnow()
            
            if reason_code == 0:
                logger.info("MQTT client disconnected cleanly")
            else:
                logger.warning(f"MQTT client disconnected unexpectedly with reason code: {reason_code}")
            
            if self.on_disconnect_callback:
                try:
                    if asyncio.iscoroutinefunction(self.on_disconnect_callback):
                        asyncio.create_task(self.on_disconnect_callback(client, userdata, flags, reason_code))
                    else:
                        self.on_disconnect_callback(client, userdata, flags, reason_code)
                except Exception as e:
                    logger.error(f"Error in disconnect callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error in MQTT disconnect callback: {e}")
    
    def _on_publish(self, client, userdata, mid, reason_code, properties):
        """Callback for MQTT publish events."""
        try:
            if reason_code == 0:
                logger.debug(f"MQTT message published successfully (mid: {mid})")
            else:
                logger.warning(f"MQTT publish failed (mid: {mid}, reason_code: {reason_code})")
                
        except Exception as e:
            logger.error(f"Error in MQTT publish callback: {e}")
    
    def _on_message(self, client, userdata, message):
        """Callback for received MQTT messages."""
        try:
            logger.debug(f"Received MQTT message on topic: {message.topic}")
            
            if self.on_message_callback:
                try:
                    if asyncio.iscoroutinefunction(self.on_message_callback):
                        asyncio.create_task(self.on_message_callback(client, userdata, message))
                    else:
                        self.on_message_callback(client, userdata, message)
                except Exception as e:
                    logger.error(f"Error in message callback: {e}")
                    
        except Exception as e:
            logger.error(f"Error in MQTT message callback: {e}")
    
    def _on_log(self, client, userdata, level, buf):
        """Callback for MQTT client logging."""
        try:
            # Map MQTT log levels to Python logging levels
            if level == mqtt_client.MQTT_LOG_DEBUG:
                logger.debug(f"MQTT: {buf}")
            elif level == mqtt_client.MQTT_LOG_INFO:
                logger.debug(f"MQTT: {buf}")  # Use debug for MQTT info to reduce noise
            elif level == mqtt_client.MQTT_LOG_NOTICE:
                logger.info(f"MQTT: {buf}")
            elif level == mqtt_client.MQTT_LOG_WARNING:
                logger.warning(f"MQTT: {buf}")
            elif level == mqtt_client.MQTT_LOG_ERR:
                logger.error(f"MQTT: {buf}")
                
        except Exception as e:
            logger.error(f"Error in MQTT log callback: {e}")


class MQTTPublisherError(OccupancyPredictionError):
    """Raised when MQTT publisher operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="MQTT_PUBLISHER_ERROR",
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            **kwargs
        )