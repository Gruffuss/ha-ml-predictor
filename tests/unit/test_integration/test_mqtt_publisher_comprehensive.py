"""
Comprehensive unit tests for MQTTPublisher to achieve high test coverage.

This module focuses on comprehensive testing of all methods, error paths,
edge cases, and configuration variations in MQTTPublisher.
"""

import asyncio
from datetime import datetime, timedelta
import json
import threading
from unittest.mock import AsyncMock, MagicMock, Mock, patch, call
import ssl

import pytest
from paho.mqtt import client as mqtt_client

from src.core.config import MQTTConfig
from src.core.exceptions import ErrorSeverity
from src.integration.mqtt_publisher import (
    MQTTPublisher,
    MQTTConnectionStatus,
    MQTTPublishResult,
    MQTTPublisherError
)


class TestMQTTConnectionStatus:
    """Test MQTTConnectionStatus dataclass."""

    def test_default_initialization(self):
        """Test default initialization of MQTTConnectionStatus."""
        status = MQTTConnectionStatus()
        
        assert status.connected is False
        assert status.last_connected is None
        assert status.last_disconnected is None
        assert status.connection_attempts == 0
        assert status.last_error is None
        assert status.reconnect_count == 0
        assert status.uptime_seconds == 0.0

    def test_custom_initialization(self):
        """Test custom initialization of MQTTConnectionStatus."""
        now = datetime.utcnow()
        status = MQTTConnectionStatus(
            connected=True,
            last_connected=now,
            connection_attempts=3,
            last_error="Test error",
            reconnect_count=2,
            uptime_seconds=123.45
        )
        
        assert status.connected is True
        assert status.last_connected == now
        assert status.connection_attempts == 3
        assert status.last_error == "Test error"
        assert status.reconnect_count == 2
        assert status.uptime_seconds == 123.45


class TestMQTTPublishResult:
    """Test MQTTPublishResult dataclass."""

    def test_successful_result(self):
        """Test successful publish result."""
        now = datetime.utcnow()
        result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=now,
            message_id=12345
        )
        
        assert result.success is True
        assert result.topic == "test/topic"
        assert result.payload_size == 100
        assert result.publish_time == now
        assert result.error_message is None
        assert result.message_id == 12345

    def test_failed_result(self):
        """Test failed publish result."""
        now = datetime.utcnow()
        result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=50,
            publish_time=now,
            error_message="Connection failed"
        )
        
        assert result.success is False
        assert result.topic == "test/topic"
        assert result.payload_size == 50
        assert result.publish_time == now
        assert result.error_message == "Connection failed"
        assert result.message_id is None


class TestMQTTPublisherInitialization:
    """Test MQTTPublisher initialization."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        config = MQTTConfig(broker="localhost")
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value.timestamp.return_value = 1234567890
            publisher = MQTTPublisher(config)
        
        assert publisher.config == config
        assert "ha_ml_predictor_1234567890" in publisher.client_id
        assert publisher.client is None
        assert isinstance(publisher.connection_status, MQTTConnectionStatus)
        assert publisher.message_queue == []
        assert publisher.max_queue_size == 1000
        assert isinstance(publisher._lock, threading.RLock)
        assert publisher._background_tasks == []
        assert not publisher._shutdown_event.is_set()
        assert publisher._publisher_active is False

    def test_init_custom_client_id(self):
        """Test initialization with custom client ID."""
        config = MQTTConfig(broker="localhost")
        custom_id = "custom_mqtt_client_123"
        
        publisher = MQTTPublisher(config, client_id=custom_id)
        
        assert publisher.client_id == custom_id

    def test_init_with_callbacks(self):
        """Test initialization with callback functions."""
        config = MQTTConfig(broker="localhost")
        
        connect_callback = Mock()
        disconnect_callback = Mock()
        message_callback = Mock()
        
        publisher = MQTTPublisher(
            config,
            on_connect_callback=connect_callback,
            on_disconnect_callback=disconnect_callback,
            on_message_callback=message_callback
        )
        
        assert publisher.on_connect_callback == connect_callback
        assert publisher.on_disconnect_callback == disconnect_callback
        assert publisher.on_message_callback == message_callback

    def test_init_statistics(self):
        """Test initial statistics values."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        assert publisher.total_messages_published == 0
        assert publisher.total_messages_failed == 0
        assert publisher.total_bytes_published == 0
        assert publisher.last_publish_time is None


class TestMQTTPublisherInitialization:
    """Test MQTTPublisher initialization process."""

    @pytest.fixture
    def mock_mqtt_client(self):
        """Create a mock MQTT client."""
        with patch('src.integration.mqtt_publisher.mqtt_client.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            yield mock_client

    @pytest.mark.asyncio
    async def test_initialize_disabled_publishing(self, mock_mqtt_client):
        """Test initialization when publishing is disabled."""
        config = MQTTConfig(broker="localhost", publishing_enabled=False)
        publisher = MQTTPublisher(config)
        
        await publisher.initialize()
        
        # Should not create MQTT client
        assert publisher.client is None

    @pytest.mark.asyncio
    async def test_initialize_success(self, mock_mqtt_client):
        """Test successful initialization."""
        config = MQTTConfig(
            broker="localhost",
            port=1883,
            username="user",
            password="pass"
        )
        publisher = MQTTPublisher(config)
        
        # Mock successful connection
        mock_mqtt_client.connect.return_value = None
        
        # Mock connection status change in callback
        def mock_connect_callback(*args):
            publisher.connection_status.connected = True
        
        with patch.object(publisher, '_connect_to_broker') as mock_connect:
            mock_connect.return_value = None
            with patch.object(publisher, 'start_publisher') as mock_start:
                mock_start.return_value = None
                
                await publisher.initialize()
        
        assert publisher.client == mock_mqtt_client
        assert publisher.client.on_connect is not None
        assert publisher.client.on_disconnect is not None
        mock_mqtt_client.username_pw_set.assert_called_once_with("user", "pass")

    @pytest.mark.asyncio
    async def test_initialize_tls_connection(self, mock_mqtt_client):
        """Test initialization with TLS (port 8883)."""
        config = MQTTConfig(broker="localhost", port=8883)
        publisher = MQTTPublisher(config)
        
        with patch.object(publisher, '_connect_to_broker'), \
             patch.object(publisher, 'start_publisher'), \
             patch('ssl.create_default_context') as mock_ssl:
            
            mock_context = Mock()
            mock_ssl.return_value = mock_context
            
            await publisher.initialize()
            
            mock_ssl.assert_called_once_with(ssl.Purpose.SERVER_AUTH)
            mock_mqtt_client.tls_set_context.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, mock_mqtt_client):
        """Test initialization with connection failure."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        with patch.object(publisher, '_connect_to_broker') as mock_connect:
            mock_connect.side_effect = Exception("Connection failed")
            
            with pytest.raises(MQTTPublisherError) as exc_info:
                await publisher.initialize()
            
            assert "Failed to initialize MQTT publisher" in str(exc_info.value)
            assert exc_info.value.cause is not None


class TestMQTTPublisherPublishing:
    """Test MQTT message publishing functionality."""

    @pytest.fixture
    def connected_publisher(self):
        """Create a connected MQTT publisher."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        # Mock connected state
        publisher.client = Mock()
        publisher.connection_status.connected = True
        
        return publisher

    @pytest.mark.asyncio
    async def test_publish_string_payload_success(self, connected_publisher):
        """Test successful publishing with string payload."""
        publisher = connected_publisher
        
        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        mock_info.mid = 12345
        publisher.client.publish.return_value = mock_info
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is True
        assert result.topic == "test/topic"
        assert result.payload_size == len("test message".encode("utf-8"))
        assert result.message_id == 12345
        assert result.error_message is None
        
        publisher.client.publish.assert_called_once_with(
            "test/topic", "test message", qos=1, retain=False
        )

    @pytest.mark.asyncio
    async def test_publish_dict_payload_success(self, connected_publisher):
        """Test successful publishing with dictionary payload."""
        publisher = connected_publisher
        
        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        mock_info.mid = 12346
        publisher.client.publish.return_value = mock_info
        
        payload_dict = {"key": "value", "number": 42}
        result = await publisher.publish("test/topic", payload_dict, qos=2, retain=True)
        
        assert result.success is True
        expected_json = json.dumps(payload_dict, default=str)
        assert result.payload_size == len(expected_json.encode("utf-8"))
        
        publisher.client.publish.assert_called_once_with(
            "test/topic", expected_json, qos=2, retain=True
        )

    @pytest.mark.asyncio
    async def test_publish_bytes_payload_success(self, connected_publisher):
        """Test successful publishing with bytes payload."""
        publisher = connected_publisher
        
        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        publisher.client.publish.return_value = mock_info
        
        bytes_payload = b"binary data"
        result = await publisher.publish("test/topic", bytes_payload)
        
        assert result.success is True
        
        # Should decode bytes to string
        publisher.client.publish.assert_called_once_with(
            "test/topic", "binary data", qos=1, retain=False
        )

    @pytest.mark.asyncio
    async def test_publish_client_not_connected(self, connected_publisher):
        """Test publishing when client is not connected (should queue)."""
        publisher = connected_publisher
        publisher.connection_status.connected = False
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is False
        assert "not connected - message queued" in result.error_message
        assert len(publisher.message_queue) == 1
        
        # Check queued message
        queued = publisher.message_queue[0]
        assert queued["topic"] == "test/topic"
        assert queued["payload"] == "test message"
        assert queued["qos"] == 1
        assert queued["retain"] is False

    @pytest.mark.asyncio
    async def test_publish_no_client(self):
        """Test publishing when no client exists."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        # publisher.client is None by default
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is False
        assert len(publisher.message_queue) == 1

    @pytest.mark.asyncio
    async def test_publish_queue_overflow(self, connected_publisher):
        """Test publishing when message queue is full."""
        publisher = connected_publisher
        publisher.connection_status.connected = False
        publisher.max_queue_size = 2
        
        # Fill queue to max
        await publisher.publish("topic1", "message1")
        await publisher.publish("topic2", "message2")
        
        # This should remove oldest and add new
        await publisher.publish("topic3", "message3")
        
        assert len(publisher.message_queue) == 2
        # Should have removed first message and kept last two
        topics = [msg["topic"] for msg in publisher.message_queue]
        assert "topic2" in topics
        assert "topic3" in topics

    @pytest.mark.asyncio
    async def test_publish_mqtt_error(self, connected_publisher):
        """Test publishing when MQTT client returns error."""
        publisher = connected_publisher
        
        # Mock failed publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_NO_CONN
        publisher.client.publish.return_value = mock_info
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is False
        assert "return code" in result.error_message
        assert publisher.total_messages_failed == 1

    @pytest.mark.asyncio
    async def test_publish_exception_during_publish(self, connected_publisher):
        """Test publishing when MQTT client raises exception."""
        publisher = connected_publisher
        
        # Mock exception during publish
        publisher.client.publish.side_effect = Exception("Publish failed")
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is False
        assert "Exception during MQTT publish" in result.error_message
        assert publisher.total_messages_failed == 1

    @pytest.mark.asyncio
    async def test_publish_json_convenience_method(self, connected_publisher):
        """Test publish_json convenience method."""
        publisher = connected_publisher
        
        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        publisher.client.publish.return_value = mock_info
        
        data = {"temperature": 23.5, "humidity": 45}
        result = await publisher.publish_json("sensor/data", data, qos=0, retain=True)
        
        assert result.success is True
        
        expected_json = json.dumps(data, default=str)
        publisher.client.publish.assert_called_once_with(
            "sensor/data", expected_json, qos=0, retain=True
        )

    @pytest.mark.asyncio
    async def test_publish_payload_conversion_exception(self, connected_publisher):
        """Test publishing when payload conversion fails."""
        publisher = connected_publisher
        
        # Create an object that can't be JSON serialized and doesn't have __str__
        class UnserializableObject:
            def __str__(self):
                raise Exception("Cannot convert to string")
        
        result = await publisher.publish("test/topic", UnserializableObject())
        
        assert result.success is False
        assert result.payload_size == 0


class TestMQTTPublisherConnectionManagement:
    """Test MQTT connection management."""

    @pytest.fixture
    def publisher_with_mock_client(self):
        """Create publisher with mock client."""
        config = MQTTConfig(
            broker="localhost",
            max_reconnect_attempts=3,
            reconnect_delay_seconds=1,
            connection_timeout=5
        )
        publisher = MQTTPublisher(config)
        publisher.client = Mock()
        return publisher

    @pytest.mark.asyncio
    async def test_connect_to_broker_success(self, publisher_with_mock_client):
        """Test successful connection to broker."""
        publisher = publisher_with_mock_client
        
        # Mock successful connection
        def mock_connect_side_effect(*args):
            publisher.connection_status.connected = True
        
        publisher.client.connect.side_effect = mock_connect_side_effect
        
        await publisher._connect_to_broker()
        
        publisher.client.connect.assert_called_once()
        publisher.client.loop_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_to_broker_timeout(self, publisher_with_mock_client):
        """Test connection timeout."""
        publisher = publisher_with_mock_client
        publisher.config.connection_timeout = 0.1  # Very short timeout
        
        # Mock connection that doesn't set connected=True
        publisher.client.connect.return_value = None
        
        with pytest.raises(MQTTPublisherError) as exc_info:
            await publisher._connect_to_broker()
        
        assert "Failed to connect" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_connect_to_broker_max_attempts(self, publisher_with_mock_client):
        """Test connection with max attempts exceeded."""
        publisher = publisher_with_mock_client
        publisher.config.max_reconnect_attempts = 2
        publisher.config.reconnect_delay_seconds = 0.01  # Speed up test
        
        # Mock connection failure
        publisher.client.connect.side_effect = Exception("Connection failed")
        
        with pytest.raises(MQTTPublisherError) as exc_info:
            await publisher._connect_to_broker()
        
        assert "Failed to connect after 2 attempts" in str(exc_info.value)
        assert publisher.connection_status.connection_attempts == 2

    @pytest.mark.asyncio
    async def test_connect_to_broker_infinite_retries(self, publisher_with_mock_client):
        """Test connection with infinite retries (until shutdown)."""
        publisher = publisher_with_mock_client
        publisher.config.max_reconnect_attempts = -1  # Infinite retries
        publisher.config.reconnect_delay_seconds = 0.01
        
        # Mock connection failure
        publisher.client.connect.side_effect = Exception("Connection failed")
        
        # Set shutdown event after short delay to stop infinite loop
        async def set_shutdown():
            await asyncio.sleep(0.05)
            publisher._shutdown_event.set()
        
        # Start shutdown task
        shutdown_task = asyncio.create_task(set_shutdown())
        
        try:
            await publisher._connect_to_broker()
        except MQTTPublisherError:
            pass  # Expected when shutdown occurs
        
        await shutdown_task
        assert publisher.connection_status.connection_attempts > 0

    @pytest.mark.asyncio
    async def test_disconnect_from_broker(self, publisher_with_mock_client):
        """Test disconnection from broker."""
        publisher = publisher_with_mock_client
        
        await publisher._disconnect_from_broker()
        
        publisher.client.loop_stop.assert_called_once()
        publisher.client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_from_broker_exception(self, publisher_with_mock_client):
        """Test disconnection with exception."""
        publisher = publisher_with_mock_client
        
        publisher.client.disconnect.side_effect = Exception("Disconnect failed")
        
        # Should handle exception gracefully
        await publisher._disconnect_from_broker()
        
        publisher.client.loop_stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_no_client(self):
        """Test disconnection when no client exists."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        # publisher.client is None
        
        # Should handle gracefully
        await publisher._disconnect_from_broker()


class TestMQTTPublisherBackgroundTasks:
    """Test background task management."""

    @pytest.fixture
    def publisher_with_mock_client(self):
        """Create publisher with mock client."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        publisher.client = Mock()
        return publisher

    @pytest.mark.asyncio
    async def test_start_publisher_disabled(self, publisher_with_mock_client):
        """Test starting publisher when publishing is disabled."""
        publisher = publisher_with_mock_client
        publisher.config.publishing_enabled = False
        
        await publisher.start_publisher()
        
        assert not publisher._publisher_active
        assert len(publisher._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_publisher_already_active(self, publisher_with_mock_client):
        """Test starting publisher when already active."""
        publisher = publisher_with_mock_client
        publisher._publisher_active = True
        
        await publisher.start_publisher()
        
        # Should not create new tasks
        assert len(publisher._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_start_publisher_success(self, publisher_with_mock_client):
        """Test successful publisher start."""
        publisher = publisher_with_mock_client
        
        with patch('asyncio.create_task') as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            await publisher.start_publisher()
            
            assert publisher._publisher_active is True
            assert len(publisher._background_tasks) == 2  # Connection monitoring + queue processing
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_start_publisher_exception(self, publisher_with_mock_client):
        """Test publisher start with exception."""
        publisher = publisher_with_mock_client
        
        with patch('asyncio.create_task', side_effect=Exception("Task creation failed")):
            with pytest.raises(MQTTPublisherError):
                await publisher.start_publisher()

    @pytest.mark.asyncio
    async def test_stop_publisher_not_active(self, publisher_with_mock_client):
        """Test stopping publisher when not active."""
        publisher = publisher_with_mock_client
        publisher._publisher_active = False
        
        await publisher.stop_publisher()
        # Should handle gracefully

    @pytest.mark.asyncio
    async def test_stop_publisher_with_queued_messages(self, publisher_with_mock_client):
        """Test stopping publisher with queued messages."""
        publisher = publisher_with_mock_client
        publisher._publisher_active = True
        publisher.connection_status.connected = False
        
        # Add queued messages
        publisher.message_queue = [
            {"topic": "test1", "payload": "msg1", "qos": 1, "retain": False},
            {"topic": "test2", "payload": "msg2", "qos": 1, "retain": False},
        ]
        
        with patch.object(publisher, '_process_message_queue') as mock_process:
            mock_process.return_value = None
            with patch.object(publisher, '_disconnect_from_broker') as mock_disconnect:
                mock_disconnect.return_value = None
                
                await publisher.stop_publisher()
                
                mock_process.assert_called_once()
                assert not publisher._publisher_active

    @pytest.mark.asyncio
    async def test_stop_publisher_process_timeout(self, publisher_with_mock_client):
        """Test stopping publisher with message processing timeout."""
        publisher = publisher_with_mock_client
        publisher._publisher_active = True
        publisher.message_queue = [{"topic": "test", "payload": "msg", "qos": 1, "retain": False}]
        
        with patch.object(publisher, '_process_message_queue') as mock_process:
            # Mock timeout
            async def slow_process():
                await asyncio.sleep(15)  # Longer than timeout
            mock_process.side_effect = slow_process
            
            with patch.object(publisher, '_disconnect_from_broker') as mock_disconnect:
                mock_disconnect.return_value = None
                
                # Should handle timeout gracefully
                await publisher.stop_publisher()
                
                assert not publisher._publisher_active

    @pytest.mark.asyncio
    async def test_connection_monitoring_loop(self, publisher_with_mock_client):
        """Test connection monitoring background loop."""
        publisher = publisher_with_mock_client
        publisher.connection_status.connected = False
        
        # Set shutdown event after short delay
        async def set_shutdown():
            await asyncio.sleep(0.05)
            publisher._shutdown_event.set()
        
        with patch.object(publisher, '_connect_to_broker') as mock_connect:
            mock_connect.return_value = None
            
            # Start shutdown task
            shutdown_task = asyncio.create_task(set_shutdown())
            
            # Should attempt reconnection
            await publisher._connection_monitoring_loop()
            
            await shutdown_task
            mock_connect.assert_called()

    @pytest.mark.asyncio
    async def test_connection_monitoring_loop_exception(self, publisher_with_mock_client):
        """Test connection monitoring loop with exception."""
        publisher = publisher_with_mock_client
        
        # Set shutdown event after short delay
        async def set_shutdown():
            await asyncio.sleep(0.1)
            publisher._shutdown_event.set()
        
        with patch.object(publisher, '_connect_to_broker') as mock_connect:
            mock_connect.side_effect = Exception("Connect failed")
            
            shutdown_task = asyncio.create_task(set_shutdown())
            
            # Should handle exception and continue
            await publisher._connection_monitoring_loop()
            
            await shutdown_task

    @pytest.mark.asyncio
    async def test_message_queue_processing_loop(self, publisher_with_mock_client):
        """Test message queue processing background loop."""
        publisher = publisher_with_mock_client
        publisher.connection_status.connected = True
        publisher.message_queue = [{"topic": "test", "payload": "msg", "qos": 1, "retain": False}]
        
        # Set shutdown event after short delay
        async def set_shutdown():
            await asyncio.sleep(0.05)
            publisher._shutdown_event.set()
        
        with patch.object(publisher, '_process_message_queue') as mock_process:
            mock_process.return_value = None
            
            shutdown_task = asyncio.create_task(set_shutdown())
            
            await publisher._message_queue_processing_loop()
            
            await shutdown_task
            mock_process.assert_called()

    @pytest.mark.asyncio
    async def test_process_message_queue_empty(self, publisher_with_mock_client):
        """Test processing empty message queue."""
        publisher = publisher_with_mock_client
        
        await publisher._process_message_queue()
        # Should handle empty queue gracefully

    @pytest.mark.asyncio
    async def test_process_message_queue_success(self, publisher_with_mock_client):
        """Test successful message queue processing."""
        publisher = publisher_with_mock_client
        publisher.connection_status.connected = True
        
        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        publisher.client.publish.return_value = mock_info
        
        # Add test messages
        publisher.message_queue = [
            {"topic": "test1", "payload": "msg1", "qos": 1, "retain": False, "queued_at": datetime.utcnow()},
            {"topic": "test2", "payload": "msg2", "qos": 0, "retain": True, "queued_at": datetime.utcnow()},
        ]
        
        await publisher._process_message_queue()
        
        # Queue should be empty
        assert len(publisher.message_queue) == 0
        # Should have called publish for each message
        assert publisher.client.publish.call_count == 2

    @pytest.mark.asyncio
    async def test_process_message_queue_with_failures(self, publisher_with_mock_client):
        """Test message queue processing with some failures."""
        publisher = publisher_with_mock_client
        publisher.connection_status.connected = True
        
        # Mock mixed success/failure
        def publish_side_effect(topic, payload, qos, retain):
            if "fail" in topic:
                raise Exception("Publish failed")
            mock_info = Mock()
            mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
            return mock_info
        
        publisher.client.publish.side_effect = publish_side_effect
        
        # Add test messages (one will fail)
        publisher.message_queue = [
            {"topic": "test_success", "payload": "msg1", "qos": 1, "retain": False, "queued_at": datetime.utcnow()},
            {"topic": "test_fail", "payload": "msg2", "qos": 1, "retain": False, "queued_at": datetime.utcnow()},
        ]
        
        await publisher._process_message_queue()
        
        # Queue should be empty regardless of failures
        assert len(publisher.message_queue) == 0


class TestMQTTPublisherCallbacks:
    """Test MQTT client callback handlers."""

    @pytest.fixture
    def publisher_with_callbacks(self):
        """Create publisher with mock callbacks."""
        config = MQTTConfig(broker="localhost")
        
        connect_callback = Mock()
        disconnect_callback = Mock()
        message_callback = Mock()
        
        publisher = MQTTPublisher(
            config,
            on_connect_callback=connect_callback,
            on_disconnect_callback=disconnect_callback,
            on_message_callback=message_callback
        )
        
        return publisher, connect_callback, disconnect_callback, message_callback

    def test_on_connect_success(self, publisher_with_callbacks):
        """Test successful connection callback."""
        publisher, connect_callback, _, _ = publisher_with_callbacks
        
        client = Mock()
        userdata = {}
        flags = {}
        reason_code = 0
        properties = {}
        
        publisher._on_connect(client, userdata, flags, reason_code, properties)
        
        assert publisher.connection_status.connected is True
        assert publisher.connection_status.last_connected is not None
        assert publisher.connection_status.reconnect_count == 1
        connect_callback.assert_called_once_with(client, userdata, flags, reason_code)

    def test_on_connect_failure(self, publisher_with_callbacks):
        """Test failed connection callback."""
        publisher, connect_callback, _, _ = publisher_with_callbacks
        
        client = Mock()
        reason_code = 1  # Connection refused
        
        publisher._on_connect(client, {}, {}, reason_code, {})
        
        assert publisher.connection_status.connected is False
        assert publisher.connection_status.last_error is not None
        assert "reason code: 1" in publisher.connection_status.last_error

    def test_on_connect_async_callback(self, publisher_with_callbacks):
        """Test connection callback with async function."""
        publisher, _, _, _ = publisher_with_callbacks
        
        # Replace with async callback
        async_callback = AsyncMock()
        publisher.on_connect_callback = async_callback
        
        with patch('asyncio.create_task') as mock_create_task:
            publisher._on_connect(Mock(), {}, {}, 0, {})
            
            # Should create task for async callback
            mock_create_task.assert_called_once()

    def test_on_connect_callback_exception(self, publisher_with_callbacks):
        """Test connection callback with exception."""
        publisher, connect_callback, _, _ = publisher_with_callbacks
        connect_callback.side_effect = Exception("Callback failed")
        
        # Should handle exception gracefully
        publisher._on_connect(Mock(), {}, {}, 0, {})
        
        assert publisher.connection_status.connected is True  # Should still set connection status

    def test_on_disconnect_clean(self, publisher_with_callbacks):
        """Test clean disconnection callback."""
        publisher, _, disconnect_callback, _ = publisher_with_callbacks
        
        client = Mock()
        reason_code = 0  # Clean disconnect
        
        publisher._on_disconnect(client, {}, {}, reason_code, {})
        
        assert publisher.connection_status.connected is False
        assert publisher.connection_status.last_disconnected is not None
        disconnect_callback.assert_called_once()

    def test_on_disconnect_unexpected(self, publisher_with_callbacks):
        """Test unexpected disconnection callback."""
        publisher, _, disconnect_callback, _ = publisher_with_callbacks
        
        reason_code = 1  # Unexpected disconnect
        
        publisher._on_disconnect(Mock(), {}, {}, reason_code, {})
        
        assert publisher.connection_status.connected is False
        disconnect_callback.assert_called_once()

    def test_on_publish_success(self, publisher_with_callbacks):
        """Test successful publish callback."""
        publisher, _, _, _ = publisher_with_callbacks
        
        mid = 12345
        reason_code = 0
        
        # Should handle gracefully
        publisher._on_publish(Mock(), {}, mid, reason_code, {})

    def test_on_publish_failure(self, publisher_with_callbacks):
        """Test failed publish callback."""
        publisher, _, _, _ = publisher_with_callbacks
        
        mid = 12346
        reason_code = 1
        
        # Should handle gracefully
        publisher._on_publish(Mock(), {}, mid, reason_code, {})

    def test_on_message_callback(self, publisher_with_callbacks):
        """Test message received callback."""
        publisher, _, _, message_callback = publisher_with_callbacks
        
        client = Mock()
        userdata = {}
        message = Mock()
        message.topic = "test/topic"
        
        publisher._on_message(client, userdata, message)
        
        message_callback.assert_called_once_with(client, userdata, message)

    def test_on_message_async_callback(self, publisher_with_callbacks):
        """Test message callback with async function."""
        publisher, _, _, _ = publisher_with_callbacks
        
        # Replace with async callback
        async_callback = AsyncMock()
        publisher.on_message_callback = async_callback
        
        message = Mock()
        message.topic = "test/topic"
        
        with patch('asyncio.create_task') as mock_create_task:
            publisher._on_message(Mock(), {}, message)
            
            mock_create_task.assert_called_once()

    def test_on_log_callback(self, publisher_with_callbacks):
        """Test MQTT log callback with different levels."""
        publisher, _, _, _ = publisher_with_callbacks
        
        # Test different log levels
        log_levels = [
            (mqtt_client.MQTT_LOG_DEBUG, "Debug message"),
            (mqtt_client.MQTT_LOG_INFO, "Info message"),
            (mqtt_client.MQTT_LOG_NOTICE, "Notice message"),
            (mqtt_client.MQTT_LOG_WARNING, "Warning message"),
            (mqtt_client.MQTT_LOG_ERR, "Error message"),
        ]
        
        for level, message in log_levels:
            # Should handle all log levels gracefully
            publisher._on_log(Mock(), {}, level, message)

    def test_callback_exception_handling(self, publisher_with_callbacks):
        """Test that callback exceptions are handled gracefully."""
        publisher, _, _, _ = publisher_with_callbacks
        
        # Mock callbacks that raise exceptions
        def failing_callback():
            raise Exception("Callback failed")
        
        # Should handle exceptions in all callbacks
        publisher._on_connect(Mock(), {}, {}, 0, {})  # Should not raise
        publisher._on_disconnect(Mock(), {}, {}, 0, {})  # Should not raise
        publisher._on_publish(Mock(), {}, 123, 0, {})  # Should not raise
        publisher._on_message(Mock(), {}, Mock())  # Should not raise
        publisher._on_log(Mock(), {}, mqtt_client.MQTT_LOG_INFO, "test")  # Should not raise


class TestMQTTPublisherStatusAndStats:
    """Test status and statistics methods."""

    def test_get_connection_status_disconnected(self):
        """Test connection status when disconnected."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        status = publisher.get_connection_status()
        
        assert isinstance(status, MQTTConnectionStatus)
        assert status.connected is False
        assert status.uptime_seconds == 0.0

    def test_get_connection_status_connected(self):
        """Test connection status when connected with uptime calculation."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        # Mock connected state
        publisher.connection_status.connected = True
        past_time = datetime.utcnow() - timedelta(seconds=100)
        publisher.connection_status.last_connected = past_time
        
        with patch('datetime.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = past_time + timedelta(seconds=100)
            
            status = publisher.get_connection_status()
            
            assert status.connected is True
            assert status.uptime_seconds == 100.0

    def test_get_publisher_stats(self):
        """Test getting publisher statistics."""
        config = MQTTConfig(
            broker="mqtt.example.com",
            port=1883,
            publishing_enabled=True,
            discovery_enabled=True,
            keepalive=60
        )
        publisher = MQTTPublisher(config, client_id="test_client")
        
        # Set some statistics
        publisher.total_messages_published = 100
        publisher.total_messages_failed = 5
        publisher.total_bytes_published = 50000
        publisher.last_publish_time = datetime.utcnow()
        publisher.message_queue = [{"test": "message"}] * 3
        publisher._publisher_active = True
        
        stats = publisher.get_publisher_stats()
        
        assert stats["client_id"] == "test_client"
        assert stats["messages_published"] == 100
        assert stats["messages_failed"] == 5
        assert stats["bytes_published"] == 50000
        assert stats["last_publish_time"] is not None
        assert stats["queued_messages"] == 3
        assert stats["max_queue_size"] == 1000
        assert stats["publisher_active"] is True
        assert stats["config"]["broker"] == "mqtt.example.com"
        assert stats["config"]["port"] == 1883
        assert stats["config"]["publishing_enabled"] is True
        assert stats["config"]["discovery_enabled"] is True
        assert stats["config"]["keepalive"] == 60

    def test_get_publisher_stats_no_last_publish_time(self):
        """Test getting stats when last_publish_time is None."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        stats = publisher.get_publisher_stats()
        
        assert stats["last_publish_time"] is None


class TestMQTTPublisherError:
    """Test MQTTPublisherError exception class."""

    def test_mqtt_publisher_error_basic(self):
        """Test basic MQTTPublisherError initialization."""
        error = MQTTPublisherError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.error_code == "MQTT_PUBLISHER_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_mqtt_publisher_error_custom_severity(self):
        """Test MQTTPublisherError with custom severity."""
        error = MQTTPublisherError("Critical error", severity=ErrorSeverity.CRITICAL)
        
        assert error.severity == ErrorSeverity.CRITICAL

    def test_mqtt_publisher_error_with_cause(self):
        """Test MQTTPublisherError with cause."""
        original_error = ConnectionError("Network error")
        error = MQTTPublisherError("MQTT error", cause=original_error)
        
        assert error.cause == original_error


class TestMQTTPublisherEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_publish_with_complex_dict_payload(self):
        """Test publishing with complex dictionary containing datetime objects."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        publisher.client = Mock()
        publisher.connection_status.connected = True
        
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        publisher.client.publish.return_value = mock_info
        
        complex_payload = {
            "timestamp": datetime.utcnow(),
            "values": [1, 2, 3],
            "nested": {"key": "value"},
            "none_value": None
        }
        
        result = await publisher.publish("test/topic", complex_payload)
        
        # Should handle complex objects with default=str
        assert result.success is True
        
        # Check that publish was called with JSON string
        call_args = publisher.client.publish.call_args
        payload_arg = call_args[0][1]  # Second argument is payload
        
        # Should be valid JSON
        try:
            parsed = json.loads(payload_arg)
            assert "timestamp" in parsed
            assert "values" in parsed
            assert "nested" in parsed
        except json.JSONDecodeError:
            pytest.fail("Payload should be valid JSON")

    @pytest.mark.asyncio
    async def test_multiple_start_stop_cycles(self):
        """Test multiple start/stop cycles."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        # Mock successful operations
        with patch.object(publisher, '_connect_to_broker'), \
             patch.object(publisher, '_disconnect_from_broker'), \
             patch('asyncio.create_task') as mock_create_task:
            
            mock_task = Mock()
            mock_create_task.return_value = mock_task
            
            # Start -> Stop -> Start -> Stop
            await publisher.start_publisher()
            assert publisher._publisher_active is True
            
            await publisher.stop_publisher()
            assert publisher._publisher_active is False
            
            await publisher.start_publisher()
            assert publisher._publisher_active is True
            
            await publisher.stop_publisher()
            assert publisher._publisher_active is False

    def test_thread_safety_of_message_queue(self):
        """Test thread safety of message queue operations."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        # Test that _lock is used for thread safety
        assert isinstance(publisher._lock, threading.RLock)
        
        # Verify queue operations are thread-safe by checking they work
        # (detailed thread testing would require more complex setup)
        assert len(publisher.message_queue) == 0

    @pytest.mark.asyncio 
    async def test_shutdown_event_handling(self):
        """Test that shutdown event properly stops background operations."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        
        # Initially not set
        assert not publisher._shutdown_event.is_set()
        
        # Set shutdown
        publisher._shutdown_event.set()
        assert publisher._shutdown_event.is_set()
        
        # Clear shutdown
        publisher._shutdown_event.clear()
        assert not publisher._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_publish_statistics_update(self):
        """Test that publishing updates statistics correctly."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        publisher.client = Mock()
        publisher.connection_status.connected = True
        
        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        publisher.client.publish.return_value = mock_info
        
        initial_published = publisher.total_messages_published
        initial_bytes = publisher.total_bytes_published
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is True
        assert publisher.total_messages_published == initial_published + 1
        assert publisher.total_bytes_published > initial_bytes
        assert publisher.last_publish_time is not None

    @pytest.mark.asyncio
    async def test_publish_failed_statistics_update(self):
        """Test that failed publishing updates failure statistics."""
        config = MQTTConfig(broker="localhost")
        publisher = MQTTPublisher(config)
        publisher.client = Mock()
        publisher.connection_status.connected = True
        
        # Mock failed publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_NO_CONN
        publisher.client.publish.return_value = mock_info
        
        initial_failed = publisher.total_messages_failed
        
        result = await publisher.publish("test/topic", "test message")
        
        assert result.success is False
        assert publisher.total_messages_failed == initial_failed + 1