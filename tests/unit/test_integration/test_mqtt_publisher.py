"""
Comprehensive unit tests for MQTT Publisher.

This module provides extensive coverage for MQTTPublisher functionality including
connection management, message publishing, error handling, and background tasks.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import ssl
import threading
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from paho.mqtt import client as mqtt_client
import pytest

from src.core.config import MQTTConfig
from src.core.exceptions import ErrorSeverity
from src.integration.mqtt_publisher import (
    MQTTConnectionStatus,
    MQTTPublisher,
    MQTTPublisherError,
    MQTTPublishResult,
)


class TestMQTTConnectionStatus:
    """Test MQTTConnectionStatus dataclass."""

    def test_connection_status_initialization(self):
        """Test default initialization of connection status."""
        status = MQTTConnectionStatus()

        assert status.connected is False
        assert status.last_connected is None
        assert status.last_disconnected is None
        assert status.connection_attempts == 0
        assert status.last_error is None
        assert status.reconnect_count == 0
        assert status.uptime_seconds == 0.0

    def test_connection_status_with_values(self):
        """Test connection status with specific values."""
        now = datetime.now(timezone.utc)
        status = MQTTConnectionStatus(
            connected=True,
            last_connected=now,
            connection_attempts=5,
            last_error="Test error",
            reconnect_count=3,
            uptime_seconds=120.5,
        )

        assert status.connected is True
        assert status.last_connected == now
        assert status.connection_attempts == 5
        assert status.last_error == "Test error"
        assert status.reconnect_count == 3
        assert status.uptime_seconds == 120.5


class TestMQTTPublishResult:
    """Test MQTTPublishResult dataclass."""

    def test_publish_result_success(self):
        """Test successful publish result."""
        now = datetime.now(timezone.utc)
        result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=now,
            message_id=12345,
        )

        assert result.success is True
        assert result.topic == "test/topic"
        assert result.payload_size == 100
        assert result.publish_time == now
        assert result.message_id == 12345
        assert result.error_message is None

    def test_publish_result_failure(self):
        """Test failed publish result."""
        now = datetime.now(timezone.utc)
        result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=50,
            publish_time=now,
            error_message="Connection failed",
        )

        assert result.success is False
        assert result.topic == "test/topic"
        assert result.payload_size == 50
        assert result.error_message == "Connection failed"
        assert result.message_id is None


class TestMQTTPublisher:
    """Test MQTTPublisher class."""

    @pytest.fixture
    def mqtt_config(self):
        """Create MQTT configuration for testing."""
        return MQTTConfig(
            broker="test-broker",
            port=1883,
            username="testuser",
            password="testpass",
            topic_prefix="test/occupancy",
            publishing_enabled=True,
            discovery_enabled=True,
            keepalive=60,
            connection_timeout=30,
            reconnect_delay_seconds=5,
            max_reconnect_attempts=3,
        )

    @pytest.fixture
    def disabled_mqtt_config(self):
        """Create disabled MQTT configuration."""
        return MQTTConfig(broker="test-broker", port=1883, publishing_enabled=False)

    @pytest.fixture
    def publisher(self, mqtt_config):
        """Create MQTTPublisher instance."""
        return MQTTPublisher(mqtt_config, client_id="test-client")

    @pytest.fixture
    def mock_mqtt_client(self):
        """Create mock MQTT client."""
        mock_client = Mock(spec=mqtt_client.Client)
        mock_client.connect = Mock(return_value=0)
        mock_client.disconnect = Mock(return_value=0)
        mock_client.publish = Mock()
        mock_client.loop_start = Mock()
        mock_client.loop_stop = Mock()
        mock_client.username_pw_set = Mock()
        mock_client.tls_set_context = Mock()
        return mock_client

    def test_publisher_initialization(self, mqtt_config):
        """Test MQTTPublisher initialization."""
        publisher = MQTTPublisher(mqtt_config, client_id="test-client")

        assert publisher.config == mqtt_config
        assert publisher.client_id == "test-client"
        assert isinstance(publisher.connection_status, MQTTConnectionStatus)
        assert publisher.client is None
        assert publisher.message_queue == []
        assert publisher.max_queue_size == 1000
        assert publisher.total_messages_published == 0
        assert publisher.total_messages_failed == 0
        assert publisher.total_bytes_published == 0
        assert publisher._publisher_active is False

    def test_publisher_auto_client_id(self, mqtt_config):
        """Test automatic client ID generation."""
        with patch("src.integration.mqtt_publisher.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.timestamp.return_value = 1234567890.0

            publisher = MQTTPublisher(mqtt_config)
            assert publisher.client_id == "ha_ml_predictor_1234567890"

    @pytest.mark.asyncio
    async def test_initialize_disabled_publishing(self, disabled_mqtt_config):
        """Test initialization with disabled publishing."""
        publisher = MQTTPublisher(disabled_mqtt_config)
        await publisher.initialize()

        assert publisher.client is None
        assert not publisher._publisher_active

    @pytest.mark.asyncio
    async def test_initialize_with_authentication(self, mqtt_config, mock_mqtt_client):
        """Test initialization with authentication."""
        with patch("paho.mqtt.client.Client", return_value=mock_mqtt_client):
            publisher = MQTTPublisher(mqtt_config)

            # Mock successful connection
            with patch.object(publisher, "_connect_to_broker", new_callable=AsyncMock):
                with patch.object(publisher, "start_publisher", new_callable=AsyncMock):
                    await publisher.initialize()

            mock_mqtt_client.username_pw_set.assert_called_once_with(
                mqtt_config.username, mqtt_config.password
            )

    @pytest.mark.asyncio
    async def test_initialize_with_tls(self, mqtt_config, mock_mqtt_client):
        """Test initialization with TLS encryption."""
        mqtt_config.port = 8883  # Enable TLS

        with patch("paho.mqtt.client.Client", return_value=mock_mqtt_client):
            with patch("ssl.create_default_context") as mock_ssl_context:
                mock_context = Mock()
                mock_ssl_context.return_value = mock_context

                publisher = MQTTPublisher(mqtt_config)

                # Mock successful connection
                with patch.object(
                    publisher, "_connect_to_broker", new_callable=AsyncMock
                ):
                    with patch.object(
                        publisher, "start_publisher", new_callable=AsyncMock
                    ):
                        await publisher.initialize()

                mock_ssl_context.assert_called_once_with(ssl.Purpose.SERVER_AUTH)
                mock_mqtt_client.tls_set_context.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_initialize_connection_failure(self, mqtt_config):
        """Test initialization with connection failure."""
        publisher = MQTTPublisher(mqtt_config)

        with patch("paho.mqtt.client.Client"):
            with patch.object(
                publisher,
                "_connect_to_broker",
                new_callable=AsyncMock,
                side_effect=Exception("Connection failed"),
            ):
                with pytest.raises(MQTTPublisherError) as exc_info:
                    await publisher.initialize()

                assert "Failed to initialize MQTT publisher" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_start_publisher_already_active(self, publisher):
        """Test starting publisher when already active."""
        publisher._publisher_active = True

        with patch("asyncio.create_task") as mock_create_task:
            await publisher.start_publisher()
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_publisher_disabled(self, disabled_mqtt_config):
        """Test starting publisher when publishing is disabled."""
        publisher = MQTTPublisher(disabled_mqtt_config)

        with patch("asyncio.create_task") as mock_create_task:
            await publisher.start_publisher()
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_start_publisher_success(self, publisher):
        """Test successful publisher startup."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_task = Mock()
            mock_create_task.return_value = mock_task

            await publisher.start_publisher()

            assert publisher._publisher_active is True
            assert len(publisher._background_tasks) == 2
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_stop_publisher_not_active(self, publisher):
        """Test stopping publisher when not active."""
        await publisher.stop_publisher()
        # Should complete without error

    @pytest.mark.asyncio
    async def test_stop_publisher_with_queued_messages(
        self, publisher, mock_mqtt_client
    ):
        """Test stopping publisher with queued messages."""
        publisher._publisher_active = True
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True
        publisher.message_queue = [
            {"topic": "test", "payload": "data", "qos": 1, "retain": False}
        ]

        with patch.object(publisher, "_process_message_queue", new_callable=AsyncMock):
            with patch.object(
                publisher, "_disconnect_from_broker", new_callable=AsyncMock
            ):
                with patch("asyncio.gather", new_callable=AsyncMock):
                    await publisher.stop_publisher()

        assert not publisher._publisher_active

    @pytest.mark.asyncio
    async def test_stop_publisher_timeout(self, publisher, mock_mqtt_client):
        """Test stopping publisher with timeout processing messages."""
        publisher._publisher_active = True
        publisher.message_queue = [
            {"topic": "test", "payload": "data", "qos": 1, "retain": False}
        ]

        async def timeout_process():
            raise asyncio.TimeoutError()

        with patch.object(
            publisher, "_process_message_queue", side_effect=timeout_process
        ):
            with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError()):
                await publisher.stop_publisher()

        assert not publisher._publisher_active

    @pytest.mark.asyncio
    async def test_publish_string_payload(self, publisher, mock_mqtt_client):
        """Test publishing string payload."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        # Mock successful publish
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        mock_info.mid = 12345
        mock_mqtt_client.publish.return_value = mock_info

        result = await publisher.publish("test/topic", "test message")

        assert result.success is True
        assert result.topic == "test/topic"
        assert result.payload_size > 0
        assert result.message_id == 12345
        assert publisher.total_messages_published == 1
        mock_mqtt_client.publish.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_dict_payload(self, publisher, mock_mqtt_client):
        """Test publishing dictionary payload."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        mock_info.mid = 12345
        mock_mqtt_client.publish.return_value = mock_info

        payload = {"key": "value", "number": 42}
        result = await publisher.publish("test/topic", payload, qos=2, retain=True)

        assert result.success is True
        expected_payload = json.dumps(payload, default=str)
        mock_mqtt_client.publish.assert_called_once_with(
            "test/topic", expected_payload, qos=2, retain=True
        )

    @pytest.mark.asyncio
    async def test_publish_bytes_payload(self, publisher, mock_mqtt_client):
        """Test publishing bytes payload."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        mock_info.mid = 12345
        mock_mqtt_client.publish.return_value = mock_info

        payload = b"binary data"
        result = await publisher.publish("test/topic", payload)

        assert result.success is True
        mock_mqtt_client.publish.assert_called_once_with(
            "test/topic", "binary data", qos=1, retain=False
        )

    @pytest.mark.asyncio
    async def test_publish_not_connected(self, publisher):
        """Test publishing when not connected."""
        publisher.client = None
        publisher.connection_status.connected = False

        result = await publisher.publish("test/topic", "test message")

        assert result.success is False
        assert "not connected" in result.error_message.lower()
        assert len(publisher.message_queue) == 1
        assert publisher.message_queue[0]["topic"] == "test/topic"

    @pytest.mark.asyncio
    async def test_publish_queue_full(self, publisher):
        """Test publishing when message queue is full."""
        publisher.client = None
        publisher.connection_status.connected = False
        publisher.max_queue_size = 2

        # Fill the queue
        publisher.message_queue = [
            {"topic": "old1", "payload": "data1"},
            {"topic": "old2", "payload": "data2"},
        ]

        result = await publisher.publish("test/topic", "new message")

        assert result.success is False
        assert len(publisher.message_queue) == 2
        # Oldest message should be removed
        assert publisher.message_queue[0]["topic"] == "old2"
        assert publisher.message_queue[1]["topic"] == "test/topic"

    @pytest.mark.asyncio
    async def test_publish_mqtt_error(self, publisher, mock_mqtt_client):
        """Test publishing with MQTT error."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_NO_CONN  # Connection error
        mock_mqtt_client.publish.return_value = mock_info

        result = await publisher.publish("test/topic", "test message")

        assert result.success is False
        assert "return code" in result.error_message.lower()
        assert publisher.total_messages_failed == 1

    @pytest.mark.asyncio
    async def test_publish_exception(self, publisher, mock_mqtt_client):
        """Test publishing with exception."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True
        mock_mqtt_client.publish.side_effect = Exception("Publish failed")

        result = await publisher.publish("test/topic", "test message")

        assert result.success is False
        assert "Exception during MQTT publish" in result.error_message
        assert publisher.total_messages_failed == 1

    @pytest.mark.asyncio
    async def test_publish_json(self, publisher, mock_mqtt_client):
        """Test publish_json method."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
        mock_info.mid = 12345
        mock_mqtt_client.publish.return_value = mock_info

        data = {"temperature": 23.5, "humidity": 45}
        result = await publisher.publish_json("sensors/data", data, qos=2)

        assert result.success is True
        expected_json = json.dumps(data, default=str)
        mock_mqtt_client.publish.assert_called_once_with(
            "sensors/data", expected_json, qos=2, retain=False
        )

    def test_get_connection_status(self, publisher):
        """Test get_connection_status method."""
        now = datetime.utcnow()
        publisher.connection_status.connected = True
        publisher.connection_status.last_connected = now

        with patch("src.integration.mqtt_publisher.datetime") as mock_dt:
            mock_dt.utcnow.return_value = now + timedelta(seconds=120)

            status = publisher.get_connection_status()

            assert status.connected is True
            assert abs(status.uptime_seconds - 120) < 1

    def test_get_connection_status_not_connected(self, publisher):
        """Test get_connection_status when not connected."""
        status = publisher.get_connection_status()

        assert status.connected is False
        assert status.uptime_seconds == 0.0

    def test_get_publisher_stats(self, publisher):
        """Test get_publisher_stats method."""
        publisher.total_messages_published = 100
        publisher.total_messages_failed = 5
        publisher.total_bytes_published = 50000
        publisher.last_publish_time = datetime.utcnow()
        publisher.message_queue = ["msg1", "msg2"]

        stats = publisher.get_publisher_stats()

        assert stats["client_id"] == publisher.client_id
        assert stats["messages_published"] == 100
        assert stats["messages_failed"] == 5
        assert stats["bytes_published"] == 50000
        assert stats["queued_messages"] == 2
        assert stats["max_queue_size"] == 1000
        assert "connection_status" in stats
        assert "config" in stats

    @pytest.mark.asyncio
    async def test_connect_to_broker_success(self, publisher, mock_mqtt_client):
        """Test successful broker connection."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = False

        async def mock_connect():
            # Simulate successful connection
            publisher.connection_status.connected = True

        mock_mqtt_client.connect = Mock(side_effect=mock_connect)

        await publisher._connect_to_broker()

        mock_mqtt_client.connect.assert_called_once_with(
            publisher.config.broker, publisher.config.port, publisher.config.keepalive
        )
        mock_mqtt_client.loop_start.assert_called_once()

    @pytest.mark.asyncio
    async def test_connect_to_broker_max_attempts(self, publisher, mock_mqtt_client):
        """Test broker connection with max attempts exceeded."""
        publisher.client = mock_mqtt_client
        publisher.config.max_reconnect_attempts = 2
        mock_mqtt_client.connect.side_effect = Exception("Connection failed")

        with pytest.raises(MQTTPublisherError) as exc_info:
            await publisher._connect_to_broker()

        assert "Failed to connect after 2 attempts" in str(exc_info.value)
        assert mock_mqtt_client.connect.call_count == 2

    @pytest.mark.asyncio
    async def test_connect_to_broker_timeout(self, publisher, mock_mqtt_client):
        """Test broker connection timeout."""
        publisher.client = mock_mqtt_client
        publisher.config.connection_timeout = 0.1  # Very short timeout
        publisher.connection_status.connected = False  # Never becomes connected

        with patch("asyncio.sleep", new_callable=AsyncMock):
            with pytest.raises(TimeoutError):
                await publisher._connect_to_broker()

    @pytest.mark.asyncio
    async def test_disconnect_from_broker(self, publisher, mock_mqtt_client):
        """Test disconnecting from broker."""
        publisher.client = mock_mqtt_client

        await publisher._disconnect_from_broker()

        mock_mqtt_client.loop_stop.assert_called_once()
        mock_mqtt_client.disconnect.assert_called_once()

    @pytest.mark.asyncio
    async def test_disconnect_from_broker_exception(self, publisher, mock_mqtt_client):
        """Test disconnecting with exception."""
        publisher.client = mock_mqtt_client
        mock_mqtt_client.loop_stop.side_effect = Exception("Disconnect failed")

        # Should not raise exception, just log error
        await publisher._disconnect_from_broker()

    @pytest.mark.asyncio
    async def test_connection_monitoring_loop(self, publisher, mock_mqtt_client):
        """Test connection monitoring loop."""
        publisher.client = mock_mqtt_client
        publisher._shutdown_event = asyncio.Event()

        # Set shutdown after short delay
        async def set_shutdown():
            await asyncio.sleep(0.1)
            publisher._shutdown_event.set()

        monitor_task = asyncio.create_task(set_shutdown())

        with patch.object(publisher, "_connect_to_broker", new_callable=AsyncMock):
            # This should exit quickly due to shutdown event
            await publisher._connection_monitoring_loop()

        await monitor_task

    @pytest.mark.asyncio
    async def test_connection_monitoring_loop_reconnect(
        self, publisher, mock_mqtt_client
    ):
        """Test connection monitoring with reconnection."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = False
        publisher._shutdown_event = asyncio.Event()

        reconnect_called = False

        async def mock_reconnect():
            nonlocal reconnect_called
            reconnect_called = True
            publisher._shutdown_event.set()  # Stop the loop

        with patch.object(publisher, "_connect_to_broker", side_effect=mock_reconnect):
            await publisher._connection_monitoring_loop()

        assert reconnect_called

    @pytest.mark.asyncio
    async def test_message_queue_processing_loop(self, publisher):
        """Test message queue processing loop."""
        publisher._shutdown_event = asyncio.Event()
        publisher.connection_status.connected = True
        publisher.message_queue = [{"topic": "test", "payload": "data"}]

        process_called = False

        async def mock_process():
            nonlocal process_called
            process_called = True
            publisher._shutdown_event.set()  # Stop the loop

        with patch.object(
            publisher, "_process_message_queue", side_effect=mock_process
        ):
            await publisher._message_queue_processing_loop()

        assert process_called

    @pytest.mark.asyncio
    async def test_process_message_queue(self, publisher, mock_mqtt_client):
        """Test processing queued messages."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        # Add messages to queue
        publisher.message_queue = [
            {"topic": "test1", "payload": "data1", "qos": 1, "retain": False},
            {"topic": "test2", "payload": "data2", "qos": 2, "retain": True},
        ]

        # Mock successful publishing
        async def mock_publish(*args, **kwargs):
            return MQTTPublishResult(
                success=True,
                topic=args[0],
                payload_size=len(args[1]),
                publish_time=datetime.utcnow(),
            )

        with patch.object(publisher, "publish", side_effect=mock_publish):
            await publisher._process_message_queue()

        # Queue should be empty
        assert len(publisher.message_queue) == 0

    @pytest.mark.asyncio
    async def test_process_message_queue_failures(self, publisher, mock_mqtt_client):
        """Test processing queued messages with some failures."""
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        publisher.message_queue = [
            {"topic": "test1", "payload": "data1", "qos": 1, "retain": False}
        ]

        # Mock failed publishing
        async def mock_publish(*args, **kwargs):
            return MQTTPublishResult(
                success=False,
                topic=args[0],
                payload_size=len(args[1]),
                publish_time=datetime.utcnow(),
                error_message="Publish failed",
            )

        with patch.object(publisher, "publish", side_effect=mock_publish):
            await publisher._process_message_queue()

        # Queue should still be empty (messages processed even if failed)
        assert len(publisher.message_queue) == 0

    def test_on_connect_success(self, publisher, mock_mqtt_client):
        """Test on_connect callback with successful connection."""
        publisher.connection_status.connected = False

        publisher._on_connect(mock_mqtt_client, None, {}, 0, None)

        assert publisher.connection_status.connected is True
        assert publisher.connection_status.reconnect_count == 1
        assert publisher.connection_status.last_connected is not None

    def test_on_connect_failure(self, publisher, mock_mqtt_client):
        """Test on_connect callback with connection failure."""
        publisher.connection_status.connected = False

        publisher._on_connect(
            mock_mqtt_client, None, {}, 1, None
        )  # Non-zero reason code

        assert publisher.connection_status.connected is False
        assert (
            "Connection failed with reason code: 1"
            in publisher.connection_status.last_error
        )

    def test_on_connect_with_callback(self, publisher, mock_mqtt_client):
        """Test on_connect with user callback."""
        callback_called = False

        def test_callback(client, userdata, flags, reason_code):
            nonlocal callback_called
            callback_called = True

        publisher.on_connect_callback = test_callback
        publisher._on_connect(mock_mqtt_client, None, {}, 0, None)

        assert callback_called

    def test_on_connect_callback_exception(self, publisher, mock_mqtt_client):
        """Test on_connect with callback that raises exception."""

        def failing_callback(client, userdata, flags, reason_code):
            raise Exception("Callback failed")

        publisher.on_connect_callback = failing_callback

        # Should not raise exception, just log error
        publisher._on_connect(mock_mqtt_client, None, {}, 0, None)

    def test_on_disconnect(self, publisher, mock_mqtt_client):
        """Test on_disconnect callback."""
        publisher.connection_status.connected = True

        publisher._on_disconnect(mock_mqtt_client, None, {}, 0, None)

        assert publisher.connection_status.connected is False
        assert publisher.connection_status.last_disconnected is not None

    def test_on_disconnect_unexpected(self, publisher, mock_mqtt_client):
        """Test on_disconnect with unexpected disconnection."""
        publisher.connection_status.connected = True

        publisher._on_disconnect(
            mock_mqtt_client, None, {}, 1, None
        )  # Non-zero reason code

        assert publisher.connection_status.connected is False

    def test_on_publish_success(self, publisher, mock_mqtt_client):
        """Test on_publish callback with success."""
        # Should not raise exception
        publisher._on_publish(mock_mqtt_client, None, 12345, 0, None)

    def test_on_publish_failure(self, publisher, mock_mqtt_client):
        """Test on_publish callback with failure."""
        # Should not raise exception, just log warning
        publisher._on_publish(mock_mqtt_client, None, 12345, 1, None)

    def test_on_message(self, publisher, mock_mqtt_client):
        """Test on_message callback."""
        mock_message = Mock()
        mock_message.topic = "test/topic"

        # Should not raise exception
        publisher._on_message(mock_mqtt_client, None, mock_message)

    def test_on_message_with_callback(self, publisher, mock_mqtt_client):
        """Test on_message with user callback."""
        callback_called = False

        def test_callback(client, userdata, message):
            nonlocal callback_called
            callback_called = True

        publisher.on_message_callback = test_callback

        mock_message = Mock()
        mock_message.topic = "test/topic"

        publisher._on_message(mock_mqtt_client, None, mock_message)

        assert callback_called

    def test_on_log(self, publisher, mock_mqtt_client):
        """Test on_log callback with different log levels."""
        # Test different MQTT log levels
        log_levels = [
            mqtt_client.MQTT_LOG_DEBUG,
            mqtt_client.MQTT_LOG_INFO,
            mqtt_client.MQTT_LOG_NOTICE,
            mqtt_client.MQTT_LOG_WARNING,
            mqtt_client.MQTT_LOG_ERR,
        ]

        for level in log_levels:
            # Should not raise exception
            publisher._on_log(mock_mqtt_client, None, level, "Test log message")


class TestMQTTPublisherError:
    """Test MQTTPublisherError exception."""

    def test_error_initialization(self):
        """Test MQTTPublisherError initialization."""
        error = MQTTPublisherError("Test error message")

        assert str(error) == "Test error message"
        assert error.error_code == "MQTT_PUBLISHER_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_error_with_cause(self):
        """Test MQTTPublisherError with cause."""
        original_error = Exception("Original error")
        error = MQTTPublisherError("Wrapper error", cause=original_error)

        assert str(error) == "Wrapper error"
        assert error.error_code == "MQTT_PUBLISHER_ERROR"

    def test_error_with_custom_severity(self):
        """Test MQTTPublisherError with custom severity."""
        error = MQTTPublisherError("Critical error", severity=ErrorSeverity.HIGH)

        assert error.severity == ErrorSeverity.HIGH


class TestMQTTPublisherIntegration:
    """Integration tests for MQTTPublisher."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self, mqtt_config):
        """Test complete publisher lifecycle."""
        publisher = MQTTPublisher(mqtt_config)

        with patch("paho.mqtt.client.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Mock successful connection
            async def mock_connect():
                publisher.connection_status.connected = True

            with patch.object(
                publisher, "_connect_to_broker", side_effect=mock_connect
            ):
                # Initialize
                await publisher.initialize()
                assert publisher._publisher_active is True

                # Publish message
                mock_info = Mock()
                mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
                mock_info.mid = 12345
                mock_client.publish.return_value = mock_info

                result = await publisher.publish("test/topic", "test message")
                assert result.success is True

                # Stop
                await publisher.stop_publisher()
                assert publisher._publisher_active is False

    @pytest.mark.asyncio
    async def test_reconnection_scenario(self, mqtt_config):
        """Test reconnection scenario."""
        publisher = MQTTPublisher(mqtt_config)

        with patch("paho.mqtt.client.Client") as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client

            # Simulate connection loss and recovery
            connection_attempts = 0

            async def mock_connect():
                nonlocal connection_attempts
                connection_attempts += 1
                if connection_attempts == 1:
                    raise Exception("First connection failed")
                else:
                    publisher.connection_status.connected = True

            with patch.object(
                publisher, "_connect_to_broker", side_effect=mock_connect
            ):
                # This should succeed on second attempt
                await publisher._connect_to_broker()
                assert publisher.connection_status.connected is True
                assert connection_attempts == 2
