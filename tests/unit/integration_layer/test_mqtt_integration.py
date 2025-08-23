"""Unit tests for MQTT integration and messaging.

Covers:
- src/integration/mqtt_publisher.py (MQTT Publisher)
- src/integration/mqtt_integration_manager.py (MQTT Integration Management)
- src/integration/enhanced_mqtt_manager.py (Enhanced MQTT Features)

This test file consolidates testing for all MQTT integration functionality.
"""

import asyncio
from datetime import datetime, timezone
import json
import ssl
import threading
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

# Mock paho.mqtt.client before importing modules
import paho.mqtt.client as mqtt_client
import pytest

# Import modules after mocking dependencies
from src.core.config import MQTTConfig, RoomConfig
from src.core.exceptions import ErrorSeverity, OccupancyPredictionError
from src.integration.mqtt_integration_manager import (
    MQTTIntegrationManager,
    MQTTIntegrationStats,
)
from src.integration.mqtt_publisher import (
    MQTTConnectionStatus,
    MQTTPublisher,
    MQTTPublishResult,
)


@pytest.fixture
def mqtt_config():
    """MQTT configuration fixture."""
    return MQTTConfig(
        broker="test-broker",
        port=1883,
        username="test_user",
        password="test_pass",
        topic_prefix="ha_ml_predictor",
        publishing_enabled=True,
        discovery_enabled=True,
        keepalive=60,
        connection_timeout=30,
        reconnect_delay_seconds=5,
        max_reconnect_attempts=3,
        publish_system_status=True,
        status_update_interval_seconds=300,
    )


@pytest.fixture
def room_config():
    """Room configuration fixture."""
    return {
        "living_room": RoomConfig(
            room_id="living_room",
            name="Living Room",
            sensors={
                "motion": ["binary_sensor.living_room_motion"],
                "door": ["binary_sensor.living_room_door"],
            },
        )
    }


@pytest.fixture
def mock_mqtt_client():
    """Mock MQTT client fixture."""
    mock_client = Mock(spec=mqtt_client.Client)
    mock_client.connect.return_value = None
    mock_client.disconnect.return_value = None
    mock_client.loop_start.return_value = None
    mock_client.loop_stop.return_value = None
    mock_client.username_pw_set.return_value = None
    mock_client.tls_set_context.return_value = None

    # Mock publish to return success
    mock_info = Mock()
    mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
    mock_info.mid = 123
    mock_client.publish.return_value = mock_info

    return mock_client


class TestMQTTConnectionStatus:
    """Test MQTT connection status data class."""

    def test_connection_status_initialization(self):
        """Test connection status initialization."""
        status = MQTTConnectionStatus()

        assert not status.connected
        assert status.last_connected is None
        assert status.last_disconnected is None
        assert status.connection_attempts == 0
        assert status.last_error is None
        assert status.reconnect_count == 0
        assert status.uptime_seconds == 0.0

    def test_connection_status_with_values(self):
        """Test connection status with specific values."""
        now = datetime.utcnow()
        status = MQTTConnectionStatus(
            connected=True,
            last_connected=now,
            connection_attempts=2,
            last_error="Test error",
            reconnect_count=1,
            uptime_seconds=120.5,
        )

        assert status.connected
        assert status.last_connected == now
        assert status.connection_attempts == 2
        assert status.last_error == "Test error"
        assert status.reconnect_count == 1
        assert status.uptime_seconds == 120.5


class TestMQTTPublishResult:
    """Test MQTT publish result data class."""

    def test_publish_result_success(self):
        """Test successful publish result."""
        now = datetime.utcnow()
        result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=now,
            message_id=123,
        )

        assert result.success
        assert result.topic == "test/topic"
        assert result.payload_size == 100
        assert result.publish_time == now
        assert result.error_message is None
        assert result.message_id == 123

    def test_publish_result_failure(self):
        """Test failed publish result."""
        now = datetime.utcnow()
        result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=100,
            publish_time=now,
            error_message="Connection failed",
        )

        assert not result.success
        assert result.topic == "test/topic"
        assert result.payload_size == 100
        assert result.publish_time == now
        assert result.error_message == "Connection failed"
        assert result.message_id is None


class TestMQTTPublisher:
    """Test MQTT publisher functionality."""

    def test_mqtt_publisher_initialization(self, mqtt_config):
        """Test MQTT publisher initialization."""
        publisher = MQTTPublisher(config=mqtt_config, client_id="test_client")

        assert publisher.config == mqtt_config
        assert publisher.client_id == "test_client"
        assert publisher.client is None
        assert not publisher.connection_status.connected
        assert publisher.message_queue == []
        assert publisher.max_queue_size == 1000
        assert not publisher._publisher_active
        assert publisher.total_messages_published == 0
        assert publisher.total_messages_failed == 0
        assert publisher.total_bytes_published == 0

    def test_mqtt_publisher_auto_client_id(self, mqtt_config):
        """Test MQTT publisher with auto-generated client ID."""
        with patch("src.integration.mqtt_publisher.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value.timestamp.return_value = 1234567890

            publisher = MQTTPublisher(config=mqtt_config)

            assert "ha_ml_predictor_1234567890" in publisher.client_id

    @patch("paho.mqtt.client.Client")
    @pytest.mark.asyncio
    async def test_mqtt_publisher_initialize_success(
        self, mock_client_class, mqtt_config, mock_mqtt_client
    ):
        """Test successful MQTT publisher initialization."""
        mock_client_class.return_value = mock_mqtt_client

        publisher = MQTTPublisher(config=mqtt_config)

        with patch.object(
            publisher, "_connect_to_broker", new_callable=AsyncMock
        ) as mock_connect:
            with patch.object(
                publisher, "start_publisher", new_callable=AsyncMock
            ) as mock_start:
                await publisher.initialize()

                mock_connect.assert_called_once()
                mock_start.assert_called_once()
                assert publisher.client == mock_mqtt_client

    @patch("paho.mqtt.client.Client")
    @pytest.mark.asyncio
    async def test_mqtt_publisher_initialize_disabled(
        self, mock_client_class, mqtt_config, mock_mqtt_client
    ):
        """Test MQTT publisher initialization when publishing is disabled."""
        mqtt_config.publishing_enabled = False
        mock_client_class.return_value = mock_mqtt_client

        publisher = MQTTPublisher(config=mqtt_config)
        await publisher.initialize()

        assert publisher.client is None
        mock_client_class.assert_not_called()

    @patch("paho.mqtt.client.Client")
    @pytest.mark.asyncio
    async def test_mqtt_publisher_initialize_with_tls(
        self, mock_client_class, mqtt_config, mock_mqtt_client
    ):
        """Test MQTT publisher initialization with TLS."""
        mqtt_config.port = 8883  # TLS port
        mock_client_class.return_value = mock_mqtt_client

        publisher = MQTTPublisher(config=mqtt_config)

        with patch.object(publisher, "_connect_to_broker", new_callable=AsyncMock):
            with patch.object(publisher, "start_publisher", new_callable=AsyncMock):
                with patch("ssl.create_default_context") as mock_ssl:
                    mock_context = Mock()
                    mock_ssl.return_value = mock_context

                    await publisher.initialize()

                    mock_ssl.assert_called_once_with(ssl.Purpose.SERVER_AUTH)
                    mock_mqtt_client.tls_set_context.assert_called_once_with(
                        mock_context
                    )

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_success(self, mqtt_config, mock_mqtt_client):
        """Test successful message publishing."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        payload = {"test": "data", "value": 123}
        result = await publisher.publish("test/topic", payload, qos=1, retain=True)

        assert result.success
        assert result.topic == "test/topic"
        assert result.payload_size > 0
        assert result.message_id == 123
        assert result.error_message is None

        mock_mqtt_client.publish.assert_called_once_with(
            "test/topic", json.dumps(payload, default=str), qos=1, retain=True
        )

        assert publisher.total_messages_published == 1
        assert publisher.total_bytes_published > 0
        assert publisher.last_publish_time is not None

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_string_payload(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test publishing string payload."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        result = await publisher.publish("test/topic", "test message")

        assert result.success
        mock_mqtt_client.publish.assert_called_once_with(
            "test/topic", "test message", qos=1, retain=False
        )

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_bytes_payload(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test publishing bytes payload."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        payload_bytes = b"test message"
        result = await publisher.publish("test/topic", payload_bytes)

        assert result.success
        mock_mqtt_client.publish.assert_called_once_with(
            "test/topic", "test message", qos=1, retain=False
        )

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_disconnected_queues_message(
        self, mqtt_config
    ):
        """Test message queueing when disconnected."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = None  # Not connected

        result = await publisher.publish("test/topic", "test message")

        assert not result.success
        assert "not connected" in result.error_message.lower()
        assert len(publisher.message_queue) == 1

        queued_msg = publisher.message_queue[0]
        assert queued_msg["topic"] == "test/topic"
        assert queued_msg["payload"] == "test message"
        assert queued_msg["qos"] == 1
        assert queued_msg["retain"] is False

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_queue_full_removes_oldest(self, mqtt_config):
        """Test message queue behavior when full."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = None  # Not connected
        publisher.max_queue_size = 2

        # Fill queue
        await publisher.publish("topic1", "message1")
        await publisher.publish("topic2", "message2")
        await publisher.publish("topic3", "message3")  # Should remove first message

        assert len(publisher.message_queue) == 2
        assert publisher.message_queue[0]["topic"] == "topic2"
        assert publisher.message_queue[1]["topic"] == "topic3"

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_failure(self, mqtt_config, mock_mqtt_client):
        """Test publish failure handling."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        # Mock publish failure
        mock_info = Mock()
        mock_info.rc = mqtt_client.MQTT_ERR_NO_CONN
        mock_mqtt_client.publish.return_value = mock_info

        result = await publisher.publish("test/topic", "test message")

        assert not result.success
        assert "return code" in result.error_message
        assert publisher.total_messages_failed == 1

    @pytest.mark.asyncio
    async def test_mqtt_publisher_publish_json_convenience(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test JSON publishing convenience method."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        data = {"temperature": 22.5, "humidity": 65}
        result = await publisher.publish_json("sensors/room1", data)

        assert result.success
        mock_mqtt_client.publish.assert_called_once_with(
            "sensors/room1", json.dumps(data, default=str), qos=1, retain=False
        )

    def test_mqtt_publisher_get_connection_status(self, mqtt_config):
        """Test getting connection status."""
        publisher = MQTTPublisher(config=mqtt_config)

        # Test disconnected status
        status = publisher.get_connection_status()
        assert not status.connected
        assert status.uptime_seconds == 0.0

        # Test connected status
        past_time = datetime.utcnow()
        publisher.connection_status.connected = True
        publisher.connection_status.last_connected = past_time

        with patch("src.integration.mqtt_publisher.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = past_time.replace(
                second=past_time.second + 60
            )

            status = publisher.get_connection_status()
            assert status.connected
            assert status.uptime_seconds == 60.0

    def test_mqtt_publisher_get_publisher_stats(self, mqtt_config):
        """Test getting publisher statistics."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.total_messages_published = 10
        publisher.total_messages_failed = 2
        publisher.total_bytes_published = 1024
        publisher.last_publish_time = datetime.utcnow()

        stats = publisher.get_publisher_stats()

        assert stats["client_id"] == publisher.client_id
        assert stats["messages_published"] == 10
        assert stats["messages_failed"] == 2
        assert stats["bytes_published"] == 1024
        assert stats["last_publish_time"] is not None
        assert stats["queued_messages"] == 0
        assert stats["max_queue_size"] == 1000
        assert stats["publisher_active"] is False
        assert stats["config"]["broker"] == mqtt_config.broker

    @pytest.mark.asyncio
    async def test_mqtt_publisher_start_stop_lifecycle(self, mqtt_config):
        """Test publisher start and stop lifecycle."""
        publisher = MQTTPublisher(config=mqtt_config)

        assert not publisher._publisher_active

        with patch.object(
            publisher, "_connection_monitoring_loop", new_callable=AsyncMock
        ) as mock_conn_loop:
            with patch.object(
                publisher, "_message_queue_processing_loop", new_callable=AsyncMock
            ) as mock_queue_loop:
                with patch("asyncio.create_task") as mock_create_task:
                    mock_task = Mock()
                    mock_create_task.return_value = mock_task

                    await publisher.start_publisher()

                    assert publisher._publisher_active
                    assert mock_create_task.call_count == 2  # Two background tasks
                    assert len(publisher._background_tasks) == 2

        # Test stop
        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await publisher.stop_publisher()

            assert not publisher._publisher_active
            assert publisher._shutdown_event.is_set()
            mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_mqtt_publisher_start_disabled(self, mqtt_config):
        """Test starting publisher when disabled."""
        mqtt_config.publishing_enabled = False
        publisher = MQTTPublisher(config=mqtt_config)

        await publisher.start_publisher()

        assert not publisher._publisher_active
        assert len(publisher._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_mqtt_publisher_connect_to_broker_success(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test successful broker connection."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client

        # Simulate successful connection
        def mock_connect(*args):
            publisher.connection_status.connected = True

        mock_mqtt_client.connect.side_effect = mock_connect

        await publisher._connect_to_broker()

        mock_mqtt_client.connect.assert_called_once_with(
            mqtt_config.broker, mqtt_config.port, mqtt_config.keepalive
        )
        mock_mqtt_client.loop_start.assert_called_once()
        assert publisher.connection_status.connected

    @pytest.mark.asyncio
    async def test_mqtt_publisher_connect_to_broker_retry(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test broker connection with retries."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client

        # First call fails, second succeeds
        call_count = 0

        def mock_connect(*args):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("Connection failed")
            else:
                publisher.connection_status.connected = True

        mock_mqtt_client.connect.side_effect = mock_connect

        with patch("asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await publisher._connect_to_broker()

            assert mock_mqtt_client.connect.call_count == 2
            mock_sleep.assert_called_once_with(mqtt_config.reconnect_delay_seconds)
            assert publisher.connection_status.connection_attempts == 2

    @pytest.mark.asyncio
    async def test_mqtt_publisher_connect_to_broker_max_attempts(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test broker connection max attempts exceeded."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client

        # Always fail
        mock_mqtt_client.connect.side_effect = ConnectionError("Connection failed")

        with pytest.raises(Exception) as exc_info:
            with patch("asyncio.sleep", new_callable=AsyncMock):
                await publisher._connect_to_broker()

        assert "Failed to connect after" in str(exc_info.value)
        assert mock_mqtt_client.connect.call_count == mqtt_config.max_reconnect_attempts

    @pytest.mark.asyncio
    async def test_mqtt_publisher_process_message_queue(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test message queue processing."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        # Add messages to queue
        publisher.message_queue.extend(
            [
                {
                    "topic": "test/topic1",
                    "payload": "message1",
                    "qos": 1,
                    "retain": False,
                    "queued_at": datetime.utcnow(),
                },
                {
                    "topic": "test/topic2",
                    "payload": "message2",
                    "qos": 2,
                    "retain": True,
                    "queued_at": datetime.utcnow(),
                },
            ]
        )

        await publisher._process_message_queue()

        # Verify messages were published
        assert mock_mqtt_client.publish.call_count == 2
        mock_mqtt_client.publish.assert_has_calls(
            [
                call("test/topic1", "message1", qos=1, retain=False),
                call("test/topic2", "message2", qos=2, retain=True),
            ]
        )

        # Verify queue is empty
        assert len(publisher.message_queue) == 0

    def test_mqtt_publisher_callbacks(self, mqtt_config):
        """Test MQTT client callback handlers."""
        mock_connect_callback = Mock()
        mock_disconnect_callback = Mock()
        mock_message_callback = Mock()

        publisher = MQTTPublisher(
            config=mqtt_config,
            on_connect_callback=mock_connect_callback,
            on_disconnect_callback=mock_disconnect_callback,
            on_message_callback=mock_message_callback,
        )

        # Test connect callback
        publisher._on_connect(None, None, None, 0, None)
        assert publisher.connection_status.connected
        mock_connect_callback.assert_called_once_with(None, None, None, 0)

        # Test disconnect callback
        publisher._on_disconnect(None, None, None, 0, None)
        assert not publisher.connection_status.connected
        mock_disconnect_callback.assert_called_once_with(None, None, None, 0)

        # Test message callback
        mock_message = Mock()
        mock_message.topic = "test/topic"
        publisher._on_message(None, None, mock_message)
        mock_message_callback.assert_called_once_with(None, None, mock_message)

    def test_mqtt_publisher_log_callback(self, mqtt_config):
        """Test MQTT client log callback."""
        publisher = MQTTPublisher(config=mqtt_config)

        with patch("src.integration.mqtt_publisher.logger") as mock_logger:
            # Test different log levels
            publisher._on_log(None, None, mqtt_client.MQTT_LOG_DEBUG, "Debug message")
            mock_logger.debug.assert_called_with("MQTT: Debug message")

            publisher._on_log(None, None, mqtt_client.MQTT_LOG_ERR, "Error message")
            mock_logger.error.assert_called_with("MQTT: Error message")

    def test_mqtt_publisher_publish_callback(self, mqtt_config):
        """Test MQTT publish callback."""
        publisher = MQTTPublisher(config=mqtt_config)

        with patch("src.integration.mqtt_publisher.logger") as mock_logger:
            # Test successful publish
            publisher._on_publish(None, None, 123, 0, None)
            mock_logger.debug.assert_called_with(
                "MQTT message published successfully (mid: 123)"
            )

            # Test failed publish
            publisher._on_publish(None, None, 124, 1, None)
            mock_logger.warning.assert_called_with(
                "MQTT publish failed (mid: 124, reason_code: 1)"
            )


@pytest.fixture
def mock_prediction_result():
    """Mock prediction result fixture."""
    from src.models.base.predictor import PredictionResult

    return PredictionResult(
        prediction_time=datetime.now(timezone.utc).replace(minute=30),
        confidence=0.85,
        model_version="test_v1.0",
        features_used={"time_since_last": 3600, "day_of_week": 2},
        prediction_type="occupied",
        room_id="living_room",
    )


class TestMQTTIntegrationStats:
    """Test MQTT integration statistics data class."""

    def test_stats_initialization(self):
        """Test statistics initialization."""
        stats = MQTTIntegrationStats()

        assert not stats.initialized
        assert not stats.mqtt_connected
        assert not stats.discovery_published
        assert stats.predictions_published == 0
        assert stats.status_updates_published == 0
        assert stats.last_prediction_published is None
        assert stats.last_status_published is None
        assert stats.total_errors == 0
        assert stats.last_error is None


class TestMQTTIntegrationManager:
    """Test MQTT integration management."""

    @patch("src.integration.mqtt_integration_manager.get_config")
    def test_integration_manager_initialization(
        self, mock_get_config, mqtt_config, room_config
    ):
        """Test integration manager initialization."""
        mock_system_config = Mock()
        mock_system_config.mqtt = mqtt_config
        mock_system_config.rooms = room_config
        mock_get_config.return_value = mock_system_config

        manager = MQTTIntegrationManager()

        assert manager.mqtt_config == mqtt_config
        assert manager.rooms == room_config
        assert manager.notification_callbacks == []
        assert manager.mqtt_publisher is None
        assert manager.prediction_publisher is None
        assert manager.discovery_publisher is None
        assert not manager.stats.initialized
        assert not manager._integration_active

    def test_integration_manager_with_explicit_config(self, mqtt_config, room_config):
        """Test integration manager with explicit configuration."""
        callbacks = [Mock(), Mock()]

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config, rooms=room_config, notification_callbacks=callbacks
        )

        assert manager.mqtt_config == mqtt_config
        assert manager.rooms == room_config
        assert manager.notification_callbacks == callbacks

    @patch("src.integration.mqtt_integration_manager.MQTTPublisher")
    @patch("src.integration.mqtt_integration_manager.PredictionPublisher")
    @patch("src.integration.mqtt_integration_manager.DiscoveryPublisher")
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_success(
        self,
        mock_discovery_cls,
        mock_prediction_cls,
        mock_mqtt_cls,
        mqtt_config,
        room_config,
    ):
        """Test successful integration manager initialization."""
        # Setup mocks
        mock_mqtt_publisher = AsyncMock()
        mock_mqtt_cls.return_value = mock_mqtt_publisher

        mock_prediction_publisher = Mock()
        mock_prediction_cls.return_value = mock_prediction_publisher

        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.publish_all_discovery.return_value = {
            "sensor1": Mock(success=True),
            "sensor2": Mock(success=True),
        }
        mock_discovery_cls.return_value = mock_discovery_publisher

        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        with patch.object(
            manager, "start_integration", new_callable=AsyncMock
        ) as mock_start:
            await manager.initialize()

            # Verify components were initialized
            mock_mqtt_cls.assert_called_once()
            mock_mqtt_publisher.initialize.assert_called_once()

            mock_prediction_cls.assert_called_once()
            mock_discovery_cls.assert_called_once()
            mock_discovery_publisher.publish_all_discovery.assert_called_once()
            mock_start.assert_called_once()

            assert manager.stats.initialized
            assert manager.stats.discovery_published

    @pytest.mark.asyncio
    async def test_integration_manager_initialize_disabled(
        self, mqtt_config, room_config
    ):
        """Test integration manager initialization when disabled."""
        mqtt_config.publishing_enabled = False

        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        await manager.initialize()

        assert manager.mqtt_publisher is None
        assert manager.prediction_publisher is None
        assert manager.discovery_publisher is None

    @patch("src.integration.mqtt_integration_manager.MQTTPublisher")
    @patch("src.integration.mqtt_integration_manager.PredictionPublisher")
    @pytest.mark.asyncio
    async def test_integration_manager_initialize_discovery_disabled(
        self, mock_prediction_cls, mock_mqtt_cls, mqtt_config, room_config
    ):
        """Test integration manager initialization with discovery disabled."""
        mqtt_config.discovery_enabled = False

        mock_mqtt_publisher = AsyncMock()
        mock_mqtt_cls.return_value = mock_mqtt_publisher

        mock_prediction_publisher = Mock()
        mock_prediction_cls.return_value = mock_prediction_publisher

        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        with patch.object(manager, "start_integration", new_callable=AsyncMock):
            await manager.initialize()

            assert manager.mqtt_publisher is not None
            assert manager.prediction_publisher is not None
            assert manager.discovery_publisher is None

    @pytest.mark.asyncio
    async def test_integration_manager_start_stop_lifecycle(
        self, mqtt_config, room_config
    ):
        """Test integration manager start and stop lifecycle."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        assert not manager._integration_active

        # Test start
        with patch.object(
            manager, "_system_status_publishing_loop", new_callable=AsyncMock
        ) as mock_status_loop:
            with patch("asyncio.create_task") as mock_create_task:
                mock_task = Mock()
                mock_create_task.return_value = mock_task

                await manager.start_integration()

                assert manager._integration_active
                mock_create_task.assert_called_once()
                assert len(manager._background_tasks) == 1

        # Test stop
        mock_mqtt_publisher = AsyncMock()
        manager.mqtt_publisher = mock_mqtt_publisher

        with patch("asyncio.gather", new_callable=AsyncMock) as mock_gather:
            await manager.stop_integration()

            assert not manager._integration_active
            assert manager._shutdown_event.is_set()
            mock_mqtt_publisher.stop_publisher.assert_called_once()
            mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_integration_manager_start_disabled(self, mqtt_config, room_config):
        """Test starting integration when disabled."""
        mqtt_config.publishing_enabled = False

        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        await manager.start_integration()

        assert not manager._integration_active
        assert len(manager._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_integration_manager_start_status_publishing_disabled(
        self, mqtt_config, room_config
    ):
        """Test starting integration with status publishing disabled."""
        mqtt_config.publish_system_status = False

        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        await manager.start_integration()

        assert manager._integration_active
        assert len(manager._background_tasks) == 0  # No status publishing task

    @pytest.mark.asyncio
    async def test_integration_manager_publish_prediction_success(
        self, mqtt_config, room_config, mock_prediction_result
    ):
        """Test successful prediction publishing."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        mock_prediction_publisher.publish_prediction.return_value = Mock(success=True)
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_prediction(
            mock_prediction_result, "living_room", "vacant"
        )

        assert result is True
        assert manager.stats.predictions_published == 1
        assert manager.stats.last_prediction_published is not None

        mock_prediction_publisher.publish_prediction.assert_called_once_with(
            prediction_result=mock_prediction_result,
            room_id="living_room",
            current_state="vacant",
        )

    @pytest.mark.asyncio
    async def test_integration_manager_publish_prediction_failure(
        self, mqtt_config, room_config, mock_prediction_result
    ):
        """Test failed prediction publishing."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        mock_prediction_publisher.publish_prediction.return_value = Mock(
            success=False, error_message="MQTT connection failed"
        )
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_prediction(mock_prediction_result, "living_room")

        assert result is False
        assert manager.stats.total_errors == 1
        assert "MQTT connection failed" in manager.stats.last_error

    @pytest.mark.asyncio
    async def test_integration_manager_publish_prediction_inactive(
        self, mqtt_config, room_config, mock_prediction_result
    ):
        """Test prediction publishing when integration is inactive."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)
        manager._integration_active = False

        result = await manager.publish_prediction(mock_prediction_result, "living_room")

        assert result is False
        assert manager.stats.predictions_published == 0

    @pytest.mark.asyncio
    async def test_integration_manager_publish_system_status_success(
        self, mqtt_config, room_config
    ):
        """Test successful system status publishing."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        mock_prediction_publisher.publish_system_status.return_value = Mock(
            success=True
        )
        manager.prediction_publisher = mock_prediction_publisher

        tracking_stats = {"total_predictions": 100, "accuracy": 0.95}
        model_stats = {"model_version": "1.0", "training_date": "2024-01-01"}

        result = await manager.publish_system_status(
            tracking_stats=tracking_stats,
            model_stats=model_stats,
            database_connected=True,
            active_alerts=2,
            last_error="Test error",
        )

        assert result is True
        assert manager.stats.status_updates_published == 1
        assert manager.stats.last_status_published is not None

        mock_prediction_publisher.publish_system_status.assert_called_once_with(
            tracking_stats=tracking_stats,
            model_stats=model_stats,
            database_connected=True,
            active_alerts=2,
            last_error="Test error",
        )

    @pytest.mark.asyncio
    async def test_integration_manager_refresh_discovery_success(
        self, mqtt_config, room_config
    ):
        """Test successful discovery refresh."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.refresh_discovery.return_value = {
            "sensor1": Mock(success=True),
            "sensor2": Mock(success=True),
        }
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.refresh_discovery()

        assert result is True
        assert manager.stats.discovery_published
        mock_discovery_publisher.refresh_discovery.assert_called_once()

    @pytest.mark.asyncio
    async def test_integration_manager_refresh_discovery_no_publisher(
        self, mqtt_config, room_config
    ):
        """Test discovery refresh without publisher."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)
        manager.discovery_publisher = None

        result = await manager.refresh_discovery()

        assert result is False

    def test_integration_manager_get_stats(self, mqtt_config, room_config):
        """Test getting integration statistics."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        # Set up some stats
        manager.stats.initialized = True
        manager.stats.predictions_published = 10
        manager.stats.total_errors = 2
        manager.stats.last_prediction_published = datetime.now(timezone.utc)

        # Add mock publishers with stats
        mock_mqtt_publisher = Mock()
        mock_mqtt_publisher.connection_status.connected = True
        mock_mqtt_publisher.get_publisher_stats.return_value = {"mqtt_stats": "test"}
        manager.mqtt_publisher = mock_mqtt_publisher

        mock_prediction_publisher = Mock()
        mock_prediction_publisher.get_publisher_stats.return_value = {
            "prediction_stats": "test"
        }
        manager.prediction_publisher = mock_prediction_publisher

        mock_discovery_publisher = Mock()
        mock_discovery_publisher.get_discovery_stats.return_value = {
            "published_entities_count": 5,
            "device_available": True,
            "available_services_count": 3,
            "entity_metadata_count": 5,
            "statistics": {"discovery_errors": 0},
        }
        manager.discovery_publisher = mock_discovery_publisher

        manager._integration_active = True

        stats = manager.get_integration_stats()

        assert stats["initialized"]
        assert stats["integration_active"]
        assert stats["mqtt_connected"]
        assert stats["predictions_published"] == 10
        assert stats["total_errors"] == 2
        assert stats["rooms_configured"] == 1
        assert "mqtt_publisher" in stats
        assert "prediction_publisher" in stats
        assert "discovery_publisher" in stats
        assert "discovery_insights" in stats
        assert "system_health" in stats

        # Check system health
        system_health = stats["system_health"]
        assert system_health["overall_status"] == "healthy"
        assert system_health["component_status"]["mqtt"] == "connected"
        assert system_health["component_status"]["predictions"] == "active"

    def test_integration_manager_is_connected(self, mqtt_config, room_config):
        """Test connection status checking."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        # Test disconnected
        assert not manager.is_connected()

        # Test connected
        manager._integration_active = True
        mock_mqtt_publisher = Mock()
        mock_mqtt_publisher.connection_status.connected = True
        manager.mqtt_publisher = mock_mqtt_publisher

        assert manager.is_connected()

    def test_integration_manager_notification_callbacks(self, mqtt_config, room_config):
        """Test notification callback management."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        callback1 = Mock()
        callback2 = Mock()

        # Test adding callbacks
        manager.add_notification_callback(callback1)
        manager.add_notification_callback(callback2)

        assert len(manager.notification_callbacks) == 2
        assert callback1 in manager.notification_callbacks
        assert callback2 in manager.notification_callbacks

        # Test removing callbacks
        manager.remove_notification_callback(callback1)

        assert len(manager.notification_callbacks) == 1
        assert callback1 not in manager.notification_callbacks
        assert callback2 in manager.notification_callbacks

    @pytest.mark.asyncio
    async def test_integration_manager_mqtt_connect_callback(
        self, mqtt_config, room_config
    ):
        """Test MQTT connect callback handling."""
        callback1 = AsyncMock()
        callback2 = Mock()

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_config,
            notification_callbacks=[callback1, callback2],
        )

        await manager._on_mqtt_connect(None, None, None, 0)

        assert manager.stats.mqtt_connected
        callback1.assert_called_once_with("mqtt_connected")
        callback2.assert_called_once_with("mqtt_connected")

    @pytest.mark.asyncio
    async def test_integration_manager_mqtt_disconnect_callback(
        self, mqtt_config, room_config
    ):
        """Test MQTT disconnect callback handling."""
        callback = AsyncMock()

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_config,
            notification_callbacks=[callback],
        )
        manager.stats.mqtt_connected = True

        await manager._on_mqtt_disconnect(None, None, None, 0)

        assert not manager.stats.mqtt_connected
        callback.assert_called_once_with("mqtt_disconnected")

    @pytest.mark.asyncio
    async def test_integration_manager_system_status_loop(
        self, mqtt_config, room_config
    ):
        """Test system status publishing loop."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        with patch.object(
            manager, "publish_system_status", new_callable=AsyncMock
        ) as mock_publish:
            with patch(
                "asyncio.wait_for", side_effect=asyncio.TimeoutError()
            ) as mock_wait:
                # Set shutdown event to exit loop after first iteration
                async def set_shutdown():
                    await asyncio.sleep(0.1)
                    manager._shutdown_event.set()

                asyncio.create_task(set_shutdown())

                await manager._system_status_publishing_loop()

                mock_publish.assert_called()

    def test_integration_manager_update_system_stats(self, mqtt_config, room_config):
        """Test system stats updating."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        stats = {"test": "value", "number": 42}
        manager.update_system_stats(stats)

        assert manager._last_system_stats == stats


class TestMQTTIntegrationErrorHandling:
    """Test MQTT integration error handling."""

    @pytest.mark.asyncio
    async def test_mqtt_publisher_initialization_error(self, mqtt_config):
        """Test MQTT publisher initialization error handling."""
        with patch(
            "paho.mqtt.client.Client", side_effect=Exception("Client creation failed")
        ):
            publisher = MQTTPublisher(config=mqtt_config)

            with pytest.raises(Exception) as exc_info:
                await publisher.initialize()

            assert "Failed to initialize MQTT publisher" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_integration_manager_initialization_error(
        self, mqtt_config, room_config
    ):
        """Test integration manager initialization error handling."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        with patch(
            "src.integration.mqtt_integration_manager.MQTTPublisher",
            side_effect=Exception("Publisher creation failed"),
        ):
            with pytest.raises(Exception) as exc_info:
                await manager.initialize()

            assert "Failed to initialize MQTT integration" in str(exc_info.value)
            assert manager.stats.total_errors == 1
            assert "Publisher creation failed" in manager.stats.last_error

    @pytest.mark.asyncio
    async def test_publish_prediction_exception_handling(
        self, mqtt_config, room_config, mock_prediction_result
    ):
        """Test prediction publishing exception handling."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        mock_prediction_publisher.publish_prediction.side_effect = Exception(
            "Publishing failed"
        )
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_prediction(mock_prediction_result, "living_room")

        assert result is False
        assert manager.stats.total_errors == 1
        assert "Publishing failed" in manager.stats.last_error

    def test_callback_error_handling(self, mqtt_config):
        """Test callback error handling."""
        # Test failing callback
        failing_callback = Mock(side_effect=Exception("Callback failed"))

        publisher = MQTTPublisher(
            config=mqtt_config, on_connect_callback=failing_callback
        )

        # Should not raise exception, just log error
        with patch("src.integration.mqtt_publisher.logger") as mock_logger:
            publisher._on_connect(None, None, None, 0, None)

            # Connection should still be marked as successful
            assert publisher.connection_status.connected
            # Error should be logged
            mock_logger.error.assert_called()


class TestMQTTIntegrationPerformance:
    """Test MQTT integration performance aspects."""

    @pytest.mark.asyncio
    async def test_message_queue_performance(self, mqtt_config):
        """Test message queue performance with many messages."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.max_queue_size = 100

        # Queue many messages quickly
        for i in range(150):
            await publisher.publish(f"test/topic/{i}", f"message {i}")

        # Should only keep max_queue_size messages
        assert len(publisher.message_queue) == 100

        # Should have kept the most recent messages
        first_msg = publisher.message_queue[0]
        last_msg = publisher.message_queue[-1]

        assert "50" in first_msg["topic"]  # First kept message
        assert "149" in last_msg["topic"]  # Last message

    def test_connection_status_uptime_calculation(self, mqtt_config):
        """Test uptime calculation performance."""
        publisher = MQTTPublisher(config=mqtt_config)

        base_time = datetime.utcnow()
        publisher.connection_status.connected = True
        publisher.connection_status.last_connected = base_time

        with patch("src.integration.mqtt_publisher.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = base_time.replace(
                second=base_time.second + 3600
            )

            status = publisher.get_connection_status()

            assert status.uptime_seconds == 3600.0

    @pytest.mark.asyncio
    async def test_concurrent_message_publishing(self, mqtt_config, mock_mqtt_client):
        """Test concurrent message publishing."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        # Publish messages concurrently
        tasks = []
        for i in range(10):
            task = asyncio.create_task(
                publisher.publish(f"test/topic/{i}", f"message {i}")
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result.success for result in results)
        assert mock_mqtt_client.publish.call_count == 10
        assert publisher.total_messages_published == 10


class TestMQTTIntegrationSecurityAspects:
    """Test MQTT integration security aspects."""

    @patch("paho.mqtt.client.Client")
    @pytest.mark.asyncio
    async def test_mqtt_authentication_configuration(
        self, mock_client_class, mqtt_config, mock_mqtt_client
    ):
        """Test MQTT authentication configuration."""
        mock_client_class.return_value = mock_mqtt_client

        publisher = MQTTPublisher(config=mqtt_config)

        with patch.object(publisher, "_connect_to_broker", new_callable=AsyncMock):
            with patch.object(publisher, "start_publisher", new_callable=AsyncMock):
                await publisher.initialize()

                # Verify authentication was configured
                mock_mqtt_client.username_pw_set.assert_called_once_with(
                    mqtt_config.username, mqtt_config.password
                )

    @patch("paho.mqtt.client.Client")
    @pytest.mark.asyncio
    async def test_mqtt_tls_configuration(
        self, mock_client_class, mqtt_config, mock_mqtt_client
    ):
        """Test MQTT TLS configuration."""
        mqtt_config.port = 8883  # TLS port
        mock_client_class.return_value = mock_mqtt_client

        publisher = MQTTPublisher(config=mqtt_config)

        with patch.object(publisher, "_connect_to_broker", new_callable=AsyncMock):
            with patch.object(publisher, "start_publisher", new_callable=AsyncMock):
                with patch("ssl.create_default_context") as mock_ssl:
                    mock_context = Mock()
                    mock_ssl.return_value = mock_context

                    await publisher.initialize()

                    # Verify TLS was configured
                    mock_ssl.assert_called_once_with(ssl.Purpose.SERVER_AUTH)
                    mock_mqtt_client.tls_set_context.assert_called_once_with(
                        mock_context
                    )

    def test_sensitive_data_not_logged_in_stats(self, mqtt_config):
        """Test that sensitive data is not exposed in statistics."""
        publisher = MQTTPublisher(config=mqtt_config)

        stats = publisher.get_publisher_stats()

        # Verify sensitive config data is not exposed
        config_stats = stats["config"]
        assert "username" not in config_stats
        assert "password" not in config_stats
        assert "token" not in config_stats

        # Should contain safe configuration info
        assert config_stats["broker"] == mqtt_config.broker
        assert config_stats["port"] == mqtt_config.port
        assert config_stats["publishing_enabled"] == mqtt_config.publishing_enabled


class TestMQTTIntegrationCompatibility:
    """Test MQTT integration compatibility and edge cases."""

    def test_mqtt_publisher_different_payload_types(
        self, mqtt_config, mock_mqtt_client
    ):
        """Test publishing different payload types."""
        publisher = MQTTPublisher(config=mqtt_config)
        publisher.client = mock_mqtt_client
        publisher.connection_status.connected = True

        test_cases = [
            ({"json": "data"}, json.dumps({"json": "data"}, default=str)),
            ("string data", "string data"),
            (b"bytes data", "bytes data"),
            (123, "123"),
            ([1, 2, 3], json.dumps([1, 2, 3], default=str)),
        ]

        for payload, expected in test_cases:
            mock_mqtt_client.reset_mock()

            asyncio.run(publisher.publish("test/topic", payload))

            mock_mqtt_client.publish.assert_called_once_with(
                "test/topic", expected, qos=1, retain=False
            )

    def test_mqtt_client_callback_api_versions(self, mqtt_config):
        """Test compatibility with different MQTT client callback API versions."""
        with patch("paho.mqtt.client.Client") as mock_client_class:
            mock_client = Mock(spec=mqtt_client.Client)
            mock_client_class.return_value = mock_client

            publisher = MQTTPublisher(config=mqtt_config)

            asyncio.run(publisher.initialize())

            # Verify callback API version is set
            mock_client_class.assert_called_with(
                callback_api_version=mqtt_client.CallbackAPIVersion.VERSION2,
                client_id=publisher.client_id,
                protocol=mqtt_client.MQTTv311,
                clean_session=True,
            )

    @pytest.mark.asyncio
    async def test_mqtt_publisher_graceful_shutdown_with_queued_messages(
        self, mqtt_config
    ):
        """Test graceful shutdown with queued messages."""
        publisher = MQTTPublisher(config=mqtt_config)

        # Add messages to queue
        await publisher.publish("test/topic1", "message1")
        await publisher.publish("test/topic2", "message2")

        assert len(publisher.message_queue) == 2

        with patch.object(
            publisher, "_process_message_queue", new_callable=AsyncMock
        ) as mock_process:
            await publisher.stop_publisher()

            # Should attempt to process remaining messages
            mock_process.assert_called_once()

    @pytest.mark.asyncio
    async def test_integration_manager_enhanced_service_commands(
        self, mqtt_config, room_config
    ):
        """Test enhanced service command handling."""
        manager = MQTTIntegrationManager(mqtt_config=mqtt_config, rooms=room_config)

        # Test manual retrain command
        result = await manager.handle_service_command(
            "manual_retrain", {"room_id": "living_room", "strategy": "incremental"}
        )
        assert result is True

        # Test refresh discovery command
        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.refresh_discovery.return_value = {
            "sensor1": Mock(success=True)
        }
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.handle_service_command("refresh_discovery", {})
        assert result is True

        # Test reset statistics command
        manager.stats.predictions_published = 10
        result = await manager.handle_service_command("reset_statistics", {})
        assert result is True
        assert manager.stats.predictions_published == 0

        # Test force prediction command
        result = await manager.handle_service_command(
            "force_prediction", {"room_id": "living_room"}
        )
        assert result is True

        # Test unknown command
        result = await manager.handle_service_command("unknown_command", {})
        assert result is False
