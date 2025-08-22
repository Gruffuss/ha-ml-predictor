"""
Comprehensive MQTT Integration Tests with Real Network Conditions.

This module provides complete integration testing for MQTT publishing infrastructure
with real broker connections, network failure simulation, and recovery testing.

Focus Areas:
- Real MQTT broker integration using test containers
- Connection failure and recovery scenarios  
- High-throughput message publishing under load
- Network partition and reconnection handling
- Message queuing and delivery guarantees
- Performance testing under realistic conditions
- SSL/TLS security integration testing
"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
import os
import ssl
import tempfile
import threading
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch
import uuid

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

# Test configuration for embedded broker
TEST_MQTT_CONFIG = MQTTConfig(
    broker="localhost",
    port=1883,
    username="",
    password="",
    topic_prefix="test/ha_ml_predictor",
    publishing_enabled=True,
    discovery_enabled=True,
    keepalive=60,
    connection_timeout=10,
    max_reconnect_attempts=5,
    reconnect_delay_seconds=2,
    device_identifier="test_device",
)

# Test configuration for TLS
TEST_MQTT_TLS_CONFIG = MQTTConfig(
    broker="localhost",
    port=8883,
    username="test_user",
    password="test_pass",
    topic_prefix="test/ha_ml_predictor",
    publishing_enabled=True,
    discovery_enabled=True,
    keepalive=60,
    connection_timeout=10,
    max_reconnect_attempts=3,
    reconnect_delay_seconds=1,
    device_identifier="test_device_tls",
)


@pytest.fixture
def embedded_mqtt_broker():
    """Create embedded MQTT broker for testing (using mosquitto)."""
    # For real testing, we'll use an embedded mosquitto instance
    # In CI/CD, this would be a Docker container
    broker_process = None

    try:
        # Check if mosquitto is available for testing
        import subprocess

        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".conf", delete=False) as f:
            f.write(
                """
listener 1883
allow_anonymous true
listener 8883
cafile /tmp/ca.crt
certfile /tmp/server.crt
keyfile /tmp/server.key
"""
            )
            config_file = f.name

        # Start mosquitto broker
        try:
            broker_process = subprocess.Popen(
                ["mosquitto", "-c", config_file],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )

            # Wait for broker to start
            time.sleep(1)

            yield "localhost:1883"

        except FileNotFoundError:
            # Mosquitto not available, skip real broker tests
            pytest.skip("Mosquitto not available for integration testing")

    finally:
        if broker_process:
            broker_process.terminate()
            broker_process.wait()

        # Clean up config file
        try:
            os.unlink(config_file)
        except Exception:
            pass


@pytest.fixture
def mock_mqtt_broker():
    """Mock MQTT broker for edge case testing."""

    class MockBroker:
        def __init__(self):
            self.connected_clients = []
            self.published_messages = []
            self.connection_failures = 0
            self.publish_failures = 0
            self.simulate_network_partition = False

        def add_client(self, client):
            self.connected_clients.append(client)

        def disconnect_client(self, client):
            if client in self.connected_clients:
                self.connected_clients.remove(client)

        def publish_message(self, topic, payload, qos=0, retain=False):
            if self.simulate_network_partition:
                return False

            if self.publish_failures > 0:
                self.publish_failures -= 1
                return False

            self.published_messages.append(
                {
                    "topic": topic,
                    "payload": payload,
                    "qos": qos,
                    "retain": retain,
                    "timestamp": datetime.utcnow(),
                }
            )
            return True

        def simulate_connection_failure(self, count=1):
            self.connection_failures = count

        def simulate_publish_failure(self, count=1):
            self.publish_failures = count

        def simulate_network_partition(self, enabled=True):
            self.simulate_network_partition = enabled

    return MockBroker()


class TestMQTTRealBrokerIntegration:
    """Test MQTT publisher with real broker integration."""

    @pytest.mark.integration
    async def test_real_broker_connection(self, embedded_mqtt_broker):
        """Test connection to real MQTT broker."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        try:
            await publisher.initialize()

            # Verify connection
            assert publisher.connection_status.connected
            assert publisher.client is not None

            # Test basic publish
            result = await publisher.publish(
                "test/topic",
                {"message": "test", "timestamp": datetime.utcnow().isoformat()},
            )

            assert result.success
            assert result.payload_size > 0

        finally:
            await publisher.stop_publisher()

    @pytest.mark.integration
    async def test_real_broker_reconnection_after_disconnect(
        self, embedded_mqtt_broker
    ):
        """Test automatic reconnection after broker disconnect."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        try:
            await publisher.initialize()
            assert publisher.connection_status.connected

            # Force disconnect
            publisher.client.disconnect()

            # Wait for reconnection
            await asyncio.sleep(3)

            # Verify reconnection
            assert publisher.connection_status.connected
            assert publisher.connection_status.reconnect_count > 0

        finally:
            await publisher.stop_publisher()

    @pytest.mark.integration
    async def test_message_queuing_during_disconnect(self, embedded_mqtt_broker):
        """Test message queuing when broker is disconnected."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        try:
            await publisher.initialize()

            # Disconnect client
            publisher.client.disconnect()
            await asyncio.sleep(0.5)

            # Publish messages while disconnected
            messages = []
            for i in range(5):
                result = await publisher.publish(
                    f"test/queue/{i}", {"message": f"queued_{i}", "id": i}
                )
                messages.append(result)
                assert not result.success  # Should be queued

            # Verify messages are queued
            assert len(publisher.message_queue) == 5

            # Reconnect and verify queue processing
            await publisher._connect_to_broker()
            await asyncio.sleep(2)  # Allow queue processing

            # Queue should be empty after processing
            assert len(publisher.message_queue) == 0

        finally:
            await publisher.stop_publisher()

    @pytest.mark.integration
    async def test_high_throughput_publishing(self, embedded_mqtt_broker):
        """Test high-throughput message publishing."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        try:
            await publisher.initialize()

            # Publish many messages rapidly
            start_time = time.time()
            tasks = []

            for i in range(100):
                task = publisher.publish(
                    f"test/throughput/{i % 10}",
                    {
                        "id": i,
                        "timestamp": datetime.utcnow().isoformat(),
                        "data": f"message_{i}",
                    },
                )
                tasks.append(task)

            results = await asyncio.gather(*tasks)
            end_time = time.time()

            # Verify all messages were published
            successful = sum(1 for r in results if r.success)
            assert successful == 100

            # Verify performance (should handle 100 msgs in < 5 seconds)
            duration = end_time - start_time
            assert duration < 5.0

            # Check publisher stats
            stats = publisher.get_publisher_stats()
            assert stats["messages_published"] >= 100
            assert stats["total_bytes_published"] > 0

        finally:
            await publisher.stop_publisher()

    @pytest.mark.integration
    async def test_concurrent_publishers(self, embedded_mqtt_broker):
        """Test multiple concurrent publishers."""
        publishers = []

        try:
            # Create multiple publishers
            for i in range(3):
                config = MQTTConfig(
                    broker=TEST_MQTT_CONFIG.broker,
                    port=TEST_MQTT_CONFIG.port,
                    topic_prefix=f"test/concurrent_{i}",
                    publishing_enabled=True,
                    device_identifier=f"test_device_{i}",
                    keepalive=60,
                    connection_timeout=10,
                    max_reconnect_attempts=3,
                    reconnect_delay_seconds=1,
                )

                publisher = MQTTPublisher(config, client_id=f"test_client_{i}")
                await publisher.initialize()
                publishers.append(publisher)

            # Publish from all publishers simultaneously
            tasks = []
            for i, publisher in enumerate(publishers):
                for j in range(10):
                    task = publisher.publish(
                        f"concurrent_test/{j}",
                        {
                            "publisher": i,
                            "message": j,
                            "timestamp": datetime.utcnow().isoformat(),
                        },
                    )
                    tasks.append(task)

            results = await asyncio.gather(*tasks)

            # Verify all publishes succeeded
            successful = sum(1 for r in results if r.success)
            assert successful == 30  # 3 publishers * 10 messages each

        finally:
            for publisher in publishers:
                await publisher.stop_publisher()


class TestMQTTNetworkFailureRecovery:
    """Test MQTT publisher network failure and recovery scenarios."""

    @pytest.mark.asyncio
    async def test_connection_timeout_handling(self):
        """Test connection timeout handling."""
        # Use invalid broker to trigger timeout
        config = MQTTConfig(
            broker="192.0.2.1",  # Invalid IP (RFC3330)
            port=1883,
            topic_prefix="test",
            publishing_enabled=True,
            connection_timeout=2,  # Short timeout
            max_reconnect_attempts=2,
            reconnect_delay_seconds=1,
            device_identifier="test_timeout",
        )

        publisher = MQTTPublisher(config)

        start_time = time.time()
        with pytest.raises(MQTTPublisherError):
            await publisher.initialize()

        duration = time.time() - start_time

        # Should timeout quickly
        assert duration < 10
        assert publisher.connection_status.connection_attempts == 2

    @pytest.mark.asyncio
    async def test_reconnection_backoff(self):
        """Test exponential backoff during reconnection attempts."""
        config = MQTTConfig(
            broker="192.0.2.1",  # Invalid IP
            port=1883,
            topic_prefix="test",
            publishing_enabled=True,
            connection_timeout=1,
            max_reconnect_attempts=3,
            reconnect_delay_seconds=1,
            device_identifier="test_backoff",
        )

        publisher = MQTTPublisher(config)

        start_time = time.time()
        with pytest.raises(MQTTPublisherError):
            await publisher.initialize()

        duration = time.time() - start_time

        # Should respect delay between attempts
        assert duration >= 3  # 3 attempts with 1s delay each
        assert publisher.connection_status.connection_attempts == 3

    @pytest.mark.asyncio
    async def test_message_queue_overflow_handling(self):
        """Test message queue overflow handling."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)
        publisher.max_queue_size = 5  # Small queue for testing

        # Don't initialize to keep client disconnected

        # Fill queue beyond capacity
        for i in range(10):
            result = await publisher.publish(f"test/overflow/{i}", {"id": i})
            assert not result.success

        # Queue should not exceed max size
        assert len(publisher.message_queue) == 5

        # Verify oldest messages were discarded
        queue_ids = [msg["payload"] for msg in publisher.message_queue]
        queue_data = [json.loads(payload) for payload in queue_ids]
        message_ids = [data["id"] for data in queue_data]

        # Should contain messages 5-9 (newest)
        assert message_ids == [5, 6, 7, 8, 9]

    @pytest.mark.asyncio
    async def test_ssl_connection_failure(self):
        """Test SSL connection failure handling."""
        config = MQTTConfig(
            broker="localhost",
            port=8883,  # SSL port
            username="test",
            password="test",
            topic_prefix="test",
            publishing_enabled=True,
            connection_timeout=5,
            max_reconnect_attempts=2,
            reconnect_delay_seconds=1,
            device_identifier="test_ssl",
        )

        publisher = MQTTPublisher(config)

        # Should fail due to SSL certificate issues
        with pytest.raises(MQTTPublisherError):
            await publisher.initialize()

        assert not publisher.connection_status.connected
        assert (
            "ssl" in publisher.connection_status.last_error.lower()
            or "certificate" in publisher.connection_status.last_error.lower()
            or "connection" in publisher.connection_status.last_error.lower()
        )


class TestMQTTMessageDeliveryGuarantees:
    """Test MQTT message delivery guarantees and QoS levels."""

    @pytest.mark.asyncio
    async def test_qos_level_handling(self):
        """Test different QoS levels."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock successful connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            # Mock MQTT client
            mock_client = Mock()
            mock_publish_info = Mock()
            mock_publish_info.rc = mqtt_client.MQTT_ERR_SUCCESS
            mock_publish_info.mid = 12345
            mock_client.publish.return_value = mock_publish_info
            publisher.client = mock_client

            # Test QoS 0
            result = await publisher.publish("test/qos0", {"test": "data"}, qos=0)
            assert result.success
            mock_client.publish.assert_called_with(
                "test/qos0", '{"test": "data"}', qos=0, retain=False
            )

            # Test QoS 1
            result = await publisher.publish("test/qos1", {"test": "data"}, qos=1)
            assert result.success
            mock_client.publish.assert_called_with(
                "test/qos1", '{"test": "data"}', qos=1, retain=False
            )

            # Test QoS 2
            result = await publisher.publish("test/qos2", {"test": "data"}, qos=2)
            assert result.success
            mock_client.publish.assert_called_with(
                "test/qos2", '{"test": "data"}', qos=2, retain=False
            )

    @pytest.mark.asyncio
    async def test_retained_message_publishing(self):
        """Test retained message publishing."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock successful connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            # Mock MQTT client
            mock_client = Mock()
            mock_publish_info = Mock()
            mock_publish_info.rc = mqtt_client.MQTT_ERR_SUCCESS
            mock_publish_info.mid = 12345
            mock_client.publish.return_value = mock_publish_info
            publisher.client = mock_client

            # Test retained message
            result = await publisher.publish(
                "test/retained", {"status": "online"}, retain=True
            )

            assert result.success
            mock_client.publish.assert_called_with(
                "test/retained", '{"status": "online"}', qos=1, retain=True
            )

    @pytest.mark.asyncio
    async def test_publish_error_codes(self):
        """Test handling of different MQTT publish error codes."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock successful connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            # Mock MQTT client with different error codes
            mock_client = Mock()
            publisher.client = mock_client

            error_codes = [
                mqtt_client.MQTT_ERR_NO_CONN,
                mqtt_client.MQTT_ERR_QUEUE_SIZE,
                mqtt_client.MQTT_ERR_PAYLOAD_SIZE,
                mqtt_client.MQTT_ERR_NOT_SUPPORTED,
                mqtt_client.MQTT_ERR_NOT_FOUND,
                mqtt_client.MQTT_ERR_CONN_LOST,
                mqtt_client.MQTT_ERR_PROTOCOL,
                mqtt_client.MQTT_ERR_ERRNO,
            ]

            for error_code in error_codes:
                mock_publish_info = Mock()
                mock_publish_info.rc = error_code
                mock_client.publish.return_value = mock_publish_info

                result = await publisher.publish("test/error", {"test": "data"})

                assert not result.success
                assert "return code" in result.error_message
                assert str(error_code) in result.error_message


class TestMQTTPerformanceAndReliability:
    """Test MQTT publisher performance and reliability under stress."""

    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self):
        """Test memory usage during high-load publishing."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock successful connection and publishing
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            mock_client = Mock()
            mock_publish_info = Mock()
            mock_publish_info.rc = mqtt_client.MQTT_ERR_SUCCESS
            mock_publish_info.mid = 12345
            mock_client.publish.return_value = mock_publish_info
            publisher.client = mock_client

            # Publish large number of messages
            import os

            import psutil

            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss

            # Publish 1000 messages
            for i in range(1000):
                large_payload = {
                    "id": i,
                    "timestamp": datetime.utcnow().isoformat(),
                    "data": "x" * 1000,  # 1KB of data
                    "metadata": {
                        "room": f"room_{i % 10}",
                        "sensor_data": list(range(100)),
                    },
                }

                result = await publisher.publish(f"test/load/{i}", large_payload)
                assert result.success

            final_memory = process.memory_info().rss
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (< 50MB)
            assert memory_increase < 50 * 1024 * 1024

    @pytest.mark.asyncio
    async def test_concurrent_publish_thread_safety(self):
        """Test thread safety of concurrent publishing."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock successful connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            mock_client = Mock()
            mock_publish_info = Mock()
            mock_publish_info.rc = mqtt_client.MQTT_ERR_SUCCESS
            mock_publish_info.mid = 12345
            mock_client.publish.return_value = mock_publish_info
            publisher.client = mock_client

            # Create multiple concurrent tasks
            async def publish_batch(batch_id: int, count: int):
                results = []
                for i in range(count):
                    result = await publisher.publish(
                        f"test/concurrent/{batch_id}/{i}",
                        {"batch": batch_id, "message": i},
                    )
                    results.append(result)
                return results

            # Run 10 concurrent batches of 50 messages each
            tasks = [publish_batch(i, 50) for i in range(10)]
            batch_results = await asyncio.gather(*tasks)

            # Verify all messages were published successfully
            total_successful = 0
            for batch in batch_results:
                successful_in_batch = sum(1 for r in batch if r.success)
                total_successful += successful_in_batch

            assert total_successful == 500  # 10 batches * 50 messages

            # Verify statistics are consistent
            stats = publisher.get_publisher_stats()
            assert stats["messages_published"] == 500

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_pending_messages(self):
        """Test graceful shutdown when messages are pending."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock connection but delay publishing
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            mock_client = Mock()

            # Simulate slow publishing
            def slow_publish(*args, **kwargs):
                time.sleep(0.1)  # 100ms delay
                mock_info = Mock()
                mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
                mock_info.mid = 12345
                return mock_info

            mock_client.publish.side_effect = slow_publish
            publisher.client = mock_client

            await publisher.start_publisher()

            # Queue several messages
            publish_tasks = []
            for i in range(10):
                task = asyncio.create_task(
                    publisher.publish(f"test/shutdown/{i}", {"id": i})
                )
                publish_tasks.append(task)

            # Allow some messages to start processing
            await asyncio.sleep(0.05)

            # Shutdown publisher
            shutdown_start = time.time()
            await publisher.stop_publisher()
            shutdown_duration = time.time() - shutdown_start

            # Shutdown should complete within reasonable time
            assert shutdown_duration < 15  # Should not hang

            # Some messages should have been processed
            stats = publisher.get_publisher_stats()
            assert stats["messages_published"] > 0

    @pytest.mark.asyncio
    async def test_connection_monitoring_loop_resilience(self):
        """Test connection monitoring loop resilience to errors."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock connection that fails intermittently
        connection_attempts = 0

        async def failing_connect():
            nonlocal connection_attempts
            connection_attempts += 1
            if connection_attempts <= 2:
                raise ConnectionError(f"Connection failed {connection_attempts}")
            publisher.connection_status.connected = True

        with patch.object(publisher, "_connect_to_broker", side_effect=failing_connect):
            await publisher.start_publisher()

            # Wait for connection monitoring to retry
            await asyncio.sleep(6)  # Should retry multiple times

            # Eventually should succeed
            assert publisher.connection_status.connected
            assert connection_attempts >= 3

        await publisher.stop_publisher()

    @pytest.mark.asyncio
    async def test_message_processing_loop_error_handling(self):
        """Test message processing loop error handling."""
        publisher = MQTTPublisher(TEST_MQTT_CONFIG)

        # Mock connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None
            publisher.connection_status.connected = True

            # Mock client that fails on first publish
            mock_client = Mock()
            call_count = 0

            def publish_with_failure(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    raise Exception("Simulated publish error")

                mock_info = Mock()
                mock_info.rc = mqtt_client.MQTT_ERR_SUCCESS
                mock_info.mid = 12345
                return mock_info

            mock_client.publish.side_effect = publish_with_failure
            publisher.client = mock_client

            await publisher.start_publisher()

            # Add messages to queue
            publisher.message_queue.extend(
                [
                    {
                        "topic": "test/error/1",
                        "payload": '{"test": 1}',
                        "qos": 1,
                        "retain": False,
                        "queued_at": datetime.utcnow(),
                    },
                    {
                        "topic": "test/error/2",
                        "payload": '{"test": 2}',
                        "qos": 1,
                        "retain": False,
                        "queued_at": datetime.utcnow(),
                    },
                ]
            )

            # Wait for processing
            await asyncio.sleep(2)

            # Should handle error gracefully and continue processing
            assert len(publisher.message_queue) == 0  # Queue should be processed

        await publisher.stop_publisher()


class TestMQTTConfigurationVariations:
    """Test MQTT publisher with different configuration variations."""

    @pytest.mark.asyncio
    async def test_disabled_publishing_configuration(self):
        """Test publisher with publishing disabled."""
        config = MQTTConfig(
            broker="localhost",
            port=1883,
            topic_prefix="test",
            publishing_enabled=False,  # Disabled
            device_identifier="test_disabled",
        )

        publisher = MQTTPublisher(config)

        # Should not initialize when disabled
        await publisher.initialize()
        assert publisher.client is None
        assert not publisher._publisher_active

    @pytest.mark.asyncio
    async def test_authentication_configuration(self):
        """Test publisher with authentication."""
        config = MQTTConfig(
            broker="localhost",
            port=1883,
            username="test_user",
            password="test_password",
            topic_prefix="test",
            publishing_enabled=True,
            device_identifier="test_auth",
        )

        publisher = MQTTPublisher(config)

        # Mock successful connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None

            # Mock client creation
            with patch(
                "src.integration.mqtt_publisher.mqtt_client.Client"
            ) as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                await publisher.initialize()

                # Verify authentication was configured
                mock_client.username_pw_set.assert_called_once_with(
                    "test_user", "test_password"
                )

    @pytest.mark.asyncio
    async def test_ssl_configuration(self):
        """Test publisher with SSL configuration."""
        config = MQTTConfig(
            broker="localhost",
            port=8883,  # SSL port
            topic_prefix="test",
            publishing_enabled=True,
            device_identifier="test_ssl_config",
        )

        publisher = MQTTPublisher(config)

        # Mock successful connection
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None

            # Mock SSL context creation
            with patch("ssl.create_default_context") as mock_ssl:
                mock_context = Mock()
                mock_ssl.return_value = mock_context

                # Mock client creation
                with patch(
                    "src.integration.mqtt_publisher.mqtt_client.Client"
                ) as mock_client_class:
                    mock_client = Mock()
                    mock_client_class.return_value = mock_client

                    await publisher.initialize()

                    # Verify SSL was configured
                    mock_ssl.assert_called_once_with(ssl.Purpose.SERVER_AUTH)
                    mock_client.tls_set_context.assert_called_once_with(mock_context)

    @pytest.mark.asyncio
    async def test_custom_callback_configuration(self):
        """Test publisher with custom callbacks."""
        callback_calls = {"connect": [], "disconnect": [], "message": []}

        async def on_connect_callback(client, userdata, flags, rc):
            callback_calls["connect"].append({"rc": rc, "time": datetime.utcnow()})

        async def on_disconnect_callback(client, userdata, flags, rc):
            callback_calls["disconnect"].append({"rc": rc, "time": datetime.utcnow()})

        async def on_message_callback(client, userdata, message):
            callback_calls["message"].append(
                {"topic": message.topic, "time": datetime.utcnow()}
            )

        publisher = MQTTPublisher(
            TEST_MQTT_CONFIG,
            on_connect_callback=on_connect_callback,
            on_disconnect_callback=on_disconnect_callback,
            on_message_callback=on_message_callback,
        )

        # Mock client and trigger callbacks
        with patch.object(publisher, "_connect_to_broker") as mock_connect:
            mock_connect.return_value = None

            with patch(
                "src.integration.mqtt_publisher.mqtt_client.Client"
            ) as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client

                await publisher.initialize()

                # Trigger connect callback
                publisher._on_connect(mock_client, None, {}, 0, None)
                await asyncio.sleep(0.1)  # Allow callback to execute

                # Verify callback was called
                assert len(callback_calls["connect"]) == 1
                assert callback_calls["connect"][0]["rc"] == 0
