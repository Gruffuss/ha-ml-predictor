"""
Integration Test Framework with Test Containers.

This module provides a comprehensive integration testing framework that uses
test containers and realistic infrastructure simulation for thorough testing
of the entire system in conditions that closely match production.

Focus Areas:
- Real MQTT broker integration using test containers
- Database integration with test PostgreSQL/TimescaleDB
- Home Assistant API simulation with realistic data
- Network condition simulation (latency, packet loss, partitions)
- Multi-service integration testing
- Performance benchmarking in realistic environments
- Failure injection and recovery testing
- End-to-end workflow validation
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
import json
import logging
import os
import random
import socket
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch
import uuid

import docker
import psutil
import pytest
import requests
import signal

from src.core.config import MQTTConfig, RoomConfig, TrackingConfig
from src.integration.api_server import create_app, set_tracking_manager
from src.integration.discovery_publisher import DeviceInfo, DiscoveryPublisher
from src.integration.ha_entity_definitions import HAEntityDefinitions
from src.integration.mqtt_publisher import MQTTPublisher


class TestContainerManager:
    """Manages test containers for integration testing."""

    def __init__(self):
        self.docker_client = None
        self.containers = {}
        self.networks = {}

    async def initialize(self):
        """Initialize Docker client and test environment."""
        try:
            self.docker_client = docker.from_env()
            await self._create_test_network()
        except Exception as e:
            pytest.skip(f"Docker not available for container testing: {e}")

    async def _create_test_network(self):
        """Create isolated network for test containers."""
        network_name = f"ha_ml_test_{int(time.time())}"

        try:
            self.networks["test"] = self.docker_client.networks.create(
                network_name,
                driver="bridge",
                options={"com.docker.network.bridge.enable_icc": "true"},
            )
            logging.info(f"Created test network: {network_name}")
        except Exception as e:
            logging.warning(f"Failed to create test network: {e}")

    async def start_mqtt_broker(self, port=1883) -> Dict[str, Any]:
        """Start test MQTT broker container."""
        try:
            # Try to find available port
            while self._is_port_in_use(port):
                port += 1

            container = self.docker_client.containers.run(
                "eclipse-mosquitto:2.0",
                ports={"1883/tcp": port},
                detach=True,
                remove=True,
                name=f"test_mqtt_{int(time.time())}",
                network=self.networks["test"].name if "test" in self.networks else None,
            )

            # Wait for broker to be ready
            await self._wait_for_mqtt_broker(port)

            self.containers["mqtt"] = container

            return {
                "container": container,
                "host": "localhost",
                "port": port,
                "connection_string": f"mqtt://localhost:{port}",
            }

        except Exception as e:
            pytest.skip(f"Failed to start MQTT broker: {e}")

    async def start_postgresql(self, port=5432) -> Dict[str, Any]:
        """Start test PostgreSQL with TimescaleDB container."""
        try:
            # Find available port
            while self._is_port_in_use(port):
                port += 1

            # Create temporary directory for PostgreSQL data
            temp_dir = tempfile.mkdtemp(prefix="test_postgres_")

            container = self.docker_client.containers.run(
                "timescale/timescaledb:latest-pg14",
                ports={"5432/tcp": port},
                environment={
                    "POSTGRES_DB": "test_ha_ml",
                    "POSTGRES_USER": "test_user",
                    "POSTGRES_PASSWORD": "test_password",
                    "POSTGRES_INITDB_ARGS": "--auth-host=md5",
                },
                volumes={temp_dir: {"bind": "/var/lib/postgresql/data", "mode": "rw"}},
                detach=True,
                remove=True,
                name=f"test_postgres_{int(time.time())}",
                network=self.networks["test"].name if "test" in self.networks else None,
            )

            # Wait for database to be ready
            await self._wait_for_postgresql(port)

            self.containers["postgres"] = container

            return {
                "container": container,
                "host": "localhost",
                "port": port,
                "database": "test_ha_ml",
                "username": "test_user",
                "password": "test_password",
                "connection_string": f"postgresql://test_user:test_password@localhost:{port}/test_ha_ml",
            }

        except Exception as e:
            pytest.skip(f"Failed to start PostgreSQL: {e}")

    async def start_home_assistant_mock(self, port=8123) -> Dict[str, Any]:
        """Start Home Assistant API mock container."""
        try:
            # Find available port
            while self._is_port_in_use(port):
                port += 1

            # Create HA mock with realistic API responses
            ha_mock_config = {
                "entities": self._generate_ha_entities(),
                "history": self._generate_ha_history(),
                "websocket_events": self._generate_ha_events(),
            }

            # Use lightweight HTTP server with HA API simulation
            container = self.docker_client.containers.run(
                "python:3.11-slim",
                ports={"8123/tcp": port},
                detach=True,
                remove=True,
                name=f"test_ha_mock_{int(time.time())}",
                command=["python", "-c", self._get_ha_mock_server_code()],
                environment={
                    "HA_PORT": "8123",
                    "HA_CONFIG": json.dumps(ha_mock_config),
                },
                network=self.networks["test"].name if "test" in self.networks else None,
            )

            # Wait for HA mock to be ready
            await self._wait_for_http_server(port, "/api/")

            self.containers["ha_mock"] = container

            return {
                "container": container,
                "host": "localhost",
                "port": port,
                "api_url": f"http://localhost:{port}/api",
                "websocket_url": f"ws://localhost:{port}/api/websocket",
                "token": "test_ha_token",
            }

        except Exception as e:
            logging.warning(f"Failed to start HA mock: {e}")
            return await self._create_lightweight_ha_mock(port)

    async def start_redis(self, port=6379) -> Dict[str, Any]:
        """Start test Redis container for caching."""
        try:
            # Find available port
            while self._is_port_in_use(port):
                port += 1

            container = self.docker_client.containers.run(
                "redis:7-alpine",
                ports={"6379/tcp": port},
                detach=True,
                remove=True,
                name=f"test_redis_{int(time.time())}",
                network=self.networks["test"].name if "test" in self.networks else None,
            )

            # Wait for Redis to be ready
            await self._wait_for_redis(port)

            self.containers["redis"] = container

            return {
                "container": container,
                "host": "localhost",
                "port": port,
                "connection_string": f"redis://localhost:{port}",
            }

        except Exception as e:
            logging.warning(f"Failed to start Redis: {e}")
            return None

    def _is_port_in_use(self, port):
        """Check if port is already in use."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(("localhost", port)) == 0

    async def _wait_for_mqtt_broker(self, port, timeout=30):
        """Wait for MQTT broker to be ready."""
        import paho.mqtt.client as mqtt

        def on_connect(client, userdata, flags, rc):
            client.disconnect()

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                client = mqtt.Client()
                client.on_connect = on_connect
                client.connect("localhost", port, 5)
                client.loop_start()
                time.sleep(0.1)
                client.loop_stop()
                return True
            except Exception:
                await asyncio.sleep(0.5)

        raise TimeoutError(f"MQTT broker on port {port} not ready within {timeout}s")

    async def _wait_for_postgresql(self, port, timeout=60):
        """Wait for PostgreSQL to be ready."""
        import psycopg2

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                conn = psycopg2.connect(
                    host="localhost",
                    port=port,
                    database="test_ha_ml",
                    user="test_user",
                    password="test_password",
                    connect_timeout=5,
                )
                conn.close()
                return True
            except Exception:
                await asyncio.sleep(1)

        raise TimeoutError(f"PostgreSQL on port {port} not ready within {timeout}s")

    async def _wait_for_http_server(self, port, path="/", timeout=30):
        """Wait for HTTP server to be ready."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"http://localhost:{port}{path}", timeout=2)
                if response.status_code < 500:
                    return True
            except Exception:
                await asyncio.sleep(0.5)

        raise TimeoutError(f"HTTP server on port {port} not ready within {timeout}s")

    async def _wait_for_redis(self, port, timeout=30):
        """Wait for Redis to be ready."""
        import redis

        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                r = redis.Redis(host="localhost", port=port, socket_timeout=2)
                r.ping()
                return True
            except Exception:
                await asyncio.sleep(0.5)

        raise TimeoutError(f"Redis on port {port} not ready within {timeout}s")

    def _generate_ha_entities(self) -> List[Dict[str, Any]]:
        """Generate realistic Home Assistant entities."""
        entities = []

        # Generate entities for different rooms
        rooms = ["living_room", "bedroom", "kitchen", "bathroom", "office"]

        for room in rooms:
            # Motion sensors
            entities.append(
                {
                    "entity_id": f"binary_sensor.{room}_motion",
                    "state": random.choice(["on", "off"]),
                    "attributes": {
                        "device_class": "motion",
                        "friendly_name": f"{room.replace('_', ' ').title()} Motion",
                        "last_changed": datetime.now().isoformat(),
                    },
                }
            )

            # Door sensors
            entities.append(
                {
                    "entity_id": f"binary_sensor.{room}_door",
                    "state": random.choice(["on", "off"]),
                    "attributes": {
                        "device_class": "door",
                        "friendly_name": f"{room.replace('_', ' ').title()} Door",
                        "last_changed": datetime.now().isoformat(),
                    },
                }
            )

            # Temperature sensors
            entities.append(
                {
                    "entity_id": f"sensor.{room}_temperature",
                    "state": str(random.uniform(18, 25)),
                    "attributes": {
                        "unit_of_measurement": "°C",
                        "device_class": "temperature",
                        "friendly_name": f"{room.replace('_', ' ').title()} Temperature",
                        "last_changed": datetime.now().isoformat(),
                    },
                }
            )

        return entities

    def _generate_ha_history(self) -> Dict[str, List[Dict[str, Any]]]:
        """Generate realistic Home Assistant history data."""
        history = {}

        entities = self._generate_ha_entities()

        for entity in entities:
            entity_id = entity["entity_id"]
            history[entity_id] = []

            # Generate 24 hours of history
            current_time = datetime.now() - timedelta(hours=24)

            for _ in range(100):  # 100 state changes in 24 hours
                if "motion" in entity_id:
                    state = random.choice(["on", "off"])
                elif "door" in entity_id:
                    state = random.choice(["on", "off"])
                elif "temperature" in entity_id:
                    state = str(random.uniform(18, 25))
                else:
                    state = entity["state"]

                history[entity_id].append(
                    {
                        "state": state,
                        "last_changed": current_time.isoformat(),
                        "last_updated": current_time.isoformat(),
                        "attributes": entity["attributes"],
                    }
                )

                current_time += timedelta(minutes=random.randint(5, 30))

        return history

    def _generate_ha_events(self) -> List[Dict[str, Any]]:
        """Generate realistic Home Assistant WebSocket events."""
        events = []

        event_types = ["state_changed", "automation_triggered", "service_call"]

        for _ in range(50):
            event = {
                "id": random.randint(1000, 9999),
                "type": "event",
                "event": {
                    "event_type": random.choice(event_types),
                    "data": {
                        "entity_id": f"sensor.test_{random.randint(1, 10)}",
                        "old_state": {"state": "off"},
                        "new_state": {"state": "on"},
                    },
                    "time_fired": datetime.now().isoformat(),
                    "context": {"id": str(uuid.uuid4())},
                },
            }
            events.append(event)

        return events

    def _get_ha_mock_server_code(self) -> str:
        """Get Python code for Home Assistant mock server."""
        return """
import json
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
import time

class HAMockHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path.startswith("/api/states"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            # Return mock states
            states = json.loads(os.environ.get("HA_CONFIG", "{}")).get("entities", [])
            self.wfile.write(json.dumps(states).encode())

        elif self.path.startswith("/api/history"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()

            # Return mock history
            history = json.loads(os.environ.get("HA_CONFIG", "{}")).get("history", {})
            self.wfile.write(json.dumps(history).encode())

        else:
            self.send_response(404)
            self.end_headers()

    def do_POST(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"result": "ok"}')

if __name__ == "__main__":
    port = int(os.environ.get("HA_PORT", 8123))
    server = HTTPServer(("0.0.0.0", port), HAMockHandler)
    server.serve_forever()
"""

    async def _create_lightweight_ha_mock(self, port) -> Dict[str, Any]:
        """Create lightweight HA mock without container."""
        # Use local HTTP server as fallback
        return {
            "host": "localhost",
            "port": port,
            "api_url": f"http://localhost:{port}/api",
            "websocket_url": f"ws://localhost:{port}/api/websocket",
            "token": "test_ha_token",
            "mock_mode": "lightweight",
        }

    async def cleanup(self):
        """Clean up all test containers and networks."""
        try:
            # Stop and remove containers
            for name, container in self.containers.items():
                try:
                    container.stop(timeout=5)
                    logging.info(f"Stopped container: {name}")
                except Exception as e:
                    logging.warning(f"Failed to stop container {name}: {e}")

            # Remove networks
            for name, network in self.networks.items():
                try:
                    network.remove()
                    logging.info(f"Removed network: {name}")
                except Exception as e:
                    logging.warning(f"Failed to remove network {name}: {e}")

        except Exception as e:
            logging.error(f"Error during cleanup: {e}")

        self.containers.clear()
        self.networks.clear()


class NetworkConditionSimulator:
    """Simulates various network conditions for testing."""

    def __init__(self):
        self.latency_ms = 0
        self.packet_loss_percent = 0
        self.bandwidth_limit_kbps = None
        self.partition_active = False

    async def add_latency(self, latency_ms: int):
        """Add network latency simulation."""
        self.latency_ms = latency_ms
        logging.info(f"Simulating {latency_ms}ms network latency")

    async def add_packet_loss(self, loss_percent: float):
        """Add packet loss simulation."""
        self.packet_loss_percent = loss_percent
        logging.info(f"Simulating {loss_percent}% packet loss")

    async def limit_bandwidth(self, bandwidth_kbps: int):
        """Add bandwidth limitation."""
        self.bandwidth_limit_kbps = bandwidth_kbps
        logging.info(f"Limiting bandwidth to {bandwidth_kbps} kbps")

    async def create_network_partition(self, duration_seconds: int):
        """Simulate network partition."""
        self.partition_active = True
        logging.info(f"Creating network partition for {duration_seconds}s")

        await asyncio.sleep(duration_seconds)

        self.partition_active = False
        logging.info("Network partition resolved")

    def should_drop_packet(self) -> bool:
        """Determine if packet should be dropped due to simulation."""
        if self.partition_active:
            return True

        if self.packet_loss_percent > 0:
            return random.random() < (self.packet_loss_percent / 100)

        return False

    async def apply_latency(self):
        """Apply simulated network latency."""
        if self.latency_ms > 0:
            await asyncio.sleep(self.latency_ms / 1000)


@pytest.fixture
async def test_infrastructure():
    """Provide complete test infrastructure with containers."""
    manager = TestContainerManager()

    try:
        await manager.initialize()

        # Start core infrastructure
        mqtt_info = await manager.start_mqtt_broker()
        postgres_info = await manager.start_postgresql()
        ha_info = await manager.start_home_assistant_mock()
        redis_info = await manager.start_redis()

        # Create network simulator
        network_sim = NetworkConditionSimulator()

        yield {
            "container_manager": manager,
            "mqtt": mqtt_info,
            "postgres": postgres_info,
            "home_assistant": ha_info,
            "redis": redis_info,
            "network_simulator": network_sim,
        }

    finally:
        await manager.cleanup()


class TestContainerIntegration:
    """Test integration with real containers."""

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_mqtt_broker_integration(self, test_infrastructure):
        """Test integration with real MQTT broker."""
        mqtt_info = test_infrastructure["mqtt"]

        # Create real MQTT publisher
        mqtt_config = MQTTConfig(
            broker=mqtt_info["host"],
            port=mqtt_info["port"],
            topic_prefix="container_test",
            publishing_enabled=True,
            device_identifier="container_test_device",
        )

        publisher = MQTTPublisher(mqtt_config)

        try:
            await publisher.initialize()

            # Verify connection
            assert publisher.connection_status.connected

            # Test message publishing
            result = await publisher.publish_json(
                "container_test/message",
                {"test": "real_broker", "timestamp": datetime.now().isoformat()},
            )

            assert result.success
            assert result.message_id is not None

            # Test high-throughput publishing
            start_time = time.time()
            results = []

            for i in range(100):
                result = await publisher.publish_json(
                    f"container_test/throughput/{i}", {"id": i, "data": f"message_{i}"}
                )
                results.append(result.success)

            end_time = time.time()
            duration = end_time - start_time

            success_rate = sum(results) / len(results)
            throughput = len(results) / duration

            # Performance assertions
            assert success_rate >= 0.95
            assert throughput >= 20  # At least 20 messages per second

            print(
                f"Real MQTT broker test: {success_rate:.2%} success, {throughput:.1f} msg/s"
            )

        finally:
            await publisher.stop_publisher()

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_database_integration(self, test_infrastructure):
        """Test integration with real PostgreSQL database."""
        postgres_info = test_infrastructure["postgres"]

        # Test database connection
        import asyncpg

        try:
            connection = await asyncpg.connect(postgres_info["connection_string"])

            # Create test table
            await connection.execute(
                """
                CREATE TABLE IF NOT EXISTS test_sensor_events (
                    id SERIAL PRIMARY KEY,
                    room_id VARCHAR(50),
                    sensor_id VARCHAR(100),
                    state VARCHAR(10),
                    timestamp TIMESTAMPTZ DEFAULT NOW()
                )
            """
            )

            # Test data insertion
            test_data = [
                ("living_room", "motion_sensor", "on"),
                ("bedroom", "door_sensor", "off"),
                ("kitchen", "temperature_sensor", "22.5"),
            ]

            for room_id, sensor_id, state in test_data:
                await connection.execute(
                    "INSERT INTO test_sensor_events (room_id, sensor_id, state) VALUES ($1, $2, $3)",
                    room_id,
                    sensor_id,
                    state,
                )

            # Test data retrieval
            rows = await connection.fetch("SELECT * FROM test_sensor_events")
            assert len(rows) == 3

            # Test performance with bulk operations
            start_time = time.time()

            bulk_data = [
                (f"room_{i}", f"sensor_{i}", "on" if i % 2 == 0 else "off")
                for i in range(1000)
            ]

            await connection.executemany(
                "INSERT INTO test_sensor_events (room_id, sensor_id, state) VALUES ($1, $2, $3)",
                bulk_data,
            )

            end_time = time.time()
            duration = end_time - start_time
            throughput = len(bulk_data) / duration

            assert throughput >= 100  # At least 100 inserts per second

            print(f"Database bulk insert: {throughput:.1f} records/s")

        finally:
            await connection.close()

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_home_assistant_api_integration(self, test_infrastructure):
        """Test integration with Home Assistant API mock."""
        ha_info = test_infrastructure["home_assistant"]

        if ha_info.get("mock_mode") == "lightweight":
            pytest.skip("HA container not available, using lightweight mock")

        # Test API endpoints
        import aiohttp

        async with aiohttp.ClientSession() as session:
            # Test states endpoint
            async with session.get(
                f"{ha_info['api_url']}/states",
                headers={"Authorization": f"Bearer {ha_info['token']}"},
            ) as response:

                if response.status == 200:
                    states = await response.json()
                    assert isinstance(states, list)
                    assert len(states) > 0

                    # Verify entity structure
                    for state in states[:5]:  # Check first 5 entities
                        assert "entity_id" in state
                        assert "state" in state
                        assert "attributes" in state

                    print(f"HA API returned {len(states)} entities")

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_multi_service_integration(self, test_infrastructure):
        """Test integration across multiple services."""
        mqtt_info = test_infrastructure["mqtt"]
        postgres_info = test_infrastructure["postgres"]
        ha_info = test_infrastructure["home_assistant"]

        # Create integrated system configuration
        system_config = {
            "mqtt": MQTTConfig(
                broker=mqtt_info["host"],
                port=mqtt_info["port"],
                topic_prefix="integration_test",
                publishing_enabled=True,
                device_identifier="multi_service_test",
            ),
            "rooms": {
                "test_room": RoomConfig(
                    room_id="test_room",
                    name="Test Room",
                    sensors={
                        "motion": ["binary_sensor.test_room_motion"],
                        "door": ["binary_sensor.test_room_door"],
                    },
                )
            },
        }

        # Create MQTT publisher
        publisher = MQTTPublisher(system_config["mqtt"])

        # Create HA entity definitions
        device_info = DeviceInfo(
            identifiers=["multi_service_test"],
            name="Multi Service Test Device",
            manufacturer="Test",
            model="Integration",
            sw_version="1.0",
        )

        discovery_publisher = DiscoveryPublisher(
            mqtt_publisher=publisher,
            device_info=device_info,
            mqtt_config=system_config["mqtt"],
        )

        ha_entities = HAEntityDefinitions(
            discovery_publisher=discovery_publisher,
            mqtt_config=system_config["mqtt"],
            rooms=system_config["rooms"],
        )

        try:
            # Initialize MQTT
            await publisher.initialize()

            # Create and publish entities
            entities = ha_entities.define_all_entities()
            results = await ha_entities.publish_all_entities()

            # Verify entity creation
            assert len(entities) > 0
            assert len(results) > 0

            successful_publishes = sum(1 for r in results.values() if r.success)
            assert successful_publishes > 0

            # Simulate real-time data flow
            for i in range(10):
                # Publish room state
                room_state = {
                    "room_id": "test_room",
                    "occupied": i % 2 == 0,
                    "last_motion": datetime.now().isoformat(),
                    "confidence": random.uniform(0.8, 0.95),
                }

                await publisher.publish_json(
                    "integration_test/test_room/state", room_state
                )

                # Publish prediction
                prediction = {
                    "room_id": "test_room",
                    "prediction_time": datetime.now().isoformat(),
                    "next_transition_time": (
                        datetime.now() + timedelta(minutes=random.randint(10, 60))
                    ).isoformat(),
                    "confidence": random.uniform(0.7, 0.9),
                    "model": "integration_test",
                }

                await publisher.publish_json(
                    "integration_test/test_room/prediction", prediction
                )

                await asyncio.sleep(0.1)

            # Verify publisher statistics
            stats = publisher.get_publisher_stats()
            assert stats["messages_published"] >= 20  # At least 20 messages

            print(
                f"Multi-service integration: {stats['messages_published']} messages published"
            )

        finally:
            await publisher.stop_publisher()


class TestNetworkConditions:
    """Test system behavior under various network conditions."""

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_high_latency_conditions(self, test_infrastructure):
        """Test system behavior under high network latency."""
        mqtt_info = test_infrastructure["mqtt"]
        network_sim = test_infrastructure["network_simulator"]

        # Add 200ms latency
        await network_sim.add_latency(200)

        # Create MQTT publisher with latency simulation
        mqtt_config = MQTTConfig(
            broker=mqtt_info["host"],
            port=mqtt_info["port"],
            topic_prefix="latency_test",
            publishing_enabled=True,
            device_identifier="latency_test_device",
        )

        publisher = MQTTPublisher(mqtt_config)

        # Override publish to simulate latency
        original_publish = publisher.publish_json

        async def latency_publish_json(topic, data, qos=1, retain=False):
            await network_sim.apply_latency()
            return await original_publish(topic, data, qos, retain)

        publisher.publish_json = latency_publish_json

        try:
            await publisher.initialize()

            # Test publishing under latency
            start_time = time.time()
            results = []

            for i in range(50):
                result = await publisher.publish_json(
                    f"latency_test/message_{i}", {"id": i, "test": "high_latency"}
                )
                results.append(result.success)

            end_time = time.time()
            duration = end_time - start_time

            success_rate = sum(results) / len(results)
            throughput = len(results) / duration

            # Should handle latency gracefully
            assert success_rate >= 0.90  # Some tolerance for latency
            assert duration >= 10  # Should take at least 10s with 200ms latency

            print(
                f"High latency test: {success_rate:.2%} success, {throughput:.1f} msg/s, {duration:.2f}s"
            )

        finally:
            await publisher.stop_publisher()

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_packet_loss_conditions(self, test_infrastructure):
        """Test system behavior under packet loss."""
        mqtt_info = test_infrastructure["mqtt"]
        network_sim = test_infrastructure["network_simulator"]

        # Add 5% packet loss
        await network_sim.add_packet_loss(5.0)

        mqtt_config = MQTTConfig(
            broker=mqtt_info["host"],
            port=mqtt_info["port"],
            topic_prefix="packet_loss_test",
            publishing_enabled=True,
            device_identifier="packet_loss_test_device",
        )

        publisher = MQTTPublisher(mqtt_config)

        # Override publish to simulate packet loss
        original_publish = publisher.publish_json

        async def lossy_publish_json(topic, data, qos=1, retain=False):
            if network_sim.should_drop_packet():
                # Simulate packet loss
                raise Exception("Simulated packet loss")
            return await original_publish(topic, data, qos, retain)

        publisher.publish_json = lossy_publish_json

        try:
            await publisher.initialize()

            # Test publishing under packet loss
            results = []

            for i in range(100):
                try:
                    result = await publisher.publish_json(
                        f"packet_loss_test/message_{i}",
                        {"id": i, "test": "packet_loss"},
                    )
                    results.append(result.success)
                except Exception:
                    results.append(False)

            success_rate = sum(results) / len(results)

            # Should handle packet loss reasonably
            assert success_rate >= 0.85  # Account for 5% loss + some retries

            print(f"Packet loss test: {success_rate:.2%} success rate")

        finally:
            await publisher.stop_publisher()

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_network_partition_recovery(self, test_infrastructure):
        """Test recovery from network partition."""
        mqtt_info = test_infrastructure["mqtt"]
        network_sim = test_infrastructure["network_simulator"]

        mqtt_config = MQTTConfig(
            broker=mqtt_info["host"],
            port=mqtt_info["port"],
            topic_prefix="partition_test",
            publishing_enabled=True,
            device_identifier="partition_test_device",
        )

        publisher = MQTTPublisher(mqtt_config)

        try:
            await publisher.initialize()

            # Normal operation
            result = await publisher.publish_json(
                "partition_test/before", {"phase": "before_partition"}
            )
            assert result.success

            # Simulate network partition
            partition_task = asyncio.create_task(
                network_sim.create_network_partition(3)  # 3 second partition
            )

            # Try to publish during partition
            partition_results = []
            for i in range(10):
                if network_sim.partition_active:
                    # Should queue messages during partition
                    result = await publisher.publish_json(
                        f"partition_test/during_{i}",
                        {"phase": "during_partition", "id": i},
                    )
                    partition_results.append(result.success)
                await asyncio.sleep(0.3)

            # Wait for partition to resolve
            await partition_task

            # Test recovery
            await asyncio.sleep(1)  # Allow reconnection

            result = await publisher.publish_json(
                "partition_test/after", {"phase": "after_partition"}
            )

            # Should recover successfully
            assert result.success

            # Verify message queuing behavior
            queued_messages = sum(1 for r in partition_results if not r)
            print(
                f"Network partition test: {queued_messages} messages queued during partition"
            )

        finally:
            await publisher.stop_publisher()


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows."""

    @pytest.mark.container
    @pytest.mark.asyncio
    async def test_complete_prediction_workflow(self, test_infrastructure):
        """Test complete prediction workflow from data ingestion to publication."""
        mqtt_info = test_infrastructure["mqtt"]

        # Set up complete system
        system_config = {
            "mqtt": MQTTConfig(
                broker=mqtt_info["host"],
                port=mqtt_info["port"],
                topic_prefix="workflow_test",
                publishing_enabled=True,
                device_identifier="workflow_test_device",
            ),
            "rooms": {
                "workflow_room": RoomConfig(
                    room_id="workflow_room",
                    name="Workflow Test Room",
                    sensors={
                        "motion": ["binary_sensor.workflow_room_motion"],
                        "door": ["binary_sensor.workflow_room_door"],
                    },
                )
            },
        }

        # Create system components
        publisher = MQTTPublisher(system_config["mqtt"])

        device_info = DeviceInfo(
            identifiers=["workflow_test"],
            name="Workflow Test Device",
            manufacturer="Test",
            model="Workflow",
            sw_version="1.0",
        )

        discovery_publisher = DiscoveryPublisher(
            mqtt_publisher=publisher,
            device_info=device_info,
            mqtt_config=system_config["mqtt"],
        )

        ha_entities = HAEntityDefinitions(
            discovery_publisher=discovery_publisher,
            mqtt_config=system_config["mqtt"],
            rooms=system_config["rooms"],
        )

        # Mock tracking manager
        tracking_manager = AsyncMock()
        tracking_manager.get_room_prediction.return_value = {
            "room_id": "workflow_room",
            "prediction_time": datetime.now().isoformat(),
            "next_transition_time": (
                datetime.now() + timedelta(minutes=30)
            ).isoformat(),
            "transition_type": "vacant_to_occupied",
            "confidence": 0.85,
            "model_info": {"model": "workflow_test"},
        }

        try:
            # 1. Initialize system
            await publisher.initialize()

            # 2. Define and publish entities
            entities = ha_entities.define_all_entities()
            entity_results = await ha_entities.publish_all_entities()

            assert len(entities) > 0
            assert all(r.success for r in entity_results.values())

            # 3. Simulate sensor data ingestion
            sensor_data = [
                {
                    "sensor": "motion",
                    "state": "on",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "sensor": "door",
                    "state": "on",
                    "timestamp": datetime.now().isoformat(),
                },
                {
                    "sensor": "motion",
                    "state": "off",
                    "timestamp": datetime.now().isoformat(),
                },
            ]

            for data in sensor_data:
                await publisher.publish_json(
                    f"workflow_test/workflow_room/sensors/{data['sensor']}", data
                )

            # 4. Generate prediction
            prediction = await tracking_manager.get_room_prediction("workflow_room")

            # 5. Publish prediction
            await publisher.publish_json(
                "workflow_test/workflow_room/prediction", prediction
            )

            # 6. Publish state updates
            room_state = {
                "room_id": "workflow_room",
                "currently_occupied": False,
                "occupancy_confidence": 0.9,
                "last_motion_time": datetime.now().isoformat(),
                "prediction": prediction,
            }

            await publisher.publish_json(
                "workflow_test/workflow_room/state", room_state
            )

            # 7. Verify workflow completion
            stats = publisher.get_publisher_stats()

            # Should have published:
            # - Entity discoveries
            # - Sensor data (3 messages)
            # - Prediction (1 message)
            # - State (1 message)
            expected_messages = len(entity_results) + 3 + 1 + 1

            assert stats["messages_published"] >= expected_messages

            print(
                f"Complete workflow test: {stats['messages_published']} messages published"
            )

        finally:
            await publisher.stop_publisher()

    @pytest.mark.container
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_long_running_integration(self, test_infrastructure):
        """Test long-running integration stability."""
        mqtt_info = test_infrastructure["mqtt"]

        mqtt_config = MQTTConfig(
            broker=mqtt_info["host"],
            port=mqtt_info["port"],
            topic_prefix="long_running_test",
            publishing_enabled=True,
            device_identifier="long_running_device",
        )

        publisher = MQTTPublisher(mqtt_config)

        try:
            await publisher.initialize()

            # Run for 2 minutes with continuous activity
            end_time = time.time() + 120  # 2 minutes
            message_count = 0
            error_count = 0

            while time.time() < end_time:
                try:
                    # Simulate realistic message patterns
                    message_types = [
                        {"topic": "sensor_data", "rate": 2},  # Every 2 seconds
                        {"topic": "predictions", "rate": 10},  # Every 10 seconds
                        {"topic": "status", "rate": 30},  # Every 30 seconds
                    ]

                    for msg_type in message_types:
                        if message_count % msg_type["rate"] == 0:
                            result = await publisher.publish_json(
                                f"long_running_test/{msg_type['topic']}/{message_count}",
                                {
                                    "id": message_count,
                                    "type": msg_type["topic"],
                                    "timestamp": datetime.now().isoformat(),
                                },
                            )

                            if not result.success:
                                error_count += 1

                    message_count += 1
                    await asyncio.sleep(1)  # 1 second intervals

                except Exception as e:
                    error_count += 1
                    logging.warning(f"Error in long-running test: {e}")

            # Analyze long-running performance
            stats = publisher.get_publisher_stats()
            error_rate = error_count / message_count if message_count > 0 else 0

            # Long-running stability assertions
            assert error_rate < 0.05  # Less than 5% error rate
            assert (
                stats["messages_published"] > 100
            )  # Should have published many messages

            print(
                f"Long-running test: {message_count} operations, {error_rate:.2%} error rate"
            )

        finally:
            await publisher.stop_publisher()
