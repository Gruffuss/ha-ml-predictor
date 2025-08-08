"""
Additional test fixtures and utilities for Sprint 5 testing.

This module provides specialized fixtures and utilities for testing
Sprint 5 integration components, including mock services, test data
factories, and integration test helpers.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest
import pytest_asyncio
import websockets
from aiohttp import ClientSession
from aiohttp import web
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.adaptation.tracking_manager import TrackingConfig
from src.adaptation.tracking_manager import TrackingManager
from src.core.config import APIConfig
from src.core.config import MQTTConfig
from src.core.config import SystemConfig
from src.integration.enhanced_mqtt_manager import EnhancedIntegrationStats
from src.integration.mqtt_integration_manager import MQTTIntegrationStats
from src.integration.realtime_publisher import PublishingChannel
from src.integration.realtime_publisher import PublishingMetrics
from src.integration.realtime_publisher import RealtimePredictionEvent

logger = logging.getLogger(__name__)


@dataclass
class MockSystemMetrics:
    """Mock system metrics for testing."""

    predictions_generated: int = 0
    mqtt_messages_sent: int = 0
    websocket_connections: int = 0
    sse_connections: int = 0
    api_requests: int = 0
    errors_occurred: int = 0
    average_response_time_ms: float = 0.0
    last_update: datetime = field(default_factory=datetime.utcnow)

    def increment_predictions(self):
        """Increment prediction counter."""
        self.predictions_generated += 1
        self.last_update = datetime.utcnow()

    def increment_mqtt_messages(self):
        """Increment MQTT message counter."""
        self.mqtt_messages_sent += 1
        self.last_update = datetime.utcnow()

    def increment_api_requests(self, response_time_ms: float = 0.0):
        """Increment API request counter and update response time."""
        self.api_requests += 1
        if response_time_ms > 0:
            # Simple moving average
            self.average_response_time_ms = (
                self.average_response_time_ms * (self.api_requests - 1)
                + response_time_ms
            ) / self.api_requests
        self.last_update = datetime.utcnow()

    def increment_errors(self):
        """Increment error counter."""
        self.errors_occurred += 1
        self.last_update = datetime.utcnow()

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "predictions_generated": self.predictions_generated,
            "mqtt_messages_sent": self.mqtt_messages_sent,
            "websocket_connections": self.websocket_connections,
            "sse_connections": self.sse_connections,
            "api_requests": self.api_requests,
            "errors_occurred": self.errors_occurred,
            "average_response_time_ms": self.average_response_time_ms,
            "last_update": self.last_update.isoformat(),
        }


class MockRealtimeClients:
    """Mock real-time clients for testing."""

    def __init__(self):
        self.websocket_clients = []
        self.sse_clients = []
        self.message_history = []

    def add_websocket_client(self, client_id: str):
        """Add a WebSocket client."""
        client = {
            "id": client_id,
            "type": "websocket",
            "connected_at": datetime.utcnow(),
            "messages_received": 0,
        }
        self.websocket_clients.append(client)
        return client

    def add_sse_client(self, client_id: str):
        """Add an SSE client."""
        client = {
            "id": client_id,
            "type": "sse",
            "connected_at": datetime.utcnow(),
            "messages_received": 0,
        }
        self.sse_clients.append(client)
        return client

    def remove_client(self, client_id: str):
        """Remove a client by ID."""
        self.websocket_clients = [
            c for c in self.websocket_clients if c["id"] != client_id
        ]
        self.sse_clients = [c for c in self.sse_clients if c["id"] != client_id]

    def broadcast_message(self, message: Dict[str, Any]):
        """Simulate broadcasting a message to all clients."""
        timestamp = datetime.utcnow()

        # Record message
        self.message_history.append(
            {
                "message": message,
                "timestamp": timestamp,
                "websocket_recipients": len(self.websocket_clients),
                "sse_recipients": len(self.sse_clients),
            }
        )

        # Update client message counts
        for client in self.websocket_clients + self.sse_clients:
            client["messages_received"] += 1

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics."""
        return {
            "websocket_connections": {
                "total_active_connections": len(self.websocket_clients),
                "clients": [
                    {
                        "id": c["id"],
                        "connected_at": c["connected_at"].isoformat(),
                        "messages_received": c["messages_received"],
                    }
                    for c in self.websocket_clients
                ],
            },
            "sse_connections": {
                "total_active_connections": len(self.sse_clients),
                "clients": [
                    {
                        "id": c["id"],
                        "connected_at": c["connected_at"].isoformat(),
                        "messages_received": c["messages_received"],
                    }
                    for c in self.sse_clients
                ],
            },
            "total_active_connections": len(self.websocket_clients)
            + len(self.sse_clients),
            "message_history_count": len(self.message_history),
        }


class MockMQTTBroker:
    """Mock MQTT broker for testing."""

    def __init__(self):
        self.topics = {}
        self.subscribers = {}
        self.published_messages = []
        self.connected_clients = []
        self.broker_stats = {
            "messages_published": 0,
            "messages_delivered": 0,
            "active_subscriptions": 0,
            "connected_clients": 0,
        }

    async def connect_client(self, client_id: str):
        """Connect a client to the broker."""
        if client_id not in self.connected_clients:
            self.connected_clients.append(client_id)
            self.broker_stats["connected_clients"] = len(self.connected_clients)

    async def disconnect_client(self, client_id: str):
        """Disconnect a client from the broker."""
        if client_id in self.connected_clients:
            self.connected_clients.remove(client_id)
            self.broker_stats["connected_clients"] = len(self.connected_clients)

    async def publish_message(self, topic: str, payload: str, client_id: str = None):
        """Publish a message to a topic."""
        message = {
            "topic": topic,
            "payload": payload,
            "client_id": client_id,
            "timestamp": datetime.utcnow(),
            "qos": 0,
        }

        self.published_messages.append(message)
        self.broker_stats["messages_published"] += 1

        # Store message in topic
        if topic not in self.topics:
            self.topics[topic] = []
        self.topics[topic].append(message)

        # Deliver to subscribers
        if topic in self.subscribers:
            for subscriber in self.subscribers[topic]:
                self.broker_stats["messages_delivered"] += 1

    async def subscribe(self, topic: str, client_id: str):
        """Subscribe a client to a topic."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []

        if client_id not in self.subscribers[topic]:
            self.subscribers[topic].append(client_id)
            self.broker_stats["active_subscriptions"] += 1

    def get_topic_messages(self, topic: str) -> List[Dict[str, Any]]:
        """Get all messages for a topic."""
        return self.topics.get(topic, [])

    def get_broker_stats(self) -> Dict[str, Any]:
        """Get broker statistics."""
        return {
            **self.broker_stats,
            "total_topics": len(self.topics),
            "total_published_messages": len(self.published_messages),
            "uptime_seconds": 3600,  # Mock uptime
        }


@pytest.fixture
def mock_system_metrics():
    """Create mock system metrics."""
    return MockSystemMetrics()


@pytest.fixture
def mock_realtime_clients():
    """Create mock real-time clients."""
    return MockRealtimeClients()


@pytest.fixture
def mock_mqtt_broker():
    """Create mock MQTT broker."""
    return MockMQTTBroker()


@pytest.fixture
def comprehensive_prediction_data():
    """Create comprehensive prediction data for testing."""
    base_time = datetime.utcnow()

    return {
        "living_room": {
            "room_id": "living_room",
            "prediction_time": base_time.isoformat(),
            "next_transition_time": (base_time + timedelta(minutes=25)).isoformat(),
            "transition_type": "occupied_to_vacant",
            "confidence": 0.87,
            "time_until_transition": "25 minutes",
            "alternatives": [
                {
                    "transition_time": (base_time + timedelta(minutes=30)).isoformat(),
                    "confidence": 0.75,
                    "scenario": "delayed_departure",
                },
                {
                    "transition_time": (base_time + timedelta(minutes=20)).isoformat(),
                    "confidence": 0.68,
                    "scenario": "early_departure",
                },
            ],
            "model_info": {
                "model_type": "ensemble",
                "version": "1.2.0",
                "base_models": ["lstm", "xgboost", "hmm"],
                "training_data_hours": 168,
                "last_retrain": (base_time - timedelta(hours=6)).isoformat(),
            },
            "features_used": {
                "temporal_features": 12,
                "sequential_features": 8,
                "contextual_features": 5,
            },
            "validation_metrics": {
                "recent_accuracy": 0.89,
                "confidence_calibration": 0.91,
                "prediction_count_24h": 48,
            },
        },
        "bedroom": {
            "room_id": "bedroom",
            "prediction_time": base_time.isoformat(),
            "next_transition_time": (base_time + timedelta(hours=8)).isoformat(),
            "transition_type": "vacant_to_occupied",
            "confidence": 0.82,
            "time_until_transition": "8 hours",
            "alternatives": [
                {
                    "transition_time": (
                        base_time + timedelta(hours=7, minutes=30)
                    ).isoformat(),
                    "confidence": 0.71,
                    "scenario": "early_sleep",
                }
            ],
            "model_info": {
                "model_type": "ensemble",
                "version": "1.2.0",
                "base_models": ["lstm", "xgboost"],
                "training_data_hours": 168,
                "last_retrain": (base_time - timedelta(hours=4)).isoformat(),
            },
        },
        "kitchen": {
            "room_id": "kitchen",
            "prediction_time": base_time.isoformat(),
            "next_transition_time": (base_time + timedelta(minutes=45)).isoformat(),
            "transition_type": "vacant_to_occupied",
            "confidence": 0.93,
            "time_until_transition": "45 minutes",
            "alternatives": [],
            "model_info": {
                "model_type": "ensemble",
                "version": "1.2.0",
                "base_models": ["lstm", "xgboost", "hmm"],
                "training_data_hours": 168,
                "last_retrain": (base_time - timedelta(hours=2)).isoformat(),
            },
        },
    }


@pytest.fixture
def mock_integration_stats():
    """Create mock integration statistics."""
    return EnhancedIntegrationStats(
        mqtt_stats=MQTTIntegrationStats(
            mqtt_connected=True,
            predictions_published=125,
            discovery_messages_sent=12,
            last_prediction_time=datetime.utcnow(),
            connection_uptime_hours=72.5,
            error_count=2,
            reconnection_count=1,
        ),
        realtime_stats=PublishingMetrics(
            messages_published=150,
            websocket_broadcasts=75,
            sse_broadcasts=25,
            mqtt_publishes=125,
            failed_publishes=3,
            average_latency_ms=25.8,
            active_websocket_connections=8,
            active_sse_connections=3,
            last_publish_time=datetime.utcnow(),
        ),
        total_channels_active=3,
        total_clients_connected=11,
        predictions_per_minute=2.5,
        average_publish_latency_ms=25.8,
        publish_success_rate=0.98,
        last_performance_update=datetime.utcnow(),
    )


class TestDataFactory:
    """Factory for creating test data."""

    @staticmethod
    def create_sensor_events(
        room_id: str = "test_room", count: int = 10, start_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """Create a series of sensor events."""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=1)

        events = []
        for i in range(count):
            event = {
                "room_id": room_id,
                "sensor_id": f"binary_sensor.{room_id}_sensor_{i % 3}",
                "sensor_type": "presence",
                "state": "on" if i % 2 == 0 else "off",
                "previous_state": "off" if i % 2 == 0 else "on",
                "timestamp": (start_time + timedelta(minutes=i * 5)).isoformat(),
                "attributes": {
                    "device_class": "motion",
                    "friendly_name": f"Test Sensor {i}",
                    "battery_level": 85 + (i % 15),
                },
                "is_human_triggered": i % 4 != 0,  # 75% human, 25% cat
                "confidence_score": 0.7 + (i * 0.03) % 0.3,
            }
            events.append(event)

        return events

    @staticmethod
    def create_room_states(
        room_id: str = "test_room", count: int = 5, start_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """Create a series of room states."""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=2)

        states = []
        for i in range(count):
            state = {
                "room_id": room_id,
                "timestamp": (start_time + timedelta(minutes=i * 20)).isoformat(),
                "is_occupied": i % 2 == 0,
                "occupancy_confidence": 0.8 + (i * 0.02),
                "occupant_type": "human" if i % 3 != 0 else "cat",
                "state_duration": 300 + i * 120,
                "transition_trigger": f"binary_sensor.{room_id}_sensor_{i % 2}",
            }
            states.append(state)

        return states

    @staticmethod
    def create_predictions(
        room_id: str = "test_room", count: int = 3, start_time: datetime = None
    ) -> List[Dict[str, Any]]:
        """Create a series of predictions."""
        if start_time is None:
            start_time = datetime.utcnow() - timedelta(hours=1)

        predictions = []
        transition_types = ["occupied_to_vacant", "vacant_to_occupied"]

        for i in range(count):
            prediction = {
                "room_id": room_id,
                "prediction_time": (start_time + timedelta(minutes=i * 15)).isoformat(),
                "predicted_transition_time": (
                    start_time + timedelta(minutes=i * 15 + 30)
                ).isoformat(),
                "transition_type": transition_types[i % 2],
                "confidence_score": 0.75 + (i * 0.05),
                "model_type": "ensemble",
                "model_version": f"v1.{i}",
                "prediction_metadata": {
                    "features_count": 25 + i * 2,
                    "training_samples": 1000 + i * 100,
                    "cross_validation_score": 0.85 + (i * 0.02),
                },
            }
            predictions.append(prediction)

        return predictions


@pytest.fixture
def test_data_factory():
    """Create test data factory."""
    return TestDataFactory()


class MockWebSocketServer:
    """Mock WebSocket server for testing."""

    def __init__(self, port: int = 8765):
        self.port = port
        self.clients = []
        self.messages_sent = []
        self.server = None
        self.is_running = False

    async def start(self):
        """Start the mock WebSocket server."""

        async def handler(websocket, path):
            self.clients.append(websocket)
            try:
                async for message in websocket:
                    # Echo message back
                    await websocket.send(f"Echo: {message}")
            except Exception:
                pass
            finally:
                if websocket in self.clients:
                    self.clients.remove(websocket)

        self.server = await websockets.serve(handler, "localhost", self.port)
        self.is_running = True

    async def stop(self):
        """Stop the mock WebSocket server."""
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.is_running = False

    async def broadcast(self, message: str):
        """Broadcast message to all connected clients."""
        if self.clients:
            disconnected = []
            for client in self.clients:
                try:
                    await client.send(message)
                    self.messages_sent.append(
                        {
                            "message": message,
                            "timestamp": datetime.utcnow(),
                            "client": str(client),
                        }
                    )
                except Exception:
                    disconnected.append(client)

            # Remove disconnected clients
            for client in disconnected:
                self.clients.remove(client)

    def get_stats(self) -> Dict[str, Any]:
        """Get server statistics."""
        return {
            "is_running": self.is_running,
            "connected_clients": len(self.clients),
            "messages_sent": len(self.messages_sent),
            "port": self.port,
        }


@pytest_asyncio.fixture
async def mock_websocket_server():
    """Create and start a mock WebSocket server."""
    server = MockWebSocketServer()
    await server.start()
    yield server
    await server.stop()


class IntegrationTestHelper:
    """Helper class for integration testing."""

    @staticmethod
    def assert_prediction_structure(prediction: Dict[str, Any]):
        """Assert that a prediction has the correct structure."""
        required_fields = [
            "room_id",
            "prediction_time",
            "confidence",
            "transition_type",
        ]

        for field in required_fields:
            assert field in prediction, f"Missing required field: {field}"

        assert isinstance(prediction["confidence"], (int, float))
        assert 0.0 <= prediction["confidence"] <= 1.0
        assert prediction["transition_type"] in [
            "occupied_to_vacant",
            "vacant_to_occupied",
        ]

    @staticmethod
    def assert_health_response_structure(health: Dict[str, Any]):
        """Assert that a health response has the correct structure."""
        required_fields = ["status", "timestamp", "components"]

        for field in required_fields:
            assert field in health, f"Missing required field: {field}"

        assert health["status"] in ["healthy", "degraded", "unhealthy"]
        assert "database" in health["components"]

    @staticmethod
    def assert_mqtt_message_structure(message: Dict[str, Any]):
        """Assert that an MQTT message has the correct structure."""
        required_fields = ["topic", "payload", "timestamp"]

        for field in required_fields:
            assert field in message, f"Missing required field: {field}"

        # Validate topic format
        assert message["topic"].startswith("occupancy/") or message["topic"].startswith(
            "homeassistant/"
        )

    @staticmethod
    def create_load_test_scenario(
        num_rooms: int = 5, predictions_per_room: int = 10, concurrent_clients: int = 20
    ) -> Dict[str, Any]:
        """Create a load testing scenario."""
        return {
            "rooms": [f"room_{i}" for i in range(num_rooms)],
            "predictions_per_room": predictions_per_room,
            "concurrent_clients": concurrent_clients,
            "total_predictions": num_rooms * predictions_per_room,
            "expected_api_calls": num_rooms * predictions_per_room * concurrent_clients,
        }


@pytest.fixture
def integration_test_helper():
    """Create integration test helper."""
    return IntegrationTestHelper()


# Custom pytest markers for Sprint 5 tests
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "sprint5: mark test as Sprint 5 integration test"
    )
    config.addinivalue_line(
        "markers", "realtime: mark test as real-time integration test"
    )
    config.addinivalue_line("markers", "mqtt: mark test as MQTT integration test")
    config.addinivalue_line("markers", "api: mark test as API integration test")
    config.addinivalue_line("markers", "websocket: mark test as WebSocket test")
    config.addinivalue_line("markers", "sse: mark test as Server-Sent Events test")
    config.addinivalue_line("markers", "performance: mark test as performance test")
    config.addinivalue_line("markers", "load: mark test as load test")
    config.addinivalue_line("markers", "e2e: mark test as end-to-end test")


# Utility functions for test data generation
def generate_realistic_sensor_pattern(
    room_id: str, days: int = 7, occupancy_probability: float = 0.3
) -> List[Dict[str, Any]]:
    """Generate realistic sensor event patterns."""
    events = []
    start_time = datetime.utcnow() - timedelta(days=days)

    for day in range(days):
        day_start = start_time + timedelta(days=day)

        # Morning activity (7-9 AM)
        for hour in range(7, 9):
            if hour == 7:  # Higher activity in morning
                event_probability = 0.8
            else:
                event_probability = 0.6

            if event_probability > occupancy_probability:
                events.extend(
                    TestDataFactory.create_sensor_events(
                        room_id=room_id,
                        count=3,
                        start_time=day_start + timedelta(hours=hour),
                    )
                )

        # Evening activity (18-22 PM)
        for hour in range(18, 22):
            event_probability = 0.7
            if event_probability > occupancy_probability:
                events.extend(
                    TestDataFactory.create_sensor_events(
                        room_id=room_id,
                        count=2,
                        start_time=day_start + timedelta(hours=hour),
                    )
                )

    return events


def create_test_mqtt_discovery_payload(room_id: str) -> Dict[str, Any]:
    """Create a test MQTT discovery payload."""
    return {
        "name": f"Occupancy Prediction {room_id.title()}",
        "unique_id": f"occupancy_prediction_{room_id}",
        "state_topic": f"occupancy/predictions/{room_id}/state",
        "json_attributes_topic": f"occupancy/predictions/{room_id}/attributes",
        "device_class": "occupancy",
        "value_template": "{{ value_json.predicted_state }}",
        "device": {
            "identifiers": [f"occupancy_predictor_{room_id}"],
            "name": f"Occupancy Predictor {room_id.title()}",
            "model": "ML Predictor v1.0",
            "manufacturer": "Home Assistant ML",
        },
        "availability": {
            "topic": "occupancy/predictions/status",
            "payload_available": "online",
            "payload_not_available": "offline",
        },
    }
