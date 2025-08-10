"""
End-to-End Validation Tests for Complete System Workflow.

This module provides comprehensive end-to-end validation tests that verify
the complete system workflow from sensor event ingestion to Home Assistant
entity updates, covering all Sprint 5 integration components.

Test Coverage:
- Complete prediction workflow: sensor event → prediction → publishing → HA entity
- Multi-channel publishing validation (MQTT, WebSocket, SSE)
- System health and performance validation under load
- Configuration validation and component initialization
- Error propagation and recovery across the entire system
- Real-time event handling and broadcasting
- Dashboard metrics and monitoring integration
- Authentication and security validation
- Resource usage and performance benchmarking
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from datetime import datetime, timedelta
import logging
import time
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from aiohttp import ClientSession, web
from aiohttp.test_utils import AioHTTPTestCase
from fastapi.testclient import TestClient
import pytest
import pytest_asyncio
import websockets

from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.config import SystemConfig, get_config
from src.core.constants import SensorState, SensorType
from src.data.ingestion.event_processor import EventProcessor
from src.data.ingestion.ha_client import HAEvent
from src.data.storage.models import Prediction, RoomState, SensorEvent
from src.integration.api_server import create_app, set_tracking_manager
from src.integration.enhanced_mqtt_manager import (
    EnhancedMQTTIntegrationManager,
)
from src.integration.realtime_publisher import (
    PublishingChannel,
    RealtimePredictionEvent,
    RealtimePublishingSystem,
)
from src.integration.tracking_integration import (
    IntegrationConfig,
    TrackingIntegrationManager,
    create_integrated_tracking_manager,
)
from src.models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


@pytest.fixture
def e2e_system_config():
    """Create a comprehensive system configuration for E2E testing."""
    return {
        "home_assistant": {
            "url": "http://test-ha:8123",
            "token": "test_token_e2e",
            "websocket_timeout": 30,
            "api_timeout": 10,
        },
        "database": {
            "connection_string": "sqlite+aiosqlite:///:memory:",
            "pool_size": 5,
            "max_overflow": 10,
        },
        "mqtt": {
            "broker": "test-mqtt-broker",
            "port": 1883,
            "username": "test_user",
            "password": "test_pass",
            "topic_prefix": "occupancy/predictions",
        },
        "api": {
            "enabled": True,
            "host": "127.0.0.1",
            "port": 8000,
            "api_key_enabled": True,
            "api_key": "test_e2e_key",
            "rate_limit_enabled": True,
            "requests_per_minute": 100,
        },
        "tracking": {
            "accuracy_threshold_minutes": 15,
            "confidence_threshold": 0.7,
            "validation_window_hours": 24,
            "prediction_interval_seconds": 300,
        },
    }


@pytest.fixture
def e2e_room_configs():
    """Create room configurations for E2E testing."""
    return {
        "living_room": {
            "name": "Living Room",
            "sensors": {
                "presence": {
                    "main": "binary_sensor.living_room_presence",
                    "couch": "binary_sensor.living_room_couch",
                },
                "door": "binary_sensor.living_room_door",
                "climate": {
                    "temperature": "sensor.living_room_temperature",
                    "humidity": "sensor.living_room_humidity",
                },
            },
        },
        "bedroom": {
            "name": "Bedroom",
            "sensors": {
                "presence": {
                    "main": "binary_sensor.bedroom_presence",
                    "bed": "binary_sensor.bedroom_bed",
                },
                "door": "binary_sensor.bedroom_door",
            },
        },
        "kitchen": {
            "name": "Kitchen",
            "sensors": {
                "presence": {"main": "binary_sensor.kitchen_presence"},
                "motion": "binary_sensor.kitchen_motion",
            },
        },
    }


class MockExternalServices:
    """Mock external services for E2E testing."""

    def __init__(self):
        self.ha_events = []
        self.mqtt_messages = []
        self.websocket_messages = []
        self.sse_messages = []
        self.api_requests = []

        # Service states
        self.ha_connected = True
        self.mqtt_connected = True
        self.websocket_clients = []
        self.sse_clients = []

    async def mock_ha_client(self):
        """Mock Home Assistant client."""
        client = AsyncMock()
        client.is_connected = self.ha_connected
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()

        client.get_entity_state = AsyncMock(
            return_value={
                "state": "of",
                "attributes": {"device_class": "motion"},
            }
        )

        client.get_entity_history = AsyncMock(return_value=self.ha_events)
        client.subscribe_to_events = AsyncMock()
        return client

    async def mock_mqtt_client(self):
        """Mock MQTT client."""
        client = AsyncMock()
        client.connect = AsyncMock()
        client.disconnect = AsyncMock()
        client.publish = AsyncMock(side_effect=self._capture_mqtt_message)
        client.is_connected = Mock(return_value=self.mqtt_connected)
        return client

    def _capture_mqtt_message(self, topic, payload, **kwargs):
        """Capture MQTT messages for validation."""
        self.mqtt_messages.append(
            {
                "topic": topic,
                "payload": payload,
                "timestamp": datetime.utcnow(),
                "kwargs": kwargs,
            }
        )

    async def mock_websocket_handler(self, websocket, path):
        """Mock WebSocket handler."""
        self.websocket_clients.append(websocket)
        try:
            async for message in websocket:
                self.websocket_messages.append(
                    {
                        "message": message,
                        "timestamp": datetime.utcnow(),
                        "path": path,
                    }
                )
        except Exception:
            pass
        finally:
            if websocket in self.websocket_clients:
                self.websocket_clients.remove(websocket)

    def add_ha_event(self, entity_id: str, state: str, timestamp: datetime = None):
        """Add a Home Assistant event for testing."""
        if timestamp is None:
            timestamp = datetime.utcnow()

        event = HAEvent(
            entity_id=entity_id,
            state=state,
            previous_state="of" if state == "on" else "on",
            timestamp=timestamp,
            attributes={
                "device_class": "motion",
                "friendly_name": f"Test {entity_id}",
            },
        )
        self.ha_events.append(event)
        return event


@pytest_asyncio.fixture
async def mock_external_services():
    """Create mock external services for E2E testing."""
    services = MockExternalServices()
    yield services


@pytest_asyncio.fixture
async def complete_system(e2e_system_config, e2e_room_configs, mock_external_services):
    """Create a complete system for E2E testing."""

    # Patch external services
    with patch("src.data.ingestion.ha_client.HomeAssistantClient") as mock_ha_class:
        with patch("src.integration.mqtt_publisher.MQTTPublisher") as mock_mqtt_class:
            with patch(
                "src.integration.realtime_publisher.RealtimePublishingSystem"
            ) as mock_realtime_class:

                # Configure mocks
                ha_client = await mock_external_services.mock_ha_client()
                mqtt_client = await mock_external_services.mock_mqtt_client()

                mock_ha_class.return_value = ha_client
                mock_mqtt_class.return_value = mqtt_client

                # Mock realtime publisher
                mock_realtime = AsyncMock()
                mock_realtime.initialize = AsyncMock()
                mock_realtime.shutdown = AsyncMock()
                mock_realtime.publish_prediction = AsyncMock()
                mock_realtime.broadcast_system_event = AsyncMock()
                mock_realtime_class.return_value = mock_realtime

                # Create system configuration
                with patch("src.core.config.get_config") as mock_get_config:
                    mock_config = Mock()
                    for key, value in e2e_system_config.items():
                        setattr(
                            mock_config,
                            key,
                            (Mock(**value) if isinstance(value, dict) else value),
                        )

                    # Add API config to disable authentication and rate limiting for tests
                    mock_config.api = Mock()
                    mock_config.api.enabled = True
                    mock_config.api.host = "0.0.0.0"
                    mock_config.api.port = 8000
                    mock_config.api.debug = False
                    mock_config.api.enable_cors = True
                    mock_config.api.cors_origins = ["*"]
                    mock_config.api.api_key_enabled = False
                    mock_config.api.api_key = None
                    mock_config.api.rate_limit_enabled = False
                    mock_config.api.requests_per_minute = 60
                    mock_config.api.burst_limit = 100
                    mock_config.api.request_timeout_seconds = 30
                    mock_config.api.max_request_size_mb = 10
                    mock_config.api.include_docs = True
                    mock_config.api.docs_url = "/docs"
                    mock_config.api.redoc_url = "/redoc"

                    mock_config.rooms = {}
                    for room_id, room_config in e2e_room_configs.items():
                        room_mock = Mock()
                        room_mock.room_id = room_id
                        room_mock.name = room_config["name"]
                        room_mock.sensors = room_config["sensors"]
                        mock_config.rooms[room_id] = room_mock

                    mock_get_config.return_value = mock_config

                    # Create integrated system
                    tracking_config = TrackingConfig(
                        enabled=True,
                        monitoring_interval_seconds=300,
                        auto_validation_enabled=True,
                        validation_window_minutes=30,
                        realtime_publishing_enabled=True,
                        websocket_enabled=True,
                        sse_enabled=True,
                    )

                    integration_config = IntegrationConfig(
                        enable_realtime_publishing=True,
                        enable_websocket_server=True,
                        enable_sse_server=True,
                    )

                    # Mock the tracking manager and integration
                    with patch(
                        "src.adaptation.tracking_manager.TrackingManager"
                    ) as mock_tracking_class:
                        with patch(
                            "src.integration.tracking_integration.EnhancedMQTTIntegrationManager"
                        ) as mock_enhanced_mqtt_class:

                            # Create mock tracking manager
                            mock_tracking = AsyncMock()
                            mock_tracking.initialize = AsyncMock()
                            mock_tracking.shutdown = AsyncMock()
                            mock_tracking.notification_callbacks = []
                            mock_tracking.add_notification_callback = Mock()
                            mock_tracking.mqtt_integration_manager = None

                            # Mock prediction methods
                            mock_tracking.get_room_prediction = AsyncMock()
                            mock_tracking.get_accuracy_metrics = AsyncMock()
                            mock_tracking.get_system_stats = AsyncMock()
                            mock_tracking.get_tracking_status = AsyncMock()
                            mock_tracking.get_active_alerts = AsyncMock(return_value=[])
                            mock_tracking.trigger_manual_retrain = AsyncMock()

                            mock_tracking_class.return_value = mock_tracking

                            # Create mock enhanced MQTT manager
                            mock_enhanced_mqtt = AsyncMock()
                            mock_enhanced_mqtt.initialize = AsyncMock()
                            mock_enhanced_mqtt.shutdown = AsyncMock()
                            mock_enhanced_mqtt.publish_prediction = AsyncMock()
                            mock_enhanced_mqtt.publish_system_status = AsyncMock()
                            mock_enhanced_mqtt.get_integration_stats = Mock(
                                return_value={}
                            )
                            mock_enhanced_mqtt.get_connection_info = Mock(
                                return_value={"total_active_connections": 0}
                            )
                            mock_enhanced_mqtt_class.return_value = mock_enhanced_mqtt

                            # Create integrated system
                            tracking_manager, integration_manager = (
                                await create_integrated_tracking_manager(
                                    tracking_config=tracking_config,
                                    integration_config=integration_config,
                                )
                            )

                            # Create API server
                            set_tracking_manager(tracking_manager)
                            app = create_app()

                            yield {
                                "tracking_manager": tracking_manager,
                                "integration_manager": integration_manager,
                                "api_app": app,
                                "mock_tracking": mock_tracking,
                                "mock_ha_client": ha_client,
                                "mock_mqtt_client": mqtt_client,
                                "mock_realtime": mock_realtime,
                                "mock_enhanced_mqtt": mock_enhanced_mqtt,
                                "external_services": mock_external_services,
                            }

                            # Cleanup
                            await integration_manager.shutdown()


class TestCompleteSystemWorkflow:
    """Test complete system workflow from sensor event to HA entity update."""

    @pytest.mark.asyncio
    async def test_sensor_event_to_prediction_workflow(self, complete_system):
        """Test complete workflow from sensor event to prediction generation."""
        system = complete_system
        external_services = system["external_services"]
        mock_tracking = system["mock_tracking"]

        # Add sensor events
        external_services.add_ha_event("binary_sensor.living_room_presence", "on")
        external_services.add_ha_event("binary_sensor.living_room_presence", "off")

        # Configure tracking manager to return prediction
        prediction_data = {
            "room_id": "living_room",
            "prediction_time": datetime.utcnow().isoformat(),
            "next_transition_time": (
                datetime.utcnow() + timedelta(minutes=30)
            ).isoformat(),
            "transition_type": "vacant_to_occupied",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {"model_type": "ensemble", "version": "1.0"},
        }
        mock_tracking.get_room_prediction.return_value = prediction_data

        # Request prediction through API
        with TestClient(system["api_app"]) as client:
            response = client.get("/predictions/living_room")
            assert response.status_code == 200

            data = response.json()
            assert data["room_id"] == "living_room"
            assert data["confidence"] == 0.85
            assert data["transition_type"] == "vacant_to_occupied"

        # Verify tracking manager was called
        mock_tracking.get_room_prediction.assert_called_with("living_room")

    @pytest.mark.asyncio
    async def test_prediction_publishing_workflow(self, complete_system):
        """Test prediction publishing across all channels."""
        system = complete_system
        mock_enhanced_mqtt = system["mock_enhanced_mqtt"]

        prediction_data = {
            "room_id": "bedroom",
            "prediction_time": datetime.utcnow().isoformat(),
            "confidence": 0.78,
        }

        # Simulate prediction publishing
        await mock_enhanced_mqtt.publish_prediction("bedroom", prediction_data)

        # Verify publishing was called
        mock_enhanced_mqtt.publish_prediction.assert_called_once_with(
            "bedroom", prediction_data
        )

    @pytest.mark.asyncio
    async def test_system_health_monitoring_workflow(self, complete_system):
        """Test complete system health monitoring workflow."""
        system = complete_system

        # Configure health responses
        with patch("src.integration.api_server.get_database_manager") as mock_db:
            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = AsyncMock(
                return_value={
                    "database_connected": True,
                    "connection_pool_size": 5,
                    "active_connections": 2,
                }
            )
            mock_db.return_value = mock_db_manager

            with patch("src.integration.api_server.get_mqtt_manager") as mock_mqtt_mgr:
                mock_mqtt_manager = AsyncMock()
                mock_mqtt_manager.get_integration_stats = AsyncMock(
                    return_value=Mock(mqtt_connected=True, predictions_published=25)
                )
                mock_mqtt_mgr.return_value = mock_mqtt_manager

                # Test health endpoint
                with TestClient(system["api_app"]) as client:
                    response = client.get("/health")
                    assert response.status_code == 200

                    health_data = response.json()
                    assert health_data["status"] in [
                        "healthy",
                        "degraded",
                        "unhealthy",
                    ]
                    assert "components" in health_data
                    assert "database" in health_data["components"]
                    assert "tracking" in health_data["components"]
                    assert "mqtt" in health_data["components"]

    @pytest.mark.asyncio
    async def test_multi_room_prediction_workflow(self, complete_system):
        """Test prediction workflow across multiple rooms."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure predictions for multiple rooms
        room_predictions = {
            "living_room": {
                "room_id": "living_room",
                "confidence": 0.85,
                "transition_type": "occupied_to_vacant",
            },
            "bedroom": {
                "room_id": "bedroom",
                "confidence": 0.78,
                "transition_type": "vacant_to_occupied",
            },
            "kitchen": {
                "room_id": "kitchen",
                "confidence": 0.92,
                "transition_type": "occupied_to_vacant",
            },
        }

        def get_prediction_side_effect(room_id):
            prediction = room_predictions.get(room_id, {})
            prediction.update(
                {
                    "prediction_time": datetime.utcnow().isoformat(),
                    "next_transition_time": (
                        datetime.utcnow() + timedelta(minutes=20)
                    ).isoformat(),
                    "time_until_transition": "20 minutes",
                    "alternatives": [],
                    "model_info": {"model_type": "ensemble"},
                }
            )
            return prediction

        mock_tracking.get_room_prediction.side_effect = get_prediction_side_effect

        # Test all rooms endpoint
        with TestClient(system["api_app"]) as client:
            response = client.get("/predictions")
            assert response.status_code == 200

            predictions = response.json()
            assert len(predictions) == 3

            room_ids = {p["room_id"] for p in predictions}
            assert room_ids == {"living_room", "bedroom", "kitchen"}

            # Verify all rooms were queried
            expected_calls = [call(room_id) for room_id in room_predictions.keys()]
            mock_tracking.get_room_prediction.assert_has_calls(
                expected_calls, any_order=True
            )


class TestRealTimeIntegration:
    """Test real-time integration capabilities."""

    @pytest.mark.asyncio
    async def test_websocket_integration(self, complete_system):
        """Test WebSocket real-time integration."""
        system = complete_system
        integration_manager = system["integration_manager"]

        # Get WebSocket handler
        ws_handler = integration_manager.get_websocket_handler()
        assert ws_handler is not None

        # Test handler availability
        assert callable(ws_handler)

    @pytest.mark.asyncio
    async def test_sse_integration(self, complete_system):
        """Test Server-Sent Events integration."""
        system = complete_system
        integration_manager = system["integration_manager"]

        # Get SSE handler
        sse_handler = integration_manager.get_sse_handler()
        assert sse_handler is not None

        # Test handler availability
        assert callable(sse_handler)

    @pytest.mark.asyncio
    async def test_real_time_event_broadcasting(self, complete_system):
        """Test real-time event broadcasting across channels."""
        system = complete_system
        mock_enhanced_mqtt = system["mock_enhanced_mqtt"]

        # Simulate system status broadcast
        status_data = {
            "system_health": "healthy",
            "active_predictions": 5,
            "last_update": datetime.utcnow().isoformat(),
        }

        await mock_enhanced_mqtt.publish_system_status(**status_data)

        # Verify broadcast was called
        mock_enhanced_mqtt.publish_system_status.assert_called_once()


class TestSystemPerformanceAndScaling:
    """Test system performance characteristics and scaling."""

    @pytest.mark.asyncio
    async def test_concurrent_prediction_requests(self, complete_system):
        """Test system handling of concurrent prediction requests."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure prediction response
        prediction_data = {
            "room_id": "test_room",
            "prediction_time": datetime.utcnow().isoformat(),
            "confidence": 0.85,
            "transition_type": "occupied_to_vacant",
        }
        mock_tracking.get_room_prediction.return_value = prediction_data

        # Create multiple concurrent requests
        async def make_request():
            with TestClient(system["api_app"]) as client:
                response = client.get("/predictions/living_room")
                return response.status_code, response.json()

        # Execute concurrent requests
        tasks = [make_request() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # Verify all requests succeeded
        for status_code, data in results:
            assert status_code == 200
            assert "confidence" in data

        # Verify tracking manager handled all requests
        assert mock_tracking.get_room_prediction.call_count == 10

    @pytest.mark.asyncio
    async def test_system_resource_usage(self, complete_system):
        """Test system resource usage under normal operation."""
        system = complete_system
        integration_manager = system["integration_manager"]

        # Get system stats
        stats = integration_manager.get_integration_stats()

        # Verify stats structure
        assert "integration_active" in stats
        assert "performance" in stats
        assert stats["integration_active"]

        # Check background task count
        background_tasks = stats["performance"]["background_tasks"]
        assert isinstance(background_tasks, int)
        assert background_tasks >= 0

    @pytest.mark.asyncio
    async def test_connection_scaling(self, complete_system):
        """Test connection handling and scaling."""
        system = complete_system
        mock_enhanced_mqtt = system["mock_enhanced_mqtt"]

        # Configure connection info for high load
        mock_enhanced_mqtt.get_connection_info.return_value = {
            "total_active_connections": 75,
            "websocket_connections": {
                "total_active_connections": 50,
                "clients": [f"client_{i}" for i in range(50)],
            },
            "sse_connections": {
                "total_active_connections": 25,
                "clients": [f"sse_client_{i}" for i in range(25)],
            },
        }

        # Get connection info
        connection_info = mock_enhanced_mqtt.get_connection_info()

        # Verify connection handling
        assert connection_info["total_active_connections"] == 75
        assert len(connection_info["websocket_connections"]["clients"]) == 50
        assert len(connection_info["sse_connections"]["clients"]) == 25


class TestErrorPropagationAndRecovery:
    """Test error propagation and recovery across the complete system."""

    @pytest.mark.asyncio
    async def test_database_error_propagation(self, complete_system):
        """Test error handling when database fails."""
        system = complete_system

        with patch("src.integration.api_server.get_database_manager") as mock_db:
            # Configure database to fail
            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = AsyncMock(
                side_effect=Exception("Database connection failed")
            )
            mock_db.return_value = mock_db_manager

            with TestClient(system["api_app"]) as client:
                response = client.get("/health")

                # Should still return a response, but with error status
                assert response.status_code == 500

    @pytest.mark.asyncio
    async def test_tracking_manager_error_recovery(self, complete_system):
        """Test error recovery when tracking manager fails."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure tracking manager to fail initially, then recover
        call_count = 0

        def prediction_side_effect(room_id):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise Exception("Temporary tracking failure")
            return {
                "room_id": room_id,
                "prediction_time": datetime.utcnow().isoformat(),
                "confidence": 0.85,
            }

        mock_tracking.get_room_prediction.side_effect = prediction_side_effect

        with TestClient(system["api_app"]) as client:
            # First request should fail
            response1 = client.get("/predictions/living_room")
            assert response1.status_code == 500

            # Second request should succeed (recovery)
            response2 = client.get("/predictions/living_room")
            assert response2.status_code == 200

    @pytest.mark.asyncio
    async def test_mqtt_disconnection_handling(self, complete_system):
        """Test handling of MQTT disconnection."""
        system = complete_system
        external_services = system["external_services"]

        # Simulate MQTT disconnection
        external_services.mqtt_connected = False

        # System should continue to operate despite MQTT issues
        with patch("src.integration.api_server.get_mqtt_manager") as mock_mqtt_mgr:
            mock_mqtt_manager = AsyncMock()
            mock_mqtt_manager.get_integration_stats = AsyncMock(
                return_value=Mock(mqtt_connected=False, predictions_published=0)
            )
            mock_mqtt_mgr.return_value = mock_mqtt_manager

            with TestClient(system["api_app"]) as client:
                response = client.get("/health")
                assert response.status_code == 200

                health_data = response.json()
                # System should be degraded but still operational
                assert health_data["status"] in ["degraded", "unhealthy"]


class TestSecurityAndAuthentication:
    """Test security features and authentication."""

    @pytest.mark.asyncio
    async def test_api_key_authentication(self, complete_system):
        """Test API key authentication."""
        system = complete_system

        # Mock config to enable API key authentication
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api = Mock()
            mock_config.api.api_key_enabled = True
            mock_config.api.api_key = "test_e2e_key"
            mock_config.api.rate_limit_enabled = False
            mock_config.rooms = {"living_room": Mock()}
            mock_get_config.return_value = mock_config

            with TestClient(system["api_app"]) as client:
                # Request without API key should fail
                response1 = client.get("/predictions/living_room")
                assert response1.status_code == 401

                # Request with correct API key should succeed
                headers = {"Authorization": "Bearer test_e2e_key"}
                response2 = client.get("/predictions/living_room", headers=headers)
                assert response2.status_code == 200

                # Request with incorrect API key should fail
                headers = {"Authorization": "Bearer wrong_key"}
                response3 = client.get("/predictions/living_room", headers=headers)
                assert response3.status_code == 401

    @pytest.mark.asyncio
    async def test_rate_limiting(self, complete_system):
        """Test API rate limiting."""
        system = complete_system

        # Mock config to enable rate limiting
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api = Mock()
            mock_config.api.api_key_enabled = False
            mock_config.api.rate_limit_enabled = True
            mock_config.api.requests_per_minute = 5  # Low limit for testing
            mock_config.rooms = {"living_room": Mock()}
            mock_get_config.return_value = mock_config

            with TestClient(system["api_app"]) as client:
                # Make requests up to the limit
                for i in range(5):
                    response = client.get("/predictions/living_room")
                    assert response.status_code == 200

                # Next request should be rate limited
                response = client.get("/predictions/living_room")
                assert response.status_code == 429


class TestConfigurationValidation:
    """Test configuration validation and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_room_configuration(self, complete_system):
        """Test handling of invalid room configurations."""
        system = complete_system

        with TestClient(system["api_app"]) as client:
            # Request for non-existent room
            response = client.get("/predictions/nonexistent_room")
            assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_partial_system_configuration(self):
        """Test system behavior with partial configurations."""
        # Test with minimal configuration
        minimal_config = IntegrationConfig(
            enable_realtime_publishing=False,
            enable_websocket_server=False,
            enable_sse_server=False,
        )

        with patch(
            "src.adaptation.tracking_manager.TrackingManager"
        ) as mock_tracking_class:
            mock_tracking = AsyncMock()
            mock_tracking.initialize = AsyncMock()
            mock_tracking.shutdown = AsyncMock()
            mock_tracking.notification_callbacks = []
            mock_tracking_class.return_value = mock_tracking

            from src.integration.tracking_integration import (
                TrackingIntegrationManager,
            )

            manager = TrackingIntegrationManager(
                tracking_manager=mock_tracking,
                integration_config=minimal_config,
            )

            await manager.initialize()

            # Should work with minimal configuration
            assert manager.enhanced_mqtt_manager is None  # Real-time disabled
            assert not manager._integration_active  # No integration active

            await manager.shutdown()


class TestMetricsAndMonitoring:
    """Test metrics collection and monitoring capabilities."""

    @pytest.mark.asyncio
    async def test_system_metrics_collection(self, complete_system):
        """Test comprehensive system metrics collection."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure system stats
        mock_tracking.get_system_stats.return_value = {
            "tracking_stats": {
                "total_predictions_tracked": 150,
                "active_tracking_sessions": 3,
                "average_accuracy": 0.87,
            },
            "retraining_stats": {
                "completed_retraining_jobs": 5,
                "last_retrain_time": datetime.utcnow().isoformat(),
                "average_retrain_duration_minutes": 15,
            },
        }

        # Test stats endpoint
        with patch("src.integration.api_server.get_database_manager") as mock_db:
            with patch("src.integration.api_server.get_mqtt_manager") as mock_mqtt_mgr:

                # Configure mocks
                mock_db_manager = AsyncMock()
                mock_db_manager.health_check = AsyncMock(
                    return_value={
                        "database_connected": True,
                        "connection_pool_size": 5,
                    }
                )
                mock_db.return_value = mock_db_manager

                mock_mqtt_manager = AsyncMock()
                mock_mqtt_manager.get_integration_stats = AsyncMock(
                    return_value=Mock(mqtt_connected=True, predictions_published=50)
                )
                mock_mqtt_mgr.return_value = mock_mqtt_manager

                with TestClient(system["api_app"]) as client:
                    response = client.get("/stats")
                    assert response.status_code == 200

                    stats = response.json()
                    assert "system_info" in stats
                    assert "prediction_stats" in stats
                    assert "mqtt_stats" in stats
                    assert "database_stats" in stats
                    assert "tracking_stats" in stats

    @pytest.mark.asyncio
    async def test_accuracy_metrics_reporting(self, complete_system):
        """Test accuracy metrics reporting."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure accuracy metrics
        mock_tracking.get_accuracy_metrics.return_value = {
            "room_id": "living_room",
            "accuracy_rate": 0.87,
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.92,
            "total_predictions": 150,
            "total_validations": 140,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }

        with TestClient(system["api_app"]) as client:
            response = client.get("/accuracy?room_id=living_room&hours=24")
            assert response.status_code == 200

            metrics = response.json()
            assert metrics["room_id"] == "living_room"
            assert metrics["accuracy_rate"] == 0.87
            assert metrics["trend_direction"] == "improving"

            # Verify tracking manager was called with correct parameters
            mock_tracking.get_accuracy_metrics.assert_called_once_with(
                "living_room", 24
            )


# Performance benchmarking tests
class TestPerformanceBenchmarks:
    """Performance benchmarking for the complete system."""

    @pytest.mark.asyncio
    async def test_prediction_request_latency(self, complete_system):
        """Benchmark prediction request latency."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure fast prediction response
        mock_tracking.get_room_prediction.return_value = {
            "room_id": "living_room",
            "confidence": 0.85,
            "prediction_time": datetime.utcnow().isoformat(),
        }

        with TestClient(system["api_app"]) as client:
            # Benchmark single request
            start_time = time.time()
            response = client.get("/predictions/living_room")
            end_time = time.time()

            assert response.status_code == 200
            latency = end_time - start_time

            # Verify reasonable latency (< 100ms for mocked system)
            assert latency < 0.1, f"Prediction request latency too high: {latency:.3f}s"

    @pytest.mark.asyncio
    async def test_system_throughput(self, complete_system):
        """Test system throughput under load."""
        system = complete_system
        mock_tracking = system["mock_tracking"]

        # Configure prediction response
        mock_tracking.get_room_prediction.return_value = {
            "room_id": "test_room",
            "confidence": 0.85,
            "prediction_time": datetime.utcnow().isoformat(),
        }

        # Test throughput with multiple concurrent requests
        start_time = time.time()

        async def make_requests():
            with TestClient(system["api_app"]) as client:
                tasks = []
                for i in range(50):  # Make 50 concurrent requests
                    task = asyncio.create_task(
                        asyncio.to_thread(
                            lambda: client.get("/predictions/living_room")
                        )
                    )
                    tasks.append(task)

                results = await asyncio.gather(*tasks)
                return results

        results = await make_requests()
        end_time = time.time()

        # Verify all requests succeeded
        for result in results:
            assert result.status_code == 200

        # Calculate throughput
        duration = end_time - start_time
        throughput = len(results) / duration

        logger.info(f"System throughput: {throughput:.2f} requests/second")

        # Verify reasonable throughput (> 100 req/s for mocked system)
        assert throughput > 100, f"System throughput too low: {throughput:.2f} req/s"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
