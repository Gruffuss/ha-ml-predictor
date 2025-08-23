"""Integration tests for API services and external interfaces.

Covers API integration with external systems, service-to-service communication,
and protocol-level integration testing.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import asyncio
from fastapi.testclient import TestClient
import json
import websockets
import paho.mqtt.client as mqtt
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any

from src.integration.api_server import create_app
from src.integration.mqtt_integration_manager import MQTTIntegrationManager
from src.data.ingestion.ha_client import HomeAssistantClient
from src.core.config import (
    SystemConfig,
    MQTTConfig,
    HomeAssistantConfig,
    APIConfig,
    DatabaseConfig,
    PredictionConfig,
    FeaturesConfig,
    LoggingConfig,
    TrackingConfig
)
from src.data.storage.database import DatabaseManager


class TestRESTAPIIntegration:
    """Test REST API integration with system components."""
    
    @pytest.fixture
    def mock_config(self):
        """Create mock system configuration."""
        return SystemConfig(
            home_assistant=HomeAssistantConfig(
                url="http://test-ha:8123",
                token="test-token"
            ),
            database=DatabaseConfig(
                connection_string="sqlite:///test.db"
            ),
            mqtt=MQTTConfig(
                broker="test-mqtt",
                port=1883,
                username="test-user",
                password="test-pass"
            ),
            api=APIConfig(
                enabled=True,
                host="127.0.0.1",
                port=8080
            ),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig()
        )
    
    @pytest.fixture
    def mock_tracking_manager(self):
        """Create mock tracking manager."""
        manager = MagicMock()
        manager.get_current_predictions.return_value = {
            "living_room": {
                "next_transition_time": datetime.now(timezone.utc) + timedelta(minutes=30),
                "transition_type": "occupied",
                "confidence": 0.85,
                "prediction_time": datetime.now(timezone.utc)
            }
        }
        manager.get_accuracy_metrics.return_value = {
            "living_room": {
                "avg_error_minutes": 12.5,
                "accuracy_percentage": 88.2,
                "total_predictions": 150
            }
        }
        manager.get_system_health.return_value = {
            "status": "healthy",
            "components": {
                "database": "healthy",
                "mqtt": "healthy",
                "predictions": "healthy"
            }
        }
        return manager

    @pytest.fixture
    def api_app(self, mock_config, mock_tracking_manager):
        """Create FastAPI app for testing."""
        with patch('src.core.config.get_config', return_value=mock_config):
            # Import handled in create_app()
            app = create_app()
            app.state.tracking_manager = mock_tracking_manager
            return app

    @pytest.fixture
    def client(self, api_app):
        """Create test client."""
        return TestClient(api_app)

    def test_api_system_integration(self, client, mock_tracking_manager):
        """Test API system integration with tracking manager."""
        # Test predictions endpoint
        response = client.get("/api/predictions/living_room")
        assert response.status_code == 200
        
        data = response.json()
        assert "room_id" in data
        assert "prediction_time" in data
        assert "confidence" in data
        assert data["confidence"] == 0.85
        
        # Verify tracking manager was called
        mock_tracking_manager.get_current_predictions.assert_called_once()

    def test_health_endpoint_integration(self, client, mock_tracking_manager):
        """Test health endpoint integration."""
        response = client.get("/api/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert "components" in data
        assert "database" in data["components"]
        
        mock_tracking_manager.get_system_health.assert_called_once()

    def test_accuracy_metrics_integration(self, client, mock_tracking_manager):
        """Test accuracy metrics endpoint integration."""
        response = client.get("/api/accuracy")
        assert response.status_code == 200
        
        data = response.json()
        assert "living_room" in data
        assert data["living_room"]["avg_error_minutes"] == 12.5
        
        mock_tracking_manager.get_accuracy_metrics.assert_called_once()

    def test_authentication_integration(self, client):
        """Test authentication integration with protected endpoints."""
        # Test without authentication (should work for public endpoints)
        response = client.get("/api/health")
        assert response.status_code == 200
        
        # Test protected endpoint (if authentication is configured)
        response = client.post("/api/retrain/living_room")
        # Should either work or return proper authentication error
        assert response.status_code in [200, 401, 403]

    def test_api_error_handling(self, client, mock_tracking_manager):
        """Test API error handling integration."""
        # Configure mock to raise exception
        mock_tracking_manager.get_current_predictions.side_effect = Exception("Test error")
        
        response = client.get("/api/predictions/living_room")
        assert response.status_code == 500
        
        data = response.json()
        assert "detail" in data

    def test_api_performance_monitoring(self, client):
        """Test API performance monitoring integration."""
        import time
        
        # Make multiple requests to test performance
        start_time = time.time()
        responses = []
        
        for _ in range(5):
            response = client.get("/api/health")
            responses.append(response)
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # All requests should succeed
        assert all(r.status_code == 200 for r in responses)
        
        # Should complete within reasonable time
        assert total_time < 5.0  # 5 seconds for 5 requests


class TestWebSocketIntegration:
    """Test WebSocket integration functionality."""
    
    @pytest.fixture
    def mock_websocket_server(self):
        """Mock WebSocket server for testing."""
        server = MagicMock()
        return server

    @pytest.mark.asyncio
    async def test_websocket_system_integration(self, mock_websocket_server):
        """Test WebSocket system integration."""
        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        mock_websocket.send = AsyncMock()
        mock_websocket.recv = AsyncMock(return_value=json.dumps({
            "type": "event",
            "event": {
                "event_type": "state_changed",
                "data": {
                    "entity_id": "binary_sensor.living_room_motion",
                    "new_state": {"state": "on"}
                }
            }
        }))
        
        # Test message handling
        with patch('websockets.connect', return_value=mock_websocket):
            # Simulate WebSocket message processing
            message = await mock_websocket.recv()
            data = json.loads(message)
            
            assert data["type"] == "event"
            assert "entity_id" in data["event"]["data"]

    @pytest.mark.asyncio
    async def test_realtime_communication(self):
        """Test real-time communication patterns."""
        # Mock Home Assistant client
        with patch('src.data.ingestion.ha_client.HomeAssistantClient') as MockClient:
            mock_client = MockClient.return_value
            mock_client.connect = AsyncMock()
            mock_client.subscribe_to_events = AsyncMock()
            
            # Test connection and subscription
            ha_client = HomeAssistantClient()
            await ha_client.connect()
            await ha_client.subscribe_to_events(["binary_sensor.test"])
            
            mock_client.connect.assert_called_once()
            mock_client.subscribe_to_events.assert_called_once()

    def test_connection_resilience(self):
        """Test connection resilience and retry mechanisms."""
        # Test retry logic with connection failures
        with patch('src.data.ingestion.ha_client.HomeAssistantClient') as MockClient:
            mock_client = MockClient.return_value
            
            # Simulate connection failure then success
            mock_client.connect.side_effect = [ConnectionError("Failed"), None]
            
            ha_client = HomeAssistantClient()
            
            # Should handle the connection failure and retry
            try:
                ha_client.connect()
            except ConnectionError:
                # Expected on first attempt
                pass


class TestMQTTIntegration:
    """Test MQTT integration with Home Assistant."""
    
    @pytest.fixture
    def mqtt_config(self):
        """Create MQTT configuration for testing."""
        return MQTTConfig(
            broker="test-mqtt",
            port=1883,
            username="test-user",
            password="test-pass",
            discovery_enabled=True
        )

    @pytest.fixture
    def mock_mqtt_client(self):
        """Mock MQTT client for testing."""
        client = MagicMock()
        client.connect.return_value = 0  # Success
        client.publish.return_value = MagicMock(rc=0)  # Success
        client.is_connected.return_value = True
        return client

    def test_mqtt_ha_integration(self, mqtt_config, mock_mqtt_client):
        """Test MQTT Home Assistant integration."""
        with patch('paho.mqtt.client.Client', return_value=mock_mqtt_client):
            mqtt_manager = MQTTIntegrationManager(mqtt_config)
            
            # Test connection
            result = mqtt_manager.connect()
            assert result is True
            mock_mqtt_client.connect.assert_called_once()

    def test_discovery_integration(self, mqtt_config, mock_mqtt_client):
        """Test Home Assistant MQTT discovery integration."""
        with patch('paho.mqtt.client.Client', return_value=mock_mqtt_client):
            mqtt_manager = MQTTIntegrationManager(mqtt_config)
            mqtt_manager.connect()
            
            # Test discovery message publishing
            room_id = "living_room"
            discovery_config = mqtt_manager.create_discovery_config(room_id)
            
            assert "device" in discovery_config
            assert "unique_id" in discovery_config
            assert discovery_config["device"]["name"] == mqtt_config.device_name

    def test_message_delivery(self, mqtt_config, mock_mqtt_client):
        """Test MQTT message delivery."""
        with patch('paho.mqtt.client.Client', return_value=mock_mqtt_client):
            mqtt_manager = MQTTIntegrationManager(mqtt_config)
            mqtt_manager.connect()
            
            # Test prediction message publishing
            prediction_data = {
                "room_id": "living_room",
                "next_transition_time": datetime.now(timezone.utc).isoformat(),
                "confidence": 0.85
            }
            
            result = mqtt_manager.publish_prediction("living_room", prediction_data)
            assert result is True
            mock_mqtt_client.publish.assert_called()

    def test_mqtt_error_handling(self, mqtt_config):
        """Test MQTT error handling integration."""
        mock_client = MagicMock()
        mock_client.connect.return_value = 1  # Connection failed
        
        with patch('paho.mqtt.client.Client', return_value=mock_client):
            mqtt_manager = MQTTIntegrationManager(mqtt_config)
            
            # Should handle connection failure gracefully
            result = mqtt_manager.connect()
            assert result is False


class TestExternalServiceIntegration:
    """Test integration with external services."""
    
    @pytest.fixture
    def ha_config(self):
        """Create Home Assistant configuration."""
        return HomeAssistantConfig(
            url="http://test-ha:8123",
            token="test-token",
            websocket_timeout=30,
            api_timeout=10
        )

    def test_ha_api_integration(self, ha_config):
        """Test Home Assistant API integration."""
        with patch('aiohttp.ClientSession') as MockSession:
            mock_session = MockSession.return_value.__aenter__.return_value
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json.return_value = [{"entity_id": "sensor.test", "state": "on"}]
            mock_session.get.return_value.__aenter__.return_value = mock_response
            
            ha_client = HomeAssistantClient(ha_config)
            
            # Test would require async context - verify setup
            assert ha_client.config.url == "http://test-ha:8123"
            assert ha_client.config.token == "test-token"

    @pytest.mark.asyncio
    async def test_service_communication(self):
        """Test service-to-service communication patterns."""
        # Test database manager communication
        with patch('src.data.storage.database.create_async_engine') as mock_engine:
            mock_engine.return_value = MagicMock()
            
            db_config = DatabaseConfig(connection_string="sqlite:///test.db")
            db_manager = DatabaseManager(db_config)
            
            # Test initialization
            await db_manager.initialize()
            assert db_manager.engine is not None

    def test_error_handling_integration(self):
        """Test error handling across service integrations."""
        # Test database connection error handling
        with patch('src.data.storage.database.create_async_engine') as mock_engine:
            mock_engine.side_effect = Exception("Connection failed")
            
            db_config = DatabaseConfig(connection_string="invalid://connection")
            db_manager = DatabaseManager(db_config)
            
            # Should handle initialization errors gracefully
            with pytest.raises(Exception):
                asyncio.run(db_manager.initialize())

    def test_service_health_monitoring(self):
        """Test health monitoring across integrated services."""
        # Mock multiple services
        services = {
            "database": MagicMock(health_check=MagicMock(return_value=True)),
            "mqtt": MagicMock(is_connected=MagicMock(return_value=True)),
            "ha_client": MagicMock(is_connected=MagicMock(return_value=True))
        }
        
        # Test health status aggregation
        health_status = {}
        for service_name, service in services.items():
            if hasattr(service, 'health_check'):
                health_status[service_name] = service.health_check()
            elif hasattr(service, 'is_connected'):
                health_status[service_name] = service.is_connected()
        
        # All services should be healthy
        assert all(health_status.values())
        assert len(health_status) == 3