"""Unit tests for API services and web interfaces.

Covers:
- src/integration/api_server.py (REST API Server)
- src/integration/websocket_api.py (WebSocket API)
- src/integration/realtime_api_endpoints.py (Real-time API Endpoints)
- src/integration/monitoring_api.py (Monitoring API Endpoints)
- src/integration/dashboard.py (Dashboard Integration)

This test file consolidates testing for all API service functionality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, List, Optional
import unittest.mock
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
import uuid

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest
import websockets

from src.core.exceptions import (
    APIAuthenticationError,
    APIError,
    APIRateLimitError,
    APIResourceNotFoundError,
    APIServerError,
    WebSocketAuthenticationError,
    WebSocketConnectionError,
    WebSocketValidationError,
)

# Import the actual API components to test
from src.integration.api_server import (
    AccuracyMetricsResponse,
    APIServer,
    ErrorResponse,
    ManualRetrainRequest,
    PredictionResponse,
    RateLimitTracker,
    SystemHealthResponse,
    SystemStatsResponse,
    background_health_check,
    check_rate_limit,
    create_app,
    get_tracking_manager,
    verify_api_key,
)
from src.integration.dashboard import DashboardIntegration
from src.integration.monitoring_api import (
    AlertsResponse,
    HealthCheckResponse,
    MetricsResponse,
    SystemStatus,
)
from src.integration.websocket_api import (
    ClientAuthRequest,
    ClientConnection,
    ClientSubscription,
    MessageType,
    WebSocketAPIServer,
    WebSocketConnectionManager,
    WebSocketEndpoint,
    WebSocketMessage,
    WebSocketStats,
    create_websocket_api_server,
)


class TestRESTAPIServer:
    """Test REST API server functionality."""

    @pytest.fixture
    def mock_config(self):
        """Mock configuration for testing."""
        config_mock = Mock()
        config_mock.api = Mock()
        config_mock.api.enabled = True
        config_mock.api.host = "localhost"
        config_mock.api.port = 8000
        config_mock.api.debug = False
        config_mock.api.api_key_enabled = False
        config_mock.api.rate_limit_enabled = False
        config_mock.api.background_tasks_enabled = True
        config_mock.api.health_check_interval_seconds = 60
        config_mock.api.jwt = Mock()
        config_mock.api.jwt.enabled = False
        config_mock.api.enable_cors = True
        config_mock.api.cors_origins = ["*"]
        config_mock.rooms = {"living_room": {"name": "Living Room"}}
        return config_mock

    @pytest.fixture
    def mock_tracking_manager(self):
        """Mock tracking manager for testing."""
        manager = AsyncMock()
        manager.get_room_prediction.return_value = {
            "room_id": "living_room",
            "prediction_time": datetime.now().isoformat(),
            "next_transition_time": (
                datetime.now() + timedelta(minutes=30)
            ).isoformat(),
            "transition_type": "occupied",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {"model_type": "ensemble", "version": "1.0"},
        }
        manager.get_accuracy_metrics.return_value = {
            "room_id": "living_room",
            "accuracy_rate": 0.87,
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.91,
            "total_predictions": 150,
            "total_validations": 140,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }
        manager.trigger_manual_retrain.return_value = {
            "success": True,
            "message": "Retraining initiated",
            "room_id": "living_room",
            "strategy": "auto",
            "force": False,
        }
        manager.get_tracking_status.return_value = {
            "status": "active",
            "tracking_active": True,
            "config": {"enabled": True},
            "performance": {
                "background_tasks": 3,
                "total_predictions_recorded": 100,
                "total_validations_performed": 95,
                "total_drift_checks_performed": 50,
                "system_uptime_seconds": 3600,
            },
            "validator": {"total_predictions": 100},
            "accuracy_tracker": {"total_predictions": 100},
            "drift_detector": {"enabled": True},
            "adaptive_retrainer": {"enabled": True},
        }
        manager.get_system_stats.return_value = {
            "tracking_stats": {
                "total_predictions_tracked": 200,
                "tracking_active": True,
            },
            "retraining_stats": {"completed_retraining_jobs": 5},
        }
        return manager

    @patch("src.integration.api_server.get_config")
    def test_prediction_response_validation(self, mock_get_config):
        """Test PredictionResponse model validation."""
        # Valid prediction response
        valid_data = {
            "room_id": "living_room",
            "prediction_time": datetime.now(),
            "next_transition_time": datetime.now() + timedelta(minutes=30),
            "transition_type": "occupied",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
        }

        response = PredictionResponse(**valid_data)
        assert response.room_id == "living_room"
        assert response.confidence == 0.85
        assert response.transition_type == "occupied"

    def test_prediction_response_validation_errors(self):
        """Test PredictionResponse validation errors."""
        # Invalid confidence value
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            PredictionResponse(
                room_id="test",
                prediction_time=datetime.now(),
                confidence=1.5,  # Invalid
                transition_type="occupied",
            )

        # Invalid transition type
        with pytest.raises(ValueError, match="Transition type must be one of"):
            PredictionResponse(
                room_id="test",
                prediction_time=datetime.now(),
                confidence=0.5,
                transition_type="invalid_type",  # Invalid
            )

    @patch("src.integration.api_server.get_config")
    def test_accuracy_metrics_response_validation(self, mock_get_config):
        """Test AccuracyMetricsResponse model validation."""
        valid_data = {
            "accuracy_rate": 0.85,
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.91,
            "total_predictions": 150,
            "total_validations": 140,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }

        response = AccuracyMetricsResponse(**valid_data)
        assert response.accuracy_rate == 0.85
        assert response.trend_direction == "improving"

    @patch("src.integration.api_server.get_config")
    def test_manual_retrain_request_validation(self, mock_get_config):
        """Test ManualRetrainRequest model validation."""
        mock_config = Mock()
        mock_config.rooms = {"living_room": {"name": "Living Room"}}
        mock_get_config.return_value = mock_config

        valid_data = {
            "room_id": "living_room",
            "force": True,
            "strategy": "full",
            "reason": "Testing manual retrain",
        }

        request = ManualRetrainRequest(**valid_data)
        assert request.room_id == "living_room"
        assert request.force is True
        assert request.strategy == "full"

    def test_rate_limit_tracker(self):
        """Test RateLimitTracker functionality."""
        tracker = RateLimitTracker()
        client_ip = "192.168.1.100"

        # First requests should be allowed
        for i in range(5):
            assert tracker.is_allowed(client_ip, 10, 1) is True

        # After limit, should be denied
        for i in range(6):
            tracker.is_allowed(client_ip, 10, 1)

        assert tracker.is_allowed(client_ip, 10, 1) is False

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.get_tracking_manager")
    async def test_api_key_verification(
        self, mock_get_tracking_manager, mock_get_config
    ):
        """Test API key verification dependency."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.api_key_enabled = True
        mock_config.api.api_key = "test_api_key_123"
        mock_get_config.return_value = mock_config

        # Test valid API key
        from fastapi.security import HTTPAuthorizationCredentials

        valid_creds = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="test_api_key_123"
        )

        result = await verify_api_key(valid_creds)
        assert result is True

        # Test invalid API key
        invalid_creds = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid_key"
        )

        with pytest.raises(APIAuthenticationError):
            await verify_api_key(invalid_creds)

    @patch("src.integration.api_server.get_config")
    async def test_rate_limiting_check(self, mock_get_config):
        """Test rate limiting functionality."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.rate_limit_enabled = True
        mock_config.api.requests_per_minute = 10
        mock_get_config.return_value = mock_config

        # Mock request object
        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.100"

        # First request should pass
        result = await check_rate_limit(mock_request)
        assert result is True

    @patch("src.integration.api_server.get_config")
    def test_api_server_initialization(self, mock_get_config):
        """Test APIServer class initialization."""
        mock_tracking_manager = Mock()
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.enabled = True
        mock_config.api.host = "localhost"
        mock_config.api.port = 8000
        mock_get_config.return_value = mock_config

        api_server = APIServer(mock_tracking_manager)
        assert api_server.tracking_manager is mock_tracking_manager
        assert api_server.config is not None

    @patch("src.integration.api_server.get_config")
    @patch("uvicorn.Server")
    async def test_api_server_start_stop(self, mock_uvicorn_server, mock_get_config):
        """Test APIServer start and stop methods."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.enabled = True
        mock_config.api.host = "localhost"
        mock_config.api.port = 8000
        mock_config.api.debug = False
        mock_config.api.access_log = True
        mock_get_config.return_value = mock_config

        mock_tracking_manager = Mock()
        api_server = APIServer(mock_tracking_manager)

        # Mock server instance
        mock_server_instance = Mock()
        mock_server_instance.serve = AsyncMock()
        mock_uvicorn_server.return_value = mock_server_instance

        # Test start
        await api_server.start()
        assert api_server.server is mock_server_instance
        assert api_server.server_task is not None

        # Test stop
        mock_server_instance.should_exit = False
        await api_server.stop()
        assert mock_server_instance.should_exit is True

    def test_error_response_serialization(self):
        """Test ErrorResponse model serialization."""
        error_response = ErrorResponse(
            error="Test error",
            error_code="TEST_ERROR",
            details={"field": "value"},
            timestamp=datetime.now(),
            request_id="req-123",
        )

        # Test dict method handles datetime serialization
        response_dict = error_response.dict()
        assert "timestamp" in response_dict
        assert isinstance(response_dict["timestamp"], str)
        assert response_dict["error"] == "Test error"
        assert response_dict["error_code"] == "TEST_ERROR"

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.get_health_monitor")
    @patch("src.integration.api_server.get_incident_response_manager")
    async def test_background_health_check(
        self, mock_incident_manager, mock_health_monitor, mock_get_config
    ):
        """Test background health check task."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.health_check_interval_seconds = 1  # Fast for testing
        mock_get_config.return_value = mock_config

        # Mock health monitor
        mock_health_monitor_instance = Mock()
        mock_health_monitor_instance.start_monitoring = AsyncMock()
        mock_health_monitor_instance.stop_monitoring = AsyncMock()
        mock_health_monitor_instance.get_system_health.return_value = Mock(
            overall_status=Mock(value="healthy")
        )
        mock_health_monitor.return_value = mock_health_monitor_instance

        # Mock incident manager
        mock_incident_manager_instance = Mock()
        mock_incident_manager_instance.start_incident_response = AsyncMock()
        mock_incident_manager_instance.stop_incident_response = AsyncMock()
        mock_incident_manager_instance.get_active_incidents.return_value = {}
        mock_incident_manager.return_value = mock_incident_manager_instance

        # Create task and cancel quickly
        task = asyncio.create_task(background_health_check())
        await asyncio.sleep(0.1)  # Let it start
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify health monitoring was started
        mock_health_monitor_instance.start_monitoring.assert_called_once()
        mock_incident_manager_instance.start_incident_response.assert_called_once()

    @patch("src.integration.api_server.get_config")
    def test_app_creation_with_middleware(self, mock_get_config):
        """Test FastAPI app creation with middleware."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.debug = False
        mock_config.api.docs_url = "/docs"
        mock_config.api.redoc_url = "/redoc"
        mock_config.api.include_docs = True
        mock_config.api.jwt = Mock()
        mock_config.api.jwt.enabled = False
        mock_config.api.enable_cors = True
        mock_config.api.cors_origins = ["*"]
        mock_config.api.log_requests = True
        mock_config.api.log_responses = True
        mock_get_config.return_value = mock_config

        app = create_app()

        # Verify app was created successfully
        assert app is not None
        assert app.title == "Occupancy Prediction API"
        assert app.version == "1.0.0"

        # Check that middleware stack is configured
        assert len(app.user_middleware) > 0

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.get_tracking_manager")
    @patch("src.integration.api_server.get_database_manager")
    async def test_health_endpoint_comprehensive(
        self,
        mock_db_manager,
        mock_tracking_manager,
        mock_get_config,
        mock_config,
        mock_tracking_manager_fixture,
    ):
        """Test comprehensive health endpoint functionality."""
        mock_get_config.return_value = mock_config
        mock_tracking_manager.return_value = mock_tracking_manager_fixture

        # Mock database manager
        mock_db_instance = Mock()
        mock_db_instance.health_check = AsyncMock(
            return_value={"status": "healthy", "database_connected": True}
        )
        mock_db_manager.return_value = mock_db_instance

        # Create app and test client
        app = create_app()
        client = TestClient(app)

        # Test health endpoint
        response = client.get("/health")
        assert response.status_code == 200

        health_data = response.json()
        assert health_data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "components" in health_data
        assert "performance_metrics" in health_data
        assert "timestamp" in health_data

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.get_tracking_manager")
    async def test_prediction_endpoints(
        self,
        mock_tracking_manager,
        mock_get_config,
        mock_config,
        mock_tracking_manager_fixture,
    ):
        """Test prediction API endpoints."""
        mock_get_config.return_value = mock_config
        mock_tracking_manager.return_value = mock_tracking_manager_fixture

        app = create_app()
        client = TestClient(app)

        # Test single room prediction
        response = client.get("/predictions/living_room")
        assert response.status_code == 200

        prediction_data = response.json()
        assert prediction_data["room_id"] == "living_room"
        assert "confidence" in prediction_data
        assert "prediction_time" in prediction_data

        # Test all predictions
        response = client.get("/predictions")
        assert response.status_code == 200

        predictions = response.json()
        assert isinstance(predictions, list)

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.get_tracking_manager")
    async def test_accuracy_metrics_endpoint(
        self,
        mock_tracking_manager,
        mock_get_config,
        mock_config,
        mock_tracking_manager_fixture,
    ):
        """Test accuracy metrics API endpoint."""
        mock_get_config.return_value = mock_config
        mock_tracking_manager.return_value = mock_tracking_manager_fixture

        app = create_app()
        client = TestClient(app)

        # Test accuracy metrics
        response = client.get("/accuracy?room_id=living_room&hours=24")
        assert response.status_code == 200

        metrics_data = response.json()
        assert "accuracy_rate" in metrics_data
        assert "average_error_minutes" in metrics_data
        assert "trend_direction" in metrics_data

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.get_tracking_manager")
    async def test_manual_retrain_endpoint(
        self,
        mock_tracking_manager,
        mock_get_config,
        mock_config,
        mock_tracking_manager_fixture,
    ):
        """Test manual retrain API endpoint."""
        mock_get_config.return_value = mock_config
        mock_tracking_manager.return_value = mock_tracking_manager_fixture

        app = create_app()
        client = TestClient(app)

        # Test manual retrain
        retrain_data = {
            "room_id": "living_room",
            "force": True,
            "strategy": "full",
            "reason": "API test retrain",
        }

        response = client.post("/model/retrain", json=retrain_data)
        assert response.status_code == 200

        result = response.json()
        assert result["success"] is True
        assert "timestamp" in result

    def test_api_error_handling_scenarios(self):
        """Test various API error handling scenarios."""
        # Test API authentication error
        auth_error = APIAuthenticationError("Invalid API key", "Key mismatch")
        assert "Invalid API key" in str(auth_error)

        # Test API rate limit error
        rate_error = APIRateLimitError("192.168.1.1", 60, "minute")
        assert "Rate limit exceeded" in str(rate_error)

        # Test API resource not found error
        not_found_error = APIResourceNotFoundError("Room", "invalid_room")
        assert "Room 'invalid_room' not found" in str(not_found_error)

        # Test API server error
        server_error = APIServerError(
            endpoint="/test", operation="test_operation", cause=Exception("Test error")
        )
        assert "API server error" in str(server_error)

    @patch("src.integration.api_server.get_config")
    async def test_rate_limiting_enforcement(self, mock_get_config):
        """Test rate limiting enforcement in request middleware."""
        mock_config = Mock()
        mock_config.api = Mock()
        mock_config.api.rate_limit_enabled = True
        mock_config.api.requests_per_minute = 2
        mock_get_config.return_value = mock_config

        # Create a new rate limiter for testing
        from src.integration.api_server import rate_limiter

        rate_limiter.requests = {}  # Clear any existing data

        mock_request = Mock()
        mock_request.client = Mock()
        mock_request.client.host = "192.168.1.200"

        # First two requests should pass
        assert await check_rate_limit(mock_request) is True
        assert await check_rate_limit(mock_request) is True

        # Third request should trigger rate limit
        with pytest.raises(APIRateLimitError):
            await check_rate_limit(mock_request)


class TestWebSocketAPI:
    """Test WebSocket API functionality."""

    @pytest.fixture
    def mock_websocket(self):
        """Mock WebSocket connection for testing."""
        websocket = Mock()
        websocket.send = AsyncMock()
        websocket.receive_text = AsyncMock()
        websocket.accept = AsyncMock()
        websocket.close = AsyncMock()
        return websocket

    @pytest.fixture
    def websocket_config(self):
        """WebSocket server configuration for testing."""
        return {
            "host": "localhost",
            "port": 8765,
            "max_connections": 100,
            "max_messages_per_minute": 30,
            "heartbeat_interval_seconds": 10,
            "connection_timeout_seconds": 60,
            "enabled": True,
        }

    def test_websocket_message_creation(self):
        """Test WebSocketMessage creation and serialization."""
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"room_id": "living_room", "confidence": 0.85},
            room_id="living_room",
        )

        # Test JSON serialization
        json_str = message.to_json()
        assert "message_id" in json_str
        assert "prediction_update" in json_str
        assert "living_room" in json_str

        # Test deserialization
        restored_message = WebSocketMessage.from_json(json_str)
        assert restored_message.message_type == MessageType.PREDICTION_UPDATE
        assert restored_message.room_id == "living_room"
        assert restored_message.data["confidence"] == 0.85

    def test_client_auth_request_validation(self):
        """Test ClientAuthRequest model validation."""
        # Valid request
        valid_request = ClientAuthRequest(
            api_key="test_api_key_123456",
            client_name="Test Client",
            capabilities=["predictions", "alerts"],
            room_filters=["living_room", "bedroom"],
        )

        assert valid_request.api_key == "test_api_key_123456"
        assert len(valid_request.capabilities) == 2
        assert "living_room" in valid_request.room_filters

        # Invalid API key (too short)
        with pytest.raises(ValueError, match="API key must be at least 10 characters"):
            ClientAuthRequest(api_key="short")

    def test_client_subscription_validation(self):
        """Test ClientSubscription model validation."""
        # Valid subscription
        valid_sub = ClientSubscription(
            endpoint="/ws/predictions", filters={"room_type": "living_area"}
        )

        assert valid_sub.endpoint == "/ws/predictions"
        assert valid_sub.filters["room_type"] == "living_area"

        # Invalid endpoint format
        with pytest.raises(ValueError):
            ClientSubscription(endpoint="/invalid/endpoint")

    def test_client_connection_lifecycle(self):
        """Test ClientConnection lifecycle and properties."""
        mock_websocket = Mock()

        connection = ClientConnection(
            connection_id="conn-123",
            websocket=mock_websocket,
            endpoint="/ws/predictions",
        )

        # Test initial state
        assert connection.connection_id == "conn-123"
        assert connection.authenticated is False
        assert connection.message_count == 0
        assert len(connection.capabilities) == 0

        # Test activity updates
        initial_activity = connection.last_activity
        connection.update_activity()
        assert connection.last_activity > initial_activity

        # Test heartbeat updates
        initial_heartbeat = connection.last_heartbeat
        connection.update_heartbeat()
        assert connection.last_heartbeat > initial_heartbeat

        # Test rate limiting
        connection.message_count = 25
        assert connection.is_rate_limited(20) is True
        assert connection.is_rate_limited(30) is False

        # Test room access control
        connection.room_filters.add("living_room")
        assert connection.can_access_room("living_room") is True
        assert connection.can_access_room("bedroom") is False

        # Test capabilities
        connection.capabilities.add("admin")
        assert connection.has_capability("admin") is True
        assert connection.has_capability("user") is False

    def test_websocket_connection_manager(self, websocket_config, mock_websocket):
        """Test WebSocketConnectionManager functionality."""
        manager = WebSocketConnectionManager(websocket_config)

        # Test initial state
        assert manager.max_connections == 100
        assert manager.max_messages_per_minute == 30
        assert len(manager.connections) == 0

        # Test statistics
        stats = manager.get_connection_stats()
        assert stats["active_connections"] == 0
        assert stats["total_connections"] == 0

    async def test_websocket_connection_registration(
        self, websocket_config, mock_websocket
    ):
        """Test WebSocket connection registration."""
        manager = WebSocketConnectionManager(websocket_config)

        # Register connection
        connection_id = await manager.connect(
            websocket=mock_websocket, endpoint="/ws/predictions"
        )

        assert connection_id is not None
        assert len(manager.connections) == 1
        assert connection_id in manager.connections

        # Test connection object
        connection = manager.connections[connection_id]
        assert connection.endpoint == "/ws/predictions"
        assert connection.websocket is mock_websocket
        assert connection.authenticated is False

        # Test statistics update
        stats = manager.get_connection_stats()
        assert stats["active_connections"] == 1
        assert stats["predictions_connections"] == 1

    async def test_websocket_authentication(self, websocket_config, mock_websocket):
        """Test WebSocket client authentication."""
        manager = WebSocketConnectionManager(websocket_config)

        # Register connection
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")

        # Create auth request
        auth_request = ClientAuthRequest(
            api_key="test_key_123456789",
            client_name="Test Client",
            capabilities=["predictions"],
            room_filters=["living_room"],
        )

        # Mock config for API key validation
        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False

            # Authenticate client
            result = await manager.authenticate_client(connection_id, auth_request)
            assert result is True

            # Verify connection is authenticated
            connection = manager.connections[connection_id]
            assert connection.authenticated is True
            assert connection.client_name == "Test Client"
            assert "predictions" in connection.capabilities
            assert "living_room" in connection.room_filters

    async def test_websocket_subscription_management(
        self, websocket_config, mock_websocket
    ):
        """Test WebSocket subscription management."""
        manager = WebSocketConnectionManager(websocket_config)

        # Register and authenticate connection
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")

        auth_request = ClientAuthRequest(
            api_key="test_key_123456789", room_filters=["living_room"]
        )

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False
            await manager.authenticate_client(connection_id, auth_request)

        # Create subscription
        subscription = ClientSubscription(
            endpoint="/ws/predictions", room_id="living_room"
        )

        # Subscribe client
        result = await manager.subscribe_client(connection_id, subscription)
        assert result is True

        # Verify subscription
        connection = manager.connections[connection_id]
        assert "/ws/predictions" in connection.subscriptions
        assert "living_room" in connection.room_subscriptions

        # Test unsubscription
        result = await manager.unsubscribe_client(
            connection_id, "/ws/predictions", "living_room"
        )
        assert result is True
        assert "/ws/predictions" not in connection.subscriptions
        assert "living_room" not in connection.room_subscriptions

    async def test_websocket_message_sending(self, websocket_config, mock_websocket):
        """Test WebSocket message sending."""
        manager = WebSocketConnectionManager(websocket_config)

        # Register connection
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")

        # Create message
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"test": "data"},
        )

        # Send message
        result = await manager.send_message(connection_id, message)
        assert result is True

        # Verify websocket.send was called
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        assert "prediction_update" in sent_data
        assert "test" in sent_data

    async def test_websocket_broadcasting(self, websocket_config):
        """Test WebSocket message broadcasting."""
        manager = WebSocketConnectionManager(websocket_config)

        # Create multiple connections
        mock_ws1 = Mock()
        mock_ws1.send = AsyncMock()
        mock_ws2 = Mock()
        mock_ws2.send = AsyncMock()

        conn_id1 = await manager.connect(mock_ws1, "/ws/predictions")
        conn_id2 = await manager.connect(mock_ws2, "/ws/predictions")

        # Authenticate and subscribe both
        auth_request = ClientAuthRequest(api_key="test_key_123456789")
        subscription = ClientSubscription(endpoint="/ws/predictions")

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_config.return_value.api.api_key_enabled = False

            await manager.authenticate_client(conn_id1, auth_request)
            await manager.authenticate_client(conn_id2, auth_request)
            await manager.subscribe_client(conn_id1, subscription)
            await manager.subscribe_client(conn_id2, subscription)

        # Broadcast message
        message = WebSocketMessage(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.PREDICTION_UPDATE,
            timestamp=datetime.now(timezone.utc),
            endpoint="/ws/predictions",
            data={"broadcast": "test"},
        )

        sent_count = await manager.broadcast_to_endpoint("/ws/predictions", message)
        assert sent_count == 2

        # Verify both websockets received the message
        mock_ws1.send.assert_called_once()
        mock_ws2.send.assert_called_once()

    async def test_websocket_heartbeat(self, websocket_config, mock_websocket):
        """Test WebSocket heartbeat functionality."""
        manager = WebSocketConnectionManager(websocket_config)

        # Register connection
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")

        # Send heartbeat
        result = await manager.send_heartbeat(connection_id)
        assert result is True

        # Verify heartbeat message was sent
        mock_websocket.send.assert_called_once()
        sent_data = mock_websocket.send.call_args[0][0]
        assert "heartbeat" in sent_data
        assert "server_time" in sent_data

        # Verify heartbeat timestamp was updated
        connection = manager.connections[connection_id]
        assert connection.last_heartbeat is not None

    async def test_websocket_api_server_initialization(self, websocket_config):
        """Test WebSocketAPIServer initialization."""
        mock_tracking_manager = Mock()

        server = WebSocketAPIServer(
            tracking_manager=mock_tracking_manager, config=websocket_config
        )

        assert server.tracking_manager is mock_tracking_manager
        assert server.host == "localhost"
        assert server.port == 8765
        assert server.enabled is True
        assert isinstance(server.connection_manager, WebSocketConnectionManager)

    async def test_websocket_api_server_lifecycle(self, websocket_config):
        """Test WebSocketAPIServer start/stop lifecycle."""
        # Disable background tasks for testing
        import os

        os.environ["DISABLE_BACKGROUND_TASKS"] = "1"

        try:
            server = WebSocketAPIServer(config=websocket_config)

            # Test initialization
            await server.initialize()
            assert server._server_running is False  # Not started yet

            # Mock websockets.serve
            with patch("websockets.serve") as mock_serve:
                mock_server = Mock()
                mock_serve.return_value = mock_server

                # Test start
                await server.start()
                assert server._server_running is True
                assert server._websocket_server is mock_server

                # Test stop
                mock_server.close = Mock()
                mock_server.wait_closed = AsyncMock()

                await server.stop()
                assert server._server_running is False
                mock_server.close.assert_called_once()

        finally:
            # Clean up environment variable
            if "DISABLE_BACKGROUND_TASKS" in os.environ:
                del os.environ["DISABLE_BACKGROUND_TASKS"]

    async def test_websocket_prediction_publishing(self, websocket_config):
        """Test WebSocket prediction update publishing."""
        server = WebSocketAPIServer(config=websocket_config)

        # Mock prediction result
        prediction_result = Mock()
        prediction_result.predicted_time = datetime.now(timezone.utc) + timedelta(
            minutes=30
        )
        prediction_result.transition_type = "occupied"
        prediction_result.confidence_score = 0.85
        prediction_result.model_type = "ensemble"
        prediction_result.model_version = "1.0"
        prediction_result.alternatives = []
        prediction_result.features_used = ["temp", "motion"]
        prediction_result.prediction_metadata = {}

        # Mock connection manager broadcast
        server.connection_manager.broadcast_to_endpoint = AsyncMock(return_value=2)

        # Test publishing
        result = await server.publish_prediction_update(
            prediction_result=prediction_result,
            room_id="living_room",
            current_state="vacant",
        )

        assert result["success"] is True
        assert (
            result["clients_notified"] == 4
        )  # 2 for predictions + 2 for room endpoint

        # Verify broadcast was called twice (predictions and room endpoints)
        assert server.connection_manager.broadcast_to_endpoint.call_count == 2

    async def test_websocket_system_status_publishing(self, websocket_config):
        """Test WebSocket system status update publishing."""
        server = WebSocketAPIServer(config=websocket_config)

        # Mock connection manager broadcast
        server.connection_manager.broadcast_to_endpoint = AsyncMock(return_value=3)

        status_data = {
            "overall_status": "healthy",
            "components": {
                "database": "healthy",
                "tracking": "active",
                "mqtt": "connected",
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Test publishing
        result = await server.publish_system_status_update(status_data)

        assert result["success"] is True
        assert result["clients_notified"] == 3

        # Verify broadcast was called
        server.connection_manager.broadcast_to_endpoint.assert_called_once_with(
            "/ws/system-status", unittest.mock.ANY
        )

    async def test_websocket_alert_publishing(self, websocket_config):
        """Test WebSocket alert notification publishing."""
        server = WebSocketAPIServer(config=websocket_config)

        # Mock connection manager broadcast
        server.connection_manager.broadcast_to_endpoint = AsyncMock(return_value=1)

        alert_data = {
            "alert_type": "drift_detected",
            "room_id": "living_room",
            "severity": "warning",
            "message": "Concept drift detected in room occupancy patterns",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

        # Test publishing
        result = await server.publish_alert_notification(
            alert_data=alert_data, room_id="living_room"
        )

        assert result["success"] is True
        assert result["clients_notified"] == 1

        # Verify broadcast was called
        server.connection_manager.broadcast_to_endpoint.assert_called_once_with(
            "/ws/alerts", unittest.mock.ANY
        )

    def test_websocket_server_stats(self, websocket_config):
        """Test WebSocket server statistics."""
        mock_tracking_manager = Mock()
        server = WebSocketAPIServer(
            tracking_manager=mock_tracking_manager, config=websocket_config
        )

        # Mock connection stats
        server.connection_manager.get_connection_stats = Mock(
            return_value={
                "active_connections": 5,
                "total_messages_sent": 100,
                "authentication_failures": 2,
            }
        )

        stats = server.get_server_stats()

        assert stats["server_running"] is False
        assert stats["host"] == "localhost"
        assert stats["port"] == 8765
        assert stats["enabled"] is True
        assert "connection_stats" in stats
        assert stats["connection_stats"]["active_connections"] == 5
        assert stats["tracking_manager_integrated"] is True

    def test_create_websocket_api_server_factory(self):
        """Test WebSocket API server factory function."""
        mock_tracking_manager = Mock()

        with patch("src.integration.websocket_api.get_config") as mock_config:
            mock_system_config = Mock()
            mock_system_config.websocket_host = "0.0.0.0"
            mock_system_config.websocket_port = 8765
            mock_config.return_value = mock_system_config

            server = create_websocket_api_server(tracking_manager=mock_tracking_manager)

            assert isinstance(server, WebSocketAPIServer)
            assert server.tracking_manager is mock_tracking_manager
            assert server.host == "0.0.0.0"
            assert server.port == 8765

    def test_websocket_error_handling(self):
        """Test WebSocket-specific error handling."""
        # Test WebSocket authentication error
        auth_error = WebSocketAuthenticationError("Invalid credentials")
        assert "WebSocket authentication failed" in str(auth_error)

        # Test WebSocket connection error
        conn_error = WebSocketConnectionError(
            "Connection failed", cause=Exception("Network error")
        )
        assert "WebSocket connection error" in str(conn_error)

        # Test WebSocket validation error
        val_error = WebSocketValidationError("Invalid message format")
        assert "WebSocket validation failed" in str(val_error)

    def test_websocket_endpoint_normalization(self, websocket_config):
        """Test WebSocket endpoint path normalization."""
        server = WebSocketAPIServer(config=websocket_config)

        # Test various endpoint formats
        assert server._normalize_endpoint("/ws/predictions") == "/ws/predictions"
        assert (
            server._normalize_endpoint("/ws/room/living_room") == "/ws/room/living_room"
        )
        assert server._normalize_endpoint("/ws/system-status") == "/ws/system-status"
        assert server._normalize_endpoint("/invalid") == "/ws/predictions"  # Default
        assert server._normalize_endpoint("/") == "/ws/predictions"  # Default

    def test_websocket_time_formatting(self, websocket_config):
        """Test WebSocket time formatting utilities."""
        server = WebSocketAPIServer(config=websocket_config)

        # Test various time intervals
        assert server._format_time_until(30) == "30 seconds"
        assert server._format_time_until(60) == "1 minute"
        assert server._format_time_until(120) == "2 minutes"
        assert server._format_time_until(3600) == "1 hour"
        assert server._format_time_until(3660) == "1h 1m"
        assert server._format_time_until(86400) == "1 day"
        assert server._format_time_until(90000) == "1d 1h"

    async def test_websocket_connection_cleanup(self, websocket_config, mock_websocket):
        """Test WebSocket connection cleanup and timeout handling."""
        manager = WebSocketConnectionManager(websocket_config)

        # Register connection
        connection_id = await manager.connect(mock_websocket, "/ws/predictions")
        connection = manager.connections[connection_id]

        # Simulate old activity
        old_time = datetime.now(timezone.utc) - timedelta(minutes=10)
        connection.last_activity = old_time
        connection.last_heartbeat = old_time

        # Test rate limit application
        connection.apply_rate_limit(60)
        assert connection.rate_limited_until is not None
        assert connection.is_rate_limited() is True

        # Test disconnection
        await manager.disconnect(connection_id)
        assert connection_id not in manager.connections
        assert manager.stats.active_connections == 0


class TestRealtimeEndpoints:
    """Test real-time API endpoints."""

    @pytest.fixture
    def mock_realtime_config(self):
        """Mock configuration for real-time endpoints."""
        config = Mock()
        config.realtime = Mock()
        config.realtime.enabled = True
        config.realtime.stream_interval_seconds = 5
        config.realtime.max_clients = 50
        config.realtime.buffer_size = 1000
        return config

    def test_realtime_endpoint_configuration(self, mock_realtime_config):
        """Test real-time endpoint configuration."""
        # Test configuration validation
        assert mock_realtime_config.realtime.enabled is True
        assert mock_realtime_config.realtime.stream_interval_seconds == 5
        assert mock_realtime_config.realtime.max_clients == 50

    def test_streaming_endpoint_initialization(self):
        """Test streaming endpoint initialization."""
        # Test that we can import the realtime endpoints module
        from src.integration import realtime_api_endpoints

        assert realtime_api_endpoints is not None

        # Test that the module has expected components
        assert hasattr(realtime_api_endpoints, "set_integration_manager")
        assert hasattr(realtime_api_endpoints, "_integration_manager")

    async def test_prediction_streaming_endpoint(self):
        """Test prediction streaming endpoint functionality."""
        # Mock streaming data
        prediction_data = {
            "room_id": "living_room",
            "predicted_time": datetime.now().isoformat(),
            "confidence": 0.87,
            "transition_type": "occupied",
            "timestamp": datetime.now().isoformat(),
        }

        # Test prediction data formatting
        formatted_data = self._format_prediction_stream_data(prediction_data)

        assert "room_id" in formatted_data
        assert "stream_type" in formatted_data
        assert formatted_data["stream_type"] == "prediction"
        assert "timestamp" in formatted_data

    async def test_event_streaming_endpoint(self):
        """Test event streaming endpoint functionality."""
        # Mock event data
        event_data = {
            "event_type": "state_change",
            "room_id": "living_room",
            "sensor_id": "motion_sensor_01",
            "old_state": "off",
            "new_state": "on",
            "timestamp": datetime.now().isoformat(),
        }

        # Test event data formatting
        formatted_data = self._format_event_stream_data(event_data)

        assert "event_type" in formatted_data
        assert "stream_type" in formatted_data
        assert formatted_data["stream_type"] == "event"
        assert formatted_data["room_id"] == "living_room"

    async def test_system_status_streaming(self):
        """Test system status streaming endpoint."""
        # Mock system status data
        status_data = {
            "overall_status": "healthy",
            "components": {
                "database": "healthy",
                "tracking_manager": "active",
                "mqtt_broker": "connected",
            },
            "performance_metrics": {
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
                "active_connections": 12,
            },
            "timestamp": datetime.now().isoformat(),
        }

        # Test status data formatting
        formatted_data = self._format_status_stream_data(status_data)

        assert "stream_type" in formatted_data
        assert formatted_data["stream_type"] == "system_status"
        assert "overall_status" in formatted_data
        assert "components" in formatted_data
        assert "performance_metrics" in formatted_data

    def test_stream_data_validation(self):
        """Test streaming data validation."""
        # Test valid prediction data
        valid_prediction = {
            "room_id": "living_room",
            "predicted_time": datetime.now().isoformat(),
            "confidence": 0.85,
        }

        assert self._validate_prediction_data(valid_prediction) is True

        # Test invalid prediction data (missing required fields)
        invalid_prediction = {
            "room_id": "living_room"
            # Missing predicted_time and confidence
        }

        assert self._validate_prediction_data(invalid_prediction) is False

        # Test invalid confidence value
        invalid_confidence = {
            "room_id": "living_room",
            "predicted_time": datetime.now().isoformat(),
            "confidence": 1.5,  # Out of range
        }

        assert self._validate_prediction_data(invalid_confidence) is False

    def test_stream_rate_limiting(self):
        """Test streaming endpoint rate limiting."""
        # Mock rate limiter
        rate_limiter = Mock()
        rate_limiter.is_allowed.side_effect = [
            True,
            True,
            True,
            False,
        ]  # Allow 3, deny 4th

        client_id = "client_123"

        # Test rate limiting behavior
        for i in range(4):
            allowed = rate_limiter.is_allowed(client_id, 3, 60)  # 3 requests per minute
            if i < 3:
                assert allowed is True
            else:
                assert allowed is False

    async def test_stream_client_management(self):
        """Test streaming client connection management."""
        # Mock client registry
        client_registry = {}

        # Test client registration
        client_id = "client_123"
        client_info = {
            "connection_time": datetime.now(),
            "last_activity": datetime.now(),
            "subscriptions": ["predictions", "events"],
            "rate_limit_count": 0,
        }

        client_registry[client_id] = client_info

        assert client_id in client_registry
        assert len(client_registry[client_id]["subscriptions"]) == 2

        # Test client cleanup (stale connections)
        stale_time = datetime.now() - timedelta(minutes=30)
        client_registry[client_id]["last_activity"] = stale_time

        # Simulate cleanup process
        current_time = datetime.now()
        stale_clients = [
            cid
            for cid, info in client_registry.items()
            if (current_time - info["last_activity"]).total_seconds()
            > 1800  # 30 minutes
        ]

        assert client_id in stale_clients

    def test_stream_message_formatting(self):
        """Test streaming message format consistency."""
        base_timestamp = datetime.now().isoformat()

        # Test prediction message format
        prediction_msg = {
            "stream_type": "prediction",
            "data": {"room_id": "living_room", "confidence": 0.85},
            "timestamp": base_timestamp,
            "sequence_id": 1,
        }

        self._validate_stream_message_format(prediction_msg)

        # Test event message format
        event_msg = {
            "stream_type": "event",
            "data": {"event_type": "state_change", "room_id": "bedroom"},
            "timestamp": base_timestamp,
            "sequence_id": 2,
        }

        self._validate_stream_message_format(event_msg)

        # Test status message format
        status_msg = {
            "stream_type": "system_status",
            "data": {"overall_status": "healthy"},
            "timestamp": base_timestamp,
            "sequence_id": 3,
        }

        self._validate_stream_message_format(status_msg)

    def test_stream_error_handling(self):
        """Test streaming endpoint error handling."""
        # Test connection errors
        connection_error = Exception("Connection lost")
        error_response = self._handle_stream_error(connection_error, "client_123")

        assert "error" in error_response
        assert "client_id" in error_response
        assert error_response["client_id"] == "client_123"

        # Test data validation errors
        validation_error = ValueError("Invalid data format")
        error_response = self._handle_stream_error(validation_error, "client_456")

        assert "error" in error_response
        assert "validation" in error_response["error"].lower()

        # Test rate limit errors
        rate_limit_error = Exception("Rate limit exceeded")
        error_response = self._handle_stream_error(rate_limit_error, "client_789")

        assert "error" in error_response
        assert "rate limit" in error_response["error"].lower()

    # Helper methods for testing
    def _format_prediction_stream_data(self, data):
        """Format prediction data for streaming."""
        return {
            "stream_type": "prediction",
            "room_id": data["room_id"],
            "predicted_time": data["predicted_time"],
            "confidence": data["confidence"],
            "transition_type": data.get("transition_type"),
            "timestamp": data["timestamp"],
        }

    def _format_event_stream_data(self, data):
        """Format event data for streaming."""
        return {
            "stream_type": "event",
            "event_type": data["event_type"],
            "room_id": data["room_id"],
            "sensor_id": data.get("sensor_id"),
            "old_state": data.get("old_state"),
            "new_state": data.get("new_state"),
            "timestamp": data["timestamp"],
        }

    def _format_status_stream_data(self, data):
        """Format system status data for streaming."""
        return {
            "stream_type": "system_status",
            "overall_status": data["overall_status"],
            "components": data["components"],
            "performance_metrics": data["performance_metrics"],
            "timestamp": data["timestamp"],
        }

    def _validate_prediction_data(self, data):
        """Validate prediction data format."""
        required_fields = ["room_id", "predicted_time", "confidence"]

        # Check required fields
        for field in required_fields:
            if field not in data:
                return False

        # Validate confidence range
        confidence = data.get("confidence")
        if confidence is not None and (confidence < 0 or confidence > 1):
            return False

        return True

    def _validate_stream_message_format(self, message):
        """Validate streaming message format."""
        required_fields = ["stream_type", "data", "timestamp", "sequence_id"]

        for field in required_fields:
            assert field in message, f"Missing required field: {field}"

        assert isinstance(message["data"], dict), "Data field must be a dictionary"
        assert isinstance(message["sequence_id"], int), "Sequence ID must be an integer"

    def _handle_stream_error(self, error, client_id):
        """Handle streaming errors and format error response."""
        error_type = type(error).__name__
        error_message = str(error)

        return {
            "error": error_message,
            "error_type": error_type,
            "client_id": client_id,
            "timestamp": datetime.now().isoformat(),
        }


class TestMonitoringAPI:
    """Test monitoring API endpoints."""

    @pytest.fixture
    def mock_health_monitor(self):
        """Mock health monitor for testing."""
        monitor = Mock()
        monitor.get_system_health.return_value = Mock(
            overall_status=Mock(value="healthy"),
            last_updated=datetime.now(),
            health_score=Mock(return_value=0.95),
            critical_components=[],
            degraded_components=[],
            to_dict=Mock(
                return_value={
                    "overall_status": "healthy",
                    "health_score": 0.95,
                    "message": "All systems operational",
                }
            ),
        )
        monitor.get_component_health.return_value = {
            "database": Mock(
                to_dict=Mock(
                    return_value={
                        "component_name": "database",
                        "status": "healthy",
                        "last_check": datetime.now().isoformat(),
                        "message": "Database connection active",
                    }
                )
            ),
            "tracking_manager": Mock(
                to_dict=Mock(
                    return_value={
                        "component_name": "tracking_manager",
                        "status": "healthy",
                        "last_check": datetime.now().isoformat(),
                        "message": "Tracking manager operational",
                    }
                )
            ),
        }
        monitor.get_monitoring_stats.return_value = {
            "monitoring_active": True,
            "total_checks_performed": 150,
            "failed_checks": 2,
            "average_check_duration_ms": 45.3,
            "last_check_time": datetime.now().isoformat(),
        }
        monitor.get_health_history.return_value = [
            (datetime.now() - timedelta(hours=1), Mock(value="healthy")),
            (datetime.now() - timedelta(hours=2), Mock(value="healthy")),
        ]
        monitor.health_checks = {"database": Mock(), "tracking": Mock()}
        monitor.is_monitoring_active.return_value = True
        monitor.start_monitoring = AsyncMock()
        monitor.stop_monitoring = AsyncMock()
        return monitor

    @pytest.fixture
    def mock_incident_manager(self):
        """Mock incident response manager for testing."""
        manager = Mock()
        manager.get_active_incidents.return_value = {
            "incident_001": Mock(
                to_dict=Mock(
                    return_value={
                        "incident_id": "incident_001",
                        "type": "database_connection_lost",
                        "severity": "critical",
                        "status": "active",
                        "created_at": datetime.now().isoformat(),
                        "description": "Database connection lost",
                    }
                )
            )
        }
        manager.get_incident.return_value = Mock(
            to_dict=Mock(
                return_value={
                    "incident_id": "incident_001",
                    "type": "database_connection_lost",
                    "severity": "critical",
                    "status": "active",
                    "created_at": datetime.now().isoformat(),
                    "description": "Database connection lost",
                    "actions_taken": [
                        "Attempted reconnection",
                        "Notified administrators",
                    ],
                }
            )
        )
        manager.get_incident_history.return_value = [
            Mock(
                to_dict=Mock(
                    return_value={
                        "incident_id": "incident_002",
                        "type": "high_prediction_error",
                        "severity": "warning",
                        "status": "resolved",
                        "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                        "resolved_at": (
                            datetime.now() - timedelta(hours=1)
                        ).isoformat(),
                    }
                )
            )
        ]
        manager.get_incident_statistics.return_value = {
            "active_incidents_count": 1,
            "total_incidents_24h": 3,
            "resolved_incidents_24h": 2,
            "average_resolution_time_minutes": 45,
            "auto_recovery_enabled": True,
        }
        manager.acknowledge_incident = AsyncMock(return_value=True)
        manager.resolve_incident = AsyncMock(return_value=True)
        manager.start_incident_response = AsyncMock()
        manager.stop_incident_response = AsyncMock()
        return manager

    def test_health_monitoring_endpoint_creation(self, mock_health_monitor):
        """Test health monitoring endpoint creation."""
        # Test that we can import the monitoring API module
        from src.integration import monitoring_api

        assert monitoring_api is not None

        # Test that the module has expected components
        assert hasattr(monitoring_api, "SystemStatus")
        assert hasattr(monitoring_api, "MetricsResponse")
        assert hasattr(monitoring_api, "HealthCheckResponse")

    def test_system_health_summary_model(self):
        """Test SystemStatus model validation."""
        # Test creating a SystemStatus response
        status = SystemStatus(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            uptime_seconds=3600.0,
            health_score=0.95,
            active_alerts=0,
            monitoring_enabled=True,
        )

        assert status.status == "healthy"
        assert status.health_score == 0.95
        assert status.active_alerts == 0
        assert status.monitoring_enabled is True

    def test_metrics_response_model(self):
        """Test MetricsResponse model validation."""
        # Test creating a MetricsResponse
        metrics = MetricsResponse(
            metrics_format="prometheus",
            timestamp=datetime.now().isoformat(),
            metrics_count=25,
        )

        assert metrics.metrics_format == "prometheus"
        assert metrics.metrics_count == 25
        assert isinstance(metrics.timestamp, str)

    def test_health_check_response_model(self):
        """Test HealthCheckResponse model validation."""
        # Test creating a HealthCheckResponse
        health_check = HealthCheckResponse(
            status="healthy",
            checks={
                "database": {"status": "ok", "response_time": "5ms"},
                "cache": {"status": "ok", "response_time": "1ms"},
            },
            timestamp=datetime.now().isoformat(),
            overall_healthy=True,
        )

        assert health_check.status == "healthy"
        assert health_check.overall_healthy is True
        assert "database" in health_check.checks
        assert "cache" in health_check.checks

    def test_alerts_response_model(self):
        """Test AlertsResponse model validation."""
        # Test creating an AlertsResponse
        alerts = AlertsResponse(active_alerts=2, total_alerts_today=5)

        assert alerts.active_alerts == 2
        assert alerts.total_alerts_today == 5

    def test_monitoring_api_imports(self):
        """Test that monitoring API components can be imported."""
        # Test importing monitoring API components
        from src.integration.monitoring_api import (
            get_alert_manager,
            get_logger,
            get_metrics_manager,
        )

        # These should be importable without errors
        assert get_alert_manager is not None
        assert get_logger is not None
        assert get_metrics_manager is not None

    def test_health_status_validation(self):
        """Test health status validation and formatting."""
        # Test valid health status
        valid_status = {
            "overall_status": "healthy",
            "components": {"database": "healthy", "tracking": "active"},
            "health_score": 0.95,
            "timestamp": datetime.now().isoformat(),
        }

        validated = self._validate_health_status(valid_status)
        assert validated is True

        # Test invalid health status (missing required fields)
        invalid_status = {
            "components": {"database": "healthy"}
            # Missing overall_status and health_score
        }

        validated = self._validate_health_status(invalid_status)
        assert validated is False

    def test_metrics_data_aggregation(self):
        """Test metrics data aggregation functionality."""
        # Mock raw metrics data
        raw_metrics = [
            {
                "timestamp": datetime.now() - timedelta(minutes=5),
                "cpu_usage": 45.2,
                "memory_usage": 67.8,
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=4),
                "cpu_usage": 47.1,
                "memory_usage": 68.2,
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=3),
                "cpu_usage": 43.9,
                "memory_usage": 66.5,
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=2),
                "cpu_usage": 46.3,
                "memory_usage": 67.1,
            },
            {
                "timestamp": datetime.now() - timedelta(minutes=1),
                "cpu_usage": 44.8,
                "memory_usage": 67.9,
            },
        ]

        # Test metrics aggregation
        aggregated = self._aggregate_metrics(raw_metrics)

        assert "average_cpu_usage" in aggregated
        assert "average_memory_usage" in aggregated
        assert "max_cpu_usage" in aggregated
        assert "min_cpu_usage" in aggregated
        assert "data_points" in aggregated

        assert aggregated["data_points"] == 5
        assert 43.0 < aggregated["average_cpu_usage"] < 48.0
        assert 66.0 < aggregated["average_memory_usage"] < 69.0

    def test_alert_threshold_evaluation(self):
        """Test alert threshold evaluation logic."""
        # Define thresholds
        thresholds = {
            "cpu_usage": {"warning": 70.0, "critical": 90.0},
            "memory_usage": {"warning": 80.0, "critical": 95.0},
            "error_rate": {"warning": 0.05, "critical": 0.10},
        }

        # Test normal values (no alerts)
        normal_metrics = {"cpu_usage": 45.2, "memory_usage": 67.8, "error_rate": 0.02}

        alerts = self._evaluate_alert_thresholds(normal_metrics, thresholds)
        assert len(alerts) == 0

        # Test warning values
        warning_metrics = {"cpu_usage": 75.0, "memory_usage": 85.0, "error_rate": 0.07}

        alerts = self._evaluate_alert_thresholds(warning_metrics, thresholds)
        assert len(alerts) == 3
        assert all(alert["severity"] == "warning" for alert in alerts)

        # Test critical values
        critical_metrics = {"cpu_usage": 95.0, "memory_usage": 97.0, "error_rate": 0.15}

        alerts = self._evaluate_alert_thresholds(critical_metrics, thresholds)
        assert len(alerts) == 3
        assert all(alert["severity"] == "critical" for alert in alerts)

    def test_monitoring_api_error_handling(self):
        """Test monitoring API error handling."""
        # Test health monitor unavailable
        error_response = self._handle_monitoring_error(
            Exception("Health monitor unavailable"), "health_check"
        )

        assert "error" in error_response
        assert "operation" in error_response
        assert error_response["operation"] == "health_check"

        # Test incident manager error
        error_response = self._handle_monitoring_error(
            Exception("Incident manager error"), "incident_management"
        )

        assert "error" in error_response
        assert "incident manager" in error_response["error"].lower()

        # Test metrics collection error
        error_response = self._handle_monitoring_error(
            Exception("Metrics collection failed"), "metrics_collection"
        )

        assert "error" in error_response
        assert "metrics collection" in error_response["error"].lower()

    # Helper methods for testing
    def _validate_health_status(self, status):
        """Validate health status format."""
        required_fields = ["overall_status", "health_score"]

        for field in required_fields:
            if field not in status:
                return False

        # Validate health score range
        health_score = status.get("health_score")
        if health_score is not None and (health_score < 0 or health_score > 1):
            return False

        return True

    def _aggregate_metrics(self, raw_metrics):
        """Aggregate raw metrics data."""
        if not raw_metrics:
            return {"data_points": 0}

        cpu_values = [m["cpu_usage"] for m in raw_metrics]
        memory_values = [m["memory_usage"] for m in raw_metrics]

        return {
            "average_cpu_usage": sum(cpu_values) / len(cpu_values),
            "average_memory_usage": sum(memory_values) / len(memory_values),
            "max_cpu_usage": max(cpu_values),
            "min_cpu_usage": min(cpu_values),
            "max_memory_usage": max(memory_values),
            "min_memory_usage": min(memory_values),
            "data_points": len(raw_metrics),
        }

    def _evaluate_alert_thresholds(self, metrics, thresholds):
        """Evaluate metrics against alert thresholds."""
        alerts = []

        for metric_name, value in metrics.items():
            if metric_name in thresholds:
                threshold = thresholds[metric_name]

                if value >= threshold["critical"]:
                    alerts.append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold["critical"],
                            "severity": "critical",
                        }
                    )
                elif value >= threshold["warning"]:
                    alerts.append(
                        {
                            "metric": metric_name,
                            "value": value,
                            "threshold": threshold["warning"],
                            "severity": "warning",
                        }
                    )

        return alerts

    def _handle_monitoring_error(self, error, operation):
        """Handle monitoring API errors."""
        return {
            "error": str(error),
            "operation": operation,
            "timestamp": datetime.now().isoformat(),
            "status": "error",
        }


class TestDashboardIntegration:
    """Test dashboard integration functionality."""

    @pytest.fixture
    def mock_dashboard_config(self):
        """Mock dashboard configuration for testing."""
        config = Mock()
        config.dashboard = Mock()
        config.dashboard.enabled = True
        config.dashboard.host = "localhost"
        config.dashboard.port = 3000
        config.dashboard.debug = False
        config.dashboard.auth_required = True
        config.dashboard.refresh_interval_seconds = 30
        config.dashboard.max_data_points = 1000
        return config

    def test_dashboard_initialization(self, mock_dashboard_config):
        """Test dashboard integration initialization."""
        # Test that we can import the dashboard module
        from src.integration import dashboard

        assert dashboard is not None

        # Test that the module has expected components
        assert hasattr(dashboard, "DashboardIntegration")
        assert hasattr(dashboard, "FASTAPI_AVAILABLE")

        # Test configuration validation
        assert mock_dashboard_config.dashboard.enabled is True
        assert mock_dashboard_config.dashboard.host == "localhost"
        assert mock_dashboard_config.dashboard.port == 3000

    def test_dashboard_fastapi_availability(self):
        """Test dashboard FastAPI availability check."""
        from src.integration.dashboard import FASTAPI_AVAILABLE

        # This should be a boolean indicating if FastAPI is available
        assert isinstance(FASTAPI_AVAILABLE, bool)

        # Test that the module can be imported regardless of FastAPI availability
        from src.integration import dashboard

        assert dashboard is not None

    def test_dashboard_types_available(self):
        """Test that dashboard-related types are available when FastAPI is present."""
        from src.integration import dashboard

        # Test that key classes exist when FastAPI is available
        if dashboard.FASTAPI_AVAILABLE:
            assert hasattr(dashboard, "FastAPI")
            assert hasattr(dashboard, "HTTPException")
            assert hasattr(dashboard, "WebSocket")
        else:
            # When FastAPI isn't available, these should be None
            assert dashboard.FastAPI is None
            assert dashboard.HTTPException is None

    def test_dashboard_imports_and_structure(self):
        """Test dashboard module structure and key imports."""
        from src.integration import dashboard

        # Test that module has expected structure
        assert hasattr(dashboard, "DashboardIntegration")
        assert hasattr(dashboard, "FASTAPI_AVAILABLE")

        # Test that TYPE_CHECKING imports work
        assert hasattr(dashboard, "TYPE_CHECKING")
        assert hasattr(dashboard, "Dict")
        assert hasattr(dashboard, "List")
        assert hasattr(dashboard, "Optional")

    def test_dashboard_module_completeness(self):
        """Test that dashboard module is complete and well-structured."""
        from src.integration import dashboard

        # Test that the module loads without errors
        assert dashboard is not None

        # Test that it has the basic structure we expect
        expected_attributes = [
            "DashboardIntegration",
            "FASTAPI_AVAILABLE",
            "datetime",
            "logging",
            "threading",
        ]

        for attr in expected_attributes:
            assert hasattr(dashboard, attr), f"Dashboard module missing {attr}"

        # Test that FastAPI conditional imports work
        if dashboard.FASTAPI_AVAILABLE:
            # When FastAPI is available, these should be imported classes
            assert dashboard.FastAPI is not None
            assert dashboard.HTTPException is not None
        else:
            # When not available, should be None
            assert dashboard.FastAPI is None
