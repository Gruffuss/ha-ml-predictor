"""
Comprehensive unit tests for the API server module.

This test suite validates the FastAPI server functionality, including endpoints,
authentication, error handling, middleware, and integration with TrackingManager.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, Mock, patch

from fastapi import status
from fastapi.testclient import TestClient
import pytest

from src.core.exceptions import (
    APIAuthenticationError,
    APIError,
    APIRateLimitError,
    APIResourceNotFoundError,
    APIServerError,
)
from src.integration.api_server import (
    AccuracyMetricsResponse,
    APIServer,
    ErrorResponse,
    ManualRetrainRequest,
    PredictionResponse,
    RateLimitTracker,
    SystemHealthResponse,
    SystemStatsResponse,
    check_rate_limit,
    create_app,
    get_app,
    get_tracking_manager,
    integrate_with_tracking_manager,
    register_routes,
    set_tracking_manager,
    verify_api_key,
)


class TestAPIServerCore:
    """Test core API server functionality."""

    def test_rate_limit_tracker_initialization(self):
        """Test RateLimitTracker initialization."""
        tracker = RateLimitTracker()
        assert tracker.requests == {}

    def test_rate_limit_tracker_allows_first_request(self):
        """Test that first request is always allowed."""
        tracker = RateLimitTracker()
        assert tracker.is_allowed("127.0.0.1", 10) is True
        assert len(tracker.requests["127.0.0.1"]) == 1

    def test_rate_limit_tracker_blocks_excessive_requests(self):
        """Test that excessive requests are blocked."""
        tracker = RateLimitTracker()

        # Make requests up to limit
        for _ in range(10):
            assert tracker.is_allowed("127.0.0.1", 10) is True

        # Next request should be blocked
        assert tracker.is_allowed("127.0.0.1", 10) is False

    def test_rate_limit_tracker_window_cleanup(self):
        """Test that old requests are cleaned up."""
        tracker = RateLimitTracker()

        # Mock old timestamp
        old_time = datetime.now() - timedelta(minutes=2)
        tracker.requests["127.0.0.1"] = [old_time]

        # Should allow request as old ones are cleaned up
        assert tracker.is_allowed("127.0.0.1", 10) is True
        assert len(tracker.requests["127.0.0.1"]) == 1  # Only new request

    def test_rate_limit_tracker_multiple_ips(self):
        """Test rate limiting works independently for different IPs."""
        tracker = RateLimitTracker()

        # Fill limit for first IP
        for _ in range(5):
            assert tracker.is_allowed("127.0.0.1", 5) is True

        # Should block further requests from first IP
        assert tracker.is_allowed("127.0.0.1", 5) is False

        # But allow requests from second IP
        assert tracker.is_allowed("192.168.1.1", 5) is True


class TestPydanticModels:
    """Test Pydantic model validation and serialization."""

    def test_prediction_response_validation(self):
        """Test PredictionResponse model validation."""
        valid_data = {
            "room_id": "living_room",
            "prediction_time": datetime.now(timezone.utc),
            "next_transition_time": datetime.now(timezone.utc) + timedelta(minutes=30),
            "transition_type": "occupied_to_vacant",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {"model_type": "lstm"},
        }

        response = PredictionResponse(**valid_data)
        assert response.room_id == "living_room"
        assert response.confidence == 0.85
        assert response.transition_type == "occupied_to_vacant"

    def test_prediction_response_invalid_confidence(self):
        """Test PredictionResponse with invalid confidence value."""
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            PredictionResponse(
                room_id="living_room",
                prediction_time=datetime.now(timezone.utc),
                transition_type="occupied",
                confidence=1.5,  # Invalid
                time_until_transition="30 minutes",
            )

    def test_prediction_response_invalid_transition_type(self):
        """Test PredictionResponse with invalid transition type."""
        with pytest.raises(ValueError, match="Transition type must be one of"):
            PredictionResponse(
                room_id="living_room",
                prediction_time=datetime.now(timezone.utc),
                transition_type="invalid_type",  # Invalid
                confidence=0.85,
                time_until_transition="30 minutes",
            )

    def test_accuracy_metrics_response_validation(self):
        """Test AccuracyMetricsResponse model validation."""
        valid_data = {
            "room_id": "bedroom",
            "accuracy_rate": 0.87,
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.92,
            "total_predictions": 1000,
            "total_validations": 950,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }

        response = AccuracyMetricsResponse(**valid_data)
        assert response.accuracy_rate == 0.87
        assert response.trend_direction == "improving"

    def test_accuracy_metrics_response_invalid_rate(self):
        """Test AccuracyMetricsResponse with invalid rate value."""
        with pytest.raises(ValueError, match="Rate must be between 0.0 and 1.0"):
            AccuracyMetricsResponse(
                accuracy_rate=1.5,  # Invalid
                average_error_minutes=10.0,
                confidence_calibration=0.9,
                total_predictions=100,
                total_validations=90,
                time_window_hours=24,
                trend_direction="stable",
            )

    def test_manual_retrain_request_validation(self):
        """Test ManualRetrainRequest model validation."""
        valid_data = {
            "room_id": "kitchen",
            "force": True,
            "strategy": "incremental",
            "reason": "Performance degradation detected",
        }

        request = ManualRetrainRequest(**valid_data)
        assert request.room_id == "kitchen"
        assert request.force is True
        assert request.strategy == "incremental"

    def test_manual_retrain_request_invalid_strategy(self):
        """Test ManualRetrainRequest with invalid strategy."""
        with pytest.raises(ValueError, match="Strategy must be one of"):
            ManualRetrainRequest(
                strategy="invalid_strategy",  # Invalid
                reason="Test reason",
            )

    def test_error_response_serialization(self):
        """Test ErrorResponse serialization with datetime."""
        error_response = ErrorResponse(
            error="Test error",
            error_code="TEST_ERROR",
            details={"key": "value"},
            timestamp=datetime.now(timezone.utc),
            request_id="12345",
        )

        serialized = error_response.dict()
        assert serialized["error"] == "Test error"
        assert isinstance(serialized["timestamp"], str)  # Should be ISO format


class TestAPIServerDependencies:
    """Test API server dependency functions."""

    @pytest.fixture
    def mock_config(self):
        """Mock system configuration."""
        config = Mock()
        config.api.api_key_enabled = True
        config.api.api_key = "test-api-key-123"
        config.api.rate_limit_enabled = True
        config.api.requests_per_minute = 60
        return config

    @pytest.fixture
    def mock_tracking_manager(self):
        """Mock tracking manager."""
        manager = AsyncMock()
        manager.get_room_prediction = AsyncMock(
            return_value={
                "room_id": "living_room",
                "prediction_time": "2024-01-01T12:00:00Z",
                "next_transition_time": "2024-01-01T12:30:00Z",
                "transition_type": "occupied_to_vacant",
                "confidence": 0.85,
                "time_until_transition": "30 minutes",
                "alternatives": [],
                "model_info": {"model_type": "lstm"},
            }
        )
        return manager

    @patch("src.integration.api_server.get_config")
    def test_verify_api_key_success(self, mock_get_config, mock_config):
        """Test successful API key verification."""
        mock_get_config.return_value = mock_config

        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="test-api-key-123"
        )

        # Should not raise exception
        result = asyncio.run(verify_api_key(credentials))
        assert result is True

    @patch("src.integration.api_server.get_config")
    def test_verify_api_key_disabled(self, mock_get_config, mock_config):
        """Test API key verification when disabled."""
        mock_config.api.api_key_enabled = False
        mock_get_config.return_value = mock_config

        result = asyncio.run(verify_api_key(None))
        assert result is True

    @patch("src.integration.api_server.get_config")
    def test_verify_api_key_missing_credentials(self, mock_get_config, mock_config):
        """Test API key verification with missing credentials."""
        mock_get_config.return_value = mock_config

        with pytest.raises(APIAuthenticationError, match="API Key required"):
            asyncio.run(verify_api_key(None))

    @patch("src.integration.api_server.get_config")
    def test_verify_api_key_invalid(self, mock_get_config, mock_config):
        """Test API key verification with invalid key."""
        mock_get_config.return_value = mock_config

        from fastapi.security import HTTPAuthorizationCredentials

        credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials="invalid-key"
        )

        with pytest.raises(APIAuthenticationError, match="Invalid API key"):
            asyncio.run(verify_api_key(credentials))

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.rate_limiter")
    def test_check_rate_limit_success(
        self, mock_rate_limiter, mock_get_config, mock_config
    ):
        """Test successful rate limit check."""
        mock_get_config.return_value = mock_config
        mock_rate_limiter.is_allowed.return_value = True

        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        result = asyncio.run(check_rate_limit(mock_request))
        assert result is True

    @patch("src.integration.api_server.get_config")
    @patch("src.integration.api_server.rate_limiter")
    def test_check_rate_limit_exceeded(
        self, mock_rate_limiter, mock_get_config, mock_config
    ):
        """Test rate limit exceeded."""
        mock_get_config.return_value = mock_config
        mock_rate_limiter.is_allowed.return_value = False

        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        with pytest.raises(APIRateLimitError):
            asyncio.run(check_rate_limit(mock_request))

    @patch("src.integration.api_server.get_config")
    def test_check_rate_limit_disabled(self, mock_get_config, mock_config):
        """Test rate limit check when disabled."""
        mock_config.api.rate_limit_enabled = False
        mock_get_config.return_value = mock_config

        mock_request = Mock()
        result = asyncio.run(check_rate_limit(mock_request))
        assert result is True

    def test_get_tracking_manager_initialization(self):
        """Test tracking manager initialization."""
        # Reset global instance
        global _tracking_manager_instance
        _tracking_manager_instance = None

        with patch("src.integration.api_server.get_config") as mock_get_config, patch(
            "src.integration.api_server.TrackingManager"
        ) as mock_tm_class, patch(
            "src.integration.api_server.TrackingConfig"
        ) as mock_tc_class:

            mock_config = Mock()
            mock_config.tracking = None
            mock_get_config.return_value = mock_config

            mock_tm_instance = AsyncMock()
            mock_tm_class.return_value = mock_tm_instance

            # Should create new instance
            result = asyncio.run(get_tracking_manager())
            assert result == mock_tm_instance
            mock_tm_instance.initialize.assert_called_once()

    def test_set_tracking_manager(self):
        """Test setting tracking manager instance."""
        mock_manager = AsyncMock()
        set_tracking_manager(mock_manager)

        # Should return the set instance
        result = asyncio.run(get_tracking_manager())
        assert result == mock_manager


class TestAPIServerEndpoints:
    """Test API server endpoint functionality."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for clean testing."""
        with patch(
            "src.integration.api_server.verify_api_key", return_value=True
        ), patch(
            "src.integration.api_server.check_rate_limit", return_value=True
        ), patch(
            "src.integration.api_server.get_database_manager"
        ) as mock_db, patch(
            "src.integration.api_server.get_tracking_manager"
        ) as mock_tm, patch(
            "src.integration.api_server.get_mqtt_manager"
        ) as mock_mqtt:

            # Setup mocks
            mock_db_instance = AsyncMock()
            mock_db_instance.health_check.return_value = {
                "status": "healthy",
                "database_connected": True,
            }
            mock_db.return_value = mock_db_instance

            mock_tm_instance = AsyncMock()
            mock_tm_instance.get_tracking_status.return_value = {
                "status": "active",
                "config": {"enabled": True},
                "performance": {"background_tasks": 3},
            }
            mock_tm.return_value = mock_tm_instance

            mock_mqtt_instance = AsyncMock()
            mock_mqtt_instance.get_integration_stats.return_value = {
                "mqtt_connected": True,
                "predictions_published": 150,
            }
            mock_mqtt.return_value = mock_mqtt_instance

            yield {
                "db": mock_db_instance,
                "tracking": mock_tm_instance,
                "mqtt": mock_mqtt_instance,
            }

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["name"] == "Occupancy Prediction API"
        assert data["version"] == "1.0.0"
        assert "timestamp" in data

    def test_health_endpoint_success(self, client, mock_dependencies):
        """Test health endpoint with healthy system."""
        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "components" in data
        assert data["components"]["database"]["status"] == "healthy"
        assert data["components"]["tracking"]["status"] == "healthy"

    @patch("src.integration.api_server.get_database_manager")
    def test_health_endpoint_database_failure(self, mock_db, client):
        """Test health endpoint when database is unhealthy."""
        mock_db_instance = AsyncMock()
        mock_db_instance.health_check.side_effect = Exception("DB connection failed")
        mock_db.return_value = mock_db_instance

        response = client.get("/health")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["status"] in ["degraded", "unhealthy"]

    def test_health_comprehensive_endpoint(self, client):
        """Test comprehensive health endpoint."""
        with patch(
            "src.integration.api_server.get_health_monitor"
        ) as mock_health_monitor:
            mock_monitor = Mock()
            mock_system_health = Mock()
            mock_system_health.overall_status.name = "HEALTHY"
            mock_system_health.last_updated = datetime.now(timezone.utc)
            mock_system_health.to_dict.return_value = {"status": "healthy"}

            mock_monitor.get_system_health.return_value = mock_system_health
            mock_monitor.get_component_health.return_value = {}
            mock_monitor.get_monitoring_stats.return_value = {"uptime": 3600}

            mock_health_monitor.return_value = mock_monitor

            response = client.get("/health/comprehensive")
            assert response.status_code == status.HTTP_200_OK

            data = response.json()
            assert data["status"] == "healthy"
            assert "system_health" in data
            assert "components" in data

    @patch("src.integration.api_server.get_config")
    def test_predictions_endpoint_room_not_found(
        self, mock_get_config, client, mock_dependencies
    ):
        """Test predictions endpoint with non-existent room."""
        mock_config = Mock()
        mock_config.rooms = {"living_room": {}}
        mock_get_config.return_value = mock_config

        response = client.get("/predictions/non_existent_room")
        assert response.status_code == status.HTTP_404_NOT_FOUND

    @patch("src.integration.api_server.get_config")
    def test_predictions_endpoint_success(
        self, mock_get_config, client, mock_dependencies
    ):
        """Test successful predictions endpoint."""
        mock_config = Mock()
        mock_config.rooms = {"living_room": {}}
        mock_get_config.return_value = mock_config

        # Mock tracking manager prediction
        mock_dependencies["tracking"].get_room_prediction.return_value = {
            "room_id": "living_room",
            "prediction_time": "2024-01-01T12:00:00Z",
            "next_transition_time": "2024-01-01T12:30:00Z",
            "transition_type": "occupied_to_vacant",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {"model_type": "lstm"},
        }

        response = client.get("/predictions/living_room")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert data["room_id"] == "living_room"
        assert data["confidence_score"] == 0.85

    def test_predictions_endpoint_no_data(self, client, mock_dependencies):
        """Test predictions endpoint when no prediction available."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.rooms = {"living_room": {}}
            mock_get_config.return_value = mock_config

            # Return None for no prediction
            mock_dependencies["tracking"].get_room_prediction.return_value = None

            response = client.get("/predictions/living_room")
            assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_all_predictions_endpoint(self, client, mock_dependencies):
        """Test endpoint for getting all predictions."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.rooms = {"living_room": {}, "bedroom": {}}
            mock_get_config.return_value = mock_config

            # Mock prediction data
            prediction_data = {
                "room_id": "living_room",
                "prediction_time": "2024-01-01T12:00:00Z",
                "next_transition_time": "2024-01-01T12:30:00Z",
                "transition_type": "occupied_to_vacant",
                "confidence": 0.85,
                "time_until_transition": "30 minutes",
                "alternatives": [],
                "model_info": {"model_type": "lstm"},
            }
            mock_dependencies["tracking"].get_room_prediction.return_value = (
                prediction_data
            )

            response = client.get("/predictions")
            assert response.status_code == status.HTTP_200_OK

            data = response.json()
            assert isinstance(data, list)

    def test_accuracy_metrics_endpoint(self, client, mock_dependencies):
        """Test accuracy metrics endpoint."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.rooms = {"living_room": {}}
            mock_get_config.return_value = mock_config

            mock_dependencies["tracking"].get_accuracy_metrics.return_value = {
                "room_id": "living_room",
                "accuracy_rate": 0.87,
                "average_error_minutes": 12.5,
                "confidence_calibration": 0.92,
                "total_predictions": 1000,
                "total_validations": 950,
                "time_window_hours": 24,
                "trend_direction": "improving",
            }

            response = client.get("/accuracy?room_id=living_room")
            assert response.status_code == status.HTTP_200_OK

            data = response.json()
            assert data["room_id"] == "living_room"
            assert data["accuracy_rate"] == 0.87

    def test_retrain_endpoint(self, client, mock_dependencies):
        """Test manual retrain endpoint."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.rooms = {"living_room": {}}
            mock_get_config.return_value = mock_config

            mock_dependencies["tracking"].trigger_manual_retrain.return_value = {
                "message": "Retraining started",
                "success": True,
                "room_id": "living_room",
                "strategy": "incremental",
                "force": False,
            }

            retrain_data = {
                "room_id": "living_room",
                "force": False,
                "strategy": "incremental",
                "reason": "Performance degradation",
            }

            response = client.post("/model/retrain", json=retrain_data)
            assert response.status_code == status.HTTP_200_OK

            data = response.json()
            assert data["success"] is True
            assert data["room_id"] == "living_room"

    def test_mqtt_refresh_endpoint(self, client, mock_dependencies):
        """Test MQTT discovery refresh endpoint."""
        response = client.post("/mqtt/refresh")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "message" in data
        assert "timestamp" in data

    def test_stats_endpoint(self, client, mock_dependencies):
        """Test system stats endpoint."""
        mock_dependencies["tracking"].get_system_stats.return_value = {
            "tracking_stats": {"total_predictions_tracked": 500},
            "retraining_stats": {"completed_retraining_jobs": 10},
        }

        response = client.get("/stats")
        assert response.status_code == status.HTTP_200_OK

        data = response.json()
        assert "system_info" in data
        assert "prediction_stats" in data
        assert "mqtt_stats" in data
        assert "database_stats" in data
        assert "tracking_stats" in data


class TestAPIServerErrorHandling:
    """Test API server error handling and exception responses."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        app = create_app()
        return TestClient(app)

    def test_api_error_handler(self, client):
        """Test APIError exception handler."""
        app = create_app()

        @app.get("/test-api-error")
        async def test_endpoint():
            raise APIError("Test API error", "TEST_ERROR", {"detail": "test"})

        test_client = TestClient(app)
        response = test_client.get("/test-api-error")

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert data["error"] == "Test API error"
        assert data["error_code"] == "TEST_ERROR"

    def test_server_error_handler(self, client):
        """Test general server error handler."""
        app = create_app()

        @app.get("/test-server-error")
        async def test_endpoint():
            raise Exception("Unexpected error")

        test_client = TestClient(app)
        response = test_client.get("/test-server-error")

        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        data = response.json()
        assert data["error"] == "Internal server error"
        assert data["error_code"] == "UNHANDLED_EXCEPTION"

    def test_rate_limit_middleware_error(self):
        """Test rate limit error in middleware."""
        app = create_app()

        with patch("src.integration.api_server.check_rate_limit") as mock_check:
            mock_check.side_effect = APIRateLimitError("127.0.0.1", 60, "minute")

            test_client = TestClient(app)
            response = test_client.get("/health")

            assert response.status_code == status.HTTP_429_TOO_MANY_REQUESTS


class TestAPIServerIntegration:
    """Test API server integration with other components."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Mock tracking manager for integration tests."""
        manager = AsyncMock()
        manager.get_room_prediction = AsyncMock()
        manager.get_accuracy_metrics = AsyncMock()
        manager.trigger_manual_retrain = AsyncMock()
        manager.get_system_stats = AsyncMock()
        return manager

    def test_api_server_initialization(self, mock_tracking_manager):
        """Test APIServer class initialization."""
        server = APIServer(mock_tracking_manager)

        assert server.tracking_manager == mock_tracking_manager
        assert server.server is None
        assert server.server_task is None

    @pytest.mark.asyncio
    async def test_api_server_start_enabled(self, mock_tracking_manager):
        """Test API server start when enabled."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.enabled = True
            mock_config.api.host = "localhost"
            mock_config.api.port = 8000
            mock_config.api.debug = False
            mock_config.api.access_log = False
            mock_get_config.return_value = mock_config

            server = APIServer(mock_tracking_manager)

            with patch("src.integration.api_server.uvicorn.Server") as mock_uvicorn:
                mock_server_instance = Mock()
                mock_server_instance.serve = AsyncMock()
                mock_uvicorn.return_value = mock_server_instance

                await server.start()

                assert server.server == mock_server_instance
                assert server.server_task is not None

    @pytest.mark.asyncio
    async def test_api_server_start_disabled(self, mock_tracking_manager):
        """Test API server start when disabled."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.enabled = False
            mock_get_config.return_value = mock_config

            server = APIServer(mock_tracking_manager)
            await server.start()

            assert server.server is None

    @pytest.mark.asyncio
    async def test_api_server_stop(self, mock_tracking_manager):
        """Test API server stop."""
        server = APIServer(mock_tracking_manager)

        # Mock server components
        mock_server = Mock()
        mock_task = AsyncMock()
        server.server = mock_server
        server.server_task = mock_task

        await server.stop()

        assert mock_server.should_exit is True
        mock_task.assert_awaited_once()

    def test_api_server_is_running(self, mock_tracking_manager):
        """Test APIServer is_running method."""
        server = APIServer(mock_tracking_manager)

        # Initially not running
        assert server.is_running() is False

        # Mock running task
        mock_task = Mock()
        mock_task.done.return_value = False
        server.server_task = mock_task

        assert server.is_running() is True

    @pytest.mark.asyncio
    async def test_integrate_with_tracking_manager(self, mock_tracking_manager):
        """Test integration function."""
        api_server = await integrate_with_tracking_manager(mock_tracking_manager)

        assert isinstance(api_server, APIServer)
        assert api_server.tracking_manager == mock_tracking_manager

    def test_create_app_configuration(self):
        """Test app creation and configuration."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.debug = True
            mock_config.api.docs_url = "/docs"
            mock_config.api.redoc_url = "/redoc"
            mock_config.api.include_docs = True
            mock_config.api.enable_cors = True
            mock_config.api.cors_origins = ["*"]
            mock_config.api.jwt.enabled = False
            mock_get_config.return_value = mock_config

            app = create_app()

            assert app.debug is True
            assert app.docs_url == "/docs"
            assert app.redoc_url == "/redoc"

    def test_get_app_with_jwt_error(self):
        """Test get_app with JWT configuration error."""
        with patch("src.integration.api_server.create_app") as mock_create:
            mock_create.side_effect = ValueError("JWT_SECRET_KEY not set")

            # Mock test environment
            with patch.dict("os.environ", {"ENVIRONMENT": "test"}):
                app = get_app()
                assert app.title == "HA ML Predictor API (Test Mode)"

    def test_get_app_production_error(self):
        """Test get_app with production environment error."""
        with patch("src.integration.api_server.create_app") as mock_create:
            mock_create.side_effect = ValueError("Production error")

            with pytest.raises(ValueError, match="Production error"):
                get_app()


class TestAPIServerMiddleware:
    """Test API server middleware functionality."""

    @pytest.fixture
    def client(self):
        """Create test client with middleware."""
        app = create_app()
        return TestClient(app)

    def test_request_logging_middleware(self, client):
        """Test request logging middleware."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.api.log_requests = True
            mock_config.api.log_responses = True
            mock_get_config.return_value = mock_config

            # Mock dependencies
            with patch(
                "src.integration.api_server.check_rate_limit", return_value=True
            ):
                response = client.get("/")
                assert response.status_code == status.HTTP_200_OK

    def test_cors_middleware(self, client):
        """Test CORS middleware configuration."""
        response = client.options("/", headers={"Origin": "http://localhost:3000"})

        # Should handle OPTIONS request
        assert response.status_code in [200, 405]  # Depends on CORS config

    def test_trusted_host_middleware(self, client):
        """Test trusted host middleware."""
        response = client.get("/", headers={"Host": "localhost"})
        assert response.status_code == status.HTTP_200_OK


@pytest.mark.asyncio
async def test_background_health_check():
    """Test background health check functionality."""
    with patch("src.integration.api_server.get_config") as mock_get_config, patch(
        "src.integration.api_server.get_health_monitor"
    ) as mock_health_monitor, patch(
        "src.integration.api_server.get_incident_response_manager"
    ) as mock_incident_manager:

        mock_config = Mock()
        mock_config.api.health_check_interval_seconds = 1  # Fast for testing
        mock_get_config.return_value = mock_config

        mock_monitor = AsyncMock()
        mock_monitor.start_monitoring = AsyncMock()
        mock_monitor.get_system_health.return_value = Mock(
            overall_status=Mock(value="healthy")
        )
        mock_health_monitor.return_value = mock_monitor

        mock_incident = AsyncMock()
        mock_incident.start_incident_response = AsyncMock()
        mock_incident.get_active_incidents.return_value = {}
        mock_incident.get_incident_statistics.return_value = {
            "active_incidents_count": 0,
            "auto_recovery_enabled": True,
        }
        mock_incident_manager.return_value = mock_incident

        # Import the background function
        from src.integration.api_server import background_health_check

        # Run for a short time then cancel
        task = asyncio.create_task(background_health_check())
        await asyncio.sleep(0.1)  # Let it run briefly
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Verify initialization was called
        mock_monitor.start_monitoring.assert_called_once()
        mock_incident.start_incident_response.assert_called_once()
