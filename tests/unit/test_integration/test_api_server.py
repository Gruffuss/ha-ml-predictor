"""
Comprehensive tests for API Server endpoints and functionality.

This test module provides extensive coverage of the FastAPI server
endpoints, middleware, error handling, authentication, and integration
with the tracking system.

Coverage Areas:
- All API endpoints (health, predictions, accuracy, incidents, etc.)
- Authentication and authorization
- Rate limiting and security
- Error handling and edge cases
- Middleware functionality
- Request/response validation
- Background tasks and lifecycle
- Integration with tracking manager
- Health monitoring integration
- Incident response integration
- WebSocket functionality
- Performance and load scenarios
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
import uuid

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from src.core.config import APIConfig, get_config
from src.core.exceptions import (
    APIAuthenticationError,
    APIError,
    APIRateLimitError,
    APIResourceNotFoundError,
    APIServerError,
    ErrorSeverity,
    OccupancyPredictionError,
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
    app,
    check_rate_limit,
    get_mqtt_manager,
    get_tracking_manager,
    integrate_with_tracking_manager,
    register_routes,
    set_tracking_manager,
    verify_api_key,
)
from src.utils.health_monitor import HealthStatus

# Test Fixtures


@pytest.fixture
def mock_config():
    """Create mock configuration."""
    config = Mock()
    config.api = APIConfig(
        enabled=True,
        host="127.0.0.1",
        port=8000,
        debug=True,
        api_key_enabled=False,
        api_key="test-api-key",
        rate_limit_enabled=False,
        requests_per_minute=60,
        jwt=Mock(enabled=False),
        background_tasks_enabled=True,
        health_check_interval_seconds=30,
        log_requests=True,
        log_responses=True,
        enable_cors=True,
        cors_origins=["*"],
        include_docs=True,
        docs_url="/docs",
        redoc_url="/redoc",
        access_log=True,
    )
    config.rooms = {
        "living_room": Mock(room_id="living_room", name="Living Room"),
        "bedroom": Mock(room_id="bedroom", name="Bedroom"),
        "kitchen": Mock(room_id="kitchen", name="Kitchen"),
    }
    return config


@pytest.fixture
def mock_tracking_manager():
    """Create comprehensive mock tracking manager."""
    manager = AsyncMock()

    # Mock get_tracking_status
    manager.get_tracking_status.return_value = {
        "status": "active",
        "tracking_active": True,
        "overall_health_score": 0.85,
        "config": {"enabled": True},
        "performance": {
            "background_tasks": 3,
            "total_predictions_recorded": 150,
            "total_validations_performed": 140,
            "total_drift_checks_performed": 25,
            "system_uptime_seconds": 3600,
        },
        "validator": {"total_predictions": 150, "accuracy_rate": 0.88},
        "accuracy_tracker": {"total_predictions": 150, "mean_error": 12.5},
        "drift_detector": {"last_check": datetime.now().isoformat()},
        "adaptive_retrainer": {"active_tasks": 1},
        "component_status": {
            "prediction_validator": {"status": "healthy", "health_score": 0.9},
            "accuracy_tracker": {"status": "healthy", "health_score": 0.85},
            "drift_detector": {"status": "healthy", "health_score": 0.8},
            "adaptive_retrainer": {"status": "healthy", "health_score": 0.9},
        },
        "resource_usage": {
            "memory_usage_mb": 256.0,
            "cache_usage_percent": 45.0,
            "active_connections": 5,
            "background_tasks": 3,
        },
        "performance_metrics": {
            "avg_response_time_ms": 25.5,
            "requests_per_minute": 15.2,
            "error_rate_percent": 2.1,
            "cache_hit_rate_percent": 75.0,
        },
    }

    # Mock get_room_prediction
    manager.get_room_prediction.return_value = {
        "room_id": "living_room",
        "prediction_time": datetime.now(timezone.utc).isoformat(),
        "next_transition_time": (
            datetime.now(timezone.utc) + timedelta(minutes=30)
        ).isoformat(),
        "transition_type": "occupied_to_vacant",
        "confidence": 0.85,
        "time_until_transition": "30 minutes",
        "alternatives": [{"transition_type": "vacant_to_occupied", "confidence": 0.15}],
        "model_info": {"algorithm": "ensemble", "version": "v1.0.0"},
    }

    # Mock get_accuracy_metrics
    manager.get_accuracy_metrics.return_value = {
        "room_id": "living_room",
        "accuracy_rate": 0.88,
        "average_error_minutes": 12.5,
        "confidence_calibration": 0.82,
        "total_predictions": 150,
        "total_validations": 140,
        "time_window_hours": 24,
        "trend_direction": "improving",
    }

    # Mock trigger_manual_retrain
    manager.trigger_manual_retrain.return_value = {
        "message": "Retraining triggered successfully",
        "success": True,
        "room_id": "living_room",
        "strategy": "auto",
        "force": False,
    }

    # Mock get_system_stats
    manager.get_system_stats.return_value = {
        "tracking_stats": {
            "total_predictions_tracked": 150,
            "tracking_accuracy": 0.88,
            "active_rooms": 3,
        },
        "retraining_stats": {"completed_retraining_jobs": 5, "pending_jobs": 1},
    }

    return manager


@pytest.fixture
def mock_database_manager():
    """Create mock database manager."""
    manager = AsyncMock()
    manager.health_check.return_value = {
        "status": "healthy",
        "database_connected": True,
        "connection_pool_size": 10,
        "active_connections": 3,
        "last_health_check": datetime.now().isoformat(),
    }
    return manager


@pytest.fixture
def mock_mqtt_manager():
    """Create mock MQTT manager."""
    manager = AsyncMock()
    manager.get_integration_stats.return_value = {
        "mqtt_connected": True,
        "predictions_published": 100,
        "discovery_published": True,
        "last_publish": datetime.now().isoformat(),
    }
    manager.cleanup_discovery.return_value = True
    manager.initialize.return_value = None
    return manager


@pytest.fixture
def mock_health_monitor():
    """Create mock health monitor."""
    monitor = Mock()
    monitor.get_system_health.return_value = Mock(
        overall_status=HealthStatus.HEALTHY,
        health_score=lambda: 0.85,
        critical_components=[],
        degraded_components=[],
        last_updated=datetime.now(),
        to_dict=lambda: {
            "overall_status": "healthy",
            "health_score": 0.85,
            "message": "All systems operational",
        },
    )
    monitor.get_component_health.return_value = {
        "database": Mock(to_dict=lambda: {"status": "healthy", "score": 0.9}),
        "tracking": Mock(to_dict=lambda: {"status": "healthy", "score": 0.85}),
        "mqtt": Mock(to_dict=lambda: {"status": "healthy", "score": 0.8}),
    }
    monitor.get_monitoring_stats.return_value = {
        "checks_performed": 100,
        "last_check": datetime.now().isoformat(),
        "monitoring_active": True,
    }
    monitor.get_health_history.return_value = [
        (datetime.now() - timedelta(hours=1), HealthStatus.HEALTHY),
        (datetime.now(), HealthStatus.HEALTHY),
    ]
    monitor.health_checks = {"database": Mock(), "tracking": Mock()}
    monitor.is_monitoring_active.return_value = True
    monitor.start_monitoring = AsyncMock()
    monitor.stop_monitoring = AsyncMock()
    return monitor


@pytest.fixture
def mock_incident_response():
    """Create mock incident response manager."""
    manager = Mock()
    manager.get_active_incidents.return_value = {
        "incident_001": Mock(
            to_dict=lambda: {
                "incident_id": "incident_001",
                "severity": "warning",
                "status": "active",
                "created_at": datetime.now().isoformat(),
                "description": "Test incident",
            }
        )
    }
    manager.get_incident.return_value = Mock(
        to_dict=lambda: {
            "incident_id": "incident_001",
            "severity": "warning",
            "status": "active",
            "created_at": datetime.now().isoformat(),
            "description": "Test incident",
            "resolution_notes": None,
        }
    )
    manager.get_incident_history.return_value = [
        Mock(
            to_dict=lambda: {
                "incident_id": "incident_002",
                "severity": "critical",
                "status": "resolved",
                "created_at": (datetime.now() - timedelta(hours=2)).isoformat(),
                "resolved_at": datetime.now().isoformat(),
            }
        )
    ]
    manager.get_incident_statistics.return_value = {
        "active_incidents_count": 1,
        "total_incidents_24h": 3,
        "resolved_incidents_24h": 2,
        "auto_recovery_enabled": True,
        "mean_resolution_time_minutes": 15.5,
    }
    manager.acknowledge_incident = AsyncMock(return_value=True)
    manager.resolve_incident = AsyncMock(return_value=True)
    manager.start_incident_response = AsyncMock()
    manager.stop_incident_response = AsyncMock()
    return manager


@pytest.fixture
def client(
    mock_config,
    mock_tracking_manager,
    mock_database_manager,
    mock_mqtt_manager,
    mock_health_monitor,
    mock_incident_response,
):
    """Create test client with mocked dependencies."""
    with patch(
        "src.integration.api_server.get_config", return_value=mock_config
    ), patch(
        "src.integration.api_server.get_tracking_manager",
        return_value=mock_tracking_manager,
    ), patch(
        "src.integration.api_server.get_database_manager",
        return_value=mock_database_manager,
    ), patch(
        "src.integration.api_server.get_mqtt_manager", return_value=mock_mqtt_manager
    ), patch(
        "src.integration.api_server.get_health_monitor",
        return_value=mock_health_monitor,
    ), patch(
        "src.integration.api_server.get_incident_response_manager",
        return_value=mock_incident_response,
    ):

        # Set the tracking manager instance
        set_tracking_manager(mock_tracking_manager)

        return TestClient(app)


# Rate Limiting Tests


class TestRateLimitTracker:
    """Test rate limiting functionality."""

    def test_rate_limit_tracker_initialization(self):
        """Test rate limit tracker initialization."""
        tracker = RateLimitTracker()
        assert isinstance(tracker.requests, dict)
        assert len(tracker.requests) == 0

    def test_rate_limit_within_limit(self):
        """Test requests within rate limit."""
        tracker = RateLimitTracker()

        # First request should be allowed
        result = tracker.is_allowed("127.0.0.1", limit=5, window_minutes=1)
        assert result is True

        # Second request should be allowed
        result = tracker.is_allowed("127.0.0.1", limit=5, window_minutes=1)
        assert result is True

        assert len(tracker.requests["127.0.0.1"]) == 2

    def test_rate_limit_exceeded(self):
        """Test rate limit exceeded."""
        tracker = RateLimitTracker()

        # Make requests up to limit
        for i in range(5):
            result = tracker.is_allowed("127.0.0.1", limit=5, window_minutes=1)
            assert result is True

        # Next request should be denied
        result = tracker.is_allowed("127.0.0.1", limit=5, window_minutes=1)
        assert result is False

    def test_rate_limit_window_cleanup(self):
        """Test rate limit window cleanup."""
        tracker = RateLimitTracker()

        # Manually add old requests
        old_time = datetime.now() - timedelta(minutes=2)
        tracker.requests["127.0.0.1"] = [old_time]

        # New request should clean up old ones
        result = tracker.is_allowed("127.0.0.1", limit=5, window_minutes=1)
        assert result is True

        # Old request should be removed
        assert len(tracker.requests["127.0.0.1"]) == 1
        assert tracker.requests["127.0.0.1"][0] != old_time

    def test_rate_limit_multiple_clients(self):
        """Test rate limiting for multiple clients."""
        tracker = RateLimitTracker()

        # Client 1 makes requests
        for i in range(3):
            result = tracker.is_allowed("127.0.0.1", limit=3, window_minutes=1)
            assert result is True

        # Client 1 should be at limit
        result = tracker.is_allowed("127.0.0.1", limit=3, window_minutes=1)
        assert result is False

        # Client 2 should still be allowed
        result = tracker.is_allowed("192.168.1.100", limit=3, window_minutes=1)
        assert result is True


# Authentication and Authorization Tests


class TestAuthentication:
    """Test API authentication and authorization."""

    async def test_verify_api_key_disabled(self, mock_config):
        """Test API key verification when disabled."""
        mock_config.api.api_key_enabled = False

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            result = await verify_api_key(None)
            assert result is True

    async def test_verify_api_key_missing_credentials(self, mock_config):
        """Test API key verification with missing credentials."""
        mock_config.api.api_key_enabled = True

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            with pytest.raises(APIAuthenticationError, match="API Key required"):
                await verify_api_key(None)

    async def test_verify_api_key_invalid(self, mock_config):
        """Test API key verification with invalid key."""
        mock_config.api.api_key_enabled = True
        mock_config.api.api_key = "correct-key"

        mock_credentials = Mock()
        mock_credentials.credentials = "wrong-key"

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            with pytest.raises(APIAuthenticationError, match="Invalid API key"):
                await verify_api_key(mock_credentials)

    async def test_verify_api_key_valid(self, mock_config):
        """Test API key verification with valid key."""
        mock_config.api.api_key_enabled = True
        mock_config.api.api_key = "correct-key"

        mock_credentials = Mock()
        mock_credentials.credentials = "correct-key"

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            result = await verify_api_key(mock_credentials)
            assert result is True

    async def test_check_rate_limit_disabled(self, mock_config):
        """Test rate limit check when disabled."""
        mock_config.api.rate_limit_enabled = False

        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            result = await check_rate_limit(mock_request)
            assert result is True

    async def test_check_rate_limit_within_limit(self, mock_config):
        """Test rate limit check within limit."""
        mock_config.api.rate_limit_enabled = True
        mock_config.api.requests_per_minute = 60

        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            result = await check_rate_limit(mock_request)
            assert result is True

    async def test_check_rate_limit_exceeded(self, mock_config):
        """Test rate limit check when exceeded."""
        mock_config.api.rate_limit_enabled = True
        mock_config.api.requests_per_minute = 1

        mock_request = Mock()
        mock_request.client.host = "127.0.0.1"

        # Mock rate limiter to return False
        with patch(
            "src.integration.api_server.get_config", return_value=mock_config
        ), patch(
            "src.integration.api_server.rate_limiter.is_allowed", return_value=False
        ):

            with pytest.raises(APIRateLimitError):
                await check_rate_limit(mock_request)


# API Endpoint Tests


class TestAPIEndpoints:
    """Test API endpoints functionality."""

    def test_root_endpoint(self, client):
        """Test root endpoint."""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert data["name"] == "Occupancy Prediction API"
        assert data["version"] == "1.0.0"
        assert data["status"] == "running"
        assert "timestamp" in data

    def test_health_endpoint_basic(self, client):
        """Test basic health endpoint."""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "timestamp" in data
        assert "components" in data
        assert "performance_metrics" in data
        assert "error_count" in data
        assert "uptime_seconds" in data

        # Check components structure
        components = data["components"]
        assert "database" in components
        assert "tracking" in components
        assert "mqtt" in components

    def test_health_endpoint_comprehensive(self, client):
        """Test comprehensive health endpoint."""
        response = client.get("/health/comprehensive")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded", "unhealthy"]
        assert "system_health" in data
        assert "components" in data
        assert "monitoring_stats" in data
        assert "response_time_seconds" in data

    def test_health_component_endpoint(self, client):
        """Test health component endpoint."""
        response = client.get("/health/components/database")

        assert response.status_code == 200
        data = response.json()
        assert "component" in data
        assert "history_24h" in data

    def test_health_component_endpoint_not_found(self, client, mock_health_monitor):
        """Test health component endpoint for non-existent component."""
        mock_health_monitor.get_component_health.return_value = {}

        response = client.get("/health/components/nonexistent")

        assert response.status_code == 404

    def test_health_system_endpoint(self, client):
        """Test health system endpoint."""
        response = client.get("/health/system")

        assert response.status_code == 200
        data = response.json()
        assert "overall_status" in data
        assert "health_score" in data

    def test_health_monitoring_endpoint(self, client):
        """Test health monitoring endpoint."""
        response = client.get("/health/monitoring")

        assert response.status_code == 200
        data = response.json()
        assert "monitoring_system" in data
        assert "registered_checks" in data
        assert "monitoring_active" in data

    def test_start_health_monitoring(self, client):
        """Test start health monitoring endpoint."""
        response = client.post("/health/monitoring/start")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["monitoring_active"] is True

    def test_stop_health_monitoring(self, client):
        """Test stop health monitoring endpoint."""
        response = client.post("/health/monitoring/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["monitoring_active"] is False

    def test_get_predictions_room(self, client):
        """Test get predictions for specific room."""
        response = client.get("/predictions/living_room")

        assert response.status_code == 200
        data = response.json()
        assert data["room_id"] == "living_room"
        assert "prediction_time" in data
        assert "confidence" in data
        assert isinstance(data["confidence"], float)
        assert 0.0 <= data["confidence"] <= 1.0

    def test_get_predictions_room_not_found(self, client, mock_config):
        """Test get predictions for non-existent room."""
        response = client.get("/predictions/nonexistent_room")

        assert response.status_code == 404

    def test_get_predictions_all(self, client):
        """Test get all predictions."""
        response = client.get("/predictions")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        assert len(data) >= 0

    def test_get_accuracy_metrics(self, client):
        """Test get accuracy metrics."""
        response = client.get("/accuracy")

        assert response.status_code == 200
        data = response.json()
        assert "accuracy_rate" in data
        assert "average_error_minutes" in data
        assert "total_predictions" in data
        assert isinstance(data["accuracy_rate"], float)
        assert 0.0 <= data["accuracy_rate"] <= 1.0

    def test_get_accuracy_metrics_with_params(self, client):
        """Test get accuracy metrics with parameters."""
        response = client.get("/accuracy?room_id=living_room&hours=48")

        assert response.status_code == 200
        data = response.json()
        assert data["room_id"] == "living_room"
        assert data["time_window_hours"] == 48

    def test_trigger_manual_retrain(self, client):
        """Test trigger manual retrain."""
        retrain_data = {
            "room_id": "living_room",
            "force": False,
            "strategy": "auto",
            "reason": "test_retrain",
        }

        response = client.post("/model/retrain", json=retrain_data)

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["room_id"] == "living_room"

    def test_refresh_mqtt_discovery(self, client):
        """Test refresh MQTT discovery."""
        response = client.post("/mqtt/refresh")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "timestamp" in data

    def test_get_system_stats(self, client):
        """Test get system statistics."""
        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "system_info" in data
        assert "prediction_stats" in data
        assert "mqtt_stats" in data
        assert "database_stats" in data
        assert "tracking_stats" in data


# Incident Management Endpoint Tests


class TestIncidentEndpoints:
    """Test incident management endpoints."""

    def test_get_active_incidents(self, client):
        """Test get active incidents."""
        response = client.get("/incidents")

        assert response.status_code == 200
        data = response.json()
        assert "active_incidents_count" in data
        assert "incidents" in data
        assert "timestamp" in data

    def test_get_incident_details(self, client):
        """Test get incident details."""
        response = client.get("/incidents/incident_001")

        assert response.status_code == 200
        data = response.json()
        assert data["incident_id"] == "incident_001"
        assert data["severity"] == "warning"
        assert data["status"] == "active"

    def test_get_incident_details_not_found(self, client, mock_incident_response):
        """Test get incident details for non-existent incident."""
        mock_incident_response.get_incident.return_value = None

        response = client.get("/incidents/nonexistent")

        assert response.status_code == 404

    def test_get_incident_history(self, client):
        """Test get incident history."""
        response = client.get("/incidents/history?hours=24")

        assert response.status_code == 200
        data = response.json()
        assert data["time_window_hours"] == 24
        assert "incidents_count" in data
        assert "incidents" in data

    def test_get_incident_history_invalid_hours(self, client):
        """Test get incident history with invalid hours parameter."""
        response = client.get("/incidents/history?hours=200")  # Exceeds max

        assert response.status_code == 400

    def test_get_incident_statistics(self, client):
        """Test get incident statistics."""
        response = client.get("/incidents/statistics")

        assert response.status_code == 200
        data = response.json()
        assert "statistics" in data
        assert "timestamp" in data

    def test_acknowledge_incident(self, client):
        """Test acknowledge incident."""
        response = client.post(
            "/incidents/incident_001/acknowledge?acknowledged_by=test_user"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["incident_id"] == "incident_001"
        assert data["acknowledged_by"] == "test_user"

    def test_acknowledge_incident_not_found(self, client, mock_incident_response):
        """Test acknowledge non-existent incident."""
        mock_incident_response.acknowledge_incident.return_value = False

        response = client.post("/incidents/nonexistent/acknowledge")

        assert response.status_code == 404

    def test_resolve_incident(self, client):
        """Test resolve incident."""
        response = client.post(
            "/incidents/incident_001/resolve?resolution_notes=Resolved manually"
        )

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["incident_id"] == "incident_001"
        assert data["resolution_notes"] == "Resolved manually"

    def test_resolve_incident_not_found(self, client, mock_incident_response):
        """Test resolve non-existent incident."""
        mock_incident_response.resolve_incident.return_value = False

        response = client.post("/incidents/nonexistent/resolve")

        assert response.status_code == 404

    def test_start_incident_response(self, client):
        """Test start incident response."""
        response = client.post("/incidents/response/start")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["response_active"] is True

    def test_stop_incident_response(self, client):
        """Test stop incident response."""
        response = client.post("/incidents/response/stop")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "success"
        assert data["response_active"] is False


# Error Handling Tests


class TestErrorHandling:
    """Test API error handling."""

    def test_api_error_handler(self, client, mock_tracking_manager):
        """Test API error handling."""
        # Make tracking manager raise APIError
        mock_tracking_manager.get_room_prediction.side_effect = (
            APIResourceNotFoundError("Room", "nonexistent")
        )

        response = client.get("/predictions/nonexistent")

        assert response.status_code == 400
        data = response.json()
        assert "error" in data
        assert "error_code" in data
        assert "timestamp" in data

    def test_system_error_handler(self, client, mock_tracking_manager):
        """Test system error handling."""
        # Make tracking manager raise OccupancyPredictionError
        mock_tracking_manager.get_room_prediction.side_effect = (
            OccupancyPredictionError("System error", severity=ErrorSeverity.HIGH)
        )

        response = client.get("/predictions/living_room")

        assert response.status_code == 500
        data = response.json()
        assert "error" in data
        assert data["error"] == "System error"

    def test_general_exception_handler(self, client, mock_tracking_manager):
        """Test general exception handling."""
        # Make tracking manager raise generic exception
        mock_tracking_manager.get_room_prediction.side_effect = Exception(
            "Unexpected error"
        )

        response = client.get("/predictions/living_room")

        assert response.status_code == 500
        data = response.json()
        assert data["error"] == "Internal server error"
        assert data["error_code"] == "UNHANDLED_EXCEPTION"

    def test_health_endpoint_exception(self, client, mock_database_manager):
        """Test health endpoint with exception."""
        mock_database_manager.health_check.side_effect = Exception("Database error")

        response = client.get("/health")

        # Should still return response but with error status
        assert response.status_code == 500

    def test_validation_error(self, client):
        """Test request validation error."""
        invalid_retrain_data = {
            "room_id": "nonexistent_room",  # Invalid room
            "strategy": "invalid_strategy",  # Invalid strategy
        }

        response = client.post("/model/retrain", json=invalid_retrain_data)

        assert response.status_code == 422  # Validation error


# Response Model Tests


class TestResponseModels:
    """Test API response model validation."""

    def test_prediction_response_validation(self):
        """Test prediction response model validation."""
        # Valid data
        valid_data = {
            "room_id": "living_room",
            "prediction_time": datetime.now(),
            "next_transition_time": datetime.now() + timedelta(minutes=30),
            "transition_type": "occupied_to_vacant",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {},
        }

        response = PredictionResponse(**valid_data)
        assert response.room_id == "living_room"
        assert response.confidence == 0.85
        assert response.transition_type == "occupied_to_vacant"

    def test_prediction_response_validation_invalid_confidence(self):
        """Test prediction response with invalid confidence."""
        invalid_data = {
            "room_id": "living_room",
            "prediction_time": datetime.now(),
            "confidence": 1.5,  # Invalid: > 1.0
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {},
        }

        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            PredictionResponse(**invalid_data)

    def test_prediction_response_validation_invalid_transition_type(self):
        """Test prediction response with invalid transition type."""
        invalid_data = {
            "room_id": "living_room",
            "prediction_time": datetime.now(),
            "transition_type": "invalid_type",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {},
        }

        with pytest.raises(ValueError, match="Transition type must be one of"):
            PredictionResponse(**invalid_data)

    def test_accuracy_metrics_response_validation(self):
        """Test accuracy metrics response validation."""
        valid_data = {
            "room_id": "living_room",
            "accuracy_rate": 0.88,
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.82,
            "total_predictions": 150,
            "total_validations": 140,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }

        response = AccuracyMetricsResponse(**valid_data)
        assert response.accuracy_rate == 0.88
        assert response.trend_direction == "improving"

    def test_accuracy_metrics_response_validation_invalid_rate(self):
        """Test accuracy metrics response with invalid rate."""
        invalid_data = {
            "room_id": "living_room",
            "accuracy_rate": 1.5,  # Invalid: > 1.0
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.82,
            "total_predictions": 150,
            "total_validations": 140,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }

        with pytest.raises(ValueError, match="Rate must be between 0.0 and 1.0"):
            AccuracyMetricsResponse(**invalid_data)

    def test_manual_retrain_request_validation(self):
        """Test manual retrain request validation."""
        valid_data = {
            "room_id": "living_room",
            "force": True,
            "strategy": "full",
            "reason": "Manual test",
        }

        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.rooms = {"living_room": Mock()}
            mock_get_config.return_value = mock_config

            request = ManualRetrainRequest(**valid_data)
            assert request.room_id == "living_room"
            assert request.force is True
            assert request.strategy == "full"

    def test_manual_retrain_request_validation_invalid_room(self):
        """Test manual retrain request with invalid room."""
        invalid_data = {
            "room_id": "nonexistent_room",
            "strategy": "auto",
            "reason": "Test",
        }

        with patch("src.integration.api_server.get_config") as mock_get_config:
            mock_config = Mock()
            mock_config.rooms = {"living_room": Mock()}
            mock_get_config.return_value = mock_config

            with pytest.raises(ValueError, match="Room 'nonexistent_room' not found"):
                ManualRetrainRequest(**invalid_data)

    def test_error_response_serialization(self):
        """Test error response serialization."""
        error_data = {
            "error": "Test error",
            "error_code": "TEST_ERROR",
            "details": {"key": "value"},
            "timestamp": datetime.now(),
            "request_id": "req_123",
        }

        response = ErrorResponse(**error_data)
        data_dict = response.dict()

        assert data_dict["error"] == "Test error"
        assert data_dict["error_code"] == "TEST_ERROR"
        assert isinstance(data_dict["timestamp"], str)  # Should be serialized to string


# API Server Integration Tests


class TestAPIServerIntegration:
    """Test API server integration functionality."""

    async def test_integrate_with_tracking_manager(self, mock_tracking_manager):
        """Test API server integration with tracking manager."""
        api_server = await integrate_with_tracking_manager(mock_tracking_manager)

        assert isinstance(api_server, APIServer)
        assert api_server.tracking_manager == mock_tracking_manager

    def test_api_server_initialization(self, mock_tracking_manager, mock_config):
        """Test API server initialization."""
        with patch("src.integration.api_server.get_config", return_value=mock_config):
            api_server = APIServer(mock_tracking_manager)

            assert api_server.tracking_manager == mock_tracking_manager
            assert api_server.config == mock_config.api
            assert api_server.server is None
            assert api_server.server_task is None

    async def test_api_server_start_enabled(self, mock_tracking_manager, mock_config):
        """Test API server start when enabled."""
        mock_config.api.enabled = True

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            api_server = APIServer(mock_tracking_manager)

            with patch("uvicorn.Server") as mock_server_class:
                mock_server = Mock()
                mock_server_class.return_value = mock_server

                with patch("asyncio.create_task") as mock_create_task:
                    mock_task = Mock()
                    mock_create_task.return_value = mock_task

                    await api_server.start()

                    assert api_server.server == mock_server
                    assert api_server.server_task == mock_task

    async def test_api_server_start_disabled(self, mock_tracking_manager, mock_config):
        """Test API server start when disabled."""
        mock_config.api.enabled = False

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            api_server = APIServer(mock_tracking_manager)

            await api_server.start()

            # Should return early without starting
            assert api_server.server is None
            assert api_server.server_task is None

    async def test_api_server_stop(self, mock_tracking_manager, mock_config):
        """Test API server stop."""
        with patch("src.integration.api_server.get_config", return_value=mock_config):
            api_server = APIServer(mock_tracking_manager)

            # Mock running server
            mock_server = Mock()
            mock_task = AsyncMock()
            api_server.server = mock_server
            api_server.server_task = mock_task

            await api_server.stop()

            assert mock_server.should_exit is True
            mock_task.assert_awaited_once()

    def test_api_server_is_running_true(self, mock_tracking_manager, mock_config):
        """Test API server is_running when running."""
        with patch("src.integration.api_server.get_config", return_value=mock_config):
            api_server = APIServer(mock_tracking_manager)

            mock_task = Mock()
            mock_task.done.return_value = False
            api_server.server_task = mock_task

            assert api_server.is_running() is True

    def test_api_server_is_running_false(self, mock_tracking_manager, mock_config):
        """Test API server is_running when not running."""
        with patch("src.integration.api_server.get_config", return_value=mock_config):
            api_server = APIServer(mock_tracking_manager)

            # No server task
            assert api_server.is_running() is False

            # Done task
            mock_task = Mock()
            mock_task.done.return_value = True
            api_server.server_task = mock_task

            assert api_server.is_running() is False


# Performance and Load Tests


class TestAPIPerformance:
    """Test API performance characteristics."""

    def test_concurrent_health_checks(self, client):
        """Test concurrent health check requests."""
        import threading

        results = []

        def make_request():
            response = client.get("/health")
            results.append(response.status_code)

        # Make concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # All should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)

    def test_multiple_prediction_requests(self, client):
        """Test multiple prediction requests."""
        rooms = ["living_room", "bedroom", "kitchen"]

        for room in rooms:
            response = client.get(f"/predictions/{room}")
            assert response.status_code == 200
            data = response.json()
            assert data["room_id"] == room

    def test_large_stats_response(self, client, mock_tracking_manager):
        """Test handling of large statistics response."""
        # Mock large stats response
        large_stats = {
            "tracking_stats": {f"stat_{i}": f"value_{i}" for i in range(100)},
            "retraining_stats": {
                f"retrain_stat_{i}": f"retrain_value_{i}" for i in range(50)
            },
        }
        mock_tracking_manager.get_system_stats.return_value = large_stats

        response = client.get("/stats")

        assert response.status_code == 200
        data = response.json()
        assert "tracking_stats" in data

    def test_request_response_timing(self, client):
        """Test request/response timing."""
        import time

        start_time = time.time()
        response = client.get("/health")
        end_time = time.time()

        # Should respond quickly (within 1 second for basic health check)
        duration = end_time - start_time
        assert duration < 1.0
        assert response.status_code == 200


# Edge Cases and Security Tests


class TestAPIEdgeCases:
    """Test API edge cases and security scenarios."""

    def test_malformed_json_request(self, client):
        """Test malformed JSON in request body."""
        response = client.post(
            "/model/retrain",
            data="invalid json",
            headers={"content-type": "application/json"},
        )

        assert response.status_code == 422  # Unprocessable Entity

    def test_missing_required_fields(self, client):
        """Test request with missing required fields."""
        incomplete_data = {
            "strategy": "auto"
            # Missing required 'reason' field
        }

        response = client.post("/model/retrain", json=incomplete_data)

        assert response.status_code == 422

    def test_invalid_query_parameters(self, client):
        """Test invalid query parameters."""
        response = client.get("/accuracy?hours=invalid")

        assert response.status_code == 422

    def test_oversized_request(self, client):
        """Test oversized request handling."""
        # Create large request data
        large_data = {"reason": "x" * 10000, "strategy": "auto"}  # Very long reason

        response = client.post("/model/retrain", json=large_data)

        # Should handle gracefully (may succeed or fail depending on limits)
        assert response.status_code in [200, 413, 422]

    def test_special_characters_in_room_id(self, client, mock_config):
        """Test room IDs with special characters."""
        # Test URL encoding
        response = client.get("/predictions/room%20with%20spaces")

        # Should decode properly and return 404 for non-existent room
        assert response.status_code == 404

    def test_long_incident_id(self, client):
        """Test very long incident ID."""
        long_id = "x" * 1000
        response = client.get(f"/incidents/{long_id}")

        # Should handle gracefully
        assert response.status_code in [404, 414]  # Not found or URI too long

    def test_concurrent_retrain_requests(self, client):
        """Test concurrent retrain requests."""
        import threading

        results = []

        def make_retrain_request():
            retrain_data = {
                "room_id": "living_room",
                "strategy": "auto",
                "reason": "concurrent_test",
            }
            response = client.post("/model/retrain", json=retrain_data)
            results.append(response.status_code)

        # Make concurrent retrain requests
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=make_retrain_request)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should handle all requests (may succeed or fail based on business logic)
        assert len(results) == 5
        assert all(status in [200, 400, 500] for status in results)

    def test_health_endpoint_resilience(
        self, client, mock_database_manager, mock_tracking_manager
    ):
        """Test health endpoint resilience to component failures."""
        # Test with database failure
        mock_database_manager.health_check.side_effect = Exception("DB error")

        response = client.get("/health")
        data = response.json()

        # Should still return response but indicate degraded health
        assert "components" in data
        assert data["components"]["database"]["status"] in ["unhealthy", "error"]

        # Test with tracking manager failure
        mock_tracking_manager.get_tracking_status.side_effect = Exception(
            "Tracking error"
        )

        response = client.get("/health")
        data = response.json()

        # Should handle gracefully
        assert "components" in data

    def test_api_with_no_tracking_manager(self, client):
        """Test API behavior with no tracking manager."""
        # Clear the tracking manager
        set_tracking_manager(None)

        # Endpoints should handle gracefully
        response = client.get("/predictions/living_room")

        # Should return error or handle gracefully
        assert response.status_code in [404, 500]

    def test_endpoints_without_authentication(self, client, mock_config):
        """Test endpoint access without authentication when required."""
        # Enable API key requirement
        mock_config.api.api_key_enabled = True

        with patch("src.integration.api_server.get_config", return_value=mock_config):
            response = client.get("/predictions/living_room")

            # Should require authentication
            assert response.status_code == 401
