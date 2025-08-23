"""
Comprehensive integration test suite for Monitoring API endpoints.
Tests REST API functionality, authentication, error handling, and system integration.
"""

import asyncio
from datetime import datetime
import json
from unittest.mock import AsyncMock, Mock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient
import pytest

from src.integration.monitoring_api import (
    AlertsResponse,
    HealthCheckResponse,
    MetricsResponse,
    SystemStatus,
    monitoring_router,
)


class TestSystemStatusModel:
    """Test SystemStatus Pydantic model."""

    def test_system_status_creation(self):
        """Test SystemStatus model creation."""
        status = SystemStatus(
            status="healthy",
            timestamp="2024-01-15T10:30:00Z",
            uptime_seconds=3600.0,
            health_score=0.95,
            active_alerts=2,
            monitoring_enabled=True,
        )

        assert status.status == "healthy"
        assert status.timestamp == "2024-01-15T10:30:00Z"
        assert status.uptime_seconds == 3600.0
        assert status.health_score == 0.95
        assert status.active_alerts == 2
        assert status.monitoring_enabled is True

    def test_system_status_validation(self):
        """Test SystemStatus model validation."""
        # Test with minimal data
        status = SystemStatus(
            status="unhealthy",
            timestamp="2024-01-15T10:30:00Z",
            uptime_seconds=0.0,
            health_score=0.0,
            active_alerts=0,
            monitoring_enabled=False,
        )

        assert status.status == "unhealthy"
        assert status.health_score == 0.0
        assert status.monitoring_enabled is False


class TestMetricsResponseModel:
    """Test MetricsResponse Pydantic model."""

    def test_metrics_response_creation(self):
        """Test MetricsResponse model creation."""
        response = MetricsResponse(
            metrics_format="prometheus",
            timestamp="2024-01-15T10:30:00Z",
            metrics_count=45,
        )

        assert response.metrics_format == "prometheus"
        assert response.timestamp == "2024-01-15T10:30:00Z"
        assert response.metrics_count == 45


class TestHealthCheckResponseModel:
    """Test HealthCheckResponse Pydantic model."""

    def test_health_check_response_creation(self):
        """Test HealthCheckResponse model creation."""
        checks = {
            "database": {
                "status": "healthy",
                "message": "Connected",
                "response_time": 0.005,
                "details": {"connections": 5},
            },
            "mqtt": {
                "status": "unhealthy",
                "message": "Connection failed",
                "response_time": 1.0,
                "details": {"error": "timeout"},
            },
        }

        response = HealthCheckResponse(
            status="degraded",
            checks=checks,
            timestamp="2024-01-15T10:30:00Z",
            overall_healthy=False,
        )

        assert response.status == "degraded"
        assert len(response.checks) == 2
        assert response.overall_healthy is False


class TestAlertsResponseModel:
    """Test AlertsResponse Pydantic model."""

    def test_alerts_response_creation(self):
        """Test AlertsResponse model creation."""
        response = AlertsResponse(
            active_alerts=3,
            total_alerts_today=12,
            alert_rules_configured=25,
            notification_channels=["email", "slack"],
            timestamp="2024-01-15T10:30:00Z",
        )

        assert response.active_alerts == 3
        assert response.total_alerts_today == 12
        assert response.alert_rules_configured == 25
        assert response.notification_channels == ["email", "slack"]


class TestMonitoringAPIEndpoints:
    """Test monitoring API endpoints."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with monitoring router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @pytest.fixture
    def mock_monitoring_integration(self):
        """Mock monitoring integration."""
        integration = Mock()

        # Mock health monitor
        health_monitor = Mock()
        health_monitor.run_health_checks = AsyncMock(
            return_value={
                "database": Mock(
                    status="healthy",
                    message="Database connected",
                    response_time=0.005,
                    details={"connections": 5, "queries_per_second": 150},
                ),
                "mqtt": Mock(
                    status="healthy",
                    message="MQTT connected",
                    response_time=0.002,
                    details={"broker": "localhost", "topics": 15},
                ),
                "ha_api": Mock(
                    status="degraded",
                    message="High response time",
                    response_time=2.5,
                    details={"avg_response_time": 2.1, "timeout_rate": 0.05},
                ),
            }
        )
        health_monitor.get_overall_health_status = Mock(
            return_value=(
                "degraded",
                {"overall_score": 0.75, "degraded_services": ["ha_api"]},
            )
        )

        # Mock monitoring manager
        monitoring_manager = Mock()
        monitoring_manager.get_health_monitor = Mock(return_value=health_monitor)
        monitoring_manager.get_performance_monitor = Mock()

        # Mock performance monitor
        performance_monitor = Mock()
        performance_monitor.get_performance_summary = Mock(
            return_value={
                "prediction_accuracy": {"avg": 0.91, "min": 0.82, "max": 0.98},
                "response_times": {"avg": 0.045, "p95": 0.12, "p99": 0.25},
                "error_rates": {"prediction_errors": 0.02, "system_errors": 0.001},
                "throughput": {"predictions_per_minute": 42.5},
            }
        )
        performance_monitor.get_trend_analysis = Mock(
            return_value={
                "trend": "improving",
                "slope": 0.05,
                "correlation": 0.87,
                "data_points": 288,
                "time_range": "24h",
            }
        )
        monitoring_manager.get_performance_monitor = Mock(
            return_value=performance_monitor
        )

        integration.get_monitoring_manager = Mock(return_value=monitoring_manager)
        integration.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "monitoring_active": True,
                    "health_details": {
                        "cpu_percent": 15.5,
                        "memory_percent": 42.1,
                        "disk_usage": 68.3,
                    },
                    "alert_system": {
                        "active_alerts": 2,
                        "total_rules": 25,
                        "notification_channels": ["email", "webhook"],
                    },
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

        return integration

    @pytest.fixture
    def mock_metrics_manager(self):
        """Mock metrics manager."""
        manager = Mock()
        manager.get_metrics = Mock(
            return_value="""
# HELP prediction_accuracy_total Total prediction accuracy
# TYPE prediction_accuracy_total gauge
prediction_accuracy_total{room="living_room"} 0.92
prediction_accuracy_total{room="bedroom"} 0.89
prediction_accuracy_total{room="kitchen"} 0.95

# HELP prediction_latency_seconds Prediction generation latency
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 145
prediction_latency_seconds_bucket{le="0.5"} 248
prediction_latency_seconds_bucket{le="1.0"} 252
prediction_latency_seconds_bucket{le="+Inf"} 252
prediction_latency_seconds_sum 15.4
prediction_latency_seconds_count 252

# HELP system_uptime_seconds System uptime
# TYPE system_uptime_seconds counter
system_uptime_seconds 86400
        """.strip()
        )
        return manager

    @pytest.fixture
    def mock_alert_manager(self):
        """Mock alert manager."""
        manager = Mock()
        manager.get_alert_status = Mock(
            return_value={
                "active_alerts": 3,
                "total_alerts_today": 15,
                "alert_rules_configured": 28,
                "notification_channels": ["email", "slack", "webhook"],
                "last_alert_time": datetime.now().isoformat(),
            }
        )
        manager.trigger_alert = AsyncMock(return_value="alert_123456")
        return manager

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_health_status_success(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test successful health status endpoint."""
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get("/monitoring/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "degraded"
        assert data["overall_healthy"] is False
        assert "checks" in data
        assert len(data["checks"]) == 3
        assert "database" in data["checks"]
        assert "mqtt" in data["checks"]
        assert "ha_api" in data["checks"]
        assert "timestamp" in data

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_health_status_error(self, mock_get_integration, client):
        """Test health status endpoint with error."""
        mock_integration = Mock()
        mock_integration.get_monitoring_manager.side_effect = Exception(
            "Health check failed"
        )
        mock_get_integration.return_value = mock_integration

        response = client.get("/monitoring/health")

        assert response.status_code == 500
        assert "Health check failed" in response.json()["detail"]

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_system_status_success(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test successful system status endpoint."""
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get("/monitoring/status")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "healthy"
        assert data["health_score"] > 0
        assert data["active_alerts"] == 2
        assert data["monitoring_enabled"] is True
        assert "timestamp" in data
        assert "uptime_seconds" in data

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_system_status_with_high_resource_usage(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test system status with high resource usage affecting health score."""
        # Modify mock to simulate high resource usage
        mock_monitoring_integration.get_monitoring_status.return_value = {
            "monitoring": {
                "health_status": "degraded",
                "monitoring_active": True,
                "health_details": {
                    "cpu_percent": 95.0,  # High CPU
                    "memory_percent": 88.0,  # High memory
                    "disk_usage": 75.0,
                },
                "alert_system": {"active_alerts": 5, "total_rules": 25},
            },
            "timestamp": datetime.now().isoformat(),
        }
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get("/monitoring/status")

        assert response.status_code == 200
        data = response.json()

        # Health score should be low due to high resource usage
        assert data["health_score"] < 0.2
        assert data["active_alerts"] == 5

    @patch("src.integration.monitoring_api.get_metrics_manager")
    def test_get_prometheus_metrics_success(
        self, mock_get_manager, client, mock_metrics_manager
    ):
        """Test successful Prometheus metrics endpoint."""
        mock_get_manager.return_value = mock_metrics_manager

        response = client.get("/monitoring/metrics")

        assert response.status_code == 200
        assert response.headers["content-type"] == "text/plain; charset=utf-8"

        content = response.content.decode()
        assert "prediction_accuracy_total" in content
        assert "prediction_latency_seconds" in content
        assert "system_uptime_seconds" in content
        assert 'room="living_room"' in content

    @patch("src.integration.monitoring_api.get_metrics_manager")
    def test_get_prometheus_metrics_error(self, mock_get_manager, client):
        """Test Prometheus metrics endpoint with error."""
        mock_manager = Mock()
        mock_manager.get_metrics.side_effect = Exception("Metrics collection failed")
        mock_get_manager.return_value = mock_manager

        response = client.get("/monitoring/metrics")

        assert response.status_code == 500
        assert "Metrics collection failed" in response.json()["detail"]

    @patch("src.integration.monitoring_api.get_metrics_manager")
    def test_get_metrics_summary_success(
        self, mock_get_manager, client, mock_metrics_manager
    ):
        """Test successful metrics summary endpoint."""
        mock_get_manager.return_value = mock_metrics_manager

        response = client.get("/monitoring/metrics/summary")

        assert response.status_code == 200
        data = response.json()

        assert data["metrics_format"] == "prometheus"
        assert data["metrics_count"] > 0
        assert "timestamp" in data

    @patch("src.integration.monitoring_api.get_alert_manager")
    def test_get_alerts_status_success(
        self, mock_get_manager, client, mock_alert_manager
    ):
        """Test successful alerts status endpoint."""
        mock_get_manager.return_value = mock_alert_manager

        response = client.get("/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()

        assert data["active_alerts"] == 3
        assert data["total_alerts_today"] == 15
        assert data["alert_rules_configured"] == 28
        assert len(data["notification_channels"]) == 3
        assert "timestamp" in data

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_performance_summary_success(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test successful performance summary endpoint."""
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get("/monitoring/performance?hours=24")

        assert response.status_code == 200
        data = response.json()

        assert data["time_window_hours"] == 24
        assert "summary" in data
        assert "prediction_accuracy" in data["summary"]
        assert "response_times" in data["summary"]
        assert "timestamp" in data

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_performance_summary_default_hours(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test performance summary endpoint with default hours."""
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get("/monitoring/performance")

        assert response.status_code == 200
        data = response.json()

        assert data["time_window_hours"] == 24  # Default value

    def test_get_performance_summary_invalid_hours(self, client):
        """Test performance summary endpoint with invalid hours parameter."""
        # Test hours too low
        response = client.get("/monitoring/performance?hours=0")
        assert response.status_code == 400
        assert "must be between 1 and 168" in response.json()["detail"]

        # Test hours too high
        response = client.get("/monitoring/performance?hours=200")
        assert response.status_code == 400
        assert "must be between 1 and 168" in response.json()["detail"]

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_performance_trend_success(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test successful performance trend endpoint."""
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get(
            "/monitoring/performance/prediction_accuracy/trend?hours=48&room_id=living_room"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metric_name"] == "prediction_accuracy"
        assert data["room_id"] == "living_room"
        assert data["time_window_hours"] == 48
        assert "analysis" in data
        assert "timestamp" in data

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_get_performance_trend_without_room_id(
        self, mock_get_integration, client, mock_monitoring_integration
    ):
        """Test performance trend endpoint without room_id."""
        mock_get_integration.return_value = mock_monitoring_integration

        response = client.get("/monitoring/performance/response_time/trend?hours=12")

        assert response.status_code == 200
        data = response.json()

        assert data["metric_name"] == "response_time"
        assert data["room_id"] is None
        assert data["time_window_hours"] == 12

    def test_get_performance_trend_invalid_hours(self, client):
        """Test performance trend endpoint with invalid hours."""
        response = client.get("/monitoring/performance/accuracy/trend?hours=300")

        assert response.status_code == 400
        assert "must be between 1 and 168" in response.json()["detail"]

    @patch("src.integration.monitoring_api.get_alert_manager")
    def test_trigger_test_alert_success(
        self, mock_get_manager, client, mock_alert_manager
    ):
        """Test successful test alert trigger."""
        mock_get_manager.return_value = mock_alert_manager

        response = client.post("/monitoring/alerts/test")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["alert_id"] == "alert_123456"
        assert "Test alert triggered successfully" in data["message"]
        assert "timestamp" in data

    @patch("src.integration.monitoring_api.get_alert_manager")
    def test_trigger_test_alert_error(self, mock_get_manager, client):
        """Test test alert trigger with error."""
        mock_manager = Mock()
        mock_manager.trigger_alert.side_effect = Exception("Alert system unavailable")
        mock_get_manager.return_value = mock_manager

        response = client.post("/monitoring/alerts/test")

        assert response.status_code == 500
        assert "Alert system unavailable" in response.json()["detail"]

    def test_get_monitoring_info(self, client):
        """Test monitoring information endpoint."""
        response = client.get("/monitoring/")

        assert response.status_code == 200
        data = response.json()

        assert data["service"] == "Home Assistant ML Predictor Monitoring"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "timestamp" in data

        # Check endpoint documentation
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "status" in endpoints
        assert "metrics" in endpoints
        assert "alerts" in endpoints


class TestAPIIntegration:
    """Test API integration scenarios."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with monitoring router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    @patch("src.integration.monitoring_api.get_metrics_manager")
    @patch("src.integration.monitoring_api.get_alert_manager")
    def test_complete_monitoring_workflow(
        self,
        mock_alert_manager,
        mock_metrics_manager,
        mock_monitoring_integration,
        client,
    ):
        """Test complete monitoring workflow."""
        # Setup mocks
        monitoring_integration = Mock()
        monitoring_integration.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "monitoring_active": True,
                    "health_details": {"cpu_percent": 25.0, "memory_percent": 60.0},
                    "alert_system": {"active_alerts": 1},
                }
            }
        )

        health_monitor = Mock()
        health_monitor.run_health_checks = AsyncMock(
            return_value={
                "database": Mock(
                    status="healthy", message="OK", response_time=0.01, details={}
                )
            }
        )
        health_monitor.get_overall_health_status = Mock(return_value=("healthy", {}))

        monitoring_manager = Mock()
        monitoring_manager.get_health_monitor = Mock(return_value=health_monitor)
        monitoring_integration.get_monitoring_manager = Mock(
            return_value=monitoring_manager
        )

        metrics_manager = Mock()
        metrics_manager.get_metrics = Mock(
            return_value="# Test metric\ntest_metric 1.0"
        )

        alert_manager = Mock()
        alert_manager.get_alert_status = Mock(
            return_value={
                "active_alerts": 1,
                "total_alerts_today": 5,
                "alert_rules_configured": 20,
                "notification_channels": ["email"],
            }
        )
        alert_manager.trigger_alert = AsyncMock(return_value="test_alert_id")

        mock_monitoring_integration.return_value = monitoring_integration
        mock_metrics_manager.return_value = metrics_manager
        mock_alert_manager.return_value = alert_manager

        # Test complete workflow
        # 1. Check system status
        response = client.get("/monitoring/status")
        assert response.status_code == 200

        # 2. Check health
        response = client.get("/monitoring/health")
        assert response.status_code == 200

        # 3. Get metrics
        response = client.get("/monitoring/metrics")
        assert response.status_code == 200

        # 4. Check alerts
        response = client.get("/monitoring/alerts")
        assert response.status_code == 200

        # 5. Trigger test alert
        response = client.post("/monitoring/alerts/test")
        assert response.status_code == 200

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_partial_system_failure_handling(self, mock_get_integration, client):
        """Test handling of partial system failures."""
        # Simulate monitoring integration failure
        mock_get_integration.side_effect = Exception("Monitoring system unavailable")

        # Health endpoint should fail gracefully
        response = client.get("/monitoring/health")
        assert response.status_code == 500

        # Status endpoint should fail gracefully
        response = client.get("/monitoring/status")
        assert response.status_code == 500

    def test_concurrent_api_requests(self, client):
        """Test handling of concurrent API requests."""
        import threading
        import time

        results = []

        def make_request():
            response = client.get("/monitoring/")
            results.append(response.status_code)

        # Make multiple concurrent requests
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
            thread.start()

        # Wait for all requests to complete
        for thread in threads:
            thread.join()

        # All requests should succeed
        assert len(results) == 10
        assert all(status == 200 for status in results)


class TestAPIErrorHandling:
    """Test API error handling scenarios."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with monitoring router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_timeout_error_handling(self, mock_get_integration, client):
        """Test handling of timeout errors."""
        mock_integration = Mock()
        mock_integration.get_monitoring_status.side_effect = asyncio.TimeoutError(
            "Request timeout"
        )
        mock_get_integration.return_value = mock_integration

        response = client.get("/monitoring/status")

        assert response.status_code == 500
        # FastAPI converts the TimeoutError to a string in the detail
        assert "timeout" in response.json()["detail"].lower()

    @patch("src.integration.monitoring_api.get_metrics_manager")
    def test_memory_error_handling(self, mock_get_manager, client):
        """Test handling of memory errors."""
        mock_manager = Mock()
        mock_manager.get_metrics.side_effect = MemoryError("Out of memory")
        mock_get_manager.return_value = mock_manager

        response = client.get("/monitoring/metrics")

        assert response.status_code == 500

    @patch("src.integration.monitoring_api.get_alert_manager")
    def test_connection_error_handling(self, mock_get_manager, client):
        """Test handling of connection errors."""
        import requests

        mock_manager = Mock()
        mock_manager.trigger_alert.side_effect = requests.ConnectionError(
            "Connection failed"
        )
        mock_get_manager.return_value = mock_manager

        response = client.post("/monitoring/alerts/test")

        assert response.status_code == 500

    def test_invalid_endpoint_error(self, client):
        """Test handling of invalid endpoint requests."""
        response = client.get("/monitoring/nonexistent")

        assert response.status_code == 404

    def test_invalid_method_error(self, client):
        """Test handling of invalid HTTP methods."""
        response = client.delete("/monitoring/health")

        assert response.status_code == 405  # Method not allowed


class TestAPIDataValidation:
    """Test API data validation."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with monitoring router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_query_parameter_validation(self, client):
        """Test query parameter validation."""
        # Test invalid hours parameter types
        response = client.get("/monitoring/performance?hours=invalid")
        assert response.status_code == 422  # Validation error

        response = client.get("/monitoring/performance?hours=-5")
        assert response.status_code == 400  # Business logic validation

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_response_data_validation(self, mock_get_integration, client):
        """Test response data validation."""
        # Mock with invalid data structure
        mock_integration = Mock()
        mock_integration.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "monitoring_active": True,
                    "health_details": {"cpu_percent": 25.0},
                    "alert_system": {"active_alerts": 1},
                }
            }
        )
        mock_get_integration.return_value = mock_integration

        response = client.get("/monitoring/status")

        # Should still return valid response with defaults
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "timestamp" in data


class TestAPIPerformance:
    """Test API performance characteristics."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with monitoring router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    @patch("src.integration.monitoring_api.get_monitoring_integration")
    def test_response_time_performance(self, mock_get_integration, client):
        """Test API response time performance."""
        # Mock fast responses
        mock_integration = Mock()
        mock_integration.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "monitoring_active": True,
                    "health_details": {"cpu_percent": 10.0},
                    "alert_system": {"active_alerts": 0},
                }
            }
        )
        mock_get_integration.return_value = mock_integration

        import time

        start_time = time.time()
        response = client.get("/monitoring/status")
        end_time = time.time()

        assert response.status_code == 200
        # Response should be fast (less than 1 second for mocked calls)
        assert (end_time - start_time) < 1.0

    @patch("src.integration.monitoring_api.get_metrics_manager")
    def test_large_metrics_handling(self, mock_get_manager, client):
        """Test handling of large metrics responses."""
        # Generate large metrics output
        large_metrics = "\n".join(
            [f'test_metric_{i}{{room="room_{i % 10}"}} {i * 0.1}' for i in range(10000)]
        )

        mock_manager = Mock()
        mock_manager.get_metrics = Mock(return_value=large_metrics)
        mock_get_manager.return_value = mock_manager

        response = client.get("/monitoring/metrics")

        assert response.status_code == 200
        assert len(response.content) > 100000  # Should handle large responses

    def test_concurrent_request_handling(self, client):
        """Test concurrent request handling capacity."""
        import concurrent.futures
        import threading

        def make_request():
            response = client.get("/monitoring/")
            return response.status_code

        # Test with multiple concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = [executor.submit(make_request) for _ in range(50)]
            results = [
                future.result() for future in concurrent.futures.as_completed(futures)
            ]

        # All requests should succeed
        assert len(results) == 50
        assert all(status == 200 for status in results)


class TestAPISecurityAspects:
    """Test API security-related functionality."""

    @pytest.fixture
    def app(self):
        """Create FastAPI app with monitoring router."""
        from fastapi import FastAPI

        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    def test_malicious_query_parameters(self, client):
        """Test handling of malicious query parameters."""
        # Test SQL injection attempts
        malicious_params = [
            "'; DROP TABLE users; --",
            "<script>alert('xss')</script>",
            "../../../etc/passwd",
            "../../windows/system32/config/sam",
        ]

        for param in malicious_params:
            response = client.get(f"/monitoring/performance/trend/{param}/trend")
            # Should either validate properly or return 404/422, not crash
            assert response.status_code in [400, 404, 422, 500]

    def test_path_traversal_protection(self, client):
        """Test protection against path traversal attacks."""
        malicious_paths = [
            "../../../monitoring/health",
            "..%2F..%2F..%2Fhealth",
            "....//....//health",
        ]

        for path in malicious_paths:
            response = client.get(f"/monitoring/{path}")
            # Should not allow path traversal
            assert response.status_code in [404, 400]

    @patch("src.integration.monitoring_api.get_alert_manager")
    def test_alert_injection_protection(self, mock_get_manager, client):
        """Test protection against alert injection attacks."""
        mock_manager = Mock()
        mock_manager.trigger_alert = AsyncMock(return_value="safe_alert_id")
        mock_get_manager.return_value = mock_manager

        response = client.post("/monitoring/alerts/test")

        assert response.status_code == 200
        # Verify alert was triggered with safe, predefined content
        mock_manager.trigger_alert.assert_called_once()
        call_args = mock_manager.trigger_alert.call_args[1]
        assert call_args["rule_name"] == "api_test_alert"
        assert "test" in call_args["context"]

    def test_resource_exhaustion_protection(self, client):
        """Test protection against resource exhaustion."""
        # Test with extremely large hour values
        response = client.get("/monitoring/performance?hours=999999")

        # Should validate and reject excessive values
        assert response.status_code == 400

    def test_information_disclosure_prevention(self, client):
        """Test prevention of information disclosure."""
        # Request non-existent endpoints to check for information leakage
        response = client.get("/monitoring/internal/config")
        assert response.status_code == 404

        response = client.get("/monitoring/admin/users")
        assert response.status_code == 404

        # Error responses should not leak sensitive information
        response_data = response.json()
        sensitive_terms = ["password", "secret", "key", "token", "credential"]
        response_text = json.dumps(response_data).lower()

        for term in sensitive_terms:
            assert term not in response_text
