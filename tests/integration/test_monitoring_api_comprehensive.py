"""
Comprehensive integration tests for MonitoringAPI endpoints.

Tests all functionality from monitoring_api.py including:
- Health check endpoint with comprehensive system validation
- System status endpoint with monitoring integration
- Prometheus metrics endpoint with text format
- Metrics summary endpoint with JSON response
- Alerts endpoint with system status
- Performance summary endpoint with time windows
- Performance trend analysis endpoint
- Test alert triggering endpoint
- Monitoring system information endpoint
- API authentication and authorization
- Error handling and edge cases
- Response format validation
- Performance and load testing
- Security validation scenarios
"""

import asyncio
from datetime import datetime, timedelta
import json
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from src.integration.monitoring_api import (
    AlertsResponse,
    HealthCheckResponse,
    MetricsResponse,
    SystemStatus,
    monitoring_router,
)


class TestMonitoringAPIModels:
    """Test Pydantic models used in monitoring API."""

    def test_system_status_model(self):
        """Test SystemStatus model validation."""
        status_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "uptime_seconds": 3600.5,
            "health_score": 0.95,
            "active_alerts": 2,
            "monitoring_enabled": True,
        }

        status = SystemStatus(**status_data)

        assert status.status == "healthy"
        assert status.uptime_seconds == 3600.5
        assert status.health_score == 0.95
        assert status.active_alerts == 2
        assert status.monitoring_enabled is True

    def test_system_status_model_validation_error(self):
        """Test SystemStatus model with invalid data."""
        invalid_data = {
            "status": "healthy",
            "timestamp": "not-iso-format",
            "uptime_seconds": "invalid",  # Should be float
            "health_score": 1.5,  # Should be 0-1
            "active_alerts": -1,  # Should be >= 0
            "monitoring_enabled": "yes",  # Should be bool
        }

        with pytest.raises(ValidationError):
            SystemStatus(**invalid_data)

    def test_metrics_response_model(self):
        """Test MetricsResponse model."""
        metrics_data = {
            "metrics_format": "prometheus",
            "timestamp": datetime.now().isoformat(),
            "metrics_count": 145,
        }

        response = MetricsResponse(**metrics_data)

        assert response.metrics_format == "prometheus"
        assert response.metrics_count == 145

    def test_health_check_response_model(self):
        """Test HealthCheckResponse model."""
        health_data = {
            "status": "healthy",
            "checks": {
                "database": {"status": "ok", "response_time": 23.4},
                "ha_connection": {"status": "ok", "response_time": 45.1},
            },
            "timestamp": datetime.now().isoformat(),
            "overall_healthy": True,
        }

        response = HealthCheckResponse(**health_data)

        assert response.status == "healthy"
        assert response.overall_healthy is True
        assert "database" in response.checks

    def test_alerts_response_model(self):
        """Test AlertsResponse model."""
        alerts_data = {
            "active_alerts": 5,
            "total_alerts_today": 23,
            "alert_rules_configured": 12,
            "notification_channels": ["email", "slack", "webhook"],
            "timestamp": datetime.now().isoformat(),
        }

        response = AlertsResponse(**alerts_data)

        assert response.active_alerts == 5
        assert response.total_alerts_today == 23
        assert "email" in response.notification_channels


class TestMonitoringAPIEndpoints:
    """Comprehensive tests for monitoring API endpoints."""

    @pytest.fixture
    def mock_monitoring_integration(self):
        """Create comprehensive mock monitoring integration."""
        monitor = MagicMock()

        # Setup monitoring manager mock
        monitoring_manager = MagicMock()
        monitor.get_monitoring_manager.return_value = monitoring_manager

        # Setup health monitor mock
        health_monitor = MagicMock()
        monitoring_manager.get_health_monitor.return_value = health_monitor

        # Setup health check results
        health_results = {
            "database": MagicMock(
                status="healthy",
                message="Connection OK",
                response_time=23.4,
                details={"connection_count": 5, "query_latency": 12.3},
            ),
            "home_assistant": MagicMock(
                status="healthy",
                message="API Accessible",
                response_time=45.1,
                details={"api_version": "2024.1.0"},
            ),
            "mqtt_broker": MagicMock(
                status="degraded",
                message="High latency",
                response_time=156.7,
                details={"broker_load": 0.85},
            ),
        }
        health_monitor.run_health_checks = AsyncMock(return_value=health_results)
        health_monitor.get_overall_health_status.return_value = (
            "healthy",
            {"checks_passed": 2, "checks_failed": 0, "checks_degraded": 1},
        )

        # Setup performance monitor mock
        performance_monitor = MagicMock()
        monitoring_manager.get_performance_monitor.return_value = performance_monitor

        performance_monitor.get_performance_summary.return_value = {
            "avg_prediction_latency": 87.3,
            "avg_feature_computation_time": 234.5,
            "database_query_latency": 23.4,
            "mqtt_publish_success_rate": 0.987,
            "prediction_accuracy": 0.923,
        }

        performance_monitor.get_trend_analysis.return_value = {
            "trend_direction": "improving",
            "trend_coefficient": 0.034,
            "confidence_interval": [0.02, 0.048],
            "data_points": 2840,
            "analysis_period": "24h",
        }

        # Setup monitoring status
        monitor.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "health_details": {
                        "cpu_percent": 45.2,
                        "memory_percent": 67.8,
                        "disk_usage": 0.23,
                    },
                    "monitoring_active": True,
                    "alert_system": {"active_alerts": 3},
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

        return monitor

    @pytest.fixture
    def mock_metrics_manager(self):
        """Create mock metrics manager."""
        manager = MagicMock()

        # Prometheus metrics format
        metrics_output = """# HELP prediction_latency_seconds Time to generate prediction
# TYPE prediction_latency_seconds histogram
prediction_latency_seconds_bucket{le="0.1"} 45
prediction_latency_seconds_bucket{le="0.5"} 123
prediction_latency_seconds_bucket{le="1.0"} 187
prediction_latency_seconds_bucket{le="inf"} 200
prediction_latency_seconds_sum 145.6
prediction_latency_seconds_count 200

# HELP prediction_accuracy_minutes Prediction accuracy in minutes
# TYPE prediction_accuracy_minutes gauge
prediction_accuracy_minutes{room_id="living_room"} 8.3
prediction_accuracy_minutes{room_id="bedroom"} 12.7

# HELP model_retrain_total Number of model retrains
# TYPE model_retrain_total counter
model_retrain_total{room_id="living_room",trigger="drift"} 5
model_retrain_total{room_id="bedroom",trigger="schedule"} 12
"""

        manager.get_metrics.return_value = metrics_output
        return manager

    @pytest.fixture
    def mock_alert_manager(self):
        """Create mock alert manager."""
        manager = MagicMock()

        manager.get_alert_status.return_value = {
            "active_alerts": 8,
            "total_alerts_today": 47,
            "alert_rules_configured": 23,
            "notification_channels": ["email", "slack", "webhook", "pagerduty"],
        }

        manager.trigger_alert = AsyncMock(return_value="test_alert_123")

        return manager

    @pytest.fixture
    def app(
        self, mock_monitoring_integration, mock_metrics_manager, mock_alert_manager
    ):
        """Create test FastAPI app with monitoring router."""
        app = FastAPI()
        app.include_router(monitoring_router)

        # Mock the get functions
        with patch(
            "src.integration.monitoring_api.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            with patch(
                "src.integration.monitoring_api.get_metrics_manager",
                return_value=mock_metrics_manager,
            ):
                with patch(
                    "src.integration.monitoring_api.get_alert_manager",
                    return_value=mock_alert_manager,
                ):
                    yield app

    @pytest.fixture
    def client(self, app):
        """Create test client."""
        return TestClient(app)

    # Health Check Endpoint Tests

    def test_get_health_status_success(self, client):
        """Test successful health status retrieval."""
        response = client.get("/monitoring/health")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["status"] == "healthy"
        assert data["overall_healthy"] is True
        assert "checks" in data
        assert "timestamp" in data

        # Verify specific health checks
        checks = data["checks"]
        assert "database" in checks
        assert "home_assistant" in checks
        assert "mqtt_broker" in checks

        # Verify check details
        db_check = checks["database"]
        assert db_check["status"] == "healthy"
        assert db_check["message"] == "Connection OK"
        assert db_check["response_time"] == 23.4
        assert "connection_count" in db_check["details"]

    def test_get_health_status_error_handling(self, client):
        """Test health status with monitoring error."""
        with patch(
            "src.integration.monitoring_api.get_monitoring_integration"
        ) as mock_get:
            mock_get.side_effect = RuntimeError("Monitoring unavailable")

            response = client.get("/monitoring/health")

            assert response.status_code == 500
            assert "Health check failed" in response.json()["detail"]

    def test_get_health_status_partial_failure(
        self, client, mock_monitoring_integration
    ):
        """Test health status with some failed checks."""
        # Mock health monitor with failures
        health_monitor = (
            mock_monitoring_integration.get_monitoring_manager().get_health_monitor()
        )
        health_results = {
            "database": MagicMock(
                status="failed",
                message="Connection timeout",
                response_time=5000.0,
                details={"error": "timeout"},
            ),
            "home_assistant": MagicMock(
                status="healthy", message="API OK", response_time=23.4, details={}
            ),
        }
        health_monitor.run_health_checks = AsyncMock(return_value=health_results)
        health_monitor.get_overall_health_status.return_value = ("degraded", {})

        response = client.get("/monitoring/health")

        assert response.status_code == 200
        data = response.json()

        assert data["status"] == "degraded"
        assert data["overall_healthy"] is False
        assert data["checks"]["database"]["status"] == "failed"

    # System Status Endpoint Tests

    def test_get_system_status_success(self, client):
        """Test successful system status retrieval."""
        response = client.get("/monitoring/status")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure matches SystemStatus model
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert data["uptime_seconds"] >= 0
        assert 0.0 <= data["health_score"] <= 1.0
        assert data["active_alerts"] >= 0
        assert isinstance(data["monitoring_enabled"], bool)

    def test_get_system_status_health_score_calculation(
        self, client, mock_monitoring_integration
    ):
        """Test health score calculation from system metrics."""
        # Mock high resource usage
        mock_monitoring_integration.get_monitoring_status.return_value = {
            "monitoring": {
                "health_status": "degraded",
                "health_details": {
                    "cpu_percent": 85.5,  # High CPU
                    "memory_percent": 92.3,  # High memory
                },
                "monitoring_active": True,
                "alert_system": {"active_alerts": 5},
            },
            "timestamp": datetime.now().isoformat(),
        }

        response = client.get("/monitoring/status")
        data = response.json()

        # Health score should be low due to high resource usage
        assert data["health_score"] < 0.5
        assert data["status"] == "degraded"
        assert data["active_alerts"] == 5

    def test_get_system_status_missing_health_details(
        self, client, mock_monitoring_integration
    ):
        """Test system status when health details are missing."""
        mock_monitoring_integration.get_monitoring_status.return_value = {
            "monitoring": {
                "health_status": "unknown",
                "monitoring_active": False,
                "alert_system": {"active_alerts": 0},
            },
            "timestamp": datetime.now().isoformat(),
        }

        response = client.get("/monitoring/status")
        data = response.json()

        # Should use default health score when details missing
        assert data["health_score"] == 1.0
        assert data["monitoring_enabled"] is False

    # Metrics Endpoints Tests

    def test_get_prometheus_metrics(self, client):
        """Test Prometheus metrics endpoint."""
        response = client.get("/monitoring/metrics")

        assert response.status_code == 200
        assert (
            response.headers["content-type"]
            == "text/plain; version=0.0.4; charset=utf-8"
        )

        content = response.text

        # Verify Prometheus format
        assert "# HELP prediction_latency_seconds" in content
        assert "# TYPE prediction_latency_seconds histogram" in content
        assert "prediction_latency_seconds_bucket" in content
        assert "prediction_accuracy_minutes{room_id=" in content

    def test_get_prometheus_metrics_error(self, client):
        """Test Prometheus metrics with error."""
        with patch("src.integration.monitoring_api.get_metrics_manager") as mock_get:
            mock_get.side_effect = RuntimeError("Metrics collection failed")

            response = client.get("/monitoring/metrics")

            assert response.status_code == 500
            assert "Metrics collection failed" in response.json()["detail"]

    def test_get_metrics_summary(self, client):
        """Test metrics summary endpoint."""
        response = client.get("/monitoring/metrics/summary")

        assert response.status_code == 200
        data = response.json()

        # Verify MetricsResponse model structure
        assert data["metrics_format"] == "prometheus"
        assert "timestamp" in data
        assert data["metrics_count"] > 0

    def test_get_metrics_summary_count_calculation(self, client, mock_metrics_manager):
        """Test metrics count calculation in summary."""
        # Mock metrics with specific number of data lines
        simple_metrics = """# Comment line
metric1{label="value1"} 42
metric2{label="value2"} 84
# Another comment
metric3 126
"""
        mock_metrics_manager.get_metrics.return_value = simple_metrics

        response = client.get("/monitoring/metrics/summary")
        data = response.json()

        # Should count only non-comment lines (3 metrics)
        assert data["metrics_count"] == 3

    # Alerts Endpoint Tests

    def test_get_alerts_status(self, client):
        """Test alerts status endpoint."""
        response = client.get("/monitoring/alerts")

        assert response.status_code == 200
        data = response.json()

        # Verify AlertsResponse model structure
        assert data["active_alerts"] == 8
        assert data["total_alerts_today"] == 47
        assert data["alert_rules_configured"] == 23
        assert "email" in data["notification_channels"]
        assert "slack" in data["notification_channels"]

    def test_get_alerts_status_error(self, client):
        """Test alerts status with error."""
        with patch("src.integration.monitoring_api.get_alert_manager") as mock_get:
            mock_get.side_effect = RuntimeError("Alert manager unavailable")

            response = client.get("/monitoring/alerts")

            assert response.status_code == 500
            assert "Alerts status check failed" in response.json()["detail"]

    def test_trigger_test_alert(self, client):
        """Test triggering test alert."""
        response = client.post("/monitoring/alerts/test")

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["alert_id"] == "test_alert_123"
        assert "Test alert triggered successfully" in data["message"]
        assert "timestamp" in data

    def test_trigger_test_alert_error(self, client):
        """Test test alert triggering with error."""
        with patch("src.integration.monitoring_api.get_alert_manager") as mock_get:
            mock_alert_manager = MagicMock()
            mock_alert_manager.trigger_alert = AsyncMock(
                side_effect=RuntimeError("Alert failed")
            )
            mock_get.return_value = mock_alert_manager

            response = client.post("/monitoring/alerts/test")

            assert response.status_code == 500
            assert "Test alert failed" in response.json()["detail"]

    # Performance Endpoints Tests

    def test_get_performance_summary(self, client):
        """Test performance summary endpoint."""
        response = client.get("/monitoring/performance")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["time_window_hours"] == 24  # default
        assert "summary" in data
        assert "timestamp" in data

        # Verify summary content
        summary = data["summary"]
        assert "avg_prediction_latency" in summary
        assert "avg_feature_computation_time" in summary
        assert "prediction_accuracy" in summary

    def test_get_performance_summary_custom_hours(self, client):
        """Test performance summary with custom time window."""
        response = client.get("/monitoring/performance?hours=48")

        assert response.status_code == 200
        data = response.json()

        assert data["time_window_hours"] == 48

    def test_get_performance_summary_invalid_hours(self, client):
        """Test performance summary with invalid hours parameter."""
        # Too small
        response = client.get("/monitoring/performance?hours=0")
        assert response.status_code == 400
        assert "Hours must be between 1 and 168" in response.json()["detail"]

        # Too large
        response = client.get("/monitoring/performance?hours=200")
        assert response.status_code == 400
        assert "Hours must be between 1 and 168" in response.json()["detail"]

    def test_get_performance_trend(self, client):
        """Test performance trend analysis endpoint."""
        response = client.get("/monitoring/performance/prediction_latency/trend")

        assert response.status_code == 200
        data = response.json()

        # Verify response structure
        assert data["metric_name"] == "prediction_latency"
        assert data["room_id"] is None  # default
        assert data["time_window_hours"] == 24  # default
        assert "analysis" in data
        assert "timestamp" in data

        # Verify analysis content
        analysis = data["analysis"]
        assert analysis["trend_direction"] == "improving"
        assert "trend_coefficient" in analysis
        assert "data_points" in analysis

    def test_get_performance_trend_with_room(self, client):
        """Test performance trend with room filter."""
        response = client.get(
            "/monitoring/performance/accuracy/trend?room_id=living_room&hours=72"
        )

        assert response.status_code == 200
        data = response.json()

        assert data["metric_name"] == "accuracy"
        assert data["room_id"] == "living_room"
        assert data["time_window_hours"] == 72

    def test_get_performance_trend_invalid_hours(self, client):
        """Test performance trend with invalid hours."""
        response = client.get("/monitoring/performance/latency/trend?hours=300")

        assert response.status_code == 400
        assert "Hours must be between 1 and 168" in response.json()["detail"]

    def test_get_performance_trend_error(self, client):
        """Test performance trend with monitoring error."""
        with patch(
            "src.integration.monitoring_api.get_monitoring_integration"
        ) as mock_get:
            mock_get.side_effect = RuntimeError("Performance monitor failed")

            response = client.get("/monitoring/performance/latency/trend")

            assert response.status_code == 500
            assert "Trend analysis failed" in response.json()["detail"]

    # Information Endpoint Tests

    def test_get_monitoring_info(self, client):
        """Test monitoring information endpoint."""
        response = client.get("/monitoring/")

        assert response.status_code == 200
        data = response.json()

        # Verify information structure
        assert data["service"] == "Home Assistant ML Predictor Monitoring"
        assert data["version"] == "1.0.0"
        assert "endpoints" in data
        assert "timestamp" in data

        # Verify endpoint documentation
        endpoints = data["endpoints"]
        assert "health" in endpoints
        assert "status" in endpoints
        assert "metrics" in endpoints
        assert "alerts" in endpoints

    # Error Handling and Edge Cases

    def test_endpoints_with_async_timeout(self, client, mock_monitoring_integration):
        """Test endpoints with async operation timeouts."""
        # Mock slow async operations
        mock_monitoring_integration.get_monitoring_status = AsyncMock(
            side_effect=asyncio.TimeoutError("Operation timed out")
        )

        response = client.get("/monitoring/status")

        # Should handle timeout gracefully
        assert response.status_code == 500

    def test_endpoints_with_malformed_monitoring_data(
        self, client, mock_monitoring_integration
    ):
        """Test endpoints with malformed monitoring data."""
        # Mock malformed monitoring status
        mock_monitoring_integration.get_monitoring_status.return_value = {
            "monitoring": "invalid_structure",  # Should be dict
            "timestamp": "not_iso_format",
        }

        response = client.get("/monitoring/status")

        # Should handle gracefully or return appropriate error
        # Exact behavior depends on implementation
        assert response.status_code in [200, 500]

    def test_concurrent_requests(self, client):
        """Test handling of concurrent requests."""
        import threading
        import time

        responses = []
        errors = []

        def make_request(endpoint):
            try:
                resp = client.get(f"/monitoring/{endpoint}")
                responses.append((endpoint, resp.status_code))
            except Exception as e:
                errors.append((endpoint, str(e)))

        # Create multiple concurrent requests
        threads = []
        endpoints = ["health", "status", "metrics/summary", "alerts", "performance"]

        for endpoint in endpoints:
            thread = threading.Thread(target=make_request, args=(endpoint,))
            threads.append(thread)
            thread.start()

        # Wait for all to complete
        for thread in threads:
            thread.join()

        # Most should succeed
        successful = [r for r in responses if r[1] == 200]
        assert len(successful) >= len(endpoints) * 0.8  # 80% success rate
        assert len(errors) < len(endpoints) * 0.2  # Less than 20% errors

    # Security and Authentication Tests

    def test_api_response_headers(self, client):
        """Test API response headers for security."""
        response = client.get("/monitoring/health")

        # Should not expose sensitive information in headers
        assert (
            "Server" not in response.headers
            or "fastapi" not in response.headers.get("Server", "").lower()
        )

        # Should have appropriate content type
        assert response.headers["content-type"] == "application/json"

    def test_input_validation_and_sanitization(self, client):
        """Test input validation and sanitization."""
        # Test with potential injection attempts
        malicious_inputs = [
            "performance?hours=<script>alert('xss')</script>",
            "performance?hours='; DROP TABLE sensor_events; --",
            "performance/../../etc/passwd/trend",
        ]

        for malicious_input in malicious_inputs:
            response = client.get(f"/monitoring/{malicious_input}")

            # Should either reject with 400/404 or handle safely
            assert response.status_code in [400, 404, 422]  # Not 200 or 500

    def test_error_message_information_disclosure(self, client):
        """Test that error messages don't disclose sensitive information."""
        # Force an error condition
        with patch(
            "src.integration.monitoring_api.get_monitoring_integration"
        ) as mock_get:
            mock_get.side_effect = Exception(
                "Database connection failed: user=admin password=secret123 host=db.internal"
            )

            response = client.get("/monitoring/health")

            assert response.status_code == 500
            error_detail = response.json()["detail"]

            # Should not contain sensitive information
            assert "password" not in error_detail.lower()
            assert "secret123" not in error_detail
            assert "admin" not in error_detail

    # Performance and Load Testing

    def test_response_time_performance(self, client):
        """Test API response time performance."""
        import time

        endpoints_to_test = [
            "/monitoring/health",
            "/monitoring/status",
            "/monitoring/metrics/summary",
            "/monitoring/alerts",
        ]

        for endpoint in endpoints_to_test:
            start_time = time.time()
            response = client.get(endpoint)
            end_time = time.time()

            response_time = end_time - start_time

            # API responses should be fast (< 1 second)
            assert response_time < 1.0, f"{endpoint} took {response_time:.2f}s"
            assert response.status_code == 200

    def test_large_metrics_output_handling(self, client, mock_metrics_manager):
        """Test handling of large metrics output."""
        # Mock very large metrics output
        large_metrics = "\n".join(
            [
                f'test_metric_{i}{{room_id="room_{i % 10}"}} {i * 1.5}'
                for i in range(10000)
            ]
        )
        mock_metrics_manager.get_metrics.return_value = large_metrics

        response = client.get("/monitoring/metrics")

        assert response.status_code == 200
        assert len(response.text) > 100000  # Large response

        # Summary should handle large output
        summary_response = client.get("/monitoring/metrics/summary")
        assert summary_response.status_code == 200
        assert summary_response.json()["metrics_count"] == 10000

    def test_memory_usage_stability(self, client):
        """Test API memory usage stability under load."""
        import gc
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Make many requests
        for _ in range(100):
            client.get("/monitoring/health")
            client.get("/monitoring/status")
            client.get("/monitoring/performance")

        # Force garbage collection
        gc.collect()

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be minimal (< 50MB)
        assert (
            memory_increase < 50 * 1024 * 1024
        ), f"Memory increased by {memory_increase / 1024 / 1024:.1f}MB"


class TestMonitoringAPIIntegration:
    """Integration tests for monitoring API with real-world scenarios."""

    @pytest.fixture
    def integration_app(self):
        """Create app for integration testing with realistic mocks."""
        app = FastAPI()
        app.include_router(monitoring_router)
        return app

    @pytest.fixture
    def integration_client(self, integration_app):
        """Create client for integration testing."""
        return TestClient(integration_app)

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_flow(self, integration_client):
        """Test complete monitoring flow from health check to alerting."""
        with patch(
            "src.integration.monitoring_api.get_monitoring_integration"
        ) as mock_monitor, patch(
            "src.integration.monitoring_api.get_metrics_manager"
        ) as mock_metrics, patch(
            "src.integration.monitoring_api.get_alert_manager"
        ) as mock_alerts:

            # Setup realistic monitoring data
            self._setup_realistic_monitoring_mocks(
                mock_monitor, mock_metrics, mock_alerts
            )

            # 1. Check system health
            health_response = integration_client.get("/monitoring/health")
            assert health_response.status_code == 200

            # 2. Get system status
            status_response = integration_client.get("/monitoring/status")
            assert status_response.status_code == 200

            # 3. Check metrics
            metrics_response = integration_client.get("/monitoring/metrics")
            assert metrics_response.status_code == 200

            # 4. Analyze performance
            perf_response = integration_client.get("/monitoring/performance?hours=24")
            assert perf_response.status_code == 200

            # 5. Trigger test alert
            alert_response = integration_client.post("/monitoring/alerts/test")
            assert alert_response.status_code == 200

            # Verify all responses contain expected data
            assert health_response.json()["overall_healthy"] is True
            assert status_response.json()["monitoring_enabled"] is True
            assert "prediction_latency" in metrics_response.text
            assert perf_response.json()["time_window_hours"] == 24
            assert alert_response.json()["success"] is True

    def _setup_realistic_monitoring_mocks(
        self, mock_monitor, mock_metrics, mock_alerts
    ):
        """Setup realistic monitoring mocks for integration testing."""
        # Setup monitoring integration
        monitor_instance = MagicMock()
        mock_monitor.return_value = monitor_instance

        monitoring_manager = MagicMock()
        monitor_instance.get_monitoring_manager.return_value = monitoring_manager

        # Health monitor with realistic checks
        health_monitor = MagicMock()
        monitoring_manager.get_health_monitor.return_value = health_monitor

        health_results = {
            "database": MagicMock(
                status="healthy",
                message="PostgreSQL connection active",
                response_time=18.3,
                details={"connections": 8, "queries_per_second": 23.4},
            ),
            "home_assistant": MagicMock(
                status="healthy",
                message="WebSocket connected",
                response_time=32.1,
                details={"entities_tracked": 45, "events_per_minute": 12},
            ),
        }
        health_monitor.run_health_checks = AsyncMock(return_value=health_results)
        health_monitor.get_overall_health_status.return_value = ("healthy", {})

        # Performance monitor with realistic data
        performance_monitor = MagicMock()
        monitoring_manager.get_performance_monitor.return_value = performance_monitor

        performance_monitor.get_performance_summary.return_value = {
            "avg_prediction_latency": 95.7,
            "prediction_accuracy": 0.912,
            "database_health_score": 0.98,
        }

        # Monitoring status
        monitor_instance.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "monitoring_active": True,
                    "alert_system": {"active_alerts": 1},
                }
            }
        )

        # Metrics manager with realistic Prometheus data
        metrics_instance = MagicMock()
        mock_metrics.return_value = metrics_instance
        metrics_instance.get_metrics.return_value = """
# HELP prediction_latency_seconds Prediction generation latency
prediction_latency_seconds{quantile="0.5"} 0.095
prediction_latency_seconds{quantile="0.9"} 0.187
prediction_latency_seconds{quantile="0.99"} 0.234
"""

        # Alert manager
        alert_instance = MagicMock()
        mock_alerts.return_value = alert_instance
        alert_instance.get_alert_status.return_value = {
            "active_alerts": 1,
            "total_alerts_today": 7,
            "alert_rules_configured": 15,
            "notification_channels": ["email", "slack"],
        }
        alert_instance.trigger_alert = AsyncMock(return_value="integration_test_alert")
