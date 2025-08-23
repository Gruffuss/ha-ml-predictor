"""
Comprehensive tests for Performance Monitoring Dashboard.

This test module provides extensive coverage of the dashboard functionality
including FastAPI endpoints, WebSocket management, data aggregation,
caching, and error handling scenarios.

Coverage Areas:
- Dashboard initialization and configuration
- FastAPI app creation and route registration
- WebSocket connection management and broadcasting
- System overview data aggregation
- Accuracy metrics processing
- Drift status monitoring
- Retraining status tracking
- System health reporting
- Alert management
- Historical trends analysis
- Service command handling
- Cache management
- Error handling and edge cases
- Security scenarios
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import logging
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

import pytest

from src.core.exceptions import ErrorSeverity, OccupancyPredictionError
from src.integration.dashboard import (
    DashboardConfig,
    DashboardError,
    DashboardMode,
    MetricType,
    PerformanceDashboard,
    SystemOverview,
    WebSocketManager,
    create_dashboard_from_tracking_manager,
    integrate_dashboard_with_tracking_system,
)

# Test Fixtures


@pytest.fixture
def mock_tracking_manager():
    """Create mock tracking manager with comprehensive status."""
    manager = AsyncMock()

    # Mock get_tracking_status method
    manager.get_tracking_status.return_value = {
        "status": "active",
        "overall_health_score": 0.85,
        "accuracy_summary": {
            "total_predictions_24h": 150,
            "accuracy_rate_24h": 0.88,
            "mean_error_minutes_24h": 12.5,
            "predictions_per_hour": 6.25,
        },
        "alerts_summary": {
            "total_active": 2,
            "critical_count": 0,
            "warning_count": 2,
        },
        "drift_summary": {
            "rooms_with_drift": 1,
        },
        "retraining_summary": {
            "active_tasks": 0,
            "completed_24h": 3,
        },
        "performance_metrics": {
            "avg_prediction_latency_ms": 45.2,
            "avg_validation_lag_minutes": 2.1,
            "cache_hit_rate": 0.75,
        },
        "tracked_rooms": ["living_room", "bedroom", "kitchen"],
        "tracked_models": ["ensemble_v1", "lstm_v2"],
        "component_status": {
            "prediction_validator": {"status": "healthy", "health_score": 0.9},
            "accuracy_tracker": {"status": "healthy", "health_score": 0.85},
            "drift_detector": {"status": "warning", "health_score": 0.7},
            "adaptive_retrainer": {"status": "healthy", "health_score": 0.8},
        },
        "resource_usage": {
            "memory_usage_mb": 256.5,
            "cache_usage_percent": 45.2,
            "active_connections": 3,
            "background_tasks": 5,
        },
    }

    # Mock real-time metrics
    manager.get_real_time_metrics.return_value = [
        {
            "room_id": "living_room",
            "accuracy_rate": 0.92,
            "confidence": 0.87,
            "last_prediction": datetime.now(timezone.utc).isoformat(),
        },
        {
            "room_id": "bedroom",
            "accuracy_rate": 0.84,
            "confidence": 0.79,
            "last_prediction": datetime.now(timezone.utc).isoformat(),
        },
    ]

    # Mock drift status
    manager.get_drift_status.return_value = {
        "summary": {
            "monitored_rooms": 3,
            "rooms_with_drift": 1,
            "rooms_with_major_drift": 0,
            "last_check_time": datetime.now(timezone.utc).isoformat(),
            "next_check_time": (
                datetime.now(timezone.utc) + timedelta(hours=1)
            ).isoformat(),
        },
        "room_drift_data": {
            "living_room": {
                "drift_score": 0.15,
                "drift_detected": False,
                "last_check": datetime.now(timezone.utc).isoformat(),
            },
            "bedroom": {
                "drift_score": 0.45,
                "drift_detected": True,
                "last_check": datetime.now(timezone.utc).isoformat(),
            },
        },
        "drift_types_summary": {
            "temporal_drift": 1,
            "concept_drift": 0,
        },
        "recent_events": [
            {
                "room_id": "bedroom",
                "event_type": "drift_detected",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": {"drift_score": 0.45},
            }
        ],
        "configuration": {
            "drift_threshold": 0.3,
            "check_interval_hours": 1,
        },
    }

    # Mock retraining status
    manager.get_retraining_status.return_value = {
        "queue_summary": {
            "pending_count": 2,
            "active_count": 1,
            "completed_today": 3,
            "failed_today": 0,
            "total_queue_size": 3,
        },
        "active_tasks": [
            {
                "task_id": "retrain_001",
                "room_id": "kitchen",
                "strategy": "incremental",
                "started_at": datetime.now(timezone.utc).isoformat(),
                "estimated_completion": (
                    datetime.now(timezone.utc) + timedelta(minutes=30)
                ).isoformat(),
            }
        ],
        "recent_completions": [
            {
                "task_id": "retrain_002",
                "room_id": "living_room",
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "improvement_percent": 5.2,
            }
        ],
        "performance_improvements": {
            "living_room": {"before": 0.82, "after": 0.87},
            "bedroom": {"before": 0.78, "after": 0.84},
        },
    }

    # Mock active alerts
    manager.get_active_alerts.return_value = [
        {
            "alert_id": "alert_001",
            "room_id": "bedroom",
            "severity": "warning",
            "alert_type": "drift_detected",
            "message": "Concept drift detected",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        {
            "alert_id": "alert_002",
            "room_id": "kitchen",
            "severity": "warning",
            "alert_type": "accuracy_degraded",
            "message": "Accuracy below threshold",
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
    ]

    # Mock accuracy tracker
    manager.accuracy_tracker = AsyncMock()
    manager.accuracy_tracker.get_accuracy_trends.return_value = {
        "accuracy_trends": [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "accuracy": 0.85},
            {
                "timestamp": (
                    datetime.now(timezone.utc) - timedelta(hours=1)
                ).isoformat(),
                "accuracy": 0.87,
            },
        ],
        "error_trends": [
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error_minutes": 12.5,
            },
            {
                "timestamp": (
                    datetime.now(timezone.utc) - timedelta(hours=1)
                ).isoformat(),
                "error_minutes": 11.8,
            },
        ],
        "prediction_volume_trends": [
            {"timestamp": datetime.now(timezone.utc).isoformat(), "count": 25},
            {
                "timestamp": (
                    datetime.now(timezone.utc) - timedelta(hours=1)
                ).isoformat(),
                "count": 22,
            },
        ],
        "trend_analysis": {
            "trend_direction": "stable",
            "trend_strength": 0.1,
        },
    }

    # Mock validator with accuracy metrics
    manager.validator = AsyncMock()
    manager.validator.get_accuracy_metrics.return_value = {
        "total_predictions": 150,
        "validated_predictions": 142,
        "accuracy_rate": 0.88,
        "mean_error_minutes": 12.5,
        "median_error_minutes": 9.2,
        "validation_rate": 0.95,
        "error_percentiles": {
            "p50": 9.2,
            "p90": 22.1,
            "p95": 28.5,
        },
        "mean_confidence": 0.82,
        "confidence_calibration_score": 0.76,
        "overconfidence_rate": 0.15,
        "to_dict": lambda: {
            "total_predictions": 150,
            "validated_predictions": 142,
            "accuracy_rate": 0.88,
            "mean_error_minutes": 12.5,
            "median_error_minutes": 9.2,
            "validation_rate": 0.95,
        },
    }

    # Mock service methods
    manager.request_manual_retraining.return_value = {
        "request_id": "req_001",
        "estimated_completion": (
            datetime.now(timezone.utc) + timedelta(hours=1)
        ).isoformat(),
    }

    manager.acknowledge_alert.return_value = {
        "status": "acknowledged",
        "previous_status": "active",
        "acknowledgment_count": 1,
    }

    return manager


@pytest.fixture
def dashboard_config():
    """Create dashboard configuration for testing."""
    return DashboardConfig(
        enabled=True,
        host="127.0.0.1",
        port=8889,  # Different port to avoid conflicts
        debug=True,
        mode=DashboardMode.DEVELOPMENT,
        websocket_enabled=True,
        update_interval_seconds=1,
        max_websocket_connections=10,
        cache_ttl_seconds=5,
    )


@pytest.fixture
def dashboard_config_production():
    """Create production dashboard configuration."""
    return DashboardConfig(
        enabled=True,
        host="0.0.0.0",
        port=8888,
        debug=False,
        mode=DashboardMode.PRODUCTION,
        websocket_enabled=True,
        update_interval_seconds=30,
        max_websocket_connections=50,
        enable_retraining_controls=True,
        enable_alert_management=True,
        cache_ttl_seconds=60,
    )


@pytest.fixture
def dashboard_config_readonly():
    """Create readonly dashboard configuration."""
    return DashboardConfig(
        enabled=True,
        host="0.0.0.0",
        port=8888,
        debug=False,
        mode=DashboardMode.READONLY,
        websocket_enabled=True,
        enable_retraining_controls=False,
        enable_alert_management=False,
        cache_ttl_seconds=30,
    )


@pytest.fixture
def mock_websocket():
    """Create mock WebSocket for testing."""
    websocket = AsyncMock()
    websocket.accept = AsyncMock()
    websocket.send_text = AsyncMock()
    websocket.receive_text = AsyncMock()
    websocket.close = AsyncMock()
    websocket.client = Mock()
    websocket.client.host = "127.0.0.1"
    websocket.client.port = 12345
    return websocket


# Dashboard Configuration Tests


class TestDashboardConfig:
    """Test dashboard configuration and validation."""

    def test_default_config(self):
        """Test default configuration values."""
        config = DashboardConfig()

        assert config.enabled is True
        assert config.host == "0.0.0.0"
        assert config.port == 8888
        assert config.debug is False
        assert config.mode == DashboardMode.PRODUCTION
        assert config.websocket_enabled is True
        assert config.update_interval_seconds == 5
        assert config.max_websocket_connections == 50
        assert config.metrics_retention_hours == 72
        assert config.cache_ttl_seconds == 30
        assert config.enable_cors is True
        assert config.api_key_required is False

    def test_development_mode_config(self):
        """Test development mode configuration."""
        config = DashboardConfig(
            mode=DashboardMode.DEVELOPMENT,
            debug=True,
            port=8889,
        )

        assert config.mode == DashboardMode.DEVELOPMENT
        assert config.debug is True
        assert config.port == 8889

    def test_readonly_mode_config(self):
        """Test readonly mode configuration."""
        config = DashboardConfig(
            mode=DashboardMode.READONLY,
            enable_retraining_controls=False,
            enable_alert_management=False,
        )

        assert config.mode == DashboardMode.READONLY
        assert config.enable_retraining_controls is False
        assert config.enable_alert_management is False

    def test_security_config(self):
        """Test security-related configuration."""
        config = DashboardConfig(
            api_key_required=True,
            api_key="test-key-12345",
            allowed_origins=["https://example.com"],
            concurrent_requests_limit=20,
        )

        assert config.api_key_required is True
        assert config.api_key == "test-key-12345"
        assert config.allowed_origins == ["https://example.com"]
        assert config.concurrent_requests_limit == 20


# WebSocket Manager Tests


class TestWebSocketManager:
    """Test WebSocket connection management."""

    def test_websocket_manager_initialization(self):
        """Test WebSocket manager initialization."""
        manager = WebSocketManager(max_connections=5)

        assert manager.max_connections == 5
        assert len(manager.active_connections) == 0
        assert len(manager.connection_metadata) == 0

    async def test_websocket_connect_success(self, mock_websocket):
        """Test successful WebSocket connection."""
        manager = WebSocketManager(max_connections=2)

        # Connect first WebSocket
        result = await manager.connect(mock_websocket, {"user": "test"})

        assert result is True
        assert mock_websocket in manager.active_connections
        assert len(manager.active_connections) == 1
        assert mock_websocket in manager.connection_metadata

        metadata = manager.connection_metadata[mock_websocket]
        assert metadata["client_info"]["user"] == "test"
        assert metadata["messages_sent"] == 0
        mock_websocket.accept.assert_called_once()

    async def test_websocket_connect_max_connections(self, mock_websocket):
        """Test WebSocket connection rejection at max capacity."""
        manager = WebSocketManager(max_connections=1)

        # Connect first WebSocket (should succeed)
        first_ws = AsyncMock()
        first_ws.accept = AsyncMock()
        result1 = await manager.connect(first_ws)
        assert result1 is True

        # Try to connect second WebSocket (should fail)
        result2 = await manager.connect(mock_websocket)
        assert result2 is False
        assert mock_websocket not in manager.active_connections
        mock_websocket.accept.assert_not_called()

    async def test_websocket_disconnect(self, mock_websocket):
        """Test WebSocket disconnection."""
        manager = WebSocketManager()

        # Connect then disconnect
        await manager.connect(mock_websocket)
        assert len(manager.active_connections) == 1

        await manager.disconnect(mock_websocket)
        assert len(manager.active_connections) == 0
        assert mock_websocket not in manager.connection_metadata

    async def test_websocket_send_personal_message(self, mock_websocket):
        """Test sending personal message to WebSocket."""
        manager = WebSocketManager()
        await manager.connect(mock_websocket)

        message = {"type": "test", "data": "hello"}
        result = await manager.send_personal_message(message, mock_websocket)

        assert result is True
        mock_websocket.send_text.assert_called_once()

        # Check metadata updated
        metadata = manager.connection_metadata[mock_websocket]
        assert metadata["messages_sent"] == 1
        assert metadata["last_message_at"] is not None

    async def test_websocket_send_to_disconnected(self):
        """Test sending message to disconnected WebSocket."""
        manager = WebSocketManager()
        disconnected_ws = AsyncMock()

        # Try to send to non-connected WebSocket
        result = await manager.send_personal_message({"type": "test"}, disconnected_ws)

        assert result is False
        disconnected_ws.send_text.assert_not_called()

    async def test_websocket_broadcast(self, mock_websocket):
        """Test broadcasting message to all WebSockets."""
        manager = WebSocketManager()

        # Connect multiple WebSockets
        ws1 = AsyncMock()
        ws1.accept = AsyncMock()
        ws1.send_text = AsyncMock()

        ws2 = AsyncMock()
        ws2.accept = AsyncMock()
        ws2.send_text = AsyncMock()

        await manager.connect(ws1)
        await manager.connect(ws2)

        message = {"type": "broadcast", "data": "hello all"}
        sent_count = await manager.broadcast(message)

        assert sent_count == 2
        ws1.send_text.assert_called_once()
        ws2.send_text.assert_called_once()

    async def test_websocket_broadcast_with_failures(self, mock_websocket):
        """Test broadcast with some connection failures."""
        manager = WebSocketManager()

        # Connect working WebSocket
        working_ws = AsyncMock()
        working_ws.accept = AsyncMock()
        working_ws.send_text = AsyncMock()
        await manager.connect(working_ws)

        # Connect failing WebSocket
        failing_ws = AsyncMock()
        failing_ws.accept = AsyncMock()
        failing_ws.send_text = AsyncMock(side_effect=Exception("Connection failed"))
        await manager.connect(failing_ws)

        message = {"type": "test"}
        sent_count = await manager.broadcast(message)

        # Should succeed for working connection only
        assert sent_count == 1
        working_ws.send_text.assert_called_once()

        # Failing connection should be removed
        assert failing_ws not in manager.active_connections

    def test_websocket_connection_stats(self, mock_websocket):
        """Test WebSocket connection statistics."""
        manager = WebSocketManager(max_connections=10)

        stats = manager.get_connection_stats()

        assert stats["active_connections"] == 0
        assert stats["max_connections"] == 10
        assert stats["total_messages_sent"] == 0
        assert stats["connections_available"] == 10


# System Overview Tests


class TestSystemOverview:
    """Test system overview data structure."""

    def test_system_overview_initialization(self):
        """Test system overview initialization."""
        overview = SystemOverview()

        assert overview.system_health_score == 0.0
        assert overview.system_status == "unknown"
        assert overview.total_predictions_24h == 0
        assert overview.accuracy_rate_24h == 0.0
        assert overview.active_rooms == 0
        assert overview.last_updated is not None
        assert overview.uptime_hours == 0.0

    def test_system_overview_with_data(self):
        """Test system overview with actual data."""
        now = datetime.utcnow()
        overview = SystemOverview(
            system_health_score=0.85,
            system_status="healthy",
            total_predictions_24h=150,
            accuracy_rate_24h=0.88,
            active_rooms=3,
            active_alerts=2,
            last_updated=now,
            uptime_hours=24.5,
        )

        assert overview.system_health_score == 0.85
        assert overview.system_status == "healthy"
        assert overview.total_predictions_24h == 150
        assert overview.accuracy_rate_24h == 0.88
        assert overview.active_rooms == 3
        assert overview.active_alerts == 2
        assert overview.last_updated == now
        assert overview.uptime_hours == 24.5

    def test_system_overview_to_dict(self):
        """Test system overview serialization to dictionary."""
        now = datetime.utcnow()
        overview = SystemOverview(
            system_health_score=0.9,
            system_status="healthy",
            total_predictions_24h=200,
            last_updated=now,
        )

        data = overview.to_dict()

        assert isinstance(data, dict)
        assert data["system_health_score"] == 0.9
        assert data["system_status"] == "healthy"
        assert data["total_predictions_24h"] == 200
        assert data["last_updated"] == now.isoformat()

        # Check all expected fields are present
        expected_fields = [
            "system_health_score",
            "system_status",
            "total_predictions_24h",
            "accuracy_rate_24h",
            "mean_error_minutes_24h",
            "active_rooms",
            "active_models",
            "predictions_per_hour",
            "active_alerts",
            "critical_alerts",
            "warnings",
            "rooms_with_drift",
            "active_retraining_tasks",
            "completed_retraining_24h",
            "avg_prediction_latency_ms",
            "avg_validation_lag_minutes",
            "cache_hit_rate",
            "last_updated",
            "uptime_hours",
        ]

        for field in expected_fields:
            assert field in data


# Performance Dashboard Tests


@patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "true"})
class TestPerformanceDashboard:
    """Test performance dashboard functionality."""

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    def test_dashboard_initialization(self, mock_tracking_manager, dashboard_config):
        """Test dashboard initialization."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        assert dashboard.tracking_manager == mock_tracking_manager
        assert dashboard.config == dashboard_config
        assert dashboard._running is False
        assert dashboard.websocket_manager is not None
        assert dashboard.app is not None
        assert isinstance(dashboard._cache, dict)

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", False)
    def test_dashboard_initialization_no_fastapi(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test dashboard initialization without FastAPI."""
        with pytest.raises(OccupancyPredictionError, match="FastAPI not available"):
            PerformanceDashboard(mock_tracking_manager, dashboard_config)

    def test_dashboard_initialization_websocket_disabled(self, mock_tracking_manager):
        """Test dashboard initialization with WebSocket disabled."""
        config = DashboardConfig(websocket_enabled=False)

        with patch("src.integration.dashboard.FASTAPI_AVAILABLE", True):
            dashboard = PerformanceDashboard(mock_tracking_manager, config)

            assert dashboard.websocket_manager is None

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_dashboard_start_stop(self, mock_tracking_manager, dashboard_config):
        """Test dashboard start and stop functionality."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Test start
        await dashboard.start_dashboard()
        assert dashboard._running is True

        # Test stop
        await dashboard.stop_dashboard()
        assert dashboard._running is False

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_dashboard_start_already_running(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test starting dashboard when already running."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)
        dashboard._running = True

        # Should not raise error
        await dashboard.start_dashboard()
        assert dashboard._running is True

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_dashboard_stop_not_running(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test stopping dashboard when not running."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Should not raise error
        await dashboard.stop_dashboard()
        assert dashboard._running is False

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_system_overview(self, mock_tracking_manager, dashboard_config):
        """Test system overview generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        overview = await dashboard._get_system_overview()

        assert isinstance(overview, SystemOverview)
        assert overview.system_health_score == 0.85
        assert overview.system_status == "active"
        assert overview.total_predictions_24h == 150
        assert overview.accuracy_rate_24h == 0.88
        assert overview.active_rooms == 3
        assert overview.active_models == 2
        assert overview.active_alerts == 2
        assert overview.uptime_hours > 0

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_system_overview_with_error(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test system overview generation with tracking manager error."""
        mock_tracking_manager.get_tracking_status.side_effect = Exception(
            "Tracking error"
        )
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        overview = await dashboard._get_system_overview()

        assert isinstance(overview, SystemOverview)
        assert overview.system_status == "error"
        assert overview.last_updated is not None

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_accuracy_dashboard_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test accuracy dashboard data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_accuracy_dashboard_data(
            room_id="living_room", model_type="ensemble", hours_back=24
        )

        assert isinstance(data, dict)
        assert "real_time_metrics" in data
        assert "accuracy_summary" in data
        assert "time_period" in data
        assert data["time_period"]["hours_back"] == 24
        assert len(data["real_time_metrics"]) == 2  # From mock

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_accuracy_dashboard_data_with_error(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test accuracy dashboard data with error."""
        mock_tracking_manager.get_real_time_metrics.side_effect = Exception(
            "Metrics error"
        )
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_accuracy_dashboard_data(None, None, 24)

        assert "error" in data
        assert data["real_time_metrics"] == []
        assert data["accuracy_summary"] == {}

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_drift_dashboard_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test drift dashboard data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_drift_dashboard_data("bedroom")

        assert isinstance(data, dict)
        assert "drift_summary" in data
        assert "drift_by_room" in data
        assert "recent_drift_events" in data
        assert data["drift_summary"]["total_rooms_monitored"] == 3
        assert "bedroom" in data["drift_by_room"]

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_drift_dashboard_data_all_rooms(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test drift dashboard data for all rooms."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_drift_dashboard_data(None)

        assert "drift_by_room" in data
        assert len(data["drift_by_room"]) == 2  # From mock setup
        assert "living_room" in data["drift_by_room"]
        assert "bedroom" in data["drift_by_room"]

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_retraining_dashboard_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test retraining dashboard data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_retraining_dashboard_data()

        assert isinstance(data, dict)
        assert "queue_summary" in data
        assert "active_retraining_tasks" in data
        assert "recent_completions" in data
        assert data["queue_summary"]["pending_requests"] == 2
        assert data["queue_summary"]["active_retraining"] == 1
        assert len(data["active_retraining_tasks"]) == 1

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_system_health_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test system health data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_system_health_data()

        assert isinstance(data, dict)
        assert "overall_health" in data
        assert "component_health" in data
        assert "resource_usage" in data
        assert data["overall_health"]["score"] == 0.85
        assert "prediction_validator" in data["component_health"]
        assert "accuracy_tracker" in data["component_health"]

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_alerts_dashboard_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test alerts dashboard data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_alerts_dashboard_data("warning", "bedroom")

        assert isinstance(data, dict)
        assert "alert_summary" in data
        assert "active_alerts" in data
        assert "alerts_by_room" in data
        assert data["alert_summary"]["total_active"] == 2
        assert data["alert_summary"]["warning_count"] == 2
        assert len(data["active_alerts"]) == 2

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_trends_dashboard_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test trends dashboard data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_trends_dashboard_data("living_room", 7)

        assert isinstance(data, dict)
        assert "time_period" in data
        assert "accuracy_trends" in data
        assert "error_trends" in data
        assert data["time_period"]["days_back"] == 7

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    def test_get_dashboard_stats(self, mock_tracking_manager, dashboard_config):
        """Test dashboard statistics generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        stats = dashboard._get_dashboard_stats()

        assert isinstance(stats, dict)
        assert "dashboard_info" in stats
        assert "configuration" in stats
        assert "performance" in stats
        assert stats["dashboard_info"]["version"] == "1.0.0"
        assert stats["dashboard_info"]["mode"] == DashboardMode.DEVELOPMENT.value
        assert stats["configuration"]["host"] == "127.0.0.1"
        assert stats["configuration"]["port"] == 8889

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_websocket_initial_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test WebSocket initial data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_websocket_initial_data()

        assert isinstance(data, dict)
        assert data["type"] == "initial_data"
        assert "timestamp" in data
        assert "data" in data
        assert "system_overview" in data["data"]
        assert "dashboard_info" in data["data"]

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_websocket_update_data(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test WebSocket update data generation."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        data = await dashboard._get_websocket_update_data()

        assert isinstance(data, dict)
        assert "system_overview" in data
        assert "alert_summary" in data
        assert "last_updated" in data

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_handle_websocket_message_ping(
        self, mock_tracking_manager, dashboard_config, mock_websocket
    ):
        """Test WebSocket ping message handling."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        message = json.dumps({"type": "ping"})
        await dashboard._handle_websocket_message(message, mock_websocket)

        # Should send pong response
        assert dashboard.websocket_manager.send_personal_message.called

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_handle_websocket_message_request_data(
        self, mock_tracking_manager, dashboard_config, mock_websocket
    ):
        """Test WebSocket data request handling."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        message = json.dumps({"type": "request_data", "data_type": "overview"})
        await dashboard._handle_websocket_message(message, mock_websocket)

        # Should send data response
        assert dashboard.websocket_manager.send_personal_message.called

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_handle_websocket_message_invalid_json(
        self, mock_tracking_manager, dashboard_config, mock_websocket
    ):
        """Test WebSocket invalid JSON handling."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Should not raise exception
        await dashboard._handle_websocket_message("invalid json", mock_websocket)

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_get_requested_data(self, mock_tracking_manager, dashboard_config):
        """Test specific data type requests."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Test overview request
        data = await dashboard._get_requested_data("overview")
        assert "system_health_score" in data

        # Test accuracy request
        data = await dashboard._get_requested_data("accuracy")
        assert "real_time_metrics" in data

        # Test drift request
        data = await dashboard._get_requested_data("drift")
        assert "drift_summary" in data

        # Test retraining request
        data = await dashboard._get_requested_data("retraining")
        assert "queue_summary" in data

        # Test alerts request
        data = await dashboard._get_requested_data("alerts")
        assert "alert_summary" in data

        # Test unknown request
        data = await dashboard._get_requested_data("unknown")
        assert "error" in data

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_trigger_manual_retraining(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test manual retraining trigger."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        request_data = {
            "room_id": "living_room",
            "model_type": "ensemble",
            "strategy": "full_retrain",
        }

        result = await dashboard._trigger_manual_retraining(request_data)

        assert result["success"] is True
        assert "request_id" in result
        mock_tracking_manager.request_manual_retraining.assert_called_once()

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_trigger_manual_retraining_no_room_id(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test manual retraining trigger without room_id."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        request_data = {"strategy": "full_retrain"}

        result = await dashboard._trigger_manual_retraining(request_data)

        assert result["success"] is False
        assert "error" in result

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_acknowledge_alert(self, mock_tracking_manager, dashboard_config):
        """Test alert acknowledgment."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        alert_data = {"alert_id": "alert_001", "user_id": "test_user"}

        result = await dashboard._acknowledge_alert(alert_data)

        assert result["success"] is True
        assert result["alert_id"] == "alert_001"
        assert result["acknowledged_by"] == "test_user"
        mock_tracking_manager.acknowledge_alert.assert_called_once()

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_acknowledge_alert_no_alert_id(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test alert acknowledgment without alert_id."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        alert_data = {"user_id": "test_user"}

        result = await dashboard._acknowledge_alert(alert_data)

        assert result["success"] is False
        assert "error" in result

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    def test_cache_functionality(self, mock_tracking_manager, dashboard_config):
        """Test dashboard caching functionality."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Cache some data
        test_data = {"test": "value"}
        dashboard._cache_data("test_key", test_data)

        # Retrieve cached data
        cached = dashboard._get_cached_data("test_key")
        assert cached == test_data

        # Test cache expiration
        import time

        time.sleep(dashboard.config.cache_ttl_seconds + 1)
        expired = dashboard._get_cached_data("test_key")
        assert expired is None

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    def test_cache_size_limit(self, mock_tracking_manager, dashboard_config):
        """Test dashboard cache size limiting."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Fill cache beyond limit
        for i in range(110):  # Exceeds limit of 100
            dashboard._cache_data(f"key_{i}", f"value_{i}")

        # Should have removed oldest entries
        assert len(dashboard._cache) <= 100


# Dashboard Route Registration Tests


@patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "true"})
@patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
class TestDashboardRoutes:
    """Test dashboard FastAPI route registration."""

    def test_route_registration_development_mode(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test route registration in development mode."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Check that FastAPI app has expected routes
        route_paths = [route.path for route in dashboard.app.routes]

        expected_paths = [
            "/api/dashboard/overview",
            "/api/dashboard/accuracy",
            "/api/dashboard/drift",
            "/api/dashboard/retraining",
            "/api/dashboard/health",
            "/api/dashboard/alerts",
            "/api/dashboard/trends",
            "/api/dashboard/stats",
            "/ws/dashboard",
        ]

        for path in expected_paths:
            assert path in route_paths

    def test_route_registration_production_mode(
        self, mock_tracking_manager, dashboard_config_production
    ):
        """Test route registration in production mode with controls."""
        dashboard = PerformanceDashboard(
            mock_tracking_manager, dashboard_config_production
        )

        route_paths = [route.path for route in dashboard.app.routes]

        # Should include control endpoints in production
        assert "/api/dashboard/actions/retrain" in route_paths
        assert "/api/dashboard/actions/acknowledge_alert" in route_paths

    def test_route_registration_readonly_mode(
        self, mock_tracking_manager, dashboard_config_readonly
    ):
        """Test route registration in readonly mode."""
        dashboard = PerformanceDashboard(
            mock_tracking_manager, dashboard_config_readonly
        )

        route_paths = [route.path for route in dashboard.app.routes]

        # Should NOT include control endpoints in readonly
        assert "/api/dashboard/actions/retrain" not in route_paths
        assert "/api/dashboard/actions/acknowledge_alert" not in route_paths

    def test_websocket_route_registration(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test WebSocket route registration."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        route_paths = [route.path for route in dashboard.app.routes]
        assert "/ws/dashboard" in route_paths

    def test_websocket_disabled_no_route(self, mock_tracking_manager):
        """Test no WebSocket route when disabled."""
        config = DashboardConfig(websocket_enabled=False)
        dashboard = PerformanceDashboard(mock_tracking_manager, config)

        route_paths = [route.path for route in dashboard.app.routes]
        assert "/ws/dashboard" not in route_paths


# Dashboard Error Handling Tests


class TestDashboardErrors:
    """Test dashboard error handling scenarios."""

    def test_dashboard_error_creation(self):
        """Test dashboard error creation."""
        error = DashboardError(
            "Test error message",
            error_context={"component": "test"},
            severity=ErrorSeverity.HIGH,
        )

        assert "Test error message" in str(error)
        assert error.error_code == "DASHBOARD_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["component"] == "test"

    def test_dashboard_error_default_severity(self):
        """Test dashboard error with default severity."""
        error = DashboardError("Test error")

        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_code == "DASHBOARD_ERROR"


# Integration Helper Tests


class TestDashboardIntegration:
    """Test dashboard integration helper functions."""

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    async def test_create_dashboard_from_tracking_manager(self, mock_tracking_manager):
        """Test dashboard creation helper function."""
        with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "true"}):
            dashboard = await create_dashboard_from_tracking_manager(
                mock_tracking_manager, host="127.0.0.1", port=8890, debug=True
            )

            assert isinstance(dashboard, PerformanceDashboard)
            assert dashboard.tracking_manager == mock_tracking_manager
            assert dashboard.config.host == "127.0.0.1"
            assert dashboard.config.port == 8890
            assert dashboard.config.debug is True

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    def test_integrate_dashboard_with_tracking_system_default_config(
        self, mock_tracking_manager
    ):
        """Test dashboard integration with default config."""
        dashboard = integrate_dashboard_with_tracking_system(mock_tracking_manager)

        assert isinstance(dashboard, PerformanceDashboard)
        assert dashboard.tracking_manager == mock_tracking_manager
        assert isinstance(dashboard.config, DashboardConfig)

    @patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
    def test_integrate_dashboard_with_tracking_system_custom_config(
        self, mock_tracking_manager
    ):
        """Test dashboard integration with custom config."""
        config = DashboardConfig(port=9999, debug=True)
        dashboard = integrate_dashboard_with_tracking_system(
            mock_tracking_manager, config
        )

        assert dashboard.config == config
        assert dashboard.config.port == 9999
        assert dashboard.config.debug is True


# Edge Cases and Error Scenarios


@patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "true"})
@patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
class TestDashboardEdgeCases:
    """Test dashboard edge cases and error scenarios."""

    async def test_tracking_manager_none_response(self, dashboard_config):
        """Test handling when tracking manager returns None."""
        mock_manager = AsyncMock()
        mock_manager.get_tracking_status.return_value = None

        dashboard = PerformanceDashboard(mock_manager, dashboard_config)
        overview = await dashboard._get_system_overview()

        # Should handle None gracefully
        assert overview.system_status == "unknown"
        assert overview.system_health_score == 0.0

    async def test_tracking_manager_partial_data(self, dashboard_config):
        """Test handling partial data from tracking manager."""
        mock_manager = AsyncMock()
        mock_manager.get_tracking_status.return_value = {
            "status": "active",  # Only minimal data
        }

        dashboard = PerformanceDashboard(mock_manager, dashboard_config)
        overview = await dashboard._get_system_overview()

        assert overview.system_status == "active"
        assert overview.total_predictions_24h == 0  # Should default to 0

    async def test_websocket_message_handling_errors(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test WebSocket message handling with various errors."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)
        mock_ws = AsyncMock()

        # Test exception in message handling
        with patch.object(
            dashboard, "_get_requested_data", side_effect=Exception("Test error")
        ):
            # Should not raise exception
            await dashboard._handle_websocket_message(
                json.dumps({"type": "request_data", "data_type": "overview"}), mock_ws
            )

    async def test_cache_with_concurrent_access(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test cache with concurrent access."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Simulate concurrent cache access
        async def cache_operation(key, value):
            dashboard._cache_data(key, value)
            return dashboard._get_cached_data(key)

        # Run multiple cache operations concurrently
        tasks = [cache_operation(f"key_{i}", f"value_{i}") for i in range(10)]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert len(results) == 10
        assert all(r is not None for r in results)

    async def test_websocket_manager_error_handling(self):
        """Test WebSocket manager error handling."""
        manager = WebSocketManager()

        # Test connection with exception
        failing_ws = AsyncMock()
        failing_ws.accept.side_effect = Exception("Connection failed")

        result = await manager.connect(failing_ws)
        assert result is False
        assert failing_ws not in manager.active_connections

    async def test_dashboard_data_methods_with_exceptions(self, dashboard_config):
        """Test dashboard data methods with tracking manager exceptions."""
        mock_manager = AsyncMock()

        # Configure various methods to raise exceptions
        mock_manager.get_tracking_status.side_effect = Exception("Tracking error")
        mock_manager.get_real_time_metrics.side_effect = Exception("Metrics error")
        mock_manager.get_drift_status.side_effect = Exception("Drift error")
        mock_manager.get_retraining_status.side_effect = Exception("Retraining error")
        mock_manager.get_active_alerts.side_effect = Exception("Alerts error")

        dashboard = PerformanceDashboard(mock_manager, dashboard_config)

        # All methods should handle exceptions gracefully
        overview = await dashboard._get_system_overview()
        assert overview.system_status == "error"

        accuracy_data = await dashboard._get_accuracy_dashboard_data(None, None, 24)
        assert "error" in accuracy_data

        drift_data = await dashboard._get_drift_dashboard_data(None)
        assert "error" in drift_data

        retraining_data = await dashboard._get_retraining_dashboard_data()
        assert "error" in retraining_data

        health_data = await dashboard._get_system_health_data()
        assert "error" in health_data

        alerts_data = await dashboard._get_alerts_dashboard_data(None, None)
        assert "error" in alerts_data

    def test_enum_values(self):
        """Test enum value definitions."""
        # Test DashboardMode values
        assert DashboardMode.DEVELOPMENT.value == "development"
        assert DashboardMode.PRODUCTION.value == "production"
        assert DashboardMode.READONLY.value == "readonly"

        # Test MetricType values
        assert MetricType.ACCURACY.value == "accuracy"
        assert MetricType.PERFORMANCE.value == "performance"
        assert MetricType.DRIFT.value == "drift"


# Performance and Load Tests


@patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "true"})
@patch("src.integration.dashboard.FASTAPI_AVAILABLE", True)
class TestDashboardPerformance:
    """Test dashboard performance characteristics."""

    async def test_multiple_websocket_connections(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test handling multiple WebSocket connections."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Connect multiple WebSockets
        websockets = []
        for i in range(5):
            ws = AsyncMock()
            ws.accept = AsyncMock()
            ws.send_text = AsyncMock()
            await dashboard.websocket_manager.connect(ws)
            websockets.append(ws)

        assert len(dashboard.websocket_manager.active_connections) == 5

        # Test broadcasting to all
        message = {"type": "test", "data": "broadcast"}
        sent_count = await dashboard.websocket_manager.broadcast(message)

        assert sent_count == 5
        for ws in websockets:
            ws.send_text.assert_called_once()

    async def test_rapid_cache_operations(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test rapid cache operations."""
        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Perform many cache operations rapidly
        for i in range(200):
            dashboard._cache_data(f"rapid_key_{i}", f"value_{i}")

        # Should maintain size limit
        assert len(dashboard._cache) <= 100

        # Recent entries should still be accessible
        recent_value = dashboard._get_cached_data("rapid_key_199")
        assert recent_value == "value_199"

    async def test_large_data_serialization(
        self, mock_tracking_manager, dashboard_config
    ):
        """Test handling of large data structures."""
        # Create large mock data
        large_data = {
            "metrics": [{"room": f"room_{i}", "value": i} for i in range(1000)],
            "history": [
                {"timestamp": datetime.now().isoformat(), "data": f"data_{i}"}
                for i in range(500)
            ],
        }

        mock_tracking_manager.get_tracking_status.return_value = large_data

        dashboard = PerformanceDashboard(mock_tracking_manager, dashboard_config)

        # Should handle large data without issues
        overview = await dashboard._get_system_overview()
        assert isinstance(overview, SystemOverview)
