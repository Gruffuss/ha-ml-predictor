"""
Unit tests for real-time accuracy tracking system.

Pure unit tests for AccuracyTracker, RealTimeMetrics, AccuracyAlert, and trend analysis.
Integration tests are in tests/adaptation/test_tracker.py.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adaptation.tracker import (
    AccuracyAlert,
    AccuracyTracker,
    AccuracyTrackingError,
    AlertSeverity,
    RealTimeMetrics,
    TrendDirection,
)
from src.adaptation.validator import (
    AccuracyLevel,
    AccuracyMetrics,
    PredictionValidator,
    ValidationRecord,
)
from src.core.constants import ModelType
from src.core.exceptions import ErrorSeverity


class TestRealTimeMetrics:
    """Unit tests for RealTimeMetrics class."""

    def test_real_time_metrics_initialization(self):
        """Test basic initialization with defaults."""
        metrics = RealTimeMetrics(room_id="test_room")

        assert metrics.room_id == "test_room"
        assert metrics.model_type is None
        assert metrics.window_1h_accuracy == 0.0
        assert metrics.window_6h_accuracy == 0.0
        assert metrics.window_24h_accuracy == 0.0
        assert metrics.accuracy_trend == TrendDirection.UNKNOWN
        assert metrics.trend_slope == 0.0
        assert metrics.active_alerts == []
        assert isinstance(metrics.last_updated, datetime)

    def test_overall_health_score_no_predictions(self):
        """Test health score calculation with no predictions."""
        metrics = RealTimeMetrics(room_id="test_room", window_24h_predictions=0)

        assert metrics.overall_health_score == 0.0
        assert not metrics.is_healthy

    def test_overall_health_score_calculation(self):
        """Test health score calculation with various metrics."""
        # High performance metrics
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=92.0,
            window_24h_accuracy=88.0,
            window_6h_predictions=15,
            window_24h_predictions=120,
            accuracy_trend=TrendDirection.IMPROVING,
        )

        # Should be healthy with high scores
        assert metrics.overall_health_score > 80.0
        assert metrics.is_healthy

    def test_health_score_degrading_trend(self):
        """Test health score impact of degrading trend."""
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=75.0,
            window_24h_accuracy=80.0,
            window_6h_predictions=10,
            window_24h_predictions=80,
            accuracy_trend=TrendDirection.DEGRADING,
            trend_slope=-2.5,
        )

        # Degrading trend should reduce health score
        assert metrics.overall_health_score < 75.0

    def test_is_healthy_criteria(self):
        """Test health criteria boundary conditions."""
        # Borderline healthy
        metrics_healthy = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=70.0,  # Exactly at threshold
            window_24h_accuracy=70.0,
            window_6h_predictions=5,
            window_24h_predictions=40,
        )

        # Borderline unhealthy
        metrics_unhealthy = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=69.0,  # Below threshold
            window_24h_accuracy=69.0,
            window_6h_predictions=5,
            window_24h_predictions=40,
        )

        assert metrics_healthy.is_healthy
        assert not metrics_unhealthy.is_healthy

    def test_to_dict_serialization(self):
        """Test dictionary conversion for serialization."""
        now = datetime.now(timezone.utc)
        metrics = RealTimeMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            window_1h_accuracy=85.0,
            window_6h_accuracy=82.0,
            window_24h_accuracy=79.0,
            window_1h_predictions=4,
            window_6h_predictions=22,
            window_24h_predictions=95,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_slope=1.2,
            last_updated=now,
        )

        result = metrics.to_dict()

        assert result["room_id"] == "test_room"
        assert result["model_type"] == "LSTM"
        assert result["window_1h_accuracy"] == 85.0
        assert result["window_6h_accuracy"] == 82.0
        assert result["window_24h_accuracy"] == 79.0
        assert result["window_1h_predictions"] == 4
        assert result["window_6h_predictions"] == 22
        assert result["window_24h_predictions"] == 95
        assert result["accuracy_trend"] == "IMPROVING"
        assert result["trend_slope"] == 1.2
        assert result["overall_health_score"] == metrics.overall_health_score
        assert result["is_healthy"] == metrics.is_healthy
        assert result["last_updated"] == now.isoformat()


class TestAccuracyAlert:
    """Unit tests for AccuracyAlert class."""

    def test_accuracy_alert_initialization(self):
        """Test alert initialization with all parameters."""
        timestamp = datetime.now(timezone.utc)
        alert = AccuracyAlert(
            alert_id="test_alert_1",
            room_id="living_room",
            model_type=ModelType.ENSEMBLE,
            severity=AlertSeverity.WARNING,
            message="Accuracy degraded to 65%",
            metric_name="6h_accuracy",
            metric_value=65.0,
            threshold=70.0,
            timestamp=timestamp,
            context={"trend": "degrading", "predictions": 15},
        )

        assert alert.alert_id == "test_alert_1"
        assert alert.room_id == "living_room"
        assert alert.model_type == ModelType.ENSEMBLE
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Accuracy degraded to 65%"
        assert alert.metric_name == "6h_accuracy"
        assert alert.metric_value == 65.0
        assert alert.threshold == 70.0
        assert alert.timestamp == timestamp
        assert alert.context == {"trend": "degrading", "predictions": 15}
        assert not alert.acknowledged
        assert not alert.resolved

    def test_age_calculation(self):
        """Test alert age calculation."""
        past_time = datetime.now(timezone.utc) - timedelta(minutes=15)
        alert = AccuracyAlert(
            alert_id="age_test",
            room_id="test_room",
            severity=AlertSeverity.CRITICAL,
            message="Test alert",
            timestamp=past_time,
        )

        age = alert.age
        assert isinstance(age, timedelta)
        assert age >= timedelta(minutes=14)  # Allow for execution time
        assert age <= timedelta(minutes=16)

    def test_escalation_requirements(self):
        """Test alert escalation logic."""
        now = datetime.now(timezone.utc)

        # Recent critical alert - should escalate
        critical_alert = AccuracyAlert(
            alert_id="critical_test",
            room_id="test_room",
            severity=AlertSeverity.CRITICAL,
            message="Critical accuracy drop",
            timestamp=now - timedelta(minutes=35),  # Older than 30 minutes
        )

        # Recent warning alert - should not escalate yet
        warning_alert = AccuracyAlert(
            alert_id="warning_test",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Warning accuracy drop",
            timestamp=now - timedelta(minutes=45),  # Less than 60 minutes
        )

        # Old warning alert - should escalate
        old_warning = AccuracyAlert(
            alert_id="old_warning",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Old warning",
            timestamp=now - timedelta(minutes=75),  # Older than 60 minutes
        )

        assert critical_alert.should_escalate
        assert not warning_alert.should_escalate
        assert old_warning.should_escalate

    def test_acknowledge_alert(self):
        """Test alert acknowledgment functionality."""
        alert = AccuracyAlert(
            alert_id="ack_test",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Test acknowledgment",
        )

        assert not alert.acknowledged
        assert alert.acknowledged_by is None
        assert alert.acknowledged_at is None

        # Acknowledge the alert
        alert.acknowledge("operator_123")

        assert alert.acknowledged
        assert alert.acknowledged_by == "operator_123"
        assert isinstance(alert.acknowledged_at, datetime)

    def test_resolve_alert(self):
        """Test alert resolution functionality."""
        alert = AccuracyAlert(
            alert_id="resolve_test",
            room_id="test_room",
            severity=AlertSeverity.CRITICAL,
            message="Test resolution",
        )

        assert not alert.resolved
        assert alert.resolved_by is None
        assert alert.resolved_at is None

        # Resolve the alert
        alert.resolve("auto_system", "Accuracy improved to 85%")

        assert alert.resolved
        assert alert.resolved_by == "auto_system"
        assert alert.resolution_reason == "Accuracy improved to 85%"
        assert isinstance(alert.resolved_at, datetime)

    def test_escalate_alert(self):
        """Test alert escalation functionality."""
        alert = AccuracyAlert(
            alert_id="escalate_test",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Test escalation",
        )

        original_severity = alert.severity
        assert not alert.escalated

        # Escalate the alert
        alert.escalate("Prolonged accuracy degradation")

        assert alert.escalated
        assert alert.severity == AlertSeverity.CRITICAL  # Escalated
        assert alert.escalation_reason == "Prolonged accuracy degradation"
        assert isinstance(alert.escalated_at, datetime)

    def test_alert_to_dict(self):
        """Test alert dictionary serialization."""
        timestamp = datetime.now(timezone.utc)
        alert = AccuracyAlert(
            alert_id="dict_test",
            room_id="test_room",
            model_type=ModelType.XGBOOST,
            severity=AlertSeverity.CRITICAL,
            message="Serialization test",
            metric_name="24h_accuracy",
            metric_value=45.0,
            threshold=60.0,
            timestamp=timestamp,
            context={"test": "data"},
        )

        # Acknowledge and resolve for complete test
        alert.acknowledge("test_user")
        alert.resolve("test_system", "Test resolution")

        result = alert.to_dict()

        assert result["alert_id"] == "dict_test"
        assert result["room_id"] == "test_room"
        assert result["model_type"] == "XGBOOST"
        assert result["severity"] == "CRITICAL"
        assert result["message"] == "Serialization test"
        assert result["metric_name"] == "24h_accuracy"
        assert result["metric_value"] == 45.0
        assert result["threshold"] == 60.0
        assert result["timestamp"] == timestamp.isoformat()
        assert result["context"] == {"test": "data"}
        assert result["acknowledged"] is True
        assert result["acknowledged_by"] == "test_user"
        assert result["resolved"] is True
        assert result["resolved_by"] == "test_system"
        assert result["resolution_reason"] == "Test resolution"


class TestAccuracyTracker:
    """Unit tests for AccuracyTracker core functionality."""

    @pytest.fixture
    def mock_validator(self):
        """Create mock prediction validator."""
        validator = MagicMock(spec=PredictionValidator)
        validator._lock = MagicMock()
        validator._validation_records = {}
        validator._pending_predictions = {}
        return validator

    @pytest.fixture
    def accuracy_tracker(self, mock_validator):
        """Create AccuracyTracker instance for testing."""
        return AccuracyTracker(
            prediction_validator=mock_validator,
            monitoring_interval_seconds=60,
            alert_thresholds={
                "accuracy_warning": 70.0,
                "accuracy_critical": 50.0,
                "error_warning": 20.0,
                "error_critical": 30.0,
            },
            max_stored_alerts=50,
            trend_analysis_points=5,
        )

    def test_tracker_initialization(self, accuracy_tracker, mock_validator):
        """Test AccuracyTracker initialization."""
        assert accuracy_tracker._prediction_validator == mock_validator
        assert accuracy_tracker._monitoring_interval == 60
        assert accuracy_tracker._alert_thresholds["accuracy_warning"] == 70.0
        assert accuracy_tracker._max_stored_alerts == 50
        assert accuracy_tracker._trend_analysis_points == 5
        assert not accuracy_tracker._monitoring_active
        assert accuracy_tracker._metrics_by_room == {}
        assert accuracy_tracker._active_alerts == []

    async def test_start_stop_monitoring(self, accuracy_tracker):
        """Test monitoring start/stop functionality."""
        # Start monitoring
        await accuracy_tracker.start_monitoring()
        assert accuracy_tracker._monitoring_active

        # Start again (should be idempotent)
        await accuracy_tracker.start_monitoring()
        assert accuracy_tracker._monitoring_active

        # Stop monitoring
        await accuracy_tracker.stop_monitoring()
        assert not accuracy_tracker._monitoring_active

    async def test_start_monitoring_already_active(self, accuracy_tracker):
        """Test starting monitoring when already active."""
        await accuracy_tracker.start_monitoring()
        assert accuracy_tracker._monitoring_active

        # Starting again should not raise error
        await accuracy_tracker.start_monitoring()
        assert accuracy_tracker._monitoring_active

        await accuracy_tracker.stop_monitoring()

    async def test_get_real_time_metrics_room_specific(self, accuracy_tracker):
        """Test getting metrics for specific room."""
        # Mock validator to return specific metrics
        mock_metrics = AccuracyMetrics(
            total_predictions=30,
            validated_predictions=28,
            accuracy_rate=85.0,
            mean_error_minutes=8.5,
            median_error_minutes=7.0,
            std_error_minutes=4.2,
            predictions_per_hour=2.5,
            confidence_calibration_score=0.88,
        )
        accuracy_tracker._prediction_validator.get_accuracy_metrics = AsyncMock(
            return_value=mock_metrics
        )

        # Get metrics for specific room
        result = await accuracy_tracker.get_real_time_metrics(room_id="living_room")

        assert result.room_id == "living_room"
        assert result.window_24h_accuracy == 85.0
        assert result.window_24h_predictions == 28

    async def test_get_real_time_metrics_model_specific(self, accuracy_tracker):
        """Test getting metrics for specific model type."""
        mock_metrics = AccuracyMetrics(
            total_predictions=20,
            validated_predictions=18,
            accuracy_rate=92.0,
            mean_error_minutes=6.2,
            median_error_minutes=5.5,
            std_error_minutes=3.8,
            predictions_per_hour=1.8,
            confidence_calibration_score=0.91,
        )
        accuracy_tracker._prediction_validator.get_accuracy_metrics = AsyncMock(
            return_value=mock_metrics
        )

        # Get metrics for specific model
        result = await accuracy_tracker.get_real_time_metrics(
            room_id="bedroom", model_type=ModelType.LSTM
        )

        assert result.room_id == "bedroom"
        assert result.model_type == ModelType.LSTM
        assert result.window_24h_accuracy == 92.0

    async def test_get_real_time_metrics_global(self, accuracy_tracker):
        """Test getting global metrics across all rooms."""
        mock_metrics = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=95,
            accuracy_rate=78.0,
            mean_error_minutes=12.8,
            median_error_minutes=10.2,
            std_error_minutes=6.5,
            predictions_per_hour=4.2,
            confidence_calibration_score=0.82,
        )
        accuracy_tracker._prediction_validator.get_accuracy_metrics = AsyncMock(
            return_value=mock_metrics
        )

        # Get global metrics
        result = await accuracy_tracker.get_real_time_metrics()

        assert result.room_id == "global"
        assert result.window_24h_accuracy == 78.0
        assert result.window_24h_predictions == 95

    async def test_get_active_alerts_no_filters(self, accuracy_tracker):
        """Test getting all active alerts without filters."""
        # Create test alerts
        alert1 = AccuracyAlert(
            alert_id="alert1",
            room_id="living_room",
            severity=AlertSeverity.WARNING,
            message="Test alert 1",
        )
        alert2 = AccuracyAlert(
            alert_id="alert2",
            room_id="bedroom",
            severity=AlertSeverity.CRITICAL,
            message="Test alert 2",
        )

        # Add alerts to tracker
        accuracy_tracker._active_alerts = [alert1, alert2]

        # Get all alerts
        result = await accuracy_tracker.get_active_alerts()

        assert len(result) == 2
        assert alert1 in result
        assert alert2 in result

    async def test_get_active_alerts_room_filter(self, accuracy_tracker):
        """Test getting alerts filtered by room."""
        # Create test alerts for different rooms
        alert1 = AccuracyAlert(
            alert_id="alert1",
            room_id="living_room",
            severity=AlertSeverity.WARNING,
            message="Living room alert",
        )
        alert2 = AccuracyAlert(
            alert_id="alert2",
            room_id="bedroom",
            severity=AlertSeverity.CRITICAL,
            message="Bedroom alert",
        )
        alert3 = AccuracyAlert(
            alert_id="alert3",
            room_id="living_room",
            severity=AlertSeverity.INFO,
            message="Another living room alert",
        )

        accuracy_tracker._active_alerts = [alert1, alert2, alert3]

        # Get alerts for living room only
        result = await accuracy_tracker.get_active_alerts(room_id="living_room")

        assert len(result) == 2
        assert alert1 in result
        assert alert3 in result
        assert alert2 not in result

    async def test_get_active_alerts_severity_filter(self, accuracy_tracker):
        """Test getting alerts filtered by severity."""
        # Create test alerts with different severities
        alert1 = AccuracyAlert(
            alert_id="alert1",
            room_id="living_room",
            severity=AlertSeverity.WARNING,
            message="Warning alert",
        )
        alert2 = AccuracyAlert(
            alert_id="alert2",
            room_id="bedroom",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
        )
        alert3 = AccuracyAlert(
            alert_id="alert3",
            room_id="kitchen",
            severity=AlertSeverity.CRITICAL,
            message="Another critical alert",
        )

        accuracy_tracker._active_alerts = [alert1, alert2, alert3]

        # Get critical alerts only
        result = await accuracy_tracker.get_active_alerts(
            min_severity=AlertSeverity.CRITICAL
        )

        assert len(result) == 2
        assert alert2 in result
        assert alert3 in result
        assert alert1 not in result

    async def test_acknowledge_alert(self, accuracy_tracker):
        """Test acknowledging an alert."""
        # Create test alert
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Test acknowledgment",
        )
        accuracy_tracker._active_alerts = [alert]

        # Acknowledge the alert
        result = await accuracy_tracker.acknowledge_alert(
            "test_alert", acknowledged_by="test_user"
        )

        assert result is True
        assert alert.acknowledged
        assert alert.acknowledged_by == "test_user"

    async def test_acknowledge_nonexistent_alert(self, accuracy_tracker):
        """Test acknowledging non-existent alert."""
        result = await accuracy_tracker.acknowledge_alert(
            "nonexistent_alert", acknowledged_by="test_user"
        )
        assert result is False

    async def test_get_accuracy_trends_room_specific(self, accuracy_tracker):
        """Test getting accuracy trends for specific room."""
        # Create historical metrics
        base_time = datetime.now(timezone.utc)
        historical_metrics = []

        for i in range(10):
            timestamp = base_time - timedelta(hours=i)
            metrics = RealTimeMetrics(
                room_id="test_room",
                window_6h_accuracy=80.0 + i,  # Improving trend
                last_updated=timestamp,
            )
            historical_metrics.append((timestamp, metrics))

        # Store historical data
        accuracy_tracker._metrics_by_room["test_room"] = historical_metrics[-1][1]
        accuracy_tracker._historical_metrics = historical_metrics

        # Get trends
        result = await accuracy_tracker.get_accuracy_trends(
            room_id="test_room", hours=24
        )

        assert "test_room" in result
        trend_data = result["test_room"]
        assert len(trend_data["datapoints"]) <= 10
        assert trend_data["trend_direction"] in [
            TrendDirection.IMPROVING.value,
            TrendDirection.STABLE.value,
            TrendDirection.DEGRADING.value,
        ]

    async def test_get_accuracy_trends_all_rooms(self, accuracy_tracker):
        """Test getting accuracy trends for all rooms."""
        # Create metrics for multiple rooms
        rooms = ["living_room", "bedroom", "kitchen"]
        base_time = datetime.now(timezone.utc)

        for room in rooms:
            metrics = RealTimeMetrics(
                room_id=room,
                window_6h_accuracy=75.0,
                last_updated=base_time,
            )
            accuracy_tracker._metrics_by_room[room] = metrics

        # Get trends for all rooms
        result = await accuracy_tracker.get_accuracy_trends(hours=12)

        assert len(result) == 3
        for room in rooms:
            assert room in result

    async def test_export_tracking_data(self, accuracy_tracker, tmp_path):
        """Test exporting tracking data to file."""
        # Create test data
        alert = AccuracyAlert(
            alert_id="export_test",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Export test alert",
        )
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=82.0,
        )

        accuracy_tracker._active_alerts = [alert]
        accuracy_tracker._metrics_by_room["test_room"] = metrics

        # Export data
        export_file = tmp_path / "export_test.json"
        result = await accuracy_tracker.export_tracking_data(str(export_file))

        assert result is True
        assert export_file.exists()

        # Verify exported content
        import json

        with open(export_file, "r") as f:
            data = json.load(f)

        assert "alerts" in data
        assert "metrics" in data
        assert "export_metadata" in data
        assert len(data["alerts"]) == 1
        assert len(data["metrics"]) == 1


class TestTrendAnalysis:
    """Unit tests for trend analysis functionality."""

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis with insufficient data points."""
        tracker = AccuracyTracker(prediction_validator=MagicMock())

        # Not enough data points
        datapoints = [75.0, 78.0]  # Only 2 points, need at least 3

        direction, slope = tracker._analyze_trend(datapoints)

        assert direction == TrendDirection.UNKNOWN
        assert slope == 0.0

    def test_analyze_trend_improving(self):
        """Test trend analysis with improving accuracy."""
        tracker = AccuracyTracker(prediction_validator=MagicMock())

        # Clearly improving trend
        datapoints = [70.0, 75.0, 80.0, 85.0, 88.0]

        direction, slope = tracker._analyze_trend(datapoints)

        assert direction == TrendDirection.IMPROVING
        assert slope > 0

    def test_analyze_trend_degrading(self):
        """Test trend analysis with degrading accuracy."""
        tracker = AccuracyTracker(prediction_validator=MagicMock())

        # Clearly degrading trend
        datapoints = [90.0, 85.0, 80.0, 75.0, 70.0]

        direction, slope = tracker._analyze_trend(datapoints)

        assert direction == TrendDirection.DEGRADING
        assert slope < 0

    def test_analyze_trend_stable(self):
        """Test trend analysis with stable accuracy."""
        tracker = AccuracyTracker(prediction_validator=MagicMock())

        # Stable trend with small variations
        datapoints = [80.0, 81.0, 79.0, 80.5, 79.5]

        direction, slope = tracker._analyze_trend(datapoints)

        assert direction == TrendDirection.STABLE
        assert abs(slope) < 1.0  # Very small slope

    def test_calculate_global_trend(self):
        """Test global trend calculation across rooms."""
        tracker = AccuracyTracker(prediction_validator=MagicMock())

        room_trends = {
            "living_room": {"trend_direction": TrendDirection.IMPROVING, "slope": 2.5},
            "bedroom": {"trend_direction": TrendDirection.STABLE, "slope": 0.2},
            "kitchen": {"trend_direction": TrendDirection.DEGRADING, "slope": -1.8},
        }

        global_trend = tracker._calculate_global_trend(room_trends)

        assert "overall_direction" in global_trend
        assert "average_slope" in global_trend
        assert "room_count" in global_trend
        assert global_trend["room_count"] == 3

    def test_calculate_global_trend_empty(self):
        """Test global trend calculation with no room data."""
        tracker = AccuracyTracker(prediction_validator=MagicMock())

        global_trend = tracker._calculate_global_trend({})

        assert global_trend["overall_direction"] == TrendDirection.UNKNOWN
        assert global_trend["average_slope"] == 0.0
        assert global_trend["room_count"] == 0


class TestErrorHandling:
    """Unit tests for error handling scenarios."""

    def test_accuracy_tracking_error(self):
        """Test AccuracyTrackingError creation and properties."""
        error = AccuracyTrackingError("Test error message")

        assert "Test error message" in str(error)
        assert error.severity == ErrorSeverity.MEDIUM

    def test_accuracy_tracking_error_with_severity(self):
        """Test AccuracyTrackingError with custom severity."""
        error = AccuracyTrackingError("Critical error", severity=ErrorSeverity.CRITICAL)

        assert "Critical error" in str(error)
        assert error.severity == ErrorSeverity.CRITICAL

    async def test_get_real_time_metrics_error_handling(self, mock_validator):
        """Test error handling in get_real_time_metrics."""
        mock_validator.get_accuracy_metrics = AsyncMock(
            side_effect=Exception("Database error")
        )

        tracker = AccuracyTracker(prediction_validator=mock_validator)

        with pytest.raises(AccuracyTrackingError):
            await tracker.get_real_time_metrics(room_id="test_room")

    async def test_get_active_alerts_error_handling(self, mock_validator):
        """Test error handling in get_active_alerts."""
        tracker = AccuracyTracker(prediction_validator=mock_validator)

        # Simulate internal error by corrupting alerts list
        tracker._active_alerts = None

        with pytest.raises(AccuracyTrackingError):
            await tracker.get_active_alerts()
