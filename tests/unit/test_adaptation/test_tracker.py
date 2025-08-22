"""
Unit tests for real-time accuracy tracking system.

Tests AccuracyTracker, RealTimeMetrics, AccuracyAlert, and trend analysis functionality.
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
    """Test RealTimeMetrics dataclass and properties."""

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
        """Test comprehensive health score calculation."""
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=85.0,
            window_6h_accuracy=80.0,
            window_24h_predictions=50,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_confidence=0.8,
            confidence_calibration=0.9,
            validation_lag_minutes=3.0,
        )

        health_score = metrics.overall_health_score
        assert health_score > 80.0  # Should be high with good metrics
        assert metrics.is_healthy

    def test_health_score_degrading_trend(self):
        """Test health score with degrading trend."""
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=70.0,
            window_6h_accuracy=65.0,
            window_24h_predictions=30,
            accuracy_trend=TrendDirection.DEGRADING,
            trend_confidence=0.7,
            confidence_calibration=0.6,
            validation_lag_minutes=20.0,
        )

        health_score = metrics.overall_health_score
        assert health_score < 60.0  # Should be lower with degrading trend
        assert not metrics.is_healthy

    def test_is_healthy_criteria(self):
        """Test healthy status determination."""
        # Healthy metrics
        healthy_metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=75.0,
            window_24h_predictions=20,
            accuracy_trend=TrendDirection.STABLE,
        )
        # Set overall health score high enough
        healthy_metrics.window_24h_accuracy = 85.0

        assert healthy_metrics.is_healthy

        # Unhealthy due to low accuracy
        unhealthy_accuracy = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=50.0,
            window_24h_predictions=20,
            accuracy_trend=TrendDirection.STABLE,
        )
        unhealthy_accuracy.window_24h_accuracy = 85.0

        assert not unhealthy_accuracy.is_healthy

        # Unhealthy due to degrading trend
        unhealthy_trend = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=75.0,
            window_24h_predictions=20,
            accuracy_trend=TrendDirection.DEGRADING,
        )
        unhealthy_trend.window_24h_accuracy = 60.0  # Lower health score

        assert not unhealthy_trend.is_healthy

    def test_to_dict_serialization(self):
        """Test dictionary serialization of metrics."""
        now = datetime.now(timezone.utc)
        metrics = RealTimeMetrics(
            room_id="test_room",
            model_type=ModelType.ENSEMBLE,
            window_1h_accuracy=80.0,
            window_6h_accuracy=75.0,
            window_24h_accuracy=78.0,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_slope=2.5,
            trend_confidence=0.8,
            active_alerts=["alert_1", "alert_2"],
            last_alert_time=now,
            dominant_accuracy_level=AccuracyLevel.GOOD,
            recent_validation_records=[],
            last_updated=now,
            measurement_start=now - timedelta(hours=12),
        )

        result_dict = metrics.to_dict()

        # Check structure
        assert "room_id" in result_dict
        assert "model_type" in result_dict
        assert "time_windows" in result_dict
        assert "trend_analysis" in result_dict
        assert "performance_indicators" in result_dict
        assert "alerts" in result_dict
        assert "accuracy_analysis" in result_dict
        assert "metadata" in result_dict

        # Check values
        assert result_dict["room_id"] == "test_room"
        assert result_dict["model_type"] == "ModelType.ENSEMBLE"
        assert result_dict["time_windows"]["1h"]["accuracy"] == 80.0
        assert result_dict["trend_analysis"]["direction"] == "improving"
        assert result_dict["alerts"]["active_alerts"] == ["alert_1", "alert_2"]
        assert result_dict["accuracy_analysis"]["dominant_accuracy_level"] == "good"


class TestAccuracyAlert:
    """Test AccuracyAlert dataclass and methods."""

    def test_accuracy_alert_initialization(self):
        """Test alert initialization."""
        now = datetime.now(timezone.utc)
        alert = AccuracyAlert(
            alert_id="test_alert_1",
            room_id="test_room",
            model_type=ModelType.LSTM,
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Accuracy below warning threshold",
            affected_metrics={"accuracy_6h": 65.0},
            recent_predictions=10,
            triggered_time=now,
        )

        assert alert.alert_id == "test_alert_1"
        assert alert.room_id == "test_room"
        assert alert.model_type == ModelType.LSTM
        assert alert.severity == AlertSeverity.WARNING
        assert alert.trigger_condition == "accuracy_warning"
        assert alert.current_value == 65.0
        assert alert.threshold_value == 70.0
        assert not alert.acknowledged
        assert not alert.resolved
        assert alert.escalation_level == 1

    def test_age_calculation(self):
        """Test alert age calculation."""
        past_time = datetime.now(timezone.utc) - timedelta(minutes=30)
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.INFO,
            trigger_condition="test_condition",
            current_value=50.0,
            threshold_value=60.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=5,
            triggered_time=past_time,
        )

        age = alert.age_minutes
        assert 29 <= age <= 31  # Should be around 30 minutes

    def test_escalation_requirements(self):
        """Test escalation logic."""
        # Recent critical alert - should not escalate yet
        recent_critical = AccuracyAlert(
            alert_id="recent_critical",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="test_condition",
            current_value=40.0,
            threshold_value=50.0,
            description="Recent critical",
            affected_metrics={},
            recent_predictions=5,
            triggered_time=datetime.now(timezone.utc) - timedelta(minutes=15),
        )
        assert not recent_critical.requires_escalation

        # Old critical alert - should escalate
        old_critical = AccuracyAlert(
            alert_id="old_critical",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="test_condition",
            current_value=40.0,
            threshold_value=50.0,
            description="Old critical",
            affected_metrics={},
            recent_predictions=5,
            triggered_time=datetime.now(timezone.utc) - timedelta(minutes=45),
        )
        assert old_critical.requires_escalation

        # Acknowledged alert - should not escalate
        acknowledged = AccuracyAlert(
            alert_id="acknowledged",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="test_condition",
            current_value=40.0,
            threshold_value=50.0,
            description="Acknowledged",
            affected_metrics={},
            recent_predictions=5,
            triggered_time=datetime.now(timezone.utc) - timedelta(hours=2),
            acknowledged=True,
        )
        assert not acknowledged.requires_escalation

    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="test_condition",
            current_value=60.0,
            threshold_value=70.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=5,
        )

        assert not alert.acknowledged

        alert.acknowledge("test_user")

        assert alert.acknowledged
        assert alert.acknowledged_by == "test_user"
        assert isinstance(alert.acknowledged_time, datetime)

    def test_resolve_alert(self):
        """Test alert resolution."""
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="test_condition",
            current_value=60.0,
            threshold_value=70.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=5,
        )

        assert not alert.resolved

        alert.resolve()

        assert alert.resolved
        assert isinstance(alert.resolved_time, datetime)

    def test_escalate_alert(self):
        """Test alert escalation."""
        old_alert = AccuracyAlert(
            alert_id="old_alert",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="test_condition",
            current_value=60.0,
            threshold_value=70.0,
            description="Old alert",
            affected_metrics={},
            recent_predictions=5,
            triggered_time=datetime.now(timezone.utc) - timedelta(hours=2),
        )

        assert old_alert.escalation_level == 1

        result = old_alert.escalate()

        assert result is True
        assert old_alert.escalation_level == 2
        assert isinstance(old_alert.last_escalation, datetime)

    def test_alert_to_dict(self):
        """Test alert dictionary serialization."""
        now = datetime.now(timezone.utc)
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            model_type=ModelType.XGBOOST,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="accuracy_critical",
            current_value=45.0,
            threshold_value=50.0,
            description="Critical accuracy drop",
            affected_metrics={"accuracy_6h": 45.0, "error_6h": 25.0},
            recent_predictions=15,
            trend_data={"direction": "degrading", "slope": -2.5},
            triggered_time=now,
        )

        result_dict = alert.to_dict()

        assert result_dict["alert_id"] == "test_alert"
        assert result_dict["room_id"] == "test_room"
        assert result_dict["model_type"] == "ModelType.XGBOOST"
        assert result_dict["severity"] == "critical"
        assert result_dict["current_value"] == 45.0
        assert result_dict["threshold_value"] == 50.0
        assert "age_minutes" in result_dict
        assert "requires_escalation" in result_dict


class TestAccuracyTracker:
    """Test AccuracyTracker main class."""

    @pytest.fixture
    def mock_validator(self):
        """Create mock PredictionValidator."""
        validator = MagicMock(spec=PredictionValidator)
        validator._lock = MagicMock()
        validator._validation_records = {}

        # Mock get_accuracy_metrics method
        mock_metrics = AccuracyMetrics(
            total_predictions=10,
            validated_predictions=8,
            accuracy_rate=75.0,
            mean_error_minutes=12.5,
            median_error_minutes=10.0,
            std_error_minutes=8.0,
            predictions_per_hour=2.0,
            confidence_calibration_score=0.8,
            accuracy_by_level={"good": 5, "fair": 3},
            error_distribution={"0-5": 2, "5-15": 4, "15-30": 2},
            trend_analysis={"slope": 1.2, "r_squared": 0.65},
        )
        validator.get_accuracy_metrics = AsyncMock(return_value=mock_metrics)

        return validator

    @pytest.fixture
    def accuracy_tracker(self, mock_validator):
        """Create AccuracyTracker instance."""
        tracker = AccuracyTracker(
            prediction_validator=mock_validator,
            monitoring_interval_seconds=30,
            alert_thresholds={
                "accuracy_warning": 70.0,
                "accuracy_critical": 50.0,
                "error_warning": 20.0,
                "error_critical": 30.0,
            },
            max_stored_alerts=100,
            trend_analysis_points=5,
        )
        return tracker

    def test_tracker_initialization(self, accuracy_tracker, mock_validator):
        """Test tracker initialization."""
        assert accuracy_tracker.validator is mock_validator
        assert accuracy_tracker.monitoring_interval.total_seconds() == 30
        assert accuracy_tracker.max_stored_alerts == 100
        assert accuracy_tracker.trend_points == 5

        # Check default thresholds
        assert accuracy_tracker.alert_thresholds["accuracy_warning"] == 70.0
        assert accuracy_tracker.alert_thresholds["accuracy_critical"] == 50.0

        # Check initial state
        assert not accuracy_tracker._monitoring_active
        assert len(accuracy_tracker._background_tasks) == 0
        assert len(accuracy_tracker._active_alerts) == 0

    async def test_start_stop_monitoring(self, accuracy_tracker):
        """Test starting and stopping monitoring tasks."""
        # Start monitoring
        await accuracy_tracker.start_monitoring()

        assert accuracy_tracker._monitoring_active
        assert (
            len(accuracy_tracker._background_tasks) == 2
        )  # monitoring + alert management

        # Stop monitoring
        await accuracy_tracker.stop_monitoring()

        assert not accuracy_tracker._monitoring_active
        assert len(accuracy_tracker._background_tasks) == 0

    async def test_start_monitoring_already_active(self, accuracy_tracker):
        """Test starting monitoring when already active."""
        await accuracy_tracker.start_monitoring()
        initial_task_count = len(accuracy_tracker._background_tasks)

        # Try to start again
        await accuracy_tracker.start_monitoring()

        # Should not create duplicate tasks
        assert len(accuracy_tracker._background_tasks) == initial_task_count

        await accuracy_tracker.stop_monitoring()

    async def test_get_real_time_metrics_room_specific(self, accuracy_tracker):
        """Test getting real-time metrics for specific room."""
        # Set up test metrics
        test_metrics = RealTimeMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            window_6h_accuracy=80.0,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._metrics_by_room["test_room"] = test_metrics

        result = await accuracy_tracker.get_real_time_metrics(room_id="test_room")

        assert isinstance(result, dict)
        assert "test_room" in [metrics.room_id for metrics in result.values()]

    async def test_get_real_time_metrics_model_specific(self, accuracy_tracker):
        """Test getting real-time metrics for specific model."""
        test_metrics = RealTimeMetrics(
            room_id="test_room",
            model_type=ModelType.XGBOOST,
            window_6h_accuracy=75.0,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._metrics_by_model["xgboost_model"] = test_metrics

        result = await accuracy_tracker.get_real_time_metrics(
            model_type=ModelType.XGBOOST
        )

        assert isinstance(result, dict)

    async def test_get_real_time_metrics_global(self, accuracy_tracker):
        """Test getting global real-time metrics."""
        global_metrics = RealTimeMetrics(
            room_id="global",
            window_6h_accuracy=78.0,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._global_metrics = global_metrics

        result = await accuracy_tracker.get_real_time_metrics()

        assert result is global_metrics
        assert result.room_id == "global"

    async def test_get_active_alerts_no_filters(self, accuracy_tracker):
        """Test getting all active alerts."""
        # Create test alerts
        alert1 = AccuracyAlert(
            alert_id="alert_1",
            room_id="room_1",
            model_type=ModelType.LSTM,
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Test alert 1",
            affected_metrics={},
            recent_predictions=5,
        )

        alert2 = AccuracyAlert(
            alert_id="alert_2",
            room_id="room_2",
            model_type=ModelType.XGBOOST,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="accuracy_critical",
            current_value=40.0,
            threshold_value=50.0,
            description="Test alert 2",
            affected_metrics={},
            recent_predictions=3,
        )

        # Add to tracker
        with accuracy_tracker._lock:
            accuracy_tracker._active_alerts[alert1.alert_id] = alert1
            accuracy_tracker._active_alerts[alert2.alert_id] = alert2

        alerts = await accuracy_tracker.get_active_alerts()

        assert len(alerts) == 2
        # Should be sorted by severity (CRITICAL first)
        assert alerts[0].severity == AlertSeverity.CRITICAL
        assert alerts[1].severity == AlertSeverity.WARNING

    async def test_get_active_alerts_room_filter(self, accuracy_tracker):
        """Test getting alerts filtered by room."""
        alert1 = AccuracyAlert(
            alert_id="alert_1",
            room_id="room_1",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Room 1 alert",
            affected_metrics={},
            recent_predictions=5,
        )

        alert2 = AccuracyAlert(
            alert_id="alert_2",
            room_id="room_2",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Room 2 alert",
            affected_metrics={},
            recent_predictions=5,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._active_alerts[alert1.alert_id] = alert1
            accuracy_tracker._active_alerts[alert2.alert_id] = alert2

        alerts = await accuracy_tracker.get_active_alerts(room_id="room_1")

        assert len(alerts) == 1
        assert alerts[0].room_id == "room_1"

    async def test_get_active_alerts_severity_filter(self, accuracy_tracker):
        """Test getting alerts filtered by severity."""
        alert1 = AccuracyAlert(
            alert_id="alert_1",
            room_id="room_1",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Warning alert",
            affected_metrics={},
            recent_predictions=5,
        )

        alert2 = AccuracyAlert(
            alert_id="alert_2",
            room_id="room_1",
            model_type=None,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="accuracy_critical",
            current_value=40.0,
            threshold_value=50.0,
            description="Critical alert",
            affected_metrics={},
            recent_predictions=3,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._active_alerts[alert1.alert_id] = alert1
            accuracy_tracker._active_alerts[alert2.alert_id] = alert2

        alerts = await accuracy_tracker.get_active_alerts(
            severity=AlertSeverity.CRITICAL
        )

        assert len(alerts) == 1
        assert alerts[0].severity == AlertSeverity.CRITICAL

    async def test_acknowledge_alert(self, accuracy_tracker):
        """Test acknowledging an alert."""
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="test_condition",
            current_value=60.0,
            threshold_value=70.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=5,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._active_alerts[alert.alert_id] = alert

        result = await accuracy_tracker.acknowledge_alert("test_alert", "test_user")

        assert result is True
        assert alert.acknowledged
        assert alert.acknowledged_by == "test_user"

    async def test_acknowledge_nonexistent_alert(self, accuracy_tracker):
        """Test acknowledging non-existent alert."""
        result = await accuracy_tracker.acknowledge_alert("nonexistent", "test_user")

        assert result is False

    async def test_get_accuracy_trends_room_specific(self, accuracy_tracker):
        """Test getting accuracy trends for specific room."""
        # Set up trend history
        with accuracy_tracker._lock:
            trend_data = [
                {
                    "timestamp": datetime.now(timezone.utc) - timedelta(hours=2),
                    "accuracy_6h": 70.0,
                },
                {
                    "timestamp": datetime.now(timezone.utc) - timedelta(hours=1),
                    "accuracy_6h": 75.0,
                },
                {"timestamp": datetime.now(timezone.utc), "accuracy_6h": 80.0},
            ]
            accuracy_tracker._accuracy_history["room_1_lstm"] = deque(
                trend_data, maxlen=5
            )
            accuracy_tracker._accuracy_history["room_2_xgboost"] = deque(
                trend_data, maxlen=5
            )

        trends = await accuracy_tracker.get_accuracy_trends(room_id="room_1")

        assert "trends_by_entity" in trends
        assert "global_trend" in trends
        assert "analysis_period_hours" in trends

        # Should only include room_1 trends
        room_1_trends = [
            key for key in trends["trends_by_entity"].keys() if key.startswith("room_1")
        ]
        assert len(room_1_trends) == 1

    async def test_get_accuracy_trends_all_rooms(self, accuracy_tracker):
        """Test getting trends for all rooms."""
        # Set up trend history for multiple entities
        with accuracy_tracker._lock:
            trend_data = [
                {
                    "timestamp": datetime.now(timezone.utc) - timedelta(hours=1),
                    "accuracy_6h": 75.0,
                },
                {"timestamp": datetime.now(timezone.utc), "accuracy_6h": 80.0},
            ]
            accuracy_tracker._accuracy_history["room_1_lstm"] = deque(
                trend_data, maxlen=5
            )
            accuracy_tracker._accuracy_history["room_2_xgboost"] = deque(
                trend_data, maxlen=5
            )
            accuracy_tracker._accuracy_history["global_ensemble"] = deque(
                trend_data, maxlen=5
            )

        trends = await accuracy_tracker.get_accuracy_trends()

        assert len(trends["trends_by_entity"]) == 3
        assert "room_1_lstm" in trends["trends_by_entity"]
        assert "room_2_xgboost" in trends["trends_by_entity"]
        assert "global_ensemble" in trends["trends_by_entity"]

    async def test_export_tracking_data(self, accuracy_tracker, tmp_path):
        """Test exporting tracking data."""
        # Set up test data
        metrics = RealTimeMetrics(room_id="test_room", window_6h_accuracy=75.0)
        alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="test_room",
            model_type=None,
            severity=AlertSeverity.WARNING,
            trigger_condition="test_condition",
            current_value=65.0,
            threshold_value=70.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=5,
        )

        with accuracy_tracker._lock:
            accuracy_tracker._metrics_by_room["test_room"] = metrics
            accuracy_tracker._alert_history.append(alert)
            accuracy_tracker._accuracy_history["test_room"] = deque(
                [{"timestamp": datetime.now(timezone.utc), "accuracy_6h": 75.0}],
                maxlen=5,
            )

        output_file = tmp_path / "tracking_export.json"
        record_count = await accuracy_tracker.export_tracking_data(
            output_path=output_file,
            include_alerts=True,
            include_trends=True,
            days_back=1,
        )

        assert record_count > 0
        assert output_file.exists()

        # Verify file content structure
        import json

        with open(output_file, "r") as f:
            data = json.load(f)

        assert "export_time" in data
        assert "metrics" in data
        assert "alerts" in data
        assert "trends" in data

    def test_notification_callback_management(self, accuracy_tracker):
        """Test adding and removing notification callbacks."""

        def test_callback(alert):
            pass

        # Add callback
        accuracy_tracker.add_notification_callback(test_callback)
        assert test_callback in accuracy_tracker.notification_callbacks

        # Remove callback
        accuracy_tracker.remove_notification_callback(test_callback)
        assert test_callback not in accuracy_tracker.notification_callbacks

    def test_get_tracker_stats(self, accuracy_tracker):
        """Test getting tracker statistics."""
        # Add some test data
        with accuracy_tracker._lock:
            accuracy_tracker._metrics_by_room["room_1"] = RealTimeMetrics(
                room_id="room_1"
            )
            accuracy_tracker._metrics_by_model["lstm"] = RealTimeMetrics(
                room_id="room_1", model_type=ModelType.LSTM
            )
            accuracy_tracker._active_alerts["alert_1"] = AccuracyAlert(
                alert_id="alert_1",
                room_id="room_1",
                model_type=None,
                severity=AlertSeverity.WARNING,
                trigger_condition="test",
                current_value=60.0,
                threshold_value=70.0,
                description="Test",
                affected_metrics={},
                recent_predictions=5,
            )
            accuracy_tracker._accuracy_history["room_1"] = deque(
                [{"accuracy_6h": 75.0}], maxlen=5
            )

        stats = accuracy_tracker.get_tracker_stats()

        assert stats["monitoring_active"] is False
        assert stats["metrics_tracked"]["rooms"] == 1
        assert stats["metrics_tracked"]["models"] == 1
        assert stats["alerts"]["active"] == 1
        assert stats["trend_tracking"]["entities_tracked"] == 1
        assert "configuration" in stats


class TestTrendAnalysis:
    """Test trend analysis functionality."""

    def test_analyze_trend_insufficient_data(self):
        """Test trend analysis with insufficient data points."""
        tracker = AccuracyTracker(
            prediction_validator=MagicMock(),
            monitoring_interval_seconds=60,
        )

        # Test with empty data
        result = tracker._analyze_trend([])
        assert result["direction"] == TrendDirection.UNKNOWN
        assert result["slope"] == 0.0
        assert result["confidence"] == 0.0

        # Test with too few points
        result = tracker._analyze_trend([{"accuracy_6h": 75.0}])
        assert result["direction"] == TrendDirection.UNKNOWN

    def test_analyze_trend_improving(self):
        """Test trend analysis with improving accuracy."""
        tracker = AccuracyTracker(
            prediction_validator=MagicMock(),
            monitoring_interval_seconds=60,
        )

        # Create improving trend data
        data_points = [
            {"accuracy_6h": 60.0},
            {"accuracy_6h": 65.0},
            {"accuracy_6h": 70.0},
            {"accuracy_6h": 75.0},
            {"accuracy_6h": 80.0},
        ]

        result = tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.IMPROVING
        assert result["slope"] > 1  # Should be positive and significant
        assert 0 <= result["confidence"] <= 1

    def test_analyze_trend_degrading(self):
        """Test trend analysis with degrading accuracy."""
        tracker = AccuracyTracker(
            prediction_validator=MagicMock(),
            monitoring_interval_seconds=60,
        )

        # Create degrading trend data
        data_points = [
            {"accuracy_6h": 80.0},
            {"accuracy_6h": 75.0},
            {"accuracy_6h": 70.0},
            {"accuracy_6h": 65.0},
            {"accuracy_6h": 60.0},
        ]

        result = tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.DEGRADING
        assert result["slope"] < -1  # Should be negative and significant
        assert 0 <= result["confidence"] <= 1

    def test_analyze_trend_stable(self):
        """Test trend analysis with stable accuracy."""
        tracker = AccuracyTracker(
            prediction_validator=MagicMock(),
            monitoring_interval_seconds=60,
        )

        # Create stable trend data
        data_points = [
            {"accuracy_6h": 75.0},
            {"accuracy_6h": 74.5},
            {"accuracy_6h": 75.2},
            {"accuracy_6h": 75.1},
            {"accuracy_6h": 74.8},
        ]

        result = tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.STABLE
        assert abs(result["slope"]) < 1  # Should be small slope
        assert 0 <= result["confidence"] <= 1

    def test_calculate_global_trend(self):
        """Test global trend calculation from individual trends."""
        tracker = AccuracyTracker(
            prediction_validator=MagicMock(),
            monitoring_interval_seconds=60,
        )

        # Test with multiple improving trends
        individual_trends = {
            "room_1": {"slope": 2.0, "confidence": 0.8},
            "room_2": {"slope": 1.5, "confidence": 0.7},
            "room_3": {"slope": 2.5, "confidence": 0.9},
        }

        global_trend = tracker._calculate_global_trend(individual_trends)

        assert global_trend["direction"] == TrendDirection.IMPROVING
        assert global_trend["average_slope"] == 2.0  # Average of 2.0, 1.5, 2.5
        assert global_trend["confidence"] > 0.7
        assert global_trend["entities_analyzed"] == 3

    def test_calculate_global_trend_empty(self):
        """Test global trend calculation with no data."""
        tracker = AccuracyTracker(
            prediction_validator=MagicMock(),
            monitoring_interval_seconds=60,
        )

        global_trend = tracker._calculate_global_trend({})

        assert global_trend["direction"] == TrendDirection.UNKNOWN
        assert global_trend["average_slope"] == 0.0
        assert global_trend["confidence"] == 0.0


class TestErrorHandling:
    """Test error handling and exception cases."""

    def test_accuracy_tracking_error(self):
        """Test custom AccuracyTrackingError exception."""
        error = AccuracyTrackingError("Test tracking error")

        assert str(error) == "Test tracking error"
        assert error.error_code == "ACCURACY_TRACKING_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_accuracy_tracking_error_with_severity(self):
        """Test AccuracyTrackingError with custom severity."""
        error = AccuracyTrackingError(
            "Critical tracking error", severity=ErrorSeverity.HIGH
        )

        assert error.severity == ErrorSeverity.HIGH

    async def test_get_real_time_metrics_error_handling(self, mock_validator):
        """Test error handling in get_real_time_metrics."""
        tracker = AccuracyTracker(mock_validator)

        # Simulate error by making validator method raise exception
        mock_validator.get_accuracy_metrics.side_effect = Exception("Validator error")

        # Should not raise exception but return None or empty results
        result = await tracker.get_real_time_metrics(room_id="test_room")

        # The method should handle the error gracefully
        assert result is None or isinstance(result, dict)

    async def test_get_active_alerts_error_handling(self, mock_validator):
        """Test error handling in get_active_alerts."""
        tracker = AccuracyTracker(mock_validator)

        # Corrupt the internal state to trigger an error
        with tracker._lock:
            tracker._active_alerts["corrupt"] = "not_an_alert_object"

        # Should handle error gracefully
        with pytest.raises(AccuracyTrackingError):
            await tracker.get_active_alerts()


class TestIntegration:
    """Integration tests for AccuracyTracker with related components."""

    @pytest.fixture
    def integration_tracker(self):
        """Create tracker with more realistic setup."""
        mock_validator = MagicMock(spec=PredictionValidator)
        mock_validator._lock = MagicMock()
        mock_validator._validation_records = {}

        # Mock realistic metrics
        mock_metrics = AccuracyMetrics(
            total_predictions=50,
            validated_predictions=45,
            accuracy_rate=78.0,
            mean_error_minutes=11.2,
            median_error_minutes=9.5,
            std_error_minutes=6.8,
            predictions_per_hour=3.5,
            confidence_calibration_score=0.85,
            accuracy_by_level={"excellent": 10, "good": 20, "fair": 15},
            error_distribution={"0-5": 15, "5-15": 20, "15-30": 10},
            trend_analysis={"slope": 0.8, "r_squared": 0.72},
        )
        mock_validator.get_accuracy_metrics = AsyncMock(return_value=mock_metrics)

        tracker = AccuracyTracker(
            prediction_validator=mock_validator,
            monitoring_interval_seconds=60,
            alert_thresholds={
                "accuracy_warning": 70.0,
                "accuracy_critical": 50.0,
                "error_warning": 15.0,
                "error_critical": 25.0,
            },
        )

        return tracker

    async def test_full_monitoring_cycle(self, integration_tracker):
        """Test complete monitoring cycle with metrics and alerts."""
        # Start monitoring
        await integration_tracker.start_monitoring()

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Check that monitoring is active
        assert integration_tracker._monitoring_active

        # Stop monitoring
        await integration_tracker.stop_monitoring()

        assert not integration_tracker._monitoring_active

    async def test_alert_lifecycle(self, integration_tracker):
        """Test complete alert lifecycle from creation to resolution."""
        # Create metrics that would trigger an alert
        poor_metrics = RealTimeMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            window_6h_accuracy=45.0,  # Below critical threshold
            window_6h_predictions=10,
        )

        with integration_tracker._lock:
            integration_tracker._metrics_by_room["test_room"] = poor_metrics

        # Check alert conditions
        await integration_tracker._check_alert_conditions()

        # Should have created an alert
        alerts = await integration_tracker.get_active_alerts()
        assert len(alerts) > 0

        critical_alert = None
        for alert in alerts:
            if alert.severity == AlertSeverity.CRITICAL:
                critical_alert = alert
                break

        assert critical_alert is not None
        assert critical_alert.trigger_condition == "accuracy_critical"

        # Acknowledge the alert
        result = await integration_tracker.acknowledge_alert(
            critical_alert.alert_id, "test_operator"
        )
        assert result is True
        assert critical_alert.acknowledged

        # Improve metrics to trigger auto-resolution
        improved_metrics = RealTimeMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            window_6h_accuracy=75.0,  # Above threshold
            window_6h_predictions=12,
        )

        with integration_tracker._lock:
            integration_tracker._metrics_by_room["test_room"] = improved_metrics

        # Check if alert should be auto-resolved
        should_resolve = await integration_tracker._should_auto_resolve_alert(
            critical_alert
        )
        assert should_resolve is True
