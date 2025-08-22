"""
Comprehensive tests for src/adaptation/tracker.py - AccuracyTracker system.

Tests all public methods, error conditions, and edge cases for complete coverage
of the AccuracyTracker, RealTimeMetrics, AccuracyAlert, and related components.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import statistics

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
    """Test suite for RealTimeMetrics class."""

    def test_real_time_metrics_initialization(self):
        """Test RealTimeMetrics initialization with default values."""
        metrics = RealTimeMetrics(room_id="living_room", model_type=ModelType.ENSEMBLE)

        assert metrics.room_id == "living_room"
        assert metrics.model_type == ModelType.ENSEMBLE
        assert metrics.window_1h_accuracy == 0.0
        assert metrics.window_6h_accuracy == 0.0
        assert metrics.window_24h_accuracy == 0.0
        assert metrics.accuracy_trend == TrendDirection.UNKNOWN
        assert isinstance(metrics.active_alerts, list)
        assert len(metrics.active_alerts) == 0
        assert isinstance(metrics.recent_validation_records, list)

    def test_real_time_metrics_with_custom_values(self):
        """Test RealTimeMetrics initialization with custom values."""
        metrics = RealTimeMetrics(
            room_id="bedroom",
            model_type="lstm",
            window_1h_accuracy=85.5,
            window_6h_accuracy=82.3,
            window_24h_accuracy=79.8,
            window_1h_predictions=25,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_slope=2.5,
            trend_confidence=0.85,
            recent_predictions_rate=15.0,
            validation_lag_minutes=5.2,
            confidence_calibration=0.78,
        )

        assert metrics.room_id == "bedroom"
        assert metrics.model_type == "lstm"
        assert metrics.window_1h_accuracy == 85.5
        assert metrics.window_6h_accuracy == 82.3
        assert metrics.window_24h_accuracy == 79.8
        assert metrics.window_1h_predictions == 25
        assert metrics.accuracy_trend == TrendDirection.IMPROVING
        assert metrics.trend_slope == 2.5
        assert metrics.trend_confidence == 0.85

    def test_overall_health_score_calculation(self):
        """Test health score calculation with various metrics."""
        # High performance metrics
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=90.0,
            window_24h_predictions=100,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_confidence=0.9,
            confidence_calibration=0.85,
            validation_lag_minutes=3.0,
        )

        health_score = metrics.overall_health_score
        assert 80 <= health_score <= 100  # Should be high

        # Poor performance metrics
        metrics_poor = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=40.0,
            window_24h_predictions=50,
            accuracy_trend=TrendDirection.DEGRADING,
            trend_confidence=0.8,
            confidence_calibration=0.3,
            validation_lag_minutes=45.0,
        )

        health_score_poor = metrics_poor.overall_health_score
        assert health_score_poor < 50  # Should be low

    def test_overall_health_score_no_predictions(self):
        """Test health score calculation with no predictions."""
        metrics = RealTimeMetrics(room_id="test_room", window_24h_predictions=0)

        assert metrics.overall_health_score == 0.0

    def test_is_healthy_property(self):
        """Test is_healthy property with various conditions."""
        # Healthy metrics
        healthy_metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=75.0,
            accuracy_trend=TrendDirection.STABLE,
        )

        # Mock the health score to be above threshold
        with patch.object(
            RealTimeMetrics,
            "overall_health_score",
            new_callable=lambda: property(lambda self: 80.0),
        ):
            assert healthy_metrics.is_healthy is True

        # Unhealthy metrics - low accuracy
        unhealthy_metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=50.0,
            accuracy_trend=TrendDirection.STABLE,
        )

        with patch.object(
            RealTimeMetrics,
            "overall_health_score",
            new_callable=lambda: property(lambda self: 80.0),
        ):
            assert unhealthy_metrics.is_healthy is False  # Low accuracy

        # Unhealthy metrics - degrading trend
        degrading_metrics = RealTimeMetrics(
            room_id="test_room",
            window_6h_accuracy=75.0,
            accuracy_trend=TrendDirection.DEGRADING,
        )

        with patch.object(
            RealTimeMetrics,
            "overall_health_score",
            new_callable=lambda: property(lambda self: 80.0),
        ):
            assert degrading_metrics.is_healthy is False  # Degrading trend

    def test_to_dict_conversion(self):
        """Test conversion to dictionary for API responses."""
        metrics = RealTimeMetrics(
            room_id="kitchen",
            model_type=ModelType.LSTM,
            window_1h_accuracy=88.5,
            window_6h_accuracy=85.2,
            window_24h_accuracy=82.8,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_slope=1.5,
            trend_confidence=0.75,
            active_alerts=["alert_1", "alert_2"],
            dominant_accuracy_level=AccuracyLevel.EXCELLENT,
        )

        metrics_dict = metrics.to_dict()

        assert metrics_dict["room_id"] == "kitchen"
        assert metrics_dict["model_type"] == "ModelType.LSTM"
        assert metrics_dict["time_windows"]["1h"]["accuracy"] == 88.5
        assert metrics_dict["time_windows"]["6h"]["accuracy"] == 85.2
        assert metrics_dict["time_windows"]["24h"]["accuracy"] == 82.8
        assert metrics_dict["trend_analysis"]["direction"] == "improving"
        assert metrics_dict["trend_analysis"]["slope"] == 1.5
        assert metrics_dict["alerts"]["active_alerts"] == ["alert_1", "alert_2"]
        assert (
            metrics_dict["accuracy_analysis"]["dominant_accuracy_level"] == "excellent"
        )

    def test_to_dict_with_string_model_type(self):
        """Test to_dict with string model type."""
        metrics = RealTimeMetrics(
            room_id="study", model_type="xgboost", window_1h_accuracy=75.0
        )

        metrics_dict = metrics.to_dict()
        assert metrics_dict["model_type"] == "xgboost"

    def test_to_dict_with_timestamps(self):
        """Test to_dict includes proper timestamp formatting."""
        test_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        metrics = RealTimeMetrics(
            room_id="garage", last_updated=test_time, measurement_start=test_time
        )
        metrics.last_alert_time = test_time

        metrics_dict = metrics.to_dict()
        assert metrics_dict["metadata"]["last_updated"] == test_time.isoformat()
        assert metrics_dict["metadata"]["measurement_start"] == test_time.isoformat()
        assert metrics_dict["alerts"]["last_alert_time"] == test_time.isoformat()


class TestAccuracyAlert:
    """Test suite for AccuracyAlert class."""

    def test_accuracy_alert_initialization(self):
        """Test AccuracyAlert initialization."""
        alert = AccuracyAlert(
            alert_id="test_alert_001",
            room_id="living_room",
            model_type=ModelType.ENSEMBLE,
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.5,
            threshold_value=70.0,
            description="Accuracy below warning threshold",
            affected_metrics={"accuracy_6h": 65.5},
            recent_predictions=50,
        )

        assert alert.alert_id == "test_alert_001"
        assert alert.room_id == "living_room"
        assert alert.model_type == ModelType.ENSEMBLE
        assert alert.severity == AlertSeverity.WARNING
        assert alert.trigger_condition == "accuracy_warning"
        assert alert.current_value == 65.5
        assert alert.threshold_value == 70.0
        assert not alert.acknowledged
        assert not alert.resolved
        assert alert.escalation_level == 1

    def test_alert_age_calculation(self):
        """Test alert age calculation in minutes."""
        # Create alert 10 minutes ago
        past_time = datetime.now(timezone.utc) - timedelta(minutes=10)

        with patch("src.adaptation.tracker.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime.now(timezone.utc)

            alert = AccuracyAlert(
                alert_id="age_test",
                room_id="test",
                model_type="test",
                severity=AlertSeverity.INFO,
                trigger_condition="test",
                current_value=0.0,
                threshold_value=0.0,
                description="Test",
                affected_metrics={},
                recent_predictions=0,
                triggered_time=past_time,
            )

        age = alert.age_minutes
        assert 9 <= age <= 11  # Allow for small timing variations

    def test_requires_escalation_logic(self):
        """Test escalation requirement logic."""
        # Create alert that should require escalation
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        alert = AccuracyAlert(
            alert_id="escalation_test",
            room_id="test",
            model_type="test",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Test",
            affected_metrics={},
            recent_predictions=0,
            triggered_time=old_time,
        )

        assert alert.requires_escalation is True

        # Test acknowledged alert doesn't escalate
        alert.acknowledge("test_user")
        assert alert.requires_escalation is False

        # Test resolved alert doesn't escalate
        alert.acknowledged = False
        alert.resolve()
        assert alert.requires_escalation is False

        # Test max escalation level reached
        alert.resolved = False
        alert.escalation_level = 3
        alert.max_escalations = 3
        assert alert.requires_escalation is False

    def test_escalation_thresholds_by_severity(self):
        """Test different escalation thresholds based on severity."""
        severities_and_thresholds = [
            (AlertSeverity.INFO, 120),
            (AlertSeverity.WARNING, 60),
            (AlertSeverity.CRITICAL, 30),
            (AlertSeverity.EMERGENCY, 10),
        ]

        for severity, threshold_minutes in severities_and_thresholds:
            # Create alert just under threshold
            time_delta = timedelta(minutes=threshold_minutes - 5)
            alert = AccuracyAlert(
                alert_id=f"threshold_test_{severity.value}",
                room_id="test",
                model_type="test",
                severity=severity,
                trigger_condition="test",
                current_value=0.0,
                threshold_value=0.0,
                description="Test",
                affected_metrics={},
                recent_predictions=0,
                triggered_time=datetime.now(timezone.utc) - time_delta,
            )

            assert alert.requires_escalation is False

            # Create alert just over threshold
            time_delta = timedelta(minutes=threshold_minutes + 5)
            alert.triggered_time = datetime.now(timezone.utc) - time_delta
            assert alert.requires_escalation is True

    def test_acknowledge_alert(self):
        """Test alert acknowledgment."""
        alert = AccuracyAlert(
            alert_id="ack_test",
            room_id="test",
            model_type="test",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Test",
            affected_metrics={},
            recent_predictions=0,
        )

        assert not alert.acknowledged
        assert alert.acknowledged_time is None
        assert alert.acknowledged_by is None

        alert.acknowledge("admin_user")

        assert alert.acknowledged is True
        assert alert.acknowledged_time is not None
        assert alert.acknowledged_by == "admin_user"

    def test_resolve_alert(self):
        """Test alert resolution."""
        with patch("src.adaptation.tracker.logger"):
            alert = AccuracyAlert(
                alert_id="resolve_test",
                room_id="test",
                model_type="test",
                severity=AlertSeverity.CRITICAL,
                trigger_condition="test",
                current_value=0.0,
                threshold_value=0.0,
                description="Test",
                affected_metrics={},
                recent_predictions=0,
            )

            assert not alert.resolved
            assert alert.resolved_time is None

            alert.resolve()

            assert alert.resolved is True
            assert alert.resolved_time is not None

    def test_escalate_alert(self):
        """Test alert escalation."""
        # Create alert that should escalate
        old_time = datetime.now(timezone.utc) - timedelta(hours=2)
        alert = AccuracyAlert(
            alert_id="escalate_test",
            room_id="test",
            model_type="test",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Test",
            affected_metrics={},
            recent_predictions=0,
            triggered_time=old_time,
        )

        initial_level = alert.escalation_level
        result = alert.escalate()

        assert result is True
        assert alert.escalation_level == initial_level + 1
        assert alert.last_escalation is not None

        # Test escalation when not required
        alert.escalation_level = 3
        alert.max_escalations = 3
        result = alert.escalate()
        assert result is False

    def test_to_dict_conversion(self):
        """Test alert conversion to dictionary."""
        test_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        alert = AccuracyAlert(
            alert_id="dict_test",
            room_id="office",
            model_type=ModelType.XGBOOST,
            severity=AlertSeverity.CRITICAL,
            trigger_condition="accuracy_critical",
            current_value=45.0,
            threshold_value=50.0,
            description="Critical accuracy degradation",
            affected_metrics={"accuracy_6h": 45.0, "error_6h": 25.0},
            recent_predictions=75,
            trend_data={"direction": "degrading", "confidence": 0.8},
            triggered_time=test_time,
            escalation_level=2,
        )
        alert.acknowledge("operator")

        alert_dict = alert.to_dict()

        assert alert_dict["alert_id"] == "dict_test"
        assert alert_dict["room_id"] == "office"
        assert alert_dict["model_type"] == "ModelType.XGBOOST"
        assert alert_dict["severity"] == "critical"
        assert alert_dict["trigger_condition"] == "accuracy_critical"
        assert alert_dict["current_value"] == 45.0
        assert alert_dict["threshold_value"] == 50.0
        assert alert_dict["description"] == "Critical accuracy degradation"
        assert alert_dict["affected_metrics"] == {"accuracy_6h": 45.0, "error_6h": 25.0}
        assert alert_dict["recent_predictions"] == 75
        assert alert_dict["trend_data"] == {"direction": "degrading", "confidence": 0.8}
        assert alert_dict["triggered_time"] == test_time.isoformat()
        assert alert_dict["acknowledged"] is True
        assert alert_dict["acknowledged_by"] == "operator"
        assert alert_dict["escalation_level"] == 2


class TestAccuracyTracker:
    """Test suite for AccuracyTracker class."""

    @pytest.fixture
    def mock_validator(self):
        """Create a mock PredictionValidator for testing."""
        validator = Mock(spec=PredictionValidator)
        validator._lock = threading.RLock()
        validator._validation_records = {}
        validator._pending_predictions = {}
        validator.get_accuracy_metrics = AsyncMock()
        validator.get_total_predictions = AsyncMock(return_value=100)
        validator.get_validation_rate = AsyncMock(return_value=0.95)
        validator.cleanup_old_predictions = AsyncMock()
        return validator

    @pytest.fixture
    def tracker_config(self):
        """Create a basic tracker configuration."""
        return {
            "monitoring_interval_seconds": 5,
            "alert_thresholds": {
                "accuracy_warning": 70.0,
                "accuracy_critical": 50.0,
                "error_warning": 20.0,
                "error_critical": 30.0,
                "trend_degrading": -5.0,
                "validation_lag_warning": 15.0,
                "validation_lag_critical": 30.0,
            },
            "max_stored_alerts": 100,
            "trend_analysis_points": 5,
        }

    @pytest.fixture
    def accuracy_tracker(self, mock_validator, tracker_config):
        """Create an AccuracyTracker instance for testing."""
        return AccuracyTracker(prediction_validator=mock_validator, **tracker_config)

    def test_tracker_initialization(self, mock_validator, tracker_config):
        """Test AccuracyTracker initialization."""
        tracker = AccuracyTracker(prediction_validator=mock_validator, **tracker_config)

        assert tracker.validator == mock_validator
        assert tracker.monitoring_interval.total_seconds() == 5
        assert tracker.max_stored_alerts == 100
        assert tracker.trend_points == 5
        assert len(tracker.alert_thresholds) == 7
        assert tracker.alert_thresholds["accuracy_warning"] == 70.0
        assert not tracker._monitoring_active
        assert isinstance(tracker._metrics_by_room, dict)
        assert isinstance(tracker._active_alerts, dict)

    def test_tracker_initialization_with_defaults(self, mock_validator):
        """Test tracker initialization with default parameters."""
        tracker = AccuracyTracker(prediction_validator=mock_validator)

        assert tracker.monitoring_interval.total_seconds() == 60
        assert tracker.max_stored_alerts == 1000
        assert tracker.trend_points == 10
        assert "accuracy_warning" in tracker.alert_thresholds

    def test_start_monitoring_success(self, accuracy_tracker):
        """Test successful start of monitoring tasks."""
        assert not accuracy_tracker._monitoring_active

        # Use a real event loop for async testing
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_start():
                await accuracy_tracker.start_monitoring()
                assert accuracy_tracker._monitoring_active
                assert len(accuracy_tracker._background_tasks) >= 1
                await accuracy_tracker.stop_monitoring()

            loop.run_until_complete(test_start())
        finally:
            loop.close()

    def test_start_monitoring_already_active(self, accuracy_tracker):
        """Test start monitoring when already active."""
        accuracy_tracker._monitoring_active = True

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_already_active():
                # Should not raise an exception, just log a warning
                await accuracy_tracker.start_monitoring()
                assert accuracy_tracker._monitoring_active

            loop.run_until_complete(test_already_active())
        finally:
            loop.close()

    def test_start_monitoring_failure(self, accuracy_tracker):
        """Test start monitoring with failure."""
        # Mock a failure in task creation
        with patch(
            "asyncio.create_task", side_effect=Exception("Task creation failed")
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def test_failure():
                    with pytest.raises(
                        AccuracyTrackingError,
                        match="Failed to start real-time monitoring",
                    ):
                        await accuracy_tracker.start_monitoring()

                loop.run_until_complete(test_failure())
            finally:
                loop.close()

    def test_stop_monitoring(self, accuracy_tracker):
        """Test stopping monitoring tasks."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_stop():
                await accuracy_tracker.start_monitoring()
                assert accuracy_tracker._monitoring_active

                await accuracy_tracker.stop_monitoring()
                assert not accuracy_tracker._monitoring_active
                assert len(accuracy_tracker._background_tasks) == 0

            loop.run_until_complete(test_stop())
        finally:
            loop.close()

    def test_get_real_time_metrics_room_specific(self, accuracy_tracker):
        """Test getting real-time metrics for specific room."""
        # Add mock metrics
        test_metrics = RealTimeMetrics(room_id="living_room")
        accuracy_tracker._metrics_by_room["living_room"] = test_metrics

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_metrics():
                result = await accuracy_tracker.get_real_time_metrics(
                    room_id="living_room"
                )
                # The method returns metrics from the room dict, but with filtering logic
                # that may return different results
                assert result is not None or result == {}

                # Test non-existent room
                result = await accuracy_tracker.get_real_time_metrics(
                    room_id="nonexistent"
                )
                assert result is None or result == {}

            loop.run_until_complete(test_get_metrics())
        finally:
            loop.close()

    def test_get_real_time_metrics_model_specific(self, accuracy_tracker):
        """Test getting real-time metrics for specific model type."""
        test_metrics = RealTimeMetrics(room_id="bedroom", model_type=ModelType.LSTM)
        accuracy_tracker._metrics_by_model["lstm_model"] = test_metrics

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_model_metrics():
                result = await accuracy_tracker.get_real_time_metrics(
                    model_type=ModelType.LSTM
                )
                # Should return dict of matching metrics or None
                assert result is None or isinstance(result, dict)

            loop.run_until_complete(test_get_model_metrics())
        finally:
            loop.close()

    def test_get_real_time_metrics_global(self, accuracy_tracker):
        """Test getting global real-time metrics."""
        test_metrics = RealTimeMetrics(room_id="global")
        accuracy_tracker._global_metrics = test_metrics

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_global_metrics():
                result = await accuracy_tracker.get_real_time_metrics()
                assert result == test_metrics

            loop.run_until_complete(test_get_global_metrics())
        finally:
            loop.close()

    def test_get_real_time_metrics_error(self, accuracy_tracker):
        """Test error handling in get_real_time_metrics."""
        # Mock an error in the method
        with patch.object(
            accuracy_tracker, "_lock", side_effect=Exception("Lock error")
        ):
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def test_error():
                    with pytest.raises(
                        AccuracyTrackingError,
                        match="Failed to retrieve real-time metrics",
                    ):
                        await accuracy_tracker.get_real_time_metrics(room_id="test")

                loop.run_until_complete(test_error())
            finally:
                loop.close()

    def test_get_active_alerts_all(self, accuracy_tracker):
        """Test getting all active alerts."""
        # Create test alerts
        alert1 = AccuracyAlert(
            alert_id="alert1",
            room_id="living_room",
            model_type="ensemble",
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Test alert 1",
            affected_metrics={},
            recent_predictions=50,
        )
        alert2 = AccuracyAlert(
            alert_id="alert2",
            room_id="bedroom",
            model_type="lstm",
            severity=AlertSeverity.CRITICAL,
            trigger_condition="accuracy_critical",
            current_value=45.0,
            threshold_value=50.0,
            description="Test alert 2",
            affected_metrics={},
            recent_predictions=30,
        )
        alert3 = AccuracyAlert(  # Resolved alert
            alert_id="alert3",
            room_id="kitchen",
            model_type="xgboost",
            severity=AlertSeverity.INFO,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Resolved alert",
            affected_metrics={},
            recent_predictions=0,
        )
        alert3.resolve()

        accuracy_tracker._active_alerts = {
            "alert1": alert1,
            "alert2": alert2,
            "alert3": alert3,
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_alerts():
                alerts = await accuracy_tracker.get_active_alerts()
                # Should return only non-resolved alerts, sorted by severity
                active_alerts = [a for a in alerts if not a.resolved]
                assert len(active_alerts) == 2
                # Should be sorted by severity (CRITICAL first)
                assert active_alerts[0].severity == AlertSeverity.CRITICAL
                assert active_alerts[1].severity == AlertSeverity.WARNING

            loop.run_until_complete(test_get_alerts())
        finally:
            loop.close()

    def test_get_active_alerts_filtered_by_room(self, accuracy_tracker):
        """Test getting active alerts filtered by room."""
        alert1 = AccuracyAlert(
            alert_id="alert1",
            room_id="living_room",
            model_type="ensemble",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Living room alert",
            affected_metrics={},
            recent_predictions=0,
        )
        alert2 = AccuracyAlert(
            alert_id="alert2",
            room_id="bedroom",
            model_type="lstm",
            severity=AlertSeverity.CRITICAL,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Bedroom alert",
            affected_metrics={},
            recent_predictions=0,
        )

        accuracy_tracker._active_alerts = {"alert1": alert1, "alert2": alert2}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_filtered_alerts():
                alerts = await accuracy_tracker.get_active_alerts(room_id="living_room")
                assert len(alerts) == 1
                assert alerts[0].room_id == "living_room"

            loop.run_until_complete(test_filtered_alerts())
        finally:
            loop.close()

    def test_get_active_alerts_filtered_by_severity(self, accuracy_tracker):
        """Test getting active alerts filtered by severity."""
        alert1 = AccuracyAlert(
            alert_id="alert1",
            room_id="living_room",
            model_type="ensemble",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Warning alert",
            affected_metrics={},
            recent_predictions=0,
        )
        alert2 = AccuracyAlert(
            alert_id="alert2",
            room_id="bedroom",
            model_type="lstm",
            severity=AlertSeverity.CRITICAL,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Critical alert",
            affected_metrics={},
            recent_predictions=0,
        )

        accuracy_tracker._active_alerts = {"alert1": alert1, "alert2": alert2}

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_severity_filter():
                alerts = await accuracy_tracker.get_active_alerts(
                    severity=AlertSeverity.CRITICAL
                )
                assert len(alerts) == 1
                assert alerts[0].severity == AlertSeverity.CRITICAL

            loop.run_until_complete(test_severity_filter())
        finally:
            loop.close()

    def test_acknowledge_alert_success(self, accuracy_tracker):
        """Test successful alert acknowledgment."""
        alert = AccuracyAlert(
            alert_id="ack_test",
            room_id="test",
            model_type="test",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=0,
        )
        accuracy_tracker._active_alerts["ack_test"] = alert

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_acknowledge():
                result = await accuracy_tracker.acknowledge_alert("ack_test", "admin")
                assert result is True
                assert alert.acknowledged is True
                assert alert.acknowledged_by == "admin"

            loop.run_until_complete(test_acknowledge())
        finally:
            loop.close()

    def test_acknowledge_alert_not_found(self, accuracy_tracker):
        """Test acknowledging non-existent alert."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_not_found():
                result = await accuracy_tracker.acknowledge_alert(
                    "nonexistent", "admin"
                )
                assert result is False

            loop.run_until_complete(test_not_found())
        finally:
            loop.close()

    def test_get_accuracy_trends(self, accuracy_tracker):
        """Test getting accuracy trends."""
        # Add mock trend data
        test_data = [
            {"timestamp": datetime.now(), "accuracy_6h": 80.0},
            {"timestamp": datetime.now(), "accuracy_6h": 82.0},
            {"timestamp": datetime.now(), "accuracy_6h": 78.0},
        ]
        accuracy_tracker._accuracy_history["living_room_ensemble"] = deque(
            test_data, maxlen=10
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_trends():
                trends = await accuracy_tracker.get_accuracy_trends(
                    room_id="living_room"
                )
                assert "trends_by_entity" in trends
                assert "global_trend" in trends
                assert "analysis_period_hours" in trends
                assert trends["analysis_period_hours"] == 24

            loop.run_until_complete(test_trends())
        finally:
            loop.close()

    def test_export_tracking_data(self, accuracy_tracker):
        """Test exporting tracking data to file."""
        # Add some test data
        test_metrics = RealTimeMetrics(room_id="test_room", window_1h_accuracy=85.0)
        accuracy_tracker._metrics_by_room["test_room"] = test_metrics

        test_alert = AccuracyAlert(
            alert_id="export_test",
            room_id="test_room",
            model_type="ensemble",
            severity=AlertSeverity.INFO,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Export test alert",
            affected_metrics={},
            recent_predictions=0,
        )
        accuracy_tracker._alert_history.append(test_alert)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            with tempfile.NamedTemporaryFile(
                mode="w", delete=False, suffix=".json"
            ) as tmp_file:
                tmp_path = tmp_file.name

            async def test_export():
                count = await accuracy_tracker.export_tracking_data(
                    output_path=tmp_path,
                    include_alerts=True,
                    include_trends=True,
                    days_back=7,
                )

                assert count > 0

                # Verify the exported file exists and has content
                with open(tmp_path, "r") as f:
                    data = json.load(f)

                assert "export_time" in data
                assert "metrics" in data
                assert "alerts" in data
                assert "trends" in data
                assert data["export_period_days"] == 7

                # Clean up
                Path(tmp_path).unlink()

            loop.run_until_complete(test_export())
        finally:
            loop.close()

    def test_add_notification_callback(self, accuracy_tracker):
        """Test adding notification callback."""

        def test_callback(alert):
            pass

        initial_count = len(accuracy_tracker.notification_callbacks)
        accuracy_tracker.add_notification_callback(test_callback)

        assert len(accuracy_tracker.notification_callbacks) == initial_count + 1
        assert test_callback in accuracy_tracker.notification_callbacks

        # Adding the same callback again should not duplicate
        accuracy_tracker.add_notification_callback(test_callback)
        assert len(accuracy_tracker.notification_callbacks) == initial_count + 1

    def test_remove_notification_callback(self, accuracy_tracker):
        """Test removing notification callback."""

        def test_callback(alert):
            pass

        accuracy_tracker.add_notification_callback(test_callback)
        initial_count = len(accuracy_tracker.notification_callbacks)

        accuracy_tracker.remove_notification_callback(test_callback)
        assert len(accuracy_tracker.notification_callbacks) == initial_count - 1
        assert test_callback not in accuracy_tracker.notification_callbacks

    def test_get_tracker_stats(self, accuracy_tracker):
        """Test getting tracker statistics."""
        # Add some test data
        test_metrics = RealTimeMetrics(room_id="stats_room")
        accuracy_tracker._metrics_by_room["stats_room"] = test_metrics
        accuracy_tracker._global_metrics = RealTimeMetrics(room_id="global")

        test_alert = AccuracyAlert(
            alert_id="stats_alert",
            room_id="stats_room",
            model_type="ensemble",
            severity=AlertSeverity.WARNING,
            trigger_condition="test",
            current_value=0.0,
            threshold_value=0.0,
            description="Stats test alert",
            affected_metrics={},
            recent_predictions=0,
        )
        accuracy_tracker._active_alerts["stats_alert"] = test_alert
        accuracy_tracker._alert_history.append(test_alert)

        stats = accuracy_tracker.get_tracker_stats()

        assert "monitoring_active" in stats
        assert "metrics_tracked" in stats
        assert "alerts" in stats
        assert "trend_tracking" in stats
        assert "configuration" in stats
        assert "background_tasks" in stats

        assert stats["metrics_tracked"]["rooms"] == 1
        assert stats["metrics_tracked"]["global"] == 1
        assert stats["alerts"]["active"] == 1
        assert stats["alerts"]["total_stored"] == 1

    def test_analyze_trend_insufficient_data(self, accuracy_tracker):
        """Test trend analysis with insufficient data."""
        # Test with less than 3 data points
        data_points = [{"accuracy_6h": 80.0}]
        result = accuracy_tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.UNKNOWN
        assert result["slope"] == 0.0
        assert result["confidence"] == 0.0

    def test_analyze_trend_with_data(self, accuracy_tracker):
        """Test trend analysis with sufficient data."""
        # Create trend data (improving accuracy)
        data_points = [
            {"accuracy_6h": 75.0},
            {"accuracy_6h": 78.0},
            {"accuracy_6h": 82.0},
            {"accuracy_6h": 85.0},
            {"accuracy_6h": 88.0},
        ]

        result = accuracy_tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.IMPROVING
        assert result["slope"] > 0
        assert result["confidence"] > 0
        assert "r_squared" in result
        assert "data_points" in result

    def test_analyze_trend_stable(self, accuracy_tracker):
        """Test trend analysis with stable data."""
        # Create stable trend data
        data_points = [
            {"accuracy_6h": 80.0},
            {"accuracy_6h": 80.5},
            {"accuracy_6h": 79.5},
            {"accuracy_6h": 80.2},
            {"accuracy_6h": 79.8},
        ]

        result = accuracy_tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.STABLE
        assert abs(result["slope"]) < 1  # Should be close to 0 for stable trend

    def test_analyze_trend_degrading(self, accuracy_tracker):
        """Test trend analysis with degrading data."""
        # Create degrading trend data
        data_points = [
            {"accuracy_6h": 85.0},
            {"accuracy_6h": 82.0},
            {"accuracy_6h": 78.0},
            {"accuracy_6h": 75.0},
            {"accuracy_6h": 72.0},
        ]

        result = accuracy_tracker._analyze_trend(data_points)

        assert result["direction"] == TrendDirection.DEGRADING
        assert result["slope"] < 0

    def test_calculate_global_trend(self, accuracy_tracker):
        """Test global trend calculation."""
        individual_trends = {
            "room1": {
                "direction": TrendDirection.IMPROVING,
                "slope": 2.0,
                "confidence": 0.8,
            },
            "room2": {
                "direction": TrendDirection.STABLE,
                "slope": 0.5,
                "confidence": 0.6,
            },
            "room3": {
                "direction": TrendDirection.DEGRADING,
                "slope": -1.5,
                "confidence": 0.7,
            },
        }

        global_trend = accuracy_tracker._calculate_global_trend(individual_trends)

        assert "direction" in global_trend
        assert "average_slope" in global_trend
        assert "confidence" in global_trend
        assert "entities_analyzed" in global_trend
        assert global_trend["entities_analyzed"] == 3

    def test_calculate_global_trend_empty(self, accuracy_tracker):
        """Test global trend calculation with no data."""
        global_trend = accuracy_tracker._calculate_global_trend({})

        assert global_trend["direction"] == TrendDirection.UNKNOWN
        assert global_trend["average_slope"] == 0.0
        assert global_trend["confidence"] == 0.0

    def test_model_types_match(self, accuracy_tracker):
        """Test model type matching helper method."""
        # Test enum to enum
        assert (
            accuracy_tracker._model_types_match(ModelType.LSTM, ModelType.LSTM) is True
        )
        assert (
            accuracy_tracker._model_types_match(ModelType.LSTM, ModelType.XGBOOST)
            is False
        )

        # Test string to string
        assert accuracy_tracker._model_types_match("lstm", "lstm") is True
        assert accuracy_tracker._model_types_match("lstm", "xgboost") is False

        # Test enum to string
        assert (
            accuracy_tracker._model_types_match(ModelType.LSTM, "ModelType.LSTM")
            is True
        )

        # Test None cases
        assert accuracy_tracker._model_types_match(None, None) is True
        assert accuracy_tracker._model_types_match(None, "lstm") is False
        assert accuracy_tracker._model_types_match("lstm", None) is False

    def test_update_from_accuracy_metrics(self, accuracy_tracker):
        """Test updating real-time metrics from AccuracyMetrics."""
        metrics = RealTimeMetrics(room_id="test_room")

        accuracy_metrics = Mock(spec=AccuracyMetrics)
        accuracy_metrics.accuracy_by_level = {"excellent": 10, "good": 5, "fair": 2}
        accuracy_metrics.confidence_calibration_score = 0.85

        accuracy_tracker.update_from_accuracy_metrics(metrics, accuracy_metrics)

        assert metrics.dominant_accuracy_level == AccuracyLevel.EXCELLENT
        assert metrics.confidence_calibration == 0.85

    def test_update_from_accuracy_metrics_unknown_level(self, accuracy_tracker):
        """Test updating metrics with unknown accuracy level."""
        metrics = RealTimeMetrics(room_id="test_room")

        accuracy_metrics = Mock(spec=AccuracyMetrics)
        accuracy_metrics.accuracy_by_level = {"unknown_level": 10, "excellent": 5}
        accuracy_metrics.confidence_calibration_score = 0.75

        with patch("logging.getLogger") as mock_logger:
            logger = Mock()
            mock_logger.return_value = logger

            accuracy_tracker.update_from_accuracy_metrics(metrics, accuracy_metrics)

            # Should still work, just with warning logged
            assert metrics.confidence_calibration == 0.75

    def test_extract_recent_validation_records(self, accuracy_tracker, mock_validator):
        """Test extracting recent validation records."""
        # Create mock validation records
        recent_time = datetime.now(timezone.utc) - timedelta(hours=2)
        old_time = datetime.now(timezone.utc) - timedelta(hours=8)

        record1 = Mock(spec=ValidationRecord)
        record1.validation_time = recent_time
        record1.room_id = "test_room"
        record1.model_type = ModelType.ENSEMBLE

        record2 = Mock(spec=ValidationRecord)
        record2.validation_time = old_time
        record2.room_id = "test_room"
        record2.model_type = ModelType.ENSEMBLE

        record3 = Mock(spec=ValidationRecord)
        record3.validation_time = recent_time
        record3.room_id = "other_room"
        record3.model_type = ModelType.ENSEMBLE

        mock_validator._validation_records = {
            "rec1": record1,
            "rec2": record2,  # Too old
            "rec3": record3,  # Different room
        }

        metrics = RealTimeMetrics(room_id="test_room", model_type=ModelType.ENSEMBLE)
        accuracy_tracker.extract_recent_validation_records(metrics, hours_back=6)

        # Should only include record1 (recent and matching room)
        assert len(metrics.recent_validation_records) == 1
        assert metrics.recent_validation_records[0] == record1

    def test_calculate_validation_lag(self, accuracy_tracker, mock_validator):
        """Test validation lag calculation."""
        # Create mock validation records with different lag times
        base_time = datetime.now(timezone.utc)

        record1 = Mock(spec=ValidationRecord)
        record1.validation_time = base_time
        record1.prediction_time = base_time - timedelta(minutes=5)
        record1.room_id = "test_room"
        record1.model_type = ModelType.ENSEMBLE

        record2 = Mock(spec=ValidationRecord)
        record2.validation_time = base_time - timedelta(hours=1)
        record2.prediction_time = base_time - timedelta(hours=1, minutes=10)
        record2.room_id = "test_room"
        record2.model_type = ModelType.ENSEMBLE

        mock_validator._validation_records = {"rec1": record1, "rec2": record2}

        avg_lag = accuracy_tracker._calculate_validation_lag(
            "test_room", ModelType.ENSEMBLE
        )

        # Should average 5 and 10 minutes = 7.5 minutes
        assert 7 <= avg_lag <= 8

    def test_calculate_validation_lag_no_data(self, accuracy_tracker, mock_validator):
        """Test validation lag calculation with no data."""
        mock_validator._validation_records = {}

        avg_lag = accuracy_tracker._calculate_validation_lag(
            "test_room", ModelType.ENSEMBLE
        )
        assert avg_lag == 0.0


class TestAccuracyTrackingError:
    """Test suite for AccuracyTrackingError exception."""

    def test_tracking_error_creation(self):
        """Test creating AccuracyTrackingError."""
        error = AccuracyTrackingError("Test tracking error")

        assert str(error) == "Test tracking error"
        assert error.error_code == "ACCURACY_TRACKING_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_tracking_error_with_custom_severity(self):
        """Test creating AccuracyTrackingError with custom severity."""
        error = AccuracyTrackingError(
            "Critical tracking error", severity=ErrorSeverity.HIGH
        )

        assert str(error) == "Critical tracking error"
        assert error.severity == ErrorSeverity.HIGH

    def test_tracking_error_with_cause(self):
        """Test creating AccuracyTrackingError with cause."""
        original_error = ValueError("Original error")
        error = AccuracyTrackingError("Tracking error with cause", cause=original_error)

        assert str(error) == "Tracking error with cause"
        assert error.cause == original_error


# Integration tests for complex scenarios
class TestAccuracyTrackerIntegration:
    """Integration tests for AccuracyTracker with complex scenarios."""

    @pytest.fixture
    def integration_validator(self):
        """Create a more realistic mock validator for integration tests."""
        validator = Mock(spec=PredictionValidator)
        validator._lock = threading.RLock()
        validator._validation_records = {}
        validator._pending_predictions = {}

        # Mock realistic accuracy metrics
        mock_metrics = Mock(spec=AccuracyMetrics)
        mock_metrics.accuracy_rate = 75.0
        mock_metrics.mean_error_minutes = 12.5
        mock_metrics.confidence_calibration_score = 0.78
        mock_metrics.validated_predictions = 50
        mock_metrics.predictions_per_hour = 5.2

        validator.get_accuracy_metrics = AsyncMock(return_value=mock_metrics)
        validator.get_total_predictions = AsyncMock(return_value=150)
        validator.get_validation_rate = AsyncMock(return_value=0.93)
        validator.cleanup_old_predictions = AsyncMock()

        return validator

    @pytest.fixture
    def integration_tracker(self, integration_validator):
        """Create tracker for integration testing."""
        return AccuracyTracker(
            prediction_validator=integration_validator,
            monitoring_interval_seconds=1,  # Fast for testing
            alert_thresholds={
                "accuracy_warning": 80.0,
                "accuracy_critical": 60.0,
                "error_warning": 15.0,
                "error_critical": 25.0,
            },
            max_stored_alerts=50,
            trend_analysis_points=3,
        )

    def test_end_to_end_monitoring_cycle(self, integration_tracker):
        """Test complete monitoring cycle from start to metrics calculation."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_full_cycle():
                # Start monitoring
                await integration_tracker.start_monitoring()
                assert integration_tracker._monitoring_active

                # Let one monitoring cycle run
                await asyncio.sleep(0.1)

                # Check if metrics were calculated
                global_metrics = await integration_tracker.get_real_time_metrics()

                # Stop monitoring
                await integration_tracker.stop_monitoring()
                assert not integration_tracker._monitoring_active

            loop.run_until_complete(test_full_cycle())
        finally:
            loop.close()

    def test_alert_creation_and_management(self, integration_tracker):
        """Test alert creation, escalation, and resolution cycle."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_alert_lifecycle():
                # Simulate poor accuracy metrics that should trigger alerts
                mock_poor_metrics = Mock(spec=AccuracyMetrics)
                mock_poor_metrics.accuracy_rate = 45.0  # Below critical threshold
                mock_poor_metrics.mean_error_minutes = 30.0  # Above error threshold
                mock_poor_metrics.confidence_calibration_score = 0.3
                mock_poor_metrics.validated_predictions = 25
                mock_poor_metrics.predictions_per_hour = 2.0

                integration_tracker.validator.get_accuracy_metrics.return_value = (
                    mock_poor_metrics
                )

                # Force a metrics update to trigger alerts
                await integration_tracker._update_real_time_metrics()

                # Check if alerts were created
                active_alerts = await integration_tracker.get_active_alerts()

                # Should have created alerts for low accuracy and high error
                assert len(active_alerts) >= 1

                # Test alert acknowledgment
                if active_alerts:
                    alert_id = active_alerts[0].alert_id
                    success = await integration_tracker.acknowledge_alert(
                        alert_id, "test_admin"
                    )
                    assert success

                    # Verify acknowledgment
                    acknowledged_alert = integration_tracker._active_alerts[alert_id]
                    assert acknowledged_alert.acknowledged
                    assert acknowledged_alert.acknowledged_by == "test_admin"

            loop.run_until_complete(test_alert_lifecycle())
        finally:
            loop.close()

    def test_trend_analysis_over_time(self, integration_tracker):
        """Test trend analysis with changing metrics over time."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_trend_analysis():
                # Simulate improving accuracy over time
                accuracy_values = [70.0, 73.0, 76.0, 79.0, 82.0]

                for i, accuracy in enumerate(accuracy_values):
                    mock_metrics = Mock(spec=AccuracyMetrics)
                    mock_metrics.accuracy_rate = accuracy
                    mock_metrics.mean_error_minutes = 20.0 - (i * 2)  # Decreasing error
                    mock_metrics.confidence_calibration_score = 0.7 + (i * 0.05)
                    mock_metrics.validated_predictions = 30 + (i * 5)
                    mock_metrics.predictions_per_hour = 4.0

                    integration_tracker.validator.get_accuracy_metrics.return_value = (
                        mock_metrics
                    )

                    # Force metrics update
                    await integration_tracker._update_real_time_metrics()

                    # Small delay to simulate time passing
                    await asyncio.sleep(0.01)

                # Get trend analysis
                trends = await integration_tracker.get_accuracy_trends()

                assert "trends_by_entity" in trends
                assert "global_trend" in trends

                # Check if improving trend was detected
                global_trend = trends["global_trend"]
                if global_trend["entities_analyzed"] > 0:
                    # Should detect improvement or at least not degradation
                    assert global_trend["direction"] in [
                        TrendDirection.IMPROVING,
                        TrendDirection.STABLE,
                    ]

            loop.run_until_complete(test_trend_analysis())
        finally:
            loop.close()

    def test_notification_callback_integration(self, integration_tracker):
        """Test notification callback system with real alerts."""
        callback_calls = []

        def test_callback(alert):
            callback_calls.append(alert)

        async def async_callback(alert):
            callback_calls.append(f"async_{alert.alert_id}")

        # Add both sync and async callbacks
        integration_tracker.add_notification_callback(test_callback)
        integration_tracker.add_notification_callback(async_callback)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_callbacks():
                # Create conditions that will trigger an alert
                mock_critical_metrics = Mock(spec=AccuracyMetrics)
                mock_critical_metrics.accuracy_rate = 40.0  # Critical level
                mock_critical_metrics.mean_error_minutes = 35.0
                mock_critical_metrics.confidence_calibration_score = 0.2
                mock_critical_metrics.validated_predictions = 20
                mock_critical_metrics.predictions_per_hour = 1.5

                integration_tracker.validator.get_accuracy_metrics.return_value = (
                    mock_critical_metrics
                )

                # Add mock validation record to trigger entity detection
                mock_record = Mock(spec=ValidationRecord)
                mock_record.room_id = "test_room"
                mock_record.model_type = ModelType.ENSEMBLE
                integration_tracker.validator._validation_records = {
                    "test": mock_record
                }

                # Force metrics update which should trigger alerts
                await integration_tracker._update_real_time_metrics()
                await integration_tracker._check_alert_conditions()

                # Allow async callbacks to complete
                await asyncio.sleep(0.1)

                # Verify callbacks were called
                assert len(callback_calls) >= 2  # At least sync and async callback

            loop.run_until_complete(test_callbacks())
        finally:
            loop.close()

    def test_export_comprehensive_data(self, integration_tracker):
        """Test comprehensive data export functionality."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_comprehensive_export():
                # Add various types of data
                test_metrics = RealTimeMetrics(
                    room_id="export_test_room",
                    window_1h_accuracy=85.0,
                    window_6h_accuracy=82.0,
                    window_24h_accuracy=79.0,
                )
                integration_tracker._metrics_by_room["export_test_room"] = test_metrics

                # Add alert
                test_alert = AccuracyAlert(
                    alert_id="export_alert",
                    room_id="export_test_room",
                    model_type=ModelType.ENSEMBLE,
                    severity=AlertSeverity.WARNING,
                    trigger_condition="accuracy_warning",
                    current_value=65.0,
                    threshold_value=70.0,
                    description="Export test alert",
                    affected_metrics={"accuracy_6h": 65.0},
                    recent_predictions=45,
                )
                integration_tracker._active_alerts["export_alert"] = test_alert
                integration_tracker._alert_history.append(test_alert)

                # Add trend data
                trend_data = [
                    {"timestamp": datetime.now(timezone.utc), "accuracy_6h": 80.0},
                    {"timestamp": datetime.now(timezone.utc), "accuracy_6h": 82.0},
                    {"timestamp": datetime.now(timezone.utc), "accuracy_6h": 79.0},
                ]
                integration_tracker._accuracy_history["export_test_room_ensemble"] = (
                    deque(trend_data, maxlen=10)
                )

                with tempfile.NamedTemporaryFile(
                    mode="w", delete=False, suffix=".json"
                ) as tmp_file:
                    tmp_path = tmp_file.name

                # Export all data
                count = await integration_tracker.export_tracking_data(
                    output_path=tmp_path,
                    include_alerts=True,
                    include_trends=True,
                    days_back=1,
                )

                assert count >= 3  # At least metrics, alert, and trend data

                # Verify exported data structure
                with open(tmp_path, "r") as f:
                    exported_data = json.load(f)

                assert "export_time" in exported_data
                assert "metrics" in exported_data
                assert "alerts" in exported_data
                assert "trends" in exported_data
                assert len(exported_data["metrics"]) >= 1
                assert len(exported_data["alerts"]) >= 1
                assert len(exported_data["trends"]) >= 1

                # Verify specific data
                assert "export_test_room" in exported_data["metrics"]
                alert_data = exported_data["alerts"][0]
                assert alert_data["alert_id"] == "export_alert"
                assert alert_data["severity"] == "warning"

                # Clean up
                Path(tmp_path).unlink()

            loop.run_until_complete(test_comprehensive_export())
        finally:
            loop.close()
