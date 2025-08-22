"""
Integration tests for AccuracyTracker system.

Tests real component interactions, end-to-end workflows, and system integration scenarios.
Unit tests are in tests/unit/test_adaptation/test_tracker.py.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import tempfile
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

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


class TestAccuracyTrackerIntegration:
    """Integration tests for AccuracyTracker with realistic scenarios."""

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
            },
            max_stored_alerts=20,
            trend_analysis_points=3,
        )

    @pytest.mark.asyncio
    async def test_end_to_end_monitoring_cycle(self, integration_tracker):
        """Test complete monitoring cycle from start to stop."""
        # Start monitoring
        await integration_tracker.start_monitoring()
        assert integration_tracker._monitoring_active

        # Let monitoring run briefly
        await asyncio.sleep(0.01)

        # Stop monitoring
        await integration_tracker.stop_monitoring()
        assert not integration_tracker._monitoring_active

    async def test_full_monitoring_cycle_with_metrics(self, integration_tracker):
        """Test complete monitoring cycle with metrics and alerts."""
        # Start monitoring
        await integration_tracker.start_monitoring()

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Check that monitoring is active
        assert integration_tracker._monitoring_active

        # Get some metrics
        metrics = await integration_tracker.get_real_time_metrics(room_id="test_room")
        assert metrics is not None
        assert metrics.room_id == "test_room"

        # Stop monitoring
        await integration_tracker.stop_monitoring()
        assert not integration_tracker._monitoring_active

    async def test_alert_lifecycle_integration(self, integration_tracker):
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

        # Acknowledge the alert
        alert_id = alerts[0].alert_id
        result = await integration_tracker.acknowledge_alert(
            alert_id, acknowledged_by="test_operator"
        )
        assert result is True

        # Verify acknowledgment
        updated_alerts = await integration_tracker.get_active_alerts()
        acknowledged_alert = next(
            (a for a in updated_alerts if a.alert_id == alert_id), None
        )
        assert acknowledged_alert is not None
        assert acknowledged_alert.acknowledged
        assert acknowledged_alert.acknowledged_by == "test_operator"

    async def test_multi_room_tracking_integration(self, integration_validator):
        """Test tracking multiple rooms simultaneously."""
        tracker = AccuracyTracker(
            prediction_validator=integration_validator,
            monitoring_interval_seconds=5,
            alert_thresholds={"accuracy_warning": 70.0, "accuracy_critical": 50.0},
        )

        # Start monitoring
        await tracker.start_monitoring()

        # Simulate metrics for multiple rooms
        rooms = ["living_room", "bedroom", "kitchen", "office"]
        for i, room in enumerate(rooms):
            metrics = RealTimeMetrics(
                room_id=room,
                window_6h_accuracy=80.0 - (i * 5),  # Varying performance
                window_6h_predictions=15 + i,
            )
            with tracker._lock:
                tracker._metrics_by_room[room] = metrics

        # Get metrics for all rooms
        all_metrics = {}
        for room in rooms:
            room_metrics = await tracker.get_real_time_metrics(room_id=room)
            all_metrics[room] = room_metrics

        # Verify each room has metrics
        assert len(all_metrics) == 4
        for room in rooms:
            assert room in all_metrics
            assert all_metrics[room].room_id == room

        # Get global trends
        trends = await tracker.get_accuracy_trends(hours=6)
        assert isinstance(trends, dict)

        await tracker.stop_monitoring()

    async def test_alert_escalation_integration(self, integration_tracker):
        """Test alert escalation in realistic scenario."""
        # Create an old warning alert that should escalate
        old_time = datetime.now(timezone.utc) - timedelta(minutes=90)
        warning_alert = AccuracyAlert(
            alert_id="escalation_test",
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Accuracy degraded",
            timestamp=old_time,
        )

        # Add to tracker
        integration_tracker._active_alerts = [warning_alert]

        # Check escalation conditions
        await integration_tracker._check_escalation_conditions()

        # Alert should have escalated
        alerts = await integration_tracker.get_active_alerts()
        escalated_alert = next(
            (a for a in alerts if a.alert_id == "escalation_test"), None
        )

        assert escalated_alert is not None
        assert escalated_alert.escalated
        assert escalated_alert.severity == AlertSeverity.CRITICAL

    async def test_trend_analysis_integration(self, integration_tracker):
        """Test trend analysis with realistic data patterns."""
        # Simulate degrading performance over time
        base_time = datetime.now(timezone.utc)
        room_id = "trend_test_room"

        # Create historical data showing degrading trend
        historical_points = []
        for i in range(10):
            timestamp = base_time - timedelta(hours=i)
            accuracy = 90.0 - (i * 2)  # Degrading from 90% to 72%

            metrics = RealTimeMetrics(
                room_id=room_id,
                window_6h_accuracy=accuracy,
                last_updated=timestamp,
            )
            historical_points.append((timestamp, metrics))

        # Store historical data in tracker
        integration_tracker._historical_metrics = historical_points
        integration_tracker._metrics_by_room[room_id] = historical_points[0][1]

        # Analyze trends
        trends = await integration_tracker.get_accuracy_trends(
            room_id=room_id, hours=24
        )

        assert room_id in trends
        room_trend = trends[room_id]
        assert room_trend["trend_direction"] == TrendDirection.DEGRADING.value
        assert room_trend["slope"] < 0  # Negative slope for degrading

    async def test_export_import_integration(self, integration_tracker):
        """Test data export and import functionality."""
        # Set up test data
        test_alert = AccuracyAlert(
            alert_id="export_test_alert",
            room_id="export_room",
            severity=AlertSeverity.WARNING,
            message="Export test alert",
        )

        test_metrics = RealTimeMetrics(
            room_id="export_room",
            window_6h_accuracy=75.0,
            window_24h_accuracy=78.0,
        )

        integration_tracker._active_alerts = [test_alert]
        integration_tracker._metrics_by_room["export_room"] = test_metrics

        # Export data
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            export_path = f.name

        success = await integration_tracker.export_tracking_data(export_path)
        assert success

        # Verify exported data
        with open(export_path, "r") as f:
            exported_data = json.load(f)

        assert "alerts" in exported_data
        assert "metrics" in exported_data
        assert "export_metadata" in exported_data

        # Verify alert data
        assert len(exported_data["alerts"]) == 1
        alert_data = exported_data["alerts"][0]
        assert alert_data["alert_id"] == "export_test_alert"
        assert alert_data["room_id"] == "export_room"

        # Verify metrics data
        assert len(exported_data["metrics"]) == 1
        metrics_data = exported_data["metrics"][0]
        assert metrics_data["room_id"] == "export_room"
        assert metrics_data["window_6h_accuracy"] == 75.0

        # Clean up
        import os

        os.unlink(export_path)

    async def test_concurrent_operations_integration(self, integration_tracker):
        """Test concurrent operations on tracker."""
        # Start monitoring
        await integration_tracker.start_monitoring()

        # Define concurrent operations
        async def add_metrics():
            for i in range(5):
                metrics = RealTimeMetrics(
                    room_id=f"room_{i}",
                    window_6h_accuracy=80.0 + i,
                )
                with integration_tracker._lock:
                    integration_tracker._metrics_by_room[f"room_{i}"] = metrics
                await asyncio.sleep(0.01)

        async def get_metrics():
            for i in range(5):
                await integration_tracker.get_real_time_metrics(room_id=f"room_{i}")
                await asyncio.sleep(0.01)

        async def check_alerts():
            for _ in range(3):
                await integration_tracker.get_active_alerts()
                await asyncio.sleep(0.02)

        # Run concurrent operations
        await asyncio.gather(
            add_metrics(), get_metrics(), check_alerts(), return_exceptions=True
        )

        # Verify system remains stable
        final_metrics = await integration_tracker.get_real_time_metrics()
        assert final_metrics is not None

        await integration_tracker.stop_monitoring()

    async def test_realistic_accuracy_degradation_scenario(self, integration_validator):
        """Test realistic scenario of accuracy degradation and recovery."""
        # Configure validator to return degrading then improving metrics
        metrics_sequence = [
            # Initial good performance
            AccuracyMetrics(
                total_predictions=100,
                validated_predictions=95,
                accuracy_rate=85.0,
                mean_error_minutes=8.0,
                predictions_per_hour=4.0,
                confidence_calibration_score=0.88,
            ),
            # Degrading performance
            AccuracyMetrics(
                total_predictions=120,
                validated_predictions=110,
                accuracy_rate=65.0,
                mean_error_minutes=18.0,
                predictions_per_hour=4.2,
                confidence_calibration_score=0.72,
            ),
            # Recovery
            AccuracyMetrics(
                total_predictions=140,
                validated_predictions=130,
                accuracy_rate=82.0,
                mean_error_minutes=9.5,
                predictions_per_hour=4.1,
                confidence_calibration_score=0.85,
            ),
        ]

        call_count = 0

        async def mock_get_metrics(*args, **kwargs):
            nonlocal call_count
            result = metrics_sequence[min(call_count, len(metrics_sequence) - 1)]
            call_count += 1
            return result

        integration_validator.get_accuracy_metrics = mock_get_metrics

        tracker = AccuracyTracker(
            prediction_validator=integration_validator,
            monitoring_interval_seconds=1,
            alert_thresholds={
                "accuracy_warning": 75.0,
                "accuracy_critical": 60.0,
            },
        )

        await tracker.start_monitoring()

        # Phase 1: Good performance (no alerts expected)
        metrics1 = await tracker.get_real_time_metrics(room_id="test_room")
        assert metrics1.window_24h_accuracy == 85.0

        alerts = await tracker.get_active_alerts()
        # Should have no critical alerts initially

        # Phase 2: Degraded performance (should trigger alerts)
        await asyncio.sleep(0.1)  # Let monitoring cycle run
        metrics2 = await tracker.get_real_time_metrics(room_id="test_room")
        assert metrics2.window_24h_accuracy == 65.0

        # Check for alerts after degradation
        await tracker._check_alert_conditions()
        alerts = await tracker.get_active_alerts()
        # Should have alerts due to degraded performance

        # Phase 3: Recovery (alerts should be resolved)
        await asyncio.sleep(0.1)
        metrics3 = await tracker.get_real_time_metrics(room_id="test_room")
        assert metrics3.window_24h_accuracy == 82.0

        await tracker.stop_monitoring()

    async def test_notification_callback_integration(self, integration_tracker):
        """Test notification callback system integration."""
        received_notifications = []

        def notification_callback(alert: AccuracyAlert):
            received_notifications.append(
                {
                    "alert_id": alert.alert_id,
                    "severity": alert.severity,
                    "room_id": alert.room_id,
                    "timestamp": alert.timestamp,
                }
            )

        # Add notification callback
        integration_tracker.add_notification_callback(notification_callback)

        # Create condition that triggers alert
        poor_metrics = RealTimeMetrics(
            room_id="notification_room",
            window_6h_accuracy=55.0,  # Below critical threshold
            window_6h_predictions=10,
        )

        with integration_tracker._lock:
            integration_tracker._metrics_by_room["notification_room"] = poor_metrics

        # Trigger alert check
        await integration_tracker._check_alert_conditions()

        # Should have received notification
        assert len(received_notifications) > 0
        notification = received_notifications[0]
        assert notification["room_id"] == "notification_room"
        assert notification["severity"] == AlertSeverity.CRITICAL

    async def test_memory_management_integration(self, integration_tracker):
        """Test memory management with large datasets."""
        # Configure tracker with limits
        integration_tracker._max_stored_alerts = 5
        integration_tracker._max_historical_points = 10

        # Generate many alerts (more than limit)
        for i in range(10):
            alert = AccuracyAlert(
                alert_id=f"memory_test_{i}",
                room_id="memory_room",
                severity=AlertSeverity.WARNING,
                message=f"Alert {i}",
                timestamp=datetime.now(timezone.utc) - timedelta(minutes=i),
            )
            integration_tracker._active_alerts.append(alert)

        # Trigger cleanup
        await integration_tracker._cleanup_old_data()

        # Should respect limits
        assert (
            len(integration_tracker._active_alerts)
            <= integration_tracker._max_stored_alerts
        )

        # Generate historical metrics (more than limit)
        base_time = datetime.now(timezone.utc)
        for i in range(15):
            timestamp = base_time - timedelta(hours=i)
            metrics = RealTimeMetrics(
                room_id="memory_room",
                window_6h_accuracy=75.0 + i,
                last_updated=timestamp,
            )
            if not hasattr(integration_tracker, "_historical_metrics"):
                integration_tracker._historical_metrics = []
            integration_tracker._historical_metrics.append((timestamp, metrics))

        # Trigger cleanup
        await integration_tracker._cleanup_old_data()

        # Should respect limits
        if hasattr(integration_tracker, "_historical_metrics"):
            assert (
                len(integration_tracker._historical_metrics)
                <= integration_tracker._max_historical_points
            )


class TestRealWorldScenarios:
    """Integration tests simulating real-world usage scenarios."""

    @pytest.fixture
    def production_like_tracker(self):
        """Create tracker with production-like configuration."""
        validator = Mock(spec=PredictionValidator)
        validator._lock = threading.RLock()

        # Production-like metrics
        production_metrics = AccuracyMetrics(
            total_predictions=500,
            validated_predictions=475,
            accuracy_rate=78.5,
            mean_error_minutes=13.2,
            median_error_minutes=10.8,
            std_error_minutes=8.4,
            predictions_per_hour=6.8,
            confidence_calibration_score=0.81,
            accuracy_by_level={"excellent": 120, "good": 200, "fair": 155},
            error_distribution={"0-5": 140, "5-15": 180, "15-30": 110, "30+": 45},
            trend_analysis={"slope": -0.3, "r_squared": 0.65},
        )

        validator.get_accuracy_metrics = AsyncMock(return_value=production_metrics)

        return AccuracyTracker(
            prediction_validator=validator,
            monitoring_interval_seconds=300,  # 5 minutes - production interval
            alert_thresholds={
                "accuracy_warning": 75.0,
                "accuracy_critical": 60.0,
                "error_warning": 20.0,
                "error_critical": 30.0,
            },
            max_stored_alerts=100,
            trend_analysis_points=24,  # 24 data points for trend analysis
        )

    async def test_24_hour_monitoring_simulation(self, production_like_tracker):
        """Simulate 24-hour monitoring with realistic intervals."""
        await production_like_tracker.start_monitoring()

        # Simulate periodic metric collection (condensed time)
        for hour in range(24):
            # Get metrics for different rooms
            rooms = ["living_room", "bedroom", "kitchen", "office"]
            for room in rooms:
                metrics = await production_like_tracker.get_real_time_metrics(
                    room_id=room
                )
                assert metrics is not None
                assert metrics.room_id == room

            # Brief pause to simulate time passage
            await asyncio.sleep(0.01)

        # Check final system state
        all_alerts = await production_like_tracker.get_active_alerts()
        global_metrics = await production_like_tracker.get_real_time_metrics()

        assert global_metrics is not None
        assert isinstance(all_alerts, list)

        await production_like_tracker.stop_monitoring()

    async def test_high_frequency_prediction_load(self, production_like_tracker):
        """Test system under high prediction load."""
        await production_like_tracker.start_monitoring()

        # Simulate high-frequency metric requests
        tasks = []
        for i in range(50):  # 50 concurrent requests
            task = asyncio.create_task(
                production_like_tracker.get_real_time_metrics(
                    room_id=f"room_{i % 10}"  # 10 different rooms
                )
            )
            tasks.append(task)

        # Wait for all requests to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Verify no exceptions occurred
        for result in results:
            assert not isinstance(result, Exception)
            assert isinstance(result, RealTimeMetrics)

        await production_like_tracker.stop_monitoring()

    async def test_alert_storm_handling(self, production_like_tracker):
        """Test handling of many simultaneous alerts."""
        # Create multiple rooms with poor performance
        rooms_with_issues = [f"problem_room_{i}" for i in range(20)]

        for room in rooms_with_issues:
            poor_metrics = RealTimeMetrics(
                room_id=room,
                window_6h_accuracy=45.0,  # Very poor performance
                window_6h_predictions=10,
                accuracy_trend=TrendDirection.DEGRADING,
            )

            with production_like_tracker._lock:
                production_like_tracker._metrics_by_room[room] = poor_metrics

        # Trigger alert checking
        await production_like_tracker._check_alert_conditions()

        # Get all alerts
        all_alerts = await production_like_tracker.get_active_alerts()

        # Should have created alerts, but system should remain stable
        assert isinstance(all_alerts, list)

        # Verify alerts can be filtered and managed
        critical_alerts = await production_like_tracker.get_active_alerts(
            min_severity=AlertSeverity.CRITICAL
        )
        assert len(critical_alerts) <= len(all_alerts)

    async def test_graceful_degradation(self, production_like_tracker):
        """Test system behavior when components fail."""
        await production_like_tracker.start_monitoring()

        # Simulate validator failure
        production_like_tracker._prediction_validator.get_accuracy_metrics = AsyncMock(
            side_effect=Exception("Database connection lost")
        )

        # System should handle errors gracefully
        with pytest.raises(AccuracyTrackingError):
            await production_like_tracker.get_real_time_metrics(room_id="test_room")

        # System should still be able to return cached data or defaults
        alerts = await production_like_tracker.get_active_alerts()
        assert isinstance(alerts, list)  # Should not crash

        await production_like_tracker.stop_monitoring()
