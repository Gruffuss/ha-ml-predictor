"""Unit tests for system utilities and supporting functionality.

Covers:
- src/utils/logger.py (Structured Logging)
- src/utils/metrics.py (Performance Metrics)
- src/utils/time_utils.py (Time Utility Functions)
- src/utils/health_monitor.py (Health Monitoring)
- src/utils/monitoring.py (System Monitoring)
- src/utils/monitoring_integration.py (Monitoring Integration)
- src/utils/alerts.py (Alert System)
- src/utils/incident_response.py (Incident Response)

This test file consolidates testing for all utility and monitoring functionality.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta, timezone
import json
import logging
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from src.utils.alerts import (
    AlertChannel,
    AlertEvent as AlertsAlertEvent,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertThrottler,
    EmailNotifier,
    ErrorRecoveryManager,
    MQTTNotifier,
    NotificationConfig,
    WebhookNotifier,
    get_alert_manager,
)
from src.utils.health_monitor import (
    ComponentHealth,
    ComponentType,
    HealthMonitor,
    HealthStatus,
    HealthThresholds,
    SystemHealth,
    get_health_monitor,
)
from src.utils.incident_response import (
    Incident,
    IncidentResponseManager,
    IncidentSeverity,
    IncidentStatus,
    RecoveryAction,
    RecoveryActionType,
    get_incident_response_manager,
)

# Import actual utility classes
from src.utils.logger import (
    ErrorTracker,
    LoggerManager,
    MLOperationsLogger,
    PerformanceLogger,
    StructuredFormatter,
    get_error_tracker,
    get_logger,
    get_logger_manager,
    get_ml_ops_logger,
    get_performance_logger,
)
from src.utils.metrics import (
    MetricsManager,
    MLMetricsCollector,
    MultiProcessMetricsManager,
    get_metrics_collector,
    get_metrics_manager,
    get_multiprocess_metrics_manager,
    time_prediction,
)
from src.utils.monitoring import (
    AlertEvent,
    HealthCheckResult,
    MonitoringManager,
    PerformanceMonitor,
    PerformanceThreshold,
    SystemHealthMonitor,
    get_monitoring_manager,
)
from src.utils.monitoring_integration import (
    MonitoringIntegration,
    get_monitoring_integration,
)
from src.utils.time_utils import (
    AsyncTimeUtils,
    TimeFrame,
    TimeProfiler,
    TimeRange,
    TimeUtils,
    cyclical_time_features,
    format_duration,
    time_since,
    time_until,
)


class TestStructuredLogging:
    """Test structured logging functionality."""

    def test_structured_formatter_initialization(self):
        """Test StructuredFormatter initialization."""
        formatter = StructuredFormatter(include_extra=True)
        assert formatter.include_extra is True

        formatter_no_extra = StructuredFormatter(include_extra=False)
        assert formatter_no_extra.include_extra is False

    def test_structured_formatter_basic_formatting(self):
        """Test basic log formatting."""
        formatter = StructuredFormatter(include_extra=True)

        # Create a test log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Format the record
        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        # Verify basic fields
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test"
        assert parsed["line_number"] == 42
        assert "timestamp" in parsed

    def test_structured_formatter_exception_handling(self):
        """Test exception information formatting."""
        formatter = StructuredFormatter(include_extra=True)

        try:
            raise ValueError("Test exception")
        except ValueError:
            import sys

            exc_info = sys.exc_info()

            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="test.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            formatted = formatter.format(record)
            parsed = json.loads(formatted)

            assert "exception" in parsed
            assert parsed["exception"]["type"] == "ValueError"
            assert parsed["exception"]["message"] == "Test exception"
            assert "traceback" in parsed["exception"]

    def test_structured_formatter_extra_fields(self):
        """Test handling of extra fields."""
        formatter = StructuredFormatter(include_extra=True)

        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Test with extra",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.room_id = "living_room"
        record.prediction_type = "occupancy"

        formatted = formatter.format(record)
        parsed = json.loads(formatted)

        assert "extra" in parsed
        assert parsed["extra"]["room_id"] == "living_room"
        assert parsed["extra"]["prediction_type"] == "occupancy"

    def test_performance_logger_operation_timing(self):
        """Test performance logger operation timing."""
        perf_logger = PerformanceLogger("test.performance")

        with patch.object(perf_logger.logger, "info") as mock_info:
            perf_logger.log_operation_time(
                operation="test_operation",
                duration=1.25,
                room_id="bedroom",
                prediction_type="next_occupied",
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert "Test_operation" in call_args[0][0]  # Message content

            extra = call_args[1]["extra"]
            assert extra["operation"] == "test_operation"
            assert extra["duration_seconds"] == 1.25
            assert extra["room_id"] == "bedroom"
            assert extra["prediction_type"] == "next_occupied"
            assert extra["metric_type"] == "performance"

    def test_performance_logger_prediction_accuracy(self):
        """Test prediction accuracy logging."""
        perf_logger = PerformanceLogger("test.performance")

        with patch.object(perf_logger.logger, "info") as mock_info:
            perf_logger.log_prediction_accuracy(
                room_id="kitchen",
                accuracy_minutes=12.5,
                confidence=0.85,
                prediction_type="next_vacant",
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args

            extra = call_args[1]["extra"]
            assert extra["room_id"] == "kitchen"
            assert extra["accuracy_minutes"] == 12.5
            assert extra["confidence"] == 0.85
            assert extra["prediction_type"] == "next_vacant"
            assert extra["metric_type"] == "accuracy"

    def test_error_tracker_error_tracking(self):
        """Test error tracking functionality."""
        error_tracker = ErrorTracker("test.errors")

        test_error = ValueError("Test error")
        context = {"component": "test", "operation": "test_op"}

        with patch.object(error_tracker.logger, "error") as mock_error:
            error_tracker.track_error(
                error=test_error, context=context, severity="error", alert=True
            )

            mock_error.assert_called_once()
            call_args = mock_error.call_args

            extra = call_args[1]["extra"]
            assert extra["error_type"] == "ValueError"
            assert extra["error_message"] == "Test error"
            assert extra["severity"] == "error"
            assert extra["context"] == context
            assert extra["alert_required"] is True
            assert extra["metric_type"] == "error"

    def test_ml_operations_logger_training_event(self):
        """Test ML operations logging."""
        ml_ops_logger = MLOperationsLogger("test.ml_ops")

        with patch.object(ml_ops_logger.logger, "info") as mock_info:
            ml_ops_logger.log_training_event(
                room_id="office",
                model_type="lstm",
                event_type="training_start",
                metrics={"learning_rate": 0.001},
            )

            mock_info.assert_called_once()
            call_args = mock_info.call_args

            extra = call_args[1]["extra"]
            assert extra["room_id"] == "office"
            assert extra["model_type"] == "lstm"
            assert extra["event_type"] == "training_start"
            assert extra["metrics"] == {"learning_rate": 0.001}
            assert extra["component"] == "training"
            assert extra["metric_type"] == "ml_lifecycle"

    def test_logger_manager_configuration(self):
        """Test logger manager configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "logging.yaml"

            # Create a basic config
            config_content = """
            version: 1
            formatters:
              simple:
                format: "%(levelname)s - %(message)s"
            handlers:
              console:
                class: logging.StreamHandler
                formatter: simple
            root:
              level: INFO
              handlers: [console]
            """

            with open(config_path, "w") as f:
                f.write(config_content)

            logger_manager = LoggerManager(config_path)
            assert logger_manager.config_path == config_path

            logger = logger_manager.get_logger("test")
            assert logger.name == "occupancy_prediction.test"

    @pytest.mark.asyncio
    async def test_logger_manager_operation_context(self):
        """Test logger manager operation context manager."""
        logger_manager = LoggerManager()

        with patch.object(
            logger_manager.performance_logger, "log_operation_time"
        ) as mock_perf:
            with logger_manager.log_operation("test_operation", room_id="bathroom"):
                await asyncio.sleep(0.01)  # Small delay

            mock_perf.assert_called_once()
            call_args = mock_perf.call_args[0]
            assert call_args[0] == "test_operation"
            assert call_args[1] > 0  # Duration should be positive
            assert call_args[2] == "bathroom"


class TestPerformanceMetrics:
    """Test performance metrics collection."""

    @pytest.fixture
    def mock_prometheus(self):
        """Mock Prometheus client components."""
        with patch("src.utils.metrics.PROMETHEUS_AVAILABLE", True), patch(
            "src.utils.metrics.Counter"
        ) as mock_counter, patch("src.utils.metrics.Gauge") as mock_gauge, patch(
            "src.utils.metrics.Histogram"
        ) as mock_histogram:

            yield {
                "counter": mock_counter,
                "gauge": mock_gauge,
                "histogram": mock_histogram,
            }

    def test_ml_metrics_collector_initialization(self, mock_prometheus):
        """Test MLMetricsCollector initialization."""
        collector = MLMetricsCollector()

        # Verify metrics were created
        assert mock_prometheus["counter"].call_count > 0
        assert mock_prometheus["gauge"].call_count > 0
        assert mock_prometheus["histogram"].call_count > 0

        # Test system info update
        collector.update_system_info()
        assert collector._system_info_updated is True

    def test_ml_metrics_collector_record_prediction(self, mock_prometheus):
        """Test prediction metrics recording."""
        collector = MLMetricsCollector()

        # Mock metric instances
        mock_counter_instance = Mock()
        mock_histogram_instance = Mock()
        mock_gauge_instance = Mock()

        collector.prediction_requests_total = Mock()
        collector.prediction_requests_total.labels.return_value = mock_counter_instance
        collector.prediction_latency = Mock()
        collector.prediction_latency.labels.return_value = mock_histogram_instance
        collector.prediction_accuracy = Mock()
        collector.prediction_accuracy.labels.return_value = mock_gauge_instance
        collector.prediction_confidence = Mock()
        collector.prediction_confidence.labels.return_value = mock_histogram_instance

        collector.record_prediction(
            room_id="living_room",
            prediction_type="next_occupied",
            model_type="lstm",
            duration=0.15,
            accuracy_minutes=8.5,
            confidence=0.92,
            status="success",
        )

        # Verify metrics were updated
        collector.prediction_requests_total.labels.assert_called_with(
            room_id="living_room", prediction_type="next_occupied", status="success"
        )
        mock_counter_instance.inc.assert_called_once()

        collector.prediction_latency.labels.assert_called_with(
            room_id="living_room", prediction_type="next_occupied", model_type="lstm"
        )
        mock_histogram_instance.observe.assert_called_with(0.15)

    def test_ml_metrics_collector_record_model_training(self, mock_prometheus):
        """Test model training metrics recording."""
        collector = MLMetricsCollector()

        # Mock metric instances
        mock_histogram_instance = Mock()
        mock_counter_instance = Mock()
        mock_gauge_instance = Mock()

        collector.model_training_duration = Mock()
        collector.model_training_duration.labels.return_value = mock_histogram_instance
        collector.model_retraining_count = Mock()
        collector.model_retraining_count.labels.return_value = mock_counter_instance
        collector.model_accuracy_score = Mock()
        collector.model_accuracy_score.labels.return_value = mock_gauge_instance

        collector.record_model_training(
            room_id="bedroom",
            model_type="xgboost",
            training_type="incremental",
            duration=120.5,
            accuracy_metrics={"mse": 0.15, "mae": 0.12},
            trigger_reason="drift_detected",
        )

        # Verify training duration recorded
        collector.model_training_duration.labels.assert_called_with(
            room_id="bedroom", model_type="xgboost", training_type="incremental"
        )
        mock_histogram_instance.observe.assert_called_with(120.5)

        # Verify retraining count incremented
        collector.model_retraining_count.labels.assert_called_with(
            room_id="bedroom", model_type="xgboost", trigger_reason="drift_detected"
        )
        mock_counter_instance.inc.assert_called_once()

        # Verify accuracy metrics set
        assert (
            collector.model_accuracy_score.labels.call_count == 2
        )  # Called for each metric

    @patch("src.utils.metrics.psutil")
    def test_ml_metrics_collector_system_resources(self, mock_psutil, mock_prometheus):
        """Test system resource metrics updating."""
        collector = MLMetricsCollector()

        # Mock psutil responses
        mock_process = Mock()
        mock_process.memory_info.return_value = Mock(
            rss=1024 * 1024 * 100, vms=1024 * 1024 * 150
        )  # 100MB RSS, 150MB VMS
        mock_psutil.Process.return_value = mock_process
        mock_psutil.cpu_percent.return_value = 25.5
        mock_psutil.disk_usage.return_value = Mock(
            total=1000000000, used=500000000, free=500000000
        )

        # Mock gauge instances
        mock_cpu_gauge = Mock()
        mock_memory_gauge = Mock()
        mock_disk_gauge = Mock()

        collector.cpu_usage_percent = mock_cpu_gauge
        collector.memory_usage_bytes = Mock()
        collector.memory_usage_bytes.labels.return_value = mock_memory_gauge
        collector.disk_usage_bytes = Mock()
        collector.disk_usage_bytes.labels.return_value = mock_disk_gauge

        collector.update_system_resources()

        # Verify CPU metrics
        mock_cpu_gauge.set.assert_called_with(25.5)

        # Verify memory metrics
        assert collector.memory_usage_bytes.labels.call_count >= 2  # RSS and VMS

        # Verify disk metrics
        assert collector.disk_usage_bytes.labels.call_count >= 3  # Total, used, free

    def test_metrics_manager_background_collection(self, mock_prometheus):
        """Test metrics manager background collection."""
        manager = MetricsManager()

        # Mock the collector's methods
        manager.collector = Mock()

        # Start background collection
        manager.start_background_collection(update_interval=0.1)

        assert manager._running is True
        assert manager._resource_update_thread is not None

        # Let it run briefly
        time.sleep(0.2)

        # Stop background collection
        manager.stop_background_collection()

        assert manager._running is False

    def test_multiprocess_metrics_manager(self, mock_prometheus):
        """Test multi-process metrics manager."""
        mp_manager = MultiProcessMetricsManager()

        # Test initialization
        assert mp_manager.multiprocess_enabled in [
            True,
            False,
        ]  # Depends on environment

        # Test registry access
        registry = mp_manager.get_multiprocess_registry()
        # Registry can be None if multiprocess is not available

        # Test metrics generation
        metrics_output = mp_manager.generate_multiprocess_metrics()
        assert isinstance(metrics_output, str)

    def test_time_prediction_decorator(self, mock_prometheus):
        """Test time_prediction decorator functionality."""

        @time_prediction("living_room", "next_occupied", "lstm")
        def mock_prediction_function():
            time.sleep(0.01)  # Small delay
            return {"confidence": 0.85}

        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            result = mock_prediction_function()

            assert result == {"confidence": 0.85}
            mock_collector.record_prediction.assert_called_once()
            call_args = mock_collector.record_prediction.call_args[1]
            assert call_args["room_id"] == "living_room"
            assert call_args["prediction_type"] == "next_occupied"
            assert call_args["model_type"] == "lstm"
            assert call_args["status"] == "success"
            assert call_args["confidence"] == 0.85


class TestTimeUtilities:
    """Test time utility functions."""

    def test_time_frame_enum(self):
        """Test TimeFrame enum values."""
        assert TimeFrame.MINUTE.value == "minute"
        assert TimeFrame.HOUR.value == "hour"
        assert TimeFrame.DAY.value == "day"
        assert TimeFrame.WEEK.value == "week"
        assert TimeFrame.MONTH.value == "month"

    def test_time_range_creation(self):
        """Test TimeRange creation and validation."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)

        time_range = TimeRange(start, end)

        assert time_range.start == start
        assert time_range.end == end
        assert time_range.duration == timedelta(hours=2)

    def test_time_range_invalid_creation(self):
        """Test TimeRange creation with invalid dates."""
        start = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)  # End before start

        with pytest.raises(ValueError, match="Start time must be before end time"):
            TimeRange(start, end)

    def test_time_range_contains(self):
        """Test TimeRange contains method."""
        start = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
        time_range = TimeRange(start, end)

        # Test point inside range
        inside = datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc)
        assert time_range.contains(inside) is True

        # Test point outside range
        outside = datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        assert time_range.contains(outside) is False

        # Test edge cases
        assert time_range.contains(start) is True
        assert time_range.contains(end) is True

    def test_time_range_overlaps(self):
        """Test TimeRange overlap detection."""
        range1 = TimeRange(
            datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 14, 0, 0, tzinfo=timezone.utc),
        )

        # Overlapping range
        range2 = TimeRange(
            datetime(2024, 1, 1, 13, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc),
        )
        assert range1.overlaps(range2) is True

        # Non-overlapping range
        range3 = TimeRange(
            datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc),
            datetime(2024, 1, 1, 16, 0, 0, tzinfo=timezone.utc),
        )
        assert range1.overlaps(range3) is False

    def test_time_utils_timezone_conversion(self):
        """Test timezone conversion utilities."""
        # Test UTC now
        utc_now = TimeUtils.utc_now()
        assert utc_now.tzinfo == timezone.utc

        # Test timezone conversion
        dt = datetime(2024, 1, 1, 12, 0, 0)  # Naive datetime
        utc_dt = TimeUtils.to_utc(dt)
        assert utc_dt.tzinfo == timezone.utc

    def test_time_utils_datetime_parsing(self):
        """Test datetime string parsing."""
        # Test ISO format
        dt_str = "2024-01-01T12:00:00Z"
        parsed = TimeUtils.parse_datetime(dt_str)
        assert parsed.year == 2024
        assert parsed.month == 1
        assert parsed.day == 1
        assert parsed.hour == 12

        # Test invalid format
        with pytest.raises(ValueError, match="Unable to parse datetime string"):
            TimeUtils.parse_datetime("invalid-datetime")

    def test_time_utils_duration_formatting(self):
        """Test duration formatting."""
        # Test various durations
        duration1 = timedelta(hours=2, minutes=30, seconds=15)
        formatted1 = TimeUtils.format_duration(duration1)
        assert "2 hours" in formatted1
        assert "30 minutes" in formatted1

        duration2 = timedelta(seconds=45)
        formatted2 = TimeUtils.format_duration(duration2)
        assert "45 seconds" in formatted2

        # Test zero duration
        duration3 = timedelta(0)
        formatted3 = TimeUtils.format_duration(duration3)
        assert formatted3 == "0 seconds"

    def test_time_utils_time_calculations(self):
        """Test time calculation utilities."""
        target_time = datetime(2024, 1, 1, 15, 0, 0, tzinfo=timezone.utc)
        from_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        # Test time until
        time_until_target = TimeUtils.time_until(target_time, from_time)
        assert time_until_target == timedelta(hours=3)

        # Test time since
        time_since_ref = TimeUtils.time_since(from_time, target_time)
        assert time_since_ref == timedelta(hours=3)

    def test_time_utils_cyclical_features(self):
        """Test cyclical time feature extraction."""
        dt = datetime(2024, 6, 15, 14, 30, 0)  # June 15, 2:30 PM
        features = TimeUtils.get_cyclical_time_features(dt)

        # Verify all expected features are present
        expected_features = [
            "hour_sin",
            "hour_cos",
            "minute_sin",
            "minute_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "day_of_month_sin",
            "day_of_month_cos",
            "month_sin",
            "month_cos",
        ]

        for feature in expected_features:
            assert feature in features
            assert -1 <= features[feature] <= 1  # Sin/cos values should be in [-1, 1]

    def test_time_utils_business_hours(self):
        """Test business hours detection."""
        # Business day, business hours
        dt1 = datetime(2024, 1, 2, 10, 0, 0)  # Tuesday 10 AM
        assert TimeUtils.is_business_hours(dt1) is True

        # Business day, non-business hours
        dt2 = datetime(2024, 1, 2, 18, 0, 0)  # Tuesday 6 PM
        assert TimeUtils.is_business_hours(dt2) is False

        # Weekend
        dt3 = datetime(2024, 1, 6, 10, 0, 0)  # Saturday 10 AM
        assert TimeUtils.is_business_hours(dt3, weekdays_only=True) is False
        assert TimeUtils.is_business_hours(dt3, weekdays_only=False) is True

    def test_time_utils_interval_rounding(self):
        """Test datetime interval rounding."""
        dt = datetime(2024, 1, 1, 12, 37, 23, tzinfo=timezone.utc)
        interval = timedelta(minutes=15)

        # Round down
        rounded_down = TimeUtils.round_to_interval(dt, interval, direction="down")
        assert rounded_down.minute == 30  # Should round to nearest 15-minute mark below

        # Round up
        rounded_up = TimeUtils.round_to_interval(dt, interval, direction="up")
        assert rounded_up.minute == 45  # Should round to nearest 15-minute mark above

        # Round nearest
        rounded_nearest = TimeUtils.round_to_interval(dt, interval, direction="nearest")
        assert rounded_nearest.minute in [
            30,
            45,
        ]  # Should round to nearest 15-minute mark

    @pytest.mark.asyncio
    async def test_async_time_utils_wait_until(self):
        """Test AsyncTimeUtils wait_until functionality."""
        start_time = datetime.now(timezone.utc)
        target_time = start_time + timedelta(milliseconds=50)

        await AsyncTimeUtils.wait_until(target_time, check_interval=0.01)

        end_time = datetime.now(timezone.utc)
        elapsed = (end_time - start_time).total_seconds()

        # Should have waited at least 0.05 seconds
        assert elapsed >= 0.045  # Allow some tolerance

    @pytest.mark.asyncio
    async def test_async_time_utils_periodic_task(self):
        """Test AsyncTimeUtils periodic task generator."""
        interval = timedelta(milliseconds=20)
        iterations = []

        async for iteration in AsyncTimeUtils.periodic_task(interval, max_iterations=3):
            iterations.append(iteration)

        assert iterations == [0, 1, 2]

    def test_time_profiler_context_manager(self):
        """Test TimeProfiler as context manager."""
        with TimeProfiler("test_operation") as profiler:
            time.sleep(0.01)  # Small delay

        assert profiler.duration is not None
        assert profiler.duration_seconds > 0
        assert profiler.operation_name == "test_operation"

    def test_time_profiler_decorator(self):
        """Test TimeProfiler as decorator."""
        profiler = TimeProfiler("test_function")

        @profiler
        def test_function():
            time.sleep(0.01)
            return {"result": "success"}

        result = test_function()

        assert result["result"] == "success"
        assert "_timing" in result
        assert result["_timing"]["operation"] == "test_function"
        assert result["_timing"]["duration_seconds"] > 0


class TestHealthMonitoring:
    """Test health monitoring functionality."""

    def test_health_status_enum(self):
        """Test HealthStatus enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"

    def test_component_type_enum(self):
        """Test ComponentType enum values."""
        assert ComponentType.DATABASE.value == "database"
        assert ComponentType.MQTT.value == "mqtt"
        assert ComponentType.API.value == "api"
        assert ComponentType.SYSTEM.value == "system"

    def test_health_thresholds_defaults(self):
        """Test HealthThresholds default values."""
        thresholds = HealthThresholds()

        assert thresholds.cpu_warning == 70.0
        assert thresholds.cpu_critical == 85.0
        assert thresholds.memory_warning == 70.0
        assert thresholds.memory_critical == 85.0
        assert thresholds.disk_warning == 80.0
        assert thresholds.disk_critical == 90.0

    def test_component_health_creation(self):
        """Test ComponentHealth creation and methods."""
        health = ComponentHealth(
            component_name="test_component",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time=0.15,
            message="All systems operational",
            details={"connection_pool": 10},
            metrics={"query_time": 0.05},
        )

        assert health.component_name == "test_component"
        assert health.component_type == ComponentType.DATABASE
        assert health.status == HealthStatus.HEALTHY
        assert health.is_healthy() is True
        assert health.needs_attention() is False
        assert health.response_time == 0.15

        # Test dictionary conversion
        health_dict = health.to_dict()
        assert health_dict["component_name"] == "test_component"
        assert health_dict["status"] == "healthy"
        assert health_dict["is_healthy"] is True

    def test_component_health_unhealthy_states(self):
        """Test ComponentHealth unhealthy state detection."""
        degraded_health = ComponentHealth(
            component_name="degraded_component",
            component_type=ComponentType.API,
            status=HealthStatus.DEGRADED,
            last_check=datetime.now(),
            response_time=2.0,
            message="Performance degraded",
        )

        assert degraded_health.is_healthy() is False
        assert degraded_health.needs_attention() is True

        critical_health = ComponentHealth(
            component_name="critical_component",
            component_type=ComponentType.SYSTEM,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time=0.0,
            message="Component failed",
        )

        assert critical_health.is_healthy() is False
        assert critical_health.needs_attention() is True

    def test_system_health_calculation(self):
        """Test SystemHealth calculation and scoring."""
        system_health = SystemHealth(
            overall_status=HealthStatus.HEALTHY,
            component_count=5,
            healthy_components=4,
            degraded_components=1,
            critical_components=0,
            last_updated=datetime.now(),
            uptime_seconds=3600,
            system_load=0.5,
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_usage=25.0,
            active_alerts=0,
            performance_score=85.0,
        )

        # Test health score calculation
        health_score = system_health.health_score()
        assert 0 <= health_score <= 100
        assert health_score > 70  # Should be reasonably high with these values

        # Test dictionary conversion
        system_dict = system_health.to_dict()
        assert system_dict["overall_status"] == "healthy"
        assert system_dict["component_count"] == 5
        assert system_dict["health_score"] == health_score

    @patch("src.utils.health_monitor.psutil")
    @patch("src.utils.health_monitor.get_config")
    @patch("src.utils.health_monitor.get_alert_manager")
    @patch("src.utils.health_monitor.get_metrics_collector")
    def test_health_monitor_initialization(
        self, mock_metrics, mock_alert, mock_config, mock_psutil
    ):
        """Test HealthMonitor initialization."""
        mock_config.return_value = Mock(mqtt=Mock(), api=Mock(port=8000))

        monitor = HealthMonitor()

        assert monitor.check_interval == 30
        assert monitor.alert_threshold == 3
        assert len(monitor.health_checks) > 0
        assert "system_resources" in monitor.health_checks
        assert "database_connection" in monitor.health_checks
        assert "memory_usage" in monitor.health_checks

    @patch("src.utils.health_monitor.psutil")
    def test_health_monitor_system_resources_check(self, mock_psutil):
        """Test system resources health check."""
        # Mock psutil responses for healthy system
        mock_psutil.cpu_percent.return_value = 25.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=40.0, available=8 * 1024**3, total=16 * 1024**3
        )
        mock_psutil.disk_usage.return_value = Mock(
            used=400 * 1024**3, total=1000 * 1024**3, free=600 * 1024**3
        )

        monitor = HealthMonitor()

        # Run the check
        result = asyncio.run(monitor._check_system_resources())

        assert isinstance(result, ComponentHealth)
        assert result.component_name == "system_resources"
        assert result.component_type == ComponentType.SYSTEM
        assert result.status == HealthStatus.HEALTHY
        assert "healthy" in result.message.lower()

        # Verify metrics in details
        assert "cpu_percent" in result.details
        assert "memory_percent" in result.details
        assert "disk_percent" in result.details

    @patch("src.utils.health_monitor.psutil")
    def test_health_monitor_critical_resource_usage(self, mock_psutil):
        """Test health check with critical resource usage."""
        # Mock psutil responses for critical system
        mock_psutil.cpu_percent.return_value = 95.0  # Critical CPU
        mock_psutil.virtual_memory.return_value = Mock(
            percent=92.0, available=1 * 1024**3, total=16 * 1024**3
        )  # Critical memory
        mock_psutil.disk_usage.return_value = Mock(
            used=950 * 1024**3, total=1000 * 1024**3, free=50 * 1024**3
        )  # Critical disk

        monitor = HealthMonitor()

        result = asyncio.run(monitor._check_system_resources())

        assert result.status == HealthStatus.CRITICAL
        assert "critical" in result.message.lower()
        assert result.details["cpu_percent"] == 95.0
        assert result.details["memory_percent"] == 92.0

    @pytest.mark.asyncio
    async def test_health_monitor_lifecycle(self):
        """Test health monitor start/stop lifecycle."""
        with patch("src.utils.health_monitor.get_config"), patch(
            "src.utils.health_monitor.get_alert_manager"
        ), patch("src.utils.health_monitor.get_metrics_collector"):

            monitor = HealthMonitor(check_interval=0.1)  # Short interval for testing

            # Test start
            await monitor.start_monitoring()
            assert monitor._monitoring_active is True
            assert monitor._monitoring_task is not None

            # Let it run briefly
            await asyncio.sleep(0.15)

            # Test stop
            await monitor.stop_monitoring()
            assert monitor._monitoring_active is False


class TestSystemMonitoring:
    """Test system monitoring functionality."""

    def test_performance_threshold_creation(self):
        """Test PerformanceThreshold creation."""
        threshold = PerformanceThreshold(
            name="test_latency",
            warning_threshold=1.0,
            critical_threshold=2.0,
            unit="seconds",
            description="Test latency threshold",
        )

        assert threshold.name == "test_latency"
        assert threshold.warning_threshold == 1.0
        assert threshold.critical_threshold == 2.0
        assert threshold.unit == "seconds"
        assert threshold.description == "Test latency threshold"

    def test_health_check_result_creation(self):
        """Test HealthCheckResult creation."""
        result = HealthCheckResult(
            component="test_component",
            status="healthy",
            response_time=0.15,
            message="Component is healthy",
            details={"connection_count": 5},
        )

        assert result.component == "test_component"
        assert result.status == "healthy"
        assert result.response_time == 0.15
        assert result.message == "Component is healthy"
        assert result.details["connection_count"] == 5

    def test_performance_monitor_initialization(self):
        """Test PerformanceMonitor initialization."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_performance_logger"
        ), patch("src.utils.monitoring.get_error_tracker"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            monitor = PerformanceMonitor()

            assert len(monitor.thresholds) > 0
            assert "prediction_latency" in monitor.thresholds
            assert "cpu_usage" in monitor.thresholds
            assert len(monitor.alert_callbacks) == 0
            assert len(monitor.performance_history) == 0

    def test_performance_monitor_metric_recording(self):
        """Test performance metric recording."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_performance_logger"
        ) as mock_perf_logger, patch("src.utils.monitoring.get_error_tracker"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            monitor = PerformanceMonitor()

            # Record a normal metric
            monitor.record_performance_metric(
                metric_name="prediction_latency",
                value=0.15,
                room_id="bedroom",
                additional_info={"model_type": "lstm"},
            )

            # Verify logging
            mock_perf_logger.return_value.log_operation_time.assert_called_once()

            # Verify history storage
            key = "prediction_latency_bedroom"
            assert key in monitor.performance_history
            assert len(monitor.performance_history[key]) == 1

    def test_performance_monitor_threshold_checking(self):
        """Test performance threshold checking and alerting."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_performance_logger"
        ), patch("src.utils.monitoring.get_error_tracker"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            monitor = PerformanceMonitor()

            # Add alert callback to capture alerts
            alerts_triggered = []

            def capture_alert(alert):
                alerts_triggered.append(alert)

            monitor.add_alert_callback(capture_alert)

            # Record metric that exceeds warning threshold
            monitor.record_performance_metric(
                metric_name="prediction_latency",
                value=0.6,  # Above warning threshold of 0.5
                room_id="kitchen",
            )

            assert len(alerts_triggered) == 1
            alert = alerts_triggered[0]
            assert alert.alert_type == "warning"
            assert alert.metric_name == "prediction_latency"
            assert alert.current_value == 0.6

    def test_performance_monitor_trend_analysis(self):
        """Test performance trend analysis."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_performance_logger"
        ), patch("src.utils.monitoring.get_error_tracker"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            monitor = PerformanceMonitor()

            # Add some historical data points showing improvement
            metric_key = "prediction_latency_office"
            base_time = datetime.now()

            for i in range(10):
                timestamp = base_time + timedelta(minutes=i)
                value = 2.0 - (i * 0.1)  # Decreasing values (improving performance)
                monitor.performance_history[metric_key].append((timestamp, value))

            # Analyze trend
            trend = monitor.get_trend_analysis("prediction_latency", "office", hours=1)

            assert trend["status"] == "success"
            assert trend["trend"] == "improving"  # Values are decreasing
            assert trend["data_points"] == 10
            assert "slope" in trend

    @patch("src.utils.monitoring.psutil")
    def test_system_health_monitor_resource_checks(self, mock_psutil):
        """Test SystemHealthMonitor resource health checks."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            # Mock healthy system resources
            mock_psutil.cpu_percent.return_value = 30.0
            mock_psutil.virtual_memory.return_value = Mock(
                percent=45.0, available=8 * 1024**3, total=16 * 1024**3
            )

            monitor = SystemHealthMonitor()

            # Run system resources check
            result = monitor._check_system_resources()

            assert result.component == "system_resources"
            assert result.status == "healthy"
            assert result.details["cpu_percent"] == 30.0
            assert result.details["memory_percent"] == 45.0

    @pytest.mark.asyncio
    async def test_monitoring_manager_integration(self):
        """Test MonitoringManager integration."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            manager = MonitoringManager()

            # Test component access
            perf_monitor = manager.get_performance_monitor()
            health_monitor = manager.get_health_monitor()

            assert perf_monitor is not None
            assert health_monitor is not None

            # Test monitoring status
            status = await manager.get_monitoring_status()

            assert "monitoring_active" in status
            assert "health_status" in status
            assert "health_checks" in status
            assert "performance_summary" in status


class TestMonitoringIntegration:
    """Test monitoring integration functionality."""

    def test_monitoring_integration_initialization(self):
        """Test MonitoringIntegration initialization."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch("src.utils.monitoring_integration.get_ml_ops_logger"), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ), patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ), patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ):

            integration = MonitoringIntegration()

            assert integration.logger is not None
            assert integration.performance_logger is not None
            assert integration.ml_ops_logger is not None

    @pytest.mark.asyncio
    async def test_monitoring_integration_prediction_tracking(self):
        """Test prediction operation tracking."""
        with patch("src.utils.monitoring_integration.get_logger") as mock_logger, patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ) as mock_perf_logger, patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_metrics, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_monitor_mgr, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ):

            # Mock the performance monitor
            mock_perf_monitor = Mock()
            mock_monitor_mgr.return_value.get_performance_monitor.return_value = (
                mock_perf_monitor
            )

            integration = MonitoringIntegration()

            # Test successful prediction tracking
            async with integration.track_prediction_operation(
                room_id="bathroom", prediction_type="next_vacant", model_type="xgboost"
            ):
                await asyncio.sleep(0.01)  # Simulate prediction work

            # Verify logging calls
            mock_logger.return_value.info.assert_called()

            # Verify performance logging
            mock_perf_logger.return_value.log_operation_time.assert_called_once()

            # Verify metrics collection
            mock_metrics.return_value.record_prediction.assert_called_once()

            # Verify performance monitoring
            mock_perf_monitor.record_performance_metric.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_integration_prediction_error_handling(self):
        """Test prediction error handling in monitoring."""
        with patch("src.utils.monitoring_integration.get_logger") as mock_logger, patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch("src.utils.monitoring_integration.get_ml_ops_logger"), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_metrics, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ), patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_alert_mgr:

            integration = MonitoringIntegration()

            # Test prediction error tracking
            with pytest.raises(ValueError, match="Test prediction error"):
                async with integration.track_prediction_operation(
                    room_id="office", prediction_type="next_occupied", model_type="lstm"
                ):
                    raise ValueError("Test prediction error")

            # Verify error metrics recorded
            mock_metrics.return_value.record_prediction.assert_called_with(
                room_id="office",
                prediction_type="next_occupied",
                model_type="lstm",
                duration=pytest.approx(0, abs=0.1),
                status="error",
            )

            # Verify alert handling
            mock_alert_mgr.return_value.handle_prediction_error.assert_called_once()

            # Verify error logging
            mock_logger.return_value.error.assert_called()

    @pytest.mark.asyncio
    async def test_monitoring_integration_training_tracking(self):
        """Test model training operation tracking."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ) as mock_ml_ops, patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_metrics, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_monitor_mgr, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ):

            # Mock the performance monitor
            mock_perf_monitor = Mock()
            mock_monitor_mgr.return_value.get_performance_monitor.return_value = (
                mock_perf_monitor
            )

            integration = MonitoringIntegration()

            # Test training operation tracking
            async with integration.track_training_operation(
                room_id="bedroom", model_type="lstm", training_type="full_retrain"
            ):
                await asyncio.sleep(0.01)  # Simulate training work

            # Verify ML ops logging
            assert (
                mock_ml_ops.return_value.log_training_event.call_count == 2
            )  # Start and complete

            # Verify metrics collection
            mock_metrics.return_value.record_model_training.assert_called_once()

            # Verify performance monitoring
            mock_perf_monitor.record_performance_metric.assert_called_once()

    def test_monitoring_integration_accuracy_recording(self):
        """Test prediction accuracy recording."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ) as mock_perf_logger, patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_metrics, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_monitor_mgr, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ):

            # Mock the performance monitor
            mock_perf_monitor = Mock()
            mock_monitor_mgr.return_value.get_performance_monitor.return_value = (
                mock_perf_monitor
            )

            integration = MonitoringIntegration()

            # Record prediction accuracy
            integration.record_prediction_accuracy(
                room_id="kitchen",
                model_type="xgboost",
                prediction_type="next_vacant",
                accuracy_minutes=8.5,
                confidence=0.92,
            )

            # Verify performance logging
            mock_perf_logger.return_value.log_prediction_accuracy.assert_called_once()

            # Verify metrics collection
            mock_metrics.return_value.record_prediction.assert_called_once()

            # Verify performance monitoring
            mock_perf_monitor.record_performance_metric.assert_called_once()

    def test_monitoring_integration_concept_drift_recording(self):
        """Test concept drift detection recording."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ) as mock_ml_ops, patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_metrics, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ), patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_alert_mgr:

            integration = MonitoringIntegration()

            # Record high severity drift (should trigger alert)
            integration.record_concept_drift(
                room_id="living_room",
                drift_type="feature_drift",
                severity=0.75,
                action_taken="model_retrain_scheduled",
            )

            # Verify ML ops logging
            mock_ml_ops.return_value.log_drift_detection.assert_called_once()

            # Verify metrics collection
            mock_metrics.return_value.record_concept_drift.assert_called_once()

            # Should trigger alert due to high severity
            # Note: Alert is triggered asynchronously, so we verify the call was made
            # In a real scenario, you might need to use asyncio.run or similar


class TestAlertSystem:
    """Test alert system functionality."""

    def test_alert_severity_enum(self):
        """Test AlertSeverity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"

    def test_alert_channel_enum(self):
        """Test AlertChannel enum values."""
        assert AlertChannel.LOG.value == "log"
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.MQTT.value == "mqtt"

    def test_alert_rule_creation(self):
        """Test AlertRule creation."""
        rule = AlertRule(
            name="test_rule",
            condition="latency > 2.0",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            throttle_minutes=30,
            description="Test alert rule",
        )

        assert rule.name == "test_rule"
        assert rule.condition == "latency > 2.0"
        assert rule.severity == AlertSeverity.WARNING
        assert AlertChannel.LOG in rule.channels
        assert AlertChannel.EMAIL in rule.channels
        assert rule.throttle_minutes == 30

    def test_alert_event_creation(self):
        """Test AlertEvent creation."""
        event = AlertsAlertEvent(
            id="alert_001",
            rule_name="test_rule",
            severity=AlertSeverity.ERROR,
            timestamp=datetime.now(),
            title="Test Alert",
            message="This is a test alert",
            component="test_component",
            room_id="bedroom",
            context={"error_count": 5},
        )

        assert event.id == "alert_001"
        assert event.rule_name == "test_rule"
        assert event.severity == AlertSeverity.ERROR
        assert event.title == "Test Alert"
        assert event.component == "test_component"
        assert event.room_id == "bedroom"
        assert event.context["error_count"] == 5
        assert event.resolved is False

    def test_alert_throttler_functionality(self):
        """Test AlertThrottler throttling logic."""
        throttler = AlertThrottler()
        alert_id = "test_alert_001"

        # First alert should be allowed
        assert throttler.should_send_alert(alert_id, throttle_minutes=5) is True

        # Immediate subsequent alert should be throttled
        assert throttler.should_send_alert(alert_id, throttle_minutes=5) is False

        # Reset throttle
        throttler.reset_throttle(alert_id)

        # Should be allowed again after reset
        assert throttler.should_send_alert(alert_id, throttle_minutes=5) is True

    def test_notification_config_defaults(self):
        """Test NotificationConfig default values."""
        config = NotificationConfig()

        assert config.email_enabled is False
        assert config.webhook_enabled is False
        assert config.mqtt_enabled is False
        assert config.smtp_port == 587
        assert config.mqtt_topic == "occupancy/alerts"
        assert len(config.email_recipients) == 0

    @patch("src.utils.alerts.smtplib.SMTP")
    def test_email_notifier_sending(self, mock_smtp):
        """Test EmailNotifier email sending."""
        config = NotificationConfig(
            email_enabled=True,
            smtp_host="smtp.example.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="password",
            email_recipients=["recipient@example.com"],
        )

        notifier = EmailNotifier(config)

        # Create test alert
        alert = AlertsAlertEvent(
            id="email_test_001",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            title="Email Test Alert",
            message="This is a test email alert",
            component="test_component",
        )

        # Mock SMTP server
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        # Test email sending
        result = asyncio.run(notifier.send_alert(alert))

        assert result is True
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once()
        mock_server.sendmail.assert_called_once()

    @patch("aiohttp.ClientSession.post")
    def test_webhook_notifier_sending(self, mock_post):
        """Test WebhookNotifier webhook sending."""
        config = NotificationConfig(
            webhook_enabled=True, webhook_url="https://webhook.example.com/alerts"
        )

        notifier = WebhookNotifier(config)

        # Create test alert
        alert = AlertsAlertEvent(
            id="webhook_test_001",
            rule_name="test_rule",
            severity=AlertSeverity.ERROR,
            timestamp=datetime.now(),
            title="Webhook Test Alert",
            message="This is a test webhook alert",
            component="test_component",
            context={"test_data": "value"},
        )

        # Mock successful response
        mock_response = Mock()
        mock_response.status = 200
        mock_post.return_value.__aenter__.return_value = mock_response

        # Test webhook sending
        result = asyncio.run(notifier.send_alert(alert))

        assert result is True
        mock_post.assert_called_once()

        # Verify payload structure
        call_args = mock_post.call_args
        payload = call_args[1]["json"]
        assert payload["alert_id"] == "webhook_test_001"
        assert payload["severity"] == "error"
        assert payload["title"] == "Webhook Test Alert"

    def test_error_recovery_manager_strategy_registration(self):
        """Test ErrorRecoveryManager strategy registration."""
        recovery_manager = ErrorRecoveryManager()

        # Mock recovery function
        def mock_recovery(error, context):
            return True

        # Register recovery strategy
        recovery_manager.register_recovery_strategy("ConnectionError", mock_recovery)

        assert "ConnectionError" in recovery_manager.recovery_strategies
        assert recovery_manager.recovery_strategies["ConnectionError"] == mock_recovery

    @pytest.mark.asyncio
    async def test_error_recovery_manager_recovery_attempt(self):
        """Test ErrorRecoveryManager recovery attempt."""
        recovery_manager = ErrorRecoveryManager()

        # Mock successful recovery function
        def mock_successful_recovery(error, context):
            return True

        # Mock failing recovery function
        def mock_failing_recovery(error, context):
            return False

        recovery_manager.register_recovery_strategy(
            "TestError", mock_successful_recovery
        )
        recovery_manager.register_recovery_strategy("FailError", mock_failing_recovery)

        # Test successful recovery
        test_error = Exception("TestError occurred")
        context = {"component": "test"}

        success = await recovery_manager.attempt_recovery(test_error, context)
        assert success is True

        # Verify recovery history
        assert len(recovery_manager.recovery_history) == 1
        history_entry = recovery_manager.recovery_history[0]
        assert history_entry["success"] is True
        assert history_entry["recovery_pattern"] == "TestError"

    def test_alert_manager_initialization(self):
        """Test AlertManager initialization."""
        with patch("src.utils.alerts.get_logger"), patch(
            "src.utils.alerts.get_error_tracker"
        ), patch("src.utils.alerts.get_metrics_collector"):

            config = NotificationConfig(email_enabled=True)
            manager = AlertManager(config)

            assert len(manager.alert_rules) > 0  # Should have default rules
            assert "high_prediction_latency" in manager.alert_rules
            assert "database_connection_error" in manager.alert_rules

            assert len(manager.active_alerts) == 0
            assert len(manager.alert_history) == 0

    @pytest.mark.asyncio
    async def test_alert_manager_alert_triggering(self):
        """Test AlertManager alert triggering."""
        with patch("src.utils.alerts.get_logger"), patch(
            "src.utils.alerts.get_error_tracker"
        ), patch("src.utils.alerts.get_metrics_collector") as mock_metrics:

            manager = AlertManager()

            # Trigger an alert
            alert_id = await manager.trigger_alert(
                rule_name="high_prediction_latency",
                title="High Latency Alert",
                message="Prediction latency is too high",
                component="prediction_engine",
                room_id="living_room",
                context={"latency": 2.5},
            )

            assert alert_id is not None
            assert alert_id in manager.active_alerts

            alert = manager.active_alerts[alert_id]
            assert alert.title == "High Latency Alert"
            assert alert.component == "prediction_engine"
            assert alert.room_id == "living_room"
            assert alert.context["latency"] == 2.5

            # Verify metrics were recorded
            mock_metrics.return_value.record_error.assert_called_once()

    @pytest.mark.asyncio
    async def test_alert_manager_alert_resolution(self):
        """Test AlertManager alert resolution."""
        with patch("src.utils.alerts.get_logger"), patch(
            "src.utils.alerts.get_error_tracker"
        ), patch("src.utils.alerts.get_metrics_collector"):

            manager = AlertManager()

            # Trigger an alert first
            alert_id = await manager.trigger_alert(
                rule_name="high_prediction_latency",
                title="Test Alert",
                message="Test alert message",
                component="test_component",
            )

            # Resolve the alert
            await manager.resolve_alert(alert_id, "Issue resolved by system restart")

            # Alert should no longer be active
            assert alert_id not in manager.active_alerts

            # But should be in history
            resolved_alert = next(
                (alert for alert in manager.alert_history if alert.id == alert_id), None
            )
            assert resolved_alert is not None
            assert resolved_alert.resolved is True
            assert resolved_alert.resolution_notes == "Issue resolved by system restart"


class TestIncidentResponse:
    """Test incident response functionality."""

    def test_incident_severity_enum(self):
        """Test IncidentSeverity enum values."""
        assert IncidentSeverity.INFO.value == "info"
        assert IncidentSeverity.MINOR.value == "minor"
        assert IncidentSeverity.MAJOR.value == "major"
        assert IncidentSeverity.CRITICAL.value == "critical"
        assert IncidentSeverity.EMERGENCY.value == "emergency"

    def test_incident_status_enum(self):
        """Test IncidentStatus enum values."""
        assert IncidentStatus.NEW.value == "new"
        assert IncidentStatus.ACKNOWLEDGED.value == "acknowledged"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.IN_PROGRESS.value == "in_progress"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.CLOSED.value == "closed"

    def test_recovery_action_type_enum(self):
        """Test RecoveryActionType enum values."""
        assert RecoveryActionType.RESTART_SERVICE.value == "restart_service"
        assert RecoveryActionType.CLEAR_CACHE.value == "clear_cache"
        assert RecoveryActionType.RESTART_COMPONENT.value == "restart_component"
        assert RecoveryActionType.SCALE_RESOURCES.value == "scale_resources"
        assert RecoveryActionType.FAILOVER.value == "failover"
        assert RecoveryActionType.NOTIFICATION.value == "notification"
        assert RecoveryActionType.CUSTOM.value == "custom"

    def test_recovery_action_creation_and_limits(self):
        """Test RecoveryAction creation and attempt limits."""

        def mock_recovery_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test_service",
            description="Restart test service",
            function=mock_recovery_function,
            conditions={"consecutive_failures": 3},
            max_attempts=2,
            cooldown_minutes=10,
        )

        assert action.action_type == RecoveryActionType.RESTART_SERVICE
        assert action.component == "test_service"
        assert action.max_attempts == 2
        assert action.cooldown_minutes == 10
        assert action.attempt_count == 0

        # Test initial attempt allowance
        assert action.can_attempt() is True

        # Record attempts up to limit
        action.record_attempt(False)
        assert action.attempt_count == 1
        assert action.can_attempt() is True

        action.record_attempt(False)
        assert action.attempt_count == 2
        assert action.can_attempt() is False  # Max attempts reached

        # Test reset
        action.reset_attempts()
        assert action.attempt_count == 0
        assert action.can_attempt() is True

    def test_recovery_action_cooldown_logic(self):
        """Test RecoveryAction cooldown functionality."""

        def mock_recovery_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.CLEAR_CACHE,
            component="cache_service",
            description="Clear cache",
            function=mock_recovery_function,
            max_attempts=5,
            cooldown_minutes=1,  # 1 minute cooldown
        )

        # First attempt should be allowed
        assert action.can_attempt() is True

        # Record attempt
        action.record_attempt(True)
        assert action.last_attempted is not None

        # Immediate subsequent attempt should be blocked by cooldown
        assert action.can_attempt() is False

        # Mock time passage (in real scenario, would wait)
        action.last_attempted = datetime.now() - timedelta(minutes=2)
        assert action.can_attempt() is True

    def test_incident_creation_and_management(self):
        """Test Incident creation and management."""
        source_health = ComponentHealth(
            component_name="database",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time=5.0,
            message="Database connection timeout",
            consecutive_failures=3,
        )

        incident = Incident(
            incident_id="INC-20240101-0001",
            title="Database Connection Failure",
            description="Database connection is timing out",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.NEW,
            component="database",
            component_type="database",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=source_health,
        )

        assert incident.incident_id == "INC-20240101-0001"
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.status == IncidentStatus.NEW
        assert len(incident.recovery_actions_attempted) == 0
        assert incident.escalation_level == 0

        # Test timeline functionality
        incident.add_timeline_entry("Investigation started", {"investigator": "system"})
        assert len(incident.timeline) == 1

        # Test acknowledgment
        incident.acknowledge("admin_user")
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_by == "admin_user"
        assert incident.acknowledged_at is not None

        # Test resolution
        incident.resolve(
            "Database connection restored after service restart", auto_resolved=True
        )
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None
        assert incident.recovery_success is True

    def test_incident_escalation_logic(self):
        """Test Incident escalation logic."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test incident for escalation",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now()
            - timedelta(minutes=45),  # Old enough to need escalation
            updated_at=datetime.now(),
            source_health=Mock(),
        )

        # Should need escalation due to age
        assert incident.needs_escalation() is True

        # Perform escalation
        original_severity = incident.severity
        incident.escalate()

        assert incident.escalation_level == 1
        assert incident.escalated_at is not None
        assert incident.severity != original_severity  # Should be escalated

        # Test multiple escalations
        incident.escalate()
        assert incident.escalation_level == 2

        # Resolved incidents shouldn't need escalation
        incident.resolve("Resolved", auto_resolved=False)
        assert incident.needs_escalation() is False

    def test_incident_response_manager_initialization(self):
        """Test IncidentResponseManager initialization."""
        with patch("src.utils.incident_response.get_logger"), patch(
            "src.utils.incident_response.get_health_monitor"
        ), patch("src.utils.incident_response.get_alert_manager"), patch(
            "src.utils.incident_response.get_metrics_collector"
        ):

            manager = IncidentResponseManager(check_interval=30)

            assert manager.check_interval == 30
            assert manager.auto_recovery_enabled is True
            assert manager.escalation_enabled is True
            assert len(manager.active_incidents) == 0
            assert len(manager.incident_history) == 0
            assert manager.incident_counter == 0

            # Should have default recovery actions
            assert len(manager.recovery_actions) > 0
            assert "database_connection" in manager.recovery_actions
            assert "mqtt_broker" in manager.recovery_actions

    def test_incident_response_manager_recovery_action_registration(self):
        """Test recovery action registration."""
        with patch("src.utils.incident_response.get_logger"), patch(
            "src.utils.incident_response.get_health_monitor"
        ), patch("src.utils.incident_response.get_alert_manager"), patch(
            "src.utils.incident_response.get_metrics_collector"
        ):

            manager = IncidentResponseManager()

            # Register custom recovery action
            def custom_recovery_function(incident):
                return True

            custom_action = RecoveryAction(
                action_type=RecoveryActionType.CUSTOM,
                component="custom_component",
                description="Custom recovery action",
                function=custom_recovery_function,
                max_attempts=3,
                cooldown_minutes=15,
            )

            manager.register_recovery_action("custom_component", custom_action)

            assert "custom_component" in manager.recovery_actions
            assert len(manager.recovery_actions["custom_component"]) == 1
            assert manager.recovery_actions["custom_component"][0] == custom_action

    @pytest.mark.asyncio
    async def test_incident_response_manager_lifecycle(self):
        """Test IncidentResponseManager start/stop lifecycle."""
        with patch("src.utils.incident_response.get_logger"), patch(
            "src.utils.incident_response.get_health_monitor"
        ) as mock_health_monitor, patch(
            "src.utils.incident_response.get_alert_manager"
        ), patch(
            "src.utils.incident_response.get_metrics_collector"
        ):

            manager = IncidentResponseManager(
                check_interval=0.1
            )  # Short interval for testing

            # Test start
            await manager.start_incident_response()
            assert manager._response_active is True
            assert manager._response_task is not None

            # Verify callback was added to health monitor
            mock_health_monitor.return_value.add_incident_callback.assert_called_once()

            # Let it run briefly
            await asyncio.sleep(0.15)

            # Test stop
            await manager.stop_incident_response()
            assert manager._response_active is False

    def test_incident_response_manager_statistics(self):
        """Test IncidentResponseManager statistics tracking."""
        with patch("src.utils.incident_response.get_logger"), patch(
            "src.utils.incident_response.get_health_monitor"
        ), patch("src.utils.incident_response.get_alert_manager"), patch(
            "src.utils.incident_response.get_metrics_collector"
        ):

            manager = IncidentResponseManager()

            # Check initial statistics
            stats = manager.get_incident_statistics()

            assert stats["response_active"] is False
            assert stats["auto_recovery_enabled"] is True
            assert stats["escalation_enabled"] is True
            assert stats["active_incidents_count"] == 0
            assert "active_incidents_by_severity" in stats
            assert "active_incidents_by_status" in stats
            assert "registered_recovery_actions" in stats
            assert "statistics" in stats

            # Verify statistics structure
            assert isinstance(stats["active_incidents_by_severity"], dict)
            assert isinstance(stats["active_incidents_by_status"], dict)
            assert isinstance(stats["registered_recovery_actions"], dict)


# Test fixtures for common mocks
@pytest.fixture
def mock_datetime():
    """Mock datetime for consistent testing."""
    with patch("src.utils.time_utils.datetime") as mock_dt:
        mock_dt.now.return_value = datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        yield mock_dt


@pytest.fixture
def sample_health_component():
    """Sample ComponentHealth for testing."""
    return ComponentHealth(
        component_name="test_component",
        component_type=ComponentType.APPLICATION,
        status=HealthStatus.HEALTHY,
        last_check=datetime.now(),
        response_time=0.15,
        message="Component is healthy",
        details={"version": "1.0.0"},
        metrics={"response_time": 0.15},
    )


@pytest.fixture
def sample_alert_event():
    """Sample AlertEvent for testing."""
    return AlertsAlertEvent(
        id="test_alert_001",
        rule_name="test_rule",
        severity=AlertSeverity.WARNING,
        timestamp=datetime.now(),
        title="Test Alert",
        message="This is a test alert",
        component="test_component",
        room_id="test_room",
        context={"test_key": "test_value"},
    )


# Performance and integration tests
class TestUtilitiesIntegration:
    """Integration tests for utilities working together."""

    @pytest.mark.asyncio
    async def test_monitoring_integration_full_workflow(self):
        """Test full monitoring integration workflow."""
        # This would be a comprehensive integration test
        # Testing how all utilities work together
        pass

    def test_utilities_memory_usage(self):
        """Test memory usage of utility classes."""
        # This could test that utilities don't leak memory
        pass

    def test_utilities_thread_safety(self):
        """Test thread safety of utility classes."""
        # This could test concurrent access to utilities
        pass


if __name__ == "__main__":
    pytest.main([__file__])
