"""
Comprehensive unit tests for monitoring.py with deeper coverage.
Tests performance monitoring, health checks, alerting, and complex scenarios.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import psutil
import pytest
import statistics

from src.utils.monitoring import (
    AlertEvent,
    HealthCheckResult,
    MonitoringManager,
    PerformanceMonitor,
    PerformanceThreshold,
    SystemHealthMonitor,
    get_monitoring_manager,
)


class TestPerformanceThreshold:
    """Test PerformanceThreshold dataclass."""

    def test_performance_threshold_creation(self):
        """Test creating PerformanceThreshold instances."""
        threshold = PerformanceThreshold(
            name="test_metric",
            warning_threshold=10.0,
            critical_threshold=20.0,
            unit="seconds",
            description="Test metric threshold",
        )

        assert threshold.name == "test_metric"
        assert threshold.warning_threshold == 10.0
        assert threshold.critical_threshold == 20.0
        assert threshold.unit == "seconds"
        assert threshold.description == "Test metric threshold"

    def test_performance_threshold_equality(self):
        """Test PerformanceThreshold equality comparison."""
        threshold1 = PerformanceThreshold("metric", 5.0, 10.0, "ms", "desc")
        threshold2 = PerformanceThreshold("metric", 5.0, 10.0, "ms", "desc")
        threshold3 = PerformanceThreshold("metric", 6.0, 10.0, "ms", "desc")

        assert threshold1 == threshold2
        assert threshold1 != threshold3


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass."""

    def test_health_check_result_creation(self):
        """Test creating HealthCheckResult instances."""
        result = HealthCheckResult(
            component="test_component",
            status="healthy",
            response_time=0.125,
            message="All systems operational",
            details={"cpu": 45.2, "memory": 67.8},
        )

        assert result.component == "test_component"
        assert result.status == "healthy"
        assert result.response_time == 0.125
        assert result.message == "All systems operational"
        assert result.details["cpu"] == 45.2
        assert result.details["memory"] == 67.8

    def test_health_check_result_without_details(self):
        """Test HealthCheckResult without details."""
        result = HealthCheckResult(
            component="simple_component",
            status="warning",
            response_time=0.5,
            message="Minor issues detected",
        )

        assert result.details is None


class TestAlertEvent:
    """Test AlertEvent dataclass."""

    def test_alert_event_creation(self):
        """Test creating AlertEvent instances."""
        timestamp = datetime.now()
        alert = AlertEvent(
            timestamp=timestamp,
            alert_type="critical",
            component="database",
            message="High response time detected",
            metric_name="database_query_time",
            current_value=5.2,
            threshold=2.0,
            additional_info={"table": "sensor_events", "query_type": "select"},
        )

        assert alert.timestamp == timestamp
        assert alert.alert_type == "critical"
        assert alert.component == "database"
        assert alert.message == "High response time detected"
        assert alert.metric_name == "database_query_time"
        assert alert.current_value == 5.2
        assert alert.threshold == 2.0
        assert alert.additional_info["table"] == "sensor_events"


class TestPerformanceMonitorAdvanced:
    """Advanced tests for PerformanceMonitor."""

    @pytest.fixture
    def mock_logger(self):
        with patch("src.utils.monitoring.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            yield mock_logger

    @pytest.fixture
    def mock_perf_logger(self):
        with patch(
            "src.utils.monitoring.get_performance_logger"
        ) as mock_get_perf_logger:
            mock_perf_logger = Mock()
            mock_get_perf_logger.return_value = mock_perf_logger
            yield mock_perf_logger

    @pytest.fixture
    def mock_error_tracker(self):
        with patch("src.utils.monitoring.get_error_tracker") as mock_get_error_tracker:
            mock_error_tracker = Mock()
            mock_get_error_tracker.return_value = mock_error_tracker
            yield mock_error_tracker

    @pytest.fixture
    def mock_metrics_collector(self):
        with patch(
            "src.utils.monitoring.get_metrics_collector"
        ) as mock_get_metrics_collector:
            mock_metrics_collector = Mock()
            mock_get_metrics_collector.return_value = mock_metrics_collector
            yield mock_metrics_collector

    @pytest.fixture
    def performance_monitor(
        self, mock_logger, mock_perf_logger, mock_error_tracker, mock_metrics_collector
    ):
        """Create PerformanceMonitor instance with mocked dependencies."""
        return PerformanceMonitor()

    def test_performance_monitor_initialization(self, performance_monitor):
        """Test PerformanceMonitor initialization."""
        assert isinstance(performance_monitor.performance_history, defaultdict)
        assert isinstance(performance_monitor.alert_callbacks, list)
        assert len(performance_monitor.alert_callbacks) == 0
        assert isinstance(performance_monitor.thresholds, dict)

        # Verify default thresholds are set
        expected_thresholds = [
            "prediction_latency",
            "feature_computation_time",
            "model_training_time",
            "database_query_time",
            "prediction_accuracy",
            "cpu_usage",
            "memory_usage",
            "disk_usage",
            "error_rate",
        ]

        for threshold_name in expected_thresholds:
            assert threshold_name in performance_monitor.thresholds
            threshold = performance_monitor.thresholds[threshold_name]
            assert isinstance(threshold, PerformanceThreshold)
            assert threshold.name == threshold_name

    def test_add_alert_callback(self, performance_monitor):
        """Test adding alert callbacks."""
        callback1 = Mock()
        callback2 = Mock()

        performance_monitor.add_alert_callback(callback1)
        assert len(performance_monitor.alert_callbacks) == 1
        assert callback1 in performance_monitor.alert_callbacks

        performance_monitor.add_alert_callback(callback2)
        assert len(performance_monitor.alert_callbacks) == 2
        assert callback2 in performance_monitor.alert_callbacks

    def test_record_performance_metric_with_room_id(
        self, performance_monitor, mock_perf_logger
    ):
        """Test recording performance metric with room ID."""
        additional_info = {"model_type": "lstm", "batch_size": 32}

        performance_monitor.record_performance_metric(
            metric_name="prediction_latency",
            value=0.25,
            room_id="living_room",
            additional_info=additional_info,
        )

        # Verify metric was stored in history
        key = "prediction_latency_living_room"
        assert key in performance_monitor.performance_history
        history = performance_monitor.performance_history[key]
        assert len(history) == 1

        timestamp, value = history[0]
        assert isinstance(timestamp, datetime)
        assert value == 0.25

        # Verify performance logger was called
        mock_perf_logger.log_operation_time.assert_called_once_with(
            operation="prediction_latency",
            duration=0.25,
            room_id="living_room",
            model_type="lstm",
            batch_size=32,
        )

    def test_record_performance_metric_without_room_id(self, performance_monitor):
        """Test recording performance metric without room ID."""
        performance_monitor.record_performance_metric(
            metric_name="cpu_usage", value=75.5
        )

        # Verify metric was stored in history
        key = "cpu_usage"
        assert key in performance_monitor.performance_history
        history = performance_monitor.performance_history[key]
        assert len(history) == 1

        timestamp, value = history[0]
        assert value == 75.5

    def test_check_threshold_warning_level(
        self, performance_monitor, mock_logger, mock_metrics_collector
    ):
        """Test threshold checking for warning level."""
        # Add a mock alert callback
        callback = Mock()
        performance_monitor.add_alert_callback(callback)

        # Record a metric that exceeds warning threshold but not critical
        performance_monitor.record_performance_metric(
            metric_name="prediction_latency",
            value=0.75,  # Between warning (0.5) and critical (2.0)
            room_id="bedroom",
        )

        # Verify warning alert was triggered
        mock_logger.warning.assert_called_once()
        warning_call = mock_logger.warning.call_args
        assert "Performance alert" in warning_call[0][0]

        # Verify metrics collector recorded error
        mock_metrics_collector.record_error.assert_called_once_with(
            error_type="performance_warning", component="bedroom", severity="warning"
        )

        # Verify callback was called
        callback.assert_called_once()
        alert = callback.call_args[0][0]
        assert isinstance(alert, AlertEvent)
        assert alert.alert_type == "warning"
        assert alert.current_value == 0.75
        assert alert.threshold == 0.5  # Warning threshold

    def test_check_threshold_critical_level(self, performance_monitor):
        """Test threshold checking for critical level."""
        callback = Mock()
        performance_monitor.add_alert_callback(callback)

        # Record a metric that exceeds critical threshold
        performance_monitor.record_performance_metric(
            metric_name="prediction_latency",
            value=2.5,  # Above critical threshold (2.0)
            room_id="kitchen",
        )

        # Verify critical alert was triggered
        callback.assert_called_once()
        alert = callback.call_args[0][0]
        assert alert.alert_type == "critical"
        assert alert.current_value == 2.5
        assert alert.threshold == 2.0  # Critical threshold

    def test_check_threshold_no_alert(self, performance_monitor):
        """Test threshold checking when no alert should be triggered."""
        callback = Mock()
        performance_monitor.add_alert_callback(callback)

        # Record a metric below warning threshold
        performance_monitor.record_performance_metric(
            metric_name="prediction_latency",
            value=0.1,  # Below warning threshold (0.5)
            room_id="office",
        )

        # Verify no alert was triggered
        callback.assert_not_called()

    def test_trigger_alert_callback_exception_handling(
        self, performance_monitor, mock_error_tracker
    ):
        """Test alert callback exception handling."""
        # Add callbacks, one that fails
        working_callback = Mock()
        failing_callback = Mock(side_effect=Exception("Callback error"))

        performance_monitor.add_alert_callback(working_callback)
        performance_monitor.add_alert_callback(failing_callback)

        # Trigger alert
        performance_monitor.record_performance_metric(
            metric_name="prediction_latency",
            value=1.0,  # Above warning threshold
            room_id="garage",
        )

        # Verify working callback was called
        working_callback.assert_called_once()

        # Verify failing callback was called and error was tracked
        failing_callback.assert_called_once()
        mock_error_tracker.track_error.assert_called_once()

    def test_get_performance_summary_with_data(self, performance_monitor):
        """Test getting performance summary with historical data."""
        # Add historical data
        base_time = datetime.now()

        # Add data for prediction_latency
        values = [0.1, 0.2, 0.15, 0.3, 0.25, 0.18, 0.22]
        for i, value in enumerate(values):
            timestamp = base_time - timedelta(minutes=i)
            performance_monitor.performance_history["prediction_latency"].append(
                (timestamp, value)
            )

        # Add data for cpu_usage
        cpu_values = [45.2, 52.1, 38.7, 61.3, 47.9]
        for i, value in enumerate(cpu_values):
            timestamp = base_time - timedelta(minutes=i)
            performance_monitor.performance_history["cpu_usage"].append(
                (timestamp, value)
            )

        summary = performance_monitor.get_performance_summary(hours=1)

        # Verify prediction_latency summary
        assert "prediction_latency" in summary
        latency_summary = summary["prediction_latency"]
        assert latency_summary["count"] == len(values)
        assert latency_summary["mean"] == statistics.mean(values)
        assert latency_summary["median"] == statistics.median(values)
        assert latency_summary["min"] == min(values)
        assert latency_summary["max"] == max(values)

        # Verify cpu_usage summary
        assert "cpu_usage" in summary
        cpu_summary = summary["cpu_usage"]
        assert cpu_summary["count"] == len(cpu_values)
        assert cpu_summary["mean"] == statistics.mean(cpu_values)

    def test_get_performance_summary_no_data(self, performance_monitor):
        """Test getting performance summary with no data."""
        summary = performance_monitor.get_performance_summary(hours=24)
        assert summary == {}

    def test_get_performance_summary_time_filtering(self, performance_monitor):
        """Test performance summary time filtering."""
        base_time = datetime.now()

        # Add old data (outside time window)
        old_timestamp = base_time - timedelta(hours=25)
        performance_monitor.performance_history["test_metric"].append(
            (old_timestamp, 1.0)
        )

        # Add recent data (within time window)
        recent_timestamp = base_time - timedelta(minutes=30)
        performance_monitor.performance_history["test_metric"].append(
            (recent_timestamp, 2.0)
        )

        summary = performance_monitor.get_performance_summary(hours=1)

        # Should only include recent data
        assert "test_metric" in summary
        assert summary["test_metric"]["count"] == 1
        assert summary["test_metric"]["mean"] == 2.0

    def test_percentile_calculation(self, performance_monitor):
        """Test percentile calculation accuracy."""
        # Test known values
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        p50 = performance_monitor._percentile(values, 50)
        assert p50 == 5.5  # Median of 1-10

        p95 = performance_monitor._percentile(values, 95)
        assert p95 == 9.55  # 95th percentile

        p99 = performance_monitor._percentile(values, 99)
        assert p99 == 9.91  # 99th percentile

        # Test edge cases
        assert performance_monitor._percentile([], 50) == 0.0
        assert performance_monitor._percentile([1], 50) == 1.0

    def test_get_trend_analysis_no_data(self, performance_monitor):
        """Test trend analysis with no data."""
        result = performance_monitor.get_trend_analysis("nonexistent_metric")
        assert result["status"] == "no_data"

    def test_get_trend_analysis_insufficient_data(self, performance_monitor):
        """Test trend analysis with insufficient data."""
        base_time = datetime.now()
        timestamp = base_time - timedelta(minutes=5)
        performance_monitor.performance_history["test_metric"].append((timestamp, 1.0))

        result = performance_monitor.get_trend_analysis("test_metric")
        assert result["status"] == "insufficient_data"

    def test_get_trend_analysis_improving_trend(self, performance_monitor):
        """Test trend analysis with improving (decreasing) trend."""
        base_time = datetime.now()

        # Add decreasing values (improving for latency metrics)
        values_and_offsets = [(1.0, 60), (0.8, 50), (0.6, 40), (0.4, 30), (0.2, 20)]

        for value, minutes_offset in values_and_offsets:
            timestamp = base_time - timedelta(minutes=minutes_offset)
            performance_monitor.performance_history["test_metric"].append(
                (timestamp, value)
            )

        result = performance_monitor.get_trend_analysis("test_metric", hours=2)

        assert result["status"] == "success"
        assert result["trend"] == "improving"  # Negative slope for decreasing values
        assert result["slope"] < 0
        assert result["data_points"] == 5

    def test_get_trend_analysis_degrading_trend(self, performance_monitor):
        """Test trend analysis with degrading (increasing) trend."""
        base_time = datetime.now()

        # Add increasing values (degrading for latency metrics)
        values_and_offsets = [(0.2, 60), (0.4, 50), (0.6, 40), (0.8, 30), (1.0, 20)]

        for value, minutes_offset in values_and_offsets:
            timestamp = base_time - timedelta(minutes=minutes_offset)
            performance_monitor.performance_history["test_metric"].append(
                (timestamp, value)
            )

        result = performance_monitor.get_trend_analysis("test_metric", hours=2)

        assert result["status"] == "success"
        assert result["trend"] == "degrading"  # Positive slope for increasing values
        assert result["slope"] > 0

    def test_get_trend_analysis_stable_trend(self, performance_monitor):
        """Test trend analysis with stable trend."""
        base_time = datetime.now()

        # Add stable values
        for minutes_offset in [60, 50, 40, 30, 20]:
            timestamp = base_time - timedelta(minutes=minutes_offset)
            performance_monitor.performance_history["test_metric"].append(
                (timestamp, 0.5)
            )

        result = performance_monitor.get_trend_analysis("test_metric", hours=2)

        assert result["status"] == "success"
        assert result["trend"] == "stable"  # Zero slope
        assert abs(result["slope"]) < 0.001  # Near zero slope

    def test_performance_history_maxlen_behavior(self, performance_monitor):
        """Test that performance history respects maxlen limit."""
        # Add more than maxlen (1000) entries
        base_time = datetime.now()

        for i in range(1200):
            timestamp = base_time - timedelta(seconds=i)
            performance_monitor.performance_history["test_metric"].append(
                (timestamp, float(i))
            )

        # Should only keep 1000 most recent entries
        history = performance_monitor.performance_history["test_metric"]
        assert len(history) == 1000

        # Should keep the most recent entries
        _, first_value = history[0]
        assert first_value == 199.0  # Should start from 199 (1200 - 1000 - 1)


class TestSystemHealthMonitorAdvanced:
    """Advanced tests for SystemHealthMonitor."""

    @pytest.fixture
    def mock_logger(self):
        with patch("src.utils.monitoring.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            yield mock_logger

    @pytest.fixture
    def mock_metrics_collector(self):
        with patch(
            "src.utils.monitoring.get_metrics_collector"
        ) as mock_get_metrics_collector:
            mock_metrics_collector = Mock()
            mock_get_metrics_collector.return_value = mock_metrics_collector
            yield mock_metrics_collector

    @pytest.fixture
    def health_monitor(self, mock_logger, mock_metrics_collector):
        """Create SystemHealthMonitor instance with mocked dependencies."""
        return SystemHealthMonitor()

    def test_system_health_monitor_initialization(self, health_monitor):
        """Test SystemHealthMonitor initialization."""
        assert isinstance(health_monitor.health_checks, dict)
        assert isinstance(health_monitor.last_health_check, dict)

        # Verify default health checks are registered
        expected_checks = [
            "system_resources",
            "disk_space",
            "memory_usage",
            "cpu_usage",
        ]

        for check_name in expected_checks:
            assert check_name in health_monitor.health_checks
            assert callable(health_monitor.health_checks[check_name])

    def test_register_custom_health_check(self, health_monitor):
        """Test registering custom health check."""

        def custom_check():
            return HealthCheckResult(
                component="custom_component",
                status="healthy",
                response_time=0.05,
                message="Custom check passed",
            )

        health_monitor.register_health_check("custom_check", custom_check)

        assert "custom_check" in health_monitor.health_checks
        assert health_monitor.health_checks["custom_check"] == custom_check

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_system_resources_healthy(
        self, mock_virtual_memory, mock_cpu_percent, health_monitor
    ):
        """Test system resources check with healthy values."""
        mock_cpu_percent.return_value = 45.2

        mock_memory = Mock()
        mock_memory.percent = 60.5
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_virtual_memory.return_value = mock_memory

        result = health_monitor._check_system_resources()

        assert isinstance(result, HealthCheckResult)
        assert result.component == "system_resources"
        assert result.status == "healthy"
        assert "Resource usage normal" in result.message
        assert result.details["cpu_percent"] == 45.2
        assert result.details["memory_percent"] == 60.5
        assert result.response_time > 0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_system_resources_warning(
        self, mock_virtual_memory, mock_cpu_percent, health_monitor
    ):
        """Test system resources check with warning values."""
        mock_cpu_percent.return_value = 75.0  # Above 70% warning threshold

        mock_memory = Mock()
        mock_memory.percent = 65.0  # Below warning threshold
        mock_memory.available = 6 * 1024**3
        mock_memory.total = 16 * 1024**3
        mock_virtual_memory.return_value = mock_memory

        result = health_monitor._check_system_resources()

        assert result.status == "warning"
        assert "Elevated resource usage" in result.message
        assert "CPU 75.0%" in result.message

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_system_resources_critical(
        self, mock_virtual_memory, mock_cpu_percent, health_monitor
    ):
        """Test system resources check with critical values."""
        mock_cpu_percent.return_value = 95.0  # Above 90% critical threshold

        mock_memory = Mock()
        mock_memory.percent = 92.0  # Above 90% critical threshold
        mock_memory.available = 1 * 1024**3
        mock_memory.total = 16 * 1024**3
        mock_virtual_memory.return_value = mock_memory

        result = health_monitor._check_system_resources()

        assert result.status == "critical"
        assert "High resource usage" in result.message
        assert "CPU 95.0%" in result.message
        assert "Memory 92.0%" in result.message

    @patch("psutil.cpu_percent")
    def test_check_system_resources_exception(self, mock_cpu_percent, health_monitor):
        """Test system resources check with exception."""
        mock_cpu_percent.side_effect = Exception("PSUtil error")

        result = health_monitor._check_system_resources()

        assert result.status == "critical"
        assert "Failed to check system resources" in result.message
        assert result.details["error"] == "PSUtil error"

    @patch("psutil.disk_usage")
    def test_check_disk_space_healthy(self, mock_disk_usage, health_monitor):
        """Test disk space check with healthy values."""
        mock_usage = Mock()
        mock_usage.total = 1000 * 1024**3  # 1TB
        mock_usage.used = 500 * 1024**3  # 500GB (50% used)
        mock_usage.free = 500 * 1024**3  # 500GB free
        mock_disk_usage.return_value = mock_usage

        result = health_monitor._check_disk_space()

        assert result.component == "disk_space"
        assert result.status == "healthy"
        assert "Disk space healthy" in result.message
        assert "50.0% used" in result.message
        assert result.details["percent"] == 50.0

    @patch("psutil.disk_usage")
    def test_check_disk_space_warning(self, mock_disk_usage, health_monitor):
        """Test disk space check with warning values."""
        mock_usage = Mock()
        mock_usage.total = 1000 * 1024**3
        mock_usage.used = 900 * 1024**3  # 90% used (warning)
        mock_usage.free = 100 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = health_monitor._check_disk_space()

        assert result.status == "warning"
        assert "Low disk space" in result.message
        assert "90.0% used" in result.message

    @patch("psutil.disk_usage")
    def test_check_disk_space_critical(self, mock_disk_usage, health_monitor):
        """Test disk space check with critical values."""
        mock_usage = Mock()
        mock_usage.total = 1000 * 1024**3
        mock_usage.used = 980 * 1024**3  # 98% used (critical)
        mock_usage.free = 20 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = health_monitor._check_disk_space()

        assert result.status == "critical"
        assert "Critical disk space" in result.message
        assert "98.0% used" in result.message

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_check_memory_usage_healthy(
        self, mock_virtual_memory, mock_process, health_monitor
    ):
        """Test memory usage check with healthy values."""
        # Setup process memory
        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 2 * 1024**3  # 2GB process memory
        mock_memory_info.vms = 4 * 1024**3  # 4GB virtual memory
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance

        # Setup system memory
        mock_system_memory = Mock()
        mock_system_memory.total = 16 * 1024**3  # 16GB total
        mock_system_memory.available = 12 * 1024**3  # 12GB available
        mock_virtual_memory.return_value = mock_system_memory

        result = health_monitor._check_memory_usage()

        assert result.component == "memory_usage"
        assert result.status == "healthy"
        assert "Memory usage normal" in result.message
        # Process is using 2GB out of 16GB = 12.5%
        assert "12.5% of system memory" in result.message

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_check_memory_usage_warning_moderate(
        self, mock_virtual_memory, mock_process, health_monitor
    ):
        """Test memory usage check with moderate warning values."""
        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 5 * 1024**3  # 5GB process memory
        mock_memory_info.vms = 8 * 1024**3
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance

        mock_system_memory = Mock()
        mock_system_memory.total = 16 * 1024**3
        mock_system_memory.available = 10 * 1024**3
        mock_virtual_memory.return_value = mock_system_memory

        result = health_monitor._check_memory_usage()

        assert result.status == "warning"
        assert "Moderate memory usage" in result.message
        # Process is using 5GB out of 16GB = 31.25%
        assert "31.2% of system memory" in result.message

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_check_memory_usage_warning_high(
        self, mock_virtual_memory, mock_process, health_monitor
    ):
        """Test memory usage check with high warning values."""
        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 9 * 1024**3  # 9GB process memory (high)
        mock_memory_info.vms = 12 * 1024**3
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance

        mock_system_memory = Mock()
        mock_system_memory.total = 16 * 1024**3
        mock_system_memory.available = 6 * 1024**3
        mock_virtual_memory.return_value = mock_system_memory

        result = health_monitor._check_memory_usage()

        assert result.status == "warning"
        assert "High memory usage" in result.message
        # Process is using 9GB out of 16GB = 56.25%
        assert "56.2% of system memory" in result.message

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_check_cpu_usage_with_load_avg(
        self, mock_cpu_count, mock_cpu_percent, health_monitor
    ):
        """Test CPU usage check including load average when available."""
        mock_cpu_percent.return_value = 45.7
        mock_cpu_count.return_value = 8

        # Mock load average (Unix systems)
        with patch(
            "psutil.getloadavg", return_value=(1.2, 1.5, 1.8)
        ) as mock_getloadavg:
            result = health_monitor._check_cpu_usage()

            assert result.status == "healthy"
            assert result.details["cpu_percent"] == 45.7
            assert result.details["cpu_count"] == 8
            assert result.details["load_avg"] == (1.2, 1.5, 1.8)

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_check_cpu_usage_without_load_avg(
        self, mock_cpu_count, mock_cpu_percent, health_monitor
    ):
        """Test CPU usage check when load average is not available (Windows)."""
        mock_cpu_percent.return_value = 65.3
        mock_cpu_count.return_value = 4

        # Simulate Windows where getloadavg doesn't exist
        result = health_monitor._check_cpu_usage()

        assert result.status == "healthy"
        assert result.details["cpu_percent"] == 65.3
        assert result.details["cpu_count"] == 4
        assert result.details["load_avg"] is None

    @pytest.mark.asyncio
    async def test_run_health_checks_all_healthy(
        self, health_monitor, mock_metrics_collector
    ):
        """Test running all health checks with healthy results."""
        # Mock all health check functions to return healthy results
        healthy_result = HealthCheckResult(
            component="test", status="healthy", response_time=0.1, message="OK"
        )

        for check_name in health_monitor.health_checks:
            health_monitor.health_checks[check_name] = Mock(return_value=healthy_result)

        results = await health_monitor.run_health_checks()

        # Verify all checks were run
        assert len(results) == len(health_monitor.health_checks)

        for check_name, result in results.items():
            assert isinstance(result, HealthCheckResult)
            assert result.status == "healthy"

            # Verify last_health_check was updated
            assert check_name in health_monitor.last_health_check
            assert isinstance(health_monitor.last_health_check[check_name], datetime)

        # Verify metrics were updated (health score should be 1.0)
        assert mock_metrics_collector.update_system_health_score.call_count >= 1

    @pytest.mark.asyncio
    async def test_run_health_checks_with_warnings(
        self, health_monitor, mock_metrics_collector
    ):
        """Test running health checks with warning results."""
        warning_result = HealthCheckResult(
            component="test", status="warning", response_time=0.2, message="Warning"
        )

        for check_name in health_monitor.health_checks:
            health_monitor.health_checks[check_name] = Mock(return_value=warning_result)

        results = await health_monitor.run_health_checks()

        for result in results.values():
            assert result.status == "warning"

        # Verify metrics show reduced health score (0.5 for warnings)
        mock_metrics_collector.update_system_health_score.assert_called()

    @pytest.mark.asyncio
    async def test_run_health_checks_with_exceptions(self, health_monitor, mock_logger):
        """Test running health checks when some checks fail with exceptions."""
        # Set up one working check and one failing check
        working_result = HealthCheckResult(
            component="working", status="healthy", response_time=0.1, message="OK"
        )

        health_monitor.health_checks = {
            "working_check": Mock(return_value=working_result),
            "failing_check": Mock(side_effect=Exception("Check failed")),
        }

        results = await health_monitor.run_health_checks()

        # Verify working check succeeded
        assert "working_check" in results
        assert results["working_check"].status == "healthy"

        # Verify failing check resulted in critical status
        assert "failing_check" in results
        assert results["failing_check"].status == "critical"
        assert "Health check failed" in results["failing_check"].message

        # Verify error was logged
        mock_logger.error.assert_called_once()

    def test_get_overall_health_status_no_checks(self, health_monitor):
        """Test overall health status when no checks have been run."""
        status, details = health_monitor.get_overall_health_status()

        assert status == "unknown"
        assert "No health checks run" in details["message"]

    def test_get_overall_health_status_outdated_checks(self, health_monitor):
        """Test overall health status with outdated checks."""
        # Add old check timestamp
        old_time = datetime.now() - timedelta(minutes=10)  # Older than 5 minute cutoff
        health_monitor.last_health_check["old_check"] = old_time

        status, details = health_monitor.get_overall_health_status()

        assert status == "warning"
        assert "outdated" in details["message"]
        assert "old_check" in details["outdated_checks"]

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_get_overall_health_status_healthy(
        self, mock_virtual_memory, mock_cpu_percent, health_monitor
    ):
        """Test overall health status with healthy system."""
        # Add recent check timestamp
        recent_time = datetime.now() - timedelta(minutes=1)
        health_monitor.last_health_check["recent_check"] = recent_time

        # Mock healthy system resources
        mock_cpu_percent.return_value = 45.0
        mock_memory = Mock()
        mock_memory.percent = 60.0
        mock_virtual_memory.return_value = mock_memory

        status, details = health_monitor.get_overall_health_status()

        assert status == "healthy"
        assert details["cpu_percent"] == 45.0
        assert details["memory_percent"] == 60.0
        assert "last_health_checks" in details

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_get_overall_health_status_warning(
        self, mock_virtual_memory, mock_cpu_percent, health_monitor
    ):
        """Test overall health status with warning system state."""
        recent_time = datetime.now() - timedelta(minutes=1)
        health_monitor.last_health_check["recent_check"] = recent_time

        # Mock warning-level system resources
        mock_cpu_percent.return_value = 75.0  # Above 70% warning threshold
        mock_memory = Mock()
        mock_memory.percent = 65.0
        mock_virtual_memory.return_value = mock_memory

        status, details = health_monitor.get_overall_health_status()

        assert status == "warning"

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_get_overall_health_status_critical(
        self, mock_virtual_memory, mock_cpu_percent, health_monitor
    ):
        """Test overall health status with critical system state."""
        recent_time = datetime.now() - timedelta(minutes=1)
        health_monitor.last_health_check["recent_check"] = recent_time

        # Mock critical-level system resources
        mock_cpu_percent.return_value = 95.0  # Above 90% critical threshold
        mock_memory = Mock()
        mock_memory.percent = 92.0  # Above 90% critical threshold
        mock_virtual_memory.return_value = mock_memory

        status, details = health_monitor.get_overall_health_status()

        assert status == "critical"

    @patch("psutil.cpu_percent")
    def test_get_overall_health_status_exception(
        self, mock_cpu_percent, health_monitor
    ):
        """Test overall health status when assessment fails."""
        recent_time = datetime.now() - timedelta(minutes=1)
        health_monitor.last_health_check["recent_check"] = recent_time

        mock_cpu_percent.side_effect = Exception("System error")

        status, details = health_monitor.get_overall_health_status()

        assert status == "critical"
        assert "Failed to assess system health" in details["message"]
        assert details["error"] == "System error"


class TestMonitoringManagerAdvanced:
    """Advanced tests for MonitoringManager."""

    @pytest.fixture
    def mock_logger(self):
        with patch("src.utils.monitoring.get_logger") as mock_get_logger:
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            yield mock_logger

    @pytest.fixture
    def mock_metrics_collector(self):
        with patch(
            "src.utils.monitoring.get_metrics_collector"
        ) as mock_get_metrics_collector:
            mock_metrics_collector = Mock()
            mock_get_metrics_collector.return_value = mock_metrics_collector
            yield mock_metrics_collector

    @pytest.fixture
    def monitoring_manager(self, mock_logger, mock_metrics_collector):
        """Create MonitoringManager instance with mocked dependencies."""
        return MonitoringManager()

    def test_monitoring_manager_initialization(self, monitoring_manager):
        """Test MonitoringManager initialization."""
        assert isinstance(monitoring_manager.performance_monitor, PerformanceMonitor)
        assert isinstance(monitoring_manager.health_monitor, SystemHealthMonitor)
        assert monitoring_manager._monitoring_task is None
        assert monitoring_manager._running is False

    @pytest.mark.asyncio
    async def test_start_monitoring_success(self, monitoring_manager, mock_logger):
        """Test starting monitoring successfully."""
        await monitoring_manager.start_monitoring(
            health_check_interval=60, performance_summary_interval=120
        )

        assert monitoring_manager._running is True
        assert monitoring_manager._monitoring_task is not None
        mock_logger.info.assert_called_with("Starting monitoring system")

        # Clean up
        await monitoring_manager.stop_monitoring()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self, monitoring_manager):
        """Test starting monitoring when already running."""
        monitoring_manager._running = True

        await monitoring_manager.start_monitoring()

        # Should not create new task
        assert monitoring_manager._monitoring_task is None

    @pytest.mark.asyncio
    async def test_stop_monitoring_success(self, monitoring_manager, mock_logger):
        """Test stopping monitoring successfully."""
        # Start monitoring first
        await monitoring_manager.start_monitoring()

        # Stop monitoring
        await monitoring_manager.stop_monitoring()

        assert monitoring_manager._running is False
        mock_logger.info.assert_called_with("Stopping monitoring system")

    @pytest.mark.asyncio
    async def test_stop_monitoring_not_running(self, monitoring_manager):
        """Test stopping monitoring when not running."""
        monitoring_manager._running = False

        await monitoring_manager.stop_monitoring()

        # Should handle gracefully
        assert monitoring_manager._running is False

    @pytest.mark.asyncio
    async def test_monitoring_loop_health_checks(self, monitoring_manager):
        """Test monitoring loop health check execution."""
        # Mock health monitor run_health_checks
        mock_health_results = {
            "test_check": HealthCheckResult("test", "healthy", 0.1, "OK")
        }

        with patch.object(
            monitoring_manager.health_monitor,
            "run_health_checks",
            new_callable=AsyncMock,
            return_value=mock_health_results,
        ) as mock_run_checks:

            # Start monitoring with very short intervals for testing
            await monitoring_manager.start_monitoring(
                health_check_interval=0.1,  # 100ms
                performance_summary_interval=0.2,  # 200ms
            )

            # Wait a bit for health checks to run
            await asyncio.sleep(0.15)

            # Stop monitoring
            await monitoring_manager.stop_monitoring()

            # Verify health checks were called
            assert mock_run_checks.call_count >= 1

    @pytest.mark.asyncio
    async def test_monitoring_loop_performance_summary(
        self, monitoring_manager, mock_logger
    ):
        """Test monitoring loop performance summary generation."""
        mock_summary = {"test_metric": {"count": 5, "mean": 0.25}}

        with patch.object(
            monitoring_manager.performance_monitor,
            "get_performance_summary",
            return_value=mock_summary,
        ) as mock_get_summary:

            # Start monitoring with short intervals
            await monitoring_manager.start_monitoring(
                health_check_interval=0.3,  # 300ms
                performance_summary_interval=0.1,  # 100ms
            )

            # Wait for performance summary to be generated
            await asyncio.sleep(0.15)

            # Stop monitoring
            await monitoring_manager.stop_monitoring()

            # Verify performance summary was generated
            assert mock_get_summary.call_count >= 1

            # Verify summary was logged
            log_calls = [
                call
                for call in mock_logger.info.call_args_list
                if "Performance summary generated" in str(call)
            ]
            assert len(log_calls) >= 1

    @pytest.mark.asyncio
    async def test_monitoring_loop_exception_handling(
        self, monitoring_manager, mock_logger
    ):
        """Test monitoring loop exception handling."""
        # Mock health monitor to raise exception
        with patch.object(
            monitoring_manager.health_monitor,
            "run_health_checks",
            new_callable=AsyncMock,
            side_effect=Exception("Health check error"),
        ):

            # Start monitoring
            await monitoring_manager.start_monitoring(
                health_check_interval=0.05, performance_summary_interval=0.1
            )

            # Wait for exception to occur
            await asyncio.sleep(0.1)

            # Stop monitoring
            await monitoring_manager.stop_monitoring()

            # Verify error was logged
            error_calls = [
                call
                for call in mock_logger.error.call_args_list
                if "Error in monitoring loop" in str(call)
            ]
            assert len(error_calls) >= 1

    def test_get_performance_monitor(self, monitoring_manager):
        """Test getting performance monitor instance."""
        perf_monitor = monitoring_manager.get_performance_monitor()
        assert perf_monitor is monitoring_manager.performance_monitor
        assert isinstance(perf_monitor, PerformanceMonitor)

    def test_get_health_monitor(self, monitoring_manager):
        """Test getting health monitor instance."""
        health_monitor = monitoring_manager.get_health_monitor()
        assert health_monitor is monitoring_manager.health_monitor
        assert isinstance(health_monitor, SystemHealthMonitor)

    @pytest.mark.asyncio
    async def test_get_monitoring_status_comprehensive(self, monitoring_manager):
        """Test getting comprehensive monitoring status."""
        # Mock health check results
        mock_health_results = {
            "system_resources": HealthCheckResult(
                "system_resources", "healthy", 0.15, "All systems operational"
            ),
            "disk_space": HealthCheckResult(
                "disk_space", "warning", 0.08, "Disk space at 88%"
            ),
        }

        mock_health_status = ("warning", {"cpu_percent": 75.0, "memory_percent": 65.0})
        mock_performance_summary = {
            "prediction_latency": {"count": 10, "mean": 0.25, "p95": 0.45}
        }

        with patch.object(
            monitoring_manager.health_monitor,
            "run_health_checks",
            new_callable=AsyncMock,
            return_value=mock_health_results,
        ) as mock_run_checks, patch.object(
            monitoring_manager.health_monitor,
            "get_overall_health_status",
            return_value=mock_health_status,
        ) as mock_get_health_status, patch.object(
            monitoring_manager.performance_monitor,
            "get_performance_summary",
            return_value=mock_performance_summary,
        ) as mock_get_perf_summary:

            # Start monitoring to set _running = True
            monitoring_manager._running = True

            status = await monitoring_manager.get_monitoring_status()

            # Verify status structure
            assert "monitoring_active" in status
            assert "health_status" in status
            assert "health_details" in status
            assert "health_checks" in status
            assert "performance_summary" in status
            assert "timestamp" in status

            # Verify monitoring is active
            assert status["monitoring_active"] is True

            # Verify health status
            assert status["health_status"] == "warning"
            assert status["health_details"]["cpu_percent"] == 75.0

            # Verify health checks
            assert len(status["health_checks"]) == 2
            assert status["health_checks"]["system_resources"]["status"] == "healthy"
            assert status["health_checks"]["disk_space"]["status"] == "warning"

            # Verify performance summary
            assert status["performance_summary"] == mock_performance_summary

            # Verify timestamp format
            assert isinstance(status["timestamp"], str)
            datetime.fromisoformat(status["timestamp"])  # Should not raise exception

    def test_get_monitoring_manager_singleton(self):
        """Test monitoring manager singleton behavior."""
        # Clear any existing global state
        import src.utils.monitoring

        original_manager = src.utils.monitoring._monitoring_manager

        try:
            # Reset global state
            src.utils.monitoring._monitoring_manager = None

            manager1 = get_monitoring_manager()
            manager2 = get_monitoring_manager()

            assert manager1 is manager2
            assert isinstance(manager1, MonitoringManager)

        finally:
            # Restore original state
            src.utils.monitoring._monitoring_manager = original_manager


if __name__ == "__main__":
    pytest.main([__file__])
