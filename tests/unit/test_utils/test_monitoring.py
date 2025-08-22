"""
Comprehensive unit tests for monitoring.py.
Tests performance monitoring, health checks, alerting, and system diagnostics.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, Mock, patch

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
    """Test PerformanceThreshold dataclass functionality."""

    def test_performance_threshold_creation(self):
        """Test creating PerformanceThreshold with all fields."""
        threshold = PerformanceThreshold(
            name="cpu_usage",
            warning_threshold=70.0,
            critical_threshold=90.0,
            unit="percent",
            description="CPU usage monitoring",
        )

        assert threshold.name == "cpu_usage"
        assert threshold.warning_threshold == 70.0
        assert threshold.critical_threshold == 90.0
        assert threshold.unit == "percent"
        assert threshold.description == "CPU usage monitoring"


class TestHealthCheckResult:
    """Test HealthCheckResult dataclass functionality."""

    def test_health_check_result_creation(self):
        """Test creating HealthCheckResult with all fields."""
        details = {"connection_count": 10, "last_query": "SELECT 1"}
        result = HealthCheckResult(
            component="database",
            status="healthy",
            response_time=0.05,
            message="Database is responding normally",
            details=details,
        )

        assert result.component == "database"
        assert result.status == "healthy"
        assert result.response_time == 0.05
        assert result.message == "Database is responding normally"
        assert result.details == details

    def test_health_check_result_defaults(self):
        """Test HealthCheckResult with default values."""
        result = HealthCheckResult(
            component="test_component",
            status="warning",
            response_time=1.0,
            message="Test warning message",
        )

        assert result.details is None


class TestAlertEvent:
    """Test AlertEvent dataclass functionality."""

    def test_alert_event_creation(self):
        """Test creating AlertEvent with all fields."""
        timestamp = datetime.now()
        additional_info = {"threshold": 80.0, "current_value": 85.5}

        event = AlertEvent(
            timestamp=timestamp,
            alert_type="warning",
            component="cpu_monitor",
            message="CPU usage exceeded warning threshold",
            metric_name="cpu_usage",
            current_value=85.5,
            threshold=80.0,
            additional_info=additional_info,
        )

        assert event.timestamp == timestamp
        assert event.alert_type == "warning"
        assert event.component == "cpu_monitor"
        assert event.message == "CPU usage exceeded warning threshold"
        assert event.metric_name == "cpu_usage"
        assert event.current_value == 85.5
        assert event.threshold == 80.0
        assert event.additional_info == additional_info

    def test_alert_event_defaults(self):
        """Test AlertEvent with default values."""
        timestamp = datetime.now()
        event = AlertEvent(
            timestamp=timestamp,
            alert_type="critical",
            component="memory_monitor",
            message="High memory usage detected",
            metric_name="memory_usage",
            current_value=95.0,
            threshold=90.0,
        )

        assert event.additional_info is None


class TestPerformanceMonitor:
    """Test PerformanceMonitor functionality."""

    @pytest.fixture
    def performance_monitor(self):
        """Create PerformanceMonitor instance."""
        with patch("src.utils.monitoring.get_logger") as mock_logger, patch(
            "src.utils.monitoring.get_performance_logger"
        ) as mock_perf_logger, patch(
            "src.utils.monitoring.get_error_tracker"
        ) as mock_error, patch(
            "src.utils.monitoring.get_metrics_collector"
        ) as mock_metrics:

            return PerformanceMonitor()

    def test_performance_monitor_initialization(self, performance_monitor):
        """Test PerformanceMonitor initialization."""
        assert isinstance(performance_monitor.performance_history, defaultdict)
        assert isinstance(performance_monitor.alert_callbacks, list)
        assert isinstance(performance_monitor.thresholds, dict)

        # Check that default thresholds are set
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

    def test_add_alert_callback(self, performance_monitor):
        """Test adding alert callback."""

        def test_callback(alert_event):
            pass

        performance_monitor.add_alert_callback(test_callback)
        assert test_callback in performance_monitor.alert_callbacks

    def test_record_performance_metric_no_room_id(self, performance_monitor):
        """Test recording performance metric without room ID."""
        with patch.object(performance_monitor, "_check_threshold") as mock_check:
            performance_monitor.record_performance_metric(
                "prediction_latency", 0.75, additional_info={"model": "lstm"}
            )

        # Check that metric was stored
        assert "prediction_latency" in performance_monitor.performance_history
        history = performance_monitor.performance_history["prediction_latency"]
        assert len(history) == 1
        assert history[0][1] == 0.75  # Value

        # Check that threshold was checked
        mock_check.assert_called_once_with(
            "prediction_latency", 0.75, None, {"model": "lstm"}
        )

    def test_record_performance_metric_with_room_id(self, performance_monitor):
        """Test recording performance metric with room ID."""
        performance_monitor.record_performance_metric(
            "prediction_latency", 1.25, room_id="living_room"
        )

        # Should store with room_id key
        key = "prediction_latency_living_room"
        assert key in performance_monitor.performance_history
        history = performance_monitor.performance_history[key]
        assert len(history) == 1
        assert history[0][1] == 1.25

    def test_record_performance_metric_unknown_threshold(self, performance_monitor):
        """Test recording metric that doesn't have a threshold."""
        performance_monitor.record_performance_metric("unknown_metric", 100.0)

        # Should store metric but not check threshold
        assert "unknown_metric" in performance_monitor.performance_history

    def test_check_threshold_warning(self, performance_monitor):
        """Test threshold checking for warning level."""
        with patch.object(performance_monitor, "_trigger_alert") as mock_trigger:
            # Use CPU usage threshold (warning: 70, critical: 85)
            performance_monitor._check_threshold("cpu_usage", 75.0, "server_room", {})

        mock_trigger.assert_called_once()
        alert = mock_trigger.call_args[0][0]
        assert alert.alert_type == "warning"
        assert alert.metric_name == "cpu_usage"
        assert alert.current_value == 75.0
        assert alert.threshold == 70.0  # Warning threshold

    def test_check_threshold_critical(self, performance_monitor):
        """Test threshold checking for critical level."""
        with patch.object(performance_monitor, "_trigger_alert") as mock_trigger:
            # Use CPU usage threshold with critical value
            performance_monitor._check_threshold("cpu_usage", 90.0, "server_room", {})

        mock_trigger.assert_called_once()
        alert = mock_trigger.call_args[0][0]
        assert alert.alert_type == "critical"
        assert alert.current_value == 90.0
        assert alert.threshold == 85.0  # Critical threshold

    def test_check_threshold_no_alert(self, performance_monitor):
        """Test threshold checking when no alert is needed."""
        with patch.object(performance_monitor, "_trigger_alert") as mock_trigger:
            # Use value below warning threshold
            performance_monitor._check_threshold("cpu_usage", 60.0, None, {})

        # Should not trigger alert
        mock_trigger.assert_not_called()

    def test_trigger_alert(self, performance_monitor):
        """Test triggering alert."""
        # Add a callback
        callback_called = False
        callback_alert = None

        def test_callback(alert):
            nonlocal callback_called, callback_alert
            callback_called = True
            callback_alert = alert

        performance_monitor.add_alert_callback(test_callback)

        alert = AlertEvent(
            timestamp=datetime.now(),
            alert_type="warning",
            component="test_component",
            message="Test alert message",
            metric_name="test_metric",
            current_value=100.0,
            threshold=80.0,
        )

        performance_monitor._trigger_alert(alert)

        # Verify callback was called
        assert callback_called
        assert callback_alert == alert

    def test_trigger_alert_callback_exception(self, performance_monitor):
        """Test alert triggering with callback that raises exception."""

        def failing_callback(alert):
            raise Exception("Callback failed")

        performance_monitor.add_alert_callback(failing_callback)

        alert = AlertEvent(
            timestamp=datetime.now(),
            alert_type="critical",
            component="test",
            message="Test",
            metric_name="test",
            current_value=100.0,
            threshold=50.0,
        )

        # Should not raise exception
        performance_monitor._trigger_alert(alert)

    def test_get_performance_summary_empty(self, performance_monitor):
        """Test getting performance summary with no data."""
        summary = performance_monitor.get_performance_summary(hours=24)
        assert summary == {}

    def test_get_performance_summary_with_data(self, performance_monitor):
        """Test getting performance summary with data."""
        # Add some test data
        key = "test_metric"
        now = datetime.now()

        # Add data within time window
        performance_monitor.performance_history[key].extend(
            [
                (now - timedelta(hours=2), 10.0),
                (now - timedelta(hours=1), 15.0),
                (now - timedelta(minutes=30), 20.0),
                (now - timedelta(minutes=10), 12.0),
            ]
        )

        # Add old data outside time window
        performance_monitor.performance_history[key].append(
            (now - timedelta(hours=25), 5.0)
        )

        summary = performance_monitor.get_performance_summary(hours=24)

        assert key in summary
        stats = summary[key]
        assert stats["count"] == 4  # Should exclude old data
        assert stats["mean"] == statistics.mean([10.0, 15.0, 20.0, 12.0])
        assert stats["median"] == statistics.median([10.0, 15.0, 20.0, 12.0])
        assert stats["min"] == 10.0
        assert stats["max"] == 20.0
        assert "p95" in stats
        assert "p99" in stats

    def test_get_performance_summary_single_value(self, performance_monitor):
        """Test performance summary with single value."""
        key = "single_metric"
        now = datetime.now()

        performance_monitor.performance_history[key].append(
            (now - timedelta(minutes=30), 42.0)
        )

        summary = performance_monitor.get_performance_summary(hours=1)

        assert key in summary
        stats = summary[key]
        assert stats["count"] == 1
        assert stats["mean"] == 42.0
        assert stats["std"] == 0  # Single value has no standard deviation

    def test_percentile_calculation(self, performance_monitor):
        """Test percentile calculation helper method."""
        values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

        # Test various percentiles
        p50 = performance_monitor._percentile(values, 50)
        p90 = performance_monitor._percentile(values, 90)
        p95 = performance_monitor._percentile(values, 95)

        assert p50 == 5.5  # Median of 1-10
        assert p90 == 9.1
        assert p95 == 9.55

    def test_percentile_empty_list(self, performance_monitor):
        """Test percentile calculation with empty list."""
        result = performance_monitor._percentile([], 95)
        assert result == 0.0

    def test_percentile_single_value(self, performance_monitor):
        """Test percentile calculation with single value."""
        result = performance_monitor._percentile([42.0], 95)
        assert result == 42.0

    def test_get_trend_analysis_no_data(self, performance_monitor):
        """Test trend analysis with no data."""
        result = performance_monitor.get_trend_analysis("nonexistent_metric")
        assert result["status"] == "no_data"

    def test_get_trend_analysis_insufficient_data(self, performance_monitor):
        """Test trend analysis with insufficient data."""
        key = "test_metric"
        now = datetime.now()

        # Add single data point
        performance_monitor.performance_history[key].append(
            (now - timedelta(minutes=30), 10.0)
        )

        result = performance_monitor.get_trend_analysis("test_metric", hours=1)
        assert result["status"] == "insufficient_data"

    def test_get_trend_analysis_improving_trend(self, performance_monitor):
        """Test trend analysis with improving (decreasing) trend."""
        key = "test_metric"
        now = datetime.now()

        # Add decreasing values (improving trend for latency-type metrics)
        performance_monitor.performance_history[key].extend(
            [
                (now - timedelta(hours=1), 20.0),
                (now - timedelta(minutes=45), 18.0),
                (now - timedelta(minutes=30), 15.0),
                (now - timedelta(minutes=15), 12.0),
                (now - timedelta(minutes=5), 10.0),
            ]
        )

        result = performance_monitor.get_trend_analysis("test_metric", hours=2)

        assert result["status"] == "success"
        assert result["trend"] == "improving"
        assert result["slope"] < 0
        assert result["data_points"] == 5

    def test_get_trend_analysis_degrading_trend(self, performance_monitor):
        """Test trend analysis with degrading (increasing) trend."""
        key = "test_metric"
        now = datetime.now()

        # Add increasing values (degrading trend)
        performance_monitor.performance_history[key].extend(
            [
                (now - timedelta(hours=1), 10.0),
                (now - timedelta(minutes=45), 12.0),
                (now - timedelta(minutes=30), 15.0),
                (now - timedelta(minutes=15), 18.0),
                (now - timedelta(minutes=5), 20.0),
            ]
        )

        result = performance_monitor.get_trend_analysis("test_metric", hours=2)

        assert result["status"] == "success"
        assert result["trend"] == "degrading"
        assert result["slope"] > 0

    def test_get_trend_analysis_stable_trend(self, performance_monitor):
        """Test trend analysis with stable trend."""
        key = "test_metric"
        now = datetime.now()

        # Add stable values
        stable_value = 15.0
        performance_monitor.performance_history[key].extend(
            [
                (now - timedelta(hours=1), stable_value),
                (now - timedelta(minutes=45), stable_value),
                (now - timedelta(minutes=30), stable_value),
                (now - timedelta(minutes=15), stable_value),
                (now - timedelta(minutes=5), stable_value),
            ]
        )

        result = performance_monitor.get_trend_analysis("test_metric", hours=2)

        assert result["status"] == "success"
        assert result["trend"] == "stable"
        assert abs(result["slope"]) < 0.001  # Nearly zero slope

    def test_get_trend_analysis_with_room_id(self, performance_monitor):
        """Test trend analysis with room ID."""
        key = "test_metric_living_room"
        now = datetime.now()

        performance_monitor.performance_history[key].extend(
            [
                (now - timedelta(minutes=30), 10.0),
                (now - timedelta(minutes=15), 15.0),
                (now - timedelta(minutes=5), 20.0),
            ]
        )

        result = performance_monitor.get_trend_analysis(
            "test_metric", room_id="living_room", hours=1
        )

        assert result["status"] == "success"
        assert result["data_points"] == 3


class TestSystemHealthMonitor:
    """Test SystemHealthMonitor functionality."""

    @pytest.fixture
    def health_monitor(self):
        """Create SystemHealthMonitor instance."""
        with patch("src.utils.monitoring.get_logger") as mock_logger, patch(
            "src.utils.monitoring.get_metrics_collector"
        ) as mock_metrics:

            return SystemHealthMonitor()

    def test_health_monitor_initialization(self, health_monitor):
        """Test SystemHealthMonitor initialization."""
        assert isinstance(health_monitor.health_checks, dict)
        assert isinstance(health_monitor.last_health_check, dict)

        # Check that default health checks are registered
        expected_checks = [
            "system_resources",
            "disk_space",
            "memory_usage",
            "cpu_usage",
        ]

        for check_name in expected_checks:
            assert check_name in health_monitor.health_checks

    def test_register_health_check(self, health_monitor):
        """Test registering custom health check."""

        def custom_check():
            return HealthCheckResult(
                component="custom_component",
                status="healthy",
                response_time=0.1,
                message="Custom check passed",
            )

        health_monitor.register_health_check("custom_check", custom_check)
        assert "custom_check" in health_monitor.health_checks

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_system_resources_healthy(
        self, mock_memory, mock_cpu, health_monitor
    ):
        """Test system resources check with healthy values."""
        mock_cpu.return_value = 25.0

        mock_memory_info = Mock()
        mock_memory_info.percent = 40.0
        mock_memory_info.available = 8 * 1024**3
        mock_memory_info.total = 16 * 1024**3
        mock_memory.return_value = mock_memory_info

        result = health_monitor._check_system_resources()

        assert result.component == "system_resources"
        assert result.status == "healthy"
        assert "normal" in result.message.lower()
        assert result.details["cpu_percent"] == 25.0
        assert result.details["memory_percent"] == 40.0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_system_resources_warning(
        self, mock_memory, mock_cpu, health_monitor
    ):
        """Test system resources check with warning values."""
        mock_cpu.return_value = 75.0  # High CPU

        mock_memory_info = Mock()
        mock_memory_info.percent = 75.0  # High memory
        mock_memory_info.available = 4 * 1024**3
        mock_memory_info.total = 16 * 1024**3
        mock_memory.return_value = mock_memory_info

        result = health_monitor._check_system_resources()

        assert result.component == "system_resources"
        assert result.status == "warning"
        assert "elevated" in result.message.lower() or "high" in result.message.lower()

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_check_system_resources_critical(
        self, mock_memory, mock_cpu, health_monitor
    ):
        """Test system resources check with critical values."""
        mock_cpu.return_value = 95.0  # Critical CPU

        mock_memory_info = Mock()
        mock_memory_info.percent = 95.0  # Critical memory
        mock_memory_info.available = 1 * 1024**3
        mock_memory_info.total = 16 * 1024**3
        mock_memory.return_value = mock_memory_info

        result = health_monitor._check_system_resources()

        assert result.component == "system_resources"
        assert result.status == "critical"
        assert "high" in result.message.lower()

    @patch("psutil.cpu_percent")
    def test_check_system_resources_exception(self, mock_cpu, health_monitor):
        """Test system resources check with exception."""
        mock_cpu.side_effect = Exception("PSUtil error")

        result = health_monitor._check_system_resources()

        assert result.component == "system_resources"
        assert result.status == "critical"
        assert "failed to check" in result.message.lower()
        assert "error" in result.details

    @patch("psutil.disk_usage")
    def test_check_disk_space_healthy(self, mock_disk_usage, health_monitor):
        """Test disk space check with healthy values."""
        mock_usage = Mock()
        mock_usage.total = 1024**4  # 1TB
        mock_usage.used = 400 * 1024**3  # 400GB (40% used)
        mock_usage.free = 624 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = health_monitor._check_disk_space()

        assert result.component == "disk_space"
        assert result.status == "healthy"
        assert "healthy" in result.message.lower()
        assert result.details["percent"] == 40.0

    @patch("psutil.disk_usage")
    def test_check_disk_space_warning(self, mock_disk_usage, health_monitor):
        """Test disk space check with warning values."""
        mock_usage = Mock()
        mock_usage.total = 1024**4  # 1TB
        mock_usage.used = 900 * 1024**3  # 900GB (90% used)
        mock_usage.free = 124 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = health_monitor._check_disk_space()

        assert result.component == "disk_space"
        assert result.status == "warning"
        assert "low" in result.message.lower()

    @patch("psutil.disk_usage")
    def test_check_disk_space_critical(self, mock_disk_usage, health_monitor):
        """Test disk space check with critical values."""
        mock_usage = Mock()
        mock_usage.total = 1024**4  # 1TB
        mock_usage.used = 970 * 1024**3  # 970GB (97% used)
        mock_usage.free = 54 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = health_monitor._check_disk_space()

        assert result.component == "disk_space"
        assert result.status == "critical"
        assert "critical" in result.message.lower()

    @patch("psutil.disk_usage")
    def test_check_disk_space_exception(self, mock_disk_usage, health_monitor):
        """Test disk space check with exception."""
        mock_disk_usage.side_effect = Exception("Disk access error")

        result = health_monitor._check_disk_space()

        assert result.component == "disk_space"
        assert result.status == "critical"
        assert "failed to check" in result.message.lower()

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_check_memory_usage_normal(
        self, mock_virtual_memory, mock_process, health_monitor
    ):
        """Test memory usage check with normal values."""
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 2 * 1024**3  # 2GB
        mock_process.return_value = mock_process_instance

        mock_virtual_memory.return_value.total = 16 * 1024**3  # 16GB

        result = health_monitor._check_memory_usage()

        assert result.component == "memory_usage"
        assert result.status == "healthy"
        assert "normal" in result.message.lower()
        assert result.details["process_percent"] == 12.5  # 2/16 * 100

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_check_memory_usage_warning(
        self, mock_virtual_memory, mock_process, health_monitor
    ):
        """Test memory usage check with warning values."""
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 8 * 1024**3  # 8GB
        mock_process.return_value = mock_process_instance

        mock_virtual_memory.return_value.total = 16 * 1024**3  # 16GB (50% usage)

        result = health_monitor._check_memory_usage()

        assert result.component == "memory_usage"
        assert result.status == "warning"
        assert "high" in result.message.lower() or "moderate" in result.message.lower()

    @patch("psutil.Process")
    @patch("psutil.virtual_memory")
    def test_check_memory_usage_exception(
        self, mock_virtual_memory, mock_process, health_monitor
    ):
        """Test memory usage check with exception."""
        mock_process.side_effect = Exception("Process error")

        result = health_monitor._check_memory_usage()

        assert result.component == "memory_usage"
        assert result.status == "critical"
        assert "failed to check" in result.message.lower()

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_check_cpu_usage_normal(
        self, mock_cpu_count, mock_cpu_percent, health_monitor
    ):
        """Test CPU usage check with normal values."""
        mock_cpu_percent.return_value = 35.0
        mock_cpu_count.return_value = 4

        result = health_monitor._check_cpu_usage()

        assert result.component == "cpu_usage"
        assert result.status == "healthy"
        assert "normal" in result.message.lower()
        assert result.details["cpu_percent"] == 35.0
        assert result.details["cpu_count"] == 4

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_check_cpu_usage_warning(
        self, mock_cpu_count, mock_cpu_percent, health_monitor
    ):
        """Test CPU usage check with warning values."""
        mock_cpu_percent.return_value = 75.0  # High CPU
        mock_cpu_count.return_value = 8

        result = health_monitor._check_cpu_usage()

        assert result.component == "cpu_usage"
        assert result.status == "warning"
        assert "high" in result.message.lower()

    @patch("psutil.cpu_percent")
    @patch("psutil.cpu_count")
    def test_check_cpu_usage_critical(
        self, mock_cpu_count, mock_cpu_percent, health_monitor
    ):
        """Test CPU usage check with critical values."""
        mock_cpu_percent.return_value = 95.0  # Critical CPU
        mock_cpu_count.return_value = 2

        result = health_monitor._check_cpu_usage()

        assert result.component == "cpu_usage"
        assert result.status == "critical"
        assert "critical" in result.message.lower()

    @patch("psutil.cpu_percent")
    def test_check_cpu_usage_exception(self, mock_cpu_percent, health_monitor):
        """Test CPU usage check with exception."""
        mock_cpu_percent.side_effect = Exception("CPU monitoring error")

        result = health_monitor._check_cpu_usage()

        assert result.component == "cpu_usage"
        assert result.status == "critical"
        assert "failed to check" in result.message.lower()

    @pytest.mark.asyncio
    async def test_run_health_checks(self, health_monitor):
        """Test running all health checks."""

        # Mock health check functions to return predictable results
        def mock_system_check():
            return HealthCheckResult("system", "healthy", 0.1, "System OK")

        def mock_disk_check():
            return HealthCheckResult("disk", "warning", 0.2, "Disk space low")

        health_monitor.health_checks = {
            "system": mock_system_check,
            "disk": mock_disk_check,
        }

        results = await health_monitor.run_health_checks()

        assert len(results) == 2
        assert "system" in results
        assert "disk" in results
        assert results["system"].status == "healthy"
        assert results["disk"].status == "warning"

        # Check that last check times were recorded
        assert "system" in health_monitor.last_health_check
        assert "disk" in health_monitor.last_health_check

    @pytest.mark.asyncio
    async def test_run_health_checks_exception(self, health_monitor):
        """Test running health checks when one fails with exception."""

        def working_check():
            return HealthCheckResult("working", "healthy", 0.1, "OK")

        def failing_check():
            raise Exception("Check failed")

        health_monitor.health_checks = {
            "working": working_check,
            "failing": failing_check,
        }

        results = await health_monitor.run_health_checks()

        assert len(results) == 2
        assert results["working"].status == "healthy"
        assert results["failing"].status == "critical"
        assert "failed" in results["failing"].message.lower()

    def test_get_overall_health_status_no_checks(self, health_monitor):
        """Test overall health status with no checks run."""
        status, details = health_monitor.get_overall_health_status()

        assert status == "unknown"
        assert "No health checks run" in details["message"]

    def test_get_overall_health_status_outdated_checks(self, health_monitor):
        """Test overall health status with outdated checks."""
        # Add outdated check
        old_time = datetime.now() - timedelta(minutes=10)
        health_monitor.last_health_check["old_check"] = old_time

        status, details = health_monitor.get_overall_health_status()

        assert status == "warning"
        assert "outdated" in details["message"].lower()
        assert "old_check" in details["outdated_checks"]

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_get_overall_health_status_healthy(
        self, mock_memory, mock_cpu, health_monitor
    ):
        """Test overall health status with healthy system."""
        mock_cpu.return_value = 30.0
        mock_memory.return_value.percent = 40.0

        # Add recent check
        health_monitor.last_health_check["test_check"] = datetime.now()

        status, details = health_monitor.get_overall_health_status()

        assert status == "healthy"
        assert details["cpu_percent"] == 30.0
        assert details["memory_percent"] == 40.0

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_get_overall_health_status_warning(
        self, mock_memory, mock_cpu, health_monitor
    ):
        """Test overall health status with warning conditions."""
        mock_cpu.return_value = 75.0  # Warning level
        mock_memory.return_value.percent = 50.0

        health_monitor.last_health_check["test_check"] = datetime.now()

        status, details = health_monitor.get_overall_health_status()

        assert status == "warning"

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    def test_get_overall_health_status_critical(
        self, mock_memory, mock_cpu, health_monitor
    ):
        """Test overall health status with critical conditions."""
        mock_cpu.return_value = 95.0  # Critical level
        mock_memory.return_value.percent = 95.0  # Critical level

        health_monitor.last_health_check["test_check"] = datetime.now()

        status, details = health_monitor.get_overall_health_status()

        assert status == "critical"

    @patch("psutil.cpu_percent")
    def test_get_overall_health_status_exception(self, mock_cpu, health_monitor):
        """Test overall health status with exception."""
        mock_cpu.side_effect = Exception("System error")

        health_monitor.last_health_check["test_check"] = datetime.now()

        status, details = health_monitor.get_overall_health_status()

        assert status == "critical"
        assert "Failed to assess" in details["message"]


class TestMonitoringManager:
    """Test MonitoringManager functionality."""

    @pytest.fixture
    def monitoring_manager(self):
        """Create MonitoringManager instance."""
        with patch("src.utils.monitoring.get_logger") as mock_logger, patch(
            "src.utils.monitoring.get_metrics_collector"
        ) as mock_metrics:

            return MonitoringManager()

    def test_monitoring_manager_initialization(self, monitoring_manager):
        """Test MonitoringManager initialization."""
        assert isinstance(monitoring_manager.performance_monitor, PerformanceMonitor)
        assert isinstance(monitoring_manager.health_monitor, SystemHealthMonitor)
        assert monitoring_manager._monitoring_task is None
        assert monitoring_manager._running is False

    @pytest.mark.asyncio
    async def test_start_monitoring(self, monitoring_manager):
        """Test starting monitoring system."""
        with patch("asyncio.create_task") as mock_create_task:
            await monitoring_manager.start_monitoring(
                health_check_interval=60, performance_summary_interval=300
            )

            assert monitoring_manager._running is True
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_running(self, monitoring_manager):
        """Test starting monitoring when already running."""
        monitoring_manager._running = True

        with patch("asyncio.create_task") as mock_create_task:
            await monitoring_manager.start_monitoring()

            # Should not create new task
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, monitoring_manager):
        """Test stopping monitoring system."""
        # Setup running monitoring
        monitoring_manager._running = True
        mock_task = Mock()
        mock_task.cancel = Mock()
        monitoring_manager._monitoring_task = mock_task

        await monitoring_manager.stop_monitoring()

        assert monitoring_manager._running is False
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_loop_health_checks(self, monitoring_manager):
        """Test monitoring loop running health checks."""
        monitoring_manager._running = True

        with patch.object(
            monitoring_manager.health_monitor, "run_health_checks"
        ) as mock_health_check, patch("time.time") as mock_time, patch(
            "asyncio.sleep"
        ) as mock_sleep:

            # Mock time progression to trigger health check
            mock_time.side_effect = [
                0,
                70,
                140,
            ]  # First call, then 70s later, then 140s
            mock_sleep.side_effect = [
                None,
                asyncio.CancelledError(),
            ]  # Cancel after first iteration

            try:
                await monitoring_manager._monitoring_loop(60, 300)
            except asyncio.CancelledError:
                pass

            # Should have run health check
            mock_health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_loop_performance_summary(self, monitoring_manager):
        """Test monitoring loop generating performance summary."""
        monitoring_manager._running = True

        with patch.object(
            monitoring_manager.performance_monitor, "get_performance_summary"
        ) as mock_summary, patch("time.time") as mock_time, patch(
            "asyncio.sleep"
        ) as mock_sleep:

            mock_summary.return_value = {"test_metric": {"count": 10}}

            # Mock time progression to trigger performance summary
            mock_time.side_effect = [0, 310, 620]  # Trigger performance summary
            mock_sleep.side_effect = [None, asyncio.CancelledError()]

            try:
                await monitoring_manager._monitoring_loop(60, 300)
            except asyncio.CancelledError:
                pass

            # Should have generated performance summary
            mock_summary.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitoring_loop_exception_handling(self, monitoring_manager):
        """Test monitoring loop exception handling."""
        monitoring_manager._running = True

        with patch.object(
            monitoring_manager.health_monitor, "run_health_checks"
        ) as mock_health_check, patch("asyncio.sleep") as mock_sleep:

            mock_health_check.side_effect = Exception("Health check failed")
            mock_sleep.side_effect = [None, asyncio.CancelledError()]

            try:
                await monitoring_manager._monitoring_loop(60, 300)
            except asyncio.CancelledError:
                pass

            # Should continue running despite exception
            mock_health_check.assert_called()

    def test_get_performance_monitor(self, monitoring_manager):
        """Test getting performance monitor instance."""
        monitor = monitoring_manager.get_performance_monitor()
        assert monitor is monitoring_manager.performance_monitor

    def test_get_health_monitor(self, monitoring_manager):
        """Test getting health monitor instance."""
        monitor = monitoring_manager.get_health_monitor()
        assert monitor is monitoring_manager.health_monitor

    @pytest.mark.asyncio
    async def test_get_monitoring_status(self, monitoring_manager):
        """Test getting monitoring status."""
        # Mock health check results
        mock_health_results = {
            "system_resources": HealthCheckResult("system", "healthy", 0.1, "OK"),
            "disk_space": HealthCheckResult("disk", "warning", 0.2, "Low space"),
        }

        with patch.object(
            monitoring_manager.health_monitor, "run_health_checks"
        ) as mock_health_check, patch.object(
            monitoring_manager.health_monitor, "get_overall_health_status"
        ) as mock_overall, patch.object(
            monitoring_manager.performance_monitor, "get_performance_summary"
        ) as mock_summary:

            mock_health_check.return_value = mock_health_results
            mock_overall.return_value = ("healthy", {"cpu_percent": 30.0})
            mock_summary.return_value = {"metric1": {"count": 5}}

            status = await monitoring_manager.get_monitoring_status()

            assert "monitoring_active" in status
            assert "health_status" in status
            assert "health_details" in status
            assert "health_checks" in status
            assert "performance_summary" in status
            assert "timestamp" in status

            assert status["monitoring_active"] == monitoring_manager._running
            assert status["health_status"] == "healthy"
            assert len(status["health_checks"]) == 2


class TestGlobalMonitoringManager:
    """Test global monitoring manager functionality."""

    def test_get_monitoring_manager_singleton(self):
        """Test that get_monitoring_manager returns singleton instance."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            manager1 = get_monitoring_manager()
            manager2 = get_monitoring_manager()

            assert manager1 is manager2
            assert isinstance(manager1, MonitoringManager)

    @patch("src.utils.monitoring._monitoring_manager", None)
    def test_get_monitoring_manager_creates_new_instance(self):
        """Test that get_monitoring_manager creates new instance when none exists."""
        with patch("src.utils.monitoring.get_logger"), patch(
            "src.utils.monitoring.get_metrics_collector"
        ):

            # Reset global instance
            import src.utils.monitoring

            src.utils.monitoring._monitoring_manager = None

            manager = get_monitoring_manager()
            assert isinstance(manager, MonitoringManager)


if __name__ == "__main__":
    pytest.main([__file__])
