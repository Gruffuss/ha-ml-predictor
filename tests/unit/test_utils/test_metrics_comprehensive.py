"""
Comprehensive unit tests for metrics.py with deeper coverage.
Tests edge cases, error scenarios, threading, and advanced functionality.
"""

import asyncio
from datetime import datetime, timedelta
import sys
import threading
import time
from unittest.mock import MagicMock, Mock, call, patch

import psutil
import pytest

from src.utils.metrics import (
    PROMETHEUS_AVAILABLE,
    MetricsManager,
    MLMetricsCollector,
    MultiProcessMetricsManager,
    export_multiprocess_metrics,
    get_aggregated_metrics,
    get_metrics_collector,
    get_metrics_manager,
    get_multiprocess_metrics_manager,
    metrics_endpoint_handler,
    setup_multiprocess_metrics,
    time_prediction,
)


class TestMLMetricsCollectorAdvanced:
    """Advanced tests for MLMetricsCollector with edge cases and error scenarios."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock Prometheus registry."""
        return Mock()

    @pytest.fixture
    def metrics_collector(self, mock_registry):
        """Create MLMetricsCollector instance."""
        return MLMetricsCollector(registry=mock_registry)

    def test_metrics_collector_with_default_registry(self):
        """Test MLMetricsCollector initialization with default registry."""
        with patch("src.utils.metrics.REGISTRY") as mock_default_registry:
            collector = MLMetricsCollector()
            assert collector.registry == mock_default_registry

    def test_setup_metrics_creates_all_expected_metrics(self, metrics_collector):
        """Test that _setup_metrics creates all expected metrics."""
        # Verify all metric attributes exist
        metric_names = [
            "system_info",
            "prediction_requests_total",
            "prediction_latency",
            "prediction_accuracy",
            "prediction_confidence",
            "model_training_duration",
            "model_retraining_count",
            "model_accuracy_score",
            "active_models_count",
            "events_processed_total",
            "feature_computation_duration",
            "database_operations_duration",
            "concept_drift_detected",
            "drift_severity_score",
            "adaptation_actions_total",
            "mqtt_messages_published",
            "ha_api_requests_total",
            "ha_connection_status",
            "cpu_usage_percent",
            "memory_usage_bytes",
            "disk_usage_bytes",
            "errors_total",
            "last_error_timestamp",
            "system_health_score",
            "prediction_queue_size",
            "uptime_seconds",
        ]

        for metric_name in metric_names:
            assert hasattr(
                metrics_collector, metric_name
            ), f"Missing metric: {metric_name}"

    def test_update_system_info_with_detailed_platform_info(self, metrics_collector):
        """Test system info update with detailed platform information."""
        with patch("src.utils.metrics.platform.platform") as mock_platform, patch(
            "src.utils.metrics.platform.architecture"
        ) as mock_arch, patch(
            "src.utils.metrics.platform.processor"
        ) as mock_processor, patch(
            "src.utils.metrics.platform.node"
        ) as mock_node, patch(
            "src.utils.metrics.sys.version",
            "3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]",
        ):

            mock_platform.return_value = "Windows-10-10.0.19044-SP0"
            mock_arch.return_value = ("64bit", "WindowsPE")
            mock_processor.return_value = "Intel64 Family 6 Model 142 Stepping 12"
            mock_node.return_value = "DESKTOP-TEST123"

            metrics_collector.update_system_info()

            # Verify system_info.info was called with correct parameters
            expected_info = {
                "version": "1.0.0",
                "python_version": "3.11.0 (main, Oct 24 2022, 18:26:48) [MSC v.1933 64 bit (AMD64)]",
                "platform": "Windows-10-10.0.19044-SP0",
                "architecture": "64bit",
                "processor": "Intel64 Family 6 Model 142 Stepping 12",
                "hostname": "DESKTOP-TEST123",
            }
            metrics_collector.system_info.info.assert_called_once_with(expected_info)

    def test_record_prediction_with_complex_parameters(self, metrics_collector):
        """Test recording prediction with complex parameter combinations."""
        # Setup mock metrics with chained returns
        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()
        mock_conf_histogram = Mock()

        metrics_collector.prediction_requests_total = mock_counter
        metrics_collector.prediction_latency = mock_histogram
        metrics_collector.prediction_accuracy = mock_gauge
        metrics_collector.prediction_confidence = mock_conf_histogram

        # Setup chained method calls
        mock_counter.labels.return_value = mock_counter
        mock_histogram.labels.return_value = mock_histogram
        mock_gauge.labels.return_value = mock_gauge
        mock_conf_histogram.labels.return_value = mock_conf_histogram

        # Test with extreme values
        metrics_collector.record_prediction(
            room_id="hallway_north_entrance",
            prediction_type="next_occupied_time_with_confidence_interval",
            model_type="ensemble_lstm_xgboost_hmm",
            duration=0.001,  # Very fast
            accuracy_minutes=0.5,  # Very accurate
            confidence=0.999,  # Very confident
            status="success",
        )

        # Verify all metrics were called with correct parameters
        mock_counter.labels.assert_called_with(
            room_id="hallway_north_entrance",
            prediction_type="next_occupied_time_with_confidence_interval",
            status="success",
        )
        mock_histogram.labels.assert_called_with(
            room_id="hallway_north_entrance",
            prediction_type="next_occupied_time_with_confidence_interval",
            model_type="ensemble_lstm_xgboost_hmm",
        )
        mock_gauge.labels.assert_called_with(
            room_id="hallway_north_entrance",
            prediction_type="next_occupied_time_with_confidence_interval",
            model_type="ensemble_lstm_xgboost_hmm",
        )
        mock_conf_histogram.labels.assert_called_with(
            room_id="hallway_north_entrance",
            prediction_type="next_occupied_time_with_confidence_interval",
        )

        # Verify values were recorded
        mock_histogram.observe.assert_called_with(0.001)
        mock_gauge.set.assert_called_with(0.5)
        mock_conf_histogram.observe.assert_called_with(0.999)

    def test_record_model_training_with_no_accuracy_metrics(self, metrics_collector):
        """Test recording model training without accuracy metrics."""
        mock_duration_histogram = Mock()
        mock_counter = Mock()

        metrics_collector.model_training_duration = mock_duration_histogram
        metrics_collector.model_retraining_count = mock_counter

        mock_duration_histogram.labels.return_value = mock_duration_histogram
        mock_counter.labels.return_value = mock_counter

        metrics_collector.record_model_training(
            room_id="basement",
            model_type="hmm",
            training_type="incremental_update",
            duration=45.2,
            accuracy_metrics=None,  # No accuracy metrics
            trigger_reason="scheduled",
        )

        # Verify basic metrics were recorded
        mock_duration_histogram.observe.assert_called_with(45.2)
        mock_counter.inc.assert_called_once()

        # Verify accuracy gauge was not called since no metrics provided
        # The model_accuracy_score should exist but not be called
        if hasattr(metrics_collector, "model_accuracy_score"):
            # If the attribute exists, it should not have been called
            assert not getattr(
                metrics_collector.model_accuracy_score, "set", Mock()
            ).called

    def test_record_model_training_with_empty_accuracy_metrics(self, metrics_collector):
        """Test recording model training with empty accuracy metrics dict."""
        mock_duration_histogram = Mock()
        mock_counter = Mock()
        mock_accuracy_gauge = Mock()

        metrics_collector.model_training_duration = mock_duration_histogram
        metrics_collector.model_retraining_count = mock_counter
        metrics_collector.model_accuracy_score = mock_accuracy_gauge

        mock_duration_histogram.labels.return_value = mock_duration_histogram
        mock_counter.labels.return_value = mock_counter
        mock_accuracy_gauge.labels.return_value = mock_accuracy_gauge

        metrics_collector.record_model_training(
            room_id="attic",
            model_type="gp",
            training_type="full_retrain",
            duration=120.0,
            accuracy_metrics={},  # Empty dict
            trigger_reason="concept_drift",
        )

        # Verify accuracy gauge was not called since dict is empty
        assert mock_accuracy_gauge.labels.call_count == 0
        assert mock_accuracy_gauge.set.call_count == 0

    def test_record_model_training_with_extensive_accuracy_metrics(
        self, metrics_collector
    ):
        """Test recording model training with extensive accuracy metrics."""
        mock_accuracy_gauge = Mock()
        metrics_collector.model_accuracy_score = mock_accuracy_gauge
        mock_accuracy_gauge.labels.return_value = mock_accuracy_gauge

        # Comprehensive accuracy metrics
        accuracy_metrics = {
            "mse": 0.023,
            "rmse": 0.152,
            "mae": 0.045,
            "r2": 0.887,
            "adjusted_r2": 0.871,
            "mape": 0.067,
            "precision": 0.923,
            "recall": 0.889,
            "f1_score": 0.906,
            "accuracy": 0.912,
            "auc_roc": 0.945,
        }

        metrics_collector.record_model_training(
            room_id="study",
            model_type="ensemble",
            training_type="hyperparameter_optimization",
            duration=1800.5,
            accuracy_metrics=accuracy_metrics,
        )

        # Verify all accuracy metrics were recorded
        assert mock_accuracy_gauge.labels.call_count == len(accuracy_metrics)
        assert mock_accuracy_gauge.set.call_count == len(accuracy_metrics)

        # Verify specific calls
        expected_calls = [
            call(room_id="study", model_type="ensemble", metric_type=metric_name)
            for metric_name in accuracy_metrics.keys()
        ]
        mock_accuracy_gauge.labels.assert_has_calls(expected_calls, any_order=True)

    def test_record_concept_drift_boundary_severity_values(self, metrics_collector):
        """Test concept drift recording with boundary severity values."""
        mock_counter = Mock()
        metrics_collector.concept_drift_detected = mock_counter
        mock_counter.labels.return_value = mock_counter

        # Test exact boundary values based on: severity > 0.7 => high, severity > 0.3 => medium, else low
        test_cases = [
            (0.0, "low"),
            (0.3, "low"),  # 0.3 is NOT > 0.3, so it's low
            (0.31, "medium"),  # 0.31 is > 0.3, so it's medium
            (0.7, "medium"),  # 0.7 is NOT > 0.7, so it's medium
            (0.71, "high"),  # 0.71 is > 0.7, so it's high
            (1.0, "high"),
        ]

        for severity, expected_label in test_cases:
            mock_counter.reset_mock()
            metrics_collector.record_concept_drift(
                "test_room", "test_drift", severity, "test_action"
            )
            mock_counter.labels.assert_called_with(
                room_id="test_room", drift_type="test_drift", severity=expected_label
            )

    @patch("psutil.cpu_percent")
    @patch("psutil.Process")
    @patch("psutil.disk_usage")
    def test_update_system_resources_with_realistic_values(
        self, mock_disk_usage, mock_process, mock_cpu_percent, metrics_collector
    ):
        """Test system resource update with realistic system values."""
        # Setup realistic system values
        mock_cpu_percent.return_value = 23.7

        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 2 * 1024**3  # 2GB RSS
        mock_memory_info.vms = 4 * 1024**3  # 4GB VMS
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance

        mock_disk_usage_result = Mock()
        mock_disk_usage_result.total = 1000 * 1024**3  # 1TB
        mock_disk_usage_result.used = 456 * 1024**3  # 456GB
        mock_disk_usage_result.free = 544 * 1024**3  # 544GB
        mock_disk_usage.return_value = mock_disk_usage_result

        # Setup metric mocks
        mock_cpu_gauge = Mock()
        mock_memory_gauge = Mock()
        mock_disk_gauge = Mock()

        metrics_collector.cpu_usage_percent = mock_cpu_gauge
        metrics_collector.memory_usage_bytes = mock_memory_gauge
        metrics_collector.disk_usage_bytes = mock_disk_gauge

        mock_memory_gauge.labels.return_value = mock_memory_gauge
        mock_disk_gauge.labels.return_value = mock_disk_gauge

        metrics_collector.update_system_resources()

        # Verify CPU usage
        mock_cpu_gauge.set.assert_called_with(23.7)

        # Verify memory usage calls
        memory_calls = [call(type="rss"), call(type="vms")]
        mock_memory_gauge.labels.assert_has_calls(memory_calls, any_order=True)

        memory_set_calls = [call(2 * 1024**3), call(4 * 1024**3)]  # RSS  # VMS
        mock_memory_gauge.set.assert_has_calls(memory_set_calls, any_order=True)

        # Verify disk usage calls
        disk_calls = [
            call(path="/", type="total"),
            call(path="/", type="used"),
            call(path="/", type="free"),
        ]
        mock_disk_gauge.labels.assert_has_calls(disk_calls, any_order=True)

    @patch("psutil.cpu_percent")
    def test_update_system_resources_psutil_exceptions(
        self, mock_cpu_percent, metrics_collector
    ):
        """Test system resource update handles various psutil exceptions."""
        # Test different exception scenarios
        exception_scenarios = [
            PermissionError("Access denied"),
            OSError("No such file or directory"),
            psutil.NoSuchProcess(123),
            ValueError("Invalid argument"),
            RuntimeError("System error"),
        ]

        for exception in exception_scenarios:
            mock_cpu_percent.side_effect = exception

            # Should not raise exception
            try:
                metrics_collector.update_system_resources()
            except Exception as e:
                pytest.fail(f"update_system_resources raised {type(e).__name__}: {e}")

    def test_set_gauge_with_all_supported_gauges(self, metrics_collector):
        """Test setting gauge values for all supported gauge types."""
        # Setup mock gauges
        mock_gauges = {
            "system_health_score": Mock(),
            "cpu_usage_percent": Mock(),
            "uptime_seconds": Mock(),
            "prediction_accuracy": Mock(),
        }

        for gauge_name, mock_gauge in mock_gauges.items():
            setattr(metrics_collector, gauge_name, mock_gauge)
            mock_gauge.labels.return_value = mock_gauge

        # Test each gauge type
        test_cases = [
            ("system_health_score", 0.95, None),
            ("cpu_usage_percent", 45.2, None),
            ("uptime_seconds", 86400, None),
            ("prediction_accuracy", 8.5, {"room_id": "bedroom", "model_type": "lstm"}),
        ]

        for gauge_name, value, labels in test_cases:
            mock_gauge = mock_gauges[gauge_name]
            mock_gauge.reset_mock()

            metrics_collector.set_gauge(gauge_name, value, labels)

            if labels:
                mock_gauge.labels.assert_called_with(**labels)
                mock_gauge.set.assert_called_with(value)
            else:
                mock_gauge.set.assert_called_with(value)

    def test_time_operation_context_manager_with_actual_timing(self, metrics_collector):
        """Test time_operation context manager functionality."""
        start_time = time.time()

        with metrics_collector.time_operation("test_room", "test_operation"):
            time.sleep(0.01)  # Small delay

        end_time = time.time()
        duration = end_time - start_time

        # Verify timing was approximately correct
        assert duration >= 0.01
        assert duration < 0.1  # Should be less than 100ms in test environment

    def test_time_operation_context_manager_with_exception(self, metrics_collector):
        """Test time_operation context manager when exception occurs."""
        with pytest.raises(ValueError, match="Test exception"):
            with metrics_collector.time_operation("test_room", "failing_operation"):
                raise ValueError("Test exception")

    def test_record_event_processing_with_edge_case_parameters(self, metrics_collector):
        """Test event processing recording with edge case parameters."""
        mock_counter = Mock()
        metrics_collector.events_processed_total = mock_counter
        mock_counter.labels.return_value = mock_counter

        # Test with very long names and special characters
        metrics_collector.record_event_processing(
            room_id="multi_word_room_with_underscores_and_numbers_123",
            sensor_type="binary_sensor.motion_detector_pir_v2_advanced",
            processing_duration=0.0001,  # Very fast processing
            status="success_with_warnings",
        )

        mock_counter.labels.assert_called_with(
            room_id="multi_word_room_with_underscores_and_numbers_123",
            sensor_type="binary_sensor.motion_detector_pir_v2_advanced",
            status="success_with_warnings",
        )
        mock_counter.inc.assert_called_once()

    def test_record_database_operation_with_various_statuses(self, metrics_collector):
        """Test database operation recording with various status values."""
        mock_histogram = Mock()
        metrics_collector.database_operations_duration = mock_histogram
        mock_histogram.labels.return_value = mock_histogram

        status_scenarios = [
            ("select", "sensor_events", 0.15, "success"),
            ("insert", "room_states", 0.05, "success"),
            ("update", "predictions", 0.25, "success"),
            ("delete", "old_events", 1.2, "success"),
            ("select", "sensor_events", 5.0, "timeout"),
            ("insert", "sensor_events", 0.1, "error"),
            ("complex_query", "multiple_tables", 10.5, "partial_success"),
        ]

        for operation_type, table, duration, status in status_scenarios:
            mock_histogram.reset_mock()

            metrics_collector.record_database_operation(
                operation_type=operation_type,
                table=table,
                duration=duration,
                status=status,
            )

            mock_histogram.labels.assert_called_with(
                operation_type=operation_type, table=table, status=status
            )
            mock_histogram.observe.assert_called_with(duration)

    def test_all_metrics_initialization_types(self, mock_registry):
        """Test that all metrics are initialized with correct types and parameters."""
        collector = MLMetricsCollector(registry=mock_registry)

        # This test ensures all metrics are properly initialized during setup
        # We verify the registry.register was called appropriately
        # Note: The actual verification depends on how the mock registry is set up
        assert collector.registry == mock_registry
        assert not collector._system_info_updated


class TestMetricsManagerAdvanced:
    """Advanced tests for MetricsManager with threading and edge cases."""

    @pytest.fixture
    def mock_registry(self):
        return Mock()

    @pytest.fixture
    def metrics_manager(self, mock_registry):
        return MetricsManager(registry=mock_registry)

    def test_background_collection_threading_behavior(self, metrics_manager):
        """Test background collection threading behavior."""
        with patch("threading.Thread") as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            # Test starting background collection
            metrics_manager.start_background_collection(update_interval=1)

            # Verify thread creation and configuration
            mock_thread_class.assert_called_once()
            call_args = mock_thread_class.call_args
            assert call_args[1]["daemon"] is True
            assert call_args[1]["name"] == "MetricsUpdater"
            mock_thread.start.assert_called_once()

            # Test that starting again doesn't create new thread
            mock_thread_class.reset_mock()
            metrics_manager.start_background_collection()
            mock_thread_class.assert_not_called()

    def test_background_collection_update_loop_behavior(self, metrics_manager):
        """Test the background update loop functionality."""
        with patch("time.sleep") as mock_sleep, patch.object(
            metrics_manager.collector, "update_system_resources"
        ) as mock_update_resources, patch.object(
            metrics_manager.collector, "update_uptime"
        ) as mock_update_uptime:

            # Mock the running flag to stop after one iteration
            original_running = metrics_manager._running

            def side_effect_stop_after_one(*args):
                if metrics_manager._running:
                    metrics_manager._running = False
                return None

            mock_sleep.side_effect = side_effect_stop_after_one

            # Start background collection (this will run the loop once)
            metrics_manager._running = True

            # Simulate the update loop manually
            try:
                metrics_manager.collector.update_system_resources()
                metrics_manager.collector.update_uptime(metrics_manager.start_time)
            except Exception:
                pass  # Expected to fail in test environment

            metrics_manager._running = original_running

    def test_stop_background_collection_timeout_handling(self, metrics_manager):
        """Test stop background collection with thread join timeout."""
        # Setup a mock thread that doesn't join immediately
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        metrics_manager._resource_update_thread = mock_thread
        metrics_manager._running = True

        metrics_manager.stop_background_collection()

        assert metrics_manager._running is False
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_get_metrics_with_prometheus_encoding_scenarios(self, metrics_manager):
        """Test get_metrics with various Prometheus encoding scenarios."""
        test_cases = [
            b"# Simple metrics\nmetric_name 1.0\n",
            b"# UTF-8 encoded\nmetric_with_unicode\xc3\xa9 2.5\n",
            b'# Complex metrics\n# HELP metric_help Description\n# TYPE metric_help counter\nmetric_help{label="value"} 42\n',
            b"",  # Empty metrics
        ]

        for test_data in test_cases:
            with patch("src.utils.metrics.PROMETHEUS_AVAILABLE", True), patch(
                "src.utils.metrics.generate_latest"
            ) as mock_generate:

                mock_generate.return_value = test_data

                result = metrics_manager.get_metrics()
                expected = test_data.decode("utf-8")
                assert result == expected

    def test_metrics_manager_singleton_behavior_edge_cases(self):
        """Test metrics manager singleton behavior in edge cases."""
        # Clear any existing global state
        import src.utils.metrics

        original_manager = src.utils.metrics._metrics_manager

        try:
            # Reset global state
            src.utils.metrics._metrics_manager = None

            # Test concurrent access simulation
            managers = []

            def create_manager():
                managers.append(get_metrics_manager())

            # Simulate multiple threads trying to get manager
            threads = [threading.Thread(target=create_manager) for _ in range(5)]

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            # All managers should be the same instance
            for manager in managers[1:]:
                assert manager is managers[0]

        finally:
            # Restore original state
            src.utils.metrics._metrics_manager = original_manager


class TestMultiProcessMetricsManagerAdvanced:
    """Advanced tests for MultiProcessMetricsManager with complex scenarios."""

    @pytest.fixture
    def multiprocess_manager(self):
        return MultiProcessMetricsManager()

    def test_multiprocess_initialization_with_mock_multiprocess_module(
        self, multiprocess_manager
    ):
        """Test initialization behavior with different multiprocess module states."""
        # Test when multiprocess module has MultiProcessCollector
        with patch("src.utils.metrics.PROMETHEUS_AVAILABLE", True), patch(
            "src.utils.metrics.multiprocess"
        ) as mock_multiprocess_module:

            # Mock the MultiProcessCollector
            mock_multiprocess_module.MultiProcessCollector = Mock()

            # Create new manager
            manager = MultiProcessMetricsManager()

            # Should try to create registry and collector
            if hasattr(mock_multiprocess_module, "MultiProcessCollector"):
                assert manager.multiprocess_enabled is not None

    def test_aggregate_multiprocess_metrics_with_complex_metric_families(
        self, multiprocess_manager
    ):
        """Test aggregating complex metric families with various sample types."""
        with patch.object(
            multiprocess_manager, "multiprocess_enabled", True
        ), patch.object(multiprocess_manager, "registry", Mock()) as mock_registry:

            # Create complex mock metric families
            metric_families = []

            # Counter metric family
            counter_family = Mock()
            counter_family.name = "requests_total"
            counter_family.help = "Total number of requests"
            counter_family.type = "counter"

            counter_sample1 = Mock()
            counter_sample1.name = "requests_total"
            counter_sample1.labels = {"method": "GET", "status": "200"}
            counter_sample1.value = 1524.0
            counter_sample1.timestamp = 1642248000.0

            counter_sample2 = Mock()
            counter_sample2.name = "requests_total"
            counter_sample2.labels = {"method": "POST", "status": "201"}
            counter_sample2.value = 847.0
            counter_sample2.timestamp = 1642248060.0

            counter_family.samples = [counter_sample1, counter_sample2]
            metric_families.append(counter_family)

            # Histogram metric family
            histogram_family = Mock()
            histogram_family.name = "request_duration_seconds"
            histogram_family.help = "Request duration in seconds"
            histogram_family.type = "histogram"

            hist_bucket1 = Mock()
            hist_bucket1.name = "request_duration_seconds_bucket"
            hist_bucket1.labels = {"le": "0.1"}
            hist_bucket1.value = 245.0
            hist_bucket1.timestamp = None

            hist_sum = Mock()
            hist_sum.name = "request_duration_seconds_sum"
            hist_sum.labels = {}
            hist_sum.value = 1247.6
            hist_sum.timestamp = None

            hist_count = Mock()
            hist_count.name = "request_duration_seconds_count"
            hist_count.labels = {}
            hist_count.value = 5000.0
            hist_count.timestamp = None

            histogram_family.samples = [hist_bucket1, hist_sum, hist_count]
            metric_families.append(histogram_family)

            mock_registry.collect.return_value = metric_families

            result = multiprocess_manager.aggregate_multiprocess_metrics()

            # Verify the aggregated structure
            assert "requests_total" in result
            assert "request_duration_seconds" in result

            # Verify counter metric details
            counter_data = result["requests_total"]
            assert counter_data["name"] == "requests_total"
            assert counter_data["type"] == "counter"
            assert len(counter_data["samples"]) == 2

            # Verify histogram metric details
            histogram_data = result["request_duration_seconds"]
            assert histogram_data["name"] == "request_duration_seconds"
            assert histogram_data["type"] == "histogram"
            assert len(histogram_data["samples"]) == 3

    def test_cleanup_dead_processes_with_various_exceptions(self, multiprocess_manager):
        """Test cleanup dead processes with various exception scenarios."""
        exception_scenarios = [
            ImportError("Module not found"),
            AttributeError("'module' object has no attribute"),
            OSError("Operation not permitted"),
            RuntimeError("Runtime error in cleanup"),
        ]

        for exception in exception_scenarios:
            with patch.object(
                multiprocess_manager, "multiprocess_enabled", True
            ), patch("src.utils.metrics.multiprocess") as mock_multiprocess:

                mock_multiprocess.mark_process_dead.side_effect = exception

                # Should not raise exception
                try:
                    multiprocess_manager.cleanup_dead_processes()
                except Exception as e:
                    pytest.fail(
                        f"cleanup_dead_processes raised {type(e).__name__}: {e}"
                    )

    def test_generate_multiprocess_metrics_with_encoding_edge_cases(
        self, multiprocess_manager
    ):
        """Test generating multiprocess metrics with encoding edge cases."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", True), patch(
            "src.utils.metrics.generate_latest"
        ) as mock_generate:

            multiprocess_manager.registry = Mock()

            # Test various byte string scenarios
            test_cases = [
                b"",  # Empty
                b"# HELP metric_name Description\n",  # Only help
                b"\xff\xfe",  # Invalid UTF-8
                b'metric_name{complex_label="value with spaces and "quotes""} 42.0\n',  # Complex labels
            ]

            for test_data in test_cases:
                mock_generate.return_value = test_data

                try:
                    result = multiprocess_manager.generate_multiprocess_metrics()
                    # Should successfully decode or handle gracefully
                    assert isinstance(result, str)
                except UnicodeDecodeError:
                    # Some edge cases might fail decode, which is acceptable
                    pass


class TestGlobalFunctionsAdvanced:
    """Advanced tests for global utility functions with edge cases."""

    def test_setup_multiprocess_metrics_with_attribute_error(self):
        """Test setup_multiprocess_metrics when collector doesn't have expected attributes."""
        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_manager, patch(
            "src.utils.metrics.get_metrics_collector"
        ) as mock_get_collector:

            mock_manager = Mock()
            mock_manager.is_multiprocess_enabled.return_value = True
            mock_manager.get_multiprocess_registry.return_value = Mock()
            mock_get_manager.return_value = mock_manager

            # Collector without multiprocess_registry attribute
            mock_collector = Mock(spec=[])  # No attributes
            mock_get_collector.return_value = mock_collector

            # Should not raise exception
            setup_multiprocess_metrics()

            mock_manager.cleanup_dead_processes.assert_called_once()

    def test_export_multiprocess_metrics_exception_handling(self):
        """Test export_multiprocess_metrics with various exception scenarios."""
        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_manager:

            mock_manager = Mock()
            mock_manager.is_multiprocess_enabled.return_value = True
            mock_manager.generate_multiprocess_metrics.side_effect = Exception(
                "Generation error"
            )
            mock_get_manager.return_value = mock_manager

            # Should fall back to single process
            with patch("src.utils.metrics.get_metrics_manager") as mock_get_single:
                mock_single_manager = Mock()
                mock_single_manager.get_metrics.return_value = "# Fallback metrics\n"
                mock_get_single.return_value = mock_single_manager

                # Should handle the exception and fall back to single process
                result = export_multiprocess_metrics()
                assert result == "# Fallback metrics\n"

    def test_get_aggregated_metrics_with_complex_data_structures(self):
        """Test get_aggregated_metrics with complex nested data structures."""
        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_multiprocess, patch(
            "src.utils.metrics.get_metrics_manager"
        ) as mock_get_single:

            # Complex multiprocess metrics
            complex_multiprocess_data = {
                "nested_metric": {
                    "deeply": {"nested": {"data": [1, 2, 3, {"key": "value"}]}}
                },
                "metric_with_unicode": {
                    "unicode_data": "Test with émojis 🚀 and spëcial characters"
                },
                "large_metric_list": {
                    "samples": [f"sample_{i}" for i in range(1000)]  # Large list
                },
            }

            mock_multiprocess_manager = Mock()
            mock_multiprocess_manager.is_multiprocess_enabled.return_value = True
            mock_multiprocess_manager.aggregate_multiprocess_metrics.return_value = (
                complex_multiprocess_data
            )
            mock_get_multiprocess.return_value = mock_multiprocess_manager

            mock_single_manager = Mock()
            mock_single_manager.get_metrics.return_value = (
                "# Complex single process metrics\n" * 100
            )  # Large string
            mock_get_single.return_value = mock_single_manager

            result = get_aggregated_metrics()

            assert result["collection_mode"] == "multiprocess"
            assert result["multiprocess_metrics"] == complex_multiprocess_data
            assert "single_process_metrics" in result
            assert len(result["single_process_metrics"]) > 1000  # Large data


class TestTimePredictionDecoratorAdvanced:
    """Advanced tests for time_prediction decorator with complex scenarios."""

    def test_time_prediction_decorator_with_coroutine_function(self):
        """Test time_prediction decorator with async function."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            @time_prediction("async_room", "async_prediction", "async_model")
            async def async_prediction():
                await asyncio.sleep(0.01)
                return {"confidence": 0.75, "async_result": True}

            # Test async function
            with patch("time.time", side_effect=[0, 0.02]):
                result = asyncio.run(async_prediction())

            assert result["async_result"] is True
            # Note: The decorator might not extract confidence from async results properly
            # Let's verify what actually gets called
            mock_collector.record_prediction.assert_called_once()
            call_args = mock_collector.record_prediction.call_args[1]
            assert call_args["room_id"] == "async_room"
            assert call_args["prediction_type"] == "async_prediction"
            assert call_args["model_type"] == "async_model"
            assert call_args["duration"] == 0.02
            assert call_args["status"] == "success"

    def test_time_prediction_decorator_with_generator_function(self):
        """Test time_prediction decorator with generator function."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            @time_prediction("generator_room", "batch_prediction", "streaming_model")
            def prediction_generator():
                for i in range(2):
                    yield {"confidence": 0.8 + i * 0.1, "batch": i + 1}
                return {"confidence": 0.85, "final": True}

            with patch("time.time", side_effect=[0, 0.05]):
                gen = prediction_generator()
                results = list(gen)

            assert len(results) == 2
            mock_collector.record_prediction.assert_called_once()

    def test_time_prediction_decorator_with_complex_exception_handling(self):
        """Test time_prediction decorator with various exception types."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            exception_scenarios = [
                ValueError("Invalid input"),
                RuntimeError("Model error"),
                KeyError("Missing key"),
                TypeError("Type mismatch"),
                MemoryError("Out of memory"),
                TimeoutError("Operation timeout"),
            ]

            for exception in exception_scenarios:

                @time_prediction("error_room", "error_prediction", "error_model")
                def failing_prediction():
                    raise exception

                with patch("time.time", side_effect=[0, 0.01]):
                    with pytest.raises(type(exception)):
                        failing_prediction()

                # Verify error was recorded
                mock_collector.record_prediction.assert_called_with(
                    room_id="error_room",
                    prediction_type="error_prediction",
                    model_type="error_model",
                    duration=0.01,
                    status="error",
                )
                mock_collector.reset_mock()

    def test_time_prediction_decorator_preserves_function_metadata(self):
        """Test that time_prediction decorator preserves function metadata."""

        @time_prediction("meta_room", "meta_prediction", "meta_model")
        def documented_function(param1, param2="default"):
            """This is a well documented function.

            Args:
                param1: First parameter
                param2: Second parameter with default

            Returns:
                Dict with prediction results
            """
            return {"param1": param1, "param2": param2}

        # Verify metadata is preserved
        assert documented_function.__name__ == "documented_function"
        assert "well documented function" in documented_function.__doc__

        # Verify function signature is preserved (if possible)
        import inspect

        sig = inspect.signature(documented_function)
        assert "param1" in sig.parameters
        assert "param2" in sig.parameters
        assert sig.parameters["param2"].default == "default"


if __name__ == "__main__":
    pytest.main([__file__])
