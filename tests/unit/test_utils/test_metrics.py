"""
Comprehensive unit tests for metrics.py.
Tests Prometheus metrics collection, ML operations tracking, and system monitoring.
"""

from datetime import datetime
import sys
import threading
import time
from unittest.mock import Mock, patch

import platform
import pytest

from src.utils.metrics import (
    MetricsManager,
    MLMetricsCollector,
    MultiProcessMetricsManager,
    get_aggregated_metrics,
    get_metrics_collector,
    get_metrics_manager,
    get_multiprocess_metrics_manager,
    metrics_endpoint_handler,
    setup_multiprocess_metrics,
    time_prediction,
)


class TestMLMetricsCollector:
    """Test MLMetricsCollector functionality."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock Prometheus registry."""
        return Mock()

    @pytest.fixture
    def metrics_collector(self, mock_registry):
        """Create MLMetricsCollector instance."""
        return MLMetricsCollector(registry=mock_registry)

    def test_metrics_collector_initialization(self, metrics_collector, mock_registry):
        """Test MLMetricsCollector initialization."""
        assert metrics_collector.registry == mock_registry
        assert hasattr(metrics_collector, "system_info")
        assert hasattr(metrics_collector, "prediction_requests_total")
        assert hasattr(metrics_collector, "prediction_latency")
        assert hasattr(metrics_collector, "prediction_accuracy")
        assert hasattr(metrics_collector, "model_training_duration")
        assert hasattr(metrics_collector, "events_processed_total")
        assert hasattr(metrics_collector, "errors_total")
        assert metrics_collector._system_info_updated is False

    def test_update_system_info_first_call(self, metrics_collector):
        """Test updating system info on first call."""
        with patch("platform.platform") as mock_platform, patch(
            "platform.architecture"
        ) as mock_arch, patch("platform.processor") as mock_processor, patch(
            "platform.node"
        ) as mock_node:

            mock_platform.return_value = "Linux-5.4.0-x86_64"
            mock_arch.return_value = ("64bit", "ELF")
            mock_processor.return_value = "x86_64"
            mock_node.return_value = "test-hostname"

            # First call should update
            metrics_collector.update_system_info()
            assert metrics_collector._system_info_updated is True
            metrics_collector.system_info.info.assert_called_once()

            # Second call should not update again
            metrics_collector.system_info.info.reset_mock()
            metrics_collector.update_system_info()
            metrics_collector.system_info.info.assert_not_called()

    def test_record_prediction_success(self, metrics_collector):
        """Test recording successful prediction metrics."""
        # Setup mock metrics
        mock_counter = Mock()
        mock_histogram = Mock()
        mock_gauge = Mock()
        mock_conf_histogram = Mock()

        metrics_collector.prediction_requests_total = mock_counter
        metrics_collector.prediction_latency = mock_histogram
        metrics_collector.prediction_accuracy = mock_gauge
        metrics_collector.prediction_confidence = mock_conf_histogram

        mock_counter.labels.return_value = mock_counter
        mock_histogram.labels.return_value = mock_histogram
        mock_gauge.labels.return_value = mock_gauge
        mock_conf_histogram.labels.return_value = mock_conf_histogram

        metrics_collector.record_prediction(
            room_id="living_room",
            prediction_type="next_occupancy",
            model_type="lstm",
            duration=0.25,
            accuracy_minutes=12.5,
            confidence=0.85,
            status="success",
        )

        # Verify counter increment
        mock_counter.labels.assert_called_with(
            room_id="living_room", prediction_type="next_occupancy", status="success"
        )
        mock_counter.inc.assert_called_once()

        # Verify latency observation
        mock_histogram.labels.assert_called_with(
            room_id="living_room", prediction_type="next_occupancy", model_type="lstm"
        )
        mock_histogram.observe.assert_called_with(0.25)

        # Verify accuracy gauge
        mock_gauge.labels.assert_called_with(
            room_id="living_room", prediction_type="next_occupancy", model_type="lstm"
        )
        mock_gauge.set.assert_called_with(12.5)

        # Verify confidence histogram
        mock_conf_histogram.labels.assert_called_with(
            room_id="living_room", prediction_type="next_occupancy"
        )
        mock_conf_histogram.observe.assert_called_with(0.85)

    def test_record_prediction_minimal(self, metrics_collector):
        """Test recording prediction with minimal parameters."""
        mock_counter = Mock()
        mock_histogram = Mock()

        metrics_collector.prediction_requests_total = mock_counter
        metrics_collector.prediction_latency = mock_histogram

        mock_counter.labels.return_value = mock_counter
        mock_histogram.labels.return_value = mock_histogram

        metrics_collector.record_prediction(
            room_id="bedroom",
            prediction_type="next_vacancy",
            model_type="xgboost",
            duration=0.15,
        )

        # Should use default status
        mock_counter.labels.assert_called_with(
            room_id="bedroom", prediction_type="next_vacancy", status="success"
        )

        # Should not call gauge or confidence metrics since they weren't provided
        assert (
            not hasattr(metrics_collector.prediction_accuracy, "set")
            or not metrics_collector.prediction_accuracy.set.called
        )

    def test_record_model_training(self, metrics_collector):
        """Test recording model training metrics."""
        mock_duration_histogram = Mock()
        mock_counter = Mock()
        mock_accuracy_gauge = Mock()

        metrics_collector.model_training_duration = mock_duration_histogram
        metrics_collector.model_retraining_count = mock_counter
        metrics_collector.model_accuracy_score = mock_accuracy_gauge

        mock_duration_histogram.labels.return_value = mock_duration_histogram
        mock_counter.labels.return_value = mock_counter
        mock_accuracy_gauge.labels.return_value = mock_accuracy_gauge

        accuracy_metrics = {"mse": 0.05, "r2": 0.92, "mae": 0.03}

        metrics_collector.record_model_training(
            room_id="kitchen",
            model_type="lstm",
            training_type="full_retrain",
            duration=180.5,
            accuracy_metrics=accuracy_metrics,
            trigger_reason="concept_drift",
        )

        # Verify duration histogram
        mock_duration_histogram.labels.assert_called_with(
            room_id="kitchen", model_type="lstm", training_type="full_retrain"
        )
        mock_duration_histogram.observe.assert_called_with(180.5)

        # Verify retraining counter
        mock_counter.labels.assert_called_with(
            room_id="kitchen", model_type="lstm", trigger_reason="concept_drift"
        )
        mock_counter.inc.assert_called_once()

        # Verify accuracy metrics
        assert mock_accuracy_gauge.labels.call_count == 3
        assert mock_accuracy_gauge.set.call_count == 3

    def test_record_concept_drift(self, metrics_collector):
        """Test recording concept drift metrics."""
        mock_counter = Mock()
        mock_gauge = Mock()
        mock_adaptation_counter = Mock()

        metrics_collector.concept_drift_detected = mock_counter
        metrics_collector.drift_severity_score = mock_gauge
        metrics_collector.adaptation_actions_total = mock_adaptation_counter

        mock_counter.labels.return_value = mock_counter
        mock_gauge.labels.return_value = mock_gauge
        mock_adaptation_counter.labels.return_value = mock_adaptation_counter

        metrics_collector.record_concept_drift(
            room_id="living_room",
            drift_type="statistical",
            severity=0.75,  # High severity
            action_taken="model_retrain",
        )

        # Verify drift counter with severity label
        mock_counter.labels.assert_called_with(
            room_id="living_room", drift_type="statistical", severity="high"
        )
        mock_counter.inc.assert_called_once()

        # Verify severity gauge
        mock_gauge.labels.assert_called_with(
            room_id="living_room", drift_type="statistical"
        )
        mock_gauge.set.assert_called_with(0.75)

        # Verify adaptation action counter
        mock_adaptation_counter.labels.assert_called_with(
            room_id="living_room", action_type="model_retrain", trigger="concept_drift"
        )
        mock_adaptation_counter.inc.assert_called_once()

    def test_record_concept_drift_severity_classification(self, metrics_collector):
        """Test concept drift severity classification."""
        mock_counter = Mock()
        metrics_collector.concept_drift_detected = mock_counter
        mock_counter.labels.return_value = mock_counter

        # Test low severity (< 0.3)
        metrics_collector.record_concept_drift("room1", "drift1", 0.2, "action1")
        mock_counter.labels.assert_called_with(
            room_id="room1", drift_type="drift1", severity="low"
        )

        # Test medium severity (0.3 <= x <= 0.7)
        metrics_collector.record_concept_drift("room1", "drift1", 0.5, "action1")
        mock_counter.labels.assert_called_with(
            room_id="room1", drift_type="drift1", severity="medium"
        )

        # Test high severity (> 0.7)
        metrics_collector.record_concept_drift("room1", "drift1", 0.8, "action1")
        mock_counter.labels.assert_called_with(
            room_id="room1", drift_type="drift1", severity="high"
        )

    def test_record_event_processing(self, metrics_collector):
        """Test recording event processing metrics."""
        mock_counter = Mock()
        metrics_collector.events_processed_total = mock_counter
        mock_counter.labels.return_value = mock_counter

        metrics_collector.record_event_processing(
            room_id="bathroom",
            sensor_type="motion",
            processing_duration=0.05,
            status="success",
        )

        mock_counter.labels.assert_called_with(
            room_id="bathroom", sensor_type="motion", status="success"
        )
        mock_counter.inc.assert_called_once()

    def test_record_feature_computation(self, metrics_collector):
        """Test recording feature computation metrics."""
        mock_histogram = Mock()
        metrics_collector.feature_computation_duration = mock_histogram
        mock_histogram.labels.return_value = mock_histogram

        metrics_collector.record_feature_computation(
            room_id="office", feature_type="temporal", duration=0.12
        )

        mock_histogram.labels.assert_called_with(
            room_id="office", feature_type="temporal"
        )
        mock_histogram.observe.assert_called_with(0.12)

    def test_record_database_operation(self, metrics_collector):
        """Test recording database operation metrics."""
        mock_histogram = Mock()
        metrics_collector.database_operations_duration = mock_histogram
        mock_histogram.labels.return_value = mock_histogram

        metrics_collector.record_database_operation(
            operation_type="select",
            table="sensor_events",
            duration=0.25,
            status="success",
        )

        mock_histogram.labels.assert_called_with(
            operation_type="select", table="sensor_events", status="success"
        )
        mock_histogram.observe.assert_called_with(0.25)

    def test_record_mqtt_publish(self, metrics_collector):
        """Test recording MQTT publish metrics."""
        mock_counter = Mock()
        metrics_collector.mqtt_messages_published = mock_counter
        mock_counter.labels.return_value = mock_counter

        metrics_collector.record_mqtt_publish(
            topic_type="prediction", room_id="garage", status="success"
        )

        mock_counter.labels.assert_called_with(
            topic_type="prediction", room_id="garage", status="success"
        )
        mock_counter.inc.assert_called_once()

    def test_record_ha_api_request(self, metrics_collector):
        """Test recording Home Assistant API request metrics."""
        mock_counter = Mock()
        metrics_collector.ha_api_requests_total = mock_counter
        mock_counter.labels.return_value = mock_counter

        metrics_collector.record_ha_api_request(
            endpoint="/api/states", method="GET", status="200"
        )

        mock_counter.labels.assert_called_with(
            endpoint="/api/states", method="GET", status="200"
        )
        mock_counter.inc.assert_called_once()

    def test_update_ha_connection_status(self, metrics_collector):
        """Test updating Home Assistant connection status."""
        mock_gauge = Mock()
        metrics_collector.ha_connection_status = mock_gauge
        mock_gauge.labels.return_value = mock_gauge

        # Test connected
        metrics_collector.update_ha_connection_status("websocket", True)
        mock_gauge.labels.assert_called_with(connection_type="websocket")
        mock_gauge.set.assert_called_with(1)

        # Test disconnected
        mock_gauge.set.reset_mock()
        metrics_collector.update_ha_connection_status("websocket", False)
        mock_gauge.set.assert_called_with(0)

    def test_record_error(self, metrics_collector):
        """Test recording error metrics."""
        mock_counter = Mock()
        mock_gauge = Mock()

        metrics_collector.errors_total = mock_counter
        metrics_collector.last_error_timestamp = mock_gauge

        mock_counter.labels.return_value = mock_counter
        mock_gauge.labels.return_value = mock_gauge

        with patch("time.time", return_value=1642248000.0):  # Fixed timestamp
            metrics_collector.record_error(
                error_type="prediction_failure",
                component="lstm_model",
                severity="critical",
            )

        # Verify error counter
        mock_counter.labels.assert_called_with(
            error_type="prediction_failure", component="lstm_model", severity="critical"
        )
        mock_counter.inc.assert_called_once()

        # Verify timestamp gauge
        mock_gauge.labels.assert_called_with(
            error_type="prediction_failure", component="lstm_model"
        )
        mock_gauge.set.assert_called_with(1642248000.0)

    @patch("psutil.cpu_percent")
    @patch("psutil.Process")
    @patch("psutil.disk_usage")
    def test_update_system_resources(
        self, mock_disk_usage, mock_process, mock_cpu_percent, metrics_collector
    ):
        """Test updating system resource metrics."""
        # Setup mocks
        mock_cpu_percent.return_value = 45.5

        mock_process_instance = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 1024 * 1024 * 512  # 512MB
        mock_memory_info.vms = 1024 * 1024 * 1024  # 1GB
        mock_process_instance.memory_info.return_value = mock_memory_info
        mock_process.return_value = mock_process_instance

        mock_disk_usage_result = Mock()
        mock_disk_usage_result.total = 1024**4  # 1TB
        mock_disk_usage_result.used = 512 * 1024**3  # 512GB
        mock_disk_usage_result.free = 512 * 1024**3  # 512GB
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
        mock_cpu_gauge.set.assert_called_with(45.5)

        # Verify memory usage
        assert mock_memory_gauge.labels.call_count == 2  # rss and vms
        assert mock_memory_gauge.set.call_count == 2

        # Verify disk usage
        assert mock_disk_gauge.labels.call_count == 3  # total, used, free
        assert mock_disk_gauge.set.call_count == 3

    def test_update_system_resources_exception_handling(self, metrics_collector):
        """Test that system resource update handles exceptions gracefully."""
        with patch("psutil.cpu_percent", side_effect=Exception("PSUtil error")):
            # Should not raise exception
            metrics_collector.update_system_resources()

    def test_update_active_models_count(self, metrics_collector):
        """Test updating active models count."""
        mock_gauge = Mock()
        metrics_collector.active_models_count = mock_gauge
        mock_gauge.labels.return_value = mock_gauge

        metrics_collector.update_active_models_count(
            room_id="bedroom", model_type="lstm", count=3
        )

        mock_gauge.labels.assert_called_with(room_id="bedroom", model_type="lstm")
        mock_gauge.set.assert_called_with(3)

    def test_update_prediction_queue_size(self, metrics_collector):
        """Test updating prediction queue size."""
        mock_gauge = Mock()
        metrics_collector.prediction_queue_size = mock_gauge
        mock_gauge.labels.return_value = mock_gauge

        metrics_collector.update_prediction_queue_size("priority", 15)

        mock_gauge.labels.assert_called_with(queue_type="priority")
        mock_gauge.set.assert_called_with(15)

    def test_update_system_health_score(self, metrics_collector):
        """Test updating system health score with bounds checking."""
        mock_gauge = Mock()
        metrics_collector.system_health_score = mock_gauge

        # Test normal value
        metrics_collector.update_system_health_score(0.85)
        mock_gauge.set.assert_called_with(0.85)

        # Test value above 1.0 (should be clamped)
        metrics_collector.update_system_health_score(1.5)
        mock_gauge.set.assert_called_with(1.0)

        # Test negative value (should be clamped)
        metrics_collector.update_system_health_score(-0.2)
        mock_gauge.set.assert_called_with(0.0)

    def test_update_uptime(self, metrics_collector):
        """Test updating uptime metric."""
        mock_gauge = Mock()
        metrics_collector.uptime_seconds = mock_gauge

        start_time = datetime(2024, 1, 15, 10, 0, 0)
        current_time = datetime(2024, 1, 15, 12, 30, 0)  # 2.5 hours later
        expected_uptime = 2.5 * 3600  # 2.5 hours in seconds

        with patch("src.utils.metrics.datetime") as mock_datetime:
            mock_datetime.now.return_value = current_time
            metrics_collector.update_uptime(start_time)

        mock_gauge.set.assert_called_with(expected_uptime)

    def test_set_gauge_with_labels(self, metrics_collector):
        """Test setting gauge value with labels."""
        mock_gauge = Mock()
        mock_gauge.labels.return_value = mock_gauge

        # Mock the gauge mapping
        metrics_collector.prediction_accuracy = mock_gauge

        metrics_collector.set_gauge(
            "prediction_accuracy",
            12.5,
            labels={"room_id": "kitchen", "model_type": "lstm"},
        )

        mock_gauge.labels.assert_called_with(room_id="kitchen", model_type="lstm")
        mock_gauge.set.assert_called_with(12.5)

    def test_set_gauge_without_labels(self, metrics_collector):
        """Test setting gauge value without labels."""
        mock_gauge = Mock()

        # Mock the gauge mapping
        metrics_collector.system_health_score = mock_gauge

        metrics_collector.set_gauge("system_health_score", 0.92)

        mock_gauge.set.assert_called_with(0.92)

    def test_set_gauge_unknown_gauge(self, metrics_collector):
        """Test setting gauge value for unknown gauge name."""
        # Should not raise exception for unknown gauge
        metrics_collector.set_gauge("unknown_gauge", 100)

    def test_set_gauge_exception_handling(self, metrics_collector):
        """Test that set_gauge handles exceptions gracefully."""
        mock_gauge = Mock()
        mock_gauge.set.side_effect = Exception("Gauge error")

        metrics_collector.system_health_score = mock_gauge

        # Should not raise exception
        metrics_collector.set_gauge("system_health_score", 0.5)

    def test_time_operation_context_manager(self, metrics_collector):
        """Test time_operation context manager."""
        with metrics_collector.time_operation("test_room", "test_operation"):
            time.sleep(0.01)  # Small delay to ensure time passes

        # Should complete without error
        # Note: The actual timing functionality is basic in the implementation

    def test_prometheus_available_true(self):
        """Test behavior when Prometheus client is available."""
        # This test assumes prometheus_client is available in the test environment
        from src.utils.metrics import PROMETHEUS_AVAILABLE

        # If prometheus_client is available, PROMETHEUS_AVAILABLE should be True
        if PROMETHEUS_AVAILABLE:
            # Test that actual Prometheus classes are used
            collector = MLMetricsCollector()
            assert hasattr(collector.prediction_requests_total, "inc")
            assert hasattr(collector.prediction_latency, "observe")

    def test_mock_classes_when_prometheus_unavailable(self):
        """Test mock classes behavior when prometheus is not available."""
        # Import the mock classes directly
        from src.utils.metrics import _MockCounter, _MockGauge, _MockHistogram

        # Test mock counter
        mock_counter = _MockCounter()
        mock_counter.inc()  # Should not raise exception
        assert mock_counter.labels() == mock_counter  # Should return self

        # Test mock gauge
        mock_gauge = _MockGauge()
        mock_gauge.set(10)  # Should not raise exception
        mock_gauge.inc()  # Should not raise exception
        mock_gauge.dec()  # Should not raise exception
        assert mock_gauge.labels() == mock_gauge

        # Test mock histogram
        mock_histogram = _MockHistogram()
        mock_histogram.observe(1.5)  # Should not raise exception
        assert mock_histogram.labels() == mock_histogram

        # Test time context manager
        with mock_histogram.time():
            pass  # Should not raise exception


class TestMetricsManager:
    """Test MetricsManager functionality."""

    @pytest.fixture
    def mock_registry(self):
        """Create mock registry."""
        return Mock()

    @pytest.fixture
    def metrics_manager(self, mock_registry):
        """Create MetricsManager instance."""
        return MetricsManager(registry=mock_registry)

    def test_metrics_manager_initialization(self, metrics_manager, mock_registry):
        """Test MetricsManager initialization."""
        assert metrics_manager.registry == mock_registry
        assert isinstance(metrics_manager.collector, MLMetricsCollector)
        assert isinstance(metrics_manager.start_time, datetime)
        assert metrics_manager._resource_update_thread is None
        assert metrics_manager._running is False

    def test_start_background_collection(self, metrics_manager):
        """Test starting background metrics collection."""
        with patch("threading.Thread") as mock_thread_class:
            mock_thread = Mock()
            mock_thread_class.return_value = mock_thread

            metrics_manager.start_background_collection(update_interval=10)

            assert metrics_manager._running is True
            mock_thread_class.assert_called_once()
            mock_thread.start.assert_called_once()

    def test_start_background_collection_already_running(self, metrics_manager):
        """Test starting background collection when already running."""
        metrics_manager._running = True

        with patch("threading.Thread") as mock_thread_class:
            metrics_manager.start_background_collection()

            # Should not create new thread
            mock_thread_class.assert_not_called()

    def test_stop_background_collection(self, metrics_manager):
        """Test stopping background metrics collection."""
        # Setup running collection
        metrics_manager._running = True
        mock_thread = Mock()
        mock_thread.is_alive.return_value = True
        metrics_manager._resource_update_thread = mock_thread

        metrics_manager.stop_background_collection()

        assert metrics_manager._running is False
        mock_thread.join.assert_called_once_with(timeout=5)

    def test_stop_background_collection_no_thread(self, metrics_manager):
        """Test stopping when no thread is running."""
        metrics_manager._running = False

        # Should not raise exception
        metrics_manager.stop_background_collection()

    def test_get_metrics_prometheus_available(self, metrics_manager):
        """Test getting metrics when Prometheus is available."""
        with patch("src.utils.metrics.PROMETHEUS_AVAILABLE", True), patch(
            "src.utils.metrics.generate_latest"
        ) as mock_generate:

            mock_generate.return_value = b"# Test metrics\nmetric_name 1.0\n"

            result = metrics_manager.get_metrics()

            assert result == "# Test metrics\nmetric_name 1.0\n"
            mock_generate.assert_called_once_with(metrics_manager.registry)

    def test_get_metrics_prometheus_unavailable(self, metrics_manager):
        """Test getting metrics when Prometheus is not available."""
        with patch("src.utils.metrics.PROMETHEUS_AVAILABLE", False):
            result = metrics_manager.get_metrics()
            assert result == "# Prometheus client not available\n"

    def test_get_collector(self, metrics_manager):
        """Test getting metrics collector."""
        collector = metrics_manager.get_collector()
        assert collector == metrics_manager.collector
        assert isinstance(collector, MLMetricsCollector)


class TestMultiProcessMetricsManager:
    """Test MultiProcessMetricsManager functionality."""

    @pytest.fixture
    def multiprocess_manager(self):
        """Create MultiProcessMetricsManager instance."""
        return MultiProcessMetricsManager()

    def test_initialization_prometheus_available(self, multiprocess_manager):
        """Test initialization when Prometheus multiprocess is available."""
        # The actual test depends on whether prometheus_client is available
        # in the test environment
        assert isinstance(multiprocess_manager.multiprocess_enabled, bool)

    def test_is_multiprocess_enabled(self, multiprocess_manager):
        """Test checking if multiprocess is enabled."""
        enabled = multiprocess_manager.is_multiprocess_enabled()
        assert isinstance(enabled, bool)

    def test_get_multiprocess_registry(self, multiprocess_manager):
        """Test getting multiprocess registry."""
        registry = multiprocess_manager.get_multiprocess_registry()
        # Could be None if multiprocess not available
        assert registry is None or hasattr(registry, "collect")

    def test_aggregate_multiprocess_metrics_disabled(self, multiprocess_manager):
        """Test aggregating metrics when multiprocess is disabled."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", False):
            result = multiprocess_manager.aggregate_multiprocess_metrics()
            assert "error" in result
            assert "not available" in result["error"]

    def test_aggregate_multiprocess_metrics_no_registry(self, multiprocess_manager):
        """Test aggregating metrics when no registry exists."""
        with patch.object(
            multiprocess_manager, "multiprocess_enabled", True
        ), patch.object(multiprocess_manager, "registry", None):

            result = multiprocess_manager.aggregate_multiprocess_metrics()
            assert "error" in result

    @patch("src.utils.metrics.multiprocess")
    def test_aggregate_multiprocess_metrics_success(
        self, mock_multiprocess, multiprocess_manager
    ):
        """Test successful multiprocess metrics aggregation."""
        # Setup multiprocess enabled
        multiprocess_manager.multiprocess_enabled = True
        multiprocess_manager.registry = Mock()

        # Mock metric families
        mock_metric_family = Mock()
        mock_metric_family.name = "test_metric"
        mock_metric_family.help = "Test metric help"
        mock_metric_family.type = "counter"

        mock_sample = Mock()
        mock_sample.name = "test_metric_total"
        mock_sample.labels = {"label": "value"}
        mock_sample.value = 10.0
        mock_sample.timestamp = 1642248000.0

        mock_metric_family.samples = [mock_sample]
        multiprocess_manager.registry.collect.return_value = [mock_metric_family]

        result = multiprocess_manager.aggregate_multiprocess_metrics()

        assert "test_metric" in result
        assert result["test_metric"]["name"] == "test_metric"
        assert result["test_metric"]["help"] == "Test metric help"
        assert result["test_metric"]["type"] == "counter"
        assert len(result["test_metric"]["samples"]) == 1

    def test_aggregate_multiprocess_metrics_exception(self, multiprocess_manager):
        """Test multiprocess metrics aggregation with exception."""
        with patch.object(
            multiprocess_manager, "multiprocess_enabled", True
        ), patch.object(multiprocess_manager, "registry", Mock()) as mock_registry:

            mock_registry.collect.side_effect = Exception("Collection failed")

            result = multiprocess_manager.aggregate_multiprocess_metrics()
            assert "error" in result
            assert "Collection failed" in result["error"]

    def test_generate_multiprocess_metrics_disabled(self, multiprocess_manager):
        """Test generating metrics when multiprocess is disabled."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", False):
            result = multiprocess_manager.generate_multiprocess_metrics()
            assert "not available" in result

    def test_generate_multiprocess_metrics_enabled(self, multiprocess_manager):
        """Test generating multiprocess metrics when enabled."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", True), patch(
            "src.utils.metrics.generate_latest"
        ) as mock_generate:

            multiprocess_manager.registry = Mock()
            mock_generate.return_value = b"# Multiprocess metrics\n"

            result = multiprocess_manager.generate_multiprocess_metrics()
            assert result == "# Multiprocess metrics\n"

    def test_generate_multiprocess_metrics_exception(self, multiprocess_manager):
        """Test generating multiprocess metrics with exception."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", True), patch(
            "src.utils.metrics.generate_latest"
        ) as mock_generate:

            multiprocess_manager.registry = Mock()
            mock_generate.side_effect = Exception("Generation failed")

            result = multiprocess_manager.generate_multiprocess_metrics()
            assert "Error generating" in result
            assert "Generation failed" in result

    def test_cleanup_dead_processes(self, multiprocess_manager):
        """Test cleaning up dead processes."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", True), patch(
            "src.utils.metrics.multiprocess"
        ) as mock_multiprocess:

            multiprocess_manager.cleanup_dead_processes()

            # Should call mark_process_dead
            mock_multiprocess.mark_process_dead.assert_called_once_with(None)

    def test_cleanup_dead_processes_disabled(self, multiprocess_manager):
        """Test cleanup when multiprocess is disabled."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", False):
            # Should not raise exception
            multiprocess_manager.cleanup_dead_processes()

    def test_cleanup_dead_processes_exception(self, multiprocess_manager):
        """Test cleanup with exception."""
        with patch.object(multiprocess_manager, "multiprocess_enabled", True), patch(
            "src.utils.metrics.multiprocess"
        ) as mock_multiprocess:

            mock_multiprocess.mark_process_dead.side_effect = Exception(
                "Cleanup failed"
            )

            # Should not raise exception
            multiprocess_manager.cleanup_dead_processes()


class TestTimePredictionDecorator:
    """Test time_prediction decorator functionality."""

    def test_time_prediction_decorator_success(self):
        """Test time_prediction decorator with successful prediction."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            @time_prediction("living_room", "next_occupancy", "lstm")
            def mock_prediction():
                time.sleep(0.01)  # Small delay
                return {"confidence": 0.85, "prediction": "result"}

            with patch("time.time", side_effect=[0, 0.1]):  # Mock 0.1 second duration
                result = mock_prediction()

            assert result == {"confidence": 0.85, "prediction": "result"}

            mock_collector.record_prediction.assert_called_once_with(
                room_id="living_room",
                prediction_type="next_occupancy",
                model_type="lstm",
                duration=0.1,
                confidence=0.85,
                status="success",
            )

    def test_time_prediction_decorator_no_confidence(self):
        """Test time_prediction decorator without confidence in result."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            @time_prediction("bedroom", "next_vacancy", "xgboost")
            def mock_prediction():
                return {"prediction": "result"}

            with patch("time.time", side_effect=[0, 0.05]):
                result = mock_prediction()

            mock_collector.record_prediction.assert_called_once_with(
                room_id="bedroom",
                prediction_type="next_vacancy",
                model_type="xgboost",
                duration=0.05,
                confidence=None,
                status="success",
            )

    def test_time_prediction_decorator_exception(self):
        """Test time_prediction decorator with exception."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            @time_prediction("kitchen", "next_occupancy", "hmm")
            def failing_prediction():
                time.sleep(0.01)
                raise ValueError("Prediction failed")

            with patch("time.time", side_effect=[0, 0.02]):
                with pytest.raises(ValueError):
                    failing_prediction()

            mock_collector.record_prediction.assert_called_once_with(
                room_id="kitchen",
                prediction_type="next_occupancy",
                model_type="hmm",
                duration=0.02,
                status="error",
            )

    def test_time_prediction_decorator_non_dict_result(self):
        """Test time_prediction decorator with non-dict result."""
        with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
            mock_collector = Mock()
            mock_get_collector.return_value = mock_collector

            @time_prediction("office", "next_vacancy", "gp")
            def simple_prediction():
                return "simple_result"

            with patch("time.time", side_effect=[0, 0.03]):
                result = simple_prediction()

            assert result == "simple_result"

            mock_collector.record_prediction.assert_called_once_with(
                room_id="office",
                prediction_type="next_vacancy",
                model_type="gp",
                duration=0.03,
                confidence=None,
                status="success",
            )


class TestGlobalFunctions:
    """Test global utility functions."""

    def test_get_metrics_manager_singleton(self):
        """Test that get_metrics_manager returns singleton."""
        manager1 = get_metrics_manager()
        manager2 = get_metrics_manager()

        assert manager1 is manager2
        assert isinstance(manager1, MetricsManager)

    def test_get_metrics_collector(self):
        """Test getting metrics collector through global function."""
        collector = get_metrics_collector()
        assert isinstance(collector, MLMetricsCollector)

        # Should be same as manager's collector
        manager = get_metrics_manager()
        assert collector is manager.get_collector()

    def test_metrics_endpoint_handler(self):
        """Test metrics endpoint handler."""
        with patch("src.utils.metrics.get_metrics_manager") as mock_get_manager:
            mock_manager = Mock()
            mock_manager.get_metrics.return_value = "# Test metrics\n"
            mock_get_manager.return_value = mock_manager

            result = metrics_endpoint_handler()
            assert result == "# Test metrics\n"

    def test_get_multiprocess_metrics_manager_singleton(self):
        """Test multiprocess metrics manager singleton."""
        manager1 = get_multiprocess_metrics_manager()
        manager2 = get_multiprocess_metrics_manager()

        assert manager1 is manager2
        assert isinstance(manager1, MultiProcessMetricsManager)

    def test_setup_multiprocess_metrics(self):
        """Test setting up multiprocess metrics."""
        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_manager:
            mock_manager = Mock()
            mock_manager.is_multiprocess_enabled.return_value = True
            mock_manager.get_multiprocess_registry.return_value = Mock()
            mock_get_manager.return_value = mock_manager

            with patch("src.utils.metrics.get_metrics_collector") as mock_get_collector:
                mock_collector = Mock()
                mock_get_collector.return_value = mock_collector

                setup_multiprocess_metrics()

                mock_manager.cleanup_dead_processes.assert_called_once()

    def test_get_aggregated_metrics_multiprocess_enabled(self):
        """Test getting aggregated metrics with multiprocess enabled."""
        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_multiprocess, patch(
            "src.utils.metrics.get_metrics_manager"
        ) as mock_get_single:

            mock_multiprocess_manager = Mock()
            mock_multiprocess_manager.is_multiprocess_enabled.return_value = True
            mock_multiprocess_manager.aggregate_multiprocess_metrics.return_value = {
                "metric": "data"
            }
            mock_get_multiprocess.return_value = mock_multiprocess_manager

            mock_single_manager = Mock()
            mock_single_manager.get_metrics.return_value = "single_process_metrics"
            mock_get_single.return_value = mock_single_manager

            result = get_aggregated_metrics()

            assert result["collection_mode"] == "multiprocess"
            assert result["multiprocess_metrics"] == {"metric": "data"}
            assert result["single_process_metrics"] == "single_process_metrics"

    def test_get_aggregated_metrics_single_process(self):
        """Test getting aggregated metrics with single process."""
        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_multiprocess, patch(
            "src.utils.metrics.get_metrics_manager"
        ) as mock_get_single:

            mock_multiprocess_manager = Mock()
            mock_multiprocess_manager.is_multiprocess_enabled.return_value = False
            mock_get_multiprocess.return_value = mock_multiprocess_manager

            mock_single_manager = Mock()
            mock_single_manager.get_metrics.return_value = "single_process_metrics"
            mock_get_single.return_value = mock_single_manager

            result = get_aggregated_metrics()

            assert result["collection_mode"] == "single_process"
            assert result["single_process_metrics"] == "single_process_metrics"
            assert "multiprocess_metrics" not in result

    def test_export_multiprocess_metrics_enabled(self):
        """Test exporting multiprocess metrics when enabled."""
        from src.utils.metrics import export_multiprocess_metrics

        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_manager:
            mock_manager = Mock()
            mock_manager.is_multiprocess_enabled.return_value = True
            mock_manager.generate_multiprocess_metrics.return_value = (
                "# Multiprocess metrics\n"
            )
            mock_get_manager.return_value = mock_manager

            result = export_multiprocess_metrics()
            assert result == "# Multiprocess metrics\n"

    def test_export_multiprocess_metrics_fallback(self):
        """Test exporting multiprocess metrics fallback to single process."""
        from src.utils.metrics import export_multiprocess_metrics

        with patch(
            "src.utils.metrics.get_multiprocess_metrics_manager"
        ) as mock_get_multiprocess, patch(
            "src.utils.metrics.get_metrics_manager"
        ) as mock_get_single:

            mock_multiprocess_manager = Mock()
            mock_multiprocess_manager.is_multiprocess_enabled.return_value = False
            mock_get_multiprocess.return_value = mock_multiprocess_manager

            mock_single_manager = Mock()
            mock_single_manager.get_metrics.return_value = "# Single process metrics\n"
            mock_get_single.return_value = mock_single_manager

            result = export_multiprocess_metrics()
            assert result == "# Single process metrics\n"


if __name__ == "__main__":
    pytest.main([__file__])
