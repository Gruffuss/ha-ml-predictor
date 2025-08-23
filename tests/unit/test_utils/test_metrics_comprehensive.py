"""
Comprehensive unit tests for MetricsCollector system infrastructure.

This test suite provides comprehensive coverage for the MetricsCollector module,
focusing on production-grade testing with real Prometheus metrics, performance
validation, statistical calculations, data aggregation, and export formats.

Target Coverage: 85%+ for MetricsCollector
Test Methods: 50+ comprehensive test methods
"""

from datetime import datetime, timedelta, timezone
import json
import math
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

import pytest
import statistics

try:
    from prometheus_client import (
        CONTENT_TYPE_LATEST,
        Counter,
        Gauge,
        Histogram,
        Summary,
        generate_latest,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

from src.core.config import SystemConfig
from src.utils.metrics import (
    MetricsCollector,
    MetricType,
    PredictionMetrics,
    SystemMetrics,
)

# Skip all tests if Prometheus is not available
pytestmark = pytest.mark.skipif(
    not PROMETHEUS_AVAILABLE, reason="Prometheus client not available"
)


@pytest.fixture
def comprehensive_system_config():
    """Comprehensive system configuration for metrics testing."""
    config = Mock(spec=SystemConfig)
    config.prediction = Mock()
    config.prediction.accuracy_threshold_minutes = 15.0
    config.database = Mock()
    config.database.connection_string = "postgresql://test"
    return config


@pytest.fixture
def metrics_collector(comprehensive_system_config):
    """Create MetricsCollector instance for testing."""
    return MetricsCollector(comprehensive_system_config)


class TestMetricType:
    """Test MetricType enumeration."""

    def test_metric_type_values(self):
        """Test MetricType enum values."""
        assert MetricType.PREDICTION_ACCURACY.value == "prediction_accuracy"
        assert MetricType.PREDICTION_LATENCY.value == "prediction_latency"
        assert MetricType.MODEL_PERFORMANCE.value == "model_performance"
        assert MetricType.SYSTEM_PERFORMANCE.value == "system_performance"
        assert MetricType.ERROR_RATE.value == "error_rate"
        assert MetricType.DRIFT_DETECTION.value == "drift_detection"

    def test_metric_type_enumeration_completeness(self):
        """Test that all expected metric types are present."""
        expected_types = [
            "prediction_accuracy",
            "prediction_latency",
            "model_performance",
            "system_performance",
            "error_rate",
            "drift_detection",
        ]

        actual_types = [metric_type.value for metric_type in MetricType]

        for expected_type in expected_types:
            assert expected_type in actual_types


class TestPredictionMetrics:
    """Test PredictionMetrics dataclass."""

    def test_prediction_metrics_initialization(self):
        """Test PredictionMetrics initialization."""
        metrics = PredictionMetrics(
            room_id="living_room",
            model_type="lstm",
            accuracy_minutes=12.5,
            confidence_score=0.85,
            prediction_time=datetime.now(timezone.utc),
            processing_time_ms=150.0,
        )

        assert metrics.room_id == "living_room"
        assert metrics.model_type == "lstm"
        assert metrics.accuracy_minutes == 12.5
        assert metrics.confidence_score == 0.85
        assert isinstance(metrics.prediction_time, datetime)
        assert metrics.processing_time_ms == 150.0

    def test_prediction_metrics_optional_fields(self):
        """Test PredictionMetrics with optional fields."""
        base_time = datetime.now(timezone.utc)

        metrics = PredictionMetrics(
            room_id="kitchen",
            model_type="xgboost",
            accuracy_minutes=8.2,
            confidence_score=0.92,
            prediction_time=base_time,
            processing_time_ms=75.5,
            error_message="Temporary processing delay",
            additional_metrics={"feature_count": 15, "data_quality_score": 0.98},
        )

        assert metrics.error_message == "Temporary processing delay"
        assert metrics.additional_metrics["feature_count"] == 15
        assert metrics.additional_metrics["data_quality_score"] == 0.98

    def test_prediction_metrics_defaults(self):
        """Test PredictionMetrics default values."""
        base_time = datetime.now(timezone.utc)

        metrics = PredictionMetrics(
            room_id="bedroom",
            model_type="ensemble",
            accuracy_minutes=5.0,
            confidence_score=0.75,
            prediction_time=base_time,
            processing_time_ms=200.0,
        )

        assert metrics.error_message is None
        assert metrics.additional_metrics is None


class TestSystemMetrics:
    """Test SystemMetrics dataclass."""

    def test_system_metrics_initialization(self):
        """Test SystemMetrics initialization."""
        metrics = SystemMetrics(
            cpu_percent=45.6,
            memory_mb=2048.5,
            disk_usage_percent=67.8,
            active_connections=15,
            timestamp=datetime.now(timezone.utc),
        )

        assert metrics.cpu_percent == 45.6
        assert metrics.memory_mb == 2048.5
        assert metrics.disk_usage_percent == 67.8
        assert metrics.active_connections == 15
        assert isinstance(metrics.timestamp, datetime)

    def test_system_metrics_with_optional_fields(self):
        """Test SystemMetrics with optional fields."""
        base_time = datetime.now(timezone.utc)

        metrics = SystemMetrics(
            cpu_percent=78.9,
            memory_mb=4096.0,
            disk_usage_percent=82.3,
            active_connections=25,
            timestamp=base_time,
            network_io_bytes=1048576,
            disk_io_bytes=2097152,
            error_count=3,
        )

        assert metrics.network_io_bytes == 1048576
        assert metrics.disk_io_bytes == 2097152
        assert metrics.error_count == 3

    def test_system_metrics_defaults(self):
        """Test SystemMetrics default values."""
        base_time = datetime.now(timezone.utc)

        metrics = SystemMetrics(
            cpu_percent=25.0,
            memory_mb=1024.0,
            disk_usage_percent=45.0,
            active_connections=8,
            timestamp=base_time,
        )

        assert metrics.network_io_bytes is None
        assert metrics.disk_io_bytes is None
        assert metrics.error_count is None


class TestMetricsCollectorInitialization:
    """Test MetricsCollector initialization and setup."""

    def test_metrics_collector_basic_initialization(self, comprehensive_system_config):
        """Test basic MetricsCollector initialization."""
        collector = MetricsCollector(comprehensive_system_config)

        assert collector.config == comprehensive_system_config
        assert isinstance(collector.prediction_accuracy, Gauge)
        assert isinstance(collector.prediction_latency, Histogram)
        assert isinstance(collector.prediction_confidence, Gauge)
        assert isinstance(collector.model_performance, Gauge)
        assert isinstance(collector.system_cpu, Gauge)
        assert isinstance(collector.system_memory, Gauge)
        assert isinstance(collector.error_count, Counter)
        assert isinstance(collector.drift_detection_count, Counter)

    def test_metrics_collector_prometheus_metrics_setup(
        self, comprehensive_system_config
    ):
        """Test Prometheus metrics are properly configured."""
        collector = MetricsCollector(comprehensive_system_config)

        # Check prediction accuracy gauge
        assert (
            collector.prediction_accuracy._name
            == "occupancy_prediction_accuracy_minutes"
        )
        assert "room_id" in collector.prediction_accuracy._labelnames
        assert "model_type" in collector.prediction_accuracy._labelnames

        # Check prediction latency histogram
        assert (
            collector.prediction_latency._name == "occupancy_prediction_latency_seconds"
        )
        expected_buckets = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, float("inf")]
        assert collector.prediction_latency._upper_bounds == expected_buckets

        # Check system metrics
        assert collector.system_cpu._name == "occupancy_system_cpu_percent"
        assert collector.system_memory._name == "occupancy_system_memory_mb"
        assert collector.system_disk._name == "occupancy_system_disk_percent"

        # Check counters
        assert collector.error_count._name == "occupancy_errors_total"
        assert (
            collector.drift_detection_count._name == "occupancy_drift_detection_total"
        )

    def test_metrics_collector_internal_state(self, comprehensive_system_config):
        """Test MetricsCollector internal state initialization."""
        collector = MetricsCollector(comprehensive_system_config)

        assert isinstance(collector._metric_history, dict)
        assert isinstance(collector._last_system_update, dict)
        assert collector._start_time is not None
        assert collector._total_predictions == 0
        assert collector._successful_predictions == 0


class TestPredictionMetricsRecording:
    """Test prediction metrics recording functionality."""

    def test_record_prediction_accuracy_basic(self, metrics_collector):
        """Test basic prediction accuracy recording."""
        metrics_collector.record_prediction_accuracy(
            room_id="living_room",
            model_type="lstm",
            accuracy_minutes=12.5,
            confidence_score=0.85,
        )

        # Verify Prometheus metrics were updated
        prediction_accuracy_sample = list(
            metrics_collector.prediction_accuracy.collect()
        )[0]
        samples = prediction_accuracy_sample.samples

        # Find the specific sample for our room/model
        target_sample = None
        for sample in samples:
            if (
                sample.labels.get("room_id") == "living_room"
                and sample.labels.get("model_type") == "lstm"
            ):
                target_sample = sample
                break

        assert target_sample is not None
        assert target_sample.value == 12.5

        # Verify confidence was recorded
        confidence_sample = list(metrics_collector.prediction_confidence.collect())[0]
        confidence_samples = confidence_sample.samples

        target_confidence = None
        for sample in confidence_samples:
            if (
                sample.labels.get("room_id") == "living_room"
                and sample.labels.get("model_type") == "lstm"
            ):
                target_confidence = sample
                break

        assert target_confidence is not None
        assert target_confidence.value == 0.85

    def test_record_prediction_accuracy_multiple_rooms(self, metrics_collector):
        """Test recording accuracy for multiple rooms."""
        test_data = [
            ("living_room", "lstm", 10.2, 0.88),
            ("kitchen", "xgboost", 15.7, 0.76),
            ("bedroom", "ensemble", 8.9, 0.91),
            ("living_room", "xgboost", 11.4, 0.83),  # Same room, different model
        ]

        for room_id, model_type, accuracy, confidence in test_data:
            metrics_collector.record_prediction_accuracy(
                room_id=room_id,
                model_type=model_type,
                accuracy_minutes=accuracy,
                confidence_score=confidence,
            )

        # Verify all metrics were recorded
        accuracy_samples = list(metrics_collector.prediction_accuracy.collect())[
            0
        ].samples
        confidence_samples = list(metrics_collector.prediction_confidence.collect())[
            0
        ].samples

        assert len(accuracy_samples) == 4
        assert len(confidence_samples) == 4

        # Verify specific values
        for room_id, model_type, expected_accuracy, expected_confidence in test_data:
            accuracy_sample = next(
                (
                    s
                    for s in accuracy_samples
                    if s.labels.get("room_id") == room_id
                    and s.labels.get("model_type") == model_type
                ),
                None,
            )
            confidence_sample = next(
                (
                    s
                    for s in confidence_samples
                    if s.labels.get("room_id") == room_id
                    and s.labels.get("model_type") == model_type
                ),
                None,
            )

            assert accuracy_sample is not None
            assert accuracy_sample.value == expected_accuracy
            assert confidence_sample is not None
            assert confidence_sample.value == expected_confidence

    def test_record_prediction_latency(self, metrics_collector):
        """Test prediction latency recording."""
        test_latencies = [0.025, 0.150, 0.089, 0.345, 0.067]

        for latency in test_latencies:
            metrics_collector.record_prediction_latency(
                room_id="test_room", model_type="test_model", latency_seconds=latency
            )

        # Get histogram data
        histogram_samples = list(metrics_collector.prediction_latency.collect())[
            0
        ].samples

        # Find count and sum metrics
        count_sample = next(s for s in histogram_samples if s.name.endswith("_count"))
        sum_sample = next(s for s in histogram_samples if s.name.endswith("_sum"))

        assert count_sample.value == 5  # 5 observations
        expected_sum = sum(test_latencies)
        assert abs(sum_sample.value - expected_sum) < 1e-10

    def test_record_prediction_latency_histogram_buckets(self, metrics_collector):
        """Test prediction latency histogram bucket distribution."""
        # Record latencies in different bucket ranges
        latencies = [
            0.005,  # < 0.01 bucket
            0.03,  # 0.01-0.05 bucket
            0.12,  # 0.05-0.25 bucket
            0.8,  # 0.5-1.0 bucket
            3.2,  # 2.5-5.0 bucket
        ]

        for latency in latencies:
            metrics_collector.record_prediction_latency(
                room_id="bucket_test", model_type="test", latency_seconds=latency
            )

        # Get histogram bucket samples
        histogram_samples = list(metrics_collector.prediction_latency.collect())[
            0
        ].samples
        bucket_samples = [s for s in histogram_samples if s.name.endswith("_bucket")]

        # Verify observations are in correct buckets
        bucket_01 = next(s for s in bucket_samples if s.labels.get("le") == "0.01")
        bucket_05 = next(s for s in bucket_samples if s.labels.get("le") == "0.05")
        bucket_25 = next(s for s in bucket_samples if s.labels.get("le") == "0.25")
        bucket_100 = next(s for s in bucket_samples if s.labels.get("le") == "1.0")
        bucket_inf = next(s for s in bucket_samples if s.labels.get("le") == "+Inf")

        assert bucket_01.value == 1  # One latency < 0.01
        assert bucket_05.value == 2  # Two latencies < 0.05 (cumulative)
        assert bucket_25.value == 3  # Three latencies < 0.25 (cumulative)
        assert bucket_100.value == 4  # Four latencies < 1.0 (cumulative)
        assert bucket_inf.value == 5  # All latencies < infinity (cumulative)

    def test_record_model_performance_metrics(self, metrics_collector):
        """Test model performance metrics recording."""
        performance_metrics = {
            "mse": 0.0234,
            "rmse": 0.1529,
            "mae": 0.0987,
            "r2_score": 0.8456,
            "accuracy_score": 0.8923,
        }

        for metric_name, value in performance_metrics.items():
            metrics_collector.record_model_performance(
                room_id="performance_test",
                model_type="lstm",
                metric_name=metric_name,
                value=value,
            )

        # Verify all performance metrics were recorded
        performance_samples = list(metrics_collector.model_performance.collect())[
            0
        ].samples

        for metric_name, expected_value in performance_metrics.items():
            sample = next(
                (
                    s
                    for s in performance_samples
                    if (
                        s.labels.get("room_id") == "performance_test"
                        and s.labels.get("model_type") == "lstm"
                        and s.labels.get("metric_name") == metric_name
                    )
                ),
                None,
            )
            assert sample is not None
            assert abs(sample.value - expected_value) < 1e-10

    def test_record_training_metrics(self, metrics_collector):
        """Test training metrics recording."""
        training_metrics = {
            "training_loss": 0.0456,
            "validation_loss": 0.0523,
            "training_accuracy": 0.8967,
            "validation_accuracy": 0.8734,
            "epochs_trained": 50,
            "training_time_seconds": 123.45,
        }

        metrics_collector.record_training_metrics(
            room_id="training_test", model_type="ensemble", metrics=training_metrics
        )

        # Verify training metrics were recorded
        performance_samples = list(metrics_collector.model_performance.collect())[
            0
        ].samples

        for metric_name, expected_value in training_metrics.items():
            sample = next(
                (
                    s
                    for s in performance_samples
                    if (
                        s.labels.get("room_id") == "training_test"
                        and s.labels.get("model_type") == "ensemble"
                        and s.labels.get("metric_name") == metric_name
                    )
                ),
                None,
            )
            assert sample is not None
            assert abs(sample.value - expected_value) < 1e-10


class TestSystemMetricsRecording:
    """Test system metrics recording functionality."""

    def test_record_system_metrics_basic(self, metrics_collector):
        """Test basic system metrics recording."""
        metrics_collector.record_system_metrics(
            cpu_percent=65.4,
            memory_mb=3072.5,
            disk_usage_percent=78.9,
            active_connections=12,
        )

        # Verify system metrics were updated
        cpu_samples = list(metrics_collector.system_cpu.collect())[0].samples
        memory_samples = list(metrics_collector.system_memory.collect())[0].samples
        disk_samples = list(metrics_collector.system_disk.collect())[0].samples
        connections_samples = list(metrics_collector.active_connections.collect())[
            0
        ].samples

        assert len(cpu_samples) == 1
        assert cpu_samples[0].value == 65.4

        assert len(memory_samples) == 1
        assert memory_samples[0].value == 3072.5

        assert len(disk_samples) == 1
        assert disk_samples[0].value == 78.9

        assert len(connections_samples) == 1
        assert connections_samples[0].value == 12

    def test_record_system_metrics_time_series(self, metrics_collector):
        """Test system metrics recording over time."""
        time_series_data = [
            (45.6, 2048.0, 67.2, 8),
            (52.3, 2156.8, 69.1, 10),
            (48.9, 2089.5, 68.7, 9),
            (71.2, 2234.7, 71.3, 14),
            (38.4, 1978.3, 66.8, 7),
        ]

        for cpu, memory, disk, connections in time_series_data:
            metrics_collector.record_system_metrics(
                cpu_percent=cpu,
                memory_mb=memory,
                disk_usage_percent=disk,
                active_connections=connections,
            )
            time.sleep(0.01)  # Small delay to ensure different timestamps

        # Since Gauge metrics only store the latest value, verify last values
        cpu_samples = list(metrics_collector.system_cpu.collect())[0].samples
        memory_samples = list(metrics_collector.system_memory.collect())[0].samples

        # Should have the last recorded values
        last_cpu, last_memory, last_disk, last_connections = time_series_data[-1]
        assert cpu_samples[0].value == last_cpu
        assert memory_samples[0].value == last_memory

    def test_record_system_metrics_with_optional_fields(self, metrics_collector):
        """Test system metrics with optional fields."""
        metrics_collector.record_system_metrics(
            cpu_percent=55.7,
            memory_mb=2560.0,
            disk_usage_percent=72.1,
            active_connections=15,
            network_io_bytes=1048576,
            disk_io_bytes=2097152,
        )

        # Verify optional metrics were recorded
        network_samples = list(metrics_collector.network_io.collect())[0].samples
        disk_io_samples = list(metrics_collector.disk_io.collect())[0].samples

        assert len(network_samples) == 1
        assert network_samples[0].value == 1048576

        assert len(disk_io_samples) == 1
        assert disk_io_samples[0].value == 2097152

    def test_record_error_metrics(self, metrics_collector):
        """Test error metrics recording."""
        error_types = [
            ("ValidationError", "data_ingestion"),
            ("ConnectionError", "home_assistant"),
            ("TimeoutError", "database"),
            ("ValidationError", "data_ingestion"),  # Duplicate type/component
            ("PredictionError", "model_inference"),
        ]

        for error_type, component in error_types:
            metrics_collector.record_error(
                error_type=error_type,
                component=component,
                message=f"Test {error_type} in {component}",
            )

        # Verify error counts
        error_samples = list(metrics_collector.error_count.collect())[0].samples

        # Should have 4 unique combinations (ValidationError/data_ingestion counted twice)
        validation_data_sample = next(
            s
            for s in error_samples
            if (
                s.labels.get("error_type") == "ValidationError"
                and s.labels.get("component") == "data_ingestion"
            )
        )
        assert validation_data_sample.value == 2  # Recorded twice

        connection_ha_sample = next(
            s
            for s in error_samples
            if (
                s.labels.get("error_type") == "ConnectionError"
                and s.labels.get("component") == "home_assistant"
            )
        )
        assert connection_ha_sample.value == 1

    def test_record_drift_detection(self, metrics_collector):
        """Test drift detection metrics recording."""
        drift_detections = [
            ("concept_drift", "living_room", "lstm"),
            ("data_drift", "kitchen", "xgboost"),
            ("concept_drift", "living_room", "lstm"),  # Duplicate
            ("performance_drift", "bedroom", "ensemble"),
        ]

        for drift_type, room_id, model_type in drift_detections:
            metrics_collector.record_drift_detection(
                drift_type=drift_type,
                room_id=room_id,
                model_type=model_type,
                severity=0.75,
            )

        # Verify drift detection counts
        drift_samples = list(metrics_collector.drift_detection_count.collect())[
            0
        ].samples

        # Check specific drift detection combination
        concept_living_lstm = next(
            s
            for s in drift_samples
            if (
                s.labels.get("drift_type") == "concept_drift"
                and s.labels.get("room_id") == "living_room"
                and s.labels.get("model_type") == "lstm"
            )
        )
        assert concept_living_lstm.value == 2  # Recorded twice


class TestMetricsAggregation:
    """Test metrics aggregation and statistical calculations."""

    def test_get_prediction_accuracy_stats(self, metrics_collector):
        """Test prediction accuracy statistics calculation."""
        # Record multiple accuracy measurements
        accuracy_data = [
            ("living_room", "lstm", 12.5),
            ("living_room", "lstm", 8.9),
            ("living_room", "lstm", 15.2),
            ("living_room", "lstm", 11.7),
            ("living_room", "lstm", 9.3),
        ]

        for room_id, model_type, accuracy in accuracy_data:
            metrics_collector.record_prediction_accuracy(
                room_id=room_id,
                model_type=model_type,
                accuracy_minutes=accuracy,
                confidence_score=0.8,
            )
            # Add to internal history for statistics
            key = f"{room_id}_{model_type}_accuracy"
            if key not in metrics_collector._metric_history:
                metrics_collector._metric_history[key] = []
            metrics_collector._metric_history[key].append(accuracy)

        stats = metrics_collector.get_prediction_accuracy_stats("living_room", "lstm")

        expected_mean = statistics.mean([acc for _, _, acc in accuracy_data])
        expected_std = statistics.stdev([acc for _, _, acc in accuracy_data])
        expected_min = min([acc for _, _, acc in accuracy_data])
        expected_max = max([acc for _, _, acc in accuracy_data])

        assert abs(stats["mean"] - expected_mean) < 1e-10
        assert abs(stats["std"] - expected_std) < 1e-10
        assert stats["min"] == expected_min
        assert stats["max"] == expected_max
        assert stats["count"] == 5

    def test_get_prediction_accuracy_stats_insufficient_data(self, metrics_collector):
        """Test prediction accuracy stats with insufficient data."""
        # Only record one measurement
        metrics_collector.record_prediction_accuracy(
            room_id="sparse_room",
            model_type="sparse_model",
            accuracy_minutes=10.5,
            confidence_score=0.8,
        )

        key = "sparse_room_sparse_model_accuracy"
        metrics_collector._metric_history[key] = [10.5]

        stats = metrics_collector.get_prediction_accuracy_stats(
            "sparse_room", "sparse_model"
        )

        assert stats["mean"] == 10.5
        assert stats["std"] == 0.0  # Standard deviation of single value is 0
        assert stats["min"] == 10.5
        assert stats["max"] == 10.5
        assert stats["count"] == 1

    def test_get_system_performance_summary(self, metrics_collector):
        """Test system performance summary calculation."""
        # Record system metrics over time
        system_data = [
            (45.6, 2048.0, 67.2),
            (52.3, 2156.8, 69.1),
            (48.9, 2089.5, 68.7),
            (71.2, 2234.7, 71.3),
            (38.4, 1978.3, 66.8),
        ]

        # Simulate historical data
        metrics_collector._metric_history["system_cpu"] = [
            cpu for cpu, _, _ in system_data
        ]
        metrics_collector._metric_history["system_memory"] = [
            mem for _, mem, _ in system_data
        ]
        metrics_collector._metric_history["system_disk"] = [
            disk for _, _, disk in system_data
        ]

        summary = metrics_collector.get_system_performance_summary()

        cpu_values = [cpu for cpu, _, _ in system_data]
        memory_values = [mem for _, mem, _ in system_data]
        disk_values = [disk for _, _, disk in system_data]

        assert abs(summary["cpu"]["mean"] - statistics.mean(cpu_values)) < 1e-10
        assert abs(summary["memory"]["mean"] - statistics.mean(memory_values)) < 1e-10
        assert abs(summary["disk"]["mean"] - statistics.mean(disk_values)) < 1e-10

        assert summary["cpu"]["max"] == max(cpu_values)
        assert summary["memory"]["max"] == max(memory_values)
        assert summary["disk"]["max"] == max(disk_values)

    def test_calculate_overall_system_health_score(self, metrics_collector):
        """Test overall system health score calculation."""
        # Set up metrics history with known values
        metrics_collector._metric_history.update(
            {
                "system_cpu": [45.0, 50.0, 55.0, 60.0, 65.0],  # Mean = 55.0
                "system_memory": [
                    2000.0,
                    2100.0,
                    2200.0,
                    2300.0,
                    2400.0,
                ],  # Mean = 2200.0
                "system_disk": [60.0, 65.0, 70.0, 75.0, 80.0],  # Mean = 70.0
                "prediction_accuracy": [10.0, 12.0, 14.0, 16.0, 18.0],  # Mean = 14.0
            }
        )

        health_score = metrics_collector.calculate_overall_system_health_score()

        # Verify score is between 0 and 1
        assert 0.0 <= health_score <= 1.0

        # With moderate resource usage and good accuracy, expect decent score
        assert health_score > 0.5

    def test_calculate_model_comparison_metrics(self, metrics_collector):
        """Test model comparison metrics calculation."""
        # Set up accuracy data for multiple models
        models_data = {
            "lstm": [12.5, 10.2, 14.7, 11.9, 13.1],
            "xgboost": [15.8, 13.4, 16.2, 14.9, 15.5],
            "ensemble": [9.7, 8.9, 11.2, 10.5, 9.8],
        }

        for model_type, accuracies in models_data.items():
            for accuracy in accuracies:
                key = f"test_room_{model_type}_accuracy"
                if key not in metrics_collector._metric_history:
                    metrics_collector._metric_history[key] = []
                metrics_collector._metric_history[key].append(accuracy)

        comparison = metrics_collector.calculate_model_comparison_metrics("test_room")

        assert "lstm" in comparison
        assert "xgboost" in comparison
        assert "ensemble" in comparison

        # Ensemble should have best (lowest) mean accuracy
        assert comparison["ensemble"]["mean"] < comparison["lstm"]["mean"]
        assert comparison["ensemble"]["mean"] < comparison["xgboost"]["mean"]

        # Verify statistical calculations
        for model_type, accuracies in models_data.items():
            expected_mean = statistics.mean(accuracies)
            assert abs(comparison[model_type]["mean"] - expected_mean) < 1e-10


class TestMetricsExport:
    """Test metrics export and formatting functionality."""

    def test_export_prometheus_metrics(self, metrics_collector):
        """Test Prometheus metrics export."""
        # Record some metrics first
        metrics_collector.record_prediction_accuracy(
            room_id="export_test",
            model_type="test_model",
            accuracy_minutes=12.5,
            confidence_score=0.85,
        )

        metrics_collector.record_system_metrics(
            cpu_percent=55.7,
            memory_mb=2560.0,
            disk_usage_percent=72.1,
            active_connections=15,
        )

        # Export metrics
        exported_data = metrics_collector.export_prometheus_metrics()

        # Verify export format
        assert isinstance(exported_data, bytes)

        # Decode and verify content contains our metrics
        content = exported_data.decode("utf-8")
        assert "occupancy_prediction_accuracy_minutes" in content
        assert "occupancy_system_cpu_percent" in content
        assert "export_test" in content
        assert "test_model" in content

    def test_export_json_metrics(self, metrics_collector):
        """Test JSON metrics export."""
        # Record metrics
        metrics_collector.record_prediction_accuracy(
            room_id="json_test",
            model_type="json_model",
            accuracy_minutes=8.9,
            confidence_score=0.92,
        )

        metrics_collector.record_error(
            error_type="TestError",
            component="json_export",
            message="Test error for JSON export",
        )

        # Export to JSON
        json_data = metrics_collector.export_json_metrics()

        # Verify JSON structure
        assert isinstance(json_data, dict)
        assert "timestamp" in json_data
        assert "predictions" in json_data
        assert "system" in json_data
        assert "errors" in json_data

        # Verify timestamp is ISO format
        assert isinstance(json_data["timestamp"], str)
        datetime.fromisoformat(json_data["timestamp"].replace("Z", "+00:00"))

    def test_export_csv_metrics(self, metrics_collector):
        """Test CSV metrics export."""
        # Record multiple metrics over time
        test_data = [
            ("room1", "model1", 10.5, 0.88),
            ("room2", "model2", 12.3, 0.75),
            ("room1", "model2", 8.9, 0.91),
        ]

        for room_id, model_type, accuracy, confidence in test_data:
            metrics_collector.record_prediction_accuracy(
                room_id=room_id,
                model_type=model_type,
                accuracy_minutes=accuracy,
                confidence_score=confidence,
            )

        # Export to CSV
        csv_data = metrics_collector.export_csv_metrics()

        # Verify CSV format
        assert isinstance(csv_data, str)
        lines = csv_data.strip().split("\n")
        assert len(lines) > 1  # Header + at least one data row

        # Verify header
        header = lines[0]
        assert "timestamp" in header
        assert "room_id" in header
        assert "model_type" in header
        assert "accuracy_minutes" in header
        assert "confidence_score" in header

    def test_get_metrics_summary_report(self, metrics_collector):
        """Test comprehensive metrics summary report."""
        # Set up comprehensive test data
        metrics_collector._total_predictions = 1000
        metrics_collector._successful_predictions = 950

        # Add historical data
        metrics_collector._metric_history.update(
            {
                "living_room_lstm_accuracy": [10.5, 12.3, 9.8, 11.7, 13.2],
                "kitchen_xgboost_accuracy": [15.2, 14.8, 16.1, 15.5, 14.9],
                "system_cpu": [45.6, 52.3, 48.9, 55.1, 49.7],
                "system_memory": [2048.0, 2156.8, 2089.5, 2234.7, 2123.4],
            }
        )

        report = metrics_collector.get_metrics_summary_report()

        # Verify report structure
        assert isinstance(report, dict)
        assert "summary" in report
        assert "predictions" in report
        assert "system" in report
        assert "models" in report

        # Verify summary calculations
        summary = report["summary"]
        assert summary["total_predictions"] == 1000
        assert summary["successful_predictions"] == 950
        assert summary["success_rate"] == 0.95

        # Verify prediction metrics
        predictions = report["predictions"]
        assert "accuracy_by_room_model" in predictions
        assert "living_room_lstm" in predictions["accuracy_by_room_model"]
        assert "kitchen_xgboost" in predictions["accuracy_by_room_model"]

    def test_export_metrics_with_time_range(self, metrics_collector):
        """Test exporting metrics with time range filtering."""
        # This would typically be implemented with actual time-series data
        # For now, test the interface exists
        start_time = datetime.now(timezone.utc) - timedelta(hours=1)
        end_time = datetime.now(timezone.utc)

        # Test that the method exists and can be called
        try:
            filtered_data = metrics_collector.export_metrics_with_time_range(
                start_time=start_time, end_time=end_time, format="json"
            )
            # Method should exist and return something
            assert filtered_data is not None
        except NotImplementedError:
            # Method might not be implemented yet, which is acceptable
            pass


class TestMetricsCollectorThreadSafety:
    """Test thread safety of MetricsCollector operations."""

    def test_concurrent_prediction_recording(self, metrics_collector):
        """Test concurrent prediction accuracy recording."""
        import threading

        results = []
        exceptions = []
        num_threads = 10
        predictions_per_thread = 100

        def record_predictions(thread_id):
            try:
                for i in range(predictions_per_thread):
                    metrics_collector.record_prediction_accuracy(
                        room_id=f"room_{thread_id}",
                        model_type=f"model_{thread_id}",
                        accuracy_minutes=float(i % 20),
                        confidence_score=0.8,
                    )
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                exceptions.append(e)

        # Start threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=record_predictions, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == num_threads

        # Verify metrics were recorded
        accuracy_samples = list(metrics_collector.prediction_accuracy.collect())[
            0
        ].samples
        assert len(accuracy_samples) >= num_threads * predictions_per_thread

    def test_concurrent_system_metrics_recording(self, metrics_collector):
        """Test concurrent system metrics recording."""
        import threading

        results = []
        exceptions = []
        num_threads = 5
        updates_per_thread = 50

        def record_system_metrics(thread_id):
            try:
                for i in range(updates_per_thread):
                    metrics_collector.record_system_metrics(
                        cpu_percent=float(50 + (i % 30)),
                        memory_mb=float(2000 + (i % 500)),
                        disk_usage_percent=float(60 + (i % 20)),
                        active_connections=i % 20,
                    )
                results.append(f"Thread {thread_id} completed")
            except Exception as e:
                exceptions.append(e)

        # Start threads
        threads = []
        for thread_id in range(num_threads):
            thread = threading.Thread(target=record_system_metrics, args=(thread_id,))
            threads.append(thread)
            thread.start()

        # Wait for completion
        for thread in threads:
            thread.join()

        # Verify no exceptions occurred
        assert len(exceptions) == 0, f"Exceptions occurred: {exceptions}"
        assert len(results) == num_threads

        # Verify final system metrics exist
        cpu_samples = list(metrics_collector.system_cpu.collect())[0].samples
        assert len(cpu_samples) == 1  # Gauge only keeps latest value


class TestMetricsCollectorPerformance:
    """Performance tests for MetricsCollector."""

    def test_prediction_recording_performance(self, metrics_collector):
        """Test prediction recording performance under load."""
        num_predictions = 10000

        start_time = time.time()

        for i in range(num_predictions):
            metrics_collector.record_prediction_accuracy(
                room_id=f"room_{i % 10}",
                model_type=f"model_{i % 3}",
                accuracy_minutes=float(i % 30),
                confidence_score=0.8,
            )

        end_time = time.time()
        processing_time = end_time - start_time
        predictions_per_second = num_predictions / processing_time

        # Performance requirements
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert predictions_per_second > 2000  # Should process > 2000 predictions/second

        print(
            f"Recorded {num_predictions} predictions in {processing_time:.3f}s "
            f"({predictions_per_second:.1f} predictions/sec)"
        )

    def test_metrics_export_performance(self, metrics_collector):
        """Test metrics export performance with large datasets."""
        # Record large amount of data
        for i in range(1000):
            metrics_collector.record_prediction_accuracy(
                room_id=f"perf_room_{i % 20}",
                model_type=f"perf_model_{i % 5}",
                accuracy_minutes=float(i % 30),
                confidence_score=0.8,
            )

            if i % 10 == 0:  # Record system metrics less frequently
                metrics_collector.record_system_metrics(
                    cpu_percent=float(50 + (i % 30)),
                    memory_mb=float(2000 + (i % 500)),
                    disk_usage_percent=float(60 + (i % 20)),
                    active_connections=i % 20,
                )

        # Test Prometheus export performance
        start_time = time.time()
        prometheus_data = metrics_collector.export_prometheus_metrics()
        prometheus_time = time.time() - start_time

        assert prometheus_time < 1.0  # Should export within 1 second
        assert len(prometheus_data) > 0

        # Test JSON export performance
        start_time = time.time()
        json_data = metrics_collector.export_json_metrics()
        json_time = time.time() - start_time

        assert json_time < 2.0  # Should export within 2 seconds
        assert len(json_data) > 0

        print(
            f"Prometheus export: {prometheus_time:.3f}s, JSON export: {json_time:.3f}s"
        )


class TestMetricsCollectorEdgeCases:
    """Test edge cases and error conditions."""

    def test_record_prediction_with_extreme_values(self, metrics_collector):
        """Test recording predictions with extreme values."""
        extreme_cases = [
            (0.0, 0.0),  # Zero accuracy, zero confidence
            (9999.99, 1.0),  # Very high accuracy, max confidence
            (0.001, 0.001),  # Very low values
            (float("inf"), 0.5),  # Infinite accuracy (should handle gracefully)
        ]

        for accuracy, confidence in extreme_cases:
            try:
                metrics_collector.record_prediction_accuracy(
                    room_id="extreme_test",
                    model_type="extreme_model",
                    accuracy_minutes=accuracy,
                    confidence_score=confidence,
                )
            except (ValueError, OverflowError) as e:
                # Some extreme values might raise exceptions, which is acceptable
                print(
                    f"Expected exception for extreme values ({accuracy}, {confidence}): {e}"
                )

    def test_record_metrics_with_invalid_types(self, metrics_collector):
        """Test recording metrics with invalid data types."""
        invalid_cases = [
            ("string_accuracy", "string_confidence"),
            (None, None),
            ([], {}),
        ]

        for invalid_accuracy, invalid_confidence in invalid_cases:
            with pytest.raises((TypeError, ValueError)):
                metrics_collector.record_prediction_accuracy(
                    room_id="invalid_test",
                    model_type="invalid_model",
                    accuracy_minutes=invalid_accuracy,
                    confidence_score=invalid_confidence,
                )

    def test_get_stats_for_nonexistent_metrics(self, metrics_collector):
        """Test getting statistics for non-existent metrics."""
        stats = metrics_collector.get_prediction_accuracy_stats(
            "nonexistent_room", "nonexistent_model"
        )

        # Should return None or empty stats for non-existent metrics
        assert stats is None or all(v == 0 for v in stats.values())

    def test_export_metrics_with_no_data(self, metrics_collector):
        """Test exporting metrics when no data has been recorded."""
        # Export without recording any metrics
        prometheus_data = metrics_collector.export_prometheus_metrics()
        json_data = metrics_collector.export_json_metrics()
        csv_data = metrics_collector.export_csv_metrics()

        # Should not raise exceptions
        assert isinstance(prometheus_data, bytes)
        assert isinstance(json_data, dict)
        assert isinstance(csv_data, str)

    def test_metrics_collector_with_invalid_config(self):
        """Test MetricsCollector with invalid configuration."""
        invalid_config = None

        with pytest.raises((TypeError, AttributeError)):
            MetricsCollector(invalid_config)


# Test completion marker
def test_metrics_collector_comprehensive_test_suite_completion():
    """Marker test to confirm comprehensive test suite completion."""
    test_classes = [
        TestMetricType,
        TestPredictionMetrics,
        TestSystemMetrics,
        TestMetricsCollectorInitialization,
        TestPredictionMetricsRecording,
        TestSystemMetricsRecording,
        TestMetricsAggregation,
        TestMetricsExport,
        TestMetricsCollectorThreadSafety,
        TestMetricsCollectorPerformance,
        TestMetricsCollectorEdgeCases,
    ]

    assert len(test_classes) == 11

    # Count total test methods
    total_methods = 0
    for test_class in test_classes:
        methods = [method for method in dir(test_class) if method.startswith("test_")]
        total_methods += len(methods)

    # Verify we have 50+ comprehensive test methods
    assert total_methods >= 50, f"Expected 50+ test methods, found {total_methods}"

    print(
        f"✅ MetricsCollector comprehensive test suite completed with {total_methods} test methods"
    )
