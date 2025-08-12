"""
Prometheus metrics collection system for Home Assistant ML Predictor.
Provides comprehensive monitoring of ML operations, performance, and system health.
"""

from contextlib import contextmanager
from datetime import datetime
from functools import wraps
import threading
import time
from typing import Any, Dict, Optional

import psutil

try:
    from prometheus_client import (
        REGISTRY,
        CollectorRegistry,
        Counter,
        Gauge,
        Histogram,
        Info,
        Summary,
        generate_latest,
        multiprocess,
    )

    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

    # Mock classes for when prometheus_client isn't available
    class _MockCounter:
        def __init__(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class _MockGauge:
        def __init__(self, *args, **kwargs):
            pass

        def set(self, *args, **kwargs):
            pass

        def inc(self, *args, **kwargs):
            pass

        def dec(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class _MockHistogram:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def time(self):
            return contextmanager(lambda: iter([None]))()

        def labels(self, *args, **kwargs):
            return self

    class _MockSummary:
        def __init__(self, *args, **kwargs):
            pass

        def observe(self, *args, **kwargs):
            pass

        def time(self):
            return contextmanager(lambda: iter([None]))()

        def labels(self, *args, **kwargs):
            return self

    class _MockInfo:
        def __init__(self, *args, **kwargs):
            pass

        def info(self, *args, **kwargs):
            pass

        def labels(self, *args, **kwargs):
            return self

    class _MockCollectorRegistry:
        def __init__(self, *args, **kwargs):
            pass

        def register(self, *args, **kwargs):
            pass

        def unregister(self, *args, **kwargs):
            pass

    # Assign mock classes to original names
    Counter = _MockCounter
    Gauge = _MockGauge
    Histogram = _MockHistogram
    Summary = _MockSummary
    Info = _MockInfo
    CollectorRegistry = _MockCollectorRegistry
    REGISTRY = _MockCollectorRegistry()
    generate_latest = lambda *args, **kwargs: b""
    multiprocess = None


class MLMetricsCollector:
    """Comprehensive metrics collection for ML operations."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self._setup_metrics()
        self._system_info_updated = False

    def _setup_metrics(self):
        """Initialize all Prometheus metrics."""

        # System and application info
        self.system_info = Info(
            "occupancy_predictor_system_info",
            "System information",
            registry=self.registry,
        )

        # Prediction metrics
        self.prediction_requests_total = Counter(
            "occupancy_predictor_predictions_total",
            "Total number of prediction requests",
            ["room_id", "prediction_type", "status"],
            registry=self.registry,
        )

        self.prediction_latency = Histogram(
            "occupancy_predictor_prediction_latency_seconds",
            "Time spent generating predictions",
            ["room_id", "prediction_type", "model_type"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
            registry=self.registry,
        )

        self.prediction_accuracy = Gauge(
            "occupancy_predictor_prediction_accuracy_minutes",
            "Prediction accuracy in minutes (lower is better)",
            ["room_id", "prediction_type", "model_type"],
            registry=self.registry,
        )

        self.prediction_confidence = Histogram(
            "occupancy_predictor_prediction_confidence",
            "Distribution of prediction confidence scores",
            ["room_id", "prediction_type"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99],
            registry=self.registry,
        )

        # Model training and lifecycle metrics
        self.model_training_duration = Histogram(
            "occupancy_predictor_model_training_seconds",
            "Time spent training models",
            ["room_id", "model_type", "training_type"],
            buckets=[1, 5, 10, 30, 60, 300, 600, 1800, 3600],
            registry=self.registry,
        )

        self.model_retraining_count = Counter(
            "occupancy_predictor_model_retraining_total",
            "Number of model retraining events",
            ["room_id", "model_type", "trigger_reason"],
            registry=self.registry,
        )

        self.model_accuracy_score = Gauge(
            "occupancy_predictor_model_accuracy_score",
            "Model accuracy score (0-1)",
            ["room_id", "model_type", "metric_type"],
            registry=self.registry,
        )

        self.active_models_count = Gauge(
            "occupancy_predictor_active_models_total",
            "Number of active models",
            ["room_id", "model_type"],
            registry=self.registry,
        )

        # Data processing metrics
        self.events_processed_total = Counter(
            "occupancy_predictor_events_processed_total",
            "Total events processed from Home Assistant",
            ["room_id", "sensor_type", "status"],
            registry=self.registry,
        )

        self.feature_computation_duration = Histogram(
            "occupancy_predictor_feature_computation_seconds",
            "Time spent computing features",
            ["room_id", "feature_type"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            registry=self.registry,
        )

        self.database_operations_duration = Histogram(
            "occupancy_predictor_database_operations_seconds",
            "Database operation duration",
            ["operation_type", "table", "status"],
            buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0],
            registry=self.registry,
        )

        # Concept drift and adaptation metrics
        self.concept_drift_detected = Counter(
            "occupancy_predictor_concept_drift_detected_total",
            "Number of concept drift detections",
            ["room_id", "drift_type", "severity"],
            registry=self.registry,
        )

        self.drift_severity_score = Gauge(
            "occupancy_predictor_drift_severity_score",
            "Current concept drift severity score",
            ["room_id", "drift_type"],
            registry=self.registry,
        )

        self.adaptation_actions_total = Counter(
            "occupancy_predictor_adaptation_actions_total",
            "Number of adaptation actions taken",
            ["room_id", "action_type", "trigger"],
            registry=self.registry,
        )

        # Integration metrics
        self.mqtt_messages_published = Counter(
            "occupancy_predictor_mqtt_messages_published_total",
            "MQTT messages published to Home Assistant",
            ["topic_type", "room_id", "status"],
            registry=self.registry,
        )

        self.ha_api_requests_total = Counter(
            "occupancy_predictor_ha_api_requests_total",
            "Home Assistant API requests",
            ["endpoint", "method", "status"],
            registry=self.registry,
        )

        self.ha_connection_status = Gauge(
            "occupancy_predictor_ha_connection_status",
            "Home Assistant connection status (1=connected, 0=disconnected)",
            ["connection_type"],
            registry=self.registry,
        )

        # System resource metrics
        self.cpu_usage_percent = Gauge(
            "occupancy_predictor_cpu_usage_percent",
            "CPU usage percentage",
            registry=self.registry,
        )

        self.memory_usage_bytes = Gauge(
            "occupancy_predictor_memory_usage_bytes",
            "Memory usage in bytes",
            ["type"],  # rss, vms, shared
            registry=self.registry,
        )

        self.disk_usage_bytes = Gauge(
            "occupancy_predictor_disk_usage_bytes",
            "Disk usage in bytes",
            ["path", "type"],  # total, used, free
            registry=self.registry,
        )

        # Error and exception metrics
        self.errors_total = Counter(
            "occupancy_predictor_errors_total",
            "Total number of errors",
            ["error_type", "component", "severity"],
            registry=self.registry,
        )

        self.last_error_timestamp = Gauge(
            "occupancy_predictor_last_error_timestamp",
            "Timestamp of last error",
            ["error_type", "component"],
            registry=self.registry,
        )

        # Performance and health metrics
        self.system_health_score = Gauge(
            "occupancy_predictor_system_health_score",
            "Overall system health score (0-1)",
            registry=self.registry,
        )

        self.prediction_queue_size = Gauge(
            "occupancy_predictor_prediction_queue_size",
            "Number of predictions in queue",
            ["queue_type"],
            registry=self.registry,
        )

        self.uptime_seconds = Gauge(
            "occupancy_predictor_uptime_seconds",
            "Application uptime in seconds",
            registry=self.registry,
        )

    def update_system_info(self):
        """Update system information metrics."""
        if not self._system_info_updated:
            import sys

            import platform

            self.system_info.info(
                {
                    "version": "1.0.0",
                    "python_version": sys.version,
                    "platform": platform.platform(),
                    "architecture": platform.architecture()[0],
                    "processor": platform.processor(),
                    "hostname": platform.node(),
                }
            )
            self._system_info_updated = True

    def record_prediction(
        self,
        room_id: str,
        prediction_type: str,
        model_type: str,
        duration: float,
        accuracy_minutes: Optional[float] = None,
        confidence: Optional[float] = None,
        status: str = "success",
    ):
        """Record prediction metrics."""
        self.prediction_requests_total.labels(
            room_id=room_id, prediction_type=prediction_type, status=status
        ).inc()

        self.prediction_latency.labels(
            room_id=room_id, prediction_type=prediction_type, model_type=model_type
        ).observe(duration)

        if accuracy_minutes is not None:
            self.prediction_accuracy.labels(
                room_id=room_id, prediction_type=prediction_type, model_type=model_type
            ).set(accuracy_minutes)

        if confidence is not None:
            self.prediction_confidence.labels(
                room_id=room_id, prediction_type=prediction_type
            ).observe(confidence)

    def record_model_training(
        self,
        room_id: str,
        model_type: str,
        training_type: str,
        duration: float,
        accuracy_metrics: Optional[Dict[str, float]] = None,
        trigger_reason: str = "scheduled",
    ):
        """Record model training metrics."""
        self.model_training_duration.labels(
            room_id=room_id, model_type=model_type, training_type=training_type
        ).observe(duration)

        self.model_retraining_count.labels(
            room_id=room_id, model_type=model_type, trigger_reason=trigger_reason
        ).inc()

        if accuracy_metrics:
            for metric_name, score in accuracy_metrics.items():
                self.model_accuracy_score.labels(
                    room_id=room_id, model_type=model_type, metric_type=metric_name
                ).set(score)

    def record_concept_drift(
        self, room_id: str, drift_type: str, severity: float, action_taken: str
    ):
        """Record concept drift detection."""
        severity_label = (
            "high" if severity > 0.7 else "medium" if severity > 0.3 else "low"
        )

        self.concept_drift_detected.labels(
            room_id=room_id, drift_type=drift_type, severity=severity_label
        ).inc()

        self.drift_severity_score.labels(room_id=room_id, drift_type=drift_type).set(
            severity
        )

        self.adaptation_actions_total.labels(
            room_id=room_id, action_type=action_taken, trigger="concept_drift"
        ).inc()

    def record_event_processing(
        self,
        room_id: str,
        sensor_type: str,
        processing_duration: float,
        status: str = "success",
    ):
        """Record event processing metrics."""
        self.events_processed_total.labels(
            room_id=room_id, sensor_type=sensor_type, status=status
        ).inc()

    def record_feature_computation(
        self, room_id: str, feature_type: str, duration: float
    ):
        """Record feature computation metrics."""
        self.feature_computation_duration.labels(
            room_id=room_id, feature_type=feature_type
        ).observe(duration)

    def record_database_operation(
        self, operation_type: str, table: str, duration: float, status: str = "success"
    ):
        """Record database operation metrics."""
        self.database_operations_duration.labels(
            operation_type=operation_type, table=table, status=status
        ).observe(duration)

    def record_mqtt_publish(
        self, topic_type: str, room_id: str, status: str = "success"
    ):
        """Record MQTT publishing metrics."""
        self.mqtt_messages_published.labels(
            topic_type=topic_type, room_id=room_id, status=status
        ).inc()

    def record_ha_api_request(self, endpoint: str, method: str, status: str):
        """Record Home Assistant API request metrics."""
        self.ha_api_requests_total.labels(
            endpoint=endpoint, method=method, status=status
        ).inc()

    def update_ha_connection_status(self, connection_type: str, connected: bool):
        """Update Home Assistant connection status."""
        self.ha_connection_status.labels(connection_type=connection_type).set(
            1 if connected else 0
        )

    def record_error(self, error_type: str, component: str, severity: str):
        """Record error occurrence."""
        self.errors_total.labels(
            error_type=error_type, component=component, severity=severity
        ).inc()

        self.last_error_timestamp.labels(
            error_type=error_type, component=component
        ).set(time.time())

    def update_system_resources(self):
        """Update system resource metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=None)
            self.cpu_usage_percent.set(cpu_percent)

            # Memory usage
            # memory = psutil.virtual_memory()  # Available if system memory metrics needed
            process = psutil.Process()
            process_memory = process.memory_info()

            self.memory_usage_bytes.labels(type="rss").set(process_memory.rss)
            self.memory_usage_bytes.labels(type="vms").set(process_memory.vms)

            # Disk usage
            disk_usage = psutil.disk_usage("/")
            self.disk_usage_bytes.labels(path="/", type="total").set(disk_usage.total)
            self.disk_usage_bytes.labels(path="/", type="used").set(disk_usage.used)
            self.disk_usage_bytes.labels(path="/", type="free").set(disk_usage.free)

        except Exception:
            # Silently ignore errors in metrics collection
            pass

    def update_active_models_count(self, room_id: str, model_type: str, count: int):
        """Update active models count."""
        self.active_models_count.labels(room_id=room_id, model_type=model_type).set(
            count
        )

    def update_prediction_queue_size(self, queue_type: str, size: int):
        """Update prediction queue size."""
        self.prediction_queue_size.labels(queue_type=queue_type).set(size)

    def update_system_health_score(self, score: float):
        """Update overall system health score."""
        self.system_health_score.set(max(0.0, min(1.0, score)))

    def update_uptime(self, start_time: datetime):
        """Update application uptime."""
        uptime = (datetime.now() - start_time).total_seconds()
        self.uptime_seconds.set(uptime)

    @contextmanager
    def time_operation(self, room_id: str, operation_type: str):
        """Context manager to time operations."""
        # start_time = time.time()  # Available for timing if needed
        try:
            yield
        finally:
            # duration = time.time() - start_time  # Available for operation-specific metrics
            # You can extend this to record operation-specific metrics
            pass


class MetricsManager:
    """Centralized metrics management and collection."""

    def __init__(self, registry: Optional[CollectorRegistry] = None):
        self.registry = registry or REGISTRY
        self.collector = MLMetricsCollector(self.registry)
        self.start_time = datetime.now()
        self._resource_update_thread: Optional[threading.Thread] = None
        self._running = False

    def start_background_collection(self, update_interval: int = 30):
        """Start background thread for periodic metrics collection."""
        if self._running:
            return

        self._running = True
        self.collector.update_system_info()

        def update_loop():
            while self._running:
                try:
                    self.collector.update_system_resources()
                    self.collector.update_uptime(self.start_time)
                    time.sleep(update_interval)
                except Exception:
                    # Silently continue on metrics collection errors
                    pass

        self._resource_update_thread = threading.Thread(
            target=update_loop, daemon=True, name="MetricsUpdater"
        )
        self._resource_update_thread.start()

    def stop_background_collection(self):
        """Stop background metrics collection."""
        self._running = False
        if self._resource_update_thread and self._resource_update_thread.is_alive():
            self._resource_update_thread.join(timeout=5)

    def get_metrics(self) -> str:
        """Get current metrics in Prometheus format."""
        if PROMETHEUS_AVAILABLE:
            result: str = generate_latest(self.registry).decode("utf-8")
            return result
        else:
            return "# Prometheus client not available\n"

    def get_collector(self) -> MLMetricsCollector:
        """Get the metrics collector instance."""
        return self.collector


# Global metrics manager instance
_metrics_manager = None


def get_metrics_manager() -> MetricsManager:
    """Get global metrics manager instance."""
    global _metrics_manager
    if _metrics_manager is None:
        _metrics_manager = MetricsManager()
    return _metrics_manager


def get_metrics_collector() -> MLMetricsCollector:
    """Convenience function to get metrics collector."""
    return get_metrics_manager().get_collector()


def metrics_endpoint_handler():
    """Handler for Prometheus metrics endpoint."""
    return get_metrics_manager().get_metrics()


def time_prediction(room_id: str, prediction_type: str, model_type: str):
    """Decorator to time prediction operations."""

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Extract confidence if available
                confidence = None
                if isinstance(result, dict) and "confidence" in result:
                    confidence = result["confidence"]

                get_metrics_collector().record_prediction(
                    room_id=room_id,
                    prediction_type=prediction_type,
                    model_type=model_type,
                    duration=duration,
                    confidence=confidence,
                    status="success",
                )
                return result

            except Exception:
                duration = time.time() - start_time
                get_metrics_collector().record_prediction(
                    room_id=room_id,
                    prediction_type=prediction_type,
                    model_type=model_type,
                    duration=duration,
                    status="error",
                )
                raise

        return wrapper

    return decorator


# Multi-process metrics support
class MultiProcessMetricsManager:
    """
    Manages metrics collection across multiple processes using prometheus_client multiprocess support.

    This enables metrics collection in multi-process deployments such as gunicorn workers.
    """

    def __init__(self):
        """Initialize multi-process metrics manager."""
        self.multiprocess_enabled = PROMETHEUS_AVAILABLE and hasattr(
            multiprocess, "MultiProcessCollector"
        )
        self.registry = None

        if self.multiprocess_enabled:
            # Create a separate registry for multi-process metrics
            self.registry = CollectorRegistry()
            # Add multiprocess collector
            multiprocess.MultiProcessCollector(self.registry)

    def is_multiprocess_enabled(self) -> bool:
        """Check if multi-process metrics collection is enabled."""
        return self.multiprocess_enabled

    def get_multiprocess_registry(self) -> Optional[CollectorRegistry]:
        """Get the multi-process registry."""
        return self.registry

    def aggregate_multiprocess_metrics(self) -> Dict[str, Any]:
        """
        Aggregate metrics from all processes.

        Returns:
            Dictionary containing aggregated metrics from all processes.
        """
        if not self.multiprocess_enabled or self.registry is None:
            return {"error": "Multi-process metrics not available"}

        try:
            # Use multiprocess.aggregate to combine metrics from all processes
            aggregated = {}

            # Get all metric families from the registry
            for metric_family in self.registry.collect():
                family_data = {
                    "name": metric_family.name,
                    "help": metric_family.help,
                    "type": metric_family.type,
                    "samples": [],
                }

                for sample in metric_family.samples:
                    family_data["samples"].append(
                        {
                            "name": sample.name,
                            "labels": dict(sample.labels),
                            "value": sample.value,
                            "timestamp": sample.timestamp,
                        }
                    )

                aggregated[metric_family.name] = family_data

            return aggregated

        except Exception as e:
            return {"error": f"Failed to aggregate multi-process metrics: {e}"}

    def generate_multiprocess_metrics(self) -> str:
        """
        Generate Prometheus-formatted metrics from all processes.

        Returns:
            Prometheus-formatted metrics string.
        """
        if not self.multiprocess_enabled:
            return "# Multi-process metrics not available\n"

        try:
            result: str = generate_latest(self.registry).decode("utf-8")
            return result
        except Exception as e:
            return f"# Error generating multi-process metrics: {e}\n"

    def cleanup_dead_processes(self):
        """Clean up metrics from dead processes."""
        if not self.multiprocess_enabled:
            return

        try:
            # Use multiprocess values functionality to clean up
            multiprocess.mark_process_dead(None)  # Clean up current process
        except Exception:
            # Log error but don't fail
            pass


# Global multi-process metrics manager
_multiprocess_manager = None


def get_multiprocess_metrics_manager() -> MultiProcessMetricsManager:
    """Get the global multi-process metrics manager."""
    global _multiprocess_manager
    if _multiprocess_manager is None:
        _multiprocess_manager = MultiProcessMetricsManager()
    return _multiprocess_manager


def setup_multiprocess_metrics():
    """
    Set up multi-process metrics collection.

    This should be called once at application startup in multi-process environments.
    """
    manager = get_multiprocess_metrics_manager()

    if manager.is_multiprocess_enabled():
        # Clean up any existing metrics from previous runs
        manager.cleanup_dead_processes()

        # Ensure metrics collector uses multi-process aware metrics
        collector = get_metrics_collector()
        if hasattr(collector, "multiprocess_registry"):
            collector.multiprocess_registry = manager.get_multiprocess_registry()


def get_aggregated_metrics() -> Dict[str, Any]:
    """
    Get aggregated metrics from all processes.

    This is the main function to call for getting comprehensive metrics
    in multi-process deployments.

    Returns:
        Dictionary containing all aggregated metrics.
    """
    manager = get_multiprocess_metrics_manager()

    if manager.is_multiprocess_enabled():
        # Return aggregated multi-process metrics
        multiprocess_metrics = manager.aggregate_multiprocess_metrics()

        # Also include single-process metrics for completeness
        single_process_metrics = get_metrics_manager().get_metrics()

        return {
            "multiprocess_metrics": multiprocess_metrics,
            "single_process_metrics": single_process_metrics,
            "collection_mode": "multiprocess",
        }
    else:
        # Fall back to single-process metrics
        return {
            "single_process_metrics": get_metrics_manager().get_metrics(),
            "collection_mode": "single_process",
        }


def export_multiprocess_metrics() -> str:
    """
    Export metrics in Prometheus format for multi-process deployments.

    Returns:
        Prometheus-formatted metrics string suitable for scraping.
    """
    manager = get_multiprocess_metrics_manager()

    if manager.is_multiprocess_enabled():
        return manager.generate_multiprocess_metrics()
    else:
        # Fall back to single-process export
        metrics_manager = get_metrics_manager()
        return metrics_manager.get_metrics()
