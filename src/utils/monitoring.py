"""
Performance monitoring and system health management for Home Assistant ML Predictor.
Provides real-time performance tracking, alerting, and health checks.
"""

import asyncio
import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics

from .logger import get_logger, get_performance_logger, get_error_tracker
from .metrics import get_metrics_collector


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""

    name: str
    warning_threshold: float
    critical_threshold: float
    unit: str
    description: str


@dataclass
class HealthCheckResult:
    """Health check result."""

    component: str
    status: str  # 'healthy', 'warning', 'critical'
    response_time: float
    message: str
    details: Dict[str, Any] = None


@dataclass
class AlertEvent:
    """Alert event information."""

    timestamp: datetime
    alert_type: str  # 'warning', 'critical'
    component: str
    message: str
    metric_name: str
    current_value: float
    threshold: float
    additional_info: Dict[str, Any] = None


class PerformanceMonitor:
    """Real-time performance monitoring and alerting."""

    def __init__(self):
        self.logger = get_logger("performance_monitor")
        self.perf_logger = get_performance_logger()
        self.error_tracker = get_error_tracker()
        self.metrics_collector = get_metrics_collector()

        # Performance history for trend analysis
        self.performance_history = defaultdict(lambda: deque(maxlen=1000))

        # Alert callbacks
        self.alert_callbacks: List[Callable[[AlertEvent], None]] = []

        # Performance thresholds
        self.thresholds = {
            "prediction_latency": PerformanceThreshold(
                "prediction_latency",
                0.5,
                2.0,
                "seconds",
                "Time to generate predictions",
            ),
            "feature_computation_time": PerformanceThreshold(
                "feature_computation_time",
                0.2,
                1.0,
                "seconds",
                "Time to compute features",
            ),
            "model_training_time": PerformanceThreshold(
                "model_training_time", 300, 1800, "seconds", "Time to train models"
            ),
            "database_query_time": PerformanceThreshold(
                "database_query_time",
                0.1,
                0.5,
                "seconds",
                "Database query response time",
            ),
            "prediction_accuracy": PerformanceThreshold(
                "prediction_accuracy", 20, 30, "minutes", "Prediction accuracy error"
            ),
            "cpu_usage": PerformanceThreshold(
                "cpu_usage", 70, 85, "percent", "CPU usage percentage"
            ),
            "memory_usage": PerformanceThreshold(
                "memory_usage", 70, 85, "percent", "Memory usage percentage"
            ),
            "disk_usage": PerformanceThreshold(
                "disk_usage", 80, 90, "percent", "Disk usage percentage"
            ),
            "error_rate": PerformanceThreshold(
                "error_rate", 0.05, 0.10, "ratio", "Error rate (errors per request)"
            ),
        }

    def add_alert_callback(self, callback: Callable[[AlertEvent], None]):
        """Add callback for alert notifications."""
        self.alert_callbacks.append(callback)

    def record_performance_metric(
        self,
        metric_name: str,
        value: float,
        room_id: Optional[str] = None,
        additional_info: Dict[str, Any] = None,
    ):
        """Record a performance metric and check thresholds."""
        timestamp = datetime.now()

        # Store in history
        key = f"{metric_name}_{room_id}" if room_id else metric_name
        self.performance_history[key].append((timestamp, value))

        # Log performance metric
        self.perf_logger.log_operation_time(
            operation=metric_name,
            duration=value,
            room_id=room_id,
            **(additional_info or {}),
        )

        # Check thresholds
        if metric_name in self.thresholds:
            self._check_threshold(metric_name, value, room_id, additional_info)

    def _check_threshold(
        self,
        metric_name: str,
        value: float,
        room_id: Optional[str] = None,
        additional_info: Dict[str, Any] = None,
    ):
        """Check if metric value exceeds thresholds."""
        threshold = self.thresholds[metric_name]
        alert_type = None

        if value >= threshold.critical_threshold:
            alert_type = "critical"
        elif value >= threshold.warning_threshold:
            alert_type = "warning"

        if alert_type:
            alert = AlertEvent(
                timestamp=datetime.now(),
                alert_type=alert_type,
                component=room_id or "system",
                message=f"{threshold.description} exceeded: {value:.2f} {threshold.unit}",
                metric_name=metric_name,
                current_value=value,
                threshold=(
                    threshold.critical_threshold
                    if alert_type == "critical"
                    else threshold.warning_threshold
                ),
                additional_info=additional_info or {},
            )

            self._trigger_alert(alert)

    def _trigger_alert(self, alert: AlertEvent):
        """Trigger alert through configured callbacks."""
        self.logger.warning(
            f"Performance alert: {alert.message}",
            extra={
                "alert_type": alert.alert_type,
                "component": alert.component,
                "metric_name": alert.metric_name,
                "current_value": alert.current_value,
                "threshold": alert.threshold,
            },
        )

        # Record alert in metrics
        self.metrics_collector.record_error(
            error_type=f"performance_{alert.alert_type}",
            component=alert.component,
            severity=alert.alert_type,
        )

        # Notify callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.error_tracker.track_error(
                    e, {"component": "alert_system", "callback": str(callback)}
                )

    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary for the specified time window."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        summary = {}

        for key, history in self.performance_history.items():
            # Filter recent data
            recent_data = [(ts, value) for ts, value in history if ts >= cutoff_time]

            if recent_data:
                values = [value for _, value in recent_data]
                summary[key] = {
                    "count": len(values),
                    "mean": statistics.mean(values),
                    "median": statistics.median(values),
                    "std": statistics.stdev(values) if len(values) > 1 else 0,
                    "min": min(values),
                    "max": max(values),
                    "p95": self._percentile(values, 95),
                    "p99": self._percentile(values, 99),
                }

        return summary

    def _percentile(self, values: List[float], percentile: float) -> float:
        """Calculate percentile of values."""
        if not values:
            return 0.0
        sorted_values = sorted(values)
        k = (len(sorted_values) - 1) * (percentile / 100)
        f = int(k)
        c = k - f
        if f == len(sorted_values) - 1:
            return sorted_values[f]
        return sorted_values[f] * (1 - c) + sorted_values[f + 1] * c

    def get_trend_analysis(
        self, metric_name: str, room_id: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """Analyze performance trends for a metric."""
        key = f"{metric_name}_{room_id}" if room_id else metric_name

        if key not in self.performance_history:
            return {"status": "no_data"}

        cutoff_time = datetime.now() - timedelta(hours=hours)
        recent_data = [
            (ts, value)
            for ts, value in self.performance_history[key]
            if ts >= cutoff_time
        ]

        if len(recent_data) < 2:
            return {"status": "insufficient_data"}

        # Calculate trend
        timestamps = [(ts - cutoff_time).total_seconds() for ts, _ in recent_data]
        values = [value for _, value in recent_data]

        # Simple linear regression for trend
        n = len(values)
        sum_x = sum(timestamps)
        sum_y = sum(values)
        sum_xy = sum(x * y for x, y in zip(timestamps, values))
        sum_x2 = sum(x * x for x in timestamps)

        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        return {
            "status": "success",
            "trend": (
                "improving" if slope < 0 else "degrading" if slope > 0 else "stable"
            ),
            "slope": slope,
            "data_points": len(recent_data),
            "current_mean": (
                statistics.mean(values[-10:])
                if len(values) >= 10
                else statistics.mean(values)
            ),
            "baseline_mean": (
                statistics.mean(values[:10])
                if len(values) >= 20
                else statistics.mean(values)
            ),
        }


class SystemHealthMonitor:
    """System health monitoring and diagnostics."""

    def __init__(self):
        self.logger = get_logger("health_monitor")
        self.metrics_collector = get_metrics_collector()
        self.health_checks: Dict[str, Callable[[], HealthCheckResult]] = {}
        self.last_health_check = {}

        # Register default health checks
        self._register_default_health_checks()

    def register_health_check(
        self, name: str, check_func: Callable[[], HealthCheckResult]
    ):
        """Register a custom health check."""
        self.health_checks[name] = check_func

    def _register_default_health_checks(self):
        """Register default system health checks."""
        self.health_checks.update(
            {
                "system_resources": self._check_system_resources,
                "disk_space": self._check_disk_space,
                "memory_usage": self._check_memory_usage,
                "cpu_usage": self._check_cpu_usage,
            }
        )

    def _check_system_resources(self) -> HealthCheckResult:
        """Check overall system resource health."""
        start_time = time.time()

        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()

            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 90:
                status = "critical"
                message = f"High resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            elif cpu_percent > 70 or memory.percent > 70:
                status = "warning"
                message = f"Elevated resource usage: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"
            else:
                status = "healthy"
                message = f"Resource usage normal: CPU {cpu_percent:.1f}%, Memory {memory.percent:.1f}%"

            response_time = time.time() - start_time

            return HealthCheckResult(
                component="system_resources",
                status=status,
                response_time=response_time,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available": memory.available,
                    "memory_total": memory.total,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                component="system_resources",
                status="critical",
                response_time=time.time() - start_time,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)},
            )

    def _check_disk_space(self) -> HealthCheckResult:
        """Check disk space availability."""
        start_time = time.time()

        try:
            disk_usage = psutil.disk_usage("/")
            usage_percent = (disk_usage.used / disk_usage.total) * 100

            if usage_percent > 95:
                status = "critical"
                message = f"Critical disk space: {usage_percent:.1f}% used"
            elif usage_percent > 85:
                status = "warning"
                message = f"Low disk space: {usage_percent:.1f}% used"
            else:
                status = "healthy"
                message = f"Disk space healthy: {usage_percent:.1f}% used"

            return HealthCheckResult(
                component="disk_space",
                status=status,
                response_time=time.time() - start_time,
                message=message,
                details={
                    "total": disk_usage.total,
                    "used": disk_usage.used,
                    "free": disk_usage.free,
                    "percent": usage_percent,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                component="disk_space",
                status="critical",
                response_time=time.time() - start_time,
                message=f"Failed to check disk space: {e}",
                details={"error": str(e)},
            )

    def _check_memory_usage(self) -> HealthCheckResult:
        """Check memory usage."""
        start_time = time.time()

        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            system_memory = psutil.virtual_memory()

            process_percent = (process_memory.rss / system_memory.total) * 100

            if process_percent > 50:
                status = "warning"
                message = f"High memory usage: {process_percent:.1f}% of system memory"
            elif process_percent > 25:
                status = "warning"
                message = (
                    f"Moderate memory usage: {process_percent:.1f}% of system memory"
                )
            else:
                status = "healthy"
                message = (
                    f"Memory usage normal: {process_percent:.1f}% of system memory"
                )

            return HealthCheckResult(
                component="memory_usage",
                status=status,
                response_time=time.time() - start_time,
                message=message,
                details={
                    "process_rss": process_memory.rss,
                    "process_vms": process_memory.vms,
                    "system_total": system_memory.total,
                    "system_available": system_memory.available,
                    "process_percent": process_percent,
                },
            )

        except Exception as e:
            return HealthCheckResult(
                component="memory_usage",
                status="critical",
                response_time=time.time() - start_time,
                message=f"Failed to check memory usage: {e}",
                details={"error": str(e)},
            )

    def _check_cpu_usage(self) -> HealthCheckResult:
        """Check CPU usage."""
        start_time = time.time()

        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()

            if cpu_percent > 90:
                status = "critical"
                message = f"Critical CPU usage: {cpu_percent:.1f}%"
            elif cpu_percent > 70:
                status = "warning"
                message = f"High CPU usage: {cpu_percent:.1f}%"
            else:
                status = "healthy"
                message = f"CPU usage normal: {cpu_percent:.1f}%"

            return HealthCheckResult(
                component="cpu_usage",
                status=status,
                response_time=time.time() - start_time,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "cpu_count": cpu_count,
                    "load_avg": (
                        psutil.getloadavg() if hasattr(psutil, "getloadavg") else None
                    ),
                },
            )

        except Exception as e:
            return HealthCheckResult(
                component="cpu_usage",
                status="critical",
                response_time=time.time() - start_time,
                message=f"Failed to check CPU usage: {e}",
                details={"error": str(e)},
            )

    async def run_health_checks(self) -> Dict[str, HealthCheckResult]:
        """Run all registered health checks."""
        results = {}

        for name, check_func in self.health_checks.items():
            try:
                result = check_func()
                results[name] = result
                self.last_health_check[name] = datetime.now()

                # Update metrics
                health_score = (
                    1.0
                    if result.status == "healthy"
                    else 0.5 if result.status == "warning" else 0.0
                )
                self.metrics_collector.update_system_health_score(health_score)

            except Exception as e:
                self.logger.error(f"Health check {name} failed: {e}")
                results[name] = HealthCheckResult(
                    component=name,
                    status="critical",
                    response_time=0,
                    message=f"Health check failed: {e}",
                    details={"error": str(e)},
                )

        return results

    def get_overall_health_status(self) -> Tuple[str, Dict[str, Any]]:
        """Get overall system health status."""
        if not self.last_health_check:
            return "unknown", {"message": "No health checks run yet"}

        # Check if any checks are outdated (> 5 minutes)
        cutoff = datetime.now() - timedelta(minutes=5)
        outdated_checks = [
            name
            for name, last_check in self.last_health_check.items()
            if last_check < cutoff
        ]

        if outdated_checks:
            return "warning", {
                "message": "Some health checks are outdated",
                "outdated_checks": outdated_checks,
            }

        # Run quick health assessment
        try:
            cpu_percent = psutil.cpu_percent(interval=None)
            memory_percent = psutil.virtual_memory().percent

            if cpu_percent > 90 or memory_percent > 90:
                status = "critical"
            elif cpu_percent > 70 or memory_percent > 70:
                status = "warning"
            else:
                status = "healthy"

            return status, {
                "cpu_percent": cpu_percent,
                "memory_percent": memory_percent,
                "last_health_checks": {
                    name: check_time.isoformat()
                    for name, check_time in self.last_health_check.items()
                },
            }

        except Exception as e:
            return "critical", {
                "message": f"Failed to assess system health: {e}",
                "error": str(e),
            }


class MonitoringManager:
    """Centralized monitoring management."""

    def __init__(self):
        self.logger = get_logger("monitoring_manager")
        self.performance_monitor = PerformanceMonitor()
        self.health_monitor = SystemHealthMonitor()
        self.metrics_collector = get_metrics_collector()

        self._monitoring_task = None
        self._running = False

    async def start_monitoring(
        self,
        health_check_interval: int = 300,  # 5 minutes
        performance_summary_interval: int = 900,
    ):  # 15 minutes
        """Start continuous monitoring."""
        if self._running:
            return

        self._running = True
        self.logger.info("Starting monitoring system")

        self._monitoring_task = asyncio.create_task(
            self._monitoring_loop(health_check_interval, performance_summary_interval)
        )

    async def stop_monitoring(self):
        """Stop continuous monitoring."""
        if not self._running:
            return

        self._running = False
        self.logger.info("Stopping monitoring system")

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

    async def _monitoring_loop(
        self, health_check_interval: int, performance_summary_interval: int
    ):
        """Main monitoring loop."""
        last_health_check = 0
        last_performance_summary = 0

        while self._running:
            try:
                current_time = time.time()

                # Run health checks
                if current_time - last_health_check >= health_check_interval:
                    await self.health_monitor.run_health_checks()
                    last_health_check = current_time

                # Generate performance summary
                if (
                    current_time - last_performance_summary
                    >= performance_summary_interval
                ):
                    summary = self.performance_monitor.get_performance_summary()
                    self.logger.info(
                        "Performance summary generated",
                        extra={
                            "summary": summary,
                            "metric_type": "performance_summary",
                        },
                    )
                    last_performance_summary = current_time

                await asyncio.sleep(30)  # Check every 30 seconds

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    def get_performance_monitor(self) -> PerformanceMonitor:
        """Get performance monitor instance."""
        return self.performance_monitor

    def get_health_monitor(self) -> SystemHealthMonitor:
        """Get health monitor instance."""
        return self.health_monitor

    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get current monitoring status."""
        health_results = await self.health_monitor.run_health_checks()
        health_status, health_details = self.health_monitor.get_overall_health_status()
        performance_summary = self.performance_monitor.get_performance_summary(hours=1)

        return {
            "monitoring_active": self._running,
            "health_status": health_status,
            "health_details": health_details,
            "health_checks": {
                name: {
                    "status": result.status,
                    "message": result.message,
                    "response_time": result.response_time,
                }
                for name, result in health_results.items()
            },
            "performance_summary": performance_summary,
            "timestamp": datetime.now().isoformat(),
        }


# Global monitoring manager instance
_monitoring_manager = None


def get_monitoring_manager() -> MonitoringManager:
    """Get global monitoring manager instance."""
    global _monitoring_manager
    if _monitoring_manager is None:
        _monitoring_manager = MonitoringManager()
    return _monitoring_manager
