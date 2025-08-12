"""
Production Health Monitoring & Metrics Collection for Home Assistant ML Predictor.

Provides comprehensive system observability with automated health checks, 
resource monitoring, database connection monitoring, MQTT/API endpoint monitoring, 
and automated incident response capabilities.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import socket
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import aiohttp
import psutil
import statistics

from ..core.config import get_config
from .alerts import get_alert_manager
from .logger import get_error_tracker, get_logger, get_performance_logger
from .metrics import get_metrics_collector


class HealthStatus(Enum):
    """Health status levels."""

    HEALTHY = "healthy"
    WARNING = "warning"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Component types for health monitoring."""

    DATABASE = "database"
    MQTT = "mqtt"
    API = "api"
    SYSTEM = "system"
    NETWORK = "network"
    APPLICATION = "application"
    EXTERNAL = "external"


@dataclass
class HealthThresholds:
    """Health monitoring thresholds."""

    cpu_warning: float = 70.0
    cpu_critical: float = 85.0
    memory_warning: float = 70.0
    memory_critical: float = 85.0
    disk_warning: float = 80.0
    disk_critical: float = 90.0
    response_time_warning: float = 1.0  # seconds
    response_time_critical: float = 5.0  # seconds
    error_rate_warning: float = 0.05  # 5%
    error_rate_critical: float = 0.10  # 10%
    uptime_warning: float = 0.95  # 95%
    uptime_critical: float = 0.90  # 90%


@dataclass
class ComponentHealth:
    """Health status for a system component."""

    component_name: str
    component_type: ComponentType
    status: HealthStatus
    last_check: datetime
    response_time: float
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    error_count: int = 0
    consecutive_failures: int = 0
    uptime_percentage: float = 100.0
    metrics: Dict[str, float] = field(default_factory=dict)

    def is_healthy(self) -> bool:
        """Check if component is healthy."""
        return self.status in [HealthStatus.HEALTHY, HealthStatus.WARNING]

    def needs_attention(self) -> bool:
        """Check if component needs attention."""
        return self.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "component_name": self.component_name,
            "component_type": self.component_type.value,
            "status": self.status.value,
            "last_check": self.last_check.isoformat(),
            "response_time": self.response_time,
            "message": self.message,
            "details": self.details,
            "error_count": self.error_count,
            "consecutive_failures": self.consecutive_failures,
            "uptime_percentage": self.uptime_percentage,
            "metrics": self.metrics,
            "is_healthy": self.is_healthy(),
            "needs_attention": self.needs_attention(),
        }


@dataclass
class SystemHealth:
    """Overall system health status."""

    overall_status: HealthStatus
    component_count: int
    healthy_components: int
    degraded_components: int
    critical_components: int
    last_updated: datetime
    uptime_seconds: float
    system_load: float
    total_memory_gb: float
    available_memory_gb: float
    cpu_usage: float
    active_alerts: int
    performance_score: float  # 0-100

    def health_score(self) -> float:
        """Calculate overall health score (0-100)."""
        if self.component_count == 0:
            return 0.0

        healthy_ratio = self.healthy_components / self.component_count
        performance_factor = self.performance_score / 100.0

        # Weight components and performance equally
        score = (healthy_ratio * 0.7 + performance_factor * 0.3) * 100

        # Penalize for critical alerts
        if self.active_alerts > 0:
            alert_penalty = min(20.0, self.active_alerts * 5.0)
            score = max(0.0, score - alert_penalty)

        return round(score, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "overall_status": self.overall_status.value,
            "health_score": self.health_score(),
            "component_count": self.component_count,
            "healthy_components": self.healthy_components,
            "degraded_components": self.degraded_components,
            "critical_components": self.critical_components,
            "last_updated": self.last_updated.isoformat(),
            "uptime_seconds": self.uptime_seconds,
            "system_load": self.system_load,
            "total_memory_gb": self.total_memory_gb,
            "available_memory_gb": self.available_memory_gb,
            "cpu_usage": self.cpu_usage,
            "active_alerts": self.active_alerts,
            "performance_score": self.performance_score,
        }


class HealthMonitor:
    """
    Comprehensive health monitoring system with automated incident response.

    Monitors system resources, database connections, MQTT/API endpoints,
    and application performance with automated alerting and recovery.
    """

    def __init__(self, check_interval: int = 30, alert_threshold: int = 3):
        self.logger = get_logger("health_monitor")
        self.perf_logger = get_performance_logger()
        self.error_tracker = get_error_tracker()
        self.metrics_collector = get_metrics_collector()
        self.alert_manager = get_alert_manager()
        self.config = get_config()

        # Configuration
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        self.thresholds = HealthThresholds()

        # Component health tracking
        self.component_health: Dict[str, ComponentHealth] = {}
        self.health_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # Monitoring state
        self._monitoring_active = False
        self._monitoring_task = None
        self._start_time = datetime.now()

        # Health check functions
        self.health_checks: Dict[str, Callable] = {}
        self._register_default_health_checks()

        # Incident response callbacks
        self.incident_callbacks: List[Callable] = []

        # Performance tracking
        self.performance_metrics = defaultdict(list)

    def _register_default_health_checks(self):
        """Register default health check functions."""
        self.health_checks.update(
            {
                "system_resources": self._check_system_resources,
                "database_connection": self._check_database_connection,
                "mqtt_broker": self._check_mqtt_broker,
                "api_endpoints": self._check_api_endpoints,
                "memory_usage": self._check_memory_usage,
                "disk_space": self._check_disk_space,
                "network_connectivity": self._check_network_connectivity,
                "application_metrics": self._check_application_metrics,
            }
        )

    def register_health_check(self, name: str, check_function: Callable):
        """Register a custom health check function."""
        self.health_checks[name] = check_function
        self.logger.info(f"Registered custom health check: {name}")

    def add_incident_callback(self, callback: Callable[[ComponentHealth], None]):
        """Add callback for incident response."""
        self.incident_callbacks.append(callback)

    async def start_monitoring(self):
        """Start continuous health monitoring."""
        if self._monitoring_active:
            self.logger.warning("Health monitoring is already active")
            return

        self._monitoring_active = True
        self._start_time = datetime.now()

        self.logger.info(
            f"Starting health monitoring with {self.check_interval}s interval",
            extra={
                "check_interval": self.check_interval,
                "registered_checks": list(self.health_checks.keys()),
                "alert_threshold": self.alert_threshold,
            },
        )

        # Start monitoring loop
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())

        # Record startup in metrics
        self.metrics_collector.increment("health_monitor_starts_total")

    async def stop_monitoring(self):
        """Stop health monitoring."""
        if not self._monitoring_active:
            return

        self._monitoring_active = False
        self.logger.info("Stopping health monitoring")

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass

        # Record shutdown in metrics
        self.metrics_collector.increment("health_monitor_stops_total")

    async def _monitoring_loop(self):
        """Main monitoring loop."""
        consecutive_errors = 0

        while self._monitoring_active:
            try:
                start_time = time.time()

                # Run all health checks
                await self._run_health_checks()

                # Update system health status
                system_health = self._calculate_system_health()

                # Check for incidents and trigger responses
                await self._process_incidents()

                # Update metrics
                self._update_monitoring_metrics(system_health)

                # Record monitoring cycle performance
                cycle_time = time.time() - start_time
                self.performance_metrics["monitoring_cycle_time"].append(cycle_time)
                self.metrics_collector.record_duration(
                    "health_check_cycle_duration_seconds", cycle_time
                )

                consecutive_errors = 0

                # Log periodic summary
                if (
                    len(self.performance_metrics["monitoring_cycle_time"]) % 20 == 0
                ):  # Every 20 cycles
                    self._log_monitoring_summary(system_health)

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                consecutive_errors += 1
                self.logger.error(
                    f"Error in health monitoring loop (attempt {consecutive_errors}): {e}",
                    extra={
                        "consecutive_errors": consecutive_errors,
                        "error_type": type(e).__name__,
                    },
                )

                self.error_tracker.track_error(
                    e,
                    {
                        "component": "health_monitor",
                        "consecutive_errors": consecutive_errors,
                    },
                )

                # Exponential backoff on errors
                sleep_time = min(
                    60, self.check_interval * (2 ** min(consecutive_errors, 5))
                )
                await asyncio.sleep(sleep_time)

    async def _run_health_checks(self):
        """Run all registered health checks."""
        tasks = []

        for name, check_func in self.health_checks.items():
            task = asyncio.create_task(self._run_single_health_check(name, check_func))
            tasks.append(task)

        # Wait for all checks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            check_name = list(self.health_checks.keys())[i]
            if isinstance(result, Exception):
                self.logger.error(f"Health check {check_name} failed: {result}")
                self._record_check_failure(check_name, result)

    async def _run_single_health_check(
        self, name: str, check_func: Callable
    ) -> ComponentHealth:
        """Run a single health check with timeout and error handling."""
        start_time = time.time()

        try:
            # Run check with timeout
            health_result = await asyncio.wait_for(
                asyncio.create_task(check_func()), timeout=30.0  # 30 second timeout
            )

            response_time = time.time() - start_time

            # Update component health
            if name in self.component_health:
                self.component_health[name].consecutive_failures = 0

            # Store result
            self.component_health[name] = health_result
            self.component_health[name].response_time = response_time
            self.component_health[name].last_check = datetime.now()

            # Record in history
            self.health_history[name].append((datetime.now(), health_result.status))

            # Update metrics
            self.metrics_collector.record_duration(
                f"health_check_{name}_duration_seconds", response_time
            )
            self.metrics_collector.set_gauge(
                f"health_check_{name}_status",
                1.0 if health_result.is_healthy() else 0.0,
            )

            return health_result

        except asyncio.TimeoutError:
            response_time = time.time() - start_time
            error_msg = f"Health check {name} timed out after {response_time:.2f}s"
            self.logger.warning(error_msg)

            health_result = ComponentHealth(
                component_name=name,
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                message=error_msg,
                consecutive_failures=self.component_health.get(
                    name,
                    ComponentHealth(
                        name,
                        ComponentType.APPLICATION,
                        HealthStatus.UNKNOWN,
                        datetime.now(),
                        0.0,
                        "",
                    ),
                ).consecutive_failures
                + 1,
            )

            self.component_health[name] = health_result
            return health_result

        except Exception as e:
            response_time = time.time() - start_time
            error_msg = f"Health check {name} failed: {e}"
            self.logger.error(error_msg)

            self.error_tracker.track_error(
                e,
                {
                    "component": "health_monitor",
                    "check_name": name,
                    "response_time": response_time,
                },
            )

            health_result = ComponentHealth(
                component_name=name,
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=response_time,
                message=error_msg,
                details={"error": str(e)},
                consecutive_failures=self.component_health.get(
                    name,
                    ComponentHealth(
                        name,
                        ComponentType.APPLICATION,
                        HealthStatus.UNKNOWN,
                        datetime.now(),
                        0.0,
                        "",
                    ),
                ).consecutive_failures
                + 1,
            )

            self.component_health[name] = health_result
            return health_result

    def _record_check_failure(self, check_name: str, error: Exception):
        """Record health check failure."""
        if check_name in self.component_health:
            self.component_health[check_name].consecutive_failures += 1
            self.component_health[check_name].error_count += 1

        self.metrics_collector.increment(f"health_check_{check_name}_failures_total")

    async def _check_system_resources(self) -> ComponentHealth:
        """Check system resource usage (CPU, memory, disk)."""
        try:
            # Get system metrics
            cpu_percent = psutil.cpu_percent(interval=0.5)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage("/")

            # Calculate status based on thresholds
            status = HealthStatus.HEALTHY
            issues = []

            if cpu_percent >= self.thresholds.cpu_critical:
                status = HealthStatus.CRITICAL
                issues.append(f"CPU usage critical: {cpu_percent:.1f}%")
            elif cpu_percent >= self.thresholds.cpu_warning:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"CPU usage high: {cpu_percent:.1f}%")

            if memory.percent >= self.thresholds.memory_critical:
                status = HealthStatus.CRITICAL
                issues.append(f"Memory usage critical: {memory.percent:.1f}%")
            elif memory.percent >= self.thresholds.memory_warning:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Memory usage high: {memory.percent:.1f}%")

            disk_percent = (disk.used / disk.total) * 100
            if disk_percent >= self.thresholds.disk_critical:
                status = HealthStatus.CRITICAL
                issues.append(f"Disk usage critical: {disk_percent:.1f}%")
            elif disk_percent >= self.thresholds.disk_warning:
                if status == HealthStatus.HEALTHY:
                    status = HealthStatus.WARNING
                issues.append(f"Disk usage high: {disk_percent:.1f}%")

            message = (
                "System resources healthy"
                if status == HealthStatus.HEALTHY
                else "; ".join(issues)
            )

            return ComponentHealth(
                component_name="system_resources",
                component_type=ComponentType.SYSTEM,
                status=status,
                last_check=datetime.now(),
                response_time=0.0,
                message=message,
                details={
                    "cpu_percent": cpu_percent,
                    "memory_percent": memory.percent,
                    "memory_available_gb": memory.available / (1024**3),
                    "memory_total_gb": memory.total / (1024**3),
                    "disk_percent": disk_percent,
                    "disk_free_gb": disk.free / (1024**3),
                    "disk_total_gb": disk.total / (1024**3),
                },
                metrics={
                    "cpu_usage": cpu_percent,
                    "memory_usage": memory.percent,
                    "disk_usage": disk_percent,
                },
            )

        except Exception as e:
            return ComponentHealth(
                component_name="system_resources",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"Failed to check system resources: {e}",
                details={"error": str(e)},
            )

    async def _check_database_connection(self) -> ComponentHealth:
        """Check database connection and performance."""
        try:
            from ..data.storage.database import get_database_manager

            start_time = time.time()
            db_manager = await get_database_manager()

            # Run database health check
            db_health = await db_manager.health_check()
            response_time = time.time() - start_time

            # Determine status
            if db_health.get("database_connected", False):
                if response_time > self.thresholds.response_time_critical:
                    status = HealthStatus.DEGRADED
                    message = f"Database responding slowly ({response_time:.2f}s)"
                elif response_time > self.thresholds.response_time_warning:
                    status = HealthStatus.WARNING
                    message = f"Database response time elevated ({response_time:.2f}s)"
                else:
                    status = HealthStatus.HEALTHY
                    message = "Database connection healthy"
            else:
                status = HealthStatus.CRITICAL
                message = "Database connection failed"

            return ComponentHealth(
                component_name="database_connection",
                component_type=ComponentType.DATABASE,
                status=status,
                last_check=datetime.now(),
                response_time=response_time,
                message=message,
                details=db_health,
                metrics={"response_time": response_time},
            )

        except Exception as e:
            return ComponentHealth(
                component_name="database_connection",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"Database health check failed: {e}",
                details={"error": str(e)},
            )

    async def _check_mqtt_broker(self) -> ComponentHealth:
        """Check MQTT broker connection and performance."""
        try:
            import paho.mqtt.client as mqtt

            mqtt_config = self.config.mqtt
            connected = False
            error_msg = None

            def on_connect(client, userdata, flags, rc):
                nonlocal connected
                connected = rc == 0
                if rc != 0:
                    nonlocal error_msg
                    error_msg = f"MQTT connection failed with code {rc}"

            # Test connection
            start_time = time.time()
            client = mqtt.Client()
            client.on_connect = on_connect

            if mqtt_config.username:
                client.username_pw_set(mqtt_config.username, mqtt_config.password)

            try:
                client.connect(mqtt_config.broker, mqtt_config.port, 10)
                client.loop_start()

                # Wait for connection
                for _ in range(50):  # 5 second timeout
                    if connected or error_msg:
                        break
                    await asyncio.sleep(0.1)

                client.disconnect()
                client.loop_stop()

            except Exception as e:
                error_msg = str(e)

            response_time = time.time() - start_time

            if connected:
                status = HealthStatus.HEALTHY
                message = f"MQTT broker connection healthy ({response_time:.2f}s)"
            else:
                status = HealthStatus.CRITICAL
                message = error_msg or "MQTT broker connection failed"

            return ComponentHealth(
                component_name="mqtt_broker",
                component_type=ComponentType.MQTT,
                status=status,
                last_check=datetime.now(),
                response_time=response_time,
                message=message,
                details={
                    "broker": mqtt_config.broker,
                    "port": mqtt_config.port,
                    "connected": connected,
                },
                metrics={"response_time": response_time},
            )

        except Exception as e:
            return ComponentHealth(
                component_name="mqtt_broker",
                component_type=ComponentType.MQTT,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"MQTT broker check failed: {e}",
                details={"error": str(e)},
            )

    async def _check_api_endpoints(self) -> ComponentHealth:
        """Check API endpoint health and performance."""
        try:
            api_config = self.config.api
            base_url = f"http://localhost:{api_config.port}"

            # Test endpoints
            endpoints_to_test = [
                ("/health", "Health endpoint"),
                ("/api/predictions", "Predictions API"),
                ("/api/status", "Status API"),
            ]

            total_response_time = 0.0
            successful_checks = 0
            issues = []

            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10)
            ) as session:
                for endpoint, description in endpoints_to_test:
                    try:
                        start_time = time.time()
                        async with session.get(f"{base_url}{endpoint}") as response:
                            response_time = time.time() - start_time
                            total_response_time += response_time

                            if response.status in [
                                200,
                                404,
                            ]:  # 404 is OK for some endpoints
                                successful_checks += 1
                            else:
                                issues.append(
                                    f"{description} returned {response.status}"
                                )

                    except asyncio.TimeoutError:
                        issues.append(f"{description} timed out")
                    except Exception as e:
                        issues.append(f"{description} failed: {str(e)[:50]}")

            avg_response_time = (
                total_response_time / len(endpoints_to_test)
                if endpoints_to_test
                else 0.0
            )
            success_rate = (
                successful_checks / len(endpoints_to_test) if endpoints_to_test else 0.0
            )

            # Determine status
            if success_rate >= 1.0:
                if avg_response_time > self.thresholds.response_time_critical:
                    status = HealthStatus.DEGRADED
                    message = f"API endpoints slow (avg {avg_response_time:.2f}s)"
                elif avg_response_time > self.thresholds.response_time_warning:
                    status = HealthStatus.WARNING
                    message = f"API endpoints elevated response time (avg {avg_response_time:.2f}s)"
                else:
                    status = HealthStatus.HEALTHY
                    message = (
                        f"All API endpoints healthy (avg {avg_response_time:.2f}s)"
                    )
            elif success_rate >= 0.5:
                status = HealthStatus.DEGRADED
                message = (
                    f"Some API endpoints failing ({success_rate*100:.0f}% success)"
                )
            else:
                status = HealthStatus.CRITICAL
                message = (
                    f"Most API endpoints failing ({success_rate*100:.0f}% success)"
                )

            if issues:
                message += f". Issues: {'; '.join(issues[:3])}"

            return ComponentHealth(
                component_name="api_endpoints",
                component_type=ComponentType.API,
                status=status,
                last_check=datetime.now(),
                response_time=avg_response_time,
                message=message,
                details={
                    "base_url": base_url,
                    "endpoints_tested": len(endpoints_to_test),
                    "successful_checks": successful_checks,
                    "success_rate": success_rate,
                    "issues": issues,
                },
                metrics={
                    "avg_response_time": avg_response_time,
                    "success_rate": success_rate,
                },
            )

        except Exception as e:
            return ComponentHealth(
                component_name="api_endpoints",
                component_type=ComponentType.API,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"API endpoint check failed: {e}",
                details={"error": str(e)},
            )

    async def _check_memory_usage(self) -> ComponentHealth:
        """Check application memory usage."""
        try:
            process = psutil.Process()
            process_memory = process.memory_info()
            system_memory = psutil.virtual_memory()

            # Calculate memory usage
            process_mb = process_memory.rss / (1024 * 1024)
            process_percent = (process_memory.rss / system_memory.total) * 100

            # Determine status
            if process_percent >= 50:  # Using > 50% of system memory
                status = HealthStatus.CRITICAL
                message = f"High memory usage: {process_mb:.1f} MB ({process_percent:.1f}% of system)"
            elif process_percent >= 25:  # Using > 25% of system memory
                status = HealthStatus.WARNING
                message = f"Elevated memory usage: {process_mb:.1f} MB ({process_percent:.1f}% of system)"
            else:
                status = HealthStatus.HEALTHY
                message = f"Memory usage normal: {process_mb:.1f} MB ({process_percent:.1f}% of system)"

            return ComponentHealth(
                component_name="memory_usage",
                component_type=ComponentType.APPLICATION,
                status=status,
                last_check=datetime.now(),
                response_time=0.0,
                message=message,
                details={
                    "process_memory_mb": process_mb,
                    "process_memory_percent": process_percent,
                    "system_total_gb": system_memory.total / (1024**3),
                    "system_available_gb": system_memory.available / (1024**3),
                },
                metrics={
                    "process_memory_mb": process_mb,
                    "process_memory_percent": process_percent,
                },
            )

        except Exception as e:
            return ComponentHealth(
                component_name="memory_usage",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"Memory usage check failed: {e}",
                details={"error": str(e)},
            )

    async def _check_disk_space(self) -> ComponentHealth:
        """Check disk space availability."""
        try:
            disk_usage = psutil.disk_usage("/")
            usage_percent = (disk_usage.used / disk_usage.total) * 100
            free_gb = disk_usage.free / (1024**3)

            # Determine status
            if usage_percent >= self.thresholds.disk_critical:
                status = HealthStatus.CRITICAL
                message = f"Critical disk space: {usage_percent:.1f}% used, {free_gb:.1f} GB free"
            elif usage_percent >= self.thresholds.disk_warning:
                status = HealthStatus.WARNING
                message = (
                    f"Low disk space: {usage_percent:.1f}% used, {free_gb:.1f} GB free"
                )
            else:
                status = HealthStatus.HEALTHY
                message = f"Disk space healthy: {usage_percent:.1f}% used, {free_gb:.1f} GB free"

            return ComponentHealth(
                component_name="disk_space",
                component_type=ComponentType.SYSTEM,
                status=status,
                last_check=datetime.now(),
                response_time=0.0,
                message=message,
                details={
                    "total_gb": disk_usage.total / (1024**3),
                    "used_gb": disk_usage.used / (1024**3),
                    "free_gb": free_gb,
                    "usage_percent": usage_percent,
                },
                metrics={"disk_usage_percent": usage_percent},
            )

        except Exception as e:
            return ComponentHealth(
                component_name="disk_space",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"Disk space check failed: {e}",
                details={"error": str(e)},
            )

    async def _check_network_connectivity(self) -> ComponentHealth:
        """Check network connectivity to external services."""
        try:
            # Test DNS resolution and HTTP connectivity
            test_urls = ["8.8.8.8", "google.com"]
            successful_tests = 0
            total_response_time = 0.0

            for target in test_urls:
                try:
                    start_time = time.time()
                    if target == "8.8.8.8":
                        # Ping test
                        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                        sock.settimeout(5)
                        result = sock.connect_ex((target, 53))
                        sock.close()
                        if result == 0:
                            successful_tests += 1
                    else:
                        # HTTP test
                        async with aiohttp.ClientSession(
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as session:
                            async with session.get(f"https://{target}") as response:
                                if response.status < 500:
                                    successful_tests += 1

                    total_response_time += time.time() - start_time

                except Exception:
                    pass  # Failed test

            avg_response_time = (
                total_response_time / len(test_urls) if test_urls else 0.0
            )
            success_rate = successful_tests / len(test_urls) if test_urls else 0.0

            # Determine status
            if success_rate >= 1.0:
                status = HealthStatus.HEALTHY
                message = f"Network connectivity healthy (avg {avg_response_time:.2f}s)"
            elif success_rate >= 0.5:
                status = HealthStatus.WARNING
                message = (
                    f"Partial network connectivity ({success_rate*100:.0f}% success)"
                )
            else:
                status = HealthStatus.CRITICAL
                message = (
                    f"Network connectivity issues ({success_rate*100:.0f}% success)"
                )

            return ComponentHealth(
                component_name="network_connectivity",
                component_type=ComponentType.NETWORK,
                status=status,
                last_check=datetime.now(),
                response_time=avg_response_time,
                message=message,
                details={
                    "tests_run": len(test_urls),
                    "successful_tests": successful_tests,
                    "success_rate": success_rate,
                },
                metrics={
                    "success_rate": success_rate,
                    "avg_response_time": avg_response_time,
                },
            )

        except Exception as e:
            return ComponentHealth(
                component_name="network_connectivity",
                component_type=ComponentType.NETWORK,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"Network connectivity check failed: {e}",
                details={"error": str(e)},
            )

    async def _check_application_metrics(self) -> ComponentHealth:
        """Check application-specific metrics and performance."""
        try:
            # Get application metrics from metrics collector
            current_time = datetime.now()
            performance_issues = []

            # Check recent performance metrics
            if hasattr(self.metrics_collector, "get_recent_metrics"):
                recent_metrics = self.metrics_collector.get_recent_metrics(minutes=5)

                # Check prediction latency
                if "prediction_latency" in recent_metrics:
                    avg_latency = statistics.mean(
                        recent_metrics["prediction_latency"][-10:]
                    )
                    if avg_latency > 2.0:
                        performance_issues.append(
                            f"High prediction latency: {avg_latency:.2f}s"
                        )

                # Check error rates
                if "error_count" in recent_metrics:
                    recent_errors = sum(recent_metrics["error_count"][-5:])
                    if recent_errors > 5:
                        performance_issues.append(
                            f"High error rate: {recent_errors} errors in 5 minutes"
                        )

            # Check if monitoring cycle time is acceptable
            if self.performance_metrics["monitoring_cycle_time"]:
                recent_cycles = self.performance_metrics["monitoring_cycle_time"][-10:]
                avg_cycle_time = statistics.mean(recent_cycles)
                if avg_cycle_time > 10.0:  # 10 seconds is too slow
                    performance_issues.append(
                        f"Slow monitoring cycles: {avg_cycle_time:.2f}s"
                    )

            # Determine status
            if not performance_issues:
                status = HealthStatus.HEALTHY
                message = "Application metrics healthy"
            elif len(performance_issues) <= 2:
                status = HealthStatus.WARNING
                message = f"Performance concerns: {'; '.join(performance_issues)}"
            else:
                status = HealthStatus.DEGRADED
                message = (
                    f"Multiple performance issues: {'; '.join(performance_issues[:3])}"
                )

            return ComponentHealth(
                component_name="application_metrics",
                component_type=ComponentType.APPLICATION,
                status=status,
                last_check=current_time,
                response_time=0.0,
                message=message,
                details={
                    "performance_issues_count": len(performance_issues),
                    "performance_issues": performance_issues,
                    "monitoring_active": self._monitoring_active,
                    "uptime_seconds": (current_time - self._start_time).total_seconds(),
                },
                metrics={
                    "performance_issues_count": len(performance_issues),
                    "uptime_seconds": (current_time - self._start_time).total_seconds(),
                },
            )

        except Exception as e:
            return ComponentHealth(
                component_name="application_metrics",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.CRITICAL,
                last_check=datetime.now(),
                response_time=0.0,
                message=f"Application metrics check failed: {e}",
                details={"error": str(e)},
            )

    def _calculate_system_health(self) -> SystemHealth:
        """Calculate overall system health status."""
        if not self.component_health:
            return SystemHealth(
                overall_status=HealthStatus.UNKNOWN,
                component_count=0,
                healthy_components=0,
                degraded_components=0,
                critical_components=0,
                last_updated=datetime.now(),
                uptime_seconds=0,
                system_load=0.0,
                total_memory_gb=0.0,
                available_memory_gb=0.0,
                cpu_usage=0.0,
                active_alerts=0,
                performance_score=0.0,
            )

        # Count components by status
        healthy_count = 0
        warning_count = 0
        degraded_count = 0
        critical_count = 0

        for health in self.component_health.values():
            if health.status == HealthStatus.HEALTHY:
                healthy_count += 1
            elif health.status == HealthStatus.WARNING:
                warning_count += 1
            elif health.status == HealthStatus.DEGRADED:
                degraded_count += 1
            elif health.status == HealthStatus.CRITICAL:
                critical_count += 1

        total_components = len(self.component_health)

        # Determine overall status
        if critical_count > 0:
            overall_status = HealthStatus.CRITICAL
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        elif warning_count > 0:
            overall_status = HealthStatus.WARNING
        else:
            overall_status = HealthStatus.HEALTHY

        # Get system metrics
        try:
            cpu_usage = psutil.cpu_percent(interval=None)
            memory = psutil.virtual_memory()
            system_load = (
                psutil.getloadavg()[0] if hasattr(psutil, "getloadavg") else 0.0
            )
        except Exception:
            cpu_usage = 0.0
            memory = psutil.virtual_memory()
            system_load = 0.0

        # Calculate performance score
        healthy_ratio = (
            (healthy_count + warning_count * 0.5) / total_components
            if total_components > 0
            else 0.0
        )
        resource_score = max(0.0, 1.0 - max(cpu_usage, memory.percent) / 100.0)
        performance_score = (healthy_ratio * 0.7 + resource_score * 0.3) * 100

        # Get active alerts count
        active_alerts = 0
        if hasattr(self.alert_manager, "get_active_alert_count"):
            active_alerts = self.alert_manager.get_active_alert_count()

        return SystemHealth(
            overall_status=overall_status,
            component_count=total_components,
            healthy_components=healthy_count
            + warning_count,  # Warning is still functional
            degraded_components=degraded_count,
            critical_components=critical_count,
            last_updated=datetime.now(),
            uptime_seconds=(datetime.now() - self._start_time).total_seconds(),
            system_load=system_load,
            total_memory_gb=memory.total / (1024**3),
            available_memory_gb=memory.available / (1024**3),
            cpu_usage=cpu_usage,
            active_alerts=active_alerts,
            performance_score=performance_score,
        )

    async def _process_incidents(self):
        """Process health incidents and trigger automated responses."""
        for name, health in self.component_health.items():
            # Check for incidents requiring attention
            if (
                health.needs_attention()
                and health.consecutive_failures >= self.alert_threshold
            ):
                await self._trigger_incident_response(health)

    async def _trigger_incident_response(self, health: ComponentHealth):
        """Trigger automated incident response for a component."""
        # Create alert
        alert_title = f"{health.component_name.title()} Health Alert"
        alert_message = f"Component {health.component_name} is {health.status.value}: {health.message}"

        try:
            await self.alert_manager.trigger_alert(
                rule_name=f"health_monitor_{health.component_name}",
                title=alert_title,
                message=alert_message,
                component=health.component_name,
                severity=(
                    "critical" if health.status == HealthStatus.CRITICAL else "warning"
                ),
                context={
                    "component_type": health.component_type.value,
                    "consecutive_failures": health.consecutive_failures,
                    "response_time": health.response_time,
                    "details": health.details,
                    "metrics": health.metrics,
                },
            )
        except Exception as e:
            self.logger.error(
                f"Failed to trigger alert for {health.component_name}: {e}"
            )

        # Run incident callbacks
        for callback in self.incident_callbacks:
            try:
                await callback(health)
            except Exception as e:
                self.logger.error(
                    f"Incident callback failed for {health.component_name}: {e}"
                )

        # Log incident
        self.logger.warning(
            f"Health incident triggered for {health.component_name}",
            extra={
                "component_name": health.component_name,
                "component_type": health.component_type.value,
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
                "message": health.message,
            },
        )

    def _update_monitoring_metrics(self, system_health: SystemHealth):
        """Update monitoring metrics."""
        try:
            # System health metrics
            self.metrics_collector.set_gauge(
                "system_health_score", system_health.health_score()
            )
            self.metrics_collector.set_gauge(
                "system_healthy_components", system_health.healthy_components
            )
            self.metrics_collector.set_gauge(
                "system_degraded_components", system_health.degraded_components
            )
            self.metrics_collector.set_gauge(
                "system_critical_components", system_health.critical_components
            )
            self.metrics_collector.set_gauge(
                "system_uptime_seconds", system_health.uptime_seconds
            )

            # Component health metrics
            for name, health in self.component_health.items():
                labels = {"component": name, "type": health.component_type.value}
                status_value = {
                    HealthStatus.HEALTHY: 1.0,
                    HealthStatus.WARNING: 0.8,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.CRITICAL: 0.0,
                    HealthStatus.UNKNOWN: -1.0,
                }.get(health.status, -1.0)

                self.metrics_collector.set_gauge_with_labels(
                    "component_health_status", status_value, labels
                )
                self.metrics_collector.set_gauge_with_labels(
                    "component_response_time", health.response_time, labels
                )
                self.metrics_collector.set_gauge_with_labels(
                    "component_consecutive_failures",
                    health.consecutive_failures,
                    labels,
                )

                # Component-specific metrics
                for metric_name, metric_value in health.metrics.items():
                    self.metrics_collector.set_gauge_with_labels(
                        f"component_{metric_name}",
                        metric_value,
                        {**labels, "metric": metric_name},
                    )

        except Exception as e:
            self.logger.error(f"Failed to update monitoring metrics: {e}")

    def _log_monitoring_summary(self, system_health: SystemHealth):
        """Log periodic monitoring summary."""
        self.logger.info(
            "Health monitoring summary",
            extra={
                "overall_status": system_health.overall_status.value,
                "health_score": system_health.health_score(),
                "component_count": system_health.component_count,
                "healthy_components": system_health.healthy_components,
                "degraded_components": system_health.degraded_components,
                "critical_components": system_health.critical_components,
                "uptime_hours": system_health.uptime_seconds / 3600,
                "cpu_usage": system_health.cpu_usage,
                "memory_usage_gb": system_health.total_memory_gb
                - system_health.available_memory_gb,
                "active_alerts": system_health.active_alerts,
                "performance_score": system_health.performance_score,
                "avg_monitoring_cycle_time": (
                    statistics.mean(
                        self.performance_metrics["monitoring_cycle_time"][-10:]
                    )
                    if self.performance_metrics["monitoring_cycle_time"]
                    else 0.0
                ),
            },
        )

    # Public API methods

    def get_system_health(self) -> SystemHealth:
        """Get current system health status."""
        return self._calculate_system_health()

    def get_component_health(
        self, component_name: Optional[str] = None
    ) -> Dict[str, ComponentHealth]:
        """Get health status for specific component or all components."""
        if component_name:
            health = self.component_health.get(component_name)
            return {component_name: health} if health is not None else {}
        return self.component_health.copy()

    def get_health_history(
        self, component_name: str, hours: int = 24
    ) -> List[Tuple[datetime, HealthStatus]]:
        """Get health history for a component."""
        if component_name not in self.health_history:
            return []

        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [
            (timestamp, status)
            for timestamp, status in self.health_history[component_name]
            if timestamp >= cutoff_time
        ]

    def is_monitoring_active(self) -> bool:
        """Check if health monitoring is active."""
        return self._monitoring_active

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """Get monitoring system statistics."""
        uptime = (datetime.now() - self._start_time).total_seconds()

        return {
            "monitoring_active": self._monitoring_active,
            "uptime_seconds": uptime,
            "uptime_hours": uptime / 3600,
            "registered_checks": len(self.health_checks),
            "component_count": len(self.component_health),
            "incident_callbacks": len(self.incident_callbacks),
            "check_interval": self.check_interval,
            "alert_threshold": self.alert_threshold,
            "avg_cycle_time": (
                statistics.mean(self.performance_metrics["monitoring_cycle_time"][-20:])
                if self.performance_metrics["monitoring_cycle_time"]
                else 0.0
            ),
            "total_cycles": len(self.performance_metrics["monitoring_cycle_time"]),
        }


# Global health monitor instance
_health_monitor = None


def get_health_monitor() -> HealthMonitor:
    """Get global health monitor instance."""
    global _health_monitor
    if _health_monitor is None:
        _health_monitor = HealthMonitor()
    return _health_monitor
