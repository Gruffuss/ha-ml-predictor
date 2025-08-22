"""
Comprehensive unit tests for health_monitor.py.
Tests health monitoring, system checks, component health tracking, and incident response.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.utils.health_monitor import (
    ComponentHealth,
    ComponentType,
    HealthMonitor,
    HealthStatus,
    HealthThresholds,
    SystemHealth,
    get_health_monitor,
)


class TestHealthStatus:
    """Test HealthStatus enum functionality."""

    def test_health_status_values(self):
        """Test all health status enum values."""
        assert HealthStatus.HEALTHY.value == "healthy"
        assert HealthStatus.WARNING.value == "warning"
        assert HealthStatus.DEGRADED.value == "degraded"
        assert HealthStatus.CRITICAL.value == "critical"
        assert HealthStatus.UNKNOWN.value == "unknown"


class TestComponentType:
    """Test ComponentType enum functionality."""

    def test_component_type_values(self):
        """Test all component type enum values."""
        assert ComponentType.DATABASE.value == "database"
        assert ComponentType.MQTT.value == "mqtt"
        assert ComponentType.API.value == "api"
        assert ComponentType.SYSTEM.value == "system"
        assert ComponentType.NETWORK.value == "network"
        assert ComponentType.APPLICATION.value == "application"
        assert ComponentType.EXTERNAL.value == "external"


class TestHealthThresholds:
    """Test HealthThresholds dataclass functionality."""

    def test_health_thresholds_creation(self):
        """Test creating HealthThresholds with custom values."""
        thresholds = HealthThresholds(
            cpu_warning=60.0,
            cpu_critical=80.0,
            memory_warning=75.0,
            memory_critical=90.0,
            disk_warning=85.0,
            disk_critical=95.0,
            response_time_warning=2.0,
            response_time_critical=10.0,
            error_rate_warning=0.1,
            error_rate_critical=0.2,
            uptime_warning=0.99,
            uptime_critical=0.95,
        )

        assert thresholds.cpu_warning == 60.0
        assert thresholds.cpu_critical == 80.0
        assert thresholds.memory_warning == 75.0
        assert thresholds.memory_critical == 90.0
        assert thresholds.disk_warning == 85.0
        assert thresholds.disk_critical == 95.0
        assert thresholds.response_time_warning == 2.0
        assert thresholds.response_time_critical == 10.0
        assert thresholds.error_rate_warning == 0.1
        assert thresholds.error_rate_critical == 0.2
        assert thresholds.uptime_warning == 0.99
        assert thresholds.uptime_critical == 0.95

    def test_health_thresholds_defaults(self):
        """Test HealthThresholds with default values."""
        thresholds = HealthThresholds()

        assert thresholds.cpu_warning == 70.0
        assert thresholds.cpu_critical == 85.0
        assert thresholds.memory_warning == 70.0
        assert thresholds.memory_critical == 85.0
        assert thresholds.disk_warning == 80.0
        assert thresholds.disk_critical == 90.0
        assert thresholds.response_time_warning == 1.0
        assert thresholds.response_time_critical == 5.0
        assert thresholds.error_rate_warning == 0.05
        assert thresholds.error_rate_critical == 0.10
        assert thresholds.uptime_warning == 0.95
        assert thresholds.uptime_critical == 0.90


class TestComponentHealth:
    """Test ComponentHealth dataclass functionality."""

    def test_component_health_creation(self):
        """Test creating ComponentHealth with all fields."""
        timestamp = datetime.now()
        details = {"connection_count": 5, "last_query": "SELECT 1"}
        metrics = {"response_time": 0.05, "query_count": 100}

        health = ComponentHealth(
            component_name="database",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.HEALTHY,
            last_check=timestamp,
            response_time=0.05,
            message="Database connection healthy",
            details=details,
            error_count=0,
            consecutive_failures=0,
            uptime_percentage=99.9,
            metrics=metrics,
        )

        assert health.component_name == "database"
        assert health.component_type == ComponentType.DATABASE
        assert health.status == HealthStatus.HEALTHY
        assert health.last_check == timestamp
        assert health.response_time == 0.05
        assert health.message == "Database connection healthy"
        assert health.details == details
        assert health.error_count == 0
        assert health.consecutive_failures == 0
        assert health.uptime_percentage == 99.9
        assert health.metrics == metrics

    def test_component_health_defaults(self):
        """Test ComponentHealth with default values."""
        health = ComponentHealth(
            component_name="test_component",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.UNKNOWN,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test component",
        )

        assert health.details == {}
        assert health.error_count == 0
        assert health.consecutive_failures == 0
        assert health.uptime_percentage == 100.0
        assert health.metrics == {}

    def test_is_healthy(self):
        """Test is_healthy method."""
        healthy_component = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test",
        )

        warning_component = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.WARNING,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test",
        )

        degraded_component = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.DEGRADED,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test",
        )

        assert healthy_component.is_healthy() is True
        assert warning_component.is_healthy() is True
        assert degraded_component.is_healthy() is False

    def test_needs_attention(self):
        """Test needs_attention method."""
        healthy_component = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test",
        )

        degraded_component = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.DEGRADED,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test",
        )

        critical_component = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time=0.0,
            message="Test",
        )

        assert healthy_component.needs_attention() is False
        assert degraded_component.needs_attention() is True
        assert critical_component.needs_attention() is True

    def test_to_dict(self):
        """Test to_dict method."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        health = ComponentHealth(
            component_name="test_component",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.WARNING,
            last_check=timestamp,
            response_time=1.5,
            message="High response time",
            details={"connection_pool": "80% utilized"},
            error_count=2,
            consecutive_failures=1,
            uptime_percentage=98.5,
            metrics={"latency": 1.5},
        )

        result = health.to_dict()

        expected = {
            "component_name": "test_component",
            "component_type": "database",
            "status": "warning",
            "last_check": "2024-01-15T14:30:00",
            "response_time": 1.5,
            "message": "High response time",
            "details": {"connection_pool": "80% utilized"},
            "error_count": 2,
            "consecutive_failures": 1,
            "uptime_percentage": 98.5,
            "metrics": {"latency": 1.5},
            "is_healthy": True,
            "needs_attention": False,
        }

        assert result == expected


class TestSystemHealth:
    """Test SystemHealth dataclass functionality."""

    def test_system_health_creation(self):
        """Test creating SystemHealth with all fields."""
        timestamp = datetime.now()

        health = SystemHealth(
            overall_status=HealthStatus.WARNING,
            component_count=5,
            healthy_components=3,
            degraded_components=1,
            critical_components=1,
            last_updated=timestamp,
            uptime_seconds=86400.0,
            system_load=0.5,
            total_memory_gb=16.0,
            available_memory_gb=8.0,
            cpu_usage=45.0,
            active_alerts=2,
            performance_score=75.0,
        )

        assert health.overall_status == HealthStatus.WARNING
        assert health.component_count == 5
        assert health.healthy_components == 3
        assert health.degraded_components == 1
        assert health.critical_components == 1
        assert health.last_updated == timestamp
        assert health.uptime_seconds == 86400.0
        assert health.system_load == 0.5
        assert health.total_memory_gb == 16.0
        assert health.available_memory_gb == 8.0
        assert health.cpu_usage == 45.0
        assert health.active_alerts == 2
        assert health.performance_score == 75.0

    def test_health_score_calculation(self):
        """Test health score calculation."""
        # Perfect health
        perfect_health = SystemHealth(
            overall_status=HealthStatus.HEALTHY,
            component_count=5,
            healthy_components=5,
            degraded_components=0,
            critical_components=0,
            last_updated=datetime.now(),
            uptime_seconds=86400.0,
            system_load=0.1,
            total_memory_gb=16.0,
            available_memory_gb=12.0,
            cpu_usage=10.0,
            active_alerts=0,
            performance_score=100.0,
        )

        assert perfect_health.health_score() == 100.0

        # Degraded health with alerts
        degraded_health = SystemHealth(
            overall_status=HealthStatus.DEGRADED,
            component_count=5,
            healthy_components=2,
            degraded_components=2,
            critical_components=1,
            last_updated=datetime.now(),
            uptime_seconds=86400.0,
            system_load=2.0,
            total_memory_gb=16.0,
            available_memory_gb=4.0,
            cpu_usage=80.0,
            active_alerts=3,
            performance_score=30.0,
        )

        score = degraded_health.health_score()
        assert 0.0 <= score <= 100.0
        assert score < 50.0  # Should be low due to degraded state and alerts

    def test_health_score_no_components(self):
        """Test health score with no components."""
        empty_health = SystemHealth(
            overall_status=HealthStatus.UNKNOWN,
            component_count=0,
            healthy_components=0,
            degraded_components=0,
            critical_components=0,
            last_updated=datetime.now(),
            uptime_seconds=0.0,
            system_load=0.0,
            total_memory_gb=0.0,
            available_memory_gb=0.0,
            cpu_usage=0.0,
            active_alerts=0,
            performance_score=0.0,
        )

        assert empty_health.health_score() == 0.0

    def test_to_dict(self):
        """Test to_dict method."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        health = SystemHealth(
            overall_status=HealthStatus.WARNING,
            component_count=3,
            healthy_components=2,
            degraded_components=1,
            critical_components=0,
            last_updated=timestamp,
            uptime_seconds=3600.0,
            system_load=1.0,
            total_memory_gb=8.0,
            available_memory_gb=4.0,
            cpu_usage=50.0,
            active_alerts=1,
            performance_score=80.0,
        )

        result = health.to_dict()

        assert result["overall_status"] == "warning"
        assert result["component_count"] == 3
        assert result["healthy_components"] == 2
        assert result["degraded_components"] == 1
        assert result["critical_components"] == 0
        assert result["last_updated"] == "2024-01-15T14:30:00"
        assert result["uptime_seconds"] == 3600.0
        assert result["system_load"] == 1.0
        assert result["total_memory_gb"] == 8.0
        assert result["available_memory_gb"] == 4.0
        assert result["cpu_usage"] == 50.0
        assert result["active_alerts"] == 1
        assert result["performance_score"] == 80.0
        assert "health_score" in result


class TestHealthMonitor:
    """Test HealthMonitor functionality."""

    @pytest.fixture
    def health_monitor(self):
        """Create HealthMonitor instance."""
        with patch("src.utils.health_monitor.get_logger"), patch(
            "src.utils.health_monitor.get_performance_logger"
        ), patch("src.utils.health_monitor.get_error_tracker"), patch(
            "src.utils.health_monitor.get_metrics_collector"
        ), patch(
            "src.utils.health_monitor.get_alert_manager"
        ), patch(
            "src.utils.health_monitor.get_config"
        ):
            return HealthMonitor()

    def test_health_monitor_initialization(self, health_monitor):
        """Test HealthMonitor initialization."""
        assert health_monitor.check_interval == 30
        assert health_monitor.alert_threshold == 3
        assert isinstance(health_monitor.thresholds, HealthThresholds)
        assert isinstance(health_monitor.component_health, dict)
        assert isinstance(health_monitor.health_history, defaultdict)
        assert health_monitor._monitoring_active is False
        assert health_monitor._monitoring_task is None

    def test_register_health_check(self, health_monitor):
        """Test registering custom health check."""

        def custom_check():
            return ComponentHealth(
                component_name="custom",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=0.1,
                message="Custom check passed",
            )

        health_monitor.register_health_check("custom_check", custom_check)

        assert "custom_check" in health_monitor.health_checks

    def test_add_incident_callback(self, health_monitor):
        """Test adding incident callback."""

        def incident_handler(health):
            pass

        health_monitor.add_incident_callback(incident_handler)

        assert incident_handler in health_monitor.incident_callbacks

    @pytest.mark.asyncio
    async def test_start_monitoring(self, health_monitor):
        """Test starting health monitoring."""
        with patch("asyncio.create_task") as mock_create_task:
            await health_monitor.start_monitoring()

            assert health_monitor._monitoring_active is True
            assert health_monitor._start_time is not None
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_monitoring_already_active(self, health_monitor):
        """Test starting monitoring when already active."""
        health_monitor._monitoring_active = True

        with patch("asyncio.create_task") as mock_create_task:
            await health_monitor.start_monitoring()

            # Should not create new task
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, health_monitor):
        """Test stopping health monitoring."""
        # Setup active monitoring
        health_monitor._monitoring_active = True
        mock_task = Mock()
        mock_task.cancel = Mock()
        health_monitor._monitoring_task = mock_task

        await health_monitor.stop_monitoring()

        assert health_monitor._monitoring_active is False
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_health_checks(self, health_monitor):
        """Test running all health checks."""

        # Setup mock health checks
        def mock_check1():
            return ComponentHealth(
                component_name="component1",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=0.1,
                message="Check 1 passed",
            )

        def mock_check2():
            return ComponentHealth(
                component_name="component2",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.WARNING,
                last_check=datetime.now(),
                response_time=0.5,
                message="Check 2 has warnings",
            )

        health_monitor.health_checks = {"check1": mock_check1, "check2": mock_check2}

        await health_monitor._run_health_checks()

        assert "check1" in health_monitor.component_health
        assert "check2" in health_monitor.component_health
        assert health_monitor.component_health["check1"].status == HealthStatus.HEALTHY
        assert health_monitor.component_health["check2"].status == HealthStatus.WARNING

    @pytest.mark.asyncio
    async def test_run_single_health_check_success(self, health_monitor):
        """Test running single health check successfully."""

        def successful_check():
            return ComponentHealth(
                component_name="test_component",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=0.1,
                message="Test passed",
            )

        result = await health_monitor._run_single_health_check(
            "test_check", successful_check
        )

        assert result.component_name == "test_component"
        assert result.status == HealthStatus.HEALTHY
        assert "test_check" in health_monitor.component_health

    @pytest.mark.asyncio
    async def test_run_single_health_check_timeout(self, health_monitor):
        """Test handling health check timeout."""

        async def slow_check():
            await asyncio.sleep(35)  # Longer than 30s timeout
            return ComponentHealth(
                component_name="slow_component",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=35.0,
                message="Slow check",
            )

        result = await health_monitor._run_single_health_check("slow_check", slow_check)

        assert result.status == HealthStatus.CRITICAL
        assert "timed out" in result.message
        assert result.component_name == "slow_check"

    @pytest.mark.asyncio
    async def test_run_single_health_check_exception(self, health_monitor):
        """Test handling health check exception."""

        def failing_check():
            raise Exception("Health check failed")

        result = await health_monitor._run_single_health_check(
            "failing_check", failing_check
        )

        assert result.status == HealthStatus.CRITICAL
        assert "Health check failing_check failed" in result.message
        assert result.component_name == "failing_check"

    @pytest.mark.asyncio
    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    async def test_check_system_resources_healthy(
        self, mock_disk, mock_memory, mock_cpu, health_monitor
    ):
        """Test system resources check with healthy values."""
        mock_cpu.return_value = 30.0
        mock_memory.return_value.percent = 40.0
        mock_memory.return_value.available = 8 * 1024**3
        mock_memory.return_value.total = 16 * 1024**3

        mock_disk_usage = Mock()
        mock_disk_usage.total = 1024**4  # 1TB
        mock_disk_usage.used = 512**3  # 128GB
        mock_disk.return_value = mock_disk_usage

        result = await health_monitor._check_system_resources()

        assert result.status == HealthStatus.HEALTHY
        assert "Resource usage normal" in result.message
        assert result.component_type == ComponentType.SYSTEM

    @pytest.mark.asyncio
    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.disk_usage")
    async def test_check_system_resources_critical(
        self, mock_disk, mock_memory, mock_cpu, health_monitor
    ):
        """Test system resources check with critical values."""
        mock_cpu.return_value = 95.0  # Critical
        mock_memory.return_value.percent = 95.0  # Critical
        mock_memory.return_value.available = 1 * 1024**3
        mock_memory.return_value.total = 16 * 1024**3

        mock_disk_usage = Mock()
        mock_disk_usage.total = 1024**4
        mock_disk_usage.used = 950 * 1024**3  # 95% used - critical
        mock_disk.return_value = mock_disk_usage

        result = await health_monitor._check_system_resources()

        assert result.status == HealthStatus.CRITICAL
        assert (
            "CPU usage critical" in result.message
            or "Memory usage critical" in result.message
            or "Disk usage critical" in result.message
        )

    @pytest.mark.asyncio
    @patch("psutil.cpu_percent")
    async def test_check_system_resources_exception(self, mock_cpu, health_monitor):
        """Test system resources check with exception."""
        mock_cpu.side_effect = Exception("PSUtil error")

        result = await health_monitor._check_system_resources()

        assert result.status == HealthStatus.CRITICAL
        assert "Failed to check system resources" in result.message

    @pytest.mark.asyncio
    async def test_check_database_connection_success(self, health_monitor):
        """Test successful database connection check."""
        with patch("src.utils.health_monitor.get_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = AsyncMock(
                return_value={"database_connected": True}
            )
            mock_get_db.return_value = mock_db_manager

            result = await health_monitor._check_database_connection()

            assert result.status == HealthStatus.HEALTHY
            assert "Database connection healthy" in result.message
            assert result.component_type == ComponentType.DATABASE

    @pytest.mark.asyncio
    async def test_check_database_connection_failure(self, health_monitor):
        """Test failed database connection check."""
        with patch("src.utils.health_monitor.get_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = AsyncMock(
                return_value={"database_connected": False}
            )
            mock_get_db.return_value = mock_db_manager

            result = await health_monitor._check_database_connection()

            assert result.status == HealthStatus.CRITICAL
            assert "Database connection failed" in result.message

    @pytest.mark.asyncio
    async def test_check_database_connection_slow(self, health_monitor):
        """Test slow database connection check."""
        with patch(
            "src.utils.health_monitor.get_database_manager"
        ) as mock_get_db, patch("time.time") as mock_time:

            mock_time.side_effect = [0, 3.0]  # 3 second response time

            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = AsyncMock(
                return_value={"database_connected": True}
            )
            mock_get_db.return_value = mock_db_manager

            result = await health_monitor._check_database_connection()

            assert result.status == HealthStatus.DEGRADED
            assert "responding slowly" in result.message

    @pytest.mark.asyncio
    @patch("paho.mqtt.client.Client")
    async def test_check_mqtt_broker_success(self, mock_client_class, health_monitor):
        """Test successful MQTT broker check."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful connection
        def mock_connect(broker, port, keepalive):
            mock_client.on_connect(mock_client, None, None, 0)  # Success code

        mock_client.connect = mock_connect

        result = await health_monitor._check_mqtt_broker()

        assert result.status == HealthStatus.HEALTHY
        assert "MQTT broker connection healthy" in result.message
        assert result.component_type == ComponentType.MQTT

    @pytest.mark.asyncio
    @patch("paho.mqtt.client.Client")
    async def test_check_mqtt_broker_failure(self, mock_client_class, health_monitor):
        """Test failed MQTT broker check."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock failed connection
        def mock_connect(broker, port, keepalive):
            mock_client.on_connect(mock_client, None, None, 4)  # Auth failure code

        mock_client.connect = mock_connect

        result = await health_monitor._check_mqtt_broker()

        assert result.status == HealthStatus.CRITICAL
        assert "MQTT broker connection failed" in result.message

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_check_api_endpoints_success(self, mock_session, health_monitor):
        """Test successful API endpoints check."""
        mock_response = Mock()
        mock_response.status = 200

        mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
            mock_response
        )

        result = await health_monitor._check_api_endpoints()

        assert result.status == HealthStatus.HEALTHY
        assert "All API endpoints healthy" in result.message
        assert result.component_type == ComponentType.API

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_check_api_endpoints_partial_failure(
        self, mock_session, health_monitor
    ):
        """Test API endpoints check with partial failures."""
        # Mock mixed responses - some success, some failure
        responses = [Mock(status=200), Mock(status=500), Mock(status=200)]

        async def mock_get(url):
            response = responses.pop(0) if responses else Mock(status=200)
            mock_context = Mock()
            mock_context.__aenter__ = AsyncMock(return_value=response)
            return mock_context

        mock_session.return_value.__aenter__.return_value.get = mock_get

        result = await health_monitor._check_api_endpoints()

        assert result.status in [HealthStatus.DEGRADED, HealthStatus.CRITICAL]
        assert "endpoints failing" in result.message

    @pytest.mark.asyncio
    @patch("psutil.Process")
    async def test_check_memory_usage_normal(self, mock_process, health_monitor):
        """Test memory usage check with normal values."""
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 1 * 1024**3  # 1GB
        mock_process.return_value = mock_process_instance

        with patch("psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value.total = 16 * 1024**3  # 16GB total

            result = await health_monitor._check_memory_usage()

            assert result.status == HealthStatus.HEALTHY
            assert "Memory usage normal" in result.message
            assert result.component_type == ComponentType.APPLICATION

    @pytest.mark.asyncio
    @patch("psutil.Process")
    async def test_check_memory_usage_high(self, mock_process, health_monitor):
        """Test memory usage check with high values."""
        mock_process_instance = Mock()
        mock_process_instance.memory_info.return_value.rss = 10 * 1024**3  # 10GB
        mock_process.return_value = mock_process_instance

        with patch("psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value.total = (
                16 * 1024**3
            )  # 16GB total (62.5% usage)

            result = await health_monitor._check_memory_usage()

            assert result.status == HealthStatus.WARNING
            assert "High memory usage" in result.message

    @pytest.mark.asyncio
    @patch("psutil.disk_usage")
    async def test_check_disk_space_healthy(self, mock_disk_usage, health_monitor):
        """Test disk space check with healthy values."""
        mock_usage = Mock()
        mock_usage.total = 1024**4  # 1TB
        mock_usage.used = 500 * 1024**3  # 500GB (50% used)
        mock_usage.free = 524 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = await health_monitor._check_disk_space()

        assert result.status == HealthStatus.HEALTHY
        assert "Disk space healthy" in result.message
        assert result.component_type == ComponentType.SYSTEM

    @pytest.mark.asyncio
    @patch("psutil.disk_usage")
    async def test_check_disk_space_critical(self, mock_disk_usage, health_monitor):
        """Test disk space check with critical values."""
        mock_usage = Mock()
        mock_usage.total = 1024**4  # 1TB
        mock_usage.used = 950 * 1024**3  # 950GB (95% used)
        mock_usage.free = 74 * 1024**3
        mock_disk_usage.return_value = mock_usage

        result = await health_monitor._check_disk_space()

        assert result.status == HealthStatus.CRITICAL
        assert "Critical disk space" in result.message

    @pytest.mark.asyncio
    async def test_check_network_connectivity_success(self, health_monitor):
        """Test successful network connectivity check."""
        with patch("socket.socket") as mock_socket, patch(
            "aiohttp.ClientSession"
        ) as mock_session:

            # Mock successful socket connection
            mock_sock = Mock()
            mock_sock.connect_ex.return_value = 0  # Success
            mock_socket.return_value = mock_sock

            # Mock successful HTTP request
            mock_response = Mock()
            mock_response.status = 200
            mock_session.return_value.__aenter__.return_value.get.return_value.__aenter__.return_value = (
                mock_response
            )

            result = await health_monitor._check_network_connectivity()

            assert result.status == HealthStatus.HEALTHY
            assert "Network connectivity healthy" in result.message
            assert result.component_type == ComponentType.NETWORK

    @pytest.mark.asyncio
    async def test_check_application_metrics_healthy(self, health_monitor):
        """Test application metrics check with healthy values."""
        # Mock healthy metrics
        health_monitor.performance_metrics = {
            "monitoring_cycle_time": [0.5, 0.6, 0.4, 0.5, 0.7]  # Good cycle times
        }

        result = await health_monitor._check_application_metrics()

        assert result.status == HealthStatus.HEALTHY
        assert "Application metrics healthy" in result.message
        assert result.component_type == ComponentType.APPLICATION

    def test_calculate_system_health_no_components(self, health_monitor):
        """Test system health calculation with no components."""
        result = health_monitor._calculate_system_health()

        assert result.overall_status == HealthStatus.UNKNOWN
        assert result.component_count == 0
        assert result.healthy_components == 0

    def test_calculate_system_health_mixed_status(self, health_monitor):
        """Test system health calculation with mixed component status."""
        # Setup mixed component health
        health_monitor.component_health = {
            "component1": ComponentHealth(
                component_name="component1",
                component_type=ComponentType.APPLICATION,
                status=HealthStatus.HEALTHY,
                last_check=datetime.now(),
                response_time=0.1,
                message="Healthy",
            ),
            "component2": ComponentHealth(
                component_name="component2",
                component_type=ComponentType.DATABASE,
                status=HealthStatus.WARNING,
                last_check=datetime.now(),
                response_time=0.5,
                message="Warning",
            ),
            "component3": ComponentHealth(
                component_name="component3",
                component_type=ComponentType.SYSTEM,
                status=HealthStatus.DEGRADED,
                last_check=datetime.now(),
                response_time=2.0,
                message="Degraded",
            ),
        }

        with patch("psutil.cpu_percent", return_value=50.0), patch(
            "psutil.virtual_memory"
        ) as mock_memory:

            mock_memory.return_value.total = 16 * 1024**3
            mock_memory.return_value.available = 8 * 1024**3
            mock_memory.return_value.percent = 50.0

            result = health_monitor._calculate_system_health()

            assert result.overall_status == HealthStatus.DEGRADED  # Worst status
            assert result.component_count == 3
            assert result.healthy_components == 2  # Healthy + Warning
            assert result.degraded_components == 1
            assert result.critical_components == 0

    @pytest.mark.asyncio
    async def test_process_incidents(self, health_monitor):
        """Test processing health incidents."""
        # Setup component needing attention
        problematic_component = ComponentHealth(
            component_name="failing_component",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time=10.0,
            message="Database connection failed",
            consecutive_failures=5,  # Above threshold
        )

        health_monitor.component_health["failing_component"] = problematic_component

        with patch.object(health_monitor, "_trigger_incident_response") as mock_trigger:
            await health_monitor._process_incidents()
            mock_trigger.assert_called_once_with(problematic_component)

    def test_get_system_health(self, health_monitor):
        """Test getting current system health."""
        result = health_monitor.get_system_health()

        assert isinstance(result, SystemHealth)
        assert result.component_count >= 0

    def test_get_component_health_all(self, health_monitor):
        """Test getting all component health."""
        # Setup some component health
        test_health = ComponentHealth(
            component_name="test_component",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time=0.1,
            message="Test",
        )
        health_monitor.component_health["test_component"] = test_health

        result = health_monitor.get_component_health()

        assert "test_component" in result
        assert result["test_component"] == test_health

    def test_get_component_health_specific(self, health_monitor):
        """Test getting specific component health."""
        test_health = ComponentHealth(
            component_name="specific_component",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.WARNING,
            last_check=datetime.now(),
            response_time=1.0,
            message="Slow response",
        )
        health_monitor.component_health["specific_component"] = test_health

        result = health_monitor.get_component_health("specific_component")

        assert "specific_component" in result
        assert result["specific_component"] == test_health

    def test_get_component_health_nonexistent(self, health_monitor):
        """Test getting health for non-existent component."""
        result = health_monitor.get_component_health("nonexistent")
        assert result == {}

    def test_get_health_history(self, health_monitor):
        """Test getting health history for component."""
        # Setup health history
        component_name = "test_component"
        now = datetime.now()

        health_monitor.health_history[component_name].extend(
            [
                (now - timedelta(hours=2), HealthStatus.HEALTHY),
                (now - timedelta(hours=1), HealthStatus.WARNING),
                (now - timedelta(minutes=30), HealthStatus.HEALTHY),
            ]
        )

        result = health_monitor.get_health_history(component_name, hours=3)

        assert len(result) == 3
        assert all(isinstance(entry[0], datetime) for entry in result)
        assert all(isinstance(entry[1], HealthStatus) for entry in result)

    def test_get_health_history_filtered_by_time(self, health_monitor):
        """Test getting health history filtered by time window."""
        component_name = "test_component"
        now = datetime.now()

        health_monitor.health_history[component_name].extend(
            [
                (now - timedelta(hours=25), HealthStatus.HEALTHY),  # Outside window
                (now - timedelta(hours=2), HealthStatus.WARNING),  # Inside window
                (now - timedelta(minutes=30), HealthStatus.HEALTHY),  # Inside window
            ]
        )

        result = health_monitor.get_health_history(component_name, hours=24)

        assert len(result) == 2  # Only entries within 24 hours

    def test_is_monitoring_active(self, health_monitor):
        """Test checking if monitoring is active."""
        assert health_monitor.is_monitoring_active() is False

        health_monitor._monitoring_active = True
        assert health_monitor.is_monitoring_active() is True

    def test_get_monitoring_stats(self, health_monitor):
        """Test getting monitoring statistics."""
        health_monitor._monitoring_active = True
        health_monitor.performance_metrics["monitoring_cycle_time"] = [0.5, 0.6, 0.7]

        stats = health_monitor.get_monitoring_stats()

        assert stats["monitoring_active"] is True
        assert stats["registered_checks"] == len(health_monitor.health_checks)
        assert stats["component_count"] == len(health_monitor.component_health)
        assert stats["check_interval"] == health_monitor.check_interval
        assert stats["alert_threshold"] == health_monitor.alert_threshold
        assert "uptime_seconds" in stats
        assert "avg_cycle_time" in stats


class TestGlobalHealthMonitor:
    """Test global health monitor functionality."""

    def test_get_health_monitor_singleton(self):
        """Test that get_health_monitor returns singleton instance."""
        with patch("src.utils.health_monitor.get_logger"), patch(
            "src.utils.health_monitor.get_performance_logger"
        ), patch("src.utils.health_monitor.get_error_tracker"), patch(
            "src.utils.health_monitor.get_metrics_collector"
        ), patch(
            "src.utils.health_monitor.get_alert_manager"
        ), patch(
            "src.utils.health_monitor.get_config"
        ):

            monitor1 = get_health_monitor()
            monitor2 = get_health_monitor()

            assert monitor1 is monitor2
            assert isinstance(monitor1, HealthMonitor)

    @patch("src.utils.health_monitor._health_monitor", None)
    def test_get_health_monitor_creates_new_instance(self):
        """Test that get_health_monitor creates new instance when none exists."""
        with patch("src.utils.health_monitor.get_logger"), patch(
            "src.utils.health_monitor.get_performance_logger"
        ), patch("src.utils.health_monitor.get_error_tracker"), patch(
            "src.utils.health_monitor.get_metrics_collector"
        ), patch(
            "src.utils.health_monitor.get_alert_manager"
        ), patch(
            "src.utils.health_monitor.get_config"
        ):

            # Reset global instance
            import src.utils.health_monitor

            src.utils.health_monitor._health_monitor = None

            monitor = get_health_monitor()
            assert isinstance(monitor, HealthMonitor)


if __name__ == "__main__":
    pytest.main([__file__])
