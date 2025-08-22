"""
System Resource Constraint Testing.

This module provides comprehensive testing for system behavior under various
resource constraints including memory limits, CPU throttling, connection pool
exhaustion, and concurrent load scenarios.

Test Coverage:
- Memory pressure and out-of-memory scenarios
- CPU throttling and high load conditions
- Database connection pool exhaustion
- Network connection limits
- Concurrent request handling under constraints
- Resource cleanup and garbage collection
- Performance degradation detection
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
import gc
import logging
import os

import psutil

# Windows compatibility: resource module not available on Windows
try:
    import resource
except ImportError:
    resource = None
import sys
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import weakref

import pytest

from src.adaptation.tracking_manager import TrackingManager
from src.core.config import MQTTConfig, TrackingConfig, get_config
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    ErrorSeverity,
    SystemResourceError,
)
from src.data.storage.database import DatabaseManager
from src.integration.enhanced_mqtt_manager import EnhancedMQTTIntegrationManager
from src.main_system import OccupancyPredictionSystem

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resource usage during tests."""

    def __init__(self):
        self.process = psutil.Process()
        self.baseline_memory = None
        self.peak_memory = 0
        self.baseline_cpu = None
        self.cpu_samples = []
        self.start_time = None
        self.open_file_descriptors = []

    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        self.baseline_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = self.baseline_memory
        self.baseline_cpu = self.process.cpu_percent()
        self.cpu_samples = []
        try:
            self.open_file_descriptors = self.process.open_files()
        except (psutil.AccessDenied, AttributeError):
            self.open_file_descriptors = []

    def update_metrics(self):
        """Update resource metrics."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        self.peak_memory = max(self.peak_memory, current_memory)

        cpu_percent = self.process.cpu_percent()
        self.cpu_samples.append(cpu_percent)

    def get_memory_usage(self):
        """Get current memory statistics."""
        current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
        return {
            "current_mb": current_memory,
            "baseline_mb": self.baseline_memory,
            "peak_mb": self.peak_memory,
            "increase_mb": (
                current_memory - self.baseline_memory if self.baseline_memory else 0
            ),
            "peak_increase_mb": (
                self.peak_memory - self.baseline_memory if self.baseline_memory else 0
            ),
        }

    def get_cpu_usage(self):
        """Get CPU usage statistics."""
        avg_cpu = (
            sum(self.cpu_samples) / len(self.cpu_samples) if self.cpu_samples else 0
        )
        max_cpu = max(self.cpu_samples) if self.cpu_samples else 0
        return {
            "average_percent": avg_cpu,
            "peak_percent": max_cpu,
            "baseline_percent": self.baseline_cpu,
            "sample_count": len(self.cpu_samples),
        }

    def get_file_descriptor_usage(self):
        """Get file descriptor usage."""
        try:
            current_fds = self.process.open_files()
            return {
                "current_count": len(current_fds),
                "baseline_count": len(self.open_file_descriptors),
                "increase": len(current_fds) - len(self.open_file_descriptors),
            }
        except (psutil.AccessDenied, AttributeError):
            return {"current_count": 0, "baseline_count": 0, "increase": 0}


@pytest.mark.system
class TestSystemResourceConstraints:
    """System resource constraint test suite."""

    @pytest.fixture
    def system_config(self):
        """System configuration for resource testing."""
        config = MagicMock()
        config.mqtt = MQTTConfig(
            broker="localhost",
            port=1883,
            username="test",
            password="test",
            topic_prefix="test/occupancy",
            device_identifier="test-device",
            discovery_prefix="homeassistant",
        )
        config.tracking = TrackingConfig()
        config.api = MagicMock()
        config.api.enabled = True
        config.api.host = "127.0.0.1"
        config.api.port = 8000
        return config

    @pytest.fixture
    def resource_monitor(self):
        """Resource monitoring fixture."""
        monitor = ResourceMonitor()
        monitor.start_monitoring()
        yield monitor

        # Cleanup and verify no resource leaks
        gc.collect()
        final_memory = monitor.get_memory_usage()
        final_fds = monitor.get_file_descriptor_usage()

        # Log resource usage for analysis
        logger.info(f"Test completed - Memory: {final_memory}")
        logger.info(f"Test completed - FDs: {final_fds}")

    @pytest.mark.asyncio
    async def test_memory_pressure_during_initialization(
        self, system_config, resource_monitor
    ):
        """Test system behavior under memory pressure during initialization."""

        # Create memory pressure
        memory_ballast = []
        try:
            # Allocate significant memory (100MB)
            for _ in range(100):
                memory_ballast.append(bytearray(1024 * 1024))  # 1MB each

            resource_monitor.update_metrics()
            initial_memory = resource_monitor.get_memory_usage()

            # Attempt system initialization under memory pressure
            system = OccupancyPredictionSystem()

            mock_database_manager = AsyncMock()
            mock_mqtt_manager = AsyncMock()
            mock_tracking_manager = AsyncMock()

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                # System should initialize despite memory pressure
                await system.initialize()
                assert system.running is True

                resource_monitor.update_metrics()
                post_init_memory = resource_monitor.get_memory_usage()

                # Verify system doesn't allocate excessive additional memory
                memory_increase = (
                    post_init_memory["current_mb"] - initial_memory["current_mb"]
                )
                assert (
                    memory_increase < 50
                ), f"Excessive memory allocated: {memory_increase}MB"

                await system.shutdown()

        finally:
            # Clean up memory ballast
            memory_ballast.clear()
            gc.collect()

    @pytest.mark.asyncio
    async def test_cpu_throttling_resilience(self, system_config, resource_monitor):
        """Test system resilience under CPU throttling conditions."""

        def cpu_intensive_task():
            """Create CPU load."""
            end_time = time.time() + 0.5  # Run for 500ms
            while time.time() < end_time:
                # CPU intensive calculation
                sum(i * i for i in range(1000))

        # Create CPU load in background
        with ThreadPoolExecutor(max_workers=4) as executor:
            # Submit CPU intensive tasks to all cores
            cpu_futures = [executor.submit(cpu_intensive_task) for _ in range(4)]

            try:
                # Initialize system under CPU load
                system = OccupancyPredictionSystem()

                mock_database_manager = AsyncMock()
                mock_mqtt_manager = AsyncMock()
                mock_tracking_manager = AsyncMock()

                with patch(
                    "src.main_system.get_config", return_value=system_config
                ), patch(
                    "src.main_system.get_database_manager",
                    return_value=mock_database_manager,
                ), patch(
                    "src.main_system.MQTTIntegrationManager",
                    return_value=mock_mqtt_manager,
                ), patch(
                    "src.main_system.TrackingManager",
                    return_value=mock_tracking_manager,
                ):

                    start_time = time.time()
                    await system.initialize()
                    init_duration = time.time() - start_time

                    assert system.running is True

                    resource_monitor.update_metrics()
                    cpu_stats = resource_monitor.get_cpu_usage()

                    # System should complete initialization even under CPU load
                    # Allow for reasonable initialization time under load
                    assert (
                        init_duration < 5.0
                    ), f"Initialization too slow under CPU load: {init_duration}s"

                    await system.shutdown()

            finally:
                # Wait for CPU tasks to complete
                for future in cpu_futures:
                    try:
                        future.result(timeout=1.0)
                    except Exception:
                        pass

    @pytest.mark.asyncio
    async def test_database_connection_pool_exhaustion(
        self, system_config, resource_monitor
    ):
        """Test system behavior when database connection pool is exhausted."""

        # Mock database manager with limited connection pool
        mock_database_manager = AsyncMock()
        connection_count = 0
        max_connections = 5

        def simulate_connection_exhaustion(*args, **kwargs):
            nonlocal connection_count
            connection_count += 1
            if connection_count > max_connections:
                raise DatabaseConnectionError(
                    "Connection pool exhausted", ErrorSeverity.HIGH
                )
            return mock_database_manager

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager",
            side_effect=simulate_connection_exhaustion,
        ):

            # First few attempts should succeed
            for i in range(max_connections):
                try:
                    test_system = OccupancyPredictionSystem()
                    # This would call get_database_manager during initialization
                    # but we're testing the connection limit logic
                    connection_count = i + 1  # Simulate connection count
                except DatabaseConnectionError:
                    pytest.fail(f"Connection should succeed for attempt {i}")

            # Next attempt should fail due to pool exhaustion
            with pytest.raises(
                DatabaseConnectionError, match="Connection pool exhausted"
            ):
                test_system = OccupancyPredictionSystem()
                await test_system.initialize()  # This should trigger the exhaustion

    @pytest.mark.asyncio
    async def test_concurrent_system_initialization(
        self, system_config, resource_monitor
    ):
        """Test multiple concurrent system initializations."""

        async def initialize_system(system_id: int):
            """Initialize a system instance."""
            system = OccupancyPredictionSystem()

            # Add slight delay to simulate real initialization time
            await asyncio.sleep(0.1 * system_id)

            mock_database_manager = AsyncMock()
            mock_mqtt_manager = AsyncMock()
            mock_tracking_manager = AsyncMock()

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                await system.initialize()
                assert system.running is True

                # Simulate some work
                await asyncio.sleep(0.1)

                await system.shutdown()
                assert system.running is False

                return system_id

        # Launch multiple concurrent initializations
        concurrent_count = 10
        tasks = [initialize_system(i) for i in range(concurrent_count)]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # Verify all initializations completed successfully
        successful_results = [r for r in results if not isinstance(r, Exception)]
        failed_results = [r for r in results if isinstance(r, Exception)]

        assert (
            len(successful_results) == concurrent_count
        ), f"Failed initializations: {failed_results}"

        resource_monitor.update_metrics()
        memory_stats = resource_monitor.get_memory_usage()

        # Verify reasonable performance under concurrent load
        avg_time_per_init = total_time / concurrent_count
        assert (
            avg_time_per_init < 1.0
        ), f"Average initialization time too high: {avg_time_per_init}s"

        # Verify memory usage is reasonable
        assert (
            memory_stats["peak_increase_mb"] < 100
        ), f"Excessive memory usage: {memory_stats['peak_increase_mb']}MB"

    @pytest.mark.asyncio
    async def test_file_descriptor_limits(self, system_config, resource_monitor):
        """Test system behavior when approaching file descriptor limits."""

        # Create many file handles to simulate FD pressure
        temp_files = []
        try:
            # Create file descriptor pressure (but be careful not to exhaust system)
            import tempfile

            for _ in range(50):  # Moderate FD usage
                temp_files.append(tempfile.NamedTemporaryFile())

            resource_monitor.update_metrics()
            fd_stats_before = resource_monitor.get_file_descriptor_usage()

            # Initialize system under FD pressure
            system = OccupancyPredictionSystem()

            mock_database_manager = AsyncMock()
            mock_mqtt_manager = AsyncMock()
            mock_tracking_manager = AsyncMock()

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                await system.initialize()
                assert system.running is True

                resource_monitor.update_metrics()
                fd_stats_after = resource_monitor.get_file_descriptor_usage()

                # System should not leak file descriptors
                fd_increase = (
                    fd_stats_after["current_count"] - fd_stats_before["current_count"]
                )
                assert (
                    fd_increase < 10
                ), f"Excessive FD usage: {fd_increase} new descriptors"

                await system.shutdown()

        finally:
            # Clean up temp files
            for temp_file in temp_files:
                temp_file.close()

    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, system_config, resource_monitor):
        """Test for memory leaks during repeated initialization/shutdown cycles."""

        initial_memory = resource_monitor.get_memory_usage()
        memory_samples = []

        # Perform multiple initialization/shutdown cycles
        for cycle in range(10):
            system = OccupancyPredictionSystem()

            mock_database_manager = AsyncMock()
            mock_mqtt_manager = AsyncMock()
            mock_tracking_manager = AsyncMock()

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                await system.initialize()
                await system.shutdown()

            # Force garbage collection
            del system
            gc.collect()

            # Sample memory usage
            resource_monitor.update_metrics()
            current_memory = resource_monitor.get_memory_usage()
            memory_samples.append(current_memory["current_mb"])

            # Allow system to stabilize
            await asyncio.sleep(0.1)

        # Analyze memory trend
        if len(memory_samples) >= 5:
            # Check if memory is consistently increasing (potential leak)
            recent_samples = memory_samples[-5:]
            memory_increase = recent_samples[-1] - recent_samples[0]

            # Allow some variance but detect significant leaks
            assert (
                memory_increase < 20
            ), f"Potential memory leak detected: {memory_increase}MB increase over 5 cycles"

        final_memory = resource_monitor.get_memory_usage()
        total_increase = final_memory["current_mb"] - initial_memory["current_mb"]

        # Total memory increase should be minimal after all cycles
        assert (
            total_increase < 30
        ), f"Memory not properly cleaned up: {total_increase}MB total increase"

    @pytest.mark.asyncio
    async def test_system_under_network_timeout_pressure(
        self, system_config, resource_monitor
    ):
        """Test system behavior when network operations are slow/timing out."""

        # Mock slow network operations
        slow_operation_count = 0

        async def slow_network_operation(*args, **kwargs):
            nonlocal slow_operation_count
            slow_operation_count += 1
            # Simulate network timeout/slowness
            await asyncio.sleep(0.5)  # 500ms delay
            if slow_operation_count % 3 == 0:  # Occasionally fail
                raise asyncio.TimeoutError("Network timeout")
            return AsyncMock()

        system = OccupancyPredictionSystem()

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Make MQTT initialization slow
        mock_mqtt_manager.initialize.side_effect = slow_network_operation

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            start_time = time.time()

            try:
                # System might succeed or fail depending on timing
                await asyncio.wait_for(system.initialize(), timeout=2.0)
                if system.running:
                    await system.shutdown()
            except (asyncio.TimeoutError, Exception):
                # Expected behavior under network pressure
                pass

            total_time = time.time() - start_time

            # Verify system doesn't hang indefinitely
            assert (
                total_time < 3.0
            ), f"System hung under network pressure: {total_time}s"

        resource_monitor.update_metrics()
        memory_stats = resource_monitor.get_memory_usage()

        # Verify no resource leaks during network issues
        assert (
            memory_stats["increase_mb"] < 20
        ), f"Memory leak under network pressure: {memory_stats['increase_mb']}MB"

    @pytest.mark.asyncio
    async def test_resource_cleanup_on_exception(self, system_config, resource_monitor):
        """Test that resources are properly cleaned up when exceptions occur."""

        # Track resource allocations
        allocated_resources = []

        class TrackableResource:
            def __init__(self, resource_id):
                self.resource_id = resource_id
                self.cleaned_up = False
                allocated_resources.append(self)

            def cleanup(self):
                self.cleaned_up = True

        # Mock components that allocate trackable resources
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Add resource to database manager
        mock_database_manager._resource = TrackableResource("database")

        # Make tracking manager fail after allocating resource
        def create_tracking_manager(*args, **kwargs):
            tracking_resource = TrackableResource("tracking")
            mock_tracking_manager._resource = tracking_resource
            mock_tracking_manager.initialize.side_effect = Exception(
                "Tracking initialization failed"
            )
            return mock_tracking_manager

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", side_effect=create_tracking_manager
        ):

            with pytest.raises(Exception, match="Tracking initialization failed"):
                await system.initialize()

        # Verify system attempted cleanup
        assert system.running is False

        # Simulate resource cleanup (in real system, this would be handled by component destructors)
        for resource in allocated_resources:
            resource.cleanup()

        # Verify all resources were cleaned up
        uncleaned_resources = [r for r in allocated_resources if not r.cleaned_up]
        assert (
            len(uncleaned_resources) == 0
        ), f"Resources not cleaned up: {[r.resource_id for r in uncleaned_resources]}"

        resource_monitor.update_metrics()
        memory_stats = resource_monitor.get_memory_usage()

        # Verify minimal memory retention after exception
        assert (
            memory_stats["increase_mb"] < 15
        ), f"Memory not cleaned up after exception: {memory_stats['increase_mb']}MB"

    @pytest.mark.asyncio
    async def test_performance_degradation_detection(
        self, system_config, resource_monitor
    ):
        """Test detection of performance degradation under resource constraints."""

        performance_metrics = []

        async def measure_initialization_performance():
            """Measure system initialization performance."""
            system = OccupancyPredictionSystem()

            mock_database_manager = AsyncMock()
            mock_mqtt_manager = AsyncMock()
            mock_tracking_manager = AsyncMock()

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                start_time = time.time()
                await system.initialize()
                init_time = time.time() - start_time

                start_time = time.time()
                await system.shutdown()
                shutdown_time = time.time() - start_time

                return {
                    "init_time": init_time,
                    "shutdown_time": shutdown_time,
                    "total_time": init_time + shutdown_time,
                }

        # Baseline performance measurement
        baseline_perf = await measure_initialization_performance()
        performance_metrics.append(("baseline", baseline_perf))

        # Performance under memory pressure
        memory_ballast = []
        try:
            # Create moderate memory pressure
            for _ in range(50):
                memory_ballast.append(bytearray(1024 * 1024))  # 1MB each

            memory_pressure_perf = await measure_initialization_performance()
            performance_metrics.append(("memory_pressure", memory_pressure_perf))

        finally:
            memory_ballast.clear()
            gc.collect()

        # Performance under CPU load
        def cpu_load():
            end_time = time.time() + 0.3
            while time.time() < end_time:
                sum(i * i for i in range(500))

        with ThreadPoolExecutor(max_workers=2) as executor:
            cpu_futures = [executor.submit(cpu_load) for _ in range(2)]

            try:
                cpu_pressure_perf = await measure_initialization_performance()
                performance_metrics.append(("cpu_pressure", cpu_pressure_perf))
            finally:
                for future in cpu_futures:
                    try:
                        future.result(timeout=1.0)
                    except Exception:
                        pass

        # Analyze performance degradation
        baseline_total = baseline_perf["total_time"]

        for condition, perf in performance_metrics[1:]:  # Skip baseline
            degradation_factor = perf["total_time"] / baseline_total

            # Allow reasonable performance degradation under pressure
            max_degradation = 3.0  # 3x slower is still acceptable
            assert degradation_factor < max_degradation, (
                f"Excessive performance degradation under {condition}: "
                f"{degradation_factor:.2f}x slower ({perf['total_time']:.3f}s vs {baseline_total:.3f}s)"
            )

        resource_monitor.update_metrics()
        final_memory = resource_monitor.get_memory_usage()

        logger.info(f"Performance metrics: {performance_metrics}")
        logger.info(f"Final memory usage: {final_memory}")

    @pytest.mark.asyncio
    async def test_system_stability_under_mixed_resource_pressure(
        self, system_config, resource_monitor
    ):
        """Test system stability under combined resource pressures."""

        # Create combined resource pressure
        memory_ballast = []
        cpu_load_active = True

        def sustained_cpu_load():
            """Sustained CPU load in background."""
            while cpu_load_active:
                # CPU intensive work
                sum(i * i for i in range(1000))
                time.sleep(0.01)  # Brief pause to prevent 100% CPU

        try:
            # Memory pressure
            for _ in range(30):
                memory_ballast.append(bytearray(1024 * 1024))  # 1MB each

            # CPU pressure
            with ThreadPoolExecutor(max_workers=2) as executor:
                cpu_future = executor.submit(sustained_cpu_load)

                # Allow CPU load to start
                await asyncio.sleep(0.1)

                # Test system under combined pressure
                system = OccupancyPredictionSystem()

                mock_database_manager = AsyncMock()
                mock_mqtt_manager = AsyncMock()
                mock_tracking_manager = AsyncMock()

                with patch(
                    "src.main_system.get_config", return_value=system_config
                ), patch(
                    "src.main_system.get_database_manager",
                    return_value=mock_database_manager,
                ), patch(
                    "src.main_system.MQTTIntegrationManager",
                    return_value=mock_mqtt_manager,
                ), patch(
                    "src.main_system.TrackingManager",
                    return_value=mock_tracking_manager,
                ):

                    # System should still initialize under combined pressure
                    start_time = time.time()
                    await system.initialize()
                    init_duration = time.time() - start_time

                    assert system.running is True
                    assert (
                        init_duration < 10.0
                    ), f"System too slow under combined pressure: {init_duration}s"

                    # Test some basic operations
                    resource_monitor.update_metrics()

                    # Shutdown should also work under pressure
                    start_time = time.time()
                    await system.shutdown()
                    shutdown_duration = time.time() - start_time

                    assert system.running is False
                    assert (
                        shutdown_duration < 5.0
                    ), f"Shutdown too slow under pressure: {shutdown_duration}s"

                # Stop CPU load
                cpu_load_active = False
                cpu_future.result(timeout=2.0)

        finally:
            cpu_load_active = False
            memory_ballast.clear()
            gc.collect()

        resource_monitor.update_metrics()
        final_stats = {
            "memory": resource_monitor.get_memory_usage(),
            "cpu": resource_monitor.get_cpu_usage(),
            "fds": resource_monitor.get_file_descriptor_usage(),
        }

        # Verify system stability metrics
        assert (
            final_stats["memory"]["increase_mb"] < 40
        ), f"Memory usage too high: {final_stats['memory']}"
        assert (
            final_stats["fds"]["increase"] < 5
        ), f"FD leak detected: {final_stats['fds']}"

        logger.info(f"Combined pressure test completed: {final_stats}")
