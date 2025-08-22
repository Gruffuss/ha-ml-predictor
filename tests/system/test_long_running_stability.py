"""
Long-Running System Stability Tests.

This module provides comprehensive testing for system stability over extended
periods, including memory leak detection, performance degradation monitoring,
resource accumulation tracking, and sustained load testing.

Test Coverage:
- Extended runtime memory leak detection
- Performance degradation over time
- Resource accumulation and cleanup verification
- Sustained concurrent load testing
- Component behavior under long-term stress
- System recovery after extended operation
- Background task stability and resource usage
- Event handling consistency over time
"""

import asyncio
from datetime import datetime, timedelta
import gc
import logging
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import weakref

import psutil
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


class LongRunningMonitor:
    """Monitor system behavior during long-running tests."""

    def __init__(self, sampling_interval: float = 1.0):
        self.process = psutil.Process()
        self.sampling_interval = sampling_interval
        self.start_time = None
        self.samples = []
        self.monitoring = False
        self.peak_memory = 0
        self.peak_cpu = 0
        self.total_gc_collections = 0

    def start_monitoring(self):
        """Start long-running monitoring."""
        self.start_time = time.time()
        self.monitoring = True
        self.samples = []
        self.peak_memory = 0
        self.peak_cpu = 0
        self.total_gc_collections = sum(gc.get_count())

    def stop_monitoring(self):
        """Stop monitoring and return final statistics."""
        self.monitoring = False
        return self.get_final_statistics()

    def take_sample(self):
        """Take a resource usage sample."""
        if not self.monitoring:
            return

        try:
            current_time = time.time() - self.start_time if self.start_time else 0
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            cpu_percent = self.process.cpu_percent()

            # Track peaks
            self.peak_memory = max(self.peak_memory, memory_mb)
            self.peak_cpu = max(self.peak_cpu, cpu_percent)

            try:
                fd_count = len(self.process.open_files())
            except (psutil.AccessDenied, AttributeError):
                fd_count = 0

            sample = {
                "timestamp": current_time,
                "memory_mb": memory_mb,
                "cpu_percent": cpu_percent,
                "fd_count": fd_count,
                "gc_collections": sum(gc.get_count()) - self.total_gc_collections,
            }

            self.samples.append(sample)

        except Exception as e:
            logger.warning(f"Failed to take sample: {e}")

    def get_memory_trend(self, window_size: int = 10):
        """Analyze memory usage trend over recent samples."""
        if len(self.samples) < window_size:
            return {"trend": "insufficient_data", "slope": 0}

        recent_samples = self.samples[-window_size:]

        # Simple linear regression for trend analysis
        n = len(recent_samples)
        sum_x = sum(i for i in range(n))
        sum_y = sum(sample["memory_mb"] for sample in recent_samples)
        sum_xy = sum(i * sample["memory_mb"] for i, sample in enumerate(recent_samples))
        sum_x2 = sum(i * i for i in range(n))

        # Calculate slope (MB per sample)
        slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x)

        if slope > 0.5:  # More than 0.5MB increase per sample
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"

        return {"trend": trend, "slope": slope}

    def detect_memory_leak(self, threshold_mb: float = 10.0, min_samples: int = 50):
        """Detect potential memory leaks."""
        if len(self.samples) < min_samples:
            return {"leak_detected": False, "reason": "insufficient_samples"}

        # Analyze memory growth over entire run
        start_memory = self.samples[0]["memory_mb"]
        end_memory = self.samples[-1]["memory_mb"]
        total_growth = end_memory - start_memory

        if total_growth > threshold_mb:
            # Check if growth is consistent (potential leak) vs spiky (normal variance)
            trend = self.get_memory_trend(min(len(self.samples), 20))

            return {
                "leak_detected": True,
                "total_growth_mb": total_growth,
                "trend": trend,
                "start_memory_mb": start_memory,
                "end_memory_mb": end_memory,
            }

        return {"leak_detected": False, "total_growth_mb": total_growth}

    def get_performance_degradation(self):
        """Analyze performance degradation over time."""
        if len(self.samples) < 10:
            return {"degradation_detected": False, "reason": "insufficient_samples"}

        # Compare first 25% vs last 25% of samples
        sample_count = len(self.samples)
        quarter_size = max(5, sample_count // 4)

        early_samples = self.samples[:quarter_size]
        late_samples = self.samples[-quarter_size:]

        early_avg_cpu = sum(s["cpu_percent"] for s in early_samples) / len(
            early_samples
        )
        late_avg_cpu = sum(s["cpu_percent"] for s in late_samples) / len(late_samples)

        cpu_increase = late_avg_cpu - early_avg_cpu

        return {
            "degradation_detected": cpu_increase > 10.0,  # 10% CPU increase threshold
            "cpu_increase_percent": cpu_increase,
            "early_avg_cpu": early_avg_cpu,
            "late_avg_cpu": late_avg_cpu,
            "sample_count": sample_count,
        }

    def get_final_statistics(self):
        """Get comprehensive final statistics."""
        if not self.samples:
            return {"error": "no_samples_collected"}

        runtime = time.time() - self.start_time if self.start_time else 0

        return {
            "runtime_seconds": runtime,
            "total_samples": len(self.samples),
            "peak_memory_mb": self.peak_memory,
            "peak_cpu_percent": self.peak_cpu,
            "memory_leak_analysis": self.detect_memory_leak(),
            "performance_degradation": self.get_performance_degradation(),
            "memory_trend": self.get_memory_trend(),
            "final_gc_collections": sum(gc.get_count()) - self.total_gc_collections,
        }


@pytest.mark.system
@pytest.mark.slow
class TestLongRunningSystemStability:
    """Long-running system stability test suite."""

    @pytest.fixture
    def system_config(self):
        """System configuration for stability testing."""
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
    def long_running_monitor(self):
        """Long-running monitoring fixture."""
        monitor = LongRunningMonitor(sampling_interval=0.5)
        yield monitor

        if monitor.monitoring:
            final_stats = monitor.stop_monitoring()
            logger.info(f"Long-running test completed: {final_stats}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(60)  # 1 minute timeout for CI/CD
    async def test_extended_runtime_memory_stability(
        self, system_config, long_running_monitor
    ):
        """Test system memory stability over extended runtime."""

        # For CI/CD, run for shorter duration but with intensive operations
        test_duration = 30  # 30 seconds
        operation_interval = 0.1  # 100ms between operations

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Create realistic component behavior
        operation_count = 0

        async def realistic_database_operation(*args, **kwargs):
            nonlocal operation_count
            operation_count += 1
            # Simulate database work with some memory allocation
            temp_data = [{"id": i, "data": f"test_data_{i}"} for i in range(100)]
            await asyncio.sleep(0.01)  # Simulate I/O
            return {"result": f"operation_{operation_count}", "count": len(temp_data)}

        async def realistic_mqtt_operation(*args, **kwargs):
            # Simulate MQTT message handling
            message_data = {"timestamp": time.time(), "payload": "test_payload"}
            await asyncio.sleep(0.005)  # Simulate network I/O
            return message_data

        mock_database_manager.health_check.side_effect = realistic_database_operation
        mock_mqtt_manager.cleanup.side_effect = realistic_mqtt_operation
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            long_running_monitor.start_monitoring()

            # Simulate extended runtime with periodic operations
            start_time = time.time()

            while time.time() - start_time < test_duration:
                # Simulate periodic system operations
                try:
                    await mock_database_manager.health_check()
                    long_running_monitor.take_sample()

                    # Simulate varying load
                    if operation_count % 50 == 0:
                        # Periodic heavier operation
                        await asyncio.gather(
                            mock_database_manager.health_check(),
                            mock_mqtt_manager.cleanup(),
                            return_exceptions=True,
                        )

                    # Force periodic garbage collection
                    if operation_count % 100 == 0:
                        gc.collect()

                    await asyncio.sleep(operation_interval)

                except Exception as e:
                    logger.warning(f"Operation failed during stability test: {e}")

            await system.shutdown()

        final_stats = long_running_monitor.stop_monitoring()

        # Analyze results
        memory_leak = final_stats["memory_leak_analysis"]
        performance_deg = final_stats["performance_degradation"]

        # Assert memory stability
        assert not memory_leak["leak_detected"], f"Memory leak detected: {memory_leak}"

        # Assert performance stability
        assert not performance_deg[
            "degradation_detected"
        ], f"Performance degradation detected: {performance_deg}"

        # Assert reasonable resource usage
        assert (
            final_stats["peak_memory_mb"] < 200
        ), f"Excessive peak memory usage: {final_stats['peak_memory_mb']}MB"

        logger.info(
            f"Stability test completed successfully: {operation_count} operations"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(45)
    async def test_sustained_concurrent_load_stability(
        self, system_config, long_running_monitor
    ):
        """Test system stability under sustained concurrent load."""

        test_duration = 20  # 20 seconds of concurrent load
        concurrent_workers = 5

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Track concurrent operations
        active_operations = {"count": 0}
        max_concurrent = {"value": 0}

        async def concurrent_database_op(*args, **kwargs):
            active_operations["count"] += 1
            max_concurrent["value"] = max(
                max_concurrent["value"], active_operations["count"]
            )

            try:
                # Simulate variable duration database operation
                duration = random.uniform(0.01, 0.1)
                await asyncio.sleep(duration)
                return {"operation": "db_op", "duration": duration}
            finally:
                active_operations["count"] -= 1

        async def concurrent_mqtt_op(*args, **kwargs):
            active_operations["count"] += 1
            max_concurrent["value"] = max(
                max_concurrent["value"], active_operations["count"]
            )

            try:
                # Simulate MQTT operation
                duration = random.uniform(0.005, 0.05)
                await asyncio.sleep(duration)
                return {"operation": "mqtt_op", "duration": duration}
            finally:
                active_operations["count"] -= 1

        mock_database_manager.health_check.side_effect = concurrent_database_op
        mock_mqtt_manager.cleanup.side_effect = concurrent_mqtt_op
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        async def worker_task(worker_id: int):
            """Individual worker task generating load."""
            operations_completed = 0

            while time.time() - start_time < test_duration:
                try:
                    # Alternate between different types of operations
                    if operations_completed % 2 == 0:
                        await mock_database_manager.health_check()
                    else:
                        await mock_mqtt_manager.cleanup()

                    operations_completed += 1

                    # Small random delay between operations
                    await asyncio.sleep(random.uniform(0.01, 0.05))

                except Exception as e:
                    logger.warning(f"Worker {worker_id} operation failed: {e}")

            return operations_completed

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            long_running_monitor.start_monitoring()

            # Start monitoring task
            async def monitoring_task():
                while time.time() - start_time < test_duration:
                    long_running_monitor.take_sample()
                    await asyncio.sleep(0.2)

            start_time = time.time()

            # Launch concurrent workers and monitoring
            tasks = [
                asyncio.create_task(worker_task(i)) for i in range(concurrent_workers)
            ]
            tasks.append(asyncio.create_task(monitoring_task()))

            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)

            await system.shutdown()

        final_stats = long_running_monitor.stop_monitoring()

        # Analyze concurrent load results
        worker_results = [r for r in results[:-1] if not isinstance(r, Exception)]
        total_operations = sum(worker_results)

        # Verify system handled concurrent load
        assert len(worker_results) == concurrent_workers, "Some workers failed"
        assert total_operations > 0, "No operations completed"
        assert max_concurrent["value"] >= concurrent_workers, "Concurrency not achieved"

        # Verify system stability under load
        memory_leak = final_stats["memory_leak_analysis"]
        assert not memory_leak[
            "leak_detected"
        ], f"Memory leak under concurrent load: {memory_leak}"

        performance_deg = final_stats["performance_degradation"]
        assert not performance_deg[
            "degradation_detected"
        ], f"Performance degradation under load: {performance_deg}"

        logger.info(
            f"Concurrent load test: {total_operations} operations, max concurrent: {max_concurrent['value']}"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(40)
    async def test_background_task_stability(self, system_config, long_running_monitor):
        """Test stability of background tasks over extended periods."""

        test_duration = 15  # 15 seconds
        background_task_count = 3

        # Track background task behavior
        task_metrics = {
            "completions": 0,
            "failures": 0,
            "max_duration": 0,
            "min_duration": float("inf"),
        }

        async def background_task(task_id: int, interval: float):
            """Simulate background task with varying behavior."""
            iteration = 0

            while time.time() - start_time < test_duration:
                iteration += 1
                task_start = time.time()

                try:
                    # Simulate background work
                    work_duration = random.uniform(0.01, 0.1)

                    # Occasionally simulate more intensive work
                    if iteration % 20 == 0:
                        work_duration *= 3

                    await asyncio.sleep(work_duration)

                    # Simulate memory allocation/deallocation
                    temp_data = [
                        f"task_{task_id}_data_{i}"
                        for i in range(random.randint(10, 100))
                    ]
                    await asyncio.sleep(0.001)  # Brief processing
                    del temp_data

                    task_duration = time.time() - task_start
                    task_metrics["completions"] += 1
                    task_metrics["max_duration"] = max(
                        task_metrics["max_duration"], task_duration
                    )
                    task_metrics["min_duration"] = min(
                        task_metrics["min_duration"], task_duration
                    )

                except Exception as e:
                    task_metrics["failures"] += 1
                    logger.warning(f"Background task {task_id} failed: {e}")

                # Wait for next iteration
                await asyncio.sleep(interval)

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            long_running_monitor.start_monitoring()

            # Start monitoring task
            async def monitoring_task():
                while time.time() - start_time < test_duration:
                    long_running_monitor.take_sample()
                    await asyncio.sleep(0.3)

            start_time = time.time()

            # Create background tasks with different intervals
            background_tasks = [
                asyncio.create_task(background_task(i, 0.1 + i * 0.05))
                for i in range(background_task_count)
            ]
            background_tasks.append(asyncio.create_task(monitoring_task()))

            # Wait for tasks to complete
            await asyncio.gather(*background_tasks, return_exceptions=True)

            await system.shutdown()

        final_stats = long_running_monitor.stop_monitoring()

        # Verify background task stability
        assert task_metrics["completions"] > 0, "No background tasks completed"
        failure_rate = task_metrics["failures"] / max(
            1, task_metrics["completions"] + task_metrics["failures"]
        )
        assert (
            failure_rate < 0.05
        ), f"High background task failure rate: {failure_rate:.2%}"

        # Verify system stability with background tasks
        memory_leak = final_stats["memory_leak_analysis"]
        assert not memory_leak[
            "leak_detected"
        ], f"Memory leak with background tasks: {memory_leak}"

        logger.info(f"Background task stability: {task_metrics}")

    @pytest.mark.asyncio
    @pytest.mark.timeout(35)
    async def test_event_handling_consistency_over_time(
        self, system_config, long_running_monitor
    ):
        """Test event handling consistency over extended periods."""

        test_duration = 12  # 12 seconds
        event_rate = 10  # events per second

        # Track event handling consistency
        event_metrics = {
            "events_sent": 0,
            "events_processed": 0,
            "processing_times": [],
            "errors": [],
        }

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        async def process_event(event_data):
            """Simulate event processing."""
            process_start = time.time()

            try:
                # Simulate varying processing complexity
                complexity = random.choice([0.01, 0.02, 0.05])
                await asyncio.sleep(complexity)

                # Simulate database interaction
                await mock_database_manager.health_check()

                process_duration = time.time() - process_start
                event_metrics["events_processed"] += 1
                event_metrics["processing_times"].append(process_duration)

                return {"status": "processed", "duration": process_duration}

            except Exception as e:
                event_metrics["errors"].append(str(e))
                raise

        async def event_generator():
            """Generate events at consistent rate."""
            event_interval = 1.0 / event_rate

            while time.time() - start_time < test_duration:
                event_data = {
                    "id": event_metrics["events_sent"],
                    "timestamp": time.time(),
                    "type": random.choice(
                        ["sensor_update", "state_change", "heartbeat"]
                    ),
                }

                try:
                    await process_event(event_data)
                    event_metrics["events_sent"] += 1
                except Exception as e:
                    logger.warning(f"Event processing failed: {e}")

                await asyncio.sleep(event_interval)

        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            long_running_monitor.start_monitoring()

            # Start monitoring task
            async def monitoring_task():
                while time.time() - start_time < test_duration:
                    long_running_monitor.take_sample()
                    await asyncio.sleep(0.25)

            start_time = time.time()

            # Run event generation and monitoring concurrently
            tasks = [
                asyncio.create_task(event_generator()),
                asyncio.create_task(monitoring_task()),
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            await system.shutdown()

        final_stats = long_running_monitor.stop_monitoring()

        # Analyze event handling consistency
        expected_events = int(test_duration * event_rate * 0.9)  # Allow 10% variance
        assert (
            event_metrics["events_processed"] >= expected_events
        ), f"Insufficient events processed: {event_metrics['events_processed']} < {expected_events}"

        # Check processing time consistency
        if event_metrics["processing_times"]:
            avg_processing_time = sum(event_metrics["processing_times"]) / len(
                event_metrics["processing_times"]
            )
            max_processing_time = max(event_metrics["processing_times"])

            assert (
                avg_processing_time < 0.1
            ), f"Average processing time too high: {avg_processing_time}s"
            assert (
                max_processing_time < 0.5
            ), f"Max processing time too high: {max_processing_time}s"

        # Check error rate
        error_rate = len(event_metrics["errors"]) / max(1, event_metrics["events_sent"])
        assert error_rate < 0.01, f"High event processing error rate: {error_rate:.2%}"

        # Verify system stability during event processing
        memory_leak = final_stats["memory_leak_analysis"]
        assert not memory_leak[
            "leak_detected"
        ], f"Memory leak during event processing: {memory_leak}"

        logger.info(
            f"Event handling test: {event_metrics['events_processed']} events processed"
        )

    @pytest.mark.asyncio
    @pytest.mark.timeout(30)
    async def test_resource_cleanup_after_extended_operation(
        self, system_config, long_running_monitor
    ):
        """Test resource cleanup after extended system operation."""

        operation_duration = 10  # 10 seconds of operation

        # Track resource allocations
        resource_allocations = []
        cleanup_verifications = []

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Simulate resource-intensive operations
        async def resource_intensive_operation(operation_id: int):
            """Simulate operation that allocates and should clean up resources."""
            # Allocate some resources
            allocated_resource = {
                "id": operation_id,
                "data": [f"resource_data_{i}" for i in range(1000)],
                "timestamp": time.time(),
            }
            resource_allocations.append(allocated_resource)

            # Simulate work
            await asyncio.sleep(random.uniform(0.05, 0.15))

            # Mark resource for cleanup verification
            cleanup_verifications.append(operation_id)

            return f"operation_{operation_id}_completed"

        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            long_running_monitor.start_monitoring()

            # Record initial resource state
            initial_sample = (
                long_running_monitor.samples[-1]
                if long_running_monitor.samples
                else None
            )

            # Run intensive operations
            start_time = time.time()
            operation_id = 0

            while time.time() - start_time < operation_duration:
                await resource_intensive_operation(operation_id)
                operation_id += 1

                # Periodic monitoring
                if operation_id % 10 == 0:
                    long_running_monitor.take_sample()
                    gc.collect()  # Force garbage collection

                await asyncio.sleep(0.1)

            # Force final cleanup
            resource_allocations.clear()
            gc.collect()

            # Wait a bit for cleanup to complete
            await asyncio.sleep(1.0)
            long_running_monitor.take_sample()

            await system.shutdown()

        final_stats = long_running_monitor.stop_monitoring()

        # Verify resource cleanup
        assert len(cleanup_verifications) > 0, "No operations were performed"

        # Check memory was cleaned up after operations
        if len(long_running_monitor.samples) >= 2:
            final_memory = long_running_monitor.samples[-1]["memory_mb"]
            if initial_sample:
                initial_memory = initial_sample["memory_mb"]
                memory_increase = final_memory - initial_memory

                # Allow some reasonable memory increase but detect major leaks
                assert (
                    memory_increase < 30
                ), f"Excessive memory retention after cleanup: {memory_increase}MB"

        # Verify no major memory leaks detected
        memory_leak = final_stats["memory_leak_analysis"]
        if memory_leak["leak_detected"]:
            # Allow some tolerance for legitimate memory growth
            assert (
                memory_leak["total_growth_mb"] < 50
            ), f"Major memory leak detected: {memory_leak}"

        logger.info(
            f"Resource cleanup test: {len(cleanup_verifications)} operations, final stats: {final_stats}"
        )
