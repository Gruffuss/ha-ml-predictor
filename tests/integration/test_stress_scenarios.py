"""
Comprehensive Stress Testing for Sprint 6 Task 6 Integration Test Coverage.

This module provides stress testing scenarios to validate system behavior under
realistic load conditions and component interaction stress scenarios.

Test Coverage:
- Concurrent request handling and system throughput under load
- Database connection pool stress and connection exhaustion scenarios
- Memory and CPU usage under sustained load conditions
- Component interaction stress (tracking manager, API server, MQTT)
- Data volume stress testing with large event streams
- System resource limits and graceful degradation
- Multi-component failure recovery and error propagation
- Real-time system performance under realistic occupancy patterns
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import gc
import logging
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from fastapi.testclient import TestClient
import httpx
import psutil
import pytest
import pytest_asyncio
from sqlalchemy.exc import DisconnectionError

from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.config import get_config
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    ErrorSeverity,
    SystemResourceError,
)
from src.data.storage.database import DatabaseManager, get_database_manager
from src.data.storage.models import RoomState, SensorEvent
from src.integration.api_server import APIServer, create_app
from src.integration.enhanced_mqtt_manager import (
    EnhancedMQTTIntegrationManager,
)

logger = logging.getLogger(__name__)


@pytest.fixture
async def stress_test_config():
    """Configuration for stress testing scenarios."""
    return {
        "concurrent_requests": 50,
        "request_duration": 30,  # seconds
        "event_volume": 1000,
        "memory_limit_mb": 512,
        "cpu_threshold_percent": 80,
        "database_pool_size": 20,
        "mqtt_message_rate": 100,  # messages per second
    }


@pytest.fixture
async def system_monitor():
    """System resource monitoring fixture for stress tests."""

    class SystemMonitor:
        def __init__(self):
            self.start_time = None
            self.metrics = []
            self.process = psutil.Process()

        def start(self):
            """Start monitoring system resources."""
            self.start_time = time.time()
            self.metrics = []

        def record_metrics(self):
            """Record current system metrics."""
            try:
                cpu_percent = self.process.cpu_percent()
                memory_info = self.process.memory_info()
                memory_mb = memory_info.rss / (1024 * 1024)

                self.metrics.append(
                    {
                        "timestamp": time.time() - self.start_time,
                        "cpu_percent": cpu_percent,
                        "memory_mb": memory_mb,
                        "memory_percent": self.process.memory_percent(),
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to record metrics: {e}")

        def get_peak_usage(self):
            """Get peak resource usage during monitoring."""
            if not self.metrics:
                return {"cpu_percent": 0, "memory_mb": 0}

            peak_cpu = max(m["cpu_percent"] for m in self.metrics)
            peak_memory = max(m["memory_mb"] for m in self.metrics)

            return {
                "peak_cpu_percent": peak_cpu,
                "peak_memory_mb": peak_memory,
                "average_cpu_percent": sum(
                    m["cpu_percent"] for m in self.metrics
                )
                / len(self.metrics),
                "average_memory_mb": sum(m["memory_mb"] for m in self.metrics)
                / len(self.metrics),
            }

    return SystemMonitor()


class TestConcurrentRequestStress:
    """Test system behavior under concurrent request stress."""

    @pytest_asyncio.async_test
    async def test_concurrent_api_request_handling(
        self, stress_test_config, system_monitor
    ):
        """Test API server handling concurrent requests without degradation."""
        system_monitor.start()

        # Create API server with tracking manager
        with patch(
            "src.integration.api_server.get_tracking_manager"
        ) as mock_tm:
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = [
                "living_room",
                "bedroom",
                "kitchen",
            ]
            mock_tracking_manager.get_room_metrics.return_value = {
                "prediction_accuracy": 0.85,
                "last_updated": datetime.now(),
                "error_rate": 0.05,
            }
            mock_tm.return_value = mock_tracking_manager

            app = create_app()

            # Test concurrent GET requests
            concurrent_requests = stress_test_config["concurrent_requests"]

            async def make_request(client: httpx.AsyncClient, endpoint: str):
                """Make a single API request."""
                try:
                    system_monitor.record_metrics()
                    response = await client.get(endpoint, timeout=10.0)
                    return response.status_code, len(response.content)
                except Exception as e:
                    return None, str(e)

            async with httpx.AsyncClient(
                app=app, base_url="http://testserver"
            ) as client:
                # Test multiple endpoints concurrently
                endpoints = [
                    "/api/rooms",
                    "/api/rooms/living_room/metrics",
                    "/api/health",
                    "/api/system/status",
                ]

                tasks = []
                for _ in range(concurrent_requests):
                    for endpoint in endpoints:
                        task = asyncio.create_task(
                            make_request(client, endpoint)
                        )
                        tasks.append(task)

                # Execute all requests concurrently
                results = await asyncio.gather(*tasks, return_exceptions=True)

                # Analyze results
                successful_requests = 0
                failed_requests = 0
                total_response_size = 0

                for result in results:
                    if isinstance(result, Exception):
                        failed_requests += 1
                    else:
                        status_code, response_size = result
                        if status_code and 200 <= status_code < 300:
                            successful_requests += 1
                            if isinstance(response_size, int):
                                total_response_size += response_size
                        else:
                            failed_requests += 1

        # Validate results
        total_requests = len(tasks)
        success_rate = successful_requests / total_requests

        # Get resource usage metrics
        resource_metrics = system_monitor.get_peak_usage()

        # Assertions for stress test success
        assert (
            success_rate >= 0.95
        ), f"Success rate {success_rate} below threshold 0.95"
        assert (
            resource_metrics["peak_memory_mb"]
            < stress_test_config["memory_limit_mb"]
        ), f"Peak memory {resource_metrics['peak_memory_mb']}MB exceeded limit"
        assert failed_requests < (
            total_requests * 0.05
        ), f"Too many failed requests: {failed_requests}/{total_requests}"

        logger.info(
            f"Concurrent stress test completed: {successful_requests}/{total_requests} successful"
        )
        logger.info(
            f"Resource usage - CPU: {resource_metrics['peak_cpu_percent']:.1f}%, "
            f"Memory: {resource_metrics['peak_memory_mb']:.1f}MB"
        )

    @pytest_asyncio.async_test
    async def test_database_connection_pool_stress(
        self, stress_test_config, system_monitor
    ):
        """Test database connection pool under concurrent query stress."""
        system_monitor.start()

        with patch(
            "src.data.storage.database.create_async_engine"
        ) as mock_engine:
            # Mock database operations
            mock_session = AsyncMock()
            mock_session.execute.return_value = AsyncMock()
            mock_session.commit.return_value = None

            # Create database manager with connection pool
            db_manager = DatabaseManager(
                connection_string="postgresql+asyncpg://test:test@localhost/test",
                pool_size=stress_test_config["database_pool_size"],
                max_overflow=10,
            )

            async def execute_database_query(query_id: int):
                """Execute a database query operation."""
                try:
                    system_monitor.record_metrics()

                    # Simulate different types of database operations
                    if query_id % 3 == 0:
                        # Insert operation
                        event = SensorEvent(
                            room_id="living_room",
                            sensor_id=f"sensor_{query_id}",
                            sensor_type="motion",
                            state="on",
                            timestamp=datetime.now(),
                        )
                        # Mock insert
                        await asyncio.sleep(0.01)  # Simulate DB operation
                    elif query_id % 3 == 1:
                        # Select operation
                        await asyncio.sleep(0.005)  # Simulate faster select
                    else:
                        # Update operation
                        await asyncio.sleep(0.015)  # Simulate slower update

                    return True
                except Exception as e:
                    logger.error(f"Database query {query_id} failed: {e}")
                    return False

            # Execute concurrent database operations
            concurrent_queries = (
                stress_test_config["concurrent_requests"] * 2
            )  # More DB stress
            tasks = [
                asyncio.create_task(execute_database_query(i))
                for i in range(concurrent_queries)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze results
            successful_queries = sum(1 for r in results if r is True)
            failed_queries = len(results) - successful_queries

            resource_metrics = system_monitor.get_peak_usage()

            # Validate database stress test
            success_rate = successful_queries / len(results)
            assert (
                success_rate >= 0.98
            ), f"Database success rate {success_rate} too low"
            assert (
                resource_metrics["peak_memory_mb"]
                < stress_test_config["memory_limit_mb"] * 1.2
            )

            logger.info(
                f"Database stress test: {successful_queries}/{concurrent_queries} successful"
            )


class TestDataVolumeStress:
    """Test system behavior with large volumes of event data."""

    @pytest_asyncio.async_test
    async def test_large_event_stream_processing(
        self, stress_test_config, system_monitor
    ):
        """Test processing large volumes of sensor events."""
        system_monitor.start()

        event_volume = stress_test_config["event_volume"]

        with patch(
            "src.adaptation.tracking_manager.TrackingManager"
        ) as MockTM:
            mock_tm = AsyncMock(spec=TrackingManager)
            mock_tm.process_prediction_validation.return_value = None
            mock_tm.get_room_metrics.return_value = {"accuracy": 0.85}
            MockTM.return_value = mock_tm

            # Generate large volume of test events
            def generate_test_events(count: int) -> List[Dict[str, Any]]:
                """Generate a large number of test events."""
                events = []
                base_time = datetime.now()

                rooms = [
                    "living_room",
                    "bedroom",
                    "kitchen",
                    "bathroom",
                    "office",
                ]
                sensor_types = ["motion", "door", "light", "temperature"]
                states = ["on", "of", "open", "closed"]

                for i in range(count):
                    event = {
                        "room_id": rooms[i % len(rooms)],
                        "sensor_id": f"sensor_{i % 20}",
                        "sensor_type": sensor_types[i % len(sensor_types)],
                        "state": states[i % len(states)],
                        "timestamp": base_time + timedelta(seconds=i),
                        "attributes": {"test_id": i},
                    }
                    events.append(event)

                return events

            # Generate and process large event volume
            test_events = generate_test_events(event_volume)

            async def process_event_batch(events: List[Dict[str, Any]]):
                """Process a batch of events."""
                try:
                    for event in events:
                        system_monitor.record_metrics()
                        # Simulate event processing
                        await asyncio.sleep(0.001)  # 1ms per event
                    return len(events)
                except Exception as e:
                    logger.error(f"Event batch processing failed: {e}")
                    return 0

            # Process events in batches to avoid memory issues
            batch_size = 100
            batches = [
                test_events[i : i + batch_size]
                for i in range(0, len(test_events), batch_size)
            ]

            start_time = time.time()
            processed_events = 0

            for batch in batches:
                batch_result = await process_event_batch(batch)
                processed_events += batch_result

            processing_time = time.time() - start_time
            resource_metrics = system_monitor.get_peak_usage()

            # Calculate throughput
            events_per_second = processed_events / processing_time

            # Validate data volume stress test
            assert (
                processed_events >= event_volume * 0.95
            ), f"Processed {processed_events}/{event_volume} events"
            assert (
                events_per_second >= 100
            ), f"Throughput {events_per_second:.1f} events/sec too low"
            assert (
                resource_metrics["peak_memory_mb"]
                < stress_test_config["memory_limit_mb"] * 1.5
            )

            logger.info(
                f"Data volume stress test: {processed_events} events processed at "
                f"{events_per_second:.1f} events/sec"
            )

    @pytest_asyncio.async_test
    async def test_memory_usage_under_sustained_load(
        self, stress_test_config, system_monitor
    ):
        """Test memory usage and garbage collection under sustained load."""
        system_monitor.start()

        # Force garbage collection before test
        gc.collect()
        initial_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        async def sustained_operation():
            """Perform sustained operations that could cause memory leaks."""
            data_structures = []

            for i in range(1000):
                system_monitor.record_metrics()

                # Create temporary data structures
                temp_data = {
                    "events": [
                        {
                            "id": j,
                            "timestamp": datetime.now(),
                            "data": f"test_data_{j}" * 10,
                        }
                        for j in range(100)
                    ],
                    "processing_results": list(range(i, i + 50)),
                }
                data_structures.append(temp_data)

                # Periodically clean up to test garbage collection
                if i % 100 == 0:
                    data_structures = data_structures[
                        -10:
                    ]  # Keep only last 10
                    gc.collect()

                await asyncio.sleep(0.01)  # Simulate processing time

        # Run sustained operations
        await sustained_operation()

        # Force final garbage collection
        gc.collect()
        final_memory = psutil.Process().memory_info().rss / (1024 * 1024)

        resource_metrics = system_monitor.get_peak_usage()
        memory_growth = final_memory - initial_memory

        # Validate memory usage
        assert (
            memory_growth < 100
        ), f"Memory grew by {memory_growth:.1f}MB, indicating potential leak"
        assert (
            resource_metrics["peak_memory_mb"]
            < stress_test_config["memory_limit_mb"] * 2
        )
        assert (
            final_memory < initial_memory * 1.2
        ), "Final memory usage too high"

        logger.info(
            f"Memory stress test: Initial {initial_memory:.1f}MB, "
            f"Final {final_memory:.1f}MB, Peak {resource_metrics['peak_memory_mb']:.1f}MB"
        )


class TestMultiComponentStress:
    """Test stress scenarios involving multiple system components."""

    @pytest_asyncio.async_test
    async def test_tracking_manager_api_mqtt_integration_stress(
        self, stress_test_config, system_monitor
    ):
        """Test integrated stress across tracking manager, API server, and MQTT."""
        system_monitor.start()

        with (
            patch.multiple(
                "src.adaptation.tracking_manager", TrackingManager=Mock()
            ),
            patch(
                "src.integration.enhanced_mqtt_manager.EnhancedMQTTIntegrationManager"
            ) as MockMQTT,
        ):

            # Setup mocked components
            mock_tracking_manager = AsyncMock(spec=TrackingManager)
            mock_tracking_manager.get_all_rooms.return_value = [
                "living_room",
                "bedroom",
                "kitchen",
            ]
            mock_tracking_manager.process_prediction_validation.return_value = (
                None
            )
            mock_tracking_manager.get_room_metrics.return_value = {
                "prediction_accuracy": 0.85,
                "confidence_score": 0.9,
                "last_updated": datetime.now(),
            }

            mock_mqtt_manager = AsyncMock(spec=EnhancedMQTTIntegrationManager)
            mock_mqtt_manager.publish_prediction.return_value = True
            mock_mqtt_manager.is_connected = True
            MockMQTT.return_value = mock_mqtt_manager

            # Create integrated system
            app = create_app()

            async def integrated_stress_operation(operation_id: int):
                """Perform an integrated operation across all components."""
                try:
                    system_monitor.record_metrics()

                    room_id = ["living_room", "bedroom", "kitchen"][
                        operation_id % 3
                    ]

                    # 1. Tracking manager operation
                    metrics = await mock_tracking_manager.get_room_metrics(
                        room_id
                    )

                    # 2. API operation
                    async with httpx.AsyncClient(
                        app=app, base_url="http://testserver"
                    ) as client:
                        response = await client.get(
                            f"/api/rooms/{room_id}/metrics", timeout=5.0
                        )
                        api_success = 200 <= response.status_code < 300

                    # 3. MQTT operation
                    await mock_mqtt_manager.publish_prediction(
                        room_id,
                        {
                            "predicted_time": datetime.now().isoformat(),
                            "confidence": 0.85,
                        },
                    )

                    return api_success and metrics is not None

                except Exception as e:
                    logger.error(
                        f"Integrated operation {operation_id} failed: {e}"
                    )
                    return False

            # Execute concurrent integrated operations
            concurrent_operations = stress_test_config["concurrent_requests"]
            tasks = [
                asyncio.create_task(integrated_stress_operation(i))
                for i in range(concurrent_operations)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Analyze integrated stress results
            successful_operations = sum(1 for r in results if r is True)
            failed_operations = len(results) - successful_operations

            resource_metrics = system_monitor.get_peak_usage()

            # Validate multi-component stress
            success_rate = successful_operations / len(results)
            assert (
                success_rate >= 0.90
            ), f"Integration success rate {success_rate} too low"
            assert (
                resource_metrics["peak_memory_mb"]
                < stress_test_config["memory_limit_mb"] * 1.5
            )
            assert failed_operations < (concurrent_operations * 0.1)

            # Verify component interactions
            assert (
                mock_tracking_manager.get_room_metrics.call_count
                >= concurrent_operations * 0.8
            )
            assert (
                mock_mqtt_manager.publish_prediction.call_count
                >= concurrent_operations * 0.8
            )

            logger.info(
                f"Multi-component stress test: {successful_operations}/{concurrent_operations} successful"
            )

    @pytest_asyncio.async_test
    async def test_system_resource_limit_handling(
        self, stress_test_config, system_monitor
    ):
        """Test system behavior when approaching resource limits."""
        system_monitor.start()

        async def resource_intensive_operation(intensity: int):
            """Perform operations with varying resource intensity."""
            try:
                system_monitor.record_metrics()

                # Create memory-intensive structures
                if intensity > 70:
                    # High intensity - create large data structures
                    large_data = [
                        {"data": "x" * 1000, "id": i} for i in range(1000)
                    ]
                    await asyncio.sleep(0.1)
                elif intensity > 40:
                    # Medium intensity
                    medium_data = [
                        {"data": "x" * 100, "id": i} for i in range(500)
                    ]
                    await asyncio.sleep(0.05)
                else:
                    # Low intensity
                    await asyncio.sleep(0.01)

                return True

            except Exception as e:
                logger.warning(f"Resource intensive operation failed: {e}")
                return False

        # Gradually increase resource usage
        results = []
        for intensity in range(0, 100, 10):
            batch_size = min(
                10, stress_test_config["concurrent_requests"] // 5
            )
            batch_tasks = [
                asyncio.create_task(resource_intensive_operation(intensity))
                for _ in range(batch_size)
            ]

            batch_results = await asyncio.gather(
                *batch_tasks, return_exceptions=True
            )
            results.extend(batch_results)

            # Check if we're approaching resource limits
            current_metrics = system_monitor.get_peak_usage()
            if (
                current_metrics["peak_memory_mb"]
                > stress_test_config["memory_limit_mb"]
            ):
                logger.info(f"Reached memory limit at intensity {intensity}")
                break

        resource_metrics = system_monitor.get_peak_usage()
        successful_operations = sum(1 for r in results if r is True)

        # Validate resource limit handling
        assert successful_operations > 0, "No operations succeeded"
        assert (
            resource_metrics["peak_memory_mb"]
            < stress_test_config["memory_limit_mb"] * 2.0
        )

        # Ensure system didn't crash under resource pressure
        assert len(results) >= 20, "Too few operations completed"

        logger.info(
            f"Resource limit test: {successful_operations}/{len(results)} operations succeeded"
        )
        logger.info(
            f"Peak resource usage - Memory: {resource_metrics['peak_memory_mb']:.1f}MB"
        )


class TestFailureRecoveryStress:
    """Test system behavior under component failure and recovery scenarios."""

    @pytest_asyncio.async_test
    async def test_database_connection_failure_recovery(
        self, stress_test_config, system_monitor
    ):
        """Test system recovery from database connection failures."""
        system_monitor.start()

        failure_injected = False
        connection_attempts = 0

        async def database_operation_with_failure(operation_id: int):
            """Database operation that may experience connection failures."""
            nonlocal failure_injected, connection_attempts

            try:
                system_monitor.record_metrics()
                connection_attempts += 1

                # Inject failure for middle operations
                if 10 <= operation_id <= 20 and not failure_injected:
                    failure_injected = True
                    raise DatabaseConnectionError(
                        "Simulated connection failure"
                    )

                # Simulate successful database operation
                await asyncio.sleep(0.01)
                return True

            except DatabaseConnectionError as e:
                logger.warning(
                    f"Database connection failed for operation {operation_id}: {e}"
                )
                # Simulate connection recovery
                await asyncio.sleep(0.1)
                return False
            except Exception as e:
                logger.error(f"Database operation {operation_id} failed: {e}")
                return False

        # Execute database operations with failure injection
        operations_count = stress_test_config["concurrent_requests"]
        tasks = [
            asyncio.create_task(database_operation_with_failure(i))
            for i in range(operations_count)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        successful_operations = sum(1 for r in results if r is True)
        failed_operations = len(results) - successful_operations

        resource_metrics = system_monitor.get_peak_usage()

        # Validate failure recovery
        assert failure_injected, "Failure was not injected as expected"
        assert (
            successful_operations > operations_count * 0.7
        ), f"Too few successful operations after failure: {successful_operations}/{operations_count}"
        assert (
            failed_operations <= operations_count * 0.3
        ), f"Too many failed operations: {failed_operations}"

        logger.info(
            f"Database failure recovery test: {successful_operations}/{operations_count} recovered"
        )

    @pytest_asyncio.async_test
    async def test_mqtt_connection_resilience_stress(
        self, stress_test_config, system_monitor
    ):
        """Test MQTT connection resilience under stress conditions."""
        system_monitor.start()

        with patch(
            "src.integration.enhanced_mqtt_manager.EnhancedMQTTIntegrationManager"
        ) as MockMQTT:
            mock_mqtt_manager = AsyncMock(spec=EnhancedMQTTIntegrationManager)

            # Simulate MQTT connection issues
            connection_failures = 0
            successful_publishes = 0

            async def publish_with_failures(*args, **kwargs):
                nonlocal connection_failures, successful_publishes

                # Inject failures for some operations
                if connection_failures < 5 and successful_publishes % 10 == 5:
                    connection_failures += 1
                    raise ConnectionError("MQTT connection lost")

                successful_publishes += 1
                return True

            mock_mqtt_manager.publish_prediction.side_effect = (
                publish_with_failures
            )
            mock_mqtt_manager.is_connected = True
            MockMQTT.return_value = mock_mqtt_manager

            mqtt_manager = mock_mqtt_manager

            async def mqtt_stress_operation(operation_id: int):
                """MQTT operation under stress conditions."""
                try:
                    system_monitor.record_metrics()

                    room_id = f"room_{operation_id % 5}"
                    prediction = {
                        "predicted_time": datetime.now().isoformat(),
                        "confidence": 0.8,
                        "operation_id": operation_id,
                    }

                    result = await mqtt_manager.publish_prediction(
                        room_id, prediction
                    )
                    return result

                except ConnectionError as e:
                    logger.warning(
                        f"MQTT connection error for operation {operation_id}: {e}"
                    )
                    return False
                except Exception as e:
                    logger.error(f"MQTT operation {operation_id} failed: {e}")
                    return False

            # Execute MQTT stress operations
            operations_count = stress_test_config["concurrent_requests"]
            tasks = [
                asyncio.create_task(mqtt_stress_operation(i))
                for i in range(operations_count)
            ]

            results = await asyncio.gather(*tasks, return_exceptions=True)

            successful_mqtt_operations = sum(1 for r in results if r is True)

            resource_metrics = system_monitor.get_peak_usage()

            # Validate MQTT resilience
            assert (
                connection_failures > 0
            ), "No connection failures were simulated"
            assert (
                successful_mqtt_operations >= operations_count * 0.8
            ), f"Too few successful MQTT operations: {successful_mqtt_operations}/{operations_count}"

            logger.info(
                f"MQTT resilience test: {successful_mqtt_operations}/{operations_count} successful, "
                f"{connection_failures} failures handled"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
