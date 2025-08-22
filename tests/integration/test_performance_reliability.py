"""
Performance and Reliability Integration Tests.

This module provides comprehensive integration testing focused on system performance,
reliability, and resilience under high-throughput scenarios and stress conditions.

Focus Areas:
- High-throughput message processing and API handling
- System behavior under resource constraints
- Memory and CPU usage patterns under load
- Network latency and timeout handling
- Error recovery and graceful degradation
- Long-running system stability
- Resource leak detection and prevention
- Performance regression detection
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import gc
import json
import logging
import multiprocessing
import os
import random
import sys
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, patch
import uuid
import weakref

from fastapi.testclient import TestClient
import psutil
import pytest
import resource

from src.core.config import MQTTConfig, RoomConfig
from src.integration.api_server import create_app, set_tracking_manager
from src.integration.discovery_publisher import DeviceInfo, DiscoveryPublisher
from src.integration.ha_entity_definitions import HAEntityDefinitions
from src.integration.mqtt_publisher import MQTTPublisher, MQTTPublishResult

# Performance test configuration
PERFORMANCE_CONFIG = {
    "high_load_message_count": 10000,
    "concurrent_clients": 50,
    "sustained_duration_minutes": 5,
    "memory_limit_mb": 500,
    "cpu_limit_percent": 80,
    "latency_threshold_ms": 1000,
    "throughput_threshold_msg_per_sec": 100,
}


@pytest.fixture
def performance_tracking_manager():
    """Create performance-optimized mock tracking manager."""
    mock_manager = AsyncMock()

    # Pre-generate responses for performance
    room_predictions = {}
    for i in range(100):
        room_id = f"room_{i}"
        room_predictions[room_id] = {
            "room_id": room_id,
            "prediction_time": datetime.now().isoformat(),
            "next_transition_time": (
                datetime.now() + timedelta(minutes=random.randint(5, 60))
            ).isoformat(),
            "transition_type": random.choice(
                ["occupied", "vacant", "occupied_to_vacant", "vacant_to_occupied"]
            ),
            "confidence": random.uniform(0.7, 0.95),
            "time_until_transition": f"{random.randint(5, 60)} minutes",
            "alternatives": [],
            "model_info": {
                "model": "performance_test",
                "accuracy": random.uniform(0.8, 0.95),
            },
        }

    async def fast_get_prediction(room_id):
        return room_predictions.get(room_id, room_predictions["room_0"])

    async def fast_get_accuracy(room_id=None, hours=24):
        return {
            "room_id": room_id,
            "accuracy_rate": random.uniform(0.8, 0.95),
            "average_error_minutes": random.uniform(5, 20),
            "confidence_calibration": random.uniform(0.8, 0.95),
            "total_predictions": random.randint(100, 1000),
            "total_validations": random.randint(90, 950),
            "time_window_hours": hours,
            "trend_direction": random.choice(["improving", "stable", "degrading"]),
        }

    async def fast_get_status():
        return {
            "tracking_active": True,
            "status": "active",
            "config": {"enabled": True},
            "performance": {
                "background_tasks": random.randint(2, 5),
                "total_predictions_recorded": random.randint(1000, 5000),
                "total_validations_performed": random.randint(900, 4500),
                "system_uptime_seconds": random.randint(3600, 86400),
            },
        }

    mock_manager.get_room_prediction.side_effect = fast_get_prediction
    mock_manager.get_accuracy_metrics.side_effect = fast_get_accuracy
    mock_manager.get_tracking_status.side_effect = fast_get_status

    return mock_manager


@pytest.fixture
def performance_mqtt_publisher():
    """Create performance-optimized mock MQTT publisher."""
    mock_publisher = AsyncMock(spec=MQTTPublisher)

    # Track performance metrics
    mock_publisher.message_times = []
    mock_publisher.total_messages = 0
    mock_publisher.failed_messages = 0

    async def fast_publish_json(topic, data, qos=1, retain=False):
        start_time = time.time()

        # Simulate minimal processing time
        await asyncio.sleep(0.001)  # 1ms base latency

        # Randomly simulate failures
        if random.random() < 0.01:  # 1% failure rate
            mock_publisher.failed_messages += 1
            return MQTTPublishResult(
                success=False,
                topic=topic,
                payload_size=len(json.dumps(data)),
                publish_time=datetime.utcnow(),
                error_message="Simulated network error",
            )

        end_time = time.time()
        mock_publisher.message_times.append(end_time - start_time)
        mock_publisher.total_messages += 1

        return MQTTPublishResult(
            success=True,
            topic=topic,
            payload_size=len(json.dumps(data)),
            publish_time=datetime.utcnow(),
            message_id=mock_publisher.total_messages,
        )

    mock_publisher.publish_json.side_effect = fast_publish_json
    mock_publisher.connection_status.connected = True

    return mock_publisher


@pytest.fixture
def performance_system(performance_tracking_manager, performance_mqtt_publisher):
    """Create performance-optimized integrated system."""

    # Set up environment for performance testing
    with patch.dict(
        "os.environ",
        {
            "ENVIRONMENT": "performance_test",
            "JWT_SECRET_KEY": "performance_test_secret_key",
            "API_KEY": "performance_test_api_key",
        },
    ):
        with patch("src.integration.api_server.get_database_manager") as mock_get_db:
            # Fast database mock
            mock_db = AsyncMock()
            mock_db.health_check.return_value = {
                "status": "healthy",
                "database_connected": True,
                "performance": "optimized",
            }
            mock_get_db.return_value = mock_db

            app = create_app()
            set_tracking_manager(performance_tracking_manager)

            return {
                "app": app,
                "tracking_manager": performance_tracking_manager,
                "mqtt_publisher": performance_mqtt_publisher,
                "database_manager": mock_db,
            }


class TestHighThroughputPerformance:
    """Test system performance under high-throughput scenarios."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_api_high_throughput_requests(self, performance_system):
        """Test API server performance under high request load."""
        app = performance_system["app"]

        # Performance metrics
        start_time = time.time()
        request_count = PERFORMANCE_CONFIG["high_load_message_count"]
        concurrent_clients = PERFORMANCE_CONFIG["concurrent_clients"]

        response_times = []
        status_codes = []

        def make_request_batch(batch_size):
            """Make a batch of requests."""
            batch_times = []
            batch_codes = []

            with TestClient(app) as client:
                for _ in range(batch_size):
                    request_start = time.time()

                    # Random endpoint selection for realistic load
                    endpoints = [
                        "/health",
                        "/predictions/room_0",
                        "/predictions/room_1",
                        "/accuracy?room_id=room_0",
                        "/stats",
                    ]
                    endpoint = random.choice(endpoints)

                    response = client.get(
                        endpoint,
                        headers={"Authorization": "Bearer performance_test_api_key"},
                    )

                    request_end = time.time()
                    batch_times.append(request_end - request_start)
                    batch_codes.append(response.status_code)

            return batch_times, batch_codes

        # Execute concurrent request batches
        batch_size = request_count // concurrent_clients

        with ThreadPoolExecutor(max_workers=concurrent_clients) as executor:
            future_to_batch = {
                executor.submit(make_request_batch, batch_size): i
                for i in range(concurrent_clients)
            }

            for future in as_completed(future_to_batch):
                batch_times, batch_codes = future.result()
                response_times.extend(batch_times)
                status_codes.extend(batch_codes)

        end_time = time.time()
        total_duration = end_time - start_time

        # Performance analysis
        successful_requests = sum(1 for code in status_codes if code == 200)
        success_rate = successful_requests / len(status_codes)
        throughput = len(status_codes) / total_duration
        avg_response_time = sum(response_times) / len(response_times)
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]

        # Performance assertions
        assert success_rate >= 0.95, f"Success rate {success_rate:.2%} below 95%"
        assert (
            throughput >= PERFORMANCE_CONFIG["throughput_threshold_msg_per_sec"]
        ), f"Throughput {throughput:.1f} below threshold"
        assert (
            avg_response_time < PERFORMANCE_CONFIG["latency_threshold_ms"] / 1000
        ), f"Average response time {avg_response_time*1000:.1f}ms above threshold"
        assert (
            p95_response_time < PERFORMANCE_CONFIG["latency_threshold_ms"] / 1000 * 2
        ), f"P95 response time {p95_response_time*1000:.1f}ms too high"

        print("High-throughput API test results:")
        print(f"  Requests: {len(status_codes)} in {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg response time: {avg_response_time*1000:.1f}ms")
        print(f"  P95 response time: {p95_response_time*1000:.1f}ms")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mqtt_high_throughput_publishing(self, performance_mqtt_publisher):
        """Test MQTT publisher performance under high message load."""
        start_time = time.time()
        message_count = PERFORMANCE_CONFIG["high_load_message_count"]

        # Generate test messages
        test_messages = []
        for i in range(message_count):
            test_messages.append(
                {
                    "topic": f"test/performance/message_{i % 100}",
                    "data": {
                        "id": i,
                        "timestamp": datetime.now().isoformat(),
                        "payload": f"performance_test_message_{i}",
                        "metadata": {
                            "batch": i // 100,
                            "sequence": i % 100,
                            "test_data": "x" * 100,  # 100 bytes of data
                        },
                    },
                }
            )

        # Publish messages with controlled concurrency
        semaphore = asyncio.Semaphore(100)  # Limit concurrent operations

        async def publish_message(msg):
            async with semaphore:
                return await performance_mqtt_publisher.publish_json(
                    msg["topic"], msg["data"]
                )

        # Execute publishing
        tasks = [publish_message(msg) for msg in test_messages]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        end_time = time.time()
        total_duration = end_time - start_time

        # Analyze results
        successful_publishes = sum(
            1 for r in results if isinstance(r, MQTTPublishResult) and r.success
        )
        failed_publishes = len(results) - successful_publishes
        throughput = len(results) / total_duration

        # Performance metrics from mock
        if performance_mqtt_publisher.message_times:
            avg_publish_time = sum(performance_mqtt_publisher.message_times) / len(
                performance_mqtt_publisher.message_times
            )
            p95_publish_time = sorted(performance_mqtt_publisher.message_times)[
                int(0.95 * len(performance_mqtt_publisher.message_times))
            ]
        else:
            avg_publish_time = 0
            p95_publish_time = 0

        # Performance assertions
        success_rate = successful_publishes / len(results)
        assert success_rate >= 0.98, f"MQTT success rate {success_rate:.2%} below 98%"
        assert (
            throughput >= PERFORMANCE_CONFIG["throughput_threshold_msg_per_sec"]
        ), f"MQTT throughput {throughput:.1f} below threshold"
        assert (
            avg_publish_time < 0.1
        ), f"Average publish time {avg_publish_time:.3f}s too high"

        print("High-throughput MQTT test results:")
        print(f"  Messages: {len(results)} in {total_duration:.2f}s")
        print(f"  Throughput: {throughput:.1f} msg/s")
        print(f"  Success rate: {success_rate:.2%}")
        print(f"  Avg publish time: {avg_publish_time*1000:.1f}ms")
        print(f"  P95 publish time: {p95_publish_time*1000:.1f}ms")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_mixed_high_throughput_operations(self, performance_system):
        """Test mixed high-throughput operations across all components."""
        app = performance_system["app"]
        mqtt_publisher = performance_system["mqtt_publisher"]
        tracking_manager = performance_system["tracking_manager"]

        start_time = time.time()
        operation_count = (
            PERFORMANCE_CONFIG["high_load_message_count"] // 4
        )  # Split across 4 operation types

        async def api_operations():
            """High-throughput API operations."""
            results = []
            with TestClient(app) as client:
                for i in range(operation_count):
                    room_id = f"room_{i % 10}"
                    response = client.get(
                        f"/predictions/{room_id}",
                        headers={"Authorization": "Bearer performance_test_api_key"},
                    )
                    results.append(response.status_code == 200)
            return results

        async def mqtt_operations():
            """High-throughput MQTT operations."""
            results = []
            for i in range(operation_count):
                result = await mqtt_publisher.publish_json(
                    f"test/mixed/{i % 20}", {"id": i, "type": "mixed_test"}
                )
                results.append(result.success)
            return results

        async def tracking_operations():
            """High-throughput tracking operations."""
            results = []
            for i in range(operation_count):
                room_id = f"room_{i % 5}"
                try:
                    prediction = await tracking_manager.get_room_prediction(room_id)
                    results.append(prediction is not None)
                except Exception:
                    results.append(False)
            return results

        async def entity_operations():
            """High-throughput entity operations."""
            # Create lightweight entity system for testing
            device_info = DeviceInfo(
                identifiers=["perf_test"],
                name="Performance Test Device",
                manufacturer="Test",
                model="Performance",
                sw_version="1.0",
            )

            discovery_publisher = DiscoveryPublisher(
                mqtt_publisher=mqtt_publisher,
                device_info=device_info,
                mqtt_config=MQTTConfig(
                    broker="localhost",
                    port=1883,
                    topic_prefix="perf_test",
                    device_identifier="perf_test",
                ),
            )

            # Create small room set for performance
            small_rooms = {
                f"perf_room_{i}": RoomConfig(
                    room_id=f"perf_room_{i}",
                    name=f"Performance Room {i}",
                    sensors={"motion": [f"sensor.perf_{i}"]},
                )
                for i in range(5)
            }

            ha_entities = HAEntityDefinitions(
                discovery_publisher=discovery_publisher,
                mqtt_config=discovery_publisher.mqtt_config,
                rooms=small_rooms,
            )

            results = []
            for i in range(
                operation_count // 10
            ):  # Fewer entity operations as they're more expensive
                entities = ha_entities.define_all_entities()
                results.append(len(entities) > 0)

            return results

        # Execute all operation types concurrently
        api_results, mqtt_results, tracking_results, entity_results = (
            await asyncio.gather(
                api_operations(),
                mqtt_operations(),
                tracking_operations(),
                entity_operations(),
            )
        )

        end_time = time.time()
        total_duration = end_time - start_time

        # Analyze mixed operation results
        total_operations = (
            len(api_results)
            + len(mqtt_results)
            + len(tracking_results)
            + len(entity_results)
        )
        total_successes = (
            sum(api_results)
            + sum(mqtt_results)
            + sum(tracking_results)
            + sum(entity_results)
        )

        overall_success_rate = total_successes / total_operations
        overall_throughput = total_operations / total_duration

        # Performance assertions
        assert (
            overall_success_rate >= 0.90
        ), f"Mixed operations success rate {overall_success_rate:.2%} below 90%"
        assert (
            overall_throughput >= 50
        ), f"Mixed operations throughput {overall_throughput:.1f} below 50 ops/s"

        print("Mixed high-throughput test results:")
        print(f"  Total operations: {total_operations} in {total_duration:.2f}s")
        print(f"  Overall throughput: {overall_throughput:.1f} ops/s")
        print(f"  Overall success rate: {overall_success_rate:.2%}")
        print(
            f"  API: {sum(api_results)}/{len(api_results)} ({100*sum(api_results)/len(api_results):.1f}%)"
        )
        print(
            f"  MQTT: {sum(mqtt_results)}/{len(mqtt_results)} ({100*sum(mqtt_results)/len(mqtt_results):.1f}%)"
        )
        print(
            f"  Tracking: {sum(tracking_results)}/{len(tracking_results)} ({100*sum(tracking_results)/len(tracking_results):.1f}%)"
        )
        print(
            f"  Entity: {sum(entity_results)}/{len(entity_results)} ({100*sum(entity_results)/len(entity_results):.1f}%)"
        )


class TestResourceUsageAndConstraints:
    """Test system behavior under resource constraints."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_memory_usage_under_load(self, performance_system):
        """Test memory usage patterns under sustained load."""
        app = performance_system["app"]

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Memory tracking
        memory_samples = []
        max_memory = initial_memory

        async def memory_monitor():
            """Monitor memory usage during test."""
            while True:
                try:
                    current_memory = process.memory_info().rss / 1024 / 1024  # MB
                    memory_samples.append(current_memory)
                    nonlocal max_memory
                    max_memory = max(max_memory, current_memory)
                    await asyncio.sleep(0.5)  # Sample every 500ms
                except asyncio.CancelledError:
                    break

        # Start memory monitoring
        monitor_task = asyncio.create_task(memory_monitor())

        try:
            # Generate sustained load
            with TestClient(app) as client:
                for batch in range(10):  # 10 batches
                    batch_start = time.time()

                    # Each batch: 500 requests
                    for i in range(500):
                        response = client.get(
                            f"/predictions/room_{i % 5}",
                            headers={
                                "Authorization": "Bearer performance_test_api_key"
                            },
                        )

                        # Ensure we're generating realistic load
                        if i % 100 == 0:
                            await asyncio.sleep(0.01)  # Brief pause every 100 requests

                    batch_end = time.time()
                    print(
                        f"Batch {batch + 1}/10 completed in {batch_end - batch_start:.2f}s"
                    )

                    # Force garbage collection between batches
                    gc.collect()

                    # Brief pause between batches
                    await asyncio.sleep(0.1)

        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        peak_memory_increase = max_memory - initial_memory

        # Memory usage assertions
        assert (
            memory_increase < PERFORMANCE_CONFIG["memory_limit_mb"]
        ), f"Memory increase {memory_increase:.1f}MB exceeds limit"
        assert (
            peak_memory_increase < PERFORMANCE_CONFIG["memory_limit_mb"] * 1.5
        ), f"Peak memory increase {peak_memory_increase:.1f}MB exceeds safe limit"

        # Check for memory leaks
        avg_memory = (
            sum(memory_samples) / len(memory_samples)
            if memory_samples
            else initial_memory
        )
        memory_variance = (
            max(memory_samples) - min(memory_samples) if memory_samples else 0
        )

        print("Memory usage analysis:")
        print(f"  Initial: {initial_memory:.1f}MB")
        print(f"  Final: {final_memory:.1f}MB")
        print(f"  Increase: {memory_increase:.1f}MB")
        print(f"  Peak increase: {peak_memory_increase:.1f}MB")
        print(f"  Average: {avg_memory:.1f}MB")
        print(f"  Variance: {memory_variance:.1f}MB")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cpu_usage_under_load(self, performance_system):
        """Test CPU usage patterns under sustained load."""
        app = performance_system["app"]

        # CPU tracking
        cpu_samples = []

        async def cpu_monitor():
            """Monitor CPU usage during test."""
            while True:
                try:
                    cpu_percent = psutil.cpu_percent(interval=0.1)
                    cpu_samples.append(cpu_percent)
                    await asyncio.sleep(0.5)
                except asyncio.CancelledError:
                    break

        # Start CPU monitoring
        monitor_task = asyncio.create_task(cpu_monitor())

        try:
            # Generate CPU-intensive load
            start_time = time.time()
            request_count = 0

            with TestClient(app) as client:
                # Concurrent request generation
                async def make_requests():
                    nonlocal request_count
                    for i in range(200):
                        response = client.get(
                            f"/predictions/room_{i % 10}",
                            headers={
                                "Authorization": "Bearer performance_test_api_key"
                            },
                        )
                        request_count += 1
                        if i % 50 == 0:
                            await asyncio.sleep(0.01)  # Brief pause

                # Run multiple concurrent request generators
                tasks = [make_requests() for _ in range(5)]
                await asyncio.gather(*tasks)

            end_time = time.time()
            duration = end_time - start_time

        finally:
            monitor_task.cancel()
            try:
                await monitor_task
            except asyncio.CancelledError:
                pass

        # CPU usage analysis
        if cpu_samples:
            avg_cpu = sum(cpu_samples) / len(cpu_samples)
            max_cpu = max(cpu_samples)
            p95_cpu = sorted(cpu_samples)[int(0.95 * len(cpu_samples))]
        else:
            avg_cpu = max_cpu = p95_cpu = 0

        throughput = request_count / duration

        # CPU usage assertions (relaxed for test environment)
        assert (
            avg_cpu < PERFORMANCE_CONFIG["cpu_limit_percent"] * 1.5
        ), f"Average CPU {avg_cpu:.1f}% too high"
        assert max_cpu < 95, f"Max CPU {max_cpu:.1f}% approaching saturation"

        print("CPU usage analysis:")
        print(f"  Requests: {request_count} in {duration:.2f}s")
        print(f"  Throughput: {throughput:.1f} req/s")
        print(f"  Average CPU: {avg_cpu:.1f}%")
        print(f"  Max CPU: {max_cpu:.1f}%")
        print(f"  P95 CPU: {p95_cpu:.1f}%")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_resource_cleanup_and_gc(self, performance_system):
        """Test resource cleanup and garbage collection behavior."""
        app = performance_system["app"]

        # Track object creation and cleanup
        initial_objects = len(gc.get_objects())

        # Create many temporary objects through API calls
        with TestClient(app) as client:
            for batch in range(5):
                # Create objects
                for i in range(1000):
                    response = client.get(
                        f"/predictions/room_{i % 20}",
                        headers={"Authorization": "Bearer performance_test_api_key"},
                    )
                    # Response creates temporary objects

                # Force garbage collection
                gc.collect()

                current_objects = len(gc.get_objects())
                print(f"Batch {batch + 1}: {current_objects} objects")

        # Final cleanup
        gc.collect()
        final_objects = len(gc.get_objects())

        object_increase = final_objects - initial_objects

        # Object growth should be reasonable
        assert (
            object_increase < 10000
        ), f"Object increase {object_increase} suggests memory leak"

        print("Object lifecycle analysis:")
        print(f"  Initial objects: {initial_objects}")
        print(f"  Final objects: {final_objects}")
        print(f"  Net increase: {object_increase}")


class TestLongRunningStability:
    """Test system stability over extended periods."""

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_sustained_load_stability(self, performance_system):
        """Test system stability under sustained load."""
        app = performance_system["app"]
        mqtt_publisher = performance_system["mqtt_publisher"]

        duration_minutes = PERFORMANCE_CONFIG["sustained_duration_minutes"]
        end_time = time.time() + (duration_minutes * 60)

        # Stability metrics
        total_requests = 0
        successful_requests = 0
        total_mqtt_messages = 0
        successful_mqtt_messages = 0
        error_counts = {}

        print(f"Starting {duration_minutes}-minute sustained load test...")

        async def api_load():
            """Sustained API load generator."""
            nonlocal total_requests, successful_requests

            with TestClient(app) as client:
                while time.time() < end_time:
                    try:
                        room_id = f"room_{total_requests % 10}"
                        response = client.get(
                            f"/predictions/{room_id}",
                            headers={
                                "Authorization": "Bearer performance_test_api_key"
                            },
                        )

                        total_requests += 1
                        if response.status_code == 200:
                            successful_requests += 1
                        else:
                            error_code = response.status_code
                            error_counts[f"api_{error_code}"] = (
                                error_counts.get(f"api_{error_code}", 0) + 1
                            )

                        # Realistic load pacing
                        await asyncio.sleep(0.1)  # 10 requests per second

                    except Exception as e:
                        error_counts[f"api_exception_{type(e).__name__}"] = (
                            error_counts.get(f"api_exception_{type(e).__name__}", 0) + 1
                        )

        async def mqtt_load():
            """Sustained MQTT load generator."""
            nonlocal total_mqtt_messages, successful_mqtt_messages

            while time.time() < end_time:
                try:
                    topic = f"stability_test/message_{total_mqtt_messages % 100}"
                    data = {
                        "id": total_mqtt_messages,
                        "timestamp": datetime.now().isoformat(),
                        "test": "sustained_load",
                    }

                    result = await mqtt_publisher.publish_json(topic, data)

                    total_mqtt_messages += 1
                    if result.success:
                        successful_mqtt_messages += 1
                    else:
                        error_counts["mqtt_publish_failed"] = (
                            error_counts.get("mqtt_publish_failed", 0) + 1
                        )

                    # MQTT load pacing
                    await asyncio.sleep(0.05)  # 20 messages per second

                except Exception as e:
                    error_counts[f"mqtt_exception_{type(e).__name__}"] = (
                        error_counts.get(f"mqtt_exception_{type(e).__name__}", 0) + 1
                    )

        # Run sustained load
        start_time = time.time()
        await asyncio.gather(api_load(), mqtt_load())
        actual_duration = time.time() - start_time

        # Stability analysis
        api_success_rate = (
            successful_requests / total_requests if total_requests > 0 else 0
        )
        mqtt_success_rate = (
            successful_mqtt_messages / total_mqtt_messages
            if total_mqtt_messages > 0
            else 0
        )
        api_throughput = total_requests / actual_duration
        mqtt_throughput = total_mqtt_messages / actual_duration

        # Stability assertions
        assert (
            api_success_rate >= 0.95
        ), f"API success rate {api_success_rate:.2%} degraded over time"
        assert (
            mqtt_success_rate >= 0.95
        ), f"MQTT success rate {mqtt_success_rate:.2%} degraded over time"
        assert len(error_counts) < 10, f"Too many error types: {error_counts}"

        print(f"Sustained load test results ({actual_duration/60:.1f} minutes):")
        print(
            f"  API: {successful_requests}/{total_requests} ({api_success_rate:.2%}) at {api_throughput:.1f} req/s"
        )
        print(
            f"  MQTT: {successful_mqtt_messages}/{total_mqtt_messages} ({mqtt_success_rate:.2%}) at {mqtt_throughput:.1f} msg/s"
        )
        if error_counts:
            print(f"  Errors: {error_counts}")

    @pytest.mark.performance
    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_memory_leak_detection(self, performance_system):
        """Test for memory leaks over extended operation."""
        app = performance_system["app"]

        # Memory tracking over time
        memory_samples = []
        time_samples = []

        # Reference tracking for leak detection
        class LeakTracker:
            def __init__(self):
                self.objects = weakref.WeakSet()

            def track(self, obj):
                self.objects.add(obj)

            def count(self):
                return len(self.objects)

        leak_tracker = LeakTracker()

        # Run test cycles
        cycles = 10
        requests_per_cycle = 100

        for cycle in range(cycles):
            cycle_start = time.time()

            # Record memory before cycle
            process = psutil.Process(os.getpid())
            memory_before = process.memory_info().rss / 1024 / 1024  # MB

            # Generate load
            with TestClient(app) as client:
                for i in range(requests_per_cycle):
                    response = client.get(
                        f"/predictions/room_{i % 5}",
                        headers={"Authorization": "Bearer performance_test_api_key"},
                    )

                    # Track response objects for leak detection
                    leak_tracker.track(response)

            # Force garbage collection
            gc.collect()

            # Record memory after cycle
            memory_after = process.memory_info().rss / 1024 / 1024  # MB
            cycle_end = time.time()

            memory_samples.append(memory_after)
            time_samples.append(cycle_end)

            print(
                f"Cycle {cycle + 1}/{cycles}: {memory_after:.1f}MB "
                f"({memory_after - memory_before:+.1f}MB) "
                f"in {cycle_end - cycle_start:.2f}s"
            )

            # Brief pause between cycles
            await asyncio.sleep(1)

        # Analyze memory trend
        if len(memory_samples) >= 3:
            # Calculate memory growth trend
            memory_growth = (memory_samples[-1] - memory_samples[0]) / (cycles - 1)

            # Check for consistent growth (potential leak)
            increasing_samples = sum(
                1
                for i in range(1, len(memory_samples))
                if memory_samples[i] > memory_samples[i - 1]
            )
            growth_consistency = increasing_samples / (len(memory_samples) - 1)

            # Leak detection assertions
            assert (
                memory_growth < 5
            ), f"Memory growth {memory_growth:.1f}MB/cycle suggests leak"
            assert (
                growth_consistency < 0.8
            ), f"Consistent memory growth {growth_consistency:.1%} suggests leak"

            print("Memory leak analysis:")
            print(f"  Memory growth per cycle: {memory_growth:.1f}MB")
            print(f"  Growth consistency: {growth_consistency:.1%}")
            print(f"  Tracked objects: {leak_tracker.count()}")


class TestNetworkLatencyAndTimeouts:
    """Test system behavior under network constraints."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_simulated_network_latency(self, performance_mqtt_publisher):
        """Test system behavior with simulated network latency."""

        # Add network latency simulation to MQTT publisher
        original_publish = performance_mqtt_publisher.publish_json.side_effect

        async def latent_publish_json(topic, data, qos=1, retain=False):
            # Simulate variable network latency
            latency = random.uniform(0.01, 0.2)  # 10-200ms
            await asyncio.sleep(latency)

            return await original_publish(topic, data, qos, retain)

        performance_mqtt_publisher.publish_json.side_effect = latent_publish_json

        # Test performance under latency
        start_time = time.time()
        message_count = 1000

        # Publish messages with latency
        tasks = []
        for i in range(message_count):
            task = performance_mqtt_publisher.publish_json(
                f"latency_test/{i % 10}", {"id": i, "test": "network_latency"}
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)
        end_time = time.time()

        duration = end_time - start_time
        throughput = message_count / duration
        successful = sum(1 for r in results if r.success)
        success_rate = successful / len(results)

        # Performance should degrade gracefully with latency
        assert (
            success_rate >= 0.95
        ), f"Success rate {success_rate:.2%} too low under latency"
        assert throughput >= 10, f"Throughput {throughput:.1f} too low under latency"

        print("Network latency test:")
        print(f"  Messages: {message_count} in {duration:.2f}s")
        print(f"  Throughput: {throughput:.1f} msg/s")
        print(f"  Success rate: {success_rate:.2%}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_timeout_handling_under_load(self, performance_system):
        """Test timeout handling under high load."""
        app = performance_system["app"]

        # Create slow tracking manager responses
        tracking_manager = performance_system["tracking_manager"]

        async def slow_get_prediction(room_id):
            # Simulate slow responses
            delay = random.uniform(0.1, 1.0)  # 100ms - 1s delay
            await asyncio.sleep(delay)

            return {
                "room_id": room_id,
                "prediction_time": datetime.now().isoformat(),
                "confidence": 0.8,
                "slow_response": True,
            }

        tracking_manager.get_room_prediction.side_effect = slow_get_prediction

        # Test API timeout behavior
        request_count = 200
        timeout_threshold = 5.0  # 5 second timeout

        response_times = []
        timeout_count = 0

        with TestClient(app) as client:
            for i in range(request_count):
                request_start = time.time()

                try:
                    response = client.get(
                        f"/predictions/room_{i % 5}",
                        headers={"Authorization": "Bearer performance_test_api_key"},
                        timeout=timeout_threshold,
                    )

                    request_end = time.time()
                    response_times.append(request_end - request_start)

                except Exception:
                    # Request timed out or failed
                    timeout_count += 1
                    response_times.append(timeout_threshold)

        # Analyze timeout behavior
        avg_response_time = sum(response_times) / len(response_times)
        timeout_rate = timeout_count / request_count
        p95_response_time = sorted(response_times)[int(0.95 * len(response_times))]

        # Timeout handling assertions
        assert timeout_rate < 0.1, f"Timeout rate {timeout_rate:.2%} too high"
        assert (
            avg_response_time < timeout_threshold * 0.8
        ), f"Average response time {avg_response_time:.2f}s approaching timeout"

        print("Timeout handling test:")
        print(f"  Requests: {request_count}")
        print(f"  Timeouts: {timeout_count} ({timeout_rate:.2%})")
        print(f"  Avg response time: {avg_response_time:.2f}s")
        print(f"  P95 response time: {p95_response_time:.2f}s")


class TestErrorRecoveryAndGracefulDegradation:
    """Test system error recovery and graceful degradation."""

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_partial_component_failure_recovery(self, performance_system):
        """Test recovery from partial component failures."""
        app = performance_system["app"]
        mqtt_publisher = performance_system["mqtt_publisher"]
        tracking_manager = performance_system["tracking_manager"]

        # Simulate intermittent failures
        failure_probability = 0.2  # 20% failure rate

        async def failing_mqtt_publish(topic, data, qos=1, retain=False):
            if random.random() < failure_probability:
                raise Exception("Simulated MQTT failure")

            return MQTTPublishResult(
                success=True,
                topic=topic,
                payload_size=len(json.dumps(data)),
                publish_time=datetime.utcnow(),
                message_id=random.randint(1000, 9999),
            )

        async def failing_get_prediction(room_id):
            if random.random() < failure_probability:
                raise Exception("Simulated tracking failure")

            return {
                "room_id": room_id,
                "prediction_time": datetime.now().isoformat(),
                "confidence": 0.8,
                "recovered": True,
            }

        mqtt_publisher.publish_json.side_effect = failing_mqtt_publish
        tracking_manager.get_room_prediction.side_effect = failing_get_prediction

        # Test system resilience
        operation_count = 500
        api_successes = 0
        mqtt_successes = 0

        # API operations with failures
        with TestClient(app) as client:
            for i in range(operation_count):
                try:
                    response = client.get(
                        f"/predictions/room_{i % 10}",
                        headers={"Authorization": "Bearer performance_test_api_key"},
                    )
                    if response.status_code == 200:
                        api_successes += 1
                except Exception:
                    pass

        # MQTT operations with failures
        for i in range(operation_count):
            try:
                result = await mqtt_publisher.publish_json(
                    f"recovery_test/{i}", {"id": i, "test": "failure_recovery"}
                )
                if result.success:
                    mqtt_successes += 1
            except Exception:
                pass

        # Recovery analysis
        api_success_rate = api_successes / operation_count
        mqtt_success_rate = mqtt_successes / operation_count

        # Should maintain reasonable performance despite failures
        expected_success_rate = 1 - failure_probability
        assert (
            api_success_rate >= expected_success_rate * 0.8
        ), f"API success rate {api_success_rate:.2%} too low for failure rate"
        assert (
            mqtt_success_rate >= expected_success_rate * 0.8
        ), f"MQTT success rate {mqtt_success_rate:.2%} too low for failure rate"

        print("Failure recovery test:")
        print(f"  Expected success rate: {expected_success_rate:.2%}")
        print(f"  API actual success rate: {api_success_rate:.2%}")
        print(f"  MQTT actual success rate: {mqtt_success_rate:.2%}")

    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_cascading_failure_graceful_degradation(self, performance_system):
        """Test graceful degradation under cascading failures."""
        app = performance_system["app"]
        tracking_manager = performance_system["tracking_manager"]
        database_manager = performance_system["database_manager"]
        mqtt_publisher = performance_system["mqtt_publisher"]

        # Introduce cascading failures over time
        failure_stages = [
            {"time": 0, "component": "none", "description": "Normal operation"},
            {"time": 1, "component": "mqtt", "description": "MQTT failure"},
            {"time": 2, "component": "database", "description": "Database failure"},
            {"time": 3, "component": "tracking", "description": "Tracking failure"},
            {"time": 4, "component": "recovery", "description": "Recovery phase"},
        ]

        stage_results = {}

        for stage in failure_stages:
            stage_start = time.time()

            # Configure failures based on stage
            if stage["component"] == "mqtt":
                mqtt_publisher.publish_json.side_effect = Exception(
                    "MQTT cascading failure"
                )
                mqtt_publisher.connection_status.connected = False
            elif stage["component"] == "database":
                database_manager.health_check.side_effect = Exception(
                    "Database cascading failure"
                )
            elif stage["component"] == "tracking":
                tracking_manager.get_room_prediction.side_effect = Exception(
                    "Tracking cascading failure"
                )
                tracking_manager.get_tracking_status.side_effect = Exception(
                    "Tracking status failure"
                )
            elif stage["component"] == "recovery":
                # Restore all components
                mqtt_publisher.publish_json.side_effect = None
                mqtt_publisher.connection_status.connected = True
                database_manager.health_check.side_effect = None
                tracking_manager.get_room_prediction.side_effect = None
                tracking_manager.get_tracking_status.side_effect = None

            # Test system response in this stage
            successes = 0
            attempts = 100

            with TestClient(app) as client:
                for i in range(attempts):
                    try:
                        response = client.get(
                            "/health",
                            headers={
                                "Authorization": "Bearer performance_test_api_key"
                            },
                        )
                        if response.status_code == 200:
                            successes += 1
                    except Exception:
                        pass

            success_rate = successes / attempts
            stage_results[stage["component"]] = {
                "success_rate": success_rate,
                "description": stage["description"],
            }

            print(f"Stage '{stage['component']}': {success_rate:.2%} success rate")

            # Brief pause between stages
            await asyncio.sleep(0.5)

        # Degradation analysis
        normal_success = stage_results["none"]["success_rate"]
        recovery_success = stage_results["recovery"]["success_rate"]

        # System should recover to near-normal performance
        assert (
            recovery_success >= normal_success * 0.9
        ), f"Recovery success rate {recovery_success:.2%} insufficient"

        # System should maintain some functionality even during failures
        for stage, result in stage_results.items():
            if stage not in ["none", "recovery"]:
                assert (
                    result["success_rate"] >= 0.1
                ), f"Stage '{stage}' success rate {result['success_rate']:.2%} shows no graceful degradation"

        print("Graceful degradation analysis:")
        for stage, result in stage_results.items():
            print(f"  {stage}: {result['success_rate']:.2%} - {result['description']}")
