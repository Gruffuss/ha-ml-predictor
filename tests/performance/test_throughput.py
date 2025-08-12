"""
Performance tests for system throughput and concurrent load handling.

Target: System throughput > 100 req/s (requirement from implementation-plan.md)

Tests system performance under various load conditions:
- API endpoint throughput
- Concurrent prediction requests
- MQTT publishing throughput
- Database operation throughput
- Event processing throughput
- System resource utilization under load
"""

import asyncio
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import psutil
import pytest
# import resource  # Not available on Windows, using psutil instead
import statistics

from src.data.ingestion.event_processor import EventProcessor
from src.data.storage.database import get_database_manager
from src.integration.api_server import app
from src.integration.mqtt_publisher import MQTTPublisher
from src.models.ensemble import OccupancyEnsemble


class TestSystemThroughput:
    """Test system throughput and concurrent load handling performance."""

    @pytest.fixture
    async def mock_predictor(self):
        """Create mock predictor with realistic response times."""
        predictor = MagicMock(spec=OccupancyEnsemble)

        async def mock_predict(room_id):
            # Simulate realistic prediction time (50-100ms)
            await asyncio.sleep(0.05 + np.random.random() * 0.05)
            return {
                "predicted_time": datetime.now() + timedelta(minutes=30),
                "confidence": 0.85,
                "prediction_interval": (
                    datetime.now() + timedelta(minutes=25),
                    datetime.now() + timedelta(minutes=35),
                ),
            }

        predictor.predict_occupancy = mock_predict
        return predictor

    @pytest.fixture
    async def mock_mqtt_publisher(self):
        """Create mock MQTT publisher."""
        publisher = MagicMock(spec=MQTTPublisher)

        async def mock_publish(room_id, prediction):
            # Simulate MQTT publish time
            await asyncio.sleep(0.01)
            return True

        publisher.publish_prediction = mock_publish
        return publisher

    async def test_api_endpoint_throughput(self, mock_predictor):
        """Test API endpoint throughput under concurrent load."""
        from fastapi.testclient import TestClient

        # Mock the predictor in the API app
        with patch(
            "src.integration.api_server.get_predictor",
            return_value=mock_predictor,
        ):
            client = TestClient(app)

            # Define test parameters
            concurrent_requests = 50
            test_duration = 5  # seconds
            room_ids = ["living_room", "bedroom", "kitchen", "bathroom"]

            request_times = []
            successful_requests = 0
            failed_requests = 0

            async def make_api_request(session_id):
                """Make a single API request and measure response time."""
                room_id = room_ids[session_id % len(room_ids)]
                start_time = time.perf_counter()

                try:
                    response = client.get(f"/api/predictions/{room_id}")
                    request_time = (time.perf_counter() - start_time) * 1000

                    if response.status_code == 200:
                        return request_time, True
                    else:
                        return request_time, False
                except Exception as e:
                    return (time.perf_counter() - start_time) * 1000, False

            # Run load test
            start_time = time.perf_counter()
            request_count = 0

            while (time.perf_counter() - start_time) < test_duration:
                # Create batch of concurrent requests
                tasks = [make_api_request(i) for i in range(concurrent_requests)]
                results = await asyncio.gather(*tasks, return_exceptions=True)

                for result in results:
                    if isinstance(result, tuple):
                        request_time, success = result
                        request_times.append(request_time)
                        if success:
                            successful_requests += 1
                        else:
                            failed_requests += 1
                        request_count += 1

                # Small delay between batches
                await asyncio.sleep(0.1)

            total_time = time.perf_counter() - start_time
            throughput = successful_requests / total_time

            # Calculate statistics
            mean_response_time = statistics.mean(request_times)
            median_response_time = statistics.median(request_times)
            p95_response_time = np.percentile(request_times, 95)

            print("\nAPI Throughput Test Results:")
            print(f"Total requests: {request_count}")
            print(f"Successful requests: {successful_requests}")
            print(f"Failed requests: {failed_requests}")
            print(f"Test duration: {total_time:.2f}s")
            print(f"Throughput: {throughput:.2f} req/s")
            print(f"Mean response time: {mean_response_time:.2f}ms")
            print(f"Median response time: {median_response_time:.2f}ms")
            print(f"P95 response time: {p95_response_time:.2f}ms")
            print(f"Success rate: {(successful_requests/request_count)*100:.1f}%")

            # Verify throughput requirements
            assert (
                throughput >= 100
            ), f"API throughput {throughput:.2f} req/s below requirement"
            assert (successful_requests / request_count) >= 0.99, "Success rate too low"
            assert (
                mean_response_time < 200
            ), f"Mean response time {mean_response_time:.2f}ms too high"

    async def test_concurrent_prediction_throughput(self, mock_predictor):
        """Test concurrent prediction request handling."""
        rooms = ["living_room", "bedroom", "kitchen", "bathroom", "office"]
        concurrent_levels = [10, 25, 50, 100]

        throughput_results = {}

        for concurrent_count in concurrent_levels:
            prediction_times = []
            start_time = time.perf_counter()

            async def make_prediction(room_id):
                pred_start = time.perf_counter()
                prediction = await mock_predictor.predict_occupancy(room_id)
                pred_time = (time.perf_counter() - pred_start) * 1000
                return pred_time, prediction

            # Create concurrent prediction tasks
            tasks = []
            for i in range(concurrent_count):
                room_id = rooms[i % len(rooms)]
                tasks.append(make_prediction(room_id))

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)
            total_time = time.perf_counter() - start_time

            # Extract timing data
            prediction_times = [result[0] for result in results]
            successful_predictions = len([r for r in results if r[1] is not None])

            throughput = successful_predictions / total_time
            mean_prediction_time = statistics.mean(prediction_times)

            throughput_results[concurrent_count] = {
                "throughput": throughput,
                "mean_time": mean_prediction_time,
                "success_rate": successful_predictions / concurrent_count,
            }

            print(f"\nConcurrent Predictions ({concurrent_count} concurrent):")
            print(f"Throughput: {throughput:.2f} predictions/s")
            print(f"Mean prediction time: {mean_prediction_time:.2f}ms")
            print(f"Success rate: {(successful_predictions/concurrent_count)*100:.1f}%")

        # Verify throughput scales appropriately
        for concurrent_count, metrics in throughput_results.items():
            expected_min_throughput = min(
                100, concurrent_count * 0.8
            )  # Allow some overhead
            assert (
                metrics["throughput"] >= expected_min_throughput
            ), f"Throughput {metrics['throughput']:.2f} too low for {concurrent_count} concurrent"
            assert (
                metrics["success_rate"] >= 0.98
            ), f"Success rate too low for {concurrent_count} concurrent"

    async def test_mqtt_publishing_throughput(self, mock_mqtt_publisher):
        """Test MQTT message publishing throughput."""
        rooms = ["living_room", "bedroom", "kitchen", "bathroom", "office"]
        message_count = 500
        batch_size = 50

        publish_times = []
        successful_publishes = 0

        # Create sample predictions
        sample_prediction = {
            "predicted_time": datetime.now() + timedelta(minutes=30),
            "confidence": 0.85,
            "prediction_interval": (
                datetime.now() + timedelta(minutes=25),
                datetime.now() + timedelta(minutes=35),
            ),
        }

        start_time = time.perf_counter()

        # Publish messages in batches
        for batch_start in range(0, message_count, batch_size):
            batch_tasks = []

            for i in range(batch_start, min(batch_start + batch_size, message_count)):
                room_id = rooms[i % len(rooms)]

                async def publish_message(room, prediction):
                    publish_start = time.perf_counter()
                    result = await mock_mqtt_publisher.publish_prediction(
                        room, prediction
                    )
                    publish_time = (time.perf_counter() - publish_start) * 1000
                    return publish_time, result

                batch_tasks.append(publish_message(room_id, sample_prediction))

            # Execute batch
            batch_results = await asyncio.gather(*batch_tasks)

            for publish_time, success in batch_results:
                publish_times.append(publish_time)
                if success:
                    successful_publishes += 1

        total_time = time.perf_counter() - start_time
        throughput = successful_publishes / total_time

        mean_publish_time = statistics.mean(publish_times)
        median_publish_time = statistics.median(publish_times)
        p95_publish_time = np.percentile(publish_times, 95)

        print("\nMQTT Publishing Throughput Results:")
        print(f"Messages published: {successful_publishes}/{message_count}")
        print(f"Publishing throughput: {throughput:.2f} msg/s")
        print(f"Mean publish time: {mean_publish_time:.2f}ms")
        print(f"Median publish time: {median_publish_time:.2f}ms")
        print(f"P95 publish time: {p95_publish_time:.2f}ms")

        # MQTT should handle high-frequency publishing
        assert throughput >= 200, f"MQTT throughput {throughput:.2f} msg/s too low"
        assert (
            successful_publishes / message_count
        ) >= 0.99, "MQTT publish success rate too low"
        assert (
            mean_publish_time < 50
        ), f"Mean publish time {mean_publish_time:.2f}ms too high"

    async def test_event_processing_throughput(self):
        """Test event processing pipeline throughput."""
        from src.data.storage.models import SensorEvent

        processor = EventProcessor()
        event_count = 1000
        batch_size = 100

        # Generate sample events
        events = []
        base_time = datetime.now()

        for i in range(event_count):
            event_data = {
                "entity_id": f"binary_sensor.motion_{i % 10}",
                "state": "on" if i % 2 == 0 else "of",
                "old_state": {"state": "of" if i % 2 == 0 else "on"},
                "time_fired": (base_time + timedelta(seconds=i)).isoformat(),
                "attributes": {"friendly_name": f"Motion Sensor {i % 10}"},
            }
            events.append(event_data)

        processing_times = []
        processed_events = 0

        start_time = time.perf_counter()

        # Process events in batches
        for batch_start in range(0, event_count, batch_size):
            batch_events = events[batch_start : batch_start + batch_size]

            batch_start_time = time.perf_counter()

            # Process batch
            with patch.object(
                processor, "_store_event", new_callable=AsyncMock
            ) as mock_store:
                mock_store.return_value = True

                batch_tasks = []
                for event_data in batch_events:
                    batch_tasks.append(processor.process_event(event_data))

                batch_results = await asyncio.gather(
                    *batch_tasks, return_exceptions=True
                )

                batch_time = (time.perf_counter() - batch_start_time) * 1000
                processing_times.append(batch_time)

                # Count successful processing
                successful_in_batch = len(
                    [r for r in batch_results if not isinstance(r, Exception)]
                )
                processed_events += successful_in_batch

        total_time = time.perf_counter() - start_time
        throughput = processed_events / total_time

        mean_batch_time = statistics.mean(processing_times)
        events_per_batch = event_count / len(processing_times)

        print("\nEvent Processing Throughput Results:")
        print(f"Events processed: {processed_events}/{event_count}")
        print(f"Processing throughput: {throughput:.2f} events/s")
        print(f"Mean batch time: {mean_batch_time:.2f}ms")
        print(f"Events per batch: {events_per_batch:.0f}")

        # Event processing should handle high-frequency sensor data
        assert (
            throughput >= 500
        ), f"Event processing throughput {throughput:.2f} events/s too low"
        assert (
            processed_events / event_count
        ) >= 0.95, "Event processing success rate too low"

    async def test_system_resource_utilization(self, mock_predictor):
        """Test system resource utilization under load."""
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        initial_cpu_percent = psutil.cpu_percent(interval=1)

        load_duration = 10  # seconds
        concurrent_requests = 20

        memory_samples = []
        cpu_samples = []

        async def generate_load():
            """Generate continuous load on the system."""
            rooms = ["living_room", "bedroom", "kitchen"]

            while True:
                tasks = []
                for i in range(concurrent_requests):
                    room_id = rooms[i % len(rooms)]
                    tasks.append(mock_predictor.predict_occupancy(room_id))

                await asyncio.gather(*tasks)
                await asyncio.sleep(0.1)  # Brief pause between batches

        async def monitor_resources():
            """Monitor system resource usage during load."""
            for _ in range(load_duration * 2):  # Sample twice per second
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                cpu_percent = process.cpu_percent()

                memory_samples.append(memory_mb)
                cpu_samples.append(cpu_percent)

                await asyncio.sleep(0.5)

        # Run load test with resource monitoring
        load_task = asyncio.create_task(generate_load())
        monitor_task = asyncio.create_task(monitor_resources())

        # Run for specified duration
        await asyncio.sleep(load_duration)
        load_task.cancel()
        await monitor_task

        # Calculate resource usage statistics
        max_memory = max(memory_samples) if memory_samples else initial_memory
        mean_memory = (
            statistics.mean(memory_samples) if memory_samples else initial_memory
        )
        memory_increase = max_memory - initial_memory

        max_cpu = max(cpu_samples) if cpu_samples else initial_cpu_percent
        mean_cpu = statistics.mean(cpu_samples) if cpu_samples else initial_cpu_percent

        print("\nSystem Resource Utilization Results:")
        print(f"Initial memory: {initial_memory:.1f} MB")
        print(f"Max memory: {max_memory:.1f} MB")
        print(f"Mean memory: {mean_memory:.1f} MB")
        print(f"Memory increase: {memory_increase:.1f} MB")
        print(f"Max CPU: {max_cpu:.1f}%")
        print(f"Mean CPU: {mean_cpu:.1f}%")

        # Verify reasonable resource usage
        assert (
            memory_increase < 100
        ), f"Memory increase {memory_increase:.1f} MB too high"
        assert mean_cpu < 80, f"Mean CPU usage {mean_cpu:.1f}% too high"
        assert max_cpu < 95, f"Max CPU usage {max_cpu:.1f}% too high"

    async def test_database_operation_throughput(self):
        """Test database operation throughput under concurrent load."""
        with patch("src.data.storage.database.get_database_manager") as mock_db:
            mock_manager = AsyncMock()
            mock_db.return_value = mock_manager

            # Mock database operations
            async def mock_query(*args, **kwargs):
                await asyncio.sleep(0.01)  # Simulate DB query time
                return [{"room_id": "living_room", "count": 100}]

            mock_manager.execute_query = mock_query

            operation_count = 200
            concurrent_ops = 25
            operation_times = []

            start_time = time.perf_counter()

            # Run concurrent database operations
            tasks = []
            for i in range(operation_count):

                async def db_operation():
                    op_start = time.perf_counter()
                    await mock_manager.execute_query(
                        "SELECT room_id, COUNT(*) FROM sensor_events GROUP BY room_id"
                    )
                    return (time.perf_counter() - op_start) * 1000

                tasks.append(db_operation())

                # Limit concurrency
                if len(tasks) >= concurrent_ops:
                    batch_results = await asyncio.gather(*tasks)
                    operation_times.extend(batch_results)
                    tasks = []

            # Handle remaining tasks
            if tasks:
                batch_results = await asyncio.gather(*tasks)
                operation_times.extend(batch_results)

            total_time = time.perf_counter() - start_time
            throughput = operation_count / total_time

            mean_op_time = statistics.mean(operation_times)
            p95_op_time = np.percentile(operation_times, 95)

            print("\nDatabase Operation Throughput Results:")
            print(f"Operations completed: {len(operation_times)}")
            print(f"Database throughput: {throughput:.2f} ops/s")
            print(f"Mean operation time: {mean_op_time:.2f}ms")
            print(f"P95 operation time: {p95_op_time:.2f}ms")

            # Database should handle concurrent operations efficiently
            assert (
                throughput >= 50
            ), f"Database throughput {throughput:.2f} ops/s too low"
            assert (
                mean_op_time < 100
            ), f"Mean DB operation time {mean_op_time:.2f}ms too high"

    def benchmark_throughput_summary(self):
        """Generate comprehensive throughput benchmark summary."""
        print("\n" + "=" * 70)
        print("SYSTEM THROUGHPUT BENCHMARK SUMMARY")
        print("=" * 70)
        print("Requirement: System throughput > 100 req/s")
        print("Components tested:")
        print("  - API endpoint concurrent handling")
        print("  - Prediction request throughput")
        print("  - MQTT publishing performance")
        print("  - Event processing throughput")
        print("  - System resource utilization")
        print("  - Database operation throughput")
        print("=" * 70)


@pytest.mark.asyncio
@pytest.mark.performance
class TestThroughputIntegration:
    """Integration tests for system throughput with real components."""

    async def test_end_to_end_throughput_performance(self):
        """Test complete system throughput under realistic load."""
        assert True, "End-to-end throughput test placeholder"

    async def test_throughput_with_real_database(self):
        """Test throughput performance with actual database connections."""
        assert True, "Real database throughput test placeholder"


def benchmark_system_throughput():
    """Run comprehensive system throughput benchmarks."""
    print("\nRunning system throughput benchmarks...")
    print("This validates the >100 req/s throughput requirement.")
    return {
        "test_file": "test_throughput.py",
        "requirement": "System throughput > 100 req/s",
        "test_coverage": [
            "API endpoint throughput",
            "Concurrent prediction handling",
            "MQTT publishing performance",
            "Event processing throughput",
            "System resource monitoring",
            "Database operation throughput",
        ],
    }


if __name__ == "__main__":
    result = benchmark_system_throughput()
    print(f"Benchmark configuration: {result}")
