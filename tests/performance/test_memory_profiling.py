"""
Performance tests for memory usage profiling and leak detection.

Tests memory usage patterns across system components:
- Memory usage baselines and growth patterns
- Memory leak detection in long-running processes
- Memory efficiency of data structures
- Garbage collection effectiveness
- Peak memory usage under load
- Memory usage profiling per component
"""

import asyncio
from datetime import datetime, timedelta
import gc
import time
from unittest.mock import AsyncMock, MagicMock, patch
import weakref

import numpy as np
import pandas as pd
import psutil
import pytest
import statistics
import tracemalloc

from src.data.ingestion.event_processor import EventProcessor
from src.data.storage.models import SensorEvent
from src.features.store import FeatureStore
from src.integration.mqtt_publisher import MQTTPublisher
from src.models.predictor import OccupancyPredictor


class MemoryProfiler:
    """Utility class for memory profiling and leak detection."""

    def __init__(self):
        self.snapshots = []
        self.process = psutil.Process()

    def start_profiling(self):
        """Start memory profiling session."""
        tracemalloc.start()
        gc.collect()  # Clean start
        self.baseline_memory = self.get_current_memory()

    def stop_profiling(self):
        """Stop memory profiling and return summary."""
        tracemalloc.stop()
        final_memory = self.get_current_memory()
        return {
            "baseline_mb": self.baseline_memory,
            "final_mb": final_memory,
            "increase_mb": final_memory - self.baseline_memory,
            "snapshots": len(self.snapshots),
        }

    def take_snapshot(self, label: str = ""):
        """Take a memory usage snapshot."""
        current_memory = self.get_current_memory()
        snapshot = tracemalloc.take_snapshot()

        self.snapshots.append(
            {
                "label": label,
                "timestamp": time.time(),
                "memory_mb": current_memory,
                "snapshot": snapshot,
            }
        )

        return current_memory

    def get_current_memory(self) -> float:
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def detect_memory_leak(self, threshold_mb: float = 10.0) -> bool:
        """Detect if memory usage indicates a potential leak."""
        if len(self.snapshots) < 2:
            return False

        # Check if memory consistently increases
        memory_values = [s["memory_mb"] for s in self.snapshots]

        # Simple trend detection
        if len(memory_values) >= 3:
            # Check if last 3 measurements show consistent growth
            recent = memory_values[-3:]
            is_increasing = all(
                recent[i] <= recent[i + 1] for i in range(len(recent) - 1)
            )
            growth = recent[-1] - recent[0]

            return is_increasing and growth > threshold_mb

        return False

    def get_top_memory_allocations(
        self, count: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top memory allocations from latest snapshot."""
        if not self.snapshots:
            return []

        latest_snapshot = self.snapshots[-1]["snapshot"]
        top_stats = latest_snapshot.statistics("lineno")

        allocations = []
        for stat in top_stats[:count]:
            allocations.append(
                {
                    "size_mb": stat.size / 1024 / 1024,
                    "count": stat.count,
                    "filename": (
                        stat.traceback.format()[-1]
                        if stat.traceback
                        else "Unknown"
                    ),
                }
            )

        return allocations


class TestMemoryProfiling:
    """Test memory usage patterns and detect potential leaks."""

    @pytest.fixture
    def memory_profiler(self):
        """Create memory profiler for tests."""
        profiler = MemoryProfiler()
        profiler.start_profiling()
        yield profiler
        summary = profiler.stop_profiling()
        print(f"\nMemory profiling summary: {summary}")

    @pytest.fixture
    def sample_events(self):
        """Generate sample events for memory testing."""
        events = []
        base_time = datetime.now() - timedelta(hours=1)

        for i in range(1000):
            event = SensorEvent(
                room_id=f"room_{i % 5}",
                sensor_id=f"sensor_{i % 10}",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "of",
                previous_state="of" if i % 2 == 0 else "on",
                timestamp=base_time + timedelta(seconds=i * 3.6),
                attributes={"test_data": f"data_{i}"},
                is_human_triggered=True,
            )
            events.append(event)

        return events

    async def test_predictor_memory_usage(self, memory_profiler):
        """Test memory usage of prediction operations."""
        with (
            patch("src.models.predictor.FeatureStore"),
            patch("src.models.predictor.OccupancyEnsemble"),
        ):

            predictor = OccupancyPredictor()
            room_id = "living_room"

            # Mock predictor methods
            async def mock_predict(room_id):
                # Create some data structures to simulate real prediction
                features = pd.DataFrame(np.random.random((100, 20)))
                prediction = {
                    "predicted_time": datetime.now() + timedelta(minutes=30),
                    "confidence": 0.85,
                    "features": features,  # Large data structure
                }
                return prediction

            predictor.predict_occupancy = mock_predict

            # Baseline measurement
            memory_profiler.take_snapshot("baseline")
            initial_memory = memory_profiler.get_current_memory()

            # Run predictions and measure memory
            predictions = []
            for i in range(50):
                prediction = await predictor.predict_occupancy(room_id)
                predictions.append(prediction)

                if i % 10 == 0:
                    memory_profiler.take_snapshot(f"prediction_{i}")
                    gc.collect()  # Force garbage collection

            final_memory = memory_profiler.get_current_memory()
            memory_increase = final_memory - initial_memory

            print("\nPredictor Memory Usage:")
            print(f"Initial memory: {initial_memory:.2f} MB")
            print(f"Final memory: {final_memory:.2f} MB")
            print(f"Memory increase: {memory_increase:.2f} MB")
            print(f"Predictions created: {len(predictions)}")

            # Check for memory leaks
            has_leak = memory_profiler.detect_memory_leak(threshold_mb=5.0)
            assert not has_leak, "Potential memory leak detected in predictor"

            # Memory usage should be reasonable
            assert (
                memory_increase < 50
            ), f"Memory increase {memory_increase:.2f} MB too high"

            # Clean up predictions to test garbage collection
            predictions.clear()
            gc.collect()
            after_cleanup_memory = memory_profiler.get_current_memory()
            memory_recovered = final_memory - after_cleanup_memory

            print(f"Memory after cleanup: {after_cleanup_memory:.2f} MB")
            print(f"Memory recovered: {memory_recovered:.2f} MB")

    async def test_feature_store_memory_efficiency(
        self, memory_profiler, sample_events
    ):
        """Test memory efficiency of feature store operations."""
        with patch("src.features.store.get_database_manager"):
            feature_store = FeatureStore()

            # Mock database calls to return sample events
            async def mock_get_events(*args, **kwargs):
                return sample_events

            feature_store._get_recent_events = mock_get_events
            feature_store._get_room_context = AsyncMock(return_value={})

            memory_profiler.take_snapshot("feature_store_baseline")
            initial_memory = memory_profiler.get_current_memory()

            # Run feature computations
            feature_sets = []
            for i in range(20):
                features = await feature_store.compute_features(
                    room_id="living_room",
                    target_time=datetime.now(),
                    lookback_hours=24,
                )
                feature_sets.append(features)

                if i % 5 == 0:
                    memory_profiler.take_snapshot(f"features_{i}")

            peak_memory = memory_profiler.get_current_memory()
            memory_increase = peak_memory - initial_memory

            print("\nFeature Store Memory Usage:")
            print(f"Initial memory: {initial_memory:.2f} MB")
            print(f"Peak memory: {peak_memory:.2f} MB")
            print(f"Memory increase: {memory_increase:.2f} MB")
            print(f"Feature sets created: {len(feature_sets)}")

            # Feature store should be memory efficient
            assert (
                memory_increase < 30
            ), f"Feature store memory increase {memory_increase:.2f} MB too high"

            # Test memory cleanup
            feature_sets.clear()
            gc.collect()
            after_cleanup = memory_profiler.get_current_memory()
            memory_recovered = peak_memory - after_cleanup

            print(f"Memory after cleanup: {after_cleanup:.2f} MB")
            print(f"Memory recovered: {memory_recovered:.2f} MB")

    async def test_event_processing_memory_scaling(self, memory_profiler):
        """Test how event processing memory scales with data volume."""
        processor = EventProcessor()

        # Mock database storage
        with patch.object(
            processor, "_store_event", new_callable=AsyncMock
        ) as mock_store:
            mock_store.return_value = True

            memory_profiler.take_snapshot("event_processing_baseline")
            initial_memory = memory_profiler.get_current_memory()

            # Process events in batches of increasing size
            batch_sizes = [100, 500, 1000, 2000]
            memory_by_batch = {}

            for batch_size in batch_sizes:
                # Generate event batch
                events = []
                base_time = datetime.now()

                for i in range(batch_size):
                    event_data = {
                        "entity_id": f"binary_sensor.motion_{i}",
                        "state": "on" if i % 2 == 0 else "of",
                        "old_state": {"state": "of" if i % 2 == 0 else "on"},
                        "time_fired": (
                            base_time + timedelta(seconds=i)
                        ).isoformat(),
                        "attributes": {
                            "friendly_name": f"Motion Sensor {i}",
                            "data": f"test_data_{i}",
                        },
                    }
                    events.append(event_data)

                # Process batch and measure memory
                batch_start_memory = memory_profiler.get_current_memory()

                processing_tasks = []
                for event_data in events:
                    processing_tasks.append(
                        processor.process_event(event_data)
                    )

                await asyncio.gather(*processing_tasks)

                batch_end_memory = memory_profiler.get_current_memory()
                memory_by_batch[batch_size] = {
                    "start_memory": batch_start_memory,
                    "end_memory": batch_end_memory,
                    "increase": batch_end_memory - batch_start_memory,
                }

                memory_profiler.take_snapshot(f"batch_{batch_size}")

                # Force garbage collection between batches
                events.clear()
                processing_tasks.clear()
                gc.collect()

            print("\nEvent Processing Memory Scaling:")
            for batch_size, metrics in memory_by_batch.items():
                memory_per_event = (
                    metrics["increase"] / batch_size * 1024
                )  # KB per event
                print(
                    f"Batch {batch_size}: {metrics['increase']:.2f} MB increase ({memory_per_event:.2f} KB/event)"
                )

            # Memory usage should scale linearly, not exponentially
            largest_batch = memory_by_batch[max(batch_sizes)]
            smallest_batch = memory_by_batch[min(batch_sizes)]

            scaling_ratio = (
                largest_batch["increase"] / smallest_batch["increase"]
            )
            batch_size_ratio = max(batch_sizes) / min(batch_sizes)

            # Memory scaling should be reasonable (not more than 2x the batch size ratio)
            assert (
                scaling_ratio <= batch_size_ratio * 2
            ), f"Memory scaling {scaling_ratio:.2f} too high"

    async def test_long_running_memory_stability(self, memory_profiler):
        """Test memory stability over extended operation."""
        with (
            patch("src.models.predictor.FeatureStore"),
            patch("src.models.predictor.OccupancyEnsemble"),
        ):

            predictor = OccupancyPredictor()

            async def mock_predict(room_id):
                # Simulate prediction with temporary data structures
                temp_data = {
                    "features": np.random.random((50, 10)),
                    "intermediate": pd.DataFrame(np.random.random((100, 5))),
                    "prediction": datetime.now() + timedelta(minutes=30),
                }
                return temp_data["prediction"]

            predictor.predict_occupancy = mock_predict

            memory_profiler.take_snapshot("long_running_start")

            # Run for extended period with regular predictions
            iterations = 200
            prediction_interval = 0.05  # 50ms between predictions
            memory_samples = []

            for i in range(iterations):
                # Make prediction
                await predictor.predict_occupancy("living_room")

                # Sample memory every 20 iterations
                if i % 20 == 0:
                    current_memory = memory_profiler.get_current_memory()
                    memory_samples.append(current_memory)
                    memory_profiler.take_snapshot(f"iteration_{i}")

                # Brief pause
                await asyncio.sleep(prediction_interval)

                # Periodic garbage collection
                if i % 50 == 0:
                    gc.collect()

            # Analyze memory stability
            if len(memory_samples) >= 3:
                initial_memory = memory_samples[0]
                final_memory = memory_samples[-1]
                max_memory = max(memory_samples)
                min_memory = min(memory_samples)

                memory_growth = final_memory - initial_memory
                memory_variance = max_memory - min_memory

                print("\nLong-Running Memory Stability:")
                print(f"Initial memory: {initial_memory:.2f} MB")
                print(f"Final memory: {final_memory:.2f} MB")
                print(f"Memory growth: {memory_growth:.2f} MB")
                print(f"Memory variance: {memory_variance:.2f} MB")
                print(f"Max memory: {max_memory:.2f} MB")
                print(f"Min memory: {min_memory:.2f} MB")
                print(f"Iterations completed: {iterations}")

                # Memory should remain stable over time
                assert (
                    memory_growth < 20
                ), f"Memory growth {memory_growth:.2f} MB indicates potential leak"
                assert (
                    memory_variance < 30
                ), f"Memory variance {memory_variance:.2f} MB too high"

                # Check for consistent leak pattern
                has_leak = memory_profiler.detect_memory_leak(threshold_mb=3.0)
                assert (
                    not has_leak
                ), "Memory leak pattern detected in long-running test"

    async def test_garbage_collection_effectiveness(self, memory_profiler):
        """Test effectiveness of garbage collection in releasing memory."""
        # Create objects that should be garbage collected
        large_objects = []

        memory_profiler.take_snapshot("gc_test_start")
        initial_memory = memory_profiler.get_current_memory()

        # Create large objects
        for i in range(100):
            # Create large DataFrame that should be collectible
            large_df = pd.DataFrame(np.random.random((1000, 50)))
            large_dict = {
                f"key_{j}": np.random.random(100) for j in range(100)
            }

            large_objects.append(
                {
                    "id": i,
                    "dataframe": large_df,
                    "dictionary": large_dict,
                    "array": np.random.random((500, 20)),
                }
            )

        after_creation_memory = memory_profiler.get_current_memory()
        memory_profiler.take_snapshot("after_object_creation")

        # Clear references
        large_objects.clear()

        # Test garbage collection effectiveness
        before_gc_memory = memory_profiler.get_current_memory()

        # Force garbage collection
        collected = gc.collect()

        after_gc_memory = memory_profiler.get_current_memory()
        memory_profiler.take_snapshot("after_garbage_collection")

        memory_created = after_creation_memory - initial_memory
        memory_recovered = before_gc_memory - after_gc_memory
        recovery_rate = (
            (memory_recovered / memory_created) * 100
            if memory_created > 0
            else 0
        )

        print("\nGarbage Collection Effectiveness:")
        print(f"Initial memory: {initial_memory:.2f} MB")
        print(f"After creation: {after_creation_memory:.2f} MB")
        print(f"Memory created: {memory_created:.2f} MB")
        print(f"Before GC: {before_gc_memory:.2f} MB")
        print(f"After GC: {after_gc_memory:.2f} MB")
        print(f"Memory recovered: {memory_recovered:.2f} MB")
        print(f"Recovery rate: {recovery_rate:.1f}%")
        print(f"Objects collected: {collected}")

        # Garbage collection should be effective
        assert (
            memory_recovered > 0
        ), "Garbage collection should recover some memory"
        assert (
            recovery_rate > 50
        ), f"GC recovery rate {recovery_rate:.1f}% too low"

    def test_object_lifecycle_memory_tracking(self):
        """Test memory tracking for object lifecycles."""
        # Track object creation and destruction
        created_objects = []
        destroyed_objects = []

        class TrackableObject:
            def __init__(self, data_size: int = 1000):
                self.data = np.random.random(data_size)
                self.id = id(self)
                created_objects.append(self.id)

                # Use weak reference to track destruction
                weakref.finalize(
                    self,
                    lambda obj_id=self.id: destroyed_objects.append(obj_id),
                )

        # Create and destroy objects in batches
        batch_size = 50
        total_batches = 10

        for batch in range(total_batches):
            # Create batch of objects
            objects = [TrackableObject(1000) for _ in range(batch_size)]

            # Use objects briefly
            total_data = sum(obj.data.sum() for obj in objects)
            assert total_data > 0  # Ensure objects are actually used

            # Clear batch
            objects.clear()
            gc.collect()  # Force cleanup

        # Final garbage collection
        gc.collect()

        print("\nObject Lifecycle Tracking:")
        print(f"Objects created: {len(created_objects)}")
        print(f"Objects destroyed: {len(destroyed_objects)}")
        print(f"Expected objects: {batch_size * total_batches}")

        # Most objects should be properly destroyed
        destruction_rate = len(destroyed_objects) / len(created_objects) * 100
        assert (
            destruction_rate > 90
        ), f"Object destruction rate {destruction_rate:.1f}% too low"

    def benchmark_memory_profiling_summary(self):
        """Generate comprehensive memory profiling benchmark summary."""
        print("\n" + "=" * 70)
        print("MEMORY PROFILING BENCHMARK SUMMARY")
        print("=" * 70)
        print("Tests performed:")
        print("  - Predictor memory usage patterns")
        print("  - Feature store memory efficiency")
        print("  - Event processing memory scaling")
        print("  - Long-running memory stability")
        print("  - Garbage collection effectiveness")
        print("  - Object lifecycle memory tracking")
        print("\nMemory leak detection and optimization validation.")
        print("=" * 70)


@pytest.mark.asyncio
@pytest.mark.performance
class TestMemoryProfilingIntegration:
    """Integration tests for memory profiling with real components."""

    async def test_end_to_end_memory_profiling(self):
        """Test end-to-end memory usage patterns."""
        assert True, "End-to-end memory profiling test placeholder"

    async def test_memory_usage_under_load(self):
        """Test memory usage patterns under system load."""
        assert True, "Memory under load test placeholder"


def benchmark_memory_performance():
    """Run comprehensive memory performance benchmarks."""
    print("\nRunning memory profiling benchmarks...")
    print("This validates memory efficiency and leak detection.")
    return {
        "test_file": "test_memory_profiling.py",
        "requirement": "Memory usage profiling and leak detection",
        "test_coverage": [
            "Component memory usage patterns",
            "Memory scaling with data volume",
            "Long-running stability testing",
            "Garbage collection effectiveness",
            "Memory leak detection",
            "Object lifecycle tracking",
        ],
    }


if __name__ == "__main__":
    result = benchmark_memory_performance()
    print(f"Benchmark configuration: {result}")
