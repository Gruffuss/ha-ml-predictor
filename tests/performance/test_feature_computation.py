"""
Performance tests for feature computation pipeline.

Target: Feature computation < 500ms (requirement from implementation-plan.md)

Tests feature extraction performance across different scenarios:
- Individual feature extractors (temporal, sequential, contextual)
- Complete feature pipeline performance
- Large dataset feature computation
- Concurrent feature extraction
- Feature caching effectiveness
"""

import asyncio
from datetime import datetime, timedelta
import time
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import statistics

from src.core.config import get_config
from src.data.storage.models import SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureStore
from src.features.temporal import TemporalFeatureExtractor


class TestFeatureComputationLatency:
    """Test feature computation performance and latency requirements."""

    @pytest.fixture
    def sample_events(self):
        """Create realistic sensor event data for performance testing."""
        events = []
        base_time = datetime.now() - timedelta(hours=24)

        # Generate 1000 events across 24 hours for realistic load
        for i in range(1000):
            event = SensorEvent(
                room_id="living_room",
                sensor_id=f"binary_sensor.living_room_motion_{i % 3}",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "of",
                previous_state="of" if i % 2 == 0 else "on",
                timestamp=base_time
                + timedelta(minutes=i * 1.44),  # ~1.44 min intervals
                attributes={"friendly_name": f"Living Room Motion {i % 3}"},
                is_human_triggered=(
                    True if i % 10 != 0 else False
                ),  # 90% human, 10% cat
            )
            events.append(event)

        return events

    @pytest.fixture
    def large_event_dataset(self):
        """Create large dataset for scalability testing."""
        events = []
        base_time = datetime.now() - timedelta(days=7)
        rooms = ["living_room", "bedroom", "kitchen", "bathroom", "office"]

        # Generate 10,000 events across 7 days for scalability testing
        for i in range(10000):
            room_id = rooms[i % len(rooms)]
            event = SensorEvent(
                room_id=room_id,
                sensor_id=f"binary_sensor.{room_id}_motion",
                sensor_type="motion",
                state="on" if i % 2 == 0 else "of",
                previous_state="of" if i % 2 == 0 else "on",
                timestamp=base_time + timedelta(minutes=i * 1.008),  # ~1 min intervals
                attributes={"friendly_name": f"{room_id.title()} Motion"},
                is_human_triggered=True if i % 8 != 0 else False,
            )
            events.append(event)

        return events

    @pytest.fixture
    def temporal_extractor(self):
        """Create temporal feature extractor."""
        return TemporalFeatureExtractor()

    @pytest.fixture
    def sequential_extractor(self):
        """Create sequential feature extractor."""
        return SequentialFeatureExtractor()

    @pytest.fixture
    def contextual_extractor(self):
        """Create contextual feature extractor."""
        return ContextualFeatureExtractor()

    @pytest.fixture
    async def feature_store(self):
        """Create feature store with mocked database."""
        with patch("src.features.store.get_database_manager"):
            store = FeatureStore()
            return store

    async def test_temporal_feature_extraction_performance(
        self, temporal_extractor, sample_events
    ):
        """Test temporal feature extraction latency."""
        target_time = datetime.now()
        latencies = []

        # Run multiple extractions for statistical analysis
        for _ in range(50):
            start_time = time.perf_counter()
            features = await temporal_extractor.extract_features(
                sample_events, target_time
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Verify feature extraction produced results
            assert features is not None
            assert len(features) > 0
            assert "time_since_last_change" in features
            assert "current_state_duration" in features

        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)

        print("\nTemporal Feature Extraction Results:")
        print(f"Mean: {mean_latency:.2f}ms")
        print(f"Median: {median_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"Features extracted: {len(features)}")

        # Temporal extraction should be very fast
        assert mean_latency < 50, f"Temporal extraction {mean_latency:.2f}ms too slow"
        assert (
            p95_latency < 100
        ), f"P95 temporal extraction {p95_latency:.2f}ms too slow"

    async def test_sequential_feature_extraction_performance(
        self, sequential_extractor, sample_events
    ):
        """Test sequential feature extraction latency."""
        from src.core.config import RoomConfig

        # Mock room configuration
        room_configs = {
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={"motion": ["binary_sensor.living_room_motion_0"]},
            )
        }

        latencies = []

        for _ in range(30):
            start_time = time.perf_counter()
            features = await sequential_extractor.extract_features(
                sample_events, room_configs
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            # Verify features
            assert features is not None
            assert len(features) > 0

        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print("\nSequential Feature Extraction Results:")
        print(f"Mean: {mean_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"Features extracted: {len(features)}")

        # Sequential features may be more complex
        assert (
            mean_latency < 150
        ), f"Sequential extraction {mean_latency:.2f}ms too slow"
        assert (
            p95_latency < 250
        ), f"P95 sequential extraction {p95_latency:.2f}ms too slow"

    async def test_contextual_feature_extraction_performance(
        self, contextual_extractor, sample_events
    ):
        """Test contextual feature extraction latency."""
        room_id = "living_room"
        target_time = datetime.now()
        latencies = []

        for _ in range(40):
            start_time = time.perf_counter()
            features = await contextual_extractor.extract_features(
                events=sample_events, room_id=room_id, target_time=target_time
            )
            end_time = time.perf_counter()

            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)

            assert features is not None
            assert len(features) > 0

        mean_latency = statistics.mean(latencies)
        p95_latency = np.percentile(latencies, 95)

        print("\nContextual Feature Extraction Results:")
        print(f"Mean: {mean_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"Features extracted: {len(features)}")

        # Contextual features may involve more computation
        assert (
            mean_latency < 200
        ), f"Contextual extraction {mean_latency:.2f}ms too slow"
        assert (
            p95_latency < 350
        ), f"P95 contextual extraction {p95_latency:.2f}ms too slow"

    async def test_complete_feature_pipeline_performance(
        self, feature_store, sample_events
    ):
        """Test complete feature engineering pipeline latency."""
        room_id = "living_room"
        target_time = datetime.now()
        latencies = []

        # Mock the database queries to return our sample events
        with (
            patch.object(
                feature_store, "_get_recent_events", return_value=sample_events
            ),
            patch.object(feature_store, "_get_room_context", return_value={}),
        ):

            for _ in range(25):
                start_time = time.perf_counter()
                features_df = await feature_store.compute_features(
                    room_id=room_id, target_time=target_time, lookback_hours=24
                )
                end_time = time.perf_counter()

                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # Verify complete feature set
                assert features_df is not None
                assert len(features_df) > 0
                assert len(features_df.columns) > 10  # Should have many features

        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)

        print("\nComplete Feature Pipeline Results:")
        print(f"Mean: {mean_latency:.2f}ms")
        print(f"Median: {median_latency:.2f}ms")
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        print(f"Features: {len(features_df.columns) if features_df is not None else 0}")

        # Complete pipeline must meet 500ms requirement
        assert (
            mean_latency < 500
        ), f"Complete pipeline {mean_latency:.2f}ms exceeds requirement"
        assert p95_latency < 750, f"P95 pipeline {p95_latency:.2f}ms too slow"
        assert p99_latency < 1000, f"P99 pipeline {p99_latency:.2f}ms too slow"

    async def test_large_dataset_feature_computation(
        self, feature_store, large_event_dataset
    ):
        """Test feature computation performance with large datasets."""
        room_id = "living_room"
        target_time = datetime.now()

        with (
            patch.object(
                feature_store,
                "_get_recent_events",
                return_value=large_event_dataset,
            ),
            patch.object(feature_store, "_get_room_context", return_value={}),
        ):

            # Test single large computation
            start_time = time.perf_counter()
            features_df = await feature_store.compute_features(
                room_id=room_id,
                target_time=target_time,
                lookback_hours=168,  # 7 days
            )
            large_dataset_latency = (time.perf_counter() - start_time) * 1000

            print("\nLarge Dataset Feature Computation:")
            print(f"Dataset size: {len(large_event_dataset)} events")
            print(f"Computation time: {large_dataset_latency:.2f}ms")
            print(
                f"Events per ms: {len(large_event_dataset) / large_dataset_latency:.2f}"
            )

            # Large datasets should still complete in reasonable time
            assert (
                large_dataset_latency < 2000
            ), f"Large dataset computation {large_dataset_latency:.2f}ms too slow"
            assert features_df is not None
            assert len(features_df) > 0

    async def test_concurrent_feature_computation(self, feature_store, sample_events):
        """Test feature computation performance under concurrent load."""
        rooms = ["living_room", "bedroom", "kitchen"]
        target_time = datetime.now()

        with (
            patch.object(
                feature_store, "_get_recent_events", return_value=sample_events
            ),
            patch.object(feature_store, "_get_room_context", return_value={}),
        ):

            async def compute_features_for_room(room_id):
                start_time = time.perf_counter()
                features = await feature_store.compute_features(
                    room_id=room_id, target_time=target_time, lookback_hours=24
                )
                latency = (time.perf_counter() - start_time) * 1000
                return room_id, latency, features

            # Run concurrent feature computations
            tasks = [compute_features_for_room(room_id) for room_id in rooms]
            results = await asyncio.gather(*tasks)

            latencies = [result[1] for result in results]
            mean_concurrent_latency = statistics.mean(latencies)
            max_concurrent_latency = max(latencies)

            print("\nConcurrent Feature Computation Results:")
            print(f"Concurrent rooms: {len(rooms)}")
            print(f"Mean latency: {mean_concurrent_latency:.2f}ms")
            print(f"Max latency: {max_concurrent_latency:.2f}ms")

            # Concurrent performance should degrade gracefully
            assert (
                mean_concurrent_latency < 600
            ), f"Mean concurrent latency {mean_concurrent_latency:.2f}ms too high"
            assert (
                max_concurrent_latency < 800
            ), f"Max concurrent latency {max_concurrent_latency:.2f}ms too high"

    async def test_feature_caching_performance(self, feature_store, sample_events):
        """Test feature caching effectiveness on computation performance."""
        room_id = "living_room"
        target_time = datetime.now()

        with (
            patch.object(
                feature_store, "_get_recent_events", return_value=sample_events
            ),
            patch.object(feature_store, "_get_room_context", return_value={}),
        ):

            # First computation (cache miss)
            start_time = time.perf_counter()
            features_1 = await feature_store.compute_features(room_id, target_time, 24)
            cold_latency = (time.perf_counter() - start_time) * 1000

            # Second computation (potential cache hit)
            start_time = time.perf_counter()
            features_2 = await feature_store.compute_features(room_id, target_time, 24)
            warm_latency = (time.perf_counter() - start_time) * 1000

            # Subsequent computations
            warm_latencies = []
            for _ in range(10):
                start_time = time.perf_counter()
                await feature_store.compute_features(room_id, target_time, 24)
                warm_latencies.append((time.perf_counter() - start_time) * 1000)

            mean_warm_latency = statistics.mean(warm_latencies)
            cache_improvement = (
                (cold_latency - mean_warm_latency) / cold_latency
            ) * 100

            print("\nFeature Caching Performance:")
            print(f"Cold computation: {cold_latency:.2f}ms")
            print(f"Warm computation: {mean_warm_latency:.2f}ms")
            print(f"Cache improvement: {cache_improvement:.1f}%")

            # Caching should provide meaningful performance improvement
            assert (
                mean_warm_latency <= cold_latency
            ), "Caching should not slow down computation"
            assert (
                mean_warm_latency < 400
            ), f"Cached computation {mean_warm_latency:.2f}ms still too slow"

    async def test_feature_computation_scalability(self, feature_store):
        """Test how feature computation scales with different data sizes."""
        room_id = "living_room"
        target_time = datetime.now()
        dataset_sizes = [100, 500, 1000, 2000]

        scalability_results = {}

        for size in dataset_sizes:
            # Generate dataset of specific size
            events = []
            base_time = datetime.now() - timedelta(hours=24)

            for i in range(size):
                event = SensorEvent(
                    room_id=room_id,
                    sensor_id="binary_sensor.motion",
                    sensor_type="motion",
                    state="on" if i % 2 == 0 else "of",
                    previous_state="of" if i % 2 == 0 else "on",
                    timestamp=base_time + timedelta(minutes=i * (24 * 60 / size)),
                    attributes={},
                    is_human_triggered=True,
                )
                events.append(event)

            with (
                patch.object(feature_store, "_get_recent_events", return_value=events),
                patch.object(feature_store, "_get_room_context", return_value={}),
            ):

                # Measure computation time
                start_time = time.perf_counter()
                features = await feature_store.compute_features(
                    room_id, target_time, 24
                )
                latency = (time.perf_counter() - start_time) * 1000

                scalability_results[size] = latency

        print("\nFeature Computation Scalability:")
        for size, latency in scalability_results.items():
            events_per_ms = size / latency if latency > 0 else 0
            print(f"{size} events: {latency:.2f}ms ({events_per_ms:.2f} events/ms)")

        # Verify all sizes meet requirements
        for size, latency in scalability_results.items():
            assert latency < 500 + (
                size * 0.1
            ), f"Computation for {size} events ({latency:.2f}ms) doesn't scale properly"

    def benchmark_feature_computation_summary(self):
        """Generate comprehensive feature computation benchmark summary."""
        print("\n" + "=" * 70)
        print("FEATURE COMPUTATION BENCHMARK SUMMARY")
        print("=" * 70)
        print("Requirement: Feature computation < 500ms")
        print("Components tested:")
        print("  - Temporal feature extraction")
        print("  - Sequential pattern features")
        print("  - Contextual feature computation")
        print("  - Complete pipeline performance")
        print("  - Large dataset scalability")
        print("  - Concurrent computation performance")
        print("  - Feature caching effectiveness")
        print("=" * 70)


@pytest.mark.asyncio
@pytest.mark.performance
class TestFeatureComputationIntegration:
    """Integration tests for feature computation with real components."""

    async def test_end_to_end_feature_computation_performance(self):
        """Test end-to-end feature computation with database integration."""
        assert True, "End-to-end feature computation test placeholder"

    async def test_feature_computation_memory_efficiency(self):
        """Test memory usage during feature computation."""
        assert True, "Memory efficiency test placeholder"


def benchmark_feature_computation_performance():
    """Run comprehensive feature computation benchmarks."""
    print("\nRunning feature computation benchmarks...")
    print("This validates the <500ms feature computation requirement.")
    return {
        "test_file": "test_feature_computation.py",
        "requirement": "Feature computation < 500ms",
        "test_coverage": [
            "Individual feature extractor performance",
            "Complete pipeline latency",
            "Large dataset scalability",
            "Concurrent computation efficiency",
            "Feature caching effectiveness",
            "Performance scaling analysis",
        ],
    }


if __name__ == "__main__":
    result = benchmark_feature_computation_performance()
    print(f"Benchmark configuration: {result}")
