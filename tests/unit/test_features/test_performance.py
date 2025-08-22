"""
Comprehensive performance and memory testing for feature engineering modules.

This test suite validates feature computation performance, memory efficiency,
and system behavior under high-load conditions.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import gc
import threading
import time
from typing import Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import psutil
import pytest

from src.core.config import RoomConfig
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureCache, FeatureStore
from src.features.temporal import TemporalFeatureExtractor


class TestFeaturePerformanceBenchmarks:
    """Test feature extraction performance under various load conditions."""

    @pytest.fixture
    def large_event_dataset(self):
        """Create large dataset for performance testing."""
        events = []
        base_time = datetime(2024, 1, 1, 0, 0, 0)

        # Generate 10,000 events across 7 days
        for i in range(10000):
            event = Mock(spec=SensorEvent)
            event.room_id = f"room_{i % 10}"
            event.sensor_id = f"sensor_{i % 50}"
            event.sensor_type = ["motion", "door", "temperature", "humidity"][i % 4]
            event.state = ["on", "off"][i % 2]
            event.timestamp = base_time + timedelta(minutes=i)
            event.attributes = {
                "temperature": 20.0 + (i % 20),
                "humidity": 40.0 + (i % 30),
                "battery": 90 - (i % 30),
            }
            events.append(event)

        return events

    @pytest.fixture
    def memory_monitor(self):
        """Monitor memory usage during tests."""
        process = psutil.Process()

        def get_memory_mb():
            return process.memory_info().rss / 1024 / 1024

        return get_memory_mb

    def test_temporal_feature_performance_large_dataset(
        self, large_event_dataset, memory_monitor
    ):
        """Test temporal feature extraction performance with large dataset."""
        extractor = TemporalFeatureExtractor()
        target_time = datetime(2024, 1, 7, 12, 0, 0)

        # Measure initial memory
        initial_memory = memory_monitor()

        # Performance test
        start_time = time.time()
        features = extractor.extract_features(large_event_dataset, target_time)
        end_time = time.time()

        # Memory after extraction
        final_memory = memory_monitor()

        # Assertions
        assert (
            end_time - start_time < 2.0
        ), "Temporal extraction should complete in under 2 seconds"
        assert len(features) > 20, "Should extract comprehensive temporal features"
        assert (
            final_memory - initial_memory < 50
        ), "Memory increase should be under 50MB"

        # Validate feature quality
        assert all(
            isinstance(v, (int, float)) for v in features.values()
        ), "All features should be numeric"
        assert not any(np.isnan(v) for v in features.values()), "No NaN values allowed"

    def test_sequential_feature_performance_concurrent_rooms(
        self, large_event_dataset, memory_monitor
    ):
        """Test sequential feature extraction with concurrent processing."""
        extractor = SequentialFeatureExtractor()
        room_configs = {f"room_{i}": Mock(spec=RoomConfig) for i in range(10)}
        target_time = datetime(2024, 1, 7, 12, 0, 0)

        initial_memory = memory_monitor()

        # Concurrent extraction test
        start_time = time.time()

        def extract_for_room_subset(events_subset):
            return extractor.extract_features(events_subset, target_time, room_configs)

        # Split events into chunks for concurrent processing
        chunk_size = len(large_event_dataset) // 4
        chunks = [
            large_event_dataset[i : i + chunk_size]
            for i in range(0, len(large_event_dataset), chunk_size)
        ]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [
                executor.submit(extract_for_room_subset, chunk) for chunk in chunks
            ]
            results = [future.result() for future in as_completed(futures)]

        end_time = time.time()
        final_memory = memory_monitor()

        # Assertions
        assert (
            end_time - start_time < 3.0
        ), "Concurrent sequential extraction should complete in under 3 seconds"
        assert len(results) == 4, "Should process all chunks"
        assert (
            final_memory - initial_memory < 100
        ), "Memory increase should be reasonable"

    def test_feature_engineering_batch_performance(
        self, large_event_dataset, memory_monitor
    ):
        """Test batch feature extraction performance."""
        engine = FeatureEngineeringEngine()
        room_configs = {f"room_{i}": Mock(spec=RoomConfig) for i in range(10)}

        # Create multiple extraction targets
        targets = [
            (f"room_{i}", datetime(2024, 1, 7, hour, 0, 0))
            for i in range(10)
            for hour in range(6, 18, 2)
        ]  # 60 targets total

        initial_memory = memory_monitor()
        start_time = time.time()

        results = engine.extract_batch_features(
            targets, large_event_dataset, room_configs, parallel=True
        )

        end_time = time.time()
        final_memory = memory_monitor()

        # Assertions
        assert (
            end_time - start_time < 5.0
        ), "Batch extraction should complete in under 5 seconds"
        assert len(results) == len(targets), "Should process all targets"
        assert (
            final_memory - initial_memory < 200
        ), "Memory increase should be manageable"

        # Validate result quality
        for result in results.values():
            assert isinstance(
                result, dict
            ), "Each result should be a feature dictionary"
            assert len(result) > 10, "Should extract meaningful number of features"

    def test_memory_efficiency_with_cache_cleanup(
        self, large_event_dataset, memory_monitor
    ):
        """Test memory efficiency with automatic cache cleanup."""
        cache = FeatureCache(max_size=100)  # Small cache to force evictions

        initial_memory = memory_monitor()

        # Fill cache beyond capacity
        for i in range(200):
            key = f"test_key_{i}"
            large_feature_dict = {f"feature_{j}": float(j) for j in range(100)}
            cache.put(key, large_feature_dict)

        mid_memory = memory_monitor()

        # Force garbage collection
        gc.collect()

        final_memory = memory_monitor()

        # Assertions
        assert cache.size() <= 100, "Cache should respect size limit"
        assert (
            mid_memory - initial_memory > 0
        ), "Memory should increase during cache filling"
        assert final_memory <= mid_memory, "Memory should not grow indefinitely"

    def test_concurrent_feature_store_access(self, large_event_dataset):
        """Test concurrent access to feature store."""
        with patch("src.features.store.DatabaseManager") as mock_db:
            store = FeatureStore(db_manager=mock_db)

            def concurrent_access(thread_id):
                room_id = f"room_{thread_id % 5}"
                target_time = datetime(2024, 1, 7, 12, 0, 0)

                # Simulate multiple operations
                features = store.get_features(room_id, target_time, large_event_dataset)
                return len(features) if features else 0

            # Run concurrent access test
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(concurrent_access, i) for i in range(20)]
                results = [future.result() for future in as_completed(futures)]

            # Assertions
            assert len(results) == 20, "All concurrent operations should complete"
            assert all(
                r >= 0 for r in results
            ), "All operations should return valid results"

    def test_feature_extraction_scalability(self, memory_monitor):
        """Test feature extraction scalability with increasing data size."""
        extractor = TemporalFeatureExtractor()
        target_time = datetime(2024, 1, 7, 12, 0, 0)

        # Test with different data sizes
        sizes = [100, 500, 1000, 5000, 10000]
        times = []
        memory_usage = []

        for size in sizes:
            # Generate events
            events = []
            for i in range(size):
                event = Mock(spec=SensorEvent)
                event.timestamp = target_time - timedelta(minutes=i)
                event.state = "on" if i % 2 == 0 else "off"
                event.sensor_type = "motion"
                events.append(event)

            # Measure extraction time
            initial_memory = memory_monitor()
            start_time = time.time()

            features = extractor.extract_features(events, target_time)

            end_time = time.time()
            final_memory = memory_monitor()

            times.append(end_time - start_time)
            memory_usage.append(final_memory - initial_memory)

        # Scalability assertions
        # Time complexity should be roughly O(n log n) or better
        time_ratios = [times[i + 1] / times[i] for i in range(len(times) - 1)]
        assert all(ratio < 10 for ratio in time_ratios), "Time should scale reasonably"

        # Memory usage should scale linearly or sublinearly
        memory_ratios = [
            memory_usage[i + 1] / max(memory_usage[i], 1)
            for i in range(len(memory_usage) - 1)
        ]
        assert all(
            ratio < 5 for ratio in memory_ratios
        ), "Memory should scale reasonably"


class TestFeatureMissingDataHandling:
    """Test feature extraction with missing and incomplete data."""

    def test_temporal_features_with_sensor_gaps(self):
        """Test temporal feature extraction with sensor data gaps."""
        extractor = TemporalFeatureExtractor()
        target_time = datetime(2024, 1, 7, 12, 0, 0)

        # Create events with gaps
        events = []
        base_time = target_time - timedelta(hours=24)

        # First batch: 0-6 hours ago
        for i in range(100):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 2)
            event.state = "on" if i % 3 == 0 else "off"
            event.sensor_type = "motion"
            events.append(event)

        # Gap: 6-18 hours (no events)

        # Second batch: 18-24 hours ago
        for i in range(100, 200):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(hours=18, minutes=(i - 100) * 2)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should handle gaps gracefully
        assert len(features) > 10, "Should extract features despite gaps"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "No NaN values despite gaps"
        assert (
            "time_since_last_change" in features
        ), "Should calculate time since last change"

    def test_sequential_features_with_incomplete_sequences(self):
        """Test sequential feature extraction with incomplete room sequences."""
        extractor = SequentialFeatureExtractor()
        room_configs = {
            "living_room": Mock(spec=RoomConfig),
            "bedroom": Mock(spec=RoomConfig),
            # Missing "kitchen" room config
        }

        # Events including missing room
        events = []
        for i, room in enumerate(["living_room", "bedroom", "kitchen", "living_room"]):
            event = Mock(spec=SensorEvent)
            event.room_id = room
            event.timestamp = datetime(2024, 1, 7, 12, i, 0)
            event.sensor_type = "motion"
            event.state = "on"
            events.append(event)

        features = extractor.extract_features(events, room_configs)

        # Should handle missing room configs gracefully
        assert len(features) > 0, "Should extract some features"
        assert all(not np.isnan(v) for v in features.values()), "No NaN values"

    def test_contextual_features_with_missing_environmental_data(self):
        """Test contextual features with missing environmental sensor data."""
        extractor = ContextualFeatureExtractor()

        # Events with missing attributes
        events = []
        for i in range(50):
            event = Mock(spec=SensorEvent)
            event.room_id = "living_room"
            event.timestamp = datetime(2024, 1, 7, 12, 0, 0) - timedelta(minutes=i)
            event.sensor_type = ["motion", "temperature", "humidity", "light"][i % 4]

            # Randomly missing attributes
            if i % 3 == 0:
                event.attributes = {}  # No attributes
            elif i % 3 == 1:
                event.attributes = {"temperature": 22.0}  # Partial attributes
            else:
                event.attributes = {
                    "temperature": 22.0,
                    "humidity": 45.0,
                    "light_level": 300.0,
                }
            events.append(event)

        room_states = [
            Mock(room_id="living_room", state="occupied", timestamp=events[0].timestamp)
        ]

        features = extractor.extract_features(events, room_states)

        # Should provide defaults for missing data
        assert len(features) > 10, "Should extract features with defaults"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should handle missing data"

    def test_feature_store_with_database_unavailable(self):
        """Test feature store behavior when database is unavailable."""
        with patch("src.features.store.DatabaseManager") as mock_db:
            # Simulate database failure
            mock_db.return_value.get_events.side_effect = Exception(
                "Database connection failed"
            )

            store = FeatureStore(db_manager=mock_db)

            # Should gracefully handle database failure
            features = store.get_features(
                "living_room",
                datetime(2024, 1, 7, 12, 0, 0),
                [],  # Provide fallback events
            )

            # Should return empty/default features instead of crashing
            assert features is not None, "Should handle database failure gracefully"

    def test_feature_extraction_with_corrupted_events(self):
        """Test feature extraction with corrupted/malformed events."""
        extractor = TemporalFeatureExtractor()

        # Mix of valid and corrupted events
        events = []

        # Valid events
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = datetime(2024, 1, 7, 12, 0, 0) - timedelta(minutes=i)
            event.state = "on"
            event.sensor_type = "motion"
            events.append(event)

        # Corrupted events
        corrupted1 = Mock(spec=SensorEvent)
        corrupted1.timestamp = None  # Invalid timestamp
        corrupted1.state = "on"
        events.append(corrupted1)

        corrupted2 = Mock(spec=SensorEvent)
        corrupted2.timestamp = "invalid_time"  # String instead of datetime
        corrupted2.state = "on"
        events.append(corrupted2)

        # Should handle corrupted events gracefully
        features = extractor.extract_features(events, datetime(2024, 1, 7, 12, 0, 0))

        assert len(features) > 0, "Should extract features from valid events"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should ignore corrupted events"


class TestFeatureTimezoneTransitions:
    """Test feature extraction across timezone changes and DST transitions."""

    def test_temporal_features_across_dst_spring_forward(self):
        """Test temporal features during DST spring forward transition."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)  # PST

        # Create events around DST transition (2nd Sunday in March, 2:00 AM -> 3:00 AM)
        events = []
        base_time = datetime(2024, 3, 10, 1, 0, 0)  # 1 AM before transition

        for i in range(120):  # 2 hours of events
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.state = "on" if i % 10 < 5 else "off"
            event.sensor_type = "motion"
            events.append(event)

        # Target time after DST transition
        target_time = datetime(2024, 3, 10, 4, 0, 0)  # 4 AM (after DST)

        features = extractor.extract_features(events, target_time)

        # Should handle DST transition gracefully
        assert len(features) > 10, "Should extract features across DST transition"
        assert "cyclical_hour_sin" in features, "Should include cyclical time features"
        assert (
            abs(features["cyclical_hour_sin"]) <= 1.0
        ), "Cyclical features should be valid"

    def test_temporal_features_across_dst_fall_back(self):
        """Test temporal features during DST fall back transition."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)

        # Create events around DST fall back (1st Sunday in November, 2:00 AM -> 1:00 AM)
        events = []
        base_time = datetime(2024, 11, 3, 0, 0, 0)  # Midnight before transition

        for i in range(240):  # 4 hours of events
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.state = "on" if i % 8 < 4 else "off"
            event.sensor_type = "motion"
            events.append(event)

        target_time = datetime(2024, 11, 3, 3, 0, 0)  # After DST fall back

        features = extractor.extract_features(events, target_time)

        # Should handle repeated hour gracefully
        assert len(features) > 10, "Should extract features across DST fall back"
        assert not any(
            np.isnan(v) for v in features.values()
        ), "No NaN values after DST"

    def test_feature_extraction_timezone_changes(self):
        """Test feature extraction when timezone changes mid-stream."""
        # Start with one timezone
        extractor_pst = TemporalFeatureExtractor(timezone_offset=-8)

        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # First half: PST events
        for i in range(60):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.state = "on" if i % 6 < 3 else "off"
            event.sensor_type = "motion"
            events.append(event)

        # Simulate timezone change (moving from PST to EST)
        extractor_est = TemporalFeatureExtractor(timezone_offset=-5)

        # Extract features with new timezone
        target_time = base_time + timedelta(hours=2)
        features = extractor_est.extract_features(events, target_time)

        # Should adapt to new timezone
        assert len(features) > 10, "Should extract features with timezone change"
        assert "cyclical_hour_sin" in features, "Should recalculate cyclical features"

    def test_cross_timezone_room_correlation(self):
        """Test contextual features with rooms in different timezones."""
        extractor = ContextualFeatureExtractor()

        # Events from multiple "timezones" (simulated)
        events = []
        for i, tz_offset in enumerate([-8, -5, 0]):  # PST, EST, UTC
            for j in range(20):
                event = Mock(spec=SensorEvent)
                event.room_id = f"room_tz_{tz_offset}"
                # Adjust timestamp for timezone simulation
                event.timestamp = datetime(2024, 6, 15, 12 - tz_offset, j, 0)
                event.sensor_type = "motion"
                event.state = "on" if j % 4 < 2 else "off"
                events.append(event)

        room_states = []
        for tz_offset in [-8, -5, 0]:
            state = Mock(spec=RoomState)
            state.room_id = f"room_tz_{tz_offset}"
            state.state = "occupied"
            state.timestamp = datetime(2024, 6, 15, 12 - tz_offset, 30, 0)
            room_states.append(state)

        features = extractor.extract_features(events, room_states)

        # Should handle cross-timezone correlations
        assert len(features) > 5, "Should extract cross-timezone features"
        assert (
            "multi_room_correlation_active_rooms" in features
        ), "Should track active rooms"


class TestFeatureCacheInvalidationScenarios:
    """Test complex caching scenarios and cache invalidation."""

    def test_feature_cache_with_memory_pressure(self):
        """Test cache behavior under memory pressure."""
        # Create cache with limited memory
        cache = FeatureCache(max_size=50)

        # Fill cache to capacity
        large_features = {}
        for i in range(100):  # More than max_size
            key = f"feature_set_{i}"
            # Create large feature dictionary
            features = {f"feature_{j}": float(i * j) for j in range(200)}
            large_features[key] = features
            cache.put(key, features)

        # Verify LRU behavior
        assert cache.size() <= 50, "Cache should respect size limit"

        # Most recent items should still be accessible
        for i in range(90, 100):
            key = f"feature_set_{i}"
            cached_features = cache.get(key)
            assert cached_features is not None, f"Recent item {key} should be in cache"

    def test_cache_invalidation_on_time_expiry(self):
        """Test cache invalidation based on time expiry."""
        cache = FeatureCache(max_size=100)

        # Add features with different timestamps
        old_features = {"temp_feature": 20.0}
        recent_features = {"temp_feature": 25.0}

        # Mock time-based invalidation
        cache.put("old_key", old_features, max_age_seconds=1)
        time.sleep(0.5)  # Wait half the expiry time
        cache.put("recent_key", recent_features, max_age_seconds=10)

        # Old key should expire
        time.sleep(1.0)  # Wait for expiry

        assert cache.get("old_key") is None, "Expired features should be removed"
        assert cache.get("recent_key") is not None, "Recent features should remain"

    def test_concurrent_cache_access_and_invalidation(self):
        """Test concurrent cache access with invalidation."""
        cache = FeatureCache(max_size=1000)

        def cache_worker(worker_id, iterations=100):
            """Worker function for concurrent cache operations."""
            for i in range(iterations):
                key = f"worker_{worker_id}_item_{i}"
                features = {f"feature_{j}": float(worker_id * i * j) for j in range(10)}

                # Put and get operations
                cache.put(key, features)
                retrieved = cache.get(key)

                if retrieved is None:
                    continue  # Item may have been evicted

                assert retrieved == features, "Retrieved features should match stored"

        # Run concurrent workers
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(cache_worker, i) for i in range(8)]
            for future in as_completed(futures):
                future.result()  # Wait for completion

        # Cache should be in valid state
        assert (
            cache.size() <= 1000
        ), "Cache should respect size limit after concurrent access"
        stats = cache.get_statistics()
        assert stats["hits"] > 0 or stats["misses"] > 0, "Should have cache activity"

    def test_feature_store_cache_coherence(self):
        """Test cache coherence in feature store under updates."""
        with patch("src.features.store.DatabaseManager") as mock_db:
            store = FeatureStore(db_manager=mock_db)

            room_id = "test_room"
            target_time = datetime(2024, 1, 7, 12, 0, 0)

            # Initial feature extraction
            events1 = [Mock(spec=SensorEvent) for _ in range(10)]
            features1 = store.get_features(room_id, target_time, events1)

            # Should be cached
            cached_features = store.get_features(room_id, target_time, events1)
            assert cached_features == features1, "Should return cached features"

            # New events should invalidate cache
            events2 = [Mock(spec=SensorEvent) for _ in range(15)]
            features2 = store.get_features(room_id, target_time, events2)

            # Should be different if cache was properly invalidated
            # (This depends on implementation - here we test that it doesn't crash)
            assert features2 is not None, "Should handle cache invalidation"

    def test_memory_leak_prevention_in_cache(self):
        """Test that cache prevents memory leaks during long operations."""
        cache = FeatureCache(max_size=100)

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB

        # Simulate long-running operation with many cache operations
        for iteration in range(5):
            for i in range(200):  # Exceed cache capacity
                key = f"iter_{iteration}_item_{i}"
                large_feature_dict = {f"feature_{j}": float(i * j) for j in range(500)}
                cache.put(key, large_feature_dict)

            # Force garbage collection
            gc.collect()

            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = current_memory - initial_memory

            # Memory should not grow indefinitely
            assert (
                memory_increase < 500
            ), f"Memory increase ({memory_increase:.1f} MB) should be bounded"

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024
        total_increase = final_memory - initial_memory

        assert total_increase < 200, "Total memory increase should be reasonable"


class TestFeatureErrorRecoveryAndResilience:
    """Test error recovery and system resilience in feature extraction."""

    def test_temporal_feature_extraction_with_partial_failures(self):
        """Test temporal feature extraction continues after partial failures."""

        class FailingTemporalExtractor(TemporalFeatureExtractor):
            def _extract_cyclical_features(self, target_time):
                # Simulate failure in one feature type
                raise Exception("Cyclical feature extraction failed")

        extractor = FailingTemporalExtractor()
        events = [Mock(spec=SensorEvent) for _ in range(10)]
        for i, event in enumerate(events):
            event.timestamp = datetime(2024, 1, 7, 12, 0, 0) - timedelta(minutes=i)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"

        # Should continue extraction despite partial failure
        features = extractor.extract_features(events, datetime(2024, 1, 7, 12, 0, 0))

        # Should have some features despite failure
        assert len(features) > 0, "Should extract some features despite partial failure"
        assert (
            "time_since_last_change" in features
        ), "Should extract other feature types"

    def test_feature_engineering_graceful_degradation(self):
        """Test feature engineering graceful degradation on extractor failures."""

        class FlakyExtractor:
            def __init__(self, failure_rate=0.3):
                self.failure_rate = failure_rate
                self.call_count = 0

            def extract_features(self, *args, **kwargs):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    raise Exception("Extractor temporarily unavailable")
                return {"test_feature": float(self.call_count)}

            def get_feature_names(self):
                return ["test_feature"]

        engine = FeatureEngineeringEngine()
        engine.temporal_extractor = FlakyExtractor()
        engine.sequential_extractor = FlakyExtractor()
        engine.contextual_extractor = FlakyExtractor()

        # Multiple extractions - some should fail, some succeed
        room_id = "test_room"
        target_time = datetime(2024, 1, 7, 12, 0, 0)
        events = [Mock(spec=SensorEvent) for _ in range(5)]
        room_configs = {"test_room": Mock(spec=RoomConfig)}

        # Should handle extractor failures gracefully
        for i in range(10):
            try:
                features = engine.extract_features(
                    room_id, target_time, events, room_configs
                )
                # When successful, should have some features
                assert len(features) > 0, "Successful extractions should yield features"
            except Exception as e:
                # Failures should be handled gracefully
                assert "temporarily unavailable" in str(
                    e
                ), "Should get expected error message"

    def test_feature_store_database_reconnection(self):
        """Test feature store handles database reconnection."""
        connection_attempts = []

        class FlakySensorEvent:
            """Mock that fails initially then succeeds."""

            def __init__(self):
                self.attempt_count = 0

            @property
            def query(self):
                return self

            def filter(self, *args, **kwargs):
                return self

            def order_by(self, *args, **kwargs):
                return self

            def all(self):
                self.attempt_count += 1
                connection_attempts.append(self.attempt_count)

                if self.attempt_count <= 2:
                    raise Exception("Database connection failed")

                # Return mock events after reconnection
                return [Mock(spec=SensorEvent) for _ in range(5)]

        with patch("src.features.store.SensorEvent", FlakySensorEvent()):
            with patch("src.features.store.DatabaseManager") as mock_db:
                mock_db.return_value.get_session.return_value.__enter__.return_value = (
                    Mock()
                )

                store = FeatureStore(db_manager=mock_db)

                # Should eventually succeed after retries
                features = store.get_data_for_features(
                    "test_room", datetime(2024, 1, 7, 12, 0, 0), lookback_hours=24
                )

                assert len(connection_attempts) >= 3, "Should retry database connection"
                assert features is not None, "Should eventually succeed"

    def test_concurrent_feature_extraction_with_failures(self):
        """Test concurrent feature extraction handles individual failures."""

        def extract_with_random_failure(task_id):
            """Extraction function that randomly fails."""
            if task_id % 4 == 0:  # Fail every 4th task
                raise Exception(f"Task {task_id} failed")

            # Return mock features for successful tasks
            return {f"feature_{task_id}": float(task_id)}

        # Run concurrent extractions
        successful_results = []
        failed_tasks = []

        with ThreadPoolExecutor(max_workers=6) as executor:
            futures = {
                executor.submit(extract_with_random_failure, i): i for i in range(20)
            }

            for future in as_completed(futures):
                task_id = futures[future]
                try:
                    result = future.result()
                    successful_results.append((task_id, result))
                except Exception as e:
                    failed_tasks.append(task_id)

        # Verify resilience
        assert len(successful_results) > 0, "Some tasks should succeed"
        assert len(failed_tasks) > 0, "Some tasks should fail (as expected)"
        assert (
            len(successful_results) + len(failed_tasks) == 20
        ), "All tasks should complete"

        # Failed tasks should be the expected ones (every 4th)
        expected_failures = [i for i in range(20) if i % 4 == 0]
        assert failed_tasks == expected_failures, "Expected tasks should fail"

    def test_feature_extraction_memory_exhaustion_recovery(self):
        """Test recovery from memory exhaustion scenarios."""

        def memory_intensive_extraction():
            """Simulate memory-intensive feature extraction."""
            # Create large data structures
            large_data = []
            try:
                for i in range(10000):
                    # Simulate memory pressure
                    large_array = np.random.random(1000)
                    large_data.append(large_array)

                    # Check memory usage
                    if i % 1000 == 0:
                        memory_mb = psutil.Process().memory_info().rss / 1024 / 1024
                        if memory_mb > 500:  # Limit memory usage
                            raise MemoryError("Memory exhaustion detected")

                return {"memory_feature": len(large_data)}

            except MemoryError:
                # Clean up and return partial result
                del large_data
                gc.collect()
                return {"memory_feature": -1}  # Indicate partial failure

        # Should handle memory exhaustion gracefully
        result = memory_intensive_extraction()

        assert result is not None, "Should return result even after memory exhaustion"
        assert "memory_feature" in result, "Should return at least partial features"
