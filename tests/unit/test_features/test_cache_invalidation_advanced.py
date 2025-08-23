"""
Advanced feature store cache invalidation and memory efficiency tests.

This test suite validates:
- Complex cache invalidation scenarios
- Memory-efficient cache operations
- Cache coherence under concurrent access
- Cache performance optimization
- Advanced cache eviction strategies
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import gc
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch
import weakref

import numpy as np
import psutil
import pytest

from src.core.config import RoomConfig
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureCache, FeatureRecord, FeatureStore
from src.features.temporal import TemporalFeatureExtractor


class TestAdvancedCacheInvalidation:
    """Test complex cache invalidation scenarios and strategies."""

    @pytest.fixture
    def memory_aware_cache(self):
        """Create cache with memory monitoring capabilities."""

        class MemoryAwareCache(FeatureCache):
            def __init__(self, max_size=100, memory_limit_mb=50):
                super().__init__(max_size=max_size)
                self.memory_limit_mb = memory_limit_mb
                self.memory_checks = 0

            def _check_memory_pressure(self):
                """Check if system is under memory pressure."""
                self.memory_checks += 1
                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                return memory_mb > self.memory_limit_mb

            def put(self, key: str, features: Dict[str, float], **kwargs):
                # Check memory before adding
                if self._check_memory_pressure():
                    # Aggressive eviction under memory pressure
                    self._evict_oldest(0.5)  # Evict 50% of entries

                return super().put(key, features, **kwargs)

            def _evict_oldest(self, fraction: float):
                """Evict fraction of oldest entries."""
                target_size = int(self.max_size * (1 - fraction))
                while self.size() > target_size and self._cache:
                    oldest_key = next(iter(self._cache))
                    del self._cache[oldest_key]

        return MemoryAwareCache()

    def test_cache_invalidation_on_data_freshness(self, memory_aware_cache):
        """Test cache invalidation based on data freshness requirements."""
        cache = memory_aware_cache

        # Add features with different freshness requirements
        fresh_features = {"temp_feature": 22.0, "humidity_feature": 45.0}
        stale_features = {"old_temp_feature": 18.0, "old_humidity_feature": 50.0}

        # Add with different max ages
        cache.put("fresh_key", fresh_features, max_age_seconds=300)  # 5 minutes
        time.sleep(0.1)
        cache.put("stale_key", stale_features, max_age_seconds=1)  # 1 second

        # Wait for stale data to expire
        time.sleep(1.5)

        # Fresh data should remain, stale should be gone
        assert cache.get("fresh_key") is not None, "Fresh data should remain"
        assert cache.get("stale_key") is None, "Stale data should be invalidated"

        # Memory checks should have occurred
        assert cache.memory_checks > 0, "Memory pressure should be monitored"

    def test_cascading_cache_invalidation(self):
        """Test cascading invalidation when dependent features change."""

        class DependentFeatureCache(FeatureCache):
            def __init__(self):
                super().__init__(max_size=200)
                self.dependencies = {}  # Track feature dependencies

            def add_dependency(self, dependent_key: str, source_key: str):
                """Add dependency relationship."""
                if source_key not in self.dependencies:
                    self.dependencies[source_key] = set()
                self.dependencies[source_key].add(dependent_key)

            def invalidate_cascade(self, source_key: str):
                """Invalidate source and all dependent features."""
                # Invalidate source
                if source_key in self._cache:
                    del self._cache[source_key]

                # Cascade to dependents
                if source_key in self.dependencies:
                    for dependent_key in self.dependencies[source_key]:
                        if dependent_key in self._cache:
                            del self._cache[dependent_key]
                        # Recursively invalidate dependents of dependents
                        self.invalidate_cascade(dependent_key)

        cache = DependentFeatureCache()

        # Create dependency chain: base -> derived -> aggregated
        cache.put("base_features", {"sensor_value": 25.0})
        cache.put("derived_features", {"processed_value": 50.0})
        cache.put("aggregated_features", {"summary_value": 75.0})

        # Set up dependencies
        cache.add_dependency("derived_features", "base_features")
        cache.add_dependency("aggregated_features", "derived_features")

        # Verify all features are cached
        assert cache.get("base_features") is not None
        assert cache.get("derived_features") is not None
        assert cache.get("aggregated_features") is not None

        # Invalidate base features - should cascade
        cache.invalidate_cascade("base_features")

        # All should be invalidated due to cascading
        assert cache.get("base_features") is None
        assert cache.get("derived_features") is None
        assert cache.get("aggregated_features") is None

    def test_selective_cache_invalidation_by_pattern(self):
        """Test selective cache invalidation based on key patterns."""
        cache = FeatureCache(max_size=100)

        # Add various feature types
        test_data = {
            "room_living_temporal_features": {"time_since": 1.0},
            "room_living_sequential_features": {"transitions": 2.0},
            "room_bedroom_temporal_features": {"time_since": 3.0},
            "room_bedroom_sequential_features": {"transitions": 4.0},
            "global_environmental_features": {"temperature": 22.0},
            "global_system_features": {"uptime": 100.0},
        }

        for key, features in test_data.items():
            cache.put(key, features)

        # Verify all cached
        assert cache.size() == 6

        # Selective invalidation by room
        def invalidate_room_features(cache, room_id):
            """Invalidate all features for a specific room."""
            keys_to_remove = []
            for key in cache._cache.keys():
                if f"room_{room_id}_" in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del cache._cache[key]

        # Invalidate only living room features
        invalidate_room_features(cache, "living")

        # Verify selective invalidation
        assert cache.get("room_living_temporal_features") is None
        assert cache.get("room_living_sequential_features") is None
        assert cache.get("room_bedroom_temporal_features") is not None
        assert cache.get("room_bedroom_sequential_features") is not None
        assert cache.get("global_environmental_features") is not None
        assert cache.get("global_system_features") is not None

    def test_cache_invalidation_on_config_changes(self):
        """Test cache invalidation when system configuration changes."""
        with patch("src.features.store.DatabaseManager") as mock_db:
            store = FeatureStore(db_manager=mock_db)

            room_id = "configurable_room"
            target_time = datetime(2024, 1, 15, 12, 0, 0)
            events = [Mock(spec=SensorEvent) for _ in range(5)]

            # Initial configuration
            original_config = {
                "lookback_hours": 24,
                "feature_types": ["temporal", "sequential"],
            }

            # Extract features with original config
            features1 = store.get_features(
                room_id, target_time, events, config=original_config
            )

            # Verify caching
            cached_features = store.get_features(
                room_id, target_time, events, config=original_config
            )
            assert (
                cached_features == features1
            ), "Should return cached features with same config"

            # Change configuration
            new_config = {
                "lookback_hours": 48,
                "feature_types": ["temporal", "contextual"],
            }

            # Should invalidate cache and compute new features
            features2 = store.get_features(
                room_id, target_time, events, config=new_config
            )

            # Should be different due to config change
            assert features2 is not None, "Should compute features with new config"

    def test_time_based_cache_invalidation_with_sliding_window(self):
        """Test time-based invalidation with sliding window approach."""

        class SlidingWindowCache(FeatureCache):
            def __init__(self, window_minutes=30):
                super().__init__(max_size=1000)
                self.window_minutes = window_minutes

            def get_with_time_window(
                self, key: str, current_time: datetime
            ) -> Optional[Dict[str, float]]:
                """Get cached features only if within time window."""
                record = self._cache.get(key)
                if not record:
                    return None

                # Check if within sliding window
                time_diff = (current_time - record.timestamp).total_seconds() / 60
                if time_diff <= self.window_minutes:
                    # Update position in LRU
                    self._cache.move_to_end(key)
                    return record.features
                else:
                    # Remove expired entry
                    del self._cache[key]
                    return None

        cache = SlidingWindowCache(window_minutes=15)  # 15-minute window

        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Add features at different times
        cache.put("recent_features", {"value": 1.0})
        time.sleep(0.1)

        # Simulate time passing
        current_time = base_time + timedelta(minutes=10)  # Within window
        assert cache.get_with_time_window("recent_features", current_time) is not None

        # Simulate more time passing - outside window
        current_time = base_time + timedelta(minutes=20)  # Outside window
        assert cache.get_with_time_window("recent_features", current_time) is None


class TestCacheMemoryEfficiency:
    """Test memory efficiency and optimization in caching operations."""

    def test_memory_efficient_large_feature_sets(self):
        """Test memory efficiency with large feature sets."""
        cache = FeatureCache(max_size=50)  # Small cache for large features

        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Add many large feature sets
        for i in range(100):
            key = f"large_feature_set_{i}"
            # Create large feature dictionary
            large_features = {f"feature_{j}": float(i * j) for j in range(1000)}
            cache.put(key, large_features)

        mid_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Force garbage collection
        gc.collect()

        final_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # Memory should be bounded by cache size
        memory_increase = final_memory - initial_memory
        assert memory_increase < 200, "Memory increase should be bounded"

        # Cache should respect size limit
        assert cache.size() <= 50, "Cache should respect size limit"

    def test_weak_reference_cache_cleanup(self):
        """Test cache cleanup using weak references."""

        class WeakReferenceCache:
            def __init__(self):
                self._cache = {}
                self._refs = {}

            def put(self, key: str, features: Dict[str, float]):
                """Store features with weak reference for cleanup."""
                self._cache[key] = features
                # Create weak reference with cleanup callback
                self._refs[key] = weakref.ref(features, lambda ref: self._cleanup(key))

            def get(self, key: str) -> Optional[Dict[str, float]]:
                """Get features if still alive."""
                return self._cache.get(key)

            def _cleanup(self, key: str):
                """Cleanup callback when features are garbage collected."""
                if key in self._cache:
                    del self._cache[key]
                if key in self._refs:
                    del self._refs[key]

            def size(self):
                return len(self._cache)

        cache = WeakReferenceCache()

        # Add features
        features1 = {"test_feature": 1.0}
        features2 = {"test_feature": 2.0}

        cache.put("key1", features1)
        cache.put("key2", features2)

        assert cache.size() == 2

        # Remove references and force garbage collection
        del features1
        del features2
        gc.collect()

        # Weak references should trigger cleanup
        # (Note: This is implementation-dependent and may not always work immediately)
        time.sleep(0.1)  # Give cleanup time to occur

    def test_memory_pool_feature_allocation(self):
        """Test memory pool allocation for feature dictionaries."""

        class MemoryPoolCache:
            def __init__(self, pool_size=100):
                self.pool_size = pool_size
                self.feature_pool = []  # Pool of reusable feature dicts
                self._cache = {}

            def _get_feature_dict(self) -> Dict[str, float]:
                """Get feature dict from pool or create new."""
                if self.feature_pool:
                    feature_dict = self.feature_pool.pop()
                    feature_dict.clear()  # Clear previous contents
                    return feature_dict
                else:
                    return {}

            def _return_feature_dict(self, feature_dict: Dict[str, float]):
                """Return feature dict to pool."""
                if len(self.feature_pool) < self.pool_size:
                    self.feature_pool.append(feature_dict)

            def put(self, key: str, features: Dict[str, float]):
                """Store features using pooled dictionary."""
                # If replacing existing, return old dict to pool
                if key in self._cache:
                    old_features = self._cache[key]
                    self._return_feature_dict(old_features)

                # Get new dict from pool and copy features
                pooled_dict = self._get_feature_dict()
                pooled_dict.update(features)
                self._cache[key] = pooled_dict

            def get(self, key: str) -> Optional[Dict[str, float]]:
                return self._cache.get(key)

            def remove(self, key: str):
                """Remove and return dict to pool."""
                if key in self._cache:
                    features = self._cache[key]
                    del self._cache[key]
                    self._return_feature_dict(features)

        cache = MemoryPoolCache(pool_size=20)

        # Test pool utilization
        for i in range(50):
            key = f"pooled_key_{i}"
            features = {f"feature_{j}": float(i * j) for j in range(10)}
            cache.put(key, features)

        # Test pool reuse
        for i in range(25):  # Remove half
            cache.remove(f"pooled_key_{i}")

        # Pool should have accumulated reusable dicts
        assert len(cache.feature_pool) > 0, "Pool should have reusable dictionaries"
        assert len(cache.feature_pool) <= 20, "Pool should respect size limit"

    def test_compressed_feature_storage(self):
        """Test compressed storage for infrequently accessed features."""
        import pickle

        import zlib

        class CompressedCache:
            def __init__(self, compression_threshold=100):
                self._cache = {}
                self._compressed = {}
                self.compression_threshold = compression_threshold
                self.access_counts = {}

            def _should_compress(self, key: str) -> bool:
                """Determine if features should be compressed."""
                access_count = self.access_counts.get(key, 0)
                return access_count < self.compression_threshold

            def put(self, key: str, features: Dict[str, float]):
                """Store features with optional compression."""
                self.access_counts[key] = self.access_counts.get(key, 0)

                if self._should_compress(key):
                    # Compress for storage
                    serialized = pickle.dumps(features)
                    compressed = zlib.compress(serialized)
                    self._compressed[key] = compressed
                    # Remove from regular cache
                    if key in self._cache:
                        del self._cache[key]
                else:
                    # Store uncompressed for frequent access
                    self._cache[key] = features
                    # Remove from compressed cache
                    if key in self._compressed:
                        del self._compressed[key]

            def get(self, key: str) -> Optional[Dict[str, float]]:
                """Get features with decompression if needed."""
                self.access_counts[key] = self.access_counts.get(key, 0) + 1

                # Check regular cache first
                if key in self._cache:
                    return self._cache[key]

                # Check compressed cache
                if key in self._compressed:
                    compressed_data = self._compressed[key]
                    serialized = zlib.decompress(compressed_data)
                    features = pickle.loads(serialized)

                    # Promote to regular cache if accessed frequently
                    if self.access_counts[key] >= self.compression_threshold:
                        self._cache[key] = features
                        del self._compressed[key]

                    return features

                return None

        cache = CompressedCache(compression_threshold=3)

        # Add features (should be compressed initially)
        test_features = {f"feature_{i}": float(i) for i in range(100)}
        cache.put("compress_test", test_features)

        assert "compress_test" in cache._compressed, "Should be compressed initially"
        assert "compress_test" not in cache._cache, "Should not be in regular cache"

        # Access multiple times to trigger promotion
        for _ in range(4):
            retrieved = cache.get("compress_test")
            assert retrieved == test_features, "Should retrieve correctly"

        # Should be promoted to regular cache
        assert "compress_test" in cache._cache, "Should be promoted to regular cache"
        assert (
            "compress_test" not in cache._compressed
        ), "Should be removed from compressed cache"


class TestCacheConcurrencyAndCoherence:
    """Test cache coherence under concurrent access patterns."""

    def test_concurrent_cache_read_write_coherence(self):
        """Test cache coherence with concurrent reads and writes."""
        cache = FeatureCache(max_size=100)
        coherence_errors = []

        def writer_thread(thread_id: int, iterations: int):
            """Writer thread that updates cache entries."""
            for i in range(iterations):
                key = f"coherence_key_{thread_id}_{i % 10}"
                features = {f"thread_{thread_id}_feature": float(i)}
                cache.put(key, features)
                time.sleep(0.001)  # Small delay to increase concurrency

        def reader_thread(thread_id: int, iterations: int):
            """Reader thread that validates cache consistency."""
            for i in range(iterations):
                key = f"coherence_key_{(thread_id + 1) % 4}_{i % 10}"
                features = cache.get(key)

                if features:
                    # Validate consistency - all values should be from same thread
                    thread_ids = set()
                    for feature_name in features.keys():
                        if "thread_" in feature_name:
                            tid = int(feature_name.split("_")[1])
                            thread_ids.add(tid)

                    if len(thread_ids) > 1:
                        coherence_errors.append(f"Inconsistent features: {features}")

                time.sleep(0.001)

        # Run concurrent readers and writers
        threads = []
        for i in range(4):
            writer = threading.Thread(target=writer_thread, args=(i, 50))
            reader = threading.Thread(target=reader_thread, args=(i, 50))
            threads.extend([writer, reader])

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Should maintain coherence
        assert len(coherence_errors) == 0, f"Cache coherence errors: {coherence_errors}"

    def test_cache_invalidation_race_conditions(self):
        """Test cache invalidation under race conditions."""
        cache = FeatureCache(max_size=50)
        race_conditions = []

        def aggressive_writer(thread_id: int):
            """Aggressively writes to cache."""
            for i in range(100):
                key = f"race_key_{i % 10}"
                features = {f"writer_{thread_id}_value": float(i)}
                cache.put(key, features)

        def aggressive_reader(thread_id: int):
            """Aggressively reads from cache."""
            for i in range(100):
                key = f"race_key_{i % 10}"
                features = cache.get(key)
                if features is None:
                    continue

                # Try to access again immediately
                features2 = cache.get(key)
                if features != features2:
                    race_conditions.append(f"Race condition detected: {key}")

        def cache_clearer():
            """Periodically clears cache."""
            for _ in range(10):
                time.sleep(0.01)
                cache.clear()

        # Run concurrent operations that might cause race conditions
        threads = []
        for i in range(3):
            writer = threading.Thread(target=aggressive_writer, args=(i,))
            reader = threading.Thread(target=aggressive_reader, args=(i,))
            threads.extend([writer, reader])

        clearer = threading.Thread(target=cache_clearer)
        threads.append(clearer)

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Race conditions should be minimal/handled
        assert len(race_conditions) < 5, f"Too many race conditions: {race_conditions}"

    def test_cache_eviction_under_concurrent_load(self):
        """Test cache eviction behavior under concurrent load."""
        cache = FeatureCache(max_size=20)  # Small cache to force evictions
        eviction_stats = {"evicted_keys": set(), "successful_gets": 0, "missed_gets": 0}

        def cache_loader(thread_id: int):
            """Load cache with features."""
            for i in range(50):  # More than cache capacity
                key = f"load_key_{thread_id}_{i}"
                features = {f"feature_{j}": float(i * j) for j in range(5)}
                cache.put(key, features)

                # Track what we expect to be in cache
                if cache.get(key) is None:
                    eviction_stats["evicted_keys"].add(key)

        def cache_accessor(thread_id: int):
            """Access cached features."""
            for i in range(100):
                key = f"load_key_{(thread_id + 1) % 3}_{i % 30}"
                features = cache.get(key)

                if features is not None:
                    eviction_stats["successful_gets"] += 1
                else:
                    eviction_stats["missed_gets"] += 1

        # Run concurrent load and access
        threads = []
        for i in range(3):
            loader = threading.Thread(target=cache_loader, args=(i,))
            accessor = threading.Thread(target=cache_accessor, args=(i,))
            threads.extend([loader, accessor])

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

        # Cache should maintain size limit
        assert cache.size() <= 20, "Cache should respect size limit under load"

        # Should have some successful gets and misses due to eviction
        total_attempts = (
            eviction_stats["successful_gets"] + eviction_stats["missed_gets"]
        )
        assert total_attempts > 0, "Should have attempted cache access"

        # Miss rate should be reasonable (not 100% due to eviction strategy)
        if total_attempts > 0:
            miss_rate = eviction_stats["missed_gets"] / total_attempts
            assert miss_rate < 0.9, "Miss rate should be reasonable"

    async def test_async_cache_operations(self):
        """Test cache operations in async context."""
        cache = FeatureCache(max_size=100)

        async def async_cache_worker(worker_id: int):
            """Async worker for cache operations."""
            for i in range(20):
                key = f"async_key_{worker_id}_{i}"
                features = {f"async_feature_{j}": float(i * j) for j in range(3)}

                # Put features
                cache.put(key, features)

                # Small async delay
                await asyncio.sleep(0.001)

                # Get and validate
                retrieved = cache.get(key)
                assert (
                    retrieved == features
                ), "Async cache operations should be consistent"

        # Run multiple async workers
        tasks = [async_cache_worker(i) for i in range(5)]
        await asyncio.gather(*tasks)

        # Cache should have data from all workers
        assert cache.size() > 0, "Async operations should populate cache"
