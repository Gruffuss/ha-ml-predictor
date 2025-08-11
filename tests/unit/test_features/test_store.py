"""
Unit tests for feature store caching and data management.

This module tests the FeatureStore and FeatureCache for caching behavior,
LRU eviction, training data generation, and performance management.
"""

import asyncio
from datetime import datetime, timedelta
import hashlib
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pandas as pd
import pytest

from src.core.config import SystemConfig
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.engineering import FeatureEngineeringEngine
from src.features.store import FeatureCache, FeatureRecord, FeatureStore


class TestFeatureRecord:
    """Test suite for FeatureRecord dataclass."""

    @pytest.fixture
    def sample_record(self):
        """Create a sample feature record."""
        return FeatureRecord(
            room_id="living_room",
            target_time=datetime(2024, 1, 15, 15, 0, 0),
            features={"feature_1": 1.0, "feature_2": 2.0},
            extraction_time=datetime(2024, 1, 15, 15, 1, 0),
            lookback_hours=24,
            feature_types=["temporal", "sequential"],
            data_hash="abc123",
        )

    def test_to_dict(self, sample_record):
        """Test conversion to dictionary."""
        record_dict = sample_record.to_dict()

        assert isinstance(record_dict, dict)
        assert record_dict["room_id"] == "living_room"
        assert record_dict["target_time"] == "2024-01-15T15:00:00"
        assert record_dict["extraction_time"] == "2024-01-15T15:01:00"
        assert record_dict["features"] == {"feature_1": 1.0, "feature_2": 2.0}
        assert record_dict["lookback_hours"] == 24
        assert record_dict["feature_types"] == ["temporal", "sequential"]
        assert record_dict["data_hash"] == "abc123"

    def test_from_dict(self, sample_record):
        """Test creation from dictionary."""
        record_dict = sample_record.to_dict()
        reconstructed_record = FeatureRecord.from_dict(record_dict)

        assert reconstructed_record.room_id == sample_record.room_id
        assert reconstructed_record.target_time == sample_record.target_time
        assert reconstructed_record.features == sample_record.features
        assert reconstructed_record.extraction_time == sample_record.extraction_time
        assert reconstructed_record.lookback_hours == sample_record.lookback_hours
        assert reconstructed_record.feature_types == sample_record.feature_types
        assert reconstructed_record.data_hash == sample_record.data_hash

    def test_is_valid_fresh(self, sample_record):
        """Test validity check for fresh record."""
        # Record extracted 1 minute ago
        with patch("src.features.store.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 15, 2, 0)

            # Should be valid (2 minutes old, max age 24 hours)
            assert sample_record.is_valid(max_age_hours=24) is True

    def test_is_valid_old(self, sample_record):
        """Test validity check for old record."""
        with patch("src.features.store.datetime") as mock_datetime:
            # Record is now 25 hours old
            mock_datetime.utcnow.return_value = datetime(2024, 1, 16, 16, 1, 0)

            # Should be invalid (25 hours old, max age 24 hours)
            assert sample_record.is_valid(max_age_hours=24) is False

    def test_is_valid_custom_max_age(self, sample_record):
        """Test validity check with custom max age."""
        with patch("src.features.store.datetime") as mock_datetime:
            # Record is 2 hours old
            mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 17, 1, 0)

            # Should be invalid with 1-hour max age
            assert sample_record.is_valid(max_age_hours=1) is False

            # Should be valid with 3-hour max age
            assert sample_record.is_valid(max_age_hours=3) is True


class TestFeatureCache:
    """Test suite for FeatureCache."""

    @pytest.fixture
    def cache(self):
        """Create a feature cache instance."""
        return FeatureCache(max_size=5)

    @pytest.fixture
    def sample_params(self):
        """Sample cache parameters."""
        return {
            "room_id": "living_room",
            "target_time": datetime(2024, 1, 15, 15, 0, 0),
            "lookback_hours": 24,
            "feature_types": ["temporal", "sequential"],
            "features": {"feature_1": 1.0, "feature_2": 2.0},
            "data_hash": "abc123",
        }

    def test_make_key(self, cache):
        """Test cache key generation."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        lookback_hours = 24
        feature_types = ["temporal", "sequential"]

        key1 = cache._make_key(room_id, target_time, lookback_hours, feature_types)
        key2 = cache._make_key(room_id, target_time, lookback_hours, feature_types)

        # Keys should be identical for same parameters
        assert key1 == key2
        assert isinstance(key1, str)
        assert len(key1) == 32  # MD5 hash length

        # Different parameters should produce different keys
        key3 = cache._make_key(
            room_id, target_time, 12, feature_types
        )  # Different lookback
        assert key1 != key3

    def test_put_and_get_hit(self, cache, sample_params):
        """Test cache put and successful get (cache hit)."""
        cache.put(**sample_params)

        retrieved_features = cache.get(
            sample_params["room_id"],
            sample_params["target_time"],
            sample_params["lookback_hours"],
            sample_params["feature_types"],
            max_age_hours=24,
        )

        assert retrieved_features == sample_params["features"]
        assert cache.hit_count == 1
        assert cache.miss_count == 0

    def test_get_miss(self, cache):
        """Test cache miss when item not in cache."""
        retrieved_features = cache.get(
            "nonexistent_room",
            datetime(2024, 1, 15, 15, 0, 0),
            24,
            ["temporal"],
            max_age_hours=24,
        )

        assert retrieved_features is None
        assert cache.hit_count == 0
        assert cache.miss_count == 1

    def test_get_expired_item(self, cache, sample_params):
        """Test cache miss when item is expired."""
        cache.put(**sample_params)

        # Mock time to make record appear old
        with patch("src.features.store.datetime") as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(
                2024, 1, 16, 16, 0, 0
            )  # 25 hours later

            retrieved_features = cache.get(
                sample_params["room_id"],
                sample_params["target_time"],
                sample_params["lookback_hours"],
                sample_params["feature_types"],
                max_age_hours=24,  # Max age 24 hours
            )

        assert retrieved_features is None
        assert cache.miss_count == 1
        # Cache should have removed expired item
        assert len(cache.cache) == 0

    def test_lru_eviction(self, cache):
        """Test LRU eviction when cache exceeds max size."""
        # Fill cache to max capacity (5 items)
        for i in range(5):
            cache.put(
                room_id=f"room_{i}",
                target_time=datetime(2024, 1, 15, 15, 0, 0),
                lookback_hours=24,
                feature_types=["temporal"],
                features={"feature": float(i)},
                data_hash=f"hash_{i}",
            )

        assert len(cache.cache) == 5

        # Add one more item (should trigger eviction)
        cache.put(
            room_id="room_new",
            target_time=datetime(2024, 1, 15, 15, 0, 0),
            lookback_hours=24,
            feature_types=["temporal"],
            features={"feature": 999.0},
            data_hash="hash_new",
        )

        # Cache size should still be 5
        assert len(cache.cache) == 5

        # First item (room_0) should be evicted
        oldest_features = cache.get(
            "room_0", datetime(2024, 1, 15, 15, 0, 0), 24, ["temporal"]
        )
        assert oldest_features is None

        # Newest item should be present
        newest_features = cache.get(
            "room_new", datetime(2024, 1, 15, 15, 0, 0), 24, ["temporal"]
        )
        assert newest_features == {"feature": 999.0}

    def test_cache_move_to_end(self, cache):
        """Test that accessed items are moved to end (most recently used)."""
        # Add items
        for i in range(3):
            cache.put(
                room_id=f"room_{i}",
                target_time=datetime(2024, 1, 15, 15, 0, 0),
                lookback_hours=24,
                feature_types=["temporal"],
                features={"feature": float(i)},
                data_hash=f"hash_{i}",
            )

        # Access first item (should move it to end)
        cache.get("room_0", datetime(2024, 1, 15, 15, 0, 0), 24, ["temporal"])

        # Add more items to trigger eviction
        for i in range(3, 6):
            cache.put(
                room_id=f"room_{i}",
                target_time=datetime(2024, 1, 15, 15, 0, 0),
                lookback_hours=24,
                feature_types=["temporal"],
                features={"feature": float(i)},
                data_hash=f"hash_{i}",
            )

        # room_0 should still be present (was moved to end when accessed)
        features = cache.get(
            "room_0", datetime(2024, 1, 15, 15, 0, 0), 24, ["temporal"]
        )
        assert features == {"feature": 0.0}

    def test_clear(self, cache, sample_params):
        """Test cache clearing."""
        cache.put(**sample_params)
        assert len(cache.cache) == 1
        assert cache.hit_count == 0

        # Access to create stats
        cache.get(
            sample_params["room_id"],
            sample_params["target_time"],
            sample_params["lookback_hours"],
            sample_params["feature_types"],
        )
        assert cache.hit_count == 1

        cache.clear()

        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_get_stats(self, cache):
        """Test cache statistics."""
        # Initially empty
        stats = cache.get_stats()
        assert stats["size"] == 0
        assert stats["max_size"] == 5
        assert stats["hit_count"] == 0
        assert stats["miss_count"] == 0
        assert stats["hit_rate"] == 0.0

        # Add item and test hit/miss
        cache.put(
            "room_1",
            datetime.now(),
            24,
            ["temporal"],
            {"feature": 1.0},
            "hash_1",
        )

        # Hit
        cache.get("room_1", datetime.now(), 24, ["temporal"])
        # Miss
        cache.get("room_2", datetime.now(), 24, ["temporal"])

        stats = cache.get_stats()
        assert stats["size"] == 1
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert stats["hit_rate"] == 0.5

    def test_feature_type_order_independence(self, cache):
        """Test that feature type order doesn't affect cache keys."""
        params1 = ["temporal", "sequential", "contextual"]
        params2 = ["contextual", "temporal", "sequential"]

        key1 = cache._make_key("room", datetime.now(), 24, params1)
        key2 = cache._make_key("room", datetime.now(), 24, params2)

        # Keys should be identical (feature types are sorted)
        assert key1 == key2


class TestFeatureStore:
    """Test suite for FeatureStore."""

    @pytest.fixture
    def mock_config(self):
        """Create mock system configuration."""
        config = Mock(spec=SystemConfig)
        config.rooms = {"living_room": Mock(), "kitchen": Mock()}
        return config

    @pytest.fixture
    def store(self, mock_config):
        """Create feature store instance."""
        return FeatureStore(config=mock_config, cache_size=10, enable_persistence=False)

    @pytest.fixture
    def store_with_persistence(self, mock_config):
        """Create feature store with persistence enabled."""
        return FeatureStore(config=mock_config, cache_size=10, enable_persistence=True)

    @pytest.fixture
    def sample_events(self) -> List[SensorEvent]:
        """Create sample sensor events."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.room_id = "living_room"
            event.state = "on" if i % 2 == 0 else "of"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        return events

    @pytest.fixture
    def sample_room_states(self) -> List[RoomState]:
        """Create sample room states."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        states = []

        for i in range(2):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 15)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8
            states.append(state)

        return states

    @pytest.mark.asyncio
    async def test_initialize_no_persistence(self, store):
        """Test initialization without persistence."""
        await store.initialize()

        assert store.db_manager is None
        assert store.enable_persistence is False

    @pytest.mark.asyncio
    async def test_initialize_with_persistence_success(self, store_with_persistence):
        """Test initialization with successful persistence setup."""
        with patch("src.features.store.get_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            await store_with_persistence.initialize()

            assert store_with_persistence.db_manager == mock_db_manager
            assert store_with_persistence.enable_persistence is True

    @pytest.mark.asyncio
    async def test_initialize_with_persistence_failure(self, store_with_persistence):
        """Test initialization with persistence setup failure."""
        with patch("src.features.store.get_database_manager") as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")

            await store_with_persistence.initialize()

            assert store_with_persistence.db_manager is None
            assert store_with_persistence.enable_persistence is False

    @pytest.mark.asyncio
    async def test_get_features_cache_hit(self, store):
        """Test getting features with cache hit."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        expected_features = {"feature_1": 1.0, "feature_2": 2.0}

        # Put features in cache
        store.cache.put(
            "living_room",
            target_time,
            24,
            ["temporal", "sequential", "contextual"],
            expected_features,
            "test_hash",
        )

        features = await store.get_features("living_room", target_time)

        assert features == expected_features
        assert store.stats["cache_hits"] == 1
        assert store.stats["cache_misses"] == 0

    @pytest.mark.asyncio
    async def test_get_features_cache_miss_compute(self, store):
        """Test getting features with cache miss and computation."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        expected_features = {"feature_1": 1.0, "feature_2": 2.0}

        with patch.object(
            store, "_compute_features", return_value=expected_features
        ) as mock_compute:
            features = await store.get_features("living_room", target_time)

            assert features == expected_features
            assert store.stats["cache_hits"] == 0
            assert store.stats["cache_misses"] == 1
            mock_compute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_features_force_recompute(self, store):
        """Test forced recomputation bypassing cache."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        cached_features = {"cached": 1.0}
        computed_features = {"computed": 2.0}

        # Put features in cache
        store.cache.put(
            "living_room",
            target_time,
            24,
            ["temporal", "sequential", "contextual"],
            cached_features,
            "test_hash",
        )

        with patch.object(
            store, "_compute_features", return_value=computed_features
        ) as mock_compute:
            features = await store.get_features(
                "living_room", target_time, force_recompute=True
            )

            assert features == computed_features  # Should get computed, not cached
            mock_compute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_batch_features(self, store):
        """Test batch feature retrieval."""
        requests = [
            ("living_room", datetime(2024, 1, 15, 15, 0, 0)),
            ("kitchen", datetime(2024, 1, 15, 15, 30, 0)),
        ]

        expected_results = [
            {"living_room_feature": 1.0},
            {"kitchen_feature": 2.0},
        ]

        with patch.object(store, "get_features", side_effect=expected_results):
            results = await store.get_batch_features(requests)

            assert results == expected_results
            assert store.stats["batch_operations"] == 1

    @pytest.mark.asyncio
    async def test_get_batch_features_with_exception(self, store):
        """Test batch feature retrieval with exception handling."""
        requests = [
            ("living_room", datetime(2024, 1, 15, 15, 0, 0)),
            ("kitchen", datetime(2024, 1, 15, 15, 30, 0)),
        ]

        with patch.object(store, "get_features") as mock_get_features:
            # First call succeeds, second fails
            mock_get_features.side_effect = [
                {"success": 1.0},
                Exception("Feature extraction failed"),
            ]

            results = await store.get_batch_features(requests)

            assert len(results) == 2
            assert results[0] == {"success": 1.0}
            assert isinstance(results[1], dict)  # Should be default features

    @pytest.mark.asyncio
    async def test_compute_training_data(self, store):
        """Test training data generation."""
        start_date = datetime(2024, 1, 15, 10, 0, 0)
        end_date = datetime(2024, 1, 15, 12, 0, 0)
        interval_minutes = 30

        mock_features = [
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 1.5, "feature_2": 2.5},
            {"feature_1": 2.0, "feature_2": 3.0},
            {"feature_1": 2.5, "feature_2": 3.5},
            {"feature_1": 3.0, "feature_2": 4.0},
        ]

        with (
            patch.object(store, "get_batch_features", return_value=mock_features),
            patch.object(
                store.feature_engine, "create_feature_dataframe"
            ) as mock_create_df,
        ):

            mock_df = pd.DataFrame(mock_features)
            mock_create_df.return_value = mock_df

            features_df, targets_df = await store.compute_training_data(
                "living_room", start_date, end_date, interval_minutes
            )

            # Should generate 5 time points (10:00, 10:30, 11:00, 11:30, 12:00)
            expected_time_points = 5
            assert len(features_df) == expected_time_points
            assert len(targets_df) == expected_time_points

            # Targets should have required columns
            required_target_columns = [
                "target_time",
                "next_transition_time",
                "transition_type",
                "room_id",
            ]
            for col in required_target_columns:
                assert col in targets_df.columns

    @pytest.mark.asyncio
    async def test_compute_features(self, store, sample_events, sample_room_states):
        """Test feature computation."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        expected_features = {"computed_feature": 1.0}

        with (
            patch.object(
                store,
                "_get_data_for_features",
                return_value=(sample_events, sample_room_states),
            ) as mock_get_data,
            patch.object(
                store.feature_engine,
                "extract_features",
                return_value=expected_features,
            ) as mock_extract,
        ):

            features = await store._compute_features(
                "living_room", target_time, 24, ["temporal"]
            )

            assert features == expected_features
            assert store.stats["feature_computations"] == 1
            mock_get_data.assert_called_once_with("living_room", target_time, 24)
            mock_extract.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_data_for_features_no_db(self, store):
        """Test data retrieval when no database manager."""
        events, room_states = await store._get_data_for_features(
            "living_room", datetime.now(), 24
        )

        assert events == []
        assert room_states == []

    @pytest.mark.asyncio
    async def test_get_data_for_features_with_db(self, store):
        """Test data retrieval with database manager."""
        store.db_manager = AsyncMock()
        mock_session = AsyncMock()
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)
        store.db_manager.get_session.return_value = mock_session

        # Mock database query results
        mock_events_result = Mock()
        mock_events_result.scalars.return_value.all.return_value = [
            Mock(),
            Mock(),
        ]

        mock_states_result = Mock()
        mock_states_result.scalars.return_value.all.return_value = [Mock()]

        mock_session.execute.side_effect = [
            mock_events_result,
            mock_states_result,
        ]

        events, room_states = await store._get_data_for_features(
            "living_room", datetime.now(), 24
        )

        assert len(events) == 2
        assert len(room_states) == 1
        assert store.stats["database_queries"] == 1

    def test_compute_data_hash(self, store):
        """Test data hash computation."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        lookback_hours = 24

        hash1 = store._compute_data_hash(room_id, target_time, lookback_hours)
        hash2 = store._compute_data_hash(room_id, target_time, lookback_hours)

        # Same parameters should produce same hash
        assert hash1 == hash2
        assert isinstance(hash1, str)
        assert len(hash1) == 32  # MD5 hash length

        # Different parameters should produce different hash
        hash3 = store._compute_data_hash(room_id, target_time, 12)  # Different lookback
        assert hash1 != hash3

    def test_get_stats(self, store):
        """Test statistics retrieval."""
        with (
            patch.object(store.cache, "get_stats", return_value={"cache_stat": 1}),
            patch.object(
                store.feature_engine,
                "get_extraction_stats",
                return_value={"engine_stat": 2},
            ),
        ):

            stats = store.get_stats()

            assert "feature_store" in stats
            assert "cache" in stats
            assert "engine" in stats
            assert stats["cache"]["cache_stat"] == 1
            assert stats["engine"]["engine_stat"] == 2

    def test_clear_cache(self, store):
        """Test cache clearing."""
        with (
            patch.object(store.cache, "clear") as mock_cache_clear,
            patch.object(store.feature_engine, "clear_caches") as mock_engine_clear,
        ):

            store.clear_cache()

            mock_cache_clear.assert_called_once()
            mock_engine_clear.assert_called_once()

    def test_reset_stats(self, store):
        """Test statistics reset."""
        # Modify stats
        store.stats["total_requests"] = 10
        store.stats["cache_hits"] = 5

        with patch.object(store.feature_engine, "reset_stats") as mock_engine_reset:
            store.reset_stats()

            assert store.stats["total_requests"] == 0
            assert store.stats["cache_hits"] == 0
            mock_engine_reset.assert_called_once()

    @pytest.mark.asyncio
    async def test_health_check(self, store):
        """Test health check functionality."""
        with patch.object(
            store.feature_engine, "validate_configuration"
        ) as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "warnings": [],
                "errors": [],
            }

            health = await store.health_check()

            assert health["status"] == "healthy"
            assert "components" in health
            assert "feature_engine" in health["components"]
            assert "cache" in health["components"]

    @pytest.mark.asyncio
    async def test_health_check_with_db(self, store):
        """Test health check with database."""
        store.enable_persistence = True
        store.db_manager = AsyncMock()
        store.db_manager.health_check.return_value = {"status": "healthy"}

        with patch.object(
            store.feature_engine, "validate_configuration"
        ) as mock_validate:
            mock_validate.return_value = {
                "valid": True,
                "warnings": [],
                "errors": [],
            }

            health = await store.health_check()

            assert "database" in health["components"]

    @pytest.mark.asyncio
    async def test_context_manager(self, store):
        """Test async context manager functionality."""
        with patch.object(store, "initialize") as mock_init:
            async with store as context_store:
                assert context_store == store
                mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_performance_large_batch(self, store):
        """Test performance with large batch requests."""
        import time

        # Create large batch of requests
        requests = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        for i in range(100):
            requests.append(
                (
                    f"room_{i % 5}",
                    base_time + timedelta(minutes=i),
                )  # 5 different rooms
            )

        with patch.object(store, "get_features") as mock_get_features:
            mock_get_features.return_value = {"test_feature": 1.0}

            start_time = time.time()
            results = await store.get_batch_features(requests)
            end_time = time.time()

            # Should complete in reasonable time
            execution_time = end_time - start_time
            assert execution_time < 5.0  # Less than 5 seconds

            assert len(results) == 100
            assert all(isinstance(result, dict) for result in results)

    @pytest.mark.asyncio
    async def test_memory_efficiency_caching(self, store):
        """Test that caching doesn't cause memory leaks."""
        import gc
        import sys

        gc.collect()
        initial_size = sys.getsizeof(store.cache)

        # Add many items to cache (more than cache size to test eviction)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        for i in range(50):  # Cache size is 10, so many will be evicted
            store.cache.put(
                f"room_{i}",
                target_time + timedelta(minutes=i),
                24,
                ["temporal"],
                {"feature": float(i)},
                f"hash_{i}",
            )

        gc.collect()
        final_size = sys.getsizeof(store.cache)

        # Cache should be limited in size
        assert len(store.cache.cache) <= store.cache.max_size
        # Memory growth should be bounded
        assert final_size - initial_size < 10000  # Less than 10KB growth

    @pytest.mark.asyncio
    async def test_concurrent_cache_operations(self, store):
        """Test thread safety of cache operations."""

        async def cache_operation(room_id: str):
            target_time = datetime(2024, 1, 15, 15, 0, 0)

            # Try to get (likely miss)
            features = await store.get_features(room_id, target_time)
            return features

        with patch.object(store, "_compute_features") as mock_compute:
            mock_compute.return_value = {"concurrent_feature": 1.0}

            # Run concurrent operations
            tasks = [cache_operation(f"room_{i}") for i in range(10)]
            results = await asyncio.gather(*tasks)

            # All operations should succeed
            assert len(results) == 10
            assert all(isinstance(result, dict) for result in results)

    @pytest.mark.parametrize("cache_size", [1, 5, 50, 100])
    def test_cache_size_limits(self, mock_config, cache_size):
        """Test cache behavior with different size limits."""
        store = FeatureStore(config=mock_config, cache_size=cache_size)

        # Add more items than cache size
        for i in range(cache_size * 2):
            store.cache.put(
                f"room_{i}",
                datetime(2024, 1, 15, 15, 0, 0),
                24,
                ["temporal"],
                {"feature": float(i)},
                f"hash_{i}",
            )

        # Cache should not exceed max size
        assert len(store.cache.cache) <= cache_size
        assert store.cache.max_size == cache_size

    @pytest.mark.asyncio
    async def test_error_propagation(self, store):
        """Test that errors are properly propagated and handled."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        with patch.object(store, "_compute_features") as mock_compute:
            mock_compute.side_effect = FeatureExtractionError("Computation failed")

            with pytest.raises(FeatureExtractionError):
                await store.get_features("living_room", target_time)

    def test_feature_type_parameter_handling(self, store):
        """Test handling of different feature type parameters."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Test None (should default to all types)
        cache_key1 = store.cache._make_key(
            "room", target_time, 24, ["temporal", "sequential", "contextual"]
        )

        # Test explicit all types
        cache_key2 = store.cache._make_key(
            "room", target_time, 24, ["temporal", "sequential", "contextual"]
        )

        assert cache_key1 == cache_key2

        # Test subset
        cache_key3 = store.cache._make_key("room", target_time, 24, ["temporal"])

        assert cache_key1 != cache_key3
