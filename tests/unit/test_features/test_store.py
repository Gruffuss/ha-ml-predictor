"""
Unit tests for feature store caching and data management.

This module tests the FeatureStore and FeatureCache for caching behavior,
LRU eviction, training data generation, and performance management.
"""

import asyncio
from datetime import datetime, timedelta
import gc
import hashlib
import sys
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
        return FeatureCache(max_size=3)  # Small cache for testing

    @pytest.fixture
    def sample_records(self):
        """Create sample feature records."""
        base_time = datetime(2024, 1, 15, 15, 0, 0)
        records = []

        for i in range(5):
            record = FeatureRecord(
                room_id=f"room_{i}",
                target_time=base_time + timedelta(hours=i),
                features={"feature": float(i)},
                extraction_time=base_time + timedelta(hours=i, minutes=1),
                lookback_hours=24,
                feature_types=["temporal"],
                data_hash=f"hash_{i}",
            )
            records.append(record)

        return records

    def test_cache_initialization(self, cache):
        """Test cache initialization."""
        assert cache.max_size == 3
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_put_and_get(self, cache, sample_records):
        """Test basic cache put and get operations."""
        record = sample_records[0]
        key = "test_key"

        # Put record in cache
        cache.put(key, record)
        assert len(cache) == 1
        assert cache.hits == 0
        assert cache.misses == 0

        # Get record from cache
        retrieved = cache.get(key)
        assert retrieved == record
        assert cache.hits == 1
        assert cache.misses == 0

        # Test cache miss
        missing = cache.get("nonexistent")
        assert missing is None
        assert cache.hits == 1
        assert cache.misses == 1

    def test_cache_lru_eviction(self, cache, sample_records):
        """Test LRU eviction when cache exceeds max size."""
        # Fill cache to capacity
        for i in range(3):
            cache.put(f"key_{i}", sample_records[i])

        assert len(cache) == 3

        # Access key_1 to make it recently used
        cache.get("key_1")

        # Add another item, should evict least recently used (key_0)
        cache.put("key_3", sample_records[3])
        assert len(cache) == 3

        # key_0 should be evicted, others should remain
        assert cache.get("key_0") is None
        assert cache.get("key_1") is not None
        assert cache.get("key_2") is not None
        assert cache.get("key_3") is not None

    def test_cache_update_existing(self, cache, sample_records):
        """Test updating existing cache entry."""
        key = "test_key"
        cache.put(key, sample_records[0])

        # Update with new record
        cache.put(key, sample_records[1])
        assert len(cache) == 1

        retrieved = cache.get(key)
        assert retrieved == sample_records[1]

    def test_cache_remove(self, cache, sample_records):
        """Test removing items from cache."""
        cache.put("key_1", sample_records[0])
        cache.put("key_2", sample_records[1])
        assert len(cache) == 2

        # Remove one item
        removed = cache.remove("key_1")
        assert removed == sample_records[0]
        assert len(cache) == 1

        # Try to remove non-existent item
        removed = cache.remove("nonexistent")
        assert removed is None
        assert len(cache) == 1

    def test_cache_clear(self, cache, sample_records):
        """Test clearing the cache."""
        for i in range(3):
            cache.put(f"key_{i}", sample_records[i])

        assert len(cache) == 3

        cache.clear()
        assert len(cache) == 0
        assert cache.hits == 0
        assert cache.misses == 0

    def test_cache_contains(self, cache, sample_records):
        """Test cache membership testing."""
        key = "test_key"
        assert key not in cache

        cache.put(key, sample_records[0])
        assert key in cache

        cache.remove(key)
        assert key not in cache

    def test_cache_keys_and_values(self, cache, sample_records):
        """Test getting all keys and values."""
        keys = ["key_1", "key_2", "key_3"]
        records = sample_records[:3]

        for key, record in zip(keys, records):
            cache.put(key, record)

        cache_keys = list(cache.keys())
        cache_values = list(cache.values())

        assert len(cache_keys) == 3
        assert len(cache_values) == 3
        for key in keys:
            assert key in cache_keys
        for record in records:
            assert record in cache_values

    def test_cache_items(self, cache, sample_records):
        """Test getting all cache items."""
        cache.put("key_1", sample_records[0])
        cache.put("key_2", sample_records[1])

        items = list(cache.items())
        assert len(items) == 2

        keys, values = zip(*items)
        assert "key_1" in keys
        assert "key_2" in keys
        assert sample_records[0] in values
        assert sample_records[1] in values

    def test_cache_statistics(self, cache, sample_records):
        """Test cache hit/miss statistics."""
        cache.put("key_1", sample_records[0])

        # Multiple hits
        for _ in range(5):
            cache.get("key_1")
        assert cache.hits == 5

        # Multiple misses
        for _ in range(3):
            cache.get("nonexistent")
        assert cache.misses == 3

        # Test hit rate
        assert cache.hit_rate == 5 / 8  # 5 hits out of 8 total

    def test_cache_expired_records(self, cache):
        """Test handling of expired records."""
        # Create record that's already expired
        old_record = FeatureRecord(
            room_id="room",
            target_time=datetime(2024, 1, 15, 15, 0, 0),
            features={"feature": 1.0},
            extraction_time=datetime(2024, 1, 14, 15, 0, 0),  # 24 hours ago
            lookback_hours=24,
            feature_types=["temporal"],
            data_hash="hash",
        )

        cache.put("old_key", old_record)

        # Should not return expired record
        with patch("src.features.store.datetime") as mock_datetime:
            # Current time is 25 hours after extraction
            mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 16, 0, 0)

            result = cache.get("old_key", max_age_hours=24)
            assert result is None

    def test_cache_memory_cleanup(self, cache):
        """Test cache memory cleanup operations."""
        # Fill cache with records
        for i in range(10):
            record = FeatureRecord(
                room_id=f"room_{i}",
                target_time=datetime.now(),
                features={"large_data": [1.0] * 1000},  # Large data
                extraction_time=datetime.now(),
                lookback_hours=24,
                feature_types=["temporal"],
                data_hash=f"hash_{i}",
            )
            cache.put(f"key_{i}", record)

        # Should only keep max_size items due to LRU eviction
        assert len(cache) == cache.max_size

        # Clear and force garbage collection
        cache.clear()
        gc.collect()

        assert len(cache) == 0


class TestFeatureStore:
    """Test suite for FeatureStore."""

    @pytest.fixture
    def mock_feature_engine(self):
        """Create mock feature engineering engine."""
        engine = Mock(spec=FeatureEngineeringEngine)
        engine.extract_features.return_value = {"test_feature": 1.0}
        engine.create_feature_dataframe.return_value = pd.DataFrame(
            {"feature": [1.0, 2.0, 3.0]}
        )
        return engine

    @pytest.fixture
    def store(self, mock_feature_engine):
        """Create feature store instance."""
        return FeatureStore(
            feature_engine=mock_feature_engine,
            cache_size=10,
            default_lookback_hours=24,
        )

    @pytest.fixture
    def sample_events(self):
        """Create sample sensor events."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.room_id = "living_room"
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        return events

    @pytest.fixture
    def sample_room_states(self):
        """Create sample room states."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        states = []

        for i in range(3):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 20)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8
            states.append(state)

        return states

    def test_store_initialization(self, store, mock_feature_engine):
        """Test feature store initialization."""
        assert store.feature_engine == mock_feature_engine
        assert store.default_lookback_hours == 24
        assert isinstance(store.cache, FeatureCache)
        assert store.cache.max_size == 10

    @pytest.mark.asyncio
    async def test_get_features_from_cache(self, store):
        """Test retrieving features from cache."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        expected_features = {"cached_feature": 1.0}

        # Put record in cache
        record = FeatureRecord(
            room_id=room_id,
            target_time=target_time,
            features=expected_features,
            extraction_time=datetime.now(),
            lookback_hours=24,
            feature_types=["temporal"],
            data_hash="hash",
        )
        cache_key = store._get_cache_key(room_id, target_time, 24, ["temporal"])
        store.cache.put(cache_key, record)

        features = await store.get_features(room_id, target_time)

        assert features == expected_features
        assert store.stats["cache_hits"] == 1
        assert store.cache.hits == 1

    @pytest.mark.asyncio
    async def test_get_features_cache_miss(self, store):
        """Test retrieving features when cache miss occurs."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        expected_features = {"computed_feature": 1.0}

        with patch.object(
            store, "_compute_features", return_value=expected_features
        ) as mock_compute:

            features = await store.get_features(room_id, target_time)

            assert features == expected_features
            assert store.stats["cache_misses"] == 1
            assert store.stats["feature_computations"] == 1
            mock_compute.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_batch_features_success(self, store):
        """Test batch feature retrieval."""
        requests = [
            ("living_room", datetime(2024, 1, 15, 15, 0, 0)),
            ("kitchen", datetime(2024, 1, 15, 15, 30, 0)),
        ]

        with patch.object(store, "get_features") as mock_get_features:
            mock_get_features.side_effect = [
                {"feature_1": 1.0},
                {"feature_2": 2.0},
            ]

            results = await store.get_batch_features(requests)

            assert len(results) == 2
            assert results[0] == {"feature_1": 1.0}
            assert results[1] == {"feature_2": 2.0}
            assert mock_get_features.call_count == 2

    @pytest.mark.asyncio
    async def test_get_batch_features_with_exceptions(self, store):
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
        # Create proper AsyncMock for database manager
        store.db_manager = AsyncMock()

        # Create proper async context manager mock for session
        mock_session = AsyncMock()

        # Properly configure the async context manager
        async def mock_get_session():
            return mock_session

        store.db_manager.get_session.return_value.__aenter__ = AsyncMock(
            return_value=mock_session
        )
        store.db_manager.get_session.return_value.__aexit__ = AsyncMock(
            return_value=None
        )

        # Mock database query results with proper structure
        mock_events = [Mock(spec=SensorEvent), Mock(spec=SensorEvent)]
        mock_room_states = [Mock(spec=RoomState)]

        # Create mock results that return the expected data structure
        mock_events_result = Mock()
        mock_events_result.scalars.return_value.all.return_value = mock_events

        mock_states_result = Mock()
        mock_states_result.scalars.return_value.all.return_value = mock_room_states

        # Configure session execute to return these results
        mock_session.execute.side_effect = [
            mock_events_result,
            mock_states_result,
        ]

        # Test the method
        events, room_states = await store._get_data_for_features(
            "living_room", datetime.now(), 24
        )

        # Verify results
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

    def test_get_cache_key(self, store):
        """Test cache key generation."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        lookback_hours = 24
        feature_types = ["temporal", "sequential"]

        key1 = store._get_cache_key(room_id, target_time, lookback_hours, feature_types)
        key2 = store._get_cache_key(room_id, target_time, lookback_hours, feature_types)

        # Same parameters should produce same key
        assert key1 == key2
        assert isinstance(key1, str)

        # Different parameters should produce different key
        key3 = store._get_cache_key(room_id, target_time, 12, feature_types)
        assert key1 != key3

    def test_get_default_features(self, store):
        """Test default features generation."""
        # Mock the feature engine's get_default_features method
        expected_defaults = {"default_feature": 0.0}

        with patch.object(
            store.feature_engine, "get_default_features", return_value=expected_defaults
        ) as mock_defaults:
            defaults = store._get_default_features()
            assert defaults == expected_defaults
            mock_defaults.assert_called_once()

    def test_store_statistics(self, store):
        """Test store statistics tracking."""
        initial_stats = store.get_stats()

        expected_keys = [
            "cache_hits",
            "cache_misses",
            "feature_computations",
            "database_queries",
            "cache_size",
            "total_requests",
        ]

        for key in expected_keys:
            assert key in initial_stats
            assert isinstance(initial_stats[key], (int, float))

    def test_clear_cache(self, store):
        """Test cache clearing functionality."""
        # Add some items to cache
        record = FeatureRecord(
            room_id="test",
            target_time=datetime.now(),
            features={"test": 1.0},
            extraction_time=datetime.now(),
            lookback_hours=24,
            feature_types=["temporal"],
            data_hash="test_hash",
        )

        store.cache.put("test_key", record)
        assert len(store.cache) == 1

        # Clear cache
        store.clear_cache()
        assert len(store.cache) == 0

    @pytest.mark.asyncio
    async def test_feature_extraction_error_handling(self, store):
        """Test error handling during feature extraction."""
        with patch.object(
            store, "_get_data_for_features", side_effect=Exception("Database error")
        ):
            # Should return default features on error
            features = await store._compute_features(
                "living_room", datetime.now(), 24, ["temporal"]
            )

            # Should be a dict (default features)
            assert isinstance(features, dict)

    def test_memory_usage_optimization(self, store):
        """Test memory usage optimization features."""
        # Test that cache doesn't grow beyond max size
        for i in range(20):  # Add more than cache max_size
            record = FeatureRecord(
                room_id=f"room_{i}",
                target_time=datetime.now() + timedelta(hours=i),
                features={"feature": float(i)},
                extraction_time=datetime.now(),
                lookback_hours=24,
                feature_types=["temporal"],
                data_hash=f"hash_{i}",
            )

            cache_key = store._get_cache_key(
                record.room_id,
                record.target_time,
                record.lookback_hours,
                record.feature_types,
            )
            store.cache.put(cache_key, record)

        # Cache should not exceed max size
        assert len(store.cache) <= store.cache.max_size

    @pytest.mark.asyncio
    async def test_concurrent_feature_requests(self, store):
        """Test handling of concurrent feature requests."""
        room_id = "living_room"
        target_times = [
            datetime(2024, 1, 15, 15, 0, 0),
            datetime(2024, 1, 15, 15, 30, 0),
            datetime(2024, 1, 15, 16, 0, 0),
        ]

        with patch.object(
            store, "_compute_features", return_value={"concurrent_feature": 1.0}
        ) as mock_compute:

            # Make concurrent requests
            tasks = [
                store.get_features(room_id, target_time) for target_time in target_times
            ]
            results = await asyncio.gather(*tasks)

            # All requests should succeed
            assert len(results) == 3
            for result in results:
                assert isinstance(result, dict)
                assert "concurrent_feature" in result

    def test_feature_type_validation(self, store):
        """Test validation of feature types."""
        valid_types = ["temporal", "sequential", "contextual"]

        for feature_type in valid_types:
            # Should not raise exception
            cache_key = store._get_cache_key("room", datetime.now(), 24, [feature_type])
            assert isinstance(cache_key, str)

        # Test mixed valid types
        cache_key = store._get_cache_key("room", datetime.now(), 24, valid_types)
        assert isinstance(cache_key, str)
