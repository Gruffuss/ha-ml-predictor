"""
Comprehensive unit tests for feature store, caching, and data management.

This test module covers feature record management, cache operations, feature store
functionality, and database integration for the ML occupancy prediction system.
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timedelta
import gc
from unittest.mock import AsyncMock, Mock, patch

import pandas as pd
import pytest

from src.data.storage.models import RoomState, SensorEvent
from src.features.engineering import FeatureEngineeringEngine
from src.features.store import FeatureCache, FeatureRecord, FeatureStore


class TestFeatureRecord:
    """Test FeatureRecord data structure and methods."""

    @pytest.fixture
    def sample_record(self):
        """Create sample feature record."""
        return FeatureRecord(
            room_id="living_room",
            target_time=datetime(2024, 1, 15, 15, 0, 0),
            features={"feature_1": 1.0, "feature_2": 2.0},
            extraction_time=datetime(2024, 1, 15, 15, 1, 0),
            lookback_hours=24,
            feature_types=["temporal", "sequential"],
            data_hash="abc123",
        )

    def test_record_initialization(self, sample_record):
        """Test feature record initialization."""
        assert sample_record.room_id == "living_room"
        assert sample_record.target_time == datetime(2024, 1, 15, 15, 0, 0)
        assert sample_record.features == {"feature_1": 1.0, "feature_2": 2.0}
        assert sample_record.lookback_hours == 24
        assert sample_record.feature_types == ["temporal", "sequential"]
        assert sample_record.data_hash == "abc123"

    def test_to_dict(self, sample_record):
        """Test serialization to dictionary."""
        record_dict = sample_record.to_dict()

        assert record_dict["room_id"] == "living_room"
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
        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_put_and_get(self, cache, sample_records):
        """Test basic cache put and get operations."""
        record = sample_records[0]

        # Put record in cache
        cache.put(
            room_id=record.room_id,
            target_time=record.target_time,
            lookback_hours=record.lookback_hours,
            feature_types=record.feature_types,
            features=record.features,
            data_hash=record.data_hash,
        )

        assert len(cache.cache) == 1
        assert cache.hit_count == 0
        assert cache.miss_count == 0

        # Get record from cache
        retrieved = cache.get(
            room_id=record.room_id,
            target_time=record.target_time,
            lookback_hours=record.lookback_hours,
            feature_types=record.feature_types,
        )

        assert retrieved == record.features
        assert cache.hit_count == 1
        assert cache.miss_count == 0

        # Test cache miss
        missing = cache.get(
            room_id="nonexistent",
            target_time=record.target_time,
            lookback_hours=record.lookback_hours,
            feature_types=record.feature_types,
        )
        assert missing is None
        assert cache.hit_count == 1
        assert cache.miss_count == 1

    def test_cache_lru_eviction(self, cache, sample_records):
        """Test LRU eviction when cache exceeds max size."""
        # Fill cache to capacity (max_size = 3)
        for i in range(3):
            record = sample_records[i]
            cache.put(
                room_id=record.room_id,
                target_time=record.target_time,
                lookback_hours=record.lookback_hours,
                feature_types=record.feature_types,
                features=record.features,
                data_hash=record.data_hash,
            )

        assert len(cache.cache) == 3

        # Access record 1 to make it recently used
        record_1 = sample_records[1]
        cache.get(
            room_id=record_1.room_id,
            target_time=record_1.target_time,
            lookback_hours=record_1.lookback_hours,
            feature_types=record_1.feature_types,
        )

        # Add another item, should evict least recently used (record 0)
        record_3 = sample_records[3]
        cache.put(
            room_id=record_3.room_id,
            target_time=record_3.target_time,
            lookback_hours=record_3.lookback_hours,
            feature_types=record_3.feature_types,
            features=record_3.features,
            data_hash=record_3.data_hash,
        )
        assert len(cache.cache) == 3

        # Record 0 should be evicted, others should remain
        record_0 = sample_records[0]
        evicted = cache.get(
            room_id=record_0.room_id,
            target_time=record_0.target_time,
            lookback_hours=record_0.lookback_hours,
            feature_types=record_0.feature_types,
        )
        assert evicted is None

        # Record 1 should still be there
        retrieved_1 = cache.get(
            room_id=record_1.room_id,
            target_time=record_1.target_time,
            lookback_hours=record_1.lookback_hours,
            feature_types=record_1.feature_types,
        )
        assert retrieved_1 is not None

    def test_cache_clear(self, cache, sample_records):
        """Test clearing the cache."""
        for i in range(3):
            record = sample_records[i]
            cache.put(
                room_id=record.room_id,
                target_time=record.target_time,
                lookback_hours=record.lookback_hours,
                feature_types=record.feature_types,
                features=record.features,
                data_hash=record.data_hash,
            )

        assert len(cache.cache) == 3

        cache.clear()
        assert len(cache.cache) == 0
        assert cache.hit_count == 0
        assert cache.miss_count == 0

    def test_cache_statistics(self, cache, sample_records):
        """Test cache hit/miss statistics."""
        record = sample_records[0]
        cache.put(
            room_id=record.room_id,
            target_time=record.target_time,
            lookback_hours=record.lookback_hours,
            feature_types=record.feature_types,
            features=record.features,
            data_hash=record.data_hash,
        )

        # Multiple hits
        for _ in range(5):
            cache.get(
                room_id=record.room_id,
                target_time=record.target_time,
                lookback_hours=record.lookback_hours,
                feature_types=record.feature_types,
            )
        assert cache.hit_count == 5

        # Multiple misses
        for _ in range(3):
            cache.get(
                room_id="nonexistent",
                target_time=record.target_time,
                lookback_hours=record.lookback_hours,
                feature_types=record.feature_types,
            )
        assert cache.miss_count == 3

        # Test stats calculation
        stats = cache.get_stats()
        assert stats["hit_count"] == 5
        assert stats["miss_count"] == 3
        assert stats["hit_rate"] == 5 / 8  # 5 hits out of 8 total

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

        cache.put(
            room_id=old_record.room_id,
            target_time=old_record.target_time,
            lookback_hours=old_record.lookback_hours,
            feature_types=old_record.feature_types,
            features=old_record.features,
            data_hash=old_record.data_hash,
            extraction_time=old_record.extraction_time,
        )

        # Should not return expired record
        with patch("src.features.store.datetime") as mock_datetime:
            # Current time is 25 hours after extraction
            mock_datetime.utcnow.return_value = datetime(2024, 1, 15, 16, 0, 0)

            result = cache.get(
                room_id=old_record.room_id,
                target_time=old_record.target_time,
                lookback_hours=old_record.lookback_hours,
                feature_types=old_record.feature_types,
                max_age_hours=24,
            )
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
            cache.put(
                room_id=record.room_id,
                target_time=record.target_time,
                lookback_hours=record.lookback_hours,
                feature_types=record.feature_types,
                features=record.features,
                data_hash=record.data_hash,
            )

        # Should only keep max_size items due to LRU eviction
        assert len(cache.cache) == cache.max_size

        # Clear and force garbage collection
        cache.clear()
        gc.collect()

        assert len(cache.cache) == 0


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
        store.cache.put(
            room_id=room_id,
            target_time=target_time,
            lookback_hours=24,
            feature_types=["temporal"],
            features=expected_features,
            data_hash="hash",
        )

        features = await store.get_features(room_id, target_time)

        assert features == expected_features
        # Check that it was a cache hit, not a miss
        assert store.cache.hit_count == 1
        assert store.cache.miss_count == 0

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
            # Should be a cache miss
            assert store.cache.miss_count > 0
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
                room_id="living_room",
                start_date=start_date,
                end_date=end_date,
                interval_minutes=interval_minutes,
            )

            assert isinstance(features_df, pd.DataFrame)
            assert isinstance(targets_df, pd.DataFrame)
            assert len(features_df) == len(mock_features)

    @pytest.mark.asyncio
    async def test_get_data_for_features_with_db(self, store):
        """Test getting data from database for feature computation."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        lookback_hours = 24

        # Mock database manager and session
        mock_db_manager = Mock()
        mock_session = AsyncMock()

        # Configure async context manager
        mock_session.__aenter__ = AsyncMock(return_value=mock_session)
        mock_session.__aexit__ = AsyncMock(return_value=None)

        # Mock query results
        mock_events_result = Mock()
        mock_events_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_events_result

        mock_db_manager.get_session.return_value = mock_session

        with patch(
            "src.features.store.get_database_manager", return_value=mock_db_manager
        ):
            events, room_states = await store._get_data_for_features(
                room_id, target_time, lookback_hours
            )

            assert isinstance(events, list)
            assert isinstance(room_states, list)

    @pytest.mark.asyncio
    async def test_feature_store_stats(self, store):
        """Test feature store statistics."""
        # Add some cache activity
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        store.cache.put(
            room_id=room_id,
            target_time=target_time,
            lookback_hours=24,
            feature_types=["temporal"],
            features={"test": 1.0},
            data_hash="hash",
        )

        # Get some hits and misses
        store.cache.get(room_id, target_time, 24, ["temporal"])  # Hit
        store.cache.get("other_room", target_time, 24, ["temporal"])  # Miss

        stats = store.get_statistics()

        assert "cache_stats" in stats
        assert stats["cache_stats"]["hit_count"] >= 1
        assert stats["cache_stats"]["miss_count"] >= 1

    def test_cache_key_generation(self, store):
        """Test cache key generation consistency."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        lookback_hours = 24
        feature_types = ["temporal", "sequential"]

        # Same parameters should generate same key
        key1 = store.cache._make_key(
            room_id, target_time, lookback_hours, feature_types
        )
        key2 = store.cache._make_key(
            room_id, target_time, lookback_hours, feature_types
        )

        assert key1 == key2

        # Different parameters should generate different keys
        key3 = store.cache._make_key(room_id, target_time, lookback_hours, ["temporal"])
        assert key1 != key3

    def test_feature_store_configuration(self):
        """Test feature store configuration options."""
        # Test with custom cache size
        store = FeatureStore(cache_size=500, enable_persistence=False)
        assert store.cache.max_size == 500

        # Test with default configuration
        default_store = FeatureStore()
        assert default_store.cache.max_size == 1000
        assert default_store.default_lookback_hours == 24
