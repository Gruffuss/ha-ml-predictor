"""
Sprint 2 Validation Tests - Feature Engineering Pipeline

This module contains comprehensive validation tests to ensure all Sprint 2 
feature engineering components are working correctly before proceeding to Sprint 3.
"""

import asyncio
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio


# Test that all core imports work
def test_sprint2_imports():
    """Test that all Sprint 2 components can be imported successfully."""

    # Feature extraction components
    from src.features.contextual import ContextualFeatureExtractor
    from src.features.engineering import FeatureEngineeringEngine
    from src.features.sequential import SequentialFeatureExtractor
    from src.features.store import FeatureCache
    from src.features.store import FeatureRecord
    from src.features.store import FeatureStore
    from src.features.temporal import TemporalFeatureExtractor

    # All imports successful
    assert True


def test_sprint2_temporal_extractor_structure():
    """Test that the temporal feature extractor is properly structured."""
    from src.data.storage.models import RoomState
    from src.data.storage.models import SensorEvent
    from src.features.temporal import TemporalFeatureExtractor

    # Test extractor initialization
    extractor = TemporalFeatureExtractor(timezone_offset=-8)
    assert extractor.timezone_offset == -8
    assert isinstance(extractor.feature_cache, dict)

    # Test feature names availability
    feature_names = extractor.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

    # Check for expected temporal features
    expected_features = [
        "time_since_last_event",
        "current_state_duration",
        "hour_sin",
        "hour_cos",
        "is_weekend",
    ]
    for feature in expected_features:
        assert feature in feature_names

    # Test default features
    default_features = extractor._get_default_features()
    assert isinstance(default_features, dict)
    assert len(default_features) > 0
    assert all(isinstance(v, (int, float)) for v in default_features.values())


def test_sprint2_sequential_extractor_structure():
    """Test that the sequential feature extractor is properly structured."""
    from src.core.config import SystemConfig
    from src.features.sequential import SequentialFeatureExtractor

    # Test extractor initialization
    extractor = SequentialFeatureExtractor()
    assert extractor.config is None  # Should handle None config gracefully
    assert isinstance(extractor.sequence_cache, dict)

    # Test with config
    from tests.conftest import test_system_config

    # Create mock config for testing
    mock_config = SystemConfig(
        home_assistant={"url": "http://test", "token": "test"},
        database={"connection_string": "test"},
        mqtt={"broker": "test"},
        prediction={"interval_seconds": 300},
        features={"lookback_hours": 24},
        logging={"level": "INFO"},
        rooms={},
    )

    extractor_with_config = SequentialFeatureExtractor(mock_config)
    assert extractor_with_config.config == mock_config
    assert extractor_with_config.classifier is not None

    # Test feature names availability
    feature_names = extractor.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

    # Check for expected sequential features
    expected_features = [
        "room_transition_count",
        "unique_sensors_triggered",
        "movement_velocity_score",
        "human_movement_probability",
    ]
    for feature in expected_features:
        assert feature in feature_names


def test_sprint2_contextual_extractor_structure():
    """Test that the contextual feature extractor is properly structured."""
    from src.features.contextual import ContextualFeatureExtractor

    # Test extractor initialization
    extractor = ContextualFeatureExtractor()
    assert isinstance(extractor.context_cache, dict)
    assert isinstance(extractor.temp_thresholds, dict)
    assert isinstance(extractor.humidity_thresholds, dict)
    assert isinstance(extractor.light_thresholds, dict)

    # Test feature names availability
    feature_names = extractor.get_feature_names()
    assert isinstance(feature_names, list)
    assert len(feature_names) > 0

    # Check for expected contextual features
    expected_features = [
        "current_temperature",
        "doors_currently_open",
        "total_active_rooms",
        "is_winter",
    ]
    for feature in expected_features:
        assert feature in feature_names


def test_sprint2_feature_engineering_engine():
    """Test that the feature engineering engine coordinates extractors properly."""
    from src.features.engineering import FeatureEngineeringEngine

    # Test engine initialization
    engine = FeatureEngineeringEngine(enable_parallel=False)
    assert engine.temporal_extractor is not None
    assert engine.sequential_extractor is not None
    assert engine.contextual_extractor is not None
    assert isinstance(engine.stats, dict)

    # Test feature names compilation
    all_feature_names = engine.get_feature_names()
    assert isinstance(all_feature_names, list)
    assert len(all_feature_names) > 0

    # Check that features have proper prefixes
    temporal_features = [f for f in all_feature_names if f.startswith("temporal_")]
    sequential_features = [f for f in all_feature_names if f.startswith("sequential_")]
    contextual_features = [f for f in all_feature_names if f.startswith("contextual_")]
    meta_features = [f for f in all_feature_names if f.startswith("meta_")]

    assert len(temporal_features) > 0
    assert len(sequential_features) > 0
    assert len(contextual_features) > 0
    assert len(meta_features) > 0

    # Test default features
    default_features = engine._get_default_features()
    assert isinstance(default_features, dict)
    assert len(default_features) > 0


@pytest.mark.asyncio
async def test_sprint2_feature_extraction_basic():
    """Test basic feature extraction with mock data."""
    from src.data.storage.models import RoomState
    from src.data.storage.models import SensorEvent
    from src.features.engineering import FeatureEngineeringEngine

    # Create engine
    engine = FeatureEngineeringEngine(enable_parallel=False)

    # Create mock sensor events
    base_time = datetime.utcnow() - timedelta(hours=2)
    mock_events = []

    for i in range(10):
        event = SensorEvent(
            room_id="test_room",
            sensor_id=f"binary_sensor.test_{i % 3}",
            sensor_type="presence",
            state="on" if i % 2 == 0 else "off",
            previous_state="off" if i % 2 == 0 else "on",
            timestamp=base_time + timedelta(minutes=i * 10),
            attributes={"test": True},
            is_human_triggered=True,
            confidence_score=0.8,
            created_at=datetime.utcnow(),
        )
        mock_events.append(event)

    # Create mock room states
    mock_room_states = []
    for i in range(5):
        state = RoomState(
            room_id="test_room",
            timestamp=base_time + timedelta(minutes=i * 20),
            is_occupied=i % 2 == 0,
            occupancy_confidence=0.9,
            created_at=datetime.utcnow(),
        )
        mock_room_states.append(state)

    # Extract features
    target_time = datetime.utcnow()
    features = await engine.extract_features(
        room_id="test_room",
        target_time=target_time,
        events=mock_events,
        room_states=mock_room_states,
        lookback_hours=6,
    )

    # Validate results
    assert isinstance(features, dict)
    assert len(features) > 0

    # Check that all feature types are present
    has_temporal = any(k.startswith("temporal_") for k in features.keys())
    has_sequential = any(k.startswith("sequential_") for k in features.keys())
    has_contextual = any(k.startswith("contextual_") for k in features.keys())
    has_meta = any(k.startswith("meta_") for k in features.keys())

    assert has_temporal
    assert has_sequential
    assert has_contextual
    assert has_meta

    # Check that all values are numeric
    assert all(isinstance(v, (int, float)) for v in features.values())
    assert all(not np.isnan(v) for v in features.values())


def test_sprint2_feature_cache():
    """Test the feature cache functionality."""
    from src.features.store import FeatureCache
    from src.features.store import FeatureRecord

    # Create cache
    cache = FeatureCache(max_size=5)
    assert cache.max_size == 5
    assert len(cache.cache) == 0

    # Test cache operations
    room_id = "test_room"
    target_time = datetime.utcnow()
    features = {"feature1": 1.0, "feature2": 2.0}

    # Put feature in cache
    cache.put(room_id, target_time, 24, ["temporal"], features, "test_hash")
    assert len(cache.cache) == 1

    # Get feature from cache
    cached_features = cache.get(room_id, target_time, 24, ["temporal"])
    assert cached_features == features
    assert cache.hit_count == 1

    # Test cache miss
    miss_features = cache.get("other_room", target_time, 24, ["temporal"])
    assert miss_features is None
    assert cache.miss_count == 1

    # Test cache stats
    stats = cache.get_stats()
    assert stats["size"] == 1
    assert stats["hit_count"] == 1
    assert stats["miss_count"] == 1
    assert stats["hit_rate"] == 0.5


def test_sprint2_feature_record():
    """Test feature record serialization and validation."""
    from src.features.store import FeatureRecord

    # Create feature record
    record = FeatureRecord(
        room_id="test_room",
        target_time=datetime.utcnow(),
        features={"test": 1.0},
        extraction_time=datetime.utcnow(),
        lookback_hours=24,
        feature_types=["temporal"],
        data_hash="test_hash",
    )

    # Test serialization
    record_dict = record.to_dict()
    assert isinstance(record_dict, dict)
    assert "room_id" in record_dict
    assert isinstance(record_dict["target_time"], str)

    # Test deserialization
    restored_record = FeatureRecord.from_dict(record_dict)
    assert restored_record.room_id == record.room_id
    assert restored_record.features == record.features

    # Test validity
    assert record.is_valid(max_age_hours=24)

    # Test invalid (old) record
    old_record = FeatureRecord(
        room_id="test_room",
        target_time=datetime.utcnow(),
        features={"test": 1.0},
        extraction_time=datetime.utcnow() - timedelta(hours=25),
        lookback_hours=24,
        feature_types=["temporal"],
        data_hash="test_hash",
    )
    assert not old_record.is_valid(max_age_hours=24)


@pytest.mark.asyncio
async def test_sprint2_feature_store_structure():
    """Test that the feature store is properly structured."""
    from src.features.store import FeatureStore

    # Test store initialization
    store = FeatureStore(enable_persistence=False)
    assert store.feature_engine is not None
    assert store.cache is not None
    assert isinstance(store.stats, dict)

    # Test async initialization
    async with store:
        # Test health check
        health = await store.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        assert "components" in health

        # Test stats
        stats = store.get_stats()
        assert isinstance(stats, dict)
        assert "feature_store" in stats
        assert "cache" in stats
        assert "engine" in stats


@pytest.mark.asyncio
async def test_sprint2_batch_processing():
    """Test batch feature processing capabilities."""
    from src.data.storage.models import SensorEvent
    from src.features.engineering import FeatureEngineeringEngine

    engine = FeatureEngineeringEngine(enable_parallel=False)

    # Create batch extraction requests
    base_time = datetime.utcnow()
    mock_events = [
        SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="presence",
            state="on",
            timestamp=base_time - timedelta(minutes=30),
            created_at=base_time,
        )
    ]

    requests = [
        ("test_room", base_time - timedelta(minutes=i), mock_events, [])
        for i in range(5)
    ]

    # Process batch
    results = await engine.extract_batch_features(requests, lookback_hours=2)

    # Validate results
    assert isinstance(results, list)
    assert len(results) == 5
    assert all(isinstance(result, dict) for result in results)
    assert all(len(result) > 0 for result in results)


def test_sprint2_dataframe_creation():
    """Test DataFrame creation from feature dictionaries."""
    from src.features.engineering import FeatureEngineeringEngine

    engine = FeatureEngineeringEngine()

    # Create mock feature dictionaries
    feature_dicts = [
        {
            "temporal_hour_sin": 0.5,
            "sequential_room_count": 2.0,
            "contextual_temperature": 22.0,
        },
        {
            "temporal_hour_sin": 0.3,
            "sequential_room_count": 1.0,
            "contextual_temperature": 23.0,
        },
        {
            "temporal_hour_sin": 0.7,
            "sequential_room_count": 3.0,
            "contextual_temperature": 21.0,
        },
    ]

    # Create DataFrame
    df = engine.create_feature_dataframe(feature_dicts)

    # Validate DataFrame
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3
    assert len(df.columns) > 0

    # Check that missing features are filled with defaults
    assert not df.isnull().any().any()


def test_sprint2_feature_validation():
    """Test feature validation and error handling."""
    from src.core.exceptions import FeatureExtractionError
    from src.features.contextual import ContextualFeatureExtractor
    from src.features.sequential import SequentialFeatureExtractor
    from src.features.temporal import TemporalFeatureExtractor

    # Test temporal extractor with empty events
    temporal_extractor = TemporalFeatureExtractor()

    empty_features = temporal_extractor.extract_features([], datetime.utcnow())
    assert isinstance(empty_features, dict)
    assert len(empty_features) > 0

    # Test sequential extractor with empty events
    sequential_extractor = SequentialFeatureExtractor()

    empty_sequential = sequential_extractor.extract_features([], datetime.utcnow())
    assert isinstance(empty_sequential, dict)
    assert len(empty_sequential) > 0

    # Test contextual extractor with empty events
    contextual_extractor = ContextualFeatureExtractor()

    empty_contextual = contextual_extractor.extract_features([], [], datetime.utcnow())
    assert isinstance(empty_contextual, dict)
    assert len(empty_contextual) > 0


def test_sprint2_feature_consistency():
    """Test that features are consistent across multiple extractions."""
    from src.data.storage.models import SensorEvent
    from src.features.temporal import TemporalFeatureExtractor

    extractor = TemporalFeatureExtractor()

    # Create identical mock events
    base_time = datetime.utcnow()
    events = [
        SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="presence",
            state="on",
            timestamp=base_time - timedelta(minutes=30),
            created_at=base_time,
        )
    ]

    target_time = base_time

    # Extract features multiple times
    features1 = extractor.extract_features(events, target_time)
    features2 = extractor.extract_features(events, target_time)

    # Features should be identical
    assert features1 == features2

    # Check specific feature consistency
    assert features1["time_since_last_event"] == features2["time_since_last_event"]


def test_sprint2_feature_ranges():
    """Test that extracted features are within reasonable ranges."""
    from src.data.storage.models import SensorEvent
    from src.features.temporal import TemporalFeatureExtractor

    extractor = TemporalFeatureExtractor()

    # Create mock events
    base_time = datetime.utcnow()
    events = [
        SensorEvent(
            room_id="test_room",
            sensor_id="binary_sensor.test",
            sensor_type="presence",
            state="on",
            timestamp=base_time - timedelta(minutes=i * 10),
            created_at=base_time,
        )
        for i in range(5)
    ]

    features = extractor.extract_features(events, base_time)

    # Check cyclical features are in [-1, 1] range
    assert -1.0 <= features["hour_sin"] <= 1.0
    assert -1.0 <= features["hour_cos"] <= 1.0
    assert -1.0 <= features["day_sin"] <= 1.0
    assert -1.0 <= features["day_cos"] <= 1.0

    # Check boolean features are 0 or 1
    assert features["is_weekend"] in [0.0, 1.0]
    assert features["is_work_hours"] in [0.0, 1.0]
    assert features["is_sleep_hours"] in [0.0, 1.0]

    # Check time-based features are positive
    assert features["time_since_last_event"] >= 0.0
    assert features["current_state_duration"] >= 0.0


def test_sprint2_file_structure():
    """Test that all expected Sprint 2 files exist."""
    base_path = Path(__file__).parent.parent

    # Feature extraction files
    assert (base_path / "src" / "features" / "__init__.py").exists()
    assert (base_path / "src" / "features" / "temporal.py").exists()
    assert (base_path / "src" / "features" / "sequential.py").exists()
    assert (base_path / "src" / "features" / "contextual.py").exists()
    assert (base_path / "src" / "features" / "engineering.py").exists()
    assert (base_path / "src" / "features" / "store.py").exists()

    # Test files
    assert (base_path / "tests" / "conftest.py").exists()
    assert (base_path / "tests" / "test_sprint2_validation.py").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sprint2_end_to_end_feature_pipeline():
    """Test a complete end-to-end feature extraction pipeline."""
    from src.data.storage.models import RoomState
    from src.data.storage.models import SensorEvent
    from src.features.store import FeatureStore

    # Create feature store
    async with FeatureStore(enable_persistence=False) as store:
        # Create comprehensive mock data
        base_time = datetime.utcnow() - timedelta(hours=4)

        # Mock sensor events with different types
        mock_events = []
        sensor_types = ["presence", "door", "temperature", "light"]

        for i in range(20):
            event = SensorEvent(
                room_id="living_room",
                sensor_id=f"sensor.living_room_{sensor_types[i % len(sensor_types)]}",
                sensor_type=sensor_types[i % len(sensor_types)],
                state="on" if i % 2 == 0 else "off",
                previous_state="off" if i % 2 == 0 else "on",
                timestamp=base_time + timedelta(minutes=i * 10),
                attributes={"device_class": sensor_types[i % len(sensor_types)]},
                is_human_triggered=True,
                confidence_score=0.8 + (i * 0.01),
                created_at=datetime.utcnow(),
            )
            mock_events.append(event)

        # Mock room states
        mock_room_states = []
        for i in range(8):
            state = RoomState(
                room_id="living_room",
                timestamp=base_time + timedelta(minutes=i * 30),
                is_occupied=i % 2 == 0,
                occupancy_confidence=0.85 + (i * 0.02),
                occupant_type="human" if i % 3 != 0 else "cat",
                state_duration=1800,  # 30 minutes
                created_at=datetime.utcnow(),
            )
            mock_room_states.append(state)

        # Override the store's data retrieval for testing
        original_method = store._get_data_for_features

        async def mock_get_data(room_id, target_time, lookback_hours):
            return mock_events, mock_room_states

        store._get_data_for_features = mock_get_data

        # Test feature extraction
        target_time = datetime.utcnow()
        features = await store.get_features(
            room_id="living_room", target_time=target_time, lookback_hours=6
        )

        # Validate comprehensive feature extraction
        assert isinstance(features, dict)
        assert len(features) > 50  # Should have many features

        # Verify all feature categories are present
        temporal_features = [k for k in features.keys() if k.startswith("temporal_")]
        sequential_features = [
            k for k in features.keys() if k.startswith("sequential_")
        ]
        contextual_features = [
            k for k in features.keys() if k.startswith("contextual_")
        ]
        meta_features = [k for k in features.keys() if k.startswith("meta_")]

        assert len(temporal_features) > 10
        assert len(sequential_features) > 10
        assert len(contextual_features) > 10
        assert len(meta_features) > 0

        # Test batch processing
        batch_requests = [
            ("living_room", target_time - timedelta(hours=i)) for i in range(3)
        ]

        batch_results = await store.get_batch_features(batch_requests)
        assert len(batch_results) == 3
        assert all(isinstance(result, dict) for result in batch_results)
        assert all(len(result) > 0 for result in batch_results)

        # Test caching (second request should be cached)
        cached_features = await store.get_features(
            room_id="living_room", target_time=target_time, lookback_hours=6
        )

        assert cached_features == features

        # Verify cache hit
        stats = store.get_stats()
        assert stats["feature_store"]["cache_hits"] > 0

        # Test training data generation
        start_date = target_time - timedelta(hours=2)
        end_date = target_time

        features_df, targets_df = await store.compute_training_data(
            room_id="living_room",
            start_date=start_date,
            end_date=end_date,
            interval_minutes=30,
        )

        assert isinstance(features_df, pd.DataFrame)
        assert isinstance(targets_df, pd.DataFrame)
        assert len(features_df) > 0
        assert len(features_df.columns) > 50
        assert len(targets_df) == len(features_df)


@pytest.mark.smoke
def test_sprint2_smoke_test():
    """Smoke test to verify basic Sprint 2 functionality."""
    # This test should run very quickly and catch major issues

    # Test imports work
    from src.features.contextual import ContextualFeatureExtractor
    from src.features.engineering import FeatureEngineeringEngine
    from src.features.sequential import SequentialFeatureExtractor
    from src.features.store import FeatureStore
    from src.features.temporal import TemporalFeatureExtractor

    # Test basic object creation
    temporal = TemporalFeatureExtractor()
    sequential = SequentialFeatureExtractor()
    contextual = ContextualFeatureExtractor()
    engine = FeatureEngineeringEngine()
    store = FeatureStore(enable_persistence=False)

    # Test that objects have expected attributes
    assert hasattr(temporal, "extract_features")
    assert hasattr(sequential, "extract_features")
    assert hasattr(contextual, "extract_features")
    assert hasattr(engine, "extract_features")
    assert hasattr(store, "get_features")

    # Test default features are available
    temporal_defaults = temporal._get_default_features()
    sequential_defaults = sequential._get_default_features()
    contextual_defaults = contextual._get_default_features()

    assert len(temporal_defaults) > 0
    assert len(sequential_defaults) > 0
    assert len(contextual_defaults) > 0


if __name__ == "__main__":
    """
    Run Sprint 2 validation tests directly.

    Usage: python tests/test_sprint2_validation.py
    """
    pytest.main([__file__, "-v", "--tb=short"])
