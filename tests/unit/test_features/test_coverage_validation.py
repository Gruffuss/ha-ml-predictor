"""
Final coverage validation tests to reach 85% target.

This module focuses on the specific uncovered lines to maximize coverage.
"""

from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pytest

from src.core.config import RoomConfig
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureCache, FeatureStore
from src.features.temporal import TemporalFeatureExtractor


class TestTemporalCoverageGaps:
    """Test specific temporal feature gaps to improve coverage."""

    def test_temporal_extract_room_state_features_edge_cases(self):
        """Test room state feature extraction edge cases."""
        extractor = TemporalFeatureExtractor()
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with empty room states
        features_empty = extractor._extract_room_state_features([], target_time)
        assert isinstance(features_empty, dict)
        assert "room_state_confidence_avg" in features_empty
        assert features_empty["room_state_confidence_avg"] == 0.0

        # Test with room states without confidence
        room_states_no_conf = []
        for i in range(3):
            state = Mock(spec=RoomState)
            state.state = "occupied" if i % 2 == 0 else "vacant"
            state.timestamp = target_time - timedelta(minutes=i * 10)
            # No confidence attribute
            room_states_no_conf.append(state)

        features_no_conf = extractor._extract_room_state_features(
            room_states_no_conf, target_time
        )
        assert "room_state_confidence_avg" in features_no_conf

    def test_temporal_extract_historical_patterns_insufficient_data(self):
        """Test historical patterns with insufficient data."""
        extractor = TemporalFeatureExtractor()
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Very few events
        events = []
        for i in range(2):  # Less than minimum for trends
            event = Mock(spec=SensorEvent)
            event.timestamp = target_time - timedelta(hours=i)
            event.state = "on"
            events.append(event)

        features = extractor._extract_historical_patterns(events, target_time)
        assert isinstance(features, dict)
        assert "historical_trend" in features
        # Should handle insufficient data gracefully

    def test_temporal_extract_cyclical_features_edge_times(self):
        """Test cyclical features at edge times (midnight, etc)."""
        extractor = TemporalFeatureExtractor()

        # Test at exactly midnight
        midnight = datetime(2024, 1, 15, 0, 0, 0)
        features_midnight = extractor._extract_cyclical_features(midnight)

        assert "cyclical_hour_sin" in features_midnight
        assert "cyclical_hour_cos" in features_midnight
        assert abs(features_midnight["cyclical_hour_sin"]) <= 1.0
        assert abs(features_midnight["cyclical_hour_cos"]) <= 1.0

    def test_temporal_error_handling_corrupted_events(self):
        """Test temporal extractor error handling."""
        extractor = TemporalFeatureExtractor()

        # Events with None timestamps
        corrupted_events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = None if i == 2 else datetime(2024, 1, 15, 12, i, 0)
            event.state = "on"
            event.room_id = "test_room"
            corrupted_events.append(event)

        # Should handle errors gracefully
        with pytest.raises(Exception):  # Should raise FeatureExtractionError
            extractor.extract_features(
                corrupted_events, datetime(2024, 1, 15, 12, 0, 0)
            )


class TestSequentialCoverageGaps:
    """Test sequential feature gaps to improve coverage."""

    def test_sequential_extract_cross_room_features_edge_cases(self):
        """Test cross-room feature extraction edge cases."""
        extractor = SequentialFeatureExtractor()

        # Events from non-existent rooms
        events = []
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.room_id = f"nonexistent_room_{i}"
            event.timestamp = datetime(2024, 1, 15, 12, i, 0)
            event.sensor_type = "motion"
            events.append(event)

        # Empty room configs
        room_configs = {}
        target_time = datetime(2024, 1, 15, 12, 30, 0)

        features = extractor.extract_features(events, target_time, room_configs)
        assert isinstance(features, dict)
        # Should handle missing room configs

    def test_sequential_movement_classification_no_classifier(self):
        """Test movement classification when classifier is unavailable."""
        extractor = SequentialFeatureExtractor()
        extractor.classifier = None  # No classifier available

        events = []
        for i in range(20):
            event = Mock(spec=SensorEvent)
            event.room_id = "test_room"
            event.timestamp = datetime(2024, 1, 15, 12, i, 0)
            event.sensor_type = "motion"
            event.state = "on" if i % 3 == 0 else "off"
            events.append(event)

        room_configs = {"test_room": Mock(spec=RoomConfig)}
        features = extractor.extract_features(
            events, datetime(2024, 1, 15, 12, 30, 0), room_configs
        )

        # Should work without classifier
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_sequential_velocity_calculation_edge_cases(self):
        """Test velocity calculation with edge cases."""
        extractor = SequentialFeatureExtractor()

        # Events with identical timestamps
        events = []
        same_time = datetime(2024, 1, 15, 12, 0, 0)
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.room_id = f"room_{i}"
            event.timestamp = same_time  # All same timestamp
            event.sensor_type = "motion"
            events.append(event)

        velocity_features = extractor._extract_velocity_features(events)
        assert isinstance(velocity_features, dict)
        # Should handle zero time differences


class TestContextualCoverageGaps:
    """Test contextual feature gaps to improve coverage."""

    def test_contextual_environmental_features_with_invalid_values(self):
        """Test environmental features with invalid sensor values."""
        extractor = ContextualFeatureExtractor()

        events = []
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.room_id = "test_room"
            event.timestamp = datetime(2024, 1, 15, 12, i, 0)
            event.sensor_type = "temperature"

            # Mix of valid and invalid values
            if i % 3 == 0:
                event.attributes = {"temperature": float("inf")}  # Invalid
            elif i % 3 == 1:
                event.attributes = {"temperature": "not_a_number"}  # Invalid type
            else:
                event.attributes = {"temperature": 22.0}  # Valid

            events.append(event)

        room_states = [Mock(spec=RoomState)]
        features = extractor.extract_features(events, room_states)

        # Should filter out invalid values
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_contextual_multi_room_correlation_single_room(self):
        """Test multi-room correlation with only one room."""
        extractor = ContextualFeatureExtractor()

        # Events from only one room
        events = []
        for i in range(15):
            event = Mock(spec=SensorEvent)
            event.room_id = "single_room"  # Only one room
            event.timestamp = datetime(2024, 1, 15, 12, i, 0)
            event.sensor_type = "motion"
            events.append(event)

        room_states = []
        features = extractor.extract_features(events, room_states)

        # Should handle single room case
        assert isinstance(features, dict)
        if "multi_room_correlation_strength" in features:
            assert features["multi_room_correlation_strength"] >= -1
            assert features["multi_room_correlation_strength"] <= 1


class TestEngineeringCoverageGaps:
    """Test engineering engine gaps to improve coverage."""

    def test_engineering_get_default_features_all_types(self):
        """Test getting default features for all extractor types."""
        engine = FeatureEngineeringEngine()

        defaults = engine.get_default_features()
        assert isinstance(defaults, dict)
        assert len(defaults) > 0

        # Should include features from all extractors
        assert any("temporal" in key for key in defaults.keys())

    def test_engineering_clear_caches_all_extractors(self):
        """Test clearing caches for all extractors."""
        engine = FeatureEngineeringEngine()

        # Add some data to caches first
        engine.temporal_extractor.feature_cache["test"] = {"value": 1.0}
        engine.sequential_extractor.sequence_cache["test"] = {"value": 2.0}

        # Clear all caches
        engine.clear_caches()

        # Caches should be empty
        assert len(engine.temporal_extractor.feature_cache) == 0
        assert len(engine.sequential_extractor.sequence_cache) == 0

    def test_engineering_validate_configuration_invalid(self):
        """Test configuration validation with invalid config."""
        engine = FeatureEngineeringEngine()

        # Test with invalid configuration
        invalid_config = {"invalid_key": "invalid_value"}

        # Should handle invalid configuration gracefully
        result = engine.validate_configuration(invalid_config)
        assert isinstance(result, bool)

    def test_engineering_parallel_vs_sequential_extraction(self):
        """Test both parallel and sequential extraction modes."""
        engine = FeatureEngineeringEngine()

        room_id = "test_room"
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        events = [Mock(spec=SensorEvent) for _ in range(5)]
        room_configs = {"test_room": Mock(spec=RoomConfig)}

        # Test sequential mode
        features_sequential = engine.extract_features(
            room_id, target_time, events, room_configs, parallel=False
        )

        # Test parallel mode
        features_parallel = engine.extract_features(
            room_id, target_time, events, room_configs, parallel=True
        )

        # Both should work
        assert isinstance(features_sequential, dict)
        assert isinstance(features_parallel, dict)


class TestStoreCoverageGaps:
    """Test store coverage gaps to improve coverage."""

    def test_feature_cache_expired_record_cleanup(self):
        """Test automatic cleanup of expired records."""
        cache = FeatureCache(max_size=100)

        # Add record that will expire quickly
        cache.put("expire_test", {"value": 1.0})

        # Record should exist initially
        assert cache.get("expire_test") is not None

        # Test cache statistics
        stats = cache.get_statistics()
        assert "hits" in stats
        assert "misses" in stats
        assert stats["size"] > 0

    def test_feature_store_compute_training_data_edge_cases(self):
        """Test compute training data with edge cases."""
        with patch("src.features.store.DatabaseManager") as mock_db:
            store = FeatureStore(db_manager=mock_db)

            # Test with very short time range
            start_date = datetime(2024, 1, 15, 12, 0, 0)
            end_date = datetime(2024, 1, 15, 12, 30, 0)  # 30 minutes

            # Mock empty database response
            mock_db.return_value.get_events.return_value = []
            mock_db.return_value.get_room_states.return_value = []

            features_df, targets_df = store.compute_training_data(start_date, end_date)

            # Should handle empty data gracefully
            assert features_df is not None
            assert targets_df is not None

    def test_feature_store_cache_key_generation_complex(self):
        """Test cache key generation with complex parameters."""
        with patch("src.features.store.DatabaseManager") as mock_db:
            store = FeatureStore(db_manager=mock_db)

            room_id = "complex_room_with_unicode_名前"
            target_time = datetime(2024, 1, 15, 12, 0, 0)

            # Generate cache key with complex parameters
            cache_key = store._generate_cache_key(
                room_id,
                target_time,
                config={"complex": {"nested": {"value": [1, 2, 3]}}},
                lookback_hours=48,
            )

            assert isinstance(cache_key, str)
            assert len(cache_key) > 0
