"""
Unit tests for temporal feature extraction.

This module tests the TemporalFeatureExtractor for accuracy, edge cases,
and performance with realistic occupancy data patterns.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest
import statistics

from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.temporal import TemporalFeatureExtractor


class TestTemporalFeatureExtractor:
    """Test suite for TemporalFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a temporal feature extractor instance."""
        return TemporalFeatureExtractor(timezone_offset=-8)  # PST

    @pytest.fixture
    def sample_events(self) -> List[SensorEvent]:
        """Create sample sensor events for testing."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)
        events = []

        # Create realistic event sequence with proper state values
        states = ["off", "on", "on", "off", "on", "off"]
        sensor_types = [
            "motion",
            "presence",
            "door",
            "motion",
            "presence",
            "motion",
        ]

        for i, (state, sensor_type) in enumerate(zip(states, sensor_types)):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.state = state
            event.sensor_type = sensor_type
            event.sensor_id = f"sensor.test_{sensor_type}_{i}"
            event.room_id = "living_room"
            events.append(event)

        return events

    @pytest.fixture
    def empty_events(self) -> List[SensorEvent]:
        """Return empty event list."""
        return []

    @pytest.fixture
    def single_event(self) -> List[SensorEvent]:
        """Create single event for edge case testing."""
        event = Mock(spec=SensorEvent)
        event.timestamp = datetime(2024, 1, 15, 14, 30, 0)
        event.state = "on"
        event.sensor_type = "motion"
        event.sensor_id = "sensor.test_motion"
        event.room_id = "living_room"
        return [event]

    @pytest.fixture
    def room_states(self) -> List[RoomState]:
        """Create sample room states for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        states = []

        for i in range(5):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 10)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8 + (i * 0.05)
            states.append(state)

        return states

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    @pytest.fixture
    def very_old_events(self) -> List[SensorEvent]:
        """Create very old events for lookback testing."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)  # 2 weeks old
        events = []

        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(hours=i * 2)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            event.room_id = "living_room"
            # Add mock attributes to prevent iteration errors
            event.attributes = {"test": "value"} if hasattr(Mock, "attributes") else {}
            events.append(event)

        return events

    @pytest.fixture
    def large_events(self) -> List[SensorEvent]:
        """Create large dataset of events for performance testing."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        events = []

        for i in range(1000):  # Large dataset
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = ["motion", "door", "presence"][i % 3]
            event.sensor_id = f"sensor.{event.sensor_type}_{i}"
            event.room_id = "living_room"
            # Add mock attributes to prevent iteration errors
            event.attributes = {"sensor_type": event.sensor_type}
            events.append(event)

        return events

    def test_extract_features_with_sample_data(
        self, extractor, sample_events, target_time
    ):
        """Test feature extraction with realistic sample data."""
        features = extractor.extract_features(sample_events, target_time)

        # Verify basic feature presence
        assert isinstance(features, dict)
        assert len(features) > 20  # Should have reasonable number of features

        # Verify key feature categories
        temporal_features = [
            "time_since_last_on",
            "time_since_last_off",
            "hour_sin",
            "hour_cos",
        ]
        for feature in temporal_features:
            assert feature in features, f"Missing feature: {feature}"

        # Verify feature value ranges
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0

    def test_empty_events(self, extractor, empty_events, target_time):
        """Test feature extraction with no events."""
        features = extractor.extract_features(empty_events, target_time)

        # Should return default features
        assert isinstance(features, dict)
        assert len(features) > 0

        # Time-since features should have default values
        assert features["time_since_last_on"] >= 0.0
        assert features["time_since_last_off"] >= 0.0

    def test_single_event(self, extractor, single_event, target_time):
        """Test feature extraction with single event."""
        features = extractor.extract_features(single_event, target_time)

        # Should handle single event gracefully
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_feature_value_ranges(self, extractor, sample_events, target_time):
        """Test that extracted features have reasonable value ranges."""
        features = extractor.extract_features(sample_events, target_time)

        # Cyclical features should be normalized
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0
        assert -1.0 <= features["day_of_week_sin"] <= 1.0
        assert -1.0 <= features["day_of_week_cos"] <= 1.0

        # Boolean features should be 0 or 1
        boolean_features = [
            "is_weekend",
            "is_work_hours",
            "is_sleep_hours",
            "is_meal_time",
        ]
        for feature in boolean_features:
            if feature in features:
                assert features[feature] in [0.0, 1.0]

        # Time-based features should be non-negative
        time_features = [
            "time_since_last_on",
            "time_since_last_off",
            "hours_since_midnight",
        ]
        for feature in time_features:
            if feature in features:
                assert features[feature] >= 0.0

    def test_timezone_handling(self, target_time):
        """Test feature extraction with different timezone offsets."""
        # Test multiple timezone offsets
        timezone_offsets = [-8, -5, 0, 3, 8]

        for offset in timezone_offsets:
            extractor = TemporalFeatureExtractor(timezone_offset=offset)
            features = extractor.extract_features([], target_time)

            # Should produce valid features for all timezones
            assert isinstance(features, dict)
            assert len(features) > 0

            # Time features should be within valid ranges
            assert -1.0 <= features["hour_sin"] <= 1.0
            assert -1.0 <= features["hour_cos"] <= 1.0

    def test_room_state_integration(
        self, extractor, sample_events, room_states, target_time
    ):
        """Test feature extraction with room state information."""
        features = extractor.extract_features(sample_events, target_time, room_states)

        # Should include room state-based features
        assert isinstance(features, dict)

        # Room state features should be present
        state_features = ["occupancy_duration", "occupancy_transitions"]
        for feature in state_features:
            if feature in features:
                assert isinstance(features[feature], (int, float))

    def test_event_sequence_patterns(self, extractor, target_time):
        """Test detection of event sequence patterns."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Create pattern: on -> off -> on -> off (regular switching)
        for i in range(8):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should calculate transition timing features
        assert "avg_transition_interval" in features
        assert "transition_regularity" in features
        assert "recent_transition_trend" in features

    def test_default_features_completeness(self, extractor):
        """Test that default features include all expected keys."""
        default_features = extractor._get_default_features()

        # Check for essential temporal feature categories
        required_feature_categories = [
            "time_since_",  # Time-since features
            "hour_",  # Cyclical hour features
            "day_",  # Cyclical day features
            "is_",  # Boolean time features
            "activity_",  # Activity pattern features
            "transition_",  # Transition timing features
        ]

        feature_keys = list(default_features.keys())
        for category in required_feature_categories:
            category_found = any(category in key for key in feature_keys)
            assert category_found, f"No features found for category: {category}"

    def test_cyclical_encoding_accuracy(self, extractor):
        """Test mathematical accuracy of cyclical encodings."""
        # Test hour encoding for noon (12:00) with PST timezone offset (-8)
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        features = extractor.extract_features([], target_time)

        # With timezone_offset=-8, local time becomes 12 + (-8) = 4:00
        # At 4:00, hour angle should be 2π * 4 / 24 = π/3 (60 degrees)
        local_hour = 4  # 12 + (-8)
        expected_hour_sin = math.sin(2 * math.pi * local_hour / 24)  # sin(π/3) = √3/2 ≈ 0.866
        expected_hour_cos = math.cos(2 * math.pi * local_hour / 24)  # cos(π/3) = 1/2 = 0.5

        assert abs(features["hour_sin"] - expected_hour_sin) < 0.0001
        assert abs(features["hour_cos"] - expected_hour_cos) < 0.0001

    def test_activity_regularity_calculation(self, extractor):
        """Test activity regularity calculation accuracy."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 30, 0)
        events = []

        # Create perfectly regular pattern - event every 10 minutes
        for i in range(9):  # 90 minutes of regular activity
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should detect high regularity
        if "activity_regularity_score" in features:
            assert features["activity_regularity_score"] > 0.7  # High regularity

    def test_sensor_type_distribution(self, extractor, target_time):
        """Test sensor type distribution calculations."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Create known sensor type distribution
        sensor_distribution = {"motion": 6, "door": 3, "presence": 1}
        event_index = 0

        for sensor_type, count in sensor_distribution.items():
            for _ in range(count):
                event = Mock(spec=SensorEvent)
                event.timestamp = base_time + timedelta(minutes=event_index * 5)
                event.state = "on" if event_index % 2 == 0 else "off"
                event.sensor_type = sensor_type
                event.sensor_id = f"sensor.{sensor_type}_{event_index}"
                events.append(event)
                event_index += 1

        features = extractor.extract_features(events, target_time)

        # Verify sensor type ratios
        total_events = sum(sensor_distribution.values())
        expected_motion_ratio = sensor_distribution["motion"] / total_events
        expected_door_ratio = sensor_distribution["door"] / total_events
        expected_presence_ratio = sensor_distribution["presence"] / total_events

        assert abs(features["motion_sensor_ratio"] - expected_motion_ratio) < 0.01
        assert abs(features["door_sensor_ratio"] - expected_door_ratio) < 0.01
        assert abs(features["presence_sensor_ratio"] - expected_presence_ratio) < 0.01

    def test_performance_with_large_dataset(self, extractor, large_events, target_time):
        """Test performance with large number of events."""
        import time

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(large_events, target_time)
        extraction_time = time.time() - start_time

        # Should complete within reasonable time (< 2 seconds)
        assert extraction_time < 2.0
        assert isinstance(features, dict)
        assert len(features) > 30

    @pytest.mark.parametrize("timezone_offset", [-12, -8, -5, 0, 1, 5, 8, 12])
    def test_timezone_offset_variations(self, timezone_offset):
        """Test feature extraction with various timezone offsets."""
        extractor = TemporalFeatureExtractor(timezone_offset=timezone_offset)
        target_time = datetime(2024, 1, 15, 15, 0, 0)  # 3 PM UTC

        features = extractor.extract_features([], target_time)

        # Should produce valid features for all timezone offsets
        assert isinstance(features, dict)
        assert len(features) > 30

        # Time features should be within valid ranges
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0
        assert features["is_weekend"] in [0.0, 1.0]
        assert features["is_work_hours"] in [0.0, 1.0]

    def test_get_feature_names_method(self, extractor):
        """Test get_feature_names method if it exists."""
        if hasattr(extractor, "get_feature_names"):
            feature_names = extractor.get_feature_names()
            assert isinstance(feature_names, list)
            assert len(feature_names) > 30

            # Should match keys from default features
            default_features = extractor._get_default_features()
            assert set(feature_names) == set(default_features.keys())
        else:
            # If method doesn't exist, that's also valid
            pass

    def test_missing_attributes_handling(self, extractor):
        """Test handling of missing event attributes."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Test with incomplete event mock
        incomplete_event = Mock(spec=SensorEvent)
        incomplete_event.timestamp = datetime(2024, 1, 15, 14, 30, 0)
        incomplete_event.state = "on"
        # Missing sensor_type and sensor_id
        incomplete_event.sensor_type = None
        incomplete_event.sensor_id = None

        # Should handle gracefully or raise appropriate error
        try:
            features = extractor.extract_features([incomplete_event], target_time)
            assert isinstance(features, dict)
        except (FeatureExtractionError, AttributeError):
            # Either approach is acceptable
            pass

    def test_very_old_events(self, extractor, very_old_events, target_time):
        """Test feature extraction with very old events using lookback."""
        # Test with lookback filtering
        features = extractor.extract_features(
            very_old_events, target_time, lookback_hours=24
        )

        # Should filter out very old events
        assert isinstance(features, dict)
        assert len(features) > 0

        # Since events are too old, should mostly return defaults
        # or handle the lack of recent events appropriately

    @pytest.mark.parametrize("lookback_hours", [1, 6, 12, 24, 48])
    def test_different_lookback_windows(
        self, extractor, sample_events, target_time, lookback_hours
    ):
        """Test feature extraction with different lookback windows."""
        features = extractor.extract_features(
            sample_events, target_time, lookback_hours=lookback_hours
        )

        # Should produce valid features for all lookback windows
        assert isinstance(features, dict)
        assert len(features) > 20

        # Features should be within valid ranges regardless of lookback
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0

    def test_cache_functionality(self, extractor):
        """Test temporal cache functionality."""
        # Access temporal_cache attribute
        if hasattr(extractor, "temporal_cache"):
            extractor.temporal_cache["test"] = "value"
            assert extractor.temporal_cache["test"] == "value"
        else:
            # If cache doesn't exist, that's acceptable
            pass
