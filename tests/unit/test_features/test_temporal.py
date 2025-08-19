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

    def test_extract_features_with_sample_data(
        self, extractor, sample_events, target_time
    ):
        """Test feature extraction with realistic sample data."""
        features = extractor.extract_features(sample_events, target_time)

        # Verify basic feature presence
        assert isinstance(features, dict)
        assert len(features) > 30  # Should have many temporal features

        # Check key temporal features exist
        expected_features = [
            "time_since_last_event",
            "time_since_last_on",
            "time_since_last_off",
            "current_state_duration",
            "avg_on_duration",
            "hour_sin",
            "hour_cos",
            "is_weekend",
            "is_work_hours",
            "overall_activity_rate",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_extract_features_empty_events(self, extractor, empty_events, target_time):
        """Test behavior with empty event list."""
        features = extractor.extract_features(empty_events, target_time)

        # Should return default features
        expected_defaults = extractor._get_default_features()
        assert features == expected_defaults

        # Verify reasonable default values
        assert features["time_since_last_event"] == 3600.0
        assert features["hour_sin"] == math.sin(2 * math.pi * 15 / 24)  # 3 PM
        assert features["is_work_hours"] == 1.0

    def test_extract_features_single_event(self, extractor, single_event, target_time):
        """Test feature extraction with single event."""
        features = extractor.extract_features(single_event, target_time)

        # Should handle single event gracefully
        assert isinstance(features, dict)
        assert len(features) > 30

        # Time since last event should be reasonable
        assert features["time_since_last_event"] > 0
        assert features["total_events"] == 1.0

    def test_time_calculations(self, extractor):
        """Test time-based feature calculations."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create events with known timing
        event_times = [0, 5, 15, 30, 45]  # Minutes from base_time
        for i, minutes in enumerate(event_times):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=minutes)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Check time calculations
        assert features["time_since_last_event"] == 900.0  # 15 minutes in seconds
        assert features["total_events"] == 5.0

        # Activity rate should be reasonable
        assert features["overall_activity_rate"] > 0

    def test_cyclical_time_features(self, extractor):
        """Test cyclical time encoding."""
        # Test different hours
        test_cases = [
            (datetime(2024, 1, 15, 0, 0, 0), 0),  # Midnight
            (datetime(2024, 1, 15, 6, 0, 0), 6),  # 6 AM
            (datetime(2024, 1, 15, 12, 0, 0), 12),  # Noon
            (datetime(2024, 1, 15, 18, 0, 0), 18),  # 6 PM
        ]

        for target_time, hour in test_cases:
            features = extractor.extract_features([], target_time)

            expected_sin = math.sin(2 * math.pi * hour / 24)
            expected_cos = math.cos(2 * math.pi * hour / 24)

            assert abs(features["hour_sin"] - expected_sin) < 0.001
            assert abs(features["hour_cos"] - expected_cos) < 0.001

    def test_day_of_week_features(self, extractor):
        """Test day of week detection."""
        # Test different days (2024-01-15 is Monday)
        test_cases = [
            (datetime(2024, 1, 15, 15, 0, 0), 0, False),  # Monday
            (datetime(2024, 1, 16, 15, 0, 0), 1, False),  # Tuesday
            (datetime(2024, 1, 20, 15, 0, 0), 5, True),  # Saturday
            (datetime(2024, 1, 21, 15, 0, 0), 6, True),  # Sunday
        ]

        for target_time, expected_day, is_weekend in test_cases:
            features = extractor.extract_features([], target_time)

            expected_sin = math.sin(2 * math.pi * expected_day / 7)
            expected_cos = math.cos(2 * math.pi * expected_day / 7)

            assert abs(features["day_sin"] - expected_sin) < 0.001
            assert abs(features["day_cos"] - expected_cos) < 0.001
            assert features["is_weekend"] == (1.0 if is_weekend else 0.0)

    def test_work_hours_detection(self, extractor):
        """Test work hours detection."""
        # Test different times
        test_cases = [
            (datetime(2024, 1, 15, 8, 0, 0), False),  # 8 AM - before work
            (datetime(2024, 1, 15, 9, 0, 0), True),  # 9 AM - work hours
            (datetime(2024, 1, 15, 12, 0, 0), True),  # Noon - work hours
            (datetime(2024, 1, 15, 17, 0, 0), True),  # 5 PM - work hours
            (datetime(2024, 1, 15, 18, 0, 0), False),  # 6 PM - after work
            (datetime(2024, 1, 20, 12, 0, 0), False),  # Saturday - weekend
        ]

        for target_time, is_work_hours in test_cases:
            features = extractor.extract_features([], target_time)
            assert features["is_work_hours"] == (1.0 if is_work_hours else 0.0)

    def test_state_duration_calculations(self, extractor):
        """Test state duration calculations."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create sequence: off -> on (5 min) -> off (10 min) -> on (current)
        state_sequence = [
            (0, "off"),
            (5, "on"),
            (15, "off"),
            (55, "on"),  # 5 minutes before target
        ]

        for minutes, state in state_sequence:
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=minutes)
            event.state = state
            event.sensor_type = "motion"
            event.sensor_id = "sensor.motion"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Current state should be "on" for 5 minutes
        assert features["current_state_duration"] == 300.0  # 5 minutes in seconds

        # Should have detected on/off durations
        assert features["avg_on_duration"] > 0
        assert features["avg_off_duration"] > 0

    def test_activity_patterns(self, extractor):
        """Test activity pattern detection."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create regular activity pattern - event every 10 minutes
        for i in range(6):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = "sensor.motion"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should detect regular activity
        assert features["overall_activity_rate"] > 0
        assert features["activity_regularity"] > 0
        assert features["total_events"] == 6.0

    def test_sensor_type_features(self, extractor):
        """Test sensor type diversity features."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create events from different sensor types
        sensor_types = ["motion", "door", "presence", "motion", "door"]
        for i, sensor_type in enumerate(sensor_types):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = sensor_type
            event.sensor_id = f"sensor.{sensor_type}_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should detect sensor diversity
        assert features["sensor_type_count"] == 3.0  # motion, door, presence
        assert features["motion_sensor_ratio"] > 0
        assert features["door_sensor_ratio"] > 0
        assert features["presence_sensor_ratio"] > 0

    def test_room_state_features(self, extractor, room_states, target_time):
        """Test room state duration features."""
        # Create minimal events list for the extractor to work
        base_time = datetime(2024, 1, 15, 14, 30, 0)
        events = []
        event = Mock(spec=SensorEvent)
        event.timestamp = base_time
        event.state = "on"
        event.sensor_type = "motion"
        event.sensor_id = "sensor.motion"
        event.room_id = "living_room"
        events.append(event)

        features = extractor.extract_features(events, target_time, room_states)

        # Should include room state features
        assert "room_occupied_duration" in features
        assert "room_vacant_duration" in features
        assert "avg_occupancy_confidence" in features

    def test_edge_cases_malformed_events(self, extractor):
        """Test handling of malformed events."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Test with events having None timestamps
        bad_events = [Mock(spec=SensorEvent)]
        bad_events[0].timestamp = None
        bad_events[0].state = "on"
        bad_events[0].sensor_type = "motion"
        bad_events[0].sensor_id = "sensor.test"
        bad_events[0].room_id = "living_room"

        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(bad_events, target_time)

    def test_timezone_offset_handling(self):
        """Test timezone offset functionality."""
        extractor_utc = TemporalFeatureExtractor(timezone_offset=0)
        extractor_pst = TemporalFeatureExtractor(timezone_offset=-8)

        target_time = datetime(2024, 1, 15, 12, 0, 0)  # Noon UTC

        features_utc = extractor_utc.extract_features([], target_time)
        features_pst = extractor_pst.extract_features([], target_time)

        # The features should be different due to timezone offset
        # UTC noon (12) vs PST 4 AM (4)
        assert features_utc["hour_sin"] != features_pst["hour_sin"]
        assert features_utc["hour_cos"] != features_pst["hour_cos"]

    def test_feature_cache_functionality(self, extractor):
        """Test feature cache operations."""
        # Test cache is initialized
        assert isinstance(extractor.feature_cache, dict)
        assert len(extractor.feature_cache) == 0

        # Add item to cache
        extractor.feature_cache["test_key"] = {"feature": 1.0}
        assert "test_key" in extractor.feature_cache

        # Clear cache if method exists
        if hasattr(extractor, "clear_cache"):
            extractor.clear_cache()
            assert len(extractor.feature_cache) == 0

    def test_historical_patterns(self, extractor):
        """Test historical pattern analysis."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create historical pattern - same time yesterday
        yesterday_time = base_time - timedelta(days=1)
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = yesterday_time + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        # Add today's events
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 15)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_today_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should detect historical patterns
        assert "same_hour_activity_rate" in features
        assert "same_day_activity_rate" in features
        assert "weekday_pattern_similarity" in features

    def test_transition_timing_features(self, extractor):
        """Test state transition timing analysis."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create alternating on/off pattern with varying durations
        durations = [5, 10, 15, 8, 12]  # minutes
        cumulative_time = 0

        for i, duration in enumerate(durations):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=cumulative_time)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)
            cumulative_time += duration

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
        # Test hour encoding for noon (12:00)
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        features = extractor.extract_features([], target_time)

        # At noon, hour angle should be π (180 degrees)
        expected_hour_sin = math.sin(2 * math.pi * 12 / 24)  # sin(π) = 0
        expected_hour_cos = math.cos(2 * math.pi * 12 / 24)  # cos(π) = -1

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

        # Regular pattern should have high regularity score
        assert features["activity_regularity"] > 0.8  # Very regular pattern

    def test_sensor_diversity_entropy(self, extractor):
        """Test sensor type diversity calculation using entropy."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create events with known sensor distribution
        # 50% motion, 30% door, 20% presence
        sensor_distribution = {"motion": 5, "door": 3, "presence": 2}

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

    def test_performance_with_large_dataset(self, extractor):
        """Test performance with large number of events."""
        import time

        # Create large dataset
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create 10,000 events over 2 weeks
        for i in range(10000):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 2)  # Every 2 minutes
            event.state = "on" if i % 3 == 0 else "off"
            event.sensor_type = ["motion", "door", "presence"][i % 3]
            event.sensor_id = f"sensor.{event.sensor_type}_{i % 100}"
            events.append(event)

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(events, target_time)
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
