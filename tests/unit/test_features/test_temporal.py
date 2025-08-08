"""
Unit tests for temporal feature extraction.

This module tests the TemporalFeatureExtractor for accuracy, edge cases,
and performance with realistic occupancy data patterns.
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import pytest

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

        # Create realistic event sequence
        states = ["off", "on", "on", "off", "on", "off"]
        sensor_types = ["motion", "presence", "door", "motion", "presence", "motion"]

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
        assert features["time_since_last_event"] > 0
        assert features["current_state_duration"] > 0

    def test_time_since_features(self, extractor, sample_events, target_time):
        """Test time-since-last-event features calculation."""
        features = extractor._extract_time_since_features(sample_events, target_time)

        # Check that time calculations are reasonable
        assert "time_since_last_event" in features
        assert "time_since_last_on" in features
        assert "time_since_last_off" in features
        assert "time_since_last_motion" in features

        # Values should be positive and capped at 24 hours
        for key in features:
            assert features[key] >= 0
            assert features[key] <= 86400.0

    def test_duration_features(self, extractor, sample_events, target_time):
        """Test state duration feature calculations."""
        features = extractor._extract_duration_features(sample_events, target_time)

        # Check duration features
        assert "current_state_duration" in features
        assert "avg_on_duration" in features
        assert "avg_off_duration" in features
        assert "max_on_duration" in features
        assert "max_off_duration" in features

        # Durations should be positive
        for key in features:
            assert features[key] >= 0

    def test_cyclical_features(self, extractor, target_time):
        """Test cyclical time encoding features."""
        features = extractor._extract_cyclical_features(target_time)

        # Check cyclical encoding features
        cyclical_features = [
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "month_sin",
            "month_cos",
            "day_of_month_sin",
            "day_of_month_cos",
        ]

        for feature in cyclical_features:
            assert feature in features
            # Sin/cos values should be between -1 and 1
            assert -1.0 <= features[feature] <= 1.0

        # Boolean features
        assert features["is_weekend"] in [0.0, 1.0]
        assert features["is_work_hours"] in [0.0, 1.0]
        assert features["is_sleep_hours"] in [0.0, 1.0]

    def test_historical_patterns(self, extractor, sample_events, target_time):
        """Test historical pattern feature extraction."""
        features = extractor._extract_historical_patterns(sample_events, target_time)

        # Check pattern features
        pattern_features = [
            "hour_activity_rate",
            "day_activity_rate",
            "overall_activity_rate",
            "similar_time_activity_rate",
        ]

        for feature in pattern_features:
            assert feature in features
            # Activity rates should be between 0 and 1
            assert 0.0 <= features[feature] <= 1.0

    def test_transition_timing_features(self, extractor, sample_events, target_time):
        """Test transition timing feature calculations."""
        features = extractor._extract_transition_timing_features(
            sample_events, target_time
        )

        # Check timing features
        assert "avg_transition_interval" in features
        assert "recent_transition_rate" in features
        assert "time_variability" in features

        # Values should be non-negative
        for key in features:
            assert features[key] >= 0.0

    def test_room_state_features(self, extractor, room_states, target_time):
        """Test room state feature extraction."""
        features = extractor._extract_room_state_features(room_states, target_time)

        # Check room state features
        assert "avg_occupancy_confidence" in features
        assert "recent_occupancy_ratio" in features
        assert "state_stability" in features

        # Confidence and ratio should be between 0 and 1
        assert 0.0 <= features["avg_occupancy_confidence"] <= 1.0
        assert 0.0 <= features["recent_occupancy_ratio"] <= 1.0
        assert features["state_stability"] >= 0.0

    def test_timezone_handling(self):
        """Test timezone offset handling in feature extraction."""
        # Test different timezone offsets
        utc_extractor = TemporalFeatureExtractor(timezone_offset=0)
        pst_extractor = TemporalFeatureExtractor(timezone_offset=-8)

        target_time = datetime(2024, 1, 15, 20, 0, 0)  # 8 PM UTC

        utc_features = utc_extractor._extract_cyclical_features(target_time)
        pst_features = pst_extractor._extract_cyclical_features(target_time)

        # Hour encodings should be different (8 PM UTC vs 12 PM PST)
        assert utc_features["hour_sin"] != pst_features["hour_sin"]
        assert utc_features["hour_cos"] != pst_features["hour_cos"]

        # Work hours should be different
        assert utc_features["is_work_hours"] != pst_features["is_work_hours"]

    def test_feature_consistency(self, extractor, sample_events, target_time):
        """Test that feature extraction is consistent across multiple calls."""
        features1 = extractor.extract_features(sample_events, target_time)
        features2 = extractor.extract_features(sample_events, target_time)

        # Results should be identical
        assert features1 == features2

    def test_feature_names_method(self, extractor):
        """Test get_feature_names method."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 30

        # Should match default features keys
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_cache_operations(self, extractor):
        """Test cache clear functionality."""
        # Add something to cache
        extractor.feature_cache["test"] = "value"
        assert "test" in extractor.feature_cache

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.feature_cache) == 0

    def test_batch_feature_extraction(
        self, extractor, sample_events, target_time, room_states
    ):
        """Test batch feature extraction method."""
        # Create batch requests
        event_batches = [
            (sample_events, target_time),
            (sample_events[:3], target_time + timedelta(minutes=10)),
        ]
        room_states_batches = [room_states, room_states[:2]]

        results = extractor.extract_batch_features(event_batches, room_states_batches)

        assert len(results) == 2
        assert all(isinstance(result, dict) for result in results)

    @pytest.mark.parametrize("timezone_offset", [-12, -8, 0, 5, 12])
    def test_timezone_offsets(self, timezone_offset):
        """Test various timezone offsets."""
        extractor = TemporalFeatureExtractor(timezone_offset=timezone_offset)
        target_time = datetime(2024, 6, 15, 12, 0, 0)  # Noon UTC

        features = extractor._extract_cyclical_features(target_time)

        # Verify timezone adjustment in hour calculations
        adjusted_hour = (12 + timezone_offset) % 24
        expected_hour_sin = math.sin(2 * math.pi * adjusted_hour / 24)
        expected_hour_cos = math.cos(2 * math.pi * adjusted_hour / 24)

        assert abs(features["hour_sin"] - expected_hour_sin) < 1e-10
        assert abs(features["hour_cos"] - expected_hour_cos) < 1e-10

    def test_edge_case_time_boundaries(self, extractor):
        """Test edge cases around time boundaries."""
        # Test midnight
        midnight = datetime(2024, 1, 1, 0, 0, 0)
        features = extractor._extract_cyclical_features(midnight)
        assert features["hour_sin"] == 0.0
        assert features["hour_cos"] == 1.0

        # Test end of year
        new_year = datetime(2024, 12, 31, 23, 59, 59)
        features = extractor._extract_cyclical_features(new_year)
        assert isinstance(features["month_sin"], float)
        assert isinstance(features["day_of_month_sin"], float)

    def test_large_event_sequences(self, extractor):
        """Test performance with large event sequences."""
        # Create large event sequence
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        large_events = []

        for i in range(1000):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i % 10}"
            event.room_id = "test_room"
            large_events.append(event)

        target_time = base_time + timedelta(hours=20)

        # Should handle large sequences without errors
        features = extractor.extract_features(large_events, target_time)
        assert isinstance(features, dict)
        assert len(features) > 30

    def test_error_handling(self, extractor):
        """Test error handling in feature extraction."""
        # Test with malformed events
        bad_events = [Mock(spec=SensorEvent)]
        bad_events[0].timestamp = None  # This should cause an error
        bad_events[0].state = "on"
        bad_events[0].sensor_type = "motion"
        bad_events[0].sensor_id = "sensor.test"
        bad_events[0].room_id = "test_room"

        target_time = datetime(2024, 1, 15, 15, 0, 0)

        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(bad_events, target_time)

    def test_statistical_calculations_accuracy(self, extractor):
        """Test accuracy of statistical calculations."""
        # Create events with known patterns
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Create 5 "on" periods of 10 minutes each, separated by 5 minutes "off"
        for i in range(5):
            # On event
            on_event = Mock(spec=SensorEvent)
            on_event.timestamp = base_time + timedelta(minutes=i * 15)
            on_event.state = "on"
            on_event.sensor_type = "motion"
            on_event.sensor_id = "sensor.motion"
            on_event.room_id = "test_room"
            events.append(on_event)

            # Off event (10 minutes later)
            off_event = Mock(spec=SensorEvent)
            off_event.timestamp = base_time + timedelta(minutes=i * 15 + 10)
            off_event.state = "off"
            off_event.sensor_type = "motion"
            off_event.sensor_id = "sensor.motion"
            off_event.room_id = "test_room"
            events.append(off_event)

        target_time = base_time + timedelta(hours=2)
        features = extractor._extract_duration_features(events, target_time)

        # Check that average durations are calculated correctly
        # Expected: 5 "on" periods of 10 minutes each
        expected_on_duration = 600.0  # 10 minutes in seconds
        assert (
            abs(features["avg_on_duration"] - expected_on_duration) < 60.0
        )  # Within 1 minute tolerance

    def test_memory_efficiency(self, extractor, sample_events):
        """Test memory usage doesn't grow excessively."""
        import sys

        initial_size = sys.getsizeof(extractor)
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Run extraction multiple times
        for _ in range(100):
            extractor.extract_features(sample_events, target_time)

        final_size = sys.getsizeof(extractor)

        # Memory usage shouldn't grow significantly
        assert final_size - initial_size < 1000  # Less than 1KB growth

    def test_feature_value_ranges(self, extractor, sample_events, target_time):
        """Test that all feature values are within expected ranges."""
        features = extractor.extract_features(sample_events, target_time)

        # Time-based features should be positive and reasonable
        time_features = [
            "time_since_last_event",
            "time_since_last_on",
            "time_since_last_off",
            "current_state_duration",
            "avg_on_duration",
            "avg_off_duration",
        ]

        for feature in time_features:
            if feature in features:
                assert features[feature] >= 0
                assert features[feature] <= 86400.0  # Max 24 hours

        # Rate features should be between 0 and 1
        rate_features = [
            "hour_activity_rate",
            "day_activity_rate",
            "overall_activity_rate",
            "recent_occupancy_ratio",
            "avg_occupancy_confidence",
        ]

        for feature in rate_features:
            if feature in features:
                assert 0.0 <= features[feature] <= 1.0

        # Boolean features should be 0 or 1
        boolean_features = [
            "is_weekend",
            "is_work_hours",
            "is_sleep_hours",
            "is_cold",
            "is_comfortable_temp",
            "is_warm",
        ]

        for feature in boolean_features:
            if feature in features:
                assert features[feature] in [0.0, 1.0]

    @pytest.mark.asyncio
    async def test_concurrent_extraction(self, extractor, sample_events, target_time):
        """Test thread safety of feature extraction."""
        import asyncio

        async def extract_features():
            return extractor.extract_features(sample_events, target_time)

        # Run multiple extractions concurrently
        tasks = [extract_features() for _ in range(10)]
        results = await asyncio.gather(*tasks)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result


class TestTemporalFeatureExtractorEdgeCases:
    """Additional edge case tests for TemporalFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a temporal feature extractor instance."""
        return TemporalFeatureExtractor()

    def test_events_in_future(self, extractor):
        """Test handling of events that occur after target time."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events that occur after target time
        future_event = Mock(spec=SensorEvent)
        future_event.timestamp = target_time + timedelta(hours=2)
        future_event.state = "on"
        future_event.sensor_type = "motion"
        future_event.sensor_id = "sensor.motion"
        future_event.room_id = "test_room"

        # Should handle gracefully (likely filter out future events)
        features = extractor.extract_features([future_event], target_time)
        assert isinstance(features, dict)

    def test_duplicate_timestamps(self, extractor):
        """Test handling of events with duplicate timestamps."""
        timestamp = datetime(2024, 1, 15, 12, 0, 0)
        target_time = timestamp + timedelta(hours=1)

        # Create events with same timestamp
        events = []
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = timestamp
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            event.room_id = "test_room"
            events.append(event)

        # Should handle duplicate timestamps
        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)

    def test_extreme_time_differences(self, extractor):
        """Test with very large time differences."""
        base_time = datetime(2020, 1, 1, 0, 0, 0)
        target_time = datetime(2024, 1, 1, 0, 0, 0)  # 4 years later

        old_event = Mock(spec=SensorEvent)
        old_event.timestamp = base_time
        old_event.state = "on"
        old_event.sensor_type = "motion"
        old_event.sensor_id = "sensor.motion"
        old_event.room_id = "test_room"

        features = extractor.extract_features([old_event], target_time)

        # Time since features should be capped at 24 hours
        assert features["time_since_last_event"] == 86400.0

    def test_rapid_state_changes(self, extractor):
        """Test with very rapid state changes (sub-second)."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(minutes=30)
        events = []

        # Create rapid state changes (every 100ms)
        for i in range(100):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(milliseconds=i * 100)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = "sensor.motion"
            event.room_id = "test_room"
            events.append(event)

        # Should handle rapid changes
        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)
        assert (
            features["recent_transition_rate"] > 0
        )  # Should detect high transition rate

    def test_missing_sensor_types(self, extractor):
        """Test handling of missing or None sensor types."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=1)

        event = Mock(spec=SensorEvent)
        event.timestamp = base_time
        event.state = "on"
        event.sensor_type = None  # Missing sensor type
        event.sensor_id = "sensor.unknown"
        event.room_id = "test_room"

        # Should handle gracefully
        features = extractor.extract_features([event], target_time)
        assert isinstance(features, dict)
