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

    def test_recent_activity_features(self, extractor):
        """Test recent activity vs historical activity."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)  # Start earlier
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Historical activity (more than 1 hour ago)
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        # Recent activity (last 30 minutes)
        recent_base = datetime(2024, 1, 15, 14, 30, 0)
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = recent_base + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_recent_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should detect both recent and historical activity
        assert features["recent_activity_rate"] > 0
        assert features["historical_activity_rate"] > 0
        assert features["activity_trend"] != 0  # Should show some trend

    def test_timezone_handling(self):
        """Test timezone offset handling."""
        # Test different timezone offsets
        extractors = [
            TemporalFeatureExtractor(timezone_offset=0),  # UTC
            TemporalFeatureExtractor(timezone_offset=-8),  # PST
            TemporalFeatureExtractor(timezone_offset=5),  # India
        ]

        target_time = datetime(2024, 1, 15, 15, 0, 0)  # 3 PM

        for tz_offset, extractor in zip([0, -8, 5], extractors):
            features = extractor.extract_features([], target_time)

            # Local hour should be adjusted by timezone offset
            local_hour = (15 + tz_offset) % 24
            expected_sin = math.sin(2 * math.pi * local_hour / 24)

            assert abs(features["hour_sin"] - expected_sin) < 0.001

    def test_feature_consistency(self, extractor, sample_events):
        """Test that features are consistent across multiple calls."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Extract features multiple times
        features1 = extractor.extract_features(sample_events, target_time)
        features2 = extractor.extract_features(sample_events, target_time)
        features3 = extractor.extract_features(sample_events, target_time)

        # Results should be identical
        assert features1 == features2
        assert features2 == features3

    def test_edge_case_no_on_events(self, extractor):
        """Test handling when no 'on' events exist."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Create events with only "off" states
        events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should handle gracefully
        assert features["time_since_last_on"] == 3600.0  # Default
        assert features["avg_on_duration"] == 0.0
        assert features["on_event_count"] == 0.0

    def test_edge_case_no_off_events(self, extractor):
        """Test handling when no 'off' events exist."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Create events with only "on" states
        events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "on"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should handle gracefully
        assert features["time_since_last_off"] == 3600.0  # Default
        assert features["avg_off_duration"] == 0.0
        assert features["off_event_count"] == 0.0

    def test_very_old_events(self, extractor):
        """Test handling of very old events."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Create very old events (more than lookback window)
        old_events = []
        very_old_time = datetime(2024, 1, 10, 12, 0, 0)  # 5 days ago

        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = very_old_time + timedelta(minutes=i * 10)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            old_events.append(event)

        features = extractor.extract_features(
            old_events, target_time, lookback_hours=24
        )

        # Should filter out old events and return defaults
        expected_defaults = extractor._get_default_features()
        assert features == expected_defaults

    def test_malformed_events(self, extractor):
        """Test error handling with malformed events."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Create malformed event
        bad_event = Mock(spec=SensorEvent)
        bad_event.timestamp = None  # This should cause an error
        bad_event.state = "on"
        bad_event.sensor_type = "motion"
        bad_event.sensor_id = "sensor.motion"

        with pytest.raises(FeatureExtractionError):
            extractor.extract_features([bad_event], target_time)

    def test_performance_large_dataset(self, extractor):
        """Test performance with large event dataset."""
        import time

        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Create large dataset
        large_events = []
        for i in range(5000):  # 5000 events
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(seconds=i * 2)
            event.state = "on" if i % 3 == 0 else "off"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i % 10}"
            large_events.append(event)

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(large_events, target_time)
        extraction_time = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert extraction_time < 5.0
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_feature_names_method(self, extractor):
        """Test get_feature_names method."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 30

        # Should match default features keys
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_cache_functionality(self, extractor):
        """Test temporal cache functionality."""
        # Add something to cache
        extractor.temporal_cache["test"] = "value"
        assert "test" in extractor.temporal_cache

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.temporal_cache) == 0

    @pytest.mark.parametrize("lookback_hours", [1, 6, 12, 24, 48])
    def test_different_lookback_windows(self, extractor, sample_events, lookback_hours):
        """Test feature extraction with different lookback windows."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        features = extractor.extract_features(
            sample_events, target_time, lookback_hours=lookback_hours
        )

        # Should return valid features regardless of lookback window
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_month_and_season_features(self, extractor):
        """Test month and season detection."""
        # Test different seasons
        test_cases = [
            (
                datetime(2024, 1, 15, 15, 0, 0),
                1,
                True,
                False,
                False,
                False,
            ),  # January - Winter
            (
                datetime(2024, 4, 15, 15, 0, 0),
                4,
                False,
                True,
                False,
                False,
            ),  # April - Spring
            (
                datetime(2024, 7, 15, 15, 0, 0),
                7,
                False,
                False,
                True,
                False,
            ),  # July - Summer
            (
                datetime(2024, 10, 15, 15, 0, 0),
                10,
                False,
                False,
                False,
                True,
            ),  # October - Autumn
        ]

        for (
            target_time,
            month,
            is_winter,
            is_spring,
            is_summer,
            is_autumn,
        ) in test_cases:
            features = extractor.extract_features([], target_time)

            # Check month encoding
            expected_sin = math.sin(2 * math.pi * month / 12)
            expected_cos = math.cos(2 * math.pi * month / 12)

            assert abs(features["month_sin"] - expected_sin) < 0.001
            assert abs(features["month_cos"] - expected_cos) < 0.001

            # Check season detection
            assert features["is_winter"] == (1.0 if is_winter else 0.0)
            assert features["is_spring"] == (1.0 if is_spring else 0.0)
            assert features["is_summer"] == (1.0 if is_summer else 0.0)
            assert features["is_autumn"] == (1.0 if is_autumn else 0.0)

    @pytest.mark.asyncio
    async def test_concurrent_extraction(self, extractor, sample_events):
        """Test thread safety of temporal feature extraction."""
        import asyncio

        target_time = datetime(2024, 1, 15, 15, 0, 0)

        async def extract_features():
            return extractor.extract_features(sample_events, target_time)

        # Run multiple extractions concurrently
        tasks = [extract_features() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_stats_tracking(self, extractor):
        """Test extraction statistics tracking."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Get initial stats
        initial_stats = extractor.get_extraction_stats()

        # Perform some extractions
        extractor.extract_features([], target_time)
        extractor.extract_features([], target_time)

        # Check stats were updated
        final_stats = extractor.get_extraction_stats()
        assert final_stats["total_extractions"] > initial_stats["total_extractions"]

        # Reset stats
        extractor.reset_stats()
        reset_stats = extractor.get_extraction_stats()
        assert reset_stats["total_extractions"] == 0
