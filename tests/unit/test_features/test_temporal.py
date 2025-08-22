"""
Comprehensive unit tests for temporal feature extraction.

This module provides complete test coverage for TemporalFeatureExtractor including:
- Initialization and configuration
- Basic feature extraction functionality
- Time-since and duration features
- Cyclical encoding features
- Historical pattern analysis
- Transition timing features
- Room state integration
- Generic sensor features
- Edge cases and error handling
- Performance and concurrency testing
"""

from datetime import datetime, timedelta
import math
import os
import queue
import threading
import time
from typing import Any, Dict, List
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import psutil
import pytest
import statistics

from src.core.constants import TEMPORAL_FEATURE_NAMES
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.temporal import TemporalFeatureExtractor


class TestTemporalFeatureExtractorInitialization:
    """Test TemporalFeatureExtractor initialization and configuration."""

    def test_init_default_timezone(self):
        """Test initialization with default timezone."""
        extractor = TemporalFeatureExtractor()

        assert extractor.timezone_offset == 0
        assert extractor.feature_cache == {}
        assert extractor.temporal_cache == {}

    def test_init_custom_timezone(self):
        """Test initialization with custom timezone."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)

        assert extractor.timezone_offset == -8
        assert extractor.feature_cache == {}
        assert extractor.temporal_cache == {}

    def test_init_positive_timezone(self):
        """Test initialization with positive timezone offset."""
        extractor = TemporalFeatureExtractor(timezone_offset=5)

        assert extractor.timezone_offset == 5

    def test_extreme_timezone_offsets(self):
        """Test with extreme timezone offsets."""
        extreme_extractor = TemporalFeatureExtractor(
            timezone_offset=14
        )  # Maximum UTC offset
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        features = extreme_extractor.extract_features([], target_time)

        # Should handle extreme offsets without error
        assert isinstance(features, dict)
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0

    def test_negative_timezone_offset(self):
        """Test with negative timezone offset."""
        extractor = TemporalFeatureExtractor(timezone_offset=-12)  # Minimum UTC offset
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        features = extractor.extract_features([], target_time)

        assert isinstance(features, dict)
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0


class TestTemporalFeatureExtractorBasicFeatures:
    """Test basic feature extraction functionality."""

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
        sensor_types = ["motion", "presence", "door", "motion", "presence", "motion"]

        for i, (state, sensor_type) in enumerate(zip(states, sensor_types)):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.state = state
            event.sensor_type = sensor_type
            event.sensor_id = f"sensor.test_{sensor_type}_{i}"
            event.room_id = "living_room"
            event.attributes = {"temperature": 20.5 + i, "humidity": 45.0}
            events.append(event)

        return events

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

    def test_empty_events(self, extractor, target_time):
        """Test feature extraction with no events."""
        features = extractor.extract_features([], target_time)

        # Should return default features
        assert isinstance(features, dict)
        assert len(features) > 0

        # Time-since features should have default values
        assert features["time_since_last_on"] >= 0.0
        assert features["time_since_last_off"] >= 0.0

    def test_extract_features_with_none_events(self, extractor, target_time):
        """Test feature extraction with None events list returns default features."""
        features = extractor.extract_features(None, target_time)

        # Should return default features without raising exception
        assert isinstance(features, dict)
        assert len(features) > 0

        # Check some expected default features
        assert "time_since_last_event" in features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert features["time_since_last_event"] == 3600.0  # Default 1 hour

    def test_extract_features_with_room_states(
        self, extractor, sample_events, target_time
    ):
        """Test feature extraction with room states."""
        # Create sample room states
        base_time = datetime(2024, 1, 15, 9, 0, 0)
        room_states = []
        for i in range(5):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(hours=i)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8 + i * 0.05
            room_states.append(state)

        features = extractor.extract_features(sample_events, target_time, room_states)

        assert isinstance(features, dict)
        # Should include room state-based features
        if "avg_occupancy_confidence" in features:
            assert "recent_occupancy_ratio" in features
            assert "state_stability" in features

    def test_extract_features_with_lookback_filter(
        self, extractor, sample_events, target_time
    ):
        """Test feature extraction with lookback hours filter."""
        # Only look back 1 hour (should filter out most events)
        features = extractor.extract_features(
            sample_events, target_time, lookback_hours=1
        )

        assert isinstance(features, dict)
        # Should still return features but with filtered events
        assert "time_since_last_event" in features

    def test_extract_features_exception_handling(self, extractor):
        """Test feature extraction exception handling."""
        # Create mock events that will cause an exception
        bad_events = [Mock()]
        bad_events[0].timestamp = None  # This should cause an error
        bad_events[0].room_id = "test_room"

        target_time = datetime(2024, 1, 15, 12, 0, 0)

        with pytest.raises(FeatureExtractionError) as exc_info:
            extractor.extract_features(bad_events, target_time)

        assert exc_info.value.context["feature_type"] == "temporal"
        assert exc_info.value.context["room_id"] == "test_room"

    def test_extract_features_sorts_events(self, extractor):
        """Test that events are sorted chronologically."""
        # Create unsorted events
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        events = []

        # Add events in reverse chronological order
        for i in [3, 1, 4, 0, 2]:
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.room_id = "test_room"
            event.sensor_type = "motion"
            event.state = "on"
            event.attributes = {}
            events.append(event)

        target_time = base_time + timedelta(hours=1)

        # Should not raise error due to sorting
        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)

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

    def test_timezone_handling(self):
        """Test feature extraction with different timezone offsets."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

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

    def test_performance_with_large_dataset(self, extractor, target_time):
        """Test performance with large number of events."""
        # Create large dataset of events for performance testing
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

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(events, target_time)
        extraction_time = time.time() - start_time

        # Should complete within reasonable time (< 2 seconds)
        assert extraction_time < 2.0
        assert isinstance(features, dict)
        assert len(features) > 30


class TestTemporalFeatureExtractorTimeSinceFeatures:
    """Test time-since feature extraction methods."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_extract_time_since_features_empty(self, extractor):
        """Test time-since features with empty events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        features = extractor._extract_time_since_features([], target_time)

        assert features["time_since_last_event"] == 3600.0
        assert features["time_since_last_on"] == 3600.0
        assert features["time_since_last_off"] == 3600.0
        assert features["time_since_last_motion"] == 3600.0

    def test_extract_time_since_features_basic(self, extractor):
        """Test basic time-since feature extraction."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(minutes=30)

        events = [
            self._create_event(base_time, "on", "motion"),
            self._create_event(base_time + timedelta(minutes=10), "off", "motion"),
            self._create_event(base_time + timedelta(minutes=20), "on", "door"),
        ]

        features = extractor._extract_time_since_features(events, target_time)

        assert features["time_since_last_event"] == 600.0  # 10 minutes
        assert features["time_since_last_on"] == 600.0  # 10 minutes to last "on"
        assert features["time_since_last_off"] == 1200.0  # 20 minutes to last "off"

    def test_extract_time_since_features_motion_sensor(self, extractor):
        """Test time-since features with motion sensors."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(minutes=30)

        events = [
            self._create_event(base_time, "on", "motion"),
            self._create_event(base_time + timedelta(minutes=10), "on", "presence"),
            self._create_event(base_time + timedelta(minutes=20), "off", "door"),
        ]

        features = extractor._extract_time_since_features(events, target_time)

        assert (
            features["time_since_last_motion"] == 1800.0
        )  # 30 minutes to motion sensor

    def test_extract_time_since_features_24_hour_cap(self, extractor):
        """Test that time-since features are capped at 24 hours."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=30)  # 30 hours later

        events = [self._create_event(base_time, "on", "motion")]

        features = extractor._extract_time_since_features(events, target_time)

        # All values should be capped at 86400 seconds (24 hours)
        assert features["time_since_last_event"] == 86400.0
        assert features["time_since_last_on"] == 86400.0

    def _create_event(self, timestamp, state, sensor_type):
        """Helper to create mock sensor events."""
        event = Mock(spec=SensorEvent)
        event.timestamp = timestamp
        event.state = state
        event.sensor_type = sensor_type
        event.room_id = "test_room"
        event.sensor_id = f"{sensor_type}_1"
        event.attributes = {}
        return event


class TestTemporalFeatureExtractorDurationFeatures:
    """Test duration feature extraction methods."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_extract_duration_features_empty(self, extractor):
        """Test duration features with empty events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        features = extractor._extract_duration_features([], target_time)

        assert features["current_state_duration"] == 0.0
        assert features["avg_on_duration"] == 1800.0
        assert features["avg_off_duration"] == 1800.0
        assert features["duration_ratio"] == 1.0

    def test_extract_duration_features_basic(self, extractor):
        """Test basic duration feature extraction."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(minutes=35)

        events = [
            self._create_event(base_time, "on"),  # State starts
            self._create_event(base_time + timedelta(minutes=10), "off"),  # 10 min on
            self._create_event(base_time + timedelta(minutes=20), "on"),  # 10 min off
            self._create_event(base_time + timedelta(minutes=30), "off"),  # 10 min on
        ]

        features = extractor._extract_duration_features(events, target_time)

        assert features["current_state_duration"] == 300.0  # 5 minutes since last event
        assert features["avg_on_duration"] == 600.0  # Average of 10-minute on periods
        assert features["avg_off_duration"] == 600.0  # Average of 10-minute off periods

    def test_extract_duration_features_24_hour_cap(self, extractor):
        """Test that current state duration is capped at 24 hours."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=30)  # 30 hours later

        events = [self._create_event(base_time, "on")]

        features = extractor._extract_duration_features(events, target_time)

        assert features["current_state_duration"] == 86400.0  # Capped at 24 hours

    def _create_event(self, timestamp, state):
        """Helper to create mock sensor events."""
        event = Mock(spec=SensorEvent)
        event.timestamp = timestamp
        event.state = state
        event.sensor_type = "motion"
        event.room_id = "test_room"
        event.attributes = {}
        return event


class TestTemporalFeatureExtractorCyclicalFeatures:
    """Test cyclical time feature extraction."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor(timezone_offset=0)

    def test_extract_cyclical_features_basic(self, extractor):
        """Test basic cyclical feature extraction."""
        # Monday, January 15, 2024, 12:00 PM
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        features = extractor._extract_cyclical_features(target_time)

        # Hour features (12 PM)
        expected_hour_sin = math.sin(2 * math.pi * 12 / 24)
        expected_hour_cos = math.cos(2 * math.pi * 12 / 24)
        assert abs(features["hour_sin"] - expected_hour_sin) < 1e-10
        assert abs(features["hour_cos"] - expected_hour_cos) < 1e-10

        # Day of week features (Monday = 0)
        expected_day_sin = math.sin(2 * math.pi * 0 / 7)
        expected_day_cos = math.cos(2 * math.pi * 0 / 7)
        assert abs(features["day_of_week_sin"] - expected_day_sin) < 1e-10
        assert abs(features["day_of_week_cos"] - expected_day_cos) < 1e-10

    def test_extract_cyclical_features_timezone_offset(self):
        """Test cyclical features with timezone offset."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)  # PST

        # UTC 8 PM = PST 12 PM
        target_time = datetime(2024, 1, 15, 20, 0, 0)  # 8 PM UTC

        features = extractor._extract_cyclical_features(target_time)

        # Should calculate based on local time (12 PM)
        expected_hour_sin = math.sin(2 * math.pi * 12 / 24)
        assert abs(features["hour_sin"] - expected_hour_sin) < 1e-10

    def test_extract_cyclical_features_weekend(self, extractor):
        """Test weekend indicator."""
        # Saturday
        saturday = datetime(2024, 1, 13, 12, 0, 0)
        features = extractor._extract_cyclical_features(saturday)
        assert features["is_weekend"] == 1.0

        # Monday
        monday = datetime(2024, 1, 15, 12, 0, 0)
        features = extractor._extract_cyclical_features(monday)
        assert features["is_weekend"] == 0.0

    def test_extract_cyclical_features_work_hours(self, extractor):
        """Test work hours indicator."""
        # 10 AM (work hours)
        work_time = datetime(2024, 1, 15, 10, 0, 0)
        features = extractor._extract_cyclical_features(work_time)
        assert features["is_work_hours"] == 1.0

        # 8 PM (not work hours)
        non_work_time = datetime(2024, 1, 15, 20, 0, 0)
        features = extractor._extract_cyclical_features(non_work_time)
        assert features["is_work_hours"] == 0.0

    def test_extract_cyclical_features_sleep_hours(self, extractor):
        """Test sleep hours indicator."""
        # 11 PM (sleep hours)
        sleep_time = datetime(2024, 1, 15, 23, 0, 0)
        features = extractor._extract_cyclical_features(sleep_time)
        assert features["is_sleep_hours"] == 1.0

        # 3 AM (sleep hours)
        early_sleep_time = datetime(2024, 1, 15, 3, 0, 0)
        features = extractor._extract_cyclical_features(early_sleep_time)
        assert features["is_sleep_hours"] == 1.0

        # 10 AM (not sleep hours)
        wake_time = datetime(2024, 1, 15, 10, 0, 0)
        features = extractor._extract_cyclical_features(wake_time)
        assert features["is_sleep_hours"] == 0.0

    def test_cyclical_encoding_accuracy(self, extractor):
        """Test mathematical accuracy of cyclical encodings."""
        # Test hour encoding for noon (12:00) with PST timezone offset (-8)
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        features = extractor.extract_features([], target_time)

        # With timezone_offset=0, local time is same as UTC: 12:00
        # At 12:00, hour angle should be 2π * 12 / 24 = π (180 degrees)
        local_hour = 12  # 12 + 0
        expected_hour_sin = math.sin(2 * math.pi * local_hour / 24)  # sin(π) = 0
        expected_hour_cos = math.cos(2 * math.pi * local_hour / 24)  # cos(π) = -1

        assert abs(features["hour_sin"] - expected_hour_sin) < 0.0001
        assert abs(features["hour_cos"] - expected_hour_cos) < 0.0001

    def test_cyclical_encoding_edge_cases(self, extractor):
        """Test cyclical encoding with edge case times."""
        edge_times = [
            datetime(2024, 1, 1, 0, 0, 0),  # Midnight Jan 1
            datetime(2024, 12, 31, 23, 59, 59),  # Almost midnight Dec 31
            datetime(2024, 2, 29, 6, 30, 0),  # Leap year date at dawn
            datetime(2024, 7, 4, 12, 0, 0),  # July 4th noon
        ]

        for target_time in edge_times:
            features = extractor._extract_cyclical_features(target_time)

            # Verify all cyclical features are in valid ranges
            assert -1.0 <= features["hour_sin"] <= 1.0
            assert -1.0 <= features["hour_cos"] <= 1.0
            assert -1.0 <= features["day_of_week_sin"] <= 1.0
            assert -1.0 <= features["day_of_week_cos"] <= 1.0
            assert -1.0 <= features["month_sin"] <= 1.0
            assert -1.0 <= features["month_cos"] <= 1.0
            assert features["is_weekend"] in [0.0, 1.0]
            assert features["is_work_hours"] in [0.0, 1.0]
            assert features["is_sleep_hours"] in [0.0, 1.0]

    def test_leap_year_date_handling(self, extractor):
        """Test handling of leap year dates."""
        leap_year_date = datetime(2024, 2, 29, 12, 0, 0)  # Feb 29, 2024 (leap year)

        features = extractor.extract_features([], leap_year_date)

        assert isinstance(features, dict)
        assert "day_of_month_sin" in features
        assert "day_of_month_cos" in features

    def test_year_end_date_handling(self, extractor):
        """Test handling of year-end dates."""
        year_end_date = datetime(2024, 12, 31, 23, 59, 59)

        features = extractor.extract_features([], year_end_date)

        assert isinstance(features, dict)
        assert "month_sin" in features
        assert "month_cos" in features


class TestTemporalFeatureExtractorHistoricalPatterns:
    """Test historical pattern analysis."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor(timezone_offset=0)

    def test_extract_historical_patterns_empty(self, extractor):
        """Test historical patterns with empty events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        features = extractor._extract_historical_patterns([], target_time)

        assert features["hour_activity_rate"] == 0.5
        assert features["day_activity_rate"] == 0.5
        assert features["overall_activity_rate"] == 0.5
        assert features["pattern_strength"] == 0.0

    def test_extract_historical_patterns_basic(self, extractor):
        """Test basic historical pattern extraction."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)  # Monday 12 PM

        # Create events with patterns
        events = []
        for day in range(7):  # One week of data
            for hour in [9, 12, 15, 18]:  # Activity at specific hours
                event_time = base_time + timedelta(days=day, hours=hour - 12)
                event = Mock(spec=SensorEvent)
                event.timestamp = event_time
                event.state = "on"
                events.append(event)

        target_time = base_time + timedelta(days=7)  # Target at same time next week

        features = extractor._extract_historical_patterns(events, target_time)

        assert features["overall_activity_rate"] == 1.0  # All events are "on"
        assert features["hour_activity_rate"] == 1.0  # 12 PM always has activity
        assert 0.0 <= features["day_activity_rate"] <= 1.0  # Monday activity rate

    def test_extract_historical_patterns_with_timezone(self):
        """Test historical patterns with timezone offset."""
        extractor = TemporalFeatureExtractor(timezone_offset=-5)  # EST

        base_time = datetime(2024, 1, 15, 17, 0, 0)  # 5 PM UTC = 12 PM EST

        events = [
            self._create_historical_event(base_time, "on"),
            self._create_historical_event(base_time + timedelta(hours=1), "on"),
        ]

        target_time = base_time + timedelta(days=1)

        features = extractor._extract_historical_patterns(events, target_time)

        # Should be calculated based on local time (12 PM EST)
        assert isinstance(features["hour_activity_rate"], float)

    def _create_historical_event(self, timestamp, state):
        """Helper to create mock events for historical analysis."""
        event = Mock(spec=SensorEvent)
        event.timestamp = timestamp
        event.state = state
        return event


class TestTemporalFeatureExtractorTransitionTiming:
    """Test transition timing feature extraction."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_extract_transition_timing_features_insufficient_events(self, extractor):
        """Test transition timing with insufficient events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Only one event
        events = [self._create_event(target_time - timedelta(minutes=30), "on")]

        features = extractor._extract_transition_timing_features(events, target_time)

        assert features["avg_transition_interval"] == 1800.0
        assert features["recent_transition_rate"] == 0.0
        assert features["time_variability"] == 0.0

    def test_extract_transition_timing_features_basic(self, extractor):
        """Test basic transition timing features."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Create events with regular 30-minute intervals
        events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 30)
            event.state = "on" if i % 2 == 0 else "off"
            events.append(event)

        features = extractor._extract_transition_timing_features(events, target_time)

        assert features["avg_transition_interval"] == 1800.0  # 30 minutes
        assert features["time_variability"] == 0.0  # No variation in intervals

    def _create_event(self, timestamp, state):
        """Helper to create mock sensor events."""
        event = Mock(spec=SensorEvent)
        event.timestamp = timestamp
        event.state = state
        return event


class TestTemporalFeatureExtractorRoomStateFeatures:
    """Test room state feature extraction."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_extract_room_state_features_empty(self, extractor):
        """Test room state features with empty states."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        features = extractor._extract_room_state_features([], target_time)

        assert features["avg_occupancy_confidence"] == 0.5
        assert features["recent_occupancy_ratio"] == 0.5
        assert features["state_stability"] == 0.5

    def test_extract_room_state_features_basic(self, extractor):
        """Test basic room state features."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=2)

        states = []
        for i in range(5):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 30)
            state.is_occupied = i % 2 == 0  # Alternating occupied/vacant
            state.occupancy_confidence = 0.8 + i * 0.02  # Increasing confidence
            states.append(state)

        features = extractor._extract_room_state_features(states, target_time)

        assert features["recent_occupancy_ratio"] == 0.6  # 3 out of 5 occupied
        assert abs(features["avg_occupancy_confidence"] - 0.84) < 0.01

    def test_room_states_edge_cases(self, extractor):
        """Test room states with edge cases."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []  # Empty events

        # Create room states with edge cases
        room_states = []

        # State with None confidence
        state1 = Mock(spec=RoomState)
        state1.timestamp = target_time - timedelta(hours=1)
        state1.room_id = "test_room"
        state1.is_occupied = True
        state1.occupancy_confidence = None
        room_states.append(state1)

        # State with extreme confidence values
        state2 = Mock(spec=RoomState)
        state2.timestamp = target_time - timedelta(hours=2)
        state2.room_id = "test_room"
        state2.is_occupied = False
        state2.occupancy_confidence = 1.5  # Invalid value > 1
        room_states.append(state2)

        features = extractor.extract_features(events, target_time, room_states)
        assert isinstance(features, dict)

    def test_empty_room_states(self, extractor):
        """Test with empty room states list."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []
        room_states = []

        features = extractor.extract_features(events, target_time, room_states)
        assert isinstance(features, dict)


class TestTemporalFeatureExtractorGenericSensorFeatures:
    """Test generic sensor feature extraction."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_extract_generic_sensor_features_empty(self, extractor):
        """Test generic sensor features with empty events."""
        features = extractor._extract_generic_sensor_features([])

        assert features == {}

    def test_extract_generic_sensor_features_numeric_attributes(self, extractor):
        """Test extraction with numeric sensor attributes."""
        events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.attributes = {
                "temperature": 20.0 + i,
                "humidity": 45.0 + i * 2,
                "pressure": 1013.25 + i * 0.5,
            }
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        assert "numeric_mean" in features
        assert "numeric_std" in features
        assert "numeric_min" in features
        assert "numeric_max" in features
        assert "numeric_range" in features
        assert "numeric_count" in features

        # Check some calculations
        assert features["numeric_count"] > 0
        assert features["numeric_range"] > 0

    def test_extract_generic_sensor_features_sensor_type_ratios(self, extractor):
        """Test sensor type ratio calculations."""
        events = []
        sensor_types = ["motion", "motion", "door", "presence", "motion"]

        for sensor_type in sensor_types:
            event = Mock(spec=SensorEvent)
            event.attributes = {}
            event.state = "on"
            event.sensor_type = sensor_type
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        assert "motion_sensor_ratio" in features
        assert "door_sensor_ratio" in features
        assert "presence_sensor_ratio" in features

        # 3 motion out of 5 total = 0.6
        assert features["motion_sensor_ratio"] == 0.6
        # 1 door out of 5 total = 0.2
        assert features["door_sensor_ratio"] == 0.2
        # 1 presence out of 5 total = 0.2
        assert features["presence_sensor_ratio"] == 0.2


class TestTemporalFeatureExtractorUtilityMethods:
    """Test utility and helper methods."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_get_default_features_no_time(self, extractor):
        """Test default features without target time."""
        features = extractor._get_default_features()

        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many default features
        assert "time_since_last_event" in features
        assert "hour_sin" in features
        assert features["hour_sin"] == 0.0  # Default cyclical value

    def test_get_default_features_with_time(self, extractor):
        """Test default features with target time."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        features = extractor._get_default_features(target_time)

        assert isinstance(features, dict)
        # Should have actual cyclical features calculated
        expected_hour_sin = math.sin(2 * math.pi * 12 / 24)
        assert abs(features["hour_sin"] - expected_hour_sin) < 1e-10

    def test_get_feature_names(self, extractor):
        """Test getting feature names list."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 20
        assert "time_since_last_event" in feature_names
        assert "hour_sin" in feature_names

    def test_validate_feature_names_with_standard_names(self, extractor):
        """Test feature name validation with standard names."""
        extracted_features = {
            "time_since_last_change": 1800.0,
            "hour_sin": 0.5,
            "custom_feature": 42.0,
        }

        with patch(
            "src.features.temporal.TEMPORAL_FEATURE_NAMES",
            ["time_since_last_change", "hour_sin", "day_of_week_cos"],
        ):
            validated = extractor.validate_feature_names(extracted_features)

            assert "time_since_last_change" in validated
            assert "hour_sin" in validated
            assert "custom_feature" in validated  # Should be preserved

    def test_clear_cache(self, extractor):
        """Test cache clearing."""
        # Add something to cache
        extractor.feature_cache["test_key"] = "test_value"

        assert len(extractor.feature_cache) == 1

        extractor.clear_cache()

        assert len(extractor.feature_cache) == 0

    def test_extract_batch_features_basic(self, extractor):
        """Test batch feature extraction."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        # Create batch of event sequences
        event_batches = []
        for i in range(3):
            events = []
            for j in range(2):
                event = Mock(spec=SensorEvent)
                event.timestamp = base_time + timedelta(minutes=i * 30 + j * 10)
                event.state = "on"
                event.sensor_type = "motion"
                event.attributes = {}
                events.append(event)

            target_time = base_time + timedelta(minutes=i * 30 + 60)
            event_batches.append((events, target_time))

        results = extractor.extract_batch_features(event_batches)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert len(result) > 10

    def test_extract_batch_features_empty_batch(self, extractor):
        """Test batch feature extraction with empty batch."""
        results = extractor.extract_batch_features([])

        assert results == []


class TestTemporalFeatureExtractorEdgeCases:
    """Test edge cases and error handling."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor(timezone_offset=0)

    def test_extract_features_malformed_events(self, extractor):
        """Test feature extraction with malformed events."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        # Create event with missing required attributes
        malformed_event = Mock()
        malformed_event.timestamp = target_time - timedelta(hours=1)
        # Missing state and sensor_type attributes

        # Should handle gracefully or raise appropriate error
        try:
            features = extractor.extract_features([malformed_event], target_time)
            # If it succeeds, should return valid features
            assert isinstance(features, dict)
        except (AttributeError, FeatureExtractionError):
            # Both are acceptable responses to malformed data
            pass

    def test_extract_features_with_future_events(self, extractor):
        """Test feature extraction with events in the future."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        future_event = Mock(spec=SensorEvent)
        future_event.timestamp = target_time + timedelta(hours=1)  # Future event
        future_event.state = "on"
        future_event.sensor_type = "motion"
        future_event.room_id = "test_room"

        features = extractor.extract_features([future_event], target_time)

        # Should handle future events gracefully
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_features_with_extreme_time_differences(self, extractor):
        """Test feature extraction with very old and very new events."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []

        # Very old event (1 year ago)
        old_event = Mock(spec=SensorEvent)
        old_event.timestamp = target_time - timedelta(days=365)
        old_event.state = "off"
        old_event.sensor_type = "door"
        old_event.room_id = "test_room"
        events.append(old_event)

        # Very recent event (1 second ago)
        recent_event = Mock(spec=SensorEvent)
        recent_event.timestamp = target_time - timedelta(seconds=1)
        recent_event.state = "on"
        recent_event.sensor_type = "motion"
        recent_event.room_id = "test_room"
        events.append(recent_event)

        features = extractor.extract_features(events, target_time)

        # Should handle extreme time differences
        assert isinstance(features, dict)
        assert "time_since_last_event" in features
        assert features["time_since_last_event"] >= 1.0  # At least 1 second

    def test_lookback_hours_filtering(self, extractor):
        """Test lookback hours filtering edge cases."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        events = []
        # Create events at different time intervals
        for hours_back in [1, 6, 12, 24, 48, 72]:
            event = Mock(spec=SensorEvent)
            event.timestamp = target_time - timedelta(hours=hours_back)
            event.state = "on" if hours_back % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.room_id = "test_room"
            events.append(event)

        # Test different lookback windows
        for lookback in [2, 8, 25, 50]:
            features = extractor.extract_features(
                events, target_time, lookback_hours=lookback
            )
            assert isinstance(features, dict)
            assert len(features) > 0

    def test_zero_lookback_hours(self, extractor):
        """Test with zero lookback hours."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        event = Mock(spec=SensorEvent)
        event.timestamp = target_time - timedelta(minutes=30)
        event.state = "on"
        event.sensor_type = "motion"
        event.room_id = "test_room"

        # Zero lookback should filter out all events
        features = extractor.extract_features([event], target_time, lookback_hours=0)

        # Should return default features since no events pass the filter
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_negative_lookback_hours(self, extractor):
        """Test with negative lookback hours."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        event = Mock(spec=SensorEvent)
        event.timestamp = target_time - timedelta(minutes=30)
        event.state = "on"
        event.sensor_type = "motion"
        event.room_id = "test_room"

        # Negative lookback - undefined behavior, should handle gracefully
        features = extractor.extract_features([event], target_time, lookback_hours=-5)
        assert isinstance(features, dict)

    def test_events_with_invalid_states(self, extractor):
        """Test events with invalid or unusual states."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []

        # Events with unusual states
        unusual_states = ["unknown", "error", "unavailable", "", None, 123, True, []]

        for i, state in enumerate(unusual_states):
            event = Mock(spec=SensorEvent)
            event.timestamp = target_time - timedelta(minutes=i * 10)
            event.state = state
            event.sensor_type = "motion"
            event.room_id = "test_room"
            event.attributes = {}
            events.append(event)

        # Should handle unusual states gracefully
        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)

    def test_events_with_complex_attributes(self, extractor):
        """Test events with complex attribute structures."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        event = Mock(spec=SensorEvent)
        event.timestamp = target_time - timedelta(minutes=30)
        event.state = "on"
        event.sensor_type = "motion"
        event.room_id = "test_room"

        # Complex nested attributes
        event.attributes = {
            "nested": {"deep": {"value": 42}},
            "list": [1, 2, 3],
            "bool": True,
            "float": 3.14,
            "null": None,
            "empty_dict": {},
            "empty_list": [],
        }

        features = extractor.extract_features([event], target_time)
        assert isinstance(features, dict)

    def test_large_number_of_events(self, extractor):
        """Test with very large number of events."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []

        # Create 10,000 events
        for i in range(10000):
            event = Mock(spec=SensorEvent)
            event.timestamp = target_time - timedelta(seconds=i)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.room_id = "test_room"
            event.attributes = {"index": i}
            events.append(event)

        # Should handle large datasets
        start_time = time.time()
        features = extractor.extract_features(events, target_time)
        extraction_time = time.time() - start_time

        assert isinstance(features, dict)
        assert extraction_time < 10.0  # Should complete within reasonable time

    def test_events_with_duplicate_timestamps(self, extractor):
        """Test events with identical timestamps."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []
        same_timestamp = target_time - timedelta(minutes=30)

        # Create multiple events with same timestamp
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = same_timestamp
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.room_id = "test_room"
            events.append(event)

        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)

    def test_unsorted_events(self, extractor):
        """Test with unsorted event timestamps."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        events = []

        # Create events in random order
        timestamps = [
            target_time - timedelta(minutes=10),
            target_time - timedelta(minutes=60),
            target_time - timedelta(minutes=5),
            target_time - timedelta(minutes=30),
            target_time - timedelta(minutes=45),
        ]

        for i, ts in enumerate(timestamps):
            event = Mock(spec=SensorEvent)
            event.timestamp = ts
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.room_id = "test_room"
            events.append(event)

        # Should sort events internally
        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)

    def test_memory_usage_with_large_datasets(self, extractor):
        """Test memory usage with large datasets."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create large dataset
        events = []
        for i in range(5000):
            event = Mock(spec=SensorEvent)
            event.timestamp = target_time - timedelta(seconds=i)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            event.room_id = "test_room"
            event.attributes = {"index": i, "data": f"event_{i}"}
            events.append(event)

        # Extract features
        features = extractor.extract_features(events, target_time)

        # Get final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        assert isinstance(features, dict)
        # Memory usage should not increase excessively (< 100MB for 5000 events)
        assert memory_increase < 100.0

    def test_concurrent_feature_extraction(self, extractor):
        """Test thread safety of feature extraction."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        results_queue = queue.Queue()

        def extract_features_worker(worker_id):
            try:
                # Create unique events for each worker
                events = []
                for i in range(100):
                    event = Mock(spec=SensorEvent)
                    event.timestamp = target_time - timedelta(
                        seconds=i + worker_id * 100
                    )
                    event.state = "on" if (i + worker_id) % 2 == 0 else "off"
                    event.sensor_type = "motion"
                    event.room_id = f"room_{worker_id}"
                    events.append(event)

                features = extractor.extract_features(events, target_time)
                results_queue.put((worker_id, features, None))
            except Exception as e:
                results_queue.put((worker_id, None, e))

        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=extract_features_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Check results
        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        assert len(results) == 5
        for worker_id, features, error in results:
            assert error is None, f"Worker {worker_id} failed with error: {error}"
            assert isinstance(features, dict)
            assert len(features) > 0
