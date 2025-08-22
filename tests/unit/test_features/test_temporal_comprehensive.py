"""
Comprehensive unit tests for TemporalFeatureExtractor to achieve high test coverage.

This module focuses on comprehensive testing of all methods, error paths,
edge cases, and configuration variations in TemporalFeatureExtractor.
"""

from datetime import datetime, timedelta
import math
from unittest.mock import Mock, patch
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import statistics

from src.core.constants import TEMPORAL_FEATURE_NAMES
from src.core.exceptions import FeatureExtractionError
from src.features.temporal import TemporalFeatureExtractor
from src.data.storage.models import RoomState, SensorEvent


class TestTemporalFeatureExtractorInitialization:
    """Test TemporalFeatureExtractor initialization."""

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


class TestTemporalFeatureExtractorBasicFeatures:
    """Test basic feature extraction functionality."""

    @pytest.fixture
    def extractor(self):
        """Create a TemporalFeatureExtractor instance."""
        return TemporalFeatureExtractor(timezone_offset=0)

    @pytest.fixture
    def sample_events(self):
        """Create sample sensor events."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)  # Monday 10 AM
        events = []
        
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.room_id = "living_room"
            event.sensor_id = "motion_1"
            event.sensor_type = "motion"
            event.state = "on" if i % 2 == 0 else "of"  # Corrected typo here for consistency
            event.attributes = {"temperature": 20.5 + i, "humidity": 45.0}
            events.append(event)
        
        return events

    @pytest.fixture
    def sample_room_states(self):
        """Create sample room states."""
        base_time = datetime(2024, 1, 15, 9, 0, 0)
        states = []
        
        for i in range(5):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(hours=i)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8 + i * 0.05
            states.append(state)
        
        return states

    def test_extract_features_empty_events(self, extractor):
        """Test feature extraction with empty events list."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        features = extractor.extract_features([], target_time)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        # Should return default features
        assert "time_since_last_event" in features
        assert "hour_sin" in features
        assert "hour_cos" in features

    def test_extract_features_basic_success(self, extractor, sample_events):
        """Test successful basic feature extraction."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        features = extractor.extract_features(sample_events, target_time)
        
        assert isinstance(features, dict)
        assert len(features) > 10  # Should have many features
        
        # Check key feature categories
        assert "time_since_last_event" in features
        assert "current_state_duration" in features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "day_of_week_sin" in features
        assert "is_weekend" in features
        assert "overall_activity_rate" in features

    def test_extract_features_with_room_states(self, extractor, sample_events, sample_room_states):
        """Test feature extraction with room states."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        features = extractor.extract_features(sample_events, target_time, sample_room_states)
        
        assert isinstance(features, dict)
        assert "avg_occupancy_confidence" in features
        assert "recent_occupancy_ratio" in features
        assert "state_stability" in features

    def test_extract_features_with_lookback_filter(self, extractor, sample_events):
        """Test feature extraction with lookback hours filter."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        # Only look back 1 hour (should filter out most events)
        features = extractor.extract_features(sample_events, target_time, lookback_hours=1)
        
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
            self._create_event(base_time + timedelta(minutes=10), "of", "motion"),
            self._create_event(base_time + timedelta(minutes=20), "on", "door"),
        ]
        
        features = extractor._extract_time_since_features(events, target_time)
        
        assert features["time_since_last_event"] == 600.0  # 10 minutes
        assert features["time_since_last_on"] == 600.0  # 10 minutes to last "on"
        assert features["time_since_last_off"] == 1200.0  # 20 minutes to last "of"

    def test_extract_time_since_features_motion_sensor(self, extractor):
        """Test time-since features with motion sensors."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(minutes=30)
        
        events = [
            self._create_event(base_time, "on", "motion"),
            self._create_event(base_time + timedelta(minutes=10), "on", "presence"),
            self._create_event(base_time + timedelta(minutes=20), "of", "door"),
        ]
        
        features = extractor._extract_time_since_features(events, target_time)
        
        assert features["time_since_last_motion"] == 1200.0  # 20 minutes to motion sensor

    def test_extract_time_since_features_24_hour_cap(self, extractor):
        """Test that time-since features are capped at 24 hours."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=30)  # 30 hours later
        
        events = [self._create_event(base_time, "on", "motion")]
        
        features = extractor._extract_time_since_features(events, target_time)
        
        # All values should be capped at 86400 seconds (24 hours)
        assert features["time_since_last_event"] == 86400.0
        assert features["time_since_last_on"] == 86400.0

    def test_extract_time_since_features_state_of_typo(self, extractor):
        """Test time-since features with 'of' state (typo handling)."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(minutes=15)
        
        events = [
            self._create_event(base_time, "on", "motion"),
            self._create_event(base_time + timedelta(minutes=5), "of", "motion"),  # Note: "of" not "off"
        ]
        
        features = extractor._extract_time_since_features(events, target_time)
        
        # Should handle "of" as "off"
        assert features["time_since_last_off"] == 600.0  # 10 minutes

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
            self._create_event(base_time + timedelta(minutes=10), "of"),  # 10 min on
            self._create_event(base_time + timedelta(minutes=20), "on"),  # 10 min off
            self._create_event(base_time + timedelta(minutes=30), "of"),  # 10 min on
        ]
        
        features = extractor._extract_duration_features(events, target_time)
        
        assert features["current_state_duration"] == 300.0  # 5 minutes since last event
        assert features["avg_on_duration"] == 600.0  # Average of 10-minute on periods
        assert features["avg_off_duration"] == 600.0  # Average of 10-minute off periods

    def test_extract_duration_features_advanced_statistics(self, extractor):
        """Test advanced statistical duration features."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=2)
        
        # Create events with varying durations
        events = [
            self._create_event(base_time, "on"),
            self._create_event(base_time + timedelta(minutes=5), "of"),  # 5 min on
            self._create_event(base_time + timedelta(minutes=15), "on"),  # 10 min off
            self._create_event(base_time + timedelta(minutes=35), "of"),  # 20 min on
            self._create_event(base_time + timedelta(minutes=40), "on"),  # 5 min off
            self._create_event(base_time + timedelta(minutes=50), "of"),  # 10 min on
        ]
        
        features = extractor._extract_duration_features(events, target_time)
        
        # Should have advanced statistical features
        assert "on_duration_std" in features
        assert "off_duration_std" in features
        assert "median_on_duration" in features
        assert "median_off_duration" in features
        assert "duration_percentile_75" in features
        assert "duration_percentile_25" in features
        
        # Check some calculations
        assert features["max_on_duration"] == 1200.0  # 20 minutes
        assert features["max_off_duration"] == 600.0  # 10 minutes

    def test_extract_duration_features_ratio_calculation(self, extractor):
        """Test duration ratio calculation."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=1)
        
        events = [
            self._create_event(base_time, "on"),
            self._create_event(base_time + timedelta(minutes=20), "of"),  # 20 min on
            self._create_event(base_time + timedelta(minutes=30), "on"),  # 10 min off
            self._create_event(base_time + timedelta(minutes=50), "of"),  # 20 min on
        ]
        
        features = extractor._extract_duration_features(events, target_time)
        
        # Duration ratio should be 20 minutes on / 10 minutes off = 2.0
        assert features["duration_ratio"] == 2.0

    def test_extract_duration_features_zero_off_duration(self, extractor):
        """Test duration features when off duration is zero."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(minutes=30)
        
        # Only "on" states, no "of"/"off" states
        events = [
            self._create_event(base_time, "on"),
            self._create_event(base_time + timedelta(minutes=10), "on"),
            self._create_event(base_time + timedelta(minutes=20), "on"),
        ]
        
        features = extractor._extract_duration_features(events, target_time)
        
        # Should handle gracefully with defaults
        assert features["duration_ratio"] == 1.0  # Should default when division by zero

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

    def test_extract_cyclical_features_month_and_day(self, extractor):
        """Test month and day of month features."""
        # March 15th
        target_time = datetime(2024, 3, 15, 12, 0, 0)
        
        features = extractor._extract_cyclical_features(target_time)
        
        # Month features (March = 3)
        expected_month_sin = math.sin(2 * math.pi * 3 / 12)
        expected_month_cos = math.cos(2 * math.pi * 3 / 12)
        assert abs(features["month_sin"] - expected_month_sin) < 1e-10
        assert abs(features["month_cos"] - expected_month_cos) < 1e-10
        
        # Day of month features (15th)
        expected_day_sin = math.sin(2 * math.pi * 15 / 31)
        expected_day_cos = math.cos(2 * math.pi * 15 / 31)
        assert abs(features["day_of_month_sin"] - expected_day_sin) < 1e-10
        assert abs(features["day_of_month_cos"] - expected_day_cos) < 1e-10


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
                event_time = base_time + timedelta(days=day, hours=hour-12)
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

    def test_extract_historical_patterns_advanced_statistics(self, extractor):
        """Test advanced statistical features in historical patterns."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        
        # Create events with varying activity levels
        events = []
        for day in range(30):  # 30 days of data
            event_time = base_time + timedelta(days=day)
            event = Mock(spec=SensorEvent)
            event.timestamp = event_time
            event.state = "on" if day % 3 == 0 else "of"  # Pattern: on every 3rd day
            events.append(event)
        
        target_time = base_time + timedelta(days=30)
        
        features = extractor._extract_historical_patterns(events, target_time)
        
        assert "activity_variance" in features
        assert "trend_coefficient" in features
        assert "seasonality_score" in features
        assert 0.0 <= features["activity_variance"] <= 1.0

    def test_extract_historical_patterns_trend_calculation(self, extractor):
        """Test trend coefficient calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        
        # Create events with increasing activity (positive trend)
        events = []
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(hours=i)
            event.state = "on" if i > 5 else "of"  # More "on" events later
            events.append(event)
        
        target_time = base_time + timedelta(hours=10)
        
        features = extractor._extract_historical_patterns(events, target_time)
        
        # Should detect positive trend
        assert -1.0 <= features["trend_coefficient"] <= 1.0

    def test_extract_historical_patterns_insufficient_data_for_trend(self, extractor):
        """Test trend calculation with insufficient data."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        
        # Only 2 events (insufficient for trend calculation)
        events = [
            self._create_historical_event(base_time, "on"),
            self._create_historical_event(base_time + timedelta(hours=1), "on"),
        ]
        
        target_time = base_time + timedelta(hours=2)
        
        features = extractor._extract_historical_patterns(events, target_time)
        
        assert features["trend_coefficient"] == 0.0  # Should default to 0

    def test_extract_historical_patterns_seasonality(self, extractor):
        """Test seasonality score calculation."""
        base_time = datetime(2024, 1, 1, 12, 0, 0)
        
        # Create events spanning multiple days with seasonal pattern
        events = []
        for day in range(30):  # 30 days
            event_time = base_time + timedelta(days=day)
            event = Mock(spec=SensorEvent)
            event.timestamp = event_time
            # Seasonal pattern: more active on certain days
            event.state = "on" if day % 7 < 2 else "of"  # Active on Mon-Tue
            events.append(event)
        
        target_time = base_time + timedelta(days=30)
        
        features = extractor._extract_historical_patterns(events, target_time)
        
        assert features["seasonality_score"] >= 0.0

    def test_extract_historical_patterns_similar_time_weighting(self, extractor):
        """Test similar time activity rate with hour weighting."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)  # 12 PM
        
        # Create events at different hours with varying activity
        events = []
        for hour_offset in [-2, -1, 0, 1, 2]:  # 10 AM to 2 PM
            event_time = base_time + timedelta(hours=hour_offset)
            event = Mock(spec=SensorEvent)
            event.timestamp = event_time
            # 12 PM (hour_offset=0) always active, others less so
            event.state = "on" if hour_offset == 0 else "of"
            events.append(event)
        
        target_time = base_time + timedelta(days=1)  # Same time next day
        
        features = extractor._extract_historical_patterns(events, target_time)
        
        # Should weight 12 PM (current hour) more heavily
        assert 0.0 <= features["similar_time_activity_rate"] <= 1.0

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
            event.state = "on" if i % 2 == 0 else "of"
            events.append(event)
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        assert features["avg_transition_interval"] == 1800.0  # 30 minutes
        assert features["time_variability"] == 0.0  # No variation in intervals

    def test_extract_transition_timing_features_variability(self, extractor):
        """Test time variability calculation."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=3)
        
        # Create events with varying intervals
        intervals = [10, 20, 30, 40]  # Minutes
        events = []
        current_time = base_time
        
        for interval in intervals:
            event = Mock(spec=SensorEvent)
            event.timestamp = current_time
            event.state = "on"
            events.append(event)
            current_time += timedelta(minutes=interval)
        
        # Add final event
        final_event = Mock(spec=SensorEvent)
        final_event.timestamp = current_time
        final_event.state = "of"
        events.append(final_event)
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        # Should calculate coefficient of variation
        expected_mean = statistics.mean([i * 60 for i in intervals])  # Convert to seconds
        expected_std = statistics.stdev([i * 60 for i in intervals])
        expected_variability = expected_std / expected_mean
        
        assert abs(features["time_variability"] - expected_variability) < 0.01

    def test_extract_transition_timing_features_recent_rate(self, extractor):
        """Test recent transition rate calculation."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=6)
        
        # Create events: some old, some recent
        events = []
        
        # Old events (beyond 4-hour window)
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.state = "on"
            events.append(event)
        
        # Recent events (within 4-hour window)
        recent_base = target_time - timedelta(hours=2)
        for i in range(4):
            event = Mock(spec=SensorEvent)
            event.timestamp = recent_base + timedelta(minutes=i * 30)
            event.state = "of" if i % 2 else "on"
            events.append(event)
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        # Should only count recent events for rate calculation
        assert features["recent_transition_rate"] > 0.0

    def test_extract_transition_timing_features_regularity(self, extractor):
        """Test transition regularity calculation."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=2)
        
        # Create very regular events (same interval)
        events = []
        for i in range(6):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 20)  # Exactly 20 min apart
            event.state = "on" if i % 2 == 0 else "of"
            events.append(event)
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        # High regularity (low variability)
        assert features["transition_regularity"] > 0.9
        assert features["time_variability"] < 0.1

    def test_extract_transition_timing_features_trend(self, extractor):
        """Test recent transition trend calculation."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=3)
        
        # Create events with increasing intervals (slowing trend)
        events = []
        intervals = [5, 10, 15, 20, 25, 30]  # Increasing intervals
        current_time = base_time
        
        for interval in intervals:
            event = Mock(spec=SensorEvent)
            event.timestamp = current_time
            event.state = "on"
            events.append(event)
            current_time += timedelta(minutes=interval)
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        # Should detect positive trend (increasing intervals)
        assert features["recent_transition_trend"] > 0.0

    def test_extract_transition_timing_features_insufficient_for_trend(self, extractor):
        """Test trend calculation with insufficient data."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=1)
        
        # Only 2 events (insufficient for trend)
        events = [
            self._create_event(base_time, "on"),
            self._create_event(base_time + timedelta(minutes=30), "of"),
        ]
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        assert features["recent_transition_trend"] == 0.0

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

    def test_extract_room_state_features_confidence_filtering(self, extractor):
        """Test confidence filtering (None values)."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=1)
        
        states = []
        for i in range(3):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 20)
            state.is_occupied = True
            # Some states have None confidence
            state.occupancy_confidence = 0.9 if i % 2 == 0 else None
            states.append(state)
        
        features = extractor._extract_room_state_features(states, target_time)
        
        # Should only average non-None confidences
        assert features["avg_occupancy_confidence"] == 0.9

    def test_extract_room_state_features_recent_filtering(self, extractor):
        """Test recent occupancy ratio with 24-hour filtering."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=30)  # 30 hours later
        
        states = []
        
        # Old states (beyond 24 hours)
        for i in range(3):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(hours=i)
            state.is_occupied = True  # All old states occupied
            state.occupancy_confidence = 0.8
            states.append(state)
        
        # Recent states (within 24 hours)
        recent_base = target_time - timedelta(hours=12)
        for i in range(4):
            state = Mock(spec=RoomState)
            state.timestamp = recent_base + timedelta(hours=i * 3)
            state.is_occupied = i < 2  # Only first 2 occupied
            state.occupancy_confidence = 0.9
            states.append(state)
        
        features = extractor._extract_room_state_features(states, target_time)
        
        # Should only consider recent states: 2 occupied out of 4 = 0.5
        assert features["recent_occupancy_ratio"] == 0.5

    def test_extract_room_state_features_stability(self, extractor):
        """Test state stability calculation."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=4)
        
        # Create states with known durations
        state_times = [base_time, base_time + timedelta(hours=1), base_time + timedelta(hours=3)]
        states = []
        
        for i, timestamp in enumerate(state_times):
            state = Mock(spec=RoomState)
            state.timestamp = timestamp
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8
            states.append(state)
        
        features = extractor._extract_room_state_features(states, target_time)
        
        # Durations: 1 hour, 2 hours -> average 1.5 hours
        assert features["state_stability"] == 1.5

    def test_extract_room_state_features_single_state(self, extractor):
        """Test with single room state."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        state = Mock(spec=RoomState)
        state.timestamp = target_time - timedelta(hours=1)
        state.is_occupied = True
        state.occupancy_confidence = 0.95
        
        features = extractor._extract_room_state_features([state], target_time)
        
        assert features["avg_occupancy_confidence"] == 0.95
        assert features["recent_occupancy_ratio"] == 1.0
        assert features["state_stability"] == 0.5  # Default when no duration calc possible

    def test_extract_room_state_features_no_confidences(self, extractor):
        """Test with states that have no confidence values."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        states = []
        for i in range(3):
            state = Mock(spec=RoomState)
            state.timestamp = target_time - timedelta(hours=i)
            state.is_occupied = True
            state.occupancy_confidence = None  # All None
            states.append(state)
        
        features = extractor._extract_room_state_features(states, target_time)
        
        # Should default when no confidences available
        assert features["avg_occupancy_confidence"] == 0.5


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
                "pressure": 1013.25 + i * 0.5
            }
            event.state = "on" if i % 2 == 0 else "of"
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

    def test_extract_generic_sensor_features_boolean_attributes(self, extractor):
        """Test extraction with boolean sensor attributes."""
        events = []
        for i in range(4):
            event = Mock(spec=SensorEvent)
            event.attributes = {
                "motion_detected": i < 2,  # First 2 are True
                "door_open": i % 2 == 0,   # Alternating
            }
            event.state = "on"
            event.sensor_type = "motion"
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        assert "boolean_true_ratio" in features
        assert "boolean_false_ratio" in features
        assert "boolean_count" in features
        
        # Should have analyzed boolean values
        assert features["boolean_count"] > 0
        assert 0.0 <= features["boolean_true_ratio"] <= 1.0
        assert 0.0 <= features["boolean_false_ratio"] <= 1.0

    def test_extract_generic_sensor_features_string_attributes(self, extractor):
        """Test extraction with string sensor attributes."""
        events = []
        string_values = ["open", "closed", "open", "ajar", "closed"]
        
        for i, string_val in enumerate(string_values):
            event = Mock(spec=SensorEvent)
            event.attributes = {"door_status": string_val}
            event.state = "on"
            event.sensor_type = "door"
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        assert "string_unique_count" in features
        assert "string_total_count" in features
        assert "string_diversity_ratio" in features
        assert "most_common_string_frequency" in features
        
        # "on" state appears 5 times out of 10 total strings (5 attributes + 5 states)
        # Strings: ['open', 'on', 'closed', 'on', 'open', 'on', 'ajar', 'on', 'closed', 'on']
        assert features["most_common_string_frequency"] == 0.5  # "on" appears 5/10 times
        assert features["string_unique_count"] == 4  # open, closed, ajar, on

    def test_extract_generic_sensor_features_mixed_types(self, extractor):
        """Test extraction with mixed attribute types."""
        events = []
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.attributes = {
                "temperature": 20.0 + i,     # numeric
                "motion": i < 1,             # boolean
                "status": "active",          # string
                "level": str(i + 1),         # numeric string
            }
            event.state = i
            event.sensor_type = "presence"
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        # Should have features for all types
        assert "numeric_count" in features
        assert "boolean_count" in features
        assert "string_total_count" in features
        assert "numeric_value_ratio" in features
        assert "boolean_value_ratio" in features
        assert "string_value_ratio" in features

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

    def test_extract_generic_sensor_features_no_attributes(self, extractor):
        """Test extraction when events have no attributes."""
        events = []
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.attributes = None  # No attributes
            event.state = "on" if i % 2 == 0 else "of"
            event.sensor_type = "motion"
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        # Should still extract sensor type ratios and state features
        assert "motion_sensor_ratio" in features
        assert features["motion_sensor_ratio"] == 1.0

    def test_extract_generic_sensor_features_mock_attributes(self, extractor):
        """Test extraction with Mock attributes (non-dict)."""
        events = []
        for i in range(2):
            event = Mock(spec=SensorEvent)
            event.attributes = Mock()  # Mock object, not dict
            event.state = "on"
            event.sensor_type = "motion"
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        # Should handle gracefully and not crash
        assert isinstance(features, dict)
        assert "motion_sensor_ratio" in features

    def test_extract_generic_sensor_features_string_to_numeric_conversion(self, extractor):
        """Test conversion of numeric strings to numeric values."""
        events = []
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.attributes = {
                "temperature": str(20.0 + i),  # Numeric string
                "invalid_number": "not_a_number"  # Invalid numeric string
            }
            event.state = "1"  # Numeric string state
            event.sensor_type = "temperature"
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        # Should convert numeric strings and include them in numeric features
        assert "numeric_count" in features
        assert features["numeric_count"] > 3  # Should include converted strings


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
            "custom_feature": 42.0
        }
        
        with patch('src.features.temporal.TEMPORAL_FEATURE_NAMES', 
                   ["time_since_last_change", "hour_sin", "day_of_week_cos"]):
            validated = extractor.validate_feature_names(extracted_features)
            
            assert "time_since_last_change" in validated
            assert "hour_sin" in validated
            assert "custom_feature" in validated  # Should be preserved

    def test_validate_feature_names_with_mappings(self, extractor):
        """Test feature name validation with name mappings."""
        extracted_features = {
            "time_since_last_off": 1200.0,
            "current_state_duration": 600.0,
            "unmapped_feature": 123.0
        }
        
        with patch('src.features.temporal.TEMPORAL_FEATURE_NAMES',
                   ["time_since_last_change", "current_state_duration"]):
            validated = extractor.validate_feature_names(extracted_features)
            
            # Should map time_since_last_off to time_since_last_change
            assert "time_since_last_change" in validated
            assert validated["time_since_last_change"] == 1200.0
            assert "current_state_duration" in validated
            assert "unmapped_feature" in validated

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
                event.timestamp = base_time + timedelta(minutes=i*30 + j*10)
                event.state = "on"
                event.sensor_type = "motion"
                event.attributes = {}
                events.append(event)
            
            target_time = base_time + timedelta(minutes=i*30 + 60)
            event_batches.append((events, target_time))
        
        results = extractor.extract_batch_features(event_batches)
        
        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert len(result) > 10

    def test_extract_batch_features_with_room_states(self, extractor):
        """Test batch feature extraction with room states."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        # Create batch of event sequences
        event_batches = []
        room_states_batches = []
        
        for i in range(2):
            # Events
            events = [Mock(spec=SensorEvent)]
            events[0].timestamp = base_time + timedelta(minutes=i*30)
            events[0].state = "on"
            events[0].sensor_type = "motion"
            events[0].attributes = {}
            
            target_time = base_time + timedelta(minutes=i*30 + 30)
            event_batches.append((events, target_time))
            
            # Room states
            room_states = [Mock(spec=RoomState)]
            room_states[0].timestamp = base_time + timedelta(minutes=i*30)
            room_states[0].is_occupied = True
            room_states[0].occupancy_confidence = 0.8
            room_states_batches.append(room_states)
        
        results = extractor.extract_batch_features(event_batches, room_states_batches)
        
        assert len(results) == 2
        for result in results:
            assert "avg_occupancy_confidence" in result  # Should have room state features

    def test_extract_batch_features_empty_batch(self, extractor):
        """Test batch feature extraction with empty batch."""
        results = extractor.extract_batch_features([])
        
        assert results == []


class TestTemporalFeatureExtractorErrorHandling:
    """Test error handling and edge cases."""

    @pytest.fixture
    def extractor(self):
        return TemporalFeatureExtractor()

    def test_feature_extraction_with_corrupted_events(self, extractor):
        """Test feature extraction with corrupted event data."""
        # Create events with missing/invalid attributes
        events = []
        
        event1 = Mock(spec=SensorEvent)
        event1.timestamp = datetime(2024, 1, 15, 10, 0, 0)
        event1.room_id = "test_room"
        event1.state = None  # Invalid state
        event1.sensor_type = "motion"
        event1.attributes = "not_a_dict"  # Invalid attributes
        events.append(event1)
        
        event2 = Mock(spec=SensorEvent)
        event2.timestamp = datetime(2024, 1, 15, 10, 30, 0)
        event2.room_id = "test_room"
        event2.state = "on"
        event2.sensor_type = None  # Invalid sensor type
        event2.attributes = {}
        events.append(event2)
        
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        # Should handle gracefully without raising unexpected errors
        features = extractor.extract_features(events, target_time)
        assert isinstance(features, dict)

    def test_duration_features_with_identical_timestamps(self, extractor):
        """Test duration features when events have identical timestamps."""
        timestamp = datetime(2024, 1, 15, 10, 0, 0)
        
        events = []
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = timestamp  # Same timestamp for all
            event.state = "on" if i % 2 == 0 else "of"
            events.append(event)
        
        target_time = timestamp + timedelta(hours=1)
        
        features = extractor._extract_duration_features(events, target_time)
        
        # Should handle zero durations gracefully
        assert isinstance(features, dict)
        assert features["current_state_duration"] > 0  # Time since last event

    def test_historical_patterns_with_single_event(self, extractor):
        """Test historical patterns with only one event."""
        event = Mock(spec=SensorEvent)
        event.timestamp = datetime(2024, 1, 15, 10, 0, 0)
        event.state = "on"
        
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        
        features = extractor._extract_historical_patterns([event], target_time)
        
        # Should handle single event gracefully
        assert isinstance(features, dict)
        assert features["overall_activity_rate"] == 1.0  # Single "on" event
        assert features["trend_coefficient"] == 0.0  # Can't calculate trend

    def test_transition_timing_zero_intervals(self, extractor):
        """Test transition timing with zero intervals."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        
        # Events with same timestamp (zero intervals)
        events = []
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time  # Same time
            event.state = f"state_{i}"
            events.append(event)
        
        target_time = base_time + timedelta(hours=1)
        
        features = extractor._extract_transition_timing_features(events, target_time)
        
        # Should handle zero intervals
        assert features["avg_transition_interval"] == 0.0
        assert features["time_variability"] == 0.0  # No variation in zero intervals

    def test_generic_sensor_features_division_by_zero(self, extractor):
        """Test generic sensor features with potential division by zero."""
        # Create events with empty sensor values
        events = []
        for i in range(2):
            event = Mock(spec=SensorEvent)
            event.attributes = {}  # No attributes
            event.state = ""  # Empty state
            event.sensor_type = ""  # Empty sensor type
            events.append(event)
        
        features = extractor._extract_generic_sensor_features(events)
        
        # Should handle division by zero gracefully
        assert isinstance(features, dict)
        # Ratios should be 0.0 when no sensor values
        assert features.get("numeric_value_ratio", 0.0) == 0.0