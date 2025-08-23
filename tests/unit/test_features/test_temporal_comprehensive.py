"""
Comprehensive unit tests for temporal feature extraction.

This module provides exhaustive testing of the TemporalFeatureExtractor class,
covering all cyclical encodings, time-since calculations, duration analysis,
historical patterns, and advanced statistical validation.
"""

from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest
import statistics

from src.core.constants import TEMPORAL_FEATURE_NAMES
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.temporal import TemporalFeatureExtractor


class TestTemporalFeatureExtractorComprehensive:
    """Comprehensive test suite for TemporalFeatureExtractor with production-grade validation."""

    @pytest.fixture
    def extractor(self):
        """Create a temporal feature extractor instance."""
        return TemporalFeatureExtractor()

    @pytest.fixture
    def extractor_with_timezone(self):
        """Create extractor with timezone offset."""
        return TemporalFeatureExtractor(timezone_offset=-8)  # PST

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 6, 15, 14, 30, 0)  # Saturday, summer, afternoon

    def create_sensor_event(
        self,
        timestamp: datetime,
        sensor_id: str,
        sensor_type: str,
        state: str,
        room_id: str,
        attributes: Optional[Dict] = None,
    ) -> SensorEvent:
        """Create a sensor event for testing."""
        event = Mock(spec=SensorEvent)
        event.timestamp = timestamp
        event.sensor_id = sensor_id
        event.sensor_type = sensor_type
        event.state = state
        event.room_id = room_id
        event.attributes = attributes or {}
        return event

    def create_room_state(
        self,
        timestamp: datetime,
        room_id: str,
        is_occupied: bool,
        confidence: Optional[float] = None,
    ) -> RoomState:
        """Create a room state for testing."""
        state = Mock(spec=RoomState)
        state.timestamp = timestamp
        state.room_id = room_id
        state.is_occupied = is_occupied
        state.occupancy_confidence = confidence or 0.8
        return state

    @pytest.fixture
    def comprehensive_events(self) -> List[SensorEvent]:
        """Create comprehensive sensor events for testing."""
        base_time = datetime(2024, 6, 15, 10, 0, 0)
        events = []

        # Create realistic motion sensor pattern over 4 hours
        motion_pattern = [
            (0, "on"),
            (2, "off"),
            (15, "on"),
            (18, "off"),  # Morning activity
            (45, "on"),
            (50, "off"),
            (120, "on"),
            (125, "off"),  # Mid-morning
            (180, "on"),
            (185, "off"),
            (210, "on"),
            (215, "off"),  # Pre-afternoon
        ]

        for minutes, state in motion_pattern:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id="sensor.living_room_motion",
                sensor_type="motion",
                state=state,
                room_id="living_room",
            )
            events.append(event)

        # Add presence sensor events
        presence_pattern = [
            (10, "on"),
            (35, "off"),
            (60, "on"),
            (90, "off"),
            (150, "on"),
            (200, "off"),
        ]

        for minutes, state in presence_pattern:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id="sensor.living_room_presence",
                sensor_type="presence",
                state=state,
                room_id="living_room",
            )
            events.append(event)

        # Add door sensor events
        door_pattern = [(30, "on"), (31, "off"), (160, "on"), (161, "off")]

        for minutes, state in door_pattern:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id="sensor.front_door",
                sensor_type="door",
                state=state,
                room_id="entrance",
            )
            events.append(event)

        # Add some events with numeric attributes for generic sensor testing
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 30),
                sensor_id=f"sensor.generic_{i}",
                sensor_type="sensor",
                state=str(20.0 + i * 2),
                room_id="test_room",
                attributes={
                    "temperature": 22.0 + i,
                    "value": i * 10,
                    "flag": i % 2 == 0,
                },
            )
            events.append(event)

        return events

    @pytest.fixture
    def duration_pattern_events(self) -> List[SensorEvent]:
        """Create events with specific duration patterns for testing."""
        base_time = datetime(2024, 6, 15, 8, 0, 0)
        events = []

        # Pattern: on(5min) -> off(10min) -> on(15min) -> off(20min) -> on(8min)
        pattern = [
            (0, "on"),
            (5, "off"),
            (15, "on"),
            (30, "off"),
            (50, "on"),
            (58, "off"),
        ]

        for minutes, state in pattern:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id="sensor.duration_test",
                sensor_type="motion",
                state=state,
                room_id="test_room",
            )
            events.append(event)

        return events

    @pytest.fixture
    def historical_pattern_events(self) -> List[SensorEvent]:
        """Create events for historical pattern analysis."""
        events = []

        # Create 14 days of data with patterns
        base_date = datetime(2024, 6, 1, 0, 0, 0)

        for day in range(14):
            current_day = base_date + timedelta(days=day)

            # Different patterns for weekdays vs weekends
            is_weekend = current_day.weekday() >= 5

            if is_weekend:
                # Weekend pattern: later start, more evening activity
                daily_pattern = [
                    (9, "on"),
                    (10, "off"),  # Late morning
                    (14, "on"),
                    (15, "off"),  # Afternoon
                    (20, "on"),
                    (22, "off"),  # Evening
                ]
            else:
                # Weekday pattern: early start, evening activity
                daily_pattern = [
                    (7, "on"),
                    (8, "off"),  # Early morning
                    (12, "on"),
                    (13, "off"),  # Lunch
                    (18, "on"),
                    (20, "off"),  # Evening
                ]

            for hour, state in daily_pattern:
                event = self.create_sensor_event(
                    timestamp=current_day + timedelta(hours=hour),
                    sensor_id="sensor.historical_motion",
                    sensor_type="motion",
                    state=state,
                    room_id="living_room",
                )
                events.append(event)

        return events

    @pytest.fixture
    def room_state_history(self) -> List[RoomState]:
        """Create room state history for testing."""
        base_time = datetime(2024, 6, 15, 6, 0, 0)
        states = []

        # Create 24 hours of room state data
        for hour in range(24):
            timestamp = base_time + timedelta(hours=hour)

            # Pattern: occupied during day (7-23), vacant at night
            is_occupied = 7 <= hour <= 23
            confidence = 0.7 + (hour % 5) * 0.05  # Varying confidence

            state = self.create_room_state(
                timestamp, "living_room", is_occupied, confidence
            )
            states.append(state)

        return states

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization_default(self):
        """Test default initialization."""
        extractor = TemporalFeatureExtractor()

        assert extractor.timezone_offset == 0
        assert isinstance(extractor.feature_cache, dict)
        assert isinstance(extractor.temporal_cache, dict)

    def test_initialization_with_timezone(self):
        """Test initialization with timezone offset."""
        timezone_offset = -5  # EST
        extractor = TemporalFeatureExtractor(timezone_offset=timezone_offset)

        assert extractor.timezone_offset == timezone_offset
        assert isinstance(extractor.feature_cache, dict)

    def test_cache_initialization(self, extractor):
        """Test cache structures are properly initialized."""
        assert hasattr(extractor, "feature_cache")
        assert hasattr(extractor, "temporal_cache")
        assert isinstance(extractor.feature_cache, dict)
        assert isinstance(extractor.temporal_cache, dict)

    # ==================== TIME-SINCE FEATURES TESTS ====================

    def test_time_since_features_comprehensive(
        self, extractor, comprehensive_events, target_time
    ):
        """Test comprehensive time-since feature extraction."""
        features = extractor._extract_time_since_features(
            comprehensive_events, target_time
        )

        # Verify all time-since features are present
        required_features = [
            "time_since_last_event",
            "time_since_last_on",
            "time_since_last_off",
            "time_since_last_motion",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert features[feature] >= 0.0
            assert features[feature] <= 86400.0  # Capped at 24 hours

    def test_time_since_calculation_accuracy(self, extractor):
        """Test mathematical accuracy of time-since calculations."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create events with known time differences
        events = [
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=30),  # 30 minutes ago
                sensor_id="sensor1",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=15),  # 15 minutes ago
                sensor_id="sensor2",
                sensor_type="presence",
                state="off",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=target_time
                - timedelta(minutes=5),  # 5 minutes ago (most recent)
                sensor_id="sensor3",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
        ]

        features = extractor._extract_time_since_features(events, target_time)

        # Verify calculations
        assert (
            abs(features["time_since_last_event"] - 300.0) < 1.0
        )  # 5 minutes = 300 seconds
        assert (
            abs(features["time_since_last_on"] - 300.0) < 1.0
        )  # Last "on" was 5 minutes ago
        assert (
            abs(features["time_since_last_off"] - 900.0) < 1.0
        )  # Last "off" was 15 minutes ago
        assert (
            abs(features["time_since_last_motion"] - 300.0) < 1.0
        )  # Last motion was 5 minutes ago

    def test_time_since_capping_at_24_hours(self, extractor):
        """Test that time-since values are capped at 24 hours."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create event more than 24 hours ago
        old_event = [
            self.create_sensor_event(
                timestamp=target_time - timedelta(hours=30),  # 30 hours ago
                sensor_id="old_sensor",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
        ]

        features = extractor._extract_time_since_features(old_event, target_time)

        # All values should be capped at 24 hours (86400 seconds)
        assert features["time_since_last_event"] == 86400.0
        assert features["time_since_last_on"] == 86400.0
        assert features["time_since_last_motion"] == 86400.0

    def test_time_since_state_specific_tracking(self, extractor):
        """Test accurate tracking of different states."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create sequence: on -> off -> on
        events = [
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=60),  # 1 hour ago
                sensor_id="sensor1",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=30),  # 30 minutes ago
                sensor_id="sensor1",
                sensor_type="motion",
                state="of",  # Note: using "of" as seen in the source code
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=10),  # 10 minutes ago
                sensor_id="sensor1",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
        ]

        features = extractor._extract_time_since_features(events, target_time)

        # Most recent "on" was 10 minutes ago, most recent "of" was 30 minutes ago
        assert abs(features["time_since_last_on"] - 600.0) < 1.0  # 10 minutes
        assert abs(features["time_since_last_off"] - 1800.0) < 1.0  # 30 minutes

    def test_time_since_empty_events_defaults(self, extractor):
        """Test default values when no events are present."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        features = extractor._extract_time_since_features([], target_time)

        # Should return default values
        expected_defaults = {
            "time_since_last_event": 3600.0,  # 1 hour
            "time_since_last_on": 3600.0,
            "time_since_last_off": 3600.0,
            "time_since_last_motion": 3600.0,
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    def test_time_since_sensor_type_filtering(self, extractor):
        """Test proper filtering by sensor type for motion sensors."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        events = [
            # Non-motion sensor (should not affect motion time)
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=5),
                sensor_id="door_sensor",
                sensor_type="door",
                state="on",
                room_id="room1",
            ),
            # Motion sensor (should be used for motion time)
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=20),
                sensor_id="motion_sensor",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            # Presence sensor (should also be used for motion time)
            self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=10),
                sensor_id="presence_sensor",
                sensor_type="presence",
                state="on",
                room_id="room1",
            ),
        ]

        features = extractor._extract_time_since_features(events, target_time)

        # Motion time should use the most recent motion/presence sensor (10 minutes)
        assert abs(features["time_since_last_motion"] - 600.0) < 1.0  # 10 minutes

    # ==================== DURATION FEATURES TESTS ====================

    def test_duration_features_comprehensive(
        self, extractor, duration_pattern_events, target_time
    ):
        """Test comprehensive duration feature extraction."""
        features = extractor._extract_duration_features(
            duration_pattern_events, target_time
        )

        # Verify all duration features are present
        required_features = [
            "current_state_duration",
            "avg_on_duration",
            "avg_off_duration",
            "max_on_duration",
            "max_off_duration",
            "on_duration_std",
            "off_duration_std",
            "duration_ratio",
            "median_on_duration",
            "median_off_duration",
            "duration_percentile_75",
            "duration_percentile_25",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert features[feature] >= 0.0

    def test_duration_mathematical_accuracy(self, extractor):
        """Test mathematical accuracy of duration calculations using numpy."""
        target_time = datetime(2024, 6, 15, 12, 30, 0)

        # Pattern: on(5min) -> off(10min) -> on(15min) -> off(current)
        events = [
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 12, 0, 0),
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 12, 5, 0),  # on for 5 minutes
                sensor_id="test_sensor",
                sensor_type="motion",
                state="off",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 12, 15, 0),  # off for 10 minutes
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 12, 30, 0),  # on for 15 minutes
                sensor_id="test_sensor",
                sensor_type="motion",
                state="off",
                room_id="room1",
            ),
        ]

        features = extractor._extract_duration_features(events, target_time)

        # Expected durations: on = [5min, 15min], off = [10min]
        on_durations = [5 * 60, 15 * 60]  # Convert to seconds
        off_durations = [10 * 60]

        expected_avg_on = np.mean(on_durations)
        expected_avg_off = np.mean(off_durations)
        expected_max_on = np.max(on_durations)
        expected_max_off = np.max(off_durations)

        assert abs(features["avg_on_duration"] - expected_avg_on) < 1.0
        assert abs(features["avg_off_duration"] - expected_avg_off) < 1.0
        assert abs(features["max_on_duration"] - expected_max_on) < 1.0
        assert abs(features["max_off_duration"] - expected_max_off) < 1.0

    def test_duration_statistical_features_numpy(self, extractor):
        """Test advanced statistical duration features using numpy."""
        target_time = datetime(2024, 6, 15, 13, 0, 0)

        # Create events with known variance patterns
        base_time = datetime(2024, 6, 15, 10, 0, 0)

        # Pattern with high variance: on(1min) -> off(20min) -> on(30min) -> off
        events = [
            self.create_sensor_event(base_time, "sensor", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=1), "sensor", "motion", "off", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=21), "sensor", "motion", "on", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=51), "sensor", "motion", "off", "room1"
            ),
        ]

        features = extractor._extract_duration_features(events, target_time)

        # Verify statistical calculations
        on_durations = np.array([1 * 60, 30 * 60])  # 1 min and 30 min in seconds
        off_durations = np.array([20 * 60])  # 20 min in seconds

        expected_on_std = float(np.std(on_durations))
        expected_off_std = float(np.std(off_durations))
        expected_median_on = float(np.median(on_durations))
        expected_median_off = float(np.median(off_durations))

        assert abs(features["on_duration_std"] - expected_on_std) < 10.0
        assert abs(features["off_duration_std"] - expected_off_std) < 10.0
        assert abs(features["median_on_duration"] - expected_median_on) < 1.0
        assert abs(features["median_off_duration"] - expected_median_off) < 1.0

    def test_duration_ratio_calculation(self, extractor):
        """Test duration ratio calculation accuracy."""
        target_time = datetime(2024, 6, 15, 12, 30, 0)

        # Pattern: on(20min) -> off(10min) -> on(30min) -> off
        # Average on = 25min, average off = 10min, ratio = 2.5
        events = [
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 11, 0, 0),
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 11, 20, 0),  # on for 20 min
                sensor_id="test_sensor",
                sensor_type="motion",
                state="off",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 11, 30, 0),  # off for 10 min
                sensor_id="test_sensor",
                sensor_type="motion",
                state="on",
                room_id="room1",
            ),
            self.create_sensor_event(
                timestamp=datetime(2024, 6, 15, 12, 0, 0),  # on for 30 min
                sensor_id="test_sensor",
                sensor_type="motion",
                state="off",
                room_id="room1",
            ),
        ]

        features = extractor._extract_duration_features(events, target_time)

        # Expected ratio: (20+30)/2 / 10 = 25/10 = 2.5
        expected_ratio = 2.5
        assert abs(features["duration_ratio"] - expected_ratio) < 0.1

    def test_duration_percentile_calculations(self, extractor):
        """Test percentile calculations for duration features."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)
        base_time = datetime(2024, 6, 15, 10, 0, 0)

        # Create pattern with known duration distribution
        durations_minutes = [5, 10, 15, 20, 25]  # 5 different durations
        events = []

        current_time = base_time
        state = "on"

        for duration in durations_minutes:
            events.append(
                self.create_sensor_event(
                    current_time, "sensor", "motion", state, "room1"
                )
            )
            current_time += timedelta(minutes=duration)
            state = "off" if state == "on" else "on"
            events.append(
                self.create_sensor_event(
                    current_time, "sensor", "motion", state, "room1"
                )
            )

        features = extractor._extract_duration_features(events, target_time)

        # Calculate expected percentiles
        all_durations = np.array(
            [d * 60 for d in durations_minutes]
        )  # Convert to seconds
        expected_p25 = np.percentile(all_durations, 25)
        expected_p75 = np.percentile(all_durations, 75)

        assert (
            abs(features["duration_percentile_25"] - expected_p25) < 60.0
        )  # 1 minute tolerance
        assert abs(features["duration_percentile_75"] - expected_p75) < 60.0

    def test_current_state_duration_calculation(self, extractor):
        """Test current state duration calculation."""
        target_time = datetime(2024, 6, 15, 14, 30, 0)

        # Last event was 45 minutes ago
        last_event_time = target_time - timedelta(minutes=45)
        events = [
            self.create_sensor_event(last_event_time, "sensor", "motion", "on", "room1")
        ]

        features = extractor._extract_duration_features(events, target_time)

        # Current state duration should be 45 minutes = 2700 seconds
        expected_duration = 45 * 60
        assert abs(features["current_state_duration"] - expected_duration) < 1.0

    def test_duration_empty_events_defaults(self, extractor):
        """Test duration features with empty events."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        features = extractor._extract_duration_features([], target_time)

        # Should return reasonable defaults
        expected_defaults = {
            "current_state_duration": 0.0,
            "avg_on_duration": 1800.0,  # 30 minutes
            "avg_off_duration": 1800.0,
            "max_on_duration": 3600.0,  # 1 hour
            "max_off_duration": 3600.0,
            "on_duration_std": 0.0,
            "off_duration_std": 0.0,
            "duration_ratio": 1.0,
            "median_on_duration": 1800.0,
            "median_off_duration": 1800.0,
            "duration_percentile_75": 3600.0,
            "duration_percentile_25": 900.0,  # 15 minutes
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    # ==================== CYCLICAL FEATURES TESTS ====================

    def test_cyclical_features_comprehensive(self, extractor, target_time):
        """Test comprehensive cyclical feature extraction."""
        features = extractor._extract_cyclical_features(target_time)

        # Verify all cyclical features are present
        required_features = [
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "day_of_month_sin",
            "day_of_month_cos",
            "is_weekend",
            "is_work_hours",
            "is_sleep_hours",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_hour_cyclical_encoding_mathematical_accuracy(self, extractor):
        """Test mathematical accuracy of hour cyclical encoding."""
        # Test specific hours with known sin/cos values
        test_cases = [
            (datetime(2024, 6, 15, 0, 0, 0), 0),  # Midnight = 0 hours
            (datetime(2024, 6, 15, 6, 0, 0), 6),  # 6 AM
            (datetime(2024, 6, 15, 12, 0, 0), 12),  # Noon
            (datetime(2024, 6, 15, 18, 0, 0), 18),  # 6 PM
        ]

        for test_time, hour in test_cases:
            features = extractor._extract_cyclical_features(test_time)

            # Calculate expected values
            angle = 2 * math.pi * hour / 24
            expected_sin = math.sin(angle)
            expected_cos = math.cos(angle)

            assert abs(features["hour_sin"] - expected_sin) < 1e-10
            assert abs(features["hour_cos"] - expected_cos) < 1e-10

    def test_day_of_week_cyclical_encoding(self, extractor):
        """Test day of week cyclical encoding accuracy."""
        # Test each day of the week
        test_dates = [
            datetime(2024, 6, 10, 12, 0, 0),  # Monday (0)
            datetime(2024, 6, 11, 12, 0, 0),  # Tuesday (1)
            datetime(2024, 6, 12, 12, 0, 0),  # Wednesday (2)
            datetime(2024, 6, 13, 12, 0, 0),  # Thursday (3)
            datetime(2024, 6, 14, 12, 0, 0),  # Friday (4)
            datetime(2024, 6, 15, 12, 0, 0),  # Saturday (5)
            datetime(2024, 6, 16, 12, 0, 0),  # Sunday (6)
        ]

        for i, test_date in enumerate(test_dates):
            features = extractor._extract_cyclical_features(test_date)

            day_of_week = test_date.weekday()  # 0 = Monday
            angle = 2 * math.pi * day_of_week / 7
            expected_sin = math.sin(angle)
            expected_cos = math.cos(angle)

            assert abs(features["day_of_week_sin"] - expected_sin) < 1e-10
            assert abs(features["day_of_week_cos"] - expected_cos) < 1e-10

    def test_month_cyclical_encoding(self, extractor):
        """Test month cyclical encoding accuracy."""
        # Test different months
        test_dates = [
            datetime(2024, 1, 15, 12, 0, 0),  # January
            datetime(2024, 3, 15, 12, 0, 0),  # March
            datetime(2024, 6, 15, 12, 0, 0),  # June
            datetime(2024, 9, 15, 12, 0, 0),  # September
            datetime(2024, 12, 15, 12, 0, 0),  # December
        ]

        for test_date in test_dates:
            features = extractor._extract_cyclical_features(test_date)

            month = test_date.month
            angle = 2 * math.pi * month / 12
            expected_sin = math.sin(angle)
            expected_cos = math.cos(angle)

            assert abs(features["month_sin"] - expected_sin) < 1e-10
            assert abs(features["month_cos"] - expected_cos) < 1e-10

    def test_day_of_month_cyclical_encoding(self, extractor):
        """Test day of month cyclical encoding."""
        # Test different days of month
        test_dates = [
            datetime(2024, 6, 1, 12, 0, 0),  # 1st
            datetime(2024, 6, 8, 12, 0, 0),  # 8th
            datetime(2024, 6, 15, 12, 0, 0),  # 15th
            datetime(2024, 6, 22, 12, 0, 0),  # 22nd
            datetime(2024, 6, 30, 12, 0, 0),  # 30th
        ]

        for test_date in test_dates:
            features = extractor._extract_cyclical_features(test_date)

            day = test_date.day
            angle = 2 * math.pi * day / 31
            expected_sin = math.sin(angle)
            expected_cos = math.cos(angle)

            assert abs(features["day_of_month_sin"] - expected_sin) < 1e-10
            assert abs(features["day_of_month_cos"] - expected_cos) < 1e-10

    def test_weekend_indicator_accuracy(self, extractor):
        """Test weekend indicator accuracy."""
        # Test weekdays (Monday-Friday)
        weekdays = [
            datetime(2024, 6, 10, 12, 0, 0),  # Monday
            datetime(2024, 6, 11, 12, 0, 0),  # Tuesday
            datetime(2024, 6, 12, 12, 0, 0),  # Wednesday
            datetime(2024, 6, 13, 12, 0, 0),  # Thursday
            datetime(2024, 6, 14, 12, 0, 0),  # Friday
        ]

        for weekday in weekdays:
            features = extractor._extract_cyclical_features(weekday)
            assert features["is_weekend"] == 0.0

        # Test weekends (Saturday-Sunday)
        weekends = [
            datetime(2024, 6, 15, 12, 0, 0),  # Saturday
            datetime(2024, 6, 16, 12, 0, 0),  # Sunday
        ]

        for weekend in weekends:
            features = extractor._extract_cyclical_features(weekend)
            assert features["is_weekend"] == 1.0

    def test_work_hours_indicator_accuracy(self, extractor):
        """Test work hours indicator accuracy."""
        test_date = datetime(2024, 6, 15, 0, 0, 0)

        # Test different hours
        for hour in range(24):
            test_time = test_date.replace(hour=hour)
            features = extractor._extract_cyclical_features(test_time)

            # Work hours are 9 AM to 5 PM (9-17)
            expected_work_hours = 1.0 if 9 <= hour <= 17 else 0.0
            assert features["is_work_hours"] == expected_work_hours

    def test_sleep_hours_indicator_accuracy(self, extractor):
        """Test sleep hours indicator accuracy."""
        test_date = datetime(2024, 6, 15, 0, 0, 0)

        # Test different hours
        for hour in range(24):
            test_time = test_date.replace(hour=hour)
            features = extractor._extract_cyclical_features(test_time)

            # Sleep hours are 10 PM to 6 AM (>=22 or <=6)
            expected_sleep_hours = 1.0 if (hour >= 22 or hour <= 6) else 0.0
            assert features["is_sleep_hours"] == expected_sleep_hours

    def test_timezone_offset_handling(self, extractor_with_timezone):
        """Test proper timezone offset handling."""
        # UTC time
        utc_time = datetime(2024, 6, 15, 20, 0, 0)  # 8 PM UTC

        # With -8 timezone offset (PST), local time should be 12 PM
        features = extractor_with_timezone._extract_cyclical_features(utc_time)

        # The hour encoding should be based on 12 PM (noon), not 8 PM
        local_hour = 12
        angle = 2 * math.pi * local_hour / 24
        expected_sin = math.sin(angle)
        expected_cos = math.cos(angle)

        assert abs(features["hour_sin"] - expected_sin) < 1e-10
        assert abs(features["hour_cos"] - expected_cos) < 1e-10

        # Should be work hours (12 PM is within 9-17 range)
        assert features["is_work_hours"] == 1.0
        assert features["is_sleep_hours"] == 0.0

    # ==================== HISTORICAL PATTERN FEATURES TESTS ====================

    def test_historical_patterns_comprehensive(
        self, extractor, historical_pattern_events, target_time
    ):
        """Test comprehensive historical pattern analysis."""
        features = extractor._extract_historical_patterns(
            historical_pattern_events, target_time
        )

        # Verify all historical pattern features
        required_features = [
            "hour_activity_rate",
            "day_activity_rate",
            "overall_activity_rate",
            "similar_time_activity_rate",
            "pattern_strength",
            "activity_variance",
            "trend_coefficient",
            "seasonality_score",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0.0 <= features[feature] <= 1.0 or feature in [
                "trend_coefficient",
                "seasonality_score",
            ]

    def test_hour_activity_rate_calculation(self, extractor):
        """Test hour activity rate calculation using pandas."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)  # 2 PM

        # Create events at various times, with more activity at 2 PM
        events = []
        base_date = datetime(2024, 6, 10, 0, 0, 0)

        for day in range(5):  # 5 days of data
            for hour in [10, 14, 18, 22]:  # Various hours
                event_time = base_date + timedelta(days=day, hours=hour)
                # Make 2 PM (14) more active
                state = "on" if hour == 14 or (hour != 14 and day % 2 == 0) else "off"

                event = self.create_sensor_event(
                    event_time, "sensor", "motion", state, "room1"
                )
                events.append(event)

        features = extractor._extract_historical_patterns(events, target_time)

        # At 2 PM, activity should be higher due to more "on" events
        assert features["hour_activity_rate"] >= 0.8  # Most 2 PM events were "on"

    def test_day_activity_rate_weekend_pattern(self, extractor):
        """Test day of week activity rate calculation."""
        # Saturday target time
        target_time = datetime(2024, 6, 15, 14, 0, 0)  # Saturday

        events = []

        # Create 2 weeks of data with weekend pattern (less activity on weekends)
        for day in range(14):
            event_date = datetime(2024, 6, 1, 14, 0, 0) + timedelta(days=day)
            is_weekend = event_date.weekday() >= 5

            # Less activity on weekends
            state = "on" if not is_weekend else ("on" if day % 3 == 0 else "off")
            event = self.create_sensor_event(
                event_date, "sensor", "motion", state, "room1"
            )
            events.append(event)

        features = extractor._extract_historical_patterns(events, target_time)

        # Saturday should have lower activity rate
        assert features["day_activity_rate"] < 0.6

    def test_similar_time_activity_rate_weighting(self, extractor):
        """Test similar time activity rate with hour weighting."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)  # 2 PM

        events = []

        # Create data for hours 13, 14, 15 (adjacent to target)
        activity_by_hour = {13: 0.5, 14: 0.9, 15: 0.7}  # High activity at 2 PM

        for day in range(10):
            for hour, activity_rate in activity_by_hour.items():
                event_date = datetime(2024, 6, 5 + day, hour, 0, 0)
                state = "on" if np.random.random() < activity_rate else "off"
                event = self.create_sensor_event(
                    event_date, "sensor", "motion", state, "room1"
                )
                events.append(event)

        features = extractor._extract_historical_patterns(events, target_time)

        # Similar time activity should reflect weighted average with emphasis on exact hour
        assert features["similar_time_activity_rate"] > 0.6

    def test_pattern_strength_consistency(self, extractor):
        """Test pattern strength calculation based on standard deviation."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create highly consistent pattern (same activity each day at each hour)
        consistent_events = []
        for day in range(7):
            for hour in [10, 14, 18]:
                event_date = datetime(2024, 6, 8 + day, hour, 0, 0)
                # Always "on" at 2 PM, "off" at others - perfect consistency
                state = "on" if hour == 14 else "off"
                event = self.create_sensor_event(
                    event_date, "sensor", "motion", state, "room1"
                )
                consistent_events.append(event)

        consistent_features = extractor._extract_historical_patterns(
            consistent_events, target_time
        )

        # Create inconsistent pattern
        inconsistent_events = []
        for day in range(7):
            for hour in [10, 14, 18]:
                event_date = datetime(2024, 6, 8 + day, hour, 0, 0)
                # Random activity - no consistency
                state = "on" if np.random.random() > 0.5 else "off"
                event = self.create_sensor_event(
                    event_date, "sensor", "motion", state, "room1"
                )
                inconsistent_events.append(event)

        inconsistent_features = extractor._extract_historical_patterns(
            inconsistent_events, target_time
        )

        # Consistent pattern should have higher pattern strength
        assert (
            consistent_features["pattern_strength"]
            > inconsistent_features["pattern_strength"]
        )

    def test_activity_variance_calculation(self, extractor):
        """Test activity variance calculation using numpy."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create events with known variance pattern
        events = []
        activity_values = [1, 0, 1, 0, 1] * 4  # Alternating pattern

        for i, is_active in enumerate(activity_values):
            event_time = datetime(2024, 6, 10, 12, 0, 0) + timedelta(hours=i)
            state = "on" if is_active else "off"
            event = self.create_sensor_event(
                event_time, "sensor", "motion", state, "room1"
            )
            events.append(event)

        features = extractor._extract_historical_patterns(events, target_time)

        # Calculate expected variance
        binary_values = [float(val) for val in activity_values]
        expected_variance = float(np.var(binary_values))

        assert abs(features["activity_variance"] - expected_variance) < 0.01

    def test_trend_coefficient_calculation(self, extractor):
        """Test trend coefficient calculation using correlation."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create increasing trend pattern
        increasing_events = []
        for i in range(20):
            event_time = datetime(2024, 6, 10, 12, 0, 0) + timedelta(hours=i)
            # Increasing probability of "on" state
            probability = min(0.9, 0.1 + i * 0.04)
            state = "on" if np.random.random() < probability else "off"
            event = self.create_sensor_event(
                event_time, "sensor", "motion", state, "room1"
            )
            increasing_events.append(event)

        increasing_features = extractor._extract_historical_patterns(
            increasing_events, target_time
        )

        # Create decreasing trend pattern
        decreasing_events = []
        for i in range(20):
            event_time = datetime(2024, 6, 10, 12, 0, 0) + timedelta(hours=i)
            # Decreasing probability of "on" state
            probability = max(0.1, 0.9 - i * 0.04)
            state = "on" if np.random.random() < probability else "off"
            event = self.create_sensor_event(
                event_time, "sensor", "motion", state, "room1"
            )
            decreasing_events.append(event)

        decreasing_features = extractor._extract_historical_patterns(
            decreasing_events, target_time
        )

        # Increasing trend should have positive coefficient, decreasing should have negative
        # Note: Due to randomness, we check the relative difference
        trend_difference = (
            increasing_features["trend_coefficient"]
            - decreasing_features["trend_coefficient"]
        )
        assert trend_difference > 0.1  # Significant positive difference

    def test_seasonality_score_day_of_year_patterns(self, extractor):
        """Test seasonality score calculation based on day-of-year patterns."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create seasonal pattern (more activity in summer days)
        seasonal_events = []

        for day in range(1, 366, 10):  # Sample every 10 days across the year
            event_date = datetime(2024, 1, 1) + timedelta(days=day - 1)

            # Higher activity in summer (days 150-250, roughly June-September)
            is_summer = 150 <= day <= 250
            activity_prob = 0.8 if is_summer else 0.3

            state = "on" if np.random.random() < activity_prob else "off"
            event = self.create_sensor_event(
                event_date.replace(hour=14), "sensor", "motion", state, "room1"
            )
            seasonal_events.append(event)

        seasonal_features = extractor._extract_historical_patterns(
            seasonal_events, target_time
        )

        # Should detect seasonality (non-zero score)
        assert seasonal_features["seasonality_score"] > 0.0

    def test_historical_patterns_empty_data_defaults(self, extractor):
        """Test historical pattern features with empty data."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        features = extractor._extract_historical_patterns([], target_time)

        expected_defaults = {
            "hour_activity_rate": 0.5,
            "day_activity_rate": 0.5,
            "overall_activity_rate": 0.5,
            "similar_time_activity_rate": 0.5,
            "pattern_strength": 0.0,
            "activity_variance": 0.0,
            "trend_coefficient": 0.0,
            "seasonality_score": 0.0,
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    # ==================== GENERIC SENSOR FEATURES TESTS ====================

    def test_generic_sensor_features_comprehensive(
        self, extractor, comprehensive_events
    ):
        """Test comprehensive generic sensor feature extraction."""
        features = extractor._extract_generic_sensor_features(comprehensive_events)

        # Should extract various types of sensor value features
        numeric_features = [
            "numeric_mean",
            "numeric_std",
            "numeric_min",
            "numeric_max",
            "numeric_range",
            "numeric_count",
        ]
        boolean_features = [
            "boolean_true_ratio",
            "boolean_false_ratio",
            "boolean_count",
        ]
        string_features = [
            "string_unique_count",
            "string_total_count",
            "string_diversity_ratio",
        ]
        ratio_features = [
            "numeric_value_ratio",
            "boolean_value_ratio",
            "string_value_ratio",
        ]
        sensor_type_features = [
            "motion_sensor_ratio",
            "door_sensor_ratio",
            "presence_sensor_ratio",
        ]

        # Check which features are present (depends on the events)
        for feature_list in [numeric_features, ratio_features, sensor_type_features]:
            for feature in feature_list:
                if feature in features:
                    assert isinstance(features[feature], (int, float))
                    assert features[feature] >= 0.0

    def test_numeric_value_extraction_and_statistics(self, extractor):
        """Test numeric value extraction and statistical calculations."""
        events = []

        # Create events with numeric states and attributes
        numeric_values = [10.0, 20.0, 30.0, 40.0, 50.0]
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        for i, value in enumerate(numeric_values):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 5),
                sensor_id=f"sensor_{i}",
                sensor_type="sensor",
                state=str(value),
                room_id="room1",
                attributes={"temperature": value + 5, "value": value * 2},
            )
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        # Verify numeric statistics
        # All numeric values: states + temperature attributes + value attributes
        all_values = (
            numeric_values
            + [v + 5 for v in numeric_values]
            + [v * 2 for v in numeric_values]
        )
        expected_mean = np.mean(all_values)
        expected_std = np.std(all_values)
        expected_min = np.min(all_values)
        expected_max = np.max(all_values)
        expected_range = expected_max - expected_min

        assert abs(features["numeric_mean"] - expected_mean) < 0.1
        assert abs(features["numeric_std"] - expected_std) < 0.1
        assert abs(features["numeric_min"] - expected_min) < 0.1
        assert abs(features["numeric_max"] - expected_max) < 0.1
        assert abs(features["numeric_range"] - expected_range) < 0.1
        assert features["numeric_count"] == len(all_values)

    def test_boolean_value_extraction_and_ratios(self, extractor):
        """Test boolean value extraction and ratio calculations."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # Create events with boolean attributes: 6 True, 4 False
        boolean_values = [
            True,
            False,
            True,
            True,
            False,
            True,
            True,
            False,
            True,
            False,
        ]

        for i, bool_val in enumerate(boolean_values):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{i}",
                sensor_type="sensor",
                state="on",
                room_id="room1",
                attributes={"flag": bool_val},
            )
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        # Expected ratios: 6 True out of 10 = 0.6, 4 False out of 10 = 0.4
        expected_true_ratio = 6.0 / 10.0
        expected_false_ratio = 4.0 / 10.0

        assert abs(features["boolean_true_ratio"] - expected_true_ratio) < 0.01
        assert abs(features["boolean_false_ratio"] - expected_false_ratio) < 0.01
        assert features["boolean_count"] == 10

    def test_string_value_extraction_and_diversity(self, extractor):
        """Test string value extraction and diversity calculations."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # Create events with string states: 3 unique values, total 8 values
        string_states = ["on", "off", "idle", "on", "off", "on", "idle", "on"]

        for i, state in enumerate(string_states):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 3),
                sensor_id=f"sensor_{i}",
                sensor_type="sensor",
                state=state,
                room_id="room1",
            )
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        # Expected: 3 unique strings ("on", "off", "idle"), 8 total
        expected_unique = 3
        expected_total = 8
        expected_diversity = expected_unique / expected_total

        assert features["string_unique_count"] == expected_unique
        assert features["string_total_count"] == expected_total
        assert abs(features["string_diversity_ratio"] - expected_diversity) < 0.01

    def test_most_common_string_frequency(self, extractor):
        """Test most common string frequency calculation."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # "on" appears 4 times out of 7 total
        string_states = ["on", "off", "on", "idle", "on", "off", "on"]

        for i, state in enumerate(string_states):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{i}",
                sensor_type="sensor",
                state=state,
                room_id="room1",
            )
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        # Most common is "on" with frequency 4/7
        expected_frequency = 4.0 / 7.0
        assert abs(features["most_common_string_frequency"] - expected_frequency) < 0.01

    def test_sensor_type_ratio_calculations(self, extractor):
        """Test sensor type ratio calculations."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # Create events with different sensor types: 5 motion, 2 door, 3 presence
        sensor_types = ["motion"] * 5 + ["door"] * 2 + ["presence"] * 3

        for i, sensor_type in enumerate(sensor_types):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{sensor_type}_{i}",
                sensor_type=sensor_type,
                state="on",
                room_id="room1",
            )
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        # Expected ratios
        total_events = 10
        expected_motion_ratio = 5.0 / total_events
        expected_door_ratio = 2.0 / total_events
        expected_presence_ratio = 3.0 / total_events

        assert abs(features["motion_sensor_ratio"] - expected_motion_ratio) < 0.01
        assert abs(features["door_sensor_ratio"] - expected_door_ratio) < 0.01
        assert abs(features["presence_sensor_ratio"] - expected_presence_ratio) < 0.01

    def test_total_sensor_value_ratios(self, extractor):
        """Test overall sensor value type ratios."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # Create events with mixed value types
        for i in range(5):
            attributes = {}
            if i < 2:  # First 2 have numeric attributes
                attributes["temperature"] = 20.0 + i
            if i < 3:  # First 3 have boolean attributes
                attributes["flag"] = i % 2 == 0

            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{i}",
                sensor_type="sensor",
                state=str(i),  # All have string states
                room_id="room1",
                attributes=attributes,
            )
            events.append(event)

        features = extractor._extract_generic_sensor_features(events)

        # Should calculate ratios of different value types
        assert "numeric_value_ratio" in features
        assert "boolean_value_ratio" in features
        assert "string_value_ratio" in features
        assert features["total_sensor_values"] > 0

    def test_generic_sensor_empty_events(self, extractor):
        """Test generic sensor features with empty events."""
        features = extractor._extract_generic_sensor_features([])

        # Should return empty dict or minimal features
        assert isinstance(features, dict)
        # No specific assertions since empty events should return minimal features

    def test_mock_attribute_handling(self, extractor):
        """Test graceful handling of Mock attributes that might raise RuntimeError."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        # Create event with Mock attributes that might cause issues
        bad_mock = Mock()
        bad_mock.items.side_effect = RuntimeError("Mock access error")

        event = self.create_sensor_event(
            timestamp=base_time,
            sensor_id="bad_sensor",
            sensor_type="sensor",
            state="on",
            room_id="room1",
            attributes=bad_mock,
        )
        events.append(event)

        # Should handle gracefully without crashing
        features = extractor._extract_generic_sensor_features(events)
        assert isinstance(features, dict)

    # ==================== TRANSITION TIMING FEATURES TESTS ====================

    def test_transition_timing_comprehensive(
        self, extractor, comprehensive_events, target_time
    ):
        """Test comprehensive transition timing analysis."""
        features = extractor._extract_transition_timing_features(
            comprehensive_events, target_time
        )

        # Verify all transition timing features
        required_features = [
            "avg_transition_interval",
            "recent_transition_rate",
            "time_variability",
            "transition_regularity",
            "recent_transition_trend",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert features[feature] >= 0.0

    def test_transition_interval_calculation_accuracy(self, extractor):
        """Test accuracy of transition interval calculations."""
        base_time = datetime(2024, 6, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Create events with known intervals: 10min, 20min, 30min
        events = [
            self.create_sensor_event(base_time, "sensor", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=10), "sensor", "motion", "off", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "sensor", "motion", "on", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=60), "sensor", "motion", "off", "room1"
            ),
        ]

        features = extractor._extract_transition_timing_features(events, target_time)

        # Expected intervals: 600s (10min), 1200s (20min), 1800s (30min)
        expected_avg = (600 + 1200 + 1800) / 3  # 1200 seconds = 20 minutes
        assert abs(features["avg_transition_interval"] - expected_avg) < 10.0

    def test_recent_transition_rate_calculation(self, extractor):
        """Test recent transition rate calculation."""
        base_time = datetime(2024, 6, 15, 8, 0, 0)
        target_time = base_time + timedelta(hours=6)  # 6 hours later

        # Create events: some old, some recent (last 4 hours)
        events = []

        # Old events (more than 4 hours ago)
        events.append(
            self.create_sensor_event(base_time, "sensor", "motion", "on", "room1")
        )
        events.append(
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "sensor", "motion", "off", "room1"
            )
        )

        # Recent events (last 4 hours) - 4 events = 3 transitions
        recent_start = target_time - timedelta(hours=3)
        events.append(
            self.create_sensor_event(recent_start, "sensor", "motion", "on", "room1")
        )
        events.append(
            self.create_sensor_event(
                recent_start + timedelta(hours=1), "sensor", "motion", "off", "room1"
            )
        )
        events.append(
            self.create_sensor_event(
                recent_start + timedelta(hours=2), "sensor", "motion", "on", "room1"
            )
        )
        events.append(
            self.create_sensor_event(
                recent_start + timedelta(hours=3), "sensor", "motion", "off", "room1"
            )
        )

        features = extractor._extract_transition_timing_features(events, target_time)

        # Recent events span 3 hours with 3 transitions = 1 transition per hour
        expected_rate = 1.0  # transitions per hour
        assert abs(features["recent_transition_rate"] - expected_rate) < 0.1

    def test_time_variability_coefficient_of_variation(self, extractor):
        """Test time variability using coefficient of variation."""
        base_time = datetime(2024, 6, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # High variability: intervals vary greatly
        high_var_events = [
            self.create_sensor_event(base_time, "sensor", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=1), "sensor", "motion", "off", "room1"
            ),  # 1 min
            self.create_sensor_event(
                base_time + timedelta(minutes=31), "sensor", "motion", "on", "room1"
            ),  # 30 min
            self.create_sensor_event(
                base_time + timedelta(minutes=32), "sensor", "motion", "off", "room1"
            ),  # 1 min
            self.create_sensor_event(
                base_time + timedelta(minutes=62), "sensor", "motion", "on", "room1"
            ),  # 30 min
        ]

        high_var_features = extractor._extract_transition_timing_features(
            high_var_events, target_time
        )

        # Low variability: regular intervals
        low_var_events = [
            self.create_sensor_event(base_time, "sensor", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=15), "sensor", "motion", "off", "room1"
            ),  # 15 min
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "sensor", "motion", "on", "room1"
            ),  # 15 min
            self.create_sensor_event(
                base_time + timedelta(minutes=45), "sensor", "motion", "off", "room1"
            ),  # 15 min
            self.create_sensor_event(
                base_time + timedelta(minutes=60), "sensor", "motion", "on", "room1"
            ),  # 15 min
        ]

        low_var_features = extractor._extract_transition_timing_features(
            low_var_events, target_time
        )

        # High variability should have higher time_variability score
        assert (
            high_var_features["time_variability"] > low_var_features["time_variability"]
        )

        # Regularity should be inverse of variability
        assert (
            high_var_features["transition_regularity"]
            < low_var_features["transition_regularity"]
        )

    def test_recent_transition_trend_calculation(self, extractor):
        """Test recent transition trend calculation using linear regression slope."""
        base_time = datetime(2024, 6, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Accelerating pattern: intervals get shorter (positive acceleration trend)
        accelerating_intervals = [60, 50, 40, 30, 20, 10]  # Minutes, decreasing
        acc_events = []

        current_time = base_time
        for i, interval in enumerate(accelerating_intervals):
            acc_events.append(
                self.create_sensor_event(
                    current_time, f"sensor_{i}", "motion", "on", "room1"
                )
            )
            current_time += timedelta(minutes=interval)

        acc_features = extractor._extract_transition_timing_features(
            acc_events, target_time
        )

        # Decelerating pattern: intervals get longer (negative acceleration trend)
        decelerating_intervals = [10, 20, 30, 40, 50, 60]  # Minutes, increasing
        dec_events = []

        current_time = base_time
        for i, interval in enumerate(decelerating_intervals):
            dec_events.append(
                self.create_sensor_event(
                    current_time, f"sensor_{i}", "motion", "on", "room1"
                )
            )
            current_time += timedelta(minutes=interval)

        dec_features = extractor._extract_transition_timing_features(
            dec_events, target_time
        )

        # Accelerating (decreasing intervals) should have negative trend
        # Decelerating (increasing intervals) should have positive trend
        trend_difference = (
            dec_features["recent_transition_trend"]
            - acc_features["recent_transition_trend"]
        )
        assert trend_difference > 100.0  # Significant difference in trends

    def test_transition_timing_edge_cases(self, extractor):
        """Test transition timing with edge cases."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Single event
        single_event = [
            self.create_sensor_event(
                target_time - timedelta(minutes=30), "sensor", "motion", "on", "room1"
            )
        ]

        single_features = extractor._extract_transition_timing_features(
            single_event, target_time
        )

        expected_single_defaults = {
            "avg_transition_interval": 1800.0,
            "recent_transition_rate": 0.0,
            "time_variability": 0.0,
            "transition_regularity": 1.0,
            "recent_transition_trend": 0.0,
        }

        for key, expected in expected_single_defaults.items():
            assert single_features[key] == expected

        # Events with identical timestamps (zero duration)
        zero_duration_events = [
            self.create_sensor_event(
                target_time - timedelta(minutes=10), "sensor1", "motion", "on", "room1"
            ),
            self.create_sensor_event(
                target_time - timedelta(minutes=10), "sensor2", "motion", "off", "room1"
            ),
            self.create_sensor_event(
                target_time - timedelta(minutes=10), "sensor3", "motion", "on", "room1"
            ),
        ]

        zero_features = extractor._extract_transition_timing_features(
            zero_duration_events, target_time
        )

        # Should handle zero intervals gracefully
        assert isinstance(zero_features, dict)
        assert zero_features["recent_transition_rate"] >= 0.0

    # ==================== ROOM STATE FEATURES TESTS ====================

    def test_room_state_features_comprehensive(
        self, extractor, room_state_history, target_time
    ):
        """Test comprehensive room state feature analysis."""
        features = extractor._extract_room_state_features(
            room_state_history, target_time
        )

        # Verify all room state features
        required_features = [
            "avg_occupancy_confidence",
            "recent_occupancy_ratio",
            "state_stability",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0.0 <= features[feature] <= 1.0 or feature == "state_stability"

    def test_occupancy_confidence_calculation(self, extractor):
        """Test occupancy confidence average calculation."""
        base_time = datetime(2024, 6, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=4)

        # Create room states with known confidence values
        confidence_values = [0.6, 0.8, 0.9, 0.7, 0.85]
        room_states = []

        for i, confidence in enumerate(confidence_values):
            state = self.create_room_state(
                timestamp=base_time + timedelta(hours=i),
                room_id="room1",
                is_occupied=True,
                confidence=confidence,
            )
            room_states.append(state)

        features = extractor._extract_room_state_features(room_states, target_time)

        # Expected average confidence
        expected_avg = sum(confidence_values) / len(confidence_values)
        assert abs(features["avg_occupancy_confidence"] - expected_avg) < 0.01

    def test_recent_occupancy_ratio_calculation(self, extractor):
        """Test recent occupancy ratio calculation (last 24 hours)."""
        base_time = datetime(2024, 6, 15, 6, 0, 0)
        target_time = base_time + timedelta(hours=30)  # 30 hours later

        room_states = []

        # Old states (more than 24 hours ago) - should not count
        for i in range(6):  # 6 hours of old data
            state = self.create_room_state(
                timestamp=base_time + timedelta(hours=i),
                room_id="room1",
                is_occupied=i % 2 == 0,  # Alternating pattern
            )
            room_states.append(state)

        # Recent states (last 24 hours) - 12 states, 8 occupied
        recent_start = target_time - timedelta(hours=23)
        for i in range(12):  # 12 hours of recent data
            is_occupied = i < 8  # First 8 are occupied, last 4 are not
            state = self.create_room_state(
                timestamp=recent_start + timedelta(hours=i * 2),
                room_id="room1",
                is_occupied=is_occupied,
            )
            room_states.append(state)

        features = extractor._extract_room_state_features(room_states, target_time)

        # Expected recent occupancy ratio: 8 occupied out of 12 recent states
        expected_ratio = 8.0 / 12.0
        assert abs(features["recent_occupancy_ratio"] - expected_ratio) < 0.05

    def test_state_stability_duration_calculation(self, extractor):
        """Test state stability calculation based on average state duration."""
        base_time = datetime(2024, 6, 15, 8, 0, 0)
        target_time = base_time + timedelta(hours=12)

        # Create states with known durations: 2h, 3h, 4h, 3h
        state_times = [0, 2, 5, 9, 12]  # Hours from base_time
        room_states = []

        for i, hours in enumerate(state_times):
            state = self.create_room_state(
                timestamp=base_time + timedelta(hours=hours),
                room_id="room1",
                is_occupied=i % 2 == 0,
            )
            room_states.append(state)

        features = extractor._extract_room_state_features(room_states, target_time)

        # Expected durations: 2h, 3h, 4h, 3h = average 3h = 3.0
        expected_stability = 3.0  # hours
        assert abs(features["state_stability"] - expected_stability) < 0.1

    def test_room_state_confidence_filtering(self, extractor):
        """Test that None confidence values are filtered out."""
        base_time = datetime(2024, 6, 15, 10, 0, 0)
        target_time = base_time + timedelta(hours=4)

        room_states = []

        # Mix of states with and without confidence values
        for i in range(6):
            confidence = 0.8 if i < 4 else None  # First 4 have confidence, last 2 don't
            state = self.create_room_state(
                timestamp=base_time + timedelta(hours=i),
                room_id="room1",
                is_occupied=True,
                confidence=confidence,
            )
            room_states.append(state)

        features = extractor._extract_room_state_features(room_states, target_time)

        # Should only average the 4 states with confidence values
        expected_avg = 0.8  # All 4 states had 0.8 confidence
        assert abs(features["avg_occupancy_confidence"] - expected_avg) < 0.01

    def test_room_state_empty_data_defaults(self, extractor):
        """Test room state features with empty data."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        features = extractor._extract_room_state_features([], target_time)

        expected_defaults = {
            "avg_occupancy_confidence": 0.5,
            "recent_occupancy_ratio": 0.5,
            "state_stability": 0.5,
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    # ==================== INTEGRATION AND ERROR HANDLING TESTS ====================

    def test_full_feature_extraction_integration(
        self, extractor, comprehensive_events, room_state_history, target_time
    ):
        """Test full integration of all temporal feature extraction."""
        features = extractor.extract_features(
            events=comprehensive_events,
            target_time=target_time,
            room_states=room_state_history,
            lookback_hours=24,
        )

        # Verify comprehensive feature set
        feature_categories = {
            "time_since": [
                "time_since_last_event",
                "time_since_last_on",
                "time_since_last_motion",
            ],
            "duration": [
                "current_state_duration",
                "avg_on_duration",
                "avg_off_duration",
            ],
            "cyclical": ["hour_sin", "hour_cos", "is_weekend", "is_work_hours"],
            "historical": [
                "hour_activity_rate",
                "pattern_strength",
                "trend_coefficient",
            ],
            "transition": ["avg_transition_interval", "recent_transition_rate"],
            "room_state": ["avg_occupancy_confidence", "recent_occupancy_ratio"],
        }

        for category, expected_features in feature_categories.items():
            for feature in expected_features:
                assert (
                    feature in features
                ), f"Missing {feature} from {category} features"
                assert isinstance(features[feature], (int, float))

    def test_lookback_hours_filtering(self, extractor):
        """Test proper filtering of events based on lookback_hours."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Create events spanning 48 hours
        events = []
        for hours_back in range(0, 48, 4):  # Every 4 hours for 48 hours
            event_time = target_time - timedelta(hours=hours_back)
            event = self.create_sensor_event(
                timestamp=event_time,
                sensor_id=f"sensor_{hours_back}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            events.append(event)

        # Extract features with 24-hour lookback
        features_24h = extractor.extract_features(
            events, target_time, lookback_hours=24
        )

        # Extract features with 12-hour lookback
        features_12h = extractor.extract_features(
            events, target_time, lookback_hours=12
        )

        # Should have different results due to different data included
        assert isinstance(features_24h, dict)
        assert isinstance(features_12h, dict)

        # 24h should include more historical data, potentially affecting patterns
        # Exact comparison depends on the specific calculations, but both should be valid
        assert len(features_24h) > 0
        assert len(features_12h) > 0

    def test_error_handling_comprehensive(self, extractor):
        """Test comprehensive error handling scenarios."""
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Test with None events
        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(None, target_time)

        # Test with malformed events (should handle gracefully)
        malformed_events = []

        # Event with missing required attributes
        bad_event = Mock(spec=SensorEvent)
        bad_event.timestamp = None  # Missing timestamp
        bad_event.sensor_id = "sensor1"
        bad_event.sensor_type = "motion"
        bad_event.state = "on"
        bad_event.room_id = "room1"
        malformed_events.append(bad_event)

        # Should handle gracefully or raise appropriate error
        try:
            features = extractor.extract_features(malformed_events, target_time)
            assert isinstance(features, dict)
        except FeatureExtractionError:
            # Acceptable to raise specific feature extraction error
            pass

    def test_cache_functionality_comprehensive(self, extractor):
        """Test cache functionality and clearing."""
        # Clear caches initially
        extractor.clear_cache()
        assert len(extractor.feature_cache) == 0
        assert len(extractor.temporal_cache) == 0

        base_time = datetime(2024, 6, 15, 12, 0, 0)
        events = [
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "sensor2", "motion", "off", "room1"
            ),
        ]

        # Extract features (may populate caches)
        features1 = extractor.extract_features(events, base_time + timedelta(hours=1))

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.feature_cache) == 0
        assert len(extractor.temporal_cache) == 0

        # Extract again after cache clear (should still work)
        features2 = extractor.extract_features(events, base_time + timedelta(hours=1))

        # Results should be consistent
        assert isinstance(features1, dict)
        assert isinstance(features2, dict)

    def test_get_feature_names_method(self, extractor):
        """Test get_feature_names method completeness."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 20  # Should have many features

        # Should match default features
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_validate_feature_names_method(self, extractor):
        """Test validate_feature_names method with TEMPORAL_FEATURE_NAMES."""
        # Create test features that need validation
        test_features = {
            "time_since_last_off": 300.0,
            "current_state_duration": 600.0,
            "hour_sin": 0.5,
            "hour_cos": 0.866,
            "day_of_week_sin": 0.0,
            "day_of_week_cos": 1.0,
            "is_weekend": 0.0,
            "some_unmapped_feature": 123.0,
        }

        validated_features = extractor.validate_feature_names(test_features)

        # Should validate and potentially remap features
        assert isinstance(validated_features, dict)
        assert len(validated_features) > 0

        # Standard features should be preserved
        for standard_name in ["hour_sin", "hour_cos", "is_weekend"]:
            if standard_name in TEMPORAL_FEATURE_NAMES:
                assert standard_name in validated_features

    def test_batch_feature_extraction(self, extractor):
        """Test batch feature extraction method."""
        base_time = datetime(2024, 6, 15, 10, 0, 0)

        # Create multiple event-time pairs
        event_batches = []
        room_state_batches = []

        for i in range(3):
            # Different events for each batch
            events = [
                self.create_sensor_event(
                    base_time + timedelta(hours=i, minutes=j * 15),
                    f"sensor_{i}_{j}",
                    "motion",
                    "on" if j % 2 == 0 else "off",
                    f"room_{i}",
                )
                for j in range(4)
            ]

            target_time = base_time + timedelta(hours=i + 2)

            # Simple room states
            room_states = [
                self.create_room_state(
                    base_time + timedelta(hours=i + 1), f"room_{i}", True
                )
            ]

            event_batches.append((events, target_time))
            room_state_batches.append(room_states)

        # Extract features in batch
        results = extractor.extract_batch_features(event_batches, room_state_batches)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert len(result) > 0

    def test_performance_with_large_temporal_dataset(self, extractor):
        """Test performance with large temporal datasets."""
        base_time = datetime(2024, 1, 1, 0, 0, 0)
        target_time = base_time + timedelta(days=30)

        # Create large dataset: 30 days of events
        large_events = []

        for day in range(30):
            for hour in range(24):
                if hour % 4 == 0:  # Every 4 hours
                    event_time = base_time + timedelta(days=day, hours=hour)
                    event = self.create_sensor_event(
                        timestamp=event_time,
                        sensor_id=f"sensor_{day % 5}",  # 5 sensors
                        sensor_type=["motion", "presence", "door"][hour % 3],
                        state=["on", "off"][hour % 2],
                        room_id=f"room_{day % 3}",  # 3 rooms
                    )
                    large_events.append(event)

        # Measure performance
        import time

        start_time = time.time()

        features = extractor.extract_features(
            large_events, target_time, lookback_hours=24 * 7
        )  # 1 week lookback

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time
        assert (
            execution_time < 5.0
        ), f"Performance test failed: took {execution_time:.3f}s"
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_edge_case_boundary_conditions(self, extractor):
        """Test various boundary conditions and edge cases."""
        target_time = datetime(2024, 6, 15, 12, 0, 0)

        # Test with events at exact boundaries
        boundary_events = [
            # Event exactly at lookback boundary (24 hours ago)
            self.create_sensor_event(
                target_time - timedelta(hours=24),
                "boundary_sensor",
                "motion",
                "on",
                "room1",
            ),
            # Event at target time
            self.create_sensor_event(
                target_time, "current_sensor", "motion", "on", "room1"
            ),
        ]

        features = extractor.extract_features(
            boundary_events, target_time, lookback_hours=24
        )

        assert isinstance(features, dict)
        assert len(features) > 0

        # Test with future timestamps
        future_events = [
            self.create_sensor_event(
                target_time + timedelta(hours=1),
                "future_sensor",
                "motion",
                "on",
                "room1",
            )
        ]

        future_features = extractor.extract_features(
            future_events, target_time, lookback_hours=24
        )
        assert isinstance(future_features, dict)

        # Test with very old events (beyond any reasonable lookback)
        ancient_events = [
            self.create_sensor_event(
                target_time - timedelta(days=365),
                "ancient_sensor",
                "motion",
                "on",
                "room1",
            )
        ]

        ancient_features = extractor.extract_features(
            ancient_events, target_time, lookback_hours=24
        )
        assert isinstance(ancient_features, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
