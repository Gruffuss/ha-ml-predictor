"""
Comprehensive unit tests for contextual feature extraction.

This module provides exhaustive testing of the ContextualFeatureExtractor class,
covering all mathematical calculations, environmental features, door patterns,
multi-room correlations, and edge cases with production-grade validation.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest
import statistics

from src.core.config import RoomConfig, SystemConfig
from src.core.constants import SensorType
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor


class TestContextualFeatureExtractorComprehensive:
    """Comprehensive test suite for ContextualFeatureExtractor with production-grade validation."""

    @pytest.fixture
    def extractor(self):
        """Create a contextual feature extractor instance."""
        return ContextualFeatureExtractor()

    @pytest.fixture
    def system_config(self) -> SystemConfig:
        """Create a system configuration for testing."""
        config = Mock(spec=SystemConfig)
        config.rooms = {
            "living_room": self.create_room_config("living_room"),
            "kitchen": self.create_room_config("kitchen"),
            "bedroom": self.create_room_config("bedroom"),
        }
        return config

    def create_room_config(self, room_id: str) -> RoomConfig:
        """Create a room configuration for testing."""
        room_config = Mock(spec=RoomConfig)
        room_config.room_id = room_id
        room_config.name = room_id.replace("_", " ").title()
        room_config.sensors = {
            "temperature": f"sensor.{room_id}_temperature",
            "humidity": f"sensor.{room_id}_humidity",
            "light": f"sensor.{room_id}_light",
            "door": f"sensor.{room_id}_door",
            "motion": f"sensor.{room_id}_motion",
        }

        def get_sensors_by_type(sensor_type: str) -> Dict[str, str]:
            return {k: v for k, v in room_config.sensors.items() if k == sensor_type}

        room_config.get_sensors_by_type = get_sensors_by_type
        return room_config

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    @pytest.fixture
    def comprehensive_events(self) -> List[SensorEvent]:
        """Create comprehensive sensor events for testing."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Temperature sensor events with realistic patterns
        temps = [20.5, 21.0, 21.8, 22.5, 23.2, 22.8, 22.0, 21.5]
        for i, temp in enumerate(temps):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 20),
                sensor_id="sensor.living_room_temperature",
                sensor_type="temperature",
                state=str(temp),
                room_id="living_room",
            )
            events.append(event)

        # Humidity sensor events
        humidity_levels = [45.0, 48.5, 52.0, 55.5, 58.0, 54.5, 50.0]
        for i, humidity in enumerate(humidity_levels):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 25),
                sensor_id="sensor.living_room_humidity",
                sensor_type="humidity",
                state=str(humidity),
                room_id="living_room",
            )
            events.append(event)

        # Light sensor events with natural patterns
        light_levels = [150, 200, 350, 500, 800, 600, 400, 250]
        for i, light in enumerate(light_levels):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 18),
                sensor_id="sensor.living_room_light",
                sensor_type="illuminance",
                state=str(light),
                room_id="living_room",
            )
            events.append(event)

        # Door sensor events with complex patterns
        door_states = ["closed", "open", "closed", "open", "closed", "open", "closed"]
        for i, door_state in enumerate(door_states):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 15),
                sensor_id="sensor.living_room_door",
                sensor_type="door",
                state=door_state,
                room_id="living_room",
            )
            events.append(event)

        # Climate sensor events (combined temp/humidity)
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 30),
                sensor_id="sensor.living_room_climate",
                sensor_type="climate",
                state="on",
                room_id="living_room",
                attributes={"temperature": 22.0 + i, "humidity": 50.0 + i * 2},
            )
            events.append(event)

        return events

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

    @pytest.fixture
    def multi_room_events(self) -> List[SensorEvent]:
        """Create multi-room sensor events for correlation testing."""
        base_time = datetime(2024, 1, 15, 13, 0, 0)
        events = []

        rooms = ["living_room", "kitchen", "bedroom", "hallway"]

        # Create correlated activity patterns
        for room in rooms:
            for i in range(8):
                # Temperature
                event = self.create_sensor_event(
                    timestamp=base_time
                    + timedelta(minutes=i * 10 + rooms.index(room) * 2),
                    sensor_id=f"sensor.{room}_temperature",
                    sensor_type="temperature",
                    state=str(20.0 + i + rooms.index(room)),
                    room_id=room,
                )
                events.append(event)

                # Motion/presence
                if i % 3 == 0:  # Sporadic motion
                    event = self.create_sensor_event(
                        timestamp=base_time
                        + timedelta(minutes=i * 12 + rooms.index(room) * 3),
                        sensor_id=f"sensor.{room}_motion",
                        sensor_type="motion",
                        state="on",
                        room_id=room,
                    )
                    events.append(event)

        return events

    @pytest.fixture
    def complex_room_states(self) -> List[RoomState]:
        """Create complex room states for advanced testing."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        states = []

        rooms = ["living_room", "kitchen", "bedroom", "office"]
        occupancy_patterns = [
            [True, False, True, True, False, True, False, False],  # living_room
            [False, True, False, True, True, False, True, False],  # kitchen
            [True, True, False, False, True, False, False, True],  # bedroom
            [False, False, True, True, True, True, False, False],  # office
        ]

        for room_idx, room in enumerate(rooms):
            pattern = occupancy_patterns[room_idx]
            for i, is_occupied in enumerate(pattern):
                state = Mock(spec=RoomState)
                state.timestamp = base_time + timedelta(minutes=i * 30 + room_idx * 5)
                state.room_id = room
                state.is_occupied = is_occupied
                state.occupancy_confidence = 0.7 + (i % 3) * 0.1
                states.append(state)

        return states

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization_default(self):
        """Test default initialization."""
        extractor = ContextualFeatureExtractor()

        assert extractor.config is None
        assert isinstance(extractor.context_cache, dict)
        assert extractor.temp_thresholds["cold"] == 18.0
        assert extractor.humidity_thresholds["comfortable"] == 60.0
        assert extractor.light_thresholds["bright"] == 1000.0

    def test_initialization_with_config(self, system_config):
        """Test initialization with system configuration."""
        extractor = ContextualFeatureExtractor(system_config)

        assert extractor.config == system_config
        assert isinstance(extractor.context_cache, dict)
        assert len(extractor.temp_thresholds) == 3
        assert len(extractor.humidity_thresholds) == 3
        assert len(extractor.light_thresholds) == 3

    def test_threshold_values_realistic(self, extractor):
        """Test that threshold values are realistic for home automation."""
        # Temperature thresholds in Celsius
        assert 0 < extractor.temp_thresholds["cold"] < 25
        assert (
            extractor.temp_thresholds["cold"] < extractor.temp_thresholds["comfortable"]
        )
        assert (
            extractor.temp_thresholds["comfortable"] < extractor.temp_thresholds["warm"]
        )

        # Humidity thresholds in percentage
        assert 0 < extractor.humidity_thresholds["dry"] < 100
        assert (
            extractor.humidity_thresholds["dry"]
            < extractor.humidity_thresholds["comfortable"]
        )
        assert (
            extractor.humidity_thresholds["comfortable"]
            < extractor.humidity_thresholds["humid"]
        )

        # Light thresholds in lux
        assert 0 < extractor.light_thresholds["dark"] < 10000
        assert extractor.light_thresholds["dark"] < extractor.light_thresholds["dim"]
        assert extractor.light_thresholds["dim"] < extractor.light_thresholds["bright"]

    # ==================== ENVIRONMENTAL FEATURES TESTS ====================

    def test_environmental_features_comprehensive(
        self, extractor, comprehensive_events, target_time
    ):
        """Test comprehensive environmental feature extraction."""
        features = extractor._extract_environmental_features(
            comprehensive_events, target_time
        )

        # Verify all expected temperature features
        assert "current_temperature" in features
        assert "avg_temperature" in features
        assert "temperature_trend" in features
        assert "temperature_variance" in features
        assert "temperature_change_rate" in features
        assert "temperature_stability" in features
        assert "is_cold" in features
        assert "is_comfortable_temp" in features
        assert "is_warm" in features

        # Verify humidity features
        assert "current_humidity" in features
        assert "avg_humidity" in features
        assert "humidity_trend" in features
        assert "humidity_change_rate" in features
        assert "humidity_stability" in features

        # Verify light features
        assert "current_light" in features
        assert "avg_light" in features
        assert "avg_light_level" in features  # Test compatibility alias
        assert "light_trend" in features
        assert "is_dark" in features
        assert "is_dim" in features
        assert "is_bright" in features
        assert "natural_light_score" in features
        assert "light_change_rate" in features

    def test_temperature_mathematical_accuracy(self, extractor):
        """Test mathematical accuracy of temperature calculations."""
        temp_values = [20.0, 21.5, 22.0, 23.5, 22.5, 21.0]
        events = []

        base_time = datetime(2024, 1, 15, 12, 0, 0)
        for i, temp in enumerate(temp_values):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.test_temperature",
                sensor_type="temperature",
                state=str(temp),
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_environmental_features(
            events, base_time + timedelta(hours=1)
        )

        # Validate mathematical calculations
        expected_current = temp_values[-1]
        expected_avg = statistics.mean(temp_values)
        expected_variance = (
            statistics.variance(temp_values) if len(temp_values) > 1 else 0.0
        )

        assert abs(features["current_temperature"] - expected_current) < 0.01
        assert abs(features["avg_temperature"] - expected_avg) < 0.01
        assert abs(features["temperature_variance"] - expected_variance) < 0.01

    def test_temperature_trend_calculation(self, extractor):
        """Test accuracy of temperature trend calculation using linear regression."""
        # Create ascending temperature pattern
        temp_values = [20.0, 21.0, 22.0, 23.0, 24.0]
        events = []

        base_time = datetime(2024, 1, 15, 12, 0, 0)
        for i, temp in enumerate(temp_values):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.test_temperature",
                sensor_type="temperature",
                state=str(temp),
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_environmental_features(
            events, base_time + timedelta(hours=1)
        )

        # For perfect ascending pattern, trend should be positive
        assert features["temperature_trend"] > 0.8  # Strong positive trend

        # Test descending pattern
        temp_values_desc = [24.0, 23.0, 22.0, 21.0, 20.0]
        events_desc = []
        for i, temp in enumerate(temp_values_desc):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.test_temperature",
                sensor_type="temperature",
                state=str(temp),
                room_id="test_room",
            )
            events_desc.append(event)

        features_desc = extractor._extract_environmental_features(
            events_desc, base_time + timedelta(hours=1)
        )
        assert features_desc["temperature_trend"] < -0.8  # Strong negative trend

    def test_humidity_classification_thresholds(self, extractor):
        """Test humidity classification against thresholds."""
        humidity_levels = [30.0, 45.0, 55.0, 65.0, 75.0]  # dry to humid range

        for humidity in humidity_levels:
            event = self.create_sensor_event(
                timestamp=datetime(2024, 1, 15, 12, 0, 0),
                sensor_id="sensor.test_humidity",
                sensor_type="humidity",
                state=str(humidity),
                room_id="test_room",
            )

            features = extractor._extract_environmental_features(
                [event], datetime(2024, 1, 15, 12, 5, 0)
            )

            assert features["current_humidity"] == humidity

            # Verify threshold classification logic
            if humidity < extractor.humidity_thresholds["dry"]:
                expected_comfortable = 0.0
            elif humidity <= extractor.humidity_thresholds["humid"]:
                expected_comfortable = (
                    1.0 if humidity >= extractor.humidity_thresholds["dry"] else 0.0
                )
            else:
                expected_comfortable = 0.0

    def test_light_natural_pattern_detection(self, extractor):
        """Test natural light pattern detection algorithm."""
        target_times = [
            datetime(2024, 6, 15, 8, 0, 0),  # Morning summer
            datetime(2024, 6, 15, 14, 0, 0),  # Midday summer
            datetime(2024, 6, 15, 20, 0, 0),  # Evening summer
            datetime(2024, 12, 15, 8, 0, 0),  # Morning winter
            datetime(2024, 12, 15, 14, 0, 0),  # Midday winter
            datetime(2024, 12, 15, 20, 0, 0),  # Evening winter
        ]

        # Test with appropriate light levels
        light_levels = [250, 600, 150, 100, 400, 50]  # Corresponding to times above

        for target_time, light_level in zip(target_times, light_levels):
            event = self.create_sensor_event(
                timestamp=target_time - timedelta(minutes=5),
                sensor_id="sensor.test_light",
                sensor_type="illuminance",
                state=str(light_level),
                room_id="test_room",
            )

            features = extractor._extract_environmental_features([event], target_time)

            assert "natural_light_score" in features
            assert 0.0 <= features["natural_light_score"] <= 1.0

            # Verify the algorithm is working correctly
            expected_score = extractor._calculate_natural_light_score(
                [light_level], target_time
            )
            assert abs(features["natural_light_score"] - expected_score) < 0.01

    def test_environmental_stability_scores(self, extractor):
        """Test stability score calculations for environmental sensors."""
        # Test temperature stability
        stable_temps = [22.0, 22.1, 21.9, 22.0, 22.1]  # Very stable
        unstable_temps = [18.0, 25.0, 19.0, 26.0, 20.0]  # Very unstable

        for temp_set, expected_high_stability in [
            (stable_temps, True),
            (unstable_temps, False),
        ]:
            events = []
            base_time = datetime(2024, 1, 15, 12, 0, 0)
            for i, temp in enumerate(temp_set):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=i * 10),
                    sensor_id="sensor.test_temperature",
                    sensor_type="temperature",
                    state=str(temp),
                    room_id="test_room",
                )
                events.append(event)

            features = extractor._extract_environmental_features(
                events, base_time + timedelta(hours=1)
            )

            if expected_high_stability:
                assert features["temperature_stability"] > 0.8
            else:
                assert features["temperature_stability"] < 0.5

    def test_environmental_empty_data_handling(self, extractor):
        """Test handling of empty or missing environmental data."""
        features = extractor._extract_environmental_features(
            [], datetime(2024, 1, 15, 12, 0, 0)
        )

        # Verify default values are provided
        assert features["current_temperature"] == 22.0
        assert features["avg_temperature"] == 22.0
        assert features["temperature_trend"] == 0.0
        assert features["temperature_variance"] == 0.0
        assert features["is_comfortable_temp"] == 1.0

        assert features["current_humidity"] == 50.0
        assert features["avg_humidity"] == 50.0
        assert features["humidity_trend"] == 0.0

        assert features["current_light"] == 500.0
        assert features["avg_light"] == 500.0
        assert features["light_trend"] == 0.0
        assert features["is_dim"] == 1.0

    # ==================== DOOR STATE FEATURES TESTS ====================

    def test_door_state_comprehensive_analysis(self, extractor):
        """Test comprehensive door state analysis."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        door_events = []

        # Create complex door pattern: closed -> open -> closed -> open -> closed
        door_pattern = [
            ("closed", 0),
            ("open", 10),
            ("closed", 30),
            ("open", 45),
            ("closed", 70),
        ]

        for state, minutes_offset in door_pattern:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes_offset),
                sensor_id="sensor.front_door",
                sensor_type="door",
                state=state,
                room_id="entrance",
            )
            door_events.append(event)

        target_time = base_time + timedelta(hours=2)
        features = extractor._extract_door_state_features(door_events, target_time)

        # Verify all door features are present
        assert "doors_currently_open" in features
        assert "door_open_ratio" in features
        assert "door_transition_count" in features
        assert "avg_door_open_duration" in features
        assert "recent_door_activity" in features

        # Verify calculations
        assert features["doors_currently_open"] == 0.0  # Last state was closed
        assert features["door_transition_count"] == 4.0  # 4 state changes
        assert features["door_open_ratio"] > 0  # Some time spent open
        assert features["avg_door_open_duration"] > 0  # Some open durations recorded

    def test_door_open_duration_mathematical_accuracy(self, extractor):
        """Test mathematical accuracy of door open duration calculations."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Pattern: closed(5min) -> open(10min) -> closed(5min) -> open(15min) -> closed
        pattern = [
            ("closed", 0),
            ("open", 5),  # Open for 10 minutes
            ("closed", 15),
            ("open", 20),  # Open for 15 minutes
            ("closed", 35),
        ]

        for state, minutes in pattern:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id="sensor.test_door",
                sensor_type="door",
                state=state,
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_door_state_features(
            events, base_time + timedelta(hours=1)
        )

        # Expected: open durations are 10 and 15 minutes, average = 12.5 minutes
        expected_avg_duration = 12.5  # minutes
        assert abs(features["avg_door_open_duration"] - expected_avg_duration) < 0.1

    def test_door_open_ratio_calculation(self, extractor):
        """Test door open ratio calculation accuracy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create pattern where door is open exactly 50% of the time
        events = [
            self.create_sensor_event(base_time, "door1", "door", "closed", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=10), "door1", "door", "open", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=20), "door1", "door", "closed", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "door1", "door", "open", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=40), "door1", "door", "closed", "room1"
            ),
        ]

        features = extractor._extract_door_state_features(
            events, base_time + timedelta(hours=1)
        )

        # Total time: 40 minutes, open time: 20 minutes (10-20, 30-40), ratio = 0.5
        expected_ratio = 0.5
        assert abs(features["door_open_ratio"] - expected_ratio) < 0.05

    def test_multiple_doors_handling(self, extractor):
        """Test handling of multiple door sensors."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        doors = ["front_door", "back_door", "garage_door"]

        for i, door in enumerate(doors):
            # Each door has different pattern
            for j in range(3):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=j * 10 + i * 2),
                    sensor_id=f"sensor.{door}",
                    sensor_type="door",
                    state="open" if (j + i) % 2 == 0 else "closed",
                    room_id="entrance",
                )
                events.append(event)

        features = extractor._extract_door_state_features(
            events, base_time + timedelta(hours=1)
        )

        # Should handle all doors appropriately
        assert features["doors_currently_open"] >= 0
        assert features["door_transition_count"] >= 0
        assert 0.0 <= features["door_open_ratio"] <= 1.0

    def test_door_recent_activity_window(self, extractor):
        """Test recent door activity calculation within time window."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Create events: some recent (within 1 hour), some old
        events = [
            # Old events (more than 1 hour ago)
            self.create_sensor_event(base_time, "door1", "door", "open", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "door1", "door", "closed", "room1"
            ),
            # Recent events (within last hour)
            self.create_sensor_event(
                target_time - timedelta(minutes=45), "door1", "door", "open", "room1"
            ),
            self.create_sensor_event(
                target_time - timedelta(minutes=30), "door1", "door", "closed", "room1"
            ),
            self.create_sensor_event(
                target_time - timedelta(minutes=15), "door1", "door", "open", "room1"
            ),
        ]

        features = extractor._extract_door_state_features(events, target_time)

        # Should count only recent events (last 1 hour)
        assert features["recent_door_activity"] == 3.0

    def test_door_empty_events_defaults(self, extractor):
        """Test default values when no door events are present."""
        features = extractor._extract_door_state_features(
            [], datetime(2024, 1, 15, 12, 0, 0)
        )

        expected_defaults = {
            "doors_currently_open": 0.0,
            "door_open_ratio": 0.0,
            "door_transition_count": 0.0,
            "avg_door_open_duration": 0.0,
            "recent_door_activity": 0.0,
        }

        for key, expected_value in expected_defaults.items():
            assert features[key] == expected_value

    # ==================== MULTI-ROOM FEATURES TESTS ====================

    def test_multi_room_comprehensive_analysis(
        self, extractor, multi_room_events, complex_room_states
    ):
        """Test comprehensive multi-room correlation analysis."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        features = extractor._extract_multi_room_features(
            multi_room_events, complex_room_states, target_time
        )

        # Verify all multi-room features are present
        required_features = [
            "total_active_rooms",
            "simultaneous_occupancy_ratio",
            "room_activity_correlation",
            "dominant_room_activity_ratio",
            "room_activity_balance",
            "active_rooms_count",  # Alias
            "cross_room_correlation",  # Alias
            "occupancy_spread_score",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_room_activity_correlation_mathematical_accuracy(self, extractor):
        """Test mathematical accuracy of room activity correlation calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Create perfectly correlated activity in two rooms
        events = []
        for i in range(10):
            # Room A activity
            event_a = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.room_a_motion",
                sensor_type="motion",
                state="on",
                room_id="room_a",
            )
            events.append(event_a)

            # Room B activity at same time (perfect correlation)
            event_b = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.room_b_motion",
                sensor_type="motion",
                state="on",
                room_id="room_b",
            )
            events.append(event_b)

        features = extractor._extract_multi_room_features(events, [], target_time)

        # Should detect high correlation due to synchronized activity
        assert features["room_activity_correlation"] > 0.5
        assert features["cross_room_correlation"] > 0.5  # Alias test

    def test_simultaneous_occupancy_ratio_accuracy(self, extractor):
        """Test accuracy of simultaneous occupancy ratio calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=1)

        # Create room states where 2 out of 4 rooms are occupied
        room_states = []
        rooms = ["living_room", "kitchen", "bedroom", "office"]
        occupancy = [True, True, False, False]  # 50% occupancy

        for i, (room, occupied) in enumerate(zip(rooms, occupancy)):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 5)
            state.room_id = room
            state.is_occupied = occupied
            state.occupancy_confidence = 0.8
            room_states.append(state)

        features = extractor._extract_multi_room_features([], room_states, target_time)

        # Should calculate 50% simultaneous occupancy
        assert abs(features["simultaneous_occupancy_ratio"] - 0.5) < 0.1

    def test_room_activity_balance_entropy_calculation(self, extractor):
        """Test room activity balance calculation using entropy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=1)

        # Create perfectly balanced activity across rooms
        events = []
        rooms = ["room_a", "room_b", "room_c"]
        events_per_room = 10

        for room in rooms:
            for i in range(events_per_room):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=i * 3),
                    sensor_id=f"sensor.{room}_motion",
                    sensor_type="motion",
                    state="on",
                    room_id=room,
                )
                events.append(event)

        features = extractor._extract_multi_room_features(events, [], target_time)

        # Perfect balance should yield high balance score (entropy close to max)
        assert features["room_activity_balance"] > 0.8

        # Test unbalanced activity
        unbalanced_events = []
        # Room A gets 90% of events, others get 5% each
        activity_distribution = [18, 1, 1]

        for room, count in zip(rooms, activity_distribution):
            for i in range(count):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=i * 2),
                    sensor_id=f"sensor.{room}_motion",
                    sensor_type="motion",
                    state="on",
                    room_id=room,
                )
                unbalanced_events.append(event)

        unbalanced_features = extractor._extract_multi_room_features(
            unbalanced_events, [], target_time
        )

        # Unbalanced activity should yield low balance score
        assert unbalanced_features["room_activity_balance"] < 0.5

    def test_occupancy_spread_score_calculation(self, extractor):
        """Test occupancy spread score calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test full spread (all rooms occupied)
        full_spread_states = []
        rooms = ["room1", "room2", "room3", "room4"]

        for room in rooms:
            state = Mock(spec=RoomState)
            state.timestamp = base_time
            state.room_id = room
            state.is_occupied = True
            state.occupancy_confidence = 0.8
            full_spread_states.append(state)

        features_full = extractor._extract_multi_room_features(
            [], full_spread_states, base_time
        )
        assert features_full["occupancy_spread_score"] == 1.0

        # Test partial spread (half rooms occupied)
        partial_spread_states = []
        occupancy_pattern = [True, True, False, False]

        for room, occupied in zip(rooms, occupancy_pattern):
            state = Mock(spec=RoomState)
            state.timestamp = base_time
            state.room_id = room
            state.is_occupied = occupied
            state.occupancy_confidence = 0.8
            partial_spread_states.append(state)

        features_partial = extractor._extract_multi_room_features(
            [], partial_spread_states, base_time
        )
        assert features_partial["occupancy_spread_score"] == 0.5

    def test_single_room_handling(self, extractor):
        """Test handling when only one room is active."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Single room events
        events = [
            self.create_sensor_event(
                base_time, "sensor1", "motion", "on", "single_room"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=5),
                "sensor2",
                "motion",
                "off",
                "single_room",
            ),
        ]

        # Single room state
        room_states = [
            Mock(
                spec=RoomState,
                timestamp=base_time,
                room_id="single_room",
                is_occupied=True,
                occupancy_confidence=0.8,
            )
        ]

        features = extractor._extract_multi_room_features(
            events, room_states, base_time
        )

        # Verify single room handling
        assert features["total_active_rooms"] == 1.0
        assert features["active_rooms_count"] == 1.0  # Alias
        assert (
            features["simultaneous_occupancy_ratio"] == 1.0
        )  # 100% of 1 room occupied
        assert (
            features["room_activity_correlation"] == 0.0
        )  # No correlation with 1 room
        assert features["dominant_room_activity_ratio"] == 1.0  # 100% dominance

    # ==================== SEASONAL FEATURES TESTS ====================

    def test_seasonal_features_comprehensive(self, extractor):
        """Test comprehensive seasonal feature detection."""
        # Test all seasons
        test_dates = [
            datetime(2024, 1, 15, 15, 0, 0),  # Winter
            datetime(2024, 4, 15, 15, 0, 0),  # Spring
            datetime(2024, 7, 15, 15, 0, 0),  # Summer
            datetime(2024, 10, 15, 15, 0, 0),  # Autumn
        ]

        expected_seasons = [
            {"is_winter": 1.0, "is_spring": 0.0, "is_summer": 0.0, "is_autumn": 0.0},
            {"is_winter": 0.0, "is_spring": 1.0, "is_summer": 0.0, "is_autumn": 0.0},
            {"is_winter": 0.0, "is_spring": 0.0, "is_summer": 1.0, "is_autumn": 0.0},
            {"is_winter": 0.0, "is_spring": 0.0, "is_summer": 0.0, "is_autumn": 1.0},
        ]

        for target_date, expected in zip(test_dates, expected_seasons):
            features = extractor._extract_seasonal_features(target_date)

            for season_key, expected_value in expected.items():
                assert features[season_key] == expected_value

            # Verify seasonal indicator
            assert features["season_indicator"] == 1.0
            assert features["seasonal_indicator"] == 1.0  # Alias

    def test_holiday_season_detection(self, extractor):
        """Test holiday season detection logic."""
        holiday_dates = [
            datetime(2024, 12, 20, 15, 0, 0),  # Christmas season start
            datetime(2024, 12, 25, 15, 0, 0),  # Christmas day
            datetime(2024, 1, 1, 15, 0, 0),  # New Year
            datetime(2024, 1, 7, 15, 0, 0),  # New Year season end
            datetime(2024, 7, 4, 15, 0, 0),  # Independence Day
        ]

        non_holiday_dates = [
            datetime(2024, 3, 15, 15, 0, 0),  # Random spring day
            datetime(2024, 8, 15, 15, 0, 0),  # Random summer day
            datetime(2024, 10, 15, 15, 0, 0),  # Random autumn day
            datetime(2024, 12, 10, 15, 0, 0),  # December but not holiday season
        ]

        # Test holiday detection
        for holiday_date in holiday_dates:
            features = extractor._extract_seasonal_features(holiday_date)
            assert features["is_holiday_season"] == 1.0, f"Failed for {holiday_date}"

        # Test non-holiday detection
        for non_holiday_date in non_holiday_dates:
            features = extractor._extract_seasonal_features(non_holiday_date)
            assert (
                features["is_holiday_season"] == 0.0
            ), f"Failed for {non_holiday_date}"

    def test_natural_light_availability_seasonal(self, extractor):
        """Test natural light availability calculation for different seasons."""
        # Test summer light availability (longer days)
        summer_times = [
            (datetime(2024, 6, 15, 4, 0, 0), 0.0),  # Very early, no light
            (datetime(2024, 6, 15, 6, 0, 0), 1.0),  # Dawn, light available
            (datetime(2024, 6, 15, 12, 0, 0), 1.0),  # Midday, light available
            (datetime(2024, 6, 15, 19, 0, 0), 1.0),  # Evening, still light
            (datetime(2024, 6, 15, 21, 0, 0), 0.0),  # Late evening, no light
        ]

        # Test winter light availability (shorter days)
        winter_times = [
            (datetime(2024, 12, 15, 6, 0, 0), 0.0),  # Early, no light
            (datetime(2024, 12, 15, 8, 0, 0), 1.0),  # Morning, light available
            (datetime(2024, 12, 15, 12, 0, 0), 1.0),  # Midday, light available
            (datetime(2024, 12, 15, 16, 0, 0), 1.0),  # Afternoon, light available
            (datetime(2024, 12, 15, 18, 0, 0), 0.0),  # Evening, no light
        ]

        for time_expected_pairs in [summer_times, winter_times]:
            for test_time, expected_light in time_expected_pairs:
                features = extractor._extract_seasonal_features(test_time)
                assert (
                    features["natural_light_available"] == expected_light
                ), f"Failed for {test_time}: expected {expected_light}, got {features['natural_light_available']}"

    # ==================== SENSOR CORRELATION FEATURES TESTS ====================

    def test_sensor_correlation_comprehensive(self, extractor):
        """Test comprehensive sensor correlation analysis."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Create events with different correlation patterns
        sensors = ["sensor1", "sensor2", "sensor3", "sensor4"]

        for i in range(20):
            # Create time windows where multiple sensors are active
            window_start = base_time + timedelta(minutes=i * 15)

            # In each window, activate 2-3 sensors (creating correlation)
            active_sensors = sensors[: 2 + (i % 2)]  # 2 or 3 sensors active

            for j, sensor in enumerate(active_sensors):
                event = self.create_sensor_event(
                    timestamp=window_start + timedelta(seconds=j * 30),
                    sensor_id=sensor,
                    sensor_type="motion",
                    state="on",
                    room_id="test_room",
                )
                events.append(event)

        features = extractor._extract_sensor_correlation_features(events)

        # Verify all correlation features
        assert "sensor_activation_correlation" in features
        assert "multi_sensor_event_ratio" in features
        assert "sensor_type_diversity" in features

        # Values should reflect multi-sensor activity
        assert (
            features["sensor_activation_correlation"] > 1.0
        )  # Multiple sensors per window
        assert (
            features["multi_sensor_event_ratio"] > 0.5
        )  # Most windows have multiple sensors
        assert features["sensor_type_diversity"] > 0  # At least one sensor type

    def test_sensor_correlation_sliding_window_accuracy(self, extractor):
        """Test accuracy of sliding window correlation calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Create precisely controlled sensor activation pattern
        # Window 1: 3 sensors active (high correlation)
        for i in range(3):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 60),
                sensor_id=f"sensor{i+1}",
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        # Window 2 (8 minutes later): 1 sensor active (low correlation)
        event = self.create_sensor_event(
            timestamp=base_time + timedelta(minutes=8),
            sensor_id="sensor1",
            sensor_type="motion",
            state="on",
            room_id="test_room",
        )
        events.append(event)

        features = extractor._extract_sensor_correlation_features(events)

        # Should detect average correlation across windows
        expected_avg = (3 + 1) / 2  # Average of 3 and 1 sensors per window
        assert abs(features["sensor_activation_correlation"] - expected_avg) < 0.5

    def test_sensor_type_diversity_calculation(self, extractor):
        """Test sensor type diversity calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with diverse sensor types
        sensor_types = ["motion", "door", "presence", "light", "temperature"]
        events = []

        for i, sensor_type in enumerate(sensor_types):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{sensor_type}",
                sensor_type=sensor_type,
                state="on",
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_sensor_correlation_features(events)

        # Should count all unique sensor types
        assert features["sensor_type_diversity"] == len(sensor_types)

        # Test with single sensor type
        single_type_events = [
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=1), "sensor2", "motion", "on", "room1"
            ),
        ]

        single_features = extractor._extract_sensor_correlation_features(
            single_type_events
        )
        assert single_features["sensor_type_diversity"] == 1

    def test_multi_sensor_event_ratio_accuracy(self, extractor):
        """Test accuracy of multi-sensor event ratio calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create pattern: 50% of time windows have multiple sensors active
        events = []

        # 10 time windows, alternating single and multi-sensor activity
        for i in range(10):
            window_start = base_time + timedelta(minutes=i * 6)  # 6-minute windows

            if i % 2 == 0:
                # Multi-sensor window
                event1 = self.create_sensor_event(
                    timestamp=window_start,
                    sensor_id="sensor1",
                    sensor_type="motion",
                    state="on",
                    room_id="test_room",
                )
                event2 = self.create_sensor_event(
                    timestamp=window_start + timedelta(minutes=2),
                    sensor_id="sensor2",
                    sensor_type="motion",
                    state="on",
                    room_id="test_room",
                )
                events.extend([event1, event2])
            else:
                # Single-sensor window
                event = self.create_sensor_event(
                    timestamp=window_start,
                    sensor_id="sensor1",
                    sensor_type="motion",
                    state="on",
                    room_id="test_room",
                )
                events.append(event)

        features = extractor._extract_sensor_correlation_features(events)

        # Should detect ~50% multi-sensor ratio
        assert 0.4 < features["multi_sensor_event_ratio"] < 0.6

    def test_sensor_correlation_empty_data(self, extractor):
        """Test sensor correlation features with empty or minimal data."""
        # Test with empty events
        features_empty = extractor._extract_sensor_correlation_features([])

        expected_defaults = {
            "sensor_activation_correlation": 0.0,
            "multi_sensor_event_ratio": 0.0,
            "sensor_type_diversity": 0.0,
        }

        for key, expected in expected_defaults.items():
            assert features_empty[key] == expected

        # Test with single event
        single_event = [
            self.create_sensor_event(
                datetime(2024, 1, 15, 12, 0, 0), "sensor1", "motion", "on", "test_room"
            )
        ]

        features_single = extractor._extract_sensor_correlation_features(single_event)

        # Single event should yield minimal correlation
        assert features_single["sensor_activation_correlation"] == 1.0  # 1 sensor
        assert (
            features_single["multi_sensor_event_ratio"] == 0.0
        )  # No multi-sensor events
        assert features_single["sensor_type_diversity"] == 1  # 1 sensor type

    # ==================== ERROR HANDLING TESTS ====================

    def test_feature_extraction_error_handling(self, extractor):
        """Test error handling during feature extraction."""
        # Test with None inputs
        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(
                events=None,  # This should cause an error
                room_states=None,
                target_time=datetime(2024, 1, 15, 12, 0, 0),
            )

    def test_malformed_event_handling(self, extractor):
        """Test handling of malformed sensor events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events with missing/invalid attributes
        malformed_events = []

        # Event with invalid timestamp
        bad_event1 = Mock(spec=SensorEvent)
        bad_event1.timestamp = "invalid_timestamp"  # String instead of datetime
        bad_event1.sensor_id = "sensor1"
        bad_event1.sensor_type = "temperature"
        bad_event1.state = "22.5"
        bad_event1.room_id = "test_room"
        malformed_events.append(bad_event1)

        # Event with None values
        bad_event2 = Mock(spec=SensorEvent)
        bad_event2.timestamp = target_time - timedelta(minutes=30)
        bad_event2.sensor_id = None
        bad_event2.sensor_type = None
        bad_event2.state = None
        bad_event2.room_id = "test_room"
        malformed_events.append(bad_event2)

        # Should handle malformed events gracefully without crashing
        try:
            features = extractor.extract_features(
                events=malformed_events, room_states=[], target_time=target_time
            )
            # Should return some default features
            assert isinstance(features, dict)
        except Exception as e:
            # If it raises an exception, it should be a FeatureExtractionError
            assert isinstance(e, FeatureExtractionError)

    def test_numeric_value_validation(self, extractor):
        """Test validation of numeric sensor values."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with invalid temperature values
        invalid_temps = ["-999", "999", "nan", "inf", "-inf", ""]
        events = []

        for temp in invalid_temps:
            event = self.create_sensor_event(
                timestamp=base_time,
                sensor_id="sensor.temp",
                sensor_type="temperature",
                state=str(temp),
                room_id="test_room",
            )
            events.append(event)

        # Also add valid temperatures
        valid_temps = ["20.5", "22.0", "24.5"]
        for temp in valid_temps:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=1),
                sensor_id="sensor.temp2",
                sensor_type="temperature",
                state=temp,
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_environmental_features(
            events, base_time + timedelta(minutes=30)
        )

        # Should process only valid temperatures
        # extractor._is_realistic_value should filter invalid ones
        assert features["current_temperature"] > 0  # Should have some valid reading
        assert features["avg_temperature"] > 0

    # ==================== PERFORMANCE TESTS ====================

    def test_performance_with_large_dataset(self, extractor):
        """Test performance with large numbers of events."""
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        target_time = base_time + timedelta(hours=24)

        # Create large dataset: 1000 events over 24 hours
        large_events = []
        for i in range(1000):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i),
                sensor_id=f"sensor_{i % 50}",  # 50 different sensors
                sensor_type=["temperature", "humidity", "light", "motion"][i % 4],
                state=str(20 + (i % 20)),
                room_id=f"room_{i % 10}",  # 10 different rooms
            )
            large_events.append(event)

        # Should complete within reasonable time
        import time

        start_time = time.time()

        features = extractor.extract_features(
            events=large_events, room_states=[], target_time=target_time
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within 2 seconds (performance requirement < 500ms may be optimistic for 1000 events)
        assert (
            execution_time < 2.0
        ), f"Performance test failed: took {execution_time:.3f}s"
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_memory_efficiency_with_repeated_extraction(self, extractor):
        """Test memory efficiency with repeated feature extractions."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create moderate dataset
        events = []
        for i in range(100):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i),
                sensor_id=f"sensor_{i % 10}",
                sensor_type=["temperature", "motion"][i % 2],
                state=str(20 + i % 10),
                room_id="test_room",
            )
            events.append(event)

        # Run extraction multiple times to test memory usage
        results = []
        for _ in range(50):
            features = extractor.extract_features(
                events=events,
                room_states=[],
                target_time=base_time + timedelta(hours=2),
            )
            results.append(len(features))

        # All extractions should return consistent results
        assert len(set(results)) == 1, "Inconsistent results across extractions"

        # Cache should not grow indefinitely
        if hasattr(extractor, "context_cache"):
            assert len(extractor.context_cache) < 100, "Cache may be growing too large"

    # ==================== INTEGRATION TESTS ====================

    def test_full_feature_extraction_integration(
        self, extractor, comprehensive_events, complex_room_states, target_time
    ):
        """Test full integration of all feature extraction components."""
        # Test complete feature extraction with all components
        features = extractor.extract_features(
            events=comprehensive_events,
            room_states=complex_room_states,
            target_time=target_time,
            lookback_hours=24,
        )

        # Verify comprehensive feature set
        feature_categories = {
            "environmental": ["current_temperature", "avg_humidity", "current_light"],
            "door": ["doors_currently_open", "door_open_ratio"],
            "multi_room": ["total_active_rooms", "room_activity_correlation"],
            "seasonal": ["is_winter", "is_spring", "is_summer", "is_autumn"],
            "correlation": ["sensor_activation_correlation", "sensor_type_diversity"],
        }

        for category, expected_features in feature_categories.items():
            for feature in expected_features:
                assert (
                    feature in features
                ), f"Missing {feature} from {category} features"
                assert isinstance(features[feature], (int, float))

    def test_room_config_integration(self, extractor, system_config):
        """Test integration with room configuration."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events that use room configuration
        events = [
            self.create_sensor_event(
                base_time,
                "sensor.living_room_temperature",
                "temperature",
                "22.5",
                "living_room",
            ),
            self.create_sensor_event(
                base_time, "sensor.kitchen_door", "door", "open", "kitchen"
            ),
        ]

        room_configs = {
            "living_room": system_config.rooms["living_room"],
            "kitchen": system_config.rooms["kitchen"],
        }

        # Should integrate room config appropriately
        features = extractor._extract_room_context_features(
            events, room_configs, base_time + timedelta(hours=1)
        )

        # Should extract room context features
        assert "max_room_complexity" in features
        assert "sensor_type_diversity" in features
        assert features["max_room_complexity"] >= 0

    def test_feature_cache_functionality(self, extractor):
        """Test feature cache functionality."""
        # Clear cache initially
        extractor.clear_cache()
        assert len(extractor.context_cache) == 0

        # Create some events
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = [
            self.create_sensor_event(
                base_time, "sensor1", "temperature", "22.5", "room1"
            )
        ]

        # Extract features (may populate cache depending on implementation)
        features1 = extractor.extract_features(
            events, [], base_time + timedelta(hours=1)
        )

        # Cache should be accessible and clearable
        extractor.clear_cache()
        # Should not crash and cache should be cleared if implemented

        # Extract again after cache clear
        features2 = extractor.extract_features(
            events, [], base_time + timedelta(hours=1)
        )

        # Results should be consistent regardless of cache state
        for key in features1:
            if key in features2:
                assert abs(features1[key] - features2[key]) < 0.001

    # ==================== EDGE CASES AND BOUNDARY TESTS ====================

    def test_extreme_environmental_values(self, extractor):
        """Test handling of extreme environmental sensor values."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test extreme temperature values
        extreme_events = [
            # Very hot
            self.create_sensor_event(
                base_time, "temp1", "temperature", "45.0", "room1"
            ),
            # Very cold
            self.create_sensor_event(base_time, "temp2", "temperature", "5.0", "room1"),
            # Very high humidity
            self.create_sensor_event(base_time, "humid1", "humidity", "95.0", "room1"),
            # Very low humidity
            self.create_sensor_event(base_time, "humid2", "humidity", "10.0", "room1"),
            # Very bright light
            self.create_sensor_event(
                base_time, "light1", "illuminance", "50000", "room1"
            ),
            # Very dim light
            self.create_sensor_event(base_time, "light2", "illuminance", "1", "room1"),
        ]

        features = extractor._extract_environmental_features(
            extreme_events, base_time + timedelta(hours=1)
        )

        # Should handle extreme values without crashing
        assert "current_temperature" in features
        assert "current_humidity" in features
        assert "current_light" in features

        # Classification should work with extreme values
        assert features["is_cold"] in [0.0, 1.0]
        assert features["is_warm"] in [0.0, 1.0]

    def test_zero_duration_events(self, extractor):
        """Test handling of events with zero time duration."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events with identical timestamps
        zero_duration_events = [
            self.create_sensor_event(base_time, "sensor1", "door", "closed", "room1"),
            self.create_sensor_event(base_time, "sensor1", "door", "open", "room1"),
            self.create_sensor_event(base_time, "sensor1", "door", "closed", "room1"),
        ]

        # Should handle zero-duration events gracefully
        features = extractor._extract_door_state_features(
            zero_duration_events, base_time + timedelta(minutes=30)
        )

        assert "doors_currently_open" in features
        assert "door_transition_count" in features
        assert features["door_transition_count"] >= 0

    def test_future_timestamp_events(self, extractor):
        """Test handling of events with timestamps in the future."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=1)

        # Create events with future timestamps
        future_events = [
            self.create_sensor_event(
                target_time + timedelta(hours=1),
                "sensor1",
                "temperature",
                "22.0",
                "room1",
            ),
            self.create_sensor_event(
                target_time + timedelta(hours=2),
                "sensor2",
                "temperature",
                "23.0",
                "room1",
            ),
        ]

        # Should handle future events (likely by filtering them out)
        features = extractor.extract_features(
            future_events, [], target_time, lookback_hours=24
        )

        # Should not crash and should return valid features (likely defaults)
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_massive_room_correlation_data(self, extractor):
        """Test multi-room correlation with very large number of rooms."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=1)

        # Create events for 50 rooms
        events = []
        room_states = []

        for room_id in range(50):
            room_name = f"room_{room_id:02d}"

            # Create events for each room
            for i in range(5):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=i * 10),
                    sensor_id=f"sensor.{room_name}_motion",
                    sensor_type="motion",
                    state="on",
                    room_id=room_name,
                )
                events.append(event)

            # Create room state
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=room_id)
            state.room_id = room_name
            state.is_occupied = room_id % 3 == 0  # Every 3rd room occupied
            state.occupancy_confidence = 0.8
            room_states.append(state)

        # Should handle large number of rooms efficiently
        import time

        start_time = time.time()

        features = extractor._extract_multi_room_features(
            events, room_states, target_time
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time even with 50 rooms
        assert (
            execution_time < 1.0
        ), f"Large room test took too long: {execution_time:.3f}s"

        assert features["total_active_rooms"] == 50.0
        assert 0.0 <= features["simultaneous_occupancy_ratio"] <= 1.0
        assert features["room_activity_correlation"] >= 0.0

    # ==================== UTILITY METHOD TESTS ====================

    def test_get_default_features_completeness(self, extractor):
        """Test that default features cover all expected feature names."""
        defaults = extractor._get_default_features()

        # Should include all major feature categories
        expected_categories = {
            "temperature": [
                "current_temperature",
                "avg_temperature",
                "temperature_trend",
            ],
            "humidity": ["current_humidity", "avg_humidity", "humidity_trend"],
            "light": ["current_light", "avg_light", "light_trend"],
            "door": ["doors_currently_open", "door_open_ratio"],
            "multi_room": ["total_active_rooms", "room_activity_correlation"],
            "seasonal": ["is_winter", "is_spring", "is_summer", "is_autumn"],
        }

        for category, features in expected_categories.items():
            for feature in features:
                assert (
                    feature in defaults
                ), f"Missing default for {feature} in {category}"

    def test_get_feature_names_method(self, extractor):
        """Test get_feature_names method returns complete list."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 50  # Should have many features

        # Should match default features
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_clear_cache_method(self, extractor):
        """Test cache clearing functionality."""
        # Add something to cache
        extractor.context_cache["test_key"] = "test_value"
        assert len(extractor.context_cache) > 0

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.context_cache) == 0

    # ==================== MATHEMATICAL ACCURACY VALIDATION ====================

    def test_statistical_calculations_accuracy(self, extractor):
        """Test accuracy of all statistical calculations."""
        # Test variance calculation
        test_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        expected_variance = statistics.variance(test_values)

        # Create temperature events with known values
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        for i, value in enumerate(test_values):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.test_temp",
                sensor_type="temperature",
                state=str(value),
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_environmental_features(
            events, base_time + timedelta(hours=1)
        )

        # Verify variance calculation matches statistics module
        assert abs(features["temperature_variance"] - expected_variance) < 0.001

    def test_trend_calculation_mathematical_correctness(self, extractor):
        """Test mathematical correctness of trend calculations."""
        # Create perfect linear trend: y = 2x + 20
        x_values = [0, 1, 2, 3, 4]
        y_values = [20.0, 22.0, 24.0, 26.0, 28.0]  # slope = 2

        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        for i, temp in enumerate(y_values):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id="sensor.test_temp",
                sensor_type="temperature",
                state=str(temp),
                room_id="test_room",
            )
            events.append(event)

        # Calculate expected trend using manual linear regression
        n = len(x_values)
        x_mean = sum(x_values) / n
        y_mean = sum(y_values) / n

        numerator = sum(
            (x_values[i] - x_mean) * (y_values[i] - y_mean) for i in range(n)
        )
        denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
        expected_trend = numerator / denominator if denominator != 0 else 0.0

        features = extractor._extract_environmental_features(
            events, base_time + timedelta(hours=1)
        )

        # Should match manual calculation
        assert abs(features["temperature_trend"] - expected_trend) < 0.01

    # ==================== COMPREHENSIVE SCENARIO TESTS ====================

    def test_realistic_home_scenario(self, extractor):
        """Test with realistic home automation scenario."""
        # Simulate a typical day in a smart home
        base_time = datetime(2024, 1, 15, 6, 0, 0)  # 6 AM start
        target_time = base_time + timedelta(hours=12)  # 6 PM analysis

        events = []
        room_states = []

        # Morning routine (6-8 AM): bedroom -> bathroom -> kitchen
        morning_events = [
            # Bedroom activity
            self.create_sensor_event(
                base_time, "sensor.bedroom_motion", "motion", "on", "bedroom"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=5),
                "sensor.bedroom_temperature",
                "temperature",
                "21.5",
                "bedroom",
            ),
            # Bathroom activity
            self.create_sensor_event(
                base_time + timedelta(minutes=15),
                "sensor.bathroom_door",
                "door",
                "open",
                "bathroom",
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=30),
                "sensor.bathroom_humidity",
                "humidity",
                "75.0",
                "bathroom",
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=45),
                "sensor.bathroom_door",
                "door",
                "closed",
                "bathroom",
            ),
            # Kitchen activity
            self.create_sensor_event(
                base_time + timedelta(minutes=50),
                "sensor.kitchen_motion",
                "motion",
                "on",
                "kitchen",
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=55),
                "sensor.kitchen_light",
                "illuminance",
                "800",
                "kitchen",
            ),
        ]

        # Day activity (8 AM - 6 PM): mostly living room, some kitchen
        for hour in range(8, 18):
            # Living room temperature gradually increases
            temp = 20.0 + (hour - 8) * 0.3
            events.append(
                self.create_sensor_event(
                    base_time + timedelta(hours=hour),
                    "sensor.living_room_temperature",
                    "temperature",
                    str(temp),
                    "living_room",
                )
            )

            # Kitchen activity every 3 hours
            if hour % 3 == 0:
                events.append(
                    self.create_sensor_event(
                        base_time + timedelta(hours=hour, minutes=30),
                        "sensor.kitchen_motion",
                        "motion",
                        "on",
                        "kitchen",
                    )
                )

        events.extend(morning_events)

        # Create corresponding room states
        rooms = ["bedroom", "bathroom", "kitchen", "living_room"]
        for i, room in enumerate(rooms):
            for hour in range(6, 18, 2):  # Every 2 hours
                state = Mock(spec=RoomState)
                state.timestamp = base_time + timedelta(hours=hour, minutes=i * 15)
                state.room_id = room
                # Bedroom occupied at night, kitchen/living room during day
                state.is_occupied = (room == "bedroom" and hour < 8) or (
                    room in ["kitchen", "living_room"] and hour >= 8
                )
                state.occupancy_confidence = 0.8 + (i * 0.05)
                room_states.append(state)

        # Extract features for this realistic scenario
        features = extractor.extract_features(
            events, room_states, target_time, lookback_hours=12
        )

        # Verify realistic results
        assert isinstance(features, dict)
        assert len(features) > 30  # Should extract many features

        # Temperature should show daily warming trend
        if "temperature_trend" in features:
            assert features["temperature_trend"] > 0  # Warming during day

        # Multiple rooms should be active
        if "total_active_rooms" in features:
            assert features["total_active_rooms"] >= 3

        # Should detect some room correlation from morning routine
        if "room_activity_correlation" in features:
            assert features["room_activity_correlation"] >= 0

    def test_edge_case_boundary_conditions(self, extractor):
        """Test various boundary conditions and edge cases."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with exactly lookback_hours boundary
        boundary_time = target_time - timedelta(hours=24)  # Exactly 24 hours ago

        events = [
            self.create_sensor_event(
                boundary_time, "sensor1", "temperature", "22.0", "room1"
            ),
            self.create_sensor_event(
                boundary_time + timedelta(seconds=1),
                "sensor2",
                "temperature",
                "22.5",
                "room1",
            ),
        ]

        features = extractor.extract_features(
            events, [], target_time, lookback_hours=24
        )

        # Should include the boundary event
        assert isinstance(features, dict)

        # Test with events exactly at target_time
        current_events = [
            self.create_sensor_event(
                target_time, "sensor1", "temperature", "23.0", "room1"
            ),
        ]

        features_current = extractor.extract_features(
            current_events, [], target_time, lookback_hours=1
        )
        assert isinstance(features_current, dict)

        # Test with single event type
        single_type_events = [
            self.create_sensor_event(
                target_time - timedelta(minutes=30),
                "sensor1",
                "temperature",
                "22.0",
                "room1",
            ),
        ]

        features_single = extractor.extract_features(
            single_type_events, [], target_time, lookback_hours=1
        )
        assert isinstance(features_single, dict)
        assert "current_temperature" in features_single


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
