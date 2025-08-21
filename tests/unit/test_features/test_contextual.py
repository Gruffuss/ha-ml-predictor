"""
Unit tests for contextual feature extraction.

This module tests the ContextualFeatureExtractor for environmental features,
door states, multi-room correlations, and seasonal patterns.
"""

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
import statistics

from src.core.config import RoomConfig, SystemConfig
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor


class TestContextualFeatureExtractor:
    """Test suite for ContextualFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a contextual feature extractor instance."""
        return ContextualFeatureExtractor()

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    @pytest.fixture
    def sample_events(self) -> List[SensorEvent]:
        """Create sample environmental sensor events."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Temperature events
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.sensor_id = "sensor.living_room_temperature"
            event.sensor_type = "temperature"
            event.state = str(22.0 + i * 0.5)  # 22.0 to 24.0
            event.room_id = "living_room"
            events.append(event)

        # Humidity events
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 15)
            event.sensor_id = "sensor.living_room_humidity"
            event.sensor_type = "humidity"
            event.state = str(45.0 + i * 5.0)  # 45%, 50%, 55%
            event.room_id = "living_room"
            events.append(event)

        # Light events
        for i in range(4):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 12)
            event.sensor_id = "sensor.living_room_light"
            event.sensor_type = "illuminance"
            event.state = str(200 + i * 100)  # 200 to 500 lux
            event.room_id = "living_room"
            events.append(event)

        # Door events
        door_states = ["closed", "open", "closed", "open", "closed"]
        for i, door_state in enumerate(door_states):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 8)
            event.sensor_id = "sensor.living_room_door"
            event.sensor_type = "door"
            event.state = door_state
            event.room_id = "living_room"
            events.append(event)

        return events

    @pytest.fixture
    def sample_room_states(self) -> List[RoomState]:
        """Create sample room states for multi-room analysis."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        states = []

        rooms = ["living_room", "kitchen", "bedroom"]
        for i in range(12):  # Multiple states per room
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 5)
            state.room_id = rooms[i % 3]
            state.is_occupied = (i % 4) < 2  # Alternating occupancy pattern
            state.occupancy_confidence = 0.8 + (i % 3) * 0.05
            states.append(state)

        return states

    def test_extract_features_comprehensive(
        self, extractor, sample_events, sample_room_states, target_time
    ):
        """Test comprehensive feature extraction with all feature types."""
        features = extractor.extract_features(
            sample_events, sample_room_states, target_time
        )

        # Verify basic structure
        assert isinstance(features, dict)
        assert len(features) > 25  # Should have many contextual features

        # Check environmental features
        environmental_features = [
            "avg_temperature",
            "temperature_change_rate",
            "temperature_stability",
            "avg_humidity",
            "humidity_change_rate",
            "humidity_stability",
            "avg_light_level",
            "light_change_rate",
            "natural_light_score",
        ]

        for feature in environmental_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

        # Check door state features
        door_features = [
            "door_open_ratio",
            "door_transition_count",
            "avg_door_open_duration",
        ]

        for feature in door_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

        # Check multi-room features
        multi_room_features = [
            "active_rooms_count",
            "simultaneous_occupancy_ratio",
            "cross_room_correlation",
            "occupancy_spread_score",
        ]

        for feature in multi_room_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_extract_features_empty_events(self, extractor, target_time):
        """Test behavior with empty event list."""
        features = extractor.extract_features([], None, target_time)

        # Should return default features
        expected_defaults = extractor._get_default_features()
        assert features == expected_defaults

        # Verify reasonable defaults
        assert features["avg_temperature"] == 22.0  # Default room temperature
        assert features["avg_humidity"] == 50.0  # Default humidity
        assert features["door_open_ratio"] == 0.0  # No door activity

    def test_environmental_features_temperature(self, extractor):
        """Test temperature-specific environmental features."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create temperature sequence with known values
        temperatures = [20.0, 22.0, 24.0, 23.0, 21.5]
        for i, temp in enumerate(temperatures):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 12)
            event.sensor_id = "sensor.temperature"
            event.sensor_type = "temperature"
            event.state = str(temp)
            event.room_id = "living_room"
            events.append(event)

        features = extractor._extract_environmental_features(events, target_time)

        # Verify temperature calculations
        expected_avg = sum(temperatures) / len(temperatures)
        assert abs(features["avg_temperature"] - expected_avg) < 0.1

        # Temperature change rate should be calculated correctly
        assert "temperature_change_rate" in features
        assert isinstance(features["temperature_change_rate"], float)

        # Temperature stability (variance-based)
        expected_variance = statistics.variance(temperatures)
        stability_score = max(0.0, 1.0 - (expected_variance / 10.0))
        assert abs(features["temperature_stability"] - stability_score) < 0.1

    def test_environmental_features_humidity(self, extractor):
        """Test humidity-specific environmental features."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create humidity sequence
        humidity_values = [40.0, 45.0, 50.0, 48.0, 52.0]
        for i, humidity in enumerate(humidity_values):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.sensor_id = "sensor.humidity"
            event.sensor_type = "humidity"
            event.state = str(humidity)
            event.room_id = "living_room"
            events.append(event)

        features = extractor._extract_environmental_features(events, target_time)

        # Verify humidity calculations
        expected_avg = sum(humidity_values) / len(humidity_values)
        assert abs(features["avg_humidity"] - expected_avg) < 0.1

        # Humidity change rate
        assert "humidity_change_rate" in features
        assert isinstance(features["humidity_change_rate"], float)

    def test_environmental_features_light(self, extractor):
        """Test light-specific environmental features."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create light level sequence (in lux)
        light_values = [100, 300, 500, 400, 200]
        for i, light in enumerate(light_values):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.sensor_id = "sensor.light"
            event.sensor_type = "illuminance"
            event.state = str(light)
            event.room_id = "living_room"
            events.append(event)

        features = extractor._extract_environmental_features(events, target_time)

        # Verify light calculations
        expected_avg = sum(light_values) / len(light_values)
        assert abs(features["avg_light_level"] - expected_avg) < 1.0

        # Natural light score should be calculated
        assert "natural_light_score" in features
        assert 0.0 <= features["natural_light_score"] <= 1.0

    def test_door_state_features(self, extractor):
        """Test door state analysis features."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create door state sequence with known pattern
        door_sequence = [
            (0, "closed"),
            (10, "open"),
            (25, "closed"),  # 15 min open
            (30, "open"),
            (40, "closed"),  # 10 min open
            (50, "open"),
            (55, "closed"),  # 5 min open
        ]

        for minutes, state in door_sequence:
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=minutes)
            event.sensor_id = "sensor.door"
            event.sensor_type = "door"
            event.state = state
            event.room_id = "living_room"
            events.append(event)

        features = extractor._extract_door_state_features(events, target_time)

        # Verify door analysis
        total_duration = 55  # minutes
        total_open_duration = 15 + 10 + 5  # 30 minutes
        expected_open_ratio = total_open_duration / total_duration

        assert abs(features["door_open_ratio"] - expected_open_ratio) < 0.05
        assert features["door_transition_count"] == 6.0  # Number of state changes

        # Average open duration: (15 + 10 + 5) / 3 = 10 minutes
        expected_avg_open = 10.0
        assert abs(features["avg_door_open_duration"] - expected_avg_open) < 1.0

    def test_multi_room_correlation_features(self, extractor):
        """Test multi-room correlation analysis."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []  # Multi-room features need events list
        room_states = []

        # Create correlated occupancy pattern
        # Pattern: living_room and kitchen often occupied together
        patterns = [
            (0, "living_room", True),
            (5, "kitchen", True),  # Both occupied
            (10, "bedroom", False),  # Bedroom vacant
            (15, "living_room", False),
            (20, "kitchen", False),  # Both vacant
            (25, "bedroom", True),  # Only bedroom occupied
            (30, "living_room", True),
            (35, "kitchen", True),  # Both occupied again
        ]

        for minutes, room, occupied in patterns:
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=minutes)
            state.room_id = room
            state.is_occupied = occupied
            state.occupancy_confidence = 0.85
            room_states.append(state)

        features = extractor._extract_multi_room_features(events, room_states, target_time)

        # Should detect correlation between living_room and kitchen
        assert features["active_rooms_count"] > 1.0
        assert features["cross_room_correlation"] > 0.0
        assert features["simultaneous_occupancy_ratio"] > 0.0

    def test_seasonal_features(self, extractor):
        """Test seasonal and holiday detection features."""
        # Test different seasonal dates
        test_dates = [
            (datetime(2024, 12, 25, 15, 0, 0), True),  # Christmas
            (datetime(2024, 7, 4, 15, 0, 0), True),  # July 4th
            (datetime(2024, 1, 1, 15, 0, 0), True),  # New Year
            (datetime(2024, 6, 15, 15, 0, 0), False),  # Regular day
            (datetime(2024, 3, 20, 15, 0, 0), False),  # Regular day
        ]

        for target_time, is_holiday in test_dates:
            features = extractor._extract_seasonal_features(target_time)

            assert "is_holiday_season" in features
            assert features["is_holiday_season"] == (1.0 if is_holiday else 0.0)

            assert "season_indicator" in features
            assert (
                0.0 <= features["season_indicator"] <= 3.0
            )  # 0=winter, 1=spring, 2=summer, 3=fall

    def test_weather_integration_features(self, extractor):
        """Test weather-based contextual features."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = base_time + timedelta(hours=1)
        events = []

        # Create weather-like environmental changes
        # Simulate cloudy weather (low light, high humidity)
        weather_data = [
            (0, "illuminance", "150"),  # Low light
            (10, "humidity", "75"),  # High humidity
            (20, "temperature", "18"),  # Cool temperature
            (30, "illuminance", "100"),  # Very low light
            (40, "humidity", "80"),  # Very high humidity
        ]

        for minutes, sensor_type, value in weather_data:
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=minutes)
            event.sensor_id = f"sensor.{sensor_type}"
            event.sensor_type = sensor_type
            event.state = value
            event.room_id = "living_room"
            events.append(event)

        features = extractor._extract_environmental_features(events, target_time)

        # Should detect poor weather conditions
        assert features["natural_light_score"] < 0.5  # Low natural light
        assert features["avg_humidity"] > 70.0  # High humidity

    def test_cross_sensor_correlation(self, extractor):
        """Test correlation analysis between different sensor types."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Create correlated sensor events
        # Light increases when doors open (natural light)
        sensor_patterns = [
            (0, "door", "closed", "illuminance", "200"),
            (10, "door", "open", "illuminance", "400"),  # Door open -> more light
            (20, "door", "closed", "illuminance", "180"),
            (30, "door", "open", "illuminance", "450"),  # Door open -> more light
            (40, "door", "closed", "illuminance", "190"),
        ]

        for minutes, door_type, door_state, light_type, light_value in sensor_patterns:
            # Door event
            door_event = Mock(spec=SensorEvent)
            door_event.timestamp = base_time + timedelta(minutes=minutes)
            door_event.sensor_id = "sensor.door"
            door_event.sensor_type = door_type
            door_event.state = door_state
            door_event.room_id = "living_room"
            events.append(door_event)

            # Light event (slightly delayed)
            light_event = Mock(spec=SensorEvent)
            light_event.timestamp = base_time + timedelta(minutes=minutes + 2)
            light_event.sensor_id = "sensor.light"
            light_event.sensor_type = light_type
            light_event.state = light_value
            light_event.room_id = "living_room"
            events.append(light_event)

        features = extractor.extract_features(
            events, None, base_time + timedelta(hours=1)
        )

        # Should detect correlation between door opening and light increase
        assert features["door_transition_count"] > 0
        assert features["avg_light_level"] > 200  # Higher than base light level

    def test_natural_light_patterns(self, extractor):
        """Test natural light pattern detection."""
        # Create daily light pattern
        base_time = datetime(2024, 1, 15, 6, 0, 0)  # Start at 6 AM
        events = []

        # Simulate natural light progression through the day
        hourly_light = [
            (6, 50),  # Dawn - low light
            (8, 200),  # Morning - increasing
            (10, 400),  # Mid-morning - bright
            (12, 600),  # Noon - brightest
            (14, 500),  # Afternoon - still bright
            (16, 300),  # Late afternoon - decreasing
            (18, 100),  # Evening - dim
            (20, 10),  # Night - very dim
        ]

        for hour, light_level in hourly_light:
            event = Mock(spec=SensorEvent)
            event.timestamp = datetime(2024, 1, 15, hour, 0, 0)
            event.sensor_id = "sensor.light"
            event.sensor_type = "illuminance"
            event.state = str(light_level)
            event.room_id = "living_room"
            events.append(event)

        target_time = datetime(2024, 1, 15, 15, 0, 0)  # 3 PM

        # Filter events to only include those up to target time (as extract_features would do)
        filtered_events = [e for e in events if e.timestamp <= target_time]

        features = extractor._extract_environmental_features(
            filtered_events, target_time
        )

        # Should detect natural light pattern
        assert features["natural_light_score"] > 0.0
        assert features["light_change_rate"] != 0.0  # Should detect light changes

    def test_occupancy_spread_analysis(self, extractor):
        """Test occupancy spread score calculation."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        room_states = []

        rooms = ["living_room", "kitchen", "bedroom", "office"]

        # Scenario 1: Concentrated occupancy (only living room)
        concentrated_pattern = [(0, "living_room", True)]
        for minutes, room, occupied in concentrated_pattern:
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=minutes)
            state.room_id = room
            state.is_occupied = occupied
            state.occupancy_confidence = 0.9
            room_states.append(state)

        features_concentrated = extractor._extract_multi_room_features([], room_states, base_time)

        room_states.clear()

        # Scenario 2: Spread occupancy (multiple rooms)
        spread_pattern = [
            (0, "living_room", True),
            (5, "kitchen", True),
            (10, "bedroom", True),
        ]
        for minutes, room, occupied in spread_pattern:
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=minutes)
            state.room_id = room
            state.is_occupied = occupied
            state.occupancy_confidence = 0.9
            room_states.append(state)

        features_spread = extractor._extract_multi_room_features([], room_states, base_time)

        # Spread scenario should have higher spread score
        assert (
            features_spread["occupancy_spread_score"]
            > features_concentrated["occupancy_spread_score"]
        )
        assert (
            features_spread["active_rooms_count"]
            > features_concentrated["active_rooms_count"]
        )

    def test_environmental_sensor_identification(self, extractor):
        """Test proper identification of environmental sensors."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Test various sensor identification patterns
        sensor_configs = [
            ("sensor.living_room_temperature", "temperature"),
            ("climate.thermostat_temperature", "temperature"),
            ("sensor.humidity_sensor", "humidity"),
            ("sensor.light_level", "illuminance"),
            ("sensor.lux_meter", "illuminance"),
            ("binary_sensor.front_door", "door"),
            (
                "switch.bedroom_light",
                "illuminance",
            ),  # Light switch as illuminance proxy
        ]

        for sensor_id, sensor_type in sensor_configs:
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time
            event.sensor_id = sensor_id
            event.sensor_type = sensor_type
            event.state = "25.0" if "temperature" in sensor_type else "50.0"
            event.room_id = "living_room"
            events.append(event)

        # Should identify environmental sensors correctly
        environmental_events = extractor._filter_environmental_events(events)
        assert len(environmental_events) >= 6  # Should identify most sensors

        door_events = extractor._filter_door_events(events)
        assert len(door_events) >= 1  # Should identify door sensor

    def test_feature_calculation_edge_cases(self, extractor, target_time):
        """Test edge cases in feature calculations."""
        # Test with single event
        single_event = Mock(spec=SensorEvent)
        single_event.timestamp = target_time - timedelta(minutes=30)
        single_event.sensor_id = "sensor.temperature"
        single_event.sensor_type = "temperature"
        single_event.state = "22.5"
        single_event.room_id = "living_room"

        features = extractor.extract_features([single_event], None, target_time)

        # Should handle single event gracefully
        assert isinstance(features, dict)
        assert features["avg_temperature"] == 22.5

        # Test with identical values (no variance)
        identical_events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = target_time - timedelta(minutes=i * 10)
            event.sensor_id = "sensor.temperature"
            event.sensor_type = "temperature"
            event.state = "23.0"  # Same value
            event.room_id = "living_room"
            identical_events.append(event)

        features = extractor.extract_features(identical_events, None, target_time)

        # Should handle no variance gracefully
        assert features["temperature_stability"] == 1.0  # Perfect stability
        assert features["temperature_change_rate"] == 0.0  # No change

    def test_time_window_filtering(self, extractor, target_time):
        """Test proper time window filtering for features."""
        # Create events at different time distances from target
        events = []

        # Recent event (within default window)
        recent_event = Mock(spec=SensorEvent)
        recent_event.timestamp = target_time - timedelta(minutes=30)
        recent_event.sensor_id = "sensor.temperature"
        recent_event.sensor_type = "temperature"
        recent_event.state = "22.0"
        recent_event.room_id = "living_room"
        events.append(recent_event)

        # Old event (outside default window)
        old_event = Mock(spec=SensorEvent)
        old_event.timestamp = target_time - timedelta(hours=25)  # Too old
        old_event.sensor_id = "sensor.temperature"
        old_event.sensor_type = "temperature"
        old_event.state = "30.0"  # Very different value
        old_event.room_id = "living_room"
        events.append(old_event)

        features = extractor.extract_features(events, None, target_time)

        # Should only use recent event, not the old one
        assert features["avg_temperature"] == 22.0  # Only recent event value

    def test_default_features_completeness(self, extractor):
        """Test that default features cover all expected categories."""
        default_features = extractor._get_default_features()

        # Check for different feature categories
        expected_categories = [
            "temperature",  # Environmental
            "humidity",  # Environmental
            "light",  # Environmental
            "door",  # Door state
            "room",  # Multi-room
            "seasonal",  # Seasonal
        ]

        feature_keys = list(default_features.keys())
        for category in expected_categories:
            category_found = any(category in key.lower() for key in feature_keys)
            assert category_found, f"No features found for category: {category}"

    def test_error_handling(self, extractor, target_time):
        """Test error handling in contextual feature extraction."""
        # Test with malformed events
        bad_events = [Mock(spec=SensorEvent)]
        bad_events[0].timestamp = None  # This should cause an error
        bad_events[0].sensor_id = "sensor.test"
        bad_events[0].sensor_type = "temperature"
        bad_events[0].state = "invalid"
        bad_events[0].room_id = "living_room"

        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(bad_events, None, target_time)

    def test_feature_names_method(self, extractor):
        """Test get_feature_names method if it exists."""
        if hasattr(extractor, "get_feature_names"):
            feature_names = extractor.get_feature_names()
            assert isinstance(feature_names, list)
            assert len(feature_names) > 25

            # Should match keys from default features
            default_features = extractor._get_default_features()
            assert set(feature_names) == set(default_features.keys())

    def test_cache_operations(self, extractor):
        """Test cache functionality if available."""
        if hasattr(extractor, "feature_cache"):
            # Test cache initialization
            assert isinstance(extractor.feature_cache, dict)

            # Test cache operations
            extractor.feature_cache["test"] = {"cached_feature": 1.0}
            assert "test" in extractor.feature_cache

            # Test cache clearing if method exists
            if hasattr(extractor, "clear_cache"):
                extractor.clear_cache()
                assert len(extractor.feature_cache) == 0


class TestContextualFeatureExtractorIntegration:
    """Integration tests for contextual feature extraction."""

    @pytest.fixture
    def extractor(self):
        """Create a contextual feature extractor instance."""
        return ContextualFeatureExtractor()

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    def test_realistic_home_scenario(self, extractor, target_time):
        """Test with realistic home automation scenario."""
        base_time = target_time - timedelta(hours=2)
        events = []
        room_states = []

        # Morning routine simulation
        # 1. Wake up - bedroom becomes vacant, bathroom becomes occupied
        bedroom_vacant = Mock(spec=RoomState)
        bedroom_vacant.timestamp = base_time + timedelta(minutes=10)
        bedroom_vacant.room_id = "bedroom"
        bedroom_vacant.is_occupied = False
        bedroom_vacant.occupancy_confidence = 0.9
        room_states.append(bedroom_vacant)

        # 2. Bathroom door opens, light increases
        bathroom_door = Mock(spec=SensorEvent)
        bathroom_door.timestamp = base_time + timedelta(minutes=12)
        bathroom_door.sensor_id = "binary_sensor.bathroom_door"
        bathroom_door.sensor_type = "door"
        bathroom_door.state = "open"
        bathroom_door.room_id = "bathroom"
        events.append(bathroom_door)
        
        # Add door closing to create a transition
        bathroom_door_close = Mock(spec=SensorEvent)
        bathroom_door_close.timestamp = base_time + timedelta(minutes=15)
        bathroom_door_close.sensor_id = "binary_sensor.bathroom_door"
        bathroom_door_close.sensor_type = "door"
        bathroom_door_close.state = "closed"
        bathroom_door_close.room_id = "bathroom"
        events.append(bathroom_door_close)

        # 3. Kitchen activity - coffee maker, temperature rise
        kitchen_temp = Mock(spec=SensorEvent)
        kitchen_temp.timestamp = base_time + timedelta(minutes=30)
        kitchen_temp.sensor_id = "sensor.kitchen_temperature"
        kitchen_temp.sensor_type = "temperature"
        kitchen_temp.state = "25.0"  # Warmer due to cooking
        kitchen_temp.room_id = "kitchen"
        events.append(kitchen_temp)

        kitchen_occupied = Mock(spec=RoomState)
        kitchen_occupied.timestamp = base_time + timedelta(minutes=32)
        kitchen_occupied.room_id = "kitchen"
        kitchen_occupied.is_occupied = True
        kitchen_occupied.occupancy_confidence = 0.85
        room_states.append(kitchen_occupied)

        # 4. Living room - TV area, normal temperature
        living_temp = Mock(spec=SensorEvent)
        living_temp.timestamp = base_time + timedelta(minutes=60)
        living_temp.sensor_id = "sensor.living_room_temperature"
        living_temp.sensor_type = "temperature"
        living_temp.state = "22.5"
        living_temp.room_id = "living_room"
        events.append(living_temp)

        # Extract features for this realistic scenario
        features = extractor.extract_features(events, room_states, target_time)

        # Verify realistic feature values
        assert isinstance(features, dict)
        assert features["active_rooms_count"] >= 1  # At least kitchen occupied
        assert features["avg_temperature"] > 20.0  # Reasonable temperature
        assert features["door_transition_count"] >= 1  # Door activity detected

    def test_seasonal_behavior_patterns(self, extractor):
        """Test seasonal behavior detection across different times of year."""
        seasonal_test_cases = [
            # Winter scenario - more indoor activity, doors closed
            (datetime(2024, 1, 15, 15, 0, 0), 18.0, 60.0, "closed", 100),
            # Summer scenario - warmer, more door activity, brighter
            (datetime(2024, 7, 15, 15, 0, 0), 26.0, 40.0, "open", 500),
            # Spring scenario - moderate conditions
            (datetime(2024, 4, 15, 15, 0, 0), 22.0, 50.0, "closed", 300),
            # Fall scenario - cooling down, variable conditions
            (datetime(2024, 10, 15, 15, 0, 0), 20.0, 55.0, "open", 250),
        ]

        for target_time, temp, humidity, door_state, light_level in seasonal_test_cases:
            events = []

            # Environmental conditions for the season
            temp_event = Mock(spec=SensorEvent)
            temp_event.timestamp = target_time - timedelta(minutes=30)
            temp_event.sensor_id = "sensor.temperature"
            temp_event.sensor_type = "temperature"
            temp_event.state = str(temp)
            temp_event.room_id = "living_room"
            events.append(temp_event)

            humidity_event = Mock(spec=SensorEvent)
            humidity_event.timestamp = target_time - timedelta(minutes=30)
            humidity_event.sensor_id = "sensor.humidity"
            humidity_event.sensor_type = "humidity"
            humidity_event.state = str(humidity)
            humidity_event.room_id = "living_room"
            events.append(humidity_event)

            # Create door events sequence to establish proper time-based ratio
            if door_state == "open":
                # Scenario: Door is open most of the time with clear transitions
                door_event_start = Mock(spec=SensorEvent)
                door_event_start.timestamp = target_time - timedelta(minutes=60)
                door_event_start.sensor_id = "sensor.door"
                door_event_start.sensor_type = "door"
                door_event_start.state = "closed"
                door_event_start.room_id = "living_room"
                events.append(door_event_start)
                
                # Open early and stay open (45 minutes out of 55 total)
                door_event_open = Mock(spec=SensorEvent)
                door_event_open.timestamp = target_time - timedelta(minutes=50)
                door_event_open.sensor_id = "sensor.door"
                door_event_open.sensor_type = "door"
                door_event_open.state = "open"
                door_event_open.room_id = "living_room"
                events.append(door_event_open)
                
                # Close briefly near the end to create a measurable open duration
                door_event_close = Mock(spec=SensorEvent)
                door_event_close.timestamp = target_time - timedelta(minutes=5)
                door_event_close.sensor_id = "sensor.door"
                door_event_close.sensor_type = "door"
                door_event_close.state = "closed"
                door_event_close.room_id = "living_room"
                events.append(door_event_close)
            else:
                # Scenario: Door stays closed for the full period
                door_event_start = Mock(spec=SensorEvent)
                door_event_start.timestamp = target_time - timedelta(minutes=60)
                door_event_start.sensor_id = "sensor.door"
                door_event_start.sensor_type = "door"
                door_event_start.state = "closed"
                door_event_start.room_id = "living_room"
                events.append(door_event_start)
                
                door_event_end = Mock(spec=SensorEvent)
                door_event_end.timestamp = target_time - timedelta(minutes=5)
                door_event_end.sensor_id = "sensor.door"
                door_event_end.sensor_type = "door"
                door_event_end.state = "closed"
                door_event_end.room_id = "living_room"
                events.append(door_event_end)

            light_event = Mock(spec=SensorEvent)
            light_event.timestamp = target_time - timedelta(minutes=30)
            light_event.sensor_id = "sensor.light"
            light_event.sensor_type = "illuminance"
            light_event.state = str(light_level)
            light_event.room_id = "living_room"
            events.append(light_event)

            features = extractor.extract_features(events, None, target_time)

            # Verify seasonal characteristics are captured
            assert abs(features["avg_temperature"] - temp) < 1.0
            assert abs(features["avg_humidity"] - humidity) < 5.0
            if door_state == "open":
                # When door is open 45 minutes out of 55 total, expect ratio > 0.7
                assert features["door_open_ratio"] > 0.7
            else:
                # When door stays closed, expect zero ratio
                assert features["door_open_ratio"] == 0.0
            assert abs(features["avg_light_level"] - light_level) < 10.0

    def test_multi_home_correlation(self, extractor, target_time):
        """Test correlation patterns across entire home."""
        base_time = target_time - timedelta(hours=1)
        events = []
        room_states = []

        # Simulate whole-house activity pattern
        # Family dinner preparation - kitchen and dining room active
        dinner_prep_rooms = ["kitchen", "dining_room"]
        for i, room in enumerate(dinner_prep_rooms):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=10 + i * 5)
            state.room_id = room
            state.is_occupied = True
            state.occupancy_confidence = 0.9
            room_states.append(state)

        # After dinner - living room becomes active, kitchen becomes inactive
        kitchen_vacant = Mock(spec=RoomState)
        kitchen_vacant.timestamp = base_time + timedelta(minutes=40)
        kitchen_vacant.room_id = "kitchen"
        kitchen_vacant.is_occupied = False
        kitchen_vacant.occupancy_confidence = 0.85
        room_states.append(kitchen_vacant)

        living_occupied = Mock(spec=RoomState)
        living_occupied.timestamp = base_time + timedelta(minutes=42)
        living_occupied.room_id = "living_room"
        living_occupied.is_occupied = True
        living_occupied.occupancy_confidence = 0.9
        room_states.append(living_occupied)

        # Bedtime - bedrooms become active, common areas vacant
        bedtime_transition = [
            ("living_room", False),
            ("dining_room", False),
            ("bedroom", True),
            ("master_bedroom", True),
        ]

        for i, (room, occupied) in enumerate(bedtime_transition):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=50 + i * 2)
            state.room_id = room
            state.is_occupied = occupied
            state.occupancy_confidence = 0.8
            room_states.append(state)

        features = extractor._extract_multi_room_features(events, room_states, target_time)

        # Should detect complex correlation patterns
        assert features["active_rooms_count"] > 1.0  # Multiple room transitions
        assert features["cross_room_correlation"] != 0.0  # Some correlation detected
        assert 0.0 <= features["occupancy_spread_score"] <= 1.0  # Valid spread score


class TestContextualFeatureExtractorEdgeCases:
    """Additional edge case tests for ContextualFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a contextual feature extractor instance."""
        return ContextualFeatureExtractor()

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    def test_no_room_states(self, extractor, target_time):
        """Test feature extraction when no room states are provided."""
        features = extractor.extract_features([], None, target_time)

        # Should handle None room states gracefully
        assert isinstance(features, dict)
        # Multi-room features should have defaults when no room states
        assert features["simultaneous_occupancy_ratio"] == 0.0

    def test_mixed_sensor_types(self, extractor, target_time):
        """Test extraction with mixed environmental sensor types."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        events = []

        # Mixed sensor events with different identification patterns
        mixed_sensors = [
            ("sensor.temp_1", "temperature", "22.5"),
            ("climate.hvac_temp", "temperature", "23.0"),
            ("sensor.humidity_main", "humidity", "45.0"),
            ("weather.indoor_humidity", "humidity", "48.0"),
            ("sensor.lux_sensor", "illuminance", "300"),
            ("light.brightness_sensor", "illuminance", "250"),
            ("binary_sensor.main_door", "door", "closed"),
            ("switch.garage_door", "door", "open"),
        ]

        for i, (sensor_id, sensor_type, state) in enumerate(mixed_sensors):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.sensor_id = sensor_id
            event.sensor_type = sensor_type
            event.state = state
            event.room_id = "living_room"
            events.append(event)

        features = extractor.extract_features(events, None, target_time)

        # Should handle mixed sensor types correctly
        assert isinstance(features, dict)
        assert features["avg_temperature"] > 20.0  # Should average temp sensors
        assert features["avg_humidity"] > 40.0  # Should average humidity sensors
        assert features["avg_light_level"] > 200  # Should average light sensors

    def test_invalid_sensor_values(self, extractor, target_time):
        """Test handling of invalid or corrupted sensor values."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Mix of valid and invalid sensor values
        sensor_data = [
            ("sensor.temp_1", "temperature", "22.5"),  # Valid
            ("sensor.temp_2", "temperature", "invalid"),  # Invalid string
            ("sensor.temp_3", "temperature", "-999"),  # Unrealistic value
            ("sensor.humidity_1", "humidity", "45.0"),  # Valid
            ("sensor.humidity_2", "humidity", ""),  # Empty string
            ("sensor.light_1", "illuminance", "300"),  # Valid
            ("sensor.light_2", "illuminance", "NaN"),  # NaN value
        ]

        for i, (sensor_id, sensor_type, state) in enumerate(sensor_data):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.sensor_id = sensor_id
            event.sensor_type = sensor_type
            event.state = state
            event.room_id = "living_room"
            events.append(event)

        features = extractor.extract_features(events, None, target_time)

        # Should handle invalid values gracefully and use only valid ones
        assert isinstance(features, dict)
        # Should only use valid temperature value (22.5)
        assert features["avg_temperature"] == 22.5
        # Should only use valid humidity value (45.0)
        assert features["avg_humidity"] == 45.0
        # Should only use valid light value (300)
        assert features["avg_light_level"] == 300.0

    def test_extreme_environmental_conditions(self, extractor, target_time):
        """Test handling of extreme environmental conditions."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Extreme conditions that might occur in real scenarios
        extreme_conditions = [
            ("temperature", "40.0"),  # Very hot (104°F)
            ("temperature", "5.0"),  # Very cold (41°F)
            ("humidity", "95.0"),  # Very humid
            ("humidity", "10.0"),  # Very dry
            ("illuminance", "10000"),  # Very bright (direct sunlight)
            ("illuminance", "1"),  # Very dark
        ]

        for i, (sensor_type, state) in enumerate(extreme_conditions):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.sensor_id = f"sensor.{sensor_type}_{i}"
            event.sensor_type = sensor_type
            event.state = state
            event.room_id = "living_room"
            events.append(event)

        features = extractor.extract_features(events, None, target_time)

        # Should handle extreme values without errors
        assert isinstance(features, dict)
        # Should calculate reasonable statistics even with extreme values
        assert features["temperature_stability"] >= 0.0
        assert features["humidity_stability"] >= 0.0
        assert 0.0 <= features["natural_light_score"] <= 1.0

    def test_rapid_state_changes(self, extractor, target_time):
        """Test handling of very rapid sensor state changes."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Rapid door state changes (every 30 seconds)
        for i in range(20):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(seconds=i * 30)
            event.sensor_id = "sensor.door"
            event.sensor_type = "door"
            event.state = "open" if i % 2 == 0 else "closed"
            event.room_id = "living_room"
            events.append(event)

        features = extractor.extract_features(events, None, target_time)

        # Should handle rapid changes correctly
        assert isinstance(features, dict)
        assert features["door_transition_count"] == 19.0  # 19 transitions
        assert abs(features["door_open_ratio"] - 0.5) < 0.1  # Approximately half open, half closed

    def test_long_time_gaps(self, extractor, target_time):
        """Test handling of large time gaps between sensor readings."""
        events = []

        # Events with large time gaps
        time_points = [
            target_time - timedelta(days=2),  # Very old
            target_time - timedelta(hours=12),  # Half day old
            target_time - timedelta(hours=1),  # Recent
            target_time - timedelta(minutes=5),  # Very recent
        ]

        for i, timestamp in enumerate(time_points):
            event = Mock(spec=SensorEvent)
            event.timestamp = timestamp
            event.sensor_id = f"sensor.temp_{i}"
            event.sensor_type = "temperature"
            event.state = str(20.0 + i)  # 20, 21, 22, 23
            event.room_id = "living_room"
            events.append(event)

        features = extractor.extract_features(events, None, target_time)

        # Should filter to appropriate time window (default 24 hours)
        assert isinstance(features, dict)
        # Should only use events from last 24 hours (not the 2-day old one)
        # Average should be (21 + 22 + 23) / 3 = 22.0
        expected_avg = (21.0 + 22.0 + 23.0) / 3
        assert abs(features["avg_temperature"] - expected_avg) < 0.1

    def test_single_room_multi_states(self, extractor, target_time):
        """Test multi-room features with only single room data."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []
        room_states = []

        # Only living room states, but multiple over time
        for i in range(5):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 10)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0  # Alternating
            state.occupancy_confidence = 0.8
            room_states.append(state)

        features = extractor._extract_multi_room_features(events, room_states, target_time)

        # Should handle single room gracefully
        assert features["active_rooms_count"] == 1.0  # Only one room
        assert (
            features["simultaneous_occupancy_ratio"] == 0.0
        )  # No simultaneous with other rooms
        assert (
            features["cross_room_correlation"] == 0.0
        )  # No correlation with other rooms

    def test_memory_and_performance(self, extractor, target_time):
        """Test memory usage and performance with large datasets."""
        import time

        # Create large dataset
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        large_events = []
        large_room_states = []

        # 1000 environmental events
        for i in range(1000):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.sensor_id = f"sensor.temp_{i % 10}"  # 10 different sensors
            event.sensor_type = "temperature"
            event.state = str(20.0 + (i % 10))  # Varying temperatures
            event.room_id = f"room_{i % 5}"  # 5 different rooms
            large_events.append(event)

        # 500 room states
        for i in range(500):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 2)
            state.room_id = f"room_{i % 5}"
            state.is_occupied = i % 3 == 0  # Varying occupancy
            state.occupancy_confidence = 0.8
            large_room_states.append(state)

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(
            large_events, large_room_states, target_time
        )
        extraction_time = time.time() - start_time

        # Should complete within reasonable time (< 2 seconds)
        assert extraction_time < 2.0
        assert isinstance(features, dict)
        assert len(features) > 20  # Should still extract comprehensive features
