"""
Unit tests for contextual feature extraction.

This module tests the ContextualFeatureExtractor for environmental features,
door states, multi-room correlations, and seasonal patterns.
"""

import statistics
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.core.config import RoomConfig, SystemConfig
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor


class TestContextualFeatureExtractor:
    """Test suite for ContextualFeatureExtractor."""

    @pytest.fixture
    def mock_config(self):
        """Create mock system configuration."""
        config = Mock(spec=SystemConfig)
        config.rooms = {
            "living_room": Mock(spec=RoomConfig),
            "kitchen": Mock(spec=RoomConfig),
            "bedroom": Mock(spec=RoomConfig),
        }
        return config

    @pytest.fixture
    def extractor(self, mock_config):
        """Create a contextual feature extractor instance."""
        return ContextualFeatureExtractor(config=mock_config)

    @pytest.fixture
    def extractor_no_config(self):
        """Create extractor without configuration for edge case testing."""
        return ContextualFeatureExtractor(config=None)

    @pytest.fixture
    def room_configs(self):
        """Create room configuration dictionary."""
        configs = {}
        for room_id in ["living_room", "kitchen", "bedroom"]:
            config = Mock(spec=RoomConfig)
            config.room_id = room_id
            config.get_sensors_by_type = Mock(
                return_value={
                    "temperature": f"sensor.{room_id}_temperature",
                    "door": f"sensor.{room_id}_door",
                }
            )
            configs[room_id] = config
        return configs

    @pytest.fixture
    def environmental_events(self) -> List[SensorEvent]:
        """Create environmental sensor events for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Temperature events
        temp_values = ["22.5", "23.0", "23.2", "22.8", "22.1"]
        for i, temp in enumerate(temp_values):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10)
            event.room_id = "living_room"
            event.state = temp
            event.sensor_type = "temperature"
            event.sensor_id = "sensor.living_room_temperature"
            event.attributes = {"temperature": float(temp), "unit_of_measurement": "°C"}
            events.append(event)

        # Humidity events
        humidity_values = ["45.0", "48.0", "52.0", "50.0", "47.0"]
        for i, humidity in enumerate(humidity_values):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10 + 2)
            event.room_id = "living_room"
            event.state = humidity
            event.sensor_type = "humidity"
            event.sensor_id = "sensor.living_room_humidity"
            event.attributes = {"humidity": float(humidity), "unit_of_measurement": "%"}
            events.append(event)

        # Light events
        light_values = ["250", "300", "800", "1200", "400"]
        for i, light in enumerate(light_values):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 10 + 5)
            event.room_id = "living_room"
            event.state = light
            event.sensor_type = "light"
            event.sensor_id = "sensor.living_room_light"
            event.attributes = {
                "illuminance": float(light),
                "unit_of_measurement": "lx",
            }
            events.append(event)

        return events

    @pytest.fixture
    def door_events(self) -> List[SensorEvent]:
        """Create door sensor events for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Door open/close sequence
        door_states = ["open", "closed", "open", "closed", "open"]
        for i, state in enumerate(door_states):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.room_id = "living_room"
            event.state = state
            event.sensor_type = "door"
            event.sensor_id = "sensor.living_room_door"
            event.attributes = {"state": state}
            events.append(event)

        return events

    @pytest.fixture
    def multi_room_events(self) -> List[SensorEvent]:
        """Create events across multiple rooms."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        rooms = ["living_room", "kitchen", "bedroom"]
        for room_idx, room_id in enumerate(rooms):
            # Motion events for each room
            for i in range(3):
                event = Mock(spec=SensorEvent)
                event.timestamp = base_time + timedelta(minutes=room_idx * 10 + i * 2)
                event.room_id = room_id
                event.state = "on" if i % 2 == 0 else "off"
                event.sensor_type = "motion"
                event.sensor_id = f"sensor.{room_id}_motion"
                event.attributes = {}
                events.append(event)

        return events

    @pytest.fixture
    def room_states(self) -> List[RoomState]:
        """Create sample room states for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        states = []

        rooms = ["living_room", "kitchen", "bedroom"]
        for room_idx, room_id in enumerate(rooms):
            for i in range(4):
                state = Mock(spec=RoomState)
                state.timestamp = base_time + timedelta(minutes=room_idx * 15 + i * 5)
                state.room_id = room_id
                state.is_occupied = (i + room_idx) % 2 == 0
                state.occupancy_confidence = 0.7 + (i * 0.05)
                states.append(state)

        return states

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    def test_extract_features_with_environmental_data(
        self, extractor, environmental_events, room_states, target_time, room_configs
    ):
        """Test feature extraction with environmental sensor data."""
        features = extractor.extract_features(
            environmental_events,
            room_states,
            target_time,
            room_configs,
            lookback_hours=2,
        )

        # Verify basic feature structure
        assert isinstance(features, dict)
        assert len(features) > 25  # Should have many contextual features

        # Check environmental features
        expected_features = [
            "current_temperature",
            "avg_temperature",
            "temperature_trend",
            "current_humidity",
            "avg_humidity",
            "humidity_trend",
            "current_light",
            "avg_light",
            "light_trend",
            "is_comfortable_temp",
            "is_dim",
            "is_winter",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_extract_features_empty_events(self, extractor, target_time):
        """Test behavior with empty event list."""
        features = extractor.extract_features([], [], target_time)

        # Should return default features
        expected_defaults = extractor._get_default_features()
        assert features == expected_defaults

    def test_environmental_features_temperature(
        self, extractor, environmental_events, target_time
    ):
        """Test temperature feature extraction."""
        features = extractor._extract_environmental_features(
            environmental_events, target_time
        )

        # Check temperature features
        assert "current_temperature" in features
        assert "avg_temperature" in features
        assert "temperature_trend" in features
        assert "temperature_variance" in features

        # Temperature should be reasonable (from test data: 22.1 - 23.2°C)
        assert 20.0 <= features["current_temperature"] <= 25.0
        assert 22.0 <= features["avg_temperature"] <= 24.0

        # Comfort zone indicators
        assert "is_cold" in features
        assert "is_comfortable_temp" in features
        assert "is_warm" in features
        assert features["is_comfortable_temp"] == 1.0  # Should be in comfort zone

    def test_environmental_features_humidity(
        self, extractor, environmental_events, target_time
    ):
        """Test humidity feature extraction."""
        features = extractor._extract_environmental_features(
            environmental_events, target_time
        )

        # Check humidity features
        assert "current_humidity" in features
        assert "avg_humidity" in features
        assert "humidity_trend" in features

        # Humidity should be reasonable (from test data: 45-52%)
        assert 40.0 <= features["current_humidity"] <= 60.0
        assert 40.0 <= features["avg_humidity"] <= 60.0

    def test_environmental_features_light(
        self, extractor, environmental_events, target_time
    ):
        """Test light/illuminance feature extraction."""
        features = extractor._extract_environmental_features(
            environmental_events, target_time
        )

        # Check light features
        assert "current_light" in features
        assert "avg_light" in features
        assert "light_trend" in features

        # Light level categories
        assert "is_dark" in features
        assert "is_dim" in features
        assert "is_bright" in features

        # Values should be reasonable
        assert features["current_light"] >= 0
        assert features["avg_light"] >= 0

    def test_door_state_features(self, extractor, door_events, target_time):
        """Test door state feature extraction."""
        features = extractor._extract_door_state_features(door_events, target_time)

        # Check door features
        assert "doors_currently_open" in features
        assert "door_open_ratio" in features
        assert "door_transition_count" in features
        assert "avg_door_open_duration" in features
        assert "recent_door_activity" in features

        # Values should be reasonable
        assert features["door_transition_count"] > 0  # Should detect transitions
        assert 0.0 <= features["door_open_ratio"] <= 1.0

    def test_multi_room_features(
        self, extractor, multi_room_events, room_states, target_time
    ):
        """Test multi-room correlation features."""
        features = extractor._extract_multi_room_features(
            multi_room_events, room_states, target_time
        )

        # Check multi-room features
        assert "total_active_rooms" in features
        assert "simultaneous_occupancy_ratio" in features
        assert "room_activity_correlation" in features
        assert "dominant_room_activity_ratio" in features
        assert "room_activity_balance" in features

        # Should detect multiple active rooms
        assert features["total_active_rooms"] >= 2
        assert 0.0 <= features["room_activity_balance"] <= 1.0

    def test_seasonal_features(self, extractor):
        """Test seasonal and external context features."""
        # Test winter date
        winter_time = datetime(2024, 1, 15, 15, 0, 0)
        features = extractor._extract_seasonal_features(winter_time)

        assert "is_winter" in features
        assert "is_spring" in features
        assert "is_summer" in features
        assert "is_autumn" in features
        assert "natural_light_available" in features
        assert "is_holiday_season" in features

        # Should detect winter
        assert features["is_winter"] == 1.0
        assert features["is_spring"] == 0.0
        assert features["is_summer"] == 0.0
        assert features["is_autumn"] == 0.0

        # Test summer date
        summer_time = datetime(2024, 7, 15, 15, 0, 0)
        summer_features = extractor._extract_seasonal_features(summer_time)
        assert summer_features["is_summer"] == 1.0
        assert summer_features["is_winter"] == 0.0

    def test_sensor_correlation_features(self, extractor, multi_room_events):
        """Test cross-sensor correlation features."""
        features = extractor._extract_sensor_correlation_features(multi_room_events)

        # Check correlation features
        assert "sensor_activation_correlation" in features
        assert "multi_sensor_event_ratio" in features
        assert "sensor_type_diversity" in features

        # Values should be reasonable
        assert features["sensor_activation_correlation"] >= 0
        assert 0.0 <= features["multi_sensor_event_ratio"] <= 1.0
        assert features["sensor_type_diversity"] >= 0

    def test_room_context_features(
        self, extractor, multi_room_events, room_configs, target_time
    ):
        """Test room-specific context features."""
        features = extractor._extract_room_context_features(
            multi_room_events, room_configs, target_time
        )

        # Check room context features
        assert "max_room_complexity" in features
        assert "avg_room_complexity" in features
        assert "max_room_activity" in features
        assert "room_activity_variance" in features

        # Values should be reasonable
        assert features["max_room_complexity"] >= 0
        assert features["avg_room_complexity"] >= 0

    def test_extract_numeric_values(self, extractor):
        """Test numeric value extraction from sensor events."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Event with numeric state
        event1 = Mock(spec=SensorEvent)
        event1.timestamp = base_time
        event1.state = "23.5"
        event1.attributes = {}
        events.append(event1)

        # Event with numeric attribute
        event2 = Mock(spec=SensorEvent)
        event2.timestamp = base_time + timedelta(minutes=5)
        event2.state = "unknown"
        event2.attributes = {"temperature": 24.2}
        events.append(event2)

        # Event with non-numeric data
        event3 = Mock(spec=SensorEvent)
        event3.timestamp = base_time + timedelta(minutes=10)
        event3.state = "on"
        event3.attributes = {"state": "active"}
        events.append(event3)

        values = extractor._extract_numeric_values(events, "temperature")

        # Should extract numeric values correctly
        assert len(values) == 2
        assert 23.5 in values
        assert 24.2 in values

    def test_calculate_trend(self, extractor):
        """Test trend calculation accuracy."""
        # Test increasing trend
        increasing_values = [1.0, 2.0, 3.0, 4.0, 5.0]
        trend = extractor._calculate_trend(increasing_values)
        assert trend > 0  # Should be positive slope

        # Test decreasing trend
        decreasing_values = [5.0, 4.0, 3.0, 2.0, 1.0]
        trend = extractor._calculate_trend(decreasing_values)
        assert trend < 0  # Should be negative slope

        # Test flat trend
        flat_values = [3.0, 3.0, 3.0, 3.0, 3.0]
        trend = extractor._calculate_trend(flat_values)
        assert abs(trend) < 0.01  # Should be near zero

        # Test single value
        single_value = [3.0]
        trend = extractor._calculate_trend(single_value)
        assert trend == 0.0

    def test_room_activity_correlation_calculation(self, extractor):
        """Test room activity correlation calculation."""
        target_time = datetime(2024, 1, 15, 20, 0, 0)

        # Create correlated room events (events happening at similar times)
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        room_events = defaultdict(list)

        # Room 1 and Room 2 have correlated activity
        for i in range(10):
            # Room 1 event
            event1 = Mock(spec=SensorEvent)
            event1.timestamp = base_time + timedelta(minutes=i * 30)
            room_events["room_1"].append(event1)

            # Room 2 event (correlated - happens shortly after room 1)
            event2 = Mock(spec=SensorEvent)
            event2.timestamp = base_time + timedelta(minutes=i * 30 + 5)
            room_events["room_2"].append(event2)

            # Room 3 event (uncorrelated - random timing)
            if i % 3 == 0:
                event3 = Mock(spec=SensorEvent)
                event3.timestamp = base_time + timedelta(minutes=i * 45 + 20)
                room_events["room_3"].append(event3)

        correlation = extractor._calculate_room_activity_correlation(
            room_events, target_time
        )

        # Should detect some correlation
        assert isinstance(correlation, float)
        assert 0.0 <= correlation <= 1.0

    def test_feature_names_method(self, extractor):
        """Test get_feature_names method."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 25

        # Should match default features keys
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_cache_operations(self, extractor):
        """Test cache clear functionality."""
        # Add something to cache
        extractor.context_cache["test"] = "value"
        assert "test" in extractor.context_cache

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.context_cache) == 0

    def test_threshold_configuration(self, extractor):
        """Test that threshold values are properly configured."""
        # Check temperature thresholds
        assert (
            extractor.temp_thresholds["cold"] < extractor.temp_thresholds["comfortable"]
        )
        assert (
            extractor.temp_thresholds["comfortable"] < extractor.temp_thresholds["warm"]
        )

        # Check humidity thresholds
        assert (
            extractor.humidity_thresholds["dry"]
            < extractor.humidity_thresholds["comfortable"]
        )
        assert (
            extractor.humidity_thresholds["comfortable"]
            < extractor.humidity_thresholds["humid"]
        )

        # Check light thresholds
        assert extractor.light_thresholds["dark"] < extractor.light_thresholds["dim"]
        assert extractor.light_thresholds["dim"] < extractor.light_thresholds["bright"]

    @pytest.mark.parametrize("lookback_hours", [1, 6, 12, 24, 48])
    def test_different_lookback_windows(
        self, extractor, environmental_events, room_states, target_time, lookback_hours
    ):
        """Test feature extraction with different lookback windows."""
        features = extractor.extract_features(
            environmental_events,
            room_states,
            target_time,
            lookback_hours=lookback_hours,
        )

        # Should return valid features regardless of lookback window
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extreme_environmental_values(self, extractor):
        """Test handling of extreme environmental sensor values."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Extreme temperature event
        extreme_temp_event = Mock(spec=SensorEvent)
        extreme_temp_event.timestamp = base_time
        extreme_temp_event.state = "-40.0"  # Very cold
        extreme_temp_event.sensor_type = "temperature"
        extreme_temp_event.sensor_id = "sensor.test_temp"
        extreme_temp_event.attributes = {"temperature": -40.0}

        # Very high humidity event
        extreme_humidity_event = Mock(spec=SensorEvent)
        extreme_humidity_event.timestamp = base_time + timedelta(minutes=5)
        extreme_humidity_event.state = "95.0"  # Very humid
        extreme_humidity_event.sensor_type = "humidity"
        extreme_humidity_event.sensor_id = "sensor.test_humidity"
        extreme_humidity_event.attributes = {"humidity": 95.0}

        events = [extreme_temp_event, extreme_humidity_event]
        features = extractor._extract_environmental_features(events, target_time)

        # Should handle extreme values gracefully
        assert features["current_temperature"] == -40.0
        assert features["is_cold"] == 1.0
        assert features["is_comfortable_temp"] == 0.0

        assert features["current_humidity"] == 95.0

    def test_missing_sensor_attributes(self, extractor):
        """Test handling of events with missing sensor attributes."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Event with no attributes
        event1 = Mock(spec=SensorEvent)
        event1.timestamp = base_time
        event1.state = "22.0"
        event1.sensor_type = "temperature"
        event1.sensor_id = "sensor.temp"
        event1.attributes = None

        # Event with empty attributes
        event2 = Mock(spec=SensorEvent)
        event2.timestamp = base_time + timedelta(minutes=5)
        event2.state = "unknown"
        event2.sensor_type = "temperature"
        event2.sensor_id = "sensor.temp2"
        event2.attributes = {}

        events = [event1, event2]
        values = extractor._extract_numeric_values(events, "temperature")

        # Should extract from state when attributes are missing
        assert 22.0 in values

    def test_error_handling(self, extractor, target_time):
        """Test error handling in feature extraction."""
        # Test with malformed events
        bad_events = [Mock(spec=SensorEvent)]
        bad_events[0].timestamp = None  # This should cause an error
        bad_events[0].room_id = "test_room"
        bad_events[0].state = "22.0"
        bad_events[0].sensor_type = "temperature"
        bad_events[0].sensor_id = "sensor.test"
        bad_events[0].attributes = {}

        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(bad_events, [], target_time)

    def test_performance_large_datasets(self, extractor):
        """Test performance with large environmental datasets."""
        import time

        base_time = datetime(2024, 1, 15, 0, 0, 0)
        target_time = datetime(2024, 1, 15, 23, 59, 59)

        # Create large dataset
        large_events = []
        for i in range(2000):  # 2000 environmental readings
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 0.7)
            event.room_id = f"room_{i % 5}"
            event.state = str(20.0 + (i % 10))  # Temperature range 20-29
            event.sensor_type = "temperature"
            event.sensor_id = f"sensor.temp_{i % 20}"
            event.attributes = {"temperature": 20.0 + (i % 10)}
            large_events.append(event)

        # Large room states dataset
        large_room_states = []
        for i in range(500):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 3)
            state.room_id = f"room_{i % 5}"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.5 + (i % 5) * 0.1
            large_room_states.append(state)

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(
            large_events, large_room_states, target_time
        )
        extraction_time = time.time() - start_time

        # Should complete in reasonable time (< 10 seconds)
        assert extraction_time < 10.0
        assert isinstance(features, dict)
        assert len(features) > 0

    @pytest.mark.asyncio
    async def test_concurrent_extraction(
        self, extractor, environmental_events, room_states, target_time
    ):
        """Test thread safety of feature extraction."""
        import asyncio

        async def extract_features():
            return extractor.extract_features(
                environmental_events, room_states, target_time
            )

        # Run multiple extractions concurrently
        tasks = [extract_features() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result

    def test_natural_light_patterns(self, extractor):
        """Test natural light availability patterns by season and time."""
        # Test different times of day in different seasons
        test_cases = [
            # Summer - long days
            (datetime(2024, 7, 15, 6, 0, 0), 1.0),  # 6 AM summer - light
            (datetime(2024, 7, 15, 21, 0, 0), 0.0),  # 9 PM summer - dark
            # Winter - short days
            (datetime(2024, 1, 15, 8, 0, 0), 0.0),  # 8 AM winter - dark
            (datetime(2024, 1, 15, 16, 0, 0), 1.0),  # 4 PM winter - light
            (datetime(2024, 1, 15, 18, 0, 0), 0.0),  # 6 PM winter - dark
            # Spring/Autumn - medium days
            (datetime(2024, 4, 15, 7, 0, 0), 0.0),  # 7 AM spring - dark
            (datetime(2024, 4, 15, 18, 0, 0), 1.0),  # 6 PM spring - light
        ]

        for test_time, expected_light in test_cases:
            features = extractor._extract_seasonal_features(test_time)
            assert features["natural_light_available"] == expected_light

    def test_holiday_season_detection(self, extractor):
        """Test holiday season detection."""
        # Test holiday season dates
        holiday_dates = [
            datetime(2024, 12, 20, 12, 0, 0),  # Late December
            datetime(2024, 12, 25, 12, 0, 0),  # Christmas
            datetime(2024, 12, 31, 12, 0, 0),  # New Year's Eve
            datetime(2024, 1, 1, 12, 0, 0),  # New Year's Day
            datetime(2024, 1, 6, 12, 0, 0),  # Early January
        ]

        for holiday_date in holiday_dates:
            features = extractor._extract_seasonal_features(holiday_date)
            assert features["is_holiday_season"] == 1.0

        # Test non-holiday dates
        non_holiday_dates = [
            datetime(2024, 3, 15, 12, 0, 0),  # March
            datetime(2024, 6, 15, 12, 0, 0),  # June
            datetime(2024, 9, 15, 12, 0, 0),  # September
            datetime(2024, 12, 10, 12, 0, 0),  # Early December
            datetime(2024, 1, 15, 12, 0, 0),  # Mid January
        ]

        for non_holiday_date in non_holiday_dates:
            features = extractor._extract_seasonal_features(non_holiday_date)
            assert features["is_holiday_season"] == 0.0


class TestContextualFeatureExtractorEdgeCases:
    """Additional edge case tests for ContextualFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create a contextual feature extractor instance."""
        return ContextualFeatureExtractor()

    def test_no_room_states(self, extractor, target_time):
        """Test feature extraction when no room states are provided."""
        features = extractor.extract_features([], None, target_time)

        # Should handle None room states gracefully
        assert isinstance(features, dict)
        # Multi-room features should have defaults when no room states
        assert features["simultaneous_occupancy_ratio"] == 0.0

    def test_mixed_sensor_types(self, extractor):
        """Test extraction with mixed environmental sensor types."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        events = []

        # Mixed sensor events with different identification patterns
        mixed_sensors = [
            ("23.5", "climate_sensor", {"temperature": 23.5}),
            ("45.0", "humidity_detector", {"humidity": 45.0}),
            ("500", "light_sensor_kitchen", {"illuminance": 500}),
            ("24.0", "temp_living_room", {"value": 24.0}),
        ]

        for i, (state, sensor_id, attributes) in enumerate(mixed_sensors):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.room_id = "test_room"
            event.state = state
            event.sensor_type = "sensor"  # Generic type
            event.sensor_id = sensor_id
            event.attributes = attributes
            events.append(event)

        features = extractor._extract_environmental_features(events, target_time)

        # Should detect environmental values even with mixed sensor types
        assert "current_temperature" in features
        assert "current_humidity" in features
        assert "current_light" in features
