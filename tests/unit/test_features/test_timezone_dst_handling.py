"""
Comprehensive tests for timezone transitions and DST handling in feature extraction.

This test suite validates feature computation accuracy across:
- Daylight Saving Time transitions (spring forward, fall back)
- Timezone changes
- Cross-timezone event correlation
- International timezone support
- Edge cases during time transitions
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List
from unittest.mock import Mock

import numpy as np
import pytest
from zoneinfo import ZoneInfo

from src.core.config import RoomConfig
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.temporal import TemporalFeatureExtractor


class TestDaylightSavingTimeTransitions:
    """Test feature extraction during DST transitions."""

    @pytest.fixture
    def dst_spring_forward_events(self):
        """Create events around DST spring forward transition."""
        events = []
        # DST Spring Forward: 2024-03-10 2:00 AM -> 3:00 AM (PST to PDT)
        base_time = datetime(2024, 3, 10, 1, 0, 0)  # 1 AM before transition

        # Events before DST transition
        for i in range(60):  # 1 hour before transition
            event = Mock(spec=SensorEvent)
            event.room_id = "living_room"
            event.sensor_id = "motion_sensor_1"
            event.sensor_type = "motion"
            event.state = "on" if i % 8 < 4 else "off"
            event.timestamp = base_time + timedelta(minutes=i)
            event.attributes = {"battery": 90.0}
            events.append(event)

        # Skip the "lost hour" (2:00 AM - 3:00 AM doesn't exist)

        # Events after DST transition
        transition_time = datetime(2024, 3, 10, 3, 0, 0)  # 3 AM after transition
        for i in range(120):  # 2 hours after transition
            event = Mock(spec=SensorEvent)
            event.room_id = "living_room"
            event.sensor_id = "motion_sensor_1"
            event.sensor_type = "motion"
            event.state = "on" if i % 6 < 3 else "off"
            event.timestamp = transition_time + timedelta(minutes=i)
            event.attributes = {"battery": 89.0}
            events.append(event)

        return events

    @pytest.fixture
    def dst_fall_back_events(self):
        """Create events around DST fall back transition."""
        events = []
        # DST Fall Back: 2024-11-03 2:00 AM -> 1:00 AM (PDT to PST)
        base_time = datetime(2024, 11, 3, 0, 30, 0)  # 12:30 AM before transition

        # Events before first 1:00-2:00 AM hour
        for i in range(30):  # 30 minutes
            event = Mock(spec=SensorEvent)
            event.room_id = "bedroom"
            event.sensor_id = "motion_sensor_2"
            event.sensor_type = "motion"
            event.state = "on" if i % 5 < 2 else "off"
            event.timestamp = base_time + timedelta(minutes=i)
            event.attributes = {"battery": 85.0}
            events.append(event)

        # First 1:00-2:00 AM hour (PDT)
        first_hour_start = datetime(2024, 11, 3, 1, 0, 0)
        for i in range(60):
            event = Mock(spec=SensorEvent)
            event.room_id = "bedroom"
            event.sensor_id = "motion_sensor_2"
            event.sensor_type = "motion"
            event.state = "on" if i % 7 < 3 else "off"
            event.timestamp = first_hour_start + timedelta(minutes=i)
            event.attributes = {"battery": 84.0, "dst_flag": "PDT"}
            events.append(event)

        # Second 1:00-2:00 AM hour (PST - repeated hour)
        second_hour_start = datetime(
            2024, 11, 3, 1, 0, 0
        )  # Same clock time, different DST
        for i in range(60):
            event = Mock(spec=SensorEvent)
            event.room_id = "bedroom"
            event.sensor_id = "motion_sensor_2"
            event.sensor_type = "motion"
            event.state = "on" if i % 4 == 0 else "off"
            event.timestamp = second_hour_start + timedelta(minutes=i)
            event.attributes = {"battery": 83.0, "dst_flag": "PST"}
            events.append(event)

        return events

    def test_temporal_features_spring_forward_transition(
        self, dst_spring_forward_events
    ):
        """Test temporal features during DST spring forward (lost hour)."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)  # PST base

        # Target time after DST transition
        target_time = datetime(2024, 3, 10, 5, 0, 0)  # 5 AM PDT

        features = extractor.extract_features(dst_spring_forward_events, target_time)

        # Should handle missing hour gracefully
        assert len(features) > 15, "Should extract full feature set despite lost hour"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "No NaN values during DST"

        # Time-based features should account for DST gap
        assert "time_since_last_change" in features
        time_since = features["time_since_last_change"]
        assert time_since >= 0, "Time since should be non-negative"

        # Cyclical features should handle DST adjustment
        assert "cyclical_hour_sin" in features
        assert "cyclical_hour_cos" in features
        hour_sin = features["cyclical_hour_sin"]
        hour_cos = features["cyclical_hour_cos"]
        assert -1 <= hour_sin <= 1, "Hour sine should be valid"
        assert -1 <= hour_cos <= 1, "Hour cosine should be valid"

        # Historical patterns should adapt to DST
        assert "historical_occupancy_rate" in features
        assert (
            0 <= features["historical_occupancy_rate"] <= 1
        ), "Occupancy rate should be valid"

    def test_temporal_features_fall_back_transition(self, dst_fall_back_events):
        """Test temporal features during DST fall back (repeated hour)."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)

        # Target time after both repeated hours
        target_time = datetime(2024, 11, 3, 3, 0, 0)  # 3 AM PST

        features = extractor.extract_features(dst_fall_back_events, target_time)

        # Should handle repeated hour gracefully
        assert len(features) > 15, "Should extract features despite repeated hour"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "No NaN values during fall back"

        # Duration features should handle overlapping times
        assert "duration_in_current_state" in features
        duration = features["duration_in_current_state"]
        assert duration >= 0, "Duration should be non-negative despite time overlap"

        # Transition timing should handle repeated hour
        assert "transition_rate" in features
        transition_rate = features["transition_rate"]
        assert transition_rate >= 0, "Transition rate should be non-negative"

    def test_sequential_features_across_dst_boundaries(self, dst_spring_forward_events):
        """Test sequential features across DST boundaries."""
        extractor = SequentialFeatureExtractor()

        # Add cross-room events during DST transition
        cross_room_events = []
        base_time = datetime(2024, 3, 10, 1, 30, 0)
        rooms = ["living_room", "kitchen", "bedroom"]

        for i, room in enumerate(rooms * 20):  # 60 events across rooms
            event = Mock(spec=SensorEvent)
            event.room_id = room
            event.sensor_id = f"sensor_{room}"
            event.sensor_type = "motion"
            event.state = "on" if i % 3 == 0 else "off"

            # Time progression that crosses DST boundary
            if i < 30:
                event.timestamp = base_time + timedelta(minutes=i)
            else:
                # After DST transition (skip lost hour)
                event.timestamp = datetime(2024, 3, 10, 3, 0, 0) + timedelta(
                    minutes=i - 30
                )

            cross_room_events.append(event)

        all_events = dst_spring_forward_events + cross_room_events
        room_configs = {room: Mock(spec=RoomConfig) for room in rooms}
        target_time = datetime(2024, 3, 10, 4, 0, 0)

        features = extractor.extract_features(all_events, room_configs, target_time)

        # Should compute room transitions across DST
        assert len(features) > 10, "Should extract sequential features across DST"
        assert "room_transition_count" in features
        assert (
            features["room_transition_count"] >= 0
        ), "Should count transitions across DST"

        # Movement patterns should handle time gap
        assert "movement_velocity_avg" in features
        velocity = features["movement_velocity_avg"]
        assert velocity >= 0, "Velocity should be non-negative"

    def test_contextual_features_dst_environmental_correlation(self):
        """Test contextual features with environmental changes during DST."""
        extractor = ContextualFeatureExtractor()

        events = []
        room_states = []

        # Simulate environmental changes around DST (spring forward)
        base_time = datetime(2024, 3, 10, 1, 0, 0)

        for i in range(100):
            event = Mock(spec=SensorEvent)
            event.room_id = "sunroom"
            event.sensor_type = ["motion", "temperature", "light"][i % 3]
            event.state = "on" if i % 5 < 3 else "off"

            # Time progression across DST
            if i < 50:
                event.timestamp = base_time + timedelta(minutes=i * 2)
            else:
                # After DST (skip lost hour)
                event.timestamp = datetime(2024, 3, 10, 3, 0, 0) + timedelta(
                    minutes=(i - 50) * 2
                )

            # Environmental attributes affected by DST/sunrise
            hour = event.timestamp.hour
            if event.sensor_type == "temperature":
                # Temperature rise after sunrise (affected by DST)
                temp = 18.0 + max(0, (hour - 6) * 2)  # Warmer after 6 AM
                event.attributes = {"temperature": min(temp, 25.0)}
            elif event.sensor_type == "light":
                # Light levels change with DST
                light = max(0, (hour - 5) * 100)  # Brighter after 5 AM
                event.attributes = {"light_level": min(light, 800.0)}
            else:
                event.attributes = {"battery": 90.0}

            events.append(event)

        # Room states across DST transition
        for i in range(0, 100, 20):
            state = Mock(spec=RoomState)
            state.room_id = "sunroom"
            state.state = "occupied" if i % 40 < 20 else "vacant"

            if i < 50:
                state.timestamp = base_time + timedelta(minutes=i * 2)
            else:
                state.timestamp = datetime(2024, 3, 10, 3, 0, 0) + timedelta(
                    minutes=(i - 50) * 2
                )

            room_states.append(state)

        features = extractor.extract_features(events, room_states)

        # Should correlate environmental changes with DST
        assert len(features) > 15, "Should extract environmental features across DST"

        # Environmental features should be valid
        env_features = [
            k
            for k in features.keys()
            if any(env in k for env in ["temperature", "light", "environmental"])
        ]
        assert len(env_features) > 0, "Should have environmental features"

        for feature_name in env_features:
            value = features[feature_name]
            assert not np.isnan(
                value
            ), f"Environmental feature {feature_name} should not be NaN"


class TestCrossTimezoneScenarios:
    """Test feature extraction across different timezones."""

    @pytest.fixture
    def multi_timezone_events(self):
        """Create events simulating multiple timezone scenarios."""
        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)  # UTC noon

        # Simulate events from different "timezones"
        timezone_offsets = {
            "pst_room": -8,  # Pacific Standard Time
            "est_room": -5,  # Eastern Standard Time
            "utc_room": 0,  # UTC
            "cet_room": +1,  # Central European Time
            "jst_room": +9,  # Japan Standard Time
        }

        for room_id, tz_offset in timezone_offsets.items():
            for i in range(20):
                event = Mock(spec=SensorEvent)
                event.room_id = room_id
                event.sensor_id = f"sensor_{room_id}"
                event.sensor_type = "motion"
                event.state = "on" if i % 4 < 2 else "off"

                # Adjust timestamp for timezone simulation
                local_time = base_time + timedelta(hours=tz_offset, minutes=i * 3)
                event.timestamp = local_time
                event.attributes = {"timezone_offset": tz_offset}
                events.append(event)

        return events

    def test_temporal_features_cross_timezone_correlation(self, multi_timezone_events):
        """Test temporal features with cross-timezone event correlation."""
        # Test with different timezone contexts
        timezone_offsets = [-8, -5, 0, +1, +9]

        for tz_offset in timezone_offsets:
            extractor = TemporalFeatureExtractor(timezone_offset=tz_offset)
            target_time = datetime(2024, 6, 15, 18, 0, 0)  # 6 PM in extractor timezone

            features = extractor.extract_features(multi_timezone_events, target_time)

            # Should extract features regardless of timezone context
            assert (
                len(features) > 10
            ), f"Should extract features with tz_offset {tz_offset}"
            assert all(
                not np.isnan(v) for v in features.values()
            ), "No NaN values across timezones"

            # Cyclical features should be relative to extractor timezone
            assert "cyclical_hour_sin" in features
            assert "cyclical_hour_cos" in features

            # Time-based features should handle timezone differences
            assert "time_since_last_change" in features
            assert (
                features["time_since_last_change"] >= 0
            ), "Time since should be non-negative"

    def test_sequential_features_global_room_transitions(self, multi_timezone_events):
        """Test sequential features with global room transitions."""
        extractor = SequentialFeatureExtractor()

        room_configs = {
            "pst_room": Mock(spec=RoomConfig),
            "est_room": Mock(spec=RoomConfig),
            "utc_room": Mock(spec=RoomConfig),
            "cet_room": Mock(spec=RoomConfig),
            "jst_room": Mock(spec=RoomConfig),
        }

        target_time = datetime(2024, 6, 15, 15, 0, 0)  # 3 PM UTC

        features = extractor.extract_features(
            multi_timezone_events, room_configs, target_time
        )

        # Should handle global room transitions
        assert len(features) > 8, "Should extract features from global room set"
        assert "room_transition_count" in features
        assert features["room_transition_count"] >= 0, "Should count global transitions"

        # Cross-room correlation should work across timezones
        assert "cross_room_correlation_strength" in features
        correlation = features["cross_room_correlation_strength"]
        assert -1 <= correlation <= 1, "Correlation should be valid"

    def test_timezone_offset_edge_cases(self):
        """Test feature extraction with extreme timezone offsets."""
        extreme_offsets = [-12, -11, +12, +14]  # Extreme but valid timezone offsets

        events = []
        base_time = datetime(2024, 6, 15, 12, 0, 0)

        for i in range(30):
            event = Mock(spec=SensorEvent)
            event.room_id = "extreme_tz_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 3 == 0 else "off"
            event.timestamp = base_time + timedelta(minutes=i * 2)
            events.append(event)

        for tz_offset in extreme_offsets:
            extractor = TemporalFeatureExtractor(timezone_offset=tz_offset)
            target_time = base_time + timedelta(hours=2)

            features = extractor.extract_features(events, target_time)

            # Should handle extreme timezones
            assert len(features) > 5, f"Should work with extreme offset {tz_offset}"
            assert all(
                not np.isnan(v) for v in features.values()
            ), "No NaN with extreme offsets"

            # Cyclical features should remain valid
            if "cyclical_hour_sin" in features and "cyclical_hour_cos" in features:
                hour_sin = features["cyclical_hour_sin"]
                hour_cos = features["cyclical_hour_cos"]
                # Verify unit circle property
                magnitude = hour_sin**2 + hour_cos**2
                assert (
                    abs(magnitude - 1.0) < 0.01
                ), "Cyclical features should maintain unit circle"

    def test_timezone_change_during_operation(self):
        """Test feature extraction when timezone changes during operation."""
        events = []
        base_time = datetime(2024, 6, 15, 10, 0, 0)

        # Create events spanning time when timezone might change
        for i in range(50):
            event = Mock(spec=SensorEvent)
            event.room_id = "mobile_room"  # Simulating mobile/traveling scenario
            event.sensor_type = "motion"
            event.state = "on" if i % 5 < 3 else "off"
            event.timestamp = base_time + timedelta(minutes=i * 6)  # 5 hours total
            events.append(event)

        # Extract features with different timezone contexts
        extractors = [
            TemporalFeatureExtractor(timezone_offset=-8),  # Start in PST
            TemporalFeatureExtractor(timezone_offset=-5),  # Move to EST
            TemporalFeatureExtractor(timezone_offset=0),  # End in UTC
        ]

        target_time = base_time + timedelta(hours=6)

        all_features = []
        for extractor in extractors:
            features = extractor.extract_features(events, target_time)
            all_features.append(features)

            # Each should work independently
            assert len(features) > 5, "Should extract features in each timezone"
            assert all(
                not np.isnan(v) for v in features.values()
            ), "No NaN in any timezone"

        # Core features should be consistent across timezones
        core_features = ["time_since_last_change", "duration_in_current_state"]
        for feature_name in core_features:
            if all(feature_name in features for features in all_features):
                values = [features[feature_name] for features in all_features]
                # Should be reasonably consistent (within 1 hour difference)
                max_diff = max(values) - min(values)
                assert (
                    max_diff < 1.0
                ), f"Core feature {feature_name} should be consistent across timezones"


class TestTimezoneEdgeCasesAndErrorHandling:
    """Test edge cases and error handling in timezone operations."""

    def test_invalid_timezone_offset_handling(self):
        """Test handling of invalid timezone offsets."""
        # Test with invalid (but possibly encountered) timezone offsets
        invalid_offsets = [-25, 25, 100, -100]

        events = [Mock(spec=SensorEvent) for _ in range(5)]
        target_time = datetime(2024, 6, 15, 12, 0, 0)

        for i, event in enumerate(events):
            event.room_id = "test_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 2 == 0 else "off"
            event.timestamp = target_time - timedelta(minutes=i * 10)

        for invalid_offset in invalid_offsets:
            extractor = TemporalFeatureExtractor(timezone_offset=invalid_offset)

            # Should handle gracefully (possibly with clamping or defaults)
            features = extractor.extract_features(events, target_time)

            assert len(features) > 0, f"Should handle invalid offset {invalid_offset}"
            assert all(
                not np.isnan(v) for v in features.values()
            ), "Should not produce NaN"

    def test_leap_year_dst_interaction(self):
        """Test DST handling during leap year."""
        # 2024 is a leap year - test DST around Feb 29
        extractor = TemporalFeatureExtractor(timezone_offset=-5)

        events = []
        # Events around leap day and later DST transition
        dates = [
            datetime(2024, 2, 28, 12, 0, 0),  # Day before leap day
            datetime(2024, 2, 29, 12, 0, 0),  # Leap day
            datetime(2024, 3, 1, 12, 0, 0),  # Day after leap day
            datetime(2024, 3, 10, 3, 0, 0),  # DST transition
        ]

        for i, date in enumerate(dates):
            for j in range(10):
                event = Mock(spec=SensorEvent)
                event.room_id = "leap_year_room"
                event.sensor_type = "motion"
                event.state = "on" if (i + j) % 3 == 0 else "off"
                event.timestamp = date + timedelta(minutes=j * 6)
                events.append(event)

        target_time = datetime(2024, 3, 11, 12, 0, 0)  # After DST transition

        features = extractor.extract_features(events, target_time)

        # Should handle leap year + DST combination
        assert len(features) > 10, "Should extract features across leap year + DST"
        assert all(not np.isnan(v) for v in features.values()), "No NaN values"

        # Date-based cyclical features should handle leap year
        if "cyclical_day_of_year_sin" in features:
            day_sin = features["cyclical_day_of_year_sin"]
            assert -1 <= day_sin <= 1, "Day of year cyclical should be valid"

    def test_dst_transition_with_rapid_events(self):
        """Test DST transition handling with rapid event sequences."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)

        events = []
        # Very rapid events around DST spring forward
        base_time = datetime(2024, 3, 10, 1, 59, 0)  # 1:59 AM before transition

        # Events every 10 seconds leading up to and after DST
        for i in range(120):  # 20 minutes of 10-second intervals
            event = Mock(spec=SensorEvent)
            event.room_id = "rapid_dst_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 6 < 3 else "off"

            if i < 60:  # Before DST (1:59-2:00 AM)
                event.timestamp = base_time + timedelta(seconds=i * 10)
            else:  # After DST (3:00-3:20 AM)
                dst_time = datetime(2024, 3, 10, 3, 0, 0)
                event.timestamp = dst_time + timedelta(seconds=(i - 60) * 10)

            events.append(event)

        target_time = datetime(2024, 3, 10, 3, 30, 0)  # 3:30 AM after DST

        features = extractor.extract_features(events, target_time)

        # Should handle rapid events across DST
        assert len(features) > 10, "Should extract features from rapid DST events"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "No NaN with rapid DST events"

        # Transition rate should be computed correctly
        if "transition_rate" in features:
            rate = features["transition_rate"]
            assert rate >= 0, "Transition rate should be non-negative"

    def test_mixed_timezone_aware_naive_events(self):
        """Test handling of mixed timezone-aware and naive datetime events."""
        extractor = TemporalFeatureExtractor(timezone_offset=-7)

        events = []
        base_time = datetime(2024, 6, 15, 14, 0, 0)

        for i in range(20):
            event = Mock(spec=SensorEvent)
            event.room_id = "mixed_tz_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 4 < 2 else "off"

            if i % 3 == 0:
                # Timezone-naive datetime
                event.timestamp = base_time + timedelta(minutes=i * 3)
            elif i % 3 == 1:
                # Timezone-aware datetime (UTC)
                naive_time = base_time + timedelta(minutes=i * 3)
                event.timestamp = naive_time.replace(tzinfo=timezone.utc)
            else:
                # Timezone-aware datetime (different timezone)
                naive_time = base_time + timedelta(minutes=i * 3)
                pst = timezone(timedelta(hours=-8))
                event.timestamp = naive_time.replace(tzinfo=pst)

            events.append(event)

        target_time = base_time + timedelta(hours=1)

        # Should handle mixed timezone scenarios
        features = extractor.extract_features(events, target_time)

        assert len(features) > 5, "Should handle mixed timezone-aware/naive events"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should normalize timezone handling"
