"""
Comprehensive unit tests for sequential feature extraction.

This module provides exhaustive testing of the SequentialFeatureExtractor class,
covering movement patterns, velocity calculations, room transitions, n-gram analysis,
human vs cat classification, and advanced mathematical validation.
"""

from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, Mock, call, patch

import numpy as np
import pytest
import statistics

from src.core.config import RoomConfig, SystemConfig
from src.core.constants import (
    CAT_MOVEMENT_PATTERNS,
    HUMAN_MOVEMENT_PATTERNS,
    MAX_SEQUENCE_GAP,
    MIN_EVENT_SEPARATION,
    SensorType,
)
from src.core.exceptions import FeatureExtractionError
from src.data.ingestion.event_processor import (
    MovementPatternClassifier,
    MovementSequence,
)
from src.data.storage.models import SensorEvent
from src.features.sequential import SequentialFeatureExtractor


class TestSequentialFeatureExtractorComprehensive:
    """Comprehensive test suite for SequentialFeatureExtractor with production-grade validation."""

    @pytest.fixture
    def extractor(self):
        """Create a sequential feature extractor instance."""
        return SequentialFeatureExtractor()

    @pytest.fixture
    def system_config(self) -> SystemConfig:
        """Create a system configuration for testing."""
        config = Mock(spec=SystemConfig)
        config.rooms = {
            "living_room": self.create_room_config("living_room"),
            "kitchen": self.create_room_config("kitchen"),
            "bedroom": self.create_room_config("bedroom"),
            "hallway": self.create_room_config("hallway"),
        }
        return config

    @pytest.fixture
    def extractor_with_config(self, system_config):
        """Create extractor with configuration and classifier."""
        extractor = SequentialFeatureExtractor(system_config)
        return extractor

    def create_room_config(self, room_id: str) -> RoomConfig:
        """Create a room configuration for testing."""
        room_config = Mock(spec=RoomConfig)
        room_config.room_id = room_id
        room_config.name = room_id.replace("_", " ").title()
        room_config.sensors = {
            "motion": f"sensor.{room_id}_motion",
            "presence": f"sensor.{room_id}_presence",
            "door": f"sensor.{room_id}_door",
            "temperature": f"sensor.{room_id}_temperature",
        }

        def get_sensors_by_type(sensor_type: str) -> Dict[str, str]:
            return {k: v for k, v in room_config.sensors.items() if k == sensor_type}

        room_config.get_sensors_by_type = get_sensors_by_type
        return room_config

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

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
    def room_transition_events(self) -> List[SensorEvent]:
        """Create events showing room transition patterns."""
        base_time = datetime(2024, 1, 15, 10, 0, 0)
        events = []

        # Simulate person moving: bedroom -> bathroom -> kitchen -> living_room -> bedroom
        rooms_sequence = ["bedroom", "bathroom", "kitchen", "living_room", "bedroom"]
        dwell_times = [30, 10, 20, 45, 60]  # Minutes in each room

        current_time = base_time
        for i, (room, dwell) in enumerate(zip(rooms_sequence, dwell_times)):
            # Entry event
            event = self.create_sensor_event(
                timestamp=current_time,
                sensor_id=f"sensor.{room}_motion",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

            # Some activity during dwell time
            for j in range(1, min(dwell // 10, 4)):  # Max 3 additional events per room
                mid_event = self.create_sensor_event(
                    timestamp=current_time + timedelta(minutes=j * (dwell // 4)),
                    sensor_id=f"sensor.{room}_presence",
                    sensor_type="presence",
                    state="on",
                    room_id=room,
                )
                events.append(mid_event)

            current_time += timedelta(minutes=dwell)

        return events

    @pytest.fixture
    def velocity_pattern_events(self) -> List[SensorEvent]:
        """Create events with specific velocity patterns for testing."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Pattern 1: Fast burst of activity (cat-like)
        for i in range(8):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 15),  # 15-second intervals
                sensor_id="sensor.living_room_motion",
                sensor_type="motion",
                state="on",
                room_id="living_room",
            )
            events.append(event)

        # Pattern 2: Long pause
        pause_start = base_time + timedelta(minutes=10)

        # Pattern 3: Slow, regular activity (human-like)
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=pause_start + timedelta(minutes=i * 5),  # 5-minute intervals
                sensor_id="sensor.kitchen_motion",
                sensor_type="motion",
                state="on",
                room_id="kitchen",
            )
            events.append(event)

        return events

    @pytest.fixture
    def complex_movement_events(self) -> List[SensorEvent]:
        """Create complex movement patterns for advanced testing."""
        base_time = datetime(2024, 1, 15, 8, 0, 0)
        events = []

        # Morning routine: bedroom -> bathroom -> kitchen -> living_room
        morning_sequence = [
            ("bedroom", 0, "motion"),
            ("bedroom", 2, "presence"),
            ("bathroom", 15, "door"),
            ("bathroom", 16, "motion"),
            ("bathroom", 25, "door"),
            ("kitchen", 30, "motion"),
            ("kitchen", 35, "presence"),
            ("kitchen", 45, "door"),  # Refrigerator?
            ("living_room", 50, "motion"),
            ("living_room", 55, "presence"),
        ]

        for room, minutes, sensor_type in morning_sequence:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor.{room}_{sensor_type}",
                sensor_type=sensor_type,
                state="on",
                room_id=room,
            )
            events.append(event)

        # Add some revisits to test revisit patterns
        revisit_events = [
            ("kitchen", 65, "motion"),  # Return to kitchen
            ("kitchen", 70, "motion"),
            ("living_room", 80, "presence"),  # Return to living room
            ("bedroom", 120, "motion"),  # Return to bedroom
        ]

        for room, minutes, sensor_type in revisit_events:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor.{room}_{sensor_type}",
                sensor_type=sensor_type,
                state="on",
                room_id=room,
            )
            events.append(event)

        return events

    @pytest.fixture
    def n_gram_pattern_events(self) -> List[SensorEvent]:
        """Create events with repeating n-gram patterns."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Repeating pattern: A -> B -> C, A -> B -> C, A -> B -> C
        pattern = ["sensor_A", "sensor_B", "sensor_C"]

        for cycle in range(4):  # 4 complete cycles
            for i, sensor_id in enumerate(pattern):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=cycle * 15 + i * 3),
                    sensor_id=sensor_id,
                    sensor_type="motion",
                    state="on",
                    room_id="test_room",
                )
                events.append(event)

        # Add some noise to test pattern detection robustness
        noise_events = [
            ("sensor_D", 65),
            ("sensor_A", 70),  # Start of another A->B->C but incomplete
            ("sensor_B", 73),
        ]

        for sensor_id, minutes in noise_events:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        return events

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization_default(self):
        """Test default initialization."""
        extractor = SequentialFeatureExtractor()

        assert extractor.config is None
        assert extractor.classifier is None
        assert isinstance(extractor.sequence_cache, dict)

    def test_initialization_with_config(self, system_config):
        """Test initialization with system configuration."""
        extractor = SequentialFeatureExtractor(system_config)

        assert extractor.config == system_config
        assert isinstance(extractor.classifier, MovementPatternClassifier)
        assert isinstance(extractor.sequence_cache, dict)

    def test_initialization_cache_structure(self, extractor):
        """Test that cache is properly initialized."""
        assert hasattr(extractor, "sequence_cache")
        assert isinstance(extractor.sequence_cache, dict)
        assert len(extractor.sequence_cache) == 0

    # ==================== ROOM TRANSITION FEATURES TESTS ====================

    def test_room_transition_comprehensive(
        self, extractor, room_transition_events, target_time
    ):
        """Test comprehensive room transition analysis."""
        features = extractor._extract_room_transition_features(room_transition_events)

        # Verify all transition features are present
        required_features = [
            "room_transition_count",
            "unique_rooms_visited",
            "room_revisit_ratio",
            "avg_room_dwell_time",
            "max_room_sequence_length",
            "transition_regularity",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert features[feature] >= 0

    def test_room_transition_count_accuracy(self, extractor):
        """Test accuracy of room transition counting."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create specific transition pattern: A -> B -> C -> A -> B
        transitions = ["room_a", "room_b", "room_c", "room_a", "room_b"]
        events = []

        for i, room in enumerate(transitions):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

        features = extractor._extract_room_transition_features(events)

        # Should count 4 transitions (5 rooms - 1)
        assert features["room_transition_count"] == 4.0
        assert features["unique_rooms_visited"] == 3.0  # A, B, C

        # Calculate expected revisit ratio: room_a appears 2 times, room_b appears 2 times
        # Total visits = 5, revisits = (2-1) + (2-1) = 2, ratio = 2/5 = 0.4
        expected_revisit_ratio = 2.0 / 5.0
        assert abs(features["room_revisit_ratio"] - expected_revisit_ratio) < 0.01

    def test_room_dwell_time_calculation(self, extractor):
        """Test mathematical accuracy of room dwell time calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Pattern: room_a (10min) -> room_b (20min) -> room_a (15min)
        events = [
            self.create_sensor_event(base_time, "sensor_a", "motion", "on", "room_a"),
            self.create_sensor_event(
                base_time + timedelta(minutes=10), "sensor_b", "motion", "on", "room_b"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=30), "sensor_a", "motion", "on", "room_a"
            ),
            self.create_sensor_event(
                base_time + timedelta(minutes=45),
                "end_marker",
                "motion",
                "on",
                "other_room",
            ),
        ]

        features = extractor._extract_room_transition_features(events)

        # Expected dwell times: room_a = 10min, room_b = 20min, room_a = 15min
        # Average = (10 + 20 + 15) / 3 = 15 minutes = 900 seconds
        expected_avg_dwell = 900.0  # seconds

        assert (
            abs(features["avg_room_dwell_time"] - expected_avg_dwell) < 60
        )  # Allow 1 minute tolerance

    def test_max_sequence_length_detection(self, extractor):
        """Test detection of maximum consecutive room sequence length."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Pattern: A, A, A, B, B, C, C, C, C, A
        room_sequence = [
            "room_a",
            "room_a",
            "room_a",
            "room_b",
            "room_b",
            "room_c",
            "room_c",
            "room_c",
            "room_c",
            "room_a",
        ]
        events = []

        for i, room in enumerate(room_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 5),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

        features = extractor._extract_room_transition_features(events)

        # Maximum consecutive sequence should be 4 (room_c appears 4 times in a row)
        assert features["max_room_sequence_length"] == 4.0

    def test_transition_regularity_calculation(self, extractor):
        """Test transition regularity calculation using coefficient of variation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with regular intervals (low CV = high regularity)
        regular_intervals = [0, 10, 20, 30, 40]  # Perfect 10-minute intervals
        regular_events = []

        for i, minutes in enumerate(regular_intervals):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id=f"room_{i}",
            )
            regular_events.append(event)

        regular_features = extractor._extract_room_transition_features(regular_events)

        # Should have high regularity (close to 1.0)
        assert regular_features["transition_regularity"] > 0.8

        # Test with irregular intervals (high CV = low regularity)
        irregular_intervals = [0, 5, 25, 30, 60]  # Highly irregular
        irregular_events = []

        for i, minutes in enumerate(irregular_intervals):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id=f"room_{i}",
            )
            irregular_events.append(event)

        irregular_features = extractor._extract_room_transition_features(
            irregular_events
        )

        # Should have low regularity
        assert irregular_features["transition_regularity"] < 0.5

    def test_room_transition_edge_cases(self, extractor):
        """Test edge cases in room transition analysis."""
        # Test with single event
        single_event = [
            self.create_sensor_event(
                datetime(2024, 1, 15, 12, 0, 0), "sensor1", "motion", "on", "room1"
            )
        ]

        features_single = extractor._extract_room_transition_features(single_event)

        assert features_single["room_transition_count"] == 0.0
        assert features_single["unique_rooms_visited"] == 1.0
        assert features_single["room_revisit_ratio"] == 0.0

        # Test with no transitions (all events from same room)
        same_room_events = []
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 10),
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="same_room",
            )
            same_room_events.append(event)

        features_same = extractor._extract_room_transition_features(same_room_events)

        assert features_same["room_transition_count"] == 0.0
        assert features_same["unique_rooms_visited"] == 1.0
        assert features_same["max_room_sequence_length"] == len(same_room_events)

    # ==================== VELOCITY FEATURES TESTS ====================

    def test_velocity_features_comprehensive(
        self, extractor, velocity_pattern_events, target_time
    ):
        """Test comprehensive velocity feature extraction."""
        features = extractor._extract_velocity_features(velocity_pattern_events)

        # Verify all velocity features are present
        required_features = [
            "avg_event_interval",
            "min_event_interval",
            "max_event_interval",
            "event_interval_variance",
            "movement_velocity_score",
            "burst_ratio",
            "pause_ratio",
            "velocity_acceleration",
            "interval_autocorr",
            "velocity_entropy",
            "movement_regularity",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_velocity_mathematical_accuracy(self, extractor):
        """Test mathematical accuracy of velocity calculations using numpy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events with known intervals: 60s, 120s, 180s, 240s
        intervals_seconds = [60, 120, 180, 240]
        events = []

        current_time = base_time
        events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in intervals_seconds:
            current_time += timedelta(seconds=interval)
            events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        features = extractor._extract_velocity_features(events)

        # Verify calculations using numpy
        intervals_array = np.array(intervals_seconds, dtype=np.float64)

        expected_avg = float(np.mean(intervals_array))
        expected_min = float(np.min(intervals_array))
        expected_max = float(np.max(intervals_array))
        expected_var = float(np.var(intervals_array))

        assert abs(features["avg_event_interval"] - expected_avg) < 1.0
        assert abs(features["min_event_interval"] - expected_min) < 1.0
        assert abs(features["max_event_interval"] - expected_max) < 1.0
        assert abs(features["event_interval_variance"] - expected_var) < 100.0

    def test_movement_velocity_score_calculation(self, extractor):
        """Test movement velocity score calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test fast movement (high velocity score)
        fast_events = []
        for i in range(10):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 30),  # 30-second intervals
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            fast_events.append(event)

        fast_features = extractor._extract_velocity_features(fast_events)

        # Fast movement should have high velocity score
        assert fast_features["movement_velocity_score"] > 0.8

        # Test slow movement (low velocity score)
        slow_events = []
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 20),  # 20-minute intervals
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            slow_events.append(event)

        slow_features = extractor._extract_velocity_features(slow_events)

        # Slow movement should have low velocity score
        assert slow_features["movement_velocity_score"] < 0.3

    def test_burst_and_pause_detection(self, extractor):
        """Test burst and pause detection algorithms."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create pattern with clear bursts and pauses
        events = []

        # Burst pattern: 5 events in 2 minutes (24-second intervals = burst)
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 24),
                sensor_id=f"burst_sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            events.append(event)

        # Long pause: 15 minutes
        pause_start = base_time + timedelta(minutes=15)

        # Normal activity: 3 events with 5-minute intervals
        for i in range(3):
            event = self.create_sensor_event(
                timestamp=pause_start + timedelta(minutes=i * 5),
                sensor_id=f"normal_sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            events.append(event)

        features = extractor._extract_velocity_features(events)

        # Should detect both bursts and pauses
        assert features["burst_ratio"] > 0.3  # Some intervals are bursts (<30s)
        assert features["pause_ratio"] > 0.1  # Some intervals are pauses (>600s)

    def test_velocity_acceleration_calculation(self, extractor):
        """Test velocity acceleration (second derivative) calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create accelerating pattern: intervals get shorter over time
        intervals = [300, 240, 180, 120, 60]  # Accelerating (decreasing intervals)
        events = []

        current_time = base_time
        events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in intervals:
            current_time += timedelta(seconds=interval)
            events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        features = extractor._extract_velocity_features(events)

        # Should detect acceleration (decreasing intervals = increasing velocity)
        assert features["velocity_acceleration"] > 0  # Some variation in acceleration

        # Test constant velocity (no acceleration)
        constant_intervals = [120, 120, 120, 120]  # Constant intervals
        constant_events = []

        current_time = base_time
        constant_events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in constant_intervals:
            current_time += timedelta(seconds=interval)
            constant_events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        constant_features = extractor._extract_velocity_features(constant_events)

        # Should have zero or very low acceleration
        assert constant_features["velocity_acceleration"] < 10.0  # Nearly constant

    def test_interval_autocorrelation_analysis(self, extractor):
        """Test interval autocorrelation for pattern regularity."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create repeating pattern: short, long, short, long
        pattern = [60, 300, 60, 300, 60, 300, 60, 300]  # Alternating pattern
        events = []

        current_time = base_time
        events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in pattern:
            current_time += timedelta(seconds=interval)
            events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        features = extractor._extract_velocity_features(events)

        # Alternating pattern should have negative autocorrelation
        assert features["interval_autocorr"] < 0.1  # Pattern should be detected

        # Test random intervals (should have low autocorrelation)
        random_intervals = [45, 120, 300, 80, 200, 150, 90]  # Random
        random_events = []

        current_time = base_time
        random_events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in random_intervals:
            current_time += timedelta(seconds=interval)
            random_events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        random_features = extractor._extract_velocity_features(random_events)

        # Random pattern should have low autocorrelation
        assert abs(random_features["interval_autocorr"]) < 0.5

    def test_velocity_entropy_calculation(self, extractor):
        """Test velocity entropy calculation for movement unpredictability."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test high entropy (very random intervals)
        random_intervals = np.random.randint(30, 600, 20).tolist()  # Random intervals
        high_entropy_events = []

        current_time = base_time
        high_entropy_events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in random_intervals:
            current_time += timedelta(seconds=interval)
            high_entropy_events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        high_entropy_features = extractor._extract_velocity_features(
            high_entropy_events
        )

        # Test low entropy (repeating interval)
        constant_interval = [120] * 15  # Same interval repeated
        low_entropy_events = []

        current_time = base_time
        low_entropy_events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in constant_interval:
            current_time += timedelta(seconds=interval)
            low_entropy_events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        low_entropy_features = extractor._extract_velocity_features(low_entropy_events)

        # High entropy should be greater than low entropy
        assert (
            high_entropy_features["velocity_entropy"]
            > low_entropy_features["velocity_entropy"]
        )

    def test_movement_regularity_coefficient_of_variation(self, extractor):
        """Test movement regularity using coefficient of variation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test highly regular movement (low CV)
        regular_interval = 120  # 2 minutes exactly
        regular_events = []

        current_time = base_time
        regular_events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for i in range(10):
            current_time += timedelta(seconds=regular_interval)
            regular_events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        regular_features = extractor._extract_velocity_features(regular_events)

        # Perfect regularity should yield high regularity score
        assert regular_features["movement_regularity"] > 0.99

        # Test irregular movement (high CV)
        irregular_intervals = [30, 300, 60, 400, 45, 350, 90, 250]  # High variation
        irregular_events = []

        current_time = base_time
        irregular_events.append(
            self.create_sensor_event(current_time, "sensor1", "motion", "on", "room1")
        )

        for interval in irregular_intervals:
            current_time += timedelta(seconds=interval)
            irregular_events.append(
                self.create_sensor_event(
                    current_time, "sensor1", "motion", "on", "room1"
                )
            )

        irregular_features = extractor._extract_velocity_features(irregular_events)

        # Irregular movement should have low regularity score
        assert irregular_features["movement_regularity"] < 0.6

    def test_velocity_empty_and_minimal_data(self, extractor):
        """Test velocity features with empty and minimal data."""
        # Test with no events
        features_empty = extractor._extract_velocity_features([])

        expected_defaults = {
            "avg_event_interval": 300.0,
            "min_event_interval": 60.0,
            "max_event_interval": 3600.0,
            "event_interval_variance": 0.0,
            "movement_velocity_score": 0.5,
            "burst_ratio": 0.0,
            "pause_ratio": 0.0,
            "velocity_acceleration": 0.0,
            "interval_autocorr": 0.0,
            "velocity_entropy": 0.0,
            "movement_regularity": 0.0,
        }

        for key, expected in expected_defaults.items():
            assert features_empty[key] == expected

        # Test with single event
        single_event = [
            self.create_sensor_event(
                datetime(2024, 1, 15, 12, 0, 0), "sensor1", "motion", "on", "room1"
            )
        ]

        features_single = extractor._extract_velocity_features(single_event)

        # Single event should return defaults
        for key, expected in expected_defaults.items():
            assert features_single[key] == expected

    # ==================== SENSOR SEQUENCE FEATURES TESTS ====================

    def test_sensor_sequence_comprehensive(self, extractor):
        """Test comprehensive sensor sequence analysis."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create diverse sensor usage pattern
        sensors = [
            "sensor_a",
            "sensor_b",
            "sensor_c",
            "sensor_a",
            "sensor_b",
            "sensor_a",
        ]
        events = []

        for i, sensor_id in enumerate(sensors):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 5),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_sensor_sequence_features(events)

        # Verify all sequence features
        required_features = [
            "unique_sensors_triggered",
            "sensor_revisit_count",
            "dominant_sensor_ratio",
            "sensor_diversity_score",
            "presence_sensor_ratio",
            "door_sensor_ratio",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_sensor_diversity_score_entropy(self, extractor):
        """Test sensor diversity score calculation using entropy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test maximum diversity (all sensors used equally)
        sensors = [
            "sensor_a",
            "sensor_b",
            "sensor_c",
            "sensor_d",
        ] * 3  # Each used 3 times
        diverse_events = []

        for i, sensor_id in enumerate(sensors):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            diverse_events.append(event)

        diverse_features = extractor._extract_sensor_sequence_features(diverse_events)

        # Maximum diversity should approach 1.0
        assert diverse_features["sensor_diversity_score"] > 0.95

        # Test minimum diversity (single sensor)
        single_sensor_events = []
        for i in range(12):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id="sensor_only",
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            single_sensor_events.append(event)

        single_features = extractor._extract_sensor_sequence_features(
            single_sensor_events
        )

        # Single sensor should have zero diversity
        assert single_features["sensor_diversity_score"] == 0.0

    def test_sensor_revisit_count_accuracy(self, extractor):
        """Test accuracy of sensor revisit counting."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Pattern: A(3 times), B(1 time), C(2 times) = A(2 revisits) + C(1 revisit) = 3 total revisits
        sensor_sequence = [
            "sensor_a",
            "sensor_b",
            "sensor_a",
            "sensor_c",
            "sensor_a",
            "sensor_c",
        ]
        events = []

        for i, sensor_id in enumerate(sensor_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 3),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_sensor_sequence_features(events)

        # sensor_a: 3 times = 2 revisits, sensor_c: 2 times = 1 revisit, total = 3
        assert features["sensor_revisit_count"] == 3.0
        assert features["unique_sensors_triggered"] == 3.0  # A, B, C

    def test_dominant_sensor_ratio_calculation(self, extractor):
        """Test dominant sensor ratio calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create pattern where one sensor dominates: A(6), B(2), C(2) = 60% dominance
        sensor_counts = {"sensor_a": 6, "sensor_b": 2, "sensor_c": 2}
        events = []

        event_idx = 0
        for sensor_id, count in sensor_counts.items():
            for i in range(count):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=event_idx * 2),
                    sensor_id=sensor_id,
                    sensor_type="motion",
                    state="on",
                    room_id="test_room",
                )
                events.append(event)
                event_idx += 1

        features = extractor._extract_sensor_sequence_features(events)

        # Dominant sensor (sensor_a) used 6 out of 10 times = 0.6 ratio
        expected_ratio = 6.0 / 10.0
        assert abs(features["dominant_sensor_ratio"] - expected_ratio) < 0.01

    def test_sensor_type_ratios(self, extractor):
        """Test sensor type ratio calculations."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Create events with known sensor type distribution
        sensor_types = [
            ("motion", 6),  # 60% motion sensors
            ("presence", 2),  # 20% presence sensors
            ("door", 2),  # 20% door sensors
        ]

        event_idx = 0
        for sensor_type, count in sensor_types:
            for i in range(count):
                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=event_idx * 2),
                    sensor_id=f"sensor_{sensor_type}_{i}",
                    sensor_type=sensor_type,
                    state="on",
                    room_id="test_room",
                )
                events.append(event)
                event_idx += 1

        features = extractor._extract_sensor_sequence_features(events)

        # Verify ratios: motion + presence = presence_sensor_ratio, door = door_sensor_ratio
        total_events = 10
        expected_presence_ratio = (6 + 2) / total_events  # motion + presence
        expected_door_ratio = 2 / total_events

        assert abs(features["presence_sensor_ratio"] - expected_presence_ratio) < 0.01
        assert abs(features["door_sensor_ratio"] - expected_door_ratio) < 0.01

    def test_sensor_sequence_empty_data(self, extractor):
        """Test sensor sequence features with empty data."""
        features = extractor._extract_sensor_sequence_features([])

        expected_defaults = {
            "unique_sensors_triggered": 1.0,
            "sensor_revisit_count": 0.0,
            "dominant_sensor_ratio": 1.0,
            "sensor_diversity_score": 0.0,
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    # ==================== CROSS-ROOM FEATURES TESTS ====================

    def test_cross_room_comprehensive_analysis(
        self, extractor, complex_movement_events
    ):
        """Test comprehensive cross-room correlation analysis."""
        features = extractor._extract_cross_room_features(complex_movement_events)

        # Verify all cross-room features
        required_features = [
            "active_room_count",
            "room_correlation_score",
            "multi_room_sequence_ratio",
            "room_switching_frequency",
            "room_activity_entropy",
            "spatial_clustering_score",
            "room_transition_predictability",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert features[feature] >= 0.0

    def test_room_correlation_sliding_window(self, extractor):
        """Test room correlation calculation using sliding window approach."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = []

        # Create synchronized activity in multiple rooms (high correlation)
        rooms = ["room_a", "room_b", "room_c"]

        for window in range(5):  # 5 time windows
            window_start = base_time + timedelta(minutes=window * 10)

            # In each window, all rooms have activity (perfect correlation)
            for i, room in enumerate(rooms):
                event = self.create_sensor_event(
                    timestamp=window_start + timedelta(seconds=i * 30),
                    sensor_id=f"sensor_{room}",
                    sensor_type="motion",
                    state="on",
                    room_id=room,
                )
                events.append(event)

        features = extractor._extract_cross_room_features(events)

        # Should detect high correlation (all rooms active in each window)
        assert features["room_correlation_score"] > 0.8  # High correlation
        assert features["active_room_count"] == 3.0

    def test_multi_room_sequence_ratio_accuracy(self, extractor):
        """Test multi-room sequence ratio calculation using numpy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create sequence: A, A, B, A, C, C, B -> transitions: A-B, B-A, A-C, C-B = 4 multi-room out of 6 intervals
        room_sequence = [
            "room_a",
            "room_a",
            "room_b",
            "room_a",
            "room_c",
            "room_c",
            "room_b",
        ]
        events = []

        for i, room in enumerate(room_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 5),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

        features = extractor._extract_cross_room_features(events)

        # Transitions: A-A(same), A-B(diff), B-A(diff), A-C(diff), C-C(same), C-B(diff)
        # Multi-room transitions: 4 out of 6 = 4/7 (total sequence length)
        expected_ratio = 4.0 / len(room_sequence)
        assert abs(features["multi_room_sequence_ratio"] - expected_ratio) < 0.05

    def test_room_switching_frequency_calculation(self, extractor):
        """Test room switching frequency calculation per hour."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create 2-hour period with known switches
        duration_hours = 2.0
        room_switches = [
            ("room_a", 0),  # Start
            ("room_b", 30),  # Switch 1 (30 min)
            ("room_c", 60),  # Switch 2 (1 hour)
            ("room_a", 90),  # Switch 3 (1.5 hours)
            ("room_b", 120),  # Switch 4 (2 hours)
        ]

        events = []
        for room, minutes in room_switches:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

        features = extractor._extract_cross_room_features(events)

        # 4 switches in 2 hours = 2 switches per hour
        expected_frequency = 4.0 / duration_hours
        assert abs(features["room_switching_frequency"] - expected_frequency) < 0.1

    def test_room_activity_entropy_calculation(self, extractor):
        """Test room activity entropy calculation using numpy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test balanced activity (high entropy)
        balanced_rooms = ["room_a", "room_b", "room_c"] * 4  # Each room appears 4 times
        balanced_events = []

        for i, room in enumerate(balanced_rooms):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            balanced_events.append(event)

        balanced_features = extractor._extract_cross_room_features(balanced_events)

        # Test unbalanced activity (low entropy)
        unbalanced_rooms = (
            ["room_a"] * 9 + ["room_b"] * 2 + ["room_c"] * 1
        )  # Heavily skewed
        unbalanced_events = []

        for i, room in enumerate(unbalanced_rooms):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            unbalanced_events.append(event)

        unbalanced_features = extractor._extract_cross_room_features(unbalanced_events)

        # Balanced should have higher entropy than unbalanced
        assert (
            balanced_features["room_activity_entropy"]
            > unbalanced_features["room_activity_entropy"]
        )

    def test_spatial_clustering_score_calculation(self, extractor):
        """Test spatial clustering score calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test high clustering (long runs in same room)
        clustered_sequence = ["room_a"] * 5 + ["room_b"] * 4 + ["room_c"] * 3
        clustered_events = []

        for i, room in enumerate(clustered_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            clustered_events.append(event)

        clustered_features = extractor._extract_cross_room_features(clustered_events)

        # Test low clustering (frequent room changes)
        scattered_sequence = [
            "room_a",
            "room_b",
            "room_c",
            "room_a",
            "room_b",
            "room_c",
        ] * 2
        scattered_events = []

        for i, room in enumerate(scattered_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            scattered_events.append(event)

        scattered_features = extractor._extract_cross_room_features(scattered_events)

        # Clustered should have higher clustering score than scattered
        assert (
            clustered_features["spatial_clustering_score"]
            > scattered_features["spatial_clustering_score"]
        )

    def test_room_transition_predictability_matrix(self, extractor):
        """Test room transition predictability using transition matrix."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create predictable pattern: A -> B -> C -> A (repeating cycle)
        pattern = ["room_a", "room_b", "room_c"] * 4  # 4 complete cycles
        events = []

        for i, room in enumerate(pattern):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 3),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

        features = extractor._extract_cross_room_features(events)

        # Predictable pattern should have high predictability
        assert features["room_transition_predictability"] > 0.8

        # Test random pattern (low predictability)
        random_rooms = [
            "room_a",
            "room_c",
            "room_b",
            "room_a",
            "room_b",
            "room_a",
            "room_c",
            "room_b",
        ]
        random_events = []

        for i, room in enumerate(random_rooms):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 3),
                sensor_id=f"sensor_{room}",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            random_events.append(event)

        random_features = extractor._extract_cross_room_features(random_events)

        # Random pattern should have lower predictability
        assert (
            features["room_transition_predictability"]
            > random_features["room_transition_predictability"]
        )

    def test_cross_room_single_room_handling(self, extractor):
        """Test cross-room features with single room."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        single_room_events = []
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 5),
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="single_room",
            )
            single_room_events.append(event)

        features = extractor._extract_cross_room_features(single_room_events)

        # Single room should yield appropriate default values
        expected_single_room = {
            "active_room_count": 1.0,
            "room_correlation_score": 0.0,
            "multi_room_sequence_ratio": 0.0,
            "room_switching_frequency": 0.0,
            "spatial_clustering_score": 0.0,
            "room_transition_predictability": 0.0,
        }

        for key, expected in expected_single_room.items():
            assert features[key] == expected

    # ==================== MOVEMENT CLASSIFICATION TESTS ====================

    @patch("src.data.ingestion.event_processor.MovementPatternClassifier")
    def test_movement_classification_comprehensive(
        self, mock_classifier_class, extractor_with_config, complex_movement_events
    ):
        """Test comprehensive movement classification features."""
        # Setup mock classifier
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        extractor_with_config.classifier = mock_classifier

        # Mock classification results
        mock_classification = Mock()
        mock_classification.is_human_triggered = True
        mock_classification.confidence_score = 0.85
        mock_classifier.classify_movement.return_value = mock_classification

        room_configs = {
            "bedroom": extractor_with_config.config.rooms["bedroom"],
            "kitchen": extractor_with_config.config.rooms["kitchen"],
        }

        features = extractor_with_config._extract_movement_classification_features(
            complex_movement_events, room_configs
        )

        # Verify all classification features
        required_features = [
            "human_movement_probability",
            "cat_movement_probability",
            "movement_confidence_score",
            "door_interaction_ratio",
            "pattern_matches_human",
            "pattern_matches_cat",
            "velocity_classification",
            "sequence_length_score",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0.0 <= features[feature] <= 1.0

    def test_movement_pattern_constants_validation(self, extractor_with_config):
        """Test validation against HUMAN_MOVEMENT_PATTERNS and CAT_MOVEMENT_PATTERNS constants."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create human-like pattern based on constants
        human_events = []
        human_duration = (
            HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"] + 60
        )  # Just above minimum
        human_velocity = (
            HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"] - 0.1
        )  # Just below max

        # Calculate interval to achieve desired velocity (events per second)
        interval_seconds = 1.0 / human_velocity if human_velocity > 0 else 60
        num_events = max(2, int(human_duration / interval_seconds))

        current_time = base_time
        for i in range(num_events):
            event = self.create_sensor_event(
                timestamp=current_time,
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="living_room",
            )
            human_events.append(event)
            current_time += timedelta(seconds=interval_seconds)

        # Create room config with door sensors
        room_configs = {
            "living_room": extractor_with_config.config.rooms["living_room"]
        }

        # Should create sequences that match human patterns
        sequences = extractor_with_config._create_sequences_for_classification(
            human_events, room_configs
        )

        assert len(sequences) > 0

        # Verify sequence properties align with human patterns
        for seq in sequences:
            assert (
                seq.duration_seconds >= HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"]
            )
            # Additional validations could check velocity, room count, etc.

    @patch("src.data.ingestion.event_processor.MovementPatternClassifier")
    def test_door_interaction_ratio_calculation(
        self, mock_classifier_class, extractor_with_config
    ):
        """Test door interaction ratio calculation."""
        # Setup classifier
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier
        extractor_with_config.classifier = mock_classifier

        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events with some door interactions
        events = []
        door_events_count = 0
        total_events = 10

        for i in range(total_events):
            if i % 3 == 0:  # Every 3rd event is a door event
                sensor_type = "door"
                sensor_id = "sensor.room_door"
                door_events_count += 1
            else:
                sensor_type = "motion"
                sensor_id = f"sensor.room_motion_{i}"

            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=sensor_id,
                sensor_type=sensor_type,
                state="on",
                room_id="test_room",
            )
            events.append(event)

        # Mock room config
        room_config = Mock()
        room_config.get_sensors_by_type.return_value = {"door": "sensor.room_door"}
        room_configs = {"test_room": room_config}

        # Mock classification
        mock_classification = Mock()
        mock_classification.is_human_triggered = True
        mock_classification.confidence_score = 0.8
        mock_classifier.classify_movement.return_value = mock_classification

        features = extractor_with_config._extract_movement_classification_features(
            events, room_configs
        )

        # Expected door interaction ratio
        expected_ratio = door_events_count / total_events
        assert abs(features["door_interaction_ratio"] - expected_ratio) < 0.1

    def test_movement_classification_no_classifier(self, extractor):
        """Test movement classification when no classifier is available."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        events = [
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1")
        ]
        room_configs = {"room1": Mock()}

        features = extractor._extract_movement_classification_features(
            events, room_configs
        )

        # Should return default values
        expected_defaults = {
            "human_movement_probability": 0.5,
            "cat_movement_probability": 0.5,
            "movement_confidence_score": 0.5,
            "door_interaction_ratio": 0.0,
            "pattern_matches_human": 0.0,
            "pattern_matches_cat": 0.0,
            "velocity_classification": 0.0,
            "sequence_length_score": 0.0,
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    # ==================== N-GRAM PATTERN TESTS ====================

    def test_ngram_features_comprehensive(self, extractor, n_gram_pattern_events):
        """Test comprehensive n-gram pattern analysis."""
        features = extractor._extract_ngram_features(n_gram_pattern_events)

        # Verify all n-gram features
        required_features = [
            "common_bigram_ratio",
            "common_trigram_ratio",
            "pattern_repetition_score",
        ]

        for feature in required_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))
            assert 0.0 <= features[feature] <= 1.0

    def test_bigram_pattern_detection(self, extractor):
        """Test bigram pattern detection accuracy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create repeating bigram pattern: A->B, A->B, A->B, C->D
        sensor_sequence = [
            "sensor_a",
            "sensor_b",
            "sensor_a",
            "sensor_b",
            "sensor_a",
            "sensor_b",
            "sensor_c",
            "sensor_d",
        ]
        events = []

        for i, sensor_id in enumerate(sensor_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_ngram_features(events)

        # Bigrams: A->B appears 3 times out of 7 total bigrams
        expected_bigram_ratio = 3.0 / 7.0
        assert abs(features["common_bigram_ratio"] - expected_bigram_ratio) < 0.05

    def test_trigram_pattern_detection(self, extractor):
        """Test trigram pattern detection accuracy."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create repeating trigram pattern: A->B->C, A->B->C, A->B->C, D->E->F
        sensor_sequence = [
            "sensor_a",
            "sensor_b",
            "sensor_c",
            "sensor_a",
            "sensor_b",
            "sensor_c",
            "sensor_a",
            "sensor_b",
            "sensor_c",
            "sensor_d",
            "sensor_e",
            "sensor_f",
        ]
        events = []

        for i, sensor_id in enumerate(sensor_sequence):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        features = extractor._extract_ngram_features(events)

        # Trigrams: A->B->C appears 3 times out of 10 total trigrams
        expected_trigram_ratio = 3.0 / 10.0
        assert abs(features["common_trigram_ratio"] - expected_trigram_ratio) < 0.05

    def test_pattern_repetition_score_calculation(self, extractor):
        """Test pattern repetition score calculation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # High repetition pattern
        high_pattern = ["sensor_a", "sensor_b"] * 10  # Perfect repetition
        high_events = []

        for i, sensor_id in enumerate(high_pattern):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            high_events.append(event)

        high_features = extractor._extract_ngram_features(high_events)

        # Low repetition pattern
        low_pattern = [f"sensor_{i}" for i in range(20)]  # All unique
        low_events = []

        for i, sensor_id in enumerate(low_pattern):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=sensor_id,
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            low_events.append(event)

        low_features = extractor._extract_ngram_features(low_events)

        # High repetition should have higher pattern repetition score
        assert (
            high_features["pattern_repetition_score"]
            > low_features["pattern_repetition_score"]
        )

    def test_ngram_insufficient_data(self, extractor):
        """Test n-gram features with insufficient data."""
        # Test with fewer than 3 events
        insufficient_events = [
            self.create_sensor_event(
                datetime(2024, 1, 15, 12, 0, 0), "sensor1", "motion", "on", "room1"
            ),
            self.create_sensor_event(
                datetime(2024, 1, 15, 12, 5, 0), "sensor2", "motion", "on", "room1"
            ),
        ]

        features = extractor._extract_ngram_features(insufficient_events)

        # Should return default values
        expected_defaults = {
            "common_bigram_ratio": 0.0,
            "common_trigram_ratio": 0.0,
            "pattern_repetition_score": 0.0,
        }

        for key, expected in expected_defaults.items():
            assert features[key] == expected

    # ==================== SEQUENCE CREATION TESTS ====================

    def test_sequence_creation_for_classification(self, extractor_with_config):
        """Test creation of movement sequences for classification."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events that should form valid sequences
        events = []
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 60),  # 1-minute intervals
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        room_configs = {"test_room": extractor_with_config.config.rooms["living_room"]}

        sequences = extractor_with_config._create_sequences_for_classification(
            events, room_configs
        )

        assert len(sequences) > 0

        # Verify sequence properties
        for seq in sequences:
            assert isinstance(seq, MovementSequence)
            assert len(seq.events) >= 2  # Minimum for valid sequence
            assert seq.duration_seconds >= 0
            assert len(seq.rooms_visited) > 0
            assert len(seq.sensors_triggered) > 0

    def test_sequence_gap_filtering(self, extractor_with_config):
        """Test sequence creation with MAX_SEQUENCE_GAP filtering."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events with large gap in middle
        events = [
            # First group (should be one sequence)
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(seconds=30), "sensor2", "motion", "on", "room1"
            ),
            # Large gap (exceeds MAX_SEQUENCE_GAP)
            # Second group (should be separate sequence)
            self.create_sensor_event(
                base_time + timedelta(seconds=MAX_SEQUENCE_GAP + 100),
                "sensor3",
                "motion",
                "on",
                "room1",
            ),
            self.create_sensor_event(
                base_time + timedelta(seconds=MAX_SEQUENCE_GAP + 130),
                "sensor4",
                "motion",
                "on",
                "room1",
            ),
        ]

        room_configs = {"room1": extractor_with_config.config.rooms["living_room"]}

        sequences = extractor_with_config._create_sequences_for_classification(
            events, room_configs
        )

        # Should create multiple sequences due to gap
        assert len(sequences) >= 2

    def test_min_event_separation_filtering(self, extractor_with_config):
        """Test MIN_EVENT_SEPARATION filtering in sequence creation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create events that are too close together (should be filtered)
        events = [
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(seconds=MIN_EVENT_SEPARATION - 1),
                "sensor2",
                "motion",
                "on",
                "room1",
            ),  # Too close
            self.create_sensor_event(
                base_time + timedelta(seconds=MIN_EVENT_SEPARATION + 10),
                "sensor3",
                "motion",
                "on",
                "room1",
            ),  # Far enough
        ]

        room_configs = {"room1": extractor_with_config.config.rooms["living_room"]}

        sequences = extractor_with_config._create_sequences_for_classification(
            events, room_configs
        )

        # Should filter out the too-close event
        if sequences:
            for seq in sequences:
                # Verify events in sequence respect minimum separation
                for i in range(1, len(seq.events)):
                    time_diff = (
                        seq.events[i].timestamp - seq.events[i - 1].timestamp
                    ).total_seconds()
                    assert time_diff >= MIN_EVENT_SEPARATION

    # ==================== PERFORMANCE AND EDGE CASE TESTS ====================

    def test_performance_with_large_dataset(self, extractor):
        """Test performance with large sequential datasets."""
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        target_time = base_time + timedelta(hours=12)

        # Create large dataset: 2000 events over 12 hours
        large_events = []
        rooms = ["room_a", "room_b", "room_c", "room_d"]

        for i in range(2000):
            room = rooms[i % len(rooms)]
            sensor_type = ["motion", "presence", "door"][i % 3]

            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 20),
                sensor_id=f"sensor_{room}_{sensor_type}",
                sensor_type=sensor_type,
                state="on",
                room_id=room,
            )
            large_events.append(event)

        # Measure performance
        import time

        start_time = time.time()

        features = extractor.extract_features(
            events=large_events, target_time=target_time, lookback_hours=12
        )

        end_time = time.time()
        execution_time = end_time - start_time

        # Should complete within reasonable time (2 seconds for 2000 events)
        assert (
            execution_time < 2.0
        ), f"Performance test failed: took {execution_time:.3f}s"
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extreme_velocity_patterns(self, extractor):
        """Test handling of extreme velocity patterns."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test extremely fast events (cat-like)
        ultra_fast_events = []
        for i in range(20):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(seconds=i * 5),  # 5-second intervals
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            ultra_fast_events.append(event)

        fast_features = extractor._extract_velocity_features(ultra_fast_events)

        # Should handle ultra-fast patterns
        assert fast_features["movement_velocity_score"] > 0.9  # Very high velocity
        assert fast_features["burst_ratio"] > 0.8  # Most intervals are bursts

        # Test extremely slow events (very slow human)
        ultra_slow_events = []
        for i in range(5):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(hours=i),  # 1-hour intervals
                sensor_id=f"sensor_{i}",
                sensor_type="motion",
                state="on",
                room_id="room1",
            )
            ultra_slow_events.append(event)

        slow_features = extractor._extract_velocity_features(ultra_slow_events)

        # Should handle ultra-slow patterns
        assert slow_features["movement_velocity_score"] < 0.1  # Very low velocity
        assert slow_features["pause_ratio"] > 0.8  # Most intervals are pauses

    def test_complex_room_transition_patterns(self, extractor):
        """Test complex room transition patterns."""
        base_time = datetime(2024, 1, 15, 8, 0, 0)

        # Create realistic daily routine
        daily_routine = [
            ("bedroom", 0),  # Wake up
            ("bathroom", 15),  # Morning routine
            ("kitchen", 30),  # Breakfast
            ("living_room", 60),  # Relax
            ("office", 120),  # Work
            ("kitchen", 240),  # Lunch
            ("office", 270),  # Work continues
            ("living_room", 420),  # Evening relax
            ("kitchen", 480),  # Dinner
            ("living_room", 540),  # TV time
            ("bathroom", 600),  # Night routine
            ("bedroom", 615),  # Sleep
        ]

        events = []
        for room, minutes in daily_routine:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor.{room}_motion",
                sensor_type="motion",
                state="on",
                room_id=room,
            )
            events.append(event)

        features = extractor._extract_room_transition_features(events)

        # Should handle complex realistic patterns
        assert features["unique_rooms_visited"] == 5.0  # 5 different rooms
        assert features["room_transition_count"] == len(daily_routine) - 1
        assert features["room_revisit_ratio"] > 0.0  # Some rooms revisited
        assert features["avg_room_dwell_time"] > 0

    def test_edge_case_boundary_conditions(self, extractor):
        """Test various boundary conditions."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with events at exact boundaries
        boundary_events = [
            # Event exactly at lookback boundary
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

        # Test with zero-duration sequences
        zero_duration_events = [
            self.create_sensor_event(target_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                target_time, "sensor2", "motion", "on", "room1"
            ),  # Same timestamp
        ]

        zero_features = extractor._extract_velocity_features(zero_duration_events)
        assert isinstance(zero_features, dict)

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

    # ==================== ERROR HANDLING AND INTEGRATION TESTS ====================

    def test_error_handling_comprehensive(self, extractor):
        """Test comprehensive error handling scenarios."""
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with None events
        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(None, target_time)

        # Test with malformed events (should handle gracefully)
        malformed_events = []

        # Event with missing attributes
        bad_event = Mock(spec=SensorEvent)
        bad_event.timestamp = target_time - timedelta(minutes=30)
        bad_event.sensor_id = None  # Missing
        bad_event.sensor_type = None  # Missing
        bad_event.state = None  # Missing
        bad_event.room_id = "room1"
        malformed_events.append(bad_event)

        # Should handle gracefully without crashing
        try:
            features = extractor.extract_features(malformed_events, target_time)
            assert isinstance(features, dict)
        except FeatureExtractionError:
            # Acceptable to raise specific feature extraction error
            pass

    def test_cache_functionality(self, extractor):
        """Test cache functionality and clearing."""
        # Clear cache initially
        extractor.clear_cache()
        assert len(extractor.sequence_cache) == 0

        base_time = datetime(2024, 1, 15, 12, 0, 0)
        events = [
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(minutes=5), "sensor2", "motion", "on", "room2"
            ),
        ]

        # Extract features (may populate cache)
        features1 = extractor.extract_features(events, base_time + timedelta(hours=1))

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.sequence_cache) == 0

        # Extract again after cache clear (should still work)
        features2 = extractor.extract_features(events, base_time + timedelta(hours=1))

        # Results should be consistent regardless of cache state
        assert isinstance(features1, dict)
        assert isinstance(features2, dict)

    def test_get_default_features_completeness(self, extractor):
        """Test that default features are complete."""
        defaults = extractor._get_default_features()

        # Should include all major feature categories
        expected_categories = [
            # Room transition features
            "room_transition_count",
            "unique_rooms_visited",
            # Velocity features
            "avg_event_interval",
            "movement_velocity_score",
            "burst_ratio",
            # Sensor sequence features
            "unique_sensors_triggered",
            "sensor_diversity_score",
            # Cross-room features
            "active_room_count",
            "room_correlation_score",
            # Classification features
            "human_movement_probability",
            "cat_movement_probability",
            # N-gram features
            "common_bigram_ratio",
            "pattern_repetition_score",
        ]

        for feature in expected_categories:
            assert feature in defaults, f"Missing default for {feature}"
            assert isinstance(defaults[feature], (int, float))

    def test_get_feature_names_method(self, extractor):
        """Test get_feature_names method completeness."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 30  # Should have many features

        # Should match default features
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_realistic_home_automation_scenario(self, extractor_with_config):
        """Test with realistic home automation scenario."""
        # Simulate a typical morning routine
        base_time = datetime(2024, 1, 15, 7, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Create realistic movement pattern
        morning_routine = [
            # Wake up in bedroom
            ("bedroom", 0, "motion"),
            ("bedroom", 2, "presence"),
            # Go to bathroom
            ("bathroom", 10, "door"),
            ("bathroom", 12, "motion"),
            ("bathroom", 20, "door"),
            # Go to kitchen for breakfast
            ("kitchen", 25, "motion"),
            ("kitchen", 30, "door"),  # Refrigerator
            ("kitchen", 35, "motion"),
            # Move to living room
            ("living_room", 50, "motion"),
            ("living_room", 55, "presence"),
            # Return to bedroom to get ready
            ("bedroom", 70, "motion"),
            ("bedroom", 80, "presence"),
            # Final trip to kitchen
            ("kitchen", 90, "motion"),
        ]

        events = []
        for room, minutes, sensor_type in morning_routine:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minutes),
                sensor_id=f"sensor.{room}_{sensor_type}",
                sensor_type=sensor_type,
                state="on",
                room_id=room,
            )
            events.append(event)

        # Extract comprehensive features
        features = extractor_with_config.extract_features(
            events, target_time, lookback_hours=2
        )

        # Verify realistic results
        assert isinstance(features, dict)
        assert len(features) > 20  # Should extract many features

        # Should detect multiple rooms
        if "unique_rooms_visited" in features:
            assert (
                features["unique_rooms_visited"] >= 4
            )  # bedroom, bathroom, kitchen, living_room

        # Should detect room transitions
        if "room_transition_count" in features:
            assert features["room_transition_count"] >= 5  # Multiple transitions

        # Should detect revisits (bedroom and kitchen visited twice)
        if "room_revisit_ratio" in features:
            assert features["room_revisit_ratio"] > 0.0

    def test_movement_sequence_mathematical_validation(self, extractor_with_config):
        """Test mathematical validation of movement sequence creation."""
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Create precisely timed events
        events = [
            self.create_sensor_event(base_time, "sensor1", "motion", "on", "room1"),
            self.create_sensor_event(
                base_time + timedelta(seconds=30), "sensor2", "motion", "on", "room1"
            ),
            self.create_sensor_event(
                base_time + timedelta(seconds=60), "sensor3", "motion", "on", "room1"
            ),
        ]

        room_configs = {"room1": extractor_with_config.config.rooms["living_room"]}

        sequence = extractor_with_config._create_movement_sequence(events)

        # Verify mathematical properties
        assert sequence is not None
        assert sequence.start_time == events[0].timestamp
        assert sequence.end_time == events[-1].timestamp
        assert sequence.duration_seconds == 60.0  # Exactly 1 minute
        assert "room1" in sequence.rooms_visited
        assert len(sequence.sensors_triggered) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
