"""
Unit tests for sequential feature extraction.

This module tests the SequentialFeatureExtractor for movement patterns,
room transitions, velocity analysis, and human vs cat classification.
"""

from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest

from src.core.config import RoomConfig, SystemConfig
from src.core.exceptions import FeatureExtractionError
from src.data.ingestion.event_processor import (
    MovementPatternClassifier,
    MovementSequence,
)
from src.data.storage.models import SensorEvent
from src.features.sequential import SequentialFeatureExtractor


class TestSequentialFeatureExtractor:
    """Test suite for SequentialFeatureExtractor."""

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
        """Create a sequential feature extractor instance."""
        return SequentialFeatureExtractor(config=mock_config)

    @pytest.fixture
    def extractor_no_config(self):
        """Create extractor without configuration for edge case testing."""
        return SequentialFeatureExtractor(config=None)

    @pytest.fixture
    def room_configs(self):
        """Create room configuration dictionary."""
        configs = {}
        for room_id in ["living_room", "kitchen", "bedroom"]:
            config = Mock(spec=RoomConfig)
            config.room_id = room_id
            config.get_sensors_by_type = Mock(
                return_value={"door": f"sensor.{room_id}_door"}
            )
            configs[room_id] = config
        return configs

    @pytest.fixture
    def multi_room_events(self) -> List[SensorEvent]:
        """Create events across multiple rooms for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Living room sequence
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 2)
            event.room_id = "living_room"
            event.state = "on" if i % 2 == 0 else "of"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.living_room_motion_{i}"
            events.append(event)

        # Kitchen sequence (with transition)
        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=10 + i)
            event.room_id = "kitchen"
            event.state = "on"
            event.sensor_type = "presence" if i < 2 else "door"
            event.sensor_id = f"sensor.kitchen_{event.sensor_type}_{i}"
            events.append(event)

        # Back to living room
        for i in range(2):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=15 + i)
            event.room_id = "living_room"
            event.state = "on"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.living_room_motion_return_{i}"
            events.append(event)

        return events

    @pytest.fixture
    def single_room_events(self) -> List[SensorEvent]:
        """Create events within a single room."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        sensor_types = ["motion", "presence", "door", "motion", "presence"]
        for i, sensor_type in enumerate(sensor_types):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 3)
            event.room_id = "living_room"
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = sensor_type
            event.sensor_id = f"sensor.living_room_{sensor_type}_{i}"
            events.append(event)

        return events

    @pytest.fixture
    def rapid_sequence_events(self) -> List[SensorEvent]:
        """Create rapid sequence of events for burst testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Create burst of events within 30 seconds
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(seconds=i * 2)
            event.room_id = "living_room"
            event.state = "on"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        # Then a long pause (15 minutes)
        event = Mock(spec=SensorEvent)
        event.timestamp = base_time + timedelta(minutes=15)
        event.room_id = "living_room"
        event.state = "of"
        event.sensor_type = "motion"
        event.sensor_id = "sensor.motion_pause"
        events.append(event)

        return events

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    def test_extract_features_multi_room(
        self, extractor, multi_room_events, room_configs, target_time
    ):
        """Test feature extraction with multi-room events."""
        features = extractor.extract_features(
            multi_room_events, target_time, room_configs, lookback_hours=2
        )

        # Verify basic feature structure
        assert isinstance(features, dict)
        assert len(features) > 20  # Should have many sequential features

        # Check key features exist
        expected_features = [
            "room_transition_count",
            "unique_rooms_visited",
            "room_revisit_ratio",
            "avg_event_interval",
            "movement_velocity_score",
            "unique_sensors_triggered",
            "active_room_count",
            "multi_room_sequence_ratio",
        ]

        for feature in expected_features:
            assert feature in features
            assert isinstance(features[feature], (int, float))

    def test_extract_features_single_room(
        self, extractor, single_room_events, room_configs, target_time
    ):
        """Test feature extraction with single room events."""
        features = extractor.extract_features(
            single_room_events, target_time, room_configs, lookback_hours=2
        )

        # Should handle single room gracefully
        assert features["unique_rooms_visited"] == 1.0
        assert features["room_transition_count"] == 0.0
        assert features["active_room_count"] == 1.0

    def test_extract_features_empty_events(self, extractor, target_time):
        """Test behavior with empty event list."""
        features = extractor.extract_features([], target_time)

        # Should return default features
        expected_defaults = extractor._get_default_features()
        assert features == expected_defaults

    def test_room_transition_features(self, extractor, multi_room_events):
        """Test room transition feature calculations."""
        features = extractor._extract_room_transition_features(multi_room_events)

        # Should detect room transitions
        assert features["room_transition_count"] > 0
        assert features["unique_rooms_visited"] >= 2  # living_room and kitchen

        # Values should be reasonable
        assert features["room_revisit_ratio"] >= 0.0
        assert features["avg_room_dwell_time"] > 0
        assert features["max_room_sequence_length"] >= 1.0

    def test_velocity_features(self, extractor, rapid_sequence_events):
        """Test movement velocity feature calculations."""
        features = extractor._extract_velocity_features(rapid_sequence_events)

        # Check velocity features
        assert "avg_event_interval" in features
        assert "min_event_interval" in features
        assert "max_event_interval" in features
        assert "movement_velocity_score" in features
        assert "burst_ratio" in features
        assert "pause_ratio" in features

        # Should detect burst activity (many events within 30 seconds)
        assert features["burst_ratio"] > 0.5

        # Should have reasonable intervals
        assert features["min_event_interval"] >= 0
        assert features["max_event_interval"] > features["min_event_interval"]

    def test_sensor_sequence_features(self, extractor, single_room_events):
        """Test sensor sequence feature calculations."""
        features = extractor._extract_sensor_sequence_features(single_room_events)

        # Check sensor features
        assert "unique_sensors_triggered" in features
        assert "sensor_revisit_count" in features
        assert "dominant_sensor_ratio" in features
        assert "sensor_diversity_score" in features
        assert "presence_sensor_ratio" in features
        assert "door_sensor_ratio" in features

        # Values should be reasonable
        assert features["unique_sensors_triggered"] >= 1
        assert 0.0 <= features["dominant_sensor_ratio"] <= 1.0
        assert 0.0 <= features["sensor_diversity_score"] <= 1.0

    def test_cross_room_features(self, extractor, multi_room_events):
        """Test cross-room correlation features."""
        features = extractor._extract_cross_room_features(multi_room_events)

        # Check cross-room features
        assert "active_room_count" in features
        assert "room_correlation_score" in features
        assert "multi_room_sequence_ratio" in features
        assert "room_switching_frequency" in features

        # Should detect multiple active rooms
        assert features["active_room_count"] >= 2  # living_room and kitchen
        assert features["multi_room_sequence_ratio"] > 0  # Has room transitions

    @patch("src.features.sequential.MovementPatternClassifier")
    def test_movement_classification_features(
        self, mock_classifier_class, extractor, multi_room_events, room_configs
    ):
        """Test movement classification features."""
        # Mock classifier instance and its classify method
        mock_classifier = Mock()
        mock_classifier_class.return_value = mock_classifier

        # Set up extractor with mock classifier
        extractor.classifier = mock_classifier

        # Mock classification result
        mock_classification = Mock()
        mock_classification.is_human_triggered = True
        mock_classification.confidence_score = 0.8
        mock_classifier.classify_movement.return_value = mock_classification

        features = extractor._extract_movement_classification_features(
            multi_room_events, room_configs
        )

        # Check classification features
        assert "human_movement_probability" in features
        assert "cat_movement_probability" in features
        assert "movement_confidence_score" in features
        assert "door_interaction_ratio" in features

        # Values should be reasonable
        assert 0.0 <= features["human_movement_probability"] <= 1.0
        assert 0.0 <= features["cat_movement_probability"] <= 1.0

    def test_ngram_features(self, extractor, single_room_events):
        """Test n-gram pattern feature extraction."""
        features = extractor._extract_ngram_features(single_room_events)

        # Check n-gram features
        assert "common_bigram_ratio" in features
        assert "common_trigram_ratio" in features
        assert "pattern_repetition_score" in features

        # Values should be between 0 and 1
        assert 0.0 <= features["common_bigram_ratio"] <= 1.0
        assert 0.0 <= features["common_trigram_ratio"] <= 1.0
        assert 0.0 <= features["pattern_repetition_score"] <= 1.0

    def test_create_sequences_for_classification(
        self, extractor, multi_room_events, room_configs
    ):
        """Test movement sequence creation for classification."""
        sequences = extractor._create_sequences_for_classification(
            multi_room_events, room_configs
        )

        # Should create sequences grouped by room and time gaps
        assert len(sequences) > 0
        assert all(isinstance(seq, MovementSequence) for seq in sequences)

    def test_create_movement_sequence(self, extractor):
        """Test individual movement sequence creation."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        for i in range(3):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.room_id = "living_room"
            event.state = "on"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        sequence = extractor._create_movement_sequence(events)

        assert sequence is not None
        assert isinstance(sequence, MovementSequence)
        assert sequence.start_time == events[0].timestamp
        assert sequence.end_time == events[-1].timestamp
        assert len(sequence.rooms_visited) == 1
        assert "living_room" in sequence.rooms_visited

    def test_feature_names_method(self, extractor):
        """Test get_feature_names method."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 20

        # Should match default features keys
        default_features = extractor._get_default_features()
        assert set(feature_names) == set(default_features.keys())

    def test_cache_operations(self, extractor):
        """Test cache clear functionality."""
        # Add something to cache
        extractor.sequence_cache["test"] = "value"
        assert "test" in extractor.sequence_cache

        # Clear cache
        extractor.clear_cache()
        assert len(extractor.sequence_cache) == 0

    @pytest.mark.parametrize("lookback_hours", [1, 6, 12, 24, 48])
    def test_different_lookback_windows(
        self, extractor, multi_room_events, target_time, lookback_hours
    ):
        """Test feature extraction with different lookback windows."""
        features = extractor.extract_features(
            multi_room_events, target_time, lookback_hours=lookback_hours
        )

        # Should return valid features regardless of lookback window
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_edge_case_single_event(self, extractor):
        """Test handling of single event."""
        event = Mock(spec=SensorEvent)
        event.timestamp = datetime(2024, 1, 15, 14, 0, 0)
        event.room_id = "living_room"
        event.state = "on"
        event.sensor_type = "motion"
        event.sensor_id = "sensor.motion"

        target_time = datetime(2024, 1, 15, 15, 0, 0)
        features = extractor.extract_features([event], target_time)

        # Should handle single event gracefully
        assert features["unique_rooms_visited"] == 1.0
        assert features["room_transition_count"] == 0.0

    def test_edge_case_duplicate_sensors(self, extractor):
        """Test handling of events from same sensor."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Multiple events from same sensor
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.room_id = "living_room"
            event.state = "on" if i % 2 == 0 else "of"
            event.sensor_type = "motion"
            event.sensor_id = "sensor.same_motion"  # Same sensor ID
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Should handle duplicate sensors
        assert features["unique_sensors_triggered"] == 1.0
        assert (
            features["sensor_revisit_count"] == 4.0
        )  # 5 total - 1 initial = 4 revisits

    def test_empty_room_configs(self, extractor, multi_room_events, target_time):
        """Test behavior with empty room configurations."""
        features = extractor.extract_features(
            multi_room_events, target_time, room_configs={}, lookback_hours=2
        )

        # Should still work but movement classification features may be defaults
        assert isinstance(features, dict)
        assert "human_movement_probability" in features
        assert "cat_movement_probability" in features

    def test_no_classifier_available(
        self, extractor_no_config, multi_room_events, target_time
    ):
        """Test behavior when no classifier is available."""
        features = extractor_no_config.extract_features(
            multi_room_events, target_time, room_configs={}, lookback_hours=2
        )

        # Should return default classification features
        assert features["human_movement_probability"] == 0.5
        assert features["cat_movement_probability"] == 0.5
        assert features["movement_confidence_score"] == 0.5

    def test_time_filtering_accuracy(self, extractor, target_time):
        """Test that time filtering works correctly."""
        # Create events both inside and outside lookback window
        lookback_hours = 2
        cutoff_time = target_time - timedelta(hours=lookback_hours)

        # Event too old (should be filtered out)
        old_event = Mock(spec=SensorEvent)
        old_event.timestamp = cutoff_time - timedelta(minutes=30)
        old_event.room_id = "old_room"
        old_event.state = "on"
        old_event.sensor_type = "motion"
        old_event.sensor_id = "sensor.old"

        # Event within window (should be included)
        new_event = Mock(spec=SensorEvent)
        new_event.timestamp = cutoff_time + timedelta(minutes=30)
        new_event.room_id = "new_room"
        new_event.state = "on"
        new_event.sensor_type = "motion"
        new_event.sensor_id = "sensor.new"

        events = [old_event, new_event]
        features = extractor.extract_features(
            events, target_time, lookback_hours=lookback_hours
        )

        # Should only count the new event
        assert features["unique_rooms_visited"] == 1.0

    def test_statistical_accuracy_intervals(self, extractor):
        """Test statistical accuracy of interval calculations."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Create events with known 5-minute intervals
        for i in range(6):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.room_id = "test_room"
            event.state = "on"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        features = extractor._extract_velocity_features(events)

        # Average interval should be 5 minutes (300 seconds)
        expected_interval = 300.0
        assert abs(features["avg_event_interval"] - expected_interval) < 1.0
        assert features["min_event_interval"] == expected_interval
        assert features["max_event_interval"] == expected_interval

    def test_performance_large_sequences(self, extractor):
        """Test performance with large event sequences."""
        import time

        # Create large event sequence
        base_time = datetime(2024, 1, 15, 0, 0, 0)
        target_time = datetime(2024, 1, 15, 23, 59, 59)
        large_events = []

        rooms = ["room_1", "room_2", "room_3"]
        for i in range(5000):  # Large number of events
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(
                minutes=i * 0.3
            )  # ~18 second intervals
            event.room_id = rooms[i % 3]
            event.state = "on" if i % 2 == 0 else "of"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i % 10}"
            large_events.append(event)

        # Measure extraction time
        start_time = time.time()
        features = extractor.extract_features(
            large_events, target_time, lookback_hours=24
        )
        extraction_time = time.time() - start_time

        # Should complete in reasonable time (< 5 seconds)
        assert extraction_time < 5.0
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_error_handling(self, extractor, target_time):
        """Test error handling in feature extraction."""
        # Test with malformed events
        bad_events = [Mock(spec=SensorEvent)]
        bad_events[0].timestamp = None  # This should cause an error
        bad_events[0].room_id = "test_room"
        bad_events[0].state = "on"
        bad_events[0].sensor_type = "motion"
        bad_events[0].sensor_id = "sensor.test"

        with pytest.raises(FeatureExtractionError):
            extractor.extract_features(bad_events, target_time)

    def test_sensor_diversity_calculation(self, extractor):
        """Test sensor diversity score calculation accuracy."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Create events with known sensor distribution
        sensor_counts = {
            "sensor_1": 4,
            "sensor_2": 2,
            "sensor_3": 2,
        }  # Total: 8 events

        event_index = 0
        for sensor_id, count in sensor_counts.items():
            for _ in range(count):
                event = Mock(spec=SensorEvent)
                event.timestamp = base_time + timedelta(minutes=event_index)
                event.room_id = "test_room"
                event.state = "on"
                event.sensor_type = "motion"
                event.sensor_id = sensor_id
                events.append(event)
                event_index += 1

        features = extractor._extract_sensor_sequence_features(events)

        # Calculate expected entropy and diversity
        total_events = 8
        expected_entropy = 0
        for count in sensor_counts.values():
            p = count / total_events
            expected_entropy -= p * (math.log2(p) if p > 0 else 0)

        max_entropy = math.log2(len(sensor_counts))
        expected_diversity = expected_entropy / max_entropy

        # Check diversity calculation accuracy
        import math

        assert abs(features["sensor_diversity_score"] - expected_diversity) < 0.01

    @pytest.mark.asyncio
    async def test_concurrent_extraction(
        self, extractor, multi_room_events, target_time
    ):
        """Test thread safety of feature extraction."""
        import asyncio

        async def extract_features():
            return extractor.extract_features(multi_room_events, target_time)

        # Run multiple extractions concurrently
        tasks = [extract_features() for _ in range(5)]
        results = await asyncio.gather(*tasks)

        # All results should be identical
        first_result = results[0]
        for result in results[1:]:
            assert result == first_result


class TestSequentialFeatureExtractorMovementPatterns:
    """Additional tests for movement pattern analysis."""

    @pytest.fixture
    def extractor(self):
        """Create extractor for movement pattern testing."""
        return SequentialFeatureExtractor()

    def test_human_like_patterns(self, extractor):
        """Test detection of human-like movement patterns."""
        # Create human-like pattern: slower, more deliberate movements
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Slower transitions between rooms
        rooms = ["living_room", "kitchen", "bedroom"]
        for i, room in enumerate(rooms):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(
                minutes=i * 10
            )  # 10-minute intervals
            event.room_id = room
            event.state = "on"
            event.sensor_type = "presence"
            event.sensor_id = f"sensor.{room}_presence"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Human patterns should show:
        # - Reasonable movement velocity (not too fast)
        # - Lower burst ratio (not rapid-fire movements)
        assert features["movement_velocity_score"] < 0.8  # Not too fast
        assert features["burst_ratio"] < 0.2  # Low burst activity

    def test_cat_like_patterns(self, extractor):
        """Test detection of cat-like movement patterns."""
        # Create cat-like pattern: rapid, erratic movements
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = []

        # Rapid movements with quick returns
        rooms = [
            "living_room",
            "kitchen",
            "living_room",
            "bedroom",
            "living_room",
        ]
        for i, room in enumerate(rooms):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(
                seconds=i * 30
            )  # 30-second intervals
            event.room_id = room
            event.state = "on"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.{room}_motion_{i}"
            events.append(event)

        features = extractor.extract_features(events, target_time)

        # Cat patterns should show:
        # - High movement velocity (rapid movements)
        # - High room revisit ratio
        # - High burst ratio
        assert features["movement_velocity_score"] > 0.6  # Fast movement
        assert features["room_revisit_ratio"] > 0.5  # Frequent returns to same room

    def test_door_interaction_patterns(self, extractor):
        """Test door interaction feature calculations."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        # Mix of door and non-door sensors
        sensor_configs = [
            ("motion", "motion"),
            ("door", "door"),
            ("presence", "presence"),
            ("door", "door"),
            ("motion", "motion"),
        ]

        for i, (sensor_type, sensor_name) in enumerate(sensor_configs):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i)
            event.room_id = "test_room"
            event.state = "on"
            event.sensor_type = sensor_type
            event.sensor_id = f"sensor.test_{sensor_name}_{i}"
            events.append(event)

        features = extractor._extract_sensor_sequence_features(events)

        # Should have 40% door sensor ratio (2 out of 5)
        expected_door_ratio = 0.4
        assert abs(features["door_sensor_ratio"] - expected_door_ratio) < 0.01
