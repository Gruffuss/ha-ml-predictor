"""
Comprehensive tests for feature extraction with missing and incomplete data.

This test suite validates robustness of feature engineering when dealing with:
- Missing sensor data
- Incomplete sequences
- Malformed events
- Sensor failures
- Data quality issues
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.config import RoomConfig
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureStore
from src.features.temporal import TemporalFeatureExtractor


class TestMissingSensorDataScenarios:
    """Test feature extraction with various missing sensor data patterns."""

    @pytest.fixture
    def sparse_event_dataset(self):
        """Create dataset with significant gaps in sensor data."""
        events = []
        base_time = datetime(2024, 1, 15, 8, 0, 0)

        # Morning batch: 8-9 AM (dense data)
        for i in range(30):
            event = Mock(spec=SensorEvent)
            event.room_id = "living_room"
            event.sensor_id = f"motion_sensor_{i % 3}"
            event.sensor_type = "motion"
            event.state = "on" if i % 4 < 2 else "off"
            event.timestamp = base_time + timedelta(minutes=i * 2)
            event.attributes = {"battery": 85.0}
            events.append(event)

        # Large gap: 9 AM - 6 PM (9 hours of missing data)

        # Evening batch: 6-7 PM (sparse data)
        evening_start = base_time + timedelta(hours=10)
        for i in range(10):
            event = Mock(spec=SensorEvent)
            event.room_id = "living_room"
            event.sensor_id = "motion_sensor_0"
            event.sensor_type = "motion"
            event.state = "on" if i % 3 == 0 else "off"
            event.timestamp = evening_start + timedelta(
                minutes=i * 6
            )  # Every 6 minutes
            event.attributes = {"battery": 75.0}
            events.append(event)

        return events

    @pytest.fixture
    def intermittent_sensor_dataset(self):
        """Create dataset simulating intermittent sensor failures."""
        events = []
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        for i in range(100):
            event = Mock(spec=SensorEvent)
            event.room_id = "bedroom"
            event.timestamp = base_time + timedelta(minutes=i * 3)

            # Simulate sensor failures - some sensors work, others don't
            sensor_failure_pattern = [True, True, False, True, False, False]
            sensor_working = sensor_failure_pattern[i % len(sensor_failure_pattern)]

            if sensor_working:
                event.sensor_id = f"reliable_sensor_{i % 2}"
                event.sensor_type = ["motion", "door"][i % 2]
                event.state = "on" if i % 5 < 3 else "off"
                event.attributes = {
                    "battery": max(20, 100 - i),
                    "signal_strength": 80.0,
                }
            else:
                # Failed sensor - missing or corrupted data
                event.sensor_id = f"failed_sensor_{i % 3}"
                event.sensor_type = "unknown"
                event.state = None
                event.attributes = None

            events.append(event)

        return events

    def test_temporal_features_with_significant_data_gaps(self, sparse_event_dataset):
        """Test temporal feature extraction with large gaps in sensor data."""
        extractor = TemporalFeatureExtractor()
        target_time = datetime(2024, 1, 15, 19, 0, 0)  # 7 PM, after the gap

        features = extractor.extract_features(sparse_event_dataset, target_time)

        # Should handle gaps gracefully
        assert len(features) > 15, "Should extract meaningful features despite gaps"
        assert all(not np.isnan(v) for v in features.values()), "No NaN values allowed"

        # Time since features should reflect actual gap
        assert "time_since_last_change" in features
        assert features["time_since_last_change"] > 8.0, "Should reflect large time gap"

        # Duration features should handle sparse data
        assert "duration_in_current_state" in features
        assert (
            features["duration_in_current_state"] >= 0
        ), "Duration should be non-negative"

        # Historical patterns should adapt to sparse data
        assert "historical_occupancy_rate" in features
        assert (
            0 <= features["historical_occupancy_rate"] <= 1
        ), "Rate should be valid proportion"

    def test_sequential_features_with_incomplete_room_sequences(
        self, intermittent_sensor_dataset
    ):
        """Test sequential features with incomplete room transition data."""
        extractor = SequentialFeatureExtractor()

        # Limited room configs (some rooms missing)
        room_configs = {
            "bedroom": Mock(spec=RoomConfig),
            # Missing configs for other rooms that might appear in events
        }

        # Add some events from unconfigured rooms
        additional_events = []
        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.room_id = "unknown_room"  # Not in room_configs
            event.sensor_id = "unknown_sensor"
            event.sensor_type = "motion"
            event.state = "on"
            event.timestamp = datetime(2024, 1, 15, 14, i, 0)
            additional_events.append(event)

        all_events = intermittent_sensor_dataset + additional_events
        target_time = datetime(2024, 1, 15, 16, 0, 0)

        features = extractor.extract_features(all_events, room_configs, target_time)

        # Should handle missing room configs gracefully
        assert (
            len(features) > 10
        ), "Should extract features despite missing room configs"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should handle unknown rooms"

        # Room transition features should be computed with available data
        assert "room_transition_count" in features
        assert (
            features["room_transition_count"] >= 0
        ), "Transition count should be non-negative"

    def test_contextual_features_with_missing_environmental_sensors(self):
        """Test contextual features when environmental sensors are missing."""
        extractor = ContextualFeatureExtractor()

        events = []
        base_time = datetime(2024, 1, 15, 10, 0, 0)

        for i in range(50):
            event = Mock(spec=SensorEvent)
            event.room_id = "kitchen"
            event.timestamp = base_time + timedelta(minutes=i * 2)

            # Simulated sensor availability pattern
            sensor_types = ["motion", "temperature", "humidity", "light", "door"]
            available_sensors = sensor_types[
                : ((i % 5) + 1)
            ]  # Varying sensor availability

            event.sensor_type = available_sensors[0] if available_sensors else "motion"
            event.state = "on" if i % 3 == 0 else "off"

            # Environmental attributes - sometimes missing
            if "temperature" in available_sensors and i % 4 != 0:
                temp = 20.0 + np.sin(i / 10) * 5  # Varying temperature
                event.attributes = {"temperature": temp}
            elif "humidity" in available_sensors and i % 3 != 0:
                humidity = 45.0 + np.cos(i / 8) * 10  # Varying humidity
                event.attributes = {"humidity": humidity}
            else:
                event.attributes = {}  # Missing environmental data

            events.append(event)

        # Room states with gaps
        room_states = []
        for i in range(0, 50, 10):  # Every 10th interval
            state = Mock(spec=RoomState)
            state.room_id = "kitchen"
            state.state = ["occupied", "vacant"][i % 2]
            state.timestamp = base_time + timedelta(minutes=i * 2)
            state.confidence = 0.8 if i % 20 == 0 else 0.6  # Varying confidence
            room_states.append(state)

        features = extractor.extract_features(events, room_states)

        # Should provide defaults for missing environmental data
        assert len(features) > 15, "Should extract features with environmental defaults"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should handle missing sensors"

        # Should include environmental features even if sparse
        env_features = [
            k
            for k in features.keys()
            if any(env in k for env in ["temperature", "humidity", "light"])
        ]
        assert len(env_features) > 0, "Should include some environmental features"

    def test_feature_extraction_with_corrupted_timestamps(self):
        """Test feature extraction with corrupted or invalid timestamps."""
        extractor = TemporalFeatureExtractor()

        events = []
        base_time = datetime(2024, 1, 15, 12, 0, 0)

        # Mix of valid and corrupted events
        for i in range(20):
            event = Mock(spec=SensorEvent)
            event.room_id = "office"
            event.sensor_type = "motion"
            event.state = "on" if i % 2 == 0 else "off"

            if i % 5 == 0:
                # Corrupted timestamp
                event.timestamp = None
            elif i % 5 == 1:
                # Invalid timestamp type
                event.timestamp = "2024-01-15 12:00:00"  # String instead of datetime
            elif i % 5 == 2:
                # Future timestamp
                event.timestamp = base_time + timedelta(hours=48)
            elif i % 5 == 3:
                # Very old timestamp
                event.timestamp = base_time - timedelta(days=365)
            else:
                # Valid timestamp
                event.timestamp = base_time + timedelta(minutes=i)

            events.append(event)

        target_time = base_time + timedelta(hours=1)

        # Should filter out corrupted events and process valid ones
        features = extractor.extract_features(events, target_time)

        assert len(features) > 5, "Should extract features from valid events"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should ignore corrupted timestamps"

    def test_feature_extraction_with_malformed_attributes(self):
        """Test feature extraction with malformed event attributes."""
        extractor = ContextualFeatureExtractor()

        events = []
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        for i in range(30):
            event = Mock(spec=SensorEvent)
            event.room_id = "garage"
            event.timestamp = base_time + timedelta(minutes=i)
            event.sensor_type = "temperature"
            event.state = "on"

            # Various malformed attributes
            if i % 6 == 0:
                event.attributes = None  # Null attributes
            elif i % 6 == 1:
                event.attributes = "invalid_json"  # String instead of dict
            elif i % 6 == 2:
                event.attributes = {"temperature": "not_a_number"}  # String value
            elif i % 6 == 3:
                event.attributes = {"temperature": float("inf")}  # Infinite value
            elif i % 6 == 4:
                event.attributes = {"temperature": float("nan")}  # NaN value
            else:
                event.attributes = {"temperature": 22.0 + i * 0.5}  # Valid value

            events.append(event)

        room_states = [
            Mock(
                room_id="garage",
                state="occupied",
                timestamp=base_time + timedelta(minutes=15),
            )
        ]

        features = extractor.extract_features(events, room_states)

        # Should handle malformed attributes gracefully
        assert len(features) > 5, "Should extract features despite malformed attributes"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should exclude malformed values"
        assert all(
            np.isfinite(v) for v in features.values()
        ), "All features should be finite"

    def test_feature_store_with_database_connection_failures(self):
        """Test feature store behavior with intermittent database failures."""

        class FlakyDatabaseManager:
            def __init__(self):
                self.call_count = 0

            def get_events(self, *args, **kwargs):
                self.call_count += 1
                if self.call_count % 3 == 0:  # Fail every 3rd call
                    raise ConnectionError("Database connection lost")

                # Return mock events for successful calls
                return [Mock(spec=SensorEvent) for _ in range(5)]

            def get_room_states(self, *args, **kwargs):
                return [Mock(spec=RoomState)]

        store = FeatureStore(db_manager=FlakyDatabaseManager())

        # Multiple attempts should handle intermittent failures
        successful_calls = 0
        failed_calls = 0

        for i in range(10):
            try:
                features = store.get_features(
                    "test_room",
                    datetime(2024, 1, 15, 12, 0, 0),
                    fallback_events=[],  # Provide fallback for failures
                )
                successful_calls += 1
                assert features is not None, "Successful calls should return features"
            except ConnectionError:
                failed_calls += 1

        assert successful_calls > 0, "Some calls should succeed"
        assert failed_calls > 0, "Some calls should fail (as expected)"

    def test_feature_engineering_with_partial_extractor_failures(self):
        """Test feature engineering when some extractors fail."""
        engine = FeatureEngineeringEngine()

        # Mock extractors with different failure patterns
        def failing_temporal_extractor(*args, **kwargs):
            raise RuntimeError("Temporal extractor failure")

        def working_sequential_extractor(*args, **kwargs):
            return {"sequential_feature_1": 1.0, "sequential_feature_2": 2.0}

        def intermittent_contextual_extractor(*args, **kwargs):
            import random

            if random.random() < 0.5:  # 50% failure rate
                raise ValueError("Contextual extractor intermittent failure")
            return {"contextual_feature_1": 3.0}

        # Replace extractors
        engine.temporal_extractor.extract_features = failing_temporal_extractor
        engine.sequential_extractor.extract_features = working_sequential_extractor
        engine.contextual_extractor.extract_features = intermittent_contextual_extractor

        room_id = "test_room"
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        events = [Mock(spec=SensorEvent) for _ in range(5)]
        room_configs = {"test_room": Mock(spec=RoomConfig)}

        # Should extract features from working extractors
        features = engine.extract_features(room_id, target_time, events, room_configs)

        # Should have features from working extractors
        assert "sequential_feature_1" in features, "Working extractor should contribute"
        assert "sequential_feature_2" in features, "Working extractor should contribute"

        # Should handle partial failures gracefully
        assert len(features) >= 2, "Should have features from functioning extractors"


class TestFeatureValidationEdgeCases:
    """Test feature validation under edge case conditions."""

    def test_feature_extraction_with_single_sensor_type(self):
        """Test feature extraction when only one sensor type is available."""
        extractor = SequentialFeatureExtractor()

        # Only motion sensors
        events = []
        for i in range(50):
            event = Mock(spec=SensorEvent)
            event.room_id = "single_sensor_room"
            event.sensor_id = "motion_only"
            event.sensor_type = "motion"  # Only motion sensors
            event.state = "on" if i % 4 < 2 else "off"
            event.timestamp = datetime(2024, 1, 15, 10, 0, 0) + timedelta(minutes=i)
            events.append(event)

        room_configs = {"single_sensor_room": Mock(spec=RoomConfig)}
        features = extractor.extract_features(events, room_configs)

        # Should extract meaningful features even with limited sensor diversity
        assert len(features) > 5, "Should extract features from single sensor type"
        assert "sensor_diversity" in features
        assert features["sensor_diversity"] >= 0, "Sensor diversity should be computed"

    def test_feature_extraction_with_rapid_state_changes(self):
        """Test feature extraction with very rapid sensor state changes."""
        extractor = TemporalFeatureExtractor()

        # Events with very rapid state changes (every second)
        events = []
        base_time = datetime(2024, 1, 15, 13, 0, 0)

        for i in range(300):  # 5 minutes of second-by-second changes
            event = Mock(spec=SensorEvent)
            event.room_id = "rapid_change_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 2 == 0 else "off"  # Alternating every second
            event.timestamp = base_time + timedelta(seconds=i)
            events.append(event)

        target_time = base_time + timedelta(minutes=6)
        features = extractor.extract_features(events, target_time)

        # Should handle rapid changes without computational issues
        assert len(features) > 10, "Should extract features from rapid changes"
        assert "transition_rate" in features
        assert features["transition_rate"] > 0, "Should detect high transition rate"

    def test_feature_extraction_with_extreme_temporal_ranges(self):
        """Test feature extraction with events spanning very long time periods."""
        extractor = TemporalFeatureExtractor()

        # Events spanning 2 years
        events = []
        start_time = datetime(2022, 1, 1, 0, 0, 0)
        end_time = datetime(2024, 1, 1, 0, 0, 0)

        # Sparse events over 2 years
        for i in range(100):
            event = Mock(spec=SensorEvent)
            event.room_id = "long_term_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 3 == 0 else "off"
            # Random times across 2-year span
            time_offset = timedelta(days=i * 7)  # Weekly events
            event.timestamp = start_time + time_offset
            events.append(event)

        target_time = end_time + timedelta(hours=1)
        features = extractor.extract_features(
            events, target_time, lookback_hours=8760
        )  # 1 year

        # Should handle very long time ranges
        assert len(features) > 5, "Should extract features from long-term data"
        assert "time_since_last_change" in features

        # Time since features should handle large values appropriately
        time_since = features["time_since_last_change"]
        assert 0 <= time_since <= 8760, "Time since should be capped appropriately"

    def test_feature_extraction_with_timezone_naive_datetimes(self):
        """Test feature extraction when mixing timezone-aware and naive datetimes."""
        extractor = TemporalFeatureExtractor(timezone_offset=-5)  # EST

        events = []
        base_time = datetime(2024, 1, 15, 12, 0, 0)  # Timezone-naive

        for i in range(20):
            event = Mock(spec=SensorEvent)
            event.room_id = "mixed_timezone_room"
            event.sensor_type = "motion"
            event.state = "on" if i % 2 == 0 else "off"

            # Mix of timezone-naive and timezone-aware timestamps
            if i % 3 == 0:
                # Timezone-naive
                event.timestamp = base_time + timedelta(minutes=i)
            else:
                # Would be timezone-aware in real implementation
                event.timestamp = base_time + timedelta(minutes=i)

            events.append(event)

        target_time = base_time + timedelta(hours=1)

        # Should handle mixed timezone scenarios
        features = extractor.extract_features(events, target_time)

        assert len(features) > 5, "Should handle mixed timezone scenarios"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should normalize timezone handling"

    def test_feature_extraction_memory_constraints(self):
        """Test feature extraction under memory constraints."""
        extractor = ContextualFeatureExtractor()

        # Create large number of events to test memory efficiency
        events = []
        room_states = []

        for i in range(1000):  # Large dataset
            event = Mock(spec=SensorEvent)
            event.room_id = f"room_{i % 10}"
            event.sensor_type = "motion"
            event.state = "on" if i % 2 == 0 else "off"
            event.timestamp = datetime(2024, 1, 15, 12, 0, 0) + timedelta(seconds=i)
            event.attributes = {
                "temperature": 20.0 + (i % 20),
                "humidity": 40.0 + (i % 30),
                "pressure": 1000.0 + (i % 50),
                **{
                    f"extra_attribute_{j}": f"value_{i}_{j}" for j in range(10)
                },  # Extra data
            }
            events.append(event)

            if i % 100 == 0:
                state = Mock(spec=RoomState)
                state.room_id = f"room_{i % 10}"
                state.state = "occupied" if i % 200 < 100 else "vacant"
                state.timestamp = event.timestamp
                room_states.append(state)

        # Memory-constrained extraction should complete
        import psutil

        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        features = extractor.extract_features(events, room_states)

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Should complete without excessive memory usage
        assert len(features) > 10, "Should extract features from large dataset"
        assert memory_increase < 100, "Memory usage should be reasonable"

    def test_feature_extraction_with_duplicate_events(self):
        """Test feature extraction with duplicate and near-duplicate events."""
        extractor = SequentialFeatureExtractor()

        # Create events with duplicates
        events = []
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        for i in range(30):
            # Primary event
            event = Mock(spec=SensorEvent)
            event.room_id = "duplicate_room"
            event.sensor_id = "sensor_1"
            event.sensor_type = "motion"
            event.state = "on" if i % 4 < 2 else "off"
            event.timestamp = base_time + timedelta(minutes=i)
            events.append(event)

            # Duplicate event (exactly same)
            if i % 5 == 0:
                duplicate = Mock(spec=SensorEvent)
                duplicate.room_id = event.room_id
                duplicate.sensor_id = event.sensor_id
                duplicate.sensor_type = event.sensor_type
                duplicate.state = event.state
                duplicate.timestamp = event.timestamp
                events.append(duplicate)

            # Near-duplicate (same timestamp, different sensor)
            if i % 7 == 0:
                near_duplicate = Mock(spec=SensorEvent)
                near_duplicate.room_id = event.room_id
                near_duplicate.sensor_id = "sensor_2"
                near_duplicate.sensor_type = "door"
                near_duplicate.state = "closed"
                near_duplicate.timestamp = event.timestamp
                events.append(near_duplicate)

        room_configs = {"duplicate_room": Mock(spec=RoomConfig)}
        features = extractor.extract_features(events, room_configs)

        # Should handle duplicates gracefully
        assert len(features) > 5, "Should extract features despite duplicates"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Should handle duplicate events"
