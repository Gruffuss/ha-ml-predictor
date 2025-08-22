"""
Comprehensive edge case and error handling tests for temporal feature extraction.

This module provides additional coverage for complex scenarios, error conditions,
and edge cases that may not be covered by the main temporal tests.
"""

from datetime import datetime, timedelta
import math
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
import statistics

from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.temporal import TemporalFeatureExtractor


class TestTemporalFeatureExtractorEdgeCases:
    """Test edge cases and error handling for TemporalFeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        """Create temporal feature extractor."""
        return TemporalFeatureExtractor(timezone_offset=0)

    @pytest.fixture
    def extreme_timezone_extractor(self):
        """Create extractor with extreme timezone offset."""
        return TemporalFeatureExtractor(timezone_offset=14)  # Maximum UTC offset

    @pytest.fixture
    def target_time(self):
        """Standard target time."""
        return datetime(2024, 3, 15, 12, 0, 0)

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

    def test_extract_features_malformed_events(self, extractor, target_time):
        """Test feature extraction with malformed events."""
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

    def test_extract_features_with_future_events(self, extractor, target_time):
        """Test feature extraction with events in the future."""
        future_event = Mock(spec=SensorEvent)
        future_event.timestamp = target_time + timedelta(hours=1)  # Future event
        future_event.state = "on"
        future_event.sensor_type = "motion"
        future_event.room_id = "test_room"

        features = extractor.extract_features([future_event], target_time)

        # Should handle future events gracefully
        assert isinstance(features, dict)
        assert len(features) > 0

    def test_extract_features_with_extreme_time_differences(
        self, extractor, target_time
    ):
        """Test feature extraction with very old and very new events."""
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

    def test_zero_lookback_hours(self, extractor, target_time):
        """Test with zero lookback hours."""
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

    def test_negative_lookback_hours(self, extractor, target_time):
        """Test with negative lookback hours."""
        event = Mock(spec=SensorEvent)
        event.timestamp = target_time - timedelta(minutes=30)
        event.state = "on"
        event.sensor_type = "motion"
        event.room_id = "test_room"

        # Negative lookback - undefined behavior, should handle gracefully
        features = extractor.extract_features([event], target_time, lookback_hours=-5)
        assert isinstance(features, dict)

    def test_extreme_timezone_offsets(self, extreme_timezone_extractor, target_time):
        """Test with extreme timezone offsets."""
        features = extreme_timezone_extractor.extract_features([], target_time)

        # Should handle extreme offsets without error
        assert isinstance(features, dict)
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0

    def test_negative_timezone_offset(self, target_time):
        """Test with negative timezone offset."""
        extractor = TemporalFeatureExtractor(timezone_offset=-12)  # Minimum UTC offset

        features = extractor.extract_features([], target_time)

        assert isinstance(features, dict)
        assert -1.0 <= features["hour_sin"] <= 1.0
        assert -1.0 <= features["hour_cos"] <= 1.0

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

    def test_events_with_invalid_states(self, extractor, target_time):
        """Test events with invalid or unusual states."""
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

    def test_events_with_complex_attributes(self, extractor, target_time):
        """Test events with complex attribute structures."""
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

    def test_events_with_non_dict_attributes(self, extractor, target_time):
        """Test events with non-dictionary attributes."""
        event = Mock(spec=SensorEvent)
        event.timestamp = target_time - timedelta(minutes=30)
        event.state = "on"
        event.sensor_type = "motion"
        event.room_id = "test_room"
        event.attributes = "string_attribute"  # Non-dict attribute

        # Should handle gracefully
        features = extractor.extract_features([event], target_time)
        assert isinstance(features, dict)

    def test_large_number_of_events(self, extractor, target_time):
        """Test with very large number of events."""
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
        import time

        start_time = time.time()
        features = extractor.extract_features(events, target_time)
        extraction_time = time.time() - start_time

        assert isinstance(features, dict)
        assert extraction_time < 10.0  # Should complete within reasonable time

    def test_events_with_duplicate_timestamps(self, extractor, target_time):
        """Test events with identical timestamps."""
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

    def test_unsorted_events(self, extractor, target_time):
        """Test with unsorted event timestamps."""
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

    def test_room_states_edge_cases(self, extractor, target_time):
        """Test room states with edge cases."""
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

    def test_empty_room_states(self, extractor, target_time):
        """Test with empty room states list."""
        events = []
        room_states = []

        features = extractor.extract_features(events, target_time, room_states)
        assert isinstance(features, dict)

    def test_feature_cache_functionality(self, extractor):
        """Test feature cache operations."""
        # Test cache initialization
        assert hasattr(extractor, "feature_cache")
        assert hasattr(extractor, "temporal_cache")

        # Test cache operations
        extractor.feature_cache["test_key"] = "test_value"
        extractor.temporal_cache["temp_key"] = "temp_value"

        assert extractor.feature_cache["test_key"] == "test_value"
        assert extractor.temporal_cache["temp_key"] == "temp_value"

        # Test cache clearing
        extractor.clear_cache()
        assert len(extractor.feature_cache) == 0

    def test_batch_feature_extraction(self, extractor):
        """Test batch feature extraction with edge cases."""
        target_time = datetime(2024, 3, 15, 12, 0, 0)

        # Create batch of event sequences
        event_batches = []

        # Empty batch
        event_batches.append(([], target_time))

        # Single event batch
        single_event = Mock(spec=SensorEvent)
        single_event.timestamp = target_time - timedelta(minutes=10)
        single_event.state = "on"
        single_event.sensor_type = "motion"
        single_event.room_id = "test_room"
        event_batches.append(([single_event], target_time))

        # Batch with None events (should handle gracefully)
        try:
            event_batches.append((None, target_time))
        except (ValueError, TypeError):
            pass  # Expected if None is not handled

        results = extractor.extract_batch_features(
            event_batches[:2]
        )  # Exclude None batch

        assert isinstance(results, list)
        assert len(results) == 2
        for result in results:
            assert isinstance(result, dict)

    def test_feature_name_validation(self, extractor):
        """Test feature name validation and standardization."""
        # Test with sample extracted features
        extracted_features = {
            "time_since_last_off": 300.0,
            "current_state_duration": 600.0,
            "hour_sin": 0.5,
            "invalid_feature": 123.0,
            "custom_feature": 456.0,
        }

        if hasattr(extractor, "validate_feature_names"):
            validated = extractor.validate_feature_names(extracted_features)
            assert isinstance(validated, dict)
            assert len(validated) > 0

    def test_get_feature_names_method(self, extractor):
        """Test get_feature_names method comprehensively."""
        feature_names = extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 30  # Should have many features

        # Check for expected feature categories
        categories = ["time_since_", "hour_", "day_", "is_", "avg_", "duration"]
        for category in categories:
            category_found = any(category in name for name in feature_names)
            assert category_found, f"No features found for category: {category}"

    def test_default_features_completeness(self, extractor):
        """Test default features with different target times."""
        test_times = [
            None,  # No target time
            datetime(2024, 1, 1, 0, 0, 0),  # New Year
            datetime(2024, 6, 21, 12, 0, 0),  # Summer solstice
            datetime(2024, 12, 25, 18, 0, 0),  # Christmas evening
        ]

        for target_time in test_times:
            defaults = extractor._get_default_features(target_time)
            assert isinstance(defaults, dict)
            assert len(defaults) > 40  # Should have comprehensive defaults

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

    def test_error_handling_in_feature_extraction(self, extractor, target_time):
        """Test error handling during feature extraction."""
        # Create event that will cause issues in processing
        problematic_event = Mock(spec=SensorEvent)
        problematic_event.timestamp = target_time - timedelta(minutes=30)
        problematic_event.state = "on"
        problematic_event.sensor_type = "motion"
        problematic_event.room_id = "test_room"

        # Mock an attribute that causes errors when accessed
        problematic_event.attributes = Mock()
        problematic_event.attributes.items.side_effect = RuntimeError(
            "Attribute access failed"
        )

        # Should handle the error and still return features
        features = extractor.extract_features([problematic_event], target_time)
        assert isinstance(features, dict)

    def test_memory_usage_with_large_datasets(self, extractor, target_time):
        """Test memory usage with large datasets."""
        import os

        import psutil

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

    def test_concurrent_feature_extraction(self, extractor, target_time):
        """Test thread safety of feature extraction."""
        import queue
        import threading

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
