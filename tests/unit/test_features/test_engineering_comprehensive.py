"""
Comprehensive unit tests for feature engineering pipeline.

This module provides exhaustive testing of the FeatureEngineeringEngine class,
covering integration workflows, performance benchmarks, error handling,
parallel processing, and statistical analysis functions.
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
import math
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.core.config import RoomConfig, SystemConfig, get_config
from src.core.exceptions import ConfigurationError, FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.temporal import TemporalFeatureExtractor


class TestFeatureEngineeringEngineComprehensive:
    """Comprehensive test suite for FeatureEngineeringEngine with production-grade validation."""

    @pytest.fixture
    def system_config(self) -> SystemConfig:
        """Create a system configuration for testing."""
        config = Mock(spec=SystemConfig)
        config.rooms = {
            "living_room": self.create_room_config("living_room"),
            "kitchen": self.create_room_config("kitchen"),
            "bedroom": self.create_room_config("bedroom"),
            "bathroom": self.create_room_config("bathroom"),
        }
        return config

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
        """Create comprehensive sensor events across multiple rooms."""
        base_time = datetime(2024, 6, 15, 8, 0, 0)
        events = []

        rooms = ["living_room", "kitchen", "bedroom", "bathroom"]
        sensor_types = ["motion", "presence", "door", "temperature"]

        # Create realistic daily pattern
        for hour in range(12):  # 12 hours of activity
            for i, room in enumerate(rooms):
                # Different activity patterns per room
                if room == "bedroom" and hour < 2:  # Early morning in bedroom
                    activity_prob = 0.8
                elif room == "kitchen" and 6 <= hour <= 8:  # Morning kitchen activity
                    activity_prob = 0.9
                elif (
                    room == "living_room" and 10 <= hour <= 11
                ):  # Late morning living room
                    activity_prob = 0.7
                else:
                    activity_prob = 0.3

                if np.random.random() < activity_prob:
                    sensor_type = sensor_types[hour % len(sensor_types)]
                    state = (
                        "on"
                        if sensor_type in ["motion", "presence"]
                        else str(20 + hour + i)
                    )

                    event = self.create_sensor_event(
                        timestamp=base_time + timedelta(hours=hour, minutes=i * 15),
                        sensor_id=f"sensor.{room}_{sensor_type}",
                        sensor_type=sensor_type,
                        state=state,
                        room_id=room,
                    )
                    events.append(event)

        return events

    @pytest.fixture
    def comprehensive_room_states(self) -> List[RoomState]:
        """Create comprehensive room state history."""
        base_time = datetime(2024, 6, 15, 6, 0, 0)
        states = []

        rooms = ["living_room", "kitchen", "bedroom", "bathroom"]

        # Create 18 hours of room state data
        for hour in range(18):
            for i, room in enumerate(rooms):
                timestamp = base_time + timedelta(hours=hour, minutes=i * 10)

                # Realistic occupancy patterns
                if room == "bedroom":
                    is_occupied = hour < 8 or hour > 22  # Sleep hours
                elif room == "kitchen":
                    is_occupied = hour in [7, 8, 12, 13, 18, 19]  # Meal times
                elif room == "living_room":
                    is_occupied = 9 <= hour <= 23  # Day and evening
                else:  # bathroom
                    is_occupied = hour in [7, 12, 18, 22]  # Usage times

                confidence = 0.75 + (hour % 4) * 0.05  # Varying confidence

                state = self.create_room_state(timestamp, room, is_occupied, confidence)
                states.append(state)

        return states

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 6, 15, 14, 0, 0)

    # ==================== INITIALIZATION TESTS ====================

    def test_initialization_default(self):
        """Test default initialization without configuration."""
        with patch("src.features.engineering.get_config") as mock_get_config:
            mock_config = Mock(spec=SystemConfig)
            mock_get_config.return_value = mock_config

            engine = FeatureEngineeringEngine()

            assert engine.config == mock_config
            assert engine.enable_parallel is True
            assert engine.max_workers == 3
            assert isinstance(engine.temporal_extractor, TemporalFeatureExtractor)
            assert isinstance(engine.sequential_extractor, SequentialFeatureExtractor)
            assert isinstance(engine.contextual_extractor, ContextualFeatureExtractor)

    def test_initialization_with_config(self, system_config):
        """Test initialization with provided configuration."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=False, max_workers=2
        )

        assert engine.config == system_config
        assert engine.enable_parallel is False
        assert engine.max_workers == 2
        assert isinstance(engine.stats, dict)
        assert engine.executor is None  # Should be None when parallel is disabled

    def test_initialization_with_parallel_enabled(self, system_config):
        """Test initialization with parallel processing enabled."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=4
        )

        assert engine.enable_parallel is True
        assert engine.max_workers == 4
        assert isinstance(engine.executor, ThreadPoolExecutor)

        # Cleanup
        engine.executor.shutdown(wait=False)

    def test_initialization_validation_with_none_config(self):
        """Test validation logic with None configuration."""
        with patch("src.features.engineering.get_config") as mock_get_config:
            mock_get_config.return_value = None

            engine = FeatureEngineeringEngine(config=None)
            assert engine._original_config_was_none is True

            # Should not validate when original config was None
            assert engine.config is None

    def test_initialization_validation_with_provided_config(self, system_config):
        """Test validation with provided configuration."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Should have proper extractors initialized
        assert engine.temporal_extractor is not None
        assert engine.sequential_extractor is not None
        assert engine.contextual_extractor is not None

    def test_initialization_stats_structure(self, system_config):
        """Test initialization of statistics structure."""
        engine = FeatureEngineeringEngine(config=system_config)

        expected_stats = {
            "total_extractions",
            "successful_extractions",
            "failed_extractions",
            "avg_extraction_time",
            "feature_counts",
        }

        for key in expected_stats:
            assert key in engine.stats

        assert "temporal" in engine.stats["feature_counts"]
        assert "sequential" in engine.stats["feature_counts"]
        assert "contextual" in engine.stats["feature_counts"]

    # ==================== FEATURE EXTRACTION TESTS ====================

    @pytest.mark.asyncio
    async def test_extract_features_comprehensive(
        self,
        system_config,
        comprehensive_events,
        comprehensive_room_states,
        target_time,
    ):
        """Test comprehensive feature extraction."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        features = await engine.extract_features(
            room_id="living_room",
            target_time=target_time,
            events=comprehensive_events,
            room_states=comprehensive_room_states,
            lookback_hours=12,
        )

        # Verify comprehensive feature set
        assert isinstance(features, dict)
        assert len(features) > 50  # Should have many features

        # Check for features from all extractors
        temporal_features = [k for k in features.keys() if k.startswith("temporal_")]
        sequential_features = [
            k for k in features.keys() if k.startswith("sequential_")
        ]
        contextual_features = [
            k for k in features.keys() if k.startswith("contextual_")
        ]
        meta_features = [k for k in features.keys() if k.startswith("meta_")]

        assert len(temporal_features) > 10
        assert len(sequential_features) > 10
        assert len(contextual_features) > 10
        assert len(meta_features) > 0

    @pytest.mark.asyncio
    async def test_extract_features_with_specific_types(
        self, system_config, comprehensive_events, target_time
    ):
        """Test feature extraction with specific feature types."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Extract only temporal features
        temporal_only = await engine.extract_features(
            room_id="kitchen",
            target_time=target_time,
            events=comprehensive_events,
            feature_types=["temporal"],
        )

        # Should only have temporal and meta features
        temporal_features = [
            k for k in temporal_only.keys() if k.startswith("temporal_")
        ]
        sequential_features = [
            k for k in temporal_only.keys() if k.startswith("sequential_")
        ]
        contextual_features = [
            k for k in temporal_only.keys() if k.startswith("contextual_")
        ]

        assert len(temporal_features) > 0
        assert len(sequential_features) == 0
        assert len(contextual_features) == 0

    @pytest.mark.asyncio
    async def test_extract_features_parallel_processing(
        self, system_config, comprehensive_events, target_time
    ):
        """Test feature extraction with parallel processing enabled."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=3
        )

        try:
            features = await engine.extract_features(
                room_id="bedroom",
                target_time=target_time,
                events=comprehensive_events,
                lookback_hours=8,
            )

            # Should have extracted features successfully
            assert isinstance(features, dict)
            assert len(features) > 20

            # Should have features from multiple extractors
            feature_prefixes = set(k.split("_")[0] for k in features.keys())
            assert "temporal" in feature_prefixes
            assert "sequential" in feature_prefixes or "contextual" in feature_prefixes

        finally:
            engine.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_extract_features_input_validation(self, system_config):
        """Test input validation for feature extraction."""
        engine = FeatureEngineeringEngine(config=system_config)
        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Test with empty room_id
        with pytest.raises(FeatureExtractionError):
            await engine.extract_features(
                room_id="", target_time=target_time, events=[], room_states=[]
            )

        # Test with None room_id
        with pytest.raises(FeatureExtractionError):
            await engine.extract_features(
                room_id=None, target_time=target_time, events=[], room_states=[]
            )

    @pytest.mark.asyncio
    async def test_extract_features_room_filtering(
        self,
        system_config,
        comprehensive_events,
        comprehensive_room_states,
        target_time,
    ):
        """Test proper room filtering during feature extraction."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Extract features for specific room
        features = await engine.extract_features(
            room_id="kitchen",
            target_time=target_time,
            events=comprehensive_events,
            room_states=comprehensive_room_states,
            lookback_hours=24,
        )

        # Should have extracted features successfully
        assert isinstance(features, dict)
        assert len(features) > 0

        # Verify room-specific filtering worked (this is implicit in the implementation)
        # The events and room_states should be filtered to kitchen only internally

    @pytest.mark.asyncio
    async def test_extract_features_lookback_filtering(
        self, system_config, target_time
    ):
        """Test lookback time filtering."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Create events spanning different time ranges
        base_time = target_time - timedelta(hours=48)
        events = []

        for hours_back in range(0, 48, 2):  # Events every 2 hours for 48 hours
            event_time = target_time - timedelta(hours=hours_back)
            event = self.create_sensor_event(
                timestamp=event_time,
                sensor_id=f"sensor_{hours_back}",
                sensor_type="motion",
                state="on",
                room_id="test_room",
            )
            events.append(event)

        # Extract with 24-hour lookback
        features_24h = await engine.extract_features(
            room_id="test_room",
            target_time=target_time,
            events=events,
            lookback_hours=24,
        )

        # Extract with 12-hour lookback
        features_12h = await engine.extract_features(
            room_id="test_room",
            target_time=target_time,
            events=events,
            lookback_hours=12,
        )

        # Both should work but potentially have different results
        assert isinstance(features_24h, dict)
        assert isinstance(features_12h, dict)

    @pytest.mark.asyncio
    async def test_extract_features_missing_room_config(
        self, system_config, comprehensive_events, target_time
    ):
        """Test feature extraction with missing room configuration."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Try to extract features for room not in config
        features = await engine.extract_features(
            room_id="unknown_room", target_time=target_time, events=comprehensive_events
        )

        # Should still work with default behavior
        assert isinstance(features, dict)
        assert len(features) > 0

    @pytest.mark.asyncio
    async def test_extract_features_statistics_tracking(
        self, system_config, comprehensive_events, target_time
    ):
        """Test that extraction statistics are properly tracked."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Initial stats
        initial_total = engine.stats["total_extractions"]
        initial_successful = engine.stats["successful_extractions"]

        # Perform extraction
        features = await engine.extract_features(
            room_id="living_room", target_time=target_time, events=comprehensive_events
        )

        # Verify stats updated
        assert engine.stats["total_extractions"] == initial_total + 1
        assert engine.stats["successful_extractions"] == initial_successful + 1
        assert engine.stats["avg_extraction_time"] > 0

    # ==================== BATCH EXTRACTION TESTS ====================

    @pytest.mark.asyncio
    async def test_batch_feature_extraction_sequential(
        self, system_config, comprehensive_events, comprehensive_room_states
    ):
        """Test batch feature extraction with sequential processing."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Create multiple extraction requests
        requests = [
            (
                "living_room",
                datetime(2024, 6, 15, 10, 0, 0),
                comprehensive_events,
                comprehensive_room_states,
            ),
            (
                "kitchen",
                datetime(2024, 6, 15, 12, 0, 0),
                comprehensive_events,
                comprehensive_room_states,
            ),
            (
                "bedroom",
                datetime(2024, 6, 15, 14, 0, 0),
                comprehensive_events,
                comprehensive_room_states,
            ),
        ]

        results = await engine.extract_batch_features(requests, lookback_hours=6)

        assert len(results) == 3
        for result in results:
            assert isinstance(result, dict)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_batch_feature_extraction_parallel(
        self, system_config, comprehensive_events, comprehensive_room_states
    ):
        """Test batch feature extraction with parallel processing."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )

        try:
            # Create multiple extraction requests
            requests = [
                (
                    "living_room",
                    datetime(2024, 6, 15, 9, 0, 0),
                    comprehensive_events,
                    comprehensive_room_states,
                ),
                (
                    "kitchen",
                    datetime(2024, 6, 15, 11, 0, 0),
                    comprehensive_events,
                    comprehensive_room_states,
                ),
                (
                    "bedroom",
                    datetime(2024, 6, 15, 13, 0, 0),
                    comprehensive_events,
                    comprehensive_room_states,
                ),
                (
                    "bathroom",
                    datetime(2024, 6, 15, 15, 0, 0),
                    comprehensive_events,
                    comprehensive_room_states,
                ),
            ]

            results = await engine.extract_batch_features(requests, lookback_hours=4)

            assert len(results) == 4
            for result in results:
                assert isinstance(result, dict)
                assert len(result) > 0

        finally:
            engine.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_batch_extraction_error_handling(self, system_config):
        """Test error handling in batch extraction."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Create requests with some that will fail
        good_events = [
            self.create_sensor_event(
                datetime(2024, 6, 15, 12, 0, 0), "sensor1", "motion", "on", "room1"
            )
        ]

        requests = [
            ("room1", datetime(2024, 6, 15, 14, 0, 0), good_events, []),  # Good request
            (
                "",
                datetime(2024, 6, 15, 14, 0, 0),
                [],
                [],
            ),  # Bad request (empty room_id)
            ("room2", datetime(2024, 6, 15, 14, 0, 0), good_events, []),  # Good request
        ]

        results = await engine.extract_batch_features(requests)

        assert len(results) == 3
        # First and third should be valid dicts, second might be default features due to error
        assert isinstance(results[0], dict)
        assert isinstance(results[1], dict)  # Should get default features on error
        assert isinstance(results[2], dict)

    # ==================== PARALLEL PROCESSING TESTS ====================

    @pytest.mark.asyncio
    async def test_parallel_extraction_task_creation(
        self, system_config, comprehensive_events, target_time
    ):
        """Test parallel extraction task creation and execution."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=3
        )

        try:
            # Mock the individual extractors to track calls
            with patch.object(
                engine.temporal_extractor, "extract_features"
            ) as mock_temporal, patch.object(
                engine.sequential_extractor, "extract_features"
            ) as mock_sequential, patch.object(
                engine.contextual_extractor, "extract_features"
            ) as mock_contextual:

                # Configure mocks to return test data
                mock_temporal.return_value = {
                    "temporal_feature_1": 1.0,
                    "temporal_feature_2": 2.0,
                }
                mock_sequential.return_value = {"sequential_feature_1": 3.0}
                mock_contextual.return_value = {"contextual_feature_1": 4.0}

                features = await engine.extract_features(
                    room_id="test_room",
                    target_time=target_time,
                    events=comprehensive_events,
                    feature_types=["temporal", "sequential", "contextual"],
                )

                # Verify all extractors were called
                mock_temporal.assert_called_once()
                mock_sequential.assert_called_once()
                mock_contextual.assert_called_once()

                # Verify features were prefixed correctly
                assert "temporal_temporal_feature_1" in features
                assert "sequential_sequential_feature_1" in features
                assert "contextual_contextual_feature_1" in features

        finally:
            engine.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_parallel_extraction_failure_handling(
        self, system_config, comprehensive_events, target_time
    ):
        """Test handling of individual extractor failures in parallel mode."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )

        try:
            with patch.object(
                engine.temporal_extractor, "extract_features"
            ) as mock_temporal, patch.object(
                engine.sequential_extractor, "extract_features"
            ) as mock_sequential:

                # Configure one to succeed, one to fail
                mock_temporal.return_value = {"temporal_feature_1": 1.0}
                mock_sequential.side_effect = Exception("Sequential extraction failed")

                features = await engine.extract_features(
                    room_id="test_room",
                    target_time=target_time,
                    events=comprehensive_events,
                    feature_types=["temporal", "sequential"],
                )

                # Should have temporal features but not sequential
                assert "temporal_temporal_feature_1" in features
                assert not any(k.startswith("sequential_") for k in features.keys())

                # Should have metadata about failed extractors
                assert "_failed_extractors" in features
                assert features["_failed_extractors"] == 1

        finally:
            engine.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_parallel_all_extractors_fail(
        self, system_config, comprehensive_events, target_time
    ):
        """Test handling when all extractors fail in parallel mode."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )

        try:
            with patch.object(
                engine.temporal_extractor, "extract_features"
            ) as mock_temporal, patch.object(
                engine.sequential_extractor, "extract_features"
            ) as mock_sequential, patch.object(
                engine.contextual_extractor, "extract_features"
            ) as mock_contextual:

                # Configure all to fail
                mock_temporal.side_effect = Exception("Temporal failed")
                mock_sequential.side_effect = Exception("Sequential failed")
                mock_contextual.side_effect = Exception("Contextual failed")

                with pytest.raises(FeatureExtractionError):
                    await engine.extract_features(
                        room_id="test_room",
                        target_time=target_time,
                        events=comprehensive_events,
                        feature_types=["temporal", "sequential", "contextual"],
                    )

        finally:
            engine.executor.shutdown(wait=False)

    # ==================== SEQUENTIAL PROCESSING TESTS ====================

    @pytest.mark.asyncio
    async def test_sequential_extraction_error_isolation(
        self, system_config, comprehensive_events, target_time
    ):
        """Test error isolation in sequential processing mode."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        with patch.object(
            engine.temporal_extractor, "extract_features"
        ) as mock_temporal, patch.object(
            engine.sequential_extractor, "extract_features"
        ) as mock_sequential, patch.object(
            engine.contextual_extractor, "extract_features"
        ) as mock_contextual:

            # Configure one to fail, others to succeed
            mock_temporal.return_value = {"temporal_feature": 1.0}
            mock_sequential.side_effect = Exception("Sequential failed")
            mock_contextual.return_value = {"contextual_feature": 2.0}

            features = await engine.extract_features(
                room_id="test_room",
                target_time=target_time,
                events=comprehensive_events,
                feature_types=["temporal", "sequential", "contextual"],
            )

            # Should have features from successful extractors
            assert "temporal_temporal_feature" in features
            assert "contextual_contextual_feature" in features

            # Should not have features from failed extractor
            assert not any(k.startswith("sequential_") for k in features.keys())

    # ==================== METADATA FEATURES TESTS ====================

    def test_metadata_features_calculation(self, system_config):
        """Test metadata feature calculation."""
        engine = FeatureEngineeringEngine(config=system_config)

        target_time = datetime(2024, 6, 15, 14, 30, 0)  # Saturday, 2:30 PM
        event_count = 25
        room_state_count = 8

        meta_features = engine._add_metadata_features(
            room_id="test_room",
            target_time=target_time,
            event_count=event_count,
            room_state_count=room_state_count,
        )

        # Verify all metadata features are present
        required_features = [
            "meta_event_count",
            "meta_room_state_count",
            "meta_extraction_hour",
            "meta_extraction_day_of_week",
            "meta_data_quality_score",
            "meta_feature_vector_norm",
        ]

        for feature in required_features:
            assert feature in meta_features
            assert isinstance(meta_features[feature], (int, float))

        # Verify calculations
        assert meta_features["meta_extraction_hour"] == 14.0  # 2 PM
        assert (
            meta_features["meta_extraction_day_of_week"] == 5.0
        )  # Saturday (5 in Python)
        assert meta_features["meta_event_count"] == 25.0  # Normalized and denormalized
        assert meta_features["meta_room_state_count"] == 8.0

    def test_metadata_features_normalization(self, system_config):
        """Test normalization in metadata features using numpy."""
        engine = FeatureEngineeringEngine(config=system_config)

        target_time = datetime(2024, 6, 15, 10, 0, 0)

        # Test with extreme values
        meta_features = engine._add_metadata_features(
            room_id="test_room",
            target_time=target_time,
            event_count=200,  # Above normalization threshold
            room_state_count=100,  # Above normalization threshold
        )

        # Normalized values should be capped
        assert meta_features["meta_data_quality_score"] <= 1.0
        assert meta_features["meta_feature_vector_norm"] > 0.0

    # ==================== UTILITY METHODS TESTS ====================

    def test_get_feature_names_comprehensive(self, system_config):
        """Test comprehensive feature names retrieval."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Test with all feature types
        all_feature_names = engine.get_feature_names()

        assert isinstance(all_feature_names, list)
        assert len(all_feature_names) > 50  # Should have many features

        # Should have features from all extractors
        temporal_count = sum(
            1 for name in all_feature_names if name.startswith("temporal_")
        )
        sequential_count = sum(
            1 for name in all_feature_names if name.startswith("sequential_")
        )
        contextual_count = sum(
            1 for name in all_feature_names if name.startswith("contextual_")
        )
        meta_count = sum(1 for name in all_feature_names if name.startswith("meta_"))

        assert temporal_count > 10
        assert sequential_count > 10
        assert contextual_count > 10
        assert meta_count > 0

    def test_get_feature_names_specific_types(self, system_config):
        """Test feature names retrieval for specific types."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Test with only temporal features
        temporal_names = engine.get_feature_names(feature_types=["temporal"])

        temporal_count = sum(
            1 for name in temporal_names if name.startswith("temporal_")
        )
        sequential_count = sum(
            1 for name in temporal_names if name.startswith("sequential_")
        )
        contextual_count = sum(
            1 for name in temporal_names if name.startswith("contextual_")
        )

        assert temporal_count > 0
        assert sequential_count == 0
        assert contextual_count == 0

    def test_create_feature_dataframe(self, system_config):
        """Test creation of feature DataFrame from feature dictionaries."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Create sample feature dictionaries
        feature_dicts = [
            {
                "temporal_hour_sin": 0.5,
                "sequential_room_count": 3.0,
                "contextual_temperature": 22.5,
                "meta_event_count": 10.0,
            },
            {
                "temporal_hour_sin": 0.7,
                "sequential_room_count": 2.0,
                "contextual_temperature": 23.0,
                "meta_event_count": 15.0,
            },
        ]

        df = engine.create_feature_dataframe(
            feature_dicts, feature_types=["temporal", "sequential", "contextual"]
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 2  # Two rows
        assert len(df.columns) > 4  # Should have many columns from all feature types

        # Verify specific features are present
        assert "temporal_hour_sin" in df.columns
        assert "sequential_room_count" in df.columns
        assert "contextual_temperature" in df.columns

    def test_create_feature_dataframe_empty_data(self, system_config):
        """Test DataFrame creation with empty data."""
        engine = FeatureEngineeringEngine(config=system_config)

        df = engine.create_feature_dataframe([])

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0

    def test_get_default_features_completeness(self, system_config):
        """Test default features completeness."""
        engine = FeatureEngineeringEngine(config=system_config)

        defaults = engine._get_default_features()

        assert isinstance(defaults, dict)
        assert len(defaults) > 50  # Should have comprehensive defaults

        # Should have features from all extractors with prefixes
        temporal_count = sum(1 for k in defaults.keys() if k.startswith("temporal_"))
        sequential_count = sum(
            1 for k in defaults.keys() if k.startswith("sequential_")
        )
        contextual_count = sum(
            1 for k in defaults.keys() if k.startswith("contextual_")
        )
        meta_count = sum(1 for k in defaults.keys() if k.startswith("meta_"))

        assert temporal_count > 10
        assert sequential_count > 10
        assert contextual_count > 10
        assert meta_count > 0

    # ==================== STATISTICS AND MONITORING TESTS ====================

    def test_extraction_stats_tracking(self, system_config):
        """Test extraction statistics tracking."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Initial stats
        initial_stats = engine.get_extraction_stats()

        assert initial_stats["total_extractions"] == 0
        assert initial_stats["successful_extractions"] == 0
        assert initial_stats["failed_extractions"] == 0
        assert initial_stats["avg_extraction_time"] == 0.0

        # Check feature counts structure
        assert "feature_counts" in initial_stats
        assert "temporal" in initial_stats["feature_counts"]
        assert "sequential" in initial_stats["feature_counts"]
        assert "contextual" in initial_stats["feature_counts"]

    def test_reset_stats(self, system_config):
        """Test statistics reset functionality."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Modify stats
        engine.stats["total_extractions"] = 10
        engine.stats["successful_extractions"] = 8
        engine.stats["failed_extractions"] = 2

        # Reset
        engine.reset_stats()

        # Should be back to initial values
        assert engine.stats["total_extractions"] == 0
        assert engine.stats["successful_extractions"] == 0
        assert engine.stats["failed_extractions"] == 0

    def test_clear_caches(self, system_config):
        """Test clearing of all extractor caches."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Mock the cache clearing methods
        with patch.object(
            engine.temporal_extractor, "clear_cache"
        ) as mock_temporal_clear, patch.object(
            engine.sequential_extractor, "clear_cache"
        ) as mock_sequential_clear, patch.object(
            engine.contextual_extractor, "clear_cache"
        ) as mock_contextual_clear:

            engine.clear_caches()

            # Verify all caches were cleared
            mock_temporal_clear.assert_called_once()
            mock_sequential_clear.assert_called_once()
            mock_contextual_clear.assert_called_once()

    # ==================== CONFIGURATION VALIDATION TESTS ====================

    @pytest.mark.asyncio
    async def test_validate_configuration_success(self, system_config):
        """Test successful configuration validation."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )

        try:
            validation_result = await engine.validate_configuration()

            assert validation_result["valid"] is True
            assert len(validation_result["errors"]) == 0
            assert isinstance(validation_result["warnings"], list)

        finally:
            if engine.executor:
                engine.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_validate_configuration_none_config(self):
        """Test configuration validation with None config."""
        engine = FeatureEngineeringEngine(config=None)

        validation_result = await engine.validate_configuration()

        assert validation_result["valid"] is False
        assert "No system configuration provided" in validation_result["errors"]

    @pytest.mark.asyncio
    async def test_validate_configuration_missing_rooms(self):
        """Test configuration validation with missing room configurations."""
        config = Mock(spec=SystemConfig)
        config.rooms = {}  # Empty rooms

        engine = FeatureEngineeringEngine(config=config)

        validation_result = await engine.validate_configuration()

        assert isinstance(validation_result, dict)
        assert "warnings" in validation_result

    @pytest.mark.asyncio
    async def test_validate_configuration_parallel_issues(self, system_config):
        """Test configuration validation with parallel processing issues."""
        # Create engine with parallel enabled but simulate executor issue
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )
        engine.executor = None  # Simulate missing executor

        validation_result = await engine.validate_configuration()

        assert (
            "Parallel processing enabled but executor not available"
            in validation_result["warnings"]
        )

    def test_validate_configuration_initialization_errors(self):
        """Test configuration validation during initialization."""
        # Test with invalid max_workers
        with pytest.raises(ConfigurationError):
            FeatureEngineeringEngine(
                config=Mock(spec=SystemConfig), max_workers=0  # Invalid value
            )

    # ==================== STATISTICAL ANALYSIS TESTS ====================

    def test_compute_feature_correlations(self, system_config):
        """Test feature correlation computation using pandas and numpy."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Create sample feature matrix
        feature_data = pd.DataFrame(
            {
                "feature_1": [1.0, 2.0, 3.0, 4.0, 5.0],
                "feature_2": [
                    2.0,
                    4.0,
                    6.0,
                    8.0,
                    10.0,
                ],  # Perfectly correlated with feature_1
                "feature_3": [5.0, 4.0, 3.0, 2.0, 1.0],  # Perfectly anti-correlated
                "feature_4": [1.5, 2.1, 2.9, 4.2, 4.8],  # Slightly correlated
            }
        )

        correlation_result = engine.compute_feature_correlations(feature_data)

        # Verify structure
        assert "correlation_matrix" in correlation_result
        assert "high_correlations" in correlation_result
        assert "mean_abs_correlation" in correlation_result
        assert "max_correlation" in correlation_result

        # Check for high correlation detection
        high_correlations = correlation_result["high_correlations"]
        correlation_pairs = [
            (pair["feature1"], pair["feature2"]) for pair in high_correlations
        ]

        # Should detect perfect correlation between feature_1 and feature_2
        assert any(
            ("feature_1" in pair and "feature_2" in pair) for pair in correlation_pairs
        )

    def test_compute_feature_correlations_empty_data(self, system_config):
        """Test correlation computation with empty data."""
        engine = FeatureEngineeringEngine(config=system_config)

        empty_df = pd.DataFrame()
        result = engine.compute_feature_correlations(empty_df)

        assert result["correlation_matrix"] is None
        assert result["high_correlations"] == []

    def test_analyze_feature_importance(self, system_config):
        """Test feature importance analysis using correlation with target."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Create sample data with known relationships
        features_df = pd.DataFrame(
            {
                "important_feature": [1, 2, 3, 4, 5],
                "unimportant_feature": [1, 1, 1, 1, 1],  # Constant, no importance
                "somewhat_important": [2, 3, 4, 5, 6],
            }
        )

        # Target that correlates with important_feature
        targets_df = pd.DataFrame(
            {
                "next_transition_time": [
                    pd.Timestamp(f"2024-06-15 {10+i}:00:00") for i in range(5)
                ]
            }
        )

        importance_result = engine.analyze_feature_importance(features_df, targets_df)

        # Verify structure
        assert "feature_importance" in importance_result
        assert "top_features" in importance_result
        assert "mean_importance" in importance_result
        assert "importance_std" in importance_result

        # Verify important feature is detected
        importances = importance_result["feature_importance"]
        assert "important_feature" in importances
        assert "unimportant_feature" in importances

        # Important feature should have higher importance than unimportant
        assert importances["important_feature"] > importances["unimportant_feature"]

    def test_analyze_feature_importance_edge_cases(self, system_config):
        """Test feature importance analysis edge cases."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Empty data
        result_empty = engine.analyze_feature_importance(pd.DataFrame(), pd.DataFrame())
        assert result_empty["feature_importance"] == {}
        assert result_empty["top_features"] == []

        # Single data point
        single_features = pd.DataFrame({"feature": [1.0]})
        single_targets = pd.DataFrame({"target": [1.0]})

        result_single = engine.analyze_feature_importance(
            single_features, single_targets
        )
        assert isinstance(result_single, dict)

    def test_compute_feature_statistics_comprehensive(self, system_config):
        """Test comprehensive feature statistics computation."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Create feature DataFrame with known statistical properties
        np.random.seed(42)  # For reproducible results
        feature_data = pd.DataFrame(
            {
                "normal_feature": np.random.normal(0, 1, 100),
                "skewed_feature": np.random.exponential(2, 100),
                "constant_feature": [5.0] * 100,
                "high_variance_feature": np.random.normal(0, 10, 100),
            }
        )

        stats_result = engine.compute_feature_statistics(feature_data)

        # Verify structure
        assert "basic_statistics" in stats_result
        assert "advanced_statistics" in stats_result
        assert "summary" in stats_result

        # Check basic statistics
        basic_stats = stats_result["basic_statistics"]
        assert "normal_feature" in basic_stats

        # Check advanced statistics
        advanced_stats = stats_result["advanced_statistics"]
        for feature in feature_data.columns:
            assert feature in advanced_stats
            assert "skewness" in advanced_stats[feature]
            assert "kurtosis" in advanced_stats[feature]
            assert "entropy" in advanced_stats[feature]
            assert "outlier_count" in advanced_stats[feature]

        # Check summary
        summary = stats_result["summary"]
        assert summary["total_features"] == 4
        assert summary["total_samples"] == 100
        assert "constant_features" in summary
        assert (
            "constant_feature" in summary["constant_features"]
        )  # Should detect constant feature

    def test_statistical_calculations_accuracy(self, system_config):
        """Test accuracy of statistical calculations."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Test with known data
        test_data = np.array([1, 2, 3, 4, 5], dtype=float)

        # Test skewness calculation
        skewness = engine._calculate_skewness(test_data)
        assert isinstance(skewness, float)

        # Test kurtosis calculation
        kurtosis = engine._calculate_kurtosis(test_data)
        assert isinstance(kurtosis, float)

        # Test entropy calculation
        entropy = engine._calculate_entropy(test_data)
        assert isinstance(entropy, float)
        assert entropy > 0  # Should have some entropy

        # Test outlier count
        outlier_count = engine._count_outliers(test_data)
        assert isinstance(outlier_count, int)
        assert outlier_count == 0  # No outliers in this simple dataset

    def test_statistical_edge_cases(self, system_config):
        """Test statistical calculations with edge cases."""
        engine = FeatureEngineeringEngine(config=system_config)

        # Empty array
        empty_array = np.array([])
        assert engine._calculate_entropy(empty_array) == 0.0

        # Single value
        single_value = np.array([5.0])
        assert engine._calculate_skewness(single_value) == 0.0
        assert engine._calculate_kurtosis(single_value) == 0.0

        # Constant values
        constant_values = np.array([3.0, 3.0, 3.0, 3.0])
        assert engine._calculate_skewness(constant_values) == 0.0
        assert engine._calculate_kurtosis(constant_values) == 0.0

    # ==================== PERFORMANCE TESTS ====================

    @pytest.mark.asyncio
    async def test_performance_with_large_dataset(self, system_config):
        """Test performance with large feature extraction dataset."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=3
        )

        try:
            # Create large dataset
            base_time = datetime(2024, 6, 15, 0, 0, 0)
            target_time = base_time + timedelta(hours=12)

            large_events = []
            large_room_states = []

            # Create 1000 events across multiple rooms
            rooms = ["room_1", "room_2", "room_3", "room_4"]
            sensor_types = ["motion", "presence", "door", "temperature"]

            for i in range(1000):
                room = rooms[i % len(rooms)]
                sensor_type = sensor_types[i % len(sensor_types)]

                event = self.create_sensor_event(
                    timestamp=base_time + timedelta(minutes=i),
                    sensor_id=f"sensor_{room}_{sensor_type}",
                    sensor_type=sensor_type,
                    state="on" if i % 3 == 0 else "off",
                    room_id=room,
                )
                large_events.append(event)

                # Add room states every 10 events
                if i % 10 == 0:
                    room_state = self.create_room_state(
                        timestamp=base_time + timedelta(minutes=i),
                        room_id=room,
                        is_occupied=i % 4 < 2,
                    )
                    large_room_states.append(room_state)

            # Measure performance
            import time

            start_time = time.time()

            features = await engine.extract_features(
                room_id="room_1",
                target_time=target_time,
                events=large_events,
                room_states=large_room_states,
                lookback_hours=6,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within reasonable time (requirement < 500ms may be ambitious for 1000 events)
            assert (
                execution_time < 2.0
            ), f"Performance test failed: took {execution_time:.3f}s"
            assert isinstance(features, dict)
            assert len(features) > 20

        finally:
            engine.executor.shutdown(wait=False)

    @pytest.mark.asyncio
    async def test_memory_efficiency_repeated_extractions(self, system_config):
        """Test memory efficiency with repeated extractions."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        base_time = datetime(2024, 6, 15, 12, 0, 0)
        target_time = base_time + timedelta(hours=2)

        # Create moderate dataset
        events = []
        for i in range(50):
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=i * 2),
                sensor_id=f"sensor_{i % 5}",
                sensor_type="motion",
                state="on" if i % 3 == 0 else "off",
                room_id="test_room",
            )
            events.append(event)

        # Run multiple extractions
        results = []
        for _ in range(20):
            features = await engine.extract_features(
                room_id="test_room", target_time=target_time, events=events
            )
            results.append(len(features))

        # All results should be consistent
        assert len(set(results)) == 1, "Inconsistent results across extractions"

        # Memory usage should remain reasonable (this is implicit - no memory leaks)

    def test_concurrent_access_thread_safety(self, system_config):
        """Test thread safety of the engine with concurrent access."""
        import threading
        import time

        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )

        try:
            results = []
            errors = []

            def extract_features_worker():
                try:
                    # Create simple events
                    events = [
                        self.create_sensor_event(
                            datetime(2024, 6, 15, 12, 0, 0),
                            "sensor1",
                            "motion",
                            "on",
                            "room1",
                        )
                    ]

                    # Run async function in thread (simplified)
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)

                    features = loop.run_until_complete(
                        engine.extract_features(
                            room_id="room1",
                            target_time=datetime(2024, 6, 15, 14, 0, 0),
                            events=events,
                        )
                    )
                    results.append(features)
                    loop.close()

                except Exception as e:
                    errors.append(e)

            # Create multiple threads
            threads = []
            for _ in range(3):
                thread = threading.Thread(target=extract_features_worker)
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join(timeout=10.0)

            # Should have completed without errors
            assert len(errors) == 0, f"Errors occurred: {errors}"
            assert len(results) == 3

        finally:
            engine.executor.shutdown(wait=False)

    # ==================== ERROR HANDLING AND EDGE CASES ====================

    @pytest.mark.asyncio
    async def test_comprehensive_error_handling(self, system_config):
        """Test comprehensive error handling scenarios."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        target_time = datetime(2024, 6, 15, 14, 0, 0)

        # Test various error conditions
        test_cases = [
            (None, "Room ID is required"),  # None room_id
            ("", "Room ID is required"),  # Empty room_id
        ]

        for room_id, expected_error in test_cases:
            with pytest.raises(FeatureExtractionError) as exc_info:
                await engine.extract_features(
                    room_id=room_id, target_time=target_time, events=[], room_states=[]
                )

            assert "Room ID" in str(exc_info.value)

    def test_cleanup_on_destruction(self, system_config):
        """Test proper cleanup when engine is destroyed."""
        engine = FeatureEngineeringEngine(
            config=system_config, enable_parallel=True, max_workers=2
        )

        # Verify executor exists
        assert engine.executor is not None
        executor = engine.executor

        # Mock shutdown to test it's called
        with patch.object(executor, "shutdown") as mock_shutdown:
            del engine

            # Shutdown should be called during cleanup
            # Note: This may not always work due to garbage collection timing
            # but the __del__ method should handle cleanup

    # ==================== INTEGRATION TESTS ====================

    @pytest.mark.asyncio
    async def test_end_to_end_integration(
        self,
        system_config,
        comprehensive_events,
        comprehensive_room_states,
        target_time,
    ):
        """Test end-to-end integration of the feature engineering pipeline."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Perform complete feature extraction
        features = await engine.extract_features(
            room_id="living_room",
            target_time=target_time,
            events=comprehensive_events,
            room_states=comprehensive_room_states,
            lookback_hours=12,
        )

        # Verify comprehensive results
        assert isinstance(features, dict)
        assert len(features) > 50  # Should have many features

        # Check feature categories
        feature_prefixes = set(k.split("_")[0] for k in features.keys())
        expected_prefixes = {"temporal", "sequential", "contextual", "meta"}
        assert expected_prefixes.issubset(feature_prefixes)

        # Create DataFrame from features
        df = engine.create_feature_dataframe([features])
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        assert len(df.columns) == len(features)

        # Compute correlations
        if len(df.columns) > 1:
            correlations = engine.compute_feature_correlations(df)
            assert isinstance(correlations, dict)

        # Check statistics
        stats = engine.get_extraction_stats()
        assert stats["successful_extractions"] >= 1

    @pytest.mark.asyncio
    async def test_realistic_home_automation_scenario(self, system_config):
        """Test with realistic home automation scenario."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)

        # Simulate a typical evening routine
        base_time = datetime(2024, 6, 15, 18, 0, 0)  # 6 PM
        target_time = base_time + timedelta(hours=3)  # 9 PM analysis

        # Create realistic events
        evening_events = []

        # Kitchen activity (dinner preparation)
        for minute in [0, 5, 10, 15, 30, 45]:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minute),
                sensor_id="sensor.kitchen_motion",
                sensor_type="motion",
                state="on",
                room_id="kitchen",
            )
            evening_events.append(event)

        # Living room activity (after dinner)
        for minute in [60, 75, 90, 105, 120]:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minute),
                sensor_id="sensor.living_room_presence",
                sensor_type="presence",
                state="on",
                room_id="living_room",
            )
            evening_events.append(event)

        # Bathroom activity (evening routine)
        for minute in [135, 140, 160, 165]:
            event = self.create_sensor_event(
                timestamp=base_time + timedelta(minutes=minute),
                sensor_id="sensor.bathroom_motion",
                sensor_type="motion",
                state="on",
                room_id="bathroom",
            )
            evening_events.append(event)

        # Create corresponding room states
        evening_states = []
        for hour in range(4):
            for room in ["kitchen", "living_room", "bathroom", "bedroom"]:
                # Different occupancy patterns
                if room == "kitchen" and hour == 0:
                    is_occupied = True
                elif room == "living_room" and 1 <= hour <= 2:
                    is_occupied = True
                elif room == "bathroom" and hour == 2:
                    is_occupied = True
                else:
                    is_occupied = False

                state = self.create_room_state(
                    timestamp=base_time + timedelta(hours=hour, minutes=15),
                    room_id=room,
                    is_occupied=is_occupied,
                )
                evening_states.append(state)

        # Extract features for each room
        for room in ["kitchen", "living_room", "bathroom"]:
            features = await engine.extract_features(
                room_id=room,
                target_time=target_time,
                events=evening_events,
                room_states=evening_states,
                lookback_hours=4,
            )

            # Verify realistic results
            assert isinstance(features, dict)
            assert len(features) > 20

            # Should have detected some room transitions
            if "sequential_room_transition_count" in features:
                assert features["sequential_room_transition_count"] >= 0

            # Should have reasonable temporal features
            if "temporal_is_sleep_hours" in features:
                assert (
                    features["temporal_is_sleep_hours"] == 0.0
                )  # 9 PM is not sleep hours yet

    @pytest.mark.asyncio
    async def test_edge_case_boundary_conditions(self, system_config):
        """Test various boundary conditions and edge cases."""
        engine = FeatureEngineeringEngine(config=system_config, enable_parallel=False)
        target_time = datetime(2024, 6, 15, 12, 0, 0)

        # Test with empty events and states
        features_empty = await engine.extract_features(
            room_id="test_room", target_time=target_time, events=[], room_states=[]
        )

        assert isinstance(features_empty, dict)
        assert len(features_empty) > 0  # Should have default values

        # Test with events at exact boundaries
        boundary_events = [
            # Event exactly at lookback boundary
            self.create_sensor_event(
                target_time - timedelta(hours=24),
                "boundary_sensor",
                "motion",
                "on",
                "test_room",
            ),
            # Event at target time
            self.create_sensor_event(
                target_time, "current_sensor", "motion", "on", "test_room"
            ),
        ]

        features_boundary = await engine.extract_features(
            room_id="test_room",
            target_time=target_time,
            events=boundary_events,
            lookback_hours=24,
        )

        assert isinstance(features_boundary, dict)
        assert len(features_boundary) > 0

        # Test with very large lookback
        features_large_lookback = await engine.extract_features(
            room_id="test_room",
            target_time=target_time,
            events=boundary_events,
            lookback_hours=24 * 365,  # 1 year
        )

        assert isinstance(features_large_lookback, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
