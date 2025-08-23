"""Unit tests for feature engineering and extraction.

Covers:
- src/features/temporal.py (Temporal Feature Extraction)
- src/features/sequential.py (Sequential Pattern Features)
- src/features/contextual.py (Contextual Features)
- src/features/engineering.py (Feature Engineering Pipeline)
- src/features/store.py (Feature Store Management)

This test file consolidates testing for all feature engineering functionality.
"""

from collections import Counter, OrderedDict, deque
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.config import RoomConfig, SystemConfig
from src.core.constants import SensorType
from src.core.exceptions import ConfigurationError, FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureCache, FeatureRecord, FeatureStore

# Import the actual feature extraction classes
from src.features.temporal import TemporalFeatureExtractor


class TestTemporalFeatures:
    """Test temporal feature extraction."""

    @pytest.fixture
    def temporal_extractor(self):
        """Create temporal feature extractor."""
        return TemporalFeatureExtractor(timezone_offset=0)

    @pytest.fixture
    def sample_events(self):
        """Create sample sensor events for testing."""
        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        events = [
            Mock(
                spec=SensorEvent,
                room_id="living_room",
                sensor_id="motion_1",
                sensor_type="motion",
                state="on",
                timestamp=base_time,
                attributes={"confidence": 0.95, "temperature": 22.5},
            ),
            Mock(
                spec=SensorEvent,
                room_id="living_room",
                sensor_id="motion_1",
                sensor_type="motion",
                state="of",  # Test typo handling
                timestamp=base_time + timedelta(minutes=30),
                attributes={"confidence": 0.85, "temperature": 23.0},
            ),
        ]
        return events

    @pytest.fixture
    def sample_room_states(self):
        """Create sample room states for testing."""
        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        states = [
            Mock(
                spec=RoomState,
                room_id="living_room",
                timestamp=base_time - timedelta(hours=1),
                is_occupied=False,
                occupancy_confidence=0.8,
            ),
            Mock(
                spec=RoomState,
                room_id="living_room",
                timestamp=base_time,
                is_occupied=True,
                occupancy_confidence=0.9,
            ),
        ]
        return states

    def test_initialization_with_timezone_offset(self):
        """Test temporal extractor initialization with timezone offset."""
        extractor = TemporalFeatureExtractor(timezone_offset=-8)  # PST
        assert extractor.timezone_offset == -8
        assert isinstance(extractor.feature_cache, dict)
        assert isinstance(extractor.temporal_cache, dict)

    def test_extract_features_with_empty_events(self, temporal_extractor):
        """Test feature extraction with empty events list returns defaults."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        features = temporal_extractor.extract_features([], target_time)

        # Should return default features
        assert isinstance(features, dict)
        assert "time_since_last_event" in features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert features["time_since_last_event"] == 3600.0

    def test_extract_features_with_valid_events(
        self, temporal_extractor, sample_events
    ):
        """Test feature extraction with valid events."""
        target_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC)
        features = temporal_extractor.extract_features(sample_events, target_time)

        # Verify all feature categories are present
        assert isinstance(features, dict)
        assert "time_since_last_event" in features
        assert "current_state_duration" in features
        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "avg_transition_interval" in features

        # Verify time-based calculations
        assert 0 <= features["time_since_last_event"] <= 86400.0
        assert -1 <= features["hour_sin"] <= 1
        assert -1 <= features["hour_cos"] <= 1

    def test_extract_time_since_features_with_empty_events(self, temporal_extractor):
        """Test time-since features with empty events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        features = temporal_extractor._extract_time_since_features([], target_time)

        assert features["time_since_last_event"] == 3600.0
        assert features["time_since_last_on"] == 3600.0
        assert features["time_since_last_off"] == 3600.0
        assert features["time_since_last_motion"] == 3600.0

    def test_extract_time_since_features_with_state_changes(
        self, temporal_extractor, sample_events
    ):
        """Test time-since features with actual state changes."""
        target_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC)
        features = temporal_extractor._extract_time_since_features(
            sample_events, target_time
        )

        # Should calculate proper time differences
        assert features["time_since_last_event"] == 1800.0  # 30 minutes
        assert all(value <= 86400.0 for value in features.values())  # 24 hour cap

    def test_extract_duration_features_empty_events(self, temporal_extractor):
        """Test duration features with empty events returns defaults."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        features = temporal_extractor._extract_duration_features([], target_time)

        assert features["current_state_duration"] == 0.0
        assert features["avg_on_duration"] == 1800.0
        assert features["duration_ratio"] == 1.0
        assert features["median_on_duration"] == 1800.0

    @patch("numpy.array")
    def test_extract_duration_features_with_state_changes(
        self, mock_np_array, temporal_extractor, sample_events
    ):
        """Test duration features with state changes and numpy operations."""
        # Mock numpy arrays to return controlled values
        mock_np_array.return_value = Mock(
            __getitem__=lambda self, key: [1800.0], spec=np.ndarray  # Mock array access
        )

        with patch("numpy.mean", return_value=1800.0), patch(
            "numpy.std", return_value=300.0
        ), patch("numpy.max", return_value=3600.0), patch(
            "numpy.median", return_value=1800.0
        ), patch(
            "numpy.percentile", return_value=2700.0
        ), patch(
            "numpy.concatenate", return_value=mock_np_array.return_value
        ):

            target_time = datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC)
            features = temporal_extractor._extract_duration_features(
                sample_events, target_time
            )

            assert "current_state_duration" in features
            assert "avg_on_duration" in features
            assert "on_duration_std" in features
            assert "duration_ratio" in features

    def test_extract_cyclical_features_timezone_adjustment(self, temporal_extractor):
        """Test cyclical features with timezone adjustments."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        # Test with different timezone offsets
        temporal_extractor.timezone_offset = -8  # PST
        features = temporal_extractor._extract_cyclical_features(target_time)

        assert "hour_sin" in features
        assert "hour_cos" in features
        assert "day_of_week_sin" in features
        assert "is_weekend" in features
        assert "is_work_hours" in features

        # Verify cyclical encoding range
        assert -1 <= features["hour_sin"] <= 1
        assert -1 <= features["hour_cos"] <= 1
        assert features["is_weekend"] in [0.0, 1.0]

    @patch("pandas.DataFrame")
    def test_extract_historical_patterns_with_dataframe(
        self, mock_df, temporal_extractor, sample_events
    ):
        """Test historical patterns extraction with pandas operations."""
        # Mock DataFrame operations
        mock_df_instance = Mock()
        mock_df_instance.groupby.return_value = Mock()
        mock_df_instance.groupby.return_value.__getitem__.return_value.agg.return_value.fillna.return_value = pd.DataFrame(
            {"mean": [0.5, 0.7], "std": [0.1, 0.2], "count": [10, 15]}
        )
        mock_df_instance.__getitem__.return_value.mean.return_value = 0.6
        mock_df_instance.__getitem__.return_value.values = np.array([0.5, 0.7])
        mock_df.return_value = mock_df_instance

        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with patch("numpy.mean", return_value=0.6), patch(
            "numpy.var", return_value=0.1
        ), patch(
            "numpy.corrcoef", return_value=np.array([[1.0, 0.5], [0.5, 1.0]])
        ), patch(
            "numpy.std", return_value=0.15
        ):

            features = temporal_extractor._extract_historical_patterns(
                sample_events, target_time
            )

            assert "hour_activity_rate" in features
            assert "overall_activity_rate" in features
            assert "pattern_strength" in features
            assert "trend_coefficient" in features

    def test_extract_generic_sensor_features_type_handling(self, temporal_extractor):
        """Test generic sensor features with different value types."""
        # Create events with various attribute types
        events = [
            Mock(
                spec=SensorEvent,
                state="on",
                sensor_type="motion",
                attributes={
                    "numeric_value": 25.5,
                    "boolean_value": True,
                    "string_value": "active",
                    "string_numeric": "42.0",
                },
            ),
            Mock(
                spec=SensorEvent,
                state="off",
                sensor_type="presence",
                attributes=None,  # Test None attributes
            ),
        ]

        features = temporal_extractor._extract_generic_sensor_features(events)

        assert "numeric_mean" in features
        assert "boolean_true_ratio" in features
        assert "string_unique_count" in features
        assert "motion_sensor_ratio" in features
        assert "presence_sensor_ratio" in features

    def test_get_feature_names_returns_complete_list(self, temporal_extractor):
        """Test get_feature_names returns all feature names."""
        feature_names = temporal_extractor.get_feature_names()

        assert isinstance(feature_names, list)
        assert len(feature_names) > 0
        assert "time_since_last_event" in feature_names
        assert "hour_sin" in feature_names
        assert "current_state_duration" in feature_names

    def test_clear_cache_functionality(self, temporal_extractor):
        """Test cache clearing functionality."""
        temporal_extractor.feature_cache["test_key"] = "test_value"
        temporal_extractor.clear_cache()
        assert len(temporal_extractor.feature_cache) == 0

    def test_extract_features_error_handling(self, temporal_extractor):
        """Test error handling in extract_features."""
        # Create an event that will cause an exception
        bad_event = Mock(spec=SensorEvent)
        bad_event.timestamp = "not_a_datetime"  # This will cause errors
        bad_event.room_id = "test_room"

        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(FeatureExtractionError) as exc_info:
            temporal_extractor.extract_features([bad_event], target_time)

        assert exc_info.value.feature_type == "temporal"
        assert exc_info.value.room_id == "test_room"


class TestSequentialFeatures:
    """Test sequential pattern features."""

    @pytest.fixture
    def sequential_extractor(self):
        """Create sequential feature extractor with mock config."""
        mock_config = Mock(spec=SystemConfig)
        return SequentialFeatureExtractor(mock_config)

    @pytest.fixture
    def multi_room_events(self):
        """Create events with multiple room transitions."""
        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        events = [
            Mock(
                spec=SensorEvent,
                room_id="living_room",
                sensor_id="motion_1",
                timestamp=base_time,
            ),
            Mock(
                spec=SensorEvent,
                room_id="kitchen",
                sensor_id="motion_2",
                timestamp=base_time + timedelta(minutes=5),
            ),
            Mock(
                spec=SensorEvent,
                room_id="living_room",
                sensor_id="motion_1",
                timestamp=base_time + timedelta(minutes=10),
            ),
        ]
        return events

    def test_initialization_with_config(self):
        """Test sequential extractor initialization."""
        mock_config = Mock(spec=SystemConfig)
        extractor = SequentialFeatureExtractor(mock_config)

        assert extractor.config == mock_config
        assert extractor.classifier is not None
        assert isinstance(extractor.sequence_cache, dict)

    def test_initialization_without_config(self):
        """Test initialization with None config."""
        extractor = SequentialFeatureExtractor(None)

        assert extractor.config is None
        assert extractor.classifier is None
        assert isinstance(extractor.sequence_cache, dict)

    def test_extract_features_empty_events(self, sequential_extractor):
        """Test feature extraction with empty events."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        features = sequential_extractor.extract_features([], target_time)

        # Should return default features
        assert isinstance(features, dict)
        assert "room_transition_count" in features
        assert "unique_rooms_visited" in features
        assert features["room_transition_count"] == 0.0

    def test_extract_room_transition_features_with_single_event(
        self, sequential_extractor
    ):
        """Test room transition features with less than 2 events."""
        event = Mock(spec=SensorEvent, room_id="living_room")
        features = sequential_extractor._extract_room_transition_features([event])

        assert features["room_transition_count"] == 0.0
        assert features["unique_rooms_visited"] == 1.0
        assert features["room_revisit_ratio"] == 0.0
        assert features["avg_room_dwell_time"] == 1800.0

    def test_extract_room_transition_features_with_transitions(
        self, sequential_extractor, multi_room_events
    ):
        """Test room transition feature extraction with real transitions."""
        features = sequential_extractor._extract_room_transition_features(
            multi_room_events
        )

        assert (
            features["room_transition_count"] == 2.0
        )  # 3 rooms visited = 2 transitions
        assert features["unique_rooms_visited"] == 2.0  # living_room and kitchen
        assert features["room_revisit_ratio"] > 0.0  # living_room visited twice
        assert "avg_room_dwell_time" in features

    @patch("numpy.diff")
    @patch("numpy.std")
    @patch("numpy.mean")
    def test_extract_velocity_features_with_numpy_operations(
        self, mock_mean, mock_std, mock_diff, sequential_extractor, multi_room_events
    ):
        """Test velocity features with mocked numpy operations."""
        # Mock numpy operations
        mock_diff.return_value = np.array([300.0, 300.0])  # 5 minute intervals
        mock_mean.return_value = 300.0
        mock_std.return_value = 50.0

        with patch("numpy.array") as mock_array:
            mock_array.return_value = Mock(
                std=Mock(return_value=50.0),
                astype=Mock(return_value=np.array([300.0, 300.0])),
            )

            features = sequential_extractor._extract_velocity_features(
                multi_room_events
            )

            assert "movement_velocity" in features
            assert "burst_events_count" in features
            assert "pause_events_count" in features
            assert "velocity_acceleration" in features

    def test_extract_sensor_sequence_features_with_counters(self, sequential_extractor):
        """Test sensor sequence features using Counter operations."""
        events = [
            Mock(spec=SensorEvent, sensor_id="motion_1", sensor_type="motion"),
            Mock(spec=SensorEvent, sensor_id="motion_1", sensor_type="motion"),
            Mock(spec=SensorEvent, sensor_id="door_1", sensor_type="door"),
        ]

        features = sequential_extractor._extract_sensor_sequence_features(events)

        assert features["unique_sensors_triggered"] == 2.0  # motion_1 and door_1
        assert features["sensor_revisit_count"] == 1.0  # motion_1 triggered twice
        assert "sensor_diversity_score" in features
        assert "door_sensor_ratio" in features

    def test_get_default_features_returns_complete_dict(self, sequential_extractor):
        """Test _get_default_features returns complete feature dictionary."""
        defaults = sequential_extractor._get_default_features()

        assert isinstance(defaults, dict)
        assert "room_transition_count" in defaults
        assert "unique_rooms_visited" in defaults
        assert "movement_velocity" in defaults
        assert "unique_sensors_triggered" in defaults

    def test_clear_cache_empties_sequence_cache(self, sequential_extractor):
        """Test clear_cache empties the sequence_cache dictionary."""
        sequential_extractor.sequence_cache["test"] = "value"
        sequential_extractor.clear_cache()
        assert len(sequential_extractor.sequence_cache) == 0

    def test_extract_features_error_handling(self, sequential_extractor):
        """Test error handling in extract_features."""
        # Create a problematic event
        bad_event = Mock(spec=SensorEvent)
        bad_event.room_id = "test_room"
        bad_event.timestamp = None  # This should cause problems

        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(FeatureExtractionError) as exc_info:
            sequential_extractor.extract_features([bad_event], target_time)

        assert exc_info.value.feature_type == "sequential"
        assert exc_info.value.room_id == "test_room"


class TestContextualFeatures:
    """Test contextual feature extraction."""

    @pytest.fixture
    def contextual_extractor(self):
        """Create contextual feature extractor."""
        mock_config = Mock(spec=SystemConfig)
        return ContextualFeatureExtractor(mock_config)

    @pytest.fixture
    def environmental_events(self):
        """Create events with environmental sensor data."""
        base_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        events = [
            Mock(
                spec=SensorEvent,
                sensor_id="temp_1",
                sensor_type=SensorType.CLIMATE,
                state="22.5",
                attributes={"temperature": 22.5, "humidity": 45.0},
                timestamp=base_time,
            ),
            Mock(
                spec=SensorEvent,
                sensor_id="light_1",
                sensor_type=SensorType.LIGHT,
                state="500",
                attributes={"lux": 500},
                timestamp=base_time + timedelta(minutes=15),
            ),
        ]
        return events

    def test_initialization_with_thresholds(self):
        """Test initialization with environmental thresholds."""
        extractor = ContextualFeatureExtractor()

        assert extractor.temp_thresholds["cold"] == 18.0
        assert extractor.temp_thresholds["comfortable"] == 22.0
        assert extractor.humidity_thresholds["dry"] == 40.0
        assert extractor.light_thresholds["dark"] == 100.0
        assert isinstance(extractor.context_cache, dict)

    def test_extract_features_empty_events_returns_defaults(self, contextual_extractor):
        """Test feature extraction with empty events returns defaults."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        features = contextual_extractor.extract_features([], None, target_time)

        defaults = contextual_extractor._get_default_features()
        assert features == defaults
        assert isinstance(features, dict)

    def test_extract_environmental_features_with_climate_sensors(
        self, contextual_extractor, environmental_events
    ):
        """Test environmental feature extraction with climate sensors."""
        # Filter for environmental events
        env_events = [
            e
            for e in environmental_events
            if e.sensor_type in [SensorType.CLIMATE, SensorType.LIGHT]
        ]

        with patch.object(
            contextual_extractor, "_extract_numeric_values", return_value=[22.5, 45.0]
        ), patch.object(
            contextual_extractor, "_is_realistic_value", return_value=True
        ), patch.object(
            contextual_extractor, "_calculate_trend", return_value=0.1
        ), patch.object(
            contextual_extractor, "_calculate_change_rate", return_value=0.05
        ):

            features = contextual_extractor._extract_environmental_features(env_events)

            assert "temperature_mean" in features
            assert "humidity_mean" in features
            assert "light_mean" in features
            assert "temperature_comfort_zone" in features

    def test_extract_door_state_features_with_door_sensors(self, contextual_extractor):
        """Test door state feature extraction."""
        door_events = [
            Mock(
                spec=SensorEvent,
                sensor_type=SensorType.DOOR,
                state="open",
                timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ),
            Mock(
                spec=SensorEvent,
                sensor_type=SensorType.DOOR,
                state="closed",
                timestamp=datetime(2024, 1, 15, 12, 30, 0, tzinfo=UTC),
            ),
        ]

        features = contextual_extractor._extract_door_state_features(door_events)

        assert "doors_currently_open" in features
        assert "door_transition_count" in features
        assert "avg_door_open_duration" in features
        assert "door_open_ratio" in features

    @patch("numpy.corrcoef")
    def test_extract_multi_room_features_with_correlation(
        self, mock_corrcoef, contextual_extractor
    ):
        """Test multi-room feature extraction with room correlation."""
        # Mock correlation coefficient calculation
        mock_corrcoef.return_value = np.array([[1.0, 0.7], [0.7, 1.0]])

        multi_room_events = [
            Mock(
                spec=SensorEvent,
                room_id="living_room",
                state="on",
                timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ),
            Mock(
                spec=SensorEvent,
                room_id="kitchen",
                state="on",
                timestamp=datetime(2024, 1, 15, 12, 5, 0, tzinfo=UTC),
            ),
        ]

        room_states = [
            Mock(
                spec=RoomState,
                room_id="living_room",
                is_occupied=True,
                timestamp=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            ),
            Mock(
                spec=RoomState,
                room_id="kitchen",
                is_occupied=False,
                timestamp=datetime(2024, 1, 15, 12, 5, 0, tzinfo=UTC),
            ),
        ]

        features = contextual_extractor._extract_multi_room_features(
            multi_room_events, room_states
        )

        assert "active_rooms_count" in features
        assert "simultaneous_occupancy_ratio" in features
        assert "room_activity_correlation" in features

    def test_is_realistic_value_with_sensor_validation(self, contextual_extractor):
        """Test realistic value validation for different sensor types."""
        # Test temperature validation
        assert contextual_extractor._is_realistic_value(25.0, "temperature") is True
        assert contextual_extractor._is_realistic_value(-60.0, "temperature") is False
        assert contextual_extractor._is_realistic_value(110.0, "temperature") is False

        # Test humidity validation
        assert contextual_extractor._is_realistic_value(50.0, "humidity") is True
        assert contextual_extractor._is_realistic_value(-10.0, "humidity") is False
        assert contextual_extractor._is_realistic_value(120.0, "humidity") is False

        # Test lux validation
        assert contextual_extractor._is_realistic_value(500.0, "lux") is True
        assert contextual_extractor._is_realistic_value(-100.0, "lux") is False

        # Test unknown sensor type
        assert contextual_extractor._is_realistic_value(42.0, "unknown") is True

    def test_calculate_trend_with_linear_slope(self, contextual_extractor):
        """Test trend calculation with linear slope calculation."""
        values = [10.0, 15.0, 20.0, 25.0]  # Positive trend
        trend = contextual_extractor._calculate_trend(values)
        assert trend > 0  # Should detect positive trend

        values = [25.0, 20.0, 15.0, 10.0]  # Negative trend
        trend = contextual_extractor._calculate_trend(values)
        assert trend < 0  # Should detect negative trend

        # Test with insufficient data
        trend = contextual_extractor._calculate_trend([10.0])
        assert trend == 0.0

    def test_clear_cache_functionality(self, contextual_extractor):
        """Test context cache clearing."""
        contextual_extractor.context_cache["test"] = "value"
        contextual_extractor.clear_cache()
        assert len(contextual_extractor.context_cache) == 0

    def test_extract_features_error_handling(self, contextual_extractor):
        """Test error handling in extract_features."""
        bad_event = Mock(spec=SensorEvent)
        bad_event.room_id = "test_room"
        bad_event.timestamp = "invalid"

        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(FeatureExtractionError) as exc_info:
            contextual_extractor.extract_features([bad_event], None, target_time)

        assert exc_info.value.feature_type == "contextual"
        assert exc_info.value.room_id == "test_room"


class TestFeatureEngineering:
    """Test feature engineering pipeline."""

    @pytest.fixture
    def mock_config(self):
        """Create mock system config."""
        config = Mock(spec=SystemConfig)
        config.rooms = {"living_room": Mock(spec=RoomConfig)}
        return config

    @pytest.fixture
    def feature_engine(self, mock_config):
        """Create feature engineering engine."""
        return FeatureEngineeringEngine(config=mock_config, enable_parallel=False)

    def test_initialization_with_default_parameters(self):
        """Test engine initialization with defaults."""
        with patch("src.features.engineering.get_config") as mock_get_config:
            mock_get_config.return_value = Mock(spec=SystemConfig)
            engine = FeatureEngineeringEngine()

            assert engine.enable_parallel is True
            assert engine.max_workers == 3
            assert isinstance(engine.stats, dict)
            assert "total_extractions" in engine.stats

    def test_initialization_with_custom_config(self, mock_config):
        """Test initialization with custom SystemConfig."""
        engine = FeatureEngineeringEngine(
            config=mock_config, enable_parallel=False, max_workers=2
        )

        assert engine.config == mock_config
        assert engine.enable_parallel is False
        assert engine.max_workers == 2
        assert engine.temporal_extractor is not None
        assert engine.sequential_extractor is not None
        assert engine.contextual_extractor is not None

    def test_initialization_extractor_setup(self, mock_config):
        """Test that all extractors are properly initialized."""
        engine = FeatureEngineeringEngine(config=mock_config)

        assert isinstance(engine.temporal_extractor, TemporalFeatureExtractor)
        assert isinstance(engine.sequential_extractor, SequentialFeatureExtractor)
        assert isinstance(engine.contextual_extractor, ContextualFeatureExtractor)

        # Verify sequential and contextual extractors have config
        assert engine.sequential_extractor.config == mock_config
        assert engine.contextual_extractor.config == mock_config

    @pytest.mark.asyncio
    async def test_extract_features_validation_empty_room_id(self, feature_engine):
        """Test extract_features raises error for empty room_id."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(FeatureExtractionError) as exc_info:
            await feature_engine.extract_features(
                room_id="", target_time=target_time  # Empty room_id should raise error
            )

        assert "room_id cannot be empty" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_extract_features_with_none_room_id(self, feature_engine):
        """Test extract_features raises error for None room_id."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        with pytest.raises(FeatureExtractionError):
            await feature_engine.extract_features(room_id=None, target_time=target_time)

    @pytest.mark.asyncio
    async def test_extract_features_sequential_processing(self, feature_engine):
        """Test sequential feature extraction processing."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)

        # Mock the individual extractors
        with patch.object(
            feature_engine.temporal_extractor,
            "extract_features",
            return_value={"temp_feature": 1.0},
        ), patch.object(
            feature_engine.sequential_extractor,
            "extract_features",
            return_value={"seq_feature": 2.0},
        ), patch.object(
            feature_engine.contextual_extractor,
            "extract_features",
            return_value={"ctx_feature": 3.0},
        ):

            features = await feature_engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=[],
                feature_types=["temporal", "sequential", "contextual"],
            )

            assert "temporal_temp_feature" in features
            assert "sequential_seq_feature" in features
            assert "contextual_ctx_feature" in features
            assert features["temporal_temp_feature"] == 1.0

    def test_get_feature_names_with_feature_types(self, feature_engine):
        """Test get_feature_names with specific feature types."""
        # Mock the extractor methods
        with patch.object(
            feature_engine.temporal_extractor,
            "get_feature_names",
            return_value=["temp1", "temp2"],
        ), patch.object(
            feature_engine.sequential_extractor,
            "get_feature_names",
            return_value=["seq1", "seq2"],
        ), patch.object(
            feature_engine.contextual_extractor,
            "get_feature_names",
            return_value=["ctx1", "ctx2"],
        ):

            # Test with specific feature types
            names = feature_engine.get_feature_names(feature_types=["temporal"])
            assert "temporal_temp1" in names
            assert "temporal_temp2" in names
            assert len([n for n in names if n.startswith("sequential_")]) == 0

            # Test with all feature types
            all_names = feature_engine.get_feature_names()
            assert len([n for n in all_names if n.startswith("temporal_")]) == 2
            assert len([n for n in all_names if n.startswith("sequential_")]) == 2
            assert len([n for n in all_names if n.startswith("contextual_")]) == 2

    def test_validate_configuration_with_none_config(self, mock_config):
        """Test configuration validation with None config."""
        # Test with _original_config_was_none = True (should not raise error)
        engine = FeatureEngineeringEngine(config=None)
        engine._original_config_was_none = True
        result = engine.validate_configuration()
        assert result is True  # Should pass validation

        # Test with _original_config_was_none = False (should raise error)
        engine._original_config_was_none = False
        engine.config = None
        with pytest.raises(ConfigurationError):
            engine.validate_configuration()

    def test_validate_configuration_with_invalid_max_workers(self):
        """Test validation with invalid max_workers."""
        mock_config = Mock(spec=SystemConfig)

        with pytest.raises(ConfigurationError) as exc_info:
            FeatureEngineeringEngine(config=mock_config, max_workers=0)

        assert "max_workers must be >= 1" in str(exc_info.value)

    @patch("pandas.DataFrame")
    def test_compute_feature_correlations_with_dataframe(self, mock_df, feature_engine):
        """Test correlation computation with pandas DataFrame."""
        # Mock DataFrame operations
        mock_df_instance = Mock()
        mock_df_instance.corr.return_value = Mock()
        mock_df_instance.empty = False
        mock_df.return_value = mock_df_instance

        with patch("numpy.abs") as mock_abs, patch("numpy.triu") as mock_triu:
            mock_abs.return_value = Mock()
            mock_triu.return_value = Mock()

            feature_dicts = [{"feat1": 1.0, "feat2": 2.0}]
            result = feature_engine.compute_feature_correlations(feature_dicts)

            assert "correlation_matrix" in result
            assert "high_correlations" in result

    def test_reset_stats_functionality(self, feature_engine):
        """Test statistics reset functionality."""
        # Set some stats
        feature_engine.stats["total_extractions"] = 10
        feature_engine.stats["successful_extractions"] = 8

        feature_engine.reset_stats()

        assert feature_engine.stats["total_extractions"] == 0
        assert feature_engine.stats["successful_extractions"] == 0
        assert feature_engine.stats["failed_extractions"] == 0

    def test_clear_caches_calls_all_extractors(self, feature_engine):
        """Test clear_caches calls clear_cache on all extractors."""
        with patch.object(
            feature_engine.temporal_extractor, "clear_cache"
        ) as mock_temp, patch.object(
            feature_engine.sequential_extractor, "clear_cache"
        ) as mock_seq, patch.object(
            feature_engine.contextual_extractor, "clear_cache"
        ) as mock_ctx:

            feature_engine.clear_caches()

            mock_temp.assert_called_once()
            mock_seq.assert_called_once()
            mock_ctx.assert_called_once()

    def test_destructor_shuts_down_executor(self, mock_config):
        """Test __del__ shuts down ThreadPoolExecutor."""
        engine = FeatureEngineeringEngine(config=mock_config, enable_parallel=True)

        # Mock the executor
        mock_executor = Mock()
        engine.executor = mock_executor

        # Call destructor
        engine.__del__()

        mock_executor.shutdown.assert_called_once_with(wait=True)


class TestFeatureStore:
    """Test feature store management."""

    @pytest.fixture
    def feature_record(self):
        """Create sample feature record."""
        return FeatureRecord(
            room_id="living_room",
            target_time=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            features={"temp_feature": 1.0, "seq_feature": 2.0},
            extraction_time=datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC),
            lookback_hours=24,
            feature_types=["temporal", "sequential"],
            data_hash="abc123",
        )

    @pytest.fixture
    def feature_cache(self):
        """Create feature cache."""
        return FeatureCache(max_size=10)

    def test_feature_record_serialization(self, feature_record):
        """Test FeatureRecord to_dict and from_dict methods."""
        # Test serialization
        record_dict = feature_record.to_dict()
        assert isinstance(record_dict, dict)
        assert record_dict["room_id"] == "living_room"
        assert isinstance(record_dict["target_time"], str)  # Should be ISO format

        # Test deserialization
        restored_record = FeatureRecord.from_dict(record_dict)
        assert restored_record.room_id == feature_record.room_id
        assert restored_record.target_time == feature_record.target_time
        assert restored_record.features == feature_record.features

    @patch("datetime.now")
    def test_feature_record_is_valid_with_mock_datetime(self, mock_now, feature_record):
        """Test FeatureRecord validity with mocked datetime."""
        # Mock current time to be 1 hour after extraction
        mock_now.return_value = datetime(2024, 1, 15, 13, 0, 0, tzinfo=UTC)

        # Should be valid (within 24 hours)
        assert feature_record.is_valid(max_age_hours=24) is True

        # Mock current time to be 25 hours after extraction
        mock_now.return_value = datetime(2024, 1, 16, 13, 0, 0, tzinfo=UTC)

        # Should be invalid (older than 24 hours)
        assert feature_record.is_valid(max_age_hours=24) is False

    def test_feature_cache_initialization(self, feature_cache):
        """Test FeatureCache initialization."""
        assert feature_cache.max_size == 10
        assert isinstance(feature_cache.cache, OrderedDict)
        assert feature_cache.hit_count == 0
        assert feature_cache.miss_count == 0

    def test_feature_cache_make_key(self, feature_cache):
        """Test cache key generation."""
        target_time = datetime(2024, 1, 15, 12, 0, 0, tzinfo=UTC)
        key = feature_cache._make_key(
            room_id="living_room",
            target_time=target_time,
            lookback_hours=24,
            feature_types=["temporal"],
        )

        assert isinstance(key, str)
        assert "living_room" in key
        assert "temporal" in key

    def test_feature_cache_put_and_get(self, feature_cache, feature_record):
        """Test cache put and get operations."""
        key = "test_key"

        # Test put
        feature_cache.put(key, feature_record)
        assert key in feature_cache.cache

        # Test get with hit
        retrieved = feature_cache.get(key)
        assert retrieved == feature_record
        assert feature_cache.hit_count == 1

        # Test get with miss
        missing = feature_cache.get("nonexistent_key")
        assert missing is None
        assert feature_cache.miss_count == 1

    def test_feature_cache_lru_eviction(self, feature_record):
        """Test LRU eviction with max_size boundaries."""
        cache = FeatureCache(max_size=2)  # Small cache for testing

        # Add records up to capacity
        cache.put("key1", feature_record)
        cache.put("key2", feature_record)
        assert len(cache.cache) == 2

        # Add one more - should evict oldest
        cache.put("key3", feature_record)
        assert len(cache.cache) == 2
        assert "key1" not in cache.cache  # Should be evicted
        assert "key2" in cache.cache
        assert "key3" in cache.cache

    def test_feature_cache_get_stats(self, feature_cache, feature_record):
        """Test cache statistics collection."""
        # Perform some operations
        feature_cache.put("key1", feature_record)
        feature_cache.get("key1")  # Hit
        feature_cache.get("nonexistent")  # Miss

        stats = feature_cache.get_stats()

        assert stats["size"] == 1
        assert stats["max_size"] == 10
        assert stats["hit_count"] == 1
        assert stats["miss_count"] == 1
        assert "hit_ratio" in stats

    def test_feature_cache_clear(self, feature_cache, feature_record):
        """Test cache clearing."""
        feature_cache.put("key1", feature_record)
        assert len(feature_cache.cache) == 1

        feature_cache.clear()
        assert len(feature_cache.cache) == 0
        assert feature_cache.hit_count == 0
        assert feature_cache.miss_count == 0

    @pytest.mark.asyncio
    async def test_feature_store_initialization(self):
        """Test FeatureStore initialization."""
        mock_config = Mock(spec=SystemConfig)

        with patch("src.features.store.get_config", return_value=mock_config), patch(
            "src.features.store.get_database_manager"
        ) as mock_get_db:
            mock_get_db.return_value = AsyncMock()

            store = FeatureStore(persist_features=False)
            assert store.persist_features is False
            assert isinstance(store.cache, FeatureCache)
            assert store.engine is not None

    @pytest.mark.asyncio
    async def test_feature_store_context_manager(self):
        """Test FeatureStore async context manager."""
        mock_config = Mock(spec=SystemConfig)

        with patch("src.features.store.get_config", return_value=mock_config), patch(
            "src.features.store.get_database_manager"
        ) as mock_get_db:

            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            store = FeatureStore()

            # Test context manager entry
            async with store as s:
                assert s == store
                mock_db_manager.initialize.assert_called_once()

            # Context manager exit should not raise errors

    @pytest.mark.asyncio
    async def test_feature_store_health_check(self):
        """Test FeatureStore health check with component status."""
        mock_config = Mock(spec=SystemConfig)

        with patch("src.features.store.get_config", return_value=mock_config):
            store = FeatureStore(persist_features=False)

            # Mock engine and cache health
            store.engine = Mock()
            store.engine.validate_configuration.return_value = True

            health = await store.health_check()

            assert "cache" in health
            assert "engine" in health
            assert "overall_healthy" in health
            assert isinstance(health["overall_healthy"], bool)

    def test_feature_store_get_statistics(self):
        """Test FeatureStore statistics collection."""
        mock_config = Mock(spec=SystemConfig)

        with patch("src.features.store.get_config", return_value=mock_config):
            store = FeatureStore(persist_features=False)

            # Mock components
            store.cache = Mock()
            store.cache.get_stats.return_value = {"hit_ratio": 0.75}
            store.engine = Mock()
            store.engine.get_extraction_stats.return_value = {"total": 100}

            stats = store.get_statistics()

            assert "cache_stats" in stats
            assert "engine_stats" in stats
            assert "extraction_stats" in stats

    def test_feature_store_clear_cache(self):
        """Test FeatureStore cache clearing."""
        mock_config = Mock(spec=SystemConfig)

        with patch("src.features.store.get_config", return_value=mock_config):
            store = FeatureStore(persist_features=False)

            # Mock cache and engine
            store.cache = Mock()
            store.engine = Mock()

            store.clear_cache()

            store.cache.clear.assert_called_once()
            store.engine.clear_caches.assert_called_once()

    def test_feature_store_reset_stats(self):
        """Test FeatureStore statistics reset."""
        mock_config = Mock(spec=SystemConfig)

        with patch("src.features.store.get_config", return_value=mock_config):
            store = FeatureStore(persist_features=False)

            # Mock components
            store.cache = Mock()
            store.engine = Mock()

            store.reset_stats()

            # Verify reset calls
            store.engine.reset_stats.assert_called_once()
