"""
Unit tests for constants and enums.

Tests all enum classes and constant values used throughout the system.
"""

import pytest

from src.core.constants import (
    ABSENCE_STATES,
    API_ENDPOINTS,
    CAT_MOVEMENT_PATTERNS,
    CONTEXTUAL_FEATURE_NAMES,
    DB_TABLES,
    DEFAULT_CONFIDENCE_THRESHOLD,
    DEFAULT_MODEL_PARAMS,
    DOOR_CLOSED_STATES,
    DOOR_OPEN_STATES,
    HUMAN_MOVEMENT_PATTERNS,
    INVALID_STATES,
    MAX_SEQUENCE_GAP,
    MIN_EVENT_SEPARATION,
    MQTT_TOPICS,
    PRESENCE_STATES,
    SEQUENTIAL_FEATURE_NAMES,
    TEMPORAL_FEATURE_NAMES,
    EventType,
    ModelType,
    PredictionType,
    SensorState,
    SensorType,
)


class TestSensorType:
    """Test SensorType enum."""

    def test_sensor_type_values(self):
        """Test that all sensor types have correct values."""
        assert SensorType.PRESENCE.value == "presence"
        assert SensorType.DOOR.value == "door"
        assert SensorType.CLIMATE.value == "climate"
        assert SensorType.LIGHT.value == "light"
        assert SensorType.MOTION.value == "motion"

    def test_sensor_type_count(self):
        """Test that we have the expected number of sensor types."""
        assert len(SensorType) == 5

    def test_sensor_type_membership(self):
        """Test sensor type membership and iteration."""
        sensor_types = list(SensorType)
        assert SensorType.PRESENCE in sensor_types
        assert SensorType.DOOR in sensor_types
        assert SensorType.CLIMATE in sensor_types
        assert SensorType.LIGHT in sensor_types
        assert SensorType.MOTION in sensor_types

    def test_sensor_type_string_representation(self):
        """Test string representation of sensor types."""
        assert str(SensorType.PRESENCE) == "SensorType.PRESENCE"
        assert repr(SensorType.PRESENCE) == "<SensorType.PRESENCE: 'presence'>"


class TestSensorState:
    """Test SensorState enum."""

    def test_sensor_state_values(self):
        """Test that all sensor states have correct values."""
        assert SensorState.ON.value == "on"
        assert SensorState.OFF.value == "of"
        assert SensorState.OPEN.value == "open"
        assert SensorState.CLOSED.value == "closed"
        assert SensorState.UNKNOWN.value == "unknown"
        assert SensorState.UNAVAILABLE.value == "unavailable"

    def test_sensor_state_count(self):
        """Test that we have the expected number of sensor states."""
        assert len(SensorState) == 6

    def test_sensor_state_membership(self):
        """Test sensor state membership."""
        states = [state.value for state in SensorState]
        assert "on" in states
        assert "of" in states
        assert "open" in states
        assert "closed" in states
        assert "unknown" in states
        assert "unavailable" in states


class TestEventType:
    """Test EventType enum."""

    def test_event_type_values(self):
        """Test that all event types have correct values."""
        assert EventType.STATE_CHANGE.value == "state_change"
        assert EventType.PREDICTION.value == "prediction"
        assert EventType.MODEL_UPDATE.value == "model_update"
        assert EventType.ACCURACY_UPDATE.value == "accuracy_update"

    def test_event_type_count(self):
        """Test that we have the expected number of event types."""
        assert len(EventType) == 4


class TestModelType:
    """Test ModelType enum."""

    def test_model_type_values(self):
        """Test that all model types have correct values."""
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.HMM.value == "hmm"
        assert ModelType.GAUSSIAN_PROCESS.value == "gp"
        assert ModelType.ENSEMBLE.value == "ensemble"

    def test_model_type_count(self):
        """Test that we have the expected number of model types."""
        assert len(ModelType) == 5


class TestPredictionType:
    """Test PredictionType enum."""

    def test_prediction_type_values(self):
        """Test that all prediction types have correct values."""
        assert PredictionType.NEXT_OCCUPIED.value == "next_occupied"
        assert PredictionType.NEXT_VACANT.value == "next_vacant"
        assert PredictionType.OCCUPANCY_DURATION.value == "occupancy_duration"
        assert PredictionType.VACANCY_DURATION.value == "vacancy_duration"

    def test_prediction_type_count(self):
        """Test that we have the expected number of prediction types."""
        assert len(PredictionType) == 4


class TestStateConstants:
    """Test state-related constants."""

    def test_presence_states(self):
        """Test presence state constants."""
        assert PRESENCE_STATES == ["on"]
        assert len(PRESENCE_STATES) == 1
        assert SensorState.ON.value in PRESENCE_STATES

    def test_absence_states(self):
        """Test absence state constants."""
        assert ABSENCE_STATES == ["of"]
        assert len(ABSENCE_STATES) == 1
        assert SensorState.OFF.value in ABSENCE_STATES

    def test_door_states(self):
        """Test door state constants."""
        assert SensorState.OPEN.value in DOOR_OPEN_STATES
        assert SensorState.ON.value in DOOR_OPEN_STATES
        assert SensorState.CLOSED.value in DOOR_CLOSED_STATES
        assert SensorState.OFF.value in DOOR_CLOSED_STATES

        # Test that open and closed states don't overlap
        assert not set(DOOR_OPEN_STATES).intersection(set(DOOR_CLOSED_STATES))

    def test_invalid_states(self):
        """Test invalid state constants."""
        assert SensorState.UNKNOWN.value in INVALID_STATES
        assert SensorState.UNAVAILABLE.value in INVALID_STATES
        assert len(INVALID_STATES) >= 2

    def test_state_categories_dont_overlap(self):
        """Test that state categories are mutually exclusive where appropriate."""
        # Presence and absence states should not overlap
        assert not set(PRESENCE_STATES).intersection(set(ABSENCE_STATES))

        # Invalid states should not be in presence or absence states
        assert not set(INVALID_STATES).intersection(set(PRESENCE_STATES))
        assert not set(INVALID_STATES).intersection(set(ABSENCE_STATES))


class TestTimingConstants:
    """Test timing-related constants."""

    def test_min_event_separation(self):
        """Test minimum event separation constant."""
        assert isinstance(MIN_EVENT_SEPARATION, int)
        assert MIN_EVENT_SEPARATION > 0
        assert MIN_EVENT_SEPARATION == 5  # 5 seconds

    def test_max_sequence_gap(self):
        """Test maximum sequence gap constant."""
        assert isinstance(MAX_SEQUENCE_GAP, int)
        assert MAX_SEQUENCE_GAP > MIN_EVENT_SEPARATION
        assert MAX_SEQUENCE_GAP == 300  # 5 minutes

    def test_timing_constants_relationship(self):
        """Test relationship between timing constants."""
        assert MAX_SEQUENCE_GAP > MIN_EVENT_SEPARATION
        assert MAX_SEQUENCE_GAP >= 60  # At least 1 minute


class TestConfidenceConstants:
    """Test confidence-related constants."""

    def test_default_confidence_threshold(self):
        """Test default confidence threshold."""
        assert isinstance(DEFAULT_CONFIDENCE_THRESHOLD, float)
        assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.7


class TestFeatureNames:
    """Test feature name constants."""

    def test_temporal_feature_names(self):
        """Test temporal feature names."""
        assert isinstance(TEMPORAL_FEATURE_NAMES, list)
        assert len(TEMPORAL_FEATURE_NAMES) > 0

        expected_features = [
            "time_since_last_change",
            "current_state_duration",
            "hour_sin",
            "hour_cos",
            "day_sin",
            "day_cos",
            "week_sin",
            "week_cos",
            "is_weekend",
            "is_holiday",
        ]

        for feature in expected_features:
            assert feature in TEMPORAL_FEATURE_NAMES

    def test_sequential_feature_names(self):
        """Test sequential feature names."""
        assert isinstance(SEQUENTIAL_FEATURE_NAMES, list)
        assert len(SEQUENTIAL_FEATURE_NAMES) > 0

        expected_features = [
            "room_transition_1gram",
            "room_transition_2gram",
            "room_transition_3gram",
            "movement_velocity",
            "trigger_sequence_pattern",
            "cross_room_correlation",
        ]

        for feature in expected_features:
            assert feature in SEQUENTIAL_FEATURE_NAMES

    def test_contextual_feature_names(self):
        """Test contextual feature names."""
        assert isinstance(CONTEXTUAL_FEATURE_NAMES, list)
        assert len(CONTEXTUAL_FEATURE_NAMES) > 0

        expected_features = [
            "temperature",
            "humidity",
            "light_level",
            "door_state",
            "other_rooms_occupied",
            "historical_pattern_similarity",
        ]

        for feature in expected_features:
            assert feature in CONTEXTUAL_FEATURE_NAMES

    def test_feature_names_no_duplicates(self):
        """Test that feature names don't have duplicates within categories."""
        assert len(TEMPORAL_FEATURE_NAMES) == len(set(TEMPORAL_FEATURE_NAMES))
        assert len(SEQUENTIAL_FEATURE_NAMES) == len(set(SEQUENTIAL_FEATURE_NAMES))
        assert len(CONTEXTUAL_FEATURE_NAMES) == len(set(CONTEXTUAL_FEATURE_NAMES))

    def test_feature_names_no_cross_category_duplicates(self):
        """Test that feature names don't overlap between categories."""
        all_features = (
            set(TEMPORAL_FEATURE_NAMES)
            | set(SEQUENTIAL_FEATURE_NAMES)
            | set(CONTEXTUAL_FEATURE_NAMES)
        )
        total_features = (
            len(TEMPORAL_FEATURE_NAMES)
            + len(SEQUENTIAL_FEATURE_NAMES)
            + len(CONTEXTUAL_FEATURE_NAMES)
        )
        assert len(all_features) == total_features


class TestMQTTTopics:
    """Test MQTT topic constants."""

    def test_mqtt_topics_structure(self):
        """Test MQTT topics dictionary structure."""
        assert isinstance(MQTT_TOPICS, dict)
        assert len(MQTT_TOPICS) > 0

        expected_topics = [
            "predictions",
            "confidence",
            "accuracy",
            "status",
            "health",
        ]
        for topic in expected_topics:
            assert topic in MQTT_TOPICS

    def test_mqtt_topic_formatting(self):
        """Test MQTT topic string formatting."""
        # Test that topics contain placeholders
        assert "{topic_prefix}" in MQTT_TOPICS["predictions"]
        assert "{room_id}" in MQTT_TOPICS["predictions"]

        # Test topic formatting
        formatted = MQTT_TOPICS["predictions"].format(
            topic_prefix="test/occupancy", room_id="living_room"
        )
        assert formatted == "test/occupancy/living_room/prediction"

    def test_mqtt_system_topics(self):
        """Test system-level MQTT topics."""
        assert "{topic_prefix}/system/status" == MQTT_TOPICS["status"]
        assert "{topic_prefix}/system/health" == MQTT_TOPICS["health"]


class TestDatabaseTables:
    """Test database table constants."""

    def test_db_tables_structure(self):
        """Test database tables dictionary structure."""
        assert isinstance(DB_TABLES, dict)
        assert len(DB_TABLES) > 0

        expected_tables = [
            "sensor_events",
            "predictions",
            "model_accuracy",
            "room_states",
            "feature_store",
        ]
        for table in expected_tables:
            assert table in DB_TABLES

    def test_db_table_names(self):
        """Test database table names are valid."""
        for table_key, table_name in DB_TABLES.items():
            assert isinstance(table_name, str)
            assert len(table_name) > 0
            assert table_name.replace("_", "").isalnum()  # Valid SQL table name


class TestAPIEndpoints:
    """Test API endpoint constants."""

    def test_api_endpoints_structure(self):
        """Test API endpoints dictionary structure."""
        assert isinstance(API_ENDPOINTS, dict)
        assert len(API_ENDPOINTS) > 0

        expected_endpoints = [
            "predictions",
            "accuracy",
            "health",
            "retrain",
            "rooms",
            "sensors",
        ]
        for endpoint in expected_endpoints:
            assert endpoint in API_ENDPOINTS

    def test_api_endpoint_formatting(self):
        """Test API endpoint string formatting."""
        # Test endpoint with parameter
        predictions_endpoint = API_ENDPOINTS["predictions"]
        assert "{room_id}" in predictions_endpoint

        formatted = predictions_endpoint.format(room_id="living_room")
        assert formatted == "/api/predictions/living_room"

    def test_api_endpoints_valid_paths(self):
        """Test that API endpoints are valid URL paths."""
        for endpoint_key, endpoint_path in API_ENDPOINTS.items():
            assert endpoint_path.startswith("/api/")
            assert len(endpoint_path) > 5  # More than just "/api/"


class TestModelParameters:
    """Test default model parameters."""

    def test_default_model_params_structure(self):
        """Test default model parameters dictionary structure."""
        assert isinstance(DEFAULT_MODEL_PARAMS, dict)
        assert len(DEFAULT_MODEL_PARAMS) > 0

        # Test that all model types have parameters
        for model_type in ModelType:
            assert model_type in DEFAULT_MODEL_PARAMS

    def test_lstm_parameters(self):
        """Test LSTM model parameters."""
        lstm_params = DEFAULT_MODEL_PARAMS[ModelType.LSTM]
        assert isinstance(lstm_params, dict)

        expected_params = [
            "sequence_length",
            "hidden_units",
            "dropout",
            "learning_rate",
        ]
        for param in expected_params:
            assert param in lstm_params

        # Test parameter value types and ranges
        assert isinstance(lstm_params["sequence_length"], int)
        assert lstm_params["sequence_length"] > 0
        assert isinstance(lstm_params["hidden_units"], int)
        assert lstm_params["hidden_units"] > 0
        assert isinstance(lstm_params["dropout"], float)
        assert 0.0 <= lstm_params["dropout"] <= 1.0
        assert isinstance(lstm_params["learning_rate"], float)
        assert lstm_params["learning_rate"] > 0.0

    def test_xgboost_parameters(self):
        """Test XGBoost model parameters."""
        xgb_params = DEFAULT_MODEL_PARAMS[ModelType.XGBOOST]
        assert isinstance(xgb_params, dict)

        expected_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
        ]
        for param in expected_params:
            assert param in xgb_params

        # Test parameter value types and ranges
        assert isinstance(xgb_params["n_estimators"], int)
        assert xgb_params["n_estimators"] > 0
        assert isinstance(xgb_params["max_depth"], int)
        assert xgb_params["max_depth"] > 0
        assert isinstance(xgb_params["learning_rate"], float)
        assert xgb_params["learning_rate"] > 0.0
        assert isinstance(xgb_params["subsample"], float)
        assert 0.0 < xgb_params["subsample"] <= 1.0

    def test_hmm_parameters(self):
        """Test HMM model parameters."""
        hmm_params = DEFAULT_MODEL_PARAMS[ModelType.HMM]
        assert isinstance(hmm_params, dict)

        expected_params = ["n_components", "covariance_type", "n_iter"]
        for param in expected_params:
            assert param in hmm_params

    def test_gp_parameters(self):
        """Test Gaussian Process model parameters."""
        gp_params = DEFAULT_MODEL_PARAMS[ModelType.GAUSSIAN_PROCESS]
        assert isinstance(gp_params, dict)

        expected_params = ["kernel", "alpha", "n_restarts_optimizer"]
        for param in expected_params:
            assert param in gp_params


class TestMovementPatterns:
    """Test human and cat movement pattern constants."""

    def test_human_movement_patterns(self):
        """Test human movement pattern constants."""
        assert isinstance(HUMAN_MOVEMENT_PATTERNS, dict)

        expected_keys = [
            "min_duration_seconds",
            "max_velocity_ms",
            "typical_room_sequence_length",
            "door_interaction_probability",
        ]
        for key in expected_keys:
            assert key in HUMAN_MOVEMENT_PATTERNS

        # Test value types and reasonableness
        assert isinstance(HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"], int)
        assert HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"] > 0
        assert isinstance(HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"], float)
        assert HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"] > 0.0
        assert isinstance(HUMAN_MOVEMENT_PATTERNS["typical_room_sequence_length"], int)
        assert HUMAN_MOVEMENT_PATTERNS["typical_room_sequence_length"] > 0
        assert isinstance(
            HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"], float
        )
        assert 0.0 <= HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"] <= 1.0

    def test_cat_movement_patterns(self):
        """Test cat movement pattern constants."""
        assert isinstance(CAT_MOVEMENT_PATTERNS, dict)

        expected_keys = [
            "min_duration_seconds",
            "max_velocity_ms",
            "typical_room_sequence_length",
            "door_interaction_probability",
        ]
        for key in expected_keys:
            assert key in CAT_MOVEMENT_PATTERNS

        # Test value types and reasonableness
        assert isinstance(CAT_MOVEMENT_PATTERNS["min_duration_seconds"], int)
        assert CAT_MOVEMENT_PATTERNS["min_duration_seconds"] > 0
        assert isinstance(CAT_MOVEMENT_PATTERNS["max_velocity_ms"], float)
        assert CAT_MOVEMENT_PATTERNS["max_velocity_ms"] > 0.0
        assert isinstance(CAT_MOVEMENT_PATTERNS["typical_room_sequence_length"], int)
        assert CAT_MOVEMENT_PATTERNS["typical_room_sequence_length"] > 0
        assert isinstance(CAT_MOVEMENT_PATTERNS["door_interaction_probability"], float)
        assert 0.0 <= CAT_MOVEMENT_PATTERNS["door_interaction_probability"] <= 1.0

    def test_movement_pattern_differences(self):
        """Test that human and cat movement patterns have logical differences."""
        # Cats should generally move faster than humans
        assert (
            CAT_MOVEMENT_PATTERNS["max_velocity_ms"]
            >= HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"]
        )

        # Cats should have lower door interaction probability
        assert (
            CAT_MOVEMENT_PATTERNS["door_interaction_probability"]
            < HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"]
        )

        # Cats may have different minimum duration (can be shorter)
        assert (
            CAT_MOVEMENT_PATTERNS["min_duration_seconds"]
            <= HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"]
        )


@pytest.mark.unit
class TestConstantsIntegration:
    """Integration tests for constants used together."""

    def test_sensor_state_coverage(self):
        """Test that state constants cover all sensor states appropriately."""
        all_defined_states = set(
            PRESENCE_STATES
            + ABSENCE_STATES
            + DOOR_OPEN_STATES
            + DOOR_CLOSED_STATES
            + INVALID_STATES
        )
        all_enum_states = set(state.value for state in SensorState)

        # All enum states should be covered by at least one category
        # (some states may be in multiple categories, like "on" and "off")
        for state in all_enum_states:
            assert (
                state in all_defined_states
            ), f"State '{state}' not covered by any category"

    def test_model_type_param_consistency(self):
        """Test that all model types have corresponding parameters."""
        model_types_in_enum = set(ModelType)
        model_types_with_params = set(DEFAULT_MODEL_PARAMS.keys())

        assert model_types_in_enum == model_types_with_params

    def test_constant_value_consistency(self):
        """Test that constants have consistent and reasonable values."""
        # Timing constants should be reasonable
        assert MIN_EVENT_SEPARATION < 60  # Less than 1 minute
        assert MAX_SEQUENCE_GAP <= 600  # Less than 10 minutes

        # Confidence threshold should be reasonable
        assert 0.5 <= DEFAULT_CONFIDENCE_THRESHOLD <= 0.9

        # Movement patterns should have reasonable values
        assert (
            HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"] >= 10
        )  # At least 10 seconds
        assert CAT_MOVEMENT_PATTERNS["min_duration_seconds"] >= 1  # At least 1 second

        # Velocities should be reasonable (m/s)
        assert (
            HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"] <= 5.0
        )  # Reasonable human speed
        assert CAT_MOVEMENT_PATTERNS["max_velocity_ms"] <= 10.0  # Reasonable cat speed

    def test_string_constants_format(self):
        """Test that string constants follow expected formats."""
        # MQTT topics should use proper placeholder format
        for topic_name, topic_template in MQTT_TOPICS.items():
            if "{" in topic_template:
                assert topic_template.count("{") == topic_template.count("}")

        # API endpoints should start with /api/
        for endpoint_name, endpoint_path in API_ENDPOINTS.items():
            assert endpoint_path.startswith("/api/")

        # Database table names should be valid identifiers
        for table_key, table_name in DB_TABLES.items():
            assert table_name.replace("_", "").isalnum()
            assert not table_name.startswith("_")
            assert not table_name.endswith("_")
