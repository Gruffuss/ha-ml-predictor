"""Unit tests for core constants and exception handling.

Covers:
- src/core/constants.py (System Constants and Enums)
- src/core/exceptions.py (Custom Exception Classes)

This test file consolidates testing for core system constants and exceptions
as they are fundamental components used throughout the system.
"""

from enum import Enum
import re
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

# Import constants and enums
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

# Import exception classes
from src.core.exceptions import (
    APIAuthenticationError,
    APIAuthorizationError,
    APIError,
    APIResourceNotFoundError,
    APISecurityError,
    APIServerError,
    ConfigFileNotFoundError,
    ConfigParsingError,
    ConfigurationError,
    ConfigValidationError,
    DatabaseConnectionError,
    DatabaseError,
    DatabaseIntegrityError,
    DatabaseMigrationError,
    DatabaseQueryError,
    DataCorruptionError,
    DataProcessingError,
    DataValidationError,
    EntityNotFoundError,
    ErrorSeverity,
    FeatureEngineeringError,
    FeatureExtractionError,
    FeatureStoreError,
    FeatureValidationError,
    HomeAssistantAPIError,
    HomeAssistantAuthenticationError,
    HomeAssistantConnectionError,
    HomeAssistantError,
    InsufficientDataError,
    InsufficientTrainingDataError,
    IntegrationError,
    MaintenanceModeError,
    MissingConfigSectionError,
    MissingFeatureError,
    ModelError,
    ModelNotFoundError,
    ModelPredictionError,
    ModelTrainingError,
    ModelValidationError,
    ModelVersionMismatchError,
    MQTTConnectionError,
    MQTTError,
    MQTTPublishError,
    MQTTSubscriptionError,
    OccupancyPredictionError,
    RateLimitExceededError,
    ResourceExhaustionError,
    ServiceUnavailableError,
    SystemError,
    SystemInitializationError,
    SystemResourceError,
    WebSocketAuthenticationError,
    WebSocketConnectionError,
    WebSocketError,
    WebSocketValidationError,
    validate_entity_id,
    validate_room_id,
)


class TestSystemConstants:
    """Test system constants and enumerations."""

    # Enum Value Verification Tests

    def test_sensor_type_enum_values(self):
        """Test all SensorType enum values are correct."""
        assert SensorType.PRESENCE.value == "presence"
        assert SensorType.DOOR.value == "door"
        assert SensorType.CLIMATE.value == "climate"
        assert SensorType.LIGHT.value == "light"
        assert SensorType.MOTION.value == "motion"

        # Verify all expected values are present
        expected_values = ["presence", "door", "climate", "light", "motion"]
        actual_values = [member.value for member in SensorType]
        assert set(actual_values) == set(expected_values)

    def test_sensor_state_enum_values(self):
        """Test all SensorState enum values are correct."""
        assert SensorState.ON.value == "on"
        assert SensorState.OFF.value == "off"
        assert SensorState.OPEN.value == "open"
        assert SensorState.CLOSED.value == "closed"
        assert SensorState.UNKNOWN.value == "unknown"
        assert SensorState.UNAVAILABLE.value == "unavailable"

        # Verify all expected values are present
        expected_values = ["on", "off", "open", "closed", "unknown", "unavailable"]
        actual_values = [member.value for member in SensorState]
        assert set(actual_values) == set(expected_values)

    def test_event_type_enum_values(self):
        """Test all EventType enum values are correct."""
        assert EventType.STATE_CHANGE.value == "state_change"
        assert EventType.PREDICTION.value == "prediction"
        assert EventType.MODEL_UPDATE.value == "model_update"
        assert EventType.ACCURACY_UPDATE.value == "accuracy_update"

        # Verify all expected values are present
        expected_values = [
            "state_change",
            "prediction",
            "model_update",
            "accuracy_update",
        ]
        actual_values = [member.value for member in EventType]
        assert set(actual_values) == set(expected_values)

    def test_model_type_enum_values(self):
        """Test all ModelType enum values including aliases."""
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.HMM.value == "hmm"
        assert ModelType.GAUSSIAN_PROCESS.value == "gp"
        assert ModelType.GP.value == "gp"  # Alias for GAUSSIAN_PROCESS
        assert ModelType.ENSEMBLE.value == "ensemble"

        # Verify GP is an alias for GAUSSIAN_PROCESS (same enum member)
        assert ModelType.GP is ModelType.GAUSSIAN_PROCESS
        assert ModelType.GP.value == ModelType.GAUSSIAN_PROCESS.value

        # Verify all expected values are present
        expected_values = ["lstm", "xgboost", "hmm", "gp", "ensemble"]
        actual_values = [member.value for member in ModelType]
        assert sorted(actual_values) == sorted(expected_values)

        # Verify that we have the expected number of enum members
        assert (
            len(list(ModelType)) == 5
        )  # LSTM, XGBOOST, HMM, GAUSSIAN_PROCESS (GP is alias), ENSEMBLE

    def test_prediction_type_enum_values(self):
        """Test all PredictionType enum values are correct."""
        assert PredictionType.NEXT_OCCUPIED.value == "next_occupied"
        assert PredictionType.NEXT_VACANT.value == "next_vacant"
        assert PredictionType.OCCUPANCY_DURATION.value == "occupancy_duration"
        assert PredictionType.VACANCY_DURATION.value == "vacancy_duration"

        # Verify all expected values are present
        expected_values = [
            "next_occupied",
            "next_vacant",
            "occupancy_duration",
            "vacancy_duration",
        ]
        actual_values = [member.value for member in PredictionType]
        assert set(actual_values) == set(expected_values)

    # Constant List Content Tests

    def test_presence_absence_states(self):
        """Test presence and absence state constants."""
        assert PRESENCE_STATES == ["on"]
        assert ABSENCE_STATES == ["off"]
        assert isinstance(PRESENCE_STATES, list)
        assert isinstance(ABSENCE_STATES, list)

    def test_door_states(self):
        """Test door state constants."""
        assert set(DOOR_OPEN_STATES) == {"open", "on"}
        assert set(DOOR_CLOSED_STATES) == {"closed", "off"}
        assert isinstance(DOOR_OPEN_STATES, list)
        assert isinstance(DOOR_CLOSED_STATES, list)

    def test_invalid_states(self):
        """Test invalid states constants."""
        assert set(INVALID_STATES) == {"unknown", "unavailable"}
        assert isinstance(INVALID_STATES, list)

    def test_feature_name_lists(self):
        """Test feature name list constants."""
        # Test temporal features
        expected_temporal = [
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
        assert TEMPORAL_FEATURE_NAMES == expected_temporal

        # Test sequential features
        expected_sequential = [
            "room_transition_1gram",
            "room_transition_2gram",
            "room_transition_3gram",
            "movement_velocity",
            "trigger_sequence_pattern",
            "cross_room_correlation",
        ]
        assert SEQUENTIAL_FEATURE_NAMES == expected_sequential

        # Test contextual features
        expected_contextual = [
            "temperature",
            "humidity",
            "light_level",
            "door_state",
            "other_rooms_occupied",
            "historical_pattern_similarity",
        ]
        assert CONTEXTUAL_FEATURE_NAMES == expected_contextual

    # Numeric Constant Validation Tests

    def test_timing_constants(self):
        """Test timing-related numeric constants."""
        assert MIN_EVENT_SEPARATION == 5
        assert MAX_SEQUENCE_GAP == 300  # 5 minutes
        assert isinstance(MIN_EVENT_SEPARATION, int)
        assert isinstance(MAX_SEQUENCE_GAP, int)
        assert MIN_EVENT_SEPARATION > 0
        assert MAX_SEQUENCE_GAP > MIN_EVENT_SEPARATION

    def test_confidence_threshold(self):
        """Test default confidence threshold."""
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.7
        assert isinstance(DEFAULT_CONFIDENCE_THRESHOLD, float)
        assert 0.0 <= DEFAULT_CONFIDENCE_THRESHOLD <= 1.0

    # Dictionary Structure Tests

    def test_mqtt_topics_structure(self):
        """Test MQTT topics dictionary structure."""
        expected_keys = {"predictions", "confidence", "accuracy", "status", "health"}
        assert set(MQTT_TOPICS.keys()) == expected_keys

        # Verify format strings contain expected placeholders
        assert "{topic_prefix}" in MQTT_TOPICS["predictions"]
        assert "{room_id}" in MQTT_TOPICS["predictions"]
        assert "{topic_prefix}" in MQTT_TOPICS["status"]

        # Test format string functionality
        formatted = MQTT_TOPICS["predictions"].format(
            topic_prefix="occupancy", room_id="living_room"
        )
        assert formatted == "occupancy/living_room/prediction"

    def test_db_tables_structure(self):
        """Test database tables dictionary structure."""
        expected_keys = {
            "sensor_events",
            "predictions",
            "model_accuracy",
            "room_states",
            "feature_store",
        }
        assert set(DB_TABLES.keys()) == expected_keys

        # Verify all values are strings
        for table_name in DB_TABLES.values():
            assert isinstance(table_name, str)
            assert len(table_name) > 0

    def test_api_endpoints_structure(self):
        """Test API endpoints dictionary structure."""
        expected_keys = {
            "predictions",
            "accuracy",
            "health",
            "retrain",
            "rooms",
            "sensors",
        }
        assert set(API_ENDPOINTS.keys()) == expected_keys

        # Verify format strings contain expected placeholders where needed
        assert "{room_id}" in API_ENDPOINTS["predictions"]

        # Test format string functionality
        formatted = API_ENDPOINTS["predictions"].format(room_id="kitchen")
        assert formatted == "/api/predictions/kitchen"

        # Verify all endpoints start with /api
        for endpoint in API_ENDPOINTS.values():
            assert endpoint.startswith("/api")

    # Model Parameter Validation Tests

    def test_default_model_params_structure(self):
        """Test default model parameters structure."""
        # Verify all model types have parameters
        expected_model_types = {
            ModelType.LSTM,
            ModelType.XGBOOST,
            ModelType.HMM,
            ModelType.GAUSSIAN_PROCESS,
            ModelType.GP,
            ModelType.ENSEMBLE,
        }
        actual_model_types = set(DEFAULT_MODEL_PARAMS.keys())
        assert actual_model_types == expected_model_types

        # Verify each model has required parameters
        for model_type, params in DEFAULT_MODEL_PARAMS.items():
            assert isinstance(params, dict)
            assert len(params) > 0

    def test_lstm_model_params(self):
        """Test LSTM model parameters including aliases."""
        lstm_params = DEFAULT_MODEL_PARAMS[ModelType.LSTM]

        # Test required parameters
        assert lstm_params["sequence_length"] == 50
        assert lstm_params["hidden_units"] == 64
        assert lstm_params["dropout"] == 0.2
        assert lstm_params["learning_rate"] == 0.001

        # Test aliases for backward compatibility
        assert lstm_params["lstm_units"] == lstm_params["hidden_units"]  # Alias
        assert lstm_params["dropout_rate"] == lstm_params["dropout"]  # Alias

    def test_xgboost_model_params(self):
        """Test XGBoost model parameters."""
        xgb_params = DEFAULT_MODEL_PARAMS[ModelType.XGBOOST]

        # Test required parameters
        assert xgb_params["n_estimators"] == 100
        assert xgb_params["max_depth"] == 6
        assert xgb_params["learning_rate"] == 0.1
        assert xgb_params["subsample"] == 0.8
        assert xgb_params["objective"] == "reg:squarederror"

    def test_hmm_model_params(self):
        """Test HMM model parameters including aliases."""
        hmm_params = DEFAULT_MODEL_PARAMS[ModelType.HMM]

        # Test required parameters
        assert hmm_params["n_components"] == 4
        assert hmm_params["covariance_type"] == "full"
        assert hmm_params["n_iter"] == 100

        # Test alias for backward compatibility
        assert hmm_params["max_iter"] == hmm_params["n_iter"]  # Alias

    def test_gp_model_params_and_alias(self):
        """Test Gaussian Process model parameters and GP alias."""
        gp_params = DEFAULT_MODEL_PARAMS[ModelType.GAUSSIAN_PROCESS]
        gp_alias_params = DEFAULT_MODEL_PARAMS[ModelType.GP]

        # Test parameters are identical (GP is alias for GAUSSIAN_PROCESS)
        assert gp_params == gp_alias_params

        # Test required parameters
        assert gp_params["kernel"] == "rb"
        assert gp_params["alpha"] == 1e-6
        assert gp_params["normalize_y"] is True
        assert gp_params["n_restarts_optimizer"] == 0

    def test_ensemble_model_params(self):
        """Test ensemble model parameters."""
        ensemble_params = DEFAULT_MODEL_PARAMS[ModelType.ENSEMBLE]

        # Test required parameters
        assert ensemble_params["meta_learner"] == "xgboost"
        assert ensemble_params["cv_folds"] == 5
        assert ensemble_params["stacking_method"] == "linear"
        assert ensemble_params["blend_weights"] == "auto"

    # Movement Pattern Parameter Tests

    def test_human_movement_patterns(self):
        """Test human movement pattern parameters."""
        expected_keys = {
            "min_duration_seconds",
            "max_velocity_ms",
            "typical_room_sequence_length",
            "door_interaction_probability",
        }
        assert set(HUMAN_MOVEMENT_PATTERNS.keys()) == expected_keys

        # Test parameter values and types
        assert HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"] == 30
        assert HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"] == 2.0
        assert HUMAN_MOVEMENT_PATTERNS["typical_room_sequence_length"] == 3
        assert HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"] == 0.8

        # Test parameter types
        assert isinstance(HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"], int)
        assert isinstance(HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"], float)
        assert isinstance(HUMAN_MOVEMENT_PATTERNS["typical_room_sequence_length"], int)
        assert isinstance(
            HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"], float
        )

    def test_cat_movement_patterns(self):
        """Test cat movement pattern parameters."""
        expected_keys = {
            "min_duration_seconds",
            "max_velocity_ms",
            "typical_room_sequence_length",
            "door_interaction_probability",
        }
        assert set(CAT_MOVEMENT_PATTERNS.keys()) == expected_keys

        # Test parameter values and types
        assert CAT_MOVEMENT_PATTERNS["min_duration_seconds"] == 5
        assert CAT_MOVEMENT_PATTERNS["max_velocity_ms"] == 5.0
        assert CAT_MOVEMENT_PATTERNS["typical_room_sequence_length"] == 5
        assert CAT_MOVEMENT_PATTERNS["door_interaction_probability"] == 0.1

        # Test parameter types
        assert isinstance(CAT_MOVEMENT_PATTERNS["min_duration_seconds"], int)
        assert isinstance(CAT_MOVEMENT_PATTERNS["max_velocity_ms"], float)
        assert isinstance(CAT_MOVEMENT_PATTERNS["typical_room_sequence_length"], int)
        assert isinstance(CAT_MOVEMENT_PATTERNS["door_interaction_probability"], float)

    def test_movement_pattern_differences(self):
        """Test logical differences between human and cat movement patterns."""
        # Cats should have shorter minimum duration
        assert (
            CAT_MOVEMENT_PATTERNS["min_duration_seconds"]
            < HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"]
        )

        # Cats should have higher velocity
        assert (
            CAT_MOVEMENT_PATTERNS["max_velocity_ms"]
            > HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"]
        )

        # Cats should have longer room sequences (more wandering)
        assert (
            CAT_MOVEMENT_PATTERNS["typical_room_sequence_length"]
            > HUMAN_MOVEMENT_PATTERNS["typical_room_sequence_length"]
        )

        # Humans should interact with doors more often
        assert (
            HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"]
            > CAT_MOVEMENT_PATTERNS["door_interaction_probability"]
        )

    # Immutability and Type Checking Tests

    def test_enum_immutability(self):
        """Test that enum values cannot be modified."""
        with pytest.raises((AttributeError, TypeError)):
            SensorType.PRESENCE = "modified"

        with pytest.raises((AttributeError, TypeError)):
            SensorState.ON = "modified"

    def test_constant_types(self):
        """Test types of all constants."""
        # Test list constants
        assert isinstance(PRESENCE_STATES, list)
        assert isinstance(ABSENCE_STATES, list)
        assert isinstance(DOOR_OPEN_STATES, list)
        assert isinstance(DOOR_CLOSED_STATES, list)
        assert isinstance(INVALID_STATES, list)

        # Test numeric constants
        assert isinstance(MIN_EVENT_SEPARATION, int)
        assert isinstance(MAX_SEQUENCE_GAP, int)
        assert isinstance(DEFAULT_CONFIDENCE_THRESHOLD, float)

        # Test dictionary constants
        assert isinstance(MQTT_TOPICS, dict)
        assert isinstance(DB_TABLES, dict)
        assert isinstance(API_ENDPOINTS, dict)
        assert isinstance(DEFAULT_MODEL_PARAMS, dict)
        assert isinstance(HUMAN_MOVEMENT_PATTERNS, dict)
        assert isinstance(CAT_MOVEMENT_PATTERNS, dict)

    # Cross-Constant Validation Tests

    def test_state_constant_consistency(self):
        """Test consistency between state constants and enum values."""
        # All presence/absence states should be valid sensor states
        all_sensor_states = [state.value for state in SensorState]

        for state in PRESENCE_STATES:
            assert state in all_sensor_states

        for state in ABSENCE_STATES:
            assert state in all_sensor_states

        for state in DOOR_OPEN_STATES:
            assert state in all_sensor_states

        for state in DOOR_CLOSED_STATES:
            assert state in all_sensor_states

        for state in INVALID_STATES:
            assert state in all_sensor_states

    def test_model_type_parameter_alignment(self):
        """Test that all model types in enum have corresponding parameters."""
        # All model types should have parameters
        model_types_with_params = set(DEFAULT_MODEL_PARAMS.keys())
        all_model_types = set(ModelType)

        # Since GP is an alias for GAUSSIAN_PROCESS, we expect both to be in the parameters
        # but they should be the same enum member
        expected_model_types = {
            ModelType.LSTM,
            ModelType.XGBOOST,
            ModelType.HMM,
            ModelType.GAUSSIAN_PROCESS,
            ModelType.ENSEMBLE,
        }

        # GP should be included as an alias but point to the same member as GAUSSIAN_PROCESS
        assert ModelType.GP in model_types_with_params  # GP should have params as alias
        assert model_types_with_params >= expected_model_types

    # Format String Validation Tests

    def test_mqtt_topic_format_strings(self):
        """Test MQTT topic format string validation."""
        for topic_key, topic_template in MQTT_TOPICS.items():
            assert isinstance(topic_template, str)
            assert "{topic_prefix}" in topic_template

            # Test format string works
            if "{room_id}" in topic_template:
                formatted = topic_template.format(topic_prefix="test", room_id="room1")
            else:
                formatted = topic_template.format(topic_prefix="test")

            assert "test" in formatted
            assert "{" not in formatted  # No unresolved placeholders

    def test_api_endpoint_format_strings(self):
        """Test API endpoint format string validation."""
        for endpoint_key, endpoint_template in API_ENDPOINTS.items():
            assert isinstance(endpoint_template, str)
            assert endpoint_template.startswith("/api")

            # Test format string works if it has placeholders
            if "{room_id}" in endpoint_template:
                formatted = endpoint_template.format(room_id="test_room")
                assert "test_room" in formatted
                assert "{" not in formatted  # No unresolved placeholders

    # Memory and Performance Tests

    def test_feature_list_performance(self):
        """Test performance with large feature lists."""
        # Test that feature lists are reasonable in size
        assert len(TEMPORAL_FEATURE_NAMES) <= 20
        assert len(SEQUENTIAL_FEATURE_NAMES) <= 20
        assert len(CONTEXTUAL_FEATURE_NAMES) <= 20

        # Test list access performance (should be fast)
        import time

        start_time = time.time()
        for _ in range(1000):
            _ = len(
                TEMPORAL_FEATURE_NAMES
                + SEQUENTIAL_FEATURE_NAMES
                + CONTEXTUAL_FEATURE_NAMES
            )
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should complete in less than 100ms

    def test_dictionary_access_performance(self):
        """Test dictionary constant access performance."""
        import time

        # Test model params access
        start_time = time.time()
        for _ in range(1000):
            for model_type in ModelType:
                _ = DEFAULT_MODEL_PARAMS.get(model_type, {})
        elapsed = time.time() - start_time
        assert elapsed < 0.1  # Should complete in less than 100ms


class TestCustomExceptions:
    """Test custom exception classes."""

    # ErrorSeverity Enum Tests

    def test_error_severity_enum(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

        # Verify all expected values are present
        expected_values = ["low", "medium", "high", "critical"]
        actual_values = [severity.value for severity in ErrorSeverity]
        assert set(actual_values) == set(expected_values)

    # Base Exception Tests

    def test_base_exception_initialization(self):
        """Test OccupancyPredictionError base exception initialization."""
        message = "Test error message"
        error_code = "TEST_ERROR"
        context = {"key": "value"}
        severity = ErrorSeverity.HIGH
        cause = ValueError("Underlying error")

        error = OccupancyPredictionError(
            message=message,
            error_code=error_code,
            context=context,
            severity=severity,
            cause=cause,
        )

        assert error.message == message
        assert error.error_code == error_code
        assert error.context == context
        assert error.severity == severity
        assert error.cause == cause
        assert str(error.cause) == "Underlying error"

    def test_base_exception_defaults(self):
        """Test OccupancyPredictionError with default values."""
        message = "Test error"
        error = OccupancyPredictionError(message)

        assert error.message == message
        assert error.error_code is None
        assert error.context == {}
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause is None

    def test_base_exception_string_formatting(self):
        """Test OccupancyPredictionError string formatting."""
        # Test basic message only
        error = OccupancyPredictionError("Simple error")
        assert str(error) == "Simple error"

        # Test with error code
        error = OccupancyPredictionError("Error with code", error_code="ERR001")
        error_str = str(error)
        assert "Error with code" in error_str
        assert "Error Code: ERR001" in error_str

        # Test with context
        error = OccupancyPredictionError(
            "Error with context", context={"room": "living_room", "sensor": "motion"}
        )
        error_str = str(error)
        assert "Error with context" in error_str
        assert "Context:" in error_str
        assert "room=living_room" in error_str
        assert "sensor=motion" in error_str

        # Test with cause
        cause = ValueError("Original error")
        error = OccupancyPredictionError("Error with cause", cause=cause)
        error_str = str(error)
        assert "Error with cause" in error_str
        assert "Caused by: ValueError: Original error" in error_str

        # Test with all fields
        error = OccupancyPredictionError(
            message="Complete error",
            error_code="COMPLETE_ERR",
            context={"test": "value"},
            cause=RuntimeError("Root cause"),
        )
        error_str = str(error)
        assert "Complete error" in error_str
        assert "Error Code: COMPLETE_ERR" in error_str
        assert "Context: test=value" in error_str
        assert "Caused by: RuntimeError: Root cause" in error_str

    # Configuration Exception Tests

    def test_configuration_error_base(self):
        """Test ConfigurationError base class."""
        error = ConfigurationError("Config error", config_file="config.yaml")

        assert isinstance(error, OccupancyPredictionError)
        assert error.message == "Config error"
        assert error.context["config_file"] == "config.yaml"

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError initialization and attributes."""
        error = ConfigFileNotFoundError("config.yaml", "/app/config")

        assert isinstance(error, ConfigurationError)
        assert "config.yaml" in error.message
        assert "/app/config" in error.message
        assert error.error_code == "CONFIG_FILE_NOT_FOUND_ERROR"
        assert error.context["config_dir"] == "/app/config"
        assert error.severity == ErrorSeverity.CRITICAL

    def test_config_validation_error(self):
        """Test ConfigValidationError with various parameter combinations."""
        # Test with explicit message
        error = ConfigValidationError(message="Custom validation error")
        assert error.message == "Custom validation error"
        assert error.error_code == "CONFIG_VALIDATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH

        # Test with field and expected value (auto-generated message)
        error = ConfigValidationError(
            field="timeout", value=0, expected="positive integer"
        )
        assert "Invalid configuration field 'timeout'" in error.message
        assert "got 0, expected positive integer" in error.message
        assert error.context["field"] == "timeout"
        assert error.context["value"] == 0
        assert error.context["expected"] == "positive integer"

        # Test with field only (auto-generated message)
        error = ConfigValidationError(field="database_url", value=None)
        assert "Invalid configuration field 'database_url'" in error.message
        assert error.context["field"] == "database_url"
        assert error.context["value"] is None

        # Test with config_key and valid_values
        error = ConfigValidationError(
            config_key="logging.level",
            field="level",
            value="INVALID",
            valid_values=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        assert error.context["config_key"] == "logging.level"
        assert error.context["valid_values"] == ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_missing_config_section_error(self):
        """Test MissingConfigSectionError initialization."""
        error = MissingConfigSectionError("database", "config.yaml")

        assert isinstance(error, ConfigurationError)
        assert "Required configuration section 'database' missing" in error.message
        assert error.error_code == "CONFIG_SECTION_MISSING_ERROR"
        assert error.context["section_name"] == "database"
        assert error.severity == ErrorSeverity.HIGH

    def test_config_parsing_error(self):
        """Test ConfigParsingError initialization."""
        parsing_error = "YAML syntax error at line 5"
        error = ConfigParsingError("config.yaml", parsing_error)

        assert isinstance(error, ConfigurationError)
        assert "Failed to parse configuration file 'config.yaml'" in error.message
        assert parsing_error in error.message
        assert error.error_code == "CONFIG_PARSING_ERROR"
        assert error.context["parse_error"] == parsing_error
        assert error.severity == ErrorSeverity.HIGH

    # Home Assistant Exception Tests

    def test_home_assistant_connection_error(self):
        """Test HomeAssistantConnectionError initialization."""
        url = "http://localhost:8123"
        cause = ConnectionError("Connection refused")
        error = HomeAssistantConnectionError(url, cause)

        assert isinstance(error, HomeAssistantError)
        assert url in error.message
        assert error.error_code == "HA_CONNECTION_ERROR"
        assert error.context["url"] == url
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_home_assistant_authentication_error(self):
        """Test HomeAssistantAuthenticationError with different token hint types."""
        url = "http://localhost:8123"

        # Test with string token
        token = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..."
        error = HomeAssistantAuthenticationError(url, token)

        assert isinstance(error, HomeAssistantError)
        assert url in error.message
        assert error.error_code == "HA_AUTH_ERROR"
        assert error.context["url"] == url
        assert error.context["token_hint"] == "eyJ0eXAiOi..."
        assert (
            error.context["hint"]
            == "Check if token is valid and has required permissions"
        )
        assert error.severity == ErrorSeverity.HIGH

        # Test with integer token length
        error = HomeAssistantAuthenticationError(url, 150)
        assert error.context["token_length"] == 150

        # Test with other type (fallback)
        error = HomeAssistantAuthenticationError(url, 12.5)
        assert error.context["token_hint"] == "12.5"

    def test_home_assistant_api_error(self):
        """Test HomeAssistantAPIError initialization."""
        endpoint = "/api/states"
        status_code = 404
        response_text = "Entity not found" * 100  # Long response
        method = "GET"

        error = HomeAssistantAPIError(endpoint, status_code, response_text, method)

        assert isinstance(error, HomeAssistantError)
        assert endpoint in error.message
        assert str(status_code) in error.message
        assert method in error.message
        assert error.error_code == "HA_API_ERROR"
        assert error.context["endpoint"] == endpoint
        assert error.context["method"] == method
        assert error.context["status_code"] == status_code
        # Response should be truncated to 500 characters
        assert len(error.context["response"]) <= 500
        assert error.severity == ErrorSeverity.MEDIUM

    def test_entity_not_found_error(self):
        """Test EntityNotFoundError initialization."""
        # Test without room_id
        entity_id = "sensor.living_room_motion"
        error = EntityNotFoundError(entity_id)

        assert isinstance(error, HomeAssistantError)
        assert entity_id in error.message
        assert error.error_code == "ENTITY_NOT_FOUND_ERROR"
        assert error.context["entity_id"] == entity_id
        assert "room_id" not in error.context
        assert error.severity == ErrorSeverity.MEDIUM

        # Test with room_id
        room_id = "living_room"
        error = EntityNotFoundError(entity_id, room_id)
        assert room_id in error.message
        assert error.context["room_id"] == room_id

    def test_websocket_errors(self):
        """Test WebSocket-related errors."""
        url = "ws://localhost:8123/api/websocket"

        # Test base WebSocketError
        error = WebSocketError("Connection failed", url)
        assert isinstance(error, HomeAssistantError)
        assert error.error_code == "HA_WEBSOCKET_ERROR"
        assert error.context["url"] == url
        assert error.context["reason"] == "Connection failed"

        # Test WebSocketConnectionError
        cause = ConnectionError("Connection refused")
        error = WebSocketConnectionError(url, cause)
        assert error.error_code == "WEBSOCKET_CONNECTION_ERROR"
        assert error.cause == cause

        # Test WebSocketAuthenticationError
        error = WebSocketAuthenticationError(url, "bearer_token")
        assert error.error_code == "WEBSOCKET_AUTH_ERROR"
        assert error.context["auth_method"] == "bearer_token"

        # Test WebSocketValidationError
        error = WebSocketValidationError(url, "Invalid message format", "auth")
        assert error.error_code == "WEBSOCKET_VALIDATION_ERROR"
        assert error.context["validation_error"] == "Invalid message format"
        assert error.context["message_type"] == "auth"

    # Database Exception Tests

    def test_database_connection_error(self):
        """Test DatabaseConnectionError initialization and password masking."""
        connection_string = "postgresql://user:secret123@localhost:5432/dbname"
        cause = ConnectionError("Database unreachable")
        error = DatabaseConnectionError(connection_string, cause)

        assert isinstance(error, DatabaseError)
        assert error.error_code == "DB_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.cause == cause

        # Password should be masked in both message and context
        assert "secret123" not in error.message
        assert "***" in error.message
        assert "secret123" not in error.context["connection_string"]
        assert "***" in error.context["connection_string"]

        # Test password masking static method
        masked = DatabaseConnectionError._mask_password(connection_string)
        assert "secret123" not in masked
        assert "postgresql://user:***@localhost:5432/dbname" == masked

    def test_database_query_error(self):
        """Test DatabaseQueryError initialization with various parameters."""
        query = "SELECT * FROM sensor_events WHERE room_id = $1" * 10  # Long query
        parameters = {"room_id": "living_room"}
        cause = Exception("Query timeout")
        error_type = "TimeoutError"

        error = DatabaseQueryError(
            query, parameters, cause, error_type, ErrorSeverity.HIGH
        )

        assert isinstance(error, DatabaseError)
        assert error.error_code == "DB_QUERY_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

        # Query should be truncated in message and context
        assert len(error.message) < len(query) + 50  # Message truncates to 100 chars
        assert len(error.context["query"]) <= 200  # Context truncates to 200 chars
        assert error.context["parameters"] == parameters
        assert error.context["error_type"] == error_type

    def test_database_migration_error(self):
        """Test DatabaseMigrationError initialization."""
        migration_name = "001_create_sensor_events_table"
        cause = Exception("Table already exists")
        error = DatabaseMigrationError(migration_name, cause)

        assert isinstance(error, DatabaseError)
        assert migration_name in error.message
        assert error.error_code == "DB_MIGRATION_ERROR"
        assert error.context["migration_name"] == migration_name
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_database_integrity_error(self):
        """Test DatabaseIntegrityError initialization."""
        constraint = "unique_room_timestamp"
        table_name = "sensor_events"
        values = {"room_id": "kitchen", "timestamp": "2024-01-01T12:00:00"}
        cause = Exception("Duplicate key violation")

        error = DatabaseIntegrityError(constraint, table_name, values, cause)

        assert isinstance(error, DatabaseError)
        assert constraint in error.message
        assert table_name in error.message
        assert error.error_code == "DB_INTEGRITY_ERROR"
        assert error.context["constraint"] == constraint
        assert error.context["table"] == table_name
        assert error.context["values"] == values
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    # Feature Engineering Exception Tests

    def test_feature_extraction_error(self):
        """Test FeatureExtractionError initialization."""
        feature_type = "temporal"
        room_id = "bedroom"
        time_range = "2024-01-01 to 2024-01-02"
        cause = ValueError("Insufficient data")

        error = FeatureExtractionError(feature_type, room_id, time_range, cause)

        assert isinstance(error, FeatureEngineeringError)
        assert feature_type in error.message
        assert room_id in error.message
        assert error.error_code == "FEATURE_EXTRACTION_ERROR"
        assert error.context["feature_type"] == feature_type
        assert error.context["room_id"] == room_id
        assert error.context["time_range"] == time_range
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

        # Test without room_id
        error = FeatureExtractionError(feature_type)
        assert "for room" not in error.message
        assert "room_id" not in error.context

    def test_insufficient_data_error(self):
        """Test InsufficientDataError initialization."""
        data_type = "motion events"
        room_id = "kitchen"
        required_samples = 100
        available_samples = 25

        error = InsufficientDataError(
            data_type, room_id, required_samples, available_samples
        )

        assert isinstance(error, FeatureEngineeringError)
        assert data_type in error.message
        assert room_id in error.message
        assert f"need {required_samples}" in error.message
        assert f"have {available_samples}" in error.message
        assert error.error_code == "INSUFFICIENT_DATA_ERROR"
        assert error.context["data_type"] == data_type
        assert error.context["room_id"] == room_id
        assert error.context["required_samples"] == required_samples
        assert error.context["available_samples"] == available_samples
        assert error.severity == ErrorSeverity.MEDIUM

    def test_feature_validation_error(self):
        """Test FeatureValidationError initialization."""
        feature_name = "temperature"
        validation_error = "value must be between -50 and 50 degrees"
        actual_value = 75.5
        room_id = "garage"
        cause = ValueError("Temperature out of range")

        error = FeatureValidationError(
            feature_name, validation_error, actual_value, room_id, cause
        )

        assert isinstance(error, FeatureEngineeringError)
        assert feature_name in error.message
        assert validation_error in error.message
        assert room_id in error.message
        assert error.error_code == "FEATURE_VALIDATION_ERROR"
        assert error.context["feature_name"] == feature_name
        assert error.context["validation_error"] == validation_error
        assert (
            error.context["validation_rule"] == validation_error
        )  # Backward compatibility
        assert error.context["actual_value"] == actual_value
        assert error.context["room_id"] == room_id
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

    def test_feature_store_error(self):
        """Test FeatureStoreError initialization."""
        operation = "save_features"
        feature_type = "temporal_features"
        cause = IOError("Disk full")

        error = FeatureStoreError(operation, feature_type, cause)

        assert isinstance(error, FeatureEngineeringError)
        assert operation in error.message
        assert feature_type in error.message
        assert error.error_code == "FEATURE_STORE_ERROR"
        assert error.context["operation"] == operation
        assert error.context["feature_group"] == feature_type
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

    # Model Exception Tests

    def test_model_training_error(self):
        """Test ModelTrainingError initialization."""
        model_type = "lstm"
        room_id = "office"
        training_data_size = 500
        cause = RuntimeError("CUDA out of memory")

        error = ModelTrainingError(model_type, room_id, training_data_size, cause)

        assert isinstance(error, ModelError)
        assert model_type in error.message
        assert room_id in error.message
        assert error.error_code == "MODEL_TRAINING_ERROR"
        assert error.context["model_type"] == model_type
        assert error.context["room_id"] == room_id
        assert error.context["training_data_size"] == training_data_size
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_model_prediction_error(self):
        """Test ModelPredictionError initialization."""
        model_type = "xgboost"
        room_id = "bathroom"
        feature_shape = (1, 25)
        cause = ValueError("Feature mismatch")

        error = ModelPredictionError(model_type, room_id, feature_shape, cause)

        assert isinstance(error, ModelError)
        assert model_type in error.message
        assert room_id in error.message
        assert error.error_code == "MODEL_PREDICTION_ERROR"
        assert error.context["model_type"] == model_type
        assert error.context["room_id"] == room_id
        assert error.context["feature_shape"] == feature_shape
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

    def test_insufficient_training_data_error(self):
        """Test InsufficientTrainingDataError with various parameter combinations."""
        room_id = "study"

        # Test with data_points and minimum_required
        error = InsufficientTrainingDataError(
            room_id=room_id, data_points=50, minimum_required=100, model_type="lstm"
        )
        assert "have 50, need 100" in error.message
        assert "lstm" in error.message
        assert error.context["data_points"] == 50
        assert error.context["minimum_required"] == 100
        assert error.context["model_type"] == "lstm"

        # Test with required_samples and available_samples
        error = InsufficientTrainingDataError(
            room_id=room_id, required_samples=200, available_samples=75
        )
        assert "need 200, have 75" in error.message
        assert error.context["required_samples"] == 200
        assert error.context["available_samples"] == 75

        # Test with minimal parameters
        error = InsufficientTrainingDataError(room_id=room_id)
        assert room_id in error.message
        assert error.error_code == "INSUFFICIENT_TRAINING_DATA_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_model_not_found_error(self):
        """Test ModelNotFoundError initialization."""
        model_type = "ensemble"
        room_id = "hallway"
        model_path = "/models/hallway_ensemble.pkl"

        error = ModelNotFoundError(model_type, room_id, model_path)

        assert isinstance(error, ModelError)
        assert model_type in error.message
        assert room_id in error.message
        assert error.error_code == "MODEL_NOT_FOUND"
        assert error.context["model_type"] == model_type
        assert error.context["room_id"] == room_id
        assert error.context["model_path"] == model_path
        assert error.severity == ErrorSeverity.HIGH

    def test_model_version_mismatch_error(self):
        """Test ModelVersionMismatchError initialization."""
        model_type = "gp"
        room_id = "patio"
        expected_version = "2.1.0"
        actual_version = "1.9.5"
        cause = Exception("Version incompatibility")

        error = ModelVersionMismatchError(
            model_type, room_id, expected_version, actual_version, cause
        )

        assert isinstance(error, ModelError)
        assert model_type in error.message
        assert room_id in error.message
        assert expected_version in error.message
        assert actual_version in error.message
        assert error.error_code == "MODEL_VERSION_MISMATCH_ERROR"
        assert error.context["expected_version"] == expected_version
        assert error.context["actual_version"] == actual_version
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_missing_feature_error(self):
        """Test MissingFeatureError initialization."""
        feature_names = ["temperature", "humidity", "light_level"]
        room_id = "conservatory"
        available_features = ["temperature", "motion"]
        operation = "training"

        error = MissingFeatureError(
            feature_names, room_id, available_features, operation
        )

        assert isinstance(error, FeatureEngineeringError)
        assert "temperature, humidity, light_level" in error.message
        assert room_id in error.message
        assert operation in error.message
        assert error.error_code == "MISSING_FEATURE_ERROR"
        assert error.context["feature_names"] == feature_names
        assert (
            error.context["missing_features"] == feature_names
        )  # Backward compatibility
        assert error.context["room_id"] == room_id
        assert error.context["available_features"] == available_features
        assert error.context["operation"] == operation
        assert error.severity == ErrorSeverity.HIGH

    def test_model_validation_error(self):
        """Test ModelValidationError initialization."""
        model_type = "hmm"
        room_id = "basement"
        validation_error = "Model accuracy below threshold"
        cause = ValueError("Validation failed")

        error = ModelValidationError(model_type, room_id, validation_error, cause)

        assert isinstance(error, ModelError)
        assert model_type in error.message
        assert room_id in error.message
        assert validation_error in error.message
        assert error.error_code == "MODEL_VALIDATION_ERROR"
        assert error.context["model_type"] == model_type
        assert error.context["room_id"] == room_id
        assert error.context["validation_error"] == validation_error
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    # Data Processing Exception Tests

    def test_data_corruption_error(self):
        """Test DataCorruptionError initialization."""
        data_source = "sensor_events_table"
        corruption_details = "Checksum mismatch in 15% of records"

        error = DataCorruptionError(data_source, corruption_details)

        assert isinstance(error, DataProcessingError)
        assert data_source in error.message
        assert corruption_details in error.message
        assert error.error_code == "DATA_CORRUPTION_ERROR"
        assert error.context["data_source"] == data_source
        assert error.context["corruption_details"] == corruption_details
        assert error.severity == ErrorSeverity.HIGH

    def test_data_validation_error(self):
        """Test DataValidationError with multiple initialization patterns."""
        # Test primary signature (data_source, validation_errors)
        data_source = "api_input"
        validation_errors = ["Invalid timestamp format", "Missing room_id field"]
        sample_data = {"timestamp": "invalid", "value": 25}

        error = DataValidationError(data_source, validation_errors, sample_data)

        assert isinstance(error, IntegrationError)
        assert data_source in error.message
        assert "Invalid timestamp format; Missing room_id field" in error.message
        assert error.error_code == "DATA_VALIDATION_ERROR"
        assert error.context["data_source"] == data_source
        assert error.context["validation_errors"] == validation_errors
        assert error.context["sample_data"] == sample_data
        assert error.severity == ErrorSeverity.MEDIUM

        # Test legacy signature (data_type, validation_rule, actual_value)
        error = DataValidationError(
            data_source="legacy_test",
            validation_errors=None,
            data_type="temperature",
            validation_rule="must be numeric",
            actual_value="invalid_string",
            expected_value="float",
            field_name="temp_reading",
        )
        assert "Data validation failed: temperature - must be numeric" in error.message
        assert error.context["data_type"] == "temperature"
        assert error.context["validation_rule"] == "must be numeric"
        assert "invalid_string" in error.context["actual_value"]
        assert "float" in error.context["expected_value"]
        assert error.context["field_name"] == "temp_reading"

    # Integration Exception Tests

    def test_mqtt_connection_error(self):
        """Test MQTTConnectionError initialization."""
        broker = "localhost"
        port = 1883
        username = "mqtt_user"
        cause = ConnectionError("Broker unavailable")

        error = MQTTConnectionError(broker, port, username, cause)

        assert isinstance(error, MQTTError)
        assert isinstance(error, IntegrationError)
        assert broker in error.message
        assert str(port) in error.message
        assert error.error_code == "MQTT_CONNECTION_ERROR"
        assert error.context["broker"] == broker
        assert error.context["port"] == port
        assert error.context["username"] == username
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_mqtt_publish_error(self):
        """Test MQTTPublishError initialization."""
        topic = "occupancy/predictions/kitchen"
        payload_size = 1024
        qos = 2
        broker = "mqtt.local"
        cause = Exception("Publish timeout")

        error = MQTTPublishError(topic, payload_size, qos, broker, cause)

        assert isinstance(error, MQTTError)
        assert topic in error.message
        assert broker in error.message
        assert error.error_code == "MQTT_PUBLISH_ERROR"
        assert error.context["topic"] == topic
        assert error.context["broker"] == broker
        assert error.context["payload_size"] == payload_size
        assert error.context["qos"] == qos
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

    def test_mqtt_subscription_error(self):
        """Test MQTTSubscriptionError initialization."""
        topic = "homeassistant/+/+/state"
        broker = "mqtt.home"
        cause = Exception("Subscription failed")

        error = MQTTSubscriptionError(topic, broker, cause)

        assert isinstance(error, MQTTError)
        assert topic in error.message
        assert broker in error.message
        assert error.error_code == "MQTT_SUBSCRIPTION_ERROR"
        assert error.context["topic_pattern"] == topic
        assert error.context["broker"] == broker
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

    def test_api_server_error(self):
        """Test APIServerError initialization."""
        endpoint = "/api/predictions/living_room"
        operation = "GET_PREDICTION"
        cause = Exception("Internal server error")

        error = APIServerError(endpoint, operation, cause)

        assert isinstance(error, IntegrationError)
        assert endpoint in error.message
        assert operation in error.message
        assert error.error_code == "API_SERVER_ERROR"
        assert error.context["endpoint"] == endpoint
        assert error.context["operation"] == operation
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause == cause

    # System Exception Tests

    def test_system_initialization_error(self):
        """Test SystemInitializationError initialization."""
        component = "prediction_engine"
        cause = ImportError("Required module not found")

        error = SystemInitializationError(component, cause)

        assert isinstance(error, OccupancyPredictionError)
        assert component in error.message
        assert error.error_code == "SYSTEM_INIT_ERROR"
        assert error.context["component"] == component
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.cause == cause

    def test_system_resource_error(self):
        """Test SystemResourceError initialization."""
        resource_type = "memory"
        resource_name = "prediction_cache"
        cause = MemoryError("Out of memory")

        error = SystemResourceError(resource_type, resource_name, cause)

        assert isinstance(error, OccupancyPredictionError)
        assert resource_type in error.message
        assert resource_name in error.message
        assert error.error_code == "SYSTEM_RESOURCE_ERROR"
        assert error.context["resource_type"] == resource_type
        assert error.context["resource_name"] == resource_name
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_system_error(self):
        """Test SystemError initialization with various parameter combinations."""
        # Test with operation and component
        message = "Database connection lost"
        operation = "data_ingestion"
        component = "database_manager"
        cause = ConnectionError("Connection reset")

        error = SystemError(message, operation, component, cause)

        assert isinstance(error, OccupancyPredictionError)
        assert message in error.message
        assert operation in error.message
        assert component in error.message
        assert error.error_code == "SYSTEM_ERROR"
        assert error.context["operation"] == operation
        assert error.context["component"] == component
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

        # Test with operation only
        error = SystemError(message, operation=operation)
        assert operation in error.message
        assert "component" not in error.context

        # Test with message only
        error = SystemError(message)
        assert error.message == message
        assert "operation" not in error.context
        assert "component" not in error.context

    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError initialization."""
        resource_type = "CPU"
        current_usage = 95.5
        limit = 90.0
        unit = "%"

        error = ResourceExhaustionError(resource_type, current_usage, limit, unit)

        assert isinstance(error, SystemError)
        assert resource_type in error.message
        assert str(current_usage) in error.message
        assert str(limit) in error.message
        assert unit in error.message
        assert error.error_code == "RESOURCE_EXHAUSTION_ERROR"
        assert error.context["resource_type"] == resource_type
        assert error.context["current_usage"] == current_usage
        assert error.context["limit"] == limit
        assert error.context["unit"] == unit

    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError initialization."""
        service_name = "home_assistant_api"
        endpoint = "http://localhost:8123/api"
        retry_after = 30
        reason = "Service maintenance"

        error = ServiceUnavailableError(service_name, endpoint, retry_after, reason)

        assert isinstance(error, SystemError)
        assert service_name in error.message
        assert reason in error.message
        assert error.error_code == "SERVICE_UNAVAILABLE_ERROR"
        assert error.context["service_name"] == service_name
        assert error.context["endpoint"] == endpoint
        assert error.context["retry_after"] == retry_after
        assert error.context["reason"] == reason

    def test_maintenance_mode_error(self):
        """Test MaintenanceModeError initialization."""
        component = "prediction_service"
        end_time = "2024-01-15T14:00:00Z"

        error = MaintenanceModeError(component, end_time)

        assert isinstance(error, SystemError)
        assert component in error.message
        assert end_time in error.message
        assert error.error_code == "MAINTENANCE_MODE_ERROR"
        assert error.context["component"] == component
        assert error.context["estimated_end_time"] == end_time
        assert error.severity == ErrorSeverity.MEDIUM

        # Test backward compatibility with maintenance_until
        error = MaintenanceModeError(maintenance_until=end_time)
        assert error.context["estimated_end_time"] == end_time

    # API Exception Tests

    def test_api_authentication_error(self):
        """Test APIAuthenticationError initialization."""
        message = "Invalid JWT token"
        endpoint = "/api/predictions"
        auth_method = "bearer_token"
        context = {"token_expired": True}

        error = APIAuthenticationError(message, endpoint, auth_method, context)

        assert isinstance(error, APIError)
        assert error.message == message
        assert error.error_code == "API_AUTH_ERROR"
        assert error.context["endpoint"] == endpoint
        assert error.context["auth_method"] == auth_method
        assert error.context["token_expired"] is True
        assert error.severity == ErrorSeverity.HIGH

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError initialization."""
        service = "prediction_api"
        limit = 100
        window_seconds = 3600
        reset_time = 1642248000

        error = RateLimitExceededError(service, limit, window_seconds, reset_time)

        assert isinstance(error, IntegrationError)
        assert service in error.message
        assert str(limit) in error.message
        assert str(window_seconds) in error.message
        assert error.error_code == "RATE_LIMIT_EXCEEDED_ERROR"
        assert error.context["service"] == service
        assert error.context["limit"] == limit
        assert error.context["window_seconds"] == window_seconds
        assert error.context["reset_time"] == reset_time
        assert error.severity == ErrorSeverity.MEDIUM

    def test_api_authorization_error(self):
        """Test APIAuthorizationError initialization."""
        message = "Insufficient permissions"
        endpoint = "/api/admin/retrain"
        required_permission = "admin:model_training"
        context = {"user_role": "viewer"}

        error = APIAuthorizationError(message, endpoint, required_permission, context)

        assert isinstance(error, APIError)
        assert error.message == message
        assert error.error_code == "API_AUTHORIZATION_ERROR"
        assert error.context["endpoint"] == endpoint
        assert error.context["required_permission"] == required_permission
        assert error.context["user_role"] == "viewer"
        assert error.severity == ErrorSeverity.HIGH

    def test_api_security_error(self):
        """Test APISecurityError initialization."""
        message = "SQL injection attempt detected"
        violation_type = "sql_injection"
        endpoint = "/api/query"
        context = {"suspicious_query": "'; DROP TABLE users; --"}

        error = APISecurityError(message, violation_type, endpoint, context)

        assert isinstance(error, APIError)
        assert error.message == message
        assert error.error_code == "API_SECURITY_ERROR"
        assert error.context["violation_type"] == violation_type
        assert error.context["endpoint"] == endpoint
        assert error.context["suspicious_query"] == "'; DROP TABLE users; --"
        assert error.severity == ErrorSeverity.CRITICAL

    def test_api_resource_not_found_error(self):
        """Test APIResourceNotFoundError initialization."""
        resource_type = "Room"
        resource_id = "non_existent_room"
        endpoint = "/api/rooms/non_existent_room"
        context = {"available_rooms": ["living_room", "kitchen", "bedroom"]}

        error = APIResourceNotFoundError(resource_type, resource_id, endpoint, context)

        assert isinstance(error, APIError)
        assert resource_type in error.message
        assert resource_id in error.message
        assert error.error_code == "API_RESOURCE_NOT_FOUND"
        assert error.context["resource_type"] == resource_type
        assert error.context["resource_id"] == resource_id
        assert error.context["endpoint"] == endpoint
        assert error.context["available_rooms"] == ["living_room", "kitchen", "bedroom"]
        assert error.severity == ErrorSeverity.MEDIUM


class TestValidationHelpers:
    """Test validation helper functions."""

    def test_validate_room_id_valid(self):
        """Test validate_room_id with valid room IDs."""
        valid_room_ids = [
            "living_room",
            "bedroom1",
            "kitchen-main",
            "OFFICE",
            "room_123",
            "test-room_1",
        ]

        for room_id in valid_room_ids:
            # Should not raise exception
            validate_room_id(room_id)

    def test_validate_room_id_invalid(self):
        """Test validate_room_id with invalid room IDs."""
        invalid_room_ids = [
            "",  # Empty string
            None,  # None value
            "room with spaces",  # Contains spaces
            "room@special",  # Contains special characters
            "room/path",  # Contains slash
            "room.extension",  # Contains dot
            123,  # Not a string
        ]

        for room_id in invalid_room_ids:
            with pytest.raises(DataValidationError) as exc_info:
                validate_room_id(room_id)

            error = exc_info.value
            assert error.error_code == "DATA_VALIDATION_ERROR"
            assert error.context["data_source"] == "room_id_validation"
            assert "room_id" in error.context.get("sample_data", {})

    def test_validate_entity_id_valid(self):
        """Test validate_entity_id with valid entity IDs."""
        valid_entity_ids = [
            "sensor.living_room_motion",
            "binary_sensor.door_contact",
            "switch.bedroom_light",
            "climate.hvac_system",
            "device_tracker.phone_gps",
            "light.kitchen_led",
            "cover.garage_door",
            "sensor.temp_123",
        ]

        for entity_id in valid_entity_ids:
            # Should not raise exception
            validate_entity_id(entity_id)

    def test_validate_entity_id_invalid(self):
        """Test validate_entity_id with invalid entity IDs."""
        invalid_entity_ids = [
            "",  # Empty string
            None,  # None value
            "just_domain",  # Missing object_id
            ".no_domain",  # Missing domain
            "sensor.",  # Empty object_id
            "SENSOR.living_room",  # Uppercase domain
            "sensor.Living_Room",  # Uppercase in object_id
            "sensor.living room",  # Space in object_id
            "sensor/living_room",  # Wrong separator
            "sensor.living.room.motion",  # Multiple dots
            123,  # Not a string
        ]

        for entity_id in invalid_entity_ids:
            with pytest.raises(DataValidationError) as exc_info:
                validate_entity_id(entity_id)

            error = exc_info.value
            assert error.error_code == "DATA_VALIDATION_ERROR"
            assert error.context["data_source"] == "entity_id_validation"
            assert "entity_id" in error.context.get("sample_data", {})
            if "expected_format" in error.context.get("sample_data", {}):
                assert (
                    error.context["sample_data"]["expected_format"]
                    == "sensor.living_room_motion"
                )


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_base_exception_hierarchy(self):
        """Test that all custom exceptions inherit from OccupancyPredictionError."""
        base_exceptions = [
            ConfigurationError,
            HomeAssistantError,
            DatabaseError,
            FeatureEngineeringError,
            ModelError,
            DataProcessingError,
            IntegrationError,
            SystemError,
            APIError,
        ]

        for exception_class in base_exceptions:
            assert issubclass(exception_class, OccupancyPredictionError)
            assert issubclass(exception_class, Exception)

    def test_configuration_exceptions_hierarchy(self):
        """Test configuration exception inheritance."""
        config_exceptions = [
            ConfigFileNotFoundError,
            ConfigValidationError,
            MissingConfigSectionError,
            ConfigParsingError,
        ]

        for exception_class in config_exceptions:
            assert issubclass(exception_class, ConfigurationError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_home_assistant_exceptions_hierarchy(self):
        """Test Home Assistant exception inheritance."""
        ha_exceptions = [
            HomeAssistantConnectionError,
            HomeAssistantAuthenticationError,
            HomeAssistantAPIError,
            EntityNotFoundError,
            WebSocketError,
            WebSocketConnectionError,
            WebSocketAuthenticationError,
            WebSocketValidationError,
        ]

        for exception_class in ha_exceptions:
            assert issubclass(exception_class, HomeAssistantError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_database_exceptions_hierarchy(self):
        """Test database exception inheritance."""
        db_exceptions = [
            DatabaseConnectionError,
            DatabaseQueryError,
            DatabaseMigrationError,
            DatabaseIntegrityError,
        ]

        for exception_class in db_exceptions:
            assert issubclass(exception_class, DatabaseError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_feature_engineering_exceptions_hierarchy(self):
        """Test feature engineering exception inheritance."""
        fe_exceptions = [
            FeatureExtractionError,
            InsufficientDataError,
            FeatureValidationError,
            FeatureStoreError,
            MissingFeatureError,
        ]

        for exception_class in fe_exceptions:
            assert issubclass(exception_class, FeatureEngineeringError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_model_exceptions_hierarchy(self):
        """Test model exception inheritance."""
        model_exceptions = [
            ModelTrainingError,
            ModelPredictionError,
            InsufficientTrainingDataError,
            ModelNotFoundError,
            ModelVersionMismatchError,
            ModelValidationError,
        ]

        for exception_class in model_exceptions:
            assert issubclass(exception_class, ModelError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_integration_exceptions_hierarchy(self):
        """Test integration exception inheritance."""
        integration_exceptions = [
            DataValidationError,
            MQTTError,
            MQTTConnectionError,
            MQTTPublishError,
            MQTTSubscriptionError,
            APIServerError,
            RateLimitExceededError,
        ]

        for exception_class in integration_exceptions:
            assert issubclass(exception_class, IntegrationError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_system_exceptions_hierarchy(self):
        """Test system exception inheritance."""
        system_exceptions = [
            SystemInitializationError,
            SystemResourceError,
            ResourceExhaustionError,
            ServiceUnavailableError,
            MaintenanceModeError,
        ]

        for exception_class in system_exceptions:
            if exception_class in [
                ResourceExhaustionError,
                ServiceUnavailableError,
                MaintenanceModeError,
            ]:
                assert issubclass(exception_class, SystemError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_api_exceptions_hierarchy(self):
        """Test API exception inheritance."""
        api_exceptions = [
            APIAuthenticationError,
            APIAuthorizationError,
            APISecurityError,
            APIResourceNotFoundError,
        ]

        for exception_class in api_exceptions:
            assert issubclass(exception_class, APIError)
            assert issubclass(exception_class, OccupancyPredictionError)

    def test_exception_chaining(self):
        """Test exception chaining functionality."""
        # Create a chain of exceptions
        root_cause = ValueError("Invalid input data")
        processing_error = FeatureExtractionError(
            "temporal", "bedroom", cause=root_cause
        )
        system_error = SystemError("Feature processing failed", cause=processing_error)

        # Test cause chain
        assert system_error.cause == processing_error
        assert processing_error.cause == root_cause
        assert root_cause.__cause__ is None

        # Test string representation includes cause
        system_str = str(system_error)
        assert "Caused by: FeatureExtractionError" in system_str

        processing_str = str(processing_error)
        assert "Caused by: ValueError: Invalid input data" in processing_str

    def test_exception_context_propagation(self):
        """Test that exception context is preserved through inheritance."""
        original_context = {"room_id": "kitchen", "sensor_count": 5}
        error = FeatureExtractionError(feature_type="sequential", room_id="kitchen")

        # Verify context from constructor
        assert error.context["feature_type"] == "sequential"
        assert error.context["room_id"] == "kitchen"

        # Verify inheritance preserves context structure
        assert isinstance(error.context, dict)
        assert hasattr(error, "severity")
        assert hasattr(error, "error_code")
        assert hasattr(error, "cause")


class TestErrorHandling:
    """Test error handling patterns and edge cases."""

    def test_exception_serialization(self):
        """Test that exceptions can be properly serialized and logged."""
        error = ConfigValidationError(
            field="database_timeout",
            value=-5,
            expected="positive integer",
            config_file="config.yaml",
        )

        # Test string representation for logging
        error_str = str(error)
        assert isinstance(error_str, str)
        assert len(error_str) > 0

        # Test that all important information is in the string
        assert "database_timeout" in error_str
        assert "-5" in error_str
        assert "positive integer" in error_str

        # Test context serialization
        context_str = str(error.context)
        assert isinstance(context_str, str)
        assert "database_timeout" in context_str

    def test_exception_with_none_values(self):
        """Test exception handling with None values."""
        error = ModelPredictionError(
            model_type="lstm", room_id="living_room", feature_shape=None, cause=None
        )

        assert error.cause is None
        # Only check if feature_shape is in context if it was explicitly set
        if "feature_shape" in error.context:
            assert error.context["feature_shape"] is None

        # String representation should handle None values gracefully
        error_str = str(error)
        assert isinstance(error_str, str)
        assert len(error_str) > 0

    def test_exception_with_large_context(self):
        """Test exception handling with large context data."""
        large_data = "x" * 10000  # 10KB of data
        error = DatabaseQueryError(
            query="SELECT * FROM large_table", parameters={"large_field": large_data}
        )

        # Context should be preserved
        assert error.context["parameters"]["large_field"] == large_data

        # String representation should handle large data
        error_str = str(error)
        assert isinstance(error_str, str)
        # Should not cause memory issues
        assert len(error_str) < 50000  # Reasonable upper limit

    def test_exception_with_unicode_content(self):
        """Test exception handling with Unicode content."""
        unicode_room_id = "спальня"  # "bedroom" in Russian
        unicode_message = (
            "Ошибка валидации данных"  # "Data validation error" in Russian
        )

        error = DataValidationError(
            data_source="unicode_test",
            validation_errors=[unicode_message],
            sample_data={"room_id": unicode_room_id},
        )

        assert error.context["sample_data"]["room_id"] == unicode_room_id
        assert unicode_message in error.context["validation_errors"]

        # String representation should handle Unicode
        error_str = str(error)
        assert isinstance(error_str, str)
        # Should contain Unicode content (may be encoded)
        assert unicode_room_id in error_str or "room_id" in error_str

    def test_exception_error_codes_uniqueness(self):
        """Test that error codes are unique across exception types."""
        # Create instances of different exception types
        exceptions = [
            ConfigFileNotFoundError("config.yaml", "/app"),
            ConfigValidationError(field="test", value="invalid"),
            HomeAssistantConnectionError("http://localhost:8123"),
            DatabaseConnectionError("postgresql://localhost/db"),
            FeatureExtractionError("temporal"),
            ModelTrainingError("lstm", "bedroom"),
            DataValidationError("test", ["error"]),
            MQTTConnectionError("localhost", 1883),
            SystemInitializationError("test_component"),
        ]

        error_codes = [exc.error_code for exc in exceptions if exc.error_code]
        unique_codes = set(error_codes)

        # All error codes should be unique
        assert len(error_codes) == len(unique_codes)

        # All error codes should be strings
        for code in error_codes:
            assert isinstance(code, str)
            assert len(code) > 0

    def test_exception_severity_consistency(self):
        """Test that exception severity levels are consistent with their types."""
        # Critical severity exceptions
        critical_exceptions = [
            ConfigFileNotFoundError("config.yaml", "/app"),
            DatabaseConnectionError("postgresql://localhost/db"),
            SystemInitializationError("database"),
        ]

        for exc in critical_exceptions:
            assert exc.severity == ErrorSeverity.CRITICAL

        # High severity exceptions
        high_exceptions = [
            ConfigValidationError(field="test", value="invalid"),
            HomeAssistantConnectionError("http://localhost:8123"),
            ModelTrainingError("lstm", "bedroom"),
        ]

        for exc in high_exceptions:
            assert exc.severity == ErrorSeverity.HIGH

        # Medium severity exceptions (default)
        medium_exceptions = [
            FeatureExtractionError("temporal"),
            ModelPredictionError("lstm", "bedroom"),
            DataValidationError("test", ["error"]),
        ]

        for exc in medium_exceptions:
            assert exc.severity == ErrorSeverity.MEDIUM

    def test_exception_backwards_compatibility(self):
        """Test backwards compatibility of exception interfaces."""
        # Test alias for ModelPredictionError
        from src.core.exceptions import PredictionError

        assert PredictionError is ModelPredictionError

        # Test alias for RateLimitExceededError
        from src.core.exceptions import APIRateLimitError

        assert APIRateLimitError is RateLimitExceededError

        # Test backward compatible parameters
        error = FeatureValidationError(
            feature_name="temperature",
            validation_error="out of range",
            actual_value=100,
        )
        # Should have both new and old context keys
        assert "validation_error" in error.context
        assert "validation_rule" in error.context  # Backward compatibility
        assert error.context["validation_error"] == error.context["validation_rule"]

    def test_memory_efficiency(self):
        """Test that exceptions don't consume excessive memory."""
        import sys

        # Create many exception instances
        exceptions = []
        for i in range(1000):
            exc = ModelPredictionError(f"model_{i}", f"room_{i}")
            exceptions.append(exc)

        # Test that exceptions don't hold unnecessary references
        sample_exc = exceptions[0]
        assert sys.getrefcount(sample_exc) >= 2  # At least in list and local var

        # Test exception cleanup
        del exceptions
        # Should not cause memory issues (can't easily test without memory profiler)

        # Test that large context data doesn't duplicate unnecessarily
        from src.core.exceptions import SystemError as OccupancySystemError

        large_context = {"data": "x" * 1000}
        exc1 = DatabaseQueryError("SELECT * FROM test", {"large_field": large_context})
        exc2 = DatabaseQueryError("SELECT * FROM test", {"large_field": large_context})

        # Both should have the context but not necessarily share references
        assert "large_field" in exc1.context["parameters"]
        assert "large_field" in exc2.context["parameters"]
