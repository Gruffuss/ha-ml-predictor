"""Unit tests for core configuration system functionality.

Covers:
- src/core/config.py (Configuration Management)
- src/core/constants.py (System Constants)
- src/core/exceptions.py (Custom Exception Classes)

This test file consolidates testing for all configuration-related components
as they are closely related and often tested together in real scenarios.
"""

import os
import re
import tempfile
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, mock_open, call
from datetime import datetime
from typing import Dict, Any

# Import actual source code classes
from src.core.config import (
    HomeAssistantConfig,
    DatabaseConfig,
    MQTTConfig,
    PredictionConfig,
    FeaturesConfig,
    LoggingConfig,
    TrackingConfig,
    JWTConfig,
    APIConfig,
    SensorConfig,
    RoomConfig,
    SystemConfig,
    ConfigLoader,
    get_config,
    reload_config,
)
from src.core.constants import (
    SensorType,
    SensorState,
    EventType,
    ModelType,
    PredictionType,
    PRESENCE_STATES,
    ABSENCE_STATES,
    DOOR_OPEN_STATES,
    DOOR_CLOSED_STATES,
    INVALID_STATES,
    MIN_EVENT_SEPARATION,
    MAX_SEQUENCE_GAP,
    DEFAULT_CONFIDENCE_THRESHOLD,
    TEMPORAL_FEATURE_NAMES,
    SEQUENTIAL_FEATURE_NAMES,
    CONTEXTUAL_FEATURE_NAMES,
    MQTT_TOPICS,
    DB_TABLES,
    API_ENDPOINTS,
    DEFAULT_MODEL_PARAMS,
    HUMAN_MOVEMENT_PATTERNS,
    CAT_MOVEMENT_PATTERNS,
)
from src.core.exceptions import (
    ErrorSeverity,
    OccupancyPredictionError,
    ConfigurationError,
    ConfigFileNotFoundError,
    ConfigValidationError,
    MissingConfigSectionError,
    ConfigParsingError,
    HomeAssistantError,
    HomeAssistantConnectionError,
    HomeAssistantAuthenticationError,
    DataValidationError,
    validate_room_id,
    validate_entity_id,
)


# Test fixtures
@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary for testing."""
    return {
        "home_assistant": {
            "url": "http://homeassistant.local:8123",
            "token": "test_token_123456789",
            "websocket_timeout": 30,
            "api_timeout": 10,
        },
        "database": {
            "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/ha_ml_predictor",
            "pool_size": 10,
            "max_overflow": 20,
        },
        "mqtt": {
            "broker": "localhost",
            "port": 1883,
            "username": "mqtt_user",
            "password": "mqtt_pass",
            "topic_prefix": "occupancy/predictions",
        },
        "prediction": {
            "interval_seconds": 300,
            "accuracy_threshold_minutes": 15,
            "confidence_threshold": 0.7,
        },
        "features": {
            "lookback_hours": 24,
            "sequence_length": 50,
            "temporal_features": True,
            "sequential_features": True,
            "contextual_features": True,
        },
        "logging": {
            "level": "INFO",
            "format": "structured",
            "file_logging": False,
        },
        "tracking": {
            "enabled": True,
            "monitoring_interval_seconds": 60,
        },
        "api": {
            "enabled": True,
            "host": "0.0.0.0",
            "port": 8000,
            "debug": False,
            "jwt": {
                "enabled": True,
                "secret_key": "test_jwt_secret_key_for_security_validation_testing_at_least_32_characters_long",
                "algorithm": "HS256",
            },
        },
    }


@pytest.fixture
def sample_rooms_dict():
    """Sample rooms configuration dictionary for testing."""
    return {
        "rooms": {
            "living_room": {
                "name": "Living Room",
                "sensors": {
                    "motion": "binary_sensor.living_room_motion",
                    "door": "binary_sensor.living_room_door",
                    "temperature": "sensor.living_room_temperature",
                },
            },
            "bedroom": {
                "name": "Bedroom",
                "sensors": {
                    "motion": "binary_sensor.bedroom_motion",
                    "door": {
                        "main": "binary_sensor.bedroom_door",
                        "closet": "binary_sensor.bedroom_closet_door",
                    },
                },
            },
            "hallway": {
                "upper": {
                    "name": "Upper Hallway",
                    "sensors": {
                        "motion": "binary_sensor.upper_hallway_motion",
                    },
                },
                "lower": {
                    "name": "Lower Hallway",
                    "sensors": {
                        "motion": "binary_sensor.lower_hallway_motion",
                    },
                },
            },
        }
    }


class TestSystemConstants:
    """Test system constants and enumerations."""

    def test_sensor_type_enum_values(self):
        """Test SensorType enum has all expected values."""
        assert SensorType.PRESENCE.value == "presence"
        assert SensorType.DOOR.value == "door"
        assert SensorType.CLIMATE.value == "climate"
        assert SensorType.LIGHT.value == "light"
        assert SensorType.MOTION.value == "motion"
        
        # Test all enum members exist
        expected_types = {"presence", "door", "climate", "light", "motion"}
        actual_types = {member.value for member in SensorType}
        assert actual_types == expected_types

    def test_sensor_state_enum_values(self):
        """Test SensorState enum has all expected values."""
        assert SensorState.ON.value == "on"
        assert SensorState.OFF.value == "off"
        assert SensorState.OPEN.value == "open"
        assert SensorState.CLOSED.value == "closed"
        assert SensorState.UNKNOWN.value == "unknown"
        assert SensorState.UNAVAILABLE.value == "unavailable"

    def test_event_type_enum_values(self):
        """Test EventType enum has all expected values."""
        assert EventType.STATE_CHANGE.value == "state_change"
        assert EventType.PREDICTION.value == "prediction"
        assert EventType.MODEL_UPDATE.value == "model_update"
        assert EventType.ACCURACY_UPDATE.value == "accuracy_update"

    def test_model_type_enum_values(self):
        """Test ModelType enum has all expected values including aliases."""
        assert ModelType.LSTM.value == "lstm"
        assert ModelType.XGBOOST.value == "xgboost"
        assert ModelType.HMM.value == "hmm"
        assert ModelType.GAUSSIAN_PROCESS.value == "gp"
        assert ModelType.GP.value == "gp"  # Alias
        assert ModelType.ENSEMBLE.value == "ensemble"

    def test_prediction_type_enum_values(self):
        """Test PredictionType enum has all expected values."""
        assert PredictionType.NEXT_OCCUPIED.value == "next_occupied"
        assert PredictionType.NEXT_VACANT.value == "next_vacant"
        assert PredictionType.OCCUPANCY_DURATION.value == "occupancy_duration"
        assert PredictionType.VACANCY_DURATION.value == "vacancy_duration"

    def test_presence_states_constant(self):
        """Test PRESENCE_STATES constant contains correct values."""
        assert PRESENCE_STATES == ["on"]
        assert SensorState.ON.value in PRESENCE_STATES

    def test_absence_states_constant(self):
        """Test ABSENCE_STATES constant contains correct values."""
        assert ABSENCE_STATES == ["off"]
        assert SensorState.OFF.value in ABSENCE_STATES

    def test_door_states_constants(self):
        """Test door state constants contain correct values."""
        assert DOOR_OPEN_STATES == ["open", "on"]
        assert DOOR_CLOSED_STATES == ["closed", "off"]
        
        assert SensorState.OPEN.value in DOOR_OPEN_STATES
        assert SensorState.ON.value in DOOR_OPEN_STATES
        assert SensorState.CLOSED.value in DOOR_CLOSED_STATES
        assert SensorState.OFF.value in DOOR_CLOSED_STATES

    def test_invalid_states_constant(self):
        """Test INVALID_STATES constant contains correct values."""
        assert INVALID_STATES == ["unknown", "unavailable"]
        assert SensorState.UNKNOWN.value in INVALID_STATES
        assert SensorState.UNAVAILABLE.value in INVALID_STATES

    def test_numeric_constants(self):
        """Test numeric constants have expected values."""
        assert MIN_EVENT_SEPARATION == 5
        assert MAX_SEQUENCE_GAP == 300
        assert DEFAULT_CONFIDENCE_THRESHOLD == 0.7

    def test_feature_name_constants(self):
        """Test feature name constants are properly defined."""
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
        assert len(TEMPORAL_FEATURE_NAMES) == 10
        
        # Test sequential features
        assert "room_transition_1gram" in SEQUENTIAL_FEATURE_NAMES
        assert "movement_velocity" in SEQUENTIAL_FEATURE_NAMES
        assert len(SEQUENTIAL_FEATURE_NAMES) == 6
        
        # Test contextual features
        assert "temperature" in CONTEXTUAL_FEATURE_NAMES
        assert "humidity" in CONTEXTUAL_FEATURE_NAMES
        assert "door_state" in CONTEXTUAL_FEATURE_NAMES
        assert len(CONTEXTUAL_FEATURE_NAMES) == 6

    def test_mqtt_topics_structure(self):
        """Test MQTT topics structure is properly defined."""
        assert "predictions" in MQTT_TOPICS
        assert "confidence" in MQTT_TOPICS
        assert "status" in MQTT_TOPICS
        
        # Test format strings
        assert "{topic_prefix}" in MQTT_TOPICS["predictions"]
        assert "{room_id}" in MQTT_TOPICS["predictions"]

    def test_database_tables_constant(self):
        """Test DB_TABLES constant contains expected table names."""
        expected_tables = {
            "sensor_events",
            "predictions",
            "model_accuracy",
            "room_states",
            "feature_store",
        }
        assert set(DB_TABLES.keys()) == expected_tables
        
        # Test actual table names
        assert DB_TABLES["sensor_events"] == "sensor_events"
        assert DB_TABLES["predictions"] == "predictions"

    def test_api_endpoints_constant(self):
        """Test API_ENDPOINTS constant contains expected endpoints."""
        expected_endpoints = {
            "predictions",
            "accuracy",
            "health",
            "retrain",
            "rooms",
            "sensors",
        }
        assert set(API_ENDPOINTS.keys()) == expected_endpoints
        
        # Test endpoint format
        assert API_ENDPOINTS["predictions"] == "/api/predictions/{room_id}"
        assert API_ENDPOINTS["health"] == "/api/health"

    def test_default_model_params_structure(self):
        """Test DEFAULT_MODEL_PARAMS structure is properly defined."""
        # Test all model types have parameters
        for model_type in [ModelType.LSTM, ModelType.XGBOOST, ModelType.HMM, ModelType.GAUSSIAN_PROCESS, ModelType.ENSEMBLE]:
            assert model_type in DEFAULT_MODEL_PARAMS
            assert isinstance(DEFAULT_MODEL_PARAMS[model_type], dict)
        
        # Test specific parameters
        lstm_params = DEFAULT_MODEL_PARAMS[ModelType.LSTM]
        assert "sequence_length" in lstm_params
        assert "hidden_units" in lstm_params
        assert "dropout" in lstm_params
        assert lstm_params["sequence_length"] == 50
        
        # Test aliases
        assert "lstm_units" in lstm_params  # Alias for hidden_units
        assert "dropout_rate" in lstm_params  # Alias for dropout
        
        xgboost_params = DEFAULT_MODEL_PARAMS[ModelType.XGBOOST]
        assert "n_estimators" in xgboost_params
        assert "objective" in xgboost_params
        assert xgboost_params["objective"] == "reg:squarederror"

    def test_movement_patterns_constants(self):
        """Test human and cat movement pattern constants."""
        # Test human patterns
        assert "min_duration_seconds" in HUMAN_MOVEMENT_PATTERNS
        assert "max_velocity_ms" in HUMAN_MOVEMENT_PATTERNS
        assert "door_interaction_probability" in HUMAN_MOVEMENT_PATTERNS
        assert HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"] == 30
        assert HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"] == 2.0
        
        # Test cat patterns
        assert "min_duration_seconds" in CAT_MOVEMENT_PATTERNS
        assert "max_velocity_ms" in CAT_MOVEMENT_PATTERNS
        assert CAT_MOVEMENT_PATTERNS["min_duration_seconds"] == 5
        assert CAT_MOVEMENT_PATTERNS["max_velocity_ms"] == 5.0
        
        # Test differences between human and cat patterns
        assert HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"] > CAT_MOVEMENT_PATTERNS["min_duration_seconds"]
        assert HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"] < CAT_MOVEMENT_PATTERNS["max_velocity_ms"]


class TestExceptionClasses:
    """Test custom exception classes and their functionality."""

    def test_error_severity_enum(self):
        """Test ErrorSeverity enum values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_base_exception_initialization(self):
        """Test OccupancyPredictionError base exception initialization."""
        error = OccupancyPredictionError(
            message="Test error message",
            error_code="TEST_ERROR",
            context={"key": "value"},
            severity=ErrorSeverity.HIGH,
        )
        
        assert str(error).startswith("Test error message")
        assert error.error_code == "TEST_ERROR"
        assert error.context == {"key": "value"}
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause is None

    def test_base_exception_string_formatting(self):
        """Test OccupancyPredictionError string formatting with context and cause."""
        cause_exception = ValueError("Underlying error")
        error = OccupancyPredictionError(
            message="Main error",
            error_code="TEST_001",
            context={"room_id": "bedroom", "sensor_count": 3},
            cause=cause_exception,
        )
        
        error_string = str(error)
        assert "Main error" in error_string
        assert "Error Code: TEST_001" in error_string
        assert "Context: room_id=bedroom, sensor_count=3" in error_string
        assert "Caused by: ValueError: Underlying error" in error_string

    def test_configuration_error_inheritance(self):
        """Test ConfigurationError inherits from base exception properly."""
        error = ConfigurationError(
            message="Config error",
            config_file="config.yaml",
            error_code="CONFIG_ERROR",
        )
        
        assert isinstance(error, OccupancyPredictionError)
        assert "config_file" in error.context
        assert error.context["config_file"] == "config.yaml"

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError specific functionality."""
        error = ConfigFileNotFoundError(
            config_file="missing.yaml",
            config_dir="/config",
        )
        
        assert "not found" in str(error)
        assert error.error_code == "CONFIG_FILE_NOT_FOUND_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["config_dir"] == "/config"

    def test_config_validation_error_auto_message(self):
        """Test ConfigValidationError auto-generates message from field info."""
        error = ConfigValidationError(
            field="database.pool_size",
            value=-5,
            expected="positive integer",
        )
        
        error_msg = str(error)
        assert "Invalid configuration field 'database.pool_size'" in error_msg
        assert "got -5" in error_msg
        assert "expected positive integer" in error_msg
        assert error.context["field"] == "database.pool_size"
        assert error.context["value"] == -5

    def test_config_validation_error_with_valid_values(self):
        """Test ConfigValidationError includes valid_values in context."""
        error = ConfigValidationError(
            message="Invalid log level",
            field="logging.level",
            value="TRACE",
            valid_values=["DEBUG", "INFO", "WARNING", "ERROR"],
        )
        
        assert "valid_values" in error.context
        assert "DEBUG" in error.context["valid_values"]
        assert error.context["value"] == "TRACE"

    def test_home_assistant_authentication_error(self):
        """Test HomeAssistantAuthenticationError with different token hint types."""
        # Test with string token
        error1 = HomeAssistantAuthenticationError(
            url="http://ha.local:8123",
            token_hint="very_long_token_string_that_should_be_truncated",
        )
        assert "token_hint" in error1.context
        assert "very_long_t..." in error1.context["token_hint"]
        
        # Test with integer token length
        error2 = HomeAssistantAuthenticationError(
            url="http://ha.local:8123",
            token_hint=180,
        )
        assert "token_length" in error2.context
        assert error2.context["token_length"] == 180
        
        # Test hint is included
        assert "hint" in error1.context
        assert "Check if token is valid" in error1.context["hint"]

    def test_database_connection_error_password_masking(self):
        """Test DatabaseConnectionError masks passwords in connection strings."""
        error = DatabaseConnectionError(
            connection_string="postgresql://user:secret_password@localhost:5432/db",
        )
        
        # Check that password is masked in both message and context
        error_msg = str(error)
        assert "secret_password" not in error_msg
        assert "***" in error_msg
        assert "postgresql://user:***@localhost:5432/db" in error_msg
        
        # Test the static method directly
        masked = DatabaseConnectionError._mask_password(
            "postgresql+asyncpg://admin:my_secret@db.example.com:5432/mydb"
        )
        assert "my_secret" not in masked
        assert "postgresql+asyncpg://admin:***@db.example.com:5432/mydb" == masked

    def test_database_query_error_with_parameters(self):
        """Test DatabaseQueryError includes query and parameters in context."""
        long_query = "SELECT * FROM sensor_events WHERE room_id = %(room_id)s " * 10
        error = DatabaseQueryError(
            query=long_query,
            parameters={"room_id": "bedroom", "limit": 100},
            error_type="TimeoutError",
            severity=ErrorSeverity.HIGH,
        )
        
        assert error.error_code == "DB_QUERY_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert "error_type" in error.context
        assert error.context["error_type"] == "TimeoutError"
        assert "parameters" in error.context
        assert error.context["parameters"]["room_id"] == "bedroom"
        
        # Check query is truncated in context
        assert len(error.context["query"]) <= 200

    def test_feature_validation_error_with_room_context(self):
        """Test FeatureValidationError includes room context in message."""
        error = FeatureValidationError(
            feature_name="temperature_variance",
            validation_error="value must be positive",
            actual_value=-2.5,
            room_id="living_room",
        )
        
        error_msg = str(error)
        assert "Feature validation failed for 'temperature_variance'" in error_msg
        assert "(room: living_room)" in error_msg
        assert error.context["room_id"] == "living_room"
        assert error.context["actual_value"] == -2.5
        
        # Test backward compatibility field
        assert "validation_rule" in error.context
        assert error.context["validation_rule"] == "value must be positive"

    def test_insufficient_training_data_error_multiple_signatures(self):
        """Test InsufficientTrainingDataError handles multiple parameter combinations."""
        # Test with data_points and minimum_required
        error1 = InsufficientTrainingDataError(
            room_id="bedroom",
            data_points=50,
            minimum_required=100,
            model_type="lstm",
        )
        assert "have 50, need 100" in str(error1)
        assert "lstm" in str(error1)
        
        # Test with required_samples and available_samples
        error2 = InsufficientTrainingDataError(
            room_id="kitchen",
            required_samples=500,
            available_samples=200,
        )
        assert "need 500, have 200" in str(error2)
        
        # Test minimal parameters
        error3 = InsufficientTrainingDataError(room_id="bathroom")
        assert "Insufficient training data for room bathroom" in str(error3)

    def test_data_validation_error_multiple_signatures(self):
        """Test DataValidationError handles both new and legacy signatures."""
        # Test new signature with validation_errors list
        error1 = DataValidationError(
            data_source="sensor_events",
            validation_errors=["Missing required field: room_id", "Invalid timestamp format"],
            sample_data={"sensor_id": "test_sensor", "state": "on"},
        )
        assert "Missing required field: room_id; Invalid timestamp format" in str(error1)
        assert "validation_errors" in error1.context
        assert "sample_data" in error1.context
        
        # Test legacy signature
        error2 = DataValidationError(
            data_source="ignored",  # Will be overridden by legacy logic
            data_type="entity_id",
            validation_rule="must follow Home Assistant format",
            actual_value="invalid.entity.id",
            expected_value="domain.entity_id",
            field_name="motion_sensor",
        )
        assert "entity_id" in str(error2)
        assert "validation_rule" in error2.context
        assert "field_name" in error2.context

    def test_api_rate_limit_error_with_reset_time(self):
        """Test RateLimitExceededError includes reset time information."""
        from src.core.exceptions import RateLimitExceededError
        
        error = RateLimitExceededError(
            service="HomeAssistant",
            limit=100,
            window_seconds=60,
            reset_time="2024-01-15T14:30:00Z",
        )
        
        assert "100 requests per 60s" in str(error)
        assert error.context["service"] == "HomeAssistant"
        assert error.context["reset_time"] == "2024-01-15T14:30:00Z"
        assert error.error_code == "RATE_LIMIT_EXCEEDED_ERROR"

    def test_validation_helper_functions(self):
        """Test validate_room_id and validate_entity_id helper functions."""
        # Test valid room IDs
        validate_room_id("living_room")
        validate_room_id("bedroom-1")
        validate_room_id("kitchen2")
        
        # Test invalid room IDs
        with pytest.raises(DataValidationError, match="must be non-empty string"):
            validate_room_id("")
        
        with pytest.raises(DataValidationError, match="must be non-empty string"):
            validate_room_id(None)
        
        with pytest.raises(DataValidationError, match="must contain only alphanumeric"):
            validate_room_id("living room")  # Space not allowed
        
        with pytest.raises(DataValidationError, match="must contain only alphanumeric"):
            validate_room_id("kitchen@home")  # Special char not allowed
        
        # Test valid entity IDs
        validate_entity_id("binary_sensor.living_room_motion")
        validate_entity_id("sensor.bedroom_temperature")
        validate_entity_id("switch.kitchen_light")
        
        # Test invalid entity IDs
        with pytest.raises(DataValidationError, match="must be non-empty string"):
            validate_entity_id("")
        
        with pytest.raises(DataValidationError, match="must follow Home Assistant format"):
            validate_entity_id("invalid_entity_id")
        
        with pytest.raises(DataValidationError, match="must follow Home Assistant format"):
            validate_entity_id("BINARY_SENSOR.motion")  # Uppercase not allowed


class TestConfigurationDataClasses:
    """Test configuration dataclass functionality."""

    def test_home_assistant_config_initialization(self):
        """Test HomeAssistantConfig initialization with required and optional parameters."""
        # Test with required parameters only
        config = HomeAssistantConfig(
            url="http://homeassistant.local:8123",
            token="test_token_123",
        )
        assert config.url == "http://homeassistant.local:8123"
        assert config.token == "test_token_123"
        assert config.websocket_timeout == 30  # Default value
        assert config.api_timeout == 10  # Default value
        
        # Test with all parameters
        config_full = HomeAssistantConfig(
            url="https://ha.example.com",
            token="long_token_string",
            websocket_timeout=60,
            api_timeout=20,
        )
        assert config_full.websocket_timeout == 60
        assert config_full.api_timeout == 20

    @patch('os.getenv')
    def test_database_config_post_init_environment_override(self, mock_getenv):
        """Test DatabaseConfig.__post_init__() with DATABASE_URL environment variable override."""
        # Test without environment variable
        mock_getenv.return_value = None
        config = DatabaseConfig(
            connection_string="postgresql://localhost:5432/test"
        )
        assert config.connection_string == "postgresql://localhost:5432/test"
        
        # Test with environment variable override
        mock_getenv.return_value = "postgresql://env-host:5432/env-db"
        config_with_env = DatabaseConfig(
            connection_string="postgresql://localhost:5432/test"
        )
        assert config_with_env.connection_string == "postgresql://env-host:5432/env-db"
        mock_getenv.assert_called_with("DATABASE_URL")

    def test_mqtt_config_extensive_options(self):
        """Test MQTTConfig initialization with extensive MQTT configuration options."""
        config = MQTTConfig(
            broker="mqtt.example.com",
            port=8883,
            username="mqtt_user",
            password="mqtt_pass",
            topic_prefix="ha_ml/predictions",
            discovery_enabled=True,
            discovery_prefix="homeassistant",
            device_name="Custom Predictor",
            publishing_enabled=True,
            prediction_qos=2,
            system_qos=1,
            keepalive=120,
        )
        
        assert config.broker == "mqtt.example.com"
        assert config.port == 8883
        assert config.username == "mqtt_user"
        assert config.password == "mqtt_pass"
        assert config.topic_prefix == "ha_ml/predictions"
        assert config.discovery_enabled is True
        assert config.discovery_prefix == "homeassistant"
        assert config.device_name == "Custom Predictor"
        assert config.prediction_qos == 2
        assert config.system_qos == 1
        assert config.keepalive == 120
        
        # Test default values
        default_config = MQTTConfig(broker="localhost")
        assert default_config.port == 1883
        assert default_config.topic_prefix == "occupancy/predictions"
        assert default_config.discovery_enabled is True
        assert default_config.device_manufacturer == "HA ML Predictor"

    def test_prediction_features_logging_configs_defaults(self):
        """Test PredictionConfig, FeaturesConfig, LoggingConfig with default values."""
        # Test PredictionConfig defaults
        pred_config = PredictionConfig()
        assert pred_config.interval_seconds == 300
        assert pred_config.accuracy_threshold_minutes == 15
        assert pred_config.confidence_threshold == 0.7
        
        # Test FeaturesConfig defaults
        feat_config = FeaturesConfig()
        assert feat_config.lookback_hours == 24
        assert feat_config.sequence_length == 50
        assert feat_config.temporal_features is True
        assert feat_config.sequential_features is True
        assert feat_config.contextual_features is True
        
        # Test LoggingConfig defaults
        log_config = LoggingConfig()
        assert log_config.level == "INFO"
        assert log_config.format == "structured"
        assert log_config.file_logging is False
        assert log_config.enable_json_logging is False
        assert log_config.file_path is None
        assert log_config.file_max_size == "10MB"
        assert log_config.file_backup_count == 5

    def test_tracking_config_post_init_default_thresholds(self):
        """Test TrackingConfig.__post_init__() with default alert_thresholds creation."""
        config = TrackingConfig()
        assert config.alert_thresholds is not None
        assert isinstance(config.alert_thresholds, dict)
        
        expected_thresholds = {
            "accuracy_warning": 70.0,
            "accuracy_critical": 50.0,
            "error_warning": 20.0,
            "error_critical": 30.0,
            "trend_degrading": -5.0,
            "validation_lag_warning": 15.0,
            "validation_lag_critical": 30.0,
        }
        
        for key, value in expected_thresholds.items():
            assert key in config.alert_thresholds
            assert config.alert_thresholds[key] == value
        
        # Test with provided alert_thresholds (should not be overridden)
        custom_thresholds = {"custom_warning": 80.0}
        custom_config = TrackingConfig(alert_thresholds=custom_thresholds)
        assert custom_config.alert_thresholds == custom_thresholds

    def test_sensor_config_parameters(self):
        """Test SensorConfig with entity_id, sensor_type, room_id parameters."""
        sensor = SensorConfig(
            entity_id="binary_sensor.living_room_motion",
            sensor_type="motion",
            room_id="living_room",
        )
        
        assert sensor.entity_id == "binary_sensor.living_room_motion"
        assert sensor.sensor_type == "motion"
        assert sensor.room_id == "living_room"


class TestJWTConfiguration:
    """Test JWT configuration with comprehensive environment variable handling."""

    @patch('os.getenv')
    def test_jwt_config_disabled_via_environment(self, mock_getenv):
        """Test JWTConfig.__post_init__() with JWT_ENABLED environment variable variations."""
        # Test "false" value
        mock_getenv.side_effect = lambda key, default=None: "false" if key == "JWT_ENABLED" else default
        config = JWTConfig()
        assert config.enabled is False
        
        # Test "0" value
        mock_getenv.side_effect = lambda key, default=None: "0" if key == "JWT_ENABLED" else default
        config = JWTConfig()
        assert config.enabled is False
        
        # Test "no" value
        mock_getenv.side_effect = lambda key, default=None: "no" if key == "JWT_ENABLED" else default
        config = JWTConfig()
        assert config.enabled is False
        
        # Test "off" value
        mock_getenv.side_effect = lambda key, default=None: "off" if key == "JWT_ENABLED" else default
        config = JWTConfig()
        assert config.enabled is False

    @patch('os.getenv')
    def test_jwt_config_secret_key_from_environment(self, mock_getenv):
        """Test JWTConfig secret key loading from JWT_SECRET_KEY environment variable."""
        mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "true",
            "JWT_SECRET_KEY": "environment_secret_key_that_is_long_enough_for_validation",
        }.get(key, default)
        
        config = JWTConfig()
        assert config.enabled is True
        assert config.secret_key == "environment_secret_key_that_is_long_enough_for_validation"

    @patch('os.getenv')
    @patch('builtins.print')
    def test_jwt_config_test_environment_fallback(self, mock_print, mock_getenv):
        """Test JWTConfig test environment fallback with default test secret key."""
        mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "true",
            "JWT_SECRET_KEY": "",
            "ENVIRONMENT": "test",
        }.get(key, default)
        
        config = JWTConfig()
        assert config.enabled is True
        assert "test_jwt_secret" in config.secret_key
        assert len(config.secret_key) >= 32
        mock_print.assert_called_once()
        assert "Warning: Using default test JWT secret key" in mock_print.call_args[0][0]
        
        # Test with CI environment
        mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "true",
            "JWT_SECRET_KEY": "",
            "ENVIRONMENT": "",
            "CI": "true",
        }.get(key, default)
        
        config_ci = JWTConfig()
        assert config_ci.enabled is True
        assert len(config_ci.secret_key) >= 32

    @patch('os.getenv')
    def test_jwt_config_secret_key_length_validation(self, mock_getenv):
        """Test JWTConfig secret key length validation (minimum 32 characters)."""
        # Test with short secret key
        mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "true",
            "JWT_SECRET_KEY": "short_key",
        }.get(key, default)
        
        with pytest.raises(ValueError, match="JWT secret key must be at least 32 characters long"):
            JWTConfig()
        
        # Test with exactly 32 characters
        mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "true",
            "JWT_SECRET_KEY": "a" * 32,
        }.get(key, default)
        
        config = JWTConfig()
        assert len(config.secret_key) == 32

    @patch('os.getenv')
    def test_jwt_config_missing_secret_key_error(self, mock_getenv):
        """Test JWTConfig ValueError for missing secret key in non-test environments."""
        mock_getenv.side_effect = lambda key, default="": {
            "JWT_ENABLED": "true",
            "JWT_SECRET_KEY": "",
            "ENVIRONMENT": "production",
            "CI": "",
        }.get(key, default)
        
        with pytest.raises(ValueError, match="JWT is enabled but JWT_SECRET_KEY environment variable is not set"):
            JWTConfig()

    def test_jwt_config_default_values(self):
        """Test JWTConfig default field values."""
        with patch('os.getenv') as mock_getenv:
            mock_getenv.side_effect = lambda key, default="": {
                "JWT_ENABLED": "true",
                "JWT_SECRET_KEY": "test_secret_key_that_is_long_enough_for_jwt_validation",
            }.get(key, default)
            
            config = JWTConfig()
            assert config.algorithm == "HS256"
            assert config.access_token_expire_minutes == 60
            assert config.refresh_token_expire_days == 30
            assert config.issuer == "ha-ml-predictor"
            assert config.audience == "ha-ml-predictor-api"
            assert config.require_https is False
            assert config.secure_cookies is False
            assert config.blacklist_enabled is True


class TestAPIConfiguration:
    """Test API configuration with extensive environment variable loading."""

    @patch('os.getenv')
    def test_api_config_environment_variable_loading(self, mock_getenv):
        """Test APIConfig.__post_init__() environment variable loading."""
        mock_getenv.side_effect = lambda key, default: {
            "API_ENABLED": "true",
            "API_HOST": "127.0.0.1",
            "API_PORT": "8080",
            "API_DEBUG": "true",
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        assert config.enabled is True
        assert config.host == "127.0.0.1"
        assert config.port == 8080
        assert config.debug is True

    @patch('os.getenv')
    def test_api_config_cors_configuration(self, mock_getenv):
        """Test APIConfig CORS configuration with CORS_ENABLED, CORS_ALLOW_ORIGINS."""
        mock_getenv.side_effect = lambda key, default: {
            "CORS_ENABLED": "false",
            "CORS_ALLOW_ORIGINS": "http://localhost:3000,https://app.example.com",
        }.get(key, str(default) if default is not None else "*")
        
        config = APIConfig()
        assert config.enable_cors is False
        assert "http://localhost:3000" in config.cors_origins
        assert "https://app.example.com" in config.cors_origins
        assert len(config.cors_origins) == 2

    @patch('os.getenv')
    def test_api_config_api_key_configuration(self, mock_getenv):
        """Test APIConfig API key configuration (API_KEY, API_KEY_ENABLED)."""
        mock_getenv.side_effect = lambda key, default: {
            "API_KEY": "secret_api_key_123",
            "API_KEY_ENABLED": "true",
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        assert config.api_key == "secret_api_key_123"
        assert config.api_key_enabled is True

    @patch('os.getenv')
    def test_api_config_rate_limiting_settings(self, mock_getenv):
        """Test APIConfig rate limiting settings."""
        mock_getenv.side_effect = lambda key, default: {
            "API_RATE_LIMIT_ENABLED": "false",
            "API_RATE_LIMIT_PER_MINUTE": "120",
            "API_RATE_LIMIT_BURST": "200",
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        assert config.rate_limit_enabled is False
        assert config.requests_per_minute == 120
        assert config.burst_limit == 200

    @patch('os.getenv')
    def test_api_config_background_tasks_settings(self, mock_getenv):
        """Test APIConfig background tasks settings."""
        mock_getenv.side_effect = lambda key, default: {
            "API_BACKGROUND_TASKS_ENABLED": "false",
            "HEALTH_CHECK_INTERVAL_SECONDS": "120",
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        assert config.background_tasks_enabled is False
        assert config.health_check_interval_seconds == 120

    @patch('os.getenv')
    def test_api_config_logging_settings(self, mock_getenv):
        """Test APIConfig logging settings."""
        mock_getenv.side_effect = lambda key, default: {
            "API_ACCESS_LOG": "false",
            "API_LOG_REQUESTS": "false",
            "API_LOG_RESPONSES": "true",
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        assert config.access_log is False
        assert config.log_requests is False
        assert config.log_responses is True

    @patch('os.getenv')
    def test_api_config_documentation_settings(self, mock_getenv):
        """Test APIConfig documentation settings."""
        mock_getenv.side_effect = lambda key, default: {
            "API_INCLUDE_DOCS": "false",
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        assert config.include_docs is False
        assert config.docs_url == "/docs"  # Default value
        assert config.redoc_url == "/redoc"  # Default value

    @patch('os.getenv')
    def test_api_config_missing_api_key_error(self, mock_getenv):
        """Test APIConfig ValueError for missing API key when enabled."""
        mock_getenv.side_effect = lambda key, default: {
            "API_KEY": "",
            "API_KEY_ENABLED": "true",
        }.get(key, str(default) if default is not None else "")
        
        with pytest.raises(ValueError, match="API key is enabled but API_KEY environment variable is not set"):
            APIConfig()

    @patch('os.getenv')
    def test_api_config_default_cors_origins_splitting(self, mock_getenv):
        """Test APIConfig default CORS origins splitting on comma."""
        # Test default behavior (should be ["*"])
        mock_getenv.side_effect = lambda key, default: {
        }.get(key, str(default) if default is not None else "*")
        
        config = APIConfig()
        assert config.cors_origins == ["*"]
        
        # Test comma-separated origins
        mock_getenv.side_effect = lambda key, default: {
            "CORS_ALLOW_ORIGINS": "origin1.com, origin2.com , origin3.com",
        }.get(key, str(default) if default is not None else "*")
        
        config_multi = APIConfig()
        expected_origins = ["origin1.com", "origin2.com", "origin3.com"]
        assert config_multi.cors_origins == expected_origins

    def test_api_config_nested_jwt_config_initialization(self):
        """Test APIConfig nested JWTConfig field initialization."""
        with patch('os.getenv') as mock_getenv:
            # Setup environment for both API and JWT configs
            mock_getenv.side_effect = lambda key, default="": {
                "JWT_ENABLED": "true",
                "JWT_SECRET_KEY": "test_secret_key_that_is_long_enough_for_jwt_validation",
            }.get(key, str(default) if default is not None else "")
            
            config = APIConfig()
            assert isinstance(config.jwt, JWTConfig)
            assert config.jwt.enabled is True
            assert len(config.jwt.secret_key) >= 32


class TestRoomConfiguration:
    """Test room configuration functionality with entity ID extraction."""

    def test_room_config_get_all_entity_ids_nested_dictionary(self, sample_rooms_dict):
        """Test RoomConfig.get_all_entity_ids() with nested dictionary sensor structures."""
        bedroom_data = sample_rooms_dict["rooms"]["bedroom"]
        room = RoomConfig(
            room_id="bedroom",
            name=bedroom_data["name"],
            sensors=bedroom_data["sensors"],
        )
        
        entity_ids = room.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.bedroom_motion",
            "binary_sensor.bedroom_door",
            "binary_sensor.bedroom_closet_door",
        ]
        
        assert len(entity_ids) == 3
        for expected_id in expected_ids:
            assert expected_id in entity_ids

    def test_room_config_get_all_entity_ids_list_structure(self):
        """Test RoomConfig.get_all_entity_ids() with list-based sensor structures."""
        sensors_with_lists = {
            "motion": ["binary_sensor.room_motion_1", "binary_sensor.room_motion_2"],
            "door": "binary_sensor.room_door",
            "climate": {
                "temperature": "sensor.room_temperature",
                "humidity": ["sensor.room_humidity_1", "sensor.room_humidity_2"],
            },
        }
        
        room = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors=sensors_with_lists,
        )
        
        entity_ids = room.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.room_motion_1",
            "binary_sensor.room_motion_2",
            "binary_sensor.room_door",
            "sensor.room_temperature",
            "sensor.room_humidity_1",
            "sensor.room_humidity_2",
        ]
        
        assert len(entity_ids) == 6
        for expected_id in expected_ids:
            assert expected_id in entity_ids

    def test_room_config_get_all_entity_ids_prefix_filtering(self):
        """Test RoomConfig.get_all_entity_ids() entity ID extraction with prefix filtering."""
        sensors_mixed = {
            "motion": "binary_sensor.room_motion",
            "door": "binary_sensor.room_door",
            "temperature": "sensor.room_temperature",
            "switch": "switch.room_light",  # Should be ignored (no binary_sensor. or sensor. prefix)
            "camera": "camera.room_security",  # Should be ignored
            "invalid": "not_an_entity_id",  # Should be ignored
        }
        
        room = RoomConfig(
            room_id="filtered_room",
            name="Filtered Room",
            sensors=sensors_mixed,
        )
        
        entity_ids = room.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.room_motion",
            "binary_sensor.room_door",
            "sensor.room_temperature",
        ]
        
        assert len(entity_ids) == 3
        for expected_id in expected_ids:
            assert expected_id in entity_ids
        
        # Ensure filtered out entities are not present
        assert "switch.room_light" not in entity_ids
        assert "camera.room_security" not in entity_ids
        assert "not_an_entity_id" not in entity_ids

    def test_room_config_get_sensors_by_type_dictionary(self):
        """Test RoomConfig.get_sensors_by_type() with dictionary sensor type."""
        room = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "motion": {
                    "main": "binary_sensor.room_motion_main",
                    "backup": "binary_sensor.room_motion_backup",
                },
                "door": "binary_sensor.room_door",
            },
        )
        
        motion_sensors = room.get_sensors_by_type("motion")
        assert isinstance(motion_sensors, dict)
        assert motion_sensors["main"] == "binary_sensor.room_motion_main"
        assert motion_sensors["backup"] == "binary_sensor.room_motion_backup"

    def test_room_config_get_sensors_by_type_string(self):
        """Test RoomConfig.get_sensors_by_type() with string sensor type."""
        room = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "motion": "binary_sensor.room_motion",
                "door": "binary_sensor.room_door",
            },
        )
        
        door_sensors = room.get_sensors_by_type("door")
        assert isinstance(door_sensors, dict)
        assert door_sensors["door"] == "binary_sensor.room_door"

    def test_room_config_get_sensors_by_type_missing(self):
        """Test RoomConfig.get_sensors_by_type() with missing sensor type."""
        room = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "motion": "binary_sensor.room_motion",
            },
        )
        
        missing_sensors = room.get_sensors_by_type("nonexistent")
        assert isinstance(missing_sensors, dict)
        assert len(missing_sensors) == 0


class TestSystemConfiguration:
    """Test SystemConfig aggregation and lookup functionality."""

    def test_system_config_get_all_entity_ids_aggregation(self, sample_config_dict, sample_rooms_dict):
        """Test SystemConfig.get_all_entity_ids() aggregation from all rooms."""
        # Create individual config objects
        ha_config = HomeAssistantConfig(**sample_config_dict["home_assistant"])
        db_config = DatabaseConfig(**sample_config_dict["database"])
        mqtt_config = MQTTConfig(**sample_config_dict["mqtt"])
        pred_config = PredictionConfig(**sample_config_dict["prediction"])
        feat_config = FeaturesConfig(**sample_config_dict["features"])
        log_config = LoggingConfig(**sample_config_dict["logging"])
        track_config = TrackingConfig(**sample_config_dict["tracking"])
        api_config = APIConfig(**sample_config_dict["api"])
        
        # Create room configs
        rooms = {}
        for room_id, room_data in sample_rooms_dict["rooms"].items():
            if room_id == "hallway":  # Handle nested structure
                for sub_room_id, sub_room_data in room_data.items():
                    full_room_id = f"{room_id}_{sub_room_id}"
                    rooms[full_room_id] = RoomConfig(
                        room_id=full_room_id,
                        name=sub_room_data["name"],
                        sensors=sub_room_data["sensors"],
                    )
            else:
                rooms[room_id] = RoomConfig(
                    room_id=room_id,
                    name=room_data["name"],
                    sensors=room_data["sensors"],
                )
        
        system_config = SystemConfig(
            home_assistant=ha_config,
            database=db_config,
            mqtt=mqtt_config,
            prediction=pred_config,
            features=feat_config,
            logging=log_config,
            tracking=track_config,
            api=api_config,
            rooms=rooms,
        )
        
        all_entity_ids = system_config.get_all_entity_ids()
        
        # Should include entities from all rooms
        expected_entities = {
            "binary_sensor.living_room_motion",
            "binary_sensor.living_room_door",
            "sensor.living_room_temperature",
            "binary_sensor.bedroom_motion",
            "binary_sensor.bedroom_door",
            "binary_sensor.bedroom_closet_door",
            "binary_sensor.upper_hallway_motion",
            "binary_sensor.lower_hallway_motion",
        }
        
        assert len(all_entity_ids) == len(expected_entities)
        for entity_id in expected_entities:
            assert entity_id in all_entity_ids

    def test_system_config_get_all_entity_ids_duplicate_removal(self):
        """Test SystemConfig.get_all_entity_ids() duplicate removal (set conversion)."""
        # Create rooms with duplicate entity IDs
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={"motion": "binary_sensor.shared_motion"},
            ),
            "room2": RoomConfig(
                room_id="room2",
                name="Room 2",
                sensors={"motion": "binary_sensor.shared_motion"},  # Duplicate
            ),
        }
        
        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="http://test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(),
            api=APIConfig(),
            rooms=rooms,
        )
        
        all_entity_ids = system_config.get_all_entity_ids()
        
        # Should have only one instance of the duplicate entity ID
        assert len(all_entity_ids) == 1
        assert "binary_sensor.shared_motion" in all_entity_ids

    def test_system_config_get_room_by_entity_id_lookup(self, sample_rooms_dict):
        """Test SystemConfig.get_room_by_entity_id() entity lookup across rooms."""
        rooms = {}
        for room_id, room_data in sample_rooms_dict["rooms"].items():
            if room_id != "hallway":  # Skip nested structure for simplicity
                rooms[room_id] = RoomConfig(
                    room_id=room_id,
                    name=room_data["name"],
                    sensors=room_data["sensors"],
                )
        
        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="http://test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(),
            api=APIConfig(),
            rooms=rooms,
        )
        
        # Test finding existing entities
        living_room = system_config.get_room_by_entity_id("binary_sensor.living_room_motion")
        assert living_room is not None
        assert living_room.room_id == "living_room"
        
        bedroom = system_config.get_room_by_entity_id("binary_sensor.bedroom_door")
        assert bedroom is not None
        assert bedroom.room_id == "bedroom"

    def test_system_config_get_room_by_entity_id_nonexistent(self):
        """Test SystemConfig.get_room_by_entity_id() with non-existent entity (None return)."""
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={"motion": "binary_sensor.room1_motion"},
            ),
        }
        
        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="http://test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(),
            api=APIConfig(),
            rooms=rooms,
        )
        
        # Test with non-existent entity
        result = system_config.get_room_by_entity_id("binary_sensor.nonexistent_sensor")
        assert result is None


class TestConfigLoader:
    """Test configuration loading and validation functionality."""

    def test_config_loader_init_valid_directory(self):
        """Test ConfigLoader.__init__() with valid config directory."""
        with patch('pathlib.Path.exists', return_value=True):
            loader = ConfigLoader(config_dir="/valid/config/dir")
            assert str(loader.config_dir) == "/valid/config/dir"

    def test_config_loader_init_missing_directory(self):
        """Test ConfigLoader.__init__() with missing config directory (FileNotFoundError)."""
        with patch('pathlib.Path.exists', return_value=False):
            with pytest.raises(FileNotFoundError, match="Configuration directory not found: /missing/dir"):
                ConfigLoader(config_dir="/missing/dir")

    def test_config_loader_load_yaml_valid_files(self, sample_config_dict):
        """Test ConfigLoader._load_yaml() with valid YAML files."""
        yaml_content = "home_assistant:\n  url: http://test\n  token: test123"
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data=yaml_content)), \
             patch('yaml.safe_load', return_value=sample_config_dict):
            
            loader = ConfigLoader()
            result = loader._load_yaml("config.yaml")
            assert result == sample_config_dict

    def test_config_loader_load_yaml_missing_file(self):
        """Test ConfigLoader._load_yaml() with missing files (FileNotFoundError)."""
        with patch('pathlib.Path.exists', side_effect=lambda: False):  # config dir exists but file doesn't
            loader = ConfigLoader()
            with patch('pathlib.Path.exists', return_value=False):  # file doesn't exist
                with pytest.raises(FileNotFoundError, match="Configuration file not found"):
                    loader._load_yaml("missing.yaml")

    def test_config_loader_load_yaml_invalid_content(self):
        """Test ConfigLoader._load_yaml() with invalid YAML content."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="invalid: yaml: content: [")):
            
            loader = ConfigLoader()
            # YAML parsing error should propagate
            with pytest.raises(Exception):  # yaml.YAMLError or similar
                loader._load_yaml("invalid.yaml")

    def test_config_loader_load_yaml_non_dict_return(self):
        """Test ConfigLoader._load_yaml() return type handling (dict vs non-dict)."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="just a string")), \
             patch('yaml.safe_load', return_value="not a dictionary"):
            
            loader = ConfigLoader()
            result = loader._load_yaml("string.yaml")
            assert result == {}  # Should return empty dict for non-dict content

    @patch('yaml.safe_load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_config_loader_load_config_with_environment(self, mock_exists, mock_open_file, mock_yaml, sample_config_dict, sample_rooms_dict):
        """Test ConfigLoader.load_config() with environment parameter."""
        mock_exists.return_value = True
        mock_yaml.side_effect = [sample_config_dict, sample_rooms_dict]
        
        loader = ConfigLoader()
        
        # Test loading with specific environment
        config = loader.load_config(environment="production")
        assert isinstance(config, SystemConfig)
        
        # Should attempt to load config.production.yaml first
        expected_calls = [
            call(Path("config") / "config.production.yaml", "r", encoding="utf-8"),
            call(Path("config") / "rooms.yaml", "r", encoding="utf-8"),
        ]
        mock_open_file.assert_has_calls(expected_calls, any_order=False)

    @patch('yaml.safe_load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_config_loader_load_config_environment_fallback(self, mock_exists, mock_open_file, mock_yaml, sample_config_dict, sample_rooms_dict):
        """Test ConfigLoader.load_config() fallback to base config.yaml when environment config missing."""
        # Simulate environment-specific config missing, base config exists
        def exists_side_effect(path_obj=None):
            if path_obj is None:  # ConfigLoader init check
                return True
            path_str = str(path_obj)
            if "config.staging.yaml" in path_str:
                return False  # Environment config doesn't exist
            return True  # Other files exist
        
        mock_exists.side_effect = exists_side_effect
        mock_yaml.side_effect = [sample_config_dict, sample_rooms_dict]
        
        # Mock FileNotFoundError for environment-specific config
        def open_side_effect(path, *args, **kwargs):
            if "config.staging.yaml" in str(path):
                raise FileNotFoundError("Environment config not found")
            return mock_open().return_value
        
        mock_open_file.side_effect = open_side_effect
        
        loader = ConfigLoader()
        
        # Should fall back to base config.yaml
        with patch.object(loader, '_load_yaml') as mock_load_yaml:
            mock_load_yaml.side_effect = [FileNotFoundError(), sample_config_dict, sample_rooms_dict]
            
            config = loader.load_config(environment="staging")
            assert isinstance(config, SystemConfig)
            
            # Should have attempted environment config first, then base config
            expected_calls = [
                call("config.staging.yaml"),
                call("config.yaml"),
                call("rooms.yaml"),
            ]
            mock_load_yaml.assert_has_calls(expected_calls)

    @patch('yaml.safe_load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_config_loader_load_config_rooms_integration(self, mock_exists, mock_open_file, mock_yaml, sample_config_dict, sample_rooms_dict):
        """Test ConfigLoader.load_config() rooms.yaml integration."""
        mock_exists.return_value = True
        mock_yaml.side_effect = [sample_config_dict, sample_rooms_dict]
        
        loader = ConfigLoader()
        config = loader.load_config()
        
        # Should have loaded rooms from rooms.yaml
        assert len(config.rooms) > 0
        assert "living_room" in config.rooms
        assert "bedroom" in config.rooms
        assert "hallway_upper" in config.rooms  # Nested room
        assert "hallway_lower" in config.rooms  # Nested room

    @patch('yaml.safe_load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_config_loader_load_config_nested_room_handling(self, mock_exists, mock_open_file, mock_yaml, sample_config_dict, sample_rooms_dict):
        """Test ConfigLoader.load_config() nested room structure handling (hallways example)."""
        mock_exists.return_value = True
        mock_yaml.side_effect = [sample_config_dict, sample_rooms_dict]
        
        loader = ConfigLoader()
        config = loader.load_config()
        
        # Test nested hallway rooms
        assert "hallway_upper" in config.rooms
        assert "hallway_lower" in config.rooms
        
        upper_hallway = config.rooms["hallway_upper"]
        lower_hallway = config.rooms["hallway_lower"]
        
        assert upper_hallway.name == "Upper Hallway"
        assert lower_hallway.name == "Lower Hallway"
        assert upper_hallway.room_id == "hallway_upper"
        assert lower_hallway.room_id == "hallway_lower"

    @patch('yaml.safe_load')
    @patch('builtins.open')
    @patch('pathlib.Path.exists')
    def test_config_loader_load_config_dataclass_creation(self, mock_exists, mock_open_file, mock_yaml, sample_config_dict, sample_rooms_dict):
        """Test ConfigLoader.load_config() dataclass object creation for all config sections."""
        mock_exists.return_value = True
        mock_yaml.side_effect = [sample_config_dict, sample_rooms_dict]
        
        loader = ConfigLoader()
        config = loader.load_config()
        
        # Test all configuration objects are created
        assert isinstance(config.home_assistant, HomeAssistantConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.mqtt, MQTTConfig)
        assert isinstance(config.prediction, PredictionConfig)
        assert isinstance(config.features, FeaturesConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.tracking, TrackingConfig)
        assert isinstance(config.api, APIConfig)
        assert isinstance(config.api.jwt, JWTConfig)
        
        # Test values are properly set
        assert config.home_assistant.url == "http://homeassistant.local:8123"
        assert config.database.pool_size == 10
        assert config.mqtt.broker == "localhost"


class TestGlobalConfiguration:
    """Test global configuration functions and singleton behavior."""

    def test_get_config_singleton_behavior(self):
        """Test get_config() singleton behavior with global _config_instance."""
        # Clear the global instance first
        import src.core.config as config_module
        config_module._config_instance = None
        
        with patch('src.core.config.ConfigLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_system_config = Mock(spec=SystemConfig)
            mock_loader.load_config.return_value = mock_system_config
            mock_loader_class.return_value = mock_loader
            
            # First call should create instance
            config1 = get_config()
            assert config1 is mock_system_config
            
            # Second call should return same instance
            config2 = get_config()
            assert config2 is config1
            
            # Loader should only be called once
            mock_loader_class.assert_called_once()
            mock_loader.load_config.assert_called_once()
        
        # Clean up
        config_module._config_instance = None

    def test_get_config_environment_manager_integration(self):
        """Test get_config() environment manager integration (when available)."""
        import src.core.config as config_module
        config_module._config_instance = None
        
        # Mock successful environment manager import and usage
        mock_env_manager = Mock()
        mock_processed_config = {"home_assistant": {"url": "http://test", "token": "test"}}
        mock_env_manager.load_environment_config.return_value = mock_processed_config
        
        mock_system_config = Mock(spec=SystemConfig)
        
        with patch('src.core.config.get_environment_manager', return_value=mock_env_manager), \
             patch('src.core.config.ConfigLoader') as mock_loader_class:
            
            mock_loader = Mock()
            mock_loader._create_system_config.return_value = mock_system_config
            mock_loader_class.return_value = mock_loader
            
            config = get_config()
            
            # Should use environment manager
            mock_env_manager.load_environment_config.assert_called_once()
            mock_loader._create_system_config.assert_called_once_with(mock_processed_config)
            assert config is mock_system_config
        
        # Clean up
        config_module._config_instance = None

    def test_get_config_import_error_fallback(self):
        """Test get_config() ImportError fallback to direct ConfigLoader."""
        import src.core.config as config_module
        config_module._config_instance = None
        
        mock_system_config = Mock(spec=SystemConfig)
        
        # Mock ImportError when trying to import environment manager
        with patch('src.core.config.get_environment_manager', side_effect=ImportError("Environment manager not available")), \
             patch('src.core.config.ConfigLoader') as mock_loader_class:
            
            mock_loader = Mock()
            mock_loader.load_config.return_value = mock_system_config
            mock_loader_class.return_value = mock_loader
            
            config = get_config(environment="test")
            
            # Should fall back to direct ConfigLoader
            mock_loader_class.assert_called_once()
            mock_loader.load_config.assert_called_once_with("test")
            assert config is mock_system_config
        
        # Clean up
        config_module._config_instance = None

    def test_reload_config_forced_reload(self):
        """Test reload_config() forced reload behavior."""
        import src.core.config as config_module
        
        # Set up initial state
        old_config = Mock(spec=SystemConfig)
        config_module._config_instance = old_config
        
        new_config = Mock(spec=SystemConfig)
        
        with patch('src.core.config.ConfigLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_config.return_value = new_config
            mock_loader_class.return_value = mock_loader
            
            # Mock ImportError to test direct loading path
            with patch('src.core.config.get_environment_manager', side_effect=ImportError()):
                reloaded_config = reload_config(environment="production")
                
                # Should create new config instance
                assert reloaded_config is new_config
                assert config_module._config_instance is new_config
                assert reloaded_config is not old_config
                
                mock_loader.load_config.assert_called_once_with("production")
        
        # Clean up
        config_module._config_instance = None

    def test_reload_config_environment_manager_path(self):
        """Test reload_config() environment manager vs direct loading paths."""
        import src.core.config as config_module
        config_module._config_instance = None
        
        # Mock environment manager path
        mock_env_manager = Mock()
        mock_processed_config = {"test": "config"}
        mock_env_manager.load_environment_config.return_value = mock_processed_config
        
        new_config = Mock(spec=SystemConfig)
        
        with patch('src.core.config.get_environment_manager', return_value=mock_env_manager), \
             patch('src.core.config.ConfigLoader') as mock_loader_class:
            
            mock_loader = Mock()
            mock_loader._create_system_config.return_value = new_config
            mock_loader_class.return_value = mock_loader
            
            reloaded_config = reload_config()
            
            # Should use environment manager path
            mock_env_manager.load_environment_config.assert_called_once()
            mock_loader._create_system_config.assert_called_once_with(mock_processed_config)
            assert reloaded_config is new_config
            assert config_module._config_instance is new_config
        
        # Clean up
        config_module._config_instance = None


class TestConfigurationEdgeCases:
    """Test configuration system edge cases and error handling."""

    def test_config_loader_with_empty_yaml_files(self):
        """Test configuration loading with empty YAML files."""
        with patch('pathlib.Path.exists', return_value=True), \
             patch('builtins.open', mock_open(read_data="")), \
             patch('yaml.safe_load', return_value=None):
            
            loader = ConfigLoader()
            result = loader._load_yaml("empty.yaml")
            assert result == {}  # Should return empty dict for None content

    def test_config_loader_with_unicode_characters(self):
        """Test configuration loading with Unicode characters in configuration values."""
        unicode_config = {
            "home_assistant": {
                "url": "http://homeassistant.local:8123",
                "token": "test_token_with_émojis_🏠",
            },
            "mqtt": {
                "broker": "mqtt.example.com",
                "device_name": "Détecteur d'Occupancy",
            },
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('yaml.safe_load', return_value=unicode_config):
            
            loader = ConfigLoader()
            result = loader._load_yaml("unicode.yaml")
            assert "émojis" in result["home_assistant"]["token"]
            assert "Détecteur" in result["mqtt"]["device_name"]

    def test_room_config_with_no_sensors(self):
        """Test room configuration with no sensors defined."""
        room = RoomConfig(room_id="empty_room", name="Empty Room", sensors={})
        
        entity_ids = room.get_all_entity_ids()
        assert len(entity_ids) == 0
        
        motion_sensors = room.get_sensors_by_type("motion")
        assert len(motion_sensors) == 0

    def test_room_config_with_null_values(self):
        """Test room configuration with null/None values in sensor definitions."""
        sensors_with_nulls = {
            "motion": "binary_sensor.room_motion",
            "door": None,
            "temperature": {
                "main": "sensor.room_temp",
                "backup": None,
            },
        }
        
        room = RoomConfig(
            room_id="null_room",
            name="Room with Nulls",
            sensors=sensors_with_nulls,
        )
        
        entity_ids = room.get_all_entity_ids()
        # Should only include non-None entity IDs that match the prefix pattern
        assert "binary_sensor.room_motion" in entity_ids
        assert "sensor.room_temp" in entity_ids
        # None values should be ignored
        assert len([eid for eid in entity_ids if eid is None]) == 0

    def test_config_validation_with_extreme_values(self):
        """Test configuration objects with extreme parameter values."""
        # Test with very large timeout values
        ha_config = HomeAssistantConfig(
            url="http://test",
            token="test",
            websocket_timeout=86400,  # 24 hours
            api_timeout=3600,  # 1 hour
        )
        assert ha_config.websocket_timeout == 86400
        assert ha_config.api_timeout == 3600
        
        # Test with very large pool sizes
        db_config = DatabaseConfig(
            connection_string="postgresql://test",
            pool_size=1000,
            max_overflow=2000,
        )
        assert db_config.pool_size == 1000
        assert db_config.max_overflow == 2000

    @patch('os.getenv')
    def test_environment_variable_edge_cases(self, mock_getenv):
        """Test environment variable handling with edge cases."""
        # Test with empty string values
        mock_getenv.side_effect = lambda key, default: {
            "API_ENABLED": "",  # Empty string
            "API_PORT": "0",    # Zero value
            "CORS_ALLOW_ORIGINS": "",  # Empty CORS origins
        }.get(key, str(default) if default is not None else "")
        
        config = APIConfig()
        # Empty string should be treated as False for boolean fields
        assert config.enabled is False
        assert config.port == 0
        assert config.cors_origins == [""]  # Empty string in list

    def test_feature_constants_immutability(self):
        """Test that feature name constants are immutable (list contents)."""
        # These should not raise errors (lists are mutable but we test they exist)
        original_temporal_length = len(TEMPORAL_FEATURE_NAMES)
        original_sequential_length = len(SEQUENTIAL_FEATURE_NAMES)
        original_contextual_length = len(CONTEXTUAL_FEATURE_NAMES)
        
        # Test that constants have expected structure
        assert isinstance(TEMPORAL_FEATURE_NAMES, list)
        assert isinstance(SEQUENTIAL_FEATURE_NAMES, list)
        assert isinstance(CONTEXTUAL_FEATURE_NAMES, list)
        
        # Test feature lists are not empty
        assert original_temporal_length > 0
        assert original_sequential_length > 0
        assert original_contextual_length > 0

    def test_model_params_backward_compatibility(self):
        """Test model parameter aliases for backward compatibility."""
        lstm_params = DEFAULT_MODEL_PARAMS[ModelType.LSTM]
        
        # Test that aliases exist alongside main parameters
        assert "hidden_units" in lstm_params
        assert "lstm_units" in lstm_params
        assert "dropout" in lstm_params
        assert "dropout_rate" in lstm_params
        
        # Test that HMM has both n_iter and max_iter
        hmm_params = DEFAULT_MODEL_PARAMS[ModelType.HMM]
        assert "n_iter" in hmm_params
        assert "max_iter" in hmm_params
        
        # Test that GP alias points to same params as GAUSSIAN_PROCESS
        gp_params = DEFAULT_MODEL_PARAMS[ModelType.GP]
        gaussian_params = DEFAULT_MODEL_PARAMS[ModelType.GAUSSIAN_PROCESS]
        assert gp_params == gaussian_params

    def test_exception_error_context_edge_cases(self):
        """Test exception error context with edge cases."""
        # Test with very long error messages
        long_message = "x" * 1000
        error = OccupancyPredictionError(message=long_message)
        error_str = str(error)
        assert len(error_str) >= len(long_message)
        
        # Test with circular references in context (should not cause infinite recursion)
        context_dict = {"key": "value"}
        context_dict["self_ref"] = context_dict  # Circular reference
        
        # This should not raise an exception
        error_with_circular = OccupancyPredictionError(
            message="Test with circular context",
            context=context_dict,
        )
        # Should be able to convert to string without infinite recursion
        error_str = str(error_with_circular)
        assert "Test with circular context" in error_str


class TestConfigurationPerformance:
    """Test configuration system performance characteristics."""

    def test_large_room_configuration_performance(self):
        """Test configuration loading with very large room configurations."""
        # Create a large number of rooms with multiple sensors
        large_rooms_config = {"rooms": {}}
        
        for i in range(100):  # 100 rooms
            room_id = f"room_{i:03d}"
            large_rooms_config["rooms"][room_id] = {
                "name": f"Room {i}",
                "sensors": {
                    "motion": f"binary_sensor.room_{i:03d}_motion",
                    "door": f"binary_sensor.room_{i:03d}_door",
                    "temperature": f"sensor.room_{i:03d}_temperature",
                    "humidity": f"sensor.room_{i:03d}_humidity",
                    "light": f"sensor.room_{i:03d}_light",
                },
            }
        
        sample_config = {
            "home_assistant": {"url": "http://test", "token": "test"},
            "database": {"connection_string": "test://test"},
            "mqtt": {"broker": "test"},
            "prediction": {},
            "features": {},
            "logging": {},
            "tracking": {},
            "api": {"jwt": {}},
        }
        
        with patch('pathlib.Path.exists', return_value=True), \
             patch('yaml.safe_load', side_effect=[sample_config, large_rooms_config]):
            
            loader = ConfigLoader()
            config = loader.load_config()
            
            # Should handle large configurations
            assert len(config.rooms) == 100
            
            # Test entity ID aggregation performance
            all_entity_ids = config.get_all_entity_ids()
            assert len(all_entity_ids) == 500  # 5 sensors per room * 100 rooms
            
            # Test room lookup performance
            test_entity = "binary_sensor.room_050_motion"
            found_room = config.get_room_by_entity_id(test_entity)
            assert found_room is not None
            assert found_room.room_id == "room_050"

    def test_deep_nested_sensor_structure_performance(self):
        """Test performance with deeply nested sensor structures."""
        # Create deeply nested sensor structure
        deep_sensors = {"level_0": {}}
        current_level = deep_sensors["level_0"]
        
        for i in range(10):  # 10 levels deep
            current_level[f"level_{i+1}"] = {}
            if i == 9:  # At the deepest level, add actual sensors
                current_level[f"level_{i+1}"] = {
                    "sensor_1": "binary_sensor.deep_sensor_1",
                    "sensor_2": "binary_sensor.deep_sensor_2",
                }
            else:
                current_level = current_level[f"level_{i+1}"]
        
        room = RoomConfig(
            room_id="deep_room",
            name="Deep Nested Room",
            sensors=deep_sensors,
        )
        
        # Should handle deep nesting without issues
        entity_ids = room.get_all_entity_ids()
        assert len(entity_ids) == 2
        assert "binary_sensor.deep_sensor_1" in entity_ids
        assert "binary_sensor.deep_sensor_2" in entity_ids