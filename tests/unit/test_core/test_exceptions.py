"""
Comprehensive unit tests for exceptions module to achieve high test coverage.

This module focuses on testing all exception classes, error paths, edge cases,
and validation functions in the exceptions module.
"""

from typing import Any, Dict

import pytest

from src.core.exceptions import (  # Base exceptions; Configuration errors; Home Assistant errors; Database errors; Feature engineering errors; Model errors; Data processing errors; Integration errors; System errors; API errors; Validation functions
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
from src.core.exceptions import APIRateLimitError  # Alias
from src.core.exceptions import PredictionError  # Alias


class TestErrorSeverity:
    """Test ErrorSeverity enum."""

    def test_error_severity_values(self):
        """Test that all severity levels exist and have correct values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"

    def test_error_severity_comparison(self):
        """Test severity level ordering."""
        severities = [
            ErrorSeverity.LOW,
            ErrorSeverity.MEDIUM,
            ErrorSeverity.HIGH,
            ErrorSeverity.CRITICAL,
        ]
        assert len(severities) == 4


class TestOccupancyPredictionError:
    """Test base OccupancyPredictionError class."""

    def test_basic_initialization(self):
        """Test basic error initialization."""
        error = OccupancyPredictionError("Test error message")

        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause is None

    def test_full_initialization(self):
        """Test error initialization with all parameters."""
        context = {"key1": "value1", "key2": 42}
        cause = ValueError("Original error")

        error = OccupancyPredictionError(
            message="Test error",
            error_code="TEST_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )

        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR"
        assert error.context == context
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_str_representation_with_all_fields(self):
        """Test string representation with all fields present."""
        context = {"param1": "value1", "param2": 123}
        cause = RuntimeError("Original cause")

        error = OccupancyPredictionError(
            message="Main error", error_code="MAIN_ERROR", context=context, cause=cause
        )

        error_str = str(error)

        assert "Main error" in error_str
        assert "Error Code: MAIN_ERROR" in error_str
        assert "Context: param1=value1, param2=123" in error_str
        assert "Caused by: RuntimeError: Original cause" in error_str

    def test_str_representation_minimal(self):
        """Test string representation with minimal fields."""
        error = OccupancyPredictionError("Simple error")

        assert str(error) == "Simple error"

    def test_context_default_empty_dict(self):
        """Test that context defaults to empty dict, not None."""
        error = OccupancyPredictionError("Test", context=None)

        assert error.context == {}
        assert isinstance(error.context, dict)


class TestConfigurationErrors:
    """Test configuration-related error classes."""

    def test_configuration_error_base(self):
        """Test ConfigurationError base class."""
        error = ConfigurationError("Config error", config_file="test.yaml")

        assert isinstance(error, OccupancyPredictionError)
        assert error.context["config_file"] == "test.yaml"

    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError."""
        error = ConfigFileNotFoundError("config.yaml", "/path/to/config")

        assert "config.yaml" in str(error)
        assert "/path/to/config" in str(error)
        assert error.error_code == "CONFIG_FILE_NOT_FOUND_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["config_dir"] == "/path/to/config"

    def test_config_validation_error_basic(self):
        """Test ConfigValidationError with basic parameters."""
        error = ConfigValidationError("Invalid value")

        assert error.message == "Invalid value"
        assert error.error_code == "CONFIG_VALIDATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH

    def test_config_validation_error_with_field_info(self):
        """Test ConfigValidationError with field information."""
        error = ConfigValidationError(
            field="database.port",
            value=999999,
            expected="1-65535",
            config_file="config.yaml",
        )

        assert "database.port" in str(error)
        assert "999999" in str(error)
        assert "1-65535" in str(error)
        assert error.context["field"] == "database.port"
        assert error.context["value"] == 999999
        assert error.context["expected"] == "1-65535"

    def test_config_validation_error_auto_message(self):
        """Test ConfigValidationError auto-generated message."""
        error = ConfigValidationError(
            message=None,  # Should auto-generate
            field="timeout",
            value=0,
            expected="positive integer",
        )

        expected_msg = (
            "Invalid configuration field 'timeout': got 0, expected positive integer"
        )
        assert error.message == expected_msg

    def test_config_validation_error_with_valid_values(self):
        """Test ConfigValidationError with valid values list."""
        error = ConfigValidationError(
            field="log_level",
            value="INVALID",
            valid_values=["DEBUG", "INFO", "WARNING", "ERROR"],
        )

        assert error.context["valid_values"] == ["DEBUG", "INFO", "WARNING", "ERROR"]

    def test_config_validation_error_with_ellipsis_sentinel(self):
        """Test ConfigValidationError with ellipsis as value sentinel."""
        error = ConfigValidationError(
            field="test_field", value=..., expected="some value"  # Ellipsis sentinel
        )

        # Should not include value in context when ellipsis is used
        assert "value" not in error.context

    def test_missing_config_section_error(self):
        """Test MissingConfigSectionError."""
        error = MissingConfigSectionError("database", "config.yaml")

        assert "database" in str(error)
        assert error.error_code == "CONFIG_SECTION_MISSING_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["section_name"] == "database"

    def test_config_parsing_error(self):
        """Test ConfigParsingError."""
        error = ConfigParsingError("config.yaml", "Invalid YAML syntax")

        assert "config.yaml" in str(error)
        assert "Invalid YAML syntax" in str(error)
        assert error.error_code == "CONFIG_PARSING_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["parse_error"] == "Invalid YAML syntax"


class TestHomeAssistantErrors:
    """Test Home Assistant integration error classes."""

    def test_home_assistant_error_base(self):
        """Test HomeAssistantError base class."""
        error = HomeAssistantError("HA error")

        assert isinstance(error, OccupancyPredictionError)

    def test_home_assistant_connection_error(self):
        """Test HomeAssistantConnectionError."""
        cause = ConnectionError("Network unreachable")
        error = HomeAssistantConnectionError("http://homeassistant:8123", cause)

        assert "http://homeassistant:8123" in str(error)
        assert error.error_code == "HA_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["url"] == "http://homeassistant:8123"
        assert error.cause == cause

    def test_home_assistant_authentication_error_with_string_token(self):
        """Test HomeAssistantAuthenticationError with string token."""
        error = HomeAssistantAuthenticationError(
            "http://homeassistant:8123",
            "very_long_token_that_should_be_truncated_for_security",
        )

        assert "http://homeassistant:8123" in str(error)
        assert error.error_code == "HA_AUTH_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert "very_long_token..." in error.context["token_hint"]
        assert "hint" in error.context

    def test_home_assistant_authentication_error_with_token_length(self):
        """Test HomeAssistantAuthenticationError with integer token length."""
        error = HomeAssistantAuthenticationError(
            "http://homeassistant:8123", 64  # Token length
        )

        assert error.context["token_length"] == 64
        assert "token_length" in error.context

    def test_home_assistant_authentication_error_with_short_token(self):
        """Test HomeAssistantAuthenticationError with short token (no truncation)."""
        error = HomeAssistantAuthenticationError(
            "http://homeassistant:8123", "short_token"
        )

        assert error.context["token_hint"] == "short_token"

    def test_home_assistant_authentication_error_with_other_type(self):
        """Test HomeAssistantAuthenticationError with non-string, non-int token."""
        error = HomeAssistantAuthenticationError(
            "http://homeassistant:8123", ["token", "as", "list"]
        )

        assert error.context["token_hint"] == "['token', 'as', 'list']"

    def test_home_assistant_api_error(self):
        """Test HomeAssistantAPIError."""
        error = HomeAssistantAPIError("/api/states", 404, "Not found", method="GET")

        assert "/api/states" in str(error)
        assert "404" in str(error)
        assert error.error_code == "HA_API_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["endpoint"] == "/api/states"
        assert error.context["method"] == "GET"
        assert error.context["status_code"] == 404
        assert "Not found" in error.context["response"]

    def test_home_assistant_api_error_long_response(self):
        """Test HomeAssistantAPIError with long response text."""
        long_response = "x" * 600  # Longer than 500 char limit
        error = HomeAssistantAPIError("/api/test", 500, long_response)

        # Should be limited to 500 characters
        assert len(error.context["response"]) == 500

    def test_entity_not_found_error_basic(self):
        """Test EntityNotFoundError without room_id."""
        error = EntityNotFoundError("sensor.temperature")

        assert "sensor.temperature" in str(error)
        assert error.error_code == "ENTITY_NOT_FOUND_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["entity_id"] == "sensor.temperature"
        assert "room_id" not in error.context

    def test_entity_not_found_error_with_room(self):
        """Test EntityNotFoundError with room_id."""
        error = EntityNotFoundError("sensor.temperature", "living_room")

        assert "sensor.temperature" in str(error)
        assert "living_room" in str(error)
        assert error.context["entity_id"] == "sensor.temperature"
        assert error.context["room_id"] == "living_room"

    def test_websocket_error_base(self):
        """Test WebSocketError base class."""
        error = WebSocketError(
            "Connection failed", "ws://homeassistant:8123/api/websocket"
        )

        assert "Connection failed" in str(error)
        assert error.error_code == "HA_WEBSOCKET_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["url"] == "ws://homeassistant:8123/api/websocket"
        assert error.context["reason"] == "Connection failed"

    def test_websocket_connection_error(self):
        """Test WebSocketConnectionError."""
        cause = ConnectionError("Connection refused")
        error = WebSocketConnectionError("ws://homeassistant:8123", cause)

        assert error.error_code == "WEBSOCKET_CONNECTION_ERROR"
        assert error.cause == cause

    def test_websocket_authentication_error_basic(self):
        """Test WebSocketAuthenticationError without auth method."""
        error = WebSocketAuthenticationError("ws://homeassistant:8123")

        assert error.error_code == "WEBSOCKET_AUTH_ERROR"
        assert "auth_method" not in error.context

    def test_websocket_authentication_error_with_method(self):
        """Test WebSocketAuthenticationError with auth method."""
        error = WebSocketAuthenticationError("ws://homeassistant:8123", "bearer_token")

        assert error.error_code == "WEBSOCKET_AUTH_ERROR"
        assert error.context["auth_method"] == "bearer_token"
        assert "bearer_token" in str(error)

    def test_websocket_validation_error_basic(self):
        """Test WebSocketValidationError without message type."""
        error = WebSocketValidationError(
            "ws://homeassistant:8123", "Invalid message format"
        )

        assert "Invalid message format" in str(error)
        assert error.error_code == "WEBSOCKET_VALIDATION_ERROR"
        assert error.context["validation_error"] == "Invalid message format"
        assert "message_type" not in error.context

    def test_websocket_validation_error_with_message_type(self):
        """Test WebSocketValidationError with message type."""
        error = WebSocketValidationError(
            "ws://homeassistant:8123", "Invalid message format", "auth"
        )

        assert error.context["message_type"] == "auth"


class TestDatabaseErrors:
    """Test database-related error classes."""

    def test_database_error_base(self):
        """Test DatabaseError base class."""
        error = DatabaseError("DB error")

        assert isinstance(error, OccupancyPredictionError)

    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        cause = ConnectionError("Connection refused")
        conn_string = "postgresql://user:password@localhost/db"
        error = DatabaseConnectionError(conn_string, cause)

        assert error.error_code == "DB_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.cause == cause
        # Password should be masked
        assert "password" not in error.context["connection_string"]
        assert "***" in error.context["connection_string"]

    def test_database_connection_error_password_masking(self):
        """Test DatabaseConnectionError password masking."""
        conn_string = "postgresql://admin:secretpass@localhost:5432/mydb"
        error = DatabaseConnectionError(conn_string)

        masked = error._mask_password(conn_string)
        assert "secretpass" not in masked
        assert "admin:***@localhost" in masked

    def test_database_connection_error_no_password(self):
        """Test DatabaseConnectionError with connection string without password."""
        conn_string = "postgresql://localhost/db"
        error = DatabaseConnectionError(conn_string)

        # Should not modify string without password
        assert error.context["connection_string"] == conn_string

    def test_database_query_error_basic(self):
        """Test DatabaseQueryError with basic parameters."""
        error = DatabaseQueryError("SELECT * FROM table")

        assert "SELECT * FROM table..." in str(error)
        assert error.error_code == "DB_QUERY_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_database_query_error_full_params(self):
        """Test DatabaseQueryError with all parameters."""
        cause = Exception("Connection lost")
        parameters = {"id": 1, "name": "test"}

        error = DatabaseQueryError(
            query="SELECT * FROM users WHERE id = :id AND name = :name",
            parameters=parameters,
            cause=cause,
            error_type="ConnectionError",
            severity=ErrorSeverity.HIGH,
        )

        assert error.context["parameters"] == parameters
        assert error.context["error_type"] == "ConnectionError"
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause

    def test_database_query_error_long_query(self):
        """Test DatabaseQueryError with long query (should be truncated)."""
        long_query = "SELECT " + "column" * 50 + " FROM table"
        error = DatabaseQueryError(long_query)

        # Query in context should be limited to 200 chars
        assert len(error.context["query"]) == 200

    def test_database_migration_error(self):
        """Test DatabaseMigrationError."""
        cause = Exception("Migration script failed")
        error = DatabaseMigrationError("001_create_tables", cause)

        assert "001_create_tables" in str(error)
        assert error.error_code == "DB_MIGRATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["migration_name"] == "001_create_tables"
        assert error.cause == cause

    def test_database_integrity_error(self):
        """Test DatabaseIntegrityError."""
        cause = Exception("Constraint violation")
        values = {"id": 1, "email": "duplicate@example.com"}

        error = DatabaseIntegrityError(
            "unique_email_constraint", "users", values, cause
        )

        assert "unique_email_constraint" in str(error)
        assert "users" in str(error)
        assert error.error_code == "DB_INTEGRITY_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["constraint"] == "unique_email_constraint"
        assert error.context["table"] == "users"
        assert error.context["values"] == values
        assert error.cause == cause

    def test_database_integrity_error_no_values(self):
        """Test DatabaseIntegrityError without values."""
        error = DatabaseIntegrityError("primary_key_constraint", "products")

        assert "values" not in error.context


class TestFeatureEngineeringErrors:
    """Test feature engineering error classes."""

    def test_feature_engineering_error_base(self):
        """Test FeatureEngineeringError base class."""
        error = FeatureEngineeringError("Feature error")

        assert isinstance(error, OccupancyPredictionError)

    def test_feature_extraction_error_basic(self):
        """Test FeatureExtractionError with basic parameters."""
        error = FeatureExtractionError("temporal")

        assert "temporal" in str(error)
        assert error.error_code == "FEATURE_EXTRACTION_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["feature_type"] == "temporal"

    def test_feature_extraction_error_full_params(self):
        """Test FeatureExtractionError with all parameters."""
        cause = ValueError("Invalid data")

        error = FeatureExtractionError(
            "sequential",
            room_id="living_room",
            time_range="2024-01-01 to 2024-01-31",
            cause=cause,
        )

        assert "sequential" in str(error)
        assert "living_room" in str(error)
        assert error.context["feature_type"] == "sequential"
        assert error.context["room_id"] == "living_room"
        assert error.context["time_range"] == "2024-01-01 to 2024-01-31"
        assert error.cause == cause

    def test_insufficient_data_error(self):
        """Test InsufficientDataError."""
        error = InsufficientDataError(
            "sensor_events", "bedroom", required_samples=1000, available_samples=50
        )

        assert "sensor_events" in str(error)
        assert "bedroom" in str(error)
        assert "1000" in str(error)
        assert "50" in str(error)
        assert error.error_code == "INSUFFICIENT_DATA_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["data_type"] == "sensor_events"
        assert error.context["room_id"] == "bedroom"
        assert error.context["required_samples"] == 1000
        assert error.context["available_samples"] == 50

    def test_feature_validation_error_basic(self):
        """Test FeatureValidationError without room_id."""
        error = FeatureValidationError("temperature", "value out of range", -50)

        assert "temperature" in str(error)
        assert "value out of range" in str(error)
        assert error.error_code == "FEATURE_VALIDATION_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["feature_name"] == "temperature"
        assert error.context["validation_error"] == "value out of range"
        assert (
            error.context["validation_rule"] == "value out of range"
        )  # Backward compat
        assert error.context["actual_value"] == -50

    def test_feature_validation_error_with_room(self):
        """Test FeatureValidationError with room_id."""
        cause = ValueError("Range error")

        error = FeatureValidationError(
            "humidity", "must be 0-100", 150, room_id="bathroom", cause=cause
        )

        assert "bathroom" in str(error)
        assert error.context["room_id"] == "bathroom"
        assert error.cause == cause

    def test_feature_store_error(self):
        """Test FeatureStoreError."""
        cause = IOError("Disk full")

        error = FeatureStoreError("save_features", "temporal", cause)

        assert "save_features" in str(error)
        assert "temporal" in str(error)
        assert error.error_code == "FEATURE_STORE_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["operation"] == "save_features"
        assert error.context["feature_group"] == "temporal"
        assert error.cause == cause


class TestModelErrors:
    """Test machine learning model error classes."""

    def test_model_error_base(self):
        """Test ModelError base class."""
        error = ModelError("Model error")

        assert isinstance(error, OccupancyPredictionError)

    def test_model_training_error_basic(self):
        """Test ModelTrainingError with basic parameters."""
        error = ModelTrainingError("lstm", "kitchen")

        assert "lstm" in str(error)
        assert "kitchen" in str(error)
        assert error.error_code == "MODEL_TRAINING_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["model_type"] == "lstm"
        assert error.context["room_id"] == "kitchen"

    def test_model_training_error_with_data_size(self):
        """Test ModelTrainingError with training data size."""
        cause = ValueError("Invalid data")

        error = ModelTrainingError(
            "xgboost", "living_room", training_data_size=500, cause=cause
        )

        assert error.context["training_data_size"] == 500
        assert error.cause == cause

    def test_model_prediction_error_basic(self):
        """Test ModelPredictionError with basic parameters."""
        error = ModelPredictionError("random_forest", "bedroom")

        assert "random_forest" in str(error)
        assert "bedroom" in str(error)
        assert error.error_code == "MODEL_PREDICTION_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["model_type"] == "random_forest"
        assert error.context["room_id"] == "bedroom"

    def test_model_prediction_error_with_feature_shape(self):
        """Test ModelPredictionError with feature shape."""
        cause = RuntimeError("Shape mismatch")

        error = ModelPredictionError(
            "svm", "office", feature_shape=(100, 50), cause=cause
        )

        assert error.context["feature_shape"] == (100, 50)
        assert error.cause == cause

    def test_prediction_error_alias(self):
        """Test that PredictionError is an alias for ModelPredictionError."""
        assert PredictionError is ModelPredictionError

    def test_insufficient_training_data_error_basic(self):
        """Test InsufficientTrainingDataError with basic room_id."""
        error = InsufficientTrainingDataError("garage")

        assert "garage" in str(error)
        assert error.error_code == "INSUFFICIENT_TRAINING_DATA_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["room_id"] == "garage"

    def test_insufficient_training_data_error_data_points(self):
        """Test InsufficientTrainingDataError with data points."""
        error = InsufficientTrainingDataError(
            "basement", data_points=10, minimum_required=100, time_span_days=7.5
        )

        expected_msg = "Insufficient training data for room basement: have 10, need 100"
        assert error.message == expected_msg
        assert error.context["data_points"] == 10
        assert error.context["minimum_required"] == 100
        assert error.context["time_span_days"] == 7.5

    def test_insufficient_training_data_error_with_samples(self):
        """Test InsufficientTrainingDataError with required/available samples."""
        error = InsufficientTrainingDataError(
            "attic", required_samples=500, available_samples=25
        )

        expected_msg = "Insufficient training data for room attic: need 500, have 25"
        assert error.message == expected_msg
        assert error.context["required_samples"] == 500
        assert error.context["available_samples"] == 25

    def test_insufficient_training_data_error_with_model_type(self):
        """Test InsufficientTrainingDataError with model type."""
        error = InsufficientTrainingDataError("pantry", model_type="lstm")

        expected_msg = "Insufficient training data for lstm in room pantry"
        assert error.message == expected_msg
        assert error.context["model_type"] == "lstm"

    def test_insufficient_training_data_error_all_params(self):
        """Test InsufficientTrainingDataError with all parameters."""
        cause = Exception("Data corrupted")

        error = InsufficientTrainingDataError(
            "study",
            data_points=5,
            minimum_required=50,
            time_span_days=30,
            model_type="gru",
            required_samples=100,
            available_samples=5,
            cause=cause,
        )

        assert error.context["room_id"] == "study"
        assert error.context["model_type"] == "gru"
        assert error.context["data_points"] == 5
        assert error.context["minimum_required"] == 50
        assert error.context["time_span_days"] == 30
        assert error.context["required_samples"] == 100
        assert error.context["available_samples"] == 5
        assert error.cause == cause

    def test_model_not_found_error_basic(self):
        """Test ModelNotFoundError without model path."""
        error = ModelNotFoundError("transformer", "den")

        assert "transformer" in str(error)
        assert "den" in str(error)
        assert error.error_code == "MODEL_NOT_FOUND"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["model_type"] == "transformer"
        assert error.context["room_id"] == "den"

    def test_model_not_found_error_with_path(self):
        """Test ModelNotFoundError with model path."""
        error = ModelNotFoundError(
            "cnn", "porch", model_path="/models/cnn_porch_v1.pkl"
        )

        assert error.context["model_path"] == "/models/cnn_porch_v1.pkl"

    def test_model_version_mismatch_error(self):
        """Test ModelVersionMismatchError."""
        cause = ValueError("Version incompatible")

        error = ModelVersionMismatchError("bert", "sunroom", "v2.0", "v1.5", cause)

        assert "bert" in str(error)
        assert "sunroom" in str(error)
        assert "v2.0" in str(error)
        assert "v1.5" in str(error)
        assert error.error_code == "MODEL_VERSION_MISMATCH_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["model_type"] == "bert"
        assert error.context["room_id"] == "sunroom"
        assert error.context["expected_version"] == "v2.0"
        assert error.context["actual_version"] == "v1.5"
        assert error.cause == cause

    def test_missing_feature_error_basic(self):
        """Test MissingFeatureError with basic parameters."""
        missing_features = ["temperature", "humidity"]

        error = MissingFeatureError(missing_features, "greenhouse")

        assert "temperature, humidity" in str(error)
        assert "greenhouse" in str(error)
        assert "prediction" in str(error)
        assert error.error_code == "MISSING_FEATURE_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["feature_names"] == missing_features
        assert error.context["missing_features"] == missing_features  # Backward compat
        assert error.context["room_id"] == "greenhouse"
        assert error.context["operation"] == "prediction"

    def test_missing_feature_error_with_available_features(self):
        """Test MissingFeatureError with available features."""
        missing = ["wind_speed", "pressure"]
        available = ["temperature", "humidity", "light"]

        error = MissingFeatureError(
            missing,
            "weather_station",
            available_features=available,
            operation="training",
        )

        assert error.context["available_features"] == available
        assert error.context["operation"] == "training"
        assert "training" in str(error)

    def test_model_validation_error(self):
        """Test ModelValidationError."""
        cause = RuntimeError("Validation failed")

        error = ModelValidationError(
            "autoencoder", "lab", "accuracy below threshold", cause
        )

        assert "autoencoder" in str(error)
        assert "lab" in str(error)
        assert "accuracy below threshold" in str(error)
        assert error.error_code == "MODEL_VALIDATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["model_type"] == "autoencoder"
        assert error.context["room_id"] == "lab"
        assert error.context["validation_error"] == "accuracy below threshold"
        assert error.cause == cause


class TestDataProcessingErrors:
    """Test data processing error classes."""

    def test_data_processing_error_base(self):
        """Test DataProcessingError base class."""
        error = DataProcessingError("Data processing error")

        assert isinstance(error, OccupancyPredictionError)

    def test_data_corruption_error(self):
        """Test DataCorruptionError."""
        error = DataCorruptionError("sensor_readings.csv", "checksums do not match")

        assert "sensor_readings.csv" in str(error)
        assert "checksums do not match" in str(error)
        assert error.error_code == "DATA_CORRUPTION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["data_source"] == "sensor_readings.csv"
        assert error.context["corruption_details"] == "checksums do not match"


class TestIntegrationErrors:
    """Test integration-related error classes."""

    def test_integration_error_base(self):
        """Test IntegrationError base class."""
        error = IntegrationError("Integration error")

        assert isinstance(error, OccupancyPredictionError)

    def test_data_validation_error_primary_signature(self):
        """Test DataValidationError with primary signature (data_source, validation_errors)."""
        validation_errors = ["Field 'name' is required", "Field 'age' must be positive"]
        sample_data = {"name": None, "age": -5}

        error = DataValidationError(
            "user_input", validation_errors, sample_data=sample_data
        )

        assert "user_input" in str(error)
        assert "Field 'name' is required" in str(error)
        assert "Field 'age' must be positive" in str(error)
        assert error.error_code == "DATA_VALIDATION_ERROR"
        assert error.context["data_source"] == "user_input"
        assert error.context["validation_errors"] == validation_errors
        assert error.context["sample_data"] == sample_data

    def test_data_validation_error_legacy_signature(self):
        """Test DataValidationError with legacy signature (data_type, validation_rule)."""
        error = DataValidationError(
            data_source="input",  # Required first param, but using legacy params
            validation_errors=[],  # Empty to trigger legacy path
            data_type="temperature",
            validation_rule="must be between -40 and 60",
            actual_value=100,
            expected_value="valid temperature",
            field_name="temp_celsius",
            severity=ErrorSeverity.HIGH,
        )

        assert "temperature" in str(error)
        assert "must be between -40 and 60" in str(error)
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["data_type"] == "temperature"
        assert error.context["validation_rule"] == "must be between -40 and 60"
        assert "100" in error.context["actual_value"]  # Converted to string
        assert error.context["expected_value"] == "valid temperature"
        assert error.context["field_name"] == "temp_celsius"

    def test_data_validation_error_long_values(self):
        """Test DataValidationError with long actual/expected values (should be truncated)."""
        long_value = "x" * 150

        error = DataValidationError(
            data_source="input",
            validation_errors=[],  # Trigger legacy path
            data_type="text",
            validation_rule="length check",
            actual_value=long_value,
            expected_value=long_value,
        )

        # Values should be truncated to 100 characters
        assert len(error.context["actual_value"]) == 100
        assert len(error.context["expected_value"]) == 100

    def test_data_validation_error_fallback(self):
        """Test DataValidationError fallback when no specific validation info."""
        error = DataValidationError("api_response", [])

        expected_msg = "Data validation failed for api_response"
        assert error.message == expected_msg
        assert error.context["data_source"] == "api_response"

    def test_mqtt_error_base(self):
        """Test MQTTError base class."""
        error = MQTTError("MQTT error")

        assert isinstance(error, IntegrationError)

    def test_mqtt_connection_error_basic(self):
        """Test MQTTConnectionError without username."""
        cause = ConnectionError("Broker unreachable")

        error = MQTTConnectionError("mqtt.broker.com", 1883, cause=cause)

        assert "mqtt.broker.com:1883" in str(error)
        assert error.error_code == "MQTT_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["broker"] == "mqtt.broker.com"
        assert error.context["port"] == 1883
        assert "username" not in error.context
        assert error.cause == cause

    def test_mqtt_connection_error_with_username(self):
        """Test MQTTConnectionError with username."""
        error = MQTTConnectionError(
            "secure.mqtt.broker.com", 8883, username="mqtt_user"
        )

        assert error.context["username"] == "mqtt_user"

    def test_mqtt_publish_error_basic(self):
        """Test MQTTPublishError with basic parameters."""
        error = MQTTPublishError("sensor/temperature")

        assert "sensor/temperature" in str(error)
        assert error.error_code == "MQTT_PUBLISH_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["topic"] == "sensor/temperature"

    def test_mqtt_publish_error_full_params(self):
        """Test MQTTPublishError with all parameters."""
        cause = TimeoutError("Publish timeout")

        error = MQTTPublishError(
            "home/living_room/occupancy",
            payload_size=256,
            qos=2,
            broker="homebroker.local",
            cause=cause,
        )

        assert "home/living_room/occupancy" in str(error)
        assert "homebroker.local" in str(error)
        assert error.context["topic"] == "home/living_room/occupancy"
        assert error.context["payload_size"] == 256
        assert error.context["qos"] == 2
        assert error.context["broker"] == "homebroker.local"
        assert error.cause == cause

    def test_mqtt_subscription_error_basic(self):
        """Test MQTTSubscriptionError without broker."""
        error = MQTTSubscriptionError("sensor/+/state")

        assert "sensor/+/state" in str(error)
        assert error.error_code == "MQTT_SUBSCRIPTION_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["topic_pattern"] == "sensor/+/state"

    def test_mqtt_subscription_error_with_broker(self):
        """Test MQTTSubscriptionError with broker."""
        cause = Exception("Subscription failed")

        error = MQTTSubscriptionError("home/+/+", broker="mqtt.home.local", cause=cause)

        assert "mqtt.home.local" in str(error)
        assert error.context["broker"] == "mqtt.home.local"
        assert error.cause == cause

    def test_api_server_error(self):
        """Test APIServerError."""
        cause = RuntimeError("Server overloaded")

        error = APIServerError("/api/predictions", "GET request processing", cause)

        assert "/api/predictions" in str(error)
        assert "GET request processing" in str(error)
        assert error.error_code == "API_SERVER_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["endpoint"] == "/api/predictions"
        assert error.context["operation"] == "GET request processing"
        assert error.cause == cause


class TestSystemErrors:
    """Test system-wide error classes."""

    def test_system_initialization_error(self):
        """Test SystemInitializationError."""
        cause = ImportError("Module not found")

        error = SystemInitializationError("database_connection", cause)

        assert "database_connection" in str(error)
        assert error.error_code == "SYSTEM_INIT_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["component"] == "database_connection"
        assert error.cause == cause

    def test_system_resource_error(self):
        """Test SystemResourceError."""
        cause = OSError("Permission denied")

        error = SystemResourceError("file", "/var/log/app.log", cause)

        assert "file" in str(error)
        assert "/var/log/app.log" in str(error)
        assert error.error_code == "SYSTEM_RESOURCE_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["resource_type"] == "file"
        assert error.context["resource_name"] == "/var/log/app.log"
        assert error.cause == cause

    def test_system_error_basic(self):
        """Test SystemError with basic message."""
        error = SystemError("Something went wrong")

        assert str(error) == "Something went wrong"
        assert error.error_code == "SYSTEM_ERROR"
        assert error.severity == ErrorSeverity.HIGH

    def test_system_error_with_operation(self):
        """Test SystemError with operation."""
        error = SystemError("Failed to process", operation="data_processing")

        expected_msg = "System error during data_processing: Failed to process"
        assert str(error) == expected_msg
        assert error.context["operation"] == "data_processing"

    def test_system_error_with_component_and_operation(self):
        """Test SystemError with both component and operation."""
        cause = ValueError("Invalid input")

        error = SystemError(
            "Validation failed",
            operation="user_input_validation",
            component="input_validator",
            cause=cause,
        )

        expected_msg = "System error in input_validator during user_input_validation: Validation failed"
        assert str(error) == expected_msg
        assert error.context["component"] == "input_validator"
        assert error.context["operation"] == "user_input_validation"
        assert error.cause == cause

    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError."""
        error = ResourceExhaustionError("memory", 95.5, 90.0, "%")

        expected_msg = "Resource exhaustion: memory at 95.5% (limit: 90.0%)"
        assert str(error) == expected_msg
        assert error.error_code == "RESOURCE_EXHAUSTION_ERROR"
        assert error.context["resource_type"] == "memory"
        assert error.context["current_usage"] == 95.5
        assert error.context["limit"] == 90.0
        assert error.context["unit"] == "%"
        assert error.context["operation"] == "resource_monitoring"
        assert error.context["component"] == "memory"

    def test_service_unavailable_error_basic(self):
        """Test ServiceUnavailableError with basic parameters."""
        error = ServiceUnavailableError("database")

        assert "database" in str(error)
        assert error.error_code == "SERVICE_UNAVAILABLE_ERROR"
        assert error.context["service_name"] == "database"
        assert error.context["operation"] == "service_access"
        assert error.context["component"] == "database"

    def test_service_unavailable_error_full_params(self):
        """Test ServiceUnavailableError with all parameters."""
        error = ServiceUnavailableError(
            "api_gateway",
            endpoint="https://api.example.com/v1",
            retry_after=300,
            reason="Maintenance in progress",
        )

        assert "api_gateway" in str(error)
        assert "Maintenance in progress" in str(error)
        assert error.context["service_name"] == "api_gateway"
        assert error.context["endpoint"] == "https://api.example.com/v1"
        assert error.context["retry_after"] == 300
        assert error.context["reason"] == "Maintenance in progress"

    def test_maintenance_mode_error_basic(self):
        """Test MaintenanceModeError without component."""
        error = MaintenanceModeError()

        expected_msg = "System in maintenance mode"
        assert str(error) == expected_msg
        assert error.error_code == "MAINTENANCE_MODE_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["operation"] == "maintenance"
        assert error.context["component"] == "system"

    def test_maintenance_mode_error_with_component(self):
        """Test MaintenanceModeError with component."""
        error = MaintenanceModeError("database")

        expected_msg = "System component in maintenance mode: database"
        assert str(error) == expected_msg
        assert error.context["component"] == "database"

    def test_maintenance_mode_error_with_end_time(self):
        """Test MaintenanceModeError with end time."""
        error = MaintenanceModeError("search_engine", end_time="2024-01-15 14:00 UTC")

        expected_msg = "System component in maintenance mode: search_engine (until 2024-01-15 14:00 UTC)"
        assert str(error) == expected_msg
        assert error.context["estimated_end_time"] == "2024-01-15 14:00 UTC"

    def test_maintenance_mode_error_backwards_compatibility(self):
        """Test MaintenanceModeError with legacy maintenance_until parameter."""
        error = MaintenanceModeError("cache", maintenance_until="2024-01-15 16:00 UTC")

        # Should use maintenance_until value
        assert "2024-01-15 16:00 UTC" in str(error)
        assert error.context["estimated_end_time"] == "2024-01-15 16:00 UTC"


class TestAPIErrors:
    """Test API-related error classes."""

    def test_api_error_base(self):
        """Test APIError base class."""
        error = APIError("API error")

        assert isinstance(error, OccupancyPredictionError)

    def test_api_authentication_error_basic(self):
        """Test APIAuthenticationError with basic parameters."""
        error = APIAuthenticationError()

        assert "Authentication failed" in str(error)
        assert error.error_code == "API_AUTH_ERROR"
        assert error.severity == ErrorSeverity.HIGH

    def test_api_authentication_error_full_params(self):
        """Test APIAuthenticationError with all parameters."""
        context = {"user_id": 123, "ip": "192.168.1.100"}

        error = APIAuthenticationError(
            message="Invalid token",
            endpoint="/api/secure",
            auth_method="bearer",
            context=context,
        )

        assert "Invalid token" in str(error)
        assert error.context["endpoint"] == "/api/secure"
        assert error.context["auth_method"] == "bearer"
        assert error.context["user_id"] == 123
        assert error.context["ip"] == "192.168.1.100"

    def test_rate_limit_exceeded_error_basic(self):
        """Test RateLimitExceededError with basic parameters."""
        error = RateLimitExceededError("api", 100, 3600)

        expected_msg = "Rate limit exceeded for api: 100 requests per 3600s"
        assert str(error) == expected_msg
        assert error.error_code == "RATE_LIMIT_EXCEEDED_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["service"] == "api"
        assert error.context["limit"] == 100
        assert error.context["window_seconds"] == 3600

    def test_rate_limit_exceeded_error_with_reset_time(self):
        """Test RateLimitExceededError with reset time."""
        context = {"client_id": "app_123"}

        error = RateLimitExceededError(
            "upload_service",
            50,
            60,
            reset_time=1642694400,  # Unix timestamp
            message="Upload limit exceeded",
            context=context,
        )

        assert "Upload limit exceeded" in str(error)
        assert error.context["reset_time"] == 1642694400
        assert error.context["client_id"] == "app_123"

    def test_api_rate_limit_error_alias(self):
        """Test that APIRateLimitError is an alias for RateLimitExceededError."""
        assert APIRateLimitError is RateLimitExceededError

    def test_api_authorization_error_basic(self):
        """Test APIAuthorizationError with basic parameters."""
        error = APIAuthorizationError()

        assert "Authorization failed" in str(error)
        assert error.error_code == "API_AUTHORIZATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH

    def test_api_authorization_error_full_params(self):
        """Test APIAuthorizationError with all parameters."""
        context = {"role": "user", "required_role": "admin"}

        error = APIAuthorizationError(
            message="Insufficient permissions",
            endpoint="/api/admin/users",
            required_permission="user:manage",
            context=context,
        )

        assert "Insufficient permissions" in str(error)
        assert error.context["endpoint"] == "/api/admin/users"
        assert error.context["required_permission"] == "user:manage"
        assert error.context["role"] == "user"
        assert error.context["required_role"] == "admin"

    def test_api_security_error_basic(self):
        """Test APISecurityError with basic parameters."""
        error = APISecurityError()

        assert "Security violation detected" in str(error)
        assert error.error_code == "API_SECURITY_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL

    def test_api_security_error_full_params(self):
        """Test APISecurityError with all parameters."""
        context = {"source_ip": "10.0.0.1", "user_agent": "suspicious"}

        error = APISecurityError(
            message="SQL injection attempt",
            violation_type="injection",
            endpoint="/api/search",
            context=context,
        )

        assert "SQL injection attempt" in str(error)
        assert error.context["violation_type"] == "injection"
        assert error.context["endpoint"] == "/api/search"
        assert error.context["source_ip"] == "10.0.0.1"
        assert error.context["user_agent"] == "suspicious"

    def test_api_resource_not_found_error_basic(self):
        """Test APIResourceNotFoundError with basic parameters."""
        error = APIResourceNotFoundError("User", "12345")

        assert "User with ID '12345' not found" in str(error)
        assert error.error_code == "API_RESOURCE_NOT_FOUND"
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.context["resource_type"] == "User"
        assert error.context["resource_id"] == "12345"

    def test_api_resource_not_found_error_with_endpoint(self):
        """Test APIResourceNotFoundError with endpoint."""
        context = {"query": "name=john", "filters": ["active"]}

        error = APIResourceNotFoundError(
            "Product", "SKU-789", endpoint="/api/products/SKU-789", context=context
        )

        assert error.context["endpoint"] == "/api/products/SKU-789"
        assert error.context["query"] == "name=john"
        assert error.context["filters"] == ["active"]


class TestValidationFunctions:
    """Test validation helper functions."""

    def test_validate_room_id_valid(self):
        """Test validate_room_id with valid room IDs."""
        valid_ids = [
            "living_room",
            "bedroom1",
            "kitchen-area",
            "Room_123",
            "a",
            "123",
            "test-room_1",
        ]

        for room_id in valid_ids:
            # Should not raise exception
            validate_room_id(room_id)

    def test_validate_room_id_invalid_empty(self):
        """Test validate_room_id with empty string."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_room_id("")

        assert "non-empty string" in str(exc_info.value)
        assert exc_info.value.context["data_type"] == "room_id"
        assert exc_info.value.context["validation_rule"] == "must be non-empty string"

    def test_validate_room_id_invalid_none(self):
        """Test validate_room_id with None."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_room_id(None)

        assert "non-empty string" in str(exc_info.value)
        assert exc_info.value.context["actual_value"] is None

    def test_validate_room_id_invalid_type(self):
        """Test validate_room_id with invalid type."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_room_id(123)

        assert "non-empty string" in str(exc_info.value)
        assert exc_info.value.context["actual_value"] == 123

    def test_validate_room_id_invalid_characters(self):
        """Test validate_room_id with invalid characters."""
        invalid_ids = [
            "room with spaces",
            "room@home",
            "room#1",
            "room/area",
            "room\\area",
            "room+area",
            "room=area",
            "room?area",
            "room.area",
        ]

        for room_id in invalid_ids:
            with pytest.raises(DataValidationError) as exc_info:
                validate_room_id(room_id)

            assert "alphanumeric characters" in str(exc_info.value)
            assert exc_info.value.context["actual_value"] == room_id

    def test_validate_entity_id_valid(self):
        """Test validate_entity_id with valid entity IDs."""
        valid_ids = [
            "sensor.temperature",
            "binary_sensor.motion_1",
            "switch.light_main",
            "climate.thermostat_bedroom",
            "light.living_room_lamp",
            "cover.garage_door",
            "lock.front_door_lock",
            "camera.security_cam_1",
        ]

        for entity_id in valid_ids:
            # Should not raise exception
            validate_entity_id(entity_id)

    def test_validate_entity_id_invalid_empty(self):
        """Test validate_entity_id with empty string."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_entity_id("")

        assert "non-empty string" in str(exc_info.value)
        assert exc_info.value.context["data_type"] == "entity_id"

    def test_validate_entity_id_invalid_none(self):
        """Test validate_entity_id with None."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_entity_id(None)

        assert exc_info.value.context["actual_value"] is None

    def test_validate_entity_id_invalid_type(self):
        """Test validate_entity_id with invalid type."""
        with pytest.raises(DataValidationError) as exc_info:
            validate_entity_id(["sensor", "temperature"])

        assert exc_info.value.context["actual_value"] == ["sensor", "temperature"]

    def test_validate_entity_id_invalid_format(self):
        """Test validate_entity_id with invalid formats."""
        invalid_ids = [
            "sensor",  # No object_id
            "sensor.",  # Empty object_id
            ".temperature",  # Empty domain
            "sensor.temperature.extra",  # Too many dots
            "SENSOR.temperature",  # Uppercase domain
            "sensor.Temperature",  # Uppercase in object_id
            "sensor-type.temperature",  # Invalid domain character
            "sensor.temp@home",  # Invalid object_id character
            "123sensor.temperature",  # Domain starts with number
            "sensor.123temperature",  # Object_id starts with number (this is actually valid)
            "sensor.temp-sensor",  # Hyphen in object_id (this is actually valid)
        ]

        invalid_cases = [
            "sensor",
            "sensor.",
            ".temperature",
            "sensor.temperature.extra",
            "SENSOR.temperature",
            "sensor.Temperature",
            "sensor-type.temperature",
            "sensor.temp@home",
            "123sensor.temperature",
        ]

        for entity_id in invalid_cases:
            with pytest.raises(DataValidationError) as exc_info:
                validate_entity_id(entity_id)

            assert "Home Assistant format" in str(exc_info.value)
            assert exc_info.value.context["actual_value"] == entity_id
            assert (
                exc_info.value.context["expected_value"] == "sensor.living_room_motion"
            )

    def test_validate_entity_id_valid_edge_cases(self):
        """Test validate_entity_id with valid edge cases."""
        valid_edge_cases = [
            "sensor.temp_1",  # Underscore and number
            "binary_sensor.motion_sensor_01",  # Multiple underscores and numbers
            "sensor.temperature123",  # Ending with numbers
            "a.b",  # Single character domain and object_id
        ]

        for entity_id in valid_edge_cases:
            # Should not raise exception
            validate_entity_id(entity_id)


class TestExceptionInheritanceAndCompatibility:
    """Test exception inheritance and backward compatibility."""

    def test_exception_inheritance_hierarchy(self):
        """Test that all exceptions properly inherit from base classes."""
        # Test inheritance chains
        assert issubclass(ConfigurationError, OccupancyPredictionError)
        assert issubclass(ConfigFileNotFoundError, ConfigurationError)
        assert issubclass(HomeAssistantError, OccupancyPredictionError)
        assert issubclass(WebSocketError, HomeAssistantError)
        assert issubclass(DatabaseError, OccupancyPredictionError)
        assert issubclass(FeatureEngineeringError, OccupancyPredictionError)
        assert issubclass(ModelError, OccupancyPredictionError)
        assert issubclass(DataProcessingError, OccupancyPredictionError)
        assert issubclass(IntegrationError, OccupancyPredictionError)
        assert issubclass(MQTTError, IntegrationError)
        assert issubclass(APIError, OccupancyPredictionError)

    def test_all_exceptions_are_exceptions(self):
        """Test that all custom exceptions inherit from Exception."""
        exception_classes = [
            OccupancyPredictionError,
            ConfigurationError,
            ConfigFileNotFoundError,
            ConfigValidationError,
            HomeAssistantError,
            HomeAssistantConnectionError,
            DatabaseError,
            DatabaseConnectionError,
            FeatureEngineeringError,
            FeatureExtractionError,
            ModelError,
            ModelTrainingError,
            DataProcessingError,
            IntegrationError,
            DataValidationError,
            MQTTError,
            SystemError,
            APIError,
        ]

        for exc_class in exception_classes:
            assert issubclass(exc_class, Exception)
            # Can be instantiated
            instance = exc_class("test message")
            assert isinstance(instance, Exception)

    def test_error_context_preservation(self):
        """Test that error context is preserved through inheritance."""
        context = {"test_key": "test_value", "number": 42}

        # Test various exception types preserve context
        config_error = ConfigValidationError("Test error", context=context)
        assert config_error.context["test_key"] == "test_value"
        assert config_error.context["number"] == 42

        db_error = DatabaseQueryError("SELECT 1", parameters={"id": 1})
        assert db_error.context["parameters"] == {"id": 1}

    def test_backward_compatibility_aliases(self):
        """Test that backward compatibility aliases work."""
        # PredictionError should be ModelPredictionError
        assert PredictionError is ModelPredictionError

        # APIRateLimitError should be RateLimitExceededError
        assert APIRateLimitError is RateLimitExceededError

        # Can create instances using aliases
        pred_error = PredictionError("lstm", "room1")
        assert isinstance(pred_error, ModelPredictionError)

        rate_error = APIRateLimitError("api", 100, 3600)
        assert isinstance(rate_error, RateLimitExceededError)

    def test_error_serialization_compatibility(self):
        """Test that errors can be converted to dict-like structures."""
        error = ModelTrainingError(
            "xgboost", "bedroom", training_data_size=100, cause=ValueError("Data issue")
        )

        # Should be able to access all attributes
        assert hasattr(error, "message")
        assert hasattr(error, "error_code")
        assert hasattr(error, "context")
        assert hasattr(error, "severity")
        assert hasattr(error, "cause")

        # Context should be dict-like
        assert isinstance(error.context, dict)
        assert error.context["model_type"] == "xgboost"
        assert error.context["room_id"] == "bedroom"
