"""
Unit tests for custom exception classes.

Tests all custom exception classes for proper inheritance, error handling,
context management, and error message formatting.
"""

import pytest
from unittest.mock import Mock

from src.core.exceptions import (
    # Base exception
    ErrorSeverity, OccupancyPredictionError,
    
    # Configuration errors
    ConfigurationError, ConfigFileNotFoundError, 
    ConfigValidationError, ConfigParsingError,
    
    # Home Assistant errors
    HomeAssistantError, HomeAssistantConnectionError,
    HomeAssistantAuthenticationError, HomeAssistantAPIError,
    EntityNotFoundError, WebSocketError,
    
    # Database errors
    DatabaseError, DatabaseConnectionError, DatabaseQueryError,
    DatabaseMigrationError, DatabaseIntegrityError,
    
    # Model errors
    ModelError, ModelTrainingError, ModelPredictionError,
    ModelNotFoundError, InsufficientTrainingDataError,
    ModelVersionMismatchError,
    
    # Feature engineering errors
    FeatureEngineeringError, FeatureExtractionError,
    FeatureValidationError, MissingFeatureError,
    FeatureStoreError,
    
    # MQTT and Integration errors
    MQTTError, MQTTConnectionError, MQTTPublishError,
    MQTTSubscriptionError, IntegrationError, DataValidationError,
    RateLimitExceededError,
    
    # System errors
    SystemError, ResourceExhaustionError, ServiceUnavailableError,
    MaintenanceModeError
)


class TestErrorSeverity:
    """Test ErrorSeverity enum."""
    
    def test_error_severity_values(self):
        """Test that error severity levels have correct values."""
        assert ErrorSeverity.LOW.value == "low"
        assert ErrorSeverity.MEDIUM.value == "medium"
        assert ErrorSeverity.HIGH.value == "high"
        assert ErrorSeverity.CRITICAL.value == "critical"
    
    def test_error_severity_count(self):
        """Test that we have the expected number of severity levels."""
        assert len(ErrorSeverity) == 4


class TestOccupancyPredictionError:
    """Test base OccupancyPredictionError class."""
    
    def test_basic_exception_creation(self):
        """Test creating basic exception with message only."""
        error = OccupancyPredictionError("Test error message")
        
        assert str(error) == "Test error message"
        assert error.message == "Test error message"
        assert error.error_code is None
        assert error.context == {}
        assert error.severity == ErrorSeverity.MEDIUM
        assert error.cause is None
    
    def test_exception_with_all_parameters(self):
        """Test creating exception with all parameters."""
        context = {"key1": "value1", "key2": 42}
        cause = ValueError("Original error")
        
        error = OccupancyPredictionError(
            message="Test error",
            error_code="TEST_ERROR_001",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause
        )
        
        assert error.message == "Test error"
        assert error.error_code == "TEST_ERROR_001"
        assert error.context == context
        assert error.severity == ErrorSeverity.HIGH
        assert error.cause == cause
    
    def test_exception_string_representation(self):
        """Test string representation with all components."""
        context = {"user": "test", "action": "login"}
        cause = ValueError("Database connection failed")
        
        error = OccupancyPredictionError(
            message="Authentication failed",
            error_code="AUTH_001",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause
        )
        
        error_str = str(error)
        assert "Authentication failed" in error_str
        assert "Error Code: AUTH_001" in error_str
        assert "Context: user=test, action=login" in error_str
        assert "Caused by: ValueError: Database connection failed" in error_str
    
    def test_exception_string_representation_minimal(self):
        """Test string representation with minimal information."""
        error = OccupancyPredictionError("Simple error")
        assert str(error) == "Simple error"
    
    def test_exception_inheritance(self):
        """Test that OccupancyPredictionError inherits from Exception."""
        error = OccupancyPredictionError("Test error")
        assert isinstance(error, Exception)


class TestConfigurationErrors:
    """Test configuration-related exception classes."""
    
    def test_configuration_error_basic(self):
        """Test basic ConfigurationError."""
        error = ConfigurationError("Config error")
        assert isinstance(error, OccupancyPredictionError)
        assert error.message == "Config error"
    
    def test_configuration_error_with_config_file(self):
        """Test ConfigurationError with config file parameter."""
        error = ConfigurationError("Config error", config_file="test.yaml")
        assert error.context["config_file"] == "test.yaml"
    
    def test_config_file_not_found_error(self):
        """Test ConfigFileNotFoundError."""
        error = ConfigFileNotFoundError("config.yaml", "/path/to/config")
        
        assert isinstance(error, ConfigurationError)
        assert "config.yaml" in error.message
        assert "/path/to/config" in error.message
        assert error.error_code == "CONFIG_FILE_NOT_FOUND"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.context["config_file"] == "config.yaml"
        assert error.context["config_dir"] == "/path/to/config"
    
    def test_config_validation_error(self):
        """Test ConfigValidationError."""
        error = ConfigValidationError(
            field="database.host",
            value=None,
            expected="string",
            config_file="config.yaml"
        )
        
        assert isinstance(error, ConfigurationError)
        assert "database.host" in error.message
        assert "None" in error.message
        assert "string" in error.message
        assert error.error_code == "CONFIG_VALIDATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["field"] == "database.host"
        assert error.context["value"] is None
        assert error.context["expected"] == "string"
    
    def test_config_parsing_error(self):
        """Test ConfigParsingError."""
        error = ConfigParsingError("config.yaml", "Invalid YAML syntax")
        
        assert isinstance(error, ConfigurationError)
        assert "config.yaml" in error.message
        assert "Invalid YAML syntax" in error.message
        assert error.error_code == "CONFIG_PARSING_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["config_file"] == "config.yaml"
        assert error.context["parse_error"] == "Invalid YAML syntax"


class TestHomeAssistantErrors:
    """Test Home Assistant-related exception classes."""
    
    def test_home_assistant_error_base(self):
        """Test base HomeAssistantError."""
        error = HomeAssistantError("HA error")
        assert isinstance(error, OccupancyPredictionError)
    
    def test_home_assistant_connection_error(self):
        """Test HomeAssistantConnectionError."""
        cause = ConnectionError("Network unreachable")
        error = HomeAssistantConnectionError("http://ha:8123", cause=cause)
        
        assert isinstance(error, HomeAssistantError)
        assert "http://ha:8123" in error.message
        assert error.error_code == "HA_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["url"] == "http://ha:8123"
        assert error.cause == cause
    
    def test_home_assistant_authentication_error(self):
        """Test HomeAssistantAuthenticationError."""
        error = HomeAssistantAuthenticationError("http://ha:8123", 64)
        
        assert isinstance(error, HomeAssistantError)
        assert "http://ha:8123" in error.message
        assert error.error_code == "HA_AUTH_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["url"] == "http://ha:8123"
        assert error.context["token_length"] == 64
        assert "Check if token is valid" in error.context["hint"]
    
    def test_home_assistant_api_error(self):
        """Test HomeAssistantAPIError."""
        error = HomeAssistantAPIError("/api/states", 404, "Not Found", "GET")
        
        assert isinstance(error, HomeAssistantError)
        assert "/api/states" in error.message
        assert "404" in error.message
        assert error.error_code == "HA_API_ERROR"
        assert error.context["endpoint"] == "/api/states"
        assert error.context["method"] == "GET"
        assert error.context["status_code"] == 404
        assert error.context["response"] == "Not Found"
    
    def test_entity_not_found_error(self):
        """Test EntityNotFoundError."""
        error = EntityNotFoundError("binary_sensor.test", "living_room")
        
        assert isinstance(error, HomeAssistantError)
        assert "binary_sensor.test" in error.message
        assert "living_room" in error.message
        assert error.error_code == "ENTITY_NOT_FOUND"
        assert error.context["entity_id"] == "binary_sensor.test"
        assert error.context["room_id"] == "living_room"
    
    def test_entity_not_found_error_no_room(self):
        """Test EntityNotFoundError without room."""
        error = EntityNotFoundError("binary_sensor.test")
        
        assert "binary_sensor.test" in error.message
        assert "room_id" not in error.context
    
    def test_websocket_error(self):
        """Test WebSocketError."""
        error = WebSocketError("Connection lost", "ws://ha:8123/api/websocket")
        
        assert isinstance(error, HomeAssistantError)
        assert "Connection lost" in error.message
        assert error.error_code == "HA_WEBSOCKET_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["url"] == "ws://ha:8123/api/websocket"
        assert error.context["reason"] == "Connection lost"


class TestDatabaseErrors:
    """Test database-related exception classes."""
    
    def test_database_error_base(self):
        """Test base DatabaseError."""
        error = DatabaseError("DB error")
        assert isinstance(error, OccupancyPredictionError)
    
    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        cause = ConnectionError("Connection refused")
        error = DatabaseConnectionError(
            "postgresql://user:password@localhost:5432/db",
            cause=cause
        )
        
        assert isinstance(error, DatabaseError)
        # Password should be masked
        assert "***" in error.message
        assert "password" not in error.message
        assert error.error_code == "DB_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.CRITICAL
        assert error.cause == cause
    
    def test_database_connection_error_password_masking(self):
        """Test password masking in connection strings."""
        error = DatabaseConnectionError("postgresql://user:secret123@host:5432/db")
        
        # Check that password is masked
        masked_string = error.context["connection_string"]
        assert "secret123" not in masked_string
        assert "***" in masked_string
        assert "user" in masked_string
        assert "host" in masked_string
    
    def test_database_query_error(self):
        """Test DatabaseQueryError."""
        cause = Exception("Syntax error")
        error = DatabaseQueryError(
            "SELECT * FROM table",
            parameters={"id": 123},
            cause=cause
        )
        
        assert isinstance(error, DatabaseError)
        assert "SELECT * FROM table" in error.message
        assert error.error_code == "DB_QUERY_ERROR"
        assert error.context["parameters"] == {"id": 123}
        assert error.cause == cause
    
    def test_database_migration_error(self):
        """Test DatabaseMigrationError."""
        cause = Exception("Migration failed")
        error = DatabaseMigrationError("001_initial_schema", cause=cause)
        
        assert isinstance(error, DatabaseError)
        assert "001_initial_schema" in error.message
        assert error.error_code == "DB_MIGRATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["migration_name"] == "001_initial_schema"
        assert error.cause == cause
    
    def test_database_integrity_error(self):
        """Test DatabaseIntegrityError."""
        error = DatabaseIntegrityError(
            "unique_constraint",
            "users",
            values={"email": "test@example.com"}
        )
        
        assert isinstance(error, DatabaseError)
        assert "unique_constraint" in error.message
        assert "users" in error.message
        assert error.error_code == "DB_INTEGRITY_ERROR"
        assert error.context["constraint"] == "unique_constraint"
        assert error.context["table"] == "users"
        assert error.context["values"] == {"email": "test@example.com"}


class TestModelErrors:
    """Test ML model-related exception classes."""
    
    def test_model_error_base(self):
        """Test base ModelError."""
        error = ModelError("Model error")
        assert isinstance(error, OccupancyPredictionError)
    
    def test_model_training_error(self):
        """Test ModelTrainingError."""
        cause = ValueError("Insufficient data")
        error = ModelTrainingError(
            "lstm",
            "living_room",
            cause=cause,
            training_data_size=100
        )
        
        assert isinstance(error, ModelError)
        assert "lstm" in error.message
        assert "living_room" in error.message
        assert error.error_code == "MODEL_TRAINING_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["model_type"] == "lstm"
        assert error.context["room_id"] == "living_room"
        assert error.context["training_data_size"] == 100
        assert error.cause == cause
    
    def test_model_prediction_error(self):
        """Test ModelPredictionError."""
        error = ModelPredictionError(
            "xgboost",
            "bedroom",
            feature_shape=(10, 5)
        )
        
        assert isinstance(error, ModelError)
        assert "xgboost" in error.message
        assert "bedroom" in error.message
        assert error.error_code == "MODEL_PREDICTION_ERROR"
        assert error.context["feature_shape"] == (10, 5)
    
    def test_model_not_found_error(self):
        """Test ModelNotFoundError."""
        error = ModelNotFoundError(
            "lstm",
            "office",
            model_path="/models/office_lstm.pkl"
        )
        
        assert isinstance(error, ModelError)
        assert "lstm" in error.message
        assert "office" in error.message
        assert error.error_code == "MODEL_NOT_FOUND"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["model_path"] == "/models/office_lstm.pkl"
    
    def test_insufficient_training_data_error(self):
        """Test InsufficientTrainingDataError."""
        error = InsufficientTrainingDataError(
            "kitchen",
            data_points=50,
            minimum_required=100,
            time_span_days=7.5
        )
        
        assert isinstance(error, ModelError)
        assert "kitchen" in error.message
        assert "50" in error.message
        assert "100" in error.message
        assert error.error_code == "INSUFFICIENT_TRAINING_DATA"
        assert error.context["room_id"] == "kitchen"
        assert error.context["data_points"] == 50
        assert error.context["minimum_required"] == 100
        assert error.context["time_span_days"] == 7.5
    
    def test_model_version_mismatch_error(self):
        """Test ModelVersionMismatchError."""
        error = ModelVersionMismatchError(
            "ensemble",
            "bathroom",
            "1.0",
            "2.0"
        )
        
        assert isinstance(error, ModelError)
        assert "ensemble" in error.message
        assert "bathroom" in error.message
        assert "1.0" in error.message
        assert "2.0" in error.message
        assert error.error_code == "MODEL_VERSION_MISMATCH"
        assert error.severity == ErrorSeverity.HIGH


class TestFeatureEngineeringErrors:
    """Test feature engineering-related exception classes."""
    
    def test_feature_engineering_error_base(self):
        """Test base FeatureEngineeringError."""
        error = FeatureEngineeringError("Feature error")
        assert isinstance(error, OccupancyPredictionError)
    
    def test_feature_extraction_error(self):
        """Test FeatureExtractionError."""
        cause = ValueError("Invalid data format")
        error = FeatureExtractionError(
            "temporal",
            "bedroom",
            time_range="2024-01-01 to 2024-01-07",
            cause=cause
        )
        
        assert isinstance(error, FeatureEngineeringError)
        assert "temporal" in error.message
        assert "bedroom" in error.message
        assert error.error_code == "FEATURE_EXTRACTION_ERROR"
        assert error.context["feature_type"] == "temporal"
        assert error.context["room_id"] == "bedroom"
        assert error.context["time_range"] == "2024-01-01 to 2024-01-07"
        assert error.cause == cause
    
    def test_feature_validation_error(self):
        """Test FeatureValidationError."""
        error = FeatureValidationError(
            "temperature",
            "must be between -50 and 50",
            75.5,
            room_id="office"
        )
        
        assert isinstance(error, FeatureEngineeringError)
        assert "temperature" in error.message
        assert error.error_code == "FEATURE_VALIDATION_ERROR"
        assert error.context["feature_name"] == "temperature"
        assert error.context["validation_rule"] == "must be between -50 and 50"
        assert error.context["actual_value"] == 75.5
        assert error.context["room_id"] == "office"
    
    def test_missing_feature_error(self):
        """Test MissingFeatureError."""
        missing_features = ["temperature", "humidity", "light_level"]
        available_features = ["motion", "door_state"]
        
        error = MissingFeatureError(
            missing_features,
            "living_room",
            available_features=available_features
        )
        
        assert isinstance(error, FeatureEngineeringError)
        assert "living_room" in error.message
        assert "temperature, humidity, light_level" in error.message
        assert error.error_code == "MISSING_FEATURE_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["missing_features"] == missing_features
        assert error.context["available_features"] == available_features
    
    def test_feature_store_error(self):
        """Test FeatureStoreError."""
        cause = ConnectionError("Redis unavailable")
        error = FeatureStoreError(
            "get_features",
            "temporal_features",
            cause=cause
        )
        
        assert isinstance(error, FeatureEngineeringError)
        assert "get_features" in error.message
        assert "temporal_features" in error.message
        assert error.error_code == "FEATURE_STORE_ERROR"
        assert error.context["operation"] == "get_features"
        assert error.context["feature_group"] == "temporal_features"
        assert error.cause == cause


class TestMQTTAndIntegrationErrors:
    """Test MQTT and integration-related exception classes."""
    
    def test_mqtt_error_base(self):
        """Test base MQTTError."""
        error = MQTTError("MQTT error")
        assert isinstance(error, OccupancyPredictionError)
    
    def test_mqtt_connection_error(self):
        """Test MQTTConnectionError."""
        cause = ConnectionError("Connection refused")
        error = MQTTConnectionError(
            "mqtt.example.com",
            1883,
            username="testuser",
            cause=cause
        )
        
        assert isinstance(error, MQTTError)
        assert "mqtt.example.com:1883" in error.message
        assert error.error_code == "MQTT_CONNECTION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["broker"] == "mqtt.example.com"
        assert error.context["port"] == 1883
        assert error.context["username"] == "testuser"
        assert error.cause == cause
    
    def test_mqtt_publish_error(self):
        """Test MQTTPublishError."""
        cause = Exception("Message too large")
        error = MQTTPublishError(
            "occupancy/predictions/living_room",
            1024,
            qos=1,
            cause=cause
        )
        
        assert isinstance(error, MQTTError)
        assert "occupancy/predictions/living_room" in error.message
        assert error.error_code == "MQTT_PUBLISH_ERROR"
        assert error.context["topic"] == "occupancy/predictions/living_room"
        assert error.context["payload_size"] == 1024
        assert error.context["qos"] == 1
        assert error.cause == cause
    
    def test_mqtt_subscription_error(self):
        """Test MQTTSubscriptionError."""
        error = MQTTSubscriptionError("occupancy/+/status")
        
        assert isinstance(error, MQTTError)
        assert "occupancy/+/status" in error.message
        assert error.error_code == "MQTT_SUBSCRIPTION_ERROR"
        assert error.context["topic_pattern"] == "occupancy/+/status"
    
    def test_integration_error_base(self):
        """Test base IntegrationError."""
        error = IntegrationError("Integration error")
        assert isinstance(error, OccupancyPredictionError)
    
    def test_data_validation_error(self):
        """Test DataValidationError."""
        validation_errors = ["Missing required field 'entity_id'", "Invalid timestamp format"]
        sample_data = {"state": "on", "timestamp": "invalid"}
        
        error = DataValidationError(
            "Home Assistant",
            validation_errors,
            sample_data=sample_data
        )
        
        assert isinstance(error, IntegrationError)
        assert "Home Assistant" in error.message
        assert error.error_code == "DATA_VALIDATION_ERROR"
        assert error.context["data_source"] == "Home Assistant"
        assert error.context["validation_errors"] == validation_errors
        assert error.context["sample_data"] == sample_data
    
    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError."""
        error = RateLimitExceededError(
            "Home Assistant API",
            300,
            3600,
            reset_time=1640995200
        )
        
        assert isinstance(error, IntegrationError)
        assert "Home Assistant API" in error.message
        assert "300 requests per 3600s" in error.message
        assert error.error_code == "RATE_LIMIT_EXCEEDED"
        assert error.context["service"] == "Home Assistant API"
        assert error.context["limit"] == 300
        assert error.context["window_seconds"] == 3600
        assert error.context["reset_time"] == 1640995200


class TestSystemErrors:
    """Test system-related exception classes."""
    
    def test_system_error_base(self):
        """Test base SystemError."""
        error = SystemError("System error")
        # Note: This tests our custom SystemError, not the built-in one
        assert isinstance(error, OccupancyPredictionError)
    
    def test_resource_exhaustion_error(self):
        """Test ResourceExhaustionError."""
        error = ResourceExhaustionError(
            "memory",
            current_usage=1024.5,
            limit=1000.0,
            unit="MB"
        )
        
        assert isinstance(error, SystemError)
        assert "memory" in error.message
        assert "1024.5MB" in error.message
        assert "1000.0MB" in error.message
        assert error.error_code == "RESOURCE_EXHAUSTION"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["resource_type"] == "memory"
        assert error.context["current_usage"] == 1024.5
        assert error.context["limit"] == 1000.0
        assert error.context["unit"] == "MB"
    
    def test_service_unavailable_error(self):
        """Test ServiceUnavailableError."""
        error = ServiceUnavailableError(
            "Database",
            endpoint="postgresql://localhost:5432/db",
            retry_after=300
        )
        
        assert isinstance(error, SystemError)
        assert "Database" in error.message
        assert error.error_code == "SERVICE_UNAVAILABLE"
        assert error.severity == ErrorSeverity.HIGH
        assert error.context["service_name"] == "Database"
        assert error.context["endpoint"] == "postgresql://localhost:5432/db"
        assert error.context["retry_after"] == 300
    
    def test_maintenance_mode_error(self):
        """Test MaintenanceModeError."""
        end_time = "2024-01-01 10:00:00 UTC"
        error = MaintenanceModeError(end_time=end_time)
        
        assert isinstance(error, SystemError)
        assert "maintenance mode" in error.message
        assert end_time in error.message
        assert error.error_code == "MAINTENANCE_MODE"
        assert error.context["estimated_end_time"] == end_time
    
    def test_maintenance_mode_error_no_end_time(self):
        """Test MaintenanceModeError without end time."""
        error = MaintenanceModeError()
        
        assert "maintenance mode" in error.message
        assert "estimated_end_time" not in error.context


@pytest.mark.unit
class TestExceptionIntegration:
    """Integration tests for exception handling patterns."""
    
    def test_exception_hierarchy(self):
        """Test that all custom exceptions inherit from base correctly."""
        # Configuration errors
        assert issubclass(ConfigurationError, OccupancyPredictionError)
        assert issubclass(ConfigFileNotFoundError, ConfigurationError)
        assert issubclass(ConfigValidationError, ConfigurationError)
        
        # Home Assistant errors
        assert issubclass(HomeAssistantError, OccupancyPredictionError)
        assert issubclass(HomeAssistantConnectionError, HomeAssistantError)
        assert issubclass(EntityNotFoundError, HomeAssistantError)
        
        # Database errors
        assert issubclass(DatabaseError, OccupancyPredictionError)
        assert issubclass(DatabaseConnectionError, DatabaseError)
        assert issubclass(DatabaseQueryError, DatabaseError)
        
        # Model errors
        assert issubclass(ModelError, OccupancyPredictionError)
        assert issubclass(ModelTrainingError, ModelError)
        assert issubclass(ModelPredictionError, ModelError)
        
        # Feature engineering errors
        assert issubclass(FeatureEngineeringError, OccupancyPredictionError)
        assert issubclass(FeatureExtractionError, FeatureEngineeringError)
        
        # MQTT and integration errors
        assert issubclass(MQTTError, OccupancyPredictionError)
        assert issubclass(IntegrationError, OccupancyPredictionError)
        assert issubclass(DataValidationError, IntegrationError)
        
        # System errors
        assert issubclass(SystemError, OccupancyPredictionError)
        assert issubclass(ResourceExhaustionError, SystemError)
    
    def test_exception_catching_patterns(self):
        """Test that exceptions can be caught at different levels."""
        # Specific exception
        with pytest.raises(ConfigFileNotFoundError):
            raise ConfigFileNotFoundError("test.yaml", "/path")
        
        # Category level
        with pytest.raises(ConfigurationError):
            raise ConfigFileNotFoundError("test.yaml", "/path")
        
        # Base level
        with pytest.raises(OccupancyPredictionError):
            raise ConfigFileNotFoundError("test.yaml", "/path")
        
        # Built-in Exception level
        with pytest.raises(Exception):
            raise ConfigFileNotFoundError("test.yaml", "/path")
    
    def test_exception_chaining(self):
        """Test exception chaining with cause parameter."""
        original_error = ValueError("Original problem")
        
        chained_error = DatabaseConnectionError(
            "postgresql://localhost/db",
            cause=original_error
        )
        
        assert chained_error.cause == original_error
        assert "Caused by: ValueError: Original problem" in str(chained_error)
    
    def test_exception_context_preservation(self):
        """Test that exception context is preserved and accessible."""
        context = {
            "user_id": 123,
            "operation": "delete_user",
            "timestamp": "2024-01-01T00:00:00Z"
        }
        
        error = DatabaseQueryError(
            "DELETE FROM users WHERE id = :user_id",
            parameters={"user_id": 123},
            cause=Exception("Foreign key constraint")
        )
        
        # Additional context should be merged
        error.context.update(context)
        
        assert error.context["user_id"] == 123
        assert error.context["operation"] == "delete_user"
        assert error.context["parameters"] == {"user_id": 123}
    
    def test_severity_based_handling(self):
        """Test handling exceptions based on severity levels."""
        critical_error = ConfigFileNotFoundError("config.yaml", "/path")
        high_error = ModelTrainingError("lstm", "room1")
        medium_error = DataValidationError("API", ["Missing field"])
        
        assert critical_error.severity == ErrorSeverity.CRITICAL
        assert high_error.severity == ErrorSeverity.HIGH
        assert medium_error.severity == ErrorSeverity.MEDIUM
        
        # Test severity comparison (if needed for logging/handling logic)
        severity_levels = {
            ErrorSeverity.LOW: 1,
            ErrorSeverity.MEDIUM: 2,
            ErrorSeverity.HIGH: 3,
            ErrorSeverity.CRITICAL: 4
        }
        
        assert severity_levels[critical_error.severity] > severity_levels[high_error.severity]
        assert severity_levels[high_error.severity] > severity_levels[medium_error.severity]
    
    def test_error_code_uniqueness(self):
        """Test that error codes are unique and meaningful."""
        errors = [
            ConfigFileNotFoundError("test.yaml", "/path"),
            ConfigValidationError("field", "value", "expected"),
            HomeAssistantConnectionError("http://ha:8123"),
            HomeAssistantAuthenticationError("http://ha:8123", 64),
            DatabaseConnectionError("postgresql://localhost/db"),
            DatabaseQueryError("SELECT 1"),
            ModelTrainingError("lstm", "room1"),
            ModelPredictionError("xgboost", "room2"),
            FeatureExtractionError("temporal", "room3"),
            MQTTConnectionError("broker", 1883),
            DataValidationError("source", ["error"]),
            ResourceExhaustionError("memory", 100, 80)
        ]
        
        error_codes = [error.error_code for error in errors if error.error_code]
        assert len(error_codes) == len(set(error_codes))  # All unique
        
        # Check that error codes follow naming convention
        for code in error_codes:
            assert code.isupper()
            assert "_" in code
            assert code.endswith("_ERROR")
    
    def test_context_serialization(self):
        """Test that exception context can be serialized (for logging)."""
        import json
        
        error = ModelTrainingError(
            "lstm",
            "living_room",
            training_data_size=1000
        )
        
        # Should be able to serialize context for logging
        try:
            serialized = json.dumps(error.context, default=str)
            deserialized = json.loads(serialized)
            assert deserialized["model_type"] == "lstm"
            assert deserialized["room_id"] == "living_room"
            assert deserialized["training_data_size"] == 1000
        except (TypeError, ValueError):
            pytest.fail("Exception context should be serializable")