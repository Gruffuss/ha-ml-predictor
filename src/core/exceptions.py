"""
Custom exceptions for the Occupancy Prediction System.

This module defines specific exception classes for different components of the system,
providing detailed error context and actionable debugging information.
"""

from enum import Enum
import re
from typing import Any, Dict, List, Optional, Sequence, Union


class ErrorSeverity(Enum):
    """Severity levels for errors."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class OccupancyPredictionError(Exception):
    """Base exception for all occupancy prediction system errors."""

    def __init__(
        self,
        message: str,
        error_code: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        cause: Optional[Exception] = None,
    ):
        """
        Initialize base exception.

        Args:
            message: Human-readable error message
            error_code: Unique error code for logging/monitoring
            context: Additional context information
            severity: Error severity level
            cause: The underlying exception that caused this error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.severity = severity
        self.cause = cause

    def __str__(self) -> str:
        """Return formatted error message with context."""
        parts = [self.message]

        if self.error_code:
            parts.append(f"Error Code: {self.error_code}")

        if self.context:
            context_str = ", ".join(f"{k}={v}" for k, v in self.context.items())
            parts.append(f"Context: {context_str}")

        if self.cause:
            parts.append(f"Caused by: {type(self.cause).__name__}: {self.cause}")

        return " | ".join(parts)


# Configuration Errors


class ConfigurationError(OccupancyPredictionError):
    """Base class for configuration-related errors."""

    def __init__(self, message: str, config_file: Optional[str] = None, **kwargs):
        context = kwargs.get("context", {})
        if config_file:
            context["config_file"] = config_file
        kwargs["context"] = context
        super().__init__(message, **kwargs)


class ConfigFileNotFoundError(ConfigurationError):
    """Raised when a required configuration file is missing."""

    def __init__(self, config_file: str, config_dir: str):
        message = (
            f"Configuration file '{config_file}' not found in directory '{config_dir}'"
        )
        super().__init__(
            message=message,
            error_code="CONFIG_FILE_NOT_FOUND_ERROR",
            config_file=config_file,
            context={"config_dir": config_dir},
            severity=ErrorSeverity.CRITICAL,
        )


class ConfigValidationError(ConfigurationError):
    """Raised when configuration values are invalid or missing."""

    def __init__(
        self,
        message: Optional[str] = None,
        config_key: Optional[str] = None,
        field: Optional[str] = None,
        value: Any = ...,  # Use ellipsis as sentinel for "not provided"
        expected: Optional[str] = None,
        config_file: Optional[str] = None,
        valid_values: Optional[List[str]] = None,
    ):
        # Auto-generate message if not provided but field info is available
        if message is None:
            if field and expected:
                message = f"Invalid configuration field '{field}': got {value}, expected {expected}"
            elif field:
                message = f"Invalid configuration field '{field}': {value}"
            else:
                message = "Configuration validation failed"

        context = {}
        if config_key:
            context["config_key"] = config_key
        if field:
            context["field"] = field
        # Include value even if None (using ellipsis as sentinel)
        if value is not ...:
            context["value"] = value
        if expected:
            context["expected"] = expected
        if valid_values:
            context["valid_values"] = valid_values

        super().__init__(
            message=message,
            error_code="CONFIG_VALIDATION_ERROR",
            config_file=config_file,
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class MissingConfigSectionError(ConfigurationError):
    """Raised when a required configuration section is missing."""

    def __init__(self, section_name: str, config_file: str):
        message = f"Required configuration section '{section_name}' missing"
        super().__init__(
            message=message,
            error_code="CONFIG_SECTION_MISSING_ERROR",
            config_file=config_file,
            context={"section_name": section_name},
            severity=ErrorSeverity.HIGH,
        )


class ConfigParsingError(ConfigurationError):
    """Raised when configuration file parsing fails."""

    def __init__(self, config_file: str, parsing_error: str):
        message = f"Failed to parse configuration file '{config_file}': {parsing_error}"
        super().__init__(
            message=message,
            error_code="CONFIG_PARSING_ERROR",
            config_file=config_file,
            context={"parse_error": parsing_error},
            severity=ErrorSeverity.HIGH,
        )


# Home Assistant Integration Errors


class HomeAssistantError(OccupancyPredictionError):
    """Base class for Home Assistant integration errors."""

    pass


class HomeAssistantConnectionError(HomeAssistantError):
    """Raised when connection to Home Assistant fails."""

    def __init__(self, url: str, cause: Optional[Exception] = None):
        message = f"Failed to connect to Home Assistant at {url}"
        super().__init__(
            message=message,
            error_code="HA_CONNECTION_ERROR",
            context={"url": url},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class HomeAssistantAuthenticationError(HomeAssistantError):
    """Raised when authentication with Home Assistant fails."""

    def __init__(self, url: str, token_hint: Union[str, int]):
        """Initialize authentication error.

        Args:
            url: The Home Assistant URL
            token_hint: Either a string token to truncate or an integer token length
        """
        message = f"Authentication failed for Home Assistant at {url}"

        context = {"url": url}

        # Handle both string tokens and integer token lengths
        if isinstance(token_hint, str):
            # If token_hint is a string, truncate it
            context["token_hint"] = (
                token_hint[:10] + "..." if len(token_hint) > 10 else token_hint
            )
        elif isinstance(token_hint, int):
            # If token_hint is an integer (token length), store as token_length
            context["token_length"] = token_hint
        else:
            # Fallback for other types
            context["token_hint"] = str(token_hint)

        # Add helpful hint
        context["hint"] = "Check if token is valid and has required permissions"

        super().__init__(
            message=message,
            error_code="HA_AUTH_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class HomeAssistantAPIError(HomeAssistantError):
    """Raised when Home Assistant API returns an error."""

    def __init__(
        self,
        endpoint: str,
        status_code: int,
        response_text: str,
        method: str = "GET",
    ):
        message = (
            f"Home Assistant API error: {method} {endpoint} returned {status_code}"
        )
        super().__init__(
            message=message,
            error_code="HA_API_ERROR",
            context={
                "endpoint": endpoint,
                "method": method,
                "status_code": status_code,
                "response": response_text[:500],  # Limit response text
            },
            severity=ErrorSeverity.MEDIUM,
        )


class EntityNotFoundError(HomeAssistantError):
    """Raised when a required entity is not found in Home Assistant."""

    def __init__(self, entity_id: str, room_id: Optional[str] = None):
        message = f"Entity '{entity_id}' not found in Home Assistant"
        context = {"entity_id": entity_id}
        if room_id:
            context["room_id"] = room_id
            message += f" (required for room '{room_id}')"

        super().__init__(
            message=message,
            error_code="ENTITY_NOT_FOUND_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


class WebSocketError(HomeAssistantError):
    """Raised when WebSocket connection to Home Assistant fails."""

    def __init__(self, reason: str, url: str):
        message = f"WebSocket connection failed: {reason}"
        super().__init__(
            message=message,
            error_code="HA_WEBSOCKET_ERROR",
            context={"url": url, "reason": reason},
            severity=ErrorSeverity.HIGH,
        )


class WebSocketConnectionError(WebSocketError):
    """Raised when WebSocket connection fails."""

    def __init__(self, url: str, cause: Optional[Exception] = None):
        super().__init__(reason="Connection failed", url=url)
        self.cause = cause
        self.error_code = "WEBSOCKET_CONNECTION_ERROR"


class WebSocketAuthenticationError(WebSocketError):
    """Raised when WebSocket authentication fails."""

    def __init__(self, url: str, auth_method: Optional[str] = None):
        reason = "Authentication failed"
        if auth_method:
            reason += f" ({auth_method})"
        super().__init__(reason=reason, url=url)
        self.error_code = "WEBSOCKET_AUTH_ERROR"
        if auth_method:
            self.context["auth_method"] = auth_method


class WebSocketValidationError(WebSocketError):
    """Raised when WebSocket message validation fails."""

    def __init__(
        self, url: str, validation_error: str, message_type: Optional[str] = None
    ):
        reason = f"Message validation failed: {validation_error}"
        super().__init__(reason=reason, url=url)
        self.error_code = "WEBSOCKET_VALIDATION_ERROR"
        self.context["validation_error"] = validation_error
        if message_type:
            self.context["message_type"] = message_type


# Database Errors


class DatabaseError(OccupancyPredictionError):
    """Base class for database-related errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Raised when database connection fails."""

    def __init__(self, connection_string: str, cause: Optional[Exception] = None):
        # Mask password in connection string for logging
        masked_conn = self._mask_password(connection_string)
        message = f"Failed to connect to database: {masked_conn}"
        super().__init__(
            message=message,
            error_code="DB_CONNECTION_ERROR",
            context={"connection_string": masked_conn},
            severity=ErrorSeverity.CRITICAL,
            cause=cause,
        )

    @staticmethod
    def _mask_password(connection_string: str) -> str:
        """Mask password in connection string for safe logging."""
        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", connection_string)


class DatabaseQueryError(DatabaseError):
    """Raised when database query execution fails."""

    def __init__(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
        error_type: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        """
        Initialize database query error.

        Args:
            query: The SQL query that failed
            parameters: Query parameters that were used
            cause: The underlying exception that caused the failure
            error_type: Type of error (e.g., 'TimeoutError')
            severity: Error severity level
        """
        message = f"Database query failed: {query[:100]}..."

        context = {
            "query": query[:200],  # Limit query text
            "parameters": parameters,
        }

        if error_type:
            context["error_type"] = error_type

        super().__init__(
            message=message,
            error_code="DB_QUERY_ERROR",
            context=context,
            severity=severity,
            cause=cause,
        )


class DatabaseMigrationError(DatabaseError):
    """Raised when database migration fails."""

    def __init__(self, migration_name: str, cause: Optional[Exception] = None):
        message = f"Database migration failed: {migration_name}"
        super().__init__(
            message=message,
            error_code="DB_MIGRATION_ERROR",
            context={"migration_name": migration_name},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class DatabaseIntegrityError(DatabaseError):
    """Raised when database integrity constraints are violated."""

    def __init__(
        self,
        constraint: str,
        table_name: str,
        values: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Database integrity error in table '{table_name}': {constraint}"
        context = {"constraint": constraint, "table": table_name}
        if values:
            context["values"] = values
        super().__init__(
            message=message,
            error_code="DB_INTEGRITY_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


# Feature Engineering Errors


class FeatureEngineeringError(OccupancyPredictionError):
    """Base class for feature engineering errors."""

    pass


class FeatureExtractionError(FeatureEngineeringError):
    """Raised when feature extraction fails."""

    def __init__(
        self,
        feature_type: str,
        room_id: Optional[str] = None,
        time_range: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        if room_id:
            message = f"Feature extraction failed: {feature_type} for room {room_id}"
            context = {"feature_type": feature_type, "room_id": room_id}
        else:
            message = f"Feature extraction failed: {feature_type}"
            context = {"feature_type": feature_type}

        if time_range:
            context["time_range"] = time_range

        super().__init__(
            message=message,
            error_code="FEATURE_EXTRACTION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class InsufficientDataError(FeatureEngineeringError):
    """Raised when insufficient data is available for feature extraction."""

    def __init__(
        self,
        data_type: str,
        room_id: str,
        required_samples: int,
        available_samples: int,
    ):
        message = (
            f"Insufficient {data_type} data for room {room_id}: "
            f"need {required_samples}, have {available_samples}"
        )
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA_ERROR",
            context={
                "data_type": data_type,
                "room_id": room_id,
                "required_samples": required_samples,
                "available_samples": available_samples,
            },
            severity=ErrorSeverity.MEDIUM,
        )


class FeatureValidationError(FeatureEngineeringError):
    """Raised when feature validation fails."""

    def __init__(
        self,
        feature_name: str,
        validation_error: str,
        actual_value: Any,
        room_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Feature validation failed for '{feature_name}': {validation_error}"
        context = {
            "feature_name": feature_name,
            "validation_error": validation_error,
            "validation_rule": validation_error,  # For backward compatibility
            "actual_value": actual_value,
        }
        if room_id:
            context["room_id"] = room_id
            message += f" (room: {room_id})"

        super().__init__(
            message=message,
            error_code="FEATURE_VALIDATION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class FeatureStoreError(FeatureEngineeringError):
    """Raised when feature store operations fail."""

    def __init__(
        self,
        operation: str,
        feature_type: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Feature store operation failed: {operation} for {feature_type}"
        super().__init__(
            message=message,
            error_code="FEATURE_STORE_ERROR",
            context={"operation": operation, "feature_group": feature_type},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


# Model Training Errors


class ModelError(OccupancyPredictionError):
    """Base class for machine learning model errors."""

    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        training_data_size: Optional[int] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Model training failed: {model_type} for room {room_id}"
        context = {"model_type": model_type, "room_id": room_id}
        if training_data_size is not None:
            context["training_data_size"] = training_data_size
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        feature_shape: Optional[tuple] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Model prediction failed: {model_type} for room {room_id}"
        context = {"model_type": model_type, "room_id": room_id}
        if feature_shape:
            context["feature_shape"] = feature_shape
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


# Alias for backward compatibility
PredictionError = ModelPredictionError


class InsufficientTrainingDataError(ModelError):
    """Raised when insufficient training data is available for model training."""

    def __init__(
        self,
        room_id: str,
        data_points: Optional[int] = None,
        minimum_required: Optional[int] = None,
        time_span_days: Optional[float] = None,
        model_type: Optional[str] = None,
        required_samples: Optional[int] = None,
        available_samples: Optional[int] = None,
        cause: Optional[Exception] = None,
    ):
        # Create message based on available parameters
        if data_points is not None and minimum_required is not None:
            base_message = f"have {data_points}, need {minimum_required}"
            if model_type:
                message = (
                    f"Insufficient training data for {model_type} in room {room_id}: "
                    f"{base_message}"
                )
            else:
                message = (
                    f"Insufficient training data for room {room_id}: "
                    f"{base_message}"
                )
        elif required_samples is not None and available_samples is not None:
            base_message = f"need {required_samples}, have {available_samples}"
            if model_type:
                message = (
                    f"Insufficient training data for {model_type} in room {room_id}: "
                    f"{base_message}"
                )
            else:
                message = (
                    f"Insufficient training data for room {room_id}: "
                    f"{base_message}"
                )
        else:
            if model_type:
                message = f"Insufficient training data for {model_type} in room {room_id}"
            else:
                message = f"Insufficient training data for room {room_id}"

        context = {"room_id": room_id}
        if model_type is not None:
            context["model_type"] = model_type
        if data_points is not None:
            context["data_points"] = data_points
        if minimum_required is not None:
            context["minimum_required"] = minimum_required
        if time_span_days is not None:
            context["time_span_days"] = time_span_days
        if required_samples is not None:
            context["required_samples"] = required_samples
        if available_samples is not None:
            context["available_samples"] = available_samples

        super().__init__(
            message=message,
            error_code="INSUFFICIENT_TRAINING_DATA_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        model_path: Optional[str] = None,
    ):
        message = f"Model not found: {model_type} for room {room_id}"
        context = {"model_type": model_type, "room_id": room_id}
        if model_path:
            context["model_path"] = model_path

        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class ModelVersionMismatchError(ModelError):
    """Raised when model version doesn't match expected version."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        expected_version: str,
        actual_version: str,
        cause: Optional[Exception] = None,
    ):
        message = (
            f"Model version mismatch for {model_type} in room {room_id}: "
            f"expected {expected_version}, got {actual_version}"
        )
        super().__init__(
            message=message,
            error_code="MODEL_VERSION_MISMATCH_ERROR",
            context={
                "model_type": model_type,
                "room_id": room_id,
                "expected_version": expected_version,
                "actual_version": actual_version,
            },
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class MissingFeatureError(FeatureEngineeringError):
    """Raised when required features are missing for model operation."""

    def __init__(
        self,
        feature_names: List[str],
        room_id: str,
        available_features: Optional[List[str]] = None,
        operation: str = "prediction",
    ):
        features_str = ", ".join(feature_names)
        message = f"Missing required features for {operation} in room {room_id}: {features_str}"
        context = {
            "feature_names": feature_names,
            "missing_features": feature_names,  # For backward compatibility
            "room_id": room_id,
            "operation": operation,
        }
        if available_features:
            context["available_features"] = available_features
        super().__init__(
            message=message,
            error_code="MISSING_FEATURE_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class ModelValidationError(ModelError):
    """Raised when model validation fails."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        validation_error: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Model validation failed: {model_type} for room {room_id} - {validation_error}"
        super().__init__(
            message=message,
            error_code="MODEL_VALIDATION_ERROR",
            context={
                "model_type": model_type,
                "room_id": room_id,
                "validation_error": validation_error,
            },
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


# Data Processing Errors


class DataProcessingError(OccupancyPredictionError):
    """Base class for data processing errors."""

    pass


# DataValidationError moved to Integration Errors section


class DataCorruptionError(DataProcessingError):
    """Raised when data corruption is detected."""

    def __init__(self, data_source: str, corruption_details: str):
        message = f"Data corruption detected in {data_source}: {corruption_details}"
        super().__init__(
            message=message,
            error_code="DATA_CORRUPTION_ERROR",
            context={
                "data_source": data_source,
                "corruption_details": corruption_details,
            },
            severity=ErrorSeverity.HIGH,
        )


# Integration Errors


class IntegrationError(OccupancyPredictionError):
    """Base class for integration-related errors."""

    pass


class DataValidationError(IntegrationError):
    """Raised when data validation fails."""

    def __init__(
        self,
        data_source: str,
        validation_errors: List[str],
        sample_data: Optional[Dict[str, Any]] = None,
        # Legacy parameters for backward compatibility
        data_type: Optional[str] = None,
        validation_rule: Optional[str] = None,
        actual_value: Any = None,
        expected_value: Any = None,
        field_name: Optional[str] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        # Primary signature: data_source, validation_errors
        context: Dict[str, Union[str, Sequence[str], Dict[str, Any], None]]
        if validation_errors:
            message = f"Data validation failed for {data_source}: {'; '.join(validation_errors)}"
            context = {
                "data_source": data_source,
                "validation_errors": validation_errors,
            }
            if sample_data is not None:
                context["sample_data"] = sample_data
        # Legacy signature: data_type, validation_rule, actual_value
        elif data_type and validation_rule:
            message = f"Data validation failed: {data_type} - {validation_rule}"
            context = {
                "data_type": data_type,
                "validation_rule": validation_rule,
                "actual_value": (
                    str(actual_value)[:100] if actual_value is not None else None
                ),
            }
            if expected_value is not None:
                context["expected_value"] = str(expected_value)[:100]
            if field_name:
                context["field_name"] = field_name
        else:
            message = f"Data validation failed for {data_source}"
            context = {"data_source": data_source}

        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context,
            severity=severity,
        )


class MQTTError(IntegrationError):
    """Base class for MQTT-related errors."""

    pass


class MQTTConnectionError(MQTTError):
    """Raised when MQTT connection fails."""

    def __init__(
        self,
        broker: str,
        port: int,
        username: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to connect to MQTT broker at {broker}:{port}"
        context = {"broker": broker, "port": port}
        if username:
            context["username"] = username
        super().__init__(
            message=message,
            error_code="MQTT_CONNECTION_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class MQTTPublishError(MQTTError):
    """Raised when MQTT publishing fails."""

    def __init__(
        self,
        topic: str,
        payload_size: Optional[int] = None,
        qos: Optional[int] = None,
        broker: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to publish to MQTT topic '{topic}'"
        if broker:
            message += f" on broker '{broker}'"
        context = {"topic": topic}
        if broker:
            context["broker"] = broker
        if payload_size is not None:
            context["payload_size"] = payload_size
        if qos is not None:
            context["qos"] = qos
        super().__init__(
            message=message,
            error_code="MQTT_PUBLISH_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class MQTTSubscriptionError(MQTTError):
    """Raised when MQTT subscription fails."""

    def __init__(
        self,
        topic: str,
        broker: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to subscribe to MQTT topic '{topic}'"
        context = {"topic_pattern": topic}
        if broker:
            message += f" on broker '{broker}'"
            context["broker"] = broker
        super().__init__(
            message=message,
            error_code="MQTT_SUBSCRIPTION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class APIServerError(IntegrationError):
    """Raised when API server operations fail."""

    def __init__(
        self,
        endpoint: str,
        operation: str,
        cause: Optional[Exception] = None,
    ):
        message = f"API server error: {operation} on {endpoint}"
        super().__init__(
            message=message,
            error_code="API_SERVER_ERROR",
            context={"endpoint": endpoint, "operation": operation},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


# System-wide Errors


class SystemInitializationError(OccupancyPredictionError):
    """Raised when system initialization fails."""

    def __init__(self, component: str, cause: Optional[Exception] = None):
        message = f"Failed to initialize system component: {component}"
        super().__init__(
            message=message,
            error_code="SYSTEM_INIT_ERROR",
            context={"component": component},
            severity=ErrorSeverity.CRITICAL,
            cause=cause,
        )


class SystemResourceError(OccupancyPredictionError):
    """Raised when system resources are exhausted or unavailable."""

    def __init__(
        self,
        resource_type: str,
        resource_name: str,
        cause: Optional[Exception] = None,
    ):
        message = f"System resource unavailable: {resource_type} - {resource_name}"
        super().__init__(
            message=message,
            error_code="SYSTEM_RESOURCE_ERROR",
            context={"resource_type": resource_type, "resource_name": resource_name},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class SystemError(OccupancyPredictionError):
    """Raised when a general system error occurs."""

    def __init__(
        self,
        message: str,
        operation: Optional[str] = None,
        component: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        # If operation is provided, incorporate it into the message
        if operation and component:
            full_message = f"System error in {component} during {operation}: {message}"
        elif operation:
            full_message = f"System error during {operation}: {message}"
        else:
            full_message = message

        context = {}
        if component:
            context["component"] = component
        if operation:
            context["operation"] = operation

        super().__init__(
            message=full_message,
            error_code="SYSTEM_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class ResourceExhaustionError(SystemError):
    """Raised when system resources are exhausted."""

    def __init__(
        self,
        resource_type: str,
        current_usage: float,
        limit: float,
        unit: str = "%",
    ):
        message = (
            f"Resource exhaustion: {resource_type} at {current_usage}{unit} "
            f"(limit: {limit}{unit})"
        )
        super().__init__(
            message=message,
            operation="resource_monitoring",
            component=resource_type,
        )
        self.error_code = "RESOURCE_EXHAUSTION_ERROR"
        self.context.update(
            {
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "unit": unit,
            }
        )


class ServiceUnavailableError(SystemError):
    """Raised when a required service is unavailable."""

    def __init__(
        self,
        service_name: str,
        endpoint: Optional[str] = None,
        retry_after: Optional[int] = None,
        reason: str = "Service unavailable",
    ):
        message = f"Service unavailable: {service_name}"
        if reason != "Service unavailable":
            message += f" - {reason}"

        context = {"service_name": service_name}
        if endpoint:
            context["endpoint"] = endpoint
        if retry_after:
            context["retry_after"] = retry_after
        if reason != "Service unavailable":
            context["reason"] = reason

        super().__init__(
            message=message,
            operation="service_access",
            component=service_name,
        )
        self.error_code = "SERVICE_UNAVAILABLE_ERROR"
        self.context = context


class MaintenanceModeError(SystemError):
    """Raised when system is in maintenance mode."""

    def __init__(
        self,
        component: Optional[str] = None,
        end_time: Optional[str] = None,
        maintenance_until: Optional[str] = None,  # backward compatibility
    ):
        # Use end_time if provided, otherwise maintenance_until for backward compatibility
        end_time = end_time or maintenance_until

        if component:
            message = f"System component in maintenance mode: {component}"
        else:
            message = "System in maintenance mode"

        context = {}
        if component:
            context["component"] = component
        if end_time:
            context["estimated_end_time"] = end_time
            message += f" (until {end_time})"

        super().__init__(
            message=message,
            operation="maintenance",
            component=component or "system",
        )
        self.error_code = "MAINTENANCE_MODE_ERROR"
        self.context = context
        self.severity = ErrorSeverity.MEDIUM


class APIError(OccupancyPredictionError):
    """Base class for API-related errors."""

    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(
        self,
        message: str = "Authentication failed",
        endpoint: Optional[str] = None,
        auth_method: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if endpoint:
            error_context["endpoint"] = endpoint
        if auth_method:
            error_context["auth_method"] = auth_method

        super().__init__(
            message=message,
            error_code="API_AUTH_ERROR",
            context=error_context,
            severity=ErrorSeverity.HIGH,
        )


class RateLimitExceededError(IntegrationError):
    """Raised when API rate limits are exceeded."""

    def __init__(
        self,
        service: str,
        limit: int,
        window_seconds: int,
        reset_time: Optional[Union[str, int]] = None,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        if message is None:
            message = f"Rate limit exceeded for {service}: {limit} requests per {window_seconds}s"

        error_context = context or {}
        error_context.update(
            {
                "service": service,
                "limit": limit,
                "window_seconds": window_seconds,
            }
        )
        if reset_time is not None:
            error_context["reset_time"] = reset_time

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED_ERROR",
            context=error_context,
            severity=ErrorSeverity.MEDIUM,
        )


# Alias for backward compatibility
APIRateLimitError = RateLimitExceededError


class APIAuthorizationError(APIError):
    """Raised when API authorization fails."""

    def __init__(
        self,
        message: str = "Authorization failed",
        endpoint: Optional[str] = None,
        required_permission: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if endpoint:
            error_context["endpoint"] = endpoint
        if required_permission:
            error_context["required_permission"] = required_permission

        super().__init__(
            message=message,
            error_code="API_AUTHORIZATION_ERROR",
            context=error_context,
            severity=ErrorSeverity.HIGH,
        )


class APISecurityError(APIError):
    """Raised when API security violations are detected."""

    def __init__(
        self,
        message: str = "Security violation detected",
        violation_type: Optional[str] = None,
        endpoint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if violation_type:
            error_context["violation_type"] = violation_type
        if endpoint:
            error_context["endpoint"] = endpoint

        super().__init__(
            message=message,
            error_code="API_SECURITY_ERROR",
            context=error_context,
            severity=ErrorSeverity.CRITICAL,
        )


class APIResourceNotFoundError(APIError):
    """Raised when a requested API resource is not found."""

    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        endpoint: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        message = f"{resource_type} with ID '{resource_id}' not found"
        error_context = context or {}
        error_context.update(
            {"resource_type": resource_type, "resource_id": resource_id}
        )
        if endpoint:
            error_context["endpoint"] = endpoint

        super().__init__(
            message=message,
            error_code="API_RESOURCE_NOT_FOUND",
            context=error_context,
            severity=ErrorSeverity.MEDIUM,
        )


# Validation Helpers


def validate_room_id(room_id: str) -> None:
    """
    Validate room ID format.

    Args:
        room_id: Room identifier to validate

    Raises:
        DataValidationError: If room ID is invalid
    """
    if not room_id or not isinstance(room_id, str):
        raise DataValidationError(
            data_source="room_id_validation",
            validation_errors=["must be non-empty string"],
            sample_data={"room_id": room_id},
        )

    if not re.match(r"^[a-zA-Z0-9_-]+$", room_id):
        raise DataValidationError(
            data_source="room_id_validation", 
            validation_errors=["must contain only alphanumeric characters, underscores, and hyphens"],
            sample_data={"room_id": room_id},
        )


def validate_entity_id(entity_id: str) -> None:
    """
    Validate Home Assistant entity ID format.

    Args:
        entity_id: Entity ID to validate

    Raises:
        DataValidationError: If entity ID is invalid
    """
    if not entity_id or not isinstance(entity_id, str):
        raise DataValidationError(
            data_source="entity_id_validation",
            validation_errors=["must be non-empty string"],
            sample_data={"entity_id": entity_id},
        )

    if not re.match(r"^[a-z_]+\.[a-z0-9_]+$", entity_id):
        raise DataValidationError(
            data_source="entity_id_validation",
            validation_errors=["must follow Home Assistant format (domain.object_id)"],
            sample_data={"entity_id": entity_id, "expected_format": "sensor.living_room_motion"},
        )
