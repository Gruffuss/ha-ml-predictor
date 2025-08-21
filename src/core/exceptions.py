"""
Custom exceptions for the Occupancy Prediction System.

This module defines specific exception classes for different components of the system,
providing detailed error context and actionable debugging information.
"""

from enum import Enum
import re
from typing import Any, Dict, List, Optional, Union


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
        message: str,
        config_key: Optional[str] = None,
        config_file: Optional[str] = None,
        valid_values: Optional[List[str]] = None,
    ):
        context = {}
        if config_key:
            context["config_key"] = config_key
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
            context={"parsing_error": parsing_error},
            severity=ErrorSeverity.CRITICAL,
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
            severity=ErrorSeverity.CRITICAL,
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

        # Handle both string tokens and integer token lengths
        if isinstance(token_hint, str):
            # If token_hint is a string, truncate it
            context_token_hint = (
                token_hint[:10] + "..." if len(token_hint) > 10 else token_hint
            )
        elif isinstance(token_hint, int):
            # If token_hint is an integer (token length), format it appropriately
            context_token_hint = f"<token_length_{token_hint}>"
        else:
            # Fallback for other types
            context_token_hint = str(token_hint)

        super().__init__(
            message=message,
            error_code="HA_AUTH_ERROR",
            context={"url": url, "token_hint": context_token_hint},
            severity=ErrorSeverity.CRITICAL,
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
        self, table_name: str, constraint: str, cause: Optional[Exception] = None
    ):
        message = f"Database integrity error in table '{table_name}': {constraint}"
        super().__init__(
            message=message,
            error_code="DB_INTEGRITY_ERROR",
            context={"table_name": table_name, "constraint": constraint},
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
        cause: Optional[Exception] = None,
    ):
        if room_id:
            message = f"Feature extraction failed: {feature_type} for room {room_id}"
            context = {"feature_type": feature_type, "room_id": room_id}
        else:
            message = f"Feature extraction failed: {feature_type}"
            context = {"feature_type": feature_type}

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
        room_id: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Feature validation failed for '{feature_name}': {validation_error}"
        context = {"feature_name": feature_name, "validation_error": validation_error}
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
            context={"operation": operation, "feature_type": feature_type},
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
        cause: Optional[Exception] = None,
    ):
        message = f"Model training failed: {model_type} for room {room_id}"
        super().__init__(
            message=message,
            error_code="MODEL_TRAINING_ERROR",
            context={"model_type": model_type, "room_id": room_id},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class ModelPredictionError(ModelError):
    """Raised when model prediction fails."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Model prediction failed: {model_type} for room {room_id}"
        super().__init__(
            message=message,
            error_code="MODEL_PREDICTION_ERROR",
            context={"model_type": model_type, "room_id": room_id},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


# Alias for backward compatibility
PredictionError = ModelPredictionError


class InsufficientTrainingDataError(ModelError):
    """Raised when insufficient training data is available for model training."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        required_samples: int,
        available_samples: int,
        cause: Optional[Exception] = None,
    ):
        message = (
            f"Insufficient training data for {model_type} in room {room_id}: "
            f"need {required_samples}, have {available_samples}"
        )
        super().__init__(
            message=message,
            error_code="INSUFFICIENT_TRAINING_DATA_ERROR",
            context={
                "model_type": model_type,
                "room_id": room_id,
                "required_samples": required_samples,
                "available_samples": available_samples,
            },
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
            error_code="MODEL_NOT_FOUND_ERROR",
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
            severity=ErrorSeverity.MEDIUM,
        )


class MissingFeatureError(FeatureEngineeringError):
    """Raised when required features are missing for model operation."""

    def __init__(
        self,
        feature_names: List[str],
        room_id: str,
        operation: str = "prediction",
    ):
        features_str = ", ".join(feature_names)
        message = f"Missing required features for {operation} in room {room_id}: {features_str}"
        super().__init__(
            message=message,
            error_code="MISSING_FEATURE_ERROR",
            context={
                "feature_names": feature_names,
                "room_id": room_id,
                "operation": operation,
            },
            severity=ErrorSeverity.MEDIUM,
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


class DataValidationError(DataProcessingError):
    """Raised when data validation fails."""

    def __init__(
        self,
        data_type: str,
        validation_rule: str,
        actual_value: Any,
        expected_value: Any = None,
    ):
        message = f"Data validation failed: {data_type} - {validation_rule}"
        context = {
            "data_type": data_type,
            "validation_rule": validation_rule,
            "actual_value": str(actual_value)[:100],  # Limit length
        }
        if expected_value is not None:
            context["expected_value"] = str(expected_value)[:100]

        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


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


class MQTTError(IntegrationError):
    """Base class for MQTT-related errors."""

    pass


class MQTTConnectionError(MQTTError):
    """Raised when MQTT connection fails."""

    def __init__(
        self,
        broker: str,
        port: int,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to connect to MQTT broker at {broker}:{port}"
        super().__init__(
            message=message,
            error_code="MQTT_CONNECTION_ERROR",
            context={"broker": broker, "port": port},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class MQTTPublishError(MQTTError):
    """Raised when MQTT publishing fails."""

    def __init__(
        self,
        topic: str,
        broker: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to publish to MQTT topic '{topic}' on broker '{broker}'"
        super().__init__(
            message=message,
            error_code="MQTT_PUBLISH_ERROR",
            context={"topic": topic, "broker": broker},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class MQTTSubscriptionError(MQTTError):
    """Raised when MQTT subscription fails."""

    def __init__(
        self,
        topic: str,
        broker: str,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to subscribe to MQTT topic '{topic}' on broker '{broker}'"
        super().__init__(
            message=message,
            error_code="MQTT_SUBSCRIPTION_ERROR",
            context={"topic": topic, "broker": broker},
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
        component: str,
        operation: str,
        cause: Optional[Exception] = None,
    ):
        message = f"System error in {component} during {operation}"
        super().__init__(
            message=message,
            error_code="SYSTEM_ERROR",
            context={"component": component, "operation": operation},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class ResourceExhaustionError(SystemResourceError):
    """Raised when system resources are exhausted."""

    def __init__(
        self,
        resource_type: str,
        current_usage: float,
        limit: float,
        unit: str = "%",
    ):
        super().__init__(
            resource_type=resource_type,
            resource_name=f"{current_usage}{unit}/{limit}{unit}",
        )
        self.message = (
            f"Resource exhaustion: {resource_type} at {current_usage}{unit} "
            f"(limit: {limit}{unit})"
        )
        self.error_code = "RESOURCE_EXHAUSTION_ERROR"
        self.context.update(
            {
                "current_usage": current_usage,
                "limit": limit,
                "unit": unit,
            }
        )


class ServiceUnavailableError(OccupancyPredictionError):
    """Raised when a required service is unavailable."""

    def __init__(
        self,
        service_name: str,
        reason: str = "Service unavailable",
        retry_after: Optional[int] = None,
    ):
        message = f"Service unavailable: {service_name} - {reason}"
        context = {"service_name": service_name, "reason": reason}
        if retry_after:
            context["retry_after_seconds"] = retry_after

        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class MaintenanceModeError(OccupancyPredictionError):
    """Raised when system is in maintenance mode."""

    def __init__(
        self,
        component: str,
        maintenance_until: Optional[str] = None,
    ):
        message = f"System component in maintenance mode: {component}"
        context = {"component": component}
        if maintenance_until:
            context["maintenance_until"] = maintenance_until
            message += f" (until {maintenance_until})"

        super().__init__(
            message=message,
            error_code="MAINTENANCE_MODE_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


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


class RateLimitExceededError(APIError):
    """Raised when API rate limits are exceeded."""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        window_seconds: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        error_context = context or {}
        if limit:
            error_context["rate_limit"] = limit
        if window_seconds:
            error_context["window_seconds"] = window_seconds

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED",
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
            data_type="room_id",
            validation_rule="must be non-empty string",
            actual_value=room_id,
        )

    if not re.match(r"^[a-zA-Z0-9_-]+$", room_id):
        raise DataValidationError(
            data_type="room_id",
            validation_rule="must contain only alphanumeric characters, underscores, and hyphens",
            actual_value=room_id,
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
            data_type="entity_id",
            validation_rule="must be non-empty string",
            actual_value=entity_id,
        )

    if not re.match(r"^[a-z_]+\.[a-z0-9_]+$", entity_id):
        raise DataValidationError(
            data_type="entity_id",
            validation_rule="must follow Home Assistant format (domain.object_id)",
            actual_value=entity_id,
            expected_value="sensor.living_room_motion",
        )
