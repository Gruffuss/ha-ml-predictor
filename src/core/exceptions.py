"""
Custom exceptions for the Occupancy Prediction System.

This module defines specific exception classes for different components of the system,
providing detailed error context and actionable debugging information.
"""

from enum import Enum
from typing import Any, Dict, List, Optional


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
            error_code="CONFIG_FILE_NOT_FOUND",
            config_file=config_file,
            context={"config_dir": config_dir},
            severity=ErrorSeverity.CRITICAL,
        )


class ConfigValidationError(ConfigurationError):
    """Raised when configuration values are invalid or missing."""

    def __init__(
        self, field: str, value: Any, expected: str, config_file: Optional[str] = None
    ):
        message = (
            f"Invalid configuration for '{field}': got '{value}', expected {expected}"
        )
        super().__init__(
            message=message,
            error_code="CONFIG_VALIDATION_ERROR",
            config_file=config_file,
            context={"field": field, "value": value, "expected": expected},
            severity=ErrorSeverity.HIGH,
        )


class ConfigParsingError(ConfigurationError):
    """Raised when configuration file cannot be parsed."""

    def __init__(self, config_file: str, parse_error: str):
        message = f"Failed to parse configuration file '{config_file}': {parse_error}"
        super().__init__(
            message=message,
            error_code="CONFIG_PARSING_ERROR",
            config_file=config_file,
            context={"parse_error": parse_error},
            severity=ErrorSeverity.HIGH,
        )


# Home Assistant Connection Errors


class HomeAssistantError(OccupancyPredictionError):
    """Base class for Home Assistant integration errors."""

    pass


class HomeAssistantConnectionError(HomeAssistantError):
    """Raised when connection to Home Assistant fails."""

    def __init__(self, url: str, cause: Optional[Exception] = None):
        message = f"Failed to connect to Home Assistant at '{url}'"
        super().__init__(
            message=message,
            error_code="HA_CONNECTION_ERROR",
            context={"url": url},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


class HomeAssistantAuthenticationError(HomeAssistantError):
    """Raised when Home Assistant authentication fails."""

    def __init__(self, url: str, token_length: int):
        message = f"Authentication failed for Home Assistant at '{url}'"
        super().__init__(
            message=message,
            error_code="HA_AUTH_ERROR",
            context={
                "url": url,
                "token_length": token_length,
                "hint": "Check if token is valid and has required permissions",
            },
            severity=ErrorSeverity.HIGH,
        )


class HomeAssistantAPIError(HomeAssistantError):
    """Raised when Home Assistant API returns an error."""

    def __init__(
        self, endpoint: str, status_code: int, response_text: str, method: str = "GET"
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
        import re

        return re.sub(r"://([^:]+):([^@]+)@", r"://\1:***@", connection_string)


class DatabaseQueryError(DatabaseError):
    """Raised when database query execution fails."""

    def __init__(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        cause: Optional[Exception] = None,
    ):
        message = f"Database query failed: {query[:100]}..."
        super().__init__(
            message=message,
            error_code="DB_QUERY_ERROR",
            context={
                "query": query[:200],  # Limit query text
                "parameters": parameters,
            },
            severity=ErrorSeverity.MEDIUM,
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
    """Raised when database integrity constraint is violated."""

    def __init__(
        self, constraint: str, table: str, values: Optional[Dict[str, Any]] = None
    ):
        message = (
            f"Database integrity constraint violated: {constraint} on table {table}"
        )
        super().__init__(
            message=message,
            error_code="DB_INTEGRITY_ERROR",
            context={"constraint": constraint, "table": table, "values": values},
            severity=ErrorSeverity.MEDIUM,
        )


# Model Training and Prediction Errors


class ModelError(OccupancyPredictionError):
    """Base class for ML model-related errors."""

    pass


class ModelTrainingError(ModelError):
    """Raised when model training fails."""

    def __init__(
        self,
        model_type: str,
        room_id: str,
        cause: Optional[Exception] = None,
        training_data_size: Optional[int] = None,
    ):
        message = f"Model training failed for {model_type} model in room '{room_id}'"
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
        message = f"Model prediction failed for {model_type} model in room '{room_id}'"
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


class ModelNotFoundError(ModelError):
    """Raised when a required model is not found."""

    def __init__(self, model_type: str, room_id: str, model_path: Optional[str] = None):
        message = f"Model not found: {model_type} for room '{room_id}'"
        context = {"model_type": model_type, "room_id": room_id}
        if model_path:
            context["model_path"] = model_path

        super().__init__(
            message=message,
            error_code="MODEL_NOT_FOUND",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class InsufficientTrainingDataError(ModelError):
    """Raised when there's insufficient data for model training."""

    def __init__(
        self,
        room_id: str,
        data_points: int,
        minimum_required: int,
        time_span_days: Optional[float] = None,
    ):
        message = (
            f"Insufficient training data for room '{room_id}': "
            f"got {data_points} data points, need at least {minimum_required}"
        )
        context = {
            "room_id": room_id,
            "data_points": data_points,
            "minimum_required": minimum_required,
        }
        if time_span_days:
            context["time_span_days"] = time_span_days

        super().__init__(
            message=message,
            error_code="INSUFFICIENT_TRAINING_DATA_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


class ModelVersionMismatchError(ModelError):
    """Raised when loaded model version doesn't match expected version."""

    def __init__(
        self, model_type: str, room_id: str, loaded_version: str, expected_version: str
    ):
        message = (
            f"Model version mismatch for {model_type} in room '{room_id}': "
            f"loaded v{loaded_version}, expected v{expected_version}"
        )
        super().__init__(
            message=message,
            error_code="MODEL_VERSION_MISMATCH_ERROR",
            context={
                "model_type": model_type,
                "room_id": room_id,
                "loaded_version": loaded_version,
                "expected_version": expected_version,
            },
            severity=ErrorSeverity.HIGH,
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
        room_id: str,
        time_range: Optional[str] = None,
        cause: Optional[Exception] = None,
    ):
        message = (
            f"Feature extraction failed for {feature_type} features in room '{room_id}'"
        )
        context = {"feature_type": feature_type, "room_id": room_id}
        if time_range:
            context["time_range"] = time_range

        super().__init__(
            message=message,
            error_code="FEATURE_EXTRACTION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class FeatureValidationError(FeatureEngineeringError):
    """Raised when feature validation fails."""

    def __init__(
        self,
        feature_name: str,
        validation_rule: str,
        actual_value: Any,
        room_id: Optional[str] = None,
    ):
        message = f"Feature validation failed for '{feature_name}': {validation_rule}"
        context = {
            "feature_name": feature_name,
            "validation_rule": validation_rule,
            "actual_value": actual_value,
        }
        if room_id:
            context["room_id"] = room_id

        super().__init__(
            message=message,
            error_code="FEATURE_VALIDATION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


class MissingFeatureError(FeatureEngineeringError):
    """Raised when required features are missing."""

    def __init__(
        self,
        missing_features: List[str],
        room_id: str,
        available_features: Optional[List[str]] = None,
    ):
        message = f"Missing required features for room '{room_id}': {', '.join(missing_features)}"
        context = {"missing_features": missing_features, "room_id": room_id}
        if available_features:
            context["available_features"] = available_features

        super().__init__(
            message=message,
            error_code="MISSING_FEATURE_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class FeatureStoreError(FeatureEngineeringError):
    """Raised when feature store operations fail."""

    def __init__(
        self, operation: str, feature_group: str, cause: Optional[Exception] = None
    ):
        message = (
            f"Feature store operation '{operation}' failed for group '{feature_group}'"
        )
        super().__init__(
            message=message,
            error_code="FEATURE_STORE_ERROR",
            context={"operation": operation, "feature_group": feature_group},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


# MQTT and Integration Errors


class MQTTError(OccupancyPredictionError):
    """Base class for MQTT-related errors."""

    pass


class MQTTConnectionError(MQTTError):
    """Raised when MQTT broker connection fails."""

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
    """Raised when MQTT message publishing fails."""

    def __init__(
        self,
        topic: str,
        payload_size: int,
        qos: int = 0,
        cause: Optional[Exception] = None,
    ):
        message = f"Failed to publish MQTT message to topic '{topic}'"
        super().__init__(
            message=message,
            error_code="MQTT_PUBLISH_ERROR",
            context={"topic": topic, "payload_size": payload_size, "qos": qos},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class MQTTSubscriptionError(MQTTError):
    """Raised when MQTT subscription fails."""

    def __init__(self, topic_pattern: str, cause: Optional[Exception] = None):
        message = f"Failed to subscribe to MQTT topic pattern: {topic_pattern}"
        super().__init__(
            message=message,
            error_code="MQTT_SUBSCRIPTION_ERROR",
            context={"topic_pattern": topic_pattern},
            severity=ErrorSeverity.MEDIUM,
            cause=cause,
        )


class IntegrationError(OccupancyPredictionError):
    """Base class for external integration errors."""

    pass


class DataValidationError(IntegrationError):
    """Raised when incoming data fails validation."""

    def __init__(
        self,
        data_source: str,
        validation_errors: List[str],
        sample_data: Optional[Dict[str, Any]] = None,
    ):
        message = (
            f"Data validation failed from {data_source}: {'; '.join(validation_errors)}"
        )
        context = {"data_source": data_source, "validation_errors": validation_errors}
        if sample_data:
            context["sample_data"] = sample_data

        super().__init__(
            message=message,
            error_code="DATA_VALIDATION_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


class RateLimitExceededError(IntegrationError):
    """Raised when API rate limits are exceeded."""

    def __init__(
        self,
        service: str,
        limit: int,
        window_seconds: int,
        reset_time: Optional[int] = None,
    ):
        message = (
            f"Rate limit exceeded for {service}: {limit} requests per {window_seconds}s"
        )
        context = {"service": service, "limit": limit, "window_seconds": window_seconds}
        if reset_time:
            context["reset_time"] = reset_time

        super().__init__(
            message=message,
            error_code="RATE_LIMIT_EXCEEDED_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


# System and Runtime Errors


class SystemError(OccupancyPredictionError):
    """Base class for system-level errors."""

    pass


class ResourceExhaustionError(SystemError):
    """Raised when system resources are exhausted."""

    def __init__(
        self, resource_type: str, current_usage: float, limit: float, unit: str = "MB"
    ):
        message = f"Resource exhaustion: {resource_type} usage {current_usage}{unit} exceeds limit {limit}{unit}"
        super().__init__(
            message=message,
            error_code="RESOURCE_EXHAUSTION_ERROR",
            context={
                "resource_type": resource_type,
                "current_usage": current_usage,
                "limit": limit,
                "unit": unit,
            },
            severity=ErrorSeverity.HIGH,
        )


class ServiceUnavailableError(SystemError):
    """Raised when a required service is unavailable."""

    def __init__(
        self,
        service_name: str,
        endpoint: Optional[str] = None,
        retry_after: Optional[int] = None,
    ):
        message = f"Service unavailable: {service_name}"
        context = {"service_name": service_name}
        if endpoint:
            context["endpoint"] = endpoint
        if retry_after:
            context["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code="SERVICE_UNAVAILABLE_ERROR",
            context=context,
            severity=ErrorSeverity.HIGH,
        )


class MaintenanceModeError(SystemError):
    """Raised when system is in maintenance mode."""

    def __init__(self, end_time: Optional[str] = None):
        message = "System is currently in maintenance mode"
        context = {}
        if end_time:
            context["estimated_end_time"] = end_time
            message += f" (estimated end: {end_time})"

        super().__init__(
            message=message,
            error_code="MAINTENANCE_MODE_ERROR",
            context=context,
            severity=ErrorSeverity.MEDIUM,
        )


# API and Integration Errors


class APIError(OccupancyPredictionError):
    """Base class for REST API-related errors."""

    pass


class APIAuthenticationError(APIError):
    """Raised when API authentication fails."""

    def __init__(self, endpoint: str, reason: str = "Invalid API key"):
        message = f"Authentication failed for endpoint '{endpoint}': {reason}"
        super().__init__(
            message=message,
            error_code="API_AUTHENTICATION_ERROR",
            context={"endpoint": endpoint, "reason": reason},
            severity=ErrorSeverity.HIGH,
        )


class APIRateLimitError(APIError):
    """Raised when API rate limit is exceeded."""

    def __init__(self, client_ip: str, limit: int, window: str):
        message = (
            f"Rate limit exceeded for client {client_ip}: {limit} requests per {window}"
        )
        super().__init__(
            message=message,
            error_code="API_RATE_LIMIT_ERROR",
            context={"client_ip": client_ip, "limit": limit, "window": window},
            severity=ErrorSeverity.MEDIUM,
        )


class APIValidationError(APIError):
    """Raised when API request validation fails."""

    def __init__(self, field: str, value: Any, validation_error: str):
        message = f"Validation failed for field '{field}': {validation_error}"
        super().__init__(
            message=message,
            error_code="API_VALIDATION_ERROR",
            context={
                "field": field,
                "value": str(value),
                "validation_error": validation_error,
            },
            severity=ErrorSeverity.LOW,
        )


class APIResourceNotFoundError(APIError):
    """Raised when requested API resource is not found."""

    def __init__(self, resource_type: str, resource_id: str):
        message = f"{resource_type} with ID '{resource_id}' not found"
        super().__init__(
            message=message,
            error_code="API_RESOURCE_NOT_FOUND",
            context={"resource_type": resource_type, "resource_id": resource_id},
            severity=ErrorSeverity.LOW,
        )


class APIServerError(APIError):
    """Raised when internal API server error occurs."""

    def __init__(self, operation: str, cause: Optional[Exception] = None):
        message = f"Internal server error during operation: {operation}"
        super().__init__(
            message=message,
            error_code="API_SERVER_ERROR",
            context={"operation": operation},
            severity=ErrorSeverity.HIGH,
            cause=cause,
        )


# WebSocket API Errors


class WebSocketAPIError(OccupancyPredictionError):
    """Base exception for WebSocket API errors."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_API_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )


class WebSocketAuthenticationError(WebSocketAPIError):
    """Raised when WebSocket authentication fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_AUTH_ERROR",
            severity=ErrorSeverity.HIGH,
            **kwargs,
        )


class WebSocketConnectionError(WebSocketAPIError):
    """Raised when WebSocket connection operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_CONNECTION_ERROR",
            severity=ErrorSeverity.MEDIUM,
            **kwargs,
        )


class WebSocketRateLimitError(WebSocketAPIError):
    """Raised when WebSocket rate limits are exceeded."""

    def __init__(self, client_id: str, limit: int, **kwargs):
        message = f"Rate limit exceeded for client {client_id}: {limit} messages/minute"
        super().__init__(
            message=message,
            error_code="WEBSOCKET_RATE_LIMIT_ERROR",
            severity=ErrorSeverity.LOW,
            context={"client_id": client_id, "limit": limit},
            **kwargs,
        )


class WebSocketValidationError(WebSocketAPIError):
    """Raised when WebSocket message validation fails."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="WEBSOCKET_VALIDATION_ERROR",
            severity=ErrorSeverity.LOW,
            **kwargs,
        )
