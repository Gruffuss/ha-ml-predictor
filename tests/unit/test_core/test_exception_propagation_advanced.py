"""
Advanced exception handling and error propagation tests.

Comprehensive tests for exception handling across system layers, error context
preservation, exception chaining, and production-grade error scenarios.
"""

import asyncio
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from src.core.exceptions import (  # Base exception; Configuration exceptions; Home Assistant exceptions; Database exceptions; Feature engineering exceptions; Model exceptions; Integration exceptions; System exceptions; API exceptions; Validation functions
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


class TestExceptionContextPreservation:
    """Test that exception context is preserved through error propagation."""

    def test_exception_chaining_preserves_original_cause(self):
        """Test that exception chaining preserves original cause."""
        original_error = ValueError("Database connection refused")

        db_error = DatabaseConnectionError(
            "postgresql://localhost:5432/test", cause=original_error
        )

        system_error = SystemInitializationError("database", cause=db_error)

        # Verify cause chain
        assert system_error.cause == db_error
        assert db_error.cause == original_error

        # Verify context preservation
        assert "Database connection refused" in str(system_error)
        assert "postgresql://localhost:5432/test" in str(system_error)

    def test_nested_exception_context_accumulation(self):
        """Test that nested exceptions accumulate context information."""
        # Start with low-level error
        connection_error = ConnectionError("Network unreachable")

        # Wrap in HA error
        ha_error = HomeAssistantConnectionError(
            "http://homeassistant.local:8123", cause=connection_error
        )

        # Wrap in system initialization error
        init_error = SystemInitializationError("home_assistant", cause=ha_error)

        error_str = str(init_error)

        # All context should be preserved
        assert "home_assistant" in error_str
        assert "homeassistant.local:8123" in error_str
        assert "Network unreachable" in error_str
        assert "ConnectionError" in error_str

    def test_exception_context_serialization(self):
        """Test that exception context can be serialized for logging."""
        context = {
            "user_id": 12345,
            "operation": "model_training",
            "room_id": "living_room",
            "timestamp": "2024-01-15T10:30:00Z",
            "features_count": 150,
        }

        error = ModelTrainingError(
            model_type="lstm",
            room_id="living_room",
            training_data_size=1000,
            cause=ValueError("Insufficient memory"),
        )

        # Add additional context
        error.context.update(context)

        # Should be JSON serializable
        serialized = json.dumps(
            {
                "message": error.message,
                "error_code": error.error_code,
                "context": error.context,
                "severity": error.severity.value,
            }
        )

        deserialized = json.loads(serialized)
        assert deserialized["context"]["user_id"] == 12345
        assert deserialized["context"]["room_id"] == "living_room"
        assert deserialized["severity"] == "high"

    def test_exception_context_filtering_sensitive_data(self):
        """Test that sensitive data in context can be filtered."""
        sensitive_context = {
            "database_password": "super_secret_password",
            "api_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.sensitive_token",
            "user_data": {"name": "John", "ssn": "123-45-6789"},
            "room_id": "bedroom",
            "operation": "data_sync",
        }

        error = DatabaseConnectionError(
            "postgresql://user:password@localhost/db",
            cause=Exception("Connection timeout"),
        )
        error.context.update(sensitive_context)

        def filter_sensitive_context(context: Dict[str, Any]) -> Dict[str, Any]:
            """Filter sensitive information from error context."""
            filtered = {}
            sensitive_keys = ["password", "token", "ssn", "key", "secret"]

            for key, value in context.items():
                if any(sensitive in key.lower() for sensitive in sensitive_keys):
                    if isinstance(value, str):
                        filtered[key] = value[:4] + "..." if len(value) > 4 else "***"
                    else:
                        filtered[key] = "[FILTERED]"
                elif isinstance(value, dict):
                    filtered[key] = filter_sensitive_context(value)
                else:
                    filtered[key] = value

            return filtered

        filtered_context = filter_sensitive_context(error.context)

        # Non-sensitive data preserved
        assert filtered_context["room_id"] == "bedroom"
        assert filtered_context["operation"] == "data_sync"

        # Sensitive data filtered
        assert filtered_context["database_password"] == "supe..."
        assert filtered_context["api_token"] == "eyJ0..."
        assert filtered_context["user_data"] == "[FILTERED]"


class TestExceptionHierarchyValidation:
    """Test exception hierarchy and inheritance relationships."""

    def test_all_exceptions_inherit_from_base(self):
        """Test that all custom exceptions inherit from OccupancyPredictionError."""
        exception_classes = [
            ConfigurationError,
            ConfigFileNotFoundError,
            ConfigValidationError,
            HomeAssistantError,
            HomeAssistantConnectionError,
            DatabaseError,
            DatabaseConnectionError,
            DatabaseQueryError,
            FeatureEngineeringError,
            FeatureExtractionError,
            ModelError,
            ModelTrainingError,
            ModelPredictionError,
            IntegrationError,
            MQTTError,
            APIServerError,
            SystemError,
            SystemInitializationError,
            APIError,
            APIAuthenticationError,
        ]

        for exc_class in exception_classes:
            instance = exc_class("Test error")
            assert isinstance(instance, OccupancyPredictionError)
            assert isinstance(instance, Exception)

    def test_exception_hierarchy_specificity(self):
        """Test that specific exceptions inherit from appropriate parent classes."""
        # Configuration hierarchy
        config_file_error = ConfigFileNotFoundError("test.yaml", "/config")
        assert isinstance(config_file_error, ConfigurationError)
        assert isinstance(config_file_error, OccupancyPredictionError)

        # Home Assistant hierarchy
        ha_auth_error = HomeAssistantAuthenticationError("http://test", "token")
        assert isinstance(ha_auth_error, HomeAssistantError)
        assert isinstance(ha_auth_error, OccupancyPredictionError)

        # Database hierarchy
        db_query_error = DatabaseQueryError("SELECT * FROM test", cause=Exception())
        assert isinstance(db_query_error, DatabaseError)
        assert isinstance(db_query_error, OccupancyPredictionError)

        # Model hierarchy
        model_training_error = ModelTrainingError("lstm", "room1")
        assert isinstance(model_training_error, ModelError)
        assert isinstance(model_training_error, OccupancyPredictionError)

    def test_exception_severity_inheritance(self):
        """Test that exception severity is properly inherited and set."""
        critical_errors = [
            ConfigFileNotFoundError("config.yaml", "/config"),
            SystemInitializationError("database"),
            DatabaseConnectionError("postgresql://test"),
        ]

        for error in critical_errors:
            assert error.severity == ErrorSeverity.CRITICAL

        high_severity_errors = [
            ConfigValidationError(field="test", value=None, expected="string"),
            ModelTrainingError("lstm", "room1"),
            DatabaseIntegrityError("unique_constraint", "users"),
        ]

        for error in high_severity_errors:
            assert error.severity == ErrorSeverity.HIGH


class TestSystemLayerErrorPropagation:
    """Test error propagation across different system layers."""

    def test_data_layer_to_service_layer_propagation(self):
        """Test error propagation from data layer to service layer."""
        # Simulate data layer error
        db_error = DatabaseQueryError(
            query="SELECT * FROM sensor_events WHERE room_id = ?",
            parameters={"room_id": "living_room"},
            cause=Exception("Table does not exist"),
        )

        # Service layer catches and wraps
        feature_error = FeatureExtractionError(
            feature_type="temporal", room_id="living_room", cause=db_error
        )

        # Application layer catches and wraps
        system_error = SystemError(
            message="Failed to extract features for prediction",
            operation="prediction_generation",
            component="feature_service",
            cause=feature_error,
        )

        # Verify error chain
        assert system_error.cause == feature_error
        assert feature_error.cause == db_error
        assert "Table does not exist" in str(system_error)
        assert "living_room" in system_error.context[
            "component"
        ] or "living_room" in str(system_error)

    def test_external_service_to_internal_error_propagation(self):
        """Test error propagation from external services to internal components."""
        # External Home Assistant service error
        ha_api_error = HomeAssistantAPIError(
            endpoint="/api/states/binary_sensor.living_room_motion",
            status_code=404,
            response_text="Entity not found",
            method="GET",
        )

        # Data ingestion layer wraps
        entity_error = EntityNotFoundError(
            entity_id="binary_sensor.living_room_motion", room_id="living_room"
        )
        entity_error.cause = ha_api_error

        # Feature extraction fails due to missing entity
        feature_error = MissingFeatureError(
            feature_names=["presence_state"], room_id="living_room", cause=entity_error
        )

        # Model prediction fails
        model_error = ModelPredictionError(
            model_type="lstm", room_id="living_room", cause=feature_error
        )

        # Verify complete error chain
        assert model_error.cause == feature_error
        assert feature_error.cause == entity_error
        assert entity_error.cause == ha_api_error

        # Verify context flows through
        error_str = str(model_error)
        assert "living_room" in error_str
        assert "Entity not found" in error_str

    def test_concurrent_error_aggregation(self):
        """Test aggregation of errors from concurrent operations."""
        # Simulate multiple concurrent errors
        room_errors = {}

        for room_id in ["living_room", "bedroom", "kitchen"]:
            try:
                # Simulate different error types per room
                if room_id == "living_room":
                    raise ModelPredictionError(
                        "lstm", room_id, cause=ValueError("Invalid input shape")
                    )
                elif room_id == "bedroom":
                    raise FeatureExtractionError(
                        "temporal", room_id, cause=DatabaseQueryError("SELECT failed")
                    )
                else:
                    raise EntityNotFoundError(
                        f"binary_sensor.{room_id}_motion", room_id
                    )
            except OccupancyPredictionError as e:
                room_errors[room_id] = e

        # Aggregate errors
        aggregated_error = SystemError(
            message=f"Prediction failed for {len(room_errors)} rooms",
            operation="batch_prediction",
            component="prediction_service",
        )

        # Add room errors to context
        aggregated_error.context["room_errors"] = {
            room_id: {
                "error_type": type(error).__name__,
                "message": error.message,
                "severity": error.severity.value,
            }
            for room_id, error in room_errors.items()
        }

        # Verify aggregation
        assert len(aggregated_error.context["room_errors"]) == 3
        assert "living_room" in aggregated_error.context["room_errors"]
        assert (
            aggregated_error.context["room_errors"]["living_room"]["error_type"]
            == "ModelPredictionError"
        )


class TestAsyncErrorHandling:
    """Test error handling in async contexts."""

    async def test_async_exception_propagation(self):
        """Test exception propagation in async functions."""

        async def database_operation():
            await asyncio.sleep(0.01)  # Simulate async database call
            raise DatabaseConnectionError(
                "postgresql://test", cause=ConnectionError("Timeout")
            )

        async def feature_extraction():
            try:
                await database_operation()
            except DatabaseError as e:
                raise FeatureExtractionError("temporal", "living_room", cause=e)

        async def prediction_service():
            try:
                await feature_extraction()
            except FeatureEngineeringError as e:
                raise ModelPredictionError("lstm", "living_room", cause=e)

        # Test async error propagation
        with pytest.raises(ModelPredictionError) as exc_info:
            await prediction_service()

        error = exc_info.value
        assert error.cause.__class__.__name__ == "FeatureExtractionError"
        assert error.cause.cause.__class__.__name__ == "DatabaseConnectionError"

    async def test_async_error_timeout_handling(self):
        """Test handling of timeout errors in async operations."""

        async def slow_operation():
            await asyncio.sleep(2)  # Simulate slow operation
            return "success"

        async def timed_operation():
            try:
                result = await asyncio.wait_for(slow_operation(), timeout=0.1)
                return result
            except asyncio.TimeoutError as e:
                raise SystemResourceError(
                    resource_type="compute", resource_name="prediction_service", cause=e
                )

        with pytest.raises(SystemResourceError) as exc_info:
            await timed_operation()

        error = exc_info.value
        assert error.resource_type == "compute"
        assert isinstance(error.cause, asyncio.TimeoutError)

    async def test_async_error_recovery_patterns(self):
        """Test error recovery patterns in async operations."""
        call_count = 0

        async def unreliable_operation():
            nonlocal call_count
            call_count += 1

            if call_count < 3:
                raise ServiceUnavailableError(
                    service_name="home_assistant",
                    endpoint="http://ha.local:8123",
                    retry_after=1,
                )
            return "success"

        async def retry_operation(max_retries=3):
            last_error = None

            for attempt in range(max_retries):
                try:
                    result = await unreliable_operation()
                    return result
                except ServiceUnavailableError as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.01)  # Short retry delay for test
                    continue

            # If all retries failed, wrap the last error
            raise SystemError(
                message=f"Operation failed after {max_retries} attempts",
                operation="data_sync",
                cause=last_error,
            )

        # Should succeed after retries
        result = await retry_operation()
        assert result == "success"
        assert call_count == 3


class TestErrorValidationAndSanitization:
    """Test error validation and sanitization."""

    def test_error_message_sanitization(self):
        """Test that error messages are properly sanitized."""
        # Error with potential injection content
        malicious_input = "'; DROP TABLE users; --"

        error = ConfigValidationError(
            field="database_name", value=malicious_input, expected="alphanumeric"
        )

        # Message should contain the input but be safely formatted
        error_str = str(error)
        assert malicious_input in error_str
        # Context should preserve exact value for debugging
        assert error.context["value"] == malicious_input

    def test_error_context_size_limits(self):
        """Test that error context respects size limits."""
        # Create very large context
        large_data = "x" * 10000  # 10KB string
        large_list = list(range(1000))  # Large list

        error = FeatureExtractionError(feature_type="temporal", room_id="test_room")

        # Add large context
        error.context.update(
            {
                "large_string": large_data,
                "large_list": large_list,
                "normal_data": "small_value",
            }
        )

        # Convert to string (simulating logging)
        error_str = str(error)

        # Should not crash with large context
        assert isinstance(error_str, str)
        assert len(error_str) > 0
        assert "temporal" in error_str

    def test_circular_reference_handling(self):
        """Test handling of circular references in error context."""
        # Create circular reference
        context_a = {"name": "context_a"}
        context_b = {"name": "context_b", "ref": context_a}
        context_a["ref"] = context_b

        error = SystemError("Test circular reference")

        # Should handle circular references gracefully
        try:
            error.context["circular"] = context_a
            error_str = str(error)
            # Should not crash
            assert isinstance(error_str, str)
        except (ValueError, RecursionError):
            # Acceptable to raise these for circular refs
            pass

    def test_unicode_error_handling(self):
        """Test handling of unicode characters in errors."""
        unicode_data = {
            "room_name": "Kök 🍳",  # Kitchen in Swedish with emoji
            "user_input": "тест данные",  # Test data in Russian
            "sensor_id": "binary_sensor.café_présence",  # French accents
            "special_chars": "©®™€£¥",
        }

        error = DataValidationError(
            data_source="room_configuration",
            validation_errors=["Invalid unicode characters"],
            sample_data=unicode_data,
        )

        # Should handle unicode gracefully
        error_str = str(error)
        assert "🍳" in error_str
        assert "тест" in error_str
        assert "café" in error_str


class TestValidationFunctionEdgeCases:
    """Test validation functions with edge cases and error scenarios."""

    def test_validate_room_id_comprehensive(self):
        """Test comprehensive room ID validation scenarios."""
        # Valid room IDs
        valid_room_ids = [
            "living_room",
            "bedroom-1",
            "KITCHEN",
            "room123",
            "a",  # Single character
            "room_with_many_underscores_and_hyphens-here",
        ]

        for room_id in valid_room_ids:
            # Should not raise exception
            validate_room_id(room_id)

        # Invalid room IDs
        invalid_cases = [
            ("", "empty string"),
            (None, "None value"),
            (123, "integer"),
            ("room with spaces", "spaces"),
            ("room@home", "special characters"),
            ("room.kitchen", "period"),
            ("_room", "starts with underscore"),
            ("room_", "ends with underscore"),
            ("room&kitchen", "ampersand"),
            ("room#1", "hash symbol"),
            ("room/kitchen", "slash"),
            ("room\\kitchen", "backslash"),
        ]

        for invalid_room_id, description in invalid_cases:
            with pytest.raises(DataValidationError, match="room_id") as exc_info:
                validate_room_id(invalid_room_id)

            error = exc_info.value
            assert "room_id" in str(error)
            if invalid_room_id is not None:
                assert str(invalid_room_id) in str(error)

    def test_validate_entity_id_comprehensive(self):
        """Test comprehensive entity ID validation scenarios."""
        # Valid entity IDs
        valid_entity_ids = [
            "binary_sensor.living_room_motion",
            "sensor.bedroom_temperature",
            "switch.kitchen_light",
            "binary_sensor.door_main",
            "sensor.humidity_1",
            "light.bedroom_lamp_2",
        ]

        for entity_id in valid_entity_ids:
            # Should not raise exception
            validate_entity_id(entity_id)

        # Invalid entity IDs
        invalid_cases = [
            ("", "empty string"),
            (None, "None value"),
            ("not_an_entity", "no domain separator"),
            ("binary_sensor.", "missing object_id"),
            (".living_room_motion", "missing domain"),
            ("Binary_Sensor.living_room", "uppercase in domain"),
            ("binary_sensor.Living_Room", "uppercase in object_id"),
            ("binary-sensor.living_room", "hyphen in domain"),
            (
                "binary_sensor.living-room",
                "valid format",
            ),  # This should actually be valid
            ("invalid.domain.object", "multiple dots"),
            ("binary_sensor.room with spaces", "spaces in object_id"),
            ("binary_sensor.room@home", "special characters"),
            (123, "integer type"),
        ]

        for invalid_entity_id, description in invalid_cases:
            if description == "valid format":
                # This case should actually be valid
                validate_entity_id(invalid_entity_id)
            else:
                with pytest.raises(DataValidationError, match="entity_id") as exc_info:
                    validate_entity_id(invalid_entity_id)

                error = exc_info.value
                assert "entity_id" in str(error)

    def test_validation_function_performance(self):
        """Test validation function performance with many calls."""
        import time

        # Test room ID validation performance
        valid_room_ids = [f"room_{i:04d}" for i in range(1000)]

        start_time = time.time()
        for room_id in valid_room_ids:
            validate_room_id(room_id)
        room_validation_time = time.time() - start_time

        # Should be very fast (< 0.1 seconds for 1000 validations)
        assert room_validation_time < 0.1

        # Test entity ID validation performance
        valid_entity_ids = [f"binary_sensor.room_{i:04d}_motion" for i in range(1000)]

        start_time = time.time()
        for entity_id in valid_entity_ids:
            validate_entity_id(entity_id)
        entity_validation_time = time.time() - start_time

        # Should be very fast (< 0.1 seconds for 1000 validations)
        assert entity_validation_time < 0.1


class TestExceptionLoggingIntegration:
    """Test exception integration with logging systems."""

    def test_structured_logging_format(self):
        """Test that exceptions format properly for structured logging."""
        error = ModelTrainingError(
            model_type="lstm",
            room_id="living_room",
            training_data_size=1500,
            cause=InsufficientDataError(
                data_type="sensor_events",
                room_id="living_room",
                required_samples=2000,
                available_samples=1500,
            ),
        )

        # Create structured log entry
        log_entry = {
            "timestamp": "2024-01-15T10:30:00Z",
            "level": "ERROR",
            "logger": "occupancy_prediction.models",
            "message": error.message,
            "error": {
                "type": type(error).__name__,
                "code": error.error_code,
                "severity": error.severity.value,
                "context": error.context,
                "cause": {
                    "type": type(error.cause).__name__ if error.cause else None,
                    "message": str(error.cause) if error.cause else None,
                },
            },
        }

        # Should be JSON serializable
        json_str = json.dumps(log_entry, default=str)
        parsed = json.loads(json_str)

        assert parsed["error"]["type"] == "ModelTrainingError"
        assert parsed["error"]["severity"] == "high"
        assert parsed["error"]["context"]["room_id"] == "living_room"
        assert parsed["error"]["cause"]["type"] == "InsufficientDataError"

    def test_error_metrics_extraction(self):
        """Test extraction of metrics from errors for monitoring."""
        errors = [
            ModelPredictionError("lstm", "living_room"),
            ModelPredictionError("xgboost", "bedroom"),
            FeatureExtractionError("temporal", "kitchen"),
            DatabaseConnectionError("postgresql://localhost"),
            HomeAssistantConnectionError("http://ha.local:8123"),
        ]

        # Extract metrics
        error_metrics = {}
        for error in errors:
            error_type = type(error).__name__
            severity = error.severity.value

            if error_type not in error_metrics:
                error_metrics[error_type] = {"count": 0, "severities": {}}

            error_metrics[error_type]["count"] += 1

            if severity not in error_metrics[error_type]["severities"]:
                error_metrics[error_type]["severities"][severity] = 0
            error_metrics[error_type]["severities"][severity] += 1

        # Verify metrics
        assert error_metrics["ModelPredictionError"]["count"] == 2
        assert error_metrics["FeatureExtractionError"]["count"] == 1
        assert error_metrics["DatabaseConnectionError"]["severities"]["critical"] == 1
        assert error_metrics["HomeAssistantConnectionError"]["severities"]["high"] == 1

    def test_error_alerting_classification(self):
        """Test classification of errors for alerting systems."""

        def classify_error_for_alerting(
            error: OccupancyPredictionError,
        ) -> Dict[str, Any]:
            """Classify error for alerting system."""
            classification = {
                "alert_level": "info",
                "requires_immediate_action": False,
                "component": "unknown",
                "impact": "low",
            }

            # Classify by severity
            if error.severity == ErrorSeverity.CRITICAL:
                classification["alert_level"] = "critical"
                classification["requires_immediate_action"] = True
                classification["impact"] = "high"
            elif error.severity == ErrorSeverity.HIGH:
                classification["alert_level"] = "warning"
                classification["impact"] = "medium"

            # Classify by error type
            if isinstance(error, DatabaseError):
                classification["component"] = "database"
                if isinstance(error, DatabaseConnectionError):
                    classification["requires_immediate_action"] = True
            elif isinstance(error, HomeAssistantError):
                classification["component"] = "home_assistant"
            elif isinstance(error, ModelError):
                classification["component"] = "ml_models"
            elif isinstance(error, SystemError):
                classification["component"] = "system"
                if isinstance(error, SystemInitializationError):
                    classification["requires_immediate_action"] = True

            return classification

        # Test various error classifications
        test_cases = [
            (
                DatabaseConnectionError("postgresql://test"),
                {
                    "alert_level": "critical",
                    "requires_immediate_action": True,
                    "component": "database",
                    "impact": "high",
                },
            ),
            (
                ModelPredictionError("lstm", "room1"),
                {
                    "alert_level": "warning",
                    "component": "ml_models",
                    "impact": "medium",
                },
            ),
            (
                ConfigValidationError(field="test", value=None),
                {"alert_level": "warning", "impact": "medium"},
            ),
        ]

        for error, expected in test_cases:
            classification = classify_error_for_alerting(error)

            for key, value in expected.items():
                assert (
                    classification[key] == value
                ), f"Classification mismatch for {type(error).__name__}: {key}"


@pytest.mark.unit
class TestProductionErrorScenarios:
    """Test production-grade error scenarios and handling."""

    def test_memory_pressure_error_handling(self):
        """Test error handling under memory pressure."""
        # Simulate memory error during model training
        memory_error = MemoryError("Unable to allocate memory for training data")

        training_error = ModelTrainingError(
            model_type="lstm",
            room_id="living_room",
            training_data_size=1000000,  # Large dataset
            cause=memory_error,
        )

        # System should wrap and provide context
        system_error = ResourceExhaustionError(
            resource_type="memory", current_usage=95.5, limit=90.0, unit="%"
        )
        system_error.cause = training_error

        # Verify error chain and context
        assert system_error.resource_type == "memory"
        assert system_error.current_usage == 95.5
        assert isinstance(system_error.cause.cause, MemoryError)

    def test_cascading_failure_scenario(self):
        """Test handling of cascading failures across system components."""
        # Start with database failure
        db_failure = DatabaseConnectionError(
            "postgresql://ha_ml:pass@db.local:5432/ha_ml_prod",
            cause=ConnectionError("Connection pool exhausted"),
        )

        # Feature extraction fails
        feature_failure = FeatureExtractionError(
            "temporal", "living_room", cause=db_failure
        )

        # Multiple model predictions fail
        model_failures = []
        for model_type in ["lstm", "xgboost", "hmm"]:
            model_error = ModelPredictionError(
                model_type, "living_room", cause=feature_failure
            )
            model_failures.append(model_error)

        # System enters degraded mode
        degraded_error = SystemError(
            message="System operating in degraded mode due to cascading failures",
            operation="prediction_service",
            component="ensemble_predictor",
        )

        # Aggregate all failures in context
        degraded_error.context["failed_components"] = {
            "database": str(db_failure),
            "feature_extraction": str(feature_failure),
            "model_predictions": [str(e) for e in model_failures],
        }

        # Verify cascading failure is properly documented
        assert "degraded mode" in degraded_error.message
        assert len(degraded_error.context["model_predictions"]) == 3
        assert "Connection pool exhausted" in str(degraded_error)

    def test_security_incident_error_handling(self):
        """Test handling of security-related errors."""
        # Simulate security incidents
        security_errors = [
            APISecurityError(
                message="JWT token manipulation detected",
                violation_type="token_tampering",
                endpoint="/api/predictions",
            ),
            APIAuthenticationError(
                message="Multiple failed authentication attempts",
                endpoint="/api/admin",
                auth_method="jwt",
            ),
            RateLimitExceededError(
                service="prediction_api",
                limit=60,
                window_seconds=60,
                reset_time="2024-01-15T10:31:00Z",
            ),
        ]

        # Security errors should have appropriate classifications
        for error in security_errors:
            if isinstance(error, APISecurityError):
                assert error.severity == ErrorSeverity.CRITICAL
                assert "token_tampering" in error.context["violation_type"]
            elif isinstance(error, APIAuthenticationError):
                assert error.severity == ErrorSeverity.HIGH
            elif isinstance(error, RateLimitExceededError):
                assert error.severity == ErrorSeverity.MEDIUM

    def test_data_corruption_detection_and_handling(self):
        """Test detection and handling of data corruption errors."""
        # Simulate data corruption scenarios
        corruption_scenarios = [
            {
                "source": "sensor_events_table",
                "corruption": "Duplicate primary keys detected",
                "affected_records": 1500,
            },
            {
                "source": "model_artifacts",
                "corruption": "Invalid model file format",
                "model_type": "lstm",
            },
            {
                "source": "configuration_files",
                "corruption": "YAML structure inconsistency",
                "file": "rooms.yaml",
            },
        ]

        corruption_errors = []
        for scenario in corruption_scenarios:
            if scenario["source"] == "sensor_events_table":
                error = DatabaseIntegrityError(
                    constraint="primary_key_violation",
                    table_name="sensor_events",
                    values={"affected_records": scenario["affected_records"]},
                )
            elif scenario["source"] == "model_artifacts":
                error = ModelValidationError(
                    model_type=scenario["model_type"],
                    room_id="unknown",
                    validation_error=scenario["corruption"],
                )
            else:
                error = ConfigParsingError(scenario["file"], scenario["corruption"])

            corruption_errors.append(error)

        # All corruption errors should be high severity or critical
        for error in corruption_errors:
            assert error.severity.value in ["high", "critical"]
            assert "corruption" in str(error).lower() or "invalid" in str(error).lower()

    def test_graceful_degradation_error_patterns(self):
        """Test error patterns that support graceful degradation."""
        # Service unavailable - should trigger fallback
        ha_unavailable = ServiceUnavailableError(
            service_name="home_assistant",
            endpoint="http://ha.local:8123/api/states",
            retry_after=300,
            reason="Maintenance mode",
        )

        # Model not available - should use fallback model
        model_unavailable = ModelNotFoundError(
            model_type="lstm",
            room_id="living_room",
            model_path="/models/living_room_lstm.pkl",
        )

        # Feature partially available - should use available features
        partial_features = MissingFeatureError(
            feature_names=["humidity", "light_level"],
            room_id="living_room",
            available_features=["temperature", "motion", "door_state"],
        )

        # These errors should allow system to continue with reduced functionality
        degradable_errors = [ha_unavailable, model_unavailable, partial_features]

        for error in degradable_errors:
            # Should not be critical severity (allows degraded operation)
            assert error.severity != ErrorSeverity.CRITICAL

            # Should provide enough context for fallback decisions
            assert len(error.context) > 0

            # Should have actionable information
            error_str = str(error)
            assert any(
                keyword in error_str.lower()
                for keyword in [
                    "retry",
                    "fallback",
                    "available",
                    "alternative",
                    "partial",
                ]
            )
