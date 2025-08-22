"""
Comprehensive Schema Validation and Data Format Verification.

This module provides robust schema validation for:
- JSON schema validation with custom validators
- Database schema consistency checking
- API request/response format validation
- Cross-platform data format standardization
- Version compatibility validation
"""

# asyncio available for future async operations
# import asyncio
from dataclasses import dataclass, field
from datetime import datetime

# timezone available for future use
# from datetime import timezone
import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid

try:
    from jsonschema import (  # validate,  # Available for future use
        Draft7Validator,
        ValidationError as JsonSchemaValidationError,
    )

    JSONSCHEMA_AVAILABLE = True
except ImportError:
    # Fallback for when jsonschema is not available
    JSONSCHEMA_AVAILABLE = False

    class JsonSchemaValidationError(Exception):
        pass

    class Draft7Validator:
        def __init__(self, schema):
            self.schema = schema

        def iter_errors(self, data):
            return []


from sqlalchemy import inspect, text
from sqlalchemy.ext.asyncio import AsyncSession

# RoomConfig, SystemConfig, and get_config available for future use
# from ...core.config import RoomConfig, SystemConfig, get_config
from ...core.constants import SensorState, SensorType
from ...core.exceptions import (
    ErrorSeverity,
)

# ConfigurationError, DatabaseError, DataValidationError available for future use
from .event_validator import ValidationError, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class SchemaDefinition:
    """Represents a schema definition with metadata."""

    schema_id: str
    name: str
    version: str
    schema: Dict[str, Any]
    description: str = ""
    tags: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    validators: List[Callable] = field(default_factory=list)


@dataclass
class SchemaValidationContext:
    """Context for schema validation operations."""

    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    strict_mode: bool = True
    allow_additional_properties: bool = False
    custom_formats: Dict[str, Callable] = field(default_factory=dict)
    validation_metadata: Dict[str, Any] = field(default_factory=dict)


class JSONSchemaValidator:
    """Advanced JSON schema validation with custom formats and validators."""

    def __init__(self):
        """Initialize JSON schema validator."""
        self.schemas = {}
        self.custom_validators = {}
        self.format_checkers = self._initialize_format_checkers()
        self._load_standard_schemas()

    def _initialize_format_checkers(self) -> Dict[str, Callable]:
        """Initialize custom format checkers."""
        return {
            "sensor_id": self._validate_sensor_id_format,
            "room_id": self._validate_room_id_format,
            "entity_id": self._validate_entity_id_format,
            "iso_datetime": self._validate_iso_datetime_format,
            "sensor_state": self._validate_sensor_state_format,
            "sensor_type": self._validate_sensor_type_format,
            "coordinate": self._validate_coordinate_format,
            "duration": self._validate_duration_format,
        }

    def _load_standard_schemas(self):
        """Load standard schemas for common data types."""
        # Sensor Event Schema
        self.schemas["sensor_event"] = SchemaDefinition(
            schema_id="sensor_event",
            name="Sensor Event",
            version="1.0.0",
            description="Schema for Home Assistant sensor events",
            schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": [
                    "room_id",
                    "sensor_id",
                    "sensor_type",
                    "state",
                    "timestamp",
                ],
                "properties": {
                    "room_id": {
                        "type": "string",
                        "format": "room_id",
                        "minLength": 1,
                        "maxLength": 100,
                        "pattern": "^[a-zA-Z0-9_-]+$",
                    },
                    "sensor_id": {
                        "type": "string",
                        "format": "sensor_id",
                        "minLength": 1,
                        "maxLength": 200,
                    },
                    "sensor_type": {"type": "string", "format": "sensor_type"},
                    "state": {"type": "string", "format": "sensor_state"},
                    "previous_state": {
                        "type": ["string", "null"],
                        "format": "sensor_state",
                    },
                    "timestamp": {"type": "string", "format": "iso_datetime"},
                    "attributes": {
                        "type": ["object", "null"],
                        "additionalProperties": True,
                    },
                    "is_human_triggered": {"type": ["boolean", "null"]},
                    "confidence_score": {
                        "type": ["number", "null"],
                        "minimum": 0,
                        "maximum": 1,
                    },
                    "correlation_id": {"type": ["string", "null"], "format": "uuid"},
                },
                "additionalProperties": False,
            },
        )

        # Room Configuration Schema
        self.schemas["room_config"] = SchemaDefinition(
            schema_id="room_config",
            name="Room Configuration",
            version="1.0.0",
            description="Schema for room configuration data",
            schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["room_id", "name"],
                "properties": {
                    "room_id": {"type": "string", "format": "room_id"},
                    "name": {"type": "string", "minLength": 1, "maxLength": 100},
                    "sensors": {
                        "type": "object",
                        "patternProperties": {
                            "^[a-zA-Z0-9_]+$": {
                                "type": "object",
                                "properties": {
                                    "entities": {
                                        "type": "array",
                                        "items": {
                                            "type": "string",
                                            "format": "entity_id",
                                        },
                                    },
                                    "timeout_minutes": {
                                        "type": "number",
                                        "minimum": 1,
                                        "maximum": 1440,
                                    },
                                },
                            }
                        },
                    },
                    "coordinates": {
                        "type": ["object", "null"],
                        "properties": {
                            "x": {"type": "number", "format": "coordinate"},
                            "y": {"type": "number", "format": "coordinate"},
                            "z": {"type": "number", "format": "coordinate"},
                        },
                    },
                    "metadata": {
                        "type": ["object", "null"],
                        "additionalProperties": True,
                    },
                },
                "additionalProperties": False,
            },
        )

        # API Request Schema
        self.schemas["api_request"] = SchemaDefinition(
            schema_id="api_request",
            name="API Request",
            version="1.0.0",
            description="Schema for API request validation",
            schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "properties": {
                    "method": {
                        "type": "string",
                        "enum": ["GET", "POST", "PUT", "DELETE", "PATCH"],
                    },
                    "path": {"type": "string", "pattern": "^/api/[a-zA-Z0-9_/-]*$"},
                    "headers": {
                        "type": "object",
                        "patternProperties": {"^[A-Za-z0-9-]+$": {"type": "string"}},
                    },
                    "query_params": {
                        "type": "object",
                        "additionalProperties": {"type": "string"},
                    },
                    "body": {"type": ["object", "null"]},
                },
            },
        )

    def _validate_sensor_id_format(self, value: str) -> bool:
        """Validate sensor ID format."""
        if not isinstance(value, str):
            return False
        # Home Assistant entity ID format: domain.object_id
        pattern = r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$"
        return re.match(pattern, value) is not None

    def _validate_room_id_format(self, value: str) -> bool:
        """Validate room ID format."""
        if not isinstance(value, str):
            return False
        return re.match(r"^[a-zA-Z0-9_-]+$", value) is not None and len(value) <= 100

    def _validate_entity_id_format(self, value: str) -> bool:
        """Validate Home Assistant entity ID format."""
        if not isinstance(value, str):
            return False
        pattern = r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$"
        return re.match(pattern, value) is not None

    def _validate_iso_datetime_format(self, value: str) -> bool:
        """Validate ISO datetime format."""
        if not isinstance(value, str):
            return False
        try:
            datetime.fromisoformat(value.replace("Z", "+00:00"))
            return True
        except ValueError:
            return False

    def _validate_sensor_state_format(self, value: str) -> bool:
        """Validate sensor state format."""
        if not isinstance(value, str):
            return False
        valid_states = {state.value for state in SensorState}
        return value in valid_states

    def _validate_sensor_type_format(self, value: str) -> bool:
        """Validate sensor type format."""
        if not isinstance(value, str):
            return False
        valid_types = {sensor_type.value for sensor_type in SensorType}
        return value in valid_types

    def _validate_coordinate_format(self, value: Union[int, float]) -> bool:
        """Validate coordinate format."""
        return isinstance(value, (int, float)) and -180 <= value <= 180

    def _validate_duration_format(self, value: str) -> bool:
        """Validate duration format (ISO 8601)."""
        if not isinstance(value, str):
            return False
        pattern = r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
        return re.match(pattern, value) is not None

    def validate_json_schema(
        self,
        data: Any,
        schema_id: str,
        context: Optional[SchemaValidationContext] = None,
    ) -> ValidationResult:
        """Validate data against a JSON schema."""
        if context is None:
            context = SchemaValidationContext()

        errors = []
        warnings = []

        if schema_id not in self.schemas:
            return ValidationResult(
                is_valid=False,
                errors=[
                    ValidationError(
                        rule_id="SCH_JSON_001",
                        field="schema_id",
                        value=schema_id,
                        message=f"Schema '{schema_id}' not found",
                        severity=ErrorSeverity.HIGH,
                        suggestion="Use a valid schema identifier",
                    )
                ],
            )

        schema_def = self.schemas[schema_id]
        schema = schema_def.schema.copy()

        # Modify schema based on context
        if not context.allow_additional_properties:
            schema["additionalProperties"] = False

        try:
            # Create validator with custom format checkers
            validator = Draft7Validator(schema)

            # Add custom format checkers
            for format_name, checker in self.format_checkers.items():
                validator.format_checker.checks(format_name)(checker)

            # Validate the data
            validation_errors = list(validator.iter_errors(data))

            for error in validation_errors:
                severity = ErrorSeverity.HIGH
                if error.validator in ["additionalProperties", "format"]:
                    severity = ErrorSeverity.MEDIUM
                elif error.validator in ["required"]:
                    severity = ErrorSeverity.CRITICAL

                errors.append(
                    ValidationError(
                        rule_id=f"SCH_JSON_{error.validator.upper()}",
                        field=".".join(str(p) for p in error.absolute_path),
                        value=error.instance,
                        message=error.message,
                        severity=severity,
                        suggestion=self._get_validation_suggestion(error),
                        context={
                            "schema_path": ".".join(str(p) for p in error.schema_path),
                            "validator": error.validator,
                            "schema_id": schema_id,
                        },
                    )
                )

            # Run custom validators
            for validator_func in schema_def.validators:
                try:
                    custom_errors = validator_func(data)
                    errors.extend(custom_errors)
                except Exception as e:
                    warnings.append(
                        ValidationError(
                            rule_id="SCH_JSON_CUSTOM",
                            field="custom_validator",
                            value=str(e),
                            message=f"Custom validator error: {e}",
                            severity=ErrorSeverity.LOW,
                            suggestion="Check custom validator implementation",
                        )
                    )

        except Exception as e:
            errors.append(
                ValidationError(
                    rule_id="SCH_JSON_ERROR",
                    field="validation_process",
                    value=str(e),
                    message=f"Schema validation error: {e}",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Check schema definition and data format",
                )
            )

        confidence_score = 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1)
        confidence_score = max(0.0, confidence_score)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score,
            validation_id=context.validation_id,
        )

    def _get_validation_suggestion(self, error: JsonSchemaValidationError) -> str:
        """Get helpful suggestion for validation error."""
        validator = error.validator

        if validator == "required":
            missing_props = (
                error.message.split("'")[1] if "'" in error.message else "field"
            )
            return f"Add the required property '{missing_props}' to the object"
        elif validator == "type":
            expected_type = error.schema.get("type", "unknown")
            return f"Change the value to type '{expected_type}'"
        elif validator == "format":
            format_name = error.schema.get("format", "unknown")
            return f"Ensure the value matches the '{format_name}' format"
        elif validator == "minLength":
            min_len = error.schema.get("minLength", 0)
            return f"Ensure the string has at least {min_len} characters"
        elif validator == "maxLength":
            max_len = error.schema.get("maxLength", 0)
            return f"Ensure the string has no more than {max_len} characters"
        elif validator == "minimum":
            min_val = error.schema.get("minimum", 0)
            return f"Ensure the value is at least {min_val}"
        elif validator == "maximum":
            max_val = error.schema.get("maximum", 0)
            return f"Ensure the value is no more than {max_val}"
        elif validator == "pattern":
            pattern = error.schema.get("pattern", "")
            return f"Ensure the value matches the pattern: {pattern}"
        elif validator == "enum":
            enum_values = error.schema.get("enum", [])
            return f"Use one of the allowed values: {', '.join(map(str, enum_values))}"
        else:
            return "Check the value format and try again"

    def register_schema(self, schema_def: SchemaDefinition):
        """Register a new schema definition."""
        self.schemas[schema_def.schema_id] = schema_def
        logger.info(f"Registered schema: {schema_def.name} (v{schema_def.version})")

    def get_schema_info(self, schema_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a registered schema."""
        if schema_id not in self.schemas:
            return None

        schema_def = self.schemas[schema_id]
        return {
            "schema_id": schema_def.schema_id,
            "name": schema_def.name,
            "version": schema_def.version,
            "description": schema_def.description,
            "tags": schema_def.tags,
            "created_at": schema_def.created_at.isoformat(),
            "custom_validators": len(schema_def.validators),
        }


class DatabaseSchemaValidator:
    """Validates database schema consistency and integrity."""

    def __init__(self, session: AsyncSession):
        """Initialize database schema validator."""
        self.session = session
        self.expected_tables = {
            "sensor_events",
            "room_states",
            "predictions",
            "model_metadata",
            "feature_cache",
        }
        self.expected_columns = {
            "sensor_events": {
                "id",
                "room_id",
                "sensor_id",
                "sensor_type",
                "state",
                "previous_state",
                "timestamp",
                "attributes",
                "is_human_triggered",
            },
            "room_states": {
                "id",
                "room_id",
                "occupancy_state",
                "confidence",
                "timestamp",
                "sensor_count",
                "last_motion",
                "attributes",
            },
            "predictions": {
                "id",
                "room_id",
                "model_type",
                "predicted_time",
                "confidence",
                "prediction_type",
                "features_hash",
                "model_version",
                "created_at",
            },
        }

    async def validate_database_schema(self) -> ValidationResult:
        """Validate the database schema against expected structure."""
        errors = []
        warnings = []

        try:
            # Get database metadata
            inspector = inspect(self.session.bind)
            existing_tables = set(inspector.get_table_names())

            # Check for missing tables
            missing_tables = self.expected_tables - existing_tables
            for table in missing_tables:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_001",
                        field="table",
                        value=table,
                        message=f"Missing required table: {table}",
                        severity=ErrorSeverity.CRITICAL,
                        suggestion=f"Create the {table} table using migration scripts",
                    )
                )

            # Check for unexpected tables (warnings only)
            unexpected_tables = existing_tables - self.expected_tables
            for table in unexpected_tables:
                if not table.startswith("_"):  # Ignore system tables
                    warnings.append(
                        ValidationError(
                            rule_id="DB_SCH_002",
                            field="table",
                            value=table,
                            message=f"Unexpected table found: {table}",
                            severity=ErrorSeverity.LOW,
                            suggestion="Review if this table is needed",
                        )
                    )

            # Check column structure for existing tables
            for table in existing_tables & self.expected_tables:
                if table in self.expected_columns:
                    table_errors = await self._validate_table_columns(
                        inspector, table, self.expected_columns[table]
                    )
                    errors.extend(table_errors)

            # Check indexes and constraints
            index_errors = await self._validate_indexes(inspector)
            errors.extend(index_errors)

            # Check TimescaleDB specific features
            timescale_errors = await self._validate_timescaledb_features()
            errors.extend(timescale_errors)

        except Exception as e:
            errors.append(
                ValidationError(
                    rule_id="DB_SCH_ERROR",
                    field="database_schema",
                    value=str(e),
                    message=f"Database schema validation error: {e}",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Check database connectivity and permissions",
                )
            )

        confidence_score = 1.0 - (len(errors) * 0.15 + len(warnings) * 0.05)
        confidence_score = max(0.0, confidence_score)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score,
        )

    async def _validate_table_columns(
        self, inspector, table_name: str, expected_columns: Set[str]
    ) -> List[ValidationError]:
        """Validate columns for a specific table."""
        errors = []

        try:
            existing_columns = {
                col["name"] for col in inspector.get_columns(table_name)
            }

            # Check for missing columns
            missing_columns = expected_columns - existing_columns
            for column in missing_columns:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_003",
                        field="column",
                        value=f"{table_name}.{column}",
                        message=f"Missing required column: {table_name}.{column}",
                        severity=ErrorSeverity.HIGH,
                        suggestion=f"Add the {column} column to {table_name} table",
                    )
                )

            # Validate column types for critical columns
            column_info = {
                col["name"]: col for col in inspector.get_columns(table_name)
            }

            if table_name == "sensor_events":
                errors.extend(self._validate_sensor_events_columns(column_info))
            elif table_name == "room_states":
                errors.extend(self._validate_room_states_columns(column_info))
            elif table_name == "predictions":
                errors.extend(self._validate_predictions_columns(column_info))

        except Exception as e:
            errors.append(
                ValidationError(
                    rule_id="DB_SCH_004",
                    field="table_validation",
                    value=f"{table_name}: {e}",
                    message=f"Error validating table {table_name}: {e}",
                    severity=ErrorSeverity.MEDIUM,
                    suggestion=f"Check {table_name} table structure",
                )
            )

        return errors

    def _validate_sensor_events_columns(
        self, column_info: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate sensor_events table column types."""
        errors = []

        # Check timestamp column
        if "timestamp" in column_info:
            col_type = str(column_info["timestamp"]["type"]).lower()
            if "timestamp" not in col_type and "datetime" not in col_type:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_005",
                        field="column_type",
                        value=f"sensor_events.timestamp: {col_type}",
                        message="timestamp column should be TIMESTAMP type",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Change timestamp column to TIMESTAMP WITH TIME ZONE",
                    )
                )

        # Check JSON attributes column
        if "attributes" in column_info:
            col_type = str(column_info["attributes"]["type"]).lower()
            if "json" not in col_type and "text" not in col_type:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_006",
                        field="column_type",
                        value=f"sensor_events.attributes: {col_type}",
                        message="attributes column should be JSON or TEXT type",
                        severity=ErrorSeverity.LOW,
                        suggestion="Consider using JSON type for attributes column",
                    )
                )

        return errors

    def _validate_room_states_columns(
        self, column_info: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate room_states table column types."""
        errors = []

        # Check confidence column
        if "confidence" in column_info:
            col_type = str(column_info["confidence"]["type"]).lower()
            if (
                "float" not in col_type
                and "decimal" not in col_type
                and "numeric" not in col_type
            ):
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_007",
                        field="column_type",
                        value=f"room_states.confidence: {col_type}",
                        message="confidence column should be FLOAT or DECIMAL type",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Change confidence column to FLOAT type",
                    )
                )

        return errors

    def _validate_predictions_columns(
        self, column_info: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate predictions table column types."""
        errors = []

        # Check predicted_time column
        if "predicted_time" in column_info:
            col_type = str(column_info["predicted_time"]["type"]).lower()
            if "timestamp" not in col_type and "datetime" not in col_type:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_008",
                        field="column_type",
                        value=f"predictions.predicted_time: {col_type}",
                        message="predicted_time column should be TIMESTAMP type",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Change predicted_time column to TIMESTAMP WITH TIME ZONE",
                    )
                )

        return errors

    async def _validate_indexes(self, inspector) -> List[ValidationError]:
        """Validate critical database indexes."""
        errors = []

        expected_indexes = {
            "sensor_events": ["idx_room_sensor_time", "idx_room_state_changes"],
            "room_states": ["idx_room_timestamp"],
            "predictions": ["idx_room_predicted_time"],
        }

        for table, expected_idx_list in expected_indexes.items():
            try:
                existing_indexes = {idx["name"] for idx in inspector.get_indexes(table)}

                for expected_idx in expected_idx_list:
                    if expected_idx not in existing_indexes:
                        errors.append(
                            ValidationError(
                                rule_id="DB_SCH_009",
                                field="index",
                                value=f"{table}.{expected_idx}",
                                message=f"Missing critical index: {expected_idx} on {table}",
                                severity=ErrorSeverity.MEDIUM,
                                suggestion=f"Create index {expected_idx} for better query performance",
                            )
                        )

            except Exception as e:
                logger.warning(f"Could not validate indexes for {table}: {e}")

        return errors

    async def _validate_timescaledb_features(self) -> List[ValidationError]:
        """Validate TimescaleDB specific features."""
        errors = []

        try:
            # Check if TimescaleDB extension is available
            result = await self.session.execute(
                text("SELECT * FROM pg_extension WHERE extname = 'timescaledb'")
            )
            timescale_installed = result.fetchone() is not None

            if not timescale_installed:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_010",
                        field="timescaledb",
                        value="extension",
                        message="TimescaleDB extension not installed",
                        severity=ErrorSeverity.HIGH,
                        suggestion="Install TimescaleDB extension for time-series optimization",
                    )
                )
                return errors

            # Check if sensor_events is a hypertable
            result = await self.session.execute(
                text(
                    "SELECT * FROM _timescaledb_catalog.hypertable WHERE table_name = 'sensor_events'"
                )
            )
            is_hypertable = result.fetchone() is not None

            if not is_hypertable:
                errors.append(
                    ValidationError(
                        rule_id="DB_SCH_011",
                        field="hypertable",
                        value="sensor_events",
                        message="sensor_events table is not a TimescaleDB hypertable",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Convert sensor_events to hypertable for time-series optimization",
                    )
                )

        except Exception as e:
            logger.warning(f"Could not validate TimescaleDB features: {e}")

        return errors


class APISchemaValidator:
    """Validates API request and response schemas."""

    def __init__(self):
        """Initialize API schema validator."""
        self.json_validator = JSONSchemaValidator()
        self.content_type_validators = {
            "application/json": self._validate_json_content,
            "application/x-www-form-urlencoded": self._validate_form_content,
            "multipart/form-data": self._validate_multipart_content,
        }

    def validate_api_request(
        self,
        method: str,
        path: str,
        headers: Dict[str, str],
        body: Any = None,
        query_params: Dict[str, str] = None,
    ) -> ValidationResult:
        """Validate an API request."""
        errors = []
        warnings = []

        # Validate HTTP method
        valid_methods = {"GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"}
        if method.upper() not in valid_methods:
            errors.append(
                ValidationError(
                    rule_id="API_SCH_001",
                    field="method",
                    value=method,
                    message=f"Invalid HTTP method: {method}",
                    severity=ErrorSeverity.HIGH,
                    suggestion=f"Use one of: {', '.join(valid_methods)}",
                )
            )

        # Validate path format
        if not path.startswith("/"):
            errors.append(
                ValidationError(
                    rule_id="API_SCH_002",
                    field="path",
                    value=path,
                    message="API path must start with '/'",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Add leading '/' to the path",
                )
            )

        # Validate headers
        header_errors = self._validate_headers(headers)
        errors.extend(header_errors)

        # Validate body based on content type
        if body is not None:
            content_type = headers.get("Content-Type", "").split(";")[0].strip()
            if content_type in self.content_type_validators:
                body_errors = self.content_type_validators[content_type](body)
                errors.extend(body_errors)
            elif content_type and not content_type.startswith("text/"):
                warnings.append(
                    ValidationError(
                        rule_id="API_SCH_003",
                        field="content_type",
                        value=content_type,
                        message=f"Unsupported content type: {content_type}",
                        severity=ErrorSeverity.LOW,
                        suggestion="Use application/json for structured data",
                    )
                )

        # Validate query parameters
        if query_params:
            query_errors = self._validate_query_params(query_params)
            errors.extend(query_errors)

        confidence_score = 1.0 - (len(errors) * 0.2 + len(warnings) * 0.1)
        confidence_score = max(0.0, confidence_score)

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score,
        )

    def _validate_headers(self, headers: Dict[str, str]) -> List[ValidationError]:
        """Validate HTTP headers."""
        errors = []

        # Check for security headers
        security_headers = ["Authorization", "X-API-Key"]
        has_auth = any(header in headers for header in security_headers)

        if not has_auth:
            errors.append(
                ValidationError(
                    rule_id="API_SCH_004",
                    field="headers",
                    value="missing_auth",
                    message="No authentication header found",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Include Authorization or X-API-Key header",
                )
            )

        # Validate header format
        for name, value in headers.items():
            if not isinstance(name, str) or not isinstance(value, str):
                errors.append(
                    ValidationError(
                        rule_id="API_SCH_005",
                        field="headers",
                        value=f"{name}: {value}",
                        message="Header names and values must be strings",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Convert header values to strings",
                    )
                )

            # Check for suspicious header values
            if len(value) > 8192:  # HTTP header value limit
                errors.append(
                    ValidationError(
                        rule_id="API_SCH_006",
                        field="headers",
                        value=f"{name}: {value[:50]}...",
                        message=f"Header value too long: {name}",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Reduce header value size",
                    )
                )

        return errors

    def _validate_json_content(self, body: Any) -> List[ValidationError]:
        """Validate JSON content."""
        errors = []

        try:
            if isinstance(body, str):
                json.loads(body)
            elif not isinstance(body, (dict, list)):
                errors.append(
                    ValidationError(
                        rule_id="API_SCH_007",
                        field="body",
                        value=type(body).__name__,
                        message="JSON body must be object or array",
                        severity=ErrorSeverity.HIGH,
                        suggestion="Use dict or list for JSON content",
                    )
                )
        except json.JSONDecodeError as e:
            errors.append(
                ValidationError(
                    rule_id="API_SCH_008",
                    field="body",
                    value=str(e),
                    message=f"Invalid JSON format: {e}",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Fix JSON syntax errors",
                )
            )

        return errors

    def _validate_form_content(self, body: Any) -> List[ValidationError]:
        """Validate form-encoded content."""
        errors = []

        if not isinstance(body, (dict, str)):
            errors.append(
                ValidationError(
                    rule_id="API_SCH_009",
                    field="body",
                    value=type(body).__name__,
                    message="Form data must be dict or string",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Use dict for form-encoded data",
                )
            )

        return errors

    def _validate_multipart_content(self, body: Any) -> List[ValidationError]:
        """Validate multipart content."""
        errors = []

        # Basic validation for multipart data
        if not isinstance(body, (dict, bytes)):
            errors.append(
                ValidationError(
                    rule_id="API_SCH_010",
                    field="body",
                    value=type(body).__name__,
                    message="Multipart data must be dict or bytes",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Use appropriate multipart encoding",
                )
            )

        return errors

    def _validate_query_params(
        self, query_params: Dict[str, str]
    ) -> List[ValidationError]:
        """Validate query parameters."""
        errors = []

        for name, value in query_params.items():
            # Check parameter name format
            if not re.match(r"^[a-zA-Z0-9_-]+$", name):
                errors.append(
                    ValidationError(
                        rule_id="API_SCH_011",
                        field="query_params",
                        value=f"{name}={value}",
                        message=f"Invalid query parameter name: {name}",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Use alphanumeric characters, underscores, and hyphens only",
                    )
                )

            # Check for excessively long values
            if len(str(value)) > 2048:
                errors.append(
                    ValidationError(
                        rule_id="API_SCH_012",
                        field="query_params",
                        value=f"{name}={str(value)[:50]}...",
                        message=f"Query parameter value too long: {name}",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Reduce query parameter value size",
                    )
                )

        return errors
