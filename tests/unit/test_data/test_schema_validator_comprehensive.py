"""
Comprehensive test suite for SchemaValidator with schema enforcement testing.

This test suite provides complete coverage of schema validation functionality including:
- JSON schema validation with custom formats and validators
- Database schema consistency checking and validation
- API request/response format validation
- Schema registration and management
- Custom format checkers and validation rules
- Edge cases and error handling scenarios
"""

import asyncio
from datetime import datetime, timezone
import json
from unittest.mock import AsyncMock, MagicMock, patch
import uuid

import pytest

from src.core.constants import SensorState, SensorType
from src.core.exceptions import ErrorSeverity
from src.data.validation.event_validator import ValidationError, ValidationResult
from src.data.validation.schema_validator import (
    APISchemaValidator,
    DatabaseSchemaValidator,
    JSONSchemaValidator,
    SchemaDefinition,
    SchemaValidationContext,
)


@pytest.fixture
def json_schema_validator():
    """Create JSONSchemaValidator instance for testing."""
    return JSONSchemaValidator()


@pytest.fixture
def mock_db_session():
    """Create mock database session for testing."""
    session = MagicMock()
    session.bind = MagicMock()
    session.execute = AsyncMock()
    return session


@pytest.fixture
def database_schema_validator(mock_db_session):
    """Create DatabaseSchemaValidator instance for testing."""
    return DatabaseSchemaValidator(mock_db_session)


@pytest.fixture
def api_schema_validator():
    """Create APISchemaValidator instance for testing."""
    return APISchemaValidator()


@pytest.fixture
def valid_sensor_event():
    """Create valid sensor event data for testing."""
    return {
        "room_id": "living_room",
        "sensor_id": "binary_sensor.motion_detector",
        "sensor_type": "motion",
        "state": "on",
        "previous_state": "off",
        "timestamp": "2024-01-15T10:30:00Z",
        "attributes": {"brightness": 100, "temperature": 20.5},
        "is_human_triggered": True,
        "confidence_score": 0.85,
    }


@pytest.fixture
def valid_room_config():
    """Create valid room configuration data for testing."""
    return {
        "room_id": "living_room",
        "name": "Living Room",
        "sensors": {
            "motion": {
                "entities": ["binary_sensor.motion_detector"],
                "timeout_minutes": 5,
            },
            "door": {"entities": ["binary_sensor.front_door"], "timeout_minutes": 10},
        },
        "coordinates": {"x": 0.0, "y": 0.0, "z": 0.0},
        "metadata": {"floor": 1, "area_sqft": 200},
    }


class TestJSONSchemaValidator:
    """Comprehensive tests for JSONSchemaValidator."""

    def test_initialization(self, json_schema_validator):
        """Test JSONSchemaValidator initialization."""
        assert len(json_schema_validator.schemas) > 0
        assert len(json_schema_validator.format_checkers) > 0
        assert len(json_schema_validator.custom_validators) == 0  # Initially empty

        # Verify standard schemas are loaded
        assert "sensor_event" in json_schema_validator.schemas
        assert "room_config" in json_schema_validator.schemas
        assert "api_request" in json_schema_validator.schemas

    def test_format_checker_initialization(self, json_schema_validator):
        """Test that format checkers are properly initialized."""
        expected_formats = {
            "sensor_id",
            "room_id",
            "entity_id",
            "iso_datetime",
            "sensor_state",
            "sensor_type",
            "coordinate",
            "duration",
        }

        actual_formats = set(json_schema_validator.format_checkers.keys())
        assert expected_formats.issubset(actual_formats)

        # Test that all format checkers are callable
        for format_name, checker in json_schema_validator.format_checkers.items():
            assert callable(checker), f"Format checker {format_name} is not callable"

    def test_sensor_id_format_validation(self, json_schema_validator):
        """Test sensor ID format validation."""
        validator = json_schema_validator._validate_sensor_id_format

        # Valid sensor IDs
        valid_ids = [
            "binary_sensor.motion_detector",
            "sensor.temperature_living_room",
            "switch.light_kitchen",
            "climate.thermostat",
            "cover.garage_door",
        ]

        for sensor_id in valid_ids:
            assert validator(sensor_id), f"Valid sensor ID rejected: {sensor_id}"

        # Invalid sensor IDs
        invalid_ids = [
            "InvalidFormat",  # No domain separator
            "sensor.",  # Empty object_id
            ".object_id",  # Empty domain
            "sensor.Object_ID",  # Uppercase not allowed
            "sensor.object-id",  # Hyphens not allowed
            "sensor..object_id",  # Double separator
            123,  # Not a string
            None,  # None value
        ]

        for sensor_id in invalid_ids:
            assert not validator(sensor_id), f"Invalid sensor ID accepted: {sensor_id}"

    def test_room_id_format_validation(self, json_schema_validator):
        """Test room ID format validation."""
        validator = json_schema_validator._validate_room_id_format

        # Valid room IDs
        valid_ids = [
            "living_room",
            "bedroom_1",
            "kitchen-area",
            "basement_2",
            "office",
            "MASTER_BEDROOM",  # Uppercase allowed
        ]

        for room_id in valid_ids:
            assert validator(room_id), f"Valid room ID rejected: {room_id}"

        # Invalid room IDs
        invalid_ids = [
            "",  # Empty string
            "room with spaces",  # Spaces not allowed
            "room@invalid",  # Special characters not allowed
            "room.with.dots",  # Dots not allowed
            "x" * 101,  # Too long (>100 chars)
            123,  # Not a string
            None,  # None value
        ]

        for room_id in invalid_ids:
            assert not validator(room_id), f"Invalid room ID accepted: {room_id}"

    def test_entity_id_format_validation(self, json_schema_validator):
        """Test Home Assistant entity ID format validation."""
        validator = json_schema_validator._validate_entity_id_format

        # Valid entity IDs (same format as sensor IDs)
        valid_ids = [
            "light.living_room_lamp",
            "switch.kitchen_outlet",
            "sensor.outdoor_temperature",
        ]

        for entity_id in valid_ids:
            assert validator(entity_id), f"Valid entity ID rejected: {entity_id}"

        # Invalid entity IDs
        invalid_ids = [
            "LIGHT.living_room",  # Domain uppercase
            "light.Living_Room",  # Object ID uppercase
            "light living_room",  # No separator
        ]

        for entity_id in invalid_ids:
            assert not validator(entity_id), f"Invalid entity ID accepted: {entity_id}"

    def test_iso_datetime_format_validation(self, json_schema_validator):
        """Test ISO datetime format validation."""
        validator = json_schema_validator._validate_iso_datetime_format

        # Valid ISO datetime formats
        valid_datetimes = [
            "2024-01-15T10:30:00Z",
            "2024-01-15T10:30:00+00:00",
            "2024-01-15T10:30:00.123Z",
            "2024-01-15T10:30:00.123456+05:30",
            "2024-12-31T23:59:59Z",
        ]

        for dt_str in valid_datetimes:
            assert validator(dt_str), f"Valid datetime rejected: {dt_str}"

        # Invalid datetime formats
        invalid_datetimes = [
            "2024-01-15 10:30:00",  # Missing T separator
            "2024/01/15T10:30:00Z",  # Wrong date format
            "15-01-2024T10:30:00Z",  # Wrong date order
            "2024-13-01T10:30:00Z",  # Invalid month
            "2024-01-32T10:30:00Z",  # Invalid day
            "2024-01-15T25:30:00Z",  # Invalid hour
            "2024-01-15T10:70:00Z",  # Invalid minute
            "not-a-datetime",  # Not a datetime
            123,  # Not a string
            None,  # None value
        ]

        for dt_str in invalid_datetimes:
            assert not validator(dt_str), f"Invalid datetime accepted: {dt_str}"

    def test_sensor_state_format_validation(self, json_schema_validator):
        """Test sensor state format validation."""
        validator = json_schema_validator._validate_sensor_state_format

        # Valid sensor states (from SensorState enum)
        valid_states = [state.value for state in SensorState]

        for state in valid_states:
            assert validator(state), f"Valid sensor state rejected: {state}"

        # Invalid sensor states
        invalid_states = [
            "invalid_state",
            "ON",  # Wrong case
            "Off",  # Wrong case
            "",  # Empty string
            123,  # Not a string
            None,  # None value
        ]

        for state in invalid_states:
            assert not validator(state), f"Invalid sensor state accepted: {state}"

    def test_sensor_type_format_validation(self, json_schema_validator):
        """Test sensor type format validation."""
        validator = json_schema_validator._validate_sensor_type_format

        # Valid sensor types (from SensorType enum)
        valid_types = [sensor_type.value for sensor_type in SensorType]

        for sensor_type in valid_types:
            assert validator(sensor_type), f"Valid sensor type rejected: {sensor_type}"

        # Invalid sensor types
        invalid_types = [
            "invalid_type",
            "MOTION",  # Wrong case
            "",  # Empty string
            123,  # Not a string
            None,  # None value
        ]

        for sensor_type in invalid_types:
            assert not validator(
                sensor_type
            ), f"Invalid sensor type accepted: {sensor_type}"

    def test_coordinate_format_validation(self, json_schema_validator):
        """Test coordinate format validation."""
        validator = json_schema_validator._validate_coordinate_format

        # Valid coordinates
        valid_coordinates = [0, 0.0, 45.5, -90.0, 180, -180, 123.456789]

        for coord in valid_coordinates:
            assert validator(coord), f"Valid coordinate rejected: {coord}"

        # Invalid coordinates
        invalid_coordinates = [
            "123.45",  # String instead of number
            181,  # Out of range
            -181,  # Out of range
            None,  # None value
            float("inf"),  # Infinity
            float("nan"),  # NaN
        ]

        for coord in invalid_coordinates:
            assert not validator(coord), f"Invalid coordinate accepted: {coord}"

    def test_duration_format_validation(self, json_schema_validator):
        """Test ISO 8601 duration format validation."""
        validator = json_schema_validator._validate_duration_format

        # Valid durations
        valid_durations = [
            "P1Y",  # 1 year
            "P1M",  # 1 month
            "P1D",  # 1 day
            "PT1H",  # 1 hour
            "PT1M",  # 1 minute
            "PT1S",  # 1 second
            "P1Y2M3DT4H5M6S",  # Full format
            "PT0.5S",  # Decimal seconds
            "P0D",  # Zero duration
        ]

        for duration in valid_durations:
            assert validator(duration), f"Valid duration rejected: {duration}"

        # Invalid durations
        invalid_durations = [
            "1 hour",  # Plain text
            "P",  # Empty
            "T1H",  # Missing P
            "P1Y1Y",  # Duplicate components
            "PT",  # No time components
            123,  # Not a string
            None,  # None value
        ]

        for duration in invalid_durations:
            assert not validator(duration), f"Invalid duration accepted: {duration}"

    def test_validate_sensor_event_schema_valid_data(
        self, json_schema_validator, valid_sensor_event
    ):
        """Test validation of valid sensor event data."""
        result = json_schema_validator.validate_json_schema(
            valid_sensor_event, "sensor_event"
        )

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence_score == 1.0
        assert result.validation_id is not None

    def test_validate_sensor_event_schema_missing_required_fields(
        self, json_schema_validator
    ):
        """Test validation with missing required fields."""
        incomplete_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.motion_detector",
            # Missing sensor_type, state, timestamp
        }

        result = json_schema_validator.validate_json_schema(
            incomplete_event, "sensor_event"
        )

        assert not result.is_valid
        assert len(result.errors) >= 3  # At least 3 missing required fields

        # Check for specific required field errors
        error_messages = [error.message for error in result.errors]
        required_fields = ["sensor_type", "state", "timestamp"]

        for field in required_fields:
            assert any(
                field in message for message in error_messages
            ), f"Missing error for required field: {field}"

    def test_validate_sensor_event_schema_invalid_types(self, json_schema_validator):
        """Test validation with invalid field types."""
        invalid_event = {
            "room_id": 123,  # Should be string
            "sensor_id": ["not", "string"],  # Should be string
            "sensor_type": "invalid_type",  # Invalid sensor type
            "state": "invalid_state",  # Invalid state
            "timestamp": "not-iso-datetime",  # Invalid datetime format
            "attributes": "not-an-object",  # Should be object
            "confidence_score": 1.5,  # Should be <= 1.0
        }

        result = json_schema_validator.validate_json_schema(
            invalid_event, "sensor_event"
        )

        assert not result.is_valid
        assert len(result.errors) > 0

        # Check for type and format errors
        error_rules = {error.rule_id for error in result.errors}
        assert any("TYPE" in rule for rule in error_rules)
        assert any("FORMAT" in rule for rule in error_rules)
        assert any("MAXIMUM" in rule for rule in error_rules)

    def test_validate_room_config_schema_valid_data(
        self, json_schema_validator, valid_room_config
    ):
        """Test validation of valid room configuration data."""
        result = json_schema_validator.validate_json_schema(
            valid_room_config, "room_config"
        )

        assert result.is_valid
        assert len(result.errors) == 0

    def test_validate_room_config_schema_invalid_structure(self, json_schema_validator):
        """Test validation of invalid room configuration structure."""
        invalid_config = {
            "room_id": "room@invalid",  # Invalid format
            "name": "",  # Too short
            "sensors": {
                "invalid-sensor-type": {  # Invalid pattern
                    "entities": "not-an-array",  # Should be array
                    "timeout_minutes": -5,  # Should be positive
                }
            },
            "coordinates": {
                "x": "not-a-number",  # Should be number
                "y": 200,  # Out of coordinate range
            },
        }

        result = json_schema_validator.validate_json_schema(
            invalid_config, "room_config"
        )

        assert not result.is_valid
        assert len(result.errors) > 0

        # Should catch format, type, and range errors
        error_types = {
            error.context.get("validator") for error in result.errors if error.context
        }
        assert "format" in error_types
        assert "type" in error_types

    def test_validation_context_strict_mode(
        self, json_schema_validator, valid_sensor_event
    ):
        """Test validation with different context settings."""
        # Add extra property to test strict mode
        event_with_extra = valid_sensor_event.copy()
        event_with_extra["extra_property"] = "should_be_rejected"

        # Strict mode (default) - should reject additional properties
        strict_context = SchemaValidationContext(strict_mode=True)
        result = json_schema_validator.validate_json_schema(
            event_with_extra, "sensor_event", strict_context
        )

        assert not result.is_valid
        additional_prop_errors = [
            e
            for e in result.errors
            if "additional" in e.message.lower() or "ADDITIONALPROPERTIES" in e.rule_id
        ]
        assert len(additional_prop_errors) > 0

    def test_validation_context_allow_additional(
        self, json_schema_validator, valid_sensor_event
    ):
        """Test validation allowing additional properties."""
        # Add extra property
        event_with_extra = valid_sensor_event.copy()
        event_with_extra["extra_property"] = "should_be_allowed"

        # Allow additional properties
        lenient_context = SchemaValidationContext(allow_additional_properties=True)
        result = json_schema_validator.validate_json_schema(
            event_with_extra, "sensor_event", lenient_context
        )

        # Should be valid since additional properties are allowed
        assert result.is_valid

    def test_register_custom_schema(self, json_schema_validator):
        """Test registering custom schema definition."""
        custom_schema = SchemaDefinition(
            schema_id="custom_test",
            name="Custom Test Schema",
            version="1.0.0",
            description="Test schema for unit testing",
            schema={
                "$schema": "http://json-schema.org/draft-07/schema#",
                "type": "object",
                "required": ["test_field"],
                "properties": {
                    "test_field": {"type": "string"},
                    "optional_field": {"type": "number"},
                },
            },
            tags=["test", "custom"],
        )

        json_schema_validator.register_schema(custom_schema)

        # Verify schema was registered
        assert "custom_test" in json_schema_validator.schemas

        # Test using the custom schema
        valid_data = {"test_field": "test_value", "optional_field": 42}
        result = json_schema_validator.validate_json_schema(valid_data, "custom_test")

        assert result.is_valid

        # Test with invalid data
        invalid_data = {"optional_field": 42}  # Missing required field
        result = json_schema_validator.validate_json_schema(invalid_data, "custom_test")

        assert not result.is_valid

    def test_get_schema_info(self, json_schema_validator):
        """Test getting schema information."""
        info = json_schema_validator.get_schema_info("sensor_event")

        assert isinstance(info, dict)
        assert info["schema_id"] == "sensor_event"
        assert info["name"] == "Sensor Event"
        assert info["version"] == "1.0.0"
        assert "created_at" in info
        assert "custom_validators" in info

        # Test with non-existent schema
        assert json_schema_validator.get_schema_info("non_existent") is None

    def test_validation_error_suggestions(self, json_schema_validator):
        """Test that validation errors include helpful suggestions."""
        invalid_event = {
            "room_id": "",  # Too short (minLength: 1)
            "sensor_id": "x" * 201,  # Too long (maxLength: 200)
            "sensor_type": "invalid",  # Wrong format
            "state": "wrong",  # Wrong format
            "timestamp": "invalid",  # Wrong format
            "confidence_score": 2.0,  # Too high (maximum: 1)
        }

        result = json_schema_validator.validate_json_schema(
            invalid_event, "sensor_event"
        )

        assert not result.is_valid
        assert len(result.errors) > 0

        # All errors should have suggestions
        for error in result.errors:
            assert error.suggestion is not None
            assert len(error.suggestion) > 0

            # Test specific suggestion patterns
            if "minLength" in error.rule_id:
                assert "at least" in error.suggestion.lower()
            elif "maxLength" in error.rule_id:
                assert "no more than" in error.suggestion.lower()
            elif "format" in error.rule_id:
                assert "format" in error.suggestion.lower()
            elif "maximum" in error.rule_id:
                assert "no more than" in error.suggestion.lower()

    def test_validation_with_nonexistent_schema(self, json_schema_validator):
        """Test validation with non-existent schema."""
        result = json_schema_validator.validate_json_schema(
            {"test": "data"}, "non_existent_schema"
        )

        assert not result.is_valid
        assert len(result.errors) == 1
        assert "not found" in result.errors[0].message
        assert result.errors[0].rule_id == "SCH_JSON_001"

    def test_custom_validator_integration(self, json_schema_validator):
        """Test custom validator integration."""

        def custom_validator(data):
            errors = []
            if isinstance(data, dict) and data.get("test_field") == "forbidden":
                errors.append(
                    ValidationError(
                        rule_id="CUSTOM_001",
                        field="test_field",
                        value="forbidden",
                        message="Value 'forbidden' is not allowed",
                        severity=ErrorSeverity.HIGH,
                        suggestion="Use a different value",
                    )
                )
            return errors

        # Create schema with custom validator
        custom_schema = SchemaDefinition(
            schema_id="custom_validator_test",
            name="Custom Validator Test",
            version="1.0.0",
            schema={"type": "object", "properties": {"test_field": {"type": "string"}}},
            validators=[custom_validator],
        )

        json_schema_validator.register_schema(custom_schema)

        # Test with forbidden value
        forbidden_data = {"test_field": "forbidden"}
        result = json_schema_validator.validate_json_schema(
            forbidden_data, "custom_validator_test"
        )

        assert not result.is_valid
        custom_errors = [e for e in result.errors if e.rule_id == "CUSTOM_001"]
        assert len(custom_errors) == 1

        # Test with allowed value
        allowed_data = {"test_field": "allowed"}
        result = json_schema_validator.validate_json_schema(
            allowed_data, "custom_validator_test"
        )

        assert result.is_valid

    def test_validation_error_handling(self, json_schema_validator):
        """Test validation error handling for malformed schemas."""
        # Create schema with invalid JSON schema syntax
        bad_schema = SchemaDefinition(
            schema_id="bad_schema",
            name="Bad Schema",
            version="1.0.0",
            schema={
                "type": "invalid_type",  # Invalid JSON schema type
                "properties": "not_an_object",  # Should be object
            },
        )

        json_schema_validator.register_schema(bad_schema)

        result = json_schema_validator.validate_json_schema(
            {"test": "data"}, "bad_schema"
        )

        # Should handle schema errors gracefully
        assert not result.is_valid
        assert len(result.errors) > 0


class TestDatabaseSchemaValidator:
    """Comprehensive tests for DatabaseSchemaValidator."""

    def test_initialization(self, database_schema_validator):
        """Test DatabaseSchemaValidator initialization."""
        assert database_schema_validator.session is not None
        assert len(database_schema_validator.expected_tables) > 0
        assert len(database_schema_validator.expected_columns) > 0

        # Verify expected table structure
        expected_tables = {
            "sensor_events",
            "room_states",
            "predictions",
            "model_metadata",
            "feature_cache",
        }
        assert database_schema_validator.expected_tables == expected_tables

    @pytest.mark.asyncio
    async def test_validate_database_schema_complete_setup(
        self, database_schema_validator
    ):
        """Test database schema validation with complete setup."""
        # Mock inspector with all expected tables
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = list(
            database_schema_validator.expected_tables
        )

        # Mock column information for sensor_events
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": "BIGINT"},
            {"name": "room_id", "type": "VARCHAR"},
            {"name": "sensor_id", "type": "VARCHAR"},
            {"name": "sensor_type", "type": "VARCHAR"},
            {"name": "state", "type": "VARCHAR"},
            {"name": "previous_state", "type": "VARCHAR"},
            {"name": "timestamp", "type": "TIMESTAMP"},
            {"name": "attributes", "type": "JSON"},
            {"name": "is_human_triggered", "type": "BOOLEAN"},
        ]

        # Mock indexes
        mock_inspector.get_indexes.return_value = [
            {"name": "idx_room_sensor_time"},
            {"name": "idx_room_state_changes"},
        ]

        # Mock TimescaleDB extension check
        mock_result = MagicMock()
        mock_result.fetchone.return_value = {
            "extname": "timescaledb"
        }  # Extension exists
        database_schema_validator.session.execute.return_value = mock_result

        with patch(
            "src.data.validation.schema_validator.inspect", return_value=mock_inspector
        ):
            result = await database_schema_validator.validate_database_schema()

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence_score > 0.8

    @pytest.mark.asyncio
    async def test_validate_database_schema_missing_tables(
        self, database_schema_validator
    ):
        """Test database schema validation with missing tables."""
        # Mock inspector with missing tables
        mock_inspector = MagicMock()
        mock_inspector.get_table_names.return_value = [
            "sensor_events",
            "room_states",  # Missing predictions, model_metadata, feature_cache
        ]

        with patch(
            "src.data.validation.schema_validator.inspect", return_value=mock_inspector
        ):
            result = await database_schema_validator.validate_database_schema()

        assert not result.is_valid
        assert len(result.errors) >= 3  # At least 3 missing tables

        # Check for specific missing table errors
        missing_table_errors = [e for e in result.errors if e.rule_id == "DB_SCH_001"]
        assert len(missing_table_errors) == 3

        missing_tables = {error.value for error in missing_table_errors}
        expected_missing = {"predictions", "model_metadata", "feature_cache"}
        assert missing_tables == expected_missing

    @pytest.mark.asyncio
    async def test_validate_database_schema_unexpected_tables(
        self, database_schema_validator
    ):
        """Test database schema validation with unexpected tables."""
        # Mock inspector with extra tables
        mock_inspector = MagicMock()
        expected_tables = list(database_schema_validator.expected_tables)
        unexpected_tables = ["legacy_table", "temp_data", "backup_events"]
        all_tables = expected_tables + unexpected_tables

        mock_inspector.get_table_names.return_value = all_tables
        mock_inspector.get_columns.return_value = []  # Empty for simplicity
        mock_inspector.get_indexes.return_value = []

        # Mock TimescaleDB checks
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None  # No extension
        database_schema_validator.session.execute.return_value = mock_result

        with patch(
            "src.data.validation.schema_validator.inspect", return_value=mock_inspector
        ):
            result = await database_schema_validator.validate_database_schema()

        # Should have warnings for unexpected tables
        unexpected_table_warnings = [
            w for w in result.warnings if w.rule_id == "DB_SCH_002"
        ]
        assert len(unexpected_table_warnings) == 3

        warning_tables = {warning.value for warning in unexpected_table_warnings}
        assert warning_tables == set(unexpected_tables)

    @pytest.mark.asyncio
    async def test_validate_table_columns_missing_columns(
        self, database_schema_validator
    ):
        """Test table column validation with missing columns."""
        mock_inspector = MagicMock()

        # Mock sensor_events table with missing columns
        mock_inspector.get_columns.return_value = [
            {"name": "id", "type": "BIGINT"},
            {"name": "room_id", "type": "VARCHAR"},
            {"name": "sensor_id", "type": "VARCHAR"},
            # Missing: sensor_type, state, previous_state, timestamp, attributes, is_human_triggered
        ]

        expected_columns = database_schema_validator.expected_columns["sensor_events"]
        errors = await database_schema_validator._validate_table_columns(
            mock_inspector, "sensor_events", expected_columns
        )

        assert len(errors) >= 6  # At least 6 missing columns
        missing_column_errors = [e for e in errors if e.rule_id == "DB_SCH_003"]
        assert len(missing_column_errors) >= 6

    def test_validate_sensor_events_columns_correct_types(
        self, database_schema_validator
    ):
        """Test sensor_events column type validation."""
        # Mock column info with correct types
        column_info = {
            "timestamp": {"type": "TIMESTAMP WITH TIME ZONE"},
            "attributes": {"type": "JSON"},
        }

        errors = database_schema_validator._validate_sensor_events_columns(column_info)
        assert len(errors) == 0

    def test_validate_sensor_events_columns_incorrect_types(
        self, database_schema_validator
    ):
        """Test sensor_events column type validation with incorrect types."""
        # Mock column info with incorrect types
        column_info = {
            "timestamp": {"type": "VARCHAR"},  # Should be TIMESTAMP
            "attributes": {"type": "INTEGER"},  # Should be JSON or TEXT
        }

        errors = database_schema_validator._validate_sensor_events_columns(column_info)
        assert len(errors) == 2

        # Check specific error types
        timestamp_errors = [e for e in errors if e.rule_id == "DB_SCH_005"]
        attributes_errors = [e for e in errors if e.rule_id == "DB_SCH_006"]

        assert len(timestamp_errors) == 1
        assert len(attributes_errors) == 1

    def test_validate_room_states_columns(self, database_schema_validator):
        """Test room_states column type validation."""
        # Test with incorrect confidence column type
        column_info = {
            "confidence": {"type": "VARCHAR"}  # Should be FLOAT/DECIMAL/NUMERIC
        }

        errors = database_schema_validator._validate_room_states_columns(column_info)
        assert len(errors) == 1
        assert errors[0].rule_id == "DB_SCH_007"

        # Test with correct type
        column_info = {"confidence": {"type": "FLOAT"}}

        errors = database_schema_validator._validate_room_states_columns(column_info)
        assert len(errors) == 0

    def test_validate_predictions_columns(self, database_schema_validator):
        """Test predictions column type validation."""
        # Test with incorrect predicted_time column type
        column_info = {"predicted_time": {"type": "VARCHAR"}}  # Should be TIMESTAMP

        errors = database_schema_validator._validate_predictions_columns(column_info)
        assert len(errors) == 1
        assert errors[0].rule_id == "DB_SCH_008"

        # Test with correct type
        column_info = {"predicted_time": {"type": "TIMESTAMP WITH TIME ZONE"}}

        errors = database_schema_validator._validate_predictions_columns(column_info)
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_validate_indexes_missing(self, database_schema_validator):
        """Test index validation with missing indexes."""
        mock_inspector = MagicMock()

        # Mock empty indexes for sensor_events
        mock_inspector.get_indexes.return_value = []

        errors = await database_schema_validator._validate_indexes(mock_inspector)

        # Should have errors for missing indexes
        index_errors = [e for e in errors if e.rule_id == "DB_SCH_009"]
        assert len(index_errors) >= 2  # At least 2 missing indexes for sensor_events

        # Check specific missing indexes
        missing_indexes = {error.value.split(".")[-1] for error in index_errors}
        expected_missing = {"idx_room_sensor_time", "idx_room_state_changes"}
        assert expected_missing.issubset(missing_indexes)

    @pytest.mark.asyncio
    async def test_validate_timescaledb_extension_missing(
        self, database_schema_validator
    ):
        """Test TimescaleDB validation with missing extension."""
        # Mock missing extension
        mock_result = MagicMock()
        mock_result.fetchone.return_value = None
        database_schema_validator.session.execute.return_value = mock_result

        errors = await database_schema_validator._validate_timescaledb_features()

        assert len(errors) == 1
        assert errors[0].rule_id == "DB_SCH_010"
        assert "extension not installed" in errors[0].message

    @pytest.mark.asyncio
    async def test_validate_timescaledb_hypertable_missing(
        self, database_schema_validator
    ):
        """Test TimescaleDB validation with missing hypertable."""
        # Mock extension exists but hypertable doesn't
        extension_result = MagicMock()
        extension_result.fetchone.return_value = {"extname": "timescaledb"}

        hypertable_result = MagicMock()
        hypertable_result.fetchone.return_value = None  # No hypertable

        database_schema_validator.session.execute.side_effect = [
            extension_result,
            hypertable_result,
        ]

        errors = await database_schema_validator._validate_timescaledb_features()

        assert len(errors) == 1
        assert errors[0].rule_id == "DB_SCH_011"
        assert "not a TimescaleDB hypertable" in errors[0].message

    @pytest.mark.asyncio
    async def test_database_validation_error_handling(self, database_schema_validator):
        """Test database validation error handling."""
        # Mock inspector that raises exception
        with patch(
            "src.data.validation.schema_validator.inspect",
            side_effect=Exception("Database connection failed"),
        ):
            result = await database_schema_validator.validate_database_schema()

        assert not result.is_valid
        assert len(result.errors) == 1
        assert result.errors[0].rule_id == "DB_SCH_ERROR"
        assert "Database schema validation error" in result.errors[0].message


class TestAPISchemaValidator:
    """Comprehensive tests for APISchemaValidator."""

    def test_initialization(self, api_schema_validator):
        """Test APISchemaValidator initialization."""
        assert api_schema_validator.json_validator is not None
        assert len(api_schema_validator.content_type_validators) > 0

        # Verify content type validators
        expected_content_types = {
            "application/json",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        }
        actual_content_types = set(api_schema_validator.content_type_validators.keys())
        assert expected_content_types == actual_content_types

    def test_validate_api_request_valid(self, api_schema_validator):
        """Test API request validation with valid request."""
        valid_request = {
            "method": "POST",
            "path": "/api/sensors/motion",
            "headers": {
                "Authorization": "Bearer valid_token",
                "Content-Type": "application/json",
                "Accept": "application/json",
            },
            "body": {"sensor_id": "binary_sensor.motion", "state": "on"},
            "query_params": {"room": "living_room", "timeout": "5"},
        }

        result = api_schema_validator.validate_api_request(**valid_request)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence_score > 0.8

    def test_validate_api_request_invalid_method(self, api_schema_validator):
        """Test API request validation with invalid HTTP method."""
        result = api_schema_validator.validate_api_request(
            method="INVALID",
            path="/api/test",
            headers={"Authorization": "Bearer token"},
        )

        assert not result.is_valid
        method_errors = [e for e in result.errors if e.rule_id == "API_SCH_001"]
        assert len(method_errors) == 1
        assert "Invalid HTTP method" in method_errors[0].message

    def test_validate_api_request_invalid_path(self, api_schema_validator):
        """Test API request validation with invalid path."""
        result = api_schema_validator.validate_api_request(
            method="GET",
            path="invalid_path",  # Missing leading slash
            headers={"Authorization": "Bearer token"},
        )

        assert not result.is_valid
        path_errors = [e for e in result.errors if e.rule_id == "API_SCH_002"]
        assert len(path_errors) == 1
        assert "must start with '/'" in path_errors[0].message

    def test_validate_api_request_missing_auth(self, api_schema_validator):
        """Test API request validation with missing authentication."""
        result = api_schema_validator.validate_api_request(
            method="POST",
            path="/api/test",
            headers={"Content-Type": "application/json"},  # No auth header
        )

        assert not result.is_valid
        auth_errors = [e for e in result.errors if e.rule_id == "API_SCH_004"]
        assert len(auth_errors) == 1
        assert "No authentication header" in auth_errors[0].message

    def test_validate_headers_format_errors(self, api_schema_validator):
        """Test header validation with format errors."""
        invalid_headers = {
            123: "invalid_name",  # Non-string key
            "valid_name": 456,  # Non-string value
            "long_header": "x" * 10000,  # Too long value
        }

        errors = api_schema_validator._validate_headers(invalid_headers)

        assert len(errors) >= 2  # Type error and length error

        type_errors = [e for e in errors if e.rule_id == "API_SCH_005"]
        length_errors = [e for e in errors if e.rule_id == "API_SCH_006"]

        assert len(type_errors) >= 1
        assert len(length_errors) >= 1

    def test_validate_json_content_valid(self, api_schema_validator):
        """Test JSON content validation with valid JSON."""
        valid_json_objects = [
            {"key": "value", "number": 42},
            [1, 2, 3, {"nested": "object"}],
            '{"json_string": "valid"}',
            [],
        ]

        for json_obj in valid_json_objects:
            errors = api_schema_validator._validate_json_content(json_obj)
            assert len(errors) == 0, f"Valid JSON flagged as invalid: {json_obj}"

    def test_validate_json_content_invalid(self, api_schema_validator):
        """Test JSON content validation with invalid JSON."""
        invalid_json_objects = [
            '{"invalid": json}',  # Invalid JSON string
            "not json at all",  # Plain string (invalid JSON)
            123,  # Number (not object or array)
            True,  # Boolean (not object or array)
        ]

        for json_obj in invalid_json_objects:
            errors = api_schema_validator._validate_json_content(json_obj)
            assert len(errors) > 0, f"Invalid JSON not detected: {json_obj}"

    def test_validate_form_content(self, api_schema_validator):
        """Test form content validation."""
        # Valid form content
        valid_forms = [
            {"field1": "value1", "field2": "value2"},
            "field1=value1&field2=value2",
            {},
        ]

        for form_data in valid_forms:
            errors = api_schema_validator._validate_form_content(form_data)
            assert len(errors) == 0, f"Valid form data flagged as invalid: {form_data}"

        # Invalid form content
        invalid_forms = [
            123,  # Not dict or string
            ["list", "not", "valid"],  # List not valid
            None,  # None not valid
        ]

        for form_data in invalid_forms:
            errors = api_schema_validator._validate_form_content(form_data)
            assert len(errors) > 0, f"Invalid form data not detected: {form_data}"

    def test_validate_multipart_content(self, api_schema_validator):
        """Test multipart content validation."""
        # Valid multipart content
        valid_multipart = [
            {"file": "file_data", "field": "value"},
            b"multipart_bytes_data",
            {},
        ]

        for mp_data in valid_multipart:
            errors = api_schema_validator._validate_multipart_content(mp_data)
            assert (
                len(errors) == 0
            ), f"Valid multipart data flagged as invalid: {mp_data}"

        # Invalid multipart content
        invalid_multipart = [
            123,  # Not dict or bytes
            "string_not_bytes",  # String not valid
            ["list", "not", "valid"],  # List not valid
        ]

        for mp_data in invalid_multipart:
            errors = api_schema_validator._validate_multipart_content(mp_data)
            assert len(errors) > 0, f"Invalid multipart data not detected: {mp_data}"

    def test_validate_query_params(self, api_schema_validator):
        """Test query parameter validation."""
        # Valid query parameters
        valid_params = {
            "room_id": "living_room",
            "timeout": "5",
            "enable_debug": "true",
            "sensor-type": "motion",
        }

        errors = api_schema_validator._validate_query_params(valid_params)
        assert len(errors) == 0

        # Invalid query parameters
        invalid_params = {
            "invalid@param": "value",  # Invalid characters
            "valid_param": "x" * 3000,  # Too long value
            "param with spaces": "value",  # Spaces not allowed
        }

        errors = api_schema_validator._validate_query_params(invalid_params)
        assert len(errors) >= 2  # Name format and length errors

        name_errors = [e for e in errors if e.rule_id == "API_SCH_011"]
        length_errors = [e for e in errors if e.rule_id == "API_SCH_012"]

        assert len(name_errors) >= 2  # Two invalid parameter names
        assert len(length_errors) >= 1  # One too-long value

    def test_validate_api_request_content_type_handling(self, api_schema_validator):
        """Test API request validation with different content types."""
        base_request = {
            "method": "POST",
            "path": "/api/test",
            "headers": {"Authorization": "Bearer token"},
        }

        # Test supported content types
        supported_types = [
            ("application/json", {"key": "value"}),
            ("application/x-www-form-urlencoded", {"field": "value"}),
            ("multipart/form-data", {"file": b"data"}),
        ]

        for content_type, body in supported_types:
            request = base_request.copy()
            request["headers"] = base_request["headers"].copy()
            request["headers"]["Content-Type"] = content_type
            request["body"] = body

            result = api_schema_validator.validate_api_request(**request)

            # Should not have unsupported content type warnings
            content_warnings = [
                w for w in result.warnings if w.rule_id == "API_SCH_003"
            ]
            assert len(content_warnings) == 0

        # Test unsupported content type
        request = base_request.copy()
        request["headers"] = base_request["headers"].copy()
        request["headers"]["Content-Type"] = "application/xml"
        request["body"] = "<xml>data</xml>"

        result = api_schema_validator.validate_api_request(**request)

        # Should have unsupported content type warning
        content_warnings = [w for w in result.warnings if w.rule_id == "API_SCH_003"]
        assert len(content_warnings) == 1

    def test_validate_api_request_comprehensive(self, api_schema_validator):
        """Test comprehensive API request validation with multiple issues."""
        problematic_request = {
            "method": "INVALID_METHOD",  # Invalid method
            "path": "no_leading_slash",  # Invalid path
            "headers": {
                # No authentication
                "Content-Type": "application/json",
                123: "invalid_header_name",  # Invalid header
                "too_long": "x" * 10000,  # Too long header value
            },
            "body": '{"invalid": json}',  # Invalid JSON
            "query_params": {
                "invalid@param": "value",  # Invalid param name
                "long_param": "x" * 3000,  # Too long param value
            },
        }

        result = api_schema_validator.validate_api_request(**problematic_request)

        assert not result.is_valid
        assert len(result.errors) >= 5  # Multiple types of errors

        # Check for specific error types
        error_rules = {error.rule_id for error in result.errors}
        expected_rules = {
            "API_SCH_001",  # Invalid method
            "API_SCH_002",  # Invalid path
            "API_SCH_004",  # Missing auth
            "API_SCH_005",  # Invalid header
            "API_SCH_008",  # Invalid JSON
            "API_SCH_011",  # Invalid param name
            "API_SCH_012",  # Long param value
        }

        # Should have most of the expected error types
        assert len(error_rules & expected_rules) >= 5


class TestSchemaDefinitionAndContext:
    """Tests for SchemaDefinition and SchemaValidationContext dataclasses."""

    def test_schema_definition_creation(self):
        """Test SchemaDefinition creation and properties."""

        def custom_validator(data):
            return []

        schema_def = SchemaDefinition(
            schema_id="test_schema",
            name="Test Schema",
            version="2.1.0",
            schema={"type": "object"},
            description="A test schema for unit testing",
            tags=["test", "validation"],
            validators=[custom_validator],
        )

        assert schema_def.schema_id == "test_schema"
        assert schema_def.name == "Test Schema"
        assert schema_def.version == "2.1.0"
        assert schema_def.schema == {"type": "object"}
        assert schema_def.description == "A test schema for unit testing"
        assert schema_def.tags == ["test", "validation"]
        assert len(schema_def.validators) == 1
        assert isinstance(schema_def.created_at, datetime)

    def test_schema_validation_context_creation(self):
        """Test SchemaValidationContext creation and properties."""
        custom_formats = {"custom_format": lambda x: True}
        metadata = {"source": "unit_test"}

        context = SchemaValidationContext(
            strict_mode=False,
            allow_additional_properties=True,
            custom_formats=custom_formats,
            validation_metadata=metadata,
        )

        assert isinstance(context.validation_id, str)
        assert len(context.validation_id) > 0  # Should be a valid UUID string
        assert not context.strict_mode
        assert context.allow_additional_properties
        assert context.custom_formats == custom_formats
        assert context.validation_metadata == metadata

    def test_schema_validation_context_defaults(self):
        """Test SchemaValidationContext default values."""
        context = SchemaValidationContext()

        assert isinstance(context.validation_id, str)
        assert context.strict_mode is True  # Default
        assert context.allow_additional_properties is False  # Default
        assert isinstance(context.custom_formats, dict)
        assert len(context.custom_formats) == 0  # Default empty
        assert isinstance(context.validation_metadata, dict)
        assert len(context.validation_metadata) == 0  # Default empty


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling scenarios."""

    def test_json_validator_with_circular_references(self, json_schema_validator):
        """Test JSON validator handling of circular references."""
        # Create data with circular reference
        circular_data = {"name": "test"}
        circular_data["self"] = circular_data

        # Should handle circular references gracefully
        try:
            result = json_schema_validator.validate_json_schema(
                circular_data, "sensor_event"
            )
            # May succeed or fail, but should not crash
            assert isinstance(result, ValidationResult)
        except Exception as e:
            # If it raises an exception, it should be handled gracefully
            assert "circular" in str(e).lower() or "recursion" in str(e).lower()

    def test_json_validator_with_very_large_data(self, json_schema_validator):
        """Test JSON validator with very large data structures."""
        # Create very large but valid sensor event
        large_event = {
            "room_id": "test_room",
            "sensor_id": "binary_sensor.test",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T10:30:00Z",
            "attributes": {
                f"key_{i}": f"value_{i}" for i in range(1000)
            },  # 1000 attributes
        }

        result = json_schema_validator.validate_json_schema(large_event, "sensor_event")

        # Should handle large data without issues
        assert isinstance(result, ValidationResult)
        # Result validity depends on schema constraints

    @pytest.mark.asyncio
    async def test_database_validator_with_connection_issues(self, mock_db_session):
        """Test database validator with connection issues."""
        # Mock session that raises connection errors
        mock_db_session.execute.side_effect = Exception("Connection lost")

        db_validator = DatabaseSchemaValidator(mock_db_session)
        result = await db_validator.validate_database_schema()

        # Should handle connection errors gracefully
        assert not result.is_valid
        assert len(result.errors) > 0
        connection_errors = [
            e for e in result.errors if "connection" in e.message.lower()
        ]
        assert len(connection_errors) > 0

    def test_api_validator_with_malformed_headers(self, api_schema_validator):
        """Test API validator with malformed headers."""
        malformed_headers = {
            "": "empty_name",  # Empty header name
            "valid": "",  # Empty value (should be ok)
            "unicode_name": "value_with_unicode_émojis_🚀",
            "\x00null_byte": "dangerous",
            "control\r\ncharacters": "newline_injection",
        }

        result = api_schema_validator.validate_api_request(
            method="GET", path="/api/test", headers=malformed_headers
        )

        # Should handle malformed headers gracefully
        assert isinstance(result, ValidationResult)
        # Some malformed headers should cause validation errors
        assert len(result.errors) > 0

    def test_format_validators_with_none_and_empty_values(self, json_schema_validator):
        """Test format validators with None and empty values."""
        validators = [
            ("sensor_id", json_schema_validator._validate_sensor_id_format),
            ("room_id", json_schema_validator._validate_room_id_format),
            ("entity_id", json_schema_validator._validate_entity_id_format),
            ("iso_datetime", json_schema_validator._validate_iso_datetime_format),
            ("sensor_state", json_schema_validator._validate_sensor_state_format),
            ("sensor_type", json_schema_validator._validate_sensor_type_format),
            ("duration", json_schema_validator._validate_duration_format),
        ]

        test_values = [None, "", " ", "   "]  # None and various empty values

        for format_name, validator in validators:
            for test_value in test_values:
                # Should handle None and empty values gracefully (return False)
                try:
                    result = validator(test_value)
                    assert (
                        result is False
                    ), f"Format validator {format_name} should reject {repr(test_value)}"
                except Exception as e:
                    pytest.fail(
                        f"Format validator {format_name} crashed on {repr(test_value)}: {e}"
                    )

    def test_coordinate_validator_with_special_float_values(
        self, json_schema_validator
    ):
        """Test coordinate validator with special float values."""
        validator = json_schema_validator._validate_coordinate_format

        special_values = [
            float("inf"),  # Positive infinity
            float("-inf"),  # Negative infinity
            float("nan"),  # Not a number
            1e308,  # Very large number
            -1e308,  # Very large negative number
            1e-308,  # Very small number
        ]

        for value in special_values:
            # Should handle special float values gracefully
            try:
                result = validator(value)
                # Infinity and NaN should be rejected, very large/small numbers might be accepted
                assert isinstance(result, bool)
            except Exception as e:
                pytest.fail(f"Coordinate validator crashed on {value}: {e}")

    def test_schema_validator_memory_efficiency(self, json_schema_validator):
        """Test schema validator memory efficiency with repeated validations."""
        import gc
        import sys

        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())

        # Perform many validations
        test_data = {
            "room_id": "test_room",
            "sensor_id": "binary_sensor.test",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T10:30:00Z",
        }

        results = []
        for i in range(100):
            result = json_schema_validator.validate_json_schema(
                test_data, "sensor_event"
            )
            results.append(result)

        # Check memory usage after validations
        gc.collect()
        final_objects = len(gc.get_objects())

        # Memory usage should not have grown excessively
        object_increase = final_objects - initial_objects
        assert (
            object_increase < 10000
        ), f"Memory usage increased by {object_increase} objects"

        # All validations should have succeeded
        assert all(result.is_valid for result in results)

    def test_concurrent_schema_validation(self, json_schema_validator):
        """Test concurrent schema validation operations."""
        import threading
        import time

        test_data = {
            "room_id": "test_room",
            "sensor_id": "binary_sensor.test",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T10:30:00Z",
        }

        results = []
        errors = []

        def validate_schema():
            try:
                result = json_schema_validator.validate_json_schema(
                    test_data, "sensor_event"
                )
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Run concurrent validations
        threads = []
        for i in range(10):
            thread = threading.Thread(target=validate_schema)
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0, f"Concurrent validation errors: {errors}"
        assert len(results) == 10
        assert all(isinstance(result, ValidationResult) for result in results)
        assert all(result.is_valid for result in results)


class TestIntegrationScenarios:
    """Integration tests combining multiple validators."""

    def test_complete_sensor_event_validation_pipeline(self, json_schema_validator):
        """Test complete sensor event validation pipeline."""
        # Test data progressing from invalid to valid
        test_cases = [
            # Case 1: Completely invalid
            {
                "data": {"invalid": "data"},
                "should_be_valid": False,
                "description": "Completely invalid data",
            },
            # Case 2: Missing required fields
            {
                "data": {"room_id": "living_room", "sensor_id": "binary_sensor.motion"},
                "should_be_valid": False,
                "description": "Missing required fields",
            },
            # Case 3: Invalid formats
            {
                "data": {
                    "room_id": "living room",  # Spaces not allowed
                    "sensor_id": "InvalidFormat",  # No domain separator
                    "sensor_type": "invalid_type",
                    "state": "invalid_state",
                    "timestamp": "not-iso-format",
                },
                "should_be_valid": False,
                "description": "Invalid formats",
            },
            # Case 4: Valid data
            {
                "data": {
                    "room_id": "living_room",
                    "sensor_id": "binary_sensor.motion_detector",
                    "sensor_type": "motion",
                    "state": "on",
                    "timestamp": "2024-01-15T10:30:00Z",
                    "attributes": {"brightness": 100},
                    "is_human_triggered": True,
                    "confidence_score": 0.85,
                },
                "should_be_valid": True,
                "description": "Completely valid data",
            },
        ]

        for case in test_cases:
            result = json_schema_validator.validate_json_schema(
                case["data"], "sensor_event"
            )

            assert (
                result.is_valid == case["should_be_valid"]
            ), f"{case['description']}: expected {case['should_be_valid']}, got {result.is_valid}"

            if case["should_be_valid"]:
                assert result.confidence_score > 0.8
            else:
                assert len(result.errors) > 0

    def test_api_request_with_json_schema_validation(self, api_schema_validator):
        """Test API request validation combined with JSON schema validation."""
        # Valid API request with valid JSON body
        valid_request = {
            "method": "POST",
            "path": "/api/sensors/events",
            "headers": {
                "Authorization": "Bearer valid_token",
                "Content-Type": "application/json",
            },
            "body": {
                "room_id": "living_room",
                "sensor_id": "binary_sensor.motion",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": "2024-01-15T10:30:00Z",
            },
        }

        result = api_schema_validator.validate_api_request(**valid_request)
        assert result.is_valid

        # Valid API request but invalid JSON body structure
        invalid_body_request = valid_request.copy()
        invalid_body_request["body"] = {"incomplete": "data"}

        result = api_schema_validator.validate_api_request(**invalid_body_request)

        # API structure is valid, but JSON content validation may flag issues
        # depending on implementation
        assert isinstance(result, ValidationResult)
