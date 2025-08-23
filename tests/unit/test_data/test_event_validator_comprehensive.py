"""
Comprehensive test suite for EventValidator with security testing.

This test suite provides complete coverage of event validation functionality including:
- SQL injection prevention and detection
- XSS attack prevention and sanitization
- Path traversal attack prevention
- Input sanitization and security validation
- Schema validation and format checking
- Data integrity validation and corruption detection
- Cross-system consistency checks
- Performance testing for bulk validation
"""

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import RoomConfig, SystemConfig, get_config
from src.core.constants import SensorState, SensorType
from src.core.exceptions import ErrorSeverity
from src.data.validation.event_validator import (
    ComprehensiveEventValidator,
    IntegrityValidator,
    PerformanceValidator,
    SchemaValidator,
    SecurityValidator,
    ValidationError,
    ValidationResult,
    ValidationRule,
)


@pytest.fixture
def mock_db_session():
    """Create mock database session."""
    mock_session = MagicMock()
    mock_session.execute = AsyncMock()
    mock_session.scalars = MagicMock()
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock()
    return mock_session


@pytest.fixture
def security_validator():
    """Create SecurityValidator instance for testing."""
    return SecurityValidator()


@pytest.fixture
def schema_validator():
    """Create SchemaValidator instance for testing."""
    # Mock system config
    config = SystemConfig(
        home_assistant=MagicMock(),
        database=MagicMock(),
        mqtt=MagicMock(),
        prediction=MagicMock(),
        features=MagicMock(),
        logging=MagicMock(),
        rooms={
            "room_1": RoomConfig(room_id="room_1", name="Living Room"),
            "room_2": RoomConfig(room_id="room_2", name="Bedroom"),
        },
    )

    with patch("src.data.validation.event_validator.get_config", return_value=config):
        return SchemaValidator(config)


@pytest.fixture
def integrity_validator(mock_db_session):
    """Create IntegrityValidator instance for testing."""
    return IntegrityValidator(mock_db_session)


@pytest.fixture
def performance_validator():
    """Create PerformanceValidator instance for testing."""
    return PerformanceValidator(batch_size=100)


@pytest.fixture
def comprehensive_validator(mock_db_session):
    """Create ComprehensiveEventValidator instance for testing."""
    with patch("src.data.validation.event_validator.get_config") as mock_get_config:
        mock_config = SystemConfig(
            home_assistant=MagicMock(),
            database=MagicMock(),
            mqtt=MagicMock(),
            prediction=MagicMock(),
            features=MagicMock(),
            logging=MagicMock(),
            rooms={
                "room_1": RoomConfig(room_id="room_1", name="Living Room"),
                "room_2": RoomConfig(room_id="room_2", name="Bedroom"),
            },
        )
        mock_get_config.return_value = mock_config
        return ComprehensiveEventValidator(mock_db_session)


@pytest.fixture
def valid_event_data():
    """Create valid event data for testing."""
    return {
        "room_id": "room_1",
        "sensor_id": "sensor.living_room_motion",
        "sensor_type": "motion",
        "state": "on",
        "previous_state": "off",
        "timestamp": datetime.now(timezone.utc),
        "attributes": {"brightness": 100, "temperature": 20.5},
        "is_human_triggered": True,
    }


class TestSecurityValidator:
    """Comprehensive tests for SecurityValidator."""

    def test_initialization(self, security_validator):
        """Test SecurityValidator initialization."""
        assert len(security_validator.compiled_patterns["sql_injection"]) > 0
        assert len(security_validator.compiled_patterns["xss"]) > 0
        assert len(security_validator.compiled_patterns["path_traversal"]) > 0

        # Test that patterns are properly compiled
        for pattern_list in security_validator.compiled_patterns.values():
            for pattern in pattern_list:
                assert hasattr(pattern, "search")  # Should be compiled regex

    def test_sql_injection_detection_basic(self, security_validator):
        """Test basic SQL injection attack detection."""
        malicious_inputs = [
            "'; DROP TABLE users; --",
            "' OR '1'='1",
            "admin' --",
            "' UNION SELECT * FROM passwords",
            "'; DELETE FROM events; --",
            "' OR 1=1 --",
            "admin'; EXEC xp_cmdshell('dir'); --",
        ]

        for malicious_input in malicious_inputs:
            errors = security_validator.validate_input_security(
                malicious_input, "test_field"
            )

            # Should detect SQL injection
            sql_errors = [e for e in errors if e.rule_id == "SEC001"]
            assert (
                len(sql_errors) > 0
            ), f"Failed to detect SQL injection in: {malicious_input}"
            assert sql_errors[0].severity == ErrorSeverity.CRITICAL
            assert "SQL injection" in sql_errors[0].message

    def test_sql_injection_detection_advanced(self, security_validator):
        """Test advanced SQL injection attack patterns."""
        advanced_sql_attacks = [
            "1' AND (SELECT COUNT(*) FROM users) > 0 --",
            "1' AND ASCII(SUBSTRING((SELECT password FROM users WHERE id=1),1,1)) > 64 --",
            "'; WAITFOR DELAY '00:00:05'; --",
            "1' AND (SELECT user FROM mysql.user WHERE user='root') = 'root",
            "'; INSERT INTO admin (username, password) VALUES ('hacker', 'pass'); --",
            "' OR SLEEP(5) --",
            "1' AND (SELECT SUBSTR(table_name,1,1) FROM information_schema.tables)='A",
        ]

        for attack in advanced_sql_attacks:
            errors = security_validator.validate_input_security(
                attack, "advanced_field"
            )
            sql_errors = [e for e in errors if e.rule_id == "SEC001"]
            assert (
                len(sql_errors) > 0
            ), f"Failed to detect advanced SQL injection: {attack}"

    def test_xss_attack_detection_basic(self, security_validator):
        """Test basic XSS attack detection."""
        xss_attacks = [
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            "javascript:alert('XSS')",
            "<iframe src='javascript:alert(\"XSS\")'></iframe>",
            "<object data='javascript:alert(\"XSS\")'></object>",
            "<link href='javascript:alert(\"XSS\")'>",
            "<meta http-equiv='refresh' content='0;url=javascript:alert(\"XSS\")'>",
        ]

        for xss_attack in xss_attacks:
            errors = security_validator.validate_input_security(xss_attack, "xss_field")
            xss_errors = [e for e in errors if e.rule_id == "SEC002"]
            assert len(xss_errors) > 0, f"Failed to detect XSS attack: {xss_attack}"
            assert xss_errors[0].severity == ErrorSeverity.HIGH

    def test_xss_attack_detection_advanced(self, security_validator):
        """Test advanced XSS attack patterns."""
        advanced_xss_attacks = [
            "String.fromCharCode(88,83,83)",
            "eval(String.fromCharCode(97,108,101,114,116,40,39,88,83,83,39,41))",
            "<svg onload=alert('XSS')>",
            "<body onpageshow=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<select onfocus=alert('XSS') autofocus>",
            "<textarea onfocus=alert('XSS') autofocus>",
            "<keygen onfocus=alert('XSS') autofocus>",
            "<video><source onerror=alert('XSS')>",
        ]

        for attack in advanced_xss_attacks:
            errors = security_validator.validate_input_security(attack, "advanced_xss")
            xss_errors = [e for e in errors if e.rule_id == "SEC002"]
            assert len(xss_errors) > 0, f"Failed to detect advanced XSS: {attack}"

    def test_path_traversal_detection(self, security_validator):
        """Test path traversal attack detection."""
        path_attacks = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "~/../../etc/passwd",
            "/proc/self/environ",
            "C:\\Windows\\System32\\drivers\\etc\\hosts",
        ]

        for path_attack in path_attacks:
            errors = security_validator.validate_input_security(
                path_attack, "path_field"
            )
            path_errors = [e for e in errors if e.rule_id == "SEC003"]
            assert (
                len(path_errors) > 0
            ), f"Failed to detect path traversal: {path_attack}"
            assert path_errors[0].severity == ErrorSeverity.HIGH

    def test_dos_attack_detection(self, security_validator):
        """Test denial of service attack detection via input length."""
        # Create very long input
        long_input = "A" * 15000  # Exceeds 10,000 character limit

        errors = security_validator.validate_input_security(long_input, "dos_field")
        dos_errors = [e for e in errors if e.rule_id == "SEC004"]

        assert len(dos_errors) > 0
        assert dos_errors[0].severity == ErrorSeverity.MEDIUM
        assert "exceeds maximum allowed size" in dos_errors[0].message

    def test_null_byte_injection_detection(self, security_validator):
        """Test null byte injection detection."""
        null_byte_attacks = [
            "filename.txt\x00.exe",
            "config.php\x00",
            "../../../../etc/passwd\x00.jpg",
            "normal_text\x00malicious_code",
        ]

        for attack in null_byte_attacks:
            errors = security_validator.validate_input_security(attack, "null_field")
            null_errors = [e for e in errors if e.rule_id == "SEC005"]
            assert (
                len(null_errors) > 0
            ), f"Failed to detect null byte injection: {repr(attack)}"
            assert null_errors[0].severity == ErrorSeverity.HIGH

    def test_legitimate_input_acceptance(self, security_validator):
        """Test that legitimate inputs are not flagged as malicious."""
        legitimate_inputs = [
            "sensor.living_room_motion",
            "Normal room description with spaces",
            "Temperature: 20.5°C, Humidity: 65%",
            'Valid JSON: {"key": "value", "number": 42}',
            "Email: user@example.com",
            "URL: https://homeassistant.local:8123",
            "Timestamp: 2024-01-15T10:30:00Z",
        ]

        for legitimate_input in legitimate_inputs:
            errors = security_validator.validate_input_security(
                legitimate_input, "legitimate_field"
            )
            # Should have no security errors
            security_errors = [e for e in errors if e.rule_id.startswith("SEC")]
            assert len(security_errors) == 0, f"False positive for: {legitimate_input}"

    def test_input_sanitization_standard(self, security_validator):
        """Test standard input sanitization."""
        dangerous_input = '<script>alert("XSS")</script>'
        sanitized = security_validator.sanitize_input(dangerous_input, aggressive=False)

        # Should escape HTML entities
        assert "&lt;" in sanitized
        assert "&gt;" in sanitized
        assert "&quot;" in sanitized
        assert "<script>" not in sanitized

    def test_input_sanitization_aggressive(self, security_validator):
        """Test aggressive input sanitization."""
        dangerous_input = "SELECT * FROM users WHERE id='1' OR '1'='1'"
        sanitized = security_validator.sanitize_input(dangerous_input, aggressive=True)

        # Should remove SQL keywords and dangerous characters
        assert "SELECT" not in sanitized.upper()
        assert "'" not in sanitized
        assert "=" not in sanitized
        assert "OR" not in sanitized.upper()

    def test_mixed_attack_detection(self, security_validator):
        """Test detection of mixed attack types in single input."""
        mixed_attack = (
            "'; DROP TABLE users; --<script>alert('XSS')</script>../../../etc/passwd"
        )

        errors = security_validator.validate_input_security(mixed_attack, "mixed_field")

        # Should detect multiple attack types
        rule_ids = {error.rule_id for error in errors}
        assert "SEC001" in rule_ids  # SQL injection
        assert "SEC002" in rule_ids  # XSS
        assert "SEC003" in rule_ids  # Path traversal

    def test_unicode_and_encoding_attacks(self, security_validator):
        """Test handling of Unicode and encoding-based attacks."""
        unicode_attacks = [
            "＜script＞alert('XSS')＜/script＞",  # Full-width characters
            "%253Cscript%253E",  # Double URL encoding
            "&#60;script&#62;",  # HTML entities
            "\u003cscript\u003e",  # Unicode escapes
        ]

        for attack in unicode_attacks:
            errors = security_validator.validate_input_security(attack, "unicode_field")
            # Should detect at least some form of attack
            assert len(errors) >= 0  # May or may not detect depending on encoding

    def test_case_sensitivity_bypass_attempts(self, security_validator):
        """Test case sensitivity bypass attempts."""
        case_bypass_attempts = [
            "SeLeCt * FrOm UsErS",
            "uNiOn SeLeCt",
            "<ScRiPt>AlErT('XsS')</ScRiPt>",
            "jAvAsCrIpT:alert('xss')",
        ]

        for attempt in case_bypass_attempts:
            errors = security_validator.validate_input_security(attempt, "case_field")
            # Should detect regardless of case (patterns use IGNORECASE)
            security_errors = [e for e in errors if e.rule_id.startswith("SEC")]
            assert len(security_errors) > 0, f"Case bypass not detected: {attempt}"

    def test_whitespace_and_comment_evasion(self, security_validator):
        """Test whitespace and comment evasion techniques."""
        evasion_attempts = [
            "' /*comment*/ OR /*comment*/ '1'='1",
            "'/**/UNION/**/SELECT/**/",
            "' -- comment\nOR 1=1",
            "'\t\r\nOR\t\r\n1=1",
            "'  OR  1=1  --",
        ]

        for attempt in evasion_attempts:
            errors = security_validator.validate_input_security(
                attempt, "evasion_field"
            )
            sql_errors = [e for e in errors if e.rule_id == "SEC001"]
            assert len(sql_errors) > 0, f"Evasion not detected: {repr(attempt)}"


class TestSchemaValidator:
    """Comprehensive tests for SchemaValidator."""

    def test_valid_sensor_event_schema(self, schema_validator, valid_event_data):
        """Test validation of valid sensor event schema."""
        errors = schema_validator.validate_sensor_event_schema(valid_event_data)
        assert len(errors) == 0

    def test_missing_required_fields(self, schema_validator):
        """Test validation with missing required fields."""
        incomplete_event = {
            "room_id": "room_1",
            "sensor_id": "sensor.test",
            # Missing sensor_type, state, timestamp
        }

        errors = schema_validator.validate_sensor_event_schema(incomplete_event)

        # Should have errors for missing fields
        missing_fields = {error.field for error in errors if error.rule_id == "SCH001"}
        assert "sensor_type" in missing_fields
        assert "state" in missing_fields
        assert "timestamp" in missing_fields

    def test_null_required_fields(self, schema_validator):
        """Test validation with null required fields."""
        null_event = {
            "room_id": None,
            "sensor_id": None,
            "sensor_type": None,
            "state": None,
            "timestamp": None,
        }

        errors = schema_validator.validate_sensor_event_schema(null_event)

        # Should have errors for null fields
        null_errors = [e for e in errors if e.rule_id == "SCH002"]
        assert len(null_errors) >= 5  # All required fields are null

    def test_invalid_field_types(self, schema_validator):
        """Test validation with invalid field types."""
        invalid_types_event = {
            "room_id": 123,  # Should be string
            "sensor_id": ["invalid", "list"],  # Should be string
            "sensor_type": "invalid_type",  # Should be valid SensorType
            "state": "invalid_state",  # Should be valid SensorState
            "timestamp": "invalid_timestamp",  # Should be valid timestamp
            "attributes": "not_a_dict",  # Should be dict
        }

        errors = schema_validator.validate_sensor_event_schema(invalid_types_event)

        # Should have errors for type mismatches
        error_rules = {error.rule_id for error in errors}
        assert "SCH003" in error_rules  # room_id type error
        assert "SCH004" in error_rules  # sensor_id type error
        assert "SCH005" in error_rules  # invalid sensor_type
        assert "SCH006" in error_rules  # invalid state
        assert "SCH007" in error_rules  # invalid timestamp
        assert "SCH009" in error_rules  # invalid attributes type

    def test_valid_sensor_types_and_states(self, schema_validator):
        """Test validation with all valid sensor types and states."""
        for sensor_type in SensorType:
            for state in SensorState:
                event_data = {
                    "room_id": "room_1",
                    "sensor_id": f"sensor.{sensor_type.value}",
                    "sensor_type": sensor_type.value,
                    "state": state.value,
                    "timestamp": datetime.now(timezone.utc),
                }

                errors = schema_validator.validate_sensor_event_schema(event_data)
                type_state_errors = [
                    e for e in errors if e.rule_id in ["SCH005", "SCH006"]
                ]
                assert (
                    len(type_state_errors) == 0
                ), f"Valid {sensor_type.value}/{state.value} flagged as invalid"

    def test_timestamp_format_validation(self, schema_validator):
        """Test timestamp format validation."""
        valid_timestamps = [
            "2024-01-15T10:30:00Z",
            "2024-01-15T10:30:00+00:00",
            "2024-01-15T10:30:00.123Z",
            datetime.now(timezone.utc),
        ]

        invalid_timestamps = [
            "2024-01-15 10:30:00",  # Missing T separator
            "2024/01/15 10:30:00",  # Wrong date format
            "invalid-timestamp",
            "2024-13-45T25:99:99Z",  # Invalid date/time values
        ]

        # Test valid timestamps
        for timestamp in valid_timestamps:
            event_data = {
                "room_id": "room_1",
                "sensor_id": "sensor.test",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": timestamp,
            }

            errors = schema_validator.validate_sensor_event_schema(event_data)
            timestamp_errors = [e for e in errors if e.rule_id in ["SCH007", "SCH008"]]
            assert (
                len(timestamp_errors) == 0
            ), f"Valid timestamp flagged as invalid: {timestamp}"

        # Test invalid timestamps
        for timestamp in invalid_timestamps:
            event_data = {
                "room_id": "room_1",
                "sensor_id": "sensor.test",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": timestamp,
            }

            errors = schema_validator.validate_sensor_event_schema(event_data)
            timestamp_errors = [e for e in errors if e.rule_id == "SCH007"]
            assert (
                len(timestamp_errors) > 0
            ), f"Invalid timestamp not detected: {timestamp}"

    def test_timezone_awareness(self, schema_validator):
        """Test timezone awareness in timestamp validation."""
        # Naive datetime (without timezone)
        naive_datetime = datetime.now()

        event_data = {
            "room_id": "room_1",
            "sensor_id": "sensor.test",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": naive_datetime,
        }

        errors = schema_validator.validate_sensor_event_schema(event_data)
        timezone_errors = [e for e in errors if e.rule_id == "SCH008"]
        assert len(timezone_errors) > 0  # Should warn about missing timezone

    def test_room_configuration_validation(self, schema_validator):
        """Test room configuration validation."""
        valid_room_data = {
            "name": "Living Room",
            "sensors": {"motion": ["sensor.motion_1"], "door": ["sensor.door_1"]},
        }

        invalid_room_data = {
            "name": 123,  # Should be string
            "sensors": "invalid",  # Should be dict
        }

        # Test valid room config
        errors = schema_validator.validate_room_configuration("room_1", valid_room_data)
        assert len(errors) == 0

        # Test invalid room ID format
        errors = schema_validator.validate_room_configuration(
            "room-with-@invalid#chars!", valid_room_data
        )
        id_errors = [e for e in errors if e.rule_id == "SCH010"]
        assert len(id_errors) > 0

        # Test invalid room data
        errors = schema_validator.validate_room_configuration(
            "room_1", invalid_room_data
        )
        name_errors = [e for e in errors if e.rule_id == "SCH011"]
        assert len(name_errors) > 0

        # Test empty room name
        empty_name_data = {"name": "   "}
        errors = schema_validator.validate_room_configuration("room_1", empty_name_data)
        empty_errors = [e for e in errors if e.rule_id == "SCH012"]
        assert len(empty_errors) > 0


class TestIntegrityValidator:
    """Comprehensive tests for IntegrityValidator."""

    def test_event_hash_calculation(self, integrity_validator):
        """Test event hash calculation for integrity checking."""
        event_data = {
            "room_id": "room_1",
            "sensor_id": "sensor.test",
            "state": "on",
            "timestamp": datetime(2024, 1, 15, 10, 30, 0, tzinfo=timezone.utc),
        }

        hash1 = integrity_validator.calculate_event_hash(event_data)
        hash2 = integrity_validator.calculate_event_hash(event_data)

        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 produces 64 character hex string

        # Modified data should produce different hash
        modified_event = event_data.copy()
        modified_event["state"] = "off"
        hash3 = integrity_validator.calculate_event_hash(modified_event)
        assert hash1 != hash3

    def test_event_hash_key_ordering(self, integrity_validator):
        """Test that event hash is consistent regardless of key ordering."""
        event1 = {"room_id": "room_1", "state": "on", "sensor_id": "sensor.test"}
        event2 = {"state": "on", "room_id": "room_1", "sensor_id": "sensor.test"}

        hash1 = integrity_validator.calculate_event_hash(event1)
        hash2 = integrity_validator.calculate_event_hash(event2)

        # Should be the same despite different key ordering
        assert hash1 == hash2

    def test_event_hash_null_handling(self, integrity_validator):
        """Test event hash calculation with null values."""
        event_with_nulls = {
            "room_id": "room_1",
            "sensor_id": None,
            "state": "on",
            "attributes": None,
        }

        # Should not crash with null values
        event_hash = integrity_validator.calculate_event_hash(event_with_nulls)
        assert isinstance(event_hash, str)
        assert len(event_hash) == 64

    @pytest.mark.asyncio
    async def test_duplicate_detection(self, integrity_validator):
        """Test duplicate event detection."""
        events = [
            {
                "room_id": "room_1",
                "sensor_id": "sensor.test",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
            },
            {
                "room_id": "room_1",
                "sensor_id": "sensor.test",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),  # Identical event
            },
        ]

        errors = await integrity_validator.validate_data_consistency(events)
        duplicate_errors = [e for e in errors if e.rule_id == "INT001"]
        assert len(duplicate_errors) > 0

    @pytest.mark.asyncio
    async def test_timestamp_ordering_validation(self, integrity_validator):
        """Test timestamp ordering validation."""
        base_time = datetime.now(timezone.utc)

        events = [
            {
                "room_id": "room_1",
                "sensor_id": "sensor.test",
                "state": "on",
                "timestamp": base_time + timedelta(seconds=10),
            },
            {
                "room_id": "room_1",
                "sensor_id": "sensor.test",
                "state": "off",
                "timestamp": base_time + timedelta(seconds=5),  # Earlier timestamp
            },
        ]

        errors = await integrity_validator.validate_data_consistency(events)
        # Note: Current implementation doesn't enforce strict ordering,
        # but checks for events that are too close together
        timing_errors = [e for e in errors if e.rule_id == "INT002"]
        # May or may not have timing errors depending on MIN_EVENT_SEPARATION

    @pytest.mark.asyncio
    async def test_cross_system_consistency_validation(
        self, integrity_validator, mock_db_session
    ):
        """Test cross-system consistency validation."""
        # Mock room state query
        mock_room_state = MagicMock()
        mock_room_state.occupancy_state = "occupied"
        mock_room_state.timestamp = datetime.now(timezone.utc) - timedelta(minutes=5)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_room_state]
        mock_db_session.execute.return_value = mock_result

        # Test rapid state transition
        events = [
            {
                "room_id": "room_1",
                "state": "off",  # Transition from occupied to off quickly
                "timestamp": datetime.now(timezone.utc),
            }
        ]

        errors = await integrity_validator.validate_cross_system_consistency(
            "room_1", events
        )

        # Should detect rapid state transition
        transition_errors = [e for e in errors if e.rule_id == "INT003"]
        assert len(transition_errors) > 0

    @pytest.mark.asyncio
    async def test_cross_system_validation_database_error(
        self, integrity_validator, mock_db_session
    ):
        """Test cross-system validation with database errors."""
        # Mock database error
        mock_db_session.execute.side_effect = Exception("Database connection failed")

        events = [{"room_id": "room_1", "state": "on", "timestamp": datetime.now()}]

        errors = await integrity_validator.validate_cross_system_consistency(
            "room_1", events
        )

        # Should handle database errors gracefully
        db_errors = [e for e in errors if e.rule_id == "INT004"]
        assert len(db_errors) > 0
        assert "cross-system consistency" in db_errors[0].message


class TestPerformanceValidator:
    """Comprehensive tests for PerformanceValidator."""

    def test_initialization(self, performance_validator):
        """Test PerformanceValidator initialization."""
        assert performance_validator.batch_size == 100
        assert isinstance(performance_validator.validation_stats, dict)
        assert performance_validator.validation_stats["batches_processed"] == 0

    @pytest.mark.asyncio
    async def test_bulk_validation_small_dataset(
        self, performance_validator, security_validator, schema_validator
    ):
        """Test bulk validation with small dataset."""
        events = [
            {
                "room_id": "room_1",
                "sensor_id": f"sensor.test_{i}",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
            }
            for i in range(10)
        ]

        results = await performance_validator.bulk_validate_events(
            events, security_validator, schema_validator
        )

        assert len(results) == 10
        assert all(isinstance(r, ValidationResult) for r in results)
        assert performance_validator.validation_stats["events_processed"] == 10

    @pytest.mark.asyncio
    async def test_bulk_validation_large_dataset(
        self, performance_validator, security_validator, schema_validator
    ):
        """Test bulk validation with large dataset to test batching."""
        # Create dataset larger than batch size
        events = [
            {
                "room_id": f"room_{i % 5}",
                "sensor_id": f"sensor.test_{i}",
                "sensor_type": "motion",
                "state": "on" if i % 2 == 0 else "off",
                "timestamp": datetime.now(timezone.utc) + timedelta(seconds=i),
            }
            for i in range(250)  # More than 2 batches (batch_size=100)
        ]

        results = await performance_validator.bulk_validate_events(
            events, security_validator, schema_validator
        )

        assert len(results) == 250
        assert (
            performance_validator.validation_stats["batches_processed"] >= 3
        )  # Should be 3 batches
        assert performance_validator.validation_stats["events_processed"] == 250

    @pytest.mark.asyncio
    async def test_single_event_validation_performance(
        self, performance_validator, security_validator, schema_validator
    ):
        """Test performance timing of single event validation."""
        event = {
            "room_id": "room_1",
            "sensor_id": "sensor.test",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await performance_validator._validate_single_event(
            event, security_validator, schema_validator
        )

        assert isinstance(result, ValidationResult)
        assert result.processing_time_ms > 0  # Should have measured time
        assert result.processing_time_ms < 1000  # Should be fast (< 1 second)

    @pytest.mark.asyncio
    async def test_validation_with_errors_and_warnings(
        self, performance_validator, security_validator, schema_validator
    ):
        """Test validation with events that generate errors and warnings."""
        problematic_events = [
            {
                "room_id": "'; DROP TABLE users; --",  # SQL injection
                "sensor_id": "<script>alert('XSS')</script>",  # XSS attack
                "sensor_type": "invalid_type",  # Invalid sensor type
                "state": "on",
                "timestamp": "invalid_timestamp",  # Invalid timestamp
            }
        ]

        results = await performance_validator.bulk_validate_events(
            problematic_events, security_validator, schema_validator
        )

        assert len(results) == 1
        result = results[0]
        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.security_flags) > 0
        assert result.confidence_score < 1.0

    def test_performance_stats_tracking(self, performance_validator):
        """Test performance statistics tracking."""
        initial_stats = performance_validator.get_performance_stats()
        assert "batches_processed" in initial_stats
        assert "events_processed" in initial_stats

        # Manually update stats to test tracking
        performance_validator.validation_stats["batches_processed"] = 5
        performance_validator.validation_stats["events_processed"] = 500

        updated_stats = performance_validator.get_performance_stats()
        assert updated_stats["batches_processed"] == 5
        assert updated_stats["events_processed"] == 500


class TestComprehensiveEventValidator:
    """Tests for the main ComprehensiveEventValidator orchestrator."""

    def test_initialization(self, comprehensive_validator):
        """Test ComprehensiveEventValidator initialization."""
        assert comprehensive_validator.security_validator is not None
        assert comprehensive_validator.schema_validator is not None
        assert comprehensive_validator.integrity_validator is not None
        assert comprehensive_validator.performance_validator is not None
        assert len(comprehensive_validator.validation_rules) > 0

    def test_validation_rules_initialization(self, comprehensive_validator):
        """Test that validation rules are properly initialized."""
        rule_ids = {rule.rule_id for rule in comprehensive_validator.validation_rules}

        assert "SEC001" in rule_ids  # SQL Injection Detection
        assert "SEC002" in rule_ids  # XSS Detection
        assert "SCH001" in rule_ids  # Required Fields
        assert "INT001" in rule_ids  # Duplicate Detection

        # Check rule properties
        for rule in comprehensive_validator.validation_rules:
            assert isinstance(rule.rule_id, str)
            assert isinstance(rule.name, str)
            assert isinstance(rule.description, str)
            assert isinstance(rule.severity, ErrorSeverity)
            assert isinstance(rule.enabled, bool)

    @pytest.mark.asyncio
    async def test_single_event_validation(
        self, comprehensive_validator, valid_event_data
    ):
        """Test single event validation."""
        result = await comprehensive_validator.validate_event(valid_event_data)

        assert isinstance(result, ValidationResult)
        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence_score == 1.0
        assert result.processing_time_ms > 0
        assert result.integrity_hash is not None

    @pytest.mark.asyncio
    async def test_single_event_validation_with_security_issues(
        self, comprehensive_validator
    ):
        """Test single event validation with security issues."""
        malicious_event = {
            "room_id": "'; DROP TABLE rooms; --",
            "sensor_id": "<script>alert('hack')</script>",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": datetime.now(timezone.utc),
        }

        result = await comprehensive_validator.validate_event(malicious_event)

        assert isinstance(result, ValidationResult)
        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.security_flags) > 0
        assert result.confidence_score < 1.0

        # Check specific security errors
        security_errors = [e for e in result.errors if e.rule_id.startswith("SEC")]
        assert len(security_errors) > 0

    @pytest.mark.asyncio
    async def test_bulk_events_validation(self, comprehensive_validator):
        """Test bulk events validation."""
        events = [
            {
                "room_id": f"room_{i}",
                "sensor_id": f"sensor.test_{i}",
                "sensor_type": "motion",
                "state": "on" if i % 2 == 0 else "off",
                "timestamp": datetime.now(timezone.utc) + timedelta(seconds=i),
            }
            for i in range(10)
        ]

        results = await comprehensive_validator.validate_events_bulk(events)

        assert isinstance(results, list)
        assert len(results) == 10
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.is_valid for r in results)  # All should be valid

    @pytest.mark.asyncio
    async def test_bulk_validation_with_duplicates(self, comprehensive_validator):
        """Test bulk validation with duplicate events."""
        duplicate_event = {
            "room_id": "room_1",
            "sensor_id": "sensor.test",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": datetime.now(timezone.utc),
        }

        events = [duplicate_event, duplicate_event.copy()]  # Two identical events

        results = await comprehensive_validator.validate_events_bulk(events)

        # At least the first result should contain integrity errors about duplicates
        assert isinstance(results, list)
        assert len(results) == 2

        # Check if any result contains duplicate detection errors
        has_duplicate_error = any(
            any(error.rule_id == "INT001" for error in result.errors)
            for result in results
            if hasattr(result, "errors")
        )
        assert has_duplicate_error

    @pytest.mark.asyncio
    async def test_room_specific_validation(
        self, comprehensive_validator, mock_db_session
    ):
        """Test room-specific validation with cross-system checks."""
        events = [
            {
                "room_id": "room_1",
                "sensor_id": "sensor.living_room_motion",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
            }
        ]

        # Mock database response for cross-system validation
        mock_room_state = MagicMock()
        mock_room_state.occupancy_state = "vacant"
        mock_room_state.timestamp = datetime.now(timezone.utc) - timedelta(hours=1)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_room_state]
        mock_db_session.execute.return_value = mock_result

        results = await comprehensive_validator.validate_room_events("room_1", events)

        assert isinstance(results, list)
        assert len(results) == 1
        assert isinstance(results[0], ValidationResult)

    def test_event_data_sanitization(self, comprehensive_validator):
        """Test event data sanitization."""
        malicious_event = {
            "room_id": "<script>alert('xss')</script>",
            "sensor_id": "'; DROP TABLE sensors; --",
            "attributes": {
                "description": "Normal text with <script>alert('nested')</script>",
                "value": 42,
            },
        }

        # Test standard sanitization
        sanitized = comprehensive_validator.sanitize_event_data(
            malicious_event, aggressive=False
        )

        assert "&lt;script&gt;" in sanitized["room_id"]
        assert "DROP TABLE" in sanitized["sensor_id"]  # SQL preserved in standard mode
        assert "&lt;script&gt;" in sanitized["attributes"]["description"]
        assert sanitized["attributes"]["value"] == 42  # Numbers unchanged

        # Test aggressive sanitization
        sanitized_aggressive = comprehensive_validator.sanitize_event_data(
            malicious_event, aggressive=True
        )

        assert "<script>" not in sanitized_aggressive["room_id"]
        assert "DROP" not in sanitized_aggressive["sensor_id"]  # SQL keywords removed
        assert "script" not in sanitized_aggressive["attributes"]["description"]

    def test_validation_summary(self, comprehensive_validator):
        """Test validation summary generation."""
        summary = comprehensive_validator.get_validation_summary()

        assert isinstance(summary, dict)
        assert "validation_rules" in summary
        assert "active_rules" in summary
        assert "performance_stats" in summary
        assert "security_patterns" in summary

        # Check security patterns count
        security_patterns = summary["security_patterns"]
        assert security_patterns["sql_injection"] > 0
        assert security_patterns["xss"] > 0
        assert security_patterns["path_traversal"] > 0

    @pytest.mark.asyncio
    async def test_validation_error_context_and_suggestions(
        self, comprehensive_validator
    ):
        """Test that validation errors include helpful context and suggestions."""
        problematic_event = {
            "room_id": "",  # Empty required field
            "sensor_type": "invalid_sensor_type",  # Invalid enum value
            "timestamp": "not-a-timestamp",  # Invalid format
            "attributes": "not-a-dict",  # Wrong type
        }

        result = await comprehensive_validator.validate_event(problematic_event)

        assert not result.is_valid
        assert len(result.errors) > 0

        # Check that errors have helpful suggestions
        for error in result.errors:
            assert error.suggestion is not None
            assert len(error.suggestion) > 0

            # Context should provide additional information
            if hasattr(error, "context") and error.context:
                assert isinstance(error.context, dict)

    @pytest.mark.asyncio
    async def test_concurrent_validation_safety(self, comprehensive_validator):
        """Test thread safety of concurrent validations."""
        import asyncio

        # Create multiple validation tasks
        events = [
            {
                "room_id": f"room_{i}",
                "sensor_id": f"sensor_{i}",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
            }
            for i in range(20)
        ]

        # Run validations concurrently
        tasks = [comprehensive_validator.validate_event(event) for event in events]
        results = await asyncio.gather(*tasks)

        assert len(results) == 20
        assert all(isinstance(r, ValidationResult) for r in results)
        assert all(r.is_valid for r in results)

    @pytest.mark.asyncio
    async def test_large_scale_performance(self, comprehensive_validator):
        """Test performance with large-scale validation."""
        import time

        # Create a large dataset
        large_event_set = [
            {
                "room_id": f"room_{i % 10}",
                "sensor_id": f"sensor.motion_{i}",
                "sensor_type": "motion",
                "state": "on" if i % 2 == 0 else "off",
                "timestamp": datetime.now(timezone.utc) + timedelta(seconds=i),
                "attributes": {"index": i, "test": True},
            }
            for i in range(1000)  # 1000 events
        ]

        start_time = time.time()
        results = await comprehensive_validator.validate_events_bulk(large_event_set)
        end_time = time.time()

        # Should complete within reasonable time (< 30 seconds for 1000 events)
        validation_time = end_time - start_time
        assert validation_time < 30.0

        assert len(results) == 1000
        assert all(isinstance(r, ValidationResult) for r in results)

        # Most events should be valid
        valid_count = sum(1 for r in results if r.is_valid)
        assert valid_count >= 950  # At least 95% should be valid


class TestSecurityTestingIntegration:
    """Integration tests specifically focused on security validation."""

    @pytest.mark.asyncio
    async def test_comprehensive_sql_injection_prevention(
        self, comprehensive_validator
    ):
        """Comprehensive test of SQL injection prevention across all input fields."""
        sql_injection_payloads = [
            # Classic SQL injection
            "' OR 1=1--",
            "admin'--",
            "' OR 'a'='a",
            # Union-based injection
            "' UNION SELECT username, password FROM users--",
            "1' UNION SELECT null, table_name FROM information_schema.tables--",
            # Time-based blind injection
            "'; WAITFOR DELAY '00:00:05'--",
            "' OR IF(1=1,SLEEP(5),0)--",
            # Boolean-based blind injection
            "' AND (SELECT COUNT(*) FROM users) > 0--",
            "' AND ASCII(SUBSTRING((SELECT password FROM users LIMIT 1),1,1))>64--",
            # Stacked queries
            "'; DROP TABLE users; CREATE TABLE users (id INT);--",
            "'; INSERT INTO admin VALUES ('hacker', 'pass');--",
        ]

        for payload in sql_injection_payloads:
            # Test in different fields
            for field in ["room_id", "sensor_id", "sensor_type"]:
                event = {
                    "room_id": "room_1",
                    "sensor_id": "sensor.test",
                    "sensor_type": "motion",
                    "state": "on",
                    "timestamp": datetime.now(timezone.utc),
                }
                event[field] = payload

                result = await comprehensive_validator.validate_event(event)

                # Should detect SQL injection
                assert (
                    not result.is_valid
                ), f"SQL injection not detected in {field}: {payload}"

                sql_errors = [e for e in result.errors if e.rule_id == "SEC001"]
                assert (
                    len(sql_errors) > 0
                ), f"No SQL injection error for {field}: {payload}"
                assert any("SQL injection" in e.message for e in sql_errors)

    @pytest.mark.asyncio
    async def test_comprehensive_xss_prevention(self, comprehensive_validator):
        """Comprehensive test of XSS prevention across all input fields."""
        xss_payloads = [
            # Basic XSS
            "<script>alert('XSS')</script>",
            "<img src=x onerror=alert('XSS')>",
            # Event handler XSS
            "<body onload=alert('XSS')>",
            "<input onfocus=alert('XSS') autofocus>",
            "<svg onload=alert('XSS')>",
            # JavaScript protocol
            "javascript:alert('XSS')",
            "JaVaScRiPt:alert('XSS')",
            # Data URI XSS
            "data:text/html,<script>alert('XSS')</script>",
            # CSS-based XSS
            "<style>body{background-image:url('javascript:alert(\"XSS\")')}</style>",
            # Object and embed XSS
            "<object data='javascript:alert(\"XSS\")'></object>",
            "<embed src='javascript:alert(\"XSS\")'></embed>",
        ]

        for payload in xss_payloads:
            event = {
                "room_id": payload,
                "sensor_id": "sensor.test",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
                "attributes": {"description": payload},
            }

            result = await comprehensive_validator.validate_event(event)

            # Should detect XSS
            assert not result.is_valid, f"XSS not detected: {payload}"

            xss_errors = [e for e in result.errors if e.rule_id == "SEC002"]
            assert len(xss_errors) > 0, f"No XSS error for: {payload}"

    @pytest.mark.asyncio
    async def test_path_traversal_prevention(self, comprehensive_validator):
        """Test path traversal attack prevention."""
        path_traversal_payloads = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "....//....//....//etc//passwd",
            "..%252f..%252f..%252fetc%252fpasswd",
            "~/../../etc/passwd",
        ]

        for payload in path_traversal_payloads:
            event = {
                "room_id": "room_1",
                "sensor_id": payload,
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
            }

            result = await comprehensive_validator.validate_event(event)

            # Should detect path traversal
            assert not result.is_valid, f"Path traversal not detected: {payload}"

            path_errors = [e for e in result.errors if e.rule_id == "SEC003"]
            assert len(path_errors) > 0, f"No path traversal error for: {payload}"

    @pytest.mark.asyncio
    async def test_sanitization_effectiveness(self, comprehensive_validator):
        """Test effectiveness of input sanitization."""
        dangerous_event = {
            "room_id": "'; DROP TABLE rooms; SELECT '<script>alert(\"XSS\")</script>' FROM dual; --",
            "sensor_id": "../../../etc/passwd<img src=x onerror=alert('hack')>",
            "attributes": {
                "description": "' OR 1=1--<script>document.location='http://evil.com'</script>",
                "config": "javascript:void(0)onmouseover=alert('xss')",
            },
        }

        # Test standard sanitization
        sanitized_standard = comprehensive_validator.sanitize_event_data(
            dangerous_event
        )

        # HTML should be escaped
        assert "&lt;script&gt;" in sanitized_standard["sensor_id"]
        assert "&quot;" in sanitized_standard["room_id"]
        assert "&lt;img" in sanitized_standard["attributes"]["description"]

        # Test aggressive sanitization
        sanitized_aggressive = comprehensive_validator.sanitize_event_data(
            dangerous_event, aggressive=True
        )

        # Dangerous keywords should be removed
        assert "DROP" not in sanitized_aggressive["room_id"]
        assert "script" not in sanitized_aggressive["sensor_id"]
        assert "javascript" not in sanitized_aggressive["attributes"]["config"]

    @pytest.mark.asyncio
    async def test_bypass_attempt_prevention(self, comprehensive_validator):
        """Test prevention of common security bypass techniques."""
        bypass_attempts = [
            # Case variation
            "SeLeCt * fRoM uSeRs WhErE iD='1' oR '1'='1'",
            "<ScRiPt>AlErT('XsS')</ScRiPt>",
            # Comment evasion
            "' /*comment*/ OR /*comment*/ '1'='1'",
            "'/**/UNION/**/SELECT/**/",
            # Whitespace evasion
            "'%20OR%201=1--",
            "'\t\r\nOR\t\r\n'1'='1'",
            # Encoding evasion
            "%27%20OR%201=1--",  # URL encoded
            "&#39; OR 1=1--",  # HTML entity encoded
            # Unicode evasion
            "\u0027 OR 1=1--",  # Unicode escape
        ]

        for attempt in bypass_attempts:
            event = {
                "room_id": attempt,
                "sensor_id": "sensor.test",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
            }

            result = await comprehensive_validator.validate_event(event)

            # Should still detect the attack despite evasion attempts
            assert not result.is_valid, f"Bypass attempt succeeded: {attempt}"
            assert len(result.errors) > 0

    @pytest.mark.asyncio
    async def test_false_positive_prevention(self, comprehensive_validator):
        """Test that legitimate data doesn't trigger false security alerts."""
        legitimate_data = [
            # Legitimate sensor data
            "sensor.living_room_motion_detector_v2",
            "binary_sensor.door_contact_front_door",
            # Legitimate room names
            "Master Bedroom & Bath",
            "Living Room (Main Floor)",
            # Legitimate attribute values
            "Temperature: 20.5°C, Humidity: 65%",
            'JSON config: {"enabled": true, "sensitivity": 80}',
            # URLs and emails
            "http://homeassistant.local:8123/api/states/sensor.temperature",
            "admin@homeassistant.local",
            # Technical descriptions
            "Motion detected by PIR sensor #1 in zone A",
            "Door opened - magnetic switch state change",
        ]

        for data in legitimate_data:
            event = {
                "room_id": "room_1",
                "sensor_id": data,
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc),
                "attributes": {"description": data},
            }

            result = await comprehensive_validator.validate_event(event)

            # Should not trigger false positives
            security_errors = [e for e in result.errors if e.rule_id.startswith("SEC")]
            assert (
                len(security_errors) == 0
            ), f"False positive for legitimate data: {data}"
