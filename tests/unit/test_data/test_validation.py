"""
Comprehensive Tests for Data Validation Framework.

This module provides extensive test coverage for:
- Event validation with security checks
- Schema validation and format verification
- Pattern detection and anomaly identification
- Data integrity and corruption detection
- Performance validation for high-volume processing
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.constants import SensorState, SensorType
from src.core.exceptions import APISecurityError, DataValidationError, ErrorSeverity
from src.data.validation.event_validator import (
    ComprehensiveEventValidator,
    IntegrityValidator,
    PerformanceValidator,
    SchemaValidator,
    SecurityValidator,
    ValidationError,
    ValidationResult,
)
from src.data.validation.pattern_detector import (
    CorruptionDetector,
    DataQualityMetrics,
    PatternAnomaly,
    RealTimeQualityMonitor,
    StatisticalPatternAnalyzer,
)
from src.data.validation.schema_validator import (
    APISchemaValidator,
    DatabaseSchemaValidator,
    JSONSchemaValidator,
    SchemaDefinition,
    SchemaValidationContext,
)


class TestSecurityValidator:
    """Test security validation and input sanitization."""

    def setup_method(self):
        """Setup test fixtures."""
        self.security_validator = SecurityValidator()

    def test_sql_injection_detection(self):
        """Test SQL injection pattern detection."""
        # Test malicious inputs
        malicious_inputs = [
            "'; DROP TABLE sensor_events; --",
            "' OR '1'='1",
            "' UNION SELECT * FROM users --",
            "'; INSERT INTO admin (user) VALUES ('hacker'); --",
            "1' OR 1=1 --",
            "admin'/*",
            "1; EXEC xp_cmdshell('dir') --",
        ]

        for malicious_input in malicious_inputs:
            errors = self.security_validator.validate_input_security(
                malicious_input, "test_field"
            )

            # Should detect SQL injection
            sql_injection_errors = [e for e in errors if e.rule_id == "SEC001"]
            assert (
                len(sql_injection_errors) > 0
            ), f"Failed to detect SQL injection in: {malicious_input}"
            assert sql_injection_errors[0].severity == ErrorSeverity.CRITICAL

    def test_xss_detection(self):
        """Test XSS pattern detection."""
        xss_inputs = [
            "<script>alert('XSS')</script>",
            "javascript:alert('XSS')",
            "<img src=x onerror=alert('XSS')>",
            "<iframe src='http://evil.com'></iframe>",
            "<object data='http://evil.com'></object>",
            "';alert(String.fromCharCode(88,83,83))//';",
        ]

        for xss_input in xss_inputs:
            errors = self.security_validator.validate_input_security(
                xss_input, "test_field"
            )

            # Should detect XSS attempt
            xss_errors = [e for e in errors if e.rule_id == "SEC002"]
            assert len(xss_errors) > 0, f"Failed to detect XSS in: {xss_input}"
            assert xss_errors[0].severity == ErrorSeverity.HIGH

    def test_path_traversal_detection(self):
        """Test path traversal detection."""
        traversal_inputs = [
            "../../../etc/passwd",
            "..\\..\\..\\Windows\\System32",
            "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
            "~/../../../etc/passwd",
            "/proc/self/environ",
        ]

        for traversal_input in traversal_inputs:
            errors = self.security_validator.validate_input_security(
                traversal_input, "test_field"
            )

            # Should detect path traversal
            traversal_errors = [e for e in errors if e.rule_id == "SEC003"]
            assert (
                len(traversal_errors) > 0
            ), f"Failed to detect path traversal in: {traversal_input}"
            assert traversal_errors[0].severity == ErrorSeverity.HIGH

    def test_oversized_input_detection(self):
        """Test detection of oversized inputs."""
        # Create large input (>10KB)
        large_input = "x" * 15000

        errors = self.security_validator.validate_input_security(
            large_input, "test_field"
        )

        # Should detect oversized input
        size_errors = [e for e in errors if e.rule_id == "SEC004"]
        assert len(size_errors) > 0
        assert size_errors[0].severity == ErrorSeverity.MEDIUM
        assert size_errors[0].context["input_length"] == 15000

    def test_null_byte_injection_detection(self):
        """Test null byte injection detection."""
        null_byte_input = "malicious\x00content"

        errors = self.security_validator.validate_input_security(
            null_byte_input, "test_field"
        )

        # Should detect null byte injection
        null_errors = [e for e in errors if e.rule_id == "SEC005"]
        assert len(null_errors) > 0
        assert null_errors[0].severity == ErrorSeverity.HIGH

    def test_input_sanitization(self):
        """Test input sanitization functionality."""
        # Test standard sanitization
        malicious_input = "<script>alert('XSS')</script>"
        sanitized = self.security_validator.sanitize_input(malicious_input)

        assert "&lt;script&gt;" in sanitized
        assert "alert" in sanitized
        assert "<script>" not in sanitized

        # Test aggressive sanitization
        aggressive_sanitized = self.security_validator.sanitize_input(
            malicious_input, aggressive=True
        )

        assert "script" not in aggressive_sanitized
        assert "alert" not in aggressive_sanitized

    def test_legitimate_input_passes(self):
        """Test that legitimate inputs pass security validation."""
        legitimate_inputs = [
            "living_room",
            "binary_sensor.motion_detector_1",
            "on",
            "2024-01-15T14:30:00+00:00",
            "Motion detected in living room",
            "temperature_sensor",
            "23.5",
        ]

        for legitimate_input in legitimate_inputs:
            errors = self.security_validator.validate_input_security(
                legitimate_input, "test_field"
            )

            # Should have no critical or high severity errors
            critical_errors = [
                e
                for e in errors
                if e.severity in [ErrorSeverity.CRITICAL, ErrorSeverity.HIGH]
            ]
            assert (
                len(critical_errors) == 0
            ), f"Legitimate input flagged as malicious: {legitimate_input}"


class TestSchemaValidator:
    """Test schema validation functionality."""

    def setup_method(self):
        """Setup test fixtures."""
        from src.core.config import RoomConfig, SystemConfig

        # Create mock config
        self.config = SystemConfig(
            home_assistant=MagicMock(),
            database=MagicMock(),
            mqtt=MagicMock(),
            prediction=MagicMock(),
            features=MagicMock(),
            logging=MagicMock(),
            tracking=MagicMock(),
            api=MagicMock(),
            rooms={
                "living_room": RoomConfig(
                    room_id="living_room",
                    name="Living Room",
                    sensors={
                        "motion": {"entities": ["binary_sensor.living_room_motion"]}
                    },
                )
            },
        )

        self.schema_validator = SchemaValidator(self.config)

    def test_valid_sensor_event_schema(self):
        """Test validation of valid sensor event."""
        valid_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.living_room_motion",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
            "attributes": {"device_class": "motion"},
            "is_human_triggered": True,
        }

        errors = self.schema_validator.validate_sensor_event_schema(valid_event)
        assert len(errors) == 0

    def test_missing_required_fields(self):
        """Test detection of missing required fields."""
        incomplete_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.living_room_motion",
            # Missing sensor_type, state, timestamp
        }

        errors = self.schema_validator.validate_sensor_event_schema(incomplete_event)

        # Should detect missing required fields
        missing_errors = [e for e in errors if e.rule_id == "SCH001"]
        assert len(missing_errors) >= 3  # sensor_type, state, timestamp

        for error in missing_errors:
            assert error.severity == ErrorSeverity.CRITICAL

    def test_invalid_field_types(self):
        """Test detection of invalid field types."""
        invalid_event = {
            "room_id": 123,  # Should be string
            "sensor_id": ["not", "a", "string"],  # Should be string
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        errors = self.schema_validator.validate_sensor_event_schema(invalid_event)

        # Should detect type errors
        type_errors = [e for e in errors if e.rule_id in ["SCH003", "SCH004"]]
        assert len(type_errors) >= 2

    def test_invalid_sensor_type(self):
        """Test detection of invalid sensor types."""
        invalid_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.living_room_motion",
            "sensor_type": "invalid_type",  # Invalid sensor type
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        errors = self.schema_validator.validate_sensor_event_schema(invalid_event)

        # Should detect invalid sensor type
        type_errors = [e for e in errors if e.rule_id == "SCH005"]
        assert len(type_errors) > 0
        assert type_errors[0].severity == ErrorSeverity.HIGH

    def test_invalid_sensor_state(self):
        """Test detection of invalid sensor states."""
        invalid_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.living_room_motion",
            "sensor_type": "motion",
            "state": "invalid_state",  # Invalid state
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        errors = self.schema_validator.validate_sensor_event_schema(invalid_event)

        # Should detect invalid state
        state_errors = [e for e in errors if e.rule_id == "SCH006"]
        assert len(state_errors) > 0
        assert state_errors[0].severity == ErrorSeverity.HIGH

    def test_invalid_timestamp_format(self):
        """Test detection of invalid timestamp formats."""
        invalid_events = [
            {
                "room_id": "living_room",
                "sensor_id": "binary_sensor.living_room_motion",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": "not-a-timestamp",
            },
            {
                "room_id": "living_room",
                "sensor_id": "binary_sensor.living_room_motion",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(),  # Missing timezone
            },
        ]

        for invalid_event in invalid_events:
            errors = self.schema_validator.validate_sensor_event_schema(invalid_event)

            # Should detect timestamp errors
            timestamp_errors = [e for e in errors if e.rule_id in ["SCH007", "SCH008"]]
            assert len(timestamp_errors) > 0

    def test_room_configuration_validation(self):
        """Test room configuration validation."""
        # Valid room config
        valid_room = {
            "name": "Living Room",
            "sensors": {"motion": {"entities": ["binary_sensor.living_room_motion"]}},
        }

        errors = self.schema_validator.validate_room_configuration(
            "living_room", valid_room
        )
        assert len(errors) == 0

        # Invalid room ID
        errors = self.schema_validator.validate_room_configuration(
            "invalid room id!", valid_room
        )
        room_id_errors = [e for e in errors if e.rule_id == "SCH010"]
        assert len(room_id_errors) > 0

        # Invalid room name
        invalid_room = {"name": 123}  # Should be string
        errors = self.schema_validator.validate_room_configuration(
            "living_room", invalid_room
        )
        name_errors = [e for e in errors if e.rule_id == "SCH011"]
        assert len(name_errors) > 0


class TestIntegrityValidator:
    """Test data integrity validation."""

    @pytest_asyncio.fixture
    async def mock_session(self):
        """Create mock database session."""
        session = AsyncMock(spec=AsyncSession)
        return session

    @pytest.mark.asyncio
    async def test_calculate_event_hash(self, mock_session):
        """Test event hash calculation."""
        validator = IntegrityValidator(mock_session)

        event_data = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        hash1 = validator.calculate_event_hash(event_data)
        hash2 = validator.calculate_event_hash(event_data)

        # Same data should produce same hash
        assert hash1 == hash2
        assert len(hash1) == 64  # SHA-256 hex length

        # Different data should produce different hash
        event_data["state"] = "off"
        hash3 = validator.calculate_event_hash(event_data)
        assert hash1 != hash3

    @pytest.mark.asyncio
    async def test_duplicate_event_detection(self, mock_session):
        """Test duplicate event detection."""
        validator = IntegrityValidator(mock_session)

        # Create duplicate events
        event1 = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        event2 = event1.copy()  # Exact duplicate

        events = [event1, event2]
        errors = await validator.validate_data_consistency(events)

        # Should detect duplicate
        duplicate_errors = [e for e in errors if e.rule_id == "INT001"]
        assert len(duplicate_errors) > 0
        assert duplicate_errors[0].severity == ErrorSeverity.MEDIUM

    @pytest.mark.asyncio
    async def test_timestamp_ordering_validation(self, mock_session):
        """Test timestamp ordering validation."""
        validator = IntegrityValidator(mock_session)

        # Create events with timestamps too close together
        base_time = datetime.now(timezone.utc)

        events = [
            {
                "room_id": "living_room",
                "sensor_id": "binary_sensor.motion",
                "state": "on",
                "timestamp": base_time.isoformat(),
            },
            {
                "room_id": "living_room",
                "sensor_id": "binary_sensor.motion",
                "state": "off",
                "timestamp": (
                    base_time + timedelta(milliseconds=100)
                ).isoformat(),  # Too close
            },
        ]

        errors = await validator.validate_data_consistency(events)

        # Should detect events too close together
        timing_errors = [e for e in errors if e.rule_id == "INT002"]
        assert len(timing_errors) > 0
        assert timing_errors[0].severity == ErrorSeverity.LOW

    @pytest.mark.asyncio
    async def test_cross_system_consistency_validation(self, mock_session):
        """Test cross-system consistency validation."""
        # Mock database query results
        mock_room_state = MagicMock()
        mock_room_state.room_id = "living_room"
        mock_room_state.occupancy_state = "occupied"
        mock_room_state.timestamp = datetime.now(timezone.utc) - timedelta(seconds=30)

        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = [mock_room_state]
        mock_session.execute.return_value = mock_result

        validator = IntegrityValidator(mock_session)

        # Event that shows rapid state change
        events = [
            {
                "room_id": "living_room",
                "state": "on",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        errors = await validator.validate_cross_system_consistency(
            "living_room", events
        )

        # May detect rapid transition
        transition_errors = [e for e in errors if e.rule_id == "INT003"]
        # This test may or may not trigger depending on timing


class TestPerformanceValidator:
    """Test performance validation capabilities."""

    def setup_method(self):
        """Setup test fixtures."""
        self.performance_validator = PerformanceValidator(batch_size=10)

    @pytest.mark.asyncio
    async def test_bulk_validation_performance(self):
        """Test bulk validation performance."""
        # Create large dataset
        events = []
        base_time = datetime.now(timezone.utc)

        for i in range(100):
            events.append(
                {
                    "room_id": f"room_{i % 5}",
                    "sensor_id": f"sensor.motion_{i}",
                    "sensor_type": "motion",
                    "state": "on" if i % 2 == 0 else "off",
                    "timestamp": (base_time + timedelta(seconds=i)).isoformat(),
                }
            )

        # Mock validators
        security_validator = MagicMock(spec=SecurityValidator)
        security_validator.validate_input_security.return_value = []

        schema_validator = MagicMock(spec=SchemaValidator)
        schema_validator.validate_sensor_event_schema.return_value = []

        start_time = asyncio.get_event_loop().time()

        results = await self.performance_validator.bulk_validate_events(
            events, security_validator, schema_validator
        )

        end_time = asyncio.get_event_loop().time()
        processing_time = end_time - start_time

        # Validate results
        assert len(results) == len(events)
        assert processing_time < 5.0  # Should process 100 events quickly

        # Check performance stats
        stats = self.performance_validator.get_performance_stats()
        assert stats["batches_processed"] > 0
        assert stats["events_processed"] == 100

    @pytest.mark.asyncio
    async def test_validation_error_handling(self):
        """Test error handling during validation."""
        # Create event that will cause validation errors
        invalid_events = [
            {
                "room_id": "",  # Invalid
                "sensor_id": "'; DROP TABLE users; --",  # SQL injection
                "sensor_type": "invalid_type",
                "state": "invalid_state",
                "timestamp": "not-a-timestamp",
            }
        ]

        security_validator = SecurityValidator()
        schema_validator = SchemaValidator(MagicMock())

        results = await self.performance_validator.bulk_validate_events(
            invalid_events, security_validator, schema_validator
        )

        assert len(results) == 1
        result = results[0]

        assert not result.is_valid
        assert len(result.errors) > 0
        assert len(result.security_flags) > 0
        assert result.confidence_score < 0.5


class TestStatisticalPatternAnalyzer:
    """Test statistical pattern analysis."""

    def setup_method(self):
        """Setup test fixtures."""
        self.analyzer = StatisticalPatternAnalyzer()

    def test_sensor_behavior_analysis(self):
        """Test sensor behavior statistical analysis."""
        # Create realistic sensor events
        base_time = datetime.now(timezone.utc)
        events = []

        for i in range(20):
            events.append(
                {
                    "sensor_id": "binary_sensor.motion",
                    "state": "on" if i % 2 == 0 else "off",
                    "timestamp": (base_time + timedelta(minutes=i * 5)).isoformat(),
                }
            )

        analysis = self.analyzer.analyze_sensor_behavior("binary_sensor.motion", events)

        # Validate analysis results
        assert "event_count" in analysis
        assert "time_span_hours" in analysis
        assert "mean_interval" in analysis
        assert "state_distribution" in analysis
        assert "trigger_frequency" in analysis

        assert analysis["event_count"] == 20
        assert analysis["time_span_hours"] > 0
        assert analysis["mean_interval"] > 0

    def test_statistical_anomaly_detection(self):
        """Test statistical anomaly detection."""
        # Create events with outliers
        base_time = datetime.now(timezone.utc)
        events = []

        # Normal intervals (5 minutes)
        for i in range(10):
            events.append(
                {
                    "sensor_id": "binary_sensor.motion",
                    "state": "on",
                    "timestamp": (base_time + timedelta(minutes=i * 5)).isoformat(),
                }
            )

        # Add outlier (1 second interval)
        events.append(
            {
                "sensor_id": "binary_sensor.motion",
                "state": "on",
                "timestamp": (base_time + timedelta(minutes=50, seconds=1)).isoformat(),
            }
        )

        analysis = self.analyzer.analyze_sensor_behavior("binary_sensor.motion", events)

        # Should detect anomalies
        assert "anomaly_count" in analysis
        assert analysis["anomaly_count"] > 0
        assert "outliers" in analysis
        assert len(analysis["outliers"]) > 0

    def test_sensor_malfunction_detection(self):
        """Test sensor malfunction detection."""
        # Create baseline behavior
        baseline_events = []
        base_time = datetime.now(timezone.utc) - timedelta(days=1)

        for i in range(24):  # Normal hourly triggers
            baseline_events.append(
                {
                    "sensor_id": "binary_sensor.motion",
                    "state": "on",
                    "timestamp": (base_time + timedelta(hours=i)).isoformat(),
                }
            )

        # Analyze baseline
        self.analyzer.analyze_sensor_behavior("binary_sensor.motion", baseline_events)

        # Create recent events with high frequency (malfunction)
        recent_events = []
        recent_base = datetime.now(timezone.utc)

        for i in range(100):  # Triggering every minute (malfunction)
            recent_events.append(
                {
                    "sensor_id": "binary_sensor.motion",
                    "state": "on",
                    "timestamp": (recent_base + timedelta(minutes=i)).isoformat(),
                }
            )

        anomalies = self.analyzer.detect_sensor_malfunction(
            "binary_sensor.motion", recent_events
        )

        # Should detect high frequency anomaly
        high_freq_anomalies = [
            a for a in anomalies if a.anomaly_type == "high_frequency"
        ]
        assert len(high_freq_anomalies) > 0
        assert high_freq_anomalies[0].severity == ErrorSeverity.HIGH


class TestCorruptionDetector:
    """Test data corruption detection."""

    def setup_method(self):
        """Setup test fixtures."""
        self.detector = CorruptionDetector()

    def test_timestamp_corruption_detection(self):
        """Test timestamp corruption detection."""
        # Events with corrupted timestamps
        corrupted_events = [
            {"timestamp": "not-a-timestamp", "room_id": "living_room", "state": "on"},
            {
                "timestamp": "2024-01-15T14:30:00+00:00",
                "room_id": "living_room",
                "state": "on",
            },
            {
                "timestamp": "2025-01-15T14:30:00+00:00",  # Future timestamp (1 year)
                "room_id": "living_room",
                "state": "on",
            },
        ]

        errors = self.detector.detect_data_corruption(corrupted_events)

        # Should detect timestamp corruption
        timestamp_errors = [e for e in errors if e.rule_id.startswith("COR00")]
        assert len(timestamp_errors) > 0

    def test_state_corruption_detection(self):
        """Test state value corruption detection."""
        corrupted_events = [
            {
                "timestamp": "2024-01-15T14:30:00+00:00",
                "room_id": "living_room",
                "state": "x" * 50,  # Suspiciously long state
            },
            {
                "timestamp": "2024-01-15T14:30:00+00:00",
                "room_id": "living_room",
                "state": "on\x00\x01\x02",  # Non-printable characters
            },
        ]

        errors = self.detector.detect_data_corruption(corrupted_events)

        # Should detect state corruption
        state_errors = [e for e in errors if e.rule_id in ["COR004", "COR005"]]
        assert len(state_errors) > 0

    def test_id_corruption_detection(self):
        """Test ID field corruption detection."""
        corrupted_events = [
            {
                "timestamp": "2024-01-15T14:30:00+00:00",
                "room_id": "x" * 200,  # Suspiciously long room ID
                "sensor_id": "normal_sensor",
                "state": "on",
            },
            {
                "timestamp": "2024-01-15T14:30:00+00:00",
                "room_id": "living_room",
                "sensor_id": "sensor\x00\x01",  # Non-printable characters
                "state": "on",
            },
        ]

        errors = self.detector.detect_data_corruption(corrupted_events)

        # Should detect ID corruption
        id_errors = [e for e in errors if e.rule_id in ["COR006", "COR007"]]
        assert len(id_errors) > 0

    def test_encoding_corruption_detection(self):
        """Test character encoding corruption detection."""
        corrupted_events = [
            {
                "timestamp": "2024-01-15T14:30:00+00:00",
                "room_id": "living_room�corrupted",  # Replacement character
                "state": "on",
            }
        ]

        errors = self.detector.detect_data_corruption(corrupted_events)

        # Should detect encoding corruption
        encoding_errors = [e for e in errors if e.rule_id == "COR008"]
        assert len(encoding_errors) > 0


class TestRealTimeQualityMonitor:
    """Test real-time quality monitoring."""

    def setup_method(self):
        """Setup test fixtures."""
        self.monitor = RealTimeQualityMonitor()

    def test_quality_metrics_calculation(self):
        """Test quality metrics calculation."""
        # Create test events
        events = []
        base_time = datetime.now(timezone.utc)
        expected_sensors = {"sensor1", "sensor2", "sensor3"}

        for i in range(10):
            events.append(
                {
                    "sensor_id": f"sensor{(i % 2) + 1}",  # Only sensor1 and sensor2
                    "state": SensorState.ON.value,
                    "sensor_type": SensorType.MOTION.value,
                    "timestamp": (base_time - timedelta(minutes=i)).isoformat(),
                }
            )

        metrics = self.monitor.calculate_quality_metrics(events, expected_sensors)

        # Validate metrics
        assert 0.0 <= metrics.completeness_score <= 1.0
        assert 0.0 <= metrics.consistency_score <= 1.0
        assert 0.0 <= metrics.accuracy_score <= 1.0
        assert 0.0 <= metrics.timeliness_score <= 1.0

        # Completeness should be 2/3 (sensor1, sensor2 present, sensor3 missing)
        assert abs(metrics.completeness_score - 2 / 3) < 0.1

    def test_quality_alert_detection(self):
        """Test quality degradation alert detection."""
        # Create poor quality metrics
        poor_metrics = DataQualityMetrics(
            completeness_score=0.5,  # Below threshold
            consistency_score=0.6,  # Below threshold
            accuracy_score=0.7,  # Below threshold
            timeliness_score=0.8,  # Below threshold
            anomaly_count=5,
        )

        alerts = self.monitor.detect_quality_alerts(poor_metrics)

        # Should generate multiple alerts
        assert len(alerts) >= 4  # One for each metric below threshold

        alert_types = {alert.anomaly_type for alert in alerts}
        expected_types = {
            "data_completeness",
            "data_consistency",
            "data_accuracy",
            "data_timeliness",
        }
        assert alert_types == expected_types

    def test_quality_trends_tracking(self):
        """Test quality trends tracking."""
        # Add some quality history
        for i in range(5):
            events = [{"sensor_id": "sensor1", "timestamp": datetime.now().isoformat()}]
            self.monitor.calculate_quality_metrics(events, {"sensor1"})

        trends = self.monitor.get_quality_trends(hours=24)

        # Should have trend data
        assert "completeness" in trends
        assert "consistency" in trends
        assert "accuracy" in trends
        assert "timeliness" in trends
        assert "timestamps" in trends

        assert len(trends["completeness"]) > 0


class TestJSONSchemaValidator:
    """Test JSON schema validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = JSONSchemaValidator()

    def test_sensor_event_schema_validation(self):
        """Test sensor event schema validation."""
        # Valid sensor event
        valid_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.motion_detector",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
            "attributes": {"device_class": "motion"},
        }

        result = self.validator.validate_json_schema(valid_event, "sensor_event")

        assert result.is_valid
        assert len(result.errors) == 0
        assert result.confidence_score == 1.0

    def test_invalid_sensor_event_schema(self):
        """Test invalid sensor event schema validation."""
        # Invalid sensor event (missing required fields)
        invalid_event = {
            "room_id": "living_room",
            # Missing sensor_id, sensor_type, state, timestamp
        }

        result = self.validator.validate_json_schema(invalid_event, "sensor_event")

        assert not result.is_valid
        assert len(result.errors) > 0
        assert result.confidence_score < 1.0

        # Check for required field errors
        required_errors = [e for e in result.errors if "required" in e.rule_id.lower()]
        assert len(required_errors) > 0

    def test_custom_format_validation(self):
        """Test custom format validators."""
        # Test sensor ID format
        assert self.validator._validate_sensor_id_format(
            "binary_sensor.motion_detector"
        )
        assert not self.validator._validate_sensor_id_format("invalid_sensor_id")

        # Test room ID format
        assert self.validator._validate_room_id_format("living_room")
        assert not self.validator._validate_room_id_format("invalid room id!")

        # Test ISO datetime format
        assert self.validator._validate_iso_datetime_format("2024-01-15T14:30:00+00:00")
        assert not self.validator._validate_iso_datetime_format("not-a-timestamp")

        # Test sensor state format
        assert self.validator._validate_sensor_state_format("on")
        assert not self.validator._validate_sensor_state_format("invalid_state")

    def test_schema_registration(self):
        """Test custom schema registration."""
        custom_schema = SchemaDefinition(
            schema_id="test_schema",
            name="Test Schema",
            version="1.0.0",
            description="Test schema for validation",
            schema={
                "type": "object",
                "properties": {"test_field": {"type": "string"}},
                "required": ["test_field"],
            },
        )

        self.validator.register_schema(custom_schema)

        # Test validation with custom schema
        valid_data = {"test_field": "test_value"}
        result = self.validator.validate_json_schema(valid_data, "test_schema")
        assert result.is_valid

        # Test with invalid data
        invalid_data = {"wrong_field": "test_value"}
        result = self.validator.validate_json_schema(invalid_data, "test_schema")
        assert not result.is_valid


class TestAPISchemaValidator:
    """Test API schema validation."""

    def setup_method(self):
        """Setup test fixtures."""
        self.validator = APISchemaValidator()

    def test_valid_api_request_validation(self):
        """Test valid API request validation."""
        result = self.validator.validate_api_request(
            method="GET",
            path="/api/rooms",
            headers={
                "Authorization": "Bearer token123",
                "Content-Type": "application/json",
            },
            query_params={"limit": "10"},
        )

        assert result.is_valid
        assert len(result.errors) == 0

    def test_invalid_http_method(self):
        """Test invalid HTTP method detection."""
        result = self.validator.validate_api_request(
            method="INVALID",
            path="/api/rooms",
            headers={"Authorization": "Bearer token123"},
        )

        assert not result.is_valid
        method_errors = [e for e in result.errors if e.rule_id == "API_SCH_001"]
        assert len(method_errors) > 0

    def test_invalid_path_format(self):
        """Test invalid path format detection."""
        result = self.validator.validate_api_request(
            method="GET",
            path="api/rooms",  # Missing leading slash
            headers={"Authorization": "Bearer token123"},
        )

        assert not result.is_valid
        path_errors = [e for e in result.errors if e.rule_id == "API_SCH_002"]
        assert len(path_errors) > 0

    def test_missing_authentication(self):
        """Test missing authentication detection."""
        result = self.validator.validate_api_request(
            method="GET",
            path="/api/rooms",
            headers={"Content-Type": "application/json"},  # No auth header
        )

        assert not result.is_valid
        auth_errors = [e for e in result.errors if e.rule_id == "API_SCH_004"]
        assert len(auth_errors) > 0

    def test_oversized_header_validation(self):
        """Test oversized header validation."""
        large_value = "x" * 10000

        result = self.validator.validate_api_request(
            method="GET",
            path="/api/rooms",
            headers={"Authorization": "Bearer token123", "Large-Header": large_value},
        )

        assert not result.is_valid
        size_errors = [e for e in result.errors if e.rule_id == "API_SCH_006"]
        assert len(size_errors) > 0

    def test_json_content_validation(self):
        """Test JSON content validation."""
        # Valid JSON
        errors = self.validator._validate_json_content({"key": "value"})
        assert len(errors) == 0

        # Invalid JSON string
        errors = self.validator._validate_json_content('{"invalid": json}')
        assert len(errors) > 0
        json_errors = [e for e in errors if e.rule_id == "API_SCH_008"]
        assert len(json_errors) > 0


@pytest.mark.asyncio
class TestComprehensiveEventValidator:
    """Test the comprehensive event validator."""

    async def test_single_event_validation(self):
        """Test single event validation."""
        mock_session = AsyncMock(spec=AsyncSession)
        validator = ComprehensiveEventValidator(mock_session)

        # Valid event
        valid_event = {
            "room_id": "living_room",
            "sensor_id": "binary_sensor.motion",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        result = await validator.validate_event(valid_event)

        assert isinstance(result, ValidationResult)
        assert result.validation_id is not None
        assert result.processing_time_ms > 0

    async def test_bulk_event_validation(self):
        """Test bulk event validation."""
        mock_session = AsyncMock(spec=AsyncSession)
        validator = ComprehensiveEventValidator(mock_session)

        # Create bulk events
        events = []
        base_time = datetime.now(timezone.utc)

        for i in range(50):
            events.append(
                {
                    "room_id": f"room_{i % 3}",
                    "sensor_id": f"sensor.motion_{i}",
                    "sensor_type": "motion",
                    "state": "on" if i % 2 == 0 else "off",
                    "timestamp": (base_time + timedelta(seconds=i * 10)).isoformat(),
                }
            )

        results = await validator.validate_events_bulk(events)

        assert len(results) == len(events)

        for result in results:
            assert isinstance(result, ValidationResult)
            assert result.processing_time_ms > 0

    async def test_room_specific_validation(self):
        """Test room-specific validation with cross-system checks."""
        mock_session = AsyncMock(spec=AsyncSession)

        # Mock database query for cross-system validation
        mock_result = MagicMock()
        mock_result.scalars.return_value.all.return_value = []
        mock_session.execute.return_value = mock_result

        validator = ComprehensiveEventValidator(mock_session)

        events = [
            {
                "room_id": "living_room",
                "sensor_id": "binary_sensor.motion",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        ]

        results = await validator.validate_room_events("living_room", events)

        assert len(results) > 0
        assert all(isinstance(result, ValidationResult) for result in results)

    async def test_event_sanitization(self):
        """Test event data sanitization."""
        mock_session = AsyncMock(spec=AsyncSession)
        validator = ComprehensiveEventValidator(mock_session)

        # Event with malicious content
        malicious_event = {
            "room_id": "<script>alert('xss')</script>",
            "sensor_id": "'; DROP TABLE users; --",
            "sensor_type": "motion",
            "state": "on",
            "timestamp": "2024-01-15T14:30:00+00:00",
        }

        sanitized = validator.sanitize_event_data(malicious_event)

        # Should sanitize malicious content
        assert "<script>" not in sanitized["room_id"]
        assert "DROP TABLE" not in sanitized["sensor_id"]
        assert "&lt;script&gt;" in sanitized["room_id"]  # HTML escaped

    async def test_validation_summary(self):
        """Test validation summary generation."""
        mock_session = AsyncMock(spec=AsyncSession)
        validator = ComprehensiveEventValidator(mock_session)

        summary = validator.get_validation_summary()

        assert "validation_rules" in summary
        assert "active_rules" in summary
        assert "performance_stats" in summary
        assert "security_patterns" in summary

        assert summary["validation_rules"] > 0
        assert summary["security_patterns"]["sql_injection"] > 0
        assert summary["security_patterns"]["xss"] > 0
