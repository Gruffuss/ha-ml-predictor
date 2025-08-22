"""
Comprehensive Event Validation System for Home Assistant ML Predictor.

This module provides comprehensive validation for sensor events including:
- Schema validation and format checking
- Data integrity validation
- Security input sanitization
- Cross-system consistency checks
- Performance-optimized bulk validation
"""

import asyncio
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import logging
import re
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ...core.config import SystemConfig, get_config
from ...core.constants import (
    MIN_EVENT_SEPARATION,
    PRESENCE_STATES,
    SensorState,
    SensorType,
)
from ...core.exceptions import (
    ErrorSeverity,
)
from ..storage.models import RoomState

logger = logging.getLogger(__name__)


@dataclass
class ValidationRule:
    """Represents a data validation rule."""

    rule_id: str
    name: str
    description: str
    severity: ErrorSeverity
    enabled: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationError:
    """Detailed validation error information."""

    rule_id: str
    field: str
    value: Any
    message: str
    severity: ErrorSeverity
    suggestion: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Result of comprehensive event validation."""

    is_valid: bool
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    confidence_score: float = 1.0
    processing_time_ms: float = 0.0
    security_flags: List[str] = field(default_factory=list)
    integrity_hash: Optional[str] = None
    validation_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    @property
    def has_errors(self) -> bool:
        """Check if validation has any errors."""
        return len(self.errors) > 0

    @property
    def has_warnings(self) -> bool:
        """Check if validation has any warnings."""
        return len(self.warnings) > 0

    @property
    def has_security_issues(self) -> bool:
        """Check if validation has security concerns."""
        return len(self.security_flags) > 0


class SecurityValidator:
    """Handles security-focused validation and input sanitization."""

    # SQL injection patterns
    SQL_INJECTION_PATTERNS = [
        r"(?i)(union\s+select)",
        r"(?i)(drop\s+table)",
        r"(?i)(delete\s+from)",
        r"(?i)(insert\s+into)",
        r"(?i)(update\s+set)",
        r"(?i)(exec\s*\()",
        r"(?i)(script\s*>)",
        r"(?i)(javascript\s*:)",
        r"['\"]\s*;\s*--",
        r"['\"]\s*or\s+['\"]*1['\"]*\s*=\s*['\"]*1",
        r"\bxp_cmdshell\b",
        r"['\"].*?/\*",  # Quote followed by /* comment
        r"['\"].*?--",  # Quote followed by -- comment
        r"['\"].*?#",  # Quote followed by # comment (MySQL style)
        r"(?i)or\s+1\s*=\s*1",
        r"(?i)sleep\s*\(",
        r"(?i)waitfor\s+delay",
        r"(?i)and\s+.*?\s*>\s*0",
        r"(?i)select\s+.*?\s+from",
        r"['\"];.*?;",  # Multiple statements
    ]

    # XSS patterns
    XSS_PATTERNS = [
        r"<\s*script[^>]*>.*?</\s*script\s*>",
        r"javascript\s*:",
        r"on\w+\s*=",
        r"<\s*iframe[^>]*>",
        r"<\s*object[^>]*>",
        r"<\s*embed[^>]*>",
        r"<\s*link[^>]*>",
        r"<\s*meta[^>]*>",
        r"alert\s*\(",
        r"String\.fromCharCode",
        r"document\.",
        r"window\.",
        r"eval\s*\(",
        r"expression\s*\(",
    ]

    # Path traversal patterns
    PATH_TRAVERSAL_PATTERNS = [
        r"\.\./",
        r"\.\.\\",
        r"%2e%2e%2f",
        r"%2e%2e\\",
        r"~",
        r"/etc/passwd",
        r"C:\\Windows",
        r"/proc/",
    ]

    def __init__(self):
        """Initialize security validator."""
        self.compiled_patterns = {
            "sql_injection": [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.SQL_INJECTION_PATTERNS
            ],
            "xss": [
                re.compile(pattern, re.IGNORECASE) for pattern in self.XSS_PATTERNS
            ],
            "path_traversal": [
                re.compile(pattern, re.IGNORECASE)
                for pattern in self.PATH_TRAVERSAL_PATTERNS
            ],
        }

    def validate_input_security(
        self, value: Any, field_name: str
    ) -> List[ValidationError]:
        """Comprehensive security validation of input values."""
        errors = []

        if not isinstance(value, (str, int, float, bool)):
            # Convert to string for security checking
            str_value = str(value)
        else:
            str_value = str(value)

        # Check for SQL injection attempts
        for pattern in self.compiled_patterns["sql_injection"]:
            if pattern.search(str_value):
                errors.append(
                    ValidationError(
                        rule_id="SEC001",
                        field=field_name,
                        value=value,
                        message=f"Potential SQL injection attempt detected in {field_name}",
                        severity=ErrorSeverity.CRITICAL,
                        suggestion="Remove SQL keywords and special characters",
                        context={"pattern_matched": pattern.pattern},
                    )
                )

        # Check for XSS attempts
        for pattern in self.compiled_patterns["xss"]:
            if pattern.search(str_value):
                errors.append(
                    ValidationError(
                        rule_id="SEC002",
                        field=field_name,
                        value=value,
                        message=f"Potential XSS attempt detected in {field_name}",
                        severity=ErrorSeverity.HIGH,
                        suggestion="Remove script tags and JavaScript references",
                        context={"pattern_matched": pattern.pattern},
                    )
                )

        # Check for path traversal attempts
        for pattern in self.compiled_patterns["path_traversal"]:
            if pattern.search(str_value):
                errors.append(
                    ValidationError(
                        rule_id="SEC003",
                        field=field_name,
                        value=value,
                        message=f"Potential path traversal attempt detected in {field_name}",
                        severity=ErrorSeverity.HIGH,
                        suggestion="Use relative paths within allowed directories",
                        context={"pattern_matched": pattern.pattern},
                    )
                )

        # Check for excessively long inputs (potential DoS)
        if len(str_value) > 10000:
            errors.append(
                ValidationError(
                    rule_id="SEC004",
                    field=field_name,
                    value=f"{str_value[:50]}...",
                    message=f"Input length exceeds maximum allowed size ({len(str_value)} characters)",
                    severity=ErrorSeverity.MEDIUM,
                    suggestion="Reduce input size to under 10,000 characters",
                    context={"input_length": len(str_value)},
                )
            )

        # Check for null byte injection
        if "\x00" in str_value:
            errors.append(
                ValidationError(
                    rule_id="SEC005",
                    field=field_name,
                    value=value,
                    message="Null byte injection attempt detected",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Remove null bytes from input",
                )
            )

        return errors

    def sanitize_input(self, value: str, aggressive: bool = False) -> str:
        """Sanitize input string to remove security threats."""
        if not isinstance(value, str):
            value = str(value)

        # Remove null bytes
        value = value.replace("\x00", "")

        # Remove or escape dangerous characters
        if aggressive:
            # Aggressive sanitization - remove all potentially dangerous content
            dangerous_chars = [
                "<",
                ">",
                '"',
                "'",
                "&",
                ";",
                "--",
                "/*",
                "*/",
                "(",
                ")",
                "{",
                "}",
            ]
            for char in dangerous_chars:
                value = value.replace(char, "")

            # Remove SQL keywords
            sql_keywords = [
                "SELECT",
                "DELETE",
                "UPDATE",
                "INSERT",
                "DROP",
                "UNION",
                "EXEC",
            ]
            for keyword in sql_keywords:
                value = re.sub(rf"\b{keyword}\b", "", value, flags=re.IGNORECASE)

            # Remove dangerous JavaScript/XSS terms
            js_keywords = [
                "script",
                "alert",
                "eval",
                "document",
                "window",
                "String",
                "fromCharCode",
            ]
            for keyword in js_keywords:
                value = re.sub(rf"{keyword}", "", value, flags=re.IGNORECASE)
        else:
            # Standard HTML/XML escaping
            value = value.replace("&", "&amp;")
            value = value.replace("<", "&lt;")
            value = value.replace(">", "&gt;")
            value = value.replace('"', "&quot;")
            value = value.replace("'", "&#x27;")

        return value.strip()


class SchemaValidator:
    """Validates data against expected schemas and formats."""

    def __init__(self, config: SystemConfig):
        """Initialize schema validator with system configuration."""
        self.config = config
        self.room_configs = {
            room_id: room_config for room_id, room_config in config.rooms.items()
        }

    def validate_sensor_event_schema(
        self, event_data: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate sensor event against expected schema."""
        errors = []

        # Required fields validation
        required_fields = ["room_id", "sensor_id", "sensor_type", "state", "timestamp"]
        for field_name in required_fields:
            if field_name not in event_data:
                errors.append(
                    ValidationError(
                        rule_id="SCH001",
                        field=field_name,
                        value=None,
                        message=f"Required field '{field_name}' is missing",
                        severity=ErrorSeverity.CRITICAL,
                        suggestion=f"Provide a valid {field_name} value",
                    )
                )
            elif event_data[field_name] is None:
                errors.append(
                    ValidationError(
                        rule_id="SCH002",
                        field=field_name,
                        value=None,
                        message=f"Required field '{field_name}' cannot be null",
                        severity=ErrorSeverity.CRITICAL,
                        suggestion=f"Provide a valid {field_name} value",
                    )
                )

        # Field type validation
        if "room_id" in event_data and not isinstance(event_data["room_id"], str):
            errors.append(
                ValidationError(
                    rule_id="SCH003",
                    field="room_id",
                    value=event_data["room_id"],
                    message="room_id must be a string",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Convert room_id to string format",
                )
            )

        if "sensor_id" in event_data and not isinstance(event_data["sensor_id"], str):
            errors.append(
                ValidationError(
                    rule_id="SCH004",
                    field="sensor_id",
                    value=event_data["sensor_id"],
                    message="sensor_id must be a string",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Convert sensor_id to string format",
                )
            )

        # Sensor type validation
        if "sensor_type" in event_data:
            valid_sensor_types = [e.value for e in SensorType]
            if event_data["sensor_type"] not in valid_sensor_types:
                errors.append(
                    ValidationError(
                        rule_id="SCH005",
                        field="sensor_type",
                        value=event_data["sensor_type"],
                        message=f"Invalid sensor_type. Must be one of: {valid_sensor_types}",
                        severity=ErrorSeverity.HIGH,
                        suggestion=f"Use one of the valid sensor types: {', '.join(valid_sensor_types)}",
                    )
                )

        # State validation
        if "state" in event_data:
            valid_states = [e.value for e in SensorState]
            if event_data["state"] not in valid_states:
                errors.append(
                    ValidationError(
                        rule_id="SCH006",
                        field="state",
                        value=event_data["state"],
                        message=f"Invalid state. Must be one of: {valid_states}",
                        severity=ErrorSeverity.HIGH,
                        suggestion=f"Use one of the valid states: {', '.join(valid_states)}",
                    )
                )

        # Timestamp validation
        if "timestamp" in event_data:
            if isinstance(event_data["timestamp"], str):
                try:
                    datetime.fromisoformat(
                        event_data["timestamp"].replace("Z", "+00:00")
                    )
                except ValueError:
                    errors.append(
                        ValidationError(
                            rule_id="SCH007",
                            field="timestamp",
                            value=event_data["timestamp"],
                            message="Invalid timestamp format",
                            severity=ErrorSeverity.HIGH,
                            suggestion="Use ISO format: YYYY-MM-DDTHH:MM:SS+TZ",
                        )
                    )
            elif isinstance(event_data["timestamp"], datetime):
                if event_data["timestamp"].tzinfo is None:
                    errors.append(
                        ValidationError(
                            rule_id="SCH008",
                            field="timestamp",
                            value=event_data["timestamp"],
                            message="Timestamp must include timezone information",
                            severity=ErrorSeverity.MEDIUM,
                            suggestion="Add timezone info to datetime objects",
                        )
                    )

        # Attributes validation
        if "attributes" in event_data and event_data["attributes"] is not None:
            if not isinstance(event_data["attributes"], dict):
                errors.append(
                    ValidationError(
                        rule_id="SCH009",
                        field="attributes",
                        value=type(event_data["attributes"]).__name__,
                        message="attributes must be a dictionary",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Convert attributes to dictionary format",
                    )
                )

        return errors

    def validate_room_configuration(
        self, room_id: str, room_data: Dict[str, Any]
    ) -> List[ValidationError]:
        """Validate room configuration data."""
        errors = []

        # Room ID format validation
        if not re.match(r"^[a-zA-Z0-9_-]+$", room_id):
            errors.append(
                ValidationError(
                    rule_id="SCH010",
                    field="room_id",
                    value=room_id,
                    message="Room ID contains invalid characters",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Use only alphanumeric characters, underscores, and hyphens",
                )
            )

        # Room name validation
        if "name" in room_data:
            if not isinstance(room_data["name"], str):
                errors.append(
                    ValidationError(
                        rule_id="SCH011",
                        field="name",
                        value=room_data["name"],
                        message="Room name must be a string",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Convert room name to string format",
                    )
                )
            elif len(room_data["name"].strip()) == 0:
                errors.append(
                    ValidationError(
                        rule_id="SCH012",
                        field="name",
                        value=room_data["name"],
                        message="Room name cannot be empty",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Provide a descriptive room name",
                    )
                )

        return errors


class IntegrityValidator:
    """Handles data integrity checking and corruption detection."""

    def __init__(self, session: AsyncSession):
        """Initialize integrity validator with database session."""
        self.session = session

    def calculate_event_hash(self, event_data: Dict[str, Any]) -> str:
        """Calculate integrity hash for event data."""
        # Sort keys for consistent hashing
        sorted_data = {
            k: event_data[k]
            for k in sorted(event_data.keys())
            if event_data[k] is not None
        }

        # Create hash string
        hash_string = ""
        for key, value in sorted_data.items():
            if isinstance(value, datetime):
                hash_string += f"{key}:{value.isoformat()}"
            else:
                hash_string += f"{key}:{str(value)}"

        # Generate SHA-256 hash
        return hashlib.sha256(hash_string.encode()).hexdigest()

    async def validate_data_consistency(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Validate data consistency across multiple events."""
        errors = []

        if not events:
            return errors

        # Check for duplicate events
        seen_hashes = set()
        for i, event in enumerate(events):
            event_hash = self.calculate_event_hash(event)
            if event_hash in seen_hashes:
                errors.append(
                    ValidationError(
                        rule_id="INT001",
                        field="event_hash",
                        value=event_hash,
                        message=f"Duplicate event detected at index {i}",
                        severity=ErrorSeverity.MEDIUM,
                        suggestion="Remove duplicate events before processing",
                        context={"event_index": i},
                    )
                )
            seen_hashes.add(event_hash)

        # Check timestamp ordering
        timestamps = []
        for i, event in enumerate(events):
            if "timestamp" in event and event["timestamp"]:
                try:
                    if isinstance(event["timestamp"], str):
                        ts = datetime.fromisoformat(
                            event["timestamp"].replace("Z", "+00:00")
                        )
                    else:
                        ts = event["timestamp"]
                    timestamps.append((i, ts))
                except Exception:
                    continue

        # Sort by timestamp and check for significant ordering issues
        timestamps.sort(key=lambda x: x[1])
        for i in range(len(timestamps) - 1):
            current_idx, current_ts = timestamps[i]
            next_idx, next_ts = timestamps[i + 1]

            # Check for events that are too close together (potential duplicates)
            time_diff = (next_ts - current_ts).total_seconds()
            if time_diff < MIN_EVENT_SEPARATION:
                errors.append(
                    ValidationError(
                        rule_id="INT002",
                        field="timestamp",
                        value=time_diff,
                        message=f"Events too close together ({time_diff}s apart)",
                        severity=ErrorSeverity.LOW,
                        suggestion=f"Ensure events are at least {MIN_EVENT_SEPARATION}s apart",
                        context={"event_indices": [current_idx, next_idx]},
                    )
                )

        return errors

    async def validate_cross_system_consistency(
        self, room_id: str, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Validate consistency across different system components."""
        errors = []

        # Check against existing room states
        try:
            stmt = (
                select(RoomState)
                .where(RoomState.room_id == room_id)
                .order_by(RoomState.timestamp.desc())
                .limit(10)
            )
            result = await self.session.execute(stmt)
            recent_states = result.scalars().all()

            # Validate state transitions make sense
            if recent_states and events:
                last_state = recent_states[0]
                first_event = events[0]

                if "state" in first_event:
                    # Check if the transition is logical
                    if (
                        last_state.occupancy_state == "occupied"
                        and first_event["state"] in PRESENCE_STATES
                        and "timestamp" in first_event
                    ):

                        event_ts = first_event["timestamp"]
                        if isinstance(event_ts, str):
                            event_ts = datetime.fromisoformat(
                                event_ts.replace("Z", "+00:00")
                            )

                        # Check if enough time has passed for a valid state change
                        if (
                            last_state.timestamp
                            and (event_ts - last_state.timestamp).total_seconds() < 60
                        ):
                            errors.append(
                                ValidationError(
                                    rule_id="INT003",
                                    field="state_transition",
                                    value=f"{last_state.occupancy_state} -> {first_event['state']}",
                                    message="Rapid state transition detected - potential noise",
                                    severity=ErrorSeverity.LOW,
                                    suggestion="Validate sensor reliability and consider debouncing",
                                    context={
                                        "last_state_time": last_state.timestamp.isoformat(),
                                        "event_time": event_ts.isoformat(),
                                    },
                                )
                            )

        except Exception as e:
            errors.append(
                ValidationError(
                    rule_id="INT004",
                    field="cross_system_validation",
                    value=str(e),
                    message="Failed to validate cross-system consistency",
                    severity=ErrorSeverity.MEDIUM,
                    suggestion="Check database connectivity and data integrity",
                )
            )

        return errors


class PerformanceValidator:
    """Optimized validation for high-volume data processing."""

    def __init__(self, batch_size: int = 1000):
        """Initialize performance validator."""
        self.batch_size = batch_size
        self.validation_stats = defaultdict(int)

    async def bulk_validate_events(
        self,
        events: List[Dict[str, Any]],
        security_validator: SecurityValidator,
        schema_validator: SchemaValidator,
    ) -> List[ValidationResult]:
        """Perform bulk validation with performance optimizations."""
        results = []

        # Process in batches to manage memory
        for i in range(0, len(events), self.batch_size):
            batch = events[i : i + self.batch_size]
            batch_results = await self._validate_batch(
                batch, security_validator, schema_validator
            )
            results.extend(batch_results)

            # Update stats
            self.validation_stats["batches_processed"] += 1
            self.validation_stats["events_processed"] += len(batch)

        return results

    async def _validate_batch(
        self,
        batch: List[Dict[str, Any]],
        security_validator: SecurityValidator,
        schema_validator: SchemaValidator,
    ) -> List[ValidationResult]:
        """Validate a batch of events with parallel processing."""
        # Use asyncio for parallel validation where possible
        validation_tasks = []

        for event in batch:
            task = self._validate_single_event(
                event, security_validator, schema_validator
            )
            validation_tasks.append(task)

        return await asyncio.gather(*validation_tasks, return_exceptions=True)

    async def _validate_single_event(
        self,
        event: Dict[str, Any],
        security_validator: SecurityValidator,
        schema_validator: SchemaValidator,
    ) -> ValidationResult:
        """Validate a single event with comprehensive checks."""
        start_time = asyncio.get_event_loop().time()
        errors = []
        warnings = []
        security_flags = []

        try:
            # Schema validation
            schema_errors = schema_validator.validate_sensor_event_schema(event)
            errors.extend(schema_errors)

            # Security validation for each field
            for field_name, value in event.items():
                if value is not None:
                    security_errors = security_validator.validate_input_security(
                        value, field_name
                    )
                    for error in security_errors:
                        if error.severity == ErrorSeverity.CRITICAL:
                            errors.append(error)
                            security_flags.append(
                                f"Critical security issue in {field_name}"
                            )
                        elif error.severity == ErrorSeverity.HIGH:
                            errors.append(error)
                            security_flags.append(f"High security risk in {field_name}")
                        else:
                            warnings.append(error)
                            security_flags.append(f"Security concern in {field_name}")

            # Calculate integrity hash
            integrity_hash = hashlib.sha256(str(event).encode()).hexdigest()

            # Calculate confidence score
            confidence_score = 1.0
            if errors:
                confidence_score -= len(errors) * 0.2
            if warnings:
                confidence_score -= len(warnings) * 0.1
            confidence_score = max(0.0, confidence_score)

        except Exception as e:
            errors.append(
                ValidationError(
                    rule_id="VAL001",
                    field="validation_process",
                    value=str(e),
                    message=f"Validation process error: {str(e)}",
                    severity=ErrorSeverity.HIGH,
                    suggestion="Check event data format and validation logic",
                )
            )
            confidence_score = 0.0
            integrity_hash = None

        processing_time_ms = (asyncio.get_event_loop().time() - start_time) * 1000

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
            security_flags=security_flags,
            integrity_hash=integrity_hash,
        )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get validation performance statistics."""
        return dict(self.validation_stats)


class ComprehensiveEventValidator:
    """Main validator that orchestrates all validation types."""

    def __init__(self, session: AsyncSession, batch_size: int = 1000):
        """Initialize comprehensive event validator."""
        self.session = session
        self.config = get_config()

        # Initialize component validators
        self.security_validator = SecurityValidator()
        self.schema_validator = SchemaValidator(self.config)
        self.integrity_validator = IntegrityValidator(session)
        self.performance_validator = PerformanceValidator(batch_size)

        self.validation_rules = self._initialize_validation_rules()

    def _initialize_validation_rules(self) -> List[ValidationRule]:
        """Initialize validation rules configuration."""
        return [
            ValidationRule(
                rule_id="SEC001",
                name="SQL Injection Detection",
                description="Detect potential SQL injection attempts",
                severity=ErrorSeverity.CRITICAL,
            ),
            ValidationRule(
                rule_id="SEC002",
                name="XSS Detection",
                description="Detect potential cross-site scripting attempts",
                severity=ErrorSeverity.HIGH,
            ),
            ValidationRule(
                rule_id="SCH001",
                name="Required Fields",
                description="Validate all required fields are present",
                severity=ErrorSeverity.CRITICAL,
            ),
            ValidationRule(
                rule_id="INT001",
                name="Duplicate Detection",
                description="Detect duplicate events",
                severity=ErrorSeverity.MEDIUM,
            ),
        ]

    async def validate_event(self, event_data: Dict[str, Any]) -> ValidationResult:
        """Validate a single event with all validation types."""
        return await self.performance_validator._validate_single_event(
            event_data, self.security_validator, self.schema_validator
        )

    async def validate_events_bulk(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """Validate multiple events with performance optimizations."""
        # Add integrity validation for the bulk set
        integrity_errors = await self.integrity_validator.validate_data_consistency(
            events
        )

        # Perform bulk validation
        results = await self.performance_validator.bulk_validate_events(
            events, self.security_validator, self.schema_validator
        )

        # Add integrity errors to first result if any
        if integrity_errors and results:
            results[0].errors.extend(integrity_errors)
            if results[0].errors:
                results[0].is_valid = False

        return results

    async def validate_room_events(
        self, room_id: str, events: List[Dict[str, Any]]
    ) -> List[ValidationResult]:
        """Validate events for a specific room with cross-system checks."""
        # Standard validation
        results = await self.validate_events_bulk(events)

        # Add cross-system consistency validation
        try:
            cross_system_errors = (
                await self.integrity_validator.validate_cross_system_consistency(
                    room_id, events
                )
            )
            if cross_system_errors and results:
                results[0].errors.extend(cross_system_errors)
                if results[0].errors:
                    results[0].is_valid = False
        except Exception as e:
            logger.error(f"Cross-system validation failed: {e}")

        return results

    def sanitize_event_data(
        self, event_data: Dict[str, Any], aggressive: bool = False
    ) -> Dict[str, Any]:
        """Sanitize event data to remove security threats."""
        sanitized = {}
        for key, value in event_data.items():
            if isinstance(value, str):
                sanitized[key] = self.security_validator.sanitize_input(
                    value, aggressive
                )
            elif isinstance(value, dict):
                sanitized[key] = {
                    k: (
                        self.security_validator.sanitize_input(str(v), aggressive)
                        if isinstance(v, str)
                        else v
                    )
                    for k, v in value.items()
                }
            else:
                sanitized[key] = value
        return sanitized

    def get_validation_summary(self) -> Dict[str, Any]:
        """Get comprehensive validation statistics and summary."""
        perf_stats = self.performance_validator.get_performance_stats()

        return {
            "validation_rules": len(self.validation_rules),
            "active_rules": len([r for r in self.validation_rules if r.enabled]),
            "performance_stats": perf_stats,
            "security_patterns": {
                "sql_injection": len(self.security_validator.SQL_INJECTION_PATTERNS),
                "xss": len(self.security_validator.XSS_PATTERNS),
                "path_traversal": len(self.security_validator.PATH_TRAVERSAL_PATTERNS),
            },
        }
