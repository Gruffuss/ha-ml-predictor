"""
Data Validation Package for Home Assistant ML Predictor.

This package provides comprehensive data validation capabilities including:
- Event validation with security checks
- Schema validation and format verification
- Pattern detection and anomaly identification
- Data integrity and corruption detection
- Performance-optimized validation for high-volume processing
"""

from .event_validator import (
    ComprehensiveEventValidator,
    IntegrityValidator,
    PerformanceValidator,
    SchemaValidator,
    SecurityValidator,
    ValidationError,
    ValidationResult,
    ValidationRule,
)
from .pattern_detector import (
    CorruptionDetector,
    DataQualityMetrics,
    PatternAnomaly,
    RealTimeQualityMonitor,
    StatisticalPatternAnalyzer,
)
from .schema_validator import (
    APISchemaValidator,
    DatabaseSchemaValidator,
    JSONSchemaValidator,
    SchemaDefinition,
    SchemaValidationContext,
)

__all__ = [
    # Event Validation
    "ComprehensiveEventValidator",
    "SecurityValidator",
    "SchemaValidator",
    "IntegrityValidator",
    "PerformanceValidator",
    "ValidationError",
    "ValidationResult",
    "ValidationRule",
    # Pattern Detection
    "StatisticalPatternAnalyzer",
    "CorruptionDetector",
    "RealTimeQualityMonitor",
    "PatternAnomaly",
    "DataQualityMetrics",
    # Schema Validation
    "JSONSchemaValidator",
    "DatabaseSchemaValidator",
    "APISchemaValidator",
    "SchemaDefinition",
    "SchemaValidationContext",
]
