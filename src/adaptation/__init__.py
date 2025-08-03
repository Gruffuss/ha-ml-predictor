"""
Self-Adaptation System for Sprint 4 - Occupancy Prediction System.

This module provides real-time prediction validation, accuracy tracking,
concept drift detection, and adaptive retraining capabilities.
"""

from .validator import (
    ValidationStatus,
    AccuracyLevel,
    ValidationRecord,
    AccuracyMetrics,
    PredictionValidator,
    ValidationError
)

from .tracker import (
    AlertSeverity,
    TrendDirection,
    RealTimeMetrics,
    AccuracyAlert,
    AccuracyTracker,
    AccuracyTrackingError
)

from .drift_detector import (
    DriftType,
    DriftSeverity,
    StatisticalTest,
    DriftMetrics,
    FeatureDriftResult,
    ConceptDriftDetector,
    FeatureDriftDetector,
    DriftDetectionError
)

__all__ = [
    # Validation components
    'ValidationStatus',
    'AccuracyLevel', 
    'ValidationRecord',
    'AccuracyMetrics',
    'PredictionValidator',
    'ValidationError',
    
    # Tracking components
    'AlertSeverity',
    'TrendDirection',
    'RealTimeMetrics',
    'AccuracyAlert',
    'AccuracyTracker',
    'AccuracyTrackingError',
    
    # Drift detection components
    'DriftType',
    'DriftSeverity',
    'StatisticalTest',
    'DriftMetrics',
    'FeatureDriftResult',
    'ConceptDriftDetector',
    'FeatureDriftDetector',
    'DriftDetectionError'
]