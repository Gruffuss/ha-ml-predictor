"""
Self-Adaptation System for Sprint 4 - Occupancy Prediction System.

This module provides real-time prediction validation, accuracy tracking,
concept drift detection, and adaptive retraining capabilities.
"""

from .drift_detector import ConceptDriftDetector
from .drift_detector import DriftDetectionError
from .drift_detector import DriftMetrics
from .drift_detector import DriftSeverity
from .drift_detector import DriftType
from .drift_detector import FeatureDriftDetector
from .drift_detector import FeatureDriftResult
from .drift_detector import StatisticalTest
from .tracker import AccuracyAlert
from .tracker import AccuracyTracker
from .tracker import AccuracyTrackingError
from .tracker import AlertSeverity
from .tracker import RealTimeMetrics
from .tracker import TrendDirection
from .validator import AccuracyLevel
from .validator import AccuracyMetrics
from .validator import PredictionValidator
from .validator import ValidationError
from .validator import ValidationRecord
from .validator import ValidationStatus

__all__ = [
    # Validation components
    "ValidationStatus",
    "AccuracyLevel",
    "ValidationRecord",
    "AccuracyMetrics",
    "PredictionValidator",
    "ValidationError",
    # Tracking components
    "AlertSeverity",
    "TrendDirection",
    "RealTimeMetrics",
    "AccuracyAlert",
    "AccuracyTracker",
    "AccuracyTrackingError",
    # Drift detection components
    "DriftType",
    "DriftSeverity",
    "StatisticalTest",
    "DriftMetrics",
    "FeatureDriftResult",
    "ConceptDriftDetector",
    "FeatureDriftDetector",
    "DriftDetectionError",
]
