"""
Self-Adaptation System for Sprint 4 - Occupancy Prediction System.

This module provides real-time prediction validation, accuracy tracking,
concept drift detection, and adaptive retraining capabilities.
"""

from .drift_detector import (
    ConceptDriftDetector,
    DriftDetectionError,
    DriftMetrics,
    DriftSeverity,
    DriftType,
    FeatureDriftDetector,
    FeatureDriftResult,
    StatisticalTest,
)
from .tracker import (
    AccuracyAlert,
    AccuracyTracker,
    AccuracyTrackingError,
    AlertSeverity,
    RealTimeMetrics,
    TrendDirection,
)
from .validator import (
    AccuracyLevel,
    AccuracyMetrics,
    PredictionValidator,
    ValidationError,
    ValidationRecord,
    ValidationStatus,
)
from .retrainer import (
    AdaptiveRetrainer,
    RetrainingRequest,
    RetrainingStatus,
)
from .optimizer import (
    ModelOptimizer,
    OptimizationConfig,
    OptimizationResult,
)
from .tracking_manager import (
    TrackingManager,
    TrackingConfig,
)

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
    # Retraining components
    "AdaptiveRetrainer",
    "RetrainingRequest",
    "RetrainingStatus",
    # Optimization components
    "ModelOptimizer",
    "OptimizationConfig",
    "OptimizationResult",
    # Tracking management components
    "TrackingManager",
    "TrackingConfig",
]
