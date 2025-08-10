"""
Concept Drift Detection System for Sprint 4 - Self-Adaptation System.

This module provides comprehensive statistical drift detection
capabilities to identify when occupancy patterns have fundamentally
changed and models need retraining. Implements robust statistical tests
for both feature drift and concept drift.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sqlalchemy import and_, select

from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..data.storage.database import get_db_session
from ..data.storage.models import Prediction, SensorEvent
from .validator import PredictionValidator

logger = logging.getLogger(__name__)


class DriftType(Enum):
    """Types of drift that can be detected."""

    FEATURE_DRIFT = "feature_drift"
    CONCEPT_DRIFT = "concept_drift"
    PREDICTION_DRIFT = "prediction_drift"
    PATTERN_DRIFT = "pattern_drift"


class DriftSeverity(Enum):
    """Severity levels for detected drift."""

    MINOR = "minor"  # Statistical but possibly noise
    MODERATE = "moderate"  # Clear drift but manageable
    MAJOR = "major"  # Significant drift requiring attention
    CRITICAL = "critical"  # Severe drift requiring immediate action


class StatisticalTest(Enum):
    """Available statistical tests for drift detection."""

    KOLMOGOROV_SMIRNOV = "kolmogorov_smirnov"
    MANN_WHITNEY_U = "mann_whitney_u"
    CHI_SQUARE = "chi_square"
    PAGE_HINKLEY = "page_hinkley"
    PSI = "population_stability_index"


@dataclass
class DriftMetrics:
    """
    Comprehensive drift detection metrics with statistical analysis.

    Contains results from various statistical tests and drift severity analysis
    for feature distributions, concept drift, and pattern changes.
    """

    # Detection metadata
    room_id: str
    detection_time: datetime
    baseline_period: Tuple[datetime, datetime]
    current_period: Tuple[datetime, datetime]

    # Statistical test results
    ks_statistic: float = 0.0
    ks_p_value: float = 1.0
    mw_statistic: float = 0.0
    mw_p_value: float = 1.0
    chi2_statistic: float = 0.0
    chi2_p_value: float = 1.0
    psi_score: float = 0.0

    # Page-Hinkley test for concept drift
    ph_statistic: float = 0.0
    ph_threshold: float = 50.0
    ph_drift_detected: bool = False

    # Feature-specific drift analysis
    drifting_features: List[str] = field(default_factory=list)
    feature_drift_scores: Dict[str, float] = field(default_factory=dict)
    feature_importance_changes: Dict[str, float] = field(default_factory=dict)

    # Prediction performance drift
    accuracy_degradation: float = 0.0
    error_distribution_change: float = 0.0
    confidence_calibration_drift: float = 0.0

    # Pattern analysis
    occupancy_pattern_changes: Dict[str, float] = field(default_factory=dict)
    temporal_pattern_drift: float = 0.0
    frequency_pattern_drift: float = 0.0

    # Overall assessment
    overall_drift_score: float = 0.0
    drift_severity: DriftSeverity = DriftSeverity.MINOR
    drift_types: List[DriftType] = field(default_factory=list)

    # Confidence and reliability
    statistical_confidence: float = 0.0
    sample_size_baseline: int = 0
    sample_size_current: int = 0

    # Recommendations
    retraining_recommended: bool = False
    immediate_attention_required: bool = False
    affected_models: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate overall metrics after initialization."""
        self._calculate_overall_drift_score()
        self._determine_drift_severity()
        self._generate_recommendations()

    def _calculate_overall_drift_score(self) -> None:
        """Calculate weighted overall drift score from all components."""
        # Statistical significance weight
        stat_weight = 0.3
        stat_score = (
            (1 - self.ks_p_value) * 0.4
            + (1 - self.mw_p_value) * 0.3
            + min(self.psi_score / 0.25, 1.0) * 0.3  # PSI > 0.25 is concerning
        )

        # Performance degradation weight
        perf_weight = 0.4
        perf_score = min(self.accuracy_degradation / 30.0, 1.0)  # 30 min threshold

        # Pattern change weight
        pattern_weight = 0.3
        pattern_score = (
            self.temporal_pattern_drift * 0.5 + self.frequency_pattern_drift * 0.5
        )

        self.overall_drift_score = (
            stat_score * stat_weight
            + perf_score * perf_weight
            + pattern_score * pattern_weight
        )

    def _determine_drift_severity(self) -> None:
        """Determine drift severity based on overall score and specific indicators."""
        if self.ph_drift_detected or self.overall_drift_score > 0.8:
            self.drift_severity = DriftSeverity.CRITICAL
        elif self.overall_drift_score > 0.6 or self.accuracy_degradation > 20:
            self.drift_severity = DriftSeverity.MAJOR
        elif self.overall_drift_score > 0.4 or self.accuracy_degradation > 10:
            self.drift_severity = DriftSeverity.MODERATE
        else:
            self.drift_severity = DriftSeverity.MINOR

    def _generate_recommendations(self) -> None:
        """Generate recommendations based on drift analysis."""
        # Immediate attention for critical drift or severe performance degradation
        self.immediate_attention_required = (
            self.drift_severity == DriftSeverity.CRITICAL
            or self.accuracy_degradation > 25
            or self.ph_drift_detected
        )

        # Retraining recommended for moderate+ drift or significant degradation
        self.retraining_recommended = (
            self.drift_severity
            in [
                DriftSeverity.MODERATE,
                DriftSeverity.MAJOR,
                DriftSeverity.CRITICAL,
            ]
            or self.accuracy_degradation > 15
            or self.overall_drift_score > 0.5
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert drift metrics to dictionary for serialization."""
        return {
            "room_id": self.room_id,
            "detection_time": self.detection_time.isoformat(),
            "baseline_period": [
                self.baseline_period[0].isoformat(),
                self.baseline_period[1].isoformat(),
            ],
            "current_period": [
                self.current_period[0].isoformat(),
                self.current_period[1].isoformat(),
            ],
            "statistical_tests": {
                "kolmogorov_smirnov": {
                    "statistic": self.ks_statistic,
                    "p_value": self.ks_p_value,
                },
                "mann_whitney_u": {
                    "statistic": self.mw_statistic,
                    "p_value": self.mw_p_value,
                },
                "chi_square": {
                    "statistic": self.chi2_statistic,
                    "p_value": self.chi2_p_value,
                },
                "population_stability_index": self.psi_score,
                "page_hinkley": {
                    "statistic": self.ph_statistic,
                    "threshold": self.ph_threshold,
                    "drift_detected": self.ph_drift_detected,
                },
            },
            "feature_analysis": {
                "drifting_features": self.drifting_features,
                "feature_drift_scores": self.feature_drift_scores,
                "importance_changes": self.feature_importance_changes,
            },
            "prediction_analysis": {
                "accuracy_degradation_minutes": self.accuracy_degradation,
                "error_distribution_change": self.error_distribution_change,
                "confidence_calibration_drift": self.confidence_calibration_drift,
            },
            "pattern_analysis": {
                "occupancy_pattern_changes": self.occupancy_pattern_changes,
                "temporal_pattern_drift": self.temporal_pattern_drift,
                "frequency_pattern_drift": self.frequency_pattern_drift,
            },
            "assessment": {
                "overall_drift_score": self.overall_drift_score,
                "drift_severity": self.drift_severity.value,
                "drift_types": [dt.value for dt in self.drift_types],
                "statistical_confidence": self.statistical_confidence,
                "sample_sizes": {
                    "baseline": self.sample_size_baseline,
                    "current": self.sample_size_current,
                },
            },
            "recommendations": {
                "retraining_recommended": self.retraining_recommended,
                "immediate_attention_required": self.immediate_attention_required,
                "affected_models": self.affected_models,
            },
        }


@dataclass
class FeatureDriftResult:
    """Results from feature distribution drift analysis."""

    feature_name: str
    drift_detected: bool
    drift_score: float
    statistical_test: StatisticalTest
    test_statistic: float
    p_value: float
    baseline_stats: Dict[str, float]
    current_stats: Dict[str, float]

    def is_significant(self, alpha: float = 0.05) -> bool:
        """Check if drift is statistically significant."""
        return self.p_value < alpha


class ConceptDriftDetector:
    """
    Statistical concept drift detection for occupancy prediction models.

    Implements multiple statistical tests to detect when the relationship between
    features and targets has changed, indicating concept drift requiring retraining.
    """

    def __init__(
        self,
        baseline_days: int = 30,
        current_days: int = 7,
        min_samples: int = 100,
        alpha: float = 0.05,
        ph_threshold: float = 50.0,
        psi_threshold: float = 0.25,
    ):
        """
        Initialize concept drift detector.

        Args:
            baseline_days: Days of historical data for baseline comparison
            current_days: Days of recent data for current comparison
            min_samples: Minimum samples required for reliable detection
            alpha: Statistical significance threshold
            ph_threshold: Page-Hinkley test threshold for drift detection
            psi_threshold: Population Stability Index threshold
        """
        self.baseline_days = baseline_days
        self.current_days = current_days
        self.min_samples = min_samples
        self.alpha = alpha
        self.ph_threshold = ph_threshold
        self.psi_threshold = psi_threshold

        # Page-Hinkley test state for continuous monitoring
        self._ph_sum = defaultdict(float)
        self._ph_min = defaultdict(float)
        self._ph_error_history = defaultdict(lambda: deque(maxlen=1000))

        # Historical statistics for comparison
        self._feature_baselines = {}
        self._pattern_baselines = {}

        logger.info(
            f"Initialized ConceptDriftDetector with {baseline_days}d baseline, "
            f"{current_days}d current window, alpha={alpha}"
        )

    async def detect_drift(
        self,
        room_id: str,
        prediction_validator: PredictionValidator,
        feature_engineering_engine=None,
    ) -> DriftMetrics:
        """
        Comprehensive drift detection for a specific room.

        Args:
            room_id: Room to analyze for drift
            prediction_validator: Validator with prediction history
            feature_engineering_engine: Engine for feature extraction

        Returns:
            DriftMetrics with comprehensive drift analysis
        """
        logger.info(f"Starting drift detection for room {room_id}")

        # Define time periods
        current_end = datetime.now()
        current_start = current_end - timedelta(days=self.current_days)
        baseline_end = current_start
        baseline_start = baseline_end - timedelta(days=self.baseline_days)

        # Initialize drift metrics
        drift_metrics = DriftMetrics(
            room_id=room_id,
            detection_time=current_end,
            baseline_period=(baseline_start, baseline_end),
            current_period=(current_start, current_end),
        )

        try:
            # Get prediction accuracy metrics for performance drift
            await self._analyze_prediction_drift(
                drift_metrics, prediction_validator, room_id
            )

            # Get feature data for feature drift analysis
            if feature_engineering_engine:
                await self._analyze_feature_drift(
                    drift_metrics, feature_engineering_engine, room_id
                )

            # Analyze occupancy pattern changes
            await self._analyze_pattern_drift(drift_metrics, room_id)

            # Run Page-Hinkley test for concept drift
            await self._run_page_hinkley_test(drift_metrics, room_id)

            # Calculate statistical confidence
            self._calculate_statistical_confidence(drift_metrics)

            logger.info(
                f"Drift detection completed for {room_id}: "
                f"severity={drift_metrics.drift_severity.value}, "
                f"score={drift_metrics.overall_drift_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Error in drift detection for {room_id}: {e}")
            raise OccupancyPredictionError(
                f"Drift detection failed for room {room_id}",
                error_context={"room_id": room_id, "error": str(e)},
                severity=ErrorSeverity.MEDIUM,
            )

        return drift_metrics

    async def _analyze_prediction_drift(
        self,
        drift_metrics: DriftMetrics,
        validator: PredictionValidator,
        room_id: str,
    ) -> None:
        """Analyze drift in prediction performance."""
        try:
            # Get baseline accuracy metrics
            baseline_metrics = await validator.get_accuracy_metrics(
                room_id=room_id,
                start_time=drift_metrics.baseline_period[0],
                end_time=drift_metrics.baseline_period[1],
            )

            # Get current accuracy metrics
            current_metrics = await validator.get_accuracy_metrics(
                room_id=room_id,
                start_time=drift_metrics.current_period[0],
                end_time=drift_metrics.current_period[1],
            )

            if baseline_metrics and current_metrics:
                # Calculate accuracy degradation
                baseline_error = baseline_metrics.mean_absolute_error_minutes
                current_error = current_metrics.mean_absolute_error_minutes
                drift_metrics.accuracy_degradation = current_error - baseline_error

                # Analyze error distribution changes using KS test
                baseline_errors = [
                    r.error_minutes for r in baseline_metrics.recent_records
                ]
                current_errors = [
                    r.error_minutes for r in current_metrics.recent_records
                ]

                if len(baseline_errors) >= 10 and len(current_errors) >= 10:
                    ks_stat, ks_p = stats.ks_2samp(baseline_errors, current_errors)
                    drift_metrics.error_distribution_change = ks_stat

                # Analyze confidence calibration drift
                baseline_conf = baseline_metrics.confidence_vs_accuracy_correlation
                current_conf = current_metrics.confidence_vs_accuracy_correlation
                drift_metrics.confidence_calibration_drift = abs(
                    current_conf - baseline_conf
                )

                # Mark prediction drift if significant
                if drift_metrics.accuracy_degradation > 10:
                    drift_metrics.drift_types.append(DriftType.PREDICTION_DRIFT)

        except Exception as e:
            logger.warning(f"Error analyzing prediction drift for {room_id}: {e}")

    async def _analyze_feature_drift(
        self, drift_metrics: DriftMetrics, feature_engine, room_id: str
    ) -> None:
        """Analyze drift in feature distributions."""
        try:
            # Get feature data for both periods
            baseline_features = await self._get_feature_data(
                feature_engine,
                room_id,
                drift_metrics.baseline_period[0],
                drift_metrics.baseline_period[1],
            )

            current_features = await self._get_feature_data(
                feature_engine,
                room_id,
                drift_metrics.current_period[0],
                drift_metrics.current_period[1],
            )

            if baseline_features is not None and current_features is not None:
                # Find common features
                common_features = set(baseline_features.columns) & set(
                    current_features.columns
                )

                # Test each feature for drift
                drifting_features = []
                feature_drift_scores = {}

                for feature in common_features:
                    if feature in ["room_id", "timestamp"]:
                        continue

                    drift_result = await self._test_feature_drift(
                        baseline_features[feature],
                        current_features[feature],
                        feature,
                    )

                    feature_drift_scores[feature] = drift_result.drift_score

                    if drift_result.is_significant(self.alpha):
                        drifting_features.append(feature)

                drift_metrics.drifting_features = drifting_features
                drift_metrics.feature_drift_scores = feature_drift_scores

                # Calculate PSI for overall feature drift
                drift_metrics.psi_score = await self._calculate_psi(
                    baseline_features, current_features, common_features
                )

                # Mark feature drift if significant
                if (
                    len(drifting_features) > 0
                    or drift_metrics.psi_score > self.psi_threshold
                ):
                    drift_metrics.drift_types.append(DriftType.FEATURE_DRIFT)

        except Exception as e:
            logger.warning(f"Error analyzing feature drift for {room_id}: {e}")

    async def _test_feature_drift(
        self,
        baseline_data: pd.Series,
        current_data: pd.Series,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Test individual feature for distribution drift."""
        try:
            # Remove NaN values
            baseline_clean = baseline_data.dropna()
            current_clean = current_data.dropna()

            if len(baseline_clean) < 10 or len(current_clean) < 10:
                # Insufficient data
                return FeatureDriftResult(
                    feature_name=feature_name,
                    drift_detected=False,
                    drift_score=0.0,
                    statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
                    test_statistic=0.0,
                    p_value=1.0,
                    baseline_stats={},
                    current_stats={},
                )

            # Choose test based on data type
            if baseline_clean.dtype in ["object", "category"]:
                # Categorical data - use Chi-square test
                return await self._test_categorical_drift(
                    baseline_clean, current_clean, feature_name
                )
            else:
                # Numerical data - use KS test
                return await self._test_numerical_drift(
                    baseline_clean, current_clean, feature_name
                )

        except Exception as e:
            logger.warning(f"Error testing feature drift for {feature_name}: {e}")
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
                test_statistic=0.0,
                p_value=1.0,
                baseline_stats={},
                current_stats={},
            )

    async def _test_numerical_drift(
        self, baseline: pd.Series, current: pd.Series, feature_name: str
    ) -> FeatureDriftResult:
        """Test numerical feature for distribution drift using KS test."""
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(baseline, current)

        # Calculate basic statistics
        baseline_stats = {
            "mean": float(baseline.mean()),
            "std": float(baseline.std()),
            "median": float(baseline.median()),
            "min": float(baseline.min()),
            "max": float(baseline.max()),
        }

        current_stats = {
            "mean": float(current.mean()),
            "std": float(current.std()),
            "median": float(current.median()),
            "min": float(current.min()),
            "max": float(current.max()),
        }

        # Calculate drift score based on statistic magnitude
        drift_score = min(ks_stat * 2, 1.0)  # Scale to 0-1

        return FeatureDriftResult(
            feature_name=feature_name,
            drift_detected=ks_p < self.alpha,
            drift_score=drift_score,
            statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
            test_statistic=ks_stat,
            p_value=ks_p,
            baseline_stats=baseline_stats,
            current_stats=current_stats,
        )

    async def _test_categorical_drift(
        self, baseline: pd.Series, current: pd.Series, feature_name: str
    ) -> FeatureDriftResult:
        """Test categorical feature for distribution drift using Chi-square test."""
        try:
            # Get value counts
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()

            # Align categories
            all_categories = set(baseline_counts.index) | set(current_counts.index)
            baseline_aligned = [baseline_counts.get(cat, 0) for cat in all_categories]
            current_aligned = [current_counts.get(cat, 0) for cat in all_categories]

            # Chi-square test
            chi2_stat, chi2_p, _, _ = stats.chi2_contingency(
                [baseline_aligned, current_aligned]
            )

            # Calculate basic statistics
            baseline_stats = {
                "mode": (baseline.mode().iloc[0] if len(baseline.mode()) > 0 else None),
                "unique_count": baseline.nunique(),
                "most_frequent_pct": (
                    baseline_counts.iloc[0] / len(baseline)
                    if len(baseline_counts) > 0
                    else 0
                ),
            }

            current_stats = {
                "mode": (current.mode().iloc[0] if len(current.mode()) > 0 else None),
                "unique_count": current.nunique(),
                "most_frequent_pct": (
                    current_counts.iloc[0] / len(current)
                    if len(current_counts) > 0
                    else 0
                ),
            }

            # Calculate drift score
            drift_score = min(chi2_stat / max(len(all_categories) * 10, 1), 1.0)

            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=chi2_p < self.alpha,
                drift_score=drift_score,
                statistical_test=StatisticalTest.CHI_SQUARE,
                test_statistic=chi2_stat,
                p_value=chi2_p,
                baseline_stats=baseline_stats,
                current_stats=current_stats,
            )

        except Exception as e:
            logger.warning(f"Error in categorical drift test for {feature_name}: {e}")
            # Fallback to simple comparison
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                statistical_test=StatisticalTest.CHI_SQUARE,
                test_statistic=0.0,
                p_value=1.0,
                baseline_stats={},
                current_stats={},
            )

    async def _calculate_psi(
        self,
        baseline_df: pd.DataFrame,
        current_df: pd.DataFrame,
        features: Set[str],
    ) -> float:
        """Calculate Population Stability Index across all features."""
        try:
            psi_scores = []

            for feature in features:
                if feature in ["room_id", "timestamp"]:
                    continue

                baseline_data = baseline_df[feature].dropna()
                current_data = current_df[feature].dropna()

                if len(baseline_data) < 10 or len(current_data) < 10:
                    continue

                # Calculate PSI for this feature
                if baseline_data.dtype in ["object", "category"]:
                    psi = self._calculate_categorical_psi(baseline_data, current_data)
                else:
                    psi = self._calculate_numerical_psi(baseline_data, current_data)

                psi_scores.append(psi)

            return np.mean(psi_scores) if psi_scores else 0.0

        except Exception as e:
            logger.warning(f"Error calculating PSI: {e}")
            return 0.0

    def _calculate_numerical_psi(
        self, baseline: pd.Series, current: pd.Series, bins: int = 10
    ) -> float:
        """Calculate PSI for numerical feature."""
        try:
            # Create bins based on baseline quantiles
            quantiles = np.linspace(0, 1, bins + 1)
            bin_edges = baseline.quantile(quantiles).values
            bin_edges[0] = -np.inf
            bin_edges[-1] = np.inf

            # Calculate distributions
            baseline_dist = (
                pd.cut(baseline, bins=bin_edges)
                .value_counts(normalize=True)
                .sort_index()
            )
            current_dist = (
                pd.cut(current, bins=bin_edges)
                .value_counts(normalize=True)
                .sort_index()
            )

            # Align distributions
            baseline_aligned = baseline_dist.reindex(
                baseline_dist.index, fill_value=0.001
            )
            current_aligned = current_dist.reindex(
                baseline_dist.index, fill_value=0.001
            )

            # Calculate PSI
            psi = np.sum(
                (current_aligned - baseline_aligned)
                * np.log(current_aligned / baseline_aligned)
            )

            return float(psi)

        except Exception as e:
            logger.warning(f"Error calculating numerical PSI: {e}")
            return 0.0

    def _calculate_categorical_psi(
        self, baseline: pd.Series, current: pd.Series
    ) -> float:
        """Calculate PSI for categorical feature."""
        try:
            # Get distributions
            baseline_dist = baseline.value_counts(normalize=True)
            current_dist = current.value_counts(normalize=True)

            # Align categories
            all_categories = set(baseline_dist.index) | set(current_dist.index)
            baseline_aligned = pd.Series(
                [baseline_dist.get(cat, 0.001) for cat in all_categories],
                index=all_categories,
            )
            current_aligned = pd.Series(
                [current_dist.get(cat, 0.001) for cat in all_categories],
                index=all_categories,
            )

            # Calculate PSI
            psi = np.sum(
                (current_aligned - baseline_aligned)
                * np.log(current_aligned / baseline_aligned)
            )

            return float(psi)

        except Exception as e:
            logger.warning(f"Error calculating categorical PSI: {e}")
            return 0.0

    async def _analyze_pattern_drift(
        self, drift_metrics: DriftMetrics, room_id: str
    ) -> None:
        """Analyze drift in occupancy patterns."""
        try:
            # Get occupancy data for both periods
            baseline_patterns = await self._get_occupancy_patterns(
                room_id,
                drift_metrics.baseline_period[0],
                drift_metrics.baseline_period[1],
            )

            current_patterns = await self._get_occupancy_patterns(
                room_id,
                drift_metrics.current_period[0],
                drift_metrics.current_period[1],
            )

            if baseline_patterns and current_patterns:
                # Analyze temporal patterns (hourly occupancy distribution)
                temporal_drift = self._compare_temporal_patterns(
                    baseline_patterns, current_patterns
                )
                drift_metrics.temporal_pattern_drift = temporal_drift

                # Analyze frequency patterns (daily occupancy frequency)
                frequency_drift = self._compare_frequency_patterns(
                    baseline_patterns, current_patterns
                )
                drift_metrics.frequency_pattern_drift = frequency_drift

                # Store specific pattern changes
                drift_metrics.occupancy_pattern_changes = {
                    "temporal_shift": temporal_drift,
                    "frequency_change": frequency_drift,
                }

                # Mark pattern drift if significant
                if temporal_drift > 0.3 or frequency_drift > 0.3:
                    drift_metrics.drift_types.append(DriftType.PATTERN_DRIFT)

        except Exception as e:
            logger.warning(f"Error analyzing pattern drift for {room_id}: {e}")

    async def _run_page_hinkley_test(
        self, drift_metrics: DriftMetrics, room_id: str
    ) -> None:
        """Run Page-Hinkley test for concept drift detection."""
        try:
            # Get recent prediction errors for PH test
            errors = await self._get_recent_prediction_errors(
                room_id, days=self.current_days
            )

            if len(errors) > 10:
                # Update Page-Hinkley statistics
                error_mean = np.mean(errors)

                for error in errors:
                    # Update cumulative sum
                    self._ph_sum[room_id] += error - error_mean - 1.0  # delta = 1.0

                    # Update minimum
                    self._ph_min[room_id] = min(
                        self._ph_min[room_id], self._ph_sum[room_id]
                    )

                    # Store error for tracking
                    self._ph_error_history[room_id].append(error)

                # Calculate PH statistic
                ph_statistic = self._ph_sum[room_id] - self._ph_min[room_id]
                drift_metrics.ph_statistic = ph_statistic
                drift_metrics.ph_threshold = self.ph_threshold

                # Check for drift
                if ph_statistic > self.ph_threshold:
                    drift_metrics.ph_drift_detected = True
                    drift_metrics.drift_types.append(DriftType.CONCEPT_DRIFT)

                    # Reset PH test after detection
                    self._ph_sum[room_id] = 0.0
                    self._ph_min[room_id] = 0.0

        except Exception as e:
            logger.warning(f"Error in Page-Hinkley test for {room_id}: {e}")

    def _calculate_statistical_confidence(self, drift_metrics: DriftMetrics) -> None:
        """Calculate overall statistical confidence in drift detection."""
        try:
            # Factors affecting confidence
            sample_size_factor = min(
                (drift_metrics.sample_size_baseline + drift_metrics.sample_size_current)
                / 200,
                1.0,
            )

            # Statistical significance factor
            significant_tests = 0
            total_tests = 0

            if drift_metrics.ks_p_value > 0:
                total_tests += 1
                if drift_metrics.ks_p_value < self.alpha:
                    significant_tests += 1

            if drift_metrics.mw_p_value > 0:
                total_tests += 1
                if drift_metrics.mw_p_value < self.alpha:
                    significant_tests += 1

            significance_factor = significant_tests / max(total_tests, 1)

            # Multiple test agreement factor
            agreement_factor = 1.0
            if len(drift_metrics.drift_types) > 1:
                agreement_factor = 1.2  # Multiple types increase confidence

            # Calculate overall confidence
            confidence = min(
                sample_size_factor * 0.4
                + significance_factor * 0.5
                + agreement_factor * 0.1,
                1.0,
            )

            drift_metrics.statistical_confidence = confidence

        except Exception as e:
            logger.warning(f"Error calculating statistical confidence: {e}")
            drift_metrics.statistical_confidence = 0.5  # Default moderate confidence

    async def _get_feature_data(
        self,
        feature_engine,
        room_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Optional[pd.DataFrame]:
        """Get feature data for the specified time period."""
        try:
            # This would integrate with the feature engineering engine
            # For now, return None to indicate no feature data available
            logger.debug(
                f"Getting feature data for {room_id} from {start_time} to {end_time}"
            )
            return None
        except Exception as e:
            logger.warning(f"Error getting feature data: {e}")
            return None

    async def _get_occupancy_patterns(
        self, room_id: str, start_time: datetime, end_time: datetime
    ) -> Optional[Dict[str, Any]]:
        """Get occupancy patterns for pattern drift analysis."""
        try:
            async with get_db_session() as session:
                # Query occupancy events
                query = (
                    select(SensorEvent)
                    .where(
                        and_(
                            SensorEvent.room_id == room_id,
                            SensorEvent.timestamp >= start_time,
                            SensorEvent.timestamp <= end_time,
                            SensorEvent.sensor_type.in_(["motion", "presence"]),
                        )
                    )
                    .order_by(SensorEvent.timestamp)
                )

                result = await session.execute(query)
                events = result.scalars().all()

                if not events:
                    return None

                # Analyze patterns
                patterns = {
                    "hourly_distribution": defaultdict(int),
                    "daily_frequency": defaultdict(int),
                    "total_events": len(events),
                }

                for event in events:
                    hour = event.timestamp.hour
                    day = event.timestamp.date()

                    patterns["hourly_distribution"][hour] += 1
                    patterns["daily_frequency"][str(day)] += 1

                return patterns

        except Exception as e:
            logger.warning(f"Error getting occupancy patterns: {e}")
            return None

    def _compare_temporal_patterns(
        self,
        baseline_patterns: Dict[str, Any],
        current_patterns: Dict[str, Any],
    ) -> float:
        """Compare temporal (hourly) occupancy patterns."""
        try:
            baseline_hourly = baseline_patterns["hourly_distribution"]
            current_hourly = current_patterns["hourly_distribution"]

            # Normalize distributions
            baseline_total = sum(baseline_hourly.values())
            current_total = sum(current_hourly.values())

            if baseline_total == 0 or current_total == 0:
                return 0.0

            # Calculate KL divergence
            kl_div = 0.0
            for hour in range(24):
                baseline_prob = baseline_hourly.get(hour, 1) / baseline_total
                current_prob = current_hourly.get(hour, 1) / current_total

                if current_prob > 0:
                    kl_div += current_prob * np.log(current_prob / baseline_prob)

            # Normalize to 0-1 scale
            return min(kl_div / 3.0, 1.0)

        except Exception as e:
            logger.warning(f"Error comparing temporal patterns: {e}")
            return 0.0

    def _compare_frequency_patterns(
        self,
        baseline_patterns: Dict[str, Any],
        current_patterns: Dict[str, Any],
    ) -> float:
        """Compare frequency (daily count) patterns."""
        try:
            baseline_daily = list(baseline_patterns["daily_frequency"].values())
            current_daily = list(current_patterns["daily_frequency"].values())

            if len(baseline_daily) < 3 or len(current_daily) < 3:
                return 0.0

            # Use Mann-Whitney U test for frequency comparison
            statistic, p_value = stats.mannwhitneyu(
                baseline_daily, current_daily, alternative="two-sided"
            )

            # Convert p-value to drift score
            return 1.0 - p_value

        except Exception as e:
            logger.warning(f"Error comparing frequency patterns: {e}")
            return 0.0

    async def _get_recent_prediction_errors(
        self, room_id: str, days: int
    ) -> List[float]:
        """Get recent prediction errors for Page-Hinkley test."""
        try:
            async with get_db_session() as session:
                # Query recent predictions with validation
                query = (
                    select(Prediction)
                    .where(
                        and_(
                            Prediction.room_id == room_id,
                            Prediction.created_at
                            >= datetime.now() - timedelta(days=days),
                            Prediction.actual_time.isnot(None),
                        )
                    )
                    .order_by(Prediction.created_at)
                )

                result = await session.execute(query)
                predictions = result.scalars().all()

                errors = []
                for pred in predictions:
                    if pred.actual_time and pred.predicted_time:
                        error_minutes = abs(
                            (pred.actual_time - pred.predicted_time).total_seconds()
                            / 60
                        )
                        errors.append(error_minutes)

                return errors

        except Exception as e:
            logger.warning(f"Error getting prediction errors: {e}")
            return []


class FeatureDriftDetector:
    """
    Specialized detector for feature distribution monitoring.

    Monitors individual feature distributions and identifies which specific
    features are experiencing drift, useful for targeted model updates.
    """

    def __init__(
        self,
        monitor_window_hours: int = 168,  # 1 week
        comparison_window_hours: int = 336,  # 2 weeks
        min_samples_per_window: int = 50,
        significance_threshold: float = 0.05,
    ):
        """
        Initialize feature drift detector.

        Args:
            monitor_window_hours: Hours of recent data to monitor
            comparison_window_hours: Hours of historical data for comparison
            min_samples_per_window: Minimum samples required per window
            significance_threshold: Statistical significance threshold
        """
        self.monitor_window_hours = monitor_window_hours
        self.comparison_window_hours = comparison_window_hours
        self.min_samples_per_window = min_samples_per_window
        self.significance_threshold = significance_threshold

        # Feature monitoring state
        self._feature_baselines = {}
        self._monitoring_active = False
        self._monitoring_task = None
        self._drift_callbacks = []

        logger.info(
            f"Initialized FeatureDriftDetector with {monitor_window_hours}h monitor window, "
            f"{comparison_window_hours}h comparison window"
        )

    async def start_monitoring(self, room_ids: List[str]) -> None:
        """Start continuous feature drift monitoring."""
        if self._monitoring_active:
            logger.warning("Feature drift monitoring already active")
            return

        self._monitoring_active = True
        self._monitoring_task = asyncio.create_task(self._monitoring_loop(room_ids))

        logger.info(f"Started feature drift monitoring for rooms: {room_ids}")

    async def stop_monitoring(self) -> None:
        """Stop continuous monitoring."""
        self._monitoring_active = False

        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None

        logger.info("Stopped feature drift monitoring")

    async def detect_feature_drift(
        self, room_id: str, feature_data: pd.DataFrame
    ) -> List[FeatureDriftResult]:
        """
        Detect drift in individual features.

        Args:
            room_id: Room to analyze
            feature_data: DataFrame with features and timestamps

        Returns:
            List of drift results for each feature
        """
        drift_results = []

        # Split data into comparison windows
        current_time = datetime.now()
        monitor_start = current_time - timedelta(hours=self.monitor_window_hours)
        comparison_start = monitor_start - timedelta(hours=self.comparison_window_hours)

        # Filter data by time windows
        if "timestamp" in feature_data.columns:
            comparison_data = feature_data[
                (feature_data["timestamp"] >= comparison_start)
                & (feature_data["timestamp"] < monitor_start)
            ]
            monitor_data = feature_data[feature_data["timestamp"] >= monitor_start]
        else:
            # If no timestamp, use recent vs older data split
            split_point = len(feature_data) // 2
            comparison_data = feature_data.iloc[:split_point]
            monitor_data = feature_data.iloc[split_point:]

        # Check minimum sample requirements
        if (
            len(comparison_data) < self.min_samples_per_window
            or len(monitor_data) < self.min_samples_per_window
        ):
            logger.warning(
                f"Insufficient samples for drift detection in {room_id}: "
                f"comparison={len(comparison_data)}, monitor={len(monitor_data)}"
            )
            return drift_results

        # Test each feature for drift
        for feature in feature_data.columns:
            if feature in ["room_id", "timestamp"]:
                continue

            try:
                # Test feature for drift
                drift_result = await self._test_single_feature_drift(
                    comparison_data[feature], monitor_data[feature], feature
                )

                drift_results.append(drift_result)

                # Trigger callbacks if significant drift detected
                if drift_result.is_significant(self.significance_threshold):
                    await self._notify_drift_callbacks(room_id, drift_result)

            except Exception as e:
                logger.warning(f"Error testing drift for feature {feature}: {e}")

        return drift_results

    async def _test_single_feature_drift(
        self,
        baseline_data: pd.Series,
        current_data: pd.Series,
        feature_name: str,
    ) -> FeatureDriftResult:
        """Test single feature for distribution drift."""
        # Clean data
        baseline_clean = baseline_data.dropna()
        current_clean = current_data.dropna()

        if len(baseline_clean) < 5 or len(current_clean) < 5:
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
                test_statistic=0.0,
                p_value=1.0,
                baseline_stats={},
                current_stats={},
            )

        # Choose appropriate test
        if baseline_clean.dtype in ["object", "category"]:
            return await self._test_categorical_feature_drift(
                baseline_clean, current_clean, feature_name
            )
        else:
            return await self._test_numerical_feature_drift(
                baseline_clean, current_clean, feature_name
            )

    async def _test_numerical_feature_drift(
        self, baseline: pd.Series, current: pd.Series, feature_name: str
    ) -> FeatureDriftResult:
        """Test numerical feature drift using KS test."""
        # Kolmogorov-Smirnov test
        ks_stat, ks_p = stats.ks_2samp(baseline, current)

        # Calculate statistics
        baseline_stats = {
            "mean": float(baseline.mean()),
            "std": float(baseline.std()),
            "median": float(baseline.median()),
            "q25": float(baseline.quantile(0.25)),
            "q75": float(baseline.quantile(0.75)),
            "skewness": float(baseline.skew()),
            "kurtosis": float(baseline.kurtosis()),
        }

        current_stats = {
            "mean": float(current.mean()),
            "std": float(current.std()),
            "median": float(current.median()),
            "q25": float(current.quantile(0.25)),
            "q75": float(current.quantile(0.75)),
            "skewness": float(current.skew()),
            "kurtosis": float(current.kurtosis()),
        }

        # Calculate drift score
        drift_score = min(ks_stat * 2, 1.0)

        return FeatureDriftResult(
            feature_name=feature_name,
            drift_detected=ks_p < self.significance_threshold,
            drift_score=drift_score,
            statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
            test_statistic=ks_stat,
            p_value=ks_p,
            baseline_stats=baseline_stats,
            current_stats=current_stats,
        )

    async def _test_categorical_feature_drift(
        self, baseline: pd.Series, current: pd.Series, feature_name: str
    ) -> FeatureDriftResult:
        """Test categorical feature drift using Chi-square test."""
        try:
            # Get value counts
            baseline_counts = baseline.value_counts()
            current_counts = current.value_counts()

            # Create contingency table
            all_values = sorted(set(baseline_counts.index) | set(current_counts.index))
            baseline_aligned = [baseline_counts.get(val, 0) for val in all_values]
            current_aligned = [current_counts.get(val, 0) for val in all_values]

            # Chi-square test
            chi2_stat, chi2_p, dof, expected = stats.chi2_contingency(
                [baseline_aligned, current_aligned]
            )

            # Calculate statistics
            baseline_stats = {
                "mode": (baseline.mode().iloc[0] if len(baseline.mode()) > 0 else None),
                "unique_values": baseline.nunique(),
                "entropy": -sum(
                    p * np.log2(p)
                    for p in baseline.value_counts(normalize=True)
                    if p > 0
                ),
            }

            current_stats = {
                "mode": (current.mode().iloc[0] if len(current.mode()) > 0 else None),
                "unique_values": current.nunique(),
                "entropy": -sum(
                    p * np.log2(p)
                    for p in current.value_counts(normalize=True)
                    if p > 0
                ),
            }

            # Calculate drift score
            drift_score = min(chi2_stat / (len(all_values) * 10), 1.0)

            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=chi2_p < self.significance_threshold,
                drift_score=drift_score,
                statistical_test=StatisticalTest.CHI_SQUARE,
                test_statistic=chi2_stat,
                p_value=chi2_p,
                baseline_stats=baseline_stats,
                current_stats=current_stats,
            )

        except Exception as e:
            logger.warning(f"Error in categorical drift test: {e}")
            return FeatureDriftResult(
                feature_name=feature_name,
                drift_detected=False,
                drift_score=0.0,
                statistical_test=StatisticalTest.CHI_SQUARE,
                test_statistic=0.0,
                p_value=1.0,
                baseline_stats={},
                current_stats={},
            )

    async def _monitoring_loop(self, room_ids: List[str]) -> None:
        """Background monitoring loop for continuous feature drift detection."""
        while self._monitoring_active:
            try:
                for room_id in room_ids:
                    # Get recent feature data
                    feature_data = await self._get_recent_feature_data(room_id)

                    if (
                        feature_data is not None
                        and len(feature_data) > self.min_samples_per_window * 2
                    ):
                        # Detect drift
                        drift_results = await self.detect_feature_drift(
                            room_id, feature_data
                        )

                        # Log significant drift
                        significant_drifts = [
                            r for r in drift_results if r.is_significant()
                        ]
                        if significant_drifts:
                            logger.info(
                                f"Feature drift detected in {room_id}: "
                                f"{[r.feature_name for r in significant_drifts]}"
                            )

                # Wait before next check
                await asyncio.sleep(3600)  # Check every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in feature monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retrying

    async def _get_recent_feature_data(self, room_id: str) -> Optional[pd.DataFrame]:
        """Get recent feature data for monitoring."""
        try:
            # This would integrate with the feature store to get recent features
            # For now, return None to indicate no data available
            return None
        except Exception as e:
            logger.warning(f"Error getting recent feature data for {room_id}: {e}")
            return None

    def add_drift_callback(
        self, callback: Callable[[str, FeatureDriftResult], None]
    ) -> None:
        """Add callback for drift notifications."""
        self._drift_callbacks.append(callback)

    def remove_drift_callback(
        self, callback: Callable[[str, FeatureDriftResult], None]
    ) -> None:
        """Remove drift notification callback."""
        if callback in self._drift_callbacks:
            self._drift_callbacks.remove(callback)

    async def _notify_drift_callbacks(
        self, room_id: str, drift_result: FeatureDriftResult
    ) -> None:
        """Notify registered callbacks about drift detection."""
        for callback in self._drift_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(room_id, drift_result)
                else:
                    callback(room_id, drift_result)
            except Exception as e:
                logger.warning(f"Error in drift callback: {e}")


class DriftDetectionError(OccupancyPredictionError):
    """Custom exception for drift detection failures."""

    def __init__(
        self,
        message: str,
        error_context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        super().__init__(message, error_context, severity)
