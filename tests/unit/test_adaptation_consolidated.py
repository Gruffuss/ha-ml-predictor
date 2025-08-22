"""
Consolidated Adaptation Module Test Suite

This comprehensive test suite consolidates all adaptation component testing,
eliminating duplications and providing complete coverage for:
- ConceptDriftDetector and FeatureDriftDetector
- AdaptiveRetrainer and ModelOptimizer
- PredictionValidator and AccuracyTracker
- TrackingManager system coordination

All duplicate test methods have been unified with the most comprehensive implementation.
"""

import asyncio
from collections import defaultdict, deque
from datetime import UTC, datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import tempfile
import threading
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio
from scipy import stats
import statistics

# Adaptation module imports
from src.adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftMetrics,
    DriftSeverity,
    DriftType,
    FeatureDriftDetector,
    FeatureDriftResult,
    StatisticalTest,
)
from src.adaptation.optimizer import (
    HyperparameterSpace,
    ModelOptimizer,
    OptimizationConfig,
    OptimizationResult,
    OptimizationStatus,
)
from src.adaptation.retrainer import (
    AdaptiveRetrainer,
    RetrainingHistory,
    RetrainingRequest,
    RetrainingStatus,
    RetrainingTrigger,
)
from src.adaptation.tracker import (
    AccuracyAlert,
    AccuracyTracker,
    AccuracyTrackingError,
    AlertSeverity,
    RealTimeMetrics,
    TrendDirection,
)
from src.adaptation.tracking_manager import (
    TrackingConfig,
    TrackingManager,
    TrackingManagerError,
)
from src.adaptation.validator import (
    AccuracyLevel,
    AccuracyMetrics,
    PredictionValidator,
    ValidationError,
    ValidationRecord,
)

# Core imports
from src.core.constants import ModelType
from src.core.exceptions import ErrorSeverity, OccupancyPredictionError
from src.models.base.predictor import PredictionResult

# ============================================================================
# FIXTURES AND TEST DATA
# ============================================================================


@pytest.fixture
def tracking_config():
    """Create comprehensive tracking configuration for testing."""
    return TrackingConfig(
        enabled=True,
        monitoring_interval_seconds=30,
        auto_validation_enabled=True,
        validation_window_minutes=15,
        alert_thresholds={
            "accuracy_warning": 70.0,
            "accuracy_critical": 50.0,
            "error_warning": 20.0,
            "error_critical": 30.0,
        },
        drift_detection_enabled=True,
        drift_check_interval_hours=6,
        auto_retraining_enabled=True,
        retraining_accuracy_threshold=65.0,
        optimization_enabled=True,
        model_persistence_enabled=True,
        alert_channels=["log", "metrics"],
    )


@pytest.fixture
def mock_prediction_validator():
    """Mock prediction validator with comprehensive test data."""
    validator = Mock(spec=PredictionValidator)

    # Mock accuracy metrics for different time periods
    baseline_metrics = AccuracyMetrics(
        total_predictions=100,
        validated_predictions=95,
        accurate_predictions=80,
        accuracy_rate=84.2,
        mae_minutes=12.5,
        confidence_accuracy_correlation=0.78,
    )

    current_metrics = AccuracyMetrics(
        total_predictions=50,
        validated_predictions=48,
        accurate_predictions=30,
        accuracy_rate=62.5,
        mae_minutes=23.8,
        confidence_accuracy_correlation=0.65,
    )

    async def get_accuracy_metrics(room_id, start_time, end_time):
        if start_time < datetime.now() - timedelta(days=20):
            return baseline_metrics
        else:
            return current_metrics

    validator.get_accuracy_metrics = AsyncMock(side_effect=get_accuracy_metrics)
    return validator


@pytest.fixture
def drift_detector():
    """Create drift detector with comprehensive test configuration."""
    return ConceptDriftDetector(
        baseline_days=14,
        current_days=3,
        min_samples=20,
        alpha=0.05,
        ph_threshold=30.0,
        psi_threshold=0.20,
    )


@pytest.fixture
def feature_drift_detector():
    """Create feature drift detector with test configuration."""
    return FeatureDriftDetector(
        monitor_window_hours=24,
        comparison_window_hours=48,
        min_samples_per_window=30,
        significance_threshold=0.05,
    )


@pytest.fixture
def synthetic_feature_data():
    """Generate synthetic feature data with and without drift."""
    np.random.seed(42)

    now = datetime.now()
    comparison_start = now - timedelta(hours=72)
    comparison_end = now - timedelta(hours=24)
    monitor_start = now - timedelta(hours=24)
    monitor_end = now

    # Stable features (no drift)
    stable_comparison = np.random.normal(100, 15, 50)
    stable_monitor = np.random.normal(102, 15, 50)

    # Drifted features (significant shift)
    drifted_comparison = np.random.normal(100, 15, 50)
    drifted_monitor = np.random.normal(130, 20, 50)

    return {
        "stable": {
            "comparison": stable_comparison,
            "monitor": stable_monitor,
            "comparison_times": [
                comparison_start + timedelta(minutes=i * 30) for i in range(50)
            ],
            "monitor_times": [
                monitor_start + timedelta(minutes=i * 30) for i in range(50)
            ],
        },
        "drifted": {
            "comparison": drifted_comparison,
            "monitor": drifted_monitor,
            "comparison_times": [
                comparison_start + timedelta(minutes=i * 30) for i in range(50)
            ],
            "monitor_times": [
                monitor_start + timedelta(minutes=i * 30) for i in range(50)
            ],
        },
    }


@pytest.fixture
def sample_prediction_records():
    """Generate sample prediction records for testing."""
    now = datetime.now()
    return [
        ValidationRecord(
            room_id="living_room",
            predicted_time=now + timedelta(minutes=30),
            actual_time=now + timedelta(minutes=25),
            confidence=0.85,
            prediction_created_at=now,
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.5, "motion": 1},
            validation_lag_minutes=2.0,
            accuracy_level=AccuracyLevel.HIGH,
        ),
        ValidationRecord(
            room_id="living_room",
            predicted_time=now + timedelta(minutes=45),
            actual_time=now + timedelta(minutes=60),
            confidence=0.72,
            prediction_created_at=now,
            model_type=ModelType.ENSEMBLE,
            features={"temp": 23.1, "motion": 0},
            validation_lag_minutes=3.5,
            accuracy_level=AccuracyLevel.MEDIUM,
        ),
    ]


# ============================================================================
# DRIFT DETECTION TESTS
# ============================================================================


class TestConceptDriftDetector:
    """Comprehensive tests for ConceptDriftDetector."""

    def test_initialization(self, drift_detector):
        """Test drift detector initialization."""
        assert drift_detector.baseline_days == 14
        assert drift_detector.current_days == 3
        assert drift_detector.min_samples == 20
        assert drift_detector.alpha == 0.05
        assert drift_detector.ph_threshold == 30.0
        assert drift_detector.psi_threshold == 0.20

    @pytest.mark.asyncio
    async def test_detect_accuracy_drift_no_drift(
        self, drift_detector, mock_prediction_validator
    ):
        """Test drift detection when no drift is present."""
        # Mock stable accuracy metrics
        stable_metrics = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=95,
            accurate_predictions=80,
            accuracy_rate=84.2,
            mae_minutes=12.5,
            confidence_accuracy_correlation=0.78,
        )
        mock_prediction_validator.get_accuracy_metrics.return_value = stable_metrics

        drift_result = await drift_detector.detect_accuracy_drift(
            "test_room", mock_prediction_validator
        )

        assert not drift_result.drift_detected
        assert drift_result.drift_severity == DriftSeverity.NONE
        assert drift_result.drift_type == DriftType.ACCURACY

    @pytest.mark.asyncio
    async def test_detect_accuracy_drift_significant_drift(
        self, drift_detector, mock_prediction_validator
    ):
        """Test drift detection with significant accuracy degradation."""
        drift_result = await drift_detector.detect_accuracy_drift(
            "test_room", mock_prediction_validator
        )

        assert drift_result.drift_detected
        assert drift_result.drift_severity in [
            DriftSeverity.MODERATE,
            DriftSeverity.SEVERE,
        ]
        assert drift_result.drift_type == DriftType.ACCURACY
        assert drift_result.statistical_significance < 0.05

    def test_calculate_population_stability_index_stable(self, drift_detector):
        """Test PSI calculation for stable distribution."""
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(102, 15, 500)

        psi = drift_detector._calculate_population_stability_index(baseline, current)
        assert psi < 0.10  # Should indicate stable distribution

    def test_calculate_population_stability_index_drift(self, drift_detector):
        """Test PSI calculation for drifted distribution."""
        baseline = np.random.normal(100, 15, 1000)
        current = np.random.normal(130, 20, 500)

        psi = drift_detector._calculate_population_stability_index(baseline, current)
        assert psi > 0.20  # Should indicate significant drift

    def test_perform_page_hinkley_test_no_drift(self, drift_detector):
        """Test Page-Hinkley test with stable accuracy."""
        accuracies = [0.85, 0.82, 0.88, 0.84, 0.87, 0.83, 0.86, 0.85]

        result = drift_detector._perform_page_hinkley_test(accuracies)
        assert not result.drift_detected

    def test_perform_page_hinkley_test_with_drift(self, drift_detector):
        """Test Page-Hinkley test with accuracy drift."""
        # Simulate accuracy degradation
        accuracies = [0.85, 0.82, 0.75, 0.68, 0.60, 0.55, 0.50, 0.45]

        result = drift_detector._perform_page_hinkley_test(accuracies)
        assert result.drift_detected
        assert result.drift_severity in [DriftSeverity.MODERATE, DriftSeverity.SEVERE]


class TestFeatureDriftDetector:
    """Comprehensive tests for FeatureDriftDetector."""

    def test_initialization(self, feature_drift_detector):
        """Test feature drift detector initialization."""
        assert feature_drift_detector.monitor_window_hours == 24
        assert feature_drift_detector.comparison_window_hours == 48
        assert feature_drift_detector.min_samples_per_window == 30
        assert feature_drift_detector.significance_threshold == 0.05

    @pytest.mark.asyncio
    async def test_detect_feature_drift_no_drift(
        self, feature_drift_detector, synthetic_feature_data
    ):
        """Test feature drift detection with stable features."""
        stable_data = synthetic_feature_data["stable"]

        with patch("src.data.storage.database.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            # Mock database query results
            comparison_results = list(
                zip(stable_data["comparison"], stable_data["comparison_times"])
            )
            monitor_results = list(
                zip(stable_data["monitor"], stable_data["monitor_times"])
            )

            async def mock_execute(query):
                mock_result = AsyncMock()
                if "monitor_start" in str(query):
                    mock_result.fetchall.return_value = monitor_results
                else:
                    mock_result.fetchall.return_value = comparison_results
                return mock_result

            mock_session.execute.side_effect = mock_execute

            drift_result = await feature_drift_detector.detect_feature_drift(
                "test_room", "temperature"
            )

        assert not drift_result.drift_detected
        assert drift_result.p_value > 0.05

    @pytest.mark.asyncio
    async def test_detect_feature_drift_with_drift(
        self, feature_drift_detector, synthetic_feature_data
    ):
        """Test feature drift detection with drifted features."""
        drifted_data = synthetic_feature_data["drifted"]

        with patch("src.data.storage.database.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            # Mock database query results
            comparison_results = list(
                zip(drifted_data["comparison"], drifted_data["comparison_times"])
            )
            monitor_results = list(
                zip(drifted_data["monitor"], drifted_data["monitor_times"])
            )

            async def mock_execute(query):
                mock_result = AsyncMock()
                if "monitor_start" in str(query):
                    mock_result.fetchall.return_value = monitor_results
                else:
                    mock_result.fetchall.return_value = comparison_results
                return mock_result

            mock_session.execute.side_effect = mock_execute

            drift_result = await feature_drift_detector.detect_feature_drift(
                "test_room", "temperature"
            )

        assert drift_result.drift_detected
        assert drift_result.p_value < 0.05
        assert drift_result.test_statistic > 0

    def test_statistical_tests(self, feature_drift_detector):
        """Test various statistical test implementations."""
        stable_baseline = np.random.normal(100, 15, 100)
        stable_current = np.random.normal(102, 15, 100)

        drifted_baseline = np.random.normal(100, 15, 100)
        drifted_current = np.random.normal(130, 20, 100)

        # Test stable data
        stable_ks = feature_drift_detector._perform_kolmogorov_smirnov_test(
            stable_baseline, stable_current
        )
        stable_mw = feature_drift_detector._perform_mann_whitney_test(
            stable_baseline, stable_current
        )

        assert stable_ks.p_value > 0.05
        assert stable_mw.p_value > 0.05

        # Test drifted data
        drifted_ks = feature_drift_detector._perform_kolmogorov_smirnov_test(
            drifted_baseline, drifted_current
        )
        drifted_mw = feature_drift_detector._perform_mann_whitney_test(
            drifted_baseline, drifted_current
        )

        assert drifted_ks.p_value < 0.05
        assert drifted_mw.p_value < 0.05


# ============================================================================
# ACCURACY TRACKING TESTS
# ============================================================================


class TestRealTimeMetrics:
    """Test RealTimeMetrics dataclass and properties."""

    def test_real_time_metrics_initialization(self):
        """Test comprehensive initialization with defaults."""
        metrics = RealTimeMetrics(room_id="test_room", model_type=ModelType.ENSEMBLE)

        assert metrics.room_id == "test_room"
        assert metrics.model_type == ModelType.ENSEMBLE
        assert metrics.window_1h_accuracy == 0.0
        assert metrics.window_6h_accuracy == 0.0
        assert metrics.window_24h_accuracy == 0.0
        assert metrics.accuracy_trend == TrendDirection.UNKNOWN
        assert metrics.trend_slope == 0.0
        assert metrics.active_alerts == []
        assert isinstance(metrics.last_updated, datetime)
        assert isinstance(metrics.recent_validation_records, list)

    def test_overall_health_score_calculation(self):
        """Test comprehensive health score calculation."""
        # High performance metrics
        metrics = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=90.0,
            window_24h_predictions=100,
            accuracy_trend=TrendDirection.IMPROVING,
            trend_confidence=0.9,
            confidence_calibration=0.85,
            validation_lag_minutes=3.0,
        )

        health_score = metrics.overall_health_score
        assert 80 <= health_score <= 100

        # Poor performance metrics
        metrics_poor = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=40.0,
            window_24h_predictions=50,
            accuracy_trend=TrendDirection.DEGRADING,
            trend_confidence=0.8,
            confidence_calibration=0.6,
            validation_lag_minutes=15.0,
        )

        health_score_poor = metrics_poor.overall_health_score
        assert health_score_poor < 50

    def test_is_healthy_property(self):
        """Test healthy status determination."""
        healthy_metrics = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=85.0,
            window_24h_predictions=50,
            accuracy_trend=TrendDirection.STABLE,
        )
        assert healthy_metrics.is_healthy

        unhealthy_metrics = RealTimeMetrics(
            room_id="test_room",
            window_24h_accuracy=50.0,
            window_24h_predictions=30,
            accuracy_trend=TrendDirection.DEGRADING,
        )
        assert not unhealthy_metrics.is_healthy


class TestAccuracyAlert:
    """Test AccuracyAlert functionality."""

    def test_alert_initialization(self):
        """Test alert creation and properties."""
        alert = AccuracyAlert(
            room_id="test_room",
            severity=AlertSeverity.WARNING,
            message="Accuracy below threshold",
            accuracy_value=65.0,
            threshold_value=70.0,
            model_type=ModelType.ENSEMBLE,
        )

        assert alert.room_id == "test_room"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.message == "Accuracy below threshold"
        assert alert.accuracy_value == 65.0
        assert alert.threshold_value == 70.0
        assert alert.model_type == ModelType.ENSEMBLE
        assert isinstance(alert.timestamp, datetime)
        assert alert.resolved is False

    def test_alert_resolution(self):
        """Test alert resolution functionality."""
        alert = AccuracyAlert(
            room_id="test_room",
            severity=AlertSeverity.CRITICAL,
            message="Critical accuracy degradation",
            accuracy_value=45.0,
            threshold_value=50.0,
        )

        assert not alert.resolved
        assert alert.resolved_at is None

        alert.resolve()
        assert alert.resolved
        assert isinstance(alert.resolved_at, datetime)


class TestAccuracyTracker:
    """Comprehensive tests for AccuracyTracker."""

    @pytest.fixture
    def accuracy_tracker(self):
        """Create accuracy tracker for testing."""
        return AccuracyTracker(
            room_id="test_room",
            max_history_size=1000,
            alert_thresholds={"warning": 70.0, "critical": 50.0},
        )

    def test_tracker_initialization(self, accuracy_tracker):
        """Test tracker initialization."""
        assert accuracy_tracker.room_id == "test_room"
        assert accuracy_tracker.max_history_size == 1000
        assert accuracy_tracker.alert_thresholds["warning"] == 70.0
        assert accuracy_tracker.alert_thresholds["critical"] == 50.0
        assert len(accuracy_tracker.validation_records) == 0

    def test_record_validation_basic(self, accuracy_tracker, sample_prediction_records):
        """Test basic validation recording."""
        record = sample_prediction_records[0]
        accuracy_tracker.record_validation(record)

        assert len(accuracy_tracker.validation_records) == 1
        assert accuracy_tracker.validation_records[0] == record

    def test_calculate_window_accuracy(
        self, accuracy_tracker, sample_prediction_records
    ):
        """Test accuracy calculation for time windows."""
        # Add multiple records
        for record in sample_prediction_records:
            accuracy_tracker.record_validation(record)

        # Test 1-hour window
        accuracy_1h = accuracy_tracker.calculate_window_accuracy(hours=1)
        assert isinstance(accuracy_1h, float)
        assert 0 <= accuracy_1h <= 100

        # Test 24-hour window
        accuracy_24h = accuracy_tracker.calculate_window_accuracy(hours=24)
        assert isinstance(accuracy_24h, float)
        assert 0 <= accuracy_24h <= 100

    def test_detect_trend(self, accuracy_tracker):
        """Test trend detection in accuracy metrics."""
        # Simulate improving trend
        improving_accuracies = [60, 65, 70, 75, 80, 85]
        trend, slope, confidence = accuracy_tracker._detect_trend(improving_accuracies)
        assert trend == TrendDirection.IMPROVING
        assert slope > 0
        assert confidence > 0.5

        # Simulate degrading trend
        degrading_accuracies = [85, 80, 75, 70, 65, 60]
        trend, slope, confidence = accuracy_tracker._detect_trend(degrading_accuracies)
        assert trend == TrendDirection.DEGRADING
        assert slope < 0
        assert confidence > 0.5

    def test_alert_generation(self, accuracy_tracker):
        """Test automatic alert generation."""
        # Create record with low accuracy that should trigger warning
        low_accuracy_record = ValidationRecord(
            room_id="test_room",
            predicted_time=datetime.now() + timedelta(minutes=30),
            actual_time=datetime.now()
            + timedelta(minutes=60),  # 30 min error = 50% accuracy
            confidence=0.65,
            prediction_created_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.5},
            validation_lag_minutes=2.0,
            accuracy_level=AccuracyLevel.LOW,
        )

        initial_alerts = len(accuracy_tracker.active_alerts)
        accuracy_tracker.record_validation(low_accuracy_record)

        # Should generate alert if accuracy is below threshold
        assert len(accuracy_tracker.active_alerts) >= initial_alerts

    def test_get_real_time_metrics(self, accuracy_tracker, sample_prediction_records):
        """Test real-time metrics generation."""
        for record in sample_prediction_records:
            accuracy_tracker.record_validation(record)

        metrics = accuracy_tracker.get_real_time_metrics()
        assert isinstance(metrics, RealTimeMetrics)
        assert metrics.room_id == "test_room"
        assert metrics.window_24h_predictions > 0


# ============================================================================
# VALIDATION TESTS
# ============================================================================


class TestPredictionValidator:
    """Comprehensive tests for PredictionValidator."""

    @pytest.fixture
    def prediction_validator(self):
        """Create prediction validator for testing."""
        return PredictionValidator(accuracy_threshold_minutes=15)

    def test_validator_initialization(self, prediction_validator):
        """Test validator initialization."""
        assert prediction_validator.accuracy_threshold_minutes == 15
        assert len(prediction_validator.pending_validations) == 0

    @pytest.mark.asyncio
    async def test_record_prediction(self, prediction_validator):
        """Test prediction recording."""
        prediction_time = datetime.now() + timedelta(minutes=30)

        await prediction_validator.record_prediction(
            room_id="test_room",
            predicted_time=prediction_time,
            confidence=0.85,
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.5, "motion": 1},
        )

        assert len(prediction_validator.pending_validations) == 1
        assert "test_room" in prediction_validator.pending_validations

    @pytest.mark.asyncio
    async def test_validate_prediction_accurate(self, prediction_validator):
        """Test validation of accurate prediction."""
        prediction_time = datetime.now() + timedelta(minutes=30)
        actual_time = prediction_time + timedelta(minutes=5)  # 5 min error

        await prediction_validator.record_prediction(
            room_id="test_room",
            predicted_time=prediction_time,
            confidence=0.85,
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.5},
        )

        with patch("src.data.storage.database.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            result = await prediction_validator.validate_prediction(
                room_id="test_room", actual_time=actual_time
            )

        assert result is not None
        assert result.accuracy_level == AccuracyLevel.HIGH
        assert abs(result.error_minutes) <= 15

    @pytest.mark.asyncio
    async def test_validate_prediction_inaccurate(self, prediction_validator):
        """Test validation of inaccurate prediction."""
        prediction_time = datetime.now() + timedelta(minutes=30)
        actual_time = prediction_time + timedelta(minutes=45)  # 45 min error

        await prediction_validator.record_prediction(
            room_id="test_room",
            predicted_time=prediction_time,
            confidence=0.85,
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.5},
        )

        with patch("src.data.storage.database.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            result = await prediction_validator.validate_prediction(
                room_id="test_room", actual_time=actual_time
            )

        assert result is not None
        assert result.accuracy_level == AccuracyLevel.LOW
        assert abs(result.error_minutes) > 15

    @pytest.mark.asyncio
    async def test_get_accuracy_metrics(self, prediction_validator):
        """Test accuracy metrics calculation."""
        with patch("src.data.storage.database.get_db_session") as mock_db:
            mock_session = AsyncMock()
            mock_db.return_value.__aenter__.return_value = mock_session

            # Mock validation records
            mock_records = [
                (100, 95, 80),  # total, validated, accurate
                (25.5,),  # mae_minutes
                (0.78,),  # confidence_correlation
            ]

            async def mock_execute(query):
                mock_result = AsyncMock()
                if "COUNT(*)" in str(query):
                    mock_result.fetchone.return_value = mock_records[0]
                elif "AVG(ABS" in str(query):
                    mock_result.fetchone.return_value = mock_records[1]
                else:
                    mock_result.fetchone.return_value = mock_records[2]
                return mock_result

            mock_session.execute.side_effect = mock_execute

            metrics = await prediction_validator.get_accuracy_metrics(
                "test_room", datetime.now() - timedelta(days=1), datetime.now()
            )

        assert isinstance(metrics, AccuracyMetrics)
        assert metrics.total_predictions == 100
        assert metrics.validated_predictions == 95
        assert metrics.accurate_predictions == 80


# ============================================================================
# RETRAINING TESTS
# ============================================================================


class TestAdaptiveRetrainer:
    """Comprehensive tests for AdaptiveRetrainer."""

    @pytest.fixture
    def adaptive_retrainer(self):
        """Create adaptive retrainer for testing."""
        return AdaptiveRetrainer(
            accuracy_threshold=70.0,
            min_samples_for_retraining=50,
            max_retraining_frequency_hours=6,
        )

    def test_retrainer_initialization(self, adaptive_retrainer):
        """Test retrainer initialization."""
        assert adaptive_retrainer.accuracy_threshold == 70.0
        assert adaptive_retrainer.min_samples_for_retraining == 50
        assert adaptive_retrainer.max_retraining_frequency_hours == 6

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_accuracy_degradation(
        self, adaptive_retrainer
    ):
        """Test retraining evaluation with accuracy degradation."""
        # Mock accuracy metrics showing degradation
        mock_metrics = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=95,
            accurate_predictions=60,  # 63% accuracy - below threshold
            accuracy_rate=63.2,
            mae_minutes=22.5,
            confidence_accuracy_correlation=0.65,
        )

        with patch.object(
            adaptive_retrainer,
            "_get_current_accuracy_metrics",
            return_value=mock_metrics,
        ) as mock_get_metrics:
            need_retraining, trigger = (
                await adaptive_retrainer.evaluate_retraining_need("test_room")
            )

        assert need_retraining
        assert trigger == RetrainingTrigger.ACCURACY_DEGRADATION

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_insufficient_samples(
        self, adaptive_retrainer
    ):
        """Test retraining evaluation with insufficient samples."""
        # Mock metrics with low sample count
        mock_metrics = AccuracyMetrics(
            total_predictions=25,  # Below min_samples_for_retraining
            validated_predictions=25,
            accurate_predictions=15,
            accuracy_rate=60.0,
            mae_minutes=25.0,
            confidence_accuracy_correlation=0.6,
        )

        with patch.object(
            adaptive_retrainer,
            "_get_current_accuracy_metrics",
            return_value=mock_metrics,
        ):
            need_retraining, trigger = (
                await adaptive_retrainer.evaluate_retraining_need("test_room")
            )

        assert not need_retraining  # Should not retrain with insufficient samples

    @pytest.mark.asyncio
    async def test_request_retraining(self, adaptive_retrainer):
        """Test retraining request creation."""
        request = await adaptive_retrainer.request_retraining(
            room_id="test_room",
            trigger=RetrainingTrigger.ACCURACY_DEGRADATION,
            priority="high",
        )

        assert isinstance(request, RetrainingRequest)
        assert request.room_id == "test_room"
        assert request.trigger == RetrainingTrigger.ACCURACY_DEGRADATION
        assert request.priority == "high"
        assert request.status == RetrainingStatus.PENDING

    @pytest.mark.asyncio
    async def test_perform_retraining(self, adaptive_retrainer):
        """Test actual retraining execution."""
        request = RetrainingRequest(
            room_id="test_room",
            trigger=RetrainingTrigger.ACCURACY_DEGRADATION,
            priority="high",
        )

        with patch("src.models.trainer.ModelTrainer") as mock_trainer_class, patch(
            "src.features.store.FeatureStore"
        ) as mock_feature_store_class:

            mock_trainer = AsyncMock()
            mock_trainer.train_room_model.return_value = {"accuracy": 85.0}
            mock_trainer_class.return_value = mock_trainer

            mock_feature_store = AsyncMock()
            mock_feature_store.get_training_data.return_value = (
                pd.DataFrame({"feature1": [1, 2, 3]}),
                pd.DataFrame({"target": [0, 1, 0]}),
            )
            mock_feature_store_class.return_value = mock_feature_store

            result = await adaptive_retrainer.perform_retraining(request)

        assert result.status == RetrainingStatus.COMPLETED
        assert result.new_accuracy > 80.0


# ============================================================================
# MODEL OPTIMIZATION TESTS
# ============================================================================


class TestModelOptimizer:
    """Comprehensive tests for ModelOptimizer."""

    @pytest.fixture
    def optimization_config(self):
        """Create optimization configuration."""
        return OptimizationConfig(
            max_trials=10,
            timeout_minutes=30,
            optimization_metric="accuracy",
            early_stopping_patience=3,
            cross_validation_folds=3,
        )

    @pytest.fixture
    def model_optimizer(self, optimization_config):
        """Create model optimizer for testing."""
        return ModelOptimizer(config=optimization_config)

    def test_optimizer_initialization(self, model_optimizer, optimization_config):
        """Test optimizer initialization."""
        assert model_optimizer.config == optimization_config
        assert model_optimizer.config.max_trials == 10
        assert model_optimizer.config.timeout_minutes == 30

    @pytest.mark.asyncio
    async def test_optimize_hyperparameters(self, model_optimizer):
        """Test hyperparameter optimization."""
        # Define hyperparameter space
        param_space = HyperparameterSpace(
            learning_rate=(0.001, 0.1), batch_size=[16, 32, 64], dropout_rate=(0.1, 0.5)
        )

        # Mock training data
        training_data = pd.DataFrame(
            {"feature1": np.random.randn(100), "feature2": np.random.randn(100)}
        )
        targets = pd.DataFrame({"target": np.random.randint(0, 2, 100)})

        with patch("src.models.base.lstm_predictor.LSTMPredictor") as mock_model_class:
            mock_model = Mock()
            mock_model.train.return_value = {"accuracy": 0.85}
            mock_model.evaluate.return_value = {"accuracy": 0.82}
            mock_model_class.return_value = mock_model

            result = await model_optimizer.optimize_hyperparameters(
                model_type=ModelType.LSTM,
                param_space=param_space,
                training_data=training_data,
                targets=targets,
                room_id="test_room",
            )

        assert isinstance(result, OptimizationResult)
        assert result.status == OptimizationStatus.COMPLETED
        assert result.best_score > 0.5
        assert result.best_params is not None

    def test_generate_hyperparameter_combinations(self, model_optimizer):
        """Test hyperparameter combination generation."""
        param_space = HyperparameterSpace(
            learning_rate=(0.001, 0.1), batch_size=[16, 32, 64], dropout_rate=(0.1, 0.5)
        )

        combinations = model_optimizer._generate_hyperparameter_combinations(
            param_space, max_combinations=5
        )

        assert len(combinations) <= 5
        for combo in combinations:
            assert "learning_rate" in combo
            assert "batch_size" in combo
            assert "dropout_rate" in combo
            assert 0.001 <= combo["learning_rate"] <= 0.1
            assert combo["batch_size"] in [16, 32, 64]
            assert 0.1 <= combo["dropout_rate"] <= 0.5


# ============================================================================
# TRACKING MANAGER INTEGRATION TESTS
# ============================================================================


class TestTrackingManager:
    """Comprehensive tests for TrackingManager system coordination."""

    @pytest.fixture
    def tracking_manager(self, tracking_config):
        """Create tracking manager with all dependencies."""
        with patch(
            "src.adaptation.validator.PredictionValidator"
        ) as mock_validator, patch(
            "src.adaptation.tracker.AccuracyTracker"
        ) as mock_tracker, patch(
            "src.adaptation.drift_detector.ConceptDriftDetector"
        ) as mock_drift, patch(
            "src.adaptation.retrainer.AdaptiveRetrainer"
        ) as mock_retrainer, patch(
            "src.adaptation.optimizer.ModelOptimizer"
        ) as mock_optimizer:

            return TrackingManager(config=tracking_config)

    def test_tracking_manager_initialization(self, tracking_manager, tracking_config):
        """Test tracking manager initialization."""
        assert tracking_manager.config == tracking_config
        assert tracking_manager.config.enabled
        assert tracking_manager.config.monitoring_interval_seconds == 30

    @pytest.mark.asyncio
    async def test_start_monitoring(self, tracking_manager):
        """Test monitoring system startup."""
        with patch.object(tracking_manager, "_monitoring_loop") as mock_loop:
            mock_loop.return_value = None

            await tracking_manager.start_monitoring()

            assert tracking_manager.is_monitoring
            mock_loop.assert_called_once()

    @pytest.mark.asyncio
    async def test_stop_monitoring(self, tracking_manager):
        """Test monitoring system shutdown."""
        tracking_manager.is_monitoring = True
        tracking_manager._monitoring_task = AsyncMock()

        await tracking_manager.stop_monitoring()

        assert not tracking_manager.is_monitoring

    @pytest.mark.asyncio
    async def test_record_prediction_integration(self, tracking_manager):
        """Test prediction recording integration."""
        prediction_time = datetime.now() + timedelta(minutes=30)

        with patch.object(
            tracking_manager.validator, "record_prediction"
        ) as mock_record:
            await tracking_manager.record_prediction(
                room_id="test_room",
                predicted_time=prediction_time,
                confidence=0.85,
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.5},
            )

            mock_record.assert_called_once_with(
                room_id="test_room",
                predicted_time=prediction_time,
                confidence=0.85,
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.5},
            )

    @pytest.mark.asyncio
    async def test_validate_prediction_integration(self, tracking_manager):
        """Test prediction validation integration."""
        actual_time = datetime.now()

        with patch.object(
            tracking_manager.validator, "validate_prediction"
        ) as mock_validate, patch.object(
            tracking_manager, "_get_room_tracker"
        ) as mock_get_tracker:

            mock_validation_record = Mock()
            mock_validate.return_value = mock_validation_record

            mock_tracker = Mock()
            mock_get_tracker.return_value = mock_tracker

            result = await tracking_manager.validate_prediction(
                room_id="test_room", actual_time=actual_time
            )

            mock_validate.assert_called_once_with("test_room", actual_time)
            mock_tracker.record_validation.assert_called_once_with(
                mock_validation_record
            )

    @pytest.mark.asyncio
    async def test_monitoring_loop_integration(self, tracking_manager):
        """Test main monitoring loop integration."""
        tracking_manager.is_monitoring = True

        with patch.object(
            tracking_manager, "_check_accuracy_alerts"
        ) as mock_alerts, patch.object(
            tracking_manager, "_check_drift_detection"
        ) as mock_drift, patch.object(
            tracking_manager, "_evaluate_retraining_needs"
        ) as mock_retrain, patch(
            "asyncio.sleep"
        ) as mock_sleep:

            # Mock sleep to break the loop after one iteration
            mock_sleep.side_effect = [None, Exception("Break loop")]

            try:
                await tracking_manager._monitoring_loop()
            except Exception:
                pass  # Expected to break the loop

            mock_alerts.assert_called()
            mock_drift.assert_called()
            mock_retrain.assert_called()

    @pytest.mark.asyncio
    async def test_get_system_status(self, tracking_manager):
        """Test system status reporting."""
        with patch.object(tracking_manager, "_get_room_tracker") as mock_get_tracker:
            mock_tracker = Mock()
            mock_metrics = Mock()
            mock_metrics.overall_health_score = 85.0
            mock_metrics.is_healthy = True
            mock_tracker.get_real_time_metrics.return_value = mock_metrics
            mock_get_tracker.return_value = mock_tracker

            # Mock room list
            tracking_manager.room_trackers = {"test_room": mock_tracker}

            status = await tracking_manager.get_system_status()

            assert "test_room" in status["room_status"]
            assert status["room_status"]["test_room"]["health_score"] == 85.0
            assert status["room_status"]["test_room"]["is_healthy"]

    def test_error_handling(self, tracking_manager):
        """Test error handling in tracking manager."""
        with pytest.raises(TrackingManagerError):
            tracking_manager._validate_config(TrackingConfig(enabled=False))

    @pytest.mark.asyncio
    async def test_resource_cleanup(self, tracking_manager):
        """Test proper resource cleanup."""
        tracking_manager.is_monitoring = True
        tracking_manager._monitoring_task = Mock()

        await tracking_manager.cleanup()

        assert not tracking_manager.is_monitoring
        assert len(tracking_manager.room_trackers) == 0


# ============================================================================
# INTEGRATION AND ERROR SCENARIOS
# ============================================================================


class TestAdaptationIntegrationScenarios:
    """Test complex integration scenarios across adaptation components."""

    @pytest.mark.asyncio
    async def test_full_adaptation_workflow(self, tracking_config):
        """Test complete adaptation workflow from prediction to retraining."""
        with patch(
            "src.adaptation.validator.PredictionValidator"
        ) as mock_validator_class, patch(
            "src.adaptation.tracker.AccuracyTracker"
        ) as mock_tracker_class, patch(
            "src.adaptation.drift_detector.ConceptDriftDetector"
        ) as mock_drift_class, patch(
            "src.adaptation.retrainer.AdaptiveRetrainer"
        ) as mock_retrainer_class:

            # Setup mocks
            mock_validator = AsyncMock()
            mock_validator_class.return_value = mock_validator

            mock_tracker = Mock()
            mock_tracker_class.return_value = mock_tracker

            mock_drift_detector = AsyncMock()
            mock_drift_class.return_value = mock_drift_detector

            mock_retrainer = AsyncMock()
            mock_retrainer_class.return_value = mock_retrainer

            # Create tracking manager
            tracking_manager = TrackingManager(config=tracking_config)

            # Simulate prediction recording
            prediction_time = datetime.now() + timedelta(minutes=30)
            await tracking_manager.record_prediction(
                room_id="test_room",
                predicted_time=prediction_time,
                confidence=0.85,
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.5},
            )

            # Simulate validation
            actual_time = prediction_time + timedelta(
                minutes=45
            )  # Inaccurate prediction
            mock_validation_record = ValidationRecord(
                room_id="test_room",
                predicted_time=prediction_time,
                actual_time=actual_time,
                confidence=0.85,
                prediction_created_at=datetime.now(),
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.5},
                validation_lag_minutes=2.0,
                accuracy_level=AccuracyLevel.LOW,
            )
            mock_validator.validate_prediction.return_value = mock_validation_record

            await tracking_manager.validate_prediction("test_room", actual_time)

            # Verify workflow execution
            mock_validator.record_prediction.assert_called_once()
            mock_validator.validate_prediction.assert_called_once()
            mock_tracker.record_validation.assert_called_once()

    @pytest.mark.asyncio
    async def test_concurrent_monitoring_operations(self, tracking_config):
        """Test concurrent monitoring operations don't interfere."""
        tracking_manager = TrackingManager(config=tracking_config)

        async def record_multiple_predictions():
            tasks = []
            for i in range(10):
                task = tracking_manager.record_prediction(
                    room_id=f"room_{i}",
                    predicted_time=datetime.now() + timedelta(minutes=30 + i),
                    confidence=0.8 + (i * 0.01),
                    model_type=ModelType.ENSEMBLE,
                    features={"temp": 22.0 + i},
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

        # Should not raise any exceptions
        await record_multiple_predictions()

    def test_error_propagation(self, tracking_config):
        """Test error propagation through adaptation components."""
        with patch(
            "src.adaptation.validator.PredictionValidator"
        ) as mock_validator_class:
            mock_validator_class.side_effect = Exception(
                "Validator initialization failed"
            )

            with pytest.raises(Exception):
                TrackingManager(config=tracking_config)

    @pytest.mark.asyncio
    async def test_recovery_from_component_failures(self, tracking_config):
        """Test system recovery from individual component failures."""
        tracking_manager = TrackingManager(config=tracking_config)

        # Simulate drift detector failure
        with patch.object(
            tracking_manager.drift_detector,
            "detect_accuracy_drift",
            side_effect=Exception("Drift detection failed"),
        ):

            # Should handle gracefully and continue monitoring
            try:
                await tracking_manager._check_drift_detection()
            except Exception:
                pytest.fail("Should handle drift detector failure gracefully")


# ============================================================================
# PERFORMANCE AND EDGE CASE TESTS
# ============================================================================


class TestAdaptationPerformanceEdgeCases:
    """Test performance characteristics and edge cases."""

    def test_large_validation_history_performance(self):
        """Test performance with large validation history."""
        tracker = AccuracyTracker(room_id="test_room", max_history_size=10000)

        # Add large number of validation records
        start_time = datetime.now()
        for i in range(5000):
            record = ValidationRecord(
                room_id="test_room",
                predicted_time=datetime.now() + timedelta(minutes=i),
                actual_time=datetime.now() + timedelta(minutes=i + 5),
                confidence=0.8,
                prediction_created_at=datetime.now(),
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.0},
                validation_lag_minutes=1.0,
                accuracy_level=AccuracyLevel.MEDIUM,
            )
            tracker.record_validation(record)

        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()

        # Should process 5000 records reasonably quickly
        assert processing_time < 10.0  # Less than 10 seconds
        assert len(tracker.validation_records) == 5000

    def test_memory_management_with_history_limit(self):
        """Test memory management with history size limits."""
        tracker = AccuracyTracker(room_id="test_room", max_history_size=100)

        # Add more records than the limit
        for i in range(150):
            record = ValidationRecord(
                room_id="test_room",
                predicted_time=datetime.now() + timedelta(minutes=i),
                actual_time=datetime.now() + timedelta(minutes=i + 5),
                confidence=0.8,
                prediction_created_at=datetime.now(),
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.0},
                validation_lag_minutes=1.0,
                accuracy_level=AccuracyLevel.MEDIUM,
            )
            tracker.record_validation(record)

        # Should maintain limit
        assert len(tracker.validation_records) == 100

    def test_edge_case_empty_data(self):
        """Test edge cases with empty or insufficient data."""
        tracker = AccuracyTracker(room_id="test_room")

        # Test with no data
        accuracy = tracker.calculate_window_accuracy(hours=24)
        assert accuracy == 0.0

        metrics = tracker.get_real_time_metrics()
        assert metrics.window_24h_accuracy == 0.0
        assert metrics.accuracy_trend == TrendDirection.UNKNOWN

    def test_extreme_accuracy_values(self):
        """Test handling of extreme accuracy values."""
        tracker = AccuracyTracker(room_id="test_room")

        # Perfect prediction
        perfect_record = ValidationRecord(
            room_id="test_room",
            predicted_time=datetime.now() + timedelta(minutes=30),
            actual_time=datetime.now() + timedelta(minutes=30),  # Exact match
            confidence=0.95,
            prediction_created_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.0},
            validation_lag_minutes=1.0,
            accuracy_level=AccuracyLevel.HIGH,
        )

        # Very poor prediction
        poor_record = ValidationRecord(
            room_id="test_room",
            predicted_time=datetime.now() + timedelta(minutes=30),
            actual_time=datetime.now() + timedelta(hours=5),  # 5 hour error
            confidence=0.2,
            prediction_created_at=datetime.now(),
            model_type=ModelType.ENSEMBLE,
            features={"temp": 22.0},
            validation_lag_minutes=1.0,
            accuracy_level=AccuracyLevel.LOW,
        )

        tracker.record_validation(perfect_record)
        tracker.record_validation(poor_record)

        # Should handle extreme values gracefully
        accuracy = tracker.calculate_window_accuracy(hours=24)
        assert 0.0 <= accuracy <= 100.0

    @pytest.mark.asyncio
    async def test_concurrent_validation_recording(self):
        """Test concurrent validation recording."""
        tracker = AccuracyTracker(room_id="test_room")

        async def record_validation(i):
            record = ValidationRecord(
                room_id="test_room",
                predicted_time=datetime.now() + timedelta(minutes=30 + i),
                actual_time=datetime.now() + timedelta(minutes=35 + i),
                confidence=0.8,
                prediction_created_at=datetime.now(),
                model_type=ModelType.ENSEMBLE,
                features={"temp": 22.0},
                validation_lag_minutes=1.0,
                accuracy_level=AccuracyLevel.MEDIUM,
            )
            tracker.record_validation(record)

        # Record 100 validations concurrently
        tasks = [record_validation(i) for i in range(100)]
        await asyncio.gather(*tasks)

        assert len(tracker.validation_records) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
