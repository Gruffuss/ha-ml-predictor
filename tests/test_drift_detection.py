"""
Comprehensive tests for the Concept Drift Detection System.

Tests statistical drift detection, feature monitoring, and integration
with the existing prediction validation infrastructure.
"""

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftDetectionError,
    DriftMetrics,
    DriftSeverity,
    DriftType,
    FeatureDriftDetector,
    FeatureDriftResult,
    StatisticalTest,
)
from src.adaptation.validator import (
    AccuracyMetrics,
    PredictionValidator,
    ValidationRecord,
)
from src.core.exceptions import OccupancyPredictionError


class TestDriftMetrics:
    """Test DriftMetrics dataclass functionality."""

    def test_drift_metrics_initialization(self):
        """Test DriftMetrics initialization and post-init calculations."""
        baseline_period = (
            datetime.now() - timedelta(days=30),
            datetime.now() - timedelta(days=7),
        )
        current_period = (datetime.now() - timedelta(days=7), datetime.now())

        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=baseline_period,
            current_period=current_period,
            ks_p_value=0.01,  # Significant
            accuracy_degradation=20.0,  # 20 minutes degradation
            temporal_pattern_drift=0.4,
            frequency_pattern_drift=0.3,
        )

        # Check that post-init calculations were performed
        assert drift_metrics.overall_drift_score > 0
        assert (
            drift_metrics.drift_severity != DriftSeverity.MINOR
        )  # Should be higher due to degradation
        assert drift_metrics.retraining_recommended

    def test_drift_severity_classification(self):
        """Test drift severity classification logic."""
        # Critical drift - Page-Hinkley detection
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(
                datetime.now() - timedelta(days=7),
                datetime.now(),
            ),
            ph_drift_detected=True,
        )
        assert drift_metrics.drift_severity == DriftSeverity.CRITICAL

        # Major drift - high accuracy degradation
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(
                datetime.now() - timedelta(days=7),
                datetime.now(),
            ),
            accuracy_degradation=25.0,
        )
        assert drift_metrics.drift_severity == DriftSeverity.MAJOR

    def test_recommendations_generation(self):
        """Test recommendation generation logic."""
        # High degradation should require immediate attention
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(
                datetime.now() - timedelta(days=7),
                datetime.now(),
            ),
            accuracy_degradation=30.0,
        )
        assert drift_metrics.immediate_attention_required
        assert drift_metrics.retraining_recommended

        # Minor drift should not require immediate attention
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(
                datetime.now() - timedelta(days=7),
                datetime.now(),
            ),
            accuracy_degradation=5.0,
        )
        assert not drift_metrics.immediate_attention_required
        assert not drift_metrics.retraining_recommended

    def test_to_dict_serialization(self):
        """Test conversion to dictionary for serialization."""
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(
                datetime.now() - timedelta(days=7),
                datetime.now(),
            ),
            drifting_features=["feature1", "feature2"],
            drift_types=[DriftType.FEATURE_DRIFT, DriftType.CONCEPT_DRIFT],
        )

        result_dict = drift_metrics.to_dict()

        assert result_dict["room_id"] == "test_room"
        assert "statistical_tests" in result_dict
        assert "feature_analysis" in result_dict
        assert "assessment" in result_dict
        assert "recommendations" in result_dict
        assert result_dict["feature_analysis"]["drifting_features"] == [
            "feature1",
            "feature2",
        ]


class TestFeatureDriftResult:
    """Test FeatureDriftResult functionality."""

    def test_feature_drift_result_creation(self):
        """Test FeatureDriftResult creation and significance testing."""
        drift_result = FeatureDriftResult(
            feature_name="test_feature",
            drift_detected=True,
            drift_score=0.7,
            statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
            test_statistic=0.35,
            p_value=0.02,
            baseline_stats={"mean": 10.0, "std": 2.0},
            current_stats={"mean": 15.0, "std": 3.0},
        )

        assert drift_result.feature_name == "test_feature"
        assert drift_result.is_significant(alpha=0.05)
        assert not drift_result.is_significant(alpha=0.01)


class TestConceptDriftDetector:
    """Test ConceptDriftDetector functionality."""

    @pytest.fixture
    def drift_detector(self):
        """Create a drift detector for testing."""
        return ConceptDriftDetector(
            baseline_days=30,
            current_days=7,
            min_samples=10,  # Lower for testing
            alpha=0.05,
            ph_threshold=50.0,
            psi_threshold=0.25,
        )

    @pytest.fixture
    def mock_prediction_validator(self):
        """Create mock prediction validator."""
        validator = Mock(spec=PredictionValidator)

        # Mock accuracy metrics
        baseline_metrics = Mock(spec=AccuracyMetrics)
        baseline_metrics.mae_minutes = 10.0
        baseline_metrics.confidence_accuracy_correlation = 0.8
        baseline_metrics.recent_records = [
            Mock(error_minutes=8.0),
            Mock(error_minutes=12.0),
        ]

        current_metrics = Mock(spec=AccuracyMetrics)
        current_metrics.mae_minutes = 20.0
        current_metrics.confidence_accuracy_correlation = 0.6
        current_metrics.recent_records = [
            Mock(error_minutes=18.0),
            Mock(error_minutes=22.0),
        ]

        validator.get_accuracy_metrics = AsyncMock(
            side_effect=[baseline_metrics, current_metrics]
        )

        return validator

    @pytest.mark.asyncio
    async def test_detect_drift_basic(self, drift_detector, mock_prediction_validator):
        """Test basic drift detection functionality."""
        room_id = "test_room"

        with (
            patch.object(drift_detector, "_get_occupancy_patterns", return_value=None),
            patch.object(
                drift_detector,
                "_get_recent_prediction_errors",
                return_value=[],
            ),
        ):

            drift_metrics = await drift_detector.detect_drift(
                room_id=room_id, prediction_validator=mock_prediction_validator
            )

            assert drift_metrics.room_id == room_id
            assert drift_metrics.accuracy_degradation == 10.0  # 20 - 10
            assert drift_metrics.confidence_calibration_drift == 0.2  # |0.6 - 0.8|

    @pytest.mark.asyncio
    async def test_numerical_feature_drift_test(self, drift_detector):
        """Test numerical feature drift detection."""
        # Create test data with different distributions
        baseline_data = pd.Series(np.random.normal(10, 2, 100))
        current_data = pd.Series(np.random.normal(15, 3, 100))  # Different mean and std

        drift_result = await drift_detector._test_numerical_drift(
            baseline_data, current_data, "test_feature"
        )

        assert drift_result.feature_name == "test_feature"
        assert drift_result.statistical_test == StatisticalTest.KOLMOGOROV_SMIRNOV
        assert drift_result.p_value < 0.05  # Should detect significant drift
        assert drift_result.drift_detected
        assert "mean" in drift_result.baseline_stats
        assert "std" in drift_result.baseline_stats

    @pytest.mark.asyncio
    async def test_categorical_feature_drift_test(self, drift_detector):
        """Test categorical feature drift detection."""
        # Create test data with different distributions
        baseline_data = pd.Series(["A"] * 50 + ["B"] * 30 + ["C"] * 20)
        current_data = pd.Series(
            ["A"] * 20 + ["B"] * 30 + ["C"] * 50
        )  # Different distribution

        drift_result = await drift_detector._test_categorical_drift(
            baseline_data, current_data, "categorical_feature"
        )

        assert drift_result.feature_name == "categorical_feature"
        assert drift_result.statistical_test == StatisticalTest.CHI_SQUARE
        assert drift_result.p_value < 0.05  # Should detect significant drift
        assert drift_result.drift_detected
        assert "mode" in drift_result.baseline_stats
        assert "unique_values" in drift_result.baseline_stats

    def test_numerical_psi_calculation(self, drift_detector):
        """Test Population Stability Index calculation for numerical features."""
        baseline_data = pd.Series(np.random.normal(10, 2, 1000))
        current_data = pd.Series(np.random.normal(15, 2, 1000))  # Shifted mean

        psi = drift_detector._calculate_numerical_psi(baseline_data, current_data)

        assert psi > 0  # Should detect shift
        assert isinstance(psi, float)

    def test_categorical_psi_calculation(self, drift_detector):
        """Test PSI calculation for categorical features."""
        baseline_data = pd.Series(["A"] * 70 + ["B"] * 20 + ["C"] * 10)
        current_data = pd.Series(
            ["A"] * 40 + ["B"] * 30 + ["C"] * 30
        )  # Different distribution

        psi = drift_detector._calculate_categorical_psi(baseline_data, current_data)

        assert psi > 0  # Should detect drift
        assert isinstance(psi, float)

    def test_temporal_pattern_comparison(self, drift_detector):
        """Test temporal pattern drift comparison."""
        baseline_patterns = {
            "hourly_distribution": {8: 10, 12: 15, 18: 20, 22: 5},
            "total_events": 50,
        }

        current_patterns = {
            "hourly_distribution": {
                8: 5,
                12: 10,
                18: 10,
                22: 25,
            },  # Different pattern
            "total_events": 50,
        }

        drift_score = drift_detector._compare_temporal_patterns(
            baseline_patterns, current_patterns
        )

        assert drift_score > 0
        assert drift_score <= 1.0

    def test_frequency_pattern_comparison(self, drift_detector):
        """Test frequency pattern drift comparison."""
        baseline_patterns = {
            "daily_frequency": {
                "2024-01-01": 10,
                "2024-01-02": 12,
                "2024-01-03": 8,
            }
        }

        current_patterns = {
            "daily_frequency": {
                "2024-01-04": 20,
                "2024-01-05": 25,
                "2024-01-06": 22,
            }  # Higher frequency
        }

        drift_score = drift_detector._compare_frequency_patterns(
            baseline_patterns, current_patterns
        )

        assert drift_score >= 0
        assert drift_score <= 1.0

    def test_statistical_confidence_calculation(self, drift_detector):
        """Test statistical confidence calculation."""
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(
                datetime.now() - timedelta(days=7),
                datetime.now(),
            ),
            ks_p_value=0.01,  # Significant
            mw_p_value=0.02,  # Significant
            sample_size_baseline=200,
            sample_size_current=200,
            drift_types=[DriftType.FEATURE_DRIFT, DriftType.CONCEPT_DRIFT],
        )

        drift_detector._calculate_statistical_confidence(drift_metrics)

        assert 0 <= drift_metrics.statistical_confidence <= 1.0
        assert (
            drift_metrics.statistical_confidence > 0.5
        )  # Should be high due to significant tests


class TestFeatureDriftDetector:
    """Test FeatureDriftDetector functionality."""

    @pytest.fixture
    def feature_detector(self):
        """Create feature drift detector for testing."""
        return FeatureDriftDetector(
            monitor_window_hours=24,  # Shorter for testing
            comparison_window_hours=48,
            min_samples_per_window=10,
            significance_threshold=0.05,
        )

    @pytest.mark.asyncio
    async def test_feature_drift_detection(self, feature_detector):
        """Test feature drift detection with sample data."""
        # Create test data with drift
        timestamps = pd.date_range(
            start=datetime.now() - timedelta(hours=72),
            end=datetime.now(),
            freq="H",
        )

        # Create features with drift in the recent period
        feature_data = pd.DataFrame(
            {
                "timestamp": timestamps,
                "feature1": np.concatenate(
                    [
                        np.random.normal(10, 2, 48),  # Baseline: mean=10
                        np.random.normal(15, 2, 25),  # Recent: mean=15 (drift)
                    ]
                ),
                "feature2": np.concatenate(
                    [
                        np.random.normal(5, 1, 48),  # Baseline: mean=5
                        np.random.normal(5, 1, 25),  # Recent: mean=5 (no drift)
                    ]
                ),
            }
        )

        drift_results = await feature_detector.detect_feature_drift(
            room_id="test_room", feature_data=feature_data
        )

        assert len(drift_results) == 2  # Two features tested

        # Find results for each feature
        feature1_result = next(r for r in drift_results if r.feature_name == "feature1")
        feature2_result = next(r for r in drift_results if r.feature_name == "feature2")

        # Feature1 should show drift, feature2 should not
        assert feature1_result.drift_score > feature2_result.drift_score

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, feature_detector):
        """Test starting and stopping continuous monitoring."""
        room_ids = ["room1", "room2"]

        # Mock the get_recent_feature_data method to avoid database calls
        with patch.object(
            feature_detector, "_get_recent_feature_data", return_value=None
        ):
            # Start monitoring
            await feature_detector.start_monitoring(room_ids)
            assert feature_detector._monitoring_active
            assert feature_detector._monitoring_task is not None

            # Stop monitoring
            await feature_detector.stop_monitoring()
            assert not feature_detector._monitoring_active
            assert feature_detector._monitoring_task is None

    def test_drift_callbacks(self, feature_detector):
        """Test drift notification callbacks."""
        callback_calls = []

        def test_callback(room_id, drift_result):
            callback_calls.append((room_id, drift_result.feature_name))

        # Add callback
        feature_detector.add_drift_callback(test_callback)
        assert test_callback in feature_detector._drift_callbacks

        # Remove callback
        feature_detector.remove_drift_callback(test_callback)
        assert test_callback not in feature_detector._drift_callbacks

    @pytest.mark.asyncio
    async def test_insufficient_data_handling(self, feature_detector):
        """Test handling of insufficient data scenarios."""
        # Create data with too few samples
        feature_data = pd.DataFrame(
            {
                "timestamp": pd.date_range(
                    start=datetime.now() - timedelta(hours=2),
                    periods=5,
                    freq="H",
                ),
                "feature1": [1, 2, 3, 4, 5],
            }
        )

        drift_results = await feature_detector.detect_feature_drift(
            room_id="test_room", feature_data=feature_data
        )

        # Should return empty list due to insufficient samples
        assert len(drift_results) == 0


class TestIntegration:
    """Test integration between drift detection and other components."""

    @pytest.mark.asyncio
    async def test_drift_detector_with_validator_integration(self):
        """Test integration between drift detector and prediction validator."""
        # Create mock components
        drift_detector = ConceptDriftDetector(min_samples=5)

        mock_validator = Mock(spec=PredictionValidator)
        baseline_metrics = Mock(spec=AccuracyMetrics)
        baseline_metrics.mae_minutes = 10.0
        baseline_metrics.confidence_accuracy_correlation = 0.8
        baseline_metrics.recent_records = []

        current_metrics = Mock(spec=AccuracyMetrics)
        current_metrics.mae_minutes = 25.0  # Degraded performance
        current_metrics.confidence_accuracy_correlation = 0.5
        current_metrics.recent_records = []

        mock_validator.get_accuracy_metrics = AsyncMock(
            side_effect=[baseline_metrics, current_metrics]
        )

        # Mock other methods to avoid database calls
        with (
            patch.object(drift_detector, "_get_occupancy_patterns", return_value=None),
            patch.object(
                drift_detector,
                "_get_recent_prediction_errors",
                return_value=[],
            ),
        ):

            drift_metrics = await drift_detector.detect_drift(
                room_id="test_room", prediction_validator=mock_validator
            )

            # Should detect performance degradation
            assert drift_metrics.accuracy_degradation == 15.0
            assert drift_metrics.retraining_recommended
            assert DriftType.PREDICTION_DRIFT in drift_metrics.drift_types

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in drift detection."""
        drift_detector = ConceptDriftDetector()

        # Test with invalid validator
        with pytest.raises(OccupancyPredictionError):
            await drift_detector.detect_drift(
                room_id="test_room", prediction_validator=None
            )


if __name__ == "__main__":
    pytest.main([__file__])
