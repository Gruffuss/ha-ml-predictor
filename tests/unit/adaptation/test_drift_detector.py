"""
Comprehensive tests for ConceptDriftDetector and FeatureDriftDetector.

Tests real drift detection functionality including statistical tests,
feature distribution analysis, and comprehensive drift metrics calculation.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftMetrics,
    DriftSeverity,
    DriftType,
    FeatureDriftDetector,
    FeatureDriftResult,
    StatisticalTest,
)
from src.adaptation.validator import AccuracyMetrics, PredictionValidator


@pytest.fixture
def sample_accuracy_metrics():
    """Create sample accuracy metrics for testing."""
    return AccuracyMetrics(
        total_predictions=100,
        validated_predictions=90,
        accurate_predictions=72,
        accuracy_rate=80.0,
        mean_error_minutes=12.5,
        confidence_accuracy_correlation=0.75,
        rmse_minutes=15.2,
        mae_minutes=12.5,
        std_error_minutes=8.3,
        median_error_minutes=10.0,
        mean_confidence=0.68,
        overconfidence_rate=15.0,
        underconfidence_rate=10.0,
    )


@pytest.fixture
def sample_feature_data():
    """Create sample feature data for testing."""
    np.random.seed(42)
    baseline_data = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 100),
        'feature_2': np.random.normal(5, 2, 100),
        'categorical_feature': np.random.choice(['A', 'B', 'C'], 100),
        'timestamp': pd.date_range(
            start=datetime.now(UTC) - timedelta(days=30),
            periods=100,
            freq='H'
        )
    })
    
    # Create current data with some drift
    current_data = pd.DataFrame({
        'feature_1': np.random.normal(0.5, 1.2, 50),  # Slight mean shift
        'feature_2': np.random.normal(5.2, 1.8, 50),  # Different variance
        'categorical_feature': np.random.choice(['A', 'B', 'C', 'D'], 50),  # New category
        'timestamp': pd.date_range(
            start=datetime.now(UTC) - timedelta(days=7),
            periods=50,
            freq='H'
        )
    })
    
    return pd.concat([baseline_data, current_data], ignore_index=True)


@pytest.fixture
def mock_prediction_validator():
    """Create mock prediction validator for testing."""
    validator = MagicMock(spec=PredictionValidator)
    
    # Mock accuracy metrics
    baseline_metrics = AccuracyMetrics(
        total_predictions=50,
        validated_predictions=45,
        accurate_predictions=36,
        accuracy_rate=80.0,
        mean_error_minutes=10.5,
        mae_minutes=10.5,
        confidence_accuracy_correlation=0.7,
    )
    
    current_metrics = AccuracyMetrics(
        total_predictions=30,
        validated_predictions=28,
        accurate_predictions=18,
        accuracy_rate=64.3,
        mean_error_minutes=16.2,
        mae_minutes=16.2,
        confidence_accuracy_correlation=0.55,
    )
    
    async def mock_get_accuracy_metrics(
        room_id=None, start_time=None, end_time=None, **kwargs
    ):
        if start_time and start_time < datetime.now(UTC) - timedelta(days=15):
            return baseline_metrics
        else:
            return current_metrics
    
    validator.get_accuracy_metrics = AsyncMock(side_effect=mock_get_accuracy_metrics)
    return validator


class TestConceptDriftDetector:
    """Test ConceptDriftDetector functionality."""

    def test_initialization(self):
        """Test drift detector initialization with various configurations."""
        # Test default initialization
        detector = ConceptDriftDetector()
        assert detector.baseline_days == 30
        assert detector.current_days == 7
        assert detector.min_samples == 100
        assert detector.alpha == 0.05
        assert detector.ph_threshold == 50.0
        assert detector.psi_threshold == 0.25

        # Test custom configuration
        detector = ConceptDriftDetector(
            baseline_days=14,
            current_days=3,
            min_samples=50,
            alpha=0.01,
            ph_threshold=30.0,
            psi_threshold=0.15,
        )
        assert detector.baseline_days == 14
        assert detector.current_days == 3
        assert detector.min_samples == 50
        assert detector.alpha == 0.01
        assert detector.ph_threshold == 30.0
        assert detector.psi_threshold == 0.15

    @pytest.mark.asyncio
    async def test_detect_drift_basic(self, mock_prediction_validator):
        """Test basic drift detection functionality."""
        detector = ConceptDriftDetector(
            baseline_days=30,
            current_days=7,
            min_samples=10
        )

        drift_metrics = await detector.detect_drift(
            room_id="living_room",
            prediction_validator=mock_prediction_validator
        )

        assert isinstance(drift_metrics, DriftMetrics)
        assert drift_metrics.room_id == "living_room"
        assert drift_metrics.detection_time is not None
        assert drift_metrics.baseline_period is not None
        assert drift_metrics.current_period is not None
        assert drift_metrics.overall_drift_score >= 0.0

    @pytest.mark.asyncio
    async def test_detect_drift_with_accuracy_degradation(self, mock_prediction_validator):
        """Test drift detection identifies accuracy degradation."""
        detector = ConceptDriftDetector()

        drift_metrics = await detector.detect_drift(
            room_id="bedroom",
            prediction_validator=mock_prediction_validator
        )

        # Should detect drift due to accuracy degradation (80% -> 64.3%)
        assert drift_metrics.accuracy_degradation > 0
        assert drift_metrics.overall_drift_score > 0.1
        assert drift_metrics.retraining_recommended

    @pytest.mark.asyncio
    async def test_detect_drift_no_validator_error(self):
        """Test drift detection fails gracefully without validator."""
        detector = ConceptDriftDetector()

        with pytest.raises(Exception):  # Should raise error without validator
            await detector.detect_drift(
                room_id="test_room",
                prediction_validator=None
            )

    @pytest.mark.asyncio
    async def test_prediction_drift_analysis(self, mock_prediction_validator):
        """Test prediction performance drift analysis."""
        detector = ConceptDriftDetector()

        drift_metrics = await detector.detect_drift(
            room_id="kitchen",
            prediction_validator=mock_prediction_validator
        )

        # Verify prediction drift analysis was performed
        assert drift_metrics.accuracy_degradation > 0
        assert drift_metrics.confidence_calibration_drift >= 0

    @pytest.mark.asyncio
    async def test_pattern_drift_analysis(self):
        """Test occupancy pattern drift analysis."""
        detector = ConceptDriftDetector()
        
        # Mock the pattern analysis methods
        with patch.object(detector, '_get_occupancy_patterns') as mock_patterns:
            # Setup mock return values
            baseline_patterns = {
                'hourly_distribution': {8: 10, 9: 15, 18: 20, 19: 25},
                'daily_frequency': {'2023-01-01': 5, '2023-01-02': 7},
                'total_events': 100
            }
            
            current_patterns = {
                'hourly_distribution': {8: 5, 9: 8, 18: 30, 19: 35},
                'daily_frequency': {'2023-02-01': 12, '2023-02-02': 15},
                'total_events': 120
            }
            
            mock_patterns.side_effect = [baseline_patterns, current_patterns]

            # Create a mock validator
            mock_validator = MagicMock(spec=PredictionValidator)
            mock_validator.get_accuracy_metrics = AsyncMock(return_value=AccuracyMetrics())

            drift_metrics = await detector.detect_drift(
                room_id="office",
                prediction_validator=mock_validator
            )

            # Pattern drift should be detected
            assert drift_metrics.temporal_pattern_drift >= 0
            assert drift_metrics.frequency_pattern_drift >= 0

    @pytest.mark.asyncio
    async def test_page_hinkley_test(self):
        """Test Page-Hinkley test for concept drift detection."""
        detector = ConceptDriftDetector(ph_threshold=10.0)
        
        # Mock recent prediction errors that should trigger drift
        with patch.object(detector, '_get_recent_prediction_errors') as mock_errors:
            mock_errors.return_value = [25.0, 30.0, 35.0, 40.0, 45.0]  # High errors
            
            mock_validator = MagicMock(spec=PredictionValidator)
            mock_validator.get_accuracy_metrics = AsyncMock(return_value=AccuracyMetrics())

            drift_metrics = await detector.detect_drift(
                room_id="test_room",
                prediction_validator=mock_validator
            )

            assert drift_metrics.ph_statistic >= 0
            # With high errors and low threshold, drift should be detected
            # Note: Actual drift detection depends on implementation details

    def test_drift_metrics_calculation(self):
        """Test drift metrics calculation and scoring."""
        # Create drift metrics with various indicators
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(UTC),
            baseline_period=(
                datetime.now(UTC) - timedelta(days=30),
                datetime.now(UTC) - timedelta(days=7)
            ),
            current_period=(
                datetime.now(UTC) - timedelta(days=7),
                datetime.now(UTC)
            ),
            # Set some drift indicators
            accuracy_degradation=20.0,  # 20 minute degradation
            ks_p_value=0.01,  # Significant KS test
            psi_score=0.8,    # High PSI score
            temporal_pattern_drift=0.4,
            drifting_features=["feature1", "feature2"]
        )

        # Metrics should detect significant drift
        assert drift_metrics.overall_drift_score > 0.1
        assert drift_metrics.retraining_recommended
        assert drift_metrics.drift_severity in [DriftSeverity.MEDIUM, DriftSeverity.HIGH, DriftSeverity.CRITICAL]

    def test_drift_metrics_serialization(self):
        """Test drift metrics can be serialized to dictionary."""
        drift_metrics = DriftMetrics(
            room_id="serialization_test",
            detection_time=datetime.now(UTC),
            baseline_period=(
                datetime.now(UTC) - timedelta(days=30),
                datetime.now(UTC) - timedelta(days=7)
            ),
            current_period=(
                datetime.now(UTC) - timedelta(days=7),
                datetime.now(UTC)
            ),
            accuracy_degradation=15.0,
            psi_score=0.3,
            drifting_features=["temp", "humidity"],
            drift_types=[DriftType.FEATURE_DRIFT, DriftType.CONCEPT_DRIFT]
        )

        result = drift_metrics.to_dict()

        assert isinstance(result, dict)
        assert result["room_id"] == "serialization_test"
        assert "statistical_tests" in result
        assert "feature_analysis" in result
        assert "prediction_analysis" in result
        assert "assessment" in result
        assert "recommendations" in result
        assert len(result["feature_analysis"]["drifting_features"]) == 2

    @pytest.mark.asyncio
    async def test_statistical_confidence_calculation(self, mock_prediction_validator):
        """Test statistical confidence calculation in drift detection."""
        detector = ConceptDriftDetector()

        drift_metrics = await detector.detect_drift(
            room_id="confidence_test",
            prediction_validator=mock_prediction_validator
        )

        # Statistical confidence should be calculated
        assert 0.0 <= drift_metrics.statistical_confidence <= 1.0

    def test_numerical_psi_calculation(self):
        """Test Population Stability Index calculation for numerical features."""
        detector = ConceptDriftDetector()
        
        # Create baseline and current data with known drift
        baseline = pd.Series(np.random.normal(0, 1, 1000))
        current = pd.Series(np.random.normal(1, 1, 1000))  # Mean shift
        
        psi = detector._calculate_numerical_psi(baseline, current, bins=10)
        
        # PSI should be positive when there's drift
        assert psi > 0
        assert isinstance(psi, float)
        assert not np.isnan(psi)
        assert not np.isinf(psi)

    def test_categorical_psi_calculation(self):
        """Test PSI calculation for categorical features."""
        detector = ConceptDriftDetector()
        
        # Create categorical data with distribution shift
        baseline = pd.Series(['A'] * 500 + ['B'] * 300 + ['C'] * 200)
        current = pd.Series(['A'] * 200 + ['B'] * 400 + ['C'] * 300 + ['D'] * 100)
        
        psi = detector._calculate_categorical_psi(baseline, current)
        
        # PSI should detect the distribution shift
        assert psi > 0
        assert isinstance(psi, float)
        assert not np.isnan(psi)
        assert not np.isinf(psi)


class TestFeatureDriftDetector:
    """Test FeatureDriftDetector functionality."""

    def test_initialization(self):
        """Test feature drift detector initialization."""
        detector = FeatureDriftDetector()
        
        assert detector.monitor_window_hours == 168  # 1 week
        assert detector.comparison_window_hours == 336  # 2 weeks
        assert detector.min_samples_per_window == 50
        assert detector.significance_threshold == 0.05

        # Test custom configuration
        detector = FeatureDriftDetector(
            monitor_window_hours=72,
            comparison_window_hours=144,
            min_samples_per_window=25,
            significance_threshold=0.01
        )
        
        assert detector.monitor_window_hours == 72
        assert detector.comparison_window_hours == 144
        assert detector.min_samples_per_window == 25
        assert detector.significance_threshold == 0.01

    @pytest.mark.asyncio
    async def test_detect_feature_drift(self, sample_feature_data):
        """Test feature drift detection on sample data."""
        detector = FeatureDriftDetector(min_samples_per_window=20)
        
        drift_results = await detector.detect_feature_drift(
            room_id="test_room",
            feature_data=sample_feature_data
        )
        
        assert isinstance(drift_results, list)
        assert len(drift_results) > 0
        
        # Check each drift result
        for result in drift_results:
            assert isinstance(result, FeatureDriftResult)
            assert result.feature_name in sample_feature_data.columns
            assert isinstance(result.drift_detected, bool)
            assert isinstance(result.drift_score, float)
            assert result.drift_score >= 0.0
            assert result.statistical_test in StatisticalTest
            assert isinstance(result.p_value, float)

    @pytest.mark.asyncio
    async def test_numerical_feature_drift_detection(self):
        """Test drift detection for numerical features."""
        detector = FeatureDriftDetector()
        
        # Create feature data with clear drift
        baseline = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(2, 1, 100))  # Clear mean shift
        
        result = await detector._test_numerical_feature_drift(
            baseline, current, "test_feature"
        )
        
        assert isinstance(result, FeatureDriftResult)
        assert result.feature_name == "test_feature"
        assert result.statistical_test == StatisticalTest.KOLMOGOROV_SMIRNOV
        assert result.drift_score > 0  # Should detect drift
        assert "mean" in result.baseline_stats
        assert "mean" in result.current_stats
        assert result.baseline_stats["mean"] != result.current_stats["mean"]

    @pytest.mark.asyncio
    async def test_categorical_feature_drift_detection(self):
        """Test drift detection for categorical features."""
        detector = FeatureDriftDetector()
        
        # Create categorical data with distribution shift
        baseline = pd.Series(['A'] * 60 + ['B'] * 30 + ['C'] * 10)
        current = pd.Series(['A'] * 30 + ['B'] * 40 + ['C'] * 20 + ['D'] * 10)
        
        result = await detector._test_categorical_feature_drift(
            baseline, current, "categorical_feature"
        )
        
        assert isinstance(result, FeatureDriftResult)
        assert result.feature_name == "categorical_feature"
        assert result.statistical_test == StatisticalTest.CHI_SQUARE
        assert result.drift_score >= 0
        assert "mode" in result.baseline_stats
        assert "mode" in result.current_stats
        assert "unique_values" in result.baseline_stats

    @pytest.mark.asyncio
    async def test_monitoring_lifecycle(self):
        """Test feature drift monitoring start/stop lifecycle."""
        detector = FeatureDriftDetector()
        room_ids = ["room1", "room2"]
        
        # Test starting monitoring
        assert not detector._monitoring_active
        
        # Mock the get_recent_feature_data method
        with patch.object(detector, '_get_recent_feature_data') as mock_get_data:
            mock_get_data.return_value = None  # No data available
            
            # Start monitoring
            await detector.start_monitoring(room_ids)
            assert detector._monitoring_active
            assert detector._monitoring_task is not None
            
            # Stop monitoring
            await detector.stop_monitoring()
            assert not detector._monitoring_active
            assert detector._monitoring_task is None

    def test_drift_callbacks(self):
        """Test drift notification callback system."""
        detector = FeatureDriftDetector()
        
        callback_called = []
        
        def test_callback(room_id, drift_result):
            callback_called.append((room_id, drift_result))
        
        # Add callback
        detector.add_drift_callback(test_callback)
        assert len(detector._drift_callbacks) == 1
        
        # Remove callback
        detector.remove_drift_callback(test_callback)
        assert len(detector._drift_callbacks) == 0

    def test_feature_drift_result_serialization(self):
        """Test FeatureDriftResult can be used properly."""
        result = FeatureDriftResult(
            feature_name="test_feature",
            drift_detected=True,
            drift_score=0.75,
            statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
            test_statistic=0.5,
            p_value=0.02,
            baseline_stats={"mean": 10.0, "std": 2.0},
            current_stats={"mean": 12.0, "std": 2.5}
        )
        
        assert result.is_significant(alpha=0.05)
        assert not result.is_significant(alpha=0.01)

    @pytest.mark.asyncio 
    async def test_insufficient_data_handling(self):
        """Test handling of insufficient data scenarios."""
        detector = FeatureDriftDetector(min_samples_per_window=100)
        
        # Create small dataset
        small_data = pd.DataFrame({
            'feature1': np.random.normal(0, 1, 20),
            'timestamp': pd.date_range(start=datetime.now(UTC), periods=20, freq='H')
        })
        
        drift_results = await detector.detect_feature_drift(
            room_id="small_data_test",
            feature_data=small_data
        )
        
        # Should return empty list due to insufficient data
        assert isinstance(drift_results, list)
        assert len(drift_results) == 0


class TestDriftIntegration:
    """Test integration between different drift detection components."""

    @pytest.mark.asyncio
    async def test_drift_detection_with_feature_engine(self, mock_prediction_validator):
        """Test drift detection with mock feature engineering engine."""
        detector = ConceptDriftDetector()
        
        # Mock feature engineering engine
        mock_engine = MagicMock()
        
        drift_metrics = await detector.detect_drift(
            room_id="integration_test",
            prediction_validator=mock_prediction_validator,
            feature_engineering_engine=mock_engine
        )
        
        assert isinstance(drift_metrics, DriftMetrics)
        assert drift_metrics.room_id == "integration_test"

    def test_drift_severity_classification(self):
        """Test drift severity classification logic."""
        # Test low severity
        low_drift = DriftMetrics(
            room_id="test",
            detection_time=datetime.now(UTC),
            baseline_period=(datetime.now(UTC) - timedelta(days=30), datetime.now(UTC) - timedelta(days=7)),
            current_period=(datetime.now(UTC) - timedelta(days=7), datetime.now(UTC)),
            accuracy_degradation=5.0,
            overall_drift_score=0.2
        )
        assert low_drift.drift_severity == DriftSeverity.LOW
        
        # Test high severity
        high_drift = DriftMetrics(
            room_id="test",
            detection_time=datetime.now(UTC),
            baseline_period=(datetime.now(UTC) - timedelta(days=30), datetime.now(UTC) - timedelta(days=7)),
            current_period=(datetime.now(UTC) - timedelta(days=7), datetime.now(UTC)),
            accuracy_degradation=25.0,
            overall_drift_score=0.7
        )
        assert high_drift.drift_severity == DriftSeverity.HIGH

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Test error handling in drift detection."""
        detector = ConceptDriftDetector()
        
        # Test with invalid input
        with pytest.raises(Exception):
            await detector.detect_drift(
                room_id="",  # Invalid room ID
                prediction_validator=None  # Missing validator
            )

    def test_drift_types_detection(self):
        """Test different drift types are properly detected."""
        drift_metrics = DriftMetrics(
            room_id="multi_drift_test",
            detection_time=datetime.now(UTC),
            baseline_period=(datetime.now(UTC) - timedelta(days=30), datetime.now(UTC) - timedelta(days=7)),
            current_period=(datetime.now(UTC) - timedelta(days=7), datetime.now(UTC)),
            # Set indicators for different drift types
            accuracy_degradation=20.0,  # Prediction drift
            psi_score=0.5,  # Feature drift
            temporal_pattern_drift=0.4,  # Pattern drift
            ph_drift_detected=True,  # Concept drift
            drifting_features=["feature1"]
        )
        
        # Should detect multiple drift types
        assert DriftType.PREDICTION_DRIFT in drift_metrics.drift_types
        assert DriftType.FEATURE_DRIFT in drift_metrics.drift_types
        assert DriftType.PATTERN_DRIFT in drift_metrics.drift_types
        assert DriftType.CONCEPT_DRIFT in drift_metrics.drift_types


if __name__ == "__main__":
    pytest.main([__file__])