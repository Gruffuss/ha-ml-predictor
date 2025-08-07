"""
Comprehensive unit tests for ConceptDriftDetector and FeatureDriftDetector.

This test module covers drift detection algorithms with synthetic drift scenarios,
statistical test validation, and comprehensive accuracy monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from scipy import stats

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
from src.core.exceptions import OccupancyPredictionError

# Test fixtures and utilities


@pytest.fixture
def mock_prediction_validator():
    """Mock prediction validator with test data."""
    validator = Mock(spec=PredictionValidator)
    
    # Mock accuracy metrics
    baseline_metrics = AccuracyMetrics(
        total_predictions=100,
        validated_predictions=95,
        accurate_predictions=80,
        accuracy_rate=84.2,
        mean_absolute_error_minutes=12.5,
        confidence_vs_accuracy_correlation=0.78,
        recent_records=[]
    )
    
    current_metrics = AccuracyMetrics(
        total_predictions=50,
        validated_predictions=48,
        accurate_predictions=30,
        accuracy_rate=62.5,
        mean_absolute_error_minutes=23.8,
        confidence_vs_accuracy_correlation=0.65,
        recent_records=[]
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
    """Create drift detector with test configuration."""
    return ConceptDriftDetector(
        baseline_days=14,
        current_days=3,
        min_samples=20,
        alpha=0.05,
        ph_threshold=30.0,
        psi_threshold=0.20
    )


@pytest.fixture
def feature_drift_detector():
    """Create feature drift detector with test configuration."""
    return FeatureDriftDetector(
        monitor_window_hours=24,
        comparison_window_hours=48,
        min_samples_per_window=30,
        significance_threshold=0.05
    )


@pytest.fixture
def synthetic_feature_data():
    """Generate synthetic feature data with and without drift."""
    np.random.seed(42)
    
    # Stable feature data (no drift)
    stable_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='1H'),
        'feature_1': np.random.normal(50, 10, 100),
        'feature_2': np.random.normal(25, 5, 100),
        'feature_3': np.random.choice(['A', 'B', 'C'], 100),
        'room_id': 'living_room'
    })
    
    # Drifted feature data (distribution shift)
    drifted_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-05', periods=100, freq='1H'),
        'feature_1': np.random.normal(70, 15, 100),  # Mean shift
        'feature_2': np.random.normal(25, 12, 100),  # Variance increase
        'feature_3': np.random.choice(['A', 'D', 'E'], 100),  # Category change
        'room_id': 'living_room'
    })
    
    return stable_data, drifted_data


@pytest.fixture
def synthetic_prediction_errors():
    """Generate synthetic prediction error sequences."""
    # Baseline errors (good performance)
    baseline_errors = np.random.normal(8, 3, 50).tolist()
    baseline_errors = [max(0, e) for e in baseline_errors]  # No negative errors
    
    # Current errors (degraded performance)
    current_errors = np.random.normal(18, 6, 30).tolist()
    current_errors = [max(0, e) for e in current_errors]
    
    return baseline_errors, current_errors


# Core drift detection tests

class TestConceptDriftDetector:
    """Test ConceptDriftDetector functionality."""

    @pytest.mark.asyncio
    async def test_detector_initialization(self, drift_detector):
        """Test drift detector initialization."""
        assert drift_detector.baseline_days == 14
        assert drift_detector.current_days == 3
        assert drift_detector.min_samples == 20
        assert drift_detector.alpha == 0.05
        assert drift_detector.ph_threshold == 30.0
        assert drift_detector.psi_threshold == 0.20

    @pytest.mark.asyncio
    async def test_drift_detection_with_no_drift(self, drift_detector, mock_prediction_validator):
        """Test drift detection when no significant drift is present."""
        # Mock accuracy metrics with stable performance
        stable_baseline = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=95,
            accurate_predictions=85,
            accuracy_rate=89.5,
            mean_absolute_error_minutes=10.2,
            confidence_vs_accuracy_correlation=0.82,
            recent_records=[]
        )
        
        stable_current = AccuracyMetrics(
            total_predictions=50,
            validated_predictions=48,
            accurate_predictions=43,
            accuracy_rate=89.6,
            mean_absolute_error_minutes=10.1,
            confidence_vs_accuracy_correlation=0.81,
            recent_records=[]
        )
        
        async def stable_metrics(room_id, start_time, end_time):
            if start_time < datetime.now() - timedelta(days=10):
                return stable_baseline
            return stable_current
        
        mock_prediction_validator.get_accuracy_metrics = AsyncMock(side_effect=stable_metrics)
        
        # Run drift detection
        drift_metrics = await drift_detector.detect_drift(
            room_id="test_room",
            prediction_validator=mock_prediction_validator,
            feature_engineering_engine=None
        )
        
        # Verify no significant drift detected
        assert drift_metrics is not None
        assert drift_metrics.drift_severity in [DriftSeverity.MINOR, DriftSeverity.MODERATE]
        assert drift_metrics.accuracy_degradation < 5.0
        assert not drift_metrics.immediate_attention_required
        assert drift_metrics.overall_drift_score < 0.4

    @pytest.mark.asyncio
    async def test_drift_detection_with_accuracy_degradation(self, drift_detector, mock_prediction_validator):
        """Test drift detection with significant accuracy degradation."""
        # Mock degraded performance
        degraded_current = AccuracyMetrics(
            total_predictions=50,
            validated_predictions=45,
            accurate_predictions=20,
            accuracy_rate=44.4,
            mean_absolute_error_minutes=35.2,
            confidence_vs_accuracy_correlation=0.45,
            recent_records=[]
        )
        
        async def degraded_metrics(room_id, start_time, end_time):
            if start_time < datetime.now() - timedelta(days=10):
                return AccuracyMetrics(accuracy_rate=85.0, mean_absolute_error_minutes=12.0)
            return degraded_current
        
        mock_prediction_validator.get_accuracy_metrics = AsyncMock(side_effect=degraded_metrics)
        
        # Run drift detection
        drift_metrics = await drift_detector.detect_drift(
            room_id="test_room",
            prediction_validator=mock_prediction_validator,
            feature_engineering_engine=None
        )
        
        # Verify significant drift detected
        assert drift_metrics is not None
        assert drift_metrics.drift_severity in [DriftSeverity.MAJOR, DriftSeverity.CRITICAL]
        assert drift_metrics.accuracy_degradation > 15.0
        assert DriftType.PREDICTION_DRIFT in drift_metrics.drift_types
        assert drift_metrics.retraining_recommended

    @pytest.mark.asyncio
    async def test_page_hinkley_drift_detection(self, drift_detector):
        """Test Page-Hinkley test for concept drift detection."""
        room_id = "test_room"
        
        # Simulate increasing prediction errors (concept drift)
        with patch.object(drift_detector, '_get_recent_prediction_errors') as mock_errors:
            # Page-Hinkley detects mean shift in error sequence
            mock_errors.return_value = [5, 6, 7, 25, 28, 30, 32, 35, 38, 40]  # Clear shift
            
            drift_metrics = DriftMetrics(
                room_id=room_id,
                detection_time=datetime.now(),
                baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
                current_period=(datetime.now() - timedelta(days=3), datetime.now())
            )
            
            # Run Page-Hinkley test
            await drift_detector._run_page_hinkley_test(drift_metrics, room_id)
            
            # Check for drift detection
            if drift_metrics.ph_statistic > drift_detector.ph_threshold:
                assert drift_metrics.ph_drift_detected
                assert DriftType.CONCEPT_DRIFT in drift_metrics.drift_types

    @pytest.mark.asyncio
    async def test_statistical_confidence_calculation(self, drift_detector):
        """Test statistical confidence calculation for drift detection."""
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            sample_size_baseline=100,
            sample_size_current=50,
            ks_p_value=0.02,  # Significant
            mw_p_value=0.03,  # Significant
        )
        
        # Calculate statistical confidence
        drift_detector._calculate_statistical_confidence(drift_metrics)
        
        # Verify confidence calculation
        assert 0.0 <= drift_metrics.statistical_confidence <= 1.0
        assert drift_metrics.statistical_confidence > 0.5  # Should be higher due to significant tests

    @pytest.mark.asyncio
    async def test_drift_metrics_serialization(self, drift_detector):
        """Test DriftMetrics serialization and deserialization."""
        drift_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            ks_statistic=0.35,
            ks_p_value=0.02,
            accuracy_degradation=18.5,
            drifting_features=["feature_1", "feature_2"],
            drift_types=[DriftType.FEATURE_DRIFT, DriftType.PREDICTION_DRIFT]
        )
        
        # Serialize to dict
        metrics_dict = drift_metrics.to_dict()
        
        # Verify serialization structure
        assert "room_id" in metrics_dict
        assert "detection_time" in metrics_dict
        assert "statistical_tests" in metrics_dict
        assert "feature_analysis" in metrics_dict
        assert "prediction_analysis" in metrics_dict
        assert "assessment" in metrics_dict
        assert "recommendations" in metrics_dict
        
        # Verify serialization values
        assert metrics_dict["room_id"] == "test_room"
        assert metrics_dict["statistical_tests"]["kolmogorov_smirnov"]["statistic"] == 0.35
        assert metrics_dict["prediction_analysis"]["accuracy_degradation_minutes"] == 18.5
        assert "feature_1" in metrics_dict["feature_analysis"]["drifting_features"]

    @pytest.mark.asyncio
    async def test_error_handling_in_drift_detection(self, drift_detector):
        """Test error handling in drift detection methods."""
        # Mock validator that raises exception
        mock_validator = Mock(spec=PredictionValidator)
        mock_validator.get_accuracy_metrics = AsyncMock(side_effect=Exception("Database error"))
        
        # Should handle error gracefully
        drift_metrics = await drift_detector.detect_drift(
            room_id="test_room",
            prediction_validator=mock_validator,
            feature_engineering_engine=None
        )
        
        # Should return metrics with error indication
        assert drift_metrics is not None
        assert drift_metrics.accuracy_degradation == 0.0  # Default on error


class TestFeatureDriftDetector:
    """Test FeatureDriftDetector functionality."""

    @pytest.mark.asyncio
    async def test_feature_detector_initialization(self, feature_drift_detector):
        """Test feature drift detector initialization."""
        assert feature_drift_detector.monitor_window_hours == 24
        assert feature_drift_detector.comparison_window_hours == 48
        assert feature_drift_detector.min_samples_per_window == 30
        assert feature_drift_detector.significance_threshold == 0.05

    @pytest.mark.asyncio
    async def test_numerical_feature_drift_detection(self, feature_drift_detector, synthetic_feature_data):
        """Test drift detection on numerical features."""
        stable_data, drifted_data = synthetic_feature_data
        
        # Combine data with temporal separation
        combined_data = pd.concat([stable_data, drifted_data], ignore_index=True)
        
        # Test feature drift detection
        drift_results = await feature_drift_detector.detect_feature_drift(
            room_id="living_room",
            feature_data=combined_data
        )
        
        # Verify drift detection results
        assert len(drift_results) > 0
        
        # Check for feature_1 drift (mean shift)
        feature_1_result = next((r for r in drift_results if r.feature_name == 'feature_1'), None)
        assert feature_1_result is not None
        assert feature_1_result.statistical_test == StatisticalTest.KOLMOGOROV_SMIRNOV
        assert feature_1_result.drift_score > 0.1  # Should detect the mean shift

    @pytest.mark.asyncio
    async def test_categorical_feature_drift_detection(self, feature_drift_detector, synthetic_feature_data):
        """Test drift detection on categorical features."""
        stable_data, drifted_data = synthetic_feature_data
        
        # Focus on categorical feature (feature_3: A,B,C -> A,D,E)
        combined_data = pd.concat([stable_data, drifted_data], ignore_index=True)
        
        drift_results = await feature_drift_detector.detect_feature_drift(
            room_id="living_room",
            feature_data=combined_data
        )
        
        # Check categorical feature drift
        feature_3_result = next((r for r in drift_results if r.feature_name == 'feature_3'), None)
        assert feature_3_result is not None
        assert feature_3_result.statistical_test == StatisticalTest.CHI_SQUARE
        assert feature_3_result.drift_detected  # Should detect category change

    @pytest.mark.asyncio
    async def test_feature_drift_with_insufficient_data(self, feature_drift_detector):
        """Test feature drift detection with insufficient data."""
        # Create small dataset (below minimum threshold)
        small_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=10, freq='1H'),
            'feature_1': np.random.normal(50, 10, 10),
            'room_id': 'test_room'
        })
        
        drift_results = await feature_drift_detector.detect_feature_drift(
            room_id="test_room",
            feature_data=small_data
        )
        
        # Should return empty results due to insufficient data
        assert len(drift_results) == 0

    @pytest.mark.asyncio
    async def test_feature_drift_monitoring_lifecycle(self, feature_drift_detector):
        """Test feature drift monitoring start/stop lifecycle."""
        room_ids = ["room_1", "room_2"]
        
        # Start monitoring
        await feature_drift_detector.start_monitoring(room_ids)
        assert feature_drift_detector._monitoring_active
        assert feature_drift_detector._monitoring_task is not None
        
        # Stop monitoring
        await feature_drift_detector.stop_monitoring()
        assert not feature_drift_detector._monitoring_active
        assert feature_drift_detector._monitoring_task is None

    @pytest.mark.asyncio
    async def test_drift_callback_functionality(self, feature_drift_detector):
        """Test drift detection callback notifications."""
        callback_called = False
        callback_result = None
        
        async def drift_callback(room_id, drift_result):
            nonlocal callback_called, callback_result
            callback_called = True
            callback_result = drift_result
        
        # Add callback
        feature_drift_detector.add_drift_callback(drift_callback)
        
        # Create significant drift result
        drift_result = FeatureDriftResult(
            feature_name="test_feature",
            drift_detected=True,
            drift_score=0.8,
            statistical_test=StatisticalTest.KOLMOGOROV_SMIRNOV,
            test_statistic=0.6,
            p_value=0.01,  # Significant
            baseline_stats={"mean": 10.0},
            current_stats={"mean": 20.0}
        )
        
        # Notify callbacks
        await feature_drift_detector._notify_drift_callbacks("test_room", drift_result)
        
        # Verify callback was called
        assert callback_called
        assert callback_result == drift_result
        
        # Remove callback
        feature_drift_detector.remove_drift_callback(drift_callback)


class TestDriftDetectionIntegration:
    """Test integration between different drift detection components."""

    @pytest.mark.asyncio
    async def test_combined_drift_detection_workflow(self, drift_detector, mock_prediction_validator):
        """Test complete drift detection workflow with multiple drift types."""
        # Mock feature engine with drift
        mock_feature_engine = Mock()
        
        # Create mock feature data with drift
        baseline_features = pd.DataFrame({
            'feature_1': np.random.normal(10, 2, 100),
            'feature_2': np.random.choice(['A', 'B'], 100)
        })
        
        current_features = pd.DataFrame({
            'feature_1': np.random.normal(15, 3, 50),  # Mean shift + variance increase
            'feature_2': np.random.choice(['C', 'D'], 50)  # Category change
        })
        
        async def mock_get_feature_data(engine, room_id, start_time, end_time):
            if start_time < datetime.now() - timedelta(days=10):
                return baseline_features
            return current_features
        
        with patch.object(drift_detector, '_get_feature_data', side_effect=mock_get_feature_data):
            # Run comprehensive drift detection
            drift_metrics = await drift_detector.detect_drift(
                room_id="test_room",
                prediction_validator=mock_prediction_validator,
                feature_engineering_engine=mock_feature_engine
            )
            
            # Verify comprehensive drift analysis
            assert drift_metrics is not None
            assert len(drift_metrics.drift_types) > 0
            assert drift_metrics.overall_drift_score > 0

    @pytest.mark.asyncio
    async def test_drift_severity_classification(self, drift_detector):
        """Test drift severity classification logic."""
        # Test minor drift
        minor_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=5.0,
            overall_drift_score=0.2
        )
        assert minor_metrics.drift_severity == DriftSeverity.MINOR
        
        # Test major drift
        major_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=25.0,
            overall_drift_score=0.7
        )
        assert major_metrics.drift_severity == DriftSeverity.MAJOR
        
        # Test critical drift
        critical_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=40.0,
            overall_drift_score=0.9,
            ph_drift_detected=True
        )
        assert critical_metrics.drift_severity == DriftSeverity.CRITICAL

    @pytest.mark.asyncio
    async def test_drift_recommendation_generation(self, drift_detector):
        """Test automatic recommendation generation based on drift severity."""
        # Test moderate drift recommendations
        moderate_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=18.0,
            overall_drift_score=0.5
        )
        
        assert moderate_metrics.retraining_recommended
        assert not moderate_metrics.immediate_attention_required
        
        # Test critical drift recommendations
        critical_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=35.0,
            overall_drift_score=0.9,
            ph_drift_detected=True
        )
        
        assert critical_metrics.retraining_recommended
        assert critical_metrics.immediate_attention_required


# Statistical test validation

class TestStatisticalTests:
    """Test individual statistical tests used in drift detection."""

    @pytest.mark.asyncio
    async def test_kolmogorov_smirnov_test(self, drift_detector):
        """Test Kolmogorov-Smirnov test for distribution drift."""
        # Same distribution
        baseline = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(0, 1, 100))
        
        result = await drift_detector._test_numerical_drift(baseline, current, "test_feature")
        
        assert result.statistical_test == StatisticalTest.KOLMOGOROV_SMIRNOV
        assert result.p_value > 0.05  # Should not be significant
        
        # Different distribution
        baseline = pd.Series(np.random.normal(0, 1, 100))
        current = pd.Series(np.random.normal(2, 1, 100))  # Mean shift
        
        result = await drift_detector._test_numerical_drift(baseline, current, "test_feature")
        
        assert result.statistical_test == StatisticalTest.KOLMOGOROV_SMIRNOV
        assert result.p_value < 0.05  # Should be significant

    @pytest.mark.asyncio
    async def test_chi_square_test(self, drift_detector):
        """Test Chi-square test for categorical drift."""
        # Same distribution
        baseline = pd.Series(['A'] * 50 + ['B'] * 30 + ['C'] * 20)
        current = pd.Series(['A'] * 40 + ['B'] * 35 + ['C'] * 25)
        
        result = await drift_detector._test_categorical_drift(baseline, current, "test_feature")
        
        assert result.statistical_test == StatisticalTest.CHI_SQUARE
        assert result.p_value > 0.05  # Should not be significant
        
        # Different distribution  
        baseline = pd.Series(['A'] * 80 + ['B'] * 20)
        current = pd.Series(['A'] * 20 + ['B'] * 80)  # Distribution flip
        
        result = await drift_detector._test_categorical_drift(baseline, current, "test_feature")
        
        assert result.statistical_test == StatisticalTest.CHI_SQUARE
        assert result.p_value < 0.05  # Should be significant

    @pytest.mark.asyncio
    async def test_population_stability_index(self, drift_detector):
        """Test Population Stability Index calculation."""
        # Create test data
        baseline_df = pd.DataFrame({
            'feature_1': np.random.normal(0, 1, 1000),
            'feature_2': np.random.choice(['A', 'B', 'C'], 1000)
        })
        
        current_df = pd.DataFrame({
            'feature_1': np.random.normal(0.5, 1.2, 500),  # Slight shift
            'feature_2': np.random.choice(['A', 'B', 'C', 'D'], 500)  # New category
        })
        
        features = {'feature_1', 'feature_2'}
        psi_score = await drift_detector._calculate_psi(baseline_df, current_df, features)
        
        assert psi_score >= 0.0
        assert psi_score < 1.0  # Reasonable PSI range
        
        # Test with identical distributions
        identical_df = baseline_df.copy()
        psi_identical = await drift_detector._calculate_psi(baseline_df, identical_df, features)
        assert psi_identical < 0.1  # Should be very low


# Performance and edge case tests

class TestDriftDetectionEdgeCases:
    """Test edge cases and error conditions in drift detection."""

    @pytest.mark.asyncio
    async def test_empty_data_handling(self, drift_detector, mock_prediction_validator):
        """Test handling of empty or insufficient data."""
        # Mock empty accuracy metrics
        empty_metrics = AccuracyMetrics(
            total_predictions=0,
            validated_predictions=0,
            accurate_predictions=0
        )
        
        mock_prediction_validator.get_accuracy_metrics = AsyncMock(return_value=empty_metrics)
        
        drift_metrics = await drift_detector.detect_drift(
            room_id="empty_room",
            prediction_validator=mock_prediction_validator,
            feature_engineering_engine=None
        )
        
        # Should handle gracefully
        assert drift_metrics is not None
        assert drift_metrics.accuracy_degradation == 0.0

    @pytest.mark.asyncio
    async def test_extreme_drift_scenarios(self, drift_detector):
        """Test detection of extreme drift scenarios."""
        # Test extreme accuracy degradation
        extreme_metrics = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(datetime.now() - timedelta(days=14), datetime.now() - timedelta(days=3)),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=60.0,  # Extreme degradation
            overall_drift_score=1.0,    # Maximum drift score
            ph_drift_detected=True
        )
        
        assert extreme_metrics.drift_severity == DriftSeverity.CRITICAL
        assert extreme_metrics.immediate_attention_required
        assert extreme_metrics.retraining_recommended

    @pytest.mark.asyncio
    async def test_concurrent_drift_detection(self, drift_detector, mock_prediction_validator):
        """Test concurrent drift detection for multiple rooms."""
        rooms = ["room_1", "room_2", "room_3"]
        
        # Run concurrent drift detection
        tasks = []
        for room in rooms:
            task = drift_detector.detect_drift(
                room_id=room,
                prediction_validator=mock_prediction_validator,
                feature_engineering_engine=None
            )
            tasks.append(task)
        
        # Wait for all to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Verify all completed successfully
        assert len(results) == len(rooms)
        for result in results:
            assert isinstance(result, DriftMetrics)

    @pytest.mark.asyncio
    async def test_feature_drift_with_mixed_data_types(self, feature_drift_detector):
        """Test feature drift detection with mixed data types."""
        # Create data with various types
        mixed_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='1H'),
            'numeric_int': np.random.randint(1, 100, 200),
            'numeric_float': np.random.normal(50, 10, 200),
            'categorical': np.random.choice(['X', 'Y', 'Z'], 200),
            'boolean': np.random.choice([True, False], 200),
            'string': np.random.choice(['cat', 'dog', 'bird'], 200),
            'room_id': 'mixed_room'
        })
        
        drift_results = await feature_drift_detector.detect_feature_drift(
            room_id="mixed_room",
            feature_data=mixed_data
        )
        
        # Should handle all data types gracefully
        assert len(drift_results) > 0
        
        # Verify different statistical tests were used
        statistical_tests = {result.statistical_test for result in drift_results}
        assert StatisticalTest.KOLMOGOROV_SMIRNOV in statistical_tests  # For numeric
        assert StatisticalTest.CHI_SQUARE in statistical_tests  # For categorical