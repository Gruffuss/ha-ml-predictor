"""
Comprehensive tests for confidence calibration validation algorithms and metrics.

This test suite validates confidence interval accuracy, calibration score calculation,
reliability diagrams, and confidence-based prediction filtering used in the
occupancy prediction system's validation framework.
"""

import asyncio
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.validator import (
    AccuracyLevel,
    AccuracyMetrics,
    PredictionValidator,
    ValidationError,
    ValidationRecord,
    ValidationStatus,
)
from src.core.constants import ModelType
from src.models.base.predictor import PredictionResult


@dataclass
class CalibrationTestCase:
    """Test case for confidence calibration validation."""

    name: str
    predictions: List[
        Tuple[datetime, float, datetime]
    ]  # (predicted_time, confidence, actual_time)
    expected_calibration_score: float
    expected_reliability_bins: int
    confidence_threshold: float = 0.8


class TestConfidenceCalibrationMetrics:
    """Test confidence calibration calculation algorithms."""

    @pytest.fixture
    def validator(self):
        """Create validator with test configuration."""
        return PredictionValidator(
            accuracy_threshold_minutes=15, confidence_threshold=0.8
        )

    @pytest.fixture
    def well_calibrated_predictions(self):
        """Create well-calibrated predictions for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        predictions = []

        # High confidence (0.9) predictions should be accurate
        for i in range(10):
            predicted = base_time + timedelta(minutes=i * 30)
            actual = predicted + timedelta(minutes=2)  # Very accurate
            predictions.append((predicted, 0.9, actual))

        # Medium confidence (0.7) predictions should be moderately accurate
        for i in range(10):
            predicted = base_time + timedelta(minutes=(i + 10) * 30)
            actual = predicted + timedelta(minutes=8)  # Moderately accurate
            predictions.append((predicted, 0.7, actual))

        # Low confidence (0.5) predictions should be less accurate
        for i in range(10):
            predicted = base_time + timedelta(minutes=(i + 20) * 30)
            actual = predicted + timedelta(minutes=20)  # Less accurate
            predictions.append((predicted, 0.5, actual))

        return predictions

    @pytest.fixture
    def poorly_calibrated_predictions(self):
        """Create poorly-calibrated predictions for testing."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        predictions = []

        # High confidence (0.9) predictions that are actually inaccurate
        for i in range(10):
            predicted = base_time + timedelta(minutes=i * 30)
            actual = predicted + timedelta(
                minutes=25
            )  # Poor accuracy despite high confidence
            predictions.append((predicted, 0.9, actual))

        # Low confidence (0.4) predictions that are actually very accurate
        for i in range(10):
            predicted = base_time + timedelta(minutes=(i + 10) * 30)
            actual = predicted + timedelta(
                minutes=1
            )  # Excellent accuracy despite low confidence
            predictions.append((predicted, 0.4, actual))

        return predictions

    def test_calibration_score_well_calibrated(
        self, validator, well_calibrated_predictions
    ):
        """Test calibration score calculation with well-calibrated predictions."""
        # Add validation records
        for predicted_time, confidence, actual_time in well_calibrated_predictions:
            record = ValidationRecord(
                room_id="living_room",
                predicted_time=predicted_time,
                confidence=confidence,
                timestamp=predicted_time - timedelta(minutes=5),
            )
            validator._pending_validations["living_room"].append(record)

            # Validate the prediction
            validator.validate_prediction("living_room", actual_time)

        # Calculate calibration metrics
        metrics = validator.get_calibration_metrics("living_room", window_hours=24)

        assert metrics is not None
        assert "calibration_score" in metrics
        assert "reliability_bins" in metrics
        assert "confidence_accuracy_correlation" in metrics

        # Well-calibrated predictions should have high calibration score
        assert metrics["calibration_score"] >= 0.8
        assert metrics["confidence_accuracy_correlation"] >= 0.7

    def test_calibration_score_poorly_calibrated(
        self, validator, poorly_calibrated_predictions
    ):
        """Test calibration score calculation with poorly-calibrated predictions."""
        # Add validation records
        for predicted_time, confidence, actual_time in poorly_calibrated_predictions:
            record = ValidationRecord(
                room_id="kitchen",
                predicted_time=predicted_time,
                confidence=confidence,
                timestamp=predicted_time - timedelta(minutes=5),
            )
            validator._pending_validations["kitchen"].append(record)

            # Validate the prediction
            validator.validate_prediction("kitchen", actual_time)

        # Calculate calibration metrics
        metrics = validator.get_calibration_metrics("kitchen", window_hours=24)

        assert metrics is not None
        assert "calibration_score" in metrics

        # Poorly-calibrated predictions should have low calibration score
        assert metrics["calibration_score"] <= 0.5
        assert metrics["confidence_accuracy_correlation"] <= 0.3

    def test_reliability_diagram_binning(self, validator):
        """Test reliability diagram bin creation and statistics."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Create predictions across different confidence levels
        confidence_levels = [0.1, 0.3, 0.5, 0.7, 0.9]
        for conf_level in confidence_levels:
            for i in range(5):  # 5 predictions per confidence level
                predicted = base_time + timedelta(minutes=i * 10)
                # Accuracy should roughly match confidence for testing
                error_minutes = (
                    1.0 - conf_level
                ) * 20  # Lower confidence = higher error
                actual = predicted + timedelta(minutes=error_minutes)

                record = ValidationRecord(
                    room_id="bedroom",
                    predicted_time=predicted,
                    confidence=conf_level,
                    timestamp=predicted - timedelta(minutes=2),
                )
                validator._pending_validations["bedroom"].append(record)
                validator.validate_prediction("bedroom", actual)

        # Get reliability diagram data
        reliability_data = validator.get_reliability_diagram("bedroom", bins=5)

        assert reliability_data is not None
        assert "bin_centers" in reliability_data
        assert "observed_frequencies" in reliability_data
        assert "predicted_probabilities" in reliability_data
        assert "bin_counts" in reliability_data

        assert len(reliability_data["bin_centers"]) == 5
        assert len(reliability_data["observed_frequencies"]) == 5
        assert len(reliability_data["predicted_probabilities"]) == 5
        assert len(reliability_data["bin_counts"]) == 5

    def test_confidence_threshold_filtering(self, validator):
        """Test prediction filtering based on confidence thresholds."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Add predictions with varying confidence levels
        confidences = [0.3, 0.6, 0.8, 0.9, 0.95]
        for i, conf in enumerate(confidences):
            predicted = base_time + timedelta(minutes=i * 30)
            actual = predicted + timedelta(minutes=5)  # All reasonably accurate

            record = ValidationRecord(
                room_id="office",
                predicted_time=predicted,
                confidence=conf,
                timestamp=predicted - timedelta(minutes=2),
            )
            validator._pending_validations["office"].append(record)
            validator.validate_prediction("office", actual)

        # Test filtering with different thresholds
        all_records = validator.get_validation_history("office", window_hours=24)
        high_conf_records = validator.get_high_confidence_predictions(
            "office", confidence_threshold=0.8, window_hours=24
        )
        very_high_conf_records = validator.get_high_confidence_predictions(
            "office", confidence_threshold=0.9, window_hours=24
        )

        assert len(all_records) == 5
        assert len(high_conf_records) == 3  # 0.8, 0.9, 0.95
        assert len(very_high_conf_records) == 2  # 0.9, 0.95

        # Verify confidence levels
        for record in high_conf_records:
            assert record.confidence >= 0.8
        for record in very_high_conf_records:
            assert record.confidence >= 0.9


class TestConfidenceIntervalValidation:
    """Test confidence interval accuracy and coverage validation."""

    @pytest.fixture
    def validator_with_intervals(self):
        """Create validator configured for interval testing."""
        return PredictionValidator(
            accuracy_threshold_minutes=15,
            confidence_threshold=0.7,
            track_prediction_intervals=True,
        )

    def test_prediction_interval_coverage_90_percent(self, validator_with_intervals):
        """Test 90% prediction interval coverage calculation."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Add predictions with 90% intervals
        for i in range(100):
            predicted = base_time + timedelta(minutes=i * 15)
            lower_bound = predicted - timedelta(minutes=10)
            upper_bound = predicted + timedelta(minutes=10)

            # 90 out of 100 predictions should fall within interval
            if i < 90:
                actual = predicted + timedelta(minutes=5)  # Within interval
            else:
                actual = predicted + timedelta(minutes=15)  # Outside interval

            record = ValidationRecord(
                room_id="living_room",
                predicted_time=predicted,
                confidence=0.9,
                prediction_interval=(lower_bound, upper_bound),
                timestamp=predicted - timedelta(minutes=2),
            )
            validator_with_intervals._pending_validations["living_room"].append(record)
            validator_with_intervals.validate_prediction("living_room", actual)

        # Check interval coverage
        coverage_metrics = validator_with_intervals.get_interval_coverage_metrics(
            "living_room", window_hours=24
        )

        assert coverage_metrics is not None
        assert "coverage_percentage" in coverage_metrics
        assert "expected_coverage" in coverage_metrics
        assert "coverage_error" in coverage_metrics

        # Should be close to 90% coverage
        assert abs(coverage_metrics["coverage_percentage"] - 90.0) <= 5.0
        assert coverage_metrics["expected_coverage"] == 90.0

    def test_prediction_interval_width_analysis(self, validator_with_intervals):
        """Test prediction interval width analysis and optimization."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Add predictions with varying interval widths
        interval_widths = [5, 10, 15, 20, 25]  # minutes
        for i, width in enumerate(interval_widths):
            for j in range(10):  # 10 predictions per width
                predicted = base_time + timedelta(minutes=(i * 10 + j) * 15)
                lower_bound = predicted - timedelta(minutes=width // 2)
                upper_bound = predicted + timedelta(minutes=width // 2)
                actual = predicted + timedelta(minutes=2)  # Accurate prediction

                record = ValidationRecord(
                    room_id="kitchen",
                    predicted_time=predicted,
                    confidence=0.8,
                    prediction_interval=(lower_bound, upper_bound),
                    timestamp=predicted - timedelta(minutes=2),
                )
                validator_with_intervals._pending_validations["kitchen"].append(record)
                validator_with_intervals.validate_prediction("kitchen", actual)

        # Analyze interval widths
        width_analysis = validator_with_intervals.get_interval_width_analysis(
            "kitchen", window_hours=24
        )

        assert width_analysis is not None
        assert "mean_width_minutes" in width_analysis
        assert "median_width_minutes" in width_analysis
        assert "width_distribution" in width_analysis
        assert "optimal_width_recommendation" in width_analysis

        # Mean width should be around 15 minutes (middle of our range)
        assert 10 <= width_analysis["mean_width_minutes"] <= 20

    def test_adaptive_confidence_calibration(self, validator_with_intervals):
        """Test adaptive confidence threshold calibration."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Simulate scenario where model confidence is systematically off
        for i in range(50):
            predicted = base_time + timedelta(minutes=i * 20)
            # Model reports high confidence but predictions are only moderately accurate
            confidence = 0.9
            actual = predicted + timedelta(
                minutes=12
            )  # ~12 min error (moderate accuracy)

            record = ValidationRecord(
                room_id="bedroom",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=5),
            )
            validator_with_intervals._pending_validations["bedroom"].append(record)
            validator_with_intervals.validate_prediction("bedroom", actual)

        # Get calibration recommendations
        calibration_recommendation = (
            validator_with_intervals.get_confidence_calibration_recommendation(
                "bedroom", window_hours=24
            )
        )

        assert calibration_recommendation is not None
        assert "current_calibration_score" in calibration_recommendation
        assert "recommended_confidence_adjustment" in calibration_recommendation
        assert "calibration_status" in calibration_recommendation

        # Should recommend lowering confidence due to poor calibration
        assert calibration_recommendation["recommended_confidence_adjustment"] < 0
        assert calibration_recommendation["calibration_status"] == "poorly_calibrated"


class TestConfidenceBasedDecisionMaking:
    """Test confidence-based prediction filtering and decision making."""

    @pytest.fixture
    def multi_confidence_validator(self):
        """Create validator with multiple confidence scenarios."""
        validator = PredictionValidator(
            accuracy_threshold_minutes=15, confidence_threshold=0.75
        )
        return validator

    def test_confidence_based_prediction_acceptance(self, multi_confidence_validator):
        """Test prediction acceptance based on confidence levels."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Test different confidence scenarios
        test_cases = [
            (0.9, True, "high_confidence_should_accept"),
            (0.8, True, "above_threshold_should_accept"),
            (0.75, True, "at_threshold_should_accept"),
            (0.7, False, "below_threshold_should_reject"),
            (0.5, False, "low_confidence_should_reject"),
        ]

        for i, (confidence, should_accept, description) in enumerate(test_cases):
            predicted = base_time + timedelta(minutes=i * 30)

            # Test acceptance decision
            should_accept_prediction = (
                multi_confidence_validator.should_accept_prediction(
                    confidence=confidence, room_id="test_room"
                )
            )

            assert should_accept_prediction == should_accept, f"Failed {description}"

    def test_confidence_weighted_accuracy_metrics(self, multi_confidence_validator):
        """Test confidence-weighted accuracy metric calculations."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Add predictions with different confidence levels and accuracies
        prediction_data = [
            (0.95, 2),  # High confidence, high accuracy
            (0.90, 3),  # High confidence, high accuracy
            (0.85, 8),  # High confidence, moderate accuracy
            (0.70, 5),  # Medium confidence, good accuracy
            (0.60, 12),  # Medium confidence, moderate accuracy
            (0.40, 18),  # Low confidence, poor accuracy
        ]

        for i, (confidence, error_minutes) in enumerate(prediction_data):
            predicted = base_time + timedelta(minutes=i * 30)
            actual = predicted + timedelta(minutes=error_minutes)

            record = ValidationRecord(
                room_id="weighted_test",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=2),
            )
            multi_confidence_validator._pending_validations["weighted_test"].append(
                record
            )
            multi_confidence_validator.validate_prediction("weighted_test", actual)

        # Calculate weighted metrics
        weighted_metrics = multi_confidence_validator.get_confidence_weighted_metrics(
            "weighted_test", window_hours=24
        )

        assert weighted_metrics is not None
        assert "weighted_mean_error" in weighted_metrics
        assert "weighted_accuracy_score" in weighted_metrics
        assert "confidence_accuracy_correlation" in weighted_metrics

        # Weighted error should be lower than unweighted due to high-confidence accurate predictions
        standard_metrics = multi_confidence_validator.get_accuracy_metrics(
            "weighted_test", window_hours=24
        )
        assert (
            weighted_metrics["weighted_mean_error"]
            <= standard_metrics.mean_error_minutes
        )

    def test_confidence_trend_analysis(self, multi_confidence_validator):
        """Test confidence level trend analysis over time."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Simulate decreasing model confidence over time (model degradation)
        for i in range(24):  # 24 hours of predictions
            predicted = base_time + timedelta(hours=i)
            confidence = max(0.5, 0.95 - (i * 0.02))  # Gradually decreasing confidence
            actual = predicted + timedelta(minutes=10)  # Consistent moderate accuracy

            record = ValidationRecord(
                room_id="trend_test",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=5),
            )
            multi_confidence_validator._pending_validations["trend_test"].append(record)
            multi_confidence_validator.validate_prediction("trend_test", actual)

        # Analyze confidence trends
        trend_analysis = multi_confidence_validator.get_confidence_trend_analysis(
            "trend_test", window_hours=24
        )

        assert trend_analysis is not None
        assert "trend_direction" in trend_analysis
        assert "trend_magnitude" in trend_analysis
        assert "confidence_volatility" in trend_analysis
        assert "trend_significance" in trend_analysis

        # Should detect decreasing confidence trend
        assert trend_analysis["trend_direction"] == "decreasing"
        assert trend_analysis["trend_magnitude"] > 0.1  # Significant decrease
        assert trend_analysis["trend_significance"] == "significant"


class TestConfidenceCalibrationIntegration:
    """Test integration of confidence calibration with the broader validation system."""

    @pytest.fixture
    def integrated_validator(self):
        """Create validator with full confidence calibration integration."""
        return PredictionValidator(
            accuracy_threshold_minutes=15,
            confidence_threshold=0.8,
            track_prediction_intervals=True,
            enable_adaptive_calibration=True,
        )

    def test_end_to_end_calibration_workflow(self, integrated_validator):
        """Test complete confidence calibration workflow."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Phase 1: Add initial predictions with poor calibration
        for i in range(20):
            predicted = base_time + timedelta(minutes=i * 15)
            confidence = 0.9  # Overconfident model
            actual = predicted + timedelta(
                minutes=20
            )  # Poor accuracy despite high confidence

            record = ValidationRecord(
                room_id="integration_test",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=5),
            )
            integrated_validator._pending_validations["integration_test"].append(record)
            integrated_validator.validate_prediction("integration_test", actual)

        # Check initial calibration
        initial_calibration = integrated_validator.get_calibration_metrics(
            "integration_test", window_hours=24
        )
        assert (
            initial_calibration["calibration_score"] < 0.6
        )  # Poor initial calibration

        # Phase 2: Apply calibration correction and add better-calibrated predictions
        calibration_adjustment = (
            integrated_validator.get_confidence_calibration_recommendation(
                "integration_test", window_hours=24
            )
        )

        adjustment_factor = calibration_adjustment["recommended_confidence_adjustment"]

        # Add new predictions with adjusted confidence
        for i in range(20):
            predicted = base_time + timedelta(hours=2, minutes=i * 15)
            original_confidence = 0.9
            adjusted_confidence = max(
                0.1, min(0.99, original_confidence + adjustment_factor)
            )
            actual = predicted + timedelta(minutes=20)  # Same accuracy

            record = ValidationRecord(
                room_id="integration_test",
                predicted_time=predicted,
                confidence=adjusted_confidence,
                timestamp=predicted - timedelta(minutes=5),
            )
            integrated_validator._pending_validations["integration_test"].append(record)
            integrated_validator.validate_prediction("integration_test", actual)

        # Check improved calibration
        final_calibration = integrated_validator.get_calibration_metrics(
            "integration_test", window_hours=24
        )
        assert (
            final_calibration["calibration_score"]
            > initial_calibration["calibration_score"]
        )

    @pytest.mark.asyncio
    async def test_real_time_calibration_monitoring(self, integrated_validator):
        """Test real-time confidence calibration monitoring."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Simulate real-time prediction stream with monitoring
        calibration_alerts = []

        async def calibration_alert_handler(room_id: str, alert_data: Dict[str, Any]):
            calibration_alerts.append((room_id, alert_data))

        # Register alert handler
        integrated_validator.register_calibration_alert_handler(
            calibration_alert_handler
        )

        # Add predictions that should trigger calibration alerts
        for i in range(30):
            predicted = base_time + timedelta(minutes=i * 10)
            confidence = 0.95  # Very high confidence

            # Alternate between accurate and inaccurate predictions
            if i % 2 == 0:
                actual = predicted + timedelta(minutes=2)  # Accurate
            else:
                actual = predicted + timedelta(minutes=25)  # Very inaccurate

            record = ValidationRecord(
                room_id="real_time_test",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=2),
            )
            integrated_validator._pending_validations["real_time_test"].append(record)

            # Simulate real-time validation
            await integrated_validator.validate_prediction_async(
                "real_time_test", actual
            )

            # Check for calibration alerts after sufficient data
            if i > 10 and i % 5 == 0:
                await integrated_validator.check_calibration_status_async(
                    "real_time_test"
                )

        # Should have received calibration alerts due to poor calibration
        assert len(calibration_alerts) > 0

        # Verify alert content
        for room_id, alert_data in calibration_alerts:
            assert room_id == "real_time_test"
            assert "calibration_score" in alert_data
            assert "alert_level" in alert_data
            assert alert_data["alert_level"] in ["warning", "critical"]

    def test_multi_room_calibration_comparison(self, integrated_validator):
        """Test calibration comparison across multiple rooms."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)

        # Room 1: Well-calibrated predictions
        for i in range(20):
            predicted = base_time + timedelta(minutes=i * 15)
            confidence = 0.8
            actual = predicted + timedelta(
                minutes=8
            )  # Good accuracy matching confidence

            record = ValidationRecord(
                room_id="room_1",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=2),
            )
            integrated_validator._pending_validations["room_1"].append(record)
            integrated_validator.validate_prediction("room_1", actual)

        # Room 2: Poorly-calibrated predictions
        for i in range(20):
            predicted = base_time + timedelta(minutes=i * 15)
            confidence = 0.9  # High confidence
            actual = predicted + timedelta(minutes=25)  # Poor accuracy

            record = ValidationRecord(
                room_id="room_2",
                predicted_time=predicted,
                confidence=confidence,
                timestamp=predicted - timedelta(minutes=2),
            )
            integrated_validator._pending_validations["room_2"].append(record)
            integrated_validator.validate_prediction("room_2", actual)

        # Compare calibration across rooms
        room_comparison = integrated_validator.compare_room_calibration(
            ["room_1", "room_2"], window_hours=24
        )

        assert room_comparison is not None
        assert len(room_comparison) == 2

        room_1_data = next(r for r in room_comparison if r["room_id"] == "room_1")
        room_2_data = next(r for r in room_comparison if r["room_id"] == "room_2")

        # Room 1 should have better calibration
        assert room_1_data["calibration_score"] > room_2_data["calibration_score"]
        assert (
            room_1_data["calibration_ranking"] < room_2_data["calibration_ranking"]
        )  # Lower rank = better
