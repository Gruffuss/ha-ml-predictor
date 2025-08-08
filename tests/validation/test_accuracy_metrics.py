"""
Comprehensive tests for accuracy metrics calculation and statistical analysis.

This test suite validates the calculation of statistical accuracy metrics,
time-series accuracy analysis, and comprehensive prediction performance
reporting used in the occupancy prediction system validation.
"""

import asyncio
import statistics
from collections import defaultdict
from datetime import datetime
from datetime import timedelta
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.validator import AccuracyLevel
from src.adaptation.validator import AccuracyMetrics
from src.adaptation.validator import PredictionValidator
from src.adaptation.validator import ValidationError
from src.adaptation.validator import ValidationRecord
from src.adaptation.validator import ValidationStatus
from src.models.base.predictor import PredictionResult


class TestAccuracyMetricsCalculation:
    """Test statistical accuracy metrics computation algorithms."""

    @pytest.fixture
    def validator(self):
        """Create validator with test configuration."""
        return PredictionValidator(accuracy_threshold_minutes=15)

    @pytest.fixture
    def sample_validation_records(self):
        """Create sample validation records with known accuracy patterns."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)
        records = []

        # Create records with specific error patterns for testing
        error_patterns = [
            (2.0, 0.95, AccuracyLevel.EXCELLENT),  # Excellent
            (4.5, 0.92, AccuracyLevel.EXCELLENT),  # Excellent
            (7.0, 0.88, AccuracyLevel.GOOD),  # Good
            (9.5, 0.85, AccuracyLevel.GOOD),  # Good
            (12.0, 0.82, AccuracyLevel.ACCEPTABLE),  # Acceptable
            (14.5, 0.78, AccuracyLevel.ACCEPTABLE),  # Acceptable
            (18.0, 0.75, AccuracyLevel.POOR),  # Poor
            (25.0, 0.68, AccuracyLevel.POOR),  # Poor
            (40.0, 0.55, AccuracyLevel.UNACCEPTABLE),  # Unacceptable
            (55.0, 0.48, AccuracyLevel.UNACCEPTABLE),  # Unacceptable
        ]

        for i, (error_minutes, confidence, level) in enumerate(error_patterns):
            predicted_time = base_time + timedelta(minutes=i * 10)
            actual_time = predicted_time + timedelta(minutes=error_minutes)

            record = ValidationRecord(
                prediction_id=f"test_prediction_{i:03d}",
                room_id=f"room_{i % 3}",  # Distribute across 3 rooms
                model_type=f"Model_{i % 2}",  # Distribute across 2 models
                model_version="1.0",
                predicted_time=predicted_time,
                transition_type="occupied" if i % 2 == 0 else "vacant",
                confidence_score=confidence,
            )

            # Validate the record to set actual time and error
            record.validate_against_actual(actual_time, threshold_minutes=15)
            records.append(record)

        return records

    def test_basic_accuracy_metrics_calculation(
        self, validator, sample_validation_records
    ):
        """Test basic accuracy metrics calculation from validation records."""
        # Calculate metrics using validator's internal method
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, hours_back=24
        )

        # Verify basic counts
        assert metrics.total_predictions == 10
        assert metrics.validated_predictions == 10
        assert metrics.expired_predictions == 0
        assert metrics.failed_predictions == 0

        # Verify accuracy rate (errors <= 15 minutes should be accurate)
        expected_accurate = sum(
            1 for r in sample_validation_records if r.error_minutes <= 15
        )
        assert metrics.accurate_predictions == expected_accurate
        assert metrics.accuracy_rate == (expected_accurate / 10) * 100

        # Verify error statistics
        expected_errors = [r.error_minutes for r in sample_validation_records]
        assert metrics.mean_error_minutes == pytest.approx(
            statistics.mean(expected_errors), rel=0.01
        )
        assert metrics.median_error_minutes == pytest.approx(
            statistics.median(expected_errors), rel=0.01
        )
        assert metrics.mae_minutes == pytest.approx(
            statistics.mean(expected_errors), rel=0.01
        )

    def test_error_distribution_metrics(self, validator, sample_validation_records):
        """Test error distribution and percentile calculations."""
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, 24
        )

        # Verify error percentiles are calculated
        assert 25 in metrics.error_percentiles
        assert 75 in metrics.error_percentiles
        assert 90 in metrics.error_percentiles
        assert 95 in metrics.error_percentiles

        # Verify percentiles are ordered correctly
        assert metrics.error_percentiles[25] <= metrics.error_percentiles[75]
        assert metrics.error_percentiles[75] <= metrics.error_percentiles[90]
        assert metrics.error_percentiles[90] <= metrics.error_percentiles[95]

        # Verify RMSE calculation
        errors = [r.error_minutes for r in sample_validation_records]
        expected_rmse = (sum(e**2 for e in errors) / len(errors)) ** 0.5
        assert metrics.rmse_minutes == pytest.approx(expected_rmse, rel=0.01)

    def test_accuracy_level_distribution(self, validator, sample_validation_records):
        """Test accuracy level distribution counting."""
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, 24
        )

        # Count expected accuracy levels from sample data
        expected_counts = defaultdict(int)
        for record in sample_validation_records:
            if record.accuracy_level:
                expected_counts[record.accuracy_level.value] += 1

        # Verify accuracy level counts
        assert metrics.accuracy_by_level == dict(expected_counts)

        # Verify all levels are represented in our test data
        assert "excellent" in metrics.accuracy_by_level
        assert "good" in metrics.accuracy_by_level
        assert "acceptable" in metrics.accuracy_by_level
        assert "poor" in metrics.accuracy_by_level
        assert "unacceptable" in metrics.accuracy_by_level

    def test_bias_analysis_calculation(self, validator, sample_validation_records):
        """Test bias analysis calculation (early vs late predictions)."""
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, 24
        )

        # Calculate expected bias from sample data
        biases = []
        for record in sample_validation_records:
            if record.actual_time and record.predicted_time:
                bias_seconds = (
                    record.actual_time - record.predicted_time
                ).total_seconds()
                bias_minutes = bias_seconds / 60
                biases.append(bias_minutes)

        expected_mean_bias = statistics.mean(biases)
        expected_bias_std = statistics.stdev(biases) if len(biases) > 1 else 0.0

        assert metrics.mean_bias_minutes == pytest.approx(expected_mean_bias, rel=0.01)
        assert metrics.bias_std_minutes == pytest.approx(expected_bias_std, rel=0.01)

        # Test bias direction classification
        if abs(expected_mean_bias) < 1:
            assert metrics.bias_direction == "unbiased"
        elif expected_mean_bias > 0:
            assert metrics.bias_direction == "predicts_late"
        else:
            assert metrics.bias_direction == "predicts_early"

    def test_confidence_analysis_calculation(
        self, validator, sample_validation_records
    ):
        """Test confidence score analysis and calibration."""
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, 24
        )

        # Verify confidence statistics
        confidences = [r.confidence_score for r in sample_validation_records]
        expected_mean_confidence = statistics.mean(confidences)

        assert metrics.mean_confidence == pytest.approx(
            expected_mean_confidence, rel=0.01
        )

        # Verify confidence-accuracy correlation calculation
        errors = [r.error_minutes for r in sample_validation_records]
        accuracies = [1 / (1 + e) for e in errors]  # Transform error to accuracy score

        correlation_matrix = np.corrcoef(confidences, accuracies)
        expected_correlation = (
            correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0
        )

        assert metrics.confidence_accuracy_correlation == pytest.approx(
            expected_correlation, abs=0.01
        )

        # Verify calibration score
        expected_calibration = max(0.0, expected_correlation)
        assert metrics.confidence_calibration_score == pytest.approx(
            expected_calibration, abs=0.01
        )

    def test_overconfidence_underconfidence_rates(
        self, validator, sample_validation_records
    ):
        """Test overconfidence and underconfidence rate calculations."""
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, 24
        )

        # Calculate expected overconfidence rate (high confidence, wrong prediction)
        high_conf_threshold = 0.8
        low_conf_threshold = 0.4
        threshold_minutes = 15

        high_conf_wrong = 0
        low_conf_right = 0
        total_predictions = len(sample_validation_records)

        for record in sample_validation_records:
            confidence = record.confidence_score
            error = record.error_minutes

            if confidence > high_conf_threshold and error > threshold_minutes:
                high_conf_wrong += 1
            elif confidence < low_conf_threshold and error <= threshold_minutes:
                low_conf_right += 1

        expected_overconfidence = (high_conf_wrong / total_predictions) * 100
        expected_underconfidence = (low_conf_right / total_predictions) * 100

        assert metrics.overconfidence_rate == pytest.approx(
            expected_overconfidence, rel=0.01
        )
        assert metrics.underconfidence_rate == pytest.approx(
            expected_underconfidence, rel=0.01
        )

    def test_time_based_analysis_metrics(self, validator, sample_validation_records):
        """Test time-based analysis metrics calculation."""
        metrics = validator._calculate_metrics_from_records(
            sample_validation_records, 24
        )

        # Verify measurement period calculation
        prediction_times = [r.prediction_time for r in sample_validation_records]
        expected_start = min(prediction_times)
        expected_end = max(prediction_times)

        assert metrics.measurement_period_start == expected_start
        assert metrics.measurement_period_end == expected_end

        # Verify predictions per hour calculation
        period_hours = (expected_end - expected_start).total_seconds() / 3600
        if period_hours > 0:
            expected_rate = len(sample_validation_records) / period_hours
            assert metrics.predictions_per_hour == pytest.approx(
                expected_rate, rel=0.01
            )
        else:
            assert metrics.predictions_per_hour == 0.0

    def test_validation_rate_properties(self, validator):
        """Test validation rate property calculations."""
        # Create mixed status records
        records = []
        base_time = datetime(2024, 1, 15, 14, 30, 0)

        # Create records with different statuses
        statuses = [
            ValidationStatus.VALIDATED,  # 5 validated
            ValidationStatus.VALIDATED,
            ValidationStatus.VALIDATED,
            ValidationStatus.VALIDATED,
            ValidationStatus.VALIDATED,
            ValidationStatus.EXPIRED,  # 2 expired
            ValidationStatus.EXPIRED,
            ValidationStatus.FAILED,  # 1 failed
            ValidationStatus.PENDING,  # 2 pending
            ValidationStatus.PENDING,
        ]

        for i, status in enumerate(statuses):
            record = ValidationRecord(
                prediction_id=f"mixed_status_{i}",
                room_id="test_room",
                model_type="TestModel",
                model_version="1.0",
                predicted_time=base_time + timedelta(minutes=i),
                transition_type="occupied",
                confidence_score=0.8,
            )
            record.status = status

            if status == ValidationStatus.VALIDATED:
                record.actual_time = base_time + timedelta(minutes=i + 5)
                record.error_minutes = 5.0
                record.accuracy_level = AccuracyLevel.EXCELLENT
                record.validation_time = datetime.utcnow()

            records.append(record)

        # Calculate metrics
        metrics = validator._calculate_metrics_from_records(records, 24)

        # Verify counts
        assert metrics.total_predictions == 10
        assert metrics.validated_predictions == 5
        assert metrics.expired_predictions == 2
        assert metrics.failed_predictions == 1

        # Verify rates
        assert metrics.validation_rate == 50.0  # 5/10 * 100
        assert metrics.expiration_rate == 20.0  # 2/10 * 100


class TestAccuracyMetricsAggregation:
    """Test accuracy metrics aggregation and filtering."""

    @pytest.fixture
    def validator(self):
        """Create validator for aggregation testing."""
        return PredictionValidator(accuracy_threshold_minutes=12)

    @pytest.fixture
    def multi_room_records(self):
        """Create validation records across multiple rooms and models."""
        records = []
        base_time = datetime(2024, 1, 15, 14, 30, 0)

        rooms = ["living_room", "bedroom", "kitchen"]
        models = ["LSTM", "XGBoost", "HMM"]

        record_id = 0
        for room in rooms:
            for model in models:
                for hour_offset in range(4):  # 4 records per room-model combination
                    predicted_time = base_time + timedelta(hours=hour_offset)
                    # Create varying error patterns
                    error_minutes = (
                        5.0 + (record_id % 8) * 3
                    )  # Errors from 5 to 26 minutes
                    actual_time = predicted_time + timedelta(minutes=error_minutes)

                    record = ValidationRecord(
                        prediction_id=f"multi_{record_id:03d}",
                        room_id=room,
                        model_type=model,
                        model_version="1.0",
                        predicted_time=predicted_time,
                        transition_type="occupied" if record_id % 2 == 0 else "vacant",
                        confidence_score=0.7 + (record_id % 5) * 0.05,
                    )

                    record.validate_against_actual(actual_time, threshold_minutes=12)
                    records.append(record)
                    record_id += 1

        return records

    @pytest.mark.asyncio
    async def test_room_specific_metrics_calculation(
        self, validator, multi_room_records
    ):
        """Test calculation of room-specific accuracy metrics."""
        # Store records in validator
        with validator._lock:
            for record in multi_room_records:
                validator._validation_records[record.prediction_id] = record
                validator._records_by_room[record.room_id].append(record.prediction_id)

        # Calculate metrics for specific room
        living_room_metrics = await validator.get_room_accuracy(
            "living_room", hours_back=24
        )

        # Verify only living room records are included
        expected_living_room_count = sum(
            1 for r in multi_room_records if r.room_id == "living_room"
        )
        assert living_room_metrics.total_predictions == expected_living_room_count
        assert living_room_metrics.validated_predictions == expected_living_room_count

        # Verify metrics calculation for room subset
        living_room_records = [
            r for r in multi_room_records if r.room_id == "living_room"
        ]
        expected_errors = [r.error_minutes for r in living_room_records]
        assert living_room_metrics.mean_error_minutes == pytest.approx(
            statistics.mean(expected_errors), rel=0.01
        )

    @pytest.mark.asyncio
    async def test_model_specific_metrics_calculation(
        self, validator, multi_room_records
    ):
        """Test calculation of model-specific accuracy metrics."""
        # Store records in validator
        with validator._lock:
            for record in multi_room_records:
                validator._validation_records[record.prediction_id] = record
                validator._records_by_model[record.model_type].append(
                    record.prediction_id
                )

        # Calculate metrics for specific model
        lstm_metrics = await validator.get_model_accuracy("LSTM", hours_back=24)

        # Verify only LSTM records are included
        expected_lstm_count = sum(
            1 for r in multi_room_records if r.model_type == "LSTM"
        )
        assert lstm_metrics.total_predictions == expected_lstm_count
        assert lstm_metrics.validated_predictions == expected_lstm_count

        # Verify accuracy calculation for model subset
        lstm_records = [r for r in multi_room_records if r.model_type == "LSTM"]
        expected_accurate = sum(1 for r in lstm_records if r.error_minutes <= 12)
        expected_accuracy_rate = (expected_accurate / len(lstm_records)) * 100

        assert lstm_metrics.accuracy_rate == pytest.approx(
            expected_accuracy_rate, rel=0.01
        )

    @pytest.mark.asyncio
    async def test_time_window_filtering(self, validator, multi_room_records):
        """Test time window filtering for metrics calculation."""
        # Store records in validator
        with validator._lock:
            for record in multi_room_records:
                validator._validation_records[record.prediction_id] = record

        # Test with narrow time window (should exclude older records)
        recent_metrics = await validator.get_accuracy_metrics(hours_back=2)
        all_metrics = await validator.get_accuracy_metrics(hours_back=24)

        # Recent metrics should have fewer records
        assert recent_metrics.total_predictions <= all_metrics.total_predictions

        # Verify time filtering logic
        cutoff_time = datetime.utcnow() - timedelta(hours=2)
        expected_recent_count = sum(
            1 for r in multi_room_records if r.prediction_time >= cutoff_time
        )

        assert recent_metrics.total_predictions == expected_recent_count

    @pytest.mark.asyncio
    async def test_combined_filtering_metrics(self, validator, multi_room_records):
        """Test combined room and model filtering for metrics."""
        # Store records in validator
        with validator._lock:
            for record in multi_room_records:
                validator._validation_records[record.prediction_id] = record
                validator._records_by_room[record.room_id].append(record.prediction_id)
                validator._records_by_model[record.model_type].append(
                    record.prediction_id
                )

        # Calculate metrics with both room and model filters
        combined_metrics = await validator.get_accuracy_metrics(
            room_id="bedroom", model_type="XGBoost", hours_back=24
        )

        # Verify filtering worked correctly
        expected_combined_count = sum(
            1
            for r in multi_room_records
            if r.room_id == "bedroom" and r.model_type == "XGBoost"
        )

        assert combined_metrics.total_predictions == expected_combined_count

        # Verify metrics are calculated correctly for filtered subset
        filtered_records = [
            r
            for r in multi_room_records
            if r.room_id == "bedroom" and r.model_type == "XGBoost"
        ]

        if filtered_records:
            expected_errors = [r.error_minutes for r in filtered_records]
            assert combined_metrics.mean_error_minutes == pytest.approx(
                statistics.mean(expected_errors), rel=0.01
            )


class TestAccuracyMetricsEdgeCases:
    """Test edge cases and error conditions in accuracy metrics calculation."""

    @pytest.fixture
    def validator(self):
        """Create validator for edge case testing."""
        return PredictionValidator(accuracy_threshold_minutes=10)

    def test_empty_records_metrics_calculation(self, validator):
        """Test metrics calculation with empty validation records."""
        metrics = validator._calculate_metrics_from_records([], hours_back=24)

        # Should return empty metrics without errors
        assert metrics.total_predictions == 0
        assert metrics.validated_predictions == 0
        assert metrics.accuracy_rate == 0.0
        assert metrics.mean_error_minutes == 0.0
        assert metrics.validation_rate == 0.0

    def test_single_record_metrics_calculation(self, validator):
        """Test metrics calculation with single validation record."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)
        predicted_time = base_time
        actual_time = base_time + timedelta(minutes=7.5)

        record = ValidationRecord(
            prediction_id="single_record",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.85,
        )

        record.validate_against_actual(actual_time, threshold_minutes=10)

        metrics = validator._calculate_metrics_from_records([record], hours_back=24)

        # Verify single record metrics
        assert metrics.total_predictions == 1
        assert metrics.validated_predictions == 1
        assert metrics.accurate_predictions == 1
        assert metrics.accuracy_rate == 100.0
        assert metrics.mean_error_minutes == 7.5
        assert metrics.median_error_minutes == 7.5
        assert metrics.std_error_minutes == 0.0  # Single value has no deviation

    def test_all_invalid_records_metrics(self, validator):
        """Test metrics calculation with all invalid/expired records."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)
        records = []

        for i in range(5):
            record = ValidationRecord(
                prediction_id=f"invalid_{i}",
                room_id="test_room",
                model_type="TestModel",
                model_version="1.0",
                predicted_time=base_time + timedelta(minutes=i * 10),
                transition_type="occupied",
                confidence_score=0.8,
            )

            # Mark records as expired or failed
            if i % 2 == 0:
                record.mark_expired()
            else:
                record.mark_failed("Test failure")

            records.append(record)

        metrics = validator._calculate_metrics_from_records(records, hours_back=24)

        # Should handle invalid records correctly
        assert metrics.total_predictions == 5
        assert metrics.validated_predictions == 0
        assert metrics.expired_predictions == 3  # 0, 2, 4
        assert metrics.failed_predictions == 2  # 1, 3
        assert metrics.accuracy_rate == 0.0
        assert metrics.validation_rate == 0.0
        assert metrics.expiration_rate == 60.0

    def test_extreme_error_values_handling(self, validator):
        """Test handling of extreme error values in metrics calculation."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)
        records = []

        # Create records with extreme error values
        extreme_errors = [0.0, 0.01, 1440.0, 10080.0]  # 0 min, 0.01 min, 1 day, 1 week

        for i, error_minutes in enumerate(extreme_errors):
            predicted_time = base_time + timedelta(minutes=i * 10)
            actual_time = predicted_time + timedelta(minutes=error_minutes)

            record = ValidationRecord(
                prediction_id=f"extreme_{i}",
                room_id="test_room",
                model_type="TestModel",
                model_version="1.0",
                predicted_time=predicted_time,
                transition_type="occupied",
                confidence_score=0.8,
            )

            record.validate_against_actual(actual_time, threshold_minutes=10)
            records.append(record)

        metrics = validator._calculate_metrics_from_records(records, hours_back=24)

        # Verify extreme values are handled correctly
        assert metrics.total_predictions == 4
        assert not np.isnan(metrics.mean_error_minutes)
        assert not np.isnan(metrics.rmse_minutes)
        assert not np.isinf(metrics.mean_error_minutes)
        assert not np.isinf(metrics.rmse_minutes)

        # Verify percentiles work with extreme values
        assert all(not np.isnan(v) for v in metrics.error_percentiles.values())
        assert all(not np.isinf(v) for v in metrics.error_percentiles.values())

    def test_nan_confidence_handling(self, validator):
        """Test handling of NaN confidence values in metrics calculation."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)

        record = ValidationRecord(
            prediction_id="nan_confidence",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=base_time,
            transition_type="occupied",
            confidence_score=float("nan"),  # NaN confidence
        )

        actual_time = base_time + timedelta(minutes=5)
        record.validate_against_actual(actual_time, threshold_minutes=10)

        metrics = validator._calculate_metrics_from_records([record], hours_back=24)

        # Should handle NaN confidence gracefully
        assert metrics.total_predictions == 1
        assert not np.isnan(metrics.mean_error_minutes)  # Error should be valid
        # Confidence-related metrics may be NaN or 0, but shouldn't crash
        assert not np.isinf(metrics.mean_confidence) or metrics.mean_confidence == 0.0
