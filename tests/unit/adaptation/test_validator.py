"""
Comprehensive tests for PredictionValidator and validation infrastructure.

Tests real validation functionality including prediction recording, validation against
actual outcomes, accuracy metrics calculation, and comprehensive validation tracking.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

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


@pytest.fixture
def sample_prediction_result():
    """Create a sample PredictionResult for testing."""
    return PredictionResult(
        predicted_time=datetime.now(UTC) + timedelta(minutes=30),
        transition_type="occupied",
        confidence_score=0.85,
        model_type="lstm",  # Use string instead of enum
        model_version="v1.2",
        prediction_interval=(
            datetime.now(UTC) + timedelta(minutes=25),
            datetime.now(UTC) + timedelta(minutes=35),
        ),
        alternatives=[
            (datetime.now(UTC) + timedelta(minutes=40), 0.75),
            (datetime.now(UTC) + timedelta(minutes=20), 0.65),
        ],
        prediction_metadata={"features_used": 15, "training_samples": 1000},
    )


@pytest.fixture
def validator():
    """Create a PredictionValidator instance for testing."""
    return PredictionValidator(
        accuracy_threshold_minutes=15,
        max_validation_delay_hours=6,
        max_memory_records=1000,
        cleanup_interval_hours=12,
        metrics_cache_ttl_minutes=30,
    )


class TestValidationRecord:
    """Test ValidationRecord functionality."""

    def test_validation_record_creation(self):
        """Test creating validation records."""
        predicted_time = datetime.now(UTC) + timedelta(minutes=30)

        record = ValidationRecord(
            prediction_id="test_001",
            room_id="bedroom",
            model_type=ModelType.XGBOOST,
            model_version="v1.0",
            predicted_time=predicted_time,
            transition_type="vacant",
            confidence_score=0.78,
        )

        assert record.prediction_id == "test_001"
        assert record.room_id == "bedroom"
        assert record.model_type == ModelType.XGBOOST
        assert record.predicted_time == predicted_time
        assert record.status == ValidationStatus.PENDING
        assert record.actual_time is None
        assert record.error_minutes is None

    def test_validation_record_validate_against_actual(self):
        """Test validation against actual transition time."""
        predicted_time = datetime.now(UTC) + timedelta(minutes=30)
        actual_time = datetime.now(UTC) + timedelta(minutes=35)  # 5 minute error

        record = ValidationRecord(
            prediction_id="validation_test",
            room_id="kitchen",
            model_type=ModelType.LSTM,
            model_version="v1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.80,
        )

        is_accurate = record.validate_against_actual(actual_time, threshold_minutes=15)

        assert is_accurate is True  # 5 minutes <= 15 minutes threshold
        assert record.status == ValidationStatus.VALIDATED
        assert record.actual_time == actual_time
        assert record.error_minutes == 5.0
        assert record.accuracy_level == AccuracyLevel.GOOD  # 5-10 min = good
        assert record.validation_time is not None

    def test_validation_record_accuracy_levels(self):
        """Test accuracy level classification."""
        predicted_time = datetime.now(UTC)

        # Test different accuracy levels
        test_cases = [
            (2, AccuracyLevel.EXCELLENT),  # < 5 min
            (7, AccuracyLevel.GOOD),  # 5-10 min
            (12, AccuracyLevel.ACCEPTABLE),  # 10-15 min
            (20, AccuracyLevel.POOR),  # 15-30 min
            (45, AccuracyLevel.UNACCEPTABLE),  # > 30 min
        ]

        for error_minutes, expected_level in test_cases:
            actual_time = predicted_time + timedelta(minutes=error_minutes)

            record = ValidationRecord(
                prediction_id=f"test_{error_minutes}",
                room_id="test_room",
                model_type=ModelType.LSTM,
                model_version="v1.0",
                predicted_time=predicted_time,
                transition_type="occupied",
                confidence_score=0.70,
            )

            record.validate_against_actual(actual_time, threshold_minutes=15)
            assert record.accuracy_level == expected_level
            assert record.error_minutes == error_minutes

    def test_validation_record_mark_expired(self):
        """Test marking validation record as expired."""
        record = ValidationRecord(
            prediction_id="expire_test",
            room_id="bathroom",
            model_type=ModelType.HMM,
            model_version="v1.0",
            predicted_time=datetime.now(UTC) - timedelta(hours=12),
            transition_type="vacant",
            confidence_score=0.65,
        )

        expiration_time = datetime.now(UTC)
        record.mark_expired(expiration_time)

        assert record.status == ValidationStatus.EXPIRED
        assert record.expiration_time == expiration_time

    def test_validation_record_mark_failed(self):
        """Test marking validation record as failed."""
        record = ValidationRecord(
            prediction_id="fail_test",
            room_id="office",
            model_type=ModelType.GP,
            model_version="v1.0",
            predicted_time=datetime.now(UTC),
            transition_type="occupied",
            confidence_score=0.55,
        )

        record.mark_failed("Invalid sensor data")

        assert record.status == ValidationStatus.FAILED
        assert record.validation_time is not None
        assert record.prediction_metadata["failure_reason"] == "Invalid sensor data"

    def test_validation_record_serialization(self):
        """Test validation record serialization to dictionary."""
        predicted_time = datetime.now(UTC)
        actual_time = predicted_time + timedelta(minutes=10)

        record = ValidationRecord(
            prediction_id="serialize_test",
            room_id="laundry_room",
            model_type=ModelType.ENSEMBLE,
            model_version="v2.0",
            predicted_time=predicted_time,
            transition_type="vacant",
            confidence_score=0.90,
            prediction_interval=(
                predicted_time - timedelta(minutes=5),
                predicted_time + timedelta(minutes=5),
            ),
            alternatives=[
                (predicted_time + timedelta(minutes=15), 0.80),
                (predicted_time - timedelta(minutes=10), 0.70),
            ],
            feature_snapshot={"temp": 22.5, "humidity": 45},
            prediction_metadata={"model_confidence": "high"},
        )

        record.validate_against_actual(actual_time, threshold_minutes=15)

        result_dict = record.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["prediction_id"] == "serialize_test"
        assert result_dict["room_id"] == "laundry_room"
        assert result_dict["error_minutes"] == 10.0
        assert (
            result_dict["accuracy_level"] == "acceptable"
        )  # 10 minutes = acceptable (10-15 min)
        assert result_dict["status"] == "validated"
        assert "prediction_interval" in result_dict
        assert "alternatives" in result_dict
        assert result_dict["feature_snapshot"]["temp"] == 22.5


class TestAccuracyMetrics:
    """Test AccuracyMetrics functionality."""

    def test_accuracy_metrics_creation(self):
        """Test accuracy metrics creation with various parameters."""
        metrics = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=85,
            accurate_predictions=68,
            expired_predictions=10,
            failed_predictions=5,
            accuracy_rate=80.0,
            mean_error_minutes=12.5,
            median_error_minutes=10.0,
            std_error_minutes=8.3,
            rmse_minutes=15.2,
            mae_minutes=12.5,
            mean_bias_minutes=2.1,
            bias_std_minutes=6.5,
            mean_confidence=0.72,
            confidence_accuracy_correlation=0.65,
            overconfidence_rate=15.0,
            underconfidence_rate=8.0,
        )

        assert metrics.total_predictions == 100
        assert metrics.accuracy_rate == 80.0
        assert metrics.validation_rate == 85.0  # 85/100 * 100
        assert metrics.expiration_rate == 10.0  # 10/100 * 100

    def test_accuracy_metrics_properties(self):
        """Test accuracy metrics computed properties."""
        metrics = AccuracyMetrics(
            total_predictions=200,
            validated_predictions=180,
            accurate_predictions=144,
            expired_predictions=15,
            mean_bias_minutes=-3.5,  # Negative bias
            confidence_accuracy_correlation=0.85,
        )

        assert metrics.validation_rate == 90.0
        assert metrics.expiration_rate == 7.5
        assert metrics.bias_direction == "predicts_early"  # Negative bias
        assert metrics.confidence_calibration_score == 0.85

    def test_accuracy_metrics_backward_compatibility(self):
        """Test backward compatibility with alternative parameter names."""
        # Test using alternative parameter names
        metrics = AccuracyMetrics(
            total_predictions=50,
            validated_predictions=45,
            avg_error_minutes=14.2,  # Alternative to mean_error_minutes
            confidence_calibration=0.78,  # Alternative to confidence_accuracy_correlation
            correct_predictions=36,  # Alternative to accurate_predictions
        )

        # Should map correctly
        assert metrics.mean_error_minutes == 14.2
        assert metrics.confidence_accuracy_correlation == 0.78
        assert metrics.accurate_predictions == 36

    def test_accuracy_metrics_serialization(self):
        """Test accuracy metrics serialization."""
        metrics = AccuracyMetrics(
            total_predictions=150,
            validated_predictions=135,
            accurate_predictions=108,
            accuracy_rate=80.0,
            mean_error_minutes=11.3,
            measurement_period_start=datetime.now(UTC) - timedelta(days=7),
            measurement_period_end=datetime.now(UTC),
            predictions_per_hour=2.5,
        )

        result_dict = metrics.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["total_predictions"] == 150
        assert result_dict["accuracy_rate"] == 80.0
        assert result_dict["validation_rate"] == 90.0  # 135/150 * 100
        assert "measurement_period_start" in result_dict
        assert "predictions_per_hour" in result_dict


class TestPredictionValidator:
    """Test PredictionValidator functionality."""

    def test_validator_initialization(self):
        """Test prediction validator initialization."""
        validator = PredictionValidator(
            accuracy_threshold_minutes=20,
            max_validation_delay_hours=8,
            max_memory_records=5000,
            cleanup_interval_hours=6,
            metrics_cache_ttl_minutes=15,
        )

        assert validator.accuracy_threshold_minutes == 20
        assert validator.accuracy_threshold == 20  # Alternative name
        assert validator.max_validation_delay == timedelta(hours=8)
        assert validator.max_memory_records == 5000
        assert validator.cleanup_interval == timedelta(hours=6)
        assert validator.metrics_cache_ttl == timedelta(minutes=15)

    @pytest.mark.asyncio
    async def test_record_prediction_with_result_object(
        self, validator, sample_prediction_result
    ):
        """Test recording prediction with PredictionResult object."""
        prediction_id = await validator.record_prediction(
            prediction=sample_prediction_result,
            room_id="living_room",
            feature_snapshot={"temperature": 22.0, "humidity": 45},
        )

        assert isinstance(prediction_id, str)
        assert "living_room" in prediction_id
        assert prediction_id in validator._validation_records

        record = validator._validation_records[prediction_id]
        assert record.room_id == "living_room"
        assert record.model_type == "lstm"
        assert record.confidence_score == 0.85
        assert record.status == ValidationStatus.PENDING
        assert record.feature_snapshot["temperature"] == 22.0

    @pytest.mark.asyncio
    async def test_record_prediction_with_individual_parameters(self, validator):
        """Test recording prediction with individual parameters."""
        predicted_time = datetime.now(UTC) + timedelta(minutes=45)

        prediction_id = await validator.record_prediction(
            room_id="bedroom",
            predicted_time=predicted_time,
            confidence=0.75,
            model_type=ModelType.XGBOOST,
            transition_type="vacant",
            prediction_metadata={"notes": "high confidence prediction"},
        )

        assert isinstance(prediction_id, str)
        assert "bedroom" in prediction_id

        record = validator._validation_records[prediction_id]
        assert record.room_id == "bedroom"
        assert record.predicted_time == predicted_time
        assert record.confidence_score == 0.75
        assert record.model_type == ModelType.XGBOOST
        assert record.transition_type == "vacant"

    @pytest.mark.asyncio
    async def test_validate_prediction_single(self, validator):
        """Test validating a single prediction."""
        # Record a prediction
        predicted_time = datetime.now(UTC) + timedelta(minutes=20)
        prediction_id = await validator.record_prediction(
            room_id="kitchen",
            predicted_time=predicted_time,
            confidence=0.80,
            model_type=ModelType.LSTM,
            transition_type="occupied",
        )

        # Validate it
        actual_time = predicted_time + timedelta(minutes=12)  # 12 minute error
        result = await validator.validate_prediction(
            room_id="kitchen",
            actual_transition_time=actual_time,
            transition_type="occupied",
            max_time_window_minutes=30,
        )

        assert result is not None
        assert isinstance(result, ValidationRecord)
        assert result.actual_time == actual_time
        assert result.error_minutes == 12.0
        assert result.status == ValidationStatus.VALIDATED
        assert result.accuracy_level == AccuracyLevel.ACCEPTABLE

    @pytest.mark.asyncio
    async def test_validate_prediction_alternative_parameter_names(self, validator):
        """Test validation with alternative parameter names."""
        # Record a prediction
        predicted_time = datetime.now(UTC) + timedelta(minutes=15)
        await validator.record_prediction(
            room_id="bathroom",
            predicted_time=predicted_time,
            confidence=0.70,
            model_type=ModelType.HMM,
            transition_type="vacant",
        )

        # Validate using alternative parameter name
        actual_time = predicted_time + timedelta(minutes=8)
        result = await validator.validate_prediction(
            room_id="bathroom",
            actual_time=actual_time,  # Alternative to actual_transition_time
            transition_type="vacant",
        )

        assert result is not None
        assert result.error_minutes == 8.0
        assert result.accuracy_level == AccuracyLevel.GOOD

    @pytest.mark.asyncio
    async def test_validate_prediction_no_candidates(self, validator):
        """Test validation when no matching predictions exist."""
        result = await validator.validate_prediction(
            room_id="nonexistent_room",
            actual_transition_time=datetime.now(UTC),
            transition_type="occupied",
        )

        assert result is None

    @pytest.mark.asyncio
    async def test_transition_type_matching(self, validator):
        """Test flexible transition type matching."""
        # Record prediction with generic type
        predicted_time = datetime.now(UTC) + timedelta(minutes=25)
        await validator.record_prediction(
            room_id="office",
            predicted_time=predicted_time,
            confidence=0.85,
            model_type=ModelType.GP,
            transition_type="occupied",
        )

        # Validate with more specific type
        actual_time = predicted_time + timedelta(minutes=3)
        result = await validator.validate_prediction(
            room_id="office",
            actual_transition_time=actual_time,
            transition_type="vacant_to_occupied",  # Should match "occupied"
        )

        assert result is not None
        assert result.error_minutes == 3.0

    @pytest.mark.asyncio
    async def test_get_accuracy_metrics_basic(self, validator):
        """Test basic accuracy metrics calculation."""
        # Create some test data
        await self._create_sample_validation_data(validator)

        metrics = await validator.get_accuracy_metrics(
            room_id="test_room", hours_back=24
        )

        assert isinstance(metrics, AccuracyMetrics)
        assert metrics.total_predictions > 0
        assert 0 <= metrics.accuracy_rate <= 100
        assert metrics.mean_error_minutes >= 0

    @pytest.mark.asyncio
    async def test_get_accuracy_metrics_with_time_range(self, validator):
        """Test accuracy metrics with specific time range."""
        await self._create_sample_validation_data(validator)

        end_time = datetime.now(UTC)
        start_time = end_time - timedelta(hours=12)

        metrics = await validator.get_accuracy_metrics(
            start_time=start_time, end_time=end_time
        )

        assert isinstance(metrics, AccuracyMetrics)
        # Should only include predictions within the time range

    @pytest.mark.asyncio
    async def test_get_room_accuracy(self, validator):
        """Test getting accuracy metrics for specific room."""
        await self._create_sample_validation_data(validator, room_id="specific_room")

        metrics = await validator.get_room_accuracy("specific_room", hours_back=24)

        assert isinstance(metrics, AccuracyMetrics)
        # All predictions should be from the specified room

    @pytest.mark.asyncio
    async def test_get_model_accuracy(self, validator):
        """Test getting accuracy metrics for specific model."""
        await self._create_sample_validation_data(validator, model_type=ModelType.LSTM)

        metrics = await validator.get_model_accuracy(ModelType.LSTM, hours_back=24)

        assert isinstance(metrics, AccuracyMetrics)

    @pytest.mark.asyncio
    async def test_get_overall_accuracy(self, validator):
        """Test getting overall accuracy metrics."""
        await self._create_sample_validation_data(validator)

        metrics = await validator.get_overall_accuracy(hours_back=48)

        assert isinstance(metrics, AccuracyMetrics)
        # Should include all rooms and models

    @pytest.mark.asyncio
    async def test_get_pending_validations(self, validator):
        """Test getting pending validations."""
        # Record some predictions without validating
        for i in range(3):
            await validator.record_prediction(
                room_id=f"pending_room_{i}",
                predicted_time=datetime.now(UTC) + timedelta(minutes=30 + i * 10),
                confidence=0.70 + i * 0.05,
                model_type=ModelType.XGBOOST,
                transition_type="occupied",
            )

        pending = await validator.get_pending_validations()

        assert len(pending) >= 3
        for record in pending:
            assert record.status == ValidationStatus.PENDING

    @pytest.mark.asyncio
    async def test_get_pending_validations_room_filter(self, validator):
        """Test getting pending validations for specific room."""
        await validator.record_prediction(
            room_id="filtered_room",
            predicted_time=datetime.now(UTC) + timedelta(minutes=30),
            confidence=0.80,
            model_type=ModelType.LSTM,
            transition_type="vacant",
        )

        pending = await validator.get_pending_validations(room_id="filtered_room")

        assert len(pending) >= 1
        for record in pending:
            assert record.room_id == "filtered_room"
            assert record.status == ValidationStatus.PENDING

    @pytest.mark.asyncio
    async def test_expire_old_predictions(self, validator):
        """Test expiring old predictions."""
        # Record old predictions
        old_time = datetime.now(UTC) - timedelta(hours=12)
        for i in range(3):
            record = ValidationRecord(
                prediction_id=f"old_prediction_{i}",
                room_id="expire_test_room",
                model_type=ModelType.HMM,
                model_version="v1.0",
                predicted_time=old_time - timedelta(hours=i),
                transition_type="occupied",
                confidence_score=0.75,
                prediction_time=old_time - timedelta(hours=i),
            )
            validator._validation_records[record.prediction_id] = record

        expired_count = await validator.expire_old_predictions(cutoff_hours=6)

        assert expired_count >= 3

        # Check that predictions were marked as expired
        for i in range(3):
            record = validator._validation_records[f"old_prediction_{i}"]
            assert record.status == ValidationStatus.EXPIRED

    @pytest.mark.asyncio
    async def test_export_validation_data_csv(self, validator, tmp_path):
        """Test exporting validation data to CSV."""
        await self._create_sample_validation_data(validator)

        output_file = tmp_path / "validation_export.csv"

        count = await validator.export_validation_data(
            output_path=output_file, format="csv", days_back=1
        )

        assert count > 0
        assert output_file.exists()

        # Verify CSV content
        import csv

        with open(output_file, "r") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            assert len(rows) == count
            assert "prediction_id" in rows[0]
            assert "room_id" in rows[0]
            assert "error_minutes" in rows[0]

    @pytest.mark.asyncio
    async def test_export_validation_data_json(self, validator, tmp_path):
        """Test exporting validation data to JSON."""
        await self._create_sample_validation_data(validator)

        output_file = tmp_path / "validation_export.json"

        count = await validator.export_validation_data(
            output_path=output_file, format="json", days_back=1
        )

        assert count > 0
        assert output_file.exists()

        # Verify JSON content
        import json

        with open(output_file, "r") as f:
            data = json.load(f)
            assert "export_time" in data
            assert "record_count" in data
            assert "records" in data
            assert len(data["records"]) == count

    @pytest.mark.asyncio
    async def test_get_validation_stats(self, validator):
        """Test getting validation system statistics."""
        await self._create_sample_validation_data(validator)

        stats = await validator.get_validation_stats()

        assert isinstance(stats, dict)
        assert "total_records" in stats
        assert "total_predictions" in stats
        assert "validation_rate" in stats
        assert "records_by_room" in stats
        assert "records_by_model" in stats
        assert "records_by_status" in stats
        assert stats["total_records"] > 0

    @pytest.mark.asyncio
    async def test_cleanup_old_records(self, validator):
        """Test cleaning up old validation records."""
        # Add old records
        old_time = datetime.now(UTC) - timedelta(days=45)
        for i in range(5):
            record = ValidationRecord(
                prediction_id=f"cleanup_test_{i}",
                room_id="cleanup_room",
                model_type=ModelType.ENSEMBLE,
                model_version="v1.0",
                predicted_time=old_time,
                transition_type="vacant",
                confidence_score=0.60,
                prediction_time=old_time - timedelta(days=i),
            )
            validator._validation_records[record.prediction_id] = record
            # Add to indexes for cleanup to work properly
            validator._records_by_room[record.room_id].append(record.prediction_id)
            validator._records_by_model[record.model_type].append(record.prediction_id)

        initial_count = len(validator._validation_records)
        removed_count = validator.cleanup_old_records(days_to_keep=30)
        final_count = len(validator._validation_records)

        assert removed_count > 0
        assert final_count < initial_count

    @pytest.mark.asyncio
    async def test_cleanup_old_predictions_async(self, validator):
        """Test async cleanup of old predictions."""
        # Add old records
        old_time = datetime.now(UTC) - timedelta(days=45)
        for i in range(3):
            record = ValidationRecord(
                prediction_id=f"async_cleanup_{i}",
                room_id="async_cleanup_room",
                model_type=ModelType.GP,
                model_version="v1.0",
                predicted_time=old_time,
                transition_type="occupied",
                confidence_score=0.65,
                prediction_time=old_time - timedelta(days=i),
            )
            validator._validation_records[record.prediction_id] = record
            # Add to indexes for cleanup to work properly
            validator._records_by_room[record.room_id].append(record.prediction_id)
            validator._records_by_model[record.model_type].append(record.prediction_id)

        cleaned_count = await validator.cleanup_old_predictions(days_to_keep=30)

        assert cleaned_count > 0

    @pytest.mark.asyncio
    async def test_get_total_predictions(self, validator):
        """Test getting total prediction count."""
        initial_count = await validator.get_total_predictions()

        # Add some predictions
        for i in range(5):
            await validator.record_prediction(
                room_id=f"count_test_room_{i}",
                predicted_time=datetime.now(UTC) + timedelta(minutes=30 + i),
                confidence=0.75,
                model_type=ModelType.LSTM,
                transition_type="occupied",
            )

        final_count = await validator.get_total_predictions()

        assert final_count >= initial_count + 5

    @pytest.mark.asyncio
    async def test_get_validation_rate(self, validator):
        """Test getting validation rate."""
        await self._create_sample_validation_data(validator)

        rate = await validator.get_validation_rate()

        assert isinstance(rate, float)
        assert 0.0 <= rate <= 100.0

    @pytest.mark.asyncio
    async def test_get_accuracy_trend(self, validator):
        """Test getting accuracy trend over time."""
        await self._create_sample_validation_data(validator)

        trend_data = await validator.get_accuracy_trend(
            room_id="test_room", hours_back=24, interval_hours=6
        )

        assert isinstance(trend_data, list)
        assert len(trend_data) > 0

        for point in trend_data:
            assert "timestamp" in point
            assert "accuracy_rate" in point
            assert "error_minutes" in point
            assert "total_predictions" in point

    @pytest.mark.asyncio
    async def test_performance_stats(self, validator):
        """Test getting performance statistics."""
        await self._create_sample_validation_data(validator)

        stats = await validator.get_performance_stats()

        assert isinstance(stats, dict)
        assert "total_predictions" in stats
        assert "records_in_memory" in stats
        assert "validation_rate_percent" in stats
        assert "memory_usage_percent" in stats
        assert "background_tasks_running" in stats

    @pytest.mark.asyncio
    async def test_background_task_lifecycle(self, validator):
        """Test background task start/stop lifecycle."""
        # Start background tasks
        await validator.start_background_tasks()

        assert len(validator._background_tasks) > 0

        # Stop background tasks
        await validator.stop_background_tasks()

        assert len(validator._background_tasks) == 0

    def test_metrics_caching(self, validator):
        """Test metrics caching functionality."""
        cache_key = "test_room_lstm_24"

        # Cache should be empty initially
        assert not validator._is_metrics_cache_valid(cache_key)

        # Cache some metrics
        test_metrics = AccuracyMetrics(total_predictions=100, accuracy_rate=85.0)
        validator._cache_metrics(cache_key, test_metrics)

        # Cache should now be valid
        assert validator._is_metrics_cache_valid(cache_key)

        # Invalidate cache
        validator._invalidate_metrics_cache("test_room", "lstm")

        # Cache should be invalidated
        assert not validator._is_metrics_cache_valid(cache_key)

    async def _create_sample_validation_data(
        self, validator, room_id="test_room", model_type=ModelType.LSTM, count=10
    ):
        """Helper method to create sample validation data."""
        for i in range(count):
            predicted_time = datetime.now(UTC) + timedelta(minutes=30 + i * 5)

            # Record prediction
            prediction_id = await validator.record_prediction(
                room_id=room_id,
                predicted_time=predicted_time,
                confidence=0.7 + (i % 3) * 0.1,
                model_type=model_type,
                transition_type="occupied" if i % 2 == 0 else "vacant",
            )

            # Validate some predictions
            if i % 2 == 0:  # Validate every other prediction
                actual_time = predicted_time + timedelta(
                    minutes=5 + i * 2
                )  # Varying error
                await validator.validate_prediction(
                    room_id=room_id,
                    actual_transition_time=actual_time,
                    transition_type="occupied" if i % 2 == 0 else "vacant",
                )


class TestValidationError:
    """Test ValidationError exception functionality."""

    def test_validation_error_creation(self):
        """Test creating validation error."""
        error = ValidationError("Test validation error")

        assert "Test validation error" in str(error)
        assert error.error_code == "VALIDATION_ERROR"

    def test_validation_error_with_cause(self):
        """Test validation error with underlying cause."""
        original_error = ValueError("Original error")
        validation_error = ValidationError("Validation failed", cause=original_error)

        assert validation_error.cause == original_error


class TestValidationIntegration:
    """Test integration scenarios for validation system."""

    @pytest.mark.asyncio
    async def test_end_to_end_validation_workflow(
        self, validator, sample_prediction_result
    ):
        """Test complete end-to-end validation workflow."""
        # 1. Record prediction
        prediction_id = await validator.record_prediction(
            prediction=sample_prediction_result,
            room_id="integration_test_room",
            feature_snapshot={"temp": 23.0, "motion_count": 5},
        )

        # 2. Check pending validations
        pending = await validator.get_pending_validations(
            room_id="integration_test_room"
        )
        assert len(pending) >= 1

        # 3. Validate prediction
        actual_time = sample_prediction_result.predicted_time + timedelta(minutes=8)
        result = await validator.validate_prediction(
            room_id="integration_test_room",
            actual_transition_time=actual_time,
            transition_type="occupied",
        )

        assert result is not None
        assert result.error_minutes == 8.0

        # 4. Check accuracy metrics
        metrics = await validator.get_room_accuracy("integration_test_room")
        assert metrics.total_predictions >= 1
        assert metrics.validated_predictions >= 1

        # 5. Check validation stats
        stats = await validator.get_validation_stats()
        assert stats["total_predictions"] >= 1

    @pytest.mark.asyncio
    async def test_concurrent_validation_operations(self, validator):
        """Test concurrent validation operations."""
        # Record multiple predictions concurrently
        tasks = []
        for i in range(10):
            task = validator.record_prediction(
                room_id=f"concurrent_room_{i % 3}",
                predicted_time=datetime.now(UTC) + timedelta(minutes=30 + i),
                confidence=0.75 + (i % 2) * 0.1,
                model_type=ModelType.XGBOOST,
                transition_type="occupied",
            )
            tasks.append(task)

        prediction_ids = await asyncio.gather(*tasks)

        assert len(prediction_ids) == 10
        assert len(set(prediction_ids)) == 10  # All IDs should be unique

    @pytest.mark.asyncio
    async def test_validation_with_database_operations(self, validator):
        """Test validation with database operations (mocked)."""
        with patch.object(validator, "_store_prediction_to_db") as mock_store:
            with patch.object(validator, "_update_predictions_in_db") as mock_update:
                # Record prediction
                prediction_id = await validator.record_prediction(
                    room_id="db_test_room",
                    predicted_time=datetime.now(UTC) + timedelta(minutes=30),
                    confidence=0.80,
                    model_type=ModelType.HMM,
                    transition_type="vacant",
                )

                # Validate prediction
                actual_time = datetime.now(UTC) + timedelta(minutes=35)
                await validator.validate_prediction(
                    room_id="db_test_room",
                    actual_transition_time=actual_time,
                    transition_type="vacant",
                )

                # Database operations should have been called
                mock_store.assert_called()
                mock_update.assert_called()


if __name__ == "__main__":
    pytest.main([__file__])
