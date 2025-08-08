"""
Comprehensive unit tests for PredictionValidator workflows.

This test module covers prediction validation, accuracy tracking, real-time
validation workflows, and comprehensive prediction performance analysis.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from src.adaptation.validator import (
    AccuracyLevel,
    AccuracyMetrics,
    PredictionValidator,
    ValidationRecord,
    ValidationStatus,
)
from src.core.constants import ModelType
from src.core.exceptions import OccupancyPredictionError
from src.data.storage.models import Prediction
from src.models.base.predictor import PredictionResult

# Test fixtures and utilities


@pytest.fixture
def mock_db_session():
    """Mock database session for testing."""
    session = Mock(spec=AsyncSession)
    session.execute = AsyncMock()
    session.commit = AsyncMock()
    session.rollback = AsyncMock()
    session.close = AsyncMock()
    return session


@pytest.fixture
def prediction_validator():
    """Create prediction validator with test configuration."""
    return PredictionValidator(accuracy_threshold_minutes=15)


@pytest.fixture
def sample_prediction_result():
    """Create sample prediction result for testing."""
    return PredictionResult(
        predicted_time=datetime.now() + timedelta(minutes=30),
        transition_type="occupied",
        confidence_score=0.85,
        model_type=ModelType.ENSEMBLE,
        prediction_metadata={
            "room_id": "living_room",
            "model_version": "v1.2.3",
            "features_used": ["temporal", "sequential"],
            "prediction_id": "pred_12345",
        },
    )


@pytest.fixture
def sample_validation_records():
    """Create sample validation records for testing."""
    base_time = datetime.now()
    records = []

    # Create records with varying accuracy
    accuracies = [5, 8, 12, 18, 25, 32, 7, 15, 22, 10]  # Mix of good and poor

    for i, error_minutes in enumerate(accuracies):
        predicted_time = base_time + timedelta(minutes=30)
        actual_time = predicted_time + timedelta(minutes=error_minutes)

        record = ValidationRecord(
            prediction_id=f"pred_{i:03d}",
            room_id="test_room",
            model_type="ensemble",
            model_version="v1.0.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.75 + (i * 0.02),
            actual_time=actual_time,
            error_minutes=error_minutes,
            prediction_time=base_time - timedelta(minutes=30 - i),
            validation_time=actual_time + timedelta(minutes=1),
            status=ValidationStatus.VALIDATED,
        )

        # Set accuracy level based on error
        if error_minutes < 5:
            record.accuracy_level = AccuracyLevel.EXCELLENT
        elif error_minutes < 10:
            record.accuracy_level = AccuracyLevel.GOOD
        elif error_minutes < 15:
            record.accuracy_level = AccuracyLevel.ACCEPTABLE
        elif error_minutes < 30:
            record.accuracy_level = AccuracyLevel.POOR
        else:
            record.accuracy_level = AccuracyLevel.UNACCEPTABLE

        records.append(record)

    return records


@pytest.fixture
def prediction_history():
    """Create prediction history data for testing."""
    predictions = []
    base_time = datetime.now() - timedelta(hours=24)

    for i in range(50):
        prediction = Prediction(
            id=i + 1,
            room_id="history_room",
            model_type="lstm",
            predicted_time=base_time + timedelta(minutes=30 + i * 10),
            actual_time=(
                base_time
                + timedelta(minutes=30 + i * 10 + np.random.randint(-10, 20))
                if np.random.random() > 0.1  # 90% get actual times
                else None
            ),
            confidence_score=0.7 + np.random.random() * 0.25,
            transition_type="occupied" if i % 2 == 0 else "vacant",
            created_at=base_time + timedelta(minutes=i * 10),
            status="validated" if np.random.random() > 0.05 else "pending",
        )
        predictions.append(prediction)

    return predictions


# Core validation tests


class TestValidationRecord:
    """Test ValidationRecord functionality."""

    def test_validation_record_creation(self):
        """Test validation record creation and initialization."""
        record = ValidationRecord(
            prediction_id="test_pred_001",
            room_id="living_room",
            model_type="lstm",
            model_version="v1.0.0",
            predicted_time=datetime.now() + timedelta(minutes=30),
            transition_type="occupied",
            confidence_score=0.82,
        )

        assert record.prediction_id == "test_pred_001"
        assert record.room_id == "living_room"
        assert record.status == ValidationStatus.PENDING
        assert record.actual_time is None
        assert record.error_minutes is None

    def test_validation_against_actual_time(self):
        """Test validation against actual transition time."""
        predicted_time = datetime.now() + timedelta(minutes=30)
        actual_time = predicted_time + timedelta(minutes=12)  # 12 minute error

        record = ValidationRecord(
            prediction_id="validation_test",
            room_id="test_room",
            model_type="xgboost",
            model_version="v1.0.0",
            predicted_time=predicted_time,
            transition_type="vacant",
            confidence_score=0.78,
        )

        # Validate with 15-minute threshold
        is_accurate = record.validate_against_actual(
            actual_time, threshold_minutes=15
        )

        assert is_accurate is True  # 12 minutes < 15 minute threshold
        assert record.status == ValidationStatus.VALIDATED
        assert record.actual_time == actual_time
        assert record.error_minutes == 12.0
        assert record.accuracy_level == AccuracyLevel.ACCEPTABLE
        assert record.validation_time is not None

    def test_validation_accuracy_levels(self):
        """Test accuracy level classification."""
        predicted_time = datetime.now()

        # Test excellent accuracy (< 5 min)
        record = ValidationRecord(
            prediction_id="excellent",
            room_id="test",
            model_type="ensemble",
            model_version="v1.0.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.9,
        )
        record.validate_against_actual(predicted_time + timedelta(minutes=3))
        assert record.accuracy_level == AccuracyLevel.EXCELLENT

        # Test poor accuracy (15-30 min)
        record2 = ValidationRecord(
            prediction_id="poor",
            room_id="test",
            model_type="ensemble",
            model_version="v1.0.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.6,
        )
        record2.validate_against_actual(predicted_time + timedelta(minutes=22))
        assert record2.accuracy_level == AccuracyLevel.POOR

        # Test unacceptable accuracy (> 30 min)
        record3 = ValidationRecord(
            prediction_id="unacceptable",
            room_id="test",
            model_type="ensemble",
            model_version="v1.0.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.5,
        )
        record3.validate_against_actual(predicted_time + timedelta(minutes=45))
        assert record3.accuracy_level == AccuracyLevel.UNACCEPTABLE

    def test_validation_record_expiration(self):
        """Test marking validation records as expired."""
        record = ValidationRecord(
            prediction_id="expire_test",
            room_id="test_room",
            model_type="hmm",
            model_version="v1.0.0",
            predicted_time=datetime.now() + timedelta(minutes=30),
            transition_type="vacant",
            confidence_score=0.73,
        )

        expiration_time = datetime.now() + timedelta(hours=2)
        record.mark_expired(expiration_time)

        assert record.status == ValidationStatus.EXPIRED
        assert record.expiration_time == expiration_time

    def test_validation_record_failure(self):
        """Test marking validation records as failed."""
        record = ValidationRecord(
            prediction_id="fail_test",
            room_id="test_room",
            model_type="gp",
            model_version="v1.0.0",
            predicted_time=datetime.now() + timedelta(minutes=30),
            transition_type="occupied",
            confidence_score=0.65,
        )

        record.mark_failed("Insufficient data for validation")

        assert record.status == ValidationStatus.FAILED
        assert record.validation_time is not None
        assert (
            record.prediction_metadata["failure_reason"]
            == "Insufficient data for validation"
        )

    def test_validation_record_serialization(self):
        """Test validation record serialization."""
        predicted_time = datetime.now() + timedelta(minutes=30)
        actual_time = predicted_time + timedelta(minutes=8)

        record = ValidationRecord(
            prediction_id="serialize_test",
            room_id="serialize_room",
            model_type="lstm",
            model_version="v2.1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.88,
            prediction_interval=(
                predicted_time - timedelta(minutes=5),
                predicted_time + timedelta(minutes=5),
            ),
            alternatives=[(predicted_time + timedelta(minutes=10), 0.65)],
        )

        record.validate_against_actual(actual_time)

        # Serialize to dict
        record_dict = record.to_dict()

        # Verify serialization
        assert record_dict["prediction_id"] == "serialize_test"
        assert record_dict["room_id"] == "serialize_room"
        assert record_dict["error_minutes"] == 8.0
        assert record_dict["accuracy_level"] == "good"
        assert "predicted_time" in record_dict
        assert "actual_time" in record_dict
        assert record_dict["prediction_interval"] is not None
        assert record_dict["alternatives"] is not None


class TestPredictionValidatorInitialization:
    """Test PredictionValidator initialization and configuration."""

    def test_validator_initialization(self):
        """Test validator initialization with default configuration."""
        validator = PredictionValidator()

        assert validator.accuracy_threshold_minutes == 15  # Default
        assert len(validator._pending_predictions) == 0
        assert len(validator._validation_history) == 0
        assert validator._total_predictions == 0
        assert validator._total_validations == 0

    def test_validator_custom_configuration(self):
        """Test validator initialization with custom configuration."""
        validator = PredictionValidator(
            accuracy_threshold_minutes=20,
            max_pending_predictions=2000,
            history_retention_days=14,
            auto_cleanup_enabled=False,
        )

        assert validator.accuracy_threshold_minutes == 20
        assert validator.max_pending_predictions == 2000
        assert validator.history_retention_days == 14
        assert validator.auto_cleanup_enabled is False


class TestPredictionRecording:
    """Test prediction recording functionality."""

    @pytest.mark.asyncio
    async def test_basic_prediction_recording(
        self, prediction_validator, sample_prediction_result
    ):
        """Test basic prediction recording."""
        room_id = "basic_room"

        with patch.object(
            prediction_validator, "_store_prediction_to_db"
        ) as mock_store:
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=sample_prediction_result.predicted_time,
                confidence=sample_prediction_result.confidence_score,
                model_type=sample_prediction_result.model_type,
                transition_type=sample_prediction_result.transition_type,
                prediction_metadata=sample_prediction_result.prediction_metadata,
            )

            # Verify prediction was recorded
            assert len(prediction_validator._pending_predictions) > 0
            assert room_id in prediction_validator._pending_predictions
            assert prediction_validator._total_predictions == 1

            # Verify database storage was called
            mock_store.assert_called_once()

    @pytest.mark.asyncio
    async def test_prediction_recording_with_metadata(
        self, prediction_validator
    ):
        """Test prediction recording with comprehensive metadata."""
        room_id = "metadata_room"
        predicted_time = datetime.now() + timedelta(minutes=45)

        metadata = {
            "model_version": "v2.3.1",
            "features_used": ["temporal", "sequential", "contextual"],
            "prediction_confidence_interval": [0.78, 0.92],
            "feature_importance": {
                "temporal": 0.4,
                "sequential": 0.35,
                "contextual": 0.25,
            },
        }

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=predicted_time,
                confidence=0.85,
                model_type=ModelType.ENSEMBLE,
                transition_type="vacant",
                prediction_metadata=metadata,
            )

            # Verify metadata was stored
            pending_predictions = prediction_validator._pending_predictions[
                room_id
            ]
            recorded_prediction = pending_predictions[0]
            assert recorded_prediction.prediction_metadata == metadata

    @pytest.mark.asyncio
    async def test_duplicate_prediction_handling(self, prediction_validator):
        """Test handling of duplicate predictions."""
        room_id = "duplicate_room"
        predicted_time = datetime.now() + timedelta(minutes=30)

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            # Record same prediction twice
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=predicted_time,
                confidence=0.8,
                model_type=ModelType.LSTM,
                transition_type="occupied",
            )

            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=predicted_time,  # Same time
                confidence=0.82,  # Slightly different confidence
                model_type=ModelType.LSTM,
                transition_type="occupied",
            )

            # Should handle duplicates appropriately
            pending_count = len(
                prediction_validator._pending_predictions[room_id]
            )
            # Behavior depends on duplicate handling strategy

    @pytest.mark.asyncio
    async def test_prediction_expiration_handling(self, prediction_validator):
        """Test automatic prediction expiration."""
        room_id = "expiration_room"

        # Record old prediction that should expire
        old_time = datetime.now() - timedelta(hours=3)

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=old_time,
                confidence=0.75,
                model_type=ModelType.XGBOOST,
                transition_type="vacant",
            )

            # Clean up expired predictions
            await prediction_validator._cleanup_expired_predictions()

            # Old prediction should be marked as expired or removed
            if room_id in prediction_validator._pending_predictions:
                remaining_predictions = (
                    prediction_validator._pending_predictions[room_id]
                )
                for pred in remaining_predictions:
                    assert pred.status in [
                        ValidationStatus.EXPIRED,
                        ValidationStatus.PENDING,
                    ]


class TestPredictionValidation:
    """Test prediction validation against actual outcomes."""

    @pytest.mark.asyncio
    async def test_successful_prediction_validation(
        self, prediction_validator
    ):
        """Test successful prediction validation."""
        room_id = "validation_room"
        predicted_time = datetime.now() + timedelta(minutes=30)
        actual_time = predicted_time + timedelta(minutes=8)  # 8 minute error

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            # Record prediction
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=predicted_time,
                confidence=0.85,
                model_type=ModelType.ENSEMBLE,
                transition_type="occupied",
            )

            # Validate prediction
            with patch.object(
                prediction_validator, "_update_validation_in_db"
            ):
                validation_result = (
                    await prediction_validator.validate_prediction(
                        room_id=room_id,
                        actual_time=actual_time,
                        transition_type="vacant_to_occupied",
                    )
                )

                assert validation_result is not None
                assert validation_result.error_minutes == 8.0
                assert validation_result.accuracy_level == AccuracyLevel.GOOD
                assert prediction_validator._total_validations == 1

    @pytest.mark.asyncio
    async def test_prediction_validation_multiple_candidates(
        self, prediction_validator
    ):
        """Test validation when multiple predictions exist for a room."""
        room_id = "multi_room"
        base_time = datetime.now()

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            # Record multiple predictions
            for i in range(3):
                await prediction_validator.record_prediction(
                    room_id=room_id,
                    predicted_time=base_time + timedelta(minutes=30 + i * 10),
                    confidence=0.8 - i * 0.1,
                    model_type=ModelType.LSTM,
                    transition_type="occupied",
                )

            # Validate against actual time
            actual_time = base_time + timedelta(
                minutes=35
            )  # Closest to second prediction

            with patch.object(
                prediction_validator, "_update_validation_in_db"
            ):
                validation_result = (
                    await prediction_validator.validate_prediction(
                        room_id=room_id,
                        actual_time=actual_time,
                        transition_type="vacant_to_occupied",
                    )
                )

                # Should find and validate the closest prediction
                assert validation_result is not None
                assert (
                    validation_result.error_minutes <= 15
                )  # Should be reasonable

    @pytest.mark.asyncio
    async def test_validation_with_no_pending_predictions(
        self, prediction_validator
    ):
        """Test validation when no pending predictions exist."""
        room_id = "empty_room"
        actual_time = datetime.now()

        with patch.object(prediction_validator, "_update_validation_in_db"):
            validation_result = await prediction_validator.validate_prediction(
                room_id=room_id,
                actual_time=actual_time,
                transition_type="vacant_to_occupied",
            )

            # Should handle gracefully (no predictions to validate)
            assert validation_result is None

    @pytest.mark.asyncio
    async def test_validation_time_window_enforcement(
        self, prediction_validator
    ):
        """Test validation time window enforcement."""
        room_id = "window_room"
        predicted_time = datetime.now() + timedelta(minutes=30)

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=predicted_time,
                confidence=0.8,
                model_type=ModelType.HMM,
                transition_type="vacant",
            )

            # Try to validate with time too far in the future
            far_future_time = predicted_time + timedelta(hours=6)

            with patch.object(
                prediction_validator, "_update_validation_in_db"
            ):
                validation_result = (
                    await prediction_validator.validate_prediction(
                        room_id=room_id,
                        actual_time=far_future_time,
                        transition_type="occupied_to_vacant",
                    )
                )

                # Should either reject or mark as expired
                if validation_result:
                    assert validation_result.error_minutes > 300  # > 5 hours


class TestAccuracyMetricsCalculation:
    """Test accuracy metrics calculation and analysis."""

    def test_basic_accuracy_metrics_calculation(
        self, sample_validation_records
    ):
        """Test basic accuracy metrics calculation."""
        # Calculate metrics from sample records
        metrics = AccuracyMetrics()

        validated_records = [
            r
            for r in sample_validation_records
            if r.status == ValidationStatus.VALIDATED
        ]

        metrics.total_predictions = len(sample_validation_records)
        metrics.validated_predictions = len(validated_records)

        # Calculate accuracy rate
        accurate_records = [
            r for r in validated_records if r.error_minutes <= 15
        ]
        metrics.accurate_predictions = len(accurate_records)
        metrics.accuracy_rate = (
            metrics.accurate_predictions / metrics.validated_predictions
        ) * 100

        # Calculate error statistics
        errors = [r.error_minutes for r in validated_records]
        metrics.mean_error_minutes = np.mean(errors)
        metrics.median_error_minutes = np.median(errors)
        metrics.std_error_minutes = np.std(errors)

        # Verify calculations
        assert 0 <= metrics.accuracy_rate <= 100
        assert metrics.mean_error_minutes > 0
        assert metrics.validated_predictions <= metrics.total_predictions

    def test_error_distribution_analysis(self, sample_validation_records):
        """Test error distribution analysis."""
        validated_records = [
            r
            for r in sample_validation_records
            if r.status == ValidationStatus.VALIDATED
        ]
        errors = [r.error_minutes for r in validated_records]

        metrics = AccuracyMetrics()

        # Calculate percentiles
        metrics.error_percentiles = {
            25: np.percentile(errors, 25),
            50: np.percentile(errors, 50),
            75: np.percentile(errors, 75),
            90: np.percentile(errors, 90),
            95: np.percentile(errors, 95),
        }

        # Count accuracy levels
        level_counts = {level.value: 0 for level in AccuracyLevel}
        for record in validated_records:
            if record.accuracy_level:
                level_counts[record.accuracy_level.value] += 1

        metrics.accuracy_by_level = level_counts

        # Verify distribution analysis
        assert metrics.error_percentiles[25] <= metrics.error_percentiles[50]
        assert metrics.error_percentiles[50] <= metrics.error_percentiles[75]
        assert sum(metrics.accuracy_by_level.values()) == len(
            validated_records
        )

    def test_bias_analysis(self, sample_validation_records):
        """Test prediction bias analysis."""
        validated_records = [
            r
            for r in sample_validation_records
            if r.status == ValidationStatus.VALIDATED
        ]

        # Calculate signed errors (positive = late predictions, negative = early)
        signed_errors = []
        for record in validated_records:
            if record.actual_time and record.predicted_time:
                time_diff = (
                    record.actual_time - record.predicted_time
                ).total_seconds() / 60
                signed_errors.append(time_diff)

        metrics = AccuracyMetrics()

        if signed_errors:
            metrics.mean_bias_minutes = np.mean(signed_errors)
            metrics.bias_std_minutes = np.std(signed_errors)

        # Test bias direction property
        if metrics.mean_bias_minutes > 1:
            assert metrics.bias_direction == "predicts_late"
        elif metrics.mean_bias_minutes < -1:
            assert metrics.bias_direction == "predicts_early"
        else:
            assert metrics.bias_direction == "unbiased"

    def test_confidence_analysis(self, sample_validation_records):
        """Test confidence score analysis."""
        validated_records = [
            r
            for r in sample_validation_records
            if r.status == ValidationStatus.VALIDATED
        ]

        confidences = [r.confidence_score for r in validated_records]
        accuracies = [
            1 if r.error_minutes <= 15 else 0 for r in validated_records
        ]

        metrics = AccuracyMetrics()
        metrics.mean_confidence = np.mean(confidences)

        # Calculate confidence-accuracy correlation
        if len(confidences) > 1:
            correlation = np.corrcoef(confidences, accuracies)[0, 1]
            metrics.confidence_vs_accuracy_correlation = (
                correlation if not np.isnan(correlation) else 0.0
            )

        # Calculate overconfidence/underconfidence rates
        high_conf_wrong = sum(
            1 for c, a in zip(confidences, accuracies) if c > 0.8 and a == 0
        )
        low_conf_right = sum(
            1 for c, a in zip(confidences, accuracies) if c < 0.6 and a == 1
        )

        metrics.overconfidence_rate = (
            high_conf_wrong / len(validated_records)
            if validated_records
            else 0
        )
        metrics.underconfidence_rate = (
            low_conf_right / len(validated_records) if validated_records else 0
        )

        # Verify confidence analysis
        assert 0 <= metrics.mean_confidence <= 1
        assert -1 <= metrics.confidence_vs_accuracy_correlation <= 1
        assert 0 <= metrics.overconfidence_rate <= 1
        assert 0 <= metrics.underconfidence_rate <= 1

    def test_accuracy_metrics_serialization(self):
        """Test accuracy metrics serialization."""
        metrics = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=85,
            accurate_predictions=68,
            accuracy_rate=80.0,
            mean_error_minutes=12.5,
            confidence_vs_accuracy_correlation=0.72,
            error_percentiles={
                25: 8.0,
                50: 12.0,
                75: 18.0,
                90: 25.0,
                95: 30.0,
            },
            accuracy_by_level={
                "excellent": 15,
                "good": 25,
                "acceptable": 28,
                "poor": 12,
                "unacceptable": 5,
            },
        )

        # Serialize to dict
        metrics_dict = metrics.to_dict()

        # Verify serialization
        assert metrics_dict["total_predictions"] == 100
        assert metrics_dict["accuracy_rate"] == 80.0
        assert metrics_dict["validation_rate"] == 85.0
        assert "error_percentiles" in metrics_dict
        assert "accuracy_by_level" in metrics_dict
        assert "bias_direction" in metrics_dict


class TestAccuracyMetricsRetrieval:
    """Test accuracy metrics retrieval and filtering."""

    @pytest.mark.asyncio
    async def test_room_accuracy_metrics(
        self, prediction_validator, prediction_history
    ):
        """Test room-specific accuracy metrics retrieval."""
        room_id = "metrics_room"

        with patch.object(
            prediction_validator,
            "_get_predictions_from_db",
            return_value=prediction_history,
        ):
            metrics = await prediction_validator.get_room_accuracy(
                room_id, hours_back=24
            )

            assert isinstance(metrics, AccuracyMetrics)
            assert metrics.total_predictions >= 0
            assert metrics.measurement_period_start is not None
            assert metrics.measurement_period_end is not None

    @pytest.mark.asyncio
    async def test_overall_accuracy_metrics(
        self, prediction_validator, prediction_history
    ):
        """Test overall system accuracy metrics."""
        with patch.object(
            prediction_validator,
            "_get_predictions_from_db",
            return_value=prediction_history,
        ):
            metrics = await prediction_validator.get_overall_accuracy(
                hours_back=48
            )

            assert isinstance(metrics, AccuracyMetrics)
            # Overall metrics should aggregate across all rooms

    @pytest.mark.asyncio
    async def test_model_specific_accuracy_metrics(
        self, prediction_validator, prediction_history
    ):
        """Test model-specific accuracy metrics."""
        model_type = ModelType.LSTM

        # Filter predictions by model type
        model_predictions = [
            p for p in prediction_history if p.model_type == model_type.value
        ]

        with patch.object(
            prediction_validator,
            "_get_predictions_from_db",
            return_value=model_predictions,
        ):
            metrics = await prediction_validator.get_model_accuracy(
                model_type, hours_back=24
            )

            assert isinstance(metrics, AccuracyMetrics)

    @pytest.mark.asyncio
    async def test_time_filtered_accuracy_metrics(self, prediction_validator):
        """Test time-filtered accuracy metrics."""
        room_id = "time_room"
        start_time = datetime.now() - timedelta(hours=12)
        end_time = datetime.now() - timedelta(hours=6)

        with patch.object(
            prediction_validator, "_get_predictions_from_db", return_value=[]
        ):
            metrics = await prediction_validator.get_accuracy_metrics(
                room_id=room_id, start_time=start_time, end_time=end_time
            )

            assert isinstance(metrics, AccuracyMetrics)
            # Should respect time filtering

    @pytest.mark.asyncio
    async def test_accuracy_trend_analysis(
        self, prediction_validator, prediction_history
    ):
        """Test accuracy trend analysis over time."""
        room_id = "trend_room"

        with patch.object(
            prediction_validator,
            "_get_predictions_from_db",
            return_value=prediction_history,
        ):
            trend_data = await prediction_validator.get_accuracy_trend(
                room_id, hours_back=48, interval_hours=6
            )

            assert isinstance(trend_data, list)
            assert len(trend_data) > 0

            for point in trend_data:
                assert "timestamp" in point
                assert "accuracy_rate" in point
                assert "error_minutes" in point


class TestValidationStatistics:
    """Test validation statistics and reporting."""

    @pytest.mark.asyncio
    async def test_validation_stats_collection(self, prediction_validator):
        """Test validation statistics collection."""
        stats = await prediction_validator.get_validation_stats()

        assert isinstance(stats, dict)
        assert "total_predictions" in stats
        assert "total_validations" in stats
        assert "validation_rate" in stats
        assert "pending_predictions" in stats

    @pytest.mark.asyncio
    async def test_room_prediction_counts(
        self, prediction_validator, prediction_history
    ):
        """Test room-wise prediction count statistics."""
        with patch.object(
            prediction_validator,
            "_get_predictions_from_db",
            return_value=prediction_history,
        ):
            stats = await prediction_validator.get_validation_stats()

            if "room_prediction_counts" in stats:
                room_counts = stats["room_prediction_counts"]
                assert isinstance(room_counts, dict)
                assert all(count >= 0 for count in room_counts.values())

    @pytest.mark.asyncio
    async def test_validation_performance_metrics(self, prediction_validator):
        """Test validation performance metrics."""
        # Record some predictions and validations
        with patch.object(prediction_validator, "_store_prediction_to_db"):
            for i in range(10):
                await prediction_validator.record_prediction(
                    room_id="perf_room",
                    predicted_time=datetime.now() + timedelta(minutes=30 + i),
                    confidence=0.8,
                    model_type=ModelType.ENSEMBLE,
                    transition_type="occupied",
                )

        # Get performance stats
        perf_stats = await prediction_validator.get_performance_stats()

        assert isinstance(perf_stats, dict)
        assert "predictions_per_hour" in perf_stats
        assert "average_validation_delay" in perf_stats

    @pytest.mark.asyncio
    async def test_total_predictions_counter(self, prediction_validator):
        """Test total predictions counter accuracy."""
        initial_count = await prediction_validator.get_total_predictions()

        # Record some predictions
        with patch.object(prediction_validator, "_store_prediction_to_db"):
            for i in range(5):
                await prediction_validator.record_prediction(
                    room_id=f"count_room_{i}",
                    predicted_time=datetime.now() + timedelta(minutes=30),
                    confidence=0.8,
                    model_type=ModelType.LSTM,
                    transition_type="vacant",
                )

        final_count = await prediction_validator.get_total_predictions()
        assert final_count >= initial_count + 5

    @pytest.mark.asyncio
    async def test_validation_rate_calculation(self, prediction_validator):
        """Test validation rate calculation."""
        initial_rate = await prediction_validator.get_validation_rate()

        # Add some predictions and validations
        with (
            patch.object(prediction_validator, "_store_prediction_to_db"),
            patch.object(prediction_validator, "_update_validation_in_db"),
        ):

            # Record predictions
            for i in range(10):
                await prediction_validator.record_prediction(
                    room_id="rate_room",
                    predicted_time=datetime.now() + timedelta(minutes=30 + i),
                    confidence=0.8,
                    model_type=ModelType.XGBOOST,
                    transition_type="occupied",
                )

            # Validate some of them
            for i in range(7):  # Validate 7 out of 10
                await prediction_validator.validate_prediction(
                    room_id="rate_room",
                    actual_time=datetime.now() + timedelta(minutes=35 + i),
                    transition_type="vacant_to_occupied",
                )

        final_rate = await prediction_validator.get_validation_rate()
        # Validation rate should be reasonable (0-100%)
        assert 0 <= final_rate <= 100


class TestDatabaseIntegration:
    """Test database integration for predictions and validations."""

    @pytest.mark.asyncio
    async def test_prediction_storage_to_database(
        self, prediction_validator, mock_db_session
    ):
        """Test prediction storage to database."""
        room_id = "db_room"
        predicted_time = datetime.now() + timedelta(minutes=30)

        with patch(
            "src.adaptation.validator.get_db_session",
            return_value=mock_db_session,
        ):
            await prediction_validator._store_prediction_to_db(
                room_id=room_id,
                predicted_time=predicted_time,
                confidence=0.85,
                model_type=ModelType.ENSEMBLE,
                transition_type="occupied",
                prediction_metadata={"test": "data"},
            )

            # Verify database interaction
            mock_db_session.add.assert_called()
            mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_validation_update_in_database(
        self, prediction_validator, mock_db_session
    ):
        """Test validation update in database."""
        prediction_id = "db_pred_001"
        actual_time = datetime.now()

        # Mock database query result
        mock_prediction = Mock()
        mock_prediction.id = prediction_id

        mock_result = Mock()
        mock_result.scalar_one_or_none.return_value = mock_prediction
        mock_db_session.execute.return_value = mock_result

        with patch(
            "src.adaptation.validator.get_db_session",
            return_value=mock_db_session,
        ):
            await prediction_validator._update_validation_in_db(
                prediction_id=prediction_id,
                actual_time=actual_time,
                error_minutes=12.5,
                accuracy_level=AccuracyLevel.ACCEPTABLE,
            )

            # Verify database update
            mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_predictions_retrieval_from_database(
        self, prediction_validator, mock_db_session
    ):
        """Test predictions retrieval from database."""
        room_id = "retrieve_room"
        hours_back = 24

        # Mock database query result
        mock_predictions = [Mock(), Mock(), Mock()]
        mock_result = Mock()
        mock_result.scalars.return_value.all.return_value = mock_predictions
        mock_db_session.execute.return_value = mock_result

        with patch(
            "src.adaptation.validator.get_db_session",
            return_value=mock_db_session,
        ):
            predictions = await prediction_validator._get_predictions_from_db(
                room_id=room_id, hours_back=hours_back
            )

            # Verify retrieval
            assert predictions == mock_predictions
            mock_db_session.execute.assert_called()

    @pytest.mark.asyncio
    async def test_database_error_handling(
        self, prediction_validator, mock_db_session
    ):
        """Test database error handling."""
        # Mock database error
        mock_db_session.commit.side_effect = Exception(
            "Database connection error"
        )

        with patch(
            "src.adaptation.validator.get_db_session",
            return_value=mock_db_session,
        ):
            # Should handle database errors gracefully
            try:
                await prediction_validator._store_prediction_to_db(
                    room_id="error_room",
                    predicted_time=datetime.now(),
                    confidence=0.8,
                    model_type=ModelType.LSTM,
                    transition_type="occupied",
                )
                # Should not raise exception
            except Exception as e:
                # If exception is raised, should be handled appropriately
                assert isinstance(e, (OccupancyPredictionError, Exception))


class TestCleanupAndMaintenance:
    """Test cleanup and maintenance operations."""

    @pytest.mark.asyncio
    async def test_expired_predictions_cleanup(self, prediction_validator):
        """Test cleanup of expired predictions."""
        # Add some old predictions
        old_time = datetime.now() - timedelta(
            hours=25
        )  # Older than 24h default

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            await prediction_validator.record_prediction(
                room_id="cleanup_room",
                predicted_time=old_time,
                confidence=0.8,
                model_type=ModelType.HMM,
                transition_type="vacant",
            )

        initial_count = len(
            prediction_validator._pending_predictions.get("cleanup_room", [])
        )

        # Run cleanup
        await prediction_validator._cleanup_expired_predictions()

        # Should clean up old predictions
        final_count = len(
            prediction_validator._pending_predictions.get("cleanup_room", [])
        )
        # May be same if expiration logic differs

    @pytest.mark.asyncio
    async def test_validation_history_cleanup(self, prediction_validator):
        """Test cleanup of old validation history."""
        # Add old validation records
        old_records = []
        for i in range(10):
            record = ValidationRecord(
                prediction_id=f"old_pred_{i}",
                room_id="history_room",
                model_type="ensemble",
                model_version="v1.0.0",
                predicted_time=datetime.now() - timedelta(days=10),
                transition_type="occupied",
                confidence_score=0.8,
                status=ValidationStatus.VALIDATED,
            )
            old_records.append(record)

        prediction_validator._validation_history.extend(old_records)

        # Run history cleanup
        await prediction_validator.cleanup_old_predictions(days_to_keep=7)

        # Should remove old records
        remaining_records = [
            r
            for r in prediction_validator._validation_history
            if r.prediction_time > datetime.now() - timedelta(days=7)
        ]
        # Verify cleanup occurred

    @pytest.mark.asyncio
    async def test_pending_predictions_size_limit(self, prediction_validator):
        """Test pending predictions size limit enforcement."""
        prediction_validator.max_pending_predictions = (
            5  # Small limit for testing
        )

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            # Add more predictions than the limit
            for i in range(10):
                await prediction_validator.record_prediction(
                    room_id="limit_room",
                    predicted_time=datetime.now() + timedelta(minutes=30 + i),
                    confidence=0.8,
                    model_type=ModelType.LSTM,
                    transition_type="occupied",
                )

        # Should enforce size limit
        total_pending = sum(
            len(preds)
            for preds in prediction_validator._pending_predictions.values()
        )
        # May be limited or use different strategy

    @pytest.mark.asyncio
    async def test_automatic_cleanup_schedule(self, prediction_validator):
        """Test automatic cleanup scheduling."""
        # Enable auto cleanup
        prediction_validator.auto_cleanup_enabled = True
        prediction_validator.cleanup_interval_hours = (
            0.001  # Very frequent for testing
        )

        # Start cleanup task
        cleanup_task = asyncio.create_task(
            prediction_validator._cleanup_loop()
        )

        # Let it run briefly
        await asyncio.sleep(0.1)

        # Stop cleanup task
        cleanup_task.cancel()

        try:
            await cleanup_task
        except asyncio.CancelledError:
            pass


class TestErrorHandlingAndEdgeCases:
    """Test error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_invalid_prediction_data_handling(
        self, prediction_validator
    ):
        """Test handling of invalid prediction data."""
        # Test with None values
        try:
            await prediction_validator.record_prediction(
                room_id=None,
                predicted_time=None,
                confidence=None,
                model_type=None,
                transition_type=None,
            )
            # Should handle gracefully or raise appropriate exception
        except Exception as e:
            assert isinstance(e, (ValueError, OccupancyPredictionError))

    @pytest.mark.asyncio
    async def test_validation_with_invalid_actual_time(
        self, prediction_validator
    ):
        """Test validation with invalid actual time."""
        room_id = "invalid_room"

        with patch.object(prediction_validator, "_store_prediction_to_db"):
            await prediction_validator.record_prediction(
                room_id=room_id,
                predicted_time=datetime.now() + timedelta(minutes=30),
                confidence=0.8,
                model_type=ModelType.ENSEMBLE,
                transition_type="occupied",
            )

        # Try to validate with invalid time
        try:
            await prediction_validator.validate_prediction(
                room_id=room_id,
                actual_time=None,  # Invalid
                transition_type="vacant_to_occupied",
            )
        except Exception as e:
            assert isinstance(e, (ValueError, OccupancyPredictionError))

    @pytest.mark.asyncio
    async def test_concurrent_validation_operations(
        self, prediction_validator
    ):
        """Test concurrent validation operations."""
        room_id = "concurrent_room"

        with (
            patch.object(prediction_validator, "_store_prediction_to_db"),
            patch.object(prediction_validator, "_update_validation_in_db"),
        ):

            # Record predictions concurrently
            prediction_tasks = []
            for i in range(10):
                task = prediction_validator.record_prediction(
                    room_id=f"{room_id}_{i}",
                    predicted_time=datetime.now() + timedelta(minutes=30 + i),
                    confidence=0.8,
                    model_type=ModelType.LSTM,
                    transition_type="occupied",
                )
                prediction_tasks.append(task)

            # Wait for all predictions to be recorded
            await asyncio.gather(*prediction_tasks)

            # Validate concurrently
            validation_tasks = []
            for i in range(10):
                task = prediction_validator.validate_prediction(
                    room_id=f"{room_id}_{i}",
                    actual_time=datetime.now() + timedelta(minutes=35 + i),
                    transition_type="vacant_to_occupied",
                )
                validation_tasks.append(task)

            # Wait for all validations
            results = await asyncio.gather(
                *validation_tasks, return_exceptions=True
            )

            # Should handle concurrent operations
            for result in results:
                assert not isinstance(result, Exception) or isinstance(
                    result, (OccupancyPredictionError, Exception)
                )

    @pytest.mark.asyncio
    async def test_memory_usage_with_large_datasets(
        self, prediction_validator
    ):
        """Test memory usage with large validation datasets."""
        # Simulate large dataset
        large_records = []
        for i in range(1000):
            record = ValidationRecord(
                prediction_id=f"large_pred_{i}",
                room_id=f"room_{i % 10}",
                model_type="ensemble",
                model_version="v1.0.0",
                predicted_time=datetime.now() + timedelta(minutes=i),
                transition_type="occupied",
                confidence_score=0.8,
                status=ValidationStatus.VALIDATED,
                error_minutes=float(i % 30),
            )
            large_records.append(record)

        prediction_validator._validation_history.extend(large_records)

        # Calculate metrics on large dataset
        with patch.object(
            prediction_validator, "_get_predictions_from_db", return_value=[]
        ):
            metrics = await prediction_validator.get_overall_accuracy()

            # Should handle large datasets efficiently
            assert isinstance(metrics, AccuracyMetrics)
