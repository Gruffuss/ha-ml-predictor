"""
Comprehensive tests for prediction accuracy validation algorithms and workflows.

This test suite validates the core prediction accuracy calculation methods,
accuracy classification systems, and real-time accuracy tracking workflows
used in the occupancy prediction system.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
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
class AccuracyTestCase:
    """Test case for accuracy validation scenarios."""

    name: str
    predicted_time: datetime
    actual_time: datetime
    expected_error_minutes: float
    expected_accuracy_level: AccuracyLevel
    expected_is_accurate: bool
    threshold_minutes: int = 15


class TestPredictionAccuracyCalculation:
    """Test core prediction accuracy calculation algorithms."""

    def test_accuracy_error_calculation_precise(self):
        """Test accuracy error calculation with precise timing."""
        predicted_time = datetime(2024, 1, 15, 14, 30, 0)
        actual_time = datetime(2024, 1, 15, 14, 35, 30)  # 5.5 minutes later

        record = ValidationRecord(
            prediction_id="test_001",
            room_id="living_room",
            model_type="LSTM",
            model_version="1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.85,
        )

        is_accurate = record.validate_against_actual(actual_time, threshold_minutes=15)

        assert record.error_minutes == pytest.approx(5.5, abs=0.1)
        assert record.accuracy_level == AccuracyLevel.GOOD
        assert is_accurate is True
        assert record.status == ValidationStatus.VALIDATED

    def test_accuracy_error_calculation_negative_time_diff(self):
        """Test accuracy calculation when prediction is later than actual."""
        predicted_time = datetime(2024, 1, 15, 14, 35, 0)
        actual_time = datetime(2024, 1, 15, 14, 30, 0)  # 5 minutes earlier

        record = ValidationRecord(
            prediction_id="test_002",
            room_id="bedroom",
            model_type="XGBoost",
            model_version="2.1",
            predicted_time=predicted_time,
            transition_type="vacant",
            confidence_score=0.92,
        )

        is_accurate = record.validate_against_actual(actual_time, threshold_minutes=10)

        assert record.error_minutes == pytest.approx(5.0, abs=0.1)
        assert record.accuracy_level == AccuracyLevel.GOOD
        assert is_accurate is True

    def test_accuracy_level_classification_comprehensive(self):
        """Test comprehensive accuracy level classification system."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)

        test_cases = [
            AccuracyTestCase(
                name="excellent_accuracy",
                predicted_time=base_time,
                actual_time=base_time + timedelta(minutes=3),
                expected_error_minutes=3.0,
                expected_accuracy_level=AccuracyLevel.EXCELLENT,
                expected_is_accurate=True,
            ),
            AccuracyTestCase(
                name="good_accuracy",
                predicted_time=base_time,
                actual_time=base_time + timedelta(minutes=7),
                expected_error_minutes=7.0,
                expected_accuracy_level=AccuracyLevel.GOOD,
                expected_is_accurate=True,
            ),
            AccuracyTestCase(
                name="acceptable_accuracy",
                predicted_time=base_time,
                actual_time=base_time + timedelta(minutes=12),
                expected_error_minutes=12.0,
                expected_accuracy_level=AccuracyLevel.ACCEPTABLE,
                expected_is_accurate=True,
            ),
            AccuracyTestCase(
                name="poor_accuracy",
                predicted_time=base_time,
                actual_time=base_time + timedelta(minutes=20),
                expected_error_minutes=20.0,
                expected_accuracy_level=AccuracyLevel.POOR,
                expected_is_accurate=False,
            ),
            AccuracyTestCase(
                name="unacceptable_accuracy",
                predicted_time=base_time,
                actual_time=base_time + timedelta(minutes=45),
                expected_error_minutes=45.0,
                expected_accuracy_level=AccuracyLevel.UNACCEPTABLE,
                expected_is_accurate=False,
            ),
        ]

        for case in test_cases:
            record = ValidationRecord(
                prediction_id=f"test_{case.name}",
                room_id="test_room",
                model_type="TestModel",
                model_version="1.0",
                predicted_time=case.predicted_time,
                transition_type="occupied",
                confidence_score=0.8,
            )

            is_accurate = record.validate_against_actual(
                case.actual_time, threshold_minutes=case.threshold_minutes
            )

            assert record.error_minutes == pytest.approx(
                case.expected_error_minutes, abs=0.1
            )
            assert record.accuracy_level == case.expected_accuracy_level
            assert is_accurate == case.expected_is_accurate

    def test_accuracy_threshold_boundary_conditions(self):
        """Test accuracy threshold boundary conditions."""
        base_time = datetime(2024, 1, 15, 14, 30, 0)

        # Test exactly at threshold
        record = ValidationRecord(
            prediction_id="threshold_exact",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=base_time,
            transition_type="occupied",
            confidence_score=0.8,
        )

        actual_time = base_time + timedelta(minutes=15)  # Exactly 15 minutes
        is_accurate = record.validate_against_actual(actual_time, threshold_minutes=15)

        assert record.error_minutes == 15.0
        assert is_accurate is True  # Should be inclusive

        # Test just over threshold
        record2 = ValidationRecord(
            prediction_id="threshold_over",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=base_time,
            transition_type="occupied",
            confidence_score=0.8,
        )

        actual_time = base_time + timedelta(
            minutes=15, seconds=1
        )  # Just over 15 minutes
        is_accurate = record2.validate_against_actual(actual_time, threshold_minutes=15)

        assert record2.error_minutes > 15.0
        assert is_accurate is False

    def test_validation_status_transitions(self):
        """Test validation status transitions and error handling."""
        record = ValidationRecord(
            prediction_id="status_test",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=datetime(2024, 1, 15, 14, 30, 0),
            transition_type="occupied",
            confidence_score=0.8,
        )

        # Initial status should be PENDING
        assert record.status == ValidationStatus.PENDING

        # Validate successfully
        actual_time = datetime(2024, 1, 15, 14, 35, 0)
        is_accurate = record.validate_against_actual(actual_time)

        assert record.status == ValidationStatus.VALIDATED
        assert record.validation_time is not None

        # Try to validate again - should raise error
        with pytest.raises(ValidationError) as exc_info:
            record.validate_against_actual(actual_time)

        assert "already validated" in str(exc_info.value)

    def test_record_expiration_workflow(self):
        """Test prediction record expiration workflow."""
        record = ValidationRecord(
            prediction_id="expire_test",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=datetime(2024, 1, 15, 14, 30, 0),
            transition_type="occupied",
            confidence_score=0.8,
        )

        # Mark as expired
        expiration_time = datetime.utcnow()
        record.mark_expired(expiration_time)

        assert record.status == ValidationStatus.EXPIRED
        assert record.expiration_time == expiration_time

        # Try to validate expired record - should raise error
        with pytest.raises(ValidationError) as exc_info:
            actual_time = datetime(2024, 1, 15, 14, 35, 0)
            record.validate_against_actual(actual_time)

        assert "already expired" in str(exc_info.value)

    def test_record_failure_handling(self):
        """Test prediction record failure handling."""
        record = ValidationRecord(
            prediction_id="failure_test",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=datetime(2024, 1, 15, 14, 30, 0),
            transition_type="occupied",
            confidence_score=0.8,
        )

        # Mark as failed
        failure_reason = "Sensor malfunction detected"
        record.mark_failed(failure_reason)

        assert record.status == ValidationStatus.FAILED
        assert record.validation_time is not None
        assert record.prediction_metadata["failure_reason"] == failure_reason


class TestPredictionValidatorAccuracy:
    """Test PredictionValidator accuracy tracking workflows."""

    @pytest.fixture
    def validator(self):
        """Create validator with test configuration."""
        return PredictionValidator(
            accuracy_threshold_minutes=15,
            max_validation_delay_hours=6,
            max_memory_records=100,
        )

    @pytest.fixture
    def sample_prediction(self):
        """Create sample prediction result."""
        return PredictionResult(
            predicted_time=datetime(2024, 1, 15, 14, 30, 0),
            transition_type="occupied",
            confidence_score=0.85,
            model_type="LSTM",
            model_version="1.0",
            prediction_interval=(
                datetime(2024, 1, 15, 14, 25, 0),
                datetime(2024, 1, 15, 14, 35, 0),
            ),
            alternatives=[
                (datetime(2024, 1, 15, 14, 32, 0), 0.12),
                (datetime(2024, 1, 15, 14, 28, 0), 0.08),
            ],
        )

    @pytest.mark.asyncio
    async def test_prediction_recording_workflow(self, validator, sample_prediction):
        """Test complete prediction recording workflow."""
        with patch("src.adaptation.validator.get_db_session"):
            # Record prediction
            prediction_id = await validator.record_prediction(
                sample_prediction,
                "living_room",
                feature_snapshot={"temperature": 22.5, "motion_count": 3},
            )

            assert prediction_id is not None
            assert "living_room" in prediction_id
            assert "LSTM" in prediction_id

            # Verify record was stored in memory
            with validator._lock:
                assert prediction_id in validator._validation_records
                record = validator._validation_records[prediction_id]

                assert record.room_id == "living_room"
                assert record.model_type == "LSTM"
                assert record.confidence_score == 0.85
                assert record.status == ValidationStatus.PENDING
                assert record.feature_snapshot["temperature"] == 22.5

    @pytest.mark.asyncio
    async def test_prediction_validation_workflow(self, validator, sample_prediction):
        """Test complete prediction validation workflow."""
        with (
            patch("src.adaptation.validator.get_db_session"),
            patch.object(validator, "_update_predictions_in_db") as mock_update,
        ):

            # Record prediction
            prediction_id = await validator.record_prediction(
                sample_prediction, "living_room"
            )

            # Validate with accurate timing
            actual_time = datetime(2024, 1, 15, 14, 33, 0)  # 3 minutes late
            validated_ids = await validator.validate_prediction(
                "living_room",
                actual_time,
                "occupied",
                max_time_window_minutes=60,
            )

            assert len(validated_ids) == 1
            assert prediction_id in validated_ids

            # Verify validation was applied
            with validator._lock:
                record = validator._validation_records[prediction_id]
                assert record.status == ValidationStatus.VALIDATED
                assert record.actual_time == actual_time
                assert record.error_minutes == 3.0
                assert record.accuracy_level == AccuracyLevel.EXCELLENT

            # Verify database update was called
            mock_update.assert_called_once()

    @pytest.mark.asyncio
    async def test_batch_prediction_validation(self, validator):
        """Test batch validation of multiple predictions."""
        with (
            patch("src.adaptation.validator.get_db_session"),
            patch.object(validator, "_update_predictions_in_db") as mock_update,
        ):

            # Create multiple predictions for the same room
            predictions = []
            for i in range(5):
                prediction = PredictionResult(
                    predicted_time=datetime(2024, 1, 15, 14, 30 + i, 0),
                    transition_type="occupied",
                    confidence_score=0.8 + i * 0.02,
                    model_type=f"Model_{i}",
                    model_version="1.0",
                )

                prediction_id = await validator.record_prediction(prediction, "bedroom")
                predictions.append(prediction_id)

            # Validate all predictions at once
            actual_time = datetime(2024, 1, 15, 14, 32, 30)  # Should match prediction 2
            validated_ids = await validator.validate_prediction(
                "bedroom",
                actual_time,
                "occupied",
                max_time_window_minutes=10,  # Narrow window
            )

            # Should validate the closest predictions within window
            assert len(validated_ids) >= 1

            # Verify database update was called with validated records
            mock_update.assert_called_once()
            call_args = mock_update.call_args[0]
            updated_records = call_args[0]
            assert len(updated_records) == len(validated_ids)

    @pytest.mark.asyncio
    async def test_validation_time_window_filtering(self, validator):
        """Test time window filtering for validation candidates."""
        with patch("src.adaptation.validator.get_db_session"):
            # Create predictions at different times
            predictions = []
            base_time = datetime(2024, 1, 15, 14, 30, 0)

            for i, offset_minutes in enumerate([-30, -10, 0, 10, 30]):
                prediction = PredictionResult(
                    predicted_time=base_time + timedelta(minutes=offset_minutes),
                    transition_type="occupied",
                    confidence_score=0.8,
                    model_type=f"Model_{i}",
                    model_version="1.0",
                )

                prediction_id = await validator.record_prediction(prediction, "kitchen")
                predictions.append(prediction_id)

            # Validate with narrow time window
            actual_time = base_time
            validated_ids = await validator.validate_prediction(
                "kitchen",
                actual_time,
                "occupied",
                max_time_window_minutes=15,  # Only middle 3 predictions should match
            )

            # Should only validate predictions within 15-minute window
            assert len(validated_ids) == 3  # -10, 0, +10 minute predictions

    @pytest.mark.asyncio
    async def test_transition_type_matching(self, validator):
        """Test that validation only matches correct transition types."""
        with patch("src.adaptation.validator.get_db_session"):
            # Create predictions for different transition types
            occupied_prediction = PredictionResult(
                predicted_time=datetime(2024, 1, 15, 14, 30, 0),
                transition_type="occupied",
                confidence_score=0.8,
                model_type="LSTM",
                model_version="1.0",
            )

            vacant_prediction = PredictionResult(
                predicted_time=datetime(2024, 1, 15, 14, 30, 0),
                transition_type="vacant",
                confidence_score=0.75,
                model_type="XGBoost",
                model_version="1.0",
            )

            occupied_id = await validator.record_prediction(
                occupied_prediction, "office"
            )
            vacant_id = await validator.record_prediction(vacant_prediction, "office")

            # Validate with "occupied" transition
            actual_time = datetime(2024, 1, 15, 14, 32, 0)
            validated_ids = await validator.validate_prediction(
                "office",
                actual_time,
                "occupied",  # Should only match occupied prediction
                max_time_window_minutes=30,
            )

            assert len(validated_ids) == 1
            assert occupied_id in validated_ids
            assert vacant_id not in validated_ids

    @pytest.mark.asyncio
    async def test_accuracy_threshold_application(self, validator):
        """Test accuracy threshold application in validation."""
        with patch("src.adaptation.validator.get_db_session"):
            # Create prediction
            prediction = PredictionResult(
                predicted_time=datetime(2024, 1, 15, 14, 30, 0),
                transition_type="occupied",
                confidence_score=0.9,
                model_type="LSTM",
                model_version="1.0",
            )

            prediction_id = await validator.record_prediction(prediction, "bathroom")

            # Validate with timing that exceeds threshold
            actual_time = datetime(2024, 1, 15, 14, 50, 0)  # 20 minutes late
            validated_ids = await validator.validate_prediction(
                "bathroom", actual_time, "occupied", max_time_window_minutes=60
            )

            assert len(validated_ids) == 1
            assert prediction_id in validated_ids

            # Check that record was marked as inaccurate
            with validator._lock:
                record = validator._validation_records[prediction_id]
                assert record.status == ValidationStatus.VALIDATED
                assert record.error_minutes == 20.0
                assert record.accuracy_level == AccuracyLevel.POOR

            # Validate against threshold in accuracy calculation
            is_accurate_by_threshold = (
                record.error_minutes <= validator.accuracy_threshold
            )
            assert is_accurate_by_threshold is False


class TestAccuracyValidationEdgeCases:
    """Test edge cases and error conditions in accuracy validation."""

    @pytest.fixture
    def validator(self):
        """Create validator for edge case testing."""
        return PredictionValidator(accuracy_threshold_minutes=10)

    def test_zero_time_difference_accuracy(self):
        """Test accuracy calculation with zero time difference."""
        predicted_time = datetime(2024, 1, 15, 14, 30, 0)
        actual_time = predicted_time  # Exactly same time

        record = ValidationRecord(
            prediction_id="zero_dif",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.95,
        )

        is_accurate = record.validate_against_actual(actual_time)

        assert record.error_minutes == 0.0
        assert record.accuracy_level == AccuracyLevel.EXCELLENT
        assert is_accurate is True

    def test_microsecond_precision_accuracy(self):
        """Test accuracy calculation with microsecond precision."""
        predicted_time = datetime(2024, 1, 15, 14, 30, 0, 123456)
        actual_time = datetime(2024, 1, 15, 14, 30, 0, 654321)

        record = ValidationRecord(
            prediction_id="microsecond_test",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.9,
        )

        is_accurate = record.validate_against_actual(actual_time)

        # Should handle microsecond differences correctly
        expected_diff_seconds = abs((actual_time - predicted_time).total_seconds())
        expected_diff_minutes = expected_diff_seconds / 60

        assert record.error_minutes == pytest.approx(expected_diff_minutes, abs=0.0001)
        assert record.accuracy_level == AccuracyLevel.EXCELLENT

    def test_large_time_difference_handling(self):
        """Test handling of very large time differences."""
        predicted_time = datetime(2024, 1, 15, 14, 30, 0)
        actual_time = datetime(2024, 1, 16, 14, 30, 0)  # 24 hours later

        record = ValidationRecord(
            prediction_id="large_dif",
            room_id="test_room",
            model_type="TestModel",
            model_version="1.0",
            predicted_time=predicted_time,
            transition_type="occupied",
            confidence_score=0.6,
        )

        is_accurate = record.validate_against_actual(actual_time, threshold_minutes=10)

        assert record.error_minutes == pytest.approx(1440.0, abs=0.1)  # 24 * 60 minutes
        assert record.accuracy_level == AccuracyLevel.UNACCEPTABLE
        assert is_accurate is False

    @pytest.mark.asyncio
    async def test_empty_room_validation_scenario(self, validator):
        """Test validation when no predictions exist for a room."""
        with patch("src.adaptation.validator.get_db_session"):
            # Try to validate for a room with no predictions
            actual_time = datetime(2024, 1, 15, 14, 30, 0)
            validated_ids = await validator.validate_prediction(
                "nonexistent_room",
                actual_time,
                "occupied",
                max_time_window_minutes=60,
            )

            assert len(validated_ids) == 0

    @pytest.mark.asyncio
    async def test_no_matching_predictions_validation(self, validator):
        """Test validation when predictions exist but don't match criteria."""
        with patch("src.adaptation.validator.get_db_session"):
            # Create prediction with different transition type
            prediction = PredictionResult(
                predicted_time=datetime(2024, 1, 15, 14, 30, 0),
                transition_type="vacant",
                confidence_score=0.8,
                model_type="LSTM",
                model_version="1.0",
            )

            await validator.record_prediction(prediction, "garage")

            # Try to validate with different transition type
            actual_time = datetime(2024, 1, 15, 14, 32, 0)
            validated_ids = await validator.validate_prediction(
                "garage",
                actual_time,
                "occupied",  # Different from prediction
                max_time_window_minutes=60,
            )

            assert len(validated_ids) == 0

    @pytest.mark.asyncio
    async def test_validation_memory_cleanup(self, validator):
        """Test validation memory cleanup under load."""
        # Set a small memory limit for testing
        validator.max_memory_records = 5

        with patch("src.adaptation.validator.get_db_session"):
            # Create more predictions than memory limit
            for i in range(10):
                prediction = PredictionResult(
                    predicted_time=datetime(2024, 1, 15, 14, 30 + i, 0),
                    transition_type="occupied",
                    confidence_score=0.8,
                    model_type=f"Model_{i}",
                    model_version="1.0",
                )

                await validator.record_prediction(prediction, f"room_{i}")

            # Verify memory limit was respected
            with validator._lock:
                assert (
                    len(validator._validation_records) <= validator.max_memory_records
                )
