"""
Advanced unit tests for logger.py with deeper coverage.
Tests performance logging, error tracking, and complex scenarios.
"""

from datetime import datetime, timedelta
import json
import logging
import sys
import time
from unittest.mock import MagicMock, Mock, call, patch

import pytest

from src.utils.logger import (
    ErrorTracker,
    LoggerManager,
    MLOperationsLogger,
    PerformanceLogger,
    StructuredFormatter,
    get_error_tracker,
    get_logger,
    get_ml_ops_logger,
    get_performance_logger,
)


class TestStructuredFormatterAdvanced:
    """Advanced tests for StructuredFormatter."""

    @pytest.fixture
    def structured_formatter(self):
        """Create StructuredFormatter instance."""
        return StructuredFormatter(include_extra=True)

    def test_structured_formatter_all_levels(self, structured_formatter):
        """Test StructuredFormatter with all log levels."""
        # Create mock log records for each level
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]

        for level in levels:
            record = logging.LogRecord(
                name="test_logger",
                level=level,
                pathname="test.py",
                lineno=42,
                msg="Test message",
                args=(),
                exc_info=None,
            )

            formatted = structured_formatter.format(record)

            # Verify JSON structure
            parsed = json.loads(formatted)
            assert parsed["level"] == logging.getLevelName(level)
            assert parsed["message"] == "Test message"
            assert parsed["logger"] == "test_logger"
            assert "timestamp" in parsed

    def test_structured_formatter_unknown_level(self, structured_formatter):
        """Test StructuredFormatter with unknown log level."""
        record = logging.LogRecord(
            name="test_logger",
            level=99,  # Unknown level
            pathname="test.py",
            lineno=42,
            msg="Unknown level message",
            args=(),
            exc_info=None,
        )

        formatted = structured_formatter.format(record)

        # Should handle unknown level gracefully
        parsed = json.loads(formatted)
        assert parsed["message"] == "Unknown level message"
        assert "level" in parsed

    def test_structured_formatter_with_exception(self, structured_formatter):
        """Test StructuredFormatter with exception information."""
        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test_logger",
            level=logging.ERROR,
            pathname="test.py",
            lineno=42,
            msg="Error with exception",
            args=(),
            exc_info=exc_info,
        )

        formatted = structured_formatter.format(record)

        # Verify JSON structure with exception
        parsed = json.loads(formatted)
        assert parsed["level"] == "ERROR"
        assert parsed["message"] == "Error with exception"
        assert "exception" in parsed
        assert "ValueError: Test exception" in parsed["exception"]

    def test_structured_formatter_without_extra(self):
        """Test StructuredFormatter without extra fields."""
        formatter_no_extra = StructuredFormatter(include_extra=False)

        record = logging.LogRecord(
            name="custom_logger",
            level=logging.INFO,
            pathname="test.py",
            lineno=42,
            msg="Custom format test",
            args=(),
            exc_info=None,
        )
        # Add extra data
        record.custom_field = "should_not_appear"

        formatted = formatter_no_extra.format(record)

        # Verify extra fields are not included
        parsed = json.loads(formatted)
        assert parsed["message"] == "Custom format test"
        assert "custom_field" not in parsed


class TestPerformanceLoggerAdvanced:
    """Advanced tests for PerformanceLogger."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def performance_logger(self, mock_logger):
        """Create PerformanceLogger with mock logger."""
        with patch("src.utils.logger.get_logger", return_value=mock_logger):
            return PerformanceLogger()

    def test_performance_logger_initialization(self, mock_logger, performance_logger):
        """Test PerformanceLogger initialization."""
        assert performance_logger.logger == mock_logger
        assert isinstance(performance_logger.operation_times, dict)
        assert isinstance(performance_logger.prediction_accuracies, dict)

    def test_log_operation_time_with_all_parameters(
        self, performance_logger, mock_logger
    ):
        """Test logging operation time with all parameters."""
        performance_logger.log_operation_time(
            operation="model_training",
            duration=125.75,
            room_id="master_bedroom",
            model_type="ensemble",
            batch_size=64,
            epochs=50,
            learning_rate=0.001,
        )

        # Verify logger was called
        mock_logger.info.assert_called_once()

        # Get the call arguments
        call_args = mock_logger.info.call_args
        message = call_args[0][0]
        extra = call_args[1]["extra"]

        # Verify message content
        assert "Operation completed" in message
        assert "model_training" in message
        assert "125.75s" in message

        # Verify extra data
        assert extra["operation"] == "model_training"
        assert extra["duration_seconds"] == 125.75
        assert extra["room_id"] == "master_bedroom"
        assert extra["model_type"] == "ensemble"
        assert extra["batch_size"] == 64
        assert extra["epochs"] == 50
        assert extra["learning_rate"] == 0.001

        # Verify operation time was stored
        assert "model_training" in performance_logger.operation_times
        assert 125.75 in performance_logger.operation_times["model_training"]

    def test_log_operation_time_minimal_parameters(
        self, performance_logger, mock_logger
    ):
        """Test logging operation time with minimal parameters."""
        performance_logger.log_operation_time(
            operation="feature_extraction", duration=0.085
        )

        # Verify logger was called
        mock_logger.info.assert_called_once()

        call_args = mock_logger.info.call_args
        extra = call_args[1]["extra"]

        # Verify minimal extra data
        assert extra["operation"] == "feature_extraction"
        assert extra["duration_seconds"] == 0.085
        assert "room_id" not in extra

        # Verify operation time was stored
        assert "feature_extraction" in performance_logger.operation_times
        assert 0.085 in performance_logger.operation_times["feature_extraction"]

    def test_log_operation_time_multiple_entries(self, performance_logger):
        """Test logging multiple operation times for same operation."""
        durations = [0.125, 0.089, 0.156, 0.098, 0.203]

        for duration in durations:
            performance_logger.log_operation_time(
                operation="prediction", duration=duration
            )

        # Verify all durations were stored
        assert "prediction" in performance_logger.operation_times
        stored_times = performance_logger.operation_times["prediction"]

        for duration in durations:
            assert duration in stored_times

        assert len(stored_times) == len(durations)

    def test_log_prediction_accuracy_with_all_parameters(
        self, performance_logger, mock_logger
    ):
        """Test logging prediction accuracy with all parameters."""
        performance_logger.log_prediction_accuracy(
            room_id="living_room",
            accuracy_minutes=12.5,
            confidence=0.87,
            prediction_type="next_occupancy",
            model_type="lstm",
            features_count=25,
            training_samples=1500,
        )

        # Verify logger was called
        mock_logger.info.assert_called_once()

        call_args = mock_logger.info.call_args
        message = call_args[0][0]
        extra = call_args[1]["extra"]

        # Verify message content
        assert "Prediction accuracy" in message
        assert "living_room" in message
        assert "12.5 minutes" in message
        assert "87.0% confidence" in message

        # Verify extra data
        assert extra["room_id"] == "living_room"
        assert extra["accuracy_minutes"] == 12.5
        assert extra["confidence"] == 0.87
        assert extra["prediction_type"] == "next_occupancy"
        assert extra["model_type"] == "lstm"
        assert extra["features_count"] == 25
        assert extra["training_samples"] == 1500

        # Verify accuracy was stored
        assert "living_room" in performance_logger.prediction_accuracies
        assert 12.5 in performance_logger.prediction_accuracies["living_room"]

    def test_log_prediction_accuracy_minimal_parameters(
        self, performance_logger, mock_logger
    ):
        """Test logging prediction accuracy with minimal parameters."""
        performance_logger.log_prediction_accuracy(
            room_id="kitchen", accuracy_minutes=8.2, confidence=0.73
        )

        call_args = mock_logger.info.call_args
        extra = call_args[1]["extra"]

        # Verify core data is present
        assert extra["room_id"] == "kitchen"
        assert extra["accuracy_minutes"] == 8.2
        assert extra["confidence"] == 0.73

        # Verify optional parameters are not present
        assert "prediction_type" not in extra
        assert "model_type" not in extra

    def test_get_average_operation_time_with_data(self, performance_logger):
        """Test getting average operation time with existing data."""
        # Add operation times
        durations = [0.125, 0.150, 0.100, 0.175, 0.130]
        for duration in durations:
            performance_logger.log_operation_time("test_op", duration)

        average = performance_logger.get_average_operation_time("test_op")
        expected_average = sum(durations) / len(durations)

        assert average == expected_average
        assert abs(average - 0.136) < 0.001  # Approximately 0.136

    def test_get_average_operation_time_no_data(self, performance_logger):
        """Test getting average operation time with no data."""
        average = performance_logger.get_average_operation_time("nonexistent_op")
        assert average is None

    def test_get_average_prediction_accuracy_with_data(self, performance_logger):
        """Test getting average prediction accuracy with existing data."""
        # Add prediction accuracies
        accuracies = [10.5, 12.0, 8.5, 15.2, 9.8]
        for accuracy in accuracies:
            performance_logger.log_prediction_accuracy(
                room_id="test_room", accuracy_minutes=accuracy, confidence=0.8
            )

        average = performance_logger.get_average_prediction_accuracy("test_room")
        expected_average = sum(accuracies) / len(accuracies)

        assert average == expected_average
        assert abs(average - 11.2) < 0.1  # Approximately 11.2

    def test_get_average_prediction_accuracy_no_data(self, performance_logger):
        """Test getting average prediction accuracy with no data."""
        average = performance_logger.get_average_prediction_accuracy("nonexistent_room")
        assert average is None

    def test_get_performance_summary(self, performance_logger):
        """Test getting comprehensive performance summary."""
        # Add various operation times
        performance_logger.log_operation_time("prediction", 0.125)
        performance_logger.log_operation_time("prediction", 0.150)
        performance_logger.log_operation_time("training", 120.5)
        performance_logger.log_operation_time("feature_extraction", 0.045)

        # Add prediction accuracies
        performance_logger.log_prediction_accuracy("room1", 10.5, 0.85)
        performance_logger.log_prediction_accuracy("room1", 12.0, 0.90)
        performance_logger.log_prediction_accuracy("room2", 8.5, 0.75)

        summary = performance_logger.get_performance_summary()

        # Verify operation times summary
        assert "operation_times" in summary
        op_times = summary["operation_times"]

        assert "prediction" in op_times
        assert op_times["prediction"]["average"] == 0.1375  # (0.125 + 0.150) / 2
        assert op_times["prediction"]["count"] == 2

        assert "training" in op_times
        assert op_times["training"]["average"] == 120.5
        assert op_times["training"]["count"] == 1

        # Verify prediction accuracies summary
        assert "prediction_accuracies" in summary
        pred_acc = summary["prediction_accuracies"]

        assert "room1" in pred_acc
        assert pred_acc["room1"]["average"] == 11.25  # (10.5 + 12.0) / 2
        assert pred_acc["room1"]["count"] == 2

        assert "room2" in pred_acc
        assert pred_acc["room2"]["average"] == 8.5
        assert pred_acc["room2"]["count"] == 1

    def test_get_performance_summary_empty(self, performance_logger):
        """Test getting performance summary with no data."""
        summary = performance_logger.get_performance_summary()

        assert summary == {"operation_times": {}, "prediction_accuracies": {}}


class TestErrorTrackerAdvanced:
    """Advanced tests for ErrorTracker."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def error_tracker(self, mock_logger):
        """Create ErrorTracker with mock logger."""
        with patch("src.utils.logger.get_logger", return_value=mock_logger):
            return ErrorTracker()

    def test_error_tracker_initialization(self, error_tracker, mock_logger):
        """Test ErrorTracker initialization."""
        assert error_tracker.logger == mock_logger
        assert isinstance(error_tracker.error_history, list)
        assert len(error_tracker.error_history) == 0

    def test_track_error_with_context(self, error_tracker, mock_logger):
        """Test tracking error with context information."""
        try:
            raise ValueError("Test error message")
        except ValueError as e:
            context = {
                "room_id": "bedroom",
                "operation": "prediction",
                "model_type": "lstm",
                "input_features": 25,
            }

            error_tracker.track_error(e, context)

        # Verify logger was called
        mock_logger.error.assert_called_once()

        call_args = mock_logger.error.call_args
        message = call_args[0][0]
        extra = call_args[1]["extra"]

        # Verify message content
        assert "Error tracked" in message
        assert "ValueError" in message
        assert "Test error message" in message

        # Verify extra data includes context
        assert extra["error_type"] == "ValueError"
        assert extra["error_message"] == "Test error message"
        assert extra["room_id"] == "bedroom"
        assert extra["operation"] == "prediction"
        assert extra["model_type"] == "lstm"
        assert extra["input_features"] == 25

        # Verify error was stored in history
        assert len(error_tracker.error_history) == 1
        stored_error = error_tracker.error_history[0]

        assert stored_error["error_type"] == "ValueError"
        assert stored_error["error_message"] == "Test error message"
        assert stored_error["context"] == context
        assert "timestamp" in stored_error
        assert isinstance(stored_error["timestamp"], datetime)

    def test_track_error_without_context(self, error_tracker, mock_logger):
        """Test tracking error without context information."""
        try:
            raise RuntimeError("Runtime error occurred")
        except RuntimeError as e:
            error_tracker.track_error(e)

        call_args = mock_logger.error.call_args
        extra = call_args[1]["extra"]

        # Verify basic error data
        assert extra["error_type"] == "RuntimeError"
        assert extra["error_message"] == "Runtime error occurred"

        # Verify no context keys are present
        context_keys = set(extra.keys()) - {"error_type", "error_message"}
        assert len(context_keys) == 0

        # Verify error was stored
        stored_error = error_tracker.error_history[0]
        assert stored_error["context"] == {}

    def test_track_error_empty_context(self, error_tracker):
        """Test tracking error with empty context."""
        try:
            raise KeyError("Missing key")
        except KeyError as e:
            error_tracker.track_error(e, {})

        stored_error = error_tracker.error_history[0]
        assert stored_error["context"] == {}

    def test_track_multiple_errors(self, error_tracker):
        """Test tracking multiple errors."""
        errors_and_contexts = [
            (ValueError("Error 1"), {"component": "A"}),
            (RuntimeError("Error 2"), {"component": "B"}),
            (TypeError("Error 3"), {"component": "C"}),
        ]

        for error, context in errors_and_contexts:
            error_tracker.track_error(error, context)

        # Verify all errors were stored
        assert len(error_tracker.error_history) == 3

        # Verify errors are stored in order
        for i, (error, context) in enumerate(errors_and_contexts):
            stored_error = error_tracker.error_history[i]
            assert stored_error["error_type"] == type(error).__name__
            assert stored_error["error_message"] == str(error)
            assert stored_error["context"] == context

    def test_get_error_summary_with_data(self, error_tracker):
        """Test getting error summary with existing data."""
        # Track various errors with timestamps spread over time
        base_time = datetime.now()

        errors_data = [
            (
                ValueError("Val error 1"),
                {"component": "A"},
                base_time - timedelta(hours=2),
            ),
            (
                ValueError("Val error 2"),
                {"component": "B"},
                base_time - timedelta(hours=1),
            ),
            (
                RuntimeError("Runtime error"),
                {"component": "A"},
                base_time - timedelta(minutes=30),
            ),
            (
                TypeError("Type error"),
                {"component": "C"},
                base_time - timedelta(minutes=10),
            ),
        ]

        # Manually add errors with specific timestamps
        for error, context, timestamp in errors_data:
            error_entry = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "timestamp": timestamp,
            }
            error_tracker.error_history.append(error_entry)

        summary = error_tracker.get_error_summary(hours=24)

        # Verify summary structure
        assert "total_errors" in summary
        assert "error_types" in summary
        assert "error_by_component" in summary
        assert "recent_errors" in summary
        assert "time_range_hours" in summary

        # Verify totals
        assert summary["total_errors"] == 4
        assert summary["time_range_hours"] == 24

        # Verify error types breakdown
        error_types = summary["error_types"]
        assert error_types["ValueError"] == 2
        assert error_types["RuntimeError"] == 1
        assert error_types["TypeError"] == 1

        # Verify component breakdown
        error_by_component = summary["error_by_component"]
        assert error_by_component["A"] == 2
        assert error_by_component["B"] == 1
        assert error_by_component["C"] == 1

        # Verify recent errors (should be all 4 within 24 hours)
        assert len(summary["recent_errors"]) == 4

    def test_get_error_summary_time_filtering(self, error_tracker):
        """Test error summary time filtering."""
        base_time = datetime.now()

        # Add errors: some recent, some old
        recent_errors = [
            (ValueError("Recent 1"), {}, base_time - timedelta(minutes=30)),
            (RuntimeError("Recent 2"), {}, base_time - timedelta(minutes=10)),
        ]

        old_errors = [
            (TypeError("Old 1"), {}, base_time - timedelta(hours=25)),
            (KeyError("Old 2"), {}, base_time - timedelta(hours=30)),
        ]

        all_errors = recent_errors + old_errors

        for error, context, timestamp in all_errors:
            error_entry = {
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context,
                "timestamp": timestamp,
            }
            error_tracker.error_history.append(error_entry)

        # Get summary for last 1 hour
        summary = error_tracker.get_error_summary(hours=1)

        # Should only include recent errors
        assert summary["total_errors"] == 2
        assert len(summary["recent_errors"]) == 2

        # Verify only recent error types are included
        error_types = summary["error_types"]
        assert "ValueError" in error_types
        assert "RuntimeError" in error_types
        assert "TypeError" not in error_types
        assert "KeyError" not in error_types

    def test_get_error_summary_no_data(self, error_tracker):
        """Test getting error summary with no data."""
        summary = error_tracker.get_error_summary()

        assert summary["total_errors"] == 0
        assert summary["error_types"] == {}
        assert summary["error_by_component"] == {}
        assert summary["recent_errors"] == []
        assert summary["time_range_hours"] == 24

    def test_get_recent_errors_with_limit(self, error_tracker):
        """Test getting recent errors with limit."""
        # Add many errors
        for i in range(10):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                error_tracker.track_error(e, {"index": i})

        # Get limited recent errors
        recent = error_tracker.get_recent_errors(limit=5)

        assert len(recent) == 5

        # Verify they are the most recent (highest indices)
        for i, error in enumerate(recent):
            expected_index = 9 - i  # Most recent first
            assert error["context"]["index"] == expected_index

    def test_get_recent_errors_no_limit(self, error_tracker):
        """Test getting recent errors without limit."""
        # Add errors
        for i in range(3):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                error_tracker.track_error(e)

        recent = error_tracker.get_recent_errors()

        # Should return all errors
        assert len(recent) == 3

    def test_get_recent_errors_empty(self, error_tracker):
        """Test getting recent errors when no errors exist."""
        recent = error_tracker.get_recent_errors()
        assert recent == []

    def test_error_history_chronological_order(self, error_tracker):
        """Test that error history maintains chronological order."""
        # Add errors with small delays to ensure different timestamps
        for i in range(5):
            try:
                raise ValueError(f"Error {i}")
            except ValueError as e:
                error_tracker.track_error(e, {"order": i})
                time.sleep(0.001)  # Small delay

        # Verify errors are in chronological order
        for i in range(len(error_tracker.error_history) - 1):
            current_time = error_tracker.error_history[i]["timestamp"]
            next_time = error_tracker.error_history[i + 1]["timestamp"]
            assert current_time <= next_time


class TestMLOperationsLoggerAdvanced:
    """Advanced tests for MLOperationsLogger."""

    @pytest.fixture
    def mock_logger(self):
        """Create mock logger."""
        return Mock(spec=logging.Logger)

    @pytest.fixture
    def mlops_logger(self, mock_logger):
        """Create MLOperationsLogger with mock logger."""
        with patch("src.utils.logger.get_logger", return_value=mock_logger):
            return MLOperationsLogger()

    def test_mlops_logger_initialization(self, mlops_logger, mock_logger):
        """Test MLOperationsLogger initialization."""
        assert mlops_logger.logger == mock_logger
        assert isinstance(mlops_logger.training_events, list)
        assert isinstance(mlops_logger.prediction_events, list)
        assert isinstance(mlops_logger.drift_events, list)

    def test_log_training_event_start(self, mlops_logger, mock_logger):
        """Test logging training start event."""
        mlops_logger.log_training_event(
            room_id="master_bedroom",
            model_type="lstm",
            event_type="training_start",
            metrics={
                "training_samples": 5000,
                "validation_samples": 1000,
                "features": 25,
                "hyperparameters": {
                    "learning_rate": 0.001,
                    "batch_size": 32,
                    "epochs": 100,
                },
            },
        )

        # Verify logger was called
        mock_logger.info.assert_called_once()

        call_args = mock_logger.info.call_args
        message = call_args[0][0]
        extra = call_args[1]["extra"]

        # Verify message content
        assert "Training event" in message
        assert "training_start" in message
        assert "master_bedroom" in message
        assert "lstm" in message

        # Verify extra data
        assert extra["room_id"] == "master_bedroom"
        assert extra["model_type"] == "lstm"
        assert extra["event_type"] == "training_start"
        assert extra["training_samples"] == 5000
        assert extra["validation_samples"] == 1000
        assert extra["features"] == 25
        assert extra["hyperparameters"]["learning_rate"] == 0.001

        # Verify event was stored in history
        assert len(mlops_logger.training_events) == 1
        stored_event = mlops_logger.training_events[0]

        assert stored_event["room_id"] == "master_bedroom"
        assert stored_event["model_type"] == "lstm"
        assert stored_event["event_type"] == "training_start"
        assert stored_event["metrics"]["training_samples"] == 5000
        assert "timestamp" in stored_event

    def test_log_training_event_completion(self, mlops_logger, mock_logger):
        """Test logging training completion event."""
        mlops_logger.log_training_event(
            room_id="kitchen",
            model_type="xgboost",
            event_type="training_complete",
            metrics={
                "duration_seconds": 245.6,
                "final_loss": 0.0234,
                "accuracy": 0.9123,
                "val_accuracy": 0.8945,
                "early_stopping_epoch": 85,
            },
        )

        call_args = mock_logger.info.call_args
        extra = call_args[1]["extra"]

        # Verify completion metrics
        assert extra["duration_seconds"] == 245.6
        assert extra["final_loss"] == 0.0234
        assert extra["accuracy"] == 0.9123
        assert extra["val_accuracy"] == 0.8945
        assert extra["early_stopping_epoch"] == 85

    def test_log_training_event_without_metrics(self, mlops_logger, mock_logger):
        """Test logging training event without metrics."""
        mlops_logger.log_training_event(
            room_id="bathroom", model_type="hmm", event_type="validation_check"
        )

        call_args = mock_logger.info.call_args
        extra = call_args[1]["extra"]

        # Verify basic data without metrics
        assert extra["room_id"] == "bathroom"
        assert extra["model_type"] == "hmm"
        assert extra["event_type"] == "validation_check"

        # Verify no metric keys are present
        metric_keys = set(extra.keys()) - {"room_id", "model_type", "event_type"}
        assert len(metric_keys) == 0

        # Verify stored event has empty metrics
        stored_event = mlops_logger.training_events[0]
        assert stored_event["metrics"] == {}

    def test_log_prediction_event_with_all_data(self, mlops_logger, mock_logger):
        """Test logging prediction event with comprehensive data."""
        mlops_logger.log_prediction_event(
            room_id="living_room",
            prediction_type="next_occupancy",
            model_type="ensemble",
            confidence=0.87,
            features_used=23,
            prediction_horizon_minutes=45,
            input_features={
                "last_motion_minutes": 15,
                "door_state": "closed",
                "time_of_day": "evening",
                "day_of_week": "weekday",
            },
        )

        # Verify logger was called
        mock_logger.info.assert_called_once()

        call_args = mock_logger.info.call_args
        message = call_args[0][0]
        extra = call_args[1]["extra"]

        # Verify message content
        assert "Prediction event" in message
        assert "next_occupancy" in message
        assert "living_room" in message
        assert "87.0% confidence" in message

        # Verify extra data
        assert extra["room_id"] == "living_room"
        assert extra["prediction_type"] == "next_occupancy"
        assert extra["model_type"] == "ensemble"
        assert extra["confidence"] == 0.87
        assert extra["features_used"] == 23
        assert extra["prediction_horizon_minutes"] == 45
        assert extra["input_features"]["last_motion_minutes"] == 15
        assert extra["input_features"]["door_state"] == "closed"

        # Verify event was stored
        assert len(mlops_logger.prediction_events) == 1
        stored_event = mlops_logger.prediction_events[0]
        assert stored_event["confidence"] == 0.87
        assert stored_event["input_features"]["time_of_day"] == "evening"

    def test_log_prediction_event_minimal_data(self, mlops_logger, mock_logger):
        """Test logging prediction event with minimal data."""
        mlops_logger.log_prediction_event(
            room_id="office",
            prediction_type="next_vacancy",
            model_type="lstm",
            confidence=0.65,
        )

        call_args = mock_logger.info.call_args
        extra = call_args[1]["extra"]

        # Verify core data
        assert extra["room_id"] == "office"
        assert extra["prediction_type"] == "next_vacancy"
        assert extra["model_type"] == "lstm"
        assert extra["confidence"] == 0.65

        # Verify optional parameters are not present
        assert "features_used" not in extra
        assert "prediction_horizon_minutes" not in extra
        assert "input_features" not in extra

    def test_log_drift_detection_with_detailed_info(self, mlops_logger, mock_logger):
        """Test logging drift detection with detailed information."""
        mlops_logger.log_drift_detection(
            room_id="guest_room",
            drift_type="statistical",
            severity=0.75,
            action_taken="model_retrain",
            detection_method="kolmogorov_smirnov",
            affected_features=["motion_frequency", "door_transitions", "time_patterns"],
            baseline_period_days=30,
            comparison_period_days=7,
        )

        # Verify logger was called
        mock_logger.warning.assert_called_once()

        call_args = mock_logger.warning.call_args
        message = call_args[0][0]
        extra = call_args[1]["extra"]

        # Verify message content
        assert "Concept drift detected" in message
        assert "guest_room" in message
        assert "statistical" in message
        assert "75.0% severity" in message
        assert "model_retrain" in message

        # Verify extra data
        assert extra["room_id"] == "guest_room"
        assert extra["drift_type"] == "statistical"
        assert extra["severity"] == 0.75
        assert extra["action_taken"] == "model_retrain"
        assert extra["detection_method"] == "kolmogorov_smirnov"
        assert extra["affected_features"] == [
            "motion_frequency",
            "door_transitions",
            "time_patterns",
        ]
        assert extra["baseline_period_days"] == 30
        assert extra["comparison_period_days"] == 7

        # Verify event was stored
        assert len(mlops_logger.drift_events) == 1
        stored_event = mlops_logger.drift_events[0]
        assert stored_event["severity"] == 0.75
        assert len(stored_event["affected_features"]) == 3

    def test_log_drift_detection_minimal_info(self, mlops_logger, mock_logger):
        """Test logging drift detection with minimal information."""
        mlops_logger.log_drift_detection(
            room_id="basement",
            drift_type="performance",
            severity=0.45,
            action_taken="monitoring",
        )

        call_args = mock_logger.warning.call_args
        extra = call_args[1]["extra"]

        # Verify core data
        assert extra["room_id"] == "basement"
        assert extra["drift_type"] == "performance"
        assert extra["severity"] == 0.45
        assert extra["action_taken"] == "monitoring"

        # Verify optional parameters are not present
        assert "detection_method" not in extra
        assert "affected_features" not in extra
        assert "baseline_period_days" not in extra

    def test_get_training_summary(self, mlops_logger):
        """Test getting training summary."""
        # Add training events
        base_time = datetime.now()

        training_events = [
            (
                "room1",
                "lstm",
                "training_start",
                {"samples": 1000},
                base_time - timedelta(hours=2),
            ),
            (
                "room1",
                "lstm",
                "training_complete",
                {"accuracy": 0.85},
                base_time - timedelta(hours=1),
            ),
            (
                "room2",
                "xgboost",
                "training_start",
                {"samples": 1500},
                base_time - timedelta(minutes=30),
            ),
            (
                "room2",
                "xgboost",
                "training_complete",
                {"accuracy": 0.90},
                base_time - timedelta(minutes=10),
            ),
        ]

        for room_id, model_type, event_type, metrics, timestamp in training_events:
            event = {
                "room_id": room_id,
                "model_type": model_type,
                "event_type": event_type,
                "metrics": metrics,
                "timestamp": timestamp,
            }
            mlops_logger.training_events.append(event)

        summary = mlops_logger.get_training_summary(hours=24)

        # Verify summary structure
        assert "total_events" in summary
        assert "events_by_room" in summary
        assert "events_by_model" in summary
        assert "recent_events" in summary

        # Verify totals
        assert summary["total_events"] == 4

        # Verify breakdown by room
        assert summary["events_by_room"]["room1"] == 2
        assert summary["events_by_room"]["room2"] == 2

        # Verify breakdown by model
        assert summary["events_by_model"]["lstm"] == 2
        assert summary["events_by_model"]["xgboost"] == 2

        # Verify recent events
        assert len(summary["recent_events"]) == 4

    def test_get_prediction_summary(self, mlops_logger):
        """Test getting prediction summary."""
        # Add prediction events
        predictions = [
            ("room1", "next_occupancy", "lstm", 0.85),
            ("room1", "next_occupancy", "lstm", 0.90),
            ("room2", "next_vacancy", "xgboost", 0.75),
            ("room2", "next_vacancy", "xgboost", 0.80),
        ]

        for room_id, pred_type, model_type, confidence in predictions:
            event = {
                "room_id": room_id,
                "prediction_type": pred_type,
                "model_type": model_type,
                "confidence": confidence,
                "timestamp": datetime.now(),
            }
            mlops_logger.prediction_events.append(event)

        summary = mlops_logger.get_prediction_summary(hours=24)

        # Verify summary structure
        assert "total_predictions" in summary
        assert "predictions_by_room" in summary
        assert "predictions_by_type" in summary
        assert "average_confidence" in summary

        # Verify totals
        assert summary["total_predictions"] == 4

        # Verify breakdown by room
        assert summary["predictions_by_room"]["room1"] == 2
        assert summary["predictions_by_room"]["room2"] == 2

        # Verify breakdown by type
        assert summary["predictions_by_type"]["next_occupancy"] == 2
        assert summary["predictions_by_type"]["next_vacancy"] == 2

        # Verify average confidence
        expected_avg = (0.85 + 0.90 + 0.75 + 0.80) / 4
        assert abs(summary["average_confidence"] - expected_avg) < 0.001

    def test_get_drift_summary(self, mlops_logger):
        """Test getting drift summary."""
        # Add drift events
        drift_events = [
            ("room1", "statistical", 0.8, "model_retrain"),
            ("room1", "performance", 0.6, "monitoring"),
            ("room2", "statistical", 0.9, "model_retrain"),
        ]

        for room_id, drift_type, severity, action in drift_events:
            event = {
                "room_id": room_id,
                "drift_type": drift_type,
                "severity": severity,
                "action_taken": action,
                "timestamp": datetime.now(),
            }
            mlops_logger.drift_events.append(event)

        summary = mlops_logger.get_drift_summary(hours=24)

        # Verify summary structure
        assert "total_drift_events" in summary
        assert "drift_by_room" in summary
        assert "drift_by_type" in summary
        assert "actions_taken" in summary
        assert "average_severity" in summary

        # Verify totals
        assert summary["total_drift_events"] == 3

        # Verify breakdown by room
        assert summary["drift_by_room"]["room1"] == 2
        assert summary["drift_by_room"]["room2"] == 1

        # Verify breakdown by type
        assert summary["drift_by_type"]["statistical"] == 2
        assert summary["drift_by_type"]["performance"] == 1

        # Verify actions taken
        assert summary["actions_taken"]["model_retrain"] == 2
        assert summary["actions_taken"]["monitoring"] == 1

        # Verify average severity
        expected_avg = (0.8 + 0.6 + 0.9) / 3
        assert abs(summary["average_severity"] - expected_avg) < 0.001


class TestLoggerManagerAdvanced:
    """Advanced tests for LoggerManager."""

    @pytest.fixture
    def logger_manager(self):
        """Create LoggerManager instance."""
        return LoggerManager()

    def test_logger_manager_initialization(self, logger_manager):
        """Test LoggerManager initialization."""
        assert hasattr(logger_manager, "_loggers")
        assert hasattr(logger_manager, "_performance_logger")
        assert hasattr(logger_manager, "_error_tracker")
        assert hasattr(logger_manager, "_ml_ops_logger")

    def test_get_logger_creates_new_logger(self, logger_manager):
        """Test getting a logger creates a new logger."""
        logger = logger_manager.get_logger("test_logger")
        assert logger is not None
        assert isinstance(logger, logging.Logger)
        assert logger.name == "test_logger"

    def test_get_logger_returns_same_instance(self, logger_manager):
        """Test getting the same logger returns the same instance."""
        logger1 = logger_manager.get_logger("same_logger")
        logger2 = logger_manager.get_logger("same_logger")
        assert logger1 is logger2


class TestGlobalFunctionsAdvanced:
    """Advanced tests for global utility functions."""

    def test_get_logger_singleton_behavior(self):
        """Test logger singleton behavior with same name."""
        logger1 = get_logger("test_logger")
        logger2 = get_logger("test_logger")

        # Should return the same logger instance
        assert logger1 is logger2
        assert logger1.name == "test_logger"

    def test_get_logger_different_names(self):
        """Test getting loggers with different names."""
        logger1 = get_logger("logger_a")
        logger2 = get_logger("logger_b")

        # Should return different logger instances
        assert logger1 is not logger2
        assert logger1.name == "logger_a"
        assert logger2.name == "logger_b"

    def test_get_performance_logger_singleton(self):
        """Test performance logger singleton behavior."""
        perf_logger1 = get_performance_logger()
        perf_logger2 = get_performance_logger()

        assert perf_logger1 is perf_logger2
        assert isinstance(perf_logger1, PerformanceLogger)

    def test_get_error_tracker_singleton(self):
        """Test error tracker singleton behavior."""
        error_tracker1 = get_error_tracker()
        error_tracker2 = get_error_tracker()

        assert error_tracker1 is error_tracker2
        assert isinstance(error_tracker1, ErrorTracker)

    def test_get_ml_ops_logger_singleton(self):
        """Test ML ops logger singleton behavior."""
        mlops_logger1 = get_ml_ops_logger()
        mlops_logger2 = get_ml_ops_logger()

        assert mlops_logger1 is mlops_logger2
        assert isinstance(mlops_logger1, MLOperationsLogger)

    def test_logger_manager_performance_logger_access(self):
        """Test accessing performance logger through manager."""
        manager = LoggerManager()
        perf_logger = manager.get_performance_logger()
        assert isinstance(perf_logger, PerformanceLogger)

    def test_logger_manager_error_tracker_access(self):
        """Test accessing error tracker through manager."""
        manager = LoggerManager()
        error_tracker = manager.get_error_tracker()
        assert isinstance(error_tracker, ErrorTracker)

    def test_logger_manager_ml_ops_logger_access(self):
        """Test accessing ML ops logger through manager."""
        manager = LoggerManager()
        ml_ops_logger = manager.get_ml_ops_logger()
        assert isinstance(ml_ops_logger, MLOperationsLogger)


if __name__ == "__main__":
    pytest.main([__file__])
