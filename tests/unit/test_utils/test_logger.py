"""
Unit tests for logger utilities.

Tests all logging components including StructuredFormatter, PerformanceLogger,
ErrorTracker, MLOperationsLogger, and LoggerManager.
"""

from datetime import datetime
import json
import logging
from pathlib import Path
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from src.utils.logger import (
    ErrorTracker,
    LoggerManager,
    MLOperationsLogger,
    PerformanceLogger,
    StructuredFormatter,
    get_error_tracker,
    get_logger,
    get_logger_manager,
    get_ml_ops_logger,
    get_performance_logger,
)


class TestStructuredFormatter:
    """Test StructuredFormatter class."""

    def test_basic_formatting(self):
        """Test basic JSON log formatting."""
        formatter = StructuredFormatter()

        # Create a log record
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test"
        assert log_data["line_number"] == 123
        assert "timestamp" in log_data

    def test_formatting_with_exception(self):
        """Test formatting with exception information."""
        formatter = StructuredFormatter()

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = sys.exc_info()

        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="test.py",
            lineno=123,
            msg="Error occurred",
            args=(),
            exc_info=exc_info,
        )

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert "exception" in log_data
        assert log_data["exception"]["type"] == "ValueError"
        assert log_data["exception"]["message"] == "Test exception"
        assert isinstance(log_data["exception"]["traceback"], list)

    def test_formatting_with_extra_fields(self):
        """Test formatting with extra fields."""
        formatter = StructuredFormatter(include_extra=True)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"
        record.user_id = 12345

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert "extra" in log_data
        assert log_data["extra"]["custom_field"] == "custom_value"
        assert log_data["extra"]["user_id"] == 12345

    def test_formatting_without_extra_fields(self):
        """Test formatting with extra fields disabled."""
        formatter = StructuredFormatter(include_extra=False)

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test message",
            args=(),
            exc_info=None,
        )
        record.custom_field = "custom_value"

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert "extra" not in log_data


class TestPerformanceLogger:
    """Test PerformanceLogger class."""

    def test_performance_logger_creation(self):
        """Test PerformanceLogger creation."""
        logger = PerformanceLogger()
        assert logger.logger.name == "occupancy_prediction.performance"

    def test_performance_logger_custom_name(self):
        """Test PerformanceLogger with custom name."""
        logger = PerformanceLogger("custom.performance")
        assert logger.logger.name == "custom.performance"

    def test_log_operation_time(self):
        """Test logging operation time."""
        logger = PerformanceLogger()

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_operation_time(
                "prediction",
                0.123,
                room_id="living_room",
                prediction_type="next_occupied",
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Operation completed: prediction" in args[0]
            assert kwargs["extra"]["operation"] == "prediction"
            assert kwargs["extra"]["duration_seconds"] == 0.123
            assert kwargs["extra"]["room_id"] == "living_room"
            assert kwargs["extra"]["prediction_type"] == "next_occupied"

    def test_log_prediction_accuracy(self):
        """Test logging prediction accuracy."""
        logger = PerformanceLogger()

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_prediction_accuracy("bedroom", 5.5, 0.85, "next_vacant")

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "5.50 minutes" in args[0]
            assert kwargs["extra"]["room_id"] == "bedroom"
            assert kwargs["extra"]["accuracy_minutes"] == 5.5
            assert kwargs["extra"]["confidence"] == 0.85

    def test_log_model_metrics(self):
        """Test logging model metrics."""
        logger = PerformanceLogger()
        metrics = {"accuracy": 0.9, "precision": 0.85}

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_model_metrics("office", "lstm", metrics)

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert kwargs["extra"]["room_id"] == "office"
            assert kwargs["extra"]["model_type"] == "lstm"
            assert kwargs["extra"]["metrics"] == metrics

    def test_log_resource_usage(self):
        """Test logging resource usage."""
        logger = PerformanceLogger()

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_resource_usage(75.5, 1024.0, 45.2)

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert kwargs["extra"]["cpu_percent"] == 75.5
            assert kwargs["extra"]["memory_mb"] == 1024.0
            assert kwargs["extra"]["disk_usage_percent"] == 45.2


class TestErrorTracker:
    """Test ErrorTracker class."""

    def test_error_tracker_creation(self):
        """Test ErrorTracker creation."""
        tracker = ErrorTracker()
        assert tracker.logger.name == "occupancy_prediction.errors"

    def test_track_error_basic(self):
        """Test basic error tracking."""
        tracker = ErrorTracker()
        error = ValueError("Test error")

        with patch.object(tracker.logger, "error") as mock_error:
            tracker.track_error(error)

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args

            assert "Error tracked: Test error" in args[0]
            assert kwargs["extra"]["error_type"] == "ValueError"
            assert kwargs["extra"]["error_message"] == "Test error"
            assert kwargs["extra"]["severity"] == "error"

    def test_track_error_with_context(self):
        """Test error tracking with context."""
        tracker = ErrorTracker()
        error = ConnectionError("Database unavailable")
        context = {"database": "postgresql", "retry_count": 3}

        with patch.object(tracker.logger, "critical") as mock_critical:
            tracker.track_error(error, context, severity="critical", alert=True)

            mock_critical.assert_called_once()
            args, kwargs = mock_critical.call_args

            assert kwargs["extra"]["context"] == context
            assert kwargs["extra"]["severity"] == "critical"
            assert kwargs["extra"]["alert_required"] is True

    def test_track_prediction_error(self):
        """Test prediction-specific error tracking."""
        tracker = ErrorTracker()
        error = RuntimeError("Model failed")

        with patch.object(tracker, "track_error") as mock_track:
            tracker.track_prediction_error("kitchen", error, "next_occupied", "lstm")

            mock_track.assert_called_once()
            args, kwargs = mock_track.call_args

            assert args[0] == error
            assert args[1]["room_id"] == "kitchen"
            assert args[1]["prediction_type"] == "next_occupied"
            assert args[1]["model_type"] == "lstm"
            assert kwargs["severity"] == "error"
            assert kwargs["alert"] is True

    def test_track_data_error(self):
        """Test data ingestion error tracking."""
        tracker = ErrorTracker()
        error = ValueError("Invalid sensor data")

        with patch.object(tracker, "track_error") as mock_track:
            tracker.track_data_error(error, "Home Assistant", "sensor.motion")

            mock_track.assert_called_once()
            args, kwargs = mock_track.call_args

            assert args[1]["data_source"] == "Home Assistant"
            assert args[1]["entity_id"] == "sensor.motion"
            assert kwargs["severity"] == "warning"

    def test_track_integration_error(self):
        """Test integration error tracking."""
        tracker = ErrorTracker()
        error = ConnectionError("MQTT broker down")

        with patch.object(tracker, "track_error") as mock_track:
            tracker.track_integration_error(error, "mqtt", "/api/predictions")

            mock_track.assert_called_once()
            args, kwargs = mock_track.call_args

            assert args[1]["integration_type"] == "mqtt"
            assert args[1]["endpoint"] == "/api/predictions"
            assert kwargs["severity"] == "error"


class TestMLOperationsLogger:
    """Test MLOperationsLogger class."""

    def test_ml_ops_logger_creation(self):
        """Test MLOperationsLogger creation."""
        logger = MLOperationsLogger()
        assert logger.logger.name == "occupancy_prediction.ml_ops"

    def test_log_training_event(self):
        """Test logging training events."""
        logger = MLOperationsLogger()
        metrics = {"loss": 0.1, "accuracy": 0.95}

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_training_event(
                "bathroom", "xgboost", "training_complete", metrics
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Training event: training_complete for xgboost" in args[0]
            assert kwargs["extra"]["room_id"] == "bathroom"
            assert kwargs["extra"]["model_type"] == "xgboost"
            assert kwargs["extra"]["metrics"] == metrics

    def test_log_drift_detection(self):
        """Test logging drift detection."""
        logger = MLOperationsLogger()

        with patch.object(logger.logger, "warning") as mock_warning:
            logger.log_drift_detection(
                "living_room", "accuracy_drop", 0.8, "retrain_scheduled"
            )

            mock_warning.assert_called_once()
            args, kwargs = mock_warning.call_args

            assert "Concept drift detected: accuracy_drop" in args[0]
            assert kwargs["extra"]["room_id"] == "living_room"
            assert kwargs["extra"]["severity"] == 0.8
            assert kwargs["extra"]["action_taken"] == "retrain_scheduled"

    def test_log_model_deployment(self):
        """Test logging model deployment."""
        logger = MLOperationsLogger()
        metrics = {"validation_accuracy": 0.92}

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_model_deployment("office", "ensemble", "v2.1.0", metrics)

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Model deployed: ensemble v2.1.0" in args[0]
            assert kwargs["extra"]["version"] == "v2.1.0"
            assert kwargs["extra"]["performance_metrics"] == metrics

    def test_log_feature_importance(self):
        """Test logging feature importance."""
        logger = MLOperationsLogger()
        importance = {"time_since_last": 0.4, "hour_of_day": 0.3}

        with patch.object(logger.logger, "info") as mock_info:
            logger.log_feature_importance("garage", "random_forest", importance)

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert kwargs["extra"]["feature_importance"] == importance


class TestLoggerManager:
    """Test LoggerManager class."""

    def test_logger_manager_creation(self):
        """Test LoggerManager creation with default config."""
        with tempfile.TemporaryDirectory() as temp_dir:
            manager = LoggerManager(Path(temp_dir) / "nonexistent.yaml")

            assert manager.performance_logger is not None
            assert manager.error_tracker is not None
            assert manager.ml_ops_logger is not None

    def test_logger_manager_with_config(self):
        """Test LoggerManager with YAML configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "logging.yaml"

            # Create test logging config
            config = {
                "version": 1,
                "handlers": {
                    "console": {
                        "class": "logging.StreamHandler",
                        "level": "INFO",
                        "formatter": "simple",
                    }
                },
                "formatters": {
                    "simple": {"format": "%(name)s - %(levelname)s - %(message)s"}
                },
                "root": {"level": "INFO", "handlers": ["console"]},
            }

            with open(config_path, "w") as f:
                yaml.dump(config, f)

            manager = LoggerManager(config_path)
            assert manager.config_path == config_path

    def test_get_logger(self):
        """Test getting logger instance."""
        manager = LoggerManager()
        logger = manager.get_logger("test_component")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "occupancy_prediction.test_component"

    def test_get_specialized_loggers(self):
        """Test getting specialized logger instances."""
        manager = LoggerManager()

        perf_logger = manager.get_performance_logger()
        error_tracker = manager.get_error_tracker()
        ml_ops_logger = manager.get_ml_ops_logger()

        assert isinstance(perf_logger, PerformanceLogger)
        assert isinstance(error_tracker, ErrorTracker)
        assert isinstance(ml_ops_logger, MLOperationsLogger)

    def test_log_operation_context_manager(self):
        """Test log operation context manager."""
        manager = LoggerManager()

        with patch.object(manager.performance_logger, "log_operation_time") as mock_log:
            with manager.log_operation("test_operation", "test_room"):
                pass  # Operation completes successfully

            mock_log.assert_called_once()
            args = mock_log.call_args[0]
            assert args[0] == "test_operation"
            assert args[2] == "test_room"
            assert isinstance(args[1], float)  # Duration

    def test_log_operation_context_manager_with_exception(self):
        """Test log operation context manager with exception."""
        manager = LoggerManager()

        with patch.object(manager.error_tracker, "track_error") as mock_track:
            with pytest.raises(ValueError):
                with manager.log_operation("failing_operation"):
                    raise ValueError("Test error")

            mock_track.assert_called_once()
            args = mock_track.call_args[0]
            assert isinstance(args[0], ValueError)
            assert args[1]["operation"] == "failing_operation"


class TestGlobalFunctions:
    """Test global convenience functions."""

    def test_get_logger_manager(self):
        """Test get_logger_manager function."""
        manager1 = get_logger_manager()
        manager2 = get_logger_manager()

        # Should return same instance (singleton pattern)
        assert manager1 is manager2
        assert isinstance(manager1, LoggerManager)

    def test_get_logger(self):
        """Test get_logger convenience function."""
        logger = get_logger("test_module")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "occupancy_prediction.test_module"

    def test_get_performance_logger(self):
        """Test get_performance_logger convenience function."""
        logger = get_performance_logger()
        assert isinstance(logger, PerformanceLogger)

    def test_get_error_tracker(self):
        """Test get_error_tracker convenience function."""
        tracker = get_error_tracker()
        assert isinstance(tracker, ErrorTracker)

    def test_get_ml_ops_logger(self):
        """Test get_ml_ops_logger convenience function."""
        logger = get_ml_ops_logger()
        assert isinstance(logger, MLOperationsLogger)


class TestIntegration:
    """Test logging system integration."""

    def test_structured_logging_integration(self):
        """Test that structured logging works end-to-end."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "test.log"

            # Create logger with file handler
            logger = logging.getLogger("test_integration")
            logger.setLevel(logging.INFO)

            handler = logging.FileHandler(log_file)
            handler.setFormatter(StructuredFormatter())
            logger.addHandler(handler)

            # Log a message
            logger.info("Test integration message", extra={"test_field": "test_value"})

            # Read and verify log content
            with open(log_file, "r") as f:
                log_content = f.read().strip()

            log_data = json.loads(log_content)
            assert log_data["message"] == "Test integration message"
            assert log_data["extra"]["test_field"] == "test_value"

    def test_performance_and_error_tracking_integration(self):
        """Test integration between performance logging and error tracking."""
        perf_logger = PerformanceLogger()
        error_tracker = ErrorTracker()

        # Mock the underlying loggers to verify calls
        with patch.object(perf_logger.logger, "info") as mock_perf, patch.object(
            error_tracker.logger, "error"
        ) as mock_error:

            # Log performance metric
            perf_logger.log_operation_time("test_op", 0.5)

            # Track an error
            error_tracker.track_error(RuntimeError("Test error"))

            # Verify both loggers were called
            mock_perf.assert_called_once()
            mock_error.assert_called_once()

    def test_logger_manager_integration(self):
        """Test LoggerManager provides consistent logger instances."""
        manager = LoggerManager()

        # Get same logger multiple times
        logger1 = manager.get_logger("integration_test")
        logger2 = manager.get_logger("integration_test")

        # Should be the same logger instance
        assert logger1 is logger2

        # Different names should be different loggers
        logger3 = manager.get_logger("different_name")
        assert logger3 is not logger1

    def test_log_levels_and_filtering(self):
        """Test that log levels work correctly."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = Path(temp_dir) / "level_test.log"

            logger = logging.getLogger("level_test")
            logger.setLevel(logging.WARNING)  # Only WARNING and above

            handler = logging.FileHandler(log_file)
            handler.setFormatter(StructuredFormatter())
            logger.addHandler(handler)

            # Log at different levels
            logger.debug("Debug message")  # Should be filtered out
            logger.info("Info message")  # Should be filtered out
            logger.warning("Warning message")  # Should appear
            logger.error("Error message")  # Should appear

            # Read log content
            with open(log_file, "r") as f:
                lines = f.readlines()

            # Should only have WARNING and ERROR messages
            assert len(lines) == 2

            warning_log = json.loads(lines[0])
            error_log = json.loads(lines[1])

            assert warning_log["level"] == "WARNING"
            assert warning_log["message"] == "Warning message"
            assert error_log["level"] == "ERROR"
            assert error_log["message"] == "Error message"


@pytest.mark.unit
class TestLoggerEdgeCases:
    """Test edge cases and error conditions in logging system."""

    def test_structured_formatter_with_none_values(self):
        """Test StructuredFormatter handles None values properly."""
        formatter = StructuredFormatter()

        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=123,
            msg="Test with None values",
            args=(),
            exc_info=None,
        )
        record.none_field = None
        record.empty_string = ""

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Should handle None and empty values
        assert log_data["extra"]["none_field"] is None
        assert log_data["extra"]["empty_string"] == ""

    def test_logger_manager_with_missing_config(self):
        """Test LoggerManager gracefully handles missing config files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            missing_config = Path(temp_dir) / "missing.yaml"

            # Should not raise exception
            manager = LoggerManager(missing_config)

            # Should still provide working loggers
            logger = manager.get_logger("test")
            assert isinstance(logger, logging.Logger)

    def test_logger_manager_with_invalid_config(self):
        """Test LoggerManager handles invalid YAML gracefully."""
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid_config = Path(temp_dir) / "invalid.yaml"

            # Create invalid YAML
            with open(invalid_config, "w") as f:
                f.write("invalid: yaml: content: [unclosed")

            # Should fallback to basic logging
            manager = LoggerManager(invalid_config)
            logger = manager.get_logger("test")
            assert isinstance(logger, logging.Logger)

    def test_performance_logger_with_none_parameters(self):
        """Test PerformanceLogger handles None parameters."""
        logger = PerformanceLogger()

        with patch.object(logger.logger, "info") as mock_info:
            # Should handle None room_id gracefully
            logger.log_operation_time("test_op", 1.0, room_id=None)

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args
            assert kwargs["extra"]["room_id"] is None

    def test_error_tracker_with_nested_exceptions(self):
        """Test ErrorTracker handles exception chaining."""
        tracker = ErrorTracker()

        try:
            try:
                raise ValueError("Inner exception")
            except ValueError as e:
                raise RuntimeError("Outer exception") from e
        except RuntimeError as e:
            with patch.object(tracker.logger, "error") as mock_error:
                tracker.track_error(e)

                mock_error.assert_called_once()
                args, kwargs = mock_error.call_args

                assert kwargs["extra"]["error_type"] == "RuntimeError"
                assert "Outer exception" in kwargs["extra"]["error_message"]
