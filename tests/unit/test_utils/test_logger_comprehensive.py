"""
Comprehensive unit tests for StructuredLogger system infrastructure.

This test suite provides comprehensive coverage for the StructuredLogger module,
focusing on production-grade testing with real structured logging, JSON formatting,
log level management, performance validation, log rotation, and error tracking.

Target Coverage: 85%+ for StructuredLogger
Test Methods: 40+ comprehensive test methods
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from io import StringIO
import json
import logging
import logging.handlers
import os
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock, patch

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


@pytest.fixture
def temp_log_dir():
    """Create temporary directory for log files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_logging_config():
    """Sample logging configuration for testing."""
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"},
            "json": {
                "()": "src.utils.logger.StructuredFormatter",
                "include_extra": True,
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "level": "INFO",
                "formatter": "standard",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "class": "logging.FileHandler",
                "level": "DEBUG",
                "formatter": "json",
                "filename": "logs/test.log",
                "mode": "a",
            },
        },
        "loggers": {
            "occupancy_prediction": {
                "level": "DEBUG",
                "handlers": ["console", "file"],
                "propagate": False,
            }
        },
        "root": {"level": "INFO", "handlers": ["console"]},
    }


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""

    def test_structured_formatter_initialization(self):
        """Test StructuredFormatter initialization."""
        formatter = StructuredFormatter(include_extra=True)
        assert formatter.include_extra is True

        formatter_no_extra = StructuredFormatter(include_extra=False)
        assert formatter_no_extra.include_extra is False

    def test_basic_log_formatting(self):
        """Test basic log record formatting to JSON."""
        formatter = StructuredFormatter(include_extra=True)

        # Create test log record
        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add standard fields
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 9876

        formatted = formatter.format(record)

        # Parse JSON to verify structure
        log_data = json.loads(formatted)

        assert log_data["level"] == "INFO"
        assert log_data["logger"] == "test_logger"
        assert log_data["message"] == "Test message"
        assert log_data["module"] == "test_module"
        assert log_data["function"] == "test_function"
        assert log_data["line_number"] == 42
        assert log_data["thread"] == 12345
        assert log_data["thread_name"] == "MainThread"
        assert log_data["process"] == 9876

        # Verify timestamp is ISO format
        timestamp = log_data["timestamp"]
        datetime.fromisoformat(timestamp.replace("Z", "+00:00"))

    def test_log_formatting_with_exception(self):
        """Test log formatting with exception information."""
        formatter = StructuredFormatter(include_extra=True)

        try:
            raise ValueError("Test exception")
        except ValueError:
            exc_info = (ValueError, ValueError("Test exception"), None)

            record = logging.LogRecord(
                name="test_logger",
                level=logging.ERROR,
                pathname="/test/path.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=exc_info,
            )

            # Mock traceback format_exception to avoid complexity
            with patch(
                "traceback.format_exception",
                return_value=["Traceback line 1\n", "ValueError: Test exception\n"],
            ):
                formatted = formatter.format(record)

            log_data = json.loads(formatted)

            assert "exception" in log_data
            assert log_data["exception"]["type"] == "ValueError"
            assert log_data["exception"]["message"] == "Test exception"
            assert isinstance(log_data["exception"]["traceback"], list)

    def test_log_formatting_with_extra_fields(self):
        """Test log formatting with extra fields."""
        formatter = StructuredFormatter(include_extra=True)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message with extras",
            args=(),
            exc_info=None,
        )

        # Add extra fields
        record.room_id = "living_room"
        record.prediction_accuracy = 12.5
        record.model_type = "lstm"
        record.custom_data = {"nested": "value", "count": 42}

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert "extra" in log_data
        assert log_data["extra"]["room_id"] == "living_room"
        assert log_data["extra"]["prediction_accuracy"] == 12.5
        assert log_data["extra"]["model_type"] == "lstm"
        assert log_data["extra"]["custom_data"]["nested"] == "value"
        assert log_data["extra"]["custom_data"]["count"] == 42

    def test_log_formatting_exclude_extra_fields(self):
        """Test log formatting with extra fields excluded."""
        formatter = StructuredFormatter(include_extra=False)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Add extra fields that should be excluded
        record.room_id = "living_room"
        record.custom_field = "should_not_appear"

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        assert "extra" not in log_data

    def test_log_formatting_with_complex_data_types(self):
        """Test log formatting with complex data types."""
        formatter = StructuredFormatter(include_extra=True)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Complex data test",
            args=(),
            exc_info=None,
        )

        # Add complex data types
        record.timestamp_field = datetime.now(timezone.utc)
        record.set_field = {"item1", "item2", "item3"}
        record.none_field = None

        formatted = formatter.format(record)
        log_data = json.loads(formatted)

        # Should handle complex types via default=str
        assert "extra" in log_data
        assert "timestamp_field" in log_data["extra"]
        assert "set_field" in log_data["extra"]
        assert log_data["extra"]["none_field"] is None


class TestPerformanceLogger:
    """Test PerformanceLogger functionality."""

    def test_performance_logger_initialization(self):
        """Test PerformanceLogger initialization."""
        logger = PerformanceLogger()
        assert logger.logger.name == "occupancy_prediction.performance"

        logger_custom = PerformanceLogger("custom.performance")
        assert logger_custom.logger.name == "custom.performance"

    def test_log_operation_time(self):
        """Test logging operation timing."""
        with patch("logging.Logger.info") as mock_info:
            logger = PerformanceLogger()

            logger.log_operation_time(
                operation="model_training",
                duration=125.67,
                room_id="living_room",
                prediction_type="next_occupied",
                feature_count=15,
                data_quality=0.95,
            )

            # Verify logging call
            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Operation completed: model_training" in args[0]
            assert "extra" in kwargs

            extra = kwargs["extra"]
            assert extra["operation"] == "model_training"
            assert extra["duration_seconds"] == 125.67
            assert extra["room_id"] == "living_room"
            assert extra["prediction_type"] == "next_occupied"
            assert extra["metric_type"] == "performance"
            assert extra["feature_count"] == 15
            assert extra["data_quality"] == 0.95

    def test_log_prediction_accuracy(self):
        """Test logging prediction accuracy."""
        with patch("logging.Logger.info") as mock_info:
            logger = PerformanceLogger()

            logger.log_prediction_accuracy(
                room_id="kitchen",
                accuracy_minutes=8.5,
                confidence=0.92,
                prediction_type="next_vacant",
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Prediction accuracy: 8.50 minutes" in args[0]

            extra = kwargs["extra"]
            assert extra["room_id"] == "kitchen"
            assert extra["accuracy_minutes"] == 8.5
            assert extra["confidence"] == 0.92
            assert extra["prediction_type"] == "next_vacant"
            assert extra["metric_type"] == "accuracy"

    def test_log_model_metrics(self):
        """Test logging model performance metrics."""
        with patch("logging.Logger.info") as mock_info:
            logger = PerformanceLogger()

            metrics = {"mse": 0.0234, "rmse": 0.1529, "mae": 0.0987, "r2_score": 0.8456}

            logger.log_model_metrics(
                room_id="bedroom", model_type="lstm", metrics=metrics
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Model metrics for lstm" in args[0]

            extra = kwargs["extra"]
            assert extra["room_id"] == "bedroom"
            assert extra["model_type"] == "lstm"
            assert extra["metrics"] == metrics
            assert extra["metric_type"] == "model_performance"

    def test_log_resource_usage(self):
        """Test logging system resource usage."""
        with patch("logging.Logger.info") as mock_info:
            logger = PerformanceLogger()

            logger.log_resource_usage(
                cpu_percent=67.8, memory_mb=3072.5, disk_usage_percent=82.1
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "System resource usage" in args[0]

            extra = kwargs["extra"]
            assert extra["cpu_percent"] == 67.8
            assert extra["memory_mb"] == 3072.5
            assert extra["disk_usage_percent"] == 82.1
            assert extra["metric_type"] == "resource_usage"


class TestErrorTracker:
    """Test ErrorTracker functionality."""

    def test_error_tracker_initialization(self):
        """Test ErrorTracker initialization."""
        tracker = ErrorTracker()
        assert tracker.logger.name == "occupancy_prediction.errors"

        tracker_custom = ErrorTracker("custom.errors")
        assert tracker_custom.logger.name == "custom.errors"

    def test_track_error_basic(self):
        """Test basic error tracking."""
        with patch("logging.Logger.error") as mock_error:
            tracker = ErrorTracker()

            test_error = ValueError("Test validation error")
            context = {
                "component": "data_validation",
                "entity_id": "binary_sensor.test",
            }

            tracker.track_error(
                error=test_error, context=context, severity="error", alert=True
            )

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args

            assert "Error tracked: Test validation error" in args[0]
            assert kwargs["exc_info"] is True

            extra = kwargs["extra"]
            assert extra["error_type"] == "ValueError"
            assert extra["error_message"] == "Test validation error"
            assert extra["severity"] == "error"
            assert extra["context"] == context
            assert extra["alert_required"] is True
            assert extra["metric_type"] == "error"

    def test_track_critical_error(self):
        """Test tracking critical errors."""
        with patch("logging.Logger.critical") as mock_critical:
            tracker = ErrorTracker()

            critical_error = RuntimeError("Critical system failure")

            tracker.track_error(
                error=critical_error,
                context={"system": "database"},
                severity="critical",
                alert=True,
            )

            mock_critical.assert_called_once()
            args, kwargs = mock_critical.call_args

            assert "Critical error: Critical system failure" in args[0]
            assert kwargs["exc_info"] is True

            extra = kwargs["extra"]
            assert extra["severity"] == "critical"
            assert extra["alert_required"] is True

    def test_track_prediction_error(self):
        """Test tracking prediction-specific errors."""
        with patch("logging.Logger.error") as mock_error:
            tracker = ErrorTracker()

            prediction_error = RuntimeError("Model inference failed")

            tracker.track_prediction_error(
                room_id="living_room",
                error=prediction_error,
                prediction_type="next_occupied",
                model_type="lstm",
            )

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args

            extra = kwargs["extra"]
            assert extra["context"]["room_id"] == "living_room"
            assert extra["context"]["prediction_type"] == "next_occupied"
            assert extra["context"]["model_type"] == "lstm"
            assert extra["context"]["component"] == "prediction_engine"
            assert extra["severity"] == "error"
            assert extra["alert_required"] is True

    def test_track_data_error(self):
        """Test tracking data ingestion errors."""
        with patch("logging.Logger.error") as mock_error:
            tracker = ErrorTracker()

            data_error = ConnectionError("Home Assistant connection failed")

            tracker.track_data_error(
                error=data_error,
                data_source="home_assistant",
                entity_id="binary_sensor.living_room_motion",
            )

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args

            extra = kwargs["extra"]
            assert extra["context"]["data_source"] == "home_assistant"
            assert extra["context"]["entity_id"] == "binary_sensor.living_room_motion"
            assert extra["context"]["component"] == "data_ingestion"
            assert extra["severity"] == "warning"
            assert extra["alert_required"] is False

    def test_track_integration_error(self):
        """Test tracking integration errors."""
        with patch("logging.Logger.error") as mock_error:
            tracker = ErrorTracker()

            integration_error = TimeoutError("MQTT publish timeout")

            tracker.track_integration_error(
                error=integration_error,
                integration_type="mqtt",
                endpoint="occupancy/predictions/living_room",
            )

            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args

            extra = kwargs["extra"]
            assert extra["context"]["integration_type"] == "mqtt"
            assert extra["context"]["endpoint"] == "occupancy/predictions/living_room"
            assert extra["context"]["component"] == "integration"
            assert extra["severity"] == "error"
            assert extra["alert_required"] is True


class TestMLOperationsLogger:
    """Test MLOperationsLogger functionality."""

    def test_ml_operations_logger_initialization(self):
        """Test MLOperationsLogger initialization."""
        logger = MLOperationsLogger()
        assert logger.logger.name == "occupancy_prediction.ml_ops"

        logger_custom = MLOperationsLogger("custom.ml_ops")
        assert logger_custom.logger.name == "custom.ml_ops"

    def test_log_training_event(self):
        """Test logging training events."""
        with patch("logging.Logger.info") as mock_info:
            logger = MLOperationsLogger()

            metrics = {"training_loss": 0.0456, "validation_loss": 0.0523, "epochs": 50}

            logger.log_training_event(
                room_id="living_room",
                model_type="lstm",
                event_type="training_completed",
                metrics=metrics,
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Training event: training_completed for lstm" in args[0]

            extra = kwargs["extra"]
            assert extra["room_id"] == "living_room"
            assert extra["model_type"] == "lstm"
            assert extra["event_type"] == "training_completed"
            assert extra["metrics"] == metrics
            assert extra["component"] == "training"
            assert extra["metric_type"] == "ml_lifecycle"

    def test_log_drift_detection(self):
        """Test logging drift detection events."""
        with patch("logging.Logger.warning") as mock_warning:
            logger = MLOperationsLogger()

            logger.log_drift_detection(
                room_id="kitchen",
                drift_type="concept_drift",
                severity=0.75,
                action_taken="model_retrain_scheduled",
            )

            mock_warning.assert_called_once()
            args, kwargs = mock_warning.call_args

            assert "Concept drift detected: concept_drift" in args[0]

            extra = kwargs["extra"]
            assert extra["room_id"] == "kitchen"
            assert extra["drift_type"] == "concept_drift"
            assert extra["severity"] == 0.75
            assert extra["action_taken"] == "model_retrain_scheduled"
            assert extra["component"] == "adaptation"
            assert extra["metric_type"] == "drift_detection"

    def test_log_model_deployment(self):
        """Test logging model deployment events."""
        with patch("logging.Logger.info") as mock_info:
            logger = MLOperationsLogger()

            performance_metrics = {
                "accuracy": 0.92,
                "precision": 0.89,
                "recall": 0.94,
                "f1_score": 0.91,
            }

            logger.log_model_deployment(
                room_id="bedroom",
                model_type="ensemble",
                version="v2.1.0",
                performance_metrics=performance_metrics,
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Model deployed: ensemble v2.1.0" in args[0]

            extra = kwargs["extra"]
            assert extra["room_id"] == "bedroom"
            assert extra["model_type"] == "ensemble"
            assert extra["version"] == "v2.1.0"
            assert extra["performance_metrics"] == performance_metrics
            assert extra["component"] == "deployment"
            assert extra["metric_type"] == "ml_lifecycle"

    def test_log_feature_importance(self):
        """Test logging feature importance analysis."""
        with patch("logging.Logger.info") as mock_info:
            logger = MLOperationsLogger()

            feature_importance = {
                "time_of_day": 0.25,
                "day_of_week": 0.18,
                "recent_occupancy": 0.22,
                "temperature": 0.12,
                "motion_history": 0.23,
            }

            logger.log_feature_importance(
                room_id="living_room",
                model_type="xgboost",
                feature_importance=feature_importance,
            )

            mock_info.assert_called_once()
            args, kwargs = mock_info.call_args

            assert "Feature importance analysis" in args[0]

            extra = kwargs["extra"]
            assert extra["room_id"] == "living_room"
            assert extra["model_type"] == "xgboost"
            assert extra["feature_importance"] == feature_importance
            assert extra["component"] == "analysis"
            assert extra["metric_type"] == "feature_analysis"


class TestLoggerManager:
    """Test LoggerManager functionality."""

    def test_logger_manager_initialization_default(self, temp_log_dir):
        """Test LoggerManager initialization with default config."""
        # Create fake config file
        config_path = temp_log_dir / "logging.yaml"

        with patch("pathlib.Path.exists", return_value=False):  # No config file
            manager = LoggerManager(config_path)

        assert manager.config_path == config_path
        assert isinstance(manager.performance_logger, PerformanceLogger)
        assert isinstance(manager.error_tracker, ErrorTracker)
        assert isinstance(manager.ml_ops_logger, MLOperationsLogger)

    def test_logger_manager_initialization_with_config(
        self, temp_log_dir, sample_logging_config
    ):
        """Test LoggerManager initialization with configuration file."""
        config_path = temp_log_dir / "logging.yaml"

        # Write config file
        with open(config_path, "w") as f:
            yaml.dump(sample_logging_config, f)

        # Create logs directory
        logs_dir = temp_log_dir / "logs"
        logs_dir.mkdir()

        with patch("pathlib.Path", return_value=logs_dir):
            manager = LoggerManager(config_path)

        assert manager.config_path == config_path

    def test_logger_manager_config_loading_failure(self, temp_log_dir):
        """Test LoggerManager when config loading fails."""
        config_path = temp_log_dir / "invalid_logging.yaml"

        # Create invalid YAML file
        with open(config_path, "w") as f:
            f.write("invalid: yaml: content: [")

        # Should handle gracefully and use fallback configuration
        manager = LoggerManager(config_path)

        assert isinstance(manager.performance_logger, PerformanceLogger)
        assert isinstance(manager.error_tracker, ErrorTracker)

    def test_get_logger(self):
        """Test getting logger instance."""
        manager = LoggerManager()

        logger = manager.get_logger("test_component")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "occupancy_prediction.test_component"

    def test_get_specialized_loggers(self):
        """Test getting specialized logger instances."""
        manager = LoggerManager()

        performance_logger = manager.get_performance_logger()
        error_tracker = manager.get_error_tracker()
        ml_ops_logger = manager.get_ml_ops_logger()

        assert isinstance(performance_logger, PerformanceLogger)
        assert isinstance(error_tracker, ErrorTracker)
        assert isinstance(ml_ops_logger, MLOperationsLogger)

        # Should return same instances
        assert performance_logger is manager.performance_logger
        assert error_tracker is manager.error_tracker
        assert ml_ops_logger is manager.ml_ops_logger

    def test_log_operation_context_manager_success(self):
        """Test log operation context manager for successful operations."""
        manager = LoggerManager()

        with patch.object(manager.performance_logger, "log_operation_time") as mock_log:
            with manager.log_operation("test_operation", "test_room"):
                time.sleep(0.01)  # Simulate some work

        # Should log successful completion
        mock_log.assert_called_once()
        args, kwargs = mock_log.call_args
        assert args[0] == "test_operation"  # operation name
        assert isinstance(args[1], float)  # duration
        assert kwargs["room_id"] == "test_room"

    def test_log_operation_context_manager_exception(self):
        """Test log operation context manager when exception occurs."""
        manager = LoggerManager()

        with patch.object(manager.error_tracker, "track_error") as mock_track_error:
            with pytest.raises(ValueError):
                with manager.log_operation("failing_operation", "test_room"):
                    raise ValueError("Test operation failed")

        # Should track the error
        mock_track_error.assert_called_once()
        args = mock_track_error.call_args[0]
        assert isinstance(args[0], ValueError)  # The exception
        assert args[1]["operation"] == "failing_operation"
        assert args[1]["room_id"] == "test_room"

    def test_json_formatter_configuration(self, temp_log_dir, sample_logging_config):
        """Test JSON formatter is properly configured."""
        config_path = temp_log_dir / "logging.yaml"

        with open(config_path, "w") as f:
            yaml.dump(sample_logging_config, f)

        logs_dir = temp_log_dir / "logs"
        logs_dir.mkdir()

        with patch("logging.config.dictConfig") as mock_dict_config:
            manager = LoggerManager(config_path)

        # Should call dictConfig with modified configuration
        mock_dict_config.assert_called_once()
        config_used = mock_dict_config.call_args[0][0]

        # Should have JSON formatter configured
        assert "json" in config_used["formatters"]
        assert (
            config_used["formatters"]["json"]["()"]
            == "src.utils.logger.StructuredFormatter"
        )
        assert config_used["formatters"]["json"]["include_extra"] is True


class TestGlobalLoggerFunctions:
    """Test global logger convenience functions."""

    def test_get_logger_manager_singleton(self):
        """Test get_logger_manager returns singleton instance."""
        # Reset global instance
        import src.utils.logger as logger_module

        logger_module._logger_manager = None

        manager1 = get_logger_manager()
        manager2 = get_logger_manager()

        assert manager1 is manager2
        assert isinstance(manager1, LoggerManager)

    def test_get_logger_convenience(self):
        """Test get_logger convenience function."""
        logger = get_logger("convenience_test")

        assert isinstance(logger, logging.Logger)
        assert logger.name == "occupancy_prediction.convenience_test"

    def test_get_performance_logger_convenience(self):
        """Test get_performance_logger convenience function."""
        performance_logger = get_performance_logger()

        assert isinstance(performance_logger, PerformanceLogger)

    def test_get_error_tracker_convenience(self):
        """Test get_error_tracker convenience function."""
        error_tracker = get_error_tracker()

        assert isinstance(error_tracker, ErrorTracker)

    def test_get_ml_ops_logger_convenience(self):
        """Test get_ml_ops_logger convenience function."""
        ml_ops_logger = get_ml_ops_logger()

        assert isinstance(ml_ops_logger, MLOperationsLogger)


class TestLoggingIntegration:
    """Test logging integration scenarios."""

    def test_end_to_end_logging_workflow(self, temp_log_dir):
        """Test complete logging workflow."""
        log_file = temp_log_dir / "test_integration.log"

        # Setup logger with file handler
        logger = logging.getLogger("integration_test")
        logger.setLevel(logging.DEBUG)

        # Create file handler with structured formatter
        handler = logging.FileHandler(log_file)
        handler.setFormatter(StructuredFormatter(include_extra=True))
        logger.addHandler(handler)

        # Log various types of messages
        logger.info("Basic info message")
        logger.error("Error message", extra={"error_code": "E001", "component": "test"})
        logger.warning(
            "Warning with data", extra={"room_id": "living_room", "value": 42}
        )

        # Flush and close handler
        handler.flush()
        handler.close()

        # Read and verify log content
        with open(log_file, "r") as f:
            log_lines = f.readlines()

        assert len(log_lines) == 3

        # Parse each log line and verify JSON structure
        for line in log_lines:
            log_data = json.loads(line.strip())
            assert "timestamp" in log_data
            assert "level" in log_data
            assert "logger" in log_data
            assert "message" in log_data

    def test_concurrent_logging(self):
        """Test concurrent logging from multiple threads."""
        log_buffer = StringIO()

        # Setup logger with stream handler
        logger = logging.getLogger("concurrent_test")
        logger.setLevel(logging.DEBUG)

        handler = logging.StreamHandler(log_buffer)
        handler.setFormatter(StructuredFormatter(include_extra=True))
        logger.addHandler(handler)

        # Function for thread logging
        def log_messages(thread_id, count):
            for i in range(count):
                logger.info(
                    f"Message {i} from thread {thread_id}",
                    extra={"thread_id": thread_id, "message_num": i},
                )

        # Start multiple threads
        threads = []
        thread_count = 5
        messages_per_thread = 20

        for thread_id in range(thread_count):
            thread = threading.Thread(
                target=log_messages, args=(thread_id, messages_per_thread)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # Verify all messages were logged
        handler.flush()
        log_content = log_buffer.getvalue()
        log_lines = [line for line in log_content.split("\n") if line.strip()]

        assert len(log_lines) == thread_count * messages_per_thread

        # Verify JSON structure is maintained under concurrent access
        for line in log_lines:
            log_data = json.loads(line)
            assert "timestamp" in log_data
            assert "extra" in log_data
            assert "thread_id" in log_data["extra"]
            assert "message_num" in log_data["extra"]

    def test_logging_performance_under_load(self):
        """Test logging performance under high load."""
        log_buffer = StringIO()

        # Setup logger
        logger = logging.getLogger("performance_test")
        logger.setLevel(logging.INFO)

        handler = logging.StreamHandler(log_buffer)
        handler.setFormatter(StructuredFormatter(include_extra=True))
        logger.addHandler(handler)

        # Log many messages and measure time
        message_count = 10000
        start_time = time.time()

        for i in range(message_count):
            logger.info(
                f"Performance test message {i}",
                extra={
                    "iteration": i,
                    "batch": i // 100,
                    "test_data": {"nested": {"value": i}},
                },
            )

        end_time = time.time()
        processing_time = end_time - start_time
        messages_per_second = message_count / processing_time

        # Performance requirements
        assert processing_time < 10.0  # Should log 10K messages in < 10 seconds
        assert messages_per_second > 1000  # Should process > 1000 messages/second

        # Verify all messages were logged correctly
        handler.flush()
        log_lines = [line for line in log_buffer.getvalue().split("\n") if line.strip()]
        assert len(log_lines) == message_count

        print(
            f"Logged {message_count} messages in {processing_time:.3f}s "
            f"({messages_per_second:.1f} messages/sec)"
        )


class TestLoggingErrorHandling:
    """Test logging error handling and edge cases."""

    def test_structured_formatter_with_circular_references(self):
        """Test structured formatter handles circular references."""
        formatter = StructuredFormatter(include_extra=True)

        record = logging.LogRecord(
            name="test_logger",
            level=logging.INFO,
            pathname="/test/path.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None,
        )

        # Create circular reference
        circular_data = {"self": None}
        circular_data["self"] = circular_data
        record.circular_field = circular_data

        # Should handle gracefully without raising exception
        try:
            formatted = formatter.format(record)
            log_data = json.loads(formatted)

            # Should have converted to string via default=str
            assert "extra" in log_data
            assert "circular_field" in log_data["extra"]
            assert isinstance(log_data["extra"]["circular_field"], str)

        except (ValueError, TypeError, RecursionError) as e:
            pytest.fail(
                f"Structured formatter should handle circular references gracefully: {e}"
            )

    def test_logger_with_invalid_log_directory(self):
        """Test logger behavior with invalid log directory."""
        invalid_config_path = Path("/nonexistent/directory/logging.yaml")

        # Should handle gracefully and use fallback configuration
        manager = LoggerManager(invalid_config_path)

        # Should still create logger instances successfully
        logger = manager.get_logger("test")
        assert isinstance(logger, logging.Logger)

    def test_error_tracker_with_complex_exception(self):
        """Test error tracker with complex exception."""
        with patch("logging.Logger.error") as mock_error:
            tracker = ErrorTracker()

            # Create complex exception with nested causes
            try:
                try:
                    raise ValueError("Inner exception")
                except ValueError as inner:
                    raise RuntimeError("Outer exception") from inner
            except RuntimeError as outer_ex:
                tracker.track_error(outer_ex, context={"test": "complex"})

            # Should handle complex exception without issues
            mock_error.assert_called_once()
            args, kwargs = mock_error.call_args

            extra = kwargs["extra"]
            assert extra["error_type"] == "RuntimeError"
            assert extra["error_message"] == "Outer exception"


class TestLoggingConfigurationEdgeCases:
    """Test logging configuration edge cases."""

    def test_logger_manager_with_empty_config(self, temp_log_dir):
        """Test LoggerManager with empty configuration file."""
        config_path = temp_log_dir / "empty_logging.yaml"

        # Create empty config file
        with open(config_path, "w") as f:
            f.write("")

        # Should handle gracefully
        manager = LoggerManager(config_path)

        assert isinstance(manager.performance_logger, PerformanceLogger)
        assert isinstance(manager.error_tracker, ErrorTracker)

    def test_logger_manager_with_partial_config(self, temp_log_dir):
        """Test LoggerManager with partial configuration."""
        config_path = temp_log_dir / "partial_logging.yaml"

        partial_config = {
            "version": 1,
            "formatters": {"simple": {"format": "%(message)s"}},
            # Missing handlers, loggers, etc.
        }

        with open(config_path, "w") as f:
            yaml.dump(partial_config, f)

        # Should handle partial config and add missing pieces
        manager = LoggerManager(config_path)

        # Should still work
        logger = manager.get_logger("partial_test")
        assert isinstance(logger, logging.Logger)


# Test completion marker
def test_structured_logger_comprehensive_test_suite_completion():
    """Marker test to confirm comprehensive test suite completion."""
    test_classes = [
        TestStructuredFormatter,
        TestPerformanceLogger,
        TestErrorTracker,
        TestMLOperationsLogger,
        TestLoggerManager,
        TestGlobalLoggerFunctions,
        TestLoggingIntegration,
        TestLoggingErrorHandling,
        TestLoggingConfigurationEdgeCases,
    ]

    assert len(test_classes) == 9

    # Count total test methods
    total_methods = 0
    for test_class in test_classes:
        methods = [method for method in dir(test_class) if method.startswith("test_")]
        total_methods += len(methods)

    # Verify we have 40+ comprehensive test methods
    assert total_methods >= 40, f"Expected 40+ test methods, found {total_methods}"

    print(
        f"✅ StructuredLogger comprehensive test suite completed with {total_methods} test methods"
    )
