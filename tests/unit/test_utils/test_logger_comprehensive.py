"""
Comprehensive unit tests for logging utilities to achieve high test coverage.

This module focuses on comprehensive testing of all logging classes, error paths,
edge cases, and configuration variations in the logging utilities.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
import json
import logging
import logging.handlers
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch, MagicMock, call
import time

import pytest
import yaml

from src.utils.logger import (
    StructuredFormatter,
    PerformanceLogger,
    ErrorTracker,
    MLOperationsLogger,
    LoggerManager,
    get_logger_manager,
    get_logger,
    get_performance_logger,
    get_error_tracker,
    get_ml_ops_logger
)


class TestStructuredFormatter:
    """Test StructuredFormatter functionality."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        formatter = StructuredFormatter()
        
        assert formatter.include_extra is True

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        formatter = StructuredFormatter(include_extra=False)
        
        assert formatter.include_extra is False

    def test_format_basic_record(self):
        """Test formatting basic log record."""
        formatter = StructuredFormatter()
        
        # Create mock log record
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0  # Fixed timestamp
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert parsed["level"] == "INFO"
        assert parsed["logger"] == "test.logger"
        assert parsed["message"] == "Test message"
        assert parsed["module"] == "test_module"
        assert parsed["function"] == "test_function"
        assert parsed["line_number"] == 42
        assert parsed["thread"] == 12345
        assert parsed["thread_name"] == "MainThread"
        assert parsed["process"] == 54321
        assert "timestamp" in parsed

    def test_format_with_exception(self):
        """Test formatting log record with exception information."""
        import sys
        formatter = StructuredFormatter()
        
        try:
            raise ValueError("Test exception")
        except ValueError:
            record = logging.LogRecord(
                name="test.logger",
                level=logging.ERROR,
                pathname="/path/to/file.py",
                lineno=42,
                msg="Error occurred",
                args=(),
                exc_info=sys.exc_info()  # Get actual exception info tuple
            )
            record.module = "test_module"
            record.funcName = "test_function"
            record.thread = 12345
            record.threadName = "MainThread"
            record.process = 54321
            record.created = 1642694400.0
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert "exception" in parsed
        assert parsed["exception"]["type"] == "ValueError"
        assert parsed["exception"]["message"] == "Test exception"
        assert "traceback" in parsed["exception"]
        assert isinstance(parsed["exception"]["traceback"], list)

    def test_format_with_extra_fields_enabled(self):
        """Test formatting with extra fields enabled."""
        formatter = StructuredFormatter(include_extra=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0
        
        # Add custom fields
        record.room_id = "living_room"
        record.operation = "prediction"
        record.duration = 1.23
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert "extra" in parsed
        assert parsed["extra"]["room_id"] == "living_room"
        assert parsed["extra"]["operation"] == "prediction"
        assert parsed["extra"]["duration"] == 1.23

    def test_format_with_extra_fields_disabled(self):
        """Test formatting with extra fields disabled."""
        formatter = StructuredFormatter(include_extra=False)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0
        
        # Add custom fields (should be ignored)
        record.custom_field = "should_not_appear"
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert "extra" not in parsed

    def test_format_with_no_extra_fields(self):
        """Test formatting when no extra fields are present."""
        formatter = StructuredFormatter(include_extra=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        # Set required fields
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        # Should not have extra field when no custom fields
        assert "extra" not in parsed

    def test_format_with_complex_message(self):
        """Test formatting with complex message containing special characters."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg='Complex message with "quotes", newlines\n, and unicode: ñoño',
            args=(),
            exc_info=None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)  # Should not raise JSON decode error
        
        assert 'Complex message with "quotes", newlines\n, and unicode: ñoño' in parsed["message"]

    def test_format_filters_standard_fields(self):
        """Test that standard logging fields are filtered from extra."""
        formatter = StructuredFormatter(include_extra=True)
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.INFO,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test message",
            args=(),
            exc_info=None
        )
        
        # Set required fields to valid values
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0
        
        # Add one custom field that should appear in extra
        record.custom_field = "should_appear"
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        # Only custom field should appear in extra
        assert "extra" in parsed
        assert parsed["extra"] == {"custom_field": "should_appear"}


class TestPerformanceLogger:
    """Test PerformanceLogger functionality."""

    def test_init_default_name(self):
        """Test initialization with default logger name."""
        perf_logger = PerformanceLogger()
        
        assert perf_logger.logger.name == "occupancy_prediction.performance"

    def test_init_custom_name(self):
        """Test initialization with custom logger name."""
        perf_logger = PerformanceLogger("custom.performance")
        
        assert perf_logger.logger.name == "custom.performance"

    def test_log_operation_time_basic(self):
        """Test basic operation time logging."""
        perf_logger = PerformanceLogger()
        
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_operation_time("test_operation", 1.23)
            
            mock_info.assert_called_once()
            call_args = mock_info.call_args
            assert "Operation completed: test_operation" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["operation"] == "test_operation"
            assert extra["duration_seconds"] == 1.23
            assert extra["metric_type"] == "performance"

    def test_log_operation_time_full_params(self):
        """Test operation time logging with all parameters."""
        perf_logger = PerformanceLogger()
        
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_operation_time(
                "prediction",
                2.45,
                room_id="bedroom",
                prediction_type="next_occupied",
                model_type="xgboost",
                accuracy=0.85
            )
            
            extra = mock_info.call_args[1]["extra"]
            assert extra["operation"] == "prediction"
            assert extra["duration_seconds"] == 2.45
            assert extra["room_id"] == "bedroom"
            assert extra["prediction_type"] == "next_occupied"
            assert extra["model_type"] == "xgboost"
            assert extra["accuracy"] == 0.85

    def test_log_prediction_accuracy(self):
        """Test prediction accuracy logging."""
        perf_logger = PerformanceLogger()
        
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_prediction_accuracy(
                "kitchen", 12.5, 0.92, "next_vacant"
            )
            
            call_args = mock_info.call_args
            assert "Prediction accuracy: 12.50 minutes" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["room_id"] == "kitchen"
            assert extra["accuracy_minutes"] == 12.5
            assert extra["confidence"] == 0.92
            assert extra["prediction_type"] == "next_vacant"
            assert extra["metric_type"] == "accuracy"

    def test_log_model_metrics(self):
        """Test model metrics logging."""
        perf_logger = PerformanceLogger()
        
        metrics = {
            "r2_score": 0.85,
            "mae": 8.5,
            "rmse": 12.3
        }
        
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_model_metrics("office", "lstm", metrics)
            
            call_args = mock_info.call_args
            assert "Model metrics for lstm" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["room_id"] == "office"
            assert extra["model_type"] == "lstm"
            assert extra["metrics"] == metrics
            assert extra["metric_type"] == "model_performance"

    def test_log_resource_usage(self):
        """Test resource usage logging."""
        perf_logger = PerformanceLogger()
        
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_resource_usage(75.5, 1024.0, 45.2)
            
            call_args = mock_info.call_args
            assert "System resource usage" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["cpu_percent"] == 75.5
            assert extra["memory_mb"] == 1024.0
            assert extra["disk_usage_percent"] == 45.2
            assert extra["metric_type"] == "resource_usage"


class TestErrorTracker:
    """Test ErrorTracker functionality."""

    def test_init_default_name(self):
        """Test initialization with default logger name."""
        error_tracker = ErrorTracker()
        
        assert error_tracker.logger.name == "occupancy_prediction.errors"

    def test_init_custom_name(self):
        """Test initialization with custom logger name."""
        error_tracker = ErrorTracker("custom.errors")
        
        assert error_tracker.logger.name == "custom.errors"

    def test_track_error_basic(self):
        """Test basic error tracking."""
        error_tracker = ErrorTracker()
        error = ValueError("Test error")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_error(error)
            
            call_args = mock_error.call_args
            assert "Error tracked: Test error" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["error_type"] == "ValueError"
            assert extra["error_message"] == "Test error"
            assert extra["severity"] == "error"
            assert extra["context"] == {}
            assert extra["alert_required"] is False
            assert extra["metric_type"] == "error"
            assert call_args[1]["exc_info"] is True

    def test_track_error_with_context(self):
        """Test error tracking with context."""
        error_tracker = ErrorTracker()
        error = RuntimeError("Runtime error")
        context = {"room_id": "bathroom", "operation": "training"}
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_error(
                error, 
                context=context, 
                severity="warning", 
                alert=True
            )
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["error_type"] == "RuntimeError"
            assert extra["context"] == context
            assert extra["severity"] == "warning"
            assert extra["alert_required"] is True

    def test_track_error_critical_severity(self):
        """Test error tracking with critical severity."""
        error_tracker = ErrorTracker()
        error = Exception("Critical error")
        
        with patch.object(error_tracker.logger, 'critical') as mock_critical:
            error_tracker.track_error(error, severity="critical")
            
            call_args = mock_critical.call_args
            assert "Critical error: Critical error" in call_args[0][0]
            assert call_args[1]["exc_info"] is True

    def test_track_prediction_error(self):
        """Test prediction error tracking."""
        error_tracker = ErrorTracker()
        error = Exception("Prediction failed")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_prediction_error(
                "garage", error, "next_occupied", "random_forest"
            )
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["context"]["room_id"] == "garage"
            assert extra["context"]["prediction_type"] == "next_occupied"
            assert extra["context"]["model_type"] == "random_forest"
            assert extra["context"]["component"] == "prediction_engine"
            assert extra["severity"] == "error"
            assert extra["alert_required"] is True

    def test_track_data_error(self):
        """Test data error tracking."""
        error_tracker = ErrorTracker()
        error = ConnectionError("Data source unavailable")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_data_error(
                error, "home_assistant", "sensor.motion_1"
            )
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["context"]["data_source"] == "home_assistant"
            assert extra["context"]["entity_id"] == "sensor.motion_1"
            assert extra["context"]["component"] == "data_ingestion"
            assert extra["severity"] == "warning"
            assert extra["alert_required"] is False

    def test_track_integration_error(self):
        """Test integration error tracking."""
        error_tracker = ErrorTracker()
        error = TimeoutError("MQTT timeout")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_integration_error(
                error, "mqtt", "mqtt://broker:1883"
            )
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["context"]["integration_type"] == "mqtt"
            assert extra["context"]["endpoint"] == "mqtt://broker:1883"
            assert extra["context"]["component"] == "integration"
            assert extra["severity"] == "error"
            assert extra["alert_required"] is True

    def test_track_data_error_no_entity_id(self):
        """Test data error tracking without entity ID."""
        error_tracker = ErrorTracker()
        error = Exception("General data error")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_data_error(error, "database")
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["context"]["data_source"] == "database"
            assert extra["context"]["entity_id"] is None

    def test_track_integration_error_no_endpoint(self):
        """Test integration error tracking without endpoint."""
        error_tracker = ErrorTracker()
        error = Exception("General integration error")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_integration_error(error, "api")
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["context"]["integration_type"] == "api"
            assert extra["context"]["endpoint"] is None


class TestMLOperationsLogger:
    """Test MLOperationsLogger functionality."""

    def test_init_default_name(self):
        """Test initialization with default logger name."""
        ml_logger = MLOperationsLogger()
        
        assert ml_logger.logger.name == "occupancy_prediction.ml_ops"

    def test_init_custom_name(self):
        """Test initialization with custom logger name."""
        ml_logger = MLOperationsLogger("custom.ml_ops")
        
        assert ml_logger.logger.name == "custom.ml_ops"

    def test_log_training_event_basic(self):
        """Test basic training event logging."""
        ml_logger = MLOperationsLogger()
        
        with patch.object(ml_logger.logger, 'info') as mock_info:
            ml_logger.log_training_event("study", "xgboost", "training_started")
            
            call_args = mock_info.call_args
            assert "Training event: training_started for xgboost" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["room_id"] == "study"
            assert extra["model_type"] == "xgboost"
            assert extra["event_type"] == "training_started"
            assert extra["metrics"] == {}
            assert extra["component"] == "training"
            assert extra["metric_type"] == "ml_lifecycle"

    def test_log_training_event_with_metrics(self):
        """Test training event logging with metrics."""
        ml_logger = MLOperationsLogger()
        metrics = {"accuracy": 0.92, "loss": 0.15}
        
        with patch.object(ml_logger.logger, 'info') as mock_info:
            ml_logger.log_training_event(
                "basement", "lstm", "training_completed", metrics
            )
            
            extra = mock_info.call_args[1]["extra"]
            assert extra["metrics"] == metrics

    def test_log_drift_detection(self):
        """Test drift detection logging."""
        ml_logger = MLOperationsLogger()
        
        with patch.object(ml_logger.logger, 'warning') as mock_warning:
            ml_logger.log_drift_detection(
                "attic", "data_drift", 0.75, "retrain_scheduled"
            )
            
            call_args = mock_warning.call_args
            assert "Concept drift detected: data_drift" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["room_id"] == "attic"
            assert extra["drift_type"] == "data_drift"
            assert extra["severity"] == 0.75
            assert extra["action_taken"] == "retrain_scheduled"
            assert extra["component"] == "adaptation"
            assert extra["metric_type"] == "drift_detection"

    def test_log_model_deployment(self):
        """Test model deployment logging."""
        ml_logger = MLOperationsLogger()
        performance_metrics = {"r2": 0.88, "mae": 9.2}
        
        with patch.object(ml_logger.logger, 'info') as mock_info:
            ml_logger.log_model_deployment(
                "patio", "svm", "v2.1", performance_metrics
            )
            
            call_args = mock_info.call_args
            assert "Model deployed: svm vv2.1" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["room_id"] == "patio"
            assert extra["model_type"] == "svm"
            assert extra["version"] == "v2.1"
            assert extra["performance_metrics"] == performance_metrics
            assert extra["component"] == "deployment"
            assert extra["metric_type"] == "ml_lifecycle"

    def test_log_feature_importance(self):
        """Test feature importance logging."""
        ml_logger = MLOperationsLogger()
        feature_importance = {
            "time_since_last_motion": 0.45,
            "hour_sin": 0.32,
            "day_of_week_cos": 0.23
        }
        
        with patch.object(ml_logger.logger, 'info') as mock_info:
            ml_logger.log_feature_importance(
                "workshop", "gradient_boost", feature_importance
            )
            
            call_args = mock_info.call_args
            assert "Feature importance analysis" in call_args[0][0]
            
            extra = call_args[1]["extra"]
            assert extra["room_id"] == "workshop"
            assert extra["model_type"] == "gradient_boost"
            assert extra["feature_importance"] == feature_importance
            assert extra["component"] == "analysis"
            assert extra["metric_type"] == "feature_analysis"


class TestLoggerManager:
    """Test LoggerManager functionality."""

    def test_init_default_config_path(self):
        """Test initialization with default config path."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
            
            assert manager.config_path == Path("config/logging.yaml")
            assert isinstance(manager.performance_logger, PerformanceLogger)
            assert isinstance(manager.error_tracker, ErrorTracker)
            assert isinstance(manager.ml_ops_logger, MLOperationsLogger)

    def test_init_custom_config_path(self):
        """Test initialization with custom config path."""
        custom_path = Path("custom/logging.yaml")
        
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager(custom_path)
            
            assert manager.config_path == custom_path

    def test_setup_logging_config_exists(self):
        """Test setup logging when config file exists."""
        config_data = {
            "version": 1,
            "handlers": {
                "file_handler": {
                    "class": "logging.FileHandler",
                    "filename": "test.log"
                }
            },
            "root": {
                "level": "INFO",
                "handlers": ["file_handler"]
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            with patch('logging.config.dictConfig') as mock_dict_config, \
                 patch('pathlib.Path.mkdir') as mock_mkdir:
                
                manager = LoggerManager(config_path)
                
                # Should have called dictConfig
                mock_dict_config.assert_called_once()
                config_used = mock_dict_config.call_args[0][0]
                
                # Should have added JSON formatter
                assert "json" in config_used["formatters"]
                assert config_used["formatters"]["json"]["()"] == "src.utils.logger.StructuredFormatter"
                
                # Should have created logs directory
                mock_mkdir.assert_called_once()
                
        finally:
            config_path.unlink()

    def test_setup_logging_config_not_exists(self):
        """Test setup logging when config file doesn't exist."""
        non_existent_path = Path("nonexistent/logging.yaml")
        
        with patch('logging.basicConfig') as mock_basic_config:
            manager = LoggerManager(non_existent_path)
            
            # Should fall back to basic configuration
            mock_basic_config.assert_called_once()
            call_kwargs = mock_basic_config.call_args[1]
            assert call_kwargs["level"] == logging.INFO

    def test_setup_logging_config_error(self):
        """Test setup logging when config loading fails."""
        # Create invalid YAML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write("invalid: yaml: content: [")
            config_path = Path(f.name)
        
        try:
            with patch('logging.basicConfig') as mock_basic_config, \
                 patch('builtins.print') as mock_print:
                
                manager = LoggerManager(config_path)
                
                # Should fall back to basic config and print warning
                mock_basic_config.assert_called_once()
                mock_print.assert_called_once()
                assert "Failed to load logging configuration" in mock_print.call_args[0][0]
                
        finally:
            config_path.unlink()

    def test_setup_logging_adds_json_formatter_to_file_handlers(self):
        """Test that JSON formatter is added to file handlers."""
        config_data = {
            "version": 1,
            "formatters": {
                "simple": {"format": "%(message)s"}
            },
            "handlers": {
                "file_handler": {
                    "class": "logging.FileHandler",
                    "filename": "test.log",
                    "formatter": "simple"
                },
                "console_handler": {
                    "class": "logging.StreamHandler",
                    "formatter": "simple"
                }
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            with patch('logging.config.dictConfig') as mock_dict_config, \
                 patch('pathlib.Path.mkdir'):
                
                manager = LoggerManager(config_path)
                
                config_used = mock_dict_config.call_args[0][0]
                
                # File handler should use JSON formatter
                assert config_used["handlers"]["file_handler"]["formatter"] == "json"
                # Console handler should keep original formatter
                assert config_used["handlers"]["console_handler"]["formatter"] == "simple"
                
        finally:
            config_path.unlink()

    def test_get_logger(self):
        """Test getting logger with consistent naming."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
            
            with patch('logging.getLogger') as mock_get_logger:
                mock_logger = Mock()
                mock_get_logger.return_value = mock_logger
                
                logger = manager.get_logger("test_module")
                
                mock_get_logger.assert_called_once_with("occupancy_prediction.test_module")
                assert logger == mock_logger

    def test_get_performance_logger(self):
        """Test getting performance logger."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
            
            perf_logger = manager.get_performance_logger()
            
            assert perf_logger == manager.performance_logger

    def test_get_error_tracker(self):
        """Test getting error tracker."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
            
            error_tracker = manager.get_error_tracker()
            
            assert error_tracker == manager.error_tracker

    def test_get_ml_ops_logger(self):
        """Test getting ML operations logger."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
            
            ml_ops_logger = manager.get_ml_ops_logger()
            
            assert ml_ops_logger == manager.ml_ops_logger

    def test_log_operation_context_manager_success(self):
        """Test log operation context manager with successful operation."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
        
        with patch.object(manager, 'get_logger') as mock_get_logger, \
             patch.object(manager.performance_logger, 'log_operation_time') as mock_log_time, \
             patch('time.time', side_effect=[1000.0, 1002.5]):  # 2.5 second operation
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with manager.log_operation("test_operation", "test_room"):
                pass  # Simulate successful operation
            
            # Should log start and completion
            assert mock_logger.info.call_count == 2
            
            # Check start log
            start_call = mock_logger.info.call_args_list[0]
            assert "Starting operation: test_operation" in start_call[0][0]
            assert start_call[1]["extra"]["event_type"] == "operation_start"
            
            # Check completion log
            completion_call = mock_logger.info.call_args_list[1]
            assert "Completed operation: test_operation" in completion_call[0][0]
            assert completion_call[1]["extra"]["event_type"] == "operation_complete"
            assert completion_call[1]["extra"]["duration_seconds"] == 2.5
            
            # Should log performance timing
            mock_log_time.assert_called_once_with("test_operation", 2.5, "test_room")

    def test_log_operation_context_manager_with_exception(self):
        """Test log operation context manager with exception."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
        
        test_error = ValueError("Test error")
        
        with patch.object(manager, 'get_logger') as mock_get_logger, \
             patch.object(manager.error_tracker, 'track_error') as mock_track_error, \
             patch('time.time', side_effect=[1000.0, 1001.5]):  # 1.5 second before error
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with pytest.raises(ValueError):
                with manager.log_operation("failing_operation", "error_room"):
                    raise test_error
            
            # Should log start only (no completion)
            mock_logger.info.assert_called_once()
            start_call = mock_logger.info.call_args
            assert "Starting operation: failing_operation" in start_call[0][0]
            
            # Should track error
            mock_track_error.assert_called_once()
            error_call = mock_track_error.call_args
            assert error_call[0][0] == test_error
            assert error_call[0][1]["operation"] == "failing_operation"
            assert error_call[0][1]["room_id"] == "error_room"
            assert error_call[0][1]["duration_seconds"] == 1.5

    def test_log_operation_context_manager_no_room_id(self):
        """Test log operation context manager without room ID."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
        
        with patch.object(manager, 'get_logger') as mock_get_logger, \
             patch.object(manager.performance_logger, 'log_operation_time') as mock_log_time, \
             patch('time.time', side_effect=[1000.0, 1001.0]):
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with manager.log_operation("test_operation"):
                pass
            
            # Should log with None room_id
            start_call = mock_logger.info.call_args_list[0]
            assert start_call[1]["extra"]["room_id"] is None
            
            mock_log_time.assert_called_once_with("test_operation", 1.0, None)


class TestGlobalLoggerFunctions:
    """Test global logger convenience functions."""

    def test_get_logger_manager_singleton(self):
        """Test that get_logger_manager returns singleton."""
        # Clear global instance
        import src.utils.logger
        src.utils.logger._logger_manager = None
        
        with patch.object(LoggerManager, '_setup_logging'):
            # First call should create instance
            manager1 = get_logger_manager()
            
            # Second call should return same instance
            manager2 = get_logger_manager()
            
            assert manager1 is manager2
            assert isinstance(manager1, LoggerManager)

    def test_get_logger_convenience(self):
        """Test get_logger convenience function."""
        mock_manager = Mock()
        mock_logger = Mock()
        mock_manager.get_logger.return_value = mock_logger
        
        with patch('src.utils.logger.get_logger_manager', return_value=mock_manager):
            logger = get_logger("test_module")
            
            mock_manager.get_logger.assert_called_once_with("test_module")
            assert logger == mock_logger

    def test_get_performance_logger_convenience(self):
        """Test get_performance_logger convenience function."""
        mock_manager = Mock()
        mock_perf_logger = Mock()
        mock_manager.get_performance_logger.return_value = mock_perf_logger
        
        with patch('src.utils.logger.get_logger_manager', return_value=mock_manager):
            perf_logger = get_performance_logger()
            
            mock_manager.get_performance_logger.assert_called_once()
            assert perf_logger == mock_perf_logger

    def test_get_error_tracker_convenience(self):
        """Test get_error_tracker convenience function."""
        mock_manager = Mock()
        mock_error_tracker = Mock()
        mock_manager.get_error_tracker.return_value = mock_error_tracker
        
        with patch('src.utils.logger.get_logger_manager', return_value=mock_manager):
            error_tracker = get_error_tracker()
            
            mock_manager.get_error_tracker.assert_called_once()
            assert error_tracker == mock_error_tracker

    def test_get_ml_ops_logger_convenience(self):
        """Test get_ml_ops_logger convenience function."""
        mock_manager = Mock()
        mock_ml_ops_logger = Mock()
        mock_manager.get_ml_ops_logger.return_value = mock_ml_ops_logger
        
        with patch('src.utils.logger.get_logger_manager', return_value=mock_manager):
            ml_ops_logger = get_ml_ops_logger()
            
            mock_manager.get_ml_ops_logger.assert_called_once()
            assert ml_ops_logger == mock_ml_ops_logger


class TestLoggerEdgeCases:
    """Test edge cases and error conditions."""

    def test_structured_formatter_with_none_exception_info(self):
        """Test structured formatter with None exception info components."""
        formatter = StructuredFormatter()
        
        record = logging.LogRecord(
            name="test.logger",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Error message",
            args=(),
            exc_info=(None, None, None)  # All None
        )
        record.module = "test_module"
        record.funcName = "test_function"
        record.thread = 12345
        record.threadName = "MainThread"
        record.process = 54321
        record.created = 1642694400.0
        
        formatted = formatter.format(record)
        parsed = json.loads(formatted)
        
        assert "exception" in parsed
        assert parsed["exception"]["type"] is None
        assert parsed["exception"]["message"] is None

    def test_performance_logger_with_none_values(self):
        """Test performance logger handling None values."""
        perf_logger = PerformanceLogger()
        
        with patch.object(perf_logger.logger, 'info') as mock_info:
            perf_logger.log_operation_time("test", 1.0, room_id=None, prediction_type=None)
            
            extra = mock_info.call_args[1]["extra"]
            assert extra["room_id"] is None
            assert extra["prediction_type"] is None

    def test_error_tracker_with_empty_context(self):
        """Test error tracker with empty context."""
        error_tracker = ErrorTracker()
        error = Exception("Test error")
        
        with patch.object(error_tracker.logger, 'error') as mock_error:
            error_tracker.track_error(error, context=None)
            
            extra = mock_error.call_args[1]["extra"]
            assert extra["context"] == {}

    def test_ml_ops_logger_with_none_metrics(self):
        """Test ML ops logger with None metrics."""
        ml_logger = MLOperationsLogger()
        
        with patch.object(ml_logger.logger, 'info') as mock_info:
            ml_logger.log_training_event("room", "model", "event", metrics=None)
            
            extra = mock_info.call_args[1]["extra"]
            assert extra["metrics"] == {}

    def test_logger_manager_config_with_no_handlers(self):
        """Test logger manager config with no handlers section."""
        config_data = {
            "version": 1,
            "formatters": {
                "simple": {"format": "%(message)s"}
            },
            "root": {"level": "INFO"}
            # No handlers section
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)
        
        try:
            with patch('logging.config.dictConfig') as mock_dict_config, \
                 patch('pathlib.Path.mkdir'):
                
                manager = LoggerManager(config_path)
                
                # Should not crash and should still add JSON formatter
                config_used = mock_dict_config.call_args[0][0]
                assert "json" in config_used["formatters"]
                
        finally:
            config_path.unlink()

    def test_log_operation_zero_duration(self):
        """Test log operation with zero duration."""
        with patch.object(LoggerManager, '_setup_logging'):
            manager = LoggerManager()
        
        with patch.object(manager, 'get_logger') as mock_get_logger, \
             patch.object(manager.performance_logger, 'log_operation_time') as mock_log_time, \
             patch('time.time', side_effect=[1000.0, 1000.0]):  # Same time = 0 duration
            
            mock_logger = Mock()
            mock_get_logger.return_value = mock_logger
            
            with manager.log_operation("instant_operation"):
                pass
            
            # Should handle zero duration gracefully
            mock_log_time.assert_called_once_with("instant_operation", 0.0, None)
            completion_call = mock_logger.info.call_args_list[1]
            assert completion_call[1]["extra"]["duration_seconds"] == 0.0