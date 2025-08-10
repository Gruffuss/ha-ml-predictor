"""
Production-grade structured logging system for Home Assistant ML Predictor.
Provides JSON formatted logging, centralized configuration, and performance monitoring.
"""

import json
import logging
import logging.config
import logging.handlers
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Union
from contextlib import contextmanager
import time

import yaml


class StructuredFormatter(logging.Formatter):
    """Custom formatter that outputs structured JSON logs."""
    
    def __init__(self, include_extra: bool = True):
        super().__init__()
        self.include_extra = include_extra
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as structured JSON."""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line_number': record.lineno,
            'thread': record.thread,
            'thread_name': record.threadName,
            'process': record.process,
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = {
                'type': record.exc_info[0].__name__ if record.exc_info[0] else None,
                'message': str(record.exc_info[1]) if record.exc_info[1] else None,
                'traceback': traceback.format_exception(*record.exc_info)
            }
        
        # Add extra fields if present and enabled
        if self.include_extra:
            extra_fields = {
                key: value for key, value in record.__dict__.items()
                if key not in [
                    'name', 'msg', 'args', 'levelname', 'levelno', 'pathname',
                    'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                    'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                    'thread', 'threadName', 'processName', 'process', 'getMessage'
                ]
            }
            if extra_fields:
                log_entry['extra'] = extra_fields
        
        return json.dumps(log_entry, default=str, separators=(',', ':'))


class PerformanceLogger:
    """Logger for performance monitoring and metrics."""
    
    def __init__(self, logger_name: str = "occupancy_prediction.performance"):
        self.logger = logging.getLogger(logger_name)
    
    def log_operation_time(self, operation: str, duration: float, 
                          room_id: Optional[str] = None, 
                          prediction_type: Optional[str] = None,
                          **kwargs):
        """Log operation timing with structured metadata."""
        self.logger.info(
            f"Operation completed: {operation}",
            extra={
                'operation': operation,
                'duration_seconds': duration,
                'room_id': room_id,
                'prediction_type': prediction_type,
                'metric_type': 'performance',
                **kwargs
            }
        )
    
    def log_prediction_accuracy(self, room_id: str, accuracy_minutes: float,
                               confidence: float, prediction_type: str):
        """Log prediction accuracy metrics."""
        self.logger.info(
            f"Prediction accuracy: {accuracy_minutes:.2f} minutes",
            extra={
                'room_id': room_id,
                'accuracy_minutes': accuracy_minutes,
                'confidence': confidence,
                'prediction_type': prediction_type,
                'metric_type': 'accuracy'
            }
        )
    
    def log_model_metrics(self, room_id: str, model_type: str, metrics: Dict[str, float]):
        """Log model performance metrics."""
        self.logger.info(
            f"Model metrics for {model_type}",
            extra={
                'room_id': room_id,
                'model_type': model_type,
                'metrics': metrics,
                'metric_type': 'model_performance'
            }
        )
    
    def log_resource_usage(self, cpu_percent: float, memory_mb: float, 
                          disk_usage_percent: float):
        """Log system resource usage."""
        self.logger.info(
            "System resource usage",
            extra={
                'cpu_percent': cpu_percent,
                'memory_mb': memory_mb,
                'disk_usage_percent': disk_usage_percent,
                'metric_type': 'resource_usage'
            }
        )


class ErrorTracker:
    """Centralized error tracking and alerting."""
    
    def __init__(self, logger_name: str = "occupancy_prediction.errors"):
        self.logger = logging.getLogger(logger_name)
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None,
                   severity: str = "error", alert: bool = False):
        """Track error with context and optional alerting."""
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'severity': severity,
            'context': context or {},
            'alert_required': alert,
            'metric_type': 'error'
        }
        
        if severity == "critical":
            self.logger.critical(
                f"Critical error: {error}",
                extra=error_info,
                exc_info=True
            )
        else:
            self.logger.error(
                f"Error tracked: {error}",
                extra=error_info,
                exc_info=True
            )
    
    def track_prediction_error(self, room_id: str, error: Exception,
                             prediction_type: str, model_type: Optional[str] = None):
        """Track prediction-specific errors."""
        context = {
            'room_id': room_id,
            'prediction_type': prediction_type,
            'model_type': model_type,
            'component': 'prediction_engine'
        }
        self.track_error(error, context, severity="error", alert=True)
    
    def track_data_error(self, error: Exception, data_source: str, 
                        entity_id: Optional[str] = None):
        """Track data ingestion errors."""
        context = {
            'data_source': data_source,
            'entity_id': entity_id,
            'component': 'data_ingestion'
        }
        self.track_error(error, context, severity="warning", alert=False)
    
    def track_integration_error(self, error: Exception, integration_type: str,
                               endpoint: Optional[str] = None):
        """Track integration errors (MQTT, HA API, etc.)."""
        context = {
            'integration_type': integration_type,
            'endpoint': endpoint,
            'component': 'integration'
        }
        self.track_error(error, context, severity="error", alert=True)


class MLOperationsLogger:
    """Specialized logger for ML operations and lifecycle events."""
    
    def __init__(self, logger_name: str = "occupancy_prediction.ml_ops"):
        self.logger = logging.getLogger(logger_name)
    
    def log_training_event(self, room_id: str, model_type: str, 
                          event_type: str, metrics: Dict[str, float] = None):
        """Log model training events."""
        self.logger.info(
            f"Training event: {event_type} for {model_type}",
            extra={
                'room_id': room_id,
                'model_type': model_type,
                'event_type': event_type,
                'metrics': metrics or {},
                'component': 'training',
                'metric_type': 'ml_lifecycle'
            }
        )
    
    def log_drift_detection(self, room_id: str, drift_type: str, 
                           severity: float, action_taken: str):
        """Log concept drift detection events."""
        self.logger.warning(
            f"Concept drift detected: {drift_type}",
            extra={
                'room_id': room_id,
                'drift_type': drift_type,
                'severity': severity,
                'action_taken': action_taken,
                'component': 'adaptation',
                'metric_type': 'drift_detection'
            }
        )
    
    def log_model_deployment(self, room_id: str, model_type: str, 
                           version: str, performance_metrics: Dict[str, float]):
        """Log model deployment events."""
        self.logger.info(
            f"Model deployed: {model_type} v{version}",
            extra={
                'room_id': room_id,
                'model_type': model_type,
                'version': version,
                'performance_metrics': performance_metrics,
                'component': 'deployment',
                'metric_type': 'ml_lifecycle'
            }
        )
    
    def log_feature_importance(self, room_id: str, model_type: str,
                             feature_importance: Dict[str, float]):
        """Log feature importance analysis."""
        self.logger.info(
            "Feature importance analysis",
            extra={
                'room_id': room_id,
                'model_type': model_type,
                'feature_importance': feature_importance,
                'component': 'analysis',
                'metric_type': 'feature_analysis'
            }
        )


class LoggerManager:
    """Centralized logging configuration and management."""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("config/logging.yaml")
        self.performance_logger = PerformanceLogger()
        self.error_tracker = ErrorTracker()
        self.ml_ops_logger = MLOperationsLogger()
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration from YAML file."""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    config = yaml.safe_load(f)
                
                # Ensure logs directory exists
                logs_dir = Path("logs")
                logs_dir.mkdir(exist_ok=True)
                
                # Add structured formatter for production
                if 'formatters' not in config:
                    config['formatters'] = {}
                
                config['formatters']['json'] = {
                    '()': 'src.utils.logger.StructuredFormatter',
                    'include_extra': True
                }
                
                # Configure handlers to use JSON formatter for file outputs
                for handler_name, handler_config in config.get('handlers', {}).items():
                    if 'file' in handler_name.lower():
                        handler_config['formatter'] = 'json'
                
                logging.config.dictConfig(config)
            else:
                # Fallback basic configuration
                logging.basicConfig(
                    level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('logs/occupancy_prediction.log')
                    ]
                )
        
        except Exception as e:
            # Fallback to basic logging if configuration fails
            print(f"Warning: Failed to load logging configuration: {e}")
            logging.basicConfig(level=logging.INFO)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get logger instance with consistent naming."""
        return logging.getLogger(f"occupancy_prediction.{name}")
    
    def get_performance_logger(self) -> PerformanceLogger:
        """Get performance logger instance."""
        return self.performance_logger
    
    def get_error_tracker(self) -> ErrorTracker:
        """Get error tracker instance."""
        return self.error_tracker
    
    def get_ml_ops_logger(self) -> MLOperationsLogger:
        """Get ML operations logger instance."""
        return self.ml_ops_logger
    
    @contextmanager
    def log_operation(self, operation_name: str, room_id: Optional[str] = None):
        """Context manager to automatically log operation timing."""
        start_time = time.time()
        logger = self.get_logger("operations")
        
        try:
            logger.info(f"Starting operation: {operation_name}", extra={
                'operation': operation_name,
                'room_id': room_id,
                'event_type': 'operation_start'
            })
            yield
            
        except Exception as e:
            duration = time.time() - start_time
            self.error_tracker.track_error(e, {
                'operation': operation_name,
                'room_id': room_id,
                'duration_seconds': duration
            })
            raise
            
        else:
            duration = time.time() - start_time
            self.performance_logger.log_operation_time(
                operation_name, duration, room_id
            )
            logger.info(f"Completed operation: {operation_name}", extra={
                'operation': operation_name,
                'room_id': room_id,
                'duration_seconds': duration,
                'event_type': 'operation_complete'
            })


# Global logger manager instance
_logger_manager = None

def get_logger_manager() -> LoggerManager:
    """Get global logger manager instance."""
    global _logger_manager
    if _logger_manager is None:
        _logger_manager = LoggerManager()
    return _logger_manager

def get_logger(name: str) -> logging.Logger:
    """Convenience function to get logger."""
    return get_logger_manager().get_logger(name)

def get_performance_logger() -> PerformanceLogger:
    """Convenience function to get performance logger."""
    return get_logger_manager().get_performance_logger()

def get_error_tracker() -> ErrorTracker:
    """Convenience function to get error tracker."""
    return get_logger_manager().get_error_tracker()

def get_ml_ops_logger() -> MLOperationsLogger:
    """Convenience function to get ML operations logger."""
    return get_logger_manager().get_ml_ops_logger()