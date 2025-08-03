"""
System-wide tracking manager for integrated accuracy monitoring.

This module provides centralized coordination of accuracy tracking across
the entire prediction system, automatically handling prediction recording,
validation, and real-time monitoring without manual intervention.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass
import threading

from ..core.exceptions import OccupancyPredictionError, ErrorSeverity
from ..core.constants import ModelType
from .validator import PredictionValidator
from .tracker import AccuracyTracker, AccuracyAlert, RealTimeMetrics
from ..models.base.predictor import PredictionResult
from ..data.storage.models import SensorEvent, RoomState


logger = logging.getLogger(__name__)


@dataclass
class TrackingConfig:
    """Configuration for system-wide tracking."""
    enabled: bool = True
    monitoring_interval_seconds: int = 60
    auto_validation_enabled: bool = True
    validation_window_minutes: int = 30
    alert_thresholds: Dict[str, float] = None
    max_stored_alerts: int = 1000
    trend_analysis_points: int = 10
    cleanup_interval_hours: int = 24
    
    def __post_init__(self):
        """Set default alert thresholds if not provided."""
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                'accuracy_warning': 70.0,
                'accuracy_critical': 50.0,
                'error_warning': 20.0,
                'error_critical': 30.0,
                'trend_degrading': -5.0,
                'validation_lag_warning': 15.0,
                'validation_lag_critical': 30.0
            }


class TrackingManager:
    """
    Centralized tracking manager for automatic accuracy monitoring.
    
    Provides seamless integration between prediction generation, validation,
    and real-time monitoring across the entire occupancy prediction system.
    
    Features:
    - Automatic prediction recording when ensemble models make predictions
    - Real-time validation based on actual room state changes
    - Background monitoring and alerting
    - Event-driven accuracy updates
    - System-wide tracking coordination
    """
    
    def __init__(
        self,
        config: TrackingConfig,
        database_manager=None,
        notification_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize the tracking manager.
        
        Args:
            config: Tracking configuration
            database_manager: Database manager for accessing room states
            notification_callbacks: List of notification callbacks for alerts
        """
        self.config = config
        self.database_manager = database_manager
        self.notification_callbacks = notification_callbacks or []
        
        # Initialize core tracking components
        self.validator: Optional[PredictionValidator] = None
        self.accuracy_tracker: Optional[AccuracyTracker] = None
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._tracking_active = False
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Prediction cache for validation correlation
        self._pending_predictions: Dict[str, List[PredictionResult]] = {}
        self._prediction_cache_lock = threading.RLock()
        
        # Performance metrics
        self._total_predictions_recorded = 0
        self._total_validations_performed = 0
        self._system_start_time = datetime.utcnow()
        
        logger.info(f"Initialized TrackingManager with config: {config}")
    
    async def initialize(self) -> None:
        """Initialize tracking components and start monitoring."""
        try:
            if not self.config.enabled:
                logger.info("Tracking is disabled in configuration")
                return
            
            # Initialize validator
            self.validator = PredictionValidator(
                accuracy_threshold_minutes=15
            )
            
            # Initialize accuracy tracker
            self.accuracy_tracker = AccuracyTracker(
                prediction_validator=self.validator,
                monitoring_interval_seconds=self.config.monitoring_interval_seconds,
                alert_thresholds=self.config.alert_thresholds,
                max_stored_alerts=self.config.max_stored_alerts,
                trend_analysis_points=self.config.trend_analysis_points,
                notification_callbacks=self.notification_callbacks
            )
            
            # Start tracking systems
            await self.start_tracking()
            
            logger.info("TrackingManager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize TrackingManager: {e}")
            raise TrackingManagerError("Failed to initialize tracking manager", cause=e)
    
    async def start_tracking(self) -> None:
        """Start background tracking tasks."""
        try:
            if self._tracking_active:
                logger.warning("Tracking already active")
                return
            
            if not self.config.enabled:
                logger.info("Tracking is disabled, not starting monitoring")
                return
            
            # Start accuracy tracker monitoring
            if self.accuracy_tracker:
                await self.accuracy_tracker.start_monitoring()
            
            # Start validation monitoring task
            validation_task = asyncio.create_task(self._validation_monitoring_loop())
            self._background_tasks.append(validation_task)
            
            # Start cleanup task
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.append(cleanup_task)
            
            self._tracking_active = True
            logger.info("Started TrackingManager monitoring tasks")
            
        except Exception as e:
            logger.error(f"Failed to start tracking: {e}")
            raise TrackingManagerError("Failed to start tracking", cause=e)
    
    async def stop_tracking(self) -> None:
        """Stop background tracking tasks gracefully."""
        try:
            if not self._tracking_active:
                return
            
            # Signal shutdown
            self._shutdown_event.set()
            
            # Stop accuracy tracker
            if self.accuracy_tracker:
                await self.accuracy_tracker.stop_monitoring()
            
            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            self._background_tasks.clear()
            self._tracking_active = False
            
            logger.info("Stopped TrackingManager monitoring tasks")
            
        except Exception as e:
            logger.error(f"Error stopping tracking: {e}")
    
    async def record_prediction(self, prediction_result: PredictionResult) -> None:
        """
        Record a prediction for tracking and future validation.
        
        This method is called automatically by ensemble models when
        they generate predictions.
        
        Args:
            prediction_result: The prediction result to record
        """
        try:
            if not self.config.enabled or not self.validator:
                return
            
            # Record prediction with validator
            await self.validator.record_prediction(
                room_id=prediction_result.prediction_metadata.get('room_id', 'unknown'),
                predicted_time=prediction_result.predicted_time,
                confidence=prediction_result.confidence_score,
                model_type=prediction_result.model_type,
                transition_type=prediction_result.transition_type,
                prediction_metadata=prediction_result.prediction_metadata
            )
            
            # Cache prediction for validation correlation
            room_id = prediction_result.prediction_metadata.get('room_id', 'unknown')
            with self._prediction_cache_lock:
                if room_id not in self._pending_predictions:
                    self._pending_predictions[room_id] = []
                self._pending_predictions[room_id].append(prediction_result)
                
                # Clean old predictions from cache
                cutoff_time = datetime.utcnow() - timedelta(hours=2)
                self._pending_predictions[room_id] = [
                    p for p in self._pending_predictions[room_id]
                    if p.predicted_time >= cutoff_time
                ]
            
            self._total_predictions_recorded += 1
            
            logger.debug(
                f"Recorded prediction for room {room_id}: "
                f"{prediction_result.predicted_time} ({prediction_result.transition_type})"
            )
            
        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
            # Don't raise exception to prevent disrupting prediction flow
    
    async def handle_room_state_change(
        self,
        room_id: str,
        new_state: str,
        change_time: datetime,
        previous_state: Optional[str] = None
    ) -> None:
        """
        Handle room state change for automatic validation.
        
        This method should be called by the event processing pipeline
        when actual room occupancy changes occur.
        
        Args:
            room_id: Room that changed state
            new_state: New occupancy state (occupied/vacant)
            change_time: When the state change occurred
            previous_state: Previous state if known
        """
        try:
            if not self.config.enabled or not self.config.auto_validation_enabled:
                return
            
            if not self.validator:
                logger.warning("No validator available for state change handling")
                return
            
            # Validate any pending predictions for this room
            await self.validator.validate_prediction(
                room_id=room_id,
                actual_time=change_time,
                transition_type=f"{previous_state or 'unknown'}_to_{new_state}"
            )
            
            self._total_validations_performed += 1
            
            logger.debug(
                f"Handled room state change for {room_id}: "
                f"{previous_state} -> {new_state} at {change_time}"
            )
            
        except Exception as e:
            logger.error(f"Failed to handle room state change: {e}")
            # Don't raise exception to prevent disrupting event processing
    
    async def get_tracking_status(self) -> Dict[str, Any]:
        """Get comprehensive tracking system status."""
        try:
            status = {
                'tracking_active': self._tracking_active,
                'config': {
                    'enabled': self.config.enabled,
                    'monitoring_interval_seconds': self.config.monitoring_interval_seconds,
                    'auto_validation_enabled': self.config.auto_validation_enabled,
                    'validation_window_minutes': self.config.validation_window_minutes
                },
                'performance': {
                    'total_predictions_recorded': self._total_predictions_recorded,
                    'total_validations_performed': self._total_validations_performed,
                    'system_uptime_seconds': (
                        datetime.utcnow() - self._system_start_time
                    ).total_seconds(),
                    'background_tasks': len(self._background_tasks)
                }
            }
            
            # Add validator status
            if self.validator:
                status['validator'] = {
                    'total_predictions': await self.validator.get_total_predictions(),
                    'validation_rate': await self.validator.get_validation_rate(),
                    'pending_validations': len(self.validator._pending_predictions)
                }
            
            # Add accuracy tracker status
            if self.accuracy_tracker:
                status['accuracy_tracker'] = self.accuracy_tracker.get_tracker_stats()
            
            # Add pending predictions cache status
            with self._prediction_cache_lock:
                cache_status = {}
                for room_id, predictions in self._pending_predictions.items():
                    cache_status[room_id] = len(predictions)
                status['prediction_cache'] = cache_status
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get tracking status: {e}")
            return {'error': str(e)}
    
    async def get_real_time_metrics(
        self,
        room_id: Optional[str] = None,
        model_type: Optional[str] = None
    ) -> Union[RealTimeMetrics, Dict[str, RealTimeMetrics], None]:
        """Get real-time accuracy metrics."""
        try:
            if not self.accuracy_tracker:
                return None
            
            return await self.accuracy_tracker.get_real_time_metrics(room_id, model_type)
            
        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return None
    
    async def get_active_alerts(
        self,
        room_id: Optional[str] = None,
        severity: Optional[str] = None
    ) -> List[AccuracyAlert]:
        """Get active accuracy alerts."""
        try:
            if not self.accuracy_tracker:
                return []
            
            from .tracker import AlertSeverity
            severity_enum = None
            if severity:
                severity_enum = AlertSeverity(severity)
            
            return await self.accuracy_tracker.get_active_alerts(room_id, severity_enum)
            
        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            return []
    
    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """Acknowledge an accuracy alert."""
        try:
            if not self.accuracy_tracker:
                return False
            
            return await self.accuracy_tracker.acknowledge_alert(alert_id, acknowledged_by)
            
        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False
    
    def add_notification_callback(self, callback: Callable[[AccuracyAlert], None]) -> None:
        """Add a notification callback for alerts."""
        if callback not in self.notification_callbacks:
            self.notification_callbacks.append(callback)
            
            if self.accuracy_tracker:
                self.accuracy_tracker.add_notification_callback(callback)
            
            logger.info("Added notification callback to TrackingManager")
    
    def remove_notification_callback(self, callback: Callable[[AccuracyAlert], None]) -> None:
        """Remove a notification callback."""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
            
            if self.accuracy_tracker:
                self.accuracy_tracker.remove_notification_callback(callback)
            
            logger.info("Removed notification callback from TrackingManager")
    
    # Private methods
    
    async def _validation_monitoring_loop(self) -> None:
        """Background loop for validation monitoring."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    if self.config.auto_validation_enabled and self.database_manager:
                        await self._check_for_room_state_changes()
                    
                    # Wait for next monitoring cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.validation_window_minutes * 60
                    )
                    
                except asyncio.TimeoutError:
                    # Expected timeout for monitoring interval
                    continue
                except Exception as e:
                    logger.error(f"Error in validation monitoring loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying
                    
        except asyncio.CancelledError:
            logger.info("Validation monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Validation monitoring loop failed: {e}")
    
    async def _check_for_room_state_changes(self) -> None:
        """Check database for recent room state changes to trigger validation."""
        try:
            if not self.database_manager:
                return
            
            # Get recent room state changes
            cutoff_time = datetime.utcnow() - timedelta(
                minutes=self.config.validation_window_minutes
            )
            
            # This would require a method in the database manager to get recent state changes
            # For now, we'll skip this implementation as it requires database integration
            # In practice, this would be called by the event processing pipeline
            
            logger.debug("Checked for room state changes")
            
        except Exception as e:
            logger.error(f"Failed to check for room state changes: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background loop for periodic cleanup."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._perform_cleanup()
                    
                    # Wait for next cleanup cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.cleanup_interval_hours * 3600
                    )
                    
                except asyncio.TimeoutError:
                    # Expected timeout for cleanup interval
                    continue
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
                    
        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Cleanup loop failed: {e}")
    
    async def _perform_cleanup(self) -> None:
        """Perform periodic cleanup of tracking data."""
        try:
            # Clean prediction cache
            cutoff_time = datetime.utcnow() - timedelta(hours=2)
            
            with self._prediction_cache_lock:
                for room_id in list(self._pending_predictions.keys()):
                    self._pending_predictions[room_id] = [
                        p for p in self._pending_predictions[room_id]
                        if p.predicted_time >= cutoff_time
                    ]
                    
                    # Remove empty room entries
                    if not self._pending_predictions[room_id]:
                        del self._pending_predictions[room_id]
            
            # Clean validator data if available
            if self.validator:
                await self.validator.cleanup_old_predictions(days_to_keep=7)
            
            logger.debug("Performed tracking data cleanup")
            
        except Exception as e:
            logger.error(f"Failed to perform cleanup: {e}")


class TrackingManagerError(OccupancyPredictionError):
    """Raised when tracking manager operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="TRACKING_MANAGER_ERROR",
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            **kwargs
        )