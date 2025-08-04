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
from .drift_detector import ConceptDriftDetector, DriftMetrics, DriftSeverity
from .retrainer import AdaptiveRetrainer, RetrainingRequest, RetrainingTrigger, RetrainingStatus
from .optimizer import ModelOptimizer, OptimizationConfig, OptimizationStrategy, OptimizationObjective
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
    
    # Drift detection configuration
    drift_detection_enabled: bool = True
    drift_check_interval_hours: int = 24
    drift_baseline_days: int = 30
    drift_current_days: int = 7
    drift_min_samples: int = 100
    drift_significance_threshold: float = 0.05
    drift_psi_threshold: float = 0.25
    drift_ph_threshold: float = 50.0
    
    # Adaptive retraining configuration
    adaptive_retraining_enabled: bool = True
    retraining_accuracy_threshold: float = 60.0  # Minimum accuracy % before retraining
    retraining_error_threshold: float = 25.0     # Maximum error minutes before retraining
    retraining_drift_threshold: float = 0.3      # Drift score threshold for retraining
    retraining_check_interval_hours: int = 6     # How often to check for retraining needs
    incremental_retraining_threshold: float = 70.0  # Use incremental if accuracy above this
    max_concurrent_retrains: int = 2             # Maximum models retraining simultaneously
    retraining_cooldown_hours: int = 12          # Minimum time between retrains for same model
    auto_feature_refresh: bool = True            # Automatically refresh features for retraining
    retraining_validation_split: float = 0.2    # Validation split for retraining
    retraining_lookback_days: int = 14           # Days of data to use for retraining
    
    # Model optimization configuration
    optimization_enabled: bool = True            # Enable automatic parameter optimization during retraining
    optimization_strategy: str = "bayesian"     # Optimization strategy: bayesian, grid_search, random_search
    optimization_max_time_minutes: int = 30     # Maximum time for optimization per model
    optimization_n_calls: int = 50              # Number of optimization iterations
    optimization_min_improvement: float = 0.01  # Minimum improvement to apply optimization results
    
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
                'validation_lag_critical': 30.0,
                'retraining_needed': 60.0,
                'retraining_urgent': 50.0,
                'drift_retraining': 0.3,
                'retraining_failure': 0.0
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
        model_registry: Optional[Dict[str, Any]] = None,
        feature_engineering_engine=None,
        notification_callbacks: Optional[List[Callable]] = None,
        mqtt_integration_manager=None
    ):
        """
        Initialize the tracking manager.
        
        Args:
            config: Tracking configuration
            database_manager: Database manager for accessing room states
            model_registry: Registry of available models for retraining
            feature_engineering_engine: Engine for feature extraction
            notification_callbacks: List of notification callbacks for alerts
            mqtt_integration_manager: MQTT integration manager for automatic publishing
        """
        self.config = config
        self.database_manager = database_manager
        self.model_registry = model_registry or {}
        self.feature_engineering_engine = feature_engineering_engine
        self.notification_callbacks = notification_callbacks or []
        
        # MQTT integration for automatic Home Assistant publishing
        self.mqtt_integration_manager = mqtt_integration_manager
        
        # Initialize core tracking components
        self.validator: Optional[PredictionValidator] = None
        self.accuracy_tracker: Optional[AccuracyTracker] = None
        self.drift_detector: Optional[ConceptDriftDetector] = None
        self.adaptive_retrainer: Optional[AdaptiveRetrainer] = None
        self.model_optimizer: Optional[ModelOptimizer] = None
        
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
        self._total_drift_checks_performed = 0
        self._last_drift_check_time: Optional[datetime] = None
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
            
            # Initialize drift detector if enabled
            if self.config.drift_detection_enabled:
                self.drift_detector = ConceptDriftDetector(
                    baseline_days=self.config.drift_baseline_days,
                    current_days=self.config.drift_current_days,
                    min_samples=self.config.drift_min_samples,
                    alpha=self.config.drift_significance_threshold,
                    ph_threshold=self.config.drift_ph_threshold,
                    psi_threshold=self.config.drift_psi_threshold
                )
            
            # Initialize model optimizer if enabled
            if self.config.optimization_enabled:
                optimization_config = OptimizationConfig(
                    enabled=self.config.optimization_enabled,
                    strategy=OptimizationStrategy(self.config.optimization_strategy),
                    max_optimization_time_minutes=self.config.optimization_max_time_minutes,
                    n_calls=self.config.optimization_n_calls,
                    min_improvement_threshold=self.config.optimization_min_improvement
                )
                
                self.model_optimizer = ModelOptimizer(
                    config=optimization_config,
                    accuracy_tracker=self.accuracy_tracker,
                    drift_detector=self.drift_detector
                )
                logger.info(f"ModelOptimizer initialized with strategy: {self.config.optimization_strategy}")
            
            # Initialize adaptive retrainer if enabled
            if self.config.adaptive_retraining_enabled:
                self.adaptive_retrainer = AdaptiveRetrainer(
                    tracking_config=self.config,
                    model_registry=self.model_registry,
                    feature_engineering_engine=self.feature_engineering_engine,
                    notification_callbacks=self.notification_callbacks,
                    model_optimizer=self.model_optimizer  # Pass optimizer to retrainer
                )
                await self.adaptive_retrainer.initialize()
            
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
            
            # Start drift detection task if enabled
            if self.config.drift_detection_enabled and self.drift_detector:
                drift_task = asyncio.create_task(self._drift_detection_loop())
                self._background_tasks.append(drift_task)
            
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
            
            # Stop adaptive retrainer
            if self.adaptive_retrainer:
                await self.adaptive_retrainer.shutdown()
            
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
        they generate predictions. Also automatically publishes to Home Assistant via MQTT.
        
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
            
            # Automatically publish prediction to Home Assistant via MQTT
            if self.mqtt_integration_manager:
                try:
                    await self.mqtt_integration_manager.publish_prediction(
                        prediction_result=prediction_result,
                        room_id=room_id,
                        current_state=None  # Could be determined from room state if available
                    )
                    logger.debug(f"Published prediction to Home Assistant for room {room_id}")
                except Exception as mqtt_error:
                    logger.warning(f"Failed to publish prediction to MQTT for room {room_id}: {mqtt_error}")
                    # Don't raise exception - MQTT publishing is optional
            
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
            
            # Check if retraining is needed based on recent accuracy
            if self.adaptive_retrainer and self.validator:
                await self._evaluate_accuracy_based_retraining(room_id)
            
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
                    'total_drift_checks_performed': self._total_drift_checks_performed,
                    'last_drift_check_time': self._last_drift_check_time.isoformat() if self._last_drift_check_time else None,
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
            
            # Add drift detector status
            if self.drift_detector:
                status['drift_detector'] = {
                    'enabled': self.config.drift_detection_enabled,
                    'check_interval_hours': self.config.drift_check_interval_hours,
                    'baseline_days': self.config.drift_baseline_days,
                    'current_days': self.config.drift_current_days,
                    'total_checks_performed': self._total_drift_checks_performed,
                    'last_check_time': self._last_drift_check_time.isoformat() if self._last_drift_check_time else None
                }
            
            # Add adaptive retrainer status
            if self.adaptive_retrainer:
                status['adaptive_retrainer'] = self.adaptive_retrainer.get_retrainer_stats()
            
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
    
    async def check_drift(self, room_id: str, feature_engineering_engine=None) -> Optional[DriftMetrics]:
        """
        Manually trigger drift detection for a specific room.
        
        Args:
            room_id: Room to check for drift
            feature_engineering_engine: Engine for feature extraction (optional)
            
        Returns:
            DriftMetrics if detection completed, None if disabled or failed
        """
        try:
            if not self.config.drift_detection_enabled or not self.drift_detector:
                logger.debug("Drift detection disabled or not initialized")
                return None
            
            if not self.validator:
                logger.warning("No validator available for drift detection")
                return None
            
            logger.info(f"Running manual drift detection for room {room_id}")
            
            drift_metrics = await self.drift_detector.detect_drift(
                room_id=room_id,
                prediction_validator=self.validator,
                feature_engineering_engine=feature_engineering_engine
            )
            
            self._total_drift_checks_performed += 1
            
            # Handle drift detection results and check for retraining needs
            await self._handle_drift_detection_results(room_id, drift_metrics)
            
            # Evaluate retraining need based on drift results
            if self.adaptive_retrainer and drift_metrics:
                await self._evaluate_drift_based_retraining(room_id, drift_metrics)
            
            logger.info(
                f"Drift detection completed for {room_id}: "
                f"severity={drift_metrics.drift_severity.value}, "
                f"score={drift_metrics.overall_drift_score:.3f}"
            )
            
            return drift_metrics
            
        except Exception as e:
            logger.error(f"Failed to check drift for {room_id}: {e}")
            return None
    
    async def get_drift_status(self, room_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get drift detection status and recent results.
        
        Args:
            room_id: Optional room filter
            
        Returns:
            Dictionary with drift detection status and metrics
        """
        try:
            status = {
                'drift_detection_enabled': self.config.drift_detection_enabled,
                'drift_detector_available': self.drift_detector is not None,
                'total_drift_checks': self._total_drift_checks_performed,
                'last_drift_check': self._last_drift_check_time.isoformat() if self._last_drift_check_time else None,
                'next_drift_check': None
            }
            
            if self._last_drift_check_time:
                next_check = self._last_drift_check_time + timedelta(
                    hours=self.config.drift_check_interval_hours
                )
                status['next_drift_check'] = next_check.isoformat()
            
            # Add configuration details
            if self.config.drift_detection_enabled:
                status['drift_config'] = {
                    'check_interval_hours': self.config.drift_check_interval_hours,
                    'baseline_days': self.config.drift_baseline_days,
                    'current_days': self.config.drift_current_days,
                    'min_samples': self.config.drift_min_samples,
                    'significance_threshold': self.config.drift_significance_threshold,
                    'psi_threshold': self.config.drift_psi_threshold,
                    'ph_threshold': self.config.drift_ph_threshold
                }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get drift status: {e}")
            return {'error': str(e)}
    
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
    
    async def _drift_detection_loop(self) -> None:
        """Background loop for automatic drift detection."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._perform_drift_detection()
                    
                    # Wait for next drift detection cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.drift_check_interval_hours * 3600
                    )
                    
                except asyncio.TimeoutError:
                    # Expected timeout for drift detection interval
                    continue
                except Exception as e:
                    logger.error(f"Error in drift detection loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying
                    
        except asyncio.CancelledError:
            logger.info("Drift detection loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Drift detection loop failed: {e}")
    
    async def _perform_drift_detection(self) -> None:
        """Perform automatic drift detection for all rooms."""
        try:
            if not self.config.drift_detection_enabled or not self.drift_detector:
                return
            
            if not self.validator:
                logger.warning("No validator available for drift detection")
                return
            
            logger.info("Starting automatic drift detection for all rooms")
            
            # Get all rooms from config (would need to be passed in or accessed globally)
            # For now, we'll get rooms from recent predictions
            rooms_to_check = await self._get_rooms_with_recent_activity()
            
            if not rooms_to_check:
                logger.debug("No rooms with recent activity for drift detection")
                return
            
            drift_results = {}
            
            for room_id in rooms_to_check:
                try:
                    logger.debug(f"Checking drift for room {room_id}")
                    
                    drift_metrics = await self.drift_detector.detect_drift(
                        room_id=room_id,
                        prediction_validator=self.validator,
                        feature_engineering_engine=None  # Would need to be injected
                    )
                    
                    drift_results[room_id] = drift_metrics
                    
                    # Handle drift detection results
                    await self._handle_drift_detection_results(room_id, drift_metrics)
                    
                    self._total_drift_checks_performed += 1
                    
                except Exception as e:
                    logger.error(f"Error checking drift for room {room_id}: {e}")
                    continue
            
            self._last_drift_check_time = datetime.utcnow()
            
            # Log summary of drift detection
            significant_drifts = [
                room_id for room_id, metrics in drift_results.items()
                if metrics.drift_severity in [DriftSeverity.MAJOR, DriftSeverity.CRITICAL]
            ]
            
            if significant_drifts:
                logger.warning(
                    f"Significant drift detected in rooms: {significant_drifts}"
                )
            else:
                logger.info(
                    f"Drift detection completed for {len(drift_results)} rooms - no significant drift"
                )
            
        except Exception as e:
            logger.error(f"Failed to perform drift detection: {e}")
    
    async def _get_rooms_with_recent_activity(self) -> List[str]:
        """Get rooms that have recent prediction activity for drift detection."""
        try:
            # Get rooms from recent predictions
            with self._prediction_cache_lock:
                rooms = list(self._pending_predictions.keys())
            
            # Also get rooms from validator if available
            if self.validator:
                validator_stats = await self.validator.get_validation_stats()
                if 'room_prediction_counts' in validator_stats:
                    rooms.extend(validator_stats['room_prediction_counts'].keys())
            
            # Remove duplicates and return
            return list(set(rooms))
            
        except Exception as e:
            logger.error(f"Error getting rooms with recent activity: {e}")
            return []
    
    async def _handle_drift_detection_results(
        self,
        room_id: str,
        drift_metrics: DriftMetrics
    ) -> None:
        """Handle drift detection results by generating alerts and notifications."""
        try:
            # Generate alert for significant drift
            if drift_metrics.drift_severity in [DriftSeverity.MAJOR, DriftSeverity.CRITICAL]:
                alert_severity = 'critical' if drift_metrics.drift_severity == DriftSeverity.CRITICAL else 'warning'
                
                # Create drift alert through accuracy tracker
                if self.accuracy_tracker:
                    # This would require extending AccuracyTracker to handle drift alerts
                    # For now, we'll log the significant drift
                    logger.warning(
                        f"Significant {drift_metrics.drift_severity.value} drift detected in {room_id}: "
                        f"score={drift_metrics.overall_drift_score:.3f}, "
                        f"retraining_recommended={drift_metrics.retraining_recommended}"
                    )
            
            # If immediate attention required, escalate notifications
            if drift_metrics.immediate_attention_required:
                logger.critical(
                    f"IMMEDIATE ATTENTION: Critical drift in {room_id} requires intervention. "
                    f"Accuracy degradation: {drift_metrics.accuracy_degradation:.1f} minutes, "
                    f"Page-Hinkley drift: {drift_metrics.ph_drift_detected}"
                )
                
                # Notify callbacks about critical drift
                for callback in self.notification_callbacks:
                    try:
                        # Create a mock alert for drift (would need proper DriftAlert class)
                        drift_alert_message = (
                            f"Critical concept drift detected in {room_id}. "
                            f"Severity: {drift_metrics.drift_severity.value}, "
                            f"Score: {drift_metrics.overall_drift_score:.3f}"
                        )
                        
                        if asyncio.iscoroutinefunction(callback):
                            await callback(drift_alert_message)
                        else:
                            callback(drift_alert_message)
                    except Exception as e:
                        logger.error(f"Error in drift notification callback: {e}")
            
            # Log drift detection results for monitoring
            logger.info(
                f"Drift detection for {room_id}: "
                f"severity={drift_metrics.drift_severity.value}, "
                f"score={drift_metrics.overall_drift_score:.3f}, "
                f"types={[dt.value for dt in drift_metrics.drift_types]}, "
                f"retraining_recommended={drift_metrics.retraining_recommended}"
            )
            
        except Exception as e:
            logger.error(f"Error handling drift detection results for {room_id}: {e}")
    
    async def _evaluate_accuracy_based_retraining(self, room_id: str) -> None:
        """Evaluate if retraining is needed based on accuracy metrics."""
        try:
            if not self.adaptive_retrainer or not self.validator:
                return
            
            # Get recent accuracy metrics for this room
            accuracy_metrics = await self.validator.get_room_accuracy(room_id, hours_back=24)
            
            if accuracy_metrics.total_predictions < 10:
                # Need more data before evaluating retraining
                return
            
            # Check all model types in the registry for this room
            models_to_evaluate = [
                model_key for model_key in self.model_registry.keys()
                if model_key.startswith(f"{room_id}_")
            ]
            
            for model_key in models_to_evaluate:
                model_type = model_key.replace(f"{room_id}_", "")
                
                # Evaluate retraining need
                retraining_request = await self.adaptive_retrainer.evaluate_retraining_need(
                    room_id=room_id,
                    model_type=model_type,
                    accuracy_metrics=accuracy_metrics
                )
                
                if retraining_request:
                    logger.info(
                        f"Accuracy-based retraining needed for {model_key}: "
                        f"accuracy={accuracy_metrics.accuracy_rate:.1f}%, "
                        f"error={accuracy_metrics.mean_error_minutes:.1f}min"
                    )
            
        except Exception as e:
            logger.error(f"Error evaluating accuracy-based retraining for {room_id}: {e}")
    
    async def _evaluate_drift_based_retraining(self, room_id: str, drift_metrics: DriftMetrics) -> None:
        """Evaluate if retraining is needed based on drift detection results."""
        try:
            if not self.adaptive_retrainer or not self.validator:
                return
            
            # Get recent accuracy metrics for context
            accuracy_metrics = await self.validator.get_room_accuracy(room_id, hours_back=24)
            
            # Check all model types in the registry for this room
            models_to_evaluate = [
                model_key for model_key in self.model_registry.keys()
                if model_key.startswith(f"{room_id}_")
            ]
            
            for model_key in models_to_evaluate:
                model_type = model_key.replace(f"{room_id}_", "")
                
                # Evaluate retraining need with drift context
                retraining_request = await self.adaptive_retrainer.evaluate_retraining_need(
                    room_id=room_id,
                    model_type=model_type,
                    accuracy_metrics=accuracy_metrics,
                    drift_metrics=drift_metrics
                )
                
                if retraining_request:
                    logger.info(
                        f"Drift-based retraining needed for {model_key}: "
                        f"drift_score={drift_metrics.overall_drift_score:.3f}, "
                        f"severity={drift_metrics.drift_severity.value}"
                    )
            
        except Exception as e:
            logger.error(f"Error evaluating drift-based retraining for {room_id}: {e}")
    
    async def request_manual_retraining(
        self,
        room_id: str,
        model_type: str,
        strategy: Optional[str] = None,
        priority: float = 5.0
    ) -> Optional[str]:
        """
        Request manual retraining for a specific model.
        
        Args:
            room_id: Room to retrain model for
            model_type: Type of model to retrain
            strategy: Retraining strategy (optional)
            priority: Request priority (0-10, higher = more urgent)
            
        Returns:
            Request ID if successful, None otherwise
        """
        try:
            if not self.adaptive_retrainer:
                logger.error("Adaptive retrainer not available for manual request")
                return None
            
            # Convert strategy string to enum if provided
            retraining_strategy = None
            if strategy:
                from .retrainer import RetrainingStrategy
                try:
                    retraining_strategy = RetrainingStrategy(strategy.lower())
                except ValueError:
                    logger.warning(f"Unknown retraining strategy: {strategy}, using default")
            
            # Request retraining
            request_id = await self.adaptive_retrainer.request_retraining(
                room_id=room_id,
                model_type=model_type,
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                strategy=retraining_strategy,
                priority=priority
            )
            
            logger.info(f"Manual retraining requested: {request_id}")
            return request_id
            
        except Exception as e:
            logger.error(f"Failed to request manual retraining: {e}")
            return None
    
    async def get_retraining_status(self, request_id: Optional[str] = None) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
        """Get status of retraining operations."""
        try:
            if not self.adaptive_retrainer:
                return None
            
            return await self.adaptive_retrainer.get_retraining_status(request_id)
            
        except Exception as e:
            logger.error(f"Failed to get retraining status: {e}")
            return None
    
    async def cancel_retraining(self, request_id: str) -> bool:
        """Cancel a retraining request."""
        try:
            if not self.adaptive_retrainer:
                return False
            
            return await self.adaptive_retrainer.cancel_retraining(request_id)
            
        except Exception as e:
            logger.error(f"Failed to cancel retraining: {e}")
            return False
    
    def register_model(self, room_id: str, model_type: str, model_instance) -> None:
        """
        Register a model instance for adaptive retraining.
        
        Args:
            room_id: Room the model is for
            model_type: Type of model
            model_instance: The actual model instance
        """
        try:
            model_key = f"{room_id}_{model_type}"
            self.model_registry[model_key] = model_instance
            
            logger.info(f"Registered model for adaptive retraining: {model_key}")
            
        except Exception as e:
            logger.error(f"Failed to register model {room_id}_{model_type}: {e}")
    
    def unregister_model(self, room_id: str, model_type: str) -> None:
        """
        Unregister a model from adaptive retraining.
        
        Args:
            room_id: Room the model is for
            model_type: Type of model
        """
        try:
            model_key = f"{room_id}_{model_type}"
            if model_key in self.model_registry:
                del self.model_registry[model_key]
                logger.info(f"Unregistered model from adaptive retraining: {model_key}")
            
        except Exception as e:
            logger.error(f"Failed to unregister model {room_id}_{model_type}: {e}")


class TrackingManagerError(OccupancyPredictionError):
    """Raised when tracking manager operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="TRACKING_MANAGER_ERROR",
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            **kwargs
        )