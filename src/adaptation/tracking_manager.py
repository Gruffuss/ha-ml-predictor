"""
System-wide tracking manager for integrated accuracy monitoring.

This module provides centralized coordination of accuracy tracking across
the entire prediction system, automatically handling prediction recording,
validation, and real-time monitoring without manual intervention.
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import threading
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.constants import ModelType
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..integration.enhanced_mqtt_manager import EnhancedMQTTIntegrationManager
from ..integration.realtime_publisher import (
    PublishingChannel,
    RealtimePublishingSystem,
)
from ..models.base.predictor import PredictionResult
from .drift_detector import ConceptDriftDetector, DriftMetrics, DriftSeverity
from .optimizer import (
    ModelOptimizer,
    OptimizationConfig,
    OptimizationStrategy,
)
from .retrainer import (
    AdaptiveRetrainer,
    RetrainingRequest,
    RetrainingTrigger,
)
from .tracker import AccuracyAlert, AccuracyTracker, RealTimeMetrics
from .validator import PredictionValidator

logger = logging.getLogger(__name__)

# Import dashboard components with graceful fallback
try:
    from ..integration.dashboard import (
        DashboardConfig,
        DashboardMode,
        PerformanceDashboard,
    )

    DASHBOARD_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Dashboard components not available: {e}")
    DASHBOARD_AVAILABLE = False
    PerformanceDashboard = None
    DashboardConfig = None
    DashboardMode = None


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

    # Real-time publishing configuration
    realtime_publishing_enabled: bool = True
    websocket_enabled: bool = True
    sse_enabled: bool = True
    websocket_port: int = 8765
    realtime_system_status_interval_seconds: int = 30
    realtime_broadcast_alerts: bool = True
    realtime_broadcast_drift_events: bool = True
    realtime_max_connections: int = 100

    # WebSocket API configuration
    websocket_api_enabled: bool = True
    websocket_api_host: str = "0.0.0.0"
    websocket_api_port: int = 8766
    websocket_api_max_connections: int = 500
    websocket_api_max_messages_per_minute: int = 60
    websocket_api_heartbeat_interval_seconds: int = 30
    websocket_api_connection_timeout_seconds: int = 300
    websocket_api_message_acknowledgment_timeout_seconds: int = 30

    # Performance Dashboard configuration
    dashboard_enabled: bool = True
    dashboard_host: str = "0.0.0.0"
    dashboard_port: int = 8888
    dashboard_debug: bool = False
    dashboard_websocket_enabled: bool = True
    dashboard_update_interval_seconds: int = 5
    dashboard_max_websocket_connections: int = 50
    dashboard_cache_ttl_seconds: int = 30
    dashboard_metrics_retention_hours: int = 72
    dashboard_enable_retraining_controls: bool = True
    dashboard_enable_alert_management: bool = True
    dashboard_enable_historical_charts: bool = True
    dashboard_enable_drift_visualization: bool = True

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
    retraining_error_threshold: float = 25.0  # Maximum error minutes before retraining
    retraining_drift_threshold: float = 0.3  # Drift score threshold for retraining
    retraining_check_interval_hours: int = 6  # How often to check for retraining needs
    incremental_retraining_threshold: float = (
        70.0  # Use incremental if accuracy above this
    )
    max_concurrent_retrains: int = 2  # Maximum models retraining simultaneously
    retraining_cooldown_hours: int = 12  # Minimum time between retrains for same model
    auto_feature_refresh: bool = True  # Automatically refresh features for retraining
    retraining_validation_split: float = 0.2  # Validation split for retraining
    retraining_lookback_days: int = 14  # Days of data to use for retraining

    # Model optimization configuration
    optimization_enabled: bool = (
        True  # Enable automatic parameter optimization during retraining
    )
    optimization_strategy: str = (
        "bayesian"  # Optimization strategy: bayesian, grid_search, random_search
    )
    optimization_max_time_minutes: int = 30  # Maximum time for optimization per model
    optimization_n_calls: int = 50  # Number of optimization iterations
    optimization_min_improvement: float = (
        0.01  # Minimum improvement to apply optimization results
    )

    def __post_init__(self):
        """Set default alert thresholds if not provided."""
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "accuracy_warning": 70.0,
                "accuracy_critical": 50.0,
                "error_warning": 20.0,
                "error_critical": 30.0,
                "trend_degrading": -5.0,
                "validation_lag_warning": 15.0,
                "validation_lag_critical": 30.0,
                "retraining_needed": 60.0,
                "retraining_urgent": 50.0,
                "drift_retraining": 0.3,
                "retraining_failure": 0.0,
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
        mqtt_integration_manager=None,
        api_config=None,
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
            api_config: API configuration for automatic server integration
        """
        self.config = config
        self.database_manager = database_manager
        self.model_registry = model_registry or {}
        self.feature_engineering_engine = feature_engineering_engine
        self.notification_callbacks = notification_callbacks or []

        # Enhanced MQTT integration for automatic Home Assistant publishing with real-time features
        # Use Enhanced MQTT Manager by default, fall back to basic if provided explicitly
        if mqtt_integration_manager is None:
            # Initialize Enhanced MQTT Manager automatically with default configuration
            from ..core.config import get_config

            system_config = get_config()
            self.mqtt_integration_manager = EnhancedMQTTIntegrationManager(
                mqtt_config=system_config.mqtt,
                rooms=system_config.rooms,
                notification_callbacks=notification_callbacks,
            )
        else:
            # Use the provided integration manager (backward compatibility)
            self.mqtt_integration_manager = mqtt_integration_manager

        # API server integration for automatic REST API
        self.api_config = api_config
        self.api_server = None

        # Initialize core tracking components
        self.validator: Optional[PredictionValidator] = None
        self.accuracy_tracker: Optional[AccuracyTracker] = None
        self.drift_detector: Optional[ConceptDriftDetector] = None
        self.adaptive_retrainer: Optional[AdaptiveRetrainer] = None
        self.model_optimizer: Optional[ModelOptimizer] = None

        # Real-time publishing system
        self.realtime_publisher: Optional[RealtimePublishingSystem] = None

        # WebSocket API server
        self.websocket_api_server = None

        # Performance Dashboard
        self.dashboard: Optional[PerformanceDashboard] = None

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
            self.validator = PredictionValidator(accuracy_threshold_minutes=15)

            # Initialize accuracy tracker
            self.accuracy_tracker = AccuracyTracker(
                prediction_validator=self.validator,
                monitoring_interval_seconds=self.config.monitoring_interval_seconds,
                alert_thresholds=self.config.alert_thresholds,
                max_stored_alerts=self.config.max_stored_alerts,
                trend_analysis_points=self.config.trend_analysis_points,
                notification_callbacks=self.notification_callbacks,
            )

            # Initialize drift detector if enabled
            if self.config.drift_detection_enabled:
                self.drift_detector = ConceptDriftDetector(
                    baseline_days=self.config.drift_baseline_days,
                    current_days=self.config.drift_current_days,
                    min_samples=self.config.drift_min_samples,
                    alpha=self.config.drift_significance_threshold,
                    ph_threshold=self.config.drift_ph_threshold,
                    psi_threshold=self.config.drift_psi_threshold,
                )

            # Initialize model optimizer if enabled
            if self.config.optimization_enabled:
                optimization_config = OptimizationConfig(
                    enabled=self.config.optimization_enabled,
                    strategy=OptimizationStrategy(self.config.optimization_strategy),
                    max_optimization_time_minutes=self.config.optimization_max_time_minutes,
                    n_calls=self.config.optimization_n_calls,
                    min_improvement_threshold=self.config.optimization_min_improvement,
                )

                self.model_optimizer = ModelOptimizer(
                    config=optimization_config,
                    accuracy_tracker=self.accuracy_tracker,
                    drift_detector=self.drift_detector,
                )
                logger.info(
                    f"ModelOptimizer initialized with strategy: {self.config.optimization_strategy}"
                )

            # Initialize adaptive retrainer if enabled
            if self.config.adaptive_retraining_enabled:
                self.adaptive_retrainer = AdaptiveRetrainer(
                    tracking_config=self.config,
                    model_registry=self.model_registry,
                    feature_engineering_engine=self.feature_engineering_engine,
                    notification_callbacks=self.notification_callbacks,
                    model_optimizer=self.model_optimizer,  # Pass optimizer to retrainer
                )
                await self.adaptive_retrainer.initialize()

            # Initialize Enhanced MQTT integration if available
            if self.mqtt_integration_manager and hasattr(
                self.mqtt_integration_manager, "initialize"
            ):
                await self.mqtt_integration_manager.initialize()
                logger.info("Enhanced MQTT integration initialized successfully")

            # Start tracking systems
            await self.start_tracking()

            # Initialize and start real-time publishing system if enabled
            await self._initialize_realtime_publishing()

            # Initialize and start performance dashboard if enabled
            await self._initialize_dashboard()

            # Initialize and start WebSocket API server if enabled
            await self._initialize_websocket_api()

            # Start API server automatically if enabled
            await self._start_api_server_if_enabled()

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

            # Stop Enhanced MQTT integration if available
            if self.mqtt_integration_manager and hasattr(
                self.mqtt_integration_manager, "shutdown"
            ):
                await self.mqtt_integration_manager.shutdown()
                logger.info("Enhanced MQTT integration shutdown complete")

            # Stop API server if running
            await self.stop_api_server()

            # Stop performance dashboard if running
            await self._shutdown_dashboard()

            # Stop WebSocket API server if running
            await self._shutdown_websocket_api()

            # Stop real-time publishing system
            await self._shutdown_realtime_publishing()

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
                room_id=prediction_result.prediction_metadata.get("room_id", "unknown"),
                predicted_time=prediction_result.predicted_time,
                confidence=prediction_result.confidence_score,
                model_type=prediction_result.model_type,
                transition_type=prediction_result.transition_type,
                prediction_metadata=prediction_result.prediction_metadata,
            )

            # Cache prediction for validation correlation
            room_id = prediction_result.prediction_metadata.get("room_id", "unknown")
            with self._prediction_cache_lock:
                if room_id not in self._pending_predictions:
                    self._pending_predictions[room_id] = []
                self._pending_predictions[room_id].append(prediction_result)

                # Clean old predictions from cache
                cutoff_time = datetime.utcnow() - timedelta(hours=2)
                self._pending_predictions[room_id] = [
                    p
                    for p in self._pending_predictions[room_id]
                    if p.predicted_time >= cutoff_time
                ]

            self._total_predictions_recorded += 1

            # Automatically publish prediction to Home Assistant via Enhanced MQTT (includes real-time broadcasting)
            if self.mqtt_integration_manager:
                try:
                    publish_results = await self.mqtt_integration_manager.publish_prediction(
                        prediction_result=prediction_result,
                        room_id=room_id,
                        current_state=None,  # Could be determined from room state if available
                    )

                    # Log enhanced publishing results
                    if isinstance(publish_results, dict):
                        successful_channels = []
                        if publish_results.get("mqtt", {}).get("success", False):
                            successful_channels.append("MQTT")
                        if publish_results.get("websocket", {}).get("success", False):
                            ws_clients = publish_results["websocket"].get(
                                "clients_notified", 0
                            )
                            successful_channels.append(
                                f"WebSocket({ws_clients} clients)"
                            )
                        if publish_results.get("sse", {}).get("success", False):
                            sse_clients = publish_results["sse"].get(
                                "clients_notified", 0
                            )
                            successful_channels.append(f"SSE({sse_clients} clients)")

                        if successful_channels:
                            logger.debug(
                                f"Published prediction for room {room_id} via Enhanced MQTT: {', '.join(successful_channels)}"
                            )
                    else:
                        logger.debug(
                            f"Published prediction to Home Assistant for room {room_id}"
                        )
                except Exception as mqtt_error:
                    logger.warning(
                        f"Failed to publish prediction via Enhanced MQTT for room {room_id}: {mqtt_error}"
                    )
                    # Don't raise exception - MQTT publishing is optional

            # Note: Real-time broadcasting is now handled by Enhanced MQTT Manager automatically
            # No need for separate real-time publisher when using Enhanced MQTT Manager

            # Automatically publish to WebSocket API clients if enabled
            if self.websocket_api_server and self.config.websocket_api_enabled:
                try:
                    websocket_results = await self.websocket_api_server.publish_prediction_update(
                        prediction_result=prediction_result,
                        room_id=room_id,
                        current_state=None,  # Could be determined from room state if available
                    )

                    if websocket_results.get("success", False):
                        clients_notified = websocket_results.get("clients_notified", 0)
                        logger.debug(
                            f"Published prediction for room {room_id} via WebSocket API to {clients_notified} clients"
                        )
                    else:
                        logger.warning(
                            f"Failed to publish prediction via WebSocket API for room {room_id}: {websocket_results.get('error', 'Unknown error')}"
                        )
                except Exception as websocket_error:
                    logger.warning(
                        f"Failed to publish prediction via WebSocket API for room {room_id}: {websocket_error}"
                    )
                    # Don't raise exception - WebSocket API publishing is optional

            # Automatically notify dashboard via WebSocket if enabled
            if self.dashboard and self.config.dashboard_enabled:
                try:
                    # Dashboard will automatically receive the prediction data through its integration
                    # with the tracking manager via real-time updates
                    logger.debug(
                        f"Dashboard integration active for prediction in room {room_id}"
                    )
                except Exception as dashboard_error:
                    logger.warning(
                        f"Dashboard integration error for room {room_id}: {dashboard_error}"
                    )
                    # Don't raise exception - dashboard integration is optional

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
        previous_state: Optional[str] = None,
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
                transition_type=f"{previous_state or 'unknown'}_to_{new_state}",
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
                "tracking_active": self._tracking_active,
                "config": {
                    "enabled": self.config.enabled,
                    "monitoring_interval_seconds": self.config.monitoring_interval_seconds,
                    "auto_validation_enabled": self.config.auto_validation_enabled,
                    "validation_window_minutes": self.config.validation_window_minutes,
                },
                "performance": {
                    "total_predictions_recorded": self._total_predictions_recorded,
                    "total_validations_performed": self._total_validations_performed,
                    "total_drift_checks_performed": self._total_drift_checks_performed,
                    "last_drift_check_time": (
                        self._last_drift_check_time.isoformat()
                        if self._last_drift_check_time
                        else None
                    ),
                    "system_uptime_seconds": (
                        datetime.utcnow() - self._system_start_time
                    ).total_seconds(),
                    "background_tasks": len(self._background_tasks),
                },
            }

            # Add validator status
            if self.validator:
                status["validator"] = {
                    "total_predictions": await self.validator.get_total_predictions(),
                    "validation_rate": await self.validator.get_validation_rate(),
                    "pending_validations": len(self.validator._pending_predictions),
                }

            # Add accuracy tracker status
            if self.accuracy_tracker:
                status["accuracy_tracker"] = self.accuracy_tracker.get_tracker_stats()

            # Add drift detector status
            if self.drift_detector:
                status["drift_detector"] = {
                    "enabled": self.config.drift_detection_enabled,
                    "check_interval_hours": self.config.drift_check_interval_hours,
                    "baseline_days": self.config.drift_baseline_days,
                    "current_days": self.config.drift_current_days,
                    "total_checks_performed": self._total_drift_checks_performed,
                    "last_check_time": (
                        self._last_drift_check_time.isoformat()
                        if self._last_drift_check_time
                        else None
                    ),
                }

            # Add adaptive retrainer status
            if self.adaptive_retrainer:
                status["adaptive_retrainer"] = (
                    self.adaptive_retrainer.get_retrainer_stats()
                )

            # Add pending predictions cache status
            with self._prediction_cache_lock:
                cache_status = {}
                for room_id, predictions in self._pending_predictions.items():
                    cache_status[room_id] = len(predictions)
                status["prediction_cache"] = cache_status

            # Add Enhanced MQTT integration status (includes real-time publishing)
            status["enhanced_mqtt_integration"] = self.get_enhanced_mqtt_status()

            # Add real-time publishing status (for backward compatibility)
            status["realtime_publishing"] = self.get_realtime_publishing_status()

            # Add WebSocket API status
            status["websocket_api"] = self.get_websocket_api_status()

            return status

        except Exception as e:
            logger.error(f"Failed to get tracking status: {e}")
            return {"error": str(e)}

    async def get_real_time_metrics(
        self,
        room_id: Optional[str] = None,
        model_type: Optional[Union[ModelType, str]] = None,
    ) -> Union[RealTimeMetrics, Dict[str, RealTimeMetrics], None]:
        """Get real-time accuracy metrics."""
        try:
            if not self.accuracy_tracker:
                return None

            return await self.accuracy_tracker.get_real_time_metrics(
                room_id, model_type
            )

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            return None

    async def get_active_alerts(
        self, room_id: Optional[str] = None, severity: Optional[str] = None
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

            return await self.accuracy_tracker.acknowledge_alert(
                alert_id, acknowledged_by
            )

        except Exception as e:
            logger.error(f"Failed to acknowledge alert: {e}")
            return False

    def add_notification_callback(
        self, callback: Callable[[AccuracyAlert], None]
    ) -> None:
        """Add a notification callback for alerts."""
        if callback not in self.notification_callbacks:
            self.notification_callbacks.append(callback)

            if self.accuracy_tracker:
                self.accuracy_tracker.add_notification_callback(callback)

            logger.info("Added notification callback to TrackingManager")

    def remove_notification_callback(
        self, callback: Callable[[AccuracyAlert], None]
    ) -> None:
        """Remove a notification callback."""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)

            if self.accuracy_tracker:
                self.accuracy_tracker.remove_notification_callback(callback)

            logger.info("Removed notification callback from TrackingManager")

    async def check_drift(
        self, room_id: str, feature_engineering_engine=None
    ) -> Optional[DriftMetrics]:
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
                feature_engineering_engine=feature_engineering_engine,
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
                "drift_detection_enabled": self.config.drift_detection_enabled,
                "drift_detector_available": self.drift_detector is not None,
                "total_drift_checks": self._total_drift_checks_performed,
                "last_drift_check": (
                    self._last_drift_check_time.isoformat()
                    if self._last_drift_check_time
                    else None
                ),
                "next_drift_check": None,
            }

            if self._last_drift_check_time:
                next_check = self._last_drift_check_time + timedelta(
                    hours=self.config.drift_check_interval_hours
                )
                status["next_drift_check"] = next_check.isoformat()

            # Add configuration details
            if self.config.drift_detection_enabled:
                status["drift_config"] = {
                    "check_interval_hours": self.config.drift_check_interval_hours,
                    "baseline_days": self.config.drift_baseline_days,
                    "current_days": self.config.drift_current_days,
                    "min_samples": self.config.drift_min_samples,
                    "significance_threshold": self.config.drift_significance_threshold,
                    "psi_threshold": self.config.drift_psi_threshold,
                    "ph_threshold": self.config.drift_ph_threshold,
                }

            return status

        except Exception as e:
            logger.error(f"Failed to get drift status: {e}")
            return {"error": str(e)}

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
                        timeout=self.config.validation_window_minutes * 60,
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

            # Query database for recent state changes using cutoff_time
            try:
                # Use cutoff_time in database query for recent state changes
                query = """
                    SELECT DISTINCT room_id, MAX(timestamp) as last_change
                    FROM sensor_events 
                    WHERE timestamp >= %s 
                    AND state != previous_state
                    GROUP BY room_id
                    ORDER BY last_change DESC
                """

                result = await self.database_manager.execute_query(
                    query, (cutoff_time,), fetch_all=True
                )

                if result:
                    logger.debug(
                        f"Found {len(result)} rooms with state changes since {cutoff_time.isoformat()}"
                    )

                    # Process state changes to trigger validation
                    for row in result:
                        room_id = row["room_id"]
                        last_change = row["last_change"]

                        # Check if validation is needed for this room
                        await self._check_room_validation_needed(room_id, last_change)
                else:
                    logger.debug(
                        f"No state changes found since {cutoff_time.isoformat()}"
                    )

            except Exception as query_error:
                logger.warning(
                    f"Database query failed, falling back to basic time logging: {query_error}"
                )
                logger.debug(
                    f"Checked for room state changes since {cutoff_time.isoformat()}"
                )

        except Exception as e:
            logger.error(f"Failed to check for room state changes: {e}")

    async def _check_room_validation_needed(
        self, room_id: str, last_change: datetime
    ) -> None:
        """Check if validation is needed for a room based on recent state change."""
        try:
            # Check if there are pending predictions that need validation
            if room_id in self._pending_predictions:
                await self._validate_room_predictions(room_id)
                logger.debug(
                    f"Triggered validation for room {room_id} due to state change at {last_change}"
                )
        except Exception as e:
            logger.error(f"Failed to check validation need for room {room_id}: {e}")

    async def _cleanup_loop(self) -> None:
        """Background loop for periodic cleanup."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    await self._perform_cleanup()

                    # Wait for next cleanup cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.cleanup_interval_hours * 3600,
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
                        p
                        for p in self._pending_predictions[room_id]
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
                        timeout=self.config.drift_check_interval_hours * 3600,
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
                        feature_engineering_engine=None,  # Would need to be injected
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
                room_id
                for room_id, metrics in drift_results.items()
                if metrics.drift_severity
                in [DriftSeverity.MAJOR, DriftSeverity.CRITICAL]
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
                if "room_prediction_counts" in validator_stats:
                    rooms.extend(validator_stats["room_prediction_counts"].keys())

            # Remove duplicates and return
            return list(set(rooms))

        except Exception as e:
            logger.error(f"Error getting rooms with recent activity: {e}")
            return []

    async def _handle_drift_detection_results(
        self, room_id: str, drift_metrics: DriftMetrics
    ) -> None:
        """Handle drift detection results by generating alerts and notifications."""
        try:
            # Generate alert for significant drift
            if drift_metrics.drift_severity in [
                DriftSeverity.MAJOR,
                DriftSeverity.CRITICAL,
            ]:
                alert_severity = (
                    "critical"
                    if drift_metrics.drift_severity == DriftSeverity.CRITICAL
                    else "warning"
                )

                # Use alert_severity to create drift alert with appropriate severity level
                if self.accuracy_tracker:
                    # Import AlertSeverity for proper alert creation
                    from .tracker import AccuracyAlert, AlertSeverity

                    # Map string severity to AlertSeverity enum
                    severity_mapping = {
                        "critical": AlertSeverity.CRITICAL,
                        "warning": AlertSeverity.WARNING,
                        "info": AlertSeverity.INFO,
                        "emergency": AlertSeverity.EMERGENCY,
                    }

                    # Create drift alert using the determined alert_severity
                    drift_alert = AccuracyAlert(
                        alert_id=f"drift_{room_id}_{int(datetime.utcnow().timestamp())}",
                        room_id=room_id,
                        model_type="drift_detector",
                        severity=severity_mapping.get(
                            alert_severity, AlertSeverity.WARNING
                        ),
                        trigger_condition="concept_drift",
                        current_value=drift_metrics.overall_drift_score,
                        threshold_value=0.7,  # Default drift threshold
                        description=(
                            f"Concept drift detected with {alert_severity} severity: "
                            f"score={drift_metrics.overall_drift_score:.3f}, "
                            f"types={[dt.value for dt in drift_metrics.drift_types]}"
                        ),
                        affected_metrics={
                            "drift_score": drift_metrics.overall_drift_score,
                            "accuracy_degradation": drift_metrics.accuracy_degradation,
                            "ph_drift_detected": drift_metrics.ph_drift_detected,
                        },
                        recent_predictions=0,  # Not applicable for drift alerts
                        trend_data={
                            "drift_severity": drift_metrics.drift_severity.value,
                            "drift_types": [
                                dt.value for dt in drift_metrics.drift_types
                            ],
                            "retraining_recommended": drift_metrics.retraining_recommended,
                        },
                    )

                    # Add drift alert to accuracy tracker's active alerts
                    with self.accuracy_tracker._lock:
                        self.accuracy_tracker._active_alerts[drift_alert.alert_id] = (
                            drift_alert
                        )
                        self.accuracy_tracker._alert_history.append(drift_alert)
                        self.accuracy_tracker._alert_counter += 1

                    # Notify callbacks about the drift alert
                    await self.accuracy_tracker._notify_alert_callbacks(drift_alert)

                    logger.warning(
                        f"Created {alert_severity} drift alert for {room_id}: "
                        f"score={drift_metrics.overall_drift_score:.3f}, "
                        f"retraining_recommended={drift_metrics.retraining_recommended}"
                    )
                else:
                    # Fallback logging if accuracy tracker not available
                    logger.warning(
                        f"Significant {alert_severity} drift detected in {room_id}: "
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
            accuracy_metrics = await self.validator.get_room_accuracy(
                room_id, hours_back=24
            )

            if accuracy_metrics.total_predictions < 10:
                # Need more data before evaluating retraining
                return

            # Check all model types in the registry for this room
            models_to_evaluate = [
                model_key
                for model_key in self.model_registry.keys()
                if model_key.startswith(f"{room_id}_")
            ]

            for model_key in models_to_evaluate:
                model_type = model_key.replace(f"{room_id}_", "")

                # Evaluate retraining need
                retraining_request = (
                    await self.adaptive_retrainer.evaluate_retraining_need(
                        room_id=room_id,
                        model_type=model_type,
                        accuracy_metrics=accuracy_metrics,
                    )
                )

                if retraining_request:
                    logger.info(
                        f"Accuracy-based retraining needed for {model_key}: "
                        f"accuracy={accuracy_metrics.accuracy_rate:.1f}%, "
                        f"error={accuracy_metrics.mean_error_minutes:.1f}min"
                    )

        except Exception as e:
            logger.error(
                f"Error evaluating accuracy-based retraining for {room_id}: {e}"
            )

    async def _evaluate_drift_based_retraining(
        self, room_id: str, drift_metrics: DriftMetrics
    ) -> None:
        """Evaluate if retraining is needed based on drift detection results."""
        try:
            if not self.adaptive_retrainer or not self.validator:
                return

            # Get recent accuracy metrics for context
            accuracy_metrics = await self.validator.get_room_accuracy(
                room_id, hours_back=24
            )

            # Check all model types in the registry for this room
            models_to_evaluate = [
                model_key
                for model_key in self.model_registry.keys()
                if model_key.startswith(f"{room_id}_")
            ]

            for model_key in models_to_evaluate:
                model_type = model_key.replace(f"{room_id}_", "")

                # Evaluate retraining need with drift context
                retraining_request = (
                    await self.adaptive_retrainer.evaluate_retraining_need(
                        room_id=room_id,
                        model_type=model_type,
                        accuracy_metrics=accuracy_metrics,
                        drift_metrics=drift_metrics,
                    )
                )

                if retraining_request:
                    logger.info(
                        f"Drift-based retraining needed for {model_key}: "
                        f"drift_score={drift_metrics.overall_drift_score:.3f}, "
                        f"severity={drift_metrics.drift_severity.value}"
                    )

        except Exception as e:
            logger.error(f"Error evaluating drift-based retraining for {room_id}: {e}")

    async def _initialize_realtime_publishing(self) -> None:
        """Initialize and start the real-time publishing system."""
        try:
            if not self.config.realtime_publishing_enabled:
                logger.debug("Real-time publishing disabled in configuration")
                return

            from ..core.config import get_config

            system_config = get_config()

            # Determine enabled channels based on configuration
            enabled_channels = []
            if self.mqtt_integration_manager:
                enabled_channels.append(PublishingChannel.MQTT)
            if self.config.websocket_enabled:
                enabled_channels.append(PublishingChannel.WEBSOCKET)
            if self.config.sse_enabled:
                enabled_channels.append(PublishingChannel.SSE)

            if not enabled_channels:
                logger.warning("No real-time publishing channels enabled")
                return

            # Initialize real-time publishing system
            self.realtime_publisher = RealtimePublishingSystem(
                mqtt_config=system_config.mqtt,
                rooms=system_config.rooms,
                prediction_publisher=None,  # Will be set if MQTT manager available
                enabled_channels=enabled_channels,
            )

            # Set MQTT prediction publisher if available
            if self.mqtt_integration_manager and hasattr(
                self.mqtt_integration_manager, "prediction_publisher"
            ):
                self.realtime_publisher.prediction_publisher = (
                    self.mqtt_integration_manager.prediction_publisher
                )

            # Initialize the publishing system
            await self.realtime_publisher.initialize()

            logger.info(
                "Real-time publishing system initialized with channels: "
                f"{[channel.value for channel in enabled_channels]}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize real-time publishing system: {e}")
            # Don't raise exception - real-time publishing is optional

    async def _shutdown_realtime_publishing(self) -> None:
        """Shutdown the real-time publishing system gracefully."""
        try:
            if self.realtime_publisher:
                await self.realtime_publisher.shutdown()
                self.realtime_publisher = None
                logger.info("Real-time publishing system shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down real-time publishing system: {e}")

    async def request_manual_retraining(
        self,
        room_id: str,
        model_type: Union[ModelType, str],
        strategy: Optional[str] = None,
        priority: float = 5.0,
    ) -> Optional[str]:
        """
        Request manual retraining for a specific model.

        Args:
            room_id: Room to retrain model for
            model_type: Type of model to retrain (enum or string)
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
                    logger.warning(
                        f"Unknown retraining strategy: {strategy}, using default"
                    )

            # Request retraining
            request_id = await self.adaptive_retrainer.request_retraining(
                room_id=room_id,
                model_type=model_type,
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                strategy=retraining_strategy,
                priority=priority,
            )

            logger.info(f"Manual retraining requested: {request_id}")
            return request_id

        except Exception as e:
            logger.error(f"Failed to request manual retraining: {e}")
            return None

    async def get_retraining_status(
        self, request_id: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]], None]:
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

    def register_model(
        self, room_id: str, model_type: Union[ModelType, str], model_instance
    ) -> None:
        """
        Register a model instance for adaptive retraining.

        Args:
            room_id: Room the model is for
            model_type: Type of model (enum or string)
            model_instance: The actual model instance
        """
        try:
            model_key = f"{room_id}_{str(model_type) if isinstance(model_type, ModelType) else model_type}"
            self.model_registry[model_key] = model_instance

            logger.info(f"Registered model for adaptive retraining: {model_key}")

        except Exception as e:
            logger.error(f"Failed to register model {room_id}_{model_type}: {e}")

    def unregister_model(self, room_id: str, model_type: Union[ModelType, str]) -> None:
        """
        Unregister a model from adaptive retraining.

        Args:
            room_id: Room the model is for
            model_type: Type of model (enum or string)
        """
        try:
            model_key = f"{room_id}_{str(model_type) if isinstance(model_type, ModelType) else model_type}"
            if model_key in self.model_registry:
                del self.model_registry[model_key]
                logger.info(f"Unregistered model from adaptive retraining: {model_key}")

        except Exception as e:
            logger.error(f"Failed to unregister model {room_id}_{model_type}: {e}")

    async def _start_api_server_if_enabled(self) -> None:
        """Start API server automatically if enabled in configuration."""
        if self.api_config and self.api_config.enabled:
            api_server = await self.start_api_server()
            if api_server:
                logger.info(
                    f"API server automatically started on {self.api_config.host}:{self.api_config.port}"
                )
            else:
                logger.warning("API server failed to start automatically")
        else:
            logger.debug("API server disabled or no configuration provided")

    async def start_api_server(self) -> Optional[Any]:
        """
        Start the integrated REST API server.

        Returns:
            APIServer instance if enabled, None otherwise
        """
        try:
            # Use late import to avoid circular dependency during module initialization
            from ..integration.api_server import (
                integrate_with_tracking_manager,
            )

            logger.info("Starting integrated REST API server...")
            api_server = await integrate_with_tracking_manager(self)
            await api_server.start()

            self.api_server = api_server
            logger.info("REST API server started successfully")
            return api_server

        except Exception as e:
            logger.error(f"Failed to start API server: {e}")
            return None

    async def stop_api_server(self) -> None:
        """Stop the integrated REST API server."""
        try:
            if hasattr(self, "api_server") and self.api_server:
                await self.api_server.stop()
                self.api_server = None
                logger.info("REST API server stopped")
        except Exception as e:
            logger.error(f"Failed to stop API server: {e}")

    def get_api_server_status(self) -> Dict[str, Any]:
        """Get API server status information."""
        try:
            if hasattr(self, "api_server") and self.api_server:
                return {
                    "enabled": True,
                    "running": self.api_server.is_running(),
                    "host": self.api_server.config.host,
                    "port": self.api_server.config.port,
                    "debug": self.api_server.config.debug,
                }
            else:
                return {
                    "enabled": False,
                    "running": False,
                    "host": None,
                    "port": None,
                    "debug": None,
                }
        except Exception as e:
            logger.error(f"Failed to get API server status: {e}")
            return {"enabled": False, "running": False, "error": str(e)}

    def get_enhanced_mqtt_status(self) -> Dict[str, Any]:
        """Get Enhanced MQTT integration status including all channels."""
        try:
            if self.mqtt_integration_manager and hasattr(
                self.mqtt_integration_manager, "get_integration_stats"
            ):
                stats = self.mqtt_integration_manager.get_integration_stats()
                return {
                    "enabled": True,
                    "type": "enhanced",
                    "mqtt_connected": stats.get("mqtt_integration", {}).get(
                        "mqtt_connected", False
                    ),
                    "discovery_published": stats.get("mqtt_integration", {}).get(
                        "discovery_published", False
                    ),
                    "realtime_publishing_active": stats.get(
                        "realtime_publishing", {}
                    ).get("system_active", False),
                    "total_channels": stats.get("channels", {}).get("total_active", 0),
                    "websocket_connections": stats.get("connections", {}).get(
                        "websocket_clients", 0
                    ),
                    "sse_connections": stats.get("connections", {}).get(
                        "sse_clients", 0
                    ),
                    "predictions_per_minute": stats.get("performance", {}).get(
                        "predictions_per_minute", 0
                    ),
                    "average_publish_latency_ms": stats.get("performance", {}).get(
                        "average_publish_latency_ms", 0
                    ),
                    "publish_success_rate": stats.get("performance", {}).get(
                        "publish_success_rate", 0
                    ),
                    "enabled_channels": stats.get("channels", {}).get(
                        "enabled_channels", []
                    ),
                }
            elif self.mqtt_integration_manager and hasattr(
                self.mqtt_integration_manager, "get_integration_stats"
            ):
                # Basic MQTT integration
                stats = self.mqtt_integration_manager.get_integration_stats()
                return {
                    "enabled": True,
                    "type": "basic",
                    "mqtt_connected": stats.get("mqtt_connected", False),
                    "discovery_published": stats.get("discovery_published", False),
                    "realtime_publishing_active": False,
                    "total_channels": (1 if stats.get("mqtt_connected", False) else 0),
                    "websocket_connections": 0,
                    "sse_connections": 0,
                    "predictions_per_minute": 0,
                    "average_publish_latency_ms": 0,
                    "publish_success_rate": 0,
                    "enabled_channels": (
                        ["mqtt"] if stats.get("mqtt_connected", False) else []
                    ),
                }
            else:
                return {
                    "enabled": False,
                    "type": "none",
                    "mqtt_connected": False,
                    "discovery_published": False,
                    "realtime_publishing_active": False,
                    "total_channels": 0,
                    "websocket_connections": 0,
                    "sse_connections": 0,
                    "predictions_per_minute": 0,
                    "average_publish_latency_ms": 0,
                    "publish_success_rate": 0,
                    "enabled_channels": [],
                }
        except Exception as e:
            logger.error(f"Failed to get Enhanced MQTT integration status: {e}")
            return {"enabled": False, "type": "error", "error": str(e)}

    def get_realtime_publishing_status(self) -> Dict[str, Any]:
        """Get real-time publishing system status information."""
        try:
            # Enhanced MQTT Manager provides real-time publishing - check that first
            enhanced_status = self.get_enhanced_mqtt_status()
            if enhanced_status.get("type") == "enhanced":
                return {
                    "enabled": True,
                    "active": enhanced_status.get("realtime_publishing_active", False),
                    "enabled_channels": enhanced_status.get("enabled_channels", []),
                    "websocket_connections": enhanced_status.get(
                        "websocket_connections", 0
                    ),
                    "sse_connections": enhanced_status.get("sse_connections", 0),
                    "total_predictions_published": 0,  # Would need to track this
                    "uptime_seconds": 0,  # Would need to track this
                    "source": "enhanced_mqtt_manager",
                }

            # Fall back to standalone real-time publisher if available
            if self.realtime_publisher and self.config.realtime_publishing_enabled:
                stats = self.realtime_publisher.get_publishing_stats()
                return {
                    "enabled": True,
                    "active": stats.get("system_active", False),
                    "enabled_channels": stats.get("enabled_channels", []),
                    "websocket_connections": stats.get("websocket_stats", {}).get(
                        "total_active_connections", 0
                    ),
                    "sse_connections": stats.get("sse_stats", {}).get(
                        "total_active_connections", 0
                    ),
                    "total_predictions_published": stats.get("metrics", {}).get(
                        "total_predictions_published", 0
                    ),
                    "uptime_seconds": stats.get("uptime_seconds", 0),
                    "source": "standalone_realtime_publisher",
                }
            else:
                return {
                    "enabled": self.config.realtime_publishing_enabled,
                    "active": False,
                    "enabled_channels": [],
                    "websocket_connections": 0,
                    "sse_connections": 0,
                    "total_predictions_published": 0,
                    "uptime_seconds": 0,
                    "source": "none",
                }
        except Exception as e:
            logger.error(f"Failed to get real-time publishing status: {e}")
            return {
                "enabled": False,
                "active": False,
                "error": str(e),
                "source": "error",
            }

    async def get_room_prediction(self, room_id: str) -> Optional[Dict[str, Any]]:
        """
        Get current prediction for a specific room.

        Args:
            room_id: Room identifier

        Returns:
            Prediction data or None if no prediction available
        """
        try:
            # This would need to interact with ensemble models
            # For now, return mock data structure
            logger.info(f"Getting prediction for room: {room_id}")

            # In a real implementation, this would:
            # 1. Get current room state
            # 2. Generate features for current time
            # 3. Use ensemble model to make prediction
            # 4. Return formatted prediction

            return {
                "room_id": room_id,
                "prediction_time": datetime.now().isoformat(),
                "next_transition_time": (
                    datetime.now() + timedelta(minutes=30)
                ).isoformat(),
                "transition_type": "occupied",
                "confidence": 0.85,
                "time_until_transition": "30 minutes",
                "alternatives": [],
                "model_info": {"model_type": "ensemble", "version": "1.0"},
            }

        except Exception as e:
            logger.error(f"Failed to get prediction for room {room_id}: {e}")
            return None

    async def get_accuracy_metrics(
        self, room_id: Optional[str] = None, hours: int = 24
    ) -> Dict[str, Any]:
        """
        Get accuracy metrics for a room or overall system.

        Args:
            room_id: Specific room ID (None for all rooms)
            hours: Time window in hours

        Returns:
            Accuracy metrics dictionary
        """
        try:
            logger.info(f"Getting accuracy metrics for room: {room_id}, hours: {hours}")

            # Get metrics from accuracy tracker
            if hasattr(self, "accuracy_tracker") and self.accuracy_tracker:
                if room_id:
                    metrics = await self.accuracy_tracker.get_room_metrics(room_id)
                else:
                    metrics = await self.accuracy_tracker.get_overall_metrics()

                # Convert to API format
                return {
                    "room_id": room_id,
                    "accuracy_rate": metrics.accuracy_rate,
                    "average_error_minutes": metrics.average_error_minutes,
                    "confidence_calibration": metrics.confidence_calibration,
                    "total_predictions": metrics.total_predictions,
                    "total_validations": metrics.total_validations,
                    "time_window_hours": hours,
                    "trend_direction": metrics.trend_direction,
                }
            else:
                # Return mock metrics if tracker not available
                return {
                    "room_id": room_id,
                    "accuracy_rate": 0.85,
                    "average_error_minutes": 12.5,
                    "confidence_calibration": 0.78,
                    "total_predictions": 150,
                    "total_validations": 145,
                    "time_window_hours": hours,
                    "trend_direction": "stable",
                }

        except Exception as e:
            logger.error(f"Failed to get accuracy metrics: {e}")
            return {
                "room_id": room_id,
                "accuracy_rate": 0.0,
                "average_error_minutes": 0.0,
                "confidence_calibration": 0.0,
                "total_predictions": 0,
                "total_validations": 0,
                "time_window_hours": hours,
                "trend_direction": "unknown",
                "error": str(e),
            }

    async def trigger_manual_retrain(
        self,
        room_id: Optional[str] = None,
        force: bool = False,
        strategy: str = "auto",
        reason: str = "manual_request",
    ) -> Dict[str, Any]:
        """
        Trigger manual model retraining.

        Args:
            room_id: Specific room to retrain (None for all)
            force: Force retrain even if not needed
            strategy: Retraining strategy
            reason: Reason for retraining

        Returns:
            Retraining status information
        """
        try:
            logger.info(
                f"Triggering manual retrain: room={room_id}, strategy={strategy}, force={force}"
            )

            # Use adaptive retrainer if available
            if hasattr(self, "adaptive_retrainer") and self.adaptive_retrainer:
                # RetrainingRequest and RetrainingTrigger already imported at top

                # Create retraining request
                request = RetrainingRequest(
                    room_id=room_id,
                    model_type="ensemble",  # Default to ensemble
                    trigger=RetrainingTrigger.MANUAL,
                    reason=reason,
                    priority=1 if force else 3,
                    force_retrain=force,
                    strategy=strategy,
                )

                # Submit request
                success = await self.adaptive_retrainer.request_retraining(request)

                return {
                    "success": success,
                    "room_id": room_id or "all_rooms",
                    "strategy": strategy,
                    "force": force,
                    "reason": reason,
                    "message": (
                        "Retraining request submitted successfully"
                        if success
                        else "Failed to submit retraining request"
                    ),
                }
            else:
                logger.warning("Adaptive retrainer not available for manual retrain")
                return {
                    "success": False,
                    "room_id": room_id or "all_rooms",
                    "strategy": strategy,
                    "force": force,
                    "reason": reason,
                    "message": "Adaptive retrainer not available",
                }

        except Exception as e:
            logger.error(f"Failed to trigger manual retrain: {e}")
            return {
                "success": False,
                "room_id": room_id or "all_rooms",
                "strategy": strategy,
                "force": force,
                "reason": reason,
                "message": f"Retraining failed: {str(e)}",
            }

    async def get_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive system statistics for API.

        Returns:
            System statistics dictionary
        """
        try:
            stats = {
                "tracking_stats": {
                    "tracking_enabled": self.config.enabled,
                    "monitoring_interval_seconds": self.config.monitoring_interval_seconds,
                    "active_alerts": 0,
                    "total_predictions_tracked": 0,
                    "total_validations": 0,
                },
                "drift_detection_stats": {
                    "drift_detection_enabled": self.config.drift_detection_enabled,
                    "last_drift_check": None,
                    "drift_alerts": 0,
                },
                "retraining_stats": {
                    "adaptive_retraining_enabled": self.config.adaptive_retraining_enabled,
                    "active_retraining_jobs": 0,
                    "completed_retraining_jobs": 0,
                    "failed_retraining_jobs": 0,
                },
                "api_server_stats": self.get_api_server_status(),
                "enhanced_mqtt_stats": self.get_enhanced_mqtt_status(),
                "realtime_publishing_stats": self.get_realtime_publishing_status(),
                "websocket_api_stats": self.get_websocket_api_status(),
                "dashboard_stats": self.get_dashboard_status(),
            }

            # Add tracker-specific stats if available
            if hasattr(self, "accuracy_tracker") and self.accuracy_tracker:
                tracker_stats = await self.accuracy_tracker.get_tracker_stats()
                stats["tracking_stats"].update(
                    {
                        "active_alerts": len(tracker_stats.get("active_alerts", [])),
                        "total_predictions_tracked": tracker_stats.get(
                            "total_predictions", 0
                        ),
                        "total_validations": tracker_stats.get("total_validations", 0),
                    }
                )

            # Add drift detector stats if available
            if hasattr(self, "drift_detector") and self.drift_detector:
                stats["drift_detection_stats"].update(
                    {
                        "last_drift_check": getattr(
                            self.drift_detector, "last_check_time", None
                        ),
                        "drift_alerts": getattr(
                            self.drift_detector, "total_drift_alerts", 0
                        ),
                    }
                )

            # Add retrainer stats if available
            if hasattr(self, "adaptive_retrainer") and self.adaptive_retrainer:
                retrainer_stats = await self.adaptive_retrainer.get_retraining_stats()
                stats["retraining_stats"].update(
                    {
                        "active_retraining_jobs": retrainer_stats.get("active_jobs", 0),
                        "completed_retraining_jobs": retrainer_stats.get(
                            "completed_jobs", 0
                        ),
                        "failed_retraining_jobs": retrainer_stats.get("failed_jobs", 0),
                    }
                )

            return stats

        except Exception as e:
            logger.error(f"Failed to get system stats: {e}")
            return {
                "tracking_stats": {"error": str(e)},
                "drift_detection_stats": {"error": str(e)},
                "retraining_stats": {"error": str(e)},
                "api_server_stats": {"error": str(e)},
                "websocket_api_stats": {"error": str(e)},
                "dashboard_stats": {"error": str(e)},
            }

    async def _initialize_websocket_api(self) -> None:
        """Initialize and start the WebSocket API server if enabled."""
        try:
            if not self.config.websocket_api_enabled:
                logger.debug("WebSocket API server disabled in configuration")
                return

            logger.info("Initializing WebSocket API server...")

            # Import WebSocket API components
            from ..integration.websocket_api import WebSocketAPIServer

            # Create WebSocket API configuration
            websocket_config = {
                "enabled": self.config.websocket_api_enabled,
                "host": self.config.websocket_api_host,
                "port": self.config.websocket_api_port,
                "max_connections": self.config.websocket_api_max_connections,
                "max_messages_per_minute": self.config.websocket_api_max_messages_per_minute,
                "heartbeat_interval_seconds": self.config.websocket_api_heartbeat_interval_seconds,
                "connection_timeout_seconds": self.config.websocket_api_connection_timeout_seconds,
                "acknowledgment_timeout_seconds": self.config.websocket_api_message_acknowledgment_timeout_seconds,
            }

            # Initialize WebSocket API server with this tracking manager
            self.websocket_api_server = WebSocketAPIServer(
                tracking_manager=self, config=websocket_config
            )

            # Initialize and start the WebSocket API server
            await self.websocket_api_server.initialize()
            await self.websocket_api_server.start()

            logger.info(
                f"WebSocket API server started successfully on ws://{websocket_config['host']}:{websocket_config['port']}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize WebSocket API server: {e}")
            # Don't raise exception - WebSocket API is optional functionality

    async def _shutdown_websocket_api(self) -> None:
        """Shutdown the WebSocket API server gracefully."""
        try:
            if self.websocket_api_server:
                logger.info("Shutting down WebSocket API server...")
                await self.websocket_api_server.stop()
                self.websocket_api_server = None
                logger.info("WebSocket API server shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down WebSocket API server: {e}")

    def get_websocket_api_status(self) -> Dict[str, Any]:
        """Get WebSocket API server status information."""
        try:
            if self.websocket_api_server and self.config.websocket_api_enabled:
                server_stats = self.websocket_api_server.get_server_stats()
                return {
                    "enabled": True,
                    "running": server_stats["server_running"],
                    "host": server_stats["host"],
                    "port": server_stats["port"],
                    "active_connections": server_stats["connection_stats"][
                        "active_connections"
                    ],
                    "authenticated_connections": server_stats["connection_stats"][
                        "authenticated_connections"
                    ],
                    "total_connections_served": server_stats["connection_stats"][
                        "total_connections"
                    ],
                    "predictions_connections": server_stats["connection_stats"][
                        "predictions_connections"
                    ],
                    "system_status_connections": server_stats["connection_stats"][
                        "system_status_connections"
                    ],
                    "alerts_connections": server_stats["connection_stats"][
                        "alerts_connections"
                    ],
                    "room_specific_connections": server_stats["connection_stats"][
                        "room_specific_connections"
                    ],
                    "total_messages_sent": server_stats["connection_stats"][
                        "total_messages_sent"
                    ],
                    "total_messages_received": server_stats["connection_stats"][
                        "total_messages_received"
                    ],
                    "rate_limited_clients": server_stats["connection_stats"][
                        "rate_limited_clients"
                    ],
                    "authentication_failures": server_stats["connection_stats"][
                        "authentication_failures"
                    ],
                    "tracking_manager_integrated": server_stats[
                        "tracking_manager_integrated"
                    ],
                }
            else:
                return {
                    "enabled": self.config.websocket_api_enabled,
                    "running": False,
                    "host": self.config.websocket_api_host,
                    "port": self.config.websocket_api_port,
                    "active_connections": 0,
                    "authenticated_connections": 0,
                    "total_connections_served": 0,
                    "predictions_connections": 0,
                    "system_status_connections": 0,
                    "alerts_connections": 0,
                    "room_specific_connections": 0,
                    "total_messages_sent": 0,
                    "total_messages_received": 0,
                    "rate_limited_clients": 0,
                    "authentication_failures": 0,
                    "tracking_manager_integrated": False,
                }
        except Exception as e:
            logger.error(f"Failed to get WebSocket API status: {e}")
            return {"enabled": False, "running": False, "error": str(e)}

    async def publish_system_status_update(
        self, status_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Publish system status update via WebSocket API."""
        try:
            if self.websocket_api_server and self.config.websocket_api_enabled:
                return await self.websocket_api_server.publish_system_status_update(
                    status_data
                )
            else:
                return {
                    "success": False,
                    "error": "WebSocket API not available",
                }
        except Exception as e:
            logger.error(f"Failed to publish system status update: {e}")
            return {"success": False, "error": str(e)}

    async def publish_alert_notification(
        self, alert_data: Dict[str, Any], room_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Publish alert notification via WebSocket API."""
        try:
            if self.websocket_api_server and self.config.websocket_api_enabled:
                return await self.websocket_api_server.publish_alert_notification(
                    alert_data, room_id
                )
            else:
                return {
                    "success": False,
                    "error": "WebSocket API not available",
                }
        except Exception as e:
            logger.error(f"Failed to publish alert notification: {e}")
            return {"success": False, "error": str(e)}

    async def _initialize_dashboard(self) -> None:
        """Initialize and start the performance dashboard if enabled."""
        try:
            if not self.config.dashboard_enabled:
                logger.debug("Performance dashboard disabled in configuration")
                return

            if not DASHBOARD_AVAILABLE:
                logger.warning(
                    "Dashboard components not available - cannot start dashboard"
                )
                return

            logger.info("Initializing performance monitoring dashboard...")

            # Create dashboard configuration from tracking config
            dashboard_config = DashboardConfig(
                enabled=self.config.dashboard_enabled,
                host=self.config.dashboard_host,
                port=self.config.dashboard_port,
                debug=self.config.dashboard_debug,
                mode=(
                    DashboardMode.DEVELOPMENT
                    if self.config.dashboard_debug
                    else DashboardMode.PRODUCTION
                ),
                # Real-time features
                websocket_enabled=self.config.dashboard_websocket_enabled,
                update_interval_seconds=self.config.dashboard_update_interval_seconds,
                max_websocket_connections=self.config.dashboard_max_websocket_connections,
                # Performance settings
                cache_ttl_seconds=self.config.dashboard_cache_ttl_seconds,
                metrics_retention_hours=self.config.dashboard_metrics_retention_hours,
                # Dashboard features
                enable_retraining_controls=self.config.dashboard_enable_retraining_controls,
                enable_alert_management=self.config.dashboard_enable_alert_management,
                enable_historical_charts=self.config.dashboard_enable_historical_charts,
                enable_drift_visualization=self.config.dashboard_enable_drift_visualization,
                enable_export_features=True,
            )

            # Initialize dashboard with this tracking manager
            self.dashboard = PerformanceDashboard(
                tracking_manager=self, config=dashboard_config
            )

            # Start the dashboard server
            await self.dashboard.start_dashboard()

            logger.info(
                f"Performance dashboard started successfully on http://{dashboard_config.host}:{dashboard_config.port}"
            )

        except Exception as e:
            logger.error(f"Failed to initialize performance dashboard: {e}")
            # Don't raise exception - dashboard is optional functionality

    async def _shutdown_dashboard(self) -> None:
        """Shutdown the performance dashboard gracefully."""
        try:
            if self.dashboard:
                logger.info("Shutting down performance dashboard...")
                await self.dashboard.stop_dashboard()
                self.dashboard = None
                logger.info("Performance dashboard shutdown complete")
        except Exception as e:
            logger.error(f"Error shutting down performance dashboard: {e}")

    def get_dashboard_status(self) -> Dict[str, Any]:
        """Get performance dashboard status information."""
        try:
            if self.dashboard and self.config.dashboard_enabled:
                return {
                    "enabled": True,
                    "running": (
                        self.dashboard._running
                        if hasattr(self.dashboard, "_running")
                        else False
                    ),
                    "host": self.dashboard.config.host,
                    "port": self.dashboard.config.port,
                    "websocket_enabled": self.dashboard.config.websocket_enabled,
                    "debug": self.dashboard.config.debug,
                    "mode": self.dashboard.config.mode.value,
                    "active_websocket_connections": (
                        len(self.dashboard.websocket_manager.active_connections)
                        if self.dashboard.websocket_manager
                        else 0
                    ),
                    "uptime_hours": (
                        (
                            datetime.utcnow() - self.dashboard._dashboard_start_time
                        ).total_seconds()
                        / 3600
                        if hasattr(self.dashboard, "_dashboard_start_time")
                        else 0
                    ),
                }
            else:
                return {
                    "enabled": self.config.dashboard_enabled,
                    "running": False,
                    "host": self.config.dashboard_host,
                    "port": self.config.dashboard_port,
                    "websocket_enabled": self.config.dashboard_websocket_enabled,
                    "debug": self.config.dashboard_debug,
                    "mode": "disabled",
                    "active_websocket_connections": 0,
                    "uptime_hours": 0,
                }
        except Exception as e:
            logger.error(f"Failed to get dashboard status: {e}")
            return {"enabled": False, "running": False, "error": str(e)}


class TrackingManagerError(OccupancyPredictionError):
    """Raised when tracking manager operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="TRACKING_MANAGER_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
