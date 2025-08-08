"""
Real-time Accuracy Tracking for Sprint 4 - Self-Adaptation System.

This module provides comprehensive real-time monitoring of prediction accuracy,
live metrics calculation, trend analysis, and automatic alerting capabilities
for the occupancy prediction system.
"""

import asyncio
import json
import logging
import statistics
import threading
from collections import defaultdict
from collections import deque
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ..core.constants import ModelType
from ..core.exceptions import ErrorSeverity
from ..core.exceptions import OccupancyPredictionError
from .validator import AccuracyLevel
from .validator import AccuracyMetrics
from .validator import PredictionValidator
from .validator import ValidationRecord

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Severity levels for accuracy alerts."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class TrendDirection(Enum):
    """Direction of accuracy trend analysis."""

    IMPROVING = "improving"
    STABLE = "stable"
    DEGRADING = "degrading"
    UNKNOWN = "unknown"


@dataclass
class RealTimeMetrics:
    """
    Real-time accuracy metrics with sliding window calculations.

    Provides live accuracy monitoring with configurable time windows,
    trend analysis, and performance indicators for immediate insights.
    """

    room_id: str
    model_type: Optional[str] = None

    # Time windows for metrics
    window_1h_accuracy: float = 0.0
    window_6h_accuracy: float = 0.0
    window_24h_accuracy: float = 0.0

    # Error statistics by window
    window_1h_mean_error: float = 0.0
    window_6h_mean_error: float = 0.0
    window_24h_mean_error: float = 0.0

    # Prediction counts by window
    window_1h_predictions: int = 0
    window_6h_predictions: int = 0
    window_24h_predictions: int = 0

    # Trend analysis
    accuracy_trend: TrendDirection = TrendDirection.UNKNOWN
    trend_slope: float = 0.0  # Positive = improving, negative = degrading
    trend_confidence: float = 0.0  # 0-1 confidence in trend detection

    # Real-time performance indicators
    recent_predictions_rate: float = 0.0  # Predictions per hour
    validation_lag_minutes: float = 0.0  # Average time to validate
    confidence_calibration: float = 0.0  # How well confidence matches accuracy

    # Alert status
    active_alerts: List[str] = field(default_factory=list)
    last_alert_time: Optional[datetime] = None

    # Metadata
    last_updated: datetime = field(default_factory=datetime.utcnow)
    measurement_start: Optional[datetime] = None

    @property
    def overall_health_score(self) -> float:
        """
        Calculate overall health score (0-100) based on multiple metrics.

        Combines accuracy, trend, validation rate, and confidence calibration
        into a single health indicator.
        """
        if self.window_24h_predictions == 0:
            return 0.0

        # Weight components
        accuracy_weight = 0.4
        trend_weight = 0.3
        calibration_weight = 0.2
        validation_weight = 0.1

        # Accuracy component (0-100)
        accuracy_score = min(100, self.window_24h_accuracy)

        # Trend component (0-100)
        if self.accuracy_trend == TrendDirection.IMPROVING:
            trend_score = 80 + (self.trend_confidence * 20)
        elif self.accuracy_trend == TrendDirection.STABLE:
            trend_score = 60 + (self.trend_confidence * 20)
        elif self.accuracy_trend == TrendDirection.DEGRADING:
            trend_score = 20 - (self.trend_confidence * 20)
        else:
            trend_score = 50  # Unknown

        # Calibration component (0-100)
        calibration_score = self.confidence_calibration * 100

        # Validation component (based on lag)
        if self.validation_lag_minutes < 5:
            validation_score = 100
        elif self.validation_lag_minutes < 15:
            validation_score = 80
        elif self.validation_lag_minutes < 30:
            validation_score = 60
        else:
            validation_score = 40

        return (
            accuracy_score * accuracy_weight
            + trend_score * trend_weight
            + calibration_score * calibration_weight
            + validation_score * validation_weight
        )

    @property
    def is_healthy(self) -> bool:
        """Check if metrics indicate healthy performance."""
        return (
            self.overall_health_score >= 70
            and self.window_6h_accuracy >= 60
            and self.accuracy_trend != TrendDirection.DEGRADING
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert real-time metrics to dictionary for API responses."""
        return {
            "room_id": self.room_id,
            "model_type": self.model_type,
            "time_windows": {
                "1h": {
                    "accuracy": self.window_1h_accuracy,
                    "mean_error_minutes": self.window_1h_mean_error,
                    "predictions": self.window_1h_predictions,
                },
                "6h": {
                    "accuracy": self.window_6h_accuracy,
                    "mean_error_minutes": self.window_6h_mean_error,
                    "predictions": self.window_6h_predictions,
                },
                "24h": {
                    "accuracy": self.window_24h_accuracy,
                    "mean_error_minutes": self.window_24h_mean_error,
                    "predictions": self.window_24h_predictions,
                },
            },
            "trend_analysis": {
                "direction": self.accuracy_trend.value,
                "slope": self.trend_slope,
                "confidence": self.trend_confidence,
            },
            "performance_indicators": {
                "overall_health_score": self.overall_health_score,
                "is_healthy": self.is_healthy,
                "predictions_per_hour": self.recent_predictions_rate,
                "validation_lag_minutes": self.validation_lag_minutes,
                "confidence_calibration": self.confidence_calibration,
            },
            "alerts": {
                "active_alerts": self.active_alerts,
                "last_alert_time": (
                    self.last_alert_time.isoformat() if self.last_alert_time else None
                ),
            },
            "metadata": {
                "last_updated": self.last_updated.isoformat(),
                "measurement_start": (
                    self.measurement_start.isoformat()
                    if self.measurement_start
                    else None
                ),
            },
        }


@dataclass
class AccuracyAlert:
    """
    Accuracy alert with context and notification capabilities.

    Represents an accuracy degradation alert with severity, context,
    and notification tracking for operational monitoring.
    """

    alert_id: str
    room_id: str
    model_type: Optional[str]
    severity: AlertSeverity

    # Alert details
    trigger_condition: str
    current_value: float
    threshold_value: float
    description: str

    # Context
    affected_metrics: Dict[str, float]
    recent_predictions: int
    trend_data: Dict[str, Any] = field(default_factory=dict)

    # Notification tracking
    triggered_time: datetime = field(default_factory=datetime.utcnow)
    acknowledged: bool = False
    acknowledged_time: Optional[datetime] = None
    acknowledged_by: Optional[str] = None
    resolved: bool = False
    resolved_time: Optional[datetime] = None

    # Escalation
    escalation_level: int = 1
    last_escalation: Optional[datetime] = None
    max_escalations: int = 3

    @property
    def age_minutes(self) -> float:
        """Calculate alert age in minutes."""
        return (datetime.utcnow() - self.triggered_time).total_seconds() / 60

    @property
    def requires_escalation(self) -> bool:
        """Check if alert requires escalation."""
        if self.escalation_level >= self.max_escalations:
            return False

        if self.acknowledged or self.resolved:
            return False

        # Escalate based on severity and age
        escalation_thresholds = {
            AlertSeverity.INFO: 120,  # 2 hours
            AlertSeverity.WARNING: 60,  # 1 hour
            AlertSeverity.CRITICAL: 30,  # 30 minutes
            AlertSeverity.EMERGENCY: 10,  # 10 minutes
        }

        threshold = escalation_thresholds.get(self.severity, 60)
        return self.age_minutes >= threshold

    def acknowledge(self, acknowledged_by: str) -> None:
        """Acknowledge the alert."""
        self.acknowledged = True
        self.acknowledged_time = datetime.utcnow()
        self.acknowledged_by = acknowledged_by

        logger.info(f"Alert {self.alert_id} acknowledged by {acknowledged_by}")

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.resolved = True
        self.resolved_time = datetime.utcnow()

        logger.info(
            f"Alert {self.alert_id} resolved after {self.age_minutes:.1f} minutes"
        )

    def escalate(self) -> bool:
        """Escalate the alert if conditions are met."""
        if not self.requires_escalation:
            return False

        self.escalation_level += 1
        self.last_escalation = datetime.utcnow()

        logger.warning(
            f"Escalating alert {self.alert_id} to level {self.escalation_level} "
            f"after {self.age_minutes:.1f} minutes"
        )

        return True

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary for serialization."""
        return {
            "alert_id": self.alert_id,
            "room_id": self.room_id,
            "model_type": self.model_type,
            "severity": self.severity.value,
            "trigger_condition": self.trigger_condition,
            "current_value": self.current_value,
            "threshold_value": self.threshold_value,
            "description": self.description,
            "affected_metrics": self.affected_metrics,
            "recent_predictions": self.recent_predictions,
            "trend_data": self.trend_data,
            "triggered_time": self.triggered_time.isoformat(),
            "age_minutes": self.age_minutes,
            "acknowledged": self.acknowledged,
            "acknowledged_time": (
                self.acknowledged_time.isoformat() if self.acknowledged_time else None
            ),
            "acknowledged_by": self.acknowledged_by,
            "resolved": self.resolved,
            "resolved_time": (
                self.resolved_time.isoformat() if self.resolved_time else None
            ),
            "escalation_level": self.escalation_level,
            "requires_escalation": self.requires_escalation,
            "last_escalation": (
                self.last_escalation.isoformat() if self.last_escalation else None
            ),
        }


class AccuracyTracker:
    """
    Production-ready real-time accuracy tracking system.

    Provides comprehensive real-time monitoring of prediction accuracy with:
    - Live metrics calculation across multiple time windows
    - Trend detection and analysis with statistical significance
    - Automatic alerting with configurable thresholds and escalation
    - Performance tracking per room and model type
    - Background monitoring tasks with memory efficiency
    - Integration with existing PredictionValidator
    """

    def __init__(
        self,
        prediction_validator: PredictionValidator,
        monitoring_interval_seconds: int = 60,
        alert_thresholds: Optional[Dict[str, float]] = None,
        max_stored_alerts: int = 1000,
        trend_analysis_points: int = 10,
        notification_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize real-time accuracy tracker.

        Args:
            prediction_validator: PredictionValidator instance for accessing validation data
            monitoring_interval_seconds: How often to update real-time metrics
            alert_thresholds: Configurable alert thresholds for accuracy degradation
            max_stored_alerts: Maximum number of alerts to keep in memory
            trend_analysis_points: Number of data points for trend analysis
            notification_callbacks: List of callback functions for alert notifications
        """
        self.validator = prediction_validator
        self.monitoring_interval = timedelta(seconds=monitoring_interval_seconds)
        self.max_stored_alerts = max_stored_alerts
        self.trend_points = trend_analysis_points

        # Alert thresholds with defaults
        self.alert_thresholds = alert_thresholds or {
            "accuracy_warning": 70.0,  # Warning if accuracy < 70%
            "accuracy_critical": 50.0,  # Critical if accuracy < 50%
            "error_warning": 20.0,  # Warning if mean error > 20 min
            "error_critical": 30.0,  # Critical if mean error > 30 min
            "trend_degrading": -5.0,  # Warning if trend slope < -5%/hour
            "validation_lag_warning": 15.0,  # Warning if validation lag > 15 min
            "validation_lag_critical": 30.0,  # Critical if validation lag > 30 min
        }

        # Notification callbacks
        self.notification_callbacks = notification_callbacks or []

        # Thread-safe storage
        self._lock = threading.RLock()
        self._metrics_by_room: Dict[str, RealTimeMetrics] = {}
        self._metrics_by_model: Dict[str, RealTimeMetrics] = {}
        self._global_metrics: Optional[RealTimeMetrics] = None

        # Alert management
        self._active_alerts: Dict[str, AccuracyAlert] = {}
        self._alert_history: deque = deque(maxlen=max_stored_alerts)
        self._alert_counter = 0

        # Trend analysis storage
        self._accuracy_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=trend_analysis_points)
        )

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._monitoring_active = False

        logger.info(
            f"Initialized AccuracyTracker with {monitoring_interval_seconds}s monitoring interval, "
            f"{len(self.alert_thresholds)} alert thresholds, "
            f"{len(self.notification_callbacks)} notification callbacks"
        )

    async def start_monitoring(self) -> None:
        """Start background monitoring tasks."""
        try:
            if self._monitoring_active:
                logger.warning("Monitoring already active")
                return

            # Start monitoring loop
            monitoring_task = asyncio.create_task(self._monitoring_loop())
            self._background_tasks.append(monitoring_task)

            # Start alert management loop
            alert_task = asyncio.create_task(self._alert_management_loop())
            self._background_tasks.append(alert_task)

            self._monitoring_active = True
            logger.info("Started AccuracyTracker monitoring tasks")

        except Exception as e:
            logger.error(f"Failed to start monitoring: {e}")
            raise AccuracyTrackingError("Failed to start real-time monitoring", cause=e)

    async def stop_monitoring(self) -> None:
        """Stop background monitoring tasks gracefully."""
        try:
            if not self._monitoring_active:
                return

            # Signal shutdown
            self._shutdown_event.set()

            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            self._background_tasks.clear()
            self._monitoring_active = False

            logger.info("Stopped AccuracyTracker monitoring tasks")

        except Exception as e:
            logger.error(f"Error stopping monitoring: {e}")

    async def get_real_time_metrics(
        self, room_id: Optional[str] = None, model_type: Optional[str] = None
    ) -> Union[RealTimeMetrics, Dict[str, RealTimeMetrics], None]:
        """
        Get current real-time metrics for room, model, or global.

        Args:
            room_id: Filter by specific room (None for all rooms)
            model_type: Filter by specific model (None for all models)

        Returns:
            Real-time metrics or dictionary of metrics
        """
        try:
            with self._lock:
                if room_id and model_type:
                    # Specific room and model combination
                    key = f"{room_id}_{model_type}"
                    return self._metrics_by_room.get(key) or self._metrics_by_model.get(
                        key
                    )

                elif room_id:
                    # All metrics for specific room
                    room_metrics = {}
                    for key, metrics in self._metrics_by_room.items():
                        if metrics.room_id == room_id:
                            room_metrics[key] = metrics
                    return room_metrics if room_metrics else None

                elif model_type:
                    # All metrics for specific model
                    model_metrics = {}
                    for key, metrics in self._metrics_by_model.items():
                        if metrics.model_type == model_type:
                            model_metrics[key] = metrics
                    return model_metrics if model_metrics else None

                else:
                    # Global metrics
                    return self._global_metrics

        except Exception as e:
            logger.error(f"Failed to get real-time metrics: {e}")
            raise AccuracyTrackingError("Failed to retrieve real-time metrics", cause=e)

    async def get_active_alerts(
        self, room_id: Optional[str] = None, severity: Optional[AlertSeverity] = None
    ) -> List[AccuracyAlert]:
        """
        Get active accuracy alerts with optional filtering.

        Args:
            room_id: Filter by specific room
            severity: Filter by alert severity

        Returns:
            List of active alerts matching criteria
        """
        try:
            with self._lock:
                alerts = []
                for alert in self._active_alerts.values():
                    if alert.resolved:
                        continue

                    if room_id and alert.room_id != room_id:
                        continue

                    if severity and alert.severity != severity:
                        continue

                    alerts.append(alert)

                # Sort by severity and age
                severity_order = {
                    AlertSeverity.EMERGENCY: 4,
                    AlertSeverity.CRITICAL: 3,
                    AlertSeverity.WARNING: 2,
                    AlertSeverity.INFO: 1,
                }

                alerts.sort(
                    key=lambda a: (severity_order.get(a.severity, 0), -a.age_minutes),
                    reverse=True,
                )

                return alerts

        except Exception as e:
            logger.error(f"Failed to get active alerts: {e}")
            raise AccuracyTrackingError("Failed to retrieve active alerts", cause=e)

    async def acknowledge_alert(self, alert_id: str, acknowledged_by: str) -> bool:
        """
        Acknowledge an active alert.

        Args:
            alert_id: ID of alert to acknowledge
            acknowledged_by: Who acknowledged the alert

        Returns:
            True if alert was acknowledged, False if not found
        """
        try:
            with self._lock:
                if alert_id in self._active_alerts:
                    alert = self._active_alerts[alert_id]
                    alert.acknowledge(acknowledged_by)
                    return True
                return False

        except Exception as e:
            logger.error(f"Failed to acknowledge alert {alert_id}: {e}")
            raise AccuracyTrackingError("Failed to acknowledge alert", cause=e)

    async def get_accuracy_trends(
        self, room_id: Optional[str] = None, hours_back: int = 24
    ) -> Dict[str, Any]:
        """
        Get accuracy trends and analysis.

        Args:
            room_id: Filter by specific room
            hours_back: How many hours of trend data to analyze

        Returns:
            Dictionary with trend analysis results
        """
        try:
            trends = {}

            with self._lock:
                if room_id:
                    # Room-specific trends
                    for key, history in self._accuracy_history.items():
                        if key.startswith(room_id):
                            trends[key] = self._analyze_trend(list(history))
                else:
                    # All trends
                    for key, history in self._accuracy_history.items():
                        trends[key] = self._analyze_trend(list(history))

            return {
                "trends_by_entity": trends,
                "global_trend": self._calculate_global_trend(trends),
                "analysis_period_hours": hours_back,
                "last_updated": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Failed to get accuracy trends: {e}")
            raise AccuracyTrackingError("Failed to get accuracy trends", cause=e)

    async def export_tracking_data(
        self,
        output_path: Union[str, Path],
        include_alerts: bool = True,
        include_trends: bool = True,
        days_back: int = 7,
    ) -> int:
        """
        Export tracking data for analysis.

        Args:
            output_path: Where to save the exported data
            include_alerts: Include alert history in export
            include_trends: Include trend analysis in export
            days_back: How many days of data to export

        Returns:
            Number of records exported
        """
        try:
            output_path = Path(output_path)
            export_data = {
                "export_time": datetime.utcnow().isoformat(),
                "export_period_days": days_back,
                "metrics": {},
                "alerts": [],
                "trends": {},
            }

            # Export current metrics
            with self._lock:
                for key, metrics in self._metrics_by_room.items():
                    export_data["metrics"][key] = metrics.to_dict()

                # Export alert history if requested
                if include_alerts:
                    cutoff_time = datetime.utcnow() - timedelta(days=days_back)
                    for alert in self._alert_history:
                        if alert.triggered_time >= cutoff_time:
                            export_data["alerts"].append(alert.to_dict())

                # Export trend data if requested
                if include_trends:
                    for key, history in self._accuracy_history.items():
                        export_data["trends"][key] = {
                            "data_points": list(history),
                            "analysis": self._analyze_trend(list(history)),
                        }

            # Save to file
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(export_data, f, indent=2, default=str)

            total_records = (
                len(export_data["metrics"])
                + len(export_data["alerts"])
                + len(export_data["trends"])
            )

            logger.info(f"Exported {total_records} tracking records to {output_path}")
            return total_records

        except Exception as e:
            logger.error(f"Failed to export tracking data: {e}")
            raise AccuracyTrackingError("Failed to export tracking data", cause=e)

    def add_notification_callback(
        self, callback: Callable[[AccuracyAlert], None]
    ) -> None:
        """Add a notification callback for alerts."""
        if callback not in self.notification_callbacks:
            self.notification_callbacks.append(callback)
            logger.info("Added notification callback")

    def remove_notification_callback(
        self, callback: Callable[[AccuracyAlert], None]
    ) -> None:
        """Remove a notification callback."""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
            logger.info("Removed notification callback")

    def get_tracker_stats(self) -> Dict[str, Any]:
        """Get tracker system statistics."""
        try:
            with self._lock:
                return {
                    "monitoring_active": self._monitoring_active,
                    "metrics_tracked": {
                        "rooms": len(self._metrics_by_room),
                        "models": len(self._metrics_by_model),
                        "global": 1 if self._global_metrics else 0,
                    },
                    "alerts": {
                        "active": len(
                            [a for a in self._active_alerts.values() if not a.resolved]
                        ),
                        "total_stored": len(self._active_alerts),
                        "history_size": len(self._alert_history),
                    },
                    "trend_tracking": {
                        "entities_tracked": len(self._accuracy_history),
                        "total_data_points": sum(
                            len(h) for h in self._accuracy_history.values()
                        ),
                    },
                    "configuration": {
                        "monitoring_interval_seconds": self.monitoring_interval.total_seconds(),
                        "alert_thresholds": self.alert_thresholds,
                        "max_stored_alerts": self.max_stored_alerts,
                        "trend_analysis_points": self.trend_points,
                        "notification_callbacks": len(self.notification_callbacks),
                    },
                    "background_tasks": len(self._background_tasks),
                }

        except Exception as e:
            logger.error(f"Failed to get tracker stats: {e}")
            return {}

    # Private methods

    async def _monitoring_loop(self) -> None:
        """Background loop for continuous accuracy monitoring."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Update real-time metrics
                    await self._update_real_time_metrics()

                    # Check for alert conditions
                    await self._check_alert_conditions()

                    # Wait for next monitoring cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.monitoring_interval.total_seconds(),
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for monitoring interval
                    continue
                except Exception as e:
                    logger.error(f"Error in monitoring loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Monitoring loop failed: {e}")

    async def _alert_management_loop(self) -> None:
        """Background loop for alert management and escalation."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check for alert escalations
                    await self._check_alert_escalations()

                    # Cleanup resolved alerts
                    await self._cleanup_resolved_alerts()

                    # Wait for next alert management cycle (shorter interval)
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=30,  # Check every 30 seconds
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for alert management interval
                    continue
                except Exception as e:
                    logger.error(f"Error in alert management loop: {e}")
                    await asyncio.sleep(15)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Alert management loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Alert management loop failed: {e}")

    async def _update_real_time_metrics(self) -> None:
        """Update real-time metrics for all tracked entities."""
        try:
            current_time = datetime.utcnow()

            # Get unique room and model combinations from validator
            entities = set()

            # Get entities from validator records
            with self.validator._lock:
                for record in self.validator._validation_records.values():
                    entities.add((record.room_id, record.model_type))
                    entities.add((record.room_id, None))  # Room-only metrics
                    entities.add((None, record.model_type))  # Model-only metrics

            # Update metrics for each entity
            for room_id, model_type in entities:
                metrics = await self._calculate_real_time_metrics(
                    room_id, model_type, current_time
                )
                if metrics:
                    key = f"{room_id or 'all'}_{model_type or 'all'}"

                    with self._lock:
                        if room_id and not model_type:
                            self._metrics_by_room[room_id] = metrics
                        elif model_type and not room_id:
                            self._metrics_by_model[model_type] = metrics
                        elif not room_id and not model_type:
                            self._global_metrics = metrics

                        # Update trend history
                        self._accuracy_history[key].append(
                            {
                                "timestamp": current_time,
                                "accuracy_1h": metrics.window_1h_accuracy,
                                "accuracy_6h": metrics.window_6h_accuracy,
                                "accuracy_24h": metrics.window_24h_accuracy,
                            }
                        )

            logger.debug(f"Updated real-time metrics for {len(entities)} entities")

        except Exception as e:
            logger.error(f"Failed to update real-time metrics: {e}")

    async def _calculate_real_time_metrics(
        self, room_id: Optional[str], model_type: Optional[str], current_time: datetime
    ) -> Optional[RealTimeMetrics]:
        """Calculate real-time metrics for specific entity."""
        try:
            # Calculate metrics for different time windows
            windows = {"1h": 1, "6h": 6, "24h": 24}
            window_metrics = {}

            for window_name, hours in windows.items():
                metrics = await self.validator.get_accuracy_metrics(
                    room_id=room_id, model_type=model_type, hours_back=hours
                )
                window_metrics[window_name] = metrics

            # Create real-time metrics object
            real_time_metrics = RealTimeMetrics(
                room_id=room_id or "global",
                model_type=model_type,
                window_1h_accuracy=window_metrics["1h"].accuracy_rate,
                window_6h_accuracy=window_metrics["6h"].accuracy_rate,
                window_24h_accuracy=window_metrics["24h"].accuracy_rate,
                window_1h_mean_error=window_metrics["1h"].mean_error_minutes,
                window_6h_mean_error=window_metrics["6h"].mean_error_minutes,
                window_24h_mean_error=window_metrics["24h"].mean_error_minutes,
                window_1h_predictions=window_metrics["1h"].validated_predictions,
                window_6h_predictions=window_metrics["6h"].validated_predictions,
                window_24h_predictions=window_metrics["24h"].validated_predictions,
                recent_predictions_rate=window_metrics["1h"].predictions_per_hour,
                confidence_calibration=window_metrics[
                    "24h"
                ].confidence_calibration_score,
                last_updated=current_time,
            )

            # Calculate trend analysis
            key = f"{room_id or 'all'}_{model_type or 'all'}"
            trend_data = self._analyze_trend_for_entity(key)
            if trend_data:
                real_time_metrics.accuracy_trend = trend_data["direction"]
                real_time_metrics.trend_slope = trend_data["slope"]
                real_time_metrics.trend_confidence = trend_data["confidence"]

            # Calculate validation lag
            real_time_metrics.validation_lag_minutes = self._calculate_validation_lag(
                room_id, model_type
            )

            # Check for active alerts
            active_alerts = []
            with self._lock:
                for alert in self._active_alerts.values():
                    if (
                        (alert.room_id == room_id or room_id is None)
                        and (alert.model_type == model_type or model_type is None)
                        and not alert.resolved
                    ):
                        active_alerts.append(alert.alert_id)

            real_time_metrics.active_alerts = active_alerts

            return real_time_metrics

        except Exception as e:
            logger.error(
                f"Failed to calculate real-time metrics for {room_id}, {model_type}: {e}"
            )
            return None

    def _analyze_trend_for_entity(self, entity_key: str) -> Optional[Dict[str, Any]]:
        """Analyze accuracy trend for specific entity."""
        try:
            with self._lock:
                if entity_key not in self._accuracy_history:
                    return None

                history = list(self._accuracy_history[entity_key])

            return self._analyze_trend(history)

        except Exception as e:
            logger.error(f"Failed to analyze trend for {entity_key}: {e}")
            return None

    def _analyze_trend(self, data_points: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trend from data points."""
        if len(data_points) < 3:
            return {
                "direction": TrendDirection.UNKNOWN,
                "slope": 0.0,
                "confidence": 0.0,
            }

        try:
            # Extract 6h accuracy values for trend analysis
            values = [
                point["accuracy_6h"] for point in data_points if "accuracy_6h" in point
            ]

            if len(values) < 3:
                return {
                    "direction": TrendDirection.UNKNOWN,
                    "slope": 0.0,
                    "confidence": 0.0,
                }

            # Simple linear regression to calculate slope
            n = len(values)
            x = list(range(n))

            x_mean = statistics.mean(x)
            y_mean = statistics.mean(values)

            numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
            denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

            if denominator == 0:
                slope = 0.0
            else:
                slope = numerator / denominator

            # Calculate R-squared for confidence
            y_pred = [slope * (i - x_mean) + y_mean for i in x]
            ss_res = sum((values[i] - y_pred[i]) ** 2 for i in range(n))
            ss_tot = sum((values[i] - y_mean) ** 2 for i in range(n))

            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            confidence = max(0, min(1, r_squared))

            # Determine trend direction
            if abs(slope) < 1:  # Less than 1% change per data point
                direction = TrendDirection.STABLE
            elif slope > 0:
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.DEGRADING

            return {
                "direction": direction,
                "slope": slope,
                "confidence": confidence,
                "r_squared": r_squared,
                "data_points": len(values),
            }

        except Exception as e:
            logger.error(f"Failed to analyze trend: {e}")
            return {
                "direction": TrendDirection.UNKNOWN,
                "slope": 0.0,
                "confidence": 0.0,
            }

    def _calculate_global_trend(
        self, individual_trends: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Calculate global trend from individual entity trends."""
        if not individual_trends:
            return {
                "direction": TrendDirection.UNKNOWN,
                "average_slope": 0.0,
                "confidence": 0.0,
            }

        try:
            slopes = []
            confidences = []

            for trend in individual_trends.values():
                if "slope" in trend and "confidence" in trend:
                    slopes.append(trend["slope"])
                    confidences.append(trend["confidence"])

            if not slopes:
                return {
                    "direction": TrendDirection.UNKNOWN,
                    "average_slope": 0.0,
                    "confidence": 0.0,
                }

            avg_slope = statistics.mean(slopes)
            avg_confidence = statistics.mean(confidences)

            # Determine global direction
            if abs(avg_slope) < 0.5:
                direction = TrendDirection.STABLE
            elif avg_slope > 0:
                direction = TrendDirection.IMPROVING
            else:
                direction = TrendDirection.DEGRADING

            return {
                "direction": direction,
                "average_slope": avg_slope,
                "confidence": avg_confidence,
                "entities_analyzed": len(slopes),
            }

        except Exception as e:
            logger.error(f"Failed to calculate global trend: {e}")
            return {
                "direction": TrendDirection.UNKNOWN,
                "average_slope": 0.0,
                "confidence": 0.0,
            }

    def _calculate_validation_lag(
        self, room_id: Optional[str], model_type: Optional[str]
    ) -> float:
        """Calculate average validation lag for entity."""
        try:
            lags = []
            cutoff_time = datetime.utcnow() - timedelta(hours=6)

            with self.validator._lock:
                for record in self.validator._validation_records.values():
                    if record.validation_time and record.validation_time >= cutoff_time:
                        if room_id and record.room_id != room_id:
                            continue
                        if model_type and record.model_type != model_type:
                            continue

                        # Calculate lag from prediction time to validation
                        lag = (
                            record.validation_time - record.prediction_time
                        ).total_seconds() / 60
                        lags.append(lag)

            return statistics.mean(lags) if lags else 0.0

        except Exception as e:
            logger.error(f"Failed to calculate validation lag: {e}")
            return 0.0

    async def _check_alert_conditions(self) -> None:
        """Check for conditions that should trigger alerts."""
        try:
            with self._lock:
                # Check each tracked entity for alert conditions
                all_metrics = {**self._metrics_by_room, **self._metrics_by_model}

                if self._global_metrics:
                    all_metrics["global"] = self._global_metrics

                for entity_key, metrics in all_metrics.items():
                    await self._check_entity_alerts(entity_key, metrics)

        except Exception as e:
            logger.error(f"Failed to check alert conditions: {e}")

    async def _check_entity_alerts(
        self, entity_key: str, metrics: RealTimeMetrics
    ) -> None:
        """Check alert conditions for specific entity."""
        try:
            alerts_to_create = []

            # Check accuracy thresholds
            if metrics.window_6h_predictions >= 5:  # Only alert if enough predictions
                if (
                    metrics.window_6h_accuracy
                    < self.alert_thresholds["accuracy_critical"]
                ):
                    alerts_to_create.append(
                        {
                            "condition": "accuracy_critical",
                            "severity": AlertSeverity.CRITICAL,
                            "value": metrics.window_6h_accuracy,
                            "threshold": self.alert_thresholds["accuracy_critical"],
                            "description": f"Accuracy critically low: {metrics.window_6h_accuracy:.1f}%",
                        }
                    )
                elif (
                    metrics.window_6h_accuracy
                    < self.alert_thresholds["accuracy_warning"]
                ):
                    alerts_to_create.append(
                        {
                            "condition": "accuracy_warning",
                            "severity": AlertSeverity.WARNING,
                            "value": metrics.window_6h_accuracy,
                            "threshold": self.alert_thresholds["accuracy_warning"],
                            "description": f"Accuracy below warning threshold: {metrics.window_6h_accuracy:.1f}%",
                        }
                    )

            # Check error thresholds
            if metrics.window_6h_mean_error > self.alert_thresholds["error_critical"]:
                alerts_to_create.append(
                    {
                        "condition": "error_critical",
                        "severity": AlertSeverity.CRITICAL,
                        "value": metrics.window_6h_mean_error,
                        "threshold": self.alert_thresholds["error_critical"],
                        "description": f"Mean error critically high: {metrics.window_6h_mean_error:.1f} minutes",
                    }
                )
            elif metrics.window_6h_mean_error > self.alert_thresholds["error_warning"]:
                alerts_to_create.append(
                    {
                        "condition": "error_warning",
                        "severity": AlertSeverity.WARNING,
                        "value": metrics.window_6h_mean_error,
                        "threshold": self.alert_thresholds["error_warning"],
                        "description": f"Mean error above warning threshold: {metrics.window_6h_mean_error:.1f} minutes",
                    }
                )

            # Check trend degradation
            if (
                metrics.accuracy_trend == TrendDirection.DEGRADING
                and metrics.trend_confidence > 0.5
                and metrics.trend_slope < self.alert_thresholds["trend_degrading"]
            ):
                alerts_to_create.append(
                    {
                        "condition": "trend_degrading",
                        "severity": AlertSeverity.WARNING,
                        "value": metrics.trend_slope,
                        "threshold": self.alert_thresholds["trend_degrading"],
                        "description": f"Accuracy trend degrading: {metrics.trend_slope:.1f}%/hour",
                    }
                )

            # Check validation lag
            if (
                metrics.validation_lag_minutes
                > self.alert_thresholds["validation_lag_critical"]
            ):
                alerts_to_create.append(
                    {
                        "condition": "validation_lag_critical",
                        "severity": AlertSeverity.CRITICAL,
                        "value": metrics.validation_lag_minutes,
                        "threshold": self.alert_thresholds["validation_lag_critical"],
                        "description": f"Validation lag critically high: {metrics.validation_lag_minutes:.1f} minutes",
                    }
                )
            elif (
                metrics.validation_lag_minutes
                > self.alert_thresholds["validation_lag_warning"]
            ):
                alerts_to_create.append(
                    {
                        "condition": "validation_lag_warning",
                        "severity": AlertSeverity.WARNING,
                        "value": metrics.validation_lag_minutes,
                        "threshold": self.alert_thresholds["validation_lag_warning"],
                        "description": f"Validation lag above threshold: {metrics.validation_lag_minutes:.1f} minutes",
                    }
                )

            # Create alerts that don't already exist
            for alert_spec in alerts_to_create:
                alert_key = f"{entity_key}_{alert_spec['condition']}"

                # Check if alert already exists and is active
                existing_alert = None
                for alert in self._active_alerts.values():
                    if (
                        alert.room_id == metrics.room_id
                        and alert.model_type == metrics.model_type
                        and alert.trigger_condition == alert_spec["condition"]
                        and not alert.resolved
                    ):
                        existing_alert = alert
                        break

                if not existing_alert:
                    # Create new alert
                    alert = AccuracyAlert(
                        alert_id=f"acc_{self._alert_counter}_{int(datetime.utcnow().timestamp())}",
                        room_id=metrics.room_id,
                        model_type=metrics.model_type,
                        severity=alert_spec["severity"],
                        trigger_condition=alert_spec["condition"],
                        current_value=alert_spec["value"],
                        threshold_value=alert_spec["threshold"],
                        description=alert_spec["description"],
                        affected_metrics={
                            "accuracy_1h": metrics.window_1h_accuracy,
                            "accuracy_6h": metrics.window_6h_accuracy,
                            "accuracy_24h": metrics.window_24h_accuracy,
                            "error_6h": metrics.window_6h_mean_error,
                            "trend_slope": metrics.trend_slope,
                        },
                        recent_predictions=metrics.window_6h_predictions,
                        trend_data={
                            "direction": metrics.accuracy_trend.value,
                            "slope": metrics.trend_slope,
                            "confidence": metrics.trend_confidence,
                        },
                    )

                    self._active_alerts[alert.alert_id] = alert
                    self._alert_history.append(alert)
                    self._alert_counter += 1

                    # Update metrics with active alert
                    metrics.active_alerts.append(alert.alert_id)
                    metrics.last_alert_time = alert.triggered_time

                    # Notify callbacks
                    await self._notify_alert_callbacks(alert)

                    logger.warning(
                        f"Created {alert_spec['severity'].value} alert: "
                        f"{alert_spec['description']} for {entity_key}"
                    )

        except Exception as e:
            logger.error(f"Failed to check entity alerts for {entity_key}: {e}")

    async def _check_alert_escalations(self) -> None:
        """Check for alerts that need escalation."""
        try:
            with self._lock:
                for alert in self._active_alerts.values():
                    if alert.escalate():
                        # Notify callbacks about escalation
                        await self._notify_alert_callbacks(alert, escalation=True)

        except Exception as e:
            logger.error(f"Failed to check alert escalations: {e}")

    async def _cleanup_resolved_alerts(self) -> None:
        """Clean up resolved alerts and check for auto-resolution."""
        try:
            with self._lock:
                # Auto-resolve alerts if conditions have improved
                for alert in list(self._active_alerts.values()):
                    if alert.resolved:
                        continue

                    # Check if alert condition has been resolved
                    should_resolve = await self._should_auto_resolve_alert(alert)
                    if should_resolve:
                        alert.resolve()

                        logger.info(
                            f"Auto-resolved alert {alert.alert_id}: "
                            f"condition {alert.trigger_condition} improved"
                        )

                # Remove very old resolved alerts from active storage
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                alerts_to_remove = [
                    alert_id
                    for alert_id, alert in self._active_alerts.items()
                    if alert.resolved
                    and alert.resolved_time
                    and alert.resolved_time < cutoff_time
                ]

                for alert_id in alerts_to_remove:
                    del self._active_alerts[alert_id]

                if alerts_to_remove:
                    logger.debug(
                        f"Cleaned up {len(alerts_to_remove)} old resolved alerts"
                    )

        except Exception as e:
            logger.error(f"Failed to cleanup resolved alerts: {e}")

    async def _should_auto_resolve_alert(self, alert: AccuracyAlert) -> bool:
        """Check if alert should be auto-resolved based on current conditions."""
        try:
            # Get current metrics for the alert's entity
            metrics = await self.get_real_time_metrics(alert.room_id, alert.model_type)
            if not metrics:
                return False

            # Check if condition has improved beyond threshold
            condition = alert.trigger_condition
            current_value = alert.current_value
            threshold = alert.threshold_value

            if condition.startswith("accuracy_"):
                # Accuracy improved above threshold + buffer
                current_accuracy = metrics.window_6h_accuracy
                return current_accuracy > (threshold + 5)  # 5% buffer

            elif condition.startswith("error_"):
                # Error reduced below threshold - buffer
                current_error = metrics.window_6h_mean_error
                return current_error < (threshold - 2)  # 2 minute buffer

            elif condition == "trend_degrading":
                # Trend is no longer degrading
                return metrics.accuracy_trend != TrendDirection.DEGRADING

            elif condition.startswith("validation_lag_"):
                # Validation lag improved below threshold - buffer
                current_lag = metrics.validation_lag_minutes
                return current_lag < (threshold - 2)  # 2 minute buffer

            return False

        except Exception as e:
            logger.error(
                f"Failed to check auto-resolution for alert {alert.alert_id}: {e}"
            )
            return False

    async def _notify_alert_callbacks(
        self, alert: AccuracyAlert, escalation: bool = False
    ) -> None:
        """Notify all registered callbacks about an alert."""
        try:
            for callback in self.notification_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(alert)
                    else:
                        callback(alert)
                except Exception as e:
                    logger.error(f"Alert callback failed: {e}")

        except Exception as e:
            logger.error(f"Failed to notify alert callbacks: {e}")


# Custom exception for tracking-specific errors
class AccuracyTrackingError(OccupancyPredictionError):
    """Raised when accuracy tracking operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="ACCURACY_TRACKING_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
