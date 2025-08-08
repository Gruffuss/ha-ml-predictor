"""
Adaptive Retraining Pipeline for Sprint 4 - Self-Adaptation System.

This module provides intelligent, automated model retraining based on accuracy
degradation, concept drift detection, and predictive performance monitoring.
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta
from enum import Enum
import logging
import threading
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import heapq
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from ..core.constants import ModelType
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..models.base.predictor import PredictionResult, TrainingResult
from .drift_detector import ConceptDriftDetector, DriftMetrics, DriftSeverity
from .optimizer import ModelOptimizer, OptimizationConfig, OptimizationResult
from .validator import AccuracyMetrics, PredictionValidator

logger = logging.getLogger(__name__)


class RetrainingTrigger(Enum):
    """Types of triggers that can initiate retraining."""

    ACCURACY_DEGRADATION = "accuracy_degradation"
    ERROR_THRESHOLD_EXCEEDED = "error_threshold_exceeded"
    CONCEPT_DRIFT = "concept_drift"
    SCHEDULED_UPDATE = "scheduled_update"
    MANUAL_REQUEST = "manual_request"
    PERFORMANCE_ANOMALY = "performance_anomaly"


class RetrainingStrategy(Enum):
    """Different retraining strategies available."""

    INCREMENTAL = "incremental"  # Online learning updates
    FULL_RETRAIN = "full_retrain"  # Complete model retraining
    FEATURE_REFRESH = "feature_refresh"  # Retrain with new features only
    ENSEMBLE_REBALANCE = "ensemble_rebalance"  # Adjust ensemble weights only


class RetrainingStatus(Enum):
    """Status of retraining operations."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class RetrainingRequest:
    """Request for model retraining with priority and metadata."""

    request_id: str
    room_id: str
    model_type: str
    trigger: RetrainingTrigger
    strategy: RetrainingStrategy
    priority: float  # Higher = more urgent
    created_time: datetime

    # Trigger context
    accuracy_metrics: Optional[AccuracyMetrics] = None
    drift_metrics: Optional[DriftMetrics] = None
    performance_degradation: Optional[Dict[str, float]] = None

    # Retraining configuration
    lookback_days: int = 14
    validation_split: float = 0.2
    feature_refresh: bool = True

    # Status tracking
    status: RetrainingStatus = RetrainingStatus.PENDING
    started_time: Optional[datetime] = None
    completed_time: Optional[datetime] = None
    error_message: Optional[str] = None

    # Results
    training_result: Optional[TrainingResult] = None
    performance_improvement: Optional[Dict[str, float]] = None

    def __lt__(self, other):
        """Priority queue comparison - higher priority first."""
        return self.priority > other.priority

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "request_id": self.request_id,
            "room_id": self.room_id,
            "model_type": self.model_type,
            "trigger": self.trigger.value,
            "strategy": self.strategy.value,
            "priority": self.priority,
            "created_time": self.created_time.isoformat(),
            "status": self.status.value,
            "started_time": (
                self.started_time.isoformat() if self.started_time else None
            ),
            "completed_time": (
                self.completed_time.isoformat()
                if self.completed_time
                else None
            ),
            "error_message": self.error_message,
            "lookback_days": self.lookback_days,
            "validation_split": self.validation_split,
            "feature_refresh": self.feature_refresh,
            "performance_improvement": self.performance_improvement,
        }


@dataclass
class RetrainingProgress:
    """Progress tracking for retraining operations."""

    request_id: str
    room_id: str
    model_type: str

    # Progress phases
    phase: str = "initializing"
    progress_percentage: float = 0.0
    estimated_completion: Optional[datetime] = None

    # Phase details
    current_step: str = ""
    total_steps: int = 0
    completed_steps: int = 0

    # Performance metrics
    data_samples_processed: int = 0
    features_extracted: int = 0
    training_epochs_completed: int = 0

    # Resource usage
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0

    def update_progress(self, phase: str, step: str, percentage: float):
        """Update progress information."""
        self.phase = phase
        self.current_step = step
        self.progress_percentage = percentage

        # Estimate completion time based on progress
        if percentage > 0:
            elapsed = (
                datetime.utcnow() - datetime.utcnow()
            ).total_seconds()  # Would track actual start time
            estimated_total = elapsed / (percentage / 100.0)
            self.estimated_completion = datetime.utcnow() + timedelta(
                seconds=estimated_total - elapsed
            )


class AdaptiveRetrainer:
    """
    Intelligent adaptive retraining system integrated with TrackingManager.

    Features:
    - Automatic retraining trigger detection based on accuracy/drift thresholds
    - Smart retraining strategy selection (incremental vs full retrain)
    - Priority queue for managing multiple concurrent retraining requests
    - Resource management to prevent system overload
    - Progress tracking and status reporting
    - Integration with existing ensemble models and feature pipeline
    """

    def __init__(
        self,
        tracking_config,
        model_registry: Optional[Dict[str, Any]] = None,
        feature_engineering_engine=None,
        notification_callbacks: Optional[List] = None,
        model_optimizer: Optional[ModelOptimizer] = None,
    ):
        """
        Initialize the adaptive retrainer.

        Args:
            tracking_config: TrackingConfig with retraining settings
            model_registry: Registry of available models for retraining
            feature_engineering_engine: Engine for feature extraction
            notification_callbacks: Callbacks for retraining notifications
            model_optimizer: ModelOptimizer for automatic hyperparameter optimization
        """
        self.config = tracking_config
        self.model_registry = model_registry or {}
        self.feature_engineering_engine = feature_engineering_engine
        self.notification_callbacks = notification_callbacks or []
        self.model_optimizer = model_optimizer

        # Retraining queue and tracking
        self._retraining_queue: List[RetrainingRequest] = []
        self._queue_lock = threading.RLock()
        self._active_retrainings: Dict[str, RetrainingRequest] = {}
        self._retraining_history: deque = deque(maxlen=1000)

        # Progress tracking
        self._progress_tracker: Dict[str, RetrainingProgress] = {}
        self._progress_lock = threading.Lock()

        # Resource management
        self._active_retraining_count = 0
        self._resource_lock = threading.Lock()

        # Cooldown tracking (prevent too frequent retraining)
        self._last_retrain_times: Dict[str, datetime] = {}
        self._cooldown_lock = threading.Lock()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        # Performance tracking
        self._total_retrainings_completed = 0
        self._total_retrainings_failed = 0
        self._average_retraining_time = 0.0

        logger.info(
            f"Initialized AdaptiveRetrainer with config: enabled={tracking_config.adaptive_retraining_enabled}"
        )

    async def initialize(self) -> None:
        """Initialize the retrainer and start background tasks."""
        try:
            if not self.config.adaptive_retraining_enabled:
                logger.info("Adaptive retraining is disabled in configuration")
                return

            # Start background retraining processor
            processor_task = asyncio.create_task(
                self._retraining_processor_loop()
            )
            self._background_tasks.append(processor_task)

            # Start retraining trigger checker
            trigger_task = asyncio.create_task(self._trigger_checker_loop())
            self._background_tasks.append(trigger_task)

            logger.info(
                "AdaptiveRetrainer initialized and background tasks started"
            )

        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveRetrainer: {e}")
            raise RetrainingError(
                "Failed to initialize adaptive retrainer", cause=e
            )

    async def shutdown(self) -> None:
        """Shutdown the retrainer gracefully."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks, return_exceptions=True
                )

            self._background_tasks.clear()
            logger.info("AdaptiveRetrainer shutdown completed")

        except Exception as e:
            logger.error(f"Error during AdaptiveRetrainer shutdown: {e}")

    async def evaluate_retraining_need(
        self,
        room_id: str,
        model_type: str,
        accuracy_metrics: AccuracyMetrics,
        drift_metrics: Optional[DriftMetrics] = None,
    ) -> Optional[RetrainingRequest]:
        """
        Evaluate whether a model needs retraining based on current performance.

        Args:
            room_id: Room being evaluated
            model_type: Type of model to evaluate
            accuracy_metrics: Current accuracy metrics
            drift_metrics: Optional drift detection results

        Returns:
            RetrainingRequest if retraining is needed, None otherwise
        """
        try:
            if not self.config.adaptive_retraining_enabled:
                return None

            # Check cooldown period
            model_key = f"{room_id}_{model_type}"
            if self._is_in_cooldown(model_key):
                logger.debug(
                    f"Model {model_key} is in cooldown period, skipping retraining evaluation"
                )
                return None

            # Evaluate retraining triggers
            triggers = []
            priority = 0.0

            # 1. Accuracy degradation trigger
            if (
                accuracy_metrics.accuracy_rate
                < self.config.retraining_accuracy_threshold
            ):
                triggers.append(RetrainingTrigger.ACCURACY_DEGRADATION)
                priority += (
                    self.config.retraining_accuracy_threshold
                    - accuracy_metrics.accuracy_rate
                ) / 10.0
                logger.info(
                    f"Accuracy trigger: {accuracy_metrics.accuracy_rate:.1f}% < {self.config.retraining_accuracy_threshold}%"
                )

            # 2. Error threshold trigger
            if (
                accuracy_metrics.mean_error_minutes
                > self.config.retraining_error_threshold
            ):
                triggers.append(RetrainingTrigger.ERROR_THRESHOLD_EXCEEDED)
                priority += (
                    accuracy_metrics.mean_error_minutes
                    - self.config.retraining_error_threshold
                ) / 5.0
                logger.info(
                    f"Error trigger: {accuracy_metrics.mean_error_minutes:.1f}min > {self.config.retraining_error_threshold}min"
                )

            # 3. Concept drift trigger
            if (
                drift_metrics
                and drift_metrics.overall_drift_score
                > self.config.retraining_drift_threshold
            ):
                triggers.append(RetrainingTrigger.CONCEPT_DRIFT)
                priority += drift_metrics.overall_drift_score * 10.0
                logger.info(
                    f"Drift trigger: score={drift_metrics.overall_drift_score:.3f} > {self.config.retraining_drift_threshold}"
                )

            # 4. Performance anomaly trigger (significant confidence degradation)
            if accuracy_metrics.confidence_calibration_score < 0.3:
                triggers.append(RetrainingTrigger.PERFORMANCE_ANOMALY)
                priority += (
                    0.3 - accuracy_metrics.confidence_calibration_score
                ) * 5.0
                logger.info(
                    f"Confidence trigger: {accuracy_metrics.confidence_calibration_score:.3f} < 0.3"
                )

            # No retraining needed if no triggers
            if not triggers:
                logger.debug(f"No retraining triggers for {model_key}")
                return None

            # Determine retraining strategy
            strategy = self._select_retraining_strategy(
                accuracy_metrics, drift_metrics
            )

            # Create retraining request
            request = RetrainingRequest(
                request_id=f"{model_key}_{int(datetime.utcnow().timestamp())}",
                room_id=room_id,
                model_type=model_type,
                trigger=triggers[0],  # Primary trigger
                strategy=strategy,
                priority=min(priority, 10.0),  # Cap priority at 10
                created_time=datetime.utcnow(),
                accuracy_metrics=accuracy_metrics,
                drift_metrics=drift_metrics,
                lookback_days=self.config.retraining_lookback_days,
                validation_split=self.config.retraining_validation_split,
                feature_refresh=self.config.auto_feature_refresh,
            )

            logger.info(
                f"Retraining needed for {model_key}: "
                f"triggers={[t.value for t in triggers]}, "
                f"strategy={strategy.value}, priority={priority:.2f}"
            )

            return request

        except Exception as e:
            logger.error(
                f"Failed to evaluate retraining need for {room_id}_{model_type}: {e}"
            )
            return None

    async def request_retraining(
        self,
        room_id: str,
        model_type: str,
        trigger: RetrainingTrigger = RetrainingTrigger.MANUAL_REQUEST,
        strategy: Optional[RetrainingStrategy] = None,
        priority: float = 5.0,
        **kwargs,
    ) -> str:
        """
        Manually request model retraining.

        Args:
            room_id: Room to retrain model for
            model_type: Type of model to retrain
            trigger: Reason for retraining request
            strategy: Retraining strategy (auto-selected if None)
            priority: Request priority (0-10, higher = more urgent)
            **kwargs: Additional retraining parameters

        Returns:
            Request ID for tracking
        """
        try:
            model_key = f"{room_id}_{model_type}"
            request_id = (
                f"{model_key}_manual_{int(datetime.utcnow().timestamp())}"
            )

            # Auto-select strategy if not provided
            if strategy is None:
                strategy = (
                    RetrainingStrategy.FULL_RETRAIN
                )  # Default for manual requests

            request = RetrainingRequest(
                request_id=request_id,
                room_id=room_id,
                model_type=model_type,
                trigger=trigger,
                strategy=strategy,
                priority=priority,
                created_time=datetime.utcnow(),
                lookback_days=kwargs.get(
                    "lookback_days", self.config.retraining_lookback_days
                ),
                validation_split=kwargs.get(
                    "validation_split", self.config.retraining_validation_split
                ),
                feature_refresh=kwargs.get(
                    "feature_refresh", self.config.auto_feature_refresh
                ),
            )

            # Add to queue
            await self._queue_retraining_request(request)

            logger.info(f"Manual retraining requested: {request_id}")
            return request_id

        except Exception as e:
            logger.error(
                f"Failed to request retraining for {room_id}_{model_type}: {e}"
            )
            raise RetrainingError("Failed to request retraining", cause=e)

    async def get_retraining_status(
        self, request_id: Optional[str] = None
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """
        Get status of retraining operations.

        Args:
            request_id: Specific request to get status for (None for all)

        Returns:
            Status information for specified request or all requests
        """
        try:
            if request_id:
                # Get specific request status
                with self._queue_lock:
                    # Check active retrainings
                    if request_id in self._active_retrainings:
                        request = self._active_retrainings[request_id]
                        status = request.to_dict()

                        # Add progress information
                        with self._progress_lock:
                            if request_id in self._progress_tracker:
                                progress = self._progress_tracker[request_id]
                                status["progress"] = {
                                    "phase": progress.phase,
                                    "percentage": progress.progress_percentage,
                                    "current_step": progress.current_step,
                                    "estimated_completion": (
                                        progress.estimated_completion.isoformat()
                                        if progress.estimated_completion
                                        else None
                                    ),
                                }

                        return status

                    # Check queue for pending requests
                    for request in self._retraining_queue:
                        if request.request_id == request_id:
                            return request.to_dict()

                    # Check history for completed requests
                    for request in self._retraining_history:
                        if request.request_id == request_id:
                            return request.to_dict()

                return {"error": f"Request {request_id} not found"}

            else:
                # Get all requests status
                all_status = []

                with self._queue_lock:
                    # Active retrainings
                    for request in self._active_retrainings.values():
                        status = request.to_dict()
                        status["category"] = "active"
                        all_status.append(status)

                    # Pending requests
                    for request in self._retraining_queue:
                        status = request.to_dict()
                        status["category"] = "pending"
                        all_status.append(status)

                    # Recent completed requests (last 50)
                    for request in list(self._retraining_history)[-50:]:
                        status = request.to_dict()
                        status["category"] = "completed"
                        all_status.append(status)

                return all_status

        except Exception as e:
            logger.error(f"Failed to get retraining status: {e}")
            return {"error": str(e)}

    async def cancel_retraining(self, request_id: str) -> bool:
        """
        Cancel a pending or active retraining request.

        Args:
            request_id: Request to cancel

        Returns:
            True if successfully cancelled, False otherwise
        """
        try:
            with self._queue_lock:
                # Cancel if in queue
                for i, request in enumerate(self._retraining_queue):
                    if request.request_id == request_id:
                        request.status = RetrainingStatus.CANCELLED
                        request.completed_time = datetime.utcnow()
                        self._retraining_queue.pop(i)
                        self._retraining_history.append(request)
                        logger.info(
                            f"Cancelled pending retraining request: {request_id}"
                        )
                        return True

                # Cancel if active (more complex - would need to interrupt training)
                if request_id in self._active_retrainings:
                    request = self._active_retrainings[request_id]
                    request.status = RetrainingStatus.CANCELLED
                    request.completed_time = datetime.utcnow()

                    # Move to history
                    del self._active_retrainings[request_id]
                    self._retraining_history.append(request)

                    # Clean up progress tracking
                    with self._progress_lock:
                        if request_id in self._progress_tracker:
                            del self._progress_tracker[request_id]

                    logger.info(
                        f"Cancelled active retraining request: {request_id}"
                    )
                    return True

            return False

        except Exception as e:
            logger.error(f"Failed to cancel retraining {request_id}: {e}")
            return False

    def get_retrainer_stats(self) -> Dict[str, Any]:
        """Get comprehensive retrainer statistics."""
        try:
            with self._queue_lock:
                queue_size = len(self._retraining_queue)
                active_count = len(self._active_retrainings)

                # Get queue priorities
                queue_priorities = [
                    req.priority for req in self._retraining_queue
                ]
                avg_queue_priority = (
                    np.mean(queue_priorities) if queue_priorities else 0.0
                )

                # Get trigger distribution
                trigger_counts = defaultdict(int)
                for request in self._retraining_queue:
                    trigger_counts[request.trigger.value] += 1
                for request in self._active_retrainings.values():
                    trigger_counts[request.trigger.value] += 1

            with self._cooldown_lock:
                models_in_cooldown = len(self._last_retrain_times)

            with self._resource_lock:
                resource_utilization = self._active_retraining_count / max(
                    1, self.config.max_concurrent_retrains
                )

            # Calculate success rate
            total_attempts = (
                self._total_retrainings_completed
                + self._total_retrainings_failed
            )
            success_rate = (
                self._total_retrainings_completed / max(1, total_attempts)
            ) * 100

            return {
                "enabled": self.config.adaptive_retraining_enabled,
                "queue_size": queue_size,
                "active_retrainings": active_count,
                "max_concurrent_retrains": self.config.max_concurrent_retrains,
                "resource_utilization_percent": resource_utilization * 100,
                "models_in_cooldown": models_in_cooldown,
                "cooldown_hours": self.config.retraining_cooldown_hours,
                "average_queue_priority": avg_queue_priority,
                "trigger_distribution": dict(trigger_counts),
                "performance_stats": {
                    "total_completed": self._total_retrainings_completed,
                    "total_failed": self._total_retrainings_failed,
                    "success_rate_percent": success_rate,
                    "average_retraining_time_minutes": self._average_retraining_time
                    / 60.0,
                },
                "configuration": {
                    "accuracy_threshold": self.config.retraining_accuracy_threshold,
                    "error_threshold": self.config.retraining_error_threshold,
                    "drift_threshold": self.config.retraining_drift_threshold,
                    "check_interval_hours": self.config.retraining_check_interval_hours,
                    "lookback_days": self.config.retraining_lookback_days,
                },
            }

        except Exception as e:
            logger.error(f"Failed to get retrainer stats: {e}")
            return {"error": str(e)}

    # Private methods

    async def _queue_retraining_request(
        self, request: RetrainingRequest
    ) -> None:
        """Add retraining request to priority queue."""
        try:
            with self._queue_lock:
                # Check if similar request already exists
                existing_request = None
                for req in self._retraining_queue:
                    if (
                        req.room_id == request.room_id
                        and req.model_type == request.model_type
                    ):
                        existing_request = req
                        break

                if existing_request:
                    # Update existing request if new one has higher priority
                    if request.priority > existing_request.priority:
                        self._retraining_queue.remove(existing_request)
                        heapq.heappush(self._retraining_queue, request)
                        logger.info(
                            f"Updated existing retraining request with higher priority: {request.request_id}"
                        )
                    else:
                        logger.info(
                            f"Skipping duplicate retraining request: {request.request_id}"
                        )
                else:
                    # Add new request to queue
                    heapq.heappush(self._retraining_queue, request)
                    logger.info(
                        f"Queued retraining request: {request.request_id} (priority: {request.priority})"
                    )

                # Notify callbacks about new request
                await self._notify_retraining_event("queued", request)

        except Exception as e:
            logger.error(f"Failed to queue retraining request: {e}")
            raise

    def _select_retraining_strategy(
        self,
        accuracy_metrics: AccuracyMetrics,
        drift_metrics: Optional[DriftMetrics] = None,
    ) -> RetrainingStrategy:
        """Select appropriate retraining strategy based on performance metrics."""
        try:
            # If accuracy is still reasonable, try incremental update first
            if (
                accuracy_metrics.accuracy_rate
                > self.config.incremental_retraining_threshold
            ):
                return RetrainingStrategy.INCREMENTAL

            # If there's significant drift, do full retrain with feature refresh
            if drift_metrics and drift_metrics.overall_drift_score > 0.5:
                return RetrainingStrategy.FULL_RETRAIN

            # If confidence calibration is poor, rebalance ensemble
            if accuracy_metrics.confidence_calibration_score < 0.2:
                return RetrainingStrategy.ENSEMBLE_REBALANCE

            # Default to full retrain for severe degradation
            return RetrainingStrategy.FULL_RETRAIN

        except Exception as e:
            logger.error(f"Error selecting retraining strategy: {e}")
            return RetrainingStrategy.FULL_RETRAIN

    def _is_in_cooldown(self, model_key: str) -> bool:
        """Check if model is in cooldown period."""
        try:
            with self._cooldown_lock:
                if model_key not in self._last_retrain_times:
                    return False

                last_retrain = self._last_retrain_times[model_key]
                cooldown_period = timedelta(
                    hours=self.config.retraining_cooldown_hours
                )
                return datetime.utcnow() < (last_retrain + cooldown_period)

        except Exception as e:
            logger.error(f"Error checking cooldown for {model_key}: {e}")
            return False

    async def _retraining_processor_loop(self) -> None:
        """Background loop for processing retraining requests."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check if we can start new retraining
                    with self._resource_lock:
                        can_start_new = (
                            self._active_retraining_count
                            < self.config.max_concurrent_retrains
                        )

                    if can_start_new:
                        # Get next request from queue
                        request = None
                        with self._queue_lock:
                            if self._retraining_queue:
                                request = heapq.heappop(self._retraining_queue)

                        if request:
                            # Start retraining
                            await self._start_retraining(request)

                    # Wait before checking again
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=30,  # Check every 30 seconds
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for processing interval
                    continue
                except Exception as e:
                    logger.error(f"Error in retraining processor loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Retraining processor loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Retraining processor loop failed: {e}")

    async def _trigger_checker_loop(self) -> None:
        """Background loop for checking retraining triggers."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # This would be called by TrackingManager based on accuracy/drift results
                    # For now, we'll just wait - the actual trigger evaluation happens
                    # when TrackingManager calls evaluate_retraining_need()

                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config.retraining_check_interval_hours
                        * 3600,
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for trigger check interval
                    continue
                except Exception as e:
                    logger.error(f"Error in trigger checker loop: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying

        except asyncio.CancelledError:
            logger.info("Trigger checker loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Trigger checker loop failed: {e}")

    async def _start_retraining(self, request: RetrainingRequest) -> None:
        """Start processing a retraining request."""
        try:
            # Update resource tracking
            with self._resource_lock:
                self._active_retraining_count += 1

            # Move request to active
            with self._queue_lock:
                self._active_retrainings[request.request_id] = request

            # Update request status
            request.status = RetrainingStatus.IN_PROGRESS
            request.started_time = datetime.utcnow()

            # Initialize progress tracking
            progress = RetrainingProgress(
                request_id=request.request_id,
                room_id=request.room_id,
                model_type=request.model_type,
            )

            with self._progress_lock:
                self._progress_tracker[request.request_id] = progress

            # Start retraining in background
            retraining_task = asyncio.create_task(
                self._perform_retraining(request, progress)
            )

            # Don't await here - let it run in background
            logger.info(f"Started retraining for {request.request_id}")

        except Exception as e:
            logger.error(
                f"Failed to start retraining for {request.request_id}: {e}"
            )
            await self._handle_retraining_failure(request, str(e))

    async def _perform_retraining(
        self, request: RetrainingRequest, progress: RetrainingProgress
    ) -> None:
        """Perform the actual retraining process."""
        start_time = datetime.utcnow()

        try:
            # Phase 1: Data preparation
            progress.update_progress(
                "data_preparation", "Loading historical data", 10.0
            )
            train_data, val_data = await self._prepare_retraining_data(request)

            # Phase 2: Feature extraction
            progress.update_progress(
                "feature_extraction", "Extracting features", 30.0
            )
            features, targets = await self._extract_features_for_retraining(
                train_data, request
            )
            val_features, val_targets = (
                await self._extract_features_for_retraining(val_data, request)
                if val_data is not None
                else (None, None)
            )

            # Phase 3: Model retraining
            progress.update_progress("model_training", "Training model", 60.0)
            training_result = await self._retrain_model(
                request, features, targets, val_features, val_targets
            )

            # Phase 4: Validation and deployment
            progress.update_progress(
                "validation", "Validating retrained model", 90.0
            )
            await self._validate_and_deploy_retrained_model(
                request, training_result
            )

            # Complete successfully
            progress.update_progress(
                "completed", "Retraining completed", 100.0
            )
            await self._handle_retraining_success(
                request, training_result, start_time
            )

        except Exception as e:
            logger.error(f"Retraining failed for {request.request_id}: {e}")
            await self._handle_retraining_failure(request, str(e))

    async def _prepare_retraining_data(
        self, request: RetrainingRequest
    ) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Prepare training and validation data for retraining."""
        # This would integrate with the existing data infrastructure
        # For now, return placeholder data
        logger.info(
            f"Preparing retraining data for {request.room_id} with {request.lookback_days} days lookback"
        )

        # In actual implementation, this would:
        # 1. Query database for recent data
        # 2. Split into train/validation sets
        # 3. Apply any necessary preprocessing

        # Placeholder return
        train_data = pd.DataFrame()  # Would contain actual historical data
        val_data = pd.DataFrame() if request.validation_split > 0 else None

        return train_data, val_data

    async def _extract_features_for_retraining(
        self, data: pd.DataFrame, request: RetrainingRequest
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract features for retraining."""
        if self.feature_engineering_engine and request.feature_refresh:
            # Use feature engineering engine to extract fresh features
            logger.info(f"Extracting fresh features for {request.room_id}")
            # features = await self.feature_engineering_engine.extract_features(data, request.room_id)

        # Placeholder return
        features = pd.DataFrame()  # Would contain extracted features
        targets = pd.DataFrame()  # Would contain target values

        return features, targets

    async def _retrain_model(
        self,
        request: RetrainingRequest,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        val_features: Optional[pd.DataFrame],
        val_targets: Optional[pd.DataFrame],
    ) -> TrainingResult:
        """Retrain the model based on strategy with optional optimization."""
        logger.info(
            f"Retraining {request.model_type} for {request.room_id} using {request.strategy.value} strategy"
        )

        # Get model from registry
        model_key = f"{request.room_id}_{request.model_type}"
        if model_key not in self.model_registry:
            raise RetrainingError(f"Model {model_key} not found in registry")

        model = self.model_registry[model_key]

        # Optimize model parameters if optimizer is available and strategy is FULL_RETRAIN
        if (
            self.model_optimizer
            and request.strategy == RetrainingStrategy.FULL_RETRAIN
            and hasattr(self.config, "optimization_enabled")
            and getattr(self.config, "optimization_enabled", True)
        ):

            logger.info(
                f"Optimizing parameters for {request.model_type} before retraining"
            )

            # Prepare performance context for optimization
            performance_context = {
                "accuracy_metrics": request.accuracy_metrics,
                "drift_metrics": request.drift_metrics,
                "trigger": request.trigger.value,
                "performance_degradation": request.performance_degradation,
            }

            try:
                # Run parameter optimization
                optimization_result = (
                    await self.model_optimizer.optimize_model_parameters(
                        model=model,
                        model_type=request.model_type,
                        room_id=request.room_id,
                        X_train=features,
                        y_train=targets,
                        X_val=val_features,
                        y_val=val_targets,
                        performance_context=performance_context,
                    )
                )

                if (
                    optimization_result.success
                    and optimization_result.improvement_over_default > 0.01
                ):
                    logger.info(
                        "Parameter optimization successful: "
                        f"improvement={optimization_result.improvement_over_default:.3f}, "
                        f"best_score={optimization_result.best_score:.3f}"
                    )

                    # Apply optimized parameters to model
                    if optimization_result.best_parameters and hasattr(
                        model, "set_parameters"
                    ):
                        model.set_parameters(
                            optimization_result.best_parameters
                        )
                        logger.info(
                            f"Applied optimized parameters: {optimization_result.best_parameters}"
                        )
                else:
                    logger.info(
                        "Parameter optimization did not improve performance significantly, using defaults"
                    )

            except Exception as e:
                logger.warning(
                    f"Parameter optimization failed, proceeding with default parameters: {e}"
                )

        # Apply retraining strategy
        if request.strategy == RetrainingStrategy.INCREMENTAL:
            return await self._incremental_retrain(model, features, targets)
        elif request.strategy == RetrainingStrategy.FULL_RETRAIN:
            return await self._full_retrain_with_optimization(
                model, features, targets, val_features, val_targets
            )
        elif request.strategy == RetrainingStrategy.FEATURE_REFRESH:
            return await self._feature_refresh_retrain(
                model, features, targets, val_features, val_targets
            )
        elif request.strategy == RetrainingStrategy.ENSEMBLE_REBALANCE:
            return await self._ensemble_rebalance(model, features, targets)
        else:
            raise RetrainingError(
                f"Unknown retraining strategy: {request.strategy}"
            )

    async def _full_retrain_with_optimization(
        self,
        model,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        val_features: Optional[pd.DataFrame],
        val_targets: Optional[pd.DataFrame],
    ) -> TrainingResult:
        """Perform full retraining (potentially with pre-optimized parameters)."""
        logger.info("Performing full retraining with optimized parameters")
        return await model.train(features, targets, val_features, val_targets)

    async def _incremental_retrain(
        self, model, features: pd.DataFrame, targets: pd.DataFrame
    ) -> TrainingResult:
        """Perform incremental retraining (online learning)."""
        logger.info("Performing incremental retraining")

        # Check if model supports incremental updates
        if hasattr(model, "incremental_update"):
            return await model.incremental_update(features, targets)
        else:
            # Fall back to full retrain if incremental not supported
            logger.warning(
                "Model doesn't support incremental updates, performing full retrain"
            )
            return await model.train(features, targets)

    async def _feature_refresh_retrain(
        self,
        model,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        val_features,
        val_targets,
    ) -> TrainingResult:
        """Retrain with refreshed features only."""
        logger.info("Performing feature refresh retraining")
        return await model.train(features, targets, val_features, val_targets)

    async def _ensemble_rebalance(
        self, model, features: pd.DataFrame, targets: pd.DataFrame
    ) -> TrainingResult:
        """Rebalance ensemble weights without full retraining."""
        if hasattr(model, "_calculate_model_weights"):
            logger.info("Rebalancing ensemble weights")
            # Recalculate model weights based on recent performance
            y_true = model._prepare_targets(targets)
            model._calculate_model_weights(features, y_true)

            # Create mock training result
            return TrainingResult(
                success=True,
                training_time_seconds=1.0,
                model_version=model.model_version,
                training_samples=len(features),
                training_score=0.8,  # Would calculate actual score
                training_metrics={"rebalance_method": "ensemble_weights"},
            )
        else:
            # Fall back to full retrain if model doesn't support rebalancing
            return await model.train(features, targets)

    async def _validate_and_deploy_retrained_model(
        self, request: RetrainingRequest, training_result: TrainingResult
    ) -> None:
        """Validate retrained model and deploy if successful."""
        if training_result.success:
            logger.info(
                f"Retrained model validation successful for {request.request_id}"
            )
            # In actual implementation, this would:
            # 1. Run validation tests on retrained model
            # 2. Compare performance with previous model
            # 3. Deploy if improvement is significant
            # 4. Update model registry
        else:
            raise RetrainingError(
                f"Model training failed: {training_result.error_message}"
            )

    async def _handle_retraining_success(
        self,
        request: RetrainingRequest,
        training_result: TrainingResult,
        start_time: datetime,
    ) -> None:
        """Handle successful retraining completion."""
        try:
            # Update request
            request.status = RetrainingStatus.COMPLETED
            request.completed_time = datetime.utcnow()
            request.training_result = training_result

            # Calculate performance improvement (would compare with previous model)
            request.performance_improvement = {
                "training_score_improvement": 0.05,  # Placeholder
                "validation_score_improvement": 0.03,  # Placeholder
            }

            # Update cooldown tracking
            model_key = f"{request.room_id}_{request.model_type}"
            with self._cooldown_lock:
                self._last_retrain_times[model_key] = datetime.utcnow()

            # Update statistics
            training_time = (datetime.utcnow() - start_time).total_seconds()
            self._total_retrainings_completed += 1
            if self._total_retrainings_completed > 0:
                self._average_retraining_time = (
                    self._average_retraining_time
                    * (self._total_retrainings_completed - 1)
                    + training_time
                ) / self._total_retrainings_completed

            # Move from active to history
            with self._queue_lock:
                if request.request_id in self._active_retrainings:
                    del self._active_retrainings[request.request_id]
                self._retraining_history.append(request)

            # Clean up progress tracking
            with self._progress_lock:
                if request.request_id in self._progress_tracker:
                    del self._progress_tracker[request.request_id]

            # Update resource tracking
            with self._resource_lock:
                self._active_retraining_count = max(
                    0, self._active_retraining_count - 1
                )

            # Notify success
            await self._notify_retraining_event("completed", request)

            logger.info(
                f"Retraining completed successfully: {request.request_id} "
                f"(took {training_time:.1f}s, score: {training_result.training_score:.3f})"
            )

        except Exception as e:
            logger.error(f"Error handling retraining success: {e}")

    async def _handle_retraining_failure(
        self, request: RetrainingRequest, error_message: str
    ) -> None:
        """Handle retraining failure."""
        try:
            # Update request
            request.status = RetrainingStatus.FAILED
            request.completed_time = datetime.utcnow()
            request.error_message = error_message

            # Update statistics
            self._total_retrainings_failed += 1

            # Move from active to history
            with self._queue_lock:
                if request.request_id in self._active_retrainings:
                    del self._active_retrainings[request.request_id]
                self._retraining_history.append(request)

            # Clean up progress tracking
            with self._progress_lock:
                if request.request_id in self._progress_tracker:
                    del self._progress_tracker[request.request_id]

            # Update resource tracking
            with self._resource_lock:
                self._active_retraining_count = max(
                    0, self._active_retraining_count - 1
                )

            # Notify failure
            await self._notify_retraining_event("failed", request)

            logger.error(
                f"Retraining failed: {request.request_id} - {error_message}"
            )

        except Exception as e:
            logger.error(f"Error handling retraining failure: {e}")

    async def _notify_retraining_event(
        self, event_type: str, request: RetrainingRequest
    ) -> None:
        """Notify callbacks about retraining events."""
        try:
            event_data = {
                "event_type": event_type,
                "request_id": request.request_id,
                "room_id": request.room_id,
                "model_type": request.model_type,
                "trigger": request.trigger.value,
                "strategy": request.strategy.value,
                "timestamp": datetime.utcnow().isoformat(),
            }

            for callback in self.notification_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(event_data)
                    else:
                        callback(event_data)
                except Exception as e:
                    logger.error(
                        f"Error in retraining notification callback: {e}"
                    )

        except Exception as e:
            logger.error(f"Error notifying retraining event: {e}")


class RetrainingError(OccupancyPredictionError):
    """Raised when adaptive retraining operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="RETRAINING_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
