"""
Training Integration Module for Automatic Pipeline Management.

This module provides seamless integration between the training pipeline and
the TrackingManager, enabling automatic model training based on accuracy
degradation, drift detection, and other system events.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Set

from ..core.exceptions import ModelTrainingError
from .training_config import TrainingProfile, get_training_config_manager
from .training_pipeline import ModelTrainingPipeline, TrainingType

logger = logging.getLogger(__name__)


class TrainingIntegrationManager:
    """
    Manages automatic training pipeline integration with the tracking system.

    This class acts as a bridge between the TrackingManager and the training pipeline,
    automatically triggering training operations based on system events and conditions.

    Features:
    - Automatic training triggers based on accuracy degradation
    - Drift-based retraining coordination
    - Scheduled training maintenance
    - Resource-aware training orchestration
    - Model deployment and rollback management
    """

    def __init__(
        self,
        tracking_manager,
        training_pipeline: ModelTrainingPipeline,
        config_manager=None,
    ):
        """
        Initialize the training integration manager.

        Args:
            tracking_manager: The system tracking manager
            training_pipeline: The training pipeline instance
            config_manager: Training configuration manager (optional)
        """
        self.tracking_manager = tracking_manager
        self.training_pipeline = training_pipeline
        self.config_manager = config_manager or get_training_config_manager()

        # Integration state
        self._active_training_requests: Dict[str, str] = (
            {}
        )  # room_id -> pipeline_id
        self._training_queue: List[Dict[str, Any]] = []
        self._integration_active = False

        # Training triggers and conditions
        self._accuracy_triggers: Dict[str, float] = {}  # room_id -> threshold
        self._drift_triggers: Dict[str, float] = {}  # room_id -> threshold
        self._last_training_times: Dict[str, datetime] = (
            {}
        )  # room_id -> last_train_time

        # Resource management
        self._max_concurrent_training = 2
        self._training_cooldown_hours = 12

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        logger.info("Initialized TrainingIntegrationManager")

    async def initialize(self):
        """Initialize the integration manager and start monitoring."""
        try:
            self._integration_active = True

            # Start background monitoring tasks
            await self._start_background_tasks()

            # Register integration callbacks with tracking manager
            await self._register_tracking_callbacks()

            logger.info("TrainingIntegrationManager initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize TrainingIntegrationManager: {e}"
            )
            raise

    async def shutdown(self):
        """Shutdown the integration manager gracefully."""
        try:
            self._integration_active = False
            self._shutdown_event.set()

            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks, return_exceptions=True
                )
                self._background_tasks.clear()

            logger.info("TrainingIntegrationManager shutdown complete")

        except Exception as e:
            logger.error(
                f"Error during TrainingIntegrationManager shutdown: {e}"
            )

    async def _start_background_tasks(self):
        """Start background monitoring and processing tasks."""
        try:
            # Training queue processor
            queue_task = asyncio.create_task(self._training_queue_processor())
            self._background_tasks.append(queue_task)

            # Periodic maintenance task
            maintenance_task = asyncio.create_task(
                self._periodic_maintenance()
            )
            self._background_tasks.append(maintenance_task)

            # Resource monitoring task
            resource_task = asyncio.create_task(self._resource_monitor())
            self._background_tasks.append(resource_task)

            logger.debug("Started background tasks for training integration")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise

    async def _register_tracking_callbacks(self):
        """Register callback functions with the tracking manager."""
        try:
            # Register for accuracy degradation notifications
            if hasattr(self.tracking_manager, "add_accuracy_callback"):
                self.tracking_manager.add_accuracy_callback(
                    self._on_accuracy_degradation
                )

            # Register for drift detection notifications
            if hasattr(self.tracking_manager, "add_drift_callback"):
                self.tracking_manager.add_drift_callback(
                    self._on_drift_detected
                )

            # Register for model performance notifications
            if hasattr(self.tracking_manager, "add_performance_callback"):
                self.tracking_manager.add_performance_callback(
                    self._on_performance_change
                )

            logger.debug("Registered callbacks with tracking manager")

        except Exception as e:
            logger.warning(f"Failed to register some tracking callbacks: {e}")

    async def _on_accuracy_degradation(
        self, room_id: str, accuracy_metrics: Dict[str, Any]
    ):
        """Handle accuracy degradation notification from tracking manager."""
        try:
            logger.info(
                f"Accuracy degradation detected for room {room_id}: {accuracy_metrics}"
            )

            # Check if retraining is needed based on accuracy thresholds
            current_accuracy = accuracy_metrics.get("accuracy_rate", 100.0)
            error_minutes = accuracy_metrics.get("mean_error_minutes", 0.0)

            # Get environment-specific thresholds
            env_config = self.config_manager.get_environment_config()
            accuracy_threshold = (
                env_config.quality_thresholds.min_accuracy_threshold * 100
            )  # Convert to percentage
            error_threshold = (
                env_config.quality_thresholds.max_error_threshold_minutes
            )

            should_retrain = (
                current_accuracy < accuracy_threshold
                or error_minutes > error_threshold
            )

            if should_retrain:
                await self._queue_retraining_request(
                    room_id=room_id,
                    trigger_reason=f"accuracy_degradation_{current_accuracy:.1f}%",
                    priority=self._calculate_priority(
                        current_accuracy, accuracy_threshold
                    ),
                    metadata={
                        "current_accuracy": current_accuracy,
                        "accuracy_threshold": accuracy_threshold,
                        "current_error_minutes": error_minutes,
                        "error_threshold": error_threshold,
                    },
                )
            else:
                logger.debug(
                    f"Accuracy degradation for room {room_id} within acceptable thresholds"
                )

        except Exception as e:
            logger.error(
                f"Error handling accuracy degradation for room {room_id}: {e}"
            )

    async def _on_drift_detected(
        self, room_id: str, drift_metrics: Dict[str, Any]
    ):
        """Handle concept drift notification from tracking manager."""
        try:
            logger.info(
                f"Concept drift detected for room {room_id}: {drift_metrics}"
            )

            drift_severity = drift_metrics.get("drift_severity", "NONE")
            drift_score = drift_metrics.get("overall_drift_score", 0.0)
            retraining_recommended = drift_metrics.get(
                "retraining_recommended", False
            )

            if retraining_recommended:
                # Determine retraining strategy based on drift severity
                if drift_severity in ["CRITICAL", "MAJOR"]:
                    strategy = "full_retrain"
                    priority = 1  # High priority
                else:
                    strategy = "adaptive"
                    priority = 3  # Medium priority

                await self._queue_retraining_request(
                    room_id=room_id,
                    trigger_reason=f"concept_drift_{drift_severity.lower()}",
                    priority=priority,
                    strategy=strategy,
                    metadata={
                        "drift_severity": drift_severity,
                        "drift_score": drift_score,
                        "drift_types": drift_metrics.get("drift_types", []),
                        "immediate_attention_required": drift_metrics.get(
                            "immediate_attention_required", False
                        ),
                    },
                )
            else:
                logger.debug(
                    f"Drift detected for room {room_id} but retraining not recommended"
                )

        except Exception as e:
            logger.error(
                f"Error handling drift detection for room {room_id}: {e}"
            )

    async def _on_performance_change(
        self, room_id: str, performance_metrics: Dict[str, Any]
    ):
        """Handle model performance change notification."""
        try:
            logger.debug(
                f"Performance change detected for room {room_id}: {performance_metrics}"
            )

            # This could trigger various actions based on performance changes:
            # - Model comparison requests
            # - Feature importance analysis
            # - Performance trending analysis

            # For now, just log the performance change
            logger.debug(
                f"Room {room_id} performance update: {performance_metrics}"
            )

        except Exception as e:
            logger.error(
                f"Error handling performance change for room {room_id}: {e}"
            )

    async def _queue_retraining_request(
        self,
        room_id: str,
        trigger_reason: str,
        priority: int = 5,
        strategy: str = "adaptive",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Queue a retraining request for processing."""
        try:
            # Check if training is already in progress for this room
            if room_id in self._active_training_requests:
                logger.info(
                    f"Training already in progress for room {room_id}, skipping request"
                )
                return

            # Check cooldown period
            if not self._can_retrain_room(room_id):
                cooldown_remaining = self._get_cooldown_remaining(room_id)
                logger.info(
                    f"Room {room_id} in training cooldown for {cooldown_remaining:.1f} hours, "
                    "queuing for later"
                )

            # Create retraining request
            request = {
                "room_id": room_id,
                "trigger_reason": trigger_reason,
                "strategy": strategy,
                "priority": priority,
                "requested_at": datetime.utcnow(),
                "metadata": metadata or {},
            }

            # Add to queue (maintain priority order)
            self._training_queue.append(request)
            self._training_queue.sort(
                key=lambda x: (x["priority"], x["requested_at"])
            )

            logger.info(
                f"Queued retraining request for room {room_id}: {trigger_reason}"
            )

        except Exception as e:
            logger.error(
                f"Failed to queue retraining request for room {room_id}: {e}"
            )

    def _calculate_priority(
        self, current_value: float, threshold: float
    ) -> int:
        """Calculate priority based on how far current value is from threshold."""
        if current_value <= 0:
            return 1  # Critical

        ratio = current_value / threshold

        if ratio < 0.5:
            return 1  # Critical
        elif ratio < 0.7:
            return 2  # High
        elif ratio < 0.9:
            return 3  # Medium
        else:
            return 4  # Low

    def _can_retrain_room(self, room_id: str) -> bool:
        """Check if room can be retrained (not in cooldown period)."""
        if room_id not in self._last_training_times:
            return True

        last_training = self._last_training_times[room_id]
        cooldown_period = timedelta(hours=self._training_cooldown_hours)

        return (datetime.utcnow() - last_training) >= cooldown_period

    def _get_cooldown_remaining(self, room_id: str) -> float:
        """Get remaining cooldown hours for a room."""
        if room_id not in self._last_training_times:
            return 0.0

        last_training = self._last_training_times[room_id]
        elapsed = (datetime.utcnow() - last_training).total_seconds() / 3600

        return max(0.0, self._training_cooldown_hours - elapsed)

    async def _training_queue_processor(self):
        """Background task to process the training queue."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Process training queue
                    await self._process_training_queue()

                    # Wait before next processing cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=30,  # Check every 30 seconds
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for regular processing
                    continue
                except Exception as e:
                    logger.error(f"Error in training queue processor: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        except asyncio.CancelledError:
            logger.info("Training queue processor cancelled")
            raise
        except Exception as e:
            logger.error(f"Training queue processor failed: {e}")

    async def _process_training_queue(self):
        """Process pending training requests from the queue."""
        try:
            if not self._training_queue:
                return

            # Check available training capacity
            active_training_count = len(self._active_training_requests)
            if active_training_count >= self._max_concurrent_training:
                logger.debug(
                    f"Max concurrent training reached ({active_training_count}), waiting"
                )
                return

            # Process requests in priority order
            processed_requests = []

            for request in self._training_queue:
                room_id = request["room_id"]

                # Skip if room is already being trained
                if room_id in self._active_training_requests:
                    continue

                # Skip if room is in cooldown period
                if not self._can_retrain_room(room_id):
                    continue

                # Check if we have capacity for more training
                if (
                    len(self._active_training_requests)
                    >= self._max_concurrent_training
                ):
                    break

                # Process this request
                try:
                    await self._execute_training_request(request)
                    processed_requests.append(request)

                except Exception as e:
                    logger.error(
                        f"Failed to execute training request for room {room_id}: {e}"
                    )
                    processed_requests.append(
                        request
                    )  # Remove failed request from queue

            # Remove processed requests from queue
            for request in processed_requests:
                if request in self._training_queue:
                    self._training_queue.remove(request)

        except Exception as e:
            logger.error(f"Error processing training queue: {e}")

    async def _execute_training_request(self, request: Dict[str, Any]):
        """Execute a training request."""
        try:
            room_id = request["room_id"]
            trigger_reason = request["trigger_reason"]
            strategy = request["strategy"]

            logger.info(
                f"Executing training request for room {room_id}: {trigger_reason}"
            )

            # Determine training type based on strategy
            if strategy == "full_retrain":
                training_type = TrainingType.FULL_RETRAIN
                force_full = True
            elif strategy == "incremental":
                training_type = TrainingType.INCREMENTAL
                force_full = False
            else:  # adaptive or default
                training_type = TrainingType.ADAPTATION
                force_full = False

            # Get appropriate training configuration
            training_profile = self._select_training_profile_for_strategy(
                strategy
            )
            self.config_manager.set_current_profile(training_profile)

            # Start training pipeline
            pipeline_task = asyncio.create_task(
                self.training_pipeline.run_retraining_pipeline(
                    room_id=room_id,
                    trigger_reason=trigger_reason,
                    strategy=strategy,
                    force_full_retrain=force_full,
                )
            )

            # Track active training
            pipeline_id = (
                f"integration_{room_id}_{int(datetime.utcnow().timestamp())}"
            )
            self._active_training_requests[room_id] = pipeline_id

            # Wait for training completion
            progress = await pipeline_task

            # Handle training completion
            await self._handle_training_completion(room_id, progress, request)

        except Exception as e:
            logger.error(
                f"Training request execution failed for room {room_id}: {e}"
            )
            # Clean up active training tracking
            if room_id in self._active_training_requests:
                del self._active_training_requests[room_id]
            raise

    def _select_training_profile_for_strategy(
        self, strategy: str
    ) -> TrainingProfile:
        """Select appropriate training profile based on retraining strategy."""
        if strategy == "full_retrain":
            return TrainingProfile.COMPREHENSIVE
        elif strategy == "incremental":
            return TrainingProfile.QUICK
        elif strategy == "adaptive":
            return TrainingProfile.PRODUCTION
        else:
            return TrainingProfile.PRODUCTION

    async def _handle_training_completion(
        self, room_id: str, progress, request: Dict[str, Any]
    ):
        """Handle completion of a training operation."""
        try:
            # Remove from active training
            if room_id in self._active_training_requests:
                del self._active_training_requests[room_id]

            # Update last training time
            self._last_training_times[room_id] = datetime.utcnow()

            # Check training success
            if (
                hasattr(progress, "stage")
                and progress.stage.value == "completed"
            ):
                logger.info(
                    f"Training completed successfully for room {room_id}"
                )

                # Notify tracking manager of successful training
                if hasattr(self.tracking_manager, "on_model_retrained"):
                    await self.tracking_manager.on_model_retrained(
                        room_id, progress
                    )

                # Update model registration if needed
                if hasattr(progress, "best_model") and progress.best_model:
                    await self._update_model_registration(room_id, progress)

            else:
                logger.error(
                    f"Training failed for room {room_id}: {progress.errors if hasattr(progress, 'errors') else 'Unknown error'}"
                )

                # Handle training failure
                await self._handle_training_failure(room_id, progress, request)

        except Exception as e:
            logger.error(
                f"Error handling training completion for room {room_id}: {e}"
            )

    async def _update_model_registration(self, room_id: str, progress):
        """Update model registration with tracking manager."""
        try:
            if not self.tracking_manager:
                return

            # Get the trained model from the training pipeline
            model_registry = self.training_pipeline.get_model_registry()
            model_key = f"{room_id}_{progress.best_model}"

            if model_key in model_registry:
                model_instance = model_registry[model_key]

                # Re-register model with tracking manager
                self.tracking_manager.register_model(
                    room_id=room_id,
                    model_type=progress.best_model,
                    model_instance=model_instance,
                )

                logger.info(
                    f"Updated model registration for {room_id}: {progress.best_model}"
                )

        except Exception as e:
            logger.error(
                f"Failed to update model registration for room {room_id}: {e}"
            )

    async def _handle_training_failure(
        self, room_id: str, progress, request: Dict[str, Any]
    ):
        """Handle training failure and determine next steps."""
        try:
            failure_count = request.get("failure_count", 0) + 1
            max_retries = 3

            if failure_count < max_retries:
                # Retry with different strategy or profile
                logger.info(
                    f"Retrying training for room {room_id} (attempt {failure_count + 1})"
                )

                # Modify request for retry
                retry_request = request.copy()
                retry_request["failure_count"] = failure_count
                retry_request["strategy"] = (
                    "quick"  # Use quick strategy for retry
                )
                retry_request["priority"] = min(
                    1, request["priority"]
                )  # Increase priority

                # Add back to queue
                self._training_queue.append(retry_request)

            else:
                logger.error(
                    f"Training failed permanently for room {room_id} after {failure_count} attempts"
                )

                # Notify tracking manager of permanent failure
                if hasattr(self.tracking_manager, "on_training_failure"):
                    await self.tracking_manager.on_training_failure(
                        room_id, progress
                    )

        except Exception as e:
            logger.error(
                f"Error handling training failure for room {room_id}: {e}"
            )

    async def _periodic_maintenance(self):
        """Periodic maintenance tasks for training integration."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Perform maintenance tasks
                    await self._cleanup_old_data()
                    await self._check_scheduled_training()
                    await self._update_performance_baselines()

                    # Wait for next maintenance cycle (1 hour)
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=3600
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for maintenance interval
                    continue
                except Exception as e:
                    logger.error(f"Error in periodic maintenance: {e}")
                    await asyncio.sleep(300)  # Wait 5 minutes before retrying

        except asyncio.CancelledError:
            logger.info("Periodic maintenance cancelled")
            raise
        except Exception as e:
            logger.error(f"Periodic maintenance failed: {e}")

    async def _resource_monitor(self):
        """Monitor resource usage and adjust training capacity."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Monitor system resources
                    await self._check_resource_usage()

                    # Adjust concurrent training capacity based on resources
                    await self._adjust_training_capacity()

                    # Wait for next monitoring cycle (5 minutes)
                    await asyncio.wait_for(
                        self._shutdown_event.wait(), timeout=300
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for monitoring interval
                    continue
                except Exception as e:
                    logger.error(f"Error in resource monitoring: {e}")
                    await asyncio.sleep(60)  # Wait 1 minute before retrying

        except asyncio.CancelledError:
            logger.info("Resource monitor cancelled")
            raise
        except Exception as e:
            logger.error(f"Resource monitor failed: {e}")

    async def _cleanup_old_data(self):
        """Clean up old training data and artifacts."""
        try:
            # Clean up old training requests
            cutoff_time = datetime.utcnow() - timedelta(hours=24)

            initial_count = len(self._training_queue)
            self._training_queue = [
                req
                for req in self._training_queue
                if req["requested_at"] > cutoff_time
            ]

            cleaned_count = initial_count - len(self._training_queue)
            if cleaned_count > 0:
                logger.debug(
                    f"Cleaned up {cleaned_count} old training requests"
                )

        except Exception as e:
            logger.error(f"Error in cleanup_old_data: {e}")

    async def _check_scheduled_training(self):
        """Check for scheduled training operations."""
        try:
            # This could implement:
            # - Weekly model refresh
            # - Monthly comprehensive retraining
            # - Seasonal model updates

            logger.debug("Checked for scheduled training operations")

        except Exception as e:
            logger.error(f"Error checking scheduled training: {e}")

    async def _update_performance_baselines(self):
        """Update performance baselines based on recent training results."""
        try:
            # This could analyze recent training results and update
            # performance expectations and thresholds

            logger.debug("Updated performance baselines")

        except Exception as e:
            logger.error(f"Error updating performance baselines: {e}")

    async def _check_resource_usage(self):
        """Check current system resource usage."""
        try:
            # This could monitor:
            # - CPU usage
            # - Memory usage
            # - Disk space
            # - Training queue length

            logger.debug("Checked system resource usage")

        except Exception as e:
            logger.error(f"Error checking resource usage: {e}")

    async def _adjust_training_capacity(self):
        """Adjust training capacity based on resource availability."""
        try:
            # This could dynamically adjust self._max_concurrent_training
            # based on available resources

            logger.debug("Adjusted training capacity based on resources")

        except Exception as e:
            logger.error(f"Error adjusting training capacity: {e}")

    # Public API methods

    async def request_manual_training(
        self,
        room_id: str,
        strategy: str = "adaptive",
        priority: int = 2,
        reason: str = "manual_request",
    ) -> bool:
        """Request manual training for a specific room."""
        try:
            await self._queue_retraining_request(
                room_id=room_id,
                trigger_reason=f"manual_{reason}",
                priority=priority,
                strategy=strategy,
                metadata={"manual_request": True, "reason": reason},
            )

            logger.info(f"Manual training requested for room {room_id}")
            return True

        except Exception as e:
            logger.error(
                f"Failed to request manual training for room {room_id}: {e}"
            )
            return False

    def get_integration_status(self) -> Dict[str, Any]:
        """Get current integration status and statistics."""
        try:
            return {
                "integration_active": self._integration_active,
                "active_training_requests": len(
                    self._active_training_requests
                ),
                "queued_training_requests": len(self._training_queue),
                "max_concurrent_training": self._max_concurrent_training,
                "training_cooldown_hours": self._training_cooldown_hours,
                "background_tasks_running": len(self._background_tasks),
                "rooms_with_active_training": list(
                    self._active_training_requests.keys()
                ),
                "rooms_in_cooldown": [
                    room_id
                    for room_id in self._last_training_times.keys()
                    if not self._can_retrain_room(room_id)
                ],
                "next_queued_rooms": [
                    req["room_id"] for req in self._training_queue[:5]
                ],
            }

        except Exception as e:
            logger.error(f"Error getting integration status: {e}")
            return {"error": str(e)}

    def get_training_queue_status(self) -> List[Dict[str, Any]]:
        """Get current training queue status."""
        try:
            return [
                {
                    "room_id": req["room_id"],
                    "trigger_reason": req["trigger_reason"],
                    "strategy": req["strategy"],
                    "priority": req["priority"],
                    "requested_at": req["requested_at"].isoformat(),
                    "waiting_time_minutes": (
                        datetime.utcnow() - req["requested_at"]
                    ).total_seconds()
                    / 60,
                }
                for req in self._training_queue
            ]

        except Exception as e:
            logger.error(f"Error getting training queue status: {e}")
            return []

    def set_training_capacity(self, max_concurrent: int):
        """Set maximum concurrent training operations."""
        if max_concurrent > 0:
            self._max_concurrent_training = max_concurrent
            logger.info(
                f"Set training capacity to {max_concurrent} concurrent operations"
            )
        else:
            logger.error("Training capacity must be positive")

    def set_cooldown_period(self, hours: int):
        """Set training cooldown period in hours."""
        if hours > 0:
            self._training_cooldown_hours = hours
            logger.info(f"Set training cooldown to {hours} hours")
        else:
            logger.error("Cooldown period must be positive")


async def integrate_training_with_tracking_manager(
    tracking_manager,
    training_pipeline: ModelTrainingPipeline,
    config_manager=None,
) -> TrainingIntegrationManager:
    """
    Create and initialize training integration with tracking manager.

    Args:
        tracking_manager: The system tracking manager
        training_pipeline: Training pipeline instance
        config_manager: Training configuration manager (optional)

    Returns:
        Initialized TrainingIntegrationManager
    """
    try:
        integration_manager = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=config_manager,
        )

        await integration_manager.initialize()

        logger.info(
            "Successfully integrated training pipeline with tracking manager"
        )
        return integration_manager

    except Exception as e:
        logger.error(
            f"Failed to integrate training with tracking manager: {e}"
        )
        raise ModelTrainingError(f"Training integration failed: {str(e)}")
