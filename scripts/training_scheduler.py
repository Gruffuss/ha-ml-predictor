#!/usr/bin/env python3
"""
Training Scheduler for Automated Model Training Operations.

This script provides scheduled training operations that can be run as a service
or cron job to automatically maintain model performance and freshness.
"""

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.adaptation.tracking_manager import TrackingManager, TrackingConfig
from src.core.config import get_config
from src.data.storage.database import get_database_manager
from src.features.engineering import FeatureEngineeringEngine
from src.features.store import FeatureStore
from src.models.training_config import TrainingProfile, get_training_config_manager
from src.models.training_integration import integrate_training_with_tracking_manager
from src.models.training_pipeline import ModelTrainingPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class TrainingScheduler:
    """
    Automated training scheduler that monitors system performance and
    triggers training operations based on predefined schedules and conditions.
    """

    def __init__(self, config_path: Optional[Path] = None):
        """
        Initialize the training scheduler.

        Args:
            config_path: Path to scheduler configuration file
        """
        self.config_path = config_path or Path("./config/training_scheduler.json")
        self.config = self._load_scheduler_config()

        # System components (initialized in setup)
        self.tracking_manager: Optional[TrackingManager] = None
        self.training_pipeline: Optional[ModelTrainingPipeline] = None
        self.integration_manager = None

        # Scheduler state
        self._running = False
        self._shutdown_event = asyncio.Event()
        self._background_tasks: List[asyncio.Task] = []

        # Schedule tracking
        self._last_daily_check = datetime.utcnow().date()
        self._last_weekly_check = self._get_current_week()
        self._last_monthly_check = datetime.utcnow().replace(day=1).date()

        # Statistics
        self._scheduler_stats = {
            "start_time": None,
            "total_scheduled_trainings": 0,
            "successful_trainings": 0,
            "failed_trainings": 0,
            "accuracy_triggered_trainings": 0,
            "drift_triggered_trainings": 0,
            "scheduled_trainings": 0,
        }

        logger.info("Initialized TrainingScheduler")

    def _load_scheduler_config(self) -> Dict:
        """Load scheduler configuration from file."""
        default_config = {
            # Schedule configuration
            "daily_check_enabled": True,
            "daily_check_hour": 2,  # 2 AM
            "weekly_training_enabled": True,
            "weekly_training_day": 6,  # Sunday
            "weekly_training_hour": 3,  # 3 AM
            "monthly_comprehensive_enabled": True,
            "monthly_training_day": 1,  # First day of month
            "monthly_training_hour": 1,  # 1 AM
            # Accuracy monitoring
            "accuracy_monitoring_enabled": True,
            "accuracy_check_interval_minutes": 60,
            "accuracy_threshold": 0.65,  # 65%
            "error_threshold_minutes": 30.0,
            "minimum_predictions_for_check": 50,
            # Drift monitoring
            "drift_monitoring_enabled": True,
            "drift_check_interval_hours": 6,
            "drift_threshold": 0.25,
            "drift_retraining_enabled": True,
            # Training profiles for different schedules
            "daily_training_profile": "quick",
            "weekly_training_profile": "production",
            "monthly_training_profile": "comprehensive",
            "accuracy_training_profile": "production",
            "drift_training_profile": "adaptive",
            # Resource management
            "max_concurrent_trainings": 2,
            "training_timeout_minutes": 120,
            "cooldown_hours": 6,
            "skip_training_if_recent": True,
            # Notification settings
            "notify_on_training_start": True,
            "notify_on_training_complete": True,
            "notify_on_training_failure": True,
            "notify_on_accuracy_issues": True,
            "notify_on_drift_detected": True,
            # Health checks
            "health_check_enabled": True,
            "health_check_interval_minutes": 30,
            "restart_on_health_failure": False,
        }

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
                    logger.info(
                        f"Loaded scheduler configuration from {self.config_path}"
                    )
            except Exception as e:
                logger.warning(f"Failed to load scheduler config: {e}, using defaults")
        else:
            logger.info("No scheduler config file found, using default configuration")
            # Save default configuration
            self._save_scheduler_config(default_config)

        return default_config

    def _save_scheduler_config(self, config: Dict):
        """Save scheduler configuration to file."""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.config_path, "w") as f:
                json.dump(config, f, indent=2)
            logger.info(f"Saved scheduler configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save scheduler config: {e}")

    async def setup(self):
        """Set up the training scheduler with all required components."""
        try:
            logger.info("Setting up training scheduler...")

            # Load system configuration
            system_config = get_config()

            # Initialize database manager
            database_manager = await get_database_manager()

            # Initialize feature engineering
            feature_engine = FeatureEngineeringEngine(
                config=system_config,
                database_manager=database_manager,
            )

            # Initialize feature store
            feature_store = FeatureStore(
                database_manager=database_manager,
                feature_engine=feature_engine,
            )

            # Initialize tracking manager
            tracking_config = TrackingConfig(
                accuracy_monitoring_enabled=self.config["accuracy_monitoring_enabled"],
                drift_detection_enabled=self.config["drift_monitoring_enabled"],
                adaptive_retraining_enabled=True,
                retraining_accuracy_threshold=self.config["accuracy_threshold"] * 100,
                retraining_error_threshold=self.config["error_threshold_minutes"],
                retraining_check_interval_hours=1,  # Check frequently for scheduler
            )

            self.tracking_manager = TrackingManager(
                config=tracking_config,
                database_manager=database_manager,
                feature_engineering_engine=feature_engine,
            )

            await self.tracking_manager.initialize()

            # Initialize training pipeline
            config_manager = get_training_config_manager()
            training_config = config_manager.get_training_config(
                TrainingProfile.PRODUCTION
            )

            self.training_pipeline = ModelTrainingPipeline(
                config=training_config,
                feature_engineering_engine=feature_engine,
                feature_store=feature_store,
                database_manager=database_manager,
                tracking_manager=self.tracking_manager,
            )

            # Initialize training integration
            self.integration_manager = await integrate_training_with_tracking_manager(
                tracking_manager=self.tracking_manager,
                training_pipeline=self.training_pipeline,
                config_manager=config_manager,
            )

            # Set training capacity
            self.integration_manager.set_training_capacity(
                self.config["max_concurrent_trainings"]
            )

            logger.info("Training scheduler setup completed successfully")

        except Exception as e:
            logger.error(f"Failed to set up training scheduler: {e}")
            raise

    async def start(self):
        """Start the training scheduler."""
        try:
            if self._running:
                logger.warning("Training scheduler is already running")
                return

            self._running = True
            self._scheduler_stats["start_time"] = datetime.utcnow()

            # Setup signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)

            # Start background tasks
            await self._start_background_tasks()

            logger.info("Training scheduler started successfully")

            # Main scheduler loop
            await self._main_scheduler_loop()

        except Exception as e:
            logger.error(f"Training scheduler failed: {e}")
            raise
        finally:
            await self.shutdown()

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        logger.info(f"Received signal {signum}, initiating graceful shutdown...")
        self._shutdown_event.set()

    async def _start_background_tasks(self):
        """Start background monitoring tasks."""
        try:
            # Health monitoring task
            if self.config["health_check_enabled"]:
                health_task = asyncio.create_task(self._health_monitor())
                self._background_tasks.append(health_task)

            # Accuracy monitoring task
            if self.config["accuracy_monitoring_enabled"]:
                accuracy_task = asyncio.create_task(self._accuracy_monitor())
                self._background_tasks.append(accuracy_task)

            # Drift monitoring task
            if self.config["drift_monitoring_enabled"]:
                drift_task = asyncio.create_task(self._drift_monitor())
                self._background_tasks.append(drift_task)

            logger.info(f"Started {len(self._background_tasks)} background tasks")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise

    async def _main_scheduler_loop(self):
        """Main scheduler loop for periodic training operations."""
        try:
            logger.info("Starting main scheduler loop")

            while not self._shutdown_event.is_set():
                try:
                    # Check for scheduled training operations
                    await self._check_scheduled_operations()

                    # Wait for next check (every 10 minutes)
                    await asyncio.wait_for(self._shutdown_event.wait(), timeout=600)

                except asyncio.TimeoutError:
                    # Expected timeout for regular checks
                    continue
                except Exception as e:
                    logger.error(f"Error in scheduler loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Main scheduler loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Main scheduler loop failed: {e}")
            raise

    async def _check_scheduled_operations(self):
        """Check for and execute scheduled training operations."""
        try:
            current_time = datetime.utcnow()
            current_date = current_time.date()
            current_hour = current_time.hour
            current_weekday = current_time.weekday()

            # Daily check
            if (
                self.config["daily_check_enabled"]
                and current_date > self._last_daily_check
                and current_hour >= self.config["daily_check_hour"]
            ):

                await self._run_daily_check()
                self._last_daily_check = current_date

            # Weekly training
            if (
                self.config["weekly_training_enabled"]
                and current_weekday == self.config["weekly_training_day"]
                and current_hour >= self.config["weekly_training_hour"]
                and self._get_current_week() > self._last_weekly_check
            ):

                await self._run_weekly_training()
                self._last_weekly_check = self._get_current_week()

            # Monthly comprehensive training
            if (
                self.config["monthly_comprehensive_enabled"]
                and current_date.day == self.config["monthly_training_day"]
                and current_hour >= self.config["monthly_training_hour"]
                and current_date > self._last_monthly_check
            ):

                await self._run_monthly_training()
                self._last_monthly_check = current_date

        except Exception as e:
            logger.error(f"Error checking scheduled operations: {e}")

    def _get_current_week(self) -> int:
        """Get current week number."""
        return datetime.utcnow().isocalendar()[1]

    async def _run_daily_check(self):
        """Run daily system check and light training if needed."""
        try:
            logger.info("Running daily system check...")

            # Check system health
            if self.tracking_manager:
                status = await self.tracking_manager.get_tracking_status()

                # Check for any critical issues
                if status.get("accuracy_tracker"):
                    active_alerts = len(
                        status["accuracy_tracker"].get("active_alerts", [])
                    )
                    if active_alerts > 0:
                        logger.warning(f"Found {active_alerts} active accuracy alerts")

                # Check for rooms with poor performance
                rooms_needing_attention = await self._identify_rooms_needing_training()

                if rooms_needing_attention:
                    logger.info(
                        f"Daily check identified {len(rooms_needing_attention)} rooms needing training"
                    )

                    # Run quick training for problematic rooms
                    profile = TrainingProfile(self.config["daily_training_profile"])
                    await self._train_specific_rooms(
                        rooms_needing_attention, profile, "daily_check"
                    )
                else:
                    logger.info(
                        "Daily check: all rooms performing within acceptable ranges"
                    )

            logger.info("Daily system check completed")

        except Exception as e:
            logger.error(f"Daily check failed: {e}")

    async def _run_weekly_training(self):
        """Run weekly training for model maintenance."""
        try:
            logger.info("Running weekly training maintenance...")

            # Run incremental training for all rooms
            profile = TrainingProfile(self.config["weekly_training_profile"])

            # Get all rooms
            system_config = get_config()
            room_ids = list(system_config.rooms.keys())

            # Filter rooms that haven't been trained recently
            rooms_to_train = await self._filter_rooms_by_cooldown(room_ids)

            if rooms_to_train:
                logger.info(f"Running weekly training for {len(rooms_to_train)} rooms")
                success_count, failure_count = await self._train_rooms_with_profile(
                    rooms_to_train, profile, "weekly_maintenance"
                )

                self._scheduler_stats["scheduled_trainings"] += len(rooms_to_train)
                self._scheduler_stats["successful_trainings"] += success_count
                self._scheduler_stats["failed_trainings"] += failure_count

                logger.info(
                    f"Weekly training completed: {success_count} successful, {failure_count} failed"
                )
            else:
                logger.info("Weekly training: all rooms recently trained, skipping")

        except Exception as e:
            logger.error(f"Weekly training failed: {e}")

    async def _run_monthly_training(self):
        """Run monthly comprehensive training."""
        try:
            logger.info("Running monthly comprehensive training...")

            # Run comprehensive training for all rooms
            profile = TrainingProfile(self.config["monthly_training_profile"])

            # Get all rooms
            system_config = get_config()
            room_ids = list(system_config.rooms.keys())

            logger.info(f"Running comprehensive training for {len(room_ids)} rooms")
            success_count, failure_count = await self._train_rooms_with_profile(
                room_ids, profile, "monthly_comprehensive"
            )

            self._scheduler_stats["scheduled_trainings"] += len(room_ids)
            self._scheduler_stats["successful_trainings"] += success_count
            self._scheduler_stats["failed_trainings"] += failure_count

            logger.info(
                f"Monthly comprehensive training completed: {success_count} successful, {failure_count} failed"
            )

        except Exception as e:
            logger.error(f"Monthly comprehensive training failed: {e}")

    async def _identify_rooms_needing_training(self) -> List[str]:
        """Identify rooms that need training based on performance metrics."""
        try:
            rooms_needing_training = []

            if not self.tracking_manager:
                return rooms_needing_training

            # Get all rooms
            system_config = get_config()
            room_ids = list(system_config.rooms.keys())

            for room_id in room_ids:
                try:
                    # Get room accuracy metrics (mock implementation)
                    # In reality, this would use tracking manager to get real metrics
                    accuracy_metrics = await self.tracking_manager.get_accuracy_metrics(
                        room_id
                    )

                    if accuracy_metrics:
                        accuracy_rate = accuracy_metrics.get("accuracy_rate", 1.0)
                        error_minutes = accuracy_metrics.get(
                            "average_error_minutes", 0.0
                        )

                        needs_training = (
                            accuracy_rate < self.config["accuracy_threshold"]
                            or error_minutes > self.config["error_threshold_minutes"]
                        )

                        if needs_training:
                            rooms_needing_training.append(room_id)
                            logger.info(
                                f"Room {room_id} needs training: "
                                f"accuracy={accuracy_rate:.3f}, error={error_minutes:.1f}min"
                            )

                except Exception as e:
                    logger.warning(f"Failed to check metrics for room {room_id}: {e}")

            return rooms_needing_training

        except Exception as e:
            logger.error(f"Failed to identify rooms needing training: {e}")
            return []

    async def _filter_rooms_by_cooldown(self, room_ids: List[str]) -> List[str]:
        """Filter rooms based on training cooldown period."""
        if not self.config["skip_training_if_recent"]:
            return room_ids

        filtered_rooms = []

        if self.integration_manager:
            status = self.integration_manager.get_integration_status()
            rooms_in_cooldown = set(status.get("rooms_in_cooldown", []))

            for room_id in room_ids:
                if room_id not in rooms_in_cooldown:
                    filtered_rooms.append(room_id)
                else:
                    logger.debug(f"Skipping room {room_id} due to training cooldown")
        else:
            filtered_rooms = room_ids

        return filtered_rooms

    async def _train_specific_rooms(
        self, room_ids: List[str], profile: TrainingProfile, reason: str
    ):
        """Train specific rooms with given profile."""
        try:
            if not self.training_pipeline:
                logger.error("Training pipeline not available")
                return 0, len(room_ids)

            # Set training profile
            config_manager = get_training_config_manager()
            config_manager.set_current_profile(profile)

            success_count = 0
            failure_count = 0

            for room_id in room_ids:
                try:
                    progress = await self.training_pipeline.run_retraining_pipeline(
                        room_id=room_id,
                        trigger_reason=f"scheduler_{reason}",
                        strategy="adaptive",
                    )

                    if (
                        hasattr(progress, "stage")
                        and progress.stage.value == "completed"
                    ):
                        success_count += 1
                        logger.info(f"Successfully trained room {room_id} ({reason})")
                    else:
                        failure_count += 1
                        logger.error(f"Failed to train room {room_id} ({reason})")

                except Exception as e:
                    failure_count += 1
                    logger.error(f"Error training room {room_id} ({reason}): {e}")

            return success_count, failure_count

        except Exception as e:
            logger.error(f"Failed to train specific rooms: {e}")
            return 0, len(room_ids)

    async def _train_rooms_with_profile(
        self, room_ids: List[str], profile: TrainingProfile, reason: str
    ) -> tuple:
        """Train rooms using a specific profile."""
        try:
            if not self.training_pipeline:
                logger.error("Training pipeline not available")
                return 0, len(room_ids)

            # Use training pipeline's batch training capability
            training_results = await self.training_pipeline.run_initial_training(
                room_ids=room_ids,
                months_of_data=3,  # Use last 3 months for scheduled training
            )

            successful_rooms = [
                room_id
                for room_id, progress in training_results.items()
                if hasattr(progress, "stage") and progress.stage.value == "completed"
            ]

            failed_rooms = [
                room_id for room_id in room_ids if room_id not in successful_rooms
            ]

            return len(successful_rooms), len(failed_rooms)

        except Exception as e:
            logger.error(f"Failed to train rooms with profile: {e}")
            return 0, len(room_ids)

    async def _health_monitor(self):
        """Monitor system health and restart if needed."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check system health
                    health_ok = await self._check_system_health()

                    if not health_ok and self.config["restart_on_health_failure"]:
                        logger.error(
                            "System health check failed, initiating restart..."
                        )
                        # In a production environment, this could trigger a service restart
                        # For now, just log the issue

                    # Wait for next health check
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config["health_check_interval_minutes"] * 60,
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in health monitor: {e}")
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("Health monitor cancelled")
        except Exception as e:
            logger.error(f"Health monitor failed: {e}")

    async def _accuracy_monitor(self):
        """Monitor accuracy and trigger training if needed."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check accuracy and trigger training if needed
                    await self._check_accuracy_and_trigger_training()

                    # Wait for next accuracy check
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config["accuracy_check_interval_minutes"] * 60,
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in accuracy monitor: {e}")
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("Accuracy monitor cancelled")
        except Exception as e:
            logger.error(f"Accuracy monitor failed: {e}")

    async def _drift_monitor(self):
        """Monitor drift and trigger training if needed."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Check drift and trigger training if needed
                    await self._check_drift_and_trigger_training()

                    # Wait for next drift check
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.config["drift_check_interval_hours"] * 3600,
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in drift monitor: {e}")
                    await asyncio.sleep(300)

        except asyncio.CancelledError:
            logger.info("Drift monitor cancelled")
        except Exception as e:
            logger.error(f"Drift monitor failed: {e}")

    async def _check_system_health(self) -> bool:
        """Check overall system health."""
        try:
            if not self.tracking_manager:
                return False

            status = await self.tracking_manager.get_tracking_status()

            # Check tracking manager health
            if not status.get("tracking_active", False):
                logger.warning("Tracking manager is not active")
                return False

            # Check database connectivity
            if self.tracking_manager.database_manager:
                # In a real implementation, check database health
                pass

            return True

        except Exception as e:
            logger.error(f"System health check failed: {e}")
            return False

    async def _check_accuracy_and_trigger_training(self):
        """Check accuracy metrics and trigger training if thresholds exceeded."""
        try:
            rooms_needing_training = await self._identify_rooms_needing_training()

            if rooms_needing_training:
                logger.info(
                    f"Accuracy monitor triggered training for {len(rooms_needing_training)} rooms"
                )

                profile = TrainingProfile(self.config["accuracy_training_profile"])
                success_count, failure_count = await self._train_specific_rooms(
                    rooms_needing_training, profile, "accuracy_degradation"
                )

                self._scheduler_stats["accuracy_triggered_trainings"] += len(
                    rooms_needing_training
                )
                self._scheduler_stats["successful_trainings"] += success_count
                self._scheduler_stats["failed_trainings"] += failure_count

        except Exception as e:
            logger.error(f"Error in accuracy-based training trigger: {e}")

    async def _check_drift_and_trigger_training(self):
        """Check drift metrics and trigger training if needed."""
        try:
            if not self.tracking_manager:
                return

            # Get all rooms and check for drift
            system_config = get_config()
            room_ids = list(system_config.rooms.keys())

            rooms_with_drift = []

            for room_id in room_ids:
                try:
                    # Check drift for this room
                    drift_metrics = await self.tracking_manager.check_drift(room_id)

                    if drift_metrics and drift_metrics.retraining_recommended:
                        rooms_with_drift.append(room_id)
                        logger.info(
                            f"Drift detected in room {room_id}, retraining recommended"
                        )

                except Exception as e:
                    logger.warning(f"Failed to check drift for room {room_id}: {e}")

            if rooms_with_drift:
                logger.info(
                    f"Drift monitor triggered training for {len(rooms_with_drift)} rooms"
                )

                profile = TrainingProfile(self.config["drift_training_profile"])
                success_count, failure_count = await self._train_specific_rooms(
                    rooms_with_drift, profile, "concept_drift"
                )

                self._scheduler_stats["drift_triggered_trainings"] += len(
                    rooms_with_drift
                )
                self._scheduler_stats["successful_trainings"] += success_count
                self._scheduler_stats["failed_trainings"] += failure_count

        except Exception as e:
            logger.error(f"Error in drift-based training trigger: {e}")

    async def shutdown(self):
        """Shutdown the training scheduler gracefully."""
        try:
            logger.info("Shutting down training scheduler...")

            self._running = False
            self._shutdown_event.set()

            # Cancel background tasks
            if self._background_tasks:
                for task in self._background_tasks:
                    task.cancel()
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
                self._background_tasks.clear()

            # Shutdown integration manager
            if self.integration_manager:
                await self.integration_manager.shutdown()

            # Shutdown tracking manager
            if self.tracking_manager:
                await self.tracking_manager.stop_tracking()

            # Print final statistics
            uptime = (
                (
                    datetime.utcnow() - self._scheduler_stats["start_time"]
                ).total_seconds()
                / 3600
                if self._scheduler_stats["start_time"]
                else 0
            )
            logger.info(
                f"Training scheduler shutdown completed after {uptime:.1f} hours"
            )
            logger.info(f"Statistics: {self._scheduler_stats}")

        except Exception as e:
            logger.error(f"Error during scheduler shutdown: {e}")

    def get_scheduler_status(self) -> Dict:
        """Get current scheduler status."""
        return {
            "running": self._running,
            "start_time": (
                self._scheduler_stats["start_time"].isoformat()
                if self._scheduler_stats["start_time"]
                else None
            ),
            "uptime_hours": (
                (
                    (
                        datetime.utcnow() - self._scheduler_stats["start_time"]
                    ).total_seconds()
                    / 3600
                )
                if self._scheduler_stats["start_time"]
                else 0
            ),
            "background_tasks": len(self._background_tasks),
            "last_daily_check": self._last_daily_check.isoformat(),
            "last_weekly_check": self._last_weekly_check,
            "last_monthly_check": self._last_monthly_check.isoformat(),
            "statistics": self._scheduler_stats.copy(),
            "configuration": self.config.copy(),
        }


async def main():
    """Main function to run the training scheduler."""
    try:
        # Create and setup scheduler
        scheduler = TrainingScheduler()
        await scheduler.setup()

        logger.info("Starting automated training scheduler...")
        logger.info("Press Ctrl+C to shutdown gracefully")

        # Start scheduler
        await scheduler.start()

    except KeyboardInterrupt:
        logger.info("Scheduler interrupted by user")
    except Exception as e:
        logger.error(f"Scheduler failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
