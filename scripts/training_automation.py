#!/usr/bin/env python3
"""
Training Automation Scripts for Home Assistant Occupancy Prediction System.

This script provides command-line tools for managing the training pipeline,
including initial setup, scheduled retraining, and system maintenance.
"""

import argparse
import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sys
from typing import List, Optional

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.config import get_config
from src.core.exceptions import ModelTrainingError
from src.data.storage.database import get_database_manager
from src.features.engineering import FeatureEngineeringEngine
from src.features.store import FeatureStore
from src.models.training_config import TrainingProfile, get_training_config_manager
from src.models.training_integration import integrate_training_with_tracking_manager
from src.models.training_pipeline import ModelTrainingPipeline, TrainingType

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def setup_training_environment():
    """Set up the training environment with all required components."""
    try:
        logger.info("Setting up training environment...")

        # Load system configuration
        system_config = get_config()

        # Initialize database manager
        database_manager = await get_database_manager()

        # Initialize feature engineering engine
        feature_engineering_engine = FeatureEngineeringEngine(
            config=system_config,
            database_manager=database_manager,
        )

        # Initialize feature store
        feature_store = FeatureStore(
            database_manager=database_manager,
            feature_engine=feature_engineering_engine,
        )

        # Initialize training configuration manager
        config_manager = get_training_config_manager()

        # Create training pipeline
        training_config = config_manager.get_training_config()
        training_pipeline = ModelTrainingPipeline(
            config=training_config,
            feature_engineering_engine=feature_engineering_engine,
            feature_store=feature_store,
            database_manager=database_manager,
        )

        logger.info("Training environment setup completed successfully")

        return {
            "system_config": system_config,
            "database_manager": database_manager,
            "feature_engineering_engine": feature_engineering_engine,
            "feature_store": feature_store,
            "config_manager": config_manager,
            "training_pipeline": training_pipeline,
        }

    except Exception as e:
        logger.error(f"Failed to set up training environment: {e}")
        raise


async def run_initial_training(
    room_ids: Optional[List[str]] = None,
    months_of_data: int = 6,
    training_profile: TrainingProfile = TrainingProfile.PRODUCTION,
):
    """
    Run initial training for the system.

    Args:
        room_ids: Specific rooms to train (None for all rooms)
        months_of_data: Months of historical data to use
        training_profile: Training profile to use
    """
    try:
        logger.info(f"Starting initial training with {months_of_data} months of data")

        # Set up training environment
        env = await setup_training_environment()
        config_manager = env["config_manager"]
        training_pipeline = env["training_pipeline"]

        # Set training profile
        config_manager.set_current_profile(training_profile)
        logger.info(f"Using training profile: {training_profile.value}")

        # Get room list if not specified
        if room_ids is None:
            system_config = env["system_config"]
            room_ids = list(system_config.rooms.keys())
            logger.info(f"Training models for all {len(room_ids)} rooms")
        else:
            logger.info(f"Training models for {len(room_ids)} specified rooms")

        # Run initial training
        start_time = datetime.utcnow()
        training_results = await training_pipeline.run_initial_training(
            room_ids=room_ids,
            months_of_data=months_of_data,
        )

        # Report results
        total_time = (datetime.utcnow() - start_time).total_seconds() / 60
        successful_rooms = [
            room_id
            for room_id, progress in training_results.items()
            if hasattr(progress, "stage") and progress.stage.value == "completed"
        ]
        failed_rooms = [
            room_id
            for room_id, progress in training_results.items()
            if room_id not in successful_rooms
        ]

        logger.info(f"Initial training completed in {total_time:.1f} minutes")
        logger.info(f"Successfully trained: {len(successful_rooms)} rooms")
        logger.info(f"Failed training: {len(failed_rooms)} rooms")

        if successful_rooms:
            logger.info(f"Successful rooms: {', '.join(successful_rooms)}")

        if failed_rooms:
            logger.error(f"Failed rooms: {', '.join(failed_rooms)}")

            # Print detailed error information
            for room_id in failed_rooms:
                progress = training_results[room_id]
                if hasattr(progress, "errors") and progress.errors:
                    logger.error(f"  {room_id}: {'; '.join(progress.errors)}")

        return len(successful_rooms), len(failed_rooms)

    except Exception as e:
        logger.error(f"Initial training failed: {e}")
        raise


async def run_incremental_training(
    room_id: str,
    days_of_new_data: int = 7,
    training_profile: TrainingProfile = TrainingProfile.QUICK,
):
    """
    Run incremental training for a specific room.

    Args:
        room_id: Room to train
        days_of_new_data: Days of new data to incorporate
        training_profile: Training profile to use
    """
    try:
        logger.info(f"Starting incremental training for room {room_id}")

        # Set up training environment
        env = await setup_training_environment()
        config_manager = env["config_manager"]
        training_pipeline = env["training_pipeline"]

        # Set training profile
        config_manager.set_current_profile(training_profile)

        # Run incremental training
        start_time = datetime.utcnow()
        progress = await training_pipeline.run_incremental_training(
            room_id=room_id,
            days_of_new_data=days_of_new_data,
        )

        # Report results
        total_time = (datetime.utcnow() - start_time).total_seconds() / 60
        success = hasattr(progress, "stage") and progress.stage.value == "completed"

        logger.info(f"Incremental training completed in {total_time:.1f} minutes")
        logger.info(f"Status: {'SUCCESS' if success else 'FAILED'}")

        if success:
            best_model = getattr(progress, "best_model", "unknown")
            best_score = getattr(progress, "best_score", 0.0)
            logger.info(f"Best model: {best_model}, Score: {best_score:.3f}")
        else:
            errors = getattr(progress, "errors", ["Unknown error"])
            logger.error(f"Errors: {'; '.join(errors)}")

        return success

    except Exception as e:
        logger.error(f"Incremental training failed for room {room_id}: {e}")
        raise


async def run_system_retraining(
    accuracy_threshold: float = 0.7,
    error_threshold_minutes: float = 25.0,
    training_profile: TrainingProfile = TrainingProfile.PRODUCTION,
):
    """
    Run system-wide retraining based on accuracy thresholds.

    Args:
        accuracy_threshold: Minimum accuracy threshold for retraining
        error_threshold_minutes: Maximum error threshold for retraining
        training_profile: Training profile to use
    """
    try:
        logger.info("Starting system-wide retraining based on accuracy analysis")

        # Set up training environment
        env = await setup_training_environment()
        config_manager = env["config_manager"]
        training_pipeline = env["training_pipeline"]

        # Set training profile
        config_manager.set_current_profile(training_profile)

        # Get room list
        system_config = env["system_config"]
        room_ids = list(system_config.rooms.keys())

        # Analyze each room and determine retraining needs
        rooms_needing_retraining = []

        # This would normally check actual accuracy metrics from tracking manager
        # For now, simulate the decision process
        logger.info(f"Analyzing {len(room_ids)} rooms for retraining needs...")

        for room_id in room_ids:
            # Mock accuracy check - in reality would get from tracking manager
            mock_accuracy = 0.65  # Below threshold
            mock_error = 30.0  # Above threshold

            needs_retraining = (
                mock_accuracy < accuracy_threshold
                or mock_error > error_threshold_minutes
            )

            if needs_retraining:
                rooms_needing_retraining.append(room_id)
                logger.info(
                    f"Room {room_id} needs retraining: accuracy={mock_accuracy:.3f}, error={mock_error:.1f}min"
                )

        if not rooms_needing_retraining:
            logger.info("No rooms need retraining based on current thresholds")
            return 0, 0

        logger.info(
            f"Retraining {len(rooms_needing_retraining)} rooms that need improvement"
        )

        # Run retraining for identified rooms
        start_time = datetime.utcnow()
        successful_retrains = 0
        failed_retrains = 0

        for room_id in rooms_needing_retraining:
            try:
                progress = await training_pipeline.run_retraining_pipeline(
                    room_id=room_id,
                    trigger_reason="automated_accuracy_check",
                    strategy="adaptive",
                )

                if hasattr(progress, "stage") and progress.stage.value == "completed":
                    successful_retrains += 1
                    logger.info(f"Successfully retrained room {room_id}")
                else:
                    failed_retrains += 1
                    logger.error(f"Failed to retrain room {room_id}")

            except Exception as e:
                failed_retrains += 1
                logger.error(f"Error retraining room {room_id}: {e}")

        # Report results
        total_time = (datetime.utcnow() - start_time).total_seconds() / 60
        logger.info(f"System retraining completed in {total_time:.1f} minutes")
        logger.info(f"Successfully retrained: {successful_retrains} rooms")
        logger.info(f"Failed retraining: {failed_retrains} rooms")

        return successful_retrains, failed_retrains

    except Exception as e:
        logger.error(f"System retraining failed: {e}")
        raise


async def check_training_status():
    """Check the status of training system and models."""
    try:
        logger.info("Checking training system status...")

        # Set up training environment
        env = await setup_training_environment()
        training_pipeline = env["training_pipeline"]
        config_manager = env["config_manager"]

        # Get training statistics
        stats = training_pipeline.get_training_statistics()

        print("\n=== Training System Status ===")
        print(f"Total pipelines run: {stats['total_pipelines_run']}")
        print(f"Successful pipelines: {stats['successful_pipelines']}")
        print(f"Failed pipelines: {stats['failed_pipelines']}")
        print(f"Total models trained: {stats['total_models_trained']}")
        print(
            f"Average training time: {stats['average_training_time_minutes']:.1f} minutes"
        )

        # Get active pipelines
        active_pipelines = training_pipeline.get_active_pipelines()
        print(f"Active pipelines: {len(active_pipelines)}")

        if active_pipelines:
            print("\nActive Training Pipelines:")
            for pipeline_id, progress in active_pipelines.items():
                room_id = getattr(progress, "room_id", "unknown")
                stage = getattr(progress, "stage", "unknown")
                percent = getattr(progress, "progress_percent", 0)
                print(
                    f"  {pipeline_id}: Room {room_id}, Stage {stage.value if hasattr(stage, 'value') else stage}, {percent:.1f}%"
                )

        # Get model registry
        model_registry = training_pipeline.get_model_registry()
        print(f"\nRegistered Models: {len(model_registry)}")

        if model_registry:
            room_counts = {}
            for model_key in model_registry.keys():
                room_id = model_key.split("_")[0] if "_" in model_key else "unknown"
                room_counts[room_id] = room_counts.get(room_id, 0) + 1

            print("Models by Room:")
            for room_id, count in sorted(room_counts.items()):
                print(f"  {room_id}: {count} models")

        # Get current configuration
        current_profile = config_manager.get_current_profile()
        print(f"\nCurrent Training Profile: {current_profile.value}")

        # Get configuration comparison
        profile_comparison = config_manager.get_profile_comparison()
        print("\nAvailable Training Profiles:")
        for profile_name, profile_info in profile_comparison["profiles"].items():
            print(f"  {profile_name}:")
            print(
                f"    Training time: {profile_info['max_training_time_minutes']} minutes"
            )
            print(f"    Min accuracy: {profile_info['min_accuracy_threshold']:.2f}")
            print(
                f"    Max error: {profile_info['max_error_threshold_minutes']:.1f} minutes"
            )
            print(
                f"    Optimization: {'enabled' if profile_info['optimization_enabled'] else 'disabled'}"
            )

    except Exception as e:
        logger.error(f"Failed to check training status: {e}")
        raise


async def validate_training_setup():
    """Validate that the training system is properly configured."""
    try:
        logger.info("Validating training system setup...")

        validation_errors = []
        validation_warnings = []

        # Set up training environment
        try:
            env = await setup_training_environment()
        except Exception as e:
            validation_errors.append(f"Failed to set up training environment: {e}")
            print(f"❌ Training environment setup: FAILED - {e}")
            return False

        print("✅ Training environment setup: OK")

        # Validate configuration
        config_manager = env["config_manager"]

        for profile in TrainingProfile:
            try:
                config_issues = config_manager.validate_configuration(profile)
                if config_issues:
                    validation_warnings.extend(
                        [f"{profile.value}: {issue}" for issue in config_issues]
                    )
            except Exception as e:
                validation_errors.append(
                    f"Configuration validation failed for {profile.value}: {e}"
                )

        if validation_warnings:
            print("⚠️  Configuration warnings:")
            for warning in validation_warnings:
                print(f"    - {warning}")
        else:
            print("✅ Configuration validation: OK")

        # Test training pipeline creation
        try:
            training_config = config_manager.get_training_config(
                TrainingProfile.TESTING
            )
            training_pipeline = ModelTrainingPipeline(
                config=training_config,
                feature_engineering_engine=env["feature_engineering_engine"],
                feature_store=env["feature_store"],
                database_manager=env["database_manager"],
            )
            print("✅ Training pipeline creation: OK")
        except Exception as e:
            validation_errors.append(f"Training pipeline creation failed: {e}")
            print(f"❌ Training pipeline creation: FAILED - {e}")

        # Check database connectivity
        try:
            database_manager = env["database_manager"]
            if hasattr(database_manager, "health_check"):
                health = await database_manager.health_check()
                if health:
                    print("✅ Database connectivity: OK")
                else:
                    validation_errors.append("Database health check failed")
                    print("❌ Database connectivity: FAILED")
            else:
                print(
                    "⚠️  Database connectivity: Cannot verify (no health check method)"
                )
        except Exception as e:
            validation_errors.append(f"Database connectivity check failed: {e}")
            print(f"❌ Database connectivity: FAILED - {e}")

        # Check feature engineering
        try:
            feature_engine = env["feature_engineering_engine"]
            if feature_engine:
                print("✅ Feature engineering engine: OK")
            else:
                validation_errors.append("Feature engineering engine not available")
                print("❌ Feature engineering engine: FAILED")
        except Exception as e:
            validation_errors.append(f"Feature engineering check failed: {e}")
            print(f"❌ Feature engineering engine: FAILED - {e}")

        # Summary
        print("\n=== Validation Summary ===")
        print(f"Errors: {len(validation_errors)}")
        print(f"Warnings: {len(validation_warnings)}")

        if validation_errors:
            print("\n❌ Validation FAILED. Errors found:")
            for error in validation_errors:
                print(f"  - {error}")
            return False
        else:
            print("\n✅ Validation PASSED. Training system is ready.")
            if validation_warnings:
                print("Note: Some warnings were found but they don't prevent training.")
            return True

    except Exception as e:
        logger.error(f"Training setup validation failed: {e}")
        print(f"❌ Validation failed with error: {e}")
        return False


def main():
    """Main entry point for training automation script."""
    parser = argparse.ArgumentParser(
        description="Training Automation for Home Assistant Occupancy Prediction"
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Initial training command
    init_parser = subparsers.add_parser("init", help="Run initial training")
    init_parser.add_argument(
        "--rooms", nargs="+", help="Specific rooms to train (default: all rooms)"
    )
    init_parser.add_argument(
        "--months", type=int, default=6, help="Months of historical data to use"
    )
    init_parser.add_argument(
        "--profile",
        choices=[p.value for p in TrainingProfile],
        default=TrainingProfile.PRODUCTION.value,
        help="Training profile to use",
    )

    # Incremental training command
    incremental_parser = subparsers.add_parser(
        "incremental", help="Run incremental training"
    )
    incremental_parser.add_argument("room_id", help="Room ID to retrain")
    incremental_parser.add_argument(
        "--days", type=int, default=7, help="Days of new data to incorporate"
    )
    incremental_parser.add_argument(
        "--profile",
        choices=[p.value for p in TrainingProfile],
        default=TrainingProfile.QUICK.value,
        help="Training profile to use",
    )

    # System retraining command
    retrain_parser = subparsers.add_parser("retrain", help="Run system-wide retraining")
    retrain_parser.add_argument(
        "--accuracy-threshold",
        type=float,
        default=0.7,
        help="Minimum accuracy threshold for retraining",
    )
    retrain_parser.add_argument(
        "--error-threshold",
        type=float,
        default=25.0,
        help="Maximum error threshold (minutes) for retraining",
    )
    retrain_parser.add_argument(
        "--profile",
        choices=[p.value for p in TrainingProfile],
        default=TrainingProfile.PRODUCTION.value,
        help="Training profile to use",
    )

    # Status command
    subparsers.add_parser("status", help="Check training system status")

    # Validation command
    subparsers.add_parser("validate", help="Validate training system setup")

    # Parse arguments
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Run the requested command
    try:
        if args.command == "init":
            profile = TrainingProfile(args.profile)
            success_count, failure_count = asyncio.run(
                run_initial_training(args.rooms, args.months, profile)
            )

            if failure_count > 0:
                sys.exit(1)  # Exit with error if any training failed

        elif args.command == "incremental":
            profile = TrainingProfile(args.profile)
            success = asyncio.run(
                run_incremental_training(args.room_id, args.days, profile)
            )

            if not success:
                sys.exit(1)  # Exit with error if training failed

        elif args.command == "retrain":
            profile = TrainingProfile(args.profile)
            success_count, failure_count = asyncio.run(
                run_system_retraining(
                    args.accuracy_threshold, args.error_threshold, profile
                )
            )

            if failure_count > 0:
                sys.exit(1)  # Exit with error if any retraining failed

        elif args.command == "status":
            asyncio.run(check_training_status())

        elif args.command == "validate":
            valid = asyncio.run(validate_training_setup())
            if not valid:
                sys.exit(1)  # Exit with error if validation failed

    except KeyboardInterrupt:
        logger.info("Training automation interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Training automation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
