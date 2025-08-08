"""
Automated Validation Runner

This module provides automated validation workflows for the occupancy prediction system.
It orchestrates scheduled validation runs, batch processing of historical data,
continuous monitoring workflows, and automated testing scenarios.

The runner serves as the primary interface for executing validation workflows
in production environments, CI/CD pipelines, and automated testing scenarios.
"""

import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from prediction_validation_framework import (
    PredictionValidationFramework,
    ValidationFrameworkStatus,
    ValidationPriority,
    ValidationReport,
    ValidationTask,
    create_validation_framework,
)
import schedule

from src.adaptation.validator import (
    AccuracyLevel,
    AccuracyMetrics,
    PredictionValidator,
    ValidationRecord,
    ValidationStatus,
)
from src.core.constants import ModelType
from src.models.base.predictor import PredictionResult
from src.utils.logger import get_logger


class ValidationRunType(Enum):
    """Types of validation runs."""

    CONTINUOUS = "continuous"
    SCHEDULED = "scheduled"
    ON_DEMAND = "on_demand"
    BATCH_HISTORICAL = "batch_historical"
    REGRESSION_TEST = "regression_test"
    PERFORMANCE_BENCHMARK = "performance_benchmark"
    CALIBRATION_CHECK = "calibration_check"


class ValidationRunStatus(Enum):
    """Status of validation runs."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


@dataclass
class ValidationRunConfig:
    """Configuration for a validation run."""

    run_id: str
    run_type: ValidationRunType
    rooms_to_validate: List[str]
    time_window_hours: int
    accuracy_threshold_minutes: int = 15
    confidence_threshold: float = 0.8
    generate_reports: bool = True
    send_notifications: bool = True
    parallel_processing: bool = True
    max_concurrent_rooms: int = 5
    timeout_minutes: int = 60
    retry_failed_validations: bool = True
    max_retries: int = 3
    artifacts_directory: Optional[Path] = None
    custom_parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationRunResult:
    """Results of a validation run."""

    run_id: str
    run_type: ValidationRunType
    status: ValidationRunStatus
    started_at: datetime
    completed_at: Optional[datetime] = None
    duration_seconds: Optional[float] = None

    # Results summary
    rooms_processed: int = 0
    rooms_succeeded: int = 0
    rooms_failed: int = 0
    total_predictions_validated: int = 0

    # Metrics summary
    overall_accuracy_percentage: float = 0.0
    overall_calibration_score: float = 0.0
    average_confidence_score: float = 0.0

    # Detailed results per room
    room_results: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    warnings: List[Dict[str, Any]] = field(default_factory=list)

    # Artifacts
    reports_generated: List[str] = field(default_factory=list)
    artifacts_saved: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary."""
        return {
            "run_id": self.run_id,
            "run_type": self.run_type.value,
            "status": self.status.value,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat() if self.completed_at else None
            ),
            "duration_seconds": self.duration_seconds,
            "rooms_processed": self.rooms_processed,
            "rooms_succeeded": self.rooms_succeeded,
            "rooms_failed": self.rooms_failed,
            "total_predictions_validated": self.total_predictions_validated,
            "overall_accuracy_percentage": self.overall_accuracy_percentage,
            "overall_calibration_score": self.overall_calibration_score,
            "average_confidence_score": self.average_confidence_score,
            "room_results": self.room_results,
            "errors": self.errors,
            "warnings": self.warnings,
            "reports_generated": self.reports_generated,
            "artifacts_saved": self.artifacts_saved,
        }


class AutomatedValidationRunner:
    """
    Automated validation runner for orchestrating validation workflows.

    This runner provides:
    - Scheduled validation runs
    - Batch historical data validation
    - Continuous monitoring workflows
    - Performance benchmark testing
    - Regression testing capabilities
    - Automated report generation
    """

    def __init__(
        self,
        framework: Optional[PredictionValidationFramework] = None,
        default_config: Optional[ValidationRunConfig] = None,
        artifacts_directory: Optional[Path] = None,
        enable_scheduling: bool = True,
        max_concurrent_runs: int = 3,
    ):
        """Initialize the automated validation runner."""
        self.logger = get_logger(__name__)

        # Core components
        self.framework = framework or create_validation_framework(
            artifacts_directory=artifacts_directory
        )
        self.artifacts_directory = artifacts_directory or Path(
            "validation_runner_artifacts"
        )
        self.artifacts_directory.mkdir(parents=True, exist_ok=True)

        # Configuration
        self.default_config = default_config
        self.enable_scheduling = enable_scheduling
        self.max_concurrent_runs = max_concurrent_runs

        # State tracking
        self.active_runs: Dict[str, ValidationRunResult] = {}
        self.completed_runs: List[ValidationRunResult] = []
        self.scheduled_jobs = []
        self.shutdown_requested = False

        # Concurrency control
        self.run_semaphore = asyncio.Semaphore(max_concurrent_runs)
        self.executor = ThreadPoolExecutor(max_workers=max_concurrent_runs * 2)

        # Notification handlers
        self.run_start_handlers: List[Callable] = []
        self.run_complete_handlers: List[Callable] = []
        self.error_handlers: List[Callable] = []

        self.logger.info("Automated Validation Runner initialized")

    async def start(self) -> None:
        """Start the automated validation runner."""
        try:
            # Start the validation framework
            await self.framework.start()

            # Setup default scheduled jobs if enabled
            if self.enable_scheduling:
                await self._setup_default_scheduled_jobs()

            self.logger.info("Automated Validation Runner started")

        except Exception as e:
            self.logger.error(f"Failed to start Automated Validation Runner: {e}")
            raise

    async def stop(self) -> None:
        """Stop the automated validation runner."""
        try:
            self.shutdown_requested = True

            # Cancel all scheduled jobs
            schedule.clear()
            self.scheduled_jobs.clear()

            # Wait for active runs to complete (with timeout)
            if self.active_runs:
                self.logger.info(
                    f"Waiting for {len(self.active_runs)} active runs to complete..."
                )
                await self._wait_for_active_runs_completion(timeout_seconds=300)

            # Shutdown framework
            await self.framework.stop()

            # Shutdown executor
            self.executor.shutdown(wait=True)

            self.logger.info("Automated Validation Runner stopped")

        except Exception as e:
            self.logger.error(f"Error stopping Automated Validation Runner: {e}")
            raise

    async def run_validation(
        self, config: ValidationRunConfig, wait_for_completion: bool = True
    ) -> ValidationRunResult:
        """Run a validation workflow with the specified configuration."""
        run_result = ValidationRunResult(
            run_id=config.run_id,
            run_type=config.run_type,
            status=ValidationRunStatus.PENDING,
            started_at=datetime.now(),
        )

        try:
            # Check if runner is shutting down
            if self.shutdown_requested:
                run_result.status = ValidationRunStatus.CANCELLED
                return run_result

            # Acquire run slot
            async with self.run_semaphore:
                self.active_runs[config.run_id] = run_result

                # Notify start handlers
                await self._notify_run_start(config, run_result)

                # Execute validation run
                if wait_for_completion:
                    await self._execute_validation_run(config, run_result)
                else:
                    # Start as background task
                    asyncio.create_task(
                        self._execute_validation_run(config, run_result)
                    )

                return run_result

        except Exception as e:
            run_result.status = ValidationRunStatus.FAILED
            run_result.errors.append(
                {
                    "error_type": "run_execution_error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            await self._notify_run_error(config, run_result, e)
            return run_result

        finally:
            if config.run_id in self.active_runs:
                # Move to completed runs
                self.completed_runs.append(run_result)
                del self.active_runs[config.run_id]

                # Notify completion handlers
                await self._notify_run_complete(config, run_result)

    async def run_continuous_validation(
        self,
        rooms: List[str],
        monitoring_interval_minutes: int = 15,
        report_interval_hours: int = 4,
    ) -> None:
        """Run continuous validation monitoring."""
        config = ValidationRunConfig(
            run_id=f"continuous_{int(datetime.now().timestamp())}",
            run_type=ValidationRunType.CONTINUOUS,
            rooms_to_validate=rooms,
            time_window_hours=24,
            custom_parameters={
                "monitoring_interval_minutes": monitoring_interval_minutes,
                "report_interval_hours": report_interval_hours,
            },
        )

        await self._run_continuous_validation_loop(config)

    async def run_batch_historical_validation(
        self,
        rooms: List[str],
        start_date: datetime,
        end_date: datetime,
        batch_size_hours: int = 24,
    ) -> ValidationRunResult:
        """Run batch validation on historical data."""
        config = ValidationRunConfig(
            run_id=f"batch_historical_{int(datetime.now().timestamp())}",
            run_type=ValidationRunType.BATCH_HISTORICAL,
            rooms_to_validate=rooms,
            time_window_hours=batch_size_hours,
            custom_parameters={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "batch_size_hours": batch_size_hours,
            },
        )

        return await self.run_validation(config, wait_for_completion=True)

    async def run_performance_benchmark(
        self,
        rooms: List[str],
        benchmark_duration_hours: int = 24,
        target_accuracy_percentage: float = 85.0,
        target_calibration_score: float = 0.8,
    ) -> ValidationRunResult:
        """Run performance benchmark validation."""
        config = ValidationRunConfig(
            run_id=f"performance_benchmark_{int(datetime.now().timestamp())}",
            run_type=ValidationRunType.PERFORMANCE_BENCHMARK,
            rooms_to_validate=rooms,
            time_window_hours=benchmark_duration_hours,
            custom_parameters={
                "target_accuracy_percentage": target_accuracy_percentage,
                "target_calibration_score": target_calibration_score,
                "benchmark_duration_hours": benchmark_duration_hours,
            },
        )

        return await self.run_validation(config, wait_for_completion=True)

    async def run_regression_test_suite(
        self, rooms: List[str], baseline_results_path: Optional[Path] = None
    ) -> ValidationRunResult:
        """Run regression test suite against baseline results."""
        config = ValidationRunConfig(
            run_id=f"regression_test_{int(datetime.now().timestamp())}",
            run_type=ValidationRunType.REGRESSION_TEST,
            rooms_to_validate=rooms,
            time_window_hours=48,
            custom_parameters={
                "baseline_results_path": (
                    str(baseline_results_path) if baseline_results_path else None
                ),
                "regression_tolerance_percentage": 5.0,
            },
        )

        return await self.run_validation(config, wait_for_completion=True)

    async def schedule_validation_run(
        self, config: ValidationRunConfig, schedule_expression: str
    ) -> str:
        """Schedule a recurring validation run."""
        if not self.enable_scheduling:
            raise ValueError("Scheduling is disabled")

        job_id = f"scheduled_{config.run_id}_{int(datetime.now().timestamp())}"

        def run_scheduled_validation():
            asyncio.create_task(self.run_validation(config, wait_for_completion=False))

        # Parse schedule expression and create job
        if schedule_expression == "daily":
            schedule.every().day.at("02:00").do(run_scheduled_validation).tag(job_id)
        elif schedule_expression == "hourly":
            schedule.every().hour.do(run_scheduled_validation).tag(job_id)
        elif schedule_expression.startswith("every_"):
            # Parse "every_X_minutes" format
            parts = schedule_expression.split("_")
            if len(parts) == 3 and parts[2] == "minutes":
                interval = int(parts[1])
                schedule.every(interval).minutes.do(run_scheduled_validation).tag(
                    job_id
                )

        self.scheduled_jobs.append(job_id)
        self.logger.info(
            f"Scheduled validation run {job_id} with schedule: {schedule_expression}"
        )

        return job_id

    def cancel_scheduled_run(self, job_id: str) -> bool:
        """Cancel a scheduled validation run."""
        try:
            schedule.clear(job_id)
            if job_id in self.scheduled_jobs:
                self.scheduled_jobs.remove(job_id)
            self.logger.info(f"Cancelled scheduled validation run {job_id}")
            return True
        except Exception as e:
            self.logger.error(f"Error cancelling scheduled run {job_id}: {e}")
            return False

    async def get_run_status(self, run_id: str) -> Optional[ValidationRunResult]:
        """Get the status of a validation run."""
        # Check active runs
        if run_id in self.active_runs:
            return self.active_runs[run_id]

        # Check completed runs
        for run_result in self.completed_runs:
            if run_result.run_id == run_id:
                return run_result

        return None

    async def get_runner_statistics(self) -> Dict[str, Any]:
        """Get statistics about the validation runner."""
        current_time = datetime.now()

        # Calculate statistics
        total_runs = len(self.completed_runs)
        successful_runs = len(
            [
                r
                for r in self.completed_runs
                if r.status == ValidationRunStatus.COMPLETED
            ]
        )
        failed_runs = len(
            [r for r in self.completed_runs if r.status == ValidationRunStatus.FAILED]
        )

        average_duration = 0.0
        if self.completed_runs:
            durations = [
                r.duration_seconds for r in self.completed_runs if r.duration_seconds
            ]
            if durations:
                average_duration = sum(durations) / len(durations)

        return {
            "runner_status": (
                "active" if not self.shutdown_requested else "shutting_down"
            ),
            "active_runs": len(self.active_runs),
            "scheduled_jobs": len(self.scheduled_jobs),
            "total_completed_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "success_rate_percentage": (successful_runs / max(1, total_runs)) * 100,
            "average_run_duration_seconds": average_duration,
            "framework_status": (await self.framework.get_system_health_status())[
                "framework_status"
            ],
        }

    # Private Methods

    async def _execute_validation_run(
        self, config: ValidationRunConfig, run_result: ValidationRunResult
    ) -> None:
        """Execute the core validation run logic."""
        try:
            run_result.status = ValidationRunStatus.RUNNING

            # Execute based on run type
            if config.run_type == ValidationRunType.BATCH_HISTORICAL:
                await self._execute_batch_historical_run(config, run_result)
            elif config.run_type == ValidationRunType.PERFORMANCE_BENCHMARK:
                await self._execute_performance_benchmark_run(config, run_result)
            elif config.run_type == ValidationRunType.REGRESSION_TEST:
                await self._execute_regression_test_run(config, run_result)
            elif config.run_type == ValidationRunType.CALIBRATION_CHECK:
                await self._execute_calibration_check_run(config, run_result)
            else:
                await self._execute_standard_validation_run(config, run_result)

            # Generate final summary
            await self._generate_run_summary(config, run_result)

            # Mark as completed
            run_result.status = ValidationRunStatus.COMPLETED
            run_result.completed_at = datetime.now()
            run_result.duration_seconds = (
                run_result.completed_at - run_result.started_at
            ).total_seconds()

        except asyncio.TimeoutError:
            run_result.status = ValidationRunStatus.TIMEOUT
            run_result.errors.append(
                {
                    "error_type": "timeout_error",
                    "message": f"Run exceeded timeout of {config.timeout_minutes} minutes",
                    "timestamp": datetime.now().isoformat(),
                }
            )

        except Exception as e:
            run_result.status = ValidationRunStatus.FAILED
            run_result.errors.append(
                {
                    "error_type": "execution_error",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat(),
                }
            )
            self.logger.error(f"Validation run {config.run_id} failed: {e}")

    async def _execute_standard_validation_run(
        self, config: ValidationRunConfig, run_result: ValidationRunResult
    ) -> None:
        """Execute a standard validation run."""
        # Register rooms with framework
        for room_id in config.rooms_to_validate:
            await self.framework.register_room(room_id)

        # Process rooms in parallel if configured
        if config.parallel_processing:
            await self._process_rooms_parallel(config, run_result)
        else:
            await self._process_rooms_sequential(config, run_result)

        # Generate reports if requested
        if config.generate_reports:
            await self._generate_validation_reports(config, run_result)

    async def _process_rooms_parallel(
        self, config: ValidationRunConfig, run_result: ValidationRunResult
    ) -> None:
        """Process rooms in parallel."""
        semaphore = asyncio.Semaphore(config.max_concurrent_rooms)

        async def process_room(room_id: str):
            async with semaphore:
                return await self._process_single_room(room_id, config, run_result)

        # Create tasks for all rooms
        tasks = [process_room(room_id) for room_id in config.rooms_to_validate]

        # Wait for all tasks to complete
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            room_id = config.rooms_to_validate[i]
            if isinstance(result, Exception):
                run_result.rooms_failed += 1
                run_result.errors.append(
                    {
                        "error_type": "room_processing_error",
                        "room_id": room_id,
                        "message": str(result),
                        "timestamp": datetime.now().isoformat(),
                    }
                )
            else:
                run_result.rooms_succeeded += 1

            run_result.rooms_processed += 1

    async def _process_single_room(
        self, room_id: str, config: ValidationRunConfig, run_result: ValidationRunResult
    ) -> Dict[str, Any]:
        """Process validation for a single room."""
        try:
            # Generate validation report for the room
            report = await self.framework.generate_validation_report(
                room_id=room_id,
                time_window_hours=config.time_window_hours,
                include_recommendations=True,
            )

            # Extract metrics
            room_metrics = {
                "accuracy_percentage": report.accuracy_metrics.get(
                    "accuracy_percentage", 0
                ),
                "calibration_score": report.calibration_metrics.get(
                    "calibration_score", 0
                ),
                "total_predictions": report.accuracy_metrics.get(
                    "total_predictions", 0
                ),
                "data_quality_score": report.data_quality_score,
                "confidence_score": report.confidence_score,
                "recommendations_count": len(report.recommendations),
                "report_generated": True,
            }

            # Store results
            run_result.room_results[room_id] = room_metrics
            run_result.total_predictions_validated += room_metrics["total_predictions"]

            return room_metrics

        except Exception as e:
            error_result = {
                "error": str(e),
                "report_generated": False,
                "accuracy_percentage": 0,
                "calibration_score": 0,
                "total_predictions": 0,
            }
            run_result.room_results[room_id] = error_result
            raise e

    async def _generate_run_summary(
        self, config: ValidationRunConfig, run_result: ValidationRunResult
    ) -> None:
        """Generate summary statistics for the validation run."""
        if run_result.room_results:
            # Calculate overall metrics
            accuracies = [
                r.get("accuracy_percentage", 0)
                for r in run_result.room_results.values()
            ]
            calibrations = [
                r.get("calibration_score", 0) for r in run_result.room_results.values()
            ]
            confidences = [
                r.get("confidence_score", 0) for r in run_result.room_results.values()
            ]

            run_result.overall_accuracy_percentage = (
                np.mean(accuracies) if accuracies else 0.0
            )
            run_result.overall_calibration_score = (
                np.mean(calibrations) if calibrations else 0.0
            )
            run_result.average_confidence_score = (
                np.mean(confidences) if confidences else 0.0
            )

        # Save run results
        await self._save_run_results(config, run_result)

    async def _save_run_results(
        self, config: ValidationRunConfig, run_result: ValidationRunResult
    ) -> None:
        """Save validation run results to artifacts."""
        try:
            results_filename = f"validation_run_{config.run_id}_{run_result.started_at.strftime('%Y%m%d_%H%M%S')}.json"
            results_path = self.artifacts_directory / results_filename

            with open(results_path, "w") as f:
                json.dump(run_result.to_dict(), f, indent=2)

            run_result.artifacts_saved.append(str(results_path))
            self.logger.info(f"Saved validation run results to {results_path}")

        except Exception as e:
            self.logger.error(f"Error saving validation run results: {e}")

    async def _setup_default_scheduled_jobs(self) -> None:
        """Setup default scheduled validation jobs."""
        try:
            # Daily comprehensive validation
            daily_config = ValidationRunConfig(
                run_id="daily_validation",
                run_type=ValidationRunType.SCHEDULED,
                rooms_to_validate=[],  # Will be populated from framework
                time_window_hours=24,
            )
            await self.schedule_validation_run(daily_config, "daily")

            # Hourly calibration checks
            hourly_config = ValidationRunConfig(
                run_id="hourly_calibration",
                run_type=ValidationRunType.CALIBRATION_CHECK,
                rooms_to_validate=[],
                time_window_hours=4,
            )
            await self.schedule_validation_run(hourly_config, "hourly")

        except Exception as e:
            self.logger.error(f"Error setting up default scheduled jobs: {e}")


# CLI Interface


def create_cli_parser() -> argparse.ArgumentParser:
    """Create command-line interface parser."""
    parser = argparse.ArgumentParser(description="Automated Validation Runner")

    parser.add_argument(
        "command",
        choices=["run", "schedule", "status", "stop"],
        help="Command to execute",
    )

    parser.add_argument(
        "--run-type",
        choices=[t.value for t in ValidationRunType],
        default="on_demand",
        help="Type of validation run",
    )

    parser.add_argument("--rooms", nargs="+", default=[], help="Rooms to validate")

    parser.add_argument(
        "--time-window", type=int, default=24, help="Time window in hours"
    )

    parser.add_argument(
        "--accuracy-threshold",
        type=int,
        default=15,
        help="Accuracy threshold in minutes",
    )

    parser.add_argument(
        "--confidence-threshold", type=float, default=0.8, help="Confidence threshold"
    )

    parser.add_argument("--artifacts-dir", type=Path, help="Artifacts directory")

    parser.add_argument(
        "--parallel", action="store_true", help="Enable parallel processing"
    )

    parser.add_argument(
        "--schedule", help="Schedule expression (daily, hourly, every_X_minutes)"
    )

    parser.add_argument("--run-id", help="Specific run ID to query")

    return parser


async def main() -> None:
    """Main CLI entry point."""
    parser = create_cli_parser()
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = get_logger(__name__)

    try:
        # Create and start runner
        runner = AutomatedValidationRunner(
            artifacts_directory=args.artifacts_dir, enable_scheduling=True
        )
        await runner.start()

        try:
            if args.command == "run":
                # Create run configuration
                config = ValidationRunConfig(
                    run_id=f"cli_run_{int(datetime.now().timestamp())}",
                    run_type=ValidationRunType(args.run_type),
                    rooms_to_validate=args.rooms,
                    time_window_hours=args.time_window,
                    accuracy_threshold_minutes=args.accuracy_threshold,
                    confidence_threshold=args.confidence_threshold,
                    parallel_processing=args.parallel,
                    artifacts_directory=args.artifacts_dir,
                )

                # Run validation
                result = await runner.run_validation(config, wait_for_completion=True)

                # Print results
                print(f"Validation run completed: {result.status.value}")
                print(f"Rooms processed: {result.rooms_processed}")
                print(f"Overall accuracy: {result.overall_accuracy_percentage:.2f}%")
                print(f"Overall calibration: {result.overall_calibration_score:.3f}")

            elif args.command == "schedule":
                if not args.schedule:
                    print("Schedule expression required for schedule command")
                    sys.exit(1)

                config = ValidationRunConfig(
                    run_id=f"scheduled_run_{int(datetime.now().timestamp())}",
                    run_type=ValidationRunType.SCHEDULED,
                    rooms_to_validate=args.rooms,
                    time_window_hours=args.time_window,
                )

                job_id = await runner.schedule_validation_run(config, args.schedule)
                print(f"Scheduled validation job: {job_id}")

            elif args.command == "status":
                if args.run_id:
                    # Get specific run status
                    status = await runner.get_run_status(args.run_id)
                    if status:
                        print(json.dumps(status.to_dict(), indent=2))
                    else:
                        print(f"Run not found: {args.run_id}")
                else:
                    # Get runner statistics
                    stats = await runner.get_runner_statistics()
                    print(json.dumps(stats, indent=2))

            elif args.command == "stop":
                print("Stopping validation runner...")
                await runner.stop()
                print("Validation runner stopped")

        finally:
            if args.command != "stop":
                await runner.stop()

    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"CLI error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
