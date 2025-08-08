"""
Central Prediction Validation Framework

This module provides the core orchestration system for comprehensive prediction
validation in the occupancy prediction system. It coordinates accuracy validation,
confidence calibration, statistical analysis, and real-time monitoring workflows.

The framework serves as the central hub that integrates all validation components
and provides unified interfaces for validation management, reporting, and
automated decision making.
"""

import asyncio
import json
import logging
from collections import defaultdict
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

import numpy as np
import pandas as pd

from src.adaptation.validator import AccuracyLevel
from src.adaptation.validator import AccuracyMetrics
from src.adaptation.validator import PredictionValidator
from src.adaptation.validator import ValidationError
from src.adaptation.validator import ValidationRecord
from src.adaptation.validator import ValidationStatus
from src.core.constants import ModelType
from src.models.base.predictor import PredictionResult
from src.utils.logger import get_logger


class ValidationFrameworkStatus(Enum):
    """Status of the validation framework."""

    INITIALIZING = "initializing"
    ACTIVE = "active"
    MONITORING = "monitoring"
    DEGRADED = "degraded"
    SUSPENDED = "suspended"
    ERROR = "error"


class ValidationPriority(Enum):
    """Priority levels for validation tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    BACKGROUND = "background"


@dataclass
class ValidationTask:
    """Represents a validation task in the framework queue."""

    task_id: str
    room_id: str
    task_type: str
    priority: ValidationPriority
    scheduled_time: datetime
    parameters: Dict[str, Any] = field(default_factory=dict)
    retry_count: int = 0
    max_retries: int = 3
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None


@dataclass
class ValidationReport:
    """Comprehensive validation report for a room or system."""

    room_id: str
    report_type: str
    generated_at: datetime
    time_window_hours: int

    # Core metrics
    accuracy_metrics: Dict[str, Any]
    calibration_metrics: Dict[str, Any]
    performance_metrics: Dict[str, Any]

    # Trend analysis
    trend_analysis: Dict[str, Any]
    anomaly_detection: Dict[str, Any]

    # Recommendations
    recommendations: List[Dict[str, Any]]
    action_items: List[Dict[str, Any]]

    # Metadata
    data_quality_score: float
    confidence_score: float
    report_completeness: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary format."""
        return {
            "room_id": self.room_id,
            "report_type": self.report_type,
            "generated_at": self.generated_at.isoformat(),
            "time_window_hours": self.time_window_hours,
            "accuracy_metrics": self.accuracy_metrics,
            "calibration_metrics": self.calibration_metrics,
            "performance_metrics": self.performance_metrics,
            "trend_analysis": self.trend_analysis,
            "anomaly_detection": self.anomaly_detection,
            "recommendations": self.recommendations,
            "action_items": self.action_items,
            "data_quality_score": self.data_quality_score,
            "confidence_score": self.confidence_score,
            "report_completeness": self.report_completeness,
        }


class PredictionValidationFramework:
    """
    Central orchestrator for prediction validation workflows.

    This framework coordinates all validation activities including:
    - Real-time prediction validation
    - Batch validation processing
    - Confidence calibration monitoring
    - Performance trend analysis
    - Automated report generation
    - Alert management
    - Validation task scheduling
    """

    def __init__(
        self,
        accuracy_threshold_minutes: int = 15,
        confidence_threshold: float = 0.8,
        monitoring_interval_seconds: int = 300,  # 5 minutes
        report_generation_interval_hours: int = 24,
        enable_real_time_monitoring: bool = True,
        enable_automated_reports: bool = True,
        validation_history_days: int = 30,
        artifacts_directory: Optional[Path] = None,
    ):
        """Initialize the validation framework."""
        self.logger = get_logger(__name__)

        # Configuration
        self.accuracy_threshold_minutes = accuracy_threshold_minutes
        self.confidence_threshold = confidence_threshold
        self.monitoring_interval_seconds = monitoring_interval_seconds
        self.report_generation_interval_hours = report_generation_interval_hours
        self.enable_real_time_monitoring = enable_real_time_monitoring
        self.enable_automated_reports = enable_automated_reports
        self.validation_history_days = validation_history_days

        # Artifacts directory
        self.artifacts_directory = artifacts_directory or Path("validation_artifacts")
        self.artifacts_directory.mkdir(parents=True, exist_ok=True)

        # Core components
        self.validator = PredictionValidator(
            accuracy_threshold_minutes=accuracy_threshold_minutes,
            confidence_threshold=confidence_threshold,
            track_prediction_intervals=True,
            enable_adaptive_calibration=True,
        )

        # Framework state
        self.status = ValidationFrameworkStatus.INITIALIZING
        self.active_rooms: set[str] = set()
        self.validation_tasks: Dict[str, List[ValidationTask]] = defaultdict(list)
        self.pending_tasks: List[ValidationTask] = []
        self.completed_tasks: List[ValidationTask] = []
        self.failed_tasks: List[ValidationTask] = []

        # Monitoring and reporting
        self.last_monitoring_check = datetime.now()
        self.last_report_generation = datetime.now()
        self.alert_handlers: List[Callable] = []
        self.report_handlers: List[Callable] = []

        # Performance tracking
        self.framework_metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "average_processing_time": 0.0,
            "uptime_seconds": 0,
            "last_restart": datetime.now(),
        }

        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._report_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None

        self.logger.info("Prediction Validation Framework initialized")

    async def start(self) -> None:
        """Start the validation framework and background tasks."""
        try:
            self.status = ValidationFrameworkStatus.ACTIVE

            # Start background tasks
            if self.enable_real_time_monitoring:
                self._monitoring_task = asyncio.create_task(self._monitoring_loop())
                self.logger.info("Started real-time monitoring task")

            if self.enable_automated_reports:
                self._report_task = asyncio.create_task(self._report_generation_loop())
                self.logger.info("Started automated report generation task")

            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
            self.logger.info("Started cleanup task")

            # Initialize validator alert handlers
            await self.validator.register_calibration_alert_handler(
                self._handle_calibration_alert
            )

            self.framework_metrics["last_restart"] = datetime.now()
            self.logger.info("Validation Framework started successfully")

        except Exception as e:
            self.status = ValidationFrameworkStatus.ERROR
            self.logger.error(f"Failed to start Validation Framework: {e}")
            raise

    async def stop(self) -> None:
        """Stop the validation framework and cleanup background tasks."""
        try:
            self.status = ValidationFrameworkStatus.SUSPENDED

            # Cancel background tasks
            for task in [self._monitoring_task, self._report_task, self._cleanup_task]:
                if task and not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

            self.logger.info("Validation Framework stopped")

        except Exception as e:
            self.logger.error(f"Error stopping Validation Framework: {e}")
            raise

    async def register_room(self, room_id: str) -> None:
        """Register a room for validation monitoring."""
        if room_id not in self.active_rooms:
            self.active_rooms.add(room_id)
            self.logger.info(f"Registered room {room_id} for validation monitoring")

            # Initialize validation task queue for room
            if room_id not in self.validation_tasks:
                self.validation_tasks[room_id] = []

            # Schedule initial validation tasks
            await self._schedule_initial_validation_tasks(room_id)

    async def unregister_room(self, room_id: str) -> None:
        """Unregister a room from validation monitoring."""
        if room_id in self.active_rooms:
            self.active_rooms.remove(room_id)
            self.logger.info(f"Unregistered room {room_id} from validation monitoring")

            # Cancel pending tasks for room
            await self._cancel_room_tasks(room_id)

    async def validate_prediction(
        self,
        room_id: str,
        prediction_result: PredictionResult,
        prediction_timestamp: datetime,
    ) -> ValidationRecord:
        """Validate a prediction and integrate with framework tracking."""
        try:
            # Record prediction with validator
            record = ValidationRecord(
                room_id=room_id,
                predicted_time=prediction_result.predicted_time,
                confidence=prediction_result.confidence,
                prediction_interval=getattr(
                    prediction_result, "prediction_interval", None
                ),
                timestamp=prediction_timestamp,
                model_type=getattr(prediction_result, "model_type", ModelType.ENSEMBLE),
                model_version=getattr(prediction_result, "model_version", "unknown"),
                feature_importance=getattr(
                    prediction_result, "feature_importance", None
                ),
            )

            self.validator.record_prediction(room_id, record)

            # Schedule follow-up validation tasks
            await self._schedule_follow_up_tasks(room_id, record)

            self.logger.debug(f"Recorded prediction for room {room_id}")
            return record

        except Exception as e:
            self.logger.error(f"Error validating prediction for room {room_id}: {e}")
            raise ValidationError(f"Prediction validation failed: {e}")

    async def validate_actual_outcome(
        self,
        room_id: str,
        actual_time: datetime,
        event_metadata: Optional[Dict[str, Any]] = None,
    ) -> List[ValidationRecord]:
        """Validate actual outcome against pending predictions."""
        try:
            # Validate with core validator
            validated_records = await self.validator.validate_prediction_async(
                room_id, actual_time
            )

            # Process validation results
            for record in validated_records:
                await self._process_validation_result(record, event_metadata)

            # Schedule analysis tasks if needed
            if len(validated_records) > 0:
                await self._schedule_analysis_tasks(room_id, validated_records)

            self.logger.debug(
                f"Validated {len(validated_records)} predictions for room {room_id}"
            )
            return validated_records

        except Exception as e:
            self.logger.error(
                f"Error validating actual outcome for room {room_id}: {e}"
            )
            raise ValidationError(f"Outcome validation failed: {e}")

    async def generate_validation_report(
        self,
        room_id: str,
        report_type: str = "comprehensive",
        time_window_hours: int = 24,
        include_recommendations: bool = True,
    ) -> ValidationReport:
        """Generate comprehensive validation report for a room."""
        try:
            generated_at = datetime.now()

            # Gather core metrics
            accuracy_metrics = self._get_accuracy_metrics_dict(
                room_id, time_window_hours
            )
            calibration_metrics = self._get_calibration_metrics_dict(
                room_id, time_window_hours
            )
            performance_metrics = self._get_performance_metrics_dict(
                room_id, time_window_hours
            )

            # Perform trend analysis
            trend_analysis = await self._perform_trend_analysis(
                room_id, time_window_hours
            )
            anomaly_detection = await self._perform_anomaly_detection(
                room_id, time_window_hours
            )

            # Generate recommendations
            recommendations = []
            action_items = []
            if include_recommendations:
                recommendations = await self._generate_recommendations(
                    room_id, accuracy_metrics, calibration_metrics, performance_metrics
                )
                action_items = await self._generate_action_items(
                    room_id, recommendations, trend_analysis
                )

            # Calculate report quality scores
            data_quality_score = self._calculate_data_quality_score(
                room_id, time_window_hours
            )
            confidence_score = self._calculate_report_confidence_score(
                accuracy_metrics, calibration_metrics
            )
            report_completeness = self._calculate_report_completeness(
                accuracy_metrics, calibration_metrics, performance_metrics
            )

            # Create report
            report = ValidationReport(
                room_id=room_id,
                report_type=report_type,
                generated_at=generated_at,
                time_window_hours=time_window_hours,
                accuracy_metrics=accuracy_metrics,
                calibration_metrics=calibration_metrics,
                performance_metrics=performance_metrics,
                trend_analysis=trend_analysis,
                anomaly_detection=anomaly_detection,
                recommendations=recommendations,
                action_items=action_items,
                data_quality_score=data_quality_score,
                confidence_score=confidence_score,
                report_completeness=report_completeness,
            )

            # Save report
            await self._save_validation_report(report)

            # Notify report handlers
            for handler in self.report_handlers:
                try:
                    await handler(report)
                except Exception as e:
                    self.logger.error(f"Error in report handler: {e}")

            self.logger.info(f"Generated validation report for room {room_id}")
            return report

        except Exception as e:
            self.logger.error(
                f"Error generating validation report for room {room_id}: {e}"
            )
            raise ValidationError(f"Report generation failed: {e}")

    async def get_system_health_status(self) -> Dict[str, Any]:
        """Get overall system health status."""
        try:
            current_time = datetime.now()
            uptime = (
                current_time - self.framework_metrics["last_restart"]
            ).total_seconds()

            # Calculate system-wide metrics
            total_predictions = sum(
                len(self.validator.get_validation_history(room_id, window_hours=24))
                for room_id in self.active_rooms
            )

            # Overall accuracy across all rooms
            overall_accuracy = self._calculate_system_accuracy()

            # Task processing health
            task_success_rate = self._calculate_task_success_rate()

            # Determine overall health status
            health_status = self._determine_health_status(
                overall_accuracy, task_success_rate, uptime
            )

            return {
                "framework_status": self.status.value,
                "health_status": health_status,
                "uptime_seconds": uptime,
                "active_rooms": len(self.active_rooms),
                "total_predictions_24h": total_predictions,
                "overall_accuracy": overall_accuracy,
                "task_success_rate": task_success_rate,
                "pending_tasks": len(self.pending_tasks),
                "failed_tasks_24h": len(
                    [
                        t
                        for t in self.failed_tasks
                        if (current_time - t.created_at).total_seconds() < 86400
                    ]
                ),
                "last_monitoring_check": self.last_monitoring_check.isoformat(),
                "last_report_generation": self.last_report_generation.isoformat(),
                "framework_metrics": self.framework_metrics.copy(),
            }

        except Exception as e:
            self.logger.error(f"Error getting system health status: {e}")
            return {
                "framework_status": ValidationFrameworkStatus.ERROR.value,
                "health_status": "critical",
                "error": str(e),
            }

    # Background Task Methods

    async def _monitoring_loop(self) -> None:
        """Background monitoring loop."""
        while self.status == ValidationFrameworkStatus.ACTIVE:
            try:
                await self._perform_monitoring_cycle()
                self.last_monitoring_check = datetime.now()
                await asyncio.sleep(self.monitoring_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def _report_generation_loop(self) -> None:
        """Background report generation loop."""
        while self.status == ValidationFrameworkStatus.ACTIVE:
            try:
                if self._should_generate_reports():
                    await self._generate_scheduled_reports()
                    self.last_report_generation = datetime.now()

                await asyncio.sleep(3600)  # Check every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in report generation loop: {e}")
                await asyncio.sleep(300)  # Wait before retry

    async def _cleanup_loop(self) -> None:
        """Background cleanup loop."""
        while self.status == ValidationFrameworkStatus.ACTIVE:
            try:
                await self._perform_cleanup_cycle()
                await asyncio.sleep(3600)  # Cleanup every hour

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait before retry

    # Task Management Methods

    async def _schedule_initial_validation_tasks(self, room_id: str) -> None:
        """Schedule initial validation tasks for a newly registered room."""
        current_time = datetime.now()

        # Schedule baseline report generation
        baseline_task = ValidationTask(
            task_id=f"baseline_report_{room_id}_{int(current_time.timestamp())}",
            room_id=room_id,
            task_type="baseline_report",
            priority=ValidationPriority.MEDIUM,
            scheduled_time=current_time + timedelta(minutes=5),
        )

        await self._schedule_task(baseline_task)

    async def _schedule_follow_up_tasks(
        self, room_id: str, validation_record: ValidationRecord
    ) -> None:
        """Schedule follow-up validation tasks after recording a prediction."""
        current_time = datetime.now()

        # Schedule validation check task for after predicted time
        validation_check_task = ValidationTask(
            task_id=f"validation_check_{room_id}_{int(current_time.timestamp())}",
            room_id=room_id,
            task_type="validation_check",
            priority=ValidationPriority.HIGH,
            scheduled_time=validation_record.predicted_time + timedelta(minutes=30),
            parameters={
                "validation_record_id": validation_record.timestamp.isoformat()
            },
        )

        await self._schedule_task(validation_check_task)

    async def _schedule_task(self, task: ValidationTask) -> None:
        """Schedule a validation task."""
        self.validation_tasks[task.room_id].append(task)
        self.pending_tasks.append(task)
        self.logger.debug(f"Scheduled task {task.task_id} for room {task.room_id}")

    # Helper Methods

    def _get_accuracy_metrics_dict(
        self, room_id: str, time_window_hours: int
    ) -> Dict[str, Any]:
        """Get accuracy metrics as dictionary."""
        try:
            metrics = self.validator.get_accuracy_metrics(
                room_id, window_hours=time_window_hours
            )
            if metrics:
                return {
                    "mean_error_minutes": metrics.mean_error_minutes,
                    "median_error_minutes": metrics.median_error_minutes,
                    "std_error_minutes": metrics.std_error_minutes,
                    "accuracy_percentage": metrics.accuracy_percentage,
                    "total_predictions": metrics.total_predictions,
                    "accurate_predictions": metrics.accurate_predictions,
                    "accuracy_level": (
                        metrics.accuracy_level.value if metrics.accuracy_level else None
                    ),
                }
            return {}
        except Exception as e:
            self.logger.error(f"Error getting accuracy metrics for room {room_id}: {e}")
            return {}

    def _get_calibration_metrics_dict(
        self, room_id: str, time_window_hours: int
    ) -> Dict[str, Any]:
        """Get calibration metrics as dictionary."""
        try:
            metrics = self.validator.get_calibration_metrics(
                room_id, window_hours=time_window_hours
            )
            return metrics if metrics else {}
        except Exception as e:
            self.logger.error(
                f"Error getting calibration metrics for room {room_id}: {e}"
            )
            return {}

    def _get_performance_metrics_dict(
        self, room_id: str, time_window_hours: int
    ) -> Dict[str, Any]:
        """Get performance metrics as dictionary."""
        try:
            # Implementation would depend on performance tracking in validator
            return {
                "average_validation_time_ms": 0,
                "validation_success_rate": 1.0,
                "data_completeness": 1.0,
            }
        except Exception as e:
            self.logger.error(
                f"Error getting performance metrics for room {room_id}: {e}"
            )
            return {}

    async def _perform_trend_analysis(
        self, room_id: str, time_window_hours: int
    ) -> Dict[str, Any]:
        """Perform trend analysis on validation data."""
        try:
            trend_analysis = self.validator.get_confidence_trend_analysis(
                room_id, window_hours=time_window_hours
            )
            return trend_analysis if trend_analysis else {}
        except Exception as e:
            self.logger.error(
                f"Error performing trend analysis for room {room_id}: {e}"
            )
            return {}

    async def _perform_anomaly_detection(
        self, room_id: str, time_window_hours: int
    ) -> Dict[str, Any]:
        """Perform anomaly detection on validation data."""
        try:
            # Placeholder for anomaly detection implementation
            return {
                "anomalies_detected": 0,
                "anomaly_types": [],
                "anomaly_severity": "none",
            }
        except Exception as e:
            self.logger.error(
                f"Error performing anomaly detection for room {room_id}: {e}"
            )
            return {}

    async def _generate_recommendations(
        self,
        room_id: str,
        accuracy_metrics: Dict[str, Any],
        calibration_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate actionable recommendations based on validation results."""
        recommendations = []

        # Accuracy-based recommendations
        if accuracy_metrics.get("accuracy_percentage", 100) < 70:
            recommendations.append(
                {
                    "type": "accuracy_improvement",
                    "priority": "high",
                    "description": "Model accuracy below acceptable threshold",
                    "suggested_action": "Consider model retraining with recent data",
                    "expected_impact": "Improve prediction accuracy by 10-20%",
                }
            )

        # Calibration-based recommendations
        if calibration_metrics.get("calibration_score", 1.0) < 0.6:
            recommendations.append(
                {
                    "type": "calibration_adjustment",
                    "priority": "medium",
                    "description": "Confidence scores are poorly calibrated",
                    "suggested_action": "Apply confidence calibration adjustment",
                    "expected_impact": "Improve confidence reliability",
                }
            )

        return recommendations

    async def _generate_action_items(
        self,
        room_id: str,
        recommendations: List[Dict[str, Any]],
        trend_analysis: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Generate specific action items from recommendations."""
        action_items = []

        for rec in recommendations:
            if rec["type"] == "accuracy_improvement":
                action_items.append(
                    {
                        "action": "schedule_model_retraining",
                        "room_id": room_id,
                        "priority": rec["priority"],
                        "due_date": (datetime.now() + timedelta(days=1)).isoformat(),
                        "description": "Schedule model retraining for improved accuracy",
                    }
                )

        return action_items

    def _calculate_data_quality_score(
        self, room_id: str, time_window_hours: int
    ) -> float:
        """Calculate data quality score for the room."""
        try:
            records = self.validator.get_validation_history(
                room_id, window_hours=time_window_hours
            )
            if not records:
                return 0.0

            # Simple quality score based on data completeness and consistency
            complete_records = len(
                [r for r in records if r.status == ValidationStatus.VALIDATED]
            )
            return min(1.0, complete_records / max(1, len(records)))
        except Exception:
            return 0.0

    def _calculate_report_confidence_score(
        self, accuracy_metrics: Dict[str, Any], calibration_metrics: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for the report."""
        accuracy_score = accuracy_metrics.get("accuracy_percentage", 0) / 100.0
        calibration_score = calibration_metrics.get("calibration_score", 0)

        return (accuracy_score + calibration_score) / 2.0

    def _calculate_report_completeness(
        self,
        accuracy_metrics: Dict[str, Any],
        calibration_metrics: Dict[str, Any],
        performance_metrics: Dict[str, Any],
    ) -> float:
        """Calculate report completeness score."""
        sections = [accuracy_metrics, calibration_metrics, performance_metrics]
        non_empty_sections = len([s for s in sections if s])
        return non_empty_sections / len(sections)

    async def _save_validation_report(self, report: ValidationReport) -> None:
        """Save validation report to artifacts directory."""
        try:
            report_filename = (
                f"validation_report_{report.room_id}_"
                f"{report.generated_at.strftime('%Y%m%d_%H%M%S')}.json"
            )
            report_path = self.artifacts_directory / report_filename

            with open(report_path, "w") as f:
                json.dump(report.to_dict(), f, indent=2)

            self.logger.debug(f"Saved validation report to {report_path}")
        except Exception as e:
            self.logger.error(f"Error saving validation report: {e}")

    async def _handle_calibration_alert(
        self, room_id: str, alert_data: Dict[str, Any]
    ) -> None:
        """Handle calibration alert from validator."""
        self.logger.warning(f"Calibration alert for room {room_id}: {alert_data}")

        # Notify alert handlers
        for handler in self.alert_handlers:
            try:
                await handler(room_id, "calibration", alert_data)
            except Exception as e:
                self.logger.error(f"Error in alert handler: {e}")


# Framework Factory and Utilities


def create_validation_framework(
    accuracy_threshold_minutes: int = 15,
    confidence_threshold: float = 0.8,
    enable_real_time_monitoring: bool = True,
    artifacts_directory: Optional[Path] = None,
) -> PredictionValidationFramework:
    """Factory function to create a configured validation framework."""
    return PredictionValidationFramework(
        accuracy_threshold_minutes=accuracy_threshold_minutes,
        confidence_threshold=confidence_threshold,
        enable_real_time_monitoring=enable_real_time_monitoring,
        artifacts_directory=artifacts_directory,
    )


async def run_validation_framework_demo() -> None:
    """Demo function showing framework usage."""
    framework = create_validation_framework(
        accuracy_threshold_minutes=15,
        confidence_threshold=0.8,
        artifacts_directory=Path("demo_validation_artifacts"),
    )

    try:
        # Start framework
        await framework.start()

        # Register rooms
        await framework.register_room("living_room")
        await framework.register_room("kitchen")

        # Simulate some predictions and validations
        # (This would be replaced with actual prediction integration)

        # Generate reports
        report = await framework.generate_validation_report("living_room")
        print(f"Generated report with {report.report_completeness:.2f} completeness")

        # Get system health
        health = await framework.get_system_health_status()
        print(f"System health: {health['health_status']}")

    finally:
        # Stop framework
        await framework.stop()


if __name__ == "__main__":
    asyncio.run(run_validation_framework_demo())
