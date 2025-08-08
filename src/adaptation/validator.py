"""
Prediction Validator Infrastructure for Sprint 4 - Self-Adaptation System.

This module provides comprehensive real-time prediction validation, accuracy tracking,
and performance analysis capabilities for the occupancy prediction system.
"""

import asyncio
from collections import defaultdict, deque
import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from sqlalchemy import and_, desc, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from ..core.constants import ModelType
from ..core.exceptions import (
    DatabaseError,
    ErrorSeverity,
    OccupancyPredictionError,
)
from ..data.storage.database import get_db_session
from ..data.storage.models import Prediction
from ..models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


class ValidationStatus(Enum):
    """Status of prediction validation."""

    PENDING = "pending"
    VALIDATED = "validated"
    EXPIRED = "expired"
    FAILED = "failed"


class AccuracyLevel(Enum):
    """Classification of prediction accuracy levels."""

    EXCELLENT = "excellent"  # < 5 min error
    GOOD = "good"  # 5-10 min error
    ACCEPTABLE = "acceptable"  # 10-15 min error
    POOR = "poor"  # 15-30 min error
    UNACCEPTABLE = "unacceptable"  # > 30 min error


@dataclass
class ValidationRecord:
    """
    Comprehensive record for storing prediction validation data.

    Tracks all aspects of a prediction's lifecycle from generation to validation,
    including performance metrics and confidence analysis.
    """

    prediction_id: str
    room_id: str
    model_type: str
    model_version: str

    # Prediction details
    predicted_time: datetime
    transition_type: str
    confidence_score: float
    prediction_interval: Optional[Tuple[datetime, datetime]] = None
    alternatives: Optional[List[Tuple[datetime, float]]] = None

    # Validation details
    actual_time: Optional[datetime] = None
    error_minutes: Optional[float] = None
    accuracy_level: Optional[AccuracyLevel] = None
    status: ValidationStatus = ValidationStatus.PENDING

    # Metadata
    prediction_time: datetime = field(default_factory=datetime.utcnow)
    validation_time: Optional[datetime] = None
    expiration_time: Optional[datetime] = None

    # Feature context (for analysis)
    feature_snapshot: Optional[Dict[str, Any]] = None
    prediction_metadata: Optional[Dict[str, Any]] = None

    def validate_against_actual(
        self, actual_time: datetime, threshold_minutes: int = 15
    ) -> bool:
        """
        Validate prediction against actual transition time.

        Args:
            actual_time: When the transition actually occurred
            threshold_minutes: Accuracy threshold in minutes

        Returns:
            True if prediction is accurate, False otherwise
        """
        if self.status != ValidationStatus.PENDING:
            raise ValidationError(
                f"Cannot validate prediction {self.prediction_id}: already {self.status.value}"
            )

        self.actual_time = actual_time
        self.validation_time = datetime.utcnow()

        # Calculate error in minutes
        time_diff = (actual_time - self.predicted_time).total_seconds() / 60
        self.error_minutes = abs(time_diff)

        # Classify accuracy level
        if self.error_minutes < 5:
            self.accuracy_level = AccuracyLevel.EXCELLENT
        elif self.error_minutes < 10:
            self.accuracy_level = AccuracyLevel.GOOD
        elif self.error_minutes < 15:
            self.accuracy_level = AccuracyLevel.ACCEPTABLE
        elif self.error_minutes < 30:
            self.accuracy_level = AccuracyLevel.POOR
        else:
            self.accuracy_level = AccuracyLevel.UNACCEPTABLE

        # Mark as validated
        self.status = ValidationStatus.VALIDATED

        # Return whether it meets threshold
        is_accurate = self.error_minutes <= threshold_minutes

        logger.debug(
            f"Validated prediction {self.prediction_id}: "
            f"error={self.error_minutes:.1f}min, accurate={is_accurate}, "
            f"level={self.accuracy_level.value}"
        )

        return is_accurate

    def mark_expired(self, expiration_time: Optional[datetime] = None) -> None:
        """Mark prediction as expired (no validation possible)."""
        if self.status == ValidationStatus.VALIDATED:
            raise ValidationError(
                f"Cannot expire prediction {self.prediction_id}: already validated"
            )

        self.status = ValidationStatus.EXPIRED
        self.expiration_time = expiration_time or datetime.utcnow()

        logger.debug(f"Marked prediction {self.prediction_id} as expired")

    def mark_failed(self, reason: str) -> None:
        """Mark prediction as failed validation."""
        self.status = ValidationStatus.FAILED
        self.validation_time = datetime.utcnow()

        if not self.prediction_metadata:
            self.prediction_metadata = {}
        self.prediction_metadata["failure_reason"] = reason

        logger.warning(
            f"Marked prediction {self.prediction_id} as failed: {reason}"
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert validation record to dictionary for serialization."""
        return {
            "prediction_id": self.prediction_id,
            "room_id": self.room_id,
            "model_type": self.model_type,
            "model_version": self.model_version,
            "predicted_time": self.predicted_time.isoformat(),
            "transition_type": self.transition_type,
            "confidence_score": self.confidence_score,
            "prediction_interval": (
                [
                    self.prediction_interval[0].isoformat(),
                    self.prediction_interval[1].isoformat(),
                ]
                if self.prediction_interval
                else None
            ),
            "alternatives": (
                [[alt[0].isoformat(), alt[1]] for alt in self.alternatives]
                if self.alternatives
                else None
            ),
            "actual_time": (
                self.actual_time.isoformat() if self.actual_time else None
            ),
            "error_minutes": self.error_minutes,
            "accuracy_level": (
                self.accuracy_level.value if self.accuracy_level else None
            ),
            "status": self.status.value,
            "prediction_time": self.prediction_time.isoformat(),
            "validation_time": (
                self.validation_time.isoformat()
                if self.validation_time
                else None
            ),
            "expiration_time": (
                self.expiration_time.isoformat()
                if self.expiration_time
                else None
            ),
            "feature_snapshot": self.feature_snapshot,
            "prediction_metadata": self.prediction_metadata,
        }


@dataclass
class AccuracyMetrics:
    """
    Comprehensive accuracy statistics and performance analysis.

    Provides detailed insights into prediction performance across different
    dimensions including accuracy rates, error distributions, and confidence calibration.
    """

    # Basic counts
    total_predictions: int = 0
    validated_predictions: int = 0
    accurate_predictions: int = 0
    expired_predictions: int = 0
    failed_predictions: int = 0

    # Accuracy statistics
    accuracy_rate: float = 0.0
    mean_error_minutes: float = 0.0
    median_error_minutes: float = 0.0
    std_error_minutes: float = 0.0
    rmse_minutes: float = 0.0
    mae_minutes: float = 0.0

    # Error distribution
    error_percentiles: Dict[int, float] = field(
        default_factory=dict
    )  # 25th, 75th, 90th, 95th
    accuracy_by_level: Dict[str, int] = field(default_factory=dict)

    # Bias analysis
    mean_bias_minutes: float = (
        0.0  # Positive = late predictions, negative = early
    )
    bias_std_minutes: float = 0.0

    # Confidence analysis
    mean_confidence: float = 0.0
    confidence_accuracy_correlation: float = 0.0
    overconfidence_rate: float = 0.0  # High confidence but wrong predictions
    underconfidence_rate: float = 0.0  # Low confidence but right predictions

    # Time-based analysis
    measurement_period_start: Optional[datetime] = None
    measurement_period_end: Optional[datetime] = None
    predictions_per_hour: float = 0.0

    @property
    def validation_rate(self) -> float:
        """Percentage of predictions that were validated (not expired/failed)."""
        if self.total_predictions == 0:
            return 0.0
        return (self.validated_predictions / self.total_predictions) * 100

    @property
    def expiration_rate(self) -> float:
        """Percentage of predictions that expired before validation."""
        if self.total_predictions == 0:
            return 0.0
        return (self.expired_predictions / self.total_predictions) * 100

    @property
    def bias_direction(self) -> str:
        """Human-readable bias direction."""
        if abs(self.mean_bias_minutes) < 1:
            return "unbiased"
        elif self.mean_bias_minutes > 0:
            return "predicts_late"
        else:
            return "predicts_early"

    @property
    def confidence_calibration_score(self) -> float:
        """
        Score from 0-1 indicating how well confidence correlates with accuracy.
        1.0 = perfect calibration, 0.0 = no correlation.
        """
        # Convert correlation to 0-1 score
        return max(0.0, self.confidence_accuracy_correlation)

    def to_dict(self) -> Dict[str, Any]:
        """Convert accuracy metrics to dictionary for serialization."""
        return {
            "total_predictions": self.total_predictions,
            "validated_predictions": self.validated_predictions,
            "accurate_predictions": self.accurate_predictions,
            "expired_predictions": self.expired_predictions,
            "failed_predictions": self.failed_predictions,
            "accuracy_rate": self.accuracy_rate,
            "validation_rate": self.validation_rate,
            "expiration_rate": self.expiration_rate,
            "mean_error_minutes": self.mean_error_minutes,
            "median_error_minutes": self.median_error_minutes,
            "std_error_minutes": self.std_error_minutes,
            "rmse_minutes": self.rmse_minutes,
            "mae_minutes": self.mae_minutes,
            "error_percentiles": self.error_percentiles,
            "accuracy_by_level": self.accuracy_by_level,
            "mean_bias_minutes": self.mean_bias_minutes,
            "bias_std_minutes": self.bias_std_minutes,
            "bias_direction": self.bias_direction,
            "mean_confidence": self.mean_confidence,
            "confidence_accuracy_correlation": self.confidence_accuracy_correlation,
            "confidence_calibration_score": self.confidence_calibration_score,
            "overconfidence_rate": self.overconfidence_rate,
            "underconfidence_rate": self.underconfidence_rate,
            "measurement_period_start": (
                self.measurement_period_start.isoformat()
                if self.measurement_period_start
                else None
            ),
            "measurement_period_end": (
                self.measurement_period_end.isoformat()
                if self.measurement_period_end
                else None
            ),
            "predictions_per_hour": self.predictions_per_hour,
        }


class PredictionValidator:
    """
    Production-ready prediction validation system with real-time accuracy tracking.

    Provides comprehensive validation infrastructure including:
    - Thread-safe prediction recording and validation
    - Real-time accuracy metrics calculation with caching
    - Automatic cleanup and memory management
    - Database persistence for validation data
    - Export capabilities for analysis
    """

    def __init__(
        self,
        accuracy_threshold_minutes: int = 15,
        max_validation_delay_hours: int = 6,
        max_memory_records: int = 10000,
        cleanup_interval_hours: int = 12,
        metrics_cache_ttl_minutes: int = 30,
    ):
        """
        Initialize prediction validator with configuration.

        Args:
            accuracy_threshold_minutes: Threshold for considering predictions accurate
            max_validation_delay_hours: Maximum time to wait for validation
            max_memory_records: Maximum number of records to keep in memory
            cleanup_interval_hours: Interval for automatic cleanup
            metrics_cache_ttl_minutes: Cache TTL for computed metrics
        """
        self.accuracy_threshold = accuracy_threshold_minutes
        self.max_validation_delay = timedelta(hours=max_validation_delay_hours)
        self.max_memory_records = max_memory_records
        self.cleanup_interval = timedelta(hours=cleanup_interval_hours)
        self.metrics_cache_ttl = timedelta(minutes=metrics_cache_ttl_minutes)

        # Thread-safe storage
        self._lock = threading.RLock()
        self._validation_records: Dict[str, ValidationRecord] = {}
        self._records_by_room: Dict[str, List[str]] = defaultdict(list)
        self._records_by_model: Dict[str, List[str]] = defaultdict(list)

        # Metrics caching
        self._metrics_cache: Dict[str, Tuple[AccuracyMetrics, datetime]] = {}
        self._cache_lock = threading.Lock()

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        logger.info(
            f"Initialized PredictionValidator with threshold={accuracy_threshold_minutes}min, "
            f"max_delay={max_validation_delay_hours}h, max_records={max_memory_records}"
        )

    async def start_background_tasks(self) -> None:
        """Start background maintenance tasks."""
        try:
            # Start cleanup loop
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.append(cleanup_task)

            logger.info("Started PredictionValidator background tasks")

        except Exception as e:
            logger.error(f"Failed to start background tasks: {e}")
            raise ValidationError(
                "Failed to start validator background tasks", cause=e
            )

    async def stop_background_tasks(self) -> None:
        """Stop background tasks gracefully."""
        try:
            # Signal shutdown
            self._shutdown_event.set()

            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(
                    *self._background_tasks, return_exceptions=True
                )

            self._background_tasks.clear()
            logger.info("Stopped PredictionValidator background tasks")

        except Exception as e:
            logger.error(f"Error stopping background tasks: {e}")

    async def record_prediction(
        self,
        prediction: PredictionResult,
        room_id: str,
        feature_snapshot: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Record a prediction for later validation with database persistence.

        Args:
            prediction: The prediction result to record
            room_id: Room where prediction was made
            feature_snapshot: Optional snapshot of features used

        Returns:
            Unique prediction ID for tracking
        """
        try:
            # Generate unique prediction ID
            prediction_id = f"{room_id}_{prediction.model_type}_{int(prediction.predicted_time.timestamp())}"

            # Create validation record
            record = ValidationRecord(
                prediction_id=prediction_id,
                room_id=room_id,
                model_type=prediction.model_type or "unknown",
                model_version=prediction.model_version or "unknown",
                predicted_time=prediction.predicted_time,
                transition_type=prediction.transition_type,
                confidence_score=prediction.confidence_score,
                prediction_interval=prediction.prediction_interval,
                alternatives=prediction.alternatives,
                feature_snapshot=feature_snapshot,
                prediction_metadata=prediction.prediction_metadata,
            )

            # Store in memory with thread safety
            with self._lock:
                self._validation_records[prediction_id] = record
                self._records_by_room[room_id].append(prediction_id)
                self._records_by_model[record.model_type].append(prediction_id)

                # Cleanup if over memory limit
                self._cleanup_if_needed()

            # Store in database asynchronously
            await self._store_prediction_in_db(record)

            # Invalidate relevant caches
            self._invalidate_metrics_cache(room_id, record.model_type)

            logger.debug(
                f"Recorded prediction {prediction_id} for room {room_id}"
            )

            return prediction_id

        except Exception as e:
            logger.error(f"Failed to record prediction: {e}")
            raise ValidationError(
                "Failed to record prediction for validation", cause=e
            )

    async def validate_prediction(
        self,
        room_id: str,
        actual_transition_time: datetime,
        transition_type: str,
        max_time_window_minutes: int = 60,
    ) -> List[str]:
        """
        Validate predictions against actual transition with batch processing.

        Args:
            room_id: Room where transition occurred
            actual_transition_time: When transition actually happened
            transition_type: Type of transition that occurred
            max_time_window_minutes: Maximum time window to look for predictions

        Returns:
            List of prediction IDs that were validated
        """
        try:
            validated_predictions = []

            # Find predictions to validate
            candidates = await self._find_predictions_for_validation(
                room_id,
                actual_transition_time,
                transition_type,
                max_time_window_minutes,
            )

            if not candidates:
                logger.debug(
                    f"No predictions found for validation in {room_id} at "
                    f"{actual_transition_time} ({transition_type})"
                )
                return []

            # Validate each candidate
            updated_records = []

            with self._lock:
                for prediction_id in candidates:
                    if prediction_id in self._validation_records:
                        record = self._validation_records[prediction_id]

                        if record.status == ValidationStatus.PENDING:
                            # Validate the prediction
                            is_accurate = record.validate_against_actual(
                                actual_transition_time, self.accuracy_threshold
                            )

                            validated_predictions.append(prediction_id)
                            updated_records.append(record)

                            logger.debug(
                                f"Validated prediction {prediction_id}: "
                                f"accurate={is_accurate}, error={record.error_minutes:.1f}min"
                            )

            # Update database in batch
            if updated_records:
                await self._update_predictions_in_db(updated_records)

                # Invalidate caches for affected entities
                affected_rooms = {r.room_id for r in updated_records}
                affected_models = {r.model_type for r in updated_records}

                for room in affected_rooms:
                    for model in affected_models:
                        self._invalidate_metrics_cache(room, model)

            logger.info(
                f"Validated {len(validated_predictions)} predictions for {room_id} "
                f"transition at {actual_transition_time}"
            )

            return validated_predictions

        except Exception as e:
            logger.error(f"Failed to validate predictions: {e}")
            raise ValidationError(
                "Failed to validate predictions against actual", cause=e
            )

    async def get_accuracy_metrics(
        self,
        room_id: Optional[str] = None,
        model_type: Optional[str] = None,
        hours_back: int = 24,
        force_recalculate: bool = False,
    ) -> AccuracyMetrics:
        """
        Calculate comprehensive accuracy metrics with caching.

        Args:
            room_id: Filter by specific room (None for all rooms)
            model_type: Filter by specific model (None for all models)
            hours_back: How many hours back to analyze
            force_recalculate: Force recalculation even if cached

        Returns:
            Comprehensive accuracy metrics
        """
        try:
            # Generate cache key
            cache_key = (
                f"{room_id or 'all'}_{model_type or 'all'}_{hours_back}"
            )

            # Check cache if not forcing recalculation
            if not force_recalculate and self._is_metrics_cache_valid(
                cache_key
            ):
                with self._cache_lock:
                    cached_metrics, _ = self._metrics_cache[cache_key]
                    logger.debug(f"Using cached metrics for {cache_key}")
                    return cached_metrics

            # Get filtered records
            records = self._get_filtered_records(
                room_id, model_type, hours_back
            )

            # Calculate metrics
            metrics = self._calculate_metrics_from_records(records, hours_back)

            # Cache the results
            self._cache_metrics(cache_key, metrics)

            logger.debug(
                f"Calculated accuracy metrics: {metrics.total_predictions} predictions, "
                f"{metrics.accuracy_rate:.1f}% accuracy, {metrics.mean_error_minutes:.1f}min avg error"
            )

            return metrics

        except Exception as e:
            logger.error(f"Failed to calculate accuracy metrics: {e}")
            raise ValidationError(
                "Failed to calculate accuracy metrics", cause=e
            )

    async def get_room_accuracy(
        self, room_id: str, hours_back: int = 24
    ) -> AccuracyMetrics:
        """Get accuracy metrics for specific room across all models."""
        return await self.get_accuracy_metrics(
            room_id=room_id, hours_back=hours_back
        )

    async def get_model_accuracy(
        self, model_type: str, hours_back: int = 24
    ) -> AccuracyMetrics:
        """Get accuracy metrics for specific model across all rooms."""
        return await self.get_accuracy_metrics(
            model_type=model_type, hours_back=hours_back
        )

    async def get_pending_validations(
        self, room_id: Optional[str] = None, expired_only: bool = False
    ) -> List[ValidationRecord]:
        """
        Get predictions that need validation or are expired.

        Args:
            room_id: Filter by specific room
            expired_only: Only return expired predictions

        Returns:
            List of validation records needing attention
        """
        try:
            pending_records = []
            cutoff_time = datetime.utcnow() - self.max_validation_delay

            with self._lock:
                for record in self._validation_records.values():
                    # Apply room filter
                    if room_id and record.room_id != room_id:
                        continue

                    # Check status conditions
                    if expired_only:
                        # Only expired predictions
                        if (
                            record.status == ValidationStatus.PENDING
                            and record.predicted_time < cutoff_time
                        ):
                            pending_records.append(record)
                    else:
                        # All pending predictions
                        if record.status == ValidationStatus.PENDING:
                            pending_records.append(record)

            logger.debug(f"Found {len(pending_records)} pending validations")
            return pending_records

        except Exception as e:
            logger.error(f"Failed to get pending validations: {e}")
            raise ValidationError("Failed to get pending validations", cause=e)

    async def expire_old_predictions(
        self, cutoff_hours: Optional[int] = None
    ) -> int:
        """
        Mark old predictions as expired if they haven't been validated.

        Args:
            cutoff_hours: Hours after which to expire (uses default if None)

        Returns:
            Number of predictions expired
        """
        try:
            if cutoff_hours is None:
                cutoff_time = datetime.utcnow() - self.max_validation_delay
            else:
                cutoff_time = datetime.utcnow() - timedelta(hours=cutoff_hours)

            expired_count = 0
            expired_records = []

            with self._lock:
                for record in self._validation_records.values():
                    if (
                        record.status == ValidationStatus.PENDING
                        and record.predicted_time < cutoff_time
                    ):

                        record.mark_expired(cutoff_time)
                        expired_records.append(record)
                        expired_count += 1

            # Update database if any records were expired
            if expired_records:
                await self._update_predictions_in_db(expired_records)

                # Invalidate caches
                affected_rooms = {r.room_id for r in expired_records}
                affected_models = {r.model_type for r in expired_records}

                for room in affected_rooms:
                    for model in affected_models:
                        self._invalidate_metrics_cache(room, model)

            if expired_count > 0:
                logger.info(f"Expired {expired_count} old predictions")

            return expired_count

        except Exception as e:
            logger.error(f"Failed to expire old predictions: {e}")
            raise ValidationError("Failed to expire old predictions", cause=e)

    async def export_validation_data(
        self,
        output_path: Union[str, Path],
        format: str = "csv",
        room_id: Optional[str] = None,
        days_back: int = 7,
    ) -> int:
        """
        Export validation data for analysis.

        Args:
            output_path: Where to save the exported data
            format: Export format ("csv" or "json")
            room_id: Filter by specific room
            days_back: How many days of data to export

        Returns:
            Number of records exported
        """
        try:
            # Get records for export
            cutoff_time = datetime.utcnow() - timedelta(days=days_back)
            export_records = []

            with self._lock:
                for record in self._validation_records.values():
                    if record.prediction_time < cutoff_time:
                        continue

                    if room_id and record.room_id != room_id:
                        continue

                    export_records.append(record)

            # Sort by prediction time
            export_records.sort(key=lambda r: r.prediction_time)

            # Export based on format
            output_path = Path(output_path)

            if format.lower() == "csv":
                await self._export_to_csv(export_records, output_path)
            elif format.lower() == "json":
                await self._export_to_json(export_records, output_path)
            else:
                raise ValidationError(f"Unsupported export format: {format}")

            logger.info(
                f"Exported {len(export_records)} validation records to {output_path}"
            )
            return len(export_records)

        except Exception as e:
            logger.error(f"Failed to export validation data: {e}")
            raise ValidationError("Failed to export validation data", cause=e)

    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation system statistics."""
        try:
            with self._lock:
                total_records = len(self._validation_records)
                room_counts = {
                    room: len(predictions)
                    for room, predictions in self._records_by_room.items()
                }
                model_counts = {
                    model: len(predictions)
                    for model, predictions in self._records_by_model.items()
                }

                status_counts = defaultdict(int)
                for record in self._validation_records.values():
                    status_counts[record.status.value] += 1

            with self._cache_lock:
                cache_size = len(self._metrics_cache)

            return {
                "total_records": total_records,
                "records_by_room": dict(room_counts),
                "records_by_model": dict(model_counts),
                "records_by_status": dict(status_counts),
                "cache_size": cache_size,
                "memory_usage_percent": (
                    total_records / self.max_memory_records
                )
                * 100,
                "background_tasks_running": len(self._background_tasks),
            }

        except Exception as e:
            logger.error(f"Failed to get validation stats: {e}")
            return {}

    def cleanup_old_records(self, days_to_keep: int = 30) -> int:
        """
        Remove old validation records from memory to free up space.

        Args:
            days_to_keep: Number of days of records to keep in memory

        Returns:
            Number of records removed
        """
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days_to_keep)
            removed_count = 0

            with self._lock:
                # Find records to remove
                records_to_remove = [
                    prediction_id
                    for prediction_id, record in self._validation_records.items()
                    if record.prediction_time < cutoff_time
                ]

                # Remove records
                for prediction_id in records_to_remove:
                    record = self._validation_records[prediction_id]

                    # Remove from main storage
                    del self._validation_records[prediction_id]

                    # Remove from indexes
                    self._records_by_room[record.room_id].remove(prediction_id)
                    self._records_by_model[record.model_type].remove(
                        prediction_id
                    )

                    removed_count += 1

                # Clean up empty lists
                self._records_by_room = {
                    room: predictions
                    for room, predictions in self._records_by_room.items()
                    if predictions
                }
                self._records_by_model = {
                    model: predictions
                    for model, predictions in self._records_by_model.items()
                    if predictions
                }

            if removed_count > 0:
                logger.info(
                    f"Cleaned up {removed_count} old validation records"
                )

            return removed_count

        except Exception as e:
            logger.error(f"Failed to cleanup old records: {e}")
            return 0

    # Private methods

    async def _store_prediction_in_db(self, record: ValidationRecord) -> None:
        """Store prediction record in database."""
        try:
            async with get_db_session() as session:
                # Create database prediction record
                db_prediction = Prediction(
                    room_id=record.room_id,
                    prediction_time=record.prediction_time,
                    predicted_transition_time=record.predicted_time,
                    transition_type=record.transition_type,
                    confidence_score=record.confidence_score,
                    model_type=record.model_type,
                    model_version=record.model_version,
                    prediction_interval_lower=(
                        record.prediction_interval[0]
                        if record.prediction_interval
                        else None
                    ),
                    prediction_interval_upper=(
                        record.prediction_interval[1]
                        if record.prediction_interval
                        else None
                    ),
                    alternatives=record.alternatives or [],
                )

                session.add(db_prediction)
                await session.commit()

        except Exception as e:
            logger.error(f"Failed to store prediction in database: {e}")
            # Don't raise exception - validation can continue without DB storage

    async def _update_predictions_in_db(
        self, records: List[ValidationRecord]
    ) -> None:
        """Update validated predictions in database."""
        try:
            if not records:
                return

            async with get_db_session() as session:
                for record in records:
                    # Find matching prediction in database
                    query = (
                        select(Prediction)
                        .where(
                            and_(
                                Prediction.room_id == record.room_id,
                                Prediction.predicted_transition_time
                                == record.predicted_time,
                                Prediction.model_type == record.model_type,
                            )
                        )
                        .limit(1)
                    )

                    result = await session.execute(query)
                    db_prediction = result.scalar_one_or_none()

                    if db_prediction:
                        # Update validation results
                        db_prediction.actual_transition_time = (
                            record.actual_time
                        )
                        db_prediction.accuracy_minutes = record.error_minutes
                        db_prediction.is_accurate = (
                            record.error_minutes <= self.accuracy_threshold
                            if record.error_minutes is not None
                            else None
                        )
                        db_prediction.validation_timestamp = (
                            record.validation_time
                        )

                await session.commit()

        except Exception as e:
            logger.error(f"Failed to update predictions in database: {e}")
            # Don't raise exception - validation system can continue

    async def _find_predictions_for_validation(
        self,
        room_id: str,
        actual_time: datetime,
        transition_type: str,
        max_window_minutes: int,
    ) -> List[str]:
        """Find prediction candidates for validation."""
        candidates = []
        time_window = timedelta(minutes=max_window_minutes)

        with self._lock:
            for prediction_id in self._records_by_room.get(room_id, []):
                record = self._validation_records.get(prediction_id)

                if not record or record.status != ValidationStatus.PENDING:
                    continue

                # Check if transition types match
                if record.transition_type != transition_type:
                    continue

                # Check if within time window
                time_diff = abs(
                    (actual_time - record.predicted_time).total_seconds()
                )
                if time_diff <= time_window.total_seconds():
                    candidates.append(prediction_id)

        return candidates

    def _get_filtered_records(
        self,
        room_id: Optional[str],
        model_type: Optional[str],
        hours_back: int,
    ) -> List[ValidationRecord]:
        """Get validation records filtered by criteria."""
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        filtered_records = []

        with self._lock:
            for record in self._validation_records.values():
                # Time filter
                if record.prediction_time < cutoff_time:
                    continue

                # Room filter
                if room_id and record.room_id != room_id:
                    continue

                # Model filter
                if model_type and record.model_type != model_type:
                    continue

                filtered_records.append(record)

        return filtered_records

    def _calculate_metrics_from_records(
        self, records: List[ValidationRecord], hours_back: int
    ) -> AccuracyMetrics:
        """Calculate comprehensive accuracy metrics from validation records."""
        if not records:
            return AccuracyMetrics()

        # Initialize metrics
        metrics = AccuracyMetrics()
        metrics.total_predictions = len(records)

        # Separate records by status
        validated_records = [
            r for r in records if r.status == ValidationStatus.VALIDATED
        ]
        expired_records = [
            r for r in records if r.status == ValidationStatus.EXPIRED
        ]
        failed_records = [
            r for r in records if r.status == ValidationStatus.FAILED
        ]

        metrics.validated_predictions = len(validated_records)
        metrics.expired_predictions = len(expired_records)
        metrics.failed_predictions = len(failed_records)

        if not validated_records:
            return metrics

        # Calculate error statistics
        errors = [
            r.error_minutes
            for r in validated_records
            if r.error_minutes is not None
        ]
        biases = [
            (r.actual_time - r.predicted_time).total_seconds() / 60
            for r in validated_records
            if r.actual_time
        ]
        confidences = [r.confidence_score for r in validated_records]

        if errors:
            import numpy as np
            import statistics

            metrics.mean_error_minutes = statistics.mean(errors)
            metrics.median_error_minutes = statistics.median(errors)
            metrics.std_error_minutes = (
                statistics.stdev(errors) if len(errors) > 1 else 0.0
            )
            metrics.rmse_minutes = (
                sum(e**2 for e in errors) / len(errors)
            ) ** 0.5
            metrics.mae_minutes = statistics.mean(errors)

            # Error percentiles
            sorted_errors = sorted(errors)
            metrics.error_percentiles = {
                25: np.percentile(sorted_errors, 25),
                75: np.percentile(sorted_errors, 75),
                90: np.percentile(sorted_errors, 90),
                95: np.percentile(sorted_errors, 95),
            }

            # Accuracy rate
            accurate_count = sum(
                1 for e in errors if e <= self.accuracy_threshold
            )
            metrics.accurate_predictions = accurate_count
            metrics.accuracy_rate = (accurate_count / len(errors)) * 100

            # Accuracy by level
            level_counts = defaultdict(int)
            for record in validated_records:
                if record.accuracy_level:
                    level_counts[record.accuracy_level.value] += 1
            metrics.accuracy_by_level = dict(level_counts)

        if biases:
            metrics.mean_bias_minutes = statistics.mean(biases)
            metrics.bias_std_minutes = (
                statistics.stdev(biases) if len(biases) > 1 else 0.0
            )

        if confidences:
            metrics.mean_confidence = statistics.mean(confidences)

            # Confidence-accuracy correlation
            if errors and len(errors) == len(confidences):
                import numpy as np

                # Calculate correlation between confidence and accuracy (inverse of error)
                accuracies = [
                    1 / (1 + e) for e in errors
                ]  # Transform error to accuracy score
                correlation_matrix = np.corrcoef(confidences, accuracies)
                metrics.confidence_accuracy_correlation = (
                    correlation_matrix[0, 1]
                    if not np.isnan(correlation_matrix[0, 1])
                    else 0.0
                )

                # Overconfidence/underconfidence rates
                high_conf_threshold = 0.8
                low_conf_threshold = 0.4

                high_conf_wrong = sum(
                    1
                    for conf, err in zip(confidences, errors)
                    if conf > high_conf_threshold
                    and err > self.accuracy_threshold
                )
                low_conf_right = sum(
                    1
                    for conf, err in zip(confidences, errors)
                    if conf < low_conf_threshold
                    and err <= self.accuracy_threshold
                )

                metrics.overconfidence_rate = (
                    high_conf_wrong / len(confidences)
                ) * 100
                metrics.underconfidence_rate = (
                    low_conf_right / len(confidences)
                ) * 100

        # Time-based analysis
        if records:
            prediction_times = [r.prediction_time for r in records]
            metrics.measurement_period_start = min(prediction_times)
            metrics.measurement_period_end = max(prediction_times)

            period_hours = (
                metrics.measurement_period_end
                - metrics.measurement_period_start
            ).total_seconds() / 3600
            if period_hours > 0:
                metrics.predictions_per_hour = len(records) / period_hours

        return metrics

    def _is_metrics_cache_valid(self, cache_key: str) -> bool:
        """Check if cached metrics are still valid."""
        with self._cache_lock:
            if cache_key not in self._metrics_cache:
                return False

            _, cache_time = self._metrics_cache[cache_key]
            return datetime.utcnow() - cache_time < self.metrics_cache_ttl

    def _cache_metrics(self, cache_key: str, metrics: AccuracyMetrics) -> None:
        """Cache metrics for faster retrieval."""
        with self._cache_lock:
            self._metrics_cache[cache_key] = (metrics, datetime.utcnow())

            # Limit cache size
            if len(self._metrics_cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(
                    self._metrics_cache.items(), key=lambda x: x[1][1]
                )
                for key, _ in sorted_cache[:10]:  # Remove oldest 10
                    del self._metrics_cache[key]

    def _invalidate_metrics_cache(self, room_id: str, model_type: str) -> None:
        """Invalidate cached metrics for affected entities."""
        with self._cache_lock:
            keys_to_remove = []
            for key in self._metrics_cache.keys():
                # Check if cache key matches room or model
                if room_id in key or model_type in key or "all" in key:
                    keys_to_remove.append(key)

            for key in keys_to_remove:
                del self._metrics_cache[key]

    def _cleanup_if_needed(self) -> None:
        """Cleanup records if memory limit is reached (must be called with lock)."""
        if len(self._validation_records) > self.max_memory_records:
            # Remove oldest 10% of records
            records_to_remove = int(0.1 * self.max_memory_records)
            oldest_records = sorted(
                self._validation_records.items(),
                key=lambda x: x[1].prediction_time,
            )[:records_to_remove]

            for prediction_id, record in oldest_records:
                del self._validation_records[prediction_id]
                self._records_by_room[record.room_id].remove(prediction_id)
                self._records_by_model[record.model_type].remove(prediction_id)

    async def _cleanup_loop(self) -> None:
        """Background loop for periodic cleanup."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Expire old predictions
                    await self.expire_old_predictions()

                    # Cleanup old records
                    self.cleanup_old_records()

                    # Wait for next cleanup cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.cleanup_interval.total_seconds(),
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for periodic cleanup
                    continue
                except Exception as e:
                    logger.error(f"Error in cleanup loop: {e}")
                    await asyncio.sleep(60)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Cleanup loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Cleanup loop failed: {e}")

    async def _export_to_csv(
        self, records: List[ValidationRecord], output_path: Path
    ) -> None:
        """Export validation records to CSV format."""
        with open(output_path, "w", newline="", encoding="utf-8") as csvfile:
            if not records:
                return

            # Get all possible fieldnames from records
            fieldnames = [
                "prediction_id",
                "room_id",
                "model_type",
                "model_version",
                "predicted_time",
                "transition_type",
                "confidence_score",
                "actual_time",
                "error_minutes",
                "accuracy_level",
                "status",
                "prediction_time",
                "validation_time",
                "expiration_time",
            ]

            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for record in records:
                row = {
                    "prediction_id": record.prediction_id,
                    "room_id": record.room_id,
                    "model_type": record.model_type,
                    "model_version": record.model_version,
                    "predicted_time": record.predicted_time.isoformat(),
                    "transition_type": record.transition_type,
                    "confidence_score": record.confidence_score,
                    "actual_time": (
                        record.actual_time.isoformat()
                        if record.actual_time
                        else ""
                    ),
                    "error_minutes": record.error_minutes,
                    "accuracy_level": (
                        record.accuracy_level.value
                        if record.accuracy_level
                        else ""
                    ),
                    "status": record.status.value,
                    "prediction_time": record.prediction_time.isoformat(),
                    "validation_time": (
                        record.validation_time.isoformat()
                        if record.validation_time
                        else ""
                    ),
                    "expiration_time": (
                        record.expiration_time.isoformat()
                        if record.expiration_time
                        else ""
                    ),
                }
                writer.writerow(row)

    async def _export_to_json(
        self, records: List[ValidationRecord], output_path: Path
    ) -> None:
        """Export validation records to JSON format."""
        export_data = {
            "export_time": datetime.utcnow().isoformat(),
            "record_count": len(records),
            "records": [record.to_dict() for record in records],
        }

        with open(output_path, "w", encoding="utf-8") as jsonfile:
            json.dump(export_data, jsonfile, indent=2, default=str)


# Custom exception for validation-specific errors
class ValidationError(OccupancyPredictionError):
    """Raised when prediction validation operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="VALIDATION_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
