"""
Event processing pipeline for Home Assistant sensor events.

This module handles validation, enrichment, and classification of incoming sensor events,
including detection of human vs cat movement patterns, deduplication, and sequence analysis.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

import statistics

from ...core.config import RoomConfig, SystemConfig, get_config
from ...core.constants import (
    ABSENCE_STATES,
    CAT_MOVEMENT_PATTERNS,
    HUMAN_MOVEMENT_PATTERNS,
    INVALID_STATES,
    MAX_SEQUENCE_GAP,
    MIN_EVENT_SEPARATION,
    PRESENCE_STATES,
    SensorState,
    SensorType,
)
from ...core.exceptions import (
    ConfigurationError,
    DataValidationError,
    FeatureExtractionError,
)
from ..storage.models import SensorEvent
from .ha_client import HAEvent

logger = logging.getLogger(__name__)


@dataclass
class MovementSequence:
    """Represents a sequence of related sensor events."""

    events: List[SensorEvent]
    start_time: datetime
    end_time: datetime
    duration_seconds: float
    rooms_visited: Set[str]
    sensors_triggered: Set[str]

    @property
    def average_velocity(self) -> float:
        """Calculate average movement velocity through sensors."""
        if len(self.events) < 2:
            return 0.0

        total_distance = len(self.sensors_triggered)  # Simplified distance metric
        return (
            total_distance / self.duration_seconds if self.duration_seconds > 0 else 0.0
        )

    @property
    def trigger_pattern(self) -> str:
        """Get string representation of sensor trigger pattern."""
        return " -> ".join([event.sensor_id.split(".")[-1] for event in self.events])


@dataclass
class ValidationResult:
    """Result of event validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    confidence_score: float = 1.0


@dataclass
class ClassificationResult:
    """Result of movement pattern classification."""

    is_human_triggered: bool
    confidence_score: float
    classification_reason: str
    movement_metrics: Dict[str, float]


class EventValidator:
    """Validates incoming sensor events for data quality and consistency."""

    def __init__(self, config: SystemConfig):
        self.config = config

    def validate_event(self, event: SensorEvent) -> ValidationResult:
        """
        Validate a sensor event for data quality.

        Args:
            event: The sensor event to validate

        Returns:
            ValidationResult with validation status and details
        """
        errors = []
        warnings = []
        confidence_score = 1.0

        # Required field validation
        if not event.room_id:
            errors.append("Missing room_id")

        if not event.sensor_id:
            errors.append("Missing sensor_id")

        if not event.state:
            errors.append("Missing state")

        if not event.timestamp:
            errors.append("Missing timestamp")

        # Enhanced state validation using imported constants
        if event.state in INVALID_STATES:
            errors.append(f"Invalid state: {event.state}")

        # Validate state transitions using PRESENCE_STATES and ABSENCE_STATES
        if event.state and event.previous_state:
            valid_presence_transition = (
                event.previous_state in ABSENCE_STATES
                and event.state in PRESENCE_STATES
            )
            valid_absence_transition = (
                event.previous_state in PRESENCE_STATES
                and event.state in ABSENCE_STATES
            )
            valid_same_category = (
                event.previous_state in PRESENCE_STATES
                and event.state in PRESENCE_STATES
            ) or (
                event.previous_state in ABSENCE_STATES and event.state in ABSENCE_STATES
            )

            if not (
                valid_presence_transition
                or valid_absence_transition
                or valid_same_category
            ):
                warnings.append(
                    f"Unusual state transition: {event.previous_state} -> {event.state}"
                )
                confidence_score *= 0.9

        # Validate sensor state consistency with SensorState enum
        if hasattr(SensorState, event.state.upper()):
            # State is valid according to SensorState enum
            pass
        else:
            warnings.append(f"State {event.state} not in SensorState enumeration")
            confidence_score *= 0.95

        # Timestamp validation
        if event.timestamp:
            now = datetime.utcnow()
            if event.timestamp > now + timedelta(minutes=5):
                warnings.append("Event timestamp is in the future")
                confidence_score *= 0.9

            # Events older than 24 hours might be historical imports
            if event.timestamp < now - timedelta(days=1):
                warnings.append("Event timestamp is more than 24 hours old")
                confidence_score *= 0.95

        # Room and sensor validation
        room_config = self.config.rooms.get(event.room_id)
        if not room_config:
            warnings.append(f"Unknown room_id: {event.room_id}")
            confidence_score *= 0.8
        elif event.sensor_id not in room_config.get_all_entity_ids():
            warnings.append(
                f"Sensor {event.sensor_id} not configured for room {event.room_id}"
            )
            confidence_score *= 0.9

        # State transition validation
        if event.state == event.previous_state:
            warnings.append("State did not change from previous state")
            confidence_score *= 0.7

        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score,
        )


class MovementPatternClassifier:
    """Classifies movement patterns as human or cat based on sensor sequences."""

    def __init__(self, config: SystemConfig):
        self.config = config
        self.human_patterns = HUMAN_MOVEMENT_PATTERNS
        self.cat_patterns = CAT_MOVEMENT_PATTERNS

    def classify_movement(
        self, sequence: MovementSequence, room_config: RoomConfig
    ) -> ClassificationResult:
        """
        Classify a movement sequence as human or cat.

        Args:
            sequence: The movement sequence to classify
            room_config: Configuration for the room

        Returns:
            ClassificationResult with classification and confidence
        """
        metrics = self._calculate_movement_metrics(sequence, room_config)

        # Calculate scores for human and cat patterns
        human_score = self._score_human_pattern(metrics)
        cat_score = self._score_cat_pattern(metrics)

        # Determine classification
        is_human = human_score > cat_score
        max_score = max(human_score, cat_score)
        confidence = min(max_score, 1.0)

        # Adjust confidence based on sequence quality
        if len(sequence.events) < 3:
            confidence *= 0.7  # Low confidence for short sequences

        if sequence.duration_seconds < 5:
            confidence *= 0.8  # Very quick movements are harder to classify

        classification_reason = self._generate_classification_reason(
            metrics, human_score, cat_score, is_human
        )

        return ClassificationResult(
            is_human_triggered=is_human,
            confidence_score=confidence,
            classification_reason=classification_reason,
            movement_metrics=metrics,
        )

    def _calculate_movement_metrics(
        self, sequence: MovementSequence, room_config: RoomConfig
    ) -> Dict[str, float]:
        """Calculate metrics for movement pattern analysis."""
        metrics = {}

        # Duration metrics
        metrics["duration_seconds"] = sequence.duration_seconds
        metrics["event_count"] = len(sequence.events)
        metrics["rooms_visited"] = len(sequence.rooms_visited)
        metrics["sensors_triggered"] = len(sequence.sensors_triggered)

        # Velocity metrics
        metrics["average_velocity"] = sequence.average_velocity
        metrics["max_velocity"] = self._calculate_max_velocity(sequence)

        # Pattern metrics
        metrics["door_interactions"] = self._count_door_interactions(
            sequence, room_config
        )
        metrics["presence_sensor_ratio"] = self._calculate_presence_ratio(
            sequence, room_config
        )
        metrics["revisit_count"] = self._count_sensor_revisits(sequence)

        # Timing metrics with advanced mathematical analysis
        metrics["avg_sensor_dwell_time"] = self._calculate_avg_dwell_time(sequence)
        metrics["inter_event_variance"] = self._calculate_timing_variance(sequence)

        # Advanced mathematical metrics
        metrics["movement_entropy"] = self._calculate_movement_entropy(sequence)
        metrics["spatial_dispersion"] = self._calculate_spatial_dispersion(
            sequence, room_config
        )

        # Mathematical complexity metrics
        if sequence.duration_seconds > 0:
            metrics["movement_complexity"] = (
                metrics["movement_entropy"]
                * math.log(1 + metrics["spatial_dispersion"])
                * math.sqrt(metrics["inter_event_variance"] + 1)
            )
        else:
            metrics["movement_complexity"] = 0.0

        return metrics

    def _calculate_max_velocity(self, sequence: MovementSequence) -> float:
        """Calculate maximum velocity between consecutive events."""
        if len(sequence.events) < 2:
            return 0.0

        max_velocity = 0.0
        for i in range(1, len(sequence.events)):
            time_diff = (
                sequence.events[i].timestamp - sequence.events[i - 1].timestamp
            ).total_seconds()
            if time_diff > 0:
                # Simplified velocity calculation (sensors per second)
                velocity = 1.0 / time_diff
                max_velocity = max(max_velocity, velocity)

        return max_velocity

    def _count_door_interactions(
        self, sequence: MovementSequence, room_config: RoomConfig
    ) -> int:
        """Count door sensor interactions in the sequence."""
        door_sensors = room_config.get_sensors_by_type("door")
        door_entity_ids = set(door_sensors.values())

        door_interactions = 0
        for event in sequence.events:
            if event.sensor_id in door_entity_ids:
                door_interactions += 1

        return door_interactions

    def _calculate_presence_ratio(
        self, sequence: MovementSequence, room_config: RoomConfig
    ) -> float:
        """Calculate ratio of presence sensor activations."""
        presence_sensors = room_config.get_sensors_by_type("presence")
        presence_entity_ids = set(presence_sensors.values())

        presence_events = sum(
            1 for event in sequence.events if event.sensor_id in presence_entity_ids
        )
        return presence_events / len(sequence.events) if sequence.events else 0.0

    def _count_sensor_revisits(self, sequence: MovementSequence) -> int:
        """Count how many sensors were triggered multiple times."""
        sensor_counts = defaultdict(int)
        for event in sequence.events:
            sensor_counts[event.sensor_id] += 1

        return sum(1 for count in sensor_counts.values() if count > 1)

    def _calculate_avg_dwell_time(self, sequence: MovementSequence) -> float:
        """Calculate average time spent at each sensor using mathematical analysis."""
        if len(sequence.events) < 2:
            return sequence.duration_seconds

        sensor_times = defaultdict(list)
        for event in sequence.events:
            sensor_times[event.sensor_id].append(event.timestamp)

        dwell_times = []
        for sensor_id, timestamps in sensor_times.items():
            if len(timestamps) >= 2:
                # Calculate time between first and last activation using mathematical functions
                dwell_time = (max(timestamps) - min(timestamps)).total_seconds()
                dwell_times.append(dwell_time)

        if not dwell_times:
            return sequence.duration_seconds

        # Use mathematical statistics for better precision
        mean_dwell = statistics.mean(dwell_times)

        # Apply mathematical normalization using log function if needed
        if mean_dwell > 0:
            # Use math.log to normalize extreme values
            normalized_dwell = mean_dwell * (1 + math.log(1 + mean_dwell / 60))
            return normalized_dwell

        return mean_dwell

    def _calculate_timing_variance(self, sequence: MovementSequence) -> float:
        """Calculate variance in inter-event timing using advanced mathematical analysis."""
        if len(sequence.events) < 3:
            return 0.0

        intervals = []
        for i in range(1, len(sequence.events)):
            interval = (
                sequence.events[i].timestamp - sequence.events[i - 1].timestamp
            ).total_seconds()
            intervals.append(interval)

        if len(intervals) <= 1:
            return 0.0

        # Calculate mathematical variance with additional statistical measures
        variance = statistics.variance(intervals)

        # Apply mathematical transformations for better analysis
        if variance > 0:
            # Use coefficient of variation for normalized comparison
            mean_interval = statistics.mean(intervals)
            if mean_interval > 0:
                coefficient_of_variation = math.sqrt(variance) / mean_interval
                # Apply mathematical scaling using exponential function
                return variance * (1 + math.exp(-coefficient_of_variation))

        return variance

    def _calculate_movement_entropy(self, sequence: MovementSequence) -> float:
        """
        Calculate movement entropy to measure randomness of movement patterns.
        Uses mathematical information theory concepts.
        """
        if len(sequence.events) < 3:
            return 0.0

        # Count sensor transitions
        transitions = defaultdict(int)
        total_transitions = 0

        for i in range(1, len(sequence.events)):
            from_sensor = sequence.events[i - 1].sensor_id
            to_sensor = sequence.events[i].sensor_id
            if from_sensor != to_sensor:
                transitions[(from_sensor, to_sensor)] += 1
                total_transitions += 1

        if total_transitions == 0:
            return 0.0

        # Calculate entropy using Shannon's formula: H = -Σ(p * log2(p))
        entropy = 0.0
        for count in transitions.values():
            probability = count / total_transitions
            if probability > 0:
                entropy -= probability * math.log2(probability)

        return entropy

    def _calculate_spatial_dispersion(
        self, sequence: MovementSequence, room_config
    ) -> float:
        """
        Calculate spatial dispersion of movement using mathematical distance metrics.
        """
        if len(sequence.sensors_triggered) < 2:
            return 0.0

        # Create a simplified spatial mapping (this could be enhanced with real coordinates)
        sensor_positions = {}
        position_index = 0

        # Assign positions based on sensor type and order in configuration
        for sensor_type, sensors in room_config.sensors.items():
            if isinstance(sensors, dict):
                for sensor_name, sensor_id in sensors.items():
                    sensor_positions[sensor_id] = position_index
                    position_index += 1
            elif isinstance(sensors, str):
                sensor_positions[sensors] = position_index
                position_index += 1

        # Calculate dispersion using mathematical distance metrics
        positions = [
            sensor_positions.get(sensor_id, 0)
            for sensor_id in sequence.sensors_triggered
        ]

        if len(positions) < 2:
            return 0.0

        # Calculate standard deviation of positions as dispersion measure
        mean_position = statistics.mean(positions)
        variance = sum((pos - mean_position) ** 2 for pos in positions) / len(positions)

        # Apply mathematical transformation using square root (standard deviation)
        return math.sqrt(variance)

    def _score_human_pattern(self, metrics: Dict[str, float]) -> float:
        """Score how well metrics match human movement patterns."""
        score = 0.0

        # Duration scoring (humans typically move slower)
        if metrics["duration_seconds"] >= self.human_patterns["min_duration_seconds"]:
            score += 0.3

        # Velocity scoring (humans move at moderate speeds)
        if metrics["max_velocity"] <= self.human_patterns["max_velocity_ms"]:
            score += 0.2

        # Door interaction scoring (humans typically open doors)
        door_ratio = metrics["door_interactions"] / max(metrics["event_count"], 1)
        if door_ratio >= self.human_patterns["door_interaction_probability"] * 0.5:
            score += 0.2

        # Sequence length scoring (humans have purposeful paths)
        if (
            metrics["event_count"]
            <= self.human_patterns["typical_room_sequence_length"] * 1.5
        ):
            score += 0.15

        # Revisit penalty (humans typically don't backtrack as much)
        revisit_ratio = metrics["revisit_count"] / max(metrics["sensors_triggered"], 1)
        if revisit_ratio < 0.3:
            score += 0.15

        return score

    def _score_cat_pattern(self, metrics: Dict[str, float]) -> float:
        """Score how well metrics match cat movement patterns."""
        score = 0.0

        # Duration scoring (cats can move very quickly)
        if metrics["duration_seconds"] >= self.cat_patterns["min_duration_seconds"]:
            score += 0.2

        # Velocity scoring (cats can move very fast)
        if metrics["max_velocity"] <= self.cat_patterns["max_velocity_ms"]:
            score += 0.25

        # Door interaction scoring (cats rarely interact with doors)
        door_ratio = metrics["door_interactions"] / max(metrics["event_count"], 1)
        if door_ratio <= self.cat_patterns["door_interaction_probability"]:
            score += 0.25

        # Sequence length scoring (cats explore more randomly)
        if (
            metrics["event_count"]
            >= self.cat_patterns["typical_room_sequence_length"] * 0.8
        ):
            score += 0.1

        # Revisit scoring (cats often backtrack and explore)
        revisit_ratio = metrics["revisit_count"] / max(metrics["sensors_triggered"], 1)
        if revisit_ratio >= 0.2:
            score += 0.2

        return score

    def _generate_classification_reason(
        self,
        metrics: Dict[str, float],
        human_score: float,
        cat_score: float,
        is_human: bool,
    ) -> str:
        """Generate human-readable reason for classification."""
        reasons = []

        if is_human:
            if (
                metrics["duration_seconds"]
                >= self.human_patterns["min_duration_seconds"]
            ):
                reasons.append("typical human movement duration")
            if metrics["door_interactions"] > 0:
                reasons.append("door interactions observed")
            if metrics["revisit_count"] == 0:
                reasons.append("direct movement pattern")
        else:
            if metrics["max_velocity"] > self.human_patterns["max_velocity_ms"]:
                reasons.append("high movement velocity")
            if metrics["door_interactions"] == 0:
                reasons.append("no door interactions")
            if metrics["revisit_count"] > 0:
                reasons.append("exploratory movement pattern")

        base_reason = f"{'Human' if is_human else 'Cat'} pattern (score: {human_score:.2f} vs {cat_score:.2f})"
        if reasons:
            return f"{base_reason}: {', '.join(reasons)}"
        return base_reason

    def analyze_sequence_patterns(
        self, sequence: MovementSequence, room_config: RoomConfig
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        Analyze movement sequence patterns returning classification, confidence, and metrics.

        Args:
            sequence: Movement sequence to analyze
            room_config: Room configuration for context

        Returns:
            Tuple of (classification, confidence, detailed_metrics)
        """
        # Get classification result
        result = self.classify_movement_sequence(sequence, room_config)

        # Calculate detailed metrics
        metrics = self._calculate_movement_metrics(sequence, room_config)

        # Add statistical analysis
        detailed_metrics = {
            **metrics,
            "statistical_confidence": result.confidence_score,
            "pattern_consistency": self._calculate_pattern_consistency(sequence),
            "anomaly_score": self._calculate_anomaly_score(metrics),
        }

        classification = "human" if result.is_human_triggered else "cat"

        return classification, result.confidence_score, detailed_metrics

    def get_sequence_time_analysis(
        self, sequence: MovementSequence
    ) -> Tuple[float, float, float, int]:
        """
        Analyze timing patterns in a movement sequence.

        Args:
            sequence: Movement sequence to analyze

        Returns:
            Tuple of (min_interval, max_interval, avg_interval, total_gaps)
        """
        if len(sequence.events) < 2:
            return 0.0, 0.0, 0.0, 0

        intervals = []
        total_gaps = 0

        for i in range(1, len(sequence.events)):
            interval = (
                sequence.events[i].timestamp - sequence.events[i - 1].timestamp
            ).total_seconds()
            intervals.append(interval)

            # Count significant gaps (> 5 seconds)
            if interval > 5.0:
                total_gaps += 1

        if intervals:
            min_interval = min(intervals)
            max_interval = max(intervals)
            avg_interval = sum(intervals) / len(intervals)
        else:
            min_interval = max_interval = avg_interval = 0.0

        return min_interval, max_interval, avg_interval, total_gaps

    def extract_movement_signature(
        self, sequence: MovementSequence, room_config: RoomConfig
    ) -> Tuple[List[str], Dict[str, int], float]:
        """
        Extract movement signature for pattern matching.

        Args:
            sequence: Movement sequence to analyze
            room_config: Room configuration

        Returns:
            Tuple of (sensor_path, sensor_frequencies, uniqueness_score)
        """
        # Extract sensor path
        sensor_path = [event.sensor_id.split(".")[-1] for event in sequence.events]

        # Count sensor frequencies
        sensor_frequencies = {}
        for sensor in sensor_path:
            sensor_frequencies[sensor] = sensor_frequencies.get(sensor, 0) + 1

        # Calculate uniqueness score (0.0 = highly repetitive, 1.0 = all unique)
        unique_sensors = len(set(sensor_path))
        total_sensors = len(sensor_path)
        uniqueness_score = (
            unique_sensors / max(total_sensors, 1) if total_sensors > 0 else 0.0
        )

        return sensor_path, sensor_frequencies, uniqueness_score

    def compare_movement_patterns(
        self,
        sequence1: MovementSequence,
        sequence2: MovementSequence,
        room_config: RoomConfig,
    ) -> Tuple[float, Dict[str, float], bool]:
        """
        Compare two movement sequences for similarity.

        Args:
            sequence1: First movement sequence
            sequence2: Second movement sequence
            room_config: Room configuration

        Returns:
            Tuple of (similarity_score, comparison_metrics, is_same_pattern_type)
        """
        # Get metrics for both sequences
        metrics1 = self._calculate_movement_metrics(sequence1, room_config)
        metrics2 = self._calculate_movement_metrics(sequence2, room_config)

        # Calculate similarity score
        similarity_components = []
        comparison_metrics = {}

        for key in metrics1.keys():
            if key in metrics2:
                val1, val2 = metrics1[key], metrics2[key]
                max_val = max(abs(val1), abs(val2), 1.0)
                diff_ratio = abs(val1 - val2) / max_val
                similarity = 1.0 - min(diff_ratio, 1.0)
                similarity_components.append(similarity)
                comparison_metrics[f"{key}_similarity"] = similarity

        overall_similarity = (
            sum(similarity_components) / len(similarity_components)
            if similarity_components
            else 0.0
        )

        # Determine if same pattern type
        class1 = self.classify_movement_sequence(sequence1, room_config)
        class2 = self.classify_movement_sequence(sequence2, room_config)
        is_same_pattern = class1.is_human_triggered == class2.is_human_triggered

        return overall_similarity, comparison_metrics, is_same_pattern

    def _calculate_pattern_consistency(self, sequence: MovementSequence) -> float:
        """Calculate how consistent the movement pattern is."""
        if len(sequence.events) < 3:
            return 1.0

        intervals = []
        for i in range(1, len(sequence.events)):
            interval = (
                sequence.events[i].timestamp - sequence.events[i - 1].timestamp
            ).total_seconds()
            intervals.append(interval)

        if not intervals:
            return 1.0

        # Calculate coefficient of variation (lower = more consistent)
        mean_interval = sum(intervals) / len(intervals)
        if mean_interval == 0:
            return 1.0

        variance = sum((x - mean_interval) ** 2 for x in intervals) / len(intervals)
        std_dev = math.sqrt(variance)
        cv = std_dev / mean_interval

        # Convert to consistency score (0 = inconsistent, 1 = perfectly consistent)
        consistency = 1.0 / (1.0 + cv)
        return consistency

    def _calculate_anomaly_score(self, metrics: Dict[str, float]) -> float:
        """Calculate anomaly score based on metric values."""
        # Define expected ranges for normal behavior
        expected_ranges = {
            "duration_seconds": (5.0, 300.0),
            "average_velocity": (0.1, 5.0),
            "movement_entropy": (0.0, 3.0),
            "spatial_dispersion": (0.0, 10.0),
        }

        anomaly_components = []
        for key, (min_val, max_val) in expected_ranges.items():
            if key in metrics:
                value = metrics[key]
                if value < min_val:
                    anomaly = (min_val - value) / min_val
                elif value > max_val:
                    anomaly = (value - max_val) / max_val
                else:
                    anomaly = 0.0
                anomaly_components.append(min(anomaly, 1.0))

        return (
            sum(anomaly_components) / len(anomaly_components)
            if anomaly_components
            else 0.0
        )


class EventProcessor:
    """
    Main event processing pipeline for Home Assistant sensor events.

    Handles:
    - Event validation and filtering
    - Deduplication based on time separation
    - Movement pattern classification
    - Event enrichment and sequence analysis
    - Bulk processing for historical imports
    """

    def __init__(self, config: Optional[SystemConfig] = None, tracking_manager=None):
        self.config = config or get_config()
        self.validator = EventValidator(self.config)
        self.classifier = MovementPatternClassifier(self.config)

        # Event tracking for sequence detection
        self._recent_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._last_processed_times: Dict[str, datetime] = {}

        # Processing statistics
        self.stats = {
            "total_processed": 0,
            "valid_events": 0,
            "invalid_events": 0,
            "human_classified": 0,
            "cat_classified": 0,
            "duplicates_filtered": 0,
        }

        # Tracking integration for automatic validation
        self.tracking_manager = tracking_manager

    async def process_event(self, ha_event: HAEvent) -> Optional[SensorEvent]:
        """
        Process a single Home Assistant event.

        Args:
            ha_event: The HA event to process

        Returns:
            Processed SensorEvent if valid, None if filtered out
        """
        self.stats["total_processed"] += 1

        # Find room configuration
        room_config = self.config.get_room_by_entity_id(ha_event.entity_id)
        if not room_config:
            logger.warning(
                f"No room configuration found for entity {ha_event.entity_id}"
            )
            return None

        # Determine sensor type
        sensor_type = self._determine_sensor_type(ha_event.entity_id, room_config)

        # Convert to SensorEvent
        sensor_event = SensorEvent(
            room_id=room_config.room_id,
            sensor_id=ha_event.entity_id,
            sensor_type=sensor_type,
            state=ha_event.state,
            previous_state=ha_event.previous_state,
            timestamp=ha_event.timestamp,
            attributes=ha_event.attributes,
            created_at=datetime.utcnow(),
        )

        # Validate event
        validation_result = self.validator.validate_event(sensor_event)
        if not validation_result.is_valid:
            logger.warning(f"Invalid event filtered out: {validation_result.errors}")
            self.stats["invalid_events"] += 1
            return None

        # Apply deduplication
        if self._is_duplicate_event(sensor_event):
            self.stats["duplicates_filtered"] += 1
            return None

        # Enrich event with classification
        await self._enrich_event(
            sensor_event, room_config, validation_result.confidence_score
        )

        # Update tracking
        self._update_event_tracking(sensor_event)

        # Check for room state changes and notify tracking manager
        await self._check_room_state_change(sensor_event, room_config)

        self.stats["valid_events"] += 1
        return sensor_event

    async def process_event_batch(
        self, ha_events: List[HAEvent], batch_size: int = 100
    ) -> List[SensorEvent]:
        """
        Process a batch of Home Assistant events efficiently.

        Args:
            ha_events: List of HA events to process
            batch_size: Size of processing batches

        Returns:
            List of processed SensorEvents
        """
        processed_events = []

        # Process in smaller batches to avoid memory issues
        for i in range(0, len(ha_events), batch_size):
            batch = ha_events[i : i + batch_size]
            batch_results = []

            for ha_event in batch:
                result = await self.process_event(ha_event)
                if result:
                    batch_results.append(result)

            processed_events.extend(batch_results)

            # Yield control periodically for other tasks
            if i % (batch_size * 10) == 0:
                await asyncio.sleep(0.01)

        return processed_events

    def _determine_sensor_type(self, entity_id: str, room_config: RoomConfig) -> str:
        """Determine sensor type from entity ID and room configuration."""
        # Check each sensor type in room config
        for sensor_type, sensors in room_config.sensors.items():
            if isinstance(sensors, dict):
                if entity_id in sensors.values():
                    return sensor_type
            elif isinstance(sensors, str) and entity_id == sensors:
                return sensor_type

        # Fallback to entity ID analysis
        if "motion" in entity_id or "presence" in entity_id:
            return SensorType.PRESENCE.value
        elif "door" in entity_id:
            return SensorType.DOOR.value
        elif "temperature" in entity_id:
            return SensorType.CLIMATE.value
        elif "light" in entity_id:
            return SensorType.LIGHT.value
        else:
            return SensorType.MOTION.value  # Default

    def _is_duplicate_event(self, event: SensorEvent) -> bool:
        """Check if event is a duplicate based on time separation."""
        key = f"{event.room_id}:{event.sensor_id}"
        last_time = self._last_processed_times.get(key)

        if last_time:
            time_diff = (event.timestamp - last_time).total_seconds()
            if time_diff < MIN_EVENT_SEPARATION:
                return True

        return False

    async def _enrich_event(
        self,
        event: SensorEvent,
        room_config: RoomConfig,
        base_confidence: float,
    ):
        """Enrich event with additional metadata and classification."""
        # Get recent events for sequence analysis
        recent_events = list(self._recent_events[event.room_id])

        # Create movement sequence if we have enough events
        if len(recent_events) >= 2:
            sequence = self._create_movement_sequence(recent_events + [event])
            if sequence:
                # Classify movement pattern
                classification = self.classifier.classify_movement(
                    sequence, room_config
                )

                event.is_human_triggered = classification.is_human_triggered
                event.confidence_score = (
                    classification.confidence_score * base_confidence
                )

                # Add classification metadata
                if not event.attributes:
                    event.attributes = {}
                event.attributes.update(
                    {
                        "classification_reason": classification.classification_reason,
                        "movement_metrics": classification.movement_metrics,
                    }
                )

                # Update statistics
                if classification.is_human_triggered:
                    self.stats["human_classified"] += 1
                else:
                    self.stats["cat_classified"] += 1
        else:
            # Default classification for isolated events
            event.is_human_triggered = True
            event.confidence_score = (
                base_confidence * 0.8
            )  # Lower confidence for isolated events

    def _create_movement_sequence(
        self, events: List[SensorEvent]
    ) -> Optional[MovementSequence]:
        """Create a movement sequence from a list of events."""
        if len(events) < 2:
            return None

        # Filter events within reasonable time window
        end_time = events[-1].timestamp
        start_time = end_time - timedelta(seconds=MAX_SEQUENCE_GAP)

        sequence_events = [e for e in events if e.timestamp >= start_time]

        if len(sequence_events) < 2:
            return None

        return MovementSequence(
            events=sequence_events,
            start_time=sequence_events[0].timestamp,
            end_time=sequence_events[-1].timestamp,
            duration_seconds=(
                sequence_events[-1].timestamp - sequence_events[0].timestamp
            ).total_seconds(),
            rooms_visited={e.room_id for e in sequence_events},
            sensors_triggered={e.sensor_id for e in sequence_events},
        )

    async def _check_room_state_change(
        self, event: SensorEvent, room_config: RoomConfig
    ):
        """
        Check if this event indicates a room state change and notify tracking manager.

        Args:
            event: The processed sensor event
            room_config: Configuration for this room
        """
        if not self.tracking_manager:
            return

        try:
            # Detect potential occupancy changes based on sensor types and states
            state_change_detected = False
            new_state = None
            previous_state = None

            # Check for presence sensor state changes
            presence_sensors = room_config.get_sensors_by_type("presence")
            if event.sensor_id in presence_sensors.values():
                if event.state in ["on", "detected", "occupied"]:
                    new_state = "occupied"
                    if event.previous_state in ["of", "clear", "vacant"]:
                        previous_state = "vacant"
                        state_change_detected = True
                elif event.state in ["of", "clear", "vacant"]:
                    new_state = "vacant"
                    if event.previous_state in ["on", "detected", "occupied"]:
                        previous_state = "occupied"
                        state_change_detected = True

            # Check for motion sensor patterns (more complex logic needed)
            motion_sensors = room_config.get_sensors_by_type("motion")
            if event.sensor_id in motion_sensors.values():
                # For motion sensors, we need to analyze recent activity patterns
                # This is a simplified approach - real implementation would be more sophisticated
                recent_events = list(self._recent_events[event.room_id])
                if len(recent_events) >= 2:
                    # Look for motion start/stop patterns
                    if event.state == "on" and event.previous_state == "of":
                        # Motion detected - potential room entry
                        new_state = "occupied"
                        previous_state = "vacant"
                        state_change_detected = True
                    elif event.state == "of" and event.previous_state == "on":
                        # Motion stopped - check if room might be vacant
                        # Only trigger if no recent motion in last few minutes
                        cutoff_time = event.timestamp - timedelta(minutes=5)
                        recent_motion = any(
                            e.timestamp > cutoff_time and e.state == "on"
                            for e in recent_events[-10:]
                            if e.sensor_id in motion_sensors.values()
                        )
                        if not recent_motion:
                            new_state = "vacant"
                            previous_state = "occupied"
                            state_change_detected = True

            # Notify tracking manager if state change detected
            if state_change_detected and new_state:
                await self.tracking_manager.handle_room_state_change(
                    room_id=event.room_id,
                    new_state=new_state,
                    change_time=event.timestamp,
                    previous_state=previous_state,
                )

                logger.debug(
                    f"Detected room state change for {event.room_id}: "
                    f"{previous_state} -> {new_state} at {event.timestamp}"
                )

        except Exception as e:
            logger.error(f"Failed to check room state change: {e}")
            # Don't raise exception to prevent disrupting event processing

    def _update_event_tracking(self, event: SensorEvent):
        """Update internal tracking for sequence detection."""
        # Add to recent events for room
        self._recent_events[event.room_id].append(event)

        # Update last processed time
        key = f"{event.room_id}:{event.sensor_id}"
        self._last_processed_times[key] = event.timestamp

    def get_processing_stats(self) -> Dict[str, int]:
        """Get processing statistics."""
        return self.stats.copy()

    def reset_stats(self):
        """Reset processing statistics."""
        self.stats = {
            "total_processed": 0,
            "valid_events": 0,
            "invalid_events": 0,
            "human_classified": 0,
            "cat_classified": 0,
            "duplicates_filtered": 0,
        }

    async def validate_event_sequence_integrity(
        self, events: List[SensorEvent], tolerance_seconds: float = 1.0
    ) -> Dict[str, Any]:
        """
        Validate the integrity of an event sequence using mathematical analysis.

        Args:
            events: List of sensor events to validate
            tolerance_seconds: Tolerance for timing anomalies

        Returns:
            Dictionary with integrity analysis results
        """
        if len(events) < 2:
            return {
                "valid": True,
                "issues": [],
                "confidence": 1.0,
                "analysis": "Insufficient events for sequence analysis",
            }

        issues = []
        confidence = 1.0

        try:
            # Check temporal ordering
            for i in range(1, len(events)):
                if events[i].timestamp < events[i - 1].timestamp:
                    issues.append(f"Temporal ordering violation at index {i}")
                    confidence *= 0.8

            # Check for timing anomalies using mathematical analysis
            intervals = []
            for i in range(1, len(events)):
                interval = (
                    events[i].timestamp - events[i - 1].timestamp
                ).total_seconds()
                intervals.append(interval)

            if intervals:
                mean_interval = statistics.mean(intervals)
                std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0

                # Use mathematical z-score to detect anomalies
                for i, interval in enumerate(intervals):
                    if std_interval > 0:
                        z_score = abs((interval - mean_interval) / std_interval)
                        if z_score > 3:  # Statistical threshold for outliers
                            issues.append(
                                f"Timing anomaly at interval {i+1}: z-score = {z_score:.2f}"
                            )
                            confidence *= 0.9

            # Check state transition patterns using mathematical entropy
            state_transitions = []
            for i in range(1, len(events)):
                if events[i].state != events[i - 1].state:
                    state_transitions.append((events[i - 1].state, events[i].state))

            if state_transitions:
                # Calculate transition entropy
                transition_counts = defaultdict(int)
                for transition in state_transitions:
                    transition_counts[transition] += 1

                total_transitions = len(state_transitions)
                entropy = 0.0
                for count in transition_counts.values():
                    probability = count / total_transitions
                    if probability > 0:
                        entropy -= probability * math.log2(probability)

                # Unusually low entropy might indicate data quality issues
                if entropy < 0.5 and len(transition_counts) > 1:
                    issues.append(
                        f"Low transition entropy: {entropy:.2f} (possible repetitive pattern)"
                    )
                    confidence *= 0.95

            # Check for missing required state information
            missing_states = sum(1 for event in events if not event.state)
            if missing_states > 0:
                issues.append(f"{missing_states} events missing state information")
                confidence *= 1 - missing_states / len(events)

            return {
                "valid": len(issues) == 0,
                "issues": issues,
                "confidence": confidence,
                "analysis": {
                    "total_events": len(events),
                    "temporal_span_seconds": (
                        events[-1].timestamp - events[0].timestamp
                    ).total_seconds(),
                    "unique_states": len(
                        set(event.state for event in events if event.state)
                    ),
                    "transition_count": len(state_transitions),
                    "mean_interval_seconds": (
                        statistics.mean(intervals) if intervals else 0
                    ),
                    "interval_std_seconds": (
                        statistics.stdev(intervals) if len(intervals) > 1 else 0
                    ),
                },
            }

        except (ConfigurationError, DataValidationError, FeatureExtractionError) as e:
            # Handle domain-specific errors
            logger.error(f"Domain-specific error during sequence validation: {e}")
            return {
                "valid": False,
                "issues": [f"Domain error: {str(e)}"],
                "confidence": 0.0,
                "analysis": "Validation failed due to domain-specific error",
            }
        except Exception as e:
            # Handle unexpected errors with proper exception handling
            logger.error(f"Unexpected error during sequence validation: {e}")
            raise FeatureExtractionError(
                feature_type="sequence_validation", room_id="unknown", cause=e
            )

    async def validate_room_configuration(self, room_id: str) -> Dict[str, Any]:
        """
        Validate room configuration for event processing.

        Args:
            room_id: Room ID to validate

        Returns:
            Dictionary with validation results
        """
        room_config = self.config.rooms.get(room_id)
        if not room_config:
            return {
                "valid": False,
                "error": f"Room {room_id} not found in configuration",
            }

        entity_ids = room_config.get_all_entity_ids()
        if not entity_ids:
            return {
                "valid": False,
                "error": f"No entities configured for room {room_id}",
            }

        # Check for required sensor types
        has_presence = bool(room_config.get_sensors_by_type("presence"))
        has_motion = bool(room_config.get_sensors_by_type("motion"))

        warnings = []
        if not (has_presence or has_motion):
            warnings.append("No presence or motion sensors configured")

        return {
            "valid": True,
            "entity_count": len(entity_ids),
            "sensor_types": list(room_config.sensors.keys()),
            "warnings": warnings,
        }
