"""
Event processing pipeline for Home Assistant sensor events.

This module handles validation, enrichment, and classification of incoming sensor events,
including detection of human vs cat movement patterns, deduplication, and sequence analysis.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass
from collections import defaultdict, deque
import statistics
import math

from ...core.config import SystemConfig, RoomConfig, get_config
from ...core.exceptions import (
    DataValidationError, FeatureExtractionError, 
    ConfigurationError
)
from ...core.constants import (
    SensorType, SensorState, MIN_EVENT_SEPARATION, MAX_SEQUENCE_GAP,
    PRESENCE_STATES, ABSENCE_STATES, INVALID_STATES,
    HUMAN_MOVEMENT_PATTERNS, CAT_MOVEMENT_PATTERNS
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
        return total_distance / self.duration_seconds if self.duration_seconds > 0 else 0.0
    
    @property
    def trigger_pattern(self) -> str:
        """Get string representation of sensor trigger pattern."""
        return " -> ".join([event.sensor_id.split('.')[-1] for event in self.events])


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
        
        # State validation
        if event.state in INVALID_STATES:
            errors.append(f"Invalid state: {event.state}")
        
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
            warnings.append(f"Sensor {event.sensor_id} not configured for room {event.room_id}")
            confidence_score *= 0.9
        
        # State transition validation
        if event.state == event.previous_state:
            warnings.append("State did not change from previous state")
            confidence_score *= 0.7
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            confidence_score=confidence_score
        )


class MovementPatternClassifier:
    """Classifies movement patterns as human or cat based on sensor sequences."""
    
    def __init__(self, config: SystemConfig):
        self.config = config
        self.human_patterns = HUMAN_MOVEMENT_PATTERNS
        self.cat_patterns = CAT_MOVEMENT_PATTERNS
    
    def classify_movement(
        self,
        sequence: MovementSequence,
        room_config: RoomConfig
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
            movement_metrics=metrics
        )
    
    def _calculate_movement_metrics(
        self,
        sequence: MovementSequence,
        room_config: RoomConfig
    ) -> Dict[str, float]:
        """Calculate metrics for movement pattern analysis."""
        metrics = {}
        
        # Duration metrics
        metrics['duration_seconds'] = sequence.duration_seconds
        metrics['event_count'] = len(sequence.events)
        metrics['rooms_visited'] = len(sequence.rooms_visited)
        metrics['sensors_triggered'] = len(sequence.sensors_triggered)
        
        # Velocity metrics
        metrics['average_velocity'] = sequence.average_velocity
        metrics['max_velocity'] = self._calculate_max_velocity(sequence)
        
        # Pattern metrics
        metrics['door_interactions'] = self._count_door_interactions(sequence, room_config)
        metrics['presence_sensor_ratio'] = self._calculate_presence_ratio(sequence, room_config)
        metrics['revisit_count'] = self._count_sensor_revisits(sequence)
        
        # Timing metrics
        metrics['avg_sensor_dwell_time'] = self._calculate_avg_dwell_time(sequence)
        metrics['inter_event_variance'] = self._calculate_timing_variance(sequence)
        
        return metrics
    
    def _calculate_max_velocity(self, sequence: MovementSequence) -> float:
        """Calculate maximum velocity between consecutive events."""
        if len(sequence.events) < 2:
            return 0.0
        
        max_velocity = 0.0
        for i in range(1, len(sequence.events)):
            time_diff = (sequence.events[i].timestamp - sequence.events[i-1].timestamp).total_seconds()
            if time_diff > 0:
                # Simplified velocity calculation (sensors per second)
                velocity = 1.0 / time_diff
                max_velocity = max(max_velocity, velocity)
        
        return max_velocity
    
    def _count_door_interactions(self, sequence: MovementSequence, room_config: RoomConfig) -> int:
        """Count door sensor interactions in the sequence."""
        door_sensors = room_config.get_sensors_by_type('door')
        door_entity_ids = set(door_sensors.values())
        
        door_interactions = 0
        for event in sequence.events:
            if event.sensor_id in door_entity_ids:
                door_interactions += 1
        
        return door_interactions
    
    def _calculate_presence_ratio(self, sequence: MovementSequence, room_config: RoomConfig) -> float:
        """Calculate ratio of presence sensor activations."""
        presence_sensors = room_config.get_sensors_by_type('presence')
        presence_entity_ids = set(presence_sensors.values())
        
        presence_events = sum(1 for event in sequence.events if event.sensor_id in presence_entity_ids)
        return presence_events / len(sequence.events) if sequence.events else 0.0
    
    def _count_sensor_revisits(self, sequence: MovementSequence) -> int:
        """Count how many sensors were triggered multiple times."""
        sensor_counts = defaultdict(int)
        for event in sequence.events:
            sensor_counts[event.sensor_id] += 1
        
        return sum(1 for count in sensor_counts.values() if count > 1)
    
    def _calculate_avg_dwell_time(self, sequence: MovementSequence) -> float:
        """Calculate average time spent at each sensor."""
        if len(sequence.events) < 2:
            return sequence.duration_seconds
        
        sensor_times = defaultdict(list)
        for event in sequence.events:
            sensor_times[event.sensor_id].append(event.timestamp)
        
        dwell_times = []
        for sensor_id, timestamps in sensor_times.items():
            if len(timestamps) >= 2:
                # Calculate time between first and last activation
                dwell_time = (max(timestamps) - min(timestamps)).total_seconds()
                dwell_times.append(dwell_time)
        
        return statistics.mean(dwell_times) if dwell_times else sequence.duration_seconds
    
    def _calculate_timing_variance(self, sequence: MovementSequence) -> float:
        """Calculate variance in inter-event timing."""
        if len(sequence.events) < 3:
            return 0.0
        
        intervals = []
        for i in range(1, len(sequence.events)):
            interval = (sequence.events[i].timestamp - sequence.events[i-1].timestamp).total_seconds()
            intervals.append(interval)
        
        return statistics.variance(intervals) if len(intervals) > 1 else 0.0
    
    def _score_human_pattern(self, metrics: Dict[str, float]) -> float:
        """Score how well metrics match human movement patterns."""
        score = 0.0
        
        # Duration scoring (humans typically move slower)
        if metrics['duration_seconds'] >= self.human_patterns['min_duration_seconds']:
            score += 0.3
        
        # Velocity scoring (humans move at moderate speeds)
        if metrics['max_velocity'] <= self.human_patterns['max_velocity_ms']:
            score += 0.2
        
        # Door interaction scoring (humans typically open doors)
        door_ratio = metrics['door_interactions'] / max(metrics['event_count'], 1)
        if door_ratio >= self.human_patterns['door_interaction_probability'] * 0.5:
            score += 0.2
        
        # Sequence length scoring (humans have purposeful paths)
        if metrics['event_count'] <= self.human_patterns['typical_room_sequence_length'] * 1.5:
            score += 0.15
        
        # Revisit penalty (humans typically don't backtrack as much)
        revisit_ratio = metrics['revisit_count'] / max(metrics['sensors_triggered'], 1)
        if revisit_ratio < 0.3:
            score += 0.15
        
        return score
    
    def _score_cat_pattern(self, metrics: Dict[str, float]) -> float:
        """Score how well metrics match cat movement patterns."""
        score = 0.0
        
        # Duration scoring (cats can move very quickly)
        if metrics['duration_seconds'] >= self.cat_patterns['min_duration_seconds']:
            score += 0.2
        
        # Velocity scoring (cats can move very fast)
        if metrics['max_velocity'] <= self.cat_patterns['max_velocity_ms']:
            score += 0.25
        
        # Door interaction scoring (cats rarely interact with doors)
        door_ratio = metrics['door_interactions'] / max(metrics['event_count'], 1)
        if door_ratio <= self.cat_patterns['door_interaction_probability']:
            score += 0.25
        
        # Sequence length scoring (cats explore more randomly)
        if metrics['event_count'] >= self.cat_patterns['typical_room_sequence_length'] * 0.8:
            score += 0.1
        
        # Revisit scoring (cats often backtrack and explore)
        revisit_ratio = metrics['revisit_count'] / max(metrics['sensors_triggered'], 1)
        if revisit_ratio >= 0.2:
            score += 0.2
        
        return score
    
    def _generate_classification_reason(
        self,
        metrics: Dict[str, float],
        human_score: float,
        cat_score: float,
        is_human: bool
    ) -> str:
        """Generate human-readable reason for classification."""
        reasons = []
        
        if is_human:
            if metrics['duration_seconds'] >= self.human_patterns['min_duration_seconds']:
                reasons.append("typical human movement duration")
            if metrics['door_interactions'] > 0:
                reasons.append("door interactions observed")
            if metrics['revisit_count'] == 0:
                reasons.append("direct movement pattern")
        else:
            if metrics['max_velocity'] > self.human_patterns['max_velocity_ms']:
                reasons.append("high movement velocity")
            if metrics['door_interactions'] == 0:
                reasons.append("no door interactions")
            if metrics['revisit_count'] > 0:
                reasons.append("exploratory movement pattern")
        
        base_reason = f"{'Human' if is_human else 'Cat'} pattern (score: {human_score:.2f} vs {cat_score:.2f})"
        if reasons:
            return f"{base_reason}: {', '.join(reasons)}"
        return base_reason


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
    
    def __init__(self, config: Optional[SystemConfig] = None):
        self.config = config or get_config()
        self.validator = EventValidator(self.config)
        self.classifier = MovementPatternClassifier(self.config)
        
        # Event tracking for sequence detection
        self._recent_events: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        self._last_processed_times: Dict[str, datetime] = {}
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'human_classified': 0,
            'cat_classified': 0,
            'duplicates_filtered': 0
        }
    
    async def process_event(self, ha_event: HAEvent) -> Optional[SensorEvent]:
        """
        Process a single Home Assistant event.
        
        Args:
            ha_event: The HA event to process
            
        Returns:
            Processed SensorEvent if valid, None if filtered out
        """
        self.stats['total_processed'] += 1
        
        # Find room configuration
        room_config = self.config.get_room_by_entity_id(ha_event.entity_id)
        if not room_config:
            logger.warning(f"No room configuration found for entity {ha_event.entity_id}")
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
            created_at=datetime.utcnow()
        )
        
        # Validate event
        validation_result = self.validator.validate_event(sensor_event)
        if not validation_result.is_valid:
            logger.warning(f"Invalid event filtered out: {validation_result.errors}")
            self.stats['invalid_events'] += 1
            return None
        
        # Apply deduplication
        if self._is_duplicate_event(sensor_event):
            self.stats['duplicates_filtered'] += 1
            return None
        
        # Enrich event with classification
        await self._enrich_event(sensor_event, room_config, validation_result.confidence_score)
        
        # Update tracking
        self._update_event_tracking(sensor_event)
        
        self.stats['valid_events'] += 1
        return sensor_event
    
    async def process_event_batch(
        self,
        ha_events: List[HAEvent],
        batch_size: int = 100
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
            batch = ha_events[i:i + batch_size]
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
        if 'motion' in entity_id or 'presence' in entity_id:
            return SensorType.PRESENCE.value
        elif 'door' in entity_id:
            return SensorType.DOOR.value
        elif 'temperature' in entity_id:
            return SensorType.CLIMATE.value
        elif 'light' in entity_id:
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
        base_confidence: float
    ):
        """Enrich event with additional metadata and classification."""
        # Get recent events for sequence analysis
        recent_events = list(self._recent_events[event.room_id])
        
        # Create movement sequence if we have enough events
        if len(recent_events) >= 2:
            sequence = self._create_movement_sequence(recent_events + [event])
            if sequence:
                # Classify movement pattern
                classification = self.classifier.classify_movement(sequence, room_config)
                
                event.is_human_triggered = classification.is_human_triggered
                event.confidence_score = classification.confidence_score * base_confidence
                
                # Add classification metadata
                if not event.attributes:
                    event.attributes = {}
                event.attributes.update({
                    'classification_reason': classification.classification_reason,
                    'movement_metrics': classification.movement_metrics
                })
                
                # Update statistics
                if classification.is_human_triggered:
                    self.stats['human_classified'] += 1
                else:
                    self.stats['cat_classified'] += 1
        else:
            # Default classification for isolated events
            event.is_human_triggered = True
            event.confidence_score = base_confidence * 0.8  # Lower confidence for isolated events
    
    def _create_movement_sequence(self, events: List[SensorEvent]) -> Optional[MovementSequence]:
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
            duration_seconds=(sequence_events[-1].timestamp - sequence_events[0].timestamp).total_seconds(),
            rooms_visited={e.room_id for e in sequence_events},
            sensors_triggered={e.sensor_id for e in sequence_events}
        )
    
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
            'total_processed': 0,
            'valid_events': 0,
            'invalid_events': 0,
            'human_classified': 0,
            'cat_classified': 0,
            'duplicates_filtered': 0
        }
    
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
                'valid': False,
                'error': f'Room {room_id} not found in configuration'
            }
        
        entity_ids = room_config.get_all_entity_ids()
        if not entity_ids:
            return {
                'valid': False,
                'error': f'No entities configured for room {room_id}'
            }
        
        # Check for required sensor types
        has_presence = bool(room_config.get_sensors_by_type('presence'))
        has_motion = bool(room_config.get_sensors_by_type('motion'))
        
        warnings = []
        if not (has_presence or has_motion):
            warnings.append('No presence or motion sensors configured')
        
        return {
            'valid': True,
            'entity_count': len(entity_ids),
            'sensor_types': list(room_config.sensors.keys()),
            'warnings': warnings
        }