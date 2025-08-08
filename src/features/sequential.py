"""
Sequential feature extraction for occupancy prediction.

This module extracts movement and transition patterns from sensor sequences,
including room transitions, movement velocity, sensor triggering patterns,
and human vs cat movement classification features.
"""

from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import statistics

from ..core.config import RoomConfig, SystemConfig
from ..core.constants import (
    CAT_MOVEMENT_PATTERNS,
    HUMAN_MOVEMENT_PATTERNS,
    MAX_SEQUENCE_GAP,
    MIN_EVENT_SEPARATION,
    SensorType,
)
from ..core.exceptions import FeatureExtractionError
from ..data.ingestion.event_processor import MovementPatternClassifier, MovementSequence
from ..data.storage.models import SensorEvent

logger = logging.getLogger(__name__)


class SequentialFeatureExtractor:
    """
    Extract movement and transition patterns from sensor event sequences.

    This extractor focuses on sequential patterns including:
    - Room transition sequences (n-grams)
    - Movement velocity and direction patterns
    - Sensor triggering order and timing
    - Cross-room correlation patterns
    - Movement classification features (human vs cat)
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the sequential feature extractor.

        Args:
            config: System configuration with room definitions
        """
        self.config = config
        self.classifier = MovementPatternClassifier(config) if config else None
        self.sequence_cache = {}

    def extract_features(
        self,
        events: List[SensorEvent],
        target_time: datetime,
        room_configs: Optional[Dict[str, RoomConfig]] = None,
        lookback_hours: int = 24,
    ) -> Dict[str, float]:
        """
        Extract sequential features from sensor events.

        Args:
            events: Chronologically ordered sensor events
            target_time: Time for which to extract features
            room_configs: Room configuration mapping
            lookback_hours: How far back to look for patterns

        Returns:
            Dictionary of sequential features

        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            if not events:
                return self._get_default_features()

            # Filter events within lookback window
            cutoff_time = target_time - timedelta(hours=lookback_hours)
            recent_events = [e for e in events if e.timestamp >= cutoff_time]

            if not recent_events:
                return self._get_default_features()

            # Sort by timestamp
            sorted_events = sorted(recent_events, key=lambda e: e.timestamp)

            features = {}

            # Room transition features
            features.update(self._extract_room_transition_features(sorted_events))

            # Movement velocity features
            features.update(self._extract_velocity_features(sorted_events))

            # Sensor sequence features
            features.update(self._extract_sensor_sequence_features(sorted_events))

            # Cross-room correlation features
            features.update(self._extract_cross_room_features(sorted_events))

            # Movement pattern classification features
            if self.classifier and room_configs:
                features.update(
                    self._extract_movement_classification_features(
                        sorted_events, room_configs
                    )
                )

            # N-gram pattern features
            features.update(self._extract_ngram_features(sorted_events))

            return features

        except Exception as e:
            logger.error(f"Failed to extract sequential features: {e}")
            raise FeatureExtractionError(f"Sequential feature extraction failed: {e}")

    def _extract_room_transition_features(
        self, events: List[SensorEvent]
    ) -> Dict[str, float]:
        """Extract room transition pattern features."""
        features = {}

        if len(events) < 2:
            return {
                "room_transition_count": 0.0,
                "unique_rooms_visited": 1.0,
                "room_revisit_ratio": 0.0,
                "avg_room_dwell_time": 1800.0,
                "max_room_sequence_length": 1.0,
            }

        # Track room transitions
        room_sequence = []
        room_dwell_times = defaultdict(list)
        current_room = None
        room_start_time = None

        for event in events:
            if event.room_id != current_room:
                # Room transition detected
                if current_room is not None and room_start_time is not None:
                    dwell_time = (event.timestamp - room_start_time).total_seconds()
                    room_dwell_times[current_room].append(dwell_time)

                room_sequence.append(event.room_id)
                current_room = event.room_id
                room_start_time = event.timestamp

        # Add final room dwell time
        if current_room and room_start_time and events:
            final_dwell = (events[-1].timestamp - room_start_time).total_seconds()
            room_dwell_times[current_room].append(final_dwell)

        # Calculate features
        features["room_transition_count"] = (
            len(room_sequence) - 1 if len(room_sequence) > 1 else 0.0
        )
        features["unique_rooms_visited"] = (
            len(set(room_sequence)) if room_sequence else 1.0
        )

        # Room revisit ratio
        room_visits = Counter(room_sequence)
        total_visits = sum(room_visits.values())
        revisits = sum(count - 1 for count in room_visits.values() if count > 1)
        features["room_revisit_ratio"] = (
            revisits / total_visits if total_visits > 0 else 0.0
        )

        # Average room dwell time
        all_dwell_times = []
        for room_dwells in room_dwell_times.values():
            all_dwell_times.extend(room_dwells)
        features["avg_room_dwell_time"] = (
            statistics.mean(all_dwell_times) if all_dwell_times else 1800.0
        )

        # Maximum consecutive room sequence length
        max_sequence = 1
        current_sequence = 1
        for i in range(1, len(room_sequence)):
            if room_sequence[i] == room_sequence[i - 1]:
                current_sequence += 1
                max_sequence = max(max_sequence, current_sequence)
            else:
                current_sequence = 1
        features["max_room_sequence_length"] = max_sequence

        return features

    def _extract_velocity_features(self, events: List[SensorEvent]) -> Dict[str, float]:
        """Extract movement velocity and timing features."""
        features = {}

        if len(events) < 2:
            return {
                "avg_event_interval": 300.0,
                "min_event_interval": 60.0,
                "max_event_interval": 3600.0,
                "event_interval_variance": 0.0,
                "movement_velocity_score": 0.5,
            }

        # Calculate intervals between consecutive events
        intervals = []
        for i in range(1, len(events)):
            interval = (events[i].timestamp - events[i - 1].timestamp).total_seconds()
            intervals.append(interval)

        # Interval statistics
        features["avg_event_interval"] = statistics.mean(intervals)
        features["min_event_interval"] = min(intervals)
        features["max_event_interval"] = max(intervals)
        features["event_interval_variance"] = (
            statistics.variance(intervals) if len(intervals) > 1 else 0.0
        )

        # Movement velocity score (inverse of average interval, normalized)
        avg_interval = features["avg_event_interval"]
        # Higher score for faster movement (shorter intervals)
        features["movement_velocity_score"] = min(1.0, 300.0 / max(avg_interval, 30.0))

        # Burst detection (rapid sequence of events)
        burst_threshold = 30.0  # seconds
        burst_count = sum(1 for interval in intervals if interval < burst_threshold)
        features["burst_ratio"] = burst_count / len(intervals) if intervals else 0.0

        # Pause detection (long intervals)
        pause_threshold = 600.0  # 10 minutes
        pause_count = sum(1 for interval in intervals if interval > pause_threshold)
        features["pause_ratio"] = pause_count / len(intervals) if intervals else 0.0

        return features

    def _extract_sensor_sequence_features(
        self, events: List[SensorEvent]
    ) -> Dict[str, float]:
        """Extract sensor triggering sequence features."""
        features = {}

        if not events:
            return {
                "unique_sensors_triggered": 1.0,
                "sensor_revisit_count": 0.0,
                "dominant_sensor_ratio": 1.0,
                "sensor_diversity_score": 0.0,
            }

        # Sensor usage patterns
        sensor_sequence = [event.sensor_id for event in events]
        sensor_counts = Counter(sensor_sequence)

        features["unique_sensors_triggered"] = len(sensor_counts)

        # Sensor revisit count
        revisits = sum(count - 1 for count in sensor_counts.values() if count > 1)
        features["sensor_revisit_count"] = revisits

        # Dominant sensor ratio
        most_common_count = sensor_counts.most_common(1)[0][1] if sensor_counts else 0
        features["dominant_sensor_ratio"] = most_common_count / len(sensor_sequence)

        # Sensor diversity score (entropy-based)
        total_events = len(sensor_sequence)
        entropy = 0.0
        for count in sensor_counts.values():
            p = count / total_events
            entropy -= p * np.log2(p) if p > 0 else 0.0

        max_entropy = np.log2(len(sensor_counts)) if len(sensor_counts) > 1 else 1.0
        features["sensor_diversity_score"] = (
            entropy / max_entropy if max_entropy > 0 else 0.0
        )

        # Sensor type distribution
        sensor_types = [event.sensor_type for event in events]
        type_counts = Counter(sensor_types)
        total_types = len(sensor_types)

        # Presence/motion sensor ratio
        presence_count = type_counts.get("presence", 0) + type_counts.get("motion", 0)
        features["presence_sensor_ratio"] = (
            presence_count / total_types if total_types > 0 else 0.0
        )

        # Door sensor ratio
        door_count = type_counts.get("door", 0)
        features["door_sensor_ratio"] = (
            door_count / total_types if total_types > 0 else 0.0
        )

        return features

    def _extract_cross_room_features(
        self, events: List[SensorEvent]
    ) -> Dict[str, float]:
        """Extract cross-room correlation and pattern features."""
        features = {}

        # Group events by room
        room_events = defaultdict(list)
        for event in events:
            room_events[event.room_id].append(event)

        room_count = len(room_events)
        features["active_room_count"] = room_count

        if room_count < 2:
            return {
                "active_room_count": room_count,
                "room_correlation_score": 0.0,
                "multi_room_sequence_ratio": 0.0,
                "room_switching_frequency": 0.0,
            }

        # Room correlation (events in multiple rooms within short time windows)
        correlation_windows = []
        window_size = 300  # 5 minutes

        for event in events:
            window_start = event.timestamp - timedelta(seconds=window_size)
            window_events = [
                e for e in events if window_start <= e.timestamp <= event.timestamp
            ]

            window_rooms = set(e.room_id for e in window_events)
            if len(window_rooms) > 1:
                correlation_windows.append(len(window_rooms))

        features["room_correlation_score"] = (
            statistics.mean(correlation_windows) / room_count
            if correlation_windows
            else 0.0
        )

        # Multi-room sequence ratio
        room_sequence = [event.room_id for event in events]
        multi_room_sequences = 0
        for i in range(1, len(room_sequence)):
            if room_sequence[i] != room_sequence[i - 1]:
                multi_room_sequences += 1

        features["multi_room_sequence_ratio"] = (
            multi_room_sequences / len(room_sequence) if room_sequence else 0.0
        )

        # Room switching frequency (switches per hour)
        if events and len(events) > 1:
            duration_hours = (
                events[-1].timestamp - events[0].timestamp
            ).total_seconds() / 3600
            features["room_switching_frequency"] = (
                multi_room_sequences / duration_hours if duration_hours > 0 else 0.0
            )
        else:
            features["room_switching_frequency"] = 0.0

        return features

    def _extract_movement_classification_features(
        self, events: List[SensorEvent], room_configs: Dict[str, RoomConfig]
    ) -> Dict[str, float]:
        """Extract features for human vs cat movement classification."""
        features = {}

        if not events or not self.classifier:
            return {
                "human_movement_probability": 0.5,
                "cat_movement_probability": 0.5,
                "movement_confidence_score": 0.5,
                "door_interaction_ratio": 0.0,
            }

        # Create movement sequences for classification
        sequences = self._create_sequences_for_classification(events, room_configs)

        if not sequences:
            return {
                "human_movement_probability": 0.5,
                "cat_movement_probability": 0.5,
                "movement_confidence_score": 0.5,
                "door_interaction_ratio": 0.0,
            }

        # Classify each sequence and aggregate results
        human_scores = []
        cat_scores = []
        confidence_scores = []
        door_interactions = 0
        total_events = 0

        for sequence in sequences:
            room_config = room_configs.get(sequence.events[0].room_id)
            if room_config:
                classification = self.classifier.classify_movement(
                    sequence, room_config
                )

                if classification.is_human_triggered:
                    human_scores.append(classification.confidence_score)
                else:
                    cat_scores.append(classification.confidence_score)

                confidence_scores.append(classification.confidence_score)

                # Count door interactions
                door_sensors = room_config.get_sensors_by_type("door")
                door_entity_ids = set(door_sensors.values()) if door_sensors else set()

                for event in sequence.events:
                    total_events += 1
                    if event.sensor_id in door_entity_ids:
                        door_interactions += 1

        # Aggregate classification results
        total_sequences = len(sequences)
        human_probability = (
            len(human_scores) / total_sequences if total_sequences > 0 else 0.5
        )
        cat_probability = (
            len(cat_scores) / total_sequences if total_sequences > 0 else 0.5
        )

        features["human_movement_probability"] = human_probability
        features["cat_movement_probability"] = cat_probability
        features["movement_confidence_score"] = (
            statistics.mean(confidence_scores) if confidence_scores else 0.5
        )
        features["door_interaction_ratio"] = (
            door_interactions / total_events if total_events > 0 else 0.0
        )

        return features

    def _extract_ngram_features(self, events: List[SensorEvent]) -> Dict[str, float]:
        """Extract n-gram pattern features from sensor sequences."""
        features = {}

        if len(events) < 3:
            return {
                "common_bigram_ratio": 0.0,
                "common_trigram_ratio": 0.0,
                "pattern_repetition_score": 0.0,
            }

        # Extract sensor ID sequences
        sensor_sequence = [event.sensor_id for event in events]

        # Generate bigrams
        bigrams = []
        for i in range(len(sensor_sequence) - 1):
            bigrams.append((sensor_sequence[i], sensor_sequence[i + 1]))

        # Generate trigrams
        trigrams = []
        for i in range(len(sensor_sequence) - 2):
            trigrams.append(
                (sensor_sequence[i], sensor_sequence[i + 1], sensor_sequence[i + 2])
            )

        # Calculate most common pattern ratios
        if bigrams:
            bigram_counts = Counter(bigrams)
            most_common_bigram = bigram_counts.most_common(1)[0][1]
            features["common_bigram_ratio"] = most_common_bigram / len(bigrams)
        else:
            features["common_bigram_ratio"] = 0.0

        if trigrams:
            trigram_counts = Counter(trigrams)
            most_common_trigram = trigram_counts.most_common(1)[0][1]
            features["common_trigram_ratio"] = most_common_trigram / len(trigrams)
        else:
            features["common_trigram_ratio"] = 0.0

        # Pattern repetition score
        pattern_score = (
            features["common_bigram_ratio"] + features["common_trigram_ratio"]
        ) / 2
        features["pattern_repetition_score"] = pattern_score

        return features

    def _create_sequences_for_classification(
        self, events: List[SensorEvent], room_configs: Dict[str, RoomConfig]
    ) -> List[MovementSequence]:
        """Create movement sequences from events for classification."""
        sequences = []

        # Group events by room and time windows
        for room_id in set(event.room_id for event in events):
            room_events = [e for e in events if e.room_id == room_id]

            if len(room_events) < 2:
                continue

            # Create sequences with time gaps
            current_sequence = []

            for event in room_events:
                if (
                    not current_sequence
                    or (
                        event.timestamp - current_sequence[-1].timestamp
                    ).total_seconds()
                    <= MAX_SEQUENCE_GAP
                ):
                    current_sequence.append(event)
                else:
                    # End current sequence, start new one
                    if len(current_sequence) >= 2:
                        sequences.append(
                            self._create_movement_sequence(current_sequence)
                        )
                    current_sequence = [event]

            # Add final sequence
            if len(current_sequence) >= 2:
                sequences.append(self._create_movement_sequence(current_sequence))

        return [seq for seq in sequences if seq is not None]

    def _create_movement_sequence(
        self, events: List[SensorEvent]
    ) -> Optional[MovementSequence]:
        """Create a MovementSequence from sensor events."""
        if len(events) < 2:
            return None

        return MovementSequence(
            events=events,
            start_time=events[0].timestamp,
            end_time=events[-1].timestamp,
            duration_seconds=(
                events[-1].timestamp - events[0].timestamp
            ).total_seconds(),
            rooms_visited={e.room_id for e in events},
            sensors_triggered={e.sensor_id for e in events},
        )

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no events are available."""
        return {
            "room_transition_count": 0.0,
            "unique_rooms_visited": 1.0,
            "room_revisit_ratio": 0.0,
            "avg_room_dwell_time": 1800.0,
            "max_room_sequence_length": 1.0,
            "avg_event_interval": 300.0,
            "min_event_interval": 60.0,
            "max_event_interval": 3600.0,
            "event_interval_variance": 0.0,
            "movement_velocity_score": 0.5,
            "burst_ratio": 0.0,
            "pause_ratio": 0.0,
            "unique_sensors_triggered": 1.0,
            "sensor_revisit_count": 0.0,
            "dominant_sensor_ratio": 1.0,
            "sensor_diversity_score": 0.0,
            "presence_sensor_ratio": 0.5,
            "door_sensor_ratio": 0.0,
            "active_room_count": 1.0,
            "room_correlation_score": 0.0,
            "multi_room_sequence_ratio": 0.0,
            "room_switching_frequency": 0.0,
            "human_movement_probability": 0.5,
            "cat_movement_probability": 0.5,
            "movement_confidence_score": 0.5,
            "door_interaction_ratio": 0.0,
            "common_bigram_ratio": 0.0,
            "common_trigram_ratio": 0.0,
            "pattern_repetition_score": 0.0,
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all sequential feature names."""
        return list(self._get_default_features().keys())

    def clear_cache(self):
        """Clear the sequence cache."""
        self.sequence_cache.clear()
