"""
Sequential feature extraction for occupancy prediction.

This module extracts movement and transition patterns from sensor sequences,
including room transitions, movement velocity, sensor triggering patterns,
and human vs cat movement classification features.
"""

from collections import Counter, defaultdict, deque
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Set

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
from ..data.ingestion.event_processor import (
    MovementPatternClassifier,
    MovementSequence,
)
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
            else:
                # Provide default classification features when no classifier/config
                features.update(
                    {
                        "human_movement_probability": 0.5,
                        "cat_movement_probability": 0.5,
                        "movement_confidence_score": 0.5,
                        "door_interaction_ratio": 0.0,
                        "pattern_matches_human": 0.0,
                        "pattern_matches_cat": 0.0,
                        "velocity_classification": 0.0,
                        "sequence_length_score": 0.0,
                    }
                )

            # N-gram pattern features
            features.update(self._extract_ngram_features(sorted_events))

            return features

        except Exception as e:
            logger.error(f"Failed to extract sequential features: {e}")
            # Extract room_id from events if available
            room_id = events[0].room_id if events else "unknown"
            raise FeatureExtractionError(
                feature_type="sequential", room_id=room_id, cause=e
            )

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
        """Extract movement velocity and timing features using advanced mathematical analysis."""
        features = {}

        if len(events) < 2:
            return {
                "avg_event_interval": 300.0,
                "min_event_interval": 60.0,
                "max_event_interval": 3600.0,
                "event_interval_variance": 0.0,
                "movement_velocity_score": 0.5,
                "burst_ratio": 0.0,
                "pause_ratio": 0.0,
                "velocity_acceleration": 0.0,
                "interval_autocorr": 0.0,
                "velocity_entropy": 0.0,
                "movement_regularity": 0.0,
            }

        # Calculate intervals between consecutive events using numpy
        timestamps = [event.timestamp.timestamp() for event in events]
        intervals = np.diff(timestamps)  # More efficient than loop

        # Basic interval statistics using numpy
        features["avg_event_interval"] = float(np.mean(intervals))
        features["min_event_interval"] = float(np.min(intervals))
        features["max_event_interval"] = float(np.max(intervals))
        features["event_interval_variance"] = float(np.var(intervals))

        # Movement velocity score (inverse of average interval, normalized)
        avg_interval = features["avg_event_interval"]
        features["movement_velocity_score"] = min(1.0, 300.0 / max(avg_interval, 30.0))

        # Burst and pause detection with numpy
        burst_threshold = 30.0  # seconds
        pause_threshold = 600.0  # 10 minutes

        burst_mask = intervals < burst_threshold
        pause_mask = intervals > pause_threshold

        features["burst_ratio"] = float(np.mean(burst_mask))
        features["pause_ratio"] = float(np.mean(pause_mask))

        # Advanced velocity features
        if len(intervals) > 2:
            # Velocity acceleration (second derivative of intervals)
            velocity_changes = np.diff(intervals)
            features["velocity_acceleration"] = float(np.std(velocity_changes))

            # Interval autocorrelation (pattern regularity)
            if len(intervals) > 3:
                intervals_norm = (intervals - np.mean(intervals)) / np.std(intervals)
                autocorr = np.corrcoef(intervals_norm[:-1], intervals_norm[1:])[0, 1]
                features["interval_autocorr"] = float(
                    autocorr if not np.isnan(autocorr) else 0.0
                )
            else:
                features["interval_autocorr"] = 0.0

            # Velocity entropy (movement unpredictability)
            # Discretize intervals into bins for entropy calculation
            bins = np.linspace(np.min(intervals), np.max(intervals), 10)
            hist, _ = np.histogram(intervals, bins=bins)
            hist = hist + 1  # Add pseudocount to avoid log(0)
            probs = hist / np.sum(hist)
            entropy = -np.sum(probs * np.log2(probs))
            features["velocity_entropy"] = float(entropy)

            # Movement regularity (coefficient of variation)
            cv = (
                np.std(intervals) / np.mean(intervals)
                if np.mean(intervals) > 0
                else 0.0
            )
            features["movement_regularity"] = float(
                1.0 / (1.0 + cv)
            )  # Inverse so higher = more regular
        else:
            features["velocity_acceleration"] = 0.0
            features["interval_autocorr"] = 0.0
            features["velocity_entropy"] = 0.0
            features["movement_regularity"] = 0.5

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
        """Extract cross-room correlation and pattern features using advanced mathematical analysis."""
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
                "room_activity_entropy": 0.0,
                "spatial_clustering_score": 0.0,
                "room_transition_predictability": 0.0,
            }

        # Room correlation using efficient sliding window
        correlation_windows = []
        window_size = 300  # 5 minutes
        window_events = deque()

        for event in events:
            window_start = event.timestamp - timedelta(seconds=window_size)
            while window_events and window_events[0].timestamp < window_start:
                window_events.popleft()
            window_events.append(event)

            window_rooms = set(e.room_id for e in window_events)
            if len(window_rooms) > 1:
                correlation_windows.append(len(window_rooms))

        features["room_correlation_score"] = (
            statistics.mean(correlation_windows) / room_count
            if correlation_windows
            else 0.0
        )

        # Multi-room sequence analysis using numpy
        room_sequence = np.array([event.room_id for event in events])

        # Room transitions (where consecutive events are in different rooms)
        if len(room_sequence) > 1:
            transitions = room_sequence[1:] != room_sequence[:-1]
            multi_room_sequences = np.sum(transitions)
            features["multi_room_sequence_ratio"] = float(
                multi_room_sequences / len(room_sequence)
            )

            # Room switching frequency (switches per hour)
            if len(events) > 1:
                duration_hours = (
                    events[-1].timestamp - events[0].timestamp
                ).total_seconds() / 3600
                features["room_switching_frequency"] = (
                    float(multi_room_sequences) / duration_hours
                    if duration_hours > 0
                    else 0.0
                )
            else:
                features["room_switching_frequency"] = 0.0
        else:
            features["multi_room_sequence_ratio"] = 0.0
            features["room_switching_frequency"] = 0.0

        # Advanced cross-room features using numpy

        # Room activity entropy (how evenly distributed activity is across rooms)
        room_activity_counts = np.array(
            [len(events) for events in room_events.values()]
        )
        if len(room_activity_counts) > 1:
            room_probs = room_activity_counts / np.sum(room_activity_counts)
            entropy = -np.sum(
                room_probs * np.log2(room_probs + 1e-10)
            )  # Add small epsilon
            max_entropy = np.log2(len(room_activity_counts))
            features["room_activity_entropy"] = float(
                entropy / max_entropy if max_entropy > 0 else 0.0
            )
        else:
            features["room_activity_entropy"] = 0.0

        # Spatial clustering score (how clustered room visits are in time)
        if len(room_sequence) > 2:
            # Calculate runs of same room
            room_runs = []
            current_room = room_sequence[0]
            run_length = 1

            for i in range(1, len(room_sequence)):
                if room_sequence[i] == current_room:
                    run_length += 1
                else:
                    room_runs.append(run_length)
                    current_room = room_sequence[i]
                    run_length = 1
            room_runs.append(run_length)

            run_array = np.array(room_runs)
            avg_run_length = np.mean(run_array)
            max_possible_run = len(room_sequence) / room_count
            features["spatial_clustering_score"] = float(
                avg_run_length / max_possible_run if max_possible_run > 0 else 0.0
            )
        else:
            features["spatial_clustering_score"] = 0.0

        # Room transition predictability using transition matrix
        if len(room_sequence) > 3:
            unique_rooms = list(set(room_sequence))
            room_to_idx = {room: i for i, room in enumerate(unique_rooms)}
            n_rooms = len(unique_rooms)

            # Build transition matrix
            transition_matrix = np.zeros((n_rooms, n_rooms))
            for i in range(len(room_sequence) - 1):
                from_idx = room_to_idx[room_sequence[i]]
                to_idx = room_to_idx[room_sequence[i + 1]]
                transition_matrix[from_idx, to_idx] += 1

            # Normalize rows to get probabilities
            row_sums = np.sum(transition_matrix, axis=1, keepdims=True)
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            transition_probs = transition_matrix / row_sums

            # Calculate predictability as average maximum transition probability
            max_probs = np.max(transition_probs, axis=1)
            features["room_transition_predictability"] = float(np.mean(max_probs))
        else:
            features["room_transition_predictability"] = 0.0

        return features

    def _extract_movement_classification_features(
        self, events: List[SensorEvent], room_configs: Dict[str, RoomConfig]
    ) -> Dict[str, float]:
        """Extract features for human vs cat movement classification using pattern constants."""
        features = {}

        if not events or not self.classifier:
            return {
                "human_movement_probability": 0.5,
                "cat_movement_probability": 0.5,
                "movement_confidence_score": 0.5,
                "door_interaction_ratio": 0.0,
                "pattern_matches_human": 0.0,
                "pattern_matches_cat": 0.0,
                "velocity_classification": 0.0,
                "sequence_length_score": 0.0,
            }

        # Create movement sequences for classification
        sequences = self._create_sequences_for_classification(events, room_configs)

        if not sequences:
            return {
                "human_movement_probability": 0.5,
                "cat_movement_probability": 0.5,
                "movement_confidence_score": 0.5,
                "door_interaction_ratio": 0.0,
                "pattern_matches_human": 0.0,
                "pattern_matches_cat": 0.0,
                "velocity_classification": 0.0,
                "sequence_length_score": 0.0,
            }

        # Track unique patterns using Set for pattern analysis
        unique_human_patterns: Set[str] = set()
        unique_cat_patterns: Set[str] = set()

        # Classification results
        human_scores = []
        cat_scores = []
        confidence_scores = []
        door_interactions = 0
        total_events = 0

        # Pattern matching scores
        human_pattern_matches = 0
        cat_pattern_matches = 0
        velocity_scores = []
        sequence_lengths = []

        for sequence in sequences:
            room_config = room_configs.get(sequence.events[0].room_id)
            if room_config:
                # Use classifier if available
                classification = self.classifier.classify_movement(
                    sequence, room_config
                )

                if classification.is_human_triggered:
                    human_scores.append(classification.confidence_score)
                    pattern_key = (
                        f"human_{len(sequence.events)}_{sequence.duration_seconds:.0f}"
                    )
                    unique_human_patterns.add(pattern_key)
                else:
                    cat_scores.append(classification.confidence_score)
                    pattern_key = (
                        f"cat_{len(sequence.events)}_{sequence.duration_seconds:.0f}"
                    )
                    unique_cat_patterns.add(pattern_key)

                confidence_scores.append(classification.confidence_score)

                # Count door interactions using SensorType filtering
                door_sensors = room_config.get_sensors_by_type(SensorType.DOOR.value)
                door_entity_ids: Set[str] = (
                    set(door_sensors.values()) if door_sensors else set()
                )

                sequence_door_interactions = 0
                for event in sequence.events:
                    total_events += 1
                    if event.sensor_id in door_entity_ids:
                        door_interactions += 1
                        sequence_door_interactions += 1

                # Pattern matching against movement constants
                duration = sequence.duration_seconds
                avg_velocity = len(sequence.events) / max(
                    duration, 1.0
                )  # events per second
                room_count = len(sequence.rooms_visited)
                door_ratio = sequence_door_interactions / len(sequence.events)

                # Compare against HUMAN_MOVEMENT_PATTERNS
                human_duration_match = (
                    duration >= HUMAN_MOVEMENT_PATTERNS["min_duration_seconds"]
                )
                human_velocity_match = (
                    avg_velocity <= HUMAN_MOVEMENT_PATTERNS["max_velocity_ms"]
                )
                human_sequence_match = (
                    room_count
                    <= HUMAN_MOVEMENT_PATTERNS["typical_room_sequence_length"]
                )
                human_door_match = door_ratio >= (
                    HUMAN_MOVEMENT_PATTERNS["door_interaction_probability"] * 0.5
                )

                if (
                    sum(
                        [
                            human_duration_match,
                            human_velocity_match,
                            human_sequence_match,
                            human_door_match,
                        ]
                    )
                    >= 3
                ):
                    human_pattern_matches += 1

                # Compare against CAT_MOVEMENT_PATTERNS
                cat_duration_match = (
                    duration >= CAT_MOVEMENT_PATTERNS["min_duration_seconds"]
                )
                cat_velocity_match = (
                    avg_velocity <= CAT_MOVEMENT_PATTERNS["max_velocity_ms"]
                )
                cat_sequence_match = (
                    room_count <= CAT_MOVEMENT_PATTERNS["typical_room_sequence_length"]
                )
                cat_door_match = (
                    door_ratio <= CAT_MOVEMENT_PATTERNS["door_interaction_probability"]
                )

                if (
                    sum(
                        [
                            cat_duration_match,
                            cat_velocity_match,
                            cat_sequence_match,
                            cat_door_match,
                        ]
                    )
                    >= 3
                ):
                    cat_pattern_matches += 1

                velocity_scores.append(avg_velocity)
                sequence_lengths.append(room_count)

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

        # Advanced pattern matching features
        features["pattern_matches_human"] = (
            human_pattern_matches / total_sequences if total_sequences > 0 else 0.0
        )
        features["pattern_matches_cat"] = (
            cat_pattern_matches / total_sequences if total_sequences > 0 else 0.0
        )
        features["velocity_classification"] = (
            statistics.mean(velocity_scores) if velocity_scores else 0.0
        )
        features["sequence_length_score"] = (
            statistics.mean(sequence_lengths) if sequence_lengths else 0.0
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
                (
                    sensor_sequence[i],
                    sensor_sequence[i + 1],
                    sensor_sequence[i + 2],
                )
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

            # Create sequences with time gaps using MIN_EVENT_SEPARATION filtering
            current_sequence = []

            for event in room_events:
                # Filter events too close together using MIN_EVENT_SEPARATION
                if (
                    current_sequence
                    and (
                        event.timestamp - current_sequence[-1].timestamp
                    ).total_seconds()
                    < MIN_EVENT_SEPARATION
                ):
                    continue  # Skip event that's too close to previous one

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
            # Advanced velocity features
            "velocity_acceleration": 0.0,
            "interval_autocorr": 0.0,
            "velocity_entropy": 0.0,
            "movement_regularity": 0.0,
            # Advanced cross-room features
            "room_activity_entropy": 0.0,
            "spatial_clustering_score": 0.0,
            "room_transition_predictability": 0.0,
            # Enhanced movement classification features
            "pattern_matches_human": 0.0,
            "pattern_matches_cat": 0.0,
            "velocity_classification": 0.0,
            "sequence_length_score": 0.0,
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all sequential feature names."""
        return list(self._get_default_features().keys())

    def clear_cache(self):
        """Clear the sequence cache."""
        self.sequence_cache.clear()
