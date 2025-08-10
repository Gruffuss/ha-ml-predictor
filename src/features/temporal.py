"""
Temporal feature extraction for occupancy prediction.

This module extracts time-based features from sensor event sequences, including
cyclical encodings, historical patterns, and state duration metrics.
"""

from collections import defaultdict
from datetime import datetime, timedelta
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statistics

from ..core.constants import TEMPORAL_FEATURE_NAMES
from ..core.exceptions import FeatureExtractionError
from ..data.storage.models import RoomState, SensorEvent

logger = logging.getLogger(__name__)


class TemporalFeatureExtractor:
    """
    Extract time-based features from sensor event sequences.

    This extractor focuses on temporal patterns including:
    - Time since last state changes
    - Duration in current states
    - Cyclical time encodings (sin/cos transformations)
    - Historical occupancy patterns
    - State transition timing patterns
    """

    def __init__(self, timezone_offset: int = 0):
        """
        Initialize the temporal feature extractor.

        Args:
            timezone_offset: Offset from UTC in hours (e.g., -8 for PST)
        """
        self.timezone_offset = timezone_offset
        self.feature_cache = {}

    def extract_features(
        self,
        events: List[SensorEvent],
        target_time: datetime,
        room_states: Optional[List[RoomState]] = None,
    ) -> Dict[str, float]:
        """
        Extract temporal features for a specific target time.

        Args:
            events: Chronologically ordered sensor events
            target_time: Time for which to extract features
            room_states: Optional room state history

        Returns:
            Dictionary of temporal features

        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            if not events:
                return self._get_default_features()

            # Sort events by timestamp to ensure chronological order
            sorted_events = sorted(events, key=lambda e: e.timestamp)

            features = {}

            # Time-based features
            features.update(
                self._extract_time_since_features(sorted_events, target_time)
            )
            features.update(self._extract_duration_features(sorted_events, target_time))
            features.update(self._extract_cyclical_features(target_time))
            features.update(
                self._extract_historical_patterns(sorted_events, target_time)
            )

            # State transition timing features
            features.update(
                self._extract_transition_timing_features(sorted_events, target_time)
            )

            # Room state duration features if available
            if room_states:
                features.update(
                    self._extract_room_state_features(room_states, target_time)
                )

            return features

        except Exception as e:
            logger.error(f"Failed to extract temporal features: {e}")
            raise FeatureExtractionError(f"Temporal feature extraction failed: {e}")

    def _extract_time_since_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract time-since-last-event features."""
        features = {}

        if not events:
            return {
                "time_since_last_event": 3600.0,  # Default 1 hour
                "time_since_last_on": 3600.0,
                "time_since_last_of": 3600.0,
                "time_since_last_motion": 3600.0,
            }

        # Get most recent event
        last_event = events[-1]
        time_since_last = (target_time - last_event.timestamp).total_seconds()
        features["time_since_last_event"] = min(
            time_since_last, 86400.0
        )  # Cap at 24 hours

        # Time since last state changes
        last_on_time = None
        last_off_time = None
        last_motion_time = None

        for event in reversed(events):
            if event.state == "on" and last_on_time is None:
                last_on_time = event.timestamp
            elif event.state == "of" and last_off_time is None:
                last_off_time = event.timestamp

            if event.sensor_type in ["motion", "presence"] and last_motion_time is None:
                last_motion_time = event.timestamp

            # Stop if we have all we need
            if all([last_on_time, last_off_time, last_motion_time]):
                break

        # Calculate time since features
        features["time_since_last_on"] = (
            (target_time - last_on_time).total_seconds() if last_on_time else 3600.0
        )
        features["time_since_last_of"] = (
            (target_time - last_off_time).total_seconds() if last_off_time else 3600.0
        )
        features["time_since_last_motion"] = (
            (target_time - last_motion_time).total_seconds()
            if last_motion_time
            else 3600.0
        )

        # Cap all values at 24 hours
        for key in [
            "time_since_last_on",
            "time_since_last_of",
            "time_since_last_motion",
        ]:
            features[key] = min(features[key], 86400.0)

        return features

    def _extract_duration_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract state duration features."""
        features = {}

        if not events:
            return {
                "current_state_duration": 0.0,
                "avg_on_duration": 1800.0,  # Default 30 minutes
                "avg_off_duration": 1800.0,
                "max_on_duration": 3600.0,
                "max_off_duration": 3600.0,
            }

        # Current state duration
        last_event = events[-1]
        current_duration = (target_time - last_event.timestamp).total_seconds()
        features["current_state_duration"] = min(current_duration, 86400.0)

        # Calculate historical state durations
        on_durations = []
        off_durations = []

        current_state = None
        state_start_time = None

        for event in events:
            if current_state != event.state:
                # State change detected
                if current_state is not None and state_start_time is not None:
                    duration = (event.timestamp - state_start_time).total_seconds()
                    if current_state == "on":
                        on_durations.append(duration)
                    elif current_state == "of":
                        off_durations.append(duration)

                current_state = event.state
                state_start_time = event.timestamp

        # Statistical features for durations
        features["avg_on_duration"] = (
            statistics.mean(on_durations) if on_durations else 1800.0
        )
        features["avg_off_duration"] = (
            statistics.mean(off_durations) if off_durations else 1800.0
        )
        features["max_on_duration"] = max(on_durations) if on_durations else 3600.0
        features["max_off_duration"] = max(off_durations) if off_durations else 3600.0

        return features

    def _extract_cyclical_features(self, target_time: datetime) -> Dict[str, float]:
        """Extract cyclical time encodings using sin/cos transformations."""
        # Adjust for timezone
        local_time = target_time + timedelta(hours=self.timezone_offset)

        features = {}

        # Hour of day (0-23)
        hour = local_time.hour
        features["hour_sin"] = math.sin(2 * math.pi * hour / 24)
        features["hour_cos"] = math.cos(2 * math.pi * hour / 24)

        # Day of week (0-6, Monday=0)
        day_of_week = local_time.weekday()
        features["day_sin"] = math.sin(2 * math.pi * day_of_week / 7)
        features["day_cos"] = math.cos(2 * math.pi * day_of_week / 7)

        # Month of year (1-12)
        month = local_time.month
        features["month_sin"] = math.sin(2 * math.pi * month / 12)
        features["month_cos"] = math.cos(2 * math.pi * month / 12)

        # Day of month (1-31) - normalized
        day = local_time.day
        features["day_of_month_sin"] = math.sin(2 * math.pi * day / 31)
        features["day_of_month_cos"] = math.cos(2 * math.pi * day / 31)

        # Weekend indicator
        features["is_weekend"] = 1.0 if day_of_week >= 5 else 0.0

        # Work hours indicator (9 AM - 5 PM)
        features["is_work_hours"] = 1.0 if 9 <= hour <= 17 else 0.0

        # Sleep hours indicator (10 PM - 6 AM)
        features["is_sleep_hours"] = 1.0 if hour >= 22 or hour <= 6 else 0.0

        return features

    def _extract_historical_patterns(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract historical occupancy patterns for similar times."""
        features = {}

        # Adjust for timezone
        local_time = target_time + timedelta(hours=self.timezone_offset)
        current_hour = local_time.hour
        current_day_of_week = local_time.weekday()

        # Group events by hour and day of week
        hourly_patterns = defaultdict(list)
        daily_patterns = defaultdict(list)

        for event in events:
            event_local = event.timestamp + timedelta(hours=self.timezone_offset)
            event_hour = event_local.hour
            event_day = event_local.weekday()

            hourly_patterns[event_hour].append(event.state == "on")
            daily_patterns[event_day].append(event.state == "on")

        # Historical patterns for current hour
        current_hour_events = hourly_patterns[current_hour]
        features["hour_activity_rate"] = (
            sum(current_hour_events) / len(current_hour_events)
            if current_hour_events
            else 0.5
        )

        # Historical patterns for current day of week
        current_day_events = daily_patterns[current_day_of_week]
        features["day_activity_rate"] = (
            sum(current_day_events) / len(current_day_events)
            if current_day_events
            else 0.5
        )

        # Overall activity rate
        all_states = [event.state == "on" for event in events]
        features["overall_activity_rate"] = (
            sum(all_states) / len(all_states) if all_states else 0.5
        )

        # Time-of-day similarity score
        similar_hour_events = []
        for hour in range(max(0, current_hour - 1), min(24, current_hour + 2)):
            similar_hour_events.extend(hourly_patterns[hour])

        features["similar_time_activity_rate"] = (
            sum(similar_hour_events) / len(similar_hour_events)
            if similar_hour_events
            else 0.5
        )

        return features

    def _extract_transition_timing_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract timing patterns around state transitions."""
        features = {}

        if len(events) < 2:
            return {
                "avg_transition_interval": 1800.0,
                "recent_transition_rate": 0.0,
                "time_variability": 0.0,
            }

        # Calculate intervals between consecutive events
        intervals = []
        for i in range(1, len(events)):
            interval = (events[i].timestamp - events[i - 1].timestamp).total_seconds()
            intervals.append(interval)

        # Average transition interval
        features["avg_transition_interval"] = (
            statistics.mean(intervals) if intervals else 1800.0
        )

        # Recent transition rate (transitions per hour in last 4 hours)
        recent_cutoff = target_time - timedelta(hours=4)
        recent_events = [e for e in events if e.timestamp >= recent_cutoff]
        if len(recent_events) > 1:
            recent_duration = (
                recent_events[-1].timestamp - recent_events[0].timestamp
            ).total_seconds()
            features["recent_transition_rate"] = (len(recent_events) - 1) / (
                recent_duration / 3600
            )
        else:
            features["recent_transition_rate"] = 0.0

        # Timing variability (coefficient of variation)
        if len(intervals) > 1:
            mean_interval = statistics.mean(intervals)
            std_interval = statistics.stdev(intervals)
            features["time_variability"] = (
                std_interval / mean_interval if mean_interval > 0 else 0.0
            )
        else:
            features["time_variability"] = 0.0

        return features

    def _extract_room_state_features(
        self, room_states: List[RoomState], target_time: datetime
    ) -> Dict[str, float]:
        """Extract features from room state history."""
        features = {}

        if not room_states:
            return {
                "avg_occupancy_confidence": 0.5,
                "recent_occupancy_ratio": 0.5,
                "state_stability": 0.5,
            }

        # Sort by timestamp
        sorted_states = sorted(room_states, key=lambda s: s.timestamp)

        # Average occupancy confidence
        confidences = [
            state.occupancy_confidence
            for state in sorted_states
            if state.occupancy_confidence is not None
        ]
        features["avg_occupancy_confidence"] = (
            statistics.mean(confidences) if confidences else 0.5
        )

        # Recent occupancy ratio (last 24 hours)
        recent_cutoff = target_time - timedelta(hours=24)
        recent_states = [s for s in sorted_states if s.timestamp >= recent_cutoff]
        if recent_states:
            occupied_count = sum(1 for s in recent_states if s.is_occupied)
            features["recent_occupancy_ratio"] = occupied_count / len(recent_states)
        else:
            features["recent_occupancy_ratio"] = 0.5

        # State stability (average state duration)
        if len(sorted_states) > 1:
            durations = []
            for i in range(1, len(sorted_states)):
                duration = (
                    sorted_states[i].timestamp - sorted_states[i - 1].timestamp
                ).total_seconds()
                durations.append(duration)
            features["state_stability"] = (
                statistics.mean(durations) / 3600.0
            )  # Convert to hours
        else:
            features["state_stability"] = 0.5

        return features

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no events are available."""
        return {
            "time_since_last_event": 3600.0,
            "time_since_last_on": 3600.0,
            "time_since_last_of": 3600.0,
            "time_since_last_motion": 3600.0,
            "current_state_duration": 0.0,
            "avg_on_duration": 1800.0,
            "avg_off_duration": 1800.0,
            "max_on_duration": 3600.0,
            "max_off_duration": 3600.0,
            "hour_sin": 0.0,
            "hour_cos": 1.0,
            "day_sin": 0.0,
            "day_cos": 1.0,
            "month_sin": 0.0,
            "month_cos": 1.0,
            "day_of_month_sin": 0.0,
            "day_of_month_cos": 1.0,
            "is_weekend": 0.0,
            "is_work_hours": 0.0,
            "is_sleep_hours": 0.0,
            "hour_activity_rate": 0.5,
            "day_activity_rate": 0.5,
            "overall_activity_rate": 0.5,
            "similar_time_activity_rate": 0.5,
            "avg_transition_interval": 1800.0,
            "recent_transition_rate": 0.0,
            "time_variability": 0.0,
            "avg_occupancy_confidence": 0.5,
            "recent_occupancy_ratio": 0.5,
            "state_stability": 0.5,
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all temporal feature names."""
        return list(self._get_default_features().keys())

    def clear_cache(self):
        """Clear the feature cache."""
        self.feature_cache.clear()

    def extract_batch_features(
        self,
        event_batches: List[Tuple[List[SensorEvent], datetime]],
        room_states_batches: Optional[List[List[RoomState]]] = None,
    ) -> List[Dict[str, float]]:
        """
        Extract temporal features for multiple event sequences efficiently.

        Args:
            event_batches: List of (events, target_time) tuples
            room_states_batches: Optional corresponding room states

        Returns:
            List of feature dictionaries
        """
        results = []

        for i, (events, target_time) in enumerate(event_batches):
            room_states = room_states_batches[i] if room_states_batches else None
            features = self.extract_features(events, target_time, room_states)
            results.append(features)

        return results
