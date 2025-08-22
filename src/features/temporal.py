"""
Temporal feature extraction for occupancy prediction.

This module extracts time-based features from sensor event sequences, including
cyclical encodings, historical patterns, and state duration metrics.
"""

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
        self.temporal_cache = {}  # Additional cache for temporal-specific features

    def extract_features(
        self,
        events: List[SensorEvent],
        target_time: datetime,
        room_states: Optional[List[RoomState]] = None,
        lookback_hours: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Extract temporal features for a specific target time.

        Args:
            events: Chronologically ordered sensor events
            target_time: Time for which to extract features
            room_states: Optional room state history
            lookback_hours: Optional hours to look back for filtering events

        Returns:
            Dictionary of temporal features

        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            if not events:
                return self._get_default_features(target_time)

            # Sort events by timestamp to ensure chronological order
            sorted_events = sorted(events, key=lambda e: e.timestamp)

            # Apply lookback filter if specified
            if lookback_hours is not None:
                cutoff_time = target_time - timedelta(hours=lookback_hours)
                sorted_events = [e for e in sorted_events if e.timestamp >= cutoff_time]

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

            # Generic sensor features using Any type for flexible value handling
            generic_features = self._extract_generic_sensor_features(sorted_events)
            features.update(generic_features)

            return features

        except Exception as e:
            logger.error(f"Failed to extract temporal features: {e}")
            # Extract room_id from events if available
            room_id = events[0].room_id if events else "unknown"
            raise FeatureExtractionError(
                feature_type="temporal", room_id=room_id, cause=e
            )

    def _extract_time_since_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract time-since-last-event features."""
        features = {}

        if not events:
            return {
                "time_since_last_event": 3600.0,  # Default 1 hour
                "time_since_last_on": 3600.0,
                "time_since_last_off": 3600.0,
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
        features["time_since_last_off"] = (
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
            "time_since_last_off",
            "time_since_last_motion",
        ]:
            features[key] = min(features[key], 86400.0)

        return features

    def _extract_duration_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract state duration features with advanced statistical analysis."""
        features = {}

        if not events:
            return {
                "current_state_duration": 0.0,
                "avg_on_duration": 1800.0,  # Default 30 minutes
                "avg_off_duration": 1800.0,
                "max_on_duration": 3600.0,
                "max_off_duration": 3600.0,
                "on_duration_std": 0.0,
                "off_duration_std": 0.0,
                "duration_ratio": 1.0,
                "median_on_duration": 1800.0,
                "median_off_duration": 1800.0,
                "duration_percentile_75": 3600.0,
                "duration_percentile_25": 900.0,
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

        # Convert to numpy arrays for efficient computation
        on_durations_array = (
            np.array(on_durations) if on_durations else np.array([1800.0])
        )
        off_durations_array = (
            np.array(off_durations) if off_durations else np.array([1800.0])
        )
        all_durations = np.concatenate([on_durations_array, off_durations_array])

        # Basic statistical features
        features["avg_on_duration"] = float(np.mean(on_durations_array))
        features["avg_off_duration"] = float(np.mean(off_durations_array))
        features["max_on_duration"] = float(np.max(on_durations_array))
        features["max_off_duration"] = float(np.max(off_durations_array))

        # Advanced statistical features using numpy
        features["on_duration_std"] = float(np.std(on_durations_array))
        features["off_duration_std"] = float(np.std(off_durations_array))

        # Duration ratio (on vs off)
        avg_on = float(np.mean(on_durations_array))
        avg_off = float(np.mean(off_durations_array))
        features["duration_ratio"] = avg_on / avg_off if avg_off > 0 else 1.0

        # Percentile features
        features["median_on_duration"] = float(np.median(on_durations_array))
        features["median_off_duration"] = float(np.median(off_durations_array))
        features["duration_percentile_75"] = float(np.percentile(all_durations, 75))
        features["duration_percentile_25"] = float(np.percentile(all_durations, 25))

        return features

    def _extract_generic_sensor_features(
        self, events: List[SensorEvent]
    ) -> Dict[str, float]:
        """
        Extract features from generic sensor values using Any type for flexibility.

        This method handles various sensor value types (numeric, boolean, string)
        and extracts meaningful features from them.
        """
        features = {}

        if not events:
            return features

        # Collect all sensor values with Any type
        sensor_values: List[Any] = []
        numeric_values: List[float] = []
        boolean_values: List[bool] = []
        string_values: List[str] = []

        for event in events:
            # Handle different attribute types
            if hasattr(event, "attributes") and event.attributes:
                try:
                    # Check if attributes is dict-like and iterable
                    if hasattr(event.attributes, "items"):
                        for key, value in event.attributes.items():
                            sensor_values.append(value)

                            # Type-specific processing
                            if isinstance(value, (int, float)):
                                numeric_values.append(float(value))
                            elif isinstance(value, bool):
                                boolean_values.append(value)
                            elif (
                                isinstance(value, str)
                                and value.replace(".", "").isdigit()
                            ):
                                try:
                                    numeric_values.append(float(value))
                                except ValueError:
                                    string_values.append(value)
                            else:
                                string_values.append(str(value))
                except (TypeError, AttributeError):
                    # Handle Mock or non-dict attributes
                    pass

            # Handle state as Any type
            state_value: Any = event.state
            sensor_values.append(state_value)

            # Try to convert state to numeric if possible
            if isinstance(state_value, str):
                try:
                    numeric_values.append(float(state_value))
                except ValueError:
                    string_values.append(state_value)
            elif isinstance(state_value, (int, float)):
                numeric_values.append(float(state_value))
            elif isinstance(state_value, bool):
                boolean_values.append(state_value)

        # Extract numeric features
        if numeric_values:
            numeric_array = np.array(numeric_values)
            features["numeric_mean"] = float(np.mean(numeric_array))
            features["numeric_std"] = float(np.std(numeric_array))
            features["numeric_min"] = float(np.min(numeric_array))
            features["numeric_max"] = float(np.max(numeric_array))
            features["numeric_range"] = (
                features["numeric_max"] - features["numeric_min"]
            )
            features["numeric_count"] = len(numeric_values)

        # Extract boolean features
        if boolean_values:
            true_count = sum(1 for v in boolean_values if v)
            features["boolean_true_ratio"] = true_count / len(boolean_values)
            features["boolean_false_ratio"] = 1.0 - features["boolean_true_ratio"]
            features["boolean_count"] = len(boolean_values)

        # Extract string features
        if string_values:
            unique_strings = set(string_values)
            features["string_unique_count"] = len(unique_strings)
            features["string_total_count"] = len(string_values)
            features["string_diversity_ratio"] = len(unique_strings) / len(
                string_values
            )

            # Most common string value
            from collections import Counter

            most_common = Counter(string_values).most_common(1)
            if most_common:
                features["most_common_string_frequency"] = most_common[0][1] / len(
                    string_values
                )

        # Overall sensor value statistics
        features["total_sensor_values"] = len(sensor_values)
        features["numeric_value_ratio"] = (
            len(numeric_values) / len(sensor_values) if sensor_values else 0.0
        )
        features["boolean_value_ratio"] = (
            len(boolean_values) / len(sensor_values) if sensor_values else 0.0
        )
        features["string_value_ratio"] = (
            len(string_values) / len(sensor_values) if sensor_values else 0.0
        )

        # Add sensor type ratios
        sensor_type_counts = {"motion": 0, "door": 0, "presence": 0}
        for event in events:
            if (
                hasattr(event, "sensor_type")
                and event.sensor_type in sensor_type_counts
            ):
                sensor_type_counts[event.sensor_type] += 1

        total_events = len(events) if events else 1
        features["motion_sensor_ratio"] = sensor_type_counts["motion"] / total_events
        features["door_sensor_ratio"] = sensor_type_counts["door"] / total_events
        features["presence_sensor_ratio"] = (
            sensor_type_counts["presence"] / total_events
        )

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
        features["day_of_week_sin"] = math.sin(2 * math.pi * day_of_week / 7)
        features["day_of_week_cos"] = math.cos(2 * math.pi * day_of_week / 7)

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
        """Extract historical occupancy patterns for similar times using advanced statistical methods."""
        features = {}

        if not events:
            return {
                "hour_activity_rate": 0.5,
                "day_activity_rate": 0.5,
                "overall_activity_rate": 0.5,
                "similar_time_activity_rate": 0.5,
                "pattern_strength": 0.0,
                "activity_variance": 0.0,
                "trend_coefficient": 0.0,
                "seasonality_score": 0.0,
            }

        # Convert events to pandas DataFrame for efficient analysis
        event_data = []
        for event in events:
            event_local = event.timestamp + timedelta(hours=self.timezone_offset)
            event_data.append(
                {
                    "timestamp": event.timestamp,
                    "hour": event_local.hour,
                    "day_of_week": event_local.weekday(),
                    "is_active": event.state == "on",
                    "day_of_year": event_local.timetuple().tm_yday,
                }
            )

        df = pd.DataFrame(event_data)

        # Adjust for timezone
        local_time = target_time + timedelta(hours=self.timezone_offset)
        current_hour = local_time.hour
        current_day_of_week = local_time.weekday()

        # Basic activity rates using pandas aggregation
        hour_activity = (
            df.groupby("hour")["is_active"].agg(["mean", "std", "count"]).fillna(0)
        )
        day_activity = (
            df.groupby("day_of_week")["is_active"]
            .agg(["mean", "std", "count"])
            .fillna(0)
        )

        features["hour_activity_rate"] = float(
            hour_activity.loc[current_hour, "mean"]
            if current_hour in hour_activity.index
            else 0.5
        )
        features["day_activity_rate"] = float(
            day_activity.loc[current_day_of_week, "mean"]
            if current_day_of_week in day_activity.index
            else 0.5
        )
        features["overall_activity_rate"] = float(df["is_active"].mean())

        # Time-of-day similarity score with weighted nearby hours
        nearby_hours = list(range(max(0, current_hour - 1), min(24, current_hour + 2)))
        similar_activities = []
        for hour in nearby_hours:
            if hour in hour_activity.index:
                weight = (
                    1.0 if hour == current_hour else 0.7
                )  # Weight current hour more
                similar_activities.extend(
                    [hour_activity.loc[hour, "mean"]] * int(weight * 10)
                )

        features["similar_time_activity_rate"] = float(
            np.mean(similar_activities) if similar_activities else 0.5
        )

        # Advanced statistical features using numpy
        activity_values = df["is_active"].values.astype(float)

        # Pattern strength (consistency of activity at similar times)
        if len(hour_activity) > 1:
            hour_stds = hour_activity["std"].fillna(0).values
            features["pattern_strength"] = float(
                1.0 - np.mean(hour_stds)
            )  # Lower std = stronger pattern
        else:
            features["pattern_strength"] = 0.0

        # Activity variance
        features["activity_variance"] = float(np.var(activity_values))

        # Trend coefficient using linear regression
        if len(df) > 2:
            time_indices = np.arange(len(df))
            trend_coeff = np.corrcoef(time_indices, activity_values)[0, 1]
            features["trend_coefficient"] = float(
                trend_coeff if not np.isnan(trend_coeff) else 0.0
            )
        else:
            features["trend_coefficient"] = 0.0

        # Seasonality score (day-of-year pattern strength)
        if len(df) > 7:
            day_of_year_activity = df.groupby("day_of_year")["is_active"].mean()
            if len(day_of_year_activity) > 1:
                features["seasonality_score"] = float(
                    np.std(day_of_year_activity.values)
                )
            else:
                features["seasonality_score"] = 0.0
        else:
            features["seasonality_score"] = 0.0

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

        # Transition regularity - inverse of variability (more regular = higher score)
        features["transition_regularity"] = max(0.0, 1.0 - features["time_variability"])

        # Recent transition trend - slope of recent intervals
        if len(intervals) >= 3:
            # Use last half of intervals to calculate trend
            recent_intervals = intervals[-len(intervals) // 2 :]
            x_values = list(range(len(recent_intervals)))
            # Simple linear regression slope
            n = len(recent_intervals)
            x_mean = sum(x_values) / n
            y_mean = sum(recent_intervals) / n
            numerator = sum(
                (x_values[i] - x_mean) * (recent_intervals[i] - y_mean)
                for i in range(n)
            )
            denominator = sum((x_values[i] - x_mean) ** 2 for i in range(n))
            features["recent_transition_trend"] = (
                numerator / denominator if denominator != 0 else 0.0
            )
        else:
            features["recent_transition_trend"] = 0.0

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

    def _get_default_features(
        self, target_time: Optional[datetime] = None
    ) -> Dict[str, float]:
        """Return default feature values when no events are available."""
        defaults = {
            "time_since_last_event": 3600.0,
            "time_since_last_on": 3600.0,
            "time_since_last_off": 3600.0,
            "time_since_last_motion": 3600.0,
            "current_state_duration": 0.0,
            "avg_on_duration": 1800.0,
            "avg_off_duration": 1800.0,
            "max_on_duration": 3600.0,
            "max_off_duration": 3600.0,
            # Added advanced duration features
            "on_duration_std": 0.0,
            "off_duration_std": 0.0,
            "duration_ratio": 1.0,
            "median_on_duration": 1800.0,
            "median_off_duration": 1800.0,
            "duration_percentile_75": 3600.0,
            "duration_percentile_25": 900.0,
        }

        # Add actual cyclical features if target_time provided
        if target_time:
            cyclical_features = self._extract_cyclical_features(target_time)
            defaults.update(cyclical_features)
        else:
            # Fallback cyclical features
            defaults.update(
                {
                    "hour_sin": 0.0,
                    "hour_cos": 1.0,
                    "day_of_week_sin": 0.0,
                    "day_of_week_cos": 1.0,
                    "month_sin": 0.0,
                    "month_cos": 1.0,
                    "day_of_month_sin": 0.0,
                    "day_of_month_cos": 1.0,
                    "is_weekend": 0.0,
                    "is_work_hours": 0.0,
                    "is_sleep_hours": 0.0,
                }
            )

        # Add remaining default features
        defaults.update(
            {
                # Basic historical patterns
                "hour_activity_rate": 0.5,
                "day_activity_rate": 0.5,
                "overall_activity_rate": 0.5,
                "similar_time_activity_rate": 0.5,
                # Pattern features
                "activity_variance": 0.0,
                "trend_coefficient": 0.0,
                "seasonality_score": 0.0,
                # Transition features
                "avg_transition_interval": 1800.0,
                "recent_transition_rate": 0.0,
                "time_variability": 0.0,
                # Room state features
                "avg_occupancy_confidence": 0.5,
                "recent_occupancy_ratio": 0.5,
                "state_stability": 0.5,
                # Sensor type ratios
                "motion_sensor_ratio": 0.0,
                "door_sensor_ratio": 0.0,
                "presence_sensor_ratio": 0.0,
            }
        )

        return defaults

    def get_feature_names(self) -> List[str]:
        """Get list of all temporal feature names using standardized names."""
        return list(self._get_default_features().keys())

    def validate_feature_names(
        self, extracted_features: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate and standardize feature names using TEMPORAL_FEATURE_NAMES constant."""
        validated_features = {}

        # Map extracted features to standardized names where possible
        name_mappings = {
            "time_since_last_off": "time_since_last_change",
            "current_state_duration": "current_state_duration",
            "hour_sin": "hour_sin",
            "hour_cos": "hour_cos",
            "day_of_week_sin": "day_of_week_sin",
            "day_of_week_cos": "day_of_week_cos",
            "is_weekend": "is_weekend",
        }

        # Use TEMPORAL_FEATURE_NAMES for standardization
        for standard_name in TEMPORAL_FEATURE_NAMES:
            if standard_name in extracted_features:
                validated_features[standard_name] = extracted_features[standard_name]
            else:
                # Try to find mapped equivalent
                for extracted_name, mapped_name in name_mappings.items():
                    if (
                        mapped_name == standard_name
                        and extracted_name in extracted_features
                    ):
                        validated_features[standard_name] = extracted_features[
                            extracted_name
                        ]
                        break

        # Add any additional features that don't have standard mappings
        for feature_name, value in extracted_features.items():
            if (
                feature_name not in validated_features
                and feature_name not in name_mappings
            ):
                validated_features[feature_name] = value

        return validated_features

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
