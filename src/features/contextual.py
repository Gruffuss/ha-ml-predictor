"""
Contextual feature extraction for occupancy prediction.

This module extracts environmental and contextual features from sensor data,
including temperature, humidity, lighting conditions, door states, and
multi-room occupancy correlations.
"""

from collections import defaultdict, deque
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional, Set

import numpy as np
import statistics

from ..core.config import RoomConfig, SystemConfig
from ..core.constants import SensorType
from ..core.exceptions import FeatureExtractionError
from ..data.storage.models import RoomState, SensorEvent

logger = logging.getLogger(__name__)


class ContextualFeatureExtractor:
    """
    Extract environmental and contextual features from sensor data.

    This extractor focuses on contextual patterns including:
    - Environmental conditions (temperature, humidity, light)
    - Door state patterns and sequences
    - Multi-room occupancy correlations
    - External factors and seasonal patterns
    - Cross-sensor contextual relationships
    """

    def __init__(self, config: Optional[SystemConfig] = None):
        """
        Initialize the contextual feature extractor.

        Args:
            config: System configuration with room definitions
        """
        self.config = config
        self.context_cache = {}

        # Environmental thresholds for feature extraction
        self.temp_thresholds = {
            "cold": 18.0,  # Celsius
            "comfortable": 22.0,
            "warm": 26.0,
        }

        self.humidity_thresholds = {
            "dry": 40.0,  # Percentage
            "comfortable": 60.0,
            "humid": 70.0,
        }

        self.light_thresholds = {
            "dark": 100.0,
            "dim": 300.0,
            "bright": 1000.0,
        }  # Lux

    def extract_features(
        self,
        events: List[SensorEvent],
        room_states: Optional[List[RoomState]],
        target_time: datetime,
        room_configs: Optional[Dict[str, RoomConfig]] = None,
        lookback_hours: int = 24,
    ) -> Dict[str, float]:
        """
        Extract contextual features from sensor events and room states.

        Args:
            events: Chronologically ordered sensor events
            room_states: Room state history
            target_time: Time for which to extract features
            room_configs: Room configuration mapping
            lookback_hours: How far back to look for patterns

        Returns:
            Dictionary of contextual features

        Raises:
            FeatureExtractionError: If feature extraction fails
        """
        try:
            if not events:
                return self._get_default_features()

            # Filter events within lookback window
            cutoff_time = target_time - timedelta(hours=lookback_hours)
            recent_events = [e for e in events if e.timestamp >= cutoff_time]
            recent_room_states = [
                r for r in (room_states or []) if r.timestamp >= cutoff_time
            ]

            features = {}

            # Environmental condition features
            features.update(
                self._extract_environmental_features(recent_events, target_time)
            )

            # Door state and transition features
            features.update(
                self._extract_door_state_features(recent_events, target_time)
            )

            # Multi-room occupancy correlation features
            features.update(
                self._extract_multi_room_features(
                    recent_events, recent_room_states, target_time
                )
            )

            # Contextual timing and seasonal features
            features.update(self._extract_seasonal_features(target_time))

            # Cross-sensor correlation features
            features.update(self._extract_sensor_correlation_features(recent_events))

            # External context features (if available)
            if room_configs:
                features.update(
                    self._extract_room_context_features(
                        recent_events, room_configs, target_time
                    )
                )

            return features

        except Exception as e:
            logger.error(f"Failed to extract contextual features: {e}")
            # Extract room_id from events if available
            room_id = events[0].room_id if events else "unknown"
            raise FeatureExtractionError(
                feature_type="contextual", room_id=room_id, cause=e
            )

    def _extract_environmental_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract environmental sensor features using SensorType filtering."""
        features = {}

        # Group events by sensor type using SensorType enum for validation
        env_events = {
            "temperature": [],
            "humidity": [],
            "light": [],
            "climate": [],
        }

        # Track unique sensor types using Set
        detected_sensor_types: Set[str] = set()

        for event in events:
            detected_sensor_types.add(event.sensor_type)

            # Use SensorType enum for proper filtering
            if event.sensor_type == SensorType.CLIMATE.value:
                env_events["climate"].append(event)
            elif event.sensor_type == SensorType.LIGHT.value:
                env_events["light"].append(event)
            elif event.sensor_type.lower() in ["temperature", "temp"]:
                env_events["temperature"].append(event)
            elif event.sensor_type.lower() in ["humidity", "humid"]:
                env_events["humidity"].append(event)
            elif event.sensor_type.lower() in ["illuminance", "light", "lux"]:
                env_events["light"].append(event)
            # Fallback to sensor_id analysis if sensor_type doesn't match
            elif "temperature" in event.sensor_id.lower():
                env_events["temperature"].append(event)
            elif "humidity" in event.sensor_id.lower():
                env_events["humidity"].append(event)
            elif "light" in event.sensor_id.lower() or "lux" in event.sensor_id.lower():
                env_events["light"].append(event)

        # Temperature features
        temp_values = self._extract_numeric_values(
            env_events["temperature"], "temperature"
        )
        if temp_values:
            features["current_temperature"] = temp_values[-1]
            features["avg_temperature"] = statistics.mean(temp_values)
            features["temperature_trend"] = self._calculate_trend(temp_values)
            features["temperature_variance"] = (
                statistics.variance(temp_values) if len(temp_values) > 1 else 0.0
            )

            # Temperature comfort zones
            current_temp = temp_values[-1]
            features["is_cold"] = (
                1.0 if current_temp < self.temp_thresholds["cold"] else 0.0
            )
            features["is_comfortable_temp"] = (
                1.0
                if (
                    self.temp_thresholds["cold"]
                    <= current_temp
                    <= self.temp_thresholds["warm"]
                )
                else 0.0
            )
            features["is_warm"] = (
                1.0 if current_temp > self.temp_thresholds["warm"] else 0.0
            )
        else:
            features.update(
                {
                    "current_temperature": 22.0,
                    "avg_temperature": 22.0,
                    "temperature_trend": 0.0,
                    "temperature_variance": 0.0,
                    "is_cold": 0.0,
                    "is_comfortable_temp": 1.0,
                    "is_warm": 0.0,
                }
            )

        # Humidity features
        humidity_values = self._extract_numeric_values(
            env_events["humidity"], "humidity"
        )
        if humidity_values:
            features["current_humidity"] = humidity_values[-1]
            features["avg_humidity"] = statistics.mean(humidity_values)
            features["humidity_trend"] = self._calculate_trend(humidity_values)
        else:
            features.update(
                {
                    "current_humidity": 50.0,
                    "avg_humidity": 50.0,
                    "humidity_trend": 0.0,
                }
            )

        # Light/illuminance features
        light_values = self._extract_numeric_values(env_events["light"], "illuminance")
        if light_values:
            features["current_light"] = light_values[-1]
            features["avg_light"] = statistics.mean(light_values)
            features["light_trend"] = self._calculate_trend(light_values)

            # Light level categories
            current_light = light_values[-1]
            features["is_dark"] = (
                1.0 if current_light < self.light_thresholds["dark"] else 0.0
            )
            features["is_dim"] = (
                1.0
                if (
                    self.light_thresholds["dark"]
                    <= current_light
                    < self.light_thresholds["bright"]
                )
                else 0.0
            )
            features["is_bright"] = (
                1.0 if current_light >= self.light_thresholds["bright"] else 0.0
            )

            # Natural light pattern detection
            features["natural_light_score"] = self._calculate_natural_light_score(
                light_values, target_time
            )
            features["light_change_rate"] = self._calculate_light_change_rate(
                light_values
            )
        else:
            features.update(
                {
                    "current_light": 500.0,
                    "avg_light": 500.0,
                    "light_trend": 0.0,
                    "is_dark": 0.0,
                    "is_dim": 1.0,
                    "is_bright": 0.0,
                    "natural_light_score": 0.0,
                    "light_change_rate": 0.0,
                }
            )

        return features

    def _extract_door_state_features(
        self, events: List[SensorEvent], target_time: datetime
    ) -> Dict[str, float]:
        """Extract door state and transition features."""
        features = {}

        # Filter door events using SensorType enum
        door_events = []
        door_sensor_ids: Set[str] = set()

        for event in events:
            # Use SensorType enum for proper door sensor filtering
            if (
                event.sensor_type == SensorType.DOOR.value
                or "door" in event.sensor_id.lower()
            ):
                door_events.append(event)
                door_sensor_ids.add(event.sensor_id)

        if not door_events:
            return {
                "doors_currently_open": 0.0,
                "door_open_ratio": 0.0,
                "door_transition_count": 0.0,
                "avg_door_open_duration": 0.0,
                "recent_door_activity": 0.0,
            }

        # Current door states
        current_door_states = {}
        door_open_durations = []
        door_transitions = 0

        for door_id in set(e.sensor_id for e in door_events):
            door_specific_events = [e for e in door_events if e.sensor_id == door_id]
            door_specific_events.sort(key=lambda x: x.timestamp)

            if door_specific_events:
                # Current state
                current_state = door_specific_events[-1].state
                current_door_states[door_id] = (
                    current_state == "on" or current_state == "open"
                )

                # Calculate transitions and durations
                previous_state = None
                state_start_time = None

                for event in door_specific_events:
                    if event.state != previous_state:
                        if previous_state is not None and state_start_time is not None:
                            if previous_state == "on" or previous_state == "open":
                                duration = (
                                    event.timestamp - state_start_time
                                ).total_seconds()
                                door_open_durations.append(duration)

                        door_transitions += 1
                        previous_state = event.state
                        state_start_time = event.timestamp

        # Calculate features
        features["doors_currently_open"] = sum(current_door_states.values())
        features["door_open_ratio"] = (
            sum(current_door_states.values()) / len(current_door_states)
            if current_door_states
            else 0.0
        )
        features["door_transition_count"] = door_transitions
        features["avg_door_open_duration"] = (
            statistics.mean(door_open_durations) if door_open_durations else 0.0
        )

        # Recent door activity (last 1 hour)
        recent_cutoff = target_time - timedelta(hours=1)
        recent_door_events = [e for e in door_events if e.timestamp >= recent_cutoff]
        features["recent_door_activity"] = len(recent_door_events)

        return features

    def _extract_multi_room_features(
        self,
        events: List[SensorEvent],
        room_states: List[RoomState],
        target_time: datetime,
    ) -> Dict[str, float]:
        """Extract multi-room occupancy correlation features."""
        features = {}

        # Group events and states by room
        room_events = defaultdict(list)
        room_state_history = defaultdict(list)

        for event in events:
            room_events[event.room_id].append(event)

        for state in room_states:
            room_state_history[state.room_id].append(state)

        room_count = len(room_events)
        features["total_active_rooms"] = room_count

        if room_count < 2:
            return {
                "total_active_rooms": room_count,
                "simultaneous_occupancy_ratio": 0.0,
                "room_activity_correlation": 0.0,
                "dominant_room_activity_ratio": 1.0,
                "room_activity_balance": 0.0,
            }

        # Current occupancy states
        current_occupancy = {}
        for room_id, states in room_state_history.items():
            if states:
                latest_state = max(states, key=lambda s: s.timestamp)
                current_occupancy[room_id] = latest_state.is_occupied

        # Simultaneous occupancy ratio
        occupied_rooms = sum(current_occupancy.values())
        features["simultaneous_occupancy_ratio"] = (
            occupied_rooms / len(current_occupancy) if current_occupancy else 0.0
        )

        # Room activity correlation (events happening in multiple rooms within time windows)
        correlation_score = self._calculate_room_activity_correlation(
            room_events, target_time
        )
        features["room_activity_correlation"] = correlation_score

        # Dominant room activity ratio
        room_activity_counts = {
            room_id: len(events) for room_id, events in room_events.items()
        }
        if room_activity_counts:
            max_activity = max(room_activity_counts.values())
            total_activity = sum(room_activity_counts.values())
            features["dominant_room_activity_ratio"] = (
                max_activity / total_activity if total_activity > 0 else 0.0
            )
        else:
            features["dominant_room_activity_ratio"] = 0.0

        # Room activity balance (entropy-based measure)
        if room_activity_counts:
            total_events = sum(room_activity_counts.values())
            entropy = 0.0
            for count in room_activity_counts.values():
                if count > 0:
                    p = count / total_events
                    entropy -= p * np.log2(p)

            max_entropy = np.log2(len(room_activity_counts))
            features["room_activity_balance"] = (
                entropy / max_entropy if max_entropy > 0 else 0.0
            )
        else:
            features["room_activity_balance"] = 0.0

        return features

    def _extract_seasonal_features(self, target_time: datetime) -> Dict[str, float]:
        """Extract seasonal and external context features."""
        features = {}

        # Basic seasonal indicators
        month = target_time.month

        # Season indicators
        features["is_winter"] = 1.0 if month in [12, 1, 2] else 0.0
        features["is_spring"] = 1.0 if month in [3, 4, 5] else 0.0
        features["is_summer"] = 1.0 if month in [6, 7, 8] else 0.0
        features["is_autumn"] = 1.0 if month in [9, 10, 11] else 0.0

        # Holiday indicators (simplified)
        day = target_time.day
        features["is_holiday_season"] = (
            1.0 if (month == 12 and day >= 20) or (month == 1 and day <= 7) else 0.0
        )

        # Natural light patterns (approximate)
        hour = target_time.hour
        if month in [6, 7, 8]:  # Summer
            features["natural_light_available"] = 1.0 if 5 <= hour <= 20 else 0.0
        elif month in [12, 1, 2]:  # Winter
            features["natural_light_available"] = 1.0 if 7 <= hour <= 17 else 0.0
        else:  # Spring/Autumn
            features["natural_light_available"] = 1.0 if 6 <= hour <= 19 else 0.0

        return features

    def _extract_sensor_correlation_features(
        self, events: List[SensorEvent]
    ) -> Dict[str, float]:
        """Extract cross-sensor correlation features."""
        features = {}

        if len(events) < 2:
            return {
                "sensor_activation_correlation": 0.0,
                "multi_sensor_event_ratio": 0.0,
                "sensor_type_diversity": 0.0,
            }

        # Time-based sensor correlation using efficient sliding window
        correlation_windows = []
        window_size = 300  # 5 minutes

        # Use deque for efficient sliding window operations
        window_events = deque()

        for event in events:
            window_start = event.timestamp - timedelta(seconds=window_size)

            # Remove events outside the window from the left
            while window_events and window_events[0].timestamp < window_start:
                window_events.popleft()

            # Add current event to window
            window_events.append(event)

            # Calculate unique sensors in current window
            unique_sensors = len(set(e.sensor_id for e in window_events))
            correlation_windows.append(unique_sensors)

        features["sensor_activation_correlation"] = (
            statistics.mean(correlation_windows) if correlation_windows else 0.0
        )

        # Multi-sensor event ratio (events with multiple sensors active simultaneously)
        multi_sensor_events = sum(1 for count in correlation_windows if count > 1)
        features["multi_sensor_event_ratio"] = (
            multi_sensor_events / len(correlation_windows)
            if correlation_windows
            else 0.0
        )

        # Sensor type diversity
        sensor_types = [event.sensor_type for event in events]
        unique_types = len(set(sensor_types))
        features["sensor_type_diversity"] = unique_types

        return features

    def _extract_room_context_features(
        self,
        events: List[SensorEvent],
        room_configs: Dict[str, RoomConfig],
        target_time: datetime,
    ) -> Dict[str, float]:
        """Extract room-specific context features using SensorType filtering."""
        features = {}

        # Room type and size indicators (based on sensor configuration)
        room_sensor_counts = defaultdict(int)
        room_sensor_types: Dict[str, Set[str]] = defaultdict(set)  # Proper Set typing
        sensor_type_distribution: Dict[str, int] = defaultdict(int)

        for event in events:
            room_id = event.room_id
            room_sensor_counts[room_id] += 1
            room_sensor_types[room_id].add(event.sensor_type)

            # Count distribution of different sensor types using SensorType validation
            if event.sensor_type in [t.value for t in SensorType]:
                sensor_type_distribution[event.sensor_type] += 1

        if room_sensor_counts:
            # Estimate room complexity from sensor diversity
            max_sensor_types = max(len(types) for types in room_sensor_types.values())
            avg_sensor_types = statistics.mean(
                [len(types) for types in room_sensor_types.values()]
            )

            features["max_room_complexity"] = max_sensor_types
            features["avg_room_complexity"] = avg_sensor_types

            # Room activity intensity
            max_activity = max(room_sensor_counts.values())
            avg_activity = statistics.mean(room_sensor_counts.values())

            features["max_room_activity"] = max_activity
            features["avg_room_activity"] = avg_activity
            features["room_activity_variance"] = (
                statistics.variance(room_sensor_counts.values())
                if len(room_sensor_counts) > 1
                else 0.0
            )

            # Sensor type diversity features using SensorType enum
            features["sensor_type_diversity"] = len(sensor_type_distribution)

            # Calculate sensor type ratios
            total_sensor_events = sum(sensor_type_distribution.values())
            if total_sensor_events > 0:
                features["presence_sensor_ratio"] = (
                    sensor_type_distribution.get(SensorType.PRESENCE.value, 0)
                    + sensor_type_distribution.get(SensorType.MOTION.value, 0)
                ) / total_sensor_events
                features["door_sensor_ratio"] = (
                    sensor_type_distribution.get(SensorType.DOOR.value, 0)
                    / total_sensor_events
                )
                features["climate_sensor_ratio"] = (
                    sensor_type_distribution.get(SensorType.CLIMATE.value, 0)
                    / total_sensor_events
                )
            else:
                features["presence_sensor_ratio"] = 0.0
                features["door_sensor_ratio"] = 0.0
                features["climate_sensor_ratio"] = 0.0
        else:
            features.update(
                {
                    "max_room_complexity": 0.0,
                    "avg_room_complexity": 0.0,
                    "max_room_activity": 0.0,
                    "avg_room_activity": 0.0,
                    "room_activity_variance": 0.0,
                }
            )

        return features

    def _extract_numeric_values(
        self, events: List[SensorEvent], value_key: str
    ) -> List[float]:
        """Extract numeric values from sensor events."""
        values = []

        for event in events:
            # Try to extract numeric value from state
            try:
                if event.state and event.state != "unknown":
                    value = float(event.state)
                    values.append(value)
            except (ValueError, TypeError):
                pass

            # Try to extract from attributes
            if event.attributes and isinstance(event.attributes, dict):
                for key in [value_key, "value", "state"]:
                    if key in event.attributes:
                        try:
                            value = float(event.attributes[key])
                            values.append(value)
                            break
                        except (ValueError, TypeError):
                            pass

        return values

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend (slope) from a series of values."""
        if len(values) < 2:
            return 0.0

        # Simple linear trend calculation
        n = len(values)
        x = list(range(n))

        x_mean = statistics.mean(x)
        y_mean = statistics.mean(values)

        numerator = sum((x[i] - x_mean) * (values[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))

        return numerator / denominator if denominator != 0 else 0.0

    def _calculate_room_activity_correlation(
        self, room_events: Dict[str, List[SensorEvent]], target_time: datetime
    ) -> float:
        """Calculate correlation between room activities."""
        if len(room_events) < 2:
            return 0.0

        # Create time windows and count events per room per window
        window_size = 600  # 10 minutes
        lookback_hours = 6
        start_time = target_time - timedelta(hours=lookback_hours)

        windows = []
        current_time = start_time
        while current_time < target_time:
            window_end = current_time + timedelta(seconds=window_size)
            windows.append((current_time, window_end))
            current_time = window_end

        # Count events per room per window
        room_activity_vectors = defaultdict(list)

        for room_id, events in room_events.items():
            for window_start, window_end in windows:
                count = sum(
                    1
                    for event in events
                    if window_start <= event.timestamp < window_end
                )
                room_activity_vectors[room_id].append(count)

        # Calculate correlation between room activity vectors
        room_ids = list(room_activity_vectors.keys())
        correlations = []

        for i in range(len(room_ids)):
            for j in range(i + 1, len(room_ids)):
                vector_i = room_activity_vectors[room_ids[i]]
                vector_j = room_activity_vectors[room_ids[j]]

                if len(vector_i) == len(vector_j) and len(vector_i) > 1:
                    correlation = np.corrcoef(vector_i, vector_j)[0, 1]
                    if not np.isnan(correlation):
                        correlations.append(abs(correlation))

        return statistics.mean(correlations) if correlations else 0.0

    def _calculate_natural_light_score(
        self, light_values: List[float], target_time: datetime
    ) -> float:
        """Calculate natural light pattern score based on time of day and light levels."""
        if not light_values or len(light_values) < 2:
            return 0.0

        # Expected natural light pattern based on time
        hour = target_time.hour

        # Define expected light levels for natural patterns
        if 5 <= hour <= 8:  # Dawn
            expected_range = (50, 300)
        elif 9 <= hour <= 11:  # Morning
            expected_range = (200, 500)
        elif 12 <= hour <= 15:  # Midday
            expected_range = (400, 800)
        elif 16 <= hour <= 18:  # Afternoon
            expected_range = (200, 600)
        elif 19 <= hour <= 21:  # Evening
            expected_range = (50, 200)
        else:  # Night
            expected_range = (0, 100)

        current_light = light_values[-1]

        # Score based on how well current light matches expected natural pattern
        if expected_range[0] <= current_light <= expected_range[1]:
            # Perfect match gets score of 1.0
            mid_range = (expected_range[0] + expected_range[1]) / 2
            deviation = abs(current_light - mid_range) / (
                expected_range[1] - expected_range[0]
            )
            return max(0.0, 1.0 - deviation)
        else:
            # Outside expected range
            return 0.0

    def _calculate_light_change_rate(self, light_values: List[float]) -> float:
        """Calculate rate of light change over time."""
        if len(light_values) < 2:
            return 0.0

        # Calculate absolute change rate
        changes = [
            abs(light_values[i] - light_values[i - 1])
            for i in range(1, len(light_values))
        ]
        return statistics.mean(changes) if changes else 0.0

    def _get_default_features(self) -> Dict[str, float]:
        """Return default feature values when no events are available."""
        return {
            "current_temperature": 22.0,
            "avg_temperature": 22.0,
            "temperature_trend": 0.0,
            "temperature_variance": 0.0,
            "is_cold": 0.0,
            "is_comfortable_temp": 1.0,
            "is_warm": 0.0,
            "current_humidity": 50.0,
            "avg_humidity": 50.0,
            "humidity_trend": 0.0,
            "current_light": 500.0,
            "avg_light": 500.0,
            "light_trend": 0.0,
            "is_dark": 0.0,
            "is_dim": 1.0,
            "is_bright": 0.0,
            "natural_light_score": 0.0,
            "light_change_rate": 0.0,
            "doors_currently_open": 0.0,
            "door_open_ratio": 0.0,
            "door_transition_count": 0.0,
            "avg_door_open_duration": 0.0,
            "recent_door_activity": 0.0,
            "total_active_rooms": 1.0,
            "simultaneous_occupancy_ratio": 0.0,
            "room_activity_correlation": 0.0,
            "dominant_room_activity_ratio": 1.0,
            "room_activity_balance": 0.0,
            "is_winter": 0.0,
            "is_spring": 0.0,
            "is_summer": 0.0,
            "is_autumn": 0.0,
            "is_holiday_season": 0.0,
            "natural_light_available": 0.0,
            "sensor_activation_correlation": 0.0,
            "multi_sensor_event_ratio": 0.0,
            "sensor_type_diversity": 0.0,
            "max_room_complexity": 0.0,
            "avg_room_complexity": 0.0,
            "max_room_activity": 0.0,
            "room_activity_variance": 0.0,
            # Added sensor type diversity features
            "presence_sensor_ratio": 0.0,
            "door_sensor_ratio": 0.0,
            "climate_sensor_ratio": 0.0,
        }

    def get_feature_names(self) -> List[str]:
        """Get list of all contextual feature names."""
        return list(self._get_default_features().keys())

    def clear_cache(self):
        """Clear the context cache."""
        self.context_cache.clear()

    def _filter_environmental_events(
        self, events: List[SensorEvent]
    ) -> List[SensorEvent]:
        """
        Filter events to include only environmental sensor types.

        Args:
            events: List of sensor events to filter

        Returns:
            List of environmental sensor events (temperature, humidity, light, etc.)
        """
        environmental_types = {
            SensorType.CLIMATE,
            SensorType.LIGHT,
            # Note: Using available enum values from constants.py
        }

        environmental_keywords = {
            "temperature",
            "temp",
            "humidity",
            "light",
            "illuminance",
            "lux",
            "climate",
            "weather",
            "pressure",
            "air",
        }

        filtered_events = []

        for event in events:
            # Check sensor type if available
            if (
                hasattr(event, "sensor_type")
                and event.sensor_type in environmental_types
            ):
                filtered_events.append(event)
                continue

            # Check sensor ID for environmental keywords
            sensor_id_lower = event.sensor_id.lower()
            if any(keyword in sensor_id_lower for keyword in environmental_keywords):
                filtered_events.append(event)
                continue

            # Check if event has numeric state (typical for environmental sensors)
            try:
                float(event.state)
                # If sensor ID suggests environmental data, include it
                if any(keyword in sensor_id_lower for keyword in ["sensor", "climate"]):
                    filtered_events.append(event)
            except (ValueError, TypeError):
                # Not a numeric value, skip
                pass

        return filtered_events

    def _filter_door_events(self, events: List[SensorEvent]) -> List[SensorEvent]:
        """
        Filter events to include only door/binary sensor events.

        Args:
            events: List of sensor events to filter

        Returns:
            List of door/binary sensor events
        """
        door_types = {
            SensorType.DOOR,
        }

        door_keywords = {
            "door",
            "binary_sensor",
            "switch",
            "contact",
            "open",
            "closed",
            "lock",
            "sensor",
        }

        filtered_events = []

        for event in events:
            # Check sensor type if available
            if hasattr(event, "sensor_type") and event.sensor_type in door_types:
                filtered_events.append(event)
                continue

            # Check sensor ID for door keywords
            sensor_id_lower = event.sensor_id.lower()
            if any(keyword in sensor_id_lower for keyword in door_keywords):
                # Include if sensor ID suggests door behavior, regardless of state format
                # (some tests may use non-binary states for door sensors)
                filtered_events.append(event)
                continue

        return filtered_events
