"""
Advanced Pattern Detection and Anomaly Validation for Sensor Data.

This module provides sophisticated pattern detection capabilities for:
- Anomaly detection in sensor data streams
- Behavioral pattern validation
- Data corruption detection
- Real-time data quality monitoring
- Statistical validation of sensor patterns
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import statistics

try:
    import numpy as np
    from scipy import stats
    from sklearn.cluster import DBSCAN
    from sklearn.preprocessing import StandardScaler

    SCIPY_AVAILABLE = True
    SKLEARN_AVAILABLE = True
except ImportError:
    # Fallback for when scipy/sklearn are not available
    SCIPY_AVAILABLE = False
    SKLEARN_AVAILABLE = False

    # Mock numpy for basic operations
    class MockNumpy:
        @staticmethod
        def array(data):
            return data

    np = MockNumpy()

    # Mock scipy stats
    class MockStats:
        @staticmethod
        def shapiro(data):
            return None, 0.5

    stats = MockStats()

from ...core.config import RoomConfig, get_config
from ...core.constants import (
    CAT_MOVEMENT_PATTERNS,
    HUMAN_MOVEMENT_PATTERNS,
    MAX_SEQUENCE_GAP,
    MIN_EVENT_SEPARATION,
    SensorState,
    SensorType,
)
from ...core.exceptions import (
    DataValidationError,
    ErrorSeverity,
    FeatureExtractionError,
)
from ..storage.models import SensorEvent
from .event_validator import ValidationError, ValidationResult

logger = logging.getLogger(__name__)


@dataclass
class PatternAnomaly:
    """Represents a detected pattern anomaly."""

    anomaly_id: str
    anomaly_type: str
    description: str
    severity: ErrorSeverity
    confidence: float
    detected_at: datetime
    affected_sensors: List[str]
    statistical_measures: Dict[str, float] = field(default_factory=dict)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DataQualityMetrics:
    """Comprehensive data quality metrics."""

    completeness_score: float  # 0-1, percentage of expected data present
    consistency_score: float  # 0-1, consistency across sensors
    accuracy_score: float  # 0-1, accuracy based on expected patterns
    timeliness_score: float  # 0-1, data freshness and ordering
    anomaly_count: int
    corruption_indicators: List[str] = field(default_factory=list)
    quality_trends: Dict[str, List[float]] = field(default_factory=dict)


class StatisticalPatternAnalyzer:
    """Advanced statistical analysis for sensor patterns."""

    def __init__(self, window_size: int = 100, confidence_level: float = 0.95):
        """Initialize statistical pattern analyzer."""
        self.window_size = window_size
        self.confidence_level = confidence_level
        self.sensor_baselines = defaultdict(dict)
        self.pattern_cache = {}

    def analyze_sensor_behavior(
        self, sensor_id: str, events: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze statistical patterns for a specific sensor."""
        if not events:
            return {"error": "No events provided for analysis"}

        # Extract timestamps and states
        timestamps = []
        states = []
        intervals = []

        for event in events:
            if "timestamp" in event and "state" in event:
                try:
                    if isinstance(event["timestamp"], str):
                        ts = datetime.fromisoformat(
                            event["timestamp"].replace("Z", "+00:00")
                        )
                    else:
                        ts = event["timestamp"]
                    timestamps.append(ts)
                    states.append(event["state"])
                except Exception:
                    continue

        if len(timestamps) < 2:
            return {"error": "Insufficient data for pattern analysis"}

        # Calculate inter-event intervals
        for i in range(1, len(timestamps)):
            interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
            intervals.append(interval)

        # Statistical measures
        analysis = {
            "event_count": len(events),
            "time_span_hours": (timestamps[-1] - timestamps[0]).total_seconds() / 3600,
            "mean_interval": statistics.mean(intervals) if intervals else 0,
            "median_interval": statistics.median(intervals) if intervals else 0,
            "std_interval": statistics.stdev(intervals) if len(intervals) > 1 else 0,
            "state_distribution": self._calculate_state_distribution(states),
            "trigger_frequency": len(events)
            / max(1, (timestamps[-1] - timestamps[0]).total_seconds() / 3600),
        }

        # Detect statistical anomalies
        if len(intervals) > 10:
            analysis.update(self._detect_statistical_anomalies(intervals))

        # Update baseline for this sensor
        self.sensor_baselines[sensor_id] = analysis

        return analysis

    def _calculate_state_distribution(self, states: List[str]) -> Dict[str, float]:
        """Calculate the distribution of states."""
        if not states:
            return {}

        state_counts = defaultdict(int)
        for state in states:
            state_counts[state] += 1

        total = len(states)
        return {state: count / total for state, count in state_counts.items()}

    def _detect_statistical_anomalies(self, intervals: List[float]) -> Dict[str, Any]:
        """Detect statistical anomalies in event intervals."""
        anomalies = {
            "outliers": [],
            "z_scores": [],
            "anomaly_count": 0,
            "distribution_type": "unknown",
        }

        if len(intervals) < 3:
            return anomalies

        # Calculate z-scores
        mean_interval = statistics.mean(intervals)
        std_interval = statistics.stdev(intervals)

        z_threshold = 2.5  # Values beyond 2.5 std deviations are anomalous

        for i, interval in enumerate(intervals):
            if std_interval > 0:
                z_score = abs(interval - mean_interval) / std_interval
                anomalies["z_scores"].append(z_score)

                if z_score > z_threshold:
                    anomalies["outliers"].append(
                        {"index": i, "value": interval, "z_score": z_score}
                    )
                    anomalies["anomaly_count"] += 1

        # Test for normality
        if len(intervals) > 8:
            try:
                _, p_value = stats.shapiro(intervals)
                if p_value > 0.05:
                    anomalies["distribution_type"] = "normal"
                else:
                    anomalies["distribution_type"] = "non_normal"
            except Exception:
                anomalies["distribution_type"] = "unknown"

        return anomalies

    def detect_sensor_malfunction(
        self, sensor_id: str, recent_events: List[Dict[str, Any]]
    ) -> List[PatternAnomaly]:
        """Detect potential sensor malfunctions based on patterns."""
        anomalies = []

        if not recent_events:
            return anomalies

        # Analyze current behavior
        current_analysis = self.analyze_sensor_behavior(sensor_id, recent_events)
        baseline = self.sensor_baselines.get(sensor_id, {})

        if not baseline:
            return anomalies  # No baseline to compare against

        # Check for frequency anomalies
        current_freq = current_analysis.get("trigger_frequency", 0)
        baseline_freq = baseline.get("trigger_frequency", 0)

        if baseline_freq > 0 and current_freq > 0:
            freq_ratio = current_freq / baseline_freq

            if freq_ratio > 5.0:  # Firing 5x more than normal
                anomalies.append(
                    PatternAnomaly(
                        anomaly_id=f"FREQ_HIGH_{sensor_id}",
                        anomaly_type="high_frequency",
                        description=f"Sensor {sensor_id} triggering {freq_ratio:.1f}x more than normal",
                        severity=ErrorSeverity.HIGH,
                        confidence=min(0.95, freq_ratio / 10),
                        detected_at=datetime.now(),
                        affected_sensors=[sensor_id],
                        statistical_measures={
                            "current_frequency": current_freq,
                            "baseline_frequency": baseline_freq,
                            "ratio": freq_ratio,
                        },
                    )
                )
            elif freq_ratio < 0.2:  # Firing 5x less than normal
                anomalies.append(
                    PatternAnomaly(
                        anomaly_id=f"FREQ_LOW_{sensor_id}",
                        anomaly_type="low_frequency",
                        description=f"Sensor {sensor_id} triggering {1/freq_ratio:.1f}x less than normal",
                        severity=ErrorSeverity.MEDIUM,
                        confidence=min(0.9, (0.2 - freq_ratio) * 5),
                        detected_at=datetime.now(),
                        affected_sensors=[sensor_id],
                        statistical_measures={
                            "current_frequency": current_freq,
                            "baseline_frequency": baseline_freq,
                            "ratio": freq_ratio,
                        },
                    )
                )

        # Check for interval pattern changes
        current_std = current_analysis.get("std_interval", 0)
        baseline_std = baseline.get("std_interval", 0)

        if baseline_std > 0 and current_std > baseline_std * 3:
            anomalies.append(
                PatternAnomaly(
                    anomaly_id=f"UNSTABLE_{sensor_id}",
                    anomaly_type="unstable_timing",
                    description=f"Sensor {sensor_id} showing unstable timing patterns",
                    severity=ErrorSeverity.MEDIUM,
                    confidence=0.8,
                    detected_at=datetime.now(),
                    affected_sensors=[sensor_id],
                    statistical_measures={
                        "current_std": current_std,
                        "baseline_std": baseline_std,
                        "stability_ratio": current_std / baseline_std,
                    },
                )
            )

        return anomalies


class CorruptionDetector:
    """Detects data corruption and integrity issues."""

    def __init__(self):
        """Initialize corruption detector."""
        self.known_corrupt_patterns = [
            # Common corruption indicators
            r"^\x00+$",  # Null bytes
            r"^[\x80-\xFF]+$",  # High ASCII junk
            r"^(.)\1{10,}$",  # Repeated characters
            r"^\d{13,}$",  # Suspiciously long numbers
        ]

    def detect_data_corruption(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Detect various forms of data corruption."""
        errors = []

        if not events:
            return errors

        # Check for timestamp corruption
        errors.extend(self._detect_timestamp_corruption(events))

        # Check for state corruption
        errors.extend(self._detect_state_corruption(events))

        # Check for ID corruption
        errors.extend(self._detect_id_corruption(events))

        # Check for encoding issues
        errors.extend(self._detect_encoding_corruption(events))

        return errors

    def _detect_timestamp_corruption(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Detect timestamp corruption patterns."""
        errors = []
        timestamps = []

        for i, event in enumerate(events):
            if "timestamp" in event and event["timestamp"]:
                try:
                    if isinstance(event["timestamp"], str):
                        ts = datetime.fromisoformat(
                            event["timestamp"].replace("Z", "+00:00")
                        )
                    else:
                        ts = event["timestamp"]
                    timestamps.append((i, ts))
                except Exception:
                    errors.append(
                        ValidationError(
                            rule_id="COR001",
                            field="timestamp",
                            value=event["timestamp"],
                            message=f"Corrupted timestamp format at event {i}",
                            severity=ErrorSeverity.HIGH,
                            suggestion="Fix timestamp format to ISO 8601",
                            context={"event_index": i},
                        )
                    )

        # Check for timestamp anomalies
        if len(timestamps) > 1:
            timestamps.sort(key=lambda x: x[1])

            for i in range(len(timestamps) - 1):
                current_idx, current_ts = timestamps[i]
                next_idx, next_ts = timestamps[i + 1]

                # Check for impossible time jumps (more than 1 year in the future)
                if (next_ts - current_ts).days > 365:
                    errors.append(
                        ValidationError(
                            rule_id="COR002",
                            field="timestamp",
                            value=f"{current_ts} -> {next_ts}",
                            message="Impossible time jump detected - potential corruption",
                            severity=ErrorSeverity.HIGH,
                            suggestion="Validate system clock and timestamp generation",
                            context={"event_indices": [current_idx, next_idx]},
                        )
                    )

                # Check for exact duplicates (potential corruption)
                if current_ts == next_ts and current_idx != next_idx:
                    errors.append(
                        ValidationError(
                            rule_id="COR003",
                            field="timestamp",
                            value=current_ts.isoformat(),
                            message="Duplicate timestamps detected - potential corruption",
                            severity=ErrorSeverity.MEDIUM,
                            suggestion="Ensure unique timestamps or add sequence numbers",
                            context={"event_indices": [current_idx, next_idx]},
                        )
                    )

        return errors

    def _detect_state_corruption(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Detect state value corruption."""
        errors = []

        valid_states = {state.value for state in SensorState}

        for i, event in enumerate(events):
            if "state" in event and event["state"] is not None:
                state = str(event["state"])

                # Check for obviously corrupted states
                if len(state) > 20:  # States shouldn't be very long
                    errors.append(
                        ValidationError(
                            rule_id="COR004",
                            field="state",
                            value=state[:50],
                            message=f"Suspiciously long state value at event {i}",
                            severity=ErrorSeverity.MEDIUM,
                            suggestion="Validate state value generation",
                            context={"event_index": i, "state_length": len(state)},
                        )
                    )

                # Check for non-printable characters
                if any(
                    ord(c) < 32 or ord(c) > 126
                    for c in state
                    if c != "\t" and c != "\n"
                ):
                    errors.append(
                        ValidationError(
                            rule_id="COR005",
                            field="state",
                            value=repr(state),
                            message=f"Non-printable characters in state at event {i}",
                            severity=ErrorSeverity.HIGH,
                            suggestion="Check data encoding and transmission",
                            context={"event_index": i},
                        )
                    )

        return errors

    def _detect_id_corruption(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Detect ID field corruption."""
        errors = []

        for i, event in enumerate(events):
            # Check room_id corruption
            if "room_id" in event and event["room_id"] is not None:
                room_id = str(event["room_id"])
                if len(room_id) > 100 or any(ord(c) < 32 for c in room_id):
                    errors.append(
                        ValidationError(
                            rule_id="COR006",
                            field="room_id",
                            value=room_id[:50],
                            message=f"Corrupted room_id at event {i}",
                            severity=ErrorSeverity.HIGH,
                            suggestion="Validate room_id generation and encoding",
                            context={"event_index": i},
                        )
                    )

            # Check sensor_id corruption
            if "sensor_id" in event and event["sensor_id"] is not None:
                sensor_id = str(event["sensor_id"])
                if len(sensor_id) > 200 or any(ord(c) < 32 for c in sensor_id):
                    errors.append(
                        ValidationError(
                            rule_id="COR007",
                            field="sensor_id",
                            value=sensor_id[:50],
                            message=f"Corrupted sensor_id at event {i}",
                            severity=ErrorSeverity.HIGH,
                            suggestion="Validate sensor_id generation and encoding",
                            context={"event_index": i},
                        )
                    )

        return errors

    def _detect_encoding_corruption(
        self, events: List[Dict[str, Any]]
    ) -> List[ValidationError]:
        """Detect character encoding corruption."""
        errors = []

        for i, event in enumerate(events):
            for field, value in event.items():
                if isinstance(value, str) and value:
                    # Check for common encoding issues
                    if "�" in value:  # Replacement character indicates encoding issues
                        errors.append(
                            ValidationError(
                                rule_id="COR008",
                                field=field,
                                value=value[:50],
                                message=f"Encoding corruption detected in {field} at event {i}",
                                severity=ErrorSeverity.MEDIUM,
                                suggestion="Fix character encoding in data pipeline",
                                context={"event_index": i},
                            )
                        )

                    # Check for mixed encodings
                    try:
                        value.encode("utf-8")
                    except UnicodeEncodeError:
                        errors.append(
                            ValidationError(
                                rule_id="COR009",
                                field=field,
                                value=repr(value[:50]),
                                message=f"Unicode encoding error in {field} at event {i}",
                                severity=ErrorSeverity.HIGH,
                                suggestion="Ensure consistent UTF-8 encoding",
                                context={"event_index": i},
                            )
                        )

        return errors


class RealTimeQualityMonitor:
    """Real-time monitoring of data quality metrics."""

    def __init__(self, window_minutes: int = 60):
        """Initialize real-time quality monitor."""
        self.window_minutes = window_minutes
        self.quality_history = deque(maxlen=1000)
        self.sensor_quality = defaultdict(lambda: deque(maxlen=100))
        self.alert_thresholds = {
            "completeness": 0.8,
            "consistency": 0.7,
            "accuracy": 0.75,
            "timeliness": 0.9,
        }

    def calculate_quality_metrics(
        self, events: List[Dict[str, Any]], expected_sensors: Set[str]
    ) -> DataQualityMetrics:
        """Calculate comprehensive data quality metrics."""
        if not events:
            return DataQualityMetrics(
                completeness_score=0.0,
                consistency_score=0.0,
                accuracy_score=0.0,
                timeliness_score=0.0,
                anomaly_count=0,
            )

        # Calculate completeness
        actual_sensors = {
            event.get("sensor_id") for event in events if event.get("sensor_id")
        }
        completeness_score = len(actual_sensors & expected_sensors) / max(
            len(expected_sensors), 1
        )

        # Calculate consistency
        consistency_score = self._calculate_consistency_score(events)

        # Calculate accuracy based on expected patterns
        accuracy_score = self._calculate_accuracy_score(events)

        # Calculate timeliness
        timeliness_score = self._calculate_timeliness_score(events)

        # Count anomalies
        analyzer = StatisticalPatternAnalyzer()
        anomaly_count = 0
        for sensor_id in actual_sensors:
            sensor_events = [e for e in events if e.get("sensor_id") == sensor_id]
            anomalies = analyzer.detect_sensor_malfunction(sensor_id, sensor_events)
            anomaly_count += len(anomalies)

        # Detect corruption
        detector = CorruptionDetector()
        corruption_errors = detector.detect_data_corruption(events)
        corruption_indicators = [error.rule_id for error in corruption_errors]

        metrics = DataQualityMetrics(
            completeness_score=completeness_score,
            consistency_score=consistency_score,
            accuracy_score=accuracy_score,
            timeliness_score=timeliness_score,
            anomaly_count=anomaly_count,
            corruption_indicators=corruption_indicators,
        )

        # Store for trending
        self.quality_history.append(
            {
                "timestamp": datetime.now(),
                "completeness": completeness_score,
                "consistency": consistency_score,
                "accuracy": accuracy_score,
                "timeliness": timeliness_score,
                "anomalies": anomaly_count,
            }
        )

        return metrics

    def _calculate_consistency_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate consistency score across sensors."""
        if not events:
            return 0.0

        # Group events by sensor
        sensor_events = defaultdict(list)
        for event in events:
            sensor_id = event.get("sensor_id")
            if sensor_id:
                sensor_events[sensor_id].append(event)

        if len(sensor_events) < 2:
            return 1.0  # Only one sensor, perfectly consistent

        # Calculate consistency metrics
        consistency_scores = []

        for sensor_id, sensor_event_list in sensor_events.items():
            if len(sensor_event_list) > 1:
                # Check timing consistency
                intervals = []
                timestamps = []

                for event in sensor_event_list:
                    if "timestamp" in event:
                        try:
                            if isinstance(event["timestamp"], str):
                                ts = datetime.fromisoformat(
                                    event["timestamp"].replace("Z", "+00:00")
                                )
                            else:
                                ts = event["timestamp"]
                            timestamps.append(ts)
                        except Exception:
                            continue

                timestamps.sort()
                for i in range(1, len(timestamps)):
                    interval = (timestamps[i] - timestamps[i - 1]).total_seconds()
                    intervals.append(interval)

                if len(intervals) > 1:
                    # Lower coefficient of variation = higher consistency
                    mean_interval = statistics.mean(intervals)
                    std_interval = statistics.stdev(intervals)

                    if mean_interval > 0:
                        cv = std_interval / mean_interval
                        consistency_score = max(0.0, 1.0 - min(cv, 1.0))
                        consistency_scores.append(consistency_score)

        return statistics.mean(consistency_scores) if consistency_scores else 0.5

    def _calculate_accuracy_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score based on expected patterns."""
        if not events:
            return 0.0

        accuracy_factors = []

        # Check state validity
        valid_states = {state.value for state in SensorState}
        valid_state_count = sum(
            1 for event in events if event.get("state") in valid_states
        )
        state_accuracy = valid_state_count / len(events)
        accuracy_factors.append(state_accuracy)

        # Check sensor type validity
        valid_sensor_types = {st.value for st in SensorType}
        valid_type_count = sum(
            1 for event in events if event.get("sensor_type") in valid_sensor_types
        )
        type_accuracy = valid_type_count / len(events)
        accuracy_factors.append(type_accuracy)

        # Check for reasonable timestamp ordering
        timestamps = []
        for event in events:
            if "timestamp" in event:
                try:
                    if isinstance(event["timestamp"], str):
                        ts = datetime.fromisoformat(
                            event["timestamp"].replace("Z", "+00:00")
                        )
                    else:
                        ts = event["timestamp"]
                    timestamps.append(ts)
                except Exception:
                    continue

        if len(timestamps) > 1:
            ordered_count = 0
            sorted_timestamps = sorted(timestamps)
            for i, ts in enumerate(timestamps):
                if i == 0 or ts >= timestamps[i - 1]:
                    ordered_count += 1

            ordering_accuracy = ordered_count / len(timestamps)
            accuracy_factors.append(ordering_accuracy)

        return statistics.mean(accuracy_factors) if accuracy_factors else 0.0

    def _calculate_timeliness_score(self, events: List[Dict[str, Any]]) -> float:
        """Calculate timeliness score based on data freshness."""
        if not events:
            return 0.0

        now = datetime.now(timezone.utc)
        timeliness_scores = []

        for event in events:
            if "timestamp" in event:
                try:
                    if isinstance(event["timestamp"], str):
                        ts = datetime.fromisoformat(
                            event["timestamp"].replace("Z", "+00:00")
                        )
                    else:
                        ts = event["timestamp"]

                    if ts.tzinfo is None:
                        ts = ts.replace(tzinfo=timezone.utc)

                    # Calculate age in hours
                    age_hours = (now - ts).total_seconds() / 3600

                    # Exponential decay - data older than 24 hours gets low scores
                    if age_hours < 0:  # Future timestamp
                        score = 0.5
                    elif age_hours < 1:  # Less than 1 hour old
                        score = 1.0
                    elif age_hours < 6:  # Less than 6 hours old
                        score = 0.8
                    elif age_hours < 24:  # Less than 24 hours old
                        score = 0.6
                    else:  # Older than 24 hours
                        score = max(0.1, 0.6 * math.exp(-(age_hours - 24) / 24))

                    timeliness_scores.append(score)

                except Exception:
                    timeliness_scores.append(0.1)  # Invalid timestamp gets low score

        return statistics.mean(timeliness_scores) if timeliness_scores else 0.0

    def get_quality_trends(self, hours: int = 24) -> Dict[str, List[float]]:
        """Get quality trends over the specified time period."""
        cutoff_time = datetime.now() - timedelta(hours=hours)

        recent_history = [
            entry for entry in self.quality_history if entry["timestamp"] >= cutoff_time
        ]

        trends = {
            "completeness": [entry["completeness"] for entry in recent_history],
            "consistency": [entry["consistency"] for entry in recent_history],
            "accuracy": [entry["accuracy"] for entry in recent_history],
            "timeliness": [entry["timeliness"] for entry in recent_history],
            "anomalies": [entry["anomalies"] for entry in recent_history],
            "timestamps": [entry["timestamp"].isoformat() for entry in recent_history],
        }

        return trends

    def detect_quality_alerts(
        self, current_metrics: DataQualityMetrics
    ) -> List[PatternAnomaly]:
        """Detect quality degradation alerts."""
        alerts = []

        # Check against thresholds
        if current_metrics.completeness_score < self.alert_thresholds["completeness"]:
            alerts.append(
                PatternAnomaly(
                    anomaly_id="QUAL001",
                    anomaly_type="data_completeness",
                    description=f"Data completeness below threshold ({current_metrics.completeness_score:.2%})",
                    severity=ErrorSeverity.MEDIUM,
                    confidence=0.9,
                    detected_at=datetime.now(),
                    affected_sensors=[],
                    statistical_measures={
                        "completeness_score": current_metrics.completeness_score
                    },
                )
            )

        if current_metrics.consistency_score < self.alert_thresholds["consistency"]:
            alerts.append(
                PatternAnomaly(
                    anomaly_id="QUAL002",
                    anomaly_type="data_consistency",
                    description=f"Data consistency below threshold ({current_metrics.consistency_score:.2%})",
                    severity=ErrorSeverity.MEDIUM,
                    confidence=0.8,
                    detected_at=datetime.now(),
                    affected_sensors=[],
                    statistical_measures={
                        "consistency_score": current_metrics.consistency_score
                    },
                )
            )

        if current_metrics.accuracy_score < self.alert_thresholds["accuracy"]:
            alerts.append(
                PatternAnomaly(
                    anomaly_id="QUAL003",
                    anomaly_type="data_accuracy",
                    description=f"Data accuracy below threshold ({current_metrics.accuracy_score:.2%})",
                    severity=ErrorSeverity.HIGH,
                    confidence=0.85,
                    detected_at=datetime.now(),
                    affected_sensors=[],
                    statistical_measures={
                        "accuracy_score": current_metrics.accuracy_score
                    },
                )
            )

        if current_metrics.timeliness_score < self.alert_thresholds["timeliness"]:
            alerts.append(
                PatternAnomaly(
                    anomaly_id="QUAL004",
                    anomaly_type="data_timeliness",
                    description=f"Data timeliness below threshold ({current_metrics.timeliness_score:.2%})",
                    severity=ErrorSeverity.MEDIUM,
                    confidence=0.7,
                    detected_at=datetime.now(),
                    affected_sensors=[],
                    statistical_measures={
                        "timeliness_score": current_metrics.timeliness_score
                    },
                )
            )

        return alerts
