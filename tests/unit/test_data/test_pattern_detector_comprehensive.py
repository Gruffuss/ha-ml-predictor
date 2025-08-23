"""
Comprehensive test suite for PatternDetector with anomaly detection validation.

This test suite provides complete coverage of pattern detection functionality including:
- Statistical pattern analysis with mathematical accuracy verification
- Sensor malfunction detection and behavioral anomaly identification
- Data corruption detection and integrity validation
- Real-time quality monitoring and metric calculation
- Performance testing with large datasets
- Edge case handling and boundary condition testing
"""

import asyncio
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import statistics

from src.core.constants import SensorState, SensorType
from src.core.exceptions import ErrorSeverity
from src.data.validation.event_validator import ValidationError
from src.data.validation.pattern_detector import (
    CorruptionDetector,
    DataQualityMetrics,
    PatternAnomaly,
    RealTimeQualityMonitor,
    StatisticalPatternAnalyzer,
)


@pytest.fixture
def pattern_analyzer():
    """Create StatisticalPatternAnalyzer instance for testing."""
    return StatisticalPatternAnalyzer(window_size=50, confidence_level=0.95)


@pytest.fixture
def corruption_detector():
    """Create CorruptionDetector instance for testing."""
    return CorruptionDetector()


@pytest.fixture
def quality_monitor():
    """Create RealTimeQualityMonitor instance for testing."""
    return RealTimeQualityMonitor(window_minutes=60)


@pytest.fixture
def sample_sensor_events():
    """Create sample sensor events for testing."""
    base_time = datetime.now(timezone.utc)
    events = []

    for i in range(100):
        # Create realistic sensor event patterns
        state = "on" if i % 10 < 3 else "off"  # 30% on, 70% off
        timestamp = base_time + timedelta(seconds=i * 30)  # Every 30 seconds

        events.append(
            {
                "sensor_id": f"sensor.motion_{i % 5}",
                "room_id": f"room_{i % 3}",
                "sensor_type": "motion",
                "state": state,
                "timestamp": timestamp,
                "attributes": {
                    "brightness": 50 + (i % 50),
                    "temperature": 20 + (i % 10),
                },
            }
        )

    return events


@pytest.fixture
def anomalous_sensor_events():
    """Create sensor events with anomalies for testing."""
    base_time = datetime.now(timezone.utc)
    events = []

    # First 50 events are normal
    for i in range(50):
        events.append(
            {
                "sensor_id": "sensor.motion_test",
                "room_id": "test_room",
                "sensor_type": "motion",
                "state": "on" if i % 10 < 3 else "off",
                "timestamp": base_time + timedelta(seconds=i * 60),  # Every minute
                "attributes": {"brightness": 50},
            }
        )

    # Next 50 events show anomalous behavior
    for i in range(50, 100):
        events.append(
            {
                "sensor_id": "sensor.motion_test",
                "room_id": "test_room",
                "sensor_type": "motion",
                "state": "on",  # Stuck on (anomalous)
                "timestamp": base_time + timedelta(seconds=i * 5),  # Much more frequent
                "attributes": {"brightness": 50},
            }
        )

    return events


@pytest.fixture
def corrupted_events():
    """Create events with various types of corruption for testing."""
    base_time = datetime.now(timezone.utc)

    return [
        # Timestamp corruption
        {
            "sensor_id": "sensor.test",
            "room_id": "room_1",
            "state": "on",
            "timestamp": "invalid_timestamp",
        },
        {
            "sensor_id": "sensor.test",
            "room_id": "room_1",
            "state": "on",
            "timestamp": base_time + timedelta(days=500),  # Far future
        },
        # ID corruption
        {
            "sensor_id": "sensor\x00corrupt\x00id",
            "room_id": "room\x00\x01corrupt",
            "state": "on",
            "timestamp": base_time,
        },
        # State corruption
        {
            "sensor_id": "sensor.test",
            "room_id": "room_1",
            "state": "\x80\x90\xA0",  # Non-printable characters
            "timestamp": base_time,
        },
        # Encoding corruption
        {
            "sensor_id": "sensor.test",
            "room_id": "room_1",
            "state": "text_with_replacement_char_�",
            "timestamp": base_time,
        },
    ]


class TestStatisticalPatternAnalyzer:
    """Comprehensive tests for StatisticalPatternAnalyzer."""

    def test_initialization(self, pattern_analyzer):
        """Test proper initialization of pattern analyzer."""
        assert pattern_analyzer.window_size == 50
        assert pattern_analyzer.confidence_level == 0.95
        assert isinstance(pattern_analyzer.sensor_baselines, defaultdict)
        assert isinstance(pattern_analyzer.pattern_cache, dict)

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        analyzer = StatisticalPatternAnalyzer(window_size=100, confidence_level=0.99)
        assert analyzer.window_size == 100
        assert analyzer.confidence_level == 0.99

    def test_analyze_sensor_behavior_normal_data(
        self, pattern_analyzer, sample_sensor_events
    ):
        """Test sensor behavior analysis with normal data."""
        # Filter events for specific sensor
        sensor_events = [
            e for e in sample_sensor_events if e["sensor_id"] == "sensor.motion_0"
        ]

        analysis = pattern_analyzer.analyze_sensor_behavior(
            "sensor.motion_0", sensor_events
        )

        assert isinstance(analysis, dict)
        assert "event_count" in analysis
        assert "time_span_hours" in analysis
        assert "mean_interval" in analysis
        assert "median_interval" in analysis
        assert "std_interval" in analysis
        assert "state_distribution" in analysis
        assert "trigger_frequency" in analysis

        # Verify mathematical accuracy
        assert analysis["event_count"] == len(sensor_events)
        assert analysis["time_span_hours"] > 0
        assert analysis["trigger_frequency"] > 0

        # Verify state distribution
        state_dist = analysis["state_distribution"]
        assert isinstance(state_dist, dict)
        assert sum(state_dist.values()) == pytest.approx(1.0, abs=1e-6)

    def test_analyze_sensor_behavior_insufficient_data(self, pattern_analyzer):
        """Test sensor behavior analysis with insufficient data."""
        # Single event
        single_event = [
            {"sensor_id": "sensor.test", "timestamp": datetime.now(), "state": "on"}
        ]

        analysis = pattern_analyzer.analyze_sensor_behavior("sensor.test", single_event)
        assert "error" in analysis
        assert "Insufficient data" in analysis["error"]

        # No events
        empty_events = []
        analysis = pattern_analyzer.analyze_sensor_behavior("sensor.test", empty_events)
        assert "error" in analysis
        assert "No events provided" in analysis["error"]

    def test_state_distribution_calculation_accuracy(self, pattern_analyzer):
        """Test mathematical accuracy of state distribution calculation."""
        # Create known distribution: 7 on, 3 off
        events = []
        base_time = datetime.now()

        for i in range(10):
            state = "on" if i < 7 else "off"
            events.append(
                {
                    "sensor_id": "sensor.test",
                    "timestamp": base_time + timedelta(seconds=i),
                    "state": state,
                }
            )

        # Use private method directly for testing
        distribution = pattern_analyzer._calculate_state_distribution(
            [e["state"] for e in events]
        )

        assert distribution["on"] == pytest.approx(0.7, abs=1e-6)
        assert distribution["off"] == pytest.approx(0.3, abs=1e-6)
        assert sum(distribution.values()) == pytest.approx(1.0, abs=1e-6)

    def test_statistical_anomaly_detection_normal_data(self, pattern_analyzer):
        """Test statistical anomaly detection with normal data."""
        # Create intervals following normal distribution
        np.random.seed(42)
        intervals = np.random.normal(60, 10, 100).tolist()  # Mean=60, std=10

        anomalies = pattern_analyzer._detect_statistical_anomalies(intervals)

        assert isinstance(anomalies, dict)
        assert "outliers" in anomalies
        assert "z_scores" in anomalies
        assert "anomaly_count" in anomalies
        assert "distribution_type" in anomalies

        # With normal data, should have few outliers
        assert anomalies["anomaly_count"] < len(intervals) * 0.1  # Less than 10%
        assert len(anomalies["z_scores"]) == len(intervals)

    def test_statistical_anomaly_detection_with_outliers(self, pattern_analyzer):
        """Test statistical anomaly detection with known outliers."""
        # Create data with clear outliers
        intervals = [10.0] * 50 + [1000.0, 2000.0, 3000.0]  # 3 extreme outliers

        anomalies = pattern_analyzer._detect_statistical_anomalies(intervals)

        # Should detect the 3 outliers
        assert anomalies["anomaly_count"] == 3
        assert len(anomalies["outliers"]) == 3

        # Verify outliers are the expected ones
        outlier_values = [o["value"] for o in anomalies["outliers"]]
        assert 1000.0 in outlier_values
        assert 2000.0 in outlier_values
        assert 3000.0 in outlier_values

    def test_statistical_anomaly_z_score_calculation(self, pattern_analyzer):
        """Test mathematical accuracy of z-score calculation."""
        # Known data for exact z-score calculation
        intervals = [8.0, 9.0, 10.0, 11.0, 12.0, 20.0]  # Last value is outlier

        anomalies = pattern_analyzer._detect_statistical_anomalies(intervals)

        # Calculate expected z-score for last value
        mean_val = statistics.mean(intervals)
        std_val = statistics.stdev(intervals)
        expected_z_score = abs((20.0 - mean_val) / std_val)

        # Find z-score for the outlier value
        outlier_z_score = None
        for i, z_score in enumerate(anomalies["z_scores"]):
            if intervals[i] == 20.0:
                outlier_z_score = z_score
                break

        assert outlier_z_score is not None
        assert outlier_z_score == pytest.approx(expected_z_score, abs=1e-6)

    def test_normality_test_integration(self, pattern_analyzer):
        """Test normality testing integration."""
        # Create clearly normal data
        np.random.seed(42)
        normal_intervals = np.random.normal(50, 5, 50).tolist()

        anomalies = pattern_analyzer._detect_statistical_anomalies(normal_intervals)

        # Should classify as normal or unknown (depending on p-value)
        assert anomalies["distribution_type"] in ["normal", "non_normal", "unknown"]

    def test_detect_sensor_malfunction_high_frequency(
        self, pattern_analyzer, anomalous_sensor_events
    ):
        """Test detection of high-frequency sensor malfunctions."""
        sensor_id = "sensor.motion_test"

        # Analyze normal behavior first to establish baseline
        normal_events = anomalous_sensor_events[:50]
        pattern_analyzer.analyze_sensor_behavior(sensor_id, normal_events)

        # Analyze anomalous behavior
        anomalous_events = anomalous_sensor_events[50:]
        anomalies = pattern_analyzer.detect_sensor_malfunction(
            sensor_id, anomalous_events
        )

        assert len(anomalies) > 0

        # Should detect high frequency anomaly
        freq_anomalies = [a for a in anomalies if a.anomaly_type == "high_frequency"]
        assert len(freq_anomalies) > 0

        freq_anomaly = freq_anomalies[0]
        assert freq_anomaly.severity == ErrorSeverity.HIGH
        assert sensor_id in freq_anomaly.affected_sensors
        assert "frequency" in freq_anomaly.statistical_measures

    def test_detect_sensor_malfunction_low_frequency(self, pattern_analyzer):
        """Test detection of low-frequency sensor malfunctions."""
        sensor_id = "sensor.motion_test"
        base_time = datetime.now()

        # Create baseline with normal frequency
        normal_events = []
        for i in range(50):
            normal_events.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": base_time + timedelta(minutes=i),
                    "state": "on" if i % 5 == 0 else "off",
                }
            )

        pattern_analyzer.analyze_sensor_behavior(sensor_id, normal_events)

        # Create low-frequency events (much less frequent)
        low_freq_events = []
        for i in range(5):
            low_freq_events.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": base_time
                    + timedelta(hours=50 + i * 24),  # Once per day
                    "state": "on",
                }
            )

        anomalies = pattern_analyzer.detect_sensor_malfunction(
            sensor_id, low_freq_events
        )

        # Should detect low frequency anomaly
        low_freq_anomalies = [a for a in anomalies if a.anomaly_type == "low_frequency"]
        assert len(low_freq_anomalies) > 0

        anomaly = low_freq_anomalies[0]
        assert anomaly.severity == ErrorSeverity.MEDIUM
        assert "frequency" in anomaly.statistical_measures

    def test_detect_sensor_malfunction_unstable_timing(self, pattern_analyzer):
        """Test detection of unstable timing patterns."""
        sensor_id = "sensor.timing_test"
        base_time = datetime.now()

        # Create baseline with stable timing
        stable_events = []
        for i in range(50):
            stable_events.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": base_time
                    + timedelta(seconds=i * 60),  # Consistent 1-minute intervals
                    "state": "on" if i % 2 == 0 else "off",
                }
            )

        pattern_analyzer.analyze_sensor_behavior(sensor_id, stable_events)

        # Create events with unstable timing
        unstable_events = []
        intervals = [5, 120, 10, 300, 2, 180, 1, 600]  # Very inconsistent intervals
        current_time = base_time + timedelta(hours=1)

        for i, interval in enumerate(intervals):
            unstable_events.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": current_time,
                    "state": "on" if i % 2 == 0 else "off",
                }
            )
            current_time += timedelta(seconds=interval)

        anomalies = pattern_analyzer.detect_sensor_malfunction(
            sensor_id, unstable_events
        )

        # Should detect unstable timing
        timing_anomalies = [a for a in anomalies if a.anomaly_type == "unstable_timing"]
        assert len(timing_anomalies) > 0

        anomaly = timing_anomalies[0]
        assert anomaly.severity == ErrorSeverity.MEDIUM
        assert "stability_ratio" in anomaly.statistical_measures

    def test_detect_sensor_malfunction_no_baseline(self, pattern_analyzer):
        """Test sensor malfunction detection without baseline data."""
        sensor_id = "new_sensor"

        events = [{"sensor_id": sensor_id, "timestamp": datetime.now(), "state": "on"}]

        # Should return empty list without baseline
        anomalies = pattern_analyzer.detect_sensor_malfunction(sensor_id, events)
        assert len(anomalies) == 0

    def test_pattern_baseline_persistence(self, pattern_analyzer):
        """Test that sensor baselines are properly stored and reused."""
        sensor_id = "persistent_sensor"

        events = [
            {"sensor_id": sensor_id, "timestamp": datetime.now(), "state": "on"},
            {
                "sensor_id": sensor_id,
                "timestamp": datetime.now() + timedelta(seconds=30),
                "state": "off",
            },
        ]

        # First analysis should create baseline
        analysis1 = pattern_analyzer.analyze_sensor_behavior(sensor_id, events)
        assert sensor_id in pattern_analyzer.sensor_baselines

        # Baseline should contain the analysis results
        baseline = pattern_analyzer.sensor_baselines[sensor_id]
        assert baseline["event_count"] == analysis1["event_count"]
        assert baseline["trigger_frequency"] == analysis1["trigger_frequency"]

    def test_edge_case_identical_timestamps(self, pattern_analyzer):
        """Test handling of events with identical timestamps."""
        sensor_id = "identical_timestamps_sensor"
        timestamp = datetime.now()

        events = []
        for i in range(5):
            events.append(
                {
                    "sensor_id": sensor_id,
                    "timestamp": timestamp,  # All identical
                    "state": "on" if i % 2 == 0 else "off",
                }
            )

        analysis = pattern_analyzer.analyze_sensor_behavior(sensor_id, events)

        # Should handle gracefully without crashing
        assert isinstance(analysis, dict)
        # Intervals will be all zeros, which should be handled
        assert "mean_interval" in analysis
        assert analysis["mean_interval"] == 0.0

    def test_edge_case_very_large_intervals(self, pattern_analyzer):
        """Test handling of very large time intervals."""
        sensor_id = "large_intervals_sensor"
        base_time = datetime.now()

        events = [
            {"sensor_id": sensor_id, "timestamp": base_time, "state": "on"},
            {
                "sensor_id": sensor_id,
                "timestamp": base_time + timedelta(days=365),  # 1 year later
                "state": "off",
            },
        ]

        analysis = pattern_analyzer.analyze_sensor_behavior(sensor_id, events)

        # Should handle large intervals
        assert analysis["mean_interval"] == 365 * 24 * 3600  # Seconds in a year
        assert analysis["time_span_hours"] == 365 * 24


class TestCorruptionDetector:
    """Comprehensive tests for CorruptionDetector."""

    def test_initialization(self, corruption_detector):
        """Test CorruptionDetector initialization."""
        assert len(corruption_detector.known_corrupt_patterns) > 0

        # Verify patterns are valid regex
        import re

        for pattern in corruption_detector.known_corrupt_patterns:
            try:
                re.compile(pattern)
            except re.error:
                pytest.fail(f"Invalid regex pattern: {pattern}")

    def test_detect_data_corruption_normal_events(
        self, corruption_detector, sample_sensor_events
    ):
        """Test corruption detection with normal events."""
        errors = corruption_detector.detect_data_corruption(sample_sensor_events)

        # Normal events should have no corruption errors
        assert len(errors) == 0

    def test_detect_timestamp_corruption(self, corruption_detector):
        """Test timestamp corruption detection."""
        corrupted_events = [
            {"sensor_id": "sensor.test", "timestamp": "not-a-timestamp", "state": "on"},
            {
                "sensor_id": "sensor.test",
                "timestamp": datetime.now() + timedelta(days=500),  # Far future
                "state": "on",
            },
            {"sensor_id": "sensor.test", "timestamp": datetime.now(), "state": "on"},
            {
                "sensor_id": "sensor.test",
                "timestamp": datetime.now(),  # Duplicate timestamp
                "state": "off",
            },
        ]

        errors = corruption_detector.detect_data_corruption(corrupted_events)

        # Should detect timestamp issues
        timestamp_errors = [e for e in errors if e.rule_id.startswith("COR00")]
        assert len(timestamp_errors) > 0

        # Check specific error types
        error_rules = {e.rule_id for e in timestamp_errors}
        assert "COR001" in error_rules  # Invalid format
        assert "COR002" in error_rules  # Impossible time jump
        assert "COR003" in error_rules  # Duplicate timestamps

    def test_detect_state_corruption(self, corruption_detector):
        """Test state value corruption detection."""
        corrupted_events = [
            {
                "sensor_id": "sensor.test",
                "state": "extremely_long_state_value_that_exceeds_normal_length_limits",
                "timestamp": datetime.now(),
            },
            {
                "sensor_id": "sensor.test",
                "state": "state\x00with\x01non\x02printable\x03chars",
                "timestamp": datetime.now(),
            },
        ]

        errors = corruption_detector.detect_data_corruption(corrupted_events)

        # Should detect state corruption
        state_errors = [e for e in errors if e.rule_id in ["COR004", "COR005"]]
        assert len(state_errors) > 0

        # Verify specific detections
        long_state_errors = [e for e in state_errors if "long state value" in e.message]
        assert len(long_state_errors) > 0

        non_printable_errors = [
            e for e in state_errors if "Non-printable characters" in e.message
        ]
        assert len(non_printable_errors) > 0

    def test_detect_id_corruption(self, corruption_detector):
        """Test ID field corruption detection."""
        corrupted_events = [
            {
                "room_id": "room\x00with\x01null\x02bytes"
                + "x" * 150,  # Too long + non-printable
                "sensor_id": "sensor" + "\x80\x90\xA0",  # High ASCII
                "state": "on",
                "timestamp": datetime.now(),
            }
        ]

        errors = corruption_detector.detect_data_corruption(corrupted_events)

        # Should detect ID corruption
        id_errors = [e for e in errors if e.rule_id in ["COR006", "COR007"]]
        assert len(id_errors) > 0

    def test_detect_encoding_corruption(self, corruption_detector):
        """Test character encoding corruption detection."""
        corrupted_events = [
            {
                "sensor_id": "sensor.test",
                "room_id": "room_with_replacement_char_�",  # Unicode replacement character
                "state": "on",
                "timestamp": datetime.now(),
                "attributes": {"description": "text_with_encoding_issues_�_here"},
            }
        ]

        errors = corruption_detector.detect_data_corruption(corrupted_events)

        # Should detect encoding issues
        encoding_errors = [e for e in errors if e.rule_id in ["COR008", "COR009"]]
        assert len(encoding_errors) > 0

        # Verify replacement character detection
        replacement_errors = [
            e for e in encoding_errors if "Encoding corruption" in e.message
        ]
        assert len(replacement_errors) > 0

    def test_comprehensive_corruption_detection(
        self, corruption_detector, corrupted_events
    ):
        """Test comprehensive corruption detection with various corruption types."""
        errors = corruption_detector.detect_data_corruption(corrupted_events)

        # Should detect multiple types of corruption
        assert len(errors) > 0

        # Verify different error types are detected
        error_rules = {e.rule_id for e in errors}
        corruption_rules = {rule for rule in error_rules if rule.startswith("COR")}
        assert len(corruption_rules) > 1  # Multiple corruption types detected

    def test_empty_events_handling(self, corruption_detector):
        """Test handling of empty event list."""
        errors = corruption_detector.detect_data_corruption([])
        assert len(errors) == 0

    def test_unicode_handling(self, corruption_detector):
        """Test proper Unicode string handling."""
        unicode_events = [
            {
                "sensor_id": "sensor.température",  # Accented characters
                "room_id": "salon_télévision",  # Accented characters
                "state": "activé",  # Accented characters
                "timestamp": datetime.now(),
                "attributes": {"description": "Capteur de température 🌡️"},  # Emoji
            }
        ]

        # Should not flag legitimate Unicode as corruption
        errors = corruption_detector.detect_data_corruption(unicode_events)
        unicode_errors = [e for e in errors if "encoding" in e.message.lower()]
        assert len(unicode_errors) == 0

    def test_null_and_none_handling(self, corruption_detector):
        """Test handling of null and None values."""
        null_events = [
            {
                "sensor_id": None,
                "room_id": "room_1",
                "state": None,
                "timestamp": datetime.now(),
            }
        ]

        # Should handle None values gracefully
        errors = corruption_detector.detect_data_corruption(null_events)

        # Should not crash and may or may not flag as corruption
        assert isinstance(errors, list)


class TestRealTimeQualityMonitor:
    """Comprehensive tests for RealTimeQualityMonitor."""

    def test_initialization(self, quality_monitor):
        """Test RealTimeQualityMonitor initialization."""
        assert quality_monitor.window_minutes == 60
        assert isinstance(quality_monitor.quality_history, deque)
        assert isinstance(quality_monitor.sensor_quality, defaultdict)
        assert isinstance(quality_monitor.alert_thresholds, dict)

        # Verify threshold values
        assert quality_monitor.alert_thresholds["completeness"] > 0
        assert quality_monitor.alert_thresholds["consistency"] > 0
        assert quality_monitor.alert_thresholds["accuracy"] > 0
        assert quality_monitor.alert_thresholds["timeliness"] > 0

    def test_calculate_quality_metrics_empty_events(self, quality_monitor):
        """Test quality metrics calculation with empty events."""
        expected_sensors = {"sensor.motion_1", "sensor.door_1", "sensor.temp_1"}

        metrics = quality_monitor.calculate_quality_metrics([], expected_sensors)

        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.completeness_score == 0.0
        assert metrics.consistency_score == 0.0
        assert metrics.accuracy_score == 0.0
        assert metrics.timeliness_score == 0.0
        assert metrics.anomaly_count == 0

    def test_calculate_quality_metrics_perfect_data(
        self, quality_monitor, sample_sensor_events
    ):
        """Test quality metrics calculation with high-quality data."""
        # Extract expected sensors from sample events
        expected_sensors = {event["sensor_id"] for event in sample_sensor_events}

        metrics = quality_monitor.calculate_quality_metrics(
            sample_sensor_events, expected_sensors
        )

        assert isinstance(metrics, DataQualityMetrics)
        assert metrics.completeness_score == 1.0  # All expected sensors present
        assert metrics.consistency_score > 0.5  # Should have decent consistency
        assert metrics.accuracy_score > 0.8  # Valid states and types
        assert metrics.timeliness_score > 0.5  # Recent data
        assert metrics.anomaly_count >= 0

    def test_completeness_score_calculation(self, quality_monitor):
        """Test completeness score calculation accuracy."""
        expected_sensors = {"sensor.a", "sensor.b", "sensor.c", "sensor.d"}

        # Events covering only 2 out of 4 expected sensors
        partial_events = [
            {"sensor_id": "sensor.a", "state": "on", "timestamp": datetime.now()},
            {"sensor_id": "sensor.b", "state": "off", "timestamp": datetime.now()},
        ]

        metrics = quality_monitor.calculate_quality_metrics(
            partial_events, expected_sensors
        )

        # Completeness should be 2/4 = 0.5
        assert metrics.completeness_score == pytest.approx(0.5, abs=1e-6)

    def test_consistency_score_calculation(self, quality_monitor):
        """Test consistency score calculation with known patterns."""
        # Create events with consistent timing for one sensor
        base_time = datetime.now()
        consistent_events = []

        # Sensor 1: very consistent 60-second intervals
        for i in range(10):
            consistent_events.append(
                {
                    "sensor_id": "sensor.consistent",
                    "timestamp": base_time + timedelta(seconds=i * 60),
                    "state": "on" if i % 2 == 0 else "off",
                }
            )

        # Sensor 2: inconsistent intervals
        for i in range(10):
            interval = 30 if i < 5 else 300  # First half 30s, second half 300s
            consistent_events.append(
                {
                    "sensor_id": "sensor.inconsistent",
                    "timestamp": base_time + timedelta(seconds=i * interval + 1000),
                    "state": "on" if i % 2 == 0 else "off",
                }
            )

        metrics = quality_monitor.calculate_quality_metrics(consistent_events, set())

        # Should have some consistency (not perfect due to inconsistent sensor)
        assert 0.0 <= metrics.consistency_score <= 1.0

    def test_accuracy_score_calculation(self, quality_monitor):
        """Test accuracy score calculation with known data quality."""
        # Mix of valid and invalid data
        mixed_events = [
            # Valid events
            {
                "sensor_id": "sensor.a",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(),
            },
            {
                "sensor_id": "sensor.b",
                "sensor_type": "door",
                "state": "off",
                "timestamp": datetime.now(),
            },
            {
                "sensor_id": "sensor.c",
                "sensor_type": "motion",
                "state": "on",
                "timestamp": datetime.now(),
            },
            # Invalid events
            {
                "sensor_id": "sensor.d",
                "sensor_type": "invalid_type",
                "state": "on",
                "timestamp": datetime.now(),
            },
            {
                "sensor_id": "sensor.e",
                "sensor_type": "motion",
                "state": "invalid_state",
                "timestamp": datetime.now(),
            },
        ]

        metrics = quality_monitor.calculate_quality_metrics(mixed_events, set())

        # Should reflect the 3/5 validity ratio in accuracy components
        assert 0.4 <= metrics.accuracy_score <= 0.8  # Approximately 60% accuracy

    def test_timeliness_score_calculation(self, quality_monitor):
        """Test timeliness score calculation with different data ages."""
        now = datetime.now(timezone.utc)

        # Mix of fresh and old data
        timed_events = [
            # Fresh data (should score highly)
            {"sensor_id": "sensor.fresh1", "timestamp": now - timedelta(minutes=5)},
            {"sensor_id": "sensor.fresh2", "timestamp": now - timedelta(minutes=30)},
            # Old data (should score poorly)
            {"sensor_id": "sensor.old1", "timestamp": now - timedelta(hours=25)},
            {"sensor_id": "sensor.old2", "timestamp": now - timedelta(days=7)},
            # Future data (should score poorly)
            {"sensor_id": "sensor.future", "timestamp": now + timedelta(hours=1)},
        ]

        metrics = quality_monitor.calculate_quality_metrics(timed_events, set())

        # Timeliness should be moderate (mix of fresh and old data)
        assert 0.2 <= metrics.timeliness_score <= 0.8

    def test_get_quality_trends(self, quality_monitor):
        """Test quality trends retrieval."""
        # Add some test history
        base_time = datetime.now()
        for i in range(5):
            quality_monitor.quality_history.append(
                {
                    "timestamp": base_time - timedelta(hours=i),
                    "completeness": 0.8 + i * 0.04,
                    "consistency": 0.7 + i * 0.05,
                    "accuracy": 0.9 - i * 0.02,
                    "timeliness": 0.85,
                    "anomalies": i,
                }
            )

        trends = quality_monitor.get_quality_trends(hours=24)

        assert isinstance(trends, dict)
        assert "completeness" in trends
        assert "consistency" in trends
        assert "accuracy" in trends
        assert "timeliness" in trends
        assert "anomalies" in trends
        assert "timestamps" in trends

        # Should have 5 entries
        assert len(trends["completeness"]) == 5
        assert len(trends["timestamps"]) == 5

    def test_detect_quality_alerts_all_thresholds(self, quality_monitor):
        """Test quality alert detection for all threshold types."""
        # Create metrics that violate all thresholds
        poor_metrics = DataQualityMetrics(
            completeness_score=0.5,  # Below 0.8 threshold
            consistency_score=0.4,  # Below 0.7 threshold
            accuracy_score=0.6,  # Below 0.75 threshold
            timeliness_score=0.7,  # Below 0.9 threshold
            anomaly_count=5,
        )

        alerts = quality_monitor.detect_quality_alerts(poor_metrics)

        assert len(alerts) == 4  # All four quality aspects should alert

        alert_types = {alert.anomaly_type for alert in alerts}
        assert "data_completeness" in alert_types
        assert "data_consistency" in alert_types
        assert "data_accuracy" in alert_types
        assert "data_timeliness" in alert_types

    def test_detect_quality_alerts_good_metrics(self, quality_monitor):
        """Test quality alert detection with good metrics."""
        good_metrics = DataQualityMetrics(
            completeness_score=0.95,
            consistency_score=0.85,
            accuracy_score=0.90,
            timeliness_score=0.95,
            anomaly_count=0,
        )

        alerts = quality_monitor.detect_quality_alerts(good_metrics)

        # Should have no alerts for good quality data
        assert len(alerts) == 0

    def test_quality_history_deque_limit(self, quality_monitor):
        """Test that quality history respects deque maxlen limit."""
        # Fill beyond maxlen
        for i in range(1500):  # More than maxlen=1000
            quality_monitor.quality_history.append(
                {
                    "timestamp": datetime.now() - timedelta(minutes=i),
                    "completeness": 0.8,
                    "consistency": 0.7,
                    "accuracy": 0.9,
                    "timeliness": 0.85,
                    "anomalies": 0,
                }
            )

        # Should not exceed maxlen
        assert len(quality_monitor.quality_history) == 1000

    def test_sensor_quality_tracking(self, quality_monitor):
        """Test individual sensor quality tracking."""
        # This tests the sensor_quality defaultdict structure
        # (Currently not fully implemented but structure should be correct)
        assert isinstance(quality_monitor.sensor_quality, defaultdict)

        # Adding sensor data should work
        quality_monitor.sensor_quality["test_sensor"].append(0.85)
        assert len(quality_monitor.sensor_quality["test_sensor"]) == 1


class TestPatternAnomalyAndDataQualityMetrics:
    """Tests for PatternAnomaly and DataQualityMetrics dataclasses."""

    def test_pattern_anomaly_creation(self):
        """Test PatternAnomaly creation and properties."""
        anomaly = PatternAnomaly(
            anomaly_id="TEST001",
            anomaly_type="test_anomaly",
            description="Test anomaly description",
            severity=ErrorSeverity.HIGH,
            confidence=0.85,
            detected_at=datetime.now(),
            affected_sensors=["sensor.test1", "sensor.test2"],
            statistical_measures={"mean": 50.0, "std": 10.0},
            context={"additional_info": "test_context"},
        )

        assert anomaly.anomaly_id == "TEST001"
        assert anomaly.anomaly_type == "test_anomaly"
        assert anomaly.severity == ErrorSeverity.HIGH
        assert anomaly.confidence == 0.85
        assert len(anomaly.affected_sensors) == 2
        assert "mean" in anomaly.statistical_measures
        assert "additional_info" in anomaly.context

    def test_data_quality_metrics_creation(self):
        """Test DataQualityMetrics creation and properties."""
        metrics = DataQualityMetrics(
            completeness_score=0.95,
            consistency_score=0.80,
            accuracy_score=0.90,
            timeliness_score=0.85,
            anomaly_count=2,
            corruption_indicators=["COR001", "COR002"],
            quality_trends={"completeness": [0.9, 0.95], "accuracy": [0.85, 0.90]},
        )

        assert metrics.completeness_score == 0.95
        assert metrics.consistency_score == 0.80
        assert metrics.accuracy_score == 0.90
        assert metrics.timeliness_score == 0.85
        assert metrics.anomaly_count == 2
        assert len(metrics.corruption_indicators) == 2
        assert len(metrics.quality_trends["completeness"]) == 2


class TestEdgeCasesAndErrorHandling:
    """Tests for edge cases and error handling scenarios."""

    def test_analyzer_with_malformed_events(self, pattern_analyzer):
        """Test pattern analyzer with malformed events."""
        malformed_events = [
            {"sensor_id": "sensor.test"},  # Missing timestamp and state
            {"timestamp": datetime.now()},  # Missing sensor_id and state
            {"state": "on"},  # Missing sensor_id and timestamp
            None,  # None event
            {"sensor_id": None, "timestamp": None, "state": None},  # All None values
        ]

        # Should handle malformed events gracefully
        analysis = pattern_analyzer.analyze_sensor_behavior(
            "test_sensor", malformed_events
        )

        # May return error or handle gracefully
        assert isinstance(analysis, dict)

    def test_corruption_detector_with_extreme_data(self, corruption_detector):
        """Test corruption detector with extreme data values."""
        extreme_events = [
            {
                "sensor_id": "x" * 10000,  # Very long ID
                "room_id": "room_1",
                "state": "y" * 10000,  # Very long state
                "timestamp": datetime.min,  # Extreme timestamp
                "attributes": {"key": "z" * 10000},  # Very long attribute
            }
        ]

        # Should handle extreme values without crashing
        errors = corruption_detector.detect_data_corruption(extreme_events)
        assert isinstance(errors, list)

    def test_quality_monitor_with_inconsistent_data_types(self, quality_monitor):
        """Test quality monitor with inconsistent data types."""
        inconsistent_events = [
            {
                "sensor_id": 123,
                "state": True,
                "timestamp": "not_a_timestamp",
            },  # Wrong types
            {"sensor_id": ["list", "id"], "state": {"dict": "state"}},  # Complex types
            {"sensor_id": None, "state": None, "timestamp": None},  # All None
        ]

        # Should handle inconsistent types gracefully
        metrics = quality_monitor.calculate_quality_metrics(inconsistent_events, set())
        assert isinstance(metrics, DataQualityMetrics)

    def test_concurrent_pattern_analysis(self, pattern_analyzer):
        """Test concurrent pattern analysis operations."""
        import threading
        import time

        results = []
        errors = []

        def analyze_patterns(sensor_id, events):
            try:
                result = pattern_analyzer.analyze_sensor_behavior(sensor_id, events)
                results.append(result)
            except Exception as e:
                errors.append(e)

        # Create test events for different sensors
        base_time = datetime.now()
        threads = []

        for i in range(5):
            events = [
                {
                    "sensor_id": f"sensor_{i}",
                    "timestamp": base_time + timedelta(seconds=j),
                    "state": "on" if j % 2 == 0 else "off",
                }
                for j in range(10)
            ]

            thread = threading.Thread(
                target=analyze_patterns, args=(f"sensor_{i}", events)
            )
            threads.append(thread)
            thread.start()

        # Wait for all threads
        for thread in threads:
            thread.join()

        # Should complete without errors
        assert len(errors) == 0
        assert len(results) == 5

    def test_memory_usage_with_large_datasets(self, pattern_analyzer, quality_monitor):
        """Test memory usage with large datasets."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Create large dataset
        large_events = []
        base_time = datetime.now()

        for i in range(10000):
            large_events.append(
                {
                    "sensor_id": f"sensor_{i % 100}",
                    "room_id": f"room_{i % 20}",
                    "sensor_type": "motion",
                    "state": "on" if i % 2 == 0 else "off",
                    "timestamp": base_time + timedelta(seconds=i),
                    "attributes": {"index": i, "data": f"data_{i}"},
                }
            )

        # Process with pattern analyzer
        for sensor_num in range(10):
            sensor_events = [
                e for e in large_events if f"sensor_{sensor_num}" in e["sensor_id"]
            ][:100]
            pattern_analyzer.analyze_sensor_behavior(
                f"sensor_{sensor_num}", sensor_events
            )

        # Process with quality monitor
        quality_monitor.calculate_quality_metrics(large_events[:1000], set())

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 200MB for large dataset)
        assert memory_increase < 200 * 1024 * 1024

    def test_performance_with_high_frequency_data(self, pattern_analyzer):
        """Test performance with high-frequency sensor data."""
        import time

        # Create high-frequency data (10,000 events over 1 hour)
        base_time = datetime.now()
        high_freq_events = []

        for i in range(10000):
            high_freq_events.append(
                {
                    "sensor_id": "high_freq_sensor",
                    "timestamp": base_time
                    + timedelta(seconds=i * 0.36),  # Every 0.36 seconds
                    "state": "on" if i % 10 < 3 else "off",
                }
            )

        start_time = time.time()
        analysis = pattern_analyzer.analyze_sensor_behavior(
            "high_freq_sensor", high_freq_events
        )
        end_time = time.time()

        # Should complete within reasonable time (< 5 seconds)
        processing_time = end_time - start_time
        assert processing_time < 5.0

        # Should produce valid analysis
        assert isinstance(analysis, dict)
        assert analysis["event_count"] == 10000
        assert "trigger_frequency" in analysis

    def test_robustness_with_corrupted_timestamp_formats(self, corruption_detector):
        """Test robustness with various corrupted timestamp formats."""
        timestamp_variants = [
            "2024-01-15T10:30:00",  # Missing timezone
            "2024/01/15 10:30:00",  # Wrong format
            "15-01-2024 10:30:00",  # Wrong order
            "2024-13-45T25:99:99Z",  # Invalid values
            "2024-01-15T10:30:00+99:99",  # Invalid timezone
            "Jan 15, 2024 10:30 AM",  # Text format
            "1642248600",  # Unix timestamp as string
            "",  # Empty string
            "null",  # String null
        ]

        corrupted_events = []
        for i, timestamp in enumerate(timestamp_variants):
            corrupted_events.append(
                {
                    "sensor_id": f"sensor_{i}",
                    "room_id": "test_room",
                    "state": "on",
                    "timestamp": timestamp,
                }
            )

        errors = corruption_detector.detect_data_corruption(corrupted_events)

        # Should detect most timestamp corruption without crashing
        timestamp_errors = [e for e in errors if "timestamp" in e.field.lower()]
        assert len(timestamp_errors) > 0

        # Should handle all variants without exceptions
        assert len(corrupted_events) == len(timestamp_variants)
