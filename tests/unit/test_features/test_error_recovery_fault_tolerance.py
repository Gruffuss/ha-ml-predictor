"""
Comprehensive error recovery and fault tolerance tests for feature engineering.

This test suite validates system resilience under:
- Component failures and recovery
- Resource exhaustion scenarios  
- Network and database failures
- Graceful degradation patterns
- Circuit breaker implementations
"""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from datetime import datetime, timedelta
import random
import threading
import time
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest

from src.core.config import RoomConfig
from src.core.exceptions import FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.store import FeatureStore
from src.features.temporal import TemporalFeatureExtractor


class TestComponentFailureRecovery:
    """Test recovery from individual component failures."""

    def test_temporal_extractor_partial_method_failures(self):
        """Test temporal extractor recovery from partial method failures."""

        class PartiallyFailingTemporalExtractor(TemporalFeatureExtractor):
            def __init__(self):
                super().__init__()
                self.failure_count = 0

            def _extract_cyclical_features(self, target_time):
                self.failure_count += 1
                if self.failure_count <= 2:  # Fail first 2 calls
                    raise RuntimeError("Cyclical feature extraction failed")
                return super()._extract_cyclical_features(target_time)

            def _extract_historical_patterns(self, events, target_time):
                # Always fails
                raise ValueError("Historical pattern analysis unavailable")

            def _extract_transition_timing_features(self, events, target_time):
                # Intermittent failures
                if random.random() < 0.3:
                    raise ConnectionError("Timing analysis service unavailable")
                return {"transition_rate": 0.5, "transition_regularity": 0.8}

        extractor = PartiallyFailingTemporalExtractor()
        events = [Mock(spec=SensorEvent) for _ in range(10)]
        for i, event in enumerate(events):
            event.timestamp = datetime(2024, 1, 15, 12, 0, 0) - timedelta(minutes=i)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"

        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # First two calls should recover from cyclical failures
        for attempt in range(3):
            features = extractor.extract_features(events, target_time)

            # Should extract some features despite partial failures
            assert (
                len(features) > 5
            ), f"Attempt {attempt}: Should extract partial features"
            assert all(
                not np.isnan(v) for v in features.values()
            ), "No NaN values allowed"

            # Working methods should contribute features
            assert (
                "time_since_last_change" in features
            ), "Working methods should contribute"

    def test_sequential_extractor_classifier_failure_fallback(self):
        """Test sequential extractor fallback when movement classifier fails."""

        class FlakySequentialExtractor(SequentialFeatureExtractor):
            def __init__(self):
                super().__init__()
                self.classifier_available = False

            def _classify_movement_pattern(self, events, room_configs):
                if not self.classifier_available:
                    raise RuntimeError("Movement classifier service down")
                return {"movement_type": "human", "confidence": 0.8}

            def _extract_room_transitions(self, events, room_configs):
                # Simulate occasional failures
                if random.random() < 0.2:
                    raise TimeoutError("Room transition analysis timeout")
                return {"room_transition_count": 3, "unique_rooms": 2}

            def _fallback_movement_analysis(self, events):
                """Fallback movement analysis without classifier."""
                return {"fallback_movement_detected": 1, "fallback_activity_level": 0.6}

        extractor = FlakySequentialExtractor()
        room_configs = {"room1": Mock(spec=RoomConfig), "room2": Mock(spec=RoomConfig)}

        events = []
        for i in range(20):
            event = Mock(spec=SensorEvent)
            event.room_id = f"room{(i % 2) + 1}"
            event.timestamp = datetime(2024, 1, 15, 12, 0, 0) + timedelta(minutes=i)
            event.sensor_type = "motion"
            event.state = "on" if i % 3 == 0 else "off"
            events.append(event)

        # Multiple attempts should handle intermittent failures
        successful_extractions = 0
        for attempt in range(10):
            try:
                features = extractor.extract_features(events, room_configs)
                successful_extractions += 1

                # Should have some features despite classifier failure
                assert len(features) > 3, "Should extract features without classifier"
                assert all(not np.isnan(v) for v in features.values()), "No NaN values"

            except (RuntimeError, TimeoutError):
                # Some failures are expected
                continue

        assert successful_extractions > 5, "Should have majority successful extractions"

    def test_contextual_extractor_environmental_sensor_degradation(self):
        """Test contextual extractor handling environmental sensor degradation."""

        class DegradingContextualExtractor(ContextualFeatureExtractor):
            def __init__(self):
                super().__init__()
                self.sensor_health = {
                    "temperature": 1.0,
                    "humidity": 1.0,
                    "light": 1.0,
                    "pressure": 1.0,
                }

            def _simulate_sensor_degradation(self):
                """Simulate gradual sensor degradation."""
                for sensor in self.sensor_health:
                    self.sensor_health[sensor] *= 0.95  # 5% degradation per call

            def _extract_environmental_features(self, events):
                self._simulate_sensor_degradation()

                features = {}
                for sensor_type, health in self.sensor_health.items():
                    if health < 0.3:  # Sensor too degraded
                        raise RuntimeError(f"{sensor_type} sensor failed")
                    elif health < 0.7:  # Partial degradation
                        # Return degraded readings
                        features[f"{sensor_type}_avg"] = health * 20.0  # Scaled reading
                        features[f"{sensor_type}_reliability"] = health
                    else:
                        # Normal operation
                        features[f"{sensor_type}_avg"] = 22.0
                        features[f"{sensor_type}_reliability"] = 1.0

                return features

        extractor = DegradingContextualExtractor()

        events = [Mock(spec=SensorEvent) for _ in range(10)]
        room_states = [Mock(spec=RoomState)]

        # Simulate progressive degradation
        degradation_results = []
        for cycle in range(15):  # Multiple cycles to trigger degradation
            try:
                features = extractor.extract_features(events, room_states)
                degradation_results.append(
                    {
                        "cycle": cycle,
                        "features_extracted": len(features),
                        "sensor_health": extractor.sensor_health.copy(),
                    }
                )
            except RuntimeError as e:
                degradation_results.append(
                    {
                        "cycle": cycle,
                        "error": str(e),
                        "sensor_health": extractor.sensor_health.copy(),
                    }
                )

        # Should show progressive degradation
        assert len(degradation_results) > 10, "Should complete multiple cycles"

        # Early cycles should succeed
        early_successes = [
            r for r in degradation_results[:5] if "features_extracted" in r
        ]
        assert len(early_successes) > 3, "Early cycles should mostly succeed"

        # Later cycles should show failures
        late_failures = [r for r in degradation_results[10:] if "error" in r]
        assert len(late_failures) > 0, "Later cycles should show sensor failures"

    def test_feature_engineering_engine_extractor_orchestration_failures(self):
        """Test engine recovery when multiple extractors fail in different patterns."""

        class ChaoticFeatureEngine(FeatureEngineeringEngine):
            def __init__(self):
                super().__init__()
                self.attempt_count = 0

            def _chaotic_temporal_extractor(self, *args, **kwargs):
                """Temporal extractor with chaos patterns."""
                self.attempt_count += 1

                if self.attempt_count % 5 == 0:  # Fail every 5th call
                    raise MemoryError("Temporal extractor out of memory")
                elif self.attempt_count % 3 == 0:  # Timeout every 3rd call
                    raise TimeoutError("Temporal analysis timeout")
                else:
                    return {"temporal_feature": float(self.attempt_count)}

            def _chaotic_sequential_extractor(self, *args, **kwargs):
                """Sequential extractor with different failure pattern."""
                if random.random() < 0.4:  # 40% failure rate
                    raise ConnectionError("Sequential analysis service unavailable")
                return {"sequential_feature": 2.0}

            def _chaotic_contextual_extractor(self, *args, **kwargs):
                """Contextual extractor with burst failures."""
                burst_start = 10
                burst_end = 15
                if burst_start <= self.attempt_count <= burst_end:
                    raise RuntimeError("Contextual service maintenance window")
                return {"contextual_feature": 3.0}

        engine = ChaoticFeatureEngine()

        # Replace extractors with chaotic versions
        engine.temporal_extractor.extract_features = engine._chaotic_temporal_extractor
        engine.sequential_extractor.extract_features = (
            engine._chaotic_sequential_extractor
        )
        engine.contextual_extractor.extract_features = (
            engine._chaotic_contextual_extractor
        )

        room_id = "chaos_room"
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        events = [Mock(spec=SensorEvent) for _ in range(5)]
        room_configs = {"chaos_room": Mock(spec=RoomConfig)}

        # Run many extractions to hit various failure patterns
        results = []
        for i in range(20):
            try:
                features = engine.extract_features(
                    room_id, target_time, events, room_configs
                )
                results.append(
                    {"attempt": i, "success": True, "features": len(features)}
                )
            except Exception as e:
                results.append(
                    {"attempt": i, "success": False, "error": type(e).__name__}
                )

        # Should have mix of successes and failures
        successes = [r for r in results if r["success"]]
        failures = [r for r in results if not r["success"]]

        assert len(successes) > 5, "Should have some successful extractions"
        assert len(failures) > 0, "Should have expected failures"

        # Successful extractions should have reasonable feature counts
        for success in successes:
            assert (
                success["features"] > 0
            ), "Successful extractions should yield features"


class TestResourceExhaustionRecovery:
    """Test recovery from resource exhaustion scenarios."""

    def test_memory_exhaustion_graceful_degradation(self):
        """Test graceful degradation under memory pressure."""

        class MemoryAwareExtractor(TemporalFeatureExtractor):
            def __init__(self):
                super().__init__()
                self.memory_limit_mb = 100
                self.degradation_mode = False

            def _check_memory_pressure(self):
                """Check current memory usage."""
                import psutil

                process = psutil.Process()
                memory_mb = process.memory_info().rss / 1024 / 1024
                return memory_mb > self.memory_limit_mb

            def extract_features(self, events, target_time, **kwargs):
                """Extract with memory awareness."""
                if self._check_memory_pressure():
                    self.degradation_mode = True
                    # Reduce feature set under memory pressure
                    return self._extract_essential_features_only(events, target_time)
                else:
                    self.degradation_mode = False
                    return super().extract_features(events, target_time, **kwargs)

            def _extract_essential_features_only(self, events, target_time):
                """Extract minimal essential features."""
                if not events:
                    return {"memory_degraded": 1.0}

                # Only most critical features
                return {
                    "time_since_last_change": (
                        target_time - events[-1].timestamp
                    ).total_seconds()
                    / 3600,
                    "event_count": len(events),
                    "memory_degraded": 1.0,
                }

        extractor = MemoryAwareExtractor()

        # Create large event dataset to stress memory
        large_events = []
        for i in range(1000):
            event = Mock(spec=SensorEvent)
            event.timestamp = datetime(2024, 1, 15, 12, 0, 0) - timedelta(minutes=i)
            event.state = "on" if i % 2 == 0 else "off"
            event.sensor_type = "motion"
            # Add large attributes to increase memory usage
            event.attributes = {
                f"large_attr_{j}": f"data_{i}_{j}" * 100 for j in range(50)
            }
            large_events.append(event)

        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Extract features - should handle memory pressure
        features = extractor.extract_features(large_events, target_time)

        assert len(features) > 0, "Should extract some features under memory pressure"
        assert all(
            not np.isnan(v) for v in features.values()
        ), "Features should be valid"

        if extractor.degradation_mode:
            assert "memory_degraded" in features, "Should indicate degradation mode"
            assert features["memory_degraded"] == 1.0, "Should flag memory degradation"

    def test_cpu_exhaustion_timeout_handling(self):
        """Test handling of CPU exhaustion and timeouts."""

        class CPUIntensiveExtractor(SequentialFeatureExtractor):
            def __init__(self, timeout_seconds=2):
                super().__init__()
                self.timeout_seconds = timeout_seconds

            def _cpu_intensive_analysis(self, events):
                """Simulate CPU-intensive analysis."""
                import math

                # Simulate heavy computation
                result = 0
                for i in range(1000000):  # Heavy loop
                    result += math.sin(i) * math.cos(i)
                return {"cpu_intensive_feature": result}

            def extract_features(self, events, room_configs, target_time=None):
                """Extract with timeout protection."""
                from concurrent.futures import ThreadPoolExecutor, TimeoutError

                def safe_extraction():
                    return super().extract_features(events, room_configs, target_time)

                try:
                    # Use timeout to prevent CPU exhaustion
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(safe_extraction)
                        return future.result(timeout=self.timeout_seconds)

                except TimeoutError:
                    # Return minimal features on timeout
                    return {
                        "extraction_timeout": 1.0,
                        "event_count": len(events),
                        "room_count": len(room_configs),
                    }

        extractor = CPUIntensiveExtractor(timeout_seconds=1)  # Short timeout

        # Create complex event pattern that might trigger heavy computation
        events = []
        room_configs = {}

        for i in range(100):
            event = Mock(spec=SensorEvent)
            event.room_id = f"room_{i % 10}"
            event.timestamp = datetime(2024, 1, 15, 12, 0, 0) + timedelta(seconds=i)
            event.sensor_type = "motion"
            event.state = "on" if i % 7 < 4 else "off"
            events.append(event)

            room_configs[event.room_id] = Mock(spec=RoomConfig)

        # Should complete within timeout or return fallback features
        start_time = time.time()
        features = extractor.extract_features(events, room_configs)
        end_time = time.time()

        assert end_time - start_time < 3.0, "Should complete within reasonable time"
        assert len(features) > 0, "Should return some features"

        if "extraction_timeout" in features:
            assert (
                features["extraction_timeout"] == 1.0
            ), "Should indicate timeout occurred"

    def test_disk_space_exhaustion_fallback(self):
        """Test fallback when disk space is exhausted."""

        class DiskAwareFeatureStore(FeatureStore):
            def __init__(self):
                super().__init__(db_manager=Mock())
                self.disk_space_mb = 1000  # Simulate available disk space
                self.cache_size_limit = 100

            def _check_disk_space(self):
                """Check available disk space."""
                return self.disk_space_mb > 50  # Need at least 50MB

            def _simulate_disk_usage(self, size_mb):
                """Simulate disk space consumption."""
                self.disk_space_mb -= size_mb

            def get_features(self, room_id, target_time, events, **kwargs):
                """Get features with disk space awareness."""
                if not self._check_disk_space():
                    # Use in-memory fallback only
                    return self._extract_features_memory_only(
                        room_id, target_time, events
                    )
                else:
                    # Normal operation with disk caching
                    self._simulate_disk_usage(10)  # Simulate cache write
                    return super().get_features(room_id, target_time, events, **kwargs)

            def _extract_features_memory_only(self, room_id, target_time, events):
                """Extract features without disk operations."""
                return {
                    "memory_only_feature": len(events),
                    "disk_space_exhausted": 1.0,
                    "fallback_mode": 1.0,
                }

        store = DiskAwareFeatureStore()

        # Simulate multiple requests that exhaust disk space
        room_id = "disk_test_room"
        target_time = datetime(2024, 1, 15, 12, 0, 0)
        events = [Mock(spec=SensorEvent) for _ in range(10)]

        results = []
        for i in range(150):  # Enough to exhaust disk space
            features = store.get_features(room_id, target_time, events)
            results.append(
                {
                    "request": i,
                    "disk_space": store.disk_space_mb,
                    "fallback_mode": features.get("fallback_mode", 0.0),
                }
            )

        # Should transition to fallback mode when disk space exhausted
        normal_mode = [r for r in results if r["fallback_mode"] == 0.0]
        fallback_mode = [r for r in results if r["fallback_mode"] == 1.0]

        assert len(normal_mode) > 0, "Should operate normally initially"
        assert len(fallback_mode) > 0, "Should enter fallback mode when disk exhausted"


class TestNetworkAndDatabaseFailureRecovery:
    """Test recovery from network and database failures."""

    def test_database_connection_failure_with_exponential_backoff(self):
        """Test database reconnection with exponential backoff."""

        class ResilientFeatureStore(FeatureStore):
            def __init__(self):
                super().__init__(db_manager=Mock())
                self.connection_attempts = 0
                self.max_attempts = 5
                self.base_delay = 0.1

            def _attempt_database_connection(self):
                """Simulate database connection attempts."""
                self.connection_attempts += 1

                # Fail first few attempts, then succeed
                if self.connection_attempts < 3:
                    raise ConnectionError(
                        f"Database connection failed (attempt {self.connection_attempts})"
                    )
                else:
                    return True

            def _exponential_backoff_delay(self, attempt):
                """Calculate exponential backoff delay."""
                return self.base_delay * (2**attempt)

            def get_data_for_features(self, room_id, target_time, **kwargs):
                """Get data with connection retry logic."""
                for attempt in range(self.max_attempts):
                    try:
                        self._attempt_database_connection()
                        # Return mock data on successful connection
                        return [Mock(spec=SensorEvent) for _ in range(5)]

                    except ConnectionError as e:
                        if attempt < self.max_attempts - 1:
                            delay = self._exponential_backoff_delay(attempt)
                            time.sleep(delay)
                            continue
                        else:
                            # Final fallback - return empty data
                            return []

                return []

        store = ResilientFeatureStore()

        room_id = "resilient_room"
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        start_time = time.time()
        data = store.get_data_for_features(room_id, target_time)
        end_time = time.time()

        # Should eventually succeed with retries
        assert len(data) > 0, "Should get data after retries"
        assert store.connection_attempts >= 3, "Should attempt multiple connections"

        # Should use exponential backoff (total time should reflect delays)
        expected_min_time = store.base_delay * (1 + 2)  # First two failures
        assert (
            end_time - start_time >= expected_min_time
        ), "Should use exponential backoff"

    def test_network_partition_graceful_degradation(self):
        """Test graceful degradation during network partitions."""

        class NetworkAwareExtractor(ContextualFeatureExtractor):
            def __init__(self):
                super().__init__()
                self.network_available = True
                self.local_cache = {}

            def _check_network_connectivity(self):
                """Simulate network connectivity check."""
                return self.network_available

            def _fetch_external_context(self, room_id):
                """Simulate fetching external context data."""
                if not self._check_network_connectivity():
                    raise ConnectionError(
                        "Network partition - external services unavailable"
                    )

                # Simulate external data
                return {
                    "weather_temperature": 22.0,
                    "external_occupancy_correlation": 0.7,
                    "regional_activity_level": 0.8,
                }

            def extract_features(self, events, room_states):
                """Extract with network resilience."""
                features = {}

                # Always extract local features
                local_features = self._extract_local_features(events, room_states)
                features.update(local_features)

                # Try to get external context
                try:
                    external_context = self._fetch_external_context("test_room")
                    features.update(external_context)
                    features["network_available"] = 1.0
                except ConnectionError:
                    # Use cached external data if available
                    if self.local_cache:
                        cached_context = self.local_cache.copy()
                        # Mark as stale
                        for key in cached_context:
                            cached_context[f"{key}_cached"] = cached_context.pop(key)
                        features.update(cached_context)

                    features["network_available"] = 0.0
                    features["using_cached_data"] = 1.0 if self.local_cache else 0.0

                return features

            def _extract_local_features(self, events, room_states):
                """Extract features using only local data."""
                return {
                    "local_event_count": len(events),
                    "local_room_states": len(room_states),
                    "local_analysis": 1.0,
                }

        extractor = NetworkAwareExtractor()

        events = [Mock(spec=SensorEvent) for _ in range(5)]
        room_states = [Mock(spec=RoomState)]

        # Test with network available
        extractor.network_available = True
        features_online = extractor.extract_features(events, room_states)

        # Cache external data for offline use
        external_keys = [
            "weather_temperature",
            "external_occupancy_correlation",
            "regional_activity_level",
        ]
        extractor.local_cache = {
            k: features_online[k] for k in external_keys if k in features_online
        }

        # Test with network partition
        extractor.network_available = False
        features_offline = extractor.extract_features(events, room_states)

        # Should have local features in both cases
        assert "local_event_count" in features_online
        assert "local_event_count" in features_offline
        assert (
            features_online["local_event_count"]
            == features_offline["local_event_count"]
        )

        # Online should have fresh external data
        assert features_online.get("network_available") == 1.0

        # Offline should use cached data
        assert features_offline.get("network_available") == 0.0
        assert features_offline.get("using_cached_data") == 1.0

    def test_database_corruption_recovery(self):
        """Test recovery from database corruption scenarios."""

        class CorruptionResilientStore(FeatureStore):
            def __init__(self):
                super().__init__(db_manager=Mock())
                self.corruption_detected = False
                self.backup_data = {}

            def _detect_data_corruption(self, data):
                """Simulate corruption detection."""
                if not data:
                    return False

                # Check for corruption indicators
                for item in data:
                    if hasattr(item, "timestamp") and item.timestamp is None:
                        return True
                    if hasattr(item, "state") and item.state == "CORRUPTED":
                        return True

                return False

            def _create_backup_data(self, room_id):
                """Create synthetic backup data."""
                backup_events = []
                base_time = datetime(2024, 1, 15, 12, 0, 0)

                for i in range(5):
                    event = Mock(spec=SensorEvent)
                    event.room_id = room_id
                    event.timestamp = base_time - timedelta(minutes=i * 10)
                    event.state = "on" if i % 2 == 0 else "off"
                    event.sensor_type = "motion"
                    event.attributes = {"synthetic": True}
                    backup_events.append(event)

                return backup_events

            def get_data_for_features(self, room_id, target_time, **kwargs):
                """Get data with corruption recovery."""
                # Simulate getting potentially corrupted data
                mock_data = []

                # Simulate corruption scenarios
                corruption_type = kwargs.get("corruption_type", "none")

                if corruption_type == "null_timestamps":
                    for i in range(3):
                        event = Mock(spec=SensorEvent)
                        event.timestamp = None  # Corrupted timestamp
                        event.state = "on"
                        mock_data.append(event)

                elif corruption_type == "corrupted_states":
                    for i in range(3):
                        event = Mock(spec=SensorEvent)
                        event.timestamp = target_time - timedelta(minutes=i)
                        event.state = "CORRUPTED"  # Corrupted state
                        mock_data.append(event)
                else:
                    # Normal data
                    for i in range(5):
                        event = Mock(spec=SensorEvent)
                        event.timestamp = target_time - timedelta(minutes=i)
                        event.state = "on" if i % 2 == 0 else "off"
                        mock_data.append(event)

                # Check for corruption
                if self._detect_data_corruption(mock_data):
                    self.corruption_detected = True
                    # Use backup data
                    return self._create_backup_data(room_id)
                else:
                    return mock_data

        store = CorruptionResilientStore()
        room_id = "corruption_test_room"
        target_time = datetime(2024, 1, 15, 12, 0, 0)

        # Test with corrupted timestamps
        data_corrupted = store.get_data_for_features(
            room_id, target_time, corruption_type="null_timestamps"
        )

        assert store.corruption_detected, "Should detect corruption"
        assert len(data_corrupted) > 0, "Should provide backup data"
        assert all(
            event.timestamp is not None for event in data_corrupted
        ), "Backup data should be valid"

        # Reset for next test
        store.corruption_detected = False

        # Test with normal data
        data_normal = store.get_data_for_features(room_id, target_time)

        assert (
            not store.corruption_detected
        ), "Should not detect corruption in normal data"
        assert len(data_normal) > 0, "Should return normal data"


class TestCircuitBreakerImplementation:
    """Test circuit breaker patterns for fault tolerance."""

    def test_feature_extraction_circuit_breaker(self):
        """Test circuit breaker for feature extraction operations."""

        class CircuitBreakerExtractor:
            def __init__(self, failure_threshold=3, timeout_seconds=5):
                self.failure_threshold = failure_threshold
                self.timeout_seconds = timeout_seconds
                self.failure_count = 0
                self.last_failure_time = None
                self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

            def _should_attempt_operation(self):
                """Determine if operation should be attempted."""
                if self.state == "CLOSED":
                    return True
                elif self.state == "OPEN":
                    # Check if timeout has passed
                    if self.last_failure_time:
                        time_since_failure = time.time() - self.last_failure_time
                        if time_since_failure > self.timeout_seconds:
                            self.state = "HALF_OPEN"
                            return True
                    return False
                elif self.state == "HALF_OPEN":
                    return True

                return False

            def _record_success(self):
                """Record successful operation."""
                self.failure_count = 0
                self.state = "CLOSED"

            def _record_failure(self):
                """Record failed operation."""
                self.failure_count += 1
                self.last_failure_time = time.time()

                if self.failure_count >= self.failure_threshold:
                    self.state = "OPEN"
                elif self.state == "HALF_OPEN":
                    self.state = "OPEN"

            def extract_features(self, events, simulate_failure=False):
                """Extract features with circuit breaker protection."""
                if not self._should_attempt_operation():
                    return {"circuit_breaker_open": 1.0, "fallback_feature": 0.5}

                try:
                    if simulate_failure:
                        raise RuntimeError("Simulated extraction failure")

                    # Normal feature extraction
                    features = {
                        "normal_feature": len(events),
                        "extraction_success": 1.0,
                    }
                    self._record_success()
                    return features

                except Exception as e:
                    self._record_failure()
                    return {"circuit_breaker_failure": 1.0, "error_feature": 0.0}

        extractor = CircuitBreakerExtractor(failure_threshold=3, timeout_seconds=1)
        events = [Mock(spec=SensorEvent) for _ in range(5)]

        # Test normal operation (circuit closed)
        features = extractor.extract_features(events)
        assert extractor.state == "CLOSED"
        assert "normal_feature" in features

        # Trigger failures to open circuit
        for i in range(3):
            features = extractor.extract_features(events, simulate_failure=True)

        assert extractor.state == "OPEN"

        # Circuit should be open - should get fallback features
        features = extractor.extract_features(events)
        assert "circuit_breaker_open" in features
        assert features["circuit_breaker_open"] == 1.0

        # Wait for timeout and test half-open
        time.sleep(1.5)

        # Should attempt operation again (half-open)
        features = extractor.extract_features(events)
        assert extractor.state == "CLOSED"  # Should close on success
        assert "normal_feature" in features

    def test_adaptive_circuit_breaker_with_health_monitoring(self):
        """Test adaptive circuit breaker that adjusts based on system health."""

        class AdaptiveCircuitBreaker:
            def __init__(self):
                self.base_failure_threshold = 3
                self.failure_threshold = self.base_failure_threshold
                self.failure_count = 0
                self.success_count = 0
                self.state = "CLOSED"
                self.health_metrics = {"cpu_usage": 0.0, "memory_usage": 0.0}

            def _update_health_metrics(self):
                """Update system health metrics."""
                import psutil

                self.health_metrics["cpu_usage"] = psutil.cpu_percent()
                self.health_metrics["memory_usage"] = psutil.virtual_memory().percent

            def _adapt_threshold(self):
                """Adapt failure threshold based on system health."""
                self._update_health_metrics()

                # Lower threshold under high system load
                if (
                    self.health_metrics["cpu_usage"] > 80
                    or self.health_metrics["memory_usage"] > 80
                ):
                    self.failure_threshold = max(1, self.base_failure_threshold - 2)
                elif (
                    self.health_metrics["cpu_usage"] < 50
                    and self.health_metrics["memory_usage"] < 50
                ):
                    self.failure_threshold = self.base_failure_threshold + 2
                else:
                    self.failure_threshold = self.base_failure_threshold

            def process_request(self, operation_func, *args, **kwargs):
                """Process request with adaptive circuit breaking."""
                self._adapt_threshold()

                if self.state == "OPEN":
                    return {
                        "circuit_open": 1.0,
                        "health_cpu": self.health_metrics["cpu_usage"],
                    }

                try:
                    result = operation_func(*args, **kwargs)
                    self.success_count += 1
                    self.failure_count = max(
                        0, self.failure_count - 1
                    )  # Decay failures
                    return result

                except Exception as e:
                    self.failure_count += 1

                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"

                    return {
                        "operation_failed": 1.0,
                        "adaptive_threshold": self.failure_threshold,
                    }

        circuit_breaker = AdaptiveCircuitBreaker()

        def flaky_operation(fail_rate=0.3):
            """Operation that fails randomly."""
            if random.random() < fail_rate:
                raise RuntimeError("Random failure")
            return {"operation_success": 1.0}

        # Test adaptive behavior
        results = []
        for i in range(20):
            result = circuit_breaker.process_request(flaky_operation, fail_rate=0.4)
            results.append(
                {
                    "attempt": i,
                    "state": circuit_breaker.state,
                    "threshold": circuit_breaker.failure_threshold,
                    "result": result,
                }
            )

        # Should adapt to failures and system health
        thresholds_used = set(r["threshold"] for r in results)
        assert len(thresholds_used) > 1, "Should adapt failure threshold"

        # Should have opened circuit at some point
        states_seen = set(r["state"] for r in results)
        assert (
            "OPEN" in states_seen or circuit_breaker.failure_count > 0
        ), "Should react to failures"
