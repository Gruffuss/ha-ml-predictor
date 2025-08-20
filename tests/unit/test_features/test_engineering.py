"""
Unit tests for feature engineering engine orchestration.

This module tests the FeatureEngineeringEngine for parallel processing,
feature integration, error handling, and performance management.
"""

import asyncio
from datetime import datetime, timedelta
import time
from typing import List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.core.config import RoomConfig, SystemConfig
from src.core.exceptions import ConfigurationError, FeatureExtractionError
from src.data.storage.models import RoomState, SensorEvent
from src.features.contextual import ContextualFeatureExtractor
from src.features.engineering import FeatureEngineeringEngine
from src.features.sequential import SequentialFeatureExtractor
from src.features.temporal import TemporalFeatureExtractor


class TestFeatureEngineeringEngine:
    """Test suite for FeatureEngineeringEngine."""

    @pytest.fixture
    def mock_config(self):
        """Create mock system configuration with properly configured RoomConfig objects."""
        config = Mock(spec=SystemConfig)

        # Create properly structured room configs with all required attributes
        living_room_config = Mock(spec=RoomConfig)
        living_room_config.room_id = "living_room"
        living_room_config.name = "Living Room"
        living_room_config.sensors = {
            "motion": "sensor.living_room_motion",
            "temperature": "sensor.living_room_temperature",
            "humidity": "sensor.living_room_humidity",
            "door": "sensor.living_room_door",
        }
        living_room_config.get_sensors_by_type = Mock(
            return_value={
                "temperature": "sensor.living_room_temperature",
                "door": "sensor.living_room_door",
            }
        )
        living_room_config.get_all_entity_ids = Mock(
            return_value=[
                "sensor.living_room_motion",
                "sensor.living_room_temperature",
                "sensor.living_room_humidity",
                "sensor.living_room_door",
            ]
        )

        kitchen_config = Mock(spec=RoomConfig)
        kitchen_config.room_id = "kitchen"
        kitchen_config.name = "Kitchen"
        kitchen_config.sensors = {
            "motion": "sensor.kitchen_motion",
            "temperature": "sensor.kitchen_temperature",
            "door": "sensor.kitchen_door",
        }
        kitchen_config.get_sensors_by_type = Mock(
            return_value={
                "temperature": "sensor.kitchen_temperature",
                "door": "sensor.kitchen_door",
            }
        )
        kitchen_config.get_all_entity_ids = Mock(
            return_value=[
                "sensor.kitchen_motion",
                "sensor.kitchen_temperature",
                "sensor.kitchen_door",
            ]
        )

        config.rooms = {
            "living_room": living_room_config,
            "kitchen": kitchen_config,
        }
        return config

    @pytest.fixture
    def engine(self, mock_config):
        """Create feature engineering engine instance."""
        return FeatureEngineeringEngine(
            config=mock_config, enable_parallel=True, max_workers=3
        )

    @pytest.fixture
    def engine_sequential(self, mock_config):
        """Create engine with sequential processing."""
        return FeatureEngineeringEngine(config=mock_config, enable_parallel=False)

    @pytest.fixture
    def sample_events(self) -> List[SensorEvent]:
        """Create sample sensor events with proper state values."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        events = []

        for i in range(5):
            event = Mock(spec=SensorEvent)
            event.timestamp = base_time + timedelta(minutes=i * 5)
            event.room_id = "living_room"
            event.state = "on" if i % 2 == 0 else "off"  # Fixed: was "of"
            event.sensor_type = "motion"
            event.sensor_id = f"sensor.motion_{i}"
            events.append(event)

        return events

    @pytest.fixture
    def sample_room_states(self) -> List[RoomState]:
        """Create sample room states."""
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        states = []

        for i in range(3):
            state = Mock(spec=RoomState)
            state.timestamp = base_time + timedelta(minutes=i * 10)
            state.room_id = "living_room"
            state.is_occupied = i % 2 == 0
            state.occupancy_confidence = 0.8
            states.append(state)

        return states

    @pytest.fixture
    def target_time(self) -> datetime:
        """Standard target time for feature extraction."""
        return datetime(2024, 1, 15, 15, 0, 0)

    @pytest.mark.asyncio
    async def test_extract_features_parallel(
        self, engine, sample_events, sample_room_states, target_time
    ):
        """Test parallel feature extraction."""
        with (
            patch.object(
                engine.temporal_extractor, "extract_features"
            ) as mock_temporal,
            patch.object(
                engine.sequential_extractor, "extract_features"
            ) as mock_sequential,
            patch.object(
                engine.contextual_extractor, "extract_features"
            ) as mock_contextual,
        ):

            # Mock extractor responses
            mock_temporal.return_value = {
                "temporal_feature_1": 1.0,
                "temporal_feature_2": 2.0,
            }
            mock_sequential.return_value = {"sequential_feature_1": 3.0}
            mock_contextual.return_value = {"contextual_feature_1": 4.0}

            features = await engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=sample_events,
                room_states=sample_room_states,
                lookback_hours=24,
                feature_types=["temporal", "sequential", "contextual"],
            )

            # Verify all extractors were called
            mock_temporal.assert_called_once()
            mock_sequential.assert_called_once()
            mock_contextual.assert_called_once()

            # Verify features are prefixed and combined
            assert "temporal_temporal_feature_1" in features
            assert "temporal_temporal_feature_2" in features
            assert "sequential_sequential_feature_1" in features
            assert "contextual_contextual_feature_1" in features

            # Verify metadata features are added
            assert "meta_event_count" in features
            assert "meta_room_state_count" in features
            assert "meta_extraction_hour" in features

    @pytest.mark.asyncio
    async def test_extract_features_sequential(
        self, engine_sequential, sample_events, sample_room_states, target_time
    ):
        """Test sequential feature extraction."""
        with (
            patch.object(
                engine_sequential.temporal_extractor, "extract_features"
            ) as mock_temporal,
            patch.object(
                engine_sequential.sequential_extractor, "extract_features"
            ) as mock_sequential,
            patch.object(
                engine_sequential.contextual_extractor, "extract_features"
            ) as mock_contextual,
        ):

            # Mock extractor responses
            mock_temporal.return_value = {"temporal_feature_1": 1.0}
            mock_sequential.return_value = {"sequential_feature_1": 2.0}
            mock_contextual.return_value = {"contextual_feature_1": 3.0}

            features = await engine_sequential.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=sample_events,
                room_states=sample_room_states,
            )

            # Verify all extractors were called
            mock_temporal.assert_called_once()
            mock_sequential.assert_called_once()
            mock_contextual.assert_called_once()

            # Verify features are properly prefixed
            assert "temporal_temporal_feature_1" in features
            assert "sequential_sequential_feature_1" in features
            assert "contextual_contextual_feature_1" in features

    @pytest.mark.asyncio
    async def test_extract_features_specific_types(
        self, engine, sample_events, sample_room_states, target_time
    ):
        """Test extraction with specific feature types only."""
        with (
            patch.object(
                engine.temporal_extractor, "extract_features"
            ) as mock_temporal,
            patch.object(
                engine.sequential_extractor, "extract_features"
            ) as mock_sequential,
            patch.object(
                engine.contextual_extractor, "extract_features"
            ) as mock_contextual,
        ):

            mock_temporal.return_value = {"temporal_feature": 1.0}

            # Only request temporal features
            features = await engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=sample_events,
                room_states=sample_room_states,
                feature_types=["temporal"],
            )

            # Only temporal extractor should be called
            mock_temporal.assert_called_once()
            mock_sequential.assert_not_called()
            mock_contextual.assert_not_called()

            assert "temporal_temporal_feature" in features
            assert len([k for k in features.keys() if k.startswith("sequential_")]) == 0
            assert len([k for k in features.keys() if k.startswith("contextual_")]) == 0

    @pytest.mark.asyncio
    async def test_extract_batch_features(self, engine):
        """Test batch feature extraction."""
        # Create batch requests
        base_time = datetime(2024, 1, 15, 14, 0, 0)
        requests = []

        for i in range(3):
            room_id = "living_room"
            target_time = base_time + timedelta(hours=i)
            events = [Mock(spec=SensorEvent) for _ in range(2)]
            room_states = [Mock(spec=RoomState) for _ in range(1)]
            requests.append((room_id, target_time, events, room_states))

        with patch.object(engine, "extract_features") as mock_extract:
            mock_extract.return_value = {"test_feature": 1.0}

            results = await engine.extract_batch_features(
                requests, lookback_hours=24, feature_types=["temporal"]
            )

            # Should call extract_features for each request
            assert mock_extract.call_count == 3
            assert len(results) == 3
            assert all(result == {"test_feature": 1.0} for result in results)

    @pytest.mark.asyncio
    async def test_extract_batch_features_with_exceptions(self, engine):
        """Test batch extraction handling exceptions."""
        requests = [
            ("room1", datetime.now(), [], []),
            ("room2", datetime.now(), [], []),
        ]

        with patch.object(engine, "extract_features") as mock_extract:
            # First call succeeds, second fails
            mock_extract.side_effect = [
                {"success_feature": 1.0},
                Exception("Test error"),
            ]

            results = await engine.extract_batch_features(requests)

            # Should handle exception gracefully
            assert len(results) == 2
            assert results[0] == {"success_feature": 1.0}
            # Second result should be default features
            assert isinstance(results[1], dict)
            assert len(results[1]) > 0

    @pytest.mark.asyncio
    async def test_error_handling_invalid_room_id(self, engine):
        """Test error handling with invalid room ID."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        with pytest.raises(FeatureExtractionError, match="Room ID is required"):
            await engine.extract_features(
                room_id="", target_time=target_time  # Empty room ID
            )

    @pytest.mark.asyncio
    async def test_error_handling_extractor_failure(
        self, engine, sample_events, target_time
    ):
        """Test error handling when extractor fails."""
        with patch.object(
            engine.temporal_extractor, "extract_features"
        ) as mock_temporal:
            mock_temporal.side_effect = Exception("Temporal extraction failed")

            with pytest.raises(FeatureExtractionError):
                await engine.extract_features(
                    room_id="living_room",
                    target_time=target_time,
                    events=sample_events,
                    feature_types=["temporal"],
                )

    def test_add_metadata_features(self, engine):
        """Test metadata feature addition."""
        room_id = "living_room"
        target_time = datetime(2024, 1, 15, 15, 30, 0)  # 3:30 PM, Monday
        event_count = 50
        room_state_count = 10

        metadata = engine._add_metadata_features(
            room_id, target_time, event_count, room_state_count
        )

        assert metadata["meta_event_count"] == 50.0
        assert metadata["meta_room_state_count"] == 10.0
        assert metadata["meta_extraction_hour"] == 15.0
        assert metadata["meta_extraction_day_of_week"] == 0.0  # Monday
        assert metadata["meta_data_quality_score"] == min(1.0, 50.0 / 100.0)

    def test_get_feature_names(self, engine):
        """Test feature names retrieval."""
        with (
            patch.object(
                engine.temporal_extractor, "get_feature_names"
            ) as mock_temporal,
            patch.object(
                engine.sequential_extractor, "get_feature_names"
            ) as mock_sequential,
            patch.object(
                engine.contextual_extractor, "get_feature_names"
            ) as mock_contextual,
        ):

            mock_temporal.return_value = ["temp_feature_1", "temp_feature_2"]
            mock_sequential.return_value = ["seq_feature_1"]
            mock_contextual.return_value = ["context_feature_1"]

            feature_names = engine.get_feature_names(
                ["temporal", "sequential", "contextual"]
            )

            # Should include prefixed feature names and metadata
            expected_features = [
                "temporal_temp_feature_1",
                "temporal_temp_feature_2",
                "sequential_seq_feature_1",
                "contextual_context_feature_1",
                "meta_event_count",
                "meta_room_state_count",
                "meta_extraction_hour",
                "meta_extraction_day_of_week",
                "meta_data_quality_score",
            ]

            for feature in expected_features:
                assert feature in feature_names

    def test_create_feature_dataframe(self, engine):
        """Test DataFrame creation from feature dictionaries."""
        feature_dicts = [
            {"feature_1": 1.0, "feature_2": 2.0},
            {"feature_1": 3.0, "feature_2": 4.0, "feature_3": 5.0},
        ]

        with patch.object(engine, "get_feature_names") as mock_get_names:
            mock_get_names.return_value = [
                "feature_1",
                "feature_2",
                "feature_3",
            ]

            df = engine.create_feature_dataframe(feature_dicts)

            assert df.shape == (2, 3)
            assert list(df.columns) == ["feature_1", "feature_2", "feature_3"]

            # Missing features should be filled with 0.0
            assert df.loc[0, "feature_3"] == 0.0

    def test_get_extraction_stats(self, engine):
        """Test extraction statistics retrieval."""
        stats = engine.get_extraction_stats()

        assert isinstance(stats, dict)
        assert "total_extractions" in stats
        assert "successful_extractions" in stats
        assert "failed_extractions" in stats
        assert "avg_extraction_time" in stats
        assert "feature_counts" in stats

    def test_reset_stats(self, engine):
        """Test statistics reset."""
        # Modify stats
        engine.stats["total_extractions"] = 10
        engine.stats["successful_extractions"] = 8

        engine.reset_stats()

        assert engine.stats["total_extractions"] == 0
        assert engine.stats["successful_extractions"] == 0
        assert engine.stats["failed_extractions"] == 0

    def test_clear_caches(self, engine):
        """Test cache clearing."""
        with (
            patch.object(engine.temporal_extractor, "clear_cache") as mock_temp_clear,
            patch.object(engine.sequential_extractor, "clear_cache") as mock_seq_clear,
            patch.object(
                engine.contextual_extractor, "clear_cache"
            ) as mock_context_clear,
        ):

            engine.clear_caches()

            mock_temp_clear.assert_called_once()
            mock_seq_clear.assert_called_once()
            mock_context_clear.assert_called_once()

    @pytest.mark.asyncio
    async def test_validate_configuration(self, engine):
        """Test configuration validation."""
        validation_results = await engine.validate_configuration()

        assert isinstance(validation_results, dict)
        assert "valid" in validation_results
        assert "warnings" in validation_results
        assert "errors" in validation_results

    @pytest.mark.asyncio
    async def test_validate_configuration_no_config(self):
        """Test configuration validation without config."""
        engine = FeatureEngineeringEngine(config=None)

        validation_results = await engine.validate_configuration()

        assert validation_results["valid"] is False
        assert len(validation_results["errors"]) > 0

    def test_get_default_features(self, engine):
        """Test default features retrieval."""
        defaults = engine._get_default_features()

        assert isinstance(defaults, dict)
        assert len(defaults) > 50  # Should have many default features

        # Should have prefixed features from all extractors
        temporal_features = [k for k in defaults.keys() if k.startswith("temporal_")]
        sequential_features = [
            k for k in defaults.keys() if k.startswith("sequential_")
        ]
        contextual_features = [
            k for k in defaults.keys() if k.startswith("contextual_")
        ]
        meta_features = [k for k in defaults.keys() if k.startswith("meta_")]

        assert len(temporal_features) > 0
        assert len(sequential_features) > 0
        assert len(contextual_features) > 0
        assert len(meta_features) > 0

    @pytest.mark.asyncio
    async def test_parallel_vs_sequential_consistency(
        self, mock_config, sample_events, sample_room_states, target_time
    ):
        """Test that parallel and sequential processing produce identical results."""
        # Create engines with same config
        parallel_engine = FeatureEngineeringEngine(
            config=mock_config, enable_parallel=True
        )
        sequential_engine = FeatureEngineeringEngine(
            config=mock_config, enable_parallel=False
        )

        with (
            patch.object(TemporalFeatureExtractor, "extract_features") as mock_temporal,
            patch.object(
                SequentialFeatureExtractor, "extract_features"
            ) as mock_sequential,
            patch.object(
                ContextualFeatureExtractor, "extract_features"
            ) as mock_contextual,
        ):

            # Mock consistent responses
            mock_temporal.return_value = {"temp_1": 1.0, "temp_2": 2.0}
            mock_sequential.return_value = {"seq_1": 3.0}
            mock_contextual.return_value = {"context_1": 4.0}

            parallel_features = await parallel_engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=sample_events,
                room_states=sample_room_states,
            )

            sequential_features = await sequential_engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=sample_events,
                room_states=sample_room_states,
            )

            # Results should be identical (excluding potentially different metadata timestamps)
            parallel_clean = {
                k: v for k, v in parallel_features.items() if not k.startswith("meta_")
            }
            sequential_clean = {
                k: v
                for k, v in sequential_features.items()
                if not k.startswith("meta_")
            }

            assert parallel_clean == sequential_clean

    @pytest.mark.asyncio
    async def test_memory_efficiency(self, engine, target_time):
        """Test memory usage doesn't grow excessively during extraction."""
        import gc
        import sys

        # Force garbage collection before test
        gc.collect()
        initial_size = sys.getsizeof(engine)

        # Run many extractions
        for i in range(100):
            events = [Mock(spec=SensorEvent) for _ in range(10)]
            room_states = [Mock(spec=RoomState) for _ in range(5)]

            # Mock extractors to return consistent results
            with (
                patch.object(
                    engine.temporal_extractor,
                    "extract_features",
                    return_value={"temp": 1.0},
                ),
                patch.object(
                    engine.sequential_extractor,
                    "extract_features",
                    return_value={"seq": 2.0},
                ),
                patch.object(
                    engine.contextual_extractor,
                    "extract_features",
                    return_value={"context": 3.0},
                ),
            ):

                await engine.extract_features(
                    room_id="test_room",
                    target_time=target_time,
                    events=events,
                    room_states=room_states,
                )

        gc.collect()
        final_size = sys.getsizeof(engine)

        # Memory usage shouldn't grow significantly
        assert final_size - initial_size < 5000  # Less than 5KB growth

    @pytest.mark.asyncio
    async def test_concurrent_extractions(self, engine, target_time):
        """Test concurrent feature extractions for thread safety."""

        async def single_extraction(room_id: str):
            events = [Mock(spec=SensorEvent) for _ in range(5)]
            room_states = [Mock(spec=RoomState) for _ in range(2)]

            with (
                patch.object(
                    engine.temporal_extractor,
                    "extract_features",
                    return_value={"temp": 1.0},
                ),
                patch.object(
                    engine.sequential_extractor,
                    "extract_features",
                    return_value={"seq": 2.0},
                ),
                patch.object(
                    engine.contextual_extractor,
                    "extract_features",
                    return_value={"context": 3.0},
                ),
            ):

                return await engine.extract_features(
                    room_id=room_id,
                    target_time=target_time,
                    events=events,
                    room_states=room_states,
                )

        # Run concurrent extractions
        tasks = [single_extraction(f"room_{i}") for i in range(10)]
        results = await asyncio.gather(*tasks)

        # All results should be valid
        assert len(results) == 10
        for result in results:
            assert isinstance(result, dict)
            assert len(result) > 0

    @pytest.mark.asyncio
    async def test_extractor_partial_failure_handling(
        self, engine, sample_events, target_time
    ):
        """Test handling when some extractors fail in parallel processing."""
        with (
            patch.object(
                engine.temporal_extractor, "extract_features"
            ) as mock_temporal,
            patch.object(
                engine.sequential_extractor, "extract_features"
            ) as mock_sequential,
            patch.object(
                engine.contextual_extractor, "extract_features"
            ) as mock_contextual,
        ):

            # Temporal succeeds, sequential fails, contextual succeeds
            mock_temporal.return_value = {"temp_feature": 1.0}
            mock_sequential.side_effect = Exception("Sequential failed")
            mock_contextual.return_value = {"context_feature": 3.0}

            # Should not raise exception, but should handle failures gracefully
            features = await engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=sample_events,
                feature_types=["temporal", "sequential", "contextual"],
            )

            # Should have successful features and metadata
            assert "temporal_temp_feature" in features
            assert "contextual_context_feature" in features
            assert "meta_event_count" in features

            # Sequential features should not be present (due to failure)
            sequential_features = [
                k for k in features.keys() if k.startswith("sequential_")
            ]
            assert len(sequential_features) == 0

    def test_initialization_without_config(self):
        """Test engine initialization without configuration."""
        with patch("src.features.engineering.get_config") as mock_get_config:
            mock_config = Mock(spec=SystemConfig)
            mock_get_config.return_value = mock_config

            engine = FeatureEngineeringEngine(config=None)

            assert engine.config == mock_config
            mock_get_config.assert_called_once()

    def test_executor_cleanup(self, engine):
        """Test that thread pool executor is properly cleaned up."""
        assert engine.executor is not None

        # Simulate destruction
        engine.__del__()

        # Executor should be shut down
        # Note: This is hard to test directly, but __del__ should not raise exceptions

    @pytest.mark.asyncio
    async def test_large_feature_set_handling(self, engine):
        """Test handling of large feature sets."""
        target_time = datetime(2024, 1, 15, 15, 0, 0)

        # Mock extractors to return large feature sets
        large_temporal = {f"temporal_feature_{i}": float(i) for i in range(100)}
        large_sequential = {
            f"sequential_feature_{i}": float(i + 100) for i in range(50)
        }
        large_contextual = {
            f"contextual_feature_{i}": float(i + 200) for i in range(75)
        }

        with (
            patch.object(
                engine.temporal_extractor,
                "extract_features",
                return_value=large_temporal,
            ),
            patch.object(
                engine.sequential_extractor,
                "extract_features",
                return_value=large_sequential,
            ),
            patch.object(
                engine.contextual_extractor,
                "extract_features",
                return_value=large_contextual,
            ),
        ):

            features = await engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=[],
                room_states=[],
            )

            # Should handle large feature sets
            temporal_count = len(
                [k for k in features.keys() if k.startswith("temporal_temporal_")]
            )
            sequential_count = len(
                [k for k in features.keys() if k.startswith("sequential_sequential_")]
            )
            contextual_count = len(
                [k for k in features.keys() if k.startswith("contextual_contextual_")]
            )

            assert temporal_count == 100
            assert sequential_count == 50
            assert contextual_count == 75

            # Total features should be all extractors + metadata
            total_expected = 100 + 50 + 75 + 6  # 6 metadata features
            assert len(features) == total_expected

    @pytest.mark.parametrize("enable_parallel", [True, False])
    @pytest.mark.asyncio
    async def test_performance_comparison(self, mock_config, enable_parallel):
        """Test performance difference between parallel and sequential processing."""
        import time

        engine = FeatureEngineeringEngine(
            config=mock_config, enable_parallel=enable_parallel, max_workers=3
        )

        target_time = datetime(2024, 1, 15, 15, 0, 0)
        events = [Mock(spec=SensorEvent) for _ in range(100)]
        room_states = [Mock(spec=RoomState) for _ in range(50)]

        # Mock extractors with slight delay to simulate work
        def mock_extraction_with_delay(*args, **kwargs):
            time.sleep(0.01)  # 10ms delay
            return {"feature": 1.0}

        with (
            patch.object(
                engine.temporal_extractor,
                "extract_features",
                side_effect=mock_extraction_with_delay,
            ),
            patch.object(
                engine.sequential_extractor,
                "extract_features",
                side_effect=mock_extraction_with_delay,
            ),
            patch.object(
                engine.contextual_extractor,
                "extract_features",
                side_effect=mock_extraction_with_delay,
            ),
        ):

            start_time = time.time()
            features = await engine.extract_features(
                room_id="living_room",
                target_time=target_time,
                events=events,
                room_states=room_states,
            )
            end_time = time.time()

            extraction_time = end_time - start_time

            # Parallel should be faster than sequential for this test
            if enable_parallel:
                # Should complete in roughly 10ms (parallel execution)
                assert extraction_time < 0.05  # Allow some overhead
            else:
                # Should take roughly 30ms (sequential execution)
                assert extraction_time > 0.025  # Should be slower than parallel

            # Results should be valid regardless of processing mode
            assert isinstance(features, dict)
            assert len(features) > 0
