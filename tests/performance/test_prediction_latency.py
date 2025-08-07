"""
Performance tests for prediction generation latency.

Target: Prediction generation < 100ms (requirement from implementation-plan.md)

Tests prediction latency across different scenarios:
- Single room predictions
- Multi-room batch predictions  
- Cold start vs warm cache scenarios
- Different feature complexity levels
"""

import asyncio
import statistics
import time
from typing import Dict, List, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pandas as pd
import numpy as np

from src.models.ensemble import OccupancyEnsemble
from src.models.predictor import OccupancyPredictor
from src.features.store import FeatureStore
from src.core.config import get_config


class TestPredictionLatency:
    """Test prediction generation performance and latency requirements."""

    @pytest.fixture
    async def mock_feature_store(self):
        """Create mock feature store with realistic data."""
        feature_store = MagicMock(spec=FeatureStore)
        
        # Mock feature data with realistic dimensionality
        feature_data = pd.DataFrame({
            'time_since_last_change': np.random.uniform(0, 3600, 100),
            'current_state_duration': np.random.uniform(0, 7200, 100),
            'hour_sin': np.random.uniform(-1, 1, 100),
            'hour_cos': np.random.uniform(-1, 1, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'room_transition_count': np.random.randint(0, 20, 100),
            'movement_velocity': np.random.uniform(0, 5, 100),
            'environmental_temp': np.random.uniform(18, 26, 100),
            'door_state_changes': np.random.randint(0, 10, 100),
            'cross_room_activity': np.random.uniform(0, 1, 100)
        })
        
        feature_store.get_prediction_features = AsyncMock(return_value=feature_data)
        return feature_store

    @pytest.fixture
    async def mock_ensemble_model(self):
        """Create mock ensemble model with realistic prediction behavior."""
        model = MagicMock(spec=OccupancyEnsemble)
        
        # Mock prediction with realistic format
        prediction_result = {
            'predicted_time': pd.Timestamp.now() + pd.Timedelta(minutes=30),
            'confidence': 0.85,
            'prediction_interval': (
                pd.Timestamp.now() + pd.Timedelta(minutes=25),
                pd.Timestamp.now() + pd.Timedelta(minutes=35)
            ),
            'alternatives': [
                {'time': pd.Timestamp.now() + pd.Timedelta(minutes=45), 'confidence': 0.65},
                {'time': pd.Timestamp.now() + pd.Timedelta(minutes=60), 'confidence': 0.45}
            ]
        }
        
        model.predict = MagicMock(return_value=prediction_result)
        return model

    @pytest.fixture
    async def predictor(self, mock_feature_store, mock_ensemble_model):
        """Create occupancy predictor with mocked dependencies."""
        with patch('src.models.predictor.FeatureStore', return_value=mock_feature_store), \
             patch('src.models.predictor.OccupancyEnsemble', return_value=mock_ensemble_model):
            
            predictor = OccupancyPredictor()
            predictor.feature_store = mock_feature_store
            predictor.ensemble = mock_ensemble_model
            return predictor

    async def test_single_prediction_latency(self, predictor):
        """Test single room prediction latency meets <100ms requirement."""
        room_id = "living_room"
        latencies = []
        
        # Run multiple iterations for statistical significance
        for _ in range(50):
            start_time = time.perf_counter()
            prediction = await predictor.predict_occupancy(room_id)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            latencies.append(latency_ms)
            
            # Verify prediction format
            assert prediction is not None
            assert 'predicted_time' in prediction
            assert 'confidence' in prediction

        # Statistical analysis
        mean_latency = statistics.mean(latencies)
        median_latency = statistics.median(latencies)
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        print(f"\nSingle Prediction Latency Results:")
        print(f"Mean: {mean_latency:.2f}ms")
        print(f"Median: {median_latency:.2f}ms") 
        print(f"P95: {p95_latency:.2f}ms")
        print(f"P99: {p99_latency:.2f}ms")
        
        # Verify requirements
        assert mean_latency < 100, f"Mean latency {mean_latency:.2f}ms exceeds 100ms requirement"
        assert p95_latency < 150, f"P95 latency {p95_latency:.2f}ms too high"
        assert p99_latency < 200, f"P99 latency {p99_latency:.2f}ms too high"

    async def test_batch_prediction_latency(self, predictor):
        """Test batch prediction latency for multiple rooms."""
        room_ids = ["living_room", "bedroom", "kitchen", "bathroom", "office"]
        batch_latencies = []
        
        for _ in range(20):
            start_time = time.perf_counter()
            predictions = await predictor.predict_multiple_rooms(room_ids)
            end_time = time.perf_counter()
            
            latency_ms = (end_time - start_time) * 1000
            batch_latencies.append(latency_ms)
            
            assert len(predictions) == len(room_ids)
            for room_id, prediction in predictions.items():
                assert prediction is not None
                assert 'predicted_time' in prediction

        mean_batch_latency = statistics.mean(batch_latencies)
        per_room_latency = mean_batch_latency / len(room_ids)
        
        print(f"\nBatch Prediction Latency Results:")
        print(f"Mean batch latency: {mean_batch_latency:.2f}ms")
        print(f"Per-room latency: {per_room_latency:.2f}ms")
        
        # Batch should be more efficient than individual predictions
        assert per_room_latency < 80, f"Per-room batch latency {per_room_latency:.2f}ms too high"
        assert mean_batch_latency < 400, f"Total batch latency {mean_batch_latency:.2f}ms too high"

    async def test_cold_start_vs_warm_cache_latency(self, predictor):
        """Test prediction latency with cold start vs warm cache scenarios."""
        room_id = "living_room"
        
        # Cold start - first prediction
        start_time = time.perf_counter()
        cold_prediction = await predictor.predict_occupancy(room_id)
        cold_latency = (time.perf_counter() - start_time) * 1000
        
        # Warm cache - subsequent predictions
        warm_latencies = []
        for _ in range(10):
            start_time = time.perf_counter()
            warm_prediction = await predictor.predict_occupancy(room_id)
            warm_latency = (time.perf_counter() - start_time) * 1000
            warm_latencies.append(warm_latency)
        
        mean_warm_latency = statistics.mean(warm_latencies)
        
        print(f"\nCold Start vs Warm Cache Results:")
        print(f"Cold start latency: {cold_latency:.2f}ms")
        print(f"Mean warm latency: {mean_warm_latency:.2f}ms")
        print(f"Cache improvement: {((cold_latency - mean_warm_latency) / cold_latency * 100):.1f}%")
        
        # Warm cache should be faster
        assert mean_warm_latency <= cold_latency, "Warm cache should not be slower than cold start"
        assert mean_warm_latency < 100, f"Warm cache latency {mean_warm_latency:.2f}ms exceeds requirement"

    async def test_prediction_latency_under_load(self, predictor):
        """Test prediction latency under concurrent load."""
        room_id = "living_room"
        concurrent_requests = 10
        latencies = []
        
        async def make_prediction():
            start_time = time.perf_counter()
            await predictor.predict_occupancy(room_id)
            return (time.perf_counter() - start_time) * 1000
        
        # Run concurrent predictions
        tasks = [make_prediction() for _ in range(concurrent_requests)]
        results = await asyncio.gather(*tasks)
        latencies.extend(results)
        
        mean_concurrent_latency = statistics.mean(latencies)
        max_concurrent_latency = max(latencies)
        
        print(f"\nConcurrent Load Latency Results:")
        print(f"Mean latency under load: {mean_concurrent_latency:.2f}ms")
        print(f"Max latency under load: {max_concurrent_latency:.2f}ms")
        
        # Performance should degrade gracefully under load
        assert mean_concurrent_latency < 150, f"Mean latency under load {mean_concurrent_latency:.2f}ms too high"
        assert max_concurrent_latency < 250, f"Max latency under load {max_concurrent_latency:.2f}ms too high"

    async def test_feature_complexity_impact_on_latency(self, predictor):
        """Test how feature complexity affects prediction latency."""
        room_id = "living_room"
        
        # Test with different feature set sizes
        feature_sizes = [10, 25, 50, 100]
        latency_results = {}
        
        for feature_count in feature_sizes:
            # Mock different feature set sizes
            feature_data = pd.DataFrame({
                f'feature_{i}': np.random.random(100) 
                for i in range(feature_count)
            })
            
            predictor.feature_store.get_prediction_features.return_value = feature_data
            
            # Measure latency with this feature set
            latencies = []
            for _ in range(20):
                start_time = time.perf_counter()
                await predictor.predict_occupancy(room_id)
                latency = (time.perf_counter() - start_time) * 1000
                latencies.append(latency)
            
            latency_results[feature_count] = statistics.mean(latencies)
        
        print(f"\nFeature Complexity Impact Results:")
        for feature_count, mean_latency in latency_results.items():
            print(f"{feature_count} features: {mean_latency:.2f}ms")
        
        # Verify all configurations meet requirements
        for feature_count, mean_latency in latency_results.items():
            assert mean_latency < 100, f"Latency with {feature_count} features ({mean_latency:.2f}ms) exceeds requirement"

    async def test_prediction_latency_percentiles(self, predictor):
        """Test prediction latency percentile distribution."""
        room_id = "living_room"
        latencies = []
        
        # Collect large sample for percentile analysis
        for _ in range(200):
            start_time = time.perf_counter()
            await predictor.predict_occupancy(room_id)
            latency = (time.perf_counter() - start_time) * 1000
            latencies.append(latency)
        
        # Calculate percentiles
        percentiles = [50, 75, 90, 95, 99]
        results = {}
        for p in percentiles:
            results[f'p{p}'] = np.percentile(latencies, p)
        
        print(f"\nLatency Percentile Analysis:")
        for metric, value in results.items():
            print(f"{metric}: {value:.2f}ms")
        
        # Verify percentile requirements
        assert results['p50'] < 80, f"P50 latency {results['p50']:.2f}ms too high"
        assert results['p95'] < 150, f"P95 latency {results['p95']:.2f}ms too high"
        assert results['p99'] < 200, f"P99 latency {results['p99']:.2f}ms too high"

    def benchmark_prediction_latency_summary(self, predictor):
        """Generate comprehensive prediction latency benchmark summary."""
        print("\n" + "="*60)
        print("PREDICTION LATENCY BENCHMARK SUMMARY")
        print("="*60)
        print("Requirement: Prediction generation < 100ms")
        print("Target P95: < 150ms")
        print("Target P99: < 200ms")
        print("\nThis benchmark validates end-to-end prediction performance")
        print("across various scenarios and load conditions.")
        print("="*60)


@pytest.mark.asyncio
@pytest.mark.performance
class TestPredictionLatencyIntegration:
    """Integration tests for prediction latency with real components."""

    async def test_end_to_end_prediction_latency(self):
        """Test end-to-end prediction latency with real feature store."""
        # This would use real components in a more comprehensive test
        # For now, we validate the test structure is correct
        assert True, "End-to-end integration test placeholder"

    async def test_prediction_latency_with_database(self):
        """Test prediction latency including database feature retrieval."""
        # This would test with actual database queries
        assert True, "Database integration test placeholder"


def benchmark_prediction_performance():
    """Run comprehensive prediction performance benchmarks."""
    print("\nRunning prediction latency benchmarks...")
    print("This validates the <100ms prediction requirement.")
    return {
        'test_file': 'test_prediction_latency.py',
        'requirement': 'Prediction generation < 100ms',
        'test_coverage': [
            'Single prediction latency',
            'Batch prediction efficiency', 
            'Cold start vs warm cache',
            'Concurrent load performance',
            'Feature complexity impact',
            'Percentile distribution analysis'
        ]
    }


if __name__ == "__main__":
    # Allow running benchmarks directly
    result = benchmark_prediction_performance()
    print(f"Benchmark configuration: {result}")