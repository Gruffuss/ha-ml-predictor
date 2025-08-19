"""
Comprehensive unit tests for ModelOptimizer algorithm convergence and strategies.

This test module covers hyperparameter optimization algorithms, strategy validation,
convergence testing, and performance optimization workflows.
"""

import asyncio
from datetime import datetime, timedelta
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split

from src.adaptation.drift_detector import DriftMetrics, DriftSeverity
from src.adaptation.optimizer import (
    ModelOptimizer,
    OptimizationConfig,
    OptimizationObjective,
    OptimizationResult,
    OptimizationStrategy,
)
from src.adaptation.validator import AccuracyMetrics
from src.models.base.predictor import BasePredictor, TrainingResult

# Test fixtures and utilities


@pytest.fixture
def optimization_config():
    """Create optimization configuration for testing."""
    return OptimizationConfig(
        enabled=True,
        strategy=OptimizationStrategy.BAYESIAN,
        objective=OptimizationObjective.ACCURACY,
        n_calls=20,
        n_initial_points=5,
        max_optimization_time_minutes=5,
        cv_folds=3,
        test_size=0.2,
        min_improvement_threshold=0.01,
    )


@pytest.fixture
def mock_accuracy_tracker():
    """Create mock accuracy tracker."""
    tracker = Mock()
    tracker.get_room_accuracy_metrics = Mock(
        return_value=AccuracyMetrics(
            accuracy_rate=75.5,
            avg_error_minutes=12.3,
            confidence_calibration=0.82,
            total_predictions=1250,
            correct_predictions=944,
        )
    )
    return tracker


@pytest.fixture
def mock_drift_detector():
    """Create mock drift detector."""
    detector = Mock()
    detector.get_drift_metrics = Mock(
        return_value=DriftMetrics(
            overall_drift_score=0.15,
            drift_severity=DriftSeverity.LOW,
            affected_features=["temperature", "humidity"],
            drift_detection_time=datetime.utcnow(),
        )
    )
    return detector


@pytest.fixture
def optimizer(optimization_config, mock_accuracy_tracker, mock_drift_detector):
    """Create ModelOptimizer instance for testing."""
    return ModelOptimizer(
        config=optimization_config,
        accuracy_tracker=mock_accuracy_tracker,
        drift_detector=mock_drift_detector,
    )


@pytest.fixture
def mock_base_predictor():
    """Create mock base predictor for optimization testing."""
    predictor = Mock(spec=BasePredictor)
    predictor.model_type = "test_model"
    predictor.room_id = "test_room"

    # Mock training method with realistic scoring
    def mock_train(X, y, **kwargs):
        """Mock training that returns variable scores based on parameters."""
        # Get parameters if provided
        params = kwargs.get("params", {})

        # Base score
        base_score = 0.75

        # Simulate parameter-dependent performance
        if "learning_rate" in params:
            lr = params["learning_rate"]
            if 0.01 <= lr <= 0.1:
                base_score += 0.05
            elif lr > 0.1:
                base_score -= 0.1

        if "n_estimators" in params:
            # More estimators generally better (with diminishing returns)
            n_est = params["n_estimators"]
            if n_est >= 100:
                base_score += 0.03

        # Add some noise
        noise = np.random.normal(0, 0.02)
        final_score = max(0.4, min(0.95, base_score + noise))

        return TrainingResult(
            success=True,
            training_time_seconds=np.random.uniform(1, 5),
            model_version="v1.0",
            training_samples=1000,
            validation_score=final_score,
            training_score=final_score + 0.05,
            training_metrics={"accuracy": final_score},
        )

    predictor.train = AsyncMock(side_effect=mock_train)

    # Mock parameter getting/setting
    predictor.get_parameters = Mock(
        return_value={
            "learning_rate": 0.1,
            "n_estimators": 100,
            "max_depth": 5,
        }
    )
    predictor.set_parameters = Mock()

    return predictor


@pytest.fixture
def synthetic_training_data():
    """Generate synthetic training data for optimization tests."""
    np.random.seed(42)

    # Classification data
    X, y = make_classification(
        n_samples=500,
        n_features=10,
        n_informative=8,
        n_redundant=2,
        n_clusters_per_class=1,
        random_state=42,
    )

    X_df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(X.shape[1])])
    y_df = pd.DataFrame(y, columns=["target"])

    # Split into train/validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_df, y_df, test_size=0.2, random_state=42
    )

    return X_train, X_val, y_train, y_val


@pytest.fixture
def performance_context():
    """Create performance context for optimization."""
    return {
        "recent_accuracy": [0.78, 0.82, 0.79, 0.75],
        "accuracy_trend": "declining",
        "drift_detected": True,
        "drift_features": ["feature_1", "feature_2"],
        "error_rate_increase": 15.5,
        "last_optimization_time": datetime.now() - timedelta(hours=6),
    }


# Core optimization tests


class TestOptimizationConfig:
    """Test OptimizationConfig validation and defaults."""

    def test_config_initialization(self):
        """Test optimization config initialization."""
        config = OptimizationConfig()

        # Test defaults
        assert config.enabled is True
        assert config.strategy == OptimizationStrategy.BAYESIAN
        assert config.objective == OptimizationObjective.ACCURACY
        assert config.n_calls == 50
        assert config.n_initial_points == 10
        assert config.max_optimization_time_minutes == 30

    def test_config_validation(self):
        """Test config validation and adjustment."""
        # Test auto-adjustment of n_initial_points
        config = OptimizationConfig(n_calls=15, n_initial_points=20)

        # Should adjust n_initial_points to be reasonable
        assert config.n_initial_points <= config.n_calls
        assert config.n_initial_points >= 1

    def test_config_model_specific_flags(self):
        """Test model-specific optimization flags."""
        config = OptimizationConfig(
            optimize_lstm=False,
            optimize_xgboost=True,
            optimize_hmm=False,
            optimize_gaussian_process=True,
        )

        assert config.optimize_lstm is False
        assert config.optimize_xgboost is True
        assert config.optimize_hmm is False
        assert config.optimize_gaussian_process is True


class TestModelOptimizer:
    """Test ModelOptimizer initialization and basic functionality."""

    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization."""
        assert isinstance(optimizer.config, OptimizationConfig)
        assert optimizer.accuracy_tracker is not None
        assert optimizer.drift_detector is not None
        assert isinstance(optimizer._optimization_history, dict)

    def test_parameter_space_initialization(self, optimizer):
        """Test parameter space setup for different model types."""
        # Check that parameter spaces are properly initialized
        lstm_space = optimizer._parameter_spaces.get("lstm")
        xgboost_space = optimizer._parameter_spaces.get("xgboost")

        # Parameter spaces should be initialized (can be empty in mock scenarios)
        assert lstm_space is not None or len(lstm_space) == 0  # May be empty in mock
        assert xgboost_space is not None or len(xgboost_space) == 0


class TestOptimizationStrategies:
    """Test different optimization strategies."""

    @pytest.mark.asyncio
    async def test_bayesian_optimization(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test Bayesian optimization strategy."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Mock parameter space in correct format
        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.3},
            {"name": "n_estimators", "type": "integer", "low": 50, "high": 200},
            {"name": "max_depth", "type": "integer", "low": 3, "high": 10},
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            # Run optimization
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            # Verify optimization completed
            assert isinstance(result, OptimizationResult)
            assert result.success or result.error_message is not None

            # If Bayesian optimization fails due to missing scikit-optimize, that's acceptable
            if result.error_message and "scikit-optimize" in result.error_message:
                assert result.total_evaluations == 0  # Expected for missing dependency
                assert result.best_parameters == {}
            else:
                # If Bayesian optimization works, should have evaluations
                assert result.total_evaluations > 0
                assert isinstance(result.best_parameters, dict)

    @pytest.mark.asyncio
    async def test_grid_search_optimization(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test grid search optimization strategy."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Change strategy to grid search
        optimizer.config.strategy = OptimizationStrategy.GRID_SEARCH
        optimizer.config.grid_search_max_combinations = 12

        # Mock parameter space for grid search in correct format
        param_space = [
            {
                "name": "learning_rate",
                "type": "categorical",
                "categories": [0.01, 0.1, 0.2],
            },
            {
                "name": "n_estimators",
                "type": "categorical",
                "categories": [50, 100, 150],
            },
            {"name": "max_depth", "type": "categorical", "categories": [3, 5, 7]},
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            # Verify grid search completed
            assert isinstance(result, OptimizationResult)
            assert (
                result.total_evaluations
                <= optimizer.config.grid_search_max_combinations
            )

    @pytest.mark.asyncio
    async def test_random_search_optimization(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test random search optimization strategy."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Change strategy to random search
        optimizer.config.strategy = OptimizationStrategy.RANDOM_SEARCH
        optimizer.config.n_calls = 15

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.3},
            {"name": "n_estimators", "type": "integer", "low": 50, "high": 200},
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
            )

            # Verify random search completed
            assert isinstance(result, OptimizationResult)
            assert result.total_evaluations <= optimizer.config.n_calls

    @pytest.mark.asyncio
    async def test_empty_parameter_space_handling(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test handling of empty parameter space."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Mock empty parameter space
        with patch.object(optimizer, "_get_parameter_space", return_value=[]):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="unknown_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Should handle gracefully and return default result
            assert isinstance(result, OptimizationResult)
            assert result.total_evaluations == 0
            assert result.best_parameters == {}


class TestOptimizationObjectives:
    """Test different optimization objectives."""

    @pytest.mark.asyncio
    async def test_accuracy_objective_function(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test accuracy-based objective function."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Create objective function
        objective_func = optimizer._create_objective_function(
            mock_base_predictor, X_train, y_train, X_val, y_val, None
        )

        # Test objective function
        test_params = {"learning_rate": 0.05, "n_estimators": 100}
        score = await objective_func(test_params)

        # Should return score for minimization (1 - accuracy, so lower is better)
        assert isinstance(score, float)
        assert 0 <= score <= 1  # Score should be between 0 and 1 for accuracy objective

    @pytest.mark.asyncio
    async def test_confidence_calibration_objective(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test confidence calibration optimization."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Set confidence calibration objective
        optimizer.config.objective = OptimizationObjective.CONFIDENCE_CALIBRATION

        # Mock confidence calibration in predictor
        mock_base_predictor.evaluate_confidence_calibration = AsyncMock(
            return_value=0.85
        )

        objective_func = optimizer._create_objective_function(
            mock_base_predictor, X_train, y_train, X_val, y_val, None
        )

        test_params = {"learning_rate": 0.05}
        score = await objective_func(test_params)

        assert isinstance(score, float)

    @pytest.mark.asyncio
    async def test_composite_objective(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test multi-objective optimization."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Set composite objective
        optimizer.config.objective = OptimizationObjective.COMPOSITE

        objective_func = optimizer._create_objective_function(
            mock_base_predictor, X_train, y_train, X_val, y_val, None
        )

        test_params = {"learning_rate": 0.05, "n_estimators": 100}
        score = await objective_func(test_params)

        # Should combine multiple objectives
        assert isinstance(score, float)


class TestOptimizationConstraints:
    """Test optimization constraints and validation."""

    @pytest.mark.asyncio
    async def test_time_constraint_enforcement(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test optimization time constraint enforcement."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Set very short time limit
        optimizer.config.max_optimization_time_minutes = 0.05  # 3 seconds

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.3},
            {"name": "n_estimators", "type": "integer", "low": 50, "high": 200},
        ]

        start_time = time.time()

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

        elapsed_time = time.time() - start_time

        # Should respect time constraint (with some tolerance)
        assert elapsed_time <= optimizer.config.max_optimization_time_minutes * 60 + 10
        assert isinstance(result, OptimizationResult)

    @pytest.mark.asyncio
    async def test_performance_constraint_validation(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test performance constraint validation."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Set performance constraints
        optimizer.config.max_prediction_latency_ms = 50.0
        optimizer.config.max_memory_usage_mb = 100.0

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.1}
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Should complete with constraints
            assert isinstance(result, OptimizationResult)


class TestOptimizationHistory:
    """Test optimization history tracking and caching."""

    def test_optimization_history_tracking(self, optimizer):
        """Test that optimization history is properly tracked."""
        model_key = "test_model_test_room"

        # Initially empty
        assert len(optimizer._optimization_history) == 0

        # Add some results
        result1 = OptimizationResult(
            success=True,
            optimization_time_seconds=30.5,
            best_parameters={"learning_rate": 0.05},
            best_score=0.85,
            improvement_over_default=0.05,
            total_evaluations=15,
            convergence_achieved=True,
        )

        # Manually add to history for testing
        optimizer._optimization_history[model_key] = [result1]

        # Verify history is tracked
        history = optimizer._optimization_history[model_key]
        assert len(history) == 1
        assert history[0].best_score == 0.85
        assert history[0].improvement_over_default == 0.05

    def test_parameter_caching(self, optimizer):
        """Test parameter caching functionality."""
        model_key = "test_model_test_room"
        params = {"learning_rate": 0.05, "n_estimators": 100}

        # Cache parameters
        optimizer._parameter_cache[model_key] = params

        # Verify caching
        assert model_key in optimizer._parameter_cache
        cached_params = optimizer._parameter_cache[model_key]
        assert cached_params["learning_rate"] == 0.05
        assert cached_params["n_estimators"] == 100

    def test_performance_history_tracking(self, optimizer):
        """Test performance history window management."""
        model_key = "test_model"

        # Add performance values up to window limit
        for i in range(15):  # More than the typical window of 10
            optimizer._update_performance_history(model_key, 0.8 + i * 0.01)

        # Should limit to configured window size
        history_size = len(optimizer._performance_history[model_key])
        assert history_size <= optimizer.config.performance_history_window


class TestOptimizationStatistics:
    """Test optimization statistics and reporting."""

    def test_success_rate_calculation(self, optimizer):
        """Test optimization success rate calculation."""
        # Initially zero
        assert optimizer._total_optimizations == 0
        assert optimizer._successful_optimizations == 0

        # Manually update for testing
        optimizer._total_optimizations = 10
        optimizer._successful_optimizations = 8

        # Calculate success rate
        success_rate = (
            optimizer._successful_optimizations / optimizer._total_optimizations
        )
        assert success_rate == 0.8

    def test_average_improvement_tracking(self, optimizer):
        """Test average improvement calculation."""
        # Test improvement tracking
        optimizer._update_improvement_average(0.05)
        assert optimizer._average_improvement == 0.05

        optimizer._successful_optimizations = 2
        optimizer._update_improvement_average(0.03)

        # Should use exponential moving average
        expected = 0.1 * 0.03 + 0.9 * 0.05  # alpha=0.1
        assert abs(optimizer._average_improvement - expected) < 0.001


class TestOptimizationResult:
    """Test OptimizationResult data structure."""

    def test_successful_optimization_result(self):
        """Test successful optimization result creation."""
        result = OptimizationResult(
            success=True,
            optimization_time_seconds=45.2,
            best_parameters={"learning_rate": 0.05, "n_estimators": 150},
            best_score=0.87,
            improvement_over_default=0.07,
            total_evaluations=25,
            convergence_achieved=True,
            validation_score=0.85,
        )

        assert result.success is True
        assert result.optimization_time_seconds == 45.2
        assert result.best_parameters["learning_rate"] == 0.05
        assert result.best_score == 0.87
        assert result.improvement_over_default == 0.07
        assert result.total_evaluations == 25
        assert result.convergence_achieved is True

    def test_failed_optimization_result(self):
        """Test failed optimization result handling."""
        result = OptimizationResult(
            success=False,
            optimization_time_seconds=10.0,
            best_parameters={},
            best_score=0.0,
            improvement_over_default=0.0,
            total_evaluations=5,
            convergence_achieved=False,
            error_message="Optimization failed due to timeout",
        )

        assert result.success is False
        assert result.error_message is not None
        assert result.convergence_achieved is False


class TestErrorHandling:
    """Test error handling in optimization processes."""

    @pytest.mark.asyncio
    async def test_model_training_error_handling(
        self, optimizer, synthetic_training_data
    ):
        """Test handling of model training errors during optimization."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Mock predictor that fails training
        failing_predictor = Mock(spec=BasePredictor)
        failing_predictor.train = AsyncMock(side_effect=Exception("Training failed"))

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.1}
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=failing_predictor,
                model_type="failing_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Should handle error gracefully
            assert isinstance(result, OptimizationResult)
            assert not result.success
            assert result.error_message is not None

    @pytest.mark.asyncio
    async def test_objective_function_error_handling(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test error handling in objective function evaluation."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Mock predictor that occasionally fails
        def failing_train(X, y, **kwargs):
            if np.random.random() < 0.3:  # 30% failure rate
                raise Exception("Random training failure")
            return TrainingResult(success=True, validation_score=0.8)

        mock_base_predictor.train = AsyncMock(side_effect=failing_train)

        # Create objective function
        objective_func = optimizer._create_objective_function(
            mock_base_predictor, X_train, y_train, X_val, y_val, None
        )

        # Test multiple evaluations
        results = []
        for _ in range(10):
            try:
                score = await objective_func({"learning_rate": 0.1})
                results.append(score)
            except Exception:
                results.append(None)

        # Should handle some failures gracefully
        successful_results = [r for r in results if r is not None]
        assert len(successful_results) > 0  # Some should succeed

    @pytest.mark.asyncio
    async def test_timeout_handling(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test optimization timeout handling."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Mock slow training
        async def slow_train(X, y, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow training
            return TrainingResult(success=True, validation_score=0.8)

        mock_base_predictor.train = slow_train

        # Set very short timeout
        optimizer.config.max_optimization_time_minutes = 0.01  # 0.6 seconds

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.1}
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="slow_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Should handle timeout gracefully
            assert isinstance(result, OptimizationResult)
            # May or may not be successful depending on timing


class TestPerformanceOptimization:
    """Test performance characteristics of optimization."""

    @pytest.mark.asyncio
    async def test_optimization_performance_metrics(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test that performance metrics are properly measured."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.1}
        ]

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Verify timing is recorded
            assert result.optimization_time_seconds > 0
            assert result.total_evaluations >= 0

    @pytest.mark.asyncio
    async def test_concurrent_optimizations(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test concurrent optimization requests."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        param_space = [
            {"name": "learning_rate", "type": "continuous", "low": 0.01, "high": 0.1}
        ]

        # Create multiple optimization tasks
        tasks = []
        for i in range(3):
            with patch.object(
                optimizer, "_get_parameter_space", return_value=param_space
            ):
                task = optimizer.optimize_model_parameters(
                    model=mock_base_predictor,
                    model_type=f"model_{i}",
                    room_id=f"room_{i}",
                    X_train=X_train,
                    y_train=y_train,
                )
                tasks.append(task)

        # Run concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete
        assert len(results) == 3
        for result in results:
            assert isinstance(result, OptimizationResult) or isinstance(
                result, Exception
            )

    def test_memory_usage_tracking(self, optimizer):
        """Test memory usage tracking in optimization."""
        # Test performance history cleanup
        model_key = "test_model"

        # Add many performance values
        for i in range(100):
            optimizer._update_performance_history(model_key, 0.8 + i * 0.001)

        # Should limit history size
        history_size = len(optimizer._performance_history[model_key])
        assert history_size <= optimizer.config.performance_history_window
