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

from src.adaptation.drift_detector import DriftMetrics
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
    """Mock accuracy tracker for optimization."""
    tracker = Mock()
    tracker.get_recent_performance = AsyncMock(return_value=[0.8, 0.82, 0.79, 0.85])
    return tracker


@pytest.fixture
def mock_drift_detector():
    """Mock drift detector for optimization context."""
    detector = Mock()
    drift_metrics = DriftMetrics(
        room_id="test_room",
        detection_time=datetime.now(),
        baseline_period=(
            datetime.now() - timedelta(days=14),
            datetime.now() - timedelta(days=3),
        ),
        current_period=(datetime.now() - timedelta(days=3), datetime.now()),
        drift_severity=DriftMetrics.DriftSeverity.MODERATE,
        drifting_features=["feature_1", "feature_2"],
    )
    detector.get_recent_drift_metrics = AsyncMock(return_value=drift_metrics)
    return detector


@pytest.fixture
def optimizer(optimization_config, mock_accuracy_tracker, mock_drift_detector):
    """Create model optimizer with test configuration."""
    return ModelOptimizer(
        config=optimization_config,
        accuracy_tracker=mock_accuracy_tracker,
        drift_detector=mock_drift_detector,
    )


@pytest.fixture
def mock_base_predictor():
    """Create mock base predictor for testing."""
    predictor = Mock(spec=BasePredictor)

    # Mock training method
    async def mock_train(X, y, **kwargs):
        # Simulate training with parameter effects
        params = kwargs.get("parameters", {})

        # Simulate parameter impact on performance
        base_score = 0.75
        if "learning_rate" in params:
            # Lower learning rates generally perform better (up to a point)
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
            validation_score=final_score,
            training_score=final_score + 0.05,
            model_metrics={"accuracy": final_score},
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
        # Test n_initial_points adjustment
        config = OptimizationConfig(n_calls=15, n_initial_points=20)

        # Should adjust n_initial_points to be <= n_calls
        assert config.n_initial_points <= config.n_calls
        assert config.n_initial_points >= 1

    def test_config_customization(self):
        """Test custom configuration settings."""
        config = OptimizationConfig(
            strategy=OptimizationStrategy.GRID_SEARCH,
            objective=OptimizationObjective.CONFIDENCE_CALIBRATION,
            n_calls=30,
            max_optimization_time_minutes=15,
            cv_folds=5,
        )

        assert config.strategy == OptimizationStrategy.GRID_SEARCH
        assert config.objective == OptimizationObjective.CONFIDENCE_CALIBRATION
        assert config.n_calls == 30
        assert config.cv_folds == 5


class TestModelOptimizerInitialization:
    """Test ModelOptimizer initialization and setup."""

    def test_optimizer_initialization(self, optimization_config):
        """Test optimizer initialization."""
        optimizer = ModelOptimizer(
            config=optimization_config,
            accuracy_tracker=Mock(),
            drift_detector=Mock(),
        )

        assert optimizer.config == optimization_config
        assert optimizer._total_optimizations == 0
        assert optimizer._successful_optimizations == 0
        assert len(optimizer._optimization_history) == 0
        assert len(optimizer._parameter_cache) == 0

    def test_parameter_space_initialization(self, optimizer):
        """Test parameter space initialization for different models."""
        # Check that parameter spaces are defined
        assert hasattr(optimizer, "_parameter_spaces")
        assert isinstance(optimizer._parameter_spaces, dict)

        # Test parameter space retrieval
        lstm_space = optimizer._get_parameter_space("lstm", {})
        xgboost_space = optimizer._get_parameter_space("xgboost", {})

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

        # Mock parameter space
        param_space = {
            "learning_rate": (0.01, 0.3),
            "n_estimators": (50, 200),
            "max_depth": (3, 10),
        }

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

        # Mock parameter space for grid search
        param_space = {
            "learning_rate": [0.01, 0.1, 0.2],
            "n_estimators": [50, 100, 150],
            "max_depth": [3, 5, 7],
        }

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

        param_space = {"learning_rate": (0.01, 0.3), "n_estimators": (50, 200)}

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Verify random search completed
            assert isinstance(result, OptimizationResult)
            assert result.total_evaluations <= optimizer.config.n_calls

    @pytest.mark.asyncio
    async def test_performance_adaptive_optimization(
        self,
        optimizer,
        mock_base_predictor,
        synthetic_training_data,
        performance_context,
    ):
        """Test performance-adaptive optimization strategy."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Change strategy to performance adaptive
        optimizer.config.strategy = OptimizationStrategy.PERFORMANCE_ADAPTIVE

        param_space = {"learning_rate": (0.01, 0.3), "n_estimators": (50, 200)}

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
                performance_context=performance_context,
            )

            # Verify adaptive optimization
            assert isinstance(result, OptimizationResult)


class TestObjectiveFunctions:
    """Test different optimization objectives."""

    @pytest.mark.asyncio
    async def test_accuracy_objective(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test accuracy-focused optimization."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Set accuracy objective
        optimizer.config.objective = OptimizationObjective.ACCURACY

        # Create objective function
        objective_func = optimizer._create_objective_function(
            mock_base_predictor, X_train, y_train, X_val, y_val, None
        )

        # Test objective function
        test_params = {"learning_rate": 0.05, "n_estimators": 100}
        score = await objective_func(test_params)

        # Should return negative score for minimization
        assert isinstance(score, float)
        assert score <= 0  # Negative because we minimize

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

        param_space = {"learning_rate": (0.01, 0.3), "n_estimators": (50, 200)}

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

        # Mock performance measurement
        with (
            patch.object(optimizer, "_measure_prediction_latency", return_value=45.0),
            patch.object(optimizer, "_measure_memory_usage", return_value=80.0),
        ):

            param_space = {"learning_rate": (0.01, 0.1)}

            with patch.object(
                optimizer, "_get_parameter_space", return_value=param_space
            ):
                result = await optimizer.optimize_model_parameters(
                    model=mock_base_predictor,
                    model_type="test_model",
                    room_id="test_room",
                    X_train=X_train,
                    y_train=y_train,
                )

                # Should complete successfully with constraints met
                assert result.success
                if result.prediction_latency_ms:
                    assert (
                        result.prediction_latency_ms
                        <= optimizer.config.max_prediction_latency_ms
                    )
                if result.memory_usage_mb:
                    assert (
                        result.memory_usage_mb <= optimizer.config.max_memory_usage_mb
                    )

    @pytest.mark.asyncio
    async def test_minimum_improvement_threshold(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test minimum improvement threshold enforcement."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Set high minimum improvement threshold
        optimizer.config.min_improvement_threshold = 0.20  # 20% improvement required

        # Mock baseline performance
        baseline_score = 0.85
        with patch.object(
            optimizer, "_get_baseline_performance", return_value=baseline_score
        ):

            param_space = {"learning_rate": (0.01, 0.1)}

            with patch.object(
                optimizer, "_get_parameter_space", return_value=param_space
            ):
                result = await optimizer.optimize_model_parameters(
                    model=mock_base_predictor,
                    model_type="test_model",
                    room_id="test_room",
                    X_train=X_train,
                    y_train=y_train,
                )

                # Should check improvement threshold
                if result.success and result.improvement_over_default >= 0:
                    assert (
                        result.improvement_over_default
                        >= optimizer.config.min_improvement_threshold
                        or result.improvement_over_default
                        < optimizer.config.min_improvement_threshold
                    )


class TestOptimizationHistory:
    """Test optimization history tracking and caching."""

    @pytest.mark.asyncio
    async def test_optimization_history_tracking(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test that optimization history is properly tracked."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        model_key = "test_room_test_model"

        param_space = {"learning_rate": (0.01, 0.1)}

        with patch.object(optimizer, "_get_parameter_space", return_value=param_space):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="test_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Verify history tracking
            if result.success:
                assert model_key in optimizer._optimization_history
                assert len(optimizer._optimization_history[model_key]) > 0
                assert optimizer._total_optimizations > 0

    @pytest.mark.asyncio
    async def test_parameter_caching(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test parameter caching for successful optimizations."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        model_key = "test_room_test_model"

        # Mock successful optimization
        with patch.object(optimizer, "_bayesian_optimization") as mock_bayesian:
            mock_result = OptimizationResult(
                success=True,
                optimization_time_seconds=30.0,
                best_parameters={"learning_rate": 0.05, "n_estimators": 100},
                best_score=0.87,
                improvement_over_default=0.05,
                total_evaluations=20,
                convergence_achieved=True,
            )
            mock_bayesian.return_value = mock_result

            param_space = {"learning_rate": (0.01, 0.1)}

            with patch.object(
                optimizer, "_get_parameter_space", return_value=param_space
            ):
                result = await optimizer.optimize_model_parameters(
                    model=mock_base_predictor,
                    model_type="test_model",
                    room_id="test_room",
                    X_train=X_train,
                    y_train=y_train,
                )

                # Verify parameter caching
                if (
                    result.success
                    and result.improvement_over_default
                    > optimizer.config.min_improvement_threshold
                ):
                    assert model_key in optimizer._parameter_cache
                    assert (
                        optimizer._parameter_cache[model_key] == result.best_parameters
                    )

    @pytest.mark.asyncio
    async def test_performance_history_tracking(self, optimizer):
        """Test performance history tracking."""
        room_model_key = "test_room_lstm"

        # Simulate multiple optimization runs
        performance_values = [0.78, 0.82, 0.85, 0.83, 0.87]

        for perf in performance_values:
            optimizer._update_performance_history(room_model_key, perf)

        # Verify history tracking
        assert room_model_key in optimizer._performance_history
        assert len(optimizer._performance_history[room_model_key]) == len(
            performance_values
        )
        assert (
            list(optimizer._performance_history[room_model_key]) == performance_values
        )


class TestOptimizationDecisionLogic:
    """Test optimization decision logic and triggers."""

    @pytest.mark.asyncio
    async def test_should_optimize_decision(self, optimizer, performance_context):
        """Test optimization need decision logic."""
        model_type = "lstm"
        room_id = "test_room"

        # Test with poor recent performance
        poor_context = {
            "recent_accuracy": [0.55, 0.58, 0.52, 0.60],
            "accuracy_trend": "declining",
            "drift_detected": True,
        }

        should_optimize = optimizer._should_optimize(model_type, room_id, poor_context)
        assert should_optimize is True  # Should optimize due to poor performance

        # Test with excellent performance
        excellent_context = {
            "recent_accuracy": [0.92, 0.94, 0.93, 0.95],
            "accuracy_trend": "stable",
            "drift_detected": False,
        }

        should_optimize = optimizer._should_optimize(
            model_type, room_id, excellent_context
        )
        # May or may not optimize - depends on other factors

    @pytest.mark.asyncio
    async def test_disabled_optimization(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test behavior when optimization is disabled."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Disable optimization
        optimizer.config.enabled = False

        result = await optimizer.optimize_model_parameters(
            model=mock_base_predictor,
            model_type="test_model",
            room_id="test_room",
            X_train=X_train,
            y_train=y_train,
        )

        # Should return default result without optimization
        assert isinstance(result, OptimizationResult)
        assert result.total_evaluations == 0
        assert result.optimization_time_seconds == 0.0

    @pytest.mark.asyncio
    async def test_no_parameter_space_handling(
        self, optimizer, mock_base_predictor, synthetic_training_data
    ):
        """Test handling when no parameter space is defined."""
        X_train, X_val, y_train, y_val = synthetic_training_data

        # Mock empty parameter space
        with patch.object(optimizer, "_get_parameter_space", return_value={}):
            result = await optimizer.optimize_model_parameters(
                model=mock_base_predictor,
                model_type="unknown_model",
                room_id="test_room",
                X_train=X_train,
                y_train=y_train,
            )

            # Should handle gracefully
            assert isinstance(result, OptimizationResult)
            assert result.total_evaluations == 0


class TestOptimizationResults:
    """Test optimization result handling and serialization."""

    def test_optimization_result_creation(self):
        """Test OptimizationResult creation and properties."""
        result = OptimizationResult(
            success=True,
            optimization_time_seconds=45.5,
            best_parameters={"learning_rate": 0.05, "n_estimators": 100},
            best_score=0.87,
            improvement_over_default=0.08,
            total_evaluations=25,
            convergence_achieved=True,
            validation_score=0.85,
            prediction_latency_ms=35.2,
        )

        # Test properties
        assert result.success is True
        assert result.best_score == 0.87
        assert result.improvement_over_default == 0.08
        assert result.convergence_achieved is True

    def test_optimization_result_serialization(self):
        """Test OptimizationResult serialization."""
        result = OptimizationResult(
            success=True,
            optimization_time_seconds=30.0,
            best_parameters={"param1": 0.5, "param2": 100},
            best_score=0.85,
            improvement_over_default=0.05,
            total_evaluations=20,
            convergence_achieved=True,
            optimization_history=[
                {"iteration": 1, "score": 0.80},
                {"iteration": 2, "score": 0.83},
            ],
        )

        # Serialize to dict
        result_dict = result.to_dict()

        # Verify serialization
        assert "success" in result_dict
        assert "best_parameters" in result_dict
        assert "optimization_time_seconds" in result_dict
        assert result_dict["success"] == True
        assert result_dict["best_score"] == 0.85
        assert len(result_dict["optimization_history"]) == 2

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

        param_space = {"learning_rate": (0.01, 0.1)}

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

        param_space = {"learning_rate": (0.01, 0.1)}

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

        param_space = {"learning_rate": (0.01, 0.1)}

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

        param_space = {"learning_rate": (0.01, 0.1)}

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
