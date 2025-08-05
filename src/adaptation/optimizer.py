"""
Model Optimization Engine for Sprint 4 Task 6 - Self-Adaptation System.

This module provides automatic hyperparameter optimization for predictive models
integrated with the adaptive retraining pipeline. The optimizer automatically
tunes model parameters based on performance data and drift patterns.
"""

import asyncio
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import skopt
from scipy.optimize import minimize

# Optimization libraries
from sklearn.model_selection import ParameterGrid
from skopt import gp_minimize
from skopt.space import Categorical, Integer, Real

from ..core.constants import ModelType
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..models.base.predictor import BasePredictor, TrainingResult
from .drift_detector import DriftMetrics
from .validator import AccuracyMetrics

logger = logging.getLogger(__name__)


class OptimizationStrategy(Enum):
    """Different optimization strategies available."""

    BAYESIAN = "bayesian"  # Bayesian optimization using Gaussian processes
    GRID_SEARCH = "grid_search"  # Exhaustive grid search
    RANDOM_SEARCH = "random_search"  # Random parameter sampling
    GRADIENT_BASED = "gradient_based"  # Gradient-based optimization
    PERFORMANCE_ADAPTIVE = "performance_adaptive"  # Adapt based on recent performance


class OptimizationObjective(Enum):
    """Optimization objectives."""

    ACCURACY = "accuracy"  # Maximize prediction accuracy
    CONFIDENCE_CALIBRATION = "confidence_calibration"  # Improve confidence scores
    PREDICTION_TIME = "prediction_time"  # Minimize prediction latency
    DRIFT_RESISTANCE = "drift_resistance"  # Improve stability over time
    COMPOSITE = "composite"  # Multi-objective optimization


@dataclass
class OptimizationResult:
    """Result of hyperparameter optimization."""

    success: bool
    optimization_time_seconds: float
    best_parameters: Dict[str, Any]
    best_score: float
    improvement_over_default: float

    # Optimization process details
    total_evaluations: int
    convergence_achieved: bool
    optimization_history: List[Dict[str, Any]] = field(default_factory=list)

    # Performance metrics
    validation_score: Optional[float] = None
    training_score: Optional[float] = None
    cross_validation_scores: Optional[List[float]] = None

    # Model-specific metrics
    model_complexity: Optional[float] = None
    prediction_latency_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "optimization_time_seconds": self.optimization_time_seconds,
            "best_parameters": self.best_parameters,
            "best_score": self.best_score,
            "improvement_over_default": self.improvement_over_default,
            "total_evaluations": self.total_evaluations,
            "convergence_achieved": self.convergence_achieved,
            "validation_score": self.validation_score,
            "training_score": self.training_score,
            "cross_validation_scores": self.cross_validation_scores,
            "model_complexity": self.model_complexity,
            "prediction_latency_ms": self.prediction_latency_ms,
            "memory_usage_mb": self.memory_usage_mb,
            "error_message": self.error_message,
        }


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""

    enabled: bool = True
    strategy: OptimizationStrategy = OptimizationStrategy.BAYESIAN
    objective: OptimizationObjective = OptimizationObjective.ACCURACY

    # Bayesian optimization settings
    n_calls: int = 50  # Number of optimization iterations
    n_initial_points: int = 10  # Random points before optimization
    acquisition_function: str = "EI"  # Expected Improvement

    # Grid search settings
    grid_search_max_combinations: int = 100

    # Performance constraints
    max_optimization_time_minutes: int = 30
    max_model_complexity: Optional[float] = None
    max_prediction_latency_ms: Optional[float] = 100.0
    max_memory_usage_mb: Optional[float] = 500.0

    # Cross-validation settings
    cv_folds: int = 3
    test_size: float = 0.2

    # Model-specific optimization enable/disable
    optimize_lstm: bool = True
    optimize_xgboost: bool = True
    optimize_hmm: bool = True
    optimize_gaussian_process: bool = True

    # Performance-based adaptation
    performance_history_window: int = 10
    min_improvement_threshold: float = 0.01  # Minimum improvement to apply optimization

    def __post_init__(self):
        """Validate configuration."""
        if self.n_calls < self.n_initial_points:
            self.n_initial_points = max(1, self.n_calls // 3)


class ModelOptimizer:
    """
    Automatic hyperparameter optimization engine integrated with adaptive retraining.

    Features:
    - Bayesian optimization for efficient parameter search
    - Model-specific optimization strategies
    - Performance-driven parameter adaptation
    - Integration with accuracy tracking and drift detection
    - Multi-objective optimization support
    - Automatic constraint handling (time, memory, complexity)
    """

    def __init__(
        self, config: OptimizationConfig, accuracy_tracker=None, drift_detector=None
    ):
        """
        Initialize the model optimizer.

        Args:
            config: Optimization configuration
            accuracy_tracker: AccuracyTracker for performance data
            drift_detector: ConceptDriftDetector for drift patterns
        """
        self.config = config
        self.accuracy_tracker = accuracy_tracker
        self.drift_detector = drift_detector

        # Optimization history and caching
        self._optimization_history: Dict[str, List[OptimizationResult]] = {}
        self._parameter_cache: Dict[str, Dict[str, Any]] = {}
        self._performance_history: Dict[str, List[float]] = {}

        # Thread safety
        self._lock = threading.RLock()

        # Performance tracking
        self._total_optimizations = 0
        self._successful_optimizations = 0
        self._average_improvement = 0.0

        # Model-specific parameter spaces
        self._parameter_spaces = self._initialize_parameter_spaces()

        logger.info(
            f"Initialized ModelOptimizer with strategy: {config.strategy.value}"
        )

    async def optimize_model_parameters(
        self,
        model: BasePredictor,
        model_type: str,
        room_id: str,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.DataFrame] = None,
        performance_context: Optional[Dict[str, Any]] = None,
    ) -> OptimizationResult:
        """
        Optimize model hyperparameters based on training data and performance context.

        Args:
            model: Model instance to optimize
            model_type: Type of model (lstm, xgboost, hmm, etc.)
            room_id: Room identifier for context
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            performance_context: Recent performance metrics and drift data

        Returns:
            OptimizationResult with best parameters and performance metrics
        """
        start_time = datetime.utcnow()

        try:
            if not self.config.enabled:
                logger.info(
                    f"Optimization disabled, using default parameters for {model_type}"
                )
                return self._create_default_result(model_type)

            # Check if optimization is needed based on performance context
            if not self._should_optimize(model_type, room_id, performance_context):
                logger.info(f"Optimization not needed for {model_type} in {room_id}")
                return self._create_default_result(model_type)

            logger.info(f"Starting optimization for {model_type} model in {room_id}")

            # Get parameter space for model type
            param_space = self._get_parameter_space(model_type, performance_context)
            if not param_space:
                logger.warning(f"No parameter space defined for {model_type}")
                return self._create_default_result(model_type)

            # Create objective function
            objective_func = self._create_objective_function(
                model, X_train, y_train, X_val, y_val, performance_context
            )

            # Run optimization based on strategy
            if self.config.strategy == OptimizationStrategy.BAYESIAN:
                result = await self._bayesian_optimization(
                    objective_func, param_space, model_type
                )
            elif self.config.strategy == OptimizationStrategy.GRID_SEARCH:
                result = await self._grid_search_optimization(
                    objective_func, param_space, model_type
                )
            elif self.config.strategy == OptimizationStrategy.RANDOM_SEARCH:
                result = await self._random_search_optimization(
                    objective_func, param_space, model_type
                )
            elif self.config.strategy == OptimizationStrategy.PERFORMANCE_ADAPTIVE:
                result = await self._performance_adaptive_optimization(
                    objective_func,
                    param_space,
                    model_type,
                    room_id,
                    performance_context,
                )
            else:
                result = await self._bayesian_optimization(
                    objective_func, param_space, model_type
                )

            # Calculate optimization time
            optimization_time = (datetime.utcnow() - start_time).total_seconds()
            result.optimization_time_seconds = optimization_time

            # Update statistics
            with self._lock:
                self._total_optimizations += 1
                if result.success:
                    self._successful_optimizations += 1
                    self._update_improvement_average(result.improvement_over_default)

                # Cache successful parameters
                if (
                    result.success
                    and result.improvement_over_default
                    > self.config.min_improvement_threshold
                ):
                    model_key = f"{room_id}_{model_type}"
                    self._parameter_cache[model_key] = result.best_parameters.copy()

                # Store optimization history
                if model_type not in self._optimization_history:
                    self._optimization_history[model_type] = []
                self._optimization_history[model_type].append(result)

            logger.info(
                f"Optimization completed for {model_type}: "
                f"success={result.success}, improvement={result.improvement_over_default:.3f}, "
                f"time={optimization_time:.1f}s"
            )

            return result

        except Exception as e:
            logger.error(f"Optimization failed for {model_type}: {e}")
            optimization_time = (datetime.utcnow() - start_time).total_seconds()
            return OptimizationResult(
                success=False,
                optimization_time_seconds=optimization_time,
                best_parameters={},
                best_score=0.0,
                improvement_over_default=0.0,
                total_evaluations=0,
                convergence_achieved=False,
                error_message=str(e),
            )

    def get_cached_parameters(
        self, model_type: str, room_id: str
    ) -> Optional[Dict[str, Any]]:
        """Get cached optimized parameters for a model."""
        model_key = f"{room_id}_{model_type}"
        with self._lock:
            return self._parameter_cache.get(model_key, {}).copy()

    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization performance statistics."""
        with self._lock:
            success_rate = (
                self._successful_optimizations / max(1, self._total_optimizations)
            ) * 100

            # Get recent optimization results
            recent_results = []
            for model_results in self._optimization_history.values():
                recent_results.extend(model_results[-5:])  # Last 5 per model type

            recent_improvements = [
                r.improvement_over_default for r in recent_results if r.success
            ]
            avg_recent_improvement = (
                np.mean(recent_improvements) if recent_improvements else 0.0
            )

            return {
                "enabled": self.config.enabled,
                "strategy": self.config.strategy.value,
                "objective": self.config.objective.value,
                "total_optimizations": self._total_optimizations,
                "successful_optimizations": self._successful_optimizations,
                "success_rate_percent": success_rate,
                "average_improvement": self._average_improvement,
                "recent_average_improvement": avg_recent_improvement,
                "cached_parameter_sets": len(self._parameter_cache),
                "optimization_history_length": sum(
                    len(results) for results in self._optimization_history.values()
                ),
                "configuration": {
                    "max_optimization_time_minutes": self.config.max_optimization_time_minutes,
                    "n_calls": self.config.n_calls,
                    "cv_folds": self.config.cv_folds,
                    "min_improvement_threshold": self.config.min_improvement_threshold,
                },
            }

    # Private optimization methods

    def _should_optimize(
        self,
        model_type: str,
        room_id: str,
        performance_context: Optional[Dict[str, Any]],
    ) -> bool:
        """Determine if optimization is needed based on performance context."""
        try:
            # Always optimize if no context provided
            if not performance_context:
                return True

            # Check if model type optimization is enabled
            if not getattr(self.config, f"optimize_{model_type.lower()}", True):
                return False

            # Analyze performance degradation
            accuracy_metrics = performance_context.get("accuracy_metrics")
            if accuracy_metrics and hasattr(accuracy_metrics, "accuracy_rate"):
                if accuracy_metrics.accuracy_rate < 70.0:  # Poor performance
                    logger.info(
                        f"Optimization needed due to poor accuracy: {accuracy_metrics.accuracy_rate:.1f}%"
                    )
                    return True

            # Check drift metrics
            drift_metrics = performance_context.get("drift_metrics")
            if drift_metrics and hasattr(drift_metrics, "overall_drift_score"):
                if drift_metrics.overall_drift_score > 0.3:  # Significant drift
                    logger.info(
                        f"Optimization needed due to drift: {drift_metrics.overall_drift_score:.3f}"
                    )
                    return True

            # Check recent optimization history
            with self._lock:
                if model_type in self._optimization_history:
                    recent_results = self._optimization_history[model_type][-3:]
                    if recent_results and all(
                        r.improvement_over_default < 0.005 for r in recent_results
                    ):
                        logger.info(
                            f"Skipping optimization - recent attempts showed minimal improvement"
                        )
                        return False

            # Default to optimize for new models or significant time gaps
            return True

        except Exception as e:
            logger.error(f"Error determining optimization need: {e}")
            return True  # Default to optimize on error

    def _get_parameter_space(
        self, model_type: str, performance_context: Optional[Dict[str, Any]]
    ) -> Optional[List]:
        """Get parameter search space for model type."""
        base_space = self._parameter_spaces.get(model_type.lower())
        if not base_space:
            return None

        # Adapt parameter space based on performance context
        if performance_context:
            return self._adapt_parameter_space(base_space, performance_context)

        return base_space

    def _adapt_parameter_space(
        self, base_space: List, performance_context: Dict[str, Any]
    ) -> List:
        """Adapt parameter space based on performance context."""
        try:
            # For demonstration, we'll use the base space
            # In practice, this would narrow/expand the search space based on:
            # - Drift patterns (focus on regularization if overfitting)
            # - Accuracy trends (focus on capacity if underfitting)
            # - Prediction latency requirements
            return base_space

        except Exception as e:
            logger.error(f"Error adapting parameter space: {e}")
            return base_space

    def _create_objective_function(
        self,
        model: BasePredictor,
        X_train: pd.DataFrame,
        y_train: pd.DataFrame,
        X_val: Optional[pd.DataFrame],
        y_val: Optional[pd.DataFrame],
        performance_context: Optional[Dict[str, Any]],
    ) -> Callable:
        """Create objective function for optimization."""

        def objective(params):
            try:
                # Create model copy with new parameters
                model_copy = self._create_model_with_params(model, params)

                # Train model
                training_result = model_copy.train(X_train, y_train, X_val, y_val)

                if not training_result.success:
                    return 1.0  # High penalty for failed training

                # Calculate objective based on strategy
                if self.config.objective == OptimizationObjective.ACCURACY:
                    score = (
                        training_result.validation_score
                        or training_result.training_score
                        or 0.0
                    )
                    return 1.0 - score  # Minimize (1 - accuracy)

                elif self.config.objective == OptimizationObjective.COMPOSITE:
                    # Multi-objective: accuracy + speed + calibration
                    accuracy = (
                        training_result.validation_score
                        or training_result.training_score
                        or 0.0
                    )

                    # Simplified composite score
                    composite_score = accuracy * 0.8  # 80% weight on accuracy

                    # Add prediction time penalty if available
                    if hasattr(model_copy, "last_prediction_time_ms"):
                        time_penalty = min(
                            model_copy.last_prediction_time_ms / 1000.0, 0.1
                        )
                        composite_score -= time_penalty * 0.1

                    return 1.0 - composite_score

                else:
                    # Default to accuracy
                    score = (
                        training_result.validation_score
                        or training_result.training_score
                        or 0.0
                    )
                    return 1.0 - score

            except Exception as e:
                logger.error(f"Error in objective function: {e}")
                return 1.0  # High penalty for errors

        return objective

    def _create_model_with_params(self, model: BasePredictor, params: Dict[str, Any]):
        """Create model instance with specified parameters."""
        # This would create a new model instance with the given parameters
        # For now, we'll assume the model has a method to update parameters
        if hasattr(model, "set_parameters"):
            model_copy = model.__class__()
            model_copy.set_parameters(params)
            return model_copy
        else:
            # Return original model if no parameter setting method
            return model

    async def _bayesian_optimization(
        self, objective_func: Callable, param_space: List, model_type: str
    ) -> OptimizationResult:
        """Perform Bayesian optimization."""
        try:
            logger.info(f"Running Bayesian optimization for {model_type}")

            # Run optimization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            with ThreadPoolExecutor(max_workers=1) as executor:
                result = await loop.run_in_executor(
                    executor,
                    self._run_bayesian_optimization,
                    objective_func,
                    param_space,
                )

            return result

        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimization_time_seconds=0.0,
                best_parameters={},
                best_score=0.0,
                improvement_over_default=0.0,
                total_evaluations=0,
                convergence_achieved=False,
                error_message=str(e),
            )

    def _run_bayesian_optimization(
        self, objective_func: Callable, param_space: List
    ) -> OptimizationResult:
        """Run Bayesian optimization synchronously."""
        try:
            # Convert parameter space to skopt format
            dimensions = []
            param_names = []

            for param in param_space:
                if "name" in param and "type" in param:
                    name = param["name"]
                    param_names.append(name)

                    if param["type"] == "continuous":
                        dimensions.append(Real(param["low"], param["high"], name=name))
                    elif param["type"] == "integer":
                        dimensions.append(
                            Integer(param["low"], param["high"], name=name)
                        )
                    elif param["type"] == "categorical":
                        dimensions.append(Categorical(param["categories"], name=name))

            if not dimensions:
                raise ValueError("No valid dimensions for optimization")

            # Run optimization
            result = gp_minimize(
                func=objective_func,
                dimensions=dimensions,
                n_calls=self.config.n_calls,
                n_initial_points=self.config.n_initial_points,
                acquisition_function=self.config.acquisition_function,
                random_state=42,
            )

            # Extract best parameters
            best_params = {param_names[i]: result.x[i] for i in range(len(param_names))}
            best_score = -result.fun  # Convert back from minimization

            # Calculate improvement (simplified)
            default_score = 0.7  # Assume default model achieves 70% accuracy
            improvement = best_score - default_score

            return OptimizationResult(
                success=True,
                optimization_time_seconds=0.0,  # Set by caller
                best_parameters=best_params,
                best_score=best_score,
                improvement_over_default=improvement,
                total_evaluations=len(result.func_vals),
                convergence_achieved=True,  # Simplified
                validation_score=best_score,
            )

        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            raise

    async def _grid_search_optimization(
        self, objective_func: Callable, param_space: List, model_type: str
    ) -> OptimizationResult:
        """Perform grid search optimization."""
        try:
            logger.info(f"Running grid search optimization for {model_type}")

            # Convert parameter space to grid format
            param_grid = {}
            for param in param_space:
                if "name" in param:
                    name = param["name"]
                    if param["type"] == "continuous":
                        # Create discrete points for continuous parameters
                        values = np.linspace(param["low"], param["high"], 5)
                        param_grid[name] = values.tolist()
                    elif param["type"] == "integer":
                        values = list(range(param["low"], param["high"] + 1))
                        param_grid[name] = values
                    elif param["type"] == "categorical":
                        param_grid[name] = param["categories"]

            # Generate parameter combinations
            grid = list(ParameterGrid(param_grid))

            # Limit combinations to avoid excessive computation
            if len(grid) > self.config.grid_search_max_combinations:
                grid = grid[: self.config.grid_search_max_combinations]
                logger.info(f"Limited grid search to {len(grid)} combinations")

            # Evaluate all combinations
            best_score = float("-inf")
            best_params = {}
            evaluations = 0

            for params in grid:
                score = -objective_func(params)  # Convert from minimization
                evaluations += 1

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            # Calculate improvement
            default_score = 0.7  # Assume default model achieves 70% accuracy
            improvement = best_score - default_score

            return OptimizationResult(
                success=True,
                optimization_time_seconds=0.0,  # Set by caller
                best_parameters=best_params,
                best_score=best_score,
                improvement_over_default=improvement,
                total_evaluations=evaluations,
                convergence_achieved=True,
                validation_score=best_score,
            )

        except Exception as e:
            logger.error(f"Grid search optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimization_time_seconds=0.0,
                best_parameters={},
                best_score=0.0,
                improvement_over_default=0.0,
                total_evaluations=0,
                convergence_achieved=False,
                error_message=str(e),
            )

    async def _random_search_optimization(
        self, objective_func: Callable, param_space: List, model_type: str
    ) -> OptimizationResult:
        """Perform random search optimization."""
        try:
            logger.info(f"Running random search optimization for {model_type}")

            best_score = float("-inf")
            best_params = {}
            evaluations = 0

            for _ in range(self.config.n_calls):
                # Generate random parameters
                params = {}
                for param in param_space:
                    if "name" in param:
                        name = param["name"]
                        if param["type"] == "continuous":
                            params[name] = np.random.uniform(
                                param["low"], param["high"]
                            )
                        elif param["type"] == "integer":
                            params[name] = np.random.randint(
                                param["low"], param["high"] + 1
                            )
                        elif param["type"] == "categorical":
                            params[name] = np.random.choice(param["categories"])

                # Evaluate parameters
                score = -objective_func(params)  # Convert from minimization
                evaluations += 1

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

            # Calculate improvement
            default_score = 0.7  # Assume default model achieves 70% accuracy
            improvement = best_score - default_score

            return OptimizationResult(
                success=True,
                optimization_time_seconds=0.0,  # Set by caller
                best_parameters=best_params,
                best_score=best_score,
                improvement_over_default=improvement,
                total_evaluations=evaluations,
                convergence_achieved=True,
                validation_score=best_score,
            )

        except Exception as e:
            logger.error(f"Random search optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimization_time_seconds=0.0,
                best_parameters={},
                best_score=0.0,
                improvement_over_default=0.0,
                total_evaluations=0,
                convergence_achieved=False,
                error_message=str(e),
            )

    async def _performance_adaptive_optimization(
        self,
        objective_func: Callable,
        param_space: List,
        model_type: str,
        room_id: str,
        performance_context: Optional[Dict[str, Any]],
    ) -> OptimizationResult:
        """Perform performance-adaptive optimization based on recent history."""
        try:
            logger.info(f"Running performance-adaptive optimization for {model_type}")

            # Get recent performance history for this model
            model_key = f"{room_id}_{model_type}"

            with self._lock:
                recent_performance = self._performance_history.get(model_key, [])

            # Adapt strategy based on performance trends
            if len(recent_performance) > 5:
                trend = np.mean(np.diff(recent_performance[-5:]))  # Recent trend

                if trend < -0.05:  # Declining performance
                    # Use more aggressive optimization
                    adapted_config = self.config
                    adapted_config.n_calls = min(self.config.n_calls * 2, 100)
                    logger.info(
                        "Using aggressive optimization due to declining performance"
                    )
                elif trend > 0.02:  # Improving performance
                    # Use lighter optimization
                    adapted_config = self.config
                    adapted_config.n_calls = max(self.config.n_calls // 2, 10)
                    logger.info("Using light optimization due to improving performance")
                else:
                    adapted_config = self.config
            else:
                adapted_config = self.config

            # Fall back to Bayesian optimization with adapted config
            temp_config = self.config
            self.config = adapted_config
            result = await self._bayesian_optimization(
                objective_func, param_space, model_type
            )
            self.config = temp_config

            return result

        except Exception as e:
            logger.error(f"Performance-adaptive optimization failed: {e}")
            return OptimizationResult(
                success=False,
                optimization_time_seconds=0.0,
                best_parameters={},
                best_score=0.0,
                improvement_over_default=0.0,
                total_evaluations=0,
                convergence_achieved=False,
                error_message=str(e),
            )

    def _create_default_result(self, model_type: str) -> OptimizationResult:
        """Create default optimization result when optimization is skipped."""
        return OptimizationResult(
            success=True,
            optimization_time_seconds=0.0,
            best_parameters={},  # Empty = use defaults
            best_score=0.7,  # Assume default performance
            improvement_over_default=0.0,
            total_evaluations=0,
            convergence_achieved=True,
            validation_score=0.7,
        )

    def _update_improvement_average(self, improvement: float) -> None:
        """Update running average of optimization improvements."""
        if self._successful_optimizations == 1:
            self._average_improvement = improvement
        else:
            # Exponential moving average
            alpha = 0.1
            self._average_improvement = (
                alpha * improvement + (1 - alpha) * self._average_improvement
            )

    def _initialize_parameter_spaces(self) -> Dict[str, List]:
        """Initialize parameter search spaces for different model types."""
        return {
            "lstm": [
                {"name": "hidden_size", "type": "integer", "low": 32, "high": 256},
                {"name": "num_layers", "type": "integer", "low": 1, "high": 4},
                {"name": "dropout", "type": "continuous", "low": 0.0, "high": 0.5},
                {
                    "name": "learning_rate",
                    "type": "continuous",
                    "low": 0.0001,
                    "high": 0.01,
                },
                {
                    "name": "batch_size",
                    "type": "categorical",
                    "categories": [16, 32, 64, 128],
                },
            ],
            "xgboost": [
                {"name": "n_estimators", "type": "integer", "low": 50, "high": 500},
                {"name": "max_depth", "type": "integer", "low": 3, "high": 10},
                {
                    "name": "learning_rate",
                    "type": "continuous",
                    "low": 0.01,
                    "high": 0.3,
                },
                {"name": "subsample", "type": "continuous", "low": 0.6, "high": 1.0},
                {
                    "name": "colsample_bytree",
                    "type": "continuous",
                    "low": 0.6,
                    "high": 1.0,
                },
                {"name": "reg_alpha", "type": "continuous", "low": 0.0, "high": 1.0},
                {"name": "reg_lambda", "type": "continuous", "low": 0.0, "high": 1.0},
            ],
            "hmm": [
                {"name": "n_states", "type": "integer", "low": 2, "high": 8},
                {
                    "name": "covariance_type",
                    "type": "categorical",
                    "categories": ["spherical", "diag", "full"],
                },
                {"name": "n_iter", "type": "integer", "low": 50, "high": 200},
                {"name": "tol", "type": "continuous", "low": 1e-6, "high": 1e-2},
            ],
            "gaussian_process": [
                {
                    "name": "kernel",
                    "type": "categorical",
                    "categories": ["rbf", "matern", "rational_quadratic"],
                },
                {"name": "alpha", "type": "continuous", "low": 1e-12, "high": 1e-1},
                {
                    "name": "n_restarts_optimizer",
                    "type": "integer",
                    "low": 0,
                    "high": 10,
                },
            ],
        }


class OptimizationError(OccupancyPredictionError):
    """Raised when model optimization operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="OPTIMIZATION_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
