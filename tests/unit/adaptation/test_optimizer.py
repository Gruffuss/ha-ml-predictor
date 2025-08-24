"""
Comprehensive tests for ModelOptimizer and HyperparameterTuner.

Tests real optimization functionality including hyperparameter tuning,
model architecture optimization, and comprehensive optimization orchestration.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.optimizer import (
    ModelOptimizer,
    OptimizationResult,
    OptimizationStrategy,
)
from src.core.constants import ModelType


@pytest.fixture
def sample_optimization_data():
    """Create sample data for optimization testing."""
    np.random.seed(42)
    
    features = pd.DataFrame({
        'feature_1': np.random.normal(0, 1, 1000),
        'feature_2': np.random.exponential(2, 1000),
        'feature_3': np.random.uniform(-1, 1, 1000),
        'categorical': np.random.choice(['A', 'B', 'C'], 1000)
    })
    
    # Create synthetic target with known relationships
    targets = pd.DataFrame({
        'target': (
            0.5 * features['feature_1'] +
            0.3 * features['feature_2'] +
            0.2 * features['feature_3'] +
            np.random.normal(0, 0.1, 1000)
        )
    })
    
    return features, targets


@pytest.fixture
def mock_model():
    """Create mock model for optimization testing."""
    model = MagicMock()
    model.model_type = ModelType.XGBOOST
    model.get_params.return_value = {
        'n_estimators': 100,
        'learning_rate': 0.1,
        'max_depth': 6,
        'subsample': 1.0,
        'random_state': 42
    }
    model.set_params = MagicMock()
    model.fit = MagicMock()
    model.predict = MagicMock(return_value=np.random.random(100))
    model.score = MagicMock(return_value=0.85)
    return model


@pytest.fixture
def sample_parameter_specs():
    """Create sample parameter specifications for testing."""
    return {
        'n_estimators': ParameterSpec(
            name='n_estimators',
            type=ParameterType.INTEGER,
            range=ParameterRange(min_val=50, max_val=500),
            default=100
        ),
        'learning_rate': ParameterSpec(
            name='learning_rate',
            type=ParameterType.FLOAT,
            range=ParameterRange(min_val=0.01, max_val=0.3),
            default=0.1
        ),
        'max_depth': ParameterSpec(
            name='max_depth',
            type=ParameterType.INTEGER,
            range=ParameterRange(min_val=3, max_val=12),
            default=6
        ),
        'subsample': ParameterSpec(
            name='subsample',
            type=ParameterType.FLOAT,
            range=ParameterRange(min_val=0.5, max_val=1.0),
            default=1.0
        ),
        'algorithm': ParameterSpec(
            name='algorithm',
            type=ParameterType.CATEGORICAL,
            options=['auto', 'exact', 'approx'],
            default='auto'
        )
    }


class TestParameterSpec:
    """Test ParameterSpec functionality."""

    def test_parameter_spec_creation(self):
        """Test parameter specification creation."""
        # Integer parameter
        int_spec = ParameterSpec(
            name='n_estimators',
            type=ParameterType.INTEGER,
            range=ParameterRange(min_val=10, max_val=100),
            default=50
        )
        assert int_spec.name == 'n_estimators'
        assert int_spec.type == ParameterType.INTEGER
        assert int_spec.default == 50
        assert int_spec.range.min_val == 10
        assert int_spec.range.max_val == 100

        # Float parameter
        float_spec = ParameterSpec(
            name='learning_rate',
            type=ParameterType.FLOAT,
            range=ParameterRange(min_val=0.01, max_val=1.0),
            default=0.1,
            log_scale=True
        )
        assert float_spec.log_scale
        assert float_spec.type == ParameterType.FLOAT

        # Categorical parameter
        cat_spec = ParameterSpec(
            name='optimizer',
            type=ParameterType.CATEGORICAL,
            options=['sgd', 'adam', 'rmsprop'],
            default='adam'
        )
        assert cat_spec.options == ['sgd', 'adam', 'rmsprop']
        assert cat_spec.default == 'adam'

    def test_parameter_validation(self):
        """Test parameter value validation."""
        int_spec = ParameterSpec(
            name='test_int',
            type=ParameterType.INTEGER,
            range=ParameterRange(min_val=1, max_val=10),
            default=5
        )
        
        assert int_spec.is_valid_value(5)
        assert int_spec.is_valid_value(1)
        assert int_spec.is_valid_value(10)
        assert not int_spec.is_valid_value(0)
        assert not int_spec.is_valid_value(11)

    def test_parameter_sampling(self):
        """Test random parameter sampling."""
        float_spec = ParameterSpec(
            name='test_float',
            type=ParameterType.FLOAT,
            range=ParameterRange(min_val=0.1, max_val=1.0),
            default=0.5
        )
        
        # Sample multiple values
        samples = [float_spec.sample_value() for _ in range(100)]
        
        # All samples should be within range
        assert all(0.1 <= s <= 1.0 for s in samples)
        # Should have variation
        assert len(set(samples)) > 10

    def test_log_scale_sampling(self):
        """Test log scale parameter sampling."""
        log_spec = ParameterSpec(
            name='learning_rate',
            type=ParameterType.FLOAT,
            range=ParameterRange(min_val=0.001, max_val=1.0),
            default=0.1,
            log_scale=True
        )
        
        samples = [log_spec.sample_value() for _ in range(100)]
        
        # All should be in range
        assert all(0.001 <= s <= 1.0 for s in samples)
        # Should favor smaller values due to log scale
        small_values = sum(1 for s in samples if s < 0.1)
        large_values = sum(1 for s in samples if s > 0.1)
        assert small_values > large_values  # Should be biased toward smaller values


class TestParameterGrid:
    """Test ParameterGrid functionality."""

    def test_grid_creation(self, sample_parameter_specs):
        """Test parameter grid creation."""
        grid = ParameterGrid(sample_parameter_specs)
        
        assert len(grid.parameters) == len(sample_parameter_specs)
        assert 'n_estimators' in grid.parameters
        assert 'learning_rate' in grid.parameters

    def test_grid_search_generation(self, sample_parameter_specs):
        """Test grid search parameter combinations generation."""
        grid = ParameterGrid(sample_parameter_specs)
        
        combinations = grid.generate_grid_search_combinations(
            n_estimators=[50, 100, 200],
            learning_rate=[0.05, 0.1, 0.2],
            max_depth=[4, 6, 8]
        )
        
        # Should have 3 * 3 * 3 = 27 combinations
        assert len(combinations) == 27
        
        # Each combination should have the specified parameters
        for combo in combinations:
            assert combo['n_estimators'] in [50, 100, 200]
            assert combo['learning_rate'] in [0.05, 0.1, 0.2]
            assert combo['max_depth'] in [4, 6, 8]

    def test_random_search_generation(self, sample_parameter_specs):
        """Test random search parameter combinations generation."""
        grid = ParameterGrid(sample_parameter_specs)
        
        combinations = grid.generate_random_search_combinations(n_combinations=50)
        
        assert len(combinations) == 50
        
        # Each combination should have valid parameter values
        for combo in combinations:
            for param_name, value in combo.items():
                if param_name in grid.parameters:
                    spec = grid.parameters[param_name]
                    assert spec.is_valid_value(value)

    def test_bayesian_optimization_suggestions(self, sample_parameter_specs):
        """Test Bayesian optimization parameter suggestions."""
        grid = ParameterGrid(sample_parameter_specs)
        
        # Mock previous evaluations
        previous_params = [
            {'n_estimators': 100, 'learning_rate': 0.1, 'max_depth': 6},
            {'n_estimators': 200, 'learning_rate': 0.05, 'max_depth': 8},
        ]
        previous_scores = [0.82, 0.85]
        
        suggestions = grid.suggest_bayesian_parameters(
            previous_params, previous_scores, n_suggestions=3
        )
        
        assert len(suggestions) == 3
        
        # Each suggestion should have valid parameters
        for suggestion in suggestions:
            for param_name, value in suggestion.items():
                if param_name in grid.parameters:
                    spec = grid.parameters[param_name]
                    assert spec.is_valid_value(value)


class TestHyperparameterOptimizer:
    """Test HyperparameterOptimizer functionality."""

    def test_initialization(self):
        """Test hyperparameter optimizer initialization."""
        optimizer = HyperparameterOptimizer()
        
        assert optimizer.default_method == TuningMethod.RANDOM_SEARCH
        assert optimizer.max_evaluations == 100
        assert optimizer.cv_folds == 5
        assert optimizer.scoring_metric == 'accuracy'

        # Test custom configuration
        optimizer = HyperparameterOptimizer(
            default_method=TuningMethod.BAYESIAN,
            max_evaluations=50,
            cv_folds=3,
            scoring_metric='f1',
            early_stopping_rounds=10
        )
        assert optimizer.default_method == TuningMethod.BAYESIAN
        assert optimizer.max_evaluations == 50
        assert optimizer.cv_folds == 3
        assert optimizer.scoring_metric == 'f1'
        assert optimizer.early_stopping_rounds == 10

    @pytest.mark.asyncio
    async def test_grid_search_optimization(self, mock_model, sample_optimization_data, sample_parameter_specs):
        """Test grid search optimization."""
        optimizer = HyperparameterOptimizer()
        features, targets = sample_optimization_data
        
        # Mock model scoring to return different values for different parameters
        def mock_score(*args, **kwargs):
            params = mock_model.get_params.return_value
            # Higher learning rate = better score (for testing)
            return 0.8 + 0.1 * params.get('learning_rate', 0.1)
        
        mock_model.score.side_effect = mock_score
        
        result = await optimizer.optimize_hyperparameters(
            model=mock_model,
            parameter_grid=ParameterGrid(sample_parameter_specs),
            training_features=features,
            training_targets=targets,
            method=TuningMethod.GRID_SEARCH,
            max_evaluations=27  # 3^3 combinations
        )
        
        assert isinstance(result, OptimizationResult)
        assert result.success
        assert result.method == TuningMethod.GRID_SEARCH
        assert result.best_score > 0.8
        assert len(result.evaluation_history) > 0
        assert result.best_parameters is not None

    @pytest.mark.asyncio
    async def test_random_search_optimization(self, mock_model, sample_optimization_data, sample_parameter_specs):
        """Test random search optimization."""
        optimizer = HyperparameterOptimizer()
        features, targets = sample_optimization_data
        
        # Mock progressive improvement
        evaluation_count = [0]
        
        def mock_score(*args, **kwargs):
            evaluation_count[0] += 1
            # Simulate improvement over iterations
            base_score = 0.75
            improvement = min(0.15, evaluation_count[0] * 0.01)
            return base_score + improvement
        
        mock_model.score.side_effect = mock_score
        
        result = await optimizer.optimize_hyperparameters(
            model=mock_model,
            parameter_grid=ParameterGrid(sample_parameter_specs),
            training_features=features,
            training_targets=targets,
            method=TuningMethod.RANDOM_SEARCH,
            max_evaluations=20
        )
        
        assert result.success
        assert result.method == TuningMethod.RANDOM_SEARCH
        assert len(result.evaluation_history) <= 20
        assert result.best_score > 0.75

    @pytest.mark.asyncio
    async def test_bayesian_optimization(self, mock_model, sample_optimization_data, sample_parameter_specs):
        """Test Bayesian optimization."""
        optimizer = HyperparameterOptimizer()
        features, targets = sample_optimization_data
        
        # Mock Bayesian-like improvement (later evaluations should be better)
        evaluation_scores = []
        
        def mock_score(*args, **kwargs):
            # Simulate Bayesian improvement: later evaluations generally better
            score = 0.7 + len(evaluation_scores) * 0.02 + np.random.random() * 0.05
            evaluation_scores.append(score)
            return score
        
        mock_model.score.side_effect = mock_score
        
        result = await optimizer.optimize_hyperparameters(
            model=mock_model,
            parameter_grid=ParameterGrid(sample_parameter_specs),
            training_features=features,
            training_targets=targets,
            method=TuningMethod.BAYESIAN,
            max_evaluations=15
        )
        
        assert result.success
        assert result.method == TuningMethod.BAYESIAN
        assert len(result.evaluation_history) <= 15
        # Later evaluations should generally be better
        assert result.best_score > min(evaluation_scores)

    @pytest.mark.asyncio
    async def test_early_stopping(self, mock_model, sample_optimization_data, sample_parameter_specs):
        """Test early stopping during optimization."""
        optimizer = HyperparameterOptimizer(early_stopping_rounds=5, early_stopping_patience=0.001)
        features, targets = sample_optimization_data
        
        # Mock convergence (no improvement after a few evaluations)
        best_score = 0.85
        
        def mock_score(*args, **kwargs):
            # Return best score + small noise (no real improvement)
            return best_score + np.random.normal(0, 0.0005)
        
        mock_model.score.side_effect = mock_score
        
        result = await optimizer.optimize_hyperparameters(
            model=mock_model,
            parameter_grid=ParameterGrid(sample_parameter_specs),
            training_features=features,
            training_targets=targets,
            method=TuningMethod.RANDOM_SEARCH,
            max_evaluations=50  # Should stop early
        )
        
        assert result.success
        assert result.early_stopping_triggered
        assert len(result.evaluation_history) < 50  # Stopped before max evaluations

    def test_cross_validation_scoring(self, mock_model, sample_optimization_data):
        """Test cross-validation scoring during optimization."""
        optimizer = HyperparameterOptimizer(cv_folds=3)
        features, targets = sample_optimization_data
        
        # Mock cross-validation scores
        cv_scores = [0.80, 0.85, 0.82]
        
        with patch('sklearn.model_selection.cross_val_score') as mock_cv:
            mock_cv.return_value = np.array(cv_scores)
            
            score = optimizer.evaluate_parameters(
                mock_model, {'n_estimators': 100}, features, targets
            )
            
            # Should return mean of CV scores
            expected_score = np.mean(cv_scores)
            assert abs(score - expected_score) < 1e-10


class TestModelOptimizer:
    """Test ModelOptimizer functionality."""

    def test_initialization(self):
        """Test model optimizer initialization."""
        optimizer = ModelOptimizer()
        
        assert optimizer.optimization_strategies is not None
        assert OptimizationStrategy.HYPERPARAMETER_TUNING in optimizer.optimization_strategies
        assert OptimizationStrategy.ARCHITECTURE_SEARCH in optimizer.optimization_strategies

    @pytest.mark.asyncio
    async def test_model_optimization_workflow(self, mock_model, sample_optimization_data):
        """Test complete model optimization workflow."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Mock hyperparameter optimizer
        mock_hp_optimizer = MagicMock(spec=HyperparameterOptimizer)
        mock_hp_result = OptimizationResult(
            success=True,
            best_score=0.88,
            best_parameters={'n_estimators': 150, 'learning_rate': 0.08},
            method=TuningMethod.RANDOM_SEARCH,
            evaluations_performed=25,
            optimization_time=120.0,
            evaluation_history=[]
        )
        mock_hp_optimizer.optimize_hyperparameters = AsyncMock(return_value=mock_hp_result)
        
        with patch.object(optimizer, '_create_hyperparameter_optimizer', return_value=mock_hp_optimizer):
            result = await optimizer.optimize_model(
                model=mock_model,
                training_features=features,
                training_targets=targets,
                room_id="test_room",
                strategies=[OptimizationStrategy.HYPERPARAMETER_TUNING]
            )
            
            assert isinstance(result, OptimizationResult)
            assert result.success
            assert result.best_score == 0.88
            assert 'n_estimators' in result.best_parameters

    @pytest.mark.asyncio
    async def test_architecture_optimization(self, mock_model, sample_optimization_data):
        """Test neural network architecture optimization."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Mock model with architecture parameters
        mock_model.model_type = ModelType.LSTM
        mock_model.get_params.return_value = {
            'hidden_units': 64,
            'num_layers': 2,
            'dropout_rate': 0.2
        }
        
        # Mock architecture evaluation
        def mock_score(*args, **kwargs):
            params = mock_model.get_params.return_value
            # More hidden units = better score (for testing)
            return 0.7 + (params.get('hidden_units', 64) / 200)
        
        mock_model.score.side_effect = mock_score
        
        result = await optimizer.optimize_model(
            model=mock_model,
            training_features=features,
            training_targets=targets,
            room_id="architecture_test",
            strategies=[OptimizationStrategy.ARCHITECTURE_SEARCH]
        )
        
        assert result.success
        assert result.best_score > 0.7

    @pytest.mark.asyncio
    async def test_ensemble_optimization(self, sample_optimization_data):
        """Test ensemble model optimization."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Create mock ensemble model
        mock_ensemble = MagicMock()
        mock_ensemble.model_type = ModelType.ENSEMBLE
        mock_ensemble.get_params.return_value = {
            'base_models': ['lstm', 'xgboost', 'hmm'],
            'meta_learner': 'linear',
            'stacking_cv': 5
        }
        mock_ensemble.score.return_value = 0.92  # Ensemble should perform well
        mock_ensemble.set_params = MagicMock()
        mock_ensemble.fit = MagicMock()
        
        result = await optimizer.optimize_model(
            model=mock_ensemble,
            training_features=features,
            training_targets=targets,
            room_id="ensemble_test",
            strategies=[OptimizationStrategy.ENSEMBLE_TUNING]
        )
        
        assert result.success
        assert result.best_score >= 0.9  # Ensemble should achieve high score

    @pytest.mark.asyncio
    async def test_feature_selection_optimization(self, mock_model, sample_optimization_data):
        """Test feature selection optimization."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Mock feature importance
        feature_importance = {
            'feature_1': 0.4,
            'feature_2': 0.3,
            'feature_3': 0.2,
            'categorical': 0.1
        }
        
        with patch.object(optimizer, '_get_feature_importance', return_value=feature_importance):
            result = await optimizer.optimize_model(
                model=mock_model,
                training_features=features,
                training_targets=targets,
                room_id="feature_test",
                strategies=[OptimizationStrategy.FEATURE_SELECTION]
            )
            
            assert result.success
            assert 'selected_features' in result.optimization_details
            # Should select most important features
            selected_features = result.optimization_details['selected_features']
            assert 'feature_1' in selected_features
            assert 'feature_2' in selected_features

    def test_optimization_metrics_calculation(self):
        """Test optimization metrics calculation."""
        metrics = OptimizationMetrics(
            initial_score=0.75,
            optimized_score=0.88,
            improvement_percent=17.3,
            evaluations_performed=45,
            optimization_time_minutes=8.5,
            best_parameters={'learning_rate': 0.05, 'n_estimators': 200},
            convergence_achieved=True,
            early_stopping_triggered=False
        )
        
        assert metrics.score_improvement == 0.13
        assert metrics.relative_improvement_percent == 17.3
        assert metrics.optimization_efficiency > 0  # Improvement per evaluation
        assert metrics.convergence_achieved

    def test_optimization_result_serialization(self):
        """Test optimization result serialization."""
        result = OptimizationResult(
            success=True,
            best_score=0.91,
            best_parameters={'param1': 0.1, 'param2': 100},
            method=TuningMethod.BAYESIAN,
            evaluations_performed=30,
            optimization_time=240.5,
            evaluation_history=[
                {'score': 0.85, 'parameters': {'param1': 0.2, 'param2': 50}},
                {'score': 0.91, 'parameters': {'param1': 0.1, 'param2': 100}}
            ],
            early_stopping_triggered=False,
            optimization_details={'convergence_curve': [0.8, 0.85, 0.91]}
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['success'] is True
        assert result_dict['best_score'] == 0.91
        assert 'evaluation_history' in result_dict
        assert 'optimization_details' in result_dict

    @pytest.mark.asyncio
    async def test_optimization_error_handling(self, mock_model, sample_optimization_data):
        """Test optimization error handling."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Mock model that raises an error
        mock_model.score.side_effect = Exception("Model evaluation failed")
        
        result = await optimizer.optimize_model(
            model=mock_model,
            training_features=features,
            training_targets=targets,
            room_id="error_test",
            strategies=[OptimizationStrategy.HYPERPARAMETER_TUNING]
        )
        
        assert not result.success
        assert "Model evaluation failed" in result.error_message

    def test_optimization_callback_system(self):
        """Test optimization progress callback system."""
        optimizer = ModelOptimizer()
        
        callback_calls = []
        
        def progress_callback(iteration, current_best_score, parameters):
            callback_calls.append((iteration, current_best_score, parameters))
        
        optimizer.add_progress_callback(progress_callback)
        
        # Simulate calling the callback
        optimizer._notify_progress_callbacks(5, 0.85, {'param': 0.1})
        
        assert len(callback_calls) == 1
        assert callback_calls[0] == (5, 0.85, {'param': 0.1})


class TestOptimizationIntegration:
    """Test integration between optimization components."""

    @pytest.mark.asyncio
    async def test_multi_strategy_optimization(self, mock_model, sample_optimization_data):
        """Test optimization with multiple strategies."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Test combining hyperparameter tuning and feature selection
        strategies = [
            OptimizationStrategy.HYPERPARAMETER_TUNING,
            OptimizationStrategy.FEATURE_SELECTION
        ]
        
        with patch.object(optimizer, '_optimize_hyperparameters') as mock_hp:
            with patch.object(optimizer, '_optimize_feature_selection') as mock_fs:
                mock_hp.return_value = OptimizationResult(
                    success=True, best_score=0.85, best_parameters={'param1': 0.1}
                )
                mock_fs.return_value = OptimizationResult(
                    success=True, best_score=0.88, best_parameters={'selected_features': ['f1', 'f2']}
                )
                
                result = await optimizer.optimize_model(
                    model=mock_model,
                    training_features=features,
                    training_targets=targets,
                    room_id="multi_strategy_test",
                    strategies=strategies
                )
                
                assert result.success
                # Should use best result from all strategies
                assert result.best_score == 0.88

    @pytest.mark.asyncio
    async def test_optimization_with_validation_data(self, mock_model, sample_optimization_data):
        """Test optimization with separate validation data."""
        optimizer = ModelOptimizer()
        features, targets = sample_optimization_data
        
        # Split data into train/validation
        split_idx = len(features) // 2
        train_features, val_features = features[:split_idx], features[split_idx:]
        train_targets, val_targets = targets[:split_idx], targets[split_idx:]
        
        result = await optimizer.optimize_model(
            model=mock_model,
            training_features=train_features,
            training_targets=train_targets,
            validation_features=val_features,
            validation_targets=val_targets,
            room_id="validation_test"
        )
        
        assert result.success
        # Should have used validation data for evaluation
        assert len(result.evaluation_history) > 0

    def test_parameter_importance_analysis(self, sample_parameter_specs):
        """Test parameter importance analysis."""
        optimizer = HyperparameterOptimizer()
        
        # Mock evaluation history
        evaluation_history = [
            {'score': 0.80, 'parameters': {'n_estimators': 100, 'learning_rate': 0.1}},
            {'score': 0.85, 'parameters': {'n_estimators': 200, 'learning_rate': 0.1}},
            {'score': 0.82, 'parameters': {'n_estimators': 100, 'learning_rate': 0.05}},
            {'score': 0.88, 'parameters': {'n_estimators': 200, 'learning_rate': 0.05}},
        ]
        
        importance = optimizer.analyze_parameter_importance(evaluation_history)
        
        assert isinstance(importance, dict)
        assert 'n_estimators' in importance
        assert 'learning_rate' in importance
        # Both parameters should have some importance
        assert all(imp > 0 for imp in importance.values())

    @pytest.mark.asyncio
    async def test_adaptive_optimization_budget(self, mock_model, sample_optimization_data):
        """Test adaptive optimization budget allocation."""
        optimizer = ModelOptimizer(adaptive_budget=True)
        features, targets = sample_optimization_data
        
        # Mock rapid convergence
        evaluation_count = [0]
        
        def mock_score(*args, **kwargs):
            evaluation_count[0] += 1
            if evaluation_count[0] < 5:
                return 0.7 + evaluation_count[0] * 0.03
            else:
                return 0.82 + np.random.normal(0, 0.001)  # Converged
        
        mock_model.score.side_effect = mock_score
        
        result = await optimizer.optimize_model(
            model=mock_model,
            training_features=features,
            training_targets=targets,
            room_id="adaptive_budget_test",
            max_optimization_time_minutes=10
        )
        
        assert result.success
        # Should have stopped early due to convergence
        assert result.evaluations_performed < 50  # Less than max possible


if __name__ == "__main__":
    pytest.main([__file__])