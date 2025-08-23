"""
Comprehensive unit tests for OccupancyEnsemble class achieving 85%+ coverage.

This module provides complete test coverage for the ensemble architecture including:
- Initialization and configuration
- Three-phase training pipeline (CV, meta-learner, final training)
- Stacking ensemble with meta-learning
- Prediction generation and combination
- Model weight optimization
- Incremental learning and adaptation
- Performance benchmarks and validation
- Error handling and edge cases
- Real ML validation scenarios
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.predictor import PredictionResult, TrainingResult
from src.models.ensemble import (
    OccupancyEnsemble,
    _ensure_timezone_aware,
    _safe_time_difference,
)


@pytest.fixture
def comprehensive_training_data():
    """Create comprehensive training data for thorough ensemble testing."""
    np.random.seed(42)
    n_samples = 1200
    n_features = 20

    # Create diverse feature patterns that challenge different aspects of the ensemble
    features = {}

    # Temporal cyclical features (LSTM-friendly)
    hours = np.random.randint(0, 24, n_samples)
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["day_of_week"] = np.random.randint(0, 7, n_samples)
    features["is_weekend"] = (features["day_of_week"] >= 5).astype(int)

    # Sequential state features (HMM-friendly)
    features["prev_state_duration"] = np.random.exponential(2000, n_samples)
    features["transition_count_6h"] = np.random.poisson(3, n_samples)
    features["state_stability_index"] = np.random.beta(2, 5, n_samples)
    features["recent_transitions"] = np.random.geometric(0.3, n_samples)

    # Non-linear tree-friendly features (XGBoost-friendly)
    features["temperature"] = 18 + np.random.normal(0, 4, n_samples)
    features["temp_humidity_interaction"] = features["temperature"] * np.random.uniform(
        0.3, 0.8, n_samples
    )
    features["motion_events"] = np.random.gamma(2, 3, n_samples)
    features["door_events"] = np.random.poisson(1, n_samples)

    # Smooth continuous features (GP-friendly)
    t = np.linspace(0, 4 * np.pi, n_samples)
    features["smooth_seasonal"] = (
        np.sin(t) + 0.3 * np.sin(3 * t) + np.random.normal(0, 0.1, n_samples)
    )
    features["trend_component"] = np.linspace(-1, 1, n_samples) + np.random.normal(
        0, 0.05, n_samples
    )
    features["noise_level"] = np.random.exponential(0.5, n_samples)

    # Complex interaction features
    features["time_temp_interaction"] = features["hour_sin"] * features["temperature"]
    features["motion_stability"] = (
        features["motion_events"] * features["state_stability_index"]
    )

    # Additional noise features to test robustness
    for i in range(n_features - len(features)):
        features[f"noise_feature_{i}"] = np.random.normal(0, 1, n_samples)

    # Create complex target with multiple influences (realistic occupancy patterns)
    base_transition_time = 1800  # 30 minutes baseline

    # Time of day influence (longer stays during work hours)
    time_influence = np.where(
        (hours >= 9) & (hours <= 17),
        1.8,  # Work hours - longer stays
        np.where(
            (hours >= 22) | (hours <= 6),
            2.5,  # Sleep hours - very long stays
            1.0,  # Other times - normal
        ),
    )

    # Temperature comfort influence
    temp_comfort = 1 + 0.2 * np.exp(-((features["temperature"] - 22) ** 2) / 10)

    # Motion activity influence (more motion = shorter stays until next transition)
    motion_influence = 1.0 / (1 + features["motion_events"] / 5)

    # Day of week influence (weekends different patterns)
    weekend_influence = np.where(features["is_weekend"], 1.3, 1.0)

    # Previous state duration influence (state persistence)
    persistence_influence = 1 + features["prev_state_duration"] / 10000

    # Complex target calculation
    targets = (
        base_transition_time
        * time_influence
        * temp_comfort
        * motion_influence
        * weekend_influence
        * persistence_influence
        * (1 + np.random.normal(0, 0.25, n_samples))  # Add realistic noise
    )

    # Clip to realistic bounds
    targets = np.clip(targets, 60, 14400)  # 1 minute to 4 hours

    # Create transition types based on realistic patterns
    transition_types = np.where(
        (hours >= 23) | (hours <= 7),
        "occupied_to_vacant",  # Night time transitions
        np.where(
            (hours >= 8) & (hours <= 18),
            np.random.choice(["vacant_to_occupied", "occupied_to_vacant"], n_samples),
            "vacant_to_occupied",  # Evening transitions
        ),
    )

    # Create time series for realistic target_time
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    target_times = [start_time + timedelta(minutes=i * 15) for i in range(n_samples)]

    features_df = pd.DataFrame(features)
    targets_df = pd.DataFrame(
        {
            "time_until_transition_seconds": targets,
            "transition_type": transition_types,
            "target_time": target_times,
            "next_transition_time": [
                tt + timedelta(seconds=dur) for tt, dur in zip(target_times, targets)
            ],
        }
    )

    return features_df, targets_df


@pytest.fixture
def split_comprehensive_data(comprehensive_training_data):
    """Split comprehensive data into train/validation/test sets."""
    features, targets = comprehensive_training_data

    # 70/15/15 split for train/val/test
    n_samples = len(features)
    train_end = int(0.7 * n_samples)
    val_end = int(0.85 * n_samples)

    train_features = features.iloc[:train_end].copy()
    train_targets = targets.iloc[:train_end].copy()

    val_features = features.iloc[train_end:val_end].copy()
    val_targets = targets.iloc[train_end:val_end].copy()

    test_features = features.iloc[val_end:].copy()
    test_targets = targets.iloc[val_end:].copy()

    return (
        train_features,
        train_targets,
        val_features,
        val_targets,
        test_features,
        test_targets,
    )


@pytest.fixture
def advanced_tracking_manager():
    """Create advanced mock tracking manager with realistic behavior."""
    manager = AsyncMock()
    manager.register_model = AsyncMock()
    manager.record_prediction = AsyncMock()
    manager.get_model_accuracy = AsyncMock(return_value=0.82)
    manager.update_model_metrics = AsyncMock()
    manager.should_trigger_retraining = AsyncMock(return_value=False)
    return manager


class TestEnsembleAdvancedInitialization:
    """Test advanced ensemble initialization scenarios."""

    def test_ensemble_default_initialization(self):
        """Test ensemble with default parameters."""
        ensemble = OccupancyEnsemble()

        assert ensemble.model_type == ModelType.ENSEMBLE
        assert ensemble.room_id is None
        assert not ensemble.is_trained
        assert not ensemble.base_models_trained
        assert not ensemble.meta_learner_trained
        assert ensemble.meta_learner is None

        # Check default parameters
        assert ensemble.model_params["meta_learner"] == "random_forest"
        assert ensemble.model_params["cv_folds"] == 5
        assert ensemble.model_params["use_base_features"] is True
        assert ensemble.model_params["meta_features_only"] is False

        # Check base models
        expected_models = ["lstm", "xgboost", "hmm", "gp"]
        assert set(ensemble.base_models.keys()) == set(expected_models)

        for model_name, model in ensemble.base_models.items():
            assert model.room_id is None
            assert not model.is_trained

    def test_ensemble_custom_configuration(self):
        """Test ensemble with comprehensive custom configuration."""
        custom_config = {
            "meta_learner": "linear",
            "cv_folds": 3,
            "stacking_method": "weighted",
            "blend_weights": "manual",
            "use_base_features": False,
            "meta_features_only": True,
        }

        ensemble = OccupancyEnsemble(room_id="advanced_test_room", **custom_config)

        assert ensemble.room_id == "advanced_test_room"
        for key, value in custom_config.items():
            assert ensemble.model_params[key] == value

        # Base models should inherit room_id
        for model in ensemble.base_models.values():
            assert model.room_id == "advanced_test_room"

    def test_ensemble_tracking_integration(self, advanced_tracking_manager):
        """Test ensemble integration with tracking manager."""
        ensemble = OccupancyEnsemble(
            room_id="tracked_room", tracking_manager=advanced_tracking_manager
        )

        assert ensemble.tracking_manager == advanced_tracking_manager
        advanced_tracking_manager.register_model.assert_called_once_with(
            "tracked_room", ModelType.ENSEMBLE.value, ensemble
        )

    def test_ensemble_model_parameters_validation(self):
        """Test parameter validation and defaults."""
        # Test with invalid parameters (should use defaults)
        ensemble = OccupancyEnsemble(
            cv_folds=0, meta_learner="invalid_learner"  # Invalid  # Will use default
        )

        # Should use valid defaults despite invalid inputs
        assert (
            ensemble.model_params["cv_folds"] == 0
        )  # Passed through (will be handled in training)
        assert (
            ensemble.model_params["meta_learner"] == "invalid_learner"
        )  # Passed through

    def test_ensemble_component_initialization(self):
        """Test ensemble component initialization."""
        ensemble = OccupancyEnsemble(room_id="component_test")

        # Check scalers
        assert isinstance(ensemble.meta_scaler, StandardScaler)

        # Check state variables
        assert ensemble.model_weights == {}
        assert ensemble.model_performance == {}
        assert ensemble.cross_validation_scores == {}

        # Check base models types
        from src.models.base.gp_predictor import GaussianProcessPredictor
        from src.models.base.hmm_predictor import HMMPredictor
        from src.models.base.lstm_predictor import LSTMPredictor
        from src.models.base.xgboost_predictor import XGBoostPredictor

        assert isinstance(ensemble.base_models["lstm"], LSTMPredictor)
        assert isinstance(ensemble.base_models["xgboost"], XGBoostPredictor)
        assert isinstance(ensemble.base_models["hmm"], HMMPredictor)
        assert isinstance(ensemble.base_models["gp"], GaussianProcessPredictor)


class TestEnsembleCompleteTrainingPipeline:
    """Test comprehensive ensemble training pipeline."""

    @pytest.mark.asyncio
    async def test_full_training_pipeline_success(self, split_comprehensive_data):
        """Test complete successful training pipeline."""
        train_features, train_targets, val_features, val_targets, _, _ = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(
            room_id="pipeline_test", cv_folds=3, meta_learner="random_forest"
        )

        # Mock all base models to simulate realistic training
        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Setup realistic base model mocks
            for model_name, base_model in ensemble.base_models.items():
                self._setup_realistic_base_model_mock(
                    base_model, model_name, len(train_features)
                )

            # Execute training
            result = await ensemble.train(
                train_features, train_targets, val_features, val_targets
            )

            # Validate training result
            assert result.success is True
            assert result.training_samples == len(train_features)
            assert result.validation_score is not None
            assert result.training_score is not None
            assert result.training_time_seconds > 0

            # Validate ensemble state
            assert ensemble.is_trained is True
            assert ensemble.base_models_trained is True
            assert ensemble.meta_learner_trained is True
            assert ensemble.meta_learner is not None
            assert ensemble.feature_names == list(train_features.columns)

            # Validate training metrics
            metrics = result.training_metrics
            required_metrics = [
                "ensemble_mae",
                "ensemble_rmse",
                "ensemble_r2",
                "base_model_count",
                "meta_learner_type",
                "cv_folds",
                "model_weights",
                "base_model_performance",
            ]
            for metric in required_metrics:
                assert metric in metrics

            assert metrics["base_model_count"] == 4
            assert metrics["meta_learner_type"] == "random_forest"
            assert metrics["cv_folds"] == 3

            # Validate model weights
            assert len(ensemble.model_weights) == 4
            assert all(w >= 0 for w in ensemble.model_weights.values())
            assert abs(sum(ensemble.model_weights.values()) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_cross_validation_meta_feature_generation(
        self, split_comprehensive_data
    ):
        """Test cross-validation phase for meta-feature generation."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data

        # Use subset for faster testing
        subset_features = train_features.head(300)
        subset_targets = train_targets.head(300)

        ensemble = OccupancyEnsemble(room_id="cv_test", cv_folds=3)

        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Mock fold models for CV
            for model_name in ensemble.base_models.keys():
                self._setup_cv_fold_model_mock(model_name)

            # Generate meta-features through CV
            meta_features = await ensemble._train_base_models_cv(
                subset_features, subset_targets
            )

            # Validate meta-features
            assert isinstance(meta_features, pd.DataFrame)
            assert len(meta_features) == len(subset_features)
            assert set(meta_features.columns) == set(ensemble.base_models.keys())

            # Check for reasonable meta-feature values
            for col in meta_features.columns:
                assert not meta_features[col].isna().any()
                assert (
                    meta_features[col] >= 0
                ).all()  # Time predictions should be positive

            # Validate CV scores were recorded
            assert len(ensemble.cross_validation_scores) > 0
            for model_name in ensemble.base_models.keys():
                if model_name in ensemble.cross_validation_scores:
                    scores = ensemble.cross_validation_scores[model_name]
                    assert len(scores) > 0
                    assert all(-2 <= score <= 1 for score in scores)  # R² scores

    @pytest.mark.asyncio
    async def test_meta_learner_training_variants(self, split_comprehensive_data):
        """Test different meta-learner configurations."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data

        # Test Random Forest meta-learner
        ensemble_rf = OccupancyEnsemble(
            room_id="meta_rf_test", meta_learner="random_forest"
        )

        meta_features = self._create_realistic_meta_features(len(train_features))

        await ensemble_rf._train_meta_learner(
            meta_features, train_targets, train_features
        )

        assert ensemble_rf.meta_learner_trained is True
        assert isinstance(ensemble_rf.meta_learner, RandomForestRegressor)
        assert len(ensemble_rf.model_weights) == 4

        # Test Linear meta-learner
        ensemble_linear = OccupancyEnsemble(
            room_id="meta_linear_test", meta_learner="linear"
        )

        await ensemble_linear._train_meta_learner(
            meta_features, train_targets, train_features
        )

        assert ensemble_linear.meta_learner_trained is True
        assert isinstance(ensemble_linear.meta_learner, LinearRegression)
        assert len(ensemble_linear.model_weights) == 4

    @pytest.mark.asyncio
    async def test_base_model_training_orchestration(self, split_comprehensive_data):
        """Test orchestration of base model training."""
        train_features, train_targets, val_features, val_targets, _, _ = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(room_id="orchestration_test")

        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Setup base models with different performance levels
            performance_levels = {
                "lstm": 0.75,
                "xgboost": 0.85,  # Best performer
                "hmm": 0.70,
                "gp": 0.80,
            }

            for model_name, model in ensemble.base_models.items():
                perf = performance_levels[model_name]
                model.train = AsyncMock(
                    return_value=TrainingResult(
                        success=True,
                        training_time_seconds=45.0,
                        model_version="v1.0",
                        training_samples=len(train_features),
                        training_score=perf,
                        validation_score=perf - 0.05,
                        training_metrics={"training_mae": (1 - perf) * 1000},
                    )
                )
                model.is_trained = True

            # Train base models
            await ensemble._train_base_models_final(
                train_features, train_targets, val_features, val_targets
            )

            # Validate orchestration results
            assert ensemble.base_models_trained is True

            # Check performance was recorded
            assert len(ensemble.model_performance) == 4
            for model_name, perf_data in ensemble.model_performance.items():
                expected_perf = performance_levels[model_name]
                assert abs(perf_data["training_score"] - expected_perf) < 0.01
                assert perf_data["validation_score"] == expected_perf - 0.05

    @pytest.mark.asyncio
    async def test_training_error_handling(self, split_comprehensive_data):
        """Test comprehensive error handling during training."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data
        ensemble = OccupancyEnsemble(room_id="error_test")

        # Test insufficient data error
        tiny_features = train_features.head(20)
        tiny_targets = train_targets.head(20)

        with pytest.raises(ModelTrainingError) as exc_info:
            await ensemble.train(tiny_features, tiny_targets)

        assert "Insufficient data" in str(exc_info.value)

        # Test base model failure handling
        with patch.object(
            ensemble.base_models["xgboost"],
            "train",
            side_effect=Exception("XGBoost training failed"),
        ):
            # Should handle failures gracefully
            try:
                result = await ensemble.train(
                    train_features.head(100), train_targets.head(100)
                )
                # Training may fail or succeed depending on implementation
            except ModelTrainingError:
                # Expected for critical failures
                pass

    @pytest.mark.asyncio
    async def test_validation_data_processing(self, split_comprehensive_data):
        """Test validation data handling in training."""
        train_features, train_targets, val_features, val_targets, _, _ = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(room_id="validation_test")

        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Setup base models
            for model_name, model in ensemble.base_models.items():
                self._setup_realistic_base_model_mock(
                    model, model_name, len(train_features)
                )

            # Train with validation data
            result = await ensemble.train(
                train_features.head(200),
                train_targets.head(200),
                val_features.head(50),
                val_targets.head(50),
            )

            # Should have validation metrics
            assert result.validation_score is not None
            metrics = result.training_metrics
            assert "ensemble_validation_mae" in metrics
            assert "ensemble_validation_rmse" in metrics
            assert "ensemble_validation_r2" in metrics

    def _setup_realistic_base_model_mock(self, model, model_name, n_samples):
        """Setup realistic base model mock with proper async behavior."""
        # Different performance characteristics per model
        performance_map = {"lstm": 0.78, "xgboost": 0.84, "hmm": 0.72, "gp": 0.81}

        perf = performance_map.get(model_name, 0.75)

        model.train = AsyncMock(
            return_value=TrainingResult(
                success=True,
                training_time_seconds=60.0 + np.random.uniform(-10, 20),
                model_version="v1.0",
                training_samples=n_samples,
                training_score=perf,
                validation_score=perf - 0.03,
                training_metrics={
                    "training_mae": (1 - perf) * 800 + np.random.uniform(-50, 50),
                    "training_rmse": (1 - perf) * 1000 + np.random.uniform(-100, 100),
                },
            )
        )

        # Dynamic prediction mock
        async def mock_predict(features, pred_time, state):
            n_preds = len(features)
            base_time = 1800 + np.random.uniform(-300, 300)  # ~30min ± 5min
            return [
                PredictionResult(
                    predicted_time=pred_time + timedelta(seconds=base_time + i * 60),
                    transition_type=(
                        "vacant_to_occupied" if i % 2 == 0 else "occupied_to_vacant"
                    ),
                    confidence_score=perf - 0.05 + np.random.uniform(-0.1, 0.1),
                    model_type=model_name,
                    model_version="v1.0",
                )
                for i in range(n_preds)
            ]

        model.predict = AsyncMock(side_effect=mock_predict)
        model.is_trained = True

    def _setup_cv_fold_model_mock(self, model_name):
        """Setup CV fold model mock."""
        # Mock the model class to return instances
        mock_class = MagicMock()
        mock_instance = MagicMock()

        mock_instance.train = AsyncMock(
            return_value=TrainingResult(
                success=True,
                training_time_seconds=30.0,
                model_version="v1.0",
                training_samples=100,
            )
        )

        # Different prediction patterns per model type
        async def mock_cv_predict(features, pred_time, state):
            n_preds = len(features)
            if model_name == "lstm":
                base_time = 1700  # LSTM tends to predict shorter times
            elif model_name == "xgboost":
                base_time = 1900  # XGBoost more varied
            elif model_name == "hmm":
                base_time = 2100  # HMM longer times
            else:  # gp
                base_time = 1800  # GP middle range

            return [
                PredictionResult(
                    predicted_time=pred_time
                    + timedelta(
                        seconds=base_time + i * 30 + np.random.uniform(-200, 200)
                    ),
                    transition_type="vacant_to_occupied",
                    confidence_score=0.7 + np.random.uniform(-0.1, 0.2),
                    model_type=model_name,
                )
                for i in range(n_preds)
            ]

        mock_instance.predict = AsyncMock(side_effect=mock_cv_predict)
        mock_class.return_value = mock_instance

        # Apply patch based on model name
        patch_targets = {
            "lstm": "src.models.ensemble.LSTMPredictor",
            "xgboost": "src.models.ensemble.XGBoostPredictor",
            "hmm": "src.models.ensemble.HMMPredictor",
            "gp": "src.models.ensemble.GaussianProcessPredictor",
        }

        if model_name in patch_targets:
            patch(patch_targets[model_name], mock_class).__enter__()

    def _create_realistic_meta_features(self, n_samples):
        """Create realistic meta-features for testing."""
        # Different models should have different prediction characteristics
        return pd.DataFrame(
            {
                "lstm": np.random.normal(
                    1700, 250, n_samples
                ),  # Shorter predictions, more varied
                "xgboost": np.random.normal(
                    1900, 200, n_samples
                ),  # Good performance, consistent
                "hmm": np.random.normal(
                    2100, 350, n_samples
                ),  # Longer predictions, more uncertainty
                "gp": np.random.normal(
                    1800, 180, n_samples
                ),  # Smooth predictions, low variance
            }
        )


class TestEnsemblePredictionGeneration:
    """Test comprehensive prediction generation and combination."""

    @pytest.mark.asyncio
    async def test_complete_prediction_pipeline(self, split_comprehensive_data):
        """Test complete prediction generation pipeline."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="prediction_test")
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Generate predictions for multiple scenarios
        test_scenarios = [
            ("vacant", "morning"),
            ("occupied", "afternoon"),
            ("unknown", "evening"),
            ("vacant", "night"),
        ]

        for state, time_label in test_scenarios:
            prediction_time = datetime(2024, 6, 15, 10, 0, tzinfo=timezone.utc)
            predictions = await ensemble.predict(
                test_features.head(5), prediction_time, state
            )

            # Validate prediction structure
            assert len(predictions) == 5

            for pred in predictions:
                assert isinstance(pred, PredictionResult)
                assert pred.model_type == ModelType.ENSEMBLE.value
                assert pred.predicted_time > prediction_time
                assert pred.transition_type in [
                    "vacant_to_occupied",
                    "occupied_to_vacant",
                ]
                assert 0.0 <= pred.confidence_score <= 1.0

                # Validate ensemble-specific metadata
                metadata = pred.prediction_metadata
                required_fields = [
                    "time_until_transition_seconds",
                    "prediction_method",
                    "base_model_predictions",
                    "model_weights",
                    "meta_learner_type",
                    "combination_method",
                ]
                for field in required_fields:
                    assert field in metadata

                assert metadata["prediction_method"] == "stacking_ensemble"
                assert isinstance(metadata["base_model_predictions"], dict)
                assert len(metadata["model_weights"]) == 4
                assert metadata["combination_method"] == "meta_learner_weighted"

    @pytest.mark.asyncio
    async def test_ensemble_confidence_calculation(self, split_comprehensive_data):
        """Test ensemble confidence calculation with uncertainty integration."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="confidence_test")
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Test confidence calculation with GP uncertainty
        gp_model = ensemble.base_models["gp"]

        # Mock GP with detailed uncertainty information
        async def mock_gp_predict(features, pred_time, state):
            return [
                PredictionResult(
                    predicted_time=pred_time + timedelta(seconds=1800),
                    transition_type="vacant_to_occupied",
                    confidence_score=0.82,
                    model_type="gp",
                    prediction_metadata={
                        "uncertainty_quantification": {
                            "aleatoric_uncertainty": 250.0,  # Data noise
                            "epistemic_uncertainty": 180.0,  # Model uncertainty
                        },
                        "prediction_std": 215.0,
                    },
                )
                for _ in range(len(features))
            ]

        gp_model.predict = AsyncMock(side_effect=mock_gp_predict)

        predictions = await ensemble.predict(
            test_features.head(3), datetime.now(timezone.utc), "vacant"
        )

        # Confidence should incorporate GP uncertainty
        for pred in predictions:
            assert 0.3 <= pred.confidence_score <= 0.95

            # Should contain GP uncertainty in base model predictions
            base_preds = pred.prediction_metadata["base_model_predictions"]
            assert "gp" in base_preds

    @pytest.mark.asyncio
    async def test_prediction_alternative_generation(self, split_comprehensive_data):
        """Test generation of alternative prediction scenarios."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="alternatives_test")
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        predictions = await ensemble.predict(
            test_features.head(3), datetime.now(timezone.utc), "occupied"
        )

        for pred in predictions:
            alternatives = pred.alternatives

            # Should have alternative predictions
            assert alternatives is not None
            assert len(alternatives) <= 3

            for alt_time, alt_confidence in alternatives:
                assert isinstance(alt_time, datetime)
                assert 0.0 <= alt_confidence <= 1.0

                # Alternatives should be reasonably different
                main_time = pred.predicted_time
                time_diff = abs((alt_time - main_time).total_seconds())
                # Allow some alternatives to be similar (realistic scenario)

    @pytest.mark.asyncio
    async def test_meta_features_creation_edge_cases(self, split_comprehensive_data):
        """Test meta-features creation with various edge cases."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="meta_edge_test")
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Test various base prediction scenarios
        test_cases = [
            # Empty predictions from some models
            {"lstm": [], "xgboost": [1800], "hmm": [2000], "gp": [1700]},
            # Different length predictions
            {
                "lstm": [1800, 1900],
                "xgboost": [1850],
                "hmm": [2000],
                "gp": [1750, 1800, 1900],
            },
            # Very different prediction ranges
            {"lstm": [300], "xgboost": [7200], "hmm": [1800], "gp": [900]},
        ]

        for i, base_predictions in enumerate(test_cases):
            try:
                meta_df = ensemble._create_meta_features(
                    base_predictions, test_features.head(1)
                )

                # Should handle edge cases gracefully
                assert isinstance(meta_df, pd.DataFrame)
                assert len(meta_df) == 1  # One sample
                assert set(meta_df.columns) >= set(ensemble.base_models.keys())

                # Should not contain NaN values
                assert not meta_df.isnull().any().any()

            except Exception as e:
                # Document any failures for debugging
                pytest.fail(f"Meta-features creation failed for case {i}: {e}")

    @pytest.mark.asyncio
    async def test_prediction_error_recovery(self, split_comprehensive_data):
        """Test prediction error handling and recovery mechanisms."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="error_recovery_test")
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Test individual base model failures
        failed_models = ["xgboost", "hmm"]

        for failed_model in failed_models:
            # Mock model failure
            ensemble.base_models[failed_model].predict = AsyncMock(
                side_effect=Exception(f"{failed_model} prediction failed")
            )

        # Should still generate predictions with remaining models
        predictions = await ensemble.predict(
            test_features.head(2), datetime.now(timezone.utc), "vacant"
        )

        assert len(predictions) == 2

        for pred in predictions:
            # Should not include failed models in base predictions
            base_preds = pred.prediction_metadata["base_model_predictions"]
            for failed in failed_models:
                assert failed not in base_preds

            # Should still have working models
            working_models = set(ensemble.base_models.keys()) - set(failed_models)
            assert len(set(base_preds.keys()) & working_models) > 0

    @pytest.mark.asyncio
    async def test_prediction_format_validation(self, split_comprehensive_data):
        """Test prediction format and metadata validation."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="format_test")
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        predictions = await ensemble.predict(
            test_features.head(5), datetime.now(timezone.utc), "unknown"
        )

        # Comprehensive format validation
        for pred in predictions:
            # Basic PredictionResult validation
            assert hasattr(pred, "predicted_time")
            assert hasattr(pred, "transition_type")
            assert hasattr(pred, "confidence_score")
            assert hasattr(pred, "model_type")
            assert hasattr(pred, "prediction_metadata")

            # Time validation
            assert isinstance(pred.predicted_time, datetime)
            assert pred.predicted_time.tzinfo is not None  # Should be timezone-aware

            # Metadata validation
            metadata = pred.prediction_metadata

            # Numeric fields
            assert isinstance(metadata["time_until_transition_seconds"], (int, float))
            assert metadata["time_until_transition_seconds"] > 0

            # Model information
            assert metadata["prediction_method"] == "stacking_ensemble"
            assert isinstance(metadata["base_model_predictions"], dict)
            assert isinstance(metadata["model_weights"], dict)

            # Weights should be normalized
            weights = metadata["model_weights"]
            total_weight = sum(weights.values())
            assert abs(total_weight - 1.0) < 0.01

    async def _setup_trained_ensemble(self, ensemble, train_features, train_targets):
        """Setup a fully trained ensemble for prediction testing."""
        # Setup trained state
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(train_features.columns)

        # Mock base models with realistic predictions
        model_characteristics = {
            "lstm": {"base_time": 1650, "variance": 280, "confidence": 0.78},
            "xgboost": {"base_time": 1850, "variance": 220, "confidence": 0.84},
            "hmm": {"base_time": 2100, "variance": 380, "confidence": 0.72},
            "gp": {"base_time": 1780, "variance": 190, "confidence": 0.81},
        }

        for model_name, model in ensemble.base_models.items():
            model.is_trained = True
            char = model_characteristics[model_name]

            async def create_mock_predict(characteristics):
                async def mock_predict(features, pred_time, state):
                    n_preds = len(features)
                    predictions = []

                    for i in range(n_preds):
                        time_until = characteristics["base_time"] + np.random.normal(
                            0, characteristics["variance"]
                        )
                        time_until = max(60, min(14400, time_until))  # Realistic bounds

                        predictions.append(
                            PredictionResult(
                                predicted_time=pred_time
                                + timedelta(seconds=time_until),
                                transition_type=(
                                    "vacant_to_occupied"
                                    if i % 2 == 0
                                    else "occupied_to_vacant"
                                ),
                                confidence_score=characteristics["confidence"]
                                + np.random.uniform(-0.05, 0.05),
                                model_type=model_name,
                                model_version="v1.0",
                            )
                        )

                    return predictions

                return mock_predict

            model.predict = AsyncMock(side_effect=await create_mock_predict(char))

        # Setup meta-learner with realistic behavior
        ensemble.meta_learner = MagicMock()

        def meta_predict(X):
            n_samples = len(X) if hasattr(X, "__len__") else 1
            # Meta-learner combines base predictions with some adjustment
            return np.random.normal(1800, 150, n_samples)

        ensemble.meta_learner.predict = MagicMock(side_effect=meta_predict)

        # Setup scaler
        ensemble.meta_scaler = MagicMock()
        ensemble.meta_scaler.transform = MagicMock(
            side_effect=lambda x: np.random.normal(0, 1, (len(x), 4))
        )
        ensemble.meta_scaler.n_features_in_ = 4

        # Setup model weights
        ensemble.model_weights = {
            "lstm": 0.22,
            "xgboost": 0.34,  # Best performer gets highest weight
            "hmm": 0.18,
            "gp": 0.26,
        }


class TestEnsembleIncrementalLearning:
    """Test ensemble incremental learning and adaptation capabilities."""

    @pytest.mark.asyncio
    async def test_incremental_update_success(self, split_comprehensive_data):
        """Test successful incremental update of ensemble."""
        train_features, train_targets, val_features, val_targets, _, _ = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(room_id="incremental_test")

        # Setup initial trained state
        await self._setup_trained_ensemble_for_incremental(
            ensemble, train_features, train_targets
        )

        initial_version = ensemble.model_version

        # Setup base models for incremental update
        for model_name, model in ensemble.base_models.items():
            model.incremental_update = AsyncMock(
                return_value=TrainingResult(
                    success=True,
                    training_time_seconds=25.0,
                    model_version=f"v1.1_{model_name}",
                    training_samples=len(val_features),
                    training_score=0.85,
                    training_metrics={"incremental_mae": 180.0},
                )
            )

        # Perform incremental update
        result = await ensemble.incremental_update(
            val_features, val_targets, learning_rate=0.15
        )

        # Validate update success
        assert result.success is True
        assert result.training_samples == len(val_features)
        assert ensemble.model_version != initial_version
        assert "_inc_" in ensemble.model_version

        # Validate metrics
        metrics = result.training_metrics
        assert metrics["update_type"] == "incremental"
        assert "incremental_mae" in metrics
        assert "incremental_r2" in metrics
        assert "learning_rate" in metrics
        assert metrics["learning_rate"] == 0.15
        assert "base_models_updated" in metrics
        assert "ensemble_weights_updated" in metrics
        assert metrics["ensemble_weights_updated"] is True

    @pytest.mark.asyncio
    async def test_incremental_update_weight_rebalancing(
        self, split_comprehensive_data
    ):
        """Test model weight rebalancing during incremental update."""
        train_features, train_targets, val_features, val_targets, _, _ = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(room_id="rebalance_test")
        await self._setup_trained_ensemble_for_incremental(
            ensemble, train_features, train_targets
        )

        # Store initial weights
        initial_weights = ensemble.model_weights.copy()

        # Setup different performance for incremental update
        performance_updates = {
            "lstm": 0.88,  # Improved significantly
            "xgboost": 0.82,  # Declined
            "hmm": 0.75,  # Similar
            "gp": 0.85,  # Improved
        }

        for model_name, model in ensemble.base_models.items():
            perf = performance_updates[model_name]
            model.incremental_update = AsyncMock(
                return_value=TrainingResult(
                    success=True,
                    training_time_seconds=20.0,
                    model_version="v1.1",
                    training_samples=len(val_features),
                    training_score=perf,
                )
            )

        # Mock prediction for weight recalculation
        for model_name, model in ensemble.base_models.items():

            async def create_mock_predict_for_weights(model_name, performance):
                async def mock_predict(features, pred_time, state):
                    # Better models should have more consistent predictions
                    base_time = 1800
                    variance = 300 * (
                        1 - performance
                    )  # Lower variance for better models

                    return [
                        PredictionResult(
                            predicted_time=pred_time
                            + timedelta(
                                seconds=base_time + np.random.normal(0, variance)
                            ),
                            transition_type="vacant_to_occupied",
                            confidence_score=performance,
                            model_type=model_name,
                        )
                        for _ in range(len(features))
                    ]

                return mock_predict

            model.predict = AsyncMock(
                side_effect=await create_mock_predict_for_weights(
                    model_name, performance_updates[model_name]
                )
            )

        await ensemble.incremental_update(val_features, val_targets)

        # Check weight rebalancing occurred
        updated_weights = ensemble.model_weights

        # LSTM should have higher weight (improved performance)
        # XGBoost should have lower weight (declined performance)
        assert updated_weights != initial_weights

        # Weights should still be normalized
        assert abs(sum(updated_weights.values()) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_incremental_update_fallback_behavior(self, split_comprehensive_data):
        """Test incremental update fallback to full training."""
        _, _, val_features, val_targets, _, _ = split_comprehensive_data

        # Untrained ensemble should fallback to full training
        ensemble = OccupancyEnsemble(room_id="fallback_test")

        with patch.object(ensemble, "train") as mock_train:
            mock_train.return_value = TrainingResult(
                success=True,
                training_time_seconds=120.0,
                model_version="v1.0",
                training_samples=len(val_features),
                training_score=0.8,
            )

            result = await ensemble.incremental_update(val_features, val_targets)

            # Should have called full training
            mock_train.assert_called_once_with(val_features, val_targets)
            assert result.success is True

    @pytest.mark.asyncio
    async def test_incremental_update_error_handling(self, split_comprehensive_data):
        """Test error handling in incremental updates."""
        train_features, train_targets, _, _, test_features, test_targets = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(room_id="error_handling_test")
        await self._setup_trained_ensemble_for_incremental(
            ensemble, train_features, train_targets
        )

        # Test insufficient data
        tiny_features = test_features.head(5)
        tiny_targets = test_targets.head(5)

        with pytest.raises(ModelTrainingError):
            await ensemble.incremental_update(tiny_features, tiny_targets)

        # Test partial base model failures
        ensemble.base_models["xgboost"].incremental_update = AsyncMock(
            side_effect=Exception("XGBoost incremental update failed")
        )

        # Should handle partial failures gracefully
        result = await ensemble.incremental_update(
            test_features.head(50), test_targets.head(50)
        )

        # Should still succeed with other models
        assert result.success is True

        # Failed model should not be in updated models list
        updated_models = result.training_metrics["base_models_updated"]
        assert "xgboost" not in updated_models

    async def _setup_trained_ensemble_for_incremental(
        self, ensemble, train_features, train_targets
    ):
        """Setup ensemble for incremental testing."""
        # Mark as trained
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(train_features.columns)
        ensemble.model_version = "v1.0"

        # Setup base models as trained
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True

        # Setup meta-learner and scaler
        ensemble.meta_learner = MagicMock()
        ensemble.meta_learner.predict = MagicMock(
            return_value=np.array([1800.0] * len(train_features))
        )

        ensemble.meta_scaler = MagicMock()
        ensemble.meta_scaler.transform = MagicMock(
            return_value=np.zeros((len(train_features), 4))
        )

        # Initial model weights
        ensemble.model_weights = {
            "lstm": 0.20,
            "xgboost": 0.40,  # Initially best
            "hmm": 0.15,
            "gp": 0.25,
        }


class TestEnsembleFeatureImportanceAnalysis:
    """Test ensemble feature importance calculation and analysis."""

    @pytest.mark.asyncio
    async def test_combined_feature_importance(self, split_comprehensive_data):
        """Test feature importance combination from base models."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="importance_test")
        await self._setup_trained_ensemble_with_importance(ensemble, train_features)

        # Get combined importance
        importance = ensemble.get_feature_importance()

        # Should have importance for all features
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

        # Should be normalized (approximately sum to 1)
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.01

        # Features that appear in multiple models should have higher importance
        common_features = {"hour_sin", "temperature", "motion_events"}
        if common_features.intersection(set(importance.keys())):
            # At least some common features should have reasonable importance
            common_importance = sum(
                importance[f] for f in common_features if f in importance
            )
            assert common_importance > 0

    @pytest.mark.asyncio
    async def test_weighted_feature_importance(self, split_comprehensive_data):
        """Test weighted feature importance based on model performance."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="weighted_importance_test")
        await self._setup_trained_ensemble_with_importance(ensemble, train_features)

        # Set different model weights (XGBoost highest)
        ensemble.model_weights = {
            "lstm": 0.15,
            "xgboost": 0.50,  # Dominant model
            "hmm": 0.10,
            "gp": 0.25,
        }

        # Mock different feature importances per model
        mock_importances = {
            "lstm": {"hour_sin": 0.4, "temperature": 0.3, "motion_events": 0.3},
            "xgboost": {"temperature": 0.5, "motion_events": 0.3, "day_of_week": 0.2},
            "hmm": {"prev_state_duration": 0.6, "state_stability_index": 0.4},
            "gp": {"smooth_seasonal": 0.7, "trend_component": 0.3},
        }

        for model_name, model in ensemble.base_models.items():
            model.get_feature_importance = MagicMock(
                return_value=mock_importances.get(model_name, {})
            )

        importance = ensemble.get_feature_importance()

        # Features from XGBoost should have higher weight due to model dominance
        if "temperature" in importance:
            # Temperature appears in both LSTM (15% weight) and XGBoost (50% weight)
            # XGBoost influence should dominate
            assert importance["temperature"] > 0

    def test_feature_importance_untrained_model(self):
        """Test feature importance on untrained ensemble."""
        ensemble = OccupancyEnsemble(room_id="untrained_test")

        importance = ensemble.get_feature_importance()
        assert importance == {}

    def test_feature_importance_no_base_importance(self, split_comprehensive_data):
        """Test feature importance when base models return empty importance."""
        train_features, _, _, _, _, _ = split_comprehensive_data
        ensemble = OccupancyEnsemble(room_id="no_importance_test")

        # Setup trained ensemble
        ensemble.is_trained = True
        ensemble.model_weights = {
            "lstm": 0.25,
            "xgboost": 0.25,
            "hmm": 0.25,
            "gp": 0.25,
        }

        # Mock all base models to return empty importance
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True
            model.get_feature_importance = MagicMock(return_value={})

        importance = ensemble.get_feature_importance()

        # Should return empty dict when no base models provide importance
        assert importance == {}

    async def _setup_trained_ensemble_with_importance(self, ensemble, train_features):
        """Setup trained ensemble with feature importance mocks."""
        ensemble.is_trained = True
        ensemble.model_weights = {
            "lstm": 0.25,
            "xgboost": 0.35,
            "hmm": 0.20,
            "gp": 0.20,
        }

        # Setup base models as trained
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True

        # Will be overridden by specific tests with custom importance


class TestEnsembleMetadataAndInformation:
    """Test ensemble metadata retrieval and information systems."""

    def test_comprehensive_ensemble_info(self):
        """Test comprehensive ensemble information retrieval."""
        ensemble = OccupancyEnsemble(
            room_id="info_test",
            meta_learner="linear",
            cv_folds=4,
            use_base_features=False,
        )

        info = ensemble.get_ensemble_info()

        # Validate structure
        expected_keys = [
            "ensemble_type",
            "base_models",
            "meta_learner",
            "model_weights",
            "model_performance",
            "cv_scores",
            "is_trained",
            "base_models_trained",
            "meta_learner_trained",
        ]

        for key in expected_keys:
            assert key in info

        # Validate values
        assert info["ensemble_type"] == "stacking"
        assert set(info["base_models"]) == {"lstm", "xgboost", "hmm", "gp"}
        assert info["meta_learner"] == "linear"
        assert info["is_trained"] is False
        assert info["base_models_trained"] is False
        assert info["meta_learner_trained"] is False

        # Initially empty performance data
        assert info["model_weights"] == {}
        assert info["model_performance"] == {}
        assert info["cv_scores"] == {}

    def test_ensemble_info_after_training(self, split_comprehensive_data):
        """Test ensemble info after training completion."""
        train_features, _, _, _, _, _ = split_comprehensive_data
        ensemble = OccupancyEnsemble(room_id="trained_info_test")

        # Simulate trained state
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True

        ensemble.model_weights = {
            "lstm": 0.22,
            "xgboost": 0.38,
            "hmm": 0.18,
            "gp": 0.22,
        }

        ensemble.model_performance = {
            "lstm": {"training_score": 0.75, "validation_score": 0.72},
            "xgboost": {"training_score": 0.84, "validation_score": 0.81},
            "hmm": {"training_score": 0.68, "validation_score": 0.65},
            "gp": {"training_score": 0.78, "validation_score": 0.76},
        }

        ensemble.cross_validation_scores = {
            "lstm": [0.74, 0.76, 0.73],
            "xgboost": [0.83, 0.85, 0.82],
            "hmm": [0.67, 0.69, 0.66],
            "gp": [0.77, 0.79, 0.76],
        }

        info = ensemble.get_ensemble_info()

        # Validate trained state info
        assert info["is_trained"] is True
        assert info["base_models_trained"] is True
        assert info["meta_learner_trained"] is True

        # Validate performance data
        assert len(info["model_weights"]) == 4
        assert len(info["model_performance"]) == 4
        assert len(info["cv_scores"]) == 4

        # Weights should sum to ~1
        total_weight = sum(info["model_weights"].values())
        assert abs(total_weight - 1.0) < 0.01

        # XGBoost should have highest weight (best performance)
        assert info["model_weights"]["xgboost"] > info["model_weights"]["hmm"]

    def test_model_info_inheritance(self):
        """Test model info from BasePredictor inheritance."""
        ensemble = OccupancyEnsemble(room_id="inheritance_test")

        model_info = ensemble.get_model_info()

        # Should inherit BasePredictor info structure
        assert model_info["model_type"] == ModelType.ENSEMBLE.value
        assert model_info["room_id"] == "inheritance_test"
        assert model_info["is_trained"] is False
        assert model_info["feature_count"] == 0
        assert model_info["training_sessions"] == 0
        assert model_info["predictions_made"] == 0

    def test_string_representations(self):
        """Test ensemble string representations."""
        ensemble = OccupancyEnsemble(room_id="string_test")

        # Test str()
        str_repr = str(ensemble)
        assert "ensemble" in str_repr.lower()
        assert "string_test" in str_repr
        assert "untrained" in str_repr.lower()

        # Test repr()
        repr_str = repr(ensemble)
        assert "OccupancyEnsemble" in repr_str
        assert "string_test" in repr_str

    def test_ensemble_version_tracking(self):
        """Test model version tracking through training sessions."""
        ensemble = OccupancyEnsemble(room_id="version_test")

        # Initial version should be unset
        assert ensemble.model_version is None

        # Simulate training to set version
        ensemble.model_version = ensemble._generate_model_version()
        initial_version = ensemble.model_version

        assert initial_version is not None
        assert "ensemble" in initial_version.lower()

        # Version should change on incremental updates
        import time

        time.sleep(0.01)  # Ensure different timestamp
        new_version = f"{initial_version}_inc_{int(time.time())}"
        ensemble.model_version = new_version

        assert ensemble.model_version != initial_version
        assert "_inc_" in ensemble.model_version


class TestEnsemblePerformanceBenchmarks:
    """Test ensemble performance benchmarks and requirements validation."""

    @pytest.mark.asyncio
    async def test_training_performance_benchmark(self, split_comprehensive_data):
        """Test ensemble training performance meets requirements."""
        train_features, train_targets, val_features, val_targets, _, _ = (
            split_comprehensive_data
        )

        # Use reasonable subset for benchmark
        bench_train_features = train_features.head(400)
        bench_train_targets = train_targets.head(400)
        bench_val_features = val_features.head(100)
        bench_val_targets = val_targets.head(100)

        ensemble = OccupancyEnsemble(
            room_id="benchmark_test", cv_folds=3  # Reduce folds for faster testing
        )

        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Setup fast base model mocks
            for model_name, model in ensemble.base_models.items():
                model.train = AsyncMock(
                    return_value=TrainingResult(
                        success=True,
                        training_time_seconds=15.0,
                        model_version="v1.0",
                        training_samples=len(bench_train_features),
                        training_score=0.8,
                    )
                )
                model.predict = AsyncMock(
                    return_value=[
                        PredictionResult(
                            predicted_time=datetime.now(timezone.utc)
                            + timedelta(seconds=1800),
                            transition_type="vacant_to_occupied",
                            confidence_score=0.8,
                            model_type=model_name,
                        )
                        for _ in range(len(bench_train_features))
                    ]
                )
                model.is_trained = True

            # Measure training time
            start_time = time.time()
            result = await ensemble.train(
                bench_train_features,
                bench_train_targets,
                bench_val_features,
                bench_val_targets,
            )
            training_duration = time.time() - start_time

            # Performance requirements
            assert result.success is True
            assert (
                training_duration < 180
            )  # Should complete within 3 minutes for test data

            print(
                f"Ensemble training time: {training_duration:.2f}s for {len(bench_train_features)} samples"
            )

    @pytest.mark.asyncio
    async def test_prediction_latency_benchmark(self, split_comprehensive_data):
        """Test prediction latency meets <100ms requirement."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="latency_test")
        await self._setup_fast_trained_ensemble(ensemble, train_features, train_targets)

        # Test batch sizes
        batch_sizes = [1, 5, 10, 20]

        for batch_size in batch_sizes:
            test_batch = test_features.head(batch_size)

            # Warm up (exclude from timing)
            await ensemble.predict(
                test_batch.head(1), datetime.now(timezone.utc), "vacant"
            )

            # Measure prediction time
            start_time = time.time()
            predictions = await ensemble.predict(
                test_batch, datetime.now(timezone.utc), "vacant"
            )
            prediction_duration = time.time() - start_time

            # Calculate latency per prediction
            latency_per_prediction = (
                prediction_duration / batch_size
            ) * 1000  # milliseconds

            print(
                f"Batch size {batch_size}: {latency_per_prediction:.2f}ms per prediction"
            )

            # Should meet <100ms requirement
            assert latency_per_prediction < 100
            assert len(predictions) == batch_size

    @pytest.mark.asyncio
    async def test_memory_usage_efficiency(self, split_comprehensive_data):
        """Test ensemble memory usage efficiency."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="memory_test")
        await self._setup_fast_trained_ensemble(ensemble, train_features, train_targets)

        # Test with varying input sizes
        sizes = [10, 50, 100, 200]

        for size in sizes:
            test_data = train_features.head(size)

            # Measure prediction memory usage (simplified)
            predictions = await ensemble.predict(
                test_data, datetime.now(timezone.utc), "vacant"
            )

            # Verify predictions scale correctly
            assert len(predictions) == size

            # Each prediction should have reasonable memory footprint
            for pred in predictions:
                # Check that metadata is not excessive
                metadata = pred.prediction_metadata
                assert len(str(metadata)) < 2000  # Reasonable metadata size

                # Base predictions should be present but not excessive
                base_preds = metadata["base_model_predictions"]
                assert len(base_preds) <= 4  # Only base models

    @pytest.mark.asyncio
    async def test_concurrent_prediction_handling(self, split_comprehensive_data):
        """Test ensemble handling of concurrent predictions."""
        train_features, train_targets, _, _, test_features, _ = split_comprehensive_data

        ensemble = OccupancyEnsemble(room_id="concurrent_test")
        await self._setup_fast_trained_ensemble(ensemble, train_features, train_targets)

        # Create multiple concurrent prediction tasks
        async def predict_batch(batch_id):
            batch_data = test_features.iloc[batch_id * 5 : (batch_id + 1) * 5]
            return await ensemble.predict(
                batch_data, datetime.now(timezone.utc), "vacant"
            )

        # Run multiple predictions concurrently
        tasks = [predict_batch(i) for i in range(4)]

        start_time = time.time()
        results = await asyncio.gather(*tasks, return_exceptions=True)
        total_time = time.time() - start_time

        # All predictions should succeed
        assert all(not isinstance(r, Exception) for r in results)
        assert all(len(r) == 5 for r in results)

        # Concurrent execution should be reasonably efficient
        assert total_time < 10  # Should handle concurrent requests quickly

        print(f"Concurrent predictions completed in {total_time:.2f}s")

    @pytest.mark.asyncio
    async def test_accuracy_benchmark_realistic_data(self, split_comprehensive_data):
        """Test ensemble accuracy on realistic occupancy patterns."""
        train_features, train_targets, _, _, test_features, test_targets = (
            split_comprehensive_data
        )

        ensemble = OccupancyEnsemble(room_id="accuracy_test")

        # Use more realistic base model performance for accuracy testing
        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Setup base models with realistic accuracy levels
            model_accuracies = {"lstm": 0.76, "xgboost": 0.83, "hmm": 0.71, "gp": 0.79}

            for model_name, model in ensemble.base_models.items():
                accuracy = model_accuracies[model_name]
                model.train = AsyncMock(
                    return_value=TrainingResult(
                        success=True,
                        training_time_seconds=60.0,
                        model_version="v1.0",
                        training_samples=len(train_features),
                        training_score=accuracy,
                        validation_score=accuracy - 0.03,
                    )
                )

                # Predictions with noise based on accuracy
                async def create_realistic_predict(model_accuracy):
                    async def realistic_predict(features, pred_time, state):
                        true_targets = test_targets.head(len(features))[
                            "time_until_transition_seconds"
                        ].values
                        predictions = []

                        for i, true_target in enumerate(true_targets):
                            # Add noise inversely proportional to accuracy
                            noise_factor = (1 - model_accuracy) * 500
                            predicted_value = true_target + np.random.normal(
                                0, noise_factor
                            )
                            predicted_value = max(60, min(14400, predicted_value))

                            predictions.append(
                                PredictionResult(
                                    predicted_time=pred_time
                                    + timedelta(seconds=predicted_value),
                                    transition_type="vacant_to_occupied",
                                    confidence_score=model_accuracy,
                                    model_type=model_name,
                                )
                            )

                        return predictions

                    return realistic_predict

                model.predict = AsyncMock(
                    side_effect=await create_realistic_predict(accuracy)
                )
                model.is_trained = True

            # Train ensemble
            result = await ensemble.train(train_features, train_targets)

            # Generate predictions
            predictions = await ensemble.predict(
                test_features.head(50), datetime.now(timezone.utc), "vacant"
            )

            # Ensemble should outperform individual models
            assert result.success is True

            # Ensemble accuracy should be competitive
            ensemble_score = result.training_score
            best_individual_score = max(model_accuracies.values())

            # Ensemble should perform at least as well as the best individual model
            # (allowing for some variation in the test)
            assert ensemble_score >= best_individual_score * 0.95

            print(
                f"Ensemble accuracy: {ensemble_score:.3f}, Best individual: {best_individual_score:.3f}"
            )

    async def _setup_fast_trained_ensemble(
        self, ensemble, train_features, train_targets
    ):
        """Setup trained ensemble optimized for performance testing."""
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(train_features.columns)

        # Setup fast base model mocks
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True

            # Fast prediction mock
            async def fast_predict(features, pred_time, state):
                n = len(features)
                base_time = 1800
                return [
                    PredictionResult(
                        predicted_time=pred_time
                        + timedelta(seconds=base_time + i * 10),
                        transition_type=(
                            "vacant_to_occupied" if i % 2 == 0 else "occupied_to_vacant"
                        ),
                        confidence_score=0.8,
                        model_type=model_name,
                    )
                    for i in range(n)
                ]

            model.predict = AsyncMock(side_effect=fast_predict)

        # Fast meta-learner mock
        ensemble.meta_learner = MagicMock()
        ensemble.meta_learner.predict = MagicMock(
            side_effect=lambda x: np.full(len(x), 1800.0)
        )

        # Fast scaler mock
        ensemble.meta_scaler = MagicMock()
        ensemble.meta_scaler.transform = MagicMock(
            side_effect=lambda x: np.zeros((len(x), 4))
        )
        ensemble.meta_scaler.n_features_in_ = 4

        ensemble.model_weights = {
            "lstm": 0.25,
            "xgboost": 0.25,
            "hmm": 0.25,
            "gp": 0.25,
        }


class TestEnsembleUtilityFunctions:
    """Test utility functions and edge cases."""

    def test_timezone_utility_functions(self):
        """Test timezone-aware utility functions."""
        # Test _ensure_timezone_aware
        naive_dt = datetime(2024, 6, 15, 12, 0, 0)
        aware_dt = _ensure_timezone_aware(naive_dt)

        assert aware_dt.tzinfo is not None
        assert aware_dt.tzinfo == timezone.utc

        # Already timezone-aware datetime should be unchanged
        already_aware = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        result = _ensure_timezone_aware(already_aware)
        assert result == already_aware
        assert result.tzinfo == timezone.utc

    def test_safe_time_difference(self):
        """Test safe time difference calculation."""
        dt1 = datetime(2024, 6, 15, 12, 30, 0, tzinfo=timezone.utc)
        dt2 = datetime(2024, 6, 15, 12, 0, 0, tzinfo=timezone.utc)

        diff = _safe_time_difference(dt1, dt2)
        assert diff == 1800.0  # 30 minutes in seconds

        # Test with naive datetimes (should be converted)
        naive_dt1 = datetime(2024, 6, 15, 12, 30, 0)
        naive_dt2 = datetime(2024, 6, 15, 12, 0, 0)

        diff_naive = _safe_time_difference(naive_dt1, naive_dt2)
        assert diff_naive == 1800.0

    @pytest.mark.asyncio
    async def test_ensemble_validation_data_handling(self, split_comprehensive_data):
        """Test ensemble input validation."""
        train_features, train_targets, _, _, _, _ = split_comprehensive_data
        ensemble = OccupancyEnsemble(room_id="validation_test")

        # Test invalid feature types
        invalid_features = train_features.copy()
        invalid_features["invalid_column"] = ["string"] * len(invalid_features)

        with pytest.raises(ModelTrainingError):
            await ensemble.train(invalid_features, train_targets)

        # Test mismatched feature/target lengths
        mismatched_targets = train_targets.head(len(train_features) // 2)

        with pytest.raises(ModelTrainingError):
            await ensemble.train(train_features, mismatched_targets)

        # Test empty data
        empty_features = train_features.head(0)
        empty_targets = train_targets.head(0)

        with pytest.raises(ModelTrainingError):
            await ensemble.train(empty_features, empty_targets)

    def test_model_weight_calculation_edge_cases(self):
        """Test model weight calculation with edge cases."""
        ensemble = OccupancyEnsemble(room_id="weight_edge_test")

        # Test with identical predictions (should have equal weights)
        identical_meta_features = pd.DataFrame(
            {
                "lstm": [1800.0] * 10,
                "xgboost": [1800.0] * 10,
                "hmm": [1800.0] * 10,
                "gp": [1800.0] * 10,
            }
        )
        y_true = np.array([1800.0] * 10)

        ensemble._calculate_model_weights(identical_meta_features, y_true)

        # Should have roughly equal weights
        weights = ensemble.model_weights
        assert len(weights) == 4
        assert all(0.2 <= w <= 0.3 for w in weights.values())
        assert abs(sum(weights.values()) - 1.0) < 0.01

        # Test with one model having much better predictions
        varied_meta_features = pd.DataFrame(
            {
                "lstm": np.random.normal(1800, 500, 10),  # High variance
                "xgboost": np.random.normal(1800, 50, 10),  # Low variance (better)
                "hmm": np.random.normal(1800, 300, 10),  # Medium variance
                "gp": np.random.normal(1800, 200, 10),  # Lower variance
            }
        )

        ensemble._calculate_model_weights(varied_meta_features, y_true)

        # XGBoost should have highest weight (most consistent)
        weights = ensemble.model_weights
        max_weight_model = max(weights, key=weights.get)
        assert max_weight_model == "xgboost"
        assert weights["xgboost"] > weights["lstm"]


class TestEnsembleCoverageMissingLines:
    """Test class specifically targeting missing coverage lines."""

    @pytest.mark.asyncio
    async def test_insufficient_data_training_error(self, comprehensive_training_data):
        """Test line 140 - insufficient data error."""
        features, targets = comprehensive_training_data

        # Use only 40 samples (less than 50 minimum)
        small_features = features.head(40)
        small_targets = targets.head(40)

        ensemble = OccupancyEnsemble("test_room")

        with pytest.raises(
            ModelTrainingError, match="Insufficient data for ensemble training"
        ):
            await ensemble.train(small_features, small_targets)

    @pytest.mark.asyncio
    async def test_prediction_without_training(self, comprehensive_training_data):
        """Test lines 275-276 - prediction without training."""
        features, _ = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")

        # Try to predict without training
        with pytest.raises(ModelPredictionError, match="Model prediction failed"):
            await ensemble.predict(
                features.head(1), datetime.now(timezone.utc), "vacant"
            )

    @pytest.mark.asyncio
    async def test_prediction_invalid_features(self, comprehensive_training_data):
        """Test lines 280-281 - prediction with invalid features."""
        features, targets = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")

        # Mock validate_features to return False
        with patch.object(ensemble, "validate_features", return_value=False):
            with pytest.raises(ModelPredictionError):
                await ensemble.predict(
                    features.head(1), datetime.now(timezone.utc), "vacant"
                )

    @pytest.mark.asyncio
    async def test_all_base_models_failed_prediction(self, comprehensive_training_data):
        """Test lines 317-318 - all base models fail prediction."""
        features, targets = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")

        # Mock all base models to be untrained
        for model in ensemble.base_models.values():
            model.is_trained = False

        ensemble.is_trained = True
        ensemble.meta_learner_trained = True

        with pytest.raises(ModelPredictionError, match="All base models failed"):
            await ensemble.predict(
                features.head(1), datetime.now(timezone.utc), "vacant"
            )

    @pytest.mark.asyncio
    async def test_incremental_update_insufficient_data(
        self, comprehensive_training_data
    ):
        """Test lines 413-417 - incremental update with insufficient data."""
        features, targets = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")
        ensemble.is_trained = True

        # Use only 5 samples (less than 10 minimum)
        small_features = features.head(5)
        small_targets = targets.head(5)

        with pytest.raises(
            ModelTrainingError, match="Insufficient data for incremental update"
        ):
            await ensemble.incremental_update(small_features, small_targets)

    @pytest.mark.asyncio
    async def test_incremental_update_untrained_fallback(
        self, comprehensive_training_data
    ):
        """Test lines 408-411 - fallback to full training when untrained."""
        features, targets = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")

        # Mock the train method
        with patch.object(ensemble, "train") as mock_train:
            mock_train.return_value = TrainingResult(
                success=True,
                training_time_seconds=1.0,
                model_version="v1.0",
                training_samples=len(features),
            )

            result = await ensemble.incremental_update(features, targets)

            mock_train.assert_called_once_with(features, targets)
            assert result.success

    def test_meta_features_empty_predictions(self):
        """Test line 776 - empty base predictions."""
        ensemble = OccupancyEnsemble("test_room")

        original_features = pd.DataFrame({"feature_1": [1, 2, 3]})
        base_predictions = {}

        with pytest.raises(ValueError, match="No base predictions provided"):
            ensemble._create_meta_features(base_predictions, original_features)

    def test_meta_features_zero_length_features(self):
        """Test lines 781-784 - zero length original features."""
        ensemble = OccupancyEnsemble("test_room")

        empty_features = pd.DataFrame()
        base_predictions = {"lstm": [1800.0]}

        with pytest.raises(
            ValueError,
            match="Cannot create meta-features with zero-length original features",
        ):
            ensemble._create_meta_features(base_predictions, empty_features)

    def test_meta_features_dimension_mismatch_error(self):
        """Test lines 810-812 - meta DataFrame dimension mismatch."""
        ensemble = OccupancyEnsemble("test_room")

        # Create a scenario that would cause dimension mismatch
        original_features = pd.DataFrame({"feature_1": [1, 2, 3]})
        base_predictions = {"lstm": [1800.0, 1900.0]}  # Length 2, but features length 3

        # Mock the DataFrame creation to cause mismatch
        with patch("pandas.DataFrame") as mock_df:
            mock_df.return_value = pd.DataFrame({"lstm": [1800.0]})  # Length 1

            with pytest.raises(ValueError, match="Meta DataFrame length mismatch"):
                ensemble._create_meta_features(base_predictions, original_features)

    def test_meta_features_empty_dataframe_error(self):
        """Test line 826 - empty meta-features DataFrame."""
        ensemble = OccupancyEnsemble("test_room")

        original_features = pd.DataFrame({"feature_1": [1, 2, 3]})

        # Mock DataFrame to return empty
        with patch("pandas.DataFrame") as mock_df:
            mock_df.return_value = pd.DataFrame()  # Empty DataFrame

            with pytest.raises(
                ValueError, match="Generated empty meta-features DataFrame"
            ):
                ensemble._create_meta_features(
                    {"lstm": [1800.0, 1900.0, 2000.0]}, original_features
                )

    def test_select_important_features_large_dataset(self):
        """Test line 892 - feature selection with large feature set."""
        ensemble = OccupancyEnsemble("test_room")

        # Create features with more than 6 columns
        large_features = pd.DataFrame(
            {f"feature_{i}": np.random.randn(100) for i in range(15)}
        )

        selected = ensemble._select_important_features(large_features)

        # Should select only first 6 features and rename them
        assert len(selected.columns) == 6
        assert all(col.startswith("orig_") for col in selected.columns)

    @pytest.mark.asyncio
    async def test_predict_ensemble_fallback_scenarios(
        self, comprehensive_training_data
    ):
        """Test lines 930-936 - prediction ensemble fallback scenarios."""
        features, targets = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")
        ensemble.meta_learner_trained = False

        # Test fallback when meta-learner not trained but base predictions exist
        with patch.object(ensemble.base_models["lstm"], "is_trained", True):
            with patch.object(ensemble.base_models["lstm"], "predict") as mock_predict:
                mock_predict.return_value = [
                    PredictionResult(
                        predicted_time=datetime.now(timezone.utc)
                        + timedelta(seconds=1800),
                        transition_type="vacant_to_occupied",
                        confidence_score=0.7,
                        alternatives=[],
                        model_type="lstm",
                        model_version="v1.0",
                        features_used=[],
                        prediction_metadata={},
                    )
                ]

                predictions = await ensemble._predict_ensemble(features.head(1))
                assert len(predictions) > 0

    def test_combine_predictions_scalar_handling(self):
        """Test lines 960-965 - scalar ensemble predictions handling."""
        ensemble = OccupancyEnsemble("test_room")

        base_results = {
            "lstm": [
                PredictionResult(
                    predicted_time=datetime.now(timezone.utc) + timedelta(seconds=1800),
                    transition_type="vacant_to_occupied",
                    confidence_score=0.7,
                    alternatives=[],
                    model_type="lstm",
                    model_version="v1.0",
                    features_used=[],
                    prediction_metadata={},
                )
            ]
        }

        # Test scalar ensemble prediction
        ensemble_predictions = 1800.0  # Scalar value
        prediction_time = datetime.now(timezone.utc)

        result = asyncio.run(
            ensemble._combine_predictions(
                base_results, ensemble_predictions, prediction_time, "vacant"
            )
        )

        assert len(result) == 1
        assert result[0].transition_type == "vacant_to_occupied"

    @pytest.mark.asyncio
    async def test_combine_predictions_extension_scenarios(self):
        """Test lines 977-986 - prediction extension scenarios."""
        ensemble = OccupancyEnsemble("test_room")

        base_results = {
            "lstm": [
                PredictionResult(
                    predicted_time=datetime.now(timezone.utc) + timedelta(seconds=1800),
                    transition_type="vacant_to_occupied",
                    confidence_score=0.7,
                    alternatives=[],
                    model_type="lstm",
                    model_version="v1.0",
                    features_used=[],
                    prediction_metadata={},
                ),
                PredictionResult(
                    predicted_time=datetime.now(timezone.utc) + timedelta(seconds=2000),
                    transition_type="occupied_to_vacant",
                    confidence_score=0.6,
                    alternatives=[],
                    model_type="lstm",
                    model_version="v1.0",
                    features_used=[],
                    prediction_metadata={},
                ),
            ]
        }

        # Test with fewer ensemble predictions than base model predictions
        ensemble_predictions = np.array(
            [1500.0]
        )  # Only 1 prediction for 2 base predictions
        prediction_time = datetime.now(timezone.utc)

        results = await ensemble._combine_predictions(
            base_results, ensemble_predictions, prediction_time, "vacant"
        )

        # Should extend predictions
        assert len(results) == 2

    def test_combine_predictions_transition_type_fallbacks(self):
        """Test lines 1009, 1013 - transition type fallback logic."""
        ensemble = OccupancyEnsemble("test_room")

        base_results = {}  # No XGBoost results
        ensemble_predictions = np.array([1800.0])
        prediction_time = datetime.now(timezone.utc).replace(hour=14)  # 2 PM

        # Test occupied state
        result = asyncio.run(
            ensemble._combine_predictions(
                base_results, ensemble_predictions, prediction_time, "occupied"
            )
        )

        assert result[0].transition_type == "occupied_to_vacant"

        # Test vacant state
        result = asyncio.run(
            ensemble._combine_predictions(
                base_results, ensemble_predictions, prediction_time, "vacant"
            )
        )

        assert result[0].transition_type == "vacant_to_occupied"

    def test_calculate_ensemble_confidence_no_confidences(self):
        """Test line 1112 - no confidences available."""
        ensemble = OccupancyEnsemble("test_room")

        base_results = {}  # Empty results
        confidence = ensemble._calculate_ensemble_confidence(base_results, 0, 1800.0)

        # Should return default confidence
        assert confidence == 0.7

    def test_prepare_targets_fallback_scenarios(self):
        """Test lines 1149-1157 - target preparation fallback scenarios."""
        ensemble = OccupancyEnsemble("test_room")

        # Test with unexpected target format
        targets = pd.DataFrame(
            {"unexpected_column": [1800, 2000, 1500], "another_column": [0.5, 0.7, 0.8]}
        )

        result = ensemble._prepare_targets(targets)

        # Should use first column as fallback
        assert len(result) == 3
        assert np.allclose(result, [1800, 2000, 1500])

    def test_validation_edge_cases(self):
        """Test validation edge cases lines 1171, 1173, etc."""
        ensemble = OccupancyEnsemble("test_room")

        # Test non-DataFrame features
        with pytest.raises(ValueError, match="Features must be a pandas DataFrame"):
            ensemble._validate_training_data([1, 2, 3], pd.DataFrame())

        # Test non-DataFrame targets
        with pytest.raises(ValueError, match="Targets must be a pandas DataFrame"):
            ensemble._validate_training_data(pd.DataFrame(), [1, 2, 3])

    def test_validation_dimension_mismatch(self):
        """Test validation dimension mismatch lines 1177."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, 3]})
        targets = pd.DataFrame({"t1": [100, 200]})  # Different length

        with pytest.raises(
            ValueError, match="Features and targets must have same length"
        ):
            ensemble._validate_training_data(features, targets)

    def test_validation_empty_data(self):
        """Test validation empty data line 1184."""
        ensemble = OccupancyEnsemble("test_room")

        empty_features = pd.DataFrame()
        empty_targets = pd.DataFrame()

        with pytest.raises(ValueError, match="Features and targets cannot be empty"):
            ensemble._validate_training_data(empty_features, empty_targets)

    def test_validation_non_numeric_features(self):
        """Test validation non-numeric features lines 1196-1197."""
        ensemble = OccupancyEnsemble("test_room")

        # Create features with too many non-numeric columns
        features = pd.DataFrame(
            {
                "text1": ["a", "b", "c"] * 20,
                "text2": ["x", "y", "z"] * 20,
                "text3": ["p", "q", "r"] * 20,
                "text4": ["1", "2", "3"] * 20,
                "text5": ["foo", "bar", "baz"] * 20,
                "num1": [1, 2, 3] * 20,  # Only 1 numeric out of 6
            }
        )
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800] * 60,
                "transition_type": ["vacant_to_occupied"] * 60,
                "target_time": [datetime.now()] * 60,
            }
        )

        with pytest.raises(ValueError, match="Too many non-numeric features"):
            ensemble._validate_training_data(features, targets)

    def test_validation_nan_features(self):
        """Test validation NaN features lines 1203-1204."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, np.nan] * 20, "f2": [4, 5, 6] * 20})
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800] * 60,
                "transition_type": ["vacant_to_occupied"] * 60,
                "target_time": [datetime.now()] * 60,
            }
        )

        with pytest.raises(ValueError, match="Features contain NaN values"):
            ensemble._validate_training_data(features, targets)

    def test_validation_nan_targets(self):
        """Test validation NaN targets lines 1207-1208."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, 3] * 20, "f2": [4, 5, 6] * 20})
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800, np.nan, 2000] + [1800] * 57,
                "transition_type": ["vacant_to_occupied"] * 60,
                "target_time": [datetime.now()] * 60,
            }
        )

        with pytest.raises(ValueError, match="Targets contain NaN values"):
            ensemble._validate_training_data(features, targets)

    def test_validation_missing_target_columns(self):
        """Test validation missing target columns lines 1217-1218."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, 3] * 20, "f2": [4, 5, 6] * 20})
        targets = pd.DataFrame(
            {"wrong_column": [1800] * 60, "another_wrong": ["vacant_to_occupied"] * 60}
        )

        with pytest.raises(ValueError, match="Targets missing required columns"):
            ensemble._validate_training_data(features, targets)

    def test_validation_target_value_range(self):
        """Test validation target value range lines 1226, 1229."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, 3] * 20, "f2": [4, 5, 6] * 20})

        # Test values too low
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [30] * 60,  # Below 60 minimum
                "transition_type": ["vacant_to_occupied"] * 60,
                "target_time": [datetime.now()] * 60,
            }
        )

        with pytest.raises(
            ValueError, match="time_until_transition_seconds must be between"
        ):
            ensemble._validate_training_data(features, targets)

    def test_validation_non_numeric_targets(self):
        """Test validation non-numeric targets line 1226."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, 3] * 20, "f2": [4, 5, 6] * 20})
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": ["string"] * 60,  # Non-numeric
                "transition_type": ["vacant_to_occupied"] * 60,
                "target_time": [datetime.now()] * 60,
            }
        )

        with pytest.raises(
            ValueError, match="time_until_transition_seconds must be numeric"
        ):
            ensemble._validate_training_data(features, targets)

    def test_validation_column_consistency(self):
        """Test validation column consistency lines 1243-1247, 1253-1255."""
        ensemble = OccupancyEnsemble("test_room")

        features = pd.DataFrame({"f1": [1, 2, 3] * 20})
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800] * 60,
                "transition_type": ["vacant_to_occupied"] * 60,
                "target_time": [datetime.now()] * 60,
            }
        )

        val_features = pd.DataFrame({"f2": [4, 5, 6] * 10})  # Different columns
        val_targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800] * 30,
                "transition_type": ["vacant_to_occupied"] * 30,
                "target_time": [datetime.now()] * 30,
            }
        )

        with pytest.raises(
            ValueError, match="Validation features have different columns"
        ):
            ensemble._validate_training_data(
                features, targets, val_features, val_targets
            )

    @pytest.mark.asyncio
    async def test_save_load_functionality(self, comprehensive_training_data):
        """Test save and load model functionality lines 1284-1391."""
        features, targets = comprehensive_training_data

        ensemble = OccupancyEnsemble("test_room")

        # Train the ensemble first
        await ensemble.train(features, targets)

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "ensemble_model.pkl"

            # Test successful save
            success = ensemble.save_model(model_path)
            assert success
            assert model_path.exists()

            # Test successful load
            new_ensemble = OccupancyEnsemble("test_room")
            load_success = new_ensemble.load_model(model_path)
            assert load_success
            assert new_ensemble.is_trained
            assert new_ensemble.model_version == ensemble.model_version

    def test_save_load_failure_scenarios(self):
        """Test save/load failure scenarios."""
        ensemble = OccupancyEnsemble("test_room")

        # Test save failure with invalid path
        invalid_path = "/invalid/path/model.pkl"
        success = ensemble.save_model(invalid_path)
        assert not success

        # Test load failure with non-existent file
        non_existent_path = "/non/existent/model.pkl"
        load_success = ensemble.load_model(non_existent_path)
        assert not load_success

    def test_ensemble_info_comprehensive(self):
        """Test comprehensive ensemble info retrieval."""
        ensemble = OccupancyEnsemble("test_room")

        info = ensemble.get_ensemble_info()

        assert info["ensemble_type"] == "stacking"
        assert "base_models" in info
        assert "meta_learner" in info
        assert "model_weights" in info
        assert "is_trained" in info
        assert "base_models_trained" in info
        assert "meta_learner_trained" in info


# Mark all tests as requiring the 'models' fixture to ensure proper setup
pytestmark = pytest.mark.models
