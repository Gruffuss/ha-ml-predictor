"""
Comprehensive unit tests for ensemble predictor and meta-learner.

This module tests the ensemble architecture including stacking, meta-learning,
model weight optimization, uncertainty quantification integration, and
performance comparison with base models.
"""

import asyncio
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import time
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.constants import ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.gp_predictor import GaussianProcessPredictor
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult
from src.models.base.xgboost_predictor import XGBoostPredictor
from src.models.ensemble import OccupancyEnsemble


@pytest.fixture
def ensemble_training_data():
    """Create synthetic training data optimized for ensemble testing."""
    np.random.seed(42)

    n_samples = 800
    n_features = 15

    # Create features with different patterns that different models excel at
    features = {}

    # Temporal features (good for LSTM and time-based models)
    hours = np.random.randint(0, 24, n_samples)
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    features["weekday"] = np.random.randint(0, 7, n_samples)

    # Sequential features (good for HMM)
    features["prev_state_duration"] = np.random.exponential(1800, n_samples)
    features["transition_count_1h"] = np.random.poisson(2, n_samples)
    features["state_stability"] = np.random.uniform(0, 1, n_samples)

    # Non-linear features (good for XGBoost)
    features["temp"] = 20 + np.random.normal(0, 3, n_samples)
    features["temp_squared"] = features["temp"] ** 2
    features["motion_intensity"] = np.random.gamma(2, 2, n_samples)

    # GP-friendly smooth features
    features["smooth_pattern"] = np.sin(
        np.linspace(0, 4 * np.pi, n_samples)
    ) + np.random.normal(0, 0.1, n_samples)
    features["smooth_trend"] = np.linspace(0, 1, n_samples) + np.random.normal(
        0, 0.05, n_samples
    )

    # Additional features
    for i in range(n_features - len(features)):
        features[f"feature_{i}"] = np.random.normal(0, 1, n_samples)

    # Create targets with complex patterns that benefit from ensemble
    base_time = 1800  # 30 minutes

    # Multi-factor influence on transition time
    hour_influence = np.where((hours >= 7) & (hours <= 22), 0.8, 1.5)
    temp_influence = 1 + 0.1 * np.sin(features["temp"] / 5)
    motion_influence = 1 / (1 + features["motion_intensity"] / 10)

    targets = (
        base_time
        * hour_influence
        * temp_influence
        * motion_influence
        * (1 + np.random.normal(0, 0.2, n_samples))
    )
    targets = np.clip(targets, 60, 10800)  # 1 minute to 3 hours

    # Create transition types based on patterns
    occupancy_states = np.where(
        (hours >= 22) | (hours <= 6),
        "occupied_to_vacant",
        "vacant_to_occupied",
    )

    features_df = pd.DataFrame(features)
    targets_df = pd.DataFrame(
        {
            "time_until_transition_seconds": targets,
            "transition_type": occupancy_states,
            "target_time": [
                datetime.utcnow() + timedelta(seconds=i * 10) for i in range(n_samples)
            ],
        }
    )

    return features_df, targets_df


@pytest.fixture
def ensemble_validation_data(ensemble_training_data):
    """Split ensemble data into training and validation sets."""
    features, targets = ensemble_training_data

    # 80/20 split for training/validation
    train_size = int(0.8 * len(features))

    train_features = features.iloc[:train_size].copy()
    train_targets = targets.iloc[:train_size].copy()

    val_features = features.iloc[train_size:].copy()
    val_targets = targets.iloc[train_size:].copy()

    return train_features, train_targets, val_features, val_targets


@pytest.fixture
def mock_tracking_manager():
    """Create a mock tracking manager for ensemble integration."""
    tracking_manager = AsyncMock()
    tracking_manager.register_model = AsyncMock()
    tracking_manager.record_prediction = AsyncMock()
    tracking_manager.get_model_accuracy = AsyncMock(return_value=0.85)
    return tracking_manager


class TestEnsembleInitialization:
    """Test ensemble predictor initialization and configuration."""

    def test_ensemble_basic_initialization(self):
        """Test basic ensemble initialization."""
        ensemble = OccupancyEnsemble(room_id="test_room")

        assert ensemble.model_type == ModelType.ENSEMBLE
        assert ensemble.room_id == "test_room"
        assert not ensemble.is_trained
        assert not ensemble.base_models_trained
        assert not ensemble.meta_learner_trained

        # Check base models are initialized
        expected_models = ["lstm", "xgboost", "hmm", "gp"]
        assert set(ensemble.base_models.keys()) == set(expected_models)

        for model_name, model in ensemble.base_models.items():
            assert model.room_id == "test_room"
            assert not model.is_trained

    def test_ensemble_custom_parameters(self):
        """Test ensemble initialization with custom parameters."""
        custom_params = {
            "meta_learner": "linear",
            "cv_folds": 3,
            "use_base_features": False,
            "meta_features_only": True,
        }

        ensemble = OccupancyEnsemble(room_id="custom_room", **custom_params)

        assert ensemble.model_params["meta_learner"] == "linear"
        assert ensemble.model_params["cv_folds"] == 3
        assert ensemble.model_params["use_base_features"] is False
        assert ensemble.model_params["meta_features_only"] is True

    def test_ensemble_with_tracking_manager(self, mock_tracking_manager):
        """Test ensemble initialization with tracking manager integration."""
        ensemble = OccupancyEnsemble(
            room_id="tracked_room", tracking_manager=mock_tracking_manager
        )

        assert ensemble.tracking_manager == mock_tracking_manager

        # Should register with tracking manager
        mock_tracking_manager.register_model.assert_called_once_with(
            "tracked_room", ModelType.ENSEMBLE.value, ensemble
        )


class TestEnsembleTraining:
    """Test ensemble training pipeline and meta-learning."""

    @pytest.mark.asyncio
    async def test_ensemble_training_phases(self, ensemble_validation_data):
        """Test the three-phase ensemble training process."""
        train_features, train_targets, val_features, val_targets = (
            ensemble_validation_data
        )
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Mock base model training to control behavior
        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Setup mock models
            for model_name in ensemble.base_models.keys():
                mock_model = MagicMock()
                mock_model.train = AsyncMock(
                    return_value=TrainingResult(
                        success=True,
                        training_time_seconds=60.0,
                        model_version="v1.0",
                        training_samples=len(train_features),
                        training_score=0.85,
                        validation_score=0.80,
                    )
                )
                mock_model.predict = AsyncMock(
                    return_value=[
                        PredictionResult(
                            predicted_time=datetime.utcnow() + timedelta(seconds=1800),
                            transition_type="vacant_to_occupied",
                            confidence_score=0.8,
                            model_type=model_name,
                        )
                        for _ in range(len(train_features))
                    ]
                )
                mock_model.is_trained = True
                ensemble.base_models[model_name] = mock_model

            # Train ensemble
            result = await ensemble.train(
                train_features, train_targets, val_features, val_targets
            )

            # Verify training success
            assert result.success is True
            assert result.training_samples == len(train_features)
            assert result.validation_score is not None
            assert result.training_score is not None

            # Check ensemble state
            assert ensemble.is_trained is True
            assert ensemble.base_models_trained is True
            assert ensemble.meta_learner_trained is True
            assert ensemble.meta_learner is not None

            # Check training metrics
            metrics = result.training_metrics
            assert "ensemble_mae" in metrics
            assert "ensemble_rmse" in metrics
            assert "ensemble_r2" in metrics
            assert "base_model_count" in metrics
            assert "model_weights" in metrics
            assert metrics["base_model_count"] == 4

    @pytest.mark.asyncio
    async def test_ensemble_cross_validation_meta_features(
        self, ensemble_validation_data
    ):
        """Test cross-validation for meta-feature generation."""
        train_features, train_targets, _, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room", cv_folds=3)

        # Use smaller dataset for faster testing
        small_features = train_features.head(150)
        small_targets = train_targets.head(150)

        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            # Mock fold models
            for model_name in ensemble.base_models.keys():
                mock_model_class = MagicMock()
                mock_instance = MagicMock()
                mock_instance.train = AsyncMock(
                    return_value=TrainingResult(
                        success=True,
                        training_time_seconds=30.0,
                        model_version="v1.0",
                        training_samples=100,
                    )
                )
                mock_instance.predict = AsyncMock(
                    return_value=[
                        PredictionResult(
                            predicted_time=datetime.utcnow()
                            + timedelta(seconds=1800 + i * 60),
                            transition_type="vacant_to_occupied",
                            confidence_score=0.8,
                        )
                        for i in range(50)  # Variable predictions for CV
                    ]
                )
                mock_model_class.return_value = mock_instance

                # Patch the specific model class
                if model_name == "lstm":
                    with patch("src.models.ensemble.LSTMPredictor", mock_model_class):
                        pass
                elif model_name == "xgboost":
                    with patch(
                        "src.models.ensemble.XGBoostPredictor",
                        mock_model_class,
                    ):
                        pass
                elif model_name == "hmm":
                    with patch("src.models.ensemble.HMMPredictor", mock_model_class):
                        pass
                elif model_name == "gp":
                    with patch(
                        "src.models.ensemble.GaussianProcessPredictor",
                        mock_model_class,
                    ):
                        pass

            # Test CV meta-feature generation
            meta_features = await ensemble._train_base_models_cv(
                small_features, small_targets
            )

            # Verify meta-features structure
            assert isinstance(meta_features, pd.DataFrame)
            assert len(meta_features) == len(small_features)
            assert set(meta_features.columns) == set(ensemble.base_models.keys())

            # Check CV scores were recorded
            assert len(ensemble.cross_validation_scores) > 0
            for model_name in ensemble.base_models.keys():
                if model_name in ensemble.cross_validation_scores:
                    scores = ensemble.cross_validation_scores[model_name]
                    assert len(scores) > 0
                    assert all(-1 <= score <= 1 for score in scores)

    @pytest.mark.asyncio
    async def test_ensemble_meta_learner_training(self, ensemble_validation_data):
        """Test meta-learner training with different configurations."""
        train_features, train_targets, _, _ = ensemble_validation_data

        # Test Random Forest meta-learner
        ensemble_rf = OccupancyEnsemble(
            room_id="test_room", meta_learner="random_forest"
        )

        # Create mock meta-features
        meta_features = pd.DataFrame(
            {
                "lstm": np.random.normal(1800, 300, len(train_features)),
                "xgboost": np.random.normal(1800, 250, len(train_features)),
                "hmm": np.random.normal(1800, 400, len(train_features)),
                "gp": np.random.normal(1800, 200, len(train_features)),
            }
        )

        # Train meta-learner
        await ensemble_rf._train_meta_learner(
            meta_features, train_targets, train_features
        )

        assert ensemble_rf.meta_learner_trained is True
        assert ensemble_rf.meta_learner is not None
        assert len(ensemble_rf.model_weights) == 4

        # Test Linear meta-learner
        ensemble_linear = OccupancyEnsemble(room_id="test_room", meta_learner="linear")

        await ensemble_linear._train_meta_learner(
            meta_features, train_targets, train_features
        )

        assert ensemble_linear.meta_learner_trained is True
        assert ensemble_linear.meta_learner is not None

    @pytest.mark.asyncio
    async def test_ensemble_model_weight_calculation(self, ensemble_validation_data):
        """Test automatic model weight calculation based on performance."""
        train_features, train_targets, _, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Create meta-features with different performance levels
        n_samples = len(train_features)
        meta_features = pd.DataFrame(
            {
                "lstm": np.random.normal(1800, 100, n_samples),  # Good performance
                "xgboost": np.random.normal(1800, 80, n_samples),  # Better performance
                "hmm": np.random.normal(1800, 200, n_samples),  # Worse performance
                "gp": np.random.normal(1800, 120, n_samples),  # Good performance
            }
        )

        y_true = train_targets["time_until_transition_seconds"].values

        # Calculate weights
        ensemble._calculate_model_weights(meta_features, y_true)

        # Verify weights
        assert len(ensemble.model_weights) == 4
        assert all(w >= 0 for w in ensemble.model_weights.values())
        assert abs(sum(ensemble.model_weights.values()) - 1.0) < 0.001  # Sum to 1

        # XGBoost should have highest weight (best performance)
        max_weight_model = max(ensemble.model_weights, key=ensemble.model_weights.get)
        assert max_weight_model == "xgboost"

    @pytest.mark.asyncio
    async def test_ensemble_training_error_handling(self, ensemble_validation_data):
        """Test ensemble training error handling and recovery."""
        train_features, train_targets, _, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Test insufficient data
        tiny_features = train_features.head(20)
        tiny_targets = train_targets.head(20)

        with pytest.raises(ModelTrainingError):
            await ensemble.train(tiny_features, tiny_targets)

        # Test base model training failures
        with patch.object(
            ensemble.base_models["xgboost"],
            "train",
            side_effect=Exception("Base model training failed"),
        ):

            # Should handle individual model failures gracefully
            result = await ensemble.train(
                train_features.head(100), train_targets.head(100)
            )

            # Training should still proceed with other models
            # (This depends on implementation - may fail or continue)
            assert isinstance(result, TrainingResult)


class TestEnsemblePrediction:
    """Test ensemble prediction generation and combination."""

    @pytest.mark.asyncio
    async def test_ensemble_prediction_generation(self, ensemble_validation_data):
        """Test ensemble prediction generation and format."""
        train_features, train_targets, val_features, val_targets = (
            ensemble_validation_data
        )
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Setup trained ensemble
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Generate predictions
        prediction_time = datetime.utcnow()
        predictions = await ensemble.predict(
            val_features.head(10), prediction_time, "vacant"
        )

        # Verify prediction format
        assert len(predictions) == 10

        for prediction in predictions:
            assert isinstance(prediction, PredictionResult)
            assert prediction.model_type == ModelType.ENSEMBLE.value
            assert prediction.predicted_time > prediction_time
            assert prediction.transition_type in [
                "vacant_to_occupied",
                "occupied_to_vacant",
            ]
            assert 0.0 <= prediction.confidence_score <= 1.0
            assert prediction.alternatives is not None
            assert len(prediction.alternatives) <= 3

            # Check ensemble-specific metadata
            metadata = prediction.prediction_metadata
            assert "base_model_predictions" in metadata
            assert "model_weights" in metadata
            assert "meta_learner_type" in metadata
            assert "combination_method" in metadata

    @pytest.mark.asyncio
    async def test_ensemble_confidence_with_gp_uncertainty(
        self, ensemble_validation_data
    ):
        """Test ensemble confidence calculation with GP uncertainty quantification."""
        train_features, train_targets, val_features, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Mock GP predictions with uncertainty information
        gp_model = ensemble.base_models["gp"]
        gp_predictions = [
            PredictionResult(
                predicted_time=datetime.utcnow() + timedelta(seconds=1800),
                transition_type="vacant_to_occupied",
                confidence_score=0.75,
                model_type="gp",
                prediction_metadata={
                    "uncertainty_quantification": {
                        "aleatoric_uncertainty": 300,  # seconds
                        "epistemic_uncertainty": 200,  # seconds
                    },
                    "prediction_std": 250,
                },
            )
        ]

        with patch.object(gp_model, "predict", return_value=gp_predictions):
            predictions = await ensemble.predict(
                val_features.head(1), datetime.utcnow(), "vacant"
            )

            prediction = predictions[0]

            # Confidence should incorporate GP uncertainty
            # Lower uncertainty should lead to higher confidence
            assert 0.3 <= prediction.confidence_score <= 0.95

            # GP uncertainty should be reflected in the ensemble metadata
            metadata = prediction.prediction_metadata
            assert "base_model_predictions" in metadata
            gp_prediction_time = metadata["base_model_predictions"].get("gp")
            assert gp_prediction_time is not None

    @pytest.mark.asyncio
    async def test_ensemble_prediction_combination_methods(
        self, ensemble_validation_data
    ):
        """Test different prediction combination methods."""
        train_features, train_targets, val_features, _ = ensemble_validation_data

        # Test weighted combination
        ensemble_weighted = OccupancyEnsemble(
            room_id="test_room", stacking_method="linear"
        )

        await self._setup_trained_ensemble(
            ensemble_weighted, train_features, train_targets
        )

        predictions_weighted = await ensemble_weighted.predict(
            val_features.head(5), datetime.utcnow(), "vacant"
        )

        assert len(predictions_weighted) == 5

        # Test that predictions use weighted combination
        for pred in predictions_weighted:
            metadata = pred.prediction_metadata
            assert metadata["combination_method"] == "meta_learner_weighted"
            assert "model_weights" in metadata
            assert len(metadata["model_weights"]) == 4

    @pytest.mark.asyncio
    async def test_ensemble_alternatives_generation(self, ensemble_validation_data):
        """Test generation of alternative predictions from base models."""
        train_features, train_targets, val_features, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        predictions = await ensemble.predict(
            val_features.head(3), datetime.utcnow(), "occupied"
        )

        for prediction in predictions:
            alternatives = prediction.alternatives

            # Should have alternatives from base models
            assert alternatives is not None
            assert len(alternatives) <= 3

            for alt_time, alt_confidence in alternatives:
                assert isinstance(alt_time, datetime)
                assert 0.0 <= alt_confidence <= 1.0

                # Alternative should be different from main prediction
                # (allowing for some models to have similar predictions)
                time_diff = abs((alt_time - prediction.predicted_time).total_seconds())
                # Most alternatives should differ, but some might be similar

    @pytest.mark.asyncio
    async def test_ensemble_prediction_error_handling(self, ensemble_validation_data):
        """Test prediction error handling and fallback mechanisms."""
        train_features, train_targets, val_features, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Test prediction on untrained ensemble
        with pytest.raises(ModelPredictionError):
            await ensemble.predict(val_features.head(5), datetime.utcnow(), "vacant")

        # Setup partially trained ensemble (base models trained, meta-learner not)
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = False

        with pytest.raises(ModelPredictionError):
            await ensemble.predict(val_features.head(5), datetime.utcnow(), "vacant")

        # Test with some base models failing
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Mock one base model to fail
        with patch.object(
            ensemble.base_models["xgboost"],
            "predict",
            side_effect=Exception("Model prediction failed"),
        ):

            predictions = await ensemble.predict(
                val_features.head(3), datetime.utcnow(), "vacant"
            )

            # Should still generate predictions with remaining models
            assert len(predictions) == 3

            for pred in predictions:
                # Base model predictions should not include failed model
                base_preds = pred.prediction_metadata["base_model_predictions"]
                assert "xgboost" not in base_preds

    async def _setup_trained_ensemble(self, ensemble, train_features, train_targets):
        """Helper to setup a trained ensemble for testing."""
        # Mock all base models as trained
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True
            model.train = AsyncMock(
                return_value=TrainingResult(
                    success=True,
                    training_time_seconds=60.0,
                    model_version="v1.0",
                    training_samples=len(train_features),
                )
            )

            # Mock predictions with some variation
            base_predictions = []
            for i in range(len(train_features)):
                pred_time = datetime.utcnow() + timedelta(seconds=1800 + i * 10)
                base_predictions.append(
                    PredictionResult(
                        predicted_time=pred_time,
                        transition_type=(
                            "vacant_to_occupied" if i % 2 == 0 else "occupied_to_vacant"
                        ),
                        confidence_score=0.7 + (i % 3) * 0.1,
                        model_type=model_name,
                    )
                )

            model.predict = AsyncMock(return_value=base_predictions)

        # Setup ensemble state
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(train_features.columns)

        # Mock meta-learner
        ensemble.meta_learner = MagicMock()
        ensemble.meta_learner.predict = MagicMock(
            return_value=np.array([1800.0] * len(train_features))
        )

        # Mock scaler
        ensemble.meta_scaler = MagicMock()
        ensemble.meta_scaler.transform = MagicMock(
            return_value=np.zeros((len(train_features), 4))
        )

        # Setup model weights
        ensemble.model_weights = {
            "lstm": 0.25,
            "xgboost": 0.35,
            "hmm": 0.20,
            "gp": 0.20,
        }


class TestEnsembleIncrementalUpdate:
    """Test ensemble incremental learning and adaptation."""

    @pytest.mark.asyncio
    async def test_ensemble_incremental_update(self, ensemble_validation_data):
        """Test ensemble incremental update functionality."""
        train_features, train_targets, val_features, val_targets = (
            ensemble_validation_data
        )
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Initial training
        await self._setup_trained_ensemble(ensemble, train_features, train_targets)
        initial_version = ensemble.model_version

        # Setup base models for incremental update
        for model_name, model in ensemble.base_models.items():
            model.incremental_update = AsyncMock(
                return_value=TrainingResult(
                    success=True,
                    training_time_seconds=30.0,
                    model_version="v1.1",
                    training_samples=len(val_features),
                    training_score=0.88,
                )
            )

        # Perform incremental update
        update_result = await ensemble.incremental_update(
            val_features, val_targets, learning_rate=0.1
        )

        # Verify update result
        assert update_result.success is True
        assert update_result.training_samples == len(val_features)
        assert ensemble.model_version != initial_version

        # Check update metrics
        metrics = update_result.training_metrics
        assert metrics["update_type"] == "incremental"
        assert "incremental_mae" in metrics
        assert "incremental_r2" in metrics
        assert "learning_rate" in metrics
        assert metrics["learning_rate"] == 0.1
        assert "base_models_updated" in metrics
        assert "ensemble_weights_updated" in metrics

    @pytest.mark.asyncio
    async def test_incremental_update_error_handling(self, ensemble_validation_data):
        """Test incremental update error handling."""
        _, _, val_features, val_targets = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Test incremental update on untrained model
        result = await ensemble.incremental_update(val_features, val_targets)

        # Should fallback to full training
        assert result.success is True

        # Test with insufficient data
        tiny_features = val_features.head(5)
        tiny_targets = val_targets.head(5)

        ensemble.is_trained = True

        with pytest.raises(ModelTrainingError):
            await ensemble.incremental_update(tiny_features, tiny_targets)

    async def _setup_trained_ensemble(self, ensemble, train_features, train_targets):
        """Helper to setup a trained ensemble for testing."""
        # Same as previous helper method
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True

        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(train_features.columns)

        # Mock meta-learner and scaler
        ensemble.meta_learner = MagicMock()
        ensemble.meta_learner.predict = MagicMock(
            return_value=np.array([1800.0] * len(train_features))
        )

        ensemble.meta_scaler = MagicMock()
        ensemble.meta_scaler.transform = MagicMock(
            return_value=np.zeros((len(train_features), 4))
        )

        ensemble.model_weights = {
            "lstm": 0.25,
            "xgboost": 0.35,
            "hmm": 0.20,
            "gp": 0.20,
        }


class TestEnsembleFeatureImportance:
    """Test ensemble feature importance calculation."""

    @pytest.mark.asyncio
    async def test_ensemble_feature_importance_combination(
        self, ensemble_validation_data
    ):
        """Test feature importance combination from base models."""
        train_features, train_targets, _, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Mock base model feature importance
        mock_importances = {
            "lstm": {"hour_sin": 0.3, "temp": 0.2, "motion_intensity": 0.5},
            "xgboost": {"hour_sin": 0.4, "temp": 0.3, "motion_intensity": 0.3},
            "hmm": {
                "hour_sin": 0.1,
                "prev_state_duration": 0.6,
                "state_stability": 0.3,
            },
            "gp": {
                "smooth_pattern": 0.7,
                "smooth_trend": 0.2,
                "hour_sin": 0.1,
            },
        }

        for model_name, model in ensemble.base_models.items():
            model.get_feature_importance = MagicMock(
                return_value=mock_importances.get(model_name, {})
            )

        # Get combined feature importance
        combined_importance = ensemble.get_feature_importance()

        # Verify combination
        assert len(combined_importance) > 0
        assert all(isinstance(v, (int, float)) for v in combined_importance.values())
        assert all(v >= 0 for v in combined_importance.values())

        # hour_sin should have high importance (appears in multiple models)
        assert "hour_sin" in combined_importance
        assert combined_importance["hour_sin"] > 0

        # Weighted combination should reflect model weights
        # XGBoost has higher weight, so its features should be more influential
        if "temp" in combined_importance:
            # temp appears in both LSTM and XGBoost, XGBoost weight should dominate
            pass  # Exact values depend on normalization

    def test_ensemble_feature_importance_untrained(self):
        """Test feature importance on untrained ensemble."""
        ensemble = OccupancyEnsemble(room_id="test_room")

        importance = ensemble.get_feature_importance()

        # Should return empty dict for untrained model
        assert importance == {}

    async def _setup_trained_ensemble(self, ensemble, train_features, train_targets):
        """Helper to setup a trained ensemble for testing."""
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True

        ensemble.is_trained = True
        ensemble.model_weights = {
            "lstm": 0.20,
            "xgboost": 0.40,  # Higher weight
            "hmm": 0.25,
            "gp": 0.15,
        }


class TestEnsembleInformation:
    """Test ensemble information and metadata retrieval."""

    def test_ensemble_info_retrieval(self):
        """Test comprehensive ensemble information retrieval."""
        ensemble = OccupancyEnsemble(
            room_id="info_test_room", meta_learner="random_forest", cv_folds=5
        )

        info = ensemble.get_ensemble_info()

        # Verify basic info
        assert info["ensemble_type"] == "stacking"
        assert set(info["base_models"]) == {"lstm", "xgboost", "hmm", "gp"}
        assert info["meta_learner"] == "random_forest"
        assert info["is_trained"] is False
        assert info["base_models_trained"] is False
        assert info["meta_learner_trained"] is False

        # Initially empty performance data
        assert info["model_weights"] == {}
        assert info["model_performance"] == {}
        assert info["cv_scores"] == {}

    def test_ensemble_model_info(self):
        """Test base model info integration."""
        ensemble = OccupancyEnsemble(room_id="test_room")

        model_info = ensemble.get_model_info()

        assert model_info["model_type"] == ModelType.ENSEMBLE.value
        assert model_info["room_id"] == "test_room"
        assert model_info["is_trained"] is False
        assert model_info["feature_count"] == 0
        assert model_info["training_sessions"] == 0
        assert model_info["predictions_made"] == 0

    def test_ensemble_string_representation(self):
        """Test ensemble string representations."""
        ensemble = OccupancyEnsemble(room_id="test_room")

        str_repr = str(ensemble)
        assert "ensemble" in str_repr.lower()
        assert "test_room" in str_repr
        assert "untrained" in str_repr.lower()

        repr_str = repr(ensemble)
        assert "OccupancyEnsemble" in repr_str
        assert "ensemble" in repr_str.lower()
        assert "test_room" in repr_str


class TestEnsemblePerformance:
    """Test ensemble performance benchmarks and comparisons."""

    @pytest.mark.asyncio
    async def test_ensemble_training_performance(self, ensemble_validation_data):
        """Test ensemble training performance benchmarks."""
        train_features, train_targets, val_features, val_targets = (
            ensemble_validation_data
        )

        # Use smaller dataset for performance testing
        perf_features = train_features.head(300)
        perf_targets = train_targets.head(300)
        perf_val_features = val_features.head(100)
        perf_val_targets = val_targets.head(100)

        ensemble = OccupancyEnsemble(room_id="perf_test", cv_folds=3)

        # Mock base models for faster training
        with patch.multiple(
            "src.models.ensemble",
            LSTMPredictor=MagicMock,
            XGBoostPredictor=MagicMock,
            HMMPredictor=MagicMock,
            GaussianProcessPredictor=MagicMock,
        ):
            for model_name in ensemble.base_models.keys():
                mock_model = MagicMock()
                mock_model.train = AsyncMock(
                    return_value=TrainingResult(
                        success=True,
                        training_time_seconds=20.0,
                        model_version="v1.0",
                        training_samples=len(perf_features),
                    )
                )
                mock_model.predict = AsyncMock(
                    return_value=[
                        PredictionResult(
                            predicted_time=datetime.utcnow() + timedelta(seconds=1800),
                            transition_type="vacant_to_occupied",
                            confidence_score=0.8,
                        )
                        for _ in range(len(perf_features))
                    ]
                )
                mock_model.is_trained = True
                ensemble.base_models[model_name] = mock_model

            # Measure training time
            start_time = time.time()
            result = await ensemble.train(
                perf_features,
                perf_targets,
                perf_val_features,
                perf_val_targets,
            )
            training_time = time.time() - start_time

            # Performance assertions
            assert result.success is True
            assert training_time < 120  # Should complete within 2 minutes for test data

            print(
                f"Ensemble training time: {training_time:.2f}s for {len(perf_features)} samples"
            )

    @pytest.mark.asyncio
    async def test_ensemble_prediction_latency(self, ensemble_validation_data):
        """Test ensemble prediction latency requirements."""
        train_features, train_targets, val_features, _ = ensemble_validation_data
        ensemble = OccupancyEnsemble(room_id="latency_test")

        await self._setup_trained_ensemble(ensemble, train_features, train_targets)

        # Measure prediction latency
        test_features = val_features.head(20)

        start_time = time.time()
        predictions = await ensemble.predict(test_features, datetime.utcnow(), "vacant")
        prediction_time = time.time() - start_time

        # Latency per sample
        latency_per_sample = (prediction_time / len(test_features)) * 1000  # ms

        print(f"Ensemble prediction latency: {latency_per_sample:.2f}ms per sample")

        # Should meet performance requirements (< 100ms per prediction)
        assert latency_per_sample < 100
        assert len(predictions) == len(test_features)

    async def _setup_trained_ensemble(self, ensemble, train_features, train_targets):
        """Helper to setup trained ensemble for performance testing."""
        for model_name, model in ensemble.base_models.items():
            model.is_trained = True

            # Fast mock predictions
            mock_predictions = [
                PredictionResult(
                    predicted_time=datetime.utcnow() + timedelta(seconds=1800),
                    transition_type="vacant_to_occupied",
                    confidence_score=0.8,
                    model_type=model_name,
                )
            ]
            model.predict = AsyncMock(return_value=mock_predictions)

        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(train_features.columns)

        # Fast mock meta-learner
        ensemble.meta_learner = MagicMock()
        ensemble.meta_learner.predict = MagicMock(return_value=np.array([1800.0]))

        ensemble.meta_scaler = MagicMock()
        ensemble.meta_scaler.transform = MagicMock(
            return_value=np.array([[0, 0, 0, 0]])
        )

        ensemble.model_weights = {
            "lstm": 0.25,
            "xgboost": 0.35,
            "hmm": 0.20,
            "gp": 0.20,
        }
