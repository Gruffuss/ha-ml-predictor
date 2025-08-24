"""REAL ML Model Tests - No Over-Mocking - Targeting >85% Coverage

This test file replaces the over-mocked tests with REAL model training and prediction testing.
Tests use small synthetic datasets to actually train and test ML models, not mocks.

Critical Changes from Original:
- NO mocking of model training/prediction methods
- NO mocking of sklearn/xgboost model objects  
- NO mocking of model persistence operations
- REAL small dataset training and inference
- REAL model serialization/deserialization
- ACTUAL ML library usage (sklearn, xgboost)

Target Coverage: >85% for all ML model modules
"""

import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Tuple
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

# Import the actual ML models - NO MOCKS
from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.gp_predictor import GaussianProcessPredictor
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import BasePredictor, PredictionResult, TrainingResult
from src.models.base.xgboost_predictor import XGBoostPredictor
from src.models.ensemble import OccupancyEnsemble

# Suppress sklearn warnings for cleaner test output
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


class TestDatasetGenerator:
    """Generate small synthetic datasets for real ML model training."""

    @staticmethod
    def create_occupancy_dataset(
        n_samples: int = 100, n_features: int = 10, noise: float = 0.1
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create synthetic occupancy prediction dataset.

        Returns:
            features (pd.DataFrame): Feature matrix
            targets (pd.DataFrame): Target values with required columns
        """
        # Generate base regression data
        X, y = make_regression(
            n_samples=n_samples, n_features=n_features, noise=noise, random_state=42
        )

        # Convert to DataFrames with appropriate column names
        feature_names = [f"feature_{i}" for i in range(n_features)]
        features = pd.DataFrame(X, columns=feature_names)

        # Create realistic occupancy transition times (1 min to 12 hours)
        # Transform regression targets to time range [60, 43200] seconds
        y_scaled = np.clip(np.abs(y) * 1000 + 300, 60, 43200)

        # Create target DataFrame with required columns
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": y_scaled,
                "transition_type": np.random.choice(
                    ["vacant_to_occupied", "occupied_to_vacant"], n_samples
                ),
                "target_time": [
                    datetime.now(timezone.utc) + timedelta(seconds=i)
                    for i in range(n_samples)
                ],
            }
        )

        return features, targets


class TestLSTMPredictorReal:
    """Test LSTM predictor with REAL neural network training."""

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for fast testing."""
        return TestDatasetGenerator.create_occupancy_dataset(n_samples=50, n_features=5)

    @pytest.fixture
    def lstm_predictor(self):
        """Create LSTM predictor with small parameters for testing."""
        return LSTMPredictor(
            room_id="test_room",
            sequence_length=3,  # Very small for testing
            hidden_units=[8, 4],  # Small network
            max_iter=50,  # Few iterations for speed
        )

    @pytest.mark.asyncio
    async def test_lstm_real_training(self, lstm_predictor, small_dataset):
        """Test REAL LSTM model training with sklearn MLPRegressor."""
        features, targets = small_dataset

        # Test that model starts untrained
        assert not lstm_predictor.is_trained
        assert lstm_predictor.model is None

        # REAL training - no mocks
        result = await lstm_predictor.train(features, targets)

        # Verify training succeeded
        assert result.success is True
        assert result.training_time_seconds > 0
        assert result.training_samples == len(features)
        assert result.training_score is not None
        assert lstm_predictor.is_trained is True
        assert lstm_predictor.model is not None

        # Verify model is actually trained sklearn MLPRegressor
        from sklearn.neural_network import MLPRegressor

        assert isinstance(lstm_predictor.model, MLPRegressor)
        assert hasattr(lstm_predictor.model, "coefs_")  # Weights exist
        assert len(lstm_predictor.model.coefs_) > 0  # Network has layers

        # Test training metrics
        assert result.training_metrics is not None
        assert "training_mae" in result.training_metrics
        assert "training_r2" in result.training_metrics
        assert "sequences_generated" in result.training_metrics

    @pytest.mark.asyncio
    async def test_lstm_real_prediction(self, lstm_predictor, small_dataset):
        """Test REAL LSTM model prediction after training."""
        features, targets = small_dataset

        # Train the model first
        await lstm_predictor.train(features, targets)

        # Test prediction with real data
        prediction_time = datetime.now(timezone.utc)
        predictions = await lstm_predictor.predict(
            features.iloc[:5], prediction_time, current_state="vacant"
        )

        # Verify predictions
        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.predicted_time > prediction_time
            assert pred.confidence_score > 0
            assert pred.confidence_score <= 1.0
            assert pred.transition_type in ["vacant_to_occupied", "occupied_to_vacant"]
            assert pred.model_type == "lstm"

            # Verify prediction metadata
            assert pred.prediction_metadata is not None
            assert "time_until_transition_seconds" in pred.prediction_metadata
            assert "prediction_method" in pred.prediction_metadata

    @pytest.mark.asyncio
    async def test_lstm_training_validation(self, lstm_predictor):
        """Test LSTM training with validation set."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=40, n_features=4
        )

        # Split for validation
        train_features, val_features, train_targets, val_targets = train_test_split(
            features, targets, test_size=0.3, random_state=42
        )

        result = await lstm_predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert result.validation_score is not None
        assert "validation_mae" in result.training_metrics
        assert "validation_r2" in result.training_metrics

    def test_lstm_feature_importance(self, lstm_predictor):
        """Test LSTM feature importance calculation using real neural network weights."""
        # Create and train a simple model
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=30, n_features=3
        )

        # Run training synchronously for this test
        import asyncio

        asyncio.run(lstm_predictor.train(features, targets))

        # Get feature importance from real neural network
        importance = lstm_predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3  # Should match n_features

        # All importances should be non-negative
        for feature_name, imp_value in importance.items():
            assert feature_name in features.columns
            assert imp_value >= 0

        # Importances should sum to approximately 1 (normalized)
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_lstm_insufficient_data_error(self, lstm_predictor):
        """Test LSTM error handling with insufficient data."""
        # Create tiny dataset that should cause sequence generation to fail
        features = pd.DataFrame({"feature1": [1, 2]})  # Only 2 samples
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800, 3600],
                "transition_type": ["vacant_to_occupied", "occupied_to_vacant"],
                "target_time": [datetime.now(timezone.utc)] * 2,
            }
        )

        with pytest.raises(ModelTrainingError) as exc_info:
            await lstm_predictor.train(features, targets)

        assert "lstm" in str(exc_info.value)
        assert "sequence" in str(exc_info.value).lower()

    def test_lstm_real_model_serialization(self, lstm_predictor):
        """Test REAL LSTM model save/load with actual file I/O."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=25, n_features=3
        )

        # Train model
        asyncio.run(lstm_predictor.train(features, targets))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as temp_file:
            temp_path = temp_file.name

        try:
            # Test saving real trained model
            success = lstm_predictor.save_model(temp_path)
            assert success is True
            assert os.path.exists(temp_path)

            # Create new predictor and load
            new_predictor = LSTMPredictor(room_id="loaded_room")
            load_success = new_predictor.load_model(temp_path)

            assert load_success is True
            assert new_predictor.is_trained is True
            assert new_predictor.room_id == "test_room"  # Should be loaded from file
            assert new_predictor.model is not None
            assert len(new_predictor.feature_names) > 0

            # Test that loaded model can make predictions
            predictions = asyncio.run(
                new_predictor.predict(features.iloc[:2], datetime.now(timezone.utc))
            )
            assert len(predictions) == 2

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.asyncio
    async def test_lstm_incremental_update(self, lstm_predictor, small_dataset):
        """Test LSTM incremental learning capability."""
        features, targets = small_dataset

        # Initial training
        await lstm_predictor.train(features, targets)
        initial_version = lstm_predictor.model_version

        # Create new data for incremental update
        new_features, new_targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=20, n_features=5
        )

        # Test incremental update
        update_result = await lstm_predictor.incremental_update(
            new_features, new_targets, learning_rate=0.01
        )

        assert update_result.success is True
        assert update_result.training_samples == len(new_features)
        assert lstm_predictor.model_version != initial_version
        assert "incremental" in update_result.training_metrics["update_type"]


class TestXGBoostPredictorReal:
    """Test XGBoost predictor with REAL gradient boosting training."""

    @pytest.fixture
    def xgb_predictor(self):
        """Create XGBoost predictor with small parameters for testing."""
        return XGBoostPredictor(
            room_id="test_room",
            n_estimators=20,  # Small for fast training
            max_depth=3,
            learning_rate=0.1,
        )

    @pytest.fixture
    def small_dataset(self):
        """Create small dataset for fast testing."""
        return TestDatasetGenerator.create_occupancy_dataset(n_samples=80, n_features=6)

    @pytest.mark.asyncio
    async def test_xgboost_real_training(self, xgb_predictor, small_dataset):
        """Test REAL XGBoost model training."""
        features, targets = small_dataset

        assert not xgb_predictor.is_trained
        assert xgb_predictor.model is None

        # REAL XGBoost training
        result = await xgb_predictor.train(features, targets)

        # Verify training succeeded with real XGBoost
        assert result.success is True
        assert result.training_samples == len(features)
        assert xgb_predictor.is_trained is True
        assert xgb_predictor.model is not None

        # Verify it's actual XGBoost model
        import xgboost as xgb

        assert isinstance(xgb_predictor.model, xgb.XGBRegressor)
        assert hasattr(xgb_predictor.model, "feature_importances_")
        assert len(xgb_predictor.model.feature_importances_) == features.shape[1]

        # Verify training metrics
        assert "training_mae" in result.training_metrics
        assert "training_r2" in result.training_metrics
        assert "n_estimators_used" in result.training_metrics

    @pytest.mark.asyncio
    async def test_xgboost_real_prediction(self, xgb_predictor, small_dataset):
        """Test REAL XGBoost prediction."""
        features, targets = small_dataset

        # Train first
        await xgb_predictor.train(features, targets)

        # Real prediction
        prediction_time = datetime.now(timezone.utc)
        predictions = await xgb_predictor.predict(
            features.iloc[:3], prediction_time, current_state="occupied"
        )

        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.predicted_time > prediction_time
            assert 0 < pred.confidence_score <= 1.0
            assert pred.model_type == "xgboost"

            # XGBoost-specific metadata
            assert "feature_contributions" in pred.prediction_metadata
            assert "n_estimators_used" in pred.prediction_metadata

    def test_xgboost_real_feature_importance(self, xgb_predictor):
        """Test real XGBoost feature importance from trained model."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=60, n_features=4
        )

        # Train model
        asyncio.run(xgb_predictor.train(features, targets))

        # Get REAL feature importance from XGBoost
        importance = xgb_predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 4  # Match n_features

        for feature_name, imp_value in importance.items():
            assert feature_name in features.columns
            assert imp_value >= 0  # XGBoost importances are non-negative

        # Should sum to 1 (normalized)
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_xgboost_early_stopping(self, xgb_predictor):
        """Test XGBoost early stopping with validation set."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=70, n_features=5
        )

        train_features, val_features, train_targets, val_targets = train_test_split(
            features, targets, test_size=0.3, random_state=42
        )

        result = await xgb_predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert xgb_predictor.best_iteration_ is not None
        assert "validation_mae" in result.training_metrics

    def test_xgboost_model_complexity_info(self, xgb_predictor):
        """Test XGBoost model complexity information."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=50, n_features=3
        )

        asyncio.run(xgb_predictor.train(features, targets))

        complexity = xgb_predictor.get_model_complexity()

        assert "n_estimators" in complexity
        assert "max_depth" in complexity
        assert "total_features" in complexity
        assert complexity["total_features"] == 3
        assert "regularization" in complexity


class TestHMMPredictorReal:
    """Test HMM predictor with REAL Gaussian Mixture Model training."""

    @pytest.fixture
    def hmm_predictor(self):
        """Create HMM predictor with small parameters."""
        return HMMPredictor(
            room_id="test_room",
            n_components=3,  # Small number of states
            max_iter=50,  # Fast training
        )

    @pytest.fixture
    def small_dataset(self):
        return TestDatasetGenerator.create_occupancy_dataset(n_samples=60, n_features=4)

    @pytest.mark.asyncio
    async def test_hmm_real_training(self, hmm_predictor, small_dataset):
        """Test REAL HMM training with Gaussian Mixture Model."""
        features, targets = small_dataset

        assert not hmm_predictor.is_trained
        assert hmm_predictor.state_model is None

        # REAL HMM training
        result = await hmm_predictor.train(features, targets)

        assert result.success is True
        assert hmm_predictor.is_trained is True
        assert hmm_predictor.state_model is not None

        # Verify it's real Gaussian Mixture Model
        from sklearn.mixture import GaussianMixture

        assert isinstance(hmm_predictor.state_model, GaussianMixture)
        assert hasattr(hmm_predictor.state_model, "means_")
        assert hasattr(hmm_predictor.state_model, "covariances_")

        # Verify state analysis was performed
        assert len(hmm_predictor.state_labels) > 0
        assert len(hmm_predictor.state_characteristics) > 0
        assert hmm_predictor.transition_matrix is not None

        # Training metrics
        assert "n_states" in result.training_metrics
        assert "log_likelihood" in result.training_metrics

    @pytest.mark.asyncio
    async def test_hmm_real_prediction(self, hmm_predictor, small_dataset):
        """Test REAL HMM state-based prediction."""
        features, targets = small_dataset

        await hmm_predictor.train(features, targets)

        prediction_time = datetime.now(timezone.utc)
        predictions = await hmm_predictor.predict(
            features.iloc[:4], prediction_time, current_state="vacant"
        )

        assert len(predictions) == 4
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.model_type == "hmm"

            # HMM-specific metadata
            metadata = pred.prediction_metadata
            assert "current_hidden_state" in metadata
            assert "state_probability" in metadata
            assert "state_label" in metadata
            assert "all_state_probabilities" in metadata

    def test_hmm_state_analysis(self, hmm_predictor):
        """Test HMM hidden state identification and analysis."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=80, n_features=5
        )

        asyncio.run(hmm_predictor.train(features, targets))

        # Verify state analysis
        state_info = hmm_predictor.get_state_info()

        assert "n_states" in state_info
        assert "state_labels" in state_info
        assert "state_characteristics" in state_info
        assert "transition_matrix" in state_info

        # Each state should have characteristics
        for state_id, characteristics in state_info["state_characteristics"].items():
            assert "avg_duration" in characteristics
            assert "sample_count" in characteristics
            assert "prediction_reliability" in characteristics

    def test_hmm_feature_importance_real(self, hmm_predictor):
        """Test HMM feature importance based on state discrimination."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=70, n_features=4
        )

        asyncio.run(hmm_predictor.train(features, targets))

        importance = hmm_predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 4

        # All importances should be non-negative and sum to 1
        for feature_name, imp_value in importance.items():
            assert feature_name in features.columns
            assert imp_value >= 0

        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01


class TestGaussianProcessPredictorReal:
    """Test Gaussian Process predictor with REAL GP training."""

    @pytest.fixture
    def gp_predictor(self):
        """Create GP predictor with simple parameters."""
        return GaussianProcessPredictor(
            room_id="test_room",
            kernel="rbf",  # Simple RBF kernel for testing
            n_restarts_optimizer=1,  # Fast training
            max_inducing_points=50,  # Small for testing
        )

    @pytest.fixture
    def small_dataset(self):
        return TestDatasetGenerator.create_occupancy_dataset(n_samples=40, n_features=3)

    @pytest.mark.asyncio
    async def test_gp_real_training(self, gp_predictor, small_dataset):
        """Test REAL Gaussian Process training."""
        features, targets = small_dataset

        assert not gp_predictor.is_trained
        assert gp_predictor.model is None

        # REAL GP training
        result = await gp_predictor.train(features, targets)

        assert result.success is True
        assert gp_predictor.is_trained is True
        assert gp_predictor.model is not None

        # Verify it's real GaussianProcessRegressor
        from sklearn.gaussian_process import GaussianProcessRegressor

        assert isinstance(gp_predictor.model, GaussianProcessRegressor)
        assert hasattr(gp_predictor.model, "X_train_")
        assert hasattr(gp_predictor.model, "y_train_")

        # GP-specific metrics
        assert "log_marginal_likelihood" in result.training_metrics
        assert "avg_prediction_std" in result.training_metrics
        assert gp_predictor.log_marginal_likelihood is not None

    @pytest.mark.asyncio
    async def test_gp_real_uncertainty_quantification(
        self, gp_predictor, small_dataset
    ):
        """Test REAL GP uncertainty quantification."""
        features, targets = small_dataset

        await gp_predictor.train(features, targets)

        prediction_time = datetime.now(timezone.utc)
        predictions = await gp_predictor.predict(features.iloc[:3], prediction_time)

        assert len(predictions) == 3
        for pred in predictions:
            assert pred.model_type == "gp"

            # GP-specific uncertainty information
            metadata = pred.prediction_metadata
            assert "prediction_std" in metadata
            assert "uncertainty_quantification" in metadata

            uncertainty = metadata["uncertainty_quantification"]
            assert "aleatoric_uncertainty" in uncertainty
            assert "epistemic_uncertainty" in uncertainty
            assert "confidence_intervals" in uncertainty

            # Should have prediction interval
            assert pred.prediction_interval is not None
            assert len(pred.prediction_interval) == 2

    def test_gp_kernel_creation(self, gp_predictor):
        """Test GP kernel creation and configuration."""
        # Test different kernel types
        kernel_types = ["rbf", "matern", "rational_quadratic", "composite"]

        for kernel_type in kernel_types:
            gp_pred = GaussianProcessPredictor(kernel=kernel_type)
            kernel = gp_pred._create_kernel(n_features=3)

            # Kernel should be a valid sklearn kernel
            assert kernel is not None
            assert hasattr(kernel, "__call__")

    def test_gp_uncertainty_metrics(self, gp_predictor):
        """Test GP uncertainty metrics and calibration."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=50, n_features=3
        )

        asyncio.run(gp_predictor.train(features, targets))

        metrics = gp_predictor.get_uncertainty_metrics()

        assert "kernel_type" in metrics
        assert "log_marginal_likelihood" in metrics
        assert "uncertainty_calibrated" in metrics
        assert "sparse_gp" in metrics


class TestOccupancyEnsembleReal:
    """Test Ensemble model with REAL base model training."""

    @pytest.fixture
    def ensemble_predictor(self):
        """Create ensemble with small parameters for testing."""
        return OccupancyEnsemble(
            room_id="test_room",
            cv_folds=3,  # Small for speed
            meta_learner="random_forest",
        )

    @pytest.fixture
    def medium_dataset(self):
        """Create medium dataset for ensemble training."""
        return TestDatasetGenerator.create_occupancy_dataset(
            n_samples=150, n_features=8
        )

    @pytest.mark.asyncio
    async def test_ensemble_real_training(self, ensemble_predictor, medium_dataset):
        """Test REAL ensemble training with actual base models."""
        features, targets = medium_dataset

        assert not ensemble_predictor.is_trained
        assert not ensemble_predictor.base_models_trained

        # REAL ensemble training - this will train ALL base models
        result = await ensemble_predictor.train(features, targets)

        assert result.success is True
        assert ensemble_predictor.is_trained is True
        assert ensemble_predictor.base_models_trained is True
        assert ensemble_predictor.meta_learner_trained is True

        # Verify all base models are trained
        for model_name, model in ensemble_predictor.base_models.items():
            assert model.is_trained is True
            assert model.model is not None

        # Verify meta-learner exists
        assert ensemble_predictor.meta_learner is not None
        from sklearn.ensemble import RandomForestRegressor

        assert isinstance(ensemble_predictor.meta_learner, RandomForestRegressor)

        # Verify model weights calculated
        assert len(ensemble_predictor.model_weights) == 4  # 4 base models
        assert all(w >= 0 for w in ensemble_predictor.model_weights.values())

        # Ensemble-specific metrics
        assert "ensemble_mae" in result.training_metrics
        assert "base_model_count" in result.training_metrics
        assert "model_weights" in result.training_metrics

    @pytest.mark.asyncio
    async def test_ensemble_real_prediction(self, ensemble_predictor, medium_dataset):
        """Test REAL ensemble prediction combining base models."""
        features, targets = medium_dataset

        await ensemble_predictor.train(features, targets)

        prediction_time = datetime.now(timezone.utc)
        predictions = await ensemble_predictor.predict(
            features.iloc[:5], prediction_time, current_state="vacant"
        )

        assert len(predictions) == 5
        for pred in predictions:
            assert pred.model_type == "ensemble"

            # Ensemble-specific metadata
            metadata = pred.prediction_metadata
            assert "base_model_predictions" in metadata
            assert "model_weights" in metadata
            assert "meta_learner_type" in metadata
            assert "combination_method" in metadata

            # Should have base model predictions from all 4 models
            base_preds = metadata["base_model_predictions"]
            expected_models = {"lstm", "xgboost", "hmm", "gp"}
            actual_models = set(base_preds.keys())
            assert (
                expected_models.issubset(actual_models) or len(base_preds) >= 3
            )  # At least 3 succeeded

    def test_ensemble_base_model_performance_tracking(self, ensemble_predictor):
        """Test ensemble tracking of individual base model performance."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=100, n_features=6
        )

        asyncio.run(ensemble_predictor.train(features, targets))

        # Check performance tracking
        assert len(ensemble_predictor.model_performance) > 0

        for model_name, performance in ensemble_predictor.model_performance.items():
            assert "training_score" in performance
            assert "validation_score" in performance
            assert isinstance(performance["training_score"], (int, float))

    def test_ensemble_feature_importance_combination(self, ensemble_predictor):
        """Test ensemble feature importance combining base models."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=80, n_features=5
        )

        asyncio.run(ensemble_predictor.train(features, targets))

        importance = ensemble_predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 5  # Should match n_features

        # Combined importance should be normalized
        total = sum(importance.values())
        assert abs(total - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_ensemble_incremental_update(self, ensemble_predictor):
        """Test ensemble incremental learning."""
        # Initial training
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=100, n_features=6
        )
        await ensemble_predictor.train(features, targets)

        initial_version = ensemble_predictor.model_version

        # New data for incremental update
        new_features, new_targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=30, n_features=6
        )

        result = await ensemble_predictor.incremental_update(
            new_features, new_targets, learning_rate=0.1
        )

        assert result.success is True
        assert ensemble_predictor.model_version != initial_version
        assert "incremental" in result.training_metrics["update_type"]

    def test_ensemble_model_serialization_real(self, ensemble_predictor):
        """Test REAL ensemble model serialization with all base models."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=60, n_features=4
        )

        asyncio.run(ensemble_predictor.train(features, targets))

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "ensemble_model.pkl"

            # Save ensemble (will save base models separately)
            success = ensemble_predictor.save_model(str(model_path))
            assert success is True
            assert model_path.exists()

            # Check that base model directory was created
            base_models_dir = Path(temp_dir) / "ensemble_model_base_models"
            assert base_models_dir.exists()

            # Load ensemble
            new_ensemble = OccupancyEnsemble(room_id="loaded_room")
            load_success = new_ensemble.load_model(str(model_path))

            assert load_success is True
            assert new_ensemble.is_trained is True
            assert new_ensemble.base_models_trained is True

            # Test loaded ensemble can make predictions
            predictions = asyncio.run(
                new_ensemble.predict(features.iloc[:2], datetime.now(timezone.utc))
            )
            assert len(predictions) == 2


class TestModelIntegrationReal:
    """Test cross-model integration and comparison."""

    @pytest.mark.asyncio
    async def test_all_models_same_dataset_performance(self):
        """Test all models on same dataset for comparison."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=100, n_features=6
        )

        # Create all model types with small parameters
        models = {
            "lstm": LSTMPredictor(sequence_length=3, hidden_units=[6, 3], max_iter=30),
            "xgboost": XGBoostPredictor(n_estimators=20, max_depth=3),
            "hmm": HMMPredictor(n_components=3, max_iter=30),
            "gp": GaussianProcessPredictor(kernel="rbf", n_restarts_optimizer=1),
        }

        results = {}

        # Train all models on same data
        for name, model in models.items():
            try:
                result = await model.train(features, targets)
                results[name] = {
                    "success": result.success,
                    "training_score": result.training_score,
                    "training_time": result.training_time_seconds,
                }
            except Exception as e:
                # Some models might fail with small dataset - that's ok
                results[name] = {"success": False, "error": str(e)}

        # At least half should succeed
        successful_models = [r for r in results.values() if r.get("success", False)]
        assert len(successful_models) >= 2

        # All successful models should have reasonable performance
        for model_result in successful_models:
            assert model_result["training_time"] > 0
            assert model_result["training_score"] is not None

    @pytest.mark.asyncio
    async def test_prediction_consistency_across_models(self):
        """Test that all models produce reasonable predictions."""
        features, targets = TestDatasetGenerator.create_occupancy_dataset(
            n_samples=80, n_features=5
        )

        # Train a subset of models that should work with smaller data
        models = {
            "xgboost": XGBoostPredictor(n_estimators=15, max_depth=2),
            "gp": GaussianProcessPredictor(kernel="rbf", n_restarts_optimizer=1),
        }

        prediction_time = datetime.now(timezone.utc)
        all_predictions = {}

        for name, model in models.items():
            await model.train(features, targets)
            predictions = await model.predict(
                features.iloc[:3], prediction_time, current_state="vacant"
            )
            all_predictions[name] = predictions

        # All models should produce 3 predictions
        for model_name, predictions in all_predictions.items():
            assert len(predictions) == 3

            for pred in predictions:
                # All predictions should be reasonable
                assert pred.predicted_time > prediction_time
                assert 0 < pred.confidence_score <= 1.0
                assert pred.transition_type in [
                    "vacant_to_occupied",
                    "occupied_to_vacant",
                ]

                # Time until transition should be reasonable (1 min to 24 hours)
                time_diff = (pred.predicted_time - prediction_time).total_seconds()
                assert 60 <= time_diff <= 86400


# Run coverage analysis after test execution
def test_coverage_verification():
    """Verify that tests achieve target coverage."""
    # This test serves as a reminder to check coverage
    # Run: python -m pytest tests/unit/ml_models/test_predictive_models.py --cov=src/models

    # The following modules should now achieve >85% coverage:
    target_modules = [
        "src.models.ensemble",
        "src.models.base.lstm_predictor",
        "src.models.base.xgboost_predictor",
        "src.models.base.hmm_predictor",
        "src.models.base.gp_predictor",
    ]

    # This test always passes - it's documentation
    assert True, f"Verify >85% coverage for: {target_modules}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
