"""
Comprehensive unit tests for HMM predictor module.

This test suite provides complete coverage for the HMMPredictor class,
including model initialization, training methods, prediction methods,
state analysis, transition modeling, and error handling.
"""

from datetime import datetime, timedelta, timezone
import pickle
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


class TestHMMPredictorInitialization:
    """Test HMM predictor initialization and configuration."""

    def test_hmm_predictor_initialization_default(self):
        """Test HMM predictor initialization with default parameters."""
        predictor = HMMPredictor()

        assert predictor.model_type == ModelType.HMM
        assert predictor.room_id is None
        assert predictor.state_model is None
        assert predictor.is_trained is False
        assert isinstance(predictor.feature_scaler, StandardScaler)
        assert predictor.model_params["n_components"] == 4
        assert predictor.model_params["covariance_type"] == "full"

    def test_hmm_predictor_initialization_with_room_id(self):
        """Test HMM predictor initialization with room ID."""
        predictor = HMMPredictor(room_id="kitchen")

        assert predictor.model_type == ModelType.HMM
        assert predictor.room_id == "kitchen"
        assert predictor.is_trained is False

    def test_hmm_predictor_initialization_with_custom_params(self):
        """Test HMM predictor initialization with custom parameters."""
        custom_params = {
            "n_components": 6,
            "covariance_type": "diag",
            "n_iter": 200,
            "max_iter": 150,
            "tol": 1e-4,
            "random_state": 123,
        }

        predictor = HMMPredictor(room_id="bedroom", **custom_params)

        assert predictor.model_params["n_components"] == 6
        assert predictor.model_params["n_states"] == 6  # Alias
        assert predictor.model_params["covariance_type"] == "diag"
        assert predictor.model_params["n_iter"] == 200
        assert predictor.model_params["max_iter"] == 150  # Should use n_iter value
        assert predictor.model_params["tol"] == 1e-4
        assert predictor.model_params["random_state"] == 123

    def test_hmm_predictor_parameter_aliases(self):
        """Test parameter aliases for compatibility."""
        predictor = HMMPredictor(n_components=8, max_iter=300)

        assert predictor.model_params["n_states"] == 8  # Alias for n_components
        assert predictor.model_params["n_iter"] == 100  # Default value
        assert predictor.model_params["max_iter"] == 300  # Custom value

    def test_hmm_predictor_default_parameters_merge(self):
        """Test that default parameters are properly merged with custom ones."""
        default_params = DEFAULT_MODEL_PARAMS[ModelType.HMM]
        custom_params = {"covariance_type": "tied"}

        predictor = HMMPredictor(**custom_params)

        # Custom parameter should override default
        assert predictor.model_params["covariance_type"] == "tied"

        # Default parameters should still be present
        assert predictor.model_params["n_components"] == default_params.get(
            "n_components", 4
        )

    def test_hmm_predictor_data_structures_initialization(self):
        """Test initialization of HMM-specific data structures."""
        predictor = HMMPredictor()

        assert isinstance(predictor.transition_models, dict)
        assert isinstance(predictor.state_labels, dict)
        assert isinstance(predictor.state_characteristics, dict)
        assert isinstance(predictor.state_durations, dict)
        assert predictor.transition_matrix is None


class TestHMMPredictorTraining:
    """Test HMM predictor training functionality."""

    def create_sample_training_data(self, n_samples=100, n_features=5):
        """Create sample training data for testing."""
        np.random.seed(42)

        # Create features with some structure for better state identification
        features = []
        for i in range(n_samples):
            # Create different "modes" of behavior
            if i < n_samples // 3:
                # Morning pattern
                feature_vec = np.random.normal([1, 0.5, 0, 0.8, 0.3], 0.2)
            elif i < 2 * n_samples // 3:
                # Evening pattern
                feature_vec = np.random.normal([0.2, 0.9, 0.7, 0.1, 0.8], 0.2)
            else:
                # Night pattern
                feature_vec = np.random.normal([0.1, 0.2, 0.1, 0.9, 0.1], 0.2)

            features.append(feature_vec)

        features_df = pd.DataFrame(features)
        # Ensure we only keep the requested number of features
        if n_features < len(features_df.columns):
            features_df = features_df.iloc[:, :n_features]
        features_df.columns = [f"feature_{i}" for i in range(features_df.shape[1])]

        # Create targets (time until next transition in seconds)
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )

        return features_df, targets

    @pytest.mark.asyncio
    async def test_hmm_training_success(self):
        """Test successful HMM training."""
        predictor = HMMPredictor(room_id="test_room", n_components=3, max_iter=10)
        features, targets = self.create_sample_training_data(100, 5)

        result = await predictor.train(features, targets)

        assert result.success is True
        assert result.training_samples == 100
        assert result.training_time_seconds > 0
        assert result.model_version is not None
        assert result.training_score is not None
        assert predictor.is_trained is True
        assert predictor.state_model is not None
        assert isinstance(predictor.state_model, GaussianMixture)

    @pytest.mark.asyncio
    async def test_hmm_training_with_validation_data(self):
        """Test HMM training with validation data."""
        predictor = HMMPredictor(room_id="test_room", n_components=3, max_iter=10)
        train_features, train_targets = self.create_sample_training_data(100, 5)
        val_features, val_targets = self.create_sample_training_data(30, 5)

        result = await predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert result.validation_score is not None
        assert "validation_mae" in result.training_metrics
        assert "validation_rmse" in result.training_metrics
        assert "validation_r2" in result.training_metrics

    @pytest.mark.asyncio
    async def test_hmm_training_insufficient_data_error(self):
        """Test HMM training with insufficient data raises error."""
        predictor = HMMPredictor(room_id="test_room", n_components=4)
        features, targets = self.create_sample_training_data(15, 3)  # Small dataset

        with pytest.raises(ModelTrainingError):
            await predictor.train(features, targets)

    @pytest.mark.asyncio
    async def test_hmm_training_state_analysis(self):
        """Test that HMM training performs proper state analysis."""
        predictor = HMMPredictor(room_id="test_room", n_components=3, max_iter=10)
        features, targets = self.create_sample_training_data(100, 5)

        await predictor.train(features, targets)

        # Check that state analysis was performed
        assert len(predictor.state_labels) > 0
        assert len(predictor.state_characteristics) > 0
        assert len(predictor.state_durations) > 0
        assert predictor.transition_matrix is not None
        assert predictor.transition_matrix.shape == (3, 3)

    @pytest.mark.asyncio
    async def test_hmm_training_transition_models(self):
        """Test that transition models are created for each state."""
        predictor = HMMPredictor(room_id="test_room", n_components=3, max_iter=10)
        features, targets = self.create_sample_training_data(100, 5)

        await predictor.train(features, targets)

        # Check that transition models were created
        assert len(predictor.transition_models) > 0

        # Each state should have a transition model
        for state_id in range(3):
            if state_id in predictor.transition_models:
                model_info = predictor.transition_models[state_id]
                assert "type" in model_info
                assert model_info["type"] in ["average", "regression"]

    @pytest.mark.asyncio
    async def test_hmm_training_metrics_calculation(self):
        """Test that training metrics are calculated correctly."""
        predictor = HMMPredictor(room_id="test_room", n_components=3, max_iter=10)
        features, targets = self.create_sample_training_data(100, 5)

        result = await predictor.train(features, targets)

        metrics = result.training_metrics
        assert "training_mae" in metrics
        assert "training_rmse" in metrics
        assert "training_r2" in metrics
        assert "n_states" in metrics
        assert "convergence_iter" in metrics
        assert "log_likelihood" in metrics
        assert "state_distribution" in metrics
        assert metrics["n_states"] == 3
        assert len(metrics["state_distribution"]) == 3

    @pytest.mark.asyncio
    async def test_hmm_training_with_kmeans_initialization(self):
        """Test HMM training uses KMeans for better initialization."""
        predictor = HMMPredictor(room_id="test_room", n_components=4, max_iter=10)
        features, targets = self.create_sample_training_data(100, 5)

        with patch("src.models.base.hmm_predictor.KMeans") as mock_kmeans:
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.cluster_centers_ = np.random.randn(4, 5)
            mock_kmeans.return_value = mock_kmeans_instance

            await predictor.train(features, targets)

            # Verify KMeans was used for initialization
            mock_kmeans.assert_called_once()
            mock_kmeans_instance.fit.assert_called_once()


class TestHMMPredictorStateAnalysis:
    """Test HMM state analysis functionality."""

    def create_mock_trained_predictor(self):
        """Create a mock trained HMM predictor."""
        predictor = HMMPredictor(n_components=3)
        predictor.is_trained = True
        predictor.feature_names = ["feature_0", "feature_1", "feature_2"]
        predictor.model_version = "test_v1.0"

        # Mock state model
        predictor.state_model = Mock(spec=GaussianMixture)
        predictor.state_model.means_ = np.array(
            [
                [0.1, 0.2, 0.3],  # State 0
                [0.5, 0.6, 0.7],  # State 1
                [0.8, 0.9, 0.1],  # State 2
            ]
        )
        predictor.state_model.covariances_ = np.array(
            [
                np.eye(3) * 0.1,  # State 0 covariance
                np.eye(3) * 0.2,  # State 1 covariance
                np.eye(3) * 0.15,  # State 2 covariance
            ]
        )
        predictor.state_model.predict.return_value = np.array([0])
        predictor.state_model.predict_proba.return_value = np.array([[0.8, 0.15, 0.05]])

        # Mock transition matrix
        predictor.transition_matrix = np.array(
            [[0.7, 0.2, 0.1], [0.3, 0.5, 0.2], [0.4, 0.3, 0.3]]
        )

        # Mock state characteristics
        predictor.state_labels = {
            0: "Short_Stay_0",
            1: "Medium_Stay_1",
            2: "Long_Stay_2",
        }
        predictor.state_characteristics = {
            0: {"avg_duration": 600, "sample_count": 20},
            1: {"avg_duration": 1800, "sample_count": 30},
            2: {"avg_duration": 3600, "sample_count": 25},
        }

        # Mock transition models
        predictor.transition_models = {
            0: {"type": "average", "value": 600},
            1: {"type": "regression", "model": Mock(spec=LinearRegression)},
            2: {"type": "average", "value": 3600},
        }
        predictor.transition_models[1]["model"].predict.return_value = np.array([1800])

        return predictor

    def test_state_label_assignment(self):
        """Test state label assignment based on characteristics."""
        predictor = HMMPredictor()

        # Test different duration ranges (check actual thresholds from implementation)
        label1 = predictor._assign_state_label(
            0, 300, np.array([1, 2, 3]), ["f1", "f2", "f3"]
        )
        assert "Quick_Transition" in label1

        label2 = predictor._assign_state_label(
            1, 1200, np.array([1, 2, 3]), ["f1", "f2", "f3"]
        )
        assert "Short_Stay" in label2

        label3 = predictor._assign_state_label(
            2, 7200, np.array([1, 2, 3]), ["f1", "f2", "f3"]
        )  # 2 hours
        assert "Medium_Stay" in label3

        label4 = predictor._assign_state_label(
            3, 18000, np.array([1, 2, 3]), ["f1", "f2", "f3"]
        )
        assert "Long_Stay" in label4

    def test_build_transition_matrix(self):
        """Test transition matrix building from state sequences."""
        predictor = HMMPredictor(n_components=3)
        predictor.state_labels = {0: "State_0", 1: "State_1", 2: "State_2"}

        # Create a sequence of state labels
        state_labels = np.array([0, 0, 1, 1, 1, 2, 0, 1, 2, 2])

        predictor._build_transition_matrix(state_labels)

        assert predictor.transition_matrix is not None
        assert predictor.transition_matrix.shape == (3, 3)

        # Check that rows sum to 1 (probability distributions)
        for i in range(3):
            assert abs(predictor.transition_matrix[i, :].sum() - 1.0) < 1e-6

    def test_analyze_states_with_probabilities(self):
        """Test state analysis with state probabilities."""
        predictor = HMMPredictor(n_components=3)

        X = np.random.randn(50, 4)
        state_labels = np.random.randint(0, 3, 50)
        durations = np.random.uniform(300, 7200, 50)
        feature_names = ["f1", "f2", "f3", "f4"]

        # Create mock state probabilities
        state_probabilities = np.random.uniform(0.3, 0.9, (50, 3))
        # Normalize each row
        state_probabilities = state_probabilities / state_probabilities.sum(
            axis=1, keepdims=True
        )

        predictor._analyze_states(
            X, state_labels, durations, feature_names, state_probabilities
        )

        # Check that analysis produced results
        assert len(predictor.state_labels) > 0
        assert len(predictor.state_characteristics) > 0

        # Check that state characteristics include probability-based metrics
        for state_id, characteristics in predictor.state_characteristics.items():
            assert "avg_state_probability" in characteristics
            assert "confidence_variance" in characteristics
            assert "prediction_reliability" in characteristics

    def test_get_state_info(self):
        """Test getting state information summary."""
        predictor = self.create_mock_trained_predictor()

        state_info = predictor.get_state_info()

        assert "n_states" in state_info
        assert "state_labels" in state_info
        assert "state_characteristics" in state_info
        assert "transition_matrix" in state_info
        assert state_info["n_states"] == 3
        assert len(state_info["state_labels"]) == 3
        assert len(state_info["transition_matrix"]) == 3


class TestHMMPredictorPrediction:
    """Test HMM predictor prediction functionality."""

    def create_mock_trained_predictor(self):
        """Create a mock trained HMM predictor for testing."""
        predictor = HMMPredictor(room_id="test_room", n_components=3)
        predictor.is_trained = True
        predictor.feature_names = ["feature_0", "feature_1", "feature_2"]
        predictor.model_version = "test_v1.0"

        # Mock feature scaler
        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.feature_scaler.transform.return_value = np.random.randn(3, 3)

        # Mock state model
        predictor.state_model = Mock(spec=GaussianMixture)
        predictor.state_model.predict_proba.return_value = np.array([[0.7, 0.2, 0.1]])

        # Mock transition matrix
        predictor.transition_matrix = np.array(
            [[0.6, 0.3, 0.1], [0.2, 0.5, 0.3], [0.4, 0.2, 0.4]]
        )

        # Mock state characteristics
        predictor.state_labels = {
            0: "Short_Stay_0",
            1: "Medium_Stay_1",
            2: "Long_Stay_2",
        }
        predictor.state_characteristics = {
            0: {"avg_duration": 600},
            1: {"avg_duration": 1800},
            2: {"avg_duration": 3600},
        }

        # Mock transition models
        predictor.transition_models = {
            0: {"type": "average", "value": 600},
            1: {"type": "regression", "model": Mock(spec=LinearRegression)},
            2: {"type": "average", "value": 3600},
        }
        predictor.transition_models[1]["model"].predict.return_value = np.array([1800])

        return predictor

    @pytest.mark.asyncio
    async def test_hmm_prediction_success(self):
        """Test successful HMM prediction."""
        predictor = self.create_mock_trained_predictor()
        features = pd.DataFrame(np.random.randn(3, 3), columns=predictor.feature_names)
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(features, prediction_time, "occupied")

        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.model_type == "hmm"
            assert pred.transition_type == "occupied_to_vacant"
            assert 0 <= pred.confidence_score <= 1
            assert pred.predicted_time > prediction_time

    @pytest.mark.asyncio
    async def test_hmm_prediction_untrained_model_error(self):
        """Test prediction with untrained model raises error."""
        predictor = HMMPredictor(room_id="test_room")
        features = pd.DataFrame(np.random.randn(5, 3))
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    @pytest.mark.asyncio
    async def test_hmm_prediction_invalid_features_error(self):
        """Test prediction with invalid features raises error."""
        predictor = self.create_mock_trained_predictor()
        predictor.validate_features = Mock(return_value=False)

        features = pd.DataFrame(np.random.randn(5, 3))
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    @pytest.mark.asyncio
    async def test_hmm_prediction_metadata_content(self):
        """Test that prediction metadata contains HMM-specific information."""
        predictor = self.create_mock_trained_predictor()
        features = pd.DataFrame(np.random.randn(1, 3), columns=predictor.feature_names)
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(features, prediction_time)

        metadata = predictions[0].prediction_metadata
        assert "time_until_transition_seconds" in metadata
        assert "prediction_method" in metadata
        assert "current_hidden_state" in metadata
        assert "state_probability" in metadata
        assert "state_label" in metadata
        assert "all_state_probabilities" in metadata
        assert "next_state_probabilities" in metadata
        assert metadata["prediction_method"] == "hidden_markov_model"

    @pytest.mark.asyncio
    async def test_hmm_prediction_transition_type_determination(self):
        """Test transition type determination from states and occupancy."""
        predictor = self.create_mock_trained_predictor()
        features = pd.DataFrame(np.random.randn(1, 3), columns=predictor.feature_names)
        prediction_time = datetime.now(timezone.utc)

        # Test occupied state
        predictions = await predictor.predict(features, prediction_time, "occupied")
        assert predictions[0].transition_type == "occupied_to_vacant"

        # Test vacant state
        predictions = await predictor.predict(features, prediction_time, "vacant")
        assert predictions[0].transition_type == "vacant_to_occupied"

        # Test unknown state with heuristics
        predictions = await predictor.predict(features, prediction_time, "unknown")
        assert predictions[0].transition_type in [
            "occupied_to_vacant",
            "vacant_to_occupied",
        ]

    def test_predict_single_duration_methods(self):
        """Test single duration prediction for different model types."""
        predictor = self.create_mock_trained_predictor()
        X = np.random.randn(1, 3)

        # Test average model
        duration1 = predictor._predict_single_duration(X, 0)
        assert duration1 == 600  # From average model

        # Test regression model
        duration2 = predictor._predict_single_duration(X, 1)
        assert duration2 == 1800  # From mocked regression model

        # Test missing model (should return default)
        duration3 = predictor._predict_single_duration(X, 99)  # Non-existent state
        assert duration3 == 1800  # Default


class TestHMMPredictorDurationModels:
    """Test HMM duration prediction models."""

    def test_train_state_duration_models_sufficient_data(self):
        """Test training duration models with sufficient data."""
        predictor = HMMPredictor(n_components=3)

        X = np.random.randn(100, 4)
        state_labels = np.random.randint(0, 3, 100)
        durations = np.random.uniform(300, 7200, 100)
        feature_names = ["f1", "f2", "f3", "f4"]

        # Ensure each state has sufficient samples
        state_labels[:30] = 0
        state_labels[30:60] = 1
        state_labels[60:] = 2

        predictor._train_state_duration_models(
            X, state_labels, durations, feature_names
        )

        # Check that models were created for each state
        assert len(predictor.transition_models) == 3

        for state_id in range(3):
            assert state_id in predictor.transition_models
            model_info = predictor.transition_models[state_id]
            assert model_info["type"] == "regression"
            assert "model" in model_info

    def test_train_state_duration_models_insufficient_data(self):
        """Test training duration models with insufficient data per state."""
        predictor = HMMPredictor(n_components=3)
        predictor.state_durations = {0: [600, 700], 1: [1800], 2: [3600, 3500, 3400]}

        X = np.random.randn(10, 4)
        state_labels = np.array([0, 0, 1, 2, 2, 2, 0, 1, 2, 1])  # Few samples per state
        durations = np.random.uniform(300, 7200, 10)
        feature_names = ["f1", "f2", "f3", "f4"]

        predictor._train_state_duration_models(
            X, state_labels, durations, feature_names
        )

        # Should create average models for states with insufficient data
        for state_id in range(3):
            if state_id in predictor.transition_models:
                model_info = predictor.transition_models[state_id]
                if np.sum(state_labels == state_id) < 5:
                    assert model_info["type"] == "average"
                    assert "value" in model_info

    def test_predict_durations_integration(self):
        """Test duration prediction integration with state identification."""
        predictor = HMMPredictor(n_components=2)

        # Mock state model
        predictor.state_model = Mock(spec=GaussianMixture)
        predictor.state_model.predict_proba.return_value = np.array(
            [
                [0.8, 0.2],  # First sample - strongly state 0
                [0.3, 0.7],  # Second sample - strongly state 1
            ]
        )

        # Mock transition models
        predictor.transition_models = {
            0: {"type": "average", "value": 900},
            1: {"type": "average", "value": 2700},
        }

        X = np.random.randn(2, 3)
        predictions = predictor._predict_durations(X)

        assert len(predictions) == 2
        # Note: The method uses argmax to select the most likely state
        # So both predictions will use state 0 (argmax([0.8, 0.2]) = 0, argmax([0.3, 0.7]) = 1)
        assert predictions[0] == 900  # State 0 duration (most likely for first sample)
        assert (
            predictions[1] == 2700
        )  # State 1 duration (most likely for second sample)


class TestHMMPredictorFeatureImportance:
    """Test HMM feature importance calculation."""

    def create_mock_trained_predictor_for_importance(self):
        """Create a mock trained predictor for feature importance testing."""
        predictor = HMMPredictor(n_components=3)
        predictor.is_trained = True
        predictor.feature_names = ["feature_0", "feature_1", "feature_2"]

        # Mock state model with means and covariances
        predictor.state_model = Mock(spec=GaussianMixture)
        predictor.state_model.means_ = np.array(
            [
                [0.1, 0.5, 0.9],  # State 0 - feature 2 is highest
                [0.8, 0.2, 0.3],  # State 1 - feature 0 is highest
                [0.4, 0.9, 0.1],  # State 2 - feature 1 is highest
            ]
        )

        # Full covariance matrices
        predictor.state_model.covariances_ = np.array(
            [
                [[0.1, 0.05, 0.02], [0.05, 0.2, 0.01], [0.02, 0.01, 0.15]],  # State 0
                [[0.2, 0.03, 0.04], [0.03, 0.1, 0.02], [0.04, 0.02, 0.25]],  # State 1
                [[0.15, 0.02, 0.01], [0.02, 0.3, 0.05], [0.01, 0.05, 0.1]],  # State 2
            ]
        )

        predictor.model_params = {"covariance_type": "full"}

        return predictor

    def test_feature_importance_calculation_full_covariance(self):
        """Test feature importance calculation with full covariance."""
        predictor = self.create_mock_trained_predictor_for_importance()

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(name in importance for name in predictor.feature_names)
        assert all(score >= 0 for score in importance.values())

        # Should sum to approximately 1 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_feature_importance_calculation_diag_covariance(self):
        """Test feature importance calculation with diagonal covariance."""
        predictor = self.create_mock_trained_predictor_for_importance()
        predictor.model_params["covariance_type"] = "diag"

        # Mock diagonal covariances
        predictor.state_model.covariances_ = np.array(
            [
                [0.1, 0.2, 0.15],  # State 0 diagonal
                [0.2, 0.1, 0.25],  # State 1 diagonal
                [0.15, 0.3, 0.1],  # State 2 diagonal
            ]
        )

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3
        assert all(score >= 0 for score in importance.values())

    def test_feature_importance_calculation_tied_covariance(self):
        """Test feature importance calculation with tied/spherical covariance."""
        predictor = self.create_mock_trained_predictor_for_importance()
        predictor.model_params["covariance_type"] = "tied"

        # Mock tied covariance (single matrix for all states)
        predictor.state_model.covariances_ = np.array(
            [[0.2, 0.05, 0.03], [0.05, 0.25, 0.02], [0.03, 0.02, 0.15]]
        )

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3

    def test_feature_importance_untrained_model(self):
        """Test feature importance for untrained model."""
        predictor = HMMPredictor()

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_feature_importance_missing_attributes(self):
        """Test feature importance when model lacks required attributes."""
        predictor = HMMPredictor()
        predictor.is_trained = True
        predictor.state_model = Mock(spec=GaussianMixture)
        # Missing means_ and covariances_ attributes

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_feature_importance_exception_handling(self):
        """Test feature importance calculation with exceptions."""
        predictor = self.create_mock_trained_predictor_for_importance()

        # Make means_ raise an exception
        predictor.state_model.means_ = Mock(side_effect=Exception("Test error"))

        with patch("src.models.base.hmm_predictor.logger") as mock_logger:
            importance = predictor.get_feature_importance()

            assert importance == {}
            mock_logger.warning.assert_called_once()


class TestHMMPredictorConfidenceCalculation:
    """Test HMM confidence calculation methods."""

    def test_confidence_calculation_entropy_based(self):
        """Test confidence calculation based on state probability entropy."""
        predictor = HMMPredictor()

        # High confidence case (low entropy)
        high_conf_probs = np.array([0.9, 0.05, 0.05])
        confidence1 = predictor._calculate_confidence(0.9, 1800, high_conf_probs)

        # Low confidence case (high entropy)
        low_conf_probs = np.array([0.4, 0.3, 0.3])
        confidence2 = predictor._calculate_confidence(0.4, 1800, low_conf_probs)

        assert (
            confidence1 > confidence2
        )  # High certainty should yield higher confidence
        assert 0.1 <= confidence1 <= 0.95
        assert 0.1 <= confidence2 <= 0.95

    def test_confidence_calculation_duration_adjustment(self):
        """Test confidence adjustment based on predicted duration."""
        predictor = HMMPredictor()
        base_probs = np.array([0.7, 0.2, 0.1])

        # Reasonable duration
        normal_confidence = predictor._calculate_confidence(0.7, 1800, base_probs)

        # Extreme durations should reduce confidence
        short_confidence = predictor._calculate_confidence(
            0.7, 100, base_probs
        )  # Too short
        long_confidence = predictor._calculate_confidence(
            0.7, 50000, base_probs
        )  # Too long

        assert short_confidence < normal_confidence * 0.85  # Should be reduced
        assert long_confidence < normal_confidence * 0.85  # Should be reduced

    def test_confidence_calculation_bounds(self):
        """Test that confidence calculation respects bounds."""
        predictor = HMMPredictor()

        # Test various inputs to ensure bounds are respected
        test_cases = [
            (0.99, 1800, np.array([0.99, 0.005, 0.005])),  # Very high confidence
            (0.01, 1800, np.array([0.33, 0.33, 0.34])),  # Very low confidence
            (0.5, 60, np.array([0.6, 0.3, 0.1])),  # Short duration
        ]

        for state_conf, duration, probs in test_cases:
            confidence = predictor._calculate_confidence(state_conf, duration, probs)
            assert 0.1 <= confidence <= 0.95  # Should always be within bounds


class TestHMMPredictorSerialization:
    """Test HMM model serialization and deserialization."""

    def create_trained_predictor_for_serialization(self):
        """Create a trained predictor for serialization testing."""
        predictor = HMMPredictor(room_id="test_room", n_components=3)
        predictor.is_trained = True
        predictor.feature_names = ["f1", "f2", "f3"]
        predictor.model_version = "test_v1.0"
        predictor.training_date = datetime.now(timezone.utc)

        # Mock state model
        predictor.state_model = Mock(spec=GaussianMixture)

        # Mock HMM-specific components
        predictor.transition_matrix = np.array([[0.7, 0.3], [0.4, 0.6]])
        predictor.state_labels = {0: "State_0", 1: "State_1"}
        predictor.state_characteristics = {
            0: {"avg_duration": 600, "sample_count": 50},
            1: {"avg_duration": 1800, "sample_count": 30},
        }
        predictor.transition_models = {
            0: {"type": "average", "value": 600},
            1: {"type": "average", "value": 1800},
        }

        # Add training history
        mock_result = TrainingResult(
            success=True,
            training_time_seconds=15.5,
            model_version="test_v1.0",
            training_samples=80,
            training_score=0.78,
        )
        predictor.training_history = [mock_result]

        return predictor

    def test_save_model_success(self):
        """Test successful HMM model saving."""
        predictor = self.create_trained_predictor_for_serialization()

        # Replace mock objects with actual serializable objects
        from sklearn.mixture import GaussianMixture
        from sklearn.preprocessing import StandardScaler

        predictor.state_model = GaussianMixture(n_components=2, max_iter=1)
        predictor.feature_scaler = StandardScaler()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            result = predictor.save_model(file_path)

            assert result is True

            # Verify file was created and contains expected data
            import os

            assert os.path.exists(file_path)

            with open(file_path, "rb") as f:
                saved_data = pickle.load(f)

            assert "state_model" in saved_data
            assert "transition_models" in saved_data
            assert "feature_scaler" in saved_data
            assert "transition_matrix" in saved_data
            assert "state_labels" in saved_data
            assert "state_characteristics" in saved_data
            assert saved_data["room_id"] == "test_room"

        finally:
            import os

            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_load_model_success(self):
        """Test successful HMM model loading."""
        original_predictor = self.create_trained_predictor_for_serialization()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            # Save the model
            original_predictor.save_model(file_path)

            # Load into a new predictor
            new_predictor = HMMPredictor()
            result = new_predictor.load_model(file_path)

            assert result is True
            assert new_predictor.room_id == "test_room"
            assert new_predictor.is_trained is True
            assert new_predictor.feature_names == ["f1", "f2", "f3"]
            assert new_predictor.model_version == "test_v1.0"
            assert new_predictor.state_labels == {0: "State_0", 1: "State_1"}
            assert len(new_predictor.training_history) == 1

        finally:
            import os

            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_save_model_failure(self):
        """Test HMM model saving failure handling."""
        predictor = self.create_trained_predictor_for_serialization()

        invalid_path = "/nonexistent/path/model.pkl"

        with patch("src.models.base.hmm_predictor.logger") as mock_logger:
            result = predictor.save_model(invalid_path)

            assert result is False
            mock_logger.error.assert_called_once()

    def test_load_model_failure(self):
        """Test HMM model loading failure handling."""
        predictor = HMMPredictor()

        nonexistent_path = "/nonexistent/model.pkl"

        with patch("src.models.base.hmm_predictor.logger") as mock_logger:
            result = predictor.load_model(nonexistent_path)

            assert result is False
            mock_logger.error.assert_called_once()


class TestHMMPredictorTargetPreparation:
    """Test HMM target preparation methods."""

    def test_prepare_targets_time_until_transition(self):
        """Test target preparation with time_until_transition_seconds column."""
        predictor = HMMPredictor()

        targets = pd.DataFrame(
            {"time_until_transition_seconds": [300, 1800, 7200, 50, 100000]}
        )

        prepared = predictor._prepare_targets(targets)

        # Should clip values to reasonable bounds
        assert np.all(prepared >= 60)  # Minimum 1 minute
        assert np.all(prepared <= 86400)  # Maximum 24 hours
        assert prepared[0] == 300
        assert prepared[1] == 1800
        assert prepared[2] == 7200
        assert prepared[3] == 60  # Clipped from 50
        assert prepared[4] == 86400  # Clipped from 100000

    def test_prepare_targets_datetime_columns(self):
        """Test target preparation with datetime columns."""
        predictor = HMMPredictor()

        base_time = datetime.now(timezone.utc)
        targets = pd.DataFrame(
            {
                "target_time": [base_time, base_time + timedelta(hours=1)],
                "next_transition_time": [
                    base_time + timedelta(minutes=30),
                    base_time + timedelta(hours=2),
                ],
            }
        )

        prepared = predictor._prepare_targets(targets)

        assert len(prepared) == 2
        assert prepared[0] == 1800  # 30 minutes in seconds
        assert prepared[1] == 3600  # 1 hour in seconds

    def test_prepare_targets_default_format(self):
        """Test target preparation with default format (first column)."""
        predictor = HMMPredictor()

        targets = pd.DataFrame([1200, 3600, 7200, 45, 120000])

        prepared = predictor._prepare_targets(targets)

        assert len(prepared) == 5
        assert np.all(prepared >= 60)
        assert np.all(prepared <= 86400)


class TestHMMPredictorErrorHandling:
    """Test HMM predictor error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_training_exception_handling(self):
        """Test training exception handling and error result creation."""
        predictor = HMMPredictor(room_id="test_room")

        # Create invalid data that will cause training to fail
        features = pd.DataFrame()  # Empty DataFrame
        targets = pd.DataFrame()

        with pytest.raises(ModelTrainingError):
            await predictor.train(features, targets)

        # Check that error result was added to training history
        assert len(predictor.training_history) == 1
        assert predictor.training_history[0].success is False
        assert predictor.training_history[0].error_message is not None

    @pytest.mark.asyncio
    async def test_prediction_exception_handling(self):
        """Test prediction exception handling."""
        predictor = HMMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.state_model = Mock(spec=GaussianMixture)
        predictor.feature_names = ["f1", "f2"]
        predictor.validate_features = Mock(return_value=True)

        # Make prediction raise an exception
        predictor.state_model.predict_proba.side_effect = Exception("Prediction failed")

        features = pd.DataFrame(np.random.randn(5, 2), columns=["f1", "f2"])
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    def test_edge_case_small_dataset(self):
        """Test behavior with very small datasets."""
        predictor = HMMPredictor(n_components=2)

        # Very small feature set
        X = np.random.randn(5, 2)
        state_labels = np.array([0, 1, 0, 1, 0])
        durations = np.array([600, 1200, 800, 1500, 700])
        feature_names = ["f1", "f2"]

        # Should handle gracefully
        predictor._train_state_duration_models(
            X, state_labels, durations, feature_names
        )

        # Should create average models due to insufficient data
        for state_id in [0, 1]:
            if state_id in predictor.transition_models:
                assert predictor.transition_models[state_id]["type"] == "average"

    def test_parameter_validation_edge_cases(self):
        """Test parameter validation with edge cases."""
        # Test with minimal components
        predictor1 = HMMPredictor(n_components=1)
        assert predictor1.model_params["n_components"] == 1

        # Test with maximum reasonable components
        predictor2 = HMMPredictor(n_components=20)
        assert predictor2.model_params["n_components"] == 20

        # Test with zero tolerance (should not break initialization)
        predictor3 = HMMPredictor(tol=0.0)
        assert predictor3.model_params["tol"] == 0.0


class TestHMMPredictorIntegration:
    """Integration tests for HMM predictor with realistic scenarios."""

    def create_realistic_state_data(self, n_samples=200):
        """Create realistic data with clear state patterns."""
        np.random.seed(42)

        features = []
        targets = []

        for i in range(n_samples):
            # Create three distinct behavioral patterns
            if i < n_samples // 3:
                # Quick transitions (bathroom visits, brief checks)
                feature_pattern = np.random.normal([0.2, 0.8, 0.1, 0.3], 0.15)
                target_duration = np.random.uniform(180, 900)  # 3-15 minutes
            elif i < 2 * n_samples // 3:
                # Medium stays (work, meals)
                feature_pattern = np.random.normal([0.7, 0.4, 0.8, 0.6], 0.2)
                target_duration = np.random.uniform(
                    1800, 5400
                )  # 30 minutes - 1.5 hours
            else:
                # Long stays (sleep, relaxation)
                feature_pattern = np.random.normal([0.9, 0.1, 0.2, 0.9], 0.1)
                target_duration = np.random.uniform(7200, 28800)  # 2-8 hours

            features.append(feature_pattern)
            targets.append(target_duration)

        features_df = pd.DataFrame(
            features,
            columns=[
                "occupancy_intensity",
                "motion_frequency",
                "door_activity",
                "time_of_day",
            ],
        )
        targets_df = pd.DataFrame({"time_until_transition_seconds": targets})

        return features_df, targets_df

    @pytest.mark.asyncio
    async def test_realistic_hmm_training_and_prediction(self):
        """Test complete HMM workflow with realistic state-based data."""
        predictor = HMMPredictor(
            room_id="living_room", n_components=3, max_iter=50, tol=1e-4
        )

        # Create realistic data with clear state patterns
        features, targets = self.create_realistic_state_data(200)

        # Train the model
        training_result = await predictor.train(features, targets)

        assert training_result.success is True
        assert predictor.is_trained is True

        # Verify state analysis was performed
        assert len(predictor.state_labels) == 3
        assert len(predictor.state_characteristics) == 3
        assert predictor.transition_matrix is not None

        # Test predictions
        test_features = features.sample(10)
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(test_features, prediction_time, "vacant")

        assert len(predictions) == 10
        assert all(isinstance(p, PredictionResult) for p in predictions)
        assert all(p.transition_type == "vacant_to_occupied" for p in predictions)

        # Test state information
        state_info = predictor.get_state_info()
        assert state_info["n_states"] == 3
        assert len(state_info["state_labels"]) == 3

        # Test feature importance
        importance = predictor.get_feature_importance()
        assert len(importance) == 4  # Number of features
        assert all(score >= 0 for score in importance.values())

    @pytest.mark.asyncio
    async def test_hmm_state_interpretation(self):
        """Test that HMM correctly interprets different behavioral states."""
        predictor = HMMPredictor(n_components=3, max_iter=30)
        features, targets = self.create_realistic_state_data(150)

        await predictor.train(features, targets)

        # Check that states have meaningful interpretations
        state_labels = list(predictor.state_labels.values())

        # Should have different types of stays
        label_types = set()
        for label in state_labels:
            if "Quick_Transition" in label:
                label_types.add("quick")
            elif "Short_Stay" in label:
                label_types.add("short")
            elif "Medium_Stay" in label:
                label_types.add("medium")
            elif "Long_Stay" in label:
                label_types.add("long")

        # Should identify at least 2 different types of patterns
        assert len(label_types) >= 2

        # Check that state characteristics reflect the patterns
        durations = [
            char["avg_duration"] for char in predictor.state_characteristics.values()
        ]

        # Should have a range of average durations
        assert max(durations) > min(durations) * 2  # At least 2x difference

    def test_hmm_memory_efficiency(self):
        """Test HMM memory efficiency with larger datasets."""
        predictor = HMMPredictor(n_components=4, max_iter=10)

        # Create larger dataset
        features, targets = self.create_realistic_state_data(1000)

        # Should handle larger datasets without issues
        # (This is more of a smoke test for memory usage)
        try:
            # Just test initialization and basic setup
            X_scaled = predictor.feature_scaler.fit_transform(features)
            y_prepared = predictor._prepare_targets(targets)

            assert len(X_scaled) == 1000
            assert len(y_prepared) == 1000

            # Test sequence creation (should not consume excessive memory)
            state_labels = np.random.randint(0, 4, 1000)
            predictor._build_transition_matrix(state_labels)

            assert predictor.transition_matrix.shape == (4, 4)

        except MemoryError:
            pytest.fail(
                "HMM predictor should handle reasonably large datasets efficiently"
            )
