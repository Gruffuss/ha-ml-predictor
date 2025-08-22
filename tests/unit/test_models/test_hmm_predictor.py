"""
Comprehensive unit tests for Hidden Markov Model predictor.

This test suite validates HMM model training, state identification, transition matrix
building, duration prediction, and all probabilistic state modeling functionality.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, PropertyMock, patch

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
        assert predictor.transition_models == {}
        assert isinstance(predictor.feature_scaler, StandardScaler)
        assert predictor.state_labels == {}
        assert predictor.state_characteristics == {}
        assert predictor.transition_matrix is None
        assert predictor.state_durations == {}

        # Check default parameters
        expected_defaults = DEFAULT_MODEL_PARAMS[ModelType.HMM]
        assert predictor.model_params["n_components"] == expected_defaults.get(
            "n_components", 4
        )
        assert predictor.model_params["covariance_type"] == expected_defaults.get(
            "covariance_type", "full"
        )
        assert predictor.model_params["n_iter"] == expected_defaults.get("n_iter", 100)

    def test_hmm_predictor_initialization_with_room_id(self):
        """Test HMM predictor initialization with room ID."""
        predictor = HMMPredictor(room_id="bedroom")

        assert predictor.room_id == "bedroom"
        assert predictor.model_type == ModelType.HMM

    def test_hmm_predictor_initialization_with_custom_params(self):
        """Test HMM predictor initialization with custom parameters."""
        custom_params = {
            "n_components": 6,
            "covariance_type": "diag",
            "n_iter": 200,
            "random_state": 123,
            "init_params": "random",
            "tol": 1e-4,
        }

        predictor = HMMPredictor(room_id="kitchen", **custom_params)

        assert predictor.model_params["n_components"] == 6
        assert predictor.model_params["covariance_type"] == "diag"
        assert predictor.model_params["n_iter"] == 200
        assert predictor.model_params["random_state"] == 123
        assert predictor.model_params["init_params"] == "random"
        assert predictor.model_params["tol"] == 1e-4

    def test_hmm_predictor_parameter_aliasing(self):
        """Test parameter aliasing for compatibility."""
        predictor = HMMPredictor()

        # Test that n_states is aliased to n_components
        assert (
            predictor.model_params["n_states"] == predictor.model_params["n_components"]
        )

        # Test that max_iter is aliased to n_iter
        assert predictor.model_params["max_iter"] == predictor.model_params["n_iter"]


class TestHMMPredictorTraining:
    """Test HMM model training functionality."""

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data fixture for HMM."""
        np.random.seed(42)

        # Create realistic occupancy prediction features
        n_samples = 200
        features = pd.DataFrame(
            {
                "time_since_last_change": np.random.exponential(3600, n_samples),
                "current_state_duration": np.random.exponential(1800, n_samples),
                "hour_sin": np.sin(
                    2 * np.pi * np.random.randint(0, 24, n_samples) / 24
                ),
                "hour_cos": np.cos(
                    2 * np.pi * np.random.randint(0, 24, n_samples) / 24
                ),
                "day_of_week": np.random.randint(0, 7, n_samples),
                "temperature": np.random.normal(22, 3, n_samples),
                "motion_events": np.random.poisson(1.5, n_samples),
                "door_openings": np.random.poisson(0.5, n_samples),
            }
        )

        # Create bimodal target distribution (short and long stays)
        short_stays = np.random.exponential(600, n_samples // 2)  # ~10 min
        long_stays = np.random.exponential(3600, n_samples // 2)  # ~1 hour
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.concatenate([short_stays, long_stays])}
        )

        return features, targets

    @pytest.fixture
    def hmm_predictor(self):
        """HMM predictor fixture."""
        return HMMPredictor(room_id="test_room", random_state=42, n_components=4)

    @pytest.mark.asyncio
    async def test_hmm_training_success(self, hmm_predictor, sample_training_data):
        """Test successful HMM training."""
        features, targets = sample_training_data

        result = await hmm_predictor.train(features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.training_samples == len(features)
        assert result.training_time_seconds > 0
        assert result.validation_score is not None
        assert result.training_score > 0  # Should have reasonable R² score

        # Verify model is trained
        assert hmm_predictor.is_trained is True
        assert hmm_predictor.state_model is not None
        assert isinstance(hmm_predictor.state_model, GaussianMixture)
        assert hmm_predictor.feature_names == list(features.columns)
        assert len(hmm_predictor.state_labels) > 0
        assert len(hmm_predictor.state_characteristics) > 0
        assert hmm_predictor.transition_matrix is not None
        assert len(hmm_predictor.transition_models) > 0

    @pytest.mark.asyncio
    async def test_hmm_training_with_validation_data(
        self, hmm_predictor, sample_training_data
    ):
        """Test HMM training with separate validation data."""
        features, targets = sample_training_data

        # Split data for validation
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]

        result = await hmm_predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert result.training_samples == len(train_features)
        assert "validation_mae" in result.training_metrics
        assert "validation_r2" in result.training_metrics
        assert "validation_rmse" in result.training_metrics

    @pytest.mark.asyncio
    async def test_hmm_training_insufficient_data(self, hmm_predictor):
        """Test HMM training with insufficient data."""
        # Very small dataset
        features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [100, 200, 150, 300, 250]}
        )

        with pytest.raises(ModelTrainingError):
            await hmm_predictor.train(features, targets)

    @pytest.mark.asyncio
    async def test_hmm_training_kmeans_initialization(
        self, hmm_predictor, sample_training_data
    ):
        """Test HMM training uses KMeans for initialization."""
        features, targets = sample_training_data

        with patch("sklearn.cluster.KMeans") as mock_kmeans:
            mock_kmeans_instance = Mock()
            mock_kmeans_instance.fit.return_value = None
            mock_kmeans_instance.cluster_centers_ = np.random.randn(
                4, features.shape[1]
            )
            mock_kmeans.return_value = mock_kmeans_instance

            result = await hmm_predictor.train(features, targets)

            # Verify KMeans was used for initialization
            mock_kmeans.assert_called_once()
            mock_kmeans_instance.fit.assert_called_once()

    @pytest.mark.asyncio
    async def test_hmm_training_state_analysis(
        self, hmm_predictor, sample_training_data
    ):
        """Test that HMM training performs state analysis."""
        features, targets = sample_training_data

        result = await hmm_predictor.train(features, targets)

        # Verify state analysis was performed
        assert (
            len(hmm_predictor.state_labels)
            == hmm_predictor.model_params["n_components"]
        )
        assert (
            len(hmm_predictor.state_characteristics)
            == hmm_predictor.model_params["n_components"]
        )
        assert (
            len(hmm_predictor.state_durations)
            == hmm_predictor.model_params["n_components"]
        )

        # Check state characteristics structure
        for state_id, characteristics in hmm_predictor.state_characteristics.items():
            assert "avg_duration" in characteristics
            assert "std_duration" in characteristics
            assert "sample_count" in characteristics
            assert "feature_means" in characteristics
            assert "prediction_reliability" in characteristics

    @pytest.mark.asyncio
    async def test_hmm_training_transition_matrix_building(
        self, hmm_predictor, sample_training_data
    ):
        """Test transition matrix is built during training."""
        features, targets = sample_training_data

        result = await hmm_predictor.train(features, targets)

        # Verify transition matrix
        n_states = hmm_predictor.model_params["n_components"]
        assert hmm_predictor.transition_matrix.shape == (n_states, n_states)

        # Each row should sum to approximately 1.0 (probability distribution)
        for i in range(n_states):
            row_sum = np.sum(hmm_predictor.transition_matrix[i, :])
            assert abs(row_sum - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_hmm_training_duration_models(
        self, hmm_predictor, sample_training_data
    ):
        """Test duration prediction models are trained for each state."""
        features, targets = sample_training_data

        result = await hmm_predictor.train(features, targets)

        # Verify duration models
        assert len(hmm_predictor.transition_models) > 0

        for state_id, model_info in hmm_predictor.transition_models.items():
            assert "type" in model_info
            assert model_info["type"] in ["regression", "average"]

            if model_info["type"] == "regression":
                assert "model" in model_info
                assert isinstance(model_info["model"], LinearRegression)
            elif model_info["type"] == "average":
                assert "value" in model_info
                assert isinstance(model_info["value"], (int, float))

    @pytest.mark.asyncio
    async def test_hmm_training_metrics_calculation(
        self, hmm_predictor, sample_training_data
    ):
        """Test training metrics calculation."""
        features, targets = sample_training_data

        result = await hmm_predictor.train(features, targets)

        metrics = result.training_metrics
        assert "training_mae" in metrics
        assert "training_rmse" in metrics
        assert "training_r2" in metrics
        assert "n_states" in metrics
        assert "state_distribution" in metrics
        assert "convergence_iter" in metrics
        assert "log_likelihood" in metrics

        # Values should be reasonable
        assert metrics["training_mae"] > 0
        assert metrics["training_rmse"] > 0
        assert -1 <= metrics["training_r2"] <= 1
        assert metrics["n_states"] == hmm_predictor.model_params["n_components"]
        assert (
            len(metrics["state_distribution"])
            == hmm_predictor.model_params["n_components"]
        )

    @pytest.mark.asyncio
    async def test_hmm_training_error_handling(self, hmm_predictor):
        """Test HMM training error handling."""
        features = pd.DataFrame({"feature1": [1, 2, 3]})
        targets = pd.DataFrame({"invalid_target": [100, 200, 300]})  # Wrong column name

        with patch.object(
            hmm_predictor,
            "_prepare_targets",
            side_effect=Exception("Preparation error"),
        ):
            with pytest.raises(ModelTrainingError):
                await hmm_predictor.train(features, targets)


class TestHMMStateAnalysis:
    """Test HMM state analysis functionality."""

    @pytest.fixture
    def hmm_with_mock_state_model(self):
        """HMM predictor with mock state model."""
        predictor = HMMPredictor()

        # Mock trained state model
        mock_state_model = Mock(spec=GaussianMixture)
        mock_state_model.predict.return_value = np.array([0, 1, 2, 0, 1, 2, 3, 3])
        mock_state_model.predict_proba.return_value = np.array(
            [
                [0.8, 0.1, 0.05, 0.05],  # High confidence in state 0
                [0.1, 0.8, 0.05, 0.05],  # High confidence in state 1
                [0.05, 0.1, 0.8, 0.05],  # High confidence in state 2
                [0.7, 0.1, 0.1, 0.1],  # Medium confidence in state 0
                [0.1, 0.7, 0.1, 0.1],  # Medium confidence in state 1
                [0.1, 0.1, 0.7, 0.1],  # Medium confidence in state 2
                [0.1, 0.1, 0.1, 0.7],  # Medium confidence in state 3
                [0.2, 0.2, 0.2, 0.4],  # Low confidence in state 3
            ]
        )
        predictor.state_model = mock_state_model
        predictor.model_params = {"n_components": 4}

        return predictor

    def test_analyze_states(self, hmm_with_mock_state_model):
        """Test state analysis with mock data."""
        X = np.random.randn(8, 5)
        state_labels = np.array([0, 1, 2, 0, 1, 2, 3, 3])
        durations = np.array([600, 1800, 3600, 900, 2400, 4800, 1200, 1500])
        feature_names = ["f1", "f2", "f3", "f4", "f5"]
        state_probabilities = (
            hmm_with_mock_state_model.state_model.predict_proba.return_value
        )

        hmm_with_mock_state_model._analyze_states(
            X, state_labels, durations, feature_names, state_probabilities
        )

        # Verify state characteristics were computed
        assert len(hmm_with_mock_state_model.state_characteristics) == 4
        assert len(hmm_with_mock_state_model.state_labels) == 4
        assert len(hmm_with_mock_state_model.state_durations) == 4

        # Check state 0 characteristics
        state_0_char = hmm_with_mock_state_model.state_characteristics[0]
        assert "avg_duration" in state_0_char
        assert "std_duration" in state_0_char
        assert "sample_count" in state_0_char
        assert "feature_means" in state_0_char
        assert "avg_state_probability" in state_0_char
        assert "confidence_variance" in state_0_char
        assert "high_confidence_samples" in state_0_char
        assert "low_confidence_samples" in state_0_char
        assert "prediction_reliability" in state_0_char

        # State 0 has samples at indices 0 and 3
        assert state_0_char["sample_count"] == 2
        expected_avg_duration = np.mean([600, 900])
        assert abs(state_0_char["avg_duration"] - expected_avg_duration) < 1e-6

    def test_assign_state_label(self, hmm_with_mock_state_model):
        """Test state label assignment based on duration."""
        # Test different duration categories
        short_label = hmm_with_mock_state_model._assign_state_label(
            0, 300, np.zeros(5), []
        )
        assert "Quick_Transition" in short_label

        medium_label = hmm_with_mock_state_model._assign_state_label(
            1, 1800, np.zeros(5), []
        )
        assert "Short_Stay" in medium_label

        long_label = hmm_with_mock_state_model._assign_state_label(
            2, 7200, np.zeros(5), []
        )
        assert "Medium_Stay" in long_label

        very_long_label = hmm_with_mock_state_model._assign_state_label(
            3, 18000, np.zeros(5), []
        )
        assert "Long_Stay" in very_long_label

    def test_build_transition_matrix(self, hmm_with_mock_state_model):
        """Test transition matrix building."""
        # Sequential state transitions: 0->1->2->0->1->3->3
        state_labels = np.array([0, 1, 2, 0, 1, 3, 3])

        hmm_with_mock_state_model._build_transition_matrix(state_labels)

        transition_matrix = hmm_with_mock_state_model.transition_matrix
        assert transition_matrix.shape == (4, 4)

        # Check specific transitions
        # State 0 transitions: 0->1 (twice)
        assert transition_matrix[0, 1] == 1.0  # Always goes to state 1

        # State 1 transitions: 1->2, 1->3
        assert transition_matrix[1, 2] == 0.5  # 50% to state 2
        assert transition_matrix[1, 3] == 0.5  # 50% to state 3

        # State 3 transitions: 3->3 (self-transition)
        assert transition_matrix[3, 3] == 1.0  # Always stays in state 3

    def test_build_transition_matrix_no_transitions(self, hmm_with_mock_state_model):
        """Test transition matrix with no observed transitions from a state."""
        # Only one sample in state 0, no transitions
        state_labels = np.array([0])

        hmm_with_mock_state_model._build_transition_matrix(state_labels)

        transition_matrix = hmm_with_mock_state_model.transition_matrix

        # Should have uniform distribution for states with no observed transitions
        for i in range(4):
            row_sum = np.sum(transition_matrix[i, :])
            assert abs(row_sum - 1.0) < 1e-6


class TestHMMDurationModeling:
    """Test HMM duration prediction modeling."""

    @pytest.fixture
    def hmm_for_duration_modeling(self):
        """HMM predictor for duration modeling tests."""
        predictor = HMMPredictor()
        predictor.model_params = {"n_components": 3}
        predictor.state_durations = {0: [600, 800, 900], 1: [1800, 2000], 2: []}
        return predictor

    def test_train_state_duration_models_sufficient_data(
        self, hmm_for_duration_modeling
    ):
        """Test duration model training with sufficient data."""
        X = np.random.randn(10, 5)
        state_labels = np.array(
            [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        )  # 5 samples each for states 0,1
        durations = np.array([600, 800, 900, 700, 750, 1800, 2000, 1900, 2100, 1950])
        feature_names = ["f1", "f2", "f3", "f4", "f5"]

        hmm_for_duration_modeling._train_state_duration_models(
            X, state_labels, durations, feature_names
        )

        # Should create regression models for states with sufficient data
        assert 0 in hmm_for_duration_modeling.transition_models
        assert 1 in hmm_for_duration_modeling.transition_models

        model_0 = hmm_for_duration_modeling.transition_models[0]
        model_1 = hmm_for_duration_modeling.transition_models[1]

        assert model_0["type"] == "regression"
        assert model_1["type"] == "regression"
        assert isinstance(model_0["model"], LinearRegression)
        assert isinstance(model_1["model"], LinearRegression)

    def test_train_state_duration_models_insufficient_data(
        self, hmm_for_duration_modeling
    ):
        """Test duration model training with insufficient data."""
        X = np.random.randn(4, 5)
        state_labels = np.array(
            [0, 1, 2, 2]
        )  # Only 1 sample for state 0, 1; 2 for state 2
        durations = np.array([600, 1800, 3000, 3200])
        feature_names = ["f1", "f2", "f3", "f4", "f5"]

        hmm_for_duration_modeling._train_state_duration_models(
            X, state_labels, durations, feature_names
        )

        # Should create average models for states with insufficient data
        for state_id in [0, 1, 2]:
            if state_id in hmm_for_duration_modeling.transition_models:
                model_info = hmm_for_duration_modeling.transition_models[state_id]
                if model_info["type"] == "average":
                    assert "value" in model_info
                    assert isinstance(model_info["value"], (int, float))

    def test_predict_durations_training(self, hmm_for_duration_modeling):
        """Test duration prediction during training."""
        # Mock state model
        mock_state_model = Mock()
        mock_state_model.predict_proba.return_value = np.array(
            [
                [0.8, 0.1, 0.1],  # Likely state 0
                [0.1, 0.8, 0.1],  # Likely state 1
                [0.1, 0.1, 0.8],  # Likely state 2
            ]
        )
        hmm_for_duration_modeling.state_model = mock_state_model

        # Mock transition models
        hmm_for_duration_modeling.transition_models = {
            0: {"type": "average", "value": 750.0},
            1: {"type": "regression", "model": Mock()},
            2: {"type": "average", "value": 2000.0},
        }

        # Mock regression model prediction
        hmm_for_duration_modeling.transition_models[1]["model"].predict.return_value = (
            np.array([1850.0])
        )

        X = np.random.randn(3, 5)
        predictions = hmm_for_duration_modeling._predict_durations(X)

        assert len(predictions) == 3
        assert predictions[0] == 750.0  # State 0 average
        assert predictions[1] == 1850.0  # State 1 regression
        assert predictions[2] == 2000.0  # State 2 average

    def test_predict_single_duration(self, hmm_for_duration_modeling):
        """Test single duration prediction."""
        # Test with average model
        hmm_for_duration_modeling.transition_models = {
            0: {"type": "average", "value": 1200.0}
        }

        X_point = np.random.randn(1, 5)
        duration = hmm_for_duration_modeling._predict_single_duration(X_point, 0)

        assert duration == 1200.0

        # Test with regression model
        mock_regression = Mock()
        mock_regression.predict.return_value = np.array([1500.0])
        hmm_for_duration_modeling.transition_models[1] = {
            "type": "regression",
            "model": mock_regression,
        }

        duration = hmm_for_duration_modeling._predict_single_duration(X_point, 1)

        assert duration == 1500.0

        # Test with missing model (default)
        duration = hmm_for_duration_modeling._predict_single_duration(X_point, 99)

        assert duration == 1800.0  # Default 30 minutes


class TestHMMPrediction:
    """Test HMM prediction functionality."""

    @pytest.fixture
    def trained_hmm_predictor(self, sample_training_data):
        """Trained HMM predictor fixture."""
        features, targets = sample_training_data
        predictor = HMMPredictor(room_id="test_room", random_state=42)

        # Mock training to avoid long setup
        predictor.is_trained = True
        predictor.feature_names = list(features.columns)
        predictor.training_date = datetime.now(timezone.utc)
        predictor.model_version = "v1.0"

        # Mock state model
        mock_state_model = Mock(spec=GaussianMixture)
        mock_state_model.predict_proba.return_value = np.array(
            [
                [0.1, 0.7, 0.1, 0.1],  # Likely state 1
                [0.2, 0.1, 0.6, 0.1],  # Likely state 2
            ]
        )
        predictor.state_model = mock_state_model

        # Mock feature scaler
        mock_scaler = Mock(spec=StandardScaler)
        mock_scaler.transform.return_value = features.iloc[:2].values
        predictor.feature_scaler = mock_scaler

        # Mock transition models
        predictor.transition_models = {
            1: {"type": "average", "value": 1800.0},
            2: {"type": "regression", "model": Mock()},
        }
        predictor.transition_models[2]["model"].predict.return_value = np.array(
            [2400.0]
        )

        # Mock transition matrix
        predictor.transition_matrix = np.array(
            [
                [0.1, 0.6, 0.2, 0.1],
                [0.2, 0.3, 0.3, 0.2],
                [0.1, 0.2, 0.5, 0.2],
                [0.3, 0.2, 0.2, 0.3],
            ]
        )

        # Mock state labels and characteristics
        predictor.state_labels = {
            0: "Quick_Transition_0",
            1: "Short_Stay_1",
            2: "Medium_Stay_2",
            3: "Long_Stay_3",
        }
        predictor.state_characteristics = {
            1: {"avg_duration": 1800, "prediction_reliability": "high"},
            2: {"avg_duration": 2400, "prediction_reliability": "medium"},
        }

        return predictor

    @pytest.fixture
    def sample_prediction_features(self):
        """Sample prediction features fixture."""
        return pd.DataFrame(
            {
                "time_since_last_change": [1800, 7200],
                "current_state_duration": [900, 3600],
                "hour_sin": [0.5, -0.5],
                "hour_cos": [0.866, -0.866],
                "day_of_week": [1, 5],
                "temperature": [22.5, 20.0],
                "motion_events": [3, 0],
                "door_openings": [1, 0],
            }
        )

    @pytest.mark.asyncio
    async def test_hmm_prediction_success(
        self, trained_hmm_predictor, sample_prediction_features
    ):
        """Test successful HMM prediction."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time, "occupied"
        )

        assert len(results) == len(sample_prediction_features)

        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.predicted_time > prediction_time
            assert result.transition_type in [
                "occupied_to_vacant",
                "vacant_to_occupied",
            ]
            assert 0.1 <= result.confidence_score <= 0.95
            assert result.model_type == ModelType.HMM.value
            assert result.model_version == "v1.0"
            assert result.features_used == trained_hmm_predictor.feature_names

            # Check HMM-specific metadata
            metadata = result.prediction_metadata
            assert "time_until_transition_seconds" in metadata
            assert "prediction_method" in metadata
            assert metadata["prediction_method"] == "hidden_markov_model"
            assert "current_hidden_state" in metadata
            assert "state_probability" in metadata
            assert "state_label" in metadata
            assert "all_state_probabilities" in metadata

    @pytest.mark.asyncio
    async def test_hmm_prediction_untrained_model(self, sample_prediction_features):
        """Test prediction with untrained model."""
        predictor = HMMPredictor()
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(sample_prediction_features, prediction_time)

    @pytest.mark.asyncio
    async def test_hmm_prediction_invalid_features(self, trained_hmm_predictor):
        """Test prediction with invalid features."""
        invalid_features = pd.DataFrame({"wrong_feature": [1, 2, 3]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await trained_hmm_predictor.predict(invalid_features, prediction_time)

    @pytest.mark.asyncio
    async def test_hmm_prediction_state_identification(
        self, trained_hmm_predictor, sample_prediction_features
    ):
        """Test state identification in predictions."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time
        )

        for result in results:
            metadata = result.prediction_metadata

            # Check state identification
            assert "current_hidden_state" in metadata
            assert "state_probability" in metadata
            assert "state_label" in metadata

            current_state = metadata["current_hidden_state"]
            assert 0 <= current_state < 4

            state_prob = metadata["state_probability"]
            assert 0.0 <= state_prob <= 1.0

            # State label should correspond to current state
            expected_label = trained_hmm_predictor.state_labels.get(
                current_state, f"State_{current_state}"
            )
            assert metadata["state_label"] == expected_label

    @pytest.mark.asyncio
    async def test_hmm_prediction_transition_probabilities(
        self, trained_hmm_predictor, sample_prediction_features
    ):
        """Test transition probability inclusion in predictions."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time
        )

        for result in results:
            metadata = result.prediction_metadata

            # Should include next state probabilities
            assert "next_state_probabilities" in metadata

            next_probs = metadata["next_state_probabilities"]
            if next_probs is not None:
                assert len(next_probs) == 4  # Number of states
                assert all(0.0 <= p <= 1.0 for p in next_probs)
                assert abs(sum(next_probs) - 1.0) < 1e-6  # Should sum to 1

    @pytest.mark.asyncio
    async def test_hmm_prediction_confidence_calculation(
        self, trained_hmm_predictor, sample_prediction_features
    ):
        """Test confidence score calculation."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time
        )

        for result in results:
            confidence = result.confidence_score
            assert 0.1 <= confidence <= 0.95

            # Confidence should be related to state probability
            metadata = result.prediction_metadata
            state_prob = metadata["state_probability"]

            # Higher state probability should generally lead to higher confidence
            assert confidence > 0.1

    @pytest.mark.asyncio
    async def test_hmm_prediction_transition_type_determination(
        self, trained_hmm_predictor, sample_prediction_features
    ):
        """Test transition type determination from states."""
        prediction_time = datetime.now(timezone.utc)

        # Test with occupied state
        results_occupied = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time, "occupied"
        )

        for result in results_occupied:
            assert result.transition_type == "occupied_to_vacant"

        # Test with vacant state
        results_vacant = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time, "vacant"
        )

        for result in results_vacant:
            assert result.transition_type == "vacant_to_occupied"

        # Test with unknown state (should use heuristics)
        results_unknown = await trained_hmm_predictor.predict(
            sample_prediction_features, prediction_time, "unknown"
        )

        for result in results_unknown:
            assert result.transition_type in [
                "occupied_to_vacant",
                "vacant_to_occupied",
            ]


class TestHMMFeatureImportance:
    """Test HMM feature importance calculation."""

    @pytest.fixture
    def trained_hmm_with_features(self):
        """Trained HMM with known features and state model."""
        predictor = HMMPredictor()
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2", "feature3"]
        predictor.model_params = {"covariance_type": "full"}

        # Mock trained state model with GMM components
        mock_state_model = Mock(spec=GaussianMixture)
        mock_state_model.means_ = np.array(
            [
                [0.0, 1.0, 2.0],  # State 0 means
                [1.0, 0.0, 1.0],  # State 1 means
                [2.0, 2.0, 0.0],  # State 2 means
            ]
        )
        mock_state_model.covariances_ = np.array(
            [
                [
                    [1.0, 0.1, 0.1],
                    [0.1, 1.0, 0.1],
                    [0.1, 0.1, 1.0],
                ],  # State 0 covariance
                [
                    [0.5, 0.0, 0.0],
                    [0.0, 0.5, 0.0],
                    [0.0, 0.0, 0.5],
                ],  # State 1 covariance
                [
                    [2.0, 0.2, 0.2],
                    [0.2, 2.0, 0.2],
                    [0.2, 0.2, 2.0],
                ],  # State 2 covariance
            ]
        )
        predictor.state_model = mock_state_model

        return predictor

    def test_get_feature_importance_full_covariance(self, trained_hmm_with_features):
        """Test feature importance with full covariance matrices."""
        importance = trained_hmm_with_features.get_feature_importance()

        assert len(importance) == 3
        assert "feature1" in importance
        assert "feature2" in importance
        assert "feature3" in importance

        # Should sum to 1.0
        assert abs(sum(importance.values()) - 1.0) < 1e-6

        # All importance values should be positive
        for feature_importance in importance.values():
            assert feature_importance > 0

    def test_get_feature_importance_diagonal_covariance(
        self, trained_hmm_with_features
    ):
        """Test feature importance with diagonal covariance."""
        trained_hmm_with_features.model_params["covariance_type"] = "diag"

        # Update mock to return diagonal covariances
        trained_hmm_with_features.state_model.covariances_ = np.array(
            [
                [1.0, 1.0, 1.0],  # State 0 diagonal
                [0.5, 0.5, 0.5],  # State 1 diagonal
                [2.0, 2.0, 2.0],  # State 2 diagonal
            ]
        )

        importance = trained_hmm_with_features.get_feature_importance()

        assert len(importance) == 3
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_get_feature_importance_tied_covariance(self, trained_hmm_with_features):
        """Test feature importance with tied covariance."""
        trained_hmm_with_features.model_params["covariance_type"] = "tied"

        # For tied covariance, should still work
        importance = trained_hmm_with_features.get_feature_importance()

        assert len(importance) == 3
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_get_feature_importance_untrained_model(self):
        """Test feature importance with untrained model."""
        predictor = HMMPredictor()

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_missing_attributes(self, trained_hmm_with_features):
        """Test feature importance when state model is missing attributes."""
        # Remove means and covariances
        del trained_hmm_with_features.state_model.means_
        del trained_hmm_with_features.state_model.covariances_

        importance = trained_hmm_with_features.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_error_handling(self, trained_hmm_with_features):
        """Test feature importance calculation error handling."""
        # Make means property raise exception
        type(trained_hmm_with_features.state_model).means_ = PropertyMock(
            side_effect=Exception("Error")
        )

        importance = trained_hmm_with_features.get_feature_importance()

        # Should return empty dict on error
        assert importance == {}


class TestHMMStateInfo:
    """Test HMM state information retrieval."""

    @pytest.fixture
    def hmm_with_state_info(self):
        """HMM with complete state information."""
        predictor = HMMPredictor()
        predictor.model_params = {"n_components": 3}
        predictor.state_labels = {
            0: "Quick_Transition_0",
            1: "Short_Stay_1",
            2: "Long_Stay_2",
        }
        predictor.state_characteristics = {
            0: {"avg_duration": 300, "sample_count": 10},
            1: {"avg_duration": 1800, "sample_count": 25},
            2: {"avg_duration": 3600, "sample_count": 15},
        }
        predictor.transition_matrix = np.array(
            [[0.2, 0.6, 0.2], [0.3, 0.4, 0.3], [0.1, 0.3, 0.6]]
        )
        return predictor

    def test_get_state_info(self, hmm_with_state_info):
        """Test state information retrieval."""
        state_info = hmm_with_state_info.get_state_info()

        assert "n_states" in state_info
        assert "state_labels" in state_info
        assert "state_characteristics" in state_info
        assert "transition_matrix" in state_info

        assert state_info["n_states"] == 3
        assert len(state_info["state_labels"]) == 3
        assert len(state_info["state_characteristics"]) == 3
        assert len(state_info["transition_matrix"]) == 3
        assert len(state_info["transition_matrix"][0]) == 3

    def test_get_state_info_no_transition_matrix(self, hmm_with_state_info):
        """Test state info when transition matrix is None."""
        hmm_with_state_info.transition_matrix = None

        state_info = hmm_with_state_info.get_state_info()

        assert state_info["transition_matrix"] is None


class TestHMMSerialization:
    """Test HMM model serialization and loading."""

    @pytest.fixture
    def trained_hmm_for_serialization(self, sample_training_data):
        """Trained HMM for serialization testing."""
        features, targets = sample_training_data
        predictor = HMMPredictor(room_id="test_room")

        # Mock trained state
        predictor.is_trained = True
        predictor.feature_names = list(features.columns)
        predictor.training_date = datetime.now(timezone.utc)
        predictor.model_version = "v1.0"

        # Mock training history
        mock_result = TrainingResult(
            success=True,
            training_time_seconds=15.0,
            model_version="v1.0",
            training_samples=200,
        )
        predictor.training_history = [mock_result]

        # Mock model components
        predictor.state_model = Mock(spec=GaussianMixture)
        predictor.transition_models = {0: {"type": "average", "value": 1800}}
        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.transition_matrix = np.array([[0.5, 0.5], [0.3, 0.7]])
        predictor.state_labels = {0: "State_0", 1: "State_1"}
        predictor.state_characteristics = {0: {"avg_duration": 1800}}

        return predictor

    def test_save_model_success(self, trained_hmm_for_serialization, tmp_path):
        """Test successful model saving."""
        model_path = tmp_path / "hmm_model.pkl"

        result = trained_hmm_for_serialization.save_model(str(model_path))

        assert result is True
        assert model_path.exists()

    def test_save_model_failure(self, trained_hmm_for_serialization):
        """Test model saving failure."""
        # Invalid path that should cause save to fail
        invalid_path = "/invalid/path/model.pkl"

        result = trained_hmm_for_serialization.save_model(invalid_path)

        assert result is False

    def test_load_model_success(self, trained_hmm_for_serialization, tmp_path):
        """Test successful model loading."""
        model_path = tmp_path / "hmm_model.pkl"

        # First save the model
        trained_hmm_for_serialization.save_model(str(model_path))

        # Create new predictor and load
        new_predictor = HMMPredictor()
        result = new_predictor.load_model(str(model_path))

        assert result is True
        assert new_predictor.room_id == "test_room"
        assert new_predictor.model_version == "v1.0"
        assert new_predictor.is_trained is True
        assert len(new_predictor.training_history) == 1
        assert new_predictor.transition_matrix is not None
        assert len(new_predictor.state_labels) == 2

    def test_load_model_failure(self):
        """Test model loading failure."""
        predictor = HMMPredictor()

        # Non-existent file
        result = predictor.load_model("nonexistent_file.pkl")

        assert result is False


class TestHMMUtilityMethods:
    """Test HMM utility and helper methods."""

    @pytest.fixture
    def hmm_predictor(self):
        """HMM predictor fixture."""
        return HMMPredictor()

    def test_determine_transition_type_from_states(self, hmm_predictor):
        """Test transition type determination from states."""
        # Mock state characteristics
        hmm_predictor.state_characteristics = {
            0: {"avg_duration": 600},  # Short duration
            1: {"avg_duration": 3600},  # Long duration (suggests occupied)
            2: {"avg_duration": 1800},  # Medium duration
        }

        # Test with explicit occupancy states
        transition = hmm_predictor._determine_transition_type_from_states(
            0, np.array([0.1, 0.6, 0.3]), "occupied"
        )
        assert transition == "occupied_to_vacant"

        transition = hmm_predictor._determine_transition_type_from_states(
            1, np.array([0.2, 0.5, 0.3]), "vacant"
        )
        assert transition == "vacant_to_occupied"

        # Test with unknown occupancy (should use duration heuristic)
        transition = hmm_predictor._determine_transition_type_from_states(
            1, None, "unknown"  # Long duration state
        )
        assert transition == "occupied_to_vacant"

        transition = hmm_predictor._determine_transition_type_from_states(
            0, None, "unknown"  # Short duration state
        )
        assert transition == "vacant_to_occupied"

    def test_calculate_confidence(self, hmm_predictor):
        """Test confidence calculation."""
        # High confidence state identification, reasonable duration
        confidence = hmm_predictor._calculate_confidence(
            0.9, 1800, np.array([0.9, 0.05, 0.03, 0.02])
        )
        assert 0.1 <= confidence <= 0.95

        # Low confidence state identification (high entropy)
        confidence = hmm_predictor._calculate_confidence(
            0.4, 1800, np.array([0.4, 0.3, 0.2, 0.1])
        )
        assert confidence < 0.8  # Should be lower due to uncertainty

        # Extreme duration should reduce confidence
        confidence = hmm_predictor._calculate_confidence(
            0.8, 100000, np.array([0.8, 0.1, 0.05, 0.05])
        )
        assert confidence < 0.8  # Should be reduced due to extreme duration

    def test_prepare_targets_time_until_column(self, hmm_predictor):
        """Test target preparation with time_until_transition_seconds column."""
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [1800, 3600, 2400, 900]}
        )

        prepared = hmm_predictor._prepare_targets(targets)

        assert isinstance(prepared, np.ndarray)
        assert len(prepared) == 4
        assert np.array_equal(prepared, [1800, 3600, 2400, 900])

    def test_prepare_targets_time_columns(self, hmm_predictor):
        """Test target preparation with time columns."""
        targets = pd.DataFrame(
            {
                "target_time": [
                    "2024-01-01T12:00:00",
                    "2024-01-01T13:00:00",
                ],
                "next_transition_time": [
                    "2024-01-01T12:30:00",
                    "2024-01-01T14:15:00",
                ],
            }
        )

        prepared = hmm_predictor._prepare_targets(targets)

        assert isinstance(prepared, np.ndarray)
        assert len(prepared) == 2
        assert prepared[0] == 1800  # 30 minutes
        assert prepared[1] == 4500  # 75 minutes

    def test_prepare_targets_clipping(self, hmm_predictor):
        """Test target value clipping to reasonable bounds."""
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [
                    30,
                    1800,
                    100000,
                    -500,
                ]  # Some out of bounds
            }
        )

        prepared = hmm_predictor._prepare_targets(targets)

        # Should be clipped to [60, 86400] range
        assert all(60 <= val <= 86400 for val in prepared)
        assert prepared[0] == 60  # 30 clipped to 60
        assert prepared[1] == 1800  # 1800 unchanged
        assert prepared[2] == 86400  # 100000 clipped to 86400
        assert prepared[3] == 60  # -500 clipped to 60


class TestHMMEdgeCases:
    """Test HMM predictor edge cases and error conditions."""

    def test_hmm_predictor_with_single_state(self):
        """Test HMM predictor behavior with single state."""
        predictor = HMMPredictor(n_components=1)

        assert predictor.model_params["n_components"] == 1

    def test_hmm_predictor_extreme_parameter_values(self):
        """Test HMM predictor with extreme parameter values."""
        extreme_params = {
            "n_components": 20,  # Many states
            "tol": 1e-15,  # Very tight tolerance
            "n_iter": 1,  # Minimal iterations
        }

        predictor = HMMPredictor(**extreme_params)

        assert predictor.model_params["n_components"] == 20
        assert predictor.model_params["tol"] == 1e-15
        assert predictor.model_params["n_iter"] == 1

    def test_hmm_predictor_empty_state_handling(self):
        """Test HMM predictor handling of empty states."""
        predictor = HMMPredictor()

        # Test with empty state characteristics
        predictor.state_characteristics = {}
        transition_type = predictor._determine_transition_type_from_states(
            99, None, "unknown"
        )

        # Should handle missing state gracefully
        assert transition_type in ["occupied_to_vacant", "vacant_to_occupied"]

    @pytest.mark.asyncio
    async def test_hmm_prediction_with_missing_transition_models(self):
        """Test HMM prediction when some states lack transition models."""
        predictor = HMMPredictor()
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2"]

        # Mock state model that predicts state without transition model
        mock_state_model = Mock()
        mock_state_model.predict_proba.return_value = np.array(
            [[0.1, 0.1, 0.1, 0.7]]
        )  # State 3
        predictor.state_model = mock_state_model

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2]])
        predictor.feature_scaler = mock_scaler

        # Only have transition model for state 0
        predictor.transition_models = {0: {"type": "average", "value": 1800}}

        predictor.transition_matrix = np.array([[1.0, 0.0, 0.0, 0.0]] * 4)
        predictor.state_labels = {3: "Unknown_State_3"}
        predictor.state_characteristics = {3: {"avg_duration": 2000}}

        features = pd.DataFrame({"feature1": [1], "feature2": [2]})
        prediction_time = datetime.now(timezone.utc)

        results = await predictor.predict(features, prediction_time)

        # Should handle missing transition model gracefully with default
        assert len(results) == 1
        result = results[0]
        time_until = result.prediction_metadata["time_until_transition_seconds"]
        assert time_until == 1800.0  # Default 30 minutes

    def test_hmm_predictor_memory_cleanup(self):
        """Test that HMM predictor can handle memory cleanup."""
        predictor = HMMPredictor()

        # Add many prediction records
        for i in range(2000):  # More than the 1000 limit
            mock_result = Mock(spec=PredictionResult)
            predictor._record_prediction(datetime.now(timezone.utc), mock_result)

        # Should have cleaned up to 500 most recent
        assert len(predictor.prediction_history) == 500
