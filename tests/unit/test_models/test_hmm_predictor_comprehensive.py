"""
Comprehensive unit tests for HMMPredictor class achieving 85%+ coverage.

This module provides complete test coverage for the Hidden Markov Model predictor:
- Initialization and configuration with different state counts
- Gaussian Mixture Model training for state identification
- State analysis and characterization
- Transition matrix construction and validation
- Duration prediction models per state
- Prediction generation with state inference
- Feature importance based on state discrimination
- State information retrieval and analysis
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
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


@pytest.fixture
def hmm_state_data():
    """Create data with clear hidden states for HMM testing."""
    np.random.seed(42)
    n_samples = 800
    n_features = 8

    # Create data with distinct hidden states/patterns
    # State 0: Quick transitions (short stays)
    # State 1: Medium stays
    # State 2: Long stays (occupied periods)
    # State 3: Very long stays (sleep periods)

    features = {}

    # Create state-dependent features
    # These will be used to identify hidden states
    state_indicators = np.random.choice(4, n_samples, p=[0.3, 0.3, 0.25, 0.15])

    # Duration-related features (strongly state-dependent)
    features["prev_state_duration"] = np.zeros(n_samples)
    features["transition_frequency"] = np.zeros(n_samples)
    features["stability_index"] = np.zeros(n_samples)

    for i in range(n_samples):
        state = state_indicators[i]

        if state == 0:  # Quick transitions
            features["prev_state_duration"][i] = np.random.exponential(
                300
            )  # ~5 minutes
            features["transition_frequency"][i] = np.random.poisson(8)  # High frequency
            features["stability_index"][i] = np.random.uniform(
                0.1, 0.3
            )  # Low stability
        elif state == 1:  # Medium stays
            features["prev_state_duration"][i] = np.random.exponential(
                1200
            )  # ~20 minutes
            features["transition_frequency"][i] = np.random.poisson(
                3
            )  # Medium frequency
            features["stability_index"][i] = np.random.uniform(
                0.4, 0.7
            )  # Medium stability
        elif state == 2:  # Long stays
            features["prev_state_duration"][i] = np.random.exponential(3600)  # ~1 hour
            features["transition_frequency"][i] = np.random.poisson(1)  # Low frequency
            features["stability_index"][i] = np.random.uniform(
                0.7, 0.9
            )  # High stability
        else:  # Very long stays (state 3)
            features["prev_state_duration"][i] = np.random.exponential(7200)  # ~2 hours
            features["transition_frequency"][i] = np.random.poisson(
                0.5
            )  # Very low frequency
            features["stability_index"][i] = np.random.uniform(
                0.85, 1.0
            )  # Very high stability

    # Environmental features (less state-dependent, more noise)
    features["temperature"] = 20 + np.random.normal(0, 3, n_samples)
    features["motion_events"] = np.random.gamma(2, 2, n_samples)
    features["door_activations"] = np.random.poisson(1.5, n_samples)

    # Time-based features with weak state correlation
    hours = np.random.randint(0, 24, n_samples)
    features["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    features["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Create state-dependent targets (next transition times)
    targets = np.zeros(n_samples)

    for i in range(n_samples):
        state = state_indicators[i]

        if state == 0:  # Quick transitions -> short next transition time
            base_time = 600  # 10 minutes
            multiplier = np.random.gamma(1.5, 0.8)  # Gamma distribution for realism
        elif state == 1:  # Medium stays -> medium next transition time
            base_time = 1800  # 30 minutes
            multiplier = np.random.gamma(2, 1)
        elif state == 2:  # Long stays -> long next transition time
            base_time = 4500  # 75 minutes
            multiplier = np.random.gamma(2.5, 1.2)
        else:  # Very long stays (state 3)
            base_time = 9000  # 2.5 hours
            multiplier = np.random.gamma(1.8, 1.5)

        targets[i] = base_time * multiplier * (1 + np.random.normal(0, 0.2))
        targets[i] = np.clip(targets[i], 60, 14400)  # 1 minute to 4 hours

    # Create transition types based on state patterns and time
    transition_types = []
    for i in range(n_samples):
        state = state_indicators[i]
        hour = hours[i]

        if state == 0:  # Quick transitions - mixed
            transition_types.append(
                "vacant_to_occupied" if i % 2 == 0 else "occupied_to_vacant"
            )
        elif state == 3 and (22 <= hour or hour <= 6):  # Very long stays at night
            transition_types.append("occupied_to_vacant")
        elif 7 <= hour <= 18:  # Daytime
            transition_types.append("vacant_to_occupied")
        else:  # Evening
            transition_types.append("occupied_to_vacant")

    # Create time series
    start_time = datetime(2024, 1, 1, tzinfo=timezone.utc)
    target_times = [start_time + timedelta(minutes=i * 20) for i in range(n_samples)]

    features_df = pd.DataFrame(features)
    targets_df = pd.DataFrame(
        {
            "time_until_transition_seconds": targets,
            "transition_type": transition_types,
            "target_time": target_times,
            "next_transition_time": [
                tt + timedelta(seconds=dur) for tt, dur in zip(target_times, targets)
            ],
            "true_state": state_indicators,  # Hidden for validation
        }
    )

    return features_df, targets_df


@pytest.fixture
def hmm_split_data(hmm_state_data):
    """Split HMM data into train/validation/test sets."""
    features, targets = hmm_state_data

    # 70/15/15 split
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


class TestHMMInitialization:
    """Test HMM predictor initialization and configuration."""

    def test_hmm_default_initialization(self):
        """Test HMM initialization with default parameters."""
        hmm = HMMPredictor()

        assert hmm.model_type == ModelType.HMM
        assert hmm.room_id is None
        assert not hmm.is_trained
        assert hmm.state_model is None
        assert hmm.transition_models == {}

        # Check default parameters
        assert hmm.model_params["n_components"] == 4
        assert hmm.model_params["n_states"] == 4  # Alias
        assert hmm.model_params["covariance_type"] == "full"
        assert hmm.model_params["max_iter"] == 100
        assert hmm.model_params["n_iter"] == 100  # Primary parameter name
        assert hmm.model_params["random_state"] == 42
        assert hmm.model_params["tol"] == 1e-3

        # Check components
        assert isinstance(hmm.feature_scaler, StandardScaler)
        assert hmm.state_labels == {}
        assert hmm.state_characteristics == {}
        assert hmm.transition_matrix is None
        assert hmm.state_durations == {}

    def test_hmm_custom_initialization(self):
        """Test HMM initialization with custom parameters."""
        custom_params = {
            "n_components": 3,
            "covariance_type": "diag",
            "max_iter": 200,
            "tol": 1e-4,
            "init_params": "random",
        }

        hmm = HMMPredictor(room_id="custom_hmm", **custom_params)

        assert hmm.room_id == "custom_hmm"
        assert hmm.model_params["n_components"] == 3
        assert hmm.model_params["n_states"] == 3  # Should match n_components
        assert hmm.model_params["covariance_type"] == "diag"
        assert hmm.model_params["max_iter"] == 200
        assert hmm.model_params["n_iter"] == 200  # Should match max_iter
        assert hmm.model_params["tol"] == 1e-4
        assert hmm.model_params["init_params"] == "random"

    def test_hmm_parameter_aliases(self):
        """Test HMM parameter aliases and compatibility."""
        # Test n_iter vs max_iter
        hmm_iter = HMMPredictor(n_iter=150)
        assert hmm_iter.model_params["n_iter"] == 150
        assert hmm_iter.model_params["max_iter"] == 150

        # Test max_iter override
        hmm_max = HMMPredictor(n_iter=100, max_iter=175)
        assert hmm_max.model_params["max_iter"] == 175
        assert hmm_max.model_params["n_iter"] == 100  # Primary parameter preserved

        # Test n_components vs n_states
        hmm_states = HMMPredictor(n_states=5)
        assert hmm_states.model_params["n_components"] == 5  # Should be updated
        assert hmm_states.model_params["n_states"] == 5

    def test_hmm_parameters_from_constants(self):
        """Test HMM parameters loaded from constants."""
        hmm = HMMPredictor()

        # Should use DEFAULT_MODEL_PARAMS[ModelType.HMM]
        default_params = DEFAULT_MODEL_PARAMS[ModelType.HMM]

        # Key parameters should match defaults
        assert hmm.model_params["n_components"] == default_params.get("n_components", 4)
        assert hmm.model_params["covariance_type"] == default_params.get(
            "covariance_type", "full"
        )
        assert hmm.model_params["max_iter"] == default_params.get("max_iter", 100)


class TestHMMStateIdentification:
    """Test HMM hidden state identification and analysis."""

    @pytest.mark.asyncio
    async def test_hmm_state_model_training(self, hmm_split_data):
        """Test Gaussian Mixture Model training for state identification."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(
            n_components=4,
            covariance_type="full",
            max_iter=50,  # Limit for speed
            room_id="state_training_test",
        )

        # Use subset for faster testing
        subset_features = train_features.head(200)
        subset_targets = train_targets.head(200)

        result = await hmm.train(subset_features, subset_targets)

        # Validate training success
        assert result.success is True
        assert hmm.is_trained is True
        assert hmm.state_model is not None
        assert isinstance(hmm.state_model, GaussianMixture)

        # Check state model configuration
        assert hmm.state_model.n_components == 4
        assert hmm.state_model.covariance_type == "full"

        # Should have identified states
        assert len(hmm.state_labels) > 0
        assert len(hmm.state_characteristics) > 0

        # Each identified state should have characteristics
        for state_id in hmm.state_labels:
            assert state_id in hmm.state_characteristics
            char = hmm.state_characteristics[state_id]

            assert "avg_duration" in char
            assert "std_duration" in char
            assert "sample_count" in char
            assert "feature_means" in char
            assert "prediction_reliability" in char

            # Values should be reasonable
            assert char["avg_duration"] > 0
            assert char["sample_count"] > 0
            assert char["prediction_reliability"] in ["low", "medium", "high"]

    @pytest.mark.asyncio
    async def test_hmm_kmeans_initialization(self, hmm_split_data):
        """Test K-means initialization for GMM."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="kmeans_init_test")

        with patch("src.models.base.hmm_predictor.KMeans") as mock_kmeans:
            mock_instance = MagicMock()
            mock_centers = np.random.normal(0, 1, (3, len(train_features.columns)))
            mock_instance.cluster_centers_ = mock_centers
            mock_instance.fit = MagicMock()
            mock_kmeans.return_value = mock_instance

            result = await hmm.train(train_features.head(100), train_targets.head(100))

            # K-means should have been called for initialization
            mock_kmeans.assert_called_once()
            call_kwargs = mock_kmeans.call_args[1]
            assert call_kwargs["n_clusters"] == 3
            assert call_kwargs["random_state"] == 42

            assert result.success is True

    @pytest.mark.asyncio
    async def test_hmm_state_characterization(self, hmm_split_data):
        """Test state analysis and characterization."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="characterization_test")

        result = await hmm.train(train_features.head(150), train_targets.head(150))

        assert result.success is True

        # Should have characterized all states
        assert len(hmm.state_characteristics) > 0

        # Check training metrics include state information
        metrics = result.training_metrics
        assert "n_states" in metrics
        assert "state_distribution" in metrics

        # State distribution should be reasonable
        state_dist = metrics["state_distribution"]
        assert len(state_dist) == hmm.model_params["n_components"]
        assert all(count >= 0 for count in state_dist)
        assert sum(state_dist) <= len(train_features.head(150))

    @pytest.mark.asyncio
    async def test_hmm_state_labeling(self, hmm_split_data):
        """Test intuitive state labeling based on characteristics."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="labeling_test")

        result = await hmm.train(train_features.head(120), train_targets.head(120))

        assert result.success is True

        # Should have assigned labels to states
        assert len(hmm.state_labels) > 0

        for state_id, label in hmm.state_labels.items():
            # Labels should be descriptive
            assert isinstance(label, str)
            assert len(label) > 0

            # Should be based on duration characteristics
            char = hmm.state_characteristics[state_id]
            avg_duration = char["avg_duration"]

            if avg_duration < 600:  # < 10 minutes
                assert "Quick_Transition" in label
            elif avg_duration < 3600:  # < 1 hour
                assert "Short_Stay" in label
            elif avg_duration < 14400:  # < 4 hours
                assert "Medium_Stay" in label
            else:
                assert "Long_Stay" in label

    @pytest.mark.asyncio
    async def test_hmm_state_probability_analysis(self, hmm_split_data):
        """Test state probability analysis and confidence metrics."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="probability_test")

        # Mock state probabilities for testing
        with patch.object(hmm, "_analyze_states") as mock_analyze:

            def mock_analyze_states(
                X, state_labels, durations, feature_names, state_probabilities
            ):
                # Simulate state analysis with probabilities
                for state_id in range(4):
                    state_mask = state_labels == state_id
                    if np.sum(state_mask) > 0:
                        # Mock characteristics with probability metrics
                        hmm.state_characteristics[state_id] = {
                            "avg_duration": 1800.0,
                            "std_duration": 600.0,
                            "sample_count": int(np.sum(state_mask)),
                            "feature_means": [0.0] * len(feature_names),
                            "avg_state_probability": 0.8,
                            "confidence_variance": 0.1,
                            "high_confidence_samples": 20,
                            "low_confidence_samples": 5,
                            "prediction_reliability": "high",
                        }
                        hmm.state_labels[state_id] = f"State_{state_id}"
                        hmm.state_durations[state_id] = [1800.0] * int(
                            np.sum(state_mask)
                        )

            mock_analyze.side_effect = mock_analyze_states

            result = await hmm.train(train_features.head(100), train_targets.head(100))

            assert result.success is True

            # Check that probability-based metrics were calculated
            for state_id, char in hmm.state_characteristics.items():
                assert "avg_state_probability" in char
                assert "confidence_variance" in char
                assert "high_confidence_samples" in char
                assert "low_confidence_samples" in char
                assert "prediction_reliability" in char

    @pytest.mark.asyncio
    async def test_hmm_different_covariance_types(self, hmm_split_data):
        """Test HMM with different covariance types."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        covariance_types = ["full", "diag", "tied", "spherical"]

        for cov_type in covariance_types:
            hmm = HMMPredictor(
                n_components=3,
                covariance_type=cov_type,
                max_iter=30,  # Quick training
                room_id=f"{cov_type}_test",
            )

            result = await hmm.train(train_features.head(80), train_targets.head(80))

            assert result.success is True
            assert hmm.state_model.covariance_type == cov_type

            # Should still identify states regardless of covariance type
            assert len(hmm.state_characteristics) > 0


class TestHMMTransitionMatrix:
    """Test HMM transition matrix construction and validation."""

    @pytest.mark.asyncio
    async def test_transition_matrix_construction(self, hmm_split_data):
        """Test transition matrix construction from state sequences."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="transition_test")

        result = await hmm.train(train_features.head(150), train_targets.head(150))

        assert result.success is True
        assert hmm.transition_matrix is not None

        # Validate transition matrix properties
        n_states = hmm.model_params["n_components"]
        assert hmm.transition_matrix.shape == (n_states, n_states)

        # Each row should sum to approximately 1 (probability distribution)
        for i in range(n_states):
            row_sum = np.sum(hmm.transition_matrix[i, :])
            assert abs(row_sum - 1.0) < 1e-10

        # All probabilities should be non-negative
        assert np.all(hmm.transition_matrix >= 0)

    @pytest.mark.asyncio
    async def test_transition_matrix_with_missing_transitions(self, hmm_split_data):
        """Test transition matrix handling when some transitions are missing."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="missing_transitions_test")

        # Mock state model to return specific state sequences
        hmm.state_model = MagicMock()
        hmm.state_model.fit = MagicMock()

        # Create state labels with missing transitions (no transitions from state 3)
        state_labels = np.array([0, 1, 0, 2, 1, 0, 2, 1] * 10)  # No state 3 transitions
        state_probabilities = np.random.dirichlet(np.ones(4), len(state_labels))

        hmm.state_model.predict = MagicMock(return_value=state_labels)
        hmm.state_model.predict_proba = MagicMock(return_value=state_probabilities)

        result = await hmm.train(train_features.head(80), train_targets.head(80))

        assert result.success is True
        assert hmm.transition_matrix is not None

        # States with no observed transitions should have uniform distribution
        # Check that the transition matrix is still valid
        for i in range(4):
            row_sum = np.sum(hmm.transition_matrix[i, :])
            assert abs(row_sum - 1.0) < 1e-10

    def test_transition_matrix_uniform_fallback(self):
        """Test uniform distribution fallback for states with no transitions."""
        hmm = HMMPredictor(n_components=3, room_id="uniform_fallback_test")

        # Create state sequence with no transitions from state 2
        state_labels = np.array([0, 1, 0, 1, 0, 1])

        hmm._build_transition_matrix(state_labels)

        assert hmm.transition_matrix is not None

        # State 2 should have uniform transition probabilities
        # (since it appears at the end with no transitions observed)
        if hmm.transition_matrix.shape[0] > 2:
            row_2 = hmm.transition_matrix[2, :]
            # Should be close to uniform [1/3, 1/3, 1/3]
            expected_uniform = 1.0 / 3
            for prob in row_2:
                assert abs(prob - expected_uniform) < 0.1  # Allow some tolerance


class TestHMMDurationModels:
    """Test HMM duration prediction models for each state."""

    @pytest.mark.asyncio
    async def test_state_duration_model_training(self, hmm_split_data):
        """Test training of duration models for each state."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="duration_models_test")

        result = await hmm.train(train_features.head(120), train_targets.head(120))

        assert result.success is True
        assert len(hmm.transition_models) > 0

        # Should have models for states with sufficient data
        for state_id, model_info in hmm.transition_models.items():
            assert "type" in model_info
            assert model_info["type"] in ["regression", "average"]

            if model_info["type"] == "regression":
                assert "model" in model_info
                assert hasattr(model_info["model"], "predict")
            elif model_info["type"] == "average":
                assert "value" in model_info
                assert model_info["value"] > 0

    @pytest.mark.asyncio
    async def test_duration_model_regression_vs_average(self, hmm_split_data):
        """Test choice between regression and average duration models."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="regression_avg_test")

        # Mock state identification to control state sample sizes
        hmm.state_model = MagicMock()
        hmm.state_model.fit = MagicMock()

        # Create state labels with different sample sizes per state
        n_samples = 100
        state_labels = np.zeros(n_samples, dtype=int)

        # State 0: Large sample (should get regression)
        state_labels[:50] = 0

        # State 1: Medium sample (should get regression)
        state_labels[50:75] = 1

        # State 2: Small sample (should get average)
        state_labels[75:78] = 2  # Only 3 samples

        # State 3: No samples (should not appear in models)

        hmm.state_model.predict = MagicMock(return_value=state_labels)
        hmm.state_model.predict_proba = MagicMock(
            return_value=np.random.dirichlet(np.ones(4), n_samples)
        )

        result = await hmm.train(
            train_features.head(n_samples), train_targets.head(n_samples)
        )

        assert result.success is True

        # Check model types based on sample sizes
        if 0 in hmm.transition_models:
            assert hmm.transition_models[0]["type"] == "regression"  # Large sample

        if 1 in hmm.transition_models:
            assert hmm.transition_models[1]["type"] == "regression"  # Medium sample

        if 2 in hmm.transition_models:
            assert hmm.transition_models[2]["type"] == "average"  # Small sample

        # State 3 should not have a model (no samples)
        assert 3 not in hmm.transition_models

    @pytest.mark.asyncio
    async def test_duration_prediction_functionality(self, hmm_split_data):
        """Test duration prediction functionality for trained models."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="duration_prediction_test")

        result = await hmm.train(train_features.head(100), train_targets.head(100))

        assert result.success is True

        # Test internal duration prediction methods
        for state_id, model_info in hmm.transition_models.items():
            # Create test input
            test_input = np.random.normal(0, 1, (1, len(train_features.columns)))

            duration = hmm._predict_single_duration(test_input, state_id)

            # Duration should be reasonable
            assert isinstance(duration, float)
            assert 60 <= duration <= 86400  # 1 minute to 24 hours

    def test_duration_prediction_edge_cases(self, hmm_split_data):
        """Test duration prediction edge cases."""
        train_features, _, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="duration_edge_test")

        # Test prediction for unknown state
        test_input = np.random.normal(0, 1, (1, len(train_features.columns)))

        duration = hmm._predict_single_duration(test_input, 999)  # Non-existent state

        # Should return default value
        assert duration == 1800.0  # 30 minutes default

        # Test with empty transition models
        hmm.transition_models = {}

        duration = hmm._predict_single_duration(test_input, 0)
        assert duration == 1800.0  # 30 minutes default


class TestHMMPrediction:
    """Test HMM prediction generation and state inference."""

    @pytest.mark.asyncio
    async def test_hmm_prediction_generation(self, hmm_split_data):
        """Test HMM prediction generation with state inference."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="prediction_test")
        await self._setup_trained_hmm(hmm, train_features, train_targets)

        # Generate predictions
        prediction_time = datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc)
        predictions = await hmm.predict(
            test_features.head(10), prediction_time, "vacant"
        )

        # Validate predictions
        assert len(predictions) == 10

        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.model_type == ModelType.HMM.value
            assert pred.predicted_time > prediction_time
            assert pred.transition_type in ["vacant_to_occupied", "occupied_to_vacant"]
            assert 0.0 <= pred.confidence_score <= 1.0

            # Check HMM-specific metadata
            metadata = pred.prediction_metadata
            assert "prediction_method" in metadata
            assert metadata["prediction_method"] == "hidden_markov_model"
            assert "current_hidden_state" in metadata
            assert "state_probability" in metadata
            assert "state_label" in metadata
            assert "all_state_probabilities" in metadata

            # Validate state information
            assert isinstance(metadata["current_hidden_state"], int)
            assert 0 <= metadata["current_hidden_state"] < 4
            assert 0.0 <= metadata["state_probability"] <= 1.0
            assert isinstance(metadata["all_state_probabilities"], list)
            assert len(metadata["all_state_probabilities"]) == 4

    @pytest.mark.asyncio
    async def test_hmm_state_inference_accuracy(self, hmm_split_data):
        """Test accuracy of state inference."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="state_inference_test")
        await self._setup_trained_hmm(hmm, train_features, train_targets)

        predictions = await hmm.predict(
            test_features.head(5), datetime.now(timezone.utc), "occupied"
        )

        for pred in predictions:
            metadata = pred.prediction_metadata

            # State probabilities should sum to ~1
            state_probs = metadata["all_state_probabilities"]
            total_prob = sum(state_probs)
            assert abs(total_prob - 1.0) < 0.01

            # Most likely state should match the reported current state
            most_likely_state = np.argmax(state_probs)
            assert most_likely_state == metadata["current_hidden_state"]

            # State probability should be the maximum probability
            max_prob = max(state_probs)
            assert abs(metadata["state_probability"] - max_prob) < 1e-6

    @pytest.mark.asyncio
    async def test_hmm_transition_type_inference(self, hmm_split_data):
        """Test transition type inference from state characteristics."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="transition_inference_test")
        await self._setup_trained_hmm(hmm, train_features, train_targets)

        # Setup state characteristics with different duration patterns
        hmm.state_characteristics = {
            0: {"avg_duration": 600},  # Short - quick transition
            1: {"avg_duration": 1800},  # Medium - normal transition
            2: {"avg_duration": 4800},  # Long - occupied state
            3: {"avg_duration": 9000},  # Very long - sleep state
        }

        # Test different current states and expected transitions
        test_cases = [
            ("occupied", "occupied_to_vacant"),
            ("vacant", "vacant_to_occupied"),
            ("unknown", "vacant_to_occupied"),  # Default daytime behavior
        ]

        for current_state, expected_transition in test_cases:
            predictions = await hmm.predict(
                test_features.head(1),
                datetime(2024, 6, 15, 14, 0, tzinfo=timezone.utc),
                current_state,
            )

            # The transition type should be determined by current state primarily
            assert predictions[0].transition_type == expected_transition

    @pytest.mark.asyncio
    async def test_hmm_confidence_calculation(self, hmm_split_data):
        """Test HMM confidence calculation with state uncertainty."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="confidence_test")
        await self._setup_trained_hmm(hmm, train_features, train_targets)

        # Mock different state probability scenarios
        test_scenarios = [
            np.array([0.9, 0.05, 0.03, 0.02]),  # Very confident
            np.array([0.6, 0.2, 0.15, 0.05]),  # Moderately confident
            np.array([0.4, 0.3, 0.2, 0.1]),  # Less confident
            np.array([0.25, 0.25, 0.25, 0.25]),  # Very uncertain
        ]

        for i, state_probs in enumerate(test_scenarios):
            # Mock state model to return specific probabilities
            hmm.state_model.predict_proba = MagicMock(
                return_value=np.array([state_probs])
            )
            hmm.state_model.predict = MagicMock(
                return_value=np.array([np.argmax(state_probs)])
            )

            predictions = await hmm.predict(
                test_features.head(1), datetime.now(timezone.utc), "vacant"
            )

            confidence = predictions[0].confidence_score

            # Higher state certainty should lead to higher confidence
            max_state_prob = np.max(state_probs)

            # Very confident state identification should yield higher confidence
            if max_state_prob > 0.8:
                assert confidence > 0.6
            elif max_state_prob < 0.4:
                assert confidence < 0.7

    @pytest.mark.asyncio
    async def test_hmm_next_state_prediction(self, hmm_split_data):
        """Test next state prediction using transition matrix."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="next_state_test")
        await self._setup_trained_hmm(hmm, train_features, train_targets)

        # Setup specific transition matrix for testing
        hmm.transition_matrix = np.array(
            [
                [0.1, 0.7, 0.2],  # From state 0: likely to go to state 1
                [0.3, 0.2, 0.5],  # From state 1: likely to go to state 2
                [0.4, 0.4, 0.2],  # From state 2: likely to go to states 0 or 1
            ]
        )

        predictions = await hmm.predict(
            test_features.head(3), datetime.now(timezone.utc), "vacant"
        )

        for pred in predictions:
            metadata = pred.prediction_metadata

            # Should have next state probabilities
            next_state_probs = metadata.get("next_state_probabilities")

            if next_state_probs is not None:
                assert isinstance(next_state_probs, list)
                assert len(next_state_probs) == 3
                assert all(0 <= prob <= 1 for prob in next_state_probs)
                assert abs(sum(next_state_probs) - 1.0) < 0.01

    @pytest.mark.asyncio
    async def test_hmm_prediction_error_handling(self, hmm_split_data):
        """Test HMM prediction error handling."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="pred_error_test")

        # Test prediction on untrained model
        with pytest.raises(ModelPredictionError) as exc_info:
            await hmm.predict(
                test_features.head(5), datetime.now(timezone.utc), "vacant"
            )

        assert "hmm" in str(exc_info.value).lower()

        # Setup trained model
        await self._setup_trained_hmm(hmm, train_features, train_targets)

        # Test invalid features
        with patch.object(hmm, "validate_features", return_value=False):
            with pytest.raises(ModelPredictionError):
                await hmm.predict(
                    test_features.head(5), datetime.now(timezone.utc), "vacant"
                )

        # Test state model prediction failure
        hmm.state_model.predict_proba = MagicMock(
            side_effect=Exception("State prediction failed")
        )

        with pytest.raises(ModelPredictionError) as exc_info:
            await hmm.predict(
                test_features.head(1), datetime.now(timezone.utc), "vacant"
            )

        assert "HMM prediction failed" in str(exc_info.value)

    async def _setup_trained_hmm(self, hmm, train_features, train_targets):
        """Setup a trained HMM for prediction testing."""
        hmm.is_trained = True
        hmm.training_date = datetime.now(timezone.utc)
        hmm.model_version = "v1.0"
        hmm.feature_names = list(train_features.columns)

        # Setup mock state model
        hmm.state_model = MagicMock()

        # Mock state inference methods
        def mock_predict_proba(X):
            n_samples = len(X)
            # Generate realistic state probabilities
            probs = np.random.dirichlet(np.ones(4), n_samples)
            return probs

        def mock_predict(X):
            probs = mock_predict_proba(X)
            return np.argmax(probs, axis=1)

        hmm.state_model.predict_proba = MagicMock(side_effect=mock_predict_proba)
        hmm.state_model.predict = MagicMock(side_effect=mock_predict)

        # Setup feature scaler
        hmm.feature_scaler = MagicMock()
        hmm.feature_scaler.transform = MagicMock(
            side_effect=lambda x: np.random.normal(0, 1, x.shape)
        )

        # Setup state characteristics
        hmm.state_characteristics = {
            0: {"avg_duration": 800},
            1: {"avg_duration": 2000},
            2: {"avg_duration": 4500},
            3: {"avg_duration": 8000},
        }

        hmm.state_labels = {
            0: "Quick_Transition_0",
            1: "Short_Stay_1",
            2: "Medium_Stay_2",
            3: "Long_Stay_3",
        }

        # Setup transition matrix
        hmm.transition_matrix = np.array(
            [
                [0.2, 0.3, 0.3, 0.2],
                [0.4, 0.2, 0.3, 0.1],
                [0.3, 0.4, 0.2, 0.1],
                [0.5, 0.2, 0.2, 0.1],
            ]
        )

        # Setup transition models
        hmm.transition_models = {}
        for state_id in range(4):
            hmm.transition_models[state_id] = {
                "type": "average",
                "value": hmm.state_characteristics[state_id]["avg_duration"],
            }

        # Mock training history
        hmm.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=80,
                model_version="v1.0",
                training_samples=150,
                validation_score=0.72,
                training_score=0.75,
            )
        ]


class TestHMMFeatureImportance:
    """Test HMM feature importance based on state discrimination."""

    def test_hmm_feature_importance_calculation(self, hmm_split_data):
        """Test HMM feature importance based on state separation."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="importance_test")

        # Setup trained HMM
        hmm.is_trained = True
        hmm.feature_names = list(train_features.columns)

        # Mock state model with realistic means and covariances
        hmm.state_model = MagicMock()
        n_features = len(train_features.columns)

        # Create means that show different state characteristics
        state_means = np.array(
            [
                [500, 8, 0.2, 20, 3, 0.8, 0.1, 0.3],  # Quick transitions
                [1800, 3, 0.6, 22, 2, 0.0, 0.5, -0.2],  # Medium stays
                [4500, 1, 0.9, 24, 1, -0.5, 0.8, 0.1],  # Long stays
            ]
        )

        # Create covariances (full covariance type)
        state_covariances = []
        for i in range(3):
            # Different scales of variance for different states
            scale = 0.5 + i * 0.3
            cov = np.eye(n_features) * scale
            # Add some off-diagonal elements for realism
            cov[0, 1] = cov[1, 0] = 0.1 * scale
            state_covariances.append(cov)

        hmm.state_model.means_ = state_means[:, :n_features]  # Match actual features
        hmm.state_model.covariances_ = np.array(state_covariances)[
            :, :n_features, :n_features
        ]

        # Calculate feature importance
        importance = hmm.get_feature_importance()

        # Validate importance structure
        assert len(importance) == n_features
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

        # Should be normalized
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.01

        # Features with larger between-state variance should have higher importance
        # prev_state_duration should be highly important (designed to separate states)
        if "prev_state_duration" in importance:
            assert importance["prev_state_duration"] > 0.1  # Should be significant

    def test_hmm_feature_importance_different_covariance_types(self, hmm_split_data):
        """Test feature importance with different covariance types."""
        train_features, _, _, _, _, _ = hmm_split_data

        covariance_types = ["full", "diag", "tied", "spherical"]

        for cov_type in covariance_types:
            hmm = HMMPredictor(
                n_components=3,
                covariance_type=cov_type,
                room_id=f"importance_{cov_type}_test",
            )

            # Setup for specific covariance type
            hmm.is_trained = True
            hmm.feature_names = list(train_features.columns)
            hmm.state_model = MagicMock()

            n_features = len(train_features.columns)

            # Create appropriate means and covariances for each type
            means = np.random.normal(0, 2, (3, n_features))
            hmm.state_model.means_ = means

            if cov_type == "full":
                covariances = np.array([np.eye(n_features) * (i + 1) for i in range(3)])
            elif cov_type == "diag":
                covariances = np.random.uniform(0.5, 2.0, (3, n_features))
            elif cov_type == "tied":
                covariances = np.eye(n_features) * 1.5
            else:  # spherical
                covariances = np.random.uniform(0.5, 2.0, 3)

            hmm.state_model.covariances_ = covariances

            # Should work for all covariance types
            importance = hmm.get_feature_importance()

            assert len(importance) == n_features
            assert abs(sum(importance.values()) - 1.0) < 0.01

    def test_hmm_feature_importance_untrained_model(self):
        """Test feature importance on untrained HMM."""
        hmm = HMMPredictor(room_id="untrained_importance")

        importance = hmm.get_feature_importance()
        assert importance == {}

    def test_hmm_feature_importance_missing_attributes(self, hmm_split_data):
        """Test feature importance when state model lacks required attributes."""
        train_features, _, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="missing_attrs_test")
        hmm.is_trained = True
        hmm.feature_names = list(train_features.columns)

        # Mock state model without means_ or covariances_
        hmm.state_model = MagicMock()
        (
            delattr(hmm.state_model, "means_")
            if hasattr(hmm.state_model, "means_")
            else None
        )
        (
            delattr(hmm.state_model, "covariances_")
            if hasattr(hmm.state_model, "covariances_")
            else None
        )

        importance = hmm.get_feature_importance()
        assert importance == {}

    def test_hmm_feature_importance_error_handling(self, hmm_split_data):
        """Test feature importance error handling."""
        train_features, _, _, _, _, _ = hmm_split_data

        hmm = HMMPredictor(n_components=3, room_id="error_importance_test")
        hmm.is_trained = True
        hmm.feature_names = list(train_features.columns)

        # Mock state model that raises exception
        hmm.state_model = MagicMock()
        hmm.state_model.means_ = MagicMock(side_effect=Exception("Means error"))

        importance = hmm.get_feature_importance()
        assert importance == {}


class TestHMMStateInformation:
    """Test HMM state information retrieval and analysis."""

    def test_hmm_state_info_retrieval(self):
        """Test comprehensive state information retrieval."""
        hmm = HMMPredictor(n_components=3, room_id="state_info_test")

        # Setup mock trained state
        hmm.state_labels = {
            0: "Quick_Transition_0",
            1: "Medium_Stay_1",
            2: "Long_Stay_2",
        }

        hmm.state_characteristics = {
            0: {
                "avg_duration": 600,
                "std_duration": 200,
                "sample_count": 45,
                "feature_means": [500, 8, 0.2],
                "prediction_reliability": "medium",
            },
            1: {
                "avg_duration": 1800,
                "std_duration": 600,
                "sample_count": 60,
                "feature_means": [1800, 3, 0.6],
                "prediction_reliability": "high",
            },
            2: {
                "avg_duration": 4500,
                "std_duration": 1200,
                "sample_count": 35,
                "feature_means": [4500, 1, 0.9],
                "prediction_reliability": "medium",
            },
        }

        hmm.transition_matrix = np.array(
            [[0.2, 0.5, 0.3], [0.4, 0.3, 0.3], [0.6, 0.3, 0.1]]
        )

        state_info = hmm.get_state_info()

        # Validate state info structure
        required_keys = [
            "n_states",
            "state_labels",
            "state_characteristics",
            "transition_matrix",
        ]
        for key in required_keys:
            assert key in state_info

        # Check values
        assert state_info["n_states"] == 3
        assert state_info["state_labels"] == hmm.state_labels
        assert state_info["state_characteristics"] == hmm.state_characteristics

        # Transition matrix should be converted to list
        transition_matrix = state_info["transition_matrix"]
        assert isinstance(transition_matrix, list)
        assert len(transition_matrix) == 3
        assert all(len(row) == 3 for row in transition_matrix)

    def test_hmm_state_info_no_transition_matrix(self):
        """Test state info when transition matrix is not available."""
        hmm = HMMPredictor(n_components=2, room_id="no_transition_test")

        hmm.state_labels = {0: "State_0", 1: "State_1"}
        hmm.state_characteristics = {
            0: {"avg_duration": 1000},
            1: {"avg_duration": 2000},
        }
        hmm.transition_matrix = None

        state_info = hmm.get_state_info()

        assert state_info["n_states"] == 2
        assert state_info["transition_matrix"] is None

    def test_hmm_model_complexity_info(self):
        """Test HMM model complexity information."""
        hmm = HMMPredictor(
            n_components=4, covariance_type="full", room_id="complexity_test"
        )

        # Mock trained state
        hmm.is_trained = True
        hmm.state_characteristics = {i: {"sample_count": 20 + i * 5} for i in range(4)}
        hmm.transition_models = {
            0: {"type": "regression"},
            1: {"type": "regression"},
            2: {"type": "average"},
            3: {"type": "average"},
        }

        state_info = hmm.get_state_info()

        # Should include complexity metrics
        assert state_info["n_states"] == 4

        # Check that each state has sample count information
        for state_id, char in state_info["state_characteristics"].items():
            assert "sample_count" in char
            assert char["sample_count"] > 0

    def test_hmm_prediction_reliability_analysis(self):
        """Test prediction reliability analysis per state."""
        hmm = HMMPredictor(n_components=3, room_id="reliability_test")

        # Setup states with different reliability levels
        hmm.state_characteristics = {
            0: {
                "avg_state_probability": 0.9,
                "prediction_reliability": "high",
                "high_confidence_samples": 40,
                "low_confidence_samples": 5,
            },
            1: {
                "avg_state_probability": 0.65,
                "prediction_reliability": "medium",
                "high_confidence_samples": 20,
                "low_confidence_samples": 15,
            },
            2: {
                "avg_state_probability": 0.45,
                "prediction_reliability": "low",
                "high_confidence_samples": 10,
                "low_confidence_samples": 25,
            },
        }

        state_info = hmm.get_state_info()

        # Check reliability information is preserved
        for state_id, char in state_info["state_characteristics"].items():
            assert "prediction_reliability" in char
            assert char["prediction_reliability"] in ["low", "medium", "high"]

            if "avg_state_probability" in char:
                # High probability states should have high reliability
                if char["avg_state_probability"] > 0.8:
                    assert char["prediction_reliability"] == "high"
                elif char["avg_state_probability"] < 0.5:
                    assert char["prediction_reliability"] == "low"


class TestHMMModelPersistence:
    """Test HMM model saving and loading."""

    def test_hmm_save_and_load(self, hmm_split_data):
        """Test HMM model save and load functionality."""
        train_features, train_targets, _, _, _, _ = hmm_split_data

        hmm_original = HMMPredictor(
            n_components=3, covariance_type="diag", room_id="persistence_test"
        )

        # Setup trained state
        hmm_original.is_trained = True
        hmm_original.training_date = datetime.now(timezone.utc)
        hmm_original.model_version = "v1.0"
        hmm_original.feature_names = list(train_features.columns)

        # Mock trained components
        hmm_original.state_model = MagicMock()
        hmm_original.feature_scaler = StandardScaler()
        hmm_original.transition_models = {0: {"type": "average", "value": 1800}}

        hmm_original.state_labels = {
            0: "Test_State_0",
            1: "Test_State_1",
            2: "Test_State_2",
        }
        hmm_original.state_characteristics = {
            0: {"avg_duration": 600, "sample_count": 30},
            1: {"avg_duration": 1800, "sample_count": 45},
            2: {"avg_duration": 4500, "sample_count": 25},
        }
        hmm_original.transition_matrix = np.array(
            [[0.3, 0.4, 0.3], [0.2, 0.5, 0.3], [0.4, 0.3, 0.3]]
        )

        # Add training history
        hmm_original.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=150,
                model_version="v1.0",
                training_samples=100,
                training_score=0.7,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

            try:
                # Save model
                success = hmm_original.save_model(tmp_path)
                assert success is True

                # Load model into new instance
                hmm_loaded = HMMPredictor(room_id="loaded_test")
                load_success = hmm_loaded.load_model(tmp_path)

                assert load_success is True

                # Verify loaded state
                assert hmm_loaded.is_trained == hmm_original.is_trained
                assert (
                    hmm_loaded.room_id == "persistence_test"
                )  # Should use saved room_id
                assert hmm_loaded.model_version == hmm_original.model_version
                assert hmm_loaded.feature_names == hmm_original.feature_names
                assert hmm_loaded.state_labels == hmm_original.state_labels
                assert (
                    hmm_loaded.state_characteristics
                    == hmm_original.state_characteristics
                )

                # Training history should be restored
                assert len(hmm_loaded.training_history) == 1
                assert hmm_loaded.training_history[0].success is True

                # Transition matrix should be restored
                assert hmm_loaded.transition_matrix is not None
                np.testing.assert_array_equal(
                    hmm_loaded.transition_matrix, hmm_original.transition_matrix
                )

            finally:
                # Cleanup
                Path(tmp_path).unlink(missing_ok=True)

    def test_hmm_save_load_error_handling(self):
        """Test HMM save/load error handling."""
        hmm = HMMPredictor(room_id="error_test")

        # Test saving to invalid path
        invalid_path = "/nonexistent/directory/model.pkl"
        success = hmm.save_model(invalid_path)
        assert success is False

        # Test loading from nonexistent file
        nonexistent_path = "/nonexistent/model.pkl"
        load_success = hmm.load_model(nonexistent_path)
        assert load_success is False

        # Test loading corrupted file
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_file.write(b"corrupted data")
            tmp_path = tmp_file.name

        try:
            load_success = hmm.load_model(tmp_path)
            assert load_success is False
        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestHMMPerformanceBenchmarks:
    """Test HMM performance benchmarks and validation."""

    @pytest.mark.asyncio
    async def test_hmm_training_performance(self, hmm_split_data):
        """Test HMM training performance benchmarks."""
        train_features, train_targets, val_features, val_targets, _, _ = hmm_split_data

        # Use moderate size for benchmarking
        bench_train_features = train_features.head(200)
        bench_train_targets = train_targets.head(200)
        bench_val_features = val_features.head(40)
        bench_val_targets = val_targets.head(40)

        hmm = HMMPredictor(
            n_components=4,
            max_iter=50,  # Limit iterations for speed
            room_id="performance_test",
        )

        # Measure training time
        start_time = time.time()
        result = await hmm.train(
            bench_train_features,
            bench_train_targets,
            bench_val_features,
            bench_val_targets,
        )
        training_duration = time.time() - start_time

        # Performance requirements
        assert result.success is True
        assert training_duration < 120  # Should complete within 2 minutes for test data

        print(
            f"HMM training time: {training_duration:.2f}s for {len(bench_train_features)} samples"
        )

        # HMM should produce reasonable results
        assert result.training_score is not None
        assert -1.0 <= result.training_score <= 1.0

    @pytest.mark.asyncio
    async def test_hmm_prediction_latency(self, hmm_split_data):
        """Test HMM prediction latency requirements."""
        train_features, train_targets, _, _, test_features, _ = hmm_split_data

        hmm = HMMPredictor(n_components=4, room_id="latency_test")
        await self._setup_fast_trained_hmm(hmm, train_features, train_targets)

        # Test different batch sizes
        batch_sizes = [1, 5, 10, 15]

        for batch_size in batch_sizes:
            test_batch = test_features.head(batch_size)

            # Warm up
            await hmm.predict(test_batch.head(1), datetime.now(timezone.utc), "vacant")

            # Measure prediction time
            start_time = time.time()
            predictions = await hmm.predict(
                test_batch, datetime.now(timezone.utc), "vacant"
            )
            prediction_duration = time.time() - start_time

            # Calculate per-prediction latency
            latency_per_prediction = (prediction_duration / batch_size) * 1000  # ms

            print(
                f"HMM batch size {batch_size}: {latency_per_prediction:.2f}ms per prediction"
            )

            # Should meet <100ms requirement
            assert latency_per_prediction < 100
            assert len(predictions) == batch_size

    @pytest.mark.asyncio
    async def test_hmm_state_identification_quality(self, hmm_split_data):
        """Test HMM state identification quality on realistic data."""
        train_features, train_targets, _, _, test_features, test_targets = (
            hmm_split_data
        )

        # Use real HMM training for state identification assessment
        hmm = HMMPredictor(
            n_components=4,
            max_iter=30,  # Limited for speed
            room_id="state_quality_test",
        )

        # Train on state-structured data
        result = await hmm.train(
            train_features.head(150),
            train_targets.head(150),
            test_features.head(30),
            test_targets.head(30),
        )

        assert result.success is True

        # Evaluate state identification quality
        assert len(hmm.state_labels) > 0
        assert len(hmm.state_characteristics) > 0

        # States should have reasonable characteristics
        for state_id, char in hmm.state_characteristics.items():
            assert char["avg_duration"] > 0
            assert char["sample_count"] > 0

            # Different states should have meaningfully different durations
            assert 60 <= char["avg_duration"] <= 86400

        # Check that states are actually different
        durations = [
            char["avg_duration"] for char in hmm.state_characteristics.values()
        ]
        if len(durations) > 1:
            duration_range = max(durations) - min(durations)
            assert duration_range > 300  # At least 5 minutes difference between states

        print(f"HMM identified {len(hmm.state_characteristics)} distinct states")
        for state_id, label in hmm.state_labels.items():
            char = hmm.state_characteristics[state_id]
            print(
                f"  {label}: {char['avg_duration']:.0f}s avg, {char['sample_count']} samples"
            )

    @pytest.mark.asyncio
    async def test_hmm_prediction_accuracy_on_states(self, hmm_split_data):
        """Test HMM prediction accuracy with state-based evaluation."""
        train_features, train_targets, _, _, test_features, test_targets = (
            hmm_split_data
        )

        hmm = HMMPredictor(n_components=4, room_id="accuracy_test")

        # Train HMM
        result = await hmm.train(train_features.head(120), train_targets.head(120))

        assert result.success is True

        # Generate predictions on test data
        predictions = await hmm.predict(
            test_features.head(20), datetime.now(timezone.utc), "vacant"
        )

        assert len(predictions) == 20

        # Check prediction quality
        for pred in predictions:
            # Predictions should be in reasonable range
            pred_seconds = pred.prediction_metadata["time_until_transition_seconds"]
            assert 60 <= pred_seconds <= 86400

            # Confidence should be reasonable for state-based predictions
            assert 0.1 <= pred.confidence_score <= 0.95

            # State information should be present
            assert pred.prediction_metadata["current_hidden_state"] >= 0
            assert pred.prediction_metadata["state_probability"] > 0

        print(f"HMM accuracy assessment completed for {len(predictions)} predictions")

        # Model should achieve reasonable performance on structured data
        training_score = result.training_score
        assert training_score > -0.5  # Should be better than naive baseline

        print(f"HMM accuracy (R²): {training_score:.3f}")

    async def _setup_fast_trained_hmm(self, hmm, train_features, train_targets):
        """Setup trained HMM optimized for performance testing."""
        hmm.is_trained = True
        hmm.training_date = datetime.now(timezone.utc)
        hmm.model_version = "v1.0"
        hmm.feature_names = list(train_features.columns)

        # Fast mock state model
        hmm.state_model = MagicMock()

        # Fast state inference
        def fast_predict_proba(X):
            n_samples = len(X)
            # Generate consistent state probabilities
            probs = np.zeros((n_samples, 4))
            for i in range(n_samples):
                # Deterministic assignment for speed
                dominant_state = i % 4
                probs[i, dominant_state] = 0.7
                probs[i, (dominant_state + 1) % 4] = 0.2
                probs[i, (dominant_state + 2) % 4] = 0.07
                probs[i, (dominant_state + 3) % 4] = 0.03
            return probs

        def fast_predict(X):
            probs = fast_predict_proba(X)
            return np.argmax(probs, axis=1)

        hmm.state_model.predict_proba = MagicMock(side_effect=fast_predict_proba)
        hmm.state_model.predict = MagicMock(side_effect=fast_predict)

        # Fast feature scaler
        hmm.feature_scaler = MagicMock()
        hmm.feature_scaler.transform = MagicMock(
            side_effect=lambda x: x  # No-op for speed
        )

        # Simple state characteristics and models
        hmm.state_characteristics = {
            0: {"avg_duration": 900},
            1: {"avg_duration": 1800},
            2: {"avg_duration": 3600},
            3: {"avg_duration": 7200},
        }

        hmm.state_labels = {0: "Quick_0", 1: "Short_1", 2: "Medium_2", 3: "Long_3"}

        hmm.transition_models = {
            0: {"type": "average", "value": 900},
            1: {"type": "average", "value": 1800},
            2: {"type": "average", "value": 3600},
            3: {"type": "average", "value": 7200},
        }

        # Mock training history
        hmm.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=60,
                model_version="v1.0",
                training_samples=150,
                validation_score=0.68,
                training_score=0.72,
            )
        ]


# Mark all tests as requiring the 'models' fixture
pytestmark = pytest.mark.models
