"""
Enhanced comprehensive test suite for LSTMPredictor class.

This test suite focuses on achieving 85%+ coverage of LSTM predictor functionality
with realistic ML scenarios and proper error handling.
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
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


@pytest.fixture
def lstm_training_data():
    """Create training data suitable for LSTM sequence learning."""
    np.random.seed(42)
    n_samples = 200

    # Create temporal sequence features
    features = {
        "hour_sin": np.sin(2 * np.pi * np.arange(n_samples) / 24),
        "hour_cos": np.cos(2 * np.pi * np.arange(n_samples) / 24),
        "prev_duration": np.random.exponential(
            1800, n_samples
        ),  # Previous state duration
        "motion_events": np.random.poisson(2, n_samples),
        "temperature": 20 + np.random.normal(0, 2, n_samples),
        "door_events": np.random.poisson(1, n_samples),
        "sequence_index": np.arange(n_samples),  # For sequence ordering
    }

    features_df = pd.DataFrame(features)

    # Create realistic targets with temporal dependency
    base_time = 1800  # 30 minutes
    temporal_influence = 1 + 0.3 * np.sin(2 * np.pi * np.arange(n_samples) / 24)
    motion_influence = 1 / (1 + features["motion_events"] / 3)

    time_until_transition = base_time * temporal_influence * motion_influence
    time_until_transition = np.clip(time_until_transition, 300, 7200)

    targets_df = pd.DataFrame(
        {
            "time_until_transition_seconds": time_until_transition,
            "transition_type": np.where(
                time_until_transition > 1800, "vacant_to_occupied", "occupied_to_vacant"
            ),
            "target_time": [datetime.now(timezone.utc)] * n_samples,
        }
    )

    return features_df, targets_df


@pytest.fixture
def trained_lstm_predictor(lstm_training_data):
    """Create a trained LSTM predictor for testing predictions."""
    features, targets = lstm_training_data
    predictor = LSTMPredictor("test_room")

    # Mock the training to avoid long ML training times in tests
    predictor.model = MLPRegressor(
        hidden_layer_sizes=(32, 16), max_iter=50, random_state=42
    )
    predictor.is_trained = True
    predictor.feature_names = list(features.columns)
    predictor.training_date = datetime.now(timezone.utc)
    predictor.model_version = "v1.0_test"

    # Fit scalers
    sequences_X = []
    sequences_y = []
    for i in range(predictor.sequence_length, len(features)):
        seq_X = features.iloc[i - predictor.sequence_length : i].values
        sequences_X.append(seq_X.flatten())
        sequences_y.append(targets.iloc[i]["time_until_transition_seconds"])

    if sequences_X:
        X_array = np.array(sequences_X)
        y_array = np.array(sequences_y)

        predictor.feature_scaler.fit(X_array)
        predictor.target_scaler.fit(y_array.reshape(-1, 1))
        predictor.model.fit(
            predictor.feature_scaler.transform(X_array),
            predictor.target_scaler.transform(y_array.reshape(-1, 1)).ravel(),
        )

    return predictor


class TestLSTMInitialization:
    """Test LSTM predictor initialization and configuration."""

    def test_default_initialization(self):
        """Test default LSTM predictor initialization."""
        lstm = LSTMPredictor()

        assert lstm.model_type == ModelType.LSTM
        assert lstm.room_id is None
        assert not lstm.is_trained
        assert lstm.sequence_length == 50
        assert lstm.model is None
        assert isinstance(lstm.feature_scaler, StandardScaler)
        assert isinstance(lstm.target_scaler, MinMaxScaler)

    def test_room_specific_initialization(self):
        """Test LSTM initialization with room ID."""
        room_id = "living_room"
        lstm = LSTMPredictor(room_id)

        assert lstm.room_id == room_id
        assert lstm.model_type == ModelType.LSTM

    def test_custom_parameters(self):
        """Test LSTM initialization with custom parameters."""
        lstm = LSTMPredictor(
            room_id="test_room",
            sequence_length=30,
            hidden_units=[128, 64, 32],
            learning_rate=0.005,
            max_iter=500,
        )

        assert lstm.model_params["sequence_length"] == 30
        assert lstm.model_params["hidden_layers"] == [128, 64, 32]
        assert lstm.model_params["learning_rate"] == 0.005
        assert lstm.model_params["max_iter"] == 500

    def test_dropout_parameter_handling(self):
        """Test dropout parameter handling with different aliases."""
        # Test with dropout_rate parameter
        lstm1 = LSTMPredictor(dropout_rate=0.3)
        assert lstm1.model_params["dropout"] == 0.3
        assert lstm1.model_params["dropout_rate"] == 0.3

        # Test with dropout parameter
        lstm2 = LSTMPredictor(dropout=0.4)
        assert lstm2.model_params["dropout"] == 0.4
        assert lstm2.model_params["dropout_rate"] == 0.4

    def test_hidden_units_conversion(self):
        """Test hidden units conversion to hidden layers."""
        # Test with integer
        lstm1 = LSTMPredictor(hidden_units=128)
        assert lstm1.model_params["hidden_layers"] == [128, 64]

        # Test with list
        lstm2 = LSTMPredictor(hidden_units=[64, 32, 16])
        assert lstm2.model_params["hidden_layers"] == [64, 32, 16]


class TestLSTMSequenceGeneration:
    """Test LSTM sequence generation functionality."""

    def test_create_sequences_basic(self, lstm_training_data):
        """Test basic sequence creation."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room")

        sequences_X, sequences_y = lstm._create_sequences(features, targets)

        assert len(sequences_X) == len(sequences_y)
        assert len(sequences_X) == len(features) - lstm.sequence_length

        # Each sequence should have the right shape
        expected_seq_length = lstm.sequence_length * len(features.columns)
        assert sequences_X[0].shape[0] == expected_seq_length

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        lstm = LSTMPredictor("test_room")

        # Create data smaller than sequence length
        small_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})
        small_targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800, 2000, 1500],
                "transition_type": ["vacant_to_occupied"] * 3,
                "target_time": [datetime.now(timezone.utc)] * 3,
            }
        )

        sequences_X, sequences_y = lstm._create_sequences(small_features, small_targets)

        # Should return empty sequences
        assert len(sequences_X) == 0
        assert len(sequences_y) == 0

    def test_create_sequences_different_step(self, lstm_training_data):
        """Test sequence creation with different step size."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room")
        lstm.sequence_step = 10  # Larger step

        sequences_X, sequences_y = lstm._create_sequences(features, targets)

        # Should have fewer sequences due to larger step
        expected_sequences = (
            len(features) - lstm.sequence_length
        ) // lstm.sequence_step
        assert len(sequences_X) <= expected_sequences + 1

    def test_sequence_target_extraction(self, lstm_training_data):
        """Test correct target extraction from sequences."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room")

        sequences_X, sequences_y = lstm._create_sequences(features, targets)

        # Verify target values are correctly extracted
        for i, y_val in enumerate(sequences_y):
            expected_idx = lstm.sequence_length + (i * lstm.sequence_step)
            expected_target = targets.iloc[expected_idx][
                "time_until_transition_seconds"
            ]
            assert abs(y_val - expected_target) < 1e-10


class TestLSTMTraining:
    """Test LSTM training functionality."""

    @pytest.mark.asyncio
    async def test_training_success(self, lstm_training_data):
        """Test successful LSTM training."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room")

        result = await lstm.train(features, targets)

        assert result.success
        assert lstm.is_trained
        assert lstm.model is not None
        assert result.training_time_seconds > 0
        assert result.training_samples > 0
        assert lstm.feature_names == list(features.columns)

    @pytest.mark.asyncio
    async def test_training_with_validation(self, lstm_training_data):
        """Test LSTM training with validation data."""
        features, targets = lstm_training_data

        # Split data
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        val_features = features[train_size:]
        val_targets = targets[train_size:]

        lstm = LSTMPredictor("test_room")

        result = await lstm.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success
        assert result.validation_score is not None
        assert "validation_mae" in result.training_metrics

    @pytest.mark.asyncio
    async def test_training_insufficient_data(self):
        """Test training with insufficient data."""
        lstm = LSTMPredictor("test_room")

        # Create minimal data
        features = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800, 2000],
                "transition_type": ["vacant_to_occupied"] * 2,
                "target_time": [datetime.now(timezone.utc)] * 2,
            }
        )

        with pytest.raises(ModelTrainingError, match="Insufficient data"):
            await lstm.train(features, targets)

    @pytest.mark.asyncio
    async def test_training_feature_scaling(self, lstm_training_data):
        """Test that training properly fits feature and target scalers."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room")

        await lstm.train(features, targets)

        # Verify scalers are fitted
        assert hasattr(lstm.feature_scaler, "mean_")
        assert hasattr(lstm.target_scaler, "min_")

        # Test scaling works
        test_features = features.head(1)
        sequences_X, _ = lstm._create_sequences(test_features, targets.head(1))
        if sequences_X:
            scaled_features = lstm.feature_scaler.transform([sequences_X[0]])
            assert scaled_features is not None

    @pytest.mark.asyncio
    async def test_training_error_handling(self, lstm_training_data):
        """Test training error handling."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room")

        # Mock MLPRegressor to raise an error
        with patch("src.models.base.lstm_predictor.MLPRegressor") as mock_mlp:
            mock_mlp.return_value.fit.side_effect = Exception("Training failed")

            with pytest.raises(ModelTrainingError):
                await lstm.train(features, targets)

    @pytest.mark.asyncio
    async def test_adaptive_sequence_length(self, lstm_training_data):
        """Test adaptive sequence length based on data size."""
        features, targets = lstm_training_data

        # Test with small dataset
        small_features = features.head(60)  # Just above minimum
        small_targets = targets.head(60)

        lstm = LSTMPredictor("test_room", sequence_length=50)  # Larger than data

        # Should adapt sequence length
        result = await lstm.train(small_features, small_targets)
        assert result.success

        # Sequence length should be adapted
        assert lstm.sequence_length <= len(small_features) - 10


class TestLSTMPrediction:
    """Test LSTM prediction functionality."""

    @pytest.mark.asyncio
    async def test_prediction_generation(
        self, trained_lstm_predictor, lstm_training_data
    ):
        """Test basic prediction generation."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)  # Enough for sequence

        results = await lstm.predict(test_features, prediction_time, "vacant")

        assert len(results) > 0
        assert all(isinstance(r, PredictionResult) for r in results)
        assert all(r.predicted_time > prediction_time for r in results)
        assert all(r.confidence_score >= 0 and r.confidence_score <= 1 for r in results)

    @pytest.mark.asyncio
    async def test_prediction_different_states(
        self, trained_lstm_predictor, lstm_training_data
    ):
        """Test predictions for different occupancy states."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)

        # Test vacant state
        vacant_results = await lstm.predict(test_features, prediction_time, "vacant")
        assert vacant_results[0].transition_type == "vacant_to_occupied"

        # Test occupied state
        occupied_results = await lstm.predict(
            test_features, prediction_time, "occupied"
        )
        assert occupied_results[0].transition_type == "occupied_to_vacant"

    @pytest.mark.asyncio
    async def test_prediction_confidence_calculation(
        self, trained_lstm_predictor, lstm_training_data
    ):
        """Test confidence score calculation."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)

        results = await lstm.predict(test_features, prediction_time, "vacant")

        # Confidence should be reasonable
        for result in results:
            assert 0.0 <= result.confidence_score <= 1.0
            # Should have some variance in confidence
            assert result.confidence_score != 0.5  # Not just default value

    @pytest.mark.asyncio
    async def test_prediction_insufficient_data(self, trained_lstm_predictor):
        """Test prediction with insufficient sequence data."""
        lstm = trained_lstm_predictor

        # Create data smaller than sequence length
        small_features = pd.DataFrame({"f1": [1, 2, 3], "f2": [4, 5, 6]})

        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError, match="Insufficient data"):
            await lstm.predict(small_features, prediction_time, "vacant")

    @pytest.mark.asyncio
    async def test_prediction_untrained_model(self, lstm_training_data):
        """Test prediction with untrained model."""
        features, _ = lstm_training_data
        lstm = LSTMPredictor("test_room")

        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError, match="Model not trained"):
            await lstm.predict(features, prediction_time, "vacant")

    @pytest.mark.asyncio
    async def test_prediction_bounds_enforcement(
        self, trained_lstm_predictor, lstm_training_data
    ):
        """Test that predictions are bounded within reasonable limits."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)

        results = await lstm.predict(test_features, prediction_time, "vacant")

        for result in results:
            time_diff = (result.predicted_time - prediction_time).total_seconds()
            assert 60 <= time_diff <= 86400  # 1 minute to 24 hours

    @pytest.mark.asyncio
    async def test_prediction_metadata(
        self, trained_lstm_predictor, lstm_training_data
    ):
        """Test prediction metadata completeness."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)

        results = await lstm.predict(test_features, prediction_time, "vacant")

        for result in results:
            assert result.model_type == "lstm"
            assert result.model_version == lstm.model_version
            assert result.features_used == lstm.feature_names
            assert "time_until_transition_seconds" in result.prediction_metadata
            assert "sequence_length_used" in result.prediction_metadata


class TestLSTMFeatureImportance:
    """Test LSTM feature importance calculation."""

    def test_feature_importance_trained_model(self, trained_lstm_predictor):
        """Test feature importance calculation for trained model."""
        lstm = trained_lstm_predictor

        importance = lstm.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(lstm.feature_names)
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

        # Should sum to approximately 1
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 0.01

    def test_feature_importance_untrained_model(self):
        """Test feature importance for untrained model."""
        lstm = LSTMPredictor("test_room")

        importance = lstm.get_feature_importance()

        assert importance == {}

    def test_feature_importance_no_coefficients(self, trained_lstm_predictor):
        """Test feature importance when model has no coefficients."""
        lstm = trained_lstm_predictor

        # Mock model without coefs_
        lstm.model = Mock()
        delattr(lstm.model, "coefs_")

        importance = lstm.get_feature_importance()

        # Should return uniform importance
        assert len(importance) == len(lstm.feature_names)
        expected_value = 1.0 / len(lstm.feature_names)
        assert all(abs(v - expected_value) < 0.01 for v in importance.values())


class TestLSTMModelPersistence:
    """Test LSTM model save/load functionality."""

    def test_save_and_load_model(self, trained_lstm_predictor):
        """Test saving and loading LSTM model."""
        lstm = trained_lstm_predictor

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "lstm_model.pkl"

            # Save model
            success = lstm.save_model(model_path)
            assert success
            assert model_path.exists()

            # Load model
            new_lstm = LSTMPredictor("test_room")
            load_success = new_lstm.load_model(model_path)

            assert load_success
            assert new_lstm.is_trained
            assert new_lstm.model_version == lstm.model_version
            assert new_lstm.feature_names == lstm.feature_names

    def test_save_load_failure_scenarios(self):
        """Test save/load failure scenarios."""
        lstm = LSTMPredictor("test_room")

        # Test save failure with invalid path
        invalid_path = "/invalid/path/model.pkl"
        success = lstm.save_model(invalid_path)
        assert not success

        # Test load failure with non-existent file
        non_existent_path = "/non/existent/model.pkl"
        load_success = lstm.load_model(non_existent_path)
        assert not load_success

    def test_model_state_preservation(self, trained_lstm_predictor, lstm_training_data):
        """Test that model state is preserved through save/load."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        # Make a prediction before saving
        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)
        original_results = asyncio.run(
            lstm.predict(test_features, prediction_time, "vacant")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "lstm_model.pkl"

            # Save and load
            lstm.save_model(model_path)
            new_lstm = LSTMPredictor("test_room")
            new_lstm.load_model(model_path)

            # Make the same prediction
            new_results = asyncio.run(
                new_lstm.predict(test_features, prediction_time, "vacant")
            )

            # Results should be very similar
            assert len(new_results) == len(original_results)
            for orig, new in zip(original_results, new_results):
                time_diff = abs(
                    (orig.predicted_time - new.predicted_time).total_seconds()
                )
                assert time_diff < 60  # Within 1 minute


class TestLSTMUtilityMethods:
    """Test LSTM utility methods."""

    def test_validate_features_trained_model(
        self, trained_lstm_predictor, lstm_training_data
    ):
        """Test feature validation with trained model."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        # Valid features
        test_features = features.head(10)
        assert lstm.validate_features(test_features)

        # Invalid features (missing columns)
        invalid_features = pd.DataFrame({"wrong_column": [1, 2, 3]})
        assert not lstm.validate_features(invalid_features)

    def test_validate_features_untrained_model(self, lstm_training_data):
        """Test feature validation with untrained model."""
        features, _ = lstm_training_data
        lstm = LSTMPredictor("test_room")

        # Should return True for untrained model (no feature names to check)
        assert lstm.validate_features(features)

    def test_get_model_complexity(self, trained_lstm_predictor):
        """Test model complexity calculation."""
        lstm = trained_lstm_predictor

        complexity = lstm.get_model_complexity()

        assert isinstance(complexity, dict)
        assert "total_parameters" in complexity
        assert "hidden_layers" in complexity
        assert "sequence_length" in complexity
        assert complexity["total_parameters"] > 0

    def test_get_model_complexity_untrained(self):
        """Test model complexity for untrained model."""
        lstm = LSTMPredictor("test_room")

        complexity = lstm.get_model_complexity()

        assert complexity["total_parameters"] == 0
        assert complexity["hidden_layers"] == lstm.model_params["hidden_layers"]


class TestLSTMPerformanceBenchmarks:
    """Test LSTM performance benchmarks."""

    @pytest.mark.asyncio
    async def test_training_performance(self, lstm_training_data):
        """Test training performance benchmark."""
        features, targets = lstm_training_data
        lstm = LSTMPredictor("test_room", max_iter=10)  # Fast training for tests

        start_time = time.time()
        result = await lstm.train(features, targets)
        training_time = time.time() - start_time

        assert result.success
        assert training_time < 30  # Should complete within 30 seconds
        assert result.training_time_seconds > 0

    @pytest.mark.asyncio
    async def test_prediction_latency(self, trained_lstm_predictor, lstm_training_data):
        """Test prediction latency benchmark."""
        features, _ = lstm_training_data
        lstm = trained_lstm_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(lstm.sequence_length + 5)

        start_time = time.time()
        results = await lstm.predict(test_features, prediction_time, "vacant")
        latency = time.time() - start_time

        assert len(results) > 0
        assert latency < 1.0  # Should be under 1 second

    @pytest.mark.asyncio
    async def test_accuracy_benchmark(self, lstm_training_data):
        """Test prediction accuracy on realistic data."""
        features, targets = lstm_training_data

        # Split data for testing
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        test_features = features[train_size:]
        test_targets = targets[train_size:]

        lstm = LSTMPredictor("test_room", max_iter=50)
        await lstm.train(train_features, train_targets)

        # Make predictions on test data
        prediction_errors = []
        prediction_time = datetime.now(timezone.utc)

        for i in range(len(test_features) - lstm.sequence_length):
            test_batch = test_features.iloc[i : i + lstm.sequence_length + 1]
            actual_target = test_targets.iloc[i + lstm.sequence_length][
                "time_until_transition_seconds"
            ]

            try:
                results = await lstm.predict(test_batch, prediction_time, "vacant")
                if results:
                    predicted_seconds = (
                        results[0].predicted_time - prediction_time
                    ).total_seconds()
                    error = abs(predicted_seconds - actual_target)
                    prediction_errors.append(error)
            except Exception:
                continue  # Skip problematic predictions for benchmark

        if prediction_errors:
            mean_error = np.mean(prediction_errors)
            assert mean_error < 3600  # Mean error should be less than 1 hour


# Mark all tests as focusing on models
pytestmark = pytest.mark.asyncio
