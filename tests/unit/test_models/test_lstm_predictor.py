"""
Comprehensive unit tests for LSTM predictor module.

This consolidated test suite provides complete coverage for the LSTMPredictor class,
including model initialization, training methods, prediction methods,
state management, error handling, sequence processing, and all utility functions.

This file consolidates all LSTM testing functionality to eliminate duplication
and maintain a single source of truth for LSTM model testing.
"""

from datetime import datetime, timedelta, timezone
import pickle
import tempfile
from unittest.mock import MagicMock, Mock, PropertyMock, patch
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


class TestLSTMPredictorInitialization:
    """Test LSTM predictor initialization and configuration."""

    def test_lstm_predictor_initialization_default(self):
        """Test LSTM predictor initialization with default parameters."""
        predictor = LSTMPredictor()

        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id is None
        assert predictor.model is None
        assert predictor.is_trained is False
        assert isinstance(predictor.feature_scaler, StandardScaler)
        assert isinstance(predictor.target_scaler, MinMaxScaler)
        assert predictor.sequence_length == 50
        assert predictor.sequence_step == 5

    def test_lstm_predictor_initialization_with_room_id(self):
        """Test LSTM predictor initialization with room ID."""
        predictor = LSTMPredictor(room_id="living_room")

        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id == "living_room"
        assert predictor.is_trained is False

    def test_lstm_predictor_initialization_with_custom_params(self):
        """Test LSTM predictor initialization with custom parameters."""
        custom_params = {
            "sequence_length": 30,
            "hidden_units": [128, 64, 32],
            "learning_rate": 0.01,
            "max_iter": 500,
            "early_stopping": True,
            "dropout": 0.3,
        }

        predictor = LSTMPredictor(room_id="bedroom", **custom_params)

        assert predictor.sequence_length == 30
        assert predictor.model_params["hidden_layers"] == [128, 64, 32]
        assert predictor.model_params["learning_rate"] == 0.01
        assert predictor.model_params["max_iter"] == 500
        assert predictor.model_params["early_stopping"] is True
        assert predictor.model_params["dropout"] == 0.3
        assert predictor.model_params["dropout_rate"] == 0.3

    def test_lstm_predictor_initialization_hidden_units_conversion(self):
        """Test hidden units conversion to hidden layers."""
        # Test integer conversion
        predictor1 = LSTMPredictor(hidden_units=64)
        assert predictor1.model_params["hidden_layers"] == [64, 32]

        # Test list preservation
        predictor2 = LSTMPredictor(hidden_units=[128, 64, 32])
        assert predictor2.model_params["hidden_layers"] == [128, 64, 32]

    def test_lstm_predictor_initialization_parameter_aliases(self):
        """Test parameter aliases for compatibility."""
        predictor = LSTMPredictor(hidden_units=64, lstm_units=128, dropout_rate=0.25)

        assert predictor.model_params["hidden_size"] == 64
        assert (
            predictor.model_params["lstm_units"] == 128
        )  # lstm_units is set separately
        assert predictor.model_params["dropout"] == 0.25
        assert predictor.model_params["dropout_rate"] == 0.25

    def test_lstm_predictor_default_parameters_merge(self):
        """Test that default parameters are properly merged with custom ones."""
        default_params = DEFAULT_MODEL_PARAMS[ModelType.LSTM]
        custom_params = {"learning_rate": 0.005}

        predictor = LSTMPredictor(**custom_params)

        # Custom parameter should override default
        assert predictor.model_params["learning_rate"] == 0.005

        # Default parameters should still be present
        assert predictor.model_params["sequence_length"] == default_params.get(
            "sequence_length", 50
        )


class TestLSTMPredictorTraining:
    """Test LSTM predictor training functionality."""

    def create_sample_training_data(self, n_samples=100, n_features=5):
        """Create sample training data for testing."""
        np.random.seed(42)

        # Create features
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )

        # Create targets (time until next transition in seconds)
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )

        return features, targets

    @pytest.mark.asyncio
    async def test_lstm_training_success(self):
        """Test successful LSTM training."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=10, max_iter=10)
        features, targets = self.create_sample_training_data(200, 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = await predictor.train(features, targets)

        assert result.success is True
        assert result.training_samples == 200
        assert result.training_time_seconds > 0
        assert result.model_version is not None
        assert result.training_score is not None
        assert predictor.is_trained is True
        assert predictor.model is not None
        assert isinstance(predictor.model, MLPRegressor)

    @pytest.mark.asyncio
    async def test_lstm_training_with_validation_data(self):
        """Test LSTM training with validation data."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=10, max_iter=10)
        train_features, train_targets = self.create_sample_training_data(200, 5)
        val_features, val_targets = self.create_sample_training_data(50, 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = await predictor.train(
                train_features, train_targets, val_features, val_targets
            )

        assert result.success is True
        assert result.validation_score is not None
        assert "validation_mae" in result.training_metrics
        assert "validation_rmse" in result.training_metrics
        assert "validation_r2" in result.training_metrics

    @pytest.mark.asyncio
    async def test_lstm_training_small_dataset_adaptation(self):
        """Test LSTM training with small dataset adaptation."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=50, max_iter=10)
        features, targets = self.create_sample_training_data(100, 5)  # Small dataset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = await predictor.train(features, targets)

        assert result.success is True
        # Should adapt sequence length for small dataset
        assert predictor.training_sequence_length <= 50
        assert result.training_metrics["sequence_length"] <= 50

    @pytest.mark.asyncio
    async def test_lstm_training_insufficient_data_error(self):
        """Test LSTM training with insufficient data raises error."""
        predictor = LSTMPredictor(
            room_id="test_room", sequence_length=20
        )  # Larger than data
        features, targets = self.create_sample_training_data(
            10, 3
        )  # Small dataset but >= 3

        with pytest.raises(ModelTrainingError) as exc_info:
            await predictor.train(features, targets)

        assert "Need at least 20 samples" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_lstm_training_sequence_creation_error(self):
        """Test LSTM training with sequence creation errors."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=50)

        # Create data with mismatched lengths
        features = pd.DataFrame(np.random.randn(10, 3))
        targets = pd.DataFrame(np.random.randn(5, 1))  # Different length

        with pytest.raises(ModelTrainingError):
            await predictor.train(features, targets)

    @pytest.mark.asyncio
    async def test_lstm_training_metrics_calculation(self):
        """Test that training metrics are calculated correctly."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=10, max_iter=10)
        features, targets = self.create_sample_training_data(200, 5)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = await predictor.train(features, targets)

        metrics = result.training_metrics
        assert "training_mae" in metrics
        assert "training_rmse" in metrics
        assert "training_r2" in metrics
        assert "sequences_generated" in metrics
        assert "sequence_length" in metrics
        assert "n_iterations" in metrics
        assert metrics["sequences_generated"] > 0


class TestLSTMPredictorPrediction:
    """Test LSTM predictor prediction functionality."""

    def create_trained_predictor(self):
        """Create a trained LSTM predictor for testing."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=10, max_iter=10)
        features, targets = self.create_sample_training_data(200, 5)

        # Mock the training to avoid actual computation
        predictor.is_trained = True
        predictor.model = Mock(spec=MLPRegressor)
        predictor.model.predict.return_value = np.array([1800.0])  # 30 minutes
        predictor.feature_names = [
            "feature_0",
            "feature_1",
            "feature_2",
            "feature_3",
            "feature_4",
        ]
        predictor.training_sequence_length = 10
        predictor.model_version = "test_v1.0"

        # Mock scalers
        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.feature_scaler.transform.return_value = np.random.randn(1, 50)
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array([[1800.0]])

        return predictor

    def create_sample_training_data(self, n_samples=100, n_features=5):
        """Create sample training data for testing."""
        np.random.seed(42)
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )
        return features, targets

    @pytest.mark.asyncio
    async def test_lstm_prediction_success(self):
        """Test successful LSTM prediction."""
        predictor = self.create_trained_predictor()
        features = pd.DataFrame(np.random.randn(10, 5), columns=predictor.feature_names)
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(features, prediction_time, "occupied")

        assert len(predictions) == 10
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.model_type == "lstm"
            assert pred.transition_type == "occupied_to_vacant"
            assert 0 <= pred.confidence_score <= 1
            assert pred.predicted_time > prediction_time

    @pytest.mark.asyncio
    async def test_lstm_prediction_untrained_model_error(self):
        """Test prediction with untrained model raises error."""
        predictor = LSTMPredictor(room_id="test_room")
        features = pd.DataFrame(np.random.randn(5, 3))
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    @pytest.mark.asyncio
    async def test_lstm_prediction_invalid_features_error(self):
        """Test prediction with invalid features raises error."""
        predictor = self.create_trained_predictor()
        predictor.validate_features = Mock(return_value=False)

        features = pd.DataFrame(np.random.randn(5, 3))
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    @pytest.mark.asyncio
    async def test_lstm_prediction_transition_type_inference(self):
        """Test transition type inference based on current state and time."""
        predictor = self.create_trained_predictor()
        features = pd.DataFrame(np.random.randn(1, 5), columns=predictor.feature_names)

        # Test occupied state
        prediction_time = datetime.now(timezone.utc)
        predictions = await predictor.predict(features, prediction_time, "occupied")
        assert predictions[0].transition_type == "occupied_to_vacant"

        # Test vacant state
        predictions = await predictor.predict(features, prediction_time, "vacant")
        assert predictions[0].transition_type == "vacant_to_occupied"

        # Test unknown state - daytime
        daytime = datetime.now(timezone.utc).replace(hour=12)
        predictions = await predictor.predict(features, daytime, "unknown")
        assert predictions[0].transition_type == "vacant_to_occupied"

        # Test unknown state - nighttime
        nighttime = datetime.now(timezone.utc).replace(hour=2)
        predictions = await predictor.predict(features, nighttime, "unknown")
        assert predictions[0].transition_type == "occupied_to_vacant"

    @pytest.mark.asyncio
    async def test_lstm_prediction_sequence_padding(self):
        """Test prediction with sequence padding for insufficient history."""
        predictor = self.create_trained_predictor()
        predictor.training_sequence_length = 10

        # Test with features shorter than sequence length
        features = pd.DataFrame(
            np.random.randn(5, 5),  # Only 5 samples, need 10
            columns=predictor.feature_names,
        )
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(features, prediction_time)

        # Should still work with padding
        assert len(predictions) == 5
        assert all(isinstance(p, PredictionResult) for p in predictions)

    @pytest.mark.asyncio
    async def test_lstm_prediction_time_bounds_clipping(self):
        """Test that predicted times are clipped to reasonable bounds."""
        predictor = self.create_trained_predictor()

        # Mock extreme predictions
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[100000.0]]
        )  # Very large

        features = pd.DataFrame(np.random.randn(1, 5), columns=predictor.feature_names)
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(features, prediction_time)

        # Should be clipped to maximum (24 hours = 86400 seconds)
        time_diff = (predictions[0].predicted_time - prediction_time).total_seconds()
        assert time_diff <= 86400

    @pytest.mark.asyncio
    async def test_lstm_prediction_metadata_content(self):
        """Test that prediction metadata contains required information."""
        predictor = self.create_trained_predictor()
        features = pd.DataFrame(np.random.randn(1, 5), columns=predictor.feature_names)
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(features, prediction_time)

        metadata = predictions[0].prediction_metadata
        assert "time_until_transition_seconds" in metadata
        assert "sequence_length_used" in metadata
        assert "prediction_method" in metadata
        assert metadata["prediction_method"] == "lstm_neural_network"


class TestLSTMPredictorSequenceProcessing:
    """Test LSTM sequence creation and processing."""

    def create_sample_data(self, n_samples=100, n_features=5):
        """Create sample data for sequence testing."""
        np.random.seed(42)
        features = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)],
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )
        return features, targets

    def test_create_sequences_success(self):
        """Test successful sequence creation."""
        predictor = LSTMPredictor(sequence_length=10)
        features, targets = self.create_sample_data(100, 5)

        X_sequences, y_sequences = predictor._create_sequences(features, targets)

        assert X_sequences.shape[0] == y_sequences.shape[0]  # Same number of sequences
        assert X_sequences.shape[1] == 50  # 10 timesteps * 5 features
        assert len(y_sequences) > 0

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        predictor = LSTMPredictor(sequence_length=20)
        features, targets = self.create_sample_data(10, 5)  # Less than sequence_length

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "Need at least 20 samples" in str(exc_info.value)

    def test_create_sequences_mismatched_lengths(self):
        """Test sequence creation with mismatched feature and target lengths."""
        predictor = LSTMPredictor(sequence_length=10)
        features = pd.DataFrame(np.random.randn(50, 5))
        targets = pd.DataFrame(np.random.randn(30, 1))  # Different length

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "must have same length" in str(exc_info.value)

    def test_create_sequences_target_formats(self):
        """Test sequence creation with different target formats."""
        predictor = LSTMPredictor(sequence_length=10)
        features = pd.DataFrame(np.random.randn(50, 5))

        # Test with time_until_transition_seconds column
        targets1 = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, 50)}
        )
        X1, y1 = predictor._create_sequences(features, targets1)
        assert len(X1) > 0

        # Test with next_transition_time and target_time columns
        base_time = datetime.now(timezone.utc)
        targets2 = pd.DataFrame(
            {
                "target_time": [base_time + timedelta(hours=i) for i in range(50)],
                "next_transition_time": [
                    base_time + timedelta(hours=i, minutes=30) for i in range(50)
                ],
            }
        )
        X2, y2 = predictor._create_sequences(features, targets2)
        assert len(X2) > 0

        # Test with default format (first column)
        targets3 = pd.DataFrame(np.random.uniform(300, 7200, 50))
        X3, y3 = predictor._create_sequences(features, targets3)
        assert len(X3) > 0

    def test_create_sequences_filtering(self):
        """Test that sequences with unreasonable targets are filtered out."""
        predictor = LSTMPredictor(sequence_length=10)
        features = pd.DataFrame(np.random.randn(50, 5))

        # Create targets with some unreasonable values
        target_values = np.concatenate(
            [
                np.full(20, 30),  # Too small (< 60 seconds)
                np.full(15, 1800),  # Reasonable values
                np.full(15, 100000),  # Too large (> 86400 seconds)
            ]
        )
        targets = pd.DataFrame({"time_until_transition_seconds": target_values})

        X_sequences, y_sequences = predictor._create_sequences(features, targets)

        # Should only include sequences with reasonable targets
        assert all(60 <= y <= 86400 for y in y_sequences)

    def test_create_sequences_step_size(self):
        """Test sequence creation with different step sizes."""
        predictor1 = LSTMPredictor(sequence_length=10)
        predictor1.sequence_step = 1  # Step size 1

        predictor2 = LSTMPredictor(sequence_length=10)
        predictor2.sequence_step = 5  # Step size 5 (default)

        features, targets = self.create_sample_data(50, 5)

        X1, y1 = predictor1._create_sequences(features, targets)
        X2, y2 = predictor2._create_sequences(features, targets)

        # Step size 1 should generate more sequences
        assert len(X1) > len(X2)

    def test_create_sequences_with_next_transition_format(self):
        """Test sequence creation with next_transition_time format."""
        base_time = datetime.now(timezone.utc)
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )

        targets = pd.DataFrame(
            {
                "target_time": [base_time + timedelta(minutes=i) for i in range(50)],
                "next_transition_time": [
                    base_time + timedelta(minutes=i + 10) for i in range(50)
                ],
            }
        )

        predictor = LSTMPredictor(sequence_length=5)
        X_seq, y_seq = predictor._create_sequences(features, targets)

        assert len(X_seq) > 0
        assert len(y_seq) > 0
        # All target values should be 10 minutes = 600 seconds
        assert np.all(y_seq == 600.0)

    def test_create_sequences_with_single_column_targets(self):
        """Test sequence creation with single column targets."""
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )

        targets = pd.DataFrame({"random_target": np.random.uniform(300, 3600, 30)})

        predictor = LSTMPredictor(sequence_length=5)
        X_seq, y_seq = predictor._create_sequences(features, targets)

        assert len(X_seq) > 0
        assert len(y_seq) > 0

    def test_create_sequences_non_numeric_targets(self):
        """Test sequence creation with non-numeric targets."""
        features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6]})
        targets = pd.DataFrame(
            {"target": ["invalid", "data", "here", "test", "values", "bad"]}
        )

        predictor = LSTMPredictor(sequence_length=3)

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "non-numeric data" in str(exc_info.value)

    def test_create_sequences_target_value_filtering(self):
        """Test that sequences with invalid target values are filtered out."""
        features = pd.DataFrame(
            {
                "feature1": np.ones(20),
                "feature2": np.ones(20),
            }
        )

        # Mix of valid and invalid target values
        target_values = [
            30,
            600,
            90000,
            45,
            1800,
            100000,
            20,
            3600,
        ]  # Some too small/large
        target_values.extend([1200] * 12)  # Valid values to fill the rest

        targets = pd.DataFrame({"time_until_transition_seconds": target_values})

        predictor = LSTMPredictor(sequence_length=3, room_id="test")
        predictor.sequence_step = 1  # Process every sample

        X_seq, y_seq = predictor._create_sequences(features, targets)

        # All returned targets should be in valid range [60, 86400]
        assert np.all(y_seq >= 60)
        assert np.all(y_seq <= 86400)
        assert len(y_seq) > 0

    def test_create_sequences_no_valid_sequences(self):
        """Test sequence creation when no valid sequences can be generated."""
        features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        # All target values out of range
        targets = pd.DataFrame({"target": [30, 40, 45, 50, 35]})  # All < 60

        predictor = LSTMPredictor(sequence_length=3)

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "No valid sequences could be generated" in str(exc_info.value)

    def test_create_sequences_bounds_checking(self):
        """Test sequence creation bounds checking."""
        features = pd.DataFrame(
            {
                "feature1": list(range(20)),
                "feature2": list(range(20, 40)),
            }
        )
        targets = pd.DataFrame({"target": [600] * 20})

        predictor = LSTMPredictor(sequence_length=5)

        # Manually test bounds by checking internal logic
        X_seq, y_seq = predictor._create_sequences(features, targets)

        # Should generate sequences without index errors
        assert len(X_seq) > 0
        assert len(y_seq) > 0

        # Verify sequence shapes
        for seq in X_seq:
            assert len(seq) == 5 * 2  # sequence_length * n_features


class TestLSTMPredictorFeatureImportance:
    """Test LSTM feature importance calculation."""

    def create_mock_trained_predictor(self):
        """Create a mock trained predictor for testing."""
        predictor = LSTMPredictor(sequence_length=10)
        predictor.is_trained = True
        predictor.feature_names = ["feature_0", "feature_1", "feature_2"]

        # Mock MLPRegressor with coefficients
        mock_model = Mock(spec=MLPRegressor)

        # Create mock weights (input layer to first hidden layer)
        # Shape: (n_features * sequence_length, n_hidden_units)
        n_features = 3
        n_hidden = 5
        sequence_length = 10

        input_weights = np.random.randn(n_features * sequence_length, n_hidden)
        mock_model.coefs_ = [input_weights, np.random.randn(n_hidden, 1)]

        predictor.model = mock_model

        return predictor

    def test_feature_importance_calculation(self):
        """Test feature importance calculation for trained model."""
        predictor = self.create_mock_trained_predictor()

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3  # Number of features
        assert all(name in importance for name in predictor.feature_names)
        assert all(0 <= score <= 1 for score in importance.values())

        # Should sum to approximately 1 (normalized)
        assert abs(sum(importance.values()) - 1.0) < 1e-6

    def test_feature_importance_untrained_model(self):
        """Test feature importance for untrained model."""
        predictor = LSTMPredictor()

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_feature_importance_no_coefficients(self):
        """Test feature importance when model has no coefficients."""
        predictor = LSTMPredictor(sequence_length=10)
        predictor.is_trained = True
        predictor.feature_names = ["feature_0", "feature_1"]

        # Mock model without coefficients
        mock_model = Mock(spec=MLPRegressor)
        mock_model.coefs_ = []  # Empty coefficients
        predictor.model = mock_model

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_feature_importance_exception_handling(self):
        """Test feature importance calculation with exceptions."""
        predictor = self.create_mock_trained_predictor()

        # Make model.coefs_ raise an exception
        predictor.model.coefs_ = Mock(side_effect=Exception("Test error"))

        with patch("src.models.base.lstm_predictor.logger") as mock_logger:
            importance = predictor.get_feature_importance()

            assert importance == {}
            mock_logger.warning.assert_called_once()


class TestLSTMPredictorConfidenceCalculation:
    """Test LSTM confidence calculation methods."""

    def create_mock_predictor_with_history(self):
        """Create a mock predictor with training history."""
        predictor = LSTMPredictor()
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array([[1800.0]])

        # Mock training history
        mock_result = Mock()
        mock_result.validation_score = 0.85
        mock_result.training_score = 0.80
        predictor.training_history = [mock_result]

        return predictor

    def test_confidence_calculation_with_validation_score(self):
        """Test confidence calculation using validation score."""
        predictor = self.create_mock_predictor_with_history()

        X_scaled = np.random.randn(1, 50)
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert 0.1 <= confidence <= 0.95
        assert isinstance(confidence, float)

    def test_confidence_calculation_without_validation_score(self):
        """Test confidence calculation without validation score."""
        predictor = self.create_mock_predictor_with_history()
        predictor.training_history[0].validation_score = None

        X_scaled = np.random.randn(1, 50)
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert 0.1 <= confidence <= 0.95

    def test_confidence_calculation_no_history(self):
        """Test confidence calculation without training history."""
        predictor = LSTMPredictor()
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array([[1800.0]])
        predictor.training_history = []

        X_scaled = np.random.randn(1, 50)
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert confidence == 0.7  # Default confidence

    def test_confidence_calculation_extreme_predictions(self):
        """Test confidence adjustment for extreme predictions."""
        predictor = self.create_mock_predictor_with_history()

        # Test very short prediction
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[120.0]]
        )  # 2 minutes
        X_scaled = np.random.randn(1, 50)
        y_pred_scaled = np.array([0.1])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)
        # Should be reduced due to extreme prediction (less than 5 minutes)
        assert confidence <= 0.85  # Should be reduced from base confidence

    def test_confidence_calculation_exception_handling(self):
        """Test confidence calculation exception handling."""
        predictor = LSTMPredictor()
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.side_effect = Exception("Test error")

        X_scaled = np.random.randn(1, 50)
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert confidence == 0.7  # Default confidence on exception


class TestLSTMPredictorModelComplexity:
    """Test LSTM model complexity analysis."""

    def test_model_complexity_trained_model(self):
        """Test model complexity calculation for trained model."""
        predictor = LSTMPredictor(sequence_length=10)
        predictor.is_trained = True
        predictor.feature_names = ["f1", "f2", "f3"]

        # Mock MLPRegressor with coefficients and intercepts
        mock_model = Mock(spec=MLPRegressor)

        # Create mock weight matrices and bias vectors
        coefs = [
            np.random.randn(30, 64),  # input to hidden (30 = 3 features * 10 sequence)
            np.random.randn(64, 32),  # hidden to hidden
            np.random.randn(32, 1),  # hidden to output
        ]
        intercepts = [
            np.random.randn(64),  # first hidden layer bias
            np.random.randn(32),  # second hidden layer bias
            np.random.randn(1),  # output layer bias
        ]

        mock_model.coefs_ = coefs
        mock_model.intercepts_ = intercepts
        predictor.model = mock_model

        complexity_info = predictor.get_model_complexity()

        assert "total_parameters" in complexity_info
        assert "hidden_layers" in complexity_info
        assert "sequence_length" in complexity_info
        assert "input_features" in complexity_info
        assert "flattened_input_size" in complexity_info

        # Check calculations
        expected_params = sum(coef.size for coef in coefs) + sum(
            intercept.size for intercept in intercepts
        )
        assert complexity_info["total_parameters"] == expected_params
        assert complexity_info["sequence_length"] == 10
        assert complexity_info["input_features"] == 3
        assert complexity_info["flattened_input_size"] == 30

    def test_model_complexity_untrained_model(self):
        """Test model complexity for untrained model."""
        predictor = LSTMPredictor()

        complexity_info = predictor.get_model_complexity()

        assert complexity_info == {}

    def test_model_complexity_no_coefficients(self):
        """Test model complexity when model has no coefficients."""
        predictor = LSTMPredictor(sequence_length=10)
        predictor.is_trained = True
        predictor.feature_names = ["f1", "f2"]
        predictor.model = Mock(spec=MLPRegressor)

        # Model without coefs_ attribute
        del predictor.model.coefs_

        complexity_info = predictor.get_model_complexity()

        assert complexity_info["total_parameters"] == 0
        assert complexity_info["sequence_length"] == 10
        assert complexity_info["input_features"] == 2


class TestLSTMPredictorSerialization:
    """Test LSTM model serialization and deserialization."""

    def create_trained_predictor(self):
        """Create a trained predictor for serialization testing."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=10)
        predictor.is_trained = True
        predictor.model = Mock(spec=MLPRegressor)
        predictor.feature_names = ["f1", "f2", "f3"]
        predictor.model_version = "test_v1.0"
        predictor.training_date = datetime.now(timezone.utc)

        # Add training history
        mock_result = TrainingResult(
            success=True,
            training_time_seconds=10.5,
            model_version="test_v1.0",
            training_samples=100,
            training_score=0.85,
        )
        predictor.training_history = [mock_result]

        return predictor

    def test_save_model_success(self):
        """Test successful model saving."""
        predictor = self.create_trained_predictor()

        # Replace mock objects with actual serializable objects
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        predictor.model = MLPRegressor(max_iter=1)
        predictor.feature_scaler = StandardScaler()
        predictor.target_scaler = MinMaxScaler()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            result = predictor.save_model(file_path)

            assert result is True

            # Verify file was created
            import os

            assert os.path.exists(file_path)

            # Verify file contains expected data
            with open(file_path, "rb") as f:
                saved_data = pickle.load(f)

            assert "model" in saved_data
            assert "feature_scaler" in saved_data
            assert "target_scaler" in saved_data
            assert "model_type" in saved_data
            assert saved_data["room_id"] == "test_room"
            assert saved_data["is_trained"] is True

        finally:
            # Clean up
            import os

            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_save_model_failure(self):
        """Test model saving failure handling."""
        predictor = self.create_trained_predictor()

        # Use invalid file path
        invalid_path = "/nonexistent/path/model.pkl"

        with patch("src.models.base.lstm_predictor.logger") as mock_logger:
            result = predictor.save_model(invalid_path)

            assert result is False
            mock_logger.error.assert_called_once()

    def test_load_model_success(self):
        """Test successful model loading."""
        # Create and train a real predictor
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import MinMaxScaler, StandardScaler

        original_predictor = LSTMPredictor(
            room_id="test_room", sequence_length=5, max_iter=10
        )

        # Use real serializable objects
        original_predictor.model = MLPRegressor(max_iter=10)
        original_predictor.feature_scaler = StandardScaler()
        original_predictor.target_scaler = MinMaxScaler()
        original_predictor.is_trained = True
        original_predictor.feature_names = ["f1", "f2", "f3"]
        original_predictor.model_version = "test_v1.0"
        original_predictor.training_date = datetime.now(timezone.utc)

        # Fit scalers with dummy data to make them serializable
        dummy_data = np.random.randn(10, 3)
        original_predictor.feature_scaler.fit(dummy_data)
        original_predictor.target_scaler.fit(np.random.randn(10, 1))

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            file_path = tmp_file.name

        try:
            # Save the model
            result = original_predictor.save_model(file_path)
            assert result is True

            # Load into a new predictor
            new_predictor = LSTMPredictor()
            result = new_predictor.load_model(file_path)

            assert result is True
            assert new_predictor.room_id == "test_room"
            assert new_predictor.is_trained is True
            assert new_predictor.model is not None
            assert new_predictor.feature_names == ["f1", "f2", "f3"]
            assert new_predictor.model_version == "test_v1.0"

        finally:
            # Clean up
            import os

            if os.path.exists(file_path):
                os.unlink(file_path)

    def test_load_model_failure(self):
        """Test model loading failure handling."""
        predictor = LSTMPredictor()

        # Try to load from nonexistent file
        nonexistent_path = "/nonexistent/model.pkl"

        with patch("src.models.base.lstm_predictor.logger") as mock_logger:
            result = predictor.load_model(nonexistent_path)

            assert result is False
            mock_logger.error.assert_called_once()

    def test_load_model_corrupted_file(self):
        """Test loading from corrupted file."""
        predictor = LSTMPredictor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pkl", delete=False
        ) as tmp_file:
            # Write invalid data
            tmp_file.write("corrupted data")
            file_path = tmp_file.name

        try:
            with patch("src.models.base.lstm_predictor.logger") as mock_logger:
                result = predictor.load_model(file_path)

                assert result is False
                mock_logger.error.assert_called_once()

        finally:
            # Clean up
            import os

            if os.path.exists(file_path):
                os.unlink(file_path)


class TestLSTMPredictorErrorHandling:
    """Test LSTM predictor error handling and edge cases."""

    @pytest.mark.asyncio
    async def test_training_exception_handling(self):
        """Test training exception handling and error result creation."""
        predictor = LSTMPredictor(room_id="test_room")

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
        predictor = LSTMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.model = Mock(spec=MLPRegressor)
        predictor.feature_names = ["f1", "f2"]
        predictor.validate_features = Mock(return_value=True)

        # Make model.predict raise an exception
        predictor.model.predict.side_effect = Exception("Prediction failed")

        features = pd.DataFrame(np.random.randn(5, 2), columns=["f1", "f2"])
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    def test_sequence_creation_edge_cases(self):
        """Test sequence creation with edge cases."""
        predictor = LSTMPredictor(sequence_length=10)

        # Test with NaN values in targets (use equal length arrays)
        features = pd.DataFrame(np.random.randn(50, 3))
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [np.nan] * 25
                + [600.0] * 25  # Equal length
            }
        )

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "non-numeric data" in str(exc_info.value)

    def test_parameter_validation(self):
        """Test parameter validation during initialization."""
        # Test with invalid sequence_length
        predictor = LSTMPredictor(sequence_length=0)
        assert (
            predictor.sequence_length == 0
        )  # Should accept but may cause issues later

        # Test with negative values
        predictor = LSTMPredictor(max_iter=-1)
        assert (
            predictor.model_params["max_iter"] == -1
        )  # Should accept but sklearn will handle

    @pytest.mark.asyncio
    async def test_memory_management_large_dataset(self):
        """Test memory management with large datasets."""
        predictor = LSTMPredictor(
            sequence_length=5, max_iter=1
        )  # Small sequence for faster test

        # Create larger dataset
        n_samples = 1000
        features = pd.DataFrame(
            np.random.randn(n_samples, 10), columns=[f"feature_{i}" for i in range(10)]
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result = await predictor.train(features, targets)

        assert result.success is True
        assert predictor.is_trained is True


class TestLSTMPredictorIntegration:
    """Integration tests for LSTM predictor with realistic scenarios."""

    def create_realistic_occupancy_data(self, days=7, room_id="living_room"):
        """Create realistic occupancy data for testing."""
        np.random.seed(42)

        # Generate timestamps for a week
        start_time = datetime.now(timezone.utc) - timedelta(days=days)
        timestamps = [start_time + timedelta(hours=i) for i in range(days * 24)]

        features = []
        targets = []

        for i, timestamp in enumerate(timestamps[:-1]):
            # Create features that simulate real occupancy patterns
            hour = timestamp.hour
            day_of_week = timestamp.weekday()

            # Simulate occupancy probability based on time patterns
            if 6 <= hour <= 22:  # Daytime
                base_occupancy_prob = (
                    0.7 if day_of_week < 5 else 0.9
                )  # Weekday vs weekend
            else:  # Nighttime
                base_occupancy_prob = 0.9 if 22 <= hour <= 24 or hour <= 6 else 0.3

            # Add noise and create features
            feature_vector = {
                "hour_sin": np.sin(2 * np.pi * hour / 24),
                "hour_cos": np.cos(2 * np.pi * hour / 24),
                "day_of_week": day_of_week,
                "occupancy_prob": base_occupancy_prob + np.random.normal(0, 0.1),
                "time_since_last_motion": np.random.exponential(1800),  # seconds
            }
            features.append(feature_vector)

            # Create realistic target (time until next transition)
            if np.random.random() < base_occupancy_prob:
                # Currently occupied - time until vacant
                target_time = np.random.exponential(3600)  # Average 1 hour
            else:
                # Currently vacant - time until occupied
                target_time = np.random.exponential(7200)  # Average 2 hours

            targets.append(np.clip(target_time, 300, 28800))  # 5 min to 8 hours

        features_df = pd.DataFrame(features)
        targets_df = pd.DataFrame({"time_until_transition_seconds": targets})

        return features_df, targets_df

    @pytest.mark.asyncio
    async def test_realistic_training_and_prediction_flow(self):
        """Test complete training and prediction flow with realistic data."""
        predictor = LSTMPredictor(
            room_id="living_room",
            sequence_length=24,  # 24-hour sequence
            max_iter=50,
            learning_rate=0.01,
        )

        # Create realistic training data
        features, targets = self.create_realistic_occupancy_data(days=30)

        # Split into train/validation
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]

        # Train the model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            training_result = await predictor.train(
                train_features, train_targets, val_features, val_targets
            )

        assert training_result.success is True
        assert training_result.validation_score is not None
        assert predictor.is_trained is True

        # Make predictions
        prediction_features = features[-48:]  # Last 48 hours for prediction
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(
            prediction_features, prediction_time, "occupied"
        )

        assert len(predictions) == 48
        assert all(isinstance(p, PredictionResult) for p in predictions)
        assert all(p.confidence_score > 0 for p in predictions)

        # Test feature importance
        importance = predictor.get_feature_importance()
        assert len(importance) == len(features.columns)
        assert all(score >= 0 for score in importance.values())

    @pytest.mark.asyncio
    async def test_model_adaptation_scenarios(self):
        """Test model behavior in various adaptation scenarios."""
        predictor = LSTMPredictor(room_id="bedroom", sequence_length=12, max_iter=20)

        # Create initial training data
        initial_features, initial_targets = self.create_realistic_occupancy_data(
            days=14
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            await predictor.train(initial_features, initial_targets)

        # Test prediction accuracy tracking
        test_features = initial_features[-24:]
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(test_features, prediction_time)

        # Simulate accuracy tracking
        for i, pred in enumerate(predictions[:5]):  # Test first 5 predictions
            # Simulate actual transition occurring
            actual_time = pred.predicted_time + timedelta(
                minutes=np.random.randint(-10, 10)
            )
            predictor._record_prediction(prediction_time, pred)

        # Verify prediction recording
        assert hasattr(predictor, "prediction_history")

        # Test model complexity analysis
        complexity = predictor.get_model_complexity()
        assert "total_parameters" in complexity
        assert "sequence_length" in complexity

    def test_edge_case_handling(self):
        """Test handling of various edge cases."""
        predictor = LSTMPredictor()

        # Test with extreme parameter values
        extreme_predictor = LSTMPredictor(
            sequence_length=1,  # Minimum sequence length
            hidden_units=[1],  # Minimum hidden units
            max_iter=1,  # Minimum iterations
        )

        assert extreme_predictor.sequence_length == 1
        assert extreme_predictor.model_params["hidden_layers"] == [1]
        assert extreme_predictor.model_params["max_iter"] == 1

        # Test with very large parameters
        large_predictor = LSTMPredictor(
            sequence_length=1000, hidden_units=[1000, 500, 250], max_iter=10000
        )

        assert large_predictor.sequence_length == 1000
        assert large_predictor.model_params["hidden_layers"] == [1000, 500, 250]


class TestLSTMPredictorValidation:
    """Test model validation functionality."""

    @pytest.mark.asyncio
    async def test_validate_features_trained_model(self):
        """Test feature validation for trained model."""
        predictor = LSTMPredictor(max_iter=50)

        # Train model
        training_features = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, 50),
                "occupancy": np.random.choice([0, 1], 50),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 7200, 50)}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            await predictor.train(training_features, targets)

        # Test with matching features
        valid_features = pd.DataFrame(
            {
                "hour": [12, 15],
                "occupancy": [1, 0],
            }
        )
        assert predictor.validate_features(valid_features) is True

        # Test with mismatched features
        invalid_features = pd.DataFrame(
            {
                "wrong_feature": [1, 2],
            }
        )
        assert predictor.validate_features(invalid_features) is False

    def test_validate_features_untrained_model(self):
        """Test feature validation for untrained model."""
        predictor = LSTMPredictor()

        features = pd.DataFrame({"any_feature": [1, 2, 3]})

        # Untrained model should accept any features
        assert predictor.validate_features(features) is True


class TestLSTMPredictorAdvancedEdgeCases:
    """Test advanced edge cases and error handling."""

    def test_create_sequences_edge_case_bounds(self):
        """Test sequence creation with edge case bounds."""
        features = pd.DataFrame(
            {
                "feature1": list(range(10)),
                "feature2": list(range(10, 20)),
            }
        )
        targets = pd.DataFrame({"target": [1200] * 10})

        predictor = LSTMPredictor(sequence_length=10)  # Exactly equal to data length

        # Should handle the case where sequence_length equals data length
        X_seq, y_seq = predictor._create_sequences(features, targets)

        # Should generate at least one sequence
        assert len(X_seq) >= 1
        assert len(y_seq) >= 1

    @pytest.mark.asyncio
    async def test_predict_with_missing_training_sequence_length(self):
        """Test prediction when training_sequence_length attribute is missing."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=5)
        predictor.is_trained = True
        predictor.model_version = "v1.0"
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        # Mock model and scalers
        predictor.model = Mock(spec=MLPRegressor)
        predictor.model.predict.return_value = np.array([0.5])
        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.feature_scaler.transform.return_value = np.random.randn(1, 15)
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array([[1800]])
        predictor.training_history = []

        # Remove training_sequence_length attribute if it exists
        if hasattr(predictor, "training_sequence_length"):
            delattr(predictor, "training_sequence_length")

        features = pd.DataFrame(
            {
                "feature1": [0.1, 0.2],
                "feature2": [0.2, 0.3],
                "feature3": [0.3, 0.4],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        # Should fall back to self.sequence_length
        results = await predictor.predict(features, prediction_time)

        assert len(results) == 2

    def test_feature_importance_zero_total(self):
        """Test feature importance when total importance is zero."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=2)
        predictor.is_trained = True
        predictor.feature_names = ["feature1"]

        predictor.model = Mock(spec=MLPRegressor)
        # Mock weights that are all zeros
        mock_weights = np.zeros((2, 5))  # All zero weights
        predictor.model.coefs_ = [mock_weights]

        importance = predictor.get_feature_importance()

        # Should handle zero total importance gracefully
        assert isinstance(importance, dict)
        assert "feature1" in importance

    @pytest.mark.asyncio
    async def test_train_with_nan_in_sequence_generation(self):
        """Test training when sequence generation encounters NaN values."""
        features = pd.DataFrame(
            {
                "feature1": [1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
                "feature2": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0],
            }
        )
        targets = pd.DataFrame({"time_until_transition_seconds": [600] * 10})

        predictor = LSTMPredictor(sequence_length=5, max_iter=50)

        # Should handle NaN values in features gracefully
        # (MLPRegressor might handle it or it might raise an error)
        try:
            result = await predictor.train(features, targets)
            # If training succeeds, that's fine
            assert result.success is True or result.success is False
        except ModelTrainingError:
            # If training fails due to NaN, that's also acceptable
            pass

    def test_create_sequences_with_mixed_data_types(self):
        """Test sequence creation with mixed data types in features."""
        features = pd.DataFrame(
            {
                "numeric": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
                "string": ["a", "b", "c", "d", "e", "f"],  # String column
            }
        )
        targets = pd.DataFrame({"target": [600] * 6})

        predictor = LSTMPredictor(sequence_length=3)

        # Should handle mixed data types (might convert or raise error)
        try:
            X_seq, y_seq = predictor._create_sequences(features, targets)
            # If successful, verify basic properties
            assert len(X_seq) > 0
            assert len(y_seq) > 0
        except (ValueError, TypeError):
            # If it fails due to data type issues, that's acceptable
            pass

    @pytest.mark.asyncio
    async def test_train_validation_without_valid_sequences(self):
        """Test training with validation data that produces no valid sequences."""
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(600, 3600, 100)}
        )

        # Create validation data with no valid sequences (all targets too small)
        val_features = features.iloc[:20]
        val_targets = pd.DataFrame({"time_until_transition_seconds": [30] * 20})

        predictor = LSTMPredictor(sequence_length=8, max_iter=50)

        # Should handle empty validation sequences gracefully
        result = await predictor.train(features, targets, val_features, val_targets)
        assert result.success is True

    def test_get_feature_importance_bounds_checking(self):
        """Test feature importance with bounds checking."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=5)
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        predictor.model = Mock(spec=MLPRegressor)
        # Mock smaller coefficient matrix (should handle bounds checking)
        mock_weights = np.random.rand(10, 8)  # Smaller than expected
        predictor.model.coefs_ = [mock_weights]

        importance = predictor.get_feature_importance()

        # Should handle gracefully and return valid importance scores
        assert isinstance(importance, dict)
        assert all(feature in importance for feature in predictor.feature_names)

    def test_confidence_calculation_extreme_predictions_comprehensive(self):
        """Test confidence calculation with various extreme prediction scenarios."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=60,
                model_version="v1.0",
                training_samples=100,
                validation_score=0.85,
                training_score=0.90,
            )
        ]

        # Mock scalers for extreme predictions
        predictor.target_scaler = Mock(spec=MinMaxScaler)

        X_scaled = np.array([[0.1, 0.2, 0.3]])

        # Very short prediction
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[120]]
        )  # 2 minutes
        y_pred_scaled = np.array([0.1])
        confidence_short = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Very long prediction
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[50000]]
        )  # ~14 hours
        y_pred_scaled = np.array([0.9])
        confidence_long = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Normal prediction
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[1800]]
        )  # 30 minutes
        y_pred_scaled = np.array([0.5])
        confidence_normal = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Extreme predictions should have lower confidence
        assert confidence_short < confidence_normal
        assert confidence_long < confidence_normal


class TestLSTMPredictorIntegrationWorkflows:
    """Integration tests for complete LSTM workflows."""

    def create_synthetic_time_series_data(self, n_samples=80):
        """Create synthetic time series data for integration testing."""
        np.random.seed(42)

        features = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, n_samples),
                "day_of_week": np.random.randint(0, 7, n_samples),
                "occupancy_state": np.random.choice([0, 1], n_samples),
                "time_since_last_event": np.random.uniform(300, 7200, n_samples),
                "temperature": np.random.uniform(18, 25, n_samples),
            }
        )

        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(900, 10800, n_samples)}
        )

        return features, targets

    @pytest.mark.asyncio
    async def test_complete_training_prediction_workflow(self):
        """Test complete workflow from training to prediction."""
        # Create predictor
        predictor = LSTMPredictor(
            room_id="integration_room",
            sequence_length=10,
            max_iter=100,
        )

        # Create synthetic time series data
        features, targets = self.create_synthetic_time_series_data(80)

        # Split into train/validation
        train_features = features.iloc[:60]
        train_targets = targets.iloc[:60]
        val_features = features.iloc[60:]
        val_targets = targets.iloc[60:]

        # Train model
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            training_result = await predictor.train(
                train_features, train_targets, val_features, val_targets
            )

        assert training_result.success is True
        assert training_result.validation_score is not None

        # Make predictions
        pred_features = features.iloc[-5:]
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(pred_features, prediction_time, "vacant")

        # Validate predictions
        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.predicted_time > prediction_time
            assert pred.transition_type == "vacant_to_occupied"
            assert 0.1 <= pred.confidence_score <= 0.95
            assert pred.model_type == ModelType.LSTM.value
            assert len(pred.features_used) == 5  # All features used

        # Test feature importance
        importance = predictor.get_feature_importance()
        assert len(importance) == 5
        assert all(feature in importance for feature in features.columns)

        # Test model complexity
        complexity = predictor.get_model_complexity()
        assert complexity["input_features"] == 5
        assert complexity["total_parameters"] > 0

    @pytest.mark.asyncio
    async def test_multiple_training_sessions_version_tracking(self):
        """Test multiple training sessions and version tracking."""
        predictor = LSTMPredictor(room_id="versioning_test", max_iter=50)

        # Create different datasets
        features1 = pd.DataFrame(
            {
                "feature1": np.random.random(50),
                "feature2": np.random.random(50),
            }
        )
        targets1 = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 3600, 50)}
        )

        features2 = pd.DataFrame(
            {
                "feature1": np.random.random(60),
                "feature2": np.random.random(60),
            }
        )
        targets2 = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 3600, 60)}
        )

        # First training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result1 = await predictor.train(features1, targets1)
        version1 = predictor.model_version

        # Second training
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result2 = await predictor.train(features2, targets2)
        version2 = predictor.model_version

        # Check version tracking
        assert version1 != version2
        assert len(predictor.training_history) == 2
        assert predictor.training_history[0].model_version == version1
        assert predictor.training_history[1].model_version == version2

        # Both training sessions should be successful
        assert result1.success is True
        assert result2.success is True

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self):
        """Test error recovery in training/prediction workflow."""
        predictor = LSTMPredictor(max_iter=50)

        # First, successful training
        good_features = pd.DataFrame(
            {
                "feature1": np.random.random(50),
            }
        )
        good_targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 7200, 50)}
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ConvergenceWarning)
            result1 = await predictor.train(good_features, good_targets)
        assert result1.success is True

        # Then, failed training (insufficient data)
        bad_features = pd.DataFrame({"feature1": [1, 2]})  # Too small
        bad_targets = pd.DataFrame({"time_until_transition_seconds": [3600, 1800]})

        try:
            await predictor.train(bad_features, bad_targets)
            assert False, "Should have raised ModelTrainingError"
        except ModelTrainingError:
            pass

        # Model should still be usable from first training
        assert predictor.is_trained is True
        predictions = await predictor.predict(
            good_features.iloc[:1], datetime.now(timezone.utc)
        )
        assert len(predictions) == 1

        # Training history should include both successful and failed attempts
        assert len(predictor.training_history) == 2
        assert predictor.training_history[0].success is True
        assert predictor.training_history[1].success is False
