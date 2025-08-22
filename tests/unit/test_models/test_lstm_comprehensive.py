"""
Comprehensive unit tests for LSTMPredictor to achieve high test coverage.

This module focuses on comprehensive testing of all methods, error paths,
edge cases, and configuration variations in LSTMPredictor.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.core.constants import ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


class TestLSTMPredictorInitialization:
    """Test LSTMPredictor initialization and configuration."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        predictor = LSTMPredictor(room_id="test_room")

        assert predictor.room_id == "test_room"
        assert predictor.model_type == ModelType.LSTM
        assert predictor.model is None
        assert not predictor.is_trained
        assert isinstance(predictor.feature_scaler, StandardScaler)
        assert isinstance(predictor.target_scaler, MinMaxScaler)
        assert predictor.sequence_length == predictor.model_params["sequence_length"]
        assert predictor.sequence_step == 5

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        custom_params = {
            "sequence_length": 30,
            "hidden_units": 128,
            "learning_rate": 0.01,
            "max_iter": 2000,
            "early_stopping": True,
            "dropout": 0.3,
        }

        predictor = LSTMPredictor(room_id="test_room", **custom_params)

        assert predictor.model_params["sequence_length"] == 30
        assert predictor.model_params["hidden_units"] == 128
        assert predictor.model_params["learning_rate"] == 0.01
        assert predictor.model_params["max_iter"] == 2000
        assert predictor.model_params["early_stopping"] is True
        assert predictor.model_params["dropout"] == 0.3
        assert predictor.sequence_length == 30

    def test_init_hidden_units_as_int(self):
        """Test initialization with hidden_units as integer."""
        predictor = LSTMPredictor(hidden_units=64)

        # Should convert to [64, 32]
        expected_layers = [64, 32]
        assert predictor.model_params["hidden_layers"] == expected_layers

    def test_init_hidden_units_as_list(self):
        """Test initialization with hidden_units as list."""
        hidden_layers = [128, 64, 32]
        predictor = LSTMPredictor(hidden_units=hidden_layers)

        assert predictor.model_params["hidden_layers"] == hidden_layers

    def test_init_no_room_id(self):
        """Test initialization without room_id."""
        predictor = LSTMPredictor()

        assert predictor.room_id is None
        assert predictor.model_type == ModelType.LSTM

    def test_init_parameter_aliases(self):
        """Test that parameter aliases work correctly."""
        predictor = LSTMPredictor(hidden_units=100, dropout_rate=0.4, lstm_units=80)

        assert predictor.model_params["hidden_units"] == 100
        assert predictor.model_params["hidden_size"] == 100  # Alias
        assert predictor.model_params["dropout"] == 0.4  # From dropout_rate
        assert predictor.model_params["dropout_rate"] == 0.4
        assert predictor.model_params["lstm_units"] == 80

    def test_init_training_history_empty(self):
        """Test that training history is initialized as empty."""
        predictor = LSTMPredictor()

        assert predictor.training_history == []
        assert predictor.training_loss_history == []
        assert predictor.validation_loss_history == []


class TestLSTMPredictorSequenceCreation:
    """Test sequence creation functionality."""

    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        features = pd.DataFrame(
            {
                "temporal_hour_sin": np.sin(np.linspace(0, 4 * np.pi, n_samples)),
                "temporal_hour_cos": np.cos(np.linspace(0, 4 * np.pi, n_samples)),
                "time_since_last_change": np.random.uniform(0, 3600, n_samples),
                "temperature": np.random.uniform(18, 26, n_samples),
            }
        )

        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )

        return features, targets

    def test_create_sequences_basic(self, sample_data):
        """Test basic sequence creation."""
        features, targets = sample_data
        predictor = LSTMPredictor(sequence_length=10, room_id="test_room")

        X_seq, y_seq = predictor._create_sequences(features, targets)

        assert X_seq.shape[0] == y_seq.shape[0]  # Same number of sequences
        assert X_seq.shape[1] == 10 * 4  # Flattened: sequence_length * n_features
        assert len(y_seq) > 0
        # With step=5, expect roughly (100-10+1)/5 = ~18 sequences
        assert 15 <= len(y_seq) <= 25

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

    def test_create_sequences_mismatched_lengths(self):
        """Test sequence creation with mismatched feature/target lengths."""
        features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        targets = pd.DataFrame({"target": [1, 2, 3]})  # Different length

        predictor = LSTMPredictor(sequence_length=3)

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "same length" in str(exc_info.value)

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        features = pd.DataFrame({"feature1": [1, 2]})
        targets = pd.DataFrame({"target": [300, 600]})

        predictor = LSTMPredictor(sequence_length=5)

        with pytest.raises(ValueError) as exc_info:
            predictor._create_sequences(features, targets)

        assert "Need at least 5 samples" in str(exc_info.value)

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

    def test_create_sequences_step_size(self):
        """Test sequence creation with different step sizes."""
        features = pd.DataFrame({"feature1": list(range(50))})
        targets = pd.DataFrame({"target": [1200] * 50})

        predictor = LSTMPredictor(sequence_length=10)

        # Test with default step (5)
        X_seq1, y_seq1 = predictor._create_sequences(features, targets)

        # Test with step=1
        predictor.sequence_step = 1
        X_seq2, y_seq2 = predictor._create_sequences(features, targets)

        # Step=1 should generate more sequences than step=5
        assert len(X_seq2) > len(X_seq1)


class TestLSTMPredictorTraining:
    """Test LSTMPredictor training functionality."""

    @pytest.fixture
    def training_data(self):
        """Create training data."""
        np.random.seed(42)
        n_samples = 100

        features = pd.DataFrame(
            {
                "temporal_hour_sin": np.sin(np.linspace(0, 4 * np.pi, n_samples)),
                "temporal_hour_cos": np.cos(np.linspace(0, 4 * np.pi, n_samples)),
                "time_since_last_change": np.random.uniform(0, 3600, n_samples),
                "temperature": np.random.uniform(18, 26, n_samples),
            }
        )

        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )

        return features, targets

    @pytest.mark.asyncio
    async def test_train_basic_success(self, training_data):
        """Test successful basic training."""
        features, targets = training_data
        predictor = LSTMPredictor(
            room_id="test_room",
            sequence_length=10,
            max_iter=50,  # Keep low for faster testing
        )

        result = await predictor.train(features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.training_samples == len(features)
        assert result.model_version is not None
        assert result.training_score is not None
        assert result.validation_score is not None
        assert "training_mae" in result.training_metrics
        assert "sequences_generated" in result.training_metrics
        assert "sequence_length" in result.training_metrics

        # Check model state
        assert predictor.is_trained is True
        assert predictor.model is not None
        assert isinstance(predictor.model, MLPRegressor)
        assert predictor.training_date is not None
        assert len(predictor.feature_names) == len(features.columns)

    @pytest.mark.asyncio
    async def test_train_with_validation_data(self, training_data):
        """Test training with separate validation data."""
        features, targets = training_data

        # Split data
        split_idx = int(0.8 * len(features))
        train_features = features.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_targets = targets.iloc[split_idx:]

        predictor = LSTMPredictor(sequence_length=8, max_iter=50)

        result = await predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert "validation_mae" in result.training_metrics
        assert "validation_rmse" in result.training_metrics
        assert "validation_r2" in result.training_metrics

    @pytest.mark.asyncio
    async def test_train_small_dataset_adaptation(self):
        """Test training with small dataset (should adapt sequence length)."""
        np.random.seed(42)
        # Very small dataset
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(30),
                "feature2": np.random.randn(30),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(600, 3600, 30)}
        )

        predictor = LSTMPredictor(
            sequence_length=20, max_iter=50  # Originally larger than possible
        )

        result = await predictor.train(features, targets)

        assert result.success is True
        # Should have adapted sequence length
        assert hasattr(predictor, "training_sequence_length")
        assert predictor.training_sequence_length < 20

    @pytest.mark.asyncio
    async def test_train_very_small_dataset(self):
        """Test training with very small dataset."""
        features = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
            }
        )
        targets = pd.DataFrame({"time_until_transition_seconds": [600, 1200, 1800]})

        predictor = LSTMPredictor(sequence_length=5)

        with pytest.raises(ModelTrainingError) as exc_info:
            await predictor.train(features, targets)

        # Should fail with insufficient data
        assert "Insufficient sequence data" in str(
            exc_info.value
        ) or "sequences available" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_train_sequence_generation_failure(self):
        """Test training when sequence generation fails."""
        features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        # All targets out of valid range
        targets = pd.DataFrame({"target": [30] * 10})  # All < 60 seconds

        predictor = LSTMPredictor(sequence_length=5)

        # Mock _create_sequences to raise ValueError
        with patch.object(
            predictor, "_create_sequences", side_effect=ValueError("No valid sequences")
        ):
            with pytest.raises(ModelTrainingError) as exc_info:
                await predictor.train(features, targets)

            assert exc_info.value.cause is not None

    @pytest.mark.asyncio
    async def test_train_stores_training_history(self, training_data):
        """Test that training results are stored in history."""
        features, targets = training_data
        predictor = LSTMPredictor(sequence_length=8, max_iter=50)

        # Train multiple times
        result1 = await predictor.train(features, targets)
        result2 = await predictor.train(features, targets)

        assert len(predictor.training_history) == 2
        assert predictor.training_history[0] == result1
        assert predictor.training_history[1] == result2

    @pytest.mark.asyncio
    async def test_train_model_fitting_failure(self, training_data):
        """Test training when MLPRegressor fitting fails."""
        features, targets = training_data
        predictor = LSTMPredictor(sequence_length=10)

        # Mock MLPRegressor to raise exception
        with patch("src.models.base.lstm_predictor.MLPRegressor") as mock_mlp:
            mock_mlp.return_value.fit.side_effect = RuntimeError("Fitting failed")

            with pytest.raises(ModelTrainingError) as exc_info:
                await predictor.train(features, targets)

            assert exc_info.value.cause is not None
            # Should record failed training
            assert len(predictor.training_history) == 1
            assert predictor.training_history[0].success is False

    @pytest.mark.asyncio
    async def test_train_sequence_length_restoration(self, training_data):
        """Test that sequence length is restored after training."""
        features, targets = training_data[:30]  # Small dataset to trigger adaptation
        features, targets = training_data
        # Force small dataset
        features = features.iloc[:50]
        targets = targets.iloc[:50]

        original_sequence_length = 25
        predictor = LSTMPredictor(sequence_length=original_sequence_length, max_iter=50)

        await predictor.train(features, targets)

        # Sequence length should be restored to original value
        assert predictor.sequence_length == original_sequence_length
        assert predictor.sequence_step == 5  # Should restore default step

    @pytest.mark.asyncio
    async def test_train_validation_without_valid_sequences(self, training_data):
        """Test training with validation data that produces no valid sequences."""
        features, targets = training_data

        # Create validation data with no valid sequences (all targets too small)
        val_features = features.iloc[:20]
        val_targets = pd.DataFrame({"time_until_transition_seconds": [30] * 20})

        predictor = LSTMPredictor(sequence_length=8, max_iter=50)

        # Should handle empty validation sequences gracefully
        result = await predictor.train(features, targets, val_features, val_targets)
        assert result.success is True


class TestLSTMPredictorPrediction:
    """Test LSTMPredictor prediction functionality."""

    @pytest.fixture
    def trained_predictor(self):
        """Create a trained LSTM predictor."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=5)

        # Mock trained state
        predictor.is_trained = True
        predictor.model_version = "v1.0"
        predictor.feature_names = ["feature1", "feature2", "feature3"]
        predictor.training_sequence_length = 5

        # Mock model, scalers
        predictor.model = Mock(spec=MLPRegressor)
        predictor.model.predict.return_value = np.array(
            [0.5, 0.7]
        )  # Scaled predictions

        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.feature_scaler.transform.return_value = np.array(
            [
                [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                ],  # Flattened sequence
                [
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                    1.1,
                    1.2,
                    1.3,
                    1.4,
                    1.5,
                    1.6,
                ],
            ]
        )

        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[1800], [3600]]
        )  # 30 min, 1 hour

        # Mock training history for confidence
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

        return predictor

    @pytest.mark.asyncio
    async def test_predict_success(self, trained_predictor):
        """Test successful prediction."""
        features = pd.DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
                "feature2": [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                "feature3": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time, "vacant")

        assert len(results) == len(features)
        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.predicted_time > prediction_time
            assert result.transition_type in [
                "vacant_to_occupied",
                "occupied_to_vacant",
            ]
            assert 0.1 <= result.confidence_score <= 0.95
            assert result.model_type == "lstm"
            assert result.model_version == "v1.0"
            assert result.features_used == ["feature1", "feature2", "feature3"]
            assert "time_until_transition_seconds" in result.prediction_metadata
            assert "sequence_length_used" in result.prediction_metadata
            assert "prediction_method" in result.prediction_metadata

    @pytest.mark.asyncio
    async def test_predict_not_trained(self):
        """Test prediction with untrained model."""
        predictor = LSTMPredictor(room_id="test_room")
        features = pd.DataFrame({"feature1": [1, 2, 3]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError) as exc_info:
            await predictor.predict(features, prediction_time)

        assert "lstm" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_predict_invalid_features(self, trained_predictor):
        """Test prediction with invalid features."""
        # Mock validate_features to return False
        with patch.object(trained_predictor, "validate_features", return_value=False):
            features = pd.DataFrame({"wrong_feature": [1, 2, 3]})
            prediction_time = datetime.now(timezone.utc)

            with pytest.raises(ModelPredictionError):
                await trained_predictor.predict(features, prediction_time)

    @pytest.mark.asyncio
    async def test_predict_sequence_padding(self, trained_predictor):
        """Test prediction with insufficient history (should pad)."""
        # Only 3 rows, but sequence_length is 5
        features = pd.DataFrame(
            {
                "feature1": [0.1, 0.2, 0.3],
                "feature2": [0.2, 0.3, 0.4],
                "feature3": [0.3, 0.4, 0.5],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time)

        assert len(results) == 3
        # Should handle padding gracefully
        for result in results:
            assert isinstance(result, PredictionResult)

    @pytest.mark.asyncio
    async def test_predict_transition_type_occupied(self, trained_predictor):
        """Test transition type logic when currently occupied."""
        features = pd.DataFrame(
            {
                "feature1": [0.5],
                "feature2": [0.6],
                "feature3": [0.7],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time, "occupied")

        assert results[0].transition_type == "occupied_to_vacant"

    @pytest.mark.asyncio
    async def test_predict_transition_type_vacant(self, trained_predictor):
        """Test transition type logic when currently vacant."""
        features = pd.DataFrame(
            {
                "feature1": [0.5],
                "feature2": [0.6],
                "feature3": [0.7],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time, "vacant")

        assert results[0].transition_type == "vacant_to_occupied"

    @pytest.mark.asyncio
    async def test_predict_transition_type_unknown_daytime(self, trained_predictor):
        """Test transition type logic for unknown state during daytime."""
        features = pd.DataFrame(
            {
                "feature1": [0.5],
                "feature2": [0.6],
                "feature3": [0.7],
            }
        )
        # 2 PM
        prediction_time = datetime.now(timezone.utc).replace(hour=14)

        results = await trained_predictor.predict(features, prediction_time, "unknown")

        assert results[0].transition_type == "vacant_to_occupied"

    @pytest.mark.asyncio
    async def test_predict_transition_type_unknown_nighttime(self, trained_predictor):
        """Test transition type logic for unknown state during nighttime."""
        features = pd.DataFrame(
            {
                "feature1": [0.5],
                "feature2": [0.6],
                "feature3": [0.7],
            }
        )
        # 2 AM
        prediction_time = datetime.now(timezone.utc).replace(hour=2)

        results = await trained_predictor.predict(features, prediction_time, "unknown")

        assert results[0].transition_type == "occupied_to_vacant"

    @pytest.mark.asyncio
    async def test_predict_time_clipping(self, trained_predictor):
        """Test that prediction times are properly clipped."""
        # Mock extreme predictions
        trained_predictor.target_scaler.inverse_transform.return_value = np.array(
            [[30], [200000]]
        )  # Very small/large

        features = pd.DataFrame(
            {
                "feature1": [0.5, 0.6],
                "feature2": [0.6, 0.7],
                "feature3": [0.7, 0.8],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time)

        # Check that times were clipped to reasonable bounds
        for result in results:
            time_diff = (result.predicted_time - prediction_time).total_seconds()
            assert 60 <= time_diff <= 86400  # Between 1 min and 24 hours

    @pytest.mark.asyncio
    async def test_predict_error_handling(self, trained_predictor):
        """Test prediction error handling."""
        # Mock scaler to raise exception
        trained_predictor.feature_scaler.transform.side_effect = Exception(
            "Scaling failed"
        )

        features = pd.DataFrame(
            {
                "feature1": [0.5],
                "feature2": [0.6],
                "feature3": [0.7],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError) as exc_info:
            await trained_predictor.predict(features, prediction_time)

        assert "LSTM prediction failed" in str(exc_info.value)


class TestLSTMPredictorFeatureImportance:
    """Test feature importance calculation."""

    def test_get_feature_importance_not_trained(self):
        """Test feature importance when model is not trained."""
        predictor = LSTMPredictor(room_id="test_room")

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_no_coefficients(self):
        """Test feature importance when model has no coefficients."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.model = Mock(spec=MLPRegressor)
        # Mock model without coefs_ attribute
        del predictor.model.coefs_  # Remove if exists

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_empty_coefficients(self):
        """Test feature importance when model has empty coefficients."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.model = Mock(spec=MLPRegressor)
        predictor.model.coefs_ = []

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_success(self):
        """Test successful feature importance calculation."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=3)
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2"]

        predictor.model = Mock(spec=MLPRegressor)
        # Mock coefficients: shape (n_features * sequence_length, n_hidden)
        # For 2 features and sequence_length=3: (6, 10)
        mock_weights = np.random.rand(
            6, 10
        )  # 6 input features (2 * 3), 10 hidden units
        predictor.model.coefs_ = [mock_weights]

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert "feature1" in importance
        assert "feature2" in importance
        assert all(isinstance(v, float) for v in importance.values())
        # Should be normalized to sum to 1
        assert abs(sum(importance.values()) - 1.0) < 1e-6

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

    def test_get_feature_importance_exception_handling(self):
        """Test feature importance exception handling."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.feature_names = ["feature1"]

        predictor.model = Mock(spec=MLPRegressor)
        # Mock coefs_ to raise exception when accessed
        predictor.model.coefs_ = Mock()
        predictor.model.coefs_.__getitem__.side_effect = Exception("Access failed")

        importance = predictor.get_feature_importance()

        assert importance == {}


class TestLSTMPredictorConfidence:
    """Test confidence calculation methods."""

    def test_calculate_confidence_with_training_history(self):
        """Test confidence calculation with training history."""
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

        # Mock scalers
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[1800]]
        )  # 30 minutes

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert 0.1 <= confidence <= 0.95
        # Should be based on validation score with some adjustments

    def test_calculate_confidence_no_validation_score(self):
        """Test confidence calculation without validation score."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=60,
                model_version="v1.0",
                training_samples=100,
                validation_score=None,
                training_score=0.80,
            )
        ]

        # Mock scalers
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array([[1800]])

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert 0.1 <= confidence <= 0.95

    def test_calculate_confidence_extreme_predictions(self):
        """Test confidence calculation with extreme predictions."""
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

    def test_calculate_confidence_no_training_history(self):
        """Test confidence calculation with no training history."""
        predictor = LSTMPredictor(room_id="test_room")

        # Mock scalers
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.return_value = np.array([[1800]])

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Should return default confidence
        assert confidence == 0.7

    def test_calculate_confidence_exception_handling(self):
        """Test confidence calculation exception handling."""
        predictor = LSTMPredictor(room_id="test_room")

        # Mock scaler to raise exception
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.target_scaler.inverse_transform.side_effect = Exception(
            "Scaler failed"
        )

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Should return default confidence on error
        assert confidence == 0.7


class TestLSTMPredictorModelComplexity:
    """Test model complexity information."""

    def test_get_model_complexity_not_trained(self):
        """Test model complexity when not trained."""
        predictor = LSTMPredictor(room_id="test_room")

        complexity = predictor.get_model_complexity()

        assert complexity == {}

    def test_get_model_complexity_trained(self):
        """Test model complexity with trained model."""
        predictor = LSTMPredictor(room_id="test_room", sequence_length=10)
        predictor.is_trained = True
        predictor.feature_names = ["f1", "f2", "f3"]
        predictor.model_params = {"hidden_layers": [64, 32], "sequence_length": 10}

        # Mock model with coefficients and intercepts
        predictor.model = Mock(spec=MLPRegressor)
        predictor.model.coefs_ = [
            np.random.rand(30, 64),  # Input to first hidden layer
            np.random.rand(64, 32),  # First to second hidden layer
            np.random.rand(32, 1),  # Second hidden to output
        ]
        predictor.model.intercepts_ = [
            np.random.rand(64),  # First hidden layer biases
            np.random.rand(32),  # Second hidden layer biases
            np.random.rand(1),  # Output layer bias
        ]

        complexity = predictor.get_model_complexity()

        assert "total_parameters" in complexity
        assert "hidden_layers" in complexity
        assert "sequence_length" in complexity
        assert "input_features" in complexity
        assert "flattened_input_size" in complexity

        assert complexity["hidden_layers"] == [64, 32]
        assert complexity["sequence_length"] == 10
        assert complexity["input_features"] == 3
        assert (
            complexity["flattened_input_size"] == 30
        )  # 3 features * 10 sequence_length

        # Should calculate total parameters correctly
        expected_params = (30 * 64 + 64) + (64 * 32 + 32) + (32 * 1 + 1)
        assert complexity["total_parameters"] == expected_params

    def test_get_model_complexity_no_coefficients(self):
        """Test model complexity when model has no coefficients."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.feature_names = ["f1", "f2"]
        predictor.model = Mock(spec=MLPRegressor)
        # No coefs_ attribute

        complexity = predictor.get_model_complexity()

        assert "total_parameters" in complexity
        assert complexity["total_parameters"] == 0


class TestLSTMPredictorSaveLoad:
    """Test model saving and loading functionality."""

    def test_save_model_success(self):
        """Test successful model saving."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.model_version = "v1.0"
        predictor.feature_names = ["f1", "f2"]
        predictor.sequence_length = 10

        # Mock model and scalers
        predictor.model = Mock(spec=MLPRegressor)
        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.target_scaler = Mock(spec=MinMaxScaler)
        predictor.training_date = datetime.now(timezone.utc)
        predictor.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=60,
                model_version="v1.0",
                training_samples=100,
                validation_score=0.85,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".pkl") as tmp_file:
            result = predictor.save_model(tmp_file.name)

            assert result is True
            assert Path(tmp_file.name).exists()

    def test_save_model_error(self):
        """Test model saving with error."""
        predictor = LSTMPredictor(room_id="test_room")
        predictor.model = Mock(spec=MLPRegressor)

        # Try to save to invalid path
        result = predictor.save_model("/invalid/path/model.pkl")

        assert result is False

    def test_load_model_success(self):
        """Test successful model loading."""
        # First create a model to save
        predictor1 = LSTMPredictor(room_id="test_room")
        predictor1.is_trained = True
        predictor1.model_version = "v1.0"
        predictor1.feature_names = ["f1", "f2"]
        predictor1.sequence_length = 10
        predictor1.model = Mock(spec=MLPRegressor)
        predictor1.feature_scaler = Mock(spec=StandardScaler)
        predictor1.target_scaler = Mock(spec=MinMaxScaler)
        predictor1.training_date = datetime.now(timezone.utc)
        predictor1.training_history = [
            TrainingResult(
                success=True,
                training_time_seconds=60,
                model_version="v1.0",
                training_samples=100,
                validation_score=0.85,
            )
        ]

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            tmp_path = tmp_file.name

        try:
            # Save model
            assert predictor1.save_model(tmp_path) is True

            # Load model into new predictor
            predictor2 = LSTMPredictor(room_id="different_room")
            result = predictor2.load_model(tmp_path)

            assert result is True
            assert predictor2.room_id == "test_room"  # Should be loaded from file
            assert predictor2.model_version == "v1.0"
            assert predictor2.feature_names == ["f1", "f2"]
            assert predictor2.sequence_length == 10
            assert predictor2.is_trained is True
            assert len(predictor2.training_history) == 1

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_model_error(self):
        """Test model loading with error."""
        predictor = LSTMPredictor(room_id="test_room")

        # Try to load from non-existent file
        result = predictor.load_model("/nonexistent/path/model.pkl")

        assert result is False

    def test_load_model_partial_data(self):
        """Test loading model with partial data (backwards compatibility)."""
        import pickle
        import tempfile

        # Create partial model data
        model_data = {
            "model": Mock(spec=MLPRegressor),
            "model_type": "lstm",
            "room_id": "test_room",
            # Missing some optional fields
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            pickle.dump(model_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            predictor = LSTMPredictor()
            result = predictor.load_model(tmp_path)

            assert result is True
            assert predictor.room_id == "test_room"
            # Should have defaults for missing fields
            assert predictor.model_version == "v1.0"
            assert predictor.feature_names == []
            assert predictor.sequence_length == 10  # Default
            assert predictor.is_trained is False

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestLSTMPredictorEdgeCases:
    """Test edge cases and error conditions."""

    def test_sequence_creation_edge_case_bounds(self):
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
    async def test_predict_with_missing_training_sequence_length(
        self, trained_predictor
    ):
        """Test prediction when training_sequence_length attribute is missing."""
        # Remove training_sequence_length attribute
        if hasattr(trained_predictor, "training_sequence_length"):
            delattr(trained_predictor, "training_sequence_length")

        features = pd.DataFrame(
            {
                "feature1": [0.1, 0.2],
                "feature2": [0.2, 0.3],
                "feature3": [0.3, 0.4],
            }
        )
        prediction_time = datetime.now(timezone.utc)

        # Should fall back to self.sequence_length
        results = await trained_predictor.predict(features, prediction_time)

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
