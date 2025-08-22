"""
Unit tests for LSTM-based occupancy prediction model.

Tests LSTMPredictor class including training, prediction, feature importance,
sequence generation, and model persistence functionality.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.neural_network import MLPRegressor

from src.core.constants import ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


class TestLSTMPredictorInitialization:
    """Test LSTM predictor initialization and configuration."""

    def test_basic_initialization(self):
        """Test basic LSTM predictor initialization."""
        predictor = LSTMPredictor(room_id="test_room")

        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id == "test_room"
        assert predictor.model is None
        assert not predictor.is_trained
        assert isinstance(predictor.model_params, dict)
        assert predictor.sequence_length > 0

    def test_initialization_with_custom_parameters(self):
        """Test initialization with custom parameters."""
        custom_params = {
            "sequence_length": 30,
            "hidden_units": 128,
            "learning_rate": 0.01,
            "max_iter": 500,
            "dropout": 0.3,
        }

        predictor = LSTMPredictor(room_id="custom_room", **custom_params)

        assert predictor.model_params["sequence_length"] == 30
        assert predictor.model_params["learning_rate"] == 0.01
        assert predictor.model_params["max_iter"] == 500
        assert predictor.model_params["dropout"] == 0.3

    def test_hidden_layers_configuration(self):
        """Test hidden layers configuration from different input formats."""
        # Test with integer hidden_units
        predictor1 = LSTMPredictor(hidden_units=64)
        assert predictor1.model_params["hidden_layers"] == [64, 32]

        # Test with list hidden_units
        predictor2 = LSTMPredictor(hidden_units=[128, 64, 32])
        assert predictor2.model_params["hidden_layers"] == [128, 64, 32]

    def test_dropout_parameter_aliasing(self):
        """Test dropout/dropout_rate parameter aliasing."""
        # Test dropout_rate takes precedence
        predictor1 = LSTMPredictor(dropout=0.2, dropout_rate=0.4)
        assert predictor1.model_params["dropout"] == 0.4
        assert predictor1.model_params["dropout_rate"] == 0.4

        # Test dropout fallback
        predictor2 = LSTMPredictor(dropout=0.3)
        assert predictor2.model_params["dropout"] == 0.3
        assert predictor2.model_params["dropout_rate"] == 0.3

    def test_default_parameters(self):
        """Test default parameter values."""
        predictor = LSTMPredictor()

        assert predictor.model_params["sequence_length"] == 50
        assert predictor.model_params["max_iter"] == 1000
        assert predictor.model_params["early_stopping"] is False
        assert predictor.sequence_step == 5


class TestSequenceGeneration:
    """Test sequence data generation for LSTM training."""

    def test_create_sequences_basic(self):
        """Test basic sequence creation."""
        predictor = LSTMPredictor()
        predictor.sequence_length = 5
        predictor.sequence_step = 1

        # Create test data
        features = pd.DataFrame(
            {
                "feature1": range(10),
                "feature2": range(10, 20),
            }
        )
        targets = pd.DataFrame({"time_until_transition_seconds": [3600] * 10})  # 1 hour

        X_sequences, y_sequences = predictor._create_sequences(features, targets)

        # Should create sequences from index 5 to 10
        expected_sequences = 6  # (10 - 5 + 1) with step 1 = 6 sequences
        assert len(X_sequences) == expected_sequences
        assert len(y_sequences) == expected_sequences

        # Each sequence should be flattened: sequence_length * n_features
        expected_feature_length = 5 * 2  # 5 timesteps * 2 features
        assert X_sequences[0].shape == (expected_feature_length,)

    def test_create_sequences_with_step(self):
        """Test sequence creation with step size."""
        predictor = LSTMPredictor()
        predictor.sequence_length = 3
        predictor.sequence_step = 2

        features = pd.DataFrame(
            {
                "feature1": range(10),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [1800] * 10}  # 30 minutes
        )

        X_sequences, y_sequences = predictor._create_sequences(features, targets)

        # With step=2: sequences at indices [3, 5, 7, 9] = 4 sequences
        assert len(X_sequences) == 4
        assert all(y == 1800 for y in y_sequences)

    def test_create_sequences_insufficient_data(self):
        """Test sequence creation with insufficient data."""
        predictor = LSTMPredictor()
        predictor.sequence_length = 10

        features = pd.DataFrame({"feature1": range(5)})  # Only 5 samples
        targets = pd.DataFrame({"time_until_transition_seconds": [3600] * 5})

        with pytest.raises(ValueError, match="Need at least 10 samples"):
            predictor._create_sequences(features, targets)

    def test_create_sequences_mismatched_lengths(self):
        """Test sequence creation with mismatched feature/target lengths."""
        predictor = LSTMPredictor()

        features = pd.DataFrame({"feature1": range(10)})
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [3600] * 5}
        )  # Different length

        with pytest.raises(ValueError, match="same length"):
            predictor._create_sequences(features, targets)

    def test_create_sequences_target_formats(self):
        """Test different target data formats."""
        predictor = LSTMPredictor()
        predictor.sequence_length = 3

        features = pd.DataFrame({"feature1": range(5)})

        # Test with time_until_transition_seconds column
        targets1 = pd.DataFrame({"time_until_transition_seconds": [1800] * 5})
        X1, y1 = predictor._create_sequences(features, targets1)
        assert all(y == 1800 for y in y1)

        # Test with next_transition_time and target_time columns
        base_time = datetime.now()
        targets2 = pd.DataFrame(
            {
                "target_time": [base_time + timedelta(hours=i) for i in range(5)],
                "next_transition_time": [
                    base_time + timedelta(hours=i, minutes=30) for i in range(5)
                ],
            }
        )
        X2, y2 = predictor._create_sequences(features, targets2)
        assert all(y == 1800 for y in y2)  # 30 minutes = 1800 seconds

        # Test with default format (first column)
        targets3 = pd.DataFrame({"target_values": [900] * 5})  # 15 minutes
        X3, y3 = predictor._create_sequences(features, targets3)
        assert all(y == 900 for y in y3)

    def test_create_sequences_invalid_targets(self):
        """Test sequence creation with invalid target values."""
        predictor = LSTMPredictor()
        predictor.sequence_length = 3

        features = pd.DataFrame({"feature1": range(5)})

        # Test with non-numeric targets
        targets = pd.DataFrame({"time_until_transition_seconds": ["invalid"] * 5})

        with pytest.raises(ValueError, match="non-numeric data"):
            predictor._create_sequences(features, targets)

    def test_create_sequences_target_filtering(self):
        """Test that unreasonable target values are filtered out."""
        predictor = LSTMPredictor()
        predictor.sequence_length = 3

        features = pd.DataFrame({"feature1": range(8)})
        # Mix of reasonable and unreasonable values
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [
                    30,
                    3600,
                    100000,
                    1800,
                    45,
                    7200,
                    200000,
                    900,
                ]
            }
        )

        X_sequences, y_sequences = predictor._create_sequences(features, targets)

        # Only values between 60 and 86400 seconds should be kept
        valid_targets = [t for t in y_sequences if 60 <= t <= 86400]
        assert len(valid_targets) == len(
            y_sequences
        )  # All kept sequences should be valid
        assert all(60 <= t <= 86400 for t in y_sequences)


class TestLSTMTraining:
    """Test LSTM model training functionality."""

    def create_training_data(self, n_samples=100):
        """Create synthetic training data."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, n_samples),
                "day_of_week": np.random.randint(0, 7, n_samples),
                "time_since_last_event": np.random.uniform(300, 7200, n_samples),
                "occupancy_state": np.random.choice([0, 1], n_samples),
            }
        )

        # Generate realistic transition times (30 min to 4 hours)
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 14400, n_samples)}
        )

        return features, targets

    async def test_successful_training(self):
        """Test successful LSTM model training."""
        predictor = LSTMPredictor(room_id="test_room", max_iter=100)
        features, targets = self.create_training_data(100)

        result = await predictor.train(features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.training_time_seconds > 0
        assert result.training_samples == 100
        assert result.model_version is not None
        assert predictor.is_trained is True
        assert predictor.model is not None
        assert isinstance(predictor.model, MLPRegressor)

    async def test_training_with_validation_data(self):
        """Test training with validation data."""
        predictor = LSTMPredictor(room_id="test_room", max_iter=50)
        train_features, train_targets = self.create_training_data(80)
        val_features, val_targets = self.create_training_data(20)

        result = await predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert result.validation_score is not None
        assert "validation_mae" in result.training_metrics
        assert "validation_rmse" in result.training_metrics
        assert "validation_r2" in result.training_metrics

    async def test_training_small_dataset_adaptation(self):
        """Test training with small dataset and sequence length adaptation."""
        predictor = LSTMPredictor(sequence_length=20)  # Large sequence length
        features, targets = self.create_training_data(30)  # Small dataset

        result = await predictor.train(features, targets)

        # Should adapt sequence length for small dataset
        assert result.success is True
        assert "sequence_length" in result.training_metrics
        # Adapted sequence length should be smaller
        adapted_length = result.training_metrics["sequence_length"]
        assert adapted_length < 20

    async def test_training_insufficient_sequences(self):
        """Test training failure with insufficient sequence data."""
        predictor = LSTMPredictor(sequence_length=50)
        features, targets = self.create_training_data(10)  # Too small

        with pytest.raises(ModelTrainingError):
            await predictor.train(features, targets)

    async def test_training_metrics_recording(self):
        """Test that training metrics are properly recorded."""
        predictor = LSTMPredictor(max_iter=50)
        features, targets = self.create_training_data(80)

        result = await predictor.train(features, targets)

        assert "training_mae" in result.training_metrics
        assert "training_rmse" in result.training_metrics
        assert "training_r2" in result.training_metrics
        assert "sequences_generated" in result.training_metrics
        assert "sequence_length" in result.training_metrics

        # Check that metrics are reasonable
        assert result.training_metrics["training_mae"] >= 0
        assert result.training_metrics["training_rmse"] >= 0
        assert result.training_metrics["sequences_generated"] > 0

    async def test_training_history_tracking(self):
        """Test that training history is tracked."""
        predictor = LSTMPredictor()
        features, targets = self.create_training_data()

        # Train multiple times
        await predictor.train(features, targets)
        first_result = predictor.training_history[-1]

        await predictor.train(features, targets)
        second_result = predictor.training_history[-1]

        assert len(predictor.training_history) == 2
        assert first_result.model_version != second_result.model_version

    async def test_training_error_handling(self):
        """Test training error handling and recording."""
        predictor = LSTMPredictor()

        # Create invalid data that will cause training to fail
        features = pd.DataFrame({"feature1": [np.inf, np.nan, 1, 2, 3]})
        targets = pd.DataFrame({"time_until_transition_seconds": [3600] * 5})

        with pytest.raises(ModelTrainingError):
            await predictor.train(features, targets)

        # Should record failed training
        assert len(predictor.training_history) == 1
        assert predictor.training_history[-1].success is False


class TestLSTMPrediction:
    """Test LSTM model prediction functionality."""

    async def setup_trained_predictor(self):
        """Set up a trained LSTM predictor for testing."""
        predictor = LSTMPredictor(room_id="test_room", max_iter=50)

        # Create and train on synthetic data
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, 100),
                "occupancy_state": np.random.choice([0, 1], 100),
                "time_since_event": np.random.uniform(300, 3600, 100),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 7200, 100)}
        )

        await predictor.train(features, targets)
        return predictor, features

    async def test_successful_prediction(self):
        """Test successful prediction generation."""
        predictor, training_features = await self.setup_trained_predictor()

        # Create prediction features (similar structure to training)
        pred_features = training_features.iloc[:5].copy()
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(pred_features, prediction_time, "vacant")

        assert len(predictions) == 5
        for pred in predictions:
            assert isinstance(pred, PredictionResult)
            assert pred.predicted_time > prediction_time
            assert pred.model_type == ModelType.LSTM.value
            assert pred.confidence_score >= 0.1
            assert pred.confidence_score <= 0.95
            assert pred.transition_type in ["vacant_to_occupied", "occupied_to_vacant"]

    async def test_prediction_with_different_states(self):
        """Test predictions with different current states."""
        predictor, training_features = await self.setup_trained_predictor()
        pred_features = training_features.iloc[:1].copy()
        prediction_time = datetime.now(timezone.utc)

        # Test occupied state
        pred_occupied = await predictor.predict(
            pred_features, prediction_time, "occupied"
        )
        assert pred_occupied[0].transition_type == "occupied_to_vacant"

        # Test vacant state
        pred_vacant = await predictor.predict(pred_features, prediction_time, "vacant")
        assert pred_vacant[0].transition_type == "vacant_to_occupied"

        # Test unknown state (should infer based on time)
        pred_unknown = await predictor.predict(
            pred_features, prediction_time, "unknown"
        )
        assert pred_unknown[0].transition_type in [
            "vacant_to_occupied",
            "occupied_to_vacant",
        ]

    async def test_prediction_untrained_model(self):
        """Test prediction with untrained model."""
        predictor = LSTMPredictor()
        features = pd.DataFrame({"feature1": [1, 2, 3]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(features, prediction_time)

    async def test_prediction_invalid_features(self):
        """Test prediction with invalid features."""
        predictor, _ = await self.setup_trained_predictor()

        # Features with different columns than training
        invalid_features = pd.DataFrame({"wrong_feature": [1, 2, 3]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(invalid_features, prediction_time)

    async def test_prediction_sequence_padding(self):
        """Test prediction with insufficient history for full sequence."""
        predictor, training_features = await self.setup_trained_predictor()

        # Use only first 2 rows (less than sequence_length)
        short_features = training_features.iloc[:2].copy()
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(short_features, prediction_time)

        # Should still make predictions by padding
        assert len(predictions) == 2
        assert all(isinstance(p, PredictionResult) for p in predictions)

    async def test_prediction_metadata(self):
        """Test prediction metadata content."""
        predictor, training_features = await self.setup_trained_predictor()
        pred_features = training_features.iloc[:1].copy()
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(pred_features, prediction_time)
        pred = predictions[0]

        assert "time_until_transition_seconds" in pred.prediction_metadata
        assert "sequence_length_used" in pred.prediction_metadata
        assert "prediction_method" in pred.prediction_metadata
        assert pred.prediction_metadata["prediction_method"] == "lstm_neural_network"

        # Time until transition should be reasonable
        time_until = pred.prediction_metadata["time_until_transition_seconds"]
        assert 60 <= time_until <= 86400  # Between 1 min and 24 hours

    async def test_prediction_time_bounds(self):
        """Test that predictions are bounded within reasonable time ranges."""
        predictor, training_features = await self.setup_trained_predictor()
        pred_features = training_features.iloc[:3].copy()
        prediction_time = datetime.now(timezone.utc)

        predictions = await predictor.predict(pred_features, prediction_time)

        for pred in predictions:
            time_diff = pred.predicted_time - prediction_time
            # Should be between 1 minute and 24 hours
            assert timedelta(minutes=1) <= time_diff <= timedelta(hours=24)


class TestConfidenceCalculation:
    """Test confidence score calculation."""

    async def test_confidence_with_training_history(self):
        """Test confidence calculation using training history."""
        predictor = LSTMPredictor()

        # Mock training history
        mock_result = Mock()
        mock_result.validation_score = 0.8
        mock_result.training_score = 0.75
        predictor.training_history = [mock_result]

        # Mock scalers and data
        predictor.target_scaler = Mock()
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[3600]]
        )  # 1 hour

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        assert 0.1 <= confidence <= 0.95
        # Should be based on validation score
        assert confidence > 0.7

    async def test_confidence_extreme_predictions(self):
        """Test confidence adjustment for extreme predictions."""
        predictor = LSTMPredictor()

        # Mock training history with good score
        mock_result = Mock()
        mock_result.validation_score = 0.9
        predictor.training_history = [mock_result]

        # Mock scaler for extreme prediction (very short time)
        predictor.target_scaler = Mock()
        predictor.target_scaler.inverse_transform.return_value = np.array(
            [[120]]
        )  # 2 minutes (extreme)

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.1])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Should be reduced due to extreme prediction
        assert confidence < 0.8

    async def test_confidence_no_history(self):
        """Test confidence calculation with no training history."""
        predictor = LSTMPredictor()
        predictor.training_history = []

        predictor.target_scaler = Mock()
        predictor.target_scaler.inverse_transform.return_value = np.array([[3600]])

        X_scaled = np.array([[0.1, 0.2, 0.3]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Should return default confidence
        assert confidence == 0.7

    async def test_confidence_error_handling(self):
        """Test confidence calculation error handling."""
        predictor = LSTMPredictor()

        # Create situation that will cause error
        predictor.target_scaler = Mock()
        predictor.target_scaler.inverse_transform.side_effect = Exception(
            "Scaler error"
        )

        X_scaled = np.array([[0.1]])
        y_pred_scaled = np.array([0.5])

        confidence = predictor._calculate_confidence(X_scaled, y_pred_scaled)

        # Should return default on error
        assert confidence == 0.7


class TestFeatureImportance:
    """Test feature importance calculation."""

    async def test_feature_importance_trained_model(self):
        """Test feature importance for trained model."""
        predictor = LSTMPredictor(max_iter=50)

        # Train model
        features = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, 50),
                "day_of_week": np.random.randint(0, 7, 50),
                "occupancy": np.random.choice([0, 1], 50),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 7200, 50)}
        )

        await predictor.train(features, targets)

        importance = predictor.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == 3  # Three features
        assert all(
            feature in importance for feature in ["hour", "day_of_week", "occupancy"]
        )
        assert all(isinstance(score, float) for score in importance.values())

        # Importance scores should be normalized (sum to ~1)
        total_importance = sum(importance.values())
        assert 0.99 <= total_importance <= 1.01

    def test_feature_importance_untrained_model(self):
        """Test feature importance for untrained model."""
        predictor = LSTMPredictor()

        importance = predictor.get_feature_importance()

        assert importance == {}

    async def test_feature_importance_error_handling(self):
        """Test feature importance calculation error handling."""
        predictor = LSTMPredictor()
        predictor.is_trained = True
        predictor.model = Mock()

        # Mock model without coefs_ attribute
        predictor.model.coefs_ = None

        importance = predictor.get_feature_importance()

        assert importance == {}


class TestModelComplexity:
    """Test model complexity analysis."""

    async def test_model_complexity_trained(self):
        """Test model complexity information for trained model."""
        predictor = LSTMPredictor(hidden_units=[32, 16], max_iter=50)

        # Train model
        features = pd.DataFrame(
            {
                "feature1": np.random.random(50),
                "feature2": np.random.random(50),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 7200, 50)}
        )

        await predictor.train(features, targets)

        complexity = predictor.get_model_complexity()

        assert "total_parameters" in complexity
        assert "hidden_layers" in complexity
        assert "sequence_length" in complexity
        assert "input_features" in complexity
        assert "flattened_input_size" in complexity

        assert complexity["hidden_layers"] == [32, 16]
        assert complexity["input_features"] == 2
        assert complexity["total_parameters"] > 0

    def test_model_complexity_untrained(self):
        """Test model complexity for untrained model."""
        predictor = LSTMPredictor()

        complexity = predictor.get_model_complexity()

        assert complexity == {}


class TestModelPersistence:
    """Test model saving and loading functionality."""

    async def test_save_and_load_model(self):
        """Test complete model save and load cycle."""
        # Train original model
        predictor1 = LSTMPredictor(room_id="test_room", max_iter=50)
        features = pd.DataFrame(
            {
                "hour": np.random.randint(0, 24, 50),
                "occupancy": np.random.choice([0, 1], 50),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(1800, 7200, 50)}
        )

        await predictor1.train(features, targets)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            temp_path = f.name

        try:
            success = predictor1.save_model(temp_path)
            assert success is True

            # Load into new predictor
            predictor2 = LSTMPredictor()
            success = predictor2.load_model(temp_path)
            assert success is True

            # Check that key attributes were restored
            assert predictor2.is_trained is True
            assert predictor2.room_id == "test_room"
            assert predictor2.model_type == ModelType.LSTM
            assert predictor2.model_version == predictor1.model_version
            assert predictor2.feature_names == predictor1.feature_names

            # Test that loaded model can make predictions
            pred_features = features.iloc[:1]
            predictions = await predictor2.predict(
                pred_features, datetime.now(timezone.utc)
            )
            assert len(predictions) == 1
            assert isinstance(predictions[0], PredictionResult)

        finally:
            # Clean up temp file
            Path(temp_path).unlink(missing_ok=True)

    def test_save_model_failure(self):
        """Test save model error handling."""
        predictor = LSTMPredictor()

        # Try to save to invalid path
        success = predictor.save_model("/invalid/path/model.pkl")
        assert success is False

    def test_load_model_failure(self):
        """Test load model error handling."""
        predictor = LSTMPredictor()

        # Try to load from non-existent file
        success = predictor.load_model("/nonexistent/file.pkl")
        assert success is False


class TestValidation:
    """Test model validation functionality."""

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


class TestIntegrationScenarios:
    """Integration tests for complete workflows."""

    async def test_complete_training_prediction_workflow(self):
        """Test complete workflow from training to prediction."""
        # Create predictor
        predictor = LSTMPredictor(
            room_id="integration_room",
            sequence_length=10,
            max_iter=100,
        )

        # Create synthetic time series data
        np.random.seed(42)
        n_samples = 80

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

        # Split into train/validation
        train_features = features.iloc[:60]
        train_targets = targets.iloc[:60]
        val_features = features.iloc[60:]
        val_targets = targets.iloc[60:]

        # Train model
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

    async def test_multiple_training_sessions(self):
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
        result1 = await predictor.train(features1, targets1)
        version1 = predictor.model_version

        # Second training
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
