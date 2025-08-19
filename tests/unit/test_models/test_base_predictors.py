"""
Comprehensive unit tests for BasePredictor implementations and model algorithms.

This test module covers the abstract BasePredictor interface, model-specific implementations,
and algorithm testing for LSTM, XGBoost, HMM, and Gaussian Process predictors.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.constants import ModelType
from src.core.exceptions import ModelTrainingError, PredictionError
from src.models.base.gaussian_process_predictor import GaussianProcessPredictor
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import BasePredictor, PredictionResult, TrainingResult
from src.models.base.xgboost_predictor import XGBoostPredictor


@pytest.fixture
def synthetic_training_data():
    """Create synthetic training data with realistic occupancy patterns."""
    n_samples = 1000
    n_features = 20

    # Generate time-based features
    hours = np.random.randint(0, 24, n_samples)
    days = np.random.randint(0, 7, n_samples)

    # Create feature matrix with temporal patterns
    features = {
        "hour_sin": np.sin(2 * np.pi * hours / 24),
        "hour_cos": np.cos(2 * np.pi * hours / 24),
        "day_sin": np.sin(2 * np.pi * days / 7),
        "day_cos": np.cos(2 * np.pi * days / 7),
        "time_since_last_occupied": np.random.exponential(3600, n_samples),  # Seconds
        "time_since_last_vacant": np.random.exponential(1800, n_samples),
        "current_state_duration": np.random.exponential(2400, n_samples),
        "temperature": np.random.normal(21, 3, n_samples),
        "humidity": np.random.normal(45, 10, n_samples),
        "light_level": np.random.exponential(200, n_samples),
        "motion_events_last_hour": np.random.poisson(5, n_samples),
        "door_events_last_hour": np.random.poisson(2, n_samples),
    }

    # Add noise features
    for i in range(8):
        features[f"noise_{i}"] = np.random.normal(0, 1, n_samples)

    features_df = pd.DataFrame(features)

    # Create realistic targets based on features
    # Higher probability of occupancy during work hours and when motion detected
    work_hour_factor = np.where((hours >= 8) & (hours <= 18), 1.5, 0.5)
    motion_factor = 1 + 0.1 * features["motion_events_last_hour"]
    base_probability = 0.3 + 0.4 * work_hour_factor * motion_factor

    # Generate time until next transition (in minutes)
    next_transition_minutes = np.random.exponential(30, n_samples)
    is_currently_occupied = np.random.binomial(1, base_probability, n_samples)

    targets_df = pd.DataFrame(
        {
            "next_transition_minutes": next_transition_minutes,
            "is_currently_occupied": is_currently_occupied,
            "confidence_score": np.random.uniform(0.6, 0.95, n_samples),
            "transition_probability": base_probability,
        }
    )

    return features_df, targets_df


@pytest.fixture
def validation_data(synthetic_training_data):
    """Create validation dataset separate from training."""
    features, targets = synthetic_training_data

    # Use last 200 samples for validation
    val_features = features.tail(200).copy()
    val_targets = targets.tail(200).copy()

    # Training data is everything except validation
    train_features = features.head(800).copy()
    train_targets = targets.head(800).copy()

    return train_features, train_targets, val_features, val_targets


@pytest.fixture
def prediction_features():
    """Create features for prediction testing."""
    n_samples = 10

    features = {
        "hour_sin": np.sin(
            2 * np.pi * np.array([14, 15, 16, 9, 10, 20, 21, 22, 7, 8]) / 24
        ),
        "hour_cos": np.cos(
            2 * np.pi * np.array([14, 15, 16, 9, 10, 20, 21, 22, 7, 8]) / 24
        ),
        "day_sin": np.sin(2 * np.pi * np.array([1, 1, 1, 2, 2, 5, 5, 5, 6, 6]) / 7),
        "day_cos": np.cos(2 * np.pi * np.array([1, 1, 1, 2, 2, 5, 5, 5, 6, 6]) / 7),
        "time_since_last_occupied": [
            3600,
            7200,
            1800,
            900,
            3600,
            14400,
            1800,
            3600,
            900,
            1800,
        ],
        "time_since_last_vacant": [
            1800,
            3600,
            900,
            7200,
            1800,
            3600,
            900,
            1800,
            3600,
            900,
        ],
        "current_state_duration": [
            1800,
            3600,
            900,
            1800,
            3600,
            900,
            1800,
            3600,
            900,
            1800,
        ],
        "temperature": [
            20.5,
            21.0,
            19.8,
            22.1,
            20.2,
            18.9,
            19.5,
            20.8,
            21.2,
            19.9,
        ],
        "humidity": [45, 48, 42, 50, 46, 44, 47, 49, 43, 45],
        "light_level": [250, 300, 180, 400, 350, 50, 80, 60, 300, 280],
        "motion_events_last_hour": [5, 8, 3, 12, 7, 1, 2, 0, 6, 9],
        "door_events_last_hour": [2, 3, 1, 4, 2, 0, 1, 0, 2, 3],
    }

    # Add noise features to match training
    for i in range(8):
        features[f"noise_{i}"] = np.random.normal(0, 1, n_samples)

    return pd.DataFrame(features)


class TestBasePredictor:
    """Test the abstract BasePredictor interface."""

    def test_base_predictor_initialization(self):
        """Test BasePredictor initialization with different parameters."""
        # Test with minimal parameters
        predictor = LSTMPredictor(room_id="test_room")
        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id == "test_room"
        assert not predictor.is_trained
        assert predictor.model_version == "v1.0"
        assert predictor.feature_names == []

        # Test with all parameters
        predictor_full = XGBoostPredictor(room_id="living_room")
        assert predictor_full.room_id == "living_room"
        assert predictor_full.model_type == ModelType.XGBOOST

    def test_model_info(self):
        """Test model information retrieval."""
        predictor = HMMPredictor(room_id="test_room")
        info = predictor.get_model_info()

        assert info["model_type"] == ModelType.HMM.value
        assert info["room_id"] == "test_room"
        assert info["is_trained"] is False
        assert info["training_date"] is None
        assert info["feature_count"] == 0
        assert info["training_sessions"] == 0
        assert info["predictions_made"] == 0

    def test_training_history_tracking(self):
        """Test training history is properly tracked."""
        predictor = LSTMPredictor(room_id="test_room")

        # Initially empty
        assert len(predictor.training_history) == 0
        assert len(predictor.get_training_history()) == 0

        # Mock training result
        training_result = TrainingResult(
            success=True,
            training_time_seconds=120.5,
            model_version="v1.0",
            training_samples=1000,
            validation_score=0.85,
            training_score=0.92,
        )

        predictor.training_history.append(training_result)

        history = predictor.get_training_history()
        assert len(history) == 1
        assert history[0]["success"] is True
        assert history[0]["training_time_seconds"] == 120.5
        assert history[0]["validation_score"] == 0.85

    def test_feature_validation(self, prediction_features):
        """Test feature validation functionality."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Untrained model should return False with warning
        assert predictor.validate_features(prediction_features) is False

        # Set up trained model state
        predictor.is_trained = True
        predictor.feature_names = list(prediction_features.columns)

        # Valid features should pass
        assert predictor.validate_features(prediction_features) is True

        # Missing features should fail
        incomplete_features = prediction_features.drop(
            columns=["hour_sin", "temperature"]
        )
        assert predictor.validate_features(incomplete_features) is False

        # Extra features should pass with warning
        extra_features = prediction_features.copy()
        extra_features["extra_feature"] = 1.0
        assert predictor.validate_features(extra_features) is True

    def test_prediction_history_management(self):
        """Test prediction history tracking and memory management."""
        predictor = GaussianProcessPredictor(room_id="test_room")

        # Add many predictions to test memory management
        prediction_time = datetime.utcnow()

        for i in range(1100):  # More than the 1000 limit
            result = PredictionResult(
                predicted_time=prediction_time + timedelta(seconds=1800),
                transition_type="vacant_to_occupied",
                confidence_score=0.8,
                model_type=ModelType.GP.value,
            )
            predictor._record_prediction(prediction_time, result)

        # Should be limited to the last 500 when cleanup triggers
        # The implementation truncates to [-500:] when exceeding 1000
        assert len(predictor.prediction_history) == 500

    def test_model_version_generation(self):
        """Test model version generation and updating."""
        predictor = LSTMPredictor(room_id="test_room")

        # Initial version
        assert predictor.model_version == "v1.0"

        # After training
        training_result = TrainingResult(
            success=True,
            training_time_seconds=60.0,
            model_version="v1.1",
            training_samples=500,
            validation_score=0.80,
            training_score=0.85,
        )

        predictor.training_history.append(training_result)
        predictor.model_version = "v1.1"

        assert predictor.model_version == "v1.1"

    def test_prediction_accuracy_calculation(self):
        """Test prediction accuracy calculation with mock data."""
        predictor = XGBoostPredictor(room_id="test_room")

        # No predictions should return None
        accuracy = predictor.get_prediction_accuracy(hours_back=24)
        assert accuracy is None

        # Add some predictions (not enough)
        base_time = datetime.utcnow()
        for i in range(3):
            result = PredictionResult(
                predicted_time=base_time + timedelta(seconds=1800),
                transition_type="occupied_to_vacant",
                confidence_score=0.9,
                model_type=ModelType.XGBOOST.value,
            )
            predictor._record_prediction(base_time - timedelta(hours=i), result)

        # Still not enough predictions
        accuracy = predictor.get_prediction_accuracy(hours_back=24)
        assert accuracy is None

        # Add more predictions
        for i in range(5):
            result = PredictionResult(
                predicted_time=base_time + timedelta(seconds=1800),
                transition_type="occupied_to_vacant",
                confidence_score=0.9,
                model_type=ModelType.XGBOOST.value,
            )
            predictor._record_prediction(base_time - timedelta(hours=i), result)

        # Now should return accuracy (placeholder value)
        accuracy = predictor.get_prediction_accuracy(hours_back=24)
        assert accuracy == 0.85  # Placeholder value from implementation

    def test_clear_prediction_history(self):
        """Test clearing prediction history."""
        predictor = HMMPredictor(room_id="test_room")

        # Add some predictions
        prediction_time = datetime.utcnow()
        for i in range(5):
            result = PredictionResult(
                predicted_time=prediction_time + timedelta(seconds=1800),
                transition_type="vacant_to_occupied",
                confidence_score=0.8,
                model_type=ModelType.HMM.value,
            )
            predictor._record_prediction(prediction_time, result)

        assert len(predictor.prediction_history) == 5

        # Clear history
        predictor.clear_prediction_history()
        assert len(predictor.prediction_history) == 0


class TestLSTMPredictor:
    """Test LSTM predictor implementation."""

    @pytest.mark.asyncio
    async def test_lstm_initialization(self):
        """Test LSTM predictor initialization."""
        predictor = LSTMPredictor(room_id="bedroom")

        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id == "bedroom"
        assert not predictor.is_trained

        # Check LSTM-specific parameters
        assert "hidden_size" in predictor.model_params
        assert "num_layers" in predictor.model_params
        assert "dropout" in predictor.model_params
        assert "learning_rate" in predictor.model_params

    @pytest.mark.asyncio
    async def test_lstm_training_convergence(self, validation_data):
        """Test LSTM training process and convergence."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = LSTMPredictor(room_id="test_room")

        # Use smaller dataset for faster training
        small_train_features = train_features.head(100)
        small_train_targets = train_targets.head(100)

        await predictor.train(small_train_features, small_train_targets)

        assert predictor.is_trained
        assert len(predictor.training_history) == 1

        training_result = predictor.training_history[0]
        assert training_result.success
        assert training_result.training_samples == len(small_train_features)
        assert 0.0 <= training_result.validation_score <= 1.0

    @pytest.mark.asyncio
    async def test_lstm_prediction_format(self, validation_data):
        """Test LSTM prediction output format."""
        train_features, train_targets, val_features, _ = validation_data
        predictor = LSTMPredictor(room_id="test_room")

        # Use smaller dataset for faster training
        small_train_features = train_features.head(100)
        small_train_targets = train_targets.head(100)

        await predictor.train(small_train_features, small_train_targets)

        predictions = await predictor.predict(
            val_features.head(5), datetime.utcnow(), "vacant"
        )

        assert len(predictions) == 5

        for prediction in predictions:
            assert isinstance(prediction, PredictionResult)
            assert isinstance(prediction.predicted_time, datetime)
            assert prediction.transition_type in [
                "vacant_to_occupied",
                "occupied_to_vacant",
            ]
            assert 0.0 <= prediction.confidence_score <= 1.0
            assert prediction.model_type == ModelType.LSTM.value

    @pytest.mark.asyncio
    async def test_lstm_sequence_creation(self, validation_data):
        """Test LSTM sequence data preparation."""
        train_features, train_targets, _, _ = validation_data
        predictor = LSTMPredictor(room_id="test_room")

        # Test with small dataset
        small_features = train_features.head(50)
        small_targets = train_targets.head(50)

        # Create sequences
        X_seq, y_seq = predictor._create_sequences(
            small_features.values, small_targets.values, sequence_length=10
        )

        # Should create sequences correctly
        expected_sequences = len(small_features) - 10 + 1  # 41 sequences
        assert len(X_seq) == expected_sequences
        assert len(y_seq) == expected_sequences

        # Check sequence dimensions
        assert X_seq.shape[1] == 10  # sequence_length
        assert X_seq.shape[2] == small_features.shape[1]  # n_features

    @pytest.mark.asyncio
    async def test_lstm_prediction_intervals(self, validation_data):
        """Test LSTM prediction intervals."""
        train_features, train_targets, val_features, _ = validation_data
        predictor = LSTMPredictor(room_id="test_room")

        # Small dataset
        await predictor.train(train_features.head(50), train_targets.head(50))

        predictions = await predictor.predict(
            val_features.head(3), datetime.utcnow(), "occupied"
        )

        # Check that prediction intervals are provided
        for pred in predictions:
            if pred.prediction_interval:
                lower, upper = pred.prediction_interval
                assert isinstance(lower, datetime)
                assert isinstance(upper, datetime)
                assert lower <= pred.predicted_time <= upper


class TestXGBoostPredictor:
    """Test XGBoost predictor implementation."""

    @pytest.mark.asyncio
    async def test_xgboost_initialization(self):
        """Test XGBoost predictor initialization."""
        predictor = XGBoostPredictor(room_id="kitchen")

        assert predictor.model_type == ModelType.XGBOOST
        assert predictor.room_id == "kitchen"
        assert not predictor.is_trained

        # Check XGBoost-specific parameters
        assert "n_estimators" in predictor.model_params
        assert "max_depth" in predictor.model_params
        assert "learning_rate" in predictor.model_params

    @pytest.mark.asyncio
    async def test_xgboost_training(self, validation_data):
        """Test XGBoost training with different parameters."""
        train_features, train_targets, _, _ = validation_data
        predictor = XGBoostPredictor(room_id="test_room")

        await predictor.train(train_features, train_targets)

        assert predictor.is_trained
        assert len(predictor.training_history) == 1

        training_result = predictor.training_history[0]
        assert training_result.success
        assert 0.0 <= training_result.validation_score <= 1.0

    @pytest.mark.asyncio
    async def test_xgboost_feature_importance(self, validation_data):
        """Test XGBoost feature importance extraction."""
        train_features, train_targets, _, _ = validation_data
        predictor = XGBoostPredictor(room_id="test_room")

        await predictor.train(train_features, train_targets)

        predictions = await predictor.predict(
            train_features.head(3), datetime.utcnow(), "vacant"
        )

        # Check predictions include feature importance if available
        for pred in predictions:
            if pred.prediction_metadata:
                # Feature importance might be included
                assert isinstance(pred.prediction_metadata, dict)

    @pytest.mark.asyncio
    async def test_xgboost_prediction_confidence(self, validation_data):
        """Test XGBoost confidence scoring."""
        train_features, train_targets, val_features, _ = validation_data
        predictor = XGBoostPredictor(room_id="test_room")

        await predictor.train(train_features, train_targets)

        predictions = await predictor.predict(
            val_features.head(5), datetime.utcnow(), "occupied"
        )

        # XGBoost should provide reasonable confidence scores
        confidences = [p.confidence_score for p in predictions]
        assert all(0.0 <= c <= 1.0 for c in confidences)
        assert len(set(confidences)) > 1  # Should have some variation


class TestHMMPredictor:
    """Test HMM predictor implementation."""

    @pytest.mark.asyncio
    async def test_hmm_initialization(self):
        """Test HMM predictor initialization."""
        predictor = HMMPredictor(room_id="bathroom")

        assert predictor.model_type == ModelType.HMM
        assert predictor.room_id == "bathroom"
        assert not predictor.is_trained

        # Check HMM-specific parameters
        assert "n_states" in predictor.model_params
        assert "covariance_type" in predictor.model_params

    @pytest.mark.asyncio
    async def test_hmm_state_modeling(self, validation_data):
        """Test HMM state transition modeling."""
        train_features, train_targets, _, _ = validation_data
        predictor = HMMPredictor(room_id="test_room")

        await predictor.train(train_features, train_targets)

        assert predictor.is_trained
        assert len(predictor.training_history) == 1

        # HMM should model state transitions
        training_result = predictor.training_history[0]
        assert training_result.success

    @pytest.mark.asyncio
    async def test_hmm_confidence_uncertainty(self, validation_data):
        """Test HMM confidence calculation based on state probabilities."""
        train_features, train_targets, val_features, _ = validation_data
        predictor = HMMPredictor(room_id="test_room")

        await predictor.train(train_features, train_targets)

        predictions = await predictor.predict(
            val_features.head(5), datetime.utcnow(), "occupied"
        )

        # HMM confidence should reflect state uncertainty
        confidences = [p.confidence_score for p in predictions]

        assert all(0.0 <= c <= 1.0 for c in confidences)
        # HMM often has more variable confidence due to state probabilities
        assert np.std(confidences) > 0.01  # Should have some variability


class TestGaussianProcessPredictor:
    """Test Gaussian Process predictor implementation."""

    @pytest.mark.asyncio
    async def test_gp_initialization(self):
        """Test Gaussian Process predictor initialization."""
        predictor = GaussianProcessPredictor(room_id="office")

        assert predictor.model_type == ModelType.GP
        assert predictor.room_id == "office"
        assert not predictor.is_trained

        # Check GP-specific parameters
        assert "kernel" in predictor.model_params
        assert "alpha" in predictor.model_params
        assert "normalize_y" in predictor.model_params

    @pytest.mark.asyncio
    async def test_gp_uncertainty_quantification(self, validation_data):
        """Test GP's uncertainty quantification capabilities."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = GaussianProcessPredictor(room_id="test_room")

        # Use smaller dataset for GP training (GP is computationally expensive)
        small_train_features = train_features.head(100)
        small_train_targets = train_targets.head(100)

        await predictor.train(small_train_features, small_train_targets)

        predictions = await predictor.predict(
            val_features.head(5), datetime.utcnow(), "vacant"
        )

        assert len(predictions) == 5

        # GP should provide prediction intervals
        for prediction in predictions:
            assert prediction.prediction_interval is not None

            # Prediction interval should be reasonable
            lower, upper = prediction.prediction_interval
            assert isinstance(lower, datetime)
            assert isinstance(upper, datetime)
            assert lower <= prediction.predicted_time <= upper

            # Should have uncertainty quantification in metadata
            if prediction.prediction_metadata:
                assert "uncertainty_quantification" in prediction.prediction_metadata
                uncertainty = prediction.prediction_metadata[
                    "uncertainty_quantification"
                ]
                assert "aleatoric_uncertainty" in uncertainty
                assert "epistemic_uncertainty" in uncertainty

    @pytest.mark.asyncio
    async def test_gp_prediction_intervals(self, validation_data):
        """Test GP prediction interval calibration."""
        train_features, train_targets, val_features, _ = validation_data
        predictor = GaussianProcessPredictor(room_id="test_room")

        # Small dataset for GP
        await predictor.train(train_features.head(50), train_targets.head(50))

        predictions = await predictor.predict(
            val_features.head(3), datetime.utcnow(), "occupied"
        )

        # Check prediction intervals
        for pred in predictions:
            if pred.prediction_interval:
                lower, upper = pred.prediction_interval
                interval_width = (upper - lower).total_seconds()

                # Interval should be reasonable (not too narrow or too wide)
                assert 60 <= interval_width <= 7200  # Between 1 minute and 2 hours

                # Predicted time should be within interval
                pred_time = pred.predicted_time
                assert lower <= pred_time <= upper

    @pytest.mark.asyncio
    async def test_gp_kernel_optimization(self, validation_data):
        """Test GP kernel hyperparameter optimization."""
        train_features, train_targets, _, _ = validation_data
        predictor = GaussianProcessPredictor(room_id="test_room")

        # Very small dataset for GP
        tiny_features = train_features.head(20)
        tiny_targets = train_targets.head(20)

        await predictor.train(tiny_features, tiny_targets)

        assert predictor.is_trained
        training_result = predictor.training_history[0]
        assert training_result.success

        # GP should optimize kernel hyperparameters during training
        assert 0.0 <= training_result.validation_score <= 1.0


class TestModelComparison:
    """Compare different model implementations."""

    @pytest.mark.asyncio
    async def test_model_prediction_consistency(
        self, validation_data, prediction_features
    ):
        """Test that all models produce consistent prediction formats."""
        train_features, train_targets, _, _ = validation_data
        current_state = "vacant"
        current_time = datetime.utcnow()

        # Use very small dataset for quick testing
        small_train_features = train_features.head(50)
        small_train_targets = train_targets.head(50)
        test_features = prediction_features.head(3)

        models = [
            LSTMPredictor(room_id="test_room"),
            XGBoostPredictor(room_id="test_room"),
            HMMPredictor(room_id="test_room"),
            GaussianProcessPredictor(room_id="test_room"),
        ]

        all_predictions = []

        for model in models:
            await model.train(small_train_features, small_train_targets)

            predictions = await model.predict(
                test_features, current_time, current_state
            )

            all_predictions.extend(predictions)

            # Verify consistent prediction format
            assert len(predictions) == 3
            for pred in predictions:
                assert isinstance(pred, PredictionResult)
                assert isinstance(pred.predicted_time, datetime)
                assert 0.0 <= pred.confidence_score <= 1.0

        # All models should produce predictions
        assert len(all_predictions) == 12  # 4 models * 3 predictions each

    @pytest.mark.asyncio
    async def test_training_performance_comparison(self, validation_data):
        """Compare training performance across models."""
        train_features, train_targets, _, _ = validation_data

        # Small dataset for performance testing
        perf_features = train_features.head(100)
        perf_targets = train_targets.head(100)

        models = [
            ("LSTM", LSTMPredictor(room_id="test_room")),
            ("XGBoost", XGBoostPredictor(room_id="test_room")),
            ("HMM", HMMPredictor(room_id="test_room")),
            ("GP", GaussianProcessPredictor(room_id="test_room")),
        ]

        training_times = {}

        for model_name, model in models:
            start_time = datetime.utcnow()
            await model.train(perf_features, perf_targets)
            end_time = datetime.utcnow()

            training_time = (end_time - start_time).total_seconds()
            training_times[model_name] = training_time

            # Verify successful training
            assert model.is_trained
            assert len(model.training_history) == 1
            assert model.training_history[0].success

        # All models should complete training in reasonable time
        for model_name, time_taken in training_times.items():
            assert time_taken < 30.0  # Should complete within 30 seconds

    def test_model_parameter_validation(self):
        """Test parameter validation across different models."""
        models = [
            LSTMPredictor(room_id="test_room"),
            XGBoostPredictor(room_id="test_room"),
            HMMPredictor(room_id="test_room"),
            GaussianProcessPredictor(room_id="test_room"),
        ]

        for model in models:
            # All models should have valid default parameters
            assert isinstance(model.model_params, dict)
            assert len(model.model_params) > 0

            # Model info should be complete
            info = model.get_model_info()
            assert "model_type" in info
            assert "room_id" in info
            assert "is_trained" in info
            assert "model_params" in info
