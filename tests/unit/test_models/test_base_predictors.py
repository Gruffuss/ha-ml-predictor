"""
Comprehensive unit tests for base ML predictors.

This module tests all base predictor implementations (LSTM, XGBoost, HMM, GP)
with focus on prediction format validation, training convergence, confidence
calibration, and performance benchmarking.
"""

import asyncio
import numpy as np
import pandas as pd
import pytest
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

from src.core.constants import ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.predictor import BasePredictor, PredictionResult, TrainingResult
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.xgboost_predictor import XGBoostPredictor
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.gp_predictor import GaussianProcessPredictor


@pytest.fixture
def synthetic_training_data():
    """Create synthetic training data with known occupancy patterns."""
    np.random.seed(42)
    
    # Generate 1000 samples with realistic occupancy patterns
    n_samples = 1000
    n_features = 20
    
    # Create features with temporal patterns
    features = {}
    
    # Time-based features (cyclical)
    hours = np.random.randint(0, 24, n_samples)
    features['hour_sin'] = np.sin(2 * np.pi * hours / 24)
    features['hour_cos'] = np.cos(2 * np.pi * hours / 24)
    
    days = np.random.randint(0, 7, n_samples)
    features['day_sin'] = np.sin(2 * np.pi * days / 7)
    features['day_cos'] = np.cos(2 * np.pi * days / 7)
    
    # Occupancy-related features
    features['time_since_last_occupied'] = np.random.exponential(3600, n_samples)
    features['time_since_last_vacant'] = np.random.exponential(7200, n_samples)
    features['current_state_duration'] = np.random.exponential(1800, n_samples)
    
    # Environmental features
    features['temperature'] = 18 + np.random.normal(0, 3, n_samples)
    features['humidity'] = 45 + np.random.normal(0, 10, n_samples)
    features['light_level'] = np.random.exponential(300, n_samples)
    
    # Movement pattern features
    features['motion_events_last_hour'] = np.random.poisson(5, n_samples)
    features['door_events_last_hour'] = np.random.poisson(2, n_samples)
    
    # Add noise features
    for i in range(n_features - len(features)):
        features[f'noise_{i}'] = np.random.normal(0, 1, n_samples)
    
    # Create realistic targets (time until next transition in seconds)
    # Pattern: shorter transitions during day, longer at night
    base_time = 1800  # 30 minutes
    day_factor = np.where(
        (hours >= 6) & (hours <= 22), 0.7, 2.0  # Shorter during day
    )
    
    # Add occupancy state influence
    occupancy_state = np.random.choice(['occupied', 'vacant'], n_samples)
    state_factor = np.where(occupancy_state == 'occupied', 1.2, 0.8)
    
    targets = base_time * day_factor * state_factor * (1 + np.random.normal(0, 0.3, n_samples))
    targets = np.clip(targets, 60, 14400)  # Clip between 1 minute and 4 hours
    
    features_df = pd.DataFrame(features)
    targets_df = pd.DataFrame({
        'time_until_transition_seconds': targets,
        'transition_type': [
            'occupied_to_vacant' if s == 'occupied' else 'vacant_to_occupied'
            for s in occupancy_state
        ],
        'target_time': [datetime.utcnow() + timedelta(seconds=i) for i in range(n_samples)]
    })
    
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
        'hour_sin': np.sin(2 * np.pi * np.array([14, 15, 16, 9, 10, 20, 21, 22, 7, 8]) / 24),
        'hour_cos': np.cos(2 * np.pi * np.array([14, 15, 16, 9, 10, 20, 21, 22, 7, 8]) / 24),
        'day_sin': np.sin(2 * np.pi * np.array([1, 1, 1, 2, 2, 5, 5, 5, 6, 6]) / 7),
        'day_cos': np.cos(2 * np.pi * np.array([1, 1, 1, 2, 2, 5, 5, 5, 6, 6]) / 7),
        'time_since_last_occupied': [3600, 7200, 1800, 900, 3600, 14400, 1800, 3600, 900, 1800],
        'time_since_last_vacant': [1800, 3600, 900, 7200, 1800, 3600, 900, 1800, 3600, 900],
        'current_state_duration': [1800, 3600, 900, 1800, 3600, 900, 1800, 3600, 900, 1800],
        'temperature': [20.5, 21.0, 19.8, 22.1, 20.2, 18.9, 19.5, 20.8, 21.2, 19.9],
        'humidity': [45, 48, 42, 50, 46, 44, 47, 49, 43, 45],
        'light_level': [250, 300, 180, 400, 350, 50, 80, 60, 300, 280],
        'motion_events_last_hour': [5, 8, 3, 12, 7, 1, 2, 0, 6, 9],
        'door_events_last_hour': [2, 3, 1, 4, 2, 0, 1, 0, 2, 3],
    }
    
    # Add noise features to match training
    for i in range(8):
        features[f'noise_{i}'] = np.random.normal(0, 1, n_samples)
    
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
        predictor_full = XGBoostPredictor(
            room_id="living_room"
        )
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
            training_score=0.92
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
        incomplete_features = prediction_features.drop(columns=['hour_sin', 'temperature'])
        assert predictor.validate_features(incomplete_features) is False
        
        # Extra features should pass with warning
        extra_features = prediction_features.copy()
        extra_features['extra_feature'] = 1.0
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
                model_type=ModelType.GP.value
            )
            predictor._record_prediction(prediction_time, result)
        
        # Should be limited to 500 (half of 1000 when cleanup triggers)
        assert len(predictor.prediction_history) == 500
    
    def test_model_version_generation(self):
        """Test automatic model version generation."""
        predictor = LSTMPredictor(room_id="test_room")
        
        # Initial version
        version1 = predictor._generate_model_version()
        assert version1 == "v1.0"
        
        # Add training history to test version increment
        predictor.training_history.append(
            TrainingResult(
                success=True,
                training_time_seconds=100,
                model_version="v1.2",
                training_samples=1000
            )
        )
        
        version2 = predictor._generate_model_version()
        assert version2 == "v1.3"


class TestLSTMPredictor:
    """Test LSTM predictor implementation."""
    
    @pytest.mark.asyncio
    async def test_lstm_initialization(self):
        """Test LSTM predictor initialization."""
        predictor = LSTMPredictor(room_id="test_room")
        
        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id == "test_room"
        assert not predictor.is_trained
        
        # Check default LSTM parameters
        assert "lstm_units" in predictor.model_params
        assert "sequence_length" in predictor.model_params
        assert "dropout_rate" in predictor.model_params
    
    @pytest.mark.asyncio
    async def test_lstm_training_convergence(self, validation_data):
        """Test LSTM training and convergence."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = LSTMPredictor(room_id="test_room")
        
        # Mock the underlying model to avoid actual neural network training
        with patch('src.models.base.lstm_predictor.MLPRegressor') as mock_mlp:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.score.return_value = 0.85
            mock_model.predict.return_value = np.array([1800.0] * len(val_features))
            mock_mlp.return_value = mock_model
            
            # Train model
            result = await predictor.train(
                train_features, train_targets, val_features, val_targets
            )
            
            # Verify training result
            assert result.success is True
            assert result.training_samples == len(train_features)
            assert result.validation_score is not None
            assert result.training_score is not None
            assert predictor.is_trained is True
            assert predictor.feature_names == list(train_features.columns)
    
    @pytest.mark.asyncio
    async def test_lstm_prediction_format(self, validation_data, prediction_features):
        """Test LSTM prediction format and consistency."""
        train_features, train_targets, _, _ = validation_data
        predictor = LSTMPredictor(room_id="test_room")
        
        # Mock trained model
        with patch('src.models.base.lstm_predictor.MLPRegressor') as mock_mlp:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.score.return_value = 0.85
            mock_model.predict.return_value = np.array([1800.0] * len(prediction_features))
            mock_mlp.return_value = mock_model
            
            # Train first
            await predictor.train(train_features, train_targets)
            
            # Make predictions
            prediction_time = datetime.utcnow()
            results = await predictor.predict(
                prediction_features, prediction_time, "vacant"
            )
            
            # Verify prediction format
            assert len(results) == len(prediction_features)
            
            for result in results:
                assert isinstance(result, PredictionResult)
                assert isinstance(result.predicted_time, datetime)
                assert result.transition_type in ["vacant_to_occupied", "occupied_to_vacant"]
                assert 0.0 <= result.confidence_score <= 1.0
                assert result.model_type == ModelType.LSTM.value
                assert result.model_version is not None
                
                # Verify prediction time is reasonable (not too far in past/future)
                time_delta = (result.predicted_time - prediction_time).total_seconds()
                assert 60 <= time_delta <= 14400  # Between 1 minute and 4 hours
    
    @pytest.mark.asyncio
    async def test_lstm_confidence_calibration(self, validation_data):
        """Test LSTM confidence score calibration."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = LSTMPredictor(room_id="test_room")
        
        with patch('src.models.base.lstm_predictor.MLPRegressor') as mock_mlp:
            mock_model = MagicMock()
            mock_model.fit.return_value = mock_model
            mock_model.score.return_value = 0.85
            
            # Mock different prediction scenarios for confidence testing
            mock_predictions = []
            for i in range(len(val_features)):
                # Vary predictions to test confidence calculation
                mock_predictions.append(1800 + (i * 100))  # Varying predictions
            
            mock_model.predict.return_value = np.array(mock_predictions)
            mock_mlp.return_value = mock_model
            
            await predictor.train(train_features, train_targets)
            
            predictions = await predictor.predict(val_features, datetime.utcnow(), "vacant")
            
            # Test confidence ranges
            confidences = [p.confidence_score for p in predictions]
            
            assert all(0.0 <= c <= 1.0 for c in confidences)
            assert len(set(confidences)) > 1  # Should have some variation
            
            # Confidence should generally be lower for predictions far from training data
            # This is a basic check - real calibration would need actual validation
            mean_confidence = np.mean(confidences)
            assert 0.3 <= mean_confidence <= 0.95


class TestXGBoostPredictor:
    """Test XGBoost predictor implementation."""
    
    @pytest.mark.asyncio
    async def test_xgboost_initialization(self):
        """Test XGBoost predictor initialization."""
        predictor = XGBoostPredictor(room_id="living_room")
        
        assert predictor.model_type == ModelType.XGBOOST
        assert predictor.room_id == "living_room"
        assert not predictor.is_trained
        
        # Check XGBoost-specific parameters
        assert "n_estimators" in predictor.model_params
        assert "max_depth" in predictor.model_params
        assert "learning_rate" in predictor.model_params
        assert "objective" in predictor.model_params
    
    @pytest.mark.asyncio
    async def test_xgboost_training_and_feature_importance(self, validation_data):
        """Test XGBoost training and feature importance calculation."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = XGBoostPredictor(room_id="test_room")
        
        # Train model
        result = await predictor.train(
            train_features, train_targets, val_features, val_targets
        )
        
        # Verify training success
        assert result.success is True
        assert result.training_samples == len(train_features)
        assert predictor.is_trained is True
        
        # Test feature importance
        importance = predictor.get_feature_importance()
        assert len(importance) > 0
        assert all(isinstance(v, (int, float)) for v in importance.values())
        assert all(v >= 0 for v in importance.values())
        
        # Should have importance for temporal features
        temporal_features = ['hour_sin', 'hour_cos', 'time_since_last_occupied']
        found_temporal = [f for f in temporal_features if f in importance]
        assert len(found_temporal) > 0
    
    @pytest.mark.asyncio
    async def test_xgboost_prediction_performance(self, validation_data, prediction_features):
        """Test XGBoost prediction performance and timing."""
        train_features, train_targets, _, _ = validation_data
        predictor = XGBoostPredictor(room_id="test_room")
        
        # Train model
        await predictor.train(train_features, train_targets)
        
        # Measure prediction performance
        prediction_time = datetime.utcnow()
        start_time = time.time()
        
        predictions = await predictor.predict(
            prediction_features, prediction_time, "occupied"
        )
        
        elapsed_time = time.time() - start_time
        
        # Verify predictions
        assert len(predictions) == len(prediction_features)
        
        # Performance benchmark: should be fast (< 1 second for 10 predictions)
        assert elapsed_time < 1.0
        
        # Test transition type logic
        transition_types = [p.transition_type for p in predictions]
        assert all(t in ["vacant_to_occupied", "occupied_to_vacant"] for t in transition_types)
    
    @pytest.mark.asyncio
    async def test_xgboost_incremental_update(self, validation_data):
        """Test XGBoost incremental learning capability."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = XGBoostPredictor(room_id="test_room")
        
        # Initial training
        initial_result = await predictor.train(train_features, train_targets)
        initial_version = predictor.model_version
        
        # Incremental update
        update_result = await predictor.incremental_update(
            val_features, val_targets, learning_rate=0.1
        )
        
        # Verify incremental update
        assert update_result.success is True
        assert update_result.training_samples == len(val_features)
        assert predictor.model_version != initial_version
        assert "incremental" in update_result.training_metrics.get("update_type", "")
        
        # Should still be able to make predictions
        test_predictions = await predictor.predict(
            val_features.head(5), datetime.utcnow(), "vacant"
        )
        assert len(test_predictions) == 5


class TestHMMPredictor:
    """Test Hidden Markov Model predictor implementation."""
    
    @pytest.mark.asyncio
    async def test_hmm_initialization(self):
        """Test HMM predictor initialization."""
        predictor = HMMPredictor(room_id="bedroom")
        
        assert predictor.model_type == ModelType.HMM
        assert predictor.room_id == "bedroom"
        assert not predictor.is_trained
        
        # Check HMM-specific parameters
        assert "n_components" in predictor.model_params
        assert "covariance_type" in predictor.model_params
        assert "n_iter" in predictor.model_params
    
    @pytest.mark.asyncio
    async def test_hmm_state_transition_modeling(self, validation_data):
        """Test HMM's ability to model state transitions."""
        train_features, train_targets, val_features, val_targets = validation_data
        predictor = HMMPredictor(room_id="test_room")
        
        # Train model
        result = await predictor.train(train_features, train_targets)
        
        assert result.success is True
        assert predictor.is_trained is True
        
        # Test that model learned some state structure
        # HMM should be able to identify occupancy states
        predictions = await predictor.predict(
            val_features.head(10), datetime.utcnow(), "vacant"
        )
        
        assert len(predictions) == 10
        
        # Check for reasonable prediction diversity
        prediction_times = [(p.predicted_time - datetime.utcnow()).total_seconds() 
                          for p in predictions]
        assert len(set(prediction_times)) > 1  # Should have some variation
    
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
                uncertainty = prediction.prediction_metadata["uncertainty_quantification"]
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
    async def test_gp_memory_usage_warning(self, validation_data):
        """Test GP behavior with larger datasets (memory considerations)."""
        train_features, train_targets, _, _ = validation_data
        predictor = GaussianProcessPredictor(room_id="test_room")
        
        # GP should handle or warn about large datasets
        # This test ensures GP doesn't crash with larger data
        large_features = train_features.head(200)  # Still reasonable for testing
        large_targets = train_targets.head(200)
        
        result = await predictor.train(large_features, large_targets)
        
        # Should still succeed but might be slower
        assert result.success is True
        
        # Training time should be recorded
        assert result.training_time_seconds > 0


class TestPredictorSerialization:
    """Test model serialization and persistence across all predictor types."""
    
    @pytest.mark.asyncio
    async def test_model_save_load_cycle(self, validation_data):
        """Test save/load cycle preserves model behavior."""
        train_features, train_targets, val_features, _ = validation_data
        
        predictors = [
            LSTMPredictor(room_id="test_room"),
            XGBoostPredictor(room_id="test_room"),
            HMMPredictor(room_id="test_room"),
        ]
        
        for predictor in predictors:
            model_name = predictor.__class__.__name__
            
            # Train model
            if model_name == "LSTMPredictor":
                with patch('src.models.base.lstm_predictor.MLPRegressor') as mock_mlp:
                    mock_model = MagicMock()
                    mock_model.fit.return_value = mock_model
                    mock_model.score.return_value = 0.85
                    mock_model.predict.return_value = np.array([1800.0] * len(val_features))
                    mock_mlp.return_value = mock_model
                    
                    await predictor.train(train_features.head(100), train_targets.head(100))
            else:
                await predictor.train(train_features.head(100), train_targets.head(100))
            
            # Get predictions before saving
            original_predictions = await predictor.predict(
                val_features.head(3), datetime.utcnow(), "vacant"
            )
            
            # Save model
            with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                save_success = predictor.save_model(temp_path)
                assert save_success is True
                
                # Create new predictor instance and load
                new_predictor = predictor.__class__(room_id="test_room")
                load_success = new_predictor.load_model(temp_path)
                assert load_success is True
                
                # Verify loaded model state
                assert new_predictor.is_trained is True
                assert new_predictor.model_type == predictor.model_type
                assert new_predictor.room_id == predictor.room_id
                assert new_predictor.feature_names == predictor.feature_names
                
                # Test predictions are similar (allowing for small numerical differences)
                if model_name != "LSTMPredictor":  # Skip LSTM due to mocking complexity
                    loaded_predictions = await new_predictor.predict(
                        val_features.head(3), datetime.utcnow(), "vacant"
                    )
                    
                    assert len(loaded_predictions) == len(original_predictions)
                    
                    # Check prediction consistency
                    for orig, loaded in zip(original_predictions, loaded_predictions):
                        time_diff = abs((orig.predicted_time - loaded.predicted_time).total_seconds())
                        assert time_diff < 60  # Allow small differences
                        assert orig.transition_type == loaded.transition_type
                        
            finally:
                # Clean up temp file
                Path(temp_path).unlink(missing_ok=True)
    
    def test_model_serialization_error_handling(self):
        """Test error handling during model serialization."""
        predictor = XGBoostPredictor(room_id="test_room")
        
        # Test saving untrained model
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as tmp_file:
            temp_path = tmp_file.name
        
        try:
            # Should still save (with untrained state)
            save_success = predictor.save_model(temp_path)
            assert save_success is True
            
            # Test loading into new instance
            new_predictor = XGBoostPredictor(room_id="test_room")
            load_success = new_predictor.load_model(temp_path)
            assert load_success is True
            assert new_predictor.is_trained is False
            
        finally:
            Path(temp_path).unlink(missing_ok=True)
        
        # Test invalid paths
        invalid_path = "/nonexistent/path/model.pkl"
        assert predictor.save_model(invalid_path) is False
        assert predictor.load_model(invalid_path) is False


class TestPredictorErrorHandling:
    """Test error handling across all predictor implementations."""
    
    @pytest.mark.asyncio
    async def test_prediction_on_untrained_model(self, prediction_features):
        """Test prediction fails appropriately on untrained models."""
        predictors = [
            LSTMPredictor(room_id="test_room"),
            XGBoostPredictor(room_id="test_room"),
            HMMPredictor(room_id="test_room"),
            GaussianProcessPredictor(room_id="test_room"),
        ]
        
        for predictor in predictors:
            with pytest.raises(ModelPredictionError):
                await predictor.predict(
                    prediction_features, datetime.utcnow(), "vacant"
                )
    
    @pytest.mark.asyncio
    async def test_training_with_insufficient_data(self):
        """Test training fails with insufficient data."""
        # Create very small dataset
        small_features = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        })
        small_targets = pd.DataFrame({
            'time_until_transition_seconds': [1800, 3600],
            'transition_type': ['vacant_to_occupied', 'occupied_to_vacant']
        })
        
        predictors = [
            XGBoostPredictor(room_id="test_room"),
            HMMPredictor(room_id="test_room"),
        ]
        
        for predictor in predictors:
            with pytest.raises(ModelTrainingError):
                await predictor.train(small_features, small_targets)
    
    @pytest.mark.asyncio
    async def test_invalid_feature_data(self, validation_data):
        """Test handling of invalid feature data."""
        train_features, train_targets, _, _ = validation_data
        predictor = XGBoostPredictor(room_id="test_room")
        
        # Train with good data
        await predictor.train(train_features.head(100), train_targets.head(100))
        
        # Test with NaN values
        nan_features = train_features.head(5).copy()
        nan_features.iloc[0, 0] = np.nan
        
        with pytest.raises(ModelPredictionError):
            await predictor.predict(nan_features, datetime.utcnow(), "vacant")
        
        # Test with completely different feature names
        wrong_features = pd.DataFrame({
            'wrong_feature1': [1, 2, 3],
            'wrong_feature2': [4, 5, 6]
        })
        
        with pytest.raises(ModelPredictionError):
            await predictor.predict(wrong_features, datetime.utcnow(), "vacant")


class TestPredictorPerformanceBenchmarks:
    """Performance benchmarks for all predictor types."""
    
    @pytest.mark.asyncio
    async def test_training_time_benchmarks(self, validation_data):
        """Test training time stays within reasonable bounds."""
        train_features, train_targets, _, _ = validation_data
        
        # Use moderate dataset size for benchmarking
        benchmark_features = train_features.head(500)
        benchmark_targets = train_targets.head(500)
        
        training_times = {}
        
        predictors = [
            ("XGBoost", XGBoostPredictor(room_id="test_room")),
            ("HMM", HMMPredictor(room_id="test_room")),
        ]
        
        for name, predictor in predictors:
            start_time = time.time()
            result = await predictor.train(benchmark_features, benchmark_targets)
            elapsed_time = time.time() - start_time
            
            training_times[name] = elapsed_time
            
            assert result.success is True
            # Training should complete within reasonable time (5 minutes max)
            assert elapsed_time < 300
            
        # XGBoost should generally be faster than HMM for this dataset size
        if "XGBoost" in training_times and "HMM" in training_times:
            print(f"Training times - XGBoost: {training_times['XGBoost']:.2f}s, "
                  f"HMM: {training_times['HMM']:.2f}s")
    
    @pytest.mark.asyncio
    async def test_prediction_latency_benchmarks(self, validation_data):
        """Test prediction latency meets performance requirements."""
        train_features, train_targets, val_features, _ = validation_data
        
        predictors = [
            ("XGBoost", XGBoostPredictor(room_id="test_room")),
            ("HMM", HMMPredictor(room_id="test_room")),
        ]
        
        for name, predictor in predictors:
            # Train model
            await predictor.train(train_features.head(200), train_targets.head(200))
            
            # Benchmark prediction time
            prediction_features = val_features.head(10)
            
            start_time = time.time()
            predictions = await predictor.predict(
                prediction_features, datetime.utcnow(), "vacant"
            )
            elapsed_time = time.time() - start_time
            
            # Should generate predictions quickly (< 100ms per sample as per requirements)
            latency_per_sample = elapsed_time / len(prediction_features) * 1000  # ms
            
            print(f"{name} prediction latency: {latency_per_sample:.2f}ms per sample")
            assert latency_per_sample < 100  # Less than 100ms per prediction
            assert len(predictions) == len(prediction_features)
    
    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, validation_data):
        """Test memory usage doesn't grow excessively during training."""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        train_features, train_targets, _, _ = validation_data
        
        # Train multiple models
        for i in range(3):
            predictor = XGBoostPredictor(room_id=f"room_{i}")
            await predictor.train(
                train_features.head(300), train_targets.head(300)
            )
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage - Initial: {initial_memory:.2f}MB, "
              f"Final: {final_memory:.2f}MB, Increase: {memory_increase:.2f}MB")
        
        # Memory increase should be reasonable (< 500MB for this test)
        assert memory_increase < 500