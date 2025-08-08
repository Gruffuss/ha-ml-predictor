"""
Sprint 3 Validation Tests - Model Development & Training

This module contains comprehensive validation tests to ensure all Sprint 3
model components are working correctly before proceeding to Sprint 4.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

logger = logging.getLogger(__name__)


# Test that all core imports work
def test_sprint3_imports():
    """Test that all Sprint 3 components can be imported successfully."""

    # Base model components
    from src.models.base.hmm_predictor import HMMPredictor
    from src.models.base.lstm_predictor import LSTMPredictor
    from src.models.base.predictor import (
        BasePredictor,
        PredictionResult,
        TrainingResult,
    )
    from src.models.base.xgboost_predictor import XGBoostPredictor
    from src.models.ensemble import OccupancyEnsemble

    # All imports successful
    assert True


def test_sprint3_base_predictor_interface():
    """Test that the base predictor interface is properly structured."""
    from src.core.constants import ModelType
    from src.models.base.predictor import (
        BasePredictor,
        PredictionResult,
        TrainingResult,
    )

    # Test PredictionResult
    pred_result = PredictionResult(
        predicted_time=datetime.utcnow() + timedelta(minutes=30),
        transition_type="vacant_to_occupied",
        confidence_score=0.85,
    )

    assert pred_result.predicted_time is not None
    assert pred_result.transition_type in [
        "vacant_to_occupied",
        "occupied_to_vacant",
    ]
    assert 0.0 <= pred_result.confidence_score <= 1.0

    # Test serialization
    pred_dict = pred_result.to_dict()
    assert isinstance(pred_dict, dict)
    assert "predicted_time" in pred_dict
    assert "confidence_score" in pred_dict

    # Test TrainingResult
    training_result = TrainingResult(
        success=True,
        training_time_seconds=120.5,
        model_version="v1.0",
        training_samples=1000,
        validation_score=0.85,
    )

    assert training_result.success is True
    assert training_result.training_time_seconds > 0
    assert training_result.training_samples > 0

    # Test serialization
    training_dict = training_result.to_dict()
    assert isinstance(training_dict, dict)
    assert "success" in training_dict


def test_sprint3_lstm_predictor_structure():
    """Test that the LSTM predictor is properly structured."""
    from src.core.constants import ModelType
    from src.models.base.lstm_predictor import LSTMPredictor

    # Test predictor initialization
    predictor = LSTMPredictor(room_id="test_room")
    assert predictor.model_type == ModelType.LSTM
    assert predictor.room_id == "test_room"
    assert predictor.is_trained is False
    assert isinstance(predictor.model_params, dict)
    assert "sequence_length" in predictor.model_params
    assert "hidden_layers" in predictor.model_params

    # Test model info
    info = predictor.get_model_info()
    assert isinstance(info, dict)
    assert info["model_type"] == "lstm"
    assert info["room_id"] == "test_room"
    assert info["is_trained"] == False


def test_sprint3_xgboost_predictor_structure():
    """Test that the XGBoost predictor is properly structured."""
    from src.core.constants import ModelType
    from src.models.base.xgboost_predictor import XGBoostPredictor

    # Test predictor initialization
    predictor = XGBoostPredictor(room_id="test_room")
    assert predictor.model_type == ModelType.XGBOOST
    assert predictor.room_id == "test_room"
    assert predictor.is_trained is False
    assert isinstance(predictor.model_params, dict)
    assert "n_estimators" in predictor.model_params
    assert "max_depth" in predictor.model_params

    # Test feature importance (empty before training)
    importance = predictor.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == 0  # Should be empty before training


def test_sprint3_hmm_predictor_structure():
    """Test that the HMM predictor is properly structured."""
    from src.core.constants import ModelType
    from src.models.base.hmm_predictor import HMMPredictor

    # Test predictor initialization
    predictor = HMMPredictor(room_id="test_room")
    assert predictor.model_type == ModelType.HMM
    assert predictor.room_id == "test_room"
    assert predictor.is_trained is False
    assert isinstance(predictor.model_params, dict)
    assert "n_components" in predictor.model_params
    assert "covariance_type" in predictor.model_params


def test_sprint3_ensemble_structure():
    """Test that the ensemble predictor is properly structured."""
    from src.core.constants import ModelType
    from src.models.ensemble import OccupancyEnsemble

    # Test ensemble initialization
    ensemble = OccupancyEnsemble(room_id="test_room")
    assert ensemble.model_type == ModelType.ENSEMBLE
    assert ensemble.room_id == "test_room"
    assert ensemble.is_trained is False
    assert isinstance(ensemble.model_params, dict)
    assert "meta_learner" in ensemble.model_params

    # Test base models are initialized
    assert "lstm" in ensemble.base_models
    assert "xgboost" in ensemble.base_models
    assert "hmm" in ensemble.base_models

    # Test ensemble info
    info = ensemble.get_ensemble_info()
    assert isinstance(info, dict)
    assert "base_models" in info
    assert "meta_learner" in info
    assert "is_trained" in info


@pytest.mark.asyncio
async def test_sprint3_model_training_basic():
    """Test basic model training with mock data."""
    from src.models.base.xgboost_predictor import XGBoostPredictor

    # Create predictor
    predictor = XGBoostPredictor(room_id="test_room")

    # Create mock training data
    n_samples = 100
    n_features = 20

    # Generate features with some temporal patterns
    np.random.seed(42)
    features_data = np.random.randn(n_samples, n_features)
    feature_names = [f"feature_{i}" for i in range(n_features)]

    features = pd.DataFrame(features_data, columns=feature_names)

    # Generate realistic targets (time until next transition in seconds)
    # Add some correlation with features to make training meaningful
    base_time = (
        1800 + features.iloc[:, 0] * 600
    )  # 30 min +/- 10 min based on first feature
    targets_data = np.clip(
        base_time + np.random.normal(0, 300, n_samples), 60, 86400
    )

    targets = pd.DataFrame({"time_until_transition_seconds": targets_data})

    # Train the model
    result = await predictor.train(features, targets)

    # Validate training result
    assert result.success is True
    assert result.training_time_seconds > 0
    assert result.training_samples == n_samples
    assert result.training_score is not None
    assert (
        result.training_score >= 0
    )  # R² can be negative, but should be reasonable

    # Check model state
    assert predictor.is_trained is True
    assert predictor.training_date is not None
    assert len(predictor.feature_names) == n_features

    # Test feature importance
    importance = predictor.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) == n_features
    assert all(isinstance(v, (int, float)) for v in importance.values())


@pytest.mark.asyncio
async def test_sprint3_model_prediction_basic():
    """Test basic model prediction with trained model."""
    from src.models.base.xgboost_predictor import XGBoostPredictor

    # Create and train predictor (similar to training test)
    predictor = XGBoostPredictor(room_id="test_room")

    # Mock training data
    n_samples = 100
    n_features = 15

    np.random.seed(42)
    train_features = pd.DataFrame(
        np.random.randn(n_samples, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    train_targets = pd.DataFrame(
        {
            "time_until_transition_seconds": np.clip(
                1800
                + train_features.iloc[:, 0] * 600
                + np.random.normal(0, 300, n_samples),
                60,
                86400,
            )
        }
    )

    # Train
    await predictor.train(train_features, train_targets)

    # Create prediction data
    pred_features = pd.DataFrame(
        np.random.randn(5, n_features),
        columns=[f"feature_{i}" for i in range(n_features)],
    )

    prediction_time = datetime.utcnow()

    # Make predictions
    predictions = await predictor.predict(
        pred_features, prediction_time, "vacant"
    )

    # Validate predictions
    assert isinstance(predictions, list)
    assert len(predictions) == 5

    for pred in predictions:
        assert isinstance(pred.predicted_time, datetime)
        assert pred.predicted_time > prediction_time
        assert pred.transition_type in [
            "vacant_to_occupied",
            "occupied_to_vacant",
        ]
        assert 0.0 <= pred.confidence_score <= 1.0
        assert pred.model_type == "xgboost"
        assert pred.model_version is not None
        assert isinstance(pred.prediction_metadata, dict)


@pytest.mark.asyncio
async def test_sprint3_lstm_sequence_handling():
    """Test LSTM-specific sequence handling."""
    from src.models.base.lstm_predictor import LSTMPredictor

    predictor = LSTMPredictor(room_id="test_room")

    # Create sequential training data (more samples for sequence generation)
    n_samples = 200  # Need more samples for LSTM sequences
    n_features = 10

    np.random.seed(42)

    # Generate time-series-like features
    features_data = []
    for i in range(n_samples):
        # Create some temporal correlation
        if i == 0:
            sample = np.random.randn(n_features)
        else:
            # Each sample is correlated with previous sample
            sample = 0.7 * features_data[-1] + 0.3 * np.random.randn(
                n_features
            )
        features_data.append(sample)

    features = pd.DataFrame(
        features_data, columns=[f"seq_feature_{i}" for i in range(n_features)]
    )

    # Generate targets with some temporal pattern
    targets_data = []
    for i in range(n_samples):
        if i < 10:
            base_time = 1800
        else:
            # Time influenced by recent features
            recent_avg = np.mean(
                [features_data[j][0] for j in range(max(0, i - 10), i)]
            )
            base_time = 1800 + recent_avg * 300

        target = np.clip(base_time + np.random.normal(0, 200), 60, 86400)
        targets_data.append(target)

    targets = pd.DataFrame({"time_until_transition_seconds": targets_data})

    # Train LSTM
    result = await predictor.train(features, targets)

    # Should succeed even with sequence generation
    assert result.success is True
    assert predictor.is_trained is True

    # Test sequence-specific metrics
    assert "sequences_generated" in result.training_metrics
    assert result.training_metrics["sequences_generated"] > 0
    assert "sequence_length" in result.training_metrics

    # Test prediction
    pred_features = features.tail(
        predictor.sequence_length + 5
    )  # Ensure enough for sequence
    predictions = await predictor.predict(
        pred_features, datetime.utcnow(), "occupied"
    )

    assert len(predictions) > 0
    assert all(pred.model_type == "lstm" for pred in predictions)


@pytest.mark.asyncio
async def test_sprint3_hmm_state_identification():
    """Test HMM-specific state identification."""
    from src.models.base.hmm_predictor import HMMPredictor

    predictor = HMMPredictor(room_id="test_room")

    # Create data with distinct state patterns
    n_samples = 150
    n_features = 8

    np.random.seed(42)
    features_data = []
    targets_data = []

    # Generate data with 3 distinct states
    for i in range(n_samples):
        state = i % 3

        if state == 0:  # Short stay state
            features = np.random.normal(
                [1, -1, 0, 0, 0, 0, 0, 0], 0.5, n_features
            )
            duration = np.random.normal(600, 100)  # ~10 minutes
        elif state == 1:  # Medium stay state
            features = np.random.normal(
                [-1, 1, 1, 0, 0, 0, 0, 0], 0.5, n_features
            )
            duration = np.random.normal(3600, 300)  # ~1 hour
        else:  # Long stay state
            features = np.random.normal(
                [0, 0, -1, 1, 1, 0, 0, 0], 0.5, n_features
            )
            duration = np.random.normal(7200, 600)  # ~2 hours

        features_data.append(features)
        targets_data.append(np.clip(duration, 60, 86400))

    features = pd.DataFrame(
        features_data,
        columns=[f"state_feature_{i}" for i in range(n_features)],
    )
    targets = pd.DataFrame({"time_until_transition_seconds": targets_data})

    # Train HMM
    result = await predictor.train(features, targets)

    # Should identify states successfully
    assert result.success is True
    assert predictor.is_trained is True

    # Check state-specific metrics
    assert "n_states" in result.training_metrics
    assert (
        result.training_metrics["n_states"]
        == predictor.model_params["n_components"]
    )
    assert "state_distribution" in result.training_metrics

    # Test state info
    state_info = predictor.get_state_info()
    assert isinstance(state_info, dict)
    assert "state_labels" in state_info
    assert "state_characteristics" in state_info

    # Test prediction with state information
    pred_features = features.head(10)
    predictions = await predictor.predict(
        pred_features, datetime.utcnow(), "unknown"
    )

    assert len(predictions) == 10
    for pred in predictions:
        assert pred.model_type == "hmm"
        assert "current_hidden_state" in pred.prediction_metadata
        assert "state_probability" in pred.prediction_metadata


@pytest.mark.asyncio
async def test_sprint3_ensemble_training():
    """Test ensemble training with base models."""
    from src.models.ensemble import OccupancyEnsemble

    ensemble = OccupancyEnsemble(room_id="test_room")

    # Create comprehensive training data for ensemble
    n_samples = 200  # More samples for ensemble training
    n_features = 25

    np.random.seed(42)

    # Generate features with mixed patterns for different models
    features_data = np.random.randn(n_samples, n_features)

    # Add some temporal correlation for LSTM
    for i in range(1, n_samples):
        features_data[i, :5] = (
            0.8 * features_data[i - 1, :5] + 0.2 * features_data[i, :5]
        )

    # Add some categorical-like features for XGBoost
    features_data[:, 15:20] = np.round(features_data[:, 15:20])

    features = pd.DataFrame(
        features_data,
        columns=[f"ensemble_feature_{i}" for i in range(n_features)],
    )

    # Generate targets with complex patterns
    targets_data = []
    for i in range(n_samples):
        # Complex relationship involving multiple features
        base_time = 1800  # 30 minutes
        base_time += features_data[i, 0] * 400  # Linear component
        base_time += (
            features_data[i, 1] * features_data[i, 2] * 200
        )  # Interaction
        base_time += np.sin(i * 0.1) * 300  # Temporal pattern

        target = np.clip(base_time + np.random.normal(0, 200), 60, 86400)
        targets_data.append(target)

    targets = pd.DataFrame({"time_until_transition_seconds": targets_data})

    # Split into train/validation
    split_idx = int(0.8 * n_samples)
    train_features = features.iloc[:split_idx]
    train_targets = targets.iloc[:split_idx]
    val_features = features.iloc[split_idx:]
    val_targets = targets.iloc[split_idx:]

    # Train ensemble
    result = await ensemble.train(
        train_features, train_targets, val_features, val_targets
    )

    # Validate ensemble training
    assert result.success is True
    assert ensemble.is_trained is True
    assert ensemble.base_models_trained is True
    assert ensemble.meta_learner_trained is True

    # Check ensemble-specific metrics
    assert "base_model_count" in result.training_metrics
    assert result.training_metrics["base_model_count"] == 3
    assert "model_weights" in result.training_metrics
    assert "base_model_performance" in result.training_metrics

    # Test model weights
    assert isinstance(ensemble.model_weights, dict)
    assert len(ensemble.model_weights) > 0
    assert all(isinstance(w, float) for w in ensemble.model_weights.values())

    # Test ensemble info
    info = ensemble.get_ensemble_info()
    assert info["ensemble_type"] == "stacking"
    assert info["is_trained"] == True
    assert len(info["base_models"]) == 3

    # Test ensemble prediction
    predictions = await ensemble.predict(
        val_features.head(5), datetime.utcnow(), "vacant"
    )

    assert len(predictions) == 5
    for pred in predictions:
        assert pred.model_type == "ensemble"
        assert "base_model_predictions" in pred.prediction_metadata
        assert "model_weights" in pred.prediction_metadata
        assert (
            pred.alternatives is not None
        )  # Should have alternative predictions


def test_sprint3_model_serialization():
    """Test model saving and loading functionality."""
    import tempfile

    from src.models.base.xgboost_predictor import XGBoostPredictor

    # Create predictor
    predictor = XGBoostPredictor(room_id="test_room")
    predictor.is_trained = True  # Simulate trained state
    predictor.model_version = "v1.5"
    predictor.feature_names = ["feature_1", "feature_2", "feature_3"]

    # Test serialization
    with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
        success = predictor.save_model(tmp_file.name)
        assert success is True

        # Test deserialization
        new_predictor = XGBoostPredictor(room_id="other_room")
        load_success = new_predictor.load_model(tmp_file.name)

        assert load_success is True
        assert (
            new_predictor.room_id == "test_room"
        )  # Should be loaded from file
        assert new_predictor.model_version == "v1.5"
        assert new_predictor.feature_names == [
            "feature_1",
            "feature_2",
            "feature_3",
        ]
        assert new_predictor.is_trained is True

    # Cleanup
    import os

    os.unlink(tmp_file.name)


def test_sprint3_feature_validation():
    """Test feature validation functionality."""
    from src.models.base.xgboost_predictor import XGBoostPredictor

    predictor = XGBoostPredictor(room_id="test_room")
    predictor.is_trained = True
    predictor.feature_names = ["feature_a", "feature_b", "feature_c"]

    # Test valid features
    valid_features = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            "feature_c": [7, 8, 9],
        }
    )

    assert predictor.validate_features(valid_features) == True

    # Test missing features
    missing_features = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            # Missing feature_c
        }
    )

    assert predictor.validate_features(missing_features) == False

    # Test extra features (should warn but not fail)
    extra_features = pd.DataFrame(
        {
            "feature_a": [1, 2, 3],
            "feature_b": [4, 5, 6],
            "feature_c": [7, 8, 9],
            "feature_d": [10, 11, 12],  # Extra feature
        }
    )

    assert predictor.validate_features(extra_features) == True


def test_sprint3_prediction_result_validation():
    """Test prediction result validation and bounds."""
    from src.models.base.predictor import PredictionResult

    # Test valid prediction result
    now = datetime.utcnow()
    result = PredictionResult(
        predicted_time=now + timedelta(minutes=30),
        transition_type="vacant_to_occupied",
        confidence_score=0.85,
    )

    # Test serialization and deserialization
    result_dict = result.to_dict()
    assert isinstance(result_dict["predicted_time"], str)
    assert result_dict["transition_type"] == "vacant_to_occupied"
    assert result_dict["confidence_score"] == 0.85

    # Test with alternatives
    result_with_alts = PredictionResult(
        predicted_time=now + timedelta(minutes=45),
        transition_type="occupied_to_vacant",
        confidence_score=0.75,
        alternatives=[
            (now + timedelta(minutes=30), 0.65),
            (now + timedelta(minutes=60), 0.55),
        ],
    )

    alt_dict = result_with_alts.to_dict()
    assert "alternatives" in alt_dict
    assert len(alt_dict["alternatives"]) == 2


def test_sprint3_file_structure():
    """Test that all expected Sprint 3 files exist."""
    base_path = Path(__file__).parent.parent

    # Base model files
    assert (base_path / "src" / "models" / "__init__.py").exists()
    assert (base_path / "src" / "models" / "base" / "__init__.py").exists()
    assert (base_path / "src" / "models" / "base" / "predictor.py").exists()
    assert (
        base_path / "src" / "models" / "base" / "lstm_predictor.py"
    ).exists()
    assert (
        base_path / "src" / "models" / "base" / "xgboost_predictor.py"
    ).exists()
    assert (
        base_path / "src" / "models" / "base" / "hmm_predictor.py"
    ).exists()

    # Ensemble file
    assert (base_path / "src" / "models" / "ensemble.py").exists()

    # Test files
    assert (base_path / "tests" / "test_sprint3_validation.py").exists()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_sprint3_end_to_end_modeling_pipeline():
    """Test a complete end-to-end modeling pipeline."""
    from src.models.ensemble import OccupancyEnsemble

    # Create comprehensive test scenario
    ensemble = OccupancyEnsemble(room_id="integration_test_room")

    # Generate realistic training data
    n_samples = 300
    n_features = 30

    np.random.seed(123)

    # Create features that would come from the feature engineering pipeline
    feature_names = []
    features_data = []

    # Temporal features
    for i in range(10):
        feature_names.append(f"temporal_feature_{i}")

    # Sequential features
    for i in range(10):
        feature_names.append(f"sequential_feature_{i}")

    # Contextual features
    for i in range(10):
        feature_names.append(f"contextual_feature_{i}")

    # Generate feature data with realistic patterns
    for sample_idx in range(n_samples):
        sample_features = []

        # Temporal features (cyclical patterns)
        hour_of_day = (sample_idx * 0.5) % 24
        sample_features.extend(
            [
                np.sin(2 * np.pi * hour_of_day / 24),  # Hour sine
                np.cos(2 * np.pi * hour_of_day / 24),  # Hour cosine
                (
                    1.0 if (hour_of_day >= 22 or hour_of_day <= 6) else 0.0
                ),  # Sleep hours
                1.0 if sample_idx % 7 >= 5 else 0.0,  # Weekend
                np.random.exponential(1800),  # Time since last event
                np.random.gamma(2, 900),  # State duration
                np.random.normal(22, 3),  # Temperature
                np.random.beta(2, 2),  # Activity ratio
                np.random.uniform(0, 1),  # Confidence score
                np.random.normal(0, 1),  # Generic temporal
            ]
        )

        # Sequential features (movement patterns)
        sample_features.extend(
            [
                np.random.poisson(2),  # Room transitions
                np.random.uniform(0, 1),  # Movement velocity
                np.random.binomial(1, 0.7),  # Human vs cat
                np.random.gamma(1, 2),  # Sequence length
                np.random.exponential(300),  # Inter-event time
                np.random.beta(1, 3),  # Revisit ratio
                np.random.normal(0.5, 0.2),  # Pattern consistency
                np.random.uniform(0, 5),  # Sensor diversity
                np.random.exponential(600),  # Burst duration
                np.random.normal(0, 1),  # Generic sequential
            ]
        )

        # Contextual features (environmental)
        sample_features.extend(
            [
                np.random.normal(50, 15),  # Humidity
                np.random.exponential(500),  # Light level
                np.random.binomial(1, 0.3),  # Door open
                np.random.poisson(1),  # Other rooms active
                np.random.uniform(0, 1),  # Seasonal indicator
                np.random.normal(0.4, 0.2),  # Activity correlation
                np.random.gamma(2, 0.5),  # Environmental stability
                np.random.beta(3, 2),  # Context confidence
                np.random.uniform(0, 1),  # External factors
                np.random.normal(0, 1),  # Generic contextual
            ]
        )

        features_data.append(sample_features)

    features = pd.DataFrame(features_data, columns=feature_names)

    # Generate realistic targets with complex dependencies
    targets_data = []
    for i in range(n_samples):
        base_duration = 1800  # 30 minutes base

        # Dependencies on features
        base_duration += features.iloc[i, 0] * 600  # Hour sine effect
        base_duration += features.iloc[i, 2] * 3600  # Sleep hours effect
        base_duration += features.iloc[i, 10] * 300  # Room transitions effect
        base_duration += features.iloc[i, 6] * 50  # Temperature effect

        # Add noise
        duration = base_duration + np.random.normal(0, 300)
        duration = np.clip(duration, 60, 86400)  # 1 min to 24 hours

        targets_data.append(duration)

    targets = pd.DataFrame({"time_until_transition_seconds": targets_data})

    # Split data
    split_idx = int(0.75 * n_samples)
    train_features = features.iloc[:split_idx]
    train_targets = targets.iloc[:split_idx]
    val_features = features.iloc[split_idx:]
    val_targets = targets.iloc[split_idx:]

    # Train the complete ensemble
    logger.info("Training complete ensemble with realistic data")
    training_result = await ensemble.train(
        train_features, train_targets, val_features, val_targets
    )

    # Validate training success
    assert training_result.success is True
    assert ensemble.is_trained is True
    assert training_result.training_score > 0.0  # Should learn something

    # Test comprehensive prediction
    test_features = val_features.head(10)
    prediction_time = datetime.utcnow()

    predictions = await ensemble.predict(
        test_features, prediction_time, "vacant"
    )

    # Validate predictions
    assert len(predictions) == 10

    for i, pred in enumerate(predictions):
        # Basic validation
        assert isinstance(pred.predicted_time, datetime)
        assert pred.predicted_time > prediction_time
        assert pred.transition_type in [
            "vacant_to_occupied",
            "occupied_to_vacant",
        ]
        assert 0.0 <= pred.confidence_score <= 1.0

        # Ensemble-specific validation
        assert pred.model_type == "ensemble"
        assert pred.alternatives is not None
        assert len(pred.alternatives) >= 1

        # Metadata validation
        metadata = pred.prediction_metadata
        assert "base_model_predictions" in metadata
        assert "model_weights" in metadata
        assert "meta_learner_type" in metadata

        # Check that predictions are reasonable (between 1 min and 24 hours)
        time_until = (pred.predicted_time - prediction_time).total_seconds()
        assert 60 <= time_until <= 86400

    # Test feature importance
    importance = ensemble.get_feature_importance()
    assert isinstance(importance, dict)
    assert len(importance) > 0
    assert all(isinstance(v, (int, float)) for v in importance.values())

    # Test model performance tracking
    assert len(ensemble.training_history) == 1
    assert ensemble.get_prediction_accuracy() is not None

    # Test ensemble info
    ensemble_info = ensemble.get_ensemble_info()
    assert ensemble_info["is_trained"] == True
    assert ensemble_info["base_models_trained"] == True
    assert ensemble_info["meta_learner_trained"] == True
    assert len(ensemble_info["model_weights"]) > 0


@pytest.mark.smoke
def test_sprint3_smoke_test():
    """Smoke test to verify basic Sprint 3 functionality."""
    # This test should run very quickly and catch major issues

    # Test imports work
    from src.core.constants import ModelType
    from src.models.base.hmm_predictor import HMMPredictor
    from src.models.base.lstm_predictor import LSTMPredictor
    from src.models.base.predictor import BasePredictor, PredictionResult
    from src.models.base.xgboost_predictor import XGBoostPredictor
    from src.models.ensemble import OccupancyEnsemble

    # Test basic object creation
    lstm = LSTMPredictor("test")
    xgb = XGBoostPredictor("test")
    hmm = HMMPredictor("test")
    ensemble = OccupancyEnsemble("test")

    # Test that objects have expected attributes
    assert lstm.model_type == ModelType.LSTM
    assert xgb.model_type == ModelType.XGBOOST
    assert hmm.model_type == ModelType.HMM
    assert ensemble.model_type == ModelType.ENSEMBLE

    # Test basic methods exist
    assert hasattr(lstm, "train")
    assert hasattr(lstm, "predict")
    assert hasattr(xgb, "get_feature_importance")
    assert hasattr(ensemble, "get_ensemble_info")

    # Test prediction result creation
    pred_result = PredictionResult(
        predicted_time=datetime.utcnow() + timedelta(minutes=30),
        transition_type="vacant_to_occupied",
        confidence_score=0.8,
    )

    assert pred_result.confidence_score == 0.8
    assert pred_result.transition_type == "vacant_to_occupied"


if __name__ == "__main__":
    """
    Run Sprint 3 validation tests directly.

    Usage: python tests/test_sprint3_validation.py
    """
    pytest.main([__file__, "-v", "--tb=short"])
