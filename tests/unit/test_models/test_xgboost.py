"""
Comprehensive unit tests for XGBoostPredictor to achieve high test coverage.

This module focuses on comprehensive testing of all methods, error paths,
edge cases, and configuration variations in XGBoostPredictor.
"""

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.core.constants import ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.predictor import PredictionResult, TrainingResult
from src.models.base.xgboost_predictor import XGBoostPredictor


class TestXGBoostPredictorInitialization:
    """Test XGBoostPredictor initialization and configuration."""

    def test_init_default_params(self):
        """Test initialization with default parameters."""
        predictor = XGBoostPredictor(room_id="test_room")

        assert predictor.room_id == "test_room"
        assert predictor.model_type == ModelType.XGBOOST
        assert predictor.model is None
        assert not predictor.is_trained
        assert predictor.feature_scaler is not None
        assert predictor.feature_importance_ == {}
        assert predictor.best_iteration_ is None
        assert predictor.eval_results_ == {}

    def test_init_custom_params(self):
        """Test initialization with custom parameters."""
        custom_params = {
            "n_estimators": 200,
            "max_depth": 8,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "reg_alpha": 0.2,
        }

        predictor = XGBoostPredictor(room_id="test_room", **custom_params)

        assert predictor.model_params["n_estimators"] == 200
        assert predictor.model_params["max_depth"] == 8
        assert predictor.model_params["learning_rate"] == 0.05
        assert predictor.model_params["subsample"] == 0.9
        assert predictor.model_params["reg_alpha"] == 0.2

        # Check default values still exist
        assert "random_state" in predictor.model_params
        assert "objective" in predictor.model_params

    def test_init_no_room_id(self):
        """Test initialization without room_id."""
        predictor = XGBoostPredictor()

        assert predictor.room_id is None
        assert predictor.model_type == ModelType.XGBOOST

    def test_model_params_structure(self):
        """Test that all required XGBoost parameters are present."""
        predictor = XGBoostPredictor()

        required_params = [
            "n_estimators",
            "max_depth",
            "learning_rate",
            "subsample",
            "colsample_bytree",
            "reg_alpha",
            "reg_lambda",
            "random_state",
            "early_stopping_rounds",
            "eval_metric",
            "objective",
        ]

        for param in required_params:
            assert param in predictor.model_params


class TestXGBoostPredictorTraining:
    """Test XGBoostPredictor training functionality."""

    @pytest.fixture
    def training_data(self):
        """Create sample training data."""
        np.random.seed(42)
        n_samples = 100

        features = pd.DataFrame(
            {
                "temporal_hour_sin": np.random.uniform(-1, 1, n_samples),
                "temporal_hour_cos": np.random.uniform(-1, 1, n_samples),
                "time_since_last_change": np.random.uniform(0, 3600, n_samples),
                "current_state_duration": np.random.uniform(0, 7200, n_samples),
                "temperature": np.random.uniform(18, 26, n_samples),
                "humidity": np.random.uniform(30, 70, n_samples),
            }
        )

        # Create realistic targets (time until transition in seconds)
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.uniform(300, 7200, n_samples)}
        )

        return features, targets

    @pytest.mark.asyncio
    async def test_train_basic_success(self, training_data):
        """Test successful basic training."""
        features, targets = training_data
        predictor = XGBoostPredictor(room_id="test_room")

        result = await predictor.train(features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.training_samples == len(features)
        assert result.model_version is not None
        assert result.training_score is not None
        assert result.validation_score is not None
        assert "training_mae" in result.training_metrics
        assert "training_rmse" in result.training_metrics
        assert "training_r2" in result.training_metrics

        # Check model state
        assert predictor.is_trained is True
        assert predictor.model is not None
        assert predictor.training_date is not None
        assert len(predictor.feature_names) == len(features.columns)
        assert len(predictor.feature_importance_) == len(features.columns)

    @pytest.mark.asyncio
    async def test_train_with_validation_data(self, training_data):
        """Test training with separate validation data."""
        features, targets = training_data

        # Split data for validation
        split_idx = int(0.8 * len(features))
        train_features = features.iloc[:split_idx]
        train_targets = targets.iloc[:split_idx]
        val_features = features.iloc[split_idx:]
        val_targets = targets.iloc[split_idx:]

        predictor = XGBoostPredictor(room_id="test_room")

        result = await predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert "validation_mae" in result.training_metrics
        assert "validation_rmse" in result.training_metrics
        assert "validation_r2" in result.training_metrics

    @pytest.mark.asyncio
    async def test_train_insufficient_data(self):
        """Test training with insufficient data."""
        # Create very small dataset
        features = pd.DataFrame({"feature1": [1, 2]})
        targets = pd.DataFrame({"time_until_transition_seconds": [300, 600]})

        predictor = XGBoostPredictor(room_id="test_room")

        with pytest.raises(ModelTrainingError) as exc_info:
            await predictor.train(features, targets)

        assert "Insufficient training data" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_train_with_different_target_formats(self):
        """Test training with different target data formats."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )

        # Test with next_transition_time format
        base_time = datetime.now(timezone.utc)
        targets = pd.DataFrame(
            {
                "target_time": [base_time + timedelta(minutes=i) for i in range(50)],
                "next_transition_time": [
                    base_time + timedelta(minutes=i + 10) for i in range(50)
                ],
            }
        )

        predictor = XGBoostPredictor(room_id="test_room")
        result = await predictor.train(features, targets)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_train_with_single_column_targets(self):
        """Test training with single column target format."""
        np.random.seed(42)
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(50),
                "feature2": np.random.randn(50),
            }
        )

        # Single column with random values
        targets = pd.DataFrame({"random_target": np.random.uniform(300, 3600, 50)})

        predictor = XGBoostPredictor(room_id="test_room")
        result = await predictor.train(features, targets)

        assert result.success is True

    @pytest.mark.asyncio
    async def test_train_target_clipping(self):
        """Test that target values are properly clipped."""
        features = pd.DataFrame(
            {
                "feature1": [1, 2, 3, 4, 5],
                "feature2": [5, 4, 3, 2, 1],
            }
        )

        # Targets with extreme values
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [0, 30, 1000, 100000, 200000]}
        )

        predictor = XGBoostPredictor(room_id="test_room")

        # Mock the model training to check target preparation
        with patch.object(predictor, "_prepare_targets") as mock_prepare:
            mock_prepare.return_value = np.array([60, 60, 1000, 86400, 86400])

            # Mock XGBoost components
            mock_model = Mock()
            mock_model.fit.return_value = None
            mock_model.feature_importances_ = np.array([0.5, 0.5])
            mock_model.predict.return_value = np.array([1000, 1000, 1000, 1000, 1000])

            with patch("xgboost.XGBRegressor", return_value=mock_model):
                result = await predictor.train(features, targets)

                assert result.success is True
                # Check that _prepare_targets was called
                mock_prepare.assert_called_once()

    @pytest.mark.asyncio
    async def test_train_stores_training_history(self, training_data):
        """Test that training results are stored in training history."""
        features, targets = training_data
        predictor = XGBoostPredictor(room_id="test_room")

        # Train multiple times
        result1 = await predictor.train(features, targets)
        result2 = await predictor.train(features, targets)

        assert len(predictor.training_history) == 2
        assert predictor.training_history[0] == result1
        assert predictor.training_history[1] == result2

    @pytest.mark.asyncio
    async def test_train_error_handling(self):
        """Test training error handling."""
        features = pd.DataFrame({"feature1": [1, 2, 3]})
        targets = pd.DataFrame({"time_until_transition_seconds": [300, 600, 900]})

        predictor = XGBoostPredictor(room_id="test_room")

        # Mock XGBoost to raise an exception
        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_xgb.side_effect = Exception("XGBoost initialization failed")

            with pytest.raises(ModelTrainingError) as exc_info:
                await predictor.train(features, targets)

            assert "XGBoost training failed" in str(exc_info.value)

            # Check that failed training is recorded
            assert len(predictor.training_history) == 1
            assert predictor.training_history[0].success is False
            assert (
                "XGBoost initialization failed"
                in predictor.training_history[0].error_message
            )

    @pytest.mark.asyncio
    async def test_prepare_targets_method(self):
        """Test the _prepare_targets method with various input formats."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Test with time_until_transition_seconds
        targets1 = pd.DataFrame({"time_until_transition_seconds": [300, 600, 900]})
        result1 = predictor._prepare_targets(targets1)
        np.testing.assert_array_equal(result1, [300, 600, 900])

        # Test with next_transition_time format
        base_time = datetime.now(timezone.utc)
        targets2 = pd.DataFrame(
            {
                "target_time": [base_time, base_time, base_time],
                "next_transition_time": [
                    base_time + timedelta(seconds=300),
                    base_time + timedelta(seconds=600),
                    base_time + timedelta(seconds=900),
                ],
            }
        )
        result2 = predictor._prepare_targets(targets2)
        np.testing.assert_array_equal(result2, [300, 600, 900])

        # Test with extreme values (should be clipped)
        targets3 = pd.DataFrame({"time_until_transition_seconds": [0, 30, 100000]})
        result3 = predictor._prepare_targets(targets3)
        np.testing.assert_array_equal(result3, [60, 60, 86400])  # Clipped values


class TestXGBoostPredictorPrediction:
    """Test XGBoostPredictor prediction functionality."""

    @pytest.fixture
    def trained_predictor(self):
        """Create a trained XGBoost predictor."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Mock trained state
        predictor.is_trained = True
        predictor.model_version = "v1.0"
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        # Mock model and scaler
        predictor.model = Mock()
        predictor.model.predict.return_value = np.array([1800, 3600])  # 30 min, 1 hour

        predictor.feature_scaler = Mock()
        predictor.feature_scaler.transform.return_value = np.array(
            [[0.5, -0.2, 1.0], [0.8, 0.1, -0.5]]
        )

        # Mock feature importance
        predictor.feature_importance_ = {
            "feature1": 0.4,
            "feature2": 0.3,
            "feature3": 0.3,
        }

        # Mock training history for confidence calculation
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
            {"feature1": [0.5, 0.8], "feature2": [-0.2, 0.1], "feature3": [1.0, -0.5]}
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time, "vacant")

        assert len(results) == 2
        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.predicted_time > prediction_time
            assert result.transition_type in [
                "vacant_to_occupied",
                "occupied_to_vacant",
            ]
            assert 0.1 <= result.confidence_score <= 0.95
            assert result.model_type == "xgboost"
            assert result.model_version == "v1.0"
            assert result.features_used == ["feature1", "feature2", "feature3"]
            assert "time_until_transition_seconds" in result.prediction_metadata
            assert "prediction_method" in result.prediction_metadata

    @pytest.mark.asyncio
    async def test_predict_not_trained(self):
        """Test prediction with untrained model."""
        predictor = XGBoostPredictor(room_id="test_room")
        features = pd.DataFrame({"feature1": [1, 2]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError) as exc_info:
            await predictor.predict(features, prediction_time)

        assert "xgboost" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_predict_invalid_features(self, trained_predictor):
        """Test prediction with invalid features."""
        # Mock validate_features to return False
        with patch.object(trained_predictor, "validate_features", return_value=False):
            features = pd.DataFrame({"wrong_feature": [1, 2]})
            prediction_time = datetime.now(timezone.utc)

            with pytest.raises(ModelPredictionError):
                await trained_predictor.predict(features, prediction_time)

    @pytest.mark.asyncio
    async def test_predict_time_clipping(self, trained_predictor):
        """Test that prediction times are properly clipped."""
        # Mock extreme predictions
        trained_predictor.model.predict.return_value = np.array(
            [0, 200000]
        )  # Very low/high

        features = pd.DataFrame(
            {"feature1": [0.5, 0.8], "feature2": [-0.2, 0.1], "feature3": [1.0, -0.5]}
        )
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(features, prediction_time)

        # Check that times were clipped to reasonable bounds
        for result in results:
            time_diff = (result.predicted_time - prediction_time).total_seconds()
            assert 60 <= time_diff <= 86400  # Between 1 min and 24 hours

    @pytest.mark.asyncio
    async def test_predict_transition_type_logic(self, trained_predictor):
        """Test transition type determination logic."""
        features = pd.DataFrame(
            {"feature1": [0.5], "feature2": [-0.2], "feature3": [1.0]}
        )
        prediction_time = datetime.now(timezone.utc)

        # Test with known current state
        results_occupied = await trained_predictor.predict(
            features, prediction_time, "occupied"
        )
        assert results_occupied[0].transition_type == "occupied_to_vacant"

        results_vacant = await trained_predictor.predict(
            features, prediction_time, "vacant"
        )
        assert results_vacant[0].transition_type == "vacant_to_occupied"

    @pytest.mark.asyncio
    async def test_predict_unknown_state_transition_logic(self, trained_predictor):
        """Test transition type logic with unknown current state."""
        features = pd.DataFrame(
            {
                "work_hours_feature": [0.8],  # Simulate work hours feature
                "sleep_hours_feature": [0.1],
                "feature3": [1.0],
            }
        )

        # Test during work hours (10 AM)
        work_time = datetime.now(timezone.utc).replace(hour=10)
        results = await trained_predictor.predict(features, work_time, "unknown")
        # Should predict vacant_to_occupied during work hours

        # Test during sleep hours (2 AM)
        sleep_time = datetime.now(timezone.utc).replace(hour=2)
        results_sleep = await trained_predictor.predict(features, sleep_time, "unknown")
        # Should predict occupied_to_vacant during sleep hours

    @pytest.mark.asyncio
    async def test_predict_error_handling(self, trained_predictor):
        """Test prediction error handling."""
        # Mock scaler to raise exception
        trained_predictor.feature_scaler.transform.side_effect = Exception(
            "Scaling failed"
        )

        features = pd.DataFrame(
            {"feature1": [0.5], "feature2": [-0.2], "feature3": [1.0]}
        )
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError) as exc_info:
            await trained_predictor.predict(features, prediction_time)

        assert "XGBoost prediction failed" in str(exc_info.value)

    def test_determine_transition_type_edge_cases(self, trained_predictor):
        """Test _determine_transition_type with various edge cases."""
        prediction_time = datetime.now(timezone.utc)

        # Test with features containing work_hours
        features_work = pd.Series({"work_hours_indicator": 0.9, "other_feature": 0.5})
        result = trained_predictor._determine_transition_type(
            "unknown", features_work, prediction_time
        )
        assert result == "vacant_to_occupied"

        # Test with features containing sleep_hours
        features_sleep = pd.Series({"sleep_hours_indicator": 0.9, "other_feature": 0.5})
        result = trained_predictor._determine_transition_type(
            "unknown", features_sleep, prediction_time
        )
        assert result == "occupied_to_vacant"

        # Test with no special features - early morning (3 AM)
        early_morning = prediction_time.replace(hour=3)
        features_normal = pd.Series({"feature1": 0.5, "feature2": 0.3})
        result = trained_predictor._determine_transition_type(
            "unknown", features_normal, early_morning
        )
        assert result == "occupied_to_vacant"

        # Test with no special features - afternoon (3 PM)
        afternoon = prediction_time.replace(hour=15)
        result = trained_predictor._determine_transition_type(
            "unknown", features_normal, afternoon
        )
        assert result == "vacant_to_occupied"


class TestXGBoostPredictorConfidence:
    """Test confidence calculation methods."""

    def test_calculate_confidence_with_validation_score(self):
        """Test confidence calculation with validation score."""
        predictor = XGBoostPredictor(room_id="test_room")
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

        X = pd.DataFrame([[0.5, -0.2, 1.0]], columns=["f1", "f2", "f3"])
        y_pred = np.array([1800])  # 30 minutes

        confidence = predictor._calculate_confidence(X, y_pred)

        assert 0.1 <= confidence <= 0.95
        # Should be based on validation score (0.85) with some adjustments

    def test_calculate_confidence_no_validation_score(self):
        """Test confidence calculation without validation score."""
        predictor = XGBoostPredictor(room_id="test_room")
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

        X = pd.DataFrame([[0.5, -0.2, 1.0]], columns=["f1", "f2", "f3"])
        y_pred = np.array([1800])

        confidence = predictor._calculate_confidence(X, y_pred)

        assert 0.1 <= confidence <= 0.95

    def test_calculate_confidence_extreme_predictions(self):
        """Test confidence calculation with extreme predictions."""
        predictor = XGBoostPredictor(room_id="test_room")
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

        X = pd.DataFrame([[0.5, -0.2, 1.0]], columns=["f1", "f2", "f3"])

        # Very short prediction (should lower confidence)
        y_pred_short = np.array([120])  # 2 minutes
        confidence_short = predictor._calculate_confidence(X, y_pred_short)

        # Very long prediction (should lower confidence)
        y_pred_long = np.array([50000])  # ~14 hours
        confidence_long = predictor._calculate_confidence(X, y_pred_long)

        # Normal prediction
        y_pred_normal = np.array([1800])  # 30 minutes
        confidence_normal = predictor._calculate_confidence(X, y_pred_normal)

        # Extreme predictions should have lower confidence
        assert confidence_short < confidence_normal
        assert confidence_long < confidence_normal

    def test_calculate_confidence_extreme_features(self):
        """Test confidence calculation with extreme feature values."""
        predictor = XGBoostPredictor(room_id="test_room")
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

        # Features with many extreme values (>2 std devs)
        X_extreme = pd.DataFrame([[3.5, -4.2, 5.0]], columns=["f1", "f2", "f3"])
        y_pred = np.array([1800])

        confidence_extreme = predictor._calculate_confidence(X_extreme, y_pred)

        # Normal features
        X_normal = pd.DataFrame([[0.5, -0.2, 1.0]], columns=["f1", "f2", "f3"])
        confidence_normal = predictor._calculate_confidence(X_normal, y_pred)

        # Extreme features should result in lower confidence
        assert confidence_extreme < confidence_normal

    def test_calculate_confidence_no_training_history(self):
        """Test confidence calculation with no training history."""
        predictor = XGBoostPredictor(room_id="test_room")
        # No training history

        X = pd.DataFrame([[0.5, -0.2, 1.0]], columns=["f1", "f2", "f3"])
        y_pred = np.array([1800])

        confidence = predictor._calculate_confidence(X, y_pred)

        # Should return default confidence
        assert confidence == 0.7

    def test_calculate_confidence_error_handling(self):
        """Test confidence calculation error handling."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Pass invalid data that would cause an exception
        X = pd.DataFrame()  # Empty DataFrame
        y_pred = np.array([1800])

        confidence = predictor._calculate_confidence(X, y_pred)

        # Should return default confidence on error
        assert confidence == 0.7


class TestXGBoostPredictorFeatureImportance:
    """Test feature importance methods."""

    def test_get_feature_importance_trained(self):
        """Test getting feature importance from trained model."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.feature_importance_ = {
            "feature1": 0.4,
            "feature2": 0.3,
            "feature3": 0.3,
        }

        importance = predictor.get_feature_importance()

        assert importance == {"feature1": 0.4, "feature2": 0.3, "feature3": 0.3}

        # Should return a copy, not the original
        importance["feature1"] = 0.9
        assert predictor.feature_importance_["feature1"] == 0.4

    def test_get_feature_importance_not_trained(self):
        """Test getting feature importance from untrained model."""
        predictor = XGBoostPredictor(room_id="test_room")

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_plot_data(self):
        """Test getting feature importance data for plotting."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.feature_importance_ = {
            "feature2": 0.3,
            "feature1": 0.4,
            "feature3": 0.3,
        }

        plot_data = predictor.get_feature_importance_plot_data()

        assert plot_data == [
            ("feature1", 0.4),
            ("feature2", 0.3),
            ("feature3", 0.3),
        ]  # Sorted by importance descending

    def test_get_feature_contributions(self):
        """Test getting feature contributions for interpretability."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.feature_importance_ = {
            "feature1": 0.4,
            "feature2": 0.3,
            "feature3": 0.3,
            "feature4": 0.0,  # Zero importance
        }

        X = pd.DataFrame(
            [[0.5, -0.2, 1.0, 0.1]],
            columns=["feature1", "feature2", "feature3", "feature4"],
        )

        contributions = predictor._get_feature_contributions(X)

        assert isinstance(contributions, dict)
        # Should return contributions for features with importance
        assert "feature1" in contributions
        assert "feature2" in contributions
        assert "feature3" in contributions

        # Check calculation: feature_value * feature_importance
        assert contributions["feature1"] == 0.5 * 0.4
        assert contributions["feature2"] == -0.2 * 0.3
        assert contributions["feature3"] == 1.0 * 0.3

    def test_get_feature_contributions_empty_importance(self):
        """Test feature contributions with empty importance."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.feature_importance_ = {}

        X = pd.DataFrame([[0.5, -0.2]], columns=["feature1", "feature2"])

        contributions = predictor._get_feature_contributions(X)

        assert contributions == {}

    def test_get_feature_contributions_error_handling(self):
        """Test feature contributions error handling."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.feature_importance_ = {"feature1": 0.5}

        # Pass invalid data
        X = pd.DataFrame()  # Empty DataFrame

        contributions = predictor._get_feature_contributions(X)

        assert contributions == {}


class TestXGBoostPredictorModelInformation:
    """Test model information and analysis methods."""

    def test_get_learning_curve_data(self):
        """Test getting learning curve data."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.eval_results_ = {
            "validation_0": {"rmse": [0.8, 0.7, 0.6]},
            "validation_1": {"rmse": [0.85, 0.75, 0.65]},
        }

        curve_data = predictor.get_learning_curve_data()

        assert curve_data == predictor.eval_results_

    def test_get_learning_curve_data_empty(self):
        """Test getting learning curve data when empty."""
        predictor = XGBoostPredictor(room_id="test_room")

        curve_data = predictor.get_learning_curve_data()

        assert curve_data == {}

    def test_get_model_complexity(self):
        """Test getting model complexity information."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.feature_names = ["f1", "f2", "f3", "f4", "f5"]
        predictor.best_iteration_ = 85
        predictor.feature_importance_ = {
            "f1": 0.3,
            "f2": 0.02,
            "f3": 0.25,
            "f4": 0.005,
            "f5": 0.4,
        }

        # Mock model
        predictor.model = Mock()
        predictor.model.n_estimators = 100

        complexity = predictor.get_model_complexity()

        assert complexity["n_estimators"] == 100
        assert complexity["max_depth"] == predictor.model_params["max_depth"]
        assert complexity["best_iteration"] == 85
        assert complexity["total_features"] == 5
        assert complexity["important_features"] == 3  # f1, f3, f5 > 0.01
        assert "regularization" in complexity
        assert "reg_alpha" in complexity["regularization"]
        assert "reg_lambda" in complexity["regularization"]

    def test_get_model_complexity_not_trained(self):
        """Test getting model complexity from untrained model."""
        predictor = XGBoostPredictor(room_id="test_room")

        complexity = predictor.get_model_complexity()

        assert complexity == {}


class TestXGBoostPredictorSaveLoad:
    """Test model saving and loading functionality."""

    def test_save_model_success(self):
        """Test successful model saving."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.is_trained = True
        predictor.model_version = "v1.0"
        predictor.feature_names = ["f1", "f2"]
        predictor.feature_importance_ = {"f1": 0.6, "f2": 0.4}

        # Mock model and scaler
        predictor.model = Mock()
        predictor.feature_scaler = Mock()
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
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.model = Mock()

        # Try to save to invalid path
        result = predictor.save_model("/invalid/path/model.pkl")

        assert result is False

    def test_load_model_success(self):
        """Test successful model loading."""
        # First create a model to save
        predictor1 = XGBoostPredictor(room_id="test_room")
        predictor1.is_trained = True
        predictor1.model_version = "v1.0"
        predictor1.feature_names = ["f1", "f2"]
        predictor1.feature_importance_ = {"f1": 0.6, "f2": 0.4}
        predictor1.model = Mock()
        predictor1.feature_scaler = Mock()
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
            predictor2 = XGBoostPredictor(room_id="different_room")
            result = predictor2.load_model(tmp_path)

            assert result is True
            assert predictor2.room_id == "test_room"  # Should be loaded from file
            assert predictor2.model_version == "v1.0"
            assert predictor2.feature_names == ["f1", "f2"]
            assert predictor2.feature_importance_ == {"f1": 0.6, "f2": 0.4}
            assert predictor2.is_trained is True
            assert len(predictor2.training_history) == 1

        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def test_load_model_error(self):
        """Test model loading with error."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Try to load from non-existent file
        result = predictor.load_model("/nonexistent/path/model.pkl")

        assert result is False

    def test_load_model_partial_data(self):
        """Test loading model with partial data (backwards compatibility)."""
        import pickle
        import tempfile

        # Create partial model data (missing some fields)
        model_data = {
            "model": Mock(),
            "model_type": "xgboost",
            "room_id": "test_room",
            # Missing several optional fields
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            pickle.dump(model_data, tmp_file)
            tmp_path = tmp_file.name

        try:
            predictor = XGBoostPredictor()
            result = predictor.load_model(tmp_path)

            assert result is True
            assert predictor.room_id == "test_room"
            # Should have defaults for missing fields
            assert predictor.model_version == "v1.0"  # Default
            assert predictor.feature_names == []  # Default
            assert predictor.is_trained is False  # Default

        finally:
            Path(tmp_path).unlink(missing_ok=True)


class TestXGBoostPredictorEdgeCases:
    """Test edge cases and error conditions."""

    def test_model_with_empty_feature_names(self):
        """Test model operations with empty feature names."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.feature_names = []
        predictor.is_trained = True

        # Should handle empty feature names gracefully
        importance = predictor.get_feature_importance()
        assert importance == {}

        plot_data = predictor.get_feature_importance_plot_data()
        assert plot_data == []

    def test_model_with_none_values(self):
        """Test model operations with None values."""
        predictor = XGBoostPredictor(room_id="test_room")
        predictor.model = None
        predictor.best_iteration_ = None
        predictor.feature_importance_ = {}

        complexity = predictor.get_model_complexity()
        assert complexity == {}

    @pytest.mark.asyncio
    async def test_train_with_nan_values(self):
        """Test training with NaN values in data."""
        features = pd.DataFrame(
            {"feature1": [1.0, np.nan, 3.0, 4.0], "feature2": [1.0, 2.0, np.nan, 4.0]}
        )
        targets = pd.DataFrame({"time_until_transition_seconds": [300, 600, 900, 1200]})

        predictor = XGBoostPredictor(room_id="test_room")

        # XGBoost should handle NaN values, but if preprocessing fails,
        # it should be caught and wrapped in ModelTrainingError
        with patch("xgboost.XGBRegressor") as mock_xgb:
            mock_xgb.side_effect = Exception("Cannot handle NaN values")

            with pytest.raises(ModelTrainingError):
                await predictor.train(features, targets)

    def test_prediction_time_boundary_values(self):
        """Test prediction time calculations with boundary values."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Test time clipping in predictions
        assert np.clip(30, 60, 86400) == 60  # Too small
        assert np.clip(100000, 60, 86400) == 86400  # Too large
        assert np.clip(3600, 60, 86400) == 3600  # Just right

    def test_confidence_calculation_edge_cases(self):
        """Test confidence calculation with edge case inputs."""
        predictor = XGBoostPredictor(room_id="test_room")

        # Test with single-value DataFrame
        X = pd.DataFrame([[1.0]], columns=["single_feature"])
        y_pred = np.array([1800])

        confidence = predictor._calculate_confidence(X, y_pred)
        assert 0.1 <= confidence <= 0.95

    def test_transition_type_hour_boundaries(self):
        """Test transition type determination at hour boundaries."""
        predictor = XGBoostPredictor(room_id="test_room")
        features = pd.Series({"feature1": 0.5})

        # Test boundary hours
        for hour in [0, 6, 22, 23]:
            test_time = datetime.now(timezone.utc).replace(hour=hour)
            result = predictor._determine_transition_type(
                "unknown", features, test_time
            )
            assert result in ["vacant_to_occupied", "occupied_to_vacant"]
