"""
Comprehensive unit tests for base predictor interface.

This test suite validates the base predictor abstract class, data models,
common functionality, model management, and all shared predictor operations.
"""

from datetime import datetime, timedelta, timezone
import pickle
import tempfile
from unittest.mock import Mock, mock_open, patch

import numpy as np
import pandas as pd
import pytest

from src.core.constants import ModelType
from src.core.exceptions import ModelPredictionError
from src.models.base.predictor import (
    BasePredictor,
    PredictionResult,
    TrainingResult,
)


class TestPredictionResult:
    """Test PredictionResult data model."""

    def test_prediction_result_creation_minimal(self):
        """Test PredictionResult creation with minimal required fields."""
        predicted_time = datetime.now(timezone.utc) + timedelta(minutes=30)

        result = PredictionResult(
            predicted_time=predicted_time,
            transition_type="occupied_to_vacant",
            confidence_score=0.85,
        )

        assert result.predicted_time == predicted_time
        assert result.transition_type == "occupied_to_vacant"
        assert result.confidence_score == 0.85
        assert result.prediction_interval is None
        assert result.alternatives is None
        assert result.model_type is None
        assert result.model_version is None
        assert result.features_used is None
        assert result.prediction_metadata is None

    def test_prediction_result_creation_full(self):
        """Test PredictionResult creation with all fields."""
        predicted_time = datetime.now(timezone.utc) + timedelta(minutes=45)
        interval_start = predicted_time - timedelta(minutes=10)
        interval_end = predicted_time + timedelta(minutes=10)

        alternatives = [
            (predicted_time + timedelta(minutes=15), 0.7),
            (predicted_time + timedelta(minutes=25), 0.6),
        ]

        features_used = ["feature1", "feature2", "feature3"]
        metadata = {
            "time_until_transition_seconds": 2700,
            "prediction_method": "test_method",
            "model_confidence": 0.9,
        }

        result = PredictionResult(
            predicted_time=predicted_time,
            transition_type="vacant_to_occupied",
            confidence_score=0.92,
            prediction_interval=(interval_start, interval_end),
            alternatives=alternatives,
            model_type="test_model",
            model_version="v2.1",
            features_used=features_used,
            prediction_metadata=metadata,
        )

        assert result.predicted_time == predicted_time
        assert result.transition_type == "vacant_to_occupied"
        assert result.confidence_score == 0.92
        assert result.prediction_interval == (interval_start, interval_end)
        assert result.alternatives == alternatives
        assert result.model_type == "test_model"
        assert result.model_version == "v2.1"
        assert result.features_used == features_used
        assert result.prediction_metadata == metadata

    def test_prediction_result_to_dict(self):
        """Test PredictionResult to_dict serialization."""
        predicted_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)
        interval_start = predicted_time - timedelta(minutes=5)
        interval_end = predicted_time + timedelta(minutes=5)

        alternatives = [
            (predicted_time + timedelta(minutes=10), 0.75),
            (predicted_time + timedelta(minutes=20), 0.65),
        ]

        result = PredictionResult(
            predicted_time=predicted_time,
            transition_type="occupied_to_vacant",
            confidence_score=0.88,
            prediction_interval=(interval_start, interval_end),
            alternatives=alternatives,
            model_type="lstm",
            model_version="v1.5",
            features_used=["temp", "motion"],
            prediction_metadata={"method": "deep_learning"},
        )

        result_dict = result.to_dict()

        assert result_dict["predicted_time"] == predicted_time.isoformat()
        assert result_dict["transition_type"] == "occupied_to_vacant"
        assert result_dict["confidence_score"] == 0.88
        assert result_dict["prediction_interval"] == [
            interval_start.isoformat(),
            interval_end.isoformat(),
        ]
        assert len(result_dict["alternatives"]) == 2
        assert (
            result_dict["alternatives"][0]["time"]
            == (predicted_time + timedelta(minutes=10)).isoformat()
        )
        assert result_dict["alternatives"][0]["confidence"] == 0.75
        assert result_dict["model_type"] == "lstm"
        assert result_dict["model_version"] == "v1.5"
        assert result_dict["features_used"] == ["temp", "motion"]
        assert result_dict["prediction_metadata"] == {"method": "deep_learning"}

    def test_prediction_result_to_dict_minimal(self):
        """Test PredictionResult to_dict with minimal fields."""
        predicted_time = datetime(2024, 1, 15, 14, 30, 0, tzinfo=timezone.utc)

        result = PredictionResult(
            predicted_time=predicted_time,
            transition_type="vacant_to_occupied",
            confidence_score=0.75,
        )

        result_dict = result.to_dict()

        assert result_dict["predicted_time"] == predicted_time.isoformat()
        assert result_dict["transition_type"] == "vacant_to_occupied"
        assert result_dict["confidence_score"] == 0.75

        # Optional fields should not be present if None
        assert "prediction_interval" not in result_dict
        assert "alternatives" not in result_dict
        assert "model_type" not in result_dict
        assert "model_version" not in result_dict
        assert "features_used" not in result_dict
        assert "prediction_metadata" not in result_dict


class TestTrainingResult:
    """Test TrainingResult data model."""

    def test_training_result_creation_minimal(self):
        """Test TrainingResult creation with minimal required fields."""
        result = TrainingResult(
            success=True,
            training_time_seconds=45.7,
            model_version="v1.0",
            training_samples=1000,
        )

        assert result.success is True
        assert result.training_time_seconds == 45.7
        assert result.model_version == "v1.0"
        assert result.training_samples == 1000
        assert result.validation_score is None
        assert result.training_score is None
        assert result.feature_importance is None
        assert result.training_metrics is None
        assert result.error_message is None

    def test_training_result_creation_full(self):
        """Test TrainingResult creation with all fields."""
        feature_importance = {"feature1": 0.4, "feature2": 0.35, "feature3": 0.25}
        training_metrics = {
            "mae": 12.5,
            "rmse": 18.2,
            "r2": 0.87,
            "convergence_iter": 150,
        }

        result = TrainingResult(
            success=True,
            training_time_seconds=123.4,
            model_version="v2.3",
            training_samples=2500,
            validation_score=0.82,
            training_score=0.85,
            feature_importance=feature_importance,
            training_metrics=training_metrics,
            error_message=None,
        )

        assert result.success is True
        assert result.training_time_seconds == 123.4
        assert result.model_version == "v2.3"
        assert result.training_samples == 2500
        assert result.validation_score == 0.82
        assert result.training_score == 0.85
        assert result.feature_importance == feature_importance
        assert result.training_metrics == training_metrics
        assert result.error_message is None

    def test_training_result_creation_failure(self):
        """Test TrainingResult creation for failed training."""
        result = TrainingResult(
            success=False,
            training_time_seconds=15.2,
            model_version="v1.0",
            training_samples=0,
            error_message="Training failed due to insufficient data",
        )

        assert result.success is False
        assert result.training_time_seconds == 15.2
        assert result.model_version == "v1.0"
        assert result.training_samples == 0
        assert result.error_message == "Training failed due to insufficient data"

    def test_training_result_to_dict(self):
        """Test TrainingResult to_dict serialization."""
        feature_importance = {"temp": 0.5, "motion": 0.3, "time": 0.2}
        training_metrics = {"accuracy": 0.91, "loss": 0.15}

        result = TrainingResult(
            success=True,
            training_time_seconds=67.3,
            model_version="v1.8",
            training_samples=1500,
            validation_score=0.89,
            training_score=0.92,
            feature_importance=feature_importance,
            training_metrics=training_metrics,
        )

        result_dict = result.to_dict()

        assert result_dict["success"] is True
        assert result_dict["training_time_seconds"] == 67.3
        assert result_dict["model_version"] == "v1.8"
        assert result_dict["training_samples"] == 1500
        assert result_dict["validation_score"] == 0.89
        assert result_dict["training_score"] == 0.92
        assert result_dict["feature_importance"] == feature_importance
        assert result_dict["training_metrics"] == training_metrics
        assert result_dict["error_message"] is None


class TestConcretePredictor(BasePredictor):
    """Concrete implementation of BasePredictor for testing."""

    def __init__(self, model_type=ModelType.LSTM, room_id=None, config=None):
        super().__init__(model_type, room_id, config)
        self.mock_model_trained = False
        self.mock_predictions = []
        self.mock_feature_importance = {}

    async def train(
        self, features, targets, validation_features=None, validation_targets=None
    ):
        """Mock training implementation."""
        training_start = datetime.now(timezone.utc)

        # Simulate training
        if len(features) < 10:
            raise ValueError("Insufficient training data")

        self.feature_names = list(features.columns)
        self.is_trained = True
        self.mock_model_trained = True
        self.training_date = datetime.now(timezone.utc)
        self.model_version = self._generate_model_version()

        # Mock training metrics
        training_score = 0.85
        validation_score = 0.82 if validation_features is not None else training_score

        training_time = (datetime.now(timezone.utc) - training_start).total_seconds()

        result = TrainingResult(
            success=True,
            training_time_seconds=training_time,
            model_version=self.model_version,
            training_samples=len(features),
            validation_score=validation_score,
            training_score=training_score,
            training_metrics={"mae": 15.2, "rmse": 22.1},
        )

        self.training_history.append(result)
        return result

    async def predict(self, features, prediction_time, current_state="unknown"):
        """Mock prediction implementation."""
        if not self.is_trained:
            raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")

        if not self.validate_features(features):
            raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")

        predictions = []

        for i in range(len(features)):
            # Mock prediction
            predicted_time = prediction_time + timedelta(minutes=30 + i * 10)
            transition_type = (
                "occupied_to_vacant"
                if current_state == "occupied"
                else "vacant_to_occupied"
            )

            result = PredictionResult(
                predicted_time=predicted_time,
                transition_type=transition_type,
                confidence_score=0.8,
                model_type=self.model_type.value,
                model_version=self.model_version,
                features_used=self.feature_names,
                prediction_metadata={"mock": True},
            )

            predictions.append(result)
            self._record_prediction(prediction_time, result)

        return predictions

    def get_feature_importance(self):
        """Mock feature importance implementation."""
        if not self.is_trained:
            return {}

        # Mock equal importance for all features
        if self.feature_names:
            importance = 1.0 / len(self.feature_names)
            return {name: importance for name in self.feature_names}

        return self.mock_feature_importance


class TestBasePredictorInitialization:
    """Test BasePredictor initialization and basic properties."""

    def test_base_predictor_initialization_default(self):
        """Test BasePredictor initialization with default parameters."""
        predictor = TestConcretePredictor()

        assert predictor.model_type == ModelType.LSTM
        assert predictor.room_id is None
        assert predictor.config is None
        assert predictor.is_trained is False
        assert predictor.model_version == "v1.0"
        assert predictor.training_date is None
        assert predictor.feature_names == []
        assert predictor.training_history == []
        assert predictor.prediction_history == []
        assert predictor.model_params == {}
        assert predictor.model is None

    def test_base_predictor_initialization_with_params(self):
        """Test BasePredictor initialization with parameters."""
        mock_config = Mock()

        predictor = TestConcretePredictor(
            model_type=ModelType.XGB, room_id="living_room", config=mock_config
        )

        assert predictor.model_type == ModelType.XGB
        assert predictor.room_id == "living_room"
        assert predictor.config == mock_config


class TestBasePredictorTraining:
    """Test BasePredictor training functionality."""

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data fixture."""
        features = pd.DataFrame(
            {
                "feature1": np.random.randn(100),
                "feature2": np.random.randn(100),
                "feature3": np.random.randn(100),
            }
        )
        targets = pd.DataFrame(
            {"time_until_transition_seconds": np.random.exponential(1800, 100)}
        )
        return features, targets

    @pytest.mark.asyncio
    async def test_base_predictor_training_success(self, sample_training_data):
        """Test successful training."""
        predictor = TestConcretePredictor(room_id="test_room")
        features, targets = sample_training_data

        result = await predictor.train(features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert predictor.is_trained is True
        assert predictor.feature_names == ["feature1", "feature2", "feature3"]
        assert predictor.training_date is not None
        assert len(predictor.training_history) == 1
        assert predictor.training_history[0] == result

    @pytest.mark.asyncio
    async def test_base_predictor_training_with_validation(self, sample_training_data):
        """Test training with validation data."""
        predictor = TestConcretePredictor()
        features, targets = sample_training_data

        # Split for validation
        train_features = features[:80]
        train_targets = targets[:80]
        val_features = features[80:]
        val_targets = targets[80:]

        result = await predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert result.validation_score is not None
        assert result.validation_score != result.training_score

    @pytest.mark.asyncio
    async def test_base_predictor_training_insufficient_data(self):
        """Test training with insufficient data."""
        predictor = TestConcretePredictor()

        # Very small dataset
        features = pd.DataFrame({"feature1": [1, 2, 3]})
        targets = pd.DataFrame({"time_until_transition_seconds": [100, 200, 300]})

        # Should raise exception from concrete implementation
        with pytest.raises(ValueError, match="Insufficient training data"):
            await predictor.train(features, targets)

    @pytest.mark.asyncio
    async def test_base_predictor_multiple_training_sessions(
        self, sample_training_data
    ):
        """Test multiple training sessions and version increment."""
        predictor = TestConcretePredictor()
        features, targets = sample_training_data

        # First training
        result1 = await predictor.train(features, targets)
        first_version = result1.model_version

        # Second training
        result2 = await predictor.train(features, targets)
        second_version = result2.model_version

        assert len(predictor.training_history) == 2
        assert first_version != second_version
        assert second_version > first_version  # Should increment


class TestBasePredictorPrediction:
    """Test BasePredictor prediction functionality."""

    @pytest.fixture
    def trained_predictor(self, sample_training_data):
        """Trained predictor fixture."""
        predictor = TestConcretePredictor(room_id="test_room")
        features, targets = sample_training_data

        # Train the predictor
        import asyncio

        asyncio.run(predictor.train(features, targets))

        return predictor

    @pytest.fixture
    def sample_prediction_features(self):
        """Sample prediction features."""
        return pd.DataFrame(
            {"feature1": [1.5, -0.5], "feature2": [0.8, 1.2], "feature3": [-1.1, 0.3]}
        )

    @pytest.mark.asyncio
    async def test_base_predictor_prediction_success(
        self, trained_predictor, sample_prediction_features
    ):
        """Test successful prediction."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_predictor.predict(
            sample_prediction_features, prediction_time, "occupied"
        )

        assert len(results) == 2

        for i, result in enumerate(results):
            assert isinstance(result, PredictionResult)
            assert result.predicted_time > prediction_time
            assert result.transition_type == "occupied_to_vacant"
            assert result.confidence_score == 0.8
            assert result.model_type == ModelType.LSTM.value
            assert result.features_used == ["feature1", "feature2", "feature3"]

    @pytest.mark.asyncio
    async def test_base_predictor_prediction_untrained(
        self, sample_prediction_features
    ):
        """Test prediction with untrained model."""
        predictor = TestConcretePredictor()
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(sample_prediction_features, prediction_time)

    @pytest.mark.asyncio
    async def test_base_predictor_prediction_invalid_features(self, trained_predictor):
        """Test prediction with invalid features."""
        invalid_features = pd.DataFrame({"wrong_feature": [1, 2, 3]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await trained_predictor.predict(invalid_features, prediction_time)

    @pytest.mark.asyncio
    async def test_base_predictor_predict_single(self, trained_predictor):
        """Test single prediction from feature dictionary."""
        prediction_time = datetime.now(timezone.utc)
        features_dict = {"feature1": 1.5, "feature2": 0.8, "feature3": -1.1}

        result = await trained_predictor.predict_single(
            features_dict, prediction_time, "vacant"
        )

        assert isinstance(result, PredictionResult)
        assert result.transition_type == "vacant_to_occupied"
        assert result.predicted_time > prediction_time

    @pytest.mark.asyncio
    async def test_base_predictor_predict_single_empty_result(self):
        """Test predict_single with empty prediction result."""
        predictor = TestConcretePredictor()
        predictor.is_trained = True
        predictor.feature_names = ["feature1"]

        # Mock predict to return empty list
        original_predict = predictor.predict

        async def mock_predict(*args, **kwargs):
            return []

        predictor.predict = mock_predict

        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict_single({"feature1": 1.0}, prediction_time)


class TestBasePredictorFeatureValidation:
    """Test BasePredictor feature validation."""

    @pytest.fixture
    def trained_predictor_with_features(self, sample_training_data):
        """Trained predictor with known features."""
        predictor = TestConcretePredictor()
        features, targets = sample_training_data

        # Set up as trained with specific features
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        return predictor

    def test_validate_features_success(self, trained_predictor_with_features):
        """Test successful feature validation."""
        valid_features = pd.DataFrame(
            {"feature1": [1, 2, 3], "feature2": [4, 5, 6], "feature3": [7, 8, 9]}
        )

        assert trained_predictor_with_features.validate_features(valid_features) is True

    def test_validate_features_missing(self, trained_predictor_with_features):
        """Test feature validation with missing features."""
        missing_features = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                # Missing feature3
            }
        )

        assert (
            trained_predictor_with_features.validate_features(missing_features) is False
        )

    def test_validate_features_extra(self, trained_predictor_with_features):
        """Test feature validation with extra features."""
        extra_features = pd.DataFrame(
            {
                "feature1": [1, 2, 3],
                "feature2": [4, 5, 6],
                "feature3": [7, 8, 9],
                "extra_feature": [10, 11, 12],
            }
        )

        # Should still be valid (extra features are acceptable)
        assert trained_predictor_with_features.validate_features(extra_features) is True

    def test_validate_features_untrained_model(self):
        """Test feature validation with untrained model."""
        predictor = TestConcretePredictor()
        features = pd.DataFrame({"any_feature": [1, 2, 3]})

        assert predictor.validate_features(features) is False

    def test_validate_features_no_stored_features(self):
        """Test feature validation when no feature names are stored."""
        predictor = TestConcretePredictor()
        predictor.is_trained = True
        predictor.feature_names = []  # Empty feature names

        features = pd.DataFrame({"any_feature": [1, 2, 3]})

        # Should be valid if no feature names stored
        assert predictor.validate_features(features) is True


class TestBasePredictorFeatureImportance:
    """Test BasePredictor feature importance functionality."""

    @pytest.fixture
    def trained_predictor_with_importance(self, sample_training_data):
        """Trained predictor with feature importance."""
        predictor = TestConcretePredictor()
        features, targets = sample_training_data

        # Train the predictor
        import asyncio

        asyncio.run(predictor.train(features, targets))

        return predictor

    def test_get_feature_importance_trained(self, trained_predictor_with_importance):
        """Test feature importance from trained model."""
        importance = trained_predictor_with_importance.get_feature_importance()

        assert len(importance) == 3
        assert "feature1" in importance
        assert "feature2" in importance
        assert "feature3" in importance

        # Should sum to 1.0 (equal importance in mock)
        total_importance = sum(importance.values())
        assert abs(total_importance - 1.0) < 1e-6

    def test_get_feature_importance_untrained(self):
        """Test feature importance from untrained model."""
        predictor = TestConcretePredictor()

        importance = predictor.get_feature_importance()

        assert importance == {}


class TestBasePredictorModelManagement:
    """Test BasePredictor model management functionality."""

    @pytest.fixture
    def trained_predictor_for_management(self, sample_training_data):
        """Trained predictor for management testing."""
        predictor = TestConcretePredictor(room_id="management_test")
        features, targets = sample_training_data

        # Train the predictor
        import asyncio

        result = asyncio.run(predictor.train(features, targets))

        # Add some prediction history
        prediction_time = datetime.now(timezone.utc)
        mock_result = PredictionResult(
            predicted_time=prediction_time + timedelta(minutes=30),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
        )
        predictor._record_prediction(prediction_time, mock_result)

        return predictor

    def test_get_model_info(self, trained_predictor_for_management):
        """Test model information retrieval."""
        info = trained_predictor_for_management.get_model_info()

        assert info["model_type"] == ModelType.LSTM.value
        assert info["room_id"] == "management_test"
        assert info["is_trained"] is True
        assert info["training_date"] is not None
        assert info["feature_count"] == 3
        assert info["feature_names"] == ["feature1", "feature2", "feature3"]
        assert info["training_sessions"] == 1
        assert info["predictions_made"] == 1
        assert info["last_training_score"] is not None

    def test_get_training_history(self, trained_predictor_for_management):
        """Test training history retrieval."""
        history = trained_predictor_for_management.get_training_history()

        assert len(history) == 1
        assert isinstance(history[0], dict)
        assert history[0]["success"] is True
        assert "training_time_seconds" in history[0]
        assert "model_version" in history[0]

    def test_get_prediction_accuracy_insufficient_data(
        self, trained_predictor_for_management
    ):
        """Test prediction accuracy with insufficient data."""
        accuracy = trained_predictor_for_management.get_prediction_accuracy()

        # Should return None for insufficient predictions
        assert accuracy is None

    def test_get_prediction_accuracy_sufficient_data(
        self, trained_predictor_for_management
    ):
        """Test prediction accuracy with sufficient data."""
        # Add more prediction history
        prediction_time = datetime.now(timezone.utc)
        for i in range(10):
            mock_result = PredictionResult(
                predicted_time=prediction_time + timedelta(minutes=30 + i),
                transition_type="occupied_to_vacant",
                confidence_score=0.8 + i * 0.01,
            )
            trained_predictor_for_management._record_prediction(
                prediction_time - timedelta(hours=i), mock_result
            )

        accuracy = trained_predictor_for_management.get_prediction_accuracy()

        # Should return placeholder accuracy
        assert accuracy == 0.85

    def test_clear_prediction_history(self, trained_predictor_for_management):
        """Test clearing prediction history."""
        # Verify there is history
        assert len(trained_predictor_for_management.prediction_history) > 0

        trained_predictor_for_management.clear_prediction_history()

        # Should be empty
        assert len(trained_predictor_for_management.prediction_history) == 0


class TestBasePredictorSerialization:
    """Test BasePredictor model serialization functionality."""

    @pytest.fixture
    def trained_predictor_for_serialization(self, sample_training_data):
        """Trained predictor for serialization testing."""
        predictor = TestConcretePredictor(room_id="serialization_test")
        features, targets = sample_training_data

        # Train and set up model
        import asyncio

        asyncio.run(predictor.train(features, targets))

        # Mock a model object
        predictor.model = {"mock_model": "test_data"}
        predictor.model_params = {"param1": "value1"}

        return predictor

    def test_save_model_success(self, trained_predictor_for_serialization):
        """Test successful model saving."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            result = trained_predictor_for_serialization.save_model(tmp_file.name)

            assert result is True

            # Verify file was created and has content
            import os

            assert os.path.exists(tmp_file.name)
            assert os.path.getsize(tmp_file.name) > 0

            # Clean up
            os.unlink(tmp_file.name)

    def test_save_model_failure(self, trained_predictor_for_serialization):
        """Test model saving failure."""
        # Invalid path that should cause save to fail
        invalid_path = "/invalid/nonexistent/directory/model.pkl"

        result = trained_predictor_for_serialization.save_model(invalid_path)

        assert result is False

    def test_load_model_success(self, trained_predictor_for_serialization):
        """Test successful model loading."""
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            # First save the model
            save_result = trained_predictor_for_serialization.save_model(tmp_file.name)
            assert save_result is True

            # Create new predictor and load
            new_predictor = TestConcretePredictor()
            load_result = new_predictor.load_model(tmp_file.name)

            assert load_result is True
            assert new_predictor.model_type == ModelType.LSTM
            assert new_predictor.room_id == "serialization_test"
            assert new_predictor.is_trained is True
            assert new_predictor.feature_names == ["feature1", "feature2", "feature3"]
            assert new_predictor.model == {"mock_model": "test_data"}
            assert len(new_predictor.training_history) == 1

            # Clean up
            import os

            os.unlink(tmp_file.name)

    def test_load_model_failure(self):
        """Test model loading failure."""
        predictor = TestConcretePredictor()

        # Non-existent file
        result = predictor.load_model("nonexistent_file.pkl")

        assert result is False

    def test_load_model_corrupted_file(self):
        """Test loading corrupted model file."""
        predictor = TestConcretePredictor()

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".pkl", delete=False
        ) as tmp_file:
            # Write invalid data
            tmp_file.write("This is not a valid pickle file")
            tmp_file.flush()

            result = predictor.load_model(tmp_file.name)

            assert result is False

            # Clean up
            import os

            os.unlink(tmp_file.name)


class TestBasePredictorUtilityMethods:
    """Test BasePredictor utility methods."""

    def test_record_prediction(self):
        """Test prediction recording."""
        predictor = TestConcretePredictor()
        prediction_time = datetime.now(timezone.utc)

        mock_result = PredictionResult(
            predicted_time=prediction_time + timedelta(minutes=30),
            transition_type="occupied_to_vacant",
            confidence_score=0.8,
        )

        predictor._record_prediction(prediction_time, mock_result)

        assert len(predictor.prediction_history) == 1
        assert predictor.prediction_history[0] == (prediction_time, mock_result)

    def test_record_prediction_memory_management(self):
        """Test prediction recording with memory management."""
        predictor = TestConcretePredictor()
        prediction_time = datetime.now(timezone.utc)

        # Add more than 1000 predictions
        for i in range(1200):
            mock_result = PredictionResult(
                predicted_time=prediction_time + timedelta(minutes=i),
                transition_type="occupied_to_vacant",
                confidence_score=0.8,
            )
            predictor._record_prediction(
                prediction_time - timedelta(minutes=i), mock_result
            )

        # Should be trimmed to 500 most recent
        assert len(predictor.prediction_history) == 500

    def test_generate_model_version_initial(self):
        """Test initial model version generation."""
        predictor = TestConcretePredictor()

        version = predictor._generate_model_version()

        assert version == "v1.0"

    def test_generate_model_version_increment(self, sample_training_data):
        """Test model version increment."""
        predictor = TestConcretePredictor()
        features, targets = sample_training_data

        # Train multiple times
        import asyncio

        asyncio.run(predictor.train(features, targets))
        first_version = predictor.model_version

        asyncio.run(predictor.train(features, targets))
        second_version = predictor.model_version

        assert first_version == "v1.0"
        assert second_version == "v1.1"

    def test_string_representations(self):
        """Test string representations of predictor."""
        predictor = TestConcretePredictor(room_id="test_room")

        str_repr = str(predictor)
        assert "LSTM Predictor" in str_repr
        assert "test_room" in str_repr
        assert "untrained" in str_repr

        repr_str = repr(predictor)
        assert "TestConcretePredictor" in repr_str
        assert "model_type=lstm" in repr_str
        assert "room_id=test_room" in repr_str
        assert "is_trained=False" in repr_str

    def test_string_representations_trained(self, sample_training_data):
        """Test string representations of trained predictor."""
        predictor = TestConcretePredictor(room_id="test_room")
        features, targets = sample_training_data

        import asyncio

        asyncio.run(predictor.train(features, targets))

        str_repr = str(predictor)
        assert "trained" in str_repr

        repr_str = repr(predictor)
        assert "is_trained=True" in repr_str


class TestBasePredictorAbstractMethods:
    """Test BasePredictor abstract method enforcement."""

    def test_abstract_methods_not_implemented(self):
        """Test that BasePredictor cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BasePredictor(ModelType.LSTM)

    def test_concrete_implementation_required(self):
        """Test that concrete implementations must implement abstract methods."""
        # This is implicitly tested by our TestConcretePredictor working correctly
        predictor = TestConcretePredictor()

        # Should be able to call abstract methods
        assert callable(predictor.train)
        assert callable(predictor.predict)
        assert callable(predictor.get_feature_importance)


class TestBasePredictorEdgeCases:
    """Test BasePredictor edge cases and error conditions."""

    def test_predictor_with_no_features(self):
        """Test predictor behavior with no features."""
        predictor = TestConcretePredictor()

        # Should handle empty feature names gracefully
        predictor.feature_names = []
        importance = predictor.get_feature_importance()
        assert importance == {}

    def test_predictor_with_none_values(self):
        """Test predictor behavior with None values."""
        predictor = TestConcretePredictor()

        # Should handle None values gracefully
        predictor.training_date = None
        predictor.model = None

        info = predictor.get_model_info()
        assert info["training_date"] is None
        assert "model_type" in info

    def test_prediction_result_with_none_alternatives(self):
        """Test PredictionResult with None alternatives."""
        predicted_time = datetime.now(timezone.utc) + timedelta(minutes=30)

        result = PredictionResult(
            predicted_time=predicted_time,
            transition_type="occupied_to_vacant",
            confidence_score=0.85,
            alternatives=None,
        )

        result_dict = result.to_dict()

        # alternatives should not be in dict if None
        assert "alternatives" not in result_dict

    def test_training_result_with_zero_samples(self):
        """Test TrainingResult with zero training samples."""
        result = TrainingResult(
            success=False,
            training_time_seconds=0.1,
            model_version="v1.0",
            training_samples=0,
            error_message="No training data provided",
        )

        assert result.success is False
        assert result.training_samples == 0
        assert result.error_message is not None

    def test_predictor_version_generation_invalid_format(self):
        """Test version generation with invalid existing format."""
        predictor = TestConcretePredictor()

        # Add training history with invalid version format
        mock_result = TrainingResult(
            success=True,
            training_time_seconds=1.0,
            model_version="invalid_version",
            training_samples=10,
        )
        predictor.training_history.append(mock_result)

        # Should fall back to base version
        version = predictor._generate_model_version()
        assert version == "v1.0"

    def test_large_prediction_history_performance(self):
        """Test performance with large prediction history."""
        predictor = TestConcretePredictor()

        # Add many predictions quickly
        base_time = datetime.now(timezone.utc)
        for i in range(100):
            mock_result = PredictionResult(
                predicted_time=base_time + timedelta(minutes=i),
                transition_type="occupied_to_vacant",
                confidence_score=0.8,
            )
            predictor._record_prediction(base_time - timedelta(minutes=i), mock_result)

        # Should complete without performance issues
        assert len(predictor.prediction_history) == 100

        # Get accuracy should handle large history
        accuracy = predictor.get_prediction_accuracy()
        # Returns None because we need at least 5 predictions within 24 hours
        # but our mock predictions don't have actual validation data
