"""
Comprehensive unit tests for model serialization and persistence.

This module tests model save/load cycles, versioning, artifact management,
metadata preservation, backwards compatibility, and error handling for
all model types.
"""

import json
import pickle
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.core.constants import ModelType
from src.core.exceptions import ModelTrainingError
from src.models.base.gp_predictor import GaussianProcessPredictor
from src.models.base.hmm_predictor import HMMPredictor
from src.models.base.lstm_predictor import LSTMPredictor
from src.models.base.predictor import BasePredictor, PredictionResult, TrainingResult
from src.models.base.xgboost_predictor import XGBoostPredictor
from src.models.ensemble import OccupancyEnsemble


@pytest.fixture
def sample_training_data():
    """Create sample training data for model serialization tests."""
    n_samples = 100
    features = pd.DataFrame(
        {
            "temporal_hour": np.random.randint(0, 24, n_samples),
            "sequential_motion": np.random.random(n_samples),
            "contextual_temp": np.random.normal(22, 3, n_samples),
            "movement_count": np.random.poisson(2, n_samples),
        }
    )

    targets = pd.DataFrame(
        {
            "time_until_transition_seconds": np.random.exponential(1800, n_samples),
            "transition_type": np.random.choice(
                ["occupied_to_vacant", "vacant_to_occupied"], n_samples
            ),
        }
    )

    return features, targets


@pytest.fixture
def trained_xgboost_model(sample_training_data):
    """Create a trained XGBoost model for testing."""
    features, targets = sample_training_data

    model = XGBoostPredictor(room_id="test_room")

    # Use asyncio to run the training
    import asyncio

    async def train_model():
        return await model.train(features, targets)

    # Run training synchronously for fixture
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    training_result = loop.run_until_complete(train_model())
    assert training_result.success is True

    return model


@pytest.fixture
def trained_hmm_model(sample_training_data):
    """Create a trained HMM model for testing."""
    features, targets = sample_training_data

    model = HMMPredictor(room_id="test_room")

    import asyncio

    async def train_model():
        return await model.train(features, targets)

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    training_result = loop.run_until_complete(train_model())
    assert training_result.success is True

    return model


class TestBasicModelSerialization:
    """Test basic model serialization functionality."""

    def test_save_load_untrained_model(self):
        """Test save/load cycle with untrained model."""
        model = XGBoostPredictor(room_id="test_room")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save untrained model
            save_success = model.save_model(temp_path)
            assert save_success is True

            # Load into new instance
            new_model = XGBoostPredictor(room_id="different_room")
            load_success = new_model.load_model(temp_path)
            assert load_success is True

            # Verify model state
            assert new_model.is_trained is False
            assert new_model.model_type == model.model_type
            assert new_model.room_id == model.room_id  # Should be loaded from file
            assert new_model.model_version == model.model_version
            assert new_model.feature_names == model.feature_names
            assert len(new_model.training_history) == 0

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_load_trained_xgboost_model(
        self, trained_xgboost_model, sample_training_data
    ):
        """Test save/load cycle with trained XGBoost model."""
        features, targets = sample_training_data
        original_model = trained_xgboost_model

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Generate predictions before saving
            import asyncio

            async def get_predictions():
                return await original_model.predict(
                    features.head(5), datetime.utcnow(), "vacant"
                )

            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

            original_predictions = loop.run_until_complete(get_predictions())

            # Save model
            save_success = original_model.save_model(temp_path)
            assert save_success is True

            # Load into new instance
            new_model = XGBoostPredictor(room_id="placeholder")
            load_success = new_model.load_model(temp_path)
            assert load_success is True

            # Verify model state preservation
            assert new_model.is_trained is True
            assert new_model.model_type == original_model.model_type
            assert new_model.room_id == original_model.room_id
            assert new_model.model_version == original_model.model_version
            assert new_model.feature_names == original_model.feature_names
            assert len(new_model.training_history) == len(
                original_model.training_history
            )

            # Test that predictions are consistent
            async def get_new_predictions():
                return await new_model.predict(
                    features.head(5), datetime.utcnow(), "vacant"
                )

            new_predictions = loop.run_until_complete(get_new_predictions())

            assert len(new_predictions) == len(original_predictions)

            # Check prediction consistency (allowing for small numerical differences)
            for orig, new in zip(original_predictions, new_predictions):
                time_diff = abs(
                    (orig.predicted_time - new.predicted_time).total_seconds()
                )
                assert time_diff < 60  # Allow up to 1 minute difference
                assert orig.transition_type == new.transition_type

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_save_load_trained_hmm_model(self, trained_hmm_model):
        """Test save/load cycle with trained HMM model."""
        original_model = trained_hmm_model

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save model
            save_success = original_model.save_model(temp_path)
            assert save_success is True

            # Load into new instance
            new_model = HMMPredictor(room_id="placeholder")
            load_success = new_model.load_model(temp_path)
            assert load_success is True

            # Verify model state preservation
            assert new_model.is_trained is True
            assert new_model.model_type == original_model.model_type
            assert new_model.room_id == original_model.room_id
            assert new_model.feature_names == original_model.feature_names
            assert len(new_model.training_history) == len(
                original_model.training_history
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_model_serialization_with_training_history(self, sample_training_data):
        """Test serialization preserves complete training history."""
        features, targets = sample_training_data
        model = XGBoostPredictor(room_id="test_room")

        import asyncio

        async def train_multiple_times():
            # Train model multiple times to build history
            result1 = await model.train(features, targets)
            result2 = await model.train(
                features.tail(80), targets.tail(80)
            )  # Incremental
            return result1, result2

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        result1, result2 = loop.run_until_complete(train_multiple_times())

        assert len(model.training_history) >= 2

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save and load model
            model.save_model(temp_path)

            new_model = XGBoostPredictor(room_id="placeholder")
            new_model.load_model(temp_path)

            # Verify training history preservation
            assert len(new_model.training_history) == len(model.training_history)

            for orig_result, new_result in zip(
                model.training_history, new_model.training_history
            ):
                assert orig_result.success == new_result.success
                assert orig_result.training_samples == new_result.training_samples
                assert orig_result.model_version == new_result.model_version

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestModelVersioning:
    """Test model versioning and version management."""

    def test_model_version_preservation(self, trained_xgboost_model):
        """Test that model versions are preserved during serialization."""
        model = trained_xgboost_model
        original_version = model.model_version

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save and load model
            model.save_model(temp_path)

            new_model = XGBoostPredictor(room_id="placeholder")
            new_model.load_model(temp_path)

            assert new_model.model_version == original_version

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_model_version_generation(self):
        """Test model version generation logic."""
        model = XGBoostPredictor(room_id="test_room")

        # Test initial version generation
        version1 = model._generate_model_version()
        assert version1 == "v1.0"

        # Add training history and test version increment
        model.training_history.append(
            TrainingResult(
                success=True,
                training_time_seconds=100,
                model_version="v1.2",
                training_samples=1000,
            )
        )

        version2 = model._generate_model_version()
        assert version2 == "v1.3"

        # Test with different version formats
        model.training_history.append(
            TrainingResult(
                success=True,
                training_time_seconds=100,
                model_version="v2.5",
                training_samples=1000,
            )
        )

        version3 = model._generate_model_version()
        assert version3 == "v2.6"

    def test_model_version_in_serialized_data(self, trained_xgboost_model):
        """Test that version information is correctly stored in serialized data."""
        model = trained_xgboost_model

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save model
            model.save_model(temp_path)

            # Read raw pickle data
            with open(temp_path, "rb") as f:
                model_data = pickle.load(f)

            # Verify version data is present
            assert "model_version" in model_data
            assert model_data["model_version"] == model.model_version
            assert "training_date" in model_data
            assert "training_history" in model_data

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestEnsembleModelSerialization:
    """Test serialization of ensemble models."""

    def test_ensemble_serialization_structure(self, sample_training_data):
        """Test ensemble model serialization preserves structure."""
        features, targets = sample_training_data
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Mock base models to avoid complex training
        for model_name in ensemble.base_models.keys():
            mock_model = MagicMock()
            mock_model.is_trained = True
            mock_model.model_type = (
                ModelType.XGBOOST if model_name == "xgboost" else ModelType.LSTM
            )
            mock_model.feature_names = list(features.columns)
            mock_model.model_version = "v1.0"
            mock_model.training_history = []
            ensemble.base_models[model_name] = mock_model

        # Set ensemble as trained
        ensemble.is_trained = True
        ensemble.base_models_trained = True
        ensemble.meta_learner_trained = True
        ensemble.feature_names = list(features.columns)
        ensemble.model_weights = {"xgboost": 0.4, "lstm": 0.3, "hmm": 0.2, "gp": 0.1}

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save ensemble
            save_success = ensemble.save_model(temp_path)
            assert save_success is True

            # Load ensemble
            new_ensemble = OccupancyEnsemble(room_id="placeholder")
            load_success = new_ensemble.load_model(temp_path)
            assert load_success is True

            # Verify ensemble structure
            assert new_ensemble.is_trained is True
            assert new_ensemble.room_id == ensemble.room_id
            assert new_ensemble.feature_names == ensemble.feature_names
            assert new_ensemble.model_type == ModelType.ENSEMBLE

            # Note: Base models are mocked, so deep verification is limited
            # In real scenario, base models would also be serialized

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_ensemble_base_model_serialization(self):
        """Test that ensemble base models are properly handled during serialization."""
        ensemble = OccupancyEnsemble(room_id="test_room")

        # Verify base models are initialized
        assert len(ensemble.base_models) == 4
        assert "lstm" in ensemble.base_models
        assert "xgboost" in ensemble.base_models
        assert "hmm" in ensemble.base_models
        assert "gp" in ensemble.base_models

        # Test serialization of untrained ensemble
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            save_success = ensemble.save_model(temp_path)
            assert save_success is True

            new_ensemble = OccupancyEnsemble(room_id="placeholder")
            load_success = new_ensemble.load_model(temp_path)
            assert load_success is True

            # Base models should be preserved
            assert len(new_ensemble.base_models) == 4
            for model_name in ["lstm", "xgboost", "hmm", "gp"]:
                assert model_name in new_ensemble.base_models
                # Base models should have same room_id as ensemble
                assert new_ensemble.base_models[model_name].room_id == ensemble.room_id

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSerializationErrorHandling:
    """Test error handling during model serialization."""

    def test_save_to_invalid_path(self):
        """Test saving model to invalid file path."""
        model = XGBoostPredictor(room_id="test_room")

        # Try to save to non-existent directory
        invalid_path = "/nonexistent/directory/model.pkl"
        save_success = model.save_model(invalid_path)
        assert save_success is False

        # Try to save to read-only path (if possible to simulate)
        # This test may be platform-specific

    def test_load_from_invalid_path(self):
        """Test loading model from invalid file path."""
        model = XGBoostPredictor(room_id="test_room")

        # Try to load from non-existent file
        invalid_path = "/nonexistent/model.pkl"
        load_success = model.load_model(invalid_path)
        assert load_success is False

        # Model state should remain unchanged
        assert model.is_trained is False
        assert model.model_version == "v1.0"  # Default version

    def test_load_corrupted_model_file(self):
        """Test loading from corrupted model file."""
        model = XGBoostPredictor(room_id="test_room")

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

            # Write corrupted data to file
            tmp_file.write(b"corrupted pickle data")

        try:
            load_success = model.load_model(temp_path)
            assert load_success is False

            # Model should remain in default state
            assert model.is_trained is False
            assert model.room_id == "test_room"  # Original room_id preserved

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_incompatible_model_file(self, trained_xgboost_model):
        """Test loading incompatible model file."""
        # Save XGBoost model
        xgb_model = trained_xgboost_model

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            xgb_model.save_model(temp_path)

            # Try to load XGBoost model into HMM model
            hmm_model = HMMPredictor(room_id="test_room")
            load_success = hmm_model.load_model(temp_path)

            # Should succeed - the base class handles cross-model loading
            # The model_type will be updated from the loaded data
            assert load_success is True
            assert hmm_model.model_type == xgb_model.model_type  # Should be updated

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_partial_model_data_loading(self):
        """Test loading model with missing data fields."""
        model = XGBoostPredictor(room_id="test_room")

        # Create partial model data (missing some fields)
        partial_model_data = {
            "model": MagicMock(),
            "model_type": ModelType.XGBOOST.value,
            "room_id": "test_room",
            # Missing model_version, training_date, etc.
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save partial data
            with open(temp_path, "wb") as f:
                pickle.dump(partial_model_data, f)

            # Load should handle missing fields gracefully
            load_success = model.load_model(temp_path)
            assert load_success is True

            # Missing fields should have default values
            assert model.model_version == "v1.0"  # Default
            assert model.training_date is None
            assert model.feature_names == []  # Default empty list

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSerializationPerformance:
    """Test serialization performance and efficiency."""

    def test_serialization_time_performance(self, trained_xgboost_model):
        """Test that serialization completes within reasonable time."""
        model = trained_xgboost_model

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Measure save time
            start_time = time.time()
            save_success = model.save_model(temp_path)
            save_time = time.time() - start_time

            assert save_success is True
            assert save_time < 5.0  # Should complete within 5 seconds

            # Measure load time
            new_model = XGBoostPredictor(room_id="placeholder")
            start_time = time.time()
            load_success = new_model.load_model(temp_path)
            load_time = time.time() - start_time

            assert load_success is True
            assert load_time < 5.0  # Should complete within 5 seconds

            print(
                f"Serialization performance - Save: {save_time:.3f}s, Load: {load_time:.3f}s"
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_serialization_file_size_efficiency(self, trained_xgboost_model):
        """Test that serialized files are reasonably sized."""
        model = trained_xgboost_model

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            model.save_model(temp_path)

            file_size = Path(temp_path).stat().st_size

            # File should be reasonable size (< 10MB for test model)
            assert file_size < 10 * 1024 * 1024  # 10MB
            assert file_size > 1024  # Should be > 1KB (not empty)

            print(
                f"Serialized model file size: {file_size:,} bytes ({file_size/1024:.1f} KB)"
            )

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestMultipleModelSerialization:
    """Test serialization of multiple models."""

    def test_multiple_model_save_load(self, sample_training_data):
        """Test saving and loading multiple different model types."""
        features, targets = sample_training_data

        # Create multiple trained models
        models = {
            "xgboost": XGBoostPredictor(room_id="test_room"),
            "hmm": HMMPredictor(room_id="test_room"),
        }

        # Train models
        import asyncio

        async def train_all_models():
            results = {}
            for name, model in models.items():
                result = await model.train(features, targets)
                results[name] = result
            return results

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        training_results = loop.run_until_complete(train_all_models())

        # Verify all trained successfully
        for name, result in training_results.items():
            assert result.success is True, f"Training failed for {name}"

        # Save all models
        model_paths = {}
        try:
            for name, model in models.items():
                tmp_file = tempfile.NamedTemporaryFile(
                    suffix=f"_{name}.pkl", delete=False
                )
                temp_path = tmp_file.name
                tmp_file.close()

                save_success = model.save_model(temp_path)
                assert save_success is True, f"Save failed for {name}"

                model_paths[name] = temp_path

            # Load all models into new instances
            loaded_models = {}
            for name, path in model_paths.items():
                if name == "xgboost":
                    new_model = XGBoostPredictor(room_id="placeholder")
                elif name == "hmm":
                    new_model = HMMPredictor(room_id="placeholder")

                load_success = new_model.load_model(path)
                assert load_success is True, f"Load failed for {name}"

                loaded_models[name] = new_model

            # Verify all models loaded correctly
            for name in models.keys():
                original = models[name]
                loaded = loaded_models[name]

                assert loaded.is_trained is True
                assert loaded.model_type == original.model_type
                assert loaded.room_id == original.room_id
                assert loaded.feature_names == original.feature_names

        finally:
            # Clean up all temp files
            for path in model_paths.values():
                Path(path).unlink(missing_ok=True)

    def test_model_comparison_after_serialization(self, sample_training_data):
        """Test that models behave consistently after serialization."""
        features, targets = sample_training_data

        # Create and train model
        original_model = XGBoostPredictor(room_id="test_room")

        import asyncio

        async def train_and_predict():
            training_result = await original_model.train(features, targets)
            original_predictions = await original_model.predict(
                features.head(3), datetime.utcnow(), "vacant"
            )
            return training_result, original_predictions

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        training_result, original_predictions = loop.run_until_complete(
            train_and_predict()
        )
        assert training_result.success is True

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save and load model
            original_model.save_model(temp_path)

            loaded_model = XGBoostPredictor(room_id="placeholder")
            loaded_model.load_model(temp_path)

            # Get predictions from loaded model
            async def get_loaded_predictions():
                return await loaded_model.predict(
                    features.head(3), datetime.utcnow(), "vacant"
                )

            loaded_predictions = loop.run_until_complete(get_loaded_predictions())

            # Compare predictions (should be very similar)
            assert len(loaded_predictions) == len(original_predictions)

            for orig, loaded in zip(original_predictions, loaded_predictions):
                # Time predictions should be close
                time_diff = abs(
                    (orig.predicted_time - loaded.predicted_time).total_seconds()
                )
                assert time_diff < 120  # Allow 2 minutes difference

                # Other attributes should match
                assert orig.transition_type == loaded.transition_type
                assert abs(orig.confidence_score - loaded.confidence_score) < 0.1

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSerializationMetadata:
    """Test serialization of model metadata and auxiliary information."""

    def test_feature_names_serialization(self, sample_training_data):
        """Test that feature names are properly serialized."""
        features, targets = sample_training_data
        model = XGBoostPredictor(room_id="test_room")

        import asyncio

        async def train_model():
            return await model.train(features, targets)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        training_result = loop.run_until_complete(train_model())
        assert training_result.success is True

        original_feature_names = model.feature_names.copy()

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            model.save_model(temp_path)

            new_model = XGBoostPredictor(room_id="placeholder")
            new_model.load_model(temp_path)

            # Feature names should be preserved exactly
            assert new_model.feature_names == original_feature_names
            assert len(new_model.feature_names) == len(features.columns)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_model_parameters_serialization(self):
        """Test that model parameters are properly serialized."""
        # Create model with custom parameters
        custom_params = {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05}

        model = XGBoostPredictor(room_id="test_room", **custom_params)

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            model.save_model(temp_path)

            new_model = XGBoostPredictor(room_id="placeholder")
            new_model.load_model(temp_path)

            # Model parameters should be preserved
            for key, value in custom_params.items():
                assert key in new_model.model_params
                assert new_model.model_params[key] == value

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_training_metadata_preservation(self, sample_training_data):
        """Test that training metadata is preserved during serialization."""
        features, targets = sample_training_data
        model = XGBoostPredictor(room_id="test_room")

        import asyncio

        async def train_model():
            return await model.train(features, targets)

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        training_result = loop.run_until_complete(train_model())
        assert training_result.success is True

        original_training_date = model.training_date

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            model.save_model(temp_path)

            new_model = XGBoostPredictor(room_id="placeholder")
            new_model.load_model(temp_path)

            # Training metadata should be preserved
            assert new_model.training_date == original_training_date
            assert new_model.is_trained is True

            # Training history should be preserved
            assert len(new_model.training_history) == len(model.training_history)

            if model.training_history:
                original_history = model.training_history[0]
                loaded_history = new_model.training_history[0]

                assert loaded_history.success == original_history.success
                assert (
                    loaded_history.training_samples == original_history.training_samples
                )
                assert loaded_history.model_version == original_history.model_version

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestBackwardsCompatibility:
    """Test backwards compatibility of model serialization."""

    def test_version_compatibility_handling(self):
        """Test handling of models saved with different versions."""
        model = XGBoostPredictor(room_id="test_room")

        # Simulate older version model data
        old_model_data = {
            "model": MagicMock(),
            "model_type": ModelType.XGBOOST.value,
            "room_id": "test_room",
            "model_version": "v0.9",  # Older version
            "is_trained": True,
            # Missing some newer fields like training_history
        }

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save old format data
            with open(temp_path, "wb") as f:
                pickle.dump(old_model_data, f)

            # Load into new model instance
            load_success = model.load_model(temp_path)
            assert load_success is True

            # Should handle missing fields gracefully
            assert model.is_trained is True
            assert model.model_version == "v0.9"
            assert model.training_history == []  # Default for missing field

        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_forward_compatibility_considerations(self):
        """Test considerations for forward compatibility."""
        model = XGBoostPredictor(room_id="test_room")

        # Add extra fields that might be added in future versions
        model.future_field = "future_value"

        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            # Save model (should handle extra fields gracefully)
            save_success = model.save_model(temp_path)
            assert save_success is True

            # Load model (extra fields should be ignored without error)
            new_model = XGBoostPredictor(room_id="placeholder")
            load_success = new_model.load_model(temp_path)
            assert load_success is True

            # Core functionality should work
            assert new_model.room_id == "test_room"
            assert new_model.model_type == ModelType.XGBOOST

        finally:
            Path(temp_path).unlink(missing_ok=True)
