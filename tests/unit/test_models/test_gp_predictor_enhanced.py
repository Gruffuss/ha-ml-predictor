"""
Enhanced comprehensive test suite for GaussianProcessPredictor class.

This test suite focuses on achieving 85%+ coverage of GP predictor functionality
with realistic ML scenarios, uncertainty quantification, and proper error handling.
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
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.gp_predictor import GaussianProcessPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


@pytest.fixture
def gp_training_data():
    """Create training data suitable for GP learning."""
    np.random.seed(42)
    n_samples = 100

    # Create smooth continuous features (GP-friendly)
    t = np.linspace(0, 4 * np.pi, n_samples)
    features = {
        "smooth_temporal": np.sin(t) + 0.2 * np.sin(3 * t),
        "trend_component": np.linspace(-1, 1, n_samples),
        "noise_level": np.random.exponential(0.3, n_samples),
        "temperature": 20 + 2 * np.sin(t / 2) + np.random.normal(0, 0.5, n_samples),
        "cyclical_pattern": np.cos(2 * t) + np.random.normal(0, 0.1, n_samples),
    }

    features_df = pd.DataFrame(features)

    # Create realistic targets with smooth transitions
    base_time = 1800  # 30 minutes
    smooth_influence = 1 + 0.4 * features["smooth_temporal"]
    temp_influence = 1 + 0.2 * np.tanh((features["temperature"] - 21) / 2)

    time_until_transition = base_time * smooth_influence * temp_influence
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
def trained_gp_predictor(gp_training_data):
    """Create a trained GP predictor for testing predictions."""
    features, targets = gp_training_data
    predictor = GaussianProcessPredictor("test_room")

    # Mock the training for faster tests
    predictor.gp_model = GaussianProcessRegressor(random_state=42)
    predictor.is_trained = True
    predictor.feature_names = list(features.columns)
    predictor.training_date = datetime.now(timezone.utc)
    predictor.model_version = "v1.0_test"

    # Fit the model and scaler
    predictor.feature_scaler.fit(features)
    scaled_features = predictor.feature_scaler.transform(features)
    target_values = targets["time_until_transition_seconds"].values
    predictor.gp_model.fit(scaled_features, target_values)

    return predictor


class TestGPInitialization:
    """Test GP predictor initialization and configuration."""

    def test_default_initialization(self):
        """Test default GP predictor initialization."""
        gp = GaussianProcessPredictor()

        assert gp.model_type == ModelType.GP
        assert gp.room_id is None
        assert not gp.is_trained
        assert gp.gp_model is None
        assert isinstance(gp.feature_scaler, StandardScaler)
        assert gp.model_params["kernel"] == "composite"

    def test_room_specific_initialization(self):
        """Test GP initialization with room ID."""
        room_id = "living_room"
        gp = GaussianProcessPredictor(room_id)

        assert gp.room_id == room_id
        assert gp.model_type == ModelType.GP

    def test_kernel_type_configurations(self):
        """Test GP initialization with different kernel types."""
        kernels = ["composite", "rbf", "matern", "periodic", "rational_quadratic"]

        for kernel_type in kernels:
            gp = GaussianProcessPredictor(kernel=kernel_type)
            assert gp.model_params["kernel"] == kernel_type
            assert gp.model_params["kernel_type"] == kernel_type

    def test_custom_parameters(self):
        """Test GP initialization with custom parameters."""
        gp = GaussianProcessPredictor(
            room_id="test_room",
            kernel="rbf",
            alpha=1e-5,
            n_restarts_optimizer=5,
            normalize_y=False,
            confidence_intervals=[0.5, 0.9, 0.99],
        )

        assert gp.model_params["kernel"] == "rbf"
        assert gp.model_params["alpha"] == 1e-5
        assert gp.model_params["n_restarts_optimizer"] == 5
        assert gp.model_params["normalize_y"] is False
        assert gp.model_params["confidence_intervals"] == [0.5, 0.9, 0.99]

    def test_uncertainty_parameters(self):
        """Test uncertainty quantification parameters."""
        gp = GaussianProcessPredictor(
            uncertainty_threshold=0.3, max_inducing_points=1000
        )

        assert gp.model_params["uncertainty_threshold"] == 0.3
        assert gp.model_params["max_inducing_points"] == 1000


class TestGPKernelCreation:
    """Test GP kernel creation for different types."""

    def test_composite_kernel_creation(self):
        """Test composite kernel creation."""
        gp = GaussianProcessPredictor(kernel="composite")
        kernel = gp._create_kernel("composite")

        assert kernel is not None
        # Composite kernel should be a sum/product of multiple kernels

    def test_rbf_kernel_creation(self):
        """Test RBF kernel creation."""
        gp = GaussianProcessPredictor(kernel="rbf")
        kernel = gp._create_kernel("rbf")

        assert kernel is not None

    def test_matern_kernel_creation(self):
        """Test Matern kernel creation."""
        gp = GaussianProcessPredictor(kernel="matern")
        kernel = gp._create_kernel("matern")

        assert kernel is not None

    def test_periodic_kernel_creation(self):
        """Test periodic kernel creation."""
        gp = GaussianProcessPredictor(kernel="periodic")
        kernel = gp._create_kernel("periodic")

        assert kernel is not None

    def test_rational_quadratic_kernel_creation(self):
        """Test rational quadratic kernel creation."""
        gp = GaussianProcessPredictor(kernel="rational_quadratic")
        kernel = gp._create_kernel("rational_quadratic")

        assert kernel is not None

    def test_unknown_kernel_fallback(self):
        """Test fallback for unknown kernel types."""
        gp = GaussianProcessPredictor()
        kernel = gp._create_kernel("unknown_kernel")

        # Should fallback to RBF
        assert kernel is not None


class TestGPTraining:
    """Test GP training functionality."""

    @pytest.mark.asyncio
    async def test_training_success(self, gp_training_data):
        """Test successful GP training."""
        features, targets = gp_training_data
        gp = GaussianProcessPredictor("test_room", kernel="rbf")

        result = await gp.train(features, targets)

        assert result.success
        assert gp.is_trained
        assert gp.gp_model is not None
        assert result.training_time_seconds > 0
        assert result.training_samples > 0
        assert gp.feature_names == list(features.columns)

    @pytest.mark.asyncio
    async def test_training_different_kernels(self, gp_training_data):
        """Test training with different kernel types."""
        features, targets = gp_training_data
        kernels = ["rbf", "matern", "composite"]

        for kernel_type in kernels:
            gp = GaussianProcessPredictor(f"room_{kernel_type}", kernel=kernel_type)

            result = await gp.train(features, targets)

            assert result.success, f"Training failed for kernel {kernel_type}"
            assert gp.is_trained
            assert kernel_type in result.training_metrics.get("kernel_type", "")

    @pytest.mark.asyncio
    async def test_training_with_validation(self, gp_training_data):
        """Test GP training with validation data."""
        features, targets = gp_training_data

        # Split data
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        val_features = features[train_size:]
        val_targets = targets[train_size:]

        gp = GaussianProcessPredictor("test_room")

        result = await gp.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success
        assert result.validation_score is not None
        assert "validation_mae" in result.training_metrics

    @pytest.mark.asyncio
    async def test_training_insufficient_data(self):
        """Test training with insufficient data."""
        gp = GaussianProcessPredictor("test_room")

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
            await gp.train(features, targets)

    @pytest.mark.asyncio
    async def test_training_inducing_points_selection(self, gp_training_data):
        """Test inducing points selection for large datasets."""
        features, targets = gp_training_data

        # Test with small max_inducing_points to trigger selection
        gp = GaussianProcessPredictor("test_room", max_inducing_points=20)

        result = await gp.train(features, targets)

        assert result.success
        # Should have training metrics about inducing points
        assert "inducing_points_selected" in result.training_metrics

    @pytest.mark.asyncio
    async def test_training_error_handling(self, gp_training_data):
        """Test training error handling."""
        features, targets = gp_training_data
        gp = GaussianProcessPredictor("test_room")

        # Mock GaussianProcessRegressor to raise an error
        with patch("src.models.base.gp_predictor.GaussianProcessRegressor") as mock_gpr:
            mock_gpr.return_value.fit.side_effect = Exception("GP training failed")

            with pytest.raises(ModelTrainingError):
                await gp.train(features, targets)


class TestGPPrediction:
    """Test GP prediction functionality."""

    @pytest.mark.asyncio
    async def test_prediction_generation(self, trained_gp_predictor, gp_training_data):
        """Test basic prediction generation with uncertainty."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(10)

        results = await gp.predict(test_features, prediction_time, "vacant")

        assert len(results) > 0
        assert all(isinstance(r, PredictionResult) for r in results)
        assert all(r.predicted_time > prediction_time for r in results)
        assert all(r.confidence_score >= 0 and r.confidence_score <= 1 for r in results)

        # GP should provide uncertainty quantification
        for result in results:
            assert "uncertainty_quantification" in result.prediction_metadata
            uncertainty = result.prediction_metadata["uncertainty_quantification"]
            assert "aleatoric_uncertainty" in uncertainty
            assert "epistemic_uncertainty" in uncertainty

    @pytest.mark.asyncio
    async def test_prediction_different_states(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test predictions for different occupancy states."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(5)

        # Test different states
        vacant_results = await gp.predict(test_features, prediction_time, "vacant")
        occupied_results = await gp.predict(test_features, prediction_time, "occupied")
        unknown_results = await gp.predict(test_features, prediction_time, "unknown")

        assert len(vacant_results) > 0
        assert len(occupied_results) > 0
        assert len(unknown_results) > 0

    @pytest.mark.asyncio
    async def test_prediction_confidence_intervals(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test confidence interval generation."""
        features, _ = gp_training_data
        gp = trained_gp_predictor
        gp.model_params["confidence_intervals"] = [0.68, 0.95]

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(5)

        results = await gp.predict(test_features, prediction_time, "vacant")

        for result in results:
            assert "confidence_intervals" in result.prediction_metadata
            intervals = result.prediction_metadata["confidence_intervals"]
            assert len(intervals) == 2
            assert all(isinstance(interval, dict) for interval in intervals)
            assert all("confidence_level" in interval for interval in intervals)
            assert all("lower_bound" in interval for interval in intervals)
            assert all("upper_bound" in interval for interval in intervals)

    @pytest.mark.asyncio
    async def test_prediction_uncertainty_thresholding(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test uncertainty thresholding for confidence adjustment."""
        features, _ = gp_training_data
        gp = trained_gp_predictor
        gp.model_params["uncertainty_threshold"] = 0.1  # Low threshold

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(5)

        results = await gp.predict(test_features, prediction_time, "vacant")

        # With low uncertainty threshold, some predictions might have adjusted confidence
        assert len(results) > 0
        for result in results:
            uncertainty = result.prediction_metadata["uncertainty_quantification"]
            total_uncertainty = (
                uncertainty["aleatoric_uncertainty"]
                + uncertainty["epistemic_uncertainty"]
            )
            # Confidence should be adjusted based on uncertainty
            assert 0.1 <= result.confidence_score <= 0.95

    @pytest.mark.asyncio
    async def test_prediction_untrained_model(self, gp_training_data):
        """Test prediction with untrained model."""
        features, _ = gp_training_data
        gp = GaussianProcessPredictor("test_room")

        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError, match="Model not trained"):
            await gp.predict(features, prediction_time, "vacant")

    @pytest.mark.asyncio
    async def test_prediction_bounds_enforcement(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test that predictions are bounded within reasonable limits."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(5)

        results = await gp.predict(test_features, prediction_time, "vacant")

        for result in results:
            time_diff = (result.predicted_time - prediction_time).total_seconds()
            assert 60 <= time_diff <= 86400  # 1 minute to 24 hours

    @pytest.mark.asyncio
    async def test_prediction_metadata_completeness(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test prediction metadata completeness."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(5)

        results = await gp.predict(test_features, prediction_time, "vacant")

        for result in results:
            assert result.model_type == "gaussian_process"
            assert result.model_version == gp.model_version
            assert result.features_used == gp.feature_names
            assert "time_until_transition_seconds" in result.prediction_metadata
            assert "kernel_type" in result.prediction_metadata
            assert "prediction_std" in result.prediction_metadata


class TestGPFeatureImportance:
    """Test GP feature importance calculation."""

    def test_feature_importance_trained_model(self, trained_gp_predictor):
        """Test feature importance calculation for trained model."""
        gp = trained_gp_predictor

        importance = gp.get_feature_importance()

        assert isinstance(importance, dict)
        assert len(importance) == len(gp.feature_names)
        assert all(isinstance(v, float) for v in importance.values())
        assert all(v >= 0 for v in importance.values())

    def test_feature_importance_untrained_model(self):
        """Test feature importance for untrained model."""
        gp = GaussianProcessPredictor("test_room")

        importance = gp.get_feature_importance()

        assert importance == {}

    def test_feature_importance_length_scale_based(self, trained_gp_predictor):
        """Test feature importance based on kernel length scales."""
        gp = trained_gp_predictor

        # Mock kernel with length_scale parameter
        mock_kernel = Mock()
        mock_kernel.get_params.return_value = {
            "k2__length_scale": np.array([1.0, 2.0, 0.5, 1.5, 0.8])  # 5 features
        }
        gp.gp_model.kernel_ = mock_kernel

        importance = gp.get_feature_importance()

        # Should have importance for each feature
        assert len(importance) == len(gp.feature_names)
        # Importance should be inversely related to length scale
        # (shorter length scale = more important)


class TestGPModelPersistence:
    """Test GP model save/load functionality."""

    def test_save_and_load_model(self, trained_gp_predictor):
        """Test saving and loading GP model."""
        gp = trained_gp_predictor

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gp_model.pkl"

            # Save model
            success = gp.save_model(model_path)
            assert success
            assert model_path.exists()

            # Load model
            new_gp = GaussianProcessPredictor("test_room")
            load_success = new_gp.load_model(model_path)

            assert load_success
            assert new_gp.is_trained
            assert new_gp.model_version == gp.model_version
            assert new_gp.feature_names == gp.feature_names

    def test_save_load_failure_scenarios(self):
        """Test save/load failure scenarios."""
        gp = GaussianProcessPredictor("test_room")

        # Test save failure with invalid path
        invalid_path = "/invalid/path/model.pkl"
        success = gp.save_model(invalid_path)
        assert not success

        # Test load failure with non-existent file
        non_existent_path = "/non/existent/model.pkl"
        load_success = gp.load_model(non_existent_path)
        assert not load_success

    def test_model_state_preservation(self, trained_gp_predictor, gp_training_data):
        """Test that model state is preserved through save/load."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        # Make a prediction before saving
        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(5)
        original_results = asyncio.run(
            gp.predict(test_features, prediction_time, "vacant")
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "gp_model.pkl"

            # Save and load
            gp.save_model(model_path)
            new_gp = GaussianProcessPredictor("test_room")
            new_gp.load_model(model_path)

            # Make the same prediction
            new_results = asyncio.run(
                new_gp.predict(test_features, prediction_time, "vacant")
            )

            # Results should be very similar
            assert len(new_results) == len(original_results)
            for orig, new in zip(original_results, new_results):
                time_diff = abs(
                    (orig.predicted_time - new.predicted_time).total_seconds()
                )
                assert time_diff < 300  # Within 5 minutes


class TestGPUtilityMethods:
    """Test GP utility methods."""

    def test_validate_features_trained_model(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test feature validation with trained model."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        # Valid features
        test_features = features.head(5)
        assert gp.validate_features(test_features)

        # Invalid features (missing columns)
        invalid_features = pd.DataFrame({"wrong_column": [1, 2, 3]})
        assert not gp.validate_features(invalid_features)

    def test_validate_features_untrained_model(self, gp_training_data):
        """Test feature validation with untrained model."""
        features, _ = gp_training_data
        gp = GaussianProcessPredictor("test_room")

        # Should return True for untrained model
        assert gp.validate_features(features)

    def test_get_model_complexity(self, trained_gp_predictor):
        """Test model complexity calculation."""
        gp = trained_gp_predictor

        complexity = gp.get_model_complexity()

        assert isinstance(complexity, dict)
        assert "kernel_type" in complexity
        assert "n_features" in complexity
        assert "training_samples" in complexity
        assert complexity["n_features"] > 0

    def test_get_model_complexity_untrained(self):
        """Test model complexity for untrained model."""
        gp = GaussianProcessPredictor("test_room")

        complexity = gp.get_model_complexity()

        assert complexity["training_samples"] == 0
        assert complexity["n_features"] == 0


class TestGPIncrementalUpdate:
    """Test GP incremental update functionality."""

    @pytest.mark.asyncio
    async def test_incremental_update_success(
        self, trained_gp_predictor, gp_training_data
    ):
        """Test successful incremental update."""
        features, targets = gp_training_data
        gp = trained_gp_predictor

        # Use subset of data for incremental update
        new_features = features.tail(20)
        new_targets = targets.tail(20)

        result = await gp.incremental_update(
            new_features, new_targets, learning_rate=0.01
        )

        assert result.success
        assert "update_type" in result.training_metrics
        assert result.training_metrics["update_type"] == "incremental"

    @pytest.mark.asyncio
    async def test_incremental_update_untrained(self, gp_training_data):
        """Test incremental update with untrained model falls back to full training."""
        features, targets = gp_training_data
        gp = GaussianProcessPredictor("test_room")

        # Should fallback to full training
        result = await gp.incremental_update(features, targets)

        assert result.success
        assert gp.is_trained

    @pytest.mark.asyncio
    async def test_incremental_update_insufficient_data(self, trained_gp_predictor):
        """Test incremental update with insufficient data."""
        gp = trained_gp_predictor

        # Minimal data
        features = pd.DataFrame({"f1": [1, 2], "f2": [3, 4]})
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [1800, 2000],
                "transition_type": ["vacant_to_occupied"] * 2,
                "target_time": [datetime.now(timezone.utc)] * 2,
            }
        )

        with pytest.raises(ModelTrainingError, match="Insufficient data"):
            await gp.incremental_update(features, targets)


class TestGPPerformanceBenchmarks:
    """Test GP performance benchmarks."""

    @pytest.mark.asyncio
    async def test_training_performance(self, gp_training_data):
        """Test training performance benchmark."""
        features, targets = gp_training_data
        gp = GaussianProcessPredictor("test_room", kernel="rbf")

        start_time = time.time()
        result = await gp.train(features, targets)
        training_time = time.time() - start_time

        assert result.success
        assert training_time < 60  # Should complete within 60 seconds
        assert result.training_time_seconds > 0

    @pytest.mark.asyncio
    async def test_prediction_latency(self, trained_gp_predictor, gp_training_data):
        """Test prediction latency benchmark."""
        features, _ = gp_training_data
        gp = trained_gp_predictor

        prediction_time = datetime.now(timezone.utc)
        test_features = features.tail(10)

        start_time = time.time()
        results = await gp.predict(test_features, prediction_time, "vacant")
        latency = time.time() - start_time

        assert len(results) > 0
        assert latency < 2.0  # Should be under 2 seconds

    @pytest.mark.asyncio
    async def test_uncertainty_quality(self, gp_training_data):
        """Test quality of uncertainty estimates."""
        features, targets = gp_training_data

        # Split data for testing
        train_size = int(0.8 * len(features))
        train_features = features[:train_size]
        train_targets = targets[:train_size]
        test_features = features[train_size:]
        test_targets = targets[train_size:]

        gp = GaussianProcessPredictor("test_room", kernel="rbf")
        await gp.train(train_features, train_targets)

        # Make predictions on test data
        prediction_time = datetime.now(timezone.utc)
        predictions = await gp.predict(test_features, prediction_time, "vacant")

        # Check that uncertainty estimates are reasonable
        uncertainties = []
        for pred in predictions:
            uncertainty = pred.prediction_metadata["uncertainty_quantification"]
            total_uncertainty = (
                uncertainty["aleatoric_uncertainty"]
                + uncertainty["epistemic_uncertainty"]
            )
            uncertainties.append(total_uncertainty)

        # Uncertainties should be positive and reasonable
        assert all(u > 0 for u in uncertainties)
        assert np.mean(uncertainties) < 3600  # Average uncertainty < 1 hour


# Mark all tests as focusing on models
pytestmark = pytest.mark.asyncio
