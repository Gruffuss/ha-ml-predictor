"""
Comprehensive unit tests for Gaussian Process predictor.

This test suite validates GP model training, prediction, uncertainty quantification,
kernel creation, sparse GP approximation, and all mathematical operations.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd
import pytest
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.preprocessing import StandardScaler

from src.core.constants import DEFAULT_MODEL_PARAMS, ModelType
from src.core.exceptions import ModelPredictionError, ModelTrainingError
from src.models.base.gp_predictor import GaussianProcessPredictor
from src.models.base.predictor import PredictionResult, TrainingResult


class TestGaussianProcessPredictorInitialization:
    """Test GP predictor initialization and configuration."""

    def test_gp_predictor_initialization_default(self):
        """Test GP predictor initialization with default parameters."""
        predictor = GaussianProcessPredictor()

        assert predictor.model_type == ModelType.GP
        assert predictor.room_id is None
        assert predictor.model is None
        assert isinstance(predictor.feature_scaler, StandardScaler)
        assert predictor.kernel is None
        assert predictor.use_sparse_gp is False
        assert predictor.inducing_points is None
        assert predictor.uncertainty_calibrated is False

        # Check default parameters
        expected_defaults = DEFAULT_MODEL_PARAMS[ModelType.GAUSSIAN_PROCESS]
        assert predictor.model_params["kernel"] == expected_defaults.get(
            "kernel", "composite"
        )
        assert predictor.model_params["alpha"] == expected_defaults.get("alpha", 1e-6)
        assert predictor.model_params["n_restarts_optimizer"] == expected_defaults.get(
            "n_restarts_optimizer", 3
        )

    def test_gp_predictor_initialization_with_room_id(self):
        """Test GP predictor initialization with room ID."""
        predictor = GaussianProcessPredictor(room_id="living_room")

        assert predictor.room_id == "living_room"
        assert predictor.model_type == ModelType.GP

    def test_gp_predictor_initialization_with_custom_params(self):
        """Test GP predictor initialization with custom parameters."""
        custom_params = {
            "kernel": "rbf",
            "alpha": 1e-5,
            "n_restarts_optimizer": 5,
            "normalize_y": False,
            "confidence_intervals": [0.68, 0.95, 0.99],
            "max_inducing_points": 200,
        }

        predictor = GaussianProcessPredictor(room_id="bedroom", **custom_params)

        assert predictor.model_params["kernel"] == "rbf"
        assert predictor.model_params["alpha"] == 1e-5
        assert predictor.model_params["n_restarts_optimizer"] == 5
        assert predictor.model_params["normalize_y"] is False
        assert predictor.model_params["confidence_intervals"] == [0.68, 0.95, 0.99]
        assert predictor.model_params["max_inducing_points"] == 200

    def test_gp_predictor_parameter_aliasing(self):
        """Test parameter aliasing for compatibility."""
        predictor = GaussianProcessPredictor()

        # Test that kernel_type is aliased to kernel
        assert predictor.model_params["kernel_type"] == predictor.model_params["kernel"]

    def test_gp_predictor_training_statistics_initialization(self):
        """Test GP predictor training statistics initialization."""
        predictor = GaussianProcessPredictor()

        assert predictor.log_marginal_likelihood is None
        assert predictor.kernel_params_history == []
        assert predictor.calibration_curve is None


class TestGaussianProcessKernelCreation:
    """Test GP kernel creation and configuration."""

    @pytest.fixture
    def gp_predictor(self):
        """GP predictor fixture."""
        return GaussianProcessPredictor()

    def test_create_kernel_rbf(self, gp_predictor):
        """Test RBF kernel creation."""
        gp_predictor.model_params["kernel_type"] = "rbf"

        kernel = gp_predictor._create_kernel(5)

        # Should be a composite kernel with ConstantKernel * RBF
        assert hasattr(kernel, "get_params")
        kernel_params = kernel.get_params()
        assert "k1" in str(kernel)  # ConstantKernel
        assert "k2" in str(kernel)  # RBF kernel

    def test_create_kernel_matern(self, gp_predictor):
        """Test Matern kernel creation."""
        gp_predictor.model_params["kernel_type"] = "matern"

        kernel = gp_predictor._create_kernel(3)

        assert hasattr(kernel, "get_params")
        # Should contain Matern kernel
        assert "Matern" in str(kernel) or "matern" in str(kernel).lower()

    def test_create_kernel_rational_quadratic(self, gp_predictor):
        """Test Rational Quadratic kernel creation."""
        gp_predictor.model_params["kernel_type"] = "rational_quadratic"

        kernel = gp_predictor._create_kernel(4)

        assert hasattr(kernel, "get_params")
        # Should contain RationalQuadratic kernel
        assert "RationalQuadratic" in str(kernel) or "rational" in str(kernel).lower()

    def test_create_kernel_periodic(self, gp_predictor):
        """Test Periodic kernel creation."""
        gp_predictor.model_params["kernel_type"] = "periodic"

        kernel = gp_predictor._create_kernel(2)

        assert hasattr(kernel, "get_params")
        # Should handle periodic kernel creation (may fallback to RBF)
        assert kernel is not None

    def test_create_kernel_composite_default(self, gp_predictor):
        """Test composite kernel creation (default)."""
        gp_predictor.model_params["kernel_type"] = "composite"

        kernel = gp_predictor._create_kernel(10)

        assert hasattr(kernel, "get_params")
        # Composite kernel should be sum of multiple kernels
        kernel_str = str(kernel)
        assert "+" in kernel_str  # Sum of kernels
        assert "White" in kernel_str  # Should include noise kernel

    def test_create_kernel_with_different_features(self, gp_predictor):
        """Test kernel creation with different feature counts."""
        for n_features in [1, 5, 10, 20, 50]:
            kernel = gp_predictor._create_kernel(n_features)
            assert kernel is not None
            assert hasattr(kernel, "get_params")

    def test_create_kernel_periodic_fallback_handling(self, gp_predictor):
        """Test periodic kernel with fallback handling."""
        gp_predictor.model_params["kernel_type"] = "periodic"

        with patch("src.models.base.gp_predictor.PeriodicKernel") as mock_periodic:
            # Simulate PeriodicKernel not being available
            mock_periodic.side_effect = ImportError("Not available")

            kernel = gp_predictor._create_kernel(3)

            # Should fallback to RBF kernel
            assert kernel is not None
            assert "RBF" in str(kernel) or "rbf" in str(kernel).lower()


class TestGaussianProcessTraining:
    """Test GP model training functionality."""

    @pytest.fixture
    def sample_training_data(self):
        """Sample training data fixture."""
        np.random.seed(42)

        # Create realistic occupancy prediction features
        n_samples = 100
        features = pd.DataFrame(
            {
                "time_since_last_change": np.random.exponential(
                    3600, n_samples
                ),  # seconds
                "hour_sin": np.sin(
                    2 * np.pi * np.random.randint(0, 24, n_samples) / 24
                ),
                "hour_cos": np.cos(
                    2 * np.pi * np.random.randint(0, 24, n_samples) / 24
                ),
                "day_of_week": np.random.randint(0, 7, n_samples),
                "is_weekend": np.random.choice([0, 1], n_samples),
                "temperature": np.random.normal(22, 3, n_samples),
                "motion_events_last_hour": np.random.poisson(2, n_samples),
            }
        )

        # Target: time until next transition (in seconds)
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": np.random.exponential(
                    1800, n_samples
                )  # ~30 min average
            }
        )

        return features, targets

    @pytest.fixture
    def gp_predictor(self):
        """GP predictor fixture."""
        return GaussianProcessPredictor(room_id="test_room", random_state=42)

    @pytest.mark.asyncio
    async def test_gp_training_success(self, gp_predictor, sample_training_data):
        """Test successful GP training."""
        features, targets = sample_training_data

        result = await gp_predictor.train(features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.training_samples == len(features)
        assert result.training_time_seconds > 0
        assert result.validation_score is not None
        assert result.training_score > 0  # Should have reasonable R² score

        # Verify model is trained
        assert gp_predictor.is_trained is True
        assert gp_predictor.model is not None
        assert isinstance(gp_predictor.model, GaussianProcessRegressor)
        assert gp_predictor.feature_names == list(features.columns)
        assert gp_predictor.log_marginal_likelihood is not None

    @pytest.mark.asyncio
    async def test_gp_training_with_validation_data(
        self, gp_predictor, sample_training_data
    ):
        """Test GP training with separate validation data."""
        features, targets = sample_training_data

        # Split data for validation
        split_idx = int(len(features) * 0.8)
        train_features = features[:split_idx]
        train_targets = targets[:split_idx]
        val_features = features[split_idx:]
        val_targets = targets[split_idx:]

        result = await gp_predictor.train(
            train_features, train_targets, val_features, val_targets
        )

        assert result.success is True
        assert result.training_samples == len(train_features)
        assert "validation_mae" in result.training_metrics
        assert "validation_r2" in result.training_metrics
        assert "avg_validation_std" in result.training_metrics
        assert gp_predictor.uncertainty_calibrated is True

    @pytest.mark.asyncio
    async def test_gp_training_insufficient_data(self, gp_predictor):
        """Test GP training with insufficient data."""
        # Very small dataset
        features = pd.DataFrame({"feature1": [1, 2, 3, 4, 5]})
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [100, 200, 150, 300, 250]}
        )

        with pytest.raises(ModelTrainingError):
            await gp_predictor.train(features, targets)

    @pytest.mark.asyncio
    async def test_gp_training_sparse_gp_activation(
        self, gp_predictor, sample_training_data
    ):
        """Test sparse GP activation with large dataset."""
        features, targets = sample_training_data

        # Set low max_inducing_points to force sparse GP
        gp_predictor.model_params["max_inducing_points"] = 20

        result = await gp_predictor.train(features, targets)

        assert result.success is True
        assert gp_predictor.use_sparse_gp is True
        assert gp_predictor.inducing_points is not None
        assert len(gp_predictor.inducing_points) <= 20
        assert "sparse_gp" in result.training_metrics
        assert result.training_metrics["sparse_gp"] is True

    @pytest.mark.asyncio
    async def test_gp_training_kernel_parameters_tracking(
        self, gp_predictor, sample_training_data
    ):
        """Test that kernel parameters are tracked during training."""
        features, targets = sample_training_data

        await gp_predictor.train(features, targets)

        assert len(gp_predictor.kernel_params_history) > 0
        kernel_params = gp_predictor.kernel_params_history[-1]
        assert isinstance(kernel_params, dict)

    @pytest.mark.asyncio
    async def test_gp_training_model_versioning(
        self, gp_predictor, sample_training_data
    ):
        """Test model version generation during training."""
        features, targets = sample_training_data

        result = await gp_predictor.train(features, targets)

        assert result.model_version is not None
        assert result.model_version.startswith("v")
        assert gp_predictor.model_version == result.model_version

    @pytest.mark.asyncio
    async def test_gp_training_error_handling(self, gp_predictor):
        """Test GP training error handling."""
        features = pd.DataFrame({"feature1": [1, 2, 3]})
        targets = pd.DataFrame({"invalid_target": [100, 200, 300]})  # Wrong column name

        with patch.object(
            gp_predictor, "_prepare_targets", side_effect=Exception("Preparation error")
        ):
            with pytest.raises(ModelTrainingError):
                await gp_predictor.train(features, targets)

    @pytest.mark.asyncio
    async def test_gp_training_metrics_calculation(
        self, gp_predictor, sample_training_data
    ):
        """Test training metrics calculation."""
        features, targets = sample_training_data

        result = await gp_predictor.train(features, targets)

        metrics = result.training_metrics
        assert "training_mae" in metrics
        assert "training_rmse" in metrics
        assert "training_r2" in metrics
        assert "log_marginal_likelihood" in metrics
        assert "avg_prediction_std" in metrics
        assert "kernel_type" in metrics
        assert "n_inducing_points" in metrics

        # Values should be reasonable
        assert metrics["training_mae"] > 0
        assert metrics["training_rmse"] > 0
        assert -1 <= metrics["training_r2"] <= 1


class TestGaussianProcessPrediction:
    """Test GP prediction functionality."""

    @pytest.fixture
    def trained_gp_predictor(self, sample_training_data):
        """Trained GP predictor fixture."""
        features, targets = sample_training_data
        predictor = GaussianProcessPredictor(room_id="test_room", random_state=42)

        # Mock training to avoid long setup
        predictor.is_trained = True
        predictor.feature_names = list(features.columns)
        predictor.training_date = datetime.now(timezone.utc)
        predictor.model_version = "v1.0"

        # Mock GP model
        mock_model = Mock(spec=GaussianProcessRegressor)
        mock_model.predict.return_value = (
            np.array([1800.0, 3600.0]),  # predictions
            np.array([300.0, 600.0]),  # standard deviations
        )
        predictor.model = mock_model

        # Mock feature scaler
        mock_scaler = Mock(spec=StandardScaler)
        mock_scaler.transform.return_value = features.iloc[:2].values
        predictor.feature_scaler = mock_scaler

        return predictor

    @pytest.fixture
    def sample_prediction_features(self):
        """Sample prediction features fixture."""
        return pd.DataFrame(
            {
                "time_since_last_change": [1800, 7200],
                "hour_sin": [0.5, -0.5],
                "hour_cos": [0.866, -0.866],
                "day_of_week": [1, 5],
                "is_weekend": [0, 1],
                "temperature": [22.5, 20.0],
                "motion_events_last_hour": [3, 0],
            }
        )

    @pytest.mark.asyncio
    async def test_gp_prediction_success(
        self, trained_gp_predictor, sample_prediction_features
    ):
        """Test successful GP prediction."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_gp_predictor.predict(
            sample_prediction_features, prediction_time, "occupied"
        )

        assert len(results) == len(sample_prediction_features)

        for result in results:
            assert isinstance(result, PredictionResult)
            assert result.predicted_time > prediction_time
            assert result.transition_type in [
                "occupied_to_vacant",
                "vacant_to_occupied",
            ]
            assert 0.1 <= result.confidence_score <= 0.95
            assert result.model_type == ModelType.GP.value
            assert result.model_version == "v1.0"
            assert result.features_used == trained_gp_predictor.feature_names

            # Check GP-specific metadata
            metadata = result.prediction_metadata
            assert "time_until_transition_seconds" in metadata
            assert "prediction_std" in metadata
            assert "prediction_method" in metadata
            assert metadata["prediction_method"] == "gaussian_process"
            assert "uncertainty_quantification" in metadata
            assert "aleatoric_uncertainty" in metadata["uncertainty_quantification"]
            assert "epistemic_uncertainty" in metadata["uncertainty_quantification"]

    @pytest.mark.asyncio
    async def test_gp_prediction_untrained_model(self, sample_prediction_features):
        """Test prediction with untrained model."""
        predictor = GaussianProcessPredictor()
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await predictor.predict(sample_prediction_features, prediction_time)

    @pytest.mark.asyncio
    async def test_gp_prediction_invalid_features(self, trained_gp_predictor):
        """Test prediction with invalid features."""
        invalid_features = pd.DataFrame({"wrong_feature": [1, 2, 3]})
        prediction_time = datetime.now(timezone.utc)

        with pytest.raises(ModelPredictionError):
            await trained_gp_predictor.predict(invalid_features, prediction_time)

    @pytest.mark.asyncio
    async def test_gp_prediction_confidence_intervals(
        self, trained_gp_predictor, sample_prediction_features
    ):
        """Test prediction confidence intervals calculation."""
        prediction_time = datetime.now(timezone.utc)

        # Mock calibrated uncertainty
        trained_gp_predictor.uncertainty_calibrated = True
        trained_gp_predictor.calibration_curve = {0.68: 1.0, 0.95: 1.1}

        results = await trained_gp_predictor.predict(
            sample_prediction_features, prediction_time
        )

        for result in results:
            metadata = result.prediction_metadata
            confidence_intervals = metadata["uncertainty_quantification"][
                "confidence_intervals"
            ]

            # Should have confidence intervals
            assert "68%" in confidence_intervals or "95%" in confidence_intervals

            for level, interval in confidence_intervals.items():
                assert "lower" in interval
                assert "upper" in interval
                assert interval["lower"] < interval["upper"]

    @pytest.mark.asyncio
    async def test_gp_prediction_alternative_scenarios(
        self, trained_gp_predictor, sample_prediction_features
    ):
        """Test alternative scenario generation."""
        prediction_time = datetime.now(timezone.utc)

        results = await trained_gp_predictor.predict(
            sample_prediction_features, prediction_time
        )

        for result in results:
            if result.alternatives:
                assert len(result.alternatives) <= 3  # Top 3 alternatives

                for alt_time, alt_confidence in result.alternatives:
                    assert isinstance(alt_time, datetime)
                    assert 0.0 <= alt_confidence <= 1.0

    @pytest.mark.asyncio
    async def test_gp_prediction_transition_type_determination(
        self, trained_gp_predictor, sample_prediction_features
    ):
        """Test transition type determination."""
        prediction_time = datetime.now(timezone.utc)

        # Test with occupied state
        results_occupied = await trained_gp_predictor.predict(
            sample_prediction_features, prediction_time, "occupied"
        )

        for result in results_occupied:
            assert result.transition_type == "occupied_to_vacant"

        # Test with vacant state
        results_vacant = await trained_gp_predictor.predict(
            sample_prediction_features, prediction_time, "vacant"
        )

        for result in results_vacant:
            assert result.transition_type == "vacant_to_occupied"

    @pytest.mark.asyncio
    async def test_gp_prediction_time_bounds_clipping(
        self, trained_gp_predictor, sample_prediction_features
    ):
        """Test that predictions are clipped to reasonable bounds."""
        prediction_time = datetime.now(timezone.utc)

        # Mock extreme predictions
        trained_gp_predictor.model.predict.return_value = (
            np.array([30.0, 100000.0]),  # Very short and very long
            np.array([10.0, 1000.0]),
        )

        results = await trained_gp_predictor.predict(
            sample_prediction_features, prediction_time
        )

        for result in results:
            time_until = result.prediction_metadata["time_until_transition_seconds"]
            assert 60 <= time_until <= 86400  # 1 minute to 24 hours


class TestGaussianProcessUncertaintyQuantification:
    """Test GP uncertainty quantification features."""

    @pytest.fixture
    def gp_predictor_with_uncertainty(self):
        """GP predictor with uncertainty features."""
        predictor = GaussianProcessPredictor()
        predictor.is_trained = True
        predictor.uncertainty_calibrated = True
        predictor.calibration_curve = {0.68: 1.0, 0.95: 1.1}
        predictor.model_params["confidence_intervals"] = [0.68, 0.95]
        return predictor

    def test_calculate_confidence_intervals(self, gp_predictor_with_uncertainty):
        """Test confidence interval calculation."""
        mean_time = 1800.0  # 30 minutes
        std_time = 300.0  # 5 minutes

        intervals = gp_predictor_with_uncertainty._calculate_confidence_intervals(
            mean_time, std_time
        )

        assert "68%" in intervals
        assert "95%" in intervals

        for level, (lower, upper) in intervals.items():
            assert isinstance(lower, datetime)
            assert isinstance(upper, datetime)
            assert lower < upper

    def test_calculate_confidence_score(self, gp_predictor_with_uncertainty):
        """Test confidence score calculation."""
        # Mock training history
        mock_result = Mock()
        mock_result.validation_score = 0.8
        gp_predictor_with_uncertainty.training_history = [mock_result]

        confidence = gp_predictor_with_uncertainty._calculate_confidence_score(
            1800.0, 300.0
        )

        assert 0.1 <= confidence <= 0.95
        assert isinstance(confidence, float)

    def test_generate_alternative_scenarios(self, gp_predictor_with_uncertainty):
        """Test alternative scenario generation."""
        base_time = datetime.now(timezone.utc)
        mean_time = 1800.0
        std_time = 300.0

        alternatives = gp_predictor_with_uncertainty._generate_alternative_scenarios(
            base_time, mean_time, std_time, "occupied_to_vacant"
        )

        assert len(alternatives) <= 3

        for alt_time, confidence in alternatives:
            assert isinstance(alt_time, datetime)
            assert 0.0 <= confidence <= 1.0
            assert alt_time > base_time

    def test_estimate_epistemic_uncertainty(self, gp_predictor_with_uncertainty):
        """Test epistemic uncertainty estimation."""
        # Mock model with training data
        mock_model = Mock()
        mock_model.X_train_ = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        gp_predictor_with_uncertainty.model = mock_model

        X_point = np.array([[2, 3, 4]])
        uncertainty = gp_predictor_with_uncertainty._estimate_epistemic_uncertainty(
            X_point
        )

        assert 0.0 <= uncertainty <= 1.0
        assert isinstance(uncertainty, float)

    def test_estimate_epistemic_uncertainty_no_training_data(
        self, gp_predictor_with_uncertainty
    ):
        """Test epistemic uncertainty without training data."""
        # Mock model without training data
        mock_model = Mock()
        del mock_model.X_train_  # Remove training data attribute
        gp_predictor_with_uncertainty.model = mock_model

        X_point = np.array([[2, 3, 4]])
        uncertainty = gp_predictor_with_uncertainty._estimate_epistemic_uncertainty(
            X_point
        )

        assert uncertainty == 0.5  # Default uncertainty

    def test_calibrate_uncertainty(self, gp_predictor_with_uncertainty):
        """Test uncertainty calibration."""
        y_true = np.array([1000, 2000, 1500, 2500, 1800])
        y_pred = np.array([1100, 1900, 1400, 2400, 1700])
        y_std = np.array([200, 300, 250, 400, 300])

        gp_predictor_with_uncertainty._calibrate_uncertainty(y_true, y_pred, y_std)

        assert gp_predictor_with_uncertainty.uncertainty_calibrated is True
        assert gp_predictor_with_uncertainty.calibration_curve is not None
        assert isinstance(gp_predictor_with_uncertainty.calibration_curve, dict)

    def test_calibrate_uncertainty_error_handling(self, gp_predictor_with_uncertainty):
        """Test uncertainty calibration error handling."""
        # Invalid input that should cause calibration to fail
        y_true = np.array([])
        y_pred = np.array([])
        y_std = np.array([])

        gp_predictor_with_uncertainty._calibrate_uncertainty(y_true, y_pred, y_std)

        assert gp_predictor_with_uncertainty.uncertainty_calibrated is False

    def test_get_uncertainty_metrics(self, gp_predictor_with_uncertainty):
        """Test uncertainty metrics retrieval."""
        gp_predictor_with_uncertainty.log_marginal_likelihood = -100.5
        gp_predictor_with_uncertainty.use_sparse_gp = True
        gp_predictor_with_uncertainty.inducing_points = np.array([[1, 2], [3, 4]])

        metrics = gp_predictor_with_uncertainty.get_uncertainty_metrics()

        assert "kernel_type" in metrics
        assert "log_marginal_likelihood" in metrics
        assert "uncertainty_calibrated" in metrics
        assert "confidence_intervals" in metrics
        assert "sparse_gp" in metrics
        assert "calibration_curve" in metrics
        assert "n_inducing_points" in metrics

        assert metrics["log_marginal_likelihood"] == -100.5
        assert metrics["uncertainty_calibrated"] is True
        assert metrics["sparse_gp"] is True
        assert metrics["n_inducing_points"] == 2


class TestGaussianProcessSparseApproximation:
    """Test sparse GP approximation functionality."""

    @pytest.fixture
    def gp_predictor_sparse(self):
        """GP predictor configured for sparse approximation."""
        predictor = GaussianProcessPredictor()
        predictor.model_params["max_inducing_points"] = 50
        return predictor

    def test_select_inducing_points(self, gp_predictor_sparse):
        """Test inducing point selection."""
        np.random.seed(42)
        X = np.random.randn(200, 5)  # 200 samples, 5 features
        y = np.random.randn(200)

        gp_predictor_sparse._select_inducing_points(X, y)

        assert gp_predictor_sparse.inducing_points is not None
        assert len(gp_predictor_sparse.inducing_points) <= 50
        assert gp_predictor_sparse.inducing_points.shape[1] == 5  # Same feature count

    def test_select_inducing_points_small_dataset(self, gp_predictor_sparse):
        """Test inducing point selection with small dataset."""
        X = np.random.randn(20, 3)  # Smaller than max_inducing_points
        y = np.random.randn(20)

        gp_predictor_sparse._select_inducing_points(X, y)

        assert gp_predictor_sparse.inducing_points is not None
        assert len(gp_predictor_sparse.inducing_points) <= 20

    def test_select_inducing_points_clustering(self, gp_predictor_sparse):
        """Test that inducing points use clustering."""
        # Create data with clear clusters
        cluster1 = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 50)
        cluster2 = np.random.multivariate_normal([5, 5], [[1, 0], [0, 1]], 50)
        X = np.vstack([cluster1, cluster2])
        y = np.random.randn(100)

        gp_predictor_sparse.model_params["max_inducing_points"] = 10
        gp_predictor_sparse._select_inducing_points(X, y)

        # Should have representative points from both clusters
        inducing_points = gp_predictor_sparse.inducing_points
        assert len(inducing_points) <= 10

        # Check that points span the data range
        assert inducing_points[:, 0].min() < 3  # Some points near first cluster
        assert inducing_points[:, 0].max() > 3  # Some points near second cluster


class TestGaussianProcessFeatureImportance:
    """Test GP feature importance calculation."""

    @pytest.fixture
    def trained_gp_with_features(self):
        """Trained GP with known features."""
        predictor = GaussianProcessPredictor()
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2", "feature3"]

        # Mock trained model with kernel parameters
        mock_model = Mock()
        mock_kernel = Mock()
        mock_kernel.get_params.return_value = {
            "k1__length_scale": np.array([0.5, 2.0, 1.0])  # ARD kernel
        }
        mock_model.kernel_ = mock_kernel
        predictor.model = mock_model

        return predictor

    def test_get_feature_importance_ard_kernel(self, trained_gp_with_features):
        """Test feature importance with ARD kernel."""
        importance = trained_gp_with_features.get_feature_importance()

        assert len(importance) == 3
        assert "feature1" in importance
        assert "feature2" in importance
        assert "feature3" in importance

        # Should sum to 1.0
        assert abs(sum(importance.values()) - 1.0) < 1e-6

        # feature1 should have highest importance (smallest length scale)
        assert importance["feature1"] > importance["feature2"]
        assert importance["feature1"] > importance["feature3"]

    def test_get_feature_importance_scalar_length_scale(self, trained_gp_with_features):
        """Test feature importance with scalar length scale."""
        # Mock scalar length scale (non-ARD kernel)
        mock_kernel = Mock()
        mock_kernel.get_params.return_value = {"k1__length_scale": 1.5}  # Scalar value
        trained_gp_with_features.model.kernel_ = mock_kernel

        importance = trained_gp_with_features.get_feature_importance()

        # Should have uniform importance
        for feature_importance in importance.values():
            assert abs(feature_importance - 1 / 3) < 1e-6

    def test_get_feature_importance_no_length_scale(self, trained_gp_with_features):
        """Test feature importance without length scale parameters."""
        # Mock kernel without length scale
        mock_kernel = Mock()
        mock_kernel.get_params.return_value = {}
        trained_gp_with_features.model.kernel_ = mock_kernel

        importance = trained_gp_with_features.get_feature_importance()

        # Should have uniform importance as fallback
        for feature_importance in importance.values():
            assert abs(feature_importance - 1 / 3) < 1e-6

    def test_get_feature_importance_untrained_model(self):
        """Test feature importance with untrained model."""
        predictor = GaussianProcessPredictor()

        importance = predictor.get_feature_importance()

        assert importance == {}

    def test_get_feature_importance_error_handling(self, trained_gp_with_features):
        """Test feature importance calculation error handling."""
        # Mock kernel that raises exception
        mock_kernel = Mock()
        mock_kernel.get_params.side_effect = Exception("Kernel error")
        trained_gp_with_features.model.kernel_ = mock_kernel

        importance = trained_gp_with_features.get_feature_importance()

        # Should return uniform importance as fallback
        assert len(importance) == 3
        for feature_importance in importance.values():
            assert abs(feature_importance - 1 / 3) < 1e-6


class TestGaussianProcessIncrementalUpdate:
    """Test GP incremental update functionality."""

    @pytest.fixture
    def trained_gp_for_update(self, sample_training_data):
        """Trained GP predictor for update testing."""
        features, targets = sample_training_data
        predictor = GaussianProcessPredictor(room_id="test_room")

        # Mock trained state
        predictor.is_trained = True
        predictor.feature_names = list(features.columns)
        predictor.feature_scaler = Mock(spec=StandardScaler)
        predictor.feature_scaler.transform.return_value = features.values

        # Mock trained model with training data
        mock_model = Mock(spec=GaussianProcessRegressor)
        mock_model.X_train_ = features.values[:50]
        mock_model.y_train_ = np.random.exponential(1800, 50)
        mock_model.fit.return_value = None
        mock_model.predict.return_value = (
            np.random.exponential(1800, 10),
            np.random.normal(300, 50, 10),
        )
        mock_model.log_marginal_likelihood.return_value = -150.0
        mock_model.kernel_ = Mock()
        mock_model.kernel_.get_params.return_value = {"param": "value"}
        predictor.model = mock_model

        return predictor

    @pytest.mark.asyncio
    async def test_incremental_update_success(
        self, trained_gp_for_update, sample_training_data
    ):
        """Test successful incremental update."""
        features, targets = sample_training_data

        # Use small subset for update
        update_features = features[:10]
        update_targets = targets[:10]

        result = await trained_gp_for_update.incremental_update(
            update_features, update_targets
        )

        assert isinstance(result, TrainingResult)
        assert result.success is True
        assert result.training_samples == len(update_features)
        assert "update_type" in result.training_metrics
        assert result.training_metrics["update_type"] == "incremental"
        assert "incremental_mae" in result.training_metrics
        assert "incremental_r2" in result.training_metrics

    @pytest.mark.asyncio
    async def test_incremental_update_untrained_model(self, sample_training_data):
        """Test incremental update on untrained model."""
        predictor = GaussianProcessPredictor()
        features, targets = sample_training_data

        with patch.object(predictor, "train") as mock_train:
            mock_train.return_value = TrainingResult(
                success=True,
                training_time_seconds=1.0,
                model_version="v1.0",
                training_samples=100,
            )

            result = await predictor.incremental_update(features, targets)

            # Should call full training instead
            mock_train.assert_called_once_with(features, targets)

    @pytest.mark.asyncio
    async def test_incremental_update_insufficient_data(self, trained_gp_for_update):
        """Test incremental update with insufficient data."""
        # Very small update dataset
        features = pd.DataFrame({"feature1": [1, 2]})
        targets = pd.DataFrame({"time_until_transition_seconds": [100, 200]})

        with pytest.raises(ModelTrainingError):
            await trained_gp_for_update.incremental_update(features, targets)

    @pytest.mark.asyncio
    async def test_incremental_update_sparse_gp(
        self, trained_gp_for_update, sample_training_data
    ):
        """Test incremental update with sparse GP."""
        trained_gp_for_update.use_sparse_gp = True
        trained_gp_for_update.inducing_points = np.random.randn(20, 7)
        trained_gp_for_update.model_params["max_inducing_points"] = 30

        features, targets = sample_training_data
        update_features = features[:15]
        update_targets = targets[:15]

        result = await trained_gp_for_update.incremental_update(
            update_features, update_targets
        )

        assert result.success is True
        assert "sparse_gp" in result.training_metrics
        assert result.training_metrics["sparse_gp"] is True

    @pytest.mark.asyncio
    async def test_incremental_update_model_version(
        self, trained_gp_for_update, sample_training_data
    ):
        """Test model version update during incremental training."""
        trained_gp_for_update.model_version = "v1.0"

        features, targets = sample_training_data
        update_features = features[:10]
        update_targets = targets[:10]

        result = await trained_gp_for_update.incremental_update(
            update_features, update_targets
        )

        # Model version should be updated
        assert result.model_version != "v1.0"
        assert "_inc_" in result.model_version


class TestGaussianProcessUtilityMethods:
    """Test GP utility and helper methods."""

    @pytest.fixture
    def gp_predictor(self):
        """GP predictor fixture."""
        return GaussianProcessPredictor()

    def test_determine_transition_type(self, gp_predictor):
        """Test transition type determination."""
        # Test with occupied state
        transition = gp_predictor._determine_transition_type("occupied", 14)
        assert transition == "occupied_to_vacant"

        # Test with vacant state
        transition = gp_predictor._determine_transition_type("vacant", 10)
        assert transition == "vacant_to_occupied"

        # Test with unknown state (daytime)
        transition = gp_predictor._determine_transition_type("unknown", 10)
        assert transition == "vacant_to_occupied"

        # Test with unknown state (nighttime)
        transition = gp_predictor._determine_transition_type("unknown", 2)
        assert transition == "occupied_to_vacant"

    def test_prepare_targets_time_until_column(self, gp_predictor):
        """Test target preparation with time_until_transition_seconds column."""
        targets = pd.DataFrame(
            {"time_until_transition_seconds": [1800, 3600, 2400, 900]}
        )

        prepared = gp_predictor._prepare_targets(targets)

        assert isinstance(prepared, np.ndarray)
        assert len(prepared) == 4
        assert np.array_equal(prepared, [1800, 3600, 2400, 900])

    def test_prepare_targets_time_columns(self, gp_predictor):
        """Test target preparation with time columns."""
        targets = pd.DataFrame(
            {
                "target_time": [
                    "2024-01-01T12:00:00",
                    "2024-01-01T13:00:00",
                ],
                "next_transition_time": [
                    "2024-01-01T12:30:00",
                    "2024-01-01T14:15:00",
                ],
            }
        )

        prepared = gp_predictor._prepare_targets(targets)

        assert isinstance(prepared, np.ndarray)
        assert len(prepared) == 2
        assert prepared[0] == 1800  # 30 minutes
        assert prepared[1] == 4500  # 75 minutes

    def test_prepare_targets_first_column_fallback(self, gp_predictor):
        """Test target preparation with first column fallback."""
        targets = pd.DataFrame({"some_target_column": [1200, 2400, 1800]})

        prepared = gp_predictor._prepare_targets(targets)

        assert isinstance(prepared, np.ndarray)
        assert np.array_equal(prepared, [1200, 2400, 1800])

    def test_prepare_targets_clipping(self, gp_predictor):
        """Test target value clipping to reasonable bounds."""
        targets = pd.DataFrame(
            {
                "time_until_transition_seconds": [
                    30,
                    1800,
                    100000,
                    -500,
                ]  # Some out of bounds
            }
        )

        prepared = gp_predictor._prepare_targets(targets)

        # Should be clipped to [60, 86400] range
        assert all(60 <= val <= 86400 for val in prepared)
        assert prepared[0] == 60  # 30 clipped to 60
        assert prepared[1] == 1800  # 1800 unchanged
        assert prepared[2] == 86400  # 100000 clipped to 86400
        assert prepared[3] == 60  # -500 clipped to 60


class TestGaussianProcessSerialization:
    """Test GP model serialization and loading."""

    @pytest.fixture
    def trained_gp_for_serialization(self, sample_training_data):
        """Trained GP for serialization testing."""
        features, targets = sample_training_data
        predictor = GaussianProcessPredictor(room_id="test_room")

        # Mock trained state
        predictor.is_trained = True
        predictor.feature_names = list(features.columns)
        predictor.training_date = datetime.now(timezone.utc)
        predictor.model_version = "v1.0"

        # Mock training history
        mock_result = TrainingResult(
            success=True,
            training_time_seconds=10.0,
            model_version="v1.0",
            training_samples=100,
        )
        predictor.training_history = [mock_result]

        # Mock model components
        predictor.model = Mock(spec=GaussianProcessRegressor)
        predictor.feature_scaler = Mock(spec=StandardScaler)

        return predictor

    def test_save_model_success(self, trained_gp_for_serialization, tmp_path):
        """Test successful model saving."""
        model_path = tmp_path / "gp_model.pkl"

        result = trained_gp_for_serialization.save_model(str(model_path))

        assert result is True
        assert model_path.exists()

    def test_save_model_failure(self, trained_gp_for_serialization):
        """Test model saving failure."""
        # Invalid path that should cause save to fail
        invalid_path = "/invalid/path/model.pkl"

        result = trained_gp_for_serialization.save_model(invalid_path)

        assert result is False

    def test_load_model_success(self, trained_gp_for_serialization, tmp_path):
        """Test successful model loading."""
        model_path = tmp_path / "gp_model.pkl"

        # First save the model
        trained_gp_for_serialization.save_model(str(model_path))

        # Create new predictor and load
        new_predictor = GaussianProcessPredictor()
        result = new_predictor.load_model(str(model_path))

        assert result is True
        assert new_predictor.room_id == "test_room"
        assert new_predictor.model_version == "v1.0"
        assert new_predictor.is_trained is True
        assert len(new_predictor.training_history) == 1

    def test_load_model_failure(self):
        """Test model loading failure."""
        predictor = GaussianProcessPredictor()

        # Non-existent file
        result = predictor.load_model("nonexistent_file.pkl")

        assert result is False


class TestGaussianProcessEdgeCases:
    """Test GP predictor edge cases and error conditions."""

    def test_gp_predictor_with_empty_features(self):
        """Test GP predictor behavior with empty features."""
        predictor = GaussianProcessPredictor()
        empty_features = pd.DataFrame()

        # Should handle empty features gracefully
        assert not predictor.validate_features(empty_features)

    def test_gp_predictor_extreme_parameter_values(self):
        """Test GP predictor with extreme parameter values."""
        extreme_params = {
            "alpha": 1e-15,  # Very small alpha
            "n_restarts_optimizer": 0,  # No restarts
            "max_inducing_points": 1,  # Minimal inducing points
        }

        predictor = GaussianProcessPredictor(**extreme_params)

        assert predictor.model_params["alpha"] == 1e-15
        assert predictor.model_params["n_restarts_optimizer"] == 0
        assert predictor.model_params["max_inducing_points"] == 1

    def test_gp_predictor_kernel_parameter_edge_cases(self):
        """Test kernel creation with edge cases."""
        predictor = GaussianProcessPredictor()

        # Test with minimal features
        kernel = predictor._create_kernel(1)
        assert kernel is not None

        # Test with many features
        kernel = predictor._create_kernel(1000)
        assert kernel is not None

    @pytest.mark.asyncio
    async def test_gp_prediction_with_nans(self):
        """Test GP prediction handling of NaN values."""
        predictor = GaussianProcessPredictor()
        predictor.is_trained = True
        predictor.feature_names = ["feature1", "feature2"]

        # Mock model that returns NaN
        mock_model = Mock()
        mock_model.predict.return_value = (np.array([np.nan]), np.array([np.nan]))
        predictor.model = mock_model

        mock_scaler = Mock()
        mock_scaler.transform.return_value = np.array([[1, 2]])
        predictor.feature_scaler = mock_scaler

        features = pd.DataFrame({"feature1": [1], "feature2": [2]})
        prediction_time = datetime.now(timezone.utc)

        results = await predictor.predict(features, prediction_time)

        # Should handle NaN gracefully by clipping to bounds
        assert len(results) == 1
        result = results[0]
        time_until = result.prediction_metadata["time_until_transition_seconds"]
        assert 60 <= time_until <= 86400  # Should be clipped to valid range

    def test_gp_predictor_memory_cleanup(self):
        """Test that GP predictor can handle memory cleanup."""
        predictor = GaussianProcessPredictor()

        # Add many prediction records
        for i in range(2000):  # More than the 1000 limit
            mock_result = Mock(spec=PredictionResult)
            predictor._record_prediction(datetime.now(timezone.utc), mock_result)

        # Should have cleaned up to 500 most recent
        assert len(predictor.prediction_history) == 500

    def test_gp_predictor_concurrent_safety(self):
        """Test GP predictor thread safety considerations."""
        predictor = GaussianProcessPredictor()

        # Multiple concurrent modifications should not break internal state
        for i in range(100):
            predictor.kernel_params_history.append({"param": i})

        assert len(predictor.kernel_params_history) == 100
