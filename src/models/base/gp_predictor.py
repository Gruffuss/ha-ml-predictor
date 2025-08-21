"""
Gaussian Process Predictor for occupancy prediction with uncertainty quantification.

This module implements a Gaussian Process predictor that provides probabilistic
predictions with confidence intervals and uncertainty quantification for occupancy
state transitions.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel as C,
    Matern,
    RationalQuadratic,
    WhiteKernel,
)

# Try to import PeriodicKernel, fallback to ExpSineSquared if not available
try:
    from sklearn.gaussian_process.kernels import PeriodicKernel
except ImportError:
    try:
        from sklearn.gaussian_process.kernels import (
            ExpSineSquared as PeriodicKernel,
        )
    except ImportError:
        # Create a placeholder class if neither is available
        class PeriodicKernel:
            def __init__(self, *args, **kwargs):
                # Fallback to RBF if no periodic kernel available
                self.kernel = C(1.0) * RBF(length_scale=1.0)

            def __call__(self, *args, **kwargs):
                return self.kernel(*args, **kwargs)


from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

from ...core.constants import DEFAULT_MODEL_PARAMS, ModelType
from ...core.exceptions import ModelPredictionError, ModelTrainingError
from .predictor import BasePredictor, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class GaussianProcessPredictor(BasePredictor):
    """
    Gaussian Process predictor for probabilistic occupancy predictions.

    This predictor uses Gaussian Processes to provide not only predictions but also
    uncertainty quantification through confidence intervals and probabilistic outputs.
    It's particularly useful for understanding prediction reliability and handling
    temporal patterns in occupancy data.
    """

    def __init__(self, room_id: Optional[str] = None, **kwargs):
        """
        Initialize the Gaussian Process predictor.

        Args:
            room_id: Specific room this model is for
            **kwargs: Additional parameters for GP configuration
        """
        super().__init__(ModelType.GP, room_id)

        # Default GP parameters
        default_params = DEFAULT_MODEL_PARAMS[ModelType.GAUSSIAN_PROCESS].copy()
        default_params.update(kwargs)

        self.model_params = {
            "kernel": default_params.get(
                "kernel", "composite"
            ),  # Primary parameter name
            "kernel_type": default_params.get("kernel", "composite"),  # Internal alias
            "alpha": default_params.get("alpha", 1e-6),
            "n_restarts_optimizer": default_params.get("n_restarts_optimizer", 3),
            "normalize_y": default_params.get("normalize_y", True),
            "copy_X_train": default_params.get("copy_X_train", True),
            "random_state": default_params.get("random_state", 42),
            # Uncertainty quantification parameters
            "confidence_intervals": default_params.get(
                "confidence_intervals", [0.68, 0.95]
            ),
            "uncertainty_threshold": default_params.get("uncertainty_threshold", 0.5),
            "max_inducing_points": default_params.get("max_inducing_points", 500),
        }

        # Model components
        self.model: Optional[GaussianProcessRegressor] = None
        self.feature_scaler = StandardScaler()
        self.kernel = None

        # Inducing points for sparse GP (computational efficiency)
        self.use_sparse_gp = False
        self.inducing_points = None

        # Uncertainty calibration
        self.uncertainty_calibrated = False
        self.calibration_curve = None

        # Training statistics
        self.log_marginal_likelihood = None
        self.kernel_params_history = []

    def _create_kernel(self, n_features: int) -> Any:
        """
        Create a composite kernel suitable for occupancy prediction.

        Args:
            n_features: Number of input features

        Returns:
            Configured sklearn kernel for the GP
        """
        kernel_type = self.model_params["kernel_type"]

        if kernel_type == "rb":
            # Simple RBF kernel
            kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
            )

        elif kernel_type == "matern":
            # Matern kernel (good for non-smooth functions)
            kernel = C(1.0, (1e-3, 1e3)) * Matern(
                length_scale=1.0, length_scale_bounds=(1e-2, 1e2), nu=1.5
            )

        elif kernel_type == "periodic":
            # Periodic kernel for daily/weekly patterns
            try:
                # Try PeriodicKernel first (newer scikit-learn versions)
                if hasattr(PeriodicKernel, "__module__") and "placeholder" not in str(
                    PeriodicKernel
                ):
                    kernel = C(1.0, (1e-3, 1e3)) * PeriodicKernel(
                        periodicity=24.0,  # 24 hour cycle
                        length_scale=1.0,
                        periodicity_bounds=(12.0, 168.0),  # 12 hours to 1 week
                        length_scale_bounds=(1e-2, 1e2),
                    )
                else:
                    # Fallback to ExpSineSquared with different parameter names
                    kernel = C(1.0, (1e-3, 1e3)) * PeriodicKernel(
                        length_scale=1.0,
                        periodicity=24.0,
                        length_scale_bounds=(1e-2, 1e2),
                        periodicity_bounds=(12.0, 168.0),
                    )
            except (TypeError, AttributeError):
                # Ultimate fallback to RBF kernel
                logger.warning(
                    "Periodic kernel not available, using RBF kernel instead"
                )
                kernel = C(1.0, (1e-3, 1e3)) * RBF(
                    length_scale=1.0, length_scale_bounds=(1e-2, 1e2)
                )

        elif kernel_type == "rational_quadratic":
            # Rational quadratic (scale mixture of RBF kernels)
            kernel = C(1.0, (1e-3, 1e3)) * RationalQuadratic(
                length_scale=1.0,
                alpha=1.0,
                length_scale_bounds=(1e-2, 1e2),
                alpha_bounds=(1e-5, 1e15),
            )

        else:  # composite (default)
            # Composite kernel combining multiple scales and patterns

            # 1. Short-term local variations
            local_kernel = C(1.0, (1e-3, 1e3)) * RBF(
                length_scale=0.5, length_scale_bounds=(1e-2, 2.0)
            )

            # 2. Medium-term trends
            trend_kernel = C(0.5, (1e-3, 1e3)) * RBF(
                length_scale=5.0, length_scale_bounds=(1.0, 20.0)
            )

            # 3. Daily periodic patterns
            try:
                daily_kernel = C(0.3, (1e-3, 1e3)) * PeriodicKernel(
                    periodicity=24.0,
                    length_scale=2.0,
                    periodicity_bounds=(20.0, 28.0),  # Around 24 hours
                    length_scale_bounds=(0.5, 10.0),
                )
            except (TypeError, AttributeError):
                logger.warning(
                    "Using RBF kernel for daily patterns (periodic kernel unavailable)"
                )
                daily_kernel = C(0.3, (1e-3, 1e3)) * RBF(
                    length_scale=2.0, length_scale_bounds=(0.5, 10.0)
                )

            # 4. Weekly periodic patterns
            try:
                weekly_kernel = C(0.2, (1e-3, 1e3)) * PeriodicKernel(
                    periodicity=168.0,  # 7 * 24 hours
                    length_scale=10.0,
                    periodicity_bounds=(144.0, 192.0),  # Around 1 week
                    length_scale_bounds=(5.0, 50.0),
                )
            except (TypeError, AttributeError):
                logger.warning(
                    "Using RBF kernel for weekly patterns (periodic kernel unavailable)"
                )
                weekly_kernel = C(0.2, (1e-3, 1e3)) * RBF(
                    length_scale=10.0, length_scale_bounds=(5.0, 50.0)
                )

            # 5. Noise kernel
            noise_kernel = WhiteKernel(
                noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)
            )

            # Combine all kernels
            kernel = (
                local_kernel
                + trend_kernel
                + daily_kernel
                + weekly_kernel
                + noise_kernel
            )

        logger.info(f"Created {kernel_type} kernel: {kernel}")
        return kernel

    async def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame] = None,
        validation_targets: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """
        Train the Gaussian Process model on occupancy data.

        Args:
            features: Training feature matrix
            targets: Training target values (time until next transition)
            validation_features: Optional validation features
            validation_targets: Optional validation targets

        Returns:
            TrainingResult with GP training statistics
        """
        start_time = datetime.now(timezone.utc)

        try:
            logger.info(f"Starting GP training for room {self.room_id}")
            logger.info(f"Training data shape: {features.shape}")

            # Prepare target values
            y_train = self._prepare_targets(targets)

            if len(features) < 10:
                raise ModelTrainingError(
                    model_type="gaussian_process",
                    room_id=self.room_id,
                    cause=ValueError(
                        f"Insufficient training data: only {len(features)} samples available"
                    ),
                )

            # Scale features
            X_scaled = self.feature_scaler.fit_transform(features)

            # Implement sparse GP if dataset is large
            if len(X_scaled) > self.model_params["max_inducing_points"]:
                self.use_sparse_gp = True
                self._select_inducing_points(X_scaled, y_train)
                logger.info(
                    f"Using sparse GP with {len(self.inducing_points)} inducing points"
                )

            # Create kernel
            self.kernel = self._create_kernel(features.shape[1])

            # Create and configure GP model
            self.model = GaussianProcessRegressor(
                kernel=self.kernel,
                alpha=self.model_params["alpha"],
                optimizer="fmin_l_bfgs_b",
                n_restarts_optimizer=self.model_params["n_restarts_optimizer"],
                normalize_y=self.model_params["normalize_y"],
                copy_X_train=self.model_params["copy_X_train"],
                random_state=self.model_params["random_state"],
            )

            # Train the model
            if self.use_sparse_gp:
                # Use inducing points for training (approximation)
                self.model.fit(
                    self.inducing_points, y_train[: len(self.inducing_points)]
                )
            else:
                self.model.fit(X_scaled, y_train)

            # Store training information
            self.feature_names = list(features.columns)
            self.log_marginal_likelihood = self.model.log_marginal_likelihood()
            self.kernel_params_history.append(self.model.kernel_.get_params())

            # Calculate training metrics
            if self.use_sparse_gp:
                y_pred_mean, y_pred_std = self.model.predict(X_scaled, return_std=True)
            else:
                y_pred_mean, y_pred_std = self.model.predict(X_scaled, return_std=True)

            training_score = r2_score(y_train, y_pred_mean)
            training_mae = mean_absolute_error(y_train, y_pred_mean)
            training_rmse = np.sqrt(mean_squared_error(y_train, y_pred_mean))

            # Average prediction uncertainty (standard deviation)
            avg_prediction_std = np.mean(y_pred_std)

            # Calculate validation metrics if provided
            validation_score = None
            validation_mae = None
            validation_rmse = None
            avg_validation_std = None

            if validation_features is not None and validation_targets is not None:
                X_val_scaled = self.feature_scaler.transform(validation_features)
                y_val_true = self._prepare_targets(validation_targets)

                y_val_pred_mean, y_val_pred_std = self.model.predict(
                    X_val_scaled, return_std=True
                )

                validation_score = r2_score(y_val_true, y_val_pred_mean)
                validation_mae = mean_absolute_error(y_val_true, y_val_pred_mean)
                validation_rmse = np.sqrt(
                    mean_squared_error(y_val_true, y_val_pred_mean)
                )
                avg_validation_std = np.mean(y_val_pred_std)

                # Calibrate uncertainty on validation data
                self._calibrate_uncertainty(y_val_true, y_val_pred_mean, y_val_pred_std)

            # Update model state
            self.is_trained = True
            self.training_date = datetime.now(timezone.utc)
            self.model_version = self._generate_model_version()

            # Calculate training time
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Create training result with GP-specific metrics
            training_metrics = {
                "training_mae": training_mae,
                "training_rmse": training_rmse,
                "training_r2": training_score,
                "log_marginal_likelihood": float(self.log_marginal_likelihood),
                "avg_prediction_std": float(avg_prediction_std),
                "kernel_type": self.model_params["kernel_type"],
                "n_inducing_points": (
                    len(self.inducing_points) if self.use_sparse_gp else len(features)
                ),
                "sparse_gp": self.use_sparse_gp,
                "uncertainty_calibrated": self.uncertainty_calibrated,
            }

            if validation_score is not None:
                training_metrics.update(
                    {
                        "validation_mae": validation_mae,
                        "validation_rmse": validation_rmse,
                        "validation_r2": validation_score,
                        "avg_validation_std": float(avg_validation_std),
                    }
                )

            result = TrainingResult(
                success=True,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=len(features),
                validation_score=validation_score,
                training_score=training_score,
                training_metrics=training_metrics,
            )

            self.training_history.append(result)

            logger.info(f"GP training completed in {training_time:.2f}s")
            logger.info(
                f"Training R²: {training_score:.4f}, "
                f"Validation R²: {validation_score}, "
                f"Log-likelihood: {self.log_marginal_likelihood:.2f}"
            )

            return result

        except Exception as e:
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"GP training failed: {str(e)}"
            logger.error(error_msg)

            result = TrainingResult(
                success=False,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=0,
                error_message=error_msg,
            )

            self.training_history.append(result)
            raise ModelTrainingError(
                model_type="gaussian_process", room_id=self.room_id, cause=e
            )

    async def predict(
        self,
        features: pd.DataFrame,
        prediction_time: datetime,
        current_state: str = "unknown",
    ) -> List[PredictionResult]:
        """
        Generate probabilistic predictions with uncertainty quantification.

        Args:
            features: Feature matrix for prediction
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known

        Returns:
            List of prediction results with confidence intervals
        """
        if not self.is_trained or self.model is None:
            raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")

        if not self.validate_features(features):
            raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")

        try:
            predictions = []
            X_scaled = self.feature_scaler.transform(features)

            # Get predictions with uncertainty
            y_pred_mean, y_pred_std = self.model.predict(X_scaled, return_std=True)

            for idx in range(len(features)):
                mean_time_until = y_pred_mean[idx]
                std_time_until = y_pred_std[idx]

                # Ensure reasonable bounds
                mean_time_until = np.clip(mean_time_until, 60, 86400)
                std_time_until = max(std_time_until, 30)  # Minimum 30 seconds std

                # Calculate predicted transition time
                predicted_time = prediction_time + timedelta(seconds=mean_time_until)

                # Determine transition type
                transition_type = self._determine_transition_type(
                    current_state, prediction_time.hour
                )

                # Calculate confidence intervals
                confidence_intervals = self._calculate_confidence_intervals(
                    mean_time_until, std_time_until
                )

                # Calculate calibrated confidence score
                confidence = self._calculate_confidence_score(
                    mean_time_until, std_time_until
                )

                # Generate alternative scenarios based on uncertainty
                alternatives = self._generate_alternative_scenarios(
                    prediction_time,
                    mean_time_until,
                    std_time_until,
                    transition_type,
                )

                # Create prediction result with GP-specific information
                result = PredictionResult(
                    predicted_time=predicted_time,
                    transition_type=transition_type,
                    confidence_score=confidence,
                    prediction_interval=confidence_intervals.get("95%"),
                    alternatives=alternatives,
                    model_type=self.model_type.value,
                    model_version=self.model_version,
                    features_used=self.feature_names,
                    prediction_metadata={
                        "time_until_transition_seconds": float(mean_time_until),
                        "prediction_std": float(std_time_until),
                        "prediction_method": "gaussian_process",
                        "uncertainty_quantification": {
                            "aleatoric_uncertainty": float(std_time_until),
                            "epistemic_uncertainty": self._estimate_epistemic_uncertainty(
                                X_scaled[idx : idx + 1]
                            ),
                            "confidence_intervals": {
                                level: {
                                    "lower": (
                                        interval[0] - prediction_time
                                    ).total_seconds(),
                                    "upper": (
                                        interval[1] - prediction_time
                                    ).total_seconds(),
                                }
                                for level, interval in confidence_intervals.items()
                            },
                        },
                        "kernel_type": self.model_params["kernel_type"],
                        "sparse_gp": self.use_sparse_gp,
                    },
                )

                predictions.append(result)

                # Record prediction for accuracy tracking
                self._record_prediction(prediction_time, result)

            return predictions

        except Exception as e:
            error_msg = f"GP prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelPredictionError(
                self.model_type.value, self.room_id or "unknown", cause=e
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance approximation based on GP kernel parameters.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            # For GP models, feature importance can be approximated from kernel parameters
            # This is an approximation since GPs don't have explicit feature importance

            importance = {}
            n_features = len(self.feature_names)

            # Get kernel parameters
            kernel_params = self.model.kernel_.get_params()

            if "k1__length_scale" in kernel_params:
                # RBF-like kernel with length scales
                length_scales = kernel_params["k1__length_scale"]

                if np.isscalar(length_scales):
                    # Single length scale for all features
                    for feature_name in self.feature_names:
                        importance[feature_name] = 1.0 / n_features
                else:
                    # Individual length scales (ARD kernel)
                    # Smaller length scale = higher importance
                    inv_length_scales = 1.0 / (length_scales + 1e-10)
                    normalized_importance = inv_length_scales / np.sum(
                        inv_length_scales
                    )

                    for i, feature_name in enumerate(self.feature_names):
                        importance[feature_name] = float(normalized_importance[i])
            else:
                # Fallback: uniform importance
                uniform_importance = 1.0 / n_features
                for feature_name in self.feature_names:
                    importance[feature_name] = uniform_importance

            return importance

        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            # Fallback: uniform importance
            uniform_importance = 1.0 / len(self.feature_names)
            return {name: uniform_importance for name in self.feature_names}

    def _select_inducing_points(self, X: np.ndarray, y: np.ndarray):
        """
        Select inducing points for sparse GP approximation.

        Args:
            X: Scaled feature matrix
            y: Target values
        """
        from sklearn.cluster import KMeans

        n_inducing = min(self.model_params["max_inducing_points"], len(X))

        # Use K-means clustering to select representative points
        kmeans = KMeans(n_clusters=n_inducing, random_state=42, n_init=10)
        kmeans.fit(X)

        # Select points closest to cluster centers
        inducing_indices = []
        for center in kmeans.cluster_centers_:
            distances = np.sum((X - center) ** 2, axis=1)
            closest_idx = np.argmin(distances)
            inducing_indices.append(closest_idx)

        # Remove duplicates and sort
        inducing_indices = sorted(list(set(inducing_indices)))
        self.inducing_points = X[inducing_indices]

        logger.info(f"Selected {len(self.inducing_points)} inducing points")

    def _calibrate_uncertainty(
        self, y_true: np.ndarray, y_pred: np.ndarray, y_std: np.ndarray
    ):
        """
        Calibrate uncertainty estimates using validation data.

        Args:
            y_true: True target values
            y_pred: Predicted mean values
            y_std: Predicted standard deviations
        """
        try:
            from scipy import stats

            # Calculate normalized residuals
            normalized_residuals = np.abs(y_true - y_pred) / (y_std + 1e-10)

            # Calculate empirical quantiles
            quantiles = [0.68, 0.95]  # 1-sigma and 2-sigma
            empirical_quantiles = [
                np.percentile(normalized_residuals, q * 100) for q in quantiles
            ]

            # Theoretical quantiles for normal distribution
            theoretical_quantiles = [stats.norm.ppf(0.5 + q / 2) for q in quantiles]

            # Store calibration curve (ratio of empirical to theoretical)
            self.calibration_curve = {
                quantiles[i]: empirical_quantiles[i] / theoretical_quantiles[i]
                for i in range(len(quantiles))
            }

            self.uncertainty_calibrated = True
            logger.info(f"Uncertainty calibration completed: {self.calibration_curve}")

        except Exception as e:
            logger.warning(f"Uncertainty calibration failed: {e}")
            self.uncertainty_calibrated = False

    def _calculate_confidence_intervals(
        self, mean: float, std: float
    ) -> Dict[str, Tuple[datetime, datetime]]:
        """
        Calculate confidence intervals for predictions.

        Args:
            mean: Predicted mean time until transition
            std: Predicted standard deviation

        Returns:
            Dictionary with confidence intervals
        """
        from datetime import datetime

        base_time = datetime.now(timezone.utc)

        intervals = {}

        # Apply calibration if available
        calibrated_std = std
        if self.uncertainty_calibrated and self.calibration_curve:
            # Use average calibration factor
            avg_calibration = np.mean(list(self.calibration_curve.values()))
            calibrated_std = std * avg_calibration

        # Calculate intervals
        for confidence_level in self.model_params["confidence_intervals"]:
            # Z-score for confidence level
            if confidence_level == 0.68:  # 1-sigma
                z_score = 1.0
            elif confidence_level == 0.95:  # 2-sigma
                z_score = 1.96
            elif confidence_level == 0.99:  # 3-sigma
                z_score = 2.58
            else:
                from scipy import stats

                z_score = stats.norm.ppf(0.5 + confidence_level / 2)

            lower_bound = max(60, mean - z_score * calibrated_std)
            upper_bound = min(86400, mean + z_score * calibrated_std)

            intervals[f"{int(confidence_level*100)}%"] = (
                base_time + timedelta(seconds=lower_bound),
                base_time + timedelta(seconds=upper_bound),
            )

        return intervals

    def _calculate_confidence_score(self, mean: float, std: float) -> float:
        """
        Calculate calibrated confidence score based on prediction uncertainty.

        Args:
            mean: Predicted mean
            std: Predicted standard deviation

        Returns:
            Confidence score between 0 and 1
        """
        # Base confidence from model fit
        if self.training_history:
            last_training = self.training_history[-1]
            if last_training.validation_score is not None:
                base_confidence = max(0.1, min(0.95, last_training.validation_score))
            else:
                base_confidence = max(
                    0.1, min(0.95, last_training.training_score or 0.7)
                )
        else:
            base_confidence = 0.7

        # Adjust based on prediction uncertainty
        # Lower uncertainty = higher confidence
        uncertainty_factor = 1.0 / (1.0 + std / 1800.0)  # Normalize by 30 minutes

        # Combine base confidence with uncertainty
        confidence = base_confidence * uncertainty_factor

        # Apply calibration if available
        if self.uncertainty_calibrated and self.calibration_curve:
            # Better calibrated uncertainty should increase confidence
            avg_calibration = np.mean(list(self.calibration_curve.values()))
            if 0.8 <= avg_calibration <= 1.2:  # Well calibrated
                confidence *= 1.1

        return float(np.clip(confidence, 0.1, 0.95))

    def _generate_alternative_scenarios(
        self,
        base_time: datetime,
        mean: float,
        std: float,
        transition_type: str,
    ) -> List[Tuple[datetime, float]]:
        """
        Generate alternative prediction scenarios based on uncertainty.

        Args:
            base_time: Base prediction time
            mean: Mean prediction
            std: Standard deviation
            transition_type: Type of transition

        Returns:
            List of alternative (time, confidence) tuples
        """
        alternatives = []

        # Generate scenarios at different standard deviations
        scenarios = [
            (mean - std, 0.68),  # Lower 1-sigma
            (mean + std, 0.68),  # Upper 1-sigma
            (mean - 2 * std, 0.95),  # Lower 2-sigma
        ]

        for alt_mean, confidence in scenarios:
            alt_mean = np.clip(alt_mean, 60, 86400)
            alt_time = base_time + timedelta(seconds=alt_mean)
            alternatives.append((alt_time, confidence))

        # Sort by time and return top 3
        alternatives.sort(key=lambda x: x[0])
        return alternatives[:3]

    def _estimate_epistemic_uncertainty(self, X_point: np.ndarray) -> float:
        """
        Estimate epistemic (model) uncertainty for a prediction point.

        Args:
            X_point: Single prediction point

        Returns:
            Estimated epistemic uncertainty
        """
        try:
            # For GP, epistemic uncertainty relates to distance from training data
            # This is a simplified approximation

            if not hasattr(self.model, "X_train_"):
                return 0.5  # Default uncertainty

            # Calculate minimum distance to training points
            training_points = self.model.X_train_
            distances = np.sqrt(np.sum((training_points - X_point) ** 2, axis=1))
            min_distance = np.min(distances)

            # Normalize distance to uncertainty (0-1 scale)
            # Larger distance = higher epistemic uncertainty
            max_reasonable_distance = 5.0  # Heuristic
            epistemic_uncertainty = min(1.0, min_distance / max_reasonable_distance)

            return float(epistemic_uncertainty)

        except Exception:
            return 0.5  # Default uncertainty

    def _determine_transition_type(self, current_state: str, hour: int) -> str:
        """Determine the type of state transition."""
        if current_state == "occupied":
            return "occupied_to_vacant"
        elif current_state == "vacant":
            return "vacant_to_occupied"
        else:
            # Heuristic based on time of day
            if 6 <= hour <= 22:  # Daytime
                return "vacant_to_occupied"
            else:  # Nighttime
                return "occupied_to_vacant"

    def _prepare_targets(self, targets: pd.DataFrame) -> np.ndarray:
        """Prepare target values from DataFrame."""
        if "time_until_transition_seconds" in targets.columns:
            target_values = targets["time_until_transition_seconds"].values
        elif (
            "next_transition_time" in targets.columns
            and "target_time" in targets.columns
        ):
            target_times = pd.to_datetime(targets["target_time"])
            next_times = pd.to_datetime(targets["next_transition_time"])
            target_values = (next_times - target_times).dt.total_seconds().values
        else:
            target_values = targets.iloc[:, 0].values

        return np.clip(target_values, 60, 86400)

    def get_uncertainty_metrics(self) -> Dict[str, Any]:
        """Get uncertainty quantification metrics."""
        if not self.is_trained:
            return {}

        metrics = {
            "kernel_type": self.model_params["kernel_type"],
            "log_marginal_likelihood": self.log_marginal_likelihood,
            "uncertainty_calibrated": self.uncertainty_calibrated,
            "confidence_intervals": self.model_params["confidence_intervals"],
            "sparse_gp": self.use_sparse_gp,
        }

        if self.uncertainty_calibrated and self.calibration_curve:
            metrics["calibration_curve"] = self.calibration_curve

        if self.use_sparse_gp and self.inducing_points is not None:
            metrics["n_inducing_points"] = len(self.inducing_points)

        return metrics

    async def incremental_update(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        learning_rate: float = 0.1,
    ) -> TrainingResult:
        """
        Perform incremental update of the GP model.

        For GP models, this involves adding new data points and re-optimizing
        hyperparameters with a warm start.

        Args:
            features: New training feature matrix
            targets: New training target values
            learning_rate: Learning rate (not directly used for GP but kept for interface)

        Returns:
            TrainingResult with incremental update statistics
        """
        start_time = datetime.now(timezone.utc)

        try:
            logger.info(f"Starting incremental GP update for room {self.room_id}")

            if not self.is_trained:
                logger.warning(
                    "Model not trained yet, performing full training instead"
                )
                return await self.train(features, targets)

            if len(features) < 5:
                raise ModelTrainingError(
                    self.model_type.value, self.room_id or "unknown", cause=None
                )

            # Prepare new data
            X_new = self.feature_scaler.transform(features)
            y_new = self._prepare_targets(targets)

            # For sparse GP, potentially update inducing points
            if self.use_sparse_gp:
                # Add some new points as inducing points if they're diverse enough
                if len(X_new) > 5:
                    # Simple approach: add a few representative points
                    n_add = min(10, len(X_new) // 2)
                    indices = np.random.choice(len(X_new), n_add, replace=False)
                    new_inducing = X_new[indices]

                    # Combine with existing inducing points
                    combined_inducing = np.vstack([self.inducing_points, new_inducing])

                    # If too many inducing points, subsample
                    if (
                        len(combined_inducing)
                        > self.model_params["max_inducing_points"]
                    ):
                        from sklearn.cluster import KMeans

                        n_keep = self.model_params["max_inducing_points"]
                        kmeans = KMeans(n_clusters=n_keep, random_state=42, n_init=10)
                        kmeans.fit(combined_inducing)

                        # Select points closest to cluster centers
                        inducing_indices = []
                        for center in kmeans.cluster_centers_:
                            distances = np.sum(
                                (combined_inducing - center) ** 2, axis=1
                            )
                            closest_idx = np.argmin(distances)
                            inducing_indices.append(closest_idx)

                        self.inducing_points = combined_inducing[inducing_indices]
                    else:
                        self.inducing_points = combined_inducing

            # For full GP, we need to retrain with combined data
            # This is not truly "incremental" but necessary for GP
            if hasattr(self.model, "X_train_"):
                # Combine with existing training data
                X_combined = np.vstack([self.model.X_train_, X_new])
                y_combined = np.hstack([self.model.y_train_, y_new])

                # Limit combined data size to prevent memory issues
                max_combined_size = 1000
                if len(X_combined) > max_combined_size:
                    # Keep most recent data
                    X_combined = X_combined[-max_combined_size:]
                    y_combined = y_combined[-max_combined_size:]

                # Retrain with warm start (use current kernel parameters)
                current_kernel = self.model.kernel_
                self.model = GaussianProcessRegressor(
                    kernel=current_kernel,
                    alpha=self.model_params["alpha"],
                    optimizer="fmin_l_bfgs_b",
                    n_restarts_optimizer=1,  # Fewer restarts for incremental
                    normalize_y=self.model_params["normalize_y"],
                    copy_X_train=self.model_params["copy_X_train"],
                    random_state=self.model_params["random_state"],
                )

                self.model.fit(X_combined, y_combined)
            else:
                # Fallback: train on new data only
                self.model.fit(X_new, y_new)

            # Update model information
            self.log_marginal_likelihood = self.model.log_marginal_likelihood()
            self.kernel_params_history.append(self.model.kernel_.get_params())

            # Calculate performance on new data
            y_pred_mean, y_pred_std = self.model.predict(X_new, return_std=True)

            training_score = r2_score(y_new, y_pred_mean)
            training_mae = mean_absolute_error(y_new, y_pred_mean)
            avg_uncertainty = np.mean(y_pred_std)

            # Update model version
            import time

            self.model_version = f"{self.model_version}_inc_{int(time.time())}"

            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            result = TrainingResult(
                success=True,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=len(features),
                training_score=training_score,
                training_metrics={
                    "update_type": "incremental",
                    "incremental_mae": training_mae,
                    "incremental_r2": training_score,
                    "avg_prediction_uncertainty": float(avg_uncertainty),
                    "log_marginal_likelihood": float(self.log_marginal_likelihood),
                    "sparse_gp": self.use_sparse_gp,
                    "n_inducing_points": (
                        len(self.inducing_points) if self.use_sparse_gp else len(X_new)
                    ),
                },
            )

            self.training_history.append(result)

            logger.info(
                f"Incremental GP update completed in {training_time:.2f}s: "
                f"R²={training_score:.4f}, MAE={training_mae:.2f}min, "
                f"Avg uncertainty={avg_uncertainty:.2f}min"
            )

            return result

        except Exception as e:
            training_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            error_msg = f"Incremental GP update failed: {str(e)}"
            logger.error(error_msg)

            result = TrainingResult(
                success=False,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=0,
                error_message=error_msg,
            )

            self.training_history.append(result)
            raise ModelTrainingError(
                self.model_type.value, self.room_id or "unknown", cause=e
            )

    def save_model(self, file_path: Union[str, Path]) -> bool:
        """
        Save the trained GP model with all components including feature scaler.
        
        Args:
            file_path: Path to save the model
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            
            model_data = {
                "model": self.gp_model,
                "feature_scaler": self.feature_scaler,
                "model_type": self.model_type.value,
                "room_id": self.room_id,
                "model_version": self.model_version,
                "training_date": self.training_date,
                "feature_names": self.feature_names,
                "model_params": self.model_params,
                "is_trained": self.is_trained,
                "training_history": [
                    result.to_dict() for result in self.training_history
                ],
                "kernel_type": self.kernel_type,
                "optimization_restarts": self.optimization_restarts,
                "scaler_fitted": getattr(self, '_scaler_fitted', False),
            }
            
            with open(file_path, "wb") as f:
                pickle.dump(model_data, f)
            
            logger.info(f"GP model saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save GP model: {e}")
            return False
    
    def load_model(self, file_path: Union[str, Path]) -> bool:
        """
        Load a trained GP model with all components including feature scaler.
        
        Args:
            file_path: Path to load the model from
        
        Returns:
            True if successful, False otherwise
        """
        try:
            import pickle
            
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)
            
            self.gp_model = model_data["model"]
            self.feature_scaler = model_data.get("feature_scaler", StandardScaler())
            self.model_type = ModelType(model_data["model_type"])
            self.room_id = model_data.get("room_id")
            self.model_version = model_data.get("model_version", "v1.0")
            self.training_date = model_data.get("training_date")
            self.feature_names = model_data.get("feature_names", [])
            self.model_params = model_data.get("model_params", {})
            self.is_trained = model_data.get("is_trained", False)
            self.kernel_type = model_data.get("kernel_type", "rbf")
            self.optimization_restarts = model_data.get("optimization_restarts", 0)
            self._scaler_fitted = model_data.get("scaler_fitted", False)
            
            # Restore training history
            history_data = model_data.get("training_history", [])
            self.training_history = []
            for result_dict in history_data:
                result = TrainingResult(**result_dict)
                self.training_history.append(result)
            
            logger.info(f"GP model loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load GP model: {e}")
            return False
