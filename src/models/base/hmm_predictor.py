"""
Hidden Markov Model predictor for occupancy state transitions.

This module implements an HMM-based predictor using scikit-learn's GaussianMixture
for modeling occupancy state transitions and duration predictions.
"""

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from ...core.constants import DEFAULT_MODEL_PARAMS, ModelType
from ...core.exceptions import ModelPredictionError, ModelTrainingError
from .predictor import BasePredictor, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)


class HMMPredictor(BasePredictor):
    """
    Hidden Markov Model predictor for occupancy state transitions.

    This predictor models the hidden states of occupancy patterns and
    predicts transition times based on probabilistic state sequences.
    """

    def __init__(self, room_id: Optional[str] = None, **kwargs):
        """
        Initialize the HMM predictor.

        Args:
            room_id: Specific room this model is for
            **kwargs: Additional parameters for model configuration
        """
        super().__init__(ModelType.HMM, room_id)

        # Default HMM parameters
        default_params = DEFAULT_MODEL_PARAMS[ModelType.HMM].copy()
        default_params.update(kwargs)

        self.model_params = {
            "n_components": default_params.get("n_components", 4),
            "covariance_type": default_params.get("covariance_type", "full"),
            "max_iter": default_params.get("n_iter", 100),
            "random_state": default_params.get("random_state", 42),
            "init_params": default_params.get("init_params", "kmeans"),
            "tol": default_params.get("tol", 1e-3),
        }

        # Model components
        self.state_model: Optional[GaussianMixture] = (
            None  # Hidden state identification
        )
        self.transition_models: Dict[int, Any] = {}  # Duration prediction per state
        self.feature_scaler = StandardScaler()

        # State interpretation
        self.state_labels: Dict[int, str] = {}  # State ID to description mapping
        self.state_characteristics: Dict[int, Dict[str, float]] = {}

        # Transition matrix (estimated from data)
        self.transition_matrix: Optional[np.ndarray] = None
        self.state_durations: Dict[int, List[float]] = {}

    async def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame] = None,
        validation_targets: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """
        Train the HMM model on state transition data.

        Args:
            features: Training feature matrix
            targets: Training target values (time until next transition)
            validation_features: Optional validation features
            validation_targets: Optional validation targets

        Returns:
            TrainingResult with training statistics
        """
        start_time = datetime.utcnow()

        try:
            logger.info(f"Starting HMM training for room {self.room_id}")
            logger.info(f"Training data shape: {features.shape}")

            # Prepare data
            y_train = self._prepare_targets(targets)

            if len(y_train) < 20:
                raise ModelTrainingError(
                    self.model_type.value, self.room_id or "unknown", cause=None
                )

            # Scale features
            X_train_scaled = self.feature_scaler.fit_transform(features)

            # Use KMeans for better initial state clustering
            logger.info(
                f"Pre-clustering with KMeans ({self.model_params['n_components']} clusters)"
            )
            kmeans = KMeans(
                n_clusters=self.model_params["n_components"],
                random_state=self.model_params["random_state"],
                n_init=10,
            )
            # Get cluster centers for GMM initialization
            kmeans.fit(X_train_scaled)  # Labels not needed, only centers
            kmeans_centers = kmeans.cluster_centers_

            # Train Gaussian Mixture Model to identify hidden states
            logger.info(
                f"Training GMM with {self.model_params['n_components']} components"
            )

            self.state_model = GaussianMixture(
                n_components=self.model_params["n_components"],
                covariance_type=self.model_params["covariance_type"],
                max_iter=self.model_params["max_iter"],
                random_state=self.model_params["random_state"],
                init_params=self.model_params["init_params"],
                tol=self.model_params["tol"],
                means_init=kmeans_centers,  # Initialize with KMeans centers
            )

            # Fit the state model with KMeans initialization
            self.state_model.fit(X_train_scaled)

            # Identify states for each sample
            state_labels = self.state_model.predict(X_train_scaled)
            state_probabilities = self.state_model.predict_proba(X_train_scaled)

            # Analyze state characteristics using state probabilities
            self._analyze_states(
                X_train_scaled,
                state_labels,
                y_train,
                features.columns,
                state_probabilities,
            )

            # Build transition matrix
            self._build_transition_matrix(state_labels)

            # Train duration prediction models for each state
            self._train_state_duration_models(
                X_train_scaled, state_labels, y_train, features.columns
            )

            # Store model information
            self.feature_names = list(features.columns)
            self.is_trained = True
            self.training_date = datetime.utcnow()
            self.model_version = self._generate_model_version()

            # Calculate training metrics
            y_pred_train = self._predict_durations(X_train_scaled)
            training_score = r2_score(y_train, y_pred_train)
            training_mae = mean_absolute_error(y_train, y_pred_train)
            training_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))

            # Calculate validation metrics if provided
            validation_score = None
            validation_mae = None
            validation_rmse = None

            if validation_features is not None and validation_targets is not None:
                y_val = self._prepare_targets(validation_targets)
                X_val_scaled = self.feature_scaler.transform(validation_features)
                y_pred_val = self._predict_durations(X_val_scaled)

                validation_score = r2_score(y_val, y_pred_val)
                validation_mae = mean_absolute_error(y_val, y_pred_val)
                validation_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))

            # Calculate training time
            training_time = (datetime.utcnow() - start_time).total_seconds()

            # Create training result
            training_metrics = {
                "training_mae": training_mae,
                "training_rmse": training_rmse,
                "training_r2": training_score,
                "n_states": self.model_params["n_components"],
                "convergence_iter": getattr(self.state_model, "n_iter_", None),
                "log_likelihood": getattr(self.state_model, "lower_bound_", None),
                "state_distribution": [
                    int(np.sum(state_labels == i))
                    for i in range(self.model_params["n_components"])
                ],
            }

            if validation_score is not None:
                training_metrics.update(
                    {
                        "validation_mae": validation_mae,
                        "validation_rmse": validation_rmse,
                        "validation_r2": validation_score,
                    }
                )

            result = TrainingResult(
                success=True,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=len(X_train_scaled),
                validation_score=validation_score,
                training_score=training_score,
                training_metrics=training_metrics,
            )

            self.training_history.append(result)

            logger.info(f"HMM training completed in {training_time:.2f}s")
            logger.info(
                f"Training R²: {training_score:.4f}, Validation R²: {validation_score}"
            )
            logger.info(f"Identified {self.model_params['n_components']} hidden states")

            return result

        except Exception as e:
            training_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"HMM training failed: {str(e)}"
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

    async def predict(
        self,
        features: pd.DataFrame,
        prediction_time: datetime,
        current_state: str = "unknown",
    ) -> List[PredictionResult]:
        """
        Generate predictions using the trained HMM model.

        Args:
            features: Feature matrix for prediction
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known

        Returns:
            List of prediction results
        """
        if not self.is_trained or self.state_model is None:
            raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")

        if not self.validate_features(features):
            raise ModelPredictionError(self.model_type.value, self.room_id or "unknown")

        try:
            # Scale features
            X_scaled = self.feature_scaler.transform(features)

            predictions = []

            for idx in range(len(X_scaled)):
                # Identify current hidden state
                current_state_probs = self.state_model.predict_proba(
                    X_scaled[idx : idx + 1]
                )[0]
                most_likely_state = np.argmax(current_state_probs)
                state_confidence = current_state_probs[most_likely_state]

                # Predict duration until next transition
                predicted_duration = self._predict_single_duration(
                    X_scaled[idx : idx + 1], most_likely_state
                )

                # Calculate predicted transition time
                predicted_time = prediction_time + timedelta(seconds=predicted_duration)

                # Predict next state using transition matrix
                next_state_probs = (
                    self.transition_matrix[most_likely_state]
                    if self.transition_matrix is not None
                    else None
                )

                # Determine transition type
                transition_type = self._determine_transition_type_from_states(
                    most_likely_state, next_state_probs, current_state
                )

                # Calculate overall confidence
                confidence = self._calculate_confidence(
                    state_confidence, predicted_duration, current_state_probs
                )

                # Create prediction result
                result = PredictionResult(
                    predicted_time=predicted_time,
                    transition_type=transition_type,
                    confidence_score=confidence,
                    model_type=self.model_type.value,
                    model_version=self.model_version,
                    features_used=self.feature_names,
                    prediction_metadata={
                        "time_until_transition_seconds": float(predicted_duration),
                        "prediction_method": "hidden_markov_model",
                        "current_hidden_state": int(most_likely_state),
                        "state_probability": float(state_confidence),
                        "state_label": self.state_labels.get(
                            most_likely_state, f"State_{most_likely_state}"
                        ),
                        "all_state_probabilities": current_state_probs.tolist(),
                        "next_state_probabilities": (
                            next_state_probs.tolist()
                            if next_state_probs is not None
                            else None
                        ),
                    },
                )

                predictions.append(result)

                # Record prediction for accuracy tracking
                self._record_prediction(prediction_time, result)

            return predictions

        except Exception as e:
            error_msg = f"HMM prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelPredictionError(
                self.model_type.value, self.room_id or "unknown", cause=e
            )

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance based on state discrimination power.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.state_model is None:
            return {}

        try:
            # Calculate feature importance based on how well features separate states
            if not hasattr(self.state_model, "means_") or not hasattr(
                self.state_model, "covariances_"
            ):
                return {}

            means = self.state_model.means_
            covariances = self.state_model.covariances_

            importance_scores = {}

            for i, feature_name in enumerate(self.feature_names):
                # Calculate variance between state means for this feature
                feature_means = means[:, i]
                between_state_variance = np.var(feature_means)

                # Calculate average within-state variance for this feature
                if self.model_params["covariance_type"] == "full":
                    within_state_variances = [cov[i, i] for cov in covariances]
                elif self.model_params["covariance_type"] == "diag":
                    within_state_variances = [cov[i] for cov in covariances]
                else:  # 'tied' or 'spherical'
                    within_state_variances = [
                        np.mean(np.diag(covariances)) for _ in range(len(means))
                    ]

                avg_within_state_variance = np.mean(within_state_variances)

                # Importance is ratio of between-state to within-state variance
                if avg_within_state_variance > 0:
                    importance = between_state_variance / avg_within_state_variance
                else:
                    importance = between_state_variance

                importance_scores[feature_name] = importance

            # Normalize to sum to 1
            total_importance = sum(importance_scores.values())
            if total_importance > 0:
                importance_scores = {
                    k: v / total_importance for k, v in importance_scores.items()
                }

            return importance_scores

        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            return {}

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

        return np.clip(target_values, 60, 86400)  # 1 min to 24 hours

    def _analyze_states(
        self,
        X: np.ndarray,
        state_labels: np.ndarray,
        durations: np.ndarray,
        feature_names: List[str],
        state_probabilities: np.ndarray,
    ):
        """Analyze characteristics of identified hidden states using state probabilities."""
        self.state_characteristics = {}
        self.state_labels = {}
        self.state_durations = {}

        for state_id in range(self.model_params["n_components"]):
            state_mask = state_labels == state_id

            if np.sum(state_mask) == 0:
                continue

            # Analyze durations for this state
            state_duration_list = durations[state_mask].tolist()
            self.state_durations[state_id] = state_duration_list

            avg_duration = np.mean(state_duration_list)
            std_duration = np.std(state_duration_list)

            # Analyze feature characteristics
            state_features = X[state_mask]
            feature_means = np.mean(state_features, axis=0)

            # Use state probabilities for enhanced state analysis
            state_probs = state_probabilities[state_mask, state_id]
            avg_probability = np.mean(state_probs)
            confidence_variance = np.var(state_probs)

            # Calculate certainty metrics using state probabilities
            high_confidence_samples = np.sum(state_probs > 0.8)
            low_confidence_samples = np.sum(state_probs < 0.6)

            # Assign intuitive labels based on characteristics
            label = self._assign_state_label(
                state_id, avg_duration, feature_means, feature_names
            )

            self.state_labels[state_id] = label
            self.state_characteristics[state_id] = {
                "avg_duration": avg_duration,
                "std_duration": std_duration,
                "sample_count": int(np.sum(state_mask)),
                "feature_means": feature_means.tolist(),
                "avg_state_probability": float(avg_probability),
                "confidence_variance": float(confidence_variance),
                "high_confidence_samples": int(high_confidence_samples),
                "low_confidence_samples": int(low_confidence_samples),
                "prediction_reliability": (
                    "high"
                    if avg_probability > 0.75
                    else "medium" if avg_probability > 0.6 else "low"
                ),
            }

        logger.info("State analysis complete:")
        for state_id, label in self.state_labels.items():
            characteristics = self.state_characteristics[state_id]
            logger.info(
                f"  {label}: avg_duration={characteristics['avg_duration']:.0f}s, "
                f"samples={characteristics['sample_count']}, "
                f"reliability={characteristics['prediction_reliability']}, "
                f"avg_probability={characteristics['avg_state_probability']:.3f}"
            )

    def _assign_state_label(
        self,
        state_id: int,
        avg_duration: float,
        feature_means: np.ndarray,
        feature_names: List[str],
    ) -> str:
        """Assign intuitive labels to hidden states."""
        # Simple heuristic-based labeling
        if avg_duration < 600:  # Less than 10 minutes
            return f"Quick_Transition_{state_id}"
        elif avg_duration < 3600:  # Less than 1 hour
            return f"Short_Stay_{state_id}"
        elif avg_duration < 14400:  # Less than 4 hours
            return f"Medium_Stay_{state_id}"
        else:
            return f"Long_Stay_{state_id}"

    def _build_transition_matrix(self, state_labels: np.ndarray):
        """Build state transition matrix from observed sequences."""
        n_states = self.model_params["n_components"]
        transition_counts = np.zeros((n_states, n_states))

        # Count transitions
        for i in range(len(state_labels) - 1):
            current_state = state_labels[i]
            next_state = state_labels[i + 1]
            transition_counts[current_state, next_state] += 1

        # Convert to probabilities
        self.transition_matrix = np.zeros((n_states, n_states))
        for i in range(n_states):
            row_sum = np.sum(transition_counts[i, :])
            if row_sum > 0:
                self.transition_matrix[i, :] = transition_counts[i, :] / row_sum
            else:
                # Uniform distribution if no transitions observed
                self.transition_matrix[i, :] = 1.0 / n_states

        logger.info("Transition matrix built:")
        for i in range(n_states):
            logger.info(
                f"  {self.state_labels.get(i, f'State_{i}')} -> "
                f"{[f'{prob:.2f}' for prob in self.transition_matrix[i, :]]}"
            )

    def _train_state_duration_models(
        self,
        X: np.ndarray,
        state_labels: np.ndarray,
        durations: np.ndarray,
        feature_names: List[str],
    ):
        """Train duration prediction models for each state."""
        from sklearn.linear_model import LinearRegression

        self.transition_models = {}

        for state_id in range(self.model_params["n_components"]):
            state_mask = state_labels == state_id

            if np.sum(state_mask) < 5:  # Need at least 5 samples
                # Use simple average duration
                if state_id in self.state_durations and self.state_durations[state_id]:
                    avg_duration = np.mean(self.state_durations[state_id])
                    self.transition_models[state_id] = {
                        "type": "average",
                        "value": avg_duration,
                    }
                continue

            # Train linear regression for this state
            X_state = X[state_mask]
            y_state = durations[state_mask]

            model = LinearRegression()
            model.fit(X_state, y_state)

            self.transition_models[state_id] = {
                "type": "regression",
                "model": model,
            }

    def _predict_durations(self, X: np.ndarray) -> np.ndarray:
        """Predict durations for training data."""
        predictions = []

        for i in range(len(X)):
            # Identify state
            state_probs = self.state_model.predict_proba(X[i : i + 1])[0]
            most_likely_state = np.argmax(state_probs)

            # Predict duration
            duration = self._predict_single_duration(X[i : i + 1], most_likely_state)
            predictions.append(duration)

        return np.array(predictions)

    def _predict_single_duration(self, X: np.ndarray, state_id: int) -> float:
        """Predict duration for a single sample in a specific state."""
        if state_id not in self.transition_models:
            # Default prediction
            return 1800.0  # 30 minutes

        model_info = self.transition_models[state_id]

        if model_info["type"] == "average":
            return model_info["value"]
        elif model_info["type"] == "regression":
            prediction = model_info["model"].predict(X)[0]
            return np.clip(prediction, 60, 86400)
        else:
            return 1800.0

    def _determine_transition_type_from_states(
        self,
        current_state: int,
        next_state_probs: Optional[np.ndarray],
        current_occupancy: str,
    ) -> str:
        """Determine transition type based on state analysis."""
        if current_occupancy == "occupied":
            return "occupied_to_vacant"
        elif current_occupancy == "vacant":
            return "vacant_to_occupied"
        else:
            # Use state characteristics to infer
            current_characteristics = self.state_characteristics.get(current_state, {})
            avg_duration = current_characteristics.get("avg_duration", 1800)

            # Heuristic: longer stays suggest currently occupied
            if avg_duration > 3600:  # More than 1 hour
                return "occupied_to_vacant"
            else:
                return "vacant_to_occupied"

    def _calculate_confidence(
        self,
        state_confidence: float,
        predicted_duration: float,
        all_state_probs: np.ndarray,
    ) -> float:
        """Calculate prediction confidence."""
        # Base confidence from state identification
        base_confidence = state_confidence

        # Adjust based on state probability distribution entropy
        entropy = -np.sum(all_state_probs * np.log2(all_state_probs + 1e-10))
        max_entropy = np.log2(len(all_state_probs))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

        # Lower confidence for high entropy (uncertain state identification)
        confidence = base_confidence * (1 - 0.3 * normalized_entropy)

        # Adjust for prediction reasonableness
        if predicted_duration < 300 or predicted_duration > 43200:
            confidence *= 0.8

        return float(np.clip(confidence, 0.1, 0.95))

    def get_state_info(self) -> Dict[str, Any]:
        """Get detailed information about identified states."""
        return {
            "n_states": self.model_params["n_components"],
            "state_labels": self.state_labels,
            "state_characteristics": self.state_characteristics,
            "transition_matrix": (
                self.transition_matrix.tolist()
                if self.transition_matrix is not None
                else None
            ),
        }
