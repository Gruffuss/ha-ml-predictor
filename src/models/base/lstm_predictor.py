"""
LSTM-based predictor for occupancy prediction using sequence patterns.

This module implements an LSTM-style neural network predictor using scikit-learn's
MLPRegressor for sequence-based occupancy predictions.
"""

from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from ...core.constants import DEFAULT_MODEL_PARAMS, ModelType
from ...core.exceptions import ModelPredictionError, ModelTrainingError
from .predictor import BasePredictor, PredictionResult, TrainingResult

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


class LSTMPredictor(BasePredictor):
    """
    LSTM-style neural network predictor for sequence-based occupancy patterns.

    This predictor uses scikit-learn's MLPRegressor to learn temporal sequences
    and predict next state transition times based on historical patterns.
    """

    def __init__(self, room_id: Optional[str] = None, **kwargs):
        """
        Initialize the LSTM predictor.

        Args:
            room_id: Specific room this model is for
            **kwargs: Additional parameters for model configuration
        """
        super().__init__(ModelType.LSTM, room_id)

        # Default LSTM parameters
        default_params = DEFAULT_MODEL_PARAMS[ModelType.LSTM].copy()
        default_params.update(kwargs)

        # Handle hidden_units conversion to hidden_layers
        hidden_units = default_params.get("hidden_units", [64, 32])
        if isinstance(hidden_units, int):
            hidden_layers = [hidden_units, hidden_units // 2]
        else:
            hidden_layers = hidden_units

        self.model_params = {
            "sequence_length": default_params.get("sequence_length", 50),
            "hidden_layers": hidden_layers,
            "learning_rate": default_params.get("learning_rate", 0.001),
            "max_iter": default_params.get("max_iter", 1000),
            "early_stopping": default_params.get("early_stopping", True),
            "validation_fraction": default_params.get("validation_fraction", 0.2),
            "alpha": default_params.get("alpha", 0.0001),  # L2 regularization
            "dropout": default_params.get(
                "dropout", 0.2
            ),  # Not directly supported, approximated with alpha
        }

        # Model components
        self.model: Optional[MLPRegressor] = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = MinMaxScaler()

        # Sequence processing parameters
        self.sequence_length = self.model_params["sequence_length"]
        self.sequence_step = 5  # Step size for sequence generation

        # Training statistics
        self.training_loss_history: List[float] = []
        self.validation_loss_history: List[float] = []

    async def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame] = None,
        validation_targets: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """
        Train the LSTM model on sequential data.

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
            logger.info(f"Starting LSTM training for room {self.room_id}")
            logger.info(f"Training data shape: {features.shape}")

            # Prepare sequence data
            X_sequences, y_sequences = self._create_sequences(features, targets)

            if len(X_sequences) < 10:
                raise ModelTrainingError(
                    model_type="lstm",
                    room_id=self.room_id,
                    cause=ValueError(
                        f"Insufficient sequence data: only {len(X_sequences)} sequences available"
                    ),
                )

            logger.info(
                f"Generated {len(X_sequences)} sequences of length {self.sequence_length}"
            )

            # Scale features and targets
            X_scaled = self.feature_scaler.fit_transform(X_sequences)
            y_scaled = self.target_scaler.fit_transform(
                y_sequences.reshape(-1, 1)
            ).ravel()

            # Prepare validation data if provided
            X_val_scaled = None
            y_val_scaled = None
            if validation_features is not None and validation_targets is not None:
                X_val_seq, y_val_seq = self._create_sequences(
                    validation_features, validation_targets
                )
                if len(X_val_seq) > 0:
                    X_val_scaled = self.feature_scaler.transform(X_val_seq)
                    y_val_scaled = self.target_scaler.transform(
                        y_val_seq.reshape(-1, 1)
                    ).ravel()

            # Create and train the neural network
            self.model = MLPRegressor(
                hidden_layer_sizes=tuple(self.model_params["hidden_layers"]),
                learning_rate_init=self.model_params["learning_rate"],
                max_iter=self.model_params["max_iter"],
                early_stopping=self.model_params["early_stopping"],
                validation_fraction=self.model_params["validation_fraction"],
                alpha=self.model_params["alpha"],
                random_state=42,
                warm_start=False,
            )

            # Train the model
            self.model.fit(X_scaled, y_scaled)

            # Store feature names
            self.feature_names = list(features.columns)

            # Calculate training metrics
            y_pred_train = self.model.predict(X_scaled)
            y_pred_train_original = self.target_scaler.inverse_transform(
                y_pred_train.reshape(-1, 1)
            ).ravel()
            y_train_original = self.target_scaler.inverse_transform(
                y_scaled.reshape(-1, 1)
            ).ravel()

            training_score = r2_score(y_train_original, y_pred_train_original)
            training_mae = mean_absolute_error(y_train_original, y_pred_train_original)
            training_rmse = np.sqrt(
                mean_squared_error(y_train_original, y_pred_train_original)
            )

            # Calculate validation metrics
            validation_score = None
            validation_mae = None
            validation_rmse = None

            if X_val_scaled is not None and y_val_scaled is not None:
                y_pred_val = self.model.predict(X_val_scaled)
                y_pred_val_original = self.target_scaler.inverse_transform(
                    y_pred_val.reshape(-1, 1)
                ).ravel()
                y_val_original = self.target_scaler.inverse_transform(
                    y_val_scaled.reshape(-1, 1)
                ).ravel()

                validation_score = r2_score(y_val_original, y_pred_val_original)
                validation_mae = mean_absolute_error(
                    y_val_original, y_pred_val_original
                )
                validation_rmse = np.sqrt(
                    mean_squared_error(y_val_original, y_pred_val_original)
                )

            # Update model state
            self.is_trained = True
            self.training_date = datetime.utcnow()
            self.model_version = self._generate_model_version()

            # Calculate training time
            training_time = (datetime.utcnow() - start_time).total_seconds()

            # Create training result
            training_metrics = {
                "training_mae": training_mae,
                "training_rmse": training_rmse,
                "training_r2": training_score,
                "sequences_generated": len(X_sequences),
                "sequence_length": self.sequence_length,
                "n_iterations": getattr(self.model, "n_iter_", None),
                "loss": getattr(self.model, "loss_", None),
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
                training_samples=len(X_sequences),
                validation_score=validation_score,
                training_score=training_score,
                training_metrics=training_metrics,
            )

            self.training_history.append(result)

            logger.info(f"LSTM training completed in {training_time:.2f}s")
            logger.info(
                f"Training R²: {training_score:.4f}, Validation R²: {validation_score}"
            )

            return result

        except Exception as e:
            training_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"LSTM training failed: {str(e)}"
            logger.error(error_msg)

            result = TrainingResult(
                success=False,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=0,
                error_message=error_msg,
            )

            self.training_history.append(result)
            raise ModelTrainingError(model_type="lstm", room_id=self.room_id, cause=e)

    async def predict(
        self,
        features: pd.DataFrame,
        prediction_time: datetime,
        current_state: str = "unknown",
    ) -> List[PredictionResult]:
        """
        Generate predictions using the trained LSTM model.

        Args:
            features: Feature matrix for prediction
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known

        Returns:
            List of prediction results
        """
        if not self.is_trained or self.model is None:
            raise ModelPredictionError("Model is not trained")

        if not self.validate_features(features):
            raise ModelPredictionError("Feature validation failed")

        try:
            predictions = []

            for idx in range(len(features)):
                # Create sequence from recent features
                if idx >= self.sequence_length - 1:
                    # Use the last sequence_length features
                    feature_sequence = features.iloc[
                        idx - self.sequence_length + 1 : idx + 1
                    ]
                else:
                    # Pad with the first available features if not enough history
                    needed_padding = self.sequence_length - (idx + 1)
                    padding = features.iloc[[0] * needed_padding]
                    feature_sequence = pd.concat(
                        [padding, features.iloc[: idx + 1]], ignore_index=True
                    )

                # Flatten sequence for MLPRegressor
                X_seq = feature_sequence.values.flatten().reshape(1, -1)

                # Scale features
                X_scaled = self.feature_scaler.transform(X_seq)

                # Make prediction
                y_pred_scaled = self.model.predict(X_scaled)

                # Inverse transform to get actual time
                time_until_transition = self.target_scaler.inverse_transform(
                    y_pred_scaled.reshape(-1, 1)
                )[0, 0]

                # Ensure reasonable bounds (between 1 minute and 24 hours)
                time_until_transition = np.clip(
                    time_until_transition, 60, 86400
                )  # 1 min to 24 hours

                # Calculate predicted transition time
                predicted_time = prediction_time + timedelta(
                    seconds=time_until_transition
                )

                # Determine transition type based on current state
                if current_state == "occupied":
                    transition_type = "occupied_to_vacant"
                elif current_state == "vacant":
                    transition_type = "vacant_to_occupied"
                else:
                    # Default assumption based on time patterns
                    hour = prediction_time.hour
                    if 6 <= hour <= 22:  # Daytime - likely to become occupied
                        transition_type = "vacant_to_occupied"
                    else:  # Nighttime - likely to become vacant
                        transition_type = "occupied_to_vacant"

                # Calculate confidence based on prediction consistency
                confidence = self._calculate_confidence(X_scaled, y_pred_scaled)

                # Create prediction result
                result = PredictionResult(
                    predicted_time=predicted_time,
                    transition_type=transition_type,
                    confidence_score=confidence,
                    model_type=self.model_type.value,
                    model_version=self.model_version,
                    features_used=self.feature_names,
                    prediction_metadata={
                        "time_until_transition_seconds": float(time_until_transition),
                        "sequence_length_used": self.sequence_length,
                        "prediction_method": "lstm_neural_network",
                    },
                )

                predictions.append(result)

                # Record prediction for accuracy tracking
                self._record_prediction(prediction_time, result)

            return predictions

        except Exception as e:
            error_msg = f"LSTM prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)

    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance approximation for neural networks.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or self.model is None:
            return {}

        try:
            # For MLPRegressor, we can approximate importance using connection weights
            # This is a simplified approach - actual feature importance for NNs is complex

            if not hasattr(self.model, "coefs_") or not self.model.coefs_:
                return {}

            # Get the input layer weights (first layer)
            input_weights = self.model.coefs_[
                0
            ]  # Shape: (n_features * sequence_length, n_hidden)

            # Calculate average absolute weight for each input feature
            n_features_per_timestep = len(self.feature_names)
            feature_importance = {}

            for i, feature_name in enumerate(self.feature_names):
                # Sum importance across all time steps in the sequence
                total_importance = 0.0
                for t in range(self.sequence_length):
                    feature_idx = t * n_features_per_timestep + i
                    if feature_idx < input_weights.shape[0]:
                        # Average absolute weight to all hidden units
                        importance = np.mean(np.abs(input_weights[feature_idx, :]))
                        total_importance += importance

                feature_importance[feature_name] = (
                    total_importance / self.sequence_length
                )

            # Normalize to sum to 1
            total_importance = sum(feature_importance.values())
            if total_importance > 0:
                feature_importance = {
                    k: v / total_importance for k, v in feature_importance.items()
                }

            return feature_importance

        except Exception as e:
            logger.warning(f"Could not calculate feature importance: {e}")
            return {}

    def _create_sequences(
        self, features: pd.DataFrame, targets: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequential data for LSTM training.

        Args:
            features: Feature DataFrame
            targets: Target DataFrame (time until next transition)

        Returns:
            Tuple of (X_sequences, y_sequences)
        """
        X_sequences = []
        y_sequences = []

        # Convert targets to time differences in seconds
        if "time_until_transition_seconds" in targets.columns:
            target_values = targets["time_until_transition_seconds"].values
        elif (
            "next_transition_time" in targets.columns
            and "target_time" in targets.columns
        ):
            # Calculate time differences
            target_times = pd.to_datetime(targets["target_time"])
            next_times = pd.to_datetime(targets["next_transition_time"])
            target_values = (next_times - target_times).dt.total_seconds().values
        else:
            # Default: assume targets are already time differences
            target_values = targets.iloc[:, 0].values

        # Generate sequences
        for i in range(self.sequence_length, len(features), self.sequence_step):
            # Input sequence: last sequence_length time steps
            X_seq = features.iloc[i - self.sequence_length : i].values

            # Flatten the sequence for MLPRegressor (which expects 1D input per sample)
            X_seq_flat = X_seq.flatten()

            # Target: time until next transition at time step i
            y_seq = target_values[i]

            # Only include sequences with reasonable target values
            if 60 <= y_seq <= 86400:  # Between 1 minute and 24 hours
                X_sequences.append(X_seq_flat)
                y_sequences.append(y_seq)

        return np.array(X_sequences), np.array(y_sequences)

    def _calculate_confidence(
        self, X_scaled: np.ndarray, y_pred_scaled: np.ndarray
    ) -> float:
        """
        Calculate prediction confidence based on model uncertainty.

        Args:
            X_scaled: Scaled input features
            y_pred_scaled: Scaled prediction

        Returns:
            Confidence score between 0 and 1
        """
        try:
            # For neural networks, confidence can be approximated by:
            # 1. Distance from training data (if we stored it)
            # 2. Prediction consistency (if we had ensemble)
            # 3. Fixed confidence based on validation performance

            # Use validation score as base confidence if available
            if self.training_history:
                last_training = self.training_history[-1]
                if last_training.validation_score is not None:
                    base_confidence = max(
                        0.1, min(0.95, last_training.validation_score)
                    )
                else:
                    base_confidence = max(
                        0.1, min(0.95, last_training.training_score or 0.7)
                    )
            else:
                base_confidence = 0.7

            # Adjust confidence based on prediction reasonableness
            pred_value = self.target_scaler.inverse_transform(
                y_pred_scaled.reshape(-1, 1)
            )[0, 0]

            # Lower confidence for extreme predictions
            if (
                pred_value < 300 or pred_value > 43200
            ):  # Less than 5 min or more than 12 hours
                base_confidence *= 0.8

            return float(np.clip(base_confidence, 0.1, 0.95))

        except Exception:
            return 0.7  # Default confidence

    def get_model_complexity(self) -> Dict[str, Any]:
        """Get information about model complexity."""
        if not self.is_trained or self.model is None:
            return {}

        complexity_info = {
            "total_parameters": 0,
            "hidden_layers": self.model_params["hidden_layers"],
            "sequence_length": self.sequence_length,
            "input_features": len(self.feature_names),
            "flattened_input_size": len(self.feature_names) * self.sequence_length,
        }

        # Calculate total parameters
        if hasattr(self.model, "coefs_"):
            total_params = 0
            for coef_matrix in self.model.coefs_:
                total_params += coef_matrix.size
            for bias_vector in self.model.intercepts_:
                total_params += bias_vector.size
            complexity_info["total_parameters"] = total_params

        return complexity_info
