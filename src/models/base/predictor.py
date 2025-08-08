"""
Base predictor interface for occupancy prediction models.

This module defines the abstract base class and common interfaces for all
predictive models in the occupancy prediction system.
"""

import logging
import pickle
from abc import ABC
from abc import abstractmethod
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from ...core.config import SystemConfig
from ...core.constants import ModelType
from ...core.constants import PredictionType
from ...core.exceptions import ModelPredictionError
from ...core.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Represents a prediction result with confidence and metadata."""

    predicted_time: datetime
    transition_type: str  # 'occupied_to_vacant' or 'vacant_to_occupied'
    confidence_score: float
    prediction_interval: Optional[Tuple[datetime, datetime]] = None
    alternatives: Optional[List[Tuple[datetime, float]]] = None
    model_type: Optional[str] = None
    model_version: Optional[str] = None
    features_used: Optional[List[str]] = None
    prediction_metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "predicted_time": self.predicted_time.isoformat(),
            "transition_type": self.transition_type,
            "confidence_score": self.confidence_score,
        }

        if self.prediction_interval:
            result["prediction_interval"] = [
                self.prediction_interval[0].isoformat(),
                self.prediction_interval[1].isoformat(),
            ]

        if self.alternatives:
            result["alternatives"] = [
                {"time": alt[0].isoformat(), "confidence": alt[1]}
                for alt in self.alternatives
            ]

        if self.model_type:
            result["model_type"] = self.model_type
        if self.model_version:
            result["model_version"] = self.model_version
        if self.features_used:
            result["features_used"] = self.features_used
        if self.prediction_metadata:
            result["prediction_metadata"] = self.prediction_metadata

        return result


@dataclass
class TrainingResult:
    """Represents the result of model training."""

    success: bool
    training_time_seconds: float
    model_version: str
    training_samples: int
    validation_score: Optional[float] = None
    training_score: Optional[float] = None
    feature_importance: Optional[Dict[str, float]] = None
    training_metrics: Optional[Dict[str, float]] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "success": self.success,
            "training_time_seconds": self.training_time_seconds,
            "model_version": self.model_version,
            "training_samples": self.training_samples,
            "validation_score": self.validation_score,
            "training_score": self.training_score,
            "feature_importance": self.feature_importance,
            "training_metrics": self.training_metrics,
            "error_message": self.error_message,
        }


class BasePredictor(ABC):
    """
    Abstract base class for all occupancy prediction models.

    This class defines the interface that all predictive models must implement,
    including training, prediction, and model management functionality.
    """

    def __init__(
        self,
        model_type: ModelType,
        room_id: Optional[str] = None,
        config: Optional[SystemConfig] = None,
    ):
        """
        Initialize the base predictor.

        Args:
            model_type: Type of the model
            room_id: Specific room this model is for (if room-specific)
            config: System configuration
        """
        self.model_type = model_type
        self.room_id = room_id
        self.config = config

        # Model state
        self.is_trained = False
        self.model_version = "v1.0"
        self.training_date: Optional[datetime] = None
        self.feature_names: List[str] = []

        # Model performance tracking
        self.training_history: List[TrainingResult] = []
        self.prediction_history: List[Tuple[datetime, PredictionResult]] = []

        # Model-specific parameters (to be set by subclasses)
        self.model_params: Dict[str, Any] = {}
        self.model: Any = None

    @abstractmethod
    async def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame] = None,
        validation_targets: Optional[pd.DataFrame] = None,
    ) -> TrainingResult:
        """
        Train the model on the provided data.

        Args:
            features: Training feature matrix
            targets: Training target values
            validation_features: Optional validation features
            validation_targets: Optional validation targets

        Returns:
            TrainingResult with training statistics

        Raises:
            ModelTrainingError: If training fails
        """
        pass

    @abstractmethod
    async def predict(
        self,
        features: pd.DataFrame,
        prediction_time: datetime,
        current_state: str = "unknown",
    ) -> List[PredictionResult]:
        """
        Generate predictions for the given features.

        Args:
            features: Feature matrix for prediction
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known

        Returns:
            List of prediction results

        Raises:
            ModelPredictionError: If prediction fails
        """
        pass

    @abstractmethod
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained model.

        Returns:
            Dictionary mapping feature names to importance scores
        """
        pass

    async def predict_single(
        self,
        features: Dict[str, float],
        prediction_time: datetime,
        current_state: str = "unknown",
    ) -> PredictionResult:
        """
        Generate a single prediction from feature dictionary.

        Args:
            features: Feature dictionary
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known

        Returns:
            Single prediction result
        """
        # Convert dict to DataFrame
        feature_df = pd.DataFrame([features])
        predictions = await self.predict(feature_df, prediction_time, current_state)

        if not predictions:
            raise ModelPredictionError("No predictions generated")

        return predictions[0]

    def save_model(self, file_path: Union[str, Path]) -> bool:
        """
        Save the trained model to disk.

        Args:
            file_path: Path to save the model

        Returns:
            True if successful, False otherwise
        """
        try:
            model_data = {
                "model": self.model,
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
            }

            with open(file_path, "wb") as f:
                pickle.dump(model_data, f)

            logger.info(f"Model saved to {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save model: {e}")
            return False

    def load_model(self, file_path: Union[str, Path]) -> bool:
        """
        Load a trained model from disk.

        Args:
            file_path: Path to load the model from

        Returns:
            True if successful, False otherwise
        """
        try:
            with open(file_path, "rb") as f:
                model_data = pickle.load(f)

            self.model = model_data["model"]
            self.model_type = ModelType(model_data["model_type"])
            self.room_id = model_data.get("room_id")
            self.model_version = model_data.get("model_version", "v1.0")
            self.training_date = model_data.get("training_date")
            self.feature_names = model_data.get("feature_names", [])
            self.model_params = model_data.get("model_params", {})
            self.is_trained = model_data.get("is_trained", False)

            # Restore training history
            history_data = model_data.get("training_history", [])
            self.training_history = []
            for result_dict in history_data:
                result = TrainingResult(**result_dict)
                self.training_history.append(result)

            logger.info(f"Model loaded from {file_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        return {
            "model_type": self.model_type.value,
            "room_id": self.room_id,
            "model_version": self.model_version,
            "is_trained": self.is_trained,
            "training_date": (
                self.training_date.isoformat() if self.training_date else None
            ),
            "feature_count": len(self.feature_names),
            "feature_names": (
                self.feature_names[:10] if self.feature_names else []
            ),  # First 10 features
            "model_params": self.model_params,
            "training_sessions": len(self.training_history),
            "predictions_made": len(self.prediction_history),
            "last_training_score": (
                self.training_history[-1].validation_score
                if self.training_history
                else None
            ),
        }

    def get_training_history(self) -> List[Dict[str, Any]]:
        """Get training history as list of dictionaries."""
        return [result.to_dict() for result in self.training_history]

    def get_prediction_accuracy(self, hours_back: int = 24) -> Optional[float]:
        """
        Calculate recent prediction accuracy.

        Args:
            hours_back: How many hours back to look for predictions

        Returns:
            Accuracy score or None if insufficient data
        """
        cutoff_time = datetime.utcnow() - timedelta(hours=hours_back)
        recent_predictions = [
            (pred_time, result)
            for pred_time, result in self.prediction_history
            if pred_time >= cutoff_time
        ]

        if len(recent_predictions) < 5:  # Need at least 5 predictions
            return None

        # This would need actual outcome data to compute accuracy
        # For now, return a placeholder
        return 0.85  # 85% accuracy placeholder

    def clear_prediction_history(self):
        """Clear the prediction history to free memory."""
        self.prediction_history.clear()

    def validate_features(self, features: pd.DataFrame) -> bool:
        """
        Validate that the provided features match the model's expectations.

        Args:
            features: Feature DataFrame to validate

        Returns:
            True if valid, False otherwise
        """
        if not self.is_trained:
            logger.warning("Cannot validate features on untrained model")
            return False

        if not self.feature_names:
            logger.warning("No feature names stored in model")
            return True  # Allow if no feature names stored

        # Check that all required features are present
        missing_features = set(self.feature_names) - set(features.columns)
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            return False

        # Check for unexpected features (warn but don't fail)
        extra_features = set(features.columns) - set(self.feature_names)
        if extra_features:
            logger.warning(f"Extra features provided: {extra_features}")

        return True

    def _record_prediction(self, prediction_time: datetime, result: PredictionResult):
        """Record a prediction for accuracy tracking."""
        self.prediction_history.append((prediction_time, result))

        # Keep only recent predictions to limit memory usage
        if len(self.prediction_history) > 1000:
            self.prediction_history = self.prediction_history[-500:]

    def _generate_model_version(self) -> str:
        """Generate a new model version string."""
        base_version = "v1.0"
        if self.training_history:
            last_version = self.training_history[-1].model_version
            if last_version.startswith("v"):
                try:
                    version_num = float(last_version[1:])
                    new_version = f"v{version_num + 0.1:.1f}"
                    return new_version
                except ValueError:
                    pass

        return base_version

    def __str__(self) -> str:
        """String representation of the model."""
        room_str = f" (room: {self.room_id})" if self.room_id else ""
        status = "trained" if self.is_trained else "untrained"
        return f"{self.model_type.value} Predictor{room_str} - {status}"

    def __repr__(self) -> str:
        """Detailed string representation."""
        return (
            f"{self.__class__.__name__}("
            f"model_type={self.model_type.value}, "
            f"room_id={self.room_id}, "
            f"is_trained={self.is_trained}, "
            f"version={self.model_version})"
        )
