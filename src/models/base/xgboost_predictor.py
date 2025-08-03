"""
XGBoost-based predictor for occupancy prediction using tabular features.

This module implements a gradient boosting predictor using XGBoost for
tabular feature-based occupancy predictions with excellent interpretability.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from ...core.constants import ModelType, DEFAULT_MODEL_PARAMS
from ...core.exceptions import ModelTrainingError, ModelPredictionError
from .predictor import BasePredictor, PredictionResult, TrainingResult


logger = logging.getLogger(__name__)


class XGBoostPredictor(BasePredictor):
    """
    XGBoost gradient boosting predictor for tabular occupancy features.
    
    This predictor excels at learning complex non-linear relationships in
    tabular data and provides excellent feature importance interpretability.
    """
    
    def __init__(self, room_id: Optional[str] = None, **kwargs):
        """
        Initialize the XGBoost predictor.
        
        Args:
            room_id: Specific room this model is for
            **kwargs: Additional parameters for model configuration
        """
        super().__init__(ModelType.XGBOOST, room_id)
        
        # Default XGBoost parameters
        default_params = DEFAULT_MODEL_PARAMS[ModelType.XGBOOST].copy()
        default_params.update(kwargs)
        
        self.model_params = {
            'n_estimators': default_params.get('n_estimators', 100),
            'max_depth': default_params.get('max_depth', 6),
            'learning_rate': default_params.get('learning_rate', 0.1),
            'subsample': default_params.get('subsample', 0.8),
            'colsample_bytree': default_params.get('colsample_bytree', 0.8),
            'reg_alpha': default_params.get('reg_alpha', 0.1),  # L1 regularization
            'reg_lambda': default_params.get('reg_lambda', 1.0),  # L2 regularization
            'random_state': default_params.get('random_state', 42),
            'early_stopping_rounds': default_params.get('early_stopping_rounds', 20),
            'eval_metric': default_params.get('eval_metric', 'rmse')
        }
        
        # Model components
        self.model: Optional[xgb.XGBRegressor] = None
        self.feature_scaler = StandardScaler()
        
        # Feature importance and model interpretation
        self.feature_importance_: Dict[str, float] = {}
        self.best_iteration_: Optional[int] = None
        
        # Training monitoring
        self.eval_results_: Dict[str, List[float]] = {}
    
    async def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame] = None,
        validation_targets: Optional[pd.DataFrame] = None
    ) -> TrainingResult:
        """
        Train the XGBoost model on tabular data.
        
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
            logger.info(f"Starting XGBoost training for room {self.room_id}")
            logger.info(f"Training data shape: {features.shape}")
            
            # Prepare target values
            y_train = self._prepare_targets(targets)
            
            if len(y_train) < 10:
                raise ModelTrainingError(
                    f"Insufficient training data: only {len(y_train)} samples available"
                )
            
            # Scale features (XGBoost can handle unscaled features, but scaling helps with regularization)
            X_train_scaled = self.feature_scaler.fit_transform(features)
            X_train_df = pd.DataFrame(X_train_scaled, columns=features.columns, index=features.index)
            
            # Prepare validation data if provided
            eval_set = None
            if validation_features is not None and validation_targets is not None:
                y_val = self._prepare_targets(validation_targets)
                X_val_scaled = self.feature_scaler.transform(validation_features)
                X_val_df = pd.DataFrame(X_val_scaled, columns=validation_features.columns)
                eval_set = [(X_train_df, y_train), (X_val_df, y_val)]
            else:
                eval_set = [(X_train_df, y_train)]
            
            # Create and configure XGBoost model
            self.model = xgb.XGBRegressor(
                n_estimators=self.model_params['n_estimators'],
                max_depth=self.model_params['max_depth'],
                learning_rate=self.model_params['learning_rate'],
                subsample=self.model_params['subsample'],
                colsample_bytree=self.model_params['colsample_bytree'],
                reg_alpha=self.model_params['reg_alpha'],
                reg_lambda=self.model_params['reg_lambda'],
                random_state=self.model_params['random_state'],
                early_stopping_rounds=self.model_params['early_stopping_rounds'],
                eval_metric=self.model_params['eval_metric'],
                verbosity=0  # Reduce XGBoost output
            )
            
            # Train the model
            self.model.fit(
                X_train_df, 
                y_train,
                eval_set=eval_set,
                verbose=False
            )
            
            # Store training results
            self.best_iteration_ = getattr(self.model, 'best_iteration', None)
            self.eval_results_ = getattr(self.model, 'evals_result_', {})
            
            # Store feature names and importance
            self.feature_names = list(features.columns)
            self.feature_importance_ = dict(zip(
                self.feature_names,
                [float(importance) for importance in self.model.feature_importances_]
            ))
            
            # Calculate training metrics
            y_pred_train = self.model.predict(X_train_df)
            training_score = r2_score(y_train, y_pred_train)
            training_mae = mean_absolute_error(y_train, y_pred_train)
            training_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            
            # Calculate validation metrics
            validation_score = None
            validation_mae = None
            validation_rmse = None
            
            if eval_set and len(eval_set) > 1:
                X_val_df = eval_set[1][0]
                y_val = eval_set[1][1]
                y_pred_val = self.model.predict(X_val_df)
                
                validation_score = r2_score(y_val, y_pred_val)
                validation_mae = mean_absolute_error(y_val, y_pred_val)
                validation_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
            
            # Update model state
            self.is_trained = True
            self.training_date = datetime.utcnow()
            self.model_version = self._generate_model_version()
            
            # Calculate training time
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create training result
            training_metrics = {
                'training_mae': training_mae,
                'training_rmse': training_rmse,
                'training_r2': training_score,
                'n_estimators_used': getattr(self.model, 'n_estimators', None),
                'best_iteration': self.best_iteration_,
                'feature_count': len(self.feature_names)
            }
            
            if validation_score is not None:
                training_metrics.update({
                    'validation_mae': validation_mae,
                    'validation_rmse': validation_rmse,
                    'validation_r2': validation_score
                })
            
            # Add top feature importances to metrics
            sorted_importance = sorted(
                self.feature_importance_.items(), 
                key=lambda x: x[1], 
                reverse=True
            )
            training_metrics['top_features'] = dict(sorted_importance[:10])
            
            result = TrainingResult(
                success=True,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=len(X_train_df),
                validation_score=validation_score,
                training_score=training_score,
                feature_importance=self.feature_importance_,
                training_metrics=training_metrics
            )
            
            self.training_history.append(result)
            
            logger.info(f"XGBoost training completed in {training_time:.2f}s")
            logger.info(f"Training R²: {training_score:.4f}, Validation R²: {validation_score}")
            logger.info(f"Best iteration: {self.best_iteration_}")
            
            return result
            
        except Exception as e:
            training_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"XGBoost training failed: {str(e)}"
            logger.error(error_msg)
            
            result = TrainingResult(
                success=False,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=0,
                error_message=error_msg
            )
            
            self.training_history.append(result)
            raise ModelTrainingError(error_msg)
    
    async def predict(
        self,
        features: pd.DataFrame,
        prediction_time: datetime,
        current_state: str = 'unknown'
    ) -> List[PredictionResult]:
        """
        Generate predictions using the trained XGBoost model.
        
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
            # Scale features
            X_scaled = self.feature_scaler.transform(features)
            X_df = pd.DataFrame(X_scaled, columns=features.columns, index=features.index)
            
            # Make predictions
            y_pred = self.model.predict(X_df)
            
            predictions = []
            
            for idx, time_until_transition in enumerate(y_pred):
                # Ensure reasonable bounds (between 1 minute and 24 hours)
                time_until_transition = np.clip(time_until_transition, 60, 86400)
                
                # Calculate predicted transition time
                predicted_time = prediction_time + timedelta(seconds=time_until_transition)
                
                # Determine transition type based on current state and features
                transition_type = self._determine_transition_type(
                    current_state, features.iloc[idx], prediction_time
                )
                
                # Calculate confidence based on prediction and feature values
                confidence = self._calculate_confidence(
                    X_df.iloc[idx:idx+1], np.array([time_until_transition])
                )
                
                # Get feature contributions for this prediction (SHAP-like)
                feature_contributions = self._get_feature_contributions(X_df.iloc[idx:idx+1])
                
                # Create prediction result
                result = PredictionResult(
                    predicted_time=predicted_time,
                    transition_type=transition_type,
                    confidence_score=confidence,
                    model_type=self.model_type.value,
                    model_version=self.model_version,
                    features_used=self.feature_names,
                    prediction_metadata={
                        'time_until_transition_seconds': float(time_until_transition),
                        'prediction_method': 'xgboost_gradient_boosting',
                        'n_estimators_used': getattr(self.model, 'n_estimators', None),
                        'feature_contributions': feature_contributions
                    }
                )
                
                predictions.append(result)
                
                # Record prediction for accuracy tracking
                self._record_prediction(prediction_time, result)
            
            return predictions
            
        except Exception as e:
            error_msg = f"XGBoost prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get feature importance scores from the trained XGBoost model.
        
        Returns:
            Dictionary mapping feature names to importance scores
        """
        if not self.is_trained or not self.feature_importance_:
            return {}
        
        return self.feature_importance_.copy()
    
    def get_feature_importance_plot_data(self) -> List[Tuple[str, float]]:
        """
        Get feature importance data suitable for plotting.
        
        Returns:
            List of (feature_name, importance) tuples sorted by importance
        """
        importance_dict = self.get_feature_importance()
        return sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    
    def _prepare_targets(self, targets: pd.DataFrame) -> np.ndarray:
        """
        Prepare target values from DataFrame.
        
        Args:
            targets: Target DataFrame
            
        Returns:
            Array of target values (time until transition in seconds)
        """
        if 'time_until_transition_seconds' in targets.columns:
            target_values = targets['time_until_transition_seconds'].values
        elif 'next_transition_time' in targets.columns and 'target_time' in targets.columns:
            # Calculate time differences
            target_times = pd.to_datetime(targets['target_time'])
            next_times = pd.to_datetime(targets['next_transition_time'])
            target_values = (next_times - target_times).dt.total_seconds().values
        else:
            # Default: assume targets are already time differences
            target_values = targets.iloc[:, 0].values
        
        # Clip to reasonable bounds
        target_values = np.clip(target_values, 60, 86400)  # 1 min to 24 hours
        
        return target_values
    
    def _determine_transition_type(
        self, 
        current_state: str, 
        features: pd.Series, 
        prediction_time: datetime
    ) -> str:
        """
        Determine transition type based on current state and features.
        
        Args:
            current_state: Current occupancy state
            features: Feature values for this prediction
            prediction_time: Time of prediction
            
        Returns:
            Transition type string
        """
        if current_state == 'occupied':
            return 'occupied_to_vacant'
        elif current_state == 'vacant':
            return 'vacant_to_occupied'
        else:
            # Use features and time to infer likely transition
            hour = prediction_time.hour
            
            # Check for time-based features
            is_work_hours = False
            is_sleep_hours = False
            
            for feature_name in features.index:
                if 'work_hours' in feature_name.lower() and features[feature_name] > 0.5:
                    is_work_hours = True
                elif 'sleep_hours' in feature_name.lower() and features[feature_name] > 0.5:
                    is_sleep_hours = True
            
            # Default logic based on time and patterns
            if is_sleep_hours or (22 <= hour or hour <= 6):
                return 'occupied_to_vacant'
            elif is_work_hours or (6 <= hour <= 22):
                return 'vacant_to_occupied'
            else:
                # Fallback to hour-based logic
                if 6 <= hour <= 22:
                    return 'vacant_to_occupied'
                else:
                    return 'occupied_to_vacant'
    
    def _calculate_confidence(self, X: pd.DataFrame, y_pred: np.ndarray) -> float:
        """
        Calculate prediction confidence based on model uncertainty.
        
        Args:
            X: Scaled input features for this prediction
            y_pred: Model prediction
            
        Returns:
            Confidence score between 0 and 1
        """
        try:
            # Base confidence from validation performance
            if self.training_history:
                last_training = self.training_history[-1]
                if last_training.validation_score is not None:
                    base_confidence = max(0.1, min(0.95, last_training.validation_score))
                else:
                    base_confidence = max(0.1, min(0.95, last_training.training_score or 0.7))
            else:
                base_confidence = 0.7
            
            # Adjust confidence based on prediction reasonableness
            pred_value = y_pred[0]
            
            # Lower confidence for extreme predictions
            if pred_value < 300 or pred_value > 43200:  # Less than 5 min or more than 12 hours
                base_confidence *= 0.8
            
            # Use feature values to adjust confidence
            # Features with extreme values might indicate uncertainty
            feature_values = X.iloc[0].values
            extreme_feature_ratio = np.mean(np.abs(feature_values) > 2.0)  # Beyond 2 std devs
            
            if extreme_feature_ratio > 0.3:  # More than 30% of features are extreme
                base_confidence *= 0.9
            
            return float(np.clip(base_confidence, 0.1, 0.95))
            
        except Exception:
            return 0.7  # Default confidence
    
    def _get_feature_contributions(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Get feature contributions for interpretability (simplified SHAP-like).
        
        Args:
            X: Single row of scaled features
            
        Returns:
            Dictionary of feature contributions
        """
        try:
            if not self.feature_importance_:
                return {}
            
            # Simplified contribution: feature_value * feature_importance
            feature_values = X.iloc[0]
            contributions = {}
            
            for feature_name in feature_values.index:
                if feature_name in self.feature_importance_:
                    contribution = (
                        float(feature_values[feature_name]) * 
                        self.feature_importance_[feature_name]
                    )
                    contributions[feature_name] = contribution
            
            # Return top 10 contributions by absolute value
            sorted_contributions = sorted(
                contributions.items(), 
                key=lambda x: abs(x[1]), 
                reverse=True
            )
            
            return dict(sorted_contributions[:10])
            
        except Exception:
            return {}
    
    def get_learning_curve_data(self) -> Dict[str, List[float]]:
        """
        Get learning curve data for analysis.
        
        Returns:
            Dictionary with training and validation curves
        """
        if not self.eval_results_:
            return {}
        
        return self.eval_results_
    
    def get_model_complexity(self) -> Dict[str, Any]:
        """Get information about model complexity."""
        if not self.is_trained or self.model is None:
            return {}
        
        return {
            'n_estimators': getattr(self.model, 'n_estimators', None),
            'max_depth': self.model_params['max_depth'],
            'best_iteration': self.best_iteration_,
            'total_features': len(self.feature_names),
            'important_features': len([f for f in self.feature_importance_.values() if f > 0.01]),
            'regularization': {
                'reg_alpha': self.model_params['reg_alpha'],
                'reg_lambda': self.model_params['reg_lambda']
            }
        }