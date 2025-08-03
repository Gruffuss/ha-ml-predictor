"""
Ensemble architecture for occupancy prediction.

This module implements a meta-learning ensemble that combines multiple base
predictors (LSTM, XGBoost, HMM) using stacking with a meta-learner.
"""

import logging
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings

from ..core.constants import ModelType, DEFAULT_MODEL_PARAMS
from ..core.exceptions import ModelTrainingError, ModelPredictionError
from .base.predictor import BasePredictor, PredictionResult, TrainingResult
from .base.lstm_predictor import LSTMPredictor
from .base.xgboost_predictor import XGBoostPredictor
from .base.hmm_predictor import HMMPredictor


logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')


class OccupancyEnsemble(BasePredictor):
    """
    Meta-learning ensemble combining multiple base predictors.
    
    This ensemble uses stacking with cross-validation to train a meta-learner
    that optimally combines predictions from LSTM, XGBoost, and HMM models.
    """
    
    def __init__(self, room_id: Optional[str] = None, tracking_manager=None, **kwargs):
        """
        Initialize the ensemble predictor.
        
        Args:
            room_id: Specific room this model is for
            tracking_manager: Optional tracking manager for automatic accuracy tracking
            **kwargs: Additional parameters for ensemble configuration
        """
        super().__init__(ModelType.ENSEMBLE, room_id)
        
        # Default ensemble parameters
        default_params = DEFAULT_MODEL_PARAMS[ModelType.ENSEMBLE].copy()
        default_params.update(kwargs)
        
        self.model_params = {
            'meta_learner': default_params.get('meta_learner', 'random_forest'),
            'cv_folds': default_params.get('cv_folds', 5),
            'stacking_method': default_params.get('stacking_method', 'linear'),
            'blend_weights': default_params.get('blend_weights', 'auto'),
            'use_base_features': default_params.get('use_base_features', True),
            'meta_features_only': default_params.get('meta_features_only', False)
        }
        
        # Base models
        self.base_models: Dict[str, BasePredictor] = {
            'lstm': LSTMPredictor(room_id),
            'xgboost': XGBoostPredictor(room_id),
            'hmm': HMMPredictor(room_id)
        }
        
        # Meta-learner
        self.meta_learner: Optional[Any] = None
        self.meta_scaler = StandardScaler()
        
        # Model weights and performance
        self.model_weights: Dict[str, float] = {}
        self.model_performance: Dict[str, Dict[str, float]] = {}
        self.cross_validation_scores: Dict[str, List[float]] = {}
        
        # Training state
        self.base_models_trained = False
        self.meta_learner_trained = False
        
        # Accuracy tracking integration
        self.tracking_manager = tracking_manager
    
    async def train(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame] = None,
        validation_targets: Optional[pd.DataFrame] = None
    ) -> TrainingResult:
        """
        Train the ensemble using stacking with cross-validation.
        
        Args:
            features: Training feature matrix
            targets: Training target values
            validation_features: Optional validation features
            validation_targets: Optional validation targets
            
        Returns:
            TrainingResult with ensemble training statistics
        """
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting ensemble training for room {self.room_id}")
            logger.info(f"Training data shape: {features.shape}")
            
            if len(features) < 50:
                raise ModelTrainingError(
                    f"Insufficient data for ensemble training: only {len(features)} samples"
                )
            
            # Phase 1: Train base models with cross-validation for meta-features
            logger.info("Phase 1: Training base models and generating meta-features")
            meta_features = await self._train_base_models_cv(features, targets)
            
            # Phase 2: Train meta-learner on meta-features
            logger.info("Phase 2: Training meta-learner")
            await self._train_meta_learner(meta_features, targets, features)
            
            # Phase 3: Final training of base models on full data
            logger.info("Phase 3: Final training of base models on full data")
            await self._train_base_models_final(features, targets, validation_features, validation_targets)
            
            # Calculate ensemble performance
            ensemble_predictions = await self._predict_ensemble(features)
            y_true = self._prepare_targets(targets)
            
            training_score = r2_score(y_true, ensemble_predictions)
            training_mae = mean_absolute_error(y_true, ensemble_predictions)
            training_rmse = np.sqrt(mean_squared_error(y_true, ensemble_predictions))
            
            # Validation performance
            validation_score = None
            validation_mae = None
            validation_rmse = None
            
            if validation_features is not None and validation_targets is not None:
                val_predictions = await self._predict_ensemble(validation_features)
                y_val_true = self._prepare_targets(validation_targets)
                
                validation_score = r2_score(y_val_true, val_predictions)
                validation_mae = mean_absolute_error(y_val_true, val_predictions)
                validation_rmse = np.sqrt(mean_squared_error(y_val_true, val_predictions))
            
            # Update model state  
            self.feature_names = list(features.columns)
            self.is_trained = True
            self.training_date = datetime.utcnow()
            self.model_version = self._generate_model_version()
            
            # Calculate training time
            training_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Compile training metrics
            training_metrics = {
                'ensemble_mae': training_mae,
                'ensemble_rmse': training_rmse,
                'ensemble_r2': training_score,
                'base_model_count': len(self.base_models),
                'meta_learner_type': self.model_params['meta_learner'],
                'cv_folds': self.model_params['cv_folds'],
                'model_weights': self.model_weights.copy(),
                'base_model_performance': self.model_performance.copy()
            }
            
            if validation_score is not None:
                training_metrics.update({
                    'ensemble_validation_mae': validation_mae,
                    'ensemble_validation_rmse': validation_rmse,
                    'ensemble_validation_r2': validation_score
                })
            
            # Add individual model CV scores
            for model_name, cv_scores in self.cross_validation_scores.items():
                training_metrics[f'{model_name}_cv_mean'] = np.mean(cv_scores)
                training_metrics[f'{model_name}_cv_std'] = np.std(cv_scores)
            
            result = TrainingResult(
                success=True,
                training_time_seconds=training_time,
                model_version=self.model_version,
                training_samples=len(features),
                validation_score=validation_score,
                training_score=training_score,
                training_metrics=training_metrics
            )
            
            self.training_history.append(result)
            
            logger.info(f"Ensemble training completed in {training_time:.2f}s")
            logger.info(f"Ensemble R²: {training_score:.4f}, Validation R²: {validation_score}")
            logger.info(f"Model weights: {self.model_weights}")
            
            return result
            
        except Exception as e:
            training_time = (datetime.utcnow() - start_time).total_seconds()
            error_msg = f"Ensemble training failed: {str(e)}"
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
        Generate ensemble predictions by combining base model predictions.
        
        Args:
            features: Feature matrix for prediction
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known
            
        Returns:
            List of ensemble prediction results
        """
        if not self.is_trained or not self.meta_learner_trained:
            raise ModelPredictionError("Ensemble is not fully trained")
        
        if not self.validate_features(features):
            raise ModelPredictionError("Feature validation failed")
        
        try:
            # Get predictions from all base models
            base_predictions = {}
            base_results = {}
            
            for model_name, model in self.base_models.items():
                if model.is_trained:
                    model_results = await model.predict(features, prediction_time, current_state)
                    base_results[model_name] = model_results
                    
                    # Extract prediction values for meta-learner
                    base_predictions[model_name] = [
                        (r.predicted_time - prediction_time).total_seconds()
                        for r in model_results
                    ]
            
            if not base_predictions:
                raise ModelPredictionError("No base models available for prediction")
            
            # Create meta-features
            meta_features_df = self._create_meta_features(base_predictions, features)
            
            # Get ensemble predictions from meta-learner
            ensemble_predictions = self.meta_learner.predict(meta_features_df)
            
            # Combine base model predictions with meta-learner output
            ensemble_results = self._combine_predictions(
                base_results, ensemble_predictions, prediction_time, current_state
            )
            
            return ensemble_results
            
        except Exception as e:
            error_msg = f"Ensemble prediction failed: {str(e)}"
            logger.error(error_msg)
            raise ModelPredictionError(error_msg)
    
    def get_feature_importance(self) -> Dict[str, float]:
        """
        Get ensemble feature importance combining all base models.
        
        Returns:
            Dictionary mapping feature names to combined importance scores
        """
        if not self.is_trained:
            return {}
        
        combined_importance = {}
        total_weight = 0
        
        # Combine feature importance from base models weighted by performance
        for model_name, model in self.base_models.items():
            if model.is_trained and model_name in self.model_weights:
                model_importance = model.get_feature_importance()
                weight = self.model_weights[model_name]
                
                for feature_name, importance in model_importance.items():
                    if feature_name not in combined_importance:
                        combined_importance[feature_name] = 0
                    combined_importance[feature_name] += importance * weight
                
                total_weight += weight
        
        # Normalize by total weight
        if total_weight > 0:
            combined_importance = {
                k: v / total_weight for k, v in combined_importance.items()
            }
        
        return combined_importance
    
    async def _train_base_models_cv(
        self, 
        features: pd.DataFrame, 
        targets: pd.DataFrame
    ) -> pd.DataFrame:
        """Train base models with cross-validation to generate meta-features."""
        cv = KFold(n_splits=self.model_params['cv_folds'], shuffle=True, random_state=42)
        
        # Initialize meta-features array
        n_samples = len(features)
        n_models = len(self.base_models)
        meta_features = np.zeros((n_samples, n_models))
        model_names = list(self.base_models.keys())
        
        # Cross-validation for each base model
        for fold, (train_idx, val_idx) in enumerate(cv.split(features)):
            logger.info(f"Processing fold {fold + 1}/{self.model_params['cv_folds']}")
            
            X_train_fold = features.iloc[train_idx]
            y_train_fold = targets.iloc[train_idx] 
            X_val_fold = features.iloc[val_idx]
            
            # Train each base model on this fold
            for model_idx, (model_name, model) in enumerate(self.base_models.items()):
                try:
                    # Create fresh model instance for this fold
                    if model_name == 'lstm':
                        fold_model = LSTMPredictor(self.room_id)
                    elif model_name == 'xgboost':
                        fold_model = XGBoostPredictor(self.room_id)
                    elif model_name == 'hmm':
                        fold_model = HMMPredictor(self.room_id)
                    else:
                        continue
                    
                    # Train on fold
                    await fold_model.train(X_train_fold, y_train_fold)
                    
                    # Predict on validation set
                    val_predictions = await fold_model.predict(
                        X_val_fold, datetime.utcnow(), 'unknown'
                    )
                    
                    # Extract time until transition for meta-features
                    for i, pred in enumerate(val_predictions):
                        original_idx = val_idx[i]
                        time_until = (pred.predicted_time - datetime.utcnow()).total_seconds()
                        meta_features[original_idx, model_idx] = time_until
                    
                except Exception as e:
                    logger.warning(f"Model {model_name} failed on fold {fold}: {e}")
                    # Fill with default predictions
                    for i in val_idx:
                        meta_features[i, model_idx] = 1800.0  # 30 minutes default
        
        # Convert to DataFrame
        meta_features_df = pd.DataFrame(meta_features, columns=model_names)
        
        # Store cross-validation performance
        y_true = self._prepare_targets(targets)
        for model_idx, model_name in enumerate(model_names):
            model_preds = meta_features[:, model_idx]
            cv_score = r2_score(y_true, model_preds)
            self.cross_validation_scores[model_name] = [cv_score]
        
        return meta_features_df
    
    async def _train_meta_learner(
        self, 
        meta_features: pd.DataFrame, 
        targets: pd.DataFrame,
        original_features: pd.DataFrame
    ):
        """Train the meta-learner on meta-features."""
        y_true = self._prepare_targets(targets)
        
        # Combine meta-features with original features if specified
        if self.model_params['use_base_features'] and not self.model_params['meta_features_only']:
            # Add subset of original features
            important_features = original_features.iloc[:, :20]  # First 20 features
            X_meta = pd.concat([meta_features, important_features], axis=1)
        else:
            X_meta = meta_features
        
        # Scale features
        X_meta_scaled = self.meta_scaler.fit_transform(X_meta)
        
        # Create meta-learner
        if self.model_params['meta_learner'] == 'random_forest':
            self.meta_learner = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
        else:
            # Default to linear regression
            from sklearn.linear_model import LinearRegression
            self.meta_learner = LinearRegression()
        
        # Train meta-learner
        self.meta_learner.fit(X_meta_scaled, y_true)
        self.meta_learner_trained = True
        
        # Calculate model weights based on meta-learner if possible
        self._calculate_model_weights(meta_features, y_true)
        
        logger.info("Meta-learner training completed")
    
    async def _train_base_models_final(
        self,
        features: pd.DataFrame,
        targets: pd.DataFrame,
        validation_features: Optional[pd.DataFrame],
        validation_targets: Optional[pd.DataFrame]
    ):
        """Train base models on full data for final predictions."""
        training_tasks = []
        
        for model_name, model in self.base_models.items():
            task = model.train(features, targets, validation_features, validation_targets)
            training_tasks.append((model_name, task))
        
        # Train models concurrently
        for model_name, task in training_tasks:
            try:
                result = await task
                self.model_performance[model_name] = {
                    'training_score': result.training_score or 0.0,
                    'validation_score': result.validation_score or 0.0,
                    'training_mae': result.training_metrics.get('training_mae', 0.0) if result.training_metrics else 0.0
                }
                logger.info(f"Base model {model_name} trained successfully")
            except Exception as e:
                logger.error(f"Base model {model_name} training failed: {e}")
                self.model_performance[model_name] = {
                    'training_score': 0.0,
                    'validation_score': 0.0,
                    'training_mae': float('inf')
                }
        
        self.base_models_trained = True
    
    def _calculate_model_weights(self, meta_features: pd.DataFrame, y_true: np.ndarray):
        """Calculate model weights based on individual performance."""
        self.model_weights = {}
        
        for model_name in meta_features.columns:
            model_preds = meta_features[model_name].values
            
            # Calculate R² score for this model
            score = r2_score(y_true, model_preds)
            
            # Convert to weight (higher score = higher weight)
            weight = max(0.1, score) if score > 0 else 0.1
            self.model_weights[model_name] = weight
        
        # Normalize weights to sum to 1
        total_weight = sum(self.model_weights.values())
        if total_weight > 0:
            self.model_weights = {
                k: v / total_weight for k, v in self.model_weights.items()
            }
    
    def _create_meta_features(
        self, 
        base_predictions: Dict[str, List[float]], 
        original_features: pd.DataFrame
    ) -> pd.DataFrame:
        """Create meta-features for ensemble prediction."""
        # Convert base predictions to DataFrame
        max_len = max(len(preds) for preds in base_predictions.values())
        
        meta_features = {}
        for model_name, preds in base_predictions.items():
            # Pad predictions if necessary
            padded_preds = preds + [preds[-1]] * (max_len - len(preds)) if preds else [1800.0] * max_len
            meta_features[model_name] = padded_preds[:max_len]
        
        meta_df = pd.DataFrame(meta_features)
        
        # Add original features if configured
        if self.model_params['use_base_features'] and not self.model_params['meta_features_only']:
            important_features = original_features.iloc[:max_len, :20]  # First 20 features
            important_features.index = meta_df.index
            meta_df = pd.concat([meta_df, important_features], axis=1)
        
        # Scale features
        meta_scaled = self.meta_scaler.transform(meta_df)
        return pd.DataFrame(meta_scaled, columns=meta_df.columns)
    
    async def _predict_ensemble(self, features: pd.DataFrame) -> np.ndarray:
        """Generate ensemble predictions for training evaluation."""
        # Get base model predictions
        base_predictions = {}
        
        for model_name, model in self.base_models.items():
            if model.is_trained:
                try:
                    results = await model.predict(features, datetime.utcnow(), 'unknown')
                    base_predictions[model_name] = [
                        (r.predicted_time - datetime.utcnow()).total_seconds()
                        for r in results
                    ]
                except Exception as e:
                    logger.warning(f"Base model {model_name} prediction failed: {e}")
                    # Use default predictions
                    base_predictions[model_name] = [1800.0] * len(features)
        
        # Create meta-features and predict
        if base_predictions and self.meta_learner_trained:
            meta_features_df = self._create_meta_features(base_predictions, features)
            return self.meta_learner.predict(meta_features_df)
        else:
            # Fallback: simple average
            if base_predictions:
                all_preds = np.array(list(base_predictions.values()))
                return np.mean(all_preds, axis=0)
            else:
                return np.full(len(features), 1800.0)
    
    def _combine_predictions(
        self,
        base_results: Dict[str, List[PredictionResult]],
        ensemble_predictions: np.ndarray,
        prediction_time: datetime,
        current_state: str
    ) -> List[PredictionResult]:
        """
        Combine base model predictions with meta-learner output into final ensemble results.
        
        Args:
            base_results: Predictions from individual base models
            ensemble_predictions: Meta-learner predictions (time until transition)
            prediction_time: Time when prediction is being made
            current_state: Current occupancy state if known
            
        Returns:
            List of combined ensemble prediction results
        """
        ensemble_results = []
        
        for idx, ensemble_time_until in enumerate(ensemble_predictions):
            # Ensure reasonable bounds
            ensemble_time_until = np.clip(ensemble_time_until, 60, 86400)
            
            # Calculate predicted time
            predicted_time = prediction_time + timedelta(seconds=ensemble_time_until)
            
            # Determine transition type (use XGBoost if available, otherwise heuristic)
            if 'xgboost' in base_results and idx < len(base_results['xgboost']):
                transition_type = base_results['xgboost'][idx].transition_type
            else:
                # Fallback logic
                hour = prediction_time.hour
                if current_state == 'occupied':
                    transition_type = 'occupied_to_vacant'
                elif current_state == 'vacant':
                    transition_type = 'vacant_to_occupied'
                else:
                    transition_type = 'vacant_to_occupied' if 6 <= hour <= 22 else 'occupied_to_vacant'
            
            # Calculate ensemble confidence
            confidence = self._calculate_ensemble_confidence(
                base_results, idx, ensemble_time_until
            )
            
            # Gather alternative predictions from base models
            alternatives = []
            for model_name, results in base_results.items():
                if idx < len(results):
                    alternatives.append((
                        results[idx].predicted_time,
                        results[idx].confidence_score
                    ))
            
            # Extract base model predictions for metadata
            base_predictions = {}
            for model_name, results in base_results.items():
                if idx < len(results):
                    time_until = (results[idx].predicted_time - prediction_time).total_seconds()
                    base_predictions[model_name] = float(time_until)
            
            # Create ensemble prediction result
            result = PredictionResult(
                predicted_time=predicted_time,
                transition_type=transition_type,
                confidence_score=confidence,
                alternatives=alternatives[:3],  # Top 3 alternatives
                model_type=self.model_type.value,
                model_version=self.model_version,
                features_used=self.feature_names,
                prediction_metadata={
                    'time_until_transition_seconds': float(ensemble_time_until),
                    'prediction_method': 'stacking_ensemble',
                    'base_model_predictions': base_predictions,
                    'model_weights': self.model_weights.copy(),
                    'meta_learner_type': self.model_params['meta_learner'],
                    'combination_method': 'meta_learner_weighted'
                }
            )
            
            ensemble_results.append(result)
            
            # Record prediction for accuracy tracking if tracking manager is available
            if self.tracking_manager:
                await self.tracking_manager.record_prediction(result)
        
        return ensemble_results
    
    def _calculate_ensemble_confidence(
        self, 
        base_results: Dict[str, List[PredictionResult]], 
        idx: int, 
        ensemble_prediction: float
    ) -> float:
        """Calculate confidence for ensemble prediction."""
        confidences = []
        predictions = []
        
        for model_name, results in base_results.items():
            if idx < len(results):
                confidences.append(results[idx].confidence_score)
                pred_time = (results[idx].predicted_time - datetime.utcnow()).total_seconds()
                predictions.append(pred_time)
        
        if not confidences:
            return 0.7
        
        # Weighted average confidence
        weights = [self.model_weights.get(name, 1.0) for name in base_results.keys()]
        weighted_confidence = np.average(confidences, weights=weights)
        
        # Adjust confidence based on prediction agreement
        if len(predictions) > 1:
            pred_std = np.std(predictions)
            agreement_factor = 1.0 / (1.0 + pred_std / 3600.0)  # Normalize by 1 hour
            weighted_confidence *= agreement_factor
        
        return float(np.clip(weighted_confidence, 0.1, 0.95))
    
    def _prepare_targets(self, targets: pd.DataFrame) -> np.ndarray:
        """Prepare target values from DataFrame."""
        if 'time_until_transition_seconds' in targets.columns:
            target_values = targets['time_until_transition_seconds'].values
        elif 'next_transition_time' in targets.columns and 'target_time' in targets.columns:
            target_times = pd.to_datetime(targets['target_time'])
            next_times = pd.to_datetime(targets['next_transition_time'])
            target_values = (next_times - target_times).dt.total_seconds().values
        else:
            target_values = targets.iloc[:, 0].values
        
        return np.clip(target_values, 60, 86400)
    
    def get_ensemble_info(self) -> Dict[str, Any]:
        """Get detailed information about the ensemble."""
        return {
            'ensemble_type': 'stacking',
            'base_models': list(self.base_models.keys()),
            'meta_learner': self.model_params['meta_learner'],
            'model_weights': self.model_weights,
            'model_performance': self.model_performance,
            'cv_scores': self.cross_validation_scores,
            'is_trained': self.is_trained,
            'base_models_trained': self.base_models_trained,
            'meta_learner_trained': self.meta_learner_trained
        }