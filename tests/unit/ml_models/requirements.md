# ML Models Testing Requirements

## Overview
This document contains detailed testing requirements for the ha-ml-predictor ML models components to achieve 85%+ test coverage. Each component has been analyzed for actual implementation details and specific testing scenarios.

### src/models/base/predictor.py - Base Predictor Interface
**Classes Found:** PredictionResult (dataclass), TrainingResult (dataclass), BasePredictor (abstract base class)
**Methods Analyzed:** PredictionResult.to_dict(), TrainingResult.to_dict(), BasePredictor.__init__(), BasePredictor.train() (abstract), BasePredictor.predict() (abstract), BasePredictor.get_feature_importance() (abstract), BasePredictor.predict_single(), BasePredictor.save_model(), BasePredictor.load_model(), BasePredictor.get_model_info(), BasePredictor.get_training_history(), BasePredictor.get_prediction_accuracy(), BasePredictor.clear_prediction_history(), BasePredictor.validate_features(), BasePredictor._record_prediction(), BasePredictor._generate_model_version(), BasePredictor.__str__(), BasePredictor.__repr__()

**Required Tests:**
**Unit Tests:** 
- **PredictionResult Tests:** Test dataclass initialization with all field combinations, test to_dict() serialization with datetime ISO formatting, test optional field handling (None values), test alternatives list serialization, test prediction_interval tuple handling, test all possible field value combinations
- **TrainingResult Tests:** Test dataclass initialization with required/optional fields, test to_dict() serialization with all field types, test success/failure states, test feature_importance dictionary handling, test training_metrics dictionary serialization
- **BasePredictor Initialization Tests:** Test __init__ with ModelType enum, room_id assignment, config parameter handling, test default values for state variables (is_trained=False, model_version="v1.0"), test training_date None initialization, test empty lists initialization (feature_names, training_history, prediction_history)
- **Model Serialization Tests:** Test save_model() with pickle serialization, test model data dictionary structure, test file path handling (Path and string), test model persistence with all state variables, test load_model() deserialization, test model state restoration, test training history restoration from dictionaries, test file error handling in save/load operations
- **Model Information Tests:** Test get_model_info() dictionary structure, test model state reflection in info, test feature count calculations, test feature names truncation (first 10), test training statistics calculations, test get_training_history() list conversion, test history ordering and completeness
- **Prediction Management Tests:** Test predict_single() DataFrame conversion from dict, test feature dictionary to DataFrame transformation, test prediction result extraction (first result), test prediction history recording via _record_prediction(), test prediction history memory management (1000 limit, 500 truncation), test get_prediction_accuracy() with time filtering, test clear_prediction_history() functionality
- **Feature Validation Tests:** Test validate_features() with trained vs untrained models, test feature name matching against self.feature_names, test missing features detection and error reporting, test extra features warning generation, test empty feature_names handling, test DataFrame column validation
- **Version Management Tests:** Test _generate_model_version() with empty history, test version incrementation logic (v1.0 -> v1.1), test version parsing and float conversion, test fallback to base version on parse errors, test version string format validation
- **String Representation Tests:** Test __str__() format with/without room_id, test trained/untrained status display, test __repr__() detailed format with all parameters, test model_type enum value display

**Integration Tests:**
- **Abstract Method Integration:** Test that BasePredictor cannot be instantiated directly (TypeError), test that concrete subclasses must implement train(), predict(), get_feature_importance() methods, test abstract method signature validation, test inheritance behavior with multiple concrete implementations
- **Model Persistence Integration:** Test complete save/load cycle with real file system, test persistence across different Python sessions, test model state consistency after load, test error recovery from corrupted pickle files
- **Configuration Integration:** Test BasePredictor with SystemConfig integration, test config parameter propagation to subclasses, test room_id validation with config room definitions
- **Feature Store Integration:** Test feature validation with actual feature engineering outputs, test DataFrame compatibility with feature store formats, test feature name consistency across system components

**Edge Cases:**
- **Data Type Edge Cases:** Test PredictionResult with extreme datetime values (timezone aware), test alternatives with empty lists vs None, test prediction_interval with same start/end times, test confidence_score boundary values (0.0, 1.0), test transition_type validation with edge strings
- **Model State Edge Cases:** Test BasePredictor with very large training histories (memory management), test prediction_history with concurrent access patterns, test model version generation with malformed version strings, test feature names with special characters and Unicode
- **Serialization Edge Cases:** Test pickle serialization with very large model objects, test model saving with insufficient disk space, test loading models created with different library versions, test corrupted pickle file recovery
- **Validation Edge Cases:** Test validate_features with empty DataFrames, test feature validation with duplicate column names, test DataFrame with mixed data types, test very large feature sets (>1000 features)

**Error Handling:**
- **Pickle Serialization Errors:** Test save_model() with pickle errors, test load_model() with corrupted files, test exception handling and logging for file operations, test graceful failure with informative error messages
- **Abstract Method Errors:** Test TypeError when instantiating BasePredictor directly, test NotImplementedError propagation from abstract methods, test proper error messages for missing implementations
- **Feature Validation Errors:** Test ModelPredictionError in predict_single() with empty predictions, test feature validation errors with detailed missing feature reporting, test validation failure recovery and logging
- **File System Errors:** Test model persistence with permission denied errors, test file not found errors in load_model(), test disk full scenarios during model saving, test network drive issues for model storage
- **Memory Management Errors:** Test prediction history overflow handling, test memory constraints with large model objects, test garbage collection of cleared histories

**Coverage Target:** 85%+

### src/models/base/xgboost_predictor.py - XGBoost Gradient Boosting
**Classes Found:** XGBoostPredictor
**Methods Analyzed:** __init__, train, predict, get_feature_importance, _prepare_targets, _calculate_confidence, _calculate_feature_contributions, save_model, load_model, _validate_prediction_time, get_model_complexity, incremental_update

**Required Tests:**
**Unit Tests:**
- Test XGBoostPredictor.__init__ with default and custom model parameters
- Test model_params initialization with DEFAULT_MODEL_PARAMS integration
- Test feature_scaler initialization with StandardScaler 
- Test BasePredictor initialization with ModelType.XGBOOST
- Test train method with valid features and targets DataFrames
- Test train method target preparation using _prepare_targets()
- Test train method feature scaling with StandardScaler.fit_transform
- Test train method XGBRegressor creation with model_params
- Test train method model fitting with prepared data
- Test train method training metrics calculation (R², MAE, RMSE)
- Test train method with validation data processing
- Test train method TrainingResult creation with success/failure states
- Test predict method with untrained model raising ModelPredictionError
- Test predict method feature validation through validate_features()
- Test predict method feature scaling with fitted scaler
- Test predict method XGBRegressor prediction generation
- Test predict method time bounds clipping (60-86400 seconds)
- Test predict method predicted time calculation with timedelta
- Test predict method transition type determination
- Test predict method confidence calculation integration
- Test predict method PredictionResult creation
- Test get_feature_importance with trained XGBoost model
- Test get_feature_importance with untrained model returning empty dict
- Test get_feature_importance score normalization
- Test _prepare_targets with different DataFrame column formats
- Test _prepare_targets with time_until_transition_seconds column
- Test _prepare_targets with datetime calculation from target columns
- Test _prepare_targets with target clipping and validation
- Test _calculate_confidence using training history and feature contributions
- Test _calculate_confidence with various prediction scenarios
- Test _calculate_feature_contributions with SHAP-like analysis
- Test save_model/load_model with complete state preservation
- Test _validate_prediction_time with boundary checking
- Test get_model_complexity with XGBoost model introspection
- Test incremental_update with new training data

**Integration Tests:**
- Test complete XGBoost training workflow with realistic data
- Test integration with StandardScaler from sklearn.preprocessing
- Test integration with XGBRegressor from xgboost library
- Test integration with ModelTrainingError and ModelPredictionError exceptions
- Test integration with PredictionResult and TrainingResult dataclasses
- Test integration with pandas DataFrame operations throughout
- Test integration with numpy array operations and clipping
- Test integration with datetime timezone handling
- Test feature validation through BasePredictor.validate_features
- Test prediction recording through BasePredictor._record_prediction
- Test model versioning through BasePredictor._generate_model_version
- Test logging integration with proper room_id context

**Edge Cases:**
- Test with empty features DataFrame
- Test with single feature column
- Test with features containing NaN or infinite values
- Test with targets containing negative or zero values
- Test with extremely large target values (> 24 hours)
- Test with features having duplicate column names
- Test with mismatched features/targets row counts
- Test with validation data having different feature columns
- Test with model parameters containing invalid values
- Test with save_model using invalid file paths or permissions
- Test with load_model using non-existent or corrupted files
- Test with pickle serialization failures
- Test with XGBoost model creation failures
- Test with feature scaling failures (constant features, etc.)
- Test with confidence calculation edge cases (NaN values, extreme features)
- Test with feature importance calculation when model not fitted
- Test with very short or very long time predictions
- Test with prediction at timezone boundaries
- Test with feature contributions when importance is empty

**Error Handling:**
- Test ModelTrainingError raised with proper model_type and room_id
- Test ModelPredictionError raised with proper context information
- Test exception handling during XGBRegressor creation and fitting
- Test exception handling during feature scaling operations
- Test exception handling during prediction generation
- Test exception handling during confidence calculation
- Test exception handling during feature contribution calculation
- Test exception handling during model save/load operations
- Test logging of error messages with appropriate levels
- Test graceful handling of pickle import/export failures
- Test handling of corrupted model files during loading
- Test handling of version compatibility issues in saved models
- Test exception propagation from BasePredictor methods

**Coverage Target:** 85%+

### src/models/base/lstm_predictor.py - LSTM Neural Networks
**Classes Found:** LSTMPredictor
**Methods Analyzed:** __init__, train, predict, get_feature_importance, _create_sequences, _calculate_confidence, get_model_complexity, save_model, load_model, incremental_update

**Required Tests:**
**Unit Tests:**
- **LSTMPredictor.__init__() Tests:**
  - Test initialization with default parameters (no room_id, no kwargs)
  - Test initialization with room_id parameter
  - Test initialization with custom kwargs overriding DEFAULT_MODEL_PARAMS
  - Test hidden_units parameter handling (int vs list conversion)
  - Test model_params dictionary structure and all parameter aliases (hidden_size, lstm_units, etc.)
  - Test sequence_length and sequence_step initialization
  - Test feature_scaler (StandardScaler) and target_scaler (MinMaxScaler) initialization
  - Test training statistics lists initialization (training_loss_history, validation_loss_history)
  - Test parameter alias handling (dropout vs dropout_rate)
  - Test MLPRegressor model initialization as None
  - Test BasePredictor parent class initialization with ModelType.LSTM

- **train() Method Core Logic Tests:**
  - Test training with valid features and targets DataFrames
  - Test adaptive sequence length reduction for small datasets (<200 samples)
  - Test sequence_step adjustment for small datasets (step = 1)
  - Test minimum sequence requirement validation (< 2 sequences raises ModelTrainingError)
  - Test _create_sequences() integration and sequence generation logging
  - Test feature and target scaling with StandardScaler and MinMaxScaler
  - Test validation data preparation when provided (validation_features, validation_targets)
  - Test MLPRegressor creation with proper parameters from model_params
  - Test model training with fit() and random_state=42, warm_start=False
  - Test feature_names storage from input DataFrame columns
  - Test training metrics calculation (MAE, RMSE, R²) with inverse_transform
  - Test validation metrics calculation when validation data provided
  - Test model state updates (is_trained=True, training_date, model_version)
  - Test TrainingResult creation with comprehensive training metrics
  - Test training_history append operation
  - Test success logging with training time and scores
  - Test sequence length restoration in finally block

- **predict() Method Core Logic Tests:**
  - Test prediction validation (is_trained and model existence checks)
  - Test feature validation with validate_features() integration
  - Test training_sequence_length usage for prediction consistency
  - Test sequence creation for prediction with padding for insufficient history
  - Test feature sequence flattening for MLPRegressor input format
  - Test feature scaling with transform() using fitted scaler
  - Test model prediction with predict() method
  - Test target inverse_transform for actual time values
  - Test time bounds clipping (60 seconds to 86400 seconds)
  - Test predicted_time calculation with timedelta addition
  - Test transition_type determination based on current_state
  - Test default transition_type logic based on hour of day (6-22 vs nighttime)
  - Test confidence calculation with _calculate_confidence()
  - Test PredictionResult creation with comprehensive metadata
  - Test prediction recording with _record_prediction()
  - Test multiple predictions for DataFrame with multiple rows

- **_create_sequences() Core Logic Tests:**
  - Test sequence creation with valid features and targets DataFrames
  - Test input validation (equal length features/targets, minimum sequence_length)
  - Test target value extraction from different column formats (time_until_transition_seconds, next_transition_time/target_time, default)
  - Test target value validation and numeric conversion with pd.to_numeric()
  - Test sequence generation with corrected bounds checking (sequence_length to len(features))
  - Test sequence step handling with self.sequence_step
  - Test X_seq flattening for MLPRegressor compatibility
  - Test target value bounds filtering (60 to 86400 seconds)
  - Test sequence validation and array creation
  - Test final validation of X_array and y_array shapes
  - Test sequence generation logging

- **Additional Method Tests:**
  - Test get_feature_importance() neural network weight analysis using model.coefs_[0]
  - Test _calculate_confidence() using training history validation scores
  - Test save_model() pickle serialization of complete model state
  - Test load_model() complete state restoration (model, scalers, parameters, history)
  - Test get_model_complexity() parameter counting from model.coefs_ and model.intercepts_
  - Test incremental_update() with new features/targets and warm_start=True

**Integration Tests:**
- **Model Training Integration:**
  - Test complete training workflow with realistic sensor event data
  - Test training with actual pandas DataFrames from feature extraction
  - Test integration with DEFAULT_MODEL_PARAMS from core.constants
  - Test integration with ModelType enum and BasePredictor inheritance
  - Test training with various DataFrame column formats and structures
  - Test validation split and cross-validation integration

- **Model Prediction Integration:**
  - Test prediction pipeline with real feature DataFrames
  - Test integration with PredictionResult and TrainingResult classes
  - Test prediction recording and accuracy tracking integration
  - Test confidence calculation with real model performance data
  - Test transition type logic with actual room occupancy patterns

- **Persistence Integration:**
  - Test model saving/loading with complete training state
  - Test pickle serialization compatibility across sessions
  - Test model version tracking and history persistence
  - Test scaler state preservation and restoration

**Edge Cases:**
- **Data Quality Edge Cases:**
  - Test with features/targets having NaN or infinite values
  - Test with features having zero variance (constant columns)
  - Test with targets having extreme outliers or impossible values
  - Test with very small datasets (< 10 samples) and sequence adaptation
  - Test with very large datasets and memory management
  - Test with features having different column counts across calls
  - Test with non-numeric data in features requiring error handling

- **Sequence Processing Edge Cases:**
  - Test sequence creation with edge sequence_length values (1, 2, very large)
  - Test sequence creation with sequence_step larger than data length
  - Test sequence generation with identical timestamps
  - Test sequence flattening with different feature dimensionalities
  - Test target value extraction with edge timestamp formats

- **Model Training Edge Cases:**
  - Test MLPRegressor with edge case parameters (single neuron, large networks)
  - Test training convergence with difficult optimization landscapes
  - Test scaler fitting with constant or near-constant data
  - Test model training with perfect correlations or rank-deficient data
  - Test adaptive sequence length with boundary conditions

**Error Handling:**
- **Training Error Scenarios:**
  - Test ModelTrainingError with various underlying causes
  - Test scaler fitting failures with appropriate error propagation
  - Test MLPRegressor initialization failures
  - Test sequence generation failures with informative error messages
  - Test training timeout or memory exhaustion scenarios

- **Prediction Error Scenarios:**
  - Test ModelPredictionError for various failure modes
  - Test feature validation failures and error messaging
  - Test scaler transform failures on new data
  - Test model prediction failures with corrupted model state
  - Test confidence calculation failures with fallback handling

**Coverage Target:** 85%+

### src/models/base/gp_predictor.py - Gaussian Process Models
**Classes Found:** GaussianProcessPredictor
**Methods Analyzed:** __init__, _create_kernel, train, predict, get_feature_importance, _select_inducing_points, _calibrate_uncertainty, _calculate_confidence_intervals, _calculate_confidence_score, _generate_alternative_scenarios, _estimate_epistemic_uncertainty, _determine_transition_type, _prepare_targets, get_uncertainty_metrics, incremental_update, save_model, load_model, get_model_complexity

**Required Tests:**
**Unit Tests:**
- **Initialization Tests:**
  - Test GaussianProcessPredictor initialization with default parameters
  - Test initialization with custom kernel types ('rbf', 'matern', 'periodic', 'rational_quadratic', 'composite')
  - Test initialization with different confidence intervals and uncertainty parameters
  - Test initialization with sparse GP parameters and max inducing points

- **Kernel Creation Tests:**
  - Test _create_kernel() with different kernel types (rbf, matern, periodic, rational_quadratic, composite)
  - Test fallback behavior when PeriodicKernel is not available
  - Test composite kernel creation with multiple components (local, trend, daily, weekly, noise)
  - Test kernel parameter bounds and initialization values

- **Training Tests:**
  - Test successful training with valid features and targets DataFrames
  - Test training with validation data provided
  - Test training with insufficient data (< 10 samples) raises ModelTrainingError
  - Test sparse GP activation when data exceeds max_inducing_points threshold
  - Test feature scaling with StandardScaler fit_transform
  - Test kernel parameter optimization and log marginal likelihood calculation
  - Test training history recording and TrainingResult creation

- **Prediction Tests:**
  - Test predict() with trained model returns PredictionResult list
  - Test prediction with uncertainty quantification (mean, std)
  - Test confidence interval calculation for different confidence levels (68%, 95%, 99%)
  - Test prediction without trained model raises ModelPredictionError
  - Test prediction with invalid features raises ModelPredictionError
  - Test alternative scenario generation based on uncertainty
  - Test transition type determination based on current state and time

- **Feature Importance Tests:**
  - Test get_feature_importance() with ARD kernel (individual length scales)
  - Test feature importance with single length scale kernel
  - Test fallback uniform importance when kernel parameters unavailable
  - Test handling of missing or invalid kernel parameters

- **Uncertainty Quantification Tests:**
  - Test _calibrate_uncertainty() with validation data
  - Test uncertainty calibration curve calculation and storage
  - Test confidence score calculation based on prediction uncertainty
  - Test epistemic uncertainty estimation using training point distances
  - Test get_uncertainty_metrics() returning comprehensive uncertainty information

- **Sparse GP Tests:**
  - Test _select_inducing_points() with KMeans clustering
  - Test inducing point selection with different dataset sizes
  - Test duplicate removal and sorting in inducing point selection
  - Test sparse GP training vs full GP training paths

- **Additional Method Tests:**
  - Test incremental_update() with new training data
  - Test save_model() serializes all model components correctly
  - Test load_model() restores complete model state
  - Test _prepare_targets() with different DataFrame column formats

**Integration Tests:**
- **GP Model Training Integration:**
  - Test full training pipeline with realistic occupancy data
  - Test integration with different kernel types and their effect on predictions
  - Test validation data processing and accuracy metric calculation
  - Test sparse GP vs full GP performance comparison with large datasets

- **Scikit-learn Integration:**
  - Test compatibility with different scikit-learn versions
  - Test PeriodicKernel availability detection and fallback mechanisms
  - Test GaussianProcessRegressor parameter passing and optimization
  - Test StandardScaler integration with feature preprocessing

- **BasePredictor Integration:**
  - Test inheritance from BasePredictor and interface compliance
  - Test validate_features() integration and error handling
  - Test _record_prediction() integration with prediction tracking
  - Test model_version generation and training history management

**Edge Cases:**
- **Data Edge Cases:**
  - Test training with exactly 10 samples (boundary condition)
  - Test prediction with single-row DataFrame input
  - Test handling of NaN or infinite values in features/targets
  - Test empty DataFrame inputs and appropriate error handling
  - Test features with zero variance or constant values

- **Kernel Edge Cases:**
  - Test kernel creation with very small or very large feature dimensions
  - Test handling of kernel parameter optimization failures
  - Test kernel bounds validation and constraint enforcement
  - Test composite kernel with disabled periodic components

- **Uncertainty Edge Cases:**
  - Test confidence intervals with extremely low/high uncertainty values
  - Test calibration with insufficient validation data
  - Test uncertainty calculations with edge case statistical distributions
  - Test handling of negative or zero standard deviations

**Error Handling:**
- **Training Errors:**
  - Test ModelTrainingError with insufficient data
  - Test training failures due to kernel optimization issues
  - Test memory errors with extremely large datasets
  - Test feature scaling failures with invalid data
  - Test validation data shape mismatches

- **Prediction Errors:**
  - Test ModelPredictionError when model not trained
  - Test prediction failures with mismatched feature dimensions
  - Test handling of scikit-learn internal errors during prediction
  - Test feature scaling errors during prediction preprocessing

- **Numerical Errors:**
  - Test handling of numerical instability in GP calculations
  - Test matrix inversion failures in GP optimization
  - Test overflow/underflow in uncertainty calculations
  - Test division by zero in confidence score calculations

**Coverage Target:** 85%+

### src/models/base/hmm_predictor.py - Hidden Markov Models
**Classes Found:** HMMPredictor (extends BasePredictor)
**Methods Analyzed:** __init__, train, predict, get_feature_importance, _prepare_targets, _analyze_states, _assign_state_label, _build_transition_matrix, _train_state_duration_models, _predict_durations, _predict_single_duration, _determine_transition_type_from_states, _calculate_confidence, get_state_info, save_model, load_model, incremental_update, get_model_complexity, _predict_hmm_internal

**Required Tests:**
**Unit Tests:**
- Test HMMPredictor.__init__ with default parameters and parameter aliases (n_states/n_components)
- Test HMMPredictor.__init__ with n_iter/max_iter synchronization logic
- Test HMMPredictor.__init__ with model_params initialization and default values
- Test HMMPredictor.__init__ with GaussianMixture component initialization
- Test train method with valid features and targets DataFrames
- Test train method with insufficient training data (< 20 samples) raising ModelTrainingError
- Test train method with feature scaling using StandardScaler.fit_transform
- Test train method with KMeans pre-clustering for GMM initialization
- Test train method with GaussianMixture fitting and state prediction
- Test train method with state analysis, transition matrix building, and duration model training
- Test train method with training metrics calculation (R², MAE, RMSE)
- Test train method with validation data processing and metrics
- Test train method with TrainingResult creation and training_history update
- Test predict method with untrained model raising ModelPredictionError
- Test predict method with invalid features raising ModelPredictionError
- Test predict method with feature scaling and state probability prediction
- Test predict method with duration prediction and transition time calculation
- Test predict method with confidence calculation and PredictionResult creation
- Test predict method with prediction recording for accuracy tracking
- Test get_feature_importance with state discrimination power calculation
- Test get_feature_importance with covariance matrix analysis (full, diag, tied, spherical)
- Test get_feature_importance with importance score normalization
- Test get_feature_importance with untrained model returning empty dict
- Test _prepare_targets with time_until_transition_seconds column
- Test _prepare_targets with next_transition_time and target_time columns
- Test _prepare_targets with fallback to first column and clipping (60-86400)
- Test _prepare_targets with exception handling returning default values
- Test _analyze_states with state characteristics calculation and duration analysis
- Test _analyze_states with state probability analysis and confidence metrics
- Test _analyze_states with state labeling and reliability assessment
- Test _assign_state_label with duration-based heuristic labeling
- Test _build_transition_matrix with state transition counting and probability calculation
- Test _build_transition_matrix with uniform distribution for unobserved transitions
- Test _train_state_duration_models with LinearRegression for each state
- Test _train_state_duration_models with insufficient samples falling back to average
- Test _predict_durations with state identification and duration prediction
- Test _predict_single_duration with average and regression model types
- Test _predict_single_duration with default prediction for missing states
- Test _determine_transition_type_from_states with current occupancy inference
- Test _determine_transition_type_from_states with state characteristics analysis
- Test _calculate_confidence with state confidence and entropy adjustment
- Test _calculate_confidence with prediction reasonableness adjustment
- Test get_state_info returning n_states, labels, characteristics, transition_matrix
- Test save_model with pickle serialization of all model components
- Test save_model with file writing and success/failure handling
- Test load_model with pickle deserialization and component restoration
- Test load_model with training history reconstruction from dictionaries
- Test load_model with exception handling and logging
- Test incremental_update with existing trained model
- Test incremental_update with untrained model falling back to full training
- Test incremental_update with insufficient data raising ModelTrainingError
- Test incremental_update with new GMM training and state analysis
- Test incremental_update with transition model updates and performance calculation
- Test get_model_complexity with trained model returning component information
- Test get_model_complexity with untrained model returning default values
- Test _predict_hmm_internal with state predictions and transition model usage

**Integration Tests:**
- Test integration with BasePredictor, PredictionResult, and TrainingResult classes
- Test integration with ModelType.HMM and DEFAULT_MODEL_PARAMS
- Test integration with StandardScaler for feature normalization
- Test integration with GaussianMixture for hidden state modeling
- Test integration with KMeans for initial clustering
- Test integration with LinearRegression for state duration modeling
- Test integration with ModelTrainingError and ModelPredictionError exceptions
- Test integration with pandas DataFrame operations and numpy array processing
- Test integration with scikit-learn metrics (r2_score, mean_absolute_error, mean_squared_error)
- Test integration with pickle for model serialization/deserialization
- Test integration with logging for training and prediction monitoring
- Test end-to-end workflow from training to prediction with realistic sensor data

**Edge Cases:**
- Test with n_components = 1 (single state HMM)
- Test with n_components > number of samples (overfitting scenario)
- Test with all samples having identical features (zero variance)
- Test with features containing NaN or infinite values
- Test with extremely short or long prediction durations
- Test with empty state_durations dictionary
- Test with transition_matrix as None (no transitions observed)
- Test with covariance_type variations and singular matrices
- Test with state_labels assignment edge cases
- Test with model convergence failures (n_iter reached)
- Test with feature_scaler not fitted properly
- Test with corrupted pickle files during load_model
- Test with mismatched feature names between training and prediction
- Test with extremely large or small confidence scores
- Test with state_model predict_proba returning edge probabilities
- Test with prediction_time and datetime timezone handling
- Test with feature importance calculation edge cases (zero covariance)
- Test with transition model training with < 5 samples per state
- Test with incremental update on completely different data distribution

**Error Handling:**
- Test ModelTrainingError raising with proper model_type, room_id, and cause
- Test ModelPredictionError raising for untrained model scenarios
- Test exception handling in train method with proper TrainingResult error recording
- Test exception handling in predict method with detailed error logging
- Test exception handling in get_feature_importance with warning logs
- Test exception handling in _prepare_targets with default value fallback
- Test exception handling in save_model/load_model with file I/O errors
- Test exception handling in incremental_update with ModelTrainingError
- Test GaussianMixture fitting failures and recovery
- Test KMeans clustering failures with random_state handling
- Test LinearRegression training failures for individual states
- Test feature scaling errors with StandardScaler
- Test numpy array operations with invalid shapes or types
- Test transition matrix calculation with division by zero
- Test confidence calculation with logarithm domain errors
- Test pickle serialization/deserialization errors

**Coverage Target:** 85%+

### src/models/ensemble.py - Ensemble Model Architecture
**Classes Found:** OccupancyEnsemble, utility functions (_ensure_timezone_aware, _safe_time_difference)
**Methods Analyzed:** __init__, train, predict, get_feature_importance, incremental_update, _train_base_models_cv, _train_meta_learner, _train_base_models_final, _calculate_model_weights, _create_meta_features, _select_important_features, _predict_ensemble, _combine_predictions, _calculate_ensemble_confidence, _prepare_targets, _validate_training_data, get_ensemble_info, save_model, load_model

**Required Tests:**
**Unit Tests:** 
- OccupancyEnsemble.__init__() - test initialization with various parameters, default params, tracking manager integration
- train() - test complete training pipeline, insufficient data handling, cross-validation, meta-learner training, validation scenarios
- predict() - test ensemble prediction generation, base model failure handling, meta-feature creation, result combination
- get_feature_importance() - test weighted feature importance calculation from base models
- incremental_update() - test online learning updates, model weight recalculation, dimension handling
- _train_base_models_cv() - test cross-validation training, fold processing, error handling for failed models
- _train_meta_learner() - test meta-learner training, feature scaling, dimension alignment, NaN handling
- _train_base_models_final() - test concurrent base model training, performance metric collection
- _calculate_model_weights() - test weight calculation based on prediction consistency and accuracy
- _create_meta_features() - test meta-feature creation, dimension alignment, feature scaling edge cases
- _select_important_features() - test feature selection strategies, column renaming, dimension limits
- _predict_ensemble() - test ensemble prediction generation for training evaluation
- _combine_predictions() - test combination of base and meta-learner predictions, confidence calculation
- _calculate_ensemble_confidence() - test confidence calculation with GP uncertainty, prediction agreement
- _prepare_targets() - test target value extraction from different DataFrame formats
- _validate_training_data() - test comprehensive data validation, error conditions
- get_ensemble_info() - test ensemble metadata retrieval
- save_model()/load_model() - test model persistence and restoration
- _ensure_timezone_aware()/_safe_time_difference() - test timezone handling utilities

**Integration Tests:**
- Full training pipeline with real base models (LSTM, XGBoost, HMM, GP)
- End-to-end prediction workflow with feature engineering integration
- Tracking manager integration for automatic accuracy tracking
- Model serialization/deserialization with complex ensemble state
- Concurrent base model training with various failure scenarios
- Cross-validation performance across different data distributions
- Meta-learner training with different learner types (RandomForest, LinearRegression)

**Edge Cases:**
- Training with insufficient data (< 50 samples)
- All base models failing during prediction
- Meta-feature dimension mismatches during scaling
- NaN values in features, targets, and meta-features
- Empty or zero-length prediction arrays
- Timezone-aware/naive datetime mixing in predictions
- Model weight calculation with zero or negative scores
- Feature importance with untrained base models
- Incremental updates without prior training
- Validation data with inconsistent column structures

**Error Handling:**
- ModelTrainingError for insufficient data, base model failures, validation errors
- ModelPredictionError for untrained models, invalid features, all base model failures
- ValueError for invalid parameters, data format issues, dimension mismatches
- Exception propagation from base model training/prediction failures
- Graceful degradation when subset of base models fail
- Error recovery in incremental update scenarios

**Coverage Target:** 85%+

### src/models/training_config.py - Training Configuration
**Classes Found:** TrainingProfile (Enum), OptimizationLevel (Enum), ResourceLimits (dataclass), QualityThresholds (dataclass), OptimizationConfig (dataclass), TrainingEnvironmentConfig (dataclass), TrainingConfigManager (main class)
**Methods Analyzed:** TrainingProfile._missing_(), TrainingProfile.from_string(), ResourceLimits.validate(), QualityThresholds.validate(), TrainingConfigManager.__init__(), _initialize_default_profiles(), _load_config_file(), _dict_to_environment_config(), get_training_config(), _get_lookback_days_for_profile(), set_current_profile(), get_current_profile(), get_environment_config(), validate_configuration(), get_optimization_config(), update_profile_config(), save_config_to_file(), get_profile_comparison(), recommend_profile_for_use_case(), get_training_config_manager(), get_training_config()

**Required Tests:**
**Unit Tests:**
- Test TrainingProfile enum values (DEVELOPMENT, PRODUCTION, TESTING, RESEARCH, QUICK, COMPREHENSIVE)
- Test TrainingProfile._missing_() with frame inspection logic for different caller contexts
- Test TrainingProfile.from_string() success and custom error messages
- Test OptimizationLevel enum values (NONE, BASIC, STANDARD, INTENSIVE)
- Test ResourceLimits.validate() with positive, negative, zero, and None values for all fields
- Test ResourceLimits validation error messages for each field type
- Test QualityThresholds.validate() with boundary conditions (0.0, 1.0) for percentage fields
- Test QualityThresholds validation for positive number requirements
- Test OptimizationConfig default factory functions for search spaces
- Test OptimizationConfig with all parameter combinations
- Test TrainingEnvironmentConfig.validate() calling sub-validators
- Test TrainingEnvironmentConfig.validate() Path conversion and error handling
- Test TrainingConfigManager.__init__() with default and custom config paths
- Test TrainingConfigManager.__init__() config file existence checking
- Test _initialize_default_profiles() creating all six profile configurations
- Test _initialize_default_profiles() profile-specific parameters (resource limits, quality thresholds)
- Test _load_config_file() YAML parsing and profile creation
- Test _load_config_file() default profile setting from config
- Test _load_config_file() error handling for malformed YAML
- Test _dict_to_environment_config() nested dataclass conversion
- Test _dict_to_environment_config() enum conversion and Path handling
- Test get_training_config() with profile parameter and None (current profile)
- Test get_training_config() profile not found fallback to production
- Test get_training_config() TrainingConfig field mapping from environment config
- Test _get_lookback_days_for_profile() for all profile types
- Test _get_lookback_days_for_profile() default value for unknown profiles
- Test set_current_profile() validation and success cases
- Test set_current_profile() error handling for unavailable profiles
- Test get_current_profile() returning current profile
- Test get_environment_config() with profile parameter and current profile default
- Test get_environment_config() error handling for missing profiles
- Test validate_configuration() calling env_config.validate()
- Test validate_configuration() handling missing profiles
- Test get_optimization_config() delegation to environment config
- Test update_profile_config() with valid and invalid configuration keys
- Test update_profile_config() error handling for missing profiles
- Test save_config_to_file() YAML serialization with enum and Path conversion
- Test save_config_to_file() directory creation and file writing
- Test save_config_to_file() error handling and logging
- Test get_profile_comparison() metrics extraction for all profiles
- Test get_profile_comparison() comparison data structure creation
- Test recommend_profile_for_use_case() with various use case strings
- Test recommend_profile_for_use_case() case-insensitive matching and default
- Test get_training_config_manager() singleton pattern
- Test get_training_config() convenience function delegation

**Integration Tests:**
- Test TrainingConfigManager integration with SystemConfig via get_config()
- Test config file loading with real YAML files containing all profile types
- Test _dict_to_environment_config() with complex nested configuration structures
- Test get_training_config() creating valid TrainingConfig objects for all profiles
- Test profile switching integration (set_current_profile → get_training_config)
- Test save_config_to_file() → _load_config_file() round-trip preservation
- Test update_profile_config() → validate_configuration() integration
- Test singleton pattern integration with multiple get_training_config_manager() calls
- Test recommend_profile_for_use_case() → set_current_profile() workflow
- Test config validation integration across all nested dataclasses

**Edge Cases:**
- Test TrainingProfile._missing_() frame inspection with complex call stacks
- Test TrainingProfile._missing_() with AttributeError during frame traversal
- Test ResourceLimits.validate() with extremely large positive values
- Test QualityThresholds.validate() with floating-point precision edge cases
- Test TrainingConfigManager.__init__() with non-existent config path
- Test _load_config_file() with empty YAML file
- Test _load_config_file() with YAML containing only partial profile data
- Test _dict_to_environment_config() with missing required nested fields
- Test _dict_to_environment_config() with invalid enum string values
- Test get_training_config() with profile conversion edge cases
- Test save_config_to_file() with unserializable configuration objects
- Test save_config_to_file() with filesystem permission errors
- Test update_profile_config() with configuration objects requiring deep copying
- Test get_profile_comparison() with profiles having None values
- Test recommend_profile_for_use_case() with empty strings and special characters

**Error Handling:**
- Test TrainingProfile._missing_() ValueError raising with context-specific messages
- Test TrainingProfile.from_string() ValueError handling with custom message
- Test ResourceLimits.validate() comprehensive issue collection
- Test QualityThresholds.validate() all validation rule combinations
- Test TrainingEnvironmentConfig.validate() error aggregation from sub-validators
- Test TrainingConfigManager._load_config_file() exception handling with logging
- Test _dict_to_environment_config() exception handling during dataclass creation
- Test set_current_profile() ValueError raising for unavailable profiles
- Test get_environment_config() ValueError raising for missing profiles
- Test update_profile_config() ValueError raising for missing profiles
- Test update_profile_config() warning logging for unknown configuration keys
- Test save_config_to_file() comprehensive exception handling and re-raising
- Test validation errors propagation through get_training_config()
- Test YAML parsing errors in _load_config_file() with proper logging
- Test filesystem errors during config file operations

**Coverage Target:** 85%+

### src/models/training_pipeline.py - Model Training Pipeline
**Classes Found:** TrainingStage, TrainingType, ValidationStrategy, TrainingConfig, TrainingProgress, DataQualityReport, ModelTrainingPipeline
**Methods Analyzed:** __init__, run_initial_training, run_incremental_training, run_retraining_pipeline, train_room_models, _prepare_training_data, _query_room_events, _validate_data_quality, _can_proceed_with_quality_issues, _extract_features_and_targets, _split_training_data, _time_series_split, _expanding_window_split, _rolling_window_split, _holdout_split, _train_models, _validate_models, _evaluate_and_select_best_model, _meets_quality_thresholds, _deploy_trained_models, _generate_model_version, _save_model_artifacts, _cleanup_training_artifacts, _register_trained_models, _notify_tracking_manager_of_completion, _update_training_stats, get_active_pipelines, get_pipeline_history, get_training_statistics, get_model_registry, get_model_versions, get_model_performance, load_model_from_artifacts, compare_models, _is_valid_model_type, _create_model_instance, _invoke_callback_if_configured

**Required Tests:**
**Unit Tests:**
- Test TrainingConfig dataclass initialization with all parameters and defaults
- Test TrainingProgress.update_stage method with stage transitions and progress calculations
- Test DataQualityReport creation and add_recommendation method
- Test ModelTrainingPipeline.__init__ with different configuration options
- Test pipeline artifacts path creation and directory structure setup
- Test run_initial_training with various room_ids configurations (None, specific list)
- Test run_initial_training parallel processing with semaphore limits
- Test run_incremental_training with different model types and new data periods
- Test run_retraining_pipeline with different trigger reasons and strategies
- Test train_room_models complete pipeline execution with all stages
- Test _prepare_training_data with different lookback periods and database scenarios
- Test _query_room_events mock data generation and error handling
- Test _validate_data_quality with comprehensive data validation scenarios
- Test _can_proceed_with_quality_issues decision logic with various quality issues
- Test _extract_features_and_targets mock feature generation and integration
- Test all data splitting strategies: time_series_split, expanding_window_split, rolling_window_split, holdout_split
- Test _train_models with ensemble and specific model type training
- Test _validate_models with prediction generation and scoring metrics
- Test _evaluate_and_select_best_model with different selection metrics (mae, rmse, r2)
- Test _meets_quality_thresholds with various accuracy and error combinations
- Test _deploy_trained_models with model registry registration and versioning
- Test _generate_model_version unique identifier generation
- Test _save_model_artifacts with pickle serialization and metadata storage
- Test _cleanup_training_artifacts cleanup operations
- Test model registry and version tracking operations
- Test training statistics tracking and updates
- Test callback invocation mechanisms for progress, optimization, and validation
- Test _is_valid_model_type with ModelType enum validation
- Test _create_model_instance dynamic model creation for all supported types

**Integration Tests:**
- Test complete training pipeline execution from initialization to deployment
- Test integration with FeatureEngineeringEngine and FeatureStore
- Test integration with database_manager for data retrieval
- Test integration with TrackingManager for model registration and monitoring
- Test ensemble model integration with base predictors
- Test model artifact persistence and loading across pipeline runs
- Test parallel training execution with resource limits
- Test cross-validation strategy execution with temporal data
- Test model comparison and A/B testing functionality
- Test pipeline failure recovery and error propagation
- Test training progress callbacks and monitoring integration
- Test model deployment to production registry
- Test incremental training with existing models
- Test retraining pipeline triggered by accuracy degradation

**Edge Cases:**
- Test with insufficient training data (below min_samples_per_room threshold)
- Test with empty or malformed raw data from database
- Test with extreme data quality issues (100% missing values, no timestamps)
- Test with data freshness validation failures
- Test with temporal consistency validation failures (non-monotonic timestamps)
- Test with feature extraction failures returning empty DataFrames
- Test with data splitting edge cases (very small datasets, single samples)
- Test with model training failures for all base models
- Test with validation failures due to prediction errors
- Test with model evaluation failures and score calculation errors
- Test with model deployment failures during artifact saving
- Test with pickle serialization failures for complex models
- Test with metadata serialization failures for JSON export
- Test with model registry conflicts and version collisions
- Test with cleanup failures during artifact management
- Test with tracking manager integration failures
- Test with callback function execution failures
- Test with extreme training configurations (zero splits, huge lookback)
- Test with concurrent pipeline execution conflicts
- Test with model loading failures from corrupted artifacts
- Test with model comparison on non-existent or failed models

**Error Handling:**
- Test ModelTrainingError raising with proper model_type, room_id, and cause
- Test InsufficientTrainingDataError with data_points and minimum_required
- Test OccupancyPredictionError during data splitting with context details
- Test exception handling in run_initial_training with training task failures
- Test exception handling in run_incremental_training with proper error propagation
- Test exception handling in run_retraining_pipeline with trigger context
- Test exception handling in train_room_models with stage-specific error tracking
- Test exception handling in _prepare_training_data with database failures
- Test exception handling in _validate_data_quality with calculation errors
- Test exception handling in _extract_features_and_targets with engine failures
- Test exception handling in data splitting methods with validation errors
- Test exception handling in _train_models with individual model failures
- Test exception handling in _validate_models with prediction failures
- Test exception handling in _evaluate_and_select_best_model with empty results
- Test exception handling in _deploy_trained_models with artifact saving failures
- Test exception handling in _save_model_artifacts with file system errors
- Test exception handling in model loading with corrupted or missing files
- Test exception handling in training statistics updates with invalid data
- Test exception handling in callback invocation with malformed functions
- Test exception handling in model registry operations with concurrent access
- Test exception handling in pipeline cleanup with resource conflicts
- Test progress tracking during failures with proper stage updates
- Test error message formatting and logging for all failure scenarios
- Test graceful degradation with partial model training failures
- Test timeout handling for long-running training operations
- Test resource limit enforcement during parallel training

**Coverage Target:** 85%+

## Summary

This comprehensive ML models testing requirements document covers all 9+ ML models components with detailed testing specifications including:

- **Base Predictor Interface**: Abstract base class functionality, dataclass serialization, model persistence, and feature validation
- **XGBoost Predictor**: Gradient boosting implementation with feature importance and confidence calculation  
- **LSTM Predictor**: Neural network implementation using MLPRegressor with sequence processing and time series prediction
- **Gaussian Process Predictor**: Uncertainty quantification with kernel selection, sparse GP, and confidence intervals
- **Hidden Markov Model Predictor**: State-based prediction with transition modeling and duration estimation
- **Ensemble Model**: Meta-learning architecture combining multiple base models with cross-validation and stacking
- **Training Configuration**: Profile management, optimization settings, and environment-specific configurations
- **Training Pipeline**: Complete model training orchestration with data preparation, validation, and deployment

Each component includes comprehensive unit tests, integration tests, edge cases, error handling scenarios, and specific coverage targets of 85%+ to ensure robust ML model functionality.

**Key Testing Focus Areas:**
- Mathematical accuracy of prediction algorithms and confidence calculations
- Model serialization/deserialization with complete state preservation
- Feature validation and preprocessing pipeline correctness
- Error handling for training failures and prediction edge cases
- Integration between different model types and ensemble coordination
- Performance optimization and memory management for large datasets
- Cross-validation and temporal data splitting strategies
- Model versioning and artifact management systems

**Mock Requirements:**
- Mock scikit-learn models (MLPRegressor, XGBRegressor, GaussianProcessRegressor, GaussianMixture)
- Mock StandardScaler and MinMaxScaler for deterministic preprocessing
- Mock pandas DataFrame and numpy array operations
- Mock pickle operations for persistence testing
- Mock datetime operations for consistent time handling
- Mock logging and configuration systems

**Test Fixtures Needed:**
- Realistic sensor feature DataFrames with temporal patterns
- Target DataFrames with occupancy transition timing
- Mock model states for persistence testing
- Training/validation datasets with known patterns
- Performance benchmarking datasets for optimization testing
- Cross-validation datasets for model evaluation
- Error scenario data for robustness testing