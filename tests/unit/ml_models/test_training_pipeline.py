"""Unit tests for machine learning training pipeline.

Covers:
- src/models/training_pipeline.py (Model Training Pipeline)
- src/models/training_integration.py (Training Integration Logic) 
- src/models/training_config.py (Training Configuration)

This test file implements comprehensive testing for all ML training functionality.
"""

import asyncio
import json
import pickle
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, AsyncMock
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Tuple

from src.models.training_pipeline import (
    TrainingStage,
    TrainingType,
    ValidationStrategy,
    TrainingConfig,
    TrainingProgress,
    DataQualityReport,
    ModelTrainingPipeline,
)
from src.models.training_integration import TrainingIntegrationManager
from src.models.training_config import (
    TrainingProfile,
    OptimizationLevel,
    ResourceLimits,
    QualityThresholds,
    OptimizationConfig,
    TrainingEnvironmentConfig,
    TrainingConfigManager,
)
from src.core.constants import ModelType
from src.core.exceptions import (
    ModelTrainingError,
    InsufficientTrainingDataError,
    OccupancyPredictionError,
)
from src.models.base.predictor import BasePredictor, TrainingResult, PredictionResult


@pytest.fixture
def mock_feature_engine():
    """Mock feature engineering engine."""
    engine = Mock()
    engine.extract_features = AsyncMock()
    return engine


@pytest.fixture
def mock_feature_store():
    """Mock feature store."""
    store = Mock()
    store.compute_features = AsyncMock()
    store.get_training_data = AsyncMock()
    return store


@pytest.fixture
def mock_database_manager():
    """Mock database manager."""
    db_manager = Mock()
    db_manager.query_room_events = AsyncMock()
    return db_manager


@pytest.fixture
def mock_tracking_manager():
    """Mock tracking manager."""
    tracking = Mock()
    tracking.register_model = Mock()
    tracking.add_accuracy_callback = Mock()
    tracking.add_drift_callback = Mock()
    tracking.on_model_retrained = AsyncMock()
    return tracking


@pytest.fixture
def sample_training_config():
    """Sample training configuration."""
    return TrainingConfig(
        lookback_days=30,
        validation_split=0.2,
        test_split=0.1,
        min_samples_per_room=50,
        ensemble_enabled=True,
        base_models_enabled=["lstm", "xgboost", "hmm"],
        model_selection_metric="mae",
        max_parallel_models=2,
    )


@pytest.fixture
def sample_features_targets():
    """Sample features and targets DataFrames."""
    n_samples = 100
    features_df = pd.DataFrame({
        'temporal_hour': np.random.randint(0, 24, n_samples),
        'temporal_day_of_week': np.random.randint(0, 7, n_samples),
        'sequential_last_motion': np.random.random(n_samples),
        'contextual_temp': np.random.normal(22, 5, n_samples),
    })
    
    targets_df = pd.DataFrame({
        'time_until_transition_seconds': np.random.exponential(3600, n_samples),
        'transition_type': np.random.choice(['occupied_to_vacant', 'vacant_to_occupied'], n_samples),
    })
    
    return features_df, targets_df


@pytest.fixture
def mock_base_predictor():
    """Mock base predictor."""
    predictor = Mock(spec=BasePredictor)
    predictor.model_type = ModelType.ENSEMBLE
    predictor.room_id = "living_room"
    predictor.is_trained = True
    predictor.feature_names = ["temporal_hour", "sequential_last_motion"]
    predictor.model_version = "v1.0"
    predictor.training_date = datetime.now(timezone.utc)
    
    # Mock async methods
    predictor.train = AsyncMock(return_value=TrainingResult(
        success=True,
        training_score=0.85,
        validation_score=0.78,
        training_samples=100,
        training_time_seconds=120,
    ))
    
    predictor.predict = AsyncMock(return_value=[
        PredictionResult(
            room_id="living_room",
            predicted_time=datetime.now(timezone.utc) + timedelta(minutes=30),
            transition_type="occupied_to_vacant",
            confidence_score=0.85,
        )
    ])
    
    return predictor


class TestTrainingConfig:
    """Test training configuration dataclass."""
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        
        assert config.lookback_days == 180
        assert config.validation_split == 0.2
        assert config.test_split == 0.1
        assert config.min_samples_per_room == 100
        assert config.ensemble_enabled is True
        assert "lstm" in config.base_models_enabled
        assert config.model_selection_metric == "mae"
        assert config.max_parallel_models == 2
        
    def test_training_config_custom_values(self):
        """Test TrainingConfig with custom values."""
        config = TrainingConfig(
            lookback_days=90,
            validation_split=0.3,
            base_models_enabled=["xgboost"],
            model_selection_metric="r2",
        )
        
        assert config.lookback_days == 90
        assert config.validation_split == 0.3
        assert config.base_models_enabled == ["xgboost"]
        assert config.model_selection_metric == "r2"
        
    def test_validation_strategy_enum(self):
        """Test ValidationStrategy enum values."""
        config = TrainingConfig(
            validation_strategy=ValidationStrategy.TIME_SERIES_SPLIT
        )
        assert config.validation_strategy == ValidationStrategy.TIME_SERIES_SPLIT
        
        # Test all enum values
        for strategy in ValidationStrategy:
            config = TrainingConfig(validation_strategy=strategy)
            assert config.validation_strategy == strategy


class TestTrainingProgress:
    """Test training progress tracking."""
    
    def test_training_progress_initialization(self):
        """Test TrainingProgress initialization."""
        progress = TrainingProgress(
            pipeline_id="test-123",
            training_type=TrainingType.INITIAL,
            room_id="bedroom",
        )
        
        assert progress.pipeline_id == "test-123"
        assert progress.training_type == TrainingType.INITIAL
        assert progress.room_id == "bedroom"
        assert progress.stage == TrainingStage.INITIALIZATION
        assert progress.progress_percent == 0.0
        assert isinstance(progress.start_time, datetime)
        
    def test_update_stage(self):
        """Test stage update with progress calculation."""
        progress = TrainingProgress(
            pipeline_id="test-123",
            training_type=TrainingType.INITIAL,
        )
        
        # Test stage progression
        progress.update_stage(TrainingStage.DATA_PREPARATION)
        assert progress.stage == TrainingStage.DATA_PREPARATION
        assert progress.progress_percent == 10.0
        
        progress.update_stage(TrainingStage.MODEL_TRAINING, {"models": ["lstm", "xgboost"]})
        assert progress.stage == TrainingStage.MODEL_TRAINING
        assert progress.progress_percent == 70.0
        assert progress.stage_details == {"models": ["lstm", "xgboost"]}
        
        progress.update_stage(TrainingStage.COMPLETED)
        assert progress.progress_percent == 100.0
        
    def test_stage_progress_mapping(self):
        """Test all stage progress percentages."""
        progress = TrainingProgress(
            pipeline_id="test-123",
            training_type=TrainingType.INITIAL,
        )
        
        expected_progress = {
            TrainingStage.INITIALIZATION: 5.0,
            TrainingStage.DATA_PREPARATION: 10.0,
            TrainingStage.DATA_VALIDATION: 15.0,
            TrainingStage.FEATURE_EXTRACTION: 25.0,
            TrainingStage.DATA_SPLITTING: 30.0,
            TrainingStage.MODEL_TRAINING: 70.0,
            TrainingStage.MODEL_VALIDATION: 85.0,
            TrainingStage.MODEL_EVALUATION: 90.0,
            TrainingStage.MODEL_DEPLOYMENT: 95.0,
            TrainingStage.CLEANUP: 98.0,
            TrainingStage.COMPLETED: 100.0,
        }
        
        for stage, expected_percent in expected_progress.items():
            progress.update_stage(stage)
            assert progress.progress_percent == expected_percent


class TestDataQualityReport:
    """Test data quality reporting."""
    
    def test_data_quality_report_initialization(self):
        """Test DataQualityReport creation."""
        report = DataQualityReport(
            passed=True,
            total_samples=1000,
            valid_samples=950,
            sufficient_samples=True,
            data_freshness_ok=True,
            feature_completeness_ok=True,
            temporal_consistency_ok=True,
            missing_values_percent=5.0,
            duplicates_count=10,
            outliers_count=5,
            data_gaps=[],
        )
        
        assert report.passed is True
        assert report.total_samples == 1000
        assert report.valid_samples == 950
        assert report.missing_values_percent == 5.0
        assert len(report.recommendations) == 0
        
    def test_add_recommendation(self):
        """Test adding recommendations to quality report."""
        report = DataQualityReport(
            passed=False,
            total_samples=50,
            valid_samples=40,
            sufficient_samples=False,
            data_freshness_ok=True,
            feature_completeness_ok=True,
            temporal_consistency_ok=True,
            missing_values_percent=20.0,
            duplicates_count=5,
            outliers_count=2,
            data_gaps=[],
        )
        
        report.add_recommendation("Insufficient samples for training")
        report.add_recommendation("High missing value percentage detected")
        
        assert len(report.recommendations) == 2
        assert "Insufficient samples" in report.recommendations[0]
        assert "High missing value" in report.recommendations[1]


class TestModelTrainingPipeline:
    """Test the main model training pipeline."""
    
    def test_pipeline_initialization(self, sample_training_config, mock_feature_engine,
                                   mock_feature_store, mock_database_manager):
        """Test pipeline initialization."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        assert pipeline.config == sample_training_config
        assert pipeline.feature_engineering_engine == mock_feature_engine
        assert pipeline.feature_store == mock_feature_store
        assert pipeline.database_manager == mock_database_manager
        assert pipeline.artifacts_path.exists()
        assert len(pipeline._active_pipelines) == 0
        assert len(pipeline._model_registry) == 0
        
    def test_artifacts_path_creation(self, sample_training_config, mock_feature_engine,
                                    mock_feature_store, mock_database_manager, tmp_path):
        """Test artifacts path creation."""
        artifacts_path = tmp_path / "test_artifacts"
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            model_artifacts_path=artifacts_path,
        )
        
        assert pipeline.artifacts_path == artifacts_path
        assert artifacts_path.exists()
        
    @pytest.mark.asyncio
    async def test_initial_training_success(self, sample_training_config, mock_feature_engine,
                                          mock_feature_store, mock_database_manager,
                                          mock_tracking_manager, mock_base_predictor):
        """Test successful initial training pipeline."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            tracking_manager=mock_tracking_manager,
        )
        
        # Mock the complete training pipeline
        with patch.object(pipeline, 'train_room_models') as mock_train:
            mock_progress = TrainingProgress(
                pipeline_id="test-123",
                training_type=TrainingType.INITIAL,
                room_id="living_room",
            )
            mock_progress.stage = TrainingStage.COMPLETED
            mock_progress.best_model = "ensemble"
            mock_train.return_value = mock_progress
            
            # Mock get_config to return room configuration
            with patch('src.models.training_pipeline.get_config') as mock_get_config:
                mock_config = Mock()
                mock_config.rooms = {"living_room": Mock(), "bedroom": Mock()}
                mock_get_config.return_value = mock_config
                
                results = await pipeline.run_initial_training()
                
                assert len(results) == 2  # Two rooms
                assert "living_room" in results
                assert "bedroom" in results
                mock_train.assert_called()
                
    @pytest.mark.asyncio
    async def test_initial_training_with_specific_rooms(self, sample_training_config,
                                                       mock_feature_engine, mock_feature_store,
                                                       mock_database_manager, mock_tracking_manager):
        """Test initial training with specific room list."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            tracking_manager=mock_tracking_manager,
        )
        
        with patch.object(pipeline, 'train_room_models') as mock_train:
            mock_progress = TrainingProgress(
                pipeline_id="test-123",
                training_type=TrainingType.INITIAL,
                room_id="kitchen",
            )
            mock_progress.stage = TrainingStage.COMPLETED
            mock_train.return_value = mock_progress
            
            results = await pipeline.run_initial_training(room_ids=["kitchen"])
            
            assert len(results) == 1
            assert "kitchen" in results
            
    @pytest.mark.asyncio
    async def test_incremental_training(self, sample_training_config, mock_feature_engine,
                                       mock_feature_store, mock_database_manager):
        """Test incremental training pipeline."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        with patch.object(pipeline, 'train_room_models') as mock_train:
            mock_progress = TrainingProgress(
                pipeline_id="test-123",
                training_type=TrainingType.INCREMENTAL,
                room_id="office",
            )
            mock_train.return_value = mock_progress
            
            result = await pipeline.run_incremental_training(
                room_id="office",
                model_type="xgboost",
                days_of_new_data=14,
            )
            
            assert result == mock_progress
            mock_train.assert_called_once_with(
                room_id="office",
                training_type=TrainingType.INCREMENTAL,
                lookback_days=14,
                target_model_type="xgboost",
            )
            
    @pytest.mark.asyncio
    async def test_retraining_pipeline(self, sample_training_config, mock_feature_engine,
                                      mock_feature_store, mock_database_manager):
        """Test retraining pipeline with different strategies."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        with patch.object(pipeline, 'train_room_models') as mock_train:
            mock_progress = TrainingProgress(
                pipeline_id="test-123",
                training_type=TrainingType.ADAPTATION,
            )
            mock_train.return_value = mock_progress
            
            # Test adaptive retraining
            result = await pipeline.run_retraining_pipeline(
                room_id="living_room",
                trigger_reason="accuracy_degradation",
                strategy="adaptive",
            )
            
            assert result == mock_progress
            mock_train.assert_called_once_with(
                room_id="living_room",
                training_type=TrainingType.ADAPTATION,
                lookback_days=sample_training_config.lookback_days,
                metadata={
                    "trigger_reason": "accuracy_degradation",
                    "strategy": "adaptive",
                },
            )
            
    @pytest.mark.asyncio
    async def test_retraining_with_full_retrain(self, sample_training_config, mock_feature_engine,
                                               mock_feature_store, mock_database_manager):
        """Test retraining with forced full retrain."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        with patch.object(pipeline, 'train_room_models') as mock_train:
            mock_progress = TrainingProgress(
                pipeline_id="test-123",
                training_type=TrainingType.FULL_RETRAIN,
            )
            mock_train.return_value = mock_progress
            
            result = await pipeline.run_retraining_pipeline(
                room_id="bedroom",
                trigger_reason="concept_drift",
                strategy="full_retrain",
                force_full_retrain=True,
            )
            
            assert result == mock_progress
            mock_train.assert_called_once_with(
                room_id="bedroom",
                training_type=TrainingType.FULL_RETRAIN,
                lookback_days=sample_training_config.lookback_days,
                metadata={
                    "trigger_reason": "concept_drift",
                    "strategy": "full_retrain",
                },
            )
            
    @pytest.mark.asyncio
    async def test_train_room_models_success(self, sample_training_config, mock_feature_engine,
                                           mock_feature_store, mock_database_manager,
                                           sample_features_targets, mock_base_predictor):
        """Test successful room model training."""
        features_df, targets_df = sample_features_targets
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Mock all pipeline methods
        with patch.object(pipeline, '_prepare_training_data') as mock_prepare, \
             patch.object(pipeline, '_validate_data_quality') as mock_validate, \
             patch.object(pipeline, '_extract_features_and_targets') as mock_extract, \
             patch.object(pipeline, '_split_training_data') as mock_split, \
             patch.object(pipeline, '_train_models') as mock_train, \
             patch.object(pipeline, '_validate_models') as mock_validate_models, \
             patch.object(pipeline, '_evaluate_and_select_best_model') as mock_evaluate, \
             patch.object(pipeline, '_deploy_trained_models') as mock_deploy, \
             patch.object(pipeline, '_cleanup_training_artifacts') as mock_cleanup:
            
            # Setup mocks
            mock_prepare.return_value = pd.DataFrame({'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min')})
            
            mock_quality_report = DataQualityReport(
                passed=True, total_samples=100, valid_samples=95,
                sufficient_samples=True, data_freshness_ok=True,
                feature_completeness_ok=True, temporal_consistency_ok=True,
                missing_values_percent=5.0, duplicates_count=0,
                outliers_count=0, data_gaps=[]
            )
            mock_validate.return_value = mock_quality_report
            
            mock_extract.return_value = (features_df, targets_df)
            mock_split.return_value = (
                (features_df[:70], targets_df[:70]),  # train
                (features_df[70:90], targets_df[70:90]),  # val
                (features_df[90:], targets_df[90:])  # test
            )
            
            trained_models = {"ensemble": mock_base_predictor}
            mock_train.return_value = trained_models
            
            validation_scores = {"ensemble": 0.15}  # MAE score
            mock_validate_models.return_value = validation_scores
            
            evaluation_metrics = {
                "best_model": "ensemble",
                "mae": 0.15,
                "rmse": 0.18,
                "r2": 0.85
            }
            mock_evaluate.return_value = ("ensemble", evaluation_metrics)
            
            deployment_info = {
                "deployed_models": [{"model_name": "ensemble", "is_best": True}],
                "best_model": "ensemble"
            }
            mock_deploy.return_value = deployment_info
            
            # Execute training
            result = await pipeline.train_room_models(
                room_id="living_room",
                training_type=TrainingType.INITIAL,
                lookback_days=30,
            )
            
            # Assertions
            assert result.stage == TrainingStage.COMPLETED
            assert result.room_id == "living_room"
            assert result.best_model == "ensemble"
            assert result.best_score == 0.15
            assert len(result.models_trained) == 1
            assert "ensemble" in result.models_trained
            
            # Verify all methods were called
            mock_prepare.assert_called_once()
            mock_validate.assert_called_once()
            mock_extract.assert_called_once()
            mock_split.assert_called_once()
            mock_train.assert_called_once()
            mock_validate_models.assert_called_once()
            mock_evaluate.assert_called_once()
            mock_deploy.assert_called_once()
            mock_cleanup.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_train_room_models_insufficient_data(self, sample_training_config,
                                                      mock_feature_engine, mock_feature_store,
                                                      mock_database_manager):
        """Test training failure due to insufficient data."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        with patch.object(pipeline, '_prepare_training_data') as mock_prepare:
            # Return insufficient data
            small_df = pd.DataFrame({'value': [1, 2, 3]})  # Only 3 samples, need 50
            mock_prepare.return_value = small_df
            
            with pytest.raises(ModelTrainingError) as exc_info:
                await pipeline.train_room_models(
                    room_id="living_room",
                    training_type=TrainingType.INITIAL,
                    lookback_days=30,
                )
                
            assert "ensemble" in str(exc_info.value)
            assert "living_room" in str(exc_info.value)
            
    @pytest.mark.asyncio
    async def test_prepare_training_data(self, sample_training_config, mock_feature_engine,
                                        mock_feature_store, mock_database_manager):
        """Test training data preparation."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        with patch.object(pipeline, '_query_room_events') as mock_query:
            mock_data = pd.DataFrame({
                'timestamp': pd.date_range('2024-01-01', periods=100, freq='5min'),
                'room_id': ['living_room'] * 100,
                'sensor_type': ['motion'] * 100,
                'state': ['on'] * 50 + ['off'] * 50,
            })
            mock_query.return_value = mock_data
            
            result = await pipeline._prepare_training_data("living_room", 30)
            
            assert result is not None
            assert len(result) == 100
            mock_query.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_prepare_training_data_no_database(self, sample_training_config,
                                                    mock_feature_engine, mock_feature_store):
        """Test training data preparation without database manager."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=None,  # No database manager
        )
        
        result = await pipeline._prepare_training_data("living_room", 30)
        
        assert result is not None
        assert len(result) == 0  # Empty DataFrame when no DB manager
        
    @pytest.mark.asyncio
    async def test_validate_data_quality_good_data(self, sample_training_config,
                                                  mock_feature_engine, mock_feature_store,
                                                  mock_database_manager):
        """Test data quality validation with good data."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Good quality data
        good_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=200, freq='5min'),
            'room_id': ['living_room'] * 200,
            'sensor_type': ['motion'] * 200,
            'state': np.random.choice(['on', 'off'], 200),
        })
        
        # Make sure timestamps are recent for freshness check
        good_data['timestamp'] = pd.date_range(datetime.now() - timedelta(hours=2),
                                               periods=200, freq='5min')
        
        report = await pipeline._validate_data_quality(good_data, "living_room")
        
        assert report.passed is True
        assert report.total_samples == 200
        assert report.sufficient_samples is True
        assert report.feature_completeness_ok is True
        assert report.temporal_consistency_ok is True
        assert report.missing_values_percent < 20.0
        
    @pytest.mark.asyncio
    async def test_validate_data_quality_poor_data(self, sample_training_config,
                                                  mock_feature_engine, mock_feature_store,
                                                  mock_database_manager):
        """Test data quality validation with poor data."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Poor quality data - insufficient samples and missing columns
        poor_data = pd.DataFrame({
            'timestamp': pd.date_range('2020-01-01', periods=10, freq='5min'),  # Old data
            'room_id': ['living_room'] * 10,
            # Missing 'sensor_type' and 'state' columns
        })
        
        report = await pipeline._validate_data_quality(poor_data, "living_room")
        
        assert report.passed is False
        assert report.total_samples == 10
        assert report.sufficient_samples is False
        assert report.feature_completeness_ok is False
        assert len(report.recommendations) > 0
        
    def test_can_proceed_with_quality_issues(self, sample_training_config,
                                            mock_feature_engine, mock_feature_store,
                                            mock_database_manager):
        """Test decision logic for proceeding with quality issues."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Good enough quality - can proceed
        acceptable_report = DataQualityReport(
            passed=False, total_samples=100, valid_samples=90,
            sufficient_samples=True, data_freshness_ok=False,
            feature_completeness_ok=True, temporal_consistency_ok=False,
            missing_values_percent=30.0, duplicates_count=5,
            outliers_count=2, data_gaps=[]
        )
        
        can_proceed = pipeline._can_proceed_with_quality_issues(acceptable_report)
        assert can_proceed is True  # Has sufficient samples and completeness
        
        # Poor quality - cannot proceed
        poor_report = DataQualityReport(
            passed=False, total_samples=20, valid_samples=10,
            sufficient_samples=False, data_freshness_ok=False,
            feature_completeness_ok=False, temporal_consistency_ok=False,
            missing_values_percent=80.0, duplicates_count=10,
            outliers_count=5, data_gaps=[]
        )
        
        can_proceed = pipeline._can_proceed_with_quality_issues(poor_report)
        assert can_proceed is False  # Insufficient samples and missing features
        
    @pytest.mark.asyncio
    async def test_extract_features_and_targets(self, sample_training_config,
                                               mock_feature_engine, mock_feature_store,
                                               mock_database_manager):
        """Test feature and target extraction."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        raw_data = pd.DataFrame({
            'timestamp': pd.date_range('2024-01-01', periods=50, freq='5min'),
            'room_id': ['living_room'] * 50,
            'sensor_type': ['motion'] * 50,
            'state': np.random.choice(['on', 'off'], 50),
        })
        
        features_df, targets_df = await pipeline._extract_features_and_targets(
            raw_data, "living_room"
        )
        
        assert not features_df.empty
        assert not targets_df.empty
        assert len(features_df) == len(raw_data)
        assert len(targets_df) == len(raw_data)
        
        # Check expected feature columns
        expected_feature_cols = [
            'temporal_hour', 'temporal_day_of_week',
            'sequential_last_motion', 'contextual_temp'
        ]
        for col in expected_feature_cols:
            assert col in features_df.columns
            
        # Check expected target columns
        expected_target_cols = ['time_until_transition_seconds', 'transition_type']
        for col in expected_target_cols:
            assert col in targets_df.columns
            
    @pytest.mark.asyncio
    async def test_extract_features_and_targets_empty_data(self, sample_training_config,
                                                          mock_feature_engine, mock_feature_store,
                                                          mock_database_manager):
        """Test feature extraction with empty data."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        empty_data = pd.DataFrame()
        
        features_df, targets_df = await pipeline._extract_features_and_targets(
            empty_data, "living_room"
        )
        
        assert features_df.empty
        assert targets_df.empty


class TestDataSplittingStrategies:
    """Test different data splitting strategies."""
    
    @pytest.mark.asyncio
    async def test_time_series_split(self, sample_training_config, mock_feature_engine,
                                    mock_feature_store, mock_database_manager,
                                    sample_features_targets):
        """Test time series data splitting."""
        features_df, targets_df = sample_features_targets
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        (train_features, train_targets), (val_features, val_targets), (test_features, test_targets) = \
            await pipeline._time_series_split(features_df, targets_df, progress)
        
        # Check split sizes approximately match configuration
        total_samples = len(features_df)
        expected_test_size = int(total_samples * sample_training_config.test_split)
        expected_val_size = int(total_samples * sample_training_config.validation_split)
        expected_train_size = total_samples - expected_test_size - expected_val_size
        
        assert len(train_features) == expected_train_size
        assert len(val_features) == expected_val_size
        assert len(test_features) == expected_test_size
        
        # Check no data leakage - training data comes first chronologically
        assert all(train_features.index < val_features.index.min())
        assert all(val_features.index < test_features.index.min())
        
        # Check progress was updated
        assert progress.training_samples == len(train_features)
        assert progress.validation_samples == len(val_features)
        assert progress.test_samples == len(test_features)
        
    @pytest.mark.asyncio
    async def test_expanding_window_split(self, sample_training_config, mock_feature_engine,
                                         mock_feature_store, mock_database_manager,
                                         sample_features_targets):
        """Test expanding window data splitting."""
        features_df, targets_df = sample_features_targets
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        (train_features, train_targets), (val_features, val_targets), (test_features, test_targets) = \
            await pipeline._expanding_window_split(features_df, targets_df, progress)
        
        # Check that all splits have data
        assert len(train_features) > 0
        assert len(val_features) > 0
        assert len(test_features) > 0
        
        # Check total samples preserved
        total_original = len(features_df)
        total_split = len(train_features) + len(val_features) + len(test_features)
        assert total_split == total_original
        
    @pytest.mark.asyncio
    async def test_rolling_window_split(self, sample_training_config, mock_feature_engine,
                                       mock_feature_store, mock_database_manager,
                                       sample_features_targets):
        """Test rolling window data splitting."""
        features_df, targets_df = sample_features_targets
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        (train_features, train_targets), (val_features, val_targets), (test_features, test_targets) = \
            await pipeline._rolling_window_split(features_df, targets_df, progress)
        
        # Check that all splits have data
        assert len(train_features) > 0
        assert len(val_features) > 0
        assert len(test_features) > 0
        
        # Check that the splits are positioned at the end (most recent data for validation/testing)
        assert test_features.index[-1] == features_df.index[-1]  # Test set uses most recent data
        
    @pytest.mark.asyncio
    async def test_holdout_split(self, sample_training_config, mock_feature_engine,
                                mock_feature_store, mock_database_manager,
                                sample_features_targets):
        """Test holdout data splitting."""
        features_df, targets_df = sample_features_targets
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        (train_features, train_targets), (val_features, val_targets), (test_features, test_targets) = \
            await pipeline._holdout_split(features_df, targets_df, progress)
        
        # Check sequential splits (no shuffling for temporal data)
        total_samples = len(features_df)
        test_size = max(1, int(total_samples * sample_training_config.test_split))
        val_size = max(1, int(total_samples * sample_training_config.validation_split))
        train_size = total_samples - test_size - val_size
        
        assert len(train_features) == train_size
        assert len(val_features) == val_size
        assert len(test_features) == test_size
        
    @pytest.mark.asyncio
    async def test_split_insufficient_samples(self, sample_training_config, mock_feature_engine,
                                             mock_feature_store, mock_database_manager):
        """Test data splitting with insufficient samples."""
        # Create very small dataset
        features_df = pd.DataFrame({'feature1': [1, 2, 3]})  # Only 3 samples
        targets_df = pd.DataFrame({'target': [10, 20, 30]})
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        with pytest.raises(OccupancyPredictionError) as exc_info:
            await pipeline._split_training_data(features_df, targets_df, progress)
            
        assert "Insufficient samples" in str(exc_info.value)
        assert "INSUFFICIENT_TRAINING_SAMPLES" in str(exc_info.value)


class TestModelTraining:
    """Test model training functionality."""
    
    @pytest.mark.asyncio
    async def test_train_models_ensemble(self, sample_training_config, mock_feature_engine,
                                        mock_feature_store, mock_database_manager,
                                        sample_features_targets, mock_base_predictor,
                                        mock_tracking_manager):
        """Test training ensemble models."""
        features_df, targets_df = sample_features_targets
        train_data = (features_df[:70], targets_df[:70])
        val_data = (features_df[70:90], targets_df[70:90])
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            tracking_manager=mock_tracking_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        # Mock OccupancyEnsemble
        with patch('src.models.training_pipeline.OccupancyEnsemble') as mock_ensemble_class:
            mock_ensemble = mock_base_predictor
            mock_ensemble_class.return_value = mock_ensemble
            
            trained_models = await pipeline._train_models(
                room_id="living_room",
                train_data=train_data,
                val_data=val_data,
                target_model_type=None,  # Train ensemble
                progress=progress,
            )
            
            assert "ensemble" in trained_models
            assert trained_models["ensemble"] == mock_ensemble
            mock_ensemble.train.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_train_models_specific_type(self, sample_training_config, mock_feature_engine,
                                             mock_feature_store, mock_database_manager,
                                             sample_features_targets):
        """Test training specific model type (not ensemble)."""
        features_df, targets_df = sample_features_targets
        train_data = (features_df[:70], targets_df[:70])
        val_data = (features_df[70:90], targets_df[70:90])
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        # Test with specific model type that's not implemented
        trained_models = await pipeline._train_models(
            room_id="living_room",
            train_data=train_data,
            val_data=val_data,
            target_model_type="lstm",  # Specific model type
            progress=progress,
        )
        
        # Should return empty dict for unimplemented specific types
        assert len(trained_models) == 0
        
    @pytest.mark.asyncio
    async def test_train_models_failure(self, sample_training_config, mock_feature_engine,
                                       mock_feature_store, mock_database_manager,
                                       sample_features_targets, mock_tracking_manager):
        """Test model training failure."""
        features_df, targets_df = sample_features_targets
        train_data = (features_df[:70], targets_df[:70])
        val_data = (features_df[70:90], targets_df[70:90])
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            tracking_manager=mock_tracking_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        
        # Mock OccupancyEnsemble to raise exception during training
        with patch('src.models.training_pipeline.OccupancyEnsemble') as mock_ensemble_class:
            mock_ensemble = Mock()
            mock_ensemble.train = AsyncMock(side_effect=Exception("Training failed"))
            mock_ensemble_class.return_value = mock_ensemble
            
            with pytest.raises(ModelTrainingError) as exc_info:
                await pipeline._train_models(
                    room_id="living_room",
                    train_data=train_data,
                    val_data=val_data,
                    target_model_type=None,  # Train ensemble
                    progress=progress,
                )
                
            assert "ensemble" in str(exc_info.value)
            assert "living_room" in str(exc_info.value)
            
    @pytest.mark.asyncio
    async def test_validate_models(self, sample_training_config, mock_feature_engine,
                                  mock_feature_store, mock_database_manager,
                                  sample_features_targets, mock_base_predictor):
        """Test model validation."""
        features_df, targets_df = sample_features_targets
        val_data = (features_df[70:90], targets_df[70:90])
        test_data = (features_df[90:], targets_df[90:])
        
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        trained_models = {"ensemble": mock_base_predictor}
        
        validation_results = await pipeline._validate_models(
            trained_models, val_data, test_data, progress
        )
        
        assert "ensemble" in validation_results
        assert isinstance(validation_results["ensemble"], (int, float))
        mock_base_predictor.predict.assert_called()
        
    @pytest.mark.asyncio
    async def test_evaluate_and_select_best_model(self, sample_training_config, mock_feature_engine,
                                                 mock_feature_store, mock_database_manager,
                                                 mock_base_predictor):
        """Test model evaluation and selection."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        trained_models = {"ensemble": mock_base_predictor, "xgboost": mock_base_predictor}
        validation_results = {"ensemble": 0.15, "xgboost": 0.20}  # MAE scores
        
        best_model, evaluation_metrics = await pipeline._evaluate_and_select_best_model(
            trained_models, validation_results, progress
        )
        
        # For MAE, lower is better, so ensemble (0.15) should be selected over xgboost (0.20)
        assert best_model == "ensemble"
        assert evaluation_metrics["best_model"] == "ensemble"
        assert evaluation_metrics["mae"] == 0.15
        assert "r2" in evaluation_metrics
        assert "rmse" in evaluation_metrics
        
    @pytest.mark.asyncio
    async def test_evaluate_and_select_best_model_empty_results(self, sample_training_config,
                                                               mock_feature_engine, mock_feature_store,
                                                               mock_database_manager):
        """Test model evaluation with empty validation results."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        trained_models = {}
        validation_results = {}
        
        with pytest.raises(ModelTrainingError) as exc_info:
            await pipeline._evaluate_and_select_best_model(
                trained_models, validation_results, progress
            )
            
        assert "ensemble" in str(exc_info.value)
        assert "model_selection" in str(exc_info.value)
        
    def test_meets_quality_thresholds(self, sample_training_config, mock_feature_engine,
                                     mock_feature_store, mock_database_manager):
        """Test quality threshold checking."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Good quality metrics - meets thresholds
        good_metrics = {
            "r2": 0.85,  # Above 0.6 threshold
            "mae": 900,  # Below 1800 seconds (30 minutes)
        }
        
        assert pipeline._meets_quality_thresholds(good_metrics) is True
        
        # Poor quality metrics - doesn't meet thresholds
        poor_metrics = {
            "r2": 0.4,  # Below 0.6 threshold
            "mae": 2400,  # Above 1800 seconds (30 minutes)
        }
        
        assert pipeline._meets_quality_thresholds(poor_metrics) is False
        
    @pytest.mark.asyncio
    async def test_deploy_trained_models(self, sample_training_config, mock_feature_engine,
                                        mock_feature_store, mock_database_manager,
                                        mock_base_predictor, tmp_path):
        """Test model deployment."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            model_artifacts_path=tmp_path,
        )
        
        progress = TrainingProgress("test-123", TrainingType.INITIAL)
        trained_models = {"ensemble": mock_base_predictor}
        
        with patch.object(pipeline, '_save_model_artifacts') as mock_save:
            mock_save.return_value = tmp_path / "ensemble"
            
            deployment_info = await pipeline._deploy_trained_models(
                "living_room", trained_models, "ensemble", progress
            )
            
            assert "deployed_models" in deployment_info
            assert "best_model" in deployment_info
            assert deployment_info["best_model"] == "ensemble"
            assert len(deployment_info["deployed_models"]) == 1
            
            deployed_model = deployment_info["deployed_models"][0]
            assert deployed_model["model_name"] == "ensemble"
            assert deployed_model["is_best"] is True
            
            # Check model registry was updated
            assert "living_room_ensemble" in pipeline._model_registry
            assert pipeline._model_registry["living_room_ensemble"] == mock_base_predictor


class TestModelArtifacts:
    """Test model artifact management."""
    
    @pytest.mark.asyncio
    async def test_save_model_artifacts(self, sample_training_config, mock_feature_engine,
                                       mock_feature_store, mock_database_manager,
                                       mock_base_predictor, tmp_path):
        """Test saving model artifacts."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            model_artifacts_path=tmp_path,
        )
        
        model_dir = await pipeline._save_model_artifacts(
            "living_room", "ensemble", mock_base_predictor, "v1.0"
        )
        
        assert model_dir.exists()
        assert (model_dir / "model.pkl").exists()
        assert (model_dir / "metadata.json").exists()
        
        # Check metadata content
        metadata_file = model_dir / "metadata.json"
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)
            
        assert metadata["room_id"] == "living_room"
        assert metadata["model_name"] == "ensemble"
        assert metadata["model_version"] == "v1.0"
        assert "training_date" in metadata
        
    @pytest.mark.asyncio
    async def test_load_model_from_artifacts(self, sample_training_config, mock_feature_engine,
                                            mock_feature_store, mock_database_manager,
                                            tmp_path):
        """Test loading model from artifacts."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            model_artifacts_path=tmp_path,
        )
        
        # Create test model artifacts
        model_dir = tmp_path / "living_room" / "ensemble" / "v1.0"
        model_dir.mkdir(parents=True)
        
        # Save a simple test object
        test_model = {"type": "test_model", "version": "v1.0"}
        with open(model_dir / "model.pkl", "wb") as f:
            pickle.dump(test_model, f)
            
        # Load the model
        loaded_model = await pipeline.load_model_from_artifacts(
            "living_room", "ensemble", "v1.0"
        )
        
        assert loaded_model is not None
        assert loaded_model["type"] == "test_model"
        assert loaded_model["version"] == "v1.0"
        
    @pytest.mark.asyncio
    async def test_load_model_from_artifacts_missing(self, sample_training_config,
                                                    mock_feature_engine, mock_feature_store,
                                                    mock_database_manager, tmp_path):
        """Test loading non-existent model artifacts."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
            model_artifacts_path=tmp_path,
        )
        
        loaded_model = await pipeline.load_model_from_artifacts(
            "nonexistent_room", "nonexistent_model", "v1.0"
        )
        
        assert loaded_model is None


class TestPipelineManagement:
    """Test pipeline management methods."""
    
    def test_get_active_pipelines(self, sample_training_config, mock_feature_engine,
                                 mock_feature_store, mock_database_manager):
        """Test getting active pipelines."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Add a test pipeline
        test_progress = TrainingProgress("test-123", TrainingType.INITIAL)
        pipeline._active_pipelines["test-123"] = test_progress
        
        active_pipelines = pipeline.get_active_pipelines()
        
        assert "test-123" in active_pipelines
        assert active_pipelines["test-123"] == test_progress
        
    def test_get_pipeline_history(self, sample_training_config, mock_feature_engine,
                                 mock_feature_store, mock_database_manager):
        """Test getting pipeline history."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Add test history
        for i in range(5):
            test_progress = TrainingProgress(f"test-{i}", TrainingType.INITIAL)
            pipeline._pipeline_history.append(test_progress)
            
        history = pipeline.get_pipeline_history(limit=3)
        
        assert len(history) == 3
        # Should return most recent (last 3)
        assert history[0].pipeline_id == "test-2"
        assert history[2].pipeline_id == "test-4"
        
    def test_get_training_statistics(self, sample_training_config, mock_feature_engine,
                                   mock_feature_store, mock_database_manager):
        """Test getting training statistics."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Update statistics
        pipeline._training_stats["total_pipelines_run"] = 10
        pipeline._training_stats["successful_pipelines"] = 8
        pipeline._training_stats["failed_pipelines"] = 2
        
        stats = pipeline.get_training_statistics()
        
        assert stats["total_pipelines_run"] == 10
        assert stats["successful_pipelines"] == 8
        assert stats["failed_pipelines"] == 2
        
    def test_get_model_registry(self, sample_training_config, mock_feature_engine,
                               mock_feature_store, mock_database_manager, mock_base_predictor):
        """Test getting model registry."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Add test model
        pipeline._model_registry["living_room_ensemble"] = mock_base_predictor
        
        registry = pipeline.get_model_registry()
        
        assert "living_room_ensemble" in registry
        assert registry["living_room_ensemble"] == mock_base_predictor
        
    @pytest.mark.asyncio
    async def test_get_model_performance(self, sample_training_config, mock_feature_engine,
                                        mock_feature_store, mock_database_manager,
                                        mock_base_predictor):
        """Test getting model performance metrics."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Add test model with training history
        mock_base_predictor.training_history = [
            TrainingResult(
                success=True,
                training_score=0.85,
                validation_score=0.78,
                training_samples=100,
                training_time_seconds=120,
            )
        ]
        pipeline._model_registry["living_room_ensemble"] = mock_base_predictor
        
        performance = await pipeline.get_model_performance("living_room", "ensemble")
        
        assert performance is not None
        assert performance["room_id"] == "living_room"
        assert performance["model_type"] == "ensemble"
        assert performance["is_trained"] is True
        assert performance["latest_training_score"] == 0.85
        assert performance["latest_validation_score"] == 0.78
        
    @pytest.mark.asyncio
    async def test_get_model_performance_missing(self, sample_training_config, mock_feature_engine,
                                                mock_feature_store, mock_database_manager):
        """Test getting performance for non-existent model."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        performance = await pipeline.get_model_performance("nonexistent_room", "nonexistent_model")
        
        assert performance is None


class TestTrainingConfigurationComponents:
    """Test training configuration system components."""
    
    def test_training_profile_enum(self):
        """Test TrainingProfile enum values and methods."""
        # Test all profile values
        assert TrainingProfile.DEVELOPMENT.value == "development"
        assert TrainingProfile.PRODUCTION.value == "production"
        assert TrainingProfile.TESTING.value == "testing"
        assert TrainingProfile.RESEARCH.value == "research"
        assert TrainingProfile.QUICK.value == "quick"
        assert TrainingProfile.COMPREHENSIVE.value == "comprehensive"
        
    def test_training_profile_from_string(self):
        """Test TrainingProfile.from_string() method."""
        # Test valid profiles
        assert TrainingProfile.from_string("development") == TrainingProfile.DEVELOPMENT
        assert TrainingProfile.from_string("production") == TrainingProfile.PRODUCTION
        
        # Test invalid profile
        with pytest.raises(ValueError) as exc_info:
            TrainingProfile.from_string("invalid_profile")
        assert "Training profile invalid_profile not available" in str(exc_info.value)
        
    def test_optimization_level_enum(self):
        """Test OptimizationLevel enum values."""
        assert OptimizationLevel.NONE.value == "none"
        assert OptimizationLevel.BASIC.value == "basic"
        assert OptimizationLevel.STANDARD.value == "standard"
        assert OptimizationLevel.INTENSIVE.value == "intensive"
        
    def test_resource_limits_validation(self):
        """Test ResourceLimits validation."""
        # Valid resource limits
        valid_limits = ResourceLimits(
            max_memory_gb=8.0,
            max_cpu_cores=4,
            max_training_time_minutes=60,
            max_parallel_models=2,
        )
        
        issues = valid_limits.validate()
        assert len(issues) == 0
        
        # Invalid resource limits
        invalid_limits = ResourceLimits(
            max_memory_gb=-2.0,  # Negative
            max_cpu_cores=0,  # Zero
            max_training_time_minutes=-30,  # Negative
            max_parallel_models=-1,  # Negative
        )
        
        issues = invalid_limits.validate()
        assert len(issues) > 0
        assert any("max_memory_gb must be positive" in issue for issue in issues)


class TestTrainingIntegrationManager:
    """Test training integration manager functionality."""
    
    @pytest.mark.asyncio
    async def test_integration_manager_initialization(self, mock_tracking_manager, 
                                                     sample_training_config, mock_feature_engine,
                                                     mock_feature_store, mock_database_manager):
        """Test TrainingIntegrationManager initialization."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        assert integration_manager.tracking_manager == mock_tracking_manager
        assert integration_manager.training_pipeline == pipeline
        assert len(integration_manager._active_training_requests) == 0
        assert len(integration_manager._training_queue) == 0
        assert integration_manager._integration_active is False
        
    @pytest.mark.asyncio
    async def test_integration_manager_initialize(self, mock_tracking_manager,
                                                 sample_training_config, mock_feature_engine,
                                                 mock_feature_store, mock_database_manager):
        """Test integration manager initialization process."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Mock background task creation to avoid actual asyncio tasks
        with patch.object(integration_manager, '_start_background_tasks') as mock_tasks, \
             patch.object(integration_manager, '_register_tracking_callbacks') as mock_callbacks:
            
            await integration_manager.initialize()
            
            assert integration_manager._integration_active is True
            mock_tasks.assert_called_once()
            mock_callbacks.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_integration_manager_shutdown(self, mock_tracking_manager,
                                               sample_training_config, mock_feature_engine,
                                               mock_feature_store, mock_database_manager):
        """Test integration manager shutdown process."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Set as active and add mock background task
        integration_manager._integration_active = True
        mock_task = Mock()
        mock_task.cancel = Mock()
        integration_manager._background_tasks.append(mock_task)
        
        # Mock asyncio.gather to avoid actual task management
        with patch('asyncio.gather', new_callable=AsyncMock) as mock_gather:
            await integration_manager.shutdown()
            
            assert integration_manager._integration_active is False
            assert integration_manager._shutdown_event.is_set()
            mock_gather.assert_called_once()
            
    @pytest.mark.asyncio
    async def test_on_accuracy_degradation(self, mock_tracking_manager, sample_training_config,
                                          mock_feature_engine, mock_feature_store, 
                                          mock_database_manager):
        """Test accuracy degradation handling."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        # Mock training config manager
        mock_config_manager = Mock()
        mock_env_config = Mock()
        mock_env_config.quality_thresholds = Mock()
        mock_env_config.quality_thresholds.min_accuracy_threshold = 0.7
        mock_env_config.quality_thresholds.max_error_threshold_minutes = 20.0
        mock_config_manager.get_environment_config.return_value = mock_env_config
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
            config_manager=mock_config_manager,
        )
        
        # Mock queue retraining request
        with patch.object(integration_manager, '_queue_retraining_request') as mock_queue:
            # Test degradation that triggers retraining
            accuracy_metrics = {
                "accuracy_rate": 60.0,  # Below 70% threshold
                "mean_error_minutes": 25.0,  # Above 20 minute threshold
            }
            
            await integration_manager._on_accuracy_degradation("living_room", accuracy_metrics)
            
            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["room_id"] == "living_room"
            assert "accuracy_degradation" in call_args[1]["trigger_reason"]
            
    @pytest.mark.asyncio
    async def test_on_drift_detected(self, mock_tracking_manager, sample_training_config,
                                    mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test concept drift detection handling."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        with patch.object(integration_manager, '_queue_retraining_request') as mock_queue:
            # Test critical drift that triggers full retraining
            drift_metrics = {
                "drift_severity": "CRITICAL",
                "overall_drift_score": 0.9,
                "retraining_recommended": True,
                "drift_types": ["feature_drift", "label_drift"],
            }
            
            await integration_manager._on_drift_detected("bedroom", drift_metrics)
            
            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["room_id"] == "bedroom"
            assert call_args[1]["strategy"] == "full_retrain"
            assert call_args[1]["priority"] == 1  # High priority for critical drift
            
    @pytest.mark.asyncio
    async def test_queue_retraining_request(self, mock_tracking_manager, sample_training_config,
                                           mock_feature_engine, mock_feature_store, 
                                           mock_database_manager):
        """Test queuing retraining requests."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Test queuing a request
        await integration_manager._queue_retraining_request(
            room_id="kitchen",
            trigger_reason="test_trigger",
            priority=2,
            strategy="adaptive",
            metadata={"test": "data"},
        )
        
        assert len(integration_manager._training_queue) == 1
        request = integration_manager._training_queue[0]
        assert request["room_id"] == "kitchen"
        assert request["trigger_reason"] == "test_trigger"
        assert request["priority"] == 2
        assert request["strategy"] == "adaptive"
        assert request["metadata"]["test"] == "data"
        
    def test_calculate_priority(self, mock_tracking_manager, sample_training_config,
                               mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test priority calculation logic."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Test different priority scenarios
        assert integration_manager._calculate_priority(0.0, 70.0) == 1  # Critical (0%)
        assert integration_manager._calculate_priority(30.0, 70.0) == 1  # Critical (<50%)
        assert integration_manager._calculate_priority(45.0, 70.0) == 2  # High (<70%)
        assert integration_manager._calculate_priority(60.0, 70.0) == 3  # Medium (<90%)
        assert integration_manager._calculate_priority(68.0, 70.0) == 4  # Low (>=90%)
        
    def test_can_retrain_room(self, mock_tracking_manager, sample_training_config,
                             mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test room retraining cooldown logic."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Test room with no previous training - can retrain
        assert integration_manager._can_retrain_room("new_room") is True
        
        # Test room with recent training - cannot retrain
        from datetime import datetime, timedelta
        recent_time = datetime.utcnow() - timedelta(hours=6)  # 6 hours ago
        integration_manager._last_training_times["recent_room"] = recent_time
        assert integration_manager._can_retrain_room("recent_room") is False
        
        # Test room with old training - can retrain
        old_time = datetime.utcnow() - timedelta(hours=24)  # 24 hours ago
        integration_manager._last_training_times["old_room"] = old_time
        assert integration_manager._can_retrain_room("old_room") is True
        
    def test_get_cooldown_remaining(self, mock_tracking_manager, sample_training_config,
                                   mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test cooldown time calculation."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Test room with no previous training
        assert integration_manager._get_cooldown_remaining("new_room") == 0.0
        
        # Test room with recent training
        from datetime import datetime, timedelta
        recent_time = datetime.utcnow() - timedelta(hours=6)  # 6 hours ago
        integration_manager._last_training_times["recent_room"] = recent_time
        remaining = integration_manager._get_cooldown_remaining("recent_room")
        assert remaining == 6.0  # 12 hour cooldown - 6 hours elapsed = 6 hours remaining
        
    @pytest.mark.asyncio
    async def test_request_manual_training(self, mock_tracking_manager, sample_training_config,
                                          mock_feature_engine, mock_feature_store, 
                                          mock_database_manager):
        """Test manual training request API."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        with patch.object(integration_manager, '_queue_retraining_request') as mock_queue:
            result = await integration_manager.request_manual_training(
                room_id="office",
                strategy="quick",
                priority=1,
                reason="user_request",
            )
            
            assert result is True
            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["room_id"] == "office"
            assert call_args[1]["strategy"] == "quick"
            assert call_args[1]["priority"] == 1
            assert "manual_user_request" in call_args[1]["trigger_reason"]
            
    def test_get_integration_status(self, mock_tracking_manager, sample_training_config,
                                   mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test integration status reporting."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Add some test data
        integration_manager._integration_active = True
        integration_manager._active_training_requests["room1"] = "pipeline1"
        integration_manager._training_queue.append({
            "room_id": "room2",
            "trigger_reason": "test",
            "strategy": "adaptive",
            "priority": 3,
        })
        
        status = integration_manager.get_integration_status()
        
        assert status["integration_active"] is True
        assert status["active_training_requests"] == 1
        assert status["queued_training_requests"] == 1
        assert status["max_concurrent_training"] == 2
        assert "room1" in status["rooms_with_active_training"]
        assert "room2" in status["next_queued_rooms"]
        
    def test_get_training_queue_status(self, mock_tracking_manager, sample_training_config,
                                      mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test training queue status reporting."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Add test queue items
        from datetime import datetime
        now = datetime.utcnow()
        integration_manager._training_queue.extend([
            {
                "room_id": "room1",
                "trigger_reason": "accuracy_degradation",
                "strategy": "adaptive",
                "priority": 2,
                "requested_at": now - timedelta(minutes=10),
            },
            {
                "room_id": "room2", 
                "trigger_reason": "concept_drift",
                "strategy": "full_retrain",
                "priority": 1,
                "requested_at": now - timedelta(minutes=5),
            },
        ])
        
        queue_status = integration_manager.get_training_queue_status()
        
        assert len(queue_status) == 2
        assert queue_status[0]["room_id"] == "room1"
        assert queue_status[0]["waiting_time_minutes"] == pytest.approx(10, rel=0.1)
        assert queue_status[1]["room_id"] == "room2"
        assert queue_status[1]["waiting_time_minutes"] == pytest.approx(5, rel=0.1)
        
    def test_set_training_capacity(self, mock_tracking_manager, sample_training_config,
                                  mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test setting training capacity."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Test valid capacity setting
        integration_manager.set_training_capacity(4)
        assert integration_manager._max_concurrent_training == 4
        
        # Test invalid capacity (should not change)
        original_capacity = integration_manager._max_concurrent_training
        integration_manager.set_training_capacity(0)
        assert integration_manager._max_concurrent_training == original_capacity
        
    def test_set_cooldown_period(self, mock_tracking_manager, sample_training_config,
                                mock_feature_engine, mock_feature_store, mock_database_manager):
        """Test setting cooldown period."""
        pipeline = ModelTrainingPipeline(
            config=sample_training_config,
            feature_engineering_engine=mock_feature_engine,
            feature_store=mock_feature_store,
            database_manager=mock_database_manager,
        )
        
        integration_manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=pipeline,
        )
        
        # Test valid cooldown setting
        integration_manager.set_cooldown_period(8)
        assert integration_manager._training_cooldown_hours == 8
        
        # Test invalid cooldown (should not change)
        original_cooldown = integration_manager._training_cooldown_hours
        integration_manager.set_cooldown_period(-1)
        assert integration_manager._training_cooldown_hours == original_cooldown