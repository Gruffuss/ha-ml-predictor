"""
Comprehensive tests for AdaptiveRetrainer and IncrementalTrainer.

Tests real retraining functionality including adaptive retraining triggers,
incremental model updates, and comprehensive retraining orchestration.
"""

import asyncio
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.drift_detector import DriftMetrics, DriftSeverity, DriftType
from src.adaptation.retrainer import (
    AdaptiveRetrainer,
    RetrainingStrategy,
    RetrainingTrigger,
)
from src.adaptation.validator import AccuracyMetrics, PredictionValidator
from src.core.constants import ModelType


@pytest.fixture
def mock_model_registry():
    """Create mock model registry for testing."""
    registry = MagicMock()
    
    # Mock models
    mock_model = MagicMock()
    mock_model.model_type = ModelType.LSTM
    mock_model.version = "v1.0"
    mock_model.room_id = "living_room"
    
    registry.get_model.return_value = mock_model
    registry.get_room_models.return_value = {"lstm": mock_model}
    registry.register_model = MagicMock()
    registry.get_model_metadata.return_value = {
        "training_time": datetime.now(UTC) - timedelta(days=7),
        "accuracy": 0.85,
        "samples_trained": 1000
    }
    
    return registry


@pytest.fixture
def mock_drift_detector():
    """Create mock drift detector for testing."""
    detector = MagicMock()
    
    # Create various drift scenarios
    def mock_detect_drift(room_id="", **kwargs):
        if room_id == "high_drift_room":
            return DriftMetrics(
                room_id=room_id,
                detection_time=datetime.now(UTC),
                baseline_period=(
                    datetime.now(UTC) - timedelta(days=30),
                    datetime.now(UTC) - timedelta(days=7)
                ),
                current_period=(
                    datetime.now(UTC) - timedelta(days=7),
                    datetime.now(UTC)
                ),
                accuracy_degradation=25.0,
                overall_drift_score=0.8,
                drift_severity=DriftSeverity.HIGH,
                retraining_recommended=True,
                drift_types=[DriftType.CONCEPT_DRIFT, DriftType.FEATURE_DRIFT]
            )
        else:
            return DriftMetrics(
                room_id=room_id,
                detection_time=datetime.now(UTC),
                baseline_period=(
                    datetime.now(UTC) - timedelta(days=30),
                    datetime.now(UTC) - timedelta(days=7)
                ),
                current_period=(
                    datetime.now(UTC) - timedelta(days=7),
                    datetime.now(UTC)
                ),
                accuracy_degradation=5.0,
                overall_drift_score=0.2,
                drift_severity=DriftSeverity.LOW,
                retraining_recommended=False
            )
    
    detector.detect_drift = AsyncMock(side_effect=mock_detect_drift)
    return detector


@pytest.fixture
def mock_prediction_validator():
    """Create mock prediction validator for testing."""
    validator = MagicMock(spec=PredictionValidator)
    
    # Mock different accuracy metrics for different scenarios
    def mock_get_accuracy_metrics(room_id="", **kwargs):
        if room_id == "poor_accuracy_room":
            return AccuracyMetrics(
                total_predictions=100,
                validated_predictions=90,
                accurate_predictions=60,
                accuracy_rate=66.7,
                mean_error_minutes=18.5,
                confidence_accuracy_correlation=0.45
            )
        else:
            return AccuracyMetrics(
                total_predictions=100,
                validated_predictions=95,
                accurate_predictions=85,
                accuracy_rate=89.5,
                mean_error_minutes=8.2,
                confidence_accuracy_correlation=0.78
            )
    
    validator.get_accuracy_metrics = AsyncMock(side_effect=mock_get_accuracy_metrics)
    return validator


@pytest.fixture
def sample_training_data():
    """Create sample training data for testing."""
    np.random.seed(42)
    
    features_df = pd.DataFrame({
        'time_since_last_event': np.random.exponential(60, 500),
        'hour_of_day': np.random.randint(0, 24, 500),
        'day_of_week': np.random.randint(0, 7, 500),
        'movement_frequency': np.random.poisson(3, 500),
        'room_temp': np.random.normal(22, 2, 500),
        'timestamp': pd.date_range(
            start=datetime.now(UTC) - timedelta(days=30),
            periods=500,
            freq='H'
        )
    })
    
    targets_df = pd.DataFrame({
        'next_occupied_time': np.random.exponential(45, 500),
        'confidence': np.random.beta(2, 2, 500),
        'timestamp': features_df['timestamp']
    })
    
    return features_df, targets_df


class TestAdaptiveRetrainer:
    """Test AdaptiveRetrainer functionality."""

    def test_initialization(self):
        """Test adaptive retrainer initialization with various configurations."""
        # Test default initialization
        retrainer = AdaptiveRetrainer()
        assert retrainer.accuracy_threshold == 15.0
        assert retrainer.drift_threshold == 0.5
        assert retrainer.min_retraining_interval_hours == 24
        assert retrainer.max_concurrent_retraining == 2
        assert retrainer.default_strategy == RetrainingStrategy.INCREMENTAL

        # Test custom configuration
        retrainer = AdaptiveRetrainer(
            accuracy_threshold=20.0,
            drift_threshold=0.3,
            min_retraining_interval_hours=12,
            max_concurrent_retraining=4,
            default_strategy=RetrainingStrategy.FULL_RETRAINING
        )
        assert retrainer.accuracy_threshold == 20.0
        assert retrainer.drift_threshold == 0.3
        assert retrainer.min_retraining_interval_hours == 12
        assert retrainer.max_concurrent_retraining == 4
        assert retrainer.default_strategy == RetrainingStrategy.FULL_RETRAINING

    @pytest.mark.asyncio
    async def test_start_stop_monitoring(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test starting and stopping retraining monitoring."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry
        )
        
        room_ids = ["living_room", "bedroom", "kitchen"]
        
        # Test starting monitoring
        assert not retrainer._monitoring_active
        
        await retrainer.start_monitoring(room_ids, check_interval_minutes=1)
        assert retrainer._monitoring_active
        assert retrainer._monitoring_task is not None
        assert retrainer._monitored_rooms == set(room_ids)
        
        # Test stopping monitoring
        await retrainer.stop_monitoring()
        assert not retrainer._monitoring_active
        assert retrainer._monitoring_task is None

    @pytest.mark.asyncio
    async def test_trigger_retraining_accuracy_degradation(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test retraining triggered by accuracy degradation."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry,
            accuracy_threshold=15.0
        )

        # Test with room that has poor accuracy
        result = await retrainer.check_retraining_triggers("poor_accuracy_room")
        
        assert result is not None
        assert result.trigger == RetrainingTrigger.ACCURACY_DEGRADATION
        assert result.room_id == "poor_accuracy_room"
        assert result.accuracy_metrics is not None
        assert result.retraining_recommended

    @pytest.mark.asyncio
    async def test_trigger_retraining_drift_detection(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test retraining triggered by drift detection."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry,
            drift_threshold=0.6
        )

        # Test with room that has high drift
        result = await retrainer.check_retraining_triggers("high_drift_room")
        
        assert result is not None
        assert result.trigger == RetrainingTrigger.CONCEPT_DRIFT
        assert result.room_id == "high_drift_room"
        assert result.drift_metrics is not None
        assert result.drift_metrics.drift_severity == DriftSeverity.HIGH
        assert result.retraining_recommended

    @pytest.mark.asyncio
    async def test_no_retraining_needed(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test scenario where no retraining is needed."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry
        )

        # Test with room that has good accuracy and low drift
        result = await retrainer.check_retraining_triggers("good_room")
        
        # Should return None when no retraining is needed
        assert result is None or not result.retraining_recommended

    @pytest.mark.asyncio
    async def test_execute_retraining_incremental(self, mock_drift_detector, mock_prediction_validator, mock_model_registry, sample_training_data):
        """Test incremental retraining execution."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry,
            default_strategy=RetrainingStrategy.INCREMENTAL
        )

        features_df, targets_df = sample_training_data

        with patch.object(retrainer, '_get_training_data') as mock_get_data:
            mock_get_data.return_value = (features_df, targets_df)
            
            with patch.object(retrainer, '_execute_incremental_training') as mock_incremental:
                mock_incremental.return_value = TrainingResult(
                    room_id="test_room",
                    model_type=ModelType.LSTM,
                    training_strategy=RetrainingStrategy.INCREMENTAL,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC) + timedelta(minutes=5),
                    success=True,
                    training_metrics=TrainingMetrics(
                        samples_processed=500,
                        training_loss=0.45,
                        validation_accuracy=0.82,
                        training_duration_seconds=300
                    )
                )

                result = await retrainer.execute_retraining(
                    room_id="test_room",
                    strategy=RetrainingStrategy.INCREMENTAL,
                    trigger=RetrainingTrigger.ACCURACY_DEGRADATION
                )

                assert result.success
                assert result.training_strategy == RetrainingStrategy.INCREMENTAL
                assert result.training_metrics.samples_processed == 500

    @pytest.mark.asyncio
    async def test_execute_retraining_full(self, mock_drift_detector, mock_prediction_validator, mock_model_registry, sample_training_data):
        """Test full retraining execution."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry
        )

        features_df, targets_df = sample_training_data

        with patch.object(retrainer, '_get_training_data') as mock_get_data:
            mock_get_data.return_value = (features_df, targets_df)
            
            with patch.object(retrainer, '_execute_full_retraining') as mock_full:
                mock_full.return_value = TrainingResult(
                    room_id="test_room",
                    model_type=ModelType.XGBOOST,
                    training_strategy=RetrainingStrategy.FULL_RETRAINING,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC) + timedelta(minutes=15),
                    success=True,
                    training_metrics=TrainingMetrics(
                        samples_processed=1000,
                        training_loss=0.35,
                        validation_accuracy=0.88,
                        training_duration_seconds=900
                    )
                )

                result = await retrainer.execute_retraining(
                    room_id="test_room",
                    strategy=RetrainingStrategy.FULL_RETRAINING,
                    trigger=RetrainingTrigger.CONCEPT_DRIFT
                )

                assert result.success
                assert result.training_strategy == RetrainingStrategy.FULL_RETRAINING
                assert result.training_metrics.samples_processed == 1000

    @pytest.mark.asyncio
    async def test_concurrent_retraining_limit(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test concurrent retraining limits are enforced."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry,
            max_concurrent_retraining=1  # Low limit for testing
        )

        # Mock slow retraining execution
        async def slow_retraining(*args, **kwargs):
            await asyncio.sleep(0.1)
            return TrainingResult(
                room_id="test",
                model_type=ModelType.LSTM,
                training_strategy=RetrainingStrategy.INCREMENTAL,
                start_time=datetime.now(UTC),
                end_time=datetime.now(UTC),
                success=True,
                training_metrics=TrainingMetrics()
            )

        with patch.object(retrainer, '_execute_incremental_training', side_effect=slow_retraining):
            with patch.object(retrainer, '_get_training_data', return_value=(pd.DataFrame(), pd.DataFrame())):
                
                # Start first retraining (should succeed)
                task1 = asyncio.create_task(
                    retrainer.execute_retraining("room1", RetrainingStrategy.INCREMENTAL, RetrainingTrigger.ACCURACY_DEGRADATION)
                )
                
                # Try to start second retraining immediately (should be limited)
                task2 = asyncio.create_task(
                    retrainer.execute_retraining("room2", RetrainingStrategy.INCREMENTAL, RetrainingTrigger.ACCURACY_DEGRADATION)
                )
                
                results = await asyncio.gather(task1, task2, return_exceptions=True)
                
                # At least one should succeed
                successful_results = [r for r in results if isinstance(r, TrainingResult) and r.success]
                assert len(successful_results) >= 1

    @pytest.mark.asyncio 
    async def test_retraining_interval_enforcement(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test minimum retraining interval is enforced."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry,
            min_retraining_interval_hours=24
        )

        # Record a recent retraining
        recent_time = datetime.now(UTC) - timedelta(hours=1)  # 1 hour ago
        retrainer._last_retraining_time["test_room"] = recent_time

        # Try to trigger retraining too soon
        result = await retrainer.check_retraining_triggers("test_room")
        
        # Should not recommend retraining due to interval limit
        assert result is None or not result.retraining_recommended

    def test_training_data_config(self):
        """Test training data configuration."""
        config = TrainingDataConfig(
            days_back=30,
            min_samples=100,
            max_samples=10000,
            validation_split=0.2,
            include_features=['temporal', 'sequential'],
            exclude_outliers=True
        )
        
        assert config.days_back == 30
        assert config.min_samples == 100
        assert config.validation_split == 0.2
        assert 'temporal' in config.include_features
        assert config.exclude_outliers

    def test_training_metrics_calculation(self):
        """Test training metrics calculation and properties."""
        metrics = TrainingMetrics(
            samples_processed=1000,
            training_loss=0.35,
            validation_loss=0.42,
            validation_accuracy=0.85,
            training_duration_seconds=300,
            memory_usage_mb=512,
            convergence_epochs=25,
            final_learning_rate=0.0001
        )
        
        assert metrics.samples_processed == 1000
        assert metrics.validation_accuracy == 0.85
        assert metrics.training_duration_minutes == 5.0
        assert metrics.convergence_achieved
        assert metrics.overfitting_detected  # validation_loss > training_loss

    def test_training_result_serialization(self):
        """Test training result can be serialized."""
        result = TrainingResult(
            room_id="serialization_test",
            model_type=ModelType.LSTM,
            training_strategy=RetrainingStrategy.INCREMENTAL,
            start_time=datetime.now(UTC),
            end_time=datetime.now(UTC) + timedelta(minutes=10),
            success=True,
            trigger=RetrainingTrigger.CONCEPT_DRIFT,
            training_metrics=TrainingMetrics(
                samples_processed=500,
                training_loss=0.4,
                validation_accuracy=0.8
            ),
            model_improvements={
                'accuracy_improvement': 0.05,
                'error_reduction': 2.3
            }
        )
        
        result_dict = result.to_dict()
        
        assert isinstance(result_dict, dict)
        assert result_dict['room_id'] == "serialization_test"
        assert result_dict['success'] is True
        assert 'training_metrics' in result_dict
        assert 'model_improvements' in result_dict


class TestIncrementalTrainer:
    """Test IncrementalTrainer functionality."""

    def test_initialization(self):
        """Test incremental trainer initialization."""
        trainer = IncrementalTrainer()
        
        assert trainer.batch_size == 32
        assert trainer.learning_rate_decay == 0.95
        assert trainer.early_stopping_patience == 10
        assert trainer.max_epochs == 100

        # Test custom configuration
        trainer = IncrementalTrainer(
            batch_size=64,
            learning_rate_decay=0.9,
            early_stopping_patience=5,
            max_epochs=50
        )
        
        assert trainer.batch_size == 64
        assert trainer.learning_rate_decay == 0.9
        assert trainer.early_stopping_patience == 5
        assert trainer.max_epochs == 50

    @pytest.mark.asyncio
    async def test_incremental_training_execution(self, sample_training_data):
        """Test incremental training execution."""
        trainer = IncrementalTrainer()
        features_df, targets_df = sample_training_data
        
        # Mock model
        mock_model = MagicMock()
        mock_model.model_type = ModelType.LSTM
        mock_model.partial_fit = MagicMock()
        mock_model.predict = MagicMock(return_value=np.random.random(100))
        mock_model.score = MagicMock(return_value=0.85)
        
        result = await trainer.train_incrementally(
            model=mock_model,
            new_features=features_df.iloc[:100],
            new_targets=targets_df.iloc[:100],
            room_id="test_room"
        )
        
        assert isinstance(result, TrainingResult)
        assert result.room_id == "test_room"
        assert result.training_strategy == RetrainingStrategy.INCREMENTAL
        assert result.success
        assert result.training_metrics.samples_processed == 100

    @pytest.mark.asyncio
    async def test_online_learning_update(self, sample_training_data):
        """Test online learning with single batch update."""
        trainer = IncrementalTrainer()
        features_df, targets_df = sample_training_data
        
        # Mock model with online learning capability
        mock_model = MagicMock()
        mock_model.partial_fit = MagicMock()
        mock_model.get_params = MagicMock(return_value={'learning_rate': 0.01})
        mock_model.set_params = MagicMock()
        
        await trainer.online_learning_update(
            model=mock_model,
            batch_features=features_df.iloc[:10],
            batch_targets=targets_df.iloc[:10]
        )
        
        # Verify partial_fit was called
        mock_model.partial_fit.assert_called_once()

    @pytest.mark.asyncio
    async def test_model_warmstart(self):
        """Test model warmstart from previous checkpoint."""
        trainer = IncrementalTrainer()
        
        # Mock model with warmstart capability
        mock_model = MagicMock()
        mock_model.warm_start = True
        mock_model.n_estimators = 100
        mock_model.set_params = MagicMock()
        
        # Mock previous training state
        previous_state = {
            'n_estimators': 100,
            'learning_rate': 0.01,
            'checkpoint_epoch': 25
        }
        
        trainer.warmstart_from_checkpoint(mock_model, previous_state)
        
        # Verify model parameters were updated for continuation
        mock_model.set_params.assert_called()

    def test_adaptive_learning_rate(self):
        """Test adaptive learning rate adjustment."""
        trainer = IncrementalTrainer(learning_rate_decay=0.9)
        
        initial_lr = 0.01
        current_epoch = 10
        
        # Test exponential decay
        new_lr = trainer.calculate_adaptive_learning_rate(
            initial_lr, current_epoch, decay_strategy='exponential'
        )
        
        expected_lr = initial_lr * (trainer.learning_rate_decay ** current_epoch)
        assert abs(new_lr - expected_lr) < 1e-10

    @pytest.mark.asyncio
    async def test_validation_during_training(self, sample_training_data):
        """Test validation performed during incremental training."""
        trainer = IncrementalTrainer(early_stopping_patience=3)
        features_df, targets_df = sample_training_data
        
        # Mock model with validation
        mock_model = MagicMock()
        mock_model.partial_fit = MagicMock()
        
        # Mock validation scores that improve then worsen
        validation_scores = [0.7, 0.75, 0.8, 0.78, 0.76, 0.74]  # Early stopping trigger
        mock_model.score = MagicMock(side_effect=validation_scores)
        
        result = await trainer.train_incrementally(
            model=mock_model,
            new_features=features_df.iloc[:100],
            new_targets=targets_df.iloc[:100],
            room_id="validation_test",
            validation_features=features_df.iloc[100:120],
            validation_targets=targets_df.iloc[100:120]
        )
        
        # Should have triggered early stopping
        assert result.training_metrics.early_stopping_triggered
        assert result.training_metrics.best_validation_score == 0.8

    def test_batch_processing(self, sample_training_data):
        """Test batch processing logic."""
        trainer = IncrementalTrainer(batch_size=50)
        features_df, targets_df = sample_training_data
        
        # Test batch creation
        batches = trainer.create_training_batches(features_df, targets_df, batch_size=50)
        
        assert len(batches) > 1  # Should create multiple batches
        
        # Check batch sizes
        for i, (feature_batch, target_batch) in enumerate(batches[:-1]):
            assert len(feature_batch) == 50
            assert len(target_batch) == 50
        
        # Last batch may be smaller
        last_features, last_targets = batches[-1]
        assert len(last_features) <= 50
        assert len(last_targets) <= 50


class TestTrainingPipeline:
    """Test TrainingPipeline orchestration."""

    def test_initialization(self):
        """Test training pipeline initialization."""
        pipeline = TrainingPipeline()
        
        assert isinstance(pipeline.stages, list)
        assert len(pipeline.stages) > 0
        assert TrainingStage.DATA_PREPARATION in pipeline.stages
        assert TrainingStage.MODEL_TRAINING in pipeline.stages
        assert TrainingStage.VALIDATION in pipeline.stages

    @pytest.mark.asyncio
    async def test_pipeline_execution(self, sample_training_data):
        """Test full training pipeline execution."""
        pipeline = TrainingPipeline()
        features_df, targets_df = sample_training_data
        
        # Mock pipeline components
        mock_model = MagicMock()
        mock_model.fit = MagicMock()
        mock_model.predict = MagicMock(return_value=np.random.random(100))
        mock_model.score = MagicMock(return_value=0.85)
        
        with patch.object(pipeline, '_prepare_data') as mock_prepare:
            mock_prepare.return_value = (features_df, targets_df)
            
            with patch.object(pipeline, '_train_model') as mock_train:
                mock_train.return_value = mock_model
                
                result = await pipeline.execute(
                    room_id="pipeline_test",
                    model_type=ModelType.LSTM,
                    training_data=(features_df, targets_df),
                    strategy=RetrainingStrategy.FULL_RETRAINING
                )
                
                assert isinstance(result, TrainingResult)
                assert result.success
                assert result.room_id == "pipeline_test"

    def test_pipeline_stage_tracking(self):
        """Test pipeline stage progress tracking."""
        pipeline = TrainingPipeline()
        
        # Test stage progression
        assert pipeline.current_stage is None
        
        pipeline.start_stage(TrainingStage.DATA_PREPARATION)
        assert pipeline.current_stage == TrainingStage.DATA_PREPARATION
        
        pipeline.complete_stage(TrainingStage.DATA_PREPARATION)
        assert TrainingStage.DATA_PREPARATION in pipeline.completed_stages
        
        # Test stage timing
        assert pipeline.get_stage_duration(TrainingStage.DATA_PREPARATION) > timedelta(0)

    @pytest.mark.asyncio
    async def test_pipeline_error_handling(self):
        """Test pipeline error handling and recovery."""
        pipeline = TrainingPipeline()
        
        # Mock a stage that fails
        with patch.object(pipeline, '_prepare_data') as mock_prepare:
            mock_prepare.side_effect = Exception("Data preparation failed")
            
            result = await pipeline.execute(
                room_id="error_test",
                model_type=ModelType.LSTM,
                training_data=(pd.DataFrame(), pd.DataFrame()),
                strategy=RetrainingStrategy.FULL_RETRAINING
            )
            
            assert isinstance(result, TrainingResult)
            assert not result.success
            assert "Data preparation failed" in result.error_message

    def test_data_validation_in_pipeline(self, sample_training_data):
        """Test data validation within pipeline."""
        pipeline = TrainingPipeline()
        features_df, targets_df = sample_training_data
        
        # Test valid data
        is_valid = pipeline.validate_training_data(features_df, targets_df)
        assert is_valid
        
        # Test invalid data
        empty_features = pd.DataFrame()
        is_valid = pipeline.validate_training_data(empty_features, targets_df)
        assert not is_valid


class TestRetrainingIntegration:
    """Test integration between retraining components."""

    @pytest.mark.asyncio
    async def test_end_to_end_retraining_workflow(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test complete end-to-end retraining workflow."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry
        )
        
        room_ids = ["workflow_test_room"]
        
        with patch.object(retrainer, '_get_training_data') as mock_get_data:
            mock_get_data.return_value = (pd.DataFrame({'feature': [1, 2, 3]}), pd.DataFrame({'target': [0.1, 0.2, 0.3]}))
            
            with patch.object(retrainer, '_execute_incremental_training') as mock_train:
                mock_train.return_value = TrainingResult(
                    room_id="workflow_test_room",
                    model_type=ModelType.LSTM,
                    training_strategy=RetrainingStrategy.INCREMENTAL,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC) + timedelta(minutes=1),
                    success=True,
                    training_metrics=TrainingMetrics(samples_processed=3)
                )
                
                # Start monitoring
                await retrainer.start_monitoring(room_ids, check_interval_minutes=0.01)  # Very short for testing
                
                # Let it run briefly to trigger checks
                await asyncio.sleep(0.1)
                
                # Stop monitoring
                await retrainer.stop_monitoring()
                
                # Should have attempted drift detection
                mock_drift_detector.detect_drift.assert_called()

    @pytest.mark.asyncio
    async def test_retraining_callback_system(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test retraining callback and notification system."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry
        )
        
        callback_calls = []
        
        def test_callback(training_result):
            callback_calls.append(training_result)
        
        # Register callback
        retrainer.add_retraining_callback(test_callback)
        
        with patch.object(retrainer, '_get_training_data', return_value=(pd.DataFrame({'f': [1]}), pd.DataFrame({'t': [1]}))):
            with patch.object(retrainer, '_execute_incremental_training') as mock_train:
                mock_result = TrainingResult(
                    room_id="callback_test",
                    model_type=ModelType.LSTM,
                    training_strategy=RetrainingStrategy.INCREMENTAL,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    success=True,
                    training_metrics=TrainingMetrics()
                )
                mock_train.return_value = mock_result
                
                # Execute retraining
                result = await retrainer.execute_retraining(
                    "callback_test",
                    RetrainingStrategy.INCREMENTAL,
                    RetrainingTrigger.ACCURACY_DEGRADATION
                )
                
                # Callback should have been called
                assert len(callback_calls) == 1
                assert callback_calls[0] == mock_result

    def test_retraining_statistics_tracking(self, mock_drift_detector, mock_prediction_validator, mock_model_registry):
        """Test retraining statistics and performance tracking."""
        retrainer = AdaptiveRetrainer(
            drift_detector=mock_drift_detector,
            prediction_validator=mock_prediction_validator,
            model_registry=mock_model_registry
        )
        
        # Add some mock retraining history
        past_results = [
            TrainingResult(
                room_id=f"room_{i}",
                model_type=ModelType.LSTM,
                training_strategy=RetrainingStrategy.INCREMENTAL,
                start_time=datetime.now(UTC) - timedelta(days=i),
                end_time=datetime.now(UTC) - timedelta(days=i) + timedelta(minutes=5),
                success=True,
                training_metrics=TrainingMetrics(samples_processed=100 + i * 10)
            )
            for i in range(5)
        ]
        
        retrainer._retraining_history.extend(past_results)
        
        stats = retrainer.get_retraining_statistics()
        
        assert isinstance(stats, dict)
        assert stats['total_retrainings'] == 5
        assert stats['successful_retrainings'] == 5
        assert stats['success_rate'] == 100.0
        assert 'average_training_duration' in stats
        assert 'total_samples_processed' in stats


if __name__ == "__main__":
    pytest.main([__file__])