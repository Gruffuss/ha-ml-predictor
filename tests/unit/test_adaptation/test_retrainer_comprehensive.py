"""
Comprehensive test suite for AdaptiveRetrainer with incremental learning validation.

This test suite provides complete coverage of adaptive retraining functionality including:
- Retraining trigger evaluation and strategy selection
- Incremental learning algorithm validation
- Priority queue management and resource allocation
- Performance improvement tracking and validation
- Integration with ModelOptimizer and drift detection
- Concurrent retraining management and edge cases
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from src.adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftMetrics,
    DriftSeverity,
)
from src.adaptation.optimizer import ModelOptimizer, OptimizationResult
from src.adaptation.retrainer import (
    AdaptiveRetrainer,
    RetrainingError,
    RetrainingHistory,
    RetrainingProgress,
    RetrainingRequest,
    RetrainingStatus,
    RetrainingStrategy,
    RetrainingTrigger,
)
from src.adaptation.validator import AccuracyMetrics, PredictionValidator
from src.core.constants import ModelType
from src.core.exceptions import ErrorSeverity
from src.models.base.predictor import PredictionResult, TrainingResult


@dataclass
class MockTrackingConfig:
    """Mock tracking configuration for testing."""

    adaptive_retraining_enabled: bool = True
    retraining_accuracy_threshold: float = 75.0
    retraining_error_threshold: float = 20.0
    retraining_drift_threshold: float = 0.3
    retraining_lookback_days: int = 14
    retraining_validation_split: float = 0.2
    auto_feature_refresh: bool = True
    retraining_cooldown_hours: int = 12
    max_concurrent_retrains: int = 2
    incremental_retraining_threshold: float = 80.0
    retraining_check_interval_hours: int = 6


@pytest.fixture
def mock_tracking_config():
    """Create mock tracking configuration."""
    return MockTrackingConfig()


@pytest.fixture
def mock_model_registry():
    """Create mock model registry."""
    mock_model = MagicMock()
    mock_model.train = AsyncMock(
        return_value=TrainingResult(
            success=True,
            training_time_seconds=120.0,
            model_version="v2.1",
            training_samples=1000,
            training_score=0.85,
            training_metrics={"accuracy": 0.85, "loss": 0.15},
        )
    )
    mock_model.incremental_update = AsyncMock(
        return_value=TrainingResult(
            success=True,
            training_time_seconds=30.0,
            model_version="v2.1.1",
            training_samples=200,
            training_score=0.87,
            training_metrics={"accuracy": 0.87, "loss": 0.13},
        )
    )

    return {"room_1_lstm": mock_model, "room_2_xgboost": mock_model}


@pytest.fixture
def mock_model_optimizer():
    """Create mock model optimizer."""
    optimizer = MagicMock(spec=ModelOptimizer)
    optimizer.optimize_model_parameters = AsyncMock(
        return_value=OptimizationResult(
            success=True,
            best_parameters={"n_estimators": 200, "learning_rate": 0.01},
            best_score=0.88,
            improvement_over_default=0.05,
            trials_completed=25,
            optimization_time_seconds=300.0,
        )
    )
    return optimizer


@pytest.fixture
def mock_drift_detector():
    """Create mock drift detector."""
    detector = MagicMock(spec=ConceptDriftDetector)
    detector.classify_drift_severity = AsyncMock(return_value=DriftSeverity.MEDIUM)
    return detector


@pytest.fixture
def mock_prediction_validator():
    """Create mock prediction validator."""
    validator = MagicMock(spec=PredictionValidator)

    # Mock accuracy metrics
    mock_metrics = MagicMock(spec=AccuracyMetrics)
    mock_metrics.accuracy_rate = 70.0  # Below threshold
    mock_metrics.mean_error_minutes = 25.0  # Above threshold
    mock_metrics.confidence_calibration_score = 0.7

    validator.get_accuracy_metrics = AsyncMock(return_value=mock_metrics)
    validator.validate_model_predictions = AsyncMock(
        return_value={
            "accuracy_rate": 82.0,
            "mean_error_minutes": 12.0,
            "test_predictions": [
                {
                    "prediction_time": datetime.now(),
                    "predicted_occupied_time": datetime.now() + timedelta(minutes=30),
                    "confidence": 0.85,
                    "features_used": ["feature_1", "feature_2"],
                    "metadata": {"model_version": "v2.1"},
                }
            ],
        }
    )

    return validator


@pytest.fixture
def adaptive_retrainer(
    mock_tracking_config,
    mock_model_registry,
    mock_model_optimizer,
    mock_drift_detector,
    mock_prediction_validator,
):
    """Create AdaptiveRetrainer instance for testing."""
    return AdaptiveRetrainer(
        tracking_config=mock_tracking_config,
        model_registry=mock_model_registry,
        model_optimizer=mock_model_optimizer,
        drift_detector=mock_drift_detector,
        prediction_validator=mock_prediction_validator,
    )


@pytest.fixture
def sample_accuracy_metrics():
    """Create sample accuracy metrics for testing."""
    return AccuracyMetrics(
        room_id="test_room",
        model_type=ModelType.LSTM,
        accuracy_rate=70.0,  # Below threshold
        mean_error_minutes=25.0,  # Above threshold
        confidence_calibration_score=0.6,
        total_predictions=500,
        correct_predictions=350,
        start_time=datetime.now() - timedelta(days=7),
        end_time=datetime.now(),
    )


@pytest.fixture
def sample_drift_metrics():
    """Create sample drift metrics for testing."""
    return DriftMetrics(
        room_id="test_room",
        detection_time=datetime.now(),
        baseline_period=(
            datetime.now() - timedelta(days=30),
            datetime.now() - timedelta(days=7),
        ),
        current_period=(datetime.now() - timedelta(days=7), datetime.now()),
        overall_drift_score=0.6,  # Above threshold
        drift_severity=DriftSeverity.MEDIUM,
        accuracy_degradation=15.0,
    )


class TestAdaptiveRetrainer:
    """Comprehensive tests for AdaptiveRetrainer."""

    def test_initialization(self, adaptive_retrainer, mock_tracking_config):
        """Test proper initialization of adaptive retrainer."""
        assert adaptive_retrainer.config == mock_tracking_config
        assert len(adaptive_retrainer.model_registry) > 0
        assert adaptive_retrainer.model_optimizer is not None
        assert adaptive_retrainer.drift_detector is not None
        assert adaptive_retrainer.prediction_validator is not None

        # Test internal state initialization
        assert isinstance(adaptive_retrainer._retraining_queue, list)
        assert isinstance(adaptive_retrainer._active_retrainings, dict)
        assert isinstance(adaptive_retrainer._progress_tracker, dict)
        assert adaptive_retrainer._active_retraining_count == 0
        assert adaptive_retrainer._total_requests == 0

    def test_initialization_with_disabled_retraining(self):
        """Test initialization with retraining disabled."""
        disabled_config = MockTrackingConfig(adaptive_retraining_enabled=False)
        retrainer = AdaptiveRetrainer(tracking_config=disabled_config)

        assert not retrainer.config.adaptive_retraining_enabled
        assert retrainer.model_registry == {}

    @pytest.mark.asyncio
    async def test_initialize_and_shutdown(self, adaptive_retrainer):
        """Test initialization and shutdown of background tasks."""
        # Initialize
        await adaptive_retrainer.initialize()
        assert len(adaptive_retrainer._background_tasks) > 0

        # Shutdown
        await adaptive_retrainer.shutdown()
        assert len(adaptive_retrainer._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_initialize_with_disabled_config(self):
        """Test initialization with disabled configuration."""
        disabled_config = MockTrackingConfig(adaptive_retraining_enabled=False)
        retrainer = AdaptiveRetrainer(tracking_config=disabled_config)

        await retrainer.initialize()

        # Should not start background tasks when disabled
        assert len(retrainer._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_accuracy_trigger(
        self, adaptive_retrainer, sample_accuracy_metrics
    ):
        """Test retraining need evaluation with accuracy degradation trigger."""
        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=sample_accuracy_metrics,
        )

        assert request is not None
        assert isinstance(request, RetrainingRequest)
        assert RetrainingTrigger.ACCURACY_DEGRADATION == request.trigger
        assert request.room_id == "test_room"
        assert request.model_type == ModelType.LSTM
        assert request.priority > 0

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_error_trigger(self, adaptive_retrainer):
        """Test retraining need evaluation with error threshold trigger."""
        # Create metrics with high error
        high_error_metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=80.0,  # Above threshold
            mean_error_minutes=35.0,  # Well above threshold
            confidence_calibration_score=0.7,
            total_predictions=100,
            correct_predictions=80,
        )

        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=high_error_metrics,
        )

        assert request is not None
        assert RetrainingTrigger.ERROR_THRESHOLD_EXCEEDED == request.trigger
        assert request.priority > 0

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_drift_trigger(
        self, adaptive_retrainer, sample_drift_metrics
    ):
        """Test retraining need evaluation with drift trigger."""
        # Create good accuracy metrics but with drift
        good_metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=85.0,  # Above threshold
            mean_error_minutes=10.0,  # Below threshold
            confidence_calibration_score=0.8,
            total_predictions=100,
            correct_predictions=85,
        )

        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=good_metrics,
            drift_metrics=sample_drift_metrics,
        )

        assert request is not None
        assert RetrainingTrigger.CONCEPT_DRIFT == request.trigger
        assert request.drift_metrics == sample_drift_metrics

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_confidence_trigger(
        self, adaptive_retrainer
    ):
        """Test retraining need evaluation with confidence calibration trigger."""
        # Create metrics with poor confidence calibration
        poor_confidence_metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=80.0,  # Above threshold
            mean_error_minutes=15.0,  # Below threshold
            confidence_calibration_score=0.2,  # Poor calibration
            total_predictions=100,
            correct_predictions=80,
        )

        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=poor_confidence_metrics,
        )

        assert request is not None
        assert RetrainingTrigger.PERFORMANCE_ANOMALY == request.trigger

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_no_triggers(self, adaptive_retrainer):
        """Test retraining need evaluation with no triggers."""
        # Create good metrics that don't trigger retraining
        good_metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=90.0,  # Well above threshold
            mean_error_minutes=10.0,  # Well below threshold
            confidence_calibration_score=0.8,  # Good calibration
            total_predictions=100,
            correct_predictions=90,
        )

        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=good_metrics,
        )

        assert request is None

    @pytest.mark.asyncio
    async def test_evaluate_retraining_need_cooldown(
        self, adaptive_retrainer, sample_accuracy_metrics
    ):
        """Test retraining need evaluation during cooldown period."""
        # Set recent retraining time
        model_key = "test_room_lstm"
        with adaptive_retrainer._cooldown_lock:
            adaptive_retrainer._last_retrain_times[model_key] = datetime.now(
                timezone.utc
            )

        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=sample_accuracy_metrics,
        )

        # Should skip due to cooldown
        assert request is None

    def test_select_retraining_strategy_incremental(self, adaptive_retrainer):
        """Test retraining strategy selection for incremental updates."""
        # Create metrics that would trigger incremental retraining
        decent_metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=82.0,  # Above incremental threshold
            mean_error_minutes=15.0,
            confidence_calibration_score=0.7,
            total_predictions=100,
            correct_predictions=82,
        )

        strategy = adaptive_retrainer._select_retraining_strategy(decent_metrics)
        assert strategy == RetrainingStrategy.INCREMENTAL

    def test_select_retraining_strategy_full_retrain(
        self, adaptive_retrainer, sample_drift_metrics
    ):
        """Test retraining strategy selection for full retraining."""
        poor_metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=65.0,  # Below incremental threshold
            mean_error_minutes=30.0,
            confidence_calibration_score=0.5,
            total_predictions=100,
            correct_predictions=65,
        )

        # With significant drift
        high_drift = DriftMetrics(
            room_id="test_room",
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=30),
                datetime.now() - timedelta(days=7),
            ),
            current_period=(datetime.now() - timedelta(days=7), datetime.now()),
            overall_drift_score=0.8,  # High drift
        )

        strategy = adaptive_retrainer._select_retraining_strategy(
            poor_metrics, high_drift
        )
        assert strategy == RetrainingStrategy.FULL_RETRAIN

    def test_select_retraining_strategy_ensemble_rebalance(self, adaptive_retrainer):
        """Test retraining strategy selection for ensemble rebalancing."""
        # Create metrics with poor confidence calibration but decent accuracy
        metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=82.0,  # Above incremental threshold
            mean_error_minutes=15.0,
            confidence_calibration_score=0.15,  # Very poor calibration
            total_predictions=100,
            correct_predictions=82,
        )

        strategy = adaptive_retrainer._select_retraining_strategy(metrics)
        assert strategy == RetrainingStrategy.ENSEMBLE_REBALANCE

    @pytest.mark.asyncio
    async def test_request_manual_retraining(self, adaptive_retrainer):
        """Test manual retraining request."""
        request_id = await adaptive_retrainer.request_retraining(
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            priority=8.0,
            lookback_days=21,
        )

        assert isinstance(request_id, str)
        assert "test_room" in request_id
        assert "manual" in request_id
        assert adaptive_retrainer._total_requests == 1

        # Check that request was queued
        assert len(adaptive_retrainer._retraining_queue) == 1
        request = adaptive_retrainer._retraining_queue[0]
        assert request.room_id == "test_room"
        assert request.model_type == ModelType.LSTM
        assert request.trigger == RetrainingTrigger.MANUAL_REQUEST
        assert request.priority == 8.0
        assert request.lookback_days == 21

    @pytest.mark.asyncio
    async def test_get_retraining_status_specific_request(self, adaptive_retrainer):
        """Test getting status for specific retraining request."""
        # Request manual retraining
        request_id = await adaptive_retrainer.request_retraining(
            room_id="test_room", model_type=ModelType.LSTM
        )

        status = await adaptive_retrainer.get_retraining_status(request_id)

        assert isinstance(status, dict)
        assert status["request_id"] == request_id
        assert status["room_id"] == "test_room"
        assert status["status"] == RetrainingStatus.PENDING.value

    @pytest.mark.asyncio
    async def test_get_retraining_status_all_requests(self, adaptive_retrainer):
        """Test getting status for all retraining requests."""
        # Create multiple requests
        request_ids = []
        for i in range(3):
            request_id = await adaptive_retrainer.request_retraining(
                room_id=f"room_{i}", model_type=ModelType.LSTM
            )
            request_ids.append(request_id)

        all_status = await adaptive_retrainer.get_retraining_status()

        assert isinstance(all_status, list)
        assert len(all_status) == 3
        for status in all_status:
            assert "request_id" in status
            assert "room_id" in status
            assert "status" in status

    @pytest.mark.asyncio
    async def test_cancel_pending_retraining(self, adaptive_retrainer):
        """Test canceling pending retraining request."""
        # Request retraining
        request_id = await adaptive_retrainer.request_retraining(
            room_id="test_room", model_type=ModelType.LSTM
        )

        # Cancel it
        result = await adaptive_retrainer.cancel_retraining(request_id)

        assert result is True
        assert len(adaptive_retrainer._retraining_queue) == 0
        assert len(adaptive_retrainer._retraining_history) == 1

        # Check that it's marked as cancelled in history
        cancelled_request = adaptive_retrainer._retraining_history[0]
        assert cancelled_request.status == RetrainingStatus.CANCELLED

    def test_get_retrainer_stats(self, adaptive_retrainer):
        """Test getting comprehensive retrainer statistics."""
        # Add some test data
        adaptive_retrainer._total_retrainings_completed = 10
        adaptive_retrainer._total_retrainings_failed = 2
        adaptive_retrainer._average_retraining_time = 180.0

        stats = adaptive_retrainer.get_retrainer_stats()

        assert isinstance(stats, dict)
        assert "enabled" in stats
        assert "queue_size" in stats
        assert "active_retrainings" in stats
        assert "performance_stats" in stats
        assert "configuration" in stats

        # Verify performance stats
        assert stats["performance_stats"]["total_completed"] == 10
        assert stats["performance_stats"]["total_failed"] == 2
        assert stats["performance_stats"]["success_rate_percent"] == 10 / 12 * 100

    @pytest.mark.asyncio
    async def test_queue_retraining_request_priority_ordering(self, adaptive_retrainer):
        """Test that retraining requests are queued in priority order."""
        # Create requests with different priorities
        low_priority_request = RetrainingRequest(
            request_id="low_priority",
            room_id="room1",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=3.0,
            created_time=datetime.now(timezone.utc),
        )

        high_priority_request = RetrainingRequest(
            request_id="high_priority",
            room_id="room2",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.CONCEPT_DRIFT,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=8.0,
            created_time=datetime.now(timezone.utc),
        )

        # Queue them in reverse priority order
        await adaptive_retrainer._queue_retraining_request(low_priority_request)
        await adaptive_retrainer._queue_retraining_request(high_priority_request)

        # Queue should be ordered by priority (highest first)
        assert len(adaptive_retrainer._retraining_queue) == 2
        assert adaptive_retrainer._retraining_queue[0].request_id == "high_priority"
        assert adaptive_retrainer._retraining_queue[1].request_id == "low_priority"

    @pytest.mark.asyncio
    async def test_queue_retraining_request_duplicate_handling(
        self, adaptive_retrainer
    ):
        """Test handling of duplicate retraining requests."""
        # Create two requests for same room/model
        first_request = RetrainingRequest(
            request_id="first",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.ACCURACY_DEGRADATION,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        second_request = RetrainingRequest(
            request_id="second",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.CONCEPT_DRIFT,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=8.0,  # Higher priority
            created_time=datetime.now(timezone.utc),
        )

        await adaptive_retrainer._queue_retraining_request(first_request)
        await adaptive_retrainer._queue_retraining_request(second_request)

        # Should only have one request (higher priority one)
        assert len(adaptive_retrainer._retraining_queue) == 1
        assert adaptive_retrainer._retraining_queue[0].request_id == "second"
        assert adaptive_retrainer._retraining_queue[0].priority == 8.0


class TestRetrainingExecution:
    """Tests for retraining execution and model training."""

    @pytest.mark.asyncio
    async def test_incremental_retrain_success(
        self, adaptive_retrainer, mock_model_registry
    ):
        """Test successful incremental retraining."""
        model = mock_model_registry["room_1_lstm"]

        # Create sample training data
        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        result = await adaptive_retrainer._incremental_retrain(model, features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success
        assert result.training_score == 0.87
        model.incremental_update.assert_called_once_with(features, targets)

    @pytest.mark.asyncio
    async def test_incremental_retrain_fallback_to_full(self, adaptive_retrainer):
        """Test incremental retraining fallback to full training."""
        # Create model without incremental_update method
        model = MagicMock()
        model.train = AsyncMock(
            return_value=TrainingResult(
                success=True,
                training_time_seconds=120.0,
                model_version="v2.1",
                training_samples=100,
                training_score=0.85,
            )
        )
        # Don't add incremental_update method

        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        result = await adaptive_retrainer._incremental_retrain(model, features, targets)

        assert result.success
        model.train.assert_called_once_with(features, targets)

    @pytest.mark.asyncio
    async def test_full_retrain_with_optimization(
        self, adaptive_retrainer, mock_model_registry
    ):
        """Test full retraining with parameter optimization."""
        model = mock_model_registry["room_1_lstm"]

        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))
        val_features = pd.DataFrame(np.random.randn(20, 5))
        val_targets = pd.DataFrame(np.random.randint(0, 2, (20, 1)))

        result = await adaptive_retrainer._full_retrain_with_optimization(
            model, features, targets, val_features, val_targets
        )

        assert isinstance(result, TrainingResult)
        assert result.success
        model.train.assert_called_once_with(
            features, targets, val_features, val_targets
        )

    @pytest.mark.asyncio
    async def test_ensemble_rebalance(self, adaptive_retrainer):
        """Test ensemble weight rebalancing."""
        # Create model with ensemble capabilities
        model = MagicMock()
        model._calculate_model_weights = MagicMock()
        model._prepare_targets = MagicMock(return_value=np.array([0, 1, 0, 1]))
        model.model_version = "v2.0"

        features = pd.DataFrame(np.random.randn(4, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (4, 1)))

        result = await adaptive_retrainer._ensemble_rebalance(model, features, targets)

        assert isinstance(result, TrainingResult)
        assert result.success
        model._calculate_model_weights.assert_called_once()

    @pytest.mark.asyncio
    async def test_ensemble_rebalance_fallback(self, adaptive_retrainer):
        """Test ensemble rebalancing fallback to full training."""
        # Create model without ensemble capabilities
        model = MagicMock()
        model.train = AsyncMock(
            return_value=TrainingResult(
                success=True,
                training_time_seconds=120.0,
                model_version="v2.1",
                training_samples=100,
                training_score=0.85,
            )
        )
        # Don't add _calculate_model_weights method

        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        result = await adaptive_retrainer._ensemble_rebalance(model, features, targets)

        assert result.success
        model.train.assert_called_once_with(features, targets)

    @pytest.mark.asyncio
    async def test_prepare_retraining_data(self, adaptive_retrainer):
        """Test data preparation for retraining."""
        request = RetrainingRequest(
            request_id="test",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
            lookback_days=14,
            validation_split=0.2,
        )

        train_data, val_data = await adaptive_retrainer._prepare_retraining_data(
            request
        )

        # Should return DataFrames (even if empty in mock)
        assert isinstance(train_data, pd.DataFrame)
        assert val_data is None or isinstance(val_data, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_extract_features_for_retraining(self, adaptive_retrainer):
        """Test feature extraction for retraining."""
        data = pd.DataFrame(np.random.randn(100, 5))
        request = RetrainingRequest(
            request_id="test",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        features, targets = await adaptive_retrainer._extract_features_for_retraining(
            data, request
        )

        assert isinstance(features, pd.DataFrame)
        assert isinstance(targets, pd.DataFrame)

    @pytest.mark.asyncio
    async def test_retrain_model_with_optimization(
        self, adaptive_retrainer, mock_model_registry
    ):
        """Test model retraining with optimization."""
        # Create request with optimization context
        request = RetrainingRequest(
            request_id="test_opt",
            room_id="room_1",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.ACCURACY_DEGRADATION,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
            accuracy_metrics=AccuracyMetrics(
                room_id="room_1",
                model_type=ModelType.LSTM,
                accuracy_rate=70.0,
                mean_error_minutes=25.0,
                total_predictions=100,
                correct_predictions=70,
            ),
        )

        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        # Should use optimizer when available
        result = await adaptive_retrainer._retrain_model(
            request, features, targets, None, None
        )

        assert isinstance(result, TrainingResult)
        assert result.success

        # Verify optimizer was called
        adaptive_retrainer.model_optimizer.optimize_model_parameters.assert_called_once()

    @pytest.mark.asyncio
    async def test_retrain_model_missing_from_registry(self, adaptive_retrainer):
        """Test retraining when model is missing from registry."""
        request = RetrainingRequest(
            request_id="test_missing",
            room_id="missing_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        features = pd.DataFrame(np.random.randn(10, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (10, 1)))

        with pytest.raises(RetrainingError) as exc_info:
            await adaptive_retrainer._retrain_model(
                request, features, targets, None, None
            )

        assert "not found in registry" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_retrain_model_insufficient_data(
        self, adaptive_retrainer, mock_model_registry
    ):
        """Test retraining with insufficient data."""
        request = RetrainingRequest(
            request_id="test_insufficient",
            room_id="room_1",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        # Empty features and targets
        features = pd.DataFrame()
        targets = pd.DataFrame()

        with pytest.raises(RetrainingError) as exc_info:
            await adaptive_retrainer._retrain_model(
                request, features, targets, None, None
            )

        assert "Insufficient training data" in str(exc_info.value)


class TestRetrainingProgress:
    """Tests for retraining progress tracking."""

    @pytest.mark.asyncio
    async def test_progress_tracking_during_retraining(self, adaptive_retrainer):
        """Test progress tracking throughout retraining process."""
        request = RetrainingRequest(
            request_id="progress_test",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        progress = RetrainingProgress(
            request_id="progress_test", room_id="test_room", model_type=ModelType.LSTM
        )

        # Test progress updates
        progress.update_progress("data_preparation", "Loading data", 10.0)
        assert progress.phase == "data_preparation"
        assert progress.current_step == "Loading data"
        assert progress.progress_percentage == 10.0

        progress.update_progress("training", "Training model", 75.0)
        assert progress.phase == "training"
        assert progress.progress_percentage == 75.0

    @pytest.mark.asyncio
    async def test_get_retraining_progress(self, adaptive_retrainer):
        """Test getting retraining progress."""
        # Start a retraining request
        request_id = await adaptive_retrainer.request_retraining(
            room_id="test_room", model_type=ModelType.LSTM
        )

        # Add progress manually for testing
        progress = RetrainingProgress(
            request_id=request_id, room_id="test_room", model_type=ModelType.LSTM
        )
        progress.update_progress("training", "Model training in progress", 60.0)

        with adaptive_retrainer._progress_lock:
            adaptive_retrainer._progress_tracker[request_id] = progress

        # Get progress
        progress_info = await adaptive_retrainer.get_retraining_progress(request_id)

        assert isinstance(progress_info, dict)
        assert "progress_percentage" in progress_info
        assert progress_info["progress_percentage"] == 60.0
        assert progress_info["phase"] == "training"

    @pytest.mark.asyncio
    async def test_execute_retraining_with_progress_tracking(
        self, adaptive_retrainer, mock_model_registry
    ):
        """Test complete retraining execution with progress tracking."""

        # Mock the perform_retraining method to avoid database dependencies
        async def mock_perform_retraining(request, progress):
            # Simulate progress updates
            progress.update_progress("data_preparation", "Loading data", 25.0)
            progress.update_progress("training", "Training model", 75.0)
            progress.update_progress("validation", "Validating model", 100.0)

            # Set successful result
            request.training_result = TrainingResult(
                success=True,
                training_time_seconds=120.0,
                model_version="v2.1",
                training_samples=100,
                training_score=0.85,
            )

        adaptive_retrainer._perform_retraining = mock_perform_retraining

        request = RetrainingRequest(
            request_id="exec_test",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        result_request = await adaptive_retrainer._execute_retraining(request)

        assert result_request.status == RetrainingStatus.COMPLETED
        assert result_request.training_result is not None
        assert result_request.training_result.success
        assert adaptive_retrainer._total_retrainings_completed == 1


class TestRetrainingHistory:
    """Tests for RetrainingHistory functionality."""

    def test_retraining_history_initialization(self):
        """Test initialization of RetrainingHistory."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        assert history.room_id == "test_room"
        assert history.model_type == ModelType.LSTM
        assert history.total_retrainings == 0
        assert history.successful_retrainings == 0
        assert history.failed_retrainings_count == 0
        assert isinstance(history.trigger_frequency, dict)

    def test_add_successful_retraining_record(self):
        """Test adding successful retraining record to history."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        # Create successful request
        request = RetrainingRequest(
            request_id="test_success",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.ACCURACY_DEGRADATION,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
            status=RetrainingStatus.COMPLETED,
            completed_time=datetime.now(timezone.utc),
            training_result=TrainingResult(
                success=True,
                training_time_seconds=120.0,
                model_version="v2.1",
                training_samples=100,
                training_score=0.85,
            ),
        )
        request.training_result.training_score = 0.85

        history.add_retraining_record(request)

        assert history.total_retrainings == 1
        assert history.successful_retrainings == 1
        assert history.failed_retrainings_count == 0
        assert history.best_achieved_accuracy == 0.85
        assert (
            history.trigger_frequency[RetrainingTrigger.ACCURACY_DEGRADATION.value] == 1
        )
        assert history.most_common_trigger == RetrainingTrigger.ACCURACY_DEGRADATION

    def test_add_failed_retraining_record(self):
        """Test adding failed retraining record to history."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        # Create failed request
        request = RetrainingRequest(
            request_id="test_failure",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.CONCEPT_DRIFT,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
            status=RetrainingStatus.FAILED,
            completed_time=datetime.now(timezone.utc),
            error_message="Training failed due to insufficient data",
        )

        history.add_retraining_record(request)

        assert history.total_retrainings == 1
        assert history.successful_retrainings == 0
        assert history.failed_retrainings_count == 1
        assert history.trigger_frequency[RetrainingTrigger.CONCEPT_DRIFT.value] == 1

    def test_analyze_accuracy_trend(self):
        """Test accuracy trend analysis."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        # Add multiple records with improving accuracy
        scores = [0.70, 0.75, 0.80, 0.85, 0.88]
        for i, score in enumerate(scores):
            request = RetrainingRequest(
                request_id=f"test_{i}",
                room_id="test_room",
                model_type=ModelType.LSTM,
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                strategy=RetrainingStrategy.FULL_RETRAIN,
                priority=5.0,
                created_time=datetime.now(timezone.utc),
                status=RetrainingStatus.COMPLETED,
                training_result=TrainingResult(
                    success=True,
                    training_time_seconds=120.0,
                    model_version=f"v2.{i}",
                    training_samples=100,
                    training_score=score,
                ),
            )
            request.training_result.training_score = score
            history.add_retraining_record(request)

        # Should detect improving trend
        assert history.accuracy_trend_direction == "improving"

    def test_get_success_rate(self):
        """Test success rate calculation."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        # Add mix of successful and failed retrainings
        history.successful_retrainings = 7
        history.failed_retrainings_count = 3
        history.total_retrainings = 10

        success_rate = history.get_success_rate()
        assert success_rate == 70.0

    def test_get_recent_performance(self):
        """Test recent performance statistics."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        # Add recent performance data
        recent_time = datetime.now(timezone.utc)
        history.performance_timeline = [
            {
                "timestamp": recent_time - timedelta(hours=2),
                "accuracy": 0.80,
                "training_time": 120.0,
                "trigger": "manual",
                "strategy": "full_retrain",
            },
            {
                "timestamp": recent_time - timedelta(hours=1),
                "accuracy": 0.85,
                "training_time": 110.0,
                "trigger": "accuracy_degradation",
                "strategy": "incremental",
            },
        ]

        performance = history.get_recent_performance(days=1)

        assert performance["retrainings"] == 2
        assert performance["average_accuracy"] == 0.825
        assert performance["min_accuracy"] == 0.80
        assert performance["max_accuracy"] == 0.85

    def test_to_dict_serialization(self):
        """Test RetrainingHistory serialization to dictionary."""
        history = RetrainingHistory(room_id="test_room", model_type=ModelType.LSTM)

        # Set some test data
        history.total_retrainings = 5
        history.successful_retrainings = 4
        history.failed_retrainings_count = 1
        history.best_achieved_accuracy = 0.90
        history.trigger_frequency = {"manual": 3, "accuracy_degradation": 2}

        result_dict = history.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["room_id"] == "test_room"
        assert result_dict["total_retrainings"] == 5
        assert result_dict["successful_retrainings"] == 4
        assert result_dict["success_rate_percent"] == 80.0
        assert result_dict["best_achieved_accuracy"] == 0.90


class TestIntegrationAndValidation:
    """Tests for integration with other components and validation."""

    @pytest.mark.asyncio
    async def test_drift_detector_integration(
        self, adaptive_retrainer, mock_drift_detector, sample_drift_metrics
    ):
        """Test integration with drift detector."""
        # Configure mock to return specific severity
        mock_drift_detector.classify_drift_severity.return_value = DriftSeverity.HIGH

        metrics = AccuracyMetrics(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_rate=85.0,  # Good accuracy
            mean_error_minutes=10.0,  # Good error
            confidence_calibration_score=0.8,
            total_predictions=100,
            correct_predictions=85,
        )

        # Should still trigger retraining due to drift
        request = await adaptive_retrainer.evaluate_retraining_need(
            room_id="test_room",
            model_type=ModelType.LSTM,
            accuracy_metrics=metrics,
            drift_metrics=sample_drift_metrics,
        )

        assert request is not None
        assert RetrainingTrigger.CONCEPT_DRIFT == request.trigger
        # Priority should be boosted due to high drift severity
        assert request.priority > 5.0  # Base drift priority + severity boost

    @pytest.mark.asyncio
    async def test_prediction_validator_integration(
        self, adaptive_retrainer, mock_prediction_validator
    ):
        """Test integration with prediction validator."""
        request = RetrainingRequest(
            request_id="validation_test",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        training_result = TrainingResult(
            success=True,
            training_time_seconds=120.0,
            model_version="v2.1",
            training_samples=100,
            training_score=0.85,
        )

        # Test validation call
        validation_results = await adaptive_retrainer._validate_retraining_predictions(
            request, training_result
        )

        assert validation_results is not None
        assert "accuracy_rate" in validation_results
        assert "mean_error_minutes" in validation_results

        # Verify that test predictions were stored in request
        assert request.prediction_results is not None
        assert len(request.prediction_results) > 0
        assert isinstance(request.prediction_results[0], PredictionResult)

    def test_model_optimizer_integration_status(
        self, adaptive_retrainer, mock_model_optimizer
    ):
        """Test model optimizer integration status."""
        # Test when optimizer is available
        assert adaptive_retrainer.model_optimizer is not None

        # Test optimizer configuration access
        stats = adaptive_retrainer.get_retrainer_stats()
        assert "configuration" in stats

    def test_drift_detector_status(self, adaptive_retrainer, mock_drift_detector):
        """Test drift detector status reporting."""
        status = adaptive_retrainer.get_drift_detector_status()

        assert isinstance(status, dict)
        assert "available" in status
        assert status["available"] is True
        assert status["status"] == "active"

    def test_prediction_validator_status(
        self, adaptive_retrainer, mock_prediction_validator
    ):
        """Test prediction validator status reporting."""
        status = adaptive_retrainer.get_prediction_validator_status()

        assert isinstance(status, dict)
        assert "available" in status
        assert status["available"] is True
        assert status["status"] == "active"


class TestConcurrencyAndResourceManagement:
    """Tests for concurrent operations and resource management."""

    @pytest.mark.asyncio
    async def test_concurrent_request_limit(self, adaptive_retrainer):
        """Test enforcement of concurrent retraining limits."""
        # Set low limit for testing
        adaptive_retrainer.config.max_concurrent_retrains = 1

        # Simulate one active retraining
        adaptive_retrainer._active_retraining_count = 1

        # Should not allow more retrainings
        can_start = adaptive_retrainer._can_start_retraining()
        assert not can_start

        # Reduce count, should allow retraining
        adaptive_retrainer._active_retraining_count = 0
        can_start = adaptive_retrainer._can_start_retraining()
        assert can_start

    @pytest.mark.asyncio
    async def test_multiple_concurrent_requests(self, adaptive_retrainer):
        """Test handling multiple concurrent retraining requests."""
        # Create multiple requests quickly
        tasks = []
        for i in range(5):
            task = adaptive_retrainer.request_retraining(
                room_id=f"room_{i}", model_type=ModelType.LSTM, priority=float(i)
            )
            tasks.append(task)

        # Execute concurrently
        request_ids = await asyncio.gather(*tasks)

        assert len(request_ids) == 5
        assert len(adaptive_retrainer._retraining_queue) == 5

        # Check that they're properly ordered by priority
        priorities = [req.priority for req in adaptive_retrainer._retraining_queue]
        assert priorities == sorted(priorities, reverse=True)

    def test_resource_tracking_accuracy(self, adaptive_retrainer):
        """Test accuracy of resource usage tracking."""
        # Initial state
        assert adaptive_retrainer._active_retraining_count == 0

        # Simulate resource allocation
        with adaptive_retrainer._resource_lock:
            adaptive_retrainer._active_retraining_count += 1

        stats = adaptive_retrainer.get_retrainer_stats()
        utilization = stats["resource_utilization_percent"]
        expected_utilization = (
            1 / adaptive_retrainer.config.max_concurrent_retrains
        ) * 100

        assert utilization == expected_utilization

    def test_cooldown_tracking_thread_safety(self, adaptive_retrainer):
        """Test thread safety of cooldown tracking."""
        import threading

        room_ids = [f"room_{i}" for i in range(10)]

        def set_cooldown(room_id):
            model_key = f"{room_id}_lstm"
            with adaptive_retrainer._cooldown_lock:
                adaptive_retrainer._last_retrain_times[model_key] = datetime.now(
                    timezone.utc
                )

        # Run multiple threads setting cooldowns
        threads = []
        for room_id in room_ids:
            thread = threading.Thread(target=set_cooldown, args=(room_id,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        # All cooldowns should be set
        assert len(adaptive_retrainer._last_retrain_times) == 10


class TestErrorHandlingAndEdgeCases:
    """Tests for error handling and edge case scenarios."""

    @pytest.mark.asyncio
    async def test_retraining_execution_failure(self, adaptive_retrainer):
        """Test handling of retraining execution failures."""

        # Mock perform_retraining to raise an exception
        async def failing_perform_retraining(request, progress):
            raise Exception("Training failed due to hardware error")

        adaptive_retrainer._perform_retraining = failing_perform_retraining

        request = RetrainingRequest(
            request_id="failure_test",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        result_request = await adaptive_retrainer._execute_retraining(request)

        assert result_request.status == RetrainingStatus.FAILED
        assert "hardware error" in result_request.error_message
        assert adaptive_retrainer._total_retrainings_failed == 1

    @pytest.mark.asyncio
    async def test_model_training_failure(
        self, adaptive_retrainer, mock_model_registry
    ):
        """Test handling of model training failures."""
        # Configure model to fail training
        model = mock_model_registry["room_1_lstm"]
        model.train = AsyncMock(
            return_value=TrainingResult(
                success=False,
                training_time_seconds=0.0,
                model_version="",
                training_samples=0,
                training_score=0.0,
                error_message="Model training failed",
            )
        )

        request = RetrainingRequest(
            request_id="model_failure",
            room_id="room_1",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        with pytest.raises(RetrainingError) as exc_info:
            await adaptive_retrainer._full_retrain_with_optimization(
                model, features, targets, None, None
            )

        assert "Model training failed" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_validation_failure_handling(self, adaptive_retrainer):
        """Test handling of validation failures."""
        request = RetrainingRequest(
            request_id="validation_failure",
            room_id="test_room",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        training_result = TrainingResult(
            success=True,
            training_time_seconds=120.0,
            model_version="v2.1",
            training_samples=100,
            training_score=0.85,
        )

        # Mock validation to return poor results
        validation_results = {
            "accuracy_rate": 50.0,  # Below threshold
            "mean_error_minutes": 40.0,  # Above threshold
        }

        with pytest.raises(RetrainingError) as exc_info:
            await adaptive_retrainer._validate_and_deploy_retrained_model(
                request, training_result, validation_results
            )

        assert "failed validation checks" in str(exc_info.value)

    def test_malformed_retraining_request(self, adaptive_retrainer):
        """Test handling of malformed retraining requests."""
        # Test with None values
        invalid_request = RetrainingRequest(
            request_id="invalid",
            room_id="",  # Empty room ID
            model_type=None,  # None model type
            trigger=RetrainingTrigger.MANUAL_REQUEST,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        # Should handle gracefully without crashing
        result_dict = invalid_request.to_dict()
        assert isinstance(result_dict, dict)
        assert result_dict["room_id"] == ""

    def test_invalid_model_type_handling(self, adaptive_retrainer):
        """Test handling of invalid model types."""

        # Test with string model type (legacy compatibility)
        async def test_string_model_type():
            return await adaptive_retrainer.request_retraining(
                room_id="test_room",
                model_type="invalid_type",  # String instead of enum
                trigger=RetrainingTrigger.MANUAL_REQUEST,
            )

        # Should not crash but handle string conversion
        request_id = asyncio.run(test_string_model_type())
        assert isinstance(request_id, str)

    @pytest.mark.asyncio
    async def test_optimizer_failure_graceful_handling(
        self, adaptive_retrainer, mock_model_optimizer
    ):
        """Test graceful handling of optimizer failures."""
        # Configure optimizer to fail
        mock_model_optimizer.optimize_model_parameters = AsyncMock(
            side_effect=Exception("Optimization server unavailable")
        )

        request = RetrainingRequest(
            request_id="opt_failure",
            room_id="room_1",
            model_type=ModelType.LSTM,
            trigger=RetrainingTrigger.ACCURACY_DEGRADATION,
            strategy=RetrainingStrategy.FULL_RETRAIN,
            priority=5.0,
            created_time=datetime.now(timezone.utc),
        )

        features = pd.DataFrame(np.random.randn(100, 5))
        targets = pd.DataFrame(np.random.randint(0, 2, (100, 1)))

        # Should fall back to default parameters and complete successfully
        result = await adaptive_retrainer._retrain_model(
            request, features, targets, None, None
        )

        # Should still succeed with default parameters
        assert isinstance(result, TrainingResult)
        assert result.success

    def test_memory_cleanup_on_shutdown(self, adaptive_retrainer):
        """Test proper memory cleanup on shutdown."""
        # Add some test data
        adaptive_retrainer._retraining_queue.append(
            RetrainingRequest(
                request_id="cleanup_test",
                room_id="test_room",
                model_type=ModelType.LSTM,
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                strategy=RetrainingStrategy.FULL_RETRAIN,
                priority=5.0,
                created_time=datetime.now(timezone.utc),
            )
        )

        # Add some history
        adaptive_retrainer._retraining_history.append(
            RetrainingRequest(
                request_id="history_test",
                room_id="test_room",
                model_type=ModelType.LSTM,
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                strategy=RetrainingStrategy.FULL_RETRAIN,
                priority=5.0,
                created_time=datetime.now(timezone.utc),
                status=RetrainingStatus.COMPLETED,
            )
        )

        # Memory should be properly managed (collections should exist)
        assert len(adaptive_retrainer._retraining_queue) > 0
        assert len(adaptive_retrainer._retraining_history) > 0

        # Test graceful cleanup (should not raise exceptions)
        asyncio.run(adaptive_retrainer.shutdown())


class TestPerformanceAndScalability:
    """Tests for performance characteristics and scalability."""

    @pytest.mark.asyncio
    async def test_high_volume_request_handling(self, adaptive_retrainer):
        """Test handling of high volume retraining requests."""
        import time

        start_time = time.time()

        # Create many requests rapidly
        tasks = []
        for i in range(50):
            task = adaptive_retrainer.request_retraining(
                room_id=f"room_{i % 10}",  # Reuse room IDs to test deduplication
                model_type=ModelType.LSTM,
                priority=float(i % 10),
            )
            tasks.append(task)

        request_ids = await asyncio.gather(*tasks)
        end_time = time.time()

        # Should complete quickly
        assert (end_time - start_time) < 5.0  # 5 seconds max
        assert len(request_ids) == 50

        # Queue should have proper deduplication (max 10 unique room/model combos)
        assert len(adaptive_retrainer._retraining_queue) <= 10

    def test_memory_efficiency_large_history(self, adaptive_retrainer):
        """Test memory efficiency with large retraining history."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss

        # Add many history entries
        for i in range(1000):
            request = RetrainingRequest(
                request_id=f"history_{i}",
                room_id=f"room_{i % 10}",
                model_type=ModelType.LSTM,
                trigger=RetrainingTrigger.MANUAL_REQUEST,
                strategy=RetrainingStrategy.FULL_RETRAIN,
                priority=5.0,
                created_time=datetime.now(timezone.utc),
                status=RetrainingStatus.COMPLETED,
            )
            adaptive_retrainer._retraining_history.append(request)

        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory

        # Memory increase should be reasonable (< 50MB for 1000 requests)
        assert memory_increase < 50 * 1024 * 1024

        # History should be limited by maxlen
        assert len(adaptive_retrainer._retraining_history) == 1000

    @pytest.mark.asyncio
    async def test_concurrent_status_queries(self, adaptive_retrainer):
        """Test performance of concurrent status queries."""
        # Add test data
        for i in range(10):
            await adaptive_retrainer.request_retraining(
                room_id=f"room_{i}", model_type=ModelType.LSTM
            )

        # Run many concurrent status queries
        tasks = []
        for _ in range(20):
            task = adaptive_retrainer.get_retraining_status()
            tasks.append(task)

        start_time = asyncio.get_event_loop().time()
        results = await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Should complete quickly
        assert (end_time - start_time) < 2.0  # 2 seconds max
        assert len(results) == 20

        # All results should be identical
        for result in results:
            assert len(result) == 10  # 10 pending requests
