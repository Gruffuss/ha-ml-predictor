"""
Comprehensive tests for the training integration manager.

This module tests the automatic training pipeline integration with tracking system,
including accuracy-based retraining, drift detection, and background task management.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.core.exceptions import ModelTrainingError
from src.models.training_config import TrainingProfile
from src.models.training_integration import (
    TrainingIntegrationManager,
    integrate_training_with_tracking_manager,
)
from src.models.training_pipeline import TrainingType


@pytest.mark.unit
class TestTrainingIntegrationManager:
    """Test cases for TrainingIntegrationManager."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Mock tracking manager."""
        manager = AsyncMock()
        manager.add_accuracy_callback = MagicMock()
        manager.add_drift_callback = MagicMock()
        manager.add_performance_callback = MagicMock()
        manager.trigger_retraining = AsyncMock()
        manager.validate_model_performance = AsyncMock()
        manager.on_model_retrained = AsyncMock()
        manager.on_training_failure = AsyncMock()
        manager.force_prediction = AsyncMock()
        manager.register_model = MagicMock()
        return manager

    @pytest.fixture
    def mock_training_pipeline(self):
        """Mock training pipeline."""
        pipeline = AsyncMock()
        pipeline.run_retraining_pipeline = AsyncMock()
        pipeline.get_model_registry = MagicMock(
            return_value={"test_room_lstm": MagicMock()}
        )
        return pipeline

    @pytest.fixture
    def mock_config_manager(self):
        """Mock configuration manager."""
        config = MagicMock()
        config.get_environment_config = MagicMock()
        config.set_current_profile = MagicMock()

        # Mock environment config
        env_config = MagicMock()
        env_config.quality_thresholds.min_accuracy_threshold = 0.8
        env_config.quality_thresholds.max_error_threshold_minutes = 15.0
        config.get_environment_config.return_value = env_config

        return config

    @pytest.fixture
    def integration_manager(
        self, mock_tracking_manager, mock_training_pipeline, mock_config_manager
    ):
        """Create integration manager for testing."""
        return TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=mock_training_pipeline,
            config_manager=mock_config_manager,
        )

    def test_initialization(
        self, mock_tracking_manager, mock_training_pipeline, mock_config_manager
    ):
        """Test TrainingIntegrationManager initialization."""
        manager = TrainingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            training_pipeline=mock_training_pipeline,
            config_manager=mock_config_manager,
        )

        assert manager.tracking_manager is mock_tracking_manager
        assert manager.training_pipeline is mock_training_pipeline
        assert manager.config_manager is mock_config_manager
        assert manager._integration_active is False
        assert len(manager._active_training_requests) == 0
        assert len(manager._training_queue) == 0
        assert manager._max_concurrent_training == 2
        assert manager._training_cooldown_hours == 12

    def test_initialization_with_default_config_manager(
        self, mock_tracking_manager, mock_training_pipeline
    ):
        """Test initialization with default config manager."""
        with patch(
            "src.models.training_integration.get_training_config_manager"
        ) as mock_get_config:
            mock_config = MagicMock()
            mock_get_config.return_value = mock_config

            manager = TrainingIntegrationManager(
                tracking_manager=mock_tracking_manager,
                training_pipeline=mock_training_pipeline,
            )

            assert manager.config_manager is mock_config
            mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_success(self, integration_manager, mock_tracking_manager):
        """Test successful initialization."""
        with patch.object(
            integration_manager, "_start_background_tasks", new_callable=AsyncMock
        ) as mock_start_tasks, patch.object(
            integration_manager, "_register_tracking_callbacks", new_callable=AsyncMock
        ) as mock_register_callbacks:

            await integration_manager.initialize()

            assert integration_manager._integration_active is True
            mock_start_tasks.assert_called_once()
            mock_register_callbacks.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_failure(self, integration_manager):
        """Test initialization failure handling."""
        with patch.object(
            integration_manager,
            "_start_background_tasks",
            side_effect=Exception("Task startup failed"),
        ):

            with pytest.raises(Exception, match="Task startup failed"):
                await integration_manager.initialize()

    @pytest.mark.asyncio
    async def test_shutdown(self, integration_manager):
        """Test shutdown process."""
        # Set up background tasks
        task1 = AsyncMock()
        task2 = AsyncMock()
        task1.done.return_value = False
        task2.done.return_value = False
        integration_manager._background_tasks = [task1, task2]
        integration_manager._integration_active = True

        await integration_manager.shutdown()

        assert integration_manager._integration_active is False
        assert integration_manager._shutdown_event.is_set()

    @pytest.mark.asyncio
    async def test_register_tracking_callbacks(
        self, integration_manager, mock_tracking_manager
    ):
        """Test registration of tracking callbacks."""
        await integration_manager._register_tracking_callbacks()

        # Verify callbacks were registered
        mock_tracking_manager.add_accuracy_callback.assert_called_once_with(
            integration_manager._on_accuracy_degradation
        )
        mock_tracking_manager.add_drift_callback.assert_called_once_with(
            integration_manager._on_drift_detected
        )
        mock_tracking_manager.add_performance_callback.assert_called_once_with(
            integration_manager._on_performance_change
        )

    @pytest.mark.asyncio
    async def test_register_tracking_callbacks_missing_methods(
        self, integration_manager
    ):
        """Test callback registration with tracking manager missing methods."""
        # Create tracking manager without callback methods
        mock_tracking_manager = AsyncMock()
        integration_manager.tracking_manager = mock_tracking_manager

        # Should not raise exception even if methods are missing
        await integration_manager._register_tracking_callbacks()

    @pytest.mark.asyncio
    async def test_on_accuracy_degradation_triggers_retraining(
        self, integration_manager, mock_config_manager
    ):
        """Test accuracy degradation callback triggers retraining."""
        room_id = "test_room"
        accuracy_metrics = {
            "accuracy_rate": 70.0,  # Below threshold
            "mean_error_minutes": 20.0,  # Above threshold
        }

        with patch.object(
            integration_manager, "_queue_retraining_request", new_callable=AsyncMock
        ) as mock_queue:

            await integration_manager._on_accuracy_degradation(
                room_id, accuracy_metrics
            )

            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["room_id"] == room_id
            assert "accuracy_degradation" in call_args[1]["trigger_reason"]

    @pytest.mark.asyncio
    async def test_on_accuracy_degradation_within_thresholds(
        self, integration_manager, mock_config_manager
    ):
        """Test accuracy degradation callback when metrics are within thresholds."""
        room_id = "test_room"
        accuracy_metrics = {
            "accuracy_rate": 85.0,  # Above threshold
            "mean_error_minutes": 10.0,  # Below threshold
        }

        with patch.object(
            integration_manager, "_queue_retraining_request", new_callable=AsyncMock
        ) as mock_queue:

            await integration_manager._on_accuracy_degradation(
                room_id, accuracy_metrics
            )

            # Should not queue retraining
            mock_queue.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_drift_detected_triggers_retraining(self, integration_manager):
        """Test drift detection callback triggers retraining."""
        room_id = "test_room"
        drift_metrics = {
            "drift_severity": "MAJOR",
            "overall_drift_score": 0.8,
            "retraining_recommended": True,
            "drift_types": ["temporal", "pattern"],
            "immediate_attention_required": True,
        }

        with patch.object(
            integration_manager, "_queue_retraining_request", new_callable=AsyncMock
        ) as mock_queue:

            await integration_manager._on_drift_detected(room_id, drift_metrics)

            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["room_id"] == room_id
            assert "concept_drift_major" in call_args[1]["trigger_reason"]
            assert call_args[1]["strategy"] == "full_retrain"
            assert call_args[1]["priority"] == 1

    @pytest.mark.asyncio
    async def test_on_drift_detected_minor_drift(self, integration_manager):
        """Test drift detection with minor drift."""
        room_id = "test_room"
        drift_metrics = {
            "drift_severity": "MINOR",
            "overall_drift_score": 0.3,
            "retraining_recommended": True,
        }

        with patch.object(
            integration_manager, "_queue_retraining_request", new_callable=AsyncMock
        ) as mock_queue:

            await integration_manager._on_drift_detected(room_id, drift_metrics)

            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["strategy"] == "adaptive"
            assert call_args[1]["priority"] == 3

    @pytest.mark.asyncio
    async def test_on_drift_detected_no_retraining_recommended(
        self, integration_manager
    ):
        """Test drift detection when retraining is not recommended."""
        room_id = "test_room"
        drift_metrics = {
            "drift_severity": "MINOR",
            "overall_drift_score": 0.2,
            "retraining_recommended": False,
        }

        with patch.object(
            integration_manager, "_queue_retraining_request", new_callable=AsyncMock
        ) as mock_queue:

            await integration_manager._on_drift_detected(room_id, drift_metrics)

            # Should not queue retraining
            mock_queue.assert_not_called()

    @pytest.mark.asyncio
    async def test_on_performance_change(self, integration_manager):
        """Test performance change callback."""
        room_id = "test_room"
        performance_metrics = {
            "accuracy": 0.85,
            "latency": 120,
            "throughput": 10,
        }

        # Should complete without error (currently just logs)
        await integration_manager._on_performance_change(room_id, performance_metrics)

    @pytest.mark.asyncio
    async def test_queue_retraining_request(self, integration_manager):
        """Test queueing retraining request."""
        room_id = "test_room"
        trigger_reason = "test_trigger"

        await integration_manager._queue_retraining_request(
            room_id=room_id,
            trigger_reason=trigger_reason,
            priority=2,
            strategy="adaptive",
            metadata={"test": True},
        )

        assert len(integration_manager._training_queue) == 1
        request = integration_manager._training_queue[0]
        assert request["room_id"] == room_id
        assert request["trigger_reason"] == trigger_reason
        assert request["priority"] == 2
        assert request["strategy"] == "adaptive"
        assert request["metadata"]["test"] is True

    @pytest.mark.asyncio
    async def test_queue_retraining_request_already_in_progress(
        self, integration_manager
    ):
        """Test queueing retraining when already in progress."""
        room_id = "test_room"
        integration_manager._active_training_requests[room_id] = "pipeline_123"

        await integration_manager._queue_retraining_request(
            room_id=room_id, trigger_reason="test_trigger"
        )

        # Should not add to queue
        assert len(integration_manager._training_queue) == 0

    @pytest.mark.asyncio
    async def test_queue_retraining_request_priority_ordering(
        self, integration_manager
    ):
        """Test that retraining requests are ordered by priority."""
        # Add requests with different priorities
        await integration_manager._queue_retraining_request(
            room_id="room1", trigger_reason="trigger1", priority=3
        )
        await integration_manager._queue_retraining_request(
            room_id="room2", trigger_reason="trigger2", priority=1
        )
        await integration_manager._queue_retraining_request(
            room_id="room3", trigger_reason="trigger3", priority=2
        )

        # Check ordering
        assert len(integration_manager._training_queue) == 3
        assert integration_manager._training_queue[0]["priority"] == 1
        assert integration_manager._training_queue[1]["priority"] == 2
        assert integration_manager._training_queue[2]["priority"] == 3

    def test_calculate_priority(self, integration_manager):
        """Test priority calculation based on current value vs threshold."""
        # Critical priority
        assert integration_manager._calculate_priority(0, 80) == 1
        assert integration_manager._calculate_priority(30, 80) == 1

        # High priority
        assert integration_manager._calculate_priority(50, 80) == 2

        # Medium priority
        assert integration_manager._calculate_priority(65, 80) == 3

        # Low priority
        assert integration_manager._calculate_priority(75, 80) == 4

    def test_can_retrain_room(self, integration_manager):
        """Test cooldown period checking."""
        room_id = "test_room"

        # No previous training - should allow
        assert integration_manager._can_retrain_room(room_id) is True

        # Recent training - should not allow
        integration_manager._last_training_times[room_id] = datetime.utcnow()
        assert integration_manager._can_retrain_room(room_id) is False

        # Old training - should allow
        integration_manager._last_training_times[room_id] = (
            datetime.utcnow() - timedelta(hours=24)
        )
        assert integration_manager._can_retrain_room(room_id) is True

    def test_get_cooldown_remaining(self, integration_manager):
        """Test cooldown remaining calculation."""
        room_id = "test_room"

        # No previous training
        assert integration_manager._get_cooldown_remaining(room_id) == 0.0

        # Recent training
        integration_manager._last_training_times[room_id] = (
            datetime.utcnow() - timedelta(hours=6)
        )
        remaining = integration_manager._get_cooldown_remaining(room_id)
        assert 5.0 < remaining < 7.0  # Should be around 6 hours remaining

        # Old training
        integration_manager._last_training_times[room_id] = (
            datetime.utcnow() - timedelta(hours=24)
        )
        assert integration_manager._get_cooldown_remaining(room_id) == 0.0

    @pytest.mark.asyncio
    async def test_process_training_queue(self, integration_manager):
        """Test training queue processing."""
        # Add a request to queue
        await integration_manager._queue_retraining_request(
            room_id="test_room", trigger_reason="test_trigger"
        )

        with patch.object(
            integration_manager, "_execute_training_request", new_callable=AsyncMock
        ) as mock_execute:

            await integration_manager._process_training_queue()

            mock_execute.assert_called_once()
            # Queue should be empty after processing
            assert len(integration_manager._training_queue) == 0

    @pytest.mark.asyncio
    async def test_process_training_queue_max_concurrent(self, integration_manager):
        """Test training queue respects max concurrent limit."""
        integration_manager._max_concurrent_training = 1
        integration_manager._active_training_requests["room1"] = "pipeline1"

        # Add request to queue
        await integration_manager._queue_retraining_request(
            room_id="test_room", trigger_reason="test_trigger"
        )

        with patch.object(
            integration_manager, "_execute_training_request", new_callable=AsyncMock
        ) as mock_execute:

            await integration_manager._process_training_queue()

            # Should not execute due to concurrent limit
            mock_execute.assert_not_called()
            assert len(integration_manager._training_queue) == 1

    @pytest.mark.asyncio
    async def test_process_training_queue_cooldown_period(self, integration_manager):
        """Test training queue respects cooldown period."""
        room_id = "test_room"
        integration_manager._last_training_times[room_id] = datetime.utcnow()

        # Add request to queue
        await integration_manager._queue_retraining_request(
            room_id=room_id, trigger_reason="test_trigger"
        )

        with patch.object(
            integration_manager, "_execute_training_request", new_callable=AsyncMock
        ) as mock_execute:

            await integration_manager._process_training_queue()

            # Should not execute due to cooldown
            mock_execute.assert_not_called()
            assert len(integration_manager._training_queue) == 1

    @pytest.mark.asyncio
    async def test_execute_training_request_full_retrain(
        self, integration_manager, mock_training_pipeline, mock_config_manager
    ):
        """Test executing training request with full retrain strategy."""
        request = {
            "room_id": "test_room",
            "trigger_reason": "test_trigger",
            "strategy": "full_retrain",
        }

        # Mock successful training
        mock_progress = MagicMock()
        mock_progress.stage.value = "completed"
        mock_progress.best_model = "lstm"
        mock_training_pipeline.run_retraining_pipeline.return_value = mock_progress

        with patch.object(
            integration_manager, "_handle_training_completion", new_callable=AsyncMock
        ) as mock_handle_completion:

            await integration_manager._execute_training_request(request)

            # Verify training pipeline was called with correct parameters
            mock_training_pipeline.run_retraining_pipeline.assert_called_once()
            call_args = mock_training_pipeline.run_retraining_pipeline.call_args
            assert call_args[1]["room_id"] == "test_room"
            assert call_args[1]["strategy"] == "full_retrain"
            assert call_args[1]["force_full_retrain"] is True
            assert call_args[1]["training_type"] == TrainingType.FULL_RETRAIN.value

            mock_handle_completion.assert_called_once_with(
                "test_room", mock_progress, request
            )

    @pytest.mark.asyncio
    async def test_execute_training_request_incremental(
        self, integration_manager, mock_training_pipeline, mock_config_manager
    ):
        """Test executing training request with incremental strategy."""
        request = {
            "room_id": "test_room",
            "trigger_reason": "test_trigger",
            "strategy": "incremental",
        }

        mock_progress = MagicMock()
        mock_progress.stage.value = "completed"
        mock_training_pipeline.run_retraining_pipeline.return_value = mock_progress

        with patch.object(
            integration_manager, "_handle_training_completion", new_callable=AsyncMock
        ):

            await integration_manager._execute_training_request(request)

            call_args = mock_training_pipeline.run_retraining_pipeline.call_args
            assert call_args[1]["force_full_retrain"] is False
            assert call_args[1]["training_type"] == TrainingType.INCREMENTAL.value

    @pytest.mark.asyncio
    async def test_execute_training_request_adaptive(
        self, integration_manager, mock_training_pipeline, mock_config_manager
    ):
        """Test executing training request with adaptive strategy."""
        request = {
            "room_id": "test_room",
            "trigger_reason": "test_trigger",
            "strategy": "adaptive",
        }

        mock_progress = MagicMock()
        mock_progress.stage.value = "completed"
        mock_training_pipeline.run_retraining_pipeline.return_value = mock_progress

        with patch.object(
            integration_manager, "_handle_training_completion", new_callable=AsyncMock
        ):

            await integration_manager._execute_training_request(request)

            call_args = mock_training_pipeline.run_retraining_pipeline.call_args
            assert call_args[1]["force_full_retrain"] is False
            assert call_args[1]["training_type"] == TrainingType.ADAPTATION.value

    @pytest.mark.asyncio
    async def test_execute_training_request_failure(
        self, integration_manager, mock_training_pipeline
    ):
        """Test handling of training request execution failure."""
        request = {
            "room_id": "test_room",
            "trigger_reason": "test_trigger",
            "strategy": "adaptive",
        }

        mock_training_pipeline.run_retraining_pipeline.side_effect = Exception(
            "Training failed"
        )

        with pytest.raises(Exception, match="Training failed"):
            await integration_manager._execute_training_request(request)

        # Should clean up active training tracking
        assert "test_room" not in integration_manager._active_training_requests

    def test_select_training_profile_for_strategy(self, integration_manager):
        """Test training profile selection based on strategy."""
        assert (
            integration_manager._select_training_profile_for_strategy("full_retrain")
            == TrainingProfile.COMPREHENSIVE
        )
        assert (
            integration_manager._select_training_profile_for_strategy("incremental")
            == TrainingProfile.QUICK
        )
        assert (
            integration_manager._select_training_profile_for_strategy("adaptive")
            == TrainingProfile.PRODUCTION
        )
        assert (
            integration_manager._select_training_profile_for_strategy("unknown")
            == TrainingProfile.PRODUCTION
        )

    @pytest.mark.asyncio
    async def test_handle_training_completion_success(
        self, integration_manager, mock_tracking_manager
    ):
        """Test handling successful training completion."""
        room_id = "test_room"
        integration_manager._active_training_requests[room_id] = "pipeline_123"

        mock_progress = MagicMock()
        mock_progress.stage.value = "completed"
        mock_progress.best_model = "lstm"

        request = {"room_id": room_id, "trigger_reason": "test"}

        with patch.object(
            integration_manager, "_update_model_registration", new_callable=AsyncMock
        ) as mock_update_model:

            await integration_manager._handle_training_completion(
                room_id, mock_progress, request
            )

            # Should remove from active training
            assert room_id not in integration_manager._active_training_requests

            # Should update last training time
            assert room_id in integration_manager._last_training_times

            # Should notify tracking manager
            mock_tracking_manager.on_model_retrained.assert_called_once_with(
                room_id, mock_progress
            )

            # Should update model registration
            mock_update_model.assert_called_once_with(room_id, mock_progress)

    @pytest.mark.asyncio
    async def test_handle_training_completion_failure(
        self, integration_manager, mock_tracking_manager
    ):
        """Test handling training completion failure."""
        room_id = "test_room"
        integration_manager._active_training_requests[room_id] = "pipeline_123"

        mock_progress = MagicMock()
        mock_progress.stage.value = "failed"
        mock_progress.errors = ["Model training error"]

        request = {"room_id": room_id, "trigger_reason": "test"}

        with patch.object(
            integration_manager, "_handle_training_failure", new_callable=AsyncMock
        ) as mock_handle_failure:

            await integration_manager._handle_training_completion(
                room_id, mock_progress, request
            )

            # Should still remove from active training
            assert room_id not in integration_manager._active_training_requests

            # Should handle training failure
            mock_handle_failure.assert_called_once_with(room_id, mock_progress, request)

    @pytest.mark.asyncio
    async def test_update_model_registration(
        self, integration_manager, mock_tracking_manager, mock_training_pipeline
    ):
        """Test model registration update."""
        room_id = "test_room"
        mock_progress = MagicMock()
        mock_progress.best_model = "lstm"

        # Mock model registry
        mock_model = MagicMock()
        mock_training_pipeline.get_model_registry.return_value = {
            "test_room_lstm": mock_model
        }

        await integration_manager._update_model_registration(room_id, mock_progress)

        # Should register model with tracking manager
        mock_tracking_manager.register_model.assert_called_once_with(
            room_id=room_id, model_type="lstm", model_instance=mock_model
        )

    @pytest.mark.asyncio
    async def test_handle_training_failure_with_retry(self, integration_manager):
        """Test handling training failure with retry logic."""
        room_id = "test_room"
        mock_progress = MagicMock()
        request = {"room_id": room_id, "trigger_reason": "test", "failure_count": 1}

        await integration_manager._handle_training_failure(
            room_id, mock_progress, request
        )

        # Should add retry request to queue
        assert len(integration_manager._training_queue) == 1
        retry_request = integration_manager._training_queue[0]
        assert retry_request["failure_count"] == 2
        assert retry_request["strategy"] == "quick"
        assert retry_request["priority"] == 1

    @pytest.mark.asyncio
    async def test_handle_training_failure_max_retries(
        self, integration_manager, mock_tracking_manager
    ):
        """Test handling training failure after max retries."""
        room_id = "test_room"
        mock_progress = MagicMock()
        request = {"room_id": room_id, "trigger_reason": "test", "failure_count": 3}

        await integration_manager._handle_training_failure(
            room_id, mock_progress, request
        )

        # Should not add retry request
        assert len(integration_manager._training_queue) == 0

        # Should notify tracking manager of permanent failure
        mock_tracking_manager.on_training_failure.assert_called_once_with(
            room_id, mock_progress
        )

    @pytest.mark.asyncio
    async def test_cleanup_old_data(self, integration_manager):
        """Test cleanup of old training data."""
        # Add old request
        old_request = {
            "room_id": "test_room",
            "trigger_reason": "old_trigger",
            "requested_at": datetime.utcnow() - timedelta(hours=48),
        }
        integration_manager._training_queue.append(old_request)

        # Add recent request
        recent_request = {
            "room_id": "test_room2",
            "trigger_reason": "recent_trigger",
            "requested_at": datetime.utcnow() - timedelta(hours=1),
        }
        integration_manager._training_queue.append(recent_request)

        await integration_manager._cleanup_old_data()

        # Only recent request should remain
        assert len(integration_manager._training_queue) == 1
        assert (
            integration_manager._training_queue[0]["trigger_reason"] == "recent_trigger"
        )

    @pytest.mark.asyncio
    async def test_request_manual_training(self, integration_manager):
        """Test manual training request."""
        room_id = "test_room"

        with patch.object(
            integration_manager, "_queue_retraining_request", new_callable=AsyncMock
        ) as mock_queue:

            result = await integration_manager.request_manual_training(
                room_id=room_id,
                strategy="adaptive",
                priority=2,
                reason="user_request",
            )

            assert result is True
            mock_queue.assert_called_once()
            call_args = mock_queue.call_args
            assert call_args[1]["room_id"] == room_id
            assert "manual_user_request" in call_args[1]["trigger_reason"]
            assert call_args[1]["priority"] == 2
            assert call_args[1]["strategy"] == "adaptive"

    @pytest.mark.asyncio
    async def test_request_manual_training_failure(self, integration_manager):
        """Test manual training request failure."""
        with patch.object(
            integration_manager,
            "_queue_retraining_request",
            side_effect=Exception("Queue failed"),
        ):

            result = await integration_manager.request_manual_training(
                room_id="test_room"
            )

            assert result is False

    def test_get_integration_status(self, integration_manager):
        """Test getting integration status."""
        integration_manager._integration_active = True
        integration_manager._active_training_requests["room1"] = "pipeline1"
        integration_manager._training_queue.append(
            {
                "room_id": "room2",
                "trigger_reason": "test",
                "requested_at": datetime.utcnow(),
            }
        )
        integration_manager._last_training_times["room1"] = datetime.utcnow()

        status = integration_manager.get_integration_status()

        assert status["integration_active"] is True
        assert status["active_training_requests"] == 1
        assert status["queued_training_requests"] == 1
        assert status["max_concurrent_training"] == 2
        assert status["training_cooldown_hours"] == 12
        assert "room1" in status["rooms_with_active_training"]
        assert "room1" in status["rooms_in_cooldown"]

    def test_get_training_queue_status(self, integration_manager):
        """Test getting training queue status."""
        request_time = datetime.utcnow() - timedelta(minutes=5)
        integration_manager._training_queue.append(
            {
                "room_id": "test_room",
                "trigger_reason": "test_trigger",
                "strategy": "adaptive",
                "priority": 2,
                "requested_at": request_time,
            }
        )

        status = integration_manager.get_training_queue_status()

        assert len(status) == 1
        assert status[0]["room_id"] == "test_room"
        assert status[0]["trigger_reason"] == "test_trigger"
        assert status[0]["strategy"] == "adaptive"
        assert status[0]["priority"] == 2
        assert 4.0 < status[0]["waiting_time_minutes"] < 6.0

    def test_set_training_capacity(self, integration_manager):
        """Test setting training capacity."""
        integration_manager.set_training_capacity(5)
        assert integration_manager._max_concurrent_training == 5

        # Should not allow zero or negative
        integration_manager.set_training_capacity(0)
        assert integration_manager._max_concurrent_training == 5

    def test_set_cooldown_period(self, integration_manager):
        """Test setting cooldown period."""
        integration_manager.set_cooldown_period(24)
        assert integration_manager._training_cooldown_hours == 24

        # Should not allow zero or negative
        integration_manager.set_cooldown_period(0)
        assert integration_manager._training_cooldown_hours == 24

    def test_get_active_training_rooms(self, integration_manager):
        """Test getting active training rooms."""
        integration_manager._active_training_requests["room1"] = "pipeline1"
        integration_manager._active_training_requests["room2"] = "pipeline2"

        active_rooms = integration_manager.get_active_training_rooms()

        assert active_rooms == {"room1", "room2"}


@pytest.mark.unit
class TestIntegrateTrainingWithTrackingManager:
    """Test cases for the integration function."""

    @pytest.mark.asyncio
    async def test_integrate_training_success(self):
        """Test successful integration."""
        mock_tracking_manager = AsyncMock()
        mock_training_pipeline = AsyncMock()
        mock_config_manager = MagicMock()

        with patch(
            "src.models.training_integration.TrainingIntegrationManager"
        ) as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager_class.return_value = mock_manager

            result = await integrate_training_with_tracking_manager(
                tracking_manager=mock_tracking_manager,
                training_pipeline=mock_training_pipeline,
                config_manager=mock_config_manager,
            )

            # Verify manager was created and initialized
            mock_manager_class.assert_called_once_with(
                tracking_manager=mock_tracking_manager,
                training_pipeline=mock_training_pipeline,
                config_manager=mock_config_manager,
            )
            mock_manager.initialize.assert_called_once()
            assert result is mock_manager

    @pytest.mark.asyncio
    async def test_integrate_training_failure(self):
        """Test integration failure handling."""
        mock_tracking_manager = AsyncMock()
        mock_training_pipeline = AsyncMock()

        with patch(
            "src.models.training_integration.TrainingIntegrationManager",
            side_effect=Exception("Integration failed"),
        ):

            with pytest.raises(ModelTrainingError) as exc_info:
                await integrate_training_with_tracking_manager(
                    tracking_manager=mock_tracking_manager,
                    training_pipeline=mock_training_pipeline,
                )

            assert exc_info.value.operation == "integration"
            assert exc_info.value.context == "tracking_context"


@pytest.mark.integration
class TestTrainingIntegrationManagerIntegration:
    """Integration test cases for training integration manager."""

    @pytest.fixture
    def mock_complete_system(self):
        """Mock complete system for integration testing."""
        tracking_manager = AsyncMock()
        tracking_manager.add_accuracy_callback = MagicMock()
        tracking_manager.add_drift_callback = MagicMock()
        tracking_manager.add_performance_callback = MagicMock()

        training_pipeline = AsyncMock()
        mock_progress = MagicMock()
        mock_progress.stage.value = "completed"
        mock_progress.best_model = "lstm"
        training_pipeline.run_retraining_pipeline.return_value = mock_progress

        config_manager = MagicMock()
        env_config = MagicMock()
        env_config.quality_thresholds.min_accuracy_threshold = 0.8
        env_config.quality_thresholds.max_error_threshold_minutes = 15.0
        config_manager.get_environment_config.return_value = env_config

        return tracking_manager, training_pipeline, config_manager

    @pytest.mark.asyncio
    async def test_full_integration_workflow(self, mock_complete_system):
        """Test complete integration workflow."""
        tracking_manager, training_pipeline, config_manager = mock_complete_system

        integration_manager = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=config_manager,
        )

        # Initialize integration
        with patch.object(
            integration_manager, "_start_background_tasks", new_callable=AsyncMock
        ):
            await integration_manager.initialize()

        # Simulate accuracy degradation trigger
        await integration_manager._on_accuracy_degradation(
            "test_room", {"accuracy_rate": 70.0, "mean_error_minutes": 20.0}
        )

        # Process training queue
        await integration_manager._process_training_queue()

        # Verify training was triggered
        training_pipeline.run_retraining_pipeline.assert_called_once()

        # Cleanup
        await integration_manager.shutdown()

    @pytest.mark.asyncio
    async def test_background_task_lifecycle(self, mock_complete_system):
        """Test background task lifecycle management."""
        tracking_manager, training_pipeline, config_manager = mock_complete_system

        integration_manager = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=config_manager,
        )

        # Mock background tasks
        with patch.object(
            integration_manager, "_training_queue_processor", new_callable=AsyncMock
        ) as mock_queue_processor, patch.object(
            integration_manager, "_periodic_maintenance", new_callable=AsyncMock
        ) as mock_maintenance, patch.object(
            integration_manager, "_resource_monitor", new_callable=AsyncMock
        ) as mock_monitor:

            # Initialize should start background tasks
            await integration_manager.initialize()

            # Verify tasks were started
            assert len(integration_manager._background_tasks) == 3

            # Shutdown should clean up tasks
            await integration_manager.shutdown()

            assert integration_manager._integration_active is False
            assert integration_manager._shutdown_event.is_set()
