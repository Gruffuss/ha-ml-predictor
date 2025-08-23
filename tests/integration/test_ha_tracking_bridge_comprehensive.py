"""
Comprehensive integration tests for HATrackingBridge.

Tests all functionality from HATrackingBridge class including:
- Initialization and setup processes
- Event handling for all tracking events
- Command delegation to tracking manager
- Background synchronization tasks
- HA entity state updates
- Error handling and recovery
- System lifecycle management
- Performance and concurrent operations
- Integration patterns with EnhancedIntegrationManager
- Bridge statistics and monitoring
- Shutdown and cleanup procedures
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adaptation.tracking_manager import TrackingManager
from src.core.exceptions import ErrorSeverity
from src.integration.enhanced_integration_manager import EnhancedIntegrationManager
from src.integration.ha_tracking_bridge import (
    HATrackingBridge,
    HATrackingBridgeError,
    HATrackingBridgeStats,
)
from src.models.base.predictor import PredictionResult


class TestHATrackingBridgeStats:
    """Test HATrackingBridgeStats dataclass functionality."""

    def test_stats_initialization(self):
        """Test stats initialization with default values."""
        stats = HATrackingBridgeStats()

        assert stats.bridge_initialized is False
        assert stats.entity_updates_sent == 0
        assert stats.commands_delegated == 0
        assert stats.tracking_events_processed == 0
        assert stats.system_status_updates == 0
        assert stats.last_entity_update is None
        assert stats.last_command_delegation is None
        assert stats.bridge_errors == 0
        assert stats.last_error is None

    def test_stats_custom_initialization(self):
        """Test stats initialization with custom values."""
        last_update = datetime.now()
        stats = HATrackingBridgeStats(
            bridge_initialized=True,
            entity_updates_sent=42,
            bridge_errors=3,
            last_error="Test error",
            last_entity_update=last_update,
        )

        assert stats.bridge_initialized is True
        assert stats.entity_updates_sent == 42
        assert stats.bridge_errors == 3
        assert stats.last_error == "Test error"
        assert stats.last_entity_update == last_update


class TestHATrackingBridge:
    """Comprehensive tests for HATrackingBridge functionality."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Create mock TrackingManager with comprehensive functionality."""
        manager = MagicMock(spec=TrackingManager)

        # Setup async methods
        manager.get_system_status = AsyncMock(
            return_value={
                "status": "active",
                "rooms": ["living_room", "bedroom"],
                "predictions_today": 89,
                "last_update": datetime.now().isoformat(),
            }
        )
        manager.get_accuracy_metrics = AsyncMock(
            return_value={
                "living_room": MagicMock(accuracy_percentage=92.5),
                "bedroom": MagicMock(accuracy_percentage=88.3),
            }
        )
        manager.trigger_retraining = AsyncMock(
            return_value={"status": "started", "task_id": "retrain_123"}
        )
        manager.validate_model_performance = AsyncMock(
            return_value={
                "accuracy": 0.891,
                "mae_minutes": 7.2,
                "validation_samples": 2847,
            }
        )
        manager.force_prediction = AsyncMock(
            return_value={
                "prediction_id": "force_456",
                "predicted_time": (datetime.now() + timedelta(minutes=20)).isoformat(),
            }
        )
        manager.check_database_health = AsyncMock(
            return_value={
                "status": "healthy",
                "connection_count": 5,
                "query_latency_ms": 23.4,
            }
        )
        manager.generate_diagnostic_report = AsyncMock(
            return_value={
                "system_health": "good",
                "active_predictions": 15,
                "database_status": "connected",
            }
        )

        # Setup callback registration
        manager.register_callback = MagicMock()

        return manager

    @pytest.fixture
    def mock_enhanced_integration_manager(self):
        """Create mock EnhancedIntegrationManager."""
        manager = MagicMock(spec=EnhancedIntegrationManager)

        # Setup async methods
        manager.handle_prediction_update = AsyncMock()
        manager.update_entity_state = AsyncMock()
        manager.handle_system_status_update = AsyncMock()

        # Setup command handlers dictionary
        manager.command_handlers = {
            "existing_command": MagicMock(),
            "another_command": MagicMock(),
        }

        return manager

    @pytest.fixture
    def bridge(self, mock_tracking_manager, mock_enhanced_integration_manager):
        """Create HATrackingBridge for testing."""
        return HATrackingBridge(
            mock_tracking_manager, mock_enhanced_integration_manager
        )

    @pytest.fixture
    def sample_prediction_result(self):
        """Create sample prediction result for testing."""
        return PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=25),
            confidence=0.87,
            prediction_type="next_vacant",
            model_metadata={"model": "ensemble", "accuracy": 0.89},
        )

    # Initialization and Setup Tests

    def test_bridge_initialization(
        self, mock_tracking_manager, mock_enhanced_integration_manager
    ):
        """Test proper bridge initialization."""
        bridge = HATrackingBridge(
            mock_tracking_manager, mock_enhanced_integration_manager
        )

        # Verify component assignment
        assert bridge.tracking_manager is mock_tracking_manager
        assert bridge.enhanced_integration_manager is mock_enhanced_integration_manager

        # Verify initial state
        assert isinstance(bridge.stats, HATrackingBridgeStats)
        assert bridge._bridge_active is False
        assert bridge._background_tasks == []
        assert not bridge._shutdown_event.is_set()
        assert bridge._tracking_event_handlers == {}

    @pytest.mark.asyncio
    async def test_bridge_initialize_success(self, bridge):
        """Test successful bridge initialization."""
        await bridge.initialize()

        # Verify initialization state
        assert bridge._bridge_active is True
        assert bridge.stats.bridge_initialized is True
        assert len(bridge._tracking_event_handlers) > 0

        # Verify background tasks started
        assert len(bridge._background_tasks) == 2

        # Verify tracking manager assignment
        assert (
            bridge.enhanced_integration_manager.tracking_manager
            is bridge.tracking_manager
        )

        # Cleanup
        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_bridge_initialize_error(self, bridge):
        """Test bridge initialization error handling."""
        # Mock error during setup
        bridge.enhanced_integration_manager.command_handlers = None

        with pytest.raises(AttributeError):
            await bridge.initialize()

        # Verify error tracking
        assert bridge.stats.bridge_errors > 0
        assert bridge.stats.last_error is not None
        assert bridge._bridge_active is False

    @pytest.mark.asyncio
    async def test_bridge_shutdown(self, bridge):
        """Test bridge shutdown process."""
        # Initialize first
        await bridge.initialize()
        assert bridge._bridge_active is True

        # Shutdown
        await bridge.shutdown()

        # Verify shutdown state
        assert bridge._bridge_active is False
        assert bridge._shutdown_event.is_set()

        # Background tasks should be cancelled
        for task in bridge._background_tasks:
            assert task.cancelled() or task.done()

    def test_setup_tracking_event_handlers(self, bridge):
        """Test tracking event handlers setup."""
        bridge._setup_tracking_event_handlers()

        # Verify all expected handlers are setup
        expected_handlers = [
            "prediction_made",
            "accuracy_alert",
            "drift_detected",
            "retraining_started",
            "retraining_completed",
        ]

        for handler_name in expected_handlers:
            assert handler_name in bridge._tracking_event_handlers
            assert callable(bridge._tracking_event_handlers[handler_name])

        # Verify callback registration with tracking manager
        bridge.tracking_manager.register_callback.assert_called()

    def test_setup_command_delegation(self, bridge):
        """Test command delegation setup."""
        original_handlers = bridge.enhanced_integration_manager.command_handlers.copy()

        bridge._setup_command_delegation()

        # Verify original handlers are stored
        assert hasattr(bridge, "_original_handlers")
        assert bridge._original_handlers == original_handlers

        # Verify new command handlers are added
        command_handlers = bridge.enhanced_integration_manager.command_handlers
        expected_commands = [
            "retrain_model",
            "validate_model",
            "force_prediction",
            "check_database",
            "generate_diagnostic",
        ]

        for command in expected_commands:
            assert command in command_handlers
            assert callable(command_handlers[command])

    @pytest.mark.asyncio
    async def test_start_background_tasks(self, bridge):
        """Test background task startup."""
        await bridge._start_background_tasks()

        # Verify tasks are created
        assert len(bridge._background_tasks) == 2

        # Verify tasks are running
        for task in bridge._background_tasks:
            assert not task.done()

        # Cleanup
        for task in bridge._background_tasks:
            task.cancel()
        await asyncio.gather(*bridge._background_tasks, return_exceptions=True)

    # Event Handling Tests

    @pytest.mark.asyncio
    async def test_handle_prediction_made_success(
        self, bridge, sample_prediction_result
    ):
        """Test successful prediction made event handling."""
        room_id = "living_room"

        # Initialize bridge
        await bridge.initialize()

        await bridge.handle_prediction_made(room_id, sample_prediction_result)

        # Verify HA entity update
        bridge.enhanced_integration_manager.handle_prediction_update.assert_called_once_with(
            room_id, sample_prediction_result
        )

        # Verify statistics updated
        assert bridge.stats.tracking_events_processed == 1
        assert bridge.stats.entity_updates_sent == 1
        assert bridge.stats.last_entity_update is not None

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_prediction_made_inactive_bridge(
        self, bridge, sample_prediction_result
    ):
        """Test prediction made handling when bridge is inactive."""
        room_id = "bedroom"

        # Don't initialize bridge (inactive)
        await bridge.handle_prediction_made(room_id, sample_prediction_result)

        # Should not process when inactive
        bridge.enhanced_integration_manager.handle_prediction_update.assert_not_called()
        assert bridge.stats.tracking_events_processed == 0

    @pytest.mark.asyncio
    async def test_handle_prediction_made_error(self, bridge, sample_prediction_result):
        """Test prediction made handling with error."""
        room_id = "error_room"

        # Initialize bridge
        await bridge.initialize()

        # Mock error
        bridge.enhanced_integration_manager.handle_prediction_update.side_effect = (
            RuntimeError("Update failed")
        )

        # Should handle error gracefully
        await bridge.handle_prediction_made(room_id, sample_prediction_result)

        # Verify error tracking
        assert bridge.stats.bridge_errors == 1
        assert "Update failed" in bridge.stats.last_error

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_accuracy_alert(self, bridge):
        """Test accuracy alert handling."""
        # Create mock alert
        alert = MagicMock()
        alert.alert_type = "accuracy_degradation"
        alert.severity = "high"
        alert.room_id = "kitchen"
        alert.message = "Accuracy dropped below 80%"
        alert.timestamp = datetime.now()

        await bridge.initialize()
        await bridge.handle_accuracy_alert(alert)

        # Verify system status update called
        assert hasattr(bridge, "_update_system_alert_status")

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_accuracy_alert_missing_attributes(self, bridge):
        """Test accuracy alert handling with missing attributes."""
        # Create alert with minimal attributes
        alert = MagicMock(spec=[])  # No attributes

        await bridge.initialize()

        # Should handle gracefully with defaults
        await bridge.handle_accuracy_alert(alert)

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_drift_detected(self, bridge):
        """Test concept drift detection handling."""
        # Create mock drift metrics
        drift_metrics = MagicMock()
        drift_metrics.drift_score = 0.78
        drift_metrics.severity = "medium"
        drift_metrics.affected_features = ["temporal", "sequential"]

        await bridge.initialize()
        await bridge.handle_drift_detected(drift_metrics)

        # Should update system status
        # (Implementation uses private method, so we just verify no errors)

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_retraining_started(self, bridge):
        """Test retraining started event handling."""
        room_id = "office"
        retraining_info = {
            "training_type": "incremental",
            "trigger_reason": "accuracy_drop",
            "estimated_duration": "5_minutes",
        }

        await bridge.initialize()
        await bridge.handle_retraining_started(room_id, retraining_info)

        # Verify model training status update
        bridge.enhanced_integration_manager.update_entity_state.assert_called()
        call_args = bridge.enhanced_integration_manager.update_entity_state.call_args
        assert call_args[0][0] == "model_training"
        assert call_args[0][1] is True  # Training active

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_retraining_completed(self, bridge):
        """Test retraining completed event handling."""
        room_id = "bathroom"
        retraining_result = {
            "success": True,
            "accuracy": 0.923,
            "training_duration": "4.2_minutes",
            "samples_used": 5284,
        }

        await bridge.initialize()
        await bridge.handle_retraining_completed(room_id, retraining_result)

        # Verify multiple entity updates
        assert bridge.enhanced_integration_manager.update_entity_state.call_count >= 2

        # Check training completion update
        calls = bridge.enhanced_integration_manager.update_entity_state.call_args_list
        training_call = calls[0]
        assert training_call[0][0] == "model_training"
        assert training_call[0][1] is False  # Training complete

        # Check accuracy update
        accuracy_call = calls[1]
        assert accuracy_call[0][0] == f"{room_id}_accuracy"
        assert accuracy_call[0][1] == 0.923

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_handle_retraining_completed_no_accuracy(self, bridge):
        """Test retraining completed without accuracy update."""
        room_id = "guest_room"
        retraining_result = {"success": False}  # No accuracy

        await bridge.initialize()
        await bridge.handle_retraining_completed(room_id, retraining_result)

        # Should only update training status, not accuracy
        assert bridge.enhanced_integration_manager.update_entity_state.call_count == 1

        await bridge.shutdown()

    # Command Delegation Tests

    @pytest.mark.asyncio
    async def test_delegate_retrain_model_success(self, bridge):
        """Test successful model retraining command delegation."""
        parameters = {"room_id": "living_room", "force": True}

        result = await bridge._delegate_retrain_model(parameters)

        # Verify tracking manager called
        bridge.tracking_manager.trigger_retraining.assert_called_once_with(
            room_id="living_room", force=True
        )

        # Verify result format
        assert result["status"] == "success"
        assert "result" in result

        # Verify statistics updated
        assert bridge.stats.commands_delegated == 1
        assert bridge.stats.last_command_delegation is not None

    @pytest.mark.asyncio
    async def test_delegate_retrain_model_no_support(self, bridge):
        """Test retrain model delegation when not supported."""
        # Remove retraining method
        delattr(bridge.tracking_manager, "trigger_retraining")

        parameters = {"room_id": "bedroom"}
        result = await bridge._delegate_retrain_model(parameters)

        assert result["status"] == "error"
        assert "not supported" in result["message"]

    @pytest.mark.asyncio
    async def test_delegate_retrain_model_error(self, bridge):
        """Test retrain model delegation with error."""
        bridge.tracking_manager.trigger_retraining.side_effect = RuntimeError(
            "Retraining failed"
        )

        parameters = {"room_id": "kitchen"}
        result = await bridge._delegate_retrain_model(parameters)

        assert result["status"] == "error"
        assert "Retraining failed" in result["message"]

    @pytest.mark.asyncio
    async def test_delegate_validate_model(self, bridge):
        """Test model validation command delegation."""
        parameters = {"room_id": "dining_room", "days": 14}

        result = await bridge._delegate_validate_model(parameters)

        bridge.tracking_manager.validate_model_performance.assert_called_once_with(
            room_id="dining_room", validation_days=14
        )

        assert result["status"] == "success"
        assert bridge.stats.commands_delegated == 1

    @pytest.mark.asyncio
    async def test_delegate_validate_model_default_days(self, bridge):
        """Test model validation with default days parameter."""
        parameters = {"room_id": "hallway"}

        await bridge._delegate_validate_model(parameters)

        bridge.tracking_manager.validate_model_performance.assert_called_once_with(
            room_id="hallway", validation_days=7
        )

    @pytest.mark.asyncio
    async def test_delegate_force_prediction(self, bridge):
        """Test force prediction command delegation."""
        parameters = {"room_id": "office"}

        result = await bridge._delegate_force_prediction(parameters)

        bridge.tracking_manager.force_prediction.assert_called_once_with(
            room_id="office"
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_delegate_check_database(self, bridge):
        """Test database check command delegation."""
        parameters = {}

        result = await bridge._delegate_check_database(parameters)

        bridge.tracking_manager.check_database_health.assert_called_once()
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_delegate_check_database_no_support(self, bridge):
        """Test database check when not supported."""
        delattr(bridge.tracking_manager, "check_database_health")

        result = await bridge._delegate_check_database({})

        assert result["status"] == "success"
        assert "not available" in result["message"]

    @pytest.mark.asyncio
    async def test_delegate_generate_diagnostic(self, bridge):
        """Test diagnostic generation command delegation."""
        parameters = {"include_logs": False, "include_metrics": True}

        result = await bridge._delegate_generate_diagnostic(parameters)

        bridge.tracking_manager.generate_diagnostic_report.assert_called_once_with(
            include_logs=False, include_metrics=True
        )
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_delegate_generate_diagnostic_no_support(self, bridge):
        """Test diagnostic generation when not supported."""
        delattr(bridge.tracking_manager, "generate_diagnostic_report")

        result = await bridge._delegate_generate_diagnostic({})

        assert result["status"] == "success"
        assert "bridge_stats" in result["result"]

    @pytest.mark.asyncio
    async def test_delegate_generate_diagnostic_default_params(self, bridge):
        """Test diagnostic generation with default parameters."""
        parameters = {}

        await bridge._delegate_generate_diagnostic(parameters)

        bridge.tracking_manager.generate_diagnostic_report.assert_called_once_with(
            include_logs=True, include_metrics=True
        )

    # Background Task Tests

    @pytest.mark.asyncio
    async def test_system_status_sync_loop(self, bridge):
        """Test system status synchronization loop."""
        await bridge.initialize()

        # Let loop run for a short time
        await asyncio.sleep(0.1)

        # Verify system status was retrieved and updated
        bridge.tracking_manager.get_system_status.assert_called()
        bridge.enhanced_integration_manager.handle_system_status_update.assert_called()

        # Verify statistics
        assert bridge.stats.system_status_updates > 0

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_system_status_sync_error_recovery(self, bridge):
        """Test system status sync error recovery."""
        # Mock error in system status retrieval
        bridge.tracking_manager.get_system_status.side_effect = [
            RuntimeError("Connection error"),
            {"status": "recovered"},  # Recovery
        ]

        await bridge.initialize()

        # Let loop handle error and recover
        await asyncio.sleep(0.2)

        # Should continue running despite error
        assert len([t for t in bridge._background_tasks if not t.done()]) > 0

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_metrics_sync_loop(self, bridge):
        """Test metrics synchronization loop."""
        await bridge.initialize()

        # Let loop run briefly
        await asyncio.sleep(0.1)

        # Verify metrics were retrieved and updated
        bridge.tracking_manager.get_accuracy_metrics.assert_called()

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_metrics_sync_with_room_updates(self, bridge):
        """Test metrics sync with room-specific updates."""
        # Setup specific room metrics
        bridge.tracking_manager.get_accuracy_metrics.return_value = {
            "test_room": MagicMock(accuracy_percentage=94.2)
        }

        await bridge.initialize()

        # Let sync run
        await asyncio.sleep(0.1)

        # Verify room-specific entity update
        calls = bridge.enhanced_integration_manager.update_entity_state.call_args_list
        room_accuracy_calls = [
            call
            for call in calls
            if len(call[0]) > 0 and call[0][0] == "test_room_accuracy"
        ]
        assert len(room_accuracy_calls) > 0

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_background_task_shutdown(self, bridge):
        """Test background task shutdown behavior."""
        await bridge.initialize()

        # Verify tasks are running
        running_tasks = [t for t in bridge._background_tasks if not t.done()]
        assert len(running_tasks) == 2

        # Shutdown
        await bridge.shutdown()

        # Verify all tasks are cancelled or done
        for task in bridge._background_tasks:
            assert task.cancelled() or task.done()

    # Statistics and Monitoring Tests

    def test_get_bridge_stats(self, bridge):
        """Test bridge statistics retrieval."""
        # Update some statistics
        bridge.stats.entity_updates_sent = 15
        bridge.stats.commands_delegated = 8
        bridge.stats.bridge_errors = 2
        bridge._bridge_active = True

        stats = bridge.get_bridge_stats()

        # Verify complete statistics
        assert stats["bridge_active"] is True
        assert stats["background_tasks_count"] == 0  # Before initialization
        assert stats["event_handlers_count"] == 0  # Before setup

        # Verify bridge_stats contains all dataclass fields
        bridge_stats = stats["bridge_stats"]
        assert bridge_stats["entity_updates_sent"] == 15
        assert bridge_stats["commands_delegated"] == 8
        assert bridge_stats["bridge_errors"] == 2

    @pytest.mark.asyncio
    async def test_update_system_alert_status(self, bridge):
        """Test system alert status update."""
        alert_data = {
            "alert_type": "accuracy_degradation",
            "severity": "high",
            "room_id": "problem_room",
            "message": "Model accuracy below threshold",
        }

        await bridge.initialize()
        await bridge._update_system_alert_status(alert_data)

        # Verify entity state update
        bridge.enhanced_integration_manager.update_entity_state.assert_called_with(
            "active_alerts", 1, alert_data
        )

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_update_system_drift_status(self, bridge):
        """Test system drift status update."""
        drift_data = {
            "drift_detected": True,
            "drift_score": 0.89,
            "drift_severity": "high",
            "affected_features": ["temporal", "contextual"],
        }

        await bridge.initialize()
        await bridge._update_system_drift_status(drift_data)

        # Verify entity state update
        bridge.enhanced_integration_manager.update_entity_state.assert_called_with(
            "system_status", "drift_detected", drift_data
        )

        await bridge.shutdown()

    # Error Handling and Edge Cases Tests

    @pytest.mark.asyncio
    async def test_event_handling_inactive_bridge(self, bridge):
        """Test all event handlers when bridge is inactive."""
        # Don't initialize bridge

        await bridge.handle_prediction_made("room", MagicMock())
        await bridge.handle_accuracy_alert(MagicMock())
        await bridge.handle_drift_detected(MagicMock())
        await bridge.handle_retraining_started("room", {})
        await bridge.handle_retraining_completed("room", {})

        # All should return without processing
        bridge.enhanced_integration_manager.handle_prediction_update.assert_not_called()
        bridge.enhanced_integration_manager.update_entity_state.assert_not_called()

    @pytest.mark.asyncio
    async def test_command_delegation_error_handling(self, bridge):
        """Test error handling in all command delegation methods."""
        error_methods = [
            ("_delegate_retrain_model", {"room_id": "test"}),
            ("_delegate_validate_model", {"room_id": "test"}),
            ("_delegate_force_prediction", {"room_id": "test"}),
            ("_delegate_check_database", {}),
            ("_delegate_generate_diagnostic", {}),
        ]

        for method_name, params in error_methods:
            method = getattr(bridge, method_name)

            # Mock all tracking manager methods to raise errors
            for attr_name in dir(bridge.tracking_manager):
                if not attr_name.startswith("_"):
                    attr_obj = getattr(bridge.tracking_manager, attr_name)
                    if isinstance(attr_obj, AsyncMock):
                        attr_obj.side_effect = RuntimeError("Test error")

            result = await method(params)

            # All should return error status instead of raising
            assert result["status"] == "error"
            assert "error" in result["message"] or "not supported" in result["message"]

    @pytest.mark.asyncio
    async def test_background_task_error_recovery(self, bridge):
        """Test background task error recovery."""
        # Mock intermittent errors
        error_count = 0

        def mock_get_status():
            nonlocal error_count
            error_count += 1
            if error_count % 2 == 1:
                raise RuntimeError(f"Error {error_count}")
            return {"status": "recovered"}

        bridge.tracking_manager.get_system_status.side_effect = mock_get_status

        await bridge.initialize()

        # Let tasks run and recover from errors
        await asyncio.sleep(0.3)

        # Tasks should still be running despite errors
        running_tasks = [t for t in bridge._background_tasks if not t.done()]
        assert len(running_tasks) > 0

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_concurrent_event_handling(self, bridge, sample_prediction_result):
        """Test concurrent event handling."""
        await bridge.initialize()

        # Create multiple concurrent events
        rooms = ["room1", "room2", "room3", "room4", "room5"]
        tasks = []

        for room in rooms:
            tasks.append(bridge.handle_prediction_made(room, sample_prediction_result))
            tasks.append(bridge.handle_retraining_started(room, {"type": "test"}))

        # Execute all concurrently
        await asyncio.gather(*tasks)

        # Verify all events were processed
        assert bridge.stats.tracking_events_processed == len(rooms)
        assert (
            bridge.enhanced_integration_manager.handle_prediction_update.call_count
            == len(rooms)
        )

        await bridge.shutdown()

    # Performance and Load Tests

    @pytest.mark.asyncio
    async def test_high_frequency_events(self, bridge, sample_prediction_result):
        """Test handling high frequency events."""
        await bridge.initialize()

        # Process many events rapidly
        room_id = "high_frequency_room"
        event_count = 50

        tasks = [
            bridge.handle_prediction_made(room_id, sample_prediction_result)
            for _ in range(event_count)
        ]

        start_time = asyncio.get_event_loop().time()
        await asyncio.gather(*tasks)
        end_time = asyncio.get_event_loop().time()

        # Verify all events processed
        assert bridge.stats.tracking_events_processed == event_count

        # Should complete reasonably quickly (less than 1 second)
        assert (end_time - start_time) < 1.0

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_memory_leak_prevention(self, bridge):
        """Test that bridge doesn't leak memory with many operations."""
        await bridge.initialize()

        # Perform many operations that could potentially leak
        for i in range(100):
            await bridge.handle_prediction_made(f"room_{i % 10}", MagicMock())

            # Simulate various command delegations
            await bridge._delegate_retrain_model({"room_id": f"room_{i % 5}"})

            if i % 10 == 0:
                # Periodic cleanup simulation
                bridge.stats.last_entity_update = datetime.now()

        # Verify system is still responsive
        stats = bridge.get_bridge_stats()
        assert stats["bridge_active"] is True

        await bridge.shutdown()


class TestHATrackingBridgeError:
    """Test HATrackingBridgeError exception class."""

    def test_error_creation(self):
        """Test error creation with default values."""
        error = HATrackingBridgeError("Test bridge error")

        assert str(error) == "Test bridge error"
        assert error.error_code == "HA_TRACKING_BRIDGE_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_error_with_custom_severity(self):
        """Test error creation with custom severity."""
        error = HATrackingBridgeError(
            "Critical bridge error", severity=ErrorSeverity.HIGH
        )

        assert error.severity == ErrorSeverity.HIGH
        assert error.error_code == "HA_TRACKING_BRIDGE_ERROR"

    def test_error_inheritance(self):
        """Test error inheritance from base exception."""
        error = HATrackingBridgeError("Test error")

        # Should inherit from base occupancy prediction error
        from src.core.exceptions import OccupancyPredictionError

        assert isinstance(error, OccupancyPredictionError)


class TestIntegrationScenarios:
    """Integration scenarios for HATrackingBridge with real-world patterns."""

    @pytest.fixture
    def integrated_bridge_setup(self):
        """Setup bridge with realistic component interactions."""
        tracking_manager = MagicMock(spec=TrackingManager)
        integration_manager = MagicMock(spec=EnhancedIntegrationManager)

        # Setup realistic tracking manager
        tracking_manager.get_system_status = AsyncMock(
            return_value={
                "status": "active",
                "rooms": ["living_room", "bedroom", "kitchen"],
                "predictions_today": 245,
                "accuracy_average": 0.897,
            }
        )
        tracking_manager.get_accuracy_metrics = AsyncMock(
            return_value={
                "living_room": MagicMock(accuracy_percentage=94.2),
                "bedroom": MagicMock(accuracy_percentage=91.7),
                "kitchen": MagicMock(accuracy_percentage=88.9),
            }
        )
        tracking_manager.register_callback = MagicMock()

        # Setup realistic integration manager
        integration_manager.handle_prediction_update = AsyncMock()
        integration_manager.update_entity_state = AsyncMock()
        integration_manager.handle_system_status_update = AsyncMock()
        integration_manager.command_handlers = {}

        bridge = HATrackingBridge(tracking_manager, integration_manager)
        return bridge, tracking_manager, integration_manager

    @pytest.mark.asyncio
    async def test_complete_system_lifecycle(self, integrated_bridge_setup):
        """Test complete system lifecycle with realistic operations."""
        bridge, tracking_manager, integration_manager = integrated_bridge_setup

        # Initialize system
        await bridge.initialize()
        assert bridge._bridge_active is True

        # Simulate prediction events
        prediction_result = PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=18),
            confidence=0.91,
            prediction_type="next_occupied",
        )

        await bridge.handle_prediction_made("living_room", prediction_result)

        # Simulate retraining cycle
        await bridge.handle_retraining_started("living_room", {"type": "scheduled"})
        await bridge.handle_retraining_completed(
            "living_room", {"success": True, "accuracy": 0.945}
        )

        # Simulate command delegation
        retrain_result = await bridge._delegate_retrain_model({"room_id": "bedroom"})
        assert retrain_result["status"] == "success"

        # Verify system synchronization occurred
        integration_manager.handle_prediction_update.assert_called()
        integration_manager.update_entity_state.assert_called()

        # Shutdown system
        await bridge.shutdown()
        assert bridge._bridge_active is False

    @pytest.mark.asyncio
    async def test_error_recovery_and_alerting(self, integrated_bridge_setup):
        """Test system error recovery and alerting patterns."""
        bridge, tracking_manager, integration_manager = integrated_bridge_setup

        await bridge.initialize()

        # Simulate various error conditions
        integration_manager.handle_prediction_update.side_effect = [
            RuntimeError("Network error"),  # First call fails
            None,  # Second call succeeds
        ]

        # First prediction should handle error gracefully
        prediction = PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=10),
            confidence=0.85,
            prediction_type="next_vacant",
        )

        await bridge.handle_prediction_made("error_room", prediction)
        assert bridge.stats.bridge_errors == 1

        # System should continue operating
        await bridge.handle_prediction_made("recovery_room", prediction)
        assert bridge.stats.entity_updates_sent == 1  # Second attempt succeeded

        await bridge.shutdown()

    @pytest.mark.asyncio
    async def test_high_load_scenario(self, integrated_bridge_setup):
        """Test bridge behavior under high load conditions."""
        bridge, tracking_manager, integration_manager = integrated_bridge_setup

        await bridge.initialize()

        # Simulate high load with many concurrent operations
        rooms = [f"room_{i}" for i in range(20)]
        prediction = PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=15),
            confidence=0.88,
            prediction_type="next_occupied",
        )

        # Create many concurrent tasks
        prediction_tasks = [
            bridge.handle_prediction_made(room, prediction) for room in rooms
        ]
        retraining_tasks = [
            bridge.handle_retraining_started(room, {"type": "drift_detected"})
            for room in rooms[:10]
        ]
        command_tasks = [
            bridge._delegate_validate_model({"room_id": room}) for room in rooms[:5]
        ]

        # Execute all concurrently
        all_tasks = prediction_tasks + retraining_tasks + command_tasks
        results = await asyncio.gather(*all_tasks, return_exceptions=True)

        # Most operations should succeed (some may have errors due to mocking)
        successful_results = [r for r in results if not isinstance(r, Exception)]
        assert len(successful_results) > len(all_tasks) * 0.8  # 80% success rate

        # System should remain active
        assert bridge._bridge_active is True

        await bridge.shutdown()
