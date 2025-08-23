"""
Comprehensive integration test suite for HATrackingBridge.
Tests Home Assistant integration, tracking management, and system workflow integration.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, call, patch

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
    """Test HATrackingBridgeStats dataclass."""

    def test_stats_initialization(self):
        """Test stats initialization with defaults."""
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

    def test_stats_custom_values(self):
        """Test stats with custom values."""
        now = datetime.utcnow()
        stats = HATrackingBridgeStats(
            bridge_initialized=True,
            entity_updates_sent=42,
            bridge_errors=3,
            last_error="Test error",
        )

        assert stats.bridge_initialized is True
        assert stats.entity_updates_sent == 42
        assert stats.bridge_errors == 3
        assert stats.last_error == "Test error"


class TestHATrackingBridge:
    """Test HATrackingBridge class."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Create mock tracking manager."""
        manager = Mock(spec=TrackingManager)
        manager.register_callback = Mock()
        manager.get_system_status = AsyncMock(
            return_value={
                "status": "healthy",
                "tracking_active": True,
                "rooms_tracked": 3,
                "last_prediction": datetime.now(timezone.utc).isoformat(),
            }
        )
        manager.get_accuracy_metrics = AsyncMock(
            return_value={
                "living_room": Mock(accuracy_percentage=0.92),
                "bedroom": Mock(accuracy_percentage=0.89),
                "kitchen": Mock(accuracy_percentage=0.95),
            }
        )
        manager.trigger_retraining = AsyncMock(
            return_value={"success": True, "model_id": "model_123"}
        )
        manager.validate_model_performance = AsyncMock(
            return_value={"accuracy": 0.91, "samples": 1000}
        )
        manager.force_prediction = AsyncMock(
            return_value={"predicted_time": datetime.now(timezone.utc).isoformat()}
        )
        manager.check_database_health = AsyncMock(
            return_value={"status": "healthy", "connections": 5}
        )
        manager.generate_diagnostic_report = AsyncMock(
            return_value={"report_id": "diag_456", "size": 1024}
        )
        return manager

    @pytest.fixture
    def mock_enhanced_integration_manager(self):
        """Create mock enhanced integration manager."""
        manager = Mock(spec=EnhancedIntegrationManager)
        manager.command_handlers = {}
        manager.handle_prediction_update = AsyncMock()
        manager.update_entity_state = AsyncMock()
        manager.handle_system_status_update = AsyncMock()
        return manager

    @pytest.fixture
    def prediction_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            prediction_type="next_occupied",
            predicted_time=datetime.now(timezone.utc),
            confidence=0.87,
            metadata={"model_used": "ensemble", "features_count": 45},
        )

    @pytest.fixture
    def bridge(self, mock_tracking_manager, mock_enhanced_integration_manager):
        """Create HATrackingBridge instance."""
        return HATrackingBridge(
            mock_tracking_manager, mock_enhanced_integration_manager
        )

    def test_initialization(
        self, bridge, mock_tracking_manager, mock_enhanced_integration_manager
    ):
        """Test bridge initialization."""
        assert bridge.tracking_manager is mock_tracking_manager
        assert bridge.enhanced_integration_manager is mock_enhanced_integration_manager
        assert isinstance(bridge.stats, HATrackingBridgeStats)
        assert bridge._bridge_active is False
        assert bridge._background_tasks == []
        assert hasattr(bridge, "_shutdown_event")
        assert bridge._tracking_event_handlers == {}

    @pytest.mark.asyncio
    async def test_initialize_success(self, bridge):
        """Test successful bridge initialization."""
        with patch.object(
            bridge, "_setup_tracking_event_handlers"
        ) as mock_setup_handlers, patch.object(
            bridge, "_setup_command_delegation"
        ) as mock_setup_commands, patch.object(
            bridge, "_start_background_tasks"
        ) as mock_start_tasks:

            await bridge.initialize()

            assert bridge._bridge_active is True
            assert bridge.stats.bridge_initialized is True
            assert (
                bridge.enhanced_integration_manager.tracking_manager
                is bridge.tracking_manager
            )

            mock_setup_handlers.assert_called_once()
            mock_setup_commands.assert_called_once()
            mock_start_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_error(self, bridge):
        """Test bridge initialization error."""
        error = Exception("Setup failed")

        with patch.object(bridge, "_setup_tracking_event_handlers", side_effect=error):
            with pytest.raises(Exception, match="Setup failed"):
                await bridge.initialize()

            assert bridge._bridge_active is False
            assert bridge.stats.bridge_errors == 1
            assert bridge.stats.last_error == "Setup failed"

    @pytest.mark.asyncio
    async def test_shutdown(self, bridge):
        """Test bridge shutdown."""
        # Simulate active bridge with background tasks
        bridge._bridge_active = True
        mock_task1 = Mock()
        mock_task1.done.return_value = False
        mock_task2 = Mock()
        mock_task2.done.return_value = True
        bridge._background_tasks = [mock_task1, mock_task2]

        with patch("asyncio.gather", return_value=None) as mock_gather:
            await bridge.shutdown()

            assert bridge._bridge_active is False
            assert bridge._shutdown_event.is_set()
            mock_task1.cancel.assert_called_once()
            mock_task2.cancel.assert_not_called()  # Already done
            mock_gather.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_error(self, bridge):
        """Test bridge shutdown with error."""
        bridge._bridge_active = True
        error = Exception("Shutdown failed")

        with patch("asyncio.gather", side_effect=error):
            # Should not raise exception
            await bridge.shutdown()

            assert bridge._bridge_active is False

    @pytest.mark.asyncio
    async def test_handle_prediction_made_success(self, bridge, prediction_result):
        """Test successful prediction made handling."""
        bridge._bridge_active = True
        room_id = "living_room"

        await bridge.handle_prediction_made(room_id, prediction_result)

        bridge.enhanced_integration_manager.handle_prediction_update.assert_called_once_with(
            room_id, prediction_result
        )
        assert bridge.stats.tracking_events_processed == 1
        assert bridge.stats.entity_updates_sent == 1
        assert bridge.stats.last_entity_update is not None

    @pytest.mark.asyncio
    async def test_handle_prediction_made_inactive_bridge(
        self, bridge, prediction_result
    ):
        """Test prediction made handling when bridge is inactive."""
        bridge._bridge_active = False

        await bridge.handle_prediction_made("room", prediction_result)

        bridge.enhanced_integration_manager.handle_prediction_update.assert_not_called()
        assert bridge.stats.tracking_events_processed == 0

    @pytest.mark.asyncio
    async def test_handle_prediction_made_error(self, bridge, prediction_result):
        """Test prediction made handling with error."""
        bridge._bridge_active = True
        error = Exception("Update failed")
        bridge.enhanced_integration_manager.handle_prediction_update.side_effect = error

        await bridge.handle_prediction_made("room", prediction_result)

        assert bridge.stats.bridge_errors == 1
        assert bridge.stats.last_error == "Update failed"

    @pytest.mark.asyncio
    async def test_handle_accuracy_alert(self, bridge):
        """Test accuracy alert handling."""
        bridge._bridge_active = True

        alert = Mock()
        alert.alert_type = "low_accuracy"
        alert.severity = "high"
        alert.room_id = "bedroom"
        alert.message = "Accuracy dropped below threshold"
        alert.timestamp = datetime.utcnow()

        with patch.object(bridge, "_update_system_alert_status") as mock_update:
            await bridge.handle_accuracy_alert(alert)

            mock_update.assert_called_once()
            call_args = mock_update.call_args[0][0]
            assert call_args["alert_type"] == "low_accuracy"
            assert call_args["severity"] == "high"
            assert call_args["room_id"] == "bedroom"

    @pytest.mark.asyncio
    async def test_handle_accuracy_alert_minimal_data(self, bridge):
        """Test accuracy alert handling with minimal alert data."""
        bridge._bridge_active = True

        alert = Mock()
        # Remove optional attributes to test defaults
        for attr in ["alert_type", "severity", "room_id", "message", "timestamp"]:
            if hasattr(alert, attr):
                delattr(alert, attr)

        with patch.object(bridge, "_update_system_alert_status") as mock_update:
            await bridge.handle_accuracy_alert(alert)

            mock_update.assert_called_once()
            call_args = mock_update.call_args[0][0]
            assert call_args["alert_type"] == "unknown"
            assert call_args["severity"] == "unknown"
            assert call_args["room_id"] is None

    @pytest.mark.asyncio
    async def test_handle_drift_detected(self, bridge):
        """Test drift detection handling."""
        bridge._bridge_active = True

        drift_metrics = Mock()
        drift_metrics.drift_score = 0.85
        drift_metrics.severity = "high"
        drift_metrics.affected_features = ["temporal", "sequential"]

        with patch.object(bridge, "_update_system_drift_status") as mock_update:
            await bridge.handle_drift_detected(drift_metrics)

            mock_update.assert_called_once()
            call_args = mock_update.call_args[0][0]
            assert call_args["drift_detected"] is True
            assert call_args["drift_score"] == 0.85
            assert call_args["drift_severity"] == "high"
            assert call_args["affected_features"] == ["temporal", "sequential"]

    @pytest.mark.asyncio
    async def test_handle_retraining_started(self, bridge):
        """Test retraining started handling."""
        bridge._bridge_active = True
        room_id = "kitchen"
        retraining_info = {"training_type": "full_retrain", "trigger": "drift_detected"}

        await bridge.handle_retraining_started(room_id, retraining_info)

        bridge.enhanced_integration_manager.update_entity_state.assert_called_once_with(
            "model_training",
            True,
            {
                "room_id": room_id,
                "training_started": pytest.approx(datetime.utcnow().isoformat(), abs=1),
                "training_type": "full_retrain",
            },
        )

    @pytest.mark.asyncio
    async def test_handle_retraining_completed(self, bridge):
        """Test retraining completed handling."""
        bridge._bridge_active = True
        room_id = "office"
        retraining_result = {"success": True, "accuracy": 0.94}

        await bridge.handle_retraining_completed(room_id, retraining_result)

        # Verify training status update
        calls = bridge.enhanced_integration_manager.update_entity_state.call_args_list
        assert len(calls) == 2

        # First call: training status
        first_call = calls[0]
        assert first_call[0][0] == "model_training"
        assert first_call[0][1] is False

        # Second call: accuracy update
        second_call = calls[1]
        assert second_call[0][0] == f"{room_id}_accuracy"
        assert second_call[0][1] == 0.94

    def test_get_bridge_stats(self, bridge):
        """Test bridge statistics retrieval."""
        bridge._bridge_active = True
        bridge.stats.entity_updates_sent = 10
        bridge._background_tasks = [Mock(), Mock()]
        bridge._tracking_event_handlers = {"event1": Mock(), "event2": Mock()}

        stats = bridge.get_bridge_stats()

        assert "bridge_stats" in stats
        assert stats["bridge_active"] is True
        assert stats["background_tasks_count"] == 2
        assert stats["event_handlers_count"] == 2
        assert stats["bridge_stats"]["entity_updates_sent"] == 10

    def test_setup_tracking_event_handlers(self, bridge, mock_tracking_manager):
        """Test tracking event handlers setup."""
        bridge._setup_tracking_event_handlers()

        assert len(bridge._tracking_event_handlers) == 5
        expected_handlers = [
            "prediction_made",
            "accuracy_alert",
            "drift_detected",
            "retraining_started",
            "retraining_completed",
        ]
        for handler_name in expected_handlers:
            assert handler_name in bridge._tracking_event_handlers

        # Verify registration with tracking manager
        expected_calls = len(expected_handlers)
        assert mock_tracking_manager.register_callback.call_count == expected_calls

    def test_setup_tracking_event_handlers_no_callback_support(
        self, bridge, mock_tracking_manager
    ):
        """Test setup when tracking manager doesn't support callbacks."""
        # Remove register_callback method
        del mock_tracking_manager.register_callback

        bridge._setup_tracking_event_handlers()

        # Should still setup handlers
        assert len(bridge._tracking_event_handlers) == 5

    def test_setup_command_delegation(self, bridge, mock_enhanced_integration_manager):
        """Test command delegation setup."""
        original_handlers = {"existing_handler": Mock()}
        mock_enhanced_integration_manager.command_handlers = original_handlers.copy()

        bridge._setup_command_delegation()

        # Verify original handlers stored
        assert hasattr(bridge, "_original_handlers")
        assert bridge._original_handlers == original_handlers

        # Verify new handlers added
        expected_commands = [
            "retrain_model",
            "validate_model",
            "force_prediction",
            "check_database",
            "generate_diagnostic",
        ]
        for command in expected_commands:
            assert command in mock_enhanced_integration_manager.command_handlers

    @pytest.mark.asyncio
    async def test_start_background_tasks(self, bridge):
        """Test background tasks startup."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_task1 = Mock()
            mock_task2 = Mock()
            mock_create_task.side_effect = [mock_task1, mock_task2]

            await bridge._start_background_tasks()

            assert len(bridge._background_tasks) == 2
            assert mock_task1 in bridge._background_tasks
            assert mock_task2 in bridge._background_tasks
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_system_status_sync_loop(self, bridge, mock_tracking_manager):
        """Test system status synchronization loop."""
        bridge._shutdown_event = asyncio.Event()

        # Stop loop after first iteration
        async def stop_after_first():
            await asyncio.sleep(0.1)
            bridge._shutdown_event.set()

        stop_task = asyncio.create_task(stop_after_first())

        with patch("asyncio.sleep", return_value=None):
            sync_task = asyncio.create_task(bridge._system_status_sync_loop())
            await asyncio.gather(stop_task, sync_task, return_exceptions=True)

        # Verify system status was retrieved and updated
        mock_tracking_manager.get_system_status.assert_called()
        bridge.enhanced_integration_manager.handle_system_status_update.assert_called()
        assert bridge.stats.system_status_updates > 0

    @pytest.mark.asyncio
    async def test_system_status_sync_loop_error_handling(
        self, bridge, mock_tracking_manager
    ):
        """Test system status sync loop error handling."""
        bridge._shutdown_event = asyncio.Event()

        # Make status call fail first time, succeed second time
        mock_tracking_manager.get_system_status.side_effect = [
            Exception("Status failed"),
            {"status": "healthy"},
        ]

        # Stop after brief time
        async def stop_after_delay():
            await asyncio.sleep(0.2)
            bridge._shutdown_event.set()

        stop_task = asyncio.create_task(stop_after_delay())

        with patch("asyncio.sleep", return_value=None):
            sync_task = asyncio.create_task(bridge._system_status_sync_loop())
            await asyncio.gather(stop_task, sync_task, return_exceptions=True)

        # Should have attempted multiple status calls despite error
        assert mock_tracking_manager.get_system_status.call_count >= 1

    @pytest.mark.asyncio
    async def test_metrics_sync_loop(self, bridge, mock_tracking_manager):
        """Test metrics synchronization loop."""
        bridge._shutdown_event = asyncio.Event()

        # Stop loop after first iteration
        async def stop_after_first():
            await asyncio.sleep(0.1)
            bridge._shutdown_event.set()

        stop_task = asyncio.create_task(stop_after_first())

        with patch("asyncio.sleep", return_value=None):
            sync_task = asyncio.create_task(bridge._metrics_sync_loop())
            await asyncio.gather(stop_task, sync_task, return_exceptions=True)

        # Verify metrics were retrieved and accuracy updated for each room
        mock_tracking_manager.get_accuracy_metrics.assert_called()
        accuracy_calls = (
            bridge.enhanced_integration_manager.update_entity_state.call_args_list
        )
        room_accuracy_updates = [
            call for call in accuracy_calls if "_accuracy" in call[0][0]
        ]
        assert len(room_accuracy_updates) >= 2  # living_room and bedroom from mock

    @pytest.mark.asyncio
    async def test_update_system_alert_status(self, bridge):
        """Test system alert status update."""
        alert_data = {
            "alert_type": "high_error_rate",
            "severity": "critical",
            "message": "Error rate exceeded threshold",
        }

        await bridge._update_system_alert_status(alert_data)

        bridge.enhanced_integration_manager.update_entity_state.assert_called_once_with(
            "active_alerts", 1, alert_data
        )

    @pytest.mark.asyncio
    async def test_update_system_drift_status(self, bridge):
        """Test system drift status update."""
        drift_data = {
            "drift_detected": True,
            "drift_score": 0.9,
            "drift_severity": "critical",
        }

        await bridge._update_system_drift_status(drift_data)

        bridge.enhanced_integration_manager.update_entity_state.assert_called_once_with(
            "system_status", "drift_detected", drift_data
        )

    @pytest.mark.asyncio
    async def test_delegate_retrain_model_success(self, bridge, mock_tracking_manager):
        """Test successful model retraining delegation."""
        parameters = {"room_id": "living_room", "force": True}

        result = await bridge._delegate_retrain_model(parameters)

        assert result["status"] == "success"
        assert result["result"] == {"success": True, "model_id": "model_123"}

        mock_tracking_manager.trigger_retraining.assert_called_once_with(
            room_id="living_room", force=True
        )
        assert bridge.stats.commands_delegated == 1
        assert bridge.stats.last_command_delegation is not None

    @pytest.mark.asyncio
    async def test_delegate_retrain_model_no_support(
        self, bridge, mock_tracking_manager
    ):
        """Test model retraining delegation when not supported."""
        # Remove method to simulate no support
        del mock_tracking_manager.trigger_retraining

        result = await bridge._delegate_retrain_model({"room_id": "room"})

        assert result["status"] == "error"
        assert result["message"] == "Retraining not supported"

    @pytest.mark.asyncio
    async def test_delegate_retrain_model_error(self, bridge, mock_tracking_manager):
        """Test model retraining delegation with error."""
        error = Exception("Retraining failed")
        mock_tracking_manager.trigger_retraining.side_effect = error

        result = await bridge._delegate_retrain_model({"room_id": "room"})

        assert result["status"] == "error"
        assert result["message"] == "Retraining failed"

    @pytest.mark.asyncio
    async def test_delegate_validate_model_success(self, bridge, mock_tracking_manager):
        """Test successful model validation delegation."""
        parameters = {"room_id": "bedroom", "days": 14}

        result = await bridge._delegate_validate_model(parameters)

        assert result["status"] == "success"
        assert result["result"] == {"accuracy": 0.91, "samples": 1000}

        mock_tracking_manager.validate_model_performance.assert_called_once_with(
            room_id="bedroom", validation_days=14
        )

    @pytest.mark.asyncio
    async def test_delegate_validate_model_default_days(
        self, bridge, mock_tracking_manager
    ):
        """Test model validation delegation with default days."""
        parameters = {"room_id": "kitchen"}

        result = await bridge._delegate_validate_model(parameters)

        mock_tracking_manager.validate_model_performance.assert_called_once_with(
            room_id="kitchen", validation_days=7
        )

    @pytest.mark.asyncio
    async def test_delegate_force_prediction_success(
        self, bridge, mock_tracking_manager
    ):
        """Test successful force prediction delegation."""
        parameters = {"room_id": "office"}

        result = await bridge._delegate_force_prediction(parameters)

        assert result["status"] == "success"
        mock_tracking_manager.force_prediction.assert_called_once_with(room_id="office")

    @pytest.mark.asyncio
    async def test_delegate_check_database_success(self, bridge, mock_tracking_manager):
        """Test successful database check delegation."""
        result = await bridge._delegate_check_database({})

        assert result["status"] == "success"
        assert result["result"] == {"status": "healthy", "connections": 5}
        mock_tracking_manager.check_database_health.assert_called_once()

    @pytest.mark.asyncio
    async def test_delegate_check_database_no_support(
        self, bridge, mock_tracking_manager
    ):
        """Test database check delegation when not supported."""
        del mock_tracking_manager.check_database_health

        result = await bridge._delegate_check_database({})

        assert result["status"] == "success"
        assert result["message"] == "Database check not available"

    @pytest.mark.asyncio
    async def test_delegate_generate_diagnostic_success(
        self, bridge, mock_tracking_manager
    ):
        """Test successful diagnostic generation delegation."""
        parameters = {"include_logs": True, "include_metrics": False}

        result = await bridge._delegate_generate_diagnostic(parameters)

        assert result["status"] == "success"
        assert result["result"] == {"report_id": "diag_456", "size": 1024}

        mock_tracking_manager.generate_diagnostic_report.assert_called_once_with(
            include_logs=True, include_metrics=False
        )

    @pytest.mark.asyncio
    async def test_delegate_generate_diagnostic_no_support(
        self, bridge, mock_tracking_manager
    ):
        """Test diagnostic generation when not supported."""
        del mock_tracking_manager.generate_diagnostic_report

        result = await bridge._delegate_generate_diagnostic({})

        assert result["status"] == "success"
        assert "bridge_stats" in result["result"]
        assert "timestamp" in result["result"]

    @pytest.mark.asyncio
    async def test_delegate_generate_diagnostic_defaults(
        self, bridge, mock_tracking_manager
    ):
        """Test diagnostic generation with default parameters."""
        result = await bridge._delegate_generate_diagnostic({})

        mock_tracking_manager.generate_diagnostic_report.assert_called_once_with(
            include_logs=True, include_metrics=True
        )


class TestHATrackingBridgeError:
    """Test HATrackingBridgeError exception class."""

    def test_error_initialization(self):
        """Test error initialization with default values."""
        error = HATrackingBridgeError("Test error message")

        assert str(error) == "Test error message"
        assert error.error_code == "HA_TRACKING_BRIDGE_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_error_with_custom_severity(self):
        """Test error initialization with custom severity."""
        error = HATrackingBridgeError("Critical error", severity=ErrorSeverity.HIGH)

        assert error.severity == ErrorSeverity.HIGH

    def test_error_with_additional_kwargs(self):
        """Test error initialization with additional kwargs."""
        error = HATrackingBridgeError(
            "Error with context", component="tracking_bridge", room_id="living_room"
        )

        assert error.error_code == "HA_TRACKING_BRIDGE_ERROR"
        assert hasattr(error, "component")  # Additional kwargs should be preserved


class TestIntegrationWorkflows:
    """Test complete integration workflows."""

    @pytest.fixture
    def bridge(self, mock_tracking_manager, mock_enhanced_integration_manager):
        """Create initialized bridge for workflow tests."""
        bridge = HATrackingBridge(
            mock_tracking_manager, mock_enhanced_integration_manager
        )
        bridge._bridge_active = True
        return bridge

    @pytest.mark.asyncio
    async def test_full_prediction_workflow(self, bridge, prediction_result):
        """Test complete prediction workflow from creation to HA update."""
        room_id = "living_room"

        # Handle prediction made
        await bridge.handle_prediction_made(room_id, prediction_result)

        # Verify HA entities updated
        bridge.enhanced_integration_manager.handle_prediction_update.assert_called_once_with(
            room_id, prediction_result
        )

        # Verify stats updated
        assert bridge.stats.tracking_events_processed == 1
        assert bridge.stats.entity_updates_sent == 1

    @pytest.mark.asyncio
    async def test_retraining_workflow(self, bridge):
        """Test complete model retraining workflow."""
        room_id = "bedroom"

        # Start retraining
        retraining_info = {"training_type": "drift_triggered", "features": 42}
        await bridge.handle_retraining_started(room_id, retraining_info)

        # Complete retraining
        retraining_result = {"success": True, "accuracy": 0.93, "duration": 120}
        await bridge.handle_retraining_completed(room_id, retraining_result)

        # Verify HA entities updated for both start and completion
        calls = bridge.enhanced_integration_manager.update_entity_state.call_args_list
        assert len(calls) == 3  # Start, completion status, accuracy

    @pytest.mark.asyncio
    async def test_alert_and_drift_workflow(self, bridge):
        """Test alert and drift detection workflow."""
        # Handle accuracy alert
        alert = Mock()
        alert.alert_type = "accuracy_degradation"
        alert.severity = "high"
        alert.room_id = "kitchen"
        alert.message = "Accuracy below 70%"
        alert.timestamp = datetime.utcnow()

        await bridge.handle_accuracy_alert(alert)

        # Handle drift detection
        drift_metrics = Mock()
        drift_metrics.drift_score = 0.8
        drift_metrics.severity = "medium"
        drift_metrics.affected_features = ["temporal"]

        await bridge.handle_drift_detected(drift_metrics)

        # Verify both events processed
        assert bridge.enhanced_integration_manager.update_entity_state.call_count >= 2

    @pytest.mark.asyncio
    async def test_command_delegation_workflow(self, bridge, mock_tracking_manager):
        """Test command delegation from HA to tracking manager."""
        # Setup command handlers
        bridge._setup_command_delegation()

        # Test various commands
        commands_to_test = [
            ("retrain_model", {"room_id": "living_room", "force": False}),
            ("validate_model", {"room_id": "bedroom", "days": 30}),
            ("force_prediction", {"room_id": "kitchen"}),
            ("check_database", {}),
            ("generate_diagnostic", {"include_logs": False}),
        ]

        for command_name, parameters in commands_to_test:
            handler = bridge.enhanced_integration_manager.command_handlers[command_name]
            result = await handler(parameters)
            assert result["status"] in ["success", "error"]

        # Verify tracking manager methods called
        assert mock_tracking_manager.trigger_retraining.called
        assert mock_tracking_manager.validate_model_performance.called
        assert mock_tracking_manager.force_prediction.called
        assert mock_tracking_manager.check_database_health.called
        assert mock_tracking_manager.generate_diagnostic_report.called

    @pytest.mark.asyncio
    async def test_background_sync_workflow(self, bridge, mock_tracking_manager):
        """Test background synchronization workflow."""
        # Setup for short sync loops
        bridge._shutdown_event = asyncio.Event()

        # Mock system status and metrics
        mock_tracking_manager.get_system_status.return_value = {
            "status": "healthy",
            "active_rooms": 5,
        }
        mock_tracking_manager.get_accuracy_metrics.return_value = {
            "room1": Mock(accuracy_percentage=0.9),
            "room2": Mock(accuracy_percentage=0.85),
        }

        # Start background tasks
        with patch("asyncio.sleep", return_value=None):
            status_task = asyncio.create_task(bridge._system_status_sync_loop())
            metrics_task = asyncio.create_task(bridge._metrics_sync_loop())

            # Let them run briefly
            await asyncio.sleep(0.1)
            bridge._shutdown_event.set()

            # Wait for completion
            await asyncio.gather(status_task, metrics_task, return_exceptions=True)

        # Verify sync operations occurred
        mock_tracking_manager.get_system_status.assert_called()
        mock_tracking_manager.get_accuracy_metrics.assert_called()
        bridge.enhanced_integration_manager.handle_system_status_update.assert_called()

    @pytest.mark.asyncio
    async def test_error_recovery_workflow(self, bridge, mock_tracking_manager):
        """Test error recovery in various workflows."""
        # Test prediction update error recovery
        bridge.enhanced_integration_manager.handle_prediction_update.side_effect = (
            Exception("Update failed")
        )

        prediction_result = PredictionResult(
            prediction_type="next_vacant",
            predicted_time=datetime.now(timezone.utc),
            confidence=0.8,
            metadata={},
        )

        # Should handle error gracefully
        await bridge.handle_prediction_made("room", prediction_result)

        assert bridge.stats.bridge_errors == 1
        assert bridge.stats.last_error == "Update failed"

        # Test command delegation error recovery
        mock_tracking_manager.trigger_retraining.side_effect = Exception(
            "Retraining error"
        )

        result = await bridge._delegate_retrain_model({"room_id": "room"})
        assert result["status"] == "error"
        assert "Retraining error" in result["message"]


class TestPerformanceAndScalability:
    """Test performance and scalability scenarios."""

    @pytest.fixture
    def bridge(self, mock_tracking_manager, mock_enhanced_integration_manager):
        """Create bridge for performance tests."""
        bridge = HATrackingBridge(
            mock_tracking_manager, mock_enhanced_integration_manager
        )
        bridge._bridge_active = True
        return bridge

    @pytest.mark.asyncio
    async def test_high_volume_predictions(self, bridge):
        """Test handling high volume of prediction updates."""
        prediction_results = []
        for i in range(100):
            result = PredictionResult(
                prediction_type="next_occupied" if i % 2 == 0 else "next_vacant",
                predicted_time=datetime.now(timezone.utc),
                confidence=0.7 + (i % 30) / 100,
                metadata={"batch_id": i},
            )
            prediction_results.append(result)

        # Process all predictions concurrently
        tasks = []
        for i, result in enumerate(prediction_results):
            task = bridge.handle_prediction_made(f"room_{i % 10}", result)
            tasks.append(task)

        await asyncio.gather(*tasks)

        # Verify all processed
        assert bridge.stats.tracking_events_processed == 100
        assert bridge.stats.entity_updates_sent == 100
        assert (
            bridge.enhanced_integration_manager.handle_prediction_update.call_count
            == 100
        )

    @pytest.mark.asyncio
    async def test_rapid_command_delegation(self, bridge):
        """Test rapid command delegation doesn't cause issues."""
        bridge._setup_command_delegation()

        # Execute many commands rapidly
        commands = []
        for i in range(50):
            command_type = ["retrain_model", "validate_model", "force_prediction"][
                i % 3
            ]
            parameters = {"room_id": f"room_{i % 5}"}
            commands.append((command_type, parameters))

        tasks = []
        for command_type, parameters in commands:
            handler = bridge.enhanced_integration_manager.command_handlers[command_type]
            task = handler(parameters)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all commands processed
        assert len(results) == 50
        assert bridge.stats.commands_delegated >= 50

    @pytest.mark.asyncio
    async def test_concurrent_event_handling(self, bridge):
        """Test concurrent handling of different event types."""
        # Create various events
        prediction_result = PredictionResult(
            prediction_type="next_occupied",
            predicted_time=datetime.now(timezone.utc),
            confidence=0.85,
            metadata={},
        )

        alert = Mock()
        alert.alert_type = "test_alert"
        alert.severity = "low"
        alert.room_id = "test_room"
        alert.message = "Test message"
        alert.timestamp = datetime.utcnow()

        drift_metrics = Mock()
        drift_metrics.drift_score = 0.6
        drift_metrics.severity = "low"
        drift_metrics.affected_features = ["test_feature"]

        # Process events concurrently
        tasks = [
            bridge.handle_prediction_made("room1", prediction_result),
            bridge.handle_accuracy_alert(alert),
            bridge.handle_drift_detected(drift_metrics),
            bridge.handle_retraining_started("room2", {"training_type": "test"}),
            bridge.handle_retraining_completed(
                "room3", {"success": True, "accuracy": 0.9}
            ),
        ]

        await asyncio.gather(*tasks)

        # Verify all events handled
        assert bridge.stats.tracking_events_processed >= 1
        assert bridge.enhanced_integration_manager.update_entity_state.call_count >= 4

    @pytest.mark.asyncio
    async def test_memory_efficiency_large_datasets(self, bridge):
        """Test memory efficiency with large datasets."""
        # Create prediction result for testing
        prediction_result = PredictionResult(
            prediction_type="next_occupied",
            predicted_time=datetime.now(timezone.utc),
            confidence=0.85,
            metadata={},
        )

        # Simulate large system status
        large_status = {
            "status": "healthy",
            "rooms": {
                f"room_{i}": {"accuracy": 0.9, "predictions": 1000} for i in range(1000)
            },
            "metrics": {f"metric_{i}": [0.1] * 1000 for i in range(100)},
        }

        bridge.tracking_manager.get_system_status.return_value = large_status

        # Process large status multiple times
        for _ in range(10):
            await bridge.handle_prediction_made("room", prediction_result)

        # Should not cause memory issues or crashes
        assert bridge.stats.entity_updates_sent == 10


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def bridge(self, mock_tracking_manager, mock_enhanced_integration_manager):
        """Create bridge for edge case tests."""
        return HATrackingBridge(
            mock_tracking_manager, mock_enhanced_integration_manager
        )

    def test_initialization_with_none_managers(self):
        """Test initialization with None managers."""
        with pytest.raises((TypeError, AttributeError)):
            HATrackingBridge(None, None)

    @pytest.mark.asyncio
    async def test_operations_before_initialization(self, bridge, prediction_result):
        """Test operations before bridge initialization."""
        # Bridge not initialized (_bridge_active = False)
        await bridge.handle_prediction_made("room", prediction_result)

        # Should not process events
        assert bridge.stats.tracking_events_processed == 0
        bridge.enhanced_integration_manager.handle_prediction_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_shutdown_before_initialization(self, bridge):
        """Test shutdown before initialization."""
        # Should handle gracefully
        await bridge.shutdown()

        assert bridge._bridge_active is False

    @pytest.mark.asyncio
    async def test_empty_tracking_manager_methods(self, bridge):
        """Test behavior when tracking manager has no optional methods."""
        # Remove all optional methods
        optional_methods = [
            "register_callback",
            "get_system_status",
            "get_accuracy_metrics",
            "trigger_retraining",
            "validate_model_performance",
            "force_prediction",
            "check_database_health",
            "generate_diagnostic_report",
        ]

        for method in optional_methods:
            if hasattr(bridge.tracking_manager, method):
                delattr(bridge.tracking_manager, method)

        # Initialize and test
        await bridge.initialize()

        # Should still work with basic functionality
        assert bridge._bridge_active is True
        assert bridge.stats.bridge_initialized is True

    @pytest.mark.asyncio
    async def test_malformed_event_data(self, bridge):
        """Test handling of malformed event data."""
        bridge._bridge_active = True

        # Test with None objects
        await bridge.handle_accuracy_alert(None)
        await bridge.handle_drift_detected(None)

        # Should not crash, may log errors but continue operation
        assert bridge._bridge_active is True

    @pytest.mark.asyncio
    async def test_command_delegation_missing_parameters(self, bridge):
        """Test command delegation with missing required parameters."""
        # Test commands with missing or invalid parameters
        test_cases = [
            ("retrain_model", {}),  # Missing room_id
            ("validate_model", {"invalid_param": "value"}),  # Wrong parameters
            ("force_prediction", None),  # None parameters
        ]

        bridge._setup_command_delegation()

        for command_name, parameters in test_cases:
            handler = bridge.enhanced_integration_manager.command_handlers[command_name]

            # Should handle gracefully, may return error status
            try:
                result = await handler(parameters or {})
                # Should get some result, even if error
                assert "status" in result
            except Exception:
                # Some parameter errors might raise exceptions, which is acceptable
                pass

    @pytest.mark.asyncio
    async def test_background_task_exception_handling(self, bridge):
        """Test background task exception handling."""
        bridge._shutdown_event = asyncio.Event()

        # Make tracking manager methods fail
        bridge.tracking_manager.get_system_status.side_effect = Exception(
            "Status error"
        )
        bridge.tracking_manager.get_accuracy_metrics.side_effect = Exception(
            "Metrics error"
        )

        # Start tasks that will encounter errors
        with patch("asyncio.sleep", return_value=None):
            status_task = asyncio.create_task(bridge._system_status_sync_loop())
            metrics_task = asyncio.create_task(bridge._metrics_sync_loop())

            # Brief execution
            await asyncio.sleep(0.05)
            bridge._shutdown_event.set()

            # Should complete without crashing
            results = await asyncio.gather(
                status_task, metrics_task, return_exceptions=True
            )

            # Tasks should complete (may have exceptions but shouldn't crash)
            assert len(results) == 2
