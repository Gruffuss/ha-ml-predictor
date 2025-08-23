"""
Comprehensive tests for the Enhanced Home Assistant Integration Manager.

This module tests the enhanced HA integration system including entity definitions,
service management, command handling, and comprehensive HA ecosystem integration.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from src.core.config import MQTTConfig, RoomConfig, TrackingConfig
from src.integration.enhanced_integration_manager import (
    CommandRequest,
    CommandResponse,
    EnhancedIntegrationError,
    EnhancedIntegrationManager,
    EnhancedIntegrationStats,
)
from src.models.base.predictor import PredictionResult


@pytest.mark.unit
class TestEnhancedIntegrationStats:
    """Test cases for EnhancedIntegrationStats."""

    def test_stats_initialization(self):
        """Test stats initialization with default values."""
        stats = EnhancedIntegrationStats()

        assert stats.entities_defined == 0
        assert stats.entities_published == 0
        assert stats.services_defined == 0
        assert stats.services_published == 0
        assert stats.commands_processed == 0
        assert stats.state_updates_sent == 0
        assert stats.last_entity_update is None
        assert stats.last_command_processed is None
        assert stats.integration_errors == 0
        assert stats.last_error is None

    def test_stats_with_values(self):
        """Test stats with specific values."""
        now = datetime.utcnow()
        stats = EnhancedIntegrationStats(
            entities_defined=10,
            entities_published=8,
            services_defined=5,
            services_published=4,
            commands_processed=20,
            state_updates_sent=50,
            last_entity_update=now,
            last_command_processed=now,
            integration_errors=2,
            last_error="Test error",
        )

        assert stats.entities_defined == 10
        assert stats.entities_published == 8
        assert stats.services_defined == 5
        assert stats.services_published == 4
        assert stats.commands_processed == 20
        assert stats.state_updates_sent == 50
        assert stats.last_entity_update == now
        assert stats.last_command_processed == now
        assert stats.integration_errors == 2
        assert stats.last_error == "Test error"


@pytest.mark.unit
class TestCommandRequest:
    """Test cases for CommandRequest."""

    def test_command_request_creation(self):
        """Test command request creation."""
        timestamp = datetime.utcnow()
        request = CommandRequest(
            command="test_command",
            parameters={"param1": "value1"},
            timestamp=timestamp,
            response_topic="test/response",
            correlation_id="abc123",
        )

        assert request.command == "test_command"
        assert request.parameters == {"param1": "value1"}
        assert request.timestamp == timestamp
        assert request.response_topic == "test/response"
        assert request.correlation_id == "abc123"

    def test_command_request_minimal(self):
        """Test command request with minimal parameters."""
        timestamp = datetime.utcnow()
        request = CommandRequest(
            command="test_command", parameters={"param1": "value1"}, timestamp=timestamp
        )

        assert request.command == "test_command"
        assert request.parameters == {"param1": "value1"}
        assert request.timestamp == timestamp
        assert request.response_topic is None
        assert request.correlation_id is None


@pytest.mark.unit
class TestCommandResponse:
    """Test cases for CommandResponse."""

    def test_command_response_success(self):
        """Test successful command response."""
        result = {"result": "success"}
        response = CommandResponse(success=True, result=result, correlation_id="abc123")

        assert response.success is True
        assert response.result == result
        assert response.error_message is None
        assert response.correlation_id == "abc123"
        assert isinstance(response.timestamp, datetime)

    def test_command_response_failure(self):
        """Test failed command response."""
        response = CommandResponse(
            success=False, error_message="Command failed", correlation_id="abc123"
        )

        assert response.success is False
        assert response.result is None
        assert response.error_message == "Command failed"
        assert response.correlation_id == "abc123"
        assert isinstance(response.timestamp, datetime)

    def test_command_response_default_timestamp(self):
        """Test command response with default timestamp."""
        response = CommandResponse(success=True)

        assert isinstance(response.timestamp, datetime)
        # Should be very recent
        assert (datetime.utcnow() - response.timestamp).total_seconds() < 1


@pytest.mark.unit
class TestEnhancedIntegrationManager:
    """Test cases for EnhancedIntegrationManager."""

    @pytest.fixture
    def mock_config(self):
        """Mock system configuration."""
        config = MagicMock()
        config.mqtt = MQTTConfig(
            broker="test-broker",
            port=1883,
            username="test",
            password="test",
            topic_prefix="test/occupancy",
            device_identifier="test-device",
            discovery_prefix="homeassistant",
        )
        config.rooms = {
            "test_room": RoomConfig(
                room_id="test_room",
                name="Test Room",
                sensors={"presence": {"main": "binary_sensor.test_presence"}},
            )
        }
        config.tracking = TrackingConfig()
        return config

    @pytest.fixture
    def mock_mqtt_integration_manager(self):
        """Mock MQTT integration manager."""
        manager = AsyncMock()
        manager.stats = MagicMock()
        manager.stats.initialized = True
        manager.initialize = AsyncMock()
        manager.mqtt_publisher = AsyncMock()
        manager.discovery_publisher = AsyncMock()
        return manager

    @pytest.fixture
    def mock_tracking_manager(self):
        """Mock tracking manager."""
        manager = AsyncMock()
        manager.trigger_retraining = AsyncMock()
        manager.validate_model_performance = AsyncMock()
        manager.on_model_retrained = AsyncMock()
        manager.on_training_failure = AsyncMock()
        manager.force_prediction = AsyncMock()
        return manager

    @pytest.fixture
    def mock_ha_entity_definitions(self):
        """Mock HA entity definitions."""
        definitions = AsyncMock()
        definitions.define_all_entities = MagicMock(return_value={})
        definitions.define_all_services = MagicMock(return_value={})
        definitions.publish_all_entities = AsyncMock(return_value={})
        definitions.publish_all_services = AsyncMock(return_value={})
        definitions.get_entity_definition = MagicMock()
        definitions.get_entity_stats = MagicMock(return_value={})
        return definitions

    @pytest.fixture
    def integration_manager(self, mock_mqtt_integration_manager, mock_tracking_manager):
        """Create integration manager for testing."""
        with patch(
            "src.integration.enhanced_integration_manager.get_config"
        ) as mock_get_config:
            mock_config = MagicMock()
            mock_config.mqtt = MagicMock()
            mock_config.rooms = {}
            mock_config.tracking = TrackingConfig()
            mock_get_config.return_value = mock_config

            return EnhancedIntegrationManager(
                mqtt_integration_manager=mock_mqtt_integration_manager,
                tracking_manager=mock_tracking_manager,
            )

    def test_initialization(self, mock_config):
        """Test EnhancedIntegrationManager initialization."""
        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_config,
        ):
            manager = EnhancedIntegrationManager()

            assert manager.config is mock_config
            assert manager.mqtt_config is mock_config.mqtt
            assert manager.rooms is mock_config.rooms
            assert isinstance(manager.tracking_config, TrackingConfig)
            assert manager.mqtt_integration_manager is None
            assert manager.tracking_manager is None
            assert manager.ha_entity_definitions is None
            assert len(manager.command_handlers) == 0
            assert len(manager.entity_states) == 0
            assert len(manager.last_state_update) == 0
            assert len(manager._background_tasks) == 0
            assert manager._enhanced_integration_active is False
            assert isinstance(manager.stats, EnhancedIntegrationStats)

    def test_initialization_with_managers(
        self, mock_config, mock_mqtt_integration_manager, mock_tracking_manager
    ):
        """Test initialization with provided managers."""
        notification_callbacks = [MagicMock()]

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_config,
        ):
            manager = EnhancedIntegrationManager(
                mqtt_integration_manager=mock_mqtt_integration_manager,
                tracking_manager=mock_tracking_manager,
                notification_callbacks=notification_callbacks,
            )

            assert manager.mqtt_integration_manager is mock_mqtt_integration_manager
            assert manager.tracking_manager is mock_tracking_manager
            assert manager.notification_callbacks == notification_callbacks

    @pytest.mark.asyncio
    async def test_initialize_success(
        self,
        integration_manager,
        mock_mqtt_integration_manager,
        mock_ha_entity_definitions,
    ):
        """Test successful initialization."""
        # Setup MQTT integration manager with discovery publisher
        mock_mqtt_integration_manager.discovery_publisher = AsyncMock()

        with patch(
            "src.integration.enhanced_integration_manager.HAEntityDefinitions",
            return_value=mock_ha_entity_definitions,
        ), patch.object(
            integration_manager, "_define_and_publish_entities", new_callable=AsyncMock
        ) as mock_define_entities, patch.object(
            integration_manager, "_define_and_publish_services", new_callable=AsyncMock
        ) as mock_define_services, patch.object(
            integration_manager, "_setup_command_handlers"
        ) as mock_setup_handlers, patch.object(
            integration_manager, "_start_background_tasks", new_callable=AsyncMock
        ) as mock_start_tasks:

            await integration_manager.initialize()

            assert integration_manager._enhanced_integration_active is True
            assert (
                integration_manager.ha_entity_definitions is mock_ha_entity_definitions
            )
            mock_define_entities.assert_called_once()
            mock_define_services.assert_called_once()
            mock_setup_handlers.assert_called_once()
            mock_start_tasks.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_without_mqtt_manager(self, integration_manager):
        """Test initialization without MQTT integration manager."""
        integration_manager.mqtt_integration_manager = None

        await integration_manager.initialize()

        assert integration_manager.ha_entity_definitions is None
        assert integration_manager._enhanced_integration_active is True

    @pytest.mark.asyncio
    async def test_initialize_uninitialized_mqtt_manager(
        self, integration_manager, mock_mqtt_integration_manager
    ):
        """Test initialization with uninitialized MQTT manager."""
        mock_mqtt_integration_manager.stats.initialized = False

        await integration_manager.initialize()

        # Should initialize MQTT manager first
        mock_mqtt_integration_manager.initialize.assert_called_once()

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

            assert integration_manager.stats.integration_errors == 1
            assert integration_manager.stats.last_error == "Task startup failed"

    @pytest.mark.asyncio
    async def test_shutdown(self, integration_manager):
        """Test shutdown process."""
        # Setup background tasks
        task1 = AsyncMock()
        task2 = AsyncMock()
        task1.done.return_value = False
        task2.done.return_value = False
        integration_manager._background_tasks = [task1, task2]
        integration_manager._enhanced_integration_active = True

        await integration_manager.shutdown()

        assert integration_manager._enhanced_integration_active is False
        assert integration_manager._shutdown_event.is_set()
        task1.cancel.assert_called_once()
        task2.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_update_entity_state_success(
        self,
        integration_manager,
        mock_mqtt_integration_manager,
        mock_ha_entity_definitions,
    ):
        """Test successful entity state update."""
        integration_manager._enhanced_integration_active = True
        integration_manager.ha_entity_definitions = mock_ha_entity_definitions
        integration_manager.mqtt_integration_manager = mock_mqtt_integration_manager

        # Mock entity definition
        mock_entity_config = MagicMock()
        mock_entity_config.state_topic = "test/state"
        mock_ha_entity_definitions.get_entity_definition.return_value = (
            mock_entity_config
        )

        # Mock successful publish
        mock_publish_result = MagicMock()
        mock_publish_result.success = True
        mock_mqtt_integration_manager.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await integration_manager.update_entity_state(
            "test_entity", "test_state", {"attr1": "value1"}
        )

        assert result is True
        assert integration_manager.entity_states["test_entity"] == "test_state"
        assert "test_entity" in integration_manager.last_state_update
        assert integration_manager.stats.state_updates_sent == 1
        assert integration_manager.stats.last_entity_update is not None

        # Verify publish was called correctly
        mock_mqtt_integration_manager.mqtt_publisher.publish_json.assert_called_once()
        call_args = mock_mqtt_integration_manager.mqtt_publisher.publish_json.call_args
        assert call_args[1]["topic"] == "test/state"
        assert call_args[1]["data"]["state"] == "test_state"
        assert call_args[1]["data"]["attr1"] == "value1"

    @pytest.mark.asyncio
    async def test_update_entity_state_not_active(self, integration_manager):
        """Test entity state update when not active."""
        integration_manager._enhanced_integration_active = False

        result = await integration_manager.update_entity_state(
            "test_entity", "test_state"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_entity_state_no_entity_definition(
        self, integration_manager, mock_ha_entity_definitions
    ):
        """Test entity state update with no entity definition."""
        integration_manager._enhanced_integration_active = True
        integration_manager.ha_entity_definitions = mock_ha_entity_definitions

        # Mock no entity definition found
        mock_ha_entity_definitions.get_entity_definition.return_value = None

        result = await integration_manager.update_entity_state(
            "test_entity", "test_state"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_update_entity_state_publish_failure(
        self,
        integration_manager,
        mock_mqtt_integration_manager,
        mock_ha_entity_definitions,
    ):
        """Test entity state update with publish failure."""
        integration_manager._enhanced_integration_active = True
        integration_manager.ha_entity_definitions = mock_ha_entity_definitions
        integration_manager.mqtt_integration_manager = mock_mqtt_integration_manager

        # Mock entity definition
        mock_entity_config = MagicMock()
        mock_entity_config.state_topic = "test/state"
        mock_ha_entity_definitions.get_entity_definition.return_value = (
            mock_entity_config
        )

        # Mock failed publish
        mock_publish_result = MagicMock()
        mock_publish_result.success = False
        mock_mqtt_integration_manager.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await integration_manager.update_entity_state(
            "test_entity", "test_state"
        )

        assert result is False

    @pytest.mark.asyncio
    async def test_process_command_success(self, integration_manager):
        """Test successful command processing."""
        # Setup command handler
        mock_handler = AsyncMock(return_value={"result": "success"})
        integration_manager.command_handlers["test_command"] = mock_handler

        request = CommandRequest(
            command="test_command",
            parameters={"param1": "value1"},
            timestamp=datetime.utcnow(),
            correlation_id="abc123",
        )

        response = await integration_manager.process_command(request)

        assert response.success is True
        assert response.result == {"result": "success"}
        assert response.correlation_id == "abc123"
        assert integration_manager.stats.commands_processed == 1
        assert integration_manager.stats.last_command_processed is not None

        mock_handler.assert_called_once_with({"param1": "value1"})

    @pytest.mark.asyncio
    async def test_process_command_sync_handler(self, integration_manager):
        """Test command processing with synchronous handler."""
        # Setup synchronous command handler
        mock_handler = MagicMock(return_value={"result": "sync_success"})
        integration_manager.command_handlers["test_command"] = mock_handler

        request = CommandRequest(
            command="test_command",
            parameters={"param1": "value1"},
            timestamp=datetime.utcnow(),
        )

        response = await integration_manager.process_command(request)

        assert response.success is True
        assert response.result == {"result": "sync_success"}
        mock_handler.assert_called_once_with({"param1": "value1"})

    @pytest.mark.asyncio
    async def test_process_command_unknown_command(self, integration_manager):
        """Test processing unknown command."""
        request = CommandRequest(
            command="unknown_command",
            parameters={},
            timestamp=datetime.utcnow(),
            correlation_id="abc123",
        )

        response = await integration_manager.process_command(request)

        assert response.success is False
        assert "Unknown command" in response.error_message
        assert response.correlation_id == "abc123"

    @pytest.mark.asyncio
    async def test_process_command_handler_exception(self, integration_manager):
        """Test command processing with handler exception."""
        # Setup command handler that raises exception
        mock_handler = AsyncMock(side_effect=Exception("Handler failed"))
        integration_manager.command_handlers["test_command"] = mock_handler

        request = CommandRequest(
            command="test_command",
            parameters={},
            timestamp=datetime.utcnow(),
            correlation_id="abc123",
        )

        response = await integration_manager.process_command(request)

        assert response.success is False
        assert "Error processing command" in response.error_message
        assert response.correlation_id == "abc123"
        assert integration_manager.stats.integration_errors == 1

    @pytest.mark.asyncio
    async def test_handle_prediction_update(self, integration_manager):
        """Test handling prediction updates."""
        integration_manager._enhanced_integration_active = True

        prediction_result = PredictionResult(
            predicted_time=datetime.utcnow() + timedelta(minutes=30),
            transition_type="occupied_to_vacant",
            confidence_score=0.85,
            time_until_human="30 minutes",
            prediction_reliability="high",
            model_used="lstm",
            alternatives=[],
        )

        with patch.object(
            integration_manager, "update_entity_state", new_callable=AsyncMock
        ) as mock_update:

            await integration_manager.handle_prediction_update(
                "test_room", prediction_result
            )

            # Verify multiple entity updates were called
            assert mock_update.call_count == 5
            call_args_list = mock_update.call_args_list

            # Check prediction entity update
            prediction_call = call_args_list[0]
            assert prediction_call[0] == ("test_room_prediction", "occupied_to_vacant")
            assert "predicted_time" in prediction_call[1]["attributes"]

            # Check confidence entity update
            confidence_call = call_args_list[1]
            assert confidence_call[0] == ("test_room_confidence", 85.0)

    @pytest.mark.asyncio
    async def test_handle_prediction_update_not_active(self, integration_manager):
        """Test handling prediction updates when not active."""
        integration_manager._enhanced_integration_active = False

        prediction_result = PredictionResult(
            predicted_time=datetime.utcnow(),
            transition_type="vacant_to_occupied",
            confidence_score=0.8,
            time_until_human="15 minutes",
            prediction_reliability="medium",
            model_used="xgboost",
            alternatives=[],
        )

        with patch.object(
            integration_manager, "update_entity_state", new_callable=AsyncMock
        ) as mock_update:

            await integration_manager.handle_prediction_update(
                "test_room", prediction_result
            )

            # Should not update entities when not active
            mock_update.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_system_status_update(self, integration_manager):
        """Test handling system status updates."""
        integration_manager._enhanced_integration_active = True

        system_status = {
            "system_status": "running",
            "uptime_seconds": 3600,
            "total_predictions_made": 100,
            "average_accuracy_percent": 85.5,
            "active_alerts": 0,
            "database_connected": True,
            "mqtt_connected": True,
            "tracking_active": True,
            "model_training_active": False,
        }

        with patch.object(
            integration_manager, "update_entity_state", new_callable=AsyncMock
        ) as mock_update:

            await integration_manager.handle_system_status_update(system_status)

            # Verify system status and individual status entities were updated
            assert mock_update.call_count == 9  # 1 main + 8 individual entities

            # Check main system status call
            main_call = mock_update.call_args_list[0]
            assert main_call[0] == ("system_status", "running")

    @pytest.mark.asyncio
    async def test_handle_system_status_update_not_active(self, integration_manager):
        """Test handling system status updates when not active."""
        integration_manager._enhanced_integration_active = False

        with patch.object(
            integration_manager, "update_entity_state", new_callable=AsyncMock
        ) as mock_update:

            await integration_manager.handle_system_status_update({"status": "test"})

            mock_update.assert_not_called()

    def test_get_integration_stats(
        self,
        integration_manager,
        mock_mqtt_integration_manager,
        mock_ha_entity_definitions,
    ):
        """Test getting integration statistics."""
        integration_manager.mqtt_integration_manager = mock_mqtt_integration_manager
        integration_manager.ha_entity_definitions = mock_ha_entity_definitions
        integration_manager.entity_states["entity1"] = "state1"
        integration_manager.command_handlers["cmd1"] = MagicMock()

        # Mock MQTT stats
        mock_mqtt_integration_manager.stats.__dict__ = {"mqtt_stat": "value"}

        # Mock entity stats
        mock_ha_entity_definitions.get_entity_stats.return_value = {
            "entity_stat": "value"
        }

        stats = integration_manager.get_integration_stats()

        assert "enhanced_integration" in stats
        assert "mqtt_integration" in stats
        assert stats["entity_states_count"] == 1
        assert stats["command_handlers_count"] == 1
        assert stats["active"] is False  # Not initialized in this test

    def test_get_integration_stats_minimal(self, integration_manager):
        """Test getting integration stats with minimal setup."""
        stats = integration_manager.get_integration_stats()

        assert "enhanced_integration" in stats
        assert "mqtt_integration" in stats
        assert stats["entity_states_count"] == 0
        assert stats["command_handlers_count"] == 0
        assert stats["active"] is False

    @pytest.mark.asyncio
    async def test_define_and_publish_entities(
        self, integration_manager, mock_ha_entity_definitions
    ):
        """Test defining and publishing entities."""
        integration_manager.ha_entity_definitions = mock_ha_entity_definitions

        # Mock entity definitions
        mock_entities = {"entity1": MagicMock(), "entity2": MagicMock()}
        mock_ha_entity_definitions.define_all_entities.return_value = mock_entities

        # Mock publish results
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result2 = MagicMock()
        mock_result2.success = False
        mock_ha_entity_definitions.publish_all_entities.return_value = {
            "entity1": mock_result1,
            "entity2": mock_result2,
        }

        await integration_manager._define_and_publish_entities()

        assert integration_manager.stats.entities_defined == 2
        assert integration_manager.stats.entities_published == 1  # Only successful ones

    @pytest.mark.asyncio
    async def test_define_and_publish_services(
        self, integration_manager, mock_ha_entity_definitions
    ):
        """Test defining and publishing services."""
        integration_manager.ha_entity_definitions = mock_ha_entity_definitions

        # Mock service definitions
        mock_services = {"service1": MagicMock(), "service2": MagicMock()}
        mock_ha_entity_definitions.define_all_services.return_value = mock_services

        # Mock publish results
        mock_result1 = MagicMock()
        mock_result1.success = True
        mock_result2 = MagicMock()
        mock_result2.success = True
        mock_ha_entity_definitions.publish_all_services.return_value = {
            "service1": mock_result1,
            "service2": mock_result2,
        }

        await integration_manager._define_and_publish_services()

        assert integration_manager.stats.services_defined == 2
        assert integration_manager.stats.services_published == 2

    def test_setup_command_handlers(self, integration_manager):
        """Test command handlers setup."""
        integration_manager._setup_command_handlers()

        # Verify all expected handlers were registered
        expected_handlers = [
            "retrain_model",
            "validate_model",
            "restart_system",
            "refresh_discovery",
            "reset_statistics",
            "generate_diagnostic",
            "check_database",
            "force_prediction",
            "prediction_enable",
            "mqtt_enable",
            "set_interval",
            "set_log_level",
        ]

        for handler in expected_handlers:
            assert handler in integration_manager.command_handlers

    @pytest.mark.asyncio
    async def test_start_background_tasks(self, integration_manager):
        """Test starting background tasks."""
        with patch("asyncio.create_task") as mock_create_task:
            mock_task1 = AsyncMock()
            mock_task2 = AsyncMock()
            mock_create_task.side_effect = [mock_task1, mock_task2]

            await integration_manager._start_background_tasks()

            assert len(integration_manager._background_tasks) == 2
            assert mock_create_task.call_count == 2

    @pytest.mark.asyncio
    async def test_command_processing_loop(self, integration_manager):
        """Test command processing loop."""
        # Create a command request
        request = CommandRequest(
            command="test_command", parameters={}, timestamp=datetime.utcnow()
        )

        # Mock queue get to return request then timeout
        async def mock_queue_get():
            if not hasattr(mock_queue_get, "called"):
                mock_queue_get.called = True
                return request
            else:
                raise asyncio.TimeoutError()

        integration_manager.command_queue.get = mock_queue_get

        with patch.object(
            integration_manager, "process_command", new_callable=AsyncMock
        ) as mock_process:
            mock_response = CommandResponse(success=True)
            mock_process.return_value = mock_response

            # Run loop briefly
            try:
                await asyncio.wait_for(
                    integration_manager._command_processing_loop(), timeout=0.1
                )
            except asyncio.TimeoutError:
                pass

            mock_process.assert_called_once_with(request)

    @pytest.mark.asyncio
    async def test_entity_monitoring_loop(self, integration_manager):
        """Test entity monitoring loop."""
        with patch.object(
            integration_manager, "_check_entity_availability", new_callable=AsyncMock
        ) as mock_check_availability, patch.object(
            integration_manager, "_cleanup_old_responses", new_callable=AsyncMock
        ) as mock_cleanup, patch(
            "asyncio.sleep", side_effect=asyncio.TimeoutError()
        ):

            # Run loop briefly
            try:
                await integration_manager._entity_monitoring_loop()
            except asyncio.TimeoutError:
                pass

            mock_check_availability.assert_called()
            mock_cleanup.assert_called()

    @pytest.mark.asyncio
    async def test_check_entity_availability(
        self, integration_manager, mock_mqtt_integration_manager
    ):
        """Test entity availability checking."""
        integration_manager.mqtt_integration_manager = mock_mqtt_integration_manager
        integration_manager._enhanced_integration_active = True

        # Mock discovery publisher
        mock_discovery_publisher = AsyncMock()
        mock_mqtt_integration_manager.discovery_publisher = mock_discovery_publisher

        await integration_manager._check_entity_availability()

        mock_discovery_publisher.publish_device_availability.assert_called_once_with(
            online=True
        )

    @pytest.mark.asyncio
    async def test_cleanup_old_responses(self, integration_manager):
        """Test cleanup of old command responses."""
        # Add old and recent responses
        old_response = CommandResponse(
            success=True, timestamp=datetime.utcnow() - timedelta(hours=2)
        )
        recent_response = CommandResponse(success=True)

        integration_manager.command_responses = {
            "old_id": old_response,
            "recent_id": recent_response,
        }

        await integration_manager._cleanup_old_responses()

        # Only recent response should remain
        assert "old_id" not in integration_manager.command_responses
        assert "recent_id" in integration_manager.command_responses

    @pytest.mark.asyncio
    async def test_handle_retrain_model_command(
        self, integration_manager, mock_tracking_manager
    ):
        """Test retrain model command handler."""
        integration_manager.tracking_manager = mock_tracking_manager
        mock_tracking_manager.trigger_retraining.return_value = {"status": "started"}

        result = await integration_manager._handle_retrain_model(
            {"room_id": "test_room", "force": True}
        )

        assert result["status"] == "success"
        mock_tracking_manager.trigger_retraining.assert_called_once_with(
            room_id="test_room", force=True
        )

    @pytest.mark.asyncio
    async def test_handle_retrain_model_no_tracking_manager(self, integration_manager):
        """Test retrain model command without tracking manager."""
        integration_manager.tracking_manager = None

        result = await integration_manager._handle_retrain_model({})

        assert result["status"] == "error"
        assert "Tracking manager not available" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_validate_model_command(
        self, integration_manager, mock_tracking_manager
    ):
        """Test validate model command handler."""
        integration_manager.tracking_manager = mock_tracking_manager
        mock_tracking_manager.validate_model_performance.return_value = {
            "accuracy": 85.0
        }

        result = await integration_manager._handle_validate_model(
            {"room_id": "test_room", "days": 7}
        )

        assert result["status"] == "success"
        mock_tracking_manager.validate_model_performance.assert_called_once_with(
            room_id="test_room", validation_days=7
        )

    @pytest.mark.asyncio
    async def test_handle_restart_system_command(self, integration_manager):
        """Test restart system command handler."""
        result = await integration_manager._handle_restart_system({})

        assert result["status"] == "acknowledged"
        assert "Restart request received" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_refresh_discovery_command(
        self, integration_manager, mock_mqtt_integration_manager
    ):
        """Test refresh discovery command handler."""
        integration_manager.mqtt_integration_manager = mock_mqtt_integration_manager

        # Mock discovery publisher
        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.refresh_discovery.return_value = {
            "entity1": MagicMock(success=True),
            "entity2": MagicMock(success=False),
        }
        mock_mqtt_integration_manager.discovery_publisher = mock_discovery_publisher

        result = await integration_manager._handle_refresh_discovery({})

        assert result["status"] == "success"
        assert result["entities_refreshed"] == 1
        assert result["total_entities"] == 2

    @pytest.mark.asyncio
    async def test_handle_reset_statistics_command(self, integration_manager):
        """Test reset statistics command handler."""
        # Test without confirmation
        result = await integration_manager._handle_reset_statistics({"confirm": False})
        assert result["status"] == "error"

        # Test with confirmation
        result = await integration_manager._handle_reset_statistics({"confirm": True})
        assert result["status"] == "success"

    @pytest.mark.asyncio
    async def test_handle_generate_diagnostic_command(self, integration_manager):
        """Test generate diagnostic command handler."""
        result = await integration_manager._handle_generate_diagnostic(
            {"include_logs": True, "include_metrics": True}
        )

        assert result["status"] == "success"
        assert "diagnostic_data" in result
        assert "timestamp" in result["diagnostic_data"]
        assert "integration_stats" in result["diagnostic_data"]
        assert "performance" in result["diagnostic_data"]

    @pytest.mark.asyncio
    async def test_handle_generate_diagnostic_minimal(self, integration_manager):
        """Test generate diagnostic command with minimal options."""
        result = await integration_manager._handle_generate_diagnostic(
            {"include_logs": False, "include_metrics": False}
        )

        assert result["status"] == "success"
        assert "diagnostic_data" in result
        assert "performance" not in result["diagnostic_data"]

    @pytest.mark.asyncio
    async def test_handle_check_database_command(self, integration_manager):
        """Test check database command handler."""
        result = await integration_manager._handle_check_database({})

        assert result["status"] == "success"
        assert result["database_status"] == "healthy"

    @pytest.mark.asyncio
    async def test_handle_force_prediction_command(
        self, integration_manager, mock_tracking_manager
    ):
        """Test force prediction command handler."""
        integration_manager.tracking_manager = mock_tracking_manager
        mock_tracking_manager.force_prediction.return_value = {"prediction": "result"}

        result = await integration_manager._handle_force_prediction(
            {"room_id": "test_room"}
        )

        assert result["status"] == "success"
        mock_tracking_manager.force_prediction.assert_called_once_with(
            room_id="test_room"
        )

    @pytest.mark.asyncio
    async def test_handle_force_prediction_no_room_id(self, integration_manager):
        """Test force prediction command without room ID."""
        result = await integration_manager._handle_force_prediction({})

        assert result["status"] == "error"
        assert "room_id required" in result["message"]

    @pytest.mark.asyncio
    async def test_handle_prediction_enable_command(self, integration_manager):
        """Test prediction enable command handler."""
        result = await integration_manager._handle_prediction_enable({"enabled": True})

        assert result["status"] == "success"
        assert result["prediction_enabled"] is True

    @pytest.mark.asyncio
    async def test_handle_mqtt_enable_command(self, integration_manager):
        """Test MQTT enable command handler."""
        result = await integration_manager._handle_mqtt_enable({"enabled": False})

        assert result["status"] == "success"
        assert result["mqtt_enabled"] is False

    @pytest.mark.asyncio
    async def test_handle_set_interval_command(self, integration_manager):
        """Test set interval command handler."""
        result = await integration_manager._handle_set_interval({"interval": 600})

        assert result["status"] == "success"
        assert result["prediction_interval"] == 600

    @pytest.mark.asyncio
    async def test_handle_set_log_level_command(self, integration_manager):
        """Test set log level command handler."""
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            result = await integration_manager._handle_set_log_level(
                {"log_level": "DEBUG"}
            )

            assert result["status"] == "success"
            assert result["log_level"] == "DEBUG"
            mock_logger.setLevel.assert_called()


@pytest.mark.unit
class TestEnhancedIntegrationError:
    """Test cases for EnhancedIntegrationError."""

    def test_enhanced_integration_error(self):
        """Test EnhancedIntegrationError creation."""
        error = EnhancedIntegrationError("Test error message")

        assert "Test error message" in str(error)
        assert error.error_code == "ENHANCED_INTEGRATION_ERROR"
        assert error.severity.name == "MEDIUM"

    def test_enhanced_integration_error_with_severity(self):
        """Test EnhancedIntegrationError with custom severity."""
        from src.core.exceptions import ErrorSeverity

        error = EnhancedIntegrationError(
            "Critical error", severity=ErrorSeverity.CRITICAL
        )

        assert str(error) == "Critical error"
        assert error.severity == ErrorSeverity.CRITICAL


@pytest.mark.integration
class TestEnhancedIntegrationManagerIntegration:
    """Integration test cases for enhanced integration manager."""

    @pytest.mark.asyncio
    async def test_full_integration_workflow(self, mock_config):
        """Test complete integration workflow."""
        # Create mock managers
        mock_mqtt_manager = AsyncMock()
        mock_mqtt_manager.stats = MagicMock()
        mock_mqtt_manager.stats.initialized = True
        mock_mqtt_manager.discovery_publisher = AsyncMock()

        mock_tracking_manager = AsyncMock()

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_config,
        ), patch(
            "src.integration.enhanced_integration_manager.HAEntityDefinitions"
        ) as mock_ha_definitions_class:

            mock_ha_definitions = AsyncMock()
            mock_ha_definitions.define_all_entities.return_value = {
                "entity1": MagicMock()
            }
            mock_ha_definitions.define_all_services.return_value = {
                "service1": MagicMock()
            }
            mock_ha_definitions.publish_all_entities.return_value = {
                "entity1": MagicMock(success=True)
            }
            mock_ha_definitions.publish_all_services.return_value = {
                "service1": MagicMock(success=True)
            }
            mock_ha_definitions.get_entity_stats.return_value = {}
            mock_ha_definitions_class.return_value = mock_ha_definitions

            # Create and initialize manager
            manager = EnhancedIntegrationManager(
                mqtt_integration_manager=mock_mqtt_manager,
                tracking_manager=mock_tracking_manager,
            )

            # Test full lifecycle
            await manager.initialize()
            assert manager._enhanced_integration_active is True

            # Test entity state update
            mock_entity_config = MagicMock()
            mock_entity_config.state_topic = "test/state"
            mock_ha_definitions.get_entity_definition.return_value = mock_entity_config

            mock_publish_result = MagicMock()
            mock_publish_result.success = True
            mock_mqtt_manager.mqtt_publisher.publish_json.return_value = (
                mock_publish_result
            )

            result = await manager.update_entity_state("test_entity", "test_state")
            assert result is True

            # Test command processing
            manager.command_handlers["test_cmd"] = AsyncMock(
                return_value={"success": True}
            )
            request = CommandRequest(
                command="test_cmd", parameters={}, timestamp=datetime.utcnow()
            )
            response = await manager.process_command(request)
            assert response.success is True

            # Test shutdown
            await manager.shutdown()
            assert manager._enhanced_integration_active is False

    @pytest.mark.asyncio
    async def test_background_task_lifecycle(self, mock_config):
        """Test background task lifecycle management."""
        mock_mqtt_manager = AsyncMock()
        mock_mqtt_manager.stats = MagicMock()
        mock_mqtt_manager.stats.initialized = True

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_config,
        ), patch("asyncio.create_task") as mock_create_task:

            mock_task1 = AsyncMock()
            mock_task2 = AsyncMock()
            mock_task1.done.return_value = False
            mock_task2.done.return_value = False
            mock_create_task.side_effect = [mock_task1, mock_task2]

            manager = EnhancedIntegrationManager(
                mqtt_integration_manager=mock_mqtt_manager
            )

            # Initialize should start background tasks
            await manager.initialize()
            assert len(manager._background_tasks) == 2

            # Shutdown should clean up tasks
            await manager.shutdown()
            mock_task1.cancel.assert_called_once()
            mock_task2.cancel.assert_called_once()
