"""
System-level integration tests for the occupancy prediction system.

This module tests the complete system integration including main system orchestration,
training integration, enhanced HA integration, and end-to-end workflows.
"""

import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import MQTTConfig, RoomConfig, TrackingConfig, get_config
from src.integration.enhanced_integration_manager import EnhancedIntegrationManager
from src.integration.ha_entity_definitions import HAEntityDefinitions
from src.main_system import OccupancyPredictionSystem
from src.models.base.predictor import PredictionResult
from src.models.training_integration import TrainingIntegrationManager


@pytest.mark.integration
class TestSystemLevelIntegration:
    """System-level integration test cases."""

    @pytest.fixture
    def mock_complete_system_config(self):
        """Mock complete system configuration."""
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
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={
                    "presence": {"main": "binary_sensor.living_room_presence"},
                    "temperature": "sensor.living_room_temperature",
                },
            ),
            "bedroom": RoomConfig(
                room_id="bedroom",
                name="Bedroom",
                sensors={
                    "presence": {"main": "binary_sensor.bedroom_presence"},
                    "door": "binary_sensor.bedroom_door",
                },
            ),
        }
        config.tracking = TrackingConfig()
        config.api = MagicMock()
        config.api.enabled = True
        config.api.host = "127.0.0.1"
        config.api.port = 8000
        return config

    @pytest.fixture
    def mock_system_components(self):
        """Mock all system components."""
        # Database manager
        mock_database_manager = AsyncMock()
        mock_database_manager.health_check = AsyncMock(
            return_value={"status": "healthy"}
        )

        # MQTT integration manager
        mock_mqtt_manager = AsyncMock()
        mock_mqtt_manager.initialize = AsyncMock()
        mock_mqtt_manager.cleanup = AsyncMock()
        mock_mqtt_manager.stats = MagicMock()
        mock_mqtt_manager.stats.initialized = True
        mock_mqtt_manager.discovery_publisher = AsyncMock()
        mock_mqtt_manager.mqtt_publisher = AsyncMock()

        # Tracking manager
        mock_tracking_manager = AsyncMock()
        mock_tracking_manager.initialize = AsyncMock()
        mock_tracking_manager.stop_tracking = AsyncMock()
        mock_tracking_manager.get_api_server_status = MagicMock(
            return_value={
                "enabled": True,
                "running": True,
                "host": "127.0.0.1",
                "port": 8000,
            }
        )
        mock_tracking_manager.add_accuracy_callback = MagicMock()
        mock_tracking_manager.add_drift_callback = MagicMock()
        mock_tracking_manager.add_performance_callback = MagicMock()
        mock_tracking_manager.trigger_retraining = AsyncMock()
        mock_tracking_manager.validate_model_performance = AsyncMock()
        mock_tracking_manager.force_prediction = AsyncMock()
        mock_tracking_manager.register_model = MagicMock()

        # Training pipeline
        mock_training_pipeline = AsyncMock()
        mock_training_pipeline.run_retraining_pipeline = AsyncMock()
        mock_training_pipeline.get_model_registry = MagicMock(return_value={})

        # Config manager
        mock_config_manager = MagicMock()
        env_config = MagicMock()
        env_config.quality_thresholds.min_accuracy_threshold = 0.8
        env_config.quality_thresholds.max_error_threshold_minutes = 15.0
        mock_config_manager.get_environment_config.return_value = env_config
        mock_config_manager.set_current_profile = MagicMock()

        return {
            "database_manager": mock_database_manager,
            "mqtt_manager": mock_mqtt_manager,
            "tracking_manager": mock_tracking_manager,
            "training_pipeline": mock_training_pipeline,
            "config_manager": mock_config_manager,
        }

    @pytest.mark.asyncio
    async def test_complete_system_startup_shutdown_cycle(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test complete system startup and shutdown cycle."""
        with patch(
            "src.main_system.get_config", return_value=mock_complete_system_config
        ), patch(
            "src.main_system.get_database_manager",
            return_value=mock_system_components["database_manager"],
        ), patch(
            "src.main_system.MQTTIntegrationManager",
            return_value=mock_system_components["mqtt_manager"],
        ), patch(
            "src.main_system.TrackingManager",
            return_value=mock_system_components["tracking_manager"],
        ):

            # Create and initialize system
            system = OccupancyPredictionSystem()
            await system.initialize()

            # Verify system is running
            assert system.running is True
            assert system.database_manager is mock_system_components["database_manager"]
            assert system.mqtt_manager is mock_system_components["mqtt_manager"]
            assert system.tracking_manager is mock_system_components["tracking_manager"]

            # Verify initialization sequence
            mock_system_components["mqtt_manager"].initialize.assert_called_once()
            mock_system_components["tracking_manager"].initialize.assert_called_once()

            # Test shutdown
            await system.shutdown()

            # Verify shutdown sequence
            assert system.running is False
            mock_system_components[
                "tracking_manager"
            ].stop_tracking.assert_called_once()
            mock_system_components["mqtt_manager"].cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_training_integration_with_tracking_system(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test training integration with tracking system."""
        tracking_manager = mock_system_components["tracking_manager"]
        training_pipeline = mock_system_components["training_pipeline"]
        config_manager = mock_system_components["config_manager"]

        # Create training integration manager
        integration_manager = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=config_manager,
        )

        # Initialize with mocked background tasks
        with patch.object(
            integration_manager, "_start_background_tasks", new_callable=AsyncMock
        ), patch.object(
            integration_manager, "_register_tracking_callbacks", new_callable=AsyncMock
        ):
            await integration_manager.initialize()

        # Test accuracy degradation trigger
        await integration_manager._on_accuracy_degradation(
            "living_room", {"accuracy_rate": 70.0, "mean_error_minutes": 20.0}
        )

        # Verify retraining was queued
        assert len(integration_manager._training_queue) == 1
        request = integration_manager._training_queue[0]
        assert request["room_id"] == "living_room"
        assert "accuracy_degradation" in request["trigger_reason"]

        # Test drift detection trigger
        await integration_manager._on_drift_detected(
            "bedroom",
            {
                "drift_severity": "MAJOR",
                "overall_drift_score": 0.8,
                "retraining_recommended": True,
            },
        )

        # Verify drift-based retraining was queued
        assert len(integration_manager._training_queue) == 2
        drift_request = integration_manager._training_queue[1]
        assert drift_request["room_id"] == "bedroom"
        assert drift_request["strategy"] == "full_retrain"

        # Test manual training request
        result = await integration_manager.request_manual_training(
            room_id="living_room", strategy="adaptive", reason="user_request"
        )
        assert result is True

        # Cleanup
        await integration_manager.shutdown()

    @pytest.mark.asyncio
    async def test_enhanced_ha_integration_with_mqtt_system(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test enhanced HA integration with MQTT system."""
        mqtt_manager = mock_system_components["mqtt_manager"]
        tracking_manager = mock_system_components["tracking_manager"]

        # Create enhanced integration manager
        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_complete_system_config,
        ):
            enhanced_manager = EnhancedIntegrationManager(
                mqtt_integration_manager=mqtt_manager,
                tracking_manager=tracking_manager,
            )

        # Mock HA entity definitions
        with patch(
            "src.integration.enhanced_integration_manager.HAEntityDefinitions"
        ) as mock_ha_definitions_class:
            mock_ha_definitions = AsyncMock()
            mock_ha_definitions.define_all_entities.return_value = {
                "living_room_prediction": MagicMock(),
                "bedroom_prediction": MagicMock(),
                "system_status": MagicMock(),
            }
            mock_ha_definitions.define_all_services.return_value = {
                "retrain_model": MagicMock(),
                "validate_model": MagicMock(),
            }
            mock_ha_definitions.publish_all_entities.return_value = {
                "living_room_prediction": MagicMock(success=True),
                "bedroom_prediction": MagicMock(success=True),
                "system_status": MagicMock(success=True),
            }
            mock_ha_definitions.publish_all_services.return_value = {
                "retrain_model": MagicMock(success=True),
                "validate_model": MagicMock(success=True),
            }
            mock_ha_definitions.get_entity_definition.return_value = None
            mock_ha_definitions.get_entity_stats.return_value = {}
            mock_ha_definitions_class.return_value = mock_ha_definitions

            # Initialize enhanced integration
            await enhanced_manager.initialize()

            # Verify entities and services were defined and published
            mock_ha_definitions.define_all_entities.assert_called_once()
            mock_ha_definitions.define_all_services.assert_called_once()
            mock_ha_definitions.publish_all_entities.assert_called_once()
            mock_ha_definitions.publish_all_services.assert_called_once()

            # Test prediction update handling
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
                enhanced_manager, "update_entity_state", new_callable=AsyncMock
            ) as mock_update_state:
                await enhanced_manager.handle_prediction_update(
                    "living_room", prediction_result
                )

                # Verify multiple entity updates were called
                assert mock_update_state.call_count == 5

            # Test system status update handling
            system_status = {
                "system_status": "running",
                "uptime_seconds": 3600,
                "total_predictions_made": 100,
                "database_connected": True,
                "mqtt_connected": True,
            }

            with patch.object(
                enhanced_manager, "update_entity_state", new_callable=AsyncMock
            ) as mock_update_state:
                await enhanced_manager.handle_system_status_update(system_status)

                # Verify system status entities were updated
                assert mock_update_state.call_count >= 6

            # Test command processing
            enhanced_manager.tracking_manager = tracking_manager
            tracking_manager.trigger_retraining.return_value = {"status": "started"}

            result = await enhanced_manager._handle_retrain_model(
                {"room_id": "living_room", "force": True}
            )

            assert result["status"] == "success"
            tracking_manager.trigger_retraining.assert_called_once_with(
                room_id="living_room", force=True
            )

            # Cleanup
            await enhanced_manager.shutdown()

    @pytest.mark.asyncio
    async def test_ha_entity_definitions_complete_workflow(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test complete HA entity definitions workflow."""
        mqtt_manager = mock_system_components["mqtt_manager"]

        # Create HA entity definitions
        entity_definitions = HAEntityDefinitions(
            discovery_publisher=mqtt_manager.discovery_publisher,
            mqtt_config=mock_complete_system_config.mqtt,
            rooms=mock_complete_system_config.rooms,
            tracking_config=mock_complete_system_config.tracking,
        )

        # Mock successful MQTT publishes
        from src.integration.mqtt_publisher import MQTTPublishResult

        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mqtt_manager.discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        # Define all entities
        entities = entity_definitions.define_all_entities()
        assert len(entities) > 0

        # Verify room-specific entities were created
        living_room_entities = [
            entity_id for entity_id in entities.keys() if "living_room" in entity_id
        ]
        bedroom_entities = [
            entity_id for entity_id in entities.keys() if "bedroom" in entity_id
        ]
        assert len(living_room_entities) > 0
        assert len(bedroom_entities) > 0

        # Define all services
        services = entity_definitions.define_all_services()
        assert len(services) > 0

        # Verify expected services
        expected_services = [
            "retrain_model",
            "validate_model",
            "restart_system",
            "force_prediction",
        ]
        for service_name in expected_services:
            assert service_name in services

        # Publish all entities
        entity_results = await entity_definitions.publish_all_entities()
        assert len(entity_results) == len(entities)

        # Publish all services
        service_results = await entity_definitions.publish_all_services()
        assert len(service_results) == len(services)

        # Verify statistics
        stats = entity_definitions.get_entity_stats()
        assert stats["entities_defined"] == len(entities)
        assert stats["services_defined"] == len(services)

    @pytest.mark.asyncio
    async def test_end_to_end_prediction_workflow(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test end-to-end prediction workflow integration."""
        # Setup components
        tracking_manager = mock_system_components["tracking_manager"]
        training_pipeline = mock_system_components["training_pipeline"]
        config_manager = mock_system_components["config_manager"]
        mqtt_manager = mock_system_components["mqtt_manager"]

        # Create training integration
        training_integration = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=config_manager,
        )

        # Create enhanced HA integration
        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_complete_system_config,
        ):
            enhanced_integration = EnhancedIntegrationManager(
                mqtt_integration_manager=mqtt_manager,
                tracking_manager=tracking_manager,
            )

        # Initialize both integrations
        with patch.object(
            training_integration, "_start_background_tasks", new_callable=AsyncMock
        ), patch.object(
            training_integration, "_register_tracking_callbacks", new_callable=AsyncMock
        ), patch(
            "src.integration.enhanced_integration_manager.HAEntityDefinitions"
        ) as mock_ha_definitions_class:

            mock_ha_definitions = AsyncMock()
            mock_ha_definitions.define_all_entities.return_value = {}
            mock_ha_definitions.define_all_services.return_value = {}
            mock_ha_definitions.publish_all_entities.return_value = {}
            mock_ha_definitions.publish_all_services.return_value = {}
            mock_ha_definitions.get_entity_stats.return_value = {}
            mock_ha_definitions_class.return_value = mock_ha_definitions

            await training_integration.initialize()
            await enhanced_integration.initialize()

            # Simulate prediction update
            prediction_result = PredictionResult(
                predicted_time=datetime.utcnow() + timedelta(minutes=15),
                transition_type="vacant_to_occupied",
                confidence_score=0.9,
                time_until_human="15 minutes",
                prediction_reliability="high",
                model_used="ensemble",
                alternatives=[],
            )

            # Handle prediction in enhanced integration
            with patch.object(
                enhanced_integration, "update_entity_state", new_callable=AsyncMock
            ) as mock_update_state:
                await enhanced_integration.handle_prediction_update(
                    "living_room", prediction_result
                )

                # Verify HA entities were updated
                assert mock_update_state.call_count == 5

            # Simulate accuracy degradation
            accuracy_metrics = {
                "accuracy_rate": 65.0,  # Below threshold
                "mean_error_minutes": 25.0,  # Above threshold
            }

            await training_integration._on_accuracy_degradation(
                "living_room", accuracy_metrics
            )

            # Verify retraining was queued
            assert len(training_integration._training_queue) == 1

            # Simulate manual command through HA integration
            enhanced_integration.tracking_manager = tracking_manager
            tracking_manager.force_prediction.return_value = {
                "prediction": prediction_result.__dict__
            }

            command_result = await enhanced_integration._handle_force_prediction(
                {"room_id": "bedroom"}
            )

            assert command_result["status"] == "success"
            tracking_manager.force_prediction.assert_called_once_with(room_id="bedroom")

            # Cleanup
            await training_integration.shutdown()
            await enhanced_integration.shutdown()

    @pytest.mark.asyncio
    async def test_system_error_recovery_scenarios(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test system error recovery scenarios."""
        # Test main system initialization failure recovery
        with patch(
            "src.main_system.get_config", return_value=mock_complete_system_config
        ), patch(
            "src.main_system.get_database_manager",
            side_effect=Exception("Database connection failed"),
        ):

            system = OccupancyPredictionSystem()

            with pytest.raises(Exception, match="Database connection failed"):
                await system.initialize()

            # Verify system is not running after failure
            assert system.running is False

        # Test training integration failure recovery
        tracking_manager = mock_system_components["tracking_manager"]
        training_pipeline = mock_system_components["training_pipeline"]

        # Make training pipeline fail
        training_pipeline.run_retraining_pipeline.side_effect = Exception(
            "Training failed"
        )

        training_integration = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=mock_system_components["config_manager"],
        )

        with patch.object(
            training_integration, "_start_background_tasks", new_callable=AsyncMock
        ), patch.object(
            training_integration, "_register_tracking_callbacks", new_callable=AsyncMock
        ):
            await training_integration.initialize()

            # Queue a training request
            await training_integration._queue_retraining_request(
                room_id="living_room", trigger_reason="test_failure"
            )

            # Process queue - should handle training failure
            await training_integration._process_training_queue()

            # Verify request was processed despite failure
            assert len(training_integration._training_queue) == 0
            assert "living_room" not in training_integration._active_training_requests

        # Test enhanced integration failure recovery
        mqtt_manager = mock_system_components["mqtt_manager"]

        # Make MQTT publisher fail
        mqtt_manager.mqtt_publisher.publish_json.side_effect = Exception(
            "MQTT publish failed"
        )

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_complete_system_config,
        ):
            enhanced_integration = EnhancedIntegrationManager(
                mqtt_integration_manager=mqtt_manager,
                tracking_manager=tracking_manager,
            )

            await enhanced_integration.initialize()

            # Try to update entity state - should handle failure gracefully
            result = await enhanced_integration.update_entity_state(
                "test_entity", "test_state"
            )

            # Should return False but not crash
            assert result is False

    @pytest.mark.asyncio
    async def test_system_performance_under_load(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test system performance under load scenarios."""
        tracking_manager = mock_system_components["tracking_manager"]
        training_pipeline = mock_system_components["training_pipeline"]
        config_manager = mock_system_components["config_manager"]

        # Mock successful but slow training
        async def slow_training(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow training
            mock_progress = MagicMock()
            mock_progress.stage.value = "completed"
            mock_progress.best_model = "lstm"
            return mock_progress

        training_pipeline.run_retraining_pipeline.side_effect = slow_training

        training_integration = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=config_manager,
        )

        # Set low concurrent training limit
        training_integration._max_concurrent_training = 2

        with patch.object(
            training_integration, "_start_background_tasks", new_callable=AsyncMock
        ), patch.object(
            training_integration, "_register_tracking_callbacks", new_callable=AsyncMock
        ):
            await training_integration.initialize()

            # Queue multiple training requests
            rooms = ["living_room", "bedroom", "kitchen", "bathroom"]
            for room in rooms:
                await training_integration._queue_retraining_request(
                    room_id=room, trigger_reason="load_test"
                )

            # Verify all requests were queued
            assert len(training_integration._training_queue) == 4

            # Process queue multiple times to simulate concurrent processing
            tasks = []
            for _ in range(3):
                task = asyncio.create_task(
                    training_integration._process_training_queue()
                )
                tasks.append(task)

            await asyncio.gather(*tasks)

            # Verify some training occurred within concurrent limits
            assert len(training_integration._active_training_requests) <= 2

            # Wait for training to complete
            await asyncio.sleep(0.2)

            # Verify queue was processed
            assert len(training_integration._training_queue) < 4

    @pytest.mark.asyncio
    async def test_system_configuration_validation(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test system configuration validation and handling."""
        # Test with invalid MQTT configuration
        invalid_config = MagicMock()
        invalid_config.mqtt = None
        invalid_config.rooms = {}
        invalid_config.tracking = TrackingConfig()

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=invalid_config,
        ):
            enhanced_integration = EnhancedIntegrationManager()

            # Should handle missing MQTT config gracefully
            await enhanced_integration.initialize()
            assert enhanced_integration.ha_entity_definitions is None

        # Test with empty rooms configuration
        empty_rooms_config = MagicMock()
        empty_rooms_config.mqtt = mock_complete_system_config.mqtt
        empty_rooms_config.rooms = {}
        empty_rooms_config.tracking = TrackingConfig()

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=empty_rooms_config,
        ):
            enhanced_integration = EnhancedIntegrationManager(
                mqtt_integration_manager=mock_system_components["mqtt_manager"]
            )

            await enhanced_integration.initialize()

            # Should still work with empty rooms
            assert enhanced_integration._enhanced_integration_active is True

    @pytest.mark.asyncio
    async def test_system_integration_statistics_and_monitoring(
        self, mock_complete_system_config, mock_system_components
    ):
        """Test system integration statistics and monitoring."""
        mqtt_manager = mock_system_components["mqtt_manager"]
        tracking_manager = mock_system_components["tracking_manager"]
        training_pipeline = mock_system_components["training_pipeline"]

        # Create all integration managers
        training_integration = TrainingIntegrationManager(
            tracking_manager=tracking_manager,
            training_pipeline=training_pipeline,
            config_manager=mock_system_components["config_manager"],
        )

        with patch(
            "src.integration.enhanced_integration_manager.get_config",
            return_value=mock_complete_system_config,
        ):
            enhanced_integration = EnhancedIntegrationManager(
                mqtt_integration_manager=mqtt_manager,
                tracking_manager=tracking_manager,
            )

        with patch.object(
            training_integration, "_start_background_tasks", new_callable=AsyncMock
        ), patch.object(
            training_integration, "_register_tracking_callbacks", new_callable=AsyncMock
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
            mock_ha_definitions.get_entity_stats.return_value = {"defined": 1}
            mock_ha_definitions_class.return_value = mock_ha_definitions

            # Initialize both managers
            await training_integration.initialize()
            await enhanced_integration.initialize()

            # Test training integration statistics
            training_stats = training_integration.get_integration_status()
            assert "integration_active" in training_stats
            assert "active_training_requests" in training_stats
            assert "queued_training_requests" in training_stats

            queue_stats = training_integration.get_training_queue_status()
            assert isinstance(queue_stats, list)

            # Test enhanced integration statistics
            enhanced_stats = enhanced_integration.get_integration_stats()
            assert "enhanced_integration" in enhanced_stats
            assert "mqtt_integration" in enhanced_stats
            assert "entity_states_count" in enhanced_stats
            assert "active" in enhanced_stats

            # Test capacity adjustments
            training_integration.set_training_capacity(5)
            assert training_integration._max_concurrent_training == 5

            training_integration.set_cooldown_period(24)
            assert training_integration._training_cooldown_hours == 24

            # Cleanup
            await training_integration.shutdown()
            await enhanced_integration.shutdown()
