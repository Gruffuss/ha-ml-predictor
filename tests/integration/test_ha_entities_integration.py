"""
Home Assistant Entity Definitions Integration Tests.

This module provides comprehensive integration testing for HA entity definitions,
MQTT discovery, state management, and real Home Assistant integration scenarios.

Focus Areas:
- Entity discovery message publishing and validation
- State publishing and subscription verification
- Entity lifecycle management (create, update, delete)
- Service integration and command handling
- Complex entity type interactions
- Performance under high entity counts
- Error recovery in discovery processes
- Cross-entity state synchronization
"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, Mock, call, patch
import uuid

import pytest

from src.core.config import MQTTConfig, RoomConfig, TrackingConfig
from src.integration.discovery_publisher import DeviceInfo, DiscoveryPublisher
from src.integration.ha_entity_definitions import (
    HABinarySensorEntityConfig,
    HAButtonEntityConfig,
    HADeviceClass,
    HAEntityCategory,
    HAEntityDefinitions,
    HAEntityType,
    HANumberEntityConfig,
    HASelectEntityConfig,
    HASensorEntityConfig,
    HAServiceDefinition,
    HAStateClass,
    HASwitchEntityConfig,
)
from src.integration.mqtt_publisher import MQTTPublisher, MQTTPublishResult

# Test configuration
TEST_MQTT_CONFIG = MQTTConfig(
    broker="localhost",
    port=1883,
    topic_prefix="homeassistant",
    discovery_prefix="homeassistant",
    publishing_enabled=True,
    discovery_enabled=True,
    device_identifier="test_ha_ml_predictor",
    keepalive=60,
    connection_timeout=10,
    max_reconnect_attempts=3,
    reconnect_delay_seconds=1,
)

TEST_DEVICE_INFO = DeviceInfo(
    identifiers=["test_ha_ml_predictor"],
    name="Test HA ML Predictor",
    manufacturer="Test Manufacturer",
    model="Test Model",
    sw_version="1.0.0-test",
    configuration_url="http://localhost:8000",
)

TEST_ROOMS = {
    "living_room": RoomConfig(
        room_id="living_room",
        name="Living Room",
        sensors={
            "motion": ["binary_sensor.living_room_motion"],
            "door": ["binary_sensor.living_room_door"],
            "temperature": ["sensor.living_room_temperature"],
        },
    ),
    "bedroom": RoomConfig(
        room_id="bedroom",
        name="Bedroom",
        sensors={
            "motion": ["binary_sensor.bedroom_motion"],
            "door": ["binary_sensor.bedroom_door"],
        },
    ),
    "kitchen": RoomConfig(
        room_id="kitchen",
        name="Kitchen",
        sensors={
            "motion": ["binary_sensor.kitchen_motion"],
            "door": ["binary_sensor.kitchen_door"],
            "temperature": ["sensor.kitchen_temperature"],
            "humidity": ["sensor.kitchen_humidity"],
        },
    ),
}


@pytest.fixture
def mock_mqtt_publisher():
    """Create mock MQTT publisher for testing."""
    mock_publisher = AsyncMock(spec=MQTTPublisher)

    # Mock successful publishing by default
    mock_publisher.publish_json.return_value = MQTTPublishResult(
        success=True,
        topic="test/topic",
        payload_size=100,
        publish_time=datetime.utcnow(),
        message_id=12345,
    )

    mock_publisher.publish.return_value = MQTTPublishResult(
        success=True,
        topic="test/topic",
        payload_size=100,
        publish_time=datetime.utcnow(),
        message_id=12345,
    )

    return mock_publisher


@pytest.fixture
def discovery_publisher(mock_mqtt_publisher):
    """Create discovery publisher with mock MQTT."""
    return DiscoveryPublisher(
        mqtt_publisher=mock_mqtt_publisher,
        device_info=TEST_DEVICE_INFO,
        mqtt_config=TEST_MQTT_CONFIG,
    )


@pytest.fixture
def ha_entity_definitions(discovery_publisher):
    """Create HA entity definitions instance."""
    tracking_config = TrackingConfig(
        enabled=True, accuracy_threshold_minutes=15, prediction_interval_seconds=300
    )

    return HAEntityDefinitions(
        discovery_publisher=discovery_publisher,
        mqtt_config=TEST_MQTT_CONFIG,
        rooms=TEST_ROOMS,
        tracking_config=tracking_config,
    )


class TestEntityDefinitionCreation:
    """Test entity definition creation and validation."""

    @pytest.mark.asyncio
    async def test_room_entity_definitions(self, ha_entity_definitions):
        """Test creation of room-specific entity definitions."""
        entities = ha_entity_definitions.define_all_entities()

        # Should create entities for all rooms
        assert len(entities) > 0

        # Check for room-specific entities
        for room_id in TEST_ROOMS.keys():
            # Prediction sensor
            assert f"{room_id}_prediction" in entities
            pred_entity = entities[f"{room_id}_prediction"]
            assert isinstance(pred_entity, HASensorEntityConfig)
            assert pred_entity.entity_type == HAEntityType.SENSOR
            assert pred_entity.icon == "mdi:home-account"

            # Next transition timestamp
            assert f"{room_id}_next_transition" in entities
            transition_entity = entities[f"{room_id}_next_transition"]
            assert transition_entity.device_class == HADeviceClass.TIMESTAMP.value

            # Confidence percentage
            assert f"{room_id}_confidence" in entities
            confidence_entity = entities[f"{room_id}_confidence"]
            assert confidence_entity.unit_of_measurement == "%"
            assert confidence_entity.state_class == HAStateClass.MEASUREMENT.value

            # Currently occupied binary sensor
            assert f"{room_id}_occupied" in entities
            occupied_entity = entities[f"{room_id}_occupied"]
            assert isinstance(occupied_entity, HABinarySensorEntityConfig)
            assert occupied_entity.device_class == HADeviceClass.CONNECTIVITY.value

    @pytest.mark.asyncio
    async def test_system_entity_definitions(self, ha_entity_definitions):
        """Test creation of system-wide entity definitions."""
        entities = ha_entity_definitions.define_all_entities()

        # System status sensor
        assert "system_status" in entities
        system_entity = entities["system_status"]
        assert isinstance(system_entity, HASensorEntityConfig)
        assert system_entity.entity_category == "diagnostic"

        # System uptime
        assert "system_uptime" in entities
        uptime_entity = entities["system_uptime"]
        assert uptime_entity.device_class == HADeviceClass.DURATION.value
        assert uptime_entity.unit_of_measurement == "s"

        # Total predictions count
        assert "predictions_count" in entities
        count_entity = entities["predictions_count"]
        assert count_entity.state_class == HAStateClass.TOTAL_INCREASING.value

        # System accuracy
        assert "system_accuracy" in entities
        accuracy_entity = entities["system_accuracy"]
        assert accuracy_entity.unit_of_measurement == "%"

    @pytest.mark.asyncio
    async def test_diagnostic_entity_definitions(self, ha_entity_definitions):
        """Test creation of diagnostic entity definitions."""
        entities = ha_entity_definitions.define_all_entities()

        # Database connection binary sensor
        assert "database_connected" in entities
        db_entity = entities["database_connected"]
        assert isinstance(db_entity, HABinarySensorEntityConfig)
        assert db_entity.device_class == HADeviceClass.CONNECTIVITY.value

        # MQTT connection status
        assert "mqtt_connected" in entities
        mqtt_entity = entities["mqtt_connected"]
        assert mqtt_entity.icon == "mdi:wifi"

        # Memory usage sensor
        assert "memory_usage" in entities
        memory_entity = entities["memory_usage"]
        assert memory_entity.device_class == HADeviceClass.DATA_SIZE.value
        assert memory_entity.unit_of_measurement == "MB"

        # CPU usage sensor
        assert "cpu_usage" in entities
        cpu_entity = entities["cpu_usage"]
        assert cpu_entity.unit_of_measurement == "%"

    @pytest.mark.asyncio
    async def test_control_entity_definitions(self, ha_entity_definitions):
        """Test creation of control entity definitions."""
        entities = ha_entity_definitions.define_all_entities()

        # Prediction system switch
        assert "prediction_enabled" in entities
        pred_switch = entities["prediction_enabled"]
        assert isinstance(pred_switch, HASwitchEntityConfig)
        assert pred_switch.entity_category == "config"

        # Prediction interval number entity
        assert "prediction_interval" in entities
        interval_entity = entities["prediction_interval"]
        assert isinstance(interval_entity, HANumberEntityConfig)
        assert interval_entity.min == 60
        assert interval_entity.max == 3600
        assert interval_entity.unit_of_measurement == "s"

        # Log level select entity
        assert "log_level" in entities
        log_entity = entities["log_level"]
        assert isinstance(log_entity, HASelectEntityConfig)
        assert "DEBUG" in log_entity.options
        assert "INFO" in log_entity.options


class TestServiceDefinitions:
    """Test service definition creation and validation."""

    @pytest.mark.asyncio
    async def test_model_service_definitions(self, ha_entity_definitions):
        """Test model management service definitions."""
        services = ha_entity_definitions.define_all_services()

        # Retrain model service
        assert "retrain_model" in services
        retrain_service = services["retrain_model"]
        assert isinstance(retrain_service, HAServiceDefinition)
        assert retrain_service.domain == "ha_ml_predictor"
        assert retrain_service.supports_response
        assert "room_id" in retrain_service.fields
        assert "force" in retrain_service.fields

        # Validate model service
        assert "validate_model" in services
        validate_service = services["validate_model"]
        assert "days" in validate_service.fields

        # Export model service
        assert "export_model" in services
        export_service = services["export_model"]
        assert "format" in export_service.fields
        assert export_service.supports_response

    @pytest.mark.asyncio
    async def test_system_service_definitions(self, ha_entity_definitions):
        """Test system control service definitions."""
        services = ha_entity_definitions.define_all_services()

        # Restart system service
        assert "restart_system" in services
        restart_service = services["restart_system"]
        assert restart_service.icon == "mdi:restart"

        # Refresh discovery service
        assert "refresh_discovery" in services
        refresh_service = services["refresh_discovery"]
        assert refresh_service.icon == "mdi:refresh"

        # Update configuration service
        assert "update_config" in services
        config_service = services["update_config"]
        assert "config_section" in config_service.fields
        assert "config_data" in config_service.fields

    @pytest.mark.asyncio
    async def test_diagnostic_service_definitions(self, ha_entity_definitions):
        """Test diagnostic service definitions."""
        services = ha_entity_definitions.define_all_services()

        # Generate diagnostic report
        assert "generate_diagnostic" in services
        diag_service = services["generate_diagnostic"]
        assert diag_service.supports_response
        assert "include_logs" in diag_service.fields

        # Database health check
        assert "check_database" in services
        db_service = services["check_database"]
        assert db_service.icon == "mdi:database-check"


class TestEntityDiscoveryPublishing:
    """Test entity discovery message publishing."""

    @pytest.mark.asyncio
    async def test_publish_sensor_entities(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test publishing sensor entity discovery messages."""
        # Define entities first
        entities = ha_entity_definitions.define_all_entities()

        # Publish all entities
        results = await ha_entity_definitions.publish_all_entities()

        # Should have published discovery messages
        assert len(results) > 0

        # Check MQTT publisher was called
        assert mock_mqtt_publisher.publish_json.call_count > 0

        # Verify sensor entity discovery format
        calls = mock_mqtt_publisher.publish_json.call_args_list

        # Find a sensor entity call
        sensor_calls = [call for call in calls if "sensor/" in call[1]["topic"]]
        assert len(sensor_calls) > 0

        # Verify discovery topic format
        sensor_call = sensor_calls[0]
        topic = sensor_call[1]["topic"]
        payload = sensor_call[1]["data"]

        # Topic should follow HA discovery format
        assert topic.startswith("homeassistant/sensor/")
        assert topic.endswith("/config")

        # Payload should have required fields
        assert "name" in payload
        assert "unique_id" in payload
        assert "device" in payload
        assert "state_topic" in payload

    @pytest.mark.asyncio
    async def test_publish_binary_sensor_entities(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test publishing binary sensor entity discovery messages."""
        await ha_entity_definitions.publish_all_entities()

        calls = mock_mqtt_publisher.publish_json.call_args_list

        # Find binary sensor calls
        binary_sensor_calls = [
            call for call in calls if "binary_sensor/" in call[1]["topic"]
        ]
        assert len(binary_sensor_calls) > 0

        # Check binary sensor specific attributes
        binary_call = binary_sensor_calls[0]
        payload = binary_call[1]["data"]

        # Should have binary sensor specific fields
        if "device_class" in payload:
            assert payload["device_class"] in [
                "connectivity",
                "problem",
                "running",
                "motion",
                "occupancy",
            ]

    @pytest.mark.asyncio
    async def test_publish_switch_entities(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test publishing switch entity discovery messages."""
        await ha_entity_definitions.publish_all_entities()

        calls = mock_mqtt_publisher.publish_json.call_args_list

        # Find switch calls
        switch_calls = [call for call in calls if "switch/" in call[1]["topic"]]
        assert len(switch_calls) > 0

        # Check switch specific attributes
        switch_call = switch_calls[0]
        payload = switch_call[1]["data"]

        # Should have command topic
        assert "command_topic" in payload
        assert payload["command_topic"].startswith("homeassistant/")

    @pytest.mark.asyncio
    async def test_publish_number_entities(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test publishing number entity discovery messages."""
        await ha_entity_definitions.publish_all_entities()

        calls = mock_mqtt_publisher.publish_json.call_args_list

        # Find number calls
        number_calls = [call for call in calls if "number/" in call[1]["topic"]]
        assert len(number_calls) > 0

        # Check number specific attributes
        number_call = number_calls[0]
        payload = number_call[1]["data"]

        # Should have min/max/step
        assert "min" in payload
        assert "max" in payload
        assert "step" in payload
        assert "command_topic" in payload

    @pytest.mark.asyncio
    async def test_publish_select_entities(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test publishing select entity discovery messages."""
        await ha_entity_definitions.publish_all_entities()

        calls = mock_mqtt_publisher.publish_json.call_args_list

        # Find select calls
        select_calls = [call for call in calls if "select/" in call[1]["topic"]]
        assert len(select_calls) > 0

        # Check select specific attributes
        select_call = select_calls[0]
        payload = select_call[1]["data"]

        # Should have options list
        assert "options" in payload
        assert isinstance(payload["options"], list)
        assert len(payload["options"]) > 0


class TestServiceButtonPublishing:
    """Test service button publishing for HA integration."""

    @pytest.mark.asyncio
    async def test_publish_service_buttons(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test publishing service buttons for HA control."""
        # Define services first
        services = ha_entity_definitions.define_all_services()

        # Publish service buttons
        results = await ha_entity_definitions.publish_all_services()

        # Should have published service buttons
        assert len(results) > 0

        # Each service should result in a button entity
        assert len(results) == len(services)

        # Check MQTT publisher was called for buttons
        calls = mock_mqtt_publisher.publish_json.call_args_list

        # Find button calls
        button_calls = [call for call in calls if "button/" in call[1]["topic"]]
        assert len(button_calls) >= len(services)

        # Verify button discovery format
        button_call = button_calls[0]
        topic = button_call[1]["topic"]
        payload = button_call[1]["data"]

        # Should have button-specific attributes
        assert "command_topic" in payload
        assert topic.startswith("homeassistant/button/")


class TestEntityLifecycleManagement:
    """Test entity lifecycle management operations."""

    @pytest.mark.asyncio
    async def test_entity_creation_and_update(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test entity creation and subsequent updates."""
        # Initial entity creation
        entities_v1 = ha_entity_definitions.define_all_entities()
        results_v1 = await ha_entity_definitions.publish_all_entities()

        initial_publish_count = mock_mqtt_publisher.publish_json.call_count

        # Modify entity definitions (simulate configuration change)
        # Add a new room
        ha_entity_definitions.rooms["garage"] = RoomConfig(
            room_id="garage",
            name="Garage",
            sensors={
                "motion": ["binary_sensor.garage_motion"],
                "door": ["binary_sensor.garage_door"],
            },
        )

        # Re-define and publish entities
        entities_v2 = ha_entity_definitions.define_all_entities()
        results_v2 = await ha_entity_definitions.publish_all_entities()

        # Should have more entities now
        assert len(entities_v2) > len(entities_v1)

        # Should have published additional discovery messages
        assert mock_mqtt_publisher.publish_json.call_count > initial_publish_count

        # New garage entities should exist
        garage_entities = [
            entity_id
            for entity_id in entities_v2.keys()
            if entity_id.startswith("garage_")
        ]
        assert len(garage_entities) > 0

    @pytest.mark.asyncio
    async def test_entity_removal(self, ha_entity_definitions, mock_mqtt_publisher):
        """Test entity removal and cleanup."""
        # Start with all entities
        initial_entities = ha_entity_definitions.define_all_entities()

        # Remove a room
        removed_room = "bedroom"
        del ha_entity_definitions.rooms[removed_room]

        # Re-define entities
        updated_entities = ha_entity_definitions.define_all_entities()

        # Should have fewer entities
        assert len(updated_entities) < len(initial_entities)

        # Bedroom entities should be gone
        bedroom_entities = [
            entity_id
            for entity_id in updated_entities.keys()
            if entity_id.startswith("bedroom_")
        ]
        assert len(bedroom_entities) == 0

    @pytest.mark.asyncio
    async def test_entity_configuration_updates(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test updating entity configurations."""
        # Get initial entity
        entities = ha_entity_definitions.define_all_entities()
        living_room_pred = entities["living_room_prediction"]
        original_icon = living_room_pred.icon

        # Simulate configuration change
        living_room_pred.icon = "mdi:crystal-ball"
        living_room_pred.entity_category = "config"  # Change category

        # Republish
        await ha_entity_definitions.publish_all_entities()

        # Verify the entity was updated
        assert living_room_pred.icon == "mdi:crystal-ball"
        assert living_room_pred.entity_category == "config"


class TestCrossEntityIntegration:
    """Test interactions between different entity types."""

    @pytest.mark.asyncio
    async def test_room_entity_consistency(self, ha_entity_definitions):
        """Test consistency across room entity definitions."""
        entities = ha_entity_definitions.define_all_entities()

        # Each room should have consistent entity set
        for room_id in TEST_ROOMS.keys():
            room_entities = [
                entity_id
                for entity_id in entities.keys()
                if entity_id.startswith(f"{room_id}_")
            ]

            # Should have core entities for each room
            core_entities = [
                f"{room_id}_prediction",
                f"{room_id}_confidence",
                f"{room_id}_next_transition",
                f"{room_id}_occupied",
            ]

            for core_entity in core_entities:
                assert core_entity in entities, f"Missing {core_entity} for {room_id}"

            # All room entities should use same device
            for entity_id in room_entities:
                entity = entities[entity_id]
                assert entity.device.identifiers == TEST_DEVICE_INFO.identifiers

    @pytest.mark.asyncio
    async def test_topic_structure_consistency(self, ha_entity_definitions):
        """Test MQTT topic structure consistency."""
        entities = ha_entity_definitions.define_all_entities()

        # Check topic patterns
        for entity_id, entity in entities.items():
            if hasattr(entity, "state_topic") and entity.state_topic:
                # Should follow consistent topic structure
                assert entity.state_topic.startswith("homeassistant/")

                # Room-specific entities should have room in topic
                if "_" in entity_id and not entity_id.startswith("system_"):
                    room_id = entity_id.split("_")[0]
                    if room_id in TEST_ROOMS:
                        assert room_id in entity.state_topic

    @pytest.mark.asyncio
    async def test_device_info_consistency(self, ha_entity_definitions):
        """Test device info consistency across all entities."""
        entities = ha_entity_definitions.define_all_entities()

        # All entities should reference same device
        for entity in entities.values():
            assert entity.device.identifiers == TEST_DEVICE_INFO.identifiers
            assert entity.device.name == TEST_DEVICE_INFO.name
            assert entity.device.manufacturer == TEST_DEVICE_INFO.manufacturer


class TestPerformanceUnderLoad:
    """Test entity system performance under high load."""

    @pytest.mark.asyncio
    async def test_large_scale_entity_creation(self, discovery_publisher):
        """Test entity creation with many rooms."""
        # Create many test rooms
        many_rooms = {}
        for i in range(50):
            many_rooms[f"room_{i}"] = RoomConfig(
                room_id=f"room_{i}",
                name=f"Room {i}",
                sensors={
                    "motion": [f"binary_sensor.room_{i}_motion"],
                    "door": [f"binary_sensor.room_{i}_door"],
                },
            )

        ha_definitions = HAEntityDefinitions(
            discovery_publisher=discovery_publisher,
            mqtt_config=TEST_MQTT_CONFIG,
            rooms=many_rooms,
        )

        # Time entity creation
        start_time = datetime.now()
        entities = ha_definitions.define_all_entities()
        end_time = datetime.now()

        creation_time = (end_time - start_time).total_seconds()

        # Should create entities quickly
        assert creation_time < 5.0  # Less than 5 seconds

        # Should have many entities
        assert len(entities) > 500  # At least 10 entities per room

        print(f"Created {len(entities)} entities in {creation_time:.2f}s")

    @pytest.mark.asyncio
    async def test_concurrent_entity_publishing(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test concurrent entity publishing performance."""
        entities = ha_entity_definitions.define_all_entities()

        # Mock concurrent publishing
        async def concurrent_publish():
            return await ha_entity_definitions.publish_all_entities()

        # Run multiple publishing operations concurrently
        start_time = datetime.now()
        tasks = [concurrent_publish() for _ in range(5)]
        results = await asyncio.gather(*tasks)
        end_time = datetime.now()

        duration = (end_time - start_time).total_seconds()

        # All should succeed
        for result in results:
            assert len(result) > 0

        # Should complete within reasonable time
        assert duration < 10.0

        print(f"5 concurrent publish operations completed in {duration:.2f}s")

    @pytest.mark.asyncio
    async def test_memory_usage_with_many_entities(self, discovery_publisher):
        """Test memory usage with large number of entities."""
        import os

        import psutil

        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Create many rooms and entities
        many_rooms = {}
        for i in range(100):
            many_rooms[f"room_{i}"] = RoomConfig(
                room_id=f"room_{i}",
                name=f"Room {i}",
                sensors={
                    "motion": [f"binary_sensor.room_{i}_motion"],
                    "door": [f"binary_sensor.room_{i}_door"],
                    "temperature": [f"sensor.room_{i}_temperature"],
                },
            )

        ha_definitions = HAEntityDefinitions(
            discovery_publisher=discovery_publisher,
            mqtt_config=TEST_MQTT_CONFIG,
            rooms=many_rooms,
        )

        # Create all entities
        entities = ha_definitions.define_all_entities()
        services = ha_definitions.define_all_services()

        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        print(f"Created {len(entities)} entities and {len(services)} services")
        print(f"Memory increase: {memory_increase:.1f}MB")

        # Memory increase should be reasonable
        assert memory_increase < 100  # Less than 100MB increase


class TestErrorRecoveryScenarios:
    """Test error recovery in entity management."""

    @pytest.mark.asyncio
    async def test_mqtt_publishing_failure_recovery(
        self, ha_entity_definitions, mock_mqtt_publisher
    ):
        """Test recovery from MQTT publishing failures."""
        # Configure mock to fail first few calls
        call_count = 0

        async def failing_publish_json(*args, **kwargs):
            nonlocal call_count
            call_count += 1

            if call_count <= 3:
                # First 3 calls fail
                return MQTTPublishResult(
                    success=False,
                    topic=kwargs.get("topic", "test"),
                    payload_size=0,
                    publish_time=datetime.utcnow(),
                    error_message="Connection failed",
                )
            else:
                # Subsequent calls succeed
                return MQTTPublishResult(
                    success=True,
                    topic=kwargs.get("topic", "test"),
                    payload_size=100,
                    publish_time=datetime.utcnow(),
                    message_id=12345,
                )

        mock_mqtt_publisher.publish_json.side_effect = failing_publish_json

        # Attempt to publish entities
        results = await ha_entity_definitions.publish_all_entities()

        # Some should fail, some should succeed
        failed_count = sum(1 for r in results.values() if not r.success)
        success_count = sum(1 for r in results.values() if r.success)

        assert failed_count > 0  # Some failures expected
        assert success_count > 0  # Some successes expected

        print(
            f"Publishing with failures: {failed_count} failed, {success_count} succeeded"
        )

    @pytest.mark.asyncio
    async def test_invalid_entity_configuration_handling(self, discovery_publisher):
        """Test handling of invalid entity configurations."""
        # Create entity definitions with some invalid configurations
        ha_definitions = HAEntityDefinitions(
            discovery_publisher=discovery_publisher,
            mqtt_config=TEST_MQTT_CONFIG,
            rooms=TEST_ROOMS,
        )

        # Define entities normally first
        entities = ha_definitions.define_all_entities()

        # Manually corrupt some entity configurations
        living_room_pred = entities["living_room_prediction"]
        living_room_pred.state_topic = ""  # Invalid empty topic
        living_room_pred.unique_id = None  # Invalid None unique_id

        # Try to publish - should handle errors gracefully
        results = await ha_definitions.publish_all_entities()

        # Should have some results (not all fail)
        assert len(results) > 0

        # Some should fail due to invalid config
        failed_count = sum(1 for r in results.values() if not r.success)
        assert failed_count > 0

    @pytest.mark.asyncio
    async def test_partial_room_configuration_errors(self, discovery_publisher):
        """Test handling of partial room configuration errors."""
        # Create rooms with some invalid configurations
        problematic_rooms = {
            "valid_room": RoomConfig(
                room_id="valid_room",
                name="Valid Room",
                sensors={"motion": ["binary_sensor.valid_motion"]},
            ),
            "invalid_room": RoomConfig(
                room_id="",  # Invalid empty room_id
                name="Invalid Room",
                sensors={},  # Empty sensors
            ),
            "another_valid_room": RoomConfig(
                room_id="another_valid_room",
                name="Another Valid Room",
                sensors={"door": ["binary_sensor.valid_door"]},
            ),
        }

        ha_definitions = HAEntityDefinitions(
            discovery_publisher=discovery_publisher,
            mqtt_config=TEST_MQTT_CONFIG,
            rooms=problematic_rooms,
        )

        # Should handle invalid rooms gracefully
        entities = ha_definitions.define_all_entities()

        # Should have entities for valid rooms
        valid_entities = [
            entity_id
            for entity_id in entities.keys()
            if entity_id.startswith("valid_room_")
            or entity_id.startswith("another_valid_room_")
        ]
        assert len(valid_entities) > 0

        # Should not have entities for invalid room
        invalid_entities = [
            entity_id
            for entity_id in entities.keys()
            if entity_id.startswith("invalid_room_") or entity_id.startswith("_")
        ]
        assert len(invalid_entities) == 0


class TestEntityStateManagement:
    """Test entity state management and synchronization."""

    @pytest.mark.asyncio
    async def test_entity_state_tracking(self, ha_entity_definitions):
        """Test entity state tracking functionality."""
        entities = ha_entity_definitions.define_all_entities()

        # Set some entity states
        ha_entity_definitions.entity_states["living_room_prediction"] = "occupied"
        ha_entity_definitions.entity_states["system_status"] = "healthy"

        # Set availability
        ha_entity_definitions.entity_availability["living_room_prediction"] = True
        ha_entity_definitions.entity_availability["system_status"] = False

        # Verify state storage
        assert (
            ha_entity_definitions.entity_states["living_room_prediction"] == "occupied"
        )
        assert ha_entity_definitions.entity_availability["system_status"] is False

    @pytest.mark.asyncio
    async def test_entity_statistics_tracking(self, ha_entity_definitions):
        """Test entity statistics tracking."""
        # Define and publish entities
        entities = ha_entity_definitions.define_all_entities()
        services = ha_entity_definitions.define_all_services()
        await ha_entity_definitions.publish_all_entities()
        await ha_entity_definitions.publish_all_services()

        # Check statistics
        stats = ha_entity_definitions.get_entity_stats()

        assert stats["entities_defined"] == len(entities)
        assert stats["services_defined"] == len(services)
        assert stats["entities_published"] > 0
        assert stats["last_update"] is not None

        # Check entity type breakdown
        assert "entity_types" in stats
        assert "sensor" in stats["entity_types"]
        assert stats["entity_types"]["sensor"] > 0

    @pytest.mark.asyncio
    async def test_entity_retrieval_methods(self, ha_entity_definitions):
        """Test entity and service retrieval methods."""
        entities = ha_entity_definitions.define_all_entities()
        services = ha_entity_definitions.define_all_services()

        # Test entity retrieval
        living_room_pred = ha_entity_definitions.get_entity_definition(
            "living_room_prediction"
        )
        assert living_room_pred is not None
        assert isinstance(living_room_pred, HASensorEntityConfig)

        # Test non-existent entity
        non_existent = ha_entity_definitions.get_entity_definition(
            "non_existent_entity"
        )
        assert non_existent is None

        # Test service retrieval
        retrain_service = ha_entity_definitions.get_service_definition("retrain_model")
        assert retrain_service is not None
        assert isinstance(retrain_service, HAServiceDefinition)

        # Test non-existent service
        non_existent_service = ha_entity_definitions.get_service_definition(
            "non_existent_service"
        )
        assert non_existent_service is None
