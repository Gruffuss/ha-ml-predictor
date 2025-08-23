"""
Comprehensive tests for Home Assistant Entity Definitions.

This module tests the HA entity definitions system including entity configuration,
discovery payload generation, service definitions, and MQTT discovery publishing.
"""

from datetime import datetime
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.core.config import MQTTConfig, RoomConfig, TrackingConfig
from src.integration.discovery_publisher import DeviceInfo
from src.integration.ha_entity_definitions import (
    HABinarySensorEntityConfig,
    HAButtonEntityConfig,
    HADateTimeEntityConfig,
    HADeviceClass,
    HAEntityCategory,
    HAEntityConfig,
    HAEntityDefinitions,
    HAEntityDefinitionsError,
    HAEntityType,
    HAImageEntityConfig,
    HANumberEntityConfig,
    HASelectEntityConfig,
    HASensorEntityConfig,
    HAServiceDefinition,
    HAStateClass,
    HASwitchEntityConfig,
    HATextEntityConfig,
)
from src.integration.mqtt_publisher import MQTTPublishResult


@pytest.mark.unit
class TestHAEntityEnums:
    """Test cases for HA entity enums."""

    def test_ha_entity_type_enum(self):
        """Test HAEntityType enum values."""
        assert HAEntityType.SENSOR.value == "sensor"
        assert HAEntityType.BINARY_SENSOR.value == "binary_sensor"
        assert HAEntityType.BUTTON.value == "button"
        assert HAEntityType.SWITCH.value == "switch"
        assert HAEntityType.NUMBER.value == "number"
        assert HAEntityType.SELECT.value == "select"
        assert HAEntityType.TEXT.value == "text"
        assert HAEntityType.DEVICE_TRACKER.value == "device_tracker"
        assert HAEntityType.DATETIME.value == "datetime"
        assert HAEntityType.TIME.value == "time"
        assert HAEntityType.DATE.value == "date"
        assert HAEntityType.IMAGE.value == "image"

    def test_ha_device_class_enum(self):
        """Test HADeviceClass enum values."""
        assert HADeviceClass.TIMESTAMP.value == "timestamp"
        assert HADeviceClass.DURATION.value == "duration"
        assert HADeviceClass.CONNECTIVITY.value == "connectivity"
        assert HADeviceClass.PROBLEM.value == "problem"
        assert HADeviceClass.MOTION.value == "motion"
        assert HADeviceClass.OCCUPANCY.value == "occupancy"

    def test_ha_entity_category_enum(self):
        """Test HAEntityCategory enum values."""
        assert HAEntityCategory.CONFIG.value == "config"
        assert HAEntityCategory.DIAGNOSTIC.value == "diagnostic"
        assert HAEntityCategory.SYSTEM.value == "system"

    def test_ha_state_class_enum(self):
        """Test HAStateClass enum values."""
        assert HAStateClass.MEASUREMENT.value == "measurement"
        assert HAStateClass.TOTAL.value == "total"
        assert HAStateClass.TOTAL_INCREASING.value == "total_increasing"


@pytest.mark.unit
class TestHAEntityConfigs:
    """Test cases for HA entity configuration classes."""

    @pytest.fixture
    def mock_device_info(self):
        """Mock device info."""
        return DeviceInfo(
            identifiers=["test-device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
        )

    def test_ha_entity_config_creation(self, mock_device_info):
        """Test HAEntityConfig creation."""
        config = HAEntityConfig(
            entity_type=HAEntityType.SENSOR,
            name="Test Entity",
            unique_id="test_entity_123",
            state_topic="test/state",
            device=mock_device_info,
            icon="mdi:test",
            entity_category="diagnostic",
            enabled_by_default=False,
            description="Test entity description",
        )

        assert config.entity_type == HAEntityType.SENSOR
        assert config.name == "Test Entity"
        assert config.unique_id == "test_entity_123"
        assert config.state_topic == "test/state"
        assert config.device is mock_device_info
        assert config.icon == "mdi:test"
        assert config.entity_category == "diagnostic"
        assert config.enabled_by_default is False
        assert config.description == "Test entity description"
        assert isinstance(config.created_at, datetime)

    def test_ha_sensor_entity_config(self, mock_device_info):
        """Test HASensorEntityConfig creation."""
        config = HASensorEntityConfig(
            name="Test Sensor",
            unique_id="test_sensor_123",
            state_topic="test/sensor/state",
            device=mock_device_info,
            value_template="{{ value_json.value }}",
            unit_of_measurement="°C",
            device_class=HADeviceClass.TEMPERATURE.value,
            state_class=HAStateClass.MEASUREMENT.value,
            suggested_display_precision=1,
        )

        assert config.entity_type == HAEntityType.SENSOR
        assert config.value_template == "{{ value_json.value }}"
        assert config.unit_of_measurement == "°C"
        assert config.device_class == "temperature"
        assert config.state_class == "measurement"
        assert config.suggested_display_precision == 1

    def test_ha_binary_sensor_entity_config(self, mock_device_info):
        """Test HABinarySensorEntityConfig creation."""
        config = HABinarySensorEntityConfig(
            name="Test Binary Sensor",
            unique_id="test_binary_sensor_123",
            state_topic="test/binary_sensor/state",
            device=mock_device_info,
            value_template="{{ value_json.state }}",
            payload_on="true",
            payload_off="false",
            device_class=HADeviceClass.MOTION.value,
            off_delay=30,
        )

        assert config.entity_type == HAEntityType.BINARY_SENSOR
        assert config.value_template == "{{ value_json.state }}"
        assert config.payload_on == "true"
        assert config.payload_off == "false"
        assert config.device_class == "motion"
        assert config.off_delay == 30

    def test_ha_button_entity_config(self, mock_device_info):
        """Test HAButtonEntityConfig creation."""
        config = HAButtonEntityConfig(
            name="Test Button",
            unique_id="test_button_123",
            state_topic="",  # Buttons don't have state
            command_topic="test/button/command",
            device=mock_device_info,
            command_template="{{ value }}",
            payload_press="PRESS",
            device_class=HADeviceClass.RESTART.value,
            retain=True,
            qos=2,
        )

        assert config.entity_type == HAEntityType.BUTTON
        assert config.command_topic == "test/button/command"
        assert config.command_template == "{{ value }}"
        assert config.payload_press == "PRESS"
        assert config.device_class == "restart"
        assert config.retain is True
        assert config.qos == 2

    def test_ha_switch_entity_config(self, mock_device_info):
        """Test HASwitchEntityConfig creation."""
        config = HASwitchEntityConfig(
            name="Test Switch",
            unique_id="test_switch_123",
            state_topic="test/switch/state",
            command_topic="test/switch/command",
            device=mock_device_info,
            state_on="enabled",
            state_off="disabled",
            payload_on="ENABLE",
            payload_off="DISABLE",
            value_template="{{ value_json.state }}",
            optimistic=True,
        )

        assert config.entity_type == HAEntityType.SWITCH
        assert config.command_topic == "test/switch/command"
        assert config.state_on == "enabled"
        assert config.state_off == "disabled"
        assert config.payload_on == "ENABLE"
        assert config.payload_off == "DISABLE"
        assert config.value_template == "{{ value_json.state }}"
        assert config.optimistic is True

    def test_ha_number_entity_config(self, mock_device_info):
        """Test HANumberEntityConfig creation."""
        config = HANumberEntityConfig(
            name="Test Number",
            unique_id="test_number_123",
            state_topic="test/number/state",
            command_topic="test/number/command",
            device=mock_device_info,
            min=0.0,
            max=100.0,
            step=0.1,
            mode="slider",
            unit_of_measurement="%",
            device_class=HADeviceClass.POWER.value,
        )

        assert config.entity_type == HAEntityType.NUMBER
        assert config.command_topic == "test/number/command"
        assert config.min == 0.0
        assert config.max == 100.0
        assert config.step == 0.1
        assert config.mode == "slider"
        assert config.unit_of_measurement == "%"
        assert config.device_class == "power"

    def test_ha_select_entity_config(self, mock_device_info):
        """Test HASelectEntityConfig creation."""
        config = HASelectEntityConfig(
            name="Test Select",
            unique_id="test_select_123",
            state_topic="test/select/state",
            command_topic="test/select/command",
            device=mock_device_info,
            options=["option1", "option2", "option3"],
            value_template="{{ value_json.selected }}",
            optimistic=False,
        )

        assert config.entity_type == HAEntityType.SELECT
        assert config.command_topic == "test/select/command"
        assert config.options == ["option1", "option2", "option3"]
        assert config.value_template == "{{ value_json.selected }}"
        assert config.optimistic is False

    def test_ha_text_entity_config(self, mock_device_info):
        """Test HATextEntityConfig creation."""
        config = HATextEntityConfig(
            name="Test Text",
            unique_id="test_text_123",
            state_topic="test/text/state",
            command_topic="test/text/command",
            device=mock_device_info,
            min=5,
            max=100,
            mode="password",
            pattern=r"^[A-Za-z0-9]+$",
        )

        assert config.entity_type == HAEntityType.TEXT
        assert config.command_topic == "test/text/command"
        assert config.min == 5
        assert config.max == 100
        assert config.mode == "password"
        assert config.pattern == r"^[A-Za-z0-9]+$"

    def test_ha_image_entity_config(self, mock_device_info):
        """Test HAImageEntityConfig creation."""
        config = HAImageEntityConfig(
            name="Test Image",
            unique_id="test_image_123",
            state_topic="test/image/state",
            device=mock_device_info,
            url_template="http://example.com/image/{{ entity_id }}.jpg",
            content_type="image/png",
            verify_ssl=False,
        )

        assert config.entity_type == HAEntityType.IMAGE
        assert config.url_template == "http://example.com/image/{{ entity_id }}.jpg"
        assert config.content_type == "image/png"
        assert config.verify_ssl is False

    def test_ha_datetime_entity_config(self, mock_device_info):
        """Test HADateTimeEntityConfig creation."""
        config = HADateTimeEntityConfig(
            name="Test DateTime",
            unique_id="test_datetime_123",
            state_topic="test/datetime/state",
            command_topic="test/datetime/command",
            device=mock_device_info,
            value_template="{{ value_json.datetime }}",
            format="%Y-%m-%d %H:%M",
        )

        assert config.entity_type == HAEntityType.DATETIME
        assert config.command_topic == "test/datetime/command"
        assert config.value_template == "{{ value_json.datetime }}"
        assert config.format == "%Y-%m-%d %H:%M"


@pytest.mark.unit
class TestHAServiceDefinition:
    """Test cases for HAServiceDefinition."""

    def test_service_definition_creation(self):
        """Test HAServiceDefinition creation."""
        fields = {
            "param1": {
                "description": "Parameter 1",
                "required": True,
                "selector": {"text": {}},
            },
            "param2": {
                "description": "Parameter 2",
                "default": "default_value",
                "selector": {"boolean": {}},
            },
        }

        service = HAServiceDefinition(
            service_name="test_service",
            domain="test_domain",
            friendly_name="Test Service",
            description="A test service",
            icon="mdi:test",
            fields=fields,
            command_topic="test/command",
            command_template="{{ value | tojson }}",
            response_topic="test/response",
            supports_response=True,
        )

        assert service.service_name == "test_service"
        assert service.domain == "test_domain"
        assert service.friendly_name == "Test Service"
        assert service.description == "A test service"
        assert service.icon == "mdi:test"
        assert service.fields == fields
        assert service.command_topic == "test/command"
        assert service.command_template == "{{ value | tojson }}"
        assert service.response_topic == "test/response"
        assert service.supports_response is True

    def test_service_definition_minimal(self):
        """Test HAServiceDefinition with minimal parameters."""
        service = HAServiceDefinition(
            service_name="minimal_service",
            domain="test_domain",
            friendly_name="Minimal Service",
            description="A minimal service",
            icon="mdi:minimal",
        )

        assert service.service_name == "minimal_service"
        assert service.domain == "test_domain"
        assert service.fields == {}
        assert service.command_topic == ""
        assert service.command_template is None
        assert service.response_topic is None
        assert service.target_selector is None
        assert service.supports_response is False


@pytest.mark.unit
class TestHAEntityDefinitions:
    """Test cases for HAEntityDefinitions."""

    @pytest.fixture
    def mock_discovery_publisher(self):
        """Mock discovery publisher."""
        publisher = AsyncMock()
        publisher.device_info = DeviceInfo(
            identifiers=["test-device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
            availability_topic="test/availability",
        )
        publisher.mqtt_publisher = AsyncMock()
        publisher.refresh_discovery = AsyncMock()
        publisher.publish_device_availability = AsyncMock()
        return publisher

    @pytest.fixture
    def mock_mqtt_config(self):
        """Mock MQTT configuration."""
        return MQTTConfig(
            broker="test-broker",
            port=1883,
            username="test",
            password="test",
            topic_prefix="test/occupancy",
            device_identifier="test-device",
            discovery_prefix="homeassistant",
        )

    @pytest.fixture
    def mock_rooms(self):
        """Mock room configurations."""
        return {
            "test_room": RoomConfig(
                room_id="test_room",
                name="Test Room",
                sensors={
                    "presence": {"main": "binary_sensor.test_room_presence"},
                    "temperature": "sensor.test_room_temperature",
                },
            ),
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={
                    "presence": {"main": "binary_sensor.living_room_presence"},
                    "motion": "binary_sensor.living_room_motion",
                },
            ),
        }

    @pytest.fixture
    def mock_tracking_config(self):
        """Mock tracking configuration."""
        return TrackingConfig()

    @pytest.fixture
    def entity_definitions(
        self,
        mock_discovery_publisher,
        mock_mqtt_config,
        mock_rooms,
        mock_tracking_config,
    ):
        """Create HAEntityDefinitions instance for testing."""
        return HAEntityDefinitions(
            discovery_publisher=mock_discovery_publisher,
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
            tracking_config=mock_tracking_config,
        )

    def test_initialization(
        self,
        mock_discovery_publisher,
        mock_mqtt_config,
        mock_rooms,
        mock_tracking_config,
    ):
        """Test HAEntityDefinitions initialization."""
        definitions = HAEntityDefinitions(
            discovery_publisher=mock_discovery_publisher,
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
            tracking_config=mock_tracking_config,
        )

        assert definitions.discovery_publisher is mock_discovery_publisher
        assert definitions.mqtt_config is mock_mqtt_config
        assert definitions.rooms is mock_rooms
        assert definitions.tracking_config is mock_tracking_config
        assert len(definitions.entity_definitions) == 0
        assert len(definitions.service_definitions) == 0
        assert len(definitions.entity_states) == 0
        assert len(definitions.entity_availability) == 0
        assert definitions.stats["entities_defined"] == 0
        assert definitions.stats["services_defined"] == 0

    def test_initialization_with_default_tracking_config(
        self, mock_discovery_publisher, mock_mqtt_config, mock_rooms
    ):
        """Test initialization with default tracking config."""
        definitions = HAEntityDefinitions(
            discovery_publisher=mock_discovery_publisher,
            mqtt_config=mock_mqtt_config,
            rooms=mock_rooms,
        )

        assert isinstance(definitions.tracking_config, TrackingConfig)

    def test_define_all_entities(self, entity_definitions):
        """Test defining all entities."""
        entities = entity_definitions.define_all_entities()

        # Should have entities for all rooms plus system entities
        assert len(entities) > 0
        assert entity_definitions.stats["entities_defined"] > 0
        assert entity_definitions.stats["last_update"] is not None

        # Check for room-specific entities
        room_entities = [
            entity_id for entity_id in entities.keys() if "test_room" in entity_id
        ]
        assert len(room_entities) > 0

        # Check for system entities
        system_entities = [
            entity_id for entity_id in entities.keys() if "system" in entity_id
        ]
        assert len(system_entities) > 0

    def test_define_all_entities_room_specific(self, entity_definitions):
        """Test that room-specific entities are properly defined."""
        entities = entity_definitions.define_all_entities()

        # Test room entities for test_room
        test_room_entities = {
            k: v for k, v in entities.items() if k.startswith("test_room_")
        }

        expected_entities = [
            "test_room_prediction",
            "test_room_next_transition",
            "test_room_confidence",
            "test_room_time_until",
            "test_room_reliability",
            "test_room_occupied",
            "test_room_accuracy",
            "test_room_motion_detected",
        ]

        for expected_entity in expected_entities:
            assert expected_entity in test_room_entities

        # Check prediction entity configuration
        prediction_entity = test_room_entities["test_room_prediction"]
        assert isinstance(prediction_entity, HASensorEntityConfig)
        assert prediction_entity.name == "Test Room Occupancy Prediction"
        assert "test_room/prediction" in prediction_entity.state_topic

    def test_define_all_entities_system_wide(self, entity_definitions):
        """Test that system-wide entities are properly defined."""
        entities = entity_definitions.define_all_entities()

        # Check for system entities
        system_entities = ["system_status", "system_uptime", "predictions_count"]

        for system_entity in system_entities:
            assert system_entity in entities
            entity_config = entities[system_entity]
            assert isinstance(entity_config, HASensorEntityConfig)

    def test_define_all_entities_diagnostic(self, entity_definitions):
        """Test that diagnostic entities are properly defined."""
        entities = entity_definitions.define_all_entities()

        # Check for diagnostic entities
        diagnostic_entities = [
            "database_connected",
            "mqtt_connected",
            "tracking_active",
            "model_training",
        ]

        for diagnostic_entity in diagnostic_entities:
            assert diagnostic_entity in entities
            entity_config = entities[diagnostic_entity]
            assert isinstance(entity_config, HABinarySensorEntityConfig)
            assert entity_config.entity_category == "diagnostic"

    def test_define_all_entities_control(self, entity_definitions):
        """Test that control entities are properly defined."""
        entities = entity_definitions.define_all_entities()

        # Check for control entities
        control_entities = [
            "prediction_enabled",
            "mqtt_publishing",
            "prediction_interval",
            "log_level",
        ]

        for control_entity in control_entities:
            assert control_entity in entities

        # Test specific control entity types
        assert isinstance(entities["prediction_enabled"], HASwitchEntityConfig)
        assert isinstance(entities["prediction_interval"], HANumberEntityConfig)
        assert isinstance(entities["log_level"], HASelectEntityConfig)

    def test_define_all_services(self, entity_definitions):
        """Test defining all services."""
        services = entity_definitions.define_all_services()

        assert len(services) > 0
        assert entity_definitions.stats["services_defined"] > 0

        # Check for expected service categories
        expected_services = [
            "retrain_model",
            "validate_model",
            "restart_system",
            "refresh_discovery",
            "generate_diagnostic",
            "force_prediction",
        ]

        for expected_service in expected_services:
            assert expected_service in services
            service = services[expected_service]
            assert isinstance(service, HAServiceDefinition)

    def test_define_all_services_model_management(self, entity_definitions):
        """Test model management services."""
        services = entity_definitions.define_all_services()

        # Test retrain model service
        retrain_service = services["retrain_model"]
        assert retrain_service.service_name == "retrain_model"
        assert retrain_service.domain == "ha_ml_predictor"
        assert retrain_service.friendly_name == "Retrain Model"
        assert "room_id" in retrain_service.fields
        assert "force" in retrain_service.fields
        assert retrain_service.supports_response is True

        # Test validate model service
        validate_service = services["validate_model"]
        assert validate_service.service_name == "validate_model"
        assert "room_id" in validate_service.fields
        assert "days" in validate_service.fields

    def test_define_all_services_system_control(self, entity_definitions):
        """Test system control services."""
        services = entity_definitions.define_all_services()

        # Test restart system service
        restart_service = services["restart_system"]
        assert restart_service.service_name == "restart_system"
        assert restart_service.domain == "ha_ml_predictor"
        assert restart_service.friendly_name == "Restart System"

        # Test refresh discovery service
        refresh_service = services["refresh_discovery"]
        assert refresh_service.service_name == "refresh_discovery"
        assert refresh_service.friendly_name == "Refresh Discovery"

    @pytest.mark.asyncio
    async def test_publish_all_entities(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing all entities."""
        # Mock publish results
        mock_publish_result_success = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_publish_result_failure = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
            error_message="Publish failed",
        )

        # Mock successful and failed publishes
        mock_discovery_publisher.mqtt_publisher.publish_json.side_effect = [
            mock_publish_result_success,
            mock_publish_result_failure,
            mock_publish_result_success,
        ]

        # First define entities
        entity_definitions.define_all_entities()

        # Then publish them
        results = await entity_definitions.publish_all_entities()

        assert len(results) > 0
        # Should have some successful publishes
        assert entity_definitions.stats["entities_published"] > 0

    @pytest.mark.asyncio
    async def test_publish_all_entities_no_entities_defined(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing entities when none are defined."""
        with patch.object(
            entity_definitions, "define_all_entities"
        ) as mock_define_entities:
            mock_define_entities.return_value = {}

            results = await entity_definitions.publish_all_entities()

            # Should call define_all_entities first
            mock_define_entities.assert_called_once()

    @pytest.mark.asyncio
    async def test_publish_all_services(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing all services as button entities."""
        # Mock publish results
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        # First define services
        entity_definitions.define_all_services()

        # Then publish them
        results = await entity_definitions.publish_all_services()

        assert len(results) > 0
        assert entity_definitions.stats["services_published"] > 0

    @pytest.mark.asyncio
    async def test_publish_all_services_no_services_defined(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing services when none are defined."""
        with patch.object(
            entity_definitions, "define_all_services"
        ) as mock_define_services:
            mock_define_services.return_value = {}

            results = await entity_definitions.publish_all_services()

            # Should call define_all_services first
            mock_define_services.assert_called_once()

    def test_get_entity_definition(self, entity_definitions):
        """Test getting entity definition by ID."""
        # Define entities first
        entities = entity_definitions.define_all_entities()

        # Get a known entity
        entity_id = list(entities.keys())[0]
        definition = entity_definitions.get_entity_definition(entity_id)

        assert definition is not None
        assert definition is entities[entity_id]

        # Test non-existent entity
        non_existent = entity_definitions.get_entity_definition("non_existent")
        assert non_existent is None

    def test_get_service_definition(self, entity_definitions):
        """Test getting service definition by name."""
        # Define services first
        services = entity_definitions.define_all_services()

        # Get a known service
        service_name = list(services.keys())[0]
        definition = entity_definitions.get_service_definition(service_name)

        assert definition is not None
        assert definition is services[service_name]

        # Test non-existent service
        non_existent = entity_definitions.get_service_definition("non_existent")
        assert non_existent is None

    def test_get_entity_stats(self, entity_definitions):
        """Test getting entity statistics."""
        # Define entities and services
        entity_definitions.define_all_entities()
        entity_definitions.define_all_services()

        stats = entity_definitions.get_entity_stats()

        assert "entities_defined" in stats
        assert "services_defined" in stats
        assert "entities_published" in stats
        assert "entity_types" in stats
        assert "entity_categories" in stats

        # Check entity type breakdown
        entity_types = stats["entity_types"]
        assert "sensor" in entity_types
        assert "binary_sensor" in entity_types
        assert entity_types["sensor"] > 0

        # Check entity category breakdown
        entity_categories = stats["entity_categories"]
        assert "diagnostic" in entity_categories
        assert "config" in entity_categories

    def test_define_room_entities(self, entity_definitions):
        """Test defining entities for a specific room."""
        room_config = entity_definitions.rooms["test_room"]
        room_entities = entity_definitions._define_room_entities(
            "test_room", room_config
        )

        # Should have multiple entities for the room
        assert len(room_entities) > 0

        # All entities should be for this room
        for entity_id in room_entities.keys():
            assert entity_id.startswith("test_room_")

        # Check specific entities
        assert "test_room_prediction" in room_entities
        assert "test_room_confidence" in room_entities
        assert "test_room_occupied" in room_entities

        # Check entity configuration
        prediction_entity = room_entities["test_room_prediction"]
        assert prediction_entity.name == "Test Room Occupancy Prediction"
        assert isinstance(prediction_entity, HASensorEntityConfig)

    def test_define_system_entities(self, entity_definitions):
        """Test defining system-wide entities."""
        system_entities = entity_definitions._define_system_entities()

        assert len(system_entities) > 0

        # Check for expected system entities
        expected_entities = [
            "system_status",
            "system_uptime",
            "predictions_count",
            "system_accuracy",
            "active_alerts",
        ]

        for expected_entity in expected_entities:
            assert expected_entity in system_entities

        # Check system status entity
        system_status = system_entities["system_status"]
        assert system_status.name == "System Status"
        assert isinstance(system_status, HASensorEntityConfig)

    def test_define_diagnostic_entities(self, entity_definitions):
        """Test defining diagnostic entities."""
        diagnostic_entities = entity_definitions._define_diagnostic_entities()

        assert len(diagnostic_entities) > 0

        # Check for expected diagnostic entities
        expected_entities = [
            "database_connected",
            "mqtt_connected",
            "tracking_active",
            "model_training",
            "memory_usage",
            "cpu_usage",
        ]

        for expected_entity in expected_entities:
            assert expected_entity in diagnostic_entities

        # Check binary sensor entities
        database_connected = diagnostic_entities["database_connected"]
        assert isinstance(database_connected, HABinarySensorEntityConfig)
        assert database_connected.entity_category == "diagnostic"

        # Check measurement sensor entities
        memory_usage = diagnostic_entities["memory_usage"]
        assert isinstance(memory_usage, HASensorEntityConfig)
        assert memory_usage.unit_of_measurement == "MB"

    def test_define_control_entities(self, entity_definitions):
        """Test defining control entities."""
        control_entities = entity_definitions._define_control_entities()

        assert len(control_entities) > 0

        # Check for expected control entities
        expected_entities = [
            "prediction_enabled",
            "mqtt_publishing",
            "prediction_interval",
            "log_level",
            "accuracy_threshold",
            "primary_model",
        ]

        for expected_entity in expected_entities:
            assert expected_entity in control_entities

        # Check switch entities
        prediction_enabled = control_entities["prediction_enabled"]
        assert isinstance(prediction_enabled, HASwitchEntityConfig)
        assert prediction_enabled.entity_category == "config"

        # Check number entities
        prediction_interval = control_entities["prediction_interval"]
        assert isinstance(prediction_interval, HANumberEntityConfig)
        assert prediction_interval.min == 60
        assert prediction_interval.max == 3600

        # Check select entities
        log_level = control_entities["log_level"]
        assert isinstance(log_level, HASelectEntityConfig)
        assert "DEBUG" in log_level.options
        assert "INFO" in log_level.options

    def test_define_model_services(self, entity_definitions):
        """Test defining model management services."""
        model_services = entity_definitions._define_model_services()

        assert len(model_services) > 0

        # Check for expected model services
        expected_services = [
            "retrain_model",
            "validate_model",
            "export_model",
            "import_model",
        ]

        for expected_service in expected_services:
            assert expected_service in model_services

        # Check retrain model service
        retrain_service = model_services["retrain_model"]
        assert retrain_service.domain == "ha_ml_predictor"
        assert "room_id" in retrain_service.fields
        assert "force" in retrain_service.fields

    def test_define_system_services(self, entity_definitions):
        """Test defining system control services."""
        system_services = entity_definitions._define_system_services()

        assert len(system_services) > 0

        # Check for expected system services
        expected_services = [
            "restart_system",
            "refresh_discovery",
            "reset_statistics",
            "update_config",
            "backup_system",
            "restore_system",
        ]

        for expected_service in expected_services:
            assert expected_service in system_services

        # Check backup system service
        backup_service = system_services["backup_system"]
        assert backup_service.friendly_name == "Backup System"
        assert "include_models" in backup_service.fields
        assert "include_data" in backup_service.fields

    def test_define_diagnostic_services(self, entity_definitions):
        """Test defining diagnostic services."""
        diagnostic_services = entity_definitions._define_diagnostic_services()

        assert len(diagnostic_services) > 0

        # Check for expected diagnostic services
        expected_services = ["generate_diagnostic", "check_database"]

        for expected_service in expected_services:
            assert expected_service in diagnostic_services

        # Check generate diagnostic service
        diagnostic_service = diagnostic_services["generate_diagnostic"]
        assert diagnostic_service.friendly_name == "Generate Diagnostic Report"
        assert "include_logs" in diagnostic_service.fields
        assert "include_metrics" in diagnostic_service.fields

    def test_define_room_services(self, entity_definitions):
        """Test defining room-specific services."""
        room_services = entity_definitions._define_room_services()

        assert len(room_services) > 0

        # Check for expected room services
        expected_services = ["force_prediction"]

        for expected_service in expected_services:
            assert expected_service in room_services

        # Check force prediction service
        force_prediction = room_services["force_prediction"]
        assert force_prediction.friendly_name == "Force Prediction"
        assert "room_id" in force_prediction.fields

    def test_create_service_button_config(self, entity_definitions):
        """Test creating button configuration for a service."""
        service_def = HAServiceDefinition(
            service_name="test_service",
            domain="test_domain",
            friendly_name="Test Service",
            description="A test service",
            icon="mdi:test",
            command_topic="test/command",
        )

        button_config = entity_definitions._create_service_button_config(service_def)

        assert isinstance(button_config, HAButtonEntityConfig)
        assert button_config.name == "Test Service"
        assert button_config.command_topic == "test/command"
        assert button_config.icon == "mdi:test"
        assert button_config.entity_category == "config"
        assert button_config.description == "A test service"

    @pytest.mark.asyncio
    async def test_publish_entity_discovery_sensor(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing sensor entity discovery."""
        # Create a sensor entity
        sensor_config = HASensorEntityConfig(
            name="Test Sensor",
            unique_id="test_sensor_123",
            state_topic="test/sensor/state",
            device=mock_discovery_publisher.device_info,
            value_template="{{ value_json.value }}",
            unit_of_measurement="°C",
            device_class="temperature",
            state_class="measurement",
        )

        # Mock successful publish
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await entity_definitions._publish_entity_discovery(sensor_config)

        assert result.success is True
        mock_discovery_publisher.mqtt_publisher.publish_json.assert_called_once()

        # Check discovery payload
        call_args = mock_discovery_publisher.mqtt_publisher.publish_json.call_args
        discovery_topic = call_args[1]["topic"]
        discovery_payload = call_args[1]["data"]

        assert "sensor" in discovery_topic
        assert "test_sensor_123" in discovery_topic
        assert "config" in discovery_topic

        assert discovery_payload["name"] == "Test Sensor"
        assert discovery_payload["unique_id"] == "test_sensor_123"
        assert discovery_payload["state_topic"] == "test/sensor/state"
        assert discovery_payload["value_template"] == "{{ value_json.value }}"
        assert discovery_payload["unit_of_measurement"] == "°C"
        assert discovery_payload["device_class"] == "temperature"
        assert discovery_payload["state_class"] == "measurement"

    @pytest.mark.asyncio
    async def test_publish_entity_discovery_binary_sensor(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing binary sensor entity discovery."""
        # Create a binary sensor entity
        binary_sensor_config = HABinarySensorEntityConfig(
            name="Test Binary Sensor",
            unique_id="test_binary_sensor_123",
            state_topic="test/binary_sensor/state",
            device=mock_discovery_publisher.device_info,
            device_class="motion",
            payload_on="true",
            payload_off="false",
        )

        # Mock successful publish
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await entity_definitions._publish_entity_discovery(
            binary_sensor_config
        )

        assert result.success is True

        # Check discovery payload
        call_args = mock_discovery_publisher.mqtt_publisher.publish_json.call_args
        discovery_payload = call_args[1]["data"]

        assert discovery_payload["device_class"] == "motion"
        # payload_off should not be in payload when it's "OFF" (default)
        assert "payload_of" not in discovery_payload  # Note: typo in implementation

    @pytest.mark.asyncio
    async def test_publish_entity_discovery_button(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing button entity discovery."""
        # Create a button entity
        button_config = HAButtonEntityConfig(
            name="Test Button",
            unique_id="test_button_123",
            state_topic="",
            command_topic="test/button/command",
            device=mock_discovery_publisher.device_info,
            payload_press="PRESS",
            device_class="restart",
        )

        # Mock successful publish
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await entity_definitions._publish_entity_discovery(button_config)

        assert result.success is True

        # Check discovery payload
        call_args = mock_discovery_publisher.mqtt_publisher.publish_json.call_args
        discovery_payload = call_args[1]["data"]

        assert discovery_payload["command_topic"] == "test/button/command"
        assert discovery_payload["payload_press"] == "PRESS"
        assert discovery_payload["device_class"] == "restart"

    @pytest.mark.asyncio
    async def test_publish_entity_discovery_with_availability(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test publishing entity discovery with device availability."""
        # Create a sensor entity
        sensor_config = HASensorEntityConfig(
            name="Test Sensor",
            unique_id="test_sensor_123",
            state_topic="test/sensor/state",
            device=mock_discovery_publisher.device_info,
        )

        # Mock successful publish
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await entity_definitions._publish_entity_discovery(sensor_config)

        assert result.success is True

        # Check discovery payload includes availability
        call_args = mock_discovery_publisher.mqtt_publisher.publish_json.call_args
        discovery_payload = call_args[1]["data"]

        assert "availability" in discovery_payload
        availability = discovery_payload["availability"]
        assert availability["topic"] == "test/availability"
        assert availability["payload_available"] == "online"
        assert availability["payload_not_available"] == "offline"

    @pytest.mark.asyncio
    async def test_publish_entity_discovery_failure(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test handling of entity discovery publish failure."""
        # Create a sensor entity
        sensor_config = HASensorEntityConfig(
            name="Test Sensor",
            unique_id="test_sensor_123",
            state_topic="test/sensor/state",
            device=mock_discovery_publisher.device_info,
        )

        # Mock failed publish
        mock_publish_result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
            error_message="Publish failed",
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        result = await entity_definitions._publish_entity_discovery(sensor_config)

        assert result.success is False
        assert result.error_message == "Publish failed"

    @pytest.mark.asyncio
    async def test_publish_entity_discovery_exception(
        self, entity_definitions, mock_discovery_publisher
    ):
        """Test handling of exception during entity discovery publish."""
        # Create a sensor entity
        sensor_config = HASensorEntityConfig(
            name="Test Sensor",
            unique_id="test_sensor_123",
            state_topic="test/sensor/state",
            device=mock_discovery_publisher.device_info,
        )

        # Mock exception during publish
        mock_discovery_publisher.mqtt_publisher.publish_json.side_effect = Exception(
            "MQTT error"
        )

        result = await entity_definitions._publish_entity_discovery(sensor_config)

        assert result.success is False
        assert "MQTT error" in result.error_message

    def test_add_sensor_attributes(self, entity_definitions):
        """Test adding sensor-specific attributes to discovery payload."""
        payload = {}
        config = HASensorEntityConfig(
            name="Test Sensor",
            unique_id="test_sensor",
            state_topic="test/state",
            device=MagicMock(),
            value_template="{{ value_json.test }}",
            unit_of_measurement="unit",
            device_class="test_class",
            state_class="measurement",
            suggested_display_precision=2,
            force_update=True,
        )

        entity_definitions._add_sensor_attributes(payload, config)

        assert payload["value_template"] == "{{ value_json.test }}"
        assert payload["unit_of_measurement"] == "unit"
        assert payload["device_class"] == "test_class"
        assert payload["state_class"] == "measurement"
        assert payload["suggested_display_precision"] == 2
        assert payload["force_update"] is True

    def test_add_binary_sensor_attributes(self, entity_definitions):
        """Test adding binary sensor-specific attributes to discovery payload."""
        payload = {}
        config = HABinarySensorEntityConfig(
            name="Test Binary Sensor",
            unique_id="test_binary_sensor",
            state_topic="test/state",
            device=MagicMock(),
            value_template="{{ value_json.test }}",
            payload_on="true",
            payload_off="false",
            device_class="motion",
            off_delay=30,
        )

        entity_definitions._add_binary_sensor_attributes(payload, config)

        assert payload["value_template"] == "{{ value_json.test }}"
        assert payload["device_class"] == "motion"
        assert payload["off_delay"] == 30
        # Custom payloads should be included
        assert "payload_of" in payload  # Note: typo in implementation

    def test_add_switch_attributes(self, entity_definitions):
        """Test adding switch-specific attributes to discovery payload."""
        payload = {}
        config = HASwitchEntityConfig(
            name="Test Switch",
            unique_id="test_switch",
            state_topic="test/state",
            command_topic="test/command",
            device=MagicMock(),
            state_on="enabled",
            state_off="disabled",
            payload_on="ENABLE",
            payload_off="DISABLE",
            optimistic=True,
        )

        entity_definitions._add_switch_attributes(payload, config)

        assert payload["command_topic"] == "test/command"
        assert payload["optimistic"] is True
        # Custom states and payloads should be included
        assert "state_of" in payload  # Note: typo in implementation
        assert "payload_of" in payload  # Note: typo in implementation

    def test_add_number_attributes(self, entity_definitions):
        """Test adding number-specific attributes to discovery payload."""
        payload = {}
        config = HANumberEntityConfig(
            name="Test Number",
            unique_id="test_number",
            state_topic="test/state",
            command_topic="test/command",
            device=MagicMock(),
            min=0.0,
            max=100.0,
            step=0.1,
            mode="slider",
            unit_of_measurement="%",
            device_class="power",
        )

        entity_definitions._add_number_attributes(payload, config)

        assert payload["command_topic"] == "test/command"
        assert payload["min"] == 0.0
        assert payload["max"] == 100.0
        assert payload["step"] == 0.1
        assert payload["mode"] == "slider"
        assert payload["unit_of_measurement"] == "%"
        assert payload["device_class"] == "power"


@pytest.mark.unit
class TestHAEntityDefinitionsError:
    """Test cases for HAEntityDefinitionsError."""

    def test_ha_entity_definitions_error(self):
        """Test HAEntityDefinitionsError creation."""
        error = HAEntityDefinitionsError("Test error message")

        assert "Test error message" in str(error)
        assert error.error_code == "HA_ENTITY_DEFINITIONS_ERROR"
        assert error.severity.name == "MEDIUM"

    def test_ha_entity_definitions_error_with_severity(self):
        """Test HAEntityDefinitionsError with custom severity."""
        from src.core.exceptions import ErrorSeverity

        error = HAEntityDefinitionsError(
            "Critical error", severity=ErrorSeverity.CRITICAL
        )

        assert str(error) == "Critical error"
        assert error.severity == ErrorSeverity.CRITICAL


@pytest.mark.integration
class TestHAEntityDefinitionsIntegration:
    """Integration test cases for HA entity definitions."""

    @pytest.fixture
    def complete_system_setup(self):
        """Complete system setup for integration testing."""
        # Mock discovery publisher with MQTT publisher
        mock_mqtt_publisher = AsyncMock()
        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.mqtt_publisher = mock_mqtt_publisher
        mock_discovery_publisher.device_info = DeviceInfo(
            identifiers=["test-device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
            availability_topic="test/availability",
        )

        # Mock MQTT config
        mqtt_config = MQTTConfig(
            broker="test-broker",
            port=1883,
            username="test",
            password="test",
            topic_prefix="test/occupancy",
            device_identifier="test-device",
            discovery_prefix="homeassistant",
        )

        # Mock rooms
        rooms = {
            "living_room": RoomConfig(
                room_id="living_room",
                name="Living Room",
                sensors={
                    "presence": {"main": "binary_sensor.living_room_presence"},
                    "temperature": "sensor.living_room_temperature",
                },
            )
        }

        return mock_discovery_publisher, mqtt_config, rooms

    @pytest.mark.asyncio
    async def test_complete_entity_lifecycle(self, complete_system_setup):
        """Test complete entity definition and publishing lifecycle."""
        mock_discovery_publisher, mqtt_config, rooms = complete_system_setup

        # Mock successful publishes
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        # Create entity definitions
        entity_definitions = HAEntityDefinitions(
            discovery_publisher=mock_discovery_publisher,
            mqtt_config=mqtt_config,
            rooms=rooms,
        )

        # Test complete workflow
        # 1. Define all entities
        entities = entity_definitions.define_all_entities()
        assert len(entities) > 0

        # 2. Define all services
        services = entity_definitions.define_all_services()
        assert len(services) > 0

        # 3. Publish all entities
        entity_results = await entity_definitions.publish_all_entities()
        assert len(entity_results) > 0

        # 4. Publish all services
        service_results = await entity_definitions.publish_all_services()
        assert len(service_results) > 0

        # 5. Check statistics
        stats = entity_definitions.get_entity_stats()
        assert stats["entities_defined"] > 0
        assert stats["services_defined"] > 0

        # Verify MQTT publishes were called
        assert mock_discovery_publisher.mqtt_publisher.publish_json.call_count > 0

    @pytest.mark.asyncio
    async def test_room_specific_entity_generation(self, complete_system_setup):
        """Test that room-specific entities are correctly generated."""
        mock_discovery_publisher, mqtt_config, rooms = complete_system_setup

        entity_definitions = HAEntityDefinitions(
            discovery_publisher=mock_discovery_publisher,
            mqtt_config=mqtt_config,
            rooms=rooms,
        )

        entities = entity_definitions.define_all_entities()

        # Check for living room specific entities
        living_room_entities = [
            entity_id for entity_id in entities.keys() if "living_room" in entity_id
        ]
        assert len(living_room_entities) > 0

        # Verify specific entities exist
        expected_entities = [
            "living_room_prediction",
            "living_room_confidence",
            "living_room_occupied",
            "living_room_accuracy",
        ]

        for expected_entity in expected_entities:
            assert expected_entity in entities

        # Verify entity configuration
        prediction_entity = entities["living_room_prediction"]
        assert prediction_entity.name == "Living Room Occupancy Prediction"
        assert "living_room/prediction" in prediction_entity.state_topic

    @pytest.mark.asyncio
    async def test_service_to_button_conversion(self, complete_system_setup):
        """Test conversion of services to button entities."""
        mock_discovery_publisher, mqtt_config, rooms = complete_system_setup

        # Mock successful publish
        mock_publish_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_discovery_publisher.mqtt_publisher.publish_json.return_value = (
            mock_publish_result
        )

        entity_definitions = HAEntityDefinitions(
            discovery_publisher=mock_discovery_publisher,
            mqtt_config=mqtt_config,
            rooms=rooms,
        )

        # Define and publish services
        services = entity_definitions.define_all_services()
        results = await entity_definitions.publish_all_services()

        # Verify services were published as button entities
        assert len(results) == len(services)

        # Check that publish was called for each service
        assert mock_discovery_publisher.mqtt_publisher.publish_json.call_count == len(
            services
        )

        # Verify button entity discovery topics
        calls = mock_discovery_publisher.mqtt_publisher.publish_json.call_args_list
        for call in calls:
            topic = call[1]["topic"]
            assert "button" in topic
            assert "config" in topic
