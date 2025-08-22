"""
Comprehensive tests for Home Assistant MQTT Discovery Publisher.

This test module provides extensive coverage of the enhanced discovery publisher
functionality including MQTT discovery message creation, device availability
tracking, entity lifecycle management, and Home Assistant integration.

Coverage Areas:
- Discovery publisher initialization and configuration
- Device information management and availability tracking
- MQTT discovery message creation and publishing
- Room-specific sensor discovery
- System status sensor discovery
- Service discovery for manual controls
- Entity state management and metadata
- Discovery message removal and cleanup
- Error handling and edge cases
- Security scenarios and validation
- Enhanced features and capabilities
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

import pytest

from src.core.config import MQTTConfig, RoomConfig
from src.core.exceptions import ErrorSeverity, OccupancyPredictionError
from src.integration.discovery_publisher import (
    DeviceClass,
    DeviceInfo,
    DiscoveryPublisher,
    EnhancedDiscoveryError,
    EntityAvailability,
    EntityCategory,
    EntityMetadata,
    EntityState,
    SensorConfig,
    ServiceConfig,
)
from src.integration.mqtt_publisher import MQTTPublisher, MQTTPublishResult

# Test Fixtures


@pytest.fixture
def mqtt_config():
    """Create MQTT configuration for testing."""
    return MQTTConfig(
        broker="localhost",
        port=1883,
        username="test_user",
        password="test_pass",
        topic_prefix="occupancy",
        discovery_enabled=True,
        discovery_prefix="homeassistant",
        device_identifier="ha_ml_predictor",
        device_name="HA ML Predictor",
        device_manufacturer="Home Assistant Community",
        device_model="ML Occupancy Predictor v1.0",
        device_sw_version="1.0.0",
        publishing_enabled=True,
        status_update_interval_seconds=30,
        publish_system_status=True,
    )


@pytest.fixture
def room_configs():
    """Create room configurations for testing."""
    return {
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
                "occupancy": ["binary_sensor.kitchen_occupancy"],
            },
        ),
    }


@pytest.fixture
def mock_mqtt_publisher():
    """Create mock MQTT publisher."""
    publisher = AsyncMock(spec=MQTTPublisher)

    # Mock successful publish results
    success_result = MQTTPublishResult(
        success=True,
        topic="test/topic",
        payload_size=100,
        publish_time=datetime.utcnow(),
        message_id=1,
    )

    publisher.publish.return_value = success_result
    publisher.publish_json.return_value = success_result

    # Mock connection status
    publisher.connection_status = Mock()
    publisher.connection_status.connected = True

    return publisher


@pytest.fixture
def mock_availability_callback():
    """Create mock availability check callback."""
    callback = AsyncMock()
    callback.return_value = True
    return callback


@pytest.fixture
def mock_state_change_callback():
    """Create mock state change callback."""
    return AsyncMock()


@pytest.fixture
def discovery_publisher(
    mqtt_config,
    room_configs,
    mock_mqtt_publisher,
    mock_availability_callback,
    mock_state_change_callback,
):
    """Create discovery publisher instance."""
    return DiscoveryPublisher(
        mqtt_publisher=mock_mqtt_publisher,
        config=mqtt_config,
        rooms=room_configs,
        availability_check_callback=mock_availability_callback,
        state_change_callback=mock_state_change_callback,
    )


# Discovery Publisher Initialization Tests


class TestDiscoveryPublisherInit:
    """Test discovery publisher initialization."""

    def test_discovery_publisher_initialization(
        self, mqtt_config, room_configs, mock_mqtt_publisher
    ):
        """Test discovery publisher initialization."""
        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
        )

        assert publisher.mqtt_publisher == mock_mqtt_publisher
        assert publisher.config == mqtt_config
        assert publisher.rooms == room_configs
        assert publisher.discovery_published is False
        assert len(publisher.published_entities) == 0
        assert len(publisher.entity_metadata) == 0
        assert publisher.device_available is True
        assert isinstance(publisher.device_info, DeviceInfo)

    def test_device_info_creation(self, mqtt_config, room_configs, mock_mqtt_publisher):
        """Test device information creation."""
        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
        )

        device_info = publisher.device_info

        assert device_info.identifiers == [mqtt_config.device_identifier]
        assert device_info.name == mqtt_config.device_name
        assert device_info.manufacturer == mqtt_config.device_manufacturer
        assert device_info.model == mqtt_config.device_model
        assert device_info.sw_version == mqtt_config.device_sw_version
        assert device_info.suggested_area == "Home Assistant ML Predictor"
        assert (
            device_info.availability_topic
            == f"{mqtt_config.topic_prefix}/device/availability"
        )
        assert "prediction_types" in device_info.capabilities
        assert device_info.capabilities["rooms_supported"] == len(room_configs)

    def test_initialization_with_callbacks(
        self, mqtt_config, room_configs, mock_mqtt_publisher
    ):
        """Test initialization with callback functions."""
        availability_cb = AsyncMock()
        state_change_cb = AsyncMock()

        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
            availability_check_callback=availability_cb,
            state_change_callback=state_change_cb,
        )

        assert publisher.availability_check_callback == availability_cb
        assert publisher.state_change_callback == state_change_cb

    def test_discovery_stats_initialization(self, discovery_publisher):
        """Test discovery statistics initialization."""
        stats = discovery_publisher.discovery_stats

        assert stats["entities_created"] == 0
        assert stats["entities_removed"] == 0
        assert stats["discovery_publishes"] == 0
        assert stats["discovery_errors"] == 0
        assert stats["last_discovery_refresh"] is None
        assert stats["availability_updates"] == 0


# Device Information Tests


class TestDeviceInfo:
    """Test device information functionality."""

    def test_device_info_creation(self):
        """Test device info dataclass creation."""
        device_info = DeviceInfo(
            identifiers=["test_device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model v1.0",
            sw_version="1.0.0",
            configuration_url="http://example.com/config",
            suggested_area="Test Area",
            device_class="connectivity",
            capabilities={"feature1": True, "feature2": False},
        )

        assert device_info.identifiers == ["test_device"]
        assert device_info.name == "Test Device"
        assert device_info.manufacturer == "Test Manufacturer"
        assert device_info.model == "Test Model v1.0"
        assert device_info.sw_version == "1.0.0"
        assert device_info.configuration_url == "http://example.com/config"
        assert device_info.suggested_area == "Test Area"
        assert device_info.device_class == "connectivity"
        assert device_info.capabilities["feature1"] is True
        assert device_info.capabilities["feature2"] is False

    def test_device_info_with_connections(self):
        """Test device info with network connections."""
        device_info = DeviceInfo(
            identifiers=["test_device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
            connections=[["mac", "aa:bb:cc:dd:ee:ff"], ["ip", "192.168.1.100"]],
            hw_version="2.1",
        )

        assert device_info.connections == [
            ["mac", "aa:bb:cc:dd:ee:ff"],
            ["ip", "192.168.1.100"],
        ]
        assert device_info.hw_version == "2.1"

    def test_device_info_diagnostic_info(self):
        """Test device info with diagnostic information."""
        now = datetime.utcnow()
        device_info = DeviceInfo(
            identifiers=["test_device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
            last_seen=now,
            diagnostic_info={
                "uptime": "24h 30m",
                "memory_usage": "45%",
                "cpu_usage": "12%",
            },
        )

        assert device_info.last_seen == now
        assert device_info.diagnostic_info["uptime"] == "24h 30m"
        assert device_info.diagnostic_info["memory_usage"] == "45%"
        assert device_info.diagnostic_info["cpu_usage"] == "12%"


# Entity Configuration Tests


class TestEntityConfiguration:
    """Test entity configuration classes."""

    def test_entity_availability_creation(self):
        """Test entity availability configuration."""
        availability = EntityAvailability(
            topic="test/availability",
            payload_available="online",
            payload_not_available="offline",
            value_template="{{ value_json.status }}",
        )

        assert availability.topic == "test/availability"
        assert availability.payload_available == "online"
        assert availability.payload_not_available == "offline"
        assert availability.value_template == "{{ value_json.status }}"

    def test_service_config_creation(self):
        """Test service configuration creation."""
        service = ServiceConfig(
            service_name="test_service",
            service_topic="test/command",
            schema="json",
            command_template="{{ value_json }}",
            retain=True,
        )

        assert service.service_name == "test_service"
        assert service.service_topic == "test/command"
        assert service.schema == "json"
        assert service.command_template == "{{ value_json }}"
        assert service.retain is True

    def test_entity_metadata_creation(self):
        """Test entity metadata creation."""
        now = datetime.utcnow()
        metadata = EntityMetadata(
            entity_id="test_entity",
            friendly_name="Test Entity",
            created_at=now,
            entity_category=EntityCategory.DIAGNOSTIC,
            state=EntityState.ONLINE,
            attributes={"attribute1": "value1"},
        )

        assert metadata.entity_id == "test_entity"
        assert metadata.friendly_name == "Test Entity"
        assert metadata.created_at == now
        assert metadata.entity_category == EntityCategory.DIAGNOSTIC
        assert metadata.state == EntityState.ONLINE
        assert metadata.attributes["attribute1"] == "value1"

    def test_sensor_config_creation(self):
        """Test sensor configuration creation."""
        device_info = DeviceInfo(
            identifiers=["test_device"],
            name="Test Device",
            manufacturer="Test Manufacturer",
            model="Test Model",
            sw_version="1.0.0",
        )

        sensor = SensorConfig(
            name="Test Sensor",
            unique_id="test_sensor_id",
            state_topic="test/sensor/state",
            device=device_info,
            value_template="{{ value_json.value }}",
            unit_of_measurement="°C",
            device_class="temperature",
            state_class="measurement",
            icon="mdi:thermometer",
            entity_category="diagnostic",
            expire_after=300,
        )

        assert sensor.name == "Test Sensor"
        assert sensor.unique_id == "test_sensor_id"
        assert sensor.state_topic == "test/sensor/state"
        assert sensor.device == device_info
        assert sensor.value_template == "{{ value_json.value }}"
        assert sensor.unit_of_measurement == "°C"
        assert sensor.device_class == "temperature"
        assert sensor.state_class == "measurement"
        assert sensor.icon == "mdi:thermometer"
        assert sensor.entity_category == "diagnostic"
        assert sensor.expire_after == 300


# Discovery Publishing Tests


class TestDiscoveryPublishing:
    """Test discovery message publishing functionality."""

    async def test_publish_all_discovery_success(self, discovery_publisher):
        """Test successful publishing of all discovery messages."""
        results = await discovery_publisher.publish_all_discovery()

        assert isinstance(results, dict)
        assert len(results) > 0

        # Should have room sensors and system sensors
        expected_room_sensors = len(discovery_publisher.rooms) * 5  # 5 sensors per room
        expected_system_sensors = 7  # System sensors
        expected_services = 4  # Service buttons
        expected_total = (
            expected_room_sensors + expected_system_sensors + expected_services
        )

        assert len(results) == expected_total

        # Check that discovery state updated
        assert discovery_publisher.discovery_published is True
        assert discovery_publisher.stats["discovery_publishes"] == 1
        assert discovery_publisher.stats["entities_created"] == len(results)

    async def test_publish_all_discovery_disabled(
        self, mqtt_config, room_configs, mock_mqtt_publisher
    ):
        """Test discovery publishing when disabled."""
        mqtt_config.discovery_enabled = False
        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
        )

        results = await publisher.publish_all_discovery()

        assert results == {}
        assert publisher.discovery_published is False

    async def test_publish_all_discovery_with_failures(
        self, discovery_publisher, mock_mqtt_publisher
    ):
        """Test discovery publishing with some failures."""
        # Make some publishes fail
        success_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )

        failure_result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Publish failed",
        )

        # Alternate between success and failure
        mock_mqtt_publisher.publish_json.side_effect = [
            success_result,
            failure_result,
        ] * 20

        results = await discovery_publisher.publish_all_discovery()

        # Should still return results for both successful and failed
        assert len(results) > 0

        # Check that some failed
        failed_count = sum(1 for r in results.values() if not r.success)
        assert failed_count > 0

        # Discovery should still be marked as published if some succeeded
        successful_count = sum(1 for r in results.values() if r.success)
        assert discovery_publisher.discovery_published == (successful_count > 0)

    async def test_publish_room_discovery(self, discovery_publisher):
        """Test publishing discovery for specific room."""
        room_id = "living_room"
        room_config = discovery_publisher.rooms[room_id]

        results = await discovery_publisher.publish_room_discovery(room_id, room_config)

        assert isinstance(results, dict)
        assert len(results) == 5  # 5 sensors per room

        expected_sensors = [
            f"{room_id}_prediction",
            f"{room_id}_next_transition",
            f"{room_id}_confidence",
            f"{room_id}_time_until",
            f"{room_id}_reliability",
        ]

        for sensor in expected_sensors:
            assert sensor in results
            assert results[sensor].success is True

    async def test_publish_system_discovery(self, discovery_publisher):
        """Test publishing system status sensors."""
        results = await discovery_publisher.publish_system_discovery()

        assert isinstance(results, dict)
        assert len(results) == 7  # 7 system sensors

        expected_sensors = [
            "system_status",
            "system_uptime",
            "predictions_count",
            "system_accuracy",
            "active_alerts",
            "database_status",
            "tracking_status",
        ]

        for sensor in expected_sensors:
            assert sensor in results
            assert results[sensor].success is True

    async def test_publish_service_discovery(self, discovery_publisher):
        """Test publishing service discovery."""
        results = await discovery_publisher.publish_service_discovery()

        assert isinstance(results, dict)
        assert len(results) == 4  # 4 services

        expected_services = [
            "manual_retrain",
            "refresh_discovery",
            "reset_statistics",
            "force_prediction",
        ]

        for service in expected_services:
            assert service in results
            assert results[service].success is True

        # Check that services were added to available services
        assert len(discovery_publisher.available_services) == 4

    async def test_publish_device_availability(self, discovery_publisher):
        """Test publishing device availability."""
        result = await discovery_publisher.publish_device_availability(online=True)

        assert result.success is True
        assert discovery_publisher.device_available is True
        assert discovery_publisher.last_availability_publish is not None
        assert discovery_publisher.stats["availability_updates"] == 1

    async def test_publish_device_availability_offline(self, discovery_publisher):
        """Test publishing device offline status."""
        result = await discovery_publisher.publish_device_availability(online=False)

        assert result.success is True
        assert discovery_publisher.device_available is False
        assert discovery_publisher.last_availability_publish is not None

    async def test_publish_device_availability_no_topic(
        self, mqtt_config, room_configs, mock_mqtt_publisher
    ):
        """Test device availability with no topic configured."""
        # Clear availability topic
        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
        )
        publisher.device_info.availability_topic = None

        result = await publisher.publish_device_availability()

        assert result.success is False
        assert "not configured" in result.error_message


# Discovery Removal and Cleanup Tests


class TestDiscoveryCleanup:
    """Test discovery message removal and cleanup."""

    async def test_remove_discovery_success(self, discovery_publisher):
        """Test successful discovery removal."""
        # First publish discovery
        await discovery_publisher.publish_all_discovery()

        # Get an entity to remove
        entity_name = list(discovery_publisher.published_entities.keys())[0]

        result = await discovery_publisher.remove_discovery(entity_name)

        assert result.success is True
        assert entity_name not in discovery_publisher.published_entities

    async def test_remove_discovery_not_found(self, discovery_publisher):
        """Test removing non-existent entity."""
        result = await discovery_publisher.remove_discovery("nonexistent_entity")

        assert result.success is False
        assert "not found" in result.error_message

    async def test_refresh_discovery(self, discovery_publisher):
        """Test discovery refresh."""
        # First publish
        initial_results = await discovery_publisher.publish_all_discovery()
        assert len(initial_results) > 0

        # Refresh
        refresh_results = await discovery_publisher.refresh_discovery()

        assert len(refresh_results) == len(initial_results)
        assert discovery_publisher.discovery_published is True

    async def test_cleanup_entities_all(self, discovery_publisher):
        """Test cleaning up all entities."""
        # Publish discovery first
        await discovery_publisher.publish_all_discovery()
        initial_count = len(discovery_publisher.published_entities)
        assert initial_count > 0

        # Cleanup all entities
        results = await discovery_publisher.cleanup_entities()

        assert len(results) == initial_count
        assert len(discovery_publisher.published_entities) == 0
        assert len(discovery_publisher.entity_metadata) == 0
        assert discovery_publisher.stats["entities_removed"] == initial_count

    async def test_cleanup_entities_specific(self, discovery_publisher):
        """Test cleaning up specific entities."""
        # Publish discovery first
        await discovery_publisher.publish_all_discovery()

        # Get some entity IDs to cleanup
        entity_ids = list(discovery_publisher.published_entities.keys())[:3]

        results = await discovery_publisher.cleanup_entities(entity_ids)

        assert len(results) == 3
        for entity_id in entity_ids:
            assert entity_id not in discovery_publisher.published_entities
            assert entity_id not in discovery_publisher.entity_metadata


# Entity State Management Tests


class TestEntityStateManagement:
    """Test entity state management functionality."""

    async def test_update_entity_state_success(self, discovery_publisher):
        """Test successful entity state update."""
        # First publish discovery to create entities
        await discovery_publisher.publish_all_discovery()

        # Get an entity ID
        entity_id = list(discovery_publisher.entity_metadata.keys())[0]

        result = await discovery_publisher.update_entity_state(
            entity_id=entity_id,
            state=EntityState.ERROR,
            attributes={"error_message": "Test error"},
        )

        assert result is True

        # Check metadata updated
        metadata = discovery_publisher.entity_metadata[entity_id]
        assert metadata.state == EntityState.ERROR
        assert metadata.attributes["error_message"] == "Test error"
        assert metadata.last_updated is not None
        assert metadata.last_seen is not None

    async def test_update_entity_state_not_found(self, discovery_publisher):
        """Test updating state for non-existent entity."""
        result = await discovery_publisher.update_entity_state(
            entity_id="nonexistent_entity", state=EntityState.ERROR
        )

        assert result is False

    async def test_update_entity_state_with_callback(
        self, discovery_publisher, mock_state_change_callback
    ):
        """Test entity state update with callback."""
        # Publish discovery first
        await discovery_publisher.publish_all_discovery()
        entity_id = list(discovery_publisher.entity_metadata.keys())[0]

        # Update state
        await discovery_publisher.update_entity_state(
            entity_id=entity_id,
            state=EntityState.WARNING,
            attributes={"warning": "test"},
        )

        # Check callback was called
        mock_state_change_callback.assert_called_once_with(
            entity_id, EntityState.WARNING, {"warning": "test"}
        )

    async def test_update_entity_state_callback_error(self, discovery_publisher):
        """Test entity state update with callback error."""
        # Setup callback that raises exception
        error_callback = AsyncMock(side_effect=Exception("Callback error"))
        discovery_publisher.state_change_callback = error_callback

        # Publish discovery first
        await discovery_publisher.publish_all_discovery()
        entity_id = list(discovery_publisher.entity_metadata.keys())[0]

        # Update should still succeed despite callback error
        result = await discovery_publisher.update_entity_state(
            entity_id=entity_id, state=EntityState.ERROR
        )

        assert result is True


# Statistics and Monitoring Tests


class TestDiscoveryStatistics:
    """Test discovery statistics and monitoring."""

    async def test_get_discovery_stats(self, discovery_publisher):
        """Test getting discovery statistics."""
        # Publish some discovery
        await discovery_publisher.publish_all_discovery()

        stats = discovery_publisher.get_discovery_stats()

        assert isinstance(stats, dict)
        assert stats["discovery_enabled"] is True
        assert stats["discovery_published"] is True
        assert stats["published_entities_count"] > 0
        assert stats["entity_metadata_count"] > 0
        assert stats["rooms_configured"] == len(discovery_publisher.rooms)
        assert stats["device_available"] is True
        assert "published_entities" in stats
        assert "device_info" in stats
        assert "statistics" in stats

    def test_get_discovery_stats_empty(self, discovery_publisher):
        """Test getting statistics when no discovery published."""
        stats = discovery_publisher.get_discovery_stats()

        assert stats["discovery_published"] is False
        assert stats["published_entities_count"] == 0
        assert stats["entity_metadata_count"] == 0
        assert stats["available_services_count"] == 0

    async def test_discovery_stats_updates(self, discovery_publisher):
        """Test that statistics update correctly."""
        initial_stats = discovery_publisher.get_discovery_stats()
        assert initial_stats["published_entities_count"] == 0

        # Publish discovery
        await discovery_publisher.publish_all_discovery()

        stats_after_publish = discovery_publisher.get_discovery_stats()
        assert stats_after_publish["published_entities_count"] > 0
        assert stats_after_publish["discovery_published"] is True

        # Cleanup some entities
        entity_ids = list(discovery_publisher.published_entities.keys())[:2]
        await discovery_publisher.cleanup_entities(entity_ids)

        stats_after_cleanup = discovery_publisher.get_discovery_stats()
        assert (
            stats_after_cleanup["published_entities_count"]
            < stats_after_publish["published_entities_count"]
        )


# Sensor Configuration Creation Tests


class TestSensorConfigCreation:
    """Test sensor configuration creation methods."""

    def test_create_prediction_sensor(self, discovery_publisher):
        """Test prediction sensor configuration creation."""
        sensor = discovery_publisher._create_prediction_sensor(
            "living_room", "Living Room"
        )

        assert isinstance(sensor, SensorConfig)
        assert sensor.name == "Living Room Occupancy Prediction"
        assert "living_room_prediction" in sensor.unique_id
        assert sensor.state_topic == "occupancy/living_room/prediction"
        assert sensor.value_template == "{{ value_json.transition_type }}"
        assert sensor.icon == "mdi:home-account"
        assert sensor.entity_category == "diagnostic"
        assert sensor.expire_after == 600

    def test_create_next_transition_sensor(self, discovery_publisher):
        """Test next transition sensor configuration creation."""
        sensor = discovery_publisher._create_next_transition_sensor(
            "bedroom", "Bedroom"
        )

        assert sensor.name == "Bedroom Next Transition"
        assert "bedroom_next_transition" in sensor.unique_id
        assert sensor.value_template == "{{ value_json.predicted_time }}"
        assert sensor.device_class == "timestamp"
        assert sensor.icon == "mdi:clock-outline"

    def test_create_confidence_sensor(self, discovery_publisher):
        """Test confidence sensor configuration creation."""
        sensor = discovery_publisher._create_confidence_sensor("kitchen", "Kitchen")

        assert sensor.name == "Kitchen Confidence"
        assert "kitchen_confidence" in sensor.unique_id
        assert (
            sensor.value_template
            == "{{ (value_json.confidence_score * 100) | round(1) }}"
        )
        assert sensor.unit_of_measurement == "%"
        assert sensor.icon == "mdi:percent"

    def test_create_time_until_sensor(self, discovery_publisher):
        """Test time until sensor configuration creation."""
        sensor = discovery_publisher._create_time_until_sensor(
            "living_room", "Living Room"
        )

        assert sensor.name == "Living Room Time Until"
        assert sensor.value_template == "{{ value_json.time_until_human }}"
        assert sensor.icon == "mdi:timer-outline"

    def test_create_reliability_sensor(self, discovery_publisher):
        """Test reliability sensor configuration creation."""
        sensor = discovery_publisher._create_reliability_sensor("bedroom", "Bedroom")

        assert sensor.name == "Bedroom Reliability"
        assert sensor.value_template == "{{ value_json.prediction_reliability }}"
        assert sensor.icon == "mdi:check-circle-outline"
        assert sensor.entity_category == "diagnostic"

    def test_create_system_status_sensor(self, discovery_publisher):
        """Test system status sensor configuration creation."""
        sensor = discovery_publisher._create_system_status_sensor()

        assert sensor.name == "System Status"
        assert sensor.state_topic == "occupancy/system/status"
        assert sensor.value_template == "{{ value_json.system_status }}"
        assert sensor.json_attributes_topic == "occupancy/system/status"
        assert sensor.icon == "mdi:server"

    def test_create_uptime_sensor(self, discovery_publisher):
        """Test uptime sensor configuration creation."""
        sensor = discovery_publisher._create_uptime_sensor()

        assert sensor.name == "System Uptime"
        assert sensor.value_template == "{{ value_json.uptime_seconds }}"
        assert sensor.unit_of_measurement == "s"
        assert sensor.device_class == "duration"
        assert sensor.state_class == "total"
        assert sensor.icon == "mdi:clock-check-outline"

    def test_create_predictions_count_sensor(self, discovery_publisher):
        """Test predictions count sensor configuration creation."""
        sensor = discovery_publisher._create_predictions_count_sensor()

        assert sensor.name == "Total Predictions"
        assert sensor.value_template == "{{ value_json.total_predictions_made }}"
        assert sensor.state_class == "total_increasing"
        assert sensor.icon == "mdi:counter"

    def test_create_accuracy_sensor(self, discovery_publisher):
        """Test accuracy sensor configuration creation."""
        sensor = discovery_publisher._create_accuracy_sensor()

        assert sensor.name == "System Accuracy"
        assert (
            sensor.value_template
            == "{{ value_json.average_accuracy_percent | round(1) }}"
        )
        assert sensor.unit_of_measurement == "%"
        assert sensor.icon == "mdi:target"

    def test_create_alerts_sensor(self, discovery_publisher):
        """Test alerts sensor configuration creation."""
        sensor = discovery_publisher._create_alerts_sensor()

        assert sensor.name == "Active Alerts"
        assert sensor.value_template == "{{ value_json.active_alerts }}"
        assert sensor.state_class == "measurement"
        assert sensor.icon == "mdi:alert-circle-outline"

    def test_create_database_status_sensor(self, discovery_publisher):
        """Test database status sensor configuration creation."""
        sensor = discovery_publisher._create_database_status_sensor()

        assert sensor.name == "Database Connected"
        assert (
            sensor.value_template
            == "{% if value_json.database_connected %}Connected{% else %}Disconnected{% endif %}"
        )
        assert sensor.icon == "mdi:database"

    def test_create_tracking_status_sensor(self, discovery_publisher):
        """Test tracking status sensor configuration creation."""
        sensor = discovery_publisher._create_tracking_status_sensor()

        assert sensor.name == "Tracking Active"
        assert (
            sensor.value_template
            == "{% if value_json.tracking_active %}Active{% else %}Inactive{% endif %}"
        )
        assert sensor.icon == "mdi:chart-line"


# Service Button Publishing Tests


class TestServiceButtonPublishing:
    """Test service button publishing functionality."""

    async def test_publish_service_button(self, discovery_publisher):
        """Test publishing service button."""
        service = {
            "service_name": "test_service",
            "friendly_name": "Test Service",
            "icon": "mdi:test",
            "description": "Test service button",
            "command_topic": "occupancy/commands/test",
            "command_template": "{{ value_json }}",
        }

        result = await discovery_publisher._publish_service_button(service)

        assert result.success is True
        assert service["service_name"] in discovery_publisher.published_entities

    async def test_publish_service_button_with_template(self, discovery_publisher):
        """Test publishing service button with command template."""
        service = {
            "service_name": "test_service_template",
            "friendly_name": "Test Service with Template",
            "icon": "mdi:test",
            "command_topic": "occupancy/commands/test_template",
            "command_template": "{{ value_json.action }}",
        }

        result = await discovery_publisher._publish_service_button(service)

        assert result.success is True

        # Check that the published discovery includes command template
        expected_topic = (
            "homeassistant/button/ha_ml_predictor/test_service_template/config"
        )
        assert (
            discovery_publisher.published_entities["test_service_template"]
            == expected_topic
        )


# Enhanced Discovery Features Tests


class TestEnhancedDiscoveryFeatures:
    """Test enhanced discovery features."""

    async def test_publish_sensor_discovery_with_availability(
        self, discovery_publisher
    ):
        """Test sensor discovery with availability configuration."""
        device_info = discovery_publisher.device_info
        sensor_config = SensorConfig(
            name="Test Sensor with Availability",
            unique_id="test_sensor_availability",
            state_topic="test/sensor/state",
            device=device_info,
            availability_topic="test/availability",
            availability_template="{{ 'online' if value == 'connected' else 'offline' }}",
        )

        result = await discovery_publisher._publish_sensor_discovery(sensor_config)

        assert result.success is True
        assert sensor_config.unique_id in discovery_publisher.published_entities
        assert sensor_config.unique_id in discovery_publisher.entity_metadata

    async def test_publish_sensor_discovery_enhanced_attributes(
        self, discovery_publisher
    ):
        """Test sensor discovery with enhanced attributes."""
        device_info = discovery_publisher.device_info
        sensor_config = SensorConfig(
            name="Enhanced Test Sensor",
            unique_id="enhanced_test_sensor",
            state_topic="test/sensor/state",
            device=device_info,
            json_attributes_topic="test/sensor/attributes",
            json_attributes_template="{{ value_json.attributes | tojson }}",
            unit_of_measurement="lux",
            device_class="illuminance",
            state_class="measurement",
            icon="mdi:brightness-6",
            entity_category="diagnostic",
            enabled_by_default=True,
            expire_after=300,
            force_update=False,
            suggested_display_precision=1,
        )

        result = await discovery_publisher._publish_sensor_discovery(sensor_config)

        assert result.success is True

        # Check metadata creation
        metadata = discovery_publisher.entity_metadata[sensor_config.unique_id]
        assert metadata.entity_id == sensor_config.unique_id
        assert metadata.friendly_name == sensor_config.name
        assert metadata.state == EntityState.ONLINE

    async def test_validate_published_entities(self, discovery_publisher):
        """Test entity validation after publishing."""
        # Publish discovery
        results = await discovery_publisher.publish_all_discovery()

        # All successful entities should have metadata
        for entity_name, result in results.items():
            if result.success:
                assert entity_name in discovery_publisher.entity_metadata

                metadata = discovery_publisher.entity_metadata[entity_name]
                assert metadata.entity_id == entity_name
                assert metadata.friendly_name is not None
                assert metadata.created_at is not None
                assert metadata.state == EntityState.ONLINE

    async def test_device_availability_with_enhanced_payload(self, discovery_publisher):
        """Test device availability with enhanced payload."""
        result = await discovery_publisher.publish_device_availability(online=True)

        assert result.success is True

        # The payload should contain enhanced information
        discovery_publisher.mqtt_publisher.publish_json.assert_called()
        call_args = discovery_publisher.mqtt_publisher.publish_json.call_args

        # Check the payload structure
        assert call_args[1]["data"]["status"] == "online"
        assert "timestamp" in call_args[1]["data"]
        assert "device_id" in call_args[1]["data"]
        assert "version" in call_args[1]["data"]
        assert "capabilities" in call_args[1]["data"]
        assert "entities_count" in call_args[1]["data"]


# Error Handling Tests


class TestDiscoveryErrorHandling:
    """Test discovery error handling scenarios."""

    async def test_mqtt_publish_failure(self, discovery_publisher, mock_mqtt_publisher):
        """Test handling of MQTT publish failures."""
        # Make MQTT publish fail
        failure_result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Connection lost",
        )
        mock_mqtt_publisher.publish_json.return_value = failure_result

        results = await discovery_publisher.publish_all_discovery()

        # Should handle failures gracefully
        assert isinstance(results, dict)
        for result in results.values():
            assert result.success is False
            assert result.error_message == "Connection lost"

    async def test_mqtt_publish_exception(
        self, discovery_publisher, mock_mqtt_publisher
    ):
        """Test handling of MQTT publish exceptions."""
        # Make MQTT publish raise exception
        mock_mqtt_publisher.publish_json.side_effect = Exception("MQTT error")

        results = await discovery_publisher.publish_all_discovery()

        # Should handle exceptions and return empty results
        assert results == {}
        assert discovery_publisher.stats["discovery_errors"] == 1

    async def test_remove_discovery_exception(
        self, discovery_publisher, mock_mqtt_publisher
    ):
        """Test removal with MQTT exception."""
        # Publish first
        await discovery_publisher.publish_all_discovery()
        entity_name = list(discovery_publisher.published_entities.keys())[0]

        # Make removal fail
        mock_mqtt_publisher.publish.side_effect = Exception("Removal failed")

        result = await discovery_publisher.remove_discovery(entity_name)

        assert result.success is False
        assert "Removal failed" in result.error_message

    async def test_cleanup_with_failures(
        self, discovery_publisher, mock_mqtt_publisher
    ):
        """Test cleanup with some failures."""
        # Publish first
        await discovery_publisher.publish_all_discovery()

        # Make some removals fail
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=0, publish_time=datetime.utcnow()
        )
        failure_result = MQTTPublishResult(
            success=False,
            topic="test",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Cleanup failed",
        )

        mock_mqtt_publisher.publish.side_effect = [success_result, failure_result] * 20

        results = await discovery_publisher.cleanup_entities()

        # Should handle mixed results
        successful = sum(1 for r in results.values() if r.success)
        failed = sum(1 for r in results.values() if not r.success)

        assert successful > 0
        assert failed > 0

    def test_enhanced_discovery_error_creation(self):
        """Test enhanced discovery error creation."""
        error = EnhancedDiscoveryError(
            "Enhanced discovery failed", severity=ErrorSeverity.HIGH
        )

        assert str(error) == "Enhanced discovery failed"
        assert error.error_code == "ENHANCED_DISCOVERY_ERROR"
        assert error.severity == ErrorSeverity.HIGH

    def test_enhanced_discovery_error_default_severity(self):
        """Test enhanced discovery error with default severity."""
        error = EnhancedDiscoveryError("Test error")

        assert error.severity == ErrorSeverity.MEDIUM


# Edge Cases and Complex Scenarios Tests


class TestDiscoveryEdgeCases:
    """Test discovery edge cases and complex scenarios."""

    async def test_empty_rooms_configuration(self, mqtt_config, mock_mqtt_publisher):
        """Test discovery with empty rooms configuration."""
        empty_rooms = {}
        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=empty_rooms,
        )

        results = await publisher.publish_all_discovery()

        # Should still publish system sensors and services
        assert len(results) > 0
        expected_count = 7 + 4  # System sensors + services
        assert len(results) == expected_count

    async def test_room_with_no_sensors(self, mqtt_config, mock_mqtt_publisher):
        """Test room configuration with no sensors."""
        rooms_no_sensors = {
            "empty_room": RoomConfig(
                room_id="empty_room", name="Empty Room", sensors={}
            )
        }

        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=rooms_no_sensors,
        )

        results = await publisher.publish_room_discovery(
            "empty_room", rooms_no_sensors["empty_room"]
        )

        # Should still create prediction sensors
        assert len(results) == 5  # Standard prediction sensors

    async def test_duplicate_entity_handling(self, discovery_publisher):
        """Test handling of duplicate entity creation."""
        # Publish discovery twice
        results1 = await discovery_publisher.publish_all_discovery()
        results2 = await discovery_publisher.refresh_discovery()

        # Should handle duplicates gracefully
        assert len(results1) == len(results2)

        # Entity count should remain consistent
        expected_entities = len(discovery_publisher.rooms) * 5 + 7 + 4
        assert len(discovery_publisher.published_entities) == expected_entities

    async def test_concurrent_discovery_operations(self, discovery_publisher):
        """Test concurrent discovery operations."""
        # Run multiple operations concurrently
        tasks = [
            discovery_publisher.publish_device_availability(),
            discovery_publisher.publish_system_discovery(),
            discovery_publisher.publish_service_discovery(),
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All should complete successfully
        assert len(results) == 3
        assert all(not isinstance(r, Exception) for r in results)

    async def test_large_number_of_rooms(self, mqtt_config, mock_mqtt_publisher):
        """Test discovery with large number of rooms."""
        # Create many rooms
        many_rooms = {}
        for i in range(50):
            many_rooms[f"room_{i}"] = RoomConfig(
                room_id=f"room_{i}",
                name=f"Room {i}",
                sensors={"motion": [f"binary_sensor.room_{i}_motion"]},
            )

        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=many_rooms,
        )

        results = await publisher.publish_all_discovery()

        # Should handle large number of entities
        expected_count = len(many_rooms) * 5 + 7 + 4  # Rooms + system + services
        assert len(results) == expected_count
        assert len(publisher.published_entities) == expected_count

    async def test_discovery_with_special_characters(
        self, mqtt_config, mock_mqtt_publisher
    ):
        """Test discovery with special characters in names."""
        special_rooms = {
            "room_with_spaces": RoomConfig(
                room_id="room with spaces",
                name="Room With Spaces & Special Chars!",
                sensors={"motion": ["binary_sensor.room_spaces_motion"]},
            ),
            "room-with-dashes": RoomConfig(
                room_id="room-with-dashes",
                name="Room-With-Dashes",
                sensors={"motion": ["binary_sensor.room_dashes_motion"]},
            ),
        }

        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=special_rooms,
        )

        results = await publisher.publish_all_discovery()

        # Should handle special characters in IDs and names
        assert len(results) > 0
        for result in results.values():
            assert result.success is True

    async def test_discovery_state_persistence(self, discovery_publisher):
        """Test discovery state persistence across operations."""
        # Initial state
        assert discovery_publisher.discovery_published is False
        assert len(discovery_publisher.published_entities) == 0

        # Publish discovery
        await discovery_publisher.publish_all_discovery()
        initial_count = len(discovery_publisher.published_entities)
        assert discovery_publisher.discovery_published is True
        assert initial_count > 0

        # Remove some entities
        entities_to_remove = list(discovery_publisher.published_entities.keys())[:3]
        for entity_id in entities_to_remove:
            await discovery_publisher.remove_discovery(entity_id)

        # State should be updated
        assert len(discovery_publisher.published_entities) == initial_count - 3
        assert discovery_publisher.stats["entities_removed"] == 3

        # Refresh discovery
        await discovery_publisher.refresh_discovery()

        # Should restore all entities
        assert len(discovery_publisher.published_entities) == initial_count


# Performance and Load Tests


class TestDiscoveryPerformance:
    """Test discovery performance characteristics."""

    async def test_bulk_discovery_publishing_performance(self, discovery_publisher):
        """Test performance of bulk discovery publishing."""
        import time

        start_time = time.time()
        results = await discovery_publisher.publish_all_discovery()
        end_time = time.time()

        # Should complete reasonably quickly
        duration = end_time - start_time
        assert duration < 5.0  # Should complete within 5 seconds
        assert len(results) > 0

    async def test_concurrent_entity_operations(self, discovery_publisher):
        """Test concurrent entity state operations."""
        # Publish discovery first
        await discovery_publisher.publish_all_discovery()
        entity_ids = list(discovery_publisher.entity_metadata.keys())

        # Perform concurrent state updates
        async def update_entity_state(entity_id, state):
            return await discovery_publisher.update_entity_state(
                entity_id, state, {"test": f"value_{entity_id}"}
            )

        tasks = [
            update_entity_state(entity_id, EntityState.WARNING)
            for entity_id in entity_ids[:10]  # Update first 10 entities
        ]

        results = await asyncio.gather(*tasks)

        # All updates should succeed
        assert all(results)

        # Check that all entities were updated
        for entity_id in entity_ids[:10]:
            metadata = discovery_publisher.entity_metadata[entity_id]
            assert metadata.state == EntityState.WARNING
            assert metadata.attributes["test"] == f"value_{entity_id}"

    async def test_rapid_availability_updates(self, discovery_publisher):
        """Test rapid device availability updates."""
        # Perform rapid availability updates
        results = []
        for i in range(20):
            online = i % 2 == 0  # Alternate between online and offline
            result = await discovery_publisher.publish_device_availability(online)
            results.append(result)

        # All updates should succeed
        assert all(r.success for r in results)
        assert discovery_publisher.stats["availability_updates"] == 20

        # Final state should match last update
        assert (
            discovery_publisher.device_available is False
        )  # Last was offline (19 % 2 != 0)


# Integration with Mock Services Tests


class TestDiscoveryMockIntegration:
    """Test discovery integration with mock services."""

    async def test_availability_callback_integration(
        self, mqtt_config, room_configs, mock_mqtt_publisher
    ):
        """Test integration with availability callback."""
        availability_callback = AsyncMock()
        availability_callback.return_value = True

        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
            availability_check_callback=availability_callback,
        )

        # Availability callback should be stored
        assert publisher.availability_check_callback == availability_callback

        # Test that callback would be used (in practice by other components)
        result = await availability_callback()
        assert result is True

    async def test_state_change_callback_integration(
        self, mqtt_config, room_configs, mock_mqtt_publisher
    ):
        """Test integration with state change callback."""
        state_change_callback = AsyncMock()

        publisher = DiscoveryPublisher(
            mqtt_publisher=mock_mqtt_publisher,
            config=mqtt_config,
            rooms=room_configs,
            state_change_callback=state_change_callback,
        )

        # Publish and update entity state
        await publisher.publish_all_discovery()
        entity_id = list(publisher.entity_metadata.keys())[0]

        await publisher.update_entity_state(
            entity_id, EntityState.ERROR, {"error": "test"}
        )

        # Callback should have been called
        state_change_callback.assert_called_once_with(
            entity_id, EntityState.ERROR, {"error": "test"}
        )

    async def test_mqtt_publisher_integration(
        self, discovery_publisher, mock_mqtt_publisher
    ):
        """Test integration with MQTT publisher."""
        # Publish discovery
        await discovery_publisher.publish_all_discovery()

        # Check MQTT publisher was called extensively
        assert mock_mqtt_publisher.publish_json.call_count > 0

        # Check that all calls were with proper parameters
        for call in mock_mqtt_publisher.publish_json.call_args_list:
            args, kwargs = call
            assert "topic" in kwargs
            assert "data" in kwargs
            assert kwargs["qos"] == 1
            assert kwargs["retain"] is True

    async def test_service_configuration_storage(self, discovery_publisher):
        """Test service configuration storage."""
        # Publish service discovery
        await discovery_publisher.publish_service_discovery()

        # Check services were stored
        assert len(discovery_publisher.available_services) == 4

        service_names = list(discovery_publisher.available_services.keys())
        expected_services = [
            "manual_retrain",
            "refresh_discovery",
            "reset_statistics",
            "force_prediction",
        ]

        for service_name in expected_services:
            assert service_name in service_names

            service_config = discovery_publisher.available_services[service_name]
            assert isinstance(service_config, ServiceConfig)
            assert service_config.service_name == service_name
            assert service_config.service_topic.startswith("occupancy/commands/")


# Enum and Constant Tests


class TestDiscoveryEnumsAndConstants:
    """Test enum values and constants."""

    def test_entity_state_enum(self):
        """Test EntityState enum values."""
        assert EntityState.UNKNOWN.value == "unknown"
        assert EntityState.UNAVAILABLE.value == "unavailable"
        assert EntityState.ONLINE.value == "online"
        assert EntityState.OFFLINE.value == "offline"
        assert EntityState.OK.value == "ok"
        assert EntityState.ERROR.value == "error"
        assert EntityState.WARNING.value == "warning"

    def test_entity_category_enum(self):
        """Test EntityCategory enum values."""
        assert EntityCategory.CONFIG.value == "config"
        assert EntityCategory.DIAGNOSTIC.value == "diagnostic"
        assert EntityCategory.SYSTEM.value == "system"

    def test_device_class_enum(self):
        """Test DeviceClass enum values."""
        assert DeviceClass.TIMESTAMP.value == "timestamp"
        assert DeviceClass.DURATION.value == "duration"
        assert DeviceClass.DATA_SIZE.value == "data_size"
        assert DeviceClass.ENUM.value == "enum"

    def test_sensor_names_and_icons(self, discovery_publisher):
        """Test that sensor configurations have appropriate names and icons."""
        # Test room sensors
        prediction_sensor = discovery_publisher._create_prediction_sensor(
            "test_room", "Test Room"
        )
        assert "Test Room" in prediction_sensor.name
        assert prediction_sensor.icon == "mdi:home-account"

        confidence_sensor = discovery_publisher._create_confidence_sensor(
            "test_room", "Test Room"
        )
        assert confidence_sensor.icon == "mdi:percent"
        assert confidence_sensor.unit_of_measurement == "%"

        # Test system sensors
        uptime_sensor = discovery_publisher._create_uptime_sensor()
        assert uptime_sensor.icon == "mdi:clock-check-outline"
        assert uptime_sensor.device_class == "duration"

        accuracy_sensor = discovery_publisher._create_accuracy_sensor()
        assert accuracy_sensor.icon == "mdi:target"
        assert accuracy_sensor.unit_of_measurement == "%"
