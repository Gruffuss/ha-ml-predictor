"""
Comprehensive tests for MQTT Integration Manager.

This test module provides extensive coverage of the MQTT integration manager
functionality including initialization, background tasks, prediction publishing,
system status monitoring, and service command handling.

Coverage Areas:
- Integration manager initialization and configuration
- MQTT component initialization and coordination
- Background task management and lifecycle
- Prediction publishing automation
- System status monitoring and publishing
- Discovery management and refresh
- Service command handling
- Error handling and recovery scenarios
- Statistics and monitoring
- Callback management and notifications
- Device availability management
- Integration with tracking systems
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch
import uuid

import pytest

from src.core.config import MQTTConfig, RoomConfig, get_config
from src.core.exceptions import ErrorSeverity, OccupancyPredictionError
from src.integration.mqtt_integration_manager import (
    MQTTIntegrationError,
    MQTTIntegrationManager,
    MQTTIntegrationStats,
)
from src.integration.mqtt_publisher import MQTTPublishResult
from src.models.base.predictor import PredictionResult

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
        device_identifier="ha_ml_predictor_test",
        device_name="HA ML Predictor Test",
        device_manufacturer="Home Assistant Community",
        device_model="ML Occupancy Predictor Test v1.0",
        device_sw_version="1.0.0-test",
        publishing_enabled=True,
        status_update_interval_seconds=10,
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
            },
        ),
        "bedroom": RoomConfig(
            room_id="bedroom",
            name="Bedroom",
            sensors={
                "motion": ["binary_sensor.bedroom_motion"],
                "occupancy": ["binary_sensor.bedroom_occupancy"],
            },
        ),
        "kitchen": RoomConfig(
            room_id="kitchen",
            name="Kitchen",
            sensors={
                "motion": ["binary_sensor.kitchen_motion"],
                "temperature": ["sensor.kitchen_temperature"],
            },
        ),
    }


@pytest.fixture
def mock_mqtt_publisher():
    """Create mock MQTT publisher."""
    publisher = AsyncMock()

    # Mock successful results
    success_result = MQTTPublishResult(
        success=True,
        topic="test/topic",
        payload_size=100,
        publish_time=datetime.utcnow(),
        message_id=1,
    )

    publisher.initialize.return_value = None
    publisher.stop_publisher.return_value = None
    publisher.publish.return_value = success_result
    publisher.publish_json.return_value = success_result

    # Mock connection status
    publisher.connection_status = Mock()
    publisher.connection_status.connected = True

    # Mock publisher stats
    publisher.get_publisher_stats.return_value = {
        "connected": True,
        "messages_published": 50,
        "messages_failed": 2,
        "last_publish": datetime.now().isoformat(),
    }

    return publisher


@pytest.fixture
def mock_prediction_publisher():
    """Create mock prediction publisher."""
    publisher = AsyncMock()

    success_result = MQTTPublishResult(
        success=True,
        topic="occupancy/living_room/prediction",
        payload_size=150,
        publish_time=datetime.utcnow(),
    )

    publisher.publish_prediction.return_value = success_result
    publisher.publish_system_status.return_value = success_result

    publisher.get_publisher_stats.return_value = {
        "predictions_published": 25,
        "status_updates_published": 10,
        "last_prediction": datetime.now().isoformat(),
    }

    return publisher


@pytest.fixture
def mock_discovery_publisher():
    """Create mock discovery publisher."""
    publisher = AsyncMock()

    success_result = MQTTPublishResult(
        success=True,
        topic="homeassistant/sensor/test/config",
        payload_size=200,
        publish_time=datetime.utcnow(),
    )

    publisher.publish_all_discovery.return_value = {
        "living_room_prediction": success_result,
        "bedroom_prediction": success_result,
        "system_status": success_result,
    }

    publisher.refresh_discovery.return_value = {
        "living_room_prediction": success_result,
        "bedroom_prediction": success_result,
    }

    publisher.cleanup_entities.return_value = {
        "entity1": success_result,
        "entity2": success_result,
    }

    publisher.publish_device_availability.return_value = success_result

    publisher.get_discovery_stats.return_value = {
        "discovery_enabled": True,
        "discovery_published": True,
        "published_entities_count": 15,
        "entity_metadata_count": 15,
        "device_available": True,
        "available_services_count": 4,
        "statistics": {"discovery_errors": 0, "entities_created": 15},
    }

    return publisher


@pytest.fixture
def mock_notification_callbacks():
    """Create mock notification callbacks."""
    callback1 = AsyncMock()
    callback2 = Mock()  # Sync callback
    return [callback1, callback2]


@pytest.fixture
def prediction_result():
    """Create sample prediction result."""
    return PredictionResult(
        room_id="living_room",
        predicted_time=datetime.now(timezone.utc) + timedelta(minutes=30),
        transition_type="occupied_to_vacant",
        confidence=0.85,
        model_version="v1.0.0",
        features_used=["temporal", "sequential"],
        prediction_metadata={
            "algorithm": "ensemble",
            "feature_count": 25,
            "training_samples": 1000,
        },
    )


# Integration Stats Tests


class TestMQTTIntegrationStats:
    """Test MQTT integration statistics."""

    def test_mqtt_integration_stats_creation(self):
        """Test integration stats initialization."""
        stats = MQTTIntegrationStats()

        assert stats.initialized is False
        assert stats.mqtt_connected is False
        assert stats.discovery_published is False
        assert stats.predictions_published == 0
        assert stats.status_updates_published == 0
        assert stats.last_prediction_published is None
        assert stats.last_status_published is None
        assert stats.total_errors == 0
        assert stats.last_error is None

    def test_mqtt_integration_stats_with_data(self):
        """Test integration stats with actual data."""
        now = datetime.now(timezone.utc)
        stats = MQTTIntegrationStats(
            initialized=True,
            mqtt_connected=True,
            discovery_published=True,
            predictions_published=100,
            status_updates_published=25,
            last_prediction_published=now,
            last_status_published=now,
            total_errors=5,
            last_error="Connection timeout",
        )

        assert stats.initialized is True
        assert stats.mqtt_connected is True
        assert stats.discovery_published is True
        assert stats.predictions_published == 100
        assert stats.status_updates_published == 25
        assert stats.last_prediction_published == now
        assert stats.last_status_published == now
        assert stats.total_errors == 5
        assert stats.last_error == "Connection timeout"


# Integration Manager Initialization Tests


class TestMQTTIntegrationManagerInit:
    """Test MQTT integration manager initialization."""

    def test_integration_manager_initialization_with_configs(
        self, mqtt_config, room_configs
    ):
        """Test integration manager initialization with provided configs."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        assert manager.mqtt_config == mqtt_config
        assert manager.rooms == room_configs
        assert len(manager.notification_callbacks) == 0
        assert manager.mqtt_publisher is None
        assert manager.prediction_publisher is None
        assert manager.discovery_publisher is None
        assert manager.stats.initialized is False
        assert manager._integration_active is False

    @patch("src.integration.mqtt_integration_manager.get_config")
    def test_integration_manager_initialization_load_config(
        self, mock_get_config, mqtt_config, room_configs
    ):
        """Test integration manager initialization loading config."""
        mock_system_config = Mock()
        mock_system_config.mqtt = mqtt_config
        mock_system_config.rooms = room_configs
        mock_get_config.return_value = mock_system_config

        manager = MQTTIntegrationManager()

        assert manager.mqtt_config == mqtt_config
        assert manager.rooms == room_configs
        mock_get_config.assert_called_once()

    def test_integration_manager_initialization_with_callbacks(
        self, mqtt_config, room_configs, mock_notification_callbacks
    ):
        """Test integration manager initialization with callbacks."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=mock_notification_callbacks,
        )

        assert len(manager.notification_callbacks) == 2
        assert manager.notification_callbacks == mock_notification_callbacks

    def test_integration_manager_system_start_time(self, mqtt_config, room_configs):
        """Test that system start time is recorded."""
        before_init = datetime.now(timezone.utc)
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        after_init = datetime.now(timezone.utc)

        assert before_init <= manager.system_start_time <= after_init


# Integration Manager Core Functionality Tests


class TestMQTTIntegrationManagerCore:
    """Test core integration manager functionality."""

    async def test_integration_manager_initialize_success(
        self, mqtt_config, room_configs
    ):
        """Test successful integration manager initialization."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        with patch.object(manager, "_on_mqtt_connect"), patch.object(
            manager, "_on_mqtt_disconnect"
        ), patch.object(manager, "start_integration") as mock_start:

            # Mock the components
            mock_mqtt_pub = AsyncMock()
            mock_pred_pub = AsyncMock()
            mock_disc_pub = AsyncMock()
            mock_disc_pub.publish_all_discovery.return_value = {
                "entity1": Mock(success=True)
            }

            with patch(
                "src.integration.mqtt_integration_manager.MQTTPublisher",
                return_value=mock_mqtt_pub,
            ), patch(
                "src.integration.mqtt_integration_manager.PredictionPublisher",
                return_value=mock_pred_pub,
            ), patch(
                "src.integration.mqtt_integration_manager.DiscoveryPublisher",
                return_value=mock_disc_pub,
            ):

                await manager.initialize()

                assert manager.mqtt_publisher == mock_mqtt_pub
                assert manager.prediction_publisher == mock_pred_pub
                assert manager.discovery_publisher == mock_disc_pub
                assert manager.stats.initialized is True
                mock_start.assert_called_once()

    async def test_integration_manager_initialize_disabled(
        self, mqtt_config, room_configs
    ):
        """Test initialization when MQTT publishing is disabled."""
        mqtt_config.publishing_enabled = False
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        await manager.initialize()

        # Should return early without initializing components
        assert manager.mqtt_publisher is None
        assert manager.prediction_publisher is None
        assert manager.discovery_publisher is None

    async def test_integration_manager_initialize_discovery_disabled(
        self, mqtt_config, room_configs
    ):
        """Test initialization when discovery is disabled."""
        mqtt_config.discovery_enabled = False
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        with patch.object(manager, "start_integration"):
            mock_mqtt_pub = AsyncMock()
            mock_pred_pub = AsyncMock()

            with patch(
                "src.integration.mqtt_integration_manager.MQTTPublisher",
                return_value=mock_mqtt_pub,
            ), patch(
                "src.integration.mqtt_integration_manager.PredictionPublisher",
                return_value=mock_pred_pub,
            ):

                await manager.initialize()

                assert manager.mqtt_publisher == mock_mqtt_pub
                assert manager.prediction_publisher == mock_pred_pub
                assert manager.discovery_publisher is None

    async def test_integration_manager_initialize_failure(
        self, mqtt_config, room_configs
    ):
        """Test initialization failure handling."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        with patch(
            "src.integration.mqtt_integration_manager.MQTTPublisher",
            side_effect=Exception("Init failed"),
        ):

            with pytest.raises(
                MQTTIntegrationError, match="Failed to initialize MQTT integration"
            ):
                await manager.initialize()

            assert manager.stats.total_errors == 1
            assert "Init failed" in manager.stats.last_error

    async def test_start_integration_success(self, mqtt_config, room_configs):
        """Test successful integration start."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        with patch.object(manager, "_system_status_publishing_loop") as mock_loop:
            mock_task = AsyncMock()

            with patch("asyncio.create_task", return_value=mock_task):
                await manager.start_integration()

                assert manager._integration_active is True
                assert len(manager._background_tasks) == 1

    async def test_start_integration_disabled(self, mqtt_config, room_configs):
        """Test starting integration when publishing is disabled."""
        mqtt_config.publishing_enabled = False
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        await manager.start_integration()

        assert manager._integration_active is False
        assert len(manager._background_tasks) == 0

    async def test_start_integration_already_active(self, mqtt_config, room_configs):
        """Test starting integration when already active."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        await manager.start_integration()

        # Should return early without changes
        assert manager._integration_active is True

    async def test_stop_integration_success(self, mqtt_config, room_configs):
        """Test successful integration stop."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Setup active integration
        manager._integration_active = True
        mock_mqtt_publisher = AsyncMock()
        manager.mqtt_publisher = mock_mqtt_publisher

        mock_task1 = AsyncMock()
        mock_task2 = AsyncMock()
        manager._background_tasks = [mock_task1, mock_task2]

        with patch("asyncio.gather", return_value=None):
            await manager.stop_integration()

            assert manager._integration_active is False
            assert len(manager._background_tasks) == 0
            assert manager._shutdown_event.is_set()
            mock_mqtt_publisher.stop_publisher.assert_called_once()

    async def test_stop_integration_not_active(self, mqtt_config, room_configs):
        """Test stopping integration when not active."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        await manager.stop_integration()

        # Should return early
        assert manager._integration_active is False


# Prediction Publishing Tests


class TestPredictionPublishing:
    """Test prediction publishing functionality."""

    async def test_publish_prediction_success(
        self, mqtt_config, room_configs, prediction_result
    ):
        """Test successful prediction publishing."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True,
            topic="test/topic",
            payload_size=100,
            publish_time=datetime.utcnow(),
        )
        mock_prediction_publisher.publish_prediction.return_value = success_result
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_prediction(
            prediction_result=prediction_result,
            room_id="living_room",
            current_state="vacant",
        )

        assert result is True
        assert manager.stats.predictions_published == 1
        assert manager.stats.last_prediction_published is not None
        mock_prediction_publisher.publish_prediction.assert_called_once_with(
            prediction_result=prediction_result,
            room_id="living_room",
            current_state="vacant",
        )

    async def test_publish_prediction_failure(
        self, mqtt_config, room_configs, prediction_result
    ):
        """Test prediction publishing failure."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        failure_result = MQTTPublishResult(
            success=False,
            topic="test/topic",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Publish failed",
        )
        mock_prediction_publisher.publish_prediction.return_value = failure_result
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_prediction(
            prediction_result=prediction_result, room_id="living_room"
        )

        assert result is False
        assert manager.stats.predictions_published == 0  # No increment on failure
        assert manager.stats.total_errors == 1
        assert manager.stats.last_error == "Publish failed"

    async def test_publish_prediction_not_active(
        self, mqtt_config, room_configs, prediction_result
    ):
        """Test prediction publishing when integration not active."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = False

        result = await manager.publish_prediction(
            prediction_result=prediction_result, room_id="living_room"
        )

        assert result is False
        assert manager.stats.predictions_published == 0

    async def test_publish_prediction_no_publisher(
        self, mqtt_config, room_configs, prediction_result
    ):
        """Test prediction publishing with no publisher."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True
        manager.prediction_publisher = None

        result = await manager.publish_prediction(
            prediction_result=prediction_result, room_id="living_room"
        )

        assert result is False

    async def test_publish_prediction_exception(
        self, mqtt_config, room_configs, prediction_result
    ):
        """Test prediction publishing with exception."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        mock_prediction_publisher.publish_prediction.side_effect = Exception(
            "Publisher error"
        )
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_prediction(
            prediction_result=prediction_result, room_id="living_room"
        )

        assert result is False
        assert manager.stats.total_errors == 1
        assert "Publisher error" in manager.stats.last_error


# System Status Publishing Tests


class TestSystemStatusPublishing:
    """Test system status publishing functionality."""

    async def test_publish_system_status_success(self, mqtt_config, room_configs):
        """Test successful system status publishing."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True,
            topic="occupancy/system/status",
            payload_size=150,
            publish_time=datetime.utcnow(),
        )
        mock_prediction_publisher.publish_system_status.return_value = success_result
        manager.prediction_publisher = mock_prediction_publisher

        tracking_stats = {"predictions_made": 100, "accuracy": 0.85}
        model_stats = {"models_trained": 3, "last_training": datetime.now().isoformat()}

        result = await manager.publish_system_status(
            tracking_stats=tracking_stats,
            model_stats=model_stats,
            database_connected=True,
            active_alerts=2,
            last_error="Test error",
        )

        assert result is True
        assert manager.stats.status_updates_published == 1
        assert manager.stats.last_status_published is not None
        mock_prediction_publisher.publish_system_status.assert_called_once_with(
            tracking_stats=tracking_stats,
            model_stats=model_stats,
            database_connected=True,
            active_alerts=2,
            last_error="Test error",
        )

    async def test_publish_system_status_failure(self, mqtt_config, room_configs):
        """Test system status publishing failure."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        failure_result = MQTTPublishResult(
            success=False,
            topic="occupancy/system/status",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Status publish failed",
        )
        mock_prediction_publisher.publish_system_status.return_value = failure_result
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_system_status()

        assert result is False
        assert manager.stats.status_updates_published == 0
        assert manager.stats.total_errors == 1
        assert manager.stats.last_error == "Status publish failed"

    async def test_publish_system_status_not_active(self, mqtt_config, room_configs):
        """Test system status publishing when not active."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = False

        result = await manager.publish_system_status()

        assert result is False
        assert manager.stats.status_updates_published == 0

    async def test_publish_system_status_exception(self, mqtt_config, room_configs):
        """Test system status publishing with exception."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        mock_prediction_publisher.publish_system_status.side_effect = Exception(
            "Status error"
        )
        manager.prediction_publisher = mock_prediction_publisher

        result = await manager.publish_system_status()

        assert result is False
        assert manager.stats.total_errors == 1
        assert "Status error" in manager.stats.last_error


# Discovery Management Tests


class TestDiscoveryManagement:
    """Test discovery management functionality."""

    async def test_refresh_discovery_success(self, mqtt_config, room_configs):
        """Test successful discovery refresh."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_discovery_publisher.refresh_discovery.return_value = {
            "entity1": success_result,
            "entity2": success_result,
            "entity3": success_result,
        }
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.refresh_discovery()

        assert result is True
        assert manager.stats.discovery_published is True
        mock_discovery_publisher.refresh_discovery.assert_called_once()

    async def test_refresh_discovery_partial_failure(self, mqtt_config, room_configs):
        """Test discovery refresh with partial failures."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        failure_result = MQTTPublishResult(
            success=False,
            topic="test",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Failed",
        )

        mock_discovery_publisher.refresh_discovery.return_value = {
            "entity1": success_result,
            "entity2": failure_result,
            "entity3": success_result,
        }
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.refresh_discovery()

        assert result is False  # Not all succeeded
        assert manager.stats.discovery_published is False

    async def test_refresh_discovery_no_publisher(self, mqtt_config, room_configs):
        """Test discovery refresh without publisher."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager.discovery_publisher = None

        result = await manager.refresh_discovery()

        assert result is False

    async def test_refresh_discovery_exception(self, mqtt_config, room_configs):
        """Test discovery refresh with exception."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.refresh_discovery.side_effect = Exception(
            "Discovery error"
        )
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.refresh_discovery()

        assert result is False
        assert manager.stats.total_errors == 1
        assert "Discovery error" in manager.stats.last_error

    async def test_cleanup_discovery_success(self, mqtt_config, room_configs):
        """Test successful discovery cleanup."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_discovery_publisher.cleanup_entities.return_value = {
            "entity1": success_result,
            "entity2": success_result,
        }
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.cleanup_discovery(["entity1", "entity2"])

        assert result is True
        mock_discovery_publisher.cleanup_entities.assert_called_once_with(
            ["entity1", "entity2"]
        )

    async def test_cleanup_discovery_no_publisher(self, mqtt_config, room_configs):
        """Test discovery cleanup without publisher."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager.discovery_publisher = None

        result = await manager.cleanup_discovery()

        assert result is False


# Device Availability Management Tests


class TestDeviceAvailabilityManagement:
    """Test device availability management."""

    async def test_update_device_availability_online(self, mqtt_config, room_configs):
        """Test updating device availability to online."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_discovery_publisher.publish_device_availability.return_value = (
            success_result
        )
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.update_device_availability(online=True)

        assert result is True
        mock_discovery_publisher.publish_device_availability.assert_called_once_with(
            online=True
        )

    async def test_update_device_availability_offline(self, mqtt_config, room_configs):
        """Test updating device availability to offline."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_discovery_publisher.publish_device_availability.return_value = (
            success_result
        )
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.update_device_availability(online=False)

        assert result is True
        mock_discovery_publisher.publish_device_availability.assert_called_once_with(
            online=False
        )

    async def test_update_device_availability_failure(self, mqtt_config, room_configs):
        """Test device availability update failure."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        failure_result = MQTTPublishResult(
            success=False,
            topic="test",
            payload_size=0,
            publish_time=datetime.utcnow(),
            error_message="Update failed",
        )
        mock_discovery_publisher.publish_device_availability.return_value = (
            failure_result
        )
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.update_device_availability(online=True)

        assert result is False

    async def test_update_device_availability_no_publisher(
        self, mqtt_config, room_configs
    ):
        """Test device availability update without publisher."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager.discovery_publisher = None

        result = await manager.update_device_availability(online=True)

        assert result is False


# Service Command Handling Tests


class TestServiceCommandHandling:
    """Test service command handling functionality."""

    async def test_handle_manual_retrain_command(self, mqtt_config, room_configs):
        """Test handling manual retrain service command."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        command_data = {"room_id": "living_room", "strategy": "incremental"}

        result = await manager.handle_service_command("manual_retrain", command_data)

        assert result is True

    async def test_handle_refresh_discovery_command(self, mqtt_config, room_configs):
        """Test handling refresh discovery service command."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        mock_discovery_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_discovery_publisher.refresh_discovery.return_value = {
            "entity1": success_result
        }
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.handle_service_command("refresh_discovery", {})

        assert result is True
        mock_discovery_publisher.refresh_discovery.assert_called_once()

    async def test_handle_reset_statistics_command(self, mqtt_config, room_configs):
        """Test handling reset statistics service command."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Set some initial stats
        manager.stats.predictions_published = 100
        manager.stats.total_errors = 5

        result = await manager.handle_service_command("reset_statistics", {})

        assert result is True
        assert manager.stats.predictions_published == 0
        assert manager.stats.total_errors == 0
        assert manager.stats.initialized is False  # Stats should be reset

    async def test_handle_force_prediction_command(self, mqtt_config, room_configs):
        """Test handling force prediction service command."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        command_data = {"room_id": "bedroom"}

        result = await manager.handle_service_command("force_prediction", command_data)

        assert result is True

    async def test_handle_force_prediction_command_no_room(
        self, mqtt_config, room_configs
    ):
        """Test handling force prediction without room_id."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        command_data = {}

        result = await manager.handle_service_command("force_prediction", command_data)

        assert result is False

    async def test_handle_unknown_service_command(self, mqtt_config, room_configs):
        """Test handling unknown service command."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        result = await manager.handle_service_command("unknown_command", {})

        assert result is False

    async def test_handle_service_command_exception(self, mqtt_config, room_configs):
        """Test service command handling with exception."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Mock discovery publisher to raise exception
        mock_discovery_publisher = AsyncMock()
        mock_discovery_publisher.refresh_discovery.side_effect = Exception(
            "Service error"
        )
        manager.discovery_publisher = mock_discovery_publisher

        result = await manager.handle_service_command("refresh_discovery", {})

        assert result is False


# Statistics and Monitoring Tests


class TestIntegrationStatisticsMonitoring:
    """Test integration statistics and monitoring."""

    def test_get_integration_stats_basic(self, mqtt_config, room_configs):
        """Test getting basic integration statistics."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        stats = manager.get_integration_stats()

        assert isinstance(stats, dict)
        assert stats["initialized"] is False
        assert stats["integration_active"] is False
        assert stats["mqtt_connected"] is False
        assert stats["discovery_published"] is False
        assert stats["predictions_published"] == 0
        assert stats["status_updates_published"] == 0
        assert stats["total_errors"] == 0
        assert stats["rooms_configured"] == len(room_configs)
        assert "system_uptime_seconds" in stats
        assert "background_tasks" in stats

    def test_get_integration_stats_with_publishers(self, mqtt_config, room_configs):
        """Test getting statistics with publisher information."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Mock publishers with stats
        mock_mqtt_publisher = Mock()
        mock_mqtt_publisher.connection_status.connected = True
        mock_mqtt_publisher.get_publisher_stats.return_value = {"mqtt_stat": "value"}
        manager.mqtt_publisher = mock_mqtt_publisher

        mock_prediction_publisher = Mock()
        mock_prediction_publisher.get_publisher_stats.return_value = {
            "pred_stat": "value"
        }
        manager.prediction_publisher = mock_prediction_publisher

        mock_discovery_publisher = Mock()
        mock_discovery_publisher.get_discovery_stats.return_value = {
            "published_entities_count": 10,
            "device_available": True,
            "statistics": {"discovery_errors": 0},
        }
        manager.discovery_publisher = mock_discovery_publisher

        stats = manager.get_integration_stats()

        assert stats["mqtt_connected"] is True
        assert "mqtt_publisher" in stats
        assert "prediction_publisher" in stats
        assert "discovery_publisher" in stats
        assert "discovery_insights" in stats
        assert "system_health" in stats

        # Check system health
        system_health = stats["system_health"]
        assert "overall_status" in system_health
        assert "component_status" in system_health
        assert "error_rate" in system_health

    def test_get_integration_stats_with_errors(self, mqtt_config, room_configs):
        """Test statistics with error conditions."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Set error conditions
        manager.stats.total_errors = 15
        manager.stats.predictions_published = 100
        manager.stats.last_error = "Recent error"

        stats = manager.get_integration_stats()

        assert stats["total_errors"] == 15
        assert stats["last_error"] == "Recent error"

        # System health should reflect high error rate
        system_health = stats["system_health"]
        assert system_health["error_rate"] == 0.15  # 15 errors / 100 predictions
        assert system_health["overall_status"] == "degraded"  # Due to high error rate

    def test_is_connected_true(self, mqtt_config, room_configs):
        """Test is_connected when fully connected."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        manager._integration_active = True
        mock_mqtt_publisher = Mock()
        mock_mqtt_publisher.connection_status.connected = True
        manager.mqtt_publisher = mock_mqtt_publisher

        assert manager.is_connected() is True

    def test_is_connected_false_conditions(self, mqtt_config, room_configs):
        """Test is_connected with various false conditions."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Not active
        manager._integration_active = False
        assert manager.is_connected() is False

        # No publisher
        manager._integration_active = True
        manager.mqtt_publisher = None
        assert manager.is_connected() is False

        # Not connected
        manager._integration_active = True
        mock_mqtt_publisher = Mock()
        mock_mqtt_publisher.connection_status.connected = False
        manager.mqtt_publisher = mock_mqtt_publisher
        assert manager.is_connected() is False

    def test_update_system_stats(self, mqtt_config, room_configs):
        """Test updating system statistics."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        test_stats = {"predictions_made": 100, "accuracy": 0.85, "models_active": 3}

        manager.update_system_stats(test_stats)

        assert manager._last_system_stats == test_stats


# Notification Callback Tests


class TestNotificationCallbacks:
    """Test notification callback functionality."""

    def test_add_notification_callback(self, mqtt_config, room_configs):
        """Test adding notification callback."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        callback = Mock()
        manager.add_notification_callback(callback)

        assert len(manager.notification_callbacks) == 1
        assert callback in manager.notification_callbacks

    def test_add_duplicate_notification_callback(self, mqtt_config, room_configs):
        """Test adding duplicate notification callback."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        callback = Mock()
        manager.add_notification_callback(callback)
        manager.add_notification_callback(callback)  # Add again

        assert len(manager.notification_callbacks) == 1  # Should not duplicate

    def test_remove_notification_callback(self, mqtt_config, room_configs):
        """Test removing notification callback."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        callback = Mock()
        manager.add_notification_callback(callback)
        assert len(manager.notification_callbacks) == 1

        manager.remove_notification_callback(callback)
        assert len(manager.notification_callbacks) == 0

    def test_remove_nonexistent_notification_callback(self, mqtt_config, room_configs):
        """Test removing nonexistent notification callback."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        callback = Mock()

        # Should not raise error
        manager.remove_notification_callback(callback)
        assert len(manager.notification_callbacks) == 0

    async def test_mqtt_connect_callback_notification(self, mqtt_config, room_configs):
        """Test MQTT connect callback with notifications."""
        async_callback = AsyncMock()
        sync_callback = Mock()

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=[async_callback, sync_callback],
        )

        await manager._on_mqtt_connect(None, None, None, None)

        assert manager.stats.mqtt_connected is True
        async_callback.assert_called_once_with("mqtt_connected")
        sync_callback.assert_called_once_with("mqtt_connected")

    async def test_mqtt_disconnect_callback_notification(
        self, mqtt_config, room_configs
    ):
        """Test MQTT disconnect callback with notifications."""
        async_callback = AsyncMock()
        sync_callback = Mock()

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=[async_callback, sync_callback],
        )

        await manager._on_mqtt_disconnect(None, None, None, None)

        assert manager.stats.mqtt_connected is False
        async_callback.assert_called_once_with("mqtt_disconnected")
        sync_callback.assert_called_once_with("mqtt_disconnected")

    async def test_callback_notification_with_exception(
        self, mqtt_config, room_configs
    ):
        """Test callback notification handling exceptions."""
        failing_callback = AsyncMock(side_effect=Exception("Callback failed"))
        working_callback = Mock()

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=[failing_callback, working_callback],
        )

        # Should not raise exception despite callback failure
        await manager._on_mqtt_connect(None, None, None, None)

        assert manager.stats.mqtt_connected is True
        failing_callback.assert_called_once()
        working_callback.assert_called_once()


# Background Task Tests


class TestBackgroundTasks:
    """Test background task functionality."""

    @patch("asyncio.wait_for")
    @patch("asyncio.sleep")
    async def test_system_status_publishing_loop(
        self, mock_sleep, mock_wait_for, mqtt_config, room_configs
    ):
        """Test system status publishing background loop."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Mock successful publishing
        manager.publish_system_status = AsyncMock(return_value=True)

        # Make wait_for raise TimeoutError to simulate interval
        mock_wait_for.side_effect = [asyncio.TimeoutError(), asyncio.CancelledError()]

        with pytest.raises(asyncio.CancelledError):
            await manager._system_status_publishing_loop()

        # Should have called publish_system_status
        manager.publish_system_status.assert_called()

        # Should have tried to wait for shutdown event
        assert mock_wait_for.call_count == 2

    @patch("asyncio.sleep")
    async def test_system_status_publishing_loop_error_handling(
        self, mock_sleep, mqtt_config, room_configs
    ):
        """Test system status publishing loop error handling."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Mock publishing to fail
        manager.publish_system_status = AsyncMock(
            side_effect=Exception("Publish error")
        )

        # Mock shutdown event to be set after one iteration
        async def mock_wait():
            if manager._shutdown_event.is_set():
                raise asyncio.CancelledError()
            else:
                manager._shutdown_event.set()
                raise asyncio.TimeoutError()

        with patch("asyncio.wait_for", side_effect=mock_wait):
            with pytest.raises(asyncio.CancelledError):
                await manager._system_status_publishing_loop()

        # Should have attempted to sleep after error
        mock_sleep.assert_called()

    async def test_system_status_publishing_loop_cancellation(
        self, mqtt_config, room_configs
    ):
        """Test system status publishing loop cancellation."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Start task and cancel immediately
        task = asyncio.create_task(manager._system_status_publishing_loop())
        await asyncio.sleep(0.01)  # Let task start
        task.cancel()

        # Should raise CancelledError
        with pytest.raises(asyncio.CancelledError):
            await task


# Enhanced Private Method Tests


class TestEnhancedPrivateMethods:
    """Test enhanced private methods."""

    async def test_check_system_availability_connected(self, mqtt_config, room_configs):
        """Test system availability check when connected."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Set up connected state
        manager._integration_active = True
        mock_mqtt_publisher = Mock()
        mock_mqtt_publisher.connection_status.connected = True
        manager.mqtt_publisher = mock_mqtt_publisher

        result = await manager._check_system_availability()

        assert result is True

    async def test_check_system_availability_not_connected(
        self, mqtt_config, room_configs
    ):
        """Test system availability check when not connected."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Set up disconnected state
        manager._integration_active = False

        result = await manager._check_system_availability()

        assert result is False

    async def test_check_system_availability_exception(self, mqtt_config, room_configs):
        """Test system availability check with exception."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Mock is_connected to raise exception
        with patch.object(
            manager, "is_connected", side_effect=Exception("Check failed")
        ):
            result = await manager._check_system_availability()

            assert result is False

    async def test_handle_entity_state_change(self, mqtt_config, room_configs):
        """Test entity state change handling."""
        callback = AsyncMock()
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=[callback],
        )

        from src.integration.discovery_publisher import EntityState

        await manager._handle_entity_state_change(
            "test_entity", EntityState.WARNING, {"warning": "test_warning"}
        )

        # Should notify callbacks
        callback.assert_called_once_with("entity_state_change:test_entity:warning")

    async def test_handle_entity_state_change_sync_callback(
        self, mqtt_config, room_configs
    ):
        """Test entity state change handling with sync callback."""
        sync_callback = Mock()
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=[sync_callback],
        )

        from src.integration.discovery_publisher import EntityState

        await manager._handle_entity_state_change(
            "test_entity", EntityState.ERROR, {"error": "test_error"}
        )

        # Should call sync callback
        sync_callback.assert_called_once_with("entity_state_change:test_entity:error")

    async def test_handle_entity_state_change_callback_error(
        self, mqtt_config, room_configs
    ):
        """Test entity state change handling with callback error."""
        failing_callback = AsyncMock(side_effect=Exception("Callback error"))
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=[failing_callback],
        )

        from src.integration.discovery_publisher import EntityState

        # Should not raise exception despite callback failure
        await manager._handle_entity_state_change(
            "test_entity", EntityState.ONLINE, {"status": "online"}
        )

        failing_callback.assert_called_once()


# Error Handling and Edge Cases Tests


class TestMQTTIntegrationErrorHandling:
    """Test integration error handling and edge cases."""

    def test_mqtt_integration_error_creation(self):
        """Test MQTT integration error creation."""
        error = MQTTIntegrationError(
            "Integration failed",
            severity=ErrorSeverity.HIGH,
            cause=Exception("Root cause"),
        )

        assert str(error) == "Integration failed"
        assert error.error_code == "MQTT_INTEGRATION_ERROR"
        assert error.severity == ErrorSeverity.HIGH
        assert error.__cause__ is not None

    def test_mqtt_integration_error_default_severity(self):
        """Test MQTT integration error with default severity."""
        error = MQTTIntegrationError("Test error")

        assert error.severity == ErrorSeverity.MEDIUM
        assert error.error_code == "MQTT_INTEGRATION_ERROR"

    async def test_integration_with_none_configs(self):
        """Test integration with None configurations."""
        with patch(
            "src.integration.mqtt_integration_manager.get_config"
        ) as mock_get_config:
            mock_config = Mock()
            mock_config.mqtt = MQTTConfig(
                broker="localhost",
                port=1883,
                topic_prefix="test",
                publishing_enabled=False,
            )
            mock_config.rooms = {}
            mock_get_config.return_value = mock_config

            manager = MQTTIntegrationManager(mqtt_config=None, rooms=None)

            assert manager.mqtt_config is not None
            assert manager.rooms is not None
            mock_get_config.assert_called_once()

    async def test_stop_integration_with_task_errors(self, mqtt_config, room_configs):
        """Test stopping integration with task errors."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        manager._integration_active = True

        # Mock tasks that raise exceptions when gathered
        failing_task = AsyncMock()
        working_task = AsyncMock()
        manager._background_tasks = [failing_task, working_task]

        with patch("asyncio.gather", return_value=[Exception("Task failed"), None]):
            await manager.stop_integration()

            assert manager._integration_active is False
            assert len(manager._background_tasks) == 0

    async def test_concurrent_integration_operations(self, mqtt_config, room_configs):
        """Test concurrent integration operations."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Mock components
        with patch.object(manager, "start_integration") as mock_start:
            mock_mqtt_pub = AsyncMock()
            mock_pred_pub = AsyncMock()

            with patch(
                "src.integration.mqtt_integration_manager.MQTTPublisher",
                return_value=mock_mqtt_pub,
            ), patch(
                "src.integration.mqtt_integration_manager.PredictionPublisher",
                return_value=mock_pred_pub,
            ):

                # Run initialization and start concurrently
                await asyncio.gather(
                    manager.initialize(),
                    manager.start_integration(),
                    return_exceptions=True,
                )

                # Should handle concurrency gracefully
                assert manager.stats.initialized is True or mock_start.called


# Integration Performance Tests


class TestMQTTIntegrationPerformance:
    """Test integration performance characteristics."""

    async def test_bulk_prediction_publishing(self, mqtt_config, room_configs):
        """Test bulk prediction publishing performance."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_prediction_publisher.publish_prediction.return_value = success_result
        manager.prediction_publisher = mock_prediction_publisher

        # Create multiple predictions
        predictions = []
        for i in range(10):
            prediction = PredictionResult(
                room_id=f"room_{i}",
                predicted_time=datetime.now(timezone.utc) + timedelta(minutes=30),
                transition_type="occupied_to_vacant",
                confidence=0.8 + (i * 0.01),
                model_version="v1.0.0",
                features_used=["temporal"],
            )
            predictions.append(prediction)

        # Publish all predictions concurrently
        tasks = [manager.publish_prediction(pred, pred.room_id) for pred in predictions]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(results)
        assert manager.stats.predictions_published == 10
        assert mock_prediction_publisher.publish_prediction.call_count == 10

    async def test_rapid_status_updates(self, mqtt_config, room_configs):
        """Test rapid status updates."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )
        manager._integration_active = True

        mock_prediction_publisher = AsyncMock()
        success_result = MQTTPublishResult(
            success=True, topic="test", payload_size=100, publish_time=datetime.utcnow()
        )
        mock_prediction_publisher.publish_system_status.return_value = success_result
        manager.prediction_publisher = mock_prediction_publisher

        # Perform rapid status updates
        results = []
        for i in range(20):
            result = await manager.publish_system_status(
                tracking_stats={"iteration": i},
                database_connected=i % 2 == 0,  # Alternate connection status
                active_alerts=i % 3,  # Varying alert counts
            )
            results.append(result)

        # All should succeed
        assert all(results)
        assert manager.stats.status_updates_published == 20
        assert mock_prediction_publisher.publish_system_status.call_count == 20

    async def test_statistics_access_performance(self, mqtt_config, room_configs):
        """Test performance of statistics access."""
        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
        )

        # Set up mock publishers with stats
        mock_mqtt_pub = Mock()
        mock_mqtt_pub.connection_status.connected = True
        mock_mqtt_pub.get_publisher_stats.return_value = {"stat": "value"}
        manager.mqtt_publisher = mock_mqtt_pub

        mock_pred_pub = Mock()
        mock_pred_pub.get_publisher_stats.return_value = {"stat": "value"}
        manager.prediction_publisher = mock_pred_pub

        mock_disc_pub = Mock()
        mock_disc_pub.get_discovery_stats.return_value = {
            "published_entities_count": 100,
            "device_available": True,
            "statistics": {"discovery_errors": 0},
        }
        manager.discovery_publisher = mock_disc_pub

        # Access stats multiple times rapidly
        stats_results = []
        for _ in range(50):
            stats = manager.get_integration_stats()
            stats_results.append(stats)

        # All should return complete stats
        assert len(stats_results) == 50
        for stats in stats_results:
            assert "system_health" in stats
            assert "discovery_insights" in stats
            assert "mqtt_publisher" in stats

    async def test_callback_notification_performance(self, mqtt_config, room_configs):
        """Test performance of callback notifications."""
        # Create many callbacks
        callbacks = []
        for i in range(20):
            if i % 2 == 0:
                callbacks.append(AsyncMock())  # Async callbacks
            else:
                callbacks.append(Mock())  # Sync callbacks

        manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=room_configs,
            notification_callbacks=callbacks,
        )

        # Trigger multiple notifications
        notification_tasks = [
            manager._on_mqtt_connect(None, None, None, None),
            manager._on_mqtt_disconnect(None, None, None, None),
        ]

        # Should handle all callbacks efficiently
        await asyncio.gather(*notification_tasks)

        # All callbacks should have been called
        for callback in callbacks:
            assert callback.called or callback.call_count > 0
