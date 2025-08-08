"""
Home Assistant MQTT Discovery Publisher - Enhanced Version.

This module handles the creation and publishing of Home Assistant MQTT discovery
messages for automatic device and entity registration in Home Assistant.

Enhanced Features:
- Automatic sensor entity creation for predictions
- System status sensors and diagnostics  
- Proper device registry integration with lifecycle management
- Advanced entity state management with proper state classes
- Device availability tracking and status reporting
- Home Assistant service integration for manual controls
- Entity configuration validation and error handling
- Advanced sensor attributes and metadata
- Automatic cleanup and entity removal capabilities
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Union

from ..core.config import MQTTConfig, RoomConfig
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from .mqtt_publisher import MQTTPublisher, MQTTPublishResult

logger = logging.getLogger(__name__)


class EntityState(Enum):
    """Home Assistant entity states."""

    UNKNOWN = "unknown"
    UNAVAILABLE = "unavailable"
    ONLINE = "online"
    OFFLINE = "offline"
    OK = "ok"
    ERROR = "error"
    WARNING = "warning"


class EntityCategory(Enum):
    """Home Assistant entity categories."""

    CONFIG = "config"
    DIAGNOSTIC = "diagnostic"
    SYSTEM = "system"


class DeviceClass(Enum):
    """Home Assistant device classes."""

    TIMESTAMP = "timestamp"
    DURATION = "duration"
    DATA_SIZE = "data_size"
    ENUM = "enum"


@dataclass
class EntityAvailability:
    """Entity availability configuration."""

    topic: str
    payload_available: str = "online"
    payload_not_available: str = "offline"
    value_template: Optional[str] = None


@dataclass
class ServiceConfig:
    """Home Assistant service configuration."""

    service_name: str
    service_topic: str
    schema: str = "json"
    command_template: Optional[str] = None
    retain: bool = False


@dataclass
class EntityMetadata:
    """Enhanced metadata for HA entities."""

    entity_id: str
    friendly_name: str
    created_at: datetime
    last_updated: Optional[datetime] = None
    entity_category: Optional[EntityCategory] = None
    availability: Optional[EntityAvailability] = None
    services: List[ServiceConfig] = field(default_factory=list)
    attributes: Dict[str, Any] = field(default_factory=dict)
    state: EntityState = EntityState.UNKNOWN
    last_seen: Optional[datetime] = None


@dataclass
class DeviceInfo:
    """Enhanced Home Assistant device information."""

    identifiers: List[str]
    name: str
    manufacturer: str
    model: str
    sw_version: str
    configuration_url: Optional[str] = None
    suggested_area: Optional[str] = None
    via_device: Optional[str] = None
    connections: Optional[List[List[str]]] = None
    hw_version: Optional[str] = None

    # Enhanced device metadata
    device_class: Optional[str] = None
    availability_topic: Optional[str] = None
    last_seen: Optional[datetime] = None
    capabilities: Dict[str, Any] = field(default_factory=dict)
    diagnostic_info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SensorConfig:
    """Enhanced Home Assistant sensor configuration."""

    # Core sensor information
    name: str
    unique_id: str
    state_topic: str
    device: DeviceInfo

    # Optional sensor attributes
    json_attributes_topic: Optional[str] = None
    value_template: Optional[str] = None
    json_attributes_template: Optional[str] = None
    unit_of_measurement: Optional[str] = None
    device_class: Optional[str] = None
    state_class: Optional[str] = None
    icon: Optional[str] = None
    entity_category: Optional[str] = None
    enabled_by_default: bool = True
    expire_after: Optional[int] = None

    # Enhanced availability
    availability_topic: Optional[str] = None
    availability_template: Optional[str] = None
    availability_mode: str = "latest"  # latest, all, any

    # Advanced configuration
    force_update: bool = False
    last_reset_topic: Optional[str] = None
    last_reset_value_template: Optional[str] = None
    suggested_display_precision: Optional[int] = None
    entity_picture: Optional[str] = None

    # Service integration
    command_topic: Optional[str] = None
    command_template: Optional[str] = None
    retain: bool = False
    qos: int = 1

    # Enhanced metadata
    metadata: Optional[EntityMetadata] = None
    validation_schema: Optional[Dict[str, Any]] = None


class DiscoveryPublisher:
    """
    Enhanced Home Assistant MQTT Discovery Publisher.

    Creates and manages MQTT discovery messages for automatic registration
    of occupancy prediction sensors and system status in Home Assistant.

    Enhanced Features:
    - Advanced entity lifecycle management
    - Device availability tracking and status reporting
    - Home Assistant service integration
    - Entity state management with proper state classes
    - Automatic cleanup and error recovery
    - Entity validation and configuration verification
    """

    def __init__(
        self,
        mqtt_publisher: MQTTPublisher,
        config: MQTTConfig,
        rooms: Dict[str, RoomConfig],
        availability_check_callback: Optional[Callable] = None,
        state_change_callback: Optional[Callable] = None,
    ):
        """
        Initialize enhanced discovery publisher.

        Args:
            mqtt_publisher: Core MQTT publisher instance
            config: MQTT configuration
            rooms: Room configuration mapping
            availability_check_callback: Optional callback for device availability checks
            state_change_callback: Optional callback for entity state changes
        """
        self.mqtt_publisher = mqtt_publisher
        self.config = config
        self.rooms = rooms
        self.availability_check_callback = availability_check_callback
        self.state_change_callback = state_change_callback

        # Enhanced device information with availability tracking
        availability_topic = f"{self.config.topic_prefix}/device/availability"
        self.device_info = DeviceInfo(
            identifiers=[self.config.device_identifier],
            name=self.config.device_name,
            manufacturer=self.config.device_manufacturer,
            model=self.config.device_model,
            sw_version=self.config.device_sw_version,
            configuration_url=getattr(self.config, "device_configuration_url", None),
            suggested_area="Home Assistant ML Predictor",
            device_class="connectivity",
            availability_topic=availability_topic,
            last_seen=datetime.utcnow(),
            capabilities={
                "prediction_types": ["occupancy", "vacancy"],
                "rooms_supported": len(self.rooms),
                "features": ["temporal", "sequential", "contextual"],
                "models": ["LSTM", "XGBoost", "HMM", "Ensemble"],
            },
            diagnostic_info={
                "initialized_at": datetime.utcnow().isoformat(),
                "version": self.config.device_sw_version,
                "mqtt_enabled": True,
                "discovery_enabled": self.config.discovery_enabled,
            },
        )

        # Enhanced discovery state tracking
        self.discovery_published = False
        self.published_entities: Dict[str, str] = {}  # entity_id -> discovery_topic
        self.entity_metadata: Dict[str, EntityMetadata] = {}  # entity_id -> metadata
        self.device_available = True
        self.last_availability_publish = None

        # Service integration
        self.available_services: Dict[str, ServiceConfig] = {}
        self.command_handlers: Dict[str, Callable] = {}

        # Statistics and monitoring
        self.discovery_stats = {
            "entities_created": 0,
            "entities_removed": 0,
            "discovery_publishes": 0,
            "discovery_errors": 0,
            "last_discovery_refresh": None,
            "availability_updates": 0,
        }

        logger.info(
            f"Initialized Enhanced DiscoveryPublisher for device: {self.config.device_name}"
        )
        logger.info(
            f"Device capabilities: {list(self.device_info.capabilities.keys())}"
        )
        logger.info(f"Availability topic: {availability_topic}")

    async def publish_all_discovery(self) -> Dict[str, MQTTPublishResult]:
        """
        Enhanced discovery publishing with device availability and service integration.

        Returns:
            Dictionary mapping entity names to publish results
        """
        if not self.config.discovery_enabled:
            logger.info("MQTT discovery is disabled")
            return {}

        try:
            results = {}

            # Step 1: Publish device availability first
            logger.info("Publishing device availability status")
            await self.publish_device_availability(online=True)

            # Step 2: Publish room prediction sensors with enhanced metadata
            logger.info(f"Publishing discovery for {len(self.rooms)} rooms")
            for room_id, room_config in self.rooms.items():
                room_results = await self.publish_room_discovery(room_id, room_config)
                results.update(room_results)

            # Step 3: Publish system sensors with enhanced diagnostics
            logger.info("Publishing system sensors discovery")
            system_results = await self.publish_system_discovery()
            results.update(system_results)

            # Step 4: Publish service configurations for manual controls
            logger.info("Publishing service configurations")
            service_results = await self.publish_service_discovery()
            results.update(service_results)

            # Step 5: Update discovery state and statistics
            self.discovery_published = True
            self.discovery_stats["discovery_publishes"] += 1
            self.discovery_stats["last_discovery_refresh"] = datetime.utcnow()
            self.discovery_stats["entities_created"] = len(results)

            successful = sum(1 for r in results.values() if r.success)
            total = len(results)

            if successful == total:
                logger.info(
                    f"Successfully published discovery for all {total} entities"
                )
            else:
                failed = total - successful
                self.discovery_stats["discovery_errors"] += failed
                logger.warning(
                    f"Discovery publishing partially failed: {successful}/{total} successful, {failed} failed"
                )

            # Step 6: Validate published entities
            await self._validate_published_entities(results)

            return results

        except Exception as e:
            self.discovery_stats["discovery_errors"] += 1
            logger.error(f"Error publishing discovery messages: {e}")
            return {}

    async def publish_room_discovery(
        self, room_id: str, room_config: RoomConfig
    ) -> Dict[str, MQTTPublishResult]:
        """
        Publish discovery messages for a specific room.

        Args:
            room_id: Room identifier
            room_config: Room configuration

        Returns:
            Dictionary mapping entity names to publish results
        """
        results = {}
        room_name = room_config.name

        try:
            # Main prediction sensor
            prediction_sensor = self._create_prediction_sensor(room_id, room_name)
            result = await self._publish_sensor_discovery(prediction_sensor)
            results[f"{room_id}_prediction"] = result

            # Next transition time sensor
            next_transition_sensor = self._create_next_transition_sensor(
                room_id, room_name
            )
            result = await self._publish_sensor_discovery(next_transition_sensor)
            results[f"{room_id}_next_transition"] = result

            # Confidence sensor
            confidence_sensor = self._create_confidence_sensor(room_id, room_name)
            result = await self._publish_sensor_discovery(confidence_sensor)
            results[f"{room_id}_confidence"] = result

            # Time until sensor
            time_until_sensor = self._create_time_until_sensor(room_id, room_name)
            result = await self._publish_sensor_discovery(time_until_sensor)
            results[f"{room_id}_time_until"] = result

            # Reliability sensor
            reliability_sensor = self._create_reliability_sensor(room_id, room_name)
            result = await self._publish_sensor_discovery(reliability_sensor)
            results[f"{room_id}_reliability"] = result

            logger.info(
                f"Published discovery for room {room_name}: {len(results)} sensors"
            )
            return results

        except Exception as e:
            logger.error(f"Error publishing room discovery for {room_id}: {e}")
            return results

    async def publish_system_discovery(self) -> Dict[str, MQTTPublishResult]:
        """
        Publish discovery messages for system status sensors.

        Returns:
            Dictionary mapping entity names to publish results
        """
        results = {}

        try:
            # System status sensor
            status_sensor = self._create_system_status_sensor()
            result = await self._publish_sensor_discovery(status_sensor)
            results["system_status"] = result

            # Uptime sensor
            uptime_sensor = self._create_uptime_sensor()
            result = await self._publish_sensor_discovery(uptime_sensor)
            results["system_uptime"] = result

            # Predictions count sensor
            predictions_sensor = self._create_predictions_count_sensor()
            result = await self._publish_sensor_discovery(predictions_sensor)
            results["predictions_count"] = result

            # Accuracy sensor
            accuracy_sensor = self._create_accuracy_sensor()
            result = await self._publish_sensor_discovery(accuracy_sensor)
            results["system_accuracy"] = result

            # Active alerts sensor
            alerts_sensor = self._create_alerts_sensor()
            result = await self._publish_sensor_discovery(alerts_sensor)
            results["active_alerts"] = result

            # Database status sensor
            database_sensor = self._create_database_status_sensor()
            result = await self._publish_sensor_discovery(database_sensor)
            results["database_status"] = result

            # Tracking status sensor
            tracking_sensor = self._create_tracking_status_sensor()
            result = await self._publish_sensor_discovery(tracking_sensor)
            results["tracking_status"] = result

            logger.info(f"Published system discovery: {len(results)} sensors")
            return results

        except Exception as e:
            logger.error(f"Error publishing system discovery: {e}")
            return results

    async def remove_discovery(self, entity_name: str) -> MQTTPublishResult:
        """
        Remove a discovery message (publish empty payload).

        Args:
            entity_name: Name of entity to remove

        Returns:
            MQTTPublishResult indicating success/failure
        """
        try:
            if entity_name in self.published_entities:
                discovery_topic = self.published_entities[entity_name]

                # Publish empty payload to remove discovery
                result = await self.mqtt_publisher.publish(
                    topic=discovery_topic, payload="", qos=1, retain=True
                )

                if result.success:
                    del self.published_entities[entity_name]
                    logger.info(f"Removed discovery for entity: {entity_name}")

                return result
            else:
                logger.warning(f"Entity {entity_name} not found in published entities")
                return MQTTPublishResult(
                    success=False,
                    topic="",
                    payload_size=0,
                    publish_time=datetime.utcnow(),
                    error_message="Entity not found",
                )

        except Exception as e:
            logger.error(f"Error removing discovery for {entity_name}: {e}")
            return MQTTPublishResult(
                success=False,
                topic="",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e),
            )

    async def refresh_discovery(self) -> Dict[str, MQTTPublishResult]:
        """
        Refresh all discovery messages.

        Returns:
            Dictionary mapping entity names to publish results
        """
        logger.info("Refreshing all discovery messages")
        self.discovery_published = False
        self.published_entities.clear()
        return await self.publish_all_discovery()

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get enhanced discovery publisher statistics."""
        return {
            "discovery_enabled": self.config.discovery_enabled,
            "discovery_published": self.discovery_published,
            "published_entities_count": len(self.published_entities),
            "published_entities": list(self.published_entities.keys()),
            "entity_metadata_count": len(self.entity_metadata),
            "rooms_configured": len(self.rooms),
            "device_available": self.device_available,
            "last_availability_publish": (
                self.last_availability_publish.isoformat()
                if self.last_availability_publish
                else None
            ),
            "available_services_count": len(self.available_services),
            "available_services": list(self.available_services.keys()),
            "device_info": {
                "name": self.device_info.name,
                "identifier": self.device_info.identifiers[0],
                "manufacturer": self.device_info.manufacturer,
                "model": self.device_info.model,
                "sw_version": self.device_info.sw_version,
                "capabilities": self.device_info.capabilities,
                "last_seen": (
                    self.device_info.last_seen.isoformat()
                    if self.device_info.last_seen
                    else None
                ),
            },
            "statistics": self.discovery_stats,
        }

    # Enhanced Discovery Methods

    async def publish_device_availability(
        self, online: bool = True
    ) -> MQTTPublishResult:
        """
        Publish device availability status to Home Assistant.

        Args:
            online: Whether device is online or offline

        Returns:
            MQTTPublishResult indicating success/failure
        """
        try:
            availability_topic = self.device_info.availability_topic
            if not availability_topic:
                logger.warning("Device availability topic not configured")
                return MQTTPublishResult(
                    success=False,
                    topic="",
                    payload_size=0,
                    publish_time=datetime.utcnow(),
                    error_message="Availability topic not configured",
                )

            # Create availability payload with enhanced metadata
            availability_payload = {
                "status": "online" if online else "offline",
                "timestamp": datetime.utcnow().isoformat(),
                "device_id": self.config.device_identifier,
                "version": self.config.device_sw_version,
                "capabilities": self.device_info.capabilities,
                "entities_count": len(self.published_entities),
                "services_count": len(self.available_services),
            }

            result = await self.mqtt_publisher.publish_json(
                topic=availability_topic, data=availability_payload, qos=1, retain=True
            )

            if result.success:
                self.device_available = online
                self.last_availability_publish = datetime.utcnow()
                self.device_info.last_seen = datetime.utcnow()
                self.discovery_stats["availability_updates"] += 1

                logger.info(
                    f"Published device availability: {'online' if online else 'offline'}"
                )
            else:
                logger.error(
                    f"Failed to publish device availability: {result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"Error publishing device availability: {e}")
            return MQTTPublishResult(
                success=False,
                topic=availability_topic or "",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e),
            )

    async def publish_service_discovery(self) -> Dict[str, MQTTPublishResult]:
        """
        Publish Home Assistant service discovery for manual controls.

        Returns:
            Dictionary mapping service names to publish results
        """
        try:
            if not self.config.discovery_enabled:
                logger.info("Service discovery disabled")
                return {}

            results = {}

            # Define available services for manual control
            services = [
                {
                    "service_name": "manual_retrain",
                    "friendly_name": "Manual Retrain Model",
                    "icon": "mdi:brain",
                    "description": "Manually trigger model retraining",
                    "command_topic": f"{self.config.topic_prefix}/commands/retrain",
                    "command_template": "{{ value_json }}",
                },
                {
                    "service_name": "refresh_discovery",
                    "friendly_name": "Refresh Discovery",
                    "icon": "mdi:refresh",
                    "description": "Refresh Home Assistant discovery messages",
                    "command_topic": f"{self.config.topic_prefix}/commands/refresh_discovery",
                    "command_template": "{{ value_json }}",
                },
                {
                    "service_name": "reset_statistics",
                    "friendly_name": "Reset Statistics",
                    "icon": "mdi:counter",
                    "description": "Reset system statistics and counters",
                    "command_topic": f"{self.config.topic_prefix}/commands/reset_stats",
                    "command_template": "{{ value_json }}",
                },
                {
                    "service_name": "force_prediction",
                    "friendly_name": "Force Prediction",
                    "icon": "mdi:play-circle",
                    "description": "Force prediction generation for specific room",
                    "command_topic": f"{self.config.topic_prefix}/commands/force_prediction",
                    "command_template": "{{ value_json }}",
                },
            ]

            # Publish each service as a button entity
            for service in services:
                service_result = await self._publish_service_button(service)
                results[service["service_name"]] = service_result

                if service_result.success:
                    # Store service configuration
                    service_config = ServiceConfig(
                        service_name=service["service_name"],
                        service_topic=service["command_topic"],
                        command_template=service["command_template"],
                    )
                    self.available_services[service["service_name"]] = service_config

            logger.info(f"Published service discovery for {len(services)} services")
            return results

        except Exception as e:
            logger.error(f"Error publishing service discovery: {e}")
            return {}

    async def update_entity_state(
        self,
        entity_id: str,
        state: EntityState,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """
        Update entity state and metadata.

        Args:
            entity_id: Entity identifier
            state: New entity state
            attributes: Optional attributes to update

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if entity_id not in self.entity_metadata:
                logger.warning(f"Entity {entity_id} not found in metadata")
                return False

            # Update entity metadata
            metadata = self.entity_metadata[entity_id]
            metadata.state = state
            metadata.last_updated = datetime.utcnow()
            metadata.last_seen = datetime.utcnow()

            if attributes:
                metadata.attributes.update(attributes)

            # Notify state change callback if available
            if self.state_change_callback:
                try:
                    if asyncio.iscoroutinefunction(self.state_change_callback):
                        await self.state_change_callback(entity_id, state, attributes)
                    else:
                        self.state_change_callback(entity_id, state, attributes)
                except Exception as e:
                    logger.error(f"Error in state change callback: {e}")

            logger.debug(f"Updated entity {entity_id} state to {state.value}")
            return True

        except Exception as e:
            logger.error(f"Error updating entity state for {entity_id}: {e}")
            return False

    async def cleanup_entities(
        self, entity_ids: Optional[List[str]] = None
    ) -> Dict[str, MQTTPublishResult]:
        """
        Clean up entities by removing their discovery messages.

        Args:
            entity_ids: Optional list of specific entities to clean up (None = all)

        Returns:
            Dictionary mapping entity names to cleanup results
        """
        try:
            cleanup_entities = entity_ids or list(self.published_entities.keys())
            results = {}

            logger.info(f"Cleaning up {len(cleanup_entities)} entities")

            for entity_id in cleanup_entities:
                result = await self.remove_discovery(entity_id)
                results[entity_id] = result

                if result.success:
                    # Remove from metadata
                    if entity_id in self.entity_metadata:
                        del self.entity_metadata[entity_id]

                    self.discovery_stats["entities_removed"] += 1

            successful = sum(1 for r in results.values() if r.success)
            logger.info(f"Cleaned up {successful}/{len(cleanup_entities)} entities")

            return results

        except Exception as e:
            logger.error(f"Error cleaning up entities: {e}")
            return {}

    # Private Methods - Enhanced Discovery Support

    async def _validate_published_entities(
        self, results: Dict[str, MQTTPublishResult]
    ) -> None:
        """
        Validate published entities and create metadata entries.

        Args:
            results: Dictionary of publish results to validate
        """
        try:
            for entity_name, result in results.items():
                if result.success and entity_name not in self.entity_metadata:
                    # Create metadata for successfully published entity
                    metadata = EntityMetadata(
                        entity_id=entity_name,
                        friendly_name=entity_name.replace("_", " ").title(),
                        created_at=datetime.utcnow(),
                        entity_category=EntityCategory.DIAGNOSTIC,
                        state=EntityState.ONLINE,
                        last_seen=datetime.utcnow(),
                    )

                    self.entity_metadata[entity_name] = metadata
                    logger.debug(f"Created metadata for entity: {entity_name}")

        except Exception as e:
            logger.error(f"Error validating published entities: {e}")

    async def _publish_service_button(
        self, service: Dict[str, Any]
    ) -> MQTTPublishResult:
        """
        Publish a service as a Home Assistant button entity.

        Args:
            service: Service configuration dictionary

        Returns:
            MQTTPublishResult indicating success/failure
        """
        try:
            # Create button discovery topic
            discovery_topic = (
                f"{self.config.discovery_prefix}/button/"
                f"{self.config.device_identifier}/"
                f"{service['service_name']}/config"
            )

            # Create button discovery payload
            discovery_payload = {
                "name": service["friendly_name"],
                "unique_id": f"{self.config.device_identifier}_{service['service_name']}",
                "command_topic": service["command_topic"],
                "device": {
                    "identifiers": self.device_info.identifiers,
                    "name": self.device_info.name,
                    "manufacturer": self.device_info.manufacturer,
                    "model": self.device_info.model,
                    "sw_version": self.device_info.sw_version,
                },
                "icon": service["icon"],
                "entity_category": "config",
                "availability_topic": self.device_info.availability_topic,
                "availability_template": "{{ 'online' if value_json.status == 'online' else 'offline' }}",
                "payload_press": json.dumps(
                    {
                        "action": service["service_name"],
                        "timestamp": "{{ now().isoformat() }}",
                    }
                ),
            }

            # Add command template if provided
            if service.get("command_template"):
                discovery_payload["command_template"] = service["command_template"]

            # Publish button discovery
            result = await self.mqtt_publisher.publish_json(
                topic=discovery_topic, data=discovery_payload, qos=1, retain=True
            )

            if result.success:
                self.published_entities[service["service_name"]] = discovery_topic
                logger.debug(f"Published service button: {service['friendly_name']}")
            else:
                logger.error(
                    f"Failed to publish service button {service['friendly_name']}: {result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"Error publishing service button: {e}")
            return MQTTPublishResult(
                success=False,
                topic="",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e),
            )

    # Private methods - Sensor configuration creators

    def _create_prediction_sensor(self, room_id: str, room_name: str) -> SensorConfig:
        """Create main prediction sensor configuration."""
        return SensorConfig(
            name=f"{room_name} Occupancy Prediction",
            unique_id=f"{self.config.device_identifier}_{room_id}_prediction",
            state_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            device=self.device_info,
            value_template="{{ value_json.transition_type }}",
            json_attributes_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            json_attributes_template="{{ value_json | tojson }}",
            icon="mdi:home-account",
            entity_category="diagnostic",
            expire_after=600,  # 10 minutes
        )

    def _create_next_transition_sensor(
        self, room_id: str, room_name: str
    ) -> SensorConfig:
        """Create next transition time sensor configuration."""
        return SensorConfig(
            name=f"{room_name} Next Transition",
            unique_id=f"{self.config.device_identifier}_{room_id}_next_transition",
            state_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            device=self.device_info,
            value_template="{{ value_json.predicted_time }}",
            device_class="timestamp",
            icon="mdi:clock-outline",
            expire_after=600,
        )

    def _create_confidence_sensor(self, room_id: str, room_name: str) -> SensorConfig:
        """Create confidence sensor configuration."""
        return SensorConfig(
            name=f"{room_name} Confidence",
            unique_id=f"{self.config.device_identifier}_{room_id}_confidence",
            state_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            device=self.device_info,
            value_template="{{ (value_json.confidence_score * 100) | round(1) }}",
            unit_of_measurement="%",
            icon="mdi:percent",
            entity_category="diagnostic",
            expire_after=600,
        )

    def _create_time_until_sensor(self, room_id: str, room_name: str) -> SensorConfig:
        """Create time until sensor configuration."""
        return SensorConfig(
            name=f"{room_name} Time Until",
            unique_id=f"{self.config.device_identifier}_{room_id}_time_until",
            state_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            device=self.device_info,
            value_template="{{ value_json.time_until_human }}",
            icon="mdi:timer-outline",
            expire_after=600,
        )

    def _create_reliability_sensor(self, room_id: str, room_name: str) -> SensorConfig:
        """Create reliability sensor configuration."""
        return SensorConfig(
            name=f"{room_name} Reliability",
            unique_id=f"{self.config.device_identifier}_{room_id}_reliability",
            state_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            device=self.device_info,
            value_template="{{ value_json.prediction_reliability }}",
            icon="mdi:check-circle-outline",
            entity_category="diagnostic",
            expire_after=600,
        )

    def _create_system_status_sensor(self) -> SensorConfig:
        """Create system status sensor configuration."""
        return SensorConfig(
            name="System Status",
            unique_id=f"{self.config.device_identifier}_system_status",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{{ value_json.system_status }}",
            json_attributes_topic=f"{self.config.topic_prefix}/system/status",
            json_attributes_template="{{ value_json | tojson }}",
            icon="mdi:server",
            entity_category="diagnostic",
        )

    def _create_uptime_sensor(self) -> SensorConfig:
        """Create uptime sensor configuration."""
        return SensorConfig(
            name="System Uptime",
            unique_id=f"{self.config.device_identifier}_uptime",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{{ value_json.uptime_seconds }}",
            unit_of_measurement="s",
            device_class="duration",
            state_class="total",
            icon="mdi:clock-check-outline",
            entity_category="diagnostic",
        )

    def _create_predictions_count_sensor(self) -> SensorConfig:
        """Create predictions count sensor configuration."""
        return SensorConfig(
            name="Total Predictions",
            unique_id=f"{self.config.device_identifier}_predictions_count",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{{ value_json.total_predictions_made }}",
            state_class="total_increasing",
            icon="mdi:counter",
            entity_category="diagnostic",
        )

    def _create_accuracy_sensor(self) -> SensorConfig:
        """Create accuracy sensor configuration."""
        return SensorConfig(
            name="System Accuracy",
            unique_id=f"{self.config.device_identifier}_accuracy",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{{ value_json.average_accuracy_percent | round(1) }}",
            unit_of_measurement="%",
            icon="mdi:target",
            entity_category="diagnostic",
        )

    def _create_alerts_sensor(self) -> SensorConfig:
        """Create active alerts sensor configuration."""
        return SensorConfig(
            name="Active Alerts",
            unique_id=f"{self.config.device_identifier}_alerts",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{{ value_json.active_alerts }}",
            state_class="measurement",
            icon="mdi:alert-circle-outline",
            entity_category="diagnostic",
        )

    def _create_database_status_sensor(self) -> SensorConfig:
        """Create database status sensor configuration."""
        return SensorConfig(
            name="Database Connected",
            unique_id=f"{self.config.device_identifier}_database",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{% if value_json.database_connected %}Connected{% else %}Disconnected{% endif %}",
            icon="mdi:database",
            entity_category="diagnostic",
        )

    def _create_tracking_status_sensor(self) -> SensorConfig:
        """Create tracking status sensor configuration."""
        return SensorConfig(
            name="Tracking Active",
            unique_id=f"{self.config.device_identifier}_tracking",
            state_topic=f"{self.config.topic_prefix}/system/status",
            device=self.device_info,
            value_template="{% if value_json.tracking_active %}Active{% else %}Inactive{% endif %}",
            icon="mdi:chart-line",
            entity_category="diagnostic",
        )

    # Private methods - Discovery publishing

    async def _publish_sensor_discovery(
        self, sensor_config: SensorConfig
    ) -> MQTTPublishResult:
        """Enhanced sensor discovery publishing with metadata and validation."""
        try:
            # Create discovery topic
            discovery_topic = (
                f"{self.config.discovery_prefix}/sensor/"
                f"{self.config.device_identifier}/"
                f"{sensor_config.unique_id}/config"
            )

            # Create enhanced discovery payload
            discovery_payload = {
                "name": sensor_config.name,
                "unique_id": sensor_config.unique_id,
                "state_topic": sensor_config.state_topic,
                "device": {
                    "identifiers": sensor_config.device.identifiers,
                    "name": sensor_config.device.name,
                    "manufacturer": sensor_config.device.manufacturer,
                    "model": sensor_config.device.model,
                    "sw_version": sensor_config.device.sw_version,
                },
            }

            # Add enhanced device information if available
            if (
                hasattr(sensor_config.device, "suggested_area")
                and sensor_config.device.suggested_area
            ):
                discovery_payload["device"][
                    "suggested_area"
                ] = sensor_config.device.suggested_area
            if (
                hasattr(sensor_config.device, "configuration_url")
                and sensor_config.device.configuration_url
            ):
                discovery_payload["device"][
                    "configuration_url"
                ] = sensor_config.device.configuration_url
            if (
                hasattr(sensor_config.device, "hw_version")
                and sensor_config.device.hw_version
            ):
                discovery_payload["device"][
                    "hw_version"
                ] = sensor_config.device.hw_version

            # Add enhanced optional attributes
            enhanced_attrs = [
                "json_attributes_topic",
                "value_template",
                "json_attributes_template",
                "unit_of_measurement",
                "device_class",
                "state_class",
                "icon",
                "entity_category",
                "enabled_by_default",
                "expire_after",
                "availability_topic",
                "availability_template",
                "availability_mode",
                "force_update",
                "last_reset_topic",
                "last_reset_value_template",
                "suggested_display_precision",
                "entity_picture",
                "command_topic",
                "command_template",
                "retain",
                "qos",
            ]

            for attr in enhanced_attrs:
                value = getattr(sensor_config, attr, None)
                if value is not None:
                    # Skip boolean attributes that are False (except for explicit False values)
                    if (
                        attr in ["enabled_by_default", "force_update", "retain"]
                        and value is False
                    ):
                        discovery_payload[attr] = value
                    elif value:
                        discovery_payload[attr] = value

            # Add device availability if configured
            if self.device_info.availability_topic:
                discovery_payload["availability"] = {
                    "topic": self.device_info.availability_topic,
                    "payload_available": "online",
                    "payload_not_available": "offline",
                    "value_template": "{{ 'online' if value_json.status == 'online' else 'offline' }}",
                }

            # Publish discovery message
            result = await self.mqtt_publisher.publish_json(
                topic=discovery_topic,
                data=discovery_payload,
                qos=sensor_config.qos,
                retain=True,
            )

            if result.success:
                self.published_entities[sensor_config.unique_id] = discovery_topic

                # Create entity metadata if not exists
                if sensor_config.unique_id not in self.entity_metadata:
                    metadata = EntityMetadata(
                        entity_id=sensor_config.unique_id,
                        friendly_name=sensor_config.name,
                        created_at=datetime.utcnow(),
                        entity_category=(
                            EntityCategory.DIAGNOSTIC
                            if sensor_config.entity_category == "diagnostic"
                            else EntityCategory.SYSTEM
                        ),
                        state=EntityState.ONLINE,
                        last_seen=datetime.utcnow(),
                    )

                    if sensor_config.metadata:
                        metadata.attributes.update(sensor_config.metadata.attributes)

                    self.entity_metadata[sensor_config.unique_id] = metadata

                logger.debug(
                    f"Published enhanced discovery for sensor: {sensor_config.name}"
                )
            else:
                logger.error(
                    f"Failed to publish sensor discovery: {result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"Error publishing enhanced sensor discovery: {e}")
            return MQTTPublishResult(
                success=False,
                topic="",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e),
            )


class EnhancedDiscoveryError(OccupancyPredictionError):
    """Raised when enhanced Home Assistant discovery operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="ENHANCED_DISCOVERY_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
