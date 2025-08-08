"""
Home Assistant Entity Definitions and MQTT Discovery Enhancements.

This module provides comprehensive Home Assistant entity definitions that build on
the existing discovery system to create a complete HA entity ecosystem with
advanced entity types, proper device classes, units of measurement, and service
definitions for system control.

Features:
- Complete entity type definitions (sensors, binary_sensors, buttons, switches)
- Advanced diagnostic entities for system monitoring
- Service definitions for HA control integration
- Proper device classes, units, and state classes
- Entity category management and lifecycle control
- Seamless integration with existing MQTT and tracking systems
"""

import json
import logging
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from datetime import timedelta
from enum import Enum
from enum import auto
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

from ..core.config import MQTTConfig
from ..core.config import RoomConfig
from ..core.config import TrackingConfig
from ..core.exceptions import ErrorSeverity
from ..core.exceptions import OccupancyPredictionError
from .discovery_publisher import DeviceInfo
from .discovery_publisher import DiscoveryPublisher
from .discovery_publisher import EntityMetadata
from .discovery_publisher import EntityState
from .discovery_publisher import SensorConfig
from .mqtt_publisher import MQTTPublisher
from .mqtt_publisher import MQTTPublishResult

logger = logging.getLogger(__name__)


class HAEntityType(Enum):
    """Home Assistant entity types supported by the system."""

    SENSOR = "sensor"
    BINARY_SENSOR = "binary_sensor"
    BUTTON = "button"
    SWITCH = "switch"
    NUMBER = "number"
    SELECT = "select"
    TEXT = "text"
    DEVICE_TRACKER = "device_tracker"
    # Additional entity types for comprehensive system control
    DATETIME = "datetime"
    TIME = "time"
    DATE = "date"
    IMAGE = "image"


class HADeviceClass(Enum):
    """Home Assistant device classes for proper categorization."""

    # Sensor device classes
    TIMESTAMP = "timestamp"
    DURATION = "duration"
    DATA_SIZE = "data_size"
    ENUM = "enum"
    FREQUENCY = "frequency"
    POWER = "power"
    ENERGY = "energy"
    TEMPERATURE = "temperature"
    HUMIDITY = "humidity"
    PRESSURE = "pressure"
    SPEED = "speed"
    VOLTAGE = "voltage"
    CURRENT = "current"

    # Binary sensor device classes
    CONNECTIVITY = "connectivity"
    PROBLEM = "problem"
    RUNNING = "running"
    UPDATE = "update"
    SAFETY = "safety"
    MOTION = "motion"
    OCCUPANCY = "occupancy"
    OPENING = "opening"

    # Number device classes
    DISTANCE = "distance"
    WEIGHT = "weight"
    VOLUME = "volume"

    # Button device classes
    RESTART = "restart"
    IDENTIFY = "identify"
    UPDATE = "update"


class HAEntityCategory(Enum):
    """Home Assistant entity categories for organization."""

    CONFIG = "config"
    DIAGNOSTIC = "diagnostic"
    SYSTEM = "system"


class HAStateClass(Enum):
    """Home Assistant state classes for sensor entities."""

    MEASUREMENT = "measurement"
    TOTAL = "total"
    TOTAL_INCREASING = "total_increasing"


@dataclass
class HAEntityConfig:
    """Base configuration for Home Assistant entities."""

    entity_type: HAEntityType
    name: str
    unique_id: str
    state_topic: str
    device: DeviceInfo

    # Common attributes
    icon: Optional[str] = None
    entity_category: Optional[str] = None
    enabled_by_default: bool = True
    availability_topic: Optional[str] = None
    availability_template: Optional[str] = None
    expire_after: Optional[int] = None

    # Entity-specific attributes
    extra_attributes: Dict[str, Any] = field(default_factory=dict)

    # Metadata
    description: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class HASensorEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant sensor entities."""

    entity_type: HAEntityType = HAEntityType.SENSOR

    # Sensor-specific attributes
    value_template: Optional[str] = None
    json_attributes_topic: Optional[str] = None
    json_attributes_template: Optional[str] = None
    unit_of_measurement: Optional[str] = None
    device_class: Optional[str] = None
    state_class: Optional[str] = None
    suggested_display_precision: Optional[int] = None
    force_update: bool = False
    last_reset_topic: Optional[str] = None
    last_reset_value_template: Optional[str] = None


@dataclass
class HABinarySensorEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant binary sensor entities."""

    entity_type: HAEntityType = HAEntityType.BINARY_SENSOR

    # Binary sensor-specific attributes
    value_template: Optional[str] = None
    payload_on: str = "ON"
    payload_off: str = "OFF"
    device_class: Optional[str] = None
    off_delay: Optional[int] = None


@dataclass
class HAButtonEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant button entities."""

    entity_type: HAEntityType = HAEntityType.BUTTON

    # Button-specific attributes
    command_topic: str = ""
    command_template: Optional[str] = None
    payload_press: Optional[str] = None
    device_class: Optional[str] = None
    retain: bool = False
    qos: int = 1


@dataclass
class HASwitchEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant switch entities."""

    entity_type: HAEntityType = HAEntityType.SWITCH

    # Switch-specific attributes
    command_topic: str
    state_on: str = "ON"
    state_off: str = "OFF"
    payload_on: str = "ON"
    payload_off: str = "OFF"
    value_template: Optional[str] = None
    state_value_template: Optional[str] = None
    optimistic: bool = False
    retain: bool = False
    qos: int = 1


@dataclass
class HANumberEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant number entities."""

    entity_type: HAEntityType = HAEntityType.NUMBER

    # Number-specific attributes
    command_topic: str
    command_template: Optional[str] = None
    min: float = 0
    max: float = 100
    step: float = 1
    mode: str = "auto"  # auto, box, slider
    unit_of_measurement: Optional[str] = None
    device_class: Optional[str] = None


@dataclass
class HASelectEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant select entities."""

    entity_type: HAEntityType = HAEntityType.SELECT

    # Select-specific attributes
    command_topic: str
    command_template: Optional[str] = None
    options: List[str] = field(default_factory=list)
    value_template: Optional[str] = None
    optimistic: bool = False


@dataclass
class HATextEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant text entities."""

    entity_type: HAEntityType = HAEntityType.TEXT

    # Text-specific attributes
    command_topic: str
    command_template: Optional[str] = None
    value_template: Optional[str] = None
    min: int = 0
    max: int = 255
    mode: str = "text"  # text, password
    pattern: Optional[str] = None


@dataclass
class HAImageEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant image entities."""

    entity_type: HAEntityType = HAEntityType.IMAGE

    # Image-specific attributes
    url_template: str
    content_type: str = "image/jpeg"
    verify_ssl: bool = True


@dataclass
class HADateTimeEntityConfig(HAEntityConfig):
    """Configuration for Home Assistant datetime entities."""

    entity_type: HAEntityType = HAEntityType.DATETIME

    # DateTime-specific attributes
    command_topic: str
    command_template: Optional[str] = None
    value_template: Optional[str] = None
    format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class HAServiceDefinition:
    """Home Assistant service definition."""

    service_name: str
    domain: str
    friendly_name: str
    description: str
    icon: str

    # Service fields
    fields: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # MQTT integration
    command_topic: str = ""
    command_template: Optional[str] = None
    response_topic: Optional[str] = None

    # Service metadata
    target_selector: Optional[Dict[str, Any]] = None
    supports_response: bool = False


class HAEntityDefinitions:
    """
    Comprehensive Home Assistant entity definitions and MQTT discovery enhancements.

    This class builds on the existing DiscoveryPublisher to create a complete
    ecosystem of HA entities with proper device classes, units, and service
    definitions for comprehensive system control and monitoring.
    """

    def __init__(
        self,
        discovery_publisher: DiscoveryPublisher,
        mqtt_config: MQTTConfig,
        rooms: Dict[str, RoomConfig],
        tracking_config: Optional[TrackingConfig] = None,
    ):
        """
        Initialize HA entity definitions system.

        Args:
            discovery_publisher: Existing discovery publisher to enhance
            mqtt_config: MQTT configuration
            rooms: Room configurations
            tracking_config: Optional tracking configuration
        """
        self.discovery_publisher = discovery_publisher
        self.mqtt_config = mqtt_config
        self.rooms = rooms
        self.tracking_config = tracking_config or TrackingConfig()

        # Entity registry
        self.entity_definitions: Dict[str, HAEntityConfig] = {}
        self.service_definitions: Dict[str, HAServiceDefinition] = {}

        # State management
        self.entity_states: Dict[str, Any] = {}
        self.entity_availability: Dict[str, bool] = {}

        # Statistics
        self.stats = {
            "entities_defined": 0,
            "services_defined": 0,
            "entities_published": 0,
            "services_published": 0,
            "last_update": None,
            "definition_errors": 0,
        }

        logger.info(f"Initialized HAEntityDefinitions for {len(rooms)} rooms")

    def define_all_entities(self) -> Dict[str, HAEntityConfig]:
        """
        Define all Home Assistant entities for the system.

        Returns:
            Dictionary mapping entity IDs to entity configurations
        """
        try:
            logger.info("Defining comprehensive HA entity ecosystem")

            # Clear existing definitions
            self.entity_definitions.clear()

            # Define room-specific entities
            for room_id, room_config in self.rooms.items():
                room_entities = self._define_room_entities(room_id, room_config)
                self.entity_definitions.update(room_entities)

            # Define system-wide entities
            system_entities = self._define_system_entities()
            self.entity_definitions.update(system_entities)

            # Define diagnostic entities
            diagnostic_entities = self._define_diagnostic_entities()
            self.entity_definitions.update(diagnostic_entities)

            # Define control entities
            control_entities = self._define_control_entities()
            self.entity_definitions.update(control_entities)

            # Update statistics
            self.stats["entities_defined"] = len(self.entity_definitions)
            self.stats["last_update"] = datetime.utcnow()

            logger.info(f"Defined {len(self.entity_definitions)} HA entities")
            return self.entity_definitions

        except Exception as e:
            self.stats["definition_errors"] += 1
            logger.error(f"Error defining HA entities: {e}")
            return {}

    def define_all_services(self) -> Dict[str, HAServiceDefinition]:
        """
        Define all Home Assistant services for system control.

        Returns:
            Dictionary mapping service names to service definitions
        """
        try:
            logger.info("Defining HA services for system control")

            # Clear existing service definitions
            self.service_definitions.clear()

            # Model management services
            model_services = self._define_model_services()
            self.service_definitions.update(model_services)

            # System control services
            system_services = self._define_system_services()
            self.service_definitions.update(system_services)

            # Diagnostic services
            diagnostic_services = self._define_diagnostic_services()
            self.service_definitions.update(diagnostic_services)

            # Room-specific services
            room_services = self._define_room_services()
            self.service_definitions.update(room_services)

            # Update statistics
            self.stats["services_defined"] = len(self.service_definitions)

            logger.info(f"Defined {len(self.service_definitions)} HA services")
            return self.service_definitions

        except Exception as e:
            self.stats["definition_errors"] += 1
            logger.error(f"Error defining HA services: {e}")
            return {}

    async def publish_all_entities(self) -> Dict[str, MQTTPublishResult]:
        """
        Publish all defined entities to Home Assistant via MQTT discovery.

        Returns:
            Dictionary mapping entity IDs to publish results
        """
        try:
            if not self.entity_definitions:
                logger.warning(
                    "No entities defined - calling define_all_entities first"
                )
                self.define_all_entities()

            results = {}

            # Publish entities by type for proper ordering
            entity_types = [
                HAEntityType.SENSOR,
                HAEntityType.BINARY_SENSOR,
                HAEntityType.BUTTON,
                HAEntityType.SWITCH,
                HAEntityType.NUMBER,
                HAEntityType.SELECT,
                HAEntityType.TEXT,
                HAEntityType.IMAGE,
                HAEntityType.DATETIME,
            ]

            for entity_type in entity_types:
                type_entities = {
                    entity_id: config
                    for entity_id, config in self.entity_definitions.items()
                    if config.entity_type == entity_type
                }

                if type_entities:
                    logger.info(
                        f"Publishing {len(type_entities)} {entity_type.value} entities"
                    )

                    for entity_id, config in type_entities.items():
                        result = await self._publish_entity_discovery(config)
                        results[entity_id] = result

                        if result.success:
                            self.stats["entities_published"] += 1

            successful = sum(1 for r in results.values() if r.success)
            logger.info(f"Published {successful}/{len(results)} entities successfully")

            return results

        except Exception as e:
            logger.error(f"Error publishing HA entities: {e}")
            return {}

    async def publish_all_services(self) -> Dict[str, MQTTPublishResult]:
        """
        Publish all defined services as HA button entities.

        Returns:
            Dictionary mapping service names to publish results
        """
        try:
            if not self.service_definitions:
                logger.warning(
                    "No services defined - calling define_all_services first"
                )
                self.define_all_services()

            results = {}

            for service_name, service_def in self.service_definitions.items():
                # Create button entity for service
                button_config = self._create_service_button_config(service_def)

                result = await self._publish_entity_discovery(button_config)
                results[service_name] = result

                if result.success:
                    self.stats["services_published"] += 1

            logger.info(f"Published {len(results)} service buttons")
            return results

        except Exception as e:
            logger.error(f"Error publishing HA services: {e}")
            return {}

    def get_entity_definition(self, entity_id: str) -> Optional[HAEntityConfig]:
        """Get entity definition by ID."""
        return self.entity_definitions.get(entity_id)

    def get_service_definition(
        self, service_name: str
    ) -> Optional[HAServiceDefinition]:
        """Get service definition by name."""
        return self.service_definitions.get(service_name)

    def get_entity_stats(self) -> Dict[str, Any]:
        """Get entity definition statistics."""
        return {
            **self.stats,
            "entity_types": {
                entity_type.value: len(
                    [
                        e
                        for e in self.entity_definitions.values()
                        if e.entity_type == entity_type
                    ]
                )
                for entity_type in HAEntityType
            },
            "entity_categories": {
                category: len(
                    [
                        e
                        for e in self.entity_definitions.values()
                        if e.entity_category == category
                    ]
                )
                for category in ["config", "diagnostic", "system", None]
            },
        }

    # Private methods - Entity definition creators

    def _define_room_entities(
        self, room_id: str, room_config: RoomConfig
    ) -> Dict[str, HAEntityConfig]:
        """Define entities specific to a room."""
        entities = {}
        room_name = room_config.name
        device_info = self.discovery_publisher.device_info

        # Main prediction sensor
        entities[f"{room_id}_prediction"] = HASensorEntityConfig(
            name=f"{room_name} Occupancy Prediction",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_prediction",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ value_json.transition_type }}",
            json_attributes_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            json_attributes_template="{{ value_json | tojson }}",
            icon="mdi:home-account",
            entity_category="diagnostic",
            expire_after=600,
            description=f"Predicted occupancy state transition for {room_name}",
        )

        # Next transition timestamp
        entities[f"{room_id}_next_transition"] = HASensorEntityConfig(
            name=f"{room_name} Next Transition",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_next_transition",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ value_json.predicted_time }}",
            device_class=HADeviceClass.TIMESTAMP.value,
            icon="mdi:clock-outline",
            expire_after=600,
            description=f"Timestamp of next predicted transition for {room_name}",
        )

        # Confidence percentage
        entities[f"{room_id}_confidence"] = HASensorEntityConfig(
            name=f"{room_name} Confidence",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_confidence",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ (value_json.confidence_score * 100) | round(1) }}",
            unit_of_measurement="%",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:percent",
            entity_category="diagnostic",
            expire_after=600,
            suggested_display_precision=1,
            description=f"Prediction confidence percentage for {room_name}",
        )

        # Time until next transition
        entities[f"{room_id}_time_until"] = HASensorEntityConfig(
            name=f"{room_name} Time Until",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_time_until",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ value_json.time_until_human }}",
            icon="mdi:timer-outline",
            expire_after=600,
            description=f"Human-readable time until next transition for {room_name}",
        )

        # Prediction reliability
        entities[f"{room_id}_reliability"] = HASensorEntityConfig(
            name=f"{room_name} Reliability",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_reliability",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ value_json.prediction_reliability }}",
            icon="mdi:check-circle-outline",
            entity_category="diagnostic",
            expire_after=600,
            description=f"Current prediction reliability status for {room_name}",
        )

        # Room occupancy binary sensor
        entities[f"{room_id}_occupied"] = HABinarySensorEntityConfig(
            name=f"{room_name} Currently Occupied",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_occupied",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/state",
            device=device_info,
            value_template="{{ 'ON' if value_json.currently_occupied else 'OFF' }}",
            device_class=HADeviceClass.CONNECTIVITY.value,
            icon="mdi:account-check",
            description=f"Current occupancy state for {room_name}",
        )

        # Room accuracy sensor
        entities[f"{room_id}_accuracy"] = HASensorEntityConfig(
            name=f"{room_name} Accuracy",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_accuracy",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/accuracy",
            device=device_info,
            value_template="{{ value_json.accuracy_percentage | round(1) }}",
            unit_of_measurement="%",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:target",
            entity_category="diagnostic",
            suggested_display_precision=1,
            description=f"Prediction accuracy percentage for {room_name}",
        )

        # Room motion detection
        entities[f"{room_id}_motion_detected"] = HABinarySensorEntityConfig(
            name=f"{room_name} Motion Detected",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_motion",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/motion",
            device=device_info,
            value_template="{{ 'ON' if value_json.motion_detected else 'OFF' }}",
            device_class=HADeviceClass.MOTION.value,
            icon="mdi:motion-sensor",
            off_delay=30,  # 30 second delay before turning off
            description=f"Motion detection status for {room_name}",
        )

        # Room occupancy confidence level
        entities[f"{room_id}_occupancy_confidence"] = HASensorEntityConfig(
            name=f"{room_name} Occupancy Confidence",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_occ_confidence",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/state",
            device=device_info,
            value_template="{{ (value_json.occupancy_confidence * 100) | round(1) }}",
            unit_of_measurement="%",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:gauge",
            entity_category="diagnostic",
            suggested_display_precision=1,
            description=f"Current occupancy detection confidence for {room_name}",
        )

        # Time since last occupancy change
        entities[f"{room_id}_time_since_change"] = HASensorEntityConfig(
            name=f"{room_name} Time Since Change",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_time_since",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/state",
            device=device_info,
            value_template="{{ value_json.minutes_since_last_change }}",
            unit_of_measurement="min",
            device_class=HADeviceClass.DURATION.value,
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:clock-outline",
            description=f"Minutes since last occupancy change in {room_name}",
        )

        # Room prediction model in use
        entities[f"{room_id}_model_used"] = HASensorEntityConfig(
            name=f"{room_name} Model Used",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_model",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ value_json.model_used }}",
            icon="mdi:brain",
            entity_category="diagnostic",
            description=f"Prediction model currently used for {room_name}",
        )

        # Alternative predictions count
        entities[f"{room_id}_alternatives_count"] = HASensorEntityConfig(
            name=f"{room_name} Alternative Predictions",
            unique_id=f"{self.mqtt_config.device_identifier}_{room_id}_alternatives",
            state_topic=f"{self.mqtt_config.topic_prefix}/{room_id}/prediction",
            device=device_info,
            value_template="{{ value_json.alternatives | length }}",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:format-list-numbered",
            entity_category="diagnostic",
            description=f"Number of alternative predictions available for {room_name}",
        )

        return entities

    def _define_system_entities(self) -> Dict[str, HAEntityConfig]:
        """Define system-wide entities."""
        entities = {}
        device_info = self.discovery_publisher.device_info

        # System status
        entities["system_status"] = HASensorEntityConfig(
            name="System Status",
            unique_id=f"{self.mqtt_config.device_identifier}_system_status",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ value_json.system_status }}",
            json_attributes_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            json_attributes_template="{{ value_json | tojson }}",
            icon="mdi:server",
            entity_category="diagnostic",
            description="Overall system operational status",
        )

        # System uptime
        entities["system_uptime"] = HASensorEntityConfig(
            name="System Uptime",
            unique_id=f"{self.mqtt_config.device_identifier}_uptime",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ value_json.uptime_seconds }}",
            unit_of_measurement="s",
            device_class=HADeviceClass.DURATION.value,
            state_class=HAStateClass.TOTAL.value,
            icon="mdi:clock-check-outline",
            entity_category="diagnostic",
            description="System uptime in seconds",
        )

        # Total predictions made
        entities["predictions_count"] = HASensorEntityConfig(
            name="Total Predictions",
            unique_id=f"{self.mqtt_config.device_identifier}_predictions_count",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ value_json.total_predictions_made }}",
            state_class=HAStateClass.TOTAL_INCREASING.value,
            icon="mdi:counter",
            entity_category="diagnostic",
            description="Total number of predictions made by the system",
        )

        # Average system accuracy
        entities["system_accuracy"] = HASensorEntityConfig(
            name="System Accuracy",
            unique_id=f"{self.mqtt_config.device_identifier}_accuracy",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ value_json.average_accuracy_percent | round(1) }}",
            unit_of_measurement="%",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:target",
            entity_category="diagnostic",
            suggested_display_precision=1,
            description="Average prediction accuracy across all rooms",
        )

        # Active alerts count
        entities["active_alerts"] = HASensorEntityConfig(
            name="Active Alerts",
            unique_id=f"{self.mqtt_config.device_identifier}_alerts",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ value_json.active_alerts }}",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:alert-circle-outline",
            entity_category="diagnostic",
            description="Number of active system alerts",
        )

        return entities

    def _define_diagnostic_entities(self) -> Dict[str, HAEntityConfig]:
        """Define diagnostic and monitoring entities."""
        entities = {}
        device_info = self.discovery_publisher.device_info

        # Database connection status
        entities["database_connected"] = HABinarySensorEntityConfig(
            name="Database Connected",
            unique_id=f"{self.mqtt_config.device_identifier}_database",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ 'ON' if value_json.database_connected else 'OFF' }}",
            device_class=HADeviceClass.CONNECTIVITY.value,
            icon="mdi:database",
            entity_category="diagnostic",
            description="Database connection status",
        )

        # MQTT connection status
        entities["mqtt_connected"] = HABinarySensorEntityConfig(
            name="MQTT Connected",
            unique_id=f"{self.mqtt_config.device_identifier}_mqtt",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ 'ON' if value_json.mqtt_connected else 'OFF' }}",
            device_class=HADeviceClass.CONNECTIVITY.value,
            icon="mdi:wifi",
            entity_category="diagnostic",
            description="MQTT broker connection status",
        )

        # Tracking system status
        entities["tracking_active"] = HABinarySensorEntityConfig(
            name="Tracking Active",
            unique_id=f"{self.mqtt_config.device_identifier}_tracking",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ 'ON' if value_json.tracking_active else 'OFF' }}",
            device_class=HADeviceClass.RUNNING.value,
            icon="mdi:chart-line",
            entity_category="diagnostic",
            description="Accuracy tracking system status",
        )

        # Model training status
        entities["model_training"] = HABinarySensorEntityConfig(
            name="Model Training",
            unique_id=f"{self.mqtt_config.device_identifier}_training",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/status",
            device=device_info,
            value_template="{{ 'ON' if value_json.model_training_active else 'OFF' }}",
            device_class=HADeviceClass.RUNNING.value,
            icon="mdi:brain",
            entity_category="diagnostic",
            description="Model training process status",
        )

        # Memory usage
        entities["memory_usage"] = HASensorEntityConfig(
            name="Memory Usage",
            unique_id=f"{self.mqtt_config.device_identifier}_memory",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ value_json.memory_usage_mb }}",
            unit_of_measurement="MB",
            device_class=HADeviceClass.DATA_SIZE.value,
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:memory",
            entity_category="diagnostic",
            description="System memory usage in megabytes",
        )

        # CPU usage
        entities["cpu_usage"] = HASensorEntityConfig(
            name="CPU Usage",
            unique_id=f"{self.mqtt_config.device_identifier}_cpu",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ value_json.cpu_usage_percent | round(1) }}",
            unit_of_measurement="%",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:speedometer",
            entity_category="diagnostic",
            suggested_display_precision=1,
            description="System CPU usage percentage",
        )

        # Disk usage
        entities["disk_usage"] = HASensorEntityConfig(
            name="Disk Usage",
            unique_id=f"{self.mqtt_config.device_identifier}_disk",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ value_json.disk_usage_percent | round(1) }}",
            unit_of_measurement="%",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:harddisk",
            entity_category="diagnostic",
            suggested_display_precision=1,
            description="System disk usage percentage",
        )

        # Network connectivity
        entities["network_status"] = HABinarySensorEntityConfig(
            name="Network Connected",
            unique_id=f"{self.mqtt_config.device_identifier}_network",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ 'ON' if value_json.network_connected else 'OFF' }}",
            device_class=HADeviceClass.CONNECTIVITY.value,
            icon="mdi:network",
            entity_category="diagnostic",
            description="Network connectivity status",
        )

        # Home Assistant connection status
        entities["ha_connection"] = HABinarySensorEntityConfig(
            name="Home Assistant Connected",
            unique_id=f"{self.mqtt_config.device_identifier}_ha_connection",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ 'ON' if value_json.ha_connected else 'OFF' }}",
            device_class=HADeviceClass.CONNECTIVITY.value,
            icon="mdi:home-assistant",
            entity_category="diagnostic",
            description="Home Assistant API connection status",
        )

        # System load average
        entities["load_average"] = HASensorEntityConfig(
            name="System Load",
            unique_id=f"{self.mqtt_config.device_identifier}_load",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ value_json.load_average_1min | round(2) }}",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:gauge",
            entity_category="diagnostic",
            suggested_display_precision=2,
            description="System load average (1 minute)",
        )

        # Process count
        entities["process_count"] = HASensorEntityConfig(
            name="Active Processes",
            unique_id=f"{self.mqtt_config.device_identifier}_processes",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/diagnostics",
            device=device_info,
            value_template="{{ value_json.process_count }}",
            state_class=HAStateClass.MEASUREMENT.value,
            icon="mdi:application-cog",
            entity_category="diagnostic",
            description="Number of active system processes",
        )

        return entities

    def _define_control_entities(self) -> Dict[str, HAEntityConfig]:
        """Define control and configuration entities."""
        entities = {}
        device_info = self.discovery_publisher.device_info

        # Prediction system enable/disable switch
        entities["prediction_enabled"] = HASwitchEntityConfig(
            name="Prediction System",
            unique_id=f"{self.mqtt_config.device_identifier}_prediction_switch",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/prediction_enable",
            device=device_info,
            value_template="{{ 'ON' if value_json.prediction_enabled else 'OFF' }}",
            payload_on="ON",
            payload_off="OFF",
            icon="mdi:toggle-switch",
            entity_category="config",
            description="Enable or disable the prediction system",
        )

        # MQTT publishing enable/disable
        entities["mqtt_publishing"] = HASwitchEntityConfig(
            name="MQTT Publishing",
            unique_id=f"{self.mqtt_config.device_identifier}_mqtt_switch",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/mqtt_enable",
            device=device_info,
            value_template="{{ 'ON' if value_json.mqtt_publishing_enabled else 'OFF' }}",
            payload_on="ON",
            payload_off="OFF",
            icon="mdi:publish",
            entity_category="config",
            description="Enable or disable MQTT publishing",
        )

        # Prediction interval configuration
        entities["prediction_interval"] = HANumberEntityConfig(
            name="Prediction Interval",
            unique_id=f"{self.mqtt_config.device_identifier}_interval",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/set_interval",
            device=device_info,
            command_template="{{ value }}",
            min=60,
            max=3600,
            step=60,
            mode="slider",
            unit_of_measurement="s",
            icon="mdi:timer",
            entity_category="config",
            description="Prediction generation interval in seconds",
        )

        # Log level selection
        entities["log_level"] = HASelectEntityConfig(
            name="Log Level",
            unique_id=f"{self.mqtt_config.device_identifier}_log_level",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/set_log_level",
            device=device_info,
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            value_template="{{ value_json.log_level }}",
            icon="mdi:file-document-outline",
            entity_category="config",
            description="System logging level",
        )

        # Accuracy threshold configuration
        entities["accuracy_threshold"] = HANumberEntityConfig(
            name="Accuracy Threshold",
            unique_id=f"{self.mqtt_config.device_identifier}_accuracy_threshold",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/set_accuracy_threshold",
            device=device_info,
            command_template="{{ value }}",
            min=1.0,
            max=60.0,
            step=0.5,
            mode="slider",
            unit_of_measurement="min",
            icon="mdi:target",
            entity_category="config",
            description="Prediction accuracy threshold in minutes",
        )

        # Feature engineering configuration
        entities["feature_lookback"] = HANumberEntityConfig(
            name="Feature Lookback Hours",
            unique_id=f"{self.mqtt_config.device_identifier}_feature_lookback",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/set_lookback_hours",
            device=device_info,
            command_template="{{ value }}",
            min=6,
            max=168,  # 1 week
            step=1,
            mode="box",
            unit_of_measurement="h",
            icon="mdi:clock-time-four",
            entity_category="config",
            description="Hours of historical data for feature engineering",
        )

        # Model selection
        entities["primary_model"] = HASelectEntityConfig(
            name="Primary Model",
            unique_id=f"{self.mqtt_config.device_identifier}_primary_model",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/set_primary_model",
            device=device_info,
            options=["ensemble", "lstm", "xgboost", "hmm"],
            value_template="{{ value_json.primary_model }}",
            icon="mdi:brain",
            entity_category="config",
            description="Primary model for predictions",
        )

        # Maintenance mode switch
        entities["maintenance_mode"] = HASwitchEntityConfig(
            name="Maintenance Mode",
            unique_id=f"{self.mqtt_config.device_identifier}_maintenance",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/maintenance_mode",
            device=device_info,
            value_template="{{ 'ON' if value_json.maintenance_mode else 'OFF' }}",
            payload_on="ON",
            payload_off="OFF",
            icon="mdi:wrench",
            entity_category="config",
            description="Enable maintenance mode (stops predictions)",
        )

        # Data collection configuration
        entities["data_collection"] = HASwitchEntityConfig(
            name="Data Collection",
            unique_id=f"{self.mqtt_config.device_identifier}_data_collection",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/config",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/data_collection",
            device=device_info,
            value_template="{{ 'ON' if value_json.data_collection_enabled else 'OFF' }}",
            payload_on="ON",
            payload_off="OFF",
            icon="mdi:database-plus",
            entity_category="config",
            description="Enable data collection from Home Assistant",
        )

        # Debug information text entity
        entities["debug_info"] = HATextEntityConfig(
            name="Debug Information",
            unique_id=f"{self.mqtt_config.device_identifier}_debug_info",
            state_topic=f"{self.mqtt_config.topic_prefix}/system/debug",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/set_debug_info",
            device=device_info,
            value_template="{{ value_json.debug_info }}",
            max=1000,
            mode="text",
            icon="mdi:bug",
            entity_category="diagnostic",
            description="System debug information and status messages",
        )

        return entities

    def _define_model_services(self) -> Dict[str, HAServiceDefinition]:
        """Define model management services."""
        services = {}

        # Manual model retraining
        services["retrain_model"] = HAServiceDefinition(
            service_name="retrain_model",
            domain="ha_ml_predictor",
            friendly_name="Retrain Model",
            description="Manually trigger model retraining for specific room or all rooms",
            icon="mdi:brain",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/retrain",
            command_template='{{ {"room_id": room_id, "force": force, "timestamp": now().isoformat()} | tojson }}',
            fields={
                "room_id": {
                    "description": "Room ID to retrain (leave empty for all rooms)",
                    "example": "living_room",
                    "selector": {"text": {}},
                },
                "force": {
                    "description": "Force retraining even if not needed",
                    "default": False,
                    "selector": {"boolean": {}},
                },
            },
            target_selector={"entity": {"domain": "sensor"}},
            supports_response=True,
        )

        # Model validation
        services["validate_model"] = HAServiceDefinition(
            service_name="validate_model",
            domain="ha_ml_predictor",
            friendly_name="Validate Model",
            description="Validate model performance and accuracy",
            icon="mdi:check-circle",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/validate",
            fields={
                "room_id": {
                    "description": "Room ID to validate (leave empty for all rooms)",
                    "example": "living_room",
                    "selector": {"text": {}},
                },
                "days": {
                    "description": "Number of days to validate against",
                    "default": 7,
                    "selector": {"number": {"min": 1, "max": 30, "mode": "box"}},
                },
            },
        )

        # Export model
        services["export_model"] = HAServiceDefinition(
            service_name="export_model",
            domain="ha_ml_predictor",
            friendly_name="Export Model",
            description="Export trained model for backup or analysis",
            icon="mdi:export",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/export_model",
            fields={
                "room_id": {
                    "description": "Room ID to export (leave empty for all rooms)",
                    "example": "living_room",
                    "selector": {"text": {}},
                },
                "format": {
                    "description": "Export format",
                    "default": "pickle",
                    "selector": {"select": {"options": ["pickle", "joblib", "onnx"]}},
                },
            },
            supports_response=True,
        )

        # Import model
        services["import_model"] = HAServiceDefinition(
            service_name="import_model",
            domain="ha_ml_predictor",
            friendly_name="Import Model",
            description="Import pre-trained model",
            icon="mdi:import",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/import_model",
            fields={
                "room_id": {
                    "description": "Room ID to import for",
                    "required": True,
                    "example": "living_room",
                    "selector": {"text": {}},
                },
                "model_path": {
                    "description": "Path to model file",
                    "required": True,
                    "selector": {"text": {}},
                },
            },
            supports_response=True,
        )

        return services

    def _define_system_services(self) -> Dict[str, HAServiceDefinition]:
        """Define system control services."""
        services = {}

        # System restart
        services["restart_system"] = HAServiceDefinition(
            service_name="restart_system",
            domain="ha_ml_predictor",
            friendly_name="Restart System",
            description="Restart the occupancy prediction system",
            icon="mdi:restart",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/restart",
            command_template='{{ {"action": "restart", "timestamp": now().isoformat()} | tojson }}',
        )

        # Refresh discovery
        services["refresh_discovery"] = HAServiceDefinition(
            service_name="refresh_discovery",
            domain="ha_ml_predictor",
            friendly_name="Refresh Discovery",
            description="Refresh Home Assistant MQTT discovery messages",
            icon="mdi:refresh",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/refresh_discovery",
            command_template='{{ {"action": "refresh_discovery", "timestamp": now().isoformat()} | tojson }}',
        )

        # Reset statistics
        services["reset_statistics"] = HAServiceDefinition(
            service_name="reset_statistics",
            domain="ha_ml_predictor",
            friendly_name="Reset Statistics",
            description="Reset system statistics and counters",
            icon="mdi:counter",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/reset_stats",
            fields={
                "confirm": {
                    "description": "Confirm reset operation",
                    "default": False,
                    "selector": {"boolean": {}},
                }
            },
        )

        # Update configuration
        services["update_config"] = HAServiceDefinition(
            service_name="update_config",
            domain="ha_ml_predictor",
            friendly_name="Update Configuration",
            description="Update system configuration parameters",
            icon="mdi:cog",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/update_config",
            fields={
                "config_section": {
                    "description": "Configuration section to update",
                    "required": True,
                    "selector": {
                        "select": {
                            "options": ["prediction", "features", "logging", "mqtt"]
                        }
                    },
                },
                "config_data": {
                    "description": "Configuration data (JSON format)",
                    "required": True,
                    "selector": {"text": {"multiline": True}},
                },
            },
            supports_response=True,
        )

        # Backup system
        services["backup_system"] = HAServiceDefinition(
            service_name="backup_system",
            domain="ha_ml_predictor",
            friendly_name="Backup System",
            description="Create system backup including models and configuration",
            icon="mdi:backup-restore",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/backup_system",
            fields={
                "include_models": {
                    "description": "Include trained models in backup",
                    "default": True,
                    "selector": {"boolean": {}},
                },
                "include_data": {
                    "description": "Include historical data in backup",
                    "default": False,
                    "selector": {"boolean": {}},
                },
            },
            supports_response=True,
        )

        # Restore system
        services["restore_system"] = HAServiceDefinition(
            service_name="restore_system",
            domain="ha_ml_predictor",
            friendly_name="Restore System",
            description="Restore system from backup",
            icon="mdi:restore",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/restore_system",
            fields={
                "backup_path": {
                    "description": "Path to backup file",
                    "required": True,
                    "selector": {"text": {}},
                },
                "restore_models": {
                    "description": "Restore trained models",
                    "default": True,
                    "selector": {"boolean": {}},
                },
                "restore_config": {
                    "description": "Restore configuration",
                    "default": True,
                    "selector": {"boolean": {}},
                },
            },
            supports_response=True,
        )

        return services

    def _define_diagnostic_services(self) -> Dict[str, HAServiceDefinition]:
        """Define diagnostic and monitoring services."""
        services = {}

        # Generate diagnostic report
        services["generate_diagnostic"] = HAServiceDefinition(
            service_name="generate_diagnostic",
            domain="ha_ml_predictor",
            friendly_name="Generate Diagnostic Report",
            description="Generate comprehensive diagnostic report",
            icon="mdi:file-chart",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/diagnostic",
            fields={
                "include_logs": {
                    "description": "Include recent log entries",
                    "default": True,
                    "selector": {"boolean": {}},
                },
                "include_metrics": {
                    "description": "Include performance metrics",
                    "default": True,
                    "selector": {"boolean": {}},
                },
            },
            supports_response=True,
        )

        # Database health check
        services["check_database"] = HAServiceDefinition(
            service_name="check_database",
            domain="ha_ml_predictor",
            friendly_name="Check Database Health",
            description="Perform database health check and optimization",
            icon="mdi:database-check",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/db_check",
            supports_response=True,
        )

        return services

    def _define_room_services(self) -> Dict[str, HAServiceDefinition]:
        """Define room-specific services."""
        services = {}

        # Force prediction for specific room
        services["force_prediction"] = HAServiceDefinition(
            service_name="force_prediction",
            domain="ha_ml_predictor",
            friendly_name="Force Prediction",
            description="Force immediate prediction generation for specific room",
            icon="mdi:play-circle",
            command_topic=f"{self.mqtt_config.topic_prefix}/commands/force_prediction",
            fields={
                "room_id": {
                    "description": "Room ID to generate prediction for",
                    "required": True,
                    "example": "living_room",
                    "selector": {"text": {}},
                }
            },
            target_selector={"entity": {"domain": "sensor"}},
            supports_response=True,
        )

        return services

    def _create_service_button_config(
        self, service_def: HAServiceDefinition
    ) -> HAButtonEntityConfig:
        """Create button entity configuration for a service."""
        return HAButtonEntityConfig(
            name=service_def.friendly_name,
            unique_id=f"{self.mqtt_config.device_identifier}_{service_def.service_name}",
            state_topic="",  # Buttons don't have state
            command_topic=service_def.command_topic,
            device=self.discovery_publisher.device_info,
            command_template=service_def.command_template,
            payload_press=json.dumps(
                {
                    "action": service_def.service_name,
                    "timestamp": "{{ now().isoformat() }}",
                }
            ),
            icon=service_def.icon,
            entity_category="config",
            description=service_def.description,
        )

    async def _publish_entity_discovery(
        self, config: HAEntityConfig
    ) -> MQTTPublishResult:
        """Publish entity discovery message based on entity type."""
        try:
            # Create discovery topic
            discovery_topic = (
                f"{self.mqtt_config.discovery_prefix}/"
                f"{config.entity_type.value}/"
                f"{self.mqtt_config.device_identifier}/"
                f"{config.unique_id}/config"
            )

            # Create base discovery payload
            discovery_payload = {
                "name": config.name,
                "unique_id": config.unique_id,
                "device": {
                    "identifiers": config.device.identifiers,
                    "name": config.device.name,
                    "manufacturer": config.device.manufacturer,
                    "model": config.device.model,
                    "sw_version": config.device.sw_version,
                },
            }

            # Add common attributes
            if config.icon:
                discovery_payload["icon"] = config.icon
            if config.entity_category:
                discovery_payload["entity_category"] = config.entity_category
            if not config.enabled_by_default:
                discovery_payload["enabled_by_default"] = config.enabled_by_default
            if config.availability_topic:
                discovery_payload["availability_topic"] = config.availability_topic
            if config.availability_template:
                discovery_payload["availability_template"] = (
                    config.availability_template
                )
            if config.expire_after:
                discovery_payload["expire_after"] = config.expire_after

            # Add entity-type specific attributes
            if hasattr(config, "state_topic") and config.state_topic:
                discovery_payload["state_topic"] = config.state_topic

            # Add type-specific attributes
            if isinstance(config, HASensorEntityConfig):
                self._add_sensor_attributes(discovery_payload, config)
            elif isinstance(config, HABinarySensorEntityConfig):
                self._add_binary_sensor_attributes(discovery_payload, config)
            elif isinstance(config, HAButtonEntityConfig):
                self._add_button_attributes(discovery_payload, config)
            elif isinstance(config, HASwitchEntityConfig):
                self._add_switch_attributes(discovery_payload, config)
            elif isinstance(config, HANumberEntityConfig):
                self._add_number_attributes(discovery_payload, config)
            elif isinstance(config, HASelectEntityConfig):
                self._add_select_attributes(discovery_payload, config)
            elif isinstance(config, HATextEntityConfig):
                self._add_text_attributes(discovery_payload, config)
            elif isinstance(config, HAImageEntityConfig):
                self._add_image_attributes(discovery_payload, config)
            elif isinstance(config, HADateTimeEntityConfig):
                self._add_datetime_attributes(discovery_payload, config)

            # Add device availability if configured
            if self.discovery_publisher.device_info.availability_topic:
                discovery_payload["availability"] = {
                    "topic": self.discovery_publisher.device_info.availability_topic,
                    "payload_available": "online",
                    "payload_not_available": "offline",
                    "value_template": "{{ 'online' if value_json.status == 'online' else 'offline' }}",
                }

            # Publish discovery message
            result = await self.discovery_publisher.mqtt_publisher.publish_json(
                topic=discovery_topic, data=discovery_payload, qos=1, retain=True
            )

            if result.success:
                logger.debug(
                    f"Published {config.entity_type.value} discovery: {config.name}"
                )
            else:
                logger.error(
                    f"Failed to publish {config.entity_type.value} discovery: {result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"Error publishing entity discovery: {e}")
            return MQTTPublishResult(
                success=False,
                topic="",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e),
            )

    def _add_sensor_attributes(
        self, payload: Dict[str, Any], config: HASensorEntityConfig
    ) -> None:
        """Add sensor-specific attributes to discovery payload."""
        if config.value_template:
            payload["value_template"] = config.value_template
        if config.json_attributes_topic:
            payload["json_attributes_topic"] = config.json_attributes_topic
        if config.json_attributes_template:
            payload["json_attributes_template"] = config.json_attributes_template
        if config.unit_of_measurement:
            payload["unit_of_measurement"] = config.unit_of_measurement
        if config.device_class:
            payload["device_class"] = config.device_class
        if config.state_class:
            payload["state_class"] = config.state_class
        if config.suggested_display_precision is not None:
            payload["suggested_display_precision"] = config.suggested_display_precision
        if config.force_update:
            payload["force_update"] = config.force_update
        if config.last_reset_topic:
            payload["last_reset_topic"] = config.last_reset_topic
        if config.last_reset_value_template:
            payload["last_reset_value_template"] = config.last_reset_value_template

    def _add_binary_sensor_attributes(
        self, payload: Dict[str, Any], config: HABinarySensorEntityConfig
    ) -> None:
        """Add binary sensor-specific attributes to discovery payload."""
        if config.value_template:
            payload["value_template"] = config.value_template
        if config.payload_on != "ON":
            payload["payload_on"] = config.payload_on
        if config.payload_off != "OFF":
            payload["payload_off"] = config.payload_off
        if config.device_class:
            payload["device_class"] = config.device_class
        if config.off_delay:
            payload["off_delay"] = config.off_delay

    def _add_button_attributes(
        self, payload: Dict[str, Any], config: HAButtonEntityConfig
    ) -> None:
        """Add button-specific attributes to discovery payload."""
        payload["command_topic"] = config.command_topic
        if config.command_template:
            payload["command_template"] = config.command_template
        if config.payload_press:
            payload["payload_press"] = config.payload_press
        if config.device_class:
            payload["device_class"] = config.device_class
        if config.retain:
            payload["retain"] = config.retain
        if config.qos != 1:
            payload["qos"] = config.qos

    def _add_switch_attributes(
        self, payload: Dict[str, Any], config: HASwitchEntityConfig
    ) -> None:
        """Add switch-specific attributes to discovery payload."""
        payload["command_topic"] = config.command_topic
        if config.state_on != "ON":
            payload["state_on"] = config.state_on
        if config.state_off != "OFF":
            payload["state_off"] = config.state_off
        if config.payload_on != "ON":
            payload["payload_on"] = config.payload_on
        if config.payload_off != "OFF":
            payload["payload_off"] = config.payload_off
        if config.value_template:
            payload["value_template"] = config.value_template
        if config.state_value_template:
            payload["state_value_template"] = config.state_value_template
        if config.optimistic:
            payload["optimistic"] = config.optimistic
        if config.retain:
            payload["retain"] = config.retain
        if config.qos != 1:
            payload["qos"] = config.qos

    def _add_number_attributes(
        self, payload: Dict[str, Any], config: HANumberEntityConfig
    ) -> None:
        """Add number-specific attributes to discovery payload."""
        payload["command_topic"] = config.command_topic
        if config.command_template:
            payload["command_template"] = config.command_template
        payload["min"] = config.min
        payload["max"] = config.max
        payload["step"] = config.step
        if config.mode != "auto":
            payload["mode"] = config.mode
        if config.unit_of_measurement:
            payload["unit_of_measurement"] = config.unit_of_measurement
        if config.device_class:
            payload["device_class"] = config.device_class

    def _add_select_attributes(
        self, payload: Dict[str, Any], config: HASelectEntityConfig
    ) -> None:
        """Add select-specific attributes to discovery payload."""
        payload["command_topic"] = config.command_topic
        if config.command_template:
            payload["command_template"] = config.command_template
        payload["options"] = config.options
        if config.value_template:
            payload["value_template"] = config.value_template
        if config.optimistic:
            payload["optimistic"] = config.optimistic

    def _add_text_attributes(
        self, payload: Dict[str, Any], config: HATextEntityConfig
    ) -> None:
        """Add text-specific attributes to discovery payload."""
        payload["command_topic"] = config.command_topic
        if config.command_template:
            payload["command_template"] = config.command_template
        if config.value_template:
            payload["value_template"] = config.value_template
        payload["min"] = config.min
        payload["max"] = config.max
        if config.mode != "text":
            payload["mode"] = config.mode
        if config.pattern:
            payload["pattern"] = config.pattern

    def _add_image_attributes(
        self, payload: Dict[str, Any], config: HAImageEntityConfig
    ) -> None:
        """Add image-specific attributes to discovery payload."""
        payload["url_template"] = config.url_template
        if config.content_type != "image/jpeg":
            payload["content_type"] = config.content_type
        if not config.verify_ssl:
            payload["verify_ssl"] = config.verify_ssl

    def _add_datetime_attributes(
        self, payload: Dict[str, Any], config: HADateTimeEntityConfig
    ) -> None:
        """Add datetime-specific attributes to discovery payload."""
        payload["command_topic"] = config.command_topic
        if config.command_template:
            payload["command_template"] = config.command_template
        if config.value_template:
            payload["value_template"] = config.value_template
        if config.format != "%Y-%m-%d %H:%M:%S":
            payload["format"] = config.format


class HAEntityDefinitionsError(OccupancyPredictionError):
    """Raised when HA entity definition operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="HA_ENTITY_DEFINITIONS_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
