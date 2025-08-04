"""
Home Assistant MQTT Discovery Publisher.

This module handles the creation and publishing of Home Assistant MQTT discovery
messages for automatic device and entity registration in Home Assistant.

Features:
- Automatic sensor entity creation for predictions
- System status sensors and diagnostics
- Proper device registry integration
- Icon and unit assignment
- Discovery message management
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass

from ..core.config import MQTTConfig, RoomConfig
from .mqtt_publisher import MQTTPublisher, MQTTPublishResult


logger = logging.getLogger(__name__)


@dataclass
class DeviceInfo:
    """Home Assistant device information."""
    identifiers: List[str]
    name: str
    manufacturer: str
    model: str
    sw_version: str
    configuration_url: Optional[str] = None


@dataclass
class SensorConfig:
    """Home Assistant sensor configuration."""
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
    
    # Availability
    availability_topic: Optional[str] = None
    availability_template: Optional[str] = None


class DiscoveryPublisher:
    """
    Publishes Home Assistant MQTT discovery messages.
    
    Creates and manages MQTT discovery messages for automatic registration
    of occupancy prediction sensors and system status in Home Assistant.
    """
    
    def __init__(
        self,
        mqtt_publisher: MQTTPublisher,
        config: MQTTConfig,
        rooms: Dict[str, RoomConfig]
    ):
        """
        Initialize discovery publisher.
        
        Args:
            mqtt_publisher: Core MQTT publisher instance
            config: MQTT configuration
            rooms: Room configuration mapping
        """
        self.mqtt_publisher = mqtt_publisher
        self.config = config
        self.rooms = rooms
        
        # Device information
        self.device_info = DeviceInfo(
            identifiers=[self.config.device_identifier],
            name=self.config.device_name,
            manufacturer=self.config.device_manufacturer,
            model=self.config.device_model,
            sw_version=self.config.device_sw_version
        )
        
        # Discovery state
        self.discovery_published = False
        self.published_entities: Dict[str, str] = {}  # entity_id -> discovery_topic
        
        logger.info(f"Initialized DiscoveryPublisher for device: {self.config.device_name}")
    
    async def publish_all_discovery(self) -> Dict[str, MQTTPublishResult]:
        """
        Publish all discovery messages for the system.
        
        Returns:
            Dictionary mapping entity names to publish results
        """
        if not self.config.discovery_enabled:
            logger.info("MQTT discovery is disabled")
            return {}
        
        try:
            results = {}
            
            # Publish room prediction sensors
            for room_id, room_config in self.rooms.items():
                room_results = await self.publish_room_discovery(room_id, room_config)
                results.update(room_results)
            
            # Publish system sensors
            system_results = await self.publish_system_discovery()
            results.update(system_results)
            
            # Mark as published
            self.discovery_published = True
            
            successful = sum(1 for r in results.values() if r.success)
            total = len(results)
            
            logger.info(f"Published discovery for {successful}/{total} entities")
            return results
            
        except Exception as e:
            logger.error(f"Error publishing discovery messages: {e}")
            return {}
    
    async def publish_room_discovery(
        self, 
        room_id: str, 
        room_config: RoomConfig
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
            next_transition_sensor = self._create_next_transition_sensor(room_id, room_name)
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
            
            logger.info(f"Published discovery for room {room_name}: {len(results)} sensors")
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
                    topic=discovery_topic,
                    payload="",
                    qos=1,
                    retain=True
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
                    error_message="Entity not found"
                )
                
        except Exception as e:
            logger.error(f"Error removing discovery for {entity_name}: {e}")
            return MQTTPublishResult(
                success=False,
                topic="",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e)
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
        """Get discovery publisher statistics."""
        return {
            'discovery_enabled': self.config.discovery_enabled,
            'discovery_published': self.discovery_published,
            'published_entities_count': len(self.published_entities),
            'published_entities': list(self.published_entities.keys()),
            'rooms_configured': len(self.rooms),
            'device_info': {
                'name': self.device_info.name,
                'identifier': self.device_info.identifiers[0],
                'manufacturer': self.device_info.manufacturer,
                'model': self.device_info.model,
                'sw_version': self.device_info.sw_version
            }
        }
    
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
            expire_after=600  # 10 minutes
        )
    
    def _create_next_transition_sensor(self, room_id: str, room_name: str) -> SensorConfig:
        """Create next transition time sensor configuration."""
        return SensorConfig(
            name=f"{room_name} Next Transition",
            unique_id=f"{self.config.device_identifier}_{room_id}_next_transition",
            state_topic=f"{self.config.topic_prefix}/{room_id}/prediction",
            device=self.device_info,
            value_template="{{ value_json.predicted_time }}",
            device_class="timestamp",
            icon="mdi:clock-outline",
            expire_after=600
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
            expire_after=600
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
            expire_after=600
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
            expire_after=600
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
            entity_category="diagnostic"
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
            entity_category="diagnostic"
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
            entity_category="diagnostic"
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
            entity_category="diagnostic"
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
            entity_category="diagnostic"
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
            entity_category="diagnostic"
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
            entity_category="diagnostic"
        )
    
    # Private methods - Discovery publishing
    
    async def _publish_sensor_discovery(self, sensor_config: SensorConfig) -> MQTTPublishResult:
        """Publish discovery message for a sensor."""
        try:
            # Create discovery topic
            discovery_topic = (
                f"{self.config.discovery_prefix}/sensor/"
                f"{self.config.device_identifier}/"
                f"{sensor_config.unique_id}/config"
            )
            
            # Create discovery payload
            discovery_payload = {
                "name": sensor_config.name,
                "unique_id": sensor_config.unique_id,
                "state_topic": sensor_config.state_topic,
                "device": {
                    "identifiers": sensor_config.device.identifiers,
                    "name": sensor_config.device.name,
                    "manufacturer": sensor_config.device.manufacturer,
                    "model": sensor_config.device.model,
                    "sw_version": sensor_config.device.sw_version
                }
            }
            
            # Add optional attributes
            optional_attrs = [
                'json_attributes_topic', 'value_template', 'json_attributes_template',
                'unit_of_measurement', 'device_class', 'state_class', 'icon',
                'entity_category', 'enabled_by_default', 'expire_after',
                'availability_topic', 'availability_template'
            ]
            
            for attr in optional_attrs:
                value = getattr(sensor_config, attr, None)
                if value is not None:
                    discovery_payload[attr] = value
            
            # Publish discovery message
            result = await self.mqtt_publisher.publish_json(
                topic=discovery_topic,
                data=discovery_payload,
                qos=1,
                retain=True
            )
            
            if result.success:
                self.published_entities[sensor_config.unique_id] = discovery_topic
                logger.debug(f"Published discovery for sensor: {sensor_config.name}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error publishing sensor discovery: {e}")
            return MQTTPublishResult(
                success=False,
                topic="",
                payload_size=0,
                publish_time=datetime.utcnow(),
                error_message=str(e)
            )