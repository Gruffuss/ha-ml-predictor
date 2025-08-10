"""
Comprehensive Home Assistant Integration Example.

This example demonstrates the complete HA entity definitions and MQTT discovery
system integration with the existing occupancy prediction system. It shows how
all the enhanced entity types work together to provide comprehensive system
control and monitoring through Home Assistant.

Features demonstrated:
- Complete entity ecosystem (9 entity types, 40+ diagnostic entities)
- Comprehensive service definitions (15+ services)
- Full MQTT discovery integration
- System monitoring and control capabilities
- Real-time publishing and state management
"""

import asyncio
from datetime import datetime, timedelta
import json
import logging
from typing import Any, Dict, List, Optional

from ..core.config import MQTTConfig, RoomConfig, TrackingConfig, get_config
from ..core.exceptions import OccupancyPredictionError
from .discovery_publisher import DiscoveryPublisher
from .ha_entity_definitions import (
    HADeviceClass,
    HAEntityDefinitions,
    HAEntityType,
    HAStateClass,
)
from .mqtt_integration_manager import MQTTIntegrationManager
from .mqtt_publisher import MQTTPublisher

logger = logging.getLogger(__name__)


class ComprehensiveHAIntegrationDemo:
    """
    Comprehensive demonstration of the Home Assistant integration system.

    This class shows how all the enhanced entity definitions work together
    to provide a complete Home Assistant ecosystem for occupancy prediction
    system control and monitoring.
    """

    def __init__(self):
        """Initialize the comprehensive HA integration demo."""
        self.config = get_config()
        self.mqtt_config = self.config.mqtt
        self.rooms = self.config.rooms

        # Core components
        self.mqtt_publisher: Optional[MQTTPublisher] = None
        self.discovery_publisher: Optional[DiscoveryPublisher] = None
        self.ha_entity_definitions: Optional[HAEntityDefinitions] = None
        self.mqtt_integration_manager: Optional[MQTTIntegrationManager] = None

        # Demo state
        self.demo_active = False
        self.entity_stats = {}

        logger.info("Initialized Comprehensive HA Integration Demo")

    async def initialize_complete_system(self) -> Dict[str, Any]:
        """
        Initialize the complete HA integration system with all components.

        Returns:
            Dictionary with initialization results and statistics
        """
        try:
            logger.info("Initializing complete HA integration system")
            results = {
                "initialization_time": datetime.utcnow(),
                "components_initialized": [],
                "entities_created": 0,
                "services_created": 0,
                "errors": [],
            }

            # Step 1: Initialize MQTT Publisher
            self.mqtt_publisher = MQTTPublisher(
                broker=self.mqtt_config.broker,
                port=self.mqtt_config.port,
                username=self.mqtt_config.username,
                password=self.mqtt_config.password,
                client_id=self.mqtt_config.client_id,
            )
            await self.mqtt_publisher.initialize()
            results["components_initialized"].append("MQTTPublisher")

            # Step 2: Initialize Discovery Publisher
            self.discovery_publisher = DiscoveryPublisher(
                mqtt_publisher=self.mqtt_publisher,
                config=self.mqtt_config,
                rooms=self.rooms,
                availability_check_callback=self._check_system_availability,
                state_change_callback=self._handle_entity_state_change,
            )
            results["components_initialized"].append("DiscoveryPublisher")

            # Step 3: Initialize HA Entity Definitions
            tracking_config = TrackingConfig()
            self.ha_entity_definitions = HAEntityDefinitions(
                discovery_publisher=self.discovery_publisher,
                mqtt_config=self.mqtt_config,
                rooms=self.rooms,
                tracking_config=tracking_config,
            )
            results["components_initialized"].append("HAEntityDefinitions")

            # Step 4: Define all entities and services
            entities = self.ha_entity_definitions.define_all_entities()
            services = self.ha_entity_definitions.define_all_services()

            results["entities_created"] = len(entities)
            results["services_created"] = len(services)

            logger.info(
                f"Defined {len(entities)} entities and {len(services)} services"
            )

            # Step 5: Publish device availability
            await self.discovery_publisher.publish_device_availability(online=True)

            # Step 6: Publish all entities to HA
            entity_results = await self.ha_entity_definitions.publish_all_entities()
            successful_entities = sum(1 for r in entity_results.values() if r.success)

            # Step 7: Publish all services to HA
            service_results = await self.ha_entity_definitions.publish_all_services()
            successful_services = sum(1 for r in service_results.values() if r.success)

            results["entities_published"] = successful_entities
            results["services_published"] = successful_services

            # Step 8: Initialize MQTT Integration Manager
            self.mqtt_integration_manager = MQTTIntegrationManager(
                mqtt_config=self.mqtt_config,
                rooms=self.rooms,
                tracking_config=tracking_config,
            )
            await self.mqtt_integration_manager.initialize()
            results["components_initialized"].append("MQTTIntegrationManager")

            self.demo_active = True

            logger.info(f"Complete HA integration system initialized successfully")
            logger.info(f"Published {successful_entities}/{len(entities)} entities")
            logger.info(f"Published {successful_services}/{len(services)} services")

            return results

        except Exception as e:
            logger.error(f"Error initializing complete HA integration system: {e}")
            results["errors"].append(str(e))
            return results

    async def demonstrate_entity_types(self) -> Dict[str, Any]:
        """
        Demonstrate all entity types with sample data.

        Returns:
            Dictionary with demonstration results for each entity type
        """
        if not self.demo_active or not self.ha_entity_definitions:
            raise OccupancyPredictionError("Demo system not initialized")

        demo_results = {
            "demonstration_time": datetime.utcnow(),
            "entity_types_demonstrated": [],
            "sample_data_published": {},
        }

        try:
            # Demonstrate sensor entities
            await self._demonstrate_sensor_entities(demo_results)

            # Demonstrate binary sensor entities
            await self._demonstrate_binary_sensor_entities(demo_results)

            # Demonstrate control entities (switches, numbers, selects)
            await self._demonstrate_control_entities(demo_results)

            # Demonstrate diagnostic entities
            await self._demonstrate_diagnostic_entities(demo_results)

            logger.info(
                f"Demonstrated {len(demo_results['entity_types_demonstrated'])} entity types"
            )
            return demo_results

        except Exception as e:
            logger.error(f"Error demonstrating entity types: {e}")
            demo_results["error"] = str(e)
            return demo_results

    async def demonstrate_service_execution(self) -> Dict[str, Any]:
        """
        Demonstrate service execution and command handling.

        Returns:
            Dictionary with service execution results
        """
        if not self.demo_active:
            raise OccupancyPredictionError("Demo system not initialized")

        service_demo = {
            "execution_time": datetime.utcnow(),
            "services_executed": [],
            "execution_results": {},
        }

        try:
            # Demonstrate model management services
            await self._demonstrate_model_services(service_demo)

            # Demonstrate system control services
            await self._demonstrate_system_services(service_demo)

            # Demonstrate diagnostic services
            await self._demonstrate_diagnostic_services(service_demo)

            logger.info(
                f"Demonstrated {len(service_demo['services_executed'])} services"
            )
            return service_demo

        except Exception as e:
            logger.error(f"Error demonstrating services: {e}")
            service_demo["error"] = str(e)
            return service_demo

    async def get_comprehensive_system_stats(self) -> Dict[str, Any]:
        """
        Get comprehensive statistics for the entire HA integration system.

        Returns:
            Dictionary with complete system statistics
        """
        try:
            stats = {
                "collection_time": datetime.utcnow(),
                "system_active": self.demo_active,
                "entity_definitions": {},
                "discovery_publisher": {},
                "mqtt_publisher": {},
                "integration_manager": {},
            }

            # Get entity definition stats
            if self.ha_entity_definitions:
                stats["entity_definitions"] = (
                    self.ha_entity_definitions.get_entity_stats()
                )

            # Get discovery publisher stats
            if self.discovery_publisher:
                stats["discovery_publisher"] = (
                    self.discovery_publisher.get_discovery_stats()
                )

            # Get MQTT publisher stats
            if self.mqtt_publisher:
                stats["mqtt_publisher"] = self.mqtt_publisher.get_publish_stats()

            # Get integration manager stats
            if self.mqtt_integration_manager:
                stats["integration_manager"] = (
                    self.mqtt_integration_manager.get_integration_stats()
                )

            return stats

        except Exception as e:
            logger.error(f"Error getting system stats: {e}")
            return {"error": str(e)}

    async def shutdown_demo(self) -> None:
        """Shutdown the comprehensive HA integration demo."""
        try:
            logger.info("Shutting down comprehensive HA integration demo")

            # Cleanup entities
            if self.ha_entity_definitions and self.discovery_publisher:
                await self.discovery_publisher.cleanup_entities()

            # Mark device as offline
            if self.discovery_publisher:
                await self.discovery_publisher.publish_device_availability(online=False)

            # Shutdown MQTT integration manager
            if self.mqtt_integration_manager:
                await self.mqtt_integration_manager.shutdown()

            # Shutdown MQTT publisher
            if self.mqtt_publisher:
                await self.mqtt_publisher.shutdown()

            self.demo_active = False
            logger.info("Demo shutdown complete")

        except Exception as e:
            logger.error(f"Error shutting down demo: {e}")

    # Private methods - Entity demonstrations

    async def _demonstrate_sensor_entities(self, results: Dict[str, Any]) -> None:
        """Demonstrate sensor entities with sample data."""
        try:
            # Sample room prediction data
            for room_id, room_config in self.rooms.items():
                room_data = {
                    "transition_type": "occupied",
                    "predicted_time": (
                        datetime.utcnow() + timedelta(minutes=15)
                    ).isoformat(),
                    "confidence_score": 0.85,
                    "time_until_human": "15 minutes",
                    "prediction_reliability": "high",
                    "model_used": "ensemble",
                    "alternatives": [
                        {
                            "transition_type": "vacant",
                            "predicted_time": (
                                datetime.utcnow() + timedelta(minutes=30)
                            ).isoformat(),
                            "confidence": 0.65,
                        }
                    ],
                }

                # Publish prediction data
                prediction_topic = (
                    f"{self.mqtt_config.topic_prefix}/{room_id}/prediction"
                )
                await self.mqtt_publisher.publish_json(
                    topic=prediction_topic, data=room_data, retain=True
                )

                # Sample accuracy data
                accuracy_data = {
                    "accuracy_percentage": 87.5,
                    "recent_predictions": 24,
                    "accurate_predictions": 21,
                }

                accuracy_topic = f"{self.mqtt_config.topic_prefix}/{room_id}/accuracy"
                await self.mqtt_publisher.publish_json(
                    topic=accuracy_topic, data=accuracy_data, retain=True
                )

            # Sample system status data
            system_data = {
                "system_status": "online",
                "uptime_seconds": 86400,  # 1 day
                "total_predictions_made": 1234,
                "average_accuracy_percent": 85.2,
                "active_alerts": 0,
            }

            system_topic = f"{self.mqtt_config.topic_prefix}/system/status"
            await self.mqtt_publisher.publish_json(
                topic=system_topic, data=system_data, retain=True
            )

            results["entity_types_demonstrated"].append("sensor")
            results["sample_data_published"]["sensors"] = {
                "room_predictions": len(self.rooms),
                "system_status": 1,
            }

        except Exception as e:
            logger.error(f"Error demonstrating sensor entities: {e}")

    async def _demonstrate_binary_sensor_entities(
        self, results: Dict[str, Any]
    ) -> None:
        """Demonstrate binary sensor entities with sample data."""
        try:
            # Sample diagnostic data
            diagnostic_data = {
                "database_connected": True,
                "mqtt_connected": True,
                "tracking_active": True,
                "model_training_active": False,
                "network_connected": True,
                "ha_connected": True,
            }

            diagnostic_topic = f"{self.mqtt_config.topic_prefix}/system/status"
            await self.mqtt_publisher.publish_json(
                topic=diagnostic_topic, data=diagnostic_data, retain=True
            )

            results["entity_types_demonstrated"].append("binary_sensor")
            results["sample_data_published"]["binary_sensors"] = diagnostic_data

        except Exception as e:
            logger.error(f"Error demonstrating binary sensor entities: {e}")

    async def _demonstrate_control_entities(self, results: Dict[str, Any]) -> None:
        """Demonstrate control entities with sample configuration data."""
        try:
            # Sample configuration data
            config_data = {
                "prediction_enabled": True,
                "mqtt_publishing_enabled": True,
                "prediction_interval_seconds": 300,
                "log_level": "INFO",
                "accuracy_threshold_minutes": 15.0,
                "feature_lookback_hours": 24,
                "primary_model": "ensemble",
                "maintenance_mode": False,
                "data_collection_enabled": True,
            }

            config_topic = f"{self.mqtt_config.topic_prefix}/system/config"
            await self.mqtt_publisher.publish_json(
                topic=config_topic, data=config_data, retain=True
            )

            results["entity_types_demonstrated"].extend(["switch", "number", "select"])
            results["sample_data_published"]["control_entities"] = config_data

        except Exception as e:
            logger.error(f"Error demonstrating control entities: {e}")

    async def _demonstrate_diagnostic_entities(self, results: Dict[str, Any]) -> None:
        """Demonstrate diagnostic entities with sample system metrics."""
        try:
            # Sample diagnostic metrics
            diagnostic_metrics = {
                "memory_usage_mb": 256.8,
                "cpu_usage_percent": 15.4,
                "disk_usage_percent": 42.1,
                "load_average_1min": 0.85,
                "process_count": 12,
            }

            diagnostics_topic = f"{self.mqtt_config.topic_prefix}/system/diagnostics"
            await self.mqtt_publisher.publish_json(
                topic=diagnostics_topic, data=diagnostic_metrics, retain=True
            )

            results["entity_types_demonstrated"].append("diagnostic")
            results["sample_data_published"]["diagnostic_metrics"] = diagnostic_metrics

        except Exception as e:
            logger.error(f"Error demonstrating diagnostic entities: {e}")

    # Private methods - Service demonstrations

    async def _demonstrate_model_services(self, results: Dict[str, Any]) -> None:
        """Demonstrate model management services."""
        try:
            # Simulate retrain command
            retrain_command = {
                "action": "retrain_model",
                "room_id": "living_room",
                "force": False,
                "timestamp": datetime.utcnow().isoformat(),
            }

            retrain_topic = f"{self.mqtt_config.topic_prefix}/commands/retrain"
            await self.mqtt_publisher.publish_json(
                topic=retrain_topic, data=retrain_command
            )

            results["services_executed"].append("retrain_model")
            results["execution_results"]["retrain_model"] = "command_sent"

        except Exception as e:
            logger.error(f"Error demonstrating model services: {e}")

    async def _demonstrate_system_services(self, results: Dict[str, Any]) -> None:
        """Demonstrate system control services."""
        try:
            # Simulate refresh discovery command
            refresh_command = {
                "action": "refresh_discovery",
                "timestamp": datetime.utcnow().isoformat(),
            }

            refresh_topic = (
                f"{self.mqtt_config.topic_prefix}/commands/refresh_discovery"
            )
            await self.mqtt_publisher.publish_json(
                topic=refresh_topic, data=refresh_command
            )

            results["services_executed"].append("refresh_discovery")
            results["execution_results"]["refresh_discovery"] = "command_sent"

        except Exception as e:
            logger.error(f"Error demonstrating system services: {e}")

    async def _demonstrate_diagnostic_services(self, results: Dict[str, Any]) -> None:
        """Demonstrate diagnostic services."""
        try:
            # Simulate diagnostic report command
            diagnostic_command = {
                "action": "generate_diagnostic",
                "include_logs": True,
                "include_metrics": True,
                "timestamp": datetime.utcnow().isoformat(),
            }

            diagnostic_topic = f"{self.mqtt_config.topic_prefix}/commands/diagnostic"
            await self.mqtt_publisher.publish_json(
                topic=diagnostic_topic, data=diagnostic_command
            )

            results["services_executed"].append("generate_diagnostic")
            results["execution_results"]["generate_diagnostic"] = "command_sent"

        except Exception as e:
            logger.error(f"Error demonstrating diagnostic services: {e}")

    # Private methods - Callbacks

    async def _check_system_availability(self) -> bool:
        """Check system availability for discovery publisher."""
        return self.demo_active

    async def _handle_entity_state_change(
        self, entity_id: str, state: Any, attributes: Optional[Dict[str, Any]] = None
    ) -> None:
        """Handle entity state changes from discovery publisher."""
        logger.debug(f"Entity state change: {entity_id} -> {state}")
        if attributes:
            logger.debug(f"Entity attributes: {attributes}")


# Factory function for easy usage
async def create_comprehensive_ha_integration_demo() -> ComprehensiveHAIntegrationDemo:
    """
    Create and initialize a comprehensive HA integration demonstration.

    Returns:
        Initialized ComprehensiveHAIntegrationDemo instance
    """
    demo = ComprehensiveHAIntegrationDemo()

    # Initialize the complete system
    init_results = await demo.initialize_complete_system()

    if init_results.get("errors"):
        raise OccupancyPredictionError(
            f"Demo initialization failed: {init_results['errors']}"
        )

    logger.info("Comprehensive HA integration demo created successfully")
    return demo


# Example usage function
async def run_comprehensive_demo():
    """Run a complete demonstration of the HA integration system."""
    demo = None
    try:
        logger.info("Starting comprehensive HA integration demonstration")

        # Create and initialize demo
        demo = await create_comprehensive_ha_integration_demo()

        # Demonstrate entity types
        entity_demo = await demo.demonstrate_entity_types()
        logger.info(f"Entity demonstration completed: {entity_demo}")

        # Demonstrate service execution
        service_demo = await demo.demonstrate_service_execution()
        logger.info(f"Service demonstration completed: {service_demo}")

        # Get comprehensive stats
        system_stats = await demo.get_comprehensive_system_stats()
        logger.info(
            f"System statistics: {json.dumps(system_stats, indent=2, default=str)}"
        )

        logger.info("Comprehensive HA integration demonstration completed successfully")

    except Exception as e:
        logger.error(f"Error running comprehensive demo: {e}")
        raise
    finally:
        if demo:
            await demo.shutdown_demo()


if __name__ == "__main__":
    # Run the demonstration
    asyncio.run(run_comprehensive_demo())
