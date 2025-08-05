"""
Enhanced Home Assistant Discovery Integration Example.

This example demonstrates the advanced Home Assistant integration features
implemented in Sprint 5 Task 2, including device availability tracking,
service integration, entity lifecycle management, and comprehensive discovery.

Usage:
    python enhanced_ha_integration_example.py
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict

from ..core.config import get_config
from .discovery_publisher import EntityCategory, EntityState
from .mqtt_integration_manager import MQTTIntegrationManager

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class EnhancedHAIntegrationDemo:
    """
    Demonstration of enhanced Home Assistant integration features.

    Shows the advanced functionality implemented in Sprint 5 Task 2.
    """

    def __init__(self):
        """Initialize the demo with system configuration."""
        self.config = get_config()
        self.integration_manager: Optional[MQTTIntegrationManager] = None

    async def run_demo(self) -> None:
        """Run the complete enhanced HA integration demonstration."""
        try:
            logger.info("🚀 Starting Enhanced Home Assistant Integration Demo")

            # Step 1: Initialize enhanced integration manager
            await self._initialize_integration()

            # Step 2: Demonstrate device availability tracking
            await self._demo_device_availability()

            # Step 3: Demonstrate service integration
            await self._demo_service_integration()

            # Step 4: Demonstrate entity state management
            await self._demo_entity_state_management()

            # Step 5: Show comprehensive statistics
            await self._demo_enhanced_statistics()

            # Step 6: Demonstrate cleanup functionality
            await self._demo_cleanup_functionality()

            logger.info(
                "✅ Enhanced Home Assistant Integration Demo completed successfully!"
            )

        except Exception as e:
            logger.error(f"❌ Demo failed: {e}")
            raise
        finally:
            if self.integration_manager:
                await self.integration_manager.stop_integration()

    async def _initialize_integration(self) -> None:
        """Initialize the enhanced MQTT integration manager."""
        logger.info("📡 Initializing Enhanced MQTT Integration Manager...")

        self.integration_manager = MQTTIntegrationManager(
            mqtt_config=self.config.mqtt,
            rooms=self.config.rooms,
            notification_callbacks=[self._handle_integration_event],
        )

        await self.integration_manager.initialize()

        logger.info("✅ Enhanced integration manager initialized")

        # Verify initialization
        stats = self.integration_manager.get_integration_stats()
        logger.info(
            f"📊 Integration Status: {stats['system_health']['overall_status']}"
        )
        logger.info(f"📡 MQTT Connected: {stats['mqtt_connected']}")
        logger.info(f"🔍 Discovery Published: {stats['discovery_published']}")

    async def _demo_device_availability(self) -> None:
        """Demonstrate device availability tracking features."""
        logger.info("🟢 Demonstrating Device Availability Tracking...")

        # Publish device as online
        result = await self.integration_manager.update_device_availability(online=True)
        logger.info(f"✅ Device online status published: {result}")

        await asyncio.sleep(2)

        # Simulate device going offline
        result = await self.integration_manager.update_device_availability(online=False)
        logger.info(f"🔴 Device offline status published: {result}")

        await asyncio.sleep(2)

        # Bring device back online
        result = await self.integration_manager.update_device_availability(online=True)
        logger.info(f"✅ Device back online: {result}")

        # Show availability statistics
        if self.integration_manager.discovery_publisher:
            discovery_stats = (
                self.integration_manager.discovery_publisher.get_discovery_stats()
            )
            logger.info(
                f"📈 Availability Updates: {discovery_stats['statistics']['availability_updates']}"
            )

    async def _demo_service_integration(self) -> None:
        """Demonstrate Home Assistant service integration."""
        logger.info("🔧 Demonstrating Service Integration...")

        # Test manual retrain service
        retrain_result = await self.integration_manager.handle_service_command(
            "manual_retrain", {"room_id": "living_room", "strategy": "incremental"}
        )
        logger.info(f"🧠 Manual retrain service: {retrain_result}")

        # Test discovery refresh service
        refresh_result = await self.integration_manager.handle_service_command(
            "refresh_discovery", {"timestamp": datetime.utcnow().isoformat()}
        )
        logger.info(f"🔄 Discovery refresh service: {refresh_result}")

        # Test statistics reset service
        reset_result = await self.integration_manager.handle_service_command(
            "reset_statistics", {}
        )
        logger.info(f"📊 Statistics reset service: {reset_result}")

        # Test force prediction service
        prediction_result = await self.integration_manager.handle_service_command(
            "force_prediction", {"room_id": "bedroom"}
        )
        logger.info(f"🔮 Force prediction service: {prediction_result}")

        # Show service statistics
        if self.integration_manager.discovery_publisher:
            discovery_stats = (
                self.integration_manager.discovery_publisher.get_discovery_stats()
            )
            services = discovery_stats.get("available_services", [])
            logger.info(f"🔧 Available Services: {services}")

    async def _demo_entity_state_management(self) -> None:
        """Demonstrate entity state management capabilities."""
        logger.info("📊 Demonstrating Entity State Management...")

        if not self.integration_manager.discovery_publisher:
            logger.warning(
                "Discovery publisher not available for state management demo"
            )
            return

        discovery_publisher = self.integration_manager.discovery_publisher

        # Get published entities
        published_entities = list(discovery_publisher.published_entities.keys())
        if not published_entities:
            logger.warning("No published entities found for state management demo")
            return

        # Demonstrate state updates for first few entities
        demo_entities = published_entities[:3]

        for entity_id in demo_entities:
            # Update entity to online state
            result = await discovery_publisher.update_entity_state(
                entity_id=entity_id,
                state=EntityState.ONLINE,
                attributes={"last_demo_update": datetime.utcnow().isoformat()},
            )
            logger.info(f"✅ Updated {entity_id} to ONLINE: {result}")

            await asyncio.sleep(1)

            # Update entity to warning state
            result = await discovery_publisher.update_entity_state(
                entity_id=entity_id,
                state=EntityState.WARNING,
                attributes={"demo_warning": "Test warning state"},
            )
            logger.info(f"⚠️ Updated {entity_id} to WARNING: {result}")

        # Show entity metadata
        entity_metadata = discovery_publisher.entity_metadata
        logger.info(f"📊 Total entities with metadata: {len(entity_metadata)}")

        for entity_id, metadata in list(entity_metadata.items())[:2]:
            logger.info(
                f"📋 Entity {entity_id}: state={metadata.state.value}, last_updated={metadata.last_updated}"
            )

    async def _demo_enhanced_statistics(self) -> None:
        """Demonstrate comprehensive enhanced statistics."""
        logger.info("📈 Demonstrating Enhanced Statistics...")

        stats = self.integration_manager.get_integration_stats()

        # Show system health summary
        health = stats.get("system_health", {})
        logger.info(
            f"🏥 Overall System Status: {health.get('overall_status', 'unknown')}"
        )
        logger.info(f"⏱️ System Uptime: {health.get('uptime_hours', 0):.2f} hours")
        logger.info(f"📊 Error Rate: {health.get('error_rate', 0):.4f}")

        # Show component status
        components = health.get("component_status", {})
        for component, status in components.items():
            logger.info(f"🔧 {component.title()}: {status}")

        # Show discovery insights
        insights = stats.get("discovery_insights", {})
        logger.info(f"🔍 Entity Health: {insights.get('entity_health', 'unknown')}")
        logger.info(f"📡 Device Available: {insights.get('device_available', False)}")
        logger.info(
            f"🔧 Services Available: {insights.get('services_available', False)}"
        )
        logger.info(f"📊 Metadata Complete: {insights.get('metadata_complete', False)}")

        # Show discovery publisher specific stats
        if "discovery_publisher" in stats:
            discovery = stats["discovery_publisher"]
            logger.info(
                f"📋 Published Entities: {discovery.get('published_entities_count', 0)}"
            )
            logger.info(
                f"📊 Entity Metadata: {discovery.get('entity_metadata_count', 0)}"
            )
            logger.info(
                f"🔧 Available Services: {discovery.get('available_services_count', 0)}"
            )

            # Show device capabilities
            device_info = discovery.get("device_info", {})
            capabilities = device_info.get("capabilities", {})
            logger.info(f"🚀 Device Capabilities: {list(capabilities.keys())}")

    async def _demo_cleanup_functionality(self) -> None:
        """Demonstrate entity cleanup functionality."""
        logger.info("🧹 Demonstrating Cleanup Functionality...")

        if not self.integration_manager.discovery_publisher:
            logger.warning("Discovery publisher not available for cleanup demo")
            return

        discovery_publisher = self.integration_manager.discovery_publisher

        # Get current entity count
        initial_count = len(discovery_publisher.published_entities)
        logger.info(f"📊 Initial entity count: {initial_count}")

        # Get a few entities to clean up (for demo purposes)
        entities_to_cleanup = list(discovery_publisher.published_entities.keys())[:2]

        if entities_to_cleanup:
            logger.info(f"🧹 Cleaning up entities: {entities_to_cleanup}")

            result = await self.integration_manager.cleanup_discovery(
                entities_to_cleanup
            )
            logger.info(f"✅ Cleanup result: {result}")

            # Show updated count
            final_count = len(discovery_publisher.published_entities)
            logger.info(f"📊 Final entity count: {final_count}")
            logger.info(f"🗑️ Entities removed: {initial_count - final_count}")
        else:
            logger.info("ℹ️ No entities available for cleanup demo")

    async def _handle_integration_event(self, event: str) -> None:
        """Handle integration events for demonstration."""
        logger.info(f"🔔 Integration Event: {event}")


async def main():
    """Main function to run the enhanced HA integration demo."""
    demo = EnhancedHAIntegrationDemo()
    await demo.run_demo()


if __name__ == "__main__":
    asyncio.run(main())
