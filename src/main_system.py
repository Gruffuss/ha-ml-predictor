"""
Main system initialization for Home Assistant Occupancy Prediction System.

This module demonstrates the complete integrated system where API server
starts automatically without any manual setup. This is the proper integration
pattern that should be used in production.
"""

import asyncio
import logging
from typing import Optional

from .core.config import get_config
from .adaptation.tracking_manager import TrackingManager
from .data.storage.database import get_database_manager
from .integration.mqtt_integration_manager import MQTTIntegrationManager


logger = logging.getLogger(__name__)


class OccupancyPredictionSystem:
    """
    Main system orchestrator for occupancy prediction.
    
    This class demonstrates proper component integration where the API server
    starts automatically as part of the main system workflow.
    """
    
    def __init__(self):
        """Initialize the system with automatic component integration."""
        self.config = get_config()
        self.tracking_manager: Optional[TrackingManager] = None
        self.database_manager = None
        self.mqtt_manager: Optional[MQTTIntegrationManager] = None
        self.running = False
    
    async def initialize(self) -> None:
        """Initialize all system components with automatic API server startup."""
        try:
            logger.info("🚀 Starting Home Assistant Occupancy Prediction System...")
            
            # Initialize database
            self.database_manager = await get_database_manager()
            logger.info("✅ Database connection initialized")
            
            # Initialize MQTT integration
            self.mqtt_manager = MQTTIntegrationManager(self.config.mqtt)
            await self.mqtt_manager.initialize()
            logger.info("✅ MQTT integration initialized")
            
            # Initialize tracking manager with automatic API server integration
            self.tracking_manager = TrackingManager(
                config=self.config.tracking,
                database_manager=self.database_manager,
                mqtt_integration_manager=self.mqtt_manager,
                api_config=self.config.api  # This enables automatic API server startup
            )
            
            # Initialize tracking manager - this will automatically start API server
            await self.tracking_manager.initialize()
            
            # Check if API server started automatically
            api_status = self.tracking_manager.get_api_server_status()
            if api_status["enabled"] and api_status["running"]:
                logger.info(f"✅ API server automatically started at http://{api_status['host']}:{api_status['port']}")
                logger.info(f"📋 API Documentation: http://{api_status['host']}:{api_status['port']}/docs")
            elif api_status["enabled"]:
                logger.warning("⚠️ API server enabled but failed to start automatically")
            else:
                logger.info("ℹ️ API server disabled in configuration")
            
            self.running = True
            logger.info("🎉 System initialization complete - all components integrated automatically!")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize system: {e}", exc_info=True)
            await self.shutdown()
            raise
    
    async def run(self) -> None:
        """Run the main system loop."""
        if not self.running:
            await self.initialize()
        
        logger.info("🔄 System running - prediction tracking active")
        logger.info("💡 Key features automatically enabled:")
        logger.info("   - Real-time prediction tracking")
        logger.info("   - Automatic model retraining")
        logger.info("   - MQTT Home Assistant integration")
        
        api_status = self.tracking_manager.get_api_server_status()
        if api_status["running"]:
            logger.info("   - REST API server (no manual setup required)")
            logger.info(f"   - Available at: http://{api_status['host']}:{api_status['port']}")
        
        try:
            # Keep system running
            while self.running:
                await asyncio.sleep(10)
                
        except KeyboardInterrupt:
            logger.info("👋 Shutdown requested by user")
        except Exception as e:
            logger.error(f"💥 System error: {e}", exc_info=True)
        finally:
            await self.shutdown()
    
    async def shutdown(self) -> None:
        """Shutdown all system components gracefully."""
        logger.info("🛑 Shutting down system...")
        
        if self.tracking_manager:
            # This automatically stops the API server too
            await self.tracking_manager.stop_tracking()
            logger.info("✅ Tracking manager stopped (API server included)")
        
        if self.mqtt_manager:
            await self.mqtt_manager.cleanup()
            logger.info("✅ MQTT integration stopped")
        
        self.running = False
        logger.info("✅ System shutdown complete")


# Convenience function for easy system startup
async def run_occupancy_prediction_system():
    """
    Main entry point for the occupancy prediction system.
    
    This demonstrates the complete integrated system where ALL components,
    including the API server, start automatically without manual setup.
    """
    system = OccupancyPredictionSystem()
    await system.run()


if __name__ == "__main__":
    print("🏠 Home Assistant Occupancy Prediction System")
    print("🤖 Fully Integrated System with Automatic API Server")
    print("📡 No manual setup required - everything starts automatically!")
    print()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    asyncio.run(run_occupancy_prediction_system())