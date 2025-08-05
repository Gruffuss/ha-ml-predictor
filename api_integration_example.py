"""
DEPRECATED: Example integration of REST API Server with TrackingManager.

⚠️ This file is now DEPRECATED. The API server is now fully integrated
into the main system and starts automatically.

✅ USE INSTEAD: src/main_system.py - Complete integrated system
✅ AUTOMATIC STARTUP: API server starts automatically with TrackingManager
✅ NO MANUAL SETUP: All components integrated seamlessly

This file remains for reference only.
"""

import asyncio
import logging
from src.core.config import get_config
from src.adaptation.tracking_manager import TrackingManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def run_integrated_system():
    """
    Example of running the complete integrated system with API server.
    
    This shows the automatic integration - no manual setup required!
    """
    try:
        # Load configuration
        config = get_config()
        logger.info("Configuration loaded successfully")
        
        # Create tracking manager (this would normally be created by main system)
        tracking_manager = TrackingManager(
            config=config.tracking,
            # In production, these would be injected:
            # database_manager=db_manager,
            # model_registry=model_registry,
            # feature_engineering_engine=feature_engine,
            # mqtt_integration_manager=mqtt_manager
        )
        
        # Start the API server automatically as part of tracking manager
        api_server = await tracking_manager.start_api_server()
        
        if api_server:
            logger.info("🚀 Integrated system started successfully!")
            logger.info(f"📊 API Server running at: http://{config.api.host}:{config.api.port}")
            logger.info(f"📋 API Documentation: http://{config.api.host}:{config.api.port}/docs")
            logger.info("🔧 Available endpoints:")
            logger.info("  GET  /health - System health check")
            logger.info("  GET  /predictions/{room_id} - Get room prediction")
            logger.info("  GET  /predictions - Get all predictions")
            logger.info("  GET  /accuracy?room_id={id}&hours={h} - Get accuracy metrics")
            logger.info("  POST /model/retrain - Trigger manual retraining")
            logger.info("  POST /mqtt/refresh - Refresh MQTT discovery")
            logger.info("  GET  /stats - Get system statistics")
            
            # Get API server status
            status = tracking_manager.get_api_server_status()
            logger.info(f"🔧 API Server Status: {status}")
            
            # Simulate running for a short time
            logger.info("⏳ System running... (press Ctrl+C to stop)")
            await asyncio.sleep(30)  # Run for 30 seconds in example
            
        else:
            logger.warning("API server failed to start or is disabled")
        
    except KeyboardInterrupt:
        logger.info("👋 Shutdown requested by user")
    except Exception as e:
        logger.error(f"❌ System error: {e}", exc_info=True)
    finally:
        # Clean shutdown
        if 'tracking_manager' in locals():
            await tracking_manager.stop_api_server()
            logger.info("✅ System shutdown complete")

if __name__ == "__main__":
    print("🏠 Home Assistant Occupancy Prediction System")
    print("🔌 Starting integrated system with REST API server...")
    print("📍 This demonstrates AUTOMATIC integration - no manual setup!")
    print()
    
    asyncio.run(run_integrated_system())