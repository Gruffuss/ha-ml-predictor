#!/usr/bin/env python3
"""
DEPRECATED: Example: Complete Real-time Prediction Publishing Integration

⚠️ This file is now DEPRECATED. Real-time prediction publishing is now fully integrated
into TrackingManager and starts automatically with the main system.

✅ USE INSTEAD: src/main_system.py - Complete integrated system
✅ AUTOMATIC OPERATION: Real-time publishing starts automatically with TrackingManager
✅ NO MANUAL SETUP: All channels (MQTT, WebSocket, SSE) start seamlessly

INTEGRATION STATUS:
- Real-time Publishing: ✅ Automatically initialized in TrackingManager
- MQTT Publishing: ✅ Enhanced MQTT Manager integrated by default
- WebSocket Server: ✅ Automatic startup when enabled in config
- SSE Server: ✅ Automatic startup when enabled in config
- API Endpoints: ✅ Real-time endpoints integrated in API server
- Multi-channel Broadcasting: ✅ Automatic across all channels

AUTOMATIC FEATURES NOW AVAILABLE:
- Multi-channel prediction broadcasting (MQTT, WebSocket, SSE)
- Real-time system monitoring and statistics
- Client connection management
- Performance tracking and metrics

This file remains for reference only.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any

# Core system imports
from src.core.config import get_config
from src.adaptation.tracking_manager import TrackingManager, TrackingConfig

# Integration imports
from src.integration import (
    integrate_tracking_with_realtime_publishing,
    IntegrationConfig,
    realtime_router,
    set_integration_manager,
    get_integration_info
)

# Model imports for example predictions
from src.models.base.predictor import PredictionResult
from src.core.constants import ModelType


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class RealtimeIntegrationExample:
    """
    Complete example of real-time prediction publishing integration.
    
    This class demonstrates how to set up and use the real-time publishing
    system with full TrackingManager integration.
    """
    
    def __init__(self):
        self.tracking_manager = None
        self.integration_manager = None
        self.api_server = None
        self.system_running = False
    
    async def setup_system(self):
        """Set up the complete integrated system."""
        try:
            logger.info("Setting up complete real-time prediction publishing system...")
            
            # Load system configuration
            config = get_config()
            
            # Create tracking configuration
            tracking_config = TrackingConfig(
                enabled=True,
                monitoring_interval_seconds=30,
                auto_validation_enabled=True,
                drift_detection_enabled=True,
                adaptive_retraining_enabled=True
            )
            
            # Create integration configuration
            integration_config = IntegrationConfig(
                enable_realtime_publishing=True,
                enable_websocket_server=True,
                enable_sse_server=True,
                websocket_port=8765,
                publish_system_status_interval_seconds=30,
                broadcast_alerts=True,
                broadcast_drift_events=True
            )
            
            # Create TrackingManager
            self.tracking_manager = TrackingManager(
                config=tracking_config,
                database_manager=None,  # Would be provided in real system
                model_registry={},
                feature_engineering_engine=None,
                notification_callbacks=[self._handle_system_notification]
            )
            
            # Initialize tracking manager
            await self.tracking_manager.initialize()
            
            # Integrate with real-time publishing
            self.integration_manager = await integrate_tracking_with_realtime_publishing(
                tracking_manager=self.tracking_manager,
                integration_config=integration_config
            )
            
            # Start API server with real-time endpoints
            await self._setup_api_server()
            
            logger.info("Real-time prediction publishing system setup complete!")
            logger.info(f"System info: {get_integration_info()}")
            
        except Exception as e:
            logger.error(f"Failed to setup system: {e}")
            raise
    
    async def _setup_api_server(self):
        """Set up API server with real-time endpoints."""
        try:
            from fastapi import FastAPI
            from src.integration.api_server import integrate_with_tracking_manager
            import uvicorn
            
            # Create API server integrated with tracking manager
            self.api_server = await integrate_with_tracking_manager(self.tracking_manager)
            
            # Set integration manager for real-time endpoints
            set_integration_manager(self.integration_manager)
            
            # Add real-time router to API server
            self.api_server.app.include_router(realtime_router)
            
            logger.info("API server with real-time endpoints configured")
            
        except Exception as e:
            logger.error(f"Failed to setup API server: {e}")
            raise
    
    async def start_system(self):
        """Start the complete system."""
        try:
            logger.info("Starting real-time prediction publishing system...")
            
            # Start API server
            if self.api_server:
                await self.api_server.start()
            
            # System is now running
            self.system_running = True
            
            logger.info("System started successfully!")
            logger.info("Available endpoints:")
            logger.info("  - REST API: http://localhost:8000")
            logger.info("  - WebSocket: ws://localhost:8000/realtime/predictions")
            logger.info("  - SSE: http://localhost:8000/realtime/events")
            logger.info("  - Stats: http://localhost:8000/realtime/stats")
            logger.info("  - Health: http://localhost:8000/realtime/health")
            
        except Exception as e:
            logger.error(f"Failed to start system: {e}")
            raise
    
    async def simulate_predictions(self):
        """Simulate prediction generation to demonstrate real-time publishing."""
        logger.info("Starting prediction simulation...")
        
        # Simulate predictions for different rooms
        rooms = ["living_room", "bedroom", "kitchen", "office"]
        
        while self.system_running:
            try:
                for room_id in rooms:
                    # Create example prediction
                    prediction_result = PredictionResult(
                        predicted_time=datetime.utcnow() + timedelta(minutes=30),
                        confidence_score=0.85,
                        model_type=ModelType.ENSEMBLE.value,
                        model_version="1.0.0",
                        transition_type="vacant_to_occupied",
                        features_used=["time_since_last_change", "hour_of_day", "day_of_week"],
                        alternatives=[
                            (datetime.utcnow() + timedelta(minutes=25), 0.75),
                            (datetime.utcnow() + timedelta(minutes=35), 0.65)
                        ],
                        prediction_metadata={
                            "room_id": room_id,
                            "base_model_predictions": {
                                "lstm": 0.82,
                                "xgboost": 0.88,
                                "hmm": 0.85
                            },
                            "model_weights": {
                                "lstm": 0.35,
                                "xgboost": 0.40,
                                "hmm": 0.25
                            }
                        }
                    )
                    
                    # Record prediction (triggers automatic real-time publishing)
                    await self.tracking_manager.record_prediction(prediction_result)
                    
                    logger.info(f"Generated prediction for {room_id}: {prediction_result.transition_type} in 30 minutes")
                    
                    # Wait between predictions
                    await asyncio.sleep(5)
                
                # Wait before next round
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in prediction simulation: {e}")
                await asyncio.sleep(10)
    
    async def monitor_system(self):
        """Monitor system performance and connections."""
        logger.info("Starting system monitoring...")
        
        while self.system_running:
            try:
                # Get integration statistics
                stats = self.integration_manager.get_integration_stats()
                
                # Log key metrics
                enhanced_stats = stats.get('enhanced_mqtt_stats', {})
                connection_info = enhanced_stats.get('connections', {})
                
                logger.info(f"System Status:")
                logger.info(f"  - Integration Active: {stats.get('integration_active', False)}")
                logger.info(f"  - Total Connections: {connection_info.get('total_clients', 0)}")
                logger.info(f"  - WebSocket Clients: {connection_info.get('websocket_clients', 0)}")
                logger.info(f"  - SSE Clients: {connection_info.get('sse_clients', 0)}")
                
                performance = enhanced_stats.get('performance', {})
                if performance:
                    logger.info(f"  - Predictions/min: {performance.get('predictions_per_minute', 0)}")
                    logger.info(f"  - Avg Latency: {performance.get('average_publish_latency_ms', 0):.2f}ms")
                    logger.info(f"  - Success Rate: {performance.get('publish_success_rate', 0):.2%}")
                
                # Check tracking manager status
                tracking_status = await self.tracking_manager.get_tracking_status()
                tracking_perf = tracking_status.get('performance', {})
                logger.info(f"  - Predictions Recorded: {tracking_perf.get('total_predictions_recorded', 0)}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(60)
                
            except Exception as e:
                logger.error(f"Error in system monitoring: {e}")
                await asyncio.sleep(30)
    
    async def shutdown_system(self):
        """Shutdown the system gracefully."""
        try:
            logger.info("Shutting down real-time prediction publishing system...")
            
            self.system_running = False
            
            # Shutdown integration manager
            if self.integration_manager:
                await self.integration_manager.shutdown()
            
            # Stop tracking manager
            if self.tracking_manager:
                await self.tracking_manager.stop_tracking()
            
            # Stop API server
            if self.api_server:
                await self.api_server.stop()
            
            logger.info("System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def _handle_system_notification(self, notification):
        """Handle system notifications (alerts, drift events, etc.)."""
        logger.info(f"System Notification: {notification}")
    
    async def run_complete_example(self):
        """Run the complete example."""
        try:
            # Setup system
            await self.setup_system()
            
            # Start system
            await self.start_system()
            
            # Start background tasks
            tasks = [
                asyncio.create_task(self.simulate_predictions()),
                asyncio.create_task(self.monitor_system())
            ]
            
            logger.info("Real-time prediction publishing system is now running!")
            logger.info("Press Ctrl+C to stop...")
            
            # Run until interrupted
            try:
                await asyncio.gather(*tasks)
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
            
        except Exception as e:
            logger.error(f"Error in complete example: {e}")
        finally:
            await self.shutdown_system()


async def demonstrate_client_connections():
    """
    Demonstrate how clients can connect to the real-time endpoints.
    
    This function shows example client code for connecting to WebSocket
    and SSE endpoints.
    """
    logger.info("Client Connection Examples:")
    
    # WebSocket client example
    websocket_example = '''
    # WebSocket Client Example
    import asyncio
    import websockets
    import json
    
    async def websocket_client():
        uri = "ws://localhost:8000/realtime/predictions"
        
        async with websockets.connect(uri) as websocket:
            # Subscribe to a specific room
            subscription = {
                "type": "subscribe",
                "room_id": "living_room"
            }
            await websocket.send(json.dumps(subscription))
            
            # Listen for predictions
            async for message in websocket:
                data = json.loads(message)
                print(f"Received prediction: {data}")
    
    asyncio.run(websocket_client())
    '''
    
    # SSE client example
    sse_example = '''
    # Server-Sent Events Client Example
    import requests
    import json
    
    def sse_client():
        url = "http://localhost:8000/realtime/events"
        
        with requests.get(url, stream=True) as response:
            for line in response.iter_lines():
                if line:
                    line = line.decode('utf-8')
                    if line.startswith('data: '):
                        data = json.loads(line[6:])
                        print(f"Received event: {data}")
    
    sse_client()
    '''
    
    logger.info("WebSocket Client Example:")
    logger.info(websocket_example)
    logger.info("SSE Client Example:")
    logger.info(sse_example)


def main():
    """Main entry point for the example."""
    print("=== Real-time Prediction Publishing Integration Example ===")
    print()
    print("This example demonstrates:")
    print("1. Complete system setup with TrackingManager integration")
    print("2. Automatic real-time publishing across multiple channels")
    print("3. WebSocket and SSE client endpoints")
    print("4. System monitoring and performance tracking")
    print()
    print("Starting example...")
    print()
    
    # Create and run example
    example = RealtimeIntegrationExample()
    
    try:
        # Run the complete example
        asyncio.run(example.run_complete_example())
    except KeyboardInterrupt:
        print("\nExample interrupted by user")
    except Exception as e:
        print(f"Example failed: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nExample completed")


if __name__ == "__main__":
    main()