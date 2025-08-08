"""
Integration Module for Home Assistant Occupancy Prediction System.

This module provides comprehensive integration capabilities including MQTT publishing,
real-time WebSocket and Server-Sent Events streaming, and REST API endpoints.

Main Integration Components:
- Enhanced MQTT Integration Manager (with real-time capabilities)
- Real-time Publishing System (WebSocket + SSE)
- TrackingManager Integration (seamless workflow integration)
- REST API Server (with real-time endpoints)

Usage Examples:

1. Basic Integration with TrackingManager:
   ```python
   from src.integration import integrate_tracking_with_realtime_publishing
   from src.adaptation.tracking_manager import TrackingManager, TrackingConfig

   # Create tracking manager
   tracking_manager = TrackingManager(TrackingConfig())
   await tracking_manager.initialize()

   # Add real-time publishing capabilities
   integration_manager = await integrate_tracking_with_realtime_publishing(tracking_manager)

   # Now predictions are automatically published to MQTT, WebSocket, and SSE
   ```

2. Enhanced MQTT Manager (for manual setup):
   ```python
   from src.integration import EnhancedMQTTIntegrationManager
   from src.core.config import get_config

   config = get_config()
   enhanced_mqtt = EnhancedMQTTIntegrationManager(
       mqtt_config=config.mqtt,
       rooms=config.rooms
   )
   await enhanced_mqtt.initialize()

   # Publish across all channels
   await enhanced_mqtt.publish_prediction(prediction_result, room_id)
   ```

3. Real-time Publishing Only:
   ```python
   from src.integration import RealtimePublishingSystem, PublishingChannel

   realtime_publisher = RealtimePublishingSystem(
       enabled_channels=[PublishingChannel.WEBSOCKET, PublishingChannel.SSE]
   )
   await realtime_publisher.initialize()

   # Publish to real-time channels only
   await realtime_publisher.publish_prediction(prediction_result, room_id)
   ```

4. API Server with Real-time Endpoints:
   ```python
   from src.integration import realtime_api_endpoints
   from fastapi import FastAPI

   app = FastAPI()
   app.include_router(realtime_api_endpoints.realtime_router)

   # Real-time endpoints now available:
   # - WebSocket: /realtime/predictions
   # - SSE: /realtime/events
   # - Stats: /realtime/stats
   ```

Key Features:
- Automatic integration with existing prediction workflow
- Multi-channel broadcasting (MQTT, WebSocket, SSE)
- Connection management and performance monitoring
- Full backward compatibility with existing MQTT system
- Comprehensive error handling and recovery
- Production-ready performance and scalability
"""

from .api_server import APIServer, integrate_with_tracking_manager
from .discovery_publisher import DeviceInfo, DiscoveryPublisher, SensorConfig

# Import main integration components
from .enhanced_mqtt_manager import (
    EnhancedIntegrationStats,
    EnhancedMQTTIntegrationError,
    EnhancedMQTTIntegrationManager,
)
from .mqtt_integration_manager import (
    MQTTIntegrationManager,
    MQTTIntegrationStats,
)

# Import existing components for backward compatibility
from .mqtt_publisher import (
    MQTTConnectionStatus,
    MQTTPublisher,
    MQTTPublishResult,
)
from .prediction_publisher import (
    PredictionPayload,
    PredictionPublisher,
    SystemStatusPayload,
)
from .realtime_api_endpoints import (
    RealtimeStatsResponse,
    WebSocketConnectionHandler,
    WebSocketSubscription,
    get_integration_manager,
    realtime_router,
    set_integration_manager,
)
from .realtime_publisher import (
    ClientConnection,
    PublishingChannel,
    PublishingMetrics,
    RealtimePredictionEvent,
    RealtimePublishingError,
    RealtimePublishingSystem,
    SSEConnectionManager,
    WebSocketConnectionManager,
)
from .tracking_integration import (
    IntegrationConfig,
    TrackingIntegrationError,
    TrackingIntegrationManager,
    create_integrated_tracking_manager,
    integrate_tracking_with_realtime_publishing,
)

# Version information
__version__ = "1.0.0"
__author__ = "HA ML Predictor Team"

# Export main integration functions
__all__ = [
    # Enhanced Integration
    "EnhancedMQTTIntegrationManager",
    "EnhancedIntegrationStats",
    "EnhancedMQTTIntegrationError",
    # Real-time Publishing
    "RealtimePublishingSystem",
    "PublishingChannel",
    "RealtimePredictionEvent",
    "ClientConnection",
    "PublishingMetrics",
    "WebSocketConnectionManager",
    "SSEConnectionManager",
    "RealtimePublishingError",
    # TrackingManager Integration
    "TrackingIntegrationManager",
    "IntegrationConfig",
    "integrate_tracking_with_realtime_publishing",
    "create_integrated_tracking_manager",
    "TrackingIntegrationError",
    # API Endpoints
    "realtime_router",
    "set_integration_manager",
    "get_integration_manager",
    "WebSocketSubscription",
    "RealtimeStatsResponse",
    "WebSocketConnectionHandler",
    # Base Components (backward compatibility)
    "MQTTPublisher",
    "MQTTPublishResult",
    "MQTTConnectionStatus",
    "PredictionPublisher",
    "PredictionPayload",
    "SystemStatusPayload",
    "DiscoveryPublisher",
    "DeviceInfo",
    "SensorConfig",
    "MQTTIntegrationManager",
    "MQTTIntegrationStats",
    "APIServer",
    "integrate_with_tracking_manager",
]


# Convenience functions for common integration patterns


async def create_enhanced_system(
    tracking_config=None, integration_config=None, **kwargs
):
    """
    Create a complete enhanced system with real-time publishing.

    This is the simplest way to get a fully-featured system with
    TrackingManager, MQTT, WebSocket, SSE, and API server.

    Args:
        tracking_config: TrackingConfig instance
        integration_config: IntegrationConfig instance
        **kwargs: Additional arguments for TrackingManager

    Returns:
        Tuple of (TrackingManager, TrackingIntegrationManager, APIServer)
    """
    # Create integrated tracking manager
    tracking_manager, integration_manager = (
        await create_integrated_tracking_manager(
            tracking_config=tracking_config,
            integration_config=integration_config,
            **kwargs,
        )
    )

    # Create API server with real-time endpoints
    from .api_server import integrate_with_tracking_manager

    api_server = await integrate_with_tracking_manager(tracking_manager)

    # Set integration manager for real-time endpoints
    set_integration_manager(integration_manager)

    return tracking_manager, integration_manager, api_server


async def setup_realtime_only_system(
    mqtt_config=None, rooms=None, enabled_channels=None
):
    """
    Setup real-time publishing system without full TrackingManager integration.

    Useful for standalone real-time publishing or custom integrations.

    Args:
        mqtt_config: MQTT configuration
        rooms: Room configurations
        enabled_channels: List of PublishingChannel enums

    Returns:
        RealtimePublishingSystem instance
    """
    from .realtime_publisher import PublishingChannel

    if enabled_channels is None:
        enabled_channels = [PublishingChannel.WEBSOCKET, PublishingChannel.SSE]

    realtime_system = RealtimePublishingSystem(
        mqtt_config=mqtt_config, rooms=rooms, enabled_channels=enabled_channels
    )

    await realtime_system.initialize()
    return realtime_system


def get_integration_info():
    """Get information about available integration options."""
    return {
        "version": __version__,
        "components": {
            "enhanced_mqtt": "Complete MQTT + real-time publishing",
            "realtime_publisher": "WebSocket and SSE streaming",
            "tracking_integration": "Seamless TrackingManager integration",
            "api_endpoints": "FastAPI real-time endpoints",
            "base_mqtt": "Original MQTT publishing (backward compatibility)",
        },
        "channels": [channel.value for channel in PublishingChannel],
        "integration_patterns": [
            "full_enhanced_system",
            "tracking_manager_integration",
            "realtime_only",
            "mqtt_enhanced",
            "api_server_with_realtime",
        ],
        "features": [
            "Multi-channel broadcasting",
            "Connection management",
            "Performance monitoring",
            "Automatic failover",
            "Production-ready scaling",
            "Backward compatibility",
        ],
    }
