"""
TrackingManager Integration for Real-time Publishing.

This module provides seamless integration between the TrackingManager and
the real-time publishing system, enabling automatic real-time prediction
broadcasting across multiple channels without manual configuration.

Features:
- Automatic TrackingManager integration
- Enhanced MQTT manager with real-time capabilities
- WebSocket and SSE server setup
- Transparent integration with existing prediction workflow
- No breaking changes to existing system
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass

from ..core.config import get_config, MQTTConfig, RoomConfig
from ..core.exceptions import OccupancyPredictionError, ErrorSeverity
from ..adaptation.tracking_manager import TrackingManager, TrackingConfig
from .enhanced_mqtt_manager import EnhancedMQTTIntegrationManager
from .realtime_publisher import PublishingChannel


logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for tracking integration with real-time publishing."""
    enable_realtime_publishing: bool = True
    enable_websocket_server: bool = True
    enable_sse_server: bool = True
    websocket_port: int = 8765
    sse_enabled_in_api: bool = True
    
    # Performance settings
    max_websocket_connections: int = 100
    max_sse_connections: int = 50
    connection_timeout_minutes: int = 60
    
    # Publishing settings
    publish_system_status_interval_seconds: int = 30
    broadcast_alerts: bool = True
    broadcast_drift_events: bool = True


class TrackingIntegrationManager:
    """
    Integration manager that connects TrackingManager with real-time publishing.
    
    This manager seamlessly enhances the existing tracking system with real-time
    publishing capabilities without requiring changes to the core tracking logic.
    """
    
    def __init__(
        self,
        tracking_manager: TrackingManager,
        integration_config: Optional[IntegrationConfig] = None
    ):
        """
        Initialize tracking integration manager.
        
        Args:
            tracking_manager: The existing TrackingManager instance
            integration_config: Configuration for real-time integration
        """
        self.tracking_manager = tracking_manager
        self.integration_config = integration_config or IntegrationConfig()
        
        # Get system configuration
        self.system_config = get_config()
        
        # Enhanced MQTT integration
        self.enhanced_mqtt_manager: Optional[EnhancedMQTTIntegrationManager] = None
        
        # Integration state
        self._integration_active = False
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        logger.info("Initialized TrackingIntegrationManager")
    
    async def initialize(self) -> None:
        """Initialize the tracking integration with real-time publishing."""
        try:
            if not self.integration_config.enable_realtime_publishing:
                logger.info("Real-time publishing disabled, skipping integration")
                return
            
            # Determine enabled channels
            enabled_channels = [PublishingChannel.MQTT]  # Always include MQTT
            
            if self.integration_config.enable_websocket_server:
                enabled_channels.append(PublishingChannel.WEBSOCKET)
            
            if self.integration_config.enable_sse_server:
                enabled_channels.append(PublishingChannel.SSE)
            
            # Initialize enhanced MQTT manager
            self.enhanced_mqtt_manager = EnhancedMQTTIntegrationManager(
                mqtt_config=self.system_config.mqtt,
                rooms=self.system_config.rooms,
                notification_callbacks=self.tracking_manager.notification_callbacks,
                enabled_realtime_channels=enabled_channels
            )
            
            # Initialize enhanced MQTT manager
            await self.enhanced_mqtt_manager.initialize()
            
            # Replace the tracking manager's MQTT integration
            self._integrate_with_tracking_manager()
            
            # Start background tasks
            await self._start_integration_tasks()
            
            self._integration_active = True
            logger.info("Tracking integration with real-time publishing initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize tracking integration: {e}")
            raise TrackingIntegrationError("Failed to initialize integration", cause=e)
    
    async def shutdown(self) -> None:
        """Shutdown the tracking integration."""
        try:
            self._shutdown_event.set()
            self._integration_active = False
            
            # Cancel background tasks
            if self._background_tasks:
                for task in self._background_tasks:
                    task.cancel()
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            # Shutdown enhanced MQTT manager
            if self.enhanced_mqtt_manager:
                await self.enhanced_mqtt_manager.shutdown()
            
            logger.info("Tracking integration shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during tracking integration shutdown: {e}")
    
    def get_websocket_handler(self):
        """Get WebSocket handler for external server integration."""
        if self.enhanced_mqtt_manager:
            return self.enhanced_mqtt_manager.handle_websocket_connection
        return None
    
    def get_sse_handler(self):
        """Get SSE handler for external server integration."""
        if self.enhanced_mqtt_manager:
            return self.enhanced_mqtt_manager.create_sse_stream
        return None
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        try:
            stats = {
                'integration_active': self._integration_active,
                'integration_config': {
                    'realtime_publishing_enabled': self.integration_config.enable_realtime_publishing,
                    'websocket_enabled': self.integration_config.enable_websocket_server,
                    'sse_enabled': self.integration_config.enable_sse_server,
                    'websocket_port': self.integration_config.websocket_port
                },
                'tracking_manager_stats': {},
                'enhanced_mqtt_stats': {},
                'performance': {
                    'background_tasks': len(self._background_tasks)
                }
            }
            
            # Get tracking manager stats
            if hasattr(self.tracking_manager, 'get_tracking_status'):
                stats['tracking_manager_stats'] = asyncio.create_task(
                    self.tracking_manager.get_tracking_status()
                )
            
            # Get enhanced MQTT stats
            if self.enhanced_mqtt_manager:
                stats['enhanced_mqtt_stats'] = self.enhanced_mqtt_manager.get_integration_stats()
                stats['connection_info'] = self.enhanced_mqtt_manager.get_connection_info()
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting integration stats: {e}")
            return {'error': str(e)}
    
    def add_realtime_callback(self, callback: Callable) -> None:
        """Add callback for real-time broadcast events."""
        if self.enhanced_mqtt_manager:
            self.enhanced_mqtt_manager.add_realtime_callback(callback)
    
    def remove_realtime_callback(self, callback: Callable) -> None:
        """Remove real-time broadcast callback."""
        if self.enhanced_mqtt_manager:
            self.enhanced_mqtt_manager.remove_realtime_callback(callback)
    
    # Private methods
    
    def _integrate_with_tracking_manager(self) -> None:
        """Integrate enhanced MQTT manager with tracking manager."""
        try:
            # Replace the MQTT integration manager in tracking manager
            self.tracking_manager.mqtt_integration_manager = self.enhanced_mqtt_manager
            
            # Add integration callbacks for alerts and drift events
            if self.integration_config.broadcast_alerts:
                self.tracking_manager.add_notification_callback(
                    self._handle_alert_broadcast
                )
            
            logger.debug("Successfully integrated with TrackingManager")
            
        except Exception as e:
            logger.error(f"Error integrating with TrackingManager: {e}")
            raise
    
    async def _start_integration_tasks(self) -> None:
        """Start background integration tasks."""
        try:
            # System status broadcasting task
            if self.integration_config.publish_system_status_interval_seconds > 0:
                status_task = asyncio.create_task(self._system_status_broadcast_loop())
                self._background_tasks.append(status_task)
            
            # Connection monitoring task
            monitor_task = asyncio.create_task(self._connection_monitoring_loop())
            self._background_tasks.append(monitor_task)
            
            logger.info(f"Started {len(self._background_tasks)} integration tasks")
            
        except Exception as e:
            logger.error(f"Failed to start integration tasks: {e}")
            raise
    
    async def _system_status_broadcast_loop(self) -> None:
        """Background loop for broadcasting system status."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Get comprehensive system status
                    tracking_status = await self.tracking_manager.get_tracking_status()
                    
                    # Publish status via enhanced MQTT manager
                    if self.enhanced_mqtt_manager:
                        await self.enhanced_mqtt_manager.publish_system_status(
                            tracking_stats=tracking_status,
                            database_connected=True,  # Would get from tracking manager
                            active_alerts=len(await self.tracking_manager.get_active_alerts())
                        )
                    
                    # Wait for next broadcast
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.integration_config.publish_system_status_interval_seconds
                    )
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in system status broadcast loop: {e}")
                    await asyncio.sleep(30)
                    
        except asyncio.CancelledError:
            logger.info("System status broadcast loop cancelled")
            raise
        except Exception as e:
            logger.error(f"System status broadcast loop failed: {e}")
    
    async def _connection_monitoring_loop(self) -> None:
        """Background loop for monitoring connections and performance."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    if self.enhanced_mqtt_manager:
                        # Get connection info
                        connection_info = self.enhanced_mqtt_manager.get_connection_info()
                        
                        # Log connection summary
                        total_connections = connection_info.get('total_active_connections', 0)
                        if total_connections > 0:
                            logger.debug(f"Real-time connections: {total_connections} active")
                        
                        # Check for connection limits
                        ws_connections = connection_info.get('websocket_connections', {}).get('total_active_connections', 0)
                        sse_connections = connection_info.get('sse_connections', {}).get('total_active_connections', 0)
                        
                        if ws_connections > self.integration_config.max_websocket_connections:
                            logger.warning(
                                f"WebSocket connections ({ws_connections}) exceeding limit "
                                f"({self.integration_config.max_websocket_connections})"
                            )
                        
                        if sse_connections > self.integration_config.max_sse_connections:
                            logger.warning(
                                f"SSE connections ({sse_connections}) exceeding limit "
                                f"({self.integration_config.max_sse_connections})"
                            )
                    
                    # Wait for next monitoring cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=60  # Monitor every minute
                    )
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in connection monitoring loop: {e}")
                    await asyncio.sleep(60)
                    
        except asyncio.CancelledError:
            logger.info("Connection monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Connection monitoring loop failed: {e}")
    
    def _handle_alert_broadcast(self, alert_message: Any) -> None:
        """Handle broadcasting of alerts via real-time channels."""
        try:
            if not self.enhanced_mqtt_manager or not self.integration_config.broadcast_alerts:
                return
            
            # Convert alert to broadcast format
            alert_data = {
                'type': 'alert',
                'message': str(alert_message),
                'timestamp': datetime.utcnow().isoformat(),
                'source': 'tracking_manager'
            }
            
            # Broadcast alert (fire and forget)
            asyncio.create_task(
                self.enhanced_mqtt_manager.publish_system_status({'alert': alert_data})
            )
            
        except Exception as e:
            logger.error(f"Error broadcasting alert: {e}")


# Integration function for easy setup
async def integrate_tracking_with_realtime_publishing(
    tracking_manager: TrackingManager,
    integration_config: Optional[IntegrationConfig] = None
) -> TrackingIntegrationManager:
    """
    Integrate TrackingManager with real-time publishing capabilities.
    
    This function provides a simple way to enhance an existing TrackingManager
    with real-time publishing across multiple channels.
    
    Args:
        tracking_manager: Existing TrackingManager instance
        integration_config: Optional integration configuration
        
    Returns:
        TrackingIntegrationManager instance
    """
    try:
        integration_manager = TrackingIntegrationManager(
            tracking_manager=tracking_manager,
            integration_config=integration_config
        )
        
        await integration_manager.initialize()
        
        logger.info("TrackingManager successfully integrated with real-time publishing")
        return integration_manager
        
    except Exception as e:
        logger.error(f"Failed to integrate TrackingManager with real-time publishing: {e}")
        raise


# Factory function for creating integrated tracking manager
async def create_integrated_tracking_manager(
    tracking_config: TrackingConfig,
    integration_config: Optional[IntegrationConfig] = None,
    **tracking_manager_kwargs
) -> tuple[TrackingManager, TrackingIntegrationManager]:
    """
    Create a TrackingManager with real-time publishing integration.
    
    This factory function creates both a TrackingManager and integrates it
    with real-time publishing capabilities in one step.
    
    Args:
        tracking_config: Configuration for TrackingManager
        integration_config: Configuration for real-time integration
        **tracking_manager_kwargs: Additional arguments for TrackingManager
        
    Returns:
        Tuple of (TrackingManager, TrackingIntegrationManager)
    """
    try:
        # Create TrackingManager
        tracking_manager = TrackingManager(
            config=tracking_config,
            **tracking_manager_kwargs
        )
        
        # Initialize TrackingManager
        await tracking_manager.initialize()
        
        # Create and initialize integration
        integration_manager = await integrate_tracking_with_realtime_publishing(
            tracking_manager=tracking_manager,
            integration_config=integration_config
        )
        
        logger.info("Created integrated TrackingManager with real-time publishing")
        return tracking_manager, integration_manager
        
    except Exception as e:
        logger.error(f"Failed to create integrated tracking manager: {e}")
        raise


class TrackingIntegrationError(OccupancyPredictionError):
    """Raised when tracking integration operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="TRACKING_INTEGRATION_ERROR",
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            **kwargs
        )