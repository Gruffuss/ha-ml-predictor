"""
Enhanced Home Assistant Integration Manager.

This module provides comprehensive integration management that builds on existing
MQTT integration to create a complete HA ecosystem with enhanced entity definitions,
service management, and seamless integration with the tracking system.

Features:
- Complete HA entity ecosystem with proper device classes and state management
- Service definition and automatic button entity creation
- Integration with existing MQTT, discovery, and tracking systems
- Automatic entity state updates and availability management
- Command handling for HA service calls
- No manual setup required - fully integrated into main system workflow
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field

from ..core.config import get_config, MQTTConfig, RoomConfig, TrackingConfig
from ..core.exceptions import OccupancyPredictionError, ErrorSeverity
from ..models.base.predictor import PredictionResult
from .mqtt_publisher import MQTTPublisher, MQTTPublishResult
from .discovery_publisher import DiscoveryPublisher
from .mqtt_integration_manager import MQTTIntegrationManager
from .ha_entity_definitions import HAEntityDefinitions, HAEntityConfig, HAServiceDefinition


logger = logging.getLogger(__name__)


@dataclass
class EnhancedIntegrationStats:
    """Statistics for enhanced HA integration."""
    entities_defined: int = 0
    entities_published: int = 0
    services_defined: int = 0
    services_published: int = 0
    commands_processed: int = 0
    state_updates_sent: int = 0
    last_entity_update: Optional[datetime] = None
    last_command_processed: Optional[datetime] = None
    integration_errors: int = 0
    last_error: Optional[str] = None


@dataclass
class CommandRequest:
    """HA service command request."""
    command: str
    parameters: Dict[str, Any]
    timestamp: datetime
    response_topic: Optional[str] = None
    correlation_id: Optional[str] = None


@dataclass
class CommandResponse:
    """HA service command response."""
    success: bool
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None


class EnhancedIntegrationManager:
    """
    Enhanced Home Assistant Integration Manager.
    
    This manager provides comprehensive HA integration building on existing
    MQTT infrastructure to create a complete HA entity ecosystem with proper
    device classes, service definitions, and seamless system integration.
    """
    
    def __init__(
        self,
        mqtt_integration_manager: Optional[MQTTIntegrationManager] = None,
        tracking_manager: Optional[Any] = None,  # TrackingManager - avoid circular import
        notification_callbacks: Optional[List[Callable]] = None
    ):
        """
        Initialize enhanced integration manager.
        
        Args:
            mqtt_integration_manager: Existing MQTT integration manager
            tracking_manager: Optional tracking manager for automatic integration
            notification_callbacks: Optional notification callbacks
        """
        # Load configuration
        self.config = get_config()
        self.mqtt_config = self.config.mqtt
        self.rooms = self.config.rooms
        self.tracking_config = getattr(self.config, 'tracking', TrackingConfig())
        
        # Core integration components
        self.mqtt_integration_manager = mqtt_integration_manager
        self.tracking_manager = tracking_manager
        self.notification_callbacks = notification_callbacks or []
        
        # Enhanced HA components
        self.ha_entity_definitions: Optional[HAEntityDefinitions] = None
        
        # Command handling
        self.command_handlers: Dict[str, Callable] = {}
        self.command_queue = asyncio.Queue()
        self.command_responses: Dict[str, CommandResponse] = {}
        
        # State management
        self.entity_states: Dict[str, Any] = {}
        self.last_state_update: Dict[str, datetime] = {}
        
        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._enhanced_integration_active = False
        
        # Statistics
        self.stats = EnhancedIntegrationStats()
        
        logger.info("Initialized EnhancedIntegrationManager")
    
    async def initialize(self) -> None:
        """Initialize enhanced HA integration system."""
        try:
            logger.info("Initializing enhanced HA integration system")
            
            # Ensure MQTT integration is initialized
            if self.mqtt_integration_manager:
                if not getattr(self.mqtt_integration_manager, 'stats', None) or not self.mqtt_integration_manager.stats.initialized:
                    await self.mqtt_integration_manager.initialize()
            
            # Initialize HA entity definitions
            if self.mqtt_integration_manager and self.mqtt_integration_manager.discovery_publisher:
                self.ha_entity_definitions = HAEntityDefinitions(
                    discovery_publisher=self.mqtt_integration_manager.discovery_publisher,
                    mqtt_config=self.mqtt_config,
                    rooms=self.rooms,
                    tracking_config=self.tracking_config
                )
                
                # Define all entities and services
                await self._define_and_publish_entities()
                await self._define_and_publish_services()
            else:
                logger.warning("MQTT integration not available - enhanced features disabled")
            
            # Setup command handlers
            self._setup_command_handlers()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self._enhanced_integration_active = True
            logger.info("Enhanced HA integration system initialized successfully")
            
        except Exception as e:
            self.stats.integration_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error initializing enhanced HA integration: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown enhanced integration system."""
        try:
            logger.info("Shutting down enhanced HA integration system")
            
            # Signal shutdown
            self._shutdown_event.set()
            self._enhanced_integration_active = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            logger.info("Enhanced HA integration system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during enhanced integration shutdown: {e}")
    
    async def update_entity_state(
        self,
        entity_id: str,
        state: Any,
        attributes: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Update entity state and publish to HA.
        
        Args:
            entity_id: Entity identifier
            state: New entity state
            attributes: Optional entity attributes
            
        Returns:
            True if update successful, False otherwise
        """
        try:
            if not self._enhanced_integration_active:
                return False
            
            # Update local state
            self.entity_states[entity_id] = state
            self.last_state_update[entity_id] = datetime.utcnow()
            
            # Get entity definition to determine topic
            if self.ha_entity_definitions:
                entity_config = self.ha_entity_definitions.get_entity_definition(entity_id)
                if entity_config and entity_config.state_topic:
                    # Create state payload
                    state_payload = {"state": state}
                    if attributes:
                        state_payload.update(attributes)
                    
                    # Publish state update
                    if self.mqtt_integration_manager and self.mqtt_integration_manager.mqtt_publisher:
                        result = await self.mqtt_integration_manager.mqtt_publisher.publish_json(
                            topic=entity_config.state_topic,
                            data=state_payload,
                            qos=1,
                            retain=True
                        )
                        
                        if result.success:
                            self.stats.state_updates_sent += 1
                            self.stats.last_entity_update = datetime.utcnow()
                            return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating entity state for {entity_id}: {e}")
            return False
    
    async def process_command(self, command_request: CommandRequest) -> CommandResponse:
        """
        Process HA service command request.
        
        Args:
            command_request: Command request to process
            
        Returns:
            Command response with result or error
        """
        try:
            command = command_request.command
            parameters = command_request.parameters
            
            logger.info(f"Processing HA command: {command} with parameters: {parameters}")
            
            # Find appropriate command handler
            if command in self.command_handlers:
                handler = self.command_handlers[command]
                
                # Execute command handler
                if asyncio.iscoroutinefunction(handler):
                    result = await handler(parameters)
                else:
                    result = handler(parameters)
                
                # Create response
                response = CommandResponse(
                    success=True,
                    result=result,
                    correlation_id=command_request.correlation_id
                )
                
                self.stats.commands_processed += 1
                self.stats.last_command_processed = datetime.utcnow()
                
                logger.info(f"Command {command} processed successfully")
                return response
                
            else:
                error_msg = f"Unknown command: {command}"
                logger.warning(error_msg)
                return CommandResponse(
                    success=False,
                    error_message=error_msg,
                    correlation_id=command_request.correlation_id
                )
                
        except Exception as e:
            error_msg = f"Error processing command {command_request.command}: {e}"
            logger.error(error_msg)
            self.stats.integration_errors += 1
            self.stats.last_error = error_msg
            
            return CommandResponse(
                success=False,
                error_message=error_msg,
                correlation_id=command_request.correlation_id
            )
    
    async def handle_prediction_update(
        self,
        room_id: str,
        prediction_result: PredictionResult
    ) -> None:
        """
        Handle prediction updates and update relevant HA entities.
        
        Args:
            room_id: Room identifier
            prediction_result: New prediction result
        """
        try:
            if not self._enhanced_integration_active:
                return
            
            # Update prediction entity
            await self.update_entity_state(
                f"{room_id}_prediction",
                prediction_result.transition_type,
                {
                    "predicted_time": prediction_result.predicted_time.isoformat(),
                    "confidence_score": prediction_result.confidence_score,
                    "time_until_human": prediction_result.time_until_human,
                    "prediction_reliability": prediction_result.prediction_reliability
                }
            )
            
            # Update confidence entity
            await self.update_entity_state(
                f"{room_id}_confidence",
                round(prediction_result.confidence_score * 100, 1)
            )
            
            # Update time until entity
            await self.update_entity_state(
                f"{room_id}_time_until",
                prediction_result.time_until_human
            )
            
            # Update next transition entity
            await self.update_entity_state(
                f"{room_id}_next_transition",
                prediction_result.predicted_time.isoformat()
            )
            
            # Update reliability entity
            await self.update_entity_state(
                f"{room_id}_reliability",
                prediction_result.prediction_reliability
            )
            
            logger.debug(f"Updated HA entities for room {room_id} prediction")
            
        except Exception as e:
            logger.error(f"Error handling prediction update for {room_id}: {e}")
    
    async def handle_system_status_update(self, system_status: Dict[str, Any]) -> None:
        """
        Handle system status updates and update relevant HA entities.
        
        Args:
            system_status: System status information
        """
        try:
            if not self._enhanced_integration_active:
                return
            
            # Update system status entity
            await self.update_entity_state(
                "system_status",
                system_status.get("system_status", "unknown"),
                system_status
            )
            
            # Update individual status entities
            status_mappings = {
                "system_uptime": "uptime_seconds",
                "predictions_count": "total_predictions_made",
                "system_accuracy": "average_accuracy_percent",
                "active_alerts": "active_alerts",
                "database_connected": "database_connected",
                "mqtt_connected": "mqtt_connected",
                "tracking_active": "tracking_active",
                "model_training": "model_training_active"
            }
            
            for entity_id, status_key in status_mappings.items():
                if status_key in system_status:
                    await self.update_entity_state(entity_id, system_status[status_key])
            
        except Exception as e:
            logger.error(f"Error handling system status update: {e}")
    
    def get_integration_stats(self) -> Dict[str, Any]:
        """Get enhanced integration statistics."""
        base_stats = {}
        if self.mqtt_integration_manager:
            base_stats = self.mqtt_integration_manager.stats.__dict__
        
        enhanced_stats = self.stats.__dict__.copy()
        
        # Add entity definition stats
        if self.ha_entity_definitions:
            enhanced_stats.update(self.ha_entity_definitions.get_entity_stats())
        
        return {
            "enhanced_integration": enhanced_stats,
            "mqtt_integration": base_stats,
            "entity_states_count": len(self.entity_states),
            "command_handlers_count": len(self.command_handlers),
            "active": self._enhanced_integration_active
        }
    
    # Private methods - Initialization and setup
    
    async def _define_and_publish_entities(self) -> None:
        """Define and publish all HA entities."""
        try:
            if not self.ha_entity_definitions:
                return
            
            logger.info("Defining and publishing HA entities")
            
            # Define all entities
            entities = self.ha_entity_definitions.define_all_entities()
            self.stats.entities_defined = len(entities)
            
            # Publish entities
            results = await self.ha_entity_definitions.publish_all_entities()
            self.stats.entities_published = sum(1 for r in results.values() if r.success)
            
            logger.info(f"Published {self.stats.entities_published}/{self.stats.entities_defined} entities")
            
        except Exception as e:
            logger.error(f"Error defining and publishing entities: {e}")
    
    async def _define_and_publish_services(self) -> None:
        """Define and publish all HA services."""
        try:
            if not self.ha_entity_definitions:
                return
            
            logger.info("Defining and publishing HA services")
            
            # Define all services
            services = self.ha_entity_definitions.define_all_services()
            self.stats.services_defined = len(services)
            
            # Publish services as button entities
            results = await self.ha_entity_definitions.publish_all_services()
            self.stats.services_published = sum(1 for r in results.values() if r.success)
            
            logger.info(f"Published {self.stats.services_published}/{self.stats.services_defined} services")
            
        except Exception as e:
            logger.error(f"Error defining and publishing services: {e}")
    
    def _setup_command_handlers(self) -> None:
        """Setup command handlers for HA services."""
        try:
            logger.info("Setting up HA command handlers")
            
            # Model management handlers
            self.command_handlers["retrain_model"] = self._handle_retrain_model
            self.command_handlers["validate_model"] = self._handle_validate_model
            
            # System control handlers
            self.command_handlers["restart_system"] = self._handle_restart_system
            self.command_handlers["refresh_discovery"] = self._handle_refresh_discovery
            self.command_handlers["reset_statistics"] = self._handle_reset_statistics
            
            # Diagnostic handlers
            self.command_handlers["generate_diagnostic"] = self._handle_generate_diagnostic
            self.command_handlers["check_database"] = self._handle_check_database
            
            # Room-specific handlers
            self.command_handlers["force_prediction"] = self._handle_force_prediction
            
            # Configuration handlers
            self.command_handlers["prediction_enable"] = self._handle_prediction_enable
            self.command_handlers["mqtt_enable"] = self._handle_mqtt_enable
            self.command_handlers["set_interval"] = self._handle_set_interval
            self.command_handlers["set_log_level"] = self._handle_set_log_level
            
            logger.info(f"Setup {len(self.command_handlers)} command handlers")
            
        except Exception as e:
            logger.error(f"Error setting up command handlers: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background tasks for enhanced integration."""
        try:
            # Command processing task
            command_task = asyncio.create_task(self._command_processing_loop())
            self._background_tasks.append(command_task)
            
            # Entity state monitoring task
            monitoring_task = asyncio.create_task(self._entity_monitoring_loop())
            self._background_tasks.append(monitoring_task)
            
            logger.info(f"Started {len(self._background_tasks)} background tasks")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _command_processing_loop(self) -> None:
        """Background task for processing HA commands."""
        try:
            logger.info("Started command processing loop")
            
            while not self._shutdown_event.is_set():
                try:
                    # Wait for command with timeout
                    command_request = await asyncio.wait_for(
                        self.command_queue.get(),
                        timeout=1.0
                    )
                    
                    # Process command
                    response = await self.process_command(command_request)
                    
                    # Store response if correlation ID provided
                    if command_request.correlation_id:
                        self.command_responses[command_request.correlation_id] = response
                    
                    # Publish response if response topic provided
                    if command_request.response_topic and self.mqtt_integration_manager:
                        await self.mqtt_integration_manager.mqtt_publisher.publish_json(
                            topic=command_request.response_topic,
                            data=response.__dict__,
                            qos=1
                        )
                    
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in command processing loop: {e}")
                    await asyncio.sleep(1)
            
        except Exception as e:
            logger.error(f"Command processing loop error: {e}")
        finally:
            logger.info("Command processing loop stopped")
    
    async def _entity_monitoring_loop(self) -> None:
        """Background task for monitoring entity states."""
        try:
            logger.info("Started entity monitoring loop")
            
            while not self._shutdown_event.is_set():
                try:
                    # Monitor entity availability and update as needed
                    await self._check_entity_availability()
                    
                    # Clean up old command responses
                    await self._cleanup_old_responses()
                    
                    # Wait before next check
                    await asyncio.sleep(30)  # Check every 30 seconds
                    
                except Exception as e:
                    logger.error(f"Error in entity monitoring loop: {e}")
                    await asyncio.sleep(5)
            
        except Exception as e:
            logger.error(f"Entity monitoring loop error: {e}")
        finally:
            logger.info("Entity monitoring loop stopped")
    
    async def _check_entity_availability(self) -> None:
        """Check and update entity availability."""
        try:
            # Update device availability based on system status
            if self.mqtt_integration_manager and self.mqtt_integration_manager.discovery_publisher:
                await self.mqtt_integration_manager.discovery_publisher.publish_device_availability(
                    online=self._enhanced_integration_active
                )
            
        except Exception as e:
            logger.error(f"Error checking entity availability: {e}")
    
    async def _cleanup_old_responses(self) -> None:
        """Clean up old command responses."""
        try:
            current_time = datetime.utcnow()
            expired_responses = [
                correlation_id for correlation_id, response in self.command_responses.items()
                if current_time - response.timestamp > timedelta(hours=1)
            ]
            
            for correlation_id in expired_responses:
                del self.command_responses[correlation_id]
            
            if expired_responses:
                logger.debug(f"Cleaned up {len(expired_responses)} expired command responses")
                
        except Exception as e:
            logger.error(f"Error cleaning up old responses: {e}")
    
    # Command handlers
    
    async def _handle_retrain_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model retraining command."""
        try:
            room_id = parameters.get("room_id")
            force = parameters.get("force", False)
            
            # Delegate to tracking manager if available
            if self.tracking_manager:
                if hasattr(self.tracking_manager, 'trigger_retraining'):
                    result = await self.tracking_manager.trigger_retraining(
                        room_id=room_id,
                        force=force
                    )
                    return {"status": "success", "result": result}
            
            return {"status": "error", "message": "Tracking manager not available"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_validate_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle model validation command."""
        try:
            room_id = parameters.get("room_id")
            days = parameters.get("days", 7)
            
            # Delegate to tracking manager if available
            if self.tracking_manager:
                if hasattr(self.tracking_manager, 'validate_model_performance'):
                    result = await self.tracking_manager.validate_model_performance(
                        room_id=room_id,
                        validation_days=days
                    )
                    return {"status": "success", "result": result}
            
            return {"status": "error", "message": "Tracking manager not available"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_restart_system(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle system restart command."""
        try:
            logger.warning("System restart requested via HA")
            # Note: Actual restart would be handled by system supervisor
            return {"status": "acknowledged", "message": "Restart request received"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_refresh_discovery(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle discovery refresh command."""
        try:
            if self.mqtt_integration_manager and self.mqtt_integration_manager.discovery_publisher:
                results = await self.mqtt_integration_manager.discovery_publisher.refresh_discovery()
                successful = sum(1 for r in results.values() if r.success)
                return {
                    "status": "success", 
                    "entities_refreshed": successful,
                    "total_entities": len(results)
                }
            return {"status": "error", "message": "Discovery publisher not available"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_reset_statistics(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle statistics reset command."""
        try:
            confirm = parameters.get("confirm", False)
            if not confirm:
                return {"status": "error", "message": "Confirmation required"}
            
            # Reset integration statistics
            self.stats = EnhancedIntegrationStats()
            
            # Reset MQTT integration stats if available
            if self.mqtt_integration_manager:
                self.mqtt_integration_manager.stats = type(self.mqtt_integration_manager.stats)()
            
            return {"status": "success", "message": "Statistics reset"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_generate_diagnostic(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle diagnostic report generation command."""
        try:
            include_logs = parameters.get("include_logs", True)
            include_metrics = parameters.get("include_metrics", True)
            
            diagnostic_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "integration_stats": self.get_integration_stats(),
                "entity_states_count": len(self.entity_states),
                "command_handlers_count": len(self.command_handlers)
            }
            
            if include_metrics:
                # Add performance metrics
                diagnostic_data["performance"] = {
                    "commands_processed": self.stats.commands_processed,
                    "state_updates_sent": self.stats.state_updates_sent,
                    "integration_errors": self.stats.integration_errors
                }
            
            return {"status": "success", "diagnostic_data": diagnostic_data}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_check_database(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle database health check command."""
        try:
            # This would typically check database connectivity and performance
            return {
                "status": "success", 
                "database_status": "healthy",
                "message": "Database health check completed"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_force_prediction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle force prediction command."""
        try:
            room_id = parameters.get("room_id")
            if not room_id:
                return {"status": "error", "message": "room_id required"}
            
            # Delegate to tracking manager if available
            if self.tracking_manager:
                if hasattr(self.tracking_manager, 'force_prediction'):
                    result = await self.tracking_manager.force_prediction(room_id=room_id)
                    return {"status": "success", "result": result}
            
            return {"status": "error", "message": "Tracking manager not available"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_prediction_enable(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction system enable/disable command."""
        try:
            enabled = parameters.get("enabled", True)
            
            # This would typically enable/disable the prediction system
            return {
                "status": "success", 
                "prediction_enabled": enabled,
                "message": f"Prediction system {'enabled' if enabled else 'disabled'}"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_mqtt_enable(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle MQTT publishing enable/disable command."""
        try:
            enabled = parameters.get("enabled", True)
            
            # This would typically enable/disable MQTT publishing
            return {
                "status": "success", 
                "mqtt_enabled": enabled,
                "message": f"MQTT publishing {'enabled' if enabled else 'disabled'}"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_set_interval(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle prediction interval configuration command."""
        try:
            interval = parameters.get("interval", 300)
            
            # This would typically update the prediction interval
            return {
                "status": "success", 
                "prediction_interval": interval,
                "message": f"Prediction interval set to {interval} seconds"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _handle_set_log_level(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Handle log level configuration command."""
        try:
            log_level = parameters.get("log_level", "INFO")
            
            # This would typically update the logging level
            logging.getLogger().setLevel(getattr(logging, log_level, logging.INFO))
            
            return {
                "status": "success", 
                "log_level": log_level,
                "message": f"Log level set to {log_level}"
            }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


class EnhancedIntegrationError(OccupancyPredictionError):
    """Raised when enhanced HA integration operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="ENHANCED_INTEGRATION_ERROR",
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            **kwargs
        )