"""
Home Assistant Integration Bridge for TrackingManager.

This module provides seamless integration between the enhanced Home Assistant
entity system and the existing TrackingManager, ensuring automatic operation
without manual intervention and proper system workflow integration.

Features:
- Automatic HA entity state updates when tracking events occur
- Command delegation from HA to TrackingManager
- Real-time synchronization between tracking system and HA entities
- System status publishing to HA diagnostic entities
- No manual setup required - fully integrated into main system workflow
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass

from ..core.exceptions import OccupancyPredictionError, ErrorSeverity
from ..models.base.predictor import PredictionResult
from ..adaptation.tracking_manager import TrackingManager
from .enhanced_integration_manager import EnhancedIntegrationManager


logger = logging.getLogger(__name__)


@dataclass
class HATrackingBridgeStats:
    """Statistics for HA tracking bridge operations."""
    bridge_initialized: bool = False
    entity_updates_sent: int = 0
    commands_delegated: int = 0
    tracking_events_processed: int = 0
    system_status_updates: int = 0
    last_entity_update: Optional[datetime] = None
    last_command_delegation: Optional[datetime] = None
    bridge_errors: int = 0
    last_error: Optional[str] = None


class HATrackingBridge:
    """
    Bridge between Home Assistant integration and TrackingManager.
    
    This bridge ensures seamless integration between the enhanced HA entity
    system and the existing tracking infrastructure, providing automatic
    synchronization and command delegation.
    """
    
    def __init__(
        self,
        tracking_manager: TrackingManager,
        enhanced_integration_manager: EnhancedIntegrationManager
    ):
        """
        Initialize HA tracking bridge.
        
        Args:
            tracking_manager: Existing tracking manager instance
            enhanced_integration_manager: Enhanced HA integration manager
        """
        self.tracking_manager = tracking_manager
        self.enhanced_integration_manager = enhanced_integration_manager
        
        # Bridge state
        self.stats = HATrackingBridgeStats()
        self._bridge_active = False
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        
        # Event handlers for tracking events
        self._tracking_event_handlers: Dict[str, Callable] = {}
        
        logger.info("Initialized HATrackingBridge")
    
    async def initialize(self) -> None:
        """Initialize HA tracking bridge and setup event handlers."""
        try:
            logger.info("Initializing HA tracking bridge")
            
            # Setup tracking event handlers
            self._setup_tracking_event_handlers()
            
            # Register bridge with enhanced integration manager
            self.enhanced_integration_manager.tracking_manager = self.tracking_manager
            
            # Setup command delegation
            self._setup_command_delegation()
            
            # Start background synchronization tasks
            await self._start_background_tasks()
            
            self._bridge_active = True
            self.stats.bridge_initialized = True
            
            logger.info("HA tracking bridge initialized successfully")
            
        except Exception as e:
            self.stats.bridge_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error initializing HA tracking bridge: {e}")
            raise
    
    async def shutdown(self) -> None:
        """Shutdown HA tracking bridge."""
        try:
            logger.info("Shutting down HA tracking bridge")
            
            # Signal shutdown
            self._shutdown_event.set()
            self._bridge_active = False
            
            # Cancel background tasks
            for task in self._background_tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)
            
            logger.info("HA tracking bridge shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during HA tracking bridge shutdown: {e}")
    
    async def handle_prediction_made(
        self,
        room_id: str,
        prediction_result: PredictionResult
    ) -> None:
        """
        Handle prediction made event and update HA entities.
        
        Args:
            room_id: Room identifier
            prediction_result: Prediction result from model
        """
        try:
            if not self._bridge_active:
                return
            
            # Update HA entities through enhanced integration manager
            await self.enhanced_integration_manager.handle_prediction_update(
                room_id, prediction_result
            )
            
            # Record tracking event
            self.stats.tracking_events_processed += 1
            self.stats.entity_updates_sent += 1
            self.stats.last_entity_update = datetime.utcnow()
            
            logger.debug(f"Updated HA entities for prediction in {room_id}")
            
        except Exception as e:
            self.stats.bridge_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error handling prediction made event: {e}")
    
    async def handle_accuracy_alert(self, alert: Any) -> None:
        """
        Handle accuracy alert and update HA entities.
        
        Args:
            alert: Accuracy alert from tracking system
        """
        try:
            if not self._bridge_active:
                return
            
            # Update alert-related HA entities
            alert_data = {
                "alert_type": getattr(alert, 'alert_type', 'unknown'),
                "severity": getattr(alert, 'severity', 'unknown'),
                "room_id": getattr(alert, 'room_id', None),
                "message": getattr(alert, 'message', ''),
                "timestamp": getattr(alert, 'timestamp', datetime.utcnow()).isoformat()
            }
            
            # Update system status with alert information
            await self._update_system_alert_status(alert_data)
            
            logger.info(f"Updated HA entities for accuracy alert: {alert_data['alert_type']}")
            
        except Exception as e:
            self.stats.bridge_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error handling accuracy alert: {e}")
    
    async def handle_drift_detected(self, drift_metrics: Any) -> None:
        """
        Handle concept drift detection and update HA entities.
        
        Args:
            drift_metrics: Drift detection metrics
        """
        try:
            if not self._bridge_active:
                return
            
            # Update drift-related HA entities
            drift_data = {
                "drift_detected": True,
                "drift_score": getattr(drift_metrics, 'drift_score', 0.0),
                "drift_severity": getattr(drift_metrics, 'severity', 'unknown'),
                "affected_features": getattr(drift_metrics, 'affected_features', []),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Update system status with drift information
            await self._update_system_drift_status(drift_data)
            
            logger.info(f"Updated HA entities for drift detection: {drift_data['drift_severity']}")
            
        except Exception as e:
            self.stats.bridge_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error handling drift detection: {e}")
    
    async def handle_retraining_started(self, room_id: str, retraining_info: Dict[str, Any]) -> None:
        """
        Handle model retraining started event.
        
        Args:
            room_id: Room identifier
            retraining_info: Retraining information
        """
        try:
            if not self._bridge_active:
                return
            
            # Update model training status in HA
            await self.enhanced_integration_manager.update_entity_state(
                "model_training",
                True,
                {
                    "room_id": room_id,
                    "training_started": datetime.utcnow().isoformat(),
                    "training_type": retraining_info.get("training_type", "unknown")
                }
            )
            
            logger.info(f"Updated HA entities for retraining started: {room_id}")
            
        except Exception as e:
            self.stats.bridge_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error handling retraining started: {e}")
    
    async def handle_retraining_completed(
        self,
        room_id: str,
        retraining_result: Dict[str, Any]
    ) -> None:
        """
        Handle model retraining completed event.
        
        Args:
            room_id: Room identifier
            retraining_result: Retraining result information
        """
        try:
            if not self._bridge_active:
                return
            
            # Update model training status in HA
            await self.enhanced_integration_manager.update_entity_state(
                "model_training",
                False,
                {
                    "room_id": room_id,
                    "training_completed": datetime.utcnow().isoformat(),
                    "training_success": retraining_result.get("success", False),
                    "new_accuracy": retraining_result.get("accuracy", 0.0)
                }
            )
            
            # Update room-specific accuracy if available
            if "accuracy" in retraining_result:
                await self.enhanced_integration_manager.update_entity_state(
                    f"{room_id}_accuracy",
                    retraining_result["accuracy"]
                )
            
            logger.info(f"Updated HA entities for retraining completed: {room_id}")
            
        except Exception as e:
            self.stats.bridge_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error handling retraining completed: {e}")
    
    def get_bridge_stats(self) -> Dict[str, Any]:
        """Get bridge statistics."""
        return {
            "bridge_stats": self.stats.__dict__,
            "bridge_active": self._bridge_active,
            "background_tasks_count": len(self._background_tasks),
            "event_handlers_count": len(self._tracking_event_handlers)
        }
    
    # Private methods - Setup and configuration
    
    def _setup_tracking_event_handlers(self) -> None:
        """Setup event handlers for tracking system events."""
        try:
            # Setup handlers for various tracking events
            self._tracking_event_handlers = {
                "prediction_made": self.handle_prediction_made,
                "accuracy_alert": self.handle_accuracy_alert,
                "drift_detected": self.handle_drift_detected,
                "retraining_started": self.handle_retraining_started,
                "retraining_completed": self.handle_retraining_completed
            }
            
            # Register handlers with tracking manager if it supports callbacks
            if hasattr(self.tracking_manager, 'register_callback'):
                for event_type, handler in self._tracking_event_handlers.items():
                    self.tracking_manager.register_callback(event_type, handler)
            
            logger.info(f"Setup {len(self._tracking_event_handlers)} tracking event handlers")
            
        except Exception as e:
            logger.error(f"Error setting up tracking event handlers: {e}")
    
    def _setup_command_delegation(self) -> None:
        """Setup command delegation from HA to tracking system."""
        try:
            # Override command handlers in enhanced integration manager
            # to delegate to tracking manager
            
            original_handlers = self.enhanced_integration_manager.command_handlers.copy()
            
            # Wrap handlers to delegate to tracking manager
            self.enhanced_integration_manager.command_handlers.update({
                "retrain_model": self._delegate_retrain_model,
                "validate_model": self._delegate_validate_model,
                "force_prediction": self._delegate_force_prediction,
                "check_database": self._delegate_check_database,
                "generate_diagnostic": self._delegate_generate_diagnostic
            })
            
            logger.info("Setup command delegation to tracking manager")
            
        except Exception as e:
            logger.error(f"Error setting up command delegation: {e}")
    
    async def _start_background_tasks(self) -> None:
        """Start background synchronization tasks."""
        try:
            # System status synchronization task
            status_sync_task = asyncio.create_task(self._system_status_sync_loop())
            self._background_tasks.append(status_sync_task)
            
            # Tracking metrics synchronization task
            metrics_sync_task = asyncio.create_task(self._metrics_sync_loop())
            self._background_tasks.append(metrics_sync_task)
            
            logger.info(f"Started {len(self._background_tasks)} background sync tasks")
            
        except Exception as e:
            logger.error(f"Error starting background tasks: {e}")
    
    async def _system_status_sync_loop(self) -> None:
        """Background task for synchronizing system status with HA."""
        try:
            logger.info("Started system status sync loop")
            
            while not self._shutdown_event.is_set():
                try:
                    # Get system status from tracking manager
                    if hasattr(self.tracking_manager, 'get_system_status'):
                        system_status = await self.tracking_manager.get_system_status()
                        
                        # Update HA entities with system status
                        await self.enhanced_integration_manager.handle_system_status_update(
                            system_status
                        )
                        
                        self.stats.system_status_updates += 1
                    
                    # Wait before next sync
                    await asyncio.sleep(60)  # Sync every minute
                    
                except Exception as e:
                    logger.error(f"Error in system status sync loop: {e}")
                    await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"System status sync loop error: {e}")
        finally:
            logger.info("System status sync loop stopped")
    
    async def _metrics_sync_loop(self) -> None:
        """Background task for synchronizing tracking metrics with HA."""
        try:
            logger.info("Started metrics sync loop")
            
            while not self._shutdown_event.is_set():
                try:
                    # Get accuracy metrics from tracking manager
                    if hasattr(self.tracking_manager, 'get_accuracy_metrics'):
                        metrics = await self.tracking_manager.get_accuracy_metrics()
                        
                        # Update room-specific accuracy entities
                        if isinstance(metrics, dict):
                            for room_id, room_metrics in metrics.items():
                                if hasattr(room_metrics, 'accuracy_percentage'):
                                    await self.enhanced_integration_manager.update_entity_state(
                                        f"{room_id}_accuracy",
                                        room_metrics.accuracy_percentage
                                    )
                    
                    # Wait before next sync
                    await asyncio.sleep(300)  # Sync every 5 minutes
                    
                except Exception as e:
                    logger.error(f"Error in metrics sync loop: {e}")
                    await asyncio.sleep(60)
            
        except Exception as e:
            logger.error(f"Metrics sync loop error: {e}")
        finally:
            logger.info("Metrics sync loop stopped")
    
    async def _update_system_alert_status(self, alert_data: Dict[str, Any]) -> None:
        """Update system status with alert information."""
        try:
            # Update active alerts count
            # This would typically query the tracking manager for current alert count
            alert_count = 1  # Simplified for this example
            
            await self.enhanced_integration_manager.update_entity_state(
                "active_alerts",
                alert_count,
                alert_data
            )
            
        except Exception as e:
            logger.error(f"Error updating system alert status: {e}")
    
    async def _update_system_drift_status(self, drift_data: Dict[str, Any]) -> None:
        """Update system status with drift information."""
        try:
            # Update system status to indicate drift detected
            await self.enhanced_integration_manager.update_entity_state(
                "system_status",
                "drift_detected",
                drift_data
            )
            
        except Exception as e:
            logger.error(f"Error updating system drift status: {e}")
    
    # Command delegation methods
    
    async def _delegate_retrain_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate model retraining command to tracking manager."""
        try:
            room_id = parameters.get("room_id")
            force = parameters.get("force", False)
            
            # Delegate to tracking manager
            if hasattr(self.tracking_manager, 'trigger_retraining'):
                result = await self.tracking_manager.trigger_retraining(
                    room_id=room_id,
                    force=force
                )
                
                self.stats.commands_delegated += 1
                self.stats.last_command_delegation = datetime.utcnow()
                
                return {"status": "success", "result": result}
            else:
                return {"status": "error", "message": "Retraining not supported"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _delegate_validate_model(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate model validation command to tracking manager."""
        try:
            room_id = parameters.get("room_id")
            days = parameters.get("days", 7)
            
            # Delegate to tracking manager
            if hasattr(self.tracking_manager, 'validate_model_performance'):
                result = await self.tracking_manager.validate_model_performance(
                    room_id=room_id,
                    validation_days=days
                )
                
                self.stats.commands_delegated += 1
                self.stats.last_command_delegation = datetime.utcnow()
                
                return {"status": "success", "result": result}
            else:
                return {"status": "error", "message": "Validation not supported"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _delegate_force_prediction(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate force prediction command to tracking manager."""
        try:
            room_id = parameters.get("room_id")
            
            # Delegate to tracking manager
            if hasattr(self.tracking_manager, 'force_prediction'):
                result = await self.tracking_manager.force_prediction(room_id=room_id)
                
                self.stats.commands_delegated += 1
                self.stats.last_command_delegation = datetime.utcnow()
                
                return {"status": "success", "result": result}
            else:
                return {"status": "error", "message": "Force prediction not supported"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _delegate_check_database(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate database check command to tracking manager."""
        try:
            # Delegate to tracking manager
            if hasattr(self.tracking_manager, 'check_database_health'):
                result = await self.tracking_manager.check_database_health()
                
                self.stats.commands_delegated += 1
                self.stats.last_command_delegation = datetime.utcnow()
                
                return {"status": "success", "result": result}
            else:
                return {"status": "success", "message": "Database check not available"}
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
    
    async def _delegate_generate_diagnostic(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate diagnostic generation command to tracking manager."""
        try:
            include_logs = parameters.get("include_logs", True)
            include_metrics = parameters.get("include_metrics", True)
            
            # Delegate to tracking manager
            if hasattr(self.tracking_manager, 'generate_diagnostic_report'):
                result = await self.tracking_manager.generate_diagnostic_report(
                    include_logs=include_logs,
                    include_metrics=include_metrics
                )
                
                self.stats.commands_delegated += 1
                self.stats.last_command_delegation = datetime.utcnow()
                
                return {"status": "success", "result": result}
            else:
                # Generate basic diagnostic from bridge stats
                return {
                    "status": "success",
                    "result": {
                        "bridge_stats": self.get_bridge_stats(),
                        "timestamp": datetime.utcnow().isoformat()
                    }
                }
            
        except Exception as e:
            return {"status": "error", "message": str(e)}


class HATrackingBridgeError(OccupancyPredictionError):
    """Raised when HA tracking bridge operations fail."""
    
    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="HA_TRACKING_BRIDGE_ERROR",
            severity=kwargs.get('severity', ErrorSeverity.MEDIUM),
            **kwargs
        )