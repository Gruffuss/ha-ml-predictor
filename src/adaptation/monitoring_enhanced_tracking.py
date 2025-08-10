"""
Monitoring-enhanced tracking manager for comprehensive observability.
Extends existing TrackingManager with integrated monitoring capabilities.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, Optional
from contextlib import asynccontextmanager

from ..core.constants import ModelType
from ..models.base.predictor import PredictionResult
from ..utils.monitoring_integration import get_monitoring_integration
from .tracking_manager import TrackingManager, TrackingConfig


class MonitoringEnhancedTrackingManager:
    """
    Enhanced tracking manager with comprehensive monitoring integration.
    
    This wrapper provides seamless monitoring capabilities while maintaining
    compatibility with the existing TrackingManager interface.
    """
    
    def __init__(self, tracking_manager: TrackingManager):
        """Initialize with existing tracking manager."""
        self.tracking_manager = tracking_manager
        self.monitoring_integration = get_monitoring_integration()
        self._original_methods = {}
        
        # Wrap key methods with monitoring
        self._wrap_tracking_methods()
    
    def _wrap_tracking_methods(self):
        """Wrap TrackingManager methods with monitoring capabilities."""
        # Store original methods
        self._original_methods = {
            'record_prediction': self.tracking_manager.record_prediction,
            'validate_prediction': self.tracking_manager.validate_prediction,
            'start_tracking': self.tracking_manager.start_tracking,
            'stop_tracking': self.tracking_manager.stop_tracking,
        }
        
        # Replace with monitored versions
        self.tracking_manager.record_prediction = self._monitored_record_prediction
        self.tracking_manager.validate_prediction = self._monitored_validate_prediction
        self.tracking_manager.start_tracking = self._monitored_start_tracking
        self.tracking_manager.stop_tracking = self._monitored_stop_tracking
    
    async def _monitored_record_prediction(self, room_id: str, 
                                         prediction_result: PredictionResult,
                                         model_type: ModelType = ModelType.ENSEMBLE,
                                         **kwargs) -> str:
        """Record prediction with monitoring integration."""
        prediction_type = prediction_result.prediction_type
        model_type_str = model_type.value if hasattr(model_type, 'value') else str(model_type)
        
        async with self.monitoring_integration.track_prediction_operation(
            room_id=room_id,
            prediction_type=prediction_type,
            model_type=model_type_str
        ):
            # Call original method
            result = await self._original_methods['record_prediction'](
                room_id, prediction_result, model_type, **kwargs
            )
            
            # Record additional monitoring data
            if hasattr(prediction_result, 'confidence') and prediction_result.confidence is not None:
                self.monitoring_integration.record_prediction_accuracy(
                    room_id=room_id,
                    model_type=model_type_str,
                    prediction_type=prediction_type,
                    accuracy_minutes=0,  # Will be updated when validated
                    confidence=prediction_result.confidence
                )
            
            return result
    
    async def _monitored_validate_prediction(self, room_id: str, 
                                           actual_time: datetime,
                                           **kwargs) -> Dict[str, Any]:
        """Validate prediction with monitoring integration."""
        start_time = datetime.now()
        
        try:
            # Call original method
            result = await self._original_methods['validate_prediction'](
                room_id, actual_time, **kwargs
            )
            
            # Extract validation results for monitoring
            if isinstance(result, dict):
                accuracy_minutes = result.get('accuracy_minutes', 0)
                prediction_type = result.get('prediction_type', 'unknown')
                model_type = result.get('model_type', 'unknown')
                confidence = result.get('confidence', 0)
                
                # Record accuracy metrics
                self.monitoring_integration.record_prediction_accuracy(
                    room_id=room_id,
                    model_type=str(model_type),
                    prediction_type=prediction_type,
                    accuracy_minutes=accuracy_minutes,
                    confidence=confidence
                )
            
            return result
            
        except Exception as e:
            # Record validation error
            duration = (datetime.now() - start_time).total_seconds()
            await self.monitoring_integration.alert_manager.trigger_alert(
                rule_name="prediction_validation_error",
                title=f"Prediction Validation Error: {room_id}",
                message=f"Failed to validate prediction for {room_id}: {e}",
                component="validation_system",
                room_id=room_id,
                context={
                    'error': str(e),
                    'duration_seconds': duration
                }
            )
            raise
    
    async def _monitored_start_tracking(self, **kwargs):
        """Start tracking with monitoring system initialization."""
        try:
            # Start monitoring system first
            await self.monitoring_integration.start_monitoring()
            
            # Then start original tracking
            result = await self._original_methods['start_tracking'](**kwargs)
            
            # Record successful startup
            await self.monitoring_integration.alert_manager.trigger_alert(
                rule_name="system_startup_success",
                title="Tracking System Started",
                message="Tracking system and monitoring started successfully",
                component="tracking_system",
                context={'monitoring_enabled': True}
            )
            
            return result
            
        except Exception as e:
            # Record startup failure
            await self.monitoring_integration.alert_manager.trigger_alert(
                rule_name="system_startup_error",
                title="Tracking System Startup Failed",
                message=f"Failed to start tracking system: {e}",
                component="tracking_system",
                context={'error': str(e)}
            )
            raise
    
    async def _monitored_stop_tracking(self, **kwargs):
        """Stop tracking with monitoring system cleanup."""
        try:
            # Stop original tracking first
            result = await self._original_methods['stop_tracking'](**kwargs)
            
            # Then stop monitoring system
            await self.monitoring_integration.stop_monitoring()
            
            return result
            
        except Exception as e:
            # Log error but try to stop monitoring anyway
            try:
                await self.monitoring_integration.stop_monitoring()
            except:
                pass
            raise
    
    def record_concept_drift(self, room_id: str, drift_type: str, 
                           severity: float, action_taken: str):
        """Record concept drift with monitoring integration."""
        # Call monitoring integration
        self.monitoring_integration.record_concept_drift(
            room_id=room_id,
            drift_type=drift_type,
            severity=severity,
            action_taken=action_taken
        )
        
        # If tracking manager has its own drift recording, call it too
        if hasattr(self.tracking_manager, 'record_concept_drift'):
            self.tracking_manager.record_concept_drift(
                room_id, drift_type, severity, action_taken
            )
    
    def record_feature_computation(self, room_id: str, feature_type: str, duration: float):
        """Record feature computation metrics."""
        self.monitoring_integration.record_feature_computation(
            room_id=room_id,
            feature_type=feature_type,
            duration=duration
        )
    
    def record_database_operation(self, operation_type: str, table: str, 
                                duration: float, status: str = 'success'):
        """Record database operation metrics."""
        self.monitoring_integration.record_database_operation(
            operation_type=operation_type,
            table=table,
            duration=duration,
            status=status
        )
    
    def record_mqtt_publish(self, topic_type: str, room_id: str, status: str = 'success'):
        """Record MQTT publishing metrics."""
        self.monitoring_integration.record_mqtt_publish(
            topic_type=topic_type,
            room_id=room_id,
            status=status
        )
    
    def update_connection_status(self, connection_type: str, connected: bool):
        """Update connection status."""
        self.monitoring_integration.update_connection_status(
            connection_type=connection_type,
            connected=connected
        )
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        monitoring_status = await self.monitoring_integration.get_monitoring_status()
        
        # Add tracking manager status if available
        tracking_status = {}
        if hasattr(self.tracking_manager, 'get_system_status'):
            try:
                tracking_status = await self.tracking_manager.get_system_status()
            except:
                tracking_status = {'error': 'Failed to get tracking status'}
        
        return {
            'monitoring': monitoring_status,
            'tracking': tracking_status,
            'integrated': True,
            'timestamp': datetime.now().isoformat()
        }
    
    @asynccontextmanager
    async def track_model_training(self, room_id: str, model_type: str, 
                                 training_type: str = 'retraining'):
        """Context manager for tracking model training operations."""
        async with self.monitoring_integration.track_training_operation(
            room_id=room_id,
            model_type=model_type,
            training_type=training_type
        ):
            yield
    
    def __getattr__(self, name):
        """Delegate unknown attributes to the original tracking manager."""
        return getattr(self.tracking_manager, name)


def create_monitoring_enhanced_tracking_manager(
    config: TrackingConfig,
    **kwargs
) -> MonitoringEnhancedTrackingManager:
    """
    Create a monitoring-enhanced tracking manager.
    
    This factory function creates a standard TrackingManager and wraps it
    with monitoring capabilities.
    """
    # Create standard tracking manager
    tracking_manager = TrackingManager(config=config, **kwargs)
    
    # Wrap with monitoring enhancements
    enhanced_manager = MonitoringEnhancedTrackingManager(tracking_manager)
    
    return enhanced_manager


# Convenience function for backward compatibility
def get_enhanced_tracking_manager(config: TrackingConfig, 
                                **kwargs) -> MonitoringEnhancedTrackingManager:
    """Get a monitoring-enhanced tracking manager instance."""
    return create_monitoring_enhanced_tracking_manager(config, **kwargs)