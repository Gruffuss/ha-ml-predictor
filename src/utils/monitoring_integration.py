"""
Monitoring integration for Home Assistant ML Predictor.
Integrates monitoring system with existing TrackingManager and components.
"""

import asyncio
from datetime import datetime
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from .logger import get_logger, get_performance_logger, get_ml_ops_logger
from .metrics import get_metrics_manager, get_metrics_collector
from .monitoring import get_monitoring_manager
from .alerts import get_alert_manager


class MonitoringIntegration:
    """Integration layer between monitoring system and existing components."""
    
    def __init__(self):
        self.logger = get_logger("monitoring_integration")
        self.performance_logger = get_performance_logger()
        self.ml_ops_logger = get_ml_ops_logger()
        self.metrics_manager = get_metrics_manager()
        self.metrics_collector = get_metrics_collector()
        self.monitoring_manager = get_monitoring_manager()
        self.alert_manager = get_alert_manager()
        
        self._setup_integrations()
    
    def _setup_integrations(self):
        """Setup integrations with monitoring systems."""
        # Setup alert callbacks for critical issues
        self.monitoring_manager.get_performance_monitor().add_alert_callback(
            self._handle_performance_alert
        )
        
        self.logger.info("Monitoring integration initialized")
    
    async def start_monitoring(self):
        """Start all monitoring components."""
        try:
            # Start metrics collection
            self.metrics_manager.start_background_collection()
            
            # Start system monitoring
            await self.monitoring_manager.start_monitoring()
            
            self.logger.info("Monitoring system started successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to start monitoring system: {e}")
            await self.alert_manager.trigger_alert(
                rule_name="system_startup_error",
                title="Monitoring System Startup Failed",
                message=f"Failed to start monitoring components: {e}",
                component="monitoring_system"
            )
            raise
    
    async def stop_monitoring(self):
        """Stop all monitoring components."""
        try:
            await self.monitoring_manager.stop_monitoring()
            self.metrics_manager.stop_background_collection()
            
            self.logger.info("Monitoring system stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping monitoring system: {e}")
    
    async def _handle_performance_alert(self, alert):
        """Handle performance alerts from monitoring system."""
        try:
            # Convert monitoring alert to alert manager format
            await self.alert_manager.trigger_alert(
                rule_name=f"performance_{alert.alert_type}",
                title=f"Performance Alert: {alert.message}",
                message=alert.message,
                component=alert.component,
                room_id=getattr(alert, 'room_id', None),
                context=alert.additional_info or {}
            )
        except Exception as e:
            self.logger.error(f"Failed to handle performance alert: {e}")
    
    @asynccontextmanager
    async def track_prediction_operation(self, room_id: str, prediction_type: str, 
                                       model_type: str):
        """Context manager to track prediction operations with comprehensive monitoring."""
        start_time = datetime.now()
        
        try:
            self.logger.info(
                f"Starting prediction: {prediction_type} for {room_id}",
                extra={
                    'room_id': room_id,
                    'prediction_type': prediction_type,
                    'model_type': model_type,
                    'operation': 'prediction_start'
                }
            )
            
            yield
            
            # Calculate duration
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record performance metrics
            self.performance_logger.log_operation_time(
                operation="prediction_generation",
                duration=duration,
                room_id=room_id,
                prediction_type=prediction_type,
                model_type=model_type
            )
            
            # Record Prometheus metrics
            self.metrics_collector.record_prediction(
                room_id=room_id,
                prediction_type=prediction_type,
                model_type=model_type,
                duration=duration,
                status='success'
            )
            
            # Check performance thresholds
            self.monitoring_manager.get_performance_monitor().record_performance_metric(
                'prediction_latency',
                duration,
                room_id=room_id,
                additional_info={
                    'prediction_type': prediction_type,
                    'model_type': model_type
                }
            )
            
            self.logger.info(
                f"Prediction completed: {prediction_type} for {room_id} in {duration:.3f}s",
                extra={
                    'room_id': room_id,
                    'prediction_type': prediction_type,
                    'model_type': model_type,
                    'duration_seconds': duration,
                    'operation': 'prediction_complete'
                }
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record error metrics
            self.metrics_collector.record_prediction(
                room_id=room_id,
                prediction_type=prediction_type,
                model_type=model_type,
                duration=duration,
                status='error'
            )
            
            # Handle prediction error through alert system
            await self.alert_manager.handle_prediction_error(
                error=e,
                room_id=room_id,
                prediction_type=prediction_type,
                model_type=model_type
            )
            
            self.logger.error(
                f"Prediction failed: {prediction_type} for {room_id} after {duration:.3f}s",
                extra={
                    'room_id': room_id,
                    'prediction_type': prediction_type,
                    'model_type': model_type,
                    'duration_seconds': duration,
                    'error': str(e),
                    'operation': 'prediction_error'
                },
                exc_info=True
            )
            
            raise
    
    @asynccontextmanager
    async def track_training_operation(self, room_id: str, model_type: str, 
                                     training_type: str):
        """Context manager to track model training operations."""
        start_time = datetime.now()
        
        try:
            self.ml_ops_logger.log_training_event(
                room_id=room_id,
                model_type=model_type,
                event_type=f"{training_type}_start"
            )
            
            yield
            
            duration = (datetime.now() - start_time).total_seconds()
            
            # Record training metrics
            self.metrics_collector.record_model_training(
                room_id=room_id,
                model_type=model_type,
                training_type=training_type,
                duration=duration
            )
            
            # Check training time thresholds
            self.monitoring_manager.get_performance_monitor().record_performance_metric(
                'model_training_time',
                duration,
                room_id=room_id,
                additional_info={
                    'model_type': model_type,
                    'training_type': training_type
                }
            )
            
            self.ml_ops_logger.log_training_event(
                room_id=room_id,
                model_type=model_type,
                event_type=f"{training_type}_complete",
                metrics={'duration_seconds': duration}
            )
            
        except Exception as e:
            duration = (datetime.now() - start_time).total_seconds()
            
            # Handle training error
            await self.alert_manager.handle_model_training_error(
                error=e,
                room_id=room_id,
                model_type=model_type
            )
            
            self.ml_ops_logger.log_training_event(
                room_id=room_id,
                model_type=model_type,
                event_type=f"{training_type}_error",
                metrics={'duration_seconds': duration, 'error': str(e)}
            )
            
            raise
    
    def record_prediction_accuracy(self, room_id: str, model_type: str,
                                 prediction_type: str, accuracy_minutes: float,
                                 confidence: float):
        """Record prediction accuracy metrics."""
        # Log performance
        self.performance_logger.log_prediction_accuracy(
            room_id=room_id,
            accuracy_minutes=accuracy_minutes,
            confidence=confidence,
            prediction_type=prediction_type
        )
        
        # Update Prometheus metrics
        self.metrics_collector.record_prediction(
            room_id=room_id,
            prediction_type=prediction_type,
            model_type=model_type,
            duration=0,  # Already recorded during prediction
            accuracy_minutes=accuracy_minutes,
            confidence=confidence
        )
        
        # Check accuracy thresholds
        self.monitoring_manager.get_performance_monitor().record_performance_metric(
            'prediction_accuracy',
            accuracy_minutes,
            room_id=room_id,
            additional_info={
                'model_type': model_type,
                'prediction_type': prediction_type,
                'confidence': confidence
            }
        )
    
    def record_concept_drift(self, room_id: str, drift_type: str, 
                           severity: float, action_taken: str):
        """Record concept drift detection."""
        # Log ML ops event
        self.ml_ops_logger.log_drift_detection(
            room_id=room_id,
            drift_type=drift_type,
            severity=severity,
            action_taken=action_taken
        )
        
        # Record metrics
        self.metrics_collector.record_concept_drift(
            room_id=room_id,
            drift_type=drift_type,
            severity=severity,
            action_taken=action_taken
        )
        
        # Trigger alert if severity is high
        if severity > 0.5:
            asyncio.create_task(
                self.alert_manager.trigger_alert(
                    rule_name="concept_drift_detected",
                    title=f"Concept Drift Detected: {room_id}",
                    message=f"Drift type: {drift_type}, Severity: {severity:.2f}, Action: {action_taken}",
                    component="adaptation_system",
                    room_id=room_id,
                    context={
                        'drift_type': drift_type,
                        'severity': severity,
                        'action_taken': action_taken
                    }
                )
            )
    
    def record_feature_computation(self, room_id: str, feature_type: str, duration: float):
        """Record feature computation metrics."""
        self.metrics_collector.record_feature_computation(
            room_id=room_id,
            feature_type=feature_type,
            duration=duration
        )
        
        # Check performance thresholds
        self.monitoring_manager.get_performance_monitor().record_performance_metric(
            'feature_computation_time',
            duration,
            room_id=room_id,
            additional_info={'feature_type': feature_type}
        )
    
    def record_database_operation(self, operation_type: str, table: str, 
                                duration: float, status: str = 'success'):
        """Record database operation metrics."""
        self.metrics_collector.record_database_operation(
            operation_type=operation_type,
            table=table,
            duration=duration,
            status=status
        )
        
        # Check database performance thresholds
        self.monitoring_manager.get_performance_monitor().record_performance_metric(
            'database_query_time',
            duration,
            additional_info={
                'operation_type': operation_type,
                'table': table,
                'status': status
            }
        )
    
    def record_mqtt_publish(self, topic_type: str, room_id: str, status: str = 'success'):
        """Record MQTT publishing metrics."""
        self.metrics_collector.record_mqtt_publish(
            topic_type=topic_type,
            room_id=room_id,
            status=status
        )
    
    def record_ha_api_request(self, endpoint: str, method: str, status: str):
        """Record Home Assistant API request metrics."""
        self.metrics_collector.record_ha_api_request(
            endpoint=endpoint,
            method=method,
            status=status
        )
    
    def update_connection_status(self, connection_type: str, connected: bool):
        """Update connection status metrics."""
        self.metrics_collector.update_ha_connection_status(
            connection_type=connection_type,
            connected=connected
        )
        
        # Trigger alert if connection lost
        if not connected:
            asyncio.create_task(
                self.alert_manager.trigger_alert(
                    rule_name="ha_connection_lost",
                    title=f"Connection Lost: {connection_type}",
                    message=f"Lost connection to {connection_type}",
                    component="integration",
                    context={'connection_type': connection_type}
                )
            )
    
    async def get_monitoring_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status."""
        try:
            # Get monitoring system status
            monitoring_status = await self.monitoring_manager.get_monitoring_status()
            
            # Get alert system status
            alert_status = self.alert_manager.get_alert_status()
            
            # Get metrics endpoint
            metrics_available = self.metrics_manager.get_metrics() != ""
            
            return {
                'monitoring_system': monitoring_status,
                'alert_system': alert_status,
                'metrics_collection': {
                    'enabled': metrics_available,
                    'endpoint_available': True
                },
                'integration_status': {
                    'performance_tracking': True,
                    'error_handling': True,
                    'ml_ops_logging': True
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get monitoring status: {e}")
            return {
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }


# Global monitoring integration instance
_monitoring_integration = None

def get_monitoring_integration() -> MonitoringIntegration:
    """Get global monitoring integration instance."""
    global _monitoring_integration
    if _monitoring_integration is None:
        _monitoring_integration = MonitoringIntegration()
    return _monitoring_integration