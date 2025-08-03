#!/usr/bin/env python3
"""
Example Usage: Performance Monitoring Dashboard Integration

This example demonstrates how to integrate the Performance Monitoring Dashboard
with an existing TrackingManager for real-time system monitoring.

The dashboard provides:
- System overview with key performance indicators
- Real-time accuracy metrics and trends
- Drift detection status and visualization
- Retraining queue and history
- Alert management and notifications
- WebSocket real-time updates
"""

import asyncio
import logging
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import system components
from src.core.config import get_config
from src.adaptation.tracking_manager import TrackingManager, TrackingConfig
from src.integration.dashboard import (
    PerformanceDashboard,
    DashboardConfig,
    DashboardMode,
    create_dashboard_from_tracking_manager
)


async def main():
    """
    Example integration of Performance Monitoring Dashboard.
    
    This shows the complete setup process for integrating the dashboard
    with an existing occupancy prediction system.
    """
    logger.info("Starting Performance Monitoring Dashboard Example")
    
    try:
        # 1. Load system configuration
        logger.info("Loading system configuration...")
        system_config = get_config()
        
        # 2. Create tracking configuration with dashboard integration
        tracking_config = TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=30,
            auto_validation_enabled=True,
            validation_window_minutes=15,
            
            # Drift detection settings
            drift_detection_enabled=True,
            drift_check_interval_hours=6,  # Check more frequently for demo
            drift_baseline_days=7,  # Shorter baseline for demo
            drift_current_days=1,
            
            # Adaptive retraining settings
            adaptive_retraining_enabled=True,
            retraining_accuracy_threshold=70.0,
            retraining_error_threshold=20.0,
            retraining_check_interval_hours=2  # Check more frequently for demo
        )
        
        # 3. Initialize TrackingManager (assumes database and models are set up)
        logger.info("Initializing TrackingManager with comprehensive monitoring...")
        tracking_manager = TrackingManager(
            config=tracking_config,
            database_manager=None,  # Would be provided in real system
            model_registry={}       # Would contain actual models
        )
        
        # Initialize tracking system
        await tracking_manager.initialize()
        
        # 4. Create dashboard configuration
        dashboard_config = DashboardConfig(
            enabled=True,
            host="0.0.0.0",          # Accept connections from any host
            port=8888,               # Dashboard port
            debug=True,              # Enable debug mode for development
            mode=DashboardMode.DEVELOPMENT,
            
            # Real-time features
            websocket_enabled=True,
            update_interval_seconds=5,   # Update every 5 seconds
            max_websocket_connections=25,
            
            # Dashboard features
            enable_historical_charts=True,
            enable_drift_visualization=True,
            enable_retraining_controls=True,
            enable_alert_management=True,
            enable_export_features=True,
            
            # Performance settings
            cache_ttl_seconds=30,
            metrics_retention_hours=48
        )
        
        # 5. Create and integrate dashboard using helper function
        logger.info("Creating Performance Monitoring Dashboard...")
        dashboard = await create_dashboard_from_tracking_manager(
            tracking_manager=tracking_manager,
            host=dashboard_config.host,
            port=dashboard_config.port,
            debug=dashboard_config.debug
        )
        
        # 6. Start tracking system background tasks
        logger.info("Starting tracking system background monitoring...")
        await tracking_manager.start_tracking()
        
        # 7. Dashboard is now running and integrated
        logger.info("=" * 60)
        logger.info("🎉 PERFORMANCE MONITORING DASHBOARD ACTIVE!")
        logger.info("=" * 60)
        logger.info(f"📊 Dashboard URL: http://{dashboard_config.host}:{dashboard_config.port}")
        logger.info(f"🔄 WebSocket URL: ws://{dashboard_config.host}:{dashboard_config.port}/ws/dashboard")
        logger.info("")
        logger.info("Available Endpoints:")
        logger.info("  📈 GET /api/dashboard/overview       - System overview")
        logger.info("  🎯 GET /api/dashboard/accuracy       - Accuracy metrics")
        logger.info("  📊 GET /api/dashboard/drift          - Drift detection")
        logger.info("  🔄 GET /api/dashboard/retraining     - Retraining status")
        logger.info("  ❤️  GET /api/dashboard/health        - System health")
        logger.info("  🚨 GET /api/dashboard/alerts         - Active alerts")
        logger.info("  📉 GET /api/dashboard/trends         - Historical trends")
        logger.info("  📋 GET /api/dashboard/stats          - Dashboard stats")
        logger.info("")
        logger.info("Manual Actions (if enabled):")
        logger.info("  🔧 POST /api/dashboard/actions/retrain           - Trigger retraining")
        logger.info("  ✅ POST /api/dashboard/actions/acknowledge_alert - Acknowledge alert")
        logger.info("")
        logger.info("Real-time Features:")
        logger.info("  🌐 WebSocket updates every 5 seconds")
        logger.info("  📱 Live system metrics and alerts")
        logger.info("  🔔 Real-time notifications")
        logger.info("=" * 60)
        
        # 8. Simulate some system activity for demonstration
        await simulate_system_activity(tracking_manager)
        
        # 9. Keep dashboard running
        logger.info("Dashboard running... Press Ctrl+C to stop")
        try:
            # Keep running until interrupted
            while True:
                await asyncio.sleep(10)
                
                # Show periodic status update
                status = await tracking_manager.get_tracking_status()
                if status:
                    logger.info(
                        f"System Status: {status.get('status', 'unknown')} | "
                        f"Health: {status.get('overall_health_score', 0):.1f}% | "
                        f"Active Rooms: {len(status.get('tracked_rooms', []))} | "
                        f"Alerts: {status.get('alerts_summary', {}).get('total_active', 0)}"
                    )
                
        except KeyboardInterrupt:
            logger.info("\nShutdown requested by user")
        
        # 10. Cleanup
        logger.info("Stopping Performance Monitoring Dashboard...")
        await dashboard.stop_dashboard()
        await tracking_manager.stop_tracking()
        
        logger.info("Dashboard stopped successfully")
        
    except Exception as e:
        logger.error(f"Error in dashboard example: {e}")
        raise


async def simulate_system_activity(tracking_manager: TrackingManager):
    """
    Simulate some system activity to populate the dashboard.
    
    In a real system, this would be driven by actual predictions
    and room state changes from the occupancy prediction models.
    """
    logger.info("Simulating system activity for dashboard demonstration...")
    
    try:
        # Simulate some prediction recording (normally done by models)
        from src.models.base.predictor import PredictionResult
        from datetime import datetime, timedelta
        import random
        
        # Simulate predictions for different rooms
        rooms = ['living_room', 'bedroom', 'kitchen', 'office']
        
        for room in rooms:
            # Create mock prediction result
            prediction = PredictionResult(
                room_id=room,
                predicted_time=datetime.utcnow() + timedelta(minutes=random.randint(10, 120)),
                transition_type='occupied' if random.random() > 0.5 else 'vacant',
                confidence_score=random.uniform(0.6, 0.95),
                model_type='ensemble'
            )
            
            # Record prediction through tracking manager
            await tracking_manager.record_prediction(prediction, room)
        
        logger.info(f"Simulated predictions recorded for {len(rooms)} rooms")
        
        # Simulate some room state changes for validation
        for room in rooms[:2]:  # Validate some predictions
            await tracking_manager.handle_room_state_change(
                room_id=room,
                new_state='occupied',
                timestamp=datetime.utcnow()
            )
        
        logger.info("Simulated room state changes for prediction validation")
        
    except Exception as e:
        logger.warning(f"Error simulating system activity: {e}")


def example_integration_patterns():
    """
    Show different integration patterns for the dashboard.
    """
    logger.info("Dashboard Integration Patterns:")
    logger.info("")
    
    # Pattern 1: Simple integration with existing TrackingManager
    logger.info("Pattern 1: Simple Integration")
    logger.info("```python")
    logger.info("# Assuming you have a TrackingManager instance")
    logger.info("dashboard = integrate_dashboard_with_tracking_system(tracking_manager)")
    logger.info("await dashboard.start_dashboard()")
    logger.info("```")
    logger.info("")
    
    # Pattern 2: Custom configuration
    logger.info("Pattern 2: Custom Configuration")
    logger.info("```python")
    logger.info("config = DashboardConfig(")
    logger.info("    host='localhost',")
    logger.info("    port=9999,")
    logger.info("    mode=DashboardMode.PRODUCTION,")
    logger.info("    websocket_enabled=True")
    logger.info(")")
    logger.info("dashboard = PerformanceDashboard(tracking_manager, config)")
    logger.info("await dashboard.start_dashboard()")
    logger.info("```")
    logger.info("")
    
    # Pattern 3: Integration with existing FastAPI app
    logger.info("Pattern 3: Integration with Existing FastAPI App")
    logger.info("```python")
    logger.info("# Add dashboard routes to existing app")
    logger.info("from fastapi import FastAPI")
    logger.info("app = FastAPI()")
    logger.info("dashboard = PerformanceDashboard(tracking_manager)")
    logger.info("app.mount('/dashboard', dashboard.app)")
    logger.info("```")
    logger.info("")


if __name__ == "__main__":
    # Show integration patterns
    example_integration_patterns()
    
    # Run the main dashboard example
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Dashboard example stopped by user")
    except Exception as e:
        logger.error(f"Dashboard example failed: {e}")
        exit(1)