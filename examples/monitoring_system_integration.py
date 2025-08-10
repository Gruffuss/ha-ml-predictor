#!/usr/bin/env python3
"""
Example: Monitoring System Integration

This example demonstrates how to integrate the comprehensive monitoring system
with the Home Assistant ML Predictor, showing automatic metrics collection,
alerting, and performance monitoring.
"""

import asyncio
from pathlib import Path
import sys

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from datetime import datetime, timedelta

from src.adaptation.monitoring_enhanced_tracking import (
    create_monitoring_enhanced_tracking_manager,
)
from src.adaptation.tracking_manager import TrackingConfig
from src.core.config import get_config
from src.models.base.predictor import PredictionResult
from src.utils.alerts import NotificationConfig, get_alert_manager
from src.utils.metrics import get_metrics_manager
from src.utils.monitoring_integration import get_monitoring_integration


async def demonstrate_monitoring_integration():
    """Demonstrate comprehensive monitoring system integration."""
    print("🔍 Home Assistant ML Predictor - Monitoring System Integration Demo")
    print("=" * 70)

    # Initialize monitoring components
    print("\n1. Initializing Monitoring System...")

    # Setup monitoring integration
    monitoring_integration = get_monitoring_integration()
    metrics_manager = get_metrics_manager()

    # Configure notifications (in production, you'd set real SMTP/webhook details)
    notification_config = NotificationConfig(
        email_enabled=False,  # Set to True with real config in production
        webhook_enabled=False,
        mqtt_enabled=False,
    )
    alert_manager = get_alert_manager()

    print("✅ Monitoring components initialized")

    # Start monitoring system
    print("\n2. Starting Background Monitoring...")
    await monitoring_integration.start_monitoring()
    print("✅ Monitoring system started")

    # Create enhanced tracking manager with monitoring
    print("\n3. Creating Monitoring-Enhanced Tracking Manager...")

    tracking_config = TrackingConfig(
        validation_window_minutes=30,
        accuracy_threshold_minutes=15,
        enable_retraining=True,
        dashboard_enabled=True,
    )

    # This creates a tracking manager with integrated monitoring
    tracking_manager = create_monitoring_enhanced_tracking_manager(
        config=tracking_config
    )

    print("✅ Enhanced tracking manager created")

    # Demonstrate prediction recording with monitoring
    print("\n4. Recording Predictions with Monitoring...")

    rooms = ["living_room", "bedroom", "kitchen"]

    for i, room_id in enumerate(rooms):
        print(f"   Recording prediction for {room_id}...")

        # Create a sample prediction result
        prediction_result = PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=30 + i * 10),
            confidence=0.85 - i * 0.05,
            prediction_type="next_occupied" if i % 2 == 0 else "next_vacant",
            model_type="ensemble",
        )

        # Record prediction - this automatically triggers monitoring
        try:
            prediction_id = await tracking_manager.record_prediction(
                room_id=room_id, prediction_result=prediction_result
            )
            print(f"   ✅ Prediction recorded: {prediction_id}")

            # Simulate some processing time for different rooms
            await asyncio.sleep(0.1 + i * 0.05)

        except Exception as e:
            print(f"   ❌ Failed to record prediction: {e}")

    # Demonstrate manual monitoring operations
    print("\n5. Recording Additional Monitoring Data...")

    # Record feature computation timing
    tracking_manager.record_feature_computation(
        room_id="living_room", feature_type="temporal", duration=0.045
    )

    # Record database operation
    tracking_manager.record_database_operation(
        operation_type="SELECT", table="sensor_events", duration=0.023, status="success"
    )

    # Record MQTT publishing
    tracking_manager.record_mqtt_publish(
        topic_type="prediction", room_id="bedroom", status="success"
    )

    # Update connection status
    tracking_manager.update_connection_status(
        connection_type="home_assistant", connected=True
    )

    print("✅ Additional monitoring data recorded")

    # Demonstrate concept drift detection
    print("\n6. Simulating Concept Drift Detection...")

    tracking_manager.record_concept_drift(
        room_id="kitchen",
        drift_type="behavioral_change",
        severity=0.6,
        action_taken="model_retraining_scheduled",
    )

    print("✅ Concept drift recorded")

    # Get monitoring status
    print("\n7. Monitoring System Status:")
    status = await tracking_manager.get_monitoring_status()

    print(f"   System Health: {status['monitoring']['health_status']}")
    print(f"   Active Alerts: {status['monitoring']['alert_system']['active_alerts']}")
    print(
        f"   Metrics Collection: {'✅ Enabled' if status['monitoring']['metrics_collection']['enabled'] else '❌ Disabled'}"
    )
    print(f"   Integrated: {'✅ Yes' if status.get('integrated') else '❌ No'}")

    # Display some metrics
    print("\n8. Current Metrics Sample:")
    metrics_output = metrics_manager.get_metrics()

    # Parse a few key metrics for display
    metrics_lines = metrics_output.split("\n")
    prediction_metrics = [
        line
        for line in metrics_lines
        if "occupancy_predictor_predictions_total" in line and not line.startswith("#")
    ]

    if prediction_metrics:
        print("   Recent prediction metrics:")
        for metric in prediction_metrics[:3]:  # Show first 3
            print(f"     {metric}")
    else:
        print("   No prediction metrics available yet")

    # Demonstrate alert triggering
    print("\n9. Triggering Test Alert...")

    alert_id = await alert_manager.trigger_alert(
        rule_name="test_alert",
        title="Monitoring System Test",
        message="This is a test alert from the monitoring system integration demo",
        component="demo_system",
        room_id="living_room",
        context={
            "test": True,
            "demo_time": datetime.now().isoformat(),
            "severity": "info",
        },
    )

    if alert_id:
        print(f"✅ Alert triggered: {alert_id}")
    else:
        print("❌ Failed to trigger alert")

    # Wait a bit for background processing
    print("\n10. Waiting for Background Processing...")
    await asyncio.sleep(2)

    # Show performance summary
    print("\n11. Performance Summary:")
    performance_monitor = (
        monitoring_integration.get_monitoring_manager().get_performance_monitor()
    )
    summary = performance_monitor.get_performance_summary(hours=1)

    if summary:
        print(f"   Tracked metrics: {len(summary)}")
        for metric_name, stats in list(summary.items())[:3]:  # Show first 3
            print(f"   {metric_name}: mean={stats['mean']:.3f}, count={stats['count']}")
    else:
        print("   No performance data available yet")

    # Cleanup
    print("\n12. Stopping Monitoring System...")
    try:
        await tracking_manager.stop_tracking()
        await monitoring_integration.stop_monitoring()
        print("✅ Monitoring system stopped cleanly")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

    print("\n" + "=" * 70)
    print("🎉 Monitoring System Integration Demo Complete!")
    print("\nKey Features Demonstrated:")
    print("  • Automatic prediction tracking with performance metrics")
    print("  • Real-time system health monitoring")
    print("  • Prometheus metrics collection")
    print("  • Alert generation and management")
    print("  • Concept drift detection and logging")
    print("  • Integration with existing TrackingManager")
    print("\nFor production use:")
    print("  • Configure SMTP/webhook notifications in NotificationConfig")
    print("  • Set up Grafana dashboards with the provided JSON configs")
    print("  • Monitor the /metrics endpoint for Prometheus scraping")
    print("  • Review logs/ directory for structured JSON logs")


async def demonstrate_metrics_endpoint():
    """Show how to access the metrics endpoint programmatically."""
    print("\n" + "=" * 50)
    print("📊 Metrics Endpoint Demo")
    print("=" * 50)

    metrics_manager = get_metrics_manager()

    # Start metrics collection
    metrics_manager.start_background_collection()
    await asyncio.sleep(1)  # Let it collect some data

    # Get metrics
    metrics_output = metrics_manager.get_metrics()

    print(f"Metrics output length: {len(metrics_output)} characters")
    print("\nSample metrics (first 10 lines):")
    lines = metrics_output.split("\n")
    for i, line in enumerate(lines[:10]):
        if line and not line.startswith("#"):
            print(f"  {i+1}: {line}")

    # Stop collection
    metrics_manager.stop_background_collection()

    print("\n✅ Metrics endpoint demo complete")


if __name__ == "__main__":
    print("Starting Home Assistant ML Predictor Monitoring Integration Demo...")

    try:
        # Run the main demo
        asyncio.run(demonstrate_monitoring_integration())

        # Run the metrics endpoint demo
        asyncio.run(demonstrate_metrics_endpoint())

    except KeyboardInterrupt:
        print("\n\n⚠️  Demo interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Demo failed: {e}")
        import traceback

        traceback.print_exc()

    print("\n👋 Demo finished. Thank you!")
