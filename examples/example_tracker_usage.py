"""
Example usage of the Real-time Accuracy Tracker system.

This script demonstrates how to use the AccuracyTracker with the PredictionValidator
for comprehensive real-time monitoring of prediction accuracy.
"""

import asyncio
from datetime import datetime, timedelta
import logging
from typing import Any, Dict

from src.adaptation.tracker import AccuracyTracker, AlertSeverity
from src.adaptation.validator import PredictionValidator
from src.models.base.predictor import PredictionResult

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Example notification callback
async def alert_notification_callback(alert):
    """Example callback for handling accuracy alerts."""
    print(
        f"""
🚨 ACCURACY ALERT 🚨
Alert ID: {alert.alert_id}
Room: {alert.room_id}
Model: {alert.model_type}
Severity: {alert.severity.value.upper()}
Condition: {alert.trigger_condition}
Description: {alert.description}
Current Value: {alert.current_value:.2f}
Threshold: {alert.threshold_value:.2f}
Age: {alert.age_minutes:.1f} minutes
"""
    )


def sync_notification_callback(alert):
    """Example synchronous callback for handling alerts."""
    logger.warning(f"SYNC ALERT: {alert.description} for {alert.room_id}")


async def demo_real_time_tracking():
    """Demonstrate real-time accuracy tracking capabilities."""

    print("=== Real-time Accuracy Tracking Demo ===\n")

    # Initialize the prediction validator
    print("1. Initializing PredictionValidator...")
    validator = PredictionValidator(
        accuracy_threshold_minutes=15,
        max_validation_delay_hours=6,
        max_memory_records=1000,
        cleanup_interval_hours=12,
    )

    # Initialize the accuracy tracker with custom thresholds
    print("2. Initializing AccuracyTracker...")
    tracker = AccuracyTracker(
        prediction_validator=validator,
        monitoring_interval_seconds=30,  # Monitor every 30 seconds
        alert_thresholds={
            "accuracy_warning": 75.0,  # Warning if accuracy < 75%
            "accuracy_critical": 60.0,  # Critical if accuracy < 60%
            "error_warning": 18.0,  # Warning if mean error > 18 min
            "error_critical": 25.0,  # Critical if mean error > 25 min
            "trend_degrading": -3.0,  # Warning if trend slope < -3%/hour
            "validation_lag_warning": 12.0,  # Warning if validation lag > 12 min
            "validation_lag_critical": 25.0,  # Critical if validation lag > 25 min
        },
        notification_callbacks=[
            alert_notification_callback,
            sync_notification_callback,
        ],
    )

    # Start background tasks
    print("3. Starting background monitoring tasks...")
    await validator.start_background_tasks()
    await tracker.start_monitoring()

    try:
        # Simulate some prediction recording and validation
        print("4. Simulating prediction recording and validation...\n")

        # Record some example predictions
        current_time = datetime.utcnow()

        for i in range(5):
            # Create a mock prediction result
            prediction = PredictionResult(
                predicted_time=current_time + timedelta(minutes=30),
                transition_type="occupied",
                confidence_score=0.85 - (i * 0.1),  # Decreasing confidence
                model_type="ensemble",
                model_version="1.0.0",
                prediction_interval=(
                    current_time + timedelta(minutes=25),
                    current_time + timedelta(minutes=35),
                ),
            )

            # Record prediction
            prediction_id = await validator.record_prediction(
                prediction=prediction,
                room_id=f"living_room_{i % 2}",  # Alternate between rooms
                feature_snapshot={"example_feature": 0.5 + i},
            )

            print(f"Recorded prediction {prediction_id}")

            # Simulate validation after some time (with varying accuracy)
            actual_time = current_time + timedelta(
                minutes=30 + (i * 5)
            )  # Increasing error

            await validator.validate_prediction(
                room_id=f"living_room_{i % 2}",
                actual_transition_time=actual_time,
                transition_type="occupied",
            )

            print(f"Validated prediction with {5 * i} minute error")

        # Wait for monitoring to process
        print("\n5. Waiting for real-time monitoring to process...")
        await asyncio.sleep(35)  # Wait for monitoring cycle

        # Get real-time metrics
        print("6. Retrieving real-time metrics...\n")

        # Global metrics
        global_metrics = await tracker.get_real_time_metrics()
        if global_metrics:
            print("=== GLOBAL METRICS ===")
            print(
                f"Overall Health Score: {global_metrics.overall_health_score:.1f}/100"
            )
            print(f"Is Healthy: {global_metrics.is_healthy}")
            print(f"24h Accuracy: {global_metrics.window_24h_accuracy:.1f}%")
            print(f"6h Mean Error: {global_metrics.window_6h_mean_error:.1f} minutes")
            print(f"Trend: {global_metrics.accuracy_trend.value}")
            print(f"Active Alerts: {len(global_metrics.active_alerts)}\n")

        # Room-specific metrics
        room_metrics = await tracker.get_real_time_metrics(room_id="living_room_0")
        if room_metrics:
            print("=== LIVING ROOM 0 METRICS ===")
            metrics_dict = (
                room_metrics.to_dict()
                if hasattr(room_metrics, "to_dict")
                else room_metrics
            )
            if isinstance(metrics_dict, dict):
                for key, value in metrics_dict.items():
                    if isinstance(value, dict):
                        print(f"{key.upper()}:")
                        for subkey, subvalue in value.items():
                            print(f"  {subkey}: {subvalue}")
                    else:
                        print(f"{key}: {value}")
            print()

        # Get active alerts
        print("7. Checking for active alerts...\n")
        active_alerts = await tracker.get_active_alerts()

        if active_alerts:
            print(f"=== ACTIVE ALERTS ({len(active_alerts)}) ===")
            for alert in active_alerts[:3]:  # Show first 3 alerts
                print(f"Alert: {alert.description}")
                print(f"Severity: {alert.severity.value}")
                print(f"Age: {alert.age_minutes:.1f} minutes")
                print(f"Requires Escalation: {alert.requires_escalation}")
                print("---")
        else:
            print("No active alerts found.")

        # Get accuracy trends
        print("\n8. Analyzing accuracy trends...")
        trends = await tracker.get_accuracy_trends()

        if trends.get("trends_by_entity"):
            print("=== ACCURACY TRENDS ===")
            for entity, trend_data in trends["trends_by_entity"].items():
                print(
                    f"{entity}: {trend_data.get('direction', 'unknown')} "
                    f"(slope: {trend_data.get('slope', 0):.2f})"
                )

        # Get tracker statistics
        print("\n9. Tracker system statistics...")
        stats = tracker.get_tracker_stats()

        print("=== TRACKER STATS ===")
        print(f"Monitoring Active: {stats.get('monitoring_active', False)}")
        print(f"Metrics Tracked: {stats.get('metrics_tracked', {})}")
        print(f"Active Alerts: {stats.get('alerts', {}).get('active', 0)}")
        print(f"Background Tasks: {stats.get('background_tasks', 0)}")

        # Export tracking data
        print("\n10. Exporting tracking data...")
        export_path = "tracking_export.json"
        records_exported = await tracker.export_tracking_data(
            output_path=export_path,
            include_alerts=True,
            include_trends=True,
            days_back=1,
        )
        print(f"Exported {records_exported} records to {export_path}")

        print("\n=== Demo completed successfully! ===")

    except Exception as e:
        logger.error(f"Demo failed: {e}")

    finally:
        # Clean shutdown
        print("\n11. Shutting down monitoring...")
        await tracker.stop_monitoring()
        await validator.stop_background_tasks()
        print("Shutdown complete.")


async def demo_alert_management():
    """Demonstrate alert management capabilities."""

    print("\n=== Alert Management Demo ===")

    # This would be run in a separate context to show alert management
    validator = PredictionValidator()
    tracker = AccuracyTracker(validator)

    # Start monitoring
    await tracker.start_monitoring()

    try:
        # Wait for some alerts to be generated
        await asyncio.sleep(10)

        # Get and acknowledge alerts
        alerts = await tracker.get_active_alerts(severity=AlertSeverity.WARNING)

        if alerts:
            alert = alerts[0]
            print(f"Acknowledging alert: {alert.alert_id}")
            success = await tracker.acknowledge_alert(alert.alert_id, "demo_user")
            print(f"Acknowledgment successful: {success}")

            # Show alert details
            alert_dict = alert.to_dict()
            print("Alert details:")
            for key, value in alert_dict.items():
                print(f"  {key}: {value}")

    finally:
        await tracker.stop_monitoring()


if __name__ == "__main__":
    print("Starting Real-time Accuracy Tracking Demo")
    print("This demonstrates integration with PredictionValidator\n")

    # Run the main demo
    asyncio.run(demo_real_time_tracking())

    print("\nDemo completed. Check the logs for detailed output.")
