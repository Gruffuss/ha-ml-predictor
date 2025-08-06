"""
DEPRECATED: System Integration Example - MQTT Publisher Infrastructure.

⚠️ This file is now DEPRECATED. MQTT publishing is now fully integrated
into TrackingManager and uses Enhanced MQTT Manager by default.

✅ USE INSTEAD: src/main_system.py - Complete integrated system
✅ AUTOMATIC OPERATION: Enhanced MQTT Manager starts automatically with TrackingManager
✅ NO MANUAL SETUP: Multi-channel publishing (MQTT, WebSocket, SSE) seamless

INTEGRATION STATUS:
- MQTT Publishing: ✅ Enhanced MQTT Manager integrated by default in TrackingManager
- Home Assistant Discovery: ✅ Automatic MQTT discovery configuration
- Multi-channel Broadcasting: ✅ MQTT + WebSocket + SSE automatically
- Prediction Publishing: ✅ Automatic when predictions are recorded

This file remains for reference only.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Any, Dict

from ..adaptation.tracking_manager import TrackingConfig, TrackingManager
from ..core.config import MQTTConfig, RoomConfig, get_config
from ..models.base.predictor import PredictionResult
from .mqtt_integration_manager import MQTTIntegrationManager

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def demonstrate_mqtt_integration():
    """
    Demonstrate the complete MQTT integration infrastructure.

    This shows how the system automatically publishes predictions to Home Assistant
    without any manual intervention required.
    """
    try:
        logger.info("=== MQTT Publisher Infrastructure Demonstration ===")

        # Load system configuration
        config = get_config()

        logger.info(f"Loaded configuration for {len(config.rooms)} rooms")
        logger.info(f"MQTT broker: {config.mqtt.broker}:{config.mqtt.port}")
        logger.info(f"MQTT discovery enabled: {config.mqtt.discovery_enabled}")
        logger.info(f"MQTT publishing enabled: {config.mqtt.publishing_enabled}")

        # Initialize MQTT Integration Manager
        logger.info("\n1. Initializing MQTT Integration Manager...")
        mqtt_manager = MQTTIntegrationManager(
            mqtt_config=config.mqtt, rooms=config.rooms
        )

        await mqtt_manager.initialize()
        logger.info("✅ MQTT Integration Manager initialized successfully")

        # Show integration stats
        stats = mqtt_manager.get_integration_stats()
        logger.info(f"MQTT connected: {stats['mqtt_connected']}")
        logger.info(f"Discovery published: {stats['discovery_published']}")
        logger.info(f"Rooms configured: {stats['rooms_configured']}")

        # Initialize Tracking Manager with MQTT integration
        logger.info("\n2. Initializing Tracking Manager with MQTT integration...")
        tracking_config = TrackingConfig(
            enabled=True, monitoring_interval_seconds=60, auto_validation_enabled=True
        )

        tracking_manager = TrackingManager(
            config=tracking_config,
            mqtt_integration_manager=mqtt_manager,  # This is the key integration!
        )

        await tracking_manager.initialize()
        logger.info("✅ Tracking Manager initialized with MQTT integration")

        # Demonstrate automatic prediction publishing
        logger.info("\n3. Demonstrating automatic prediction publishing...")

        # Get a sample room
        sample_room_id = list(config.rooms.keys())[0]
        sample_room = config.rooms[sample_room_id]

        # Create a mock prediction result
        mock_prediction = PredictionResult(
            predicted_time=datetime.utcnow() + timedelta(minutes=30),
            transition_type="vacant_to_occupied",
            confidence_score=0.85,
            alternatives=[(datetime.utcnow() + timedelta(minutes=25), 0.75)],
            model_type="ensemble",
            model_version="1.0.0",
            features_used=["time_since_last_change", "hour_of_day", "day_of_week"],
            prediction_metadata={
                "room_id": sample_room_id,
                "time_until_transition_seconds": 1800,
                "prediction_method": "stacking_ensemble",
                "base_model_predictions": {
                    "lstm": 1750.0,
                    "xgboost": 1850.0,
                    "hmm": 1800.0,
                },
                "model_weights": {"lstm": 0.35, "xgboost": 0.40, "hmm": 0.25},
            },
        )

        logger.info(f"Mock prediction for {sample_room.name}:")
        logger.info(f"  - Transition: {mock_prediction.transition_type}")
        logger.info(f"  - Time: {mock_prediction.predicted_time}")
        logger.info(f"  - Confidence: {mock_prediction.confidence_score:.2f}")

        # Record prediction - this will automatically publish to MQTT!
        logger.info("\n4. Recording prediction (automatic MQTT publishing)...")
        await tracking_manager.record_prediction(mock_prediction)

        logger.info(
            "✅ Prediction recorded and automatically published to Home Assistant!"
        )

        # Show updated stats
        updated_stats = mqtt_manager.get_integration_stats()
        logger.info(f"Predictions published: {updated_stats['predictions_published']}")
        logger.info(
            f"Last prediction published: {updated_stats['last_prediction_published']}"
        )

        # Demonstrate manual system status publishing
        logger.info("\n5. Publishing system status...")
        system_published = await mqtt_manager.publish_system_status(
            tracking_stats=await tracking_manager.get_tracking_status(),
            database_connected=True,
            active_alerts=0,
        )

        if system_published:
            logger.info("✅ System status published to Home Assistant")
        else:
            logger.warning("❌ Failed to publish system status")

        # Show final integration statistics
        logger.info("\n6. Final Integration Statistics:")
        final_stats = mqtt_manager.get_integration_stats()

        logger.info(f"✅ MQTT Integration Summary:")
        logger.info(f"  - Initialized: {final_stats['initialized']}")
        logger.info(f"  - MQTT Connected: {final_stats['mqtt_connected']}")
        logger.info(f"  - Discovery Published: {final_stats['discovery_published']}")
        logger.info(
            f"  - Predictions Published: {final_stats['predictions_published']}"
        )
        logger.info(
            f"  - Status Updates Published: {final_stats['status_updates_published']}"
        )
        logger.info(f"  - Total Errors: {final_stats['total_errors']}")
        logger.info(f"  - System Uptime: {final_stats['system_uptime_seconds']:.1f}s")

        logger.info(
            "\n🎉 MQTT Publisher Infrastructure demonstration completed successfully!"
        )
        logger.info("\nKey Benefits:")
        logger.info(
            "✅ Automatic prediction publishing - NO manual intervention required"
        )
        logger.info("✅ Seamless TrackingManager integration")
        logger.info("✅ Home Assistant MQTT discovery support")
        logger.info("✅ System status monitoring and publishing")
        logger.info("✅ Comprehensive error handling and resilience")
        logger.info("✅ Production-ready MQTT infrastructure")

        # Clean shutdown
        logger.info("\n7. Shutting down components...")
        await tracking_manager.stop_tracking()
        await mqtt_manager.stop_integration()
        logger.info("✅ Clean shutdown completed")

    except Exception as e:
        logger.error(f"❌ Demonstration failed: {e}")
        raise


def demonstrate_topic_structure():
    """Show the Home Assistant MQTT topic structure."""
    logger.info("\n=== Home Assistant MQTT Topic Structure ===")

    # Load config to show actual room structure
    config = get_config()

    logger.info("Prediction Topics:")
    for room_id, room_config in config.rooms.items():
        base_topic = f"{config.mqtt.topic_prefix}/{room_id}"
        logger.info(f"  Room: {room_config.name}")
        logger.info(f"    Main: {base_topic}/prediction")
        logger.info(f"    Legacy: {base_topic}/next_transition_time")
        logger.info(f"    Legacy: {base_topic}/confidence")
        logger.info(f"    Legacy: {base_topic}/time_until")

    logger.info("\nSystem Topics:")
    logger.info(f"  Status: {config.mqtt.topic_prefix}/system/status")

    logger.info("\nHome Assistant Discovery Topics:")
    for room_id, room_config in config.rooms.items():
        device_id = config.mqtt.device_identifier
        logger.info(f"  Room {room_config.name}:")
        logger.info(
            f"    Prediction: {config.mqtt.discovery_prefix}/sensor/{device_id}/{device_id}_{room_id}_prediction/config"
        )
        logger.info(
            f"    Confidence: {config.mqtt.discovery_prefix}/sensor/{device_id}/{device_id}_{room_id}_confidence/config"
        )
        logger.info(
            f"    Time Until: {config.mqtt.discovery_prefix}/sensor/{device_id}/{device_id}_{room_id}_time_until/config"
        )

    logger.info("\nExample Payload Structure:")
    example_payload = {
        "predicted_time": "2024-01-15T14:30:00",
        "transition_type": "vacant_to_occupied",
        "confidence_score": 0.85,
        "time_until_seconds": 1800,
        "time_until_human": "30 minutes",
        "model_type": "ensemble",
        "room_name": "Living Room",
        "prediction_reliability": "high",
        "base_predictions": {"lstm": 1750.0, "xgboost": 1850.0, "hmm": 1800.0},
        "model_weights": {"lstm": 0.35, "xgboost": 0.40, "hmm": 0.25},
    }

    import json

    logger.info(json.dumps(example_payload, indent=2))


if __name__ == "__main__":
    """Run the demonstration."""

    print("MQTT Publisher Infrastructure - Sprint 5 Task 1 Complete")
    print("=" * 60)

    # Show topic structure first
    demonstrate_topic_structure()

    # Run the full demonstration
    asyncio.run(demonstrate_mqtt_integration())
