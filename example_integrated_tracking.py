#!/usr/bin/env python3
"""
Example: Integrated Accuracy Tracking System

This example demonstrates how the AccuracyTracker is now fully integrated
into the main prediction system for automatic tracking without manual setup.

The tracking system automatically:
1. Records predictions when ensemble models generate them
2. Validates predictions when actual room state changes occur
3. Provides real-time monitoring and alerting
4. Requires no manual intervention - it's seamless
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any
import pandas as pd
import numpy as np

# Import the integrated tracking system components
from src.core.config import get_config, TrackingConfig
from src.adaptation.tracking_manager import TrackingManager
from src.models.ensemble import OccupancyEnsemble
from src.data.ingestion.event_processor import EventProcessor
from src.data.storage.database import get_database_manager


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def create_sample_features(room_id: str, num_samples: int = 50) -> pd.DataFrame:
    """Create sample feature data for demonstration."""
    np.random.seed(42)
    
    # Create realistic occupancy prediction features
    features = {
        'time_since_last_change': np.random.exponential(1800, num_samples),  # seconds
        'current_state_duration': np.random.exponential(3600, num_samples),  # seconds
        'hour_sin': np.sin(2 * np.pi * np.random.randint(0, 24, num_samples) / 24),
        'hour_cos': np.cos(2 * np.pi * np.random.randint(0, 24, num_samples) / 24),
        'day_of_week': np.random.randint(0, 7, num_samples),
        'is_weekend': np.random.choice([0, 1], num_samples, p=[0.7, 0.3]),
        'room_transition_count': np.random.poisson(2, num_samples),
        'sensor_trigger_rate': np.random.gamma(2, 0.5, num_samples),
        'environmental_temp': np.random.normal(22, 3, num_samples),
        'environmental_humidity': np.random.normal(45, 10, num_samples),
    }
    
    # Add room-specific features
    for i in range(20):
        features[f'feature_{i}'] = np.random.normal(0, 1, num_samples)
    
    return pd.DataFrame(features)


async def create_sample_targets(num_samples: int = 50) -> pd.DataFrame:
    """Create sample target data for demonstration."""
    np.random.seed(42)
    
    # Time until next transition (in seconds)
    time_until_transition = np.random.exponential(1800, num_samples)  # ~30 min average
    time_until_transition = np.clip(time_until_transition, 60, 86400)  # 1 min to 1 day
    
    return pd.DataFrame({
        'time_until_transition_seconds': time_until_transition,
        'transition_type': np.random.choice(['vacant_to_occupied', 'occupied_to_vacant'], num_samples)
    })


async def simulate_room_state_changes(tracking_manager: TrackingManager, room_id: str):
    """Simulate actual room state changes for validation."""
    logger.info(f"Starting room state change simulation for {room_id}")
    
    # Simulate some room state changes over time
    states = ['vacant', 'occupied']
    current_state = 'vacant'
    
    for i in range(10):
        # Wait a bit between state changes
        await asyncio.sleep(2)
        
        # Change state
        new_state = 'occupied' if current_state == 'vacant' else 'vacant'
        change_time = datetime.utcnow()
        
        # Notify tracking manager of state change
        await tracking_manager.handle_room_state_change(
            room_id=room_id,
            new_state=new_state,
            change_time=change_time,
            previous_state=current_state
        )
        
        logger.info(f"Simulated state change: {room_id} {current_state} -> {new_state}")
        current_state = new_state


async def demonstrate_integrated_tracking():
    """Demonstrate the fully integrated accuracy tracking system."""
    logger.info("=== Integrated Accuracy Tracking System Demo ===")
    
    try:
        # 1. Initialize configuration with tracking enabled
        logger.info("1. Loading configuration with tracking enabled...")
        config = get_config()
        
        # Override tracking config for demo
        config.tracking = TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=30,
            auto_validation_enabled=True,
            validation_window_minutes=5,
            alert_thresholds={
                'accuracy_warning': 70.0,
                'accuracy_critical': 50.0,
                'error_warning': 15.0,
                'error_critical': 25.0,
            }
        )
        
        logger.info(f"Tracking configuration: enabled={config.tracking.enabled}, "
                   f"monitoring_interval={config.tracking.monitoring_interval_seconds}s")
        
        # 2. Initialize tracking manager
        logger.info("2. Initializing tracking manager...")
        tracking_manager = TrackingManager(
            config=config.tracking,
            database_manager=None,  # Would be real database manager in production
            notification_callbacks=[
                lambda alert: logger.warning(f"ALERT: {alert.description} (severity: {alert.severity.value})")
            ]
        )
        
        await tracking_manager.initialize()
        logger.info("Tracking manager initialized and monitoring started")
        
        # 3. Create ensemble with integrated tracking
        logger.info("3. Creating ensemble model with integrated tracking...")
        room_id = "living_room"
        ensemble = OccupancyEnsemble(
            room_id=room_id,
            tracking_manager=tracking_manager  # This enables automatic tracking!
        )
        
        # 4. Train the ensemble (required before prediction)
        logger.info("4. Training ensemble model...")
        features_df = await create_sample_features(room_id, 100)
        targets_df = await create_sample_targets(100)
        
        training_result = await ensemble.train(features_df, targets_df)
        logger.info(f"Training completed: R² = {training_result.training_score:.3f}")
        
        # 5. Create event processor with tracking integration
        logger.info("5. Creating event processor with tracking integration...")
        event_processor = EventProcessor(
            config=config,
            tracking_manager=tracking_manager  # This enables automatic validation!
        )
        
        # 6. Start room state change simulation
        logger.info("6. Starting room state change simulation...")
        simulation_task = asyncio.create_task(
            simulate_room_state_changes(tracking_manager, room_id)
        )
        
        # 7. Generate predictions with automatic tracking
        logger.info("7. Generating predictions with automatic tracking...")
        test_features = await create_sample_features(room_id, 5)
        
        for i in range(5):
            # Generate prediction - this automatically records it for tracking!
            prediction_results = await ensemble.predict(
                features=test_features.iloc[[i]],
                prediction_time=datetime.utcnow(),
                current_state='vacant' if i % 2 == 0 else 'occupied'
            )
            
            if prediction_results:
                result = prediction_results[0]
                logger.info(
                    f"Prediction {i+1}: {result.predicted_time.strftime('%H:%M:%S')} "
                    f"({result.transition_type}) confidence={result.confidence_score:.2f}"
                )
                
                # The prediction is automatically recorded by the tracking manager!
                # No manual tracking.record_prediction() call needed!
            
            await asyncio.sleep(1)
        
        # 8. Wait for some validation to occur
        logger.info("8. Waiting for validation and monitoring...")
        await asyncio.sleep(10)
        
        # 9. Check tracking status
        logger.info("9. Checking tracking system status...")
        status = await tracking_manager.get_tracking_status()
        logger.info(f"Tracking status: {status['tracking_active']}")
        logger.info(f"Predictions recorded: {status['performance']['total_predictions_recorded']}")
        logger.info(f"Validations performed: {status['performance']['total_validations_performed']}")
        
        # 10. Check real-time metrics
        logger.info("10. Checking real-time accuracy metrics...")
        metrics = await tracking_manager.get_real_time_metrics(room_id=room_id)
        if metrics:
            logger.info(f"Room {room_id} accuracy metrics:")
            logger.info(f"  1h accuracy: {metrics.window_1h_accuracy:.1f}%")
            logger.info(f"  6h accuracy: {metrics.window_6h_accuracy:.1f}%")
            logger.info(f"  24h accuracy: {metrics.window_24h_accuracy:.1f}%")
            logger.info(f"  Health score: {metrics.overall_health_score:.1f}/100")
            logger.info(f"  Is healthy: {metrics.is_healthy}")
        else:
            logger.info("No metrics available yet (insufficient data)")
        
        # 11. Check active alerts
        logger.info("11. Checking active alerts...")
        alerts = await tracking_manager.get_active_alerts(room_id=room_id)
        if alerts:
            logger.info(f"Active alerts for {room_id}: {len(alerts)}")
            for alert in alerts:
                logger.info(f"  - {alert.description} (severity: {alert.severity.value})")
        else:
            logger.info(f"No active alerts for {room_id}")
        
        # 12. Wait for simulation to complete
        await simulation_task
        
        # 13. Final status check
        logger.info("12. Final tracking system status...")
        final_status = await tracking_manager.get_tracking_status()
        logger.info(f"Final predictions recorded: {final_status['performance']['total_predictions_recorded']}")
        logger.info(f"Final validations performed: {final_status['performance']['total_validations_performed']}")
        
        # 14. Cleanup
        logger.info("13. Stopping tracking system...")
        await tracking_manager.stop_tracking()
        
        logger.info("=== Demo completed successfully! ===")
        
        print("\n" + "="*60)
        print("KEY INTEGRATION FEATURES DEMONSTRATED:")
        print("="*60)
        print("✅ Automatic prediction recording when ensemble.predict() is called")
        print("✅ Automatic validation when room state changes are detected")
        print("✅ Real-time monitoring and alerting without manual setup")
        print("✅ Seamless integration with existing prediction pipeline")
        print("✅ Configuration-driven tracking (can be enabled/disabled)")
        print("✅ Background monitoring tasks for continuous accuracy tracking")
        print("✅ No manual tracking code required in application logic")
        print("="*60)
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


async def main():
    """Main demo function."""
    try:
        await demonstrate_integrated_tracking()
    except KeyboardInterrupt:
        logger.info("Demo interrupted by user")
    except Exception as e:
        logger.error(f"Demo failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())