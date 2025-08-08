"""
Comprehensive integration tests for Sprint 4: Self-Adaptation System.

These tests validate the complete integration of all Sprint 4 components:
- PredictionValidator: Prediction validation infrastructure
- AccuracyTracker: Real-time accuracy tracking and alerting
- ConceptDriftDetector: Statistical drift detection
- AdaptiveRetrainer: Automatic model retraining
- ModelOptimizer: Automatic parameter optimization
- PerformanceDashboard: Monitoring dashboard with API/WebSocket
- TrackingManager: System-wide coordination

Tests ensure all components work together as a unified self-adaptation system.
"""

import asyncio
import json
import logging
import tempfile
from datetime import datetime
from datetime import timedelta
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
import websockets

from src.adaptation.drift_detector import ConceptDriftDetector
from src.adaptation.drift_detector import DriftMetrics
from src.adaptation.drift_detector import DriftSeverity
from src.adaptation.drift_detector import DriftType
from src.adaptation.optimizer import ModelOptimizer
from src.adaptation.optimizer import OptimizationConfig
from src.adaptation.optimizer import OptimizationObjective
from src.adaptation.optimizer import OptimizationStrategy
from src.adaptation.retrainer import AdaptiveRetrainer
from src.adaptation.retrainer import RetrainingRequest
from src.adaptation.retrainer import RetrainingStatus
from src.adaptation.retrainer import RetrainingTrigger
from src.adaptation.tracker import AccuracyAlert
from src.adaptation.tracker import AccuracyTracker
from src.adaptation.tracker import AlertSeverity
from src.adaptation.tracker import RealTimeMetrics
from src.adaptation.tracker import TrendDirection
from src.adaptation.tracking_manager import TrackingConfig
from src.adaptation.tracking_manager import TrackingManager
# Import all Sprint 4 components
from src.adaptation.validator import AccuracyLevel
from src.adaptation.validator import AccuracyMetrics
from src.adaptation.validator import PredictionValidator
from src.adaptation.validator import ValidationRecord
from src.core.constants import ModelType
from src.core.exceptions import OccupancyPredictionError
from src.integration.dashboard import DashboardConfig
from src.integration.dashboard import DashboardMode
from src.integration.dashboard import PerformanceDashboard
from src.models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


@pytest.fixture
async def mock_database_manager():
    """Mock database manager for testing."""
    mock_db = AsyncMock()
    mock_db.get_sensor_events.return_value = []
    mock_db.get_room_states.return_value = []
    mock_db.health_check.return_value = {"status": "healthy"}
    return mock_db


@pytest.fixture
async def mock_model_registry():
    """Mock model registry with test models."""
    return {
        "living_room_ensemble": AsyncMock(),
        "bedroom_ensemble": AsyncMock(),
        "kitchen_ensemble": AsyncMock(),
    }


@pytest.fixture
async def mock_feature_engine():
    """Mock feature engineering engine."""
    mock_engine = AsyncMock()
    mock_engine.extract_features.return_value = {
        "temporal_features": {"time_since_last_change": 300},
        "sequential_features": {"room_transitions": 2},
        "contextual_features": {"temperature": 22.5},
    }
    return mock_engine


@pytest.fixture
async def tracking_config():
    """Standard tracking configuration for tests."""
    return TrackingConfig(
        enabled=True,
        monitoring_interval_seconds=1,  # Fast for testing
        auto_validation_enabled=True,
        validation_window_minutes=5,
        drift_detection_enabled=True,
        drift_check_interval_hours=1,
        adaptive_retraining_enabled=True,
        retraining_check_interval_hours=1,
        optimization_enabled=True,
        optimization_max_time_minutes=1,  # Quick optimization for tests
    )


@pytest.fixture
async def dashboard_config():
    """Dashboard configuration for tests."""
    return DashboardConfig(
        enabled=True,
        host="127.0.0.1",
        port=8889,  # Different port for tests
        debug=True,
        mode=DashboardMode.DEVELOPMENT,
        update_interval_seconds=1,
        websocket_enabled=True,
    )


@pytest.fixture
async def prediction_validator():
    """Initialized prediction validator."""
    return PredictionValidator(accuracy_threshold_minutes=15)


@pytest.fixture
async def integrated_tracking_system(
    tracking_config, mock_database_manager, mock_model_registry, mock_feature_engine
):
    """Fully integrated tracking system for testing."""
    # Create notification callback for testing
    alerts_received = []

    async def alert_callback(alert):
        alerts_received.append(alert)

    # Initialize tracking manager with all components
    tracking_manager = TrackingManager(
        config=tracking_config,
        database_manager=mock_database_manager,
        model_registry=mock_model_registry,
        feature_engineering_engine=mock_feature_engine,
        notification_callbacks=[alert_callback],
    )

    await tracking_manager.initialize()

    # Return system with reference to received alerts for testing
    tracking_manager._test_alerts_received = alerts_received

    yield tracking_manager

    # Cleanup
    await tracking_manager.stop_tracking()


@pytest.fixture
async def performance_dashboard(dashboard_config, integrated_tracking_system):
    """Performance dashboard integrated with tracking system."""
    dashboard = PerformanceDashboard(
        config=dashboard_config, tracking_manager=integrated_tracking_system
    )

    await dashboard.initialize()

    yield dashboard

    # Cleanup
    await dashboard.shutdown()


class TestSprintFourIntegrationComplete:
    """Complete integration tests for Sprint 4 self-adaptation system."""

    @pytest.mark.asyncio
    async def test_complete_prediction_lifecycle(self, integrated_tracking_system):
        """
        Test Scenario 1: Complete prediction lifecycle

        Tests the full flow: prediction → validation → accuracy tracking → alerting
        """
        logger.info("Testing complete prediction lifecycle integration")

        # Wait for system to start
        await asyncio.sleep(2)

        # Create test prediction
        prediction = PredictionResult(
            room_id="living_room",
            model_type=ModelType.ENSEMBLE,
            predicted_time=datetime.utcnow() + timedelta(minutes=30),
            confidence=0.85,
            transition_type="occupied_to_vacant",
            features_used={"time_since_last": 300},
            model_version="1.0.0",
        )

        # Record prediction through tracking manager
        await integrated_tracking_system.record_prediction(prediction)

        # Simulate actual room state change for validation
        actual_time = prediction.predicted_time + timedelta(minutes=2)  # 2 min error
        await integrated_tracking_system.validate_prediction_with_actual(
            prediction_id=prediction.prediction_id,
            actual_time=actual_time,
            actual_state="vacant",
        )

        # Wait for processing
        await asyncio.sleep(3)

        # Verify prediction was recorded and validated
        validator = integrated_tracking_system.validator
        assert validator is not None

        # Check validation record exists
        with validator._lock:
            validation_records = list(validator._validation_records.values())
            assert len(validation_records) > 0

            record = validation_records[0]
            assert record.room_id == "living_room"
            assert record.model_type == ModelType.ENSEMBLE
            assert record.prediction_time is not None
            assert record.validation_time is not None
            assert record.error_minutes == 2.0

        # Verify accuracy tracking is working
        tracker = integrated_tracking_system.accuracy_tracker
        assert tracker is not None

        metrics = await tracker.get_real_time_metrics(room_id="living_room")
        assert metrics is not None
        assert metrics.window_1h_predictions >= 1

        logger.info("✅ Complete prediction lifecycle integration test passed")

    @pytest.mark.asyncio
    async def test_drift_detection_triggers_retraining(
        self, integrated_tracking_system
    ):
        """
        Test Scenario 2: Drift detection triggering automatic retraining

        Tests drift detection → retraining trigger → model optimization integration
        """
        logger.info("Testing drift detection → retraining integration")

        # Wait for system initialization
        await asyncio.sleep(2)

        # Access components
        drift_detector = integrated_tracking_system.drift_detector
        retrainer = integrated_tracking_system.adaptive_retrainer
        assert drift_detector is not None
        assert retrainer is not None

        # Mock drift detection to return significant drift
        with patch.object(drift_detector, "detect_drift") as mock_detect:
            mock_detect.return_value = DriftMetrics(
                room_id="living_room",
                model_type=ModelType.ENSEMBLE,
                drift_detected=True,
                drift_type=DriftType.COVARIATE_SHIFT,
                drift_severity=DriftSeverity.HIGH,
                drift_score=0.45,  # Above threshold
                statistical_power=0.95,
                baseline_period_days=30,
                current_period_days=7,
                samples_analyzed=500,
                drift_features=["time_since_last_change", "room_transitions"],
                detection_method="kolmogorov_smirnov",
                p_value=0.001,
                confidence_level=0.99,
                details={
                    "ks_statistic": 0.45,
                    "psi_score": 0.35,
                    "significant_features": ["time_since_last_change"],
                },
            )

            # Mock model registry to simulate successful retraining
            mock_model = AsyncMock()
            mock_model.retrain.return_value = {"accuracy": 0.82, "loss": 0.15}
            integrated_tracking_system.model_registry["living_room_ensemble"] = (
                mock_model
            )

            # Trigger drift detection manually
            await integrated_tracking_system._drift_detection_loop_iteration()

            # Wait for retraining to be triggered
            await asyncio.sleep(3)

            # Verify retraining was triggered
            retraining_requests = await retrainer.get_retraining_status()

            # Check if any retraining was scheduled
            found_retraining = False
            for request in retraining_requests:
                if (
                    request.room_id == "living_room"
                    and RetrainingTrigger.CONCEPT_DRIFT in request.triggers
                ):
                    found_retraining = True
                    break

            assert found_retraining, "Drift detection should trigger retraining"

        logger.info("✅ Drift detection → retraining integration test passed")

    @pytest.mark.asyncio
    async def test_performance_dashboard_real_time_data(
        self, performance_dashboard, integrated_tracking_system
    ):
        """
        Test Scenario 3: Performance monitoring dashboard serving real-time data

        Tests dashboard integration with real-time tracking data
        """
        logger.info("Testing performance dashboard real-time data integration")

        # Wait for systems to initialize
        await asyncio.sleep(2)

        # Create some test data through tracking system
        prediction = PredictionResult(
            room_id="kitchen",
            model_type=ModelType.LSTM,
            predicted_time=datetime.utcnow() + timedelta(minutes=15),
            confidence=0.78,
            features_used={"temperature": 23.5},
        )

        await integrated_tracking_system.record_prediction(prediction)

        # Simulate validation
        await integrated_tracking_system.validate_prediction_with_actual(
            prediction_id=prediction.prediction_id,
            actual_time=prediction.predicted_time + timedelta(minutes=3),
            actual_state="occupied",
        )

        # Wait for data propagation
        await asyncio.sleep(3)

        # Test REST API endpoints
        if performance_dashboard.app:
            # Test metrics endpoint
            metrics_response = await performance_dashboard._get_system_metrics()
            assert metrics_response is not None
            assert "accuracy_metrics" in metrics_response
            assert "performance_metrics" in metrics_response

            # Test room-specific metrics
            room_metrics = await performance_dashboard._get_room_metrics("kitchen")
            assert room_metrics is not None
            assert room_metrics["room_id"] == "kitchen"

            # Test alerts endpoint
            alerts_response = await performance_dashboard._get_active_alerts()
            assert alerts_response is not None
            assert "alerts" in alerts_response

        logger.info("✅ Performance dashboard real-time data integration test passed")

    @pytest.mark.asyncio
    async def test_model_optimization_during_retraining(
        self, integrated_tracking_system
    ):
        """
        Test Scenario 4: Model optimization during retraining process

        Tests automatic parameter optimization integration with retraining
        """
        logger.info("Testing model optimization during retraining integration")

        # Wait for initialization
        await asyncio.sleep(2)

        optimizer = integrated_tracking_system.model_optimizer
        retrainer = integrated_tracking_system.adaptive_retrainer

        assert optimizer is not None
        assert retrainer is not None

        # Mock accuracy degradation to trigger retraining
        with patch.object(
            integrated_tracking_system.accuracy_tracker, "get_real_time_metrics"
        ) as mock_metrics:
            mock_metrics.return_value = RealTimeMetrics(
                room_id="bedroom",
                model_type=ModelType.XGBOOST.value,
                window_6h_accuracy=45.0,  # Below threshold
                window_6h_mean_error=28.0,  # Above threshold
                window_6h_predictions=20,
                accuracy_trend=TrendDirection.DEGRADING,
                trend_slope=-8.0,
            )

            # Mock model for optimization
            mock_model = AsyncMock()
            mock_model.get_params.return_value = {"learning_rate": 0.1, "max_depth": 6}
            mock_model.set_params.return_value = None
            mock_model.retrain.return_value = {"accuracy": 0.85}
            integrated_tracking_system.model_registry["bedroom_ensemble"] = mock_model

            # Mock optimization to return improved parameters
            with patch.object(optimizer, "optimize_model_parameters") as mock_optimize:
                mock_optimize.return_value = {
                    "optimization_successful": True,
                    "best_params": {"learning_rate": 0.05, "max_depth": 8},
                    "best_score": 0.87,
                    "improvement": 0.05,
                    "optimization_time_seconds": 45,
                }

                # Trigger retraining with optimization
                retraining_request = RetrainingRequest(
                    room_id="bedroom",
                    model_type=ModelType.XGBOOST,
                    triggers=[RetrainingTrigger.ACCURACY_DEGRADATION],
                    priority=1,
                    use_optimization=True,
                )

                await retrainer.add_retraining_request(retraining_request)

                # Wait for processing
                await asyncio.sleep(5)

                # Verify optimization was called
                mock_optimize.assert_called_once()

                # Verify model parameters were updated
                mock_model.set_params.assert_called_with(
                    learning_rate=0.05, max_depth=8
                )

        logger.info("✅ Model optimization during retraining integration test passed")

    @pytest.mark.asyncio
    async def test_alert_system_integration(self, integrated_tracking_system):
        """
        Test Scenario 5: Alert system integration across all components

        Tests alert generation, escalation, and notification across system
        """
        logger.info("Testing alert system integration")

        # Wait for initialization
        await asyncio.sleep(2)

        # Access test alerts collection
        alerts_received = integrated_tracking_system._test_alerts_received
        initial_alert_count = len(alerts_received)

        # Create conditions that should trigger alerts
        tracker = integrated_tracking_system.accuracy_tracker

        # Mock metrics that trigger critical alert
        with patch.object(tracker, "_calculate_real_time_metrics") as mock_calc:
            mock_calc.return_value = RealTimeMetrics(
                room_id="living_room",
                model_type=ModelType.ENSEMBLE.value,
                window_6h_accuracy=35.0,  # Critical threshold
                window_6h_mean_error=35.0,  # Critical threshold
                window_6h_predictions=15,
                accuracy_trend=TrendDirection.DEGRADING,
                trend_slope=-10.0,
                validation_lag_minutes=35.0,  # Critical threshold
            )

            # Force metrics update
            await tracker._update_real_time_metrics()

            # Force alert check
            await tracker._check_alert_conditions()

            # Wait for alert processing
            await asyncio.sleep(3)

            # Verify alerts were generated and delivered to callback
            assert len(alerts_received) > initial_alert_count

            # Check alert properties
            critical_alerts = [
                alert
                for alert in alerts_received[initial_alert_count:]
                if alert.severity == AlertSeverity.CRITICAL
            ]

            assert len(critical_alerts) > 0, "Should generate critical alerts"

            # Verify alert contains proper information
            alert = critical_alerts[0]
            assert alert.room_id == "living_room"
            assert alert.trigger_condition in [
                "accuracy_critical",
                "error_critical",
                "validation_lag_critical",
            ]
            assert alert.current_value is not None
            assert alert.threshold_value is not None
            assert not alert.resolved

        logger.info("✅ Alert system integration test passed")

    @pytest.mark.asyncio
    async def test_tracking_manager_coordination(self, integrated_tracking_system):
        """
        Test Scenario 6: TrackingManager coordination of all components

        Tests that TrackingManager properly coordinates all background tasks
        """
        logger.info("Testing TrackingManager coordination")

        # Verify all components are initialized
        assert integrated_tracking_system.validator is not None
        assert integrated_tracking_system.accuracy_tracker is not None
        assert integrated_tracking_system.drift_detector is not None
        assert integrated_tracking_system.adaptive_retrainer is not None
        assert integrated_tracking_system.model_optimizer is not None

        # Verify background tasks are running
        assert integrated_tracking_system._tracking_active
        assert len(integrated_tracking_system._background_tasks) > 0

        # Test system statistics
        stats = integrated_tracking_system.get_system_stats()
        assert stats is not None
        assert "tracking_active" in stats
        assert "total_predictions_recorded" in stats
        assert "components_status" in stats

        # Verify component coordination
        assert stats["tracking_active"] is True
        assert stats["components_status"]["validator"] == "active"
        assert stats["components_status"]["accuracy_tracker"] == "active"
        assert stats["components_status"]["drift_detector"] == "active"
        assert stats["components_status"]["adaptive_retrainer"] == "active"

        # Test prediction recording through manager
        prediction = PredictionResult(
            room_id="test_room",
            model_type=ModelType.HMM,
            predicted_time=datetime.utcnow() + timedelta(minutes=20),
            confidence=0.92,
        )

        initial_count = stats["total_predictions_recorded"]
        await integrated_tracking_system.record_prediction(prediction)

        # Wait for processing
        await asyncio.sleep(2)

        # Verify prediction was recorded
        updated_stats = integrated_tracking_system.get_system_stats()
        assert updated_stats["total_predictions_recorded"] > initial_count

        logger.info("✅ TrackingManager coordination test passed")

    @pytest.mark.asyncio
    async def test_configuration_system_integration(self, tracking_config):
        """
        Test Scenario 7: Configuration system works across all components

        Tests that configuration is properly applied to all integrated components
        """
        logger.info("Testing configuration system integration")

        # Test custom configuration
        custom_config = TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=2,
            alert_thresholds={
                "accuracy_warning": 80.0,  # Custom threshold
                "accuracy_critical": 60.0,
                "error_warning": 15.0,
                "error_critical": 25.0,
            },
            drift_detection_enabled=True,
            drift_psi_threshold=0.15,  # Custom drift threshold
            adaptive_retraining_enabled=True,
            retraining_accuracy_threshold=65.0,  # Custom retraining threshold
            optimization_enabled=True,
            optimization_strategy="bayesian",
        )

        # Create tracking manager with custom config
        tracking_manager = TrackingManager(
            config=custom_config,
            database_manager=AsyncMock(),
            model_registry={},
            feature_engineering_engine=AsyncMock(),
        )

        await tracking_manager.initialize()

        try:
            # Verify configuration is applied to components
            assert tracking_manager.config.alert_thresholds["accuracy_warning"] == 80.0
            assert tracking_manager.config.drift_psi_threshold == 0.15
            assert tracking_manager.config.retraining_accuracy_threshold == 65.0

            # Verify accuracy tracker uses custom thresholds
            tracker = tracking_manager.accuracy_tracker
            assert tracker.alert_thresholds["accuracy_warning"] == 80.0

            # Verify drift detector uses custom settings
            drift_detector = tracking_manager.drift_detector
            assert drift_detector.psi_threshold == 0.15

            # Verify retrainer uses custom thresholds
            retrainer = tracking_manager.adaptive_retrainer
            assert retrainer.config.retraining_accuracy_threshold == 65.0

        finally:
            await tracking_manager.stop_tracking()

        logger.info("✅ Configuration system integration test passed")

    @pytest.mark.asyncio
    async def test_system_resilience_and_error_handling(
        self, integrated_tracking_system
    ):
        """
        Test Scenario 8: System resilience and error handling

        Tests that the integrated system handles errors gracefully
        """
        logger.info("Testing system resilience and error handling")

        # Test with invalid prediction data
        invalid_prediction = PredictionResult(
            room_id="",  # Invalid empty room_id
            model_type=None,  # Invalid model type
            predicted_time=datetime.utcnow() - timedelta(hours=1),  # Past time
            confidence=1.5,  # Invalid confidence > 1.0
        )

        # Recording should handle gracefully without crashing
        try:
            await integrated_tracking_system.record_prediction(invalid_prediction)
        except Exception as e:
            logger.info(f"Expected error handled: {e}")

        # System should still be operational
        assert integrated_tracking_system._tracking_active

        # Test with database errors
        with patch.object(
            integrated_tracking_system.database_manager, "get_sensor_events"
        ) as mock_db:
            mock_db.side_effect = Exception("Database connection failed")

            # System should continue running despite database errors
            await asyncio.sleep(2)
            assert integrated_tracking_system._tracking_active

        # Test component recovery
        original_validator = integrated_tracking_system.validator
        integrated_tracking_system.validator = None

        # System should detect and handle missing component
        stats = integrated_tracking_system.get_system_stats()
        assert "error" in stats or stats["components_status"]["validator"] == "error"

        # Restore validator
        integrated_tracking_system.validator = original_validator

        logger.info("✅ System resilience and error handling test passed")


class TestSprintFourWebSocketIntegration:
    """Test WebSocket integration for real-time dashboard updates."""

    @pytest.mark.asyncio
    async def test_websocket_real_time_updates(self, performance_dashboard):
        """Test WebSocket real-time updates from dashboard."""
        if not performance_dashboard.config.websocket_enabled:
            pytest.skip("WebSocket not enabled")

        logger.info("Testing WebSocket real-time updates")

        # Mock WebSocket connection
        mock_websocket = AsyncMock()
        messages_sent = []

        async def mock_send(message):
            messages_sent.append(json.loads(message))

        mock_websocket.send = mock_send

        # Add mock connection to dashboard
        performance_dashboard._websocket_connections.add(mock_websocket)

        # Trigger update broadcast
        test_data = {
            "type": "metrics_update",
            "timestamp": datetime.utcnow().isoformat(),
            "data": {"system_health": 85.0, "active_alerts": 2},
        }

        await performance_dashboard._broadcast_to_websockets(test_data)

        # Verify message was sent
        assert len(messages_sent) == 1
        assert messages_sent[0]["type"] == "metrics_update"
        assert messages_sent[0]["data"]["system_health"] == 85.0

        logger.info("✅ WebSocket real-time updates test passed")


class TestSprintFourPerformanceValidation:
    """Performance validation tests for Sprint 4 integration."""

    @pytest.mark.asyncio
    async def test_system_performance_under_load(self, integrated_tracking_system):
        """Test system performance under prediction load."""
        logger.info("Testing system performance under load")

        # Generate multiple predictions quickly
        predictions = []
        for i in range(50):
            prediction = PredictionResult(
                room_id=f"room_{i % 5}",
                model_type=ModelType.ENSEMBLE,
                predicted_time=datetime.utcnow() + timedelta(minutes=i),
                confidence=0.8 + (i % 20) / 100,
                features_used={"load_test": i},
            )
            predictions.append(prediction)

        # Record all predictions
        start_time = datetime.utcnow()

        for prediction in predictions:
            await integrated_tracking_system.record_prediction(prediction)

        processing_time = (datetime.utcnow() - start_time).total_seconds()

        # Should process 50 predictions quickly (< 5 seconds)
        assert processing_time < 5.0, f"Processing too slow: {processing_time}s"

        # System should remain responsive
        stats = integrated_tracking_system.get_system_stats()
        assert stats["tracking_active"] is True

        # Wait for background processing
        await asyncio.sleep(3)

        # Verify all predictions were processed
        assert stats["total_predictions_recorded"] >= 50

        logger.info(
            f"✅ Performance test passed: {processing_time:.2f}s for 50 predictions"
        )

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self, integrated_tracking_system):
        """Test memory usage remains stable during extended operation."""
        logger.info("Testing memory usage stability")

        import gc
        import os

        import psutil

        # Get initial memory usage
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Generate continuous predictions for extended period
        for cycle in range(10):
            for i in range(20):
                prediction = PredictionResult(
                    room_id=f"room_{i % 3}",
                    model_type=ModelType.LSTM,
                    predicted_time=datetime.utcnow() + timedelta(minutes=i),
                    confidence=0.75,
                )
                await integrated_tracking_system.record_prediction(prediction)

            # Force garbage collection
            gc.collect()
            await asyncio.sleep(1)

        # Check final memory usage
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        logger.info(
            f"Memory usage: {initial_memory:.1f}MB → {final_memory:.1f}MB (+{memory_increase:.1f}MB)"
        )

        # Memory increase should be reasonable (< 50MB for this test)
        assert (
            memory_increase < 50
        ), f"Memory leak detected: {memory_increase:.1f}MB increase"

        logger.info("✅ Memory usage stability test passed")


# Helper functions for integration testing


def create_test_prediction(
    room_id: str = "test_room",
    model_type: ModelType = ModelType.ENSEMBLE,
    confidence: float = 0.85,
    minutes_ahead: int = 30,
) -> PredictionResult:
    """Create a test prediction result."""
    return PredictionResult(
        room_id=room_id,
        model_type=model_type,
        predicted_time=datetime.utcnow() + timedelta(minutes=minutes_ahead),
        confidence=confidence,
        features_used={"test_feature": 1.0},
    )


async def wait_for_background_processing(seconds: float = 2.0):
    """Wait for background processing to complete."""
    await asyncio.sleep(seconds)


def verify_component_integration(tracking_manager: TrackingManager) -> Dict[str, bool]:
    """Verify all components are properly integrated."""
    return {
        "validator_initialized": tracking_manager.validator is not None,
        "tracker_initialized": tracking_manager.accuracy_tracker is not None,
        "drift_detector_initialized": tracking_manager.drift_detector is not None,
        "retrainer_initialized": tracking_manager.adaptive_retrainer is not None,
        "optimizer_initialized": tracking_manager.model_optimizer is not None,
        "tracking_active": tracking_manager._tracking_active,
        "background_tasks_running": len(tracking_manager._background_tasks) > 0,
    }


if __name__ == "__main__":
    # Run integration tests
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])
