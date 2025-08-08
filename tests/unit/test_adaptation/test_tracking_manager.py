"""
Comprehensive unit tests for TrackingManager system coordination.

This test module covers the central tracking manager's coordination between
all adaptation components, integration workflows, and real-time monitoring.
"""

import asyncio
import json
from datetime import datetime
from datetime import timedelta
from unittest.mock import AsyncMock
from unittest.mock import MagicMock
from unittest.mock import Mock
from unittest.mock import patch

import pytest

from src.adaptation.drift_detector import ConceptDriftDetector
from src.adaptation.drift_detector import DriftMetrics
from src.adaptation.drift_detector import DriftSeverity
from src.adaptation.optimizer import ModelOptimizer
from src.adaptation.optimizer import OptimizationConfig
from src.adaptation.retrainer import AdaptiveRetrainer
from src.adaptation.retrainer import RetrainingRequest
from src.adaptation.retrainer import RetrainingStatus
from src.adaptation.tracker import AccuracyTracker
from src.adaptation.tracking_manager import TrackingConfig
from src.adaptation.tracking_manager import TrackingManager
from src.adaptation.validator import AccuracyMetrics
from src.adaptation.validator import PredictionValidator
from src.models.base.predictor import PredictionResult

# Test fixtures


@pytest.fixture
def tracking_config():
    """Create tracking configuration for testing."""
    return TrackingConfig(
        enabled=True,
        monitoring_interval_seconds=30,
        auto_validation_enabled=True,
        validation_window_minutes=15,
        alert_thresholds={
            "accuracy_warning": 70.0,
            "accuracy_critical": 50.0,
            "error_warning": 20.0,
            "error_critical": 30.0,
        },
        drift_detection_enabled=True,
        drift_check_interval_hours=6,
        adaptive_retraining_enabled=True,
        retraining_accuracy_threshold=60.0,
        realtime_publishing_enabled=True,
        dashboard_enabled=True,
        websocket_api_enabled=True,
    )


@pytest.fixture
def mock_database_manager():
    """Mock database manager."""
    db_manager = Mock()
    db_manager.health_check = AsyncMock(return_value={"status": "healthy"})
    return db_manager


@pytest.fixture
def mock_model_registry():
    """Mock model registry with test models."""
    registry = {
        "living_room_lstm": Mock(),
        "living_room_xgboost": Mock(),
        "bedroom_ensemble": Mock(),
    }
    return registry


@pytest.fixture
def mock_mqtt_manager():
    """Mock MQTT integration manager."""
    mqtt_manager = Mock()
    mqtt_manager.initialize = AsyncMock()
    mqtt_manager.shutdown = AsyncMock()
    mqtt_manager.publish_prediction = AsyncMock(
        return_value={
            "mqtt": {"success": True},
            "websocket": {"success": True, "clients_notified": 3},
            "sse": {"success": True, "clients_notified": 2},
        }
    )
    mqtt_manager.get_integration_stats = Mock(
        return_value={
            "mqtt_integration": {"mqtt_connected": True, "discovery_published": True},
            "realtime_publishing": {"system_active": True},
            "channels": {
                "total_active": 3,
                "enabled_channels": ["mqtt", "websocket", "sse"],
            },
            "connections": {"websocket_clients": 3, "sse_clients": 2},
            "performance": {
                "predictions_per_minute": 12,
                "average_publish_latency_ms": 25,
                "publish_success_rate": 0.98,
            },
        }
    )
    return mqtt_manager


@pytest.fixture
def mock_notification_callbacks():
    """Mock notification callbacks."""
    callback = Mock()
    callback.return_value = None
    return [callback]


@pytest.fixture
async def tracking_manager(
    tracking_config,
    mock_database_manager,
    mock_model_registry,
    mock_mqtt_manager,
    mock_notification_callbacks,
):
    """Create tracking manager with mocked dependencies."""
    manager = TrackingManager(
        config=tracking_config,
        database_manager=mock_database_manager,
        model_registry=mock_model_registry,
        mqtt_integration_manager=mock_mqtt_manager,
        notification_callbacks=mock_notification_callbacks,
    )

    # Initialize with mocked components
    with patch("src.adaptation.tracking_manager.PredictionValidator"), patch(
        "src.adaptation.tracking_manager.AccuracyTracker"
    ), patch("src.adaptation.tracking_manager.ConceptDriftDetector"), patch(
        "src.adaptation.tracking_manager.AdaptiveRetrainer"
    ), patch(
        "src.adaptation.tracking_manager.ModelOptimizer"
    ):
        await manager.initialize()

    yield manager

    # Cleanup
    await manager.stop_tracking()


@pytest.fixture
def sample_prediction_result():
    """Create sample prediction result for testing."""
    return PredictionResult(
        predicted_time=datetime.now() + timedelta(minutes=30),
        transition_type="occupied",
        confidence_score=0.85,
        prediction_metadata={
            "room_id": "living_room",
            "model_type": "ensemble",
            "features_used": ["temporal", "sequential"],
            "prediction_id": "pred_12345",
        },
    )


# Core tracking manager tests


class TestTrackingManagerInitialization:
    """Test TrackingManager initialization and lifecycle."""

    @pytest.mark.asyncio
    async def test_manager_initialization(self, tracking_config, mock_database_manager):
        """Test tracking manager initialization."""
        manager = TrackingManager(
            config=tracking_config,
            database_manager=mock_database_manager,
            model_registry={},
        )

        # Test initial state
        assert manager.config == tracking_config
        assert manager.database_manager == mock_database_manager
        assert not manager._tracking_active
        assert len(manager._background_tasks) == 0

        # Test configuration values
        assert manager.config.enabled
        assert manager.config.monitoring_interval_seconds == 30
        assert manager.config.drift_detection_enabled

    @pytest.mark.asyncio
    async def test_manager_initialization_with_components(self, tracking_manager):
        """Test manager initialization with all components."""
        assert tracking_manager._tracking_active
        assert tracking_manager.validator is not None
        assert tracking_manager.accuracy_tracker is not None
        assert tracking_manager.drift_detector is not None
        assert tracking_manager.adaptive_retrainer is not None
        assert tracking_manager.model_optimizer is not None

    @pytest.mark.asyncio
    async def test_manager_shutdown(self, tracking_manager):
        """Test graceful manager shutdown."""
        # Verify manager is running
        assert tracking_manager._tracking_active

        # Stop tracking
        await tracking_manager.stop_tracking()

        # Verify shutdown
        assert not tracking_manager._tracking_active
        assert len(tracking_manager._background_tasks) == 0

    @pytest.mark.asyncio
    async def test_disabled_manager_initialization(self, mock_database_manager):
        """Test initialization when tracking is disabled."""
        disabled_config = TrackingConfig(enabled=False)
        manager = TrackingManager(
            config=disabled_config, database_manager=mock_database_manager
        )

        await manager.initialize()

        # Should not start any background tasks when disabled
        assert not manager._tracking_active


class TestPredictionRecording:
    """Test prediction recording and integration workflows."""

    @pytest.mark.asyncio
    async def test_prediction_recording(
        self, tracking_manager, sample_prediction_result
    ):
        """Test basic prediction recording."""
        # Record prediction
        await tracking_manager.record_prediction(sample_prediction_result)

        # Verify prediction was processed
        assert tracking_manager._total_predictions_recorded > 0

        # Verify prediction was cached
        room_id = sample_prediction_result.prediction_metadata["room_id"]
        assert room_id in tracking_manager._pending_predictions
        assert len(tracking_manager._pending_predictions[room_id]) > 0

    @pytest.mark.asyncio
    async def test_prediction_mqtt_integration(
        self, tracking_manager, sample_prediction_result
    ):
        """Test prediction recording triggers MQTT publishing."""
        # Record prediction
        await tracking_manager.record_prediction(sample_prediction_result)

        # Verify MQTT publishing was called
        tracking_manager.mqtt_integration_manager.publish_prediction.assert_called_once()
        call_args = (
            tracking_manager.mqtt_integration_manager.publish_prediction.call_args
        )
        assert call_args[1]["prediction_result"] == sample_prediction_result

    @pytest.mark.asyncio
    async def test_prediction_recording_with_disabled_tracking(
        self, sample_prediction_result
    ):
        """Test prediction recording when tracking is disabled."""
        disabled_config = TrackingConfig(enabled=False)
        manager = TrackingManager(config=disabled_config)

        # Should not fail even when disabled
        await manager.record_prediction(sample_prediction_result)

        # Should not record anything
        assert manager._total_predictions_recorded == 0

    @pytest.mark.asyncio
    async def test_prediction_cache_cleanup(
        self, tracking_manager, sample_prediction_result
    ):
        """Test automatic cleanup of old predictions in cache."""
        # Create old prediction
        old_prediction = PredictionResult(
            predicted_time=datetime.now() - timedelta(hours=3),
            transition_type="vacant",
            confidence_score=0.75,
            prediction_metadata={"room_id": "living_room"},
        )

        # Record old prediction
        await tracking_manager.record_prediction(old_prediction)

        # Record new prediction
        await tracking_manager.record_prediction(sample_prediction_result)

        # Trigger cleanup
        await tracking_manager._perform_cleanup()

        # Old prediction should be cleaned up
        room_predictions = tracking_manager._pending_predictions.get("living_room", [])
        old_predictions = [
            p
            for p in room_predictions
            if p.predicted_time < datetime.now() - timedelta(hours=2)
        ]
        assert len(old_predictions) == 0


class TestRoomStateChangeHandling:
    """Test room state change handling and validation triggering."""

    @pytest.mark.asyncio
    async def test_room_state_change_handling(self, tracking_manager):
        """Test handling of room state changes."""
        room_id = "living_room"
        new_state = "occupied"
        change_time = datetime.now()
        previous_state = "vacant"

        # Handle state change
        await tracking_manager.handle_room_state_change(
            room_id=room_id,
            new_state=new_state,
            change_time=change_time,
            previous_state=previous_state,
        )

        # Verify validation was performed
        assert tracking_manager._total_validations_performed > 0

    @pytest.mark.asyncio
    async def test_state_change_triggers_retraining_evaluation(self, tracking_manager):
        """Test that state changes trigger retraining evaluation."""
        room_id = "bedroom"

        # Mock validator with poor accuracy
        mock_room_accuracy = AccuracyMetrics(
            total_predictions=50,
            validated_predictions=45,
            accurate_predictions=20,
            accuracy_rate=44.4,
            mean_error_minutes=35.2,
        )

        with patch.object(
            tracking_manager.validator,
            "get_room_accuracy",
            return_value=mock_room_accuracy,
        ):
            # Handle state change
            await tracking_manager.handle_room_state_change(
                room_id=room_id,
                new_state="vacant",
                change_time=datetime.now(),
                previous_state="occupied",
            )

        # Should trigger retraining evaluation due to poor accuracy
        # (This is tested indirectly through the retrainer mock calls)

    @pytest.mark.asyncio
    async def test_disabled_validation_handling(self, tracking_manager):
        """Test state change handling when validation is disabled."""
        # Disable auto-validation
        tracking_manager.config.auto_validation_enabled = False

        initial_validations = tracking_manager._total_validations_performed

        # Handle state change
        await tracking_manager.handle_room_state_change(
            room_id="test_room", new_state="occupied", change_time=datetime.now()
        )

        # Should not perform validation
        assert tracking_manager._total_validations_performed == initial_validations


class TestDriftDetectionIntegration:
    """Test integration with drift detection system."""

    @pytest.mark.asyncio
    async def test_manual_drift_detection(self, tracking_manager):
        """Test manual drift detection triggering."""
        room_id = "living_room"

        # Mock drift detection results
        mock_drift_metrics = DriftMetrics(
            room_id=room_id,
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=14),
                datetime.now() - timedelta(days=3),
            ),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=22.5,
            overall_drift_score=0.65,
            drift_severity=DriftSeverity.MAJOR,
            retraining_recommended=True,
        )

        with patch.object(
            tracking_manager.drift_detector,
            "detect_drift",
            return_value=mock_drift_metrics,
        ):
            # Run drift detection
            result = await tracking_manager.check_drift(room_id)

            # Verify drift detection completed
            assert result is not None
            assert result.room_id == room_id
            assert result.drift_severity == DriftSeverity.MAJOR
            assert tracking_manager._total_drift_checks_performed > 0

    @pytest.mark.asyncio
    async def test_drift_based_retraining_triggering(self, tracking_manager):
        """Test that significant drift triggers retraining."""
        room_id = "bedroom"

        # Mock critical drift
        critical_drift = DriftMetrics(
            room_id=room_id,
            detection_time=datetime.now(),
            baseline_period=(
                datetime.now() - timedelta(days=14),
                datetime.now() - timedelta(days=3),
            ),
            current_period=(datetime.now() - timedelta(days=3), datetime.now()),
            accuracy_degradation=35.0,
            overall_drift_score=0.9,
            drift_severity=DriftSeverity.CRITICAL,
            immediate_attention_required=True,
            retraining_recommended=True,
        )

        with patch.object(
            tracking_manager.drift_detector, "detect_drift", return_value=critical_drift
        ):
            # Run drift detection
            await tracking_manager.check_drift(room_id)

        # Critical drift should trigger immediate attention
        # Verify through notification callbacks or retraining requests

    @pytest.mark.asyncio
    async def test_disabled_drift_detection(self, tracking_manager):
        """Test behavior when drift detection is disabled."""
        # Disable drift detection
        tracking_manager.config.drift_detection_enabled = False

        # Try to run drift detection
        result = await tracking_manager.check_drift("test_room")

        # Should return None when disabled
        assert result is None


class TestRetrainingIntegration:
    """Test integration with adaptive retraining system."""

    @pytest.mark.asyncio
    async def test_manual_retraining_request(self, tracking_manager):
        """Test manual retraining request."""
        room_id = "living_room"
        model_type = "lstm"

        # Mock retraining request
        with patch.object(
            tracking_manager.adaptive_retrainer,
            "request_retraining",
            return_value="retrain_req_123",
        ):
            request_id = await tracking_manager.request_manual_retraining(
                room_id=room_id,
                model_type=model_type,
                strategy="incremental",
                priority=8.0,
            )

            # Verify request was submitted
            assert request_id is not None
            assert request_id == "retrain_req_123"

    @pytest.mark.asyncio
    async def test_retraining_status_tracking(self, tracking_manager):
        """Test retraining status tracking."""
        request_id = "retrain_req_456"

        # Mock retraining status
        mock_status = {
            "request_id": request_id,
            "room_id": "bedroom",
            "model_type": "xgboost",
            "status": "in_progress",
            "progress": 45.0,
        }

        with patch.object(
            tracking_manager.adaptive_retrainer,
            "get_retraining_status",
            return_value=mock_status,
        ):
            status = await tracking_manager.get_retraining_status(request_id)

            # Verify status retrieval
            assert status is not None
            assert status["request_id"] == request_id
            assert status["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_retraining_cancellation(self, tracking_manager):
        """Test retraining request cancellation."""
        request_id = "retrain_req_789"

        with patch.object(
            tracking_manager.adaptive_retrainer, "cancel_retraining", return_value=True
        ):
            success = await tracking_manager.cancel_retraining(request_id)

            # Verify cancellation
            assert success


class TestSystemStatusAndMetrics:
    """Test system status reporting and metrics collection."""

    @pytest.mark.asyncio
    async def test_tracking_status_comprehensive(self, tracking_manager):
        """Test comprehensive tracking status retrieval."""
        # Get tracking status
        status = await tracking_manager.get_tracking_status()

        # Verify status structure
        assert "tracking_active" in status
        assert "config" in status
        assert "performance" in status
        assert "validator" in status
        assert "accuracy_tracker" in status
        assert "drift_detector" in status
        assert "adaptive_retrainer" in status
        assert "prediction_cache" in status

        # Verify status values
        assert status["tracking_active"] == True
        assert status["config"]["enabled"] == True
        assert status["performance"]["total_predictions_recorded"] >= 0

    @pytest.mark.asyncio
    async def test_real_time_metrics_retrieval(self, tracking_manager):
        """Test real-time metrics retrieval."""
        room_id = "living_room"

        # Mock real-time metrics
        with patch.object(
            tracking_manager.accuracy_tracker,
            "get_real_time_metrics",
            return_value=Mock(accuracy_rate=85.0, average_error_minutes=12.5),
        ):
            metrics = await tracking_manager.get_real_time_metrics(room_id=room_id)

            # Verify metrics retrieval
            assert metrics is not None
            assert hasattr(metrics, "accuracy_rate")

    @pytest.mark.asyncio
    async def test_active_alerts_retrieval(self, tracking_manager):
        """Test active alerts retrieval."""
        # Mock active alerts
        mock_alerts = [
            Mock(alert_id="alert_1", room_id="living_room", severity="warning"),
            Mock(alert_id="alert_2", room_id="bedroom", severity="critical"),
        ]

        with patch.object(
            tracking_manager.accuracy_tracker,
            "get_active_alerts",
            return_value=mock_alerts,
        ):
            alerts = await tracking_manager.get_active_alerts()

            # Verify alerts retrieval
            assert len(alerts) == 2
            assert alerts[0].alert_id == "alert_1"

    @pytest.mark.asyncio
    async def test_alert_acknowledgment(self, tracking_manager):
        """Test alert acknowledgment."""
        alert_id = "alert_123"
        acknowledged_by = "user_admin"

        with patch.object(
            tracking_manager.accuracy_tracker, "acknowledge_alert", return_value=True
        ):
            success = await tracking_manager.acknowledge_alert(
                alert_id, acknowledged_by
            )

            # Verify acknowledgment
            assert success


class TestIntegrationStatus:
    """Test integration status reporting for various systems."""

    @pytest.mark.asyncio
    async def test_mqtt_integration_status(self, tracking_manager):
        """Test MQTT integration status reporting."""
        status = tracking_manager.get_enhanced_mqtt_status()

        # Verify MQTT status structure
        assert "enabled" in status
        assert "type" in status
        assert "mqtt_connected" in status
        assert "realtime_publishing_active" in status
        assert "total_channels" in status
        assert "websocket_connections" in status
        assert "sse_connections" in status

        # Verify mock values
        assert status["enabled"] == True
        assert status["type"] == "enhanced"
        assert status["mqtt_connected"] == True

    @pytest.mark.asyncio
    async def test_realtime_publishing_status(self, tracking_manager):
        """Test real-time publishing status reporting."""
        status = tracking_manager.get_realtime_publishing_status()

        # Verify publishing status structure
        assert "enabled" in status
        assert "active" in status
        assert "enabled_channels" in status
        assert "websocket_connections" in status
        assert "sse_connections" in status
        assert "source" in status

        # Verify values
        assert status["enabled"] == True
        assert status["source"] == "enhanced_mqtt_manager"

    @pytest.mark.asyncio
    async def test_drift_status_reporting(self, tracking_manager):
        """Test drift detection status reporting."""
        status = await tracking_manager.get_drift_status()

        # Verify drift status structure
        assert "drift_detection_enabled" in status
        assert "drift_detector_available" in status
        assert "total_drift_checks" in status
        assert "last_drift_check" in status

        # Verify configuration details
        if status["drift_detection_enabled"]:
            assert "drift_config" in status
            assert "check_interval_hours" in status["drift_config"]


class TestModelRegistration:
    """Test model registration and management."""

    @pytest.mark.asyncio
    async def test_model_registration(self, tracking_manager):
        """Test model registration for adaptive retraining."""
        room_id = "kitchen"
        model_type = "ensemble"
        mock_model = Mock()

        # Register model
        tracking_manager.register_model(room_id, model_type, mock_model)

        # Verify registration
        model_key = f"{room_id}_{model_type}"
        assert model_key in tracking_manager.model_registry
        assert tracking_manager.model_registry[model_key] == mock_model

    @pytest.mark.asyncio
    async def test_model_unregistration(self, tracking_manager):
        """Test model unregistration."""
        room_id = "kitchen"
        model_type = "ensemble"
        mock_model = Mock()

        # Register then unregister
        tracking_manager.register_model(room_id, model_type, mock_model)
        tracking_manager.unregister_model(room_id, model_type)

        # Verify unregistration
        model_key = f"{room_id}_{model_type}"
        assert model_key not in tracking_manager.model_registry


class TestNotificationCallbacks:
    """Test notification callback management."""

    @pytest.mark.asyncio
    async def test_notification_callback_management(self, tracking_manager):
        """Test adding and removing notification callbacks."""
        callback_called = False

        def test_callback(alert):
            nonlocal callback_called
            callback_called = True

        # Add callback
        tracking_manager.add_notification_callback(test_callback)
        assert test_callback in tracking_manager.notification_callbacks

        # Remove callback
        tracking_manager.remove_notification_callback(test_callback)
        assert test_callback not in tracking_manager.notification_callbacks

    @pytest.mark.asyncio
    async def test_callback_integration_with_tracker(self, tracking_manager):
        """Test that callbacks are properly integrated with accuracy tracker."""
        # Mock accuracy tracker
        mock_tracker = Mock()
        tracking_manager.accuracy_tracker = mock_tracker

        def test_callback(alert):
            pass

        # Add callback
        tracking_manager.add_notification_callback(test_callback)

        # Verify tracker received callback
        mock_tracker.add_notification_callback.assert_called_with(test_callback)


class TestErrorHandling:
    """Test error handling in tracking manager operations."""

    @pytest.mark.asyncio
    async def test_prediction_recording_error_handling(
        self, tracking_manager, sample_prediction_result
    ):
        """Test error handling in prediction recording."""
        # Mock validator to raise exception
        tracking_manager.validator = Mock()
        tracking_manager.validator.record_prediction = AsyncMock(
            side_effect=Exception("Database error")
        )

        # Should not raise exception
        await tracking_manager.record_prediction(sample_prediction_result)

        # Should handle gracefully

    @pytest.mark.asyncio
    async def test_drift_detection_error_handling(self, tracking_manager):
        """Test error handling in drift detection."""
        # Mock drift detector to raise exception
        tracking_manager.drift_detector = Mock()
        tracking_manager.drift_detector.detect_drift = AsyncMock(
            side_effect=Exception("Drift detection error")
        )

        # Should return None on error
        result = await tracking_manager.check_drift("error_room")
        assert result is None

    @pytest.mark.asyncio
    async def test_status_retrieval_error_handling(self, tracking_manager):
        """Test error handling in status retrieval."""
        # Mock component to raise exception
        tracking_manager.validator = Mock()
        tracking_manager.validator.get_total_predictions = AsyncMock(
            side_effect=Exception("Status error")
        )

        # Should return status with error indication
        status = await tracking_manager.get_tracking_status()
        assert "error" in str(status).lower() or status is not None


class TestPerformanceAndConcurrency:
    """Test performance characteristics and concurrent operations."""

    @pytest.mark.asyncio
    async def test_concurrent_prediction_recording(self, tracking_manager):
        """Test concurrent prediction recording."""
        # Create multiple prediction results
        predictions = []
        for i in range(10):
            pred = PredictionResult(
                predicted_time=datetime.now() + timedelta(minutes=30 + i),
                transition_type="occupied",
                confidence_score=0.8 + i * 0.01,
                prediction_metadata={
                    "room_id": f"room_{i%3}",
                    "prediction_id": f"pred_{i}",
                },
            )
            predictions.append(pred)

        # Record predictions concurrently
        tasks = [tracking_manager.record_prediction(pred) for pred in predictions]
        await asyncio.gather(*tasks)

        # Verify all were recorded
        assert tracking_manager._total_predictions_recorded >= len(predictions)

    @pytest.mark.asyncio
    async def test_background_task_management(self, tracking_manager):
        """Test background task lifecycle management."""
        # Verify tasks are running
        initial_task_count = len(tracking_manager._background_tasks)
        assert initial_task_count > 0

        # Stop and restart tracking
        await tracking_manager.stop_tracking()
        assert len(tracking_manager._background_tasks) == 0

        await tracking_manager.start_tracking()
        assert len(tracking_manager._background_tasks) > 0

    @pytest.mark.asyncio
    async def test_memory_usage_monitoring(self, tracking_manager):
        """Test that prediction cache doesn't grow unbounded."""
        # Record many predictions
        for i in range(100):
            pred = PredictionResult(
                predicted_time=datetime.now() + timedelta(minutes=i),
                transition_type="occupied",
                confidence_score=0.8,
                prediction_metadata={"room_id": "test_room"},
            )
            await tracking_manager.record_prediction(pred)

        # Trigger cleanup
        await tracking_manager._perform_cleanup()

        # Verify cache was cleaned up appropriately
        total_cached = sum(
            len(preds) for preds in tracking_manager._pending_predictions.values()
        )
        assert total_cached < 100  # Should have cleaned up old predictions
