"""
Comprehensive tests for src/adaptation/tracking_manager.py - TrackingManager system.

Tests all public methods, error conditions, and complex integration scenarios
for the TrackingManager class and TrackingConfig for complete coverage.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import json
import os
import tempfile
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftMetrics,
    DriftSeverity,
)
from src.adaptation.optimizer import ModelOptimizer
from src.adaptation.retrainer import (
    AdaptiveRetrainer,
    RetrainingRequest,
    RetrainingTrigger,
)
from src.adaptation.tracker import (
    AccuracyAlert,
    AccuracyTracker,
    AlertSeverity,
    RealTimeMetrics,
)
from src.adaptation.tracking_manager import (
    TrackingConfig,
    TrackingManager,
    TrackingManagerError,
)
from src.adaptation.validator import AccuracyMetrics, PredictionValidator
from src.core.constants import ModelType
from src.core.exceptions import ErrorSeverity
from src.models.base.predictor import PredictionResult


class TestTrackingConfig:
    """Test suite for TrackingConfig class."""

    def test_tracking_config_defaults(self):
        """Test TrackingConfig initialization with default values."""
        config = TrackingConfig()

        assert config.enabled is True
        assert config.monitoring_interval_seconds == 60
        assert config.auto_validation_enabled is True
        assert config.validation_window_minutes == 30
        assert config.max_stored_alerts == 1000
        assert config.trend_analysis_points == 10
        assert config.cleanup_interval_hours == 24

        # Real-time publishing defaults
        assert config.realtime_publishing_enabled is True
        assert config.websocket_enabled is True
        assert config.sse_enabled is True
        assert config.websocket_port == 8765

        # Dashboard defaults
        assert config.dashboard_enabled is True
        assert config.dashboard_host == "0.0.0.0"
        assert config.dashboard_port == 8888

        # Drift detection defaults
        assert config.drift_detection_enabled is True
        assert config.drift_baseline_days == 30
        assert config.drift_current_days == 7

        # Adaptive retraining defaults
        assert config.adaptive_retraining_enabled is True
        assert config.retraining_accuracy_threshold == 60.0
        assert config.max_concurrent_retrains == 2

    def test_tracking_config_custom_values(self):
        """Test TrackingConfig with custom values."""
        config = TrackingConfig(
            enabled=False,
            monitoring_interval_seconds=30,
            auto_validation_enabled=False,
            validation_window_minutes=15,
            max_stored_alerts=500,
            trend_analysis_points=5,
            realtime_publishing_enabled=False,
            dashboard_enabled=False,
            drift_detection_enabled=False,
            adaptive_retraining_enabled=False,
            retraining_accuracy_threshold=70.0,
        )

        assert config.enabled is False
        assert config.monitoring_interval_seconds == 30
        assert config.auto_validation_enabled is False
        assert config.validation_window_minutes == 15
        assert config.max_stored_alerts == 500
        assert config.trend_analysis_points == 5
        assert config.realtime_publishing_enabled is False
        assert config.dashboard_enabled is False
        assert config.drift_detection_enabled is False
        assert config.adaptive_retraining_enabled is False
        assert config.retraining_accuracy_threshold == 70.0

    def test_post_init_alert_thresholds(self):
        """Test that post_init sets default alert thresholds."""
        config = TrackingConfig()

        assert config.alert_thresholds is not None
        assert "accuracy_warning" in config.alert_thresholds
        assert "accuracy_critical" in config.alert_thresholds
        assert "error_warning" in config.alert_thresholds
        assert "error_critical" in config.alert_thresholds
        assert config.alert_thresholds["accuracy_warning"] == 70.0
        assert config.alert_thresholds["accuracy_critical"] == 50.0

    def test_custom_alert_thresholds_preserved(self):
        """Test that custom alert thresholds are preserved in post_init."""
        custom_thresholds = {
            "accuracy_warning": 80.0,
            "accuracy_critical": 60.0,
            "custom_threshold": 25.0,
        }

        config = TrackingConfig(alert_thresholds=custom_thresholds)

        assert config.alert_thresholds == custom_thresholds
        assert config.alert_thresholds["accuracy_warning"] == 80.0
        assert config.alert_thresholds["custom_threshold"] == 25.0


class TestTrackingManager:
    """Test suite for TrackingManager class."""

    @pytest.fixture
    def basic_config(self):
        """Create basic tracking configuration for testing."""
        return TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=5,
            max_stored_alerts=50,
            trend_analysis_points=3,
            drift_detection_enabled=False,  # Disable complex features for basic tests
            adaptive_retraining_enabled=False,
            realtime_publishing_enabled=False,
            dashboard_enabled=False,
            websocket_api_enabled=False,
        )

    @pytest.fixture
    def mock_database_manager(self):
        """Create mock database manager."""
        manager = Mock()
        manager.execute_query = AsyncMock(return_value=[])
        manager.health_check = AsyncMock(return_value={"status": "healthy"})
        return manager

    @pytest.fixture
    def mock_model_registry(self):
        """Create mock model registry."""
        return {
            "living_room_ensemble": Mock(),
            "bedroom_lstm": Mock(),
            "kitchen_xgboost": Mock(),
        }

    @pytest.fixture
    def mock_feature_engine(self):
        """Create mock feature engineering engine."""
        engine = Mock()
        engine.extract_features = AsyncMock(
            return_value={"feature1": 1.0, "feature2": 2.0}
        )
        return engine

    @pytest.fixture
    def basic_tracking_manager(self, basic_config, mock_database_manager):
        """Create basic TrackingManager for testing."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {}
            mock_get_config.return_value = mock_system_config

            return TrackingManager(
                config=basic_config, database_manager=mock_database_manager
            )

    def test_tracking_manager_initialization(
        self, basic_config, mock_database_manager, mock_model_registry
    ):
        """Test TrackingManager initialization."""
        notification_callbacks = [Mock(), Mock()]

        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {"living_room": Mock()}
            mock_get_config.return_value = mock_system_config

            manager = TrackingManager(
                config=basic_config,
                database_manager=mock_database_manager,
                model_registry=mock_model_registry,
                notification_callbacks=notification_callbacks,
            )

            assert manager.config == basic_config
            assert manager.database_manager == mock_database_manager
            assert manager.model_registry == mock_model_registry
            assert len(manager.notification_callbacks) == 2
            assert not manager._tracking_active
            assert isinstance(manager._pending_predictions, dict)
            assert manager._total_predictions_recorded == 0
            assert manager._total_validations_performed == 0

    def test_initialization_with_enhanced_mqtt_manager(self, basic_config):
        """Test initialization with Enhanced MQTT Manager."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            with patch(
                "src.adaptation.tracking_manager.EnhancedMQTTIntegrationManager"
            ) as mock_enhanced_mqtt:
                mock_system_config = Mock()
                mock_system_config.mqtt = Mock()
                mock_system_config.rooms = {}
                mock_get_config.return_value = mock_system_config

                mock_mqtt_instance = Mock()
                mock_enhanced_mqtt.return_value = mock_mqtt_instance

                manager = TrackingManager(config=basic_config)

                # Should create Enhanced MQTT Manager automatically
                mock_enhanced_mqtt.assert_called_once()
                assert manager.mqtt_integration_manager == mock_mqtt_instance

    def test_initialization_with_provided_mqtt_manager(self, basic_config):
        """Test initialization with provided MQTT manager."""
        custom_mqtt_manager = Mock()

        with patch("src.adaptation.tracking_manager.get_config"):
            manager = TrackingManager(
                config=basic_config, mqtt_integration_manager=custom_mqtt_manager
            )

            assert manager.mqtt_integration_manager == custom_mqtt_manager

    def test_initialize_disabled_tracking(self):
        """Test initialization when tracking is disabled."""
        disabled_config = TrackingConfig(enabled=False)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_disabled():
                with patch(
                    "src.adaptation.tracking_manager.get_config"
                ) as mock_get_config:
                    mock_system_config = Mock()
                    mock_system_config.mqtt = Mock()
                    mock_system_config.rooms = {}
                    mock_get_config.return_value = mock_system_config

                    manager = TrackingManager(config=disabled_config)
                    await manager.initialize()

                    # Should not initialize components when disabled
                    assert manager.validator is None
                    assert manager.accuracy_tracker is None
                    assert not manager._tracking_active

            loop.run_until_complete(test_disabled())
        finally:
            loop.close()

    def test_initialize_full_system(
        self, mock_database_manager, mock_model_registry, mock_feature_engine
    ):
        """Test full system initialization with all features enabled."""
        full_config = TrackingConfig(
            enabled=True,
            drift_detection_enabled=True,
            adaptive_retraining_enabled=True,
            optimization_enabled=True,
            realtime_publishing_enabled=True,
            dashboard_enabled=False,  # Disable dashboard to avoid import issues
            websocket_api_enabled=False,
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_full_init():
                with patch(
                    "src.adaptation.tracking_manager.get_config"
                ) as mock_get_config:
                    with patch(
                        "src.adaptation.tracking_manager.EnhancedMQTTIntegrationManager"
                    ) as mock_mqtt:
                        with patch.dict(
                            "os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}
                        ):
                            mock_system_config = Mock()
                            mock_system_config.mqtt = Mock()
                            mock_system_config.rooms = {}
                            mock_get_config.return_value = mock_system_config

                            mock_mqtt_instance = Mock()
                            mock_mqtt_instance.initialize = AsyncMock()
                            mock_mqtt.return_value = mock_mqtt_instance

                            manager = TrackingManager(
                                config=full_config,
                                database_manager=mock_database_manager,
                                model_registry=mock_model_registry,
                                feature_engineering_engine=mock_feature_engine,
                            )

                            await manager.initialize()

                            # Verify all components were initialized
                            assert manager.validator is not None
                            assert manager.accuracy_tracker is not None
                            assert manager.drift_detector is not None
                            assert manager.adaptive_retrainer is not None
                            assert manager.model_optimizer is not None
                            assert manager._tracking_active

            loop.run_until_complete(test_full_init())
        finally:
            loop.close()

    def test_start_tracking_disabled(self, basic_tracking_manager):
        """Test start_tracking when tracking is disabled."""
        basic_tracking_manager.config.enabled = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_disabled_start():
                await basic_tracking_manager.start_tracking()
                assert not basic_tracking_manager._tracking_active

            loop.run_until_complete(test_disabled_start())
        finally:
            loop.close()

    def test_start_tracking_already_active(self, basic_tracking_manager):
        """Test start_tracking when already active."""
        basic_tracking_manager._tracking_active = True

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_already_active():
                await basic_tracking_manager.start_tracking()
                assert basic_tracking_manager._tracking_active

            loop.run_until_complete(test_already_active())
        finally:
            loop.close()

    def test_start_tracking_success(self, basic_tracking_manager):
        """Test successful start_tracking."""
        basic_tracking_manager.accuracy_tracker = Mock()
        basic_tracking_manager.accuracy_tracker.start_monitoring = AsyncMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_start_success():
                with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
                    await basic_tracking_manager.start_tracking()

                    assert basic_tracking_manager._tracking_active
                    basic_tracking_manager.accuracy_tracker.start_monitoring.assert_called_once()

            loop.run_until_complete(test_start_success())
        finally:
            loop.close()

    def test_start_tracking_with_background_tasks(self, basic_tracking_manager):
        """Test start_tracking with background tasks enabled."""
        basic_tracking_manager.accuracy_tracker = Mock()
        basic_tracking_manager.accuracy_tracker.start_monitoring = AsyncMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_with_tasks():
                # Don't set DISABLE_BACKGROUND_TASKS
                with patch("asyncio.create_task") as mock_create_task:
                    mock_task = Mock()
                    mock_create_task.return_value = mock_task

                    await basic_tracking_manager.start_tracking()

                    assert basic_tracking_manager._tracking_active
                    assert len(basic_tracking_manager._background_tasks) >= 1
                    mock_create_task.assert_called()

            loop.run_until_complete(test_with_tasks())
        finally:
            loop.close()

    def test_start_tracking_error(self, basic_tracking_manager):
        """Test start_tracking with error."""
        basic_tracking_manager.accuracy_tracker = Mock()
        basic_tracking_manager.accuracy_tracker.start_monitoring = AsyncMock(
            side_effect=Exception("Start monitoring failed")
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_start_error():
                with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
                    with pytest.raises(
                        TrackingManagerError, match="Failed to start tracking"
                    ):
                        await basic_tracking_manager.start_tracking()

            loop.run_until_complete(test_start_error())
        finally:
            loop.close()

    def test_stop_tracking_not_active(self, basic_tracking_manager):
        """Test stop_tracking when not active."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_stop_inactive():
                await basic_tracking_manager.stop_tracking()
                # Should not raise error, just return
                assert not basic_tracking_manager._tracking_active

            loop.run_until_complete(test_stop_inactive())
        finally:
            loop.close()

    def test_stop_tracking_success(self, basic_tracking_manager):
        """Test successful stop_tracking."""
        # Setup active state
        basic_tracking_manager._tracking_active = True
        basic_tracking_manager.accuracy_tracker = Mock()
        basic_tracking_manager.accuracy_tracker.stop_monitoring = AsyncMock()
        basic_tracking_manager.adaptive_retrainer = Mock()
        basic_tracking_manager.adaptive_retrainer.shutdown = AsyncMock()

        mock_task = Mock()
        mock_task.cancel = Mock()
        basic_tracking_manager._background_tasks = [mock_task]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_stop_success():
                await basic_tracking_manager.stop_tracking()

                assert not basic_tracking_manager._tracking_active
                assert len(basic_tracking_manager._background_tasks) == 0
                basic_tracking_manager.accuracy_tracker.stop_monitoring.assert_called_once()
                basic_tracking_manager.adaptive_retrainer.shutdown.assert_called_once()

            loop.run_until_complete(test_stop_success())
        finally:
            loop.close()

    def test_record_prediction_disabled(self, basic_tracking_manager):
        """Test record_prediction when tracking is disabled."""
        basic_tracking_manager.config.enabled = False

        mock_prediction = Mock(spec=PredictionResult)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_record_disabled():
                await basic_tracking_manager.record_prediction(mock_prediction)
                # Should return without error
                assert basic_tracking_manager._total_predictions_recorded == 0

            loop.run_until_complete(test_record_disabled())
        finally:
            loop.close()

    def test_record_prediction_success(self, basic_tracking_manager):
        """Test successful record_prediction."""
        # Setup validator
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.record_prediction = AsyncMock()

        # Setup MQTT manager
        basic_tracking_manager.mqtt_integration_manager = Mock()
        basic_tracking_manager.mqtt_integration_manager.publish_prediction = AsyncMock(
            return_value={"mqtt": {"success": True}}
        )

        # Create mock prediction result
        mock_prediction = Mock(spec=PredictionResult)
        mock_prediction.prediction_metadata = {"room_id": "living_room"}
        mock_prediction.predicted_time = datetime.now(timezone.utc)
        mock_prediction.confidence_score = 0.85
        mock_prediction.model_type = ModelType.ENSEMBLE
        mock_prediction.transition_type = "next_occupied"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_record_success():
                await basic_tracking_manager.record_prediction(mock_prediction)

                # Verify validator was called
                basic_tracking_manager.validator.record_prediction.assert_called_once()

                # Verify MQTT publishing was attempted
                basic_tracking_manager.mqtt_integration_manager.publish_prediction.assert_called_once()

                # Verify prediction was cached
                assert "living_room" in basic_tracking_manager._pending_predictions
                assert (
                    len(basic_tracking_manager._pending_predictions["living_room"]) == 1
                )

                # Verify counter was incremented
                assert basic_tracking_manager._total_predictions_recorded == 1

            loop.run_until_complete(test_record_success())
        finally:
            loop.close()

    def test_record_prediction_mqtt_error(self, basic_tracking_manager):
        """Test record_prediction with MQTT publishing error."""
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.record_prediction = AsyncMock()

        basic_tracking_manager.mqtt_integration_manager = Mock()
        basic_tracking_manager.mqtt_integration_manager.publish_prediction = AsyncMock(
            side_effect=Exception("MQTT connection failed")
        )

        mock_prediction = Mock(spec=PredictionResult)
        mock_prediction.prediction_metadata = {"room_id": "bedroom"}
        mock_prediction.predicted_time = datetime.now(timezone.utc)
        mock_prediction.confidence_score = 0.75
        mock_prediction.model_type = ModelType.LSTM
        mock_prediction.transition_type = "next_vacant"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_mqtt_error():
                # Should not raise exception - MQTT error should be logged but not propagated
                await basic_tracking_manager.record_prediction(mock_prediction)

                # Prediction should still be recorded despite MQTT error
                assert basic_tracking_manager._total_predictions_recorded == 1
                basic_tracking_manager.validator.record_prediction.assert_called_once()

            loop.run_until_complete(test_mqtt_error())
        finally:
            loop.close()

    def test_record_prediction_cache_cleanup(self, basic_tracking_manager):
        """Test prediction cache cleanup in record_prediction."""
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.record_prediction = AsyncMock()

        # Add old predictions to cache
        old_time = datetime.now(timezone.utc) - timedelta(hours=3)
        recent_time = datetime.now(timezone.utc) - timedelta(minutes=30)

        old_prediction = Mock(spec=PredictionResult)
        old_prediction.predicted_time = old_time

        recent_prediction = Mock(spec=PredictionResult)
        recent_prediction.predicted_time = recent_time

        basic_tracking_manager._pending_predictions["test_room"] = [
            old_prediction,
            recent_prediction,
        ]

        # Create new prediction
        new_prediction = Mock(spec=PredictionResult)
        new_prediction.prediction_metadata = {"room_id": "test_room"}
        new_prediction.predicted_time = datetime.now(timezone.utc)
        new_prediction.confidence_score = 0.8
        new_prediction.model_type = ModelType.XGBOOST
        new_prediction.transition_type = "next_occupied"

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_cache_cleanup():
                await basic_tracking_manager.record_prediction(new_prediction)

                # Old prediction should be removed, recent and new should remain
                room_predictions = basic_tracking_manager._pending_predictions[
                    "test_room"
                ]
                assert len(room_predictions) == 2  # recent + new
                assert old_prediction not in room_predictions
                assert recent_prediction in room_predictions
                assert new_prediction in room_predictions

            loop.run_until_complete(test_cache_cleanup())
        finally:
            loop.close()

    def test_handle_room_state_change_disabled(self, basic_tracking_manager):
        """Test handle_room_state_change when disabled."""
        basic_tracking_manager.config.enabled = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_disabled():
                await basic_tracking_manager.handle_room_state_change(
                    room_id="kitchen",
                    new_state="occupied",
                    change_time=datetime.now(timezone.utc),
                )

                # Should return without error
                assert basic_tracking_manager._total_validations_performed == 0

            loop.run_until_complete(test_disabled())
        finally:
            loop.close()

    def test_handle_room_state_change_success(self, basic_tracking_manager):
        """Test successful handle_room_state_change."""
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.validate_prediction = AsyncMock()

        change_time = datetime.now(timezone.utc)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_handle_success():
                await basic_tracking_manager.handle_room_state_change(
                    room_id="office",
                    new_state="vacant",
                    change_time=change_time,
                    previous_state="occupied",
                )

                # Verify validator was called
                basic_tracking_manager.validator.validate_prediction.assert_called_once_with(
                    room_id="office",
                    actual_time=change_time,
                    transition_type="occupied_to_vacant",
                )

                # Verify counter was incremented
                assert basic_tracking_manager._total_validations_performed == 1

            loop.run_until_complete(test_handle_success())
        finally:
            loop.close()

    def test_handle_room_state_change_no_validator(self, basic_tracking_manager):
        """Test handle_room_state_change without validator."""
        basic_tracking_manager.validator = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_no_validator():
                await basic_tracking_manager.handle_room_state_change(
                    room_id="garage",
                    new_state="occupied",
                    change_time=datetime.now(timezone.utc),
                )

                # Should log warning but not raise error
                assert basic_tracking_manager._total_validations_performed == 0

            loop.run_until_complete(test_no_validator())
        finally:
            loop.close()

    def test_get_tracking_status_comprehensive(self, basic_tracking_manager):
        """Test comprehensive get_tracking_status."""
        # Setup components
        basic_tracking_manager._tracking_active = True
        basic_tracking_manager._total_predictions_recorded = 150
        basic_tracking_manager._total_validations_performed = 145
        basic_tracking_manager._total_drift_checks_performed = 12
        basic_tracking_manager._last_drift_check_time = datetime.now(timezone.utc)

        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.get_total_predictions = AsyncMock(
            return_value=150
        )
        basic_tracking_manager.validator.get_validation_rate = AsyncMock(
            return_value=0.97
        )
        basic_tracking_manager.validator._pending_predictions = {
            "room1": [],
            "room2": [],
        }

        basic_tracking_manager.accuracy_tracker = Mock(spec=AccuracyTracker)
        basic_tracking_manager.accuracy_tracker.get_tracker_stats = Mock(
            return_value={
                "monitoring_active": True,
                "metrics_tracked": {"rooms": 3, "models": 2},
                "alerts": {"active": 2, "total_stored": 5},
            }
        )

        basic_tracking_manager.adaptive_retrainer = Mock(spec=AdaptiveRetrainer)
        basic_tracking_manager.adaptive_retrainer.get_retrainer_stats = Mock(
            return_value={"active_jobs": 1, "completed_jobs": 8, "failed_jobs": 0}
        )

        # Add predictions to cache
        basic_tracking_manager._pending_predictions["room1"] = [Mock(), Mock()]
        basic_tracking_manager._pending_predictions["room2"] = [Mock()]

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_comprehensive_status():
                status = await basic_tracking_manager.get_tracking_status()

                assert status["tracking_active"] is True
                assert status["config"]["enabled"] is True
                assert status["performance"]["total_predictions_recorded"] == 150
                assert status["performance"]["total_validations_performed"] == 145
                assert status["performance"]["total_drift_checks_performed"] == 12
                assert "last_drift_check_time" in status["performance"]
                assert status["performance"]["system_uptime_seconds"] >= 0

                assert status["validator"]["total_predictions"] == 150
                assert status["validator"]["validation_rate"] == 0.97
                assert status["validator"]["pending_validations"] == 2

                assert status["accuracy_tracker"]["monitoring_active"] is True
                assert status["prediction_cache"]["room1"] == 2
                assert status["prediction_cache"]["room2"] == 1

            loop.run_until_complete(test_comprehensive_status())
        finally:
            loop.close()

    def test_get_tracking_status_error(self, basic_tracking_manager):
        """Test get_tracking_status with error."""
        # Setup validator to raise error
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.get_total_predictions = AsyncMock(
            side_effect=Exception("Database error")
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_status_error():
                status = await basic_tracking_manager.get_tracking_status()

                # Should return error in status
                assert "error" in status
                assert "Database error" in str(status["error"])

            loop.run_until_complete(test_status_error())
        finally:
            loop.close()

    def test_get_real_time_metrics(self, basic_tracking_manager):
        """Test get_real_time_metrics."""
        mock_metrics = RealTimeMetrics(room_id="test_room", window_1h_accuracy=85.0)

        basic_tracking_manager.accuracy_tracker = Mock(spec=AccuracyTracker)
        basic_tracking_manager.accuracy_tracker.get_real_time_metrics = AsyncMock(
            return_value=mock_metrics
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_metrics():
                result = await basic_tracking_manager.get_real_time_metrics(
                    room_id="test_room", model_type=ModelType.ENSEMBLE
                )

                assert result == mock_metrics
                basic_tracking_manager.accuracy_tracker.get_real_time_metrics.assert_called_once_with(
                    "test_room", ModelType.ENSEMBLE
                )

            loop.run_until_complete(test_get_metrics())
        finally:
            loop.close()

    def test_get_real_time_metrics_no_tracker(self, basic_tracking_manager):
        """Test get_real_time_metrics without tracker."""
        basic_tracking_manager.accuracy_tracker = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_no_tracker():
                result = await basic_tracking_manager.get_real_time_metrics()
                assert result is None

            loop.run_until_complete(test_no_tracker())
        finally:
            loop.close()

    def test_get_active_alerts(self, basic_tracking_manager):
        """Test get_active_alerts."""
        mock_alert = AccuracyAlert(
            alert_id="test_alert",
            room_id="study",
            model_type="ensemble",
            severity=AlertSeverity.WARNING,
            trigger_condition="accuracy_warning",
            current_value=65.0,
            threshold_value=70.0,
            description="Test alert",
            affected_metrics={},
            recent_predictions=30,
        )

        basic_tracking_manager.accuracy_tracker = Mock(spec=AccuracyTracker)
        basic_tracking_manager.accuracy_tracker.get_active_alerts = AsyncMock(
            return_value=[mock_alert]
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_alerts():
                alerts = await basic_tracking_manager.get_active_alerts(
                    room_id="study", severity="warning"
                )

                assert len(alerts) == 1
                assert alerts[0] == mock_alert
                basic_tracking_manager.accuracy_tracker.get_active_alerts.assert_called_once()

            loop.run_until_complete(test_get_alerts())
        finally:
            loop.close()

    def test_acknowledge_alert(self, basic_tracking_manager):
        """Test acknowledge_alert."""
        basic_tracking_manager.accuracy_tracker = Mock(spec=AccuracyTracker)
        basic_tracking_manager.accuracy_tracker.acknowledge_alert = AsyncMock(
            return_value=True
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_acknowledge():
                result = await basic_tracking_manager.acknowledge_alert(
                    "alert_123", "admin"
                )

                assert result is True
                basic_tracking_manager.accuracy_tracker.acknowledge_alert.assert_called_once_with(
                    "alert_123", "admin"
                )

            loop.run_until_complete(test_acknowledge())
        finally:
            loop.close()

    def test_add_notification_callback(self, basic_tracking_manager):
        """Test add_notification_callback."""

        def test_callback(alert):
            pass

        basic_tracking_manager.accuracy_tracker = Mock(spec=AccuracyTracker)
        basic_tracking_manager.accuracy_tracker.add_notification_callback = Mock()

        basic_tracking_manager.add_notification_callback(test_callback)

        assert test_callback in basic_tracking_manager.notification_callbacks
        basic_tracking_manager.accuracy_tracker.add_notification_callback.assert_called_once_with(
            test_callback
        )

    def test_remove_notification_callback(self, basic_tracking_manager):
        """Test remove_notification_callback."""

        def test_callback(alert):
            pass

        basic_tracking_manager.notification_callbacks.append(test_callback)
        basic_tracking_manager.accuracy_tracker = Mock(spec=AccuracyTracker)
        basic_tracking_manager.accuracy_tracker.remove_notification_callback = Mock()

        basic_tracking_manager.remove_notification_callback(test_callback)

        assert test_callback not in basic_tracking_manager.notification_callbacks
        basic_tracking_manager.accuracy_tracker.remove_notification_callback.assert_called_once_with(
            test_callback
        )

    def test_register_model(self, basic_tracking_manager):
        """Test register_model."""
        mock_model = Mock()

        basic_tracking_manager.register_model("bathroom", ModelType.LSTM, mock_model)

        assert "bathroom_ModelType.LSTM" in basic_tracking_manager.model_registry
        assert (
            basic_tracking_manager.model_registry["bathroom_ModelType.LSTM"]
            == mock_model
        )

    def test_register_model_string_type(self, basic_tracking_manager):
        """Test register_model with string model type."""
        mock_model = Mock()

        basic_tracking_manager.register_model("hallway", "xgboost", mock_model)

        assert "hallway_xgboost" in basic_tracking_manager.model_registry
        assert basic_tracking_manager.model_registry["hallway_xgboost"] == mock_model

    def test_unregister_model(self, basic_tracking_manager):
        """Test unregister_model."""
        mock_model = Mock()
        basic_tracking_manager.model_registry["closet_ensemble"] = mock_model

        basic_tracking_manager.unregister_model("closet", "ensemble")

        assert "closet_ensemble" not in basic_tracking_manager.model_registry

    def test_unregister_nonexistent_model(self, basic_tracking_manager):
        """Test unregister_model for nonexistent model."""
        # Should not raise error
        basic_tracking_manager.unregister_model("nonexistent", "model")

        # Should be fine (no assertion needed, just testing it doesn't crash)

    def test_calculate_uptime_seconds(self, basic_tracking_manager):
        """Test _calculate_uptime_seconds method."""
        uptime = basic_tracking_manager._calculate_uptime_seconds()

        # Should be small positive number since just created
        assert uptime >= 0
        assert uptime < 60  # Less than a minute

    def test_calculate_uptime_timezone_naive(self, basic_tracking_manager):
        """Test uptime calculation with timezone-naive start time."""
        # Set start time to timezone-naive datetime
        basic_tracking_manager._system_start_time = datetime(2024, 1, 1, 12, 0, 0)

        uptime = basic_tracking_manager._calculate_uptime_seconds()

        # Should handle timezone conversion gracefully
        assert uptime >= 0

    def test_calculate_uptime_error_handling(self, basic_tracking_manager):
        """Test uptime calculation error handling."""
        # Set invalid start time
        basic_tracking_manager._system_start_time = "invalid"

        uptime = basic_tracking_manager._calculate_uptime_seconds()

        # Should return 0 on error
        assert uptime == 0.0


class TestTrackingManagerDriftDetection:
    """Test suite for drift detection functionality in TrackingManager."""

    @pytest.fixture
    def drift_enabled_config(self):
        """Create config with drift detection enabled."""
        return TrackingConfig(
            drift_detection_enabled=True,
            drift_check_interval_hours=1,
            drift_baseline_days=7,
            drift_current_days=2,
            adaptive_retraining_enabled=False,
            realtime_publishing_enabled=False,
            dashboard_enabled=False,
            websocket_api_enabled=False,
        )

    @pytest.fixture
    def drift_tracking_manager(self, drift_enabled_config, mock_database_manager):
        """Create TrackingManager with drift detection enabled."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {}
            mock_get_config.return_value = mock_system_config

            return TrackingManager(
                config=drift_enabled_config, database_manager=mock_database_manager
            )

    def test_check_drift_disabled(self, basic_tracking_manager):
        """Test check_drift when drift detection is disabled."""
        basic_tracking_manager.config.drift_detection_enabled = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_drift_disabled():
                result = await basic_tracking_manager.check_drift("test_room")
                assert result is None

            loop.run_until_complete(test_drift_disabled())
        finally:
            loop.close()

    def test_check_drift_no_validator(self, drift_tracking_manager):
        """Test check_drift without validator."""
        drift_tracking_manager.validator = None
        drift_tracking_manager.drift_detector = Mock(spec=ConceptDriftDetector)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_no_validator():
                result = await drift_tracking_manager.check_drift("test_room")
                assert result is None

            loop.run_until_complete(test_no_validator())
        finally:
            loop.close()

    def test_check_drift_success(self, drift_tracking_manager):
        """Test successful drift detection."""
        # Setup components
        drift_tracking_manager.validator = Mock(spec=PredictionValidator)
        drift_tracking_manager.drift_detector = Mock(spec=ConceptDriftDetector)
        drift_tracking_manager.adaptive_retrainer = Mock(spec=AdaptiveRetrainer)

        # Create mock drift metrics
        mock_drift_metrics = Mock(spec=DriftMetrics)
        mock_drift_metrics.drift_severity = DriftSeverity.MEDIUM
        mock_drift_metrics.overall_drift_score = 0.6
        mock_drift_metrics.drift_types = []
        mock_drift_metrics.retraining_recommended = True

        drift_tracking_manager.drift_detector.detect_drift = AsyncMock(
            return_value=mock_drift_metrics
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_drift_success():
                result = await drift_tracking_manager.check_drift("workshop")

                assert result == mock_drift_metrics
                assert drift_tracking_manager._total_drift_checks_performed == 1

                # Verify drift detector was called
                drift_tracking_manager.drift_detector.detect_drift.assert_called_once_with(
                    room_id="workshop",
                    prediction_validator=drift_tracking_manager.validator,
                    feature_engineering_engine=None,
                )

            loop.run_until_complete(test_drift_success())
        finally:
            loop.close()

    def test_get_drift_status(self, drift_tracking_manager):
        """Test get_drift_status."""
        drift_tracking_manager._total_drift_checks_performed = 15
        drift_tracking_manager._last_drift_check_time = datetime.now(
            timezone.utc
        ) - timedelta(hours=2)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_drift_status():
                status = await drift_tracking_manager.get_drift_status()

                assert status["drift_detection_enabled"] is True
                assert status["drift_detector_available"] is True
                assert status["total_drift_checks"] == 15
                assert "last_drift_check" in status
                assert "next_drift_check" in status
                assert "drift_config" in status

                # Check config details
                config = status["drift_config"]
                assert config["check_interval_hours"] == 1
                assert config["baseline_days"] == 7
                assert config["current_days"] == 2

            loop.run_until_complete(test_drift_status())
        finally:
            loop.close()

    def test_get_drift_status_disabled(self, basic_tracking_manager):
        """Test get_drift_status when drift detection is disabled."""
        basic_tracking_manager.config.drift_detection_enabled = False

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_status_disabled():
                status = await basic_tracking_manager.get_drift_status()

                assert status["drift_detection_enabled"] is False
                assert status["drift_detector_available"] is False
                assert "drift_config" not in status

            loop.run_until_complete(test_status_disabled())
        finally:
            loop.close()


class TestTrackingManagerRetraining:
    """Test suite for retraining functionality in TrackingManager."""

    @pytest.fixture
    def retraining_config(self):
        """Create config with adaptive retraining enabled."""
        return TrackingConfig(
            adaptive_retraining_enabled=True,
            retraining_accuracy_threshold=65.0,
            retraining_error_threshold=20.0,
            max_concurrent_retrains=1,
            drift_detection_enabled=False,
            realtime_publishing_enabled=False,
            dashboard_enabled=False,
            websocket_api_enabled=False,
        )

    @pytest.fixture
    def retraining_manager(self, retraining_config, mock_database_manager):
        """Create TrackingManager with retraining enabled."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {}
            mock_get_config.return_value = mock_system_config

            return TrackingManager(
                config=retraining_config, database_manager=mock_database_manager
            )

    def test_request_manual_retraining_success(self, retraining_manager):
        """Test successful manual retraining request."""
        retraining_manager.adaptive_retrainer = Mock(spec=AdaptiveRetrainer)
        retraining_manager.adaptive_retrainer.request_retraining = AsyncMock(
            return_value="request_123"
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_manual_retrain():
                request_id = await retraining_manager.request_manual_retraining(
                    room_id="laundry",
                    model_type=ModelType.LSTM,
                    strategy="full_retrain",
                    priority=8.0,
                )

                assert request_id == "request_123"
                retraining_manager.adaptive_retrainer.request_retraining.assert_called_once()

            loop.run_until_complete(test_manual_retrain())
        finally:
            loop.close()

    def test_request_manual_retraining_no_retrainer(self, retraining_manager):
        """Test manual retraining request without retrainer."""
        retraining_manager.adaptive_retrainer = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_no_retrainer():
                request_id = await retraining_manager.request_manual_retraining(
                    room_id="pantry", model_type="xgboost"
                )

                assert request_id is None

            loop.run_until_complete(test_no_retrainer())
        finally:
            loop.close()

    def test_get_retraining_status(self, retraining_manager):
        """Test get_retraining_status."""
        mock_status = {
            "active_jobs": 2,
            "completed_jobs": 15,
            "failed_jobs": 1,
            "queue_size": 0,
        }

        retraining_manager.adaptive_retrainer = Mock(spec=AdaptiveRetrainer)
        retraining_manager.adaptive_retrainer.get_retraining_status = AsyncMock(
            return_value=mock_status
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_get_status():
                status = await retraining_manager.get_retraining_status("request_456")

                assert status == mock_status
                retraining_manager.adaptive_retrainer.get_retraining_status.assert_called_once_with(
                    "request_456"
                )

            loop.run_until_complete(test_get_status())
        finally:
            loop.close()

    def test_cancel_retraining(self, retraining_manager):
        """Test cancel_retraining."""
        retraining_manager.adaptive_retrainer = Mock(spec=AdaptiveRetrainer)
        retraining_manager.adaptive_retrainer.cancel_retraining = AsyncMock(
            return_value=True
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_cancel():
                result = await retraining_manager.cancel_retraining("request_789")

                assert result is True
                retraining_manager.adaptive_retrainer.cancel_retraining.assert_called_once_with(
                    "request_789"
                )

            loop.run_until_complete(test_cancel())
        finally:
            loop.close()


class TestTrackingManagerIntegration:
    """Integration test scenarios for TrackingManager complex workflows."""

    @pytest.fixture
    def integration_config(self):
        """Create comprehensive configuration for integration testing."""
        return TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=1,
            auto_validation_enabled=True,
            drift_detection_enabled=True,
            drift_check_interval_hours=1,
            adaptive_retraining_enabled=True,
            retraining_accuracy_threshold=70.0,
            optimization_enabled=True,
            realtime_publishing_enabled=False,  # Disable to avoid import issues
            dashboard_enabled=False,
            websocket_api_enabled=False,
            max_stored_alerts=20,
        )

    @pytest.fixture
    def integration_manager(
        self, integration_config, mock_database_manager, mock_model_registry
    ):
        """Create fully configured TrackingManager for integration testing."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {
                "living_room": Mock(),
                "bedroom": Mock(),
                "kitchen": Mock(),
            }
            mock_get_config.return_value = mock_system_config

            return TrackingManager(
                config=integration_config,
                database_manager=mock_database_manager,
                model_registry=mock_model_registry,
            )

    def test_full_initialization_workflow(self, integration_manager):
        """Test complete initialization workflow with all features."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_full_init():
                with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
                    await integration_manager.initialize()

                    # Verify all components were initialized
                    assert integration_manager.validator is not None
                    assert integration_manager.accuracy_tracker is not None
                    assert integration_manager.drift_detector is not None
                    assert integration_manager.adaptive_retrainer is not None
                    assert integration_manager.model_optimizer is not None
                    assert integration_manager._tracking_active

                    # Test graceful shutdown
                    await integration_manager.stop_tracking()
                    assert not integration_manager._tracking_active

            loop.run_until_complete(test_full_init())
        finally:
            loop.close()

    def test_prediction_validation_workflow(self, integration_manager):
        """Test complete prediction-to-validation workflow."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_prediction_workflow():
                with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
                    await integration_manager.initialize()

                    # Create and record prediction
                    mock_prediction = Mock(spec=PredictionResult)
                    mock_prediction.prediction_metadata = {
                        "room_id": "integration_room"
                    }
                    mock_prediction.predicted_time = datetime.now(
                        timezone.utc
                    ) + timedelta(minutes=30)
                    mock_prediction.confidence_score = 0.82
                    mock_prediction.model_type = ModelType.ENSEMBLE
                    mock_prediction.transition_type = "next_occupied"

                    await integration_manager.record_prediction(mock_prediction)

                    # Verify prediction was recorded
                    assert integration_manager._total_predictions_recorded == 1
                    assert (
                        "integration_room" in integration_manager._pending_predictions
                    )

                    # Simulate room state change
                    state_change_time = datetime.now(timezone.utc) + timedelta(
                        minutes=32
                    )
                    await integration_manager.handle_room_state_change(
                        room_id="integration_room",
                        new_state="occupied",
                        change_time=state_change_time,
                        previous_state="vacant",
                    )

                    # Verify validation was triggered
                    assert integration_manager._total_validations_performed == 1

                    await integration_manager.stop_tracking()

            loop.run_until_complete(test_prediction_workflow())
        finally:
            loop.close()

    def test_error_recovery_workflow(self, integration_manager):
        """Test error recovery in various scenarios."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_error_recovery():
                with patch.dict("os.environ", {"DISABLE_BACKGROUND_TASKS": "1"}):
                    await integration_manager.initialize()

                    # Test prediction recording with MQTT error (should continue)
                    mock_prediction = Mock(spec=PredictionResult)
                    mock_prediction.prediction_metadata = {"room_id": "error_room"}
                    mock_prediction.predicted_time = datetime.now(timezone.utc)
                    mock_prediction.confidence_score = 0.75
                    mock_prediction.model_type = ModelType.LSTM
                    mock_prediction.transition_type = "next_vacant"

                    # Make MQTT publishing fail
                    if (
                        hasattr(integration_manager, "mqtt_integration_manager")
                        and integration_manager.mqtt_integration_manager
                    ):
                        integration_manager.mqtt_integration_manager.publish_prediction = AsyncMock(
                            side_effect=Exception("MQTT error")
                        )

                    # Should not raise exception
                    await integration_manager.record_prediction(mock_prediction)

                    # Prediction should still be recorded
                    assert integration_manager._total_predictions_recorded == 1

                    # Test status retrieval with component errors
                    if (
                        hasattr(integration_manager, "validator")
                        and integration_manager.validator
                    ):
                        integration_manager.validator.get_total_predictions = AsyncMock(
                            side_effect=Exception("Validator error")
                        )

                    # Should return status with error information
                    status = await integration_manager.get_tracking_status()
                    assert "error" in status or "performance" in status

                    await integration_manager.stop_tracking()

            loop.run_until_complete(test_error_recovery())
        finally:
            loop.close()


class TestTrackingManagerError:
    """Test suite for TrackingManagerError exception."""

    def test_tracking_manager_error_creation(self):
        """Test creating TrackingManagerError."""
        error = TrackingManagerError("Test tracking manager error")

        assert str(error) == "Test tracking manager error"
        assert error.error_code == "TRACKING_MANAGER_ERROR"
        assert error.severity == ErrorSeverity.MEDIUM

    def test_tracking_manager_error_with_custom_severity(self):
        """Test creating TrackingManagerError with custom severity."""
        error = TrackingManagerError(
            "Critical tracking manager error", severity=ErrorSeverity.HIGH
        )

        assert str(error) == "Critical tracking manager error"
        assert error.severity == ErrorSeverity.HIGH

    def test_tracking_manager_error_with_cause(self):
        """Test creating TrackingManagerError with cause."""
        original_error = RuntimeError("Original error")
        error = TrackingManagerError(
            "Tracking manager error with cause", cause=original_error
        )

        assert str(error) == "Tracking manager error with cause"
        assert error.cause == original_error


class TestTrackingManagerBackgroundTasks:
    """Test suite for background task functionality in TrackingManager."""

    @pytest.fixture
    def task_manager(self, basic_config, mock_database_manager):
        """Create TrackingManager for background task testing."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {}
            mock_get_config.return_value = mock_system_config

            return TrackingManager(
                config=basic_config, database_manager=mock_database_manager
            )

    def test_cleanup_loop_execution(self, task_manager):
        """Test cleanup loop background task execution."""
        # Add predictions to cache for cleanup testing
        old_prediction = Mock(spec=PredictionResult)
        old_prediction.predicted_time = datetime.now(timezone.utc) - timedelta(hours=3)

        recent_prediction = Mock(spec=PredictionResult)
        recent_prediction.predicted_time = datetime.now(timezone.utc) - timedelta(
            minutes=30
        )

        task_manager._pending_predictions["cleanup_room"] = [
            old_prediction,
            recent_prediction,
        ]
        task_manager.validator = Mock(spec=PredictionValidator)
        task_manager.validator.cleanup_old_predictions = AsyncMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_cleanup():
                # Run cleanup manually
                await task_manager._perform_cleanup()

                # Old prediction should be removed
                remaining_predictions = task_manager._pending_predictions.get(
                    "cleanup_room", []
                )
                assert old_prediction not in remaining_predictions
                assert recent_prediction in remaining_predictions

                # Validator cleanup should be called
                task_manager.validator.cleanup_old_predictions.assert_called_once_with(
                    days_to_keep=7
                )

            loop.run_until_complete(test_cleanup())
        finally:
            loop.close()

    def test_cleanup_with_large_cache(self, task_manager):
        """Test cleanup with large prediction cache."""
        # Create many predictions for a room
        many_predictions = []
        base_time = datetime.now(timezone.utc)

        for i in range(100):  # Create 100 predictions
            pred = Mock(spec=PredictionResult)
            pred.predicted_time = base_time - timedelta(minutes=i)
            many_predictions.append(pred)

        task_manager._pending_predictions["large_cache_room"] = many_predictions

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_large_cleanup():
                await task_manager._perform_cleanup()

                # Should keep only 80% of predictions (80 predictions)
                remaining_predictions = task_manager._pending_predictions.get(
                    "large_cache_room", []
                )
                assert len(remaining_predictions) == 80

                # Should keep the most recent ones
                for pred in remaining_predictions:
                    assert pred in many_predictions[-80:]  # Last 80 predictions

            loop.run_until_complete(test_large_cleanup())
        finally:
            loop.close()

    def test_validation_monitoring_loop(self, task_manager):
        """Test validation monitoring background task."""
        # Setup database query mock
        task_manager.database_manager.execute_query = AsyncMock(
            return_value=[
                {"room_id": "monitor_room", "last_change": datetime.now(timezone.utc)}
            ]
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_monitoring():
                await task_manager._check_for_room_state_changes()

                # Verify database query was made
                task_manager.database_manager.execute_query.assert_called_once()

                # Verify query parameters
                call_args = task_manager.database_manager.execute_query.call_args
                assert "SELECT DISTINCT room_id" in call_args[0][0]  # SQL query
                assert len(call_args[0][1]) == 1  # One parameter (cutoff_time)

            loop.run_until_complete(test_monitoring())
        finally:
            loop.close()

    def test_validation_monitoring_database_error(self, task_manager):
        """Test validation monitoring with database error."""
        # Make database query fail
        task_manager.database_manager.execute_query = AsyncMock(
            side_effect=Exception("Database connection failed")
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_db_error():
                # Should not raise exception, should log warning and continue
                await task_manager._check_for_room_state_changes()

                # Verify fallback was used (database query was attempted)
                task_manager.database_manager.execute_query.assert_called_once()

            loop.run_until_complete(test_db_error())
        finally:
            loop.close()


class TestTrackingManagerMQTTIntegration:
    """Test suite for MQTT integration functionality."""

    @pytest.fixture
    def mqtt_manager(self, basic_config):
        """Create TrackingManager with MQTT integration."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_system_config

            manager = TrackingManager(config=basic_config)
            return manager

    def test_get_enhanced_mqtt_status(self, mqtt_manager):
        """Test get_enhanced_mqtt_status method."""
        # Mock enhanced MQTT manager with stats
        mock_stats = {
            "mqtt_integration": {"mqtt_connected": True, "discovery_published": True},
            "realtime_publishing": {"system_active": True},
            "channels": {
                "total_active": 3,
                "enabled_channels": ["mqtt", "websocket", "sse"],
            },
            "connections": {"websocket_clients": 5, "sse_clients": 2},
            "performance": {
                "predictions_per_minute": 12.5,
                "average_publish_latency_ms": 45,
                "publish_success_rate": 0.98,
            },
        }

        mqtt_manager.mqtt_integration_manager = Mock()
        mqtt_manager.mqtt_integration_manager.get_integration_stats = Mock(
            return_value=mock_stats
        )

        status = mqtt_manager.get_enhanced_mqtt_status()

        assert status["enabled"] is True
        assert status["type"] == "enhanced"
        assert status["mqtt_connected"] is True
        assert status["discovery_published"] is True
        assert status["realtime_publishing_active"] is True
        assert status["total_channels"] == 3
        assert status["websocket_connections"] == 5
        assert status["sse_connections"] == 2
        assert status["predictions_per_minute"] == 12.5
        assert status["average_publish_latency_ms"] == 45
        assert status["publish_success_rate"] == 0.98
        assert status["enabled_channels"] == ["mqtt", "websocket", "sse"]

    def test_get_enhanced_mqtt_status_basic(self, mqtt_manager):
        """Test get_enhanced_mqtt_status with basic MQTT integration."""
        # Mock basic MQTT manager
        basic_stats = {"mqtt_connected": True, "discovery_published": False}

        mqtt_manager.mqtt_integration_manager = Mock()
        mqtt_manager.mqtt_integration_manager.get_integration_stats = Mock(
            return_value=basic_stats
        )
        # Remove the enhanced method to simulate basic integration
        del mqtt_manager.mqtt_integration_manager.get_integration_stats
        mqtt_manager.mqtt_integration_manager.get_integration_stats = Mock(
            return_value=basic_stats
        )

        status = mqtt_manager.get_enhanced_mqtt_status()

        assert status["enabled"] is True
        assert status["type"] == "basic"
        assert status["mqtt_connected"] is True
        assert status["discovery_published"] is False
        assert status["realtime_publishing_active"] is False
        assert status["total_channels"] == 1
        assert status["enabled_channels"] == ["mqtt"]

    def test_get_enhanced_mqtt_status_disabled(self, basic_config):
        """Test get_enhanced_mqtt_status when MQTT is disabled."""
        with patch("src.adaptation.tracking_manager.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_system_config.mqtt = Mock()
            mock_system_config.rooms = {}
            mock_get_config.return_value = mock_system_config

            manager = TrackingManager(
                config=basic_config, mqtt_integration_manager=None  # No MQTT manager
            )

            status = manager.get_enhanced_mqtt_status()

            assert status["enabled"] is False
            assert status["type"] == "none"
            assert status["mqtt_connected"] is False
            assert status["total_channels"] == 0
            assert status["enabled_channels"] == []

    def test_get_enhanced_mqtt_status_error(self, mqtt_manager):
        """Test get_enhanced_mqtt_status with error."""
        mqtt_manager.mqtt_integration_manager = Mock()
        mqtt_manager.mqtt_integration_manager.get_integration_stats = Mock(
            side_effect=Exception("MQTT stats error")
        )

        status = mqtt_manager.get_enhanced_mqtt_status()

        assert status["enabled"] is False
        assert status["type"] == "error"
        assert "error" in status
        assert "MQTT stats error" in str(status["error"])


# Performance and stress testing
class TestTrackingManagerPerformance:
    """Performance and stress tests for TrackingManager."""

    def test_many_predictions_performance(self, basic_tracking_manager):
        """Test performance with many prediction recordings."""
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.record_prediction = AsyncMock()

        # Create many predictions
        predictions = []
        for i in range(100):
            mock_prediction = Mock(spec=PredictionResult)
            mock_prediction.prediction_metadata = {"room_id": f"room_{i % 10}"}
            mock_prediction.predicted_time = datetime.now(timezone.utc)
            mock_prediction.confidence_score = 0.8
            mock_prediction.model_type = ModelType.ENSEMBLE
            mock_prediction.transition_type = "next_occupied"
            predictions.append(mock_prediction)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_many_predictions():
                start_time = datetime.now()

                # Record all predictions
                for prediction in predictions:
                    await basic_tracking_manager.record_prediction(prediction)

                end_time = datetime.now()
                duration = (end_time - start_time).total_seconds()

                # Should complete in reasonable time (less than 5 seconds)
                assert duration < 5.0
                assert basic_tracking_manager._total_predictions_recorded == 100

                # Verify caching worked correctly
                assert (
                    len(basic_tracking_manager._pending_predictions) <= 10
                )  # 10 different rooms

            loop.run_until_complete(test_many_predictions())
        finally:
            loop.close()

    def test_concurrent_operations(self, basic_tracking_manager):
        """Test concurrent tracking operations."""
        basic_tracking_manager.validator = Mock(spec=PredictionValidator)
        basic_tracking_manager.validator.record_prediction = AsyncMock()
        basic_tracking_manager.validator.validate_prediction = AsyncMock()

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_concurrent():
                # Create concurrent tasks
                prediction_tasks = []
                validation_tasks = []

                # Create prediction tasks
                for i in range(20):
                    mock_prediction = Mock(spec=PredictionResult)
                    mock_prediction.prediction_metadata = {
                        "room_id": f"concurrent_room_{i % 5}"
                    }
                    mock_prediction.predicted_time = datetime.now(timezone.utc)
                    mock_prediction.confidence_score = 0.75
                    mock_prediction.model_type = ModelType.LSTM
                    mock_prediction.transition_type = "next_vacant"

                    task = basic_tracking_manager.record_prediction(mock_prediction)
                    prediction_tasks.append(task)

                # Create validation tasks
                for i in range(20):
                    task = basic_tracking_manager.handle_room_state_change(
                        room_id=f"concurrent_room_{i % 5}",
                        new_state="vacant",
                        change_time=datetime.now(timezone.utc),
                    )
                    validation_tasks.append(task)

                # Run all tasks concurrently
                await asyncio.gather(*prediction_tasks, *validation_tasks)

                # Verify all operations completed
                assert basic_tracking_manager._total_predictions_recorded == 20
                assert basic_tracking_manager._total_validations_performed == 20

            loop.run_until_complete(test_concurrent())
        finally:
            loop.close()
