"""
Unit tests for TrackingManager core functionality.

Pure unit tests for TrackingManager, TrackingConfig, and basic coordination logic.
Integration tests are in tests/adaptation/test_tracking_manager.py.
"""

import asyncio
from datetime import UTC, datetime, timedelta
import os
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import pytest_asyncio

from src.adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftMetrics,
    DriftSeverity,
)
from src.adaptation.retrainer import AdaptiveRetrainer, RetrainingError
from src.adaptation.tracker import AccuracyTracker, AlertSeverity
from src.adaptation.tracking_manager import (
    TrackingConfig,
    TrackingManager,
    TrackingManagerError,
)
from src.adaptation.validator import AccuracyMetrics, PredictionValidator
from src.core.constants import ModelType
from src.core.exceptions import ErrorSeverity
from src.data.storage.database import DatabaseManager
from src.integration.mqtt_publisher import MQTTPublisher


class TestTrackingConfig:
    """Unit tests for TrackingConfig class."""

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
        """Test __post_init__ sets up alert thresholds correctly."""
        config = TrackingConfig()

        expected_thresholds = {
            "accuracy_warning": 70.0,
            "accuracy_critical": 50.0,
            "error_warning": 20.0,
            "error_critical": 30.0,
        }
        assert config.alert_thresholds == expected_thresholds

    def test_custom_alert_thresholds_preserved(self):
        """Test that custom alert thresholds are preserved."""
        custom_thresholds = {
            "accuracy_warning": 80.0,
            "accuracy_critical": 60.0,
            "error_warning": 15.0,
            "error_critical": 25.0,
        }

        config = TrackingConfig(alert_thresholds=custom_thresholds)
        assert config.alert_thresholds == custom_thresholds


class TestTrackingManagerInitialization:
    """Unit tests for TrackingManager initialization."""

    @pytest.fixture
    def tracking_config(self):
        """Create test tracking configuration."""
        return TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=60,
            auto_validation_enabled=True,
            drift_detection_enabled=True,
            adaptive_retraining_enabled=True,
        )

    @pytest.fixture
    def mock_database_manager(self):
        """Create mock database manager."""
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.health_check = AsyncMock(return_value={"status": "healthy"})
        return mock_db

    @pytest.fixture
    def mock_mqtt_publisher(self):
        """Create mock MQTT publisher."""
        mock_mqtt = MagicMock(spec=MQTTPublisher)
        mock_mqtt.connect = AsyncMock()
        mock_mqtt.disconnect = AsyncMock()
        return mock_mqtt

    async def test_manager_initialization(self, tracking_config, mock_database_manager):
        """Test basic TrackingManager initialization."""
        manager = TrackingManager(
            config=tracking_config,
            database_manager=mock_database_manager,
        )

        assert manager._config == tracking_config
        assert manager._database_manager == mock_database_manager
        assert manager._prediction_validator is None  # Created on start
        assert manager._accuracy_tracker is None  # Created on start
        assert manager._drift_detector is None  # Created on start
        assert manager._retrainer is None  # Created on start
        assert not manager._running
        assert manager._background_tasks == []

    async def test_manager_initialization_with_components(
        self, tracking_config, mock_database_manager
    ):
        """Test initialization with pre-built components."""
        mock_validator = MagicMock(spec=PredictionValidator)
        mock_tracker = MagicMock(spec=AccuracyTracker)
        mock_drift = MagicMock(spec=ConceptDriftDetector)
        mock_retrainer = MagicMock(spec=AdaptiveRetrainer)

        manager = TrackingManager(
            config=tracking_config,
            database_manager=mock_database_manager,
            prediction_validator=mock_validator,
            accuracy_tracker=mock_tracker,
            drift_detector=mock_drift,
            retrainer=mock_retrainer,
        )

        assert manager._prediction_validator == mock_validator
        assert manager._accuracy_tracker == mock_tracker
        assert manager._drift_detector == mock_drift
        assert manager._retrainer == mock_retrainer

    async def test_manager_shutdown(self, tracking_config, mock_database_manager):
        """Test manager shutdown functionality."""
        manager = TrackingManager(
            config=tracking_config,
            database_manager=mock_database_manager,
        )

        # Simulate running state
        manager._running = True
        mock_task = AsyncMock()
        manager._background_tasks = [mock_task]

        await manager.shutdown()

        assert not manager._running
        mock_task.cancel.assert_called_once()

    async def test_disabled_manager_initialization(self, mock_database_manager):
        """Test initialization with tracking disabled."""
        disabled_config = TrackingConfig(enabled=False)

        manager = TrackingManager(
            config=disabled_config,
            database_manager=mock_database_manager,
        )

        assert not manager._config.enabled
        assert manager._prediction_validator is None
        assert manager._accuracy_tracker is None


class TestTrackingManagerCore:
    """Unit tests for core TrackingManager functionality."""

    @pytest.fixture
    async def tracking_manager(self):
        """Create configured tracking manager for testing."""
        config = TrackingConfig(
            enabled=True,
            monitoring_interval_seconds=1,  # Fast for testing
            auto_validation_enabled=True,
        )

        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.health_check = AsyncMock(return_value={"status": "healthy"})

        manager = TrackingManager(
            config=config,
            database_manager=mock_db,
        )

        # Mock components
        manager._prediction_validator = MagicMock(spec=PredictionValidator)
        manager._accuracy_tracker = MagicMock(spec=AccuracyTracker)
        manager._drift_detector = MagicMock(spec=ConceptDriftDetector)
        manager._retrainer = MagicMock(spec=AdaptiveRetrainer)

        return manager

    async def test_prediction_recording_basic(self, tracking_manager):
        """Test basic prediction recording functionality."""
        prediction_data = {
            "room_id": "living_room",
            "model_type": ModelType.ENSEMBLE,
            "predicted_time": datetime.now(UTC),
            "confidence": 0.85,
        }

        tracking_manager._prediction_validator.record_prediction = AsyncMock()

        await tracking_manager.record_prediction(**prediction_data)

        tracking_manager._prediction_validator.record_prediction.assert_called_once()

    async def test_prediction_recording_disabled_tracking(self, mock_database_manager):
        """Test prediction recording with disabled tracking."""
        disabled_config = TrackingConfig(enabled=False)
        manager = TrackingManager(
            config=disabled_config,
            database_manager=mock_database_manager,
        )

        # Should not raise error but should not record
        await manager.record_prediction(
            room_id="test_room",
            model_type=ModelType.LSTM,
            predicted_time=datetime.now(UTC),
            confidence=0.8,
        )

        # No components should be called since tracking is disabled

    async def test_validation_basic(self, tracking_manager):
        """Test basic validation functionality."""
        tracking_manager._prediction_validator.validate_prediction = AsyncMock()

        await tracking_manager.validate_prediction(
            room_id="bedroom",
            actual_time=datetime.now(UTC),
        )

        tracking_manager._prediction_validator.validate_prediction.assert_called_once()

    async def test_get_accuracy_metrics(self, tracking_manager):
        """Test getting accuracy metrics."""
        mock_metrics = AccuracyMetrics(
            total_predictions=100,
            validated_predictions=95,
            accuracy_rate=85.0,
            mean_error_minutes=10.5,
            predictions_per_hour=4.2,
            confidence_calibration_score=0.88,
        )

        tracking_manager._prediction_validator.get_accuracy_metrics = AsyncMock(
            return_value=mock_metrics
        )

        result = await tracking_manager.get_accuracy_metrics(room_id="kitchen")

        assert result == mock_metrics
        tracking_manager._prediction_validator.get_accuracy_metrics.assert_called_once_with(
            room_id="kitchen", hours=24, model_type=None
        )

    async def test_get_real_time_metrics(self, tracking_manager):
        """Test getting real-time tracking metrics."""
        from src.adaptation.tracker import RealTimeMetrics

        mock_rt_metrics = RealTimeMetrics(
            room_id="office",
            window_6h_accuracy=82.0,
            window_24h_accuracy=79.0,
        )

        tracking_manager._accuracy_tracker.get_real_time_metrics = AsyncMock(
            return_value=mock_rt_metrics
        )

        result = await tracking_manager.get_real_time_metrics(room_id="office")

        assert result == mock_rt_metrics
        tracking_manager._accuracy_tracker.get_real_time_metrics.assert_called_once_with(
            room_id="office", model_type=None
        )

    async def test_get_active_alerts(self, tracking_manager):
        """Test getting active alerts."""
        from src.adaptation.tracker import AccuracyAlert

        mock_alerts = [
            AccuracyAlert(
                alert_id="test_alert",
                room_id="living_room",
                severity=AlertSeverity.WARNING,
                message="Test alert",
            )
        ]

        tracking_manager._accuracy_tracker.get_active_alerts = AsyncMock(
            return_value=mock_alerts
        )

        result = await tracking_manager.get_active_alerts(room_id="living_room")

        assert result == mock_alerts
        tracking_manager._accuracy_tracker.get_active_alerts.assert_called_once()

    async def test_acknowledge_alert(self, tracking_manager):
        """Test acknowledging an alert."""
        tracking_manager._accuracy_tracker.acknowledge_alert = AsyncMock(
            return_value=True
        )

        result = await tracking_manager.acknowledge_alert(
            alert_id="test_alert", acknowledged_by="operator"
        )

        assert result is True
        tracking_manager._accuracy_tracker.acknowledge_alert.assert_called_once_with(
            "test_alert", acknowledged_by="operator"
        )

    async def test_manual_drift_detection(self, tracking_manager):
        """Test manual drift detection trigger."""
        mock_drift_metrics = DriftMetrics(
            severity=DriftSeverity.MODERATE,
            drift_score=0.65,
            baseline_accuracy=82.0,
            current_accuracy=75.0,
            drift_indicators={"pattern_change": 0.7},
        )

        tracking_manager._drift_detector.detect_drift = AsyncMock(
            return_value=mock_drift_metrics
        )

        result = await tracking_manager.detect_drift(room_id="test_room")

        assert result == mock_drift_metrics
        tracking_manager._drift_detector.detect_drift.assert_called_once()

    async def test_manual_retraining_request(self, tracking_manager):
        """Test manual retraining request."""
        tracking_manager._retrainer.schedule_retraining = AsyncMock(
            return_value="retraining_job_123"
        )

        result = await tracking_manager.request_retraining(
            room_id="bedroom", reason="manual_request"
        )

        assert result == "retraining_job_123"
        tracking_manager._retrainer.schedule_retraining.assert_called_once()

    async def test_get_tracking_status(self, tracking_manager):
        """Test getting comprehensive tracking status."""
        # Mock all status components
        tracking_manager._prediction_validator.get_total_predictions = AsyncMock(
            return_value=150
        )
        tracking_manager._accuracy_tracker.get_tracker_stats = AsyncMock(
            return_value={"active_alerts": 2, "rooms_monitored": 5}
        )
        tracking_manager._drift_detector.get_detector_status = AsyncMock(
            return_value={"last_check": datetime.now(UTC).isoformat()}
        )
        tracking_manager._retrainer.get_retraining_status = AsyncMock(
            return_value={"active_jobs": 1, "completed_jobs": 10}
        )

        status = await tracking_manager.get_tracking_status()

        assert "enabled" in status
        assert "prediction_validator" in status
        assert "accuracy_tracker" in status
        assert "drift_detector" in status
        assert "retrainer" in status
        assert status["enabled"] is True


class TestTrackingManagerErrorHandling:
    """Unit tests for error handling in TrackingManager."""

    @pytest.fixture
    def error_manager(self):
        """Create manager for error testing."""
        config = TrackingConfig(enabled=True)
        mock_db = MagicMock(spec=DatabaseManager)

        manager = TrackingManager(
            config=config,
            database_manager=mock_db,
        )

        # Add mock components that will fail
        manager._prediction_validator = MagicMock(spec=PredictionValidator)
        manager._accuracy_tracker = MagicMock(spec=AccuracyTracker)

        return manager

    async def test_record_prediction_error_handling(self, error_manager):
        """Test error handling in record_prediction."""
        error_manager._prediction_validator.record_prediction = AsyncMock(
            side_effect=Exception("Database error")
        )

        with pytest.raises(TrackingManagerError):
            await error_manager.record_prediction(
                room_id="test_room",
                model_type=ModelType.LSTM,
                predicted_time=datetime.now(UTC),
                confidence=0.8,
            )

    async def test_validate_prediction_error_handling(self, error_manager):
        """Test error handling in validate_prediction."""
        error_manager._prediction_validator.validate_prediction = AsyncMock(
            side_effect=Exception("Validation error")
        )

        with pytest.raises(TrackingManagerError):
            await error_manager.validate_prediction(
                room_id="test_room",
                actual_time=datetime.now(UTC),
            )

    async def test_get_metrics_error_handling(self, error_manager):
        """Test error handling in get_accuracy_metrics."""
        error_manager._prediction_validator.get_accuracy_metrics = AsyncMock(
            side_effect=Exception("Metrics error")
        )

        with pytest.raises(TrackingManagerError):
            await error_manager.get_accuracy_metrics(room_id="test_room")

    async def test_component_initialization_error(self):
        """Test error handling during component initialization."""
        config = TrackingConfig(enabled=True)

        # Mock database that fails health check
        mock_db = MagicMock(spec=DatabaseManager)
        mock_db.health_check = AsyncMock(side_effect=Exception("DB connection failed"))

        manager = TrackingManager(
            config=config,
            database_manager=mock_db,
        )

        with pytest.raises(TrackingManagerError):
            await manager.start()


class TestTrackingManagerValidation:
    """Unit tests for input validation in TrackingManager."""

    @pytest.fixture
    def validation_manager(self):
        """Create manager for validation testing."""
        config = TrackingConfig(enabled=True)
        mock_db = MagicMock(spec=DatabaseManager)

        manager = TrackingManager(
            config=config,
            database_manager=mock_db,
        )

        manager._prediction_validator = MagicMock(spec=PredictionValidator)
        return manager

    async def test_record_prediction_validation(self, validation_manager):
        """Test input validation for record_prediction."""
        # Invalid room_id
        with pytest.raises(ValueError, match="room_id"):
            await validation_manager.record_prediction(
                room_id="",  # Empty string
                model_type=ModelType.LSTM,
                predicted_time=datetime.now(UTC),
                confidence=0.8,
            )

        # Invalid confidence
        with pytest.raises(ValueError, match="confidence"):
            await validation_manager.record_prediction(
                room_id="test_room",
                model_type=ModelType.LSTM,
                predicted_time=datetime.now(UTC),
                confidence=1.5,  # > 1.0
            )

        # Invalid confidence (negative)
        with pytest.raises(ValueError, match="confidence"):
            await validation_manager.record_prediction(
                room_id="test_room",
                model_type=ModelType.LSTM,
                predicted_time=datetime.now(UTC),
                confidence=-0.1,  # < 0.0
            )

    async def test_validate_prediction_validation(self, validation_manager):
        """Test input validation for validate_prediction."""
        # Invalid room_id
        with pytest.raises(ValueError, match="room_id"):
            await validation_manager.validate_prediction(
                room_id=None,  # None
                actual_time=datetime.now(UTC),
            )

        # Invalid actual_time
        with pytest.raises(ValueError, match="actual_time"):
            await validation_manager.validate_prediction(
                room_id="test_room",
                actual_time=None,  # None
            )

    async def test_get_accuracy_metrics_validation(self, validation_manager):
        """Test input validation for get_accuracy_metrics."""
        # Invalid hours
        with pytest.raises(ValueError, match="hours"):
            await validation_manager.get_accuracy_metrics(
                room_id="test_room",
                hours=0,  # Must be > 0
            )

        # Invalid hours (negative)
        with pytest.raises(ValueError, match="hours"):
            await validation_manager.get_accuracy_metrics(
                room_id="test_room",
                hours=-5,  # Must be > 0
            )


class TestTrackingManagerUtilities:
    """Unit tests for TrackingManager utility methods."""

    def test_tracking_manager_error_creation(self):
        """Test TrackingManagerError creation."""
        error = TrackingManagerError("Test error message")

        assert "Test error message" in str(error)
        assert error.severity == ErrorSeverity.MEDIUM

    def test_tracking_manager_error_with_severity(self):
        """Test TrackingManagerError with custom severity."""
        error = TrackingManagerError("Critical error", severity=ErrorSeverity.CRITICAL)

        assert "Critical error" in str(error)
        assert error.severity == ErrorSeverity.CRITICAL

    def test_config_validation(self):
        """Test TrackingConfig validation edge cases."""
        # Test minimum values
        config = TrackingConfig(
            monitoring_interval_seconds=1,  # Minimum allowed
            validation_window_minutes=1,  # Minimum allowed
            max_stored_alerts=1,  # Minimum allowed
            trend_analysis_points=2,  # Minimum allowed
        )

        assert config.monitoring_interval_seconds == 1
        assert config.validation_window_minutes == 1
        assert config.max_stored_alerts == 1
        assert config.trend_analysis_points == 2
