"""
Comprehensive test suite for MonitoringEnhancedTrackingManager.

Tests all functionality from MonitoringEnhancedTrackingManager class including:
- Initialization and method wrapping
- Prediction recording with monitoring
- Validation with monitoring
- System lifecycle with monitoring integration
- Concept drift recording
- Feature computation tracking
- Database operation monitoring
- MQTT publish tracking
- Connection status monitoring
- Error handling and exception scenarios
- Performance and timeout scenarios
- Integration patterns and edge cases
"""

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.adaptation.monitoring_enhanced_tracking import (
    MonitoringEnhancedTrackingManager,
    create_monitoring_enhanced_tracking_manager,
    get_enhanced_tracking_manager,
)
from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.constants import ModelType
from src.models.base.predictor import PredictionResult
from src.utils.monitoring_integration import get_monitoring_integration


class TestMonitoringEnhancedTrackingManager:
    """Comprehensive tests for MonitoringEnhancedTrackingManager."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Create mock TrackingManager."""
        tracking_manager = MagicMock(spec=TrackingManager)
        tracking_manager.record_prediction = AsyncMock(return_value="pred_123")
        tracking_manager.validate_prediction = AsyncMock(
            return_value={
                "accuracy_minutes": 5.2,
                "prediction_type": "next_occupied",
                "model_type": "ensemble",
                "confidence": 0.85,
            }
        )
        tracking_manager.start_tracking = AsyncMock(return_value=True)
        tracking_manager.stop_tracking = AsyncMock(return_value=True)
        tracking_manager.get_system_status = AsyncMock(
            return_value={"status": "active", "rooms": ["living_room"]}
        )
        tracking_manager.get_accuracy_metrics = AsyncMock(
            return_value={"living_room": MagicMock(accuracy_percentage=88.5)}
        )
        return tracking_manager

    @pytest.fixture
    def mock_monitoring_integration(self):
        """Create mock monitoring integration."""
        monitor = MagicMock()
        monitor.track_prediction_operation = AsyncMock()
        monitor.track_training_operation = AsyncMock()
        monitor.start_monitoring = AsyncMock()
        monitor.stop_monitoring = AsyncMock()
        monitor.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "health_details": {"cpu_percent": 45, "memory_percent": 60},
                    "monitoring_active": True,
                    "alert_system": {"active_alerts": 2},
                },
                "timestamp": datetime.now().isoformat(),
            }
        )

        # Setup alert manager mock
        alert_manager = MagicMock()
        alert_manager.trigger_alert = AsyncMock(return_value="alert_123")
        monitor.alert_manager = alert_manager

        # Setup context manager behavior
        monitor.track_prediction_operation.return_value.__aenter__ = AsyncMock()
        monitor.track_prediction_operation.return_value.__aexit__ = AsyncMock()
        monitor.track_training_operation.return_value.__aenter__ = AsyncMock()
        monitor.track_training_operation.return_value.__aexit__ = AsyncMock()

        # Setup tracking methods
        monitor.record_prediction_accuracy = MagicMock()
        monitor.record_concept_drift = MagicMock()
        monitor.record_feature_computation = MagicMock()
        monitor.record_database_operation = MagicMock()
        monitor.record_mqtt_publish = MagicMock()
        monitor.update_connection_status = MagicMock()

        return monitor

    @pytest.fixture
    def enhanced_tracking_manager(
        self, mock_tracking_manager, mock_monitoring_integration
    ):
        """Create MonitoringEnhancedTrackingManager for testing."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)
            return manager

    @pytest.fixture
    def sample_prediction_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=15),
            confidence=0.85,
            prediction_type="next_occupied",
            model_metadata={"model": "ensemble", "version": "1.0"},
        )

    # Initialization and Setup Tests

    def test_initialization(self, mock_tracking_manager, mock_monitoring_integration):
        """Test proper initialization of enhanced tracking manager."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):

            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)

            # Verify tracking manager assignment
            assert manager.tracking_manager is mock_tracking_manager
            assert manager.monitoring_integration is mock_monitoring_integration

            # Verify original methods are stored
            assert "record_prediction" in manager._original_methods
            assert "validate_prediction" in manager._original_methods
            assert "start_tracking" in manager._original_methods
            assert "stop_tracking" in manager._original_methods

            # Verify methods are wrapped
            assert (
                manager.tracking_manager.record_prediction
                != mock_tracking_manager.record_prediction
            )
            assert (
                manager.tracking_manager.validate_prediction
                != mock_tracking_manager.validate_prediction
            )

    def test_method_wrapping(self, mock_tracking_manager, mock_monitoring_integration):
        """Test that tracking manager methods are properly wrapped."""
        with patch(
            "src.adaptation.monitoring_integration.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            original_record = mock_tracking_manager.record_prediction
            original_validate = mock_tracking_manager.validate_prediction

            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)

            # Verify original methods are stored
            assert manager._original_methods["record_prediction"] is original_record
            assert manager._original_methods["validate_prediction"] is original_validate

            # Verify methods are replaced with monitored versions
            assert (
                mock_tracking_manager.record_prediction
                == manager._monitored_record_prediction
            )
            assert (
                mock_tracking_manager.validate_prediction
                == manager._monitored_validate_prediction
            )

    # Prediction Recording Tests

    @pytest.mark.asyncio
    async def test_monitored_record_prediction_success(
        self, enhanced_tracking_manager, sample_prediction_result
    ):
        """Test successful prediction recording with monitoring."""
        room_id = "living_room"
        model_type = ModelType.ENSEMBLE

        result = await enhanced_tracking_manager._monitored_record_prediction(
            room_id, sample_prediction_result, model_type
        )

        # Verify original method was called
        enhanced_tracking_manager.tracking_manager.record_prediction.assert_called_once()

        # Verify monitoring integration was used
        enhanced_tracking_manager.monitoring_integration.track_prediction_operation.assert_called_once_with(
            room_id=room_id, prediction_type="next_occupied", model_type="ensemble"
        )

        # Verify accuracy recording
        enhanced_tracking_manager.monitoring_integration.record_prediction_accuracy.assert_called_once_with(
            room_id=room_id,
            model_type="ensemble",
            prediction_type="next_occupied",
            accuracy_minutes=0,
            confidence=0.85,
        )

        assert result == "pred_123"

    @pytest.mark.asyncio
    async def test_monitored_record_prediction_with_kwargs(
        self, enhanced_tracking_manager, sample_prediction_result
    ):
        """Test prediction recording with additional kwargs."""
        room_id = "bedroom"
        model_type = ModelType.LSTM
        kwargs = {"force": True, "retrain_if_needed": False}

        await enhanced_tracking_manager._monitored_record_prediction(
            room_id, sample_prediction_result, model_type, **kwargs
        )

        # Verify kwargs are passed through
        enhanced_tracking_manager._original_methods[
            "record_prediction"
        ].assert_called_once_with(
            room_id, sample_prediction_result, model_type, **kwargs
        )

    @pytest.mark.asyncio
    async def test_monitored_record_prediction_no_confidence(
        self, enhanced_tracking_manager
    ):
        """Test prediction recording when confidence is not available."""
        room_id = "kitchen"
        prediction_result = PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=20),
            confidence=None,
            prediction_type="next_vacant",
        )

        await enhanced_tracking_manager._monitored_record_prediction(
            room_id, prediction_result
        )

        # Should still call monitoring but not record accuracy
        enhanced_tracking_manager.monitoring_integration.track_prediction_operation.assert_called_once()
        enhanced_tracking_manager.monitoring_integration.record_prediction_accuracy.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitored_record_prediction_string_model_type(
        self, enhanced_tracking_manager, sample_prediction_result
    ):
        """Test prediction recording with string model type."""
        room_id = "office"
        model_type = "xgboost"  # String instead of enum

        await enhanced_tracking_manager._monitored_record_prediction(
            room_id, sample_prediction_result, model_type
        )

        enhanced_tracking_manager.monitoring_integration.track_prediction_operation.assert_called_once_with(
            room_id=room_id, prediction_type="next_occupied", model_type="xgboost"
        )

    # Prediction Validation Tests

    @pytest.mark.asyncio
    async def test_monitored_validate_prediction_success(
        self, enhanced_tracking_manager
    ):
        """Test successful prediction validation with monitoring."""
        room_id = "dining_room"
        actual_time = datetime.now()

        result = await enhanced_tracking_manager._monitored_validate_prediction(
            room_id, actual_time
        )

        # Verify original method was called
        enhanced_tracking_manager._original_methods[
            "validate_prediction"
        ].assert_called_once_with(room_id, actual_time)

        # Verify monitoring recorded the accuracy
        enhanced_tracking_manager.monitoring_integration.record_prediction_accuracy.assert_called_once_with(
            room_id=room_id,
            model_type="ensemble",
            prediction_type="next_occupied",
            accuracy_minutes=5.2,
            confidence=0.85,
        )

        assert result == {
            "accuracy_minutes": 5.2,
            "prediction_type": "next_occupied",
            "model_type": "ensemble",
            "confidence": 0.85,
        }

    @pytest.mark.asyncio
    async def test_monitored_validate_prediction_with_kwargs(
        self, enhanced_tracking_manager
    ):
        """Test prediction validation with additional kwargs."""
        room_id = "guest_room"
        actual_time = datetime.now()
        kwargs = {"tolerance_minutes": 10, "update_metrics": True}

        await enhanced_tracking_manager._monitored_validate_prediction(
            room_id, actual_time, **kwargs
        )

        enhanced_tracking_manager._original_methods[
            "validate_prediction"
        ].assert_called_once_with(room_id, actual_time, **kwargs)

    @pytest.mark.asyncio
    async def test_monitored_validate_prediction_non_dict_result(
        self, enhanced_tracking_manager
    ):
        """Test validation when result is not a dictionary."""
        room_id = "bathroom"
        actual_time = datetime.now()

        # Mock returning a non-dict result
        enhanced_tracking_manager._original_methods[
            "validate_prediction"
        ].return_value = "validation_complete"

        result = await enhanced_tracking_manager._monitored_validate_prediction(
            room_id, actual_time
        )

        # Should not try to record accuracy metrics
        enhanced_tracking_manager.monitoring_integration.record_prediction_accuracy.assert_not_called()
        assert result == "validation_complete"

    @pytest.mark.asyncio
    async def test_monitored_validate_prediction_error(self, enhanced_tracking_manager):
        """Test validation error handling and alerting."""
        room_id = "hallway"
        actual_time = datetime.now()

        # Mock validation error
        validation_error = RuntimeError("Database connection lost")
        enhanced_tracking_manager._original_methods[
            "validate_prediction"
        ].side_effect = validation_error

        with pytest.raises(RuntimeError, match="Database connection lost"):
            await enhanced_tracking_manager._monitored_validate_prediction(
                room_id, actual_time
            )

        # Verify alert was triggered
        enhanced_tracking_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
        alert_call = (
            enhanced_tracking_manager.monitoring_integration.alert_manager.trigger_alert.call_args
        )
        assert alert_call[1]["rule_name"] == "prediction_validation_error"
        assert room_id in alert_call[1]["title"]
        assert "Database connection lost" in alert_call[1]["message"]

    # System Lifecycle Tests

    @pytest.mark.asyncio
    async def test_monitored_start_tracking_success(self, enhanced_tracking_manager):
        """Test successful tracking system start with monitoring."""
        kwargs = {"background_tasks": True, "health_checks": True}

        result = await enhanced_tracking_manager._monitored_start_tracking(**kwargs)

        # Verify monitoring started first
        enhanced_tracking_manager.monitoring_integration.start_monitoring.assert_called_once()

        # Verify original start tracking called
        enhanced_tracking_manager._original_methods[
            "start_tracking"
        ].assert_called_once_with(**kwargs)

        # Verify success alert
        enhanced_tracking_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
        alert_call = (
            enhanced_tracking_manager.monitoring_integration.alert_manager.trigger_alert.call_args
        )
        assert alert_call[1]["rule_name"] == "system_startup_success"

        assert result is True

    @pytest.mark.asyncio
    async def test_monitored_start_tracking_error(self, enhanced_tracking_manager):
        """Test tracking system start error handling."""
        # Mock start tracking error
        start_error = ConnectionError("Cannot connect to database")
        enhanced_tracking_manager._original_methods["start_tracking"].side_effect = (
            start_error
        )

        with pytest.raises(ConnectionError, match="Cannot connect to database"):
            await enhanced_tracking_manager._monitored_start_tracking()

        # Verify error alert was triggered
        enhanced_tracking_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
        alert_call = (
            enhanced_tracking_manager.monitoring_integration.alert_manager.trigger_alert.call_args
        )
        assert alert_call[1]["rule_name"] == "system_startup_error"
        assert "Cannot connect to database" in alert_call[1]["message"]

    @pytest.mark.asyncio
    async def test_monitored_stop_tracking_success(self, enhanced_tracking_manager):
        """Test successful tracking system stop with monitoring cleanup."""
        kwargs = {"graceful_shutdown": True}

        result = await enhanced_tracking_manager._monitored_stop_tracking(**kwargs)

        # Verify original stop tracking called first
        enhanced_tracking_manager._original_methods[
            "stop_tracking"
        ].assert_called_once_with(**kwargs)

        # Verify monitoring stopped after
        enhanced_tracking_manager.monitoring_integration.stop_monitoring.assert_called_once()

        assert result is True

    @pytest.mark.asyncio
    async def test_monitored_stop_tracking_error(self, enhanced_tracking_manager):
        """Test tracking system stop with error - still attempts monitoring cleanup."""
        stop_error = RuntimeError("Error during shutdown")
        enhanced_tracking_manager._original_methods["stop_tracking"].side_effect = (
            stop_error
        )

        with pytest.raises(RuntimeError, match="Error during shutdown"):
            await enhanced_tracking_manager._monitored_stop_tracking()

        # Should still attempt to stop monitoring
        enhanced_tracking_manager.monitoring_integration.stop_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitored_stop_tracking_monitoring_error(
        self, enhanced_tracking_manager
    ):
        """Test stop tracking when monitoring stop also fails."""
        stop_error = RuntimeError("Tracking error")
        monitoring_error = RuntimeError("Monitoring error")

        enhanced_tracking_manager._original_methods["stop_tracking"].side_effect = (
            stop_error
        )
        enhanced_tracking_manager.monitoring_integration.stop_monitoring.side_effect = (
            monitoring_error
        )

        # Should raise the original tracking error, not monitoring error
        with pytest.raises(RuntimeError, match="Tracking error"):
            await enhanced_tracking_manager._monitored_stop_tracking()

        # Monitoring stop should still be attempted
        enhanced_tracking_manager.monitoring_integration.stop_monitoring.assert_called_once()

    # Concept Drift Recording Tests

    def test_record_concept_drift(self, enhanced_tracking_manager):
        """Test concept drift recording with monitoring integration."""
        room_id = "living_room"
        drift_type = "seasonal_pattern_change"
        severity = 0.75
        action_taken = "incremental_retraining"

        enhanced_tracking_manager.record_concept_drift(
            room_id, drift_type, severity, action_taken
        )

        # Verify monitoring integration called
        enhanced_tracking_manager.monitoring_integration.record_concept_drift.assert_called_once_with(
            room_id=room_id,
            drift_type=drift_type,
            severity=severity,
            action_taken=action_taken,
        )

    def test_record_concept_drift_with_tracking_manager_support(
        self, enhanced_tracking_manager
    ):
        """Test concept drift recording when tracking manager also supports it."""
        room_id = "bedroom"
        drift_type = "occupancy_pattern_shift"
        severity = 0.85
        action_taken = "full_retraining"

        # Add drift recording method to tracking manager
        enhanced_tracking_manager.tracking_manager.record_concept_drift = MagicMock()

        enhanced_tracking_manager.record_concept_drift(
            room_id, drift_type, severity, action_taken
        )

        # Both should be called
        enhanced_tracking_manager.monitoring_integration.record_concept_drift.assert_called_once()
        enhanced_tracking_manager.tracking_manager.record_concept_drift.assert_called_once_with(
            room_id, drift_type, severity, action_taken
        )

    # Feature and Operations Recording Tests

    def test_record_feature_computation(self, enhanced_tracking_manager):
        """Test feature computation metrics recording."""
        room_id = "kitchen"
        feature_type = "temporal_features"
        duration = 0.234

        enhanced_tracking_manager.record_feature_computation(
            room_id, feature_type, duration
        )

        enhanced_tracking_manager.monitoring_integration.record_feature_computation.assert_called_once_with(
            room_id=room_id, feature_type=feature_type, duration=duration
        )

    def test_record_database_operation(self, enhanced_tracking_manager):
        """Test database operation metrics recording."""
        operation_type = "bulk_insert"
        table = "sensor_events"
        duration = 1.456
        status = "success"

        enhanced_tracking_manager.record_database_operation(
            operation_type, table, duration, status
        )

        enhanced_tracking_manager.monitoring_integration.record_database_operation.assert_called_once_with(
            operation_type=operation_type, table=table, duration=duration, status=status
        )

    def test_record_database_operation_default_status(self, enhanced_tracking_manager):
        """Test database operation recording with default status."""
        enhanced_tracking_manager.record_database_operation(
            "select", "room_states", 0.045
        )

        enhanced_tracking_manager.monitoring_integration.record_database_operation.assert_called_once_with(
            operation_type="select",
            table="room_states",
            duration=0.045,
            status="success",
        )

    def test_record_mqtt_publish(self, enhanced_tracking_manager):
        """Test MQTT publishing metrics recording."""
        topic_type = "prediction"
        room_id = "office"
        status = "success"

        enhanced_tracking_manager.record_mqtt_publish(topic_type, room_id, status)

        enhanced_tracking_manager.monitoring_integration.record_mqtt_publish.assert_called_once_with(
            topic_type=topic_type, room_id=room_id, status=status
        )

    def test_record_mqtt_publish_default_status(self, enhanced_tracking_manager):
        """Test MQTT publish recording with default status."""
        enhanced_tracking_manager.record_mqtt_publish("accuracy", "guest_room")

        enhanced_tracking_manager.monitoring_integration.record_mqtt_publish.assert_called_once_with(
            topic_type="accuracy", room_id="guest_room", status="success"
        )

    def test_update_connection_status(self, enhanced_tracking_manager):
        """Test connection status updates."""
        connection_type = "home_assistant"
        connected = True

        enhanced_tracking_manager.update_connection_status(connection_type, connected)

        enhanced_tracking_manager.monitoring_integration.update_connection_status.assert_called_once_with(
            connection_type=connection_type, connected=connected
        )

    # Monitoring Status Tests

    @pytest.mark.asyncio
    async def test_get_monitoring_status_success(self, enhanced_tracking_manager):
        """Test comprehensive monitoring status retrieval."""
        # Mock monitoring status
        mock_monitoring_status = {
            "health": "good",
            "active_alerts": 0,
            "monitoring_active": True,
        }
        enhanced_tracking_manager.monitoring_integration.get_monitoring_status.return_value = (
            mock_monitoring_status
        )

        # Mock tracking status
        mock_tracking_status = {
            "active_rooms": ["living_room", "bedroom"],
            "predictions_today": 145,
        }
        enhanced_tracking_manager.tracking_manager.get_system_status.return_value = (
            mock_tracking_status
        )

        result = await enhanced_tracking_manager.get_monitoring_status()

        # Verify comprehensive status
        assert result["monitoring"] == mock_monitoring_status
        assert result["tracking"] == mock_tracking_status
        assert result["integrated"] is True
        assert "timestamp" in result

        # Verify ISO format timestamp
        datetime.fromisoformat(result["timestamp"])

    @pytest.mark.asyncio
    async def test_get_monitoring_status_tracking_error(
        self, enhanced_tracking_manager
    ):
        """Test monitoring status when tracking status fails."""
        # Mock monitoring status success
        mock_monitoring_status = {"health": "good"}
        enhanced_tracking_manager.monitoring_integration.get_monitoring_status.return_value = (
            mock_monitoring_status
        )

        # Mock tracking status error
        enhanced_tracking_manager.tracking_manager.get_system_status.side_effect = (
            RuntimeError("Tracking unavailable")
        )

        result = await enhanced_tracking_manager.get_monitoring_status()

        # Should handle error gracefully
        assert result["monitoring"] == mock_monitoring_status
        assert result["tracking"]["error"] == "Failed to get tracking status"
        assert result["integrated"] is True

    @pytest.mark.asyncio
    async def test_get_monitoring_status_no_tracking_method(
        self, enhanced_tracking_manager
    ):
        """Test monitoring status when tracking manager doesn't support status."""
        # Remove get_system_status method
        delattr(enhanced_tracking_manager.tracking_manager, "get_system_status")

        result = await enhanced_tracking_manager.get_monitoring_status()

        # Should work with empty tracking status
        assert result["tracking"] == {}
        assert result["integrated"] is True

    # Model Training Context Manager Tests

    @pytest.mark.asyncio
    async def test_track_model_training_context(self, enhanced_tracking_manager):
        """Test model training tracking context manager."""
        room_id = "dining_room"
        model_type = "lstm"
        training_type = "incremental"

        async with enhanced_tracking_manager.track_model_training(
            room_id, model_type, training_type
        ):
            # Context should be active
            pass

        enhanced_tracking_manager.monitoring_integration.track_training_operation.assert_called_once_with(
            room_id=room_id, model_type=model_type, training_type=training_type
        )

    @pytest.mark.asyncio
    async def test_track_model_training_default_type(self, enhanced_tracking_manager):
        """Test model training tracking with default training type."""
        room_id = "hallway"
        model_type = "xgboost"

        async with enhanced_tracking_manager.track_model_training(room_id, model_type):
            pass

        enhanced_tracking_manager.monitoring_integration.track_training_operation.assert_called_once_with(
            room_id=room_id, model_type=model_type, training_type="retraining"
        )

    # Attribute Delegation Tests

    def test_getattr_delegation(self, enhanced_tracking_manager):
        """Test that unknown attributes are delegated to tracking manager."""
        # Add custom attribute to tracking manager
        enhanced_tracking_manager.tracking_manager.custom_method = MagicMock(
            return_value="custom_result"
        )

        # Access through enhanced manager
        result = enhanced_tracking_manager.custom_method()

        assert result == "custom_result"
        enhanced_tracking_manager.tracking_manager.custom_method.assert_called_once()

    def test_getattr_delegation_error(self, enhanced_tracking_manager):
        """Test attribute delegation when attribute doesn't exist."""
        with pytest.raises(AttributeError):
            enhanced_tracking_manager.nonexistent_method()

    # Performance and Timeout Tests

    @pytest.mark.asyncio
    async def test_prediction_monitoring_timeout_handling(
        self, enhanced_tracking_manager, sample_prediction_result
    ):
        """Test handling of monitoring timeouts during prediction recording."""
        room_id = "timeout_room"

        # Mock slow monitoring operation
        async def slow_context_manager(*args, **kwargs):
            await asyncio.sleep(0.1)  # Simulate slow operation
            return AsyncMock()

        enhanced_tracking_manager.monitoring_integration.track_prediction_operation.return_value.__aenter__ = (
            slow_context_manager
        )

        # Should still complete successfully
        result = await enhanced_tracking_manager._monitored_record_prediction(
            room_id, sample_prediction_result
        )

        assert result == "pred_123"

    @pytest.mark.asyncio
    async def test_concurrent_prediction_recording(
        self, enhanced_tracking_manager, sample_prediction_result
    ):
        """Test concurrent prediction recording operations."""
        room_ids = ["room1", "room2", "room3", "room4", "room5"]

        # Create concurrent prediction recording tasks
        tasks = [
            enhanced_tracking_manager._monitored_record_prediction(
                room_id, sample_prediction_result
            )
            for room_id in room_ids
        ]

        results = await asyncio.gather(*tasks)

        # All should succeed
        assert all(result == "pred_123" for result in results)

        # All monitoring calls should have been made
        assert (
            enhanced_tracking_manager.monitoring_integration.track_prediction_operation.call_count
            == len(room_ids)
        )

    # Edge Cases and Error Scenarios

    @pytest.mark.asyncio
    async def test_monitoring_integration_none_error(self, mock_tracking_manager):
        """Test behavior when monitoring integration is None."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=None,
        ):
            with pytest.raises(AttributeError):
                MonitoringEnhancedTrackingManager(mock_tracking_manager)

    def test_empty_prediction_result_attributes(self, enhanced_tracking_manager):
        """Test handling of prediction result with missing attributes."""
        room_id = "test_room"

        # Create prediction result without standard attributes
        prediction_result = MagicMock()
        prediction_result.prediction_type = None
        prediction_result.confidence = None

        # Should handle gracefully without errors
        asyncio.run(
            enhanced_tracking_manager._monitored_record_prediction(
                room_id, prediction_result
            )
        )

    @pytest.mark.asyncio
    async def test_monitoring_context_manager_error(
        self, enhanced_tracking_manager, sample_prediction_result
    ):
        """Test handling when monitoring context manager fails."""
        room_id = "error_room"

        # Mock context manager error
        enhanced_tracking_manager.monitoring_integration.track_prediction_operation.side_effect = RuntimeError(
            "Context error"
        )

        with pytest.raises(RuntimeError, match="Context error"):
            await enhanced_tracking_manager._monitored_record_prediction(
                room_id, sample_prediction_result
            )


class TestFactoryFunctions:
    """Test factory functions for creating enhanced tracking managers."""

    @pytest.fixture
    def mock_tracking_config(self):
        """Create mock tracking configuration."""
        config = MagicMock(spec=TrackingConfig)
        config.accuracy_threshold_minutes = 15
        config.drift_detection_enabled = True
        return config

    @patch("src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration")
    @patch("src.adaptation.monitoring_enhanced_tracking.TrackingManager")
    def test_create_monitoring_enhanced_tracking_manager(
        self, mock_tracking_manager_class, mock_get_monitoring, mock_tracking_config
    ):
        """Test factory function for creating enhanced tracking manager."""
        mock_tracking_instance = MagicMock()
        mock_tracking_manager_class.return_value = mock_tracking_instance
        mock_monitoring = MagicMock()
        mock_get_monitoring.return_value = mock_monitoring

        result = create_monitoring_enhanced_tracking_manager(
            config=mock_tracking_config, additional_param="test"
        )

        # Verify TrackingManager created with correct parameters
        mock_tracking_manager_class.assert_called_once_with(
            config=mock_tracking_config, additional_param="test"
        )

        # Verify enhanced manager created
        assert isinstance(result, MonitoringEnhancedTrackingManager)
        assert result.tracking_manager is mock_tracking_instance

    @patch(
        "src.adaptation.monitoring_enhanced_tracking.create_monitoring_enhanced_tracking_manager"
    )
    def test_get_enhanced_tracking_manager(
        self, mock_create_enhanced, mock_tracking_config
    ):
        """Test convenience function for getting enhanced tracking manager."""
        mock_enhanced_manager = MagicMock()
        mock_create_enhanced.return_value = mock_enhanced_manager

        result = get_enhanced_tracking_manager(
            config=mock_tracking_config, test_param="value"
        )

        # Verify delegation to create function
        mock_create_enhanced.assert_called_once_with(
            config=mock_tracking_config, test_param="value"
        )

        assert result is mock_enhanced_manager


class TestIntegrationScenarios:
    """Integration scenario tests for enhanced tracking manager."""

    @pytest.fixture
    def integrated_manager_setup(self):
        """Setup integrated manager with real-like mocks."""
        tracking_manager = MagicMock(spec=TrackingManager)
        monitoring_integration = MagicMock()

        # Setup realistic return values
        tracking_manager.record_prediction = AsyncMock(return_value="pred_789")
        tracking_manager.validate_prediction = AsyncMock(
            return_value={
                "accuracy_minutes": 8.3,
                "prediction_type": "next_vacant",
                "model_type": "lstm",
                "confidence": 0.92,
            }
        )
        tracking_manager.start_tracking = AsyncMock(return_value=True)
        tracking_manager.stop_tracking = AsyncMock(return_value=True)

        # Setup monitoring with context managers
        monitoring_integration.track_prediction_operation.return_value = AsyncMock()
        monitoring_integration.start_monitoring = AsyncMock()
        monitoring_integration.stop_monitoring = AsyncMock()
        monitoring_integration.alert_manager.trigger_alert = AsyncMock(
            return_value="alert_456"
        )
        monitoring_integration.record_prediction_accuracy = MagicMock()

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=monitoring_integration,
        ):
            manager = MonitoringEnhancedTrackingManager(tracking_manager)

        return manager, tracking_manager, monitoring_integration

    @pytest.mark.asyncio
    async def test_complete_prediction_lifecycle(self, integrated_manager_setup):
        """Test complete prediction lifecycle with monitoring."""
        manager, tracking_manager, monitoring_integration = integrated_manager_setup

        room_id = "integrated_room"
        prediction_result = PredictionResult(
            predicted_time=datetime.now() + timedelta(minutes=12),
            confidence=0.78,
            prediction_type="next_occupied",
        )
        actual_time = datetime.now() + timedelta(minutes=10)

        # Record prediction
        pred_id = await manager._monitored_record_prediction(
            room_id, prediction_result, ModelType.ENSEMBLE
        )

        # Validate prediction
        validation_result = await manager._monitored_validate_prediction(
            room_id, actual_time
        )

        # Verify complete flow
        assert pred_id == "pred_789"
        assert validation_result["accuracy_minutes"] == 8.3

        # Verify monitoring integration throughout
        monitoring_integration.track_prediction_operation.assert_called()
        monitoring_integration.record_prediction_accuracy.assert_called()

    @pytest.mark.asyncio
    async def test_system_lifecycle_with_monitoring(self, integrated_manager_setup):
        """Test system startup and shutdown with monitoring integration."""
        manager, tracking_manager, monitoring_integration = integrated_manager_setup

        # Start system
        await manager._monitored_start_tracking(
            enable_health_checks=True, background_tasks=True
        )

        # Stop system
        await manager._monitored_stop_tracking(graceful=True)

        # Verify monitoring lifecycle
        monitoring_integration.start_monitoring.assert_called_once()
        monitoring_integration.stop_monitoring.assert_called_once()

        # Verify alerts for system events
        assert monitoring_integration.alert_manager.trigger_alert.call_count >= 1

    @pytest.mark.asyncio
    async def test_error_recovery_scenarios(self, integrated_manager_setup):
        """Test error recovery and monitoring alerting."""
        manager, tracking_manager, monitoring_integration = integrated_manager_setup

        room_id = "error_room"

        # Simulate validation error
        tracking_manager.validate_prediction.side_effect = ConnectionError(
            "DB connection lost"
        )

        with pytest.raises(ConnectionError):
            await manager._monitored_validate_prediction(room_id, datetime.now())

        # Verify error alert was triggered
        monitoring_integration.alert_manager.trigger_alert.assert_called()
        alert_call = monitoring_integration.alert_manager.trigger_alert.call_args
        assert "prediction_validation_error" in alert_call[1]["rule_name"]
        assert "DB connection lost" in alert_call[1]["message"]

    def test_metrics_recording_integration(self, integrated_manager_setup):
        """Test comprehensive metrics recording integration."""
        manager, _, monitoring_integration = integrated_manager_setup

        # Test all monitoring recording methods
        manager.record_concept_drift("room1", "drift_type", 0.8, "action")
        manager.record_feature_computation("room1", "temporal", 1.2)
        manager.record_database_operation("insert", "events", 0.5, "success")
        manager.record_mqtt_publish("prediction", "room1", "success")
        manager.update_connection_status("ha", True)

        # Verify all monitoring calls were made
        monitoring_integration.record_concept_drift.assert_called_once()
        monitoring_integration.record_feature_computation.assert_called_once()
        monitoring_integration.record_database_operation.assert_called_once()
        monitoring_integration.record_mqtt_publish.assert_called_once()
        monitoring_integration.update_connection_status.assert_called_once()
