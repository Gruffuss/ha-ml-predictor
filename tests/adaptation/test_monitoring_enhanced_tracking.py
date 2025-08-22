"""
Comprehensive tests for src/adaptation/monitoring_enhanced_tracking.py.

Tests the MonitoringEnhancedTrackingManager wrapper and factory functions 
for complete coverage of monitoring integration capabilities.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.adaptation.monitoring_enhanced_tracking import (
    MonitoringEnhancedTrackingManager,
    create_monitoring_enhanced_tracking_manager,
    get_enhanced_tracking_manager,
)
from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.constants import ModelType
from src.models.base.predictor import PredictionResult


class TestMonitoringEnhancedTrackingManager:
    """Test suite for MonitoringEnhancedTrackingManager class."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Create a mock TrackingManager for testing."""
        manager = Mock(spec=TrackingManager)
        manager.record_prediction = AsyncMock()
        manager.validate_prediction = AsyncMock()
        manager.start_tracking = AsyncMock()
        manager.stop_tracking = AsyncMock()
        manager.get_system_status = AsyncMock(return_value={"status": "active"})
        return manager

    @pytest.fixture
    def mock_monitoring_integration(self):
        """Create a mock monitoring integration."""
        integration = Mock()
        integration.track_prediction_operation = AsyncMock()
        integration.record_prediction_accuracy = Mock()
        integration.start_monitoring = AsyncMock()
        integration.stop_monitoring = AsyncMock()
        integration.get_monitoring_status = AsyncMock(
            return_value={"monitoring": "active"}
        )
        integration.alert_manager = Mock()
        integration.alert_manager.trigger_alert = AsyncMock()

        # Make track_prediction_operation work as async context manager
        @asyncio.coroutine
        def async_context_manager(*args, **kwargs):
            class MockAsyncContext:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            return MockAsyncContext()

        integration.track_prediction_operation.side_effect = async_context_manager
        integration.track_training_operation = Mock(side_effect=async_context_manager)

        return integration

    @pytest.fixture
    def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
        """Create MonitoringEnhancedTrackingManager for testing."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)
            return manager

    def test_initialization(self, mock_tracking_manager, mock_monitoring_integration):
        """Test MonitoringEnhancedTrackingManager initialization."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)

            assert manager.tracking_manager == mock_tracking_manager
            assert manager.monitoring_integration == mock_monitoring_integration
            assert isinstance(manager._original_methods, dict)
            assert len(manager._original_methods) == 4  # 4 wrapped methods

    def test_method_wrapping(self, enhanced_manager, mock_tracking_manager):
        """Test that original methods are properly wrapped."""
        # Verify original methods are stored
        assert "record_prediction" in enhanced_manager._original_methods
        assert "validate_prediction" in enhanced_manager._original_methods
        assert "start_tracking" in enhanced_manager._original_methods
        assert "stop_tracking" in enhanced_manager._original_methods

        # Verify methods are replaced with monitored versions
        assert (
            mock_tracking_manager.record_prediction
            != enhanced_manager._original_methods["record_prediction"]
        )
        assert (
            mock_tracking_manager.validate_prediction
            != enhanced_manager._original_methods["validate_prediction"]
        )
        assert (
            mock_tracking_manager.start_tracking
            != enhanced_manager._original_methods["start_tracking"]
        )
        assert (
            mock_tracking_manager.stop_tracking
            != enhanced_manager._original_methods["stop_tracking"]
        )

    def test_monitored_record_prediction(self, enhanced_manager):
        """Test monitored record_prediction method."""
        # Create mock prediction result
        mock_prediction = Mock(spec=PredictionResult)
        mock_prediction.prediction_type = "next_occupied"
        mock_prediction.confidence = 0.85

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_record():
                # Setup original method mock to return a value
                enhanced_manager._original_methods["record_prediction"].return_value = (
                    "prediction_id_123"
                )

                result = await enhanced_manager._monitored_record_prediction(
                    room_id="living_room",
                    prediction_result=mock_prediction,
                    model_type=ModelType.ENSEMBLE,
                )

                # Verify original method was called
                enhanced_manager._original_methods[
                    "record_prediction"
                ].assert_called_once_with(
                    "living_room", mock_prediction, ModelType.ENSEMBLE
                )

                # Verify monitoring integration was called
                enhanced_manager.monitoring_integration.track_prediction_operation.assert_called_once()
                enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_called_once()

                assert result == "prediction_id_123"

            loop.run_until_complete(test_record())
        finally:
            loop.close()

    def test_monitored_record_prediction_no_confidence(self, enhanced_manager):
        """Test monitored record_prediction with no confidence value."""
        mock_prediction = Mock(spec=PredictionResult)
        mock_prediction.prediction_type = "next_vacant"
        mock_prediction.confidence = None

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_no_confidence():
                enhanced_manager._original_methods["record_prediction"].return_value = (
                    "prediction_id_456"
                )

                result = await enhanced_manager._monitored_record_prediction(
                    room_id="bedroom", prediction_result=mock_prediction
                )

                # Should still work but not record accuracy
                enhanced_manager._original_methods[
                    "record_prediction"
                ].assert_called_once()
                enhanced_manager.monitoring_integration.track_prediction_operation.assert_called_once()
                # Should not call record_prediction_accuracy when confidence is None
                enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_not_called()

                assert result == "prediction_id_456"

            loop.run_until_complete(test_no_confidence())
        finally:
            loop.close()

    def test_monitored_validate_prediction_success(self, enhanced_manager):
        """Test successful monitored validate_prediction."""
        validation_result = {
            "accuracy_minutes": 12.5,
            "prediction_type": "next_occupied",
            "model_type": ModelType.LSTM,
            "confidence": 0.78,
        }

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_validate():
                enhanced_manager._original_methods[
                    "validate_prediction"
                ].return_value = validation_result

                result = await enhanced_manager._monitored_validate_prediction(
                    room_id="kitchen", actual_time=datetime.now(timezone.utc)
                )

                # Verify original method was called
                enhanced_manager._original_methods[
                    "validate_prediction"
                ].assert_called_once()

                # Verify monitoring integration recorded the accuracy
                enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_called_once_with(
                    room_id="kitchen",
                    model_type="ModelType.LSTM",
                    prediction_type="next_occupied",
                    accuracy_minutes=12.5,
                    confidence=0.78,
                )

                assert result == validation_result

            loop.run_until_complete(test_validate())
        finally:
            loop.close()

    def test_monitored_validate_prediction_error(self, enhanced_manager):
        """Test monitored validate_prediction with error."""
        test_error = Exception("Validation failed")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_validate_error():
                enhanced_manager._original_methods[
                    "validate_prediction"
                ].side_effect = test_error

                with pytest.raises(Exception, match="Validation failed"):
                    await enhanced_manager._monitored_validate_prediction(
                        room_id="study", actual_time=datetime.now(timezone.utc)
                    )

                # Verify alert was triggered
                enhanced_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
                alert_call = (
                    enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_args
                )
                assert alert_call[1]["rule_name"] == "prediction_validation_error"
                assert "study" in alert_call[1]["title"]

            loop.run_until_complete(test_validate_error())
        finally:
            loop.close()

    def test_monitored_start_tracking_success(self, enhanced_manager):
        """Test successful monitored start_tracking."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_start():
                enhanced_manager._original_methods["start_tracking"].return_value = (
                    "started"
                )

                result = await enhanced_manager._monitored_start_tracking()

                # Verify monitoring system was started first
                enhanced_manager.monitoring_integration.start_monitoring.assert_called_once()

                # Verify original method was called
                enhanced_manager._original_methods[
                    "start_tracking"
                ].assert_called_once()

                # Verify success alert was triggered
                enhanced_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
                alert_call = (
                    enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_args
                )
                assert alert_call[1]["rule_name"] == "system_startup_success"

                assert result == "started"

            loop.run_until_complete(test_start())
        finally:
            loop.close()

    def test_monitored_start_tracking_error(self, enhanced_manager):
        """Test monitored start_tracking with error."""
        test_error = Exception("Start failed")

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_start_error():
                enhanced_manager._original_methods["start_tracking"].side_effect = (
                    test_error
                )

                with pytest.raises(Exception, match="Start failed"):
                    await enhanced_manager._monitored_start_tracking()

                # Verify error alert was triggered
                enhanced_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
                alert_call = (
                    enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_args
                )
                assert alert_call[1]["rule_name"] == "system_startup_error"

            loop.run_until_complete(test_start_error())
        finally:
            loop.close()

    def test_monitored_stop_tracking_success(self, enhanced_manager):
        """Test successful monitored stop_tracking."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_stop():
                enhanced_manager._original_methods["stop_tracking"].return_value = (
                    "stopped"
                )

                result = await enhanced_manager._monitored_stop_tracking()

                # Verify original method was called first
                enhanced_manager._original_methods["stop_tracking"].assert_called_once()

                # Verify monitoring system was stopped
                enhanced_manager.monitoring_integration.stop_monitoring.assert_called_once()

                assert result == "stopped"

            loop.run_until_complete(test_stop())
        finally:
            loop.close()

    def test_monitored_stop_tracking_with_errors(self, enhanced_manager):
        """Test monitored stop_tracking with errors in both tracking and monitoring."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_stop_errors():
                # Both stop methods fail
                enhanced_manager._original_methods["stop_tracking"].side_effect = (
                    Exception("Stop tracking failed")
                )
                enhanced_manager.monitoring_integration.stop_monitoring.side_effect = (
                    Exception("Stop monitoring failed")
                )

                with pytest.raises(Exception, match="Stop tracking failed"):
                    await enhanced_manager._monitored_stop_tracking()

                # Both methods should have been called despite errors
                enhanced_manager._original_methods["stop_tracking"].assert_called_once()
                enhanced_manager.monitoring_integration.stop_monitoring.assert_called_once()

            loop.run_until_complete(test_stop_errors())
        finally:
            loop.close()

    def test_record_concept_drift(self, enhanced_manager, mock_tracking_manager):
        """Test record_concept_drift method."""
        # Add the method to tracking manager for this test
        mock_tracking_manager.record_concept_drift = Mock()

        enhanced_manager.record_concept_drift(
            room_id="garage",
            drift_type="accuracy_degradation",
            severity=0.7,
            action_taken="retrain_scheduled",
        )

        # Verify monitoring integration was called
        enhanced_manager.monitoring_integration.record_concept_drift.assert_called_once_with(
            room_id="garage",
            drift_type="accuracy_degradation",
            severity=0.7,
            action_taken="retrain_scheduled",
        )

        # Verify tracking manager method was called
        mock_tracking_manager.record_concept_drift.assert_called_once_with(
            "garage", "accuracy_degradation", 0.7, "retrain_scheduled"
        )

    def test_record_concept_drift_no_tracking_method(self, enhanced_manager):
        """Test record_concept_drift when tracking manager doesn't have the method."""
        enhanced_manager.record_concept_drift(
            room_id="basement",
            drift_type="feature_drift",
            severity=0.5,
            action_taken="monitoring",
        )

        # Should only call monitoring integration
        enhanced_manager.monitoring_integration.record_concept_drift.assert_called_once_with(
            room_id="basement",
            drift_type="feature_drift",
            severity=0.5,
            action_taken="monitoring",
        )

    def test_record_feature_computation(self, enhanced_manager):
        """Test record_feature_computation method."""
        enhanced_manager.record_feature_computation(
            room_id="office", feature_type="temporal", duration=0.15
        )

        enhanced_manager.monitoring_integration.record_feature_computation.assert_called_once_with(
            room_id="office", feature_type="temporal", duration=0.15
        )

    def test_record_database_operation(self, enhanced_manager):
        """Test record_database_operation method."""
        enhanced_manager.record_database_operation(
            operation_type="insert",
            table="sensor_events",
            duration=0.05,
            status="success",
        )

        enhanced_manager.monitoring_integration.record_database_operation.assert_called_once_with(
            operation_type="insert",
            table="sensor_events",
            duration=0.05,
            status="success",
        )

    def test_record_database_operation_default_status(self, enhanced_manager):
        """Test record_database_operation with default status."""
        enhanced_manager.record_database_operation(
            operation_type="query", table="room_states", duration=0.03
        )

        enhanced_manager.monitoring_integration.record_database_operation.assert_called_once_with(
            operation_type="query", table="room_states", duration=0.03, status="success"
        )

    def test_record_mqtt_publish(self, enhanced_manager):
        """Test record_mqtt_publish method."""
        enhanced_manager.record_mqtt_publish(
            topic_type="prediction", room_id="den", status="success"
        )

        enhanced_manager.monitoring_integration.record_mqtt_publish.assert_called_once_with(
            topic_type="prediction", room_id="den", status="success"
        )

    def test_record_mqtt_publish_default_status(self, enhanced_manager):
        """Test record_mqtt_publish with default status."""
        enhanced_manager.record_mqtt_publish(topic_type="status", room_id="patio")

        enhanced_manager.monitoring_integration.record_mqtt_publish.assert_called_once_with(
            topic_type="status", room_id="patio", status="success"
        )

    def test_update_connection_status(self, enhanced_manager):
        """Test update_connection_status method."""
        enhanced_manager.update_connection_status(
            connection_type="mqtt", connected=True
        )

        enhanced_manager.monitoring_integration.update_connection_status.assert_called_once_with(
            connection_type="mqtt", connected=True
        )

    def test_get_monitoring_status(self, enhanced_manager, mock_tracking_manager):
        """Test get_monitoring_status method."""
        monitoring_status = {"system": "active", "alerts": 0}
        tracking_status = {"predictions": 150, "validations": 145}

        enhanced_manager.monitoring_integration.get_monitoring_status.return_value = (
            monitoring_status
        )
        mock_tracking_manager.get_system_status = AsyncMock(
            return_value=tracking_status
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_status():
                result = await enhanced_manager.get_monitoring_status()

                assert result["monitoring"] == monitoring_status
                assert result["tracking"] == tracking_status
                assert result["integrated"] is True
                assert "timestamp" in result

                enhanced_manager.monitoring_integration.get_monitoring_status.assert_called_once()
                mock_tracking_manager.get_system_status.assert_called_once()

            loop.run_until_complete(test_status())
        finally:
            loop.close()

    def test_get_monitoring_status_tracking_error(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test get_monitoring_status when tracking manager fails."""
        monitoring_status = {"system": "active"}
        enhanced_manager.monitoring_integration.get_monitoring_status.return_value = (
            monitoring_status
        )
        mock_tracking_manager.get_system_status = AsyncMock(
            side_effect=Exception("Status failed")
        )

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_status_error():
                result = await enhanced_manager.get_monitoring_status()

                assert result["monitoring"] == monitoring_status
                assert result["tracking"]["error"] == "Failed to get tracking status"
                assert result["integrated"] is True

            loop.run_until_complete(test_status_error())
        finally:
            loop.close()

    def test_get_monitoring_status_no_tracking_method(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test get_monitoring_status when tracking manager has no get_system_status method."""
        monitoring_status = {"system": "active"}
        enhanced_manager.monitoring_integration.get_monitoring_status.return_value = (
            monitoring_status
        )
        # Remove the method
        del mock_tracking_manager.get_system_status

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_no_method():
                result = await enhanced_manager.get_monitoring_status()

                assert result["monitoring"] == monitoring_status
                assert result["tracking"] == {}
                assert result["integrated"] is True

            loop.run_until_complete(test_no_method())
        finally:
            loop.close()

    def test_track_model_training_context_manager(self, enhanced_manager):
        """Test track_model_training context manager."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_training_context():
                async with enhanced_manager.track_model_training(
                    room_id="workshop", model_type="lstm", training_type="retraining"
                ):
                    # Context manager should work
                    pass

                # Verify monitoring integration context manager was called
                enhanced_manager.monitoring_integration.track_training_operation.assert_called_once_with(
                    room_id="workshop", model_type="lstm", training_type="retraining"
                )

            loop.run_until_complete(test_training_context())
        finally:
            loop.close()

    def test_track_model_training_default_type(self, enhanced_manager):
        """Test track_model_training with default training type."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:

            async def test_default_type():
                async with enhanced_manager.track_model_training(
                    room_id="attic", model_type="xgboost"
                ):
                    pass

                enhanced_manager.monitoring_integration.track_training_operation.assert_called_once_with(
                    room_id="attic", model_type="xgboost", training_type="retraining"
                )

            loop.run_until_complete(test_default_type())
        finally:
            loop.close()

    def test_getattr_delegation(self, enhanced_manager, mock_tracking_manager):
        """Test that unknown attributes are delegated to tracking manager."""
        # Add a custom attribute to tracking manager
        mock_tracking_manager.custom_method = Mock(return_value="custom_result")
        mock_tracking_manager.custom_property = "custom_value"

        # Test method delegation
        result = enhanced_manager.custom_method()
        assert result == "custom_result"
        mock_tracking_manager.custom_method.assert_called_once()

        # Test property delegation
        value = enhanced_manager.custom_property
        assert value == "custom_value"

    def test_getattr_delegation_nonexistent(self, enhanced_manager):
        """Test delegation of nonexistent attributes raises AttributeError."""
        with pytest.raises(AttributeError):
            _ = enhanced_manager.nonexistent_attribute


class TestFactoryFunctions:
    """Test suite for factory functions."""

    @pytest.fixture
    def mock_tracking_config(self):
        """Create a mock TrackingConfig."""
        return TrackingConfig(enabled=True, monitoring_interval_seconds=30)

    def test_create_monitoring_enhanced_tracking_manager(self, mock_tracking_config):
        """Test create_monitoring_enhanced_tracking_manager function."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.TrackingManager"
        ) as mock_tm_class:
            with patch(
                "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration"
            ) as mock_get_mi:
                mock_tracking_manager = Mock(spec=TrackingManager)
                mock_tm_class.return_value = mock_tracking_manager
                mock_monitoring_integration = Mock()
                mock_get_mi.return_value = mock_monitoring_integration

                result = create_monitoring_enhanced_tracking_manager(
                    config=mock_tracking_config, custom_param="test_value"
                )

                # Verify TrackingManager was created with correct parameters
                mock_tm_class.assert_called_once_with(
                    config=mock_tracking_config, custom_param="test_value"
                )

                # Verify result is MonitoringEnhancedTrackingManager
                assert isinstance(result, MonitoringEnhancedTrackingManager)
                assert result.tracking_manager == mock_tracking_manager
                assert result.monitoring_integration == mock_monitoring_integration

    def test_get_enhanced_tracking_manager(self, mock_tracking_config):
        """Test get_enhanced_tracking_manager convenience function."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.create_monitoring_enhanced_tracking_manager"
        ) as mock_create:
            mock_enhanced_manager = Mock(spec=MonitoringEnhancedTrackingManager)
            mock_create.return_value = mock_enhanced_manager

            result = get_enhanced_tracking_manager(
                config=mock_tracking_config, extra_param="extra_value"
            )

            # Verify create function was called with correct parameters
            mock_create.assert_called_once_with(
                mock_tracking_config, extra_param="extra_value"
            )

            assert result == mock_enhanced_manager

    def test_create_with_additional_kwargs(self, mock_tracking_config):
        """Test factory function with additional keyword arguments."""
        additional_kwargs = {
            "database_manager": Mock(),
            "model_registry": {"model1": Mock()},
            "notification_callbacks": [Mock()],
        }

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.TrackingManager"
        ) as mock_tm_class:
            with patch(
                "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration"
            ) as mock_get_mi:
                mock_tracking_manager = Mock(spec=TrackingManager)
                mock_tm_class.return_value = mock_tracking_manager
                mock_monitoring_integration = Mock()
                mock_get_mi.return_value = mock_monitoring_integration

                result = create_monitoring_enhanced_tracking_manager(
                    config=mock_tracking_config, **additional_kwargs
                )

                # Verify all kwargs were passed to TrackingManager
                mock_tm_class.assert_called_once_with(
                    config=mock_tracking_config, **additional_kwargs
                )

                assert isinstance(result, MonitoringEnhancedTrackingManager)


class TestIntegrationScenarios:
    """Integration test scenarios for MonitoringEnhancedTrackingManager."""

    @pytest.fixture
    def full_mock_setup(self):
        """Create a complete mock setup for integration testing."""
        # Mock tracking manager
        tracking_manager = Mock(spec=TrackingManager)
        tracking_manager.record_prediction = AsyncMock(return_value="pred_123")
        tracking_manager.validate_prediction = AsyncMock(
            return_value={
                "accuracy_minutes": 8.5,
                "prediction_type": "next_vacant",
                "model_type": "ensemble",
                "confidence": 0.82,
            }
        )
        tracking_manager.start_tracking = AsyncMock(return_value="started")
        tracking_manager.stop_tracking = AsyncMock(return_value="stopped")
        tracking_manager.get_system_status = AsyncMock(return_value={"active": True})

        # Mock monitoring integration
        monitoring_integration = Mock()
        monitoring_integration.start_monitoring = AsyncMock()
        monitoring_integration.stop_monitoring = AsyncMock()
        monitoring_integration.get_monitoring_status = AsyncMock(
            return_value={"status": "ok"}
        )
        monitoring_integration.record_prediction_accuracy = Mock()
        monitoring_integration.record_concept_drift = Mock()
        monitoring_integration.alert_manager = Mock()
        monitoring_integration.alert_manager.trigger_alert = AsyncMock()

        # Setup async context managers
        @asyncio.coroutine
        def async_context_manager(*args, **kwargs):
            class MockAsyncContext:
                async def __aenter__(self):
                    return self

                async def __aexit__(self, exc_type, exc_val, exc_tb):
                    return None

            return MockAsyncContext()

        monitoring_integration.track_prediction_operation = Mock(
            side_effect=async_context_manager
        )
        monitoring_integration.track_training_operation = Mock(
            side_effect=async_context_manager
        )

        return tracking_manager, monitoring_integration

    def test_complete_prediction_lifecycle(self, full_mock_setup):
        """Test complete prediction recording and validation lifecycle."""
        tracking_manager, monitoring_integration = full_mock_setup

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=monitoring_integration,
        ):
            enhanced_manager = MonitoringEnhancedTrackingManager(tracking_manager)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def test_lifecycle():
                    # Create mock prediction
                    prediction = Mock(spec=PredictionResult)
                    prediction.prediction_type = "next_occupied"
                    prediction.confidence = 0.75

                    # Record prediction
                    pred_id = await enhanced_manager._monitored_record_prediction(
                        room_id="main_hall",
                        prediction_result=prediction,
                        model_type=ModelType.ENSEMBLE,
                    )

                    assert pred_id == "pred_123"

                    # Verify tracking and monitoring calls
                    tracking_manager.record_prediction.assert_called_once()
                    monitoring_integration.track_prediction_operation.assert_called_once()
                    monitoring_integration.record_prediction_accuracy.assert_called_once()

                    # Validate prediction
                    validation_result = (
                        await enhanced_manager._monitored_validate_prediction(
                            room_id="main_hall", actual_time=datetime.now(timezone.utc)
                        )
                    )

                    assert validation_result["accuracy_minutes"] == 8.5

                    # Verify validation calls
                    tracking_manager.validate_prediction.assert_called_once()
                    # Should have 2 calls to record_prediction_accuracy now
                    assert (
                        monitoring_integration.record_prediction_accuracy.call_count
                        == 2
                    )

                loop.run_until_complete(test_lifecycle())
            finally:
                loop.close()

    def test_system_startup_and_shutdown(self, full_mock_setup):
        """Test complete system startup and shutdown with monitoring."""
        tracking_manager, monitoring_integration = full_mock_setup

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=monitoring_integration,
        ):
            enhanced_manager = MonitoringEnhancedTrackingManager(tracking_manager)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def test_startup_shutdown():
                    # Start system
                    start_result = await enhanced_manager._monitored_start_tracking()
                    assert start_result == "started"

                    # Verify monitoring started first, then tracking
                    monitoring_integration.start_monitoring.assert_called_once()
                    tracking_manager.start_tracking.assert_called_once()

                    # Verify success alert
                    monitoring_integration.alert_manager.trigger_alert.assert_called_once()
                    alert_call = (
                        monitoring_integration.alert_manager.trigger_alert.call_args
                    )
                    assert alert_call[1]["rule_name"] == "system_startup_success"

                    # Stop system
                    stop_result = await enhanced_manager._monitored_stop_tracking()
                    assert stop_result == "stopped"

                    # Verify tracking stopped first, then monitoring
                    tracking_manager.stop_tracking.assert_called_once()
                    monitoring_integration.stop_monitoring.assert_called_once()

                loop.run_until_complete(test_startup_shutdown())
            finally:
                loop.close()

    def test_error_handling_and_alerting(self, full_mock_setup):
        """Test error handling and alert generation."""
        tracking_manager, monitoring_integration = full_mock_setup

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=monitoring_integration,
        ):
            enhanced_manager = MonitoringEnhancedTrackingManager(tracking_manager)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def test_error_handling():
                    # Setup validation to fail
                    validation_error = Exception("Database connection lost")
                    tracking_manager.validate_prediction.side_effect = validation_error

                    # Attempt validation
                    with pytest.raises(Exception, match="Database connection lost"):
                        await enhanced_manager._monitored_validate_prediction(
                            room_id="server_room",
                            actual_time=datetime.now(timezone.utc),
                        )

                    # Verify error alert was triggered
                    monitoring_integration.alert_manager.trigger_alert.assert_called_once()
                    alert_call = (
                        monitoring_integration.alert_manager.trigger_alert.call_args
                    )
                    assert alert_call[1]["rule_name"] == "prediction_validation_error"
                    assert "server_room" in alert_call[1]["title"]
                    assert "Database connection lost" in alert_call[1]["message"]

                    # Verify context includes error details
                    assert "error" in alert_call[1]["context"]
                    assert "duration_seconds" in alert_call[1]["context"]

                loop.run_until_complete(test_error_handling())
            finally:
                loop.close()

    def test_comprehensive_monitoring_integration(self, full_mock_setup):
        """Test comprehensive monitoring integration across all methods."""
        tracking_manager, monitoring_integration = full_mock_setup

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=monitoring_integration,
        ):
            enhanced_manager = MonitoringEnhancedTrackingManager(tracking_manager)

            # Test all monitoring methods
            enhanced_manager.record_concept_drift(
                "lab", "data_drift", 0.6, "investigate"
            )
            enhanced_manager.record_feature_computation("lab", "sequential", 0.25)
            enhanced_manager.record_database_operation(
                "select", "predictions", 0.08, "success"
            )
            enhanced_manager.record_mqtt_publish("alert", "lab", "failed")
            enhanced_manager.update_connection_status("database", False)

            # Verify all monitoring calls were made
            monitoring_integration.record_concept_drift.assert_called_once_with(
                room_id="lab",
                drift_type="data_drift",
                severity=0.6,
                action_taken="investigate",
            )
            monitoring_integration.record_feature_computation.assert_called_once_with(
                room_id="lab", feature_type="sequential", duration=0.25
            )
            monitoring_integration.record_database_operation.assert_called_once_with(
                operation_type="select",
                table="predictions",
                duration=0.08,
                status="success",
            )
            monitoring_integration.record_mqtt_publish.assert_called_once_with(
                topic_type="alert", room_id="lab", status="failed"
            )
            monitoring_integration.update_connection_status.assert_called_once_with(
                connection_type="database", connected=False
            )

    def test_model_training_context_integration(self, full_mock_setup):
        """Test model training context manager integration."""
        tracking_manager, monitoring_integration = full_mock_setup

        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=monitoring_integration,
        ):
            enhanced_manager = MonitoringEnhancedTrackingManager(tracking_manager)

            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:

                async def test_training_context():
                    training_completed = False

                    async with enhanced_manager.track_model_training(
                        room_id="training_room",
                        model_type="deep_learning",
                        training_type="initial_training",
                    ):
                        # Simulate training work
                        await asyncio.sleep(0.01)
                        training_completed = True

                    assert training_completed

                    # Verify monitoring context manager was used
                    monitoring_integration.track_training_operation.assert_called_once_with(
                        room_id="training_room",
                        model_type="deep_learning",
                        training_type="initial_training",
                    )

                loop.run_until_complete(test_training_context())
            finally:
                loop.close()
