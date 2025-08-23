"""
Comprehensive test suite for MonitoringEnhancedTrackingManager.
Tests all monitoring integration, tracking enhancement, and system workflow functionality.
"""

import asyncio
from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from src.adaptation.monitoring_enhanced_tracking import (
    MonitoringEnhancedTrackingManager,
    create_monitoring_enhanced_tracking_manager,
    get_enhanced_tracking_manager,
)
from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.constants import ModelType
from src.models.base.predictor import PredictionResult
from src.utils.monitoring_integration import MonitoringIntegration


class TestMonitoringEnhancedTrackingManager:
    """Test MonitoringEnhancedTrackingManager class."""

    @pytest.fixture
    def mock_tracking_manager(self):
        """Create mock tracking manager."""
        manager = Mock(spec=TrackingManager)
        manager.record_prediction = AsyncMock(return_value="prediction_id_123")
        manager.validate_prediction = AsyncMock(
            return_value={
                "accuracy_minutes": 5.2,
                "prediction_type": "next_occupied",
                "model_type": "ensemble",
                "confidence": 0.85,
            }
        )
        manager.start_tracking = AsyncMock(return_value=True)
        manager.stop_tracking = AsyncMock(return_value=True)
        manager.get_system_status = AsyncMock(
            return_value={
                "status": "healthy",
                "tracking_active": True,
                "rooms_tracked": 5,
            }
        )
        manager.get_accuracy_metrics = AsyncMock(
            return_value={
                "living_room": Mock(accuracy_percentage=0.92),
                "bedroom": Mock(accuracy_percentage=0.89),
            }
        )
        return manager

    @pytest.fixture
    def mock_monitoring_integration(self):
        """Create mock monitoring integration."""
        integration = Mock(spec=MonitoringIntegration)
        integration.track_prediction_operation = Mock(
            return_value=MockAsyncContextManager()
        )
        integration.record_prediction_accuracy = Mock()
        integration.start_monitoring = AsyncMock()
        integration.stop_monitoring = AsyncMock()
        integration.record_concept_drift = Mock()
        integration.record_feature_computation = Mock()
        integration.record_database_operation = Mock()
        integration.record_mqtt_publish = Mock()
        integration.update_connection_status = Mock()
        integration.get_monitoring_status = AsyncMock(
            return_value={
                "monitoring": {
                    "health_status": "healthy",
                    "monitoring_active": True,
                    "alert_system": {"active_alerts": 0},
                }
            }
        )
        integration.track_training_operation = Mock(
            return_value=MockAsyncContextManager()
        )
        integration.alert_manager = Mock()
        integration.alert_manager.trigger_alert = AsyncMock(return_value="alert_id_123")
        return integration

    @pytest.fixture
    def prediction_result(self):
        """Create sample prediction result."""
        return PredictionResult(
            prediction_type="next_occupied",
            predicted_time=datetime.now(timezone.utc),
            confidence=0.85,
            metadata={"model_used": "ensemble"},
        )

    @pytest.fixture
    def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
        """Create enhanced tracking manager with mocks."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)
            return manager

    def test_initialization(self, mock_tracking_manager):
        """Test enhanced manager initialization."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration"
        ) as mock_get:
            mock_monitoring = Mock()
            mock_get.return_value = mock_monitoring

            manager = MonitoringEnhancedTrackingManager(mock_tracking_manager)

            assert manager.tracking_manager is mock_tracking_manager
            assert manager.monitoring_integration is mock_monitoring
            assert len(manager._original_methods) == 4
            assert "record_prediction" in manager._original_methods
            assert "validate_prediction" in manager._original_methods
            assert "start_tracking" in manager._original_methods
            assert "stop_tracking" in manager._original_methods

    def test_method_wrapping(self, enhanced_manager, mock_tracking_manager):
        """Test that original methods are properly wrapped."""
        # Verify original methods are stored
        assert "record_prediction" in enhanced_manager._original_methods

        # Verify methods are replaced with monitored versions
        assert (
            mock_tracking_manager.record_prediction
            != enhanced_manager._original_methods["record_prediction"]
        )
        assert callable(mock_tracking_manager.record_prediction)

    @pytest.mark.asyncio
    async def test_monitored_record_prediction_success(
        self, enhanced_manager, prediction_result
    ):
        """Test successful prediction recording with monitoring."""
        room_id = "living_room"
        model_type = ModelType.ENSEMBLE

        result = await enhanced_manager._monitored_record_prediction(
            room_id, prediction_result, model_type
        )

        assert result == "prediction_id_123"

        # Verify monitoring integration calls
        enhanced_manager.monitoring_integration.track_prediction_operation.assert_called_once_with(
            room_id=room_id, prediction_type="next_occupied", model_type="ensemble"
        )
        enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_called_once_with(
            room_id=room_id,
            model_type="ensemble",
            prediction_type="next_occupied",
            accuracy_minutes=0,
            confidence=0.85,
        )

    @pytest.mark.asyncio
    async def test_monitored_record_prediction_without_confidence(
        self, enhanced_manager
    ):
        """Test prediction recording without confidence value."""
        prediction_result = PredictionResult(
            prediction_type="next_vacant",
            predicted_time=datetime.now(timezone.utc),
            confidence=None,
            metadata={},
        )

        await enhanced_manager._monitored_record_prediction(
            "bedroom", prediction_result, ModelType.LSTM
        )

        # Should not call record_prediction_accuracy without confidence
        enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_not_called()

    @pytest.mark.asyncio
    async def test_monitored_validate_prediction_success(self, enhanced_manager):
        """Test successful prediction validation with monitoring."""
        room_id = "kitchen"
        actual_time = datetime.now(timezone.utc)

        result = await enhanced_manager._monitored_validate_prediction(
            room_id, actual_time
        )

        expected_result = {
            "accuracy_minutes": 5.2,
            "prediction_type": "next_occupied",
            "model_type": "ensemble",
            "confidence": 0.85,
        }
        assert result == expected_result

        # Verify accuracy recording
        enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_called_once_with(
            room_id=room_id,
            model_type="ensemble",
            prediction_type="next_occupied",
            accuracy_minutes=5.2,
            confidence=0.85,
        )

    @pytest.mark.asyncio
    async def test_monitored_validate_prediction_error(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test prediction validation error handling."""
        room_id = "bathroom"
        actual_time = datetime.now(timezone.utc)

        # Make original method raise exception
        error = Exception("Validation failed")
        enhanced_manager._original_methods["validate_prediction"].side_effect = error

        with pytest.raises(Exception, match="Validation failed"):
            await enhanced_manager._monitored_validate_prediction(room_id, actual_time)

        # Verify alert was triggered
        enhanced_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
        call_args = (
            enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_args
        )
        assert call_args[1]["rule_name"] == "prediction_validation_error"
        assert room_id in call_args[1]["title"]

    @pytest.mark.asyncio
    async def test_monitored_start_tracking_success(self, enhanced_manager):
        """Test successful tracking start with monitoring."""
        result = await enhanced_manager._monitored_start_tracking(debug=True)

        assert result is True

        # Verify monitoring started first
        enhanced_manager.monitoring_integration.start_monitoring.assert_called_once()

        # Verify success alert triggered
        enhanced_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
        call_args = (
            enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_args
        )
        assert call_args[1]["rule_name"] == "system_startup_success"
        assert call_args[1]["context"]["monitoring_enabled"] is True

    @pytest.mark.asyncio
    async def test_monitored_start_tracking_error(self, enhanced_manager):
        """Test tracking start error handling."""
        error = Exception("Start failed")
        enhanced_manager._original_methods["start_tracking"].side_effect = error

        with pytest.raises(Exception, match="Start failed"):
            await enhanced_manager._monitored_start_tracking()

        # Verify startup failure alert
        enhanced_manager.monitoring_integration.alert_manager.trigger_alert.assert_called_once()
        call_args = (
            enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_args
        )
        assert call_args[1]["rule_name"] == "system_startup_error"

    @pytest.mark.asyncio
    async def test_monitored_stop_tracking_success(self, enhanced_manager):
        """Test successful tracking stop with monitoring cleanup."""
        result = await enhanced_manager._monitored_stop_tracking()

        assert result is True

        # Verify original stop called first
        enhanced_manager._original_methods["stop_tracking"].assert_called_once()

        # Verify monitoring stopped
        enhanced_manager.monitoring_integration.stop_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitored_stop_tracking_with_error(self, enhanced_manager):
        """Test tracking stop with error but monitoring cleanup."""
        error = Exception("Stop failed")
        enhanced_manager._original_methods["stop_tracking"].side_effect = error

        with pytest.raises(Exception, match="Stop failed"):
            await enhanced_manager._monitored_stop_tracking()

        # Verify monitoring still attempted to stop
        enhanced_manager.monitoring_integration.stop_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_monitored_stop_tracking_monitoring_error(self, enhanced_manager):
        """Test tracking stop with monitoring cleanup error."""
        stop_error = Exception("Stop failed")
        monitoring_error = Exception("Monitoring stop failed")
        enhanced_manager._original_methods["stop_tracking"].side_effect = stop_error
        enhanced_manager.monitoring_integration.stop_monitoring.side_effect = (
            monitoring_error
        )

        with pytest.raises(Exception, match="Stop failed"):
            await enhanced_manager._monitored_stop_tracking()

    def test_record_concept_drift(self, enhanced_manager, mock_tracking_manager):
        """Test concept drift recording."""
        room_id = "living_room"
        drift_type = "seasonal_change"
        severity = 0.7
        action_taken = "retrain_model"

        enhanced_manager.record_concept_drift(
            room_id, drift_type, severity, action_taken
        )

        # Verify monitoring integration called
        enhanced_manager.monitoring_integration.record_concept_drift.assert_called_once_with(
            room_id=room_id,
            drift_type=drift_type,
            severity=severity,
            action_taken=action_taken,
        )

    def test_record_concept_drift_with_tracking_manager_support(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test concept drift recording when tracking manager supports it."""
        mock_tracking_manager.record_concept_drift = Mock()

        enhanced_manager.record_concept_drift(
            "bedroom", "pattern_shift", 0.5, "adjust_weights"
        )

        # Verify both monitoring and tracking manager called
        enhanced_manager.monitoring_integration.record_concept_drift.assert_called_once()
        mock_tracking_manager.record_concept_drift.assert_called_once_with(
            "bedroom", "pattern_shift", 0.5, "adjust_weights"
        )

    def test_record_feature_computation(self, enhanced_manager):
        """Test feature computation recording."""
        enhanced_manager.record_feature_computation("kitchen", "temporal", 0.025)

        enhanced_manager.monitoring_integration.record_feature_computation.assert_called_once_with(
            room_id="kitchen", feature_type="temporal", duration=0.025
        )

    def test_record_database_operation(self, enhanced_manager):
        """Test database operation recording."""
        enhanced_manager.record_database_operation(
            "INSERT", "sensor_events", 0.15, "success"
        )

        enhanced_manager.monitoring_integration.record_database_operation.assert_called_once_with(
            operation_type="INSERT",
            table="sensor_events",
            duration=0.15,
            status="success",
        )

    def test_record_database_operation_default_status(self, enhanced_manager):
        """Test database operation recording with default status."""
        enhanced_manager.record_database_operation("SELECT", "room_states", 0.08)

        enhanced_manager.monitoring_integration.record_database_operation.assert_called_once_with(
            operation_type="SELECT",
            table="room_states",
            duration=0.08,
            status="success",
        )

    def test_record_mqtt_publish(self, enhanced_manager):
        """Test MQTT publish recording."""
        enhanced_manager.record_mqtt_publish("prediction", "living_room", "success")

        enhanced_manager.monitoring_integration.record_mqtt_publish.assert_called_once_with(
            topic_type="prediction", room_id="living_room", status="success"
        )

    def test_record_mqtt_publish_default_status(self, enhanced_manager):
        """Test MQTT publish recording with default status."""
        enhanced_manager.record_mqtt_publish("status", "bedroom")

        enhanced_manager.monitoring_integration.record_mqtt_publish.assert_called_once_with(
            topic_type="status", room_id="bedroom", status="success"
        )

    def test_update_connection_status(self, enhanced_manager):
        """Test connection status updates."""
        enhanced_manager.update_connection_status("database", True)

        enhanced_manager.monitoring_integration.update_connection_status.assert_called_once_with(
            connection_type="database", connected=True
        )

    @pytest.mark.asyncio
    async def test_get_monitoring_status(self, enhanced_manager, mock_tracking_manager):
        """Test comprehensive monitoring status retrieval."""
        mock_monitoring_status = {
            "monitoring": {"health_status": "healthy", "monitoring_active": True}
        }
        enhanced_manager.monitoring_integration.get_monitoring_status.return_value = (
            mock_monitoring_status
        )

        status = await enhanced_manager.get_monitoring_status()

        assert "monitoring" in status
        assert "tracking" in status
        assert status["integrated"] is True
        assert "timestamp" in status
        assert status["tracking"]["status"] == "healthy"

    @pytest.mark.asyncio
    async def test_get_monitoring_status_tracking_error(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test monitoring status with tracking manager error."""
        mock_tracking_manager.get_system_status.side_effect = Exception(
            "Tracking error"
        )

        status = await enhanced_manager.get_monitoring_status()

        assert "monitoring" in status
        assert "tracking" in status
        assert status["tracking"]["error"] == "Failed to get tracking status"
        assert status["integrated"] is True

    @pytest.mark.asyncio
    async def test_get_monitoring_status_no_tracking_support(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test monitoring status when tracking manager doesn't support status."""
        # Remove get_system_status method
        del mock_tracking_manager.get_system_status

        status = await enhanced_manager.get_monitoring_status()

        assert status["tracking"] == {}

    @pytest.mark.asyncio
    async def test_track_model_training_context_manager(self, enhanced_manager):
        """Test model training tracking context manager."""
        room_id = "office"
        model_type = "lstm"
        training_type = "incremental"

        async with enhanced_manager.track_model_training(
            room_id, model_type, training_type
        ):
            # Context manager should be active
            pass

        # Verify monitoring integration context manager called
        enhanced_manager.monitoring_integration.track_training_operation.assert_called_once_with(
            room_id=room_id, model_type=model_type, training_type=training_type
        )

    def test_getattr_delegation(self, enhanced_manager, mock_tracking_manager):
        """Test attribute delegation to original tracking manager."""
        mock_tracking_manager.some_custom_method = Mock(return_value="custom_result")

        result = enhanced_manager.some_custom_method()

        assert result == "custom_result"
        mock_tracking_manager.some_custom_method.assert_called_once()

    def test_getattr_missing_attribute(self, enhanced_manager, mock_tracking_manager):
        """Test attribute delegation for missing attributes."""
        with pytest.raises(AttributeError):
            enhanced_manager.nonexistent_method()


class TestFactoryFunctions:
    """Test factory functions for creating enhanced tracking managers."""

    @pytest.fixture
    def mock_tracking_config(self):
        """Create mock tracking configuration."""
        config = Mock(spec=TrackingConfig)
        config.tracking_interval = 60
        config.accuracy_threshold = 15
        return config

    @patch("src.adaptation.monitoring_enhanced_tracking.TrackingManager")
    @patch("src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration")
    def test_create_monitoring_enhanced_tracking_manager(
        self, mock_get_monitoring, mock_tracking_class, mock_tracking_config
    ):
        """Test factory function for creating enhanced tracking manager."""
        mock_tracking_instance = Mock()
        mock_tracking_class.return_value = mock_tracking_instance
        mock_monitoring = Mock()
        mock_get_monitoring.return_value = mock_monitoring

        result = create_monitoring_enhanced_tracking_manager(
            mock_tracking_config, extra_param=True
        )

        # Verify TrackingManager created with config and kwargs
        mock_tracking_class.assert_called_once_with(
            config=mock_tracking_config, extra_param=True
        )

        # Verify enhanced manager created
        assert isinstance(result, MonitoringEnhancedTrackingManager)
        assert result.tracking_manager is mock_tracking_instance

    @patch(
        "src.adaptation.monitoring_enhanced_tracking.create_monitoring_enhanced_tracking_manager"
    )
    def test_get_enhanced_tracking_manager(self, mock_create, mock_tracking_config):
        """Test convenience function for getting enhanced tracking manager."""
        mock_enhanced_manager = Mock()
        mock_create.return_value = mock_enhanced_manager

        result = get_enhanced_tracking_manager(mock_tracking_config, debug=True)

        mock_create.assert_called_once_with(mock_tracking_config, debug=True)
        assert result is mock_enhanced_manager


class TestIntegrationScenarios:
    """Test complex integration scenarios."""

    @pytest.fixture
    def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
        """Create enhanced manager for integration tests."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            return MonitoringEnhancedTrackingManager(mock_tracking_manager)

    @pytest.mark.asyncio
    async def test_full_prediction_workflow(self, enhanced_manager, prediction_result):
        """Test complete prediction workflow with monitoring."""
        room_id = "living_room"

        # Record prediction
        prediction_id = await enhanced_manager._monitored_record_prediction(
            room_id, prediction_result, ModelType.ENSEMBLE
        )

        # Validate prediction later
        actual_time = datetime.now(timezone.utc)
        validation_result = await enhanced_manager._monitored_validate_prediction(
            room_id, actual_time
        )

        # Verify workflow
        assert prediction_id == "prediction_id_123"
        assert validation_result["accuracy_minutes"] == 5.2

        # Verify monitoring calls
        assert (
            enhanced_manager.monitoring_integration.record_prediction_accuracy.call_count
            == 2
        )

    @pytest.mark.asyncio
    async def test_system_lifecycle_with_monitoring(self, enhanced_manager):
        """Test system start/stop lifecycle with monitoring."""
        # Start system
        await enhanced_manager._monitored_start_tracking()

        # Verify startup
        enhanced_manager.monitoring_integration.start_monitoring.assert_called_once()

        # Record some metrics during operation
        enhanced_manager.record_feature_computation("kitchen", "sequential", 0.032)
        enhanced_manager.record_database_operation("UPDATE", "predictions", 0.045)

        # Stop system
        await enhanced_manager._monitored_stop_tracking()

        # Verify shutdown
        enhanced_manager.monitoring_integration.stop_monitoring.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_scenarios_with_alerts(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test various error scenarios with alert generation."""
        # Validation error
        validation_error = Exception("Database connection lost")
        enhanced_manager._original_methods["validate_prediction"].side_effect = (
            validation_error
        )

        with pytest.raises(Exception, match="Database connection lost"):
            await enhanced_manager._monitored_validate_prediction(
                "bedroom", datetime.now(timezone.utc)
            )

        # Startup error
        startup_error = Exception("Configuration invalid")
        enhanced_manager._original_methods["start_tracking"].side_effect = startup_error

        with pytest.raises(Exception, match="Configuration invalid"):
            await enhanced_manager._monitored_start_tracking()

        # Verify alerts generated
        assert (
            enhanced_manager.monitoring_integration.alert_manager.trigger_alert.call_count
            == 2
        )

    def test_monitoring_integration_coverage(self, enhanced_manager):
        """Test coverage of all monitoring integration features."""
        # Test all monitoring methods
        enhanced_manager.record_concept_drift("room1", "drift1", 0.8, "action1")
        enhanced_manager.record_feature_computation("room2", "feature1", 0.1)
        enhanced_manager.record_database_operation("op1", "table1", 0.2)
        enhanced_manager.record_mqtt_publish("topic1", "room3")
        enhanced_manager.update_connection_status("conn1", False)

        # Verify all calls made
        enhanced_manager.monitoring_integration.record_concept_drift.assert_called_once()
        enhanced_manager.monitoring_integration.record_feature_computation.assert_called_once()
        enhanced_manager.monitoring_integration.record_database_operation.assert_called_once()
        enhanced_manager.monitoring_integration.record_mqtt_publish.assert_called_once()
        enhanced_manager.monitoring_integration.update_connection_status.assert_called_once()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
        """Create enhanced manager for edge case tests."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            return MonitoringEnhancedTrackingManager(mock_tracking_manager)

    @pytest.mark.asyncio
    async def test_prediction_with_string_model_type(
        self, enhanced_manager, prediction_result
    ):
        """Test prediction recording with string model type."""
        result = await enhanced_manager._monitored_record_prediction(
            "room", prediction_result, "custom_model"
        )

        assert result == "prediction_id_123"

        # Verify string model type handled correctly
        call_args = (
            enhanced_manager.monitoring_integration.track_prediction_operation.call_args
        )
        assert call_args[1]["model_type"] == "custom_model"

    @pytest.mark.asyncio
    async def test_validation_with_non_dict_result(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test validation with non-dictionary result."""
        enhanced_manager._original_methods["validate_prediction"].return_value = (
            "simple_result"
        )

        result = await enhanced_manager._monitored_validate_prediction(
            "room", datetime.now(timezone.utc)
        )

        assert result == "simple_result"
        # Should not attempt to record accuracy with non-dict result
        enhanced_manager.monitoring_integration.record_prediction_accuracy.assert_not_called()

    def test_concept_drift_without_tracking_manager_support(
        self, enhanced_manager, mock_tracking_manager
    ):
        """Test concept drift recording when tracking manager doesn't support it."""
        # Ensure tracking manager doesn't have record_concept_drift
        if hasattr(mock_tracking_manager, "record_concept_drift"):
            delattr(mock_tracking_manager, "record_concept_drift")

        enhanced_manager.record_concept_drift("room", "drift", 0.5, "action")

        # Should only call monitoring integration
        enhanced_manager.monitoring_integration.record_concept_drift.assert_called_once()

    @pytest.mark.asyncio
    async def test_context_manager_exception_handling(self, enhanced_manager):
        """Test model training context manager with exceptions."""
        enhanced_manager.monitoring_integration.track_training_operation.return_value = (
            MockAsyncContextManagerWithError()
        )

        with pytest.raises(RuntimeError, match="Context manager error"):
            async with enhanced_manager.track_model_training("room", "model", "type"):
                raise RuntimeError("Training error")


class MockAsyncContextManager:
    """Mock async context manager for testing."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False


class MockAsyncContextManagerWithError:
    """Mock async context manager that raises error on exit."""

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        raise RuntimeError("Context manager error")


class TestPerformanceAndStress:
    """Test performance and stress scenarios."""

    @pytest.fixture
    def enhanced_manager(self, mock_tracking_manager, mock_monitoring_integration):
        """Create enhanced manager for performance tests."""
        with patch(
            "src.adaptation.monitoring_enhanced_tracking.get_monitoring_integration",
            return_value=mock_monitoring_integration,
        ):
            return MonitoringEnhancedTrackingManager(mock_tracking_manager)

    @pytest.mark.asyncio
    async def test_high_volume_predictions(self, enhanced_manager):
        """Test handling high volume of predictions."""
        predictions = []
        for i in range(100):
            prediction = PredictionResult(
                prediction_type="next_occupied",
                predicted_time=datetime.now(timezone.utc),
                confidence=0.8 + (i % 20) / 100,
                metadata={"batch": i},
            )
            predictions.append(prediction)

        # Process all predictions
        tasks = []
        for i, prediction in enumerate(predictions):
            task = enhanced_manager._monitored_record_prediction(
                f"room_{i % 10}", prediction, ModelType.ENSEMBLE
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks)

        # Verify all processed
        assert len(results) == 100
        assert all(result == "prediction_id_123" for result in results)

    def test_rapid_metric_recording(self, enhanced_manager):
        """Test rapid metric recording doesn't cause issues."""
        for i in range(1000):
            enhanced_manager.record_feature_computation(
                f"room_{i % 5}", "temporal", 0.001 * i
            )
            enhanced_manager.record_database_operation("SELECT", "events", 0.002 * i)
            enhanced_manager.record_mqtt_publish("prediction", f"room_{i % 3}")

        # Should not raise any exceptions
        assert (
            enhanced_manager.monitoring_integration.record_feature_computation.call_count
            == 1000
        )
        assert (
            enhanced_manager.monitoring_integration.record_database_operation.call_count
            == 1000
        )
        assert (
            enhanced_manager.monitoring_integration.record_mqtt_publish.call_count
            == 1000
        )
