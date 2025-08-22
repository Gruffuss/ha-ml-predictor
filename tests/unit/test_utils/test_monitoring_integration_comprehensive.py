"""
Comprehensive unit tests for monitoring_integration.py.
Provides complete coverage for integration layer between monitoring and components.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, Mock, call, patch

import pytest

from src.utils.monitoring_integration import (
    MonitoringIntegration,
    get_monitoring_integration,
)


class TestMonitoringIntegrationInitialization:
    """Test MonitoringIntegration initialization and setup."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for MonitoringIntegration."""
        with patch(
            "src.utils.monitoring_integration.get_logger"
        ) as mock_get_logger, patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ) as mock_get_perf_logger, patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ) as mock_get_ml_ops_logger, patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ) as mock_get_metrics_manager, patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_get_metrics_collector, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_get_monitoring_manager, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            yield {
                "logger": mock_get_logger.return_value,
                "performance_logger": mock_get_perf_logger.return_value,
                "ml_ops_logger": mock_get_ml_ops_logger.return_value,
                "metrics_manager": mock_get_metrics_manager.return_value,
                "metrics_collector": mock_get_metrics_collector.return_value,
                "monitoring_manager": mock_get_monitoring_manager.return_value,
                "alert_manager": mock_get_alert_manager.return_value,
            }

    def test_monitoring_integration_initialization(self, mock_dependencies):
        """Test MonitoringIntegration initialization."""
        integration = MonitoringIntegration()

        # Verify all dependencies are set
        assert integration.logger == mock_dependencies["logger"]
        assert integration.performance_logger == mock_dependencies["performance_logger"]
        assert integration.ml_ops_logger == mock_dependencies["ml_ops_logger"]
        assert integration.metrics_manager == mock_dependencies["metrics_manager"]
        assert integration.metrics_collector == mock_dependencies["metrics_collector"]
        assert integration.monitoring_manager == mock_dependencies["monitoring_manager"]
        assert integration.alert_manager == mock_dependencies["alert_manager"]

        # Verify setup was called
        mock_dependencies["logger"].info.assert_called_with(
            "Monitoring integration initialized"
        )

    def test_setup_integrations(self, mock_dependencies):
        """Test setup integrations configures alert callbacks."""
        mock_performance_monitor = Mock()
        mock_dependencies["monitoring_manager"].get_performance_monitor.return_value = (
            mock_performance_monitor
        )

        integration = MonitoringIntegration()

        # Verify alert callback was added to performance monitor
        mock_performance_monitor.add_alert_callback.assert_called_once()
        callback_arg = mock_performance_monitor.add_alert_callback.call_args[0][0]

        # Verify the callback is the _handle_performance_alert method
        assert callback_arg == integration._handle_performance_alert


class TestMonitoringIntegrationStartStop:
    """Test starting and stopping monitoring integration."""

    @pytest.fixture
    def mock_integration(self):
        """Create MonitoringIntegration with mocked dependencies."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch("src.utils.monitoring_integration.get_ml_ops_logger"), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ) as mock_get_metrics_manager, patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ), patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_get_monitoring_manager, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            mock_metrics_manager = mock_get_metrics_manager.return_value
            mock_monitoring_manager = mock_get_monitoring_manager.return_value
            mock_alert_manager = mock_get_alert_manager.return_value

            integration = MonitoringIntegration()
            integration.metrics_manager = mock_metrics_manager
            integration.monitoring_manager = mock_monitoring_manager
            integration.alert_manager = mock_alert_manager

            yield integration

    @pytest.mark.asyncio
    async def test_start_monitoring_success(self, mock_integration):
        """Test successful monitoring startup."""
        mock_integration.monitoring_manager.start_monitoring = AsyncMock()

        await mock_integration.start_monitoring()

        # Verify components were started
        mock_integration.metrics_manager.start_background_collection.assert_called_once()
        mock_integration.monitoring_manager.start_monitoring.assert_called_once()

        # Verify success was logged
        mock_integration.logger.info.assert_called_with(
            "Monitoring system started successfully"
        )

    @pytest.mark.asyncio
    async def test_start_monitoring_failure(self, mock_integration):
        """Test monitoring startup failure."""
        # Mock failure in metrics manager
        mock_integration.metrics_manager.start_background_collection.side_effect = (
            Exception("Metrics startup failed")
        )
        mock_integration.alert_manager.trigger_alert = AsyncMock()

        with pytest.raises(Exception, match="Metrics startup failed"):
            await mock_integration.start_monitoring()

        # Verify error was logged
        mock_integration.logger.error.assert_called_once()
        error_call = mock_integration.logger.error.call_args[0][0]
        assert "Failed to start monitoring system" in error_call

        # Verify alert was triggered
        mock_integration.alert_manager.trigger_alert.assert_called_once_with(
            rule_name="system_startup_error",
            title="Monitoring System Startup Failed",
            message="Failed to start monitoring components: Metrics startup failed",
            component="monitoring_system",
        )

    @pytest.mark.asyncio
    async def test_stop_monitoring_success(self, mock_integration):
        """Test successful monitoring shutdown."""
        mock_integration.monitoring_manager.stop_monitoring = AsyncMock()

        await mock_integration.stop_monitoring()

        # Verify components were stopped
        mock_integration.monitoring_manager.stop_monitoring.assert_called_once()
        mock_integration.metrics_manager.stop_background_collection.assert_called_once()

        # Verify success was logged
        mock_integration.logger.info.assert_called_with("Monitoring system stopped")

    @pytest.mark.asyncio
    async def test_stop_monitoring_failure(self, mock_integration):
        """Test monitoring shutdown with errors."""
        # Mock failure in monitoring manager
        mock_integration.monitoring_manager.stop_monitoring = AsyncMock(
            side_effect=Exception("Stop failed")
        )

        # Should not raise exception
        await mock_integration.stop_monitoring()

        # Verify error was logged
        mock_integration.logger.error.assert_called_once()
        error_call = mock_integration.logger.error.call_args[0][0]
        assert "Error stopping monitoring system" in error_call


class TestMonitoringIntegrationAlertHandling:
    """Test alert handling functionality."""

    @pytest.fixture
    def mock_integration(self):
        """Create MonitoringIntegration with mocked alert manager."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch("src.utils.monitoring_integration.get_ml_ops_logger"), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ), patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ), patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            mock_alert_manager = mock_get_alert_manager.return_value
            mock_alert_manager.trigger_alert = AsyncMock()

            integration = MonitoringIntegration()
            integration.alert_manager = mock_alert_manager

            yield integration

    @pytest.mark.asyncio
    async def test_handle_performance_alert_success(self, mock_integration):
        """Test successful performance alert handling."""
        # Create mock alert with minimal required attributes
        mock_alert = Mock()
        mock_alert.alert_type = "warning"
        mock_alert.message = "High CPU usage detected"
        mock_alert.component = "system"
        mock_alert.additional_info = {"cpu_percent": 85.5}

        # Add room_id attribute to test optional handling
        mock_alert.room_id = "living_room"

        await mock_integration._handle_performance_alert(mock_alert)

        # Verify alert was triggered
        mock_integration.alert_manager.trigger_alert.assert_called_once_with(
            rule_name="performance_warning",
            title="Performance Alert: High CPU usage detected",
            message="High CPU usage detected",
            component="system",
            room_id="living_room",
            context={"cpu_percent": 85.5},
        )

    @pytest.mark.asyncio
    async def test_handle_performance_alert_without_room_id(self, mock_integration):
        """Test performance alert handling without room_id attribute."""
        mock_alert = Mock()
        mock_alert.alert_type = "critical"
        mock_alert.message = "Memory exhaustion imminent"
        mock_alert.component = "memory_manager"
        mock_alert.additional_info = {"memory_percent": 95.2}

        # No room_id attribute
        del mock_alert.room_id

        await mock_integration._handle_performance_alert(mock_alert)

        # Verify alert was triggered with room_id=None
        mock_integration.alert_manager.trigger_alert.assert_called_once_with(
            rule_name="performance_critical",
            title="Performance Alert: Memory exhaustion imminent",
            message="Memory exhaustion imminent",
            component="memory_manager",
            room_id=None,
            context={"memory_percent": 95.2},
        )

    @pytest.mark.asyncio
    async def test_handle_performance_alert_no_additional_info(self, mock_integration):
        """Test performance alert handling without additional info."""
        mock_alert = Mock()
        mock_alert.alert_type = "warning"
        mock_alert.message = "Database slow"
        mock_alert.component = "database"
        mock_alert.additional_info = None

        await mock_integration._handle_performance_alert(mock_alert)

        # Verify alert was triggered with empty context
        mock_integration.alert_manager.trigger_alert.assert_called_once_with(
            rule_name="performance_warning",
            title="Performance Alert: Database slow",
            message="Database slow",
            component="database",
            room_id=None,
            context={},
        )

    @pytest.mark.asyncio
    async def test_handle_performance_alert_failure(self, mock_integration):
        """Test performance alert handling with failure."""
        mock_alert = Mock()
        mock_alert.alert_type = "critical"
        mock_alert.message = "System failure"
        mock_alert.component = "core"
        mock_alert.additional_info = {}

        # Mock alert manager to fail
        mock_integration.alert_manager.trigger_alert.side_effect = Exception(
            "Alert system down"
        )

        # Should not raise exception
        await mock_integration._handle_performance_alert(mock_alert)

        # Verify error was logged
        mock_integration.logger.error.assert_called_once()
        error_call = mock_integration.logger.error.call_args[0][0]
        assert "Failed to handle performance alert" in error_call


class TestMonitoringIntegrationPredictionTracking:
    """Test prediction operation tracking."""

    @pytest.fixture
    def mock_integration(self):
        """Create MonitoringIntegration with mocked dependencies."""
        with patch(
            "src.utils.monitoring_integration.get_logger"
        ) as mock_get_logger, patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ) as mock_get_perf_logger, patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_get_metrics_collector, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_get_monitoring_manager, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            mock_logger = mock_get_logger.return_value
            mock_perf_logger = mock_get_perf_logger.return_value
            mock_metrics_collector = mock_get_metrics_collector.return_value
            mock_monitoring_manager = mock_get_monitoring_manager.return_value
            mock_alert_manager = mock_get_alert_manager.return_value

            # Setup performance monitor mock
            mock_performance_monitor = Mock()
            mock_monitoring_manager.get_performance_monitor.return_value = (
                mock_performance_monitor
            )

            integration = MonitoringIntegration()
            integration.logger = mock_logger
            integration.performance_logger = mock_perf_logger
            integration.metrics_collector = mock_metrics_collector
            integration.monitoring_manager = mock_monitoring_manager
            integration.alert_manager = mock_alert_manager

            yield integration

    @pytest.mark.asyncio
    async def test_track_prediction_operation_success(self, mock_integration):
        """Test successful prediction operation tracking."""
        with patch("src.utils.monitoring_integration.datetime") as mock_datetime:
            start_time = datetime(2024, 1, 15, 10, 0, 0)
            end_time = datetime(2024, 1, 15, 10, 0, 0, 250000)  # 250ms later
            mock_datetime.now.side_effect = [start_time, end_time]

            async with mock_integration.track_prediction_operation(
                room_id="bedroom", prediction_type="next_occupancy", model_type="lstm"
            ):
                # Simulate prediction work
                await asyncio.sleep(0.001)

            # Verify start logging
            start_log_calls = [
                call
                for call in mock_integration.logger.info.call_args_list
                if "Starting prediction" in str(call)
            ]
            assert len(start_log_calls) == 1

            # Verify performance logging
            mock_integration.performance_logger.log_operation_time.assert_called_once_with(
                operation="prediction_generation",
                duration=0.25,  # 250ms in seconds
                room_id="bedroom",
                prediction_type="next_occupancy",
                model_type="lstm",
            )

            # Verify metrics recording
            mock_integration.metrics_collector.record_prediction.assert_called_once_with(
                room_id="bedroom",
                prediction_type="next_occupancy",
                model_type="lstm",
                duration=0.25,
                status="success",
            )

            # Verify performance monitoring
            mock_performance_monitor = (
                mock_integration.monitoring_manager.get_performance_monitor()
            )
            mock_performance_monitor.record_performance_metric.assert_called_once_with(
                "prediction_latency",
                0.25,
                room_id="bedroom",
                additional_info={
                    "prediction_type": "next_occupancy",
                    "model_type": "lstm",
                },
            )

            # Verify completion logging
            completion_log_calls = [
                call
                for call in mock_integration.logger.info.call_args_list
                if "Prediction completed" in str(call)
            ]
            assert len(completion_log_calls) == 1

    @pytest.mark.asyncio
    async def test_track_prediction_operation_failure(self, mock_integration):
        """Test prediction operation tracking with failure."""
        mock_integration.alert_manager.handle_prediction_error = AsyncMock()

        with patch("src.utils.monitoring_integration.datetime") as mock_datetime:
            start_time = datetime(2024, 1, 15, 10, 0, 0)
            end_time = datetime(2024, 1, 15, 10, 0, 0, 150000)  # 150ms later
            mock_datetime.now.side_effect = [start_time, end_time]

            with pytest.raises(ValueError, match="Prediction failed"):
                async with mock_integration.track_prediction_operation(
                    room_id="kitchen",
                    prediction_type="next_vacancy",
                    model_type="xgboost",
                ):
                    raise ValueError("Prediction failed")

            # Verify error metrics recording
            mock_integration.metrics_collector.record_prediction.assert_called_once_with(
                room_id="kitchen",
                prediction_type="next_vacancy",
                model_type="xgboost",
                duration=0.15,
                status="error",
            )

            # Verify error handling
            mock_integration.alert_manager.handle_prediction_error.assert_called_once()
            error_call = (
                mock_integration.alert_manager.handle_prediction_error.call_args
            )
            assert error_call[1]["room_id"] == "kitchen"
            assert error_call[1]["prediction_type"] == "next_vacancy"
            assert error_call[1]["model_type"] == "xgboost"

            # Verify error logging
            error_log_calls = [
                call
                for call in mock_integration.logger.error.call_args_list
                if "Prediction failed" in str(call)
            ]
            assert len(error_log_calls) == 1


class TestMonitoringIntegrationTrainingTracking:
    """Test model training operation tracking."""

    @pytest.fixture
    def mock_integration(self):
        """Create MonitoringIntegration with mocked dependencies."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ) as mock_get_ml_ops_logger, patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_get_metrics_collector, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_get_monitoring_manager, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            mock_ml_ops_logger = mock_get_ml_ops_logger.return_value
            mock_metrics_collector = mock_get_metrics_collector.return_value
            mock_monitoring_manager = mock_get_monitoring_manager.return_value
            mock_alert_manager = mock_get_alert_manager.return_value

            # Setup performance monitor mock
            mock_performance_monitor = Mock()
            mock_monitoring_manager.get_performance_monitor.return_value = (
                mock_performance_monitor
            )

            integration = MonitoringIntegration()
            integration.ml_ops_logger = mock_ml_ops_logger
            integration.metrics_collector = mock_metrics_collector
            integration.monitoring_manager = mock_monitoring_manager
            integration.alert_manager = mock_alert_manager

            yield integration

    @pytest.mark.asyncio
    async def test_track_training_operation_success(self, mock_integration):
        """Test successful training operation tracking."""
        with patch("src.utils.monitoring_integration.datetime") as mock_datetime:
            start_time = datetime(2024, 1, 15, 10, 0, 0)
            end_time = datetime(2024, 1, 15, 10, 2, 30)  # 2.5 minutes later
            mock_datetime.now.side_effect = [start_time, end_time]

            async with mock_integration.track_training_operation(
                room_id="living_room", model_type="lstm", training_type="full_retrain"
            ):
                # Simulate training work
                await asyncio.sleep(0.001)

            # Verify start logging
            mock_integration.ml_ops_logger.log_training_event.assert_any_call(
                room_id="living_room",
                model_type="lstm",
                event_type="full_retrain_start",
            )

            # Verify training metrics recording
            mock_integration.metrics_collector.record_model_training.assert_called_once_with(
                room_id="living_room",
                model_type="lstm",
                training_type="full_retrain",
                duration=150.0,  # 2.5 minutes in seconds
            )

            # Verify performance monitoring
            mock_performance_monitor = (
                mock_integration.monitoring_manager.get_performance_monitor()
            )
            mock_performance_monitor.record_performance_metric.assert_called_once_with(
                "model_training_time",
                150.0,
                room_id="living_room",
                additional_info={"model_type": "lstm", "training_type": "full_retrain"},
            )

            # Verify completion logging
            mock_integration.ml_ops_logger.log_training_event.assert_any_call(
                room_id="living_room",
                model_type="lstm",
                event_type="full_retrain_complete",
                metrics={"duration_seconds": 150.0},
            )

    @pytest.mark.asyncio
    async def test_track_training_operation_failure(self, mock_integration):
        """Test training operation tracking with failure."""
        mock_integration.alert_manager.handle_model_training_error = AsyncMock()

        with patch("src.utils.monitoring_integration.datetime") as mock_datetime:
            start_time = datetime(2024, 1, 15, 10, 0, 0)
            end_time = datetime(2024, 1, 15, 10, 0, 45)  # 45 seconds later
            mock_datetime.now.side_effect = [start_time, end_time]

            with pytest.raises(RuntimeError, match="Training failed"):
                async with mock_integration.track_training_operation(
                    room_id="office",
                    model_type="xgboost",
                    training_type="incremental_update",
                ):
                    raise RuntimeError("Training failed")

            # Verify error handling
            mock_integration.alert_manager.handle_model_training_error.assert_called_once()
            error_call = (
                mock_integration.alert_manager.handle_model_training_error.call_args
            )
            assert error_call[1]["room_id"] == "office"
            assert error_call[1]["model_type"] == "xgboost"

            # Verify error logging
            mock_integration.ml_ops_logger.log_training_event.assert_any_call(
                room_id="office",
                model_type="xgboost",
                event_type="incremental_update_error",
                metrics={"duration_seconds": 45.0, "error": "Training failed"},
            )


class TestMonitoringIntegrationRecordingMethods:
    """Test various recording methods in MonitoringIntegration."""

    @pytest.fixture
    def mock_integration(self):
        """Create MonitoringIntegration with mocked dependencies."""
        with patch("src.utils.monitoring_integration.get_logger"), patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ) as mock_get_perf_logger, patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ) as mock_get_ml_ops_logger, patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ) as mock_get_metrics_collector, patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_get_monitoring_manager, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            mock_perf_logger = mock_get_perf_logger.return_value
            mock_ml_ops_logger = mock_get_ml_ops_logger.return_value
            mock_metrics_collector = mock_get_metrics_collector.return_value
            mock_monitoring_manager = mock_get_monitoring_manager.return_value
            mock_alert_manager = mock_get_alert_manager.return_value

            # Setup performance monitor mock
            mock_performance_monitor = Mock()
            mock_monitoring_manager.get_performance_monitor.return_value = (
                mock_performance_monitor
            )

            integration = MonitoringIntegration()
            integration.performance_logger = mock_perf_logger
            integration.ml_ops_logger = mock_ml_ops_logger
            integration.metrics_collector = mock_metrics_collector
            integration.monitoring_manager = mock_monitoring_manager
            integration.alert_manager = mock_alert_manager

            yield integration

    def test_record_prediction_accuracy(self, mock_integration):
        """Test recording prediction accuracy metrics."""
        mock_integration.record_prediction_accuracy(
            room_id="bedroom",
            model_type="ensemble",
            prediction_type="next_occupancy",
            accuracy_minutes=8.5,
            confidence=0.92,
        )

        # Verify performance logging
        mock_integration.performance_logger.log_prediction_accuracy.assert_called_once_with(
            room_id="bedroom",
            accuracy_minutes=8.5,
            confidence=0.92,
            prediction_type="next_occupancy",
        )

        # Verify metrics recording
        mock_integration.metrics_collector.record_prediction.assert_called_once_with(
            room_id="bedroom",
            prediction_type="next_occupancy",
            model_type="ensemble",
            duration=0,  # Already recorded during prediction
            accuracy_minutes=8.5,
            confidence=0.92,
        )

        # Verify performance monitoring
        mock_performance_monitor = (
            mock_integration.monitoring_manager.get_performance_monitor()
        )
        mock_performance_monitor.record_performance_metric.assert_called_once_with(
            "prediction_accuracy",
            8.5,
            room_id="bedroom",
            additional_info={
                "model_type": "ensemble",
                "prediction_type": "next_occupancy",
                "confidence": 0.92,
            },
        )

    def test_record_concept_drift_low_severity(self, mock_integration):
        """Test recording concept drift with low severity."""
        mock_integration.record_concept_drift(
            room_id="kitchen",
            drift_type="statistical",
            severity=0.3,  # Low severity
            action_taken="monitoring",
        )

        # Verify ML ops logging
        mock_integration.ml_ops_logger.log_drift_detection.assert_called_once_with(
            room_id="kitchen",
            drift_type="statistical",
            severity=0.3,
            action_taken="monitoring",
        )

        # Verify metrics recording
        mock_integration.metrics_collector.record_concept_drift.assert_called_once_with(
            room_id="kitchen",
            drift_type="statistical",
            severity=0.3,
            action_taken="monitoring",
        )

        # Verify no alert was triggered (severity <= 0.5)
        # We can't easily verify this directly, but alert_manager.trigger_alert should not be called

    @patch("asyncio.create_task")
    def test_record_concept_drift_high_severity(
        self, mock_create_task, mock_integration
    ):
        """Test recording concept drift with high severity triggers alert."""
        mock_integration.record_concept_drift(
            room_id="garage",
            drift_type="performance",
            severity=0.8,  # High severity
            action_taken="model_retrain",
        )

        # Verify alert task was created
        mock_create_task.assert_called_once()

        # Get the coroutine that was passed to create_task
        alert_coro = mock_create_task.call_args[0][0]

        # We can't easily test the coroutine content without running it,
        # but we can verify it was created
        assert alert_coro is not None

    def test_record_feature_computation(self, mock_integration):
        """Test recording feature computation metrics."""
        mock_integration.record_feature_computation(
            room_id="living_room", feature_type="temporal", duration=0.125
        )

        # Verify metrics recording
        mock_integration.metrics_collector.record_feature_computation.assert_called_once_with(
            room_id="living_room", feature_type="temporal", duration=0.125
        )

        # Verify performance monitoring
        mock_performance_monitor = (
            mock_integration.monitoring_manager.get_performance_monitor()
        )
        mock_performance_monitor.record_performance_metric.assert_called_once_with(
            "feature_computation_time",
            0.125,
            room_id="living_room",
            additional_info={"feature_type": "temporal"},
        )

    def test_record_database_operation(self, mock_integration):
        """Test recording database operation metrics."""
        mock_integration.record_database_operation(
            operation_type="select",
            table="sensor_events",
            duration=0.45,
            status="success",
        )

        # Verify metrics recording
        mock_integration.metrics_collector.record_database_operation.assert_called_once_with(
            operation_type="select",
            table="sensor_events",
            duration=0.45,
            status="success",
        )

        # Verify performance monitoring
        mock_performance_monitor = (
            mock_integration.monitoring_manager.get_performance_monitor()
        )
        mock_performance_monitor.record_performance_metric.assert_called_once_with(
            "database_query_time",
            0.45,
            additional_info={
                "operation_type": "select",
                "table": "sensor_events",
                "status": "success",
            },
        )

    def test_record_mqtt_publish(self, mock_integration):
        """Test recording MQTT publish metrics."""
        mock_integration.record_mqtt_publish(
            topic_type="prediction", room_id="bathroom", status="success"
        )

        # Verify metrics recording
        mock_integration.metrics_collector.record_mqtt_publish.assert_called_once_with(
            topic_type="prediction", room_id="bathroom", status="success"
        )

    def test_record_ha_api_request(self, mock_integration):
        """Test recording Home Assistant API request metrics."""
        mock_integration.record_ha_api_request(
            endpoint="/api/states", method="GET", status="200"
        )

        # Verify metrics recording
        mock_integration.metrics_collector.record_ha_api_request.assert_called_once_with(
            endpoint="/api/states", method="GET", status="200"
        )

    @patch("asyncio.create_task")
    def test_update_connection_status_connected(
        self, mock_create_task, mock_integration
    ):
        """Test updating connection status when connected."""
        mock_integration.update_connection_status(
            connection_type="websocket", connected=True
        )

        # Verify metrics recording
        mock_integration.metrics_collector.update_ha_connection_status.assert_called_once_with(
            connection_type="websocket", connected=True
        )

        # Verify no alert task was created (connection is up)
        mock_create_task.assert_not_called()

    @patch("asyncio.create_task")
    def test_update_connection_status_disconnected(
        self, mock_create_task, mock_integration
    ):
        """Test updating connection status when disconnected triggers alert."""
        mock_integration.update_connection_status(
            connection_type="rest_api", connected=False
        )

        # Verify metrics recording
        mock_integration.metrics_collector.update_ha_connection_status.assert_called_once_with(
            connection_type="rest_api", connected=False
        )

        # Verify alert task was created
        mock_create_task.assert_called_once()


class TestMonitoringIntegrationStatus:
    """Test monitoring status retrieval."""

    @pytest.fixture
    def mock_integration(self):
        """Create MonitoringIntegration with mocked dependencies."""
        with patch(
            "src.utils.monitoring_integration.get_logger"
        ) as mock_get_logger, patch(
            "src.utils.monitoring_integration.get_performance_logger"
        ), patch(
            "src.utils.monitoring_integration.get_ml_ops_logger"
        ), patch(
            "src.utils.monitoring_integration.get_metrics_manager"
        ) as mock_get_metrics_manager, patch(
            "src.utils.monitoring_integration.get_metrics_collector"
        ), patch(
            "src.utils.monitoring_integration.get_monitoring_manager"
        ) as mock_get_monitoring_manager, patch(
            "src.utils.monitoring_integration.get_alert_manager"
        ) as mock_get_alert_manager:

            mock_logger = mock_get_logger.return_value
            mock_metrics_manager = mock_get_metrics_manager.return_value
            mock_monitoring_manager = mock_get_monitoring_manager.return_value
            mock_alert_manager = mock_get_alert_manager.return_value

            integration = MonitoringIntegration()
            integration.logger = mock_logger
            integration.metrics_manager = mock_metrics_manager
            integration.monitoring_manager = mock_monitoring_manager
            integration.alert_manager = mock_alert_manager

            yield integration

    @pytest.mark.asyncio
    async def test_get_monitoring_status_success(self, mock_integration):
        """Test successful monitoring status retrieval."""
        # Mock monitoring manager status
        mock_monitoring_status = {
            "monitoring_active": True,
            "health_status": "healthy",
            "health_details": {"cpu_percent": 45.0},
            "performance_summary": {"prediction_latency": {"mean": 0.2}},
        }
        mock_integration.monitoring_manager.get_monitoring_status = AsyncMock(
            return_value=mock_monitoring_status
        )

        # Mock alert manager status
        mock_alert_status = {"active_alerts": 2, "alert_rules": 15}
        mock_integration.alert_manager.get_alert_status.return_value = mock_alert_status

        # Mock metrics manager
        mock_integration.metrics_manager.get_metrics.return_value = "# Test metrics\n"

        status = await mock_integration.get_monitoring_status()

        # Verify status structure
        assert "monitoring_system" in status
        assert "alert_system" in status
        assert "metrics_collection" in status
        assert "integration_status" in status
        assert "timestamp" in status

        # Verify monitoring system status
        assert status["monitoring_system"] == mock_monitoring_status

        # Verify alert system status
        assert status["alert_system"] == mock_alert_status

        # Verify metrics collection status
        assert status["metrics_collection"]["enabled"] is True
        assert status["metrics_collection"]["endpoint_available"] is True

        # Verify integration status
        integration_status = status["integration_status"]
        assert integration_status["performance_tracking"] is True
        assert integration_status["error_handling"] is True
        assert integration_status["ml_ops_logging"] is True

        # Verify timestamp
        assert isinstance(status["timestamp"], str)
        datetime.fromisoformat(status["timestamp"])  # Should not raise

    @pytest.mark.asyncio
    async def test_get_monitoring_status_with_empty_metrics(self, mock_integration):
        """Test monitoring status when metrics are empty."""
        mock_integration.monitoring_manager.get_monitoring_status = AsyncMock(
            return_value={"monitoring_active": False}
        )
        mock_integration.alert_manager.get_alert_status.return_value = {}
        mock_integration.metrics_manager.get_metrics.return_value = ""  # Empty metrics

        status = await mock_integration.get_monitoring_status()

        # Verify metrics collection shows as disabled
        assert status["metrics_collection"]["enabled"] is False

    @pytest.mark.asyncio
    async def test_get_monitoring_status_failure(self, mock_integration):
        """Test monitoring status retrieval with failure."""
        # Mock monitoring manager to fail
        mock_integration.monitoring_manager.get_monitoring_status = AsyncMock(
            side_effect=Exception("Monitoring status failed")
        )

        status = await mock_integration.get_monitoring_status()

        # Verify error response
        assert "error" in status
        assert "Monitoring status failed" in status["error"]
        assert "timestamp" in status

        # Verify error was logged
        mock_integration.logger.error.assert_called_once()
        error_call = mock_integration.logger.error.call_args[0][0]
        assert "Failed to get monitoring status" in error_call


class TestMonitoringIntegrationGlobalFunction:
    """Test global function for MonitoringIntegration."""

    def test_get_monitoring_integration_singleton(self):
        """Test monitoring integration singleton behavior."""
        # Clear any existing global state
        import src.utils.monitoring_integration

        original_integration = src.utils.monitoring_integration._monitoring_integration

        try:
            # Reset global state
            src.utils.monitoring_integration._monitoring_integration = None

            integration1 = get_monitoring_integration()
            integration2 = get_monitoring_integration()

            assert integration1 is integration2
            assert isinstance(integration1, MonitoringIntegration)

        finally:
            # Restore original state
            src.utils.monitoring_integration._monitoring_integration = (
                original_integration
            )


if __name__ == "__main__":
    pytest.main([__file__])
