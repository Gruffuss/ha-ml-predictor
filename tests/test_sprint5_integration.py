"""
Comprehensive Sprint 5 Integration Tests.

This module provides comprehensive integration testing for Sprint 5 components,
focusing on the integration between TrackingManager, API server, MQTT discovery,
real-time publishing, and enhanced MQTT manager.

Test Coverage:
- TrackingManager integration with API server and MQTT
- Enhanced MQTT manager with real-time publishing capabilities
- API server endpoints with authentication and rate limiting
- MQTT discovery configuration and entity creation
- Real-time publishing system (WebSocket, SSE, MQTT channels)
- Dashboard integration and metrics collection
- Error handling and graceful degradation
- Performance and resource usage validation
"""

import asyncio
from dataclasses import asdict
from datetime import datetime, timedelta
import logging
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from aiohttp import web
from aiohttp.test_utils import AioHTTPTestCase
from fastapi.testclient import TestClient
from httpx import AsyncClient
import pytest
import pytest_asyncio
import websockets

from src.adaptation.tracking_manager import TrackingConfig, TrackingManager
from src.core.config import APIConfig, MQTTConfig, get_config
from src.core.exceptions import (
    APIAuthenticationError,
    APIRateLimitError,
    APIServerError,
    ErrorSeverity,
)
from src.integration.api_server import (
    APIServer,
    create_app,
    integrate_with_tracking_manager,
    set_tracking_manager,
)
from src.integration.enhanced_mqtt_manager import (
    EnhancedIntegrationStats,
    EnhancedMQTTIntegrationManager,
)
from src.integration.mqtt_integration_manager import MQTTIntegrationStats
from src.integration.realtime_publisher import (
    PublishingChannel,
    PublishingMetrics,
    RealtimePredictionEvent,
    RealtimePublishingSystem,
)
from src.integration.tracking_integration import (
    IntegrationConfig,
    TrackingIntegrationManager,
    create_integrated_tracking_manager,
    integrate_tracking_with_realtime_publishing,
)
from src.models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


@pytest.fixture
def test_tracking_config():
    """Create a test tracking configuration."""
    return TrackingConfig(
        enabled=True,
        monitoring_interval_seconds=300,  # 5 minutes
        auto_validation_enabled=True,
        validation_window_minutes=30,  # 30 minutes
        alert_thresholds={
            "accuracy_threshold_minutes": 15,
            "confidence_threshold": 0.7,
        },
        realtime_publishing_enabled=True,
        websocket_enabled=True,
        sse_enabled=True,
        drift_detection_enabled=True,
        drift_check_interval_hours=6,
    )


@pytest.fixture
def test_integration_config():
    """Create a test integration configuration."""
    return IntegrationConfig(
        enable_realtime_publishing=True,
        enable_websocket_server=True,
        enable_sse_server=True,
        websocket_port=8765,
        sse_enabled_in_api=True,
        max_websocket_connections=100,
        max_sse_connections=50,
        connection_timeout_minutes=60,
        publish_system_status_interval_seconds=30,
        broadcast_alerts=True,
        broadcast_drift_events=True,
    )


@pytest.fixture
def mock_tracking_manager():
    """Create a comprehensive mock TrackingManager."""
    manager = AsyncMock(spec=TrackingManager)

    # Mock basic methods
    manager.initialize = AsyncMock()
    manager.shutdown = AsyncMock()
    manager.is_running = Mock(return_value=True)

    # Mock prediction methods
    manager.get_room_prediction = AsyncMock(
        return_value={
            "room_id": "test_room",
            "prediction_time": datetime.utcnow().isoformat(),
            "next_transition_time": (
                datetime.utcnow() + timedelta(minutes=30)
            ).isoformat(),
            "transition_type": "occupied_to_vacant",
            "confidence": 0.85,
            "time_until_transition": "30 minutes",
            "alternatives": [],
            "model_info": {"model_type": "ensemble", "version": "1.0"},
        }
    )

    # Mock accuracy methods
    manager.get_accuracy_metrics = AsyncMock(
        return_value={
            "room_id": "test_room",
            "accuracy_rate": 0.87,
            "average_error_minutes": 12.5,
            "confidence_calibration": 0.92,
            "total_predictions": 150,
            "total_validations": 140,
            "time_window_hours": 24,
            "trend_direction": "improving",
        }
    )

    # Mock system stats
    manager.get_system_stats = AsyncMock(
        return_value={
            "tracking_stats": {
                "total_predictions_tracked": 150,
                "active_tracking_sessions": 3,
                "average_accuracy": 0.87,
            },
            "retraining_stats": {
                "completed_retraining_jobs": 5,
                "last_retrain_time": datetime.utcnow().isoformat(),
                "average_retrain_duration_minutes": 15,
            },
        }
    )

    # Mock tracking status
    manager.get_tracking_status = AsyncMock(
        return_value={
            "system_health": "healthy",
            "active_predictions": 12,
            "tracking_accuracy": 0.87,
            "last_update": datetime.utcnow().isoformat(),
        }
    )

    # Mock retraining
    manager.trigger_manual_retrain = AsyncMock(
        return_value={
            "message": "Retraining initiated successfully",
            "success": True,
            "room_id": "test_room",
            "strategy": "auto",
            "force": False,
        }
    )

    # Mock alerts
    manager.get_active_alerts = AsyncMock(return_value=[])
    manager.notification_callbacks = []
    manager.add_notification_callback = Mock()

    # Mock MQTT integration attribute
    manager.mqtt_integration_manager = None

    return manager


@pytest.fixture
def mock_enhanced_mqtt_manager():
    """Create a mock enhanced MQTT manager."""
    manager = AsyncMock(spec=EnhancedMQTTIntegrationManager)

    # Mock initialization
    manager.initialize = AsyncMock()
    manager.shutdown = AsyncMock()

    # Mock publishing methods
    manager.publish_prediction = AsyncMock()
    manager.publish_system_status = AsyncMock()

    # Mock real-time capabilities
    manager.handle_websocket_connection = AsyncMock()
    manager.create_sse_stream = AsyncMock()
    manager.add_realtime_callback = Mock()
    manager.remove_realtime_callback = Mock()

    # Mock stats
    manager.get_integration_stats = Mock(
        return_value={
            "mqtt_connected": True,
            "predictions_published": 50,
            "realtime_clients": 3,
            "websocket_connections": 2,
            "sse_connections": 1,
        }
    )

    manager.get_connection_info = Mock(
        return_value={
            "total_active_connections": 3,
            "websocket_connections": {
                "total_active_connections": 2,
                "clients": [],
            },
            "sse_connections": {"total_active_connections": 1, "clients": []},
        }
    )

    return manager


@pytest.fixture
def mock_api_config():
    """Create a mock API configuration."""
    return APIConfig(
        enabled=True,
        host="127.0.0.1",
        port=8000,
        debug=True,
        api_key_enabled=False,
        api_key="test_api_key",
        rate_limit_enabled=False,
        requests_per_minute=60,
        enable_cors=True,
        cors_origins=["*"],
        include_docs=True,
        docs_url="/docs",
        redoc_url="/redoc",
        background_tasks_enabled=True,
        health_check_interval_seconds=30,
        log_requests=True,
        log_responses=False,
        access_log=True,
    )


class TestTrackingIntegrationManager:
    """Test the TrackingIntegrationManager for system integration."""

    @pytest_asyncio.fixture
    async def integration_manager(
        self,
        mock_tracking_manager,
        test_integration_config,
        mock_enhanced_mqtt_manager,
    ):
        """Create a TrackingIntegrationManager for testing."""
        with patch(
            "src.integration.tracking_integration.EnhancedMQTTIntegrationManager",
            return_value=mock_enhanced_mqtt_manager,
        ):
            manager = TrackingIntegrationManager(
                tracking_manager=mock_tracking_manager,
                integration_config=test_integration_config,
            )
            await manager.initialize()
            yield manager
            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_integration_manager_initialization(self, integration_manager):
        """Test TrackingIntegrationManager initializes correctly."""
        assert integration_manager._integration_active
        assert integration_manager.enhanced_mqtt_manager is not None
        assert len(integration_manager._background_tasks) > 0

    @pytest.mark.asyncio
    async def test_integration_manager_stats(self, integration_manager):
        """Test getting integration statistics."""
        stats = integration_manager.get_integration_stats()

        assert stats["integration_active"]
        assert "integration_config" in stats
        assert "enhanced_mqtt_stats" in stats
        assert "performance" in stats

    @pytest.mark.asyncio
    async def test_websocket_and_sse_handlers(self, integration_manager):
        """Test WebSocket and SSE handler availability."""
        ws_handler = integration_manager.get_websocket_handler()
        sse_handler = integration_manager.get_sse_handler()

        assert ws_handler is not None
        assert sse_handler is not None

    @pytest.mark.asyncio
    async def test_realtime_callback_management(self, integration_manager):
        """Test real-time callback management."""
        callback = Mock()

        # Add callback
        integration_manager.add_realtime_callback(callback)
        integration_manager.enhanced_mqtt_manager.add_realtime_callback.assert_called_once_with(
            callback
        )

        # Remove callback
        integration_manager.remove_realtime_callback(callback)
        integration_manager.enhanced_mqtt_manager.remove_realtime_callback.assert_called_once_with(
            callback
        )

    @pytest.mark.asyncio
    async def test_system_status_broadcast_loop(self, integration_manager):
        """Test the system status broadcast functionality."""
        # Let the background task run briefly
        await asyncio.sleep(0.1)

        # Check that system status broadcasting was attempted
        integration_manager.enhanced_mqtt_manager.publish_system_status.assert_called()


class TestEnhancedMQTTIntegrationManager:
    """Test the Enhanced MQTT Integration Manager."""

    @pytest_asyncio.fixture
    async def enhanced_mqtt_manager(self, test_system_config):
        """Create an Enhanced MQTT Integration Manager for testing."""
        with patch(
            "src.integration.enhanced_mqtt_manager.MQTTIntegrationManager"
        ) as mock_base:
            with patch(
                "src.integration.enhanced_mqtt_manager.RealtimePublishingSystem"
            ) as mock_realtime:

                # Configure mocks
                mock_base_instance = AsyncMock()
                mock_base.return_value = mock_base_instance

                mock_realtime_instance = AsyncMock()
                mock_realtime.return_value = mock_realtime_instance

                manager = EnhancedMQTTIntegrationManager(
                    mqtt_config=test_system_config.mqtt,
                    rooms=test_system_config.rooms,
                    enabled_realtime_channels=[
                        PublishingChannel.MQTT,
                        PublishingChannel.WEBSOCKET,
                        PublishingChannel.SSE,
                    ],
                )

                await manager.initialize()
                yield manager
                await manager.shutdown()

    @pytest.mark.asyncio
    async def test_enhanced_mqtt_manager_initialization(self, enhanced_mqtt_manager):
        """Test enhanced MQTT manager initializes with real-time capabilities."""
        assert enhanced_mqtt_manager.base_mqtt_manager is not None
        assert enhanced_mqtt_manager.realtime_publisher is not None
        assert enhanced_mqtt_manager._integration_active

    @pytest.mark.asyncio
    async def test_prediction_publishing_with_realtime(self, enhanced_mqtt_manager):
        """Test prediction publishing across multiple channels."""
        prediction_data = {
            "room_id": "test_room",
            "prediction_time": datetime.utcnow().isoformat(),
            "confidence": 0.85,
        }

        await enhanced_mqtt_manager.publish_prediction("test_room", prediction_data)

        # Verify both MQTT and real-time publishing were called
        enhanced_mqtt_manager.base_mqtt_manager.publish_prediction.assert_called_once()
        enhanced_mqtt_manager.realtime_publisher.publish_prediction.assert_called_once()

    @pytest.mark.asyncio
    async def test_system_status_publishing(self, enhanced_mqtt_manager):
        """Test system status publishing with real-time broadcast."""
        status_data = {"system_health": "healthy", "active_predictions": 5}

        await enhanced_mqtt_manager.publish_system_status(**status_data)

        # Verify real-time broadcasting
        enhanced_mqtt_manager.realtime_publisher.broadcast_system_event.assert_called_once()


class TestAPIServerIntegration:
    """Test API server integration with TrackingManager."""

    @pytest_asyncio.fixture
    async def api_server(self, mock_tracking_manager, mock_api_config):
        """Create an API server with tracking manager integration."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            # Configure mock config
            mock_config = Mock()
            mock_config.api = mock_api_config
            mock_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_config

            # Set tracking manager for API endpoints
            set_tracking_manager(mock_tracking_manager)

            server = await integrate_with_tracking_manager(mock_tracking_manager)
            yield server

            if server.is_running():
                await server.stop()

    @pytest.mark.asyncio
    async def test_api_server_integration(self, api_server, mock_tracking_manager):
        """Test API server integrates correctly with TrackingManager."""
        assert api_server.tracking_manager == mock_tracking_manager
        assert not api_server.is_running()  # Not started yet

    @pytest.mark.asyncio
    async def test_api_endpoints_with_tracking_manager(
        self, mock_tracking_manager, mock_api_config
    ):
        """Test API endpoints work with integrated TrackingManager."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            # Configure mock config
            mock_config = Mock()
            mock_config.api = mock_api_config
            mock_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_config

            # Set tracking manager
            set_tracking_manager(mock_tracking_manager)

            # Create test client
            app = create_app()

            with TestClient(app) as client:
                # Test root endpoint
                response = client.get("/")
                assert response.status_code == 200
                assert "Occupancy Prediction API" in response.json()["name"]

                # Test health endpoint
                with patch(
                    "src.integration.api_server.get_database_manager"
                ) as mock_db:
                    mock_db_manager = AsyncMock()
                    mock_db_manager.health_check = AsyncMock(
                        return_value={"database_connected": True}
                    )
                    mock_db.return_value = mock_db_manager

                    with patch(
                        "src.integration.api_server.get_mqtt_manager"
                    ) as mock_mqtt:
                        mock_mqtt_manager = AsyncMock()
                        mock_mqtt_manager.get_integration_stats = AsyncMock(
                            return_value=Mock(
                                mqtt_connected=True, predictions_published=10
                            )
                        )
                        mock_mqtt.return_value = mock_mqtt_manager

                        response = client.get("/health")
                        assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_api_prediction_endpoint(
        self, mock_tracking_manager, mock_api_config
    ):
        """Test prediction endpoint with TrackingManager integration."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            # Configure mock config
            mock_config = Mock()
            mock_config.api = mock_api_config
            mock_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_config

            # Set tracking manager
            set_tracking_manager(mock_tracking_manager)

            app = create_app()

            with TestClient(app) as client:
                response = client.get("/predictions/test_room")
                assert response.status_code == 200

                data = response.json()
                assert data["room_id"] == "test_room"
                assert "confidence" in data

                # Verify TrackingManager was called
                mock_tracking_manager.get_room_prediction.assert_called_once_with(
                    "test_room"
                )

    @pytest.mark.asyncio
    async def test_api_retrain_endpoint(self, mock_tracking_manager, mock_api_config):
        """Test manual retrain endpoint."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            # Configure mock config
            mock_config = Mock()
            mock_config.api = mock_api_config
            mock_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_config

            # Set tracking manager
            set_tracking_manager(mock_tracking_manager)

            app = create_app()

            with TestClient(app) as client:
                headers = {
                    "Authorization": "Bearer test_api_key_for_security_validation_testing"
                }
                response = client.post(
                    "/model/retrain",
                    json={
                        "room_id": "test_room",
                        "force": True,
                        "strategy": "full",
                        "reason": "test_request",
                    },
                    headers=headers,
                )
                assert response.status_code == 200

                # Verify TrackingManager was called
                mock_tracking_manager.trigger_manual_retrain.assert_called_once()


class TestSystemIntegrationFlow:
    """Test complete system integration flow."""

    @pytest_asyncio.fixture
    async def integrated_system(
        self, test_tracking_config, test_integration_config, mock_api_config
    ):
        """Create a fully integrated system for testing."""
        with patch(
            "src.adaptation.tracking_manager.TrackingManager"
        ) as mock_tracking_class:
            with patch(
                "src.integration.tracking_integration.EnhancedMQTTIntegrationManager"
            ) as mock_mqtt_class:

                # Configure mocks
                mock_tracking = AsyncMock()
                mock_tracking.initialize = AsyncMock()
                mock_tracking.shutdown = AsyncMock()
                mock_tracking.notification_callbacks = []
                mock_tracking.get_tracking_status = AsyncMock(
                    return_value={"status": "healthy"}
                )
                mock_tracking.get_active_alerts = AsyncMock(return_value=[])
                mock_tracking_class.return_value = mock_tracking

                mock_mqtt = AsyncMock()
                mock_mqtt.initialize = AsyncMock()
                mock_mqtt.shutdown = AsyncMock()
                mock_mqtt_class.return_value = mock_mqtt

                # Create integrated system
                tracking_manager, integration_manager = (
                    await create_integrated_tracking_manager(
                        tracking_config=test_tracking_config,
                        integration_config=test_integration_config,
                    )
                )

                # Create API server integration
                with patch("src.integration.api_server.get_config") as mock_get_config:
                    mock_config = Mock()
                    mock_config.api = mock_api_config
                    mock_config.rooms = {"test_room": Mock()}
                    mock_get_config.return_value = mock_config

                    api_server = await integrate_with_tracking_manager(tracking_manager)

                yield {
                    "tracking_manager": tracking_manager,
                    "integration_manager": integration_manager,
                    "api_server": api_server,
                    "mock_tracking": mock_tracking,
                    "mock_mqtt": mock_mqtt,
                }

                # Cleanup
                await integration_manager.shutdown()
                if api_server.is_running():
                    await api_server.stop()

    @pytest.mark.asyncio
    async def test_integrated_system_initialization(self, integrated_system):
        """Test complete integrated system initializes correctly."""
        tracking_manager = integrated_system["tracking_manager"]
        integration_manager = integrated_system["integration_manager"]
        api_server = integrated_system["api_server"]

        # Verify all components are initialized
        assert tracking_manager is not None
        assert integration_manager is not None
        assert integration_manager._integration_active
        assert api_server is not None

    @pytest.mark.asyncio
    async def test_prediction_flow_integration(self, integrated_system):
        """Test complete prediction flow through integrated system."""
        tracking_manager = integrated_system["tracking_manager"]
        integration_manager = integrated_system["integration_manager"]
        mock_tracking = integrated_system["mock_tracking"]
        mock_mqtt = integrated_system["mock_mqtt"]

        # Mock a prediction result
        prediction_data = {
            "room_id": "test_room",
            "prediction_time": datetime.utcnow().isoformat(),
            "confidence": 0.85,
        }
        mock_tracking.get_room_prediction = AsyncMock(return_value=prediction_data)

        # Simulate prediction request
        result = await tracking_manager.get_room_prediction("test_room")

        assert result == prediction_data
        mock_tracking.get_room_prediction.assert_called_once_with("test_room")

    @pytest.mark.asyncio
    async def test_system_stats_integration(self, integrated_system):
        """Test system statistics integration across components."""
        integration_manager = integrated_system["integration_manager"]

        # Get integration stats
        stats = integration_manager.get_integration_stats()

        # Verify comprehensive stats
        assert "integration_active" in stats
        assert "enhanced_mqtt_stats" in stats
        assert "performance" in stats
        assert stats["integration_active"]


class TestErrorHandlingAndRecovery:
    """Test error handling and recovery mechanisms in integration."""

    @pytest.mark.asyncio
    async def test_api_server_error_handling(self, mock_api_config):
        """Test API server handles errors gracefully."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            # Configure mock config
            mock_config = Mock()
            mock_config.api = mock_api_config
            mock_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_config

            # Create tracking manager that raises errors
            mock_tracking = AsyncMock()
            mock_tracking.get_room_prediction = AsyncMock(
                side_effect=Exception("Test error")
            )
            set_tracking_manager(mock_tracking)

            app = create_app()

            with TestClient(app) as client:
                response = client.get("/predictions/test_room")
                assert response.status_code == 500
                assert "error" in response.json()

    @pytest.mark.asyncio
    async def test_integration_manager_error_recovery(
        self, mock_tracking_manager, test_integration_config
    ):
        """Test integration manager handles initialization errors."""
        with patch(
            "src.integration.tracking_integration.EnhancedMQTTIntegrationManager"
        ) as mock_mqtt_class:
            # Make MQTT manager initialization fail
            mock_mqtt_class.side_effect = Exception("MQTT initialization failed")

            manager = TrackingIntegrationManager(
                tracking_manager=mock_tracking_manager,
                integration_config=test_integration_config,
            )

            # Initialization should fail
            with pytest.raises(Exception):
                await manager.initialize()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_with_errors(
        self,
        mock_tracking_manager,
        test_integration_config,
        mock_enhanced_mqtt_manager,
    ):
        """Test graceful shutdown even when errors occur."""
        with patch(
            "src.integration.tracking_integration.EnhancedMQTTIntegrationManager",
            return_value=mock_enhanced_mqtt_manager,
        ):

            # Make shutdown raise an error
            mock_enhanced_mqtt_manager.shutdown.side_effect = Exception(
                "Shutdown error"
            )

            manager = TrackingIntegrationManager(
                tracking_manager=mock_tracking_manager,
                integration_config=test_integration_config,
            )
            await manager.initialize()

            # Shutdown should not raise exception despite error
            await manager.shutdown()
            assert not manager._integration_active


class TestPerformanceAndResourceUsage:
    """Test performance characteristics and resource usage."""

    @pytest.mark.asyncio
    async def test_connection_monitoring(
        self,
        mock_tracking_manager,
        test_integration_config,
        mock_enhanced_mqtt_manager,
    ):
        """Test connection monitoring and limits."""
        # Configure connection info
        mock_enhanced_mqtt_manager.get_connection_info.return_value = {
            "total_active_connections": 150,
            "websocket_connections": {"total_active_connections": 100},
            "sse_connections": {"total_active_connections": 50},
        }

        with patch(
            "src.integration.tracking_integration.EnhancedMQTTIntegrationManager",
            return_value=mock_enhanced_mqtt_manager,
        ):

            manager = TrackingIntegrationManager(
                tracking_manager=mock_tracking_manager,
                integration_config=test_integration_config,
            )
            await manager.initialize()

            # Let monitoring loop run briefly
            await asyncio.sleep(0.1)

            # Verify connection monitoring was called
            mock_enhanced_mqtt_manager.get_connection_info.assert_called()

            await manager.shutdown()

    @pytest.mark.asyncio
    async def test_background_task_management(
        self,
        mock_tracking_manager,
        test_integration_config,
        mock_enhanced_mqtt_manager,
    ):
        """Test background task creation and cleanup."""
        with patch(
            "src.integration.tracking_integration.EnhancedMQTTIntegrationManager",
            return_value=mock_enhanced_mqtt_manager,
        ):

            manager = TrackingIntegrationManager(
                tracking_manager=mock_tracking_manager,
                integration_config=test_integration_config,
            )
            await manager.initialize()

            # Verify background tasks were created
            assert len(manager._background_tasks) > 0

            # Verify tasks are running
            for task in manager._background_tasks:
                assert not task.done()

            # Shutdown and verify tasks are cancelled
            await manager.shutdown()

            for task in manager._background_tasks:
                assert task.done() or task.cancelled()


class TestConfigurationAndDiscovery:
    """Test configuration handling and MQTT discovery."""

    @pytest.mark.asyncio
    async def test_mqtt_discovery_integration(self, mock_api_config):
        """Test MQTT discovery refresh endpoint."""
        with patch("src.integration.api_server.get_config") as mock_get_config:
            # Configure mock config
            mock_config = Mock()
            mock_config.api = mock_api_config
            mock_config.rooms = {"test_room": Mock()}
            mock_get_config.return_value = mock_config

            with patch("src.integration.api_server.get_mqtt_manager") as mock_mqtt:
                mock_mqtt_manager = AsyncMock()
                mock_mqtt_manager.cleanup_discovery = AsyncMock()
                mock_mqtt_manager.initialize = AsyncMock()
                mock_mqtt.return_value = mock_mqtt_manager

                app = create_app()

                with TestClient(app) as client:
                    response = client.post("/mqtt/refresh")
                    assert response.status_code == 200

                    # Verify discovery refresh was called
                    mock_mqtt_manager.cleanup_discovery.assert_called_once()
                    mock_mqtt_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_integration_config_validation(self, mock_tracking_manager):
        """Test integration configuration validation."""
        # Test with disabled real-time publishing
        disabled_config = IntegrationConfig(enable_realtime_publishing=False)

        manager = TrackingIntegrationManager(
            tracking_manager=mock_tracking_manager,
            integration_config=disabled_config,
        )
        await manager.initialize()

        # Should not have enhanced MQTT manager when disabled
        assert manager.enhanced_mqtt_manager is None

        await manager.shutdown()


@pytest.mark.asyncio
async def test_integration_helper_functions():
    """Test integration helper functions."""
    # Test the integration factory function
    with patch(
        "src.adaptation.tracking_manager.TrackingManager"
    ) as mock_tracking_class:
        with patch(
            "src.integration.tracking_integration.integrate_tracking_with_realtime_publishing"
        ) as mock_integrate:

            mock_tracking = AsyncMock()
            mock_tracking.initialize = AsyncMock()
            mock_tracking_class.return_value = mock_tracking

            mock_integration_manager = AsyncMock()
            mock_integrate.return_value = mock_integration_manager

            tracking_config = TrackingConfig()
            integration_config = IntegrationConfig()

            tracking_manager, integration_manager = (
                await create_integrated_tracking_manager(
                    tracking_config=tracking_config,
                    integration_config=integration_config,
                )
            )

            # Verify factory function worked
            assert tracking_manager == mock_tracking
            assert integration_manager == mock_integration_manager

            # Verify initialization was called
            mock_tracking.initialize.assert_called_once()
            mock_integrate.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
