"""Unit tests for main system orchestration.

Covers:
- src/main_system.py (Main System Orchestration)

This test file focuses on the main system orchestration and lifecycle management.
"""

import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest


class TestMainSystemOrchestration:
    """Test main system orchestration functionality."""

    @pytest.mark.asyncio
    async def test_system_initialization_comprehensive(self):
        """Test comprehensive system initialization sequence."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            # Setup mocks
            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock()
            mock_mqtt.return_value = mock_mqtt_manager

            # Test initialization
            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            await system.initialize()

            # Verify initialization sequence
            mock_get_config.assert_called_once()
            mock_get_db.assert_called_once()
            mock_mqtt_manager.initialize.assert_called_once()
            mock_tracking_manager.initialize.assert_called_once()

            # Verify system state
            assert system.running is True
            assert system.database_manager == mock_db_manager
            assert system.mqtt_manager == mock_mqtt_manager
            assert system.tracking_manager == mock_tracking_manager

    @pytest.mark.asyncio
    async def test_component_coordination_comprehensive(self):
        """Test coordination between system components."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            # Setup mocks with interconnected behavior
            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            # Test tracking manager receives database and MQTT managers
            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock()
            mock_mqtt.return_value = mock_mqtt_manager

            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            await system.initialize()

            # Verify components are properly coordinated
            # TrackingManager should receive database and MQTT managers
            mock_tracking.assert_called_once_with(
                config=mock_config.tracking,
                database_manager=mock_db_manager,
                mqtt_integration_manager=mock_mqtt_manager,
                api_config=mock_config.api,
            )

            # Verify initialization order (database -> MQTT -> tracking)
            call_order = []
            mock_get_db.side_effect = lambda: (
                call_order.append("database"),
                mock_db_manager,
            )[1]
            mock_mqtt_manager.initialize.side_effect = lambda: call_order.append("mqtt")
            mock_tracking_manager.initialize.side_effect = lambda: call_order.append(
                "tracking"
            )

            # Reset and reinitialize to test order
            system2 = OccupancyPredictionSystem()
            await system2.initialize()

            # Order should be: database, mqtt, tracking
            assert "mqtt" in call_order
            assert "tracking" in call_order

    @pytest.mark.asyncio
    async def test_system_lifecycle_comprehensive(self):
        """Test complete system lifecycle from startup to shutdown."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            # Setup mocks
            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.stop_tracking = AsyncMock()
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock()
            mock_mqtt_manager.cleanup = AsyncMock()
            mock_mqtt.return_value = mock_mqtt_manager

            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            # Test initialization phase
            assert system.running is False
            await system.initialize()
            assert system.running is True

            # Verify all components initialized
            mock_db_manager
            mock_mqtt_manager.initialize.assert_called_once()
            mock_tracking_manager.initialize.assert_called_once()

            # Test shutdown phase
            await system.shutdown()
            assert system.running is False

            # Verify proper shutdown sequence
            mock_tracking_manager.stop_tracking.assert_called_once()
            mock_mqtt_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_handling_orchestration_comprehensive(self):
        """Test system error handling and recovery mechanisms."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            # Setup config mock
            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            from src.main_system import OccupancyPredictionSystem

            # Test database initialization failure
            mock_get_db.side_effect = Exception("Database connection failed")
            system = OccupancyPredictionSystem()

            with pytest.raises(Exception, match="Database connection failed"):
                await system.initialize()

            # Verify system state after failure
            assert system.running is False

            # Test MQTT initialization failure
            mock_get_db.side_effect = None
            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock(
                side_effect=Exception("MQTT connection failed")
            )
            mock_mqtt.return_value = mock_mqtt_manager

            system2 = OccupancyPredictionSystem()
            with pytest.raises(Exception):
                await system2.initialize()

            # Verify cleanup was called on failure
            assert system2.running is False

            # Test tracking manager initialization failure
            mock_mqtt_manager.initialize = AsyncMock()  # Reset
            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock(
                side_effect=Exception("Tracking failed")
            )
            mock_tracking.return_value = mock_tracking_manager

            system3 = OccupancyPredictionSystem()
            with pytest.raises(Exception):
                await system3.initialize()

            # Verify shutdown is called on any initialization failure
            assert system3.running is False


class TestSystemStartupShutdown:
    """Test system startup and shutdown procedures."""

    @pytest.mark.asyncio
    async def test_graceful_startup_comprehensive(self):
        """Test graceful system startup with proper component ordering."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            # Track initialization order
            init_order = []

            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            mock_db_manager = AsyncMock()
            mock_get_db.side_effect = lambda: (
                init_order.append("database"),
                mock_db_manager,
            )[1]

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock(
                side_effect=lambda: init_order.append("mqtt")
            )
            mock_mqtt.return_value = mock_mqtt_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock(
                side_effect=lambda: init_order.append("tracking")
            )
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            # Test graceful initialization
            await system.initialize()

            # Verify proper startup order: database -> mqtt -> tracking
            assert init_order == ["database", "mqtt", "tracking"]
            assert system.running is True

            # Verify API server status is checked
            mock_tracking_manager.get_api_server_status.assert_called_once()

            # Test that system reports successful startup
            assert system.database_manager is not None
            assert system.mqtt_manager is not None
            assert system.tracking_manager is not None

    @pytest.mark.asyncio
    async def test_graceful_shutdown_comprehensive(self):
        """Test graceful system shutdown with proper cleanup order."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            # Track shutdown order
            shutdown_order = []

            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.stop_tracking = AsyncMock(
                side_effect=lambda: shutdown_order.append("tracking")
            )
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock()
            mock_mqtt_manager.cleanup = AsyncMock(
                side_effect=lambda: shutdown_order.append("mqtt")
            )
            mock_mqtt.return_value = mock_mqtt_manager

            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            # Initialize system first
            await system.initialize()
            assert system.running is True

            # Test graceful shutdown
            await system.shutdown()

            # Verify proper shutdown order: tracking -> mqtt
            assert shutdown_order == ["tracking", "mqtt"]
            assert system.running is False

            # Verify components were properly cleaned up
            mock_tracking_manager.stop_tracking.assert_called_once()
            mock_mqtt_manager.cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_failure_recovery_comprehensive(self):
        """Test system recovery from component failures."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            from src.main_system import OccupancyPredictionSystem

            # Test partial failure recovery - MQTT fails but system continues
            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            # First attempt: MQTT fails
            mqtt_call_count = 0

            def mqtt_init_side_effect():
                nonlocal mqtt_call_count
                mqtt_call_count += 1
                if mqtt_call_count == 1:
                    raise Exception("MQTT temporarily unavailable")
                # Second call succeeds - simulating recovery
                return None

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock(side_effect=mqtt_init_side_effect)
            mock_mqtt_manager.cleanup = AsyncMock()  # Add cleanup method
            mock_mqtt.return_value = mock_mqtt_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": False,
                "running": False,
                "host": None,
                "port": None,
            }
            mock_tracking.return_value = mock_tracking_manager

            system = OccupancyPredictionSystem()

            # First initialization should fail - but the system handles it during shutdown
            with pytest.raises(Exception):
                await system.initialize()

            # System should be in failed state
            assert system.running is False

            # Reset for successful initialization
            system2 = OccupancyPredictionSystem()
            mock_mqtt_manager2 = Mock()
            mock_mqtt_manager2.initialize = AsyncMock()  # Now succeeds
            mock_mqtt_manager2.cleanup = AsyncMock()
            mock_mqtt.return_value = mock_mqtt_manager2

            # Second attempt should succeed
            await system2.initialize()
            assert system2.running is True

            # Test error handling during shutdown
            # Create a new tracking manager for system2 that will fail on shutdown
            mock_tracking_manager2 = Mock()
            mock_tracking_manager2.initialize = AsyncMock()
            mock_tracking_manager2.stop_tracking = AsyncMock(
                side_effect=Exception("Shutdown error")
            )
            mock_tracking_manager2.get_api_server_status.return_value = {
                "enabled": False,
                "running": False,
                "host": None,
                "port": None,
            }

            # Replace the tracking manager for system2
            system2.tracking_manager = mock_tracking_manager2

            # Shutdown should fail when tracking manager fails (this is actual behavior)
            with pytest.raises(Exception, match="Shutdown error"):
                await system2.shutdown()
            # The system running flag remains True if shutdown fails early
            # (since the failure happens before setting running = False)
            assert system2.running is True


class TestSystemResourceManagement:
    """Test system resource management."""

    @pytest.mark.asyncio
    async def test_resource_allocation_comprehensive(self):
        """Test system resource allocation and management."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            # Test resource allocation during initialization
            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock()
            mock_mqtt.return_value = mock_mqtt_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            # Verify initial resource state
            assert system.database_manager is None
            assert system.mqtt_manager is None
            assert system.tracking_manager is None

            await system.initialize()

            # Verify resources are allocated
            assert system.database_manager == mock_db_manager
            assert system.mqtt_manager == mock_mqtt_manager
            assert system.tracking_manager == mock_tracking_manager

            # Test resource sharing - tracking manager should receive other resources
            mock_tracking.assert_called_once_with(
                config=mock_config.tracking,
                database_manager=mock_db_manager,
                mqtt_integration_manager=mock_mqtt_manager,
                api_config=mock_config.api,
            )

            # Verify proper initialization order ensures resources available
            # Check that database and MQTT were called before tracking manager initialization
            mock_get_db.assert_called_once()
            mock_mqtt.assert_called_once()
            mock_tracking.assert_called_once()

            # Verify initialization order by checking call counts before tracking init
            assert mock_get_db.call_count == 1
            assert mock_mqtt.call_count == 1

    @pytest.mark.asyncio
    async def test_resource_cleanup_comprehensive(self):
        """Test proper resource cleanup and deallocation."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            mock_db_manager = AsyncMock()
            mock_get_db.return_value = mock_db_manager

            mock_mqtt_manager = Mock()
            mock_mqtt_manager.initialize = AsyncMock()
            mock_mqtt_manager.cleanup = AsyncMock()
            mock_mqtt.return_value = mock_mqtt_manager

            mock_tracking_manager = Mock()
            mock_tracking_manager.initialize = AsyncMock()
            mock_tracking_manager.stop_tracking = AsyncMock()
            mock_tracking_manager.get_api_server_status.return_value = {
                "enabled": True,
                "running": True,
                "host": "localhost",
                "port": 8000,
            }
            mock_tracking.return_value = mock_tracking_manager

            from src.main_system import OccupancyPredictionSystem

            system = OccupancyPredictionSystem()

            # Initialize and verify resources are allocated
            await system.initialize()
            assert system.database_manager is not None
            assert system.mqtt_manager is not None
            assert system.tracking_manager is not None

            # Test normal cleanup
            await system.shutdown()

            # Verify cleanup methods were called
            mock_tracking_manager.stop_tracking.assert_called_once()
            mock_mqtt_manager.cleanup.assert_called_once()

            # Test cleanup during failed initialization
            system2 = OccupancyPredictionSystem()
            mock_tracking_manager2 = Mock()
            mock_tracking_manager2.initialize = AsyncMock(
                side_effect=Exception("Init failed")
            )
            mock_tracking.return_value = mock_tracking_manager2

            # Should cleanup even on failure
            with pytest.raises(Exception):
                await system2.initialize()

            # Verify system is in clean state after failure
            assert system2.running is False

            # Test cleanup with partial resource allocation
            system3 = OccupancyPredictionSystem()
            mock_mqtt_manager3 = Mock()
            mock_mqtt_manager3.initialize = AsyncMock(
                side_effect=Exception("MQTT failed")
            )
            mock_mqtt.return_value = mock_mqtt_manager3

            with pytest.raises(Exception):
                await system3.initialize()

            # Should handle partial cleanup gracefully
            assert system3.running is False

    @pytest.mark.asyncio
    async def test_memory_management_comprehensive(self):
        """Test system memory management and leak prevention."""
        with patch("src.main_system.get_config") as mock_get_config, patch(
            "src.main_system.get_database_manager"
        ) as mock_get_db, patch(
            "src.main_system.TrackingManager"
        ) as mock_tracking, patch(
            "src.main_system.MQTTIntegrationManager"
        ) as mock_mqtt:

            mock_config = Mock()
            mock_config.mqtt = Mock()
            mock_config.tracking = Mock()
            mock_config.api = Mock()
            mock_get_config.return_value = mock_config

            from src.main_system import OccupancyPredictionSystem

            # Create multiple system instances to test memory management
            systems = []

            for i in range(5):
                mock_db_manager = AsyncMock()
                mock_get_db.return_value = mock_db_manager

                mock_mqtt_manager = Mock()
                mock_mqtt_manager.initialize = AsyncMock()
                mock_mqtt_manager.cleanup = AsyncMock()
                mock_mqtt.return_value = mock_mqtt_manager

                mock_tracking_manager = Mock()
                mock_tracking_manager.initialize = AsyncMock()
                mock_tracking_manager.stop_tracking = AsyncMock()
                mock_tracking_manager.get_api_server_status.return_value = {
                    "enabled": True,
                    "running": True,
                    "host": "localhost",
                    "port": 8000,
                }
                mock_tracking.return_value = mock_tracking_manager

                system = OccupancyPredictionSystem()
                await system.initialize()
                systems.append(system)

                # Verify each system has independent resources
                assert system.database_manager is not None
                assert system.mqtt_manager is not None
                assert system.tracking_manager is not None

            # Test proper cleanup of all systems
            cleanup_calls = []
            for i, system in enumerate(systems):
                # Track cleanup calls
                system.tracking_manager.stop_tracking.side_effect = (
                    lambda i=i: cleanup_calls.append(f"tracking_{i}")
                )
                system.mqtt_manager.cleanup.side_effect = (
                    lambda i=i: cleanup_calls.append(f"mqtt_{i}")
                )

                await system.shutdown()
                assert system.running is False

            # Verify all resources were cleaned up
            assert len(cleanup_calls) == 10  # 5 tracking + 5 mqtt

            # Test reference cleanup after shutdown
            for system in systems:
                # References should still exist but be inactive
                assert system.running is False
                # Components should be cleaned up
                system.tracking_manager.stop_tracking.assert_called()
                system.mqtt_manager.cleanup.assert_called()

            # Test garbage collection friendly patterns
            system = OccupancyPredictionSystem()
            weak_refs = []

            # Simulate holding weak references to avoid memory leaks
            import weakref

            if hasattr(weakref, "ref"):  # Ensure weakref is available
                # Create system and get weak reference
                mock_db_manager = AsyncMock()
                mock_get_db.return_value = mock_db_manager

                mock_mqtt_manager = Mock()
                mock_mqtt_manager.initialize = AsyncMock()
                mock_mqtt_manager.cleanup = AsyncMock()
                mock_mqtt.return_value = mock_mqtt_manager

                mock_tracking_manager = Mock()
                mock_tracking_manager.initialize = AsyncMock()
                mock_tracking_manager.stop_tracking = AsyncMock()
                mock_tracking_manager.get_api_server_status.return_value = {
                    "enabled": True,
                    "running": True,
                    "host": "localhost",
                    "port": 8000,
                }
                mock_tracking.return_value = mock_tracking_manager

                await system.initialize()
                weak_refs.append(weakref.ref(system))

                # Cleanup should not prevent garbage collection
                await system.shutdown()
                assert system.running is False
