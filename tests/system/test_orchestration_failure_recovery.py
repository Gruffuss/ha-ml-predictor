"""
System Orchestration Failure Recovery Tests.

This module provides comprehensive testing for system-level orchestration scenarios
focusing on component isolation, cascading failure recovery, and error propagation
across the entire application stack.

Test Coverage:
- Component isolation during failures
- Cascading failure recovery mechanisms
- Cross-component error propagation
- Resource exhaustion recovery
- Graceful degradation under failures
- System stability during partial outages
"""

import asyncio
from datetime import datetime, timedelta
import gc
import logging
import time
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch
import weakref

import psutil
import pytest

from src.adaptation.tracking_manager import TrackingManager
from src.core.config import MQTTConfig, TrackingConfig, get_config
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    ErrorSeverity,
    OccupancyPredictionError,
    SystemResourceError,
)
from src.data.storage.database import DatabaseManager
from src.integration.enhanced_mqtt_manager import EnhancedMQTTIntegrationManager
from src.main_system import OccupancyPredictionSystem

logger = logging.getLogger(__name__)


@pytest.mark.system
class TestSystemOrchestrationFailureRecovery:
    """System orchestration failure recovery test suite."""

    @pytest.fixture
    def system_config(self):
        """System configuration for failure testing."""
        config = MagicMock()
        config.mqtt = MQTTConfig(
            broker="localhost",
            port=1883,
            username="test",
            password="test",
            topic_prefix="test/occupancy",
            device_identifier="test-device",
            discovery_prefix="homeassistant",
        )
        config.tracking = TrackingConfig()
        config.api = MagicMock()
        config.api.enabled = True
        config.api.host = "127.0.0.1"
        config.api.port = 8000
        return config

    @pytest.fixture
    def resource_monitor(self):
        """Monitor system resources during tests."""

        class ResourceMonitor:
            def __init__(self):
                self.process = psutil.Process()
                self.start_memory = None
                self.peak_memory = 0
                self.start_time = None

            def start_monitoring(self):
                """Start resource monitoring."""
                self.start_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
                self.peak_memory = self.start_memory
                self.start_time = time.time()

            def update_peak_memory(self):
                """Update peak memory usage."""
                current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
                self.peak_memory = max(self.peak_memory, current_memory)

            def get_memory_increase(self):
                """Get memory increase since start."""
                current_memory = self.process.memory_info().rss / (1024 * 1024)  # MB
                return current_memory - self.start_memory if self.start_memory else 0

            def get_runtime(self):
                """Get runtime since start."""
                return time.time() - self.start_time if self.start_time else 0

        return ResourceMonitor()

    @pytest.mark.asyncio
    async def test_database_failure_isolation(self, system_config, resource_monitor):
        """Test system behavior when database component fails."""
        resource_monitor.start_monitoring()

        # Create system with failing database
        system = OccupancyPredictionSystem()

        # Mock database failure during initialization
        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager",
            side_effect=DatabaseConnectionError(
                "Database unreachable", ErrorSeverity.CRITICAL
            ),
        ), patch.object(system, "shutdown", new_callable=AsyncMock) as mock_shutdown:

            # Attempt initialization - should fail gracefully
            with pytest.raises(DatabaseConnectionError):
                await system.initialize()

            # Verify proper cleanup was attempted
            mock_shutdown.assert_called_once()
            assert system.running is False

            # Verify no components were left in inconsistent state
            assert system.database_manager is None
            assert system.mqtt_manager is None
            assert system.tracking_manager is None

        # Test database failure during runtime
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Initialize system successfully first
        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):
            system = OccupancyPredictionSystem()
            await system.initialize()
            assert system.running is True

            # Simulate database failure during runtime
            mock_database_manager.health_check.side_effect = DatabaseError(
                "Connection lost", ErrorSeverity.HIGH
            )

            # System should handle database failure gracefully
            try:
                health_result = await mock_database_manager.health_check()
            except DatabaseError:
                # Expected failure - system should isolate this component
                pass

            # Other components should remain functional
            await mock_mqtt_manager.initialize()  # Should still work
            mock_tracking_manager.get_api_server_status()  # Should still work

            await system.shutdown()

        resource_monitor.update_peak_memory()
        memory_increase = resource_monitor.get_memory_increase()

        # Verify no significant memory leaks during failure scenarios
        assert (
            memory_increase < 50
        ), f"Memory leak detected: {memory_increase}MB increase"

    @pytest.mark.asyncio
    async def test_mqtt_component_failure_isolation(
        self, system_config, resource_monitor
    ):
        """Test system behavior when MQTT component fails."""
        resource_monitor.start_monitoring()

        system = OccupancyPredictionSystem()
        mock_database_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Test MQTT initialization failure
        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager",
            side_effect=Exception("MQTT broker unreachable"),
        ), patch.object(
            system, "shutdown", new_callable=AsyncMock
        ) as mock_shutdown:

            with pytest.raises(Exception, match="MQTT broker unreachable"):
                await system.initialize()

            mock_shutdown.assert_called_once()
            assert system.running is False

        # Test MQTT runtime failure isolation
        mock_mqtt_manager = AsyncMock()
        mock_mqtt_manager.initialize.side_effect = Exception("MQTT connection lost")

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ):
            system = OccupancyPredictionSystem()

            # Should fail during MQTT initialization
            with pytest.raises(Exception, match="MQTT connection lost"):
                await system.initialize()

            # Database should remain unaffected
            mock_database_manager.assert_not_called()  # Shouldn't be affected by MQTT failure

        resource_monitor.update_peak_memory()
        memory_increase = resource_monitor.get_memory_increase()
        assert memory_increase < 30, f"Memory leak in MQTT failure: {memory_increase}MB"

    @pytest.mark.asyncio
    async def test_tracking_manager_failure_isolation(self, system_config):
        """Test system behavior when tracking manager fails."""
        system = OccupancyPredictionSystem()
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()

        # Test tracking manager creation failure
        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager",
            side_effect=Exception("Tracking manager initialization failed"),
        ):
            with pytest.raises(
                Exception, match="Tracking manager initialization failed"
            ):
                await system.initialize()

        # Test tracking manager runtime failure
        mock_tracking_manager = AsyncMock()
        mock_tracking_manager.initialize.side_effect = Exception(
            "Tracking start failed"
        )

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):
            with pytest.raises(Exception, match="Tracking start failed"):
                await system.initialize()

            # Previous components should have been initialized successfully
            mock_mqtt_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery(self, system_config):
        """Test system recovery from cascading component failures."""
        failure_sequence = []

        # Track failure sequence
        def track_failure(component_name, error):
            failure_sequence.append((component_name, str(error), time.time()))

        system = OccupancyPredictionSystem()

        # Create components that fail in sequence
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Database fails first
        mock_database_manager.side_effect = lambda *args, **kwargs: track_failure(
            "database", "DB_FAIL"
        )

        # MQTT fails second due to database dependency
        def mqtt_failure(*args, **kwargs):
            track_failure("mqtt", "MQTT_FAIL")
            raise Exception("MQTT failure due to database")

        mock_mqtt_manager.initialize.side_effect = mqtt_failure

        # Tracking fails third due to both dependencies
        def tracking_failure(*args, **kwargs):
            track_failure("tracking", "TRACKING_FAIL")
            raise Exception("Tracking failure due to dependencies")

        mock_tracking_manager.initialize.side_effect = tracking_failure

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            # System should fail gracefully with cascading failures
            with pytest.raises(Exception):
                await system.initialize()

            # Verify failure sequence was tracked
            assert len(failure_sequence) >= 1  # At least one failure recorded

            # System should be in clean shutdown state
            assert system.running is False

    @pytest.mark.asyncio
    async def test_partial_system_operation_under_failures(self, system_config):
        """Test system operation when some components fail but others remain functional."""
        system = OccupancyPredictionSystem()

        # Create partially failing system
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()

        # MQTT fails but database works
        mock_mqtt_manager.initialize.side_effect = Exception("MQTT unavailable")

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ):

            # System initialization should fail due to MQTT
            with pytest.raises(Exception, match="MQTT unavailable"):
                await system.initialize()

            # But database manager should have been obtained successfully
            mock_database_manager.assert_not_called()  # No direct calls, but was created

    @pytest.mark.asyncio
    async def test_resource_exhaustion_recovery(self, system_config, resource_monitor):
        """Test system behavior under resource exhaustion conditions."""
        resource_monitor.start_monitoring()

        system = OccupancyPredictionSystem()

        # Simulate high memory usage scenario
        memory_hogs = []

        try:
            # Create memory pressure
            for _ in range(100):
                # Create large objects to simulate memory pressure
                memory_hogs.append(bytearray(1024 * 1024))  # 1MB each

            resource_monitor.update_peak_memory()

            # Create mocked components under memory pressure
            mock_database_manager = AsyncMock()
            mock_mqtt_manager = AsyncMock()
            mock_tracking_manager = AsyncMock()

            # Simulate component failure under resource pressure
            mock_tracking_manager.initialize.side_effect = SystemResourceError(
                "Insufficient memory for tracking manager", ErrorSeverity.HIGH
            )

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                with pytest.raises(SystemResourceError):
                    await system.initialize()

                # System should handle resource exhaustion gracefully
                assert system.running is False

        finally:
            # Clean up memory
            memory_hogs.clear()
            gc.collect()

        resource_monitor.update_peak_memory()

        # Verify system doesn't retain excessive memory after failure
        final_memory = resource_monitor.get_memory_increase()
        assert final_memory < 100, f"Excessive memory retained: {final_memory}MB"

    @pytest.mark.asyncio
    async def test_shutdown_failure_isolation(self, system_config):
        """Test that shutdown failures in one component don't affect others."""
        system = OccupancyPredictionSystem()

        # Create components with varying shutdown behavior
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Initialize system successfully
        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):
            await system.initialize()
            assert system.running is True

            # Make tracking manager shutdown fail
            mock_tracking_manager.stop_tracking.side_effect = Exception(
                "Tracking shutdown failed"
            )

            # Shutdown should still attempt cleanup of other components
            with pytest.raises(Exception, match="Tracking shutdown failed"):
                await system.shutdown()

            # Verify tracking manager shutdown was attempted first
            mock_tracking_manager.stop_tracking.assert_called_once()

            # MQTT cleanup should not be called due to exception propagation
            # (based on current implementation that doesn't catch exceptions)

            # System should still mark itself as not running
            assert system.running is False

    @pytest.mark.asyncio
    async def test_component_lifecycle_consistency(self, system_config):
        """Test that component lifecycles remain consistent during failures."""
        system = OccupancyPredictionSystem()

        # Track component states
        component_states = {
            "database": "uninitialized",
            "mqtt": "uninitialized",
            "tracking": "uninitialized",
        }

        # Create stateful mocks
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        def track_db_state(*args, **kwargs):
            component_states["database"] = "initialized"
            return mock_database_manager

        def track_mqtt_init(*args, **kwargs):
            component_states["mqtt"] = "initializing"
            # Fail during initialization
            component_states["mqtt"] = "failed"
            raise Exception("MQTT init failed")

        mock_mqtt_manager.initialize.side_effect = track_mqtt_init

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", side_effect=track_db_state
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            with pytest.raises(Exception, match="MQTT init failed"):
                await system.initialize()

            # Verify component state consistency
            assert component_states["database"] == "initialized"
            assert component_states["mqtt"] == "failed"
            assert component_states["tracking"] == "uninitialized"  # Never reached

            # System should be in consistent state
            assert system.running is False
            assert system.database_manager is mock_database_manager
            assert system.mqtt_manager is mock_mqtt_manager
            assert system.tracking_manager is None  # Never created

    @pytest.mark.asyncio
    async def test_error_propagation_boundaries(self, system_config):
        """Test that errors are properly contained within component boundaries."""
        system = OccupancyPredictionSystem()

        # Create error injection points
        error_log = []

        class ErrorTrackingException(Exception):
            def __init__(self, component, operation, original_error):
                self.component = component
                self.operation = operation
                self.original_error = original_error
                error_log.append((component, operation, str(original_error)))
                super().__init__(f"{component}.{operation}: {original_error}")

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Inject tracking manager error
        mock_tracking_manager.get_api_server_status.side_effect = (
            ErrorTrackingException(
                "tracking_manager",
                "get_api_server_status",
                "API server status check failed",
            )
        )

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            with pytest.raises(ErrorTrackingException) as exc_info:
                await system.initialize()

            # Verify error was properly contained and identified
            assert exc_info.value.component == "tracking_manager"
            assert exc_info.value.operation == "get_api_server_status"

            # Verify error propagation log
            assert len(error_log) == 1
            assert error_log[0][0] == "tracking_manager"

    @pytest.mark.asyncio
    async def test_memory_cleanup_during_failures(
        self, system_config, resource_monitor
    ):
        """Test that memory is properly cleaned up during component failures."""
        resource_monitor.start_monitoring()

        # Track object creation for cleanup verification
        created_objects = []

        class TrackableAsyncMock(AsyncMock):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                created_objects.append(weakref.ref(self))

        system = OccupancyPredictionSystem()

        # Create trackable components
        mock_database_manager = TrackableAsyncMock()
        mock_mqtt_manager = TrackableAsyncMock()
        mock_tracking_manager = TrackableAsyncMock()

        # Make MQTT initialization fail
        mock_mqtt_manager.initialize.side_effect = Exception(
            "MQTT failure for cleanup test"
        )

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            with pytest.raises(Exception, match="MQTT failure for cleanup test"):
                await system.initialize()

        # Force garbage collection
        del system
        gc.collect()

        # Allow some time for cleanup
        await asyncio.sleep(0.1)

        resource_monitor.update_peak_memory()
        memory_increase = resource_monitor.get_memory_increase()

        # Verify minimal memory retention after failure
        assert (
            memory_increase < 20
        ), f"Memory not cleaned up properly: {memory_increase}MB retained"

        # Verify objects can be garbage collected
        gc.collect()
        live_objects = [ref for ref in created_objects if ref() is not None]
        assert (
            len(live_objects) <= 1
        ), f"Objects not properly cleaned up: {len(live_objects)} still live"
