"""
System Lifecycle Management Tests.

This module provides comprehensive testing for system lifecycle management
including startup sequence validation, graceful shutdown procedures,
component dependency ordering, and state consistency during transitions.

Test Coverage:
- Startup sequence validation and dependency ordering
- Graceful shutdown procedures and cleanup verification
- Component state consistency during lifecycle transitions
- Dependency injection and configuration propagation
- Resource initialization and cleanup timing
- Service discovery and registration lifecycle
- Health check integration during startup/shutdown
- Component restart and recovery procedures
"""

import asyncio
from collections import OrderedDict
from datetime import datetime, timedelta
import logging
import time
from typing import Any, Dict, List, Optional, Set
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

import pytest

from src.adaptation.tracking_manager import TrackingManager
from src.core.config import MQTTConfig, TrackingConfig, get_config
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseError,
    ErrorSeverity,
    OccupancyPredictionError,
)
from src.data.storage.database import DatabaseManager
from src.integration.enhanced_mqtt_manager import EnhancedMQTTIntegrationManager
from src.main_system import OccupancyPredictionSystem

logger = logging.getLogger(__name__)


class LifecycleTracker:
    """Track component lifecycle events during testing."""

    def __init__(self):
        self.events = []
        self.component_states = {}
        self.dependency_violations = []

    def record_event(self, component: str, event: str, timestamp: float = None):
        """Record a lifecycle event."""
        if timestamp is None:
            timestamp = time.time()
        self.events.append(
            {"component": component, "event": event, "timestamp": timestamp}
        )

    def set_component_state(self, component: str, state: str):
        """Set component state."""
        self.component_states[component] = state
        self.record_event(component, f"state_changed_to_{state}")

    def check_dependency_order(self, component: str, dependencies: List[str]):
        """Check if component dependencies were satisfied before initialization."""
        component_events = [e for e in self.events if e["component"] == component]
        component_init_time = None

        for event in component_events:
            if "initialized" in event["event"]:
                component_init_time = event["timestamp"]
                break

        if component_init_time is None:
            return  # Component not initialized yet

        for dependency in dependencies:
            dep_events = [e for e in self.events if e["component"] == dependency]
            dep_init_time = None

            for event in dep_events:
                if "initialized" in event["event"]:
                    dep_init_time = event["timestamp"]
                    break

            if dep_init_time is None or dep_init_time > component_init_time:
                self.dependency_violations.append(
                    {
                        "component": component,
                        "dependency": dependency,
                        "violation": f"{dependency} not initialized before {component}",
                    }
                )

    def get_initialization_order(self):
        """Get the order in which components were initialized."""
        init_events = [e for e in self.events if "initialized" in e["event"]]
        return [
            e["component"] for e in sorted(init_events, key=lambda x: x["timestamp"])
        ]

    def get_shutdown_order(self):
        """Get the order in which components were shut down."""
        shutdown_events = [
            e
            for e in self.events
            if "shutdown" in e["event"] or "stopped" in e["event"]
        ]
        return [
            e["component"]
            for e in sorted(shutdown_events, key=lambda x: x["timestamp"])
        ]

    def get_component_lifecycle_duration(self, component: str):
        """Get the duration a component was active."""
        events = [e for e in self.events if e["component"] == component]
        init_time = None
        shutdown_time = None

        for event in events:
            if "initialized" in event["event"] and init_time is None:
                init_time = event["timestamp"]
            elif (
                "shutdown" in event["event"] or "stopped" in event["event"]
            ) and shutdown_time is None:
                shutdown_time = event["timestamp"]

        if init_time and shutdown_time:
            return shutdown_time - init_time
        return None


@pytest.mark.system
class TestSystemLifecycleManagement:
    """System lifecycle management test suite."""

    @pytest.fixture
    def system_config(self):
        """System configuration for lifecycle testing."""
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
    def lifecycle_tracker(self):
        """Lifecycle event tracking fixture."""
        return LifecycleTracker()

    @pytest.mark.asyncio
    async def test_startup_sequence_dependency_ordering(
        self, system_config, lifecycle_tracker
    ):
        """Test that components are initialized in correct dependency order."""

        # Create trackable mocks
        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Add tracking to database manager
        original_db_func = mock_database_manager

        async def tracked_db_manager(*args, **kwargs):
            lifecycle_tracker.record_event("database", "initialization_started")
            await asyncio.sleep(0.1)  # Simulate initialization time
            lifecycle_tracker.record_event("database", "initialized")
            return original_db_func

        # Add tracking to MQTT manager
        async def tracked_mqtt_init(*args, **kwargs):
            lifecycle_tracker.record_event("mqtt", "initialization_started")
            await asyncio.sleep(0.1)
            lifecycle_tracker.record_event("mqtt", "initialized")

        mock_mqtt_manager.initialize.side_effect = tracked_mqtt_init

        # Add tracking to tracking manager
        async def tracked_tracking_init(*args, **kwargs):
            lifecycle_tracker.record_event("tracking", "initialization_started")
            await asyncio.sleep(0.1)
            lifecycle_tracker.record_event("tracking", "initialized")

        mock_tracking_manager.initialize.side_effect = tracked_tracking_init
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", side_effect=tracked_db_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            assert system.running is True

            # Verify initialization order
            init_order = lifecycle_tracker.get_initialization_order()

            # Database should be first (no dependencies)
            # MQTT should be second (depends on database)
            # Tracking should be third (depends on both database and MQTT)
            expected_order = ["database", "mqtt", "tracking"]

            # Allow some flexibility in ordering, but check key dependencies
            assert "database" in init_order
            assert "mqtt" in init_order
            assert "tracking" in init_order

            # Check specific dependency violations
            lifecycle_tracker.check_dependency_order("mqtt", ["database"])
            lifecycle_tracker.check_dependency_order("tracking", ["database", "mqtt"])

            assert (
                len(lifecycle_tracker.dependency_violations) == 0
            ), f"Dependency violations found: {lifecycle_tracker.dependency_violations}"

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_graceful_shutdown_sequence(self, system_config, lifecycle_tracker):
        """Test that components are shut down in correct reverse dependency order."""

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Add tracking to shutdown methods
        async def tracked_tracking_shutdown(*args, **kwargs):
            lifecycle_tracker.record_event("tracking", "shutdown_started")
            await asyncio.sleep(0.1)
            lifecycle_tracker.record_event("tracking", "shutdown_completed")

        async def tracked_mqtt_cleanup(*args, **kwargs):
            lifecycle_tracker.record_event("mqtt", "cleanup_started")
            await asyncio.sleep(0.1)
            lifecycle_tracker.record_event("mqtt", "cleanup_completed")

        mock_tracking_manager.stop_tracking.side_effect = tracked_tracking_shutdown
        mock_mqtt_manager.cleanup.side_effect = tracked_mqtt_cleanup
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()
            assert system.running is True

            # Record database manager as initialized
            lifecycle_tracker.record_event("database", "initialized")

            await system.shutdown()
            assert system.running is False

            # Verify shutdown order (reverse of initialization)
            shutdown_events = [
                e
                for e in lifecycle_tracker.events
                if "shutdown" in e["event"] or "cleanup" in e["event"]
            ]
            shutdown_order = [
                e["component"] for e in shutdown_events if "started" in e["event"]
            ]

            # Tracking should shutdown first, then MQTT
            # Database doesn't have explicit shutdown in current implementation
            expected_shutdown_order = ["tracking", "mqtt"]

            assert (
                shutdown_order == expected_shutdown_order
            ), f"Incorrect shutdown order: {shutdown_order}, expected: {expected_shutdown_order}"

    @pytest.mark.asyncio
    async def test_component_state_consistency_during_transitions(
        self, system_config, lifecycle_tracker
    ):
        """Test that component states remain consistent during lifecycle transitions."""

        system = OccupancyPredictionSystem()

        # Track system state transitions
        original_running = system.running
        lifecycle_tracker.set_component_state("system", "created")

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Add state consistency checks
        async def checked_mqtt_init(*args, **kwargs):
            # MQTT initialization should only happen when system is initializing
            if not hasattr(system, "_initializing"):
                lifecycle_tracker.dependency_violations.append(
                    {
                        "component": "mqtt",
                        "violation": "MQTT initialized outside of system initialization",
                    }
                )
            lifecycle_tracker.set_component_state("mqtt", "initialized")

        async def checked_tracking_init(*args, **kwargs):
            # Tracking should only initialize after MQTT
            if lifecycle_tracker.component_states.get("mqtt") != "initialized":
                lifecycle_tracker.dependency_violations.append(
                    {
                        "component": "tracking",
                        "violation": "Tracking initialized before MQTT",
                    }
                )
            lifecycle_tracker.set_component_state("tracking", "initialized")

        mock_mqtt_manager.initialize.side_effect = checked_mqtt_init
        mock_tracking_manager.initialize.side_effect = checked_tracking_init
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            # Mark system as initializing
            system._initializing = True
            lifecycle_tracker.set_component_state("system", "initializing")

            await system.initialize()

            lifecycle_tracker.set_component_state("system", "running")
            lifecycle_tracker.set_component_state("database", "initialized")

            # Remove initializing flag
            delattr(system, "_initializing")

            assert system.running is True

            # Verify no state consistency violations
            assert (
                len(lifecycle_tracker.dependency_violations) == 0
            ), f"State consistency violations: {lifecycle_tracker.dependency_violations}"

            lifecycle_tracker.set_component_state("system", "shutting_down")
            await system.shutdown()
            lifecycle_tracker.set_component_state("system", "stopped")

            assert system.running is False

    @pytest.mark.asyncio
    async def test_configuration_propagation_during_startup(
        self, system_config, lifecycle_tracker
    ):
        """Test that configuration is properly propagated to all components during startup."""

        # Track configuration passed to each component
        config_received = {}

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Track MQTT manager configuration
        def track_mqtt_config(mqtt_config):
            config_received["mqtt"] = mqtt_config
            lifecycle_tracker.record_event("mqtt", "config_received")
            return mock_mqtt_manager

        # Track tracking manager configuration
        def track_tracking_config(
            config=None,
            database_manager=None,
            mqtt_integration_manager=None,
            api_config=None,
        ):
            config_received["tracking"] = {
                "config": config,
                "database_manager": database_manager,
                "mqtt_integration_manager": mqtt_integration_manager,
                "api_config": api_config,
            }
            lifecycle_tracker.record_event("tracking", "config_received")
            return mock_tracking_manager

        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", side_effect=track_mqtt_config
        ), patch(
            "src.main_system.TrackingManager", side_effect=track_tracking_config
        ):

            await system.initialize()

            # Verify all components received configuration
            assert (
                "mqtt" in config_received
            ), "MQTT manager did not receive configuration"
            assert (
                "tracking" in config_received
            ), "Tracking manager did not receive configuration"

            # Verify correct configuration was passed
            assert (
                config_received["mqtt"] is system_config.mqtt
            ), "MQTT received incorrect config"

            tracking_config = config_received["tracking"]
            assert (
                tracking_config["config"] is system_config.tracking
            ), "Tracking received incorrect config"
            assert (
                tracking_config["database_manager"] is mock_database_manager
            ), "Tracking received incorrect database manager"
            assert (
                tracking_config["mqtt_integration_manager"] is mock_mqtt_manager
            ), "Tracking received incorrect MQTT manager"
            assert (
                tracking_config["api_config"] is system_config.api
            ), "Tracking received incorrect API config"

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_resource_initialization_timing(
        self, system_config, lifecycle_tracker
    ):
        """Test that resources are initialized and cleaned up at appropriate times."""

        resource_timeline = []

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Track resource lifecycle
        async def track_resource_init(component, init_func):
            """Track resource initialization timing."""
            start_time = time.time()
            resource_timeline.append((component, "init_start", start_time))

            try:
                result = await init_func()
                end_time = time.time()
                resource_timeline.append((component, "init_success", end_time))
                return result
            except Exception as e:
                end_time = time.time()
                resource_timeline.append((component, "init_failed", end_time))
                raise

        async def tracked_mqtt_init(*args, **kwargs):
            return await track_resource_init("mqtt", lambda: asyncio.sleep(0.1))

        async def tracked_tracking_init(*args, **kwargs):
            return await track_resource_init("tracking", lambda: asyncio.sleep(0.1))

        mock_mqtt_manager.initialize.side_effect = tracked_mqtt_init
        mock_tracking_manager.initialize.side_effect = tracked_tracking_init
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            init_start = time.time()
            await system.initialize()
            init_end = time.time()

            # Verify all resources were initialized successfully
            mqtt_events = [event for event in resource_timeline if event[0] == "mqtt"]
            tracking_events = [
                event for event in resource_timeline if event[0] == "tracking"
            ]

            assert any(
                "init_success" in event[1] for event in mqtt_events
            ), "MQTT initialization not completed"
            assert any(
                "init_success" in event[1] for event in tracking_events
            ), "Tracking initialization not completed"

            # Verify timing constraints
            total_init_time = init_end - init_start
            assert (
                total_init_time < 5.0
            ), f"System initialization took too long: {total_init_time}s"

            # Verify no resource initialization failures
            failed_events = [
                event for event in resource_timeline if "failed" in event[1]
            ]
            assert (
                len(failed_events) == 0
            ), f"Resource initialization failures: {failed_events}"

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_component_restart_procedures(self, system_config, lifecycle_tracker):
        """Test component restart procedures and recovery."""

        restart_count = {"mqtt": 0, "tracking": 0}

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Simulate component that fails first time, succeeds second time
        async def flaky_mqtt_init(*args, **kwargs):
            restart_count["mqtt"] += 1
            lifecycle_tracker.record_event(
                "mqtt", f"init_attempt_{restart_count['mqtt']}"
            )
            if restart_count["mqtt"] == 1:
                raise Exception("MQTT initialization failed")
            # Success on retry
            lifecycle_tracker.record_event("mqtt", "init_success")

        mock_mqtt_manager.initialize.side_effect = flaky_mqtt_init
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            # First initialization attempt should fail
            with pytest.raises(Exception, match="MQTT initialization failed"):
                await system.initialize()

            assert system.running is False
            assert restart_count["mqtt"] == 1

            # Second attempt should succeed (simulating restart)
            system = OccupancyPredictionSystem()
            await system.initialize()

            assert system.running is True
            assert restart_count["mqtt"] == 2

            # Verify successful restart was recorded
            mqtt_events = [
                e for e in lifecycle_tracker.events if e["component"] == "mqtt"
            ]
            success_events = [e for e in mqtt_events if "success" in e["event"]]
            assert len(success_events) == 1, "MQTT restart success not recorded"

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_health_check_integration_during_lifecycle(
        self, system_config, lifecycle_tracker
    ):
        """Test health check integration during startup and shutdown."""

        health_checks = []

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Mock database health check
        async def database_health_check(*args, **kwargs):
            health_result = {"status": "healthy", "timestamp": time.time()}
            health_checks.append(("database", health_result))
            lifecycle_tracker.record_event("database", "health_check_performed")
            return health_result

        mock_database_manager.health_check = database_health_check
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            await system.initialize()

            # Perform health checks after initialization
            if hasattr(system.database_manager, "health_check"):
                await system.database_manager.health_check()

            # Verify health checks were performed
            db_health_checks = [hc for hc in health_checks if hc[0] == "database"]
            assert len(db_health_checks) >= 1, "Database health check not performed"

            # Verify health check results
            for component, result in health_checks:
                assert (
                    result["status"] == "healthy"
                ), f"{component} health check failed: {result}"

            await system.shutdown()

            # Verify health check events were recorded
            health_events = [
                e for e in lifecycle_tracker.events if "health_check" in e["event"]
            ]
            assert len(health_events) >= 1, "Health check events not recorded"

    @pytest.mark.asyncio
    async def test_component_dependency_injection_validation(
        self, system_config, lifecycle_tracker
    ):
        """Test that component dependency injection is properly validated."""

        injection_log = []

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Track dependency injection
        def track_tracking_creation(
            config=None,
            database_manager=None,
            mqtt_integration_manager=None,
            api_config=None,
        ):
            injection_log.append(
                {
                    "component": "tracking_manager",
                    "dependencies": {
                        "config": config is not None,
                        "database_manager": database_manager is not None,
                        "mqtt_integration_manager": mqtt_integration_manager
                        is not None,
                        "api_config": api_config is not None,
                    },
                }
            )
            lifecycle_tracker.record_event("tracking", "dependencies_injected")
            return mock_tracking_manager

        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", side_effect=track_tracking_creation
        ):

            await system.initialize()

            # Verify dependency injection occurred
            assert len(injection_log) == 1, "Dependency injection not recorded"

            injected_deps = injection_log[0]["dependencies"]

            # Verify all required dependencies were injected
            assert injected_deps["config"], "Config not injected into tracking manager"
            assert injected_deps[
                "database_manager"
            ], "Database manager not injected into tracking manager"
            assert injected_deps[
                "mqtt_integration_manager"
            ], "MQTT manager not injected into tracking manager"
            assert injected_deps[
                "api_config"
            ], "API config not injected into tracking manager"

            # Verify injection event was recorded
            injection_events = [
                e
                for e in lifecycle_tracker.events
                if "dependencies_injected" in e["event"]
            ]
            assert len(injection_events) == 1, "Dependency injection event not recorded"

            await system.shutdown()

    @pytest.mark.asyncio
    async def test_lifecycle_performance_benchmarks(
        self, system_config, lifecycle_tracker
    ):
        """Test lifecycle performance and establish benchmarks."""

        performance_metrics = {}

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Add realistic delays to component initialization
        async def timed_mqtt_init(*args, **kwargs):
            await asyncio.sleep(0.05)  # 50ms initialization time

        async def timed_tracking_init(*args, **kwargs):
            await asyncio.sleep(0.1)  # 100ms initialization time

        mock_mqtt_manager.initialize.side_effect = timed_mqtt_init
        mock_tracking_manager.initialize.side_effect = timed_tracking_init
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        system = OccupancyPredictionSystem()

        with patch("src.main_system.get_config", return_value=system_config), patch(
            "src.main_system.get_database_manager", return_value=mock_database_manager
        ), patch(
            "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
        ), patch(
            "src.main_system.TrackingManager", return_value=mock_tracking_manager
        ):

            # Measure initialization performance
            start_time = time.time()
            await system.initialize()
            init_duration = time.time() - start_time
            performance_metrics["initialization_duration"] = init_duration

            # Measure shutdown performance
            start_time = time.time()
            await system.shutdown()
            shutdown_duration = time.time() - start_time
            performance_metrics["shutdown_duration"] = shutdown_duration

        # Establish performance benchmarks
        assert (
            performance_metrics["initialization_duration"] < 1.0
        ), f"Initialization too slow: {performance_metrics['initialization_duration']}s"
        assert (
            performance_metrics["shutdown_duration"] < 0.5
        ), f"Shutdown too slow: {performance_metrics['shutdown_duration']}s"

        # Log performance metrics for analysis
        logger.info(f"Lifecycle performance metrics: {performance_metrics}")

        # Record performance metrics in lifecycle tracker
        lifecycle_tracker.record_event(
            "system",
            f"init_duration_{performance_metrics['initialization_duration']:.3f}s",
        )
        lifecycle_tracker.record_event(
            "system",
            f"shutdown_duration_{performance_metrics['shutdown_duration']:.3f}s",
        )
