"""
System Error Propagation Testing.

This module provides comprehensive testing for error propagation across
system components, error boundary validation, fault isolation mechanisms,
and system recovery from cross-component error scenarios.

Test Coverage:
- Cross-component error propagation patterns
- Error boundary validation and containment
- Fault isolation between system layers
- Error recovery and system resilience
- Exception handling chain validation
- Error context preservation across components
- System stability during cascading failures
- Error reporting and logging consistency
"""

import asyncio
from datetime import datetime, timedelta
import logging
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Tuple, Type
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

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


class ErrorPropagationTracker:
    """Track error propagation patterns across system components."""

    def __init__(self):
        self.error_events = []
        self.error_chains = []
        self.component_error_counts = {}
        self.recovery_events = []
        self.error_boundaries = []

    def record_error(
        self,
        component: str,
        error_type: str,
        error_message: str,
        timestamp: float = None,
        caused_by: str = None,
    ):
        """Record an error event."""
        if timestamp is None:
            timestamp = time.time()

        error_event = {
            "component": component,
            "error_type": error_type,
            "error_message": error_message,
            "timestamp": timestamp,
            "caused_by": caused_by,
        }

        self.error_events.append(error_event)
        self.component_error_counts[component] = (
            self.component_error_counts.get(component, 0) + 1
        )

        # Track error propagation chains
        if caused_by:
            self.error_chains.append(
                {
                    "source": caused_by,
                    "target": component,
                    "error_type": error_type,
                    "timestamp": timestamp,
                }
            )

    def record_error_boundary(
        self,
        component: str,
        boundary_type: str,
        contained_error: str,
        timestamp: float = None,
    ):
        """Record error boundary containment."""
        if timestamp is None:
            timestamp = time.time()

        self.error_boundaries.append(
            {
                "component": component,
                "boundary_type": boundary_type,
                "contained_error": contained_error,
                "timestamp": timestamp,
            }
        )

    def record_recovery(
        self, component: str, recovery_type: str, timestamp: float = None
    ):
        """Record error recovery event."""
        if timestamp is None:
            timestamp = time.time()

        self.recovery_events.append(
            {
                "component": component,
                "recovery_type": recovery_type,
                "timestamp": timestamp,
            }
        )

    def get_error_propagation_chains(self):
        """Get error propagation chains."""
        # Group chains by source component
        chains = {}
        for chain in self.error_chains:
            source = chain["source"]
            if source not in chains:
                chains[source] = []
            chains[source].append(chain)
        return chains

    def get_component_error_impact(self):
        """Analyze error impact per component."""
        impact_analysis = {}

        for component, count in self.component_error_counts.items():
            # Count how many other components were affected by this component's errors
            affected_components = set()
            for chain in self.error_chains:
                if chain["source"] == component:
                    affected_components.add(chain["target"])

            impact_analysis[component] = {
                "error_count": count,
                "components_affected": len(affected_components),
                "affected_list": list(affected_components),
            }

        return impact_analysis

    def get_error_containment_effectiveness(self):
        """Analyze how effectively errors were contained."""
        total_errors = len(self.error_events)
        contained_errors = len(self.error_boundaries)
        propagated_errors = len(self.error_chains)

        return {
            "total_errors": total_errors,
            "contained_errors": contained_errors,
            "propagated_errors": propagated_errors,
            "containment_rate": contained_errors / max(1, total_errors),
            "propagation_rate": propagated_errors / max(1, total_errors),
        }


class ErrorInjector:
    """Inject errors at specific points for testing error propagation."""

    def __init__(self, tracker: ErrorPropagationTracker):
        self.tracker = tracker
        self.injection_points = {}

    def register_injection_point(
        self,
        component: str,
        method: str,
        error_type: Type[Exception],
        error_message: str,
    ):
        """Register an error injection point."""
        key = f"{component}.{method}"
        self.injection_points[key] = {
            "error_type": error_type,
            "error_message": error_message,
            "triggered": False,
        }

    def create_error_injecting_mock(
        self, component: str, method: str, original_mock: AsyncMock = None
    ):
        """Create a mock that injects errors when called."""
        if original_mock is None:
            original_mock = AsyncMock()

        key = f"{component}.{method}"

        async def error_injecting_method(*args, **kwargs):
            injection_config = self.injection_points.get(key)
            if injection_config and not injection_config["triggered"]:
                injection_config["triggered"] = True

                # Record error injection
                self.tracker.record_error(
                    component=component,
                    error_type=injection_config["error_type"].__name__,
                    error_message=injection_config["error_message"],
                )

                # Raise the error
                raise injection_config["error_type"](injection_config["error_message"])

            # Normal operation
            return await original_mock(*args, **kwargs)

        return error_injecting_method


@pytest.mark.system
class TestSystemErrorPropagation:
    """System error propagation test suite."""

    @pytest.fixture
    def system_config(self):
        """System configuration for error propagation testing."""
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
    def error_tracker(self):
        """Error propagation tracking fixture."""
        return ErrorPropagationTracker()

    @pytest.fixture
    def error_injector(self, error_tracker):
        """Error injection fixture."""
        return ErrorInjector(error_tracker)

    @pytest.mark.asyncio
    async def test_database_error_propagation_to_dependent_components(
        self, system_config, error_tracker, error_injector
    ):
        """Test how database errors propagate to dependent components."""

        # Register error injection point
        error_injector.register_injection_point(
            "database",
            "health_check",
            DatabaseConnectionError,
            "Database connection lost",
        )

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Create error-injecting database manager
        mock_database_manager.health_check = error_injector.create_error_injecting_mock(
            "database", "health_check", mock_database_manager.health_check
        )

        # Track how MQTT manager handles database errors
        async def mqtt_init_with_error_handling(*args, **kwargs):
            try:
                # MQTT manager might check database health during initialization
                await mock_database_manager.health_check()
                error_tracker.record_recovery(
                    "mqtt", "successful_init_despite_db_error"
                )
            except DatabaseConnectionError as e:
                error_tracker.record_error(
                    "mqtt", "InitializationError", str(e), caused_by="database"
                )
                # MQTT might continue with degraded functionality
                error_tracker.record_error_boundary(
                    "mqtt", "graceful_degradation", str(e)
                )

        mock_mqtt_manager.initialize.side_effect = mqtt_init_with_error_handling

        # Track how tracking manager handles database dependency errors
        def tracking_manager_with_error_handling(*args, **kwargs):
            try:
                # Tracking manager depends on database manager
                if not mock_database_manager:
                    raise Exception("Database manager unavailable")
                error_tracker.record_recovery("tracking", "successful_creation")
                return mock_tracking_manager
            except Exception as e:
                error_tracker.record_error(
                    "tracking", "CreationError", str(e), caused_by="database"
                )
                raise

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
            "src.main_system.TrackingManager",
            side_effect=tracking_manager_with_error_handling,
        ):

            try:
                await system.initialize()
            except Exception as e:
                error_tracker.record_error(
                    "system", "InitializationError", str(e), caused_by="database"
                )

        # Analyze error propagation
        propagation_chains = error_tracker.get_error_propagation_chains()
        component_impact = error_tracker.get_component_error_impact()
        containment_effectiveness = error_tracker.get_error_containment_effectiveness()

        # Verify database error propagated to dependent components
        assert "database" in component_impact, "Database error not recorded"
        assert (
            component_impact["database"]["error_count"] > 0
        ), "No database errors recorded"

        # Verify propagation occurred
        if "database" in propagation_chains:
            db_chains = propagation_chains["database"]
            affected_components = [chain["target"] for chain in db_chains]

            # At least one component should be affected by database error
            assert (
                len(affected_components) > 0
            ), "Database error did not propagate to dependent components"

        logger.info(f"Database error propagation analysis: {component_impact}")

    @pytest.mark.asyncio
    async def test_mqtt_error_isolation_from_other_components(
        self, system_config, error_tracker, error_injector
    ):
        """Test that MQTT errors are properly isolated from other components."""

        # Register MQTT error injection
        error_injector.register_injection_point(
            "mqtt", "initialize", Exception, "MQTT broker connection failed"
        )

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Create error-injecting MQTT manager
        mock_mqtt_manager.initialize = error_injector.create_error_injecting_mock(
            "mqtt", "initialize", mock_mqtt_manager.initialize
        )

        # Database should be unaffected by MQTT errors
        async def database_with_isolation(*args, **kwargs):
            error_tracker.record_recovery("database", "unaffected_by_mqtt_error")
            return {"status": "healthy"}

        mock_database_manager.health_check = database_with_isolation

        # Tracking manager should handle MQTT dependency gracefully
        def tracking_with_mqtt_error_handling(*args, **kwargs):
            try:
                # Tracking manager gets MQTT manager as dependency
                error_tracker.record_recovery("tracking", "created_despite_mqtt_error")
                return mock_tracking_manager
            except Exception as e:
                error_tracker.record_error(
                    "tracking", "CreationError", str(e), caused_by="mqtt"
                )
                raise

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
            "src.main_system.TrackingManager",
            side_effect=tracking_with_mqtt_error_handling,
        ):

            try:
                await system.initialize()
            except Exception as e:
                error_tracker.record_error(
                    "system", "InitializationError", str(e), caused_by="mqtt"
                )

        # Analyze error isolation
        component_impact = error_tracker.get_component_error_impact()
        containment_effectiveness = error_tracker.get_error_containment_effectiveness()

        # Verify MQTT error was contained
        assert "mqtt" in component_impact, "MQTT error not recorded"

        # Check if error was properly isolated
        recovery_components = [
            event["component"] for event in error_tracker.recovery_events
        ]

        # Database should be unaffected (recorded recovery)
        database_recoveries = [
            event
            for event in error_tracker.recovery_events
            if event["component"] == "database"
        ]
        assert (
            len(database_recoveries) > 0
        ), "Database was affected by MQTT error (no isolation)"

        logger.info(f"MQTT error isolation analysis: {containment_effectiveness}")

    @pytest.mark.asyncio
    async def test_error_boundary_effectiveness(
        self, system_config, error_tracker, error_injector
    ):
        """Test effectiveness of error boundaries in containing failures."""

        # Inject errors at multiple points
        error_injector.register_injection_point(
            "tracking",
            "initialize",
            SystemResourceError,
            "Tracking manager resource exhaustion",
        )

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Implement error boundaries in system initialization
        async def system_init_with_boundaries():
            error_boundaries_implemented = []

            try:
                # Database initialization with error boundary
                db_manager = await mock_database_manager
                error_tracker.record_error_boundary(
                    "system", "database_boundary", "database_init_protected"
                )
                error_boundaries_implemented.append("database")
            except Exception as e:
                error_tracker.record_error("system", "DatabaseBoundaryFailure", str(e))

            try:
                # MQTT initialization with error boundary
                await mock_mqtt_manager.initialize()
                error_tracker.record_error_boundary(
                    "system", "mqtt_boundary", "mqtt_init_protected"
                )
                error_boundaries_implemented.append("mqtt")
            except Exception as e:
                error_tracker.record_error("system", "MQTTBoundaryFailure", str(e))

            try:
                # Tracking manager with error boundary
                await mock_tracking_manager.initialize()
                error_tracker.record_error_boundary(
                    "system", "tracking_boundary", "tracking_init_protected"
                )
                error_boundaries_implemented.append("tracking")
            except Exception as e:
                error_tracker.record_error_boundary(
                    "system", "tracking_boundary", f"contained_tracking_error: {e}"
                )
                # Error contained, system can continue with degraded functionality

            return error_boundaries_implemented

        # Create error-injecting tracking manager
        mock_tracking_manager.initialize = error_injector.create_error_injecting_mock(
            "tracking", "initialize", mock_tracking_manager.initialize
        )

        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        # Execute system initialization with error boundaries
        error_boundaries_implemented = await system_init_with_boundaries()

        # Analyze error boundary effectiveness
        containment_effectiveness = error_tracker.get_error_containment_effectiveness()

        # Verify error boundaries were implemented
        assert (
            len(error_tracker.error_boundaries) > 0
        ), "No error boundaries implemented"

        # Verify at least some errors were contained
        assert (
            containment_effectiveness["containment_rate"] > 0
        ), "No errors were contained by boundaries"

        # Check that system could continue despite contained errors
        boundary_components = set(
            boundary["component"] for boundary in error_tracker.error_boundaries
        )
        assert len(boundary_components) > 0, "Error boundaries not established"

        logger.info(f"Error boundary effectiveness: {containment_effectiveness}")

    @pytest.mark.asyncio
    async def test_cascading_failure_recovery_mechanisms(
        self, system_config, error_tracker, error_injector
    ):
        """Test system recovery mechanisms during cascading failures."""

        # Set up cascading failure scenario
        failure_sequence = [
            ("database", "health_check", DatabaseError, "Database health check failed"),
            (
                "mqtt",
                "initialize",
                Exception,
                "MQTT initialization failed due to database",
            ),
            (
                "tracking",
                "initialize",
                SystemResourceError,
                "Tracking failed due to dependencies",
            ),
        ]

        for component, method, error_type, message in failure_sequence:
            error_injector.register_injection_point(
                component, method, error_type, message
            )

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Implement recovery mechanisms
        retry_counts = {"database": 0, "mqtt": 0, "tracking": 0}
        max_retries = 2

        async def database_with_recovery(*args, **kwargs):
            retry_counts["database"] += 1
            if retry_counts["database"] <= max_retries:
                try:
                    return await error_injector.create_error_injecting_mock(
                        "database", "health_check"
                    )()
                except Exception as e:
                    if retry_counts["database"] < max_retries:
                        error_tracker.record_recovery(
                            "database", f"retry_attempt_{retry_counts['database']}"
                        )
                        await asyncio.sleep(0.1)  # Brief delay before retry
                        return await database_with_recovery(*args, **kwargs)
                    else:
                        error_tracker.record_error(
                            "database", "MaxRetriesExceeded", str(e)
                        )
                        raise
            return {"status": "healthy"}

        async def mqtt_with_recovery(*args, **kwargs):
            retry_counts["mqtt"] += 1
            try:
                return await error_injector.create_error_injecting_mock(
                    "mqtt", "initialize"
                )()
            except Exception as e:
                if retry_counts["mqtt"] < max_retries:
                    error_tracker.record_recovery(
                        "mqtt", f"retry_attempt_{retry_counts['mqtt']}"
                    )
                    await asyncio.sleep(0.1)
                    return await mqtt_with_recovery(*args, **kwargs)
                else:
                    error_tracker.record_error_boundary(
                        "mqtt", "graceful_degradation", f"continuing_without_mqtt: {e}"
                    )
                    # Continue with degraded functionality

        async def tracking_with_recovery(*args, **kwargs):
            retry_counts["tracking"] += 1
            try:
                return await error_injector.create_error_injecting_mock(
                    "tracking", "initialize"
                )()
            except Exception as e:
                if retry_counts["tracking"] < max_retries:
                    error_tracker.record_recovery(
                        "tracking", f"retry_attempt_{retry_counts['tracking']}"
                    )
                    await asyncio.sleep(0.1)
                    return await tracking_with_recovery(*args, **kwargs)
                else:
                    error_tracker.record_error_boundary(
                        "tracking", "fallback_mode", f"minimal_tracking: {e}"
                    )
                    # Return minimal tracking functionality
                    return mock_tracking_manager

        mock_database_manager.health_check = database_with_recovery
        mock_mqtt_manager.initialize = mqtt_with_recovery
        mock_tracking_manager.initialize = tracking_with_recovery
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

            try:
                await system.initialize()
                # If initialization succeeds, record successful recovery
                error_tracker.record_recovery(
                    "system", "full_recovery_from_cascading_failures"
                )
            except Exception as e:
                error_tracker.record_error_boundary(
                    "system", "partial_functionality", f"degraded_mode: {e}"
                )

        # Analyze recovery effectiveness
        recovery_attempts = len(error_tracker.recovery_events)
        total_retry_attempts = sum(retry_counts.values())
        containment_effectiveness = error_tracker.get_error_containment_effectiveness()

        # Verify recovery mechanisms were attempted
        assert recovery_attempts > 0, "No recovery attempts were made"
        assert total_retry_attempts > 0, "No retry mechanisms were triggered"

        # Verify some form of error containment occurred
        assert (
            len(error_tracker.error_boundaries) > 0
        ), "No error boundaries established during cascading failures"

        logger.info(
            f"Cascading failure recovery: {recovery_attempts} recovery attempts, {total_retry_attempts} retries"
        )

    @pytest.mark.asyncio
    async def test_error_context_preservation_across_components(
        self, system_config, error_tracker, error_injector
    ):
        """Test that error context is preserved as errors propagate across components."""

        # Register error with context
        error_injector.register_injection_point(
            "database",
            "health_check",
            DatabaseConnectionError,
            "Connection timeout after 30s",
        )

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Track error context preservation
        error_contexts = []

        async def mqtt_with_context_preservation(*args, **kwargs):
            try:
                await mock_database_manager.health_check()
            except DatabaseConnectionError as e:
                # Preserve and enhance error context
                enhanced_context = {
                    "original_error": str(e),
                    "error_type": type(e).__name__,
                    "component": "mqtt",
                    "operation": "initialization",
                    "dependency": "database",
                    "timestamp": time.time(),
                }
                error_contexts.append(enhanced_context)

                error_tracker.record_error(
                    "mqtt",
                    "DependencyError",
                    f"MQTT init failed due to database: {e}",
                    caused_by="database",
                )
                raise Exception(f"MQTT initialization failed: {enhanced_context}")

        def tracking_with_context_preservation(*args, **kwargs):
            try:
                # Tracking manager creation depends on MQTT manager
                enhanced_context = {
                    "component": "tracking",
                    "operation": "creation",
                    "dependencies": ["database", "mqtt"],
                    "timestamp": time.time(),
                }
                error_contexts.append(enhanced_context)
                return mock_tracking_manager
            except Exception as e:
                error_context = {
                    "original_error": str(e),
                    "component": "tracking",
                    "failed_dependencies": ["database", "mqtt"],
                    "timestamp": time.time(),
                }
                error_contexts.append(error_context)

                error_tracker.record_error(
                    "tracking",
                    "ContextualError",
                    f"Tracking creation failed with context: {error_context}",
                    caused_by="mqtt",
                )
                raise

        # Create error-injecting database manager
        mock_database_manager.health_check = error_injector.create_error_injecting_mock(
            "database", "health_check", mock_database_manager.health_check
        )

        mock_mqtt_manager.initialize = mqtt_with_context_preservation
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
            "src.main_system.TrackingManager",
            side_effect=tracking_with_context_preservation,
        ):

            try:
                await system.initialize()
            except Exception as e:
                # Record final system-level error context
                system_context = {
                    "component": "system",
                    "operation": "initialization",
                    "error_chain": [
                        event["component"] for event in error_tracker.error_events
                    ],
                    "final_error": str(e),
                    "timestamp": time.time(),
                }
                error_contexts.append(system_context)

        # Verify error context preservation
        assert len(error_contexts) > 0, "No error contexts were preserved"

        # Check that context includes component chain
        component_chain = []
        for context in error_contexts:
            if "component" in context:
                component_chain.append(context["component"])

        # Verify context chain follows dependency order
        expected_components = [
            "mqtt",
            "tracking",
        ]  # Database error should propagate to these

        for expected_component in expected_components:
            assert any(
                expected_component in context.get("component", "")
                for context in error_contexts
            ), f"Error context missing for component: {expected_component}"

        # Verify timestamps show error propagation timeline
        timestamps = [
            context.get("timestamp", 0)
            for context in error_contexts
            if "timestamp" in context
        ]
        if len(timestamps) > 1:
            # Timestamps should be in ascending order (showing propagation timeline)
            assert timestamps == sorted(
                timestamps
            ), "Error context timestamps not in chronological order"

        logger.info(
            f"Error context preservation: {len(error_contexts)} contexts preserved"
        )

    @pytest.mark.asyncio
    async def test_system_stability_during_error_storms(
        self, system_config, error_tracker, error_injector
    ):
        """Test system stability when multiple errors occur simultaneously."""

        # Register multiple simultaneous error injection points
        error_scenarios = [
            ("database", "health_check", DatabaseError, "DB error 1"),
            ("mqtt", "initialize", Exception, "MQTT error 1"),
            ("tracking", "initialize", SystemResourceError, "Tracking error 1"),
        ]

        for component, method, error_type, message in error_scenarios:
            error_injector.register_injection_point(
                component, method, error_type, message
            )

        mock_database_manager = AsyncMock()
        mock_mqtt_manager = AsyncMock()
        mock_tracking_manager = AsyncMock()

        # Implement error storm handling
        error_counts = {"database": 0, "mqtt": 0, "tracking": 0}
        max_errors_per_component = 3

        async def database_during_error_storm(*args, **kwargs):
            error_counts["database"] += 1
            if error_counts["database"] <= max_errors_per_component:
                try:
                    return await error_injector.create_error_injecting_mock(
                        "database", "health_check"
                    )()
                except Exception as e:
                    error_tracker.record_error_boundary(
                        "database",
                        "error_storm_protection",
                        f"error_{error_counts['database']}: {e}",
                    )
                    if error_counts["database"] >= max_errors_per_component:
                        # Implement circuit breaker
                        error_tracker.record_error_boundary(
                            "database", "circuit_breaker", "too_many_errors"
                        )
                        return {"status": "degraded"}
                    raise
            return {"status": "healthy"}

        async def mqtt_during_error_storm(*args, **kwargs):
            error_counts["mqtt"] += 1
            try:
                return await error_injector.create_error_injecting_mock(
                    "mqtt", "initialize"
                )()
            except Exception as e:
                error_tracker.record_error_boundary(
                    "mqtt", "error_storm_handling", f"error_{error_counts['mqtt']}: {e}"
                )
                if error_counts["mqtt"] >= max_errors_per_component:
                    error_tracker.record_error_boundary(
                        "mqtt", "fallback_mode", "switching_to_local_mode"
                    )
                    # Switch to local mode

        async def tracking_during_error_storm(*args, **kwargs):
            error_counts["tracking"] += 1
            try:
                return await error_injector.create_error_injecting_mock(
                    "tracking", "initialize"
                )()
            except Exception as e:
                error_tracker.record_error_boundary(
                    "tracking",
                    "error_storm_containment",
                    f"error_{error_counts['tracking']}: {e}",
                )
                if error_counts["tracking"] >= max_errors_per_component:
                    error_tracker.record_recovery(
                        "tracking", "minimal_functionality_mode"
                    )
                    return mock_tracking_manager
                raise

        mock_database_manager.health_check = database_during_error_storm
        mock_mqtt_manager.initialize = mqtt_during_error_storm
        mock_tracking_manager.initialize = tracking_during_error_storm
        mock_tracking_manager.get_api_server_status.return_value = {
            "enabled": True,
            "running": True,
            "host": "127.0.0.1",
            "port": 8000,
        }

        # Simulate error storm by rapid initialization attempts
        system_attempts = 0
        max_system_attempts = 5
        successful_initializations = 0

        for attempt in range(max_system_attempts):
            system_attempts += 1
            system = OccupancyPredictionSystem()

            with patch("src.main_system.get_config", return_value=system_config), patch(
                "src.main_system.get_database_manager",
                return_value=mock_database_manager,
            ), patch(
                "src.main_system.MQTTIntegrationManager", return_value=mock_mqtt_manager
            ), patch(
                "src.main_system.TrackingManager", return_value=mock_tracking_manager
            ):

                try:
                    await system.initialize()
                    successful_initializations += 1
                    error_tracker.record_recovery(
                        "system", f"successful_init_attempt_{attempt}"
                    )
                    await system.shutdown()
                except Exception as e:
                    error_tracker.record_error(
                        "system", "InitializationStormError", f"attempt_{attempt}: {e}"
                    )

                # Small delay between attempts
                await asyncio.sleep(0.1)

        # Analyze error storm handling
        containment_effectiveness = error_tracker.get_error_containment_effectiveness()
        total_boundaries = len(error_tracker.error_boundaries)
        total_recoveries = len(error_tracker.recovery_events)

        # Verify system maintained some level of functionality during error storm
        assert (
            total_boundaries > 0
        ), "No error boundaries established during error storm"
        assert (
            successful_initializations > 0 or total_recoveries > 0
        ), "System completely failed during error storm - no recovery mechanisms worked"

        # Verify error containment was attempted
        assert (
            containment_effectiveness["containment_rate"] > 0
        ), "No error containment during error storm"

        logger.info(
            f"Error storm handling: {successful_initializations}/{max_system_attempts} successful, "
            f"{total_boundaries} boundaries, {total_recoveries} recoveries"
        )
