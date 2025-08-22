"""
Comprehensive unit tests for incident_response.py.
Tests incident management, automated recovery, escalation, and response workflows.
"""

import asyncio
from datetime import datetime, timedelta
import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest

from src.utils.health_monitor import ComponentHealth, ComponentType, HealthStatus
from src.utils.incident_response import (
    Incident,
    IncidentResponseManager,
    IncidentSeverity,
    IncidentStatus,
    RecoveryAction,
    RecoveryActionType,
    get_incident_response_manager,
)


class TestIncidentSeverity:
    """Test IncidentSeverity enum functionality."""

    def test_incident_severity_values(self):
        """Test all incident severity enum values."""
        assert IncidentSeverity.INFO.value == "info"
        assert IncidentSeverity.MINOR.value == "minor"
        assert IncidentSeverity.MAJOR.value == "major"
        assert IncidentSeverity.CRITICAL.value == "critical"
        assert IncidentSeverity.EMERGENCY.value == "emergency"


class TestIncidentStatus:
    """Test IncidentStatus enum functionality."""

    def test_incident_status_values(self):
        """Test all incident status enum values."""
        assert IncidentStatus.NEW.value == "new"
        assert IncidentStatus.ACKNOWLEDGED.value == "acknowledged"
        assert IncidentStatus.INVESTIGATING.value == "investigating"
        assert IncidentStatus.IN_PROGRESS.value == "in_progress"
        assert IncidentStatus.RESOLVED.value == "resolved"
        assert IncidentStatus.CLOSED.value == "closed"


class TestRecoveryActionType:
    """Test RecoveryActionType enum functionality."""

    def test_recovery_action_type_values(self):
        """Test all recovery action type enum values."""
        assert RecoveryActionType.RESTART_SERVICE.value == "restart_service"
        assert RecoveryActionType.CLEAR_CACHE.value == "clear_cache"
        assert RecoveryActionType.RESTART_COMPONENT.value == "restart_component"
        assert RecoveryActionType.SCALE_RESOURCES.value == "scale_resources"
        assert RecoveryActionType.FAILOVER.value == "failover"
        assert RecoveryActionType.NOTIFICATION.value == "notification"
        assert RecoveryActionType.CUSTOM.value == "custom"


class TestRecoveryAction:
    """Test RecoveryAction dataclass functionality."""

    def test_recovery_action_creation(self):
        """Test creating RecoveryAction with all fields."""

        def test_function():
            return True

        conditions = {"consecutive_failures": 3}
        last_attempted = datetime.now()

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="database",
            description="Restart database service",
            function=test_function,
            conditions=conditions,
            max_attempts=5,
            cooldown_minutes=30,
            last_attempted=last_attempted,
            attempt_count=2,
            success_count=1,
        )

        assert action.action_type == RecoveryActionType.RESTART_SERVICE
        assert action.component == "database"
        assert action.description == "Restart database service"
        assert action.function == test_function
        assert action.conditions == conditions
        assert action.max_attempts == 5
        assert action.cooldown_minutes == 30
        assert action.last_attempted == last_attempted
        assert action.attempt_count == 2
        assert action.success_count == 1

    def test_recovery_action_defaults(self):
        """Test RecoveryAction with default values."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.CLEAR_CACHE,
            component="memory",
            description="Clear application cache",
            function=test_function,
        )

        assert action.conditions == {}
        assert action.max_attempts == 3
        assert action.cooldown_minutes == 15
        assert action.last_attempted is None
        assert action.attempt_count == 0
        assert action.success_count == 0

    def test_can_attempt_first_time(self):
        """Test can_attempt returns True for first attempt."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
        )

        assert action.can_attempt() is True

    def test_can_attempt_within_max_attempts(self):
        """Test can_attempt with attempts within limit."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
            max_attempts=3,
        )

        action.attempt_count = 2  # Below max_attempts

        assert action.can_attempt() is True

    def test_can_attempt_exceeded_max_attempts(self):
        """Test can_attempt returns False when max attempts exceeded."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
            max_attempts=3,
        )

        action.attempt_count = 3  # Equal to max_attempts

        assert action.can_attempt() is False

    def test_can_attempt_within_cooldown(self):
        """Test can_attempt returns False within cooldown period."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
            cooldown_minutes=15,
        )

        action.last_attempted = datetime.now() - timedelta(
            minutes=10
        )  # Within cooldown

        assert action.can_attempt() is False

    def test_can_attempt_after_cooldown(self):
        """Test can_attempt returns True after cooldown period."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
            cooldown_minutes=15,
        )

        action.last_attempted = datetime.now() - timedelta(minutes=20)  # After cooldown

        assert action.can_attempt() is True

    def test_record_attempt_success(self):
        """Test recording successful attempt."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
        )

        initial_attempts = action.attempt_count
        initial_successes = action.success_count

        action.record_attempt(True)

        assert action.attempt_count == initial_attempts + 1
        assert action.success_count == initial_successes + 1
        assert action.last_attempted is not None

    def test_record_attempt_failure(self):
        """Test recording failed attempt."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
        )

        initial_attempts = action.attempt_count
        initial_successes = action.success_count

        action.record_attempt(False)

        assert action.attempt_count == initial_attempts + 1
        assert action.success_count == initial_successes  # No change
        assert action.last_attempted is not None

    def test_reset_attempts(self):
        """Test resetting attempt counters."""

        def test_function():
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test action",
            function=test_function,
        )

        action.attempt_count = 5
        action.last_attempted = datetime.now()

        action.reset_attempts()

        assert action.attempt_count == 0
        assert action.last_attempted is None


class TestIncident:
    """Test Incident dataclass functionality."""

    @pytest.fixture
    def sample_component_health(self):
        """Create sample component health for testing."""
        return ComponentHealth(
            component_name="database",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time=5.0,
            message="Database connection failed",
            consecutive_failures=5,
        )

    def test_incident_creation(self, sample_component_health):
        """Test creating Incident with all fields."""
        created_at = datetime.now()
        updated_at = created_at + timedelta(minutes=5)
        context = {"error_code": "DB_CONNECTION_FAILED"}
        timeline = [{"timestamp": created_at.isoformat(), "event": "Incident created"}]

        incident = Incident(
            incident_id="INC-20240115-001",
            title="Database Connection Failure",
            description="Unable to connect to primary database",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.NEW,
            component="database",
            component_type="database",
            created_at=created_at,
            updated_at=updated_at,
            source_health=sample_component_health,
            acknowledged_by="admin",
            acknowledged_at=created_at + timedelta(minutes=2),
            resolved_at=None,
            resolution_notes=None,
            recovery_actions_attempted=["restart_database"],
            recovery_success=False,
            auto_recovery_enabled=True,
            escalation_level=1,
            escalated_at=created_at + timedelta(minutes=10),
            escalation_threshold_minutes=30,
            context=context,
            timeline=timeline,
        )

        assert incident.incident_id == "INC-20240115-001"
        assert incident.title == "Database Connection Failure"
        assert incident.description == "Unable to connect to primary database"
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.status == IncidentStatus.NEW
        assert incident.component == "database"
        assert incident.component_type == "database"
        assert incident.created_at == created_at
        assert incident.updated_at == updated_at
        assert incident.source_health == sample_component_health
        assert incident.acknowledged_by == "admin"
        assert incident.acknowledged_at == created_at + timedelta(minutes=2)
        assert incident.resolved_at is None
        assert incident.resolution_notes is None
        assert incident.recovery_actions_attempted == ["restart_database"]
        assert incident.recovery_success is False
        assert incident.auto_recovery_enabled is True
        assert incident.escalation_level == 1
        assert incident.escalated_at == created_at + timedelta(minutes=10)
        assert incident.escalation_threshold_minutes == 30
        assert incident.context == context
        assert incident.timeline == timeline

    def test_incident_defaults(self, sample_component_health):
        """Test Incident with default values."""
        created_at = datetime.now()
        updated_at = created_at

        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test incident description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=created_at,
            updated_at=updated_at,
            source_health=sample_component_health,
        )

        assert incident.acknowledged_by is None
        assert incident.acknowledged_at is None
        assert incident.resolved_at is None
        assert incident.resolution_notes is None
        assert incident.recovery_actions_attempted == []
        assert incident.recovery_success is False
        assert incident.auto_recovery_enabled is True
        assert incident.escalation_level == 0
        assert incident.escalated_at is None
        assert incident.escalation_threshold_minutes == 30
        assert incident.context == {}
        assert incident.timeline == []

    def test_add_timeline_entry(self, sample_component_health):
        """Test adding timeline entry."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        initial_updated_at = incident.updated_at
        details = {"action": "restart", "result": "success"}

        incident.add_timeline_entry("Recovery attempted", details)

        assert len(incident.timeline) == 1
        assert incident.timeline[0]["event"] == "Recovery attempted"
        assert incident.timeline[0]["details"] == details
        assert incident.updated_at > initial_updated_at

    def test_acknowledge(self, sample_component_health):
        """Test acknowledging incident."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        incident.acknowledge("admin_user")

        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_by == "admin_user"
        assert incident.acknowledged_at is not None
        assert len(incident.timeline) == 1
        assert incident.timeline[0]["event"] == "Incident acknowledged"

    def test_resolve(self, sample_component_health):
        """Test resolving incident."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.IN_PROGRESS,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        resolution_notes = "Database restarted successfully"

        incident.resolve(resolution_notes, auto_resolved=True)

        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None
        assert incident.resolution_notes == resolution_notes
        assert incident.recovery_success is True
        assert len(incident.timeline) == 1
        assert incident.timeline[0]["event"] == "Incident resolved"

    def test_escalate(self, sample_component_health):
        """Test escalating incident."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        initial_severity = incident.severity
        incident.escalate()

        assert incident.escalation_level == 1
        assert incident.escalated_at is not None
        assert incident.severity == IncidentSeverity.MAJOR  # Should escalate from MINOR
        assert len(incident.timeline) == 1
        assert incident.timeline[0]["event"] == "Incident escalated"

    def test_escalate_severity_progression(self, sample_component_health):
        """Test escalation severity progression."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        # First escalation: MINOR -> MAJOR
        incident.escalate()
        assert incident.severity == IncidentSeverity.MAJOR
        assert incident.escalation_level == 1

        # Second escalation: MAJOR -> CRITICAL
        incident.escalate()
        assert incident.severity == IncidentSeverity.CRITICAL
        assert incident.escalation_level == 2

        # Third escalation: CRITICAL -> EMERGENCY
        incident.escalate()
        assert incident.severity == IncidentSeverity.EMERGENCY
        assert incident.escalation_level == 3

    def test_needs_escalation_no_escalation_needed(self, sample_component_health):
        """Test needs_escalation returns False when no escalation needed."""
        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),  # Just created
            updated_at=datetime.now(),
            source_health=sample_component_health,
            escalation_threshold_minutes=30,
        )

        assert incident.needs_escalation() is False

    def test_needs_escalation_time_exceeded(self, sample_component_health):
        """Test needs_escalation returns True when threshold exceeded."""
        old_time = datetime.now() - timedelta(minutes=45)

        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=old_time,  # Created 45 minutes ago
            updated_at=old_time,
            source_health=sample_component_health,
            escalation_threshold_minutes=30,
        )

        assert incident.needs_escalation() is True

    def test_needs_escalation_resolved_incident(self, sample_component_health):
        """Test needs_escalation returns False for resolved incident."""
        old_time = datetime.now() - timedelta(minutes=45)

        incident = Incident(
            incident_id="INC-TEST-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.RESOLVED,  # Already resolved
            component="test",
            component_type="application",
            created_at=old_time,
            updated_at=old_time,
            source_health=sample_component_health,
            escalation_threshold_minutes=30,
        )

        assert incident.needs_escalation() is False

    def test_to_dict(self, sample_component_health):
        """Test converting incident to dictionary."""
        created_at = datetime(2024, 1, 15, 14, 30, 0)
        updated_at = datetime(2024, 1, 15, 14, 35, 0)
        acknowledged_at = datetime(2024, 1, 15, 14, 32, 0)

        incident = Incident(
            incident_id="INC-20240115-001",
            title="Test Incident",
            description="Test description",
            severity=IncidentSeverity.MAJOR,
            status=IncidentStatus.ACKNOWLEDGED,
            component="database",
            component_type="database",
            created_at=created_at,
            updated_at=updated_at,
            source_health=sample_component_health,
            acknowledged_by="admin",
            acknowledged_at=acknowledged_at,
            recovery_actions_attempted=["restart_db"],
            recovery_success=False,
            escalation_level=1,
            context={"error": "CONNECTION_FAILED"},
            timeline=[{"timestamp": created_at.isoformat(), "event": "Created"}],
        )

        result = incident.to_dict()

        expected_keys = [
            "incident_id",
            "title",
            "description",
            "severity",
            "status",
            "component",
            "component_type",
            "created_at",
            "updated_at",
            "acknowledged_by",
            "acknowledged_at",
            "resolved_at",
            "resolution_notes",
            "recovery_actions_attempted",
            "recovery_success",
            "auto_recovery_enabled",
            "escalation_level",
            "escalated_at",
            "context",
            "timeline",
            "source_health",
        ]

        for key in expected_keys:
            assert key in result

        assert result["incident_id"] == "INC-20240115-001"
        assert result["severity"] == "major"
        assert result["status"] == "acknowledged"
        assert result["created_at"] == "2024-01-15T14:30:00"
        assert result["acknowledged_at"] == "2024-01-15T14:32:00"


class TestIncidentResponseManager:
    """Test IncidentResponseManager functionality."""

    @pytest.fixture
    def mock_dependencies(self):
        """Mock all dependencies for IncidentResponseManager."""
        with patch("src.utils.incident_response.get_logger") as mock_logger, patch(
            "src.utils.incident_response.get_health_monitor"
        ) as mock_health, patch(
            "src.utils.incident_response.get_alert_manager"
        ) as mock_alert, patch(
            "src.utils.incident_response.get_metrics_collector"
        ) as mock_metrics:

            yield {
                "logger": mock_logger.return_value,
                "health_monitor": mock_health.return_value,
                "alert_manager": mock_alert.return_value,
                "metrics_collector": mock_metrics.return_value,
            }

    @pytest.fixture
    def incident_manager(self, mock_dependencies):
        """Create IncidentResponseManager instance."""
        return IncidentResponseManager(check_interval=10)

    @pytest.fixture
    def sample_component_health(self):
        """Create sample component health for testing."""
        return ComponentHealth(
            component_name="test_component",
            component_type=ComponentType.DATABASE,
            status=HealthStatus.CRITICAL,
            last_check=datetime.now(),
            response_time=5.0,
            message="Component is failing",
            consecutive_failures=5,
        )

    def test_incident_manager_initialization(self, incident_manager):
        """Test IncidentResponseManager initialization."""
        assert incident_manager.check_interval == 10
        assert incident_manager.auto_recovery_enabled is True
        assert incident_manager.escalation_enabled is True
        assert isinstance(incident_manager.active_incidents, dict)
        assert isinstance(incident_manager.incident_history, list)
        assert len(incident_manager.recovery_actions) > 0  # Should have default actions
        assert incident_manager._response_active is False

    def test_default_recovery_actions_registered(self, incident_manager):
        """Test that default recovery actions are registered."""
        expected_components = [
            "database_connection",
            "mqtt_broker",
            "memory_usage",
            "api_endpoints",
        ]

        for component in expected_components:
            assert component in incident_manager.recovery_actions
            assert len(incident_manager.recovery_actions[component]) > 0

    def test_register_recovery_action(self, incident_manager):
        """Test registering custom recovery action."""

        def custom_recovery(incident):
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.CUSTOM,
            component="custom_component",
            description="Custom recovery action",
            function=custom_recovery,
        )

        incident_manager.register_recovery_action("custom_component", action)

        assert "custom_component" in incident_manager.recovery_actions
        assert action in incident_manager.recovery_actions["custom_component"]

    @pytest.mark.asyncio
    async def test_start_incident_response(self, incident_manager):
        """Test starting incident response system."""
        with patch("asyncio.create_task") as mock_create_task:
            await incident_manager.start_incident_response()

            assert incident_manager._response_active is True
            mock_create_task.assert_called_once()

    @pytest.mark.asyncio
    async def test_start_incident_response_already_active(self, incident_manager):
        """Test starting incident response when already active."""
        incident_manager._response_active = True

        with patch("asyncio.create_task") as mock_create_task:
            await incident_manager.start_incident_response()

            # Should not create new task
            mock_create_task.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_incident_response(self, incident_manager):
        """Test stopping incident response system."""
        # Setup active response
        incident_manager._response_active = True
        mock_task = Mock()
        mock_task.cancel = Mock()
        incident_manager._response_task = mock_task

        await incident_manager.stop_incident_response()

        assert incident_manager._response_active is False
        mock_task.cancel.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_health_incident_new_incident(
        self, incident_manager, sample_component_health
    ):
        """Test handling new health incident."""
        with patch.object(
            incident_manager, "_create_incident_from_health"
        ) as mock_create, patch.object(
            incident_manager, "_attempt_automated_recovery"
        ) as mock_recovery:

            mock_incident = Mock()
            mock_incident.auto_recovery_enabled = True
            mock_create.return_value = mock_incident

            await incident_manager._handle_health_incident(sample_component_health)

            mock_create.assert_called_once_with(sample_component_health)
            mock_recovery.assert_called_once_with(mock_incident)

    @pytest.mark.asyncio
    async def test_handle_health_incident_existing_incident(
        self, incident_manager, sample_component_health
    ):
        """Test handling health incident with existing incident."""
        # Setup existing incident
        existing_incident = Mock()
        existing_incident.component = "test_component"
        existing_incident.status = IncidentStatus.NEW
        existing_incident.add_timeline_entry = Mock()

        incident_manager.active_incidents["existing_id"] = existing_incident

        with patch.object(
            incident_manager, "_create_incident_from_health"
        ) as mock_create:
            await incident_manager._handle_health_incident(sample_component_health)

            # Should not create new incident
            mock_create.assert_not_called()
            # Should update existing incident
            existing_incident.add_timeline_entry.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_incident_from_health(
        self, incident_manager, sample_component_health
    ):
        """Test creating incident from component health."""
        with patch.object(
            incident_manager.alert_manager, "trigger_alert"
        ) as mock_alert:
            incident = await incident_manager._create_incident_from_health(
                sample_component_health
            )

            assert incident.component == "test_component"
            assert incident.severity in [
                IncidentSeverity.MINOR,
                IncidentSeverity.MAJOR,
                IncidentSeverity.CRITICAL,
            ]
            assert incident.status == IncidentStatus.NEW
            assert incident.incident_id in incident_manager.active_incidents
            assert incident in incident_manager.incident_history
            assert incident_manager.stats["incidents_created"] == 1

            mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_incident_severity_mapping(self, incident_manager):
        """Test incident severity mapping from health status."""
        health_status_mappings = [
            (HealthStatus.DEGRADED, IncidentSeverity.MINOR),
            (HealthStatus.CRITICAL, IncidentSeverity.MAJOR),
            (HealthStatus.UNKNOWN, IncidentSeverity.MAJOR),
        ]

        for health_status, expected_severity in health_status_mappings:
            health = ComponentHealth(
                component_name="test",
                component_type=ComponentType.APPLICATION,
                status=health_status,
                last_check=datetime.now(),
                response_time=1.0,
                message="Test",
                consecutive_failures=1,
            )

            with patch.object(incident_manager.alert_manager, "trigger_alert"):
                incident = await incident_manager._create_incident_from_health(health)

                assert incident.severity == expected_severity

    @pytest.mark.asyncio
    async def test_create_incident_escalation_on_consecutive_failures(
        self, incident_manager
    ):
        """Test severity escalation based on consecutive failures."""
        health = ComponentHealth(
            component_name="test",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.DEGRADED,  # Would normally be MINOR
            last_check=datetime.now(),
            response_time=1.0,
            message="Test",
            consecutive_failures=6,  # Should escalate severity
        )

        with patch.object(incident_manager.alert_manager, "trigger_alert"):
            incident = await incident_manager._create_incident_from_health(health)

            # Should escalate from MINOR to MAJOR due to high consecutive failures
            assert incident.severity == IncidentSeverity.MAJOR

    @pytest.mark.asyncio
    async def test_attempt_automated_recovery_no_actions(
        self, incident_manager, sample_component_health
    ):
        """Test automated recovery when no actions are registered."""
        incident = Incident(
            incident_id="test_incident",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="unknown_component",  # No actions registered
            component_type="unknown",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        await incident_manager._attempt_automated_recovery(incident)

        # Should complete without error, no recovery attempted
        assert len(incident.recovery_actions_attempted) == 0

    @pytest.mark.asyncio
    async def test_attempt_automated_recovery_success(
        self, incident_manager, sample_component_health
    ):
        """Test successful automated recovery."""

        # Setup successful recovery action
        async def successful_recovery(incident):
            return True

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test_component",
            description="Test recovery",
            function=successful_recovery,
            conditions={"consecutive_failures": 3},
        )

        incident_manager.recovery_actions["test_component"] = [action]

        incident = Incident(
            incident_id="test_incident",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        with patch.object(
            incident_manager, "_should_auto_resolve_incident", return_value=True
        ), patch.object(incident_manager, "_auto_resolve_incident") as mock_resolve:

            await incident_manager._attempt_automated_recovery(incident)

            assert incident.recovery_success is True
            assert len(incident.recovery_actions_attempted) == 1
            assert "SUCCESS" in incident.recovery_actions_attempted[0]
            mock_resolve.assert_called_once_with(incident)

    @pytest.mark.asyncio
    async def test_attempt_automated_recovery_failure(
        self, incident_manager, sample_component_health
    ):
        """Test failed automated recovery."""

        # Setup failing recovery action
        async def failing_recovery(incident):
            return False

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test_component",
            description="Test recovery",
            function=failing_recovery,
            conditions={"consecutive_failures": 3},
        )

        incident_manager.recovery_actions["test_component"] = [action]

        incident = Incident(
            incident_id="test_incident",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        await incident_manager._attempt_automated_recovery(incident)

        assert incident.recovery_success is False
        assert len(incident.recovery_actions_attempted) == 1
        assert "FAILED" in incident.recovery_actions_attempted[0]

    @pytest.mark.asyncio
    async def test_attempt_automated_recovery_exception(
        self, incident_manager, sample_component_health
    ):
        """Test automated recovery with exception."""

        # Setup exception-throwing recovery action
        async def exception_recovery(incident):
            raise Exception("Recovery failed with exception")

        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test_component",
            description="Test recovery",
            function=exception_recovery,
            conditions={"consecutive_failures": 3},
        )

        incident_manager.recovery_actions["test_component"] = [action]

        incident = Incident(
            incident_id="test_incident",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        await incident_manager._attempt_automated_recovery(incident)

        assert len(incident.recovery_actions_attempted) == 1
        assert "ERROR" in incident.recovery_actions_attempted[0]

    def test_check_recovery_conditions_consecutive_failures(
        self, incident_manager, sample_component_health
    ):
        """Test checking recovery conditions for consecutive failures."""
        action = RecoveryAction(
            action_type=RecoveryActionType.RESTART_SERVICE,
            component="test",
            description="Test",
            function=lambda x: True,
            conditions={"consecutive_failures": 5},
        )

        incident = Incident(
            incident_id="test",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        # Health has 5 consecutive failures, condition requires 5
        result = incident_manager._check_recovery_conditions(action, incident)
        assert result is True

        # Change condition to require 6
        action.conditions["consecutive_failures"] = 6
        result = incident_manager._check_recovery_conditions(action, incident)
        assert result is False

    def test_check_recovery_conditions_memory_usage(self, incident_manager):
        """Test checking recovery conditions for memory usage."""
        action = RecoveryAction(
            action_type=RecoveryActionType.CLEAR_CACHE,
            component="memory",
            description="Test",
            function=lambda x: True,
            conditions={"memory_percent": 80},
        )

        # Create health with high memory usage
        health = ComponentHealth(
            component_name="memory",
            component_type=ComponentType.SYSTEM,
            status=HealthStatus.WARNING,
            last_check=datetime.now(),
            response_time=1.0,
            message="High memory usage",
            metrics={"memory_usage": 85},  # Above threshold
        )

        incident = Incident(
            incident_id="test",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="memory",
            component_type="system",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=health,
        )

        result = incident_manager._check_recovery_conditions(action, incident)
        assert result is True

        # Change health to low memory usage
        health.metrics["memory_usage"] = 70  # Below threshold
        result = incident_manager._check_recovery_conditions(action, incident)
        assert result is False

    @pytest.mark.asyncio
    async def test_should_auto_resolve_incident_success(
        self, incident_manager, sample_component_health
    ):
        """Test auto-resolve check with healthy component."""
        incident = Incident(
            incident_id="test",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
            recovery_success=True,
        )

        # Mock healthy component
        healthy_health = ComponentHealth(
            component_name="test_component",
            component_type=ComponentType.APPLICATION,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time=0.1,
            message="Component is healthy",
            consecutive_failures=0,
        )

        incident_manager.health_monitor.get_component_health.return_value = {
            "test_component": healthy_health
        }

        result = await incident_manager._should_auto_resolve_incident(incident)
        assert result is True

    @pytest.mark.asyncio
    async def test_should_auto_resolve_incident_still_failing(
        self, incident_manager, sample_component_health
    ):
        """Test auto-resolve check with still failing component."""
        incident = Incident(
            incident_id="test",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
            recovery_success=True,
        )

        incident_manager.health_monitor.get_component_health.return_value = {
            "test_component": sample_component_health  # Still critical
        }

        result = await incident_manager._should_auto_resolve_incident(incident)
        assert result is False

    @pytest.mark.asyncio
    async def test_auto_resolve_incident(
        self, incident_manager, sample_component_health
    ):
        """Test auto-resolving incident."""
        incident = Incident(
            incident_id="test",
            title="Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test_component",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        await incident_manager._auto_resolve_incident(incident)

        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolved_at is not None
        assert incident.recovery_success is True
        assert "Automatically resolved" in incident.resolution_notes
        assert incident_manager.stats["incidents_resolved"] == 1

    @pytest.mark.asyncio
    async def test_check_escalations(self, incident_manager, sample_component_health):
        """Test checking for incidents needing escalation."""
        # Create incident that needs escalation
        old_time = datetime.now() - timedelta(minutes=45)

        incident = Incident(
            incident_id="escalation_test",
            title="Test Escalation",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=old_time,  # 45 minutes ago
            updated_at=old_time,
            source_health=sample_component_health,
            escalation_threshold_minutes=30,
        )

        incident_manager.active_incidents["escalation_test"] = incident

        with patch.object(
            incident_manager.alert_manager, "trigger_alert"
        ) as mock_alert:
            await incident_manager._check_escalations()

            assert incident.escalation_level == 1
            assert incident.severity == IncidentSeverity.MAJOR  # Escalated from MINOR
            assert incident_manager.stats["escalations"] == 1
            mock_alert.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_incidents(self, incident_manager, sample_component_health):
        """Test cleaning up resolved incidents."""
        old_time = datetime.now() - timedelta(hours=2)

        # Create resolved incident that should be closed
        resolved_incident = Incident(
            incident_id="cleanup_test",
            title="Cleanup Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.RESOLVED,
            component="test",
            component_type="application",
            created_at=old_time,
            updated_at=old_time,
            source_health=sample_component_health,
            resolved_at=old_time,
        )

        incident_manager.active_incidents["cleanup_test"] = resolved_incident

        await incident_manager._cleanup_incidents()

        # Should be removed from active incidents
        assert "cleanup_test" not in incident_manager.active_incidents
        assert resolved_incident.status == IncidentStatus.CLOSED

    def test_get_active_incidents(self, incident_manager, sample_component_health):
        """Test getting active incidents."""
        incident = Incident(
            incident_id="active_test",
            title="Active Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        incident_manager.active_incidents["active_test"] = incident

        result = incident_manager.get_active_incidents()

        assert "active_test" in result
        assert result["active_test"] == incident

    def test_get_incident(self, incident_manager, sample_component_health):
        """Test getting specific incident by ID."""
        incident = Incident(
            incident_id="specific_test",
            title="Specific Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        incident_manager.active_incidents["specific_test"] = incident

        result = incident_manager.get_incident("specific_test")
        assert result == incident

        # Test non-existent incident
        result = incident_manager.get_incident("nonexistent")
        assert result is None

    def test_get_incident_history(self, incident_manager, sample_component_health):
        """Test getting incident history within time window."""
        now = datetime.now()
        old_time = now - timedelta(hours=30)  # Outside 24 hour window
        recent_time = now - timedelta(hours=12)  # Inside 24 hour window

        old_incident = Incident(
            incident_id="old_incident",
            title="Old Incident",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.RESOLVED,
            component="test",
            component_type="application",
            created_at=old_time,
            updated_at=old_time,
            source_health=sample_component_health,
        )

        recent_incident = Incident(
            incident_id="recent_incident",
            title="Recent Incident",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.RESOLVED,
            component="test",
            component_type="application",
            created_at=recent_time,
            updated_at=recent_time,
            source_health=sample_component_health,
        )

        incident_manager.incident_history = [old_incident, recent_incident]

        result = incident_manager.get_incident_history(hours=24)

        assert len(result) == 1
        assert result[0] == recent_incident

    def test_get_incident_statistics(self, incident_manager, sample_component_health):
        """Test getting incident statistics."""
        # Setup some incidents
        incidents = [
            Incident(
                "id1",
                "Test 1",
                "Test",
                IncidentSeverity.MINOR,
                IncidentStatus.NEW,
                "test",
                "app",
                datetime.now(),
                datetime.now(),
                sample_component_health,
            ),
            Incident(
                "id2",
                "Test 2",
                "Test",
                IncidentSeverity.MAJOR,
                IncidentStatus.ACKNOWLEDGED,
                "test",
                "app",
                datetime.now(),
                datetime.now(),
                sample_component_health,
            ),
            Incident(
                "id3",
                "Test 3",
                "Test",
                IncidentSeverity.CRITICAL,
                IncidentStatus.RESOLVED,
                "test",
                "app",
                datetime.now(),
                datetime.now(),
                sample_component_health,
            ),
        ]

        incident_manager.active_incidents = {
            "id1": incidents[0],
            "id2": incidents[1],
            "id3": incidents[2],
        }

        # Setup recovery actions
        incident_manager.recovery_actions = {
            "component1": [Mock(), Mock()],
            "component2": [Mock()],
        }

        # Setup stats
        incident_manager.stats = {
            "incidents_created": 5,
            "incidents_resolved": 3,
            "auto_recoveries_attempted": 8,
            "auto_recoveries_successful": 6,
            "escalations": 2,
        }

        result = incident_manager.get_incident_statistics()

        assert result["response_active"] == incident_manager._response_active
        assert result["auto_recovery_enabled"] == incident_manager.auto_recovery_enabled
        assert result["escalation_enabled"] == incident_manager.escalation_enabled
        assert result["active_incidents_count"] == 3
        assert result["active_incidents_by_severity"]["minor"] == 1
        assert result["active_incidents_by_severity"]["major"] == 1
        assert result["active_incidents_by_severity"]["critical"] == 1
        assert result["active_incidents_by_status"]["new"] == 1
        assert result["active_incidents_by_status"]["acknowledged"] == 1
        assert result["active_incidents_by_status"]["resolved"] == 1
        assert result["registered_recovery_actions"]["component1"] == 2
        assert result["registered_recovery_actions"]["component2"] == 1
        assert result["statistics"] == incident_manager.stats

    @pytest.mark.asyncio
    async def test_acknowledge_incident(
        self, incident_manager, sample_component_health
    ):
        """Test acknowledging incident."""
        incident = Incident(
            incident_id="ack_test",
            title="Acknowledge Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.NEW,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        incident_manager.active_incidents["ack_test"] = incident

        result = await incident_manager.acknowledge_incident("ack_test", "admin_user")

        assert result is True
        assert incident.status == IncidentStatus.ACKNOWLEDGED
        assert incident.acknowledged_by == "admin_user"
        assert incident.acknowledged_at is not None

    @pytest.mark.asyncio
    async def test_acknowledge_incident_not_found(self, incident_manager):
        """Test acknowledging non-existent incident."""
        result = await incident_manager.acknowledge_incident("nonexistent", "admin")
        assert result is False

    @pytest.mark.asyncio
    async def test_resolve_incident(self, incident_manager, sample_component_health):
        """Test manually resolving incident."""
        incident = Incident(
            incident_id="resolve_test",
            title="Resolve Test",
            description="Test",
            severity=IncidentSeverity.MINOR,
            status=IncidentStatus.IN_PROGRESS,
            component="test",
            component_type="application",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        incident_manager.active_incidents["resolve_test"] = incident

        result = await incident_manager.resolve_incident(
            "resolve_test", "Issue fixed manually"
        )

        assert result is True
        assert incident.status == IncidentStatus.RESOLVED
        assert incident.resolution_notes == "Issue fixed manually"
        assert incident.recovery_success is False  # Manual resolution
        assert incident_manager.stats["incidents_resolved"] == 1

    @pytest.mark.asyncio
    async def test_resolve_incident_not_found(self, incident_manager):
        """Test resolving non-existent incident."""
        result = await incident_manager.resolve_incident("nonexistent", "Fixed")
        assert result is False

    @pytest.mark.asyncio
    async def test_recovery_database_connection(
        self, incident_manager, sample_component_health
    ):
        """Test database connection recovery."""
        incident = Incident(
            incident_id="db_recovery_test",
            title="DB Recovery Test",
            description="Test",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.NEW,
            component="database_connection",
            component_type="database",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        with patch("src.utils.incident_response.get_database_manager") as mock_get_db:
            mock_db_manager = AsyncMock()
            mock_db_manager.health_check = AsyncMock(
                return_value={"database_connected": True}
            )
            mock_get_db.return_value = mock_db_manager

            result = await incident_manager._recover_database_connection(incident)

            assert result is True
            mock_db_manager.health_check.assert_called_once()

    @pytest.mark.asyncio
    async def test_recovery_database_connection_failure(
        self, incident_manager, sample_component_health
    ):
        """Test database connection recovery failure."""
        incident = Incident(
            incident_id="db_recovery_test",
            title="DB Recovery Test",
            description="Test",
            severity=IncidentSeverity.CRITICAL,
            status=IncidentStatus.NEW,
            component="database_connection",
            component_type="database",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        with patch("src.utils.incident_response.get_database_manager") as mock_get_db:
            mock_get_db.side_effect = Exception("Database connection failed")

            result = await incident_manager._recover_database_connection(incident)

            assert result is False

    @pytest.mark.asyncio
    async def test_recovery_memory_usage(
        self, incident_manager, sample_component_health
    ):
        """Test memory usage recovery."""
        incident = Incident(
            incident_id="memory_recovery_test",
            title="Memory Recovery Test",
            description="Test",
            severity=IncidentSeverity.WARNING,
            status=IncidentStatus.NEW,
            component="memory_usage",
            component_type="system",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=sample_component_health,
        )

        # Mock improved memory health
        healthy_memory = ComponentHealth(
            component_name="memory_usage",
            component_type=ComponentType.SYSTEM,
            status=HealthStatus.HEALTHY,
            last_check=datetime.now(),
            response_time=1.0,
            message="Memory usage normal",
            metrics={"process_memory_percent": 50},  # Under 80%
        )

        incident_manager.health_monitor.get_component_health.return_value = {
            "memory_usage": healthy_memory
        }

        with patch("gc.collect"), patch("asyncio.sleep"):
            result = await incident_manager._recover_memory_usage(incident)

            assert result is True


class TestGlobalIncidentResponseManager:
    """Test global incident response manager functionality."""

    def test_get_incident_response_manager_singleton(self):
        """Test that get_incident_response_manager returns singleton instance."""
        manager1 = get_incident_response_manager()
        manager2 = get_incident_response_manager()

        assert manager1 is manager2
        assert isinstance(manager1, IncidentResponseManager)

    @patch("src.utils.incident_response._incident_response_manager", None)
    def test_get_incident_response_manager_creates_new_instance(self):
        """Test that get_incident_response_manager creates new instance when none exists."""
        # Reset global instance
        import src.utils.incident_response

        src.utils.incident_response._incident_response_manager = None

        manager = get_incident_response_manager()
        assert isinstance(manager, IncidentResponseManager)


if __name__ == "__main__":
    pytest.main([__file__])
