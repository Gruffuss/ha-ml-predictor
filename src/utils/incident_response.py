"""
Automated Incident Response System for Home Assistant ML Predictor.

Provides automated incident detection, classification, escalation, and recovery
capabilities with integration into the health monitoring and alerting systems.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Set

from .alerts import get_alert_manager
from .health_monitor import ComponentHealth, HealthStatus, get_health_monitor
from .logger import get_logger
from .metrics import get_metrics_collector


class IncidentSeverity(Enum):
    """Incident severity levels."""

    INFO = "info"
    MINOR = "minor"
    MAJOR = "major"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class IncidentStatus(Enum):
    """Incident status values."""

    NEW = "new"
    ACKNOWLEDGED = "acknowledged"
    INVESTIGATING = "investigating"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    CLOSED = "closed"


class RecoveryActionType(Enum):
    """Types of recovery actions."""

    RESTART_SERVICE = "restart_service"
    CLEAR_CACHE = "clear_cache"
    RESTART_COMPONENT = "restart_component"
    SCALE_RESOURCES = "scale_resources"
    FAILOVER = "failover"
    NOTIFICATION = "notification"
    CUSTOM = "custom"


@dataclass
class RecoveryAction:
    """Automated recovery action definition."""

    action_type: RecoveryActionType
    component: str
    description: str
    function: Callable
    conditions: Dict[str, Any] = field(default_factory=dict)
    max_attempts: int = 3
    cooldown_minutes: int = 15
    last_attempted: Optional[datetime] = None
    attempt_count: int = 0
    success_count: int = 0

    def can_attempt(self) -> bool:
        """Check if action can be attempted."""
        if self.attempt_count >= self.max_attempts:
            return False

        if self.last_attempted:
            cooldown = timedelta(minutes=self.cooldown_minutes)
            return datetime.now() - self.last_attempted >= cooldown

        return True

    def record_attempt(self, success: bool):
        """Record an attempt of this recovery action."""
        self.last_attempted = datetime.now()
        self.attempt_count += 1
        if success:
            self.success_count += 1

    def reset_attempts(self):
        """Reset attempt counter."""
        self.attempt_count = 0
        self.last_attempted = None


@dataclass
class Incident:
    """System incident with automated response capabilities."""

    incident_id: str
    title: str
    description: str
    severity: IncidentSeverity
    status: IncidentStatus
    component: str
    component_type: str
    created_at: datetime
    updated_at: datetime
    source_health: ComponentHealth

    # Incident tracking
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    resolution_notes: Optional[str] = None

    # Recovery tracking
    recovery_actions_attempted: List[str] = field(default_factory=list)
    recovery_success: bool = False
    auto_recovery_enabled: bool = True

    # Escalation tracking
    escalation_level: int = 0
    escalated_at: Optional[datetime] = None
    escalation_threshold_minutes: int = 30

    # Additional context
    context: Dict[str, Any] = field(default_factory=dict)
    timeline: List[Dict[str, Any]] = field(default_factory=list)

    def add_timeline_entry(self, event: str, details: Optional[Dict[str, Any]] = None):
        """Add entry to incident timeline."""
        entry = {
            "timestamp": datetime.now().isoformat(),
            "event": event,
            "details": details or {},
        }
        self.timeline.append(entry)
        self.updated_at = datetime.now()

    def acknowledge(self, acknowledged_by: str):
        """Acknowledge the incident."""
        self.status = IncidentStatus.ACKNOWLEDGED
        self.acknowledged_by = acknowledged_by
        self.acknowledged_at = datetime.now()
        self.updated_at = datetime.now()
        self.add_timeline_entry(
            "Incident acknowledged", {"acknowledged_by": acknowledged_by}
        )

    def resolve(self, resolution_notes: str, auto_resolved: bool = False):
        """Resolve the incident."""
        self.status = IncidentStatus.RESOLVED
        self.resolved_at = datetime.now()
        self.updated_at = datetime.now()
        self.resolution_notes = resolution_notes
        self.recovery_success = auto_resolved

        self.add_timeline_entry(
            "Incident resolved",
            {
                "resolution_notes": resolution_notes,
                "auto_resolved": auto_resolved,
                "recovery_actions_attempted": len(self.recovery_actions_attempted),
            },
        )

    def escalate(self):
        """Escalate the incident."""
        self.escalation_level += 1
        self.escalated_at = datetime.now()
        self.updated_at = datetime.now()

        # Update severity if escalating
        if self.severity == IncidentSeverity.MINOR and self.escalation_level >= 1:
            self.severity = IncidentSeverity.MAJOR
        elif self.severity == IncidentSeverity.MAJOR and self.escalation_level >= 2:
            self.severity = IncidentSeverity.CRITICAL
        elif self.severity == IncidentSeverity.CRITICAL and self.escalation_level >= 3:
            self.severity = IncidentSeverity.EMERGENCY

        self.add_timeline_entry(
            "Incident escalated",
            {
                "escalation_level": self.escalation_level,
                "new_severity": self.severity.value,
            },
        )

    def needs_escalation(self) -> bool:
        """Check if incident needs escalation."""
        if self.status in [IncidentStatus.RESOLVED, IncidentStatus.CLOSED]:
            return False

        if self.created_at < datetime.now() - timedelta(
            minutes=self.escalation_threshold_minutes
        ):
            return True

        return False

    def to_dict(self) -> Dict[str, Any]:
        """Convert incident to dictionary for API responses."""
        return {
            "incident_id": self.incident_id,
            "title": self.title,
            "description": self.description,
            "severity": self.severity.value,
            "status": self.status.value,
            "component": self.component,
            "component_type": self.component_type,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "acknowledged_by": self.acknowledged_by,
            "acknowledged_at": (
                self.acknowledged_at.isoformat() if self.acknowledged_at else None
            ),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "resolution_notes": self.resolution_notes,
            "recovery_actions_attempted": self.recovery_actions_attempted,
            "recovery_success": self.recovery_success,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "escalation_level": self.escalation_level,
            "escalated_at": (
                self.escalated_at.isoformat() if self.escalated_at else None
            ),
            "context": self.context,
            "timeline": self.timeline,
            "source_health": self.source_health.to_dict(),
        }


class IncidentResponseManager:
    """
    Automated incident response and recovery management system.

    Monitors health status, detects incidents, triggers automated recovery,
    and manages escalation workflows with comprehensive logging and metrics.
    """

    def __init__(self, check_interval: int = 60):
        self.logger = get_logger("incident_response")
        self.health_monitor = get_health_monitor()
        self.alert_manager = get_alert_manager()
        self.metrics_collector = get_metrics_collector()

        # Configuration
        self.check_interval = check_interval
        self.auto_recovery_enabled = True
        self.escalation_enabled = True

        # Incident tracking
        self.active_incidents: Dict[str, Incident] = {}
        self.incident_history: List[Incident] = []
        self.incident_counter = 0

        # Recovery actions registry
        self.recovery_actions: Dict[str, List[RecoveryAction]] = {}

        # Response state
        self._response_active = False
        self._response_task = None

        # Statistics
        self.stats = {
            "incidents_created": 0,
            "incidents_resolved": 0,
            "auto_recoveries_attempted": 0,
            "auto_recoveries_successful": 0,
            "escalations": 0,
        }

        # Register default recovery actions
        self._register_default_recovery_actions()

    def _register_default_recovery_actions(self):
        """Register default automated recovery actions."""

        # Database connection recovery
        self.register_recovery_action(
            component="database_connection",
            action=RecoveryAction(
                action_type=RecoveryActionType.RESTART_SERVICE,
                component="database_connection",
                description="Attempt to reconnect to database",
                function=self._recover_database_connection,
                conditions={"consecutive_failures": 3},
                max_attempts=3,
                cooldown_minutes=10,
            ),
        )

        # MQTT broker recovery
        self.register_recovery_action(
            component="mqtt_broker",
            action=RecoveryAction(
                action_type=RecoveryActionType.RESTART_SERVICE,
                component="mqtt_broker",
                description="Attempt to reconnect to MQTT broker",
                function=self._recover_mqtt_connection,
                conditions={"consecutive_failures": 2},
                max_attempts=5,
                cooldown_minutes=5,
            ),
        )

        # High memory usage recovery
        self.register_recovery_action(
            component="memory_usage",
            action=RecoveryAction(
                action_type=RecoveryActionType.CLEAR_CACHE,
                component="memory_usage",
                description="Clear application caches to reduce memory usage",
                function=self._recover_memory_usage,
                conditions={"memory_percent": 90},
                max_attempts=2,
                cooldown_minutes=20,
            ),
        )

        # API endpoints recovery
        self.register_recovery_action(
            component="api_endpoints",
            action=RecoveryAction(
                action_type=RecoveryActionType.RESTART_COMPONENT,
                component="api_endpoints",
                description="Restart API server components",
                function=self._recover_api_endpoints,
                conditions={"success_rate": 0.5},
                max_attempts=2,
                cooldown_minutes=15,
            ),
        )

    def register_recovery_action(self, component: str, action: RecoveryAction):
        """Register a recovery action for a component."""
        if component not in self.recovery_actions:
            self.recovery_actions[component] = []

        self.recovery_actions[component].append(action)
        self.logger.info(
            f"Registered recovery action for {component}: {action.description}",
            extra={
                "component": component,
                "action_type": action.action_type.value,
                "max_attempts": action.max_attempts,
                "cooldown_minutes": action.cooldown_minutes,
            },
        )

    async def start_incident_response(self):
        """Start automated incident response monitoring."""
        if self._response_active:
            self.logger.warning("Incident response is already active")
            return

        self._response_active = True
        self.logger.info("Starting automated incident response system")

        # Start response monitoring loop
        self._response_task = asyncio.create_task(self._response_loop())

        # Add incident callback to health monitor
        self.health_monitor.add_incident_callback(self._handle_health_incident)

        # Record startup
        self.metrics_collector.increment("incident_response_starts_total")

    async def stop_incident_response(self):
        """Stop automated incident response monitoring."""
        if not self._response_active:
            return

        self._response_active = False
        self.logger.info("Stopping automated incident response system")

        if self._response_task:
            self._response_task.cancel()
            try:
                await self._response_task
            except asyncio.CancelledError:
                pass

        # Record shutdown
        self.metrics_collector.increment("incident_response_stops_total")

    async def _response_loop(self):
        """Main incident response loop."""
        while self._response_active:
            try:
                # Check for escalations needed
                if self.escalation_enabled:
                    await self._check_escalations()

                # Clean up resolved incidents
                await self._cleanup_incidents()

                # Update incident metrics
                self._update_incident_metrics()

                await asyncio.sleep(self.check_interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in incident response loop: {e}")
                await asyncio.sleep(60)  # Wait longer on error

    async def _handle_health_incident(self, health: ComponentHealth):
        """Handle health incident from health monitor."""
        if not health.needs_attention():
            return

        # Check if incident already exists for this component
        existing_incident = None
        for incident in self.active_incidents.values():
            if incident.component == health.component_name and incident.status not in [
                IncidentStatus.RESOLVED,
                IncidentStatus.CLOSED,
            ]:
                existing_incident = incident
                break

        if existing_incident:
            # Update existing incident
            existing_incident.add_timeline_entry(
                "Health status updated",
                {
                    "status": health.status.value,
                    "message": health.message,
                    "consecutive_failures": health.consecutive_failures,
                    "response_time": health.response_time,
                },
            )
        else:
            # Create new incident
            incident = await self._create_incident_from_health(health)

            # Attempt automated recovery if enabled
            if self.auto_recovery_enabled and incident.auto_recovery_enabled:
                await self._attempt_automated_recovery(incident)

    async def _create_incident_from_health(self, health: ComponentHealth) -> Incident:
        """Create incident from component health status."""
        self.incident_counter += 1
        incident_id = (
            f"INC-{datetime.now().strftime('%Y%m%d')}-{self.incident_counter:04d}"
        )

        # Determine severity based on health status
        severity_map = {
            HealthStatus.DEGRADED: IncidentSeverity.MINOR,
            HealthStatus.CRITICAL: IncidentSeverity.MAJOR,
            HealthStatus.UNKNOWN: IncidentSeverity.MAJOR,
        }
        severity = severity_map.get(health.status, IncidentSeverity.MINOR)

        # Escalate severity based on consecutive failures
        if health.consecutive_failures >= 5:
            if severity == IncidentSeverity.MINOR:
                severity = IncidentSeverity.MAJOR
            elif severity == IncidentSeverity.MAJOR:
                severity = IncidentSeverity.CRITICAL

        incident = Incident(
            incident_id=incident_id,
            title=f"{health.component_name.title()} Health Alert",
            description=f"Component {health.component_name} is {health.status.value}: {health.message}",
            severity=severity,
            status=IncidentStatus.NEW,
            component=health.component_name,
            component_type=health.component_type.value,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            source_health=health,
            context={
                "consecutive_failures": health.consecutive_failures,
                "response_time": health.response_time,
                "error_count": health.error_count,
                "uptime_percentage": health.uptime_percentage,
                "metrics": health.metrics,
                "details": health.details,
            },
        )

        incident.add_timeline_entry("Incident created", {"source": "health_monitor"})

        # Store incident
        self.active_incidents[incident_id] = incident
        self.incident_history.append(incident)

        # Update statistics
        self.stats["incidents_created"] += 1

        # Create alert
        try:
            await self.alert_manager.trigger_alert(
                rule_name=f"incident_{health.component_name}",
                title=incident.title,
                message=incident.description,
                component=health.component_name,
                severity=severity.value,
                context={
                    "incident_id": incident_id,
                    "component_type": health.component_type.value,
                    "consecutive_failures": health.consecutive_failures,
                    "auto_recovery_enabled": incident.auto_recovery_enabled,
                },
            )
        except Exception as e:
            self.logger.error(f"Failed to create alert for incident {incident_id}: {e}")

        self.logger.warning(
            f"Created incident {incident_id} for {health.component_name}",
            extra={
                "incident_id": incident_id,
                "component": health.component_name,
                "severity": severity.value,
                "status": health.status.value,
                "consecutive_failures": health.consecutive_failures,
            },
        )

        return incident

    async def _attempt_automated_recovery(self, incident: Incident):
        """Attempt automated recovery for an incident."""
        component = incident.component

        if component not in self.recovery_actions:
            self.logger.info(
                f"No recovery actions registered for component {component}"
            )
            return

        recovery_attempted = False

        for action in self.recovery_actions[component]:
            if not action.can_attempt():
                continue

            # Check if action conditions are met
            if not self._check_recovery_conditions(action, incident):
                continue

            try:
                self.logger.info(
                    f"Attempting recovery action for incident {incident.incident_id}: {action.description}",
                    extra={
                        "incident_id": incident.incident_id,
                        "component": component,
                        "action_type": action.action_type.value,
                        "attempt": action.attempt_count + 1,
                        "max_attempts": action.max_attempts,
                    },
                )

                # Update statistics
                self.stats["auto_recoveries_attempted"] += 1

                # Attempt recovery
                success = await action.function(incident)
                action.record_attempt(success)

                # Record in incident
                incident.recovery_actions_attempted.append(
                    f"{action.action_type.value}:{action.description} ({'SUCCESS' if success else 'FAILED'})"
                )

                incident.add_timeline_entry(
                    "Recovery action attempted",
                    {
                        "action_type": action.action_type.value,
                        "description": action.description,
                        "success": success,
                        "attempt": action.attempt_count,
                    },
                )

                if success:
                    self.stats["auto_recoveries_successful"] += 1
                    incident.recovery_success = True

                    self.logger.info(
                        f"Recovery successful for incident {incident.incident_id}",
                        extra={
                            "incident_id": incident.incident_id,
                            "action_type": action.action_type.value,
                            "component": component,
                        },
                    )

                    # Check if incident should be auto-resolved
                    if await self._should_auto_resolve_incident(incident):
                        await self._auto_resolve_incident(incident)

                    recovery_attempted = True
                    break
                else:
                    self.logger.warning(
                        f"Recovery failed for incident {incident.incident_id}",
                        extra={
                            "incident_id": incident.incident_id,
                            "action_type": action.action_type.value,
                            "component": component,
                            "attempt": action.attempt_count,
                        },
                    )

                recovery_attempted = True

            except Exception as e:
                action.record_attempt(False)
                incident.recovery_actions_attempted.append(
                    f"{action.action_type.value}:{action.description} (ERROR: {str(e)[:100]})"
                )

                self.logger.error(
                    f"Recovery action failed with exception for incident {incident.incident_id}: {e}",
                    extra={
                        "incident_id": incident.incident_id,
                        "action_type": action.action_type.value,
                        "component": component,
                    },
                )

        if not recovery_attempted:
            self.logger.info(
                f"No suitable recovery actions available for incident {incident.incident_id}",
                extra={"incident_id": incident.incident_id, "component": component},
            )

    def _check_recovery_conditions(
        self, action: RecoveryAction, incident: Incident
    ) -> bool:
        """Check if recovery action conditions are met."""
        health = incident.source_health

        for condition, threshold in action.conditions.items():
            if condition == "consecutive_failures":
                if health.consecutive_failures < threshold:
                    return False
            elif condition == "memory_percent":
                memory_usage = health.metrics.get("memory_usage", 0)
                if memory_usage < threshold:
                    return False
            elif condition == "success_rate":
                success_rate = health.metrics.get("success_rate", 1.0)
                if success_rate >= threshold:
                    return False
            # Add more conditions as needed

        return True

    async def _should_auto_resolve_incident(self, incident: Incident) -> bool:
        """Check if incident should be automatically resolved."""
        # Get current health status for component
        current_health = self.health_monitor.get_component_health(incident.component)

        if not current_health:
            return False

        component_health = current_health[incident.component]

        # Auto-resolve if component is now healthy and recovery was successful
        return (
            component_health.is_healthy()
            and incident.recovery_success
            and component_health.consecutive_failures == 0
        )

    async def _auto_resolve_incident(self, incident: Incident):
        """Automatically resolve an incident."""
        incident.resolve(
            resolution_notes="Automatically resolved after successful recovery action",
            auto_resolved=True,
        )

        self.stats["incidents_resolved"] += 1

        self.logger.info(
            f"Auto-resolved incident {incident.incident_id}",
            extra={
                "incident_id": incident.incident_id,
                "component": incident.component,
                "recovery_actions": len(incident.recovery_actions_attempted),
            },
        )

    async def _check_escalations(self):
        """Check for incidents that need escalation."""
        for incident in self.active_incidents.values():
            if incident.needs_escalation():
                incident.escalate()
                self.stats["escalations"] += 1

                # Create escalation alert
                try:
                    await self.alert_manager.trigger_alert(
                        rule_name=f"incident_escalation_{incident.component}",
                        title=f"ESCALATED: {incident.title}",
                        message=f"Incident {incident.incident_id} has been escalated to {incident.severity.value}",
                        component=incident.component,
                        severity="critical",
                        context={
                            "incident_id": incident.incident_id,
                            "escalation_level": incident.escalation_level,
                            "original_severity": incident.severity.value,
                        },
                    )
                except Exception as e:
                    self.logger.error(
                        f"Failed to create escalation alert for {incident.incident_id}: {e}"
                    )

                self.logger.warning(
                    f"Escalated incident {incident.incident_id} to level {incident.escalation_level}",
                    extra={
                        "incident_id": incident.incident_id,
                        "escalation_level": incident.escalation_level,
                        "new_severity": incident.severity.value,
                    },
                )

    async def _cleanup_incidents(self):
        """Clean up resolved incidents."""
        # Move resolved incidents to history and remove from active
        resolved_incidents = []

        for incident_id, incident in list(self.active_incidents.items()):
            if incident.status == IncidentStatus.RESOLVED:
                # Auto-close resolved incidents after 1 hour
                if (
                    incident.resolved_at
                    and incident.resolved_at < datetime.now() - timedelta(hours=1)
                ):
                    incident.status = IncidentStatus.CLOSED
                    incident.add_timeline_entry(
                        "Incident closed", {"auto_closed": True}
                    )
                    resolved_incidents.append(incident_id)

        for incident_id in resolved_incidents:
            del self.active_incidents[incident_id]

    def _update_incident_metrics(self):
        """Update incident response metrics."""
        try:
            # Active incidents by status
            status_counts = {}
            for status in IncidentStatus:
                count = sum(
                    1 for i in self.active_incidents.values() if i.status == status
                )
                status_counts[status.value] = count
                self.metrics_collector.set_gauge(
                    f"incidents_active_{status.value}", count
                )

            # Active incidents by severity
            severity_counts = {}
            for severity in IncidentSeverity:
                count = sum(
                    1 for i in self.active_incidents.values() if i.severity == severity
                )
                severity_counts[severity.value] = count
                self.metrics_collector.set_gauge(
                    f"incidents_active_{severity.value}", count
                )

            # Overall statistics
            self.metrics_collector.set_gauge(
                "incidents_total_created", self.stats["incidents_created"]
            )
            self.metrics_collector.set_gauge(
                "incidents_total_resolved", self.stats["incidents_resolved"]
            )
            self.metrics_collector.set_gauge(
                "auto_recoveries_attempted", self.stats["auto_recoveries_attempted"]
            )
            self.metrics_collector.set_gauge(
                "auto_recoveries_successful", self.stats["auto_recoveries_successful"]
            )
            self.metrics_collector.set_gauge(
                "incident_escalations_total", self.stats["escalations"]
            )

        except Exception as e:
            self.logger.error(f"Failed to update incident metrics: {e}")

    # Recovery action implementations

    async def _recover_database_connection(self, incident: Incident) -> bool:
        """Attempt to recover database connection."""
        try:
            from ..data.storage.database import get_database_manager

            db_manager = await get_database_manager()

            # Try to reconnect
            health = await db_manager.health_check()

            return health.get("database_connected", False)

        except Exception as e:
            self.logger.error(f"Database recovery failed: {e}")
            return False

    async def _recover_mqtt_connection(self, incident: Incident) -> bool:
        """Attempt to recover MQTT broker connection."""
        try:
            # This would implement MQTT reconnection logic
            # For now, simulate recovery attempt
            await asyncio.sleep(1)  # Simulate reconnection time

            # Check if MQTT is now working
            health = self.health_monitor.get_component_health("mqtt_broker")
            if health:
                return health["mqtt_broker"].is_healthy()

            return False

        except Exception as e:
            self.logger.error(f"MQTT recovery failed: {e}")
            return False

    async def _recover_memory_usage(self, incident: Incident) -> bool:
        """Attempt to recover from high memory usage."""
        try:
            # Implement cache clearing, garbage collection, etc.
            import gc

            gc.collect()

            # Wait a moment for memory to be freed
            await asyncio.sleep(2)

            # Check if memory usage improved
            health = self.health_monitor.get_component_health("memory_usage")
            if health:
                current_health = health["memory_usage"]
                memory_percent = current_health.metrics.get(
                    "process_memory_percent", 100
                )
                return memory_percent < 80  # Consider recovered if under 80%

            return False

        except Exception as e:
            self.logger.error(f"Memory recovery failed: {e}")
            return False

    async def _recover_api_endpoints(self, incident: Incident) -> bool:
        """Attempt to recover API endpoint functionality."""
        try:
            # This would implement API restart/recovery logic
            # For now, simulate recovery attempt
            await asyncio.sleep(1)

            # Check if API endpoints are now working
            health = self.health_monitor.get_component_health("api_endpoints")
            if health:
                current_health = health["api_endpoints"]
                success_rate = current_health.metrics.get("success_rate", 0.0)
                return success_rate >= 0.8  # Consider recovered if 80% success rate

            return False

        except Exception as e:
            self.logger.error(f"API recovery failed: {e}")
            return False

    # Public API methods

    def get_active_incidents(self) -> Dict[str, Incident]:
        """Get all active incidents."""
        return self.active_incidents.copy()

    def get_incident(self, incident_id: str) -> Optional[Incident]:
        """Get specific incident by ID."""
        return self.active_incidents.get(incident_id)

    def get_incident_history(self, hours: int = 24) -> List[Incident]:
        """Get incident history for specified time period."""
        cutoff = datetime.now() - timedelta(hours=hours)
        return [i for i in self.incident_history if i.created_at >= cutoff]

    def get_incident_statistics(self) -> Dict[str, Any]:
        """Get incident response statistics."""
        active_by_severity = {}
        for severity in IncidentSeverity:
            count = sum(
                1 for i in self.active_incidents.values() if i.severity == severity
            )
            active_by_severity[severity.value] = count

        active_by_status = {}
        for status in IncidentStatus:
            count = sum(1 for i in self.active_incidents.values() if i.status == status)
            active_by_status[status.value] = count

        return {
            "response_active": self._response_active,
            "auto_recovery_enabled": self.auto_recovery_enabled,
            "escalation_enabled": self.escalation_enabled,
            "active_incidents_count": len(self.active_incidents),
            "active_incidents_by_severity": active_by_severity,
            "active_incidents_by_status": active_by_status,
            "registered_recovery_actions": {
                component: len(actions)
                for component, actions in self.recovery_actions.items()
            },
            "statistics": self.stats.copy(),
        }

    async def acknowledge_incident(
        self, incident_id: str, acknowledged_by: str
    ) -> bool:
        """Acknowledge an incident."""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.acknowledge(acknowledged_by)

            self.logger.info(
                f"Incident {incident_id} acknowledged by {acknowledged_by}",
                extra={"incident_id": incident_id, "acknowledged_by": acknowledged_by},
            )

            return True
        return False

    async def resolve_incident(self, incident_id: str, resolution_notes: str) -> bool:
        """Manually resolve an incident."""
        if incident_id in self.active_incidents:
            incident = self.active_incidents[incident_id]
            incident.resolve(resolution_notes, auto_resolved=False)

            self.stats["incidents_resolved"] += 1

            self.logger.info(
                f"Incident {incident_id} manually resolved",
                extra={
                    "incident_id": incident_id,
                    "resolution_notes": resolution_notes,
                },
            )

            return True
        return False


# Global incident response manager instance
_incident_response_manager = None


def get_incident_response_manager() -> IncidentResponseManager:
    """Get global incident response manager instance."""
    global _incident_response_manager
    if _incident_response_manager is None:
        _incident_response_manager = IncidentResponseManager()
    return _incident_response_manager
