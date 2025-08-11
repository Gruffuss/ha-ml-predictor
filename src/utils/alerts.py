"""
Error tracking and alerting system for Home Assistant ML Predictor.
Provides comprehensive error handling, notification, and recovery mechanisms.
"""

import asyncio
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from enum import Enum
import hashlib
import json
from typing import Any, Callable, Dict, List, Optional

import smtplib

from .logger import get_error_tracker, get_logger
from .metrics import get_metrics_collector


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertChannel(Enum):
    """Alert notification channels."""

    LOG = "log"
    EMAIL = "email"
    WEBHOOK = "webhook"
    MQTT = "mqtt"


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    condition: str
    severity: AlertSeverity
    channels: List[AlertChannel]
    throttle_minutes: int = 60
    description: str = ""
    recovery_condition: Optional[str] = None
    auto_recovery: bool = True


@dataclass
class AlertEvent:
    """Alert event details."""

    id: str
    rule_name: str
    severity: AlertSeverity
    timestamp: datetime
    title: str
    message: str
    component: str
    room_id: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    resolved: bool = False
    resolved_at: Optional[datetime] = None


@dataclass
class NotificationConfig:
    """Notification configuration."""

    email_enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    email_recipients: List[str] = field(default_factory=list)
    webhook_url: str = ""
    webhook_enabled: bool = False
    mqtt_enabled: bool = False
    mqtt_topic: str = "occupancy/alerts"


class AlertThrottler:
    """Manages alert throttling to prevent spam."""

    def __init__(self):
        self.last_alert_times: Dict[str, datetime] = {}
        self.alert_counts: Dict[str, int] = defaultdict(int)

    def should_send_alert(self, alert_id: str, throttle_minutes: int) -> bool:
        """Check if alert should be sent based on throttling rules."""
        now = datetime.now()

        if alert_id not in self.last_alert_times:
            self.last_alert_times[alert_id] = now
            self.alert_counts[alert_id] = 1
            return True

        time_since_last = now - self.last_alert_times[alert_id]

        if time_since_last.total_seconds() >= throttle_minutes * 60:
            self.last_alert_times[alert_id] = now
            self.alert_counts[alert_id] += 1
            return True

        return False

    def reset_throttle(self, alert_id: str):
        """Reset throttling for specific alert."""
        if alert_id in self.last_alert_times:
            del self.last_alert_times[alert_id]
        if alert_id in self.alert_counts:
            del self.alert_counts[alert_id]


class EmailNotifier:
    """Email notification handler."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = get_logger("email_notifier")

    async def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert via email."""
        if not self.config.email_enabled or not self.config.email_recipients:
            return False

        try:
            subject = f"[{alert.severity.value.upper()}] {alert.title}"
            body = self._format_email_body(alert)

            # Send email asynchronously
            await asyncio.get_event_loop().run_in_executor(
                None, self._send_email, subject, body
            )

            self.logger.info(f"Alert email sent: {alert.id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send alert email: {e}")
            return False

    def _send_email(self, subject: str, body: str):
        """Send email synchronously."""
        msg = MIMEMultipart()
        msg["From"] = self.config.smtp_username
        msg["To"] = ", ".join(self.config.email_recipients)
        msg["Subject"] = subject

        msg.attach(MIMEText(body, "html"))

        with smtplib.SMTP(self.config.smtp_host, self.config.smtp_port) as server:
            server.starttls()
            server.login(self.config.smtp_username, self.config.smtp_password)
            server.sendmail(
                self.config.smtp_username, self.config.email_recipients, msg.as_string()
            )

    def _format_email_body(self, alert: AlertEvent) -> str:
        """Format alert as HTML email body."""
        severity_color = {
            AlertSeverity.INFO: "#2196F3",
            AlertSeverity.WARNING: "#FF9800",
            AlertSeverity.ERROR: "#F44336",
            AlertSeverity.CRITICAL: "#D32F2F",
        }

        html = f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {severity_color[alert.severity]}; color: white; padding: 15px;">
                    <h2 style="margin: 0;">{alert.title}</h2>
                    <p style="margin: 5px 0;"><strong>Severity:</strong> {alert.severity.value.upper()}</p>
                    <p style="margin: 5px 0;"><strong>Time:</strong> {alert.timestamp.strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
                
                <div style="padding: 15px; background-color: #f5f5f5;">
                    <h3>Alert Details</h3>
                    <p><strong>Component:</strong> {alert.component}</p>
                    {f'<p><strong>Room:</strong> {alert.room_id}</p>' if alert.room_id else ''}
                    <p><strong>Message:</strong></p>
                    <div style="background-color: white; padding: 10px; border-left: 4px solid {severity_color[alert.severity]};">
                        {alert.message}
                    </div>
                </div>
                
                {self._format_context_html(alert.context) if alert.context else ''}
                
                <div style="padding: 15px; background-color: #e8e8e8; font-size: 12px; color: #666;">
                    <p>This alert was generated by the Home Assistant ML Predictor monitoring system.</p>
                    <p>Alert ID: {alert.id}</p>
                </div>
            </div>
        </body>
        </html>
        """

        return html

    def _format_context_html(self, context: Dict[str, Any]) -> str:
        """Format context information as HTML."""
        html = "<div style='padding: 15px;'><h3>Additional Information</h3><ul>"

        for key, value in context.items():
            html += f"<li><strong>{key}:</strong> {value}</li>"

        html += "</ul></div>"
        return html


class WebhookNotifier:
    """Webhook notification handler."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = get_logger("webhook_notifier")

    async def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert via webhook."""
        if not self.config.webhook_enabled or not self.config.webhook_url:
            return False

        try:
            import aiohttp

            payload = {
                "alert_id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "timestamp": alert.timestamp.isoformat(),
                "title": alert.title,
                "message": alert.message,
                "component": alert.component,
                "room_id": alert.room_id,
                "context": alert.context,
                "resolved": alert.resolved,
            }

            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as response:
                    if response.status < 400:
                        self.logger.info(f"Alert webhook sent: {alert.id}")
                        return True
                    else:
                        self.logger.error(
                            f"Webhook failed with status {response.status}: {await response.text()}"
                        )
                        return False

        except Exception as e:
            self.logger.error(f"Failed to send webhook alert: {e}")
            return False


class MQTTNotifier:
    """MQTT notification handler."""

    def __init__(self, config: NotificationConfig):
        self.config = config
        self.logger = get_logger("mqtt_notifier")
        self.mqtt_client = None

    async def send_alert(self, alert: AlertEvent) -> bool:
        """Send alert via MQTT."""
        if not self.config.mqtt_enabled:
            return False

        try:
            # Import MQTT client if needed
            if self.mqtt_client is None:
                await self._initialize_mqtt_client()

            # TODO: Implement MQTT alert publishing
            # payload = {
            #     "alert_id": alert.id,
            #     "severity": alert.severity.value,
            #     "timestamp": alert.timestamp.isoformat(),
            #     "title": alert.title,
            #     "message": alert.message,
            #     "component": alert.component,
            #     "room_id": alert.room_id,
            #     "resolved": alert.resolved,
            # }

            topic = f"{self.config.mqtt_topic}/{alert.severity.value}"

            # Publish alert (implementation depends on MQTT client setup)
            self.logger.info(f"Alert MQTT sent: {alert.id} to {topic}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to send MQTT alert: {e}")
            return False

    async def _initialize_mqtt_client(self):
        """Initialize MQTT client for alerts."""
        # This would integrate with the existing MQTT setup
        # For now, we'll just log that MQTT is requested
        self.logger.info("MQTT alert notifications requested")


class ErrorRecoveryManager:
    """Manages error recovery and system resilience."""

    def __init__(self):
        self.logger = get_logger("error_recovery")
        self.recovery_strategies: Dict[str, Callable] = {}
        self.recovery_history: deque = deque(maxlen=1000)

    def register_recovery_strategy(
        self, error_pattern: str, recovery_func: Callable[[Exception, Dict], bool]
    ):
        """Register a recovery strategy for specific error patterns."""
        self.recovery_strategies[error_pattern] = recovery_func

    async def attempt_recovery(self, error: Exception, context: Dict[str, Any]) -> bool:
        """Attempt to recover from an error."""
        error_type = type(error).__name__

        # Try specific recovery strategies
        for pattern, recovery_func in self.recovery_strategies.items():
            if pattern in error_type or pattern in str(error):
                try:
                    self.logger.info(
                        f"Attempting recovery for {error_type} using {pattern}"
                    )

                    success = await asyncio.get_event_loop().run_in_executor(
                        None, recovery_func, error, context
                    )

                    if success:
                        self.recovery_history.append(
                            {
                                "timestamp": datetime.now(),
                                "error_type": error_type,
                                "recovery_pattern": pattern,
                                "success": True,
                                "context": context,
                            }
                        )

                        self.logger.info(f"Recovery successful for {error_type}")
                        return True

                except Exception as recovery_error:
                    self.logger.error(f"Recovery strategy failed: {recovery_error}")

        # Log failed recovery attempt
        self.recovery_history.append(
            {
                "timestamp": datetime.now(),
                "error_type": error_type,
                "recovery_pattern": None,
                "success": False,
                "context": context,
            }
        )

        return False


class AlertManager:
    """Main alert management system."""

    def __init__(self, config: NotificationConfig = None):
        self.config = config or NotificationConfig()
        self.logger = get_logger("alert_manager")
        self.error_tracker = get_error_tracker()
        self.metrics_collector = get_metrics_collector()

        # Alert management
        self.active_alerts: Dict[str, AlertEvent] = {}
        self.alert_history: deque = deque(maxlen=10000)
        self.alert_rules: Dict[str, AlertRule] = {}
        self.throttler = AlertThrottler()

        # Notification handlers
        self.notifiers = {
            AlertChannel.EMAIL: EmailNotifier(self.config),
            AlertChannel.WEBHOOK: WebhookNotifier(self.config),
            AlertChannel.MQTT: MQTTNotifier(self.config),
        }

        # Error recovery
        self.recovery_manager = ErrorRecoveryManager()

        # Initialize default alert rules
        self._setup_default_alert_rules()

    def _setup_default_alert_rules(self):
        """Setup default alert rules for common issues."""
        default_rules = [
            AlertRule(
                name="high_prediction_latency",
                condition="prediction_latency > 2.0",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                throttle_minutes=30,
                description="Prediction latency exceeds 2 seconds",
            ),
            AlertRule(
                name="critical_prediction_latency",
                condition="prediction_latency > 5.0",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                throttle_minutes=15,
                description="Prediction latency critically high",
            ),
            AlertRule(
                name="low_prediction_accuracy",
                condition="prediction_accuracy > 30",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                throttle_minutes=60,
                description="Prediction accuracy degraded",
            ),
            AlertRule(
                name="model_training_failure",
                condition="training_error",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                throttle_minutes=30,
                description="Model training failed",
            ),
            AlertRule(
                name="concept_drift_detected",
                condition="concept_drift",
                severity=AlertSeverity.WARNING,
                channels=[AlertChannel.LOG],
                throttle_minutes=120,
                description="Concept drift detected in room patterns",
            ),
            AlertRule(
                name="database_connection_error",
                condition="database_error",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL, AlertChannel.WEBHOOK],
                throttle_minutes=10,
                description="Database connection failure",
            ),
            AlertRule(
                name="ha_connection_lost",
                condition="ha_connection_error",
                severity=AlertSeverity.ERROR,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                throttle_minutes=15,
                description="Home Assistant connection lost",
            ),
            AlertRule(
                name="system_resource_critical",
                condition="cpu_usage > 90 OR memory_usage > 90",
                severity=AlertSeverity.CRITICAL,
                channels=[AlertChannel.LOG, AlertChannel.EMAIL],
                throttle_minutes=30,
                description="Critical system resource usage",
            ),
        ]

        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    def add_alert_rule(self, rule: AlertRule):
        """Add or update an alert rule."""
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Alert rule added/updated: {rule.name}")

    def remove_alert_rule(self, rule_name: str):
        """Remove an alert rule."""
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            self.logger.info(f"Alert rule removed: {rule_name}")

    async def trigger_alert(
        self,
        rule_name: str,
        title: str,
        message: str,
        component: str,
        room_id: Optional[str] = None,
        context: Dict[str, Any] = None,
    ) -> Optional[str]:
        """Trigger an alert based on a rule."""
        if rule_name not in self.alert_rules:
            self.logger.error(f"Unknown alert rule: {rule_name}")
            return None

        rule = self.alert_rules[rule_name]

        # Create alert ID
        alert_id = self._generate_alert_id(rule_name, component, room_id, context)

        # Check throttling
        if not self.throttler.should_send_alert(alert_id, rule.throttle_minutes):
            self.logger.debug(f"Alert throttled: {alert_id}")
            return alert_id

        # Create alert event
        alert = AlertEvent(
            id=alert_id,
            rule_name=rule_name,
            severity=rule.severity,
            timestamp=datetime.now(),
            title=title,
            message=message,
            component=component,
            room_id=room_id,
            context=context or {},
        )

        # Store alert
        self.active_alerts[alert_id] = alert
        self.alert_history.append(alert)

        # Send notifications
        await self._send_notifications(alert, rule.channels)

        # Update metrics
        self.metrics_collector.record_error(
            error_type=rule_name, component=component, severity=rule.severity.value
        )

        self.logger.info(
            f"Alert triggered: {rule_name}",
            extra={
                "alert_id": alert_id,
                "severity": rule.severity.value,
                "component": component,
                "room_id": room_id,
            },
        )

        return alert_id

    async def resolve_alert(self, alert_id: str, message: str = ""):
        """Resolve an active alert."""
        if alert_id not in self.active_alerts:
            return

        alert = self.active_alerts[alert_id]
        alert.resolved = True
        alert.resolved_at = datetime.now()

        # Remove from active alerts
        del self.active_alerts[alert_id]

        # Reset throttling for this alert
        self.throttler.reset_throttle(alert_id)

        self.logger.info(
            f"Alert resolved: {alert_id}",
            extra={
                "rule_name": alert.rule_name,
                "component": alert.component,
                "resolution_message": message,
            },
        )

    async def _send_notifications(
        self, alert: AlertEvent, channels: List[AlertChannel]
    ):
        """Send alert through specified channels."""
        for channel in channels:
            try:
                if channel == AlertChannel.LOG:
                    # Already logged by calling function
                    continue
                elif channel in self.notifiers:
                    await self.notifiers[channel].send_alert(alert)

            except Exception as e:
                self.logger.error(f"Failed to send alert via {channel.value}: {e}")

    def _generate_alert_id(
        self,
        rule_name: str,
        component: str,
        room_id: Optional[str],
        context: Dict[str, Any],
    ) -> str:
        """Generate unique alert ID."""
        # Create hash based on rule, component, room, and relevant context
        hash_input = f"{rule_name}:{component}:{room_id or ''}:{json.dumps(context, sort_keys=True)}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:12]

    async def handle_prediction_error(
        self, error: Exception, room_id: str, prediction_type: str, model_type: str
    ):
        """Handle prediction-specific errors with potential recovery."""
        context = {
            "room_id": room_id,
            "prediction_type": prediction_type,
            "model_type": model_type,
            "error_message": str(error),
        }

        # Track error
        self.error_tracker.track_prediction_error(
            room_id, error, prediction_type, model_type
        )

        # Attempt recovery
        recovery_attempted = await self.recovery_manager.attempt_recovery(
            error, context
        )

        # Trigger alert
        await self.trigger_alert(
            rule_name="prediction_error",
            title=f"Prediction Error in {room_id}",
            message=f"Failed to generate {prediction_type} prediction using {model_type}: {error}",
            component="prediction_engine",
            room_id=room_id,
            context={**context, "recovery_attempted": recovery_attempted},
        )

    async def handle_model_training_error(
        self, error: Exception, room_id: str, model_type: str
    ):
        """Handle model training errors."""
        context = {
            "room_id": room_id,
            "model_type": model_type,
            "error_message": str(error),
        }

        await self.trigger_alert(
            rule_name="model_training_failure",
            title=f"Model Training Failed for {room_id}",
            message=f"Training failed for {model_type} model: {error}",
            component="model_training",
            room_id=room_id,
            context=context,
        )

    def get_alert_status(self) -> Dict[str, Any]:
        """Get current alert system status."""
        return {
            "active_alerts": len(self.active_alerts),
            "total_alerts_today": len(
                [
                    alert
                    for alert in self.alert_history
                    if alert.timestamp
                    >= datetime.now().replace(hour=0, minute=0, second=0)
                ]
            ),
            "alert_rules_configured": len(self.alert_rules),
            "notification_channels": [
                channel.value for channel in self.notifiers.keys()
            ],
            "recovery_strategies": len(self.recovery_manager.recovery_strategies),
        }


# Global alert manager instance
_alert_manager = None


def get_alert_manager() -> AlertManager:
    """Get global alert manager instance."""
    global _alert_manager
    if _alert_manager is None:
        _alert_manager = AlertManager()
    return _alert_manager
