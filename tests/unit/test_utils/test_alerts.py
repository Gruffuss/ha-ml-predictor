"""
Comprehensive unit tests for alerts.py.
Tests alert management, notification systems, error recovery, and incident response.
"""

import asyncio
from collections import deque
from datetime import datetime, timedelta
import hashlib
import json
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import smtplib

from src.utils.alerts import (
    AlertChannel,
    AlertEvent,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertThrottler,
    EmailNotifier,
    ErrorRecoveryManager,
    MQTTNotifier,
    NotificationConfig,
    WebhookNotifier,
    get_alert_manager,
)


class TestAlertSeverity:
    """Test AlertSeverity enum functionality."""

    def test_alert_severity_values(self):
        """Test all alert severity enum values."""
        assert AlertSeverity.INFO.value == "info"
        assert AlertSeverity.WARNING.value == "warning"
        assert AlertSeverity.ERROR.value == "error"
        assert AlertSeverity.CRITICAL.value == "critical"


class TestAlertChannel:
    """Test AlertChannel enum functionality."""

    def test_alert_channel_values(self):
        """Test all alert channel enum values."""
        assert AlertChannel.LOG.value == "log"
        assert AlertChannel.EMAIL.value == "email"
        assert AlertChannel.WEBHOOK.value == "webhook"
        assert AlertChannel.MQTT.value == "mqtt"


class TestAlertRule:
    """Test AlertRule dataclass functionality."""

    def test_alert_rule_creation(self):
        """Test creating AlertRule with all fields."""
        rule = AlertRule(
            name="test_rule",
            condition="cpu_usage > 80",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG, AlertChannel.EMAIL],
            throttle_minutes=30,
            description="Test alert rule",
            recovery_condition="cpu_usage < 70",
            auto_recovery=True,
        )

        assert rule.name == "test_rule"
        assert rule.condition == "cpu_usage > 80"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.channels == [AlertChannel.LOG, AlertChannel.EMAIL]
        assert rule.throttle_minutes == 30
        assert rule.description == "Test alert rule"
        assert rule.recovery_condition == "cpu_usage < 70"
        assert rule.auto_recovery is True

    def test_alert_rule_defaults(self):
        """Test AlertRule with default values."""
        rule = AlertRule(
            name="simple_rule",
            condition="error_rate > 0.1",
            severity=AlertSeverity.ERROR,
            channels=[AlertChannel.LOG],
        )

        assert rule.throttle_minutes == 60
        assert rule.description == ""
        assert rule.recovery_condition is None
        assert rule.auto_recovery is True


class TestAlertEvent:
    """Test AlertEvent dataclass functionality."""

    def test_alert_event_creation(self):
        """Test creating AlertEvent with all fields."""
        timestamp = datetime.now()
        context = {"metric_value": 85.5, "threshold": 80.0}

        event = AlertEvent(
            id="alert_123",
            rule_name="high_cpu",
            severity=AlertSeverity.WARNING,
            timestamp=timestamp,
            title="High CPU Usage",
            message="CPU usage exceeded threshold",
            component="system",
            room_id="living_room",
            context=context,
            resolved=True,
            resolved_at=timestamp + timedelta(minutes=5),
        )

        assert event.id == "alert_123"
        assert event.rule_name == "high_cpu"
        assert event.severity == AlertSeverity.WARNING
        assert event.timestamp == timestamp
        assert event.title == "High CPU Usage"
        assert event.message == "CPU usage exceeded threshold"
        assert event.component == "system"
        assert event.room_id == "living_room"
        assert event.context == context
        assert event.resolved is True
        assert event.resolved_at == timestamp + timedelta(minutes=5)

    def test_alert_event_defaults(self):
        """Test AlertEvent with default values."""
        event = AlertEvent(
            id="alert_123",
            rule_name="test_rule",
            severity=AlertSeverity.INFO,
            timestamp=datetime.now(),
            title="Test Alert",
            message="Test message",
            component="test",
        )

        assert event.room_id is None
        assert event.context == {}
        assert event.resolved is False
        assert event.resolved_at is None


class TestNotificationConfig:
    """Test NotificationConfig dataclass functionality."""

    def test_notification_config_creation(self):
        """Test creating NotificationConfig with all fields."""
        config = NotificationConfig(
            email_enabled=True,
            smtp_host="smtp.gmail.com",
            smtp_port=587,
            smtp_username="test@example.com",
            smtp_password="password",
            email_recipients=["admin@example.com", "alerts@example.com"],
            webhook_url="https://hooks.slack.com/webhook",
            webhook_enabled=True,
            mqtt_enabled=True,
            mqtt_topic="alerts/notifications",
        )

        assert config.email_enabled is True
        assert config.smtp_host == "smtp.gmail.com"
        assert config.smtp_port == 587
        assert config.smtp_username == "test@example.com"
        assert config.smtp_password == "password"
        assert config.email_recipients == ["admin@example.com", "alerts@example.com"]
        assert config.webhook_url == "https://hooks.slack.com/webhook"
        assert config.webhook_enabled is True
        assert config.mqtt_enabled is True
        assert config.mqtt_topic == "alerts/notifications"

    def test_notification_config_defaults(self):
        """Test NotificationConfig with default values."""
        config = NotificationConfig()

        assert config.email_enabled is False
        assert config.smtp_host == ""
        assert config.smtp_port == 587
        assert config.smtp_username == ""
        assert config.smtp_password == ""
        assert config.email_recipients == []
        assert config.webhook_url == ""
        assert config.webhook_enabled is False
        assert config.mqtt_enabled is False
        assert config.mqtt_topic == "occupancy/alerts"


class TestAlertThrottler:
    """Test AlertThrottler functionality."""

    @pytest.fixture
    def throttler(self):
        """Create AlertThrottler instance."""
        return AlertThrottler()

    def test_should_send_alert_first_time(self, throttler):
        """Test that first alert should be sent."""
        result = throttler.should_send_alert("alert_1", throttle_minutes=60)
        assert result is True
        assert throttler.alert_counts["alert_1"] == 1

    def test_should_send_alert_within_throttle_period(self, throttler):
        """Test that alert within throttle period should not be sent."""
        # First alert should be sent
        assert throttler.should_send_alert("alert_1", throttle_minutes=60) is True

        # Immediate second alert should be throttled
        assert throttler.should_send_alert("alert_1", throttle_minutes=60) is False

    def test_should_send_alert_after_throttle_period(self, throttler):
        """Test that alert after throttle period should be sent."""
        # First alert
        assert throttler.should_send_alert("alert_1", throttle_minutes=1) is True

        # Mock time passage
        past_time = datetime.now() - timedelta(minutes=2)
        throttler.last_alert_times["alert_1"] = past_time

        # Should be sent after throttle period
        assert throttler.should_send_alert("alert_1", throttle_minutes=1) is True
        assert throttler.alert_counts["alert_1"] == 2

    def test_reset_throttle(self, throttler):
        """Test resetting throttle for specific alert."""
        # Set up throttled alert
        throttler.should_send_alert("alert_1", throttle_minutes=60)
        assert "alert_1" in throttler.last_alert_times
        assert "alert_1" in throttler.alert_counts

        # Reset throttle
        throttler.reset_throttle("alert_1")

        # Should be cleared
        assert "alert_1" not in throttler.last_alert_times
        assert "alert_1" not in throttler.alert_counts

    def test_different_alerts_not_throttled_together(self, throttler):
        """Test that different alerts are throttled independently."""
        assert throttler.should_send_alert("alert_1", throttle_minutes=60) is True
        assert throttler.should_send_alert("alert_2", throttle_minutes=60) is True

        # Both should be throttled on second attempt
        assert throttler.should_send_alert("alert_1", throttle_minutes=60) is False
        assert throttler.should_send_alert("alert_2", throttle_minutes=60) is False


class TestEmailNotifier:
    """Test EmailNotifier functionality."""

    @pytest.fixture
    def email_config(self):
        """Create email notification configuration."""
        return NotificationConfig(
            email_enabled=True,
            smtp_host="smtp.test.com",
            smtp_port=587,
            smtp_username="sender@test.com",
            smtp_password="password",
            email_recipients=["admin@test.com", "alerts@test.com"],
        )

    @pytest.fixture
    def email_notifier(self, email_config):
        """Create EmailNotifier instance."""
        return EmailNotifier(email_config)

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert event."""
        return AlertEvent(
            id="test_alert_123",
            rule_name="high_cpu",
            severity=AlertSeverity.WARNING,
            timestamp=datetime(2024, 1, 15, 14, 30, 0),
            title="High CPU Usage Alert",
            message="CPU usage exceeded 80%",
            component="system_monitor",
            room_id="server_room",
            context={"cpu_percent": 85.5, "threshold": 80.0},
        )

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, email_notifier, sample_alert):
        """Test that disabled email notifier doesn't send."""
        email_notifier.config.email_enabled = False

        result = await email_notifier.send_alert(sample_alert)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_no_recipients(self, email_notifier, sample_alert):
        """Test that notifier with no recipients doesn't send."""
        email_notifier.config.email_recipients = []

        result = await email_notifier.send_alert(sample_alert)
        assert result is False

    @pytest.mark.asyncio
    @patch("asyncio.get_event_loop")
    async def test_send_alert_success(
        self, mock_get_loop, email_notifier, sample_alert
    ):
        """Test successful email sending."""
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor = AsyncMock(return_value=None)

        result = await email_notifier.send_alert(sample_alert)

        assert result is True
        mock_loop.run_in_executor.assert_called_once()

    @pytest.mark.asyncio
    @patch("asyncio.get_event_loop")
    async def test_send_alert_failure(
        self, mock_get_loop, email_notifier, sample_alert
    ):
        """Test email sending failure."""
        mock_loop = Mock()
        mock_get_loop.return_value = mock_loop
        mock_loop.run_in_executor = AsyncMock(side_effect=Exception("SMTP error"))

        result = await email_notifier.send_alert(sample_alert)

        assert result is False

    @patch("smtplib.SMTP")
    def test_send_email_success(self, mock_smtp, email_notifier):
        """Test actual email sending mechanism."""
        mock_server = Mock()
        mock_smtp.return_value.__enter__.return_value = mock_server

        email_notifier._send_email("Test Subject", "Test Body")

        mock_smtp.assert_called_once_with("smtp.test.com", 587)
        mock_server.starttls.assert_called_once()
        mock_server.login.assert_called_once_with("sender@test.com", "password")
        mock_server.sendmail.assert_called_once()

    def test_format_email_body(self, email_notifier, sample_alert):
        """Test email body formatting."""
        body = email_notifier._format_email_body(sample_alert)

        assert "High CPU Usage Alert" in body
        assert "WARNING" in body
        assert "CPU usage exceeded 80%" in body
        assert "server_room" in body
        assert "2024-01-15 14:30:00" in body
        assert "test_alert_123" in body

    def test_format_context_html(self, email_notifier):
        """Test context formatting as HTML."""
        context = {"cpu_percent": 85.5, "threshold": 80.0, "server": "web-01"}

        html = email_notifier._format_context_html(context)

        assert "Additional Information" in html
        assert "cpu_percent" in html
        assert "85.5" in html
        assert "threshold" in html
        assert "80.0" in html


class TestWebhookNotifier:
    """Test WebhookNotifier functionality."""

    @pytest.fixture
    def webhook_config(self):
        """Create webhook notification configuration."""
        return NotificationConfig(
            webhook_enabled=True, webhook_url="https://hooks.example.com/webhook"
        )

    @pytest.fixture
    def webhook_notifier(self, webhook_config):
        """Create WebhookNotifier instance."""
        return WebhookNotifier(webhook_config)

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert event."""
        return AlertEvent(
            id="webhook_alert_123",
            rule_name="database_error",
            severity=AlertSeverity.ERROR,
            timestamp=datetime(2024, 1, 15, 15, 45, 0),
            title="Database Connection Error",
            message="Failed to connect to database",
            component="database",
            context={"error_code": "CONNECTION_REFUSED", "retry_count": 3},
        )

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, webhook_notifier, sample_alert):
        """Test that disabled webhook notifier doesn't send."""
        webhook_notifier.config.webhook_enabled = False

        result = await webhook_notifier.send_alert(sample_alert)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_no_url(self, webhook_notifier, sample_alert):
        """Test that notifier with no URL doesn't send."""
        webhook_notifier.config.webhook_url = ""

        result = await webhook_notifier.send_alert(sample_alert)
        assert result is False

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_send_alert_success(
        self, mock_session, webhook_notifier, sample_alert
    ):
        """Test successful webhook sending."""
        mock_response = Mock()
        mock_response.status = 200

        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = (
            mock_response
        )

        result = await webhook_notifier.send_alert(sample_alert)

        assert result is True

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_send_alert_http_error(
        self, mock_session, webhook_notifier, sample_alert
    ):
        """Test webhook sending with HTTP error."""
        mock_response = Mock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")

        mock_session.return_value.__aenter__.return_value.post.return_value.__aenter__.return_value = (
            mock_response
        )

        result = await webhook_notifier.send_alert(sample_alert)

        assert result is False

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_send_alert_exception(
        self, mock_session, webhook_notifier, sample_alert
    ):
        """Test webhook sending with exception."""
        mock_session.return_value.__aenter__.side_effect = Exception("Network error")

        result = await webhook_notifier.send_alert(sample_alert)

        assert result is False

    @pytest.mark.asyncio
    @patch("aiohttp.ClientSession")
    async def test_webhook_payload_format(
        self, mock_session, webhook_notifier, sample_alert
    ):
        """Test webhook payload format."""
        mock_response = Mock()
        mock_response.status = 200

        mock_post = Mock()
        mock_post.__aenter__.return_value = mock_response
        mock_session.return_value.__aenter__.return_value.post.return_value = mock_post

        await webhook_notifier.send_alert(sample_alert)

        # Verify the payload structure
        call_args = mock_session.return_value.__aenter__.return_value.post.call_args

        assert call_args[0][0] == "https://hooks.example.com/webhook"

        payload = call_args[1]["json"]
        assert payload["alert_id"] == "webhook_alert_123"
        assert payload["rule_name"] == "database_error"
        assert payload["severity"] == "error"
        assert payload["title"] == "Database Connection Error"
        assert payload["component"] == "database"
        assert payload["context"]["error_code"] == "CONNECTION_REFUSED"


class TestMQTTNotifier:
    """Test MQTTNotifier functionality."""

    @pytest.fixture
    def mqtt_config(self):
        """Create MQTT notification configuration."""
        return NotificationConfig(mqtt_enabled=True, mqtt_topic="alerts/notifications")

    @pytest.fixture
    def mqtt_notifier(self, mqtt_config):
        """Create MQTTNotifier instance."""
        return MQTTNotifier(mqtt_config)

    @pytest.fixture
    def sample_alert(self):
        """Create sample alert event."""
        return AlertEvent(
            id="mqtt_alert_123",
            rule_name="prediction_accuracy",
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            title="Prediction Accuracy Degraded",
            message="Model accuracy below threshold",
            component="ml_predictor",
            room_id="living_room",
        )

    @pytest.mark.asyncio
    async def test_send_alert_disabled(self, mqtt_notifier, sample_alert):
        """Test that disabled MQTT notifier doesn't send."""
        mqtt_notifier.config.mqtt_enabled = False

        result = await mqtt_notifier.send_alert(sample_alert)
        assert result is False

    @pytest.mark.asyncio
    async def test_send_alert_enabled(self, mqtt_notifier, sample_alert):
        """Test that enabled MQTT notifier processes alert."""
        # Mock MQTT client initialization
        with patch.object(mqtt_notifier, "_initialize_mqtt_client") as mock_init:
            mock_init.return_value = None

            result = await mqtt_notifier.send_alert(sample_alert)

            assert result is True
            mock_init.assert_called_once()

    @pytest.mark.asyncio
    async def test_send_alert_exception(self, mqtt_notifier, sample_alert):
        """Test MQTT alert sending with exception."""
        with patch.object(
            mqtt_notifier,
            "_initialize_mqtt_client",
            side_effect=Exception("MQTT error"),
        ):
            result = await mqtt_notifier.send_alert(sample_alert)
            assert result is False


class TestErrorRecoveryManager:
    """Test ErrorRecoveryManager functionality."""

    @pytest.fixture
    def recovery_manager(self):
        """Create ErrorRecoveryManager instance."""
        return ErrorRecoveryManager()

    def test_register_recovery_strategy(self, recovery_manager):
        """Test registering recovery strategy."""

        def test_recovery(error, context):
            return True

        recovery_manager.register_recovery_strategy("DatabaseError", test_recovery)

        assert "DatabaseError" in recovery_manager.recovery_strategies
        assert recovery_manager.recovery_strategies["DatabaseError"] == test_recovery

    @pytest.mark.asyncio
    async def test_attempt_recovery_success(self, recovery_manager):
        """Test successful error recovery."""

        def successful_recovery(error, context):
            return True

        recovery_manager.register_recovery_strategy("TestError", successful_recovery)

        test_error = Exception("TestError occurred")
        context = {"component": "test"}

        result = await recovery_manager.attempt_recovery(test_error, context)

        assert result is True
        assert len(recovery_manager.recovery_history) == 1
        assert recovery_manager.recovery_history[0]["success"] is True

    @pytest.mark.asyncio
    async def test_attempt_recovery_failure(self, recovery_manager):
        """Test failed error recovery."""

        def failing_recovery(error, context):
            return False

        recovery_manager.register_recovery_strategy("TestError", failing_recovery)

        test_error = Exception("TestError occurred")
        context = {"component": "test"}

        result = await recovery_manager.attempt_recovery(test_error, context)

        assert result is False
        assert len(recovery_manager.recovery_history) == 1
        assert recovery_manager.recovery_history[0]["success"] is False

    @pytest.mark.asyncio
    async def test_attempt_recovery_no_strategy(self, recovery_manager):
        """Test recovery attempt with no matching strategy."""
        test_error = Exception("UnknownError occurred")
        context = {"component": "test"}

        result = await recovery_manager.attempt_recovery(test_error, context)

        assert result is False
        assert len(recovery_manager.recovery_history) == 1
        assert recovery_manager.recovery_history[0]["recovery_pattern"] is None

    @pytest.mark.asyncio
    async def test_attempt_recovery_strategy_exception(self, recovery_manager):
        """Test recovery when strategy raises exception."""

        def exception_recovery(error, context):
            raise Exception("Recovery failed")

        recovery_manager.register_recovery_strategy("TestError", exception_recovery)

        test_error = Exception("TestError occurred")
        context = {"component": "test"}

        result = await recovery_manager.attempt_recovery(test_error, context)

        assert result is False


class TestAlertManager:
    """Test AlertManager functionality."""

    @pytest.fixture
    def notification_config(self):
        """Create notification configuration."""
        return NotificationConfig(
            email_enabled=True,
            email_recipients=["test@example.com"],
            webhook_enabled=True,
            webhook_url="https://webhook.example.com",
        )

    @pytest.fixture
    def alert_manager(self, notification_config):
        """Create AlertManager instance."""
        with patch("src.utils.alerts.get_error_tracker"), patch(
            "src.utils.alerts.get_metrics_collector"
        ):
            return AlertManager(notification_config)

    def test_alert_manager_initialization(self, alert_manager):
        """Test AlertManager initialization."""
        assert alert_manager.config is not None
        assert len(alert_manager.alert_rules) > 0  # Should have default rules
        assert isinstance(alert_manager.throttler, AlertThrottler)
        assert AlertChannel.EMAIL in alert_manager.notifiers
        assert AlertChannel.WEBHOOK in alert_manager.notifiers

    def test_default_alert_rules_setup(self, alert_manager):
        """Test that default alert rules are created."""
        expected_rules = [
            "high_prediction_latency",
            "critical_prediction_latency",
            "low_prediction_accuracy",
            "model_training_failure",
            "database_connection_error",
            "ha_connection_lost",
            "system_resource_critical",
        ]

        for rule_name in expected_rules:
            assert rule_name in alert_manager.alert_rules

    def test_add_alert_rule(self, alert_manager):
        """Test adding custom alert rule."""
        custom_rule = AlertRule(
            name="custom_test_rule",
            condition="test_metric > 100",
            severity=AlertSeverity.WARNING,
            channels=[AlertChannel.LOG],
        )

        alert_manager.add_alert_rule(custom_rule)

        assert "custom_test_rule" in alert_manager.alert_rules
        assert alert_manager.alert_rules["custom_test_rule"] == custom_rule

    def test_remove_alert_rule(self, alert_manager):
        """Test removing alert rule."""
        # Add a rule first
        test_rule = AlertRule(
            name="removable_rule",
            condition="test > 0",
            severity=AlertSeverity.INFO,
            channels=[AlertChannel.LOG],
        )
        alert_manager.add_alert_rule(test_rule)

        # Remove it
        alert_manager.remove_alert_rule("removable_rule")

        assert "removable_rule" not in alert_manager.alert_rules

    @pytest.mark.asyncio
    async def test_trigger_alert_success(self, alert_manager):
        """Test successful alert triggering."""
        with patch.object(alert_manager, "_send_notifications") as mock_send:
            mock_send.return_value = None

            alert_id = await alert_manager.trigger_alert(
                rule_name="high_prediction_latency",
                title="Test Alert",
                message="Test alert message",
                component="test_component",
                room_id="test_room",
                context={"value": 5.0},
            )

            assert alert_id is not None
            assert alert_id in alert_manager.active_alerts
            assert len(alert_manager.alert_history) > 0

            # Verify alert properties
            alert = alert_manager.active_alerts[alert_id]
            assert alert.rule_name == "high_prediction_latency"
            assert alert.title == "Test Alert"
            assert alert.component == "test_component"
            assert alert.room_id == "test_room"

    @pytest.mark.asyncio
    async def test_trigger_alert_unknown_rule(self, alert_manager):
        """Test triggering alert with unknown rule."""
        alert_id = await alert_manager.trigger_alert(
            rule_name="nonexistent_rule",
            title="Test Alert",
            message="Test message",
            component="test",
        )

        assert alert_id is None

    @pytest.mark.asyncio
    async def test_trigger_alert_throttled(self, alert_manager):
        """Test alert throttling."""
        # First alert should succeed
        alert_id_1 = await alert_manager.trigger_alert(
            rule_name="high_prediction_latency",
            title="First Alert",
            message="First message",
            component="test",
        )

        # Immediate second alert should be throttled
        alert_id_2 = await alert_manager.trigger_alert(
            rule_name="high_prediction_latency",
            title="Second Alert",
            message="Second message",
            component="test",
        )

        assert alert_id_1 is not None
        assert alert_id_2 == alert_id_1  # Same ID returned for throttled alert

    @pytest.mark.asyncio
    async def test_resolve_alert(self, alert_manager):
        """Test resolving active alert."""
        # Create alert first
        alert_id = await alert_manager.trigger_alert(
            rule_name="high_prediction_latency",
            title="Test Alert",
            message="Test message",
            component="test",
        )

        assert alert_id in alert_manager.active_alerts

        # Resolve it
        await alert_manager.resolve_alert(alert_id, "Issue resolved")

        assert alert_id not in alert_manager.active_alerts

        # Find in history and verify it's resolved
        resolved_alert = None
        for alert in alert_manager.alert_history:
            if alert.id == alert_id:
                resolved_alert = alert
                break

        assert resolved_alert is not None
        assert resolved_alert.resolved is True
        assert resolved_alert.resolved_at is not None

    def test_generate_alert_id_consistency(self, alert_manager):
        """Test that alert ID generation is consistent for same inputs."""
        id1 = alert_manager._generate_alert_id(
            "test_rule", "test_component", "test_room", {"key": "value"}
        )
        id2 = alert_manager._generate_alert_id(
            "test_rule", "test_component", "test_room", {"key": "value"}
        )

        assert id1 == id2

    def test_generate_alert_id_different_inputs(self, alert_manager):
        """Test that different inputs generate different alert IDs."""
        id1 = alert_manager._generate_alert_id("rule1", "component1", "room1", {})
        id2 = alert_manager._generate_alert_id("rule2", "component2", "room2", {})

        assert id1 != id2

    @pytest.mark.asyncio
    async def test_handle_prediction_error(self, alert_manager):
        """Test handling prediction-specific errors."""
        test_error = Exception("Prediction failed")

        with patch.object(
            alert_manager.error_tracker, "track_prediction_error"
        ), patch.object(
            alert_manager.recovery_manager, "attempt_recovery"
        ) as mock_recovery, patch.object(
            alert_manager, "trigger_alert"
        ) as mock_trigger:

            mock_recovery.return_value = True

            await alert_manager.handle_prediction_error(
                test_error, "living_room", "next_occupancy", "lstm"
            )

            mock_recovery.assert_called_once()
            mock_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_model_training_error(self, alert_manager):
        """Test handling model training errors."""
        test_error = Exception("Training failed")

        with patch.object(alert_manager, "trigger_alert") as mock_trigger:
            await alert_manager.handle_model_training_error(
                test_error, "bedroom", "xgboost"
            )

            mock_trigger.assert_called_once()
            call_args = mock_trigger.call_args[1]
            assert call_args["rule_name"] == "model_training_failure"
            assert call_args["room_id"] == "bedroom"

    def test_get_alert_status(self, alert_manager):
        """Test getting alert system status."""
        status = alert_manager.get_alert_status()

        assert "active_alerts" in status
        assert "total_alerts_today" in status
        assert "alert_rules_configured" in status
        assert "notification_channels" in status
        assert "recovery_strategies" in status

        assert isinstance(status["active_alerts"], int)
        assert isinstance(status["alert_rules_configured"], int)
        assert isinstance(status["notification_channels"], list)

    @pytest.mark.asyncio
    async def test_send_notifications_all_channels(self, alert_manager):
        """Test sending notifications through all configured channels."""
        alert = AlertEvent(
            id="test_alert",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            timestamp=datetime.now(),
            title="Test Alert",
            message="Test message",
            component="test",
        )

        channels = [AlertChannel.EMAIL, AlertChannel.WEBHOOK]

        with patch.object(
            alert_manager.notifiers[AlertChannel.EMAIL], "send_alert"
        ) as mock_email, patch.object(
            alert_manager.notifiers[AlertChannel.WEBHOOK], "send_alert"
        ) as mock_webhook:

            mock_email.return_value = True
            mock_webhook.return_value = True

            await alert_manager._send_notifications(alert, channels)

            mock_email.assert_called_once_with(alert)
            mock_webhook.assert_called_once_with(alert)

    @pytest.mark.asyncio
    async def test_send_notifications_exception_handling(self, alert_manager):
        """Test that notification exceptions don't break the flow."""
        alert = AlertEvent(
            id="test_alert",
            rule_name="test_rule",
            severity=AlertSeverity.ERROR,
            timestamp=datetime.now(),
            title="Test Alert",
            message="Test message",
            component="test",
        )

        channels = [AlertChannel.EMAIL]

        with patch.object(
            alert_manager.notifiers[AlertChannel.EMAIL], "send_alert"
        ) as mock_email:
            mock_email.side_effect = Exception("Email sending failed")

            # Should not raise exception
            await alert_manager._send_notifications(alert, channels)


class TestGlobalAlertManager:
    """Test global alert manager functionality."""

    def test_get_alert_manager_singleton(self):
        """Test that get_alert_manager returns singleton instance."""
        with patch("src.utils.alerts.get_error_tracker"), patch(
            "src.utils.alerts.get_metrics_collector"
        ):

            manager1 = get_alert_manager()
            manager2 = get_alert_manager()

            assert manager1 is manager2
            assert isinstance(manager1, AlertManager)

    @patch("src.utils.alerts._alert_manager", None)
    def test_get_alert_manager_creates_new_instance(self):
        """Test that get_alert_manager creates new instance when none exists."""
        with patch("src.utils.alerts.get_error_tracker"), patch(
            "src.utils.alerts.get_metrics_collector"
        ):

            # Reset global instance
            import src.utils.alerts

            src.utils.alerts._alert_manager = None

            manager = get_alert_manager()
            assert isinstance(manager, AlertManager)


if __name__ == "__main__":
    pytest.main([__file__])
