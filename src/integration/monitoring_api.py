"""
Monitoring API endpoints for Home Assistant ML Predictor.
Provides REST API access to monitoring data, metrics, and system health.
"""

from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel

from ..utils.monitoring_integration import get_monitoring_integration
from ..utils.metrics import get_metrics_manager
from ..utils.alerts import get_alert_manager
from ..utils.logger import get_logger


class SystemStatus(BaseModel):
    """System status response model."""

    status: str
    timestamp: str
    uptime_seconds: float
    health_score: float
    active_alerts: int
    monitoring_enabled: bool


class MetricsResponse(BaseModel):
    """Metrics response model."""

    metrics_format: str
    timestamp: str
    metrics_count: int


class HealthCheckResponse(BaseModel):
    """Health check response model."""

    status: str
    checks: Dict[str, Any]
    timestamp: str
    overall_healthy: bool


class AlertsResponse(BaseModel):
    """Alerts response model."""

    active_alerts: int
    total_alerts_today: int
    alert_rules_configured: int
    notification_channels: list
    timestamp: str


# Create API router
monitoring_router = APIRouter(prefix="/monitoring", tags=["monitoring"])
logger = get_logger("monitoring_api")


@monitoring_router.get("/health", response_model=HealthCheckResponse)
async def get_health_status():
    """Get comprehensive system health status."""
    try:
        monitoring_integration = get_monitoring_integration()
        health_monitor = (
            monitoring_integration.get_monitoring_manager().get_health_monitor()
        )

        # Run health checks
        health_results = await health_monitor.run_health_checks()
        overall_status, overall_details = health_monitor.get_overall_health_status()

        # Format response
        checks = {}
        for name, result in health_results.items():
            checks[name] = {
                "status": result.status,
                "message": result.message,
                "response_time": result.response_time,
                "details": result.details,
            }

        return HealthCheckResponse(
            status=overall_status,
            checks=checks,
            timestamp=datetime.now().isoformat(),
            overall_healthy=(overall_status == "healthy"),
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {e}")


@monitoring_router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status."""
    try:
        monitoring_integration = get_monitoring_integration()
        status = await monitoring_integration.get_monitoring_status()

        # Extract key information
        health_score = 1.0  # Default to healthy
        if "health_details" in status["monitoring"]:
            health_details = status["monitoring"]["health_details"]
            if isinstance(health_details, dict):
                cpu_percent = health_details.get("cpu_percent", 0)
                memory_percent = health_details.get("memory_percent", 0)
                health_score = max(0.0, 1.0 - max(cpu_percent, memory_percent) / 100.0)

        return SystemStatus(
            status=status["monitoring"].get("health_status", "unknown"),
            timestamp=status.get("timestamp", datetime.now().isoformat()),
            uptime_seconds=0,  # Would need to track startup time
            health_score=health_score,
            active_alerts=status["monitoring"]["alert_system"]["active_alerts"],
            monitoring_enabled=status["monitoring"]["monitoring_active"],
        )

    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Status check failed: {e}")


@monitoring_router.get("/metrics")
async def get_prometheus_metrics():
    """Get Prometheus metrics in text format."""
    try:
        metrics_manager = get_metrics_manager()
        metrics_output = metrics_manager.get_metrics()

        # Return as plain text for Prometheus scraping
        return Response(
            content=metrics_output,
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )

    except Exception as e:
        logger.error(f"Metrics collection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics collection failed: {e}")


@monitoring_router.get("/metrics/summary", response_model=MetricsResponse)
async def get_metrics_summary():
    """Get metrics summary information."""
    try:
        metrics_manager = get_metrics_manager()
        metrics_output = metrics_manager.get_metrics()

        # Count metrics
        lines = metrics_output.split("\n")
        metrics_count = len(
            [line for line in lines if line and not line.startswith("#")]
        )

        return MetricsResponse(
            metrics_format="prometheus",
            timestamp=datetime.now().isoformat(),
            metrics_count=metrics_count,
        )

    except Exception as e:
        logger.error(f"Metrics summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Metrics summary failed: {e}")


@monitoring_router.get("/alerts", response_model=AlertsResponse)
async def get_alerts_status():
    """Get alerts system status."""
    try:
        alert_manager = get_alert_manager()
        alert_status = alert_manager.get_alert_status()

        return AlertsResponse(
            active_alerts=alert_status["active_alerts"],
            total_alerts_today=alert_status["total_alerts_today"],
            alert_rules_configured=alert_status["alert_rules_configured"],
            notification_channels=alert_status["notification_channels"],
            timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Alerts status check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Alerts status check failed: {e}")


@monitoring_router.get("/performance")
async def get_performance_summary(hours: int = 24):
    """Get performance summary for specified time window."""
    try:
        if hours < 1 or hours > 168:  # 1 week max
            raise HTTPException(
                status_code=400, detail="Hours must be between 1 and 168"
            )

        monitoring_integration = get_monitoring_integration()
        performance_monitor = (
            monitoring_integration.get_monitoring_manager().get_performance_monitor()
        )

        summary = performance_monitor.get_performance_summary(hours=hours)

        return {
            "time_window_hours": hours,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Performance summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance summary failed: {e}")


@monitoring_router.get("/performance/{metric_name}/trend")
async def get_performance_trend(
    metric_name: str, hours: int = 24, room_id: Optional[str] = None
):
    """Get performance trend analysis for a specific metric."""
    try:
        if hours < 1 or hours > 168:  # 1 week max
            raise HTTPException(
                status_code=400, detail="Hours must be between 1 and 168"
            )

        monitoring_integration = get_monitoring_integration()
        performance_monitor = (
            monitoring_integration.get_monitoring_manager().get_performance_monitor()
        )

        trend_analysis = performance_monitor.get_trend_analysis(
            metric_name=metric_name, room_id=room_id, hours=hours
        )

        return {
            "metric_name": metric_name,
            "room_id": room_id,
            "time_window_hours": hours,
            "analysis": trend_analysis,
            "timestamp": datetime.now().isoformat(),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Trend analysis failed: {e}")
        raise HTTPException(status_code=500, detail=f"Trend analysis failed: {e}")


@monitoring_router.post("/alerts/test")
async def trigger_test_alert():
    """Trigger a test alert for testing notification systems."""
    try:
        alert_manager = get_alert_manager()

        alert_id = await alert_manager.trigger_alert(
            rule_name="api_test_alert",
            title="Test Alert from API",
            message="This is a test alert triggered via the monitoring API",
            component="monitoring_api",
            context={
                "triggered_by": "api_endpoint",
                "test": True,
                "timestamp": datetime.now().isoformat(),
            },
        )

        return {
            "success": True,
            "alert_id": alert_id,
            "message": "Test alert triggered successfully",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Test alert failed: {e}")
        raise HTTPException(status_code=500, detail=f"Test alert failed: {e}")


@monitoring_router.get("/")
async def get_monitoring_info():
    """Get monitoring system information."""
    return {
        "service": "Home Assistant ML Predictor Monitoring",
        "version": "1.0.0",
        "endpoints": {
            "health": "/monitoring/health - System health checks",
            "status": "/monitoring/status - Overall system status",
            "metrics": "/monitoring/metrics - Prometheus metrics (text format)",
            "metrics_summary": "/monitoring/metrics/summary - Metrics summary (JSON)",
            "alerts": "/monitoring/alerts - Alerts system status",
            "performance": "/monitoring/performance?hours=24 - Performance summary",
            "trend": "/monitoring/performance/{metric_name}/trend?hours=24&room_id=xxx - Trend analysis",
            "test_alert": "POST /monitoring/alerts/test - Trigger test alert",
        },
        "timestamp": datetime.now().isoformat(),
    }
