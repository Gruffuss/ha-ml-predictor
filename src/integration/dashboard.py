"""
Performance Monitoring Dashboard for Sprint 4 Task 5 - Self-Adaptation System.

This module provides a real-time performance monitoring dashboard with integrated
REST API endpoints and WebSocket support for live system monitoring and visualization.

Seamlessly integrates with existing TrackingManager, AccuracyTracker, ConceptDriftDetector,
and AdaptiveRetrainer to provide comprehensive system visibility without manual setup.
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
import logging
from pathlib import Path
import threading
from typing import Any, Callable, Dict, List, Optional, Set, Union
import uuid

try:
    from fastapi import (
        Depends,
        FastAPI,
        HTTPException,
        Query,
        WebSocket,
        WebSocketDisconnect,
    )
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from fastapi.staticfiles import StaticFiles
    from pydantic import BaseModel
    import uvicorn

    FASTAPI_AVAILABLE = True
except ImportError:
    # Graceful fallback if FastAPI not available
    FASTAPI_AVAILABLE = False
    FastAPI = None
    HTTPException = None
    WebSocket = None
    WebSocketDisconnect = None
    JSONResponse = None
    FileResponse = None
    StaticFiles = None
    CORSMiddleware = None
    BaseModel = None
    uvicorn = None

# Defer imports to prevent circular dependency
from typing import TYPE_CHECKING

from ..adaptation.drift_detector import (
    ConceptDriftDetector,
    DriftMetrics,
    DriftSeverity,
    DriftType,
)
from ..adaptation.retrainer import (
    AdaptiveRetrainer,
    RetrainingRequest,
    RetrainingStatus,
    RetrainingTrigger,
)
from ..adaptation.tracker import (
    AccuracyAlert,
    AccuracyTracker,
    AlertSeverity,
    RealTimeMetrics,
    TrendDirection,
)

if TYPE_CHECKING:
    from ..adaptation.tracking_manager import TrackingConfig, TrackingManager

from ..adaptation.validator import AccuracyLevel, AccuracyMetrics
from ..core.constants import ModelType
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError

logger = logging.getLogger(__name__)


class DashboardMode(Enum):
    """Dashboard operation modes."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"
    READONLY = "readonly"


class MetricType(Enum):
    """Types of metrics available in dashboard."""

    ACCURACY = "accuracy"
    PERFORMANCE = "performance"
    DRIFT = "drift"
    RETRAINING = "retraining"
    SYSTEM = "system"
    ALERTS = "alerts"


@dataclass
class DashboardConfig:
    """Configuration for performance monitoring dashboard."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8888
    debug: bool = False
    mode: DashboardMode = DashboardMode.PRODUCTION

    # Real-time updates configuration
    websocket_enabled: bool = True
    update_interval_seconds: int = 5
    max_websocket_connections: int = 50

    # Data retention for dashboard
    metrics_retention_hours: int = 72
    alert_retention_hours: int = 168
    max_chart_points: int = 1000

    # Security and access
    enable_cors: bool = True
    allowed_origins: List[str] = field(default_factory=lambda: ["*"])
    api_key_required: bool = False
    api_key: Optional[str] = None

    # Dashboard features
    enable_historical_charts: bool = True
    enable_drift_visualization: bool = True
    enable_retraining_controls: bool = True
    enable_alert_management: bool = True
    enable_export_features: bool = True

    # Performance tuning
    cache_ttl_seconds: int = 30
    concurrent_requests_limit: int = 100
    response_timeout_seconds: int = 30


@dataclass
class SystemOverview:
    """System overview metrics for dashboard."""

    # Overall health
    system_health_score: float = 0.0
    system_status: str = "unknown"

    # Prediction metrics
    total_predictions_24h: int = 0
    accuracy_rate_24h: float = 0.0
    mean_error_minutes_24h: float = 0.0

    # System activity
    active_rooms: int = 0
    active_models: int = 0
    predictions_per_hour: float = 0.0

    # Alerts and issues
    active_alerts: int = 0
    critical_alerts: int = 0
    warnings: int = 0

    # Drift and retraining
    rooms_with_drift: int = 0
    active_retraining_tasks: int = 0
    completed_retraining_24h: int = 0

    # Performance indicators
    avg_prediction_latency_ms: float = 0.0
    avg_validation_lag_minutes: float = 0.0
    cache_hit_rate: float = 0.0

    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    uptime_hours: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API response."""
        return {
            "system_health_score": self.system_health_score,
            "system_status": self.system_status,
            "total_predictions_24h": self.total_predictions_24h,
            "accuracy_rate_24h": self.accuracy_rate_24h,
            "mean_error_minutes_24h": self.mean_error_minutes_24h,
            "active_rooms": self.active_rooms,
            "active_models": self.active_models,
            "predictions_per_hour": self.predictions_per_hour,
            "active_alerts": self.active_alerts,
            "critical_alerts": self.critical_alerts,
            "warnings": self.warnings,
            "rooms_with_drift": self.rooms_with_drift,
            "active_retraining_tasks": self.active_retraining_tasks,
            "completed_retraining_24h": self.completed_retraining_24h,
            "avg_prediction_latency_ms": self.avg_prediction_latency_ms,
            "avg_validation_lag_minutes": self.avg_validation_lag_minutes,
            "cache_hit_rate": self.cache_hit_rate,
            "last_updated": self.last_updated.isoformat(),
            "uptime_hours": self.uptime_hours,
        }


class WebSocketManager:
    """
    WebSocket connection manager for real-time dashboard updates.

    Manages multiple client connections, broadcasts updates, and handles
    connection lifecycle for real-time monitoring dashboard.
    """

    def __init__(self, max_connections: int = 50):
        self.max_connections = max_connections
        self.active_connections: Set[WebSocket] = set()
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self._lock = threading.Lock()

        logger.info(
            f"Initialized WebSocketManager with max {max_connections} connections"
        )

    async def connect(
        self, websocket: WebSocket, client_info: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Accept and manage new WebSocket connection.

        Args:
            websocket: WebSocket connection to accept
            client_info: Optional client metadata

        Returns:
            True if connection accepted, False if rejected (max connections reached)
        """
        try:
            with self._lock:
                if len(self.active_connections) >= self.max_connections:
                    logger.warning(
                        f"WebSocket connection rejected: max connections ({self.max_connections}) reached"
                    )
                    return False

                await websocket.accept()
                self.active_connections.add(websocket)
                self.connection_metadata[websocket] = {
                    "connected_at": datetime.utcnow(),
                    "client_info": client_info or {},
                    "messages_sent": 0,
                    "last_message_at": None,
                }

            logger.info(
                f"WebSocket connection accepted: {len(self.active_connections)} active connections"
            )
            return True

        except Exception as e:
            logger.error(f"Error accepting WebSocket connection: {e}")
            return False

    async def disconnect(self, websocket: WebSocket) -> None:
        """Disconnect and clean up WebSocket connection."""
        try:
            with self._lock:
                self.active_connections.discard(websocket)
                self.connection_metadata.pop(websocket, None)

            logger.info(
                f"WebSocket disconnected: {len(self.active_connections)} active connections"
            )

        except Exception as e:
            logger.error(f"Error disconnecting WebSocket: {e}")

    async def send_personal_message(
        self, message: Dict[str, Any], websocket: WebSocket
    ) -> bool:
        """
        Send message to specific WebSocket connection.

        Args:
            message: Message to send
            websocket: Target WebSocket connection

        Returns:
            True if message sent successfully, False otherwise
        """
        try:
            if websocket in self.active_connections:
                await websocket.send_text(json.dumps(message, default=str))

                # Update metadata
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["messages_sent"] += 1
                    self.connection_metadata[websocket][
                        "last_message_at"
                    ] = datetime.utcnow()

                return True
            return False

        except WebSocketDisconnect:
            await self.disconnect(websocket)
            return False
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {e}")
            await self.disconnect(websocket)
            return False

    async def broadcast(self, message: Dict[str, Any]) -> int:
        """
        Broadcast message to all active WebSocket connections.

        Args:
            message: Message to broadcast

        Returns:
            Number of connections that received the message
        """
        if not self.active_connections:
            return 0

        disconnected_websockets = []
        successful_sends = 0

        # Send to all connections
        for websocket in list(self.active_connections):
            success = await self.send_personal_message(message, websocket)
            if success:
                successful_sends += 1
            else:
                disconnected_websockets.append(websocket)

        # Clean up disconnected connections
        for websocket in disconnected_websockets:
            await self.disconnect(websocket)

        if successful_sends > 0:
            logger.debug(
                f"Broadcast message to {successful_sends} WebSocket connections"
            )

        return successful_sends

    def get_connection_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        with self._lock:
            total_messages = sum(
                metadata.get("messages_sent", 0)
                for metadata in self.connection_metadata.values()
            )

            return {
                "active_connections": len(self.active_connections),
                "max_connections": self.max_connections,
                "total_messages_sent": total_messages,
                "connections_available": self.max_connections
                - len(self.active_connections),
            }


class PerformanceDashboard:
    """
    Real-time performance monitoring dashboard with integrated system visibility.

    Provides comprehensive monitoring interface with REST API endpoints, WebSocket
    real-time updates, and seamless integration with all Sprint 4 tracking components.

    Features:
    - System overview with key performance indicators
    - Real-time accuracy metrics and trends
    - Drift detection status and visualization
    - Retraining queue and history
    - Alert management and notifications
    - Historical data export and analysis
    - WebSocket broadcasting for live updates
    """

    def __init__(
        self, tracking_manager: "TrackingManager", config: DashboardConfig = None
    ):
        """
        Initialize performance dashboard with tracking manager integration.

        Args:
            tracking_manager: Integrated tracking manager with all components
            config: Dashboard configuration settings
        """
        if not FASTAPI_AVAILABLE:
            raise OccupancyPredictionError(
                "FastAPI not available - install with: pip install fastapi uvicorn websockets",
                severity=ErrorSeverity.HIGH,
            )

        self.tracking_manager = tracking_manager
        self.config = config or DashboardConfig()

        # Dashboard state
        self._dashboard_start_time = datetime.utcnow()
        self._update_task: Optional[asyncio.Task] = None
        self._server_task: Optional[asyncio.Task] = None
        self._running = False

        # Caching for performance
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._cache_lock = threading.Lock()

        # WebSocket management
        if self.config.websocket_enabled:
            self.websocket_manager = WebSocketManager(
                self.config.max_websocket_connections
            )
        else:
            self.websocket_manager = None

        # Initialize FastAPI app
        self.app = self._create_fastapi_app()

        logger.info(
            f"Initialized PerformanceDashboard on {self.config.host}:{self.config.port}, "
            f"WebSocket={'enabled' if self.config.websocket_enabled else 'disabled'}, "
            f"mode={self.config.mode.value}"
        )

    def _create_fastapi_app(self) -> FastAPI:
        """Create and configure FastAPI application with all endpoints."""
        app = FastAPI(
            title="Occupancy Prediction Performance Dashboard",
            description="Real-time monitoring dashboard for occupancy prediction system",
            version="1.0.0",
            debug=self.config.debug,
        )

        # CORS middleware
        if self.config.enable_cors:
            app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.allowed_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

        # Add all routes
        self._register_routes(app)

        return app

    def _register_routes(self, app: FastAPI) -> None:
        """Register all dashboard API routes."""

        # System overview endpoint
        @app.get("/api/dashboard/overview")
        async def get_system_overview():
            """Get comprehensive system overview with key metrics."""
            try:
                overview = await self._get_system_overview()
                return overview.to_dict()
            except Exception as e:
                logger.error(f"Error getting system overview: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Real-time accuracy metrics
        @app.get("/api/dashboard/accuracy")
        async def get_accuracy_metrics(
            room_id: Optional[str] = Query(None, description="Filter by room ID"),
            model_type: Optional[str] = Query(None, description="Filter by model type"),
            hours_back: int = Query(24, description="Hours of data to include"),
        ):
            """Get real-time accuracy metrics with optional filtering."""
            try:
                metrics = await self._get_accuracy_dashboard_data(
                    room_id, model_type, hours_back
                )
                return metrics
            except Exception as e:
                logger.error(f"Error getting accuracy metrics: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Drift detection status
        @app.get("/api/dashboard/drift")
        async def get_drift_status(
            room_id: Optional[str] = Query(None, description="Filter by room ID")
        ):
            """Get drift detection status and recent analysis results."""
            try:
                drift_data = await self._get_drift_dashboard_data(room_id)
                return drift_data
            except Exception as e:
                logger.error(f"Error getting drift status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Retraining information
        @app.get("/api/dashboard/retraining")
        async def get_retraining_status():
            """Get retraining queue status and history."""
            try:
                retraining_data = await self._get_retraining_dashboard_data()
                return retraining_data
            except Exception as e:
                logger.error(f"Error getting retraining status: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # System health check
        @app.get("/api/dashboard/health")
        async def get_system_health():
            """Get detailed system health information."""
            try:
                health_data = await self._get_system_health_data()
                return health_data
            except Exception as e:
                logger.error(f"Error getting system health: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Active alerts
        @app.get("/api/dashboard/alerts")
        async def get_active_alerts(
            severity: Optional[str] = Query(
                None, description="Filter by alert severity"
            ),
            room_id: Optional[str] = Query(None, description="Filter by room ID"),
        ):
            """Get active alerts with optional filtering."""
            try:
                alerts_data = await self._get_alerts_dashboard_data(severity, room_id)
                return alerts_data
            except Exception as e:
                logger.error(f"Error getting alerts: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Historical trends
        @app.get("/api/dashboard/trends")
        async def get_accuracy_trends(
            room_id: Optional[str] = Query(None, description="Filter by room ID"),
            days_back: int = Query(7, description="Days of historical data"),
        ):
            """Get historical accuracy trends for visualization."""
            try:
                trends_data = await self._get_trends_dashboard_data(room_id, days_back)
                return trends_data
            except Exception as e:
                logger.error(f"Error getting trends: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Dashboard statistics
        @app.get("/api/dashboard/stats")
        async def get_dashboard_stats():
            """Get dashboard system statistics."""
            try:
                stats = self._get_dashboard_stats()
                return stats
            except Exception as e:
                logger.error(f"Error getting dashboard stats: {e}")
                raise HTTPException(status_code=500, detail=str(e))

        # Manual actions (if enabled)
        if (
            self.config.enable_retraining_controls
            and self.config.mode != DashboardMode.READONLY
        ):

            @app.post("/api/dashboard/actions/retrain")
            async def trigger_manual_retraining(request_data: dict):
                """Trigger manual retraining for specified room/model."""
                try:
                    result = await self._trigger_manual_retraining(request_data)
                    return result
                except Exception as e:
                    logger.error(f"Error triggering retraining: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

        if (
            self.config.enable_alert_management
            and self.config.mode != DashboardMode.READONLY
        ):

            @app.post("/api/dashboard/actions/acknowledge_alert")
            async def acknowledge_alert(alert_data: dict):
                """Acknowledge active alert."""
                try:
                    result = await self._acknowledge_alert(alert_data)
                    return result
                except Exception as e:
                    logger.error(f"Error acknowledging alert: {e}")
                    raise HTTPException(status_code=500, detail=str(e))

        # WebSocket endpoint for real-time updates
        if self.config.websocket_enabled:

            @app.websocket("/ws/dashboard")
            async def websocket_endpoint(websocket: WebSocket):
                """WebSocket endpoint for real-time dashboard updates."""
                client_info = {
                    "client_host": (
                        websocket.client.host if websocket.client else "unknown"
                    ),
                    "client_port": websocket.client.port if websocket.client else 0,
                }

                connected = await self.websocket_manager.connect(websocket, client_info)
                if not connected:
                    await websocket.close(code=1013, reason="Server at capacity")
                    return

                try:
                    # Send initial data
                    initial_data = await self._get_websocket_initial_data()
                    await self.websocket_manager.send_personal_message(
                        initial_data, websocket
                    )

                    # Keep connection alive and handle incoming messages
                    while True:
                        try:
                            # Wait for messages (ping/pong, subscription changes, etc.)
                            message = await asyncio.wait_for(
                                websocket.receive_text(), timeout=30.0
                            )

                            # Handle client messages if needed
                            await self._handle_websocket_message(message, websocket)

                        except asyncio.TimeoutError:
                            # Send keepalive ping
                            await self.websocket_manager.send_personal_message(
                                {
                                    "type": "ping",
                                    "timestamp": datetime.utcnow().isoformat(),
                                },
                                websocket,
                            )

                except WebSocketDisconnect:
                    logger.info("WebSocket client disconnected normally")
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                finally:
                    await self.websocket_manager.disconnect(websocket)

    async def start_dashboard(self) -> None:
        """Start the dashboard server and background tasks."""
        if self._running:
            logger.warning("Dashboard already running")
            return

        try:
            self._running = True

            # Start background update task
            if self.config.websocket_enabled:
                self._update_task = asyncio.create_task(self._update_loop())

            # Start FastAPI server
            config = uvicorn.Config(
                app=self.app,
                host=self.config.host,
                port=self.config.port,
                log_level="info" if self.config.debug else "warning",
                access_log=self.config.debug,
            )

            server = uvicorn.Server(config)
            self._server_task = asyncio.create_task(server.serve())

            logger.info(
                f"Performance dashboard started on http://{self.config.host}:{self.config.port}"
            )

            # Wait for server to start
            await asyncio.sleep(1)

        except Exception as e:
            logger.error(f"Failed to start dashboard: {e}")
            self._running = False
            raise

    async def stop_dashboard(self) -> None:
        """Stop the dashboard server and background tasks."""
        if not self._running:
            logger.info("Dashboard not running")
            return

        try:
            self._running = False

            # Stop background tasks
            if self._update_task:
                self._update_task.cancel()
                try:
                    await self._update_task
                except asyncio.CancelledError:
                    pass
                self._update_task = None

            if self._server_task:
                self._server_task.cancel()
                try:
                    await self._server_task
                except asyncio.CancelledError:
                    pass
                self._server_task = None

            # Disconnect all WebSocket connections
            if self.websocket_manager:
                for websocket in list(self.websocket_manager.active_connections):
                    await self.websocket_manager.disconnect(websocket)

            logger.info("Performance dashboard stopped")

        except Exception as e:
            logger.error(f"Error stopping dashboard: {e}")

    async def _update_loop(self) -> None:
        """Background loop for sending real-time updates via WebSocket."""
        try:
            while self._running:
                try:
                    if (
                        self.websocket_manager
                        and self.websocket_manager.active_connections
                    ):
                        # Get current data
                        update_data = await self._get_websocket_update_data()

                        # Broadcast to all connected clients
                        sent_count = await self.websocket_manager.broadcast(
                            {
                                "type": "dashboard_update",
                                "timestamp": datetime.utcnow().isoformat(),
                                "data": update_data,
                            }
                        )

                        if sent_count > 0:
                            logger.debug(
                                f"Sent dashboard update to {sent_count} WebSocket clients"
                            )

                    # Wait for next update
                    await asyncio.sleep(self.config.update_interval_seconds)

                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in dashboard update loop: {e}")
                    await asyncio.sleep(5)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("Dashboard update loop cancelled")
        except Exception as e:
            logger.error(f"Dashboard update loop failed: {e}")

    async def _get_system_overview(self) -> SystemOverview:
        """Get comprehensive system overview metrics."""
        cache_key = "system_overview"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            overview = SystemOverview()

            # Get tracking status from TrackingManager
            tracking_status = await self.tracking_manager.get_tracking_status()

            # System health and status
            if tracking_status:
                overview.system_health_score = tracking_status.get(
                    "overall_health_score", 0.0
                )
                overview.system_status = tracking_status.get("status", "unknown")

                # Prediction metrics
                accuracy_summary = tracking_status.get("accuracy_summary", {})
                overview.total_predictions_24h = accuracy_summary.get(
                    "total_predictions_24h", 0
                )
                overview.accuracy_rate_24h = accuracy_summary.get(
                    "accuracy_rate_24h", 0.0
                )
                overview.mean_error_minutes_24h = accuracy_summary.get(
                    "mean_error_minutes_24h", 0.0
                )

                # Activity metrics
                overview.active_rooms = len(tracking_status.get("tracked_rooms", []))
                overview.active_models = len(tracking_status.get("tracked_models", []))
                overview.predictions_per_hour = accuracy_summary.get(
                    "predictions_per_hour", 0.0
                )

                # Alerts
                alerts_summary = tracking_status.get("alerts_summary", {})
                overview.active_alerts = alerts_summary.get("total_active", 0)
                overview.critical_alerts = alerts_summary.get("critical_count", 0)
                overview.warnings = alerts_summary.get("warning_count", 0)

                # Drift and retraining
                drift_summary = tracking_status.get("drift_summary", {})
                overview.rooms_with_drift = drift_summary.get("rooms_with_drift", 0)

                retraining_summary = tracking_status.get("retraining_summary", {})
                overview.active_retraining_tasks = retraining_summary.get(
                    "active_tasks", 0
                )
                overview.completed_retraining_24h = retraining_summary.get(
                    "completed_24h", 0
                )

                # Performance indicators
                performance = tracking_status.get("performance_metrics", {})
                overview.avg_prediction_latency_ms = performance.get(
                    "avg_prediction_latency_ms", 0.0
                )
                overview.avg_validation_lag_minutes = performance.get(
                    "avg_validation_lag_minutes", 0.0
                )
                overview.cache_hit_rate = performance.get("cache_hit_rate", 0.0)

            # Dashboard-specific metrics
            overview.uptime_hours = (
                datetime.utcnow() - self._dashboard_start_time
            ).total_seconds() / 3600
            overview.last_updated = datetime.utcnow()

            # Cache the result
            self._cache_data(cache_key, overview)

            return overview

        except Exception as e:
            logger.error(f"Error generating system overview: {e}")
            # Return empty overview with error status
            overview = SystemOverview()
            overview.system_status = "error"
            overview.last_updated = datetime.utcnow()
            return overview

    async def _get_accuracy_dashboard_data(
        self, room_id: Optional[str], model_type: Optional[str], hours_back: int
    ) -> Dict[str, Any]:
        """Get accuracy metrics formatted for dashboard display."""
        cache_key = f"accuracy_{room_id or 'all'}_{model_type or 'all'}_{hours_back}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get real-time metrics from tracking manager
            real_time_metrics = await self.tracking_manager.get_real_time_metrics(
                room_id, model_type
            )

            # Get detailed accuracy metrics from validator
            if (
                hasattr(self.tracking_manager, "validator")
                and self.tracking_manager.validator
            ):
                accuracy_metrics = (
                    await self.tracking_manager.validator.get_accuracy_metrics(
                        room_id=room_id, model_type=model_type, hours_back=hours_back
                    )
                )
            else:
                accuracy_metrics = None

            dashboard_data = {
                "real_time_metrics": [],
                "accuracy_summary": {},
                "error_distribution": {},
                "confidence_analysis": {},
                "room_breakdown": {},
                "model_breakdown": {},
                "time_period": {
                    "hours_back": hours_back,
                    "start_time": (
                        datetime.utcnow() - timedelta(hours=hours_back)
                    ).isoformat(),
                    "end_time": datetime.utcnow().isoformat(),
                },
            }

            # Process real-time metrics
            if real_time_metrics:
                if isinstance(real_time_metrics, list):
                    dashboard_data["real_time_metrics"] = [
                        metric.to_dict() if hasattr(metric, "to_dict") else metric
                        for metric in real_time_metrics
                    ]
                else:
                    dashboard_data["real_time_metrics"] = [
                        (
                            real_time_metrics.to_dict()
                            if hasattr(real_time_metrics, "to_dict")
                            else real_time_metrics
                        )
                    ]

            # Process accuracy metrics
            if accuracy_metrics:
                accuracy_dict = (
                    accuracy_metrics.to_dict()
                    if hasattr(accuracy_metrics, "to_dict")
                    else accuracy_metrics
                )

                dashboard_data["accuracy_summary"] = {
                    "total_predictions": accuracy_dict.get("total_predictions", 0),
                    "validated_predictions": accuracy_dict.get(
                        "validated_predictions", 0
                    ),
                    "accuracy_rate": accuracy_dict.get("accuracy_rate", 0.0),
                    "mean_error_minutes": accuracy_dict.get("mean_error_minutes", 0.0),
                    "median_error_minutes": accuracy_dict.get(
                        "median_error_minutes", 0.0
                    ),
                    "validation_rate": accuracy_dict.get("validation_rate", 0.0),
                }

                dashboard_data["error_distribution"] = accuracy_dict.get(
                    "error_percentiles", {}
                )
                dashboard_data["confidence_analysis"] = {
                    "mean_confidence": accuracy_dict.get("mean_confidence", 0.0),
                    "confidence_calibration_score": accuracy_dict.get(
                        "confidence_calibration_score", 0.0
                    ),
                    "overconfidence_rate": accuracy_dict.get(
                        "overconfidence_rate", 0.0
                    ),
                }

            # Cache and return
            self._cache_data(cache_key, dashboard_data)
            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting accuracy dashboard data: {e}")
            return {
                "error": str(e),
                "real_time_metrics": [],
                "accuracy_summary": {},
                "time_period": {"hours_back": hours_back},
            }

    async def _get_drift_dashboard_data(self, room_id: Optional[str]) -> Dict[str, Any]:
        """Get drift detection data formatted for dashboard display."""
        cache_key = f"drift_{room_id or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get drift status from tracking manager
            drift_status = await self.tracking_manager.get_drift_status()

            dashboard_data = {
                "drift_summary": {
                    "total_rooms_monitored": 0,
                    "rooms_with_drift": 0,
                    "rooms_with_major_drift": 0,
                    "last_check_time": None,
                },
                "drift_by_room": {},
                "drift_types_detected": {},
                "recent_drift_events": [],
                "drift_configuration": {},
            }

            if drift_status:
                # Overall drift summary
                summary = drift_status.get("summary", {})
                dashboard_data["drift_summary"] = {
                    "total_rooms_monitored": summary.get("monitored_rooms", 0),
                    "rooms_with_drift": summary.get("rooms_with_drift", 0),
                    "rooms_with_major_drift": summary.get("rooms_with_major_drift", 0),
                    "last_check_time": summary.get("last_check_time"),
                    "next_check_time": summary.get("next_check_time"),
                }

                # Room-specific drift information
                room_drift_data = drift_status.get("room_drift_data", {})
                if room_id and room_id in room_drift_data:
                    dashboard_data["drift_by_room"] = {
                        room_id: room_drift_data[room_id]
                    }
                else:
                    dashboard_data["drift_by_room"] = room_drift_data

                # Drift types summary
                dashboard_data["drift_types_detected"] = drift_status.get(
                    "drift_types_summary", {}
                )

                # Recent drift events
                dashboard_data["recent_drift_events"] = drift_status.get(
                    "recent_events", []
                )

                # Configuration
                dashboard_data["drift_configuration"] = drift_status.get(
                    "configuration", {}
                )

            # Cache and return
            self._cache_data(cache_key, dashboard_data)
            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting drift dashboard data: {e}")
            return {
                "error": str(e),
                "drift_summary": {},
                "drift_by_room": {},
                "drift_types_detected": {},
                "recent_drift_events": [],
            }

    async def _get_retraining_dashboard_data(self) -> Dict[str, Any]:
        """Get retraining status data formatted for dashboard display."""
        cache_key = "retraining_status"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get retraining status from tracking manager
            retraining_status = await self.tracking_manager.get_retraining_status()

            dashboard_data = {
                "queue_summary": {
                    "pending_requests": 0,
                    "active_retraining": 0,
                    "completed_today": 0,
                    "failed_today": 0,
                },
                "active_retraining_tasks": [],
                "recent_completions": [],
                "retraining_history": [],
                "performance_improvements": {},
                "retrainer_configuration": {},
            }

            if retraining_status:
                # Queue summary
                queue_summary = retraining_status.get("queue_summary", {})
                dashboard_data["queue_summary"] = {
                    "pending_requests": queue_summary.get("pending_count", 0),
                    "active_retraining": queue_summary.get("active_count", 0),
                    "completed_today": queue_summary.get("completed_today", 0),
                    "failed_today": queue_summary.get("failed_today", 0),
                    "queue_size": queue_summary.get("total_queue_size", 0),
                }

                # Active tasks
                dashboard_data["active_retraining_tasks"] = retraining_status.get(
                    "active_tasks", []
                )

                # Recent completions
                dashboard_data["recent_completions"] = retraining_status.get(
                    "recent_completions", []
                )

                # History
                dashboard_data["retraining_history"] = retraining_status.get(
                    "history", []
                )

                # Performance improvements
                dashboard_data["performance_improvements"] = retraining_status.get(
                    "performance_improvements", {}
                )

                # Configuration
                dashboard_data["retrainer_configuration"] = retraining_status.get(
                    "configuration", {}
                )

            # Cache and return
            self._cache_data(cache_key, dashboard_data)
            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting retraining dashboard data: {e}")
            return {
                "error": str(e),
                "queue_summary": {},
                "active_retraining_tasks": [],
                "recent_completions": [],
            }

    async def _get_system_health_data(self) -> Dict[str, Any]:
        """Get detailed system health information."""
        cache_key = "system_health"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get comprehensive health data
            tracking_status = await self.tracking_manager.get_tracking_status()

            health_data = {
                "overall_health": {
                    "score": 0.0,
                    "status": "unknown",
                    "uptime_hours": (
                        datetime.utcnow() - self._dashboard_start_time
                    ).total_seconds()
                    / 3600,
                },
                "component_health": {
                    "prediction_validator": {"status": "unknown", "health_score": 0.0},
                    "accuracy_tracker": {"status": "unknown", "health_score": 0.0},
                    "drift_detector": {"status": "unknown", "health_score": 0.0},
                    "adaptive_retrainer": {"status": "unknown", "health_score": 0.0},
                    "database": {"status": "unknown", "health_score": 0.0},
                },
                "resource_usage": {
                    "memory_usage_mb": 0.0,
                    "cache_usage_percent": 0.0,
                    "active_connections": 0,
                    "background_tasks": 0,
                },
                "performance_metrics": {
                    "avg_response_time_ms": 0.0,
                    "requests_per_minute": 0.0,
                    "error_rate_percent": 0.0,
                    "cache_hit_rate_percent": 0.0,
                },
            }

            if tracking_status:
                # Overall health
                health_data["overall_health"] = {
                    "score": tracking_status.get("overall_health_score", 0.0),
                    "status": tracking_status.get("status", "unknown"),
                    "uptime_hours": (
                        datetime.utcnow() - self._dashboard_start_time
                    ).total_seconds()
                    / 3600,
                    "last_updated": datetime.utcnow().isoformat(),
                }

                # Component health
                components = tracking_status.get("component_status", {})
                for component, status in components.items():
                    if component in health_data["component_health"]:
                        health_data["component_health"][component] = status

                # Resource usage
                resources = tracking_status.get("resource_usage", {})
                health_data["resource_usage"].update(resources)

                # Performance metrics
                performance = tracking_status.get("performance_metrics", {})
                health_data["performance_metrics"].update(performance)

            # Dashboard-specific metrics
            if self.websocket_manager:
                ws_stats = self.websocket_manager.get_connection_stats()
                health_data["resource_usage"]["websocket_connections"] = ws_stats[
                    "active_connections"
                ]
                health_data["resource_usage"]["websocket_capacity"] = ws_stats[
                    "max_connections"
                ]

            # Cache usage
            with self._cache_lock:
                cache_count = len(self._cache)
                health_data["resource_usage"]["dashboard_cache_entries"] = cache_count

            # Cache and return
            self._cache_data(cache_key, health_data)
            return health_data

        except Exception as e:
            logger.error(f"Error getting system health data: {e}")
            return {
                "error": str(e),
                "overall_health": {"status": "error", "score": 0.0},
                "component_health": {},
                "resource_usage": {},
                "performance_metrics": {},
            }

    async def _get_alerts_dashboard_data(
        self, severity: Optional[str], room_id: Optional[str]
    ) -> Dict[str, Any]:
        """Get active alerts data formatted for dashboard display."""
        cache_key = f"alerts_{severity or 'all'}_{room_id or 'all'}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get alerts from tracking manager
            active_alerts = await self.tracking_manager.get_active_alerts(
                room_id, severity
            )

            dashboard_data = {
                "alert_summary": {
                    "total_active": 0,
                    "critical_count": 0,
                    "warning_count": 0,
                    "info_count": 0,
                },
                "active_alerts": [],
                "alerts_by_room": {},
                "alerts_by_type": {},
                "recent_escalations": [],
                "alert_trends": {},
            }

            if active_alerts:
                # Process alerts
                alerts_list = (
                    active_alerts
                    if isinstance(active_alerts, list)
                    else [active_alerts]
                )

                critical_count = 0
                warning_count = 0
                info_count = 0
                alerts_by_room = {}
                alerts_by_type = {}

                processed_alerts = []
                for alert in alerts_list:
                    alert_dict = alert.to_dict() if hasattr(alert, "to_dict") else alert
                    processed_alerts.append(alert_dict)

                    # Count by severity
                    severity_level = alert_dict.get("severity", "info")
                    if severity_level == "critical":
                        critical_count += 1
                    elif severity_level == "warning":
                        warning_count += 1
                    else:
                        info_count += 1

                    # Group by room
                    alert_room = alert_dict.get("room_id", "unknown")
                    if alert_room not in alerts_by_room:
                        alerts_by_room[alert_room] = []
                    alerts_by_room[alert_room].append(alert_dict)

                    # Group by type
                    alert_type = alert_dict.get("alert_type", "unknown")
                    if alert_type not in alerts_by_type:
                        alerts_by_type[alert_type] = 0
                    alerts_by_type[alert_type] += 1

                dashboard_data["alert_summary"] = {
                    "total_active": len(processed_alerts),
                    "critical_count": critical_count,
                    "warning_count": warning_count,
                    "info_count": info_count,
                }

                dashboard_data["active_alerts"] = processed_alerts
                dashboard_data["alerts_by_room"] = alerts_by_room
                dashboard_data["alerts_by_type"] = alerts_by_type

            # Cache and return
            self._cache_data(cache_key, dashboard_data)
            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting alerts dashboard data: {e}")
            return {
                "error": str(e),
                "alert_summary": {},
                "active_alerts": [],
                "alerts_by_room": {},
                "alerts_by_type": {},
            }

    async def _get_trends_dashboard_data(
        self, room_id: Optional[str], days_back: int
    ) -> Dict[str, Any]:
        """Get historical trends data for visualization."""
        cache_key = f"trends_{room_id or 'all'}_{days_back}"
        cached_data = self._get_cached_data(cache_key)
        if cached_data:
            return cached_data

        try:
            # Get accuracy trends from tracking manager
            trends_data = (
                await self.tracking_manager.accuracy_tracker.get_accuracy_trends(
                    room_id=room_id, days_back=days_back
                )
            )

            dashboard_data = {
                "time_period": {
                    "days_back": days_back,
                    "start_date": (
                        datetime.utcnow() - timedelta(days=days_back)
                    ).isoformat(),
                    "end_date": datetime.utcnow().isoformat(),
                },
                "accuracy_trends": [],
                "error_trends": [],
                "prediction_volume_trends": [],
                "trend_analysis": {},
            }

            if trends_data:
                dashboard_data.update(trends_data)

            # Cache and return
            self._cache_data(cache_key, dashboard_data)
            return dashboard_data

        except Exception as e:
            logger.error(f"Error getting trends dashboard data: {e}")
            return {
                "error": str(e),
                "time_period": {"days_back": days_back},
                "accuracy_trends": [],
                "error_trends": [],
                "prediction_volume_trends": [],
            }

    def _get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard system statistics."""
        try:
            uptime_seconds = (
                datetime.utcnow() - self._dashboard_start_time
            ).total_seconds()

            stats = {
                "dashboard_info": {
                    "version": "1.0.0",
                    "mode": self.config.mode.value,
                    "uptime_seconds": uptime_seconds,
                    "started_at": self._dashboard_start_time.isoformat(),
                },
                "configuration": {
                    "host": self.config.host,
                    "port": self.config.port,
                    "websocket_enabled": self.config.websocket_enabled,
                    "update_interval_seconds": self.config.update_interval_seconds,
                    "cache_ttl_seconds": self.config.cache_ttl_seconds,
                },
                "performance": {
                    "cache_entries": 0,
                    "cache_hit_rate": 0.0,
                    "memory_usage_estimate_mb": 0.0,
                },
            }

            # Cache statistics
            with self._cache_lock:
                stats["performance"]["cache_entries"] = len(self._cache)

            # WebSocket statistics
            if self.websocket_manager:
                ws_stats = self.websocket_manager.get_connection_stats()
                stats["websocket"] = ws_stats

            return stats

        except Exception as e:
            logger.error(f"Error getting dashboard stats: {e}")
            return {"error": str(e)}

    async def _get_websocket_initial_data(self) -> Dict[str, Any]:
        """Get initial data to send to new WebSocket connections."""
        try:
            overview = await self._get_system_overview()
            return {
                "type": "initial_data",
                "timestamp": datetime.utcnow().isoformat(),
                "data": {
                    "system_overview": overview.to_dict(),
                    "dashboard_info": {
                        "update_interval": self.config.update_interval_seconds,
                        "features_enabled": {
                            "drift_detection": self.config.enable_drift_visualization,
                            "retraining_controls": self.config.enable_retraining_controls,
                            "alert_management": self.config.enable_alert_management,
                            "historical_charts": self.config.enable_historical_charts,
                        },
                    },
                },
            }
        except Exception as e:
            logger.error(f"Error getting WebSocket initial data: {e}")
            return {
                "type": "error",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e),
            }

    async def _get_websocket_update_data(self) -> Dict[str, Any]:
        """Get update data for WebSocket broadcasting."""
        try:
            # Get key metrics for real-time updates
            overview = await self._get_system_overview()

            # Get latest alerts (if any)
            recent_alerts = await self._get_alerts_dashboard_data(
                severity=None, room_id=None
            )

            return {
                "system_overview": overview.to_dict(),
                "alert_summary": recent_alerts.get("alert_summary", {}),
                "last_updated": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            logger.error(f"Error getting WebSocket update data: {e}")
            return {"error": str(e), "last_updated": datetime.utcnow().isoformat()}

    async def _handle_websocket_message(
        self, message: str, websocket: WebSocket
    ) -> None:
        """Handle incoming WebSocket messages from clients."""
        try:
            data = json.loads(message)
            message_type = data.get("type", "unknown")

            if message_type == "ping":
                # Respond to ping with pong
                await self.websocket_manager.send_personal_message(
                    {"type": "pong", "timestamp": datetime.utcnow().isoformat()},
                    websocket,
                )
            elif message_type == "subscribe":
                # Handle subscription changes (if needed)
                pass
            elif message_type == "request_data":
                # Handle specific data requests
                requested_data = data.get("data_type", "overview")
                response_data = await self._get_requested_data(requested_data)
                await self.websocket_manager.send_personal_message(
                    {
                        "type": "data_response",
                        "data_type": requested_data,
                        "data": response_data,
                        "timestamp": datetime.utcnow().isoformat(),
                    },
                    websocket,
                )

        except json.JSONDecodeError:
            logger.warning(f"Invalid JSON received from WebSocket client: {message}")
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")

    async def _get_requested_data(self, data_type: str) -> Dict[str, Any]:
        """Get specific data type requested by WebSocket client."""
        try:
            if data_type == "overview":
                overview = await self._get_system_overview()
                return overview.to_dict()
            elif data_type == "accuracy":
                return await self._get_accuracy_dashboard_data(None, None, 24)
            elif data_type == "drift":
                return await self._get_drift_dashboard_data(None)
            elif data_type == "retraining":
                return await self._get_retraining_dashboard_data()
            elif data_type == "alerts":
                return await self._get_alerts_dashboard_data(None, None)
            else:
                return {"error": f"Unknown data type: {data_type}"}
        except Exception as e:
            logger.error(f"Error getting requested data '{data_type}': {e}")
            return {"error": str(e)}

    async def _trigger_manual_retraining(self, request_data: dict) -> Dict[str, Any]:
        """Trigger manual retraining request."""
        try:
            room_id = request_data.get("room_id")
            model_type = request_data.get("model_type")
            strategy = request_data.get("strategy", "full_retrain")

            if not room_id:
                raise ValueError("room_id is required for manual retraining")

            # Use tracking manager to request retraining
            result = await self.tracking_manager.request_manual_retraining(
                room_id=room_id, model_type=model_type, strategy=strategy
            )

            logger.info(
                f"Manual retraining requested for {room_id}/{model_type}: {result}"
            )
            return {
                "success": True,
                "message": "Retraining request submitted successfully",
                "request_id": result.get("request_id"),
                "estimated_completion": result.get("estimated_completion"),
            }

        except Exception as e:
            logger.error(f"Error triggering manual retraining: {e}")
            return {"success": False, "error": str(e)}

    async def _acknowledge_alert(self, alert_data: dict) -> Dict[str, Any]:
        """Acknowledge an active alert."""
        try:
            alert_id = alert_data.get("alert_id")
            user_id = alert_data.get("user_id", "dashboard_user")

            if not alert_id:
                raise ValueError("alert_id is required to acknowledge alert")

            # Use tracking manager to acknowledge alert
            result = await self.tracking_manager.acknowledge_alert(alert_id, user_id)

            logger.info(f"Alert {alert_id} acknowledged by {user_id}")
            return {
                "success": True,
                "message": "Alert acknowledged successfully",
                "alert_id": alert_id,
                "acknowledged_by": user_id,
                "acknowledged_at": datetime.utcnow().isoformat(),
            }

        except Exception as e:
            logger.error(f"Error acknowledging alert: {e}")
            return {"success": False, "error": str(e)}

    def _get_cached_data(self, cache_key: str) -> Optional[Any]:
        """Get data from cache if valid."""
        with self._cache_lock:
            if cache_key in self._cache:
                data, cached_at = self._cache[cache_key]
                if datetime.utcnow() - cached_at < timedelta(
                    seconds=self.config.cache_ttl_seconds
                ):
                    return data
                else:
                    # Remove expired cache entry
                    del self._cache[cache_key]
        return None

    def _cache_data(self, cache_key: str, data: Any) -> None:
        """Cache data with timestamp."""
        with self._cache_lock:
            self._cache[cache_key] = (data, datetime.utcnow())

            # Limit cache size
            if len(self._cache) > 100:
                # Remove oldest entries
                sorted_cache = sorted(self._cache.items(), key=lambda x: x[1][1])
                for key, _ in sorted_cache[:10]:  # Remove oldest 10
                    del self._cache[key]


class DashboardError(OccupancyPredictionError):
    """Custom exception for dashboard operation failures."""

    def __init__(
        self,
        message: str,
        error_context: Optional[Dict[str, Any]] = None,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
    ):
        super().__init__(
            message=message,
            error_code="DASHBOARD_ERROR",
            error_context=error_context,
            severity=severity,
        )


# Example usage and testing functions
async def create_dashboard_from_tracking_manager(
    tracking_manager: "TrackingManager",
    host: str = "0.0.0.0",
    port: int = 8888,
    debug: bool = False,
) -> PerformanceDashboard:
    """
    Create and configure dashboard from existing tracking manager.

    This is the primary integration point - pass your initialized
    TrackingManager and get a ready-to-use dashboard.
    """
    config = DashboardConfig(
        host=host,
        port=port,
        debug=debug,
        websocket_enabled=True,
        update_interval_seconds=5,
    )

    dashboard = PerformanceDashboard(tracking_manager, config)
    await dashboard.start_dashboard()

    logger.info(f"Dashboard created and started on http://{host}:{port}")
    return dashboard


# Integration helper for main system initialization
def integrate_dashboard_with_tracking_system(
    tracking_manager: "TrackingManager",
    dashboard_config: Optional[DashboardConfig] = None,
) -> PerformanceDashboard:
    """
    Integrate dashboard with existing tracking system.

    This function provides seamless integration - just pass your
    TrackingManager and optionally a config, and get a configured dashboard.
    """
    if not dashboard_config:
        dashboard_config = DashboardConfig()

    dashboard = PerformanceDashboard(tracking_manager, dashboard_config)

    logger.info("Dashboard integrated with tracking system - ready to start")
    return dashboard
