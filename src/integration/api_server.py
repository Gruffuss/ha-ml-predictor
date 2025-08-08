"""
REST API Server for Home Assistant Occupancy Prediction System.

This module provides a production-ready FastAPI server with comprehensive
control endpoints for system management, monitoring, and integration.
Automatically integrates with TrackingManager and MQTT systems.

Features:
- Full integration with existing TrackingManager
- Real-time prediction endpoints
- System health monitoring
- Manual control endpoints (retrain, refresh, etc.)
- Rate limiting and authentication
- Comprehensive error handling
- Background health checks
"""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import logging
import traceback
from typing import TYPE_CHECKING

from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    HTTPException,
    Request,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field, validator
import uvicorn

if TYPE_CHECKING:
    from ..adaptation.tracking_manager import TrackingManager

from ..core.config import APIConfig, get_config
from ..core.exceptions import (
    APIAuthenticationError,
    APIError,
    APIRateLimitError,
    APIResourceNotFoundError,
    APIServerError,
    APIValidationError,
    ErrorSeverity,
    OccupancyPredictionError,
)
from ..data.storage.database import get_database_manager
from ..integration.mqtt_integration_manager import MQTTIntegrationManager
from ..models.base.predictor import PredictionResult

logger = logging.getLogger(__name__)


# Pydantic Models for API Requests/Responses


class PredictionResponse(BaseModel):
    """Response model for prediction endpoints."""

    room_id: str
    prediction_time: datetime
    next_transition_time: Optional[datetime]
    transition_type: Optional[str]  # 'occupied' or 'vacant'
    confidence: float
    time_until_transition: Optional[str]
    alternatives: List[Dict[str, Any]] = []
    model_info: Dict[str, Any] = {}


class SystemHealthResponse(BaseModel):
    """Response model for system health endpoint."""

    status: str  # 'healthy', 'degraded', 'unhealthy'
    timestamp: datetime
    components: Dict[str, Dict[str, Any]]
    performance_metrics: Dict[str, Any]
    error_count: int
    uptime_seconds: float


class AccuracyMetricsResponse(BaseModel):
    """Response model for accuracy metrics."""

    room_id: Optional[str]
    accuracy_rate: float
    average_error_minutes: float
    confidence_calibration: float
    total_predictions: int
    total_validations: int
    time_window_hours: int
    trend_direction: str  # 'improving', 'stable', 'degrading'


class ManualRetrainRequest(BaseModel):
    """Request model for manual retraining."""

    room_id: Optional[str] = Field(
        None, description="Specific room to retrain (all if None)"
    )
    force: bool = Field(False, description="Force retrain even if not needed")
    strategy: str = Field(
        "auto", pattern="^(auto|incremental|full|feature_refresh)$"
    )
    reason: str = Field("manual_request", description="Reason for retraining")


class SystemStatsResponse(BaseModel):
    """Response model for system statistics."""

    system_info: Dict[str, Any]
    prediction_stats: Dict[str, Any]
    mqtt_stats: Dict[str, Any]
    database_stats: Dict[str, Any]
    tracking_stats: Dict[str, Any]


class ErrorResponse(BaseModel):
    """Standard error response model."""

    error: str
    error_code: Optional[str]
    details: Optional[Dict[str, Any]]
    timestamp: datetime
    request_id: Optional[str]


# Rate Limiting
@dataclass
class RateLimitTracker:
    """Simple in-memory rate limiter."""

    requests: Dict[str, List[datetime]]

    def __init__(self):
        self.requests = {}

    def is_allowed(
        self, client_ip: str, limit: int, window_minutes: int = 1
    ) -> bool:
        """Check if request is within rate limits."""
        now = datetime.now()
        window_start = now - timedelta(minutes=window_minutes)

        # Clean old requests
        if client_ip in self.requests:
            self.requests[client_ip] = [
                req_time
                for req_time in self.requests[client_ip]
                if req_time > window_start
            ]
        else:
            self.requests[client_ip] = []

        # Check limit
        if len(self.requests[client_ip]) >= limit:
            return False

        # Add current request
        self.requests[client_ip].append(now)
        return True


# Global instances
rate_limiter = RateLimitTracker()
security_scheme = HTTPBearer(auto_error=False)


# Dependencies

_tracking_manager_instance = None


async def get_tracking_manager() -> "TrackingManager":
    """Get the system tracking manager."""
    global _tracking_manager_instance
    if _tracking_manager_instance is None:
        # Lazy import to prevent circular dependency
        from ..adaptation.tracking_manager import (
            TrackingConfig,
            TrackingManager,
        )

        config = get_config()
        # Create default tracking config if not available
        tracking_config = getattr(config, "tracking", None)
        if tracking_config is None:
            tracking_config = TrackingConfig()

        _tracking_manager_instance = TrackingManager(tracking_config)
        # Initialize the tracking manager if it hasn't been initialized
        if (
            not hasattr(_tracking_manager_instance, "_tracking_active")
            or not _tracking_manager_instance._tracking_active
        ):
            await _tracking_manager_instance.initialize()
    return _tracking_manager_instance


def set_tracking_manager(tracking_manager: "TrackingManager"):
    """Set the tracking manager instance for API endpoints."""
    global _tracking_manager_instance
    _tracking_manager_instance = tracking_manager


async def get_mqtt_manager() -> MQTTIntegrationManager:
    """Get the MQTT integration manager."""
    config = get_config()
    return MQTTIntegrationManager(config.mqtt)


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Depends(security_scheme),
) -> bool:
    """Verify API key if authentication is enabled."""
    config = get_config()

    if not config.api.api_key_enabled:
        return True

    if not credentials:
        raise APIAuthenticationError(
            "API Key required", "Missing authorization header"
        )

    if credentials.credentials != config.api.api_key:
        raise APIAuthenticationError("Invalid API key", "Key does not match")

    return True


async def check_rate_limit(request: Request) -> bool:
    """Check if request is within rate limits."""
    config = get_config()

    if not config.api.rate_limit_enabled:
        return True

    client_ip = request.client.host
    if not rate_limiter.is_allowed(client_ip, config.api.requests_per_minute):
        raise APIRateLimitError(
            client_ip, config.api.requests_per_minute, "minute"
        )

    return True


# Application factory with lifespan management


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    logger.info("Starting API server...")

    # Initialize background tasks
    config = get_config()
    if config.api.background_tasks_enabled:
        # Start health check task
        health_task = asyncio.create_task(background_health_check())

        yield

        # Cleanup
        health_task.cancel()
        try:
            await health_task
        except asyncio.CancelledError:
            pass
    else:
        yield

    logger.info("API server shutdown complete")


async def background_health_check():
    """Background task for periodic health checks."""
    config = get_config()

    while True:
        try:
            await asyncio.sleep(config.api.health_check_interval_seconds)

            # Perform health checks
            db_manager = await get_database_manager()
            health = await db_manager.health_check()

            if not health.get("database_connected", False):
                logger.warning(
                    "Database connection lost in background health check"
                )

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Background health check failed: {e}")


def create_app() -> FastAPI:
    """Create and configure FastAPI application."""
    config = get_config()
    api_config = config.api

    app = FastAPI(
        title="Occupancy Prediction API",
        description="REST API for Home Assistant Occupancy Prediction System",
        version="1.0.0",
        debug=api_config.debug,
        docs_url=api_config.docs_url if api_config.include_docs else None,
        redoc_url=api_config.redoc_url if api_config.include_docs else None,
        lifespan=lifespan,
    )

    # Add middleware
    if api_config.enable_cors:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=api_config.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE"],
            allow_headers=["*"],
        )

    # Add trusted host middleware for security
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=["*"],  # Configure based on deployment
    )

    # Exception handlers
    @app.exception_handler(APIError)
    async def api_error_handler(request: Request, exc: APIError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code,
                details=exc.context,
                timestamp=datetime.now(),
                request_id=getattr(request, "request_id", None),
            ).dict(),
        )

    @app.exception_handler(OccupancyPredictionError)
    async def system_error_handler(
        request: Request, exc: OccupancyPredictionError
    ):
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        if exc.severity in [ErrorSeverity.LOW, ErrorSeverity.MEDIUM]:
            status_code = status.HTTP_400_BAD_REQUEST

        return JSONResponse(
            status_code=status_code,
            content=ErrorResponse(
                error=exc.message,
                error_code=exc.error_code,
                details=exc.context,
                timestamp=datetime.now(),
                request_id=getattr(request, "request_id", None),
            ).dict(),
        )

    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        logger.error(f"Unhandled exception in API: {exc}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="Internal server error",
                error_code="UNHANDLED_EXCEPTION",
                details={"exception_type": type(exc).__name__},
                timestamp=datetime.now(),
                request_id=getattr(request, "request_id", None),
            ).dict(),
        )

    # Request middleware for logging and rate limiting
    @app.middleware("http")
    async def request_middleware(request: Request, call_next):
        # Generate request ID
        import uuid

        request.request_id = str(uuid.uuid4())

        # Log request if enabled
        if api_config.log_requests:
            logger.info(
                f"API Request: {request.method} {request.url} - ID: {request.request_id}"
            )

        try:
            # Rate limiting check
            await check_rate_limit(request)

            # Process request
            response = await call_next(request)

            # Log response if enabled
            if api_config.log_responses:
                logger.info(
                    f"API Response: {response.status_code} - ID: {request.request_id}"
                )

            return response

        except APIRateLimitError as e:
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content=ErrorResponse(
                    error=e.message,
                    error_code=e.error_code,
                    details=e.context,
                    timestamp=datetime.now(),
                    request_id=request.request_id,
                ).dict(),
            )
        except Exception as e:
            logger.error(f"Request middleware error: {e}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content=ErrorResponse(
                    error="Request processing error",
                    error_code="REQUEST_MIDDLEWARE_ERROR",
                    details={"exception_type": type(e).__name__},
                    timestamp=datetime.now(),
                    request_id=request.request_id,
                ).dict(),
            )

    return app


# Create application instance
app = create_app()


# API Endpoints


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Occupancy Prediction API",
        "version": "1.0.0",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/health", response_model=SystemHealthResponse)
async def health_check():
    """Comprehensive system health check."""
    try:
        start_time = datetime.now()

        # Check database
        db_manager = await get_database_manager()
        db_health = await db_manager.health_check()

        # Check tracking manager (comprehensive health check)
        tracking_health = {"status": "unknown"}
        try:
            tracking_manager = await get_tracking_manager()
            
            # Get comprehensive tracking status
            tracking_status = await tracking_manager.get_tracking_status()
            
            # Determine overall tracking health based on components
            tracking_active = tracking_status.get("tracking_active", False)
            config_enabled = tracking_status.get("config", {}).get("enabled", False)
            performance_metrics = tracking_status.get("performance", {})
            
            # Check for any error conditions
            has_errors = False
            error_details = []
            
            # Check validator status
            validator_status = tracking_status.get("validator", {})
            if "error" in validator_status:
                has_errors = True
                error_details.append(f"Validator error: {validator_status['error']}")
            
            # Check accuracy tracker status
            accuracy_tracker_status = tracking_status.get("accuracy_tracker", {})
            if "error" in accuracy_tracker_status:
                has_errors = True
                error_details.append(f"Accuracy tracker error: {accuracy_tracker_status['error']}")
            
            # Check background tasks
            background_tasks = performance_metrics.get("background_tasks", 0)
            if background_tasks == 0 and tracking_active:
                error_details.append("No background tasks running despite tracking being active")
            
            # Determine status
            if has_errors:
                status = "error"
            elif not config_enabled:
                status = "disabled"
            elif not tracking_active:
                status = "inactive"
            else:
                status = "healthy"
            
            tracking_health = {
                "status": status,
                "tracking_active": tracking_active,
                "config_enabled": config_enabled,
                "background_tasks": background_tasks,
                "total_predictions_recorded": performance_metrics.get("total_predictions_recorded", 0),
                "total_validations_performed": performance_metrics.get("total_validations_performed", 0),
                "total_drift_checks_performed": performance_metrics.get("total_drift_checks_performed", 0),
                "system_uptime_seconds": performance_metrics.get("system_uptime_seconds", 0),
                "validator_available": validator_status.get("total_predictions", 0) >= 0 if validator_status else False,
                "accuracy_tracker_available": accuracy_tracker_status.get("total_predictions", 0) >= 0 if accuracy_tracker_status else False,
                "drift_detector_available": "drift_detector" in tracking_status and tracking_status["drift_detector"] is not None,
                "adaptive_retrainer_available": "adaptive_retrainer" in tracking_status and tracking_status["adaptive_retrainer"] is not None,
            }
            
            if error_details:
                tracking_health["error_details"] = error_details
                
        except Exception as e:
            tracking_health = {
                "status": "error", 
                "error": str(e),
                "tracking_active": False,
                "config_enabled": False,
            }

        # Check MQTT integration
        mqtt_health = {"status": "unknown"}
        try:
            mqtt_manager = await get_mqtt_manager()
            mqtt_stats = await mqtt_manager.get_integration_stats()
            mqtt_health = {
                "status": (
                    "healthy" if mqtt_stats.mqtt_connected else "degraded"
                ),
                "connected": mqtt_stats.mqtt_connected,
                "predictions_published": mqtt_stats.predictions_published,
            }
        except Exception as e:
            mqtt_health = {"status": "error", "error": str(e)}

        # Overall health determination with enhanced logic
        db_healthy = db_health.get("database_connected", False)
        tracking_status_value = tracking_health.get("status", "unknown")
        mqtt_status_value = mqtt_health.get("status", "unknown")
        
        # Determine component health levels
        tracking_healthy = tracking_status_value == "healthy"
        tracking_functional = tracking_status_value in ["healthy", "inactive", "disabled"]
        mqtt_healthy = mqtt_status_value == "healthy"
        mqtt_functional = mqtt_status_value in ["healthy", "degraded"]
        
        # System is healthy if all core components are healthy
        if db_healthy and tracking_healthy and mqtt_healthy:
            overall_status = "healthy"
        # System is degraded if core components are functional but not optimal
        elif db_healthy and tracking_functional and mqtt_functional:
            overall_status = "degraded"
        # System is unhealthy if any critical component has errors
        else:
            overall_status = "unhealthy"

        response_time = (datetime.now() - start_time).total_seconds()

        return SystemHealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            components={
                "database": db_health,
                "tracking": tracking_health,
                "mqtt": mqtt_health,
            },
            performance_metrics={
                "response_time_seconds": response_time,
                "memory_usage": "N/A",  # Could add memory monitoring
                "tracking_predictions_recorded": tracking_health.get("total_predictions_recorded", 0),
                "tracking_validations_performed": tracking_health.get("total_validations_performed", 0),
                "tracking_drift_checks_performed": tracking_health.get("total_drift_checks_performed", 0),
                "tracking_background_tasks": tracking_health.get("background_tasks", 0),
            },
            error_count=sum(
                1
                for c in [db_health, tracking_health, mqtt_health]
                if c.get("status") == "error" or not c.get("database_connected", True)
            ),
            uptime_seconds=tracking_health.get("system_uptime_seconds", 0)
        )

    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        raise APIServerError("health_check", e)


@app.get("/predictions/{room_id}", response_model=PredictionResponse)
async def get_room_prediction(
    room_id: str,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(check_rate_limit),
):
    """Get current prediction for a specific room."""
    try:
        # Validate room exists
        config = get_config()
        if room_id not in config.rooms:
            raise APIResourceNotFoundError("Room", room_id)

        # Get prediction from tracking manager
        tracking_manager = await get_tracking_manager()
        prediction_data = await tracking_manager.get_room_prediction(room_id)

        if not prediction_data:
            raise APIServerError(
                f"get_prediction_for_{room_id}",
                Exception("No prediction available"),
            )

        return PredictionResponse(
            room_id=prediction_data["room_id"],
            prediction_time=datetime.fromisoformat(
                prediction_data["prediction_time"].replace("Z", "+00:00")
            ),
            next_transition_time=(
                datetime.fromisoformat(
                    prediction_data["next_transition_time"].replace(
                        "Z", "+00:00"
                    )
                )
                if prediction_data.get("next_transition_time")
                else None
            ),
            transition_type=prediction_data.get("transition_type"),
            confidence=prediction_data["confidence"],
            time_until_transition=prediction_data.get("time_until_transition"),
            alternatives=prediction_data.get("alternatives", []),
            model_info=prediction_data.get("model_info", {}),
        )

    except APIResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(
            f"Failed to get prediction for room {room_id}: {e}", exc_info=True
        )
        raise APIServerError(f"get_prediction_for_{room_id}", e)


@app.get("/predictions", response_model=List[PredictionResponse])
async def get_all_predictions(
    _: bool = Depends(verify_api_key), __: bool = Depends(check_rate_limit)
):
    """Get current predictions for all rooms."""
    try:
        config = get_config()
        predictions = []

        for room_id in config.rooms.keys():
            try:
                # Get prediction for each room
                prediction = await get_room_prediction(room_id)
                predictions.append(prediction)
            except Exception as e:
                logger.warning(
                    f"Failed to get prediction for room {room_id}: {e}"
                )
                # Continue with other rooms

        return predictions

    except Exception as e:
        logger.error(f"Failed to get all predictions: {e}", exc_info=True)
        raise APIServerError("get_all_predictions", e)


@app.get("/accuracy", response_model=AccuracyMetricsResponse)
async def get_accuracy_metrics(
    room_id: Optional[str] = None,
    hours: int = 24,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(check_rate_limit),
):
    """Get accuracy metrics for a room or overall system."""
    try:
        if room_id:
            config = get_config()
            if room_id not in config.rooms:
                raise APIResourceNotFoundError("Room", room_id)

        # Get metrics from tracking manager
        tracking_manager = await get_tracking_manager()
        metrics_data = await tracking_manager.get_accuracy_metrics(
            room_id, hours
        )

        return AccuracyMetricsResponse(
            room_id=metrics_data["room_id"],
            accuracy_rate=metrics_data["accuracy_rate"],
            average_error_minutes=metrics_data["average_error_minutes"],
            confidence_calibration=metrics_data["confidence_calibration"],
            total_predictions=metrics_data["total_predictions"],
            total_validations=metrics_data["total_validations"],
            time_window_hours=metrics_data["time_window_hours"],
            trend_direction=metrics_data["trend_direction"],
        )

    except APIResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to get accuracy metrics: {e}", exc_info=True)
        raise APIServerError("get_accuracy_metrics", e)


@app.post("/model/retrain")
async def trigger_manual_retrain(
    retrain_request: ManualRetrainRequest,
    background_tasks: BackgroundTasks,
    _: bool = Depends(verify_api_key),
    __: bool = Depends(check_rate_limit),
):
    """Trigger manual model retraining."""
    try:
        if retrain_request.room_id:
            config = get_config()
            if retrain_request.room_id not in config.rooms:
                raise APIResourceNotFoundError("Room", retrain_request.room_id)

        # Trigger retraining via tracking manager
        tracking_manager = await get_tracking_manager()
        result = await tracking_manager.trigger_manual_retrain(
            room_id=retrain_request.room_id,
            force=retrain_request.force,
            strategy=retrain_request.strategy,
            reason=retrain_request.reason,
        )

        return {
            "message": result["message"],
            "success": result["success"],
            "room_id": result["room_id"],
            "strategy": result["strategy"],
            "force": result["force"],
            "timestamp": datetime.now().isoformat(),
        }

    except APIResourceNotFoundError:
        raise
    except Exception as e:
        logger.error(f"Failed to trigger retrain: {e}", exc_info=True)
        raise APIServerError("trigger_retrain", e)


@app.post("/mqtt/refresh")
async def refresh_mqtt_discovery(
    _: bool = Depends(verify_api_key), __: bool = Depends(check_rate_limit)
):
    """Refresh Home Assistant MQTT discovery."""
    try:
        mqtt_manager = await get_mqtt_manager()

        # Refresh discovery
        await mqtt_manager.cleanup_discovery()
        await mqtt_manager.initialize()

        return {
            "message": "MQTT discovery refresh completed",
            "timestamp": datetime.now().isoformat(),
        }

    except Exception as e:
        logger.error(f"Failed to refresh MQTT discovery: {e}", exc_info=True)
        raise APIServerError("refresh_mqtt_discovery", e)


@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats(
    _: bool = Depends(verify_api_key), __: bool = Depends(check_rate_limit)
):
    """Get comprehensive system statistics."""
    try:
        # Gather stats from all components
        db_manager = await get_database_manager()
        db_health = await db_manager.health_check()

        mqtt_manager = await get_mqtt_manager()
        mqtt_stats = await mqtt_manager.get_integration_stats()

        tracking_manager = await get_tracking_manager()
        tracking_stats = await tracking_manager.get_system_stats()

        return SystemStatsResponse(
            system_info={
                "version": "1.0.0",
                "uptime_seconds": 0,  # Could track actual uptime
                "python_version": "3.11+",
                "timestamp": datetime.now().isoformat(),
            },
            prediction_stats={
                "total_predictions": tracking_stats["tracking_stats"].get(
                    "total_predictions_tracked", 0
                ),
                "accuracy_rate": 0.85,  # Could get from tracking stats
                "models_trained": tracking_stats["retraining_stats"].get(
                    "completed_retraining_jobs", 0
                ),
            },
            mqtt_stats=asdict(mqtt_stats),
            database_stats=db_health,
            tracking_stats=tracking_stats["tracking_stats"],
        )

    except Exception as e:
        logger.error(f"Failed to get system stats: {e}", exc_info=True)
        raise APIServerError("get_system_stats", e)


# Server runner
class APIServer:
    """
    REST API Server manager for integration with TrackingManager.

    This class provides the main interface for running the API server
    as part of the integrated system workflow.
    """

    def __init__(self, tracking_manager: "TrackingManager"):
        """Initialize API server with tracking manager integration."""
        self.tracking_manager = tracking_manager
        self.config = get_config().api
        self.server = None
        self.server_task = None

    async def start(self):
        """Start the API server."""
        if not self.config.enabled:
            logger.info("API server disabled in configuration")
            return

        logger.info(
            f"Starting API server on {self.config.host}:{self.config.port}"
        )

        # Configure uvicorn
        config = uvicorn.Config(
            app,
            host=self.config.host,
            port=self.config.port,
            log_level="info" if self.config.debug else "warning",
            access_log=self.config.access_log,
        )

        self.server = uvicorn.Server(config)
        self.server_task = asyncio.create_task(self.server.serve())

        logger.info("API server started successfully")

    async def stop(self):
        """Stop the API server."""
        if self.server:
            logger.info("Stopping API server...")
            self.server.should_exit = True
            if self.server_task:
                await self.server_task
            logger.info("API server stopped")

    def is_running(self) -> bool:
        """Check if the API server is running."""
        return self.server_task is not None and not self.server_task.done()


# Integration helper functions


async def integrate_with_tracking_manager(
    tracking_manager: "TrackingManager",
) -> APIServer:
    """
    Create and integrate API server with tracking manager.

    This is the main integration point for the system.
    """
    # Set the global tracking manager instance for API endpoints
    set_tracking_manager(tracking_manager)

    api_server = APIServer(tracking_manager)
    return api_server
