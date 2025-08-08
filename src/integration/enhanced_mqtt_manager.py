"""
Enhanced MQTT Integration Manager with Real-time Publishing.

This module extends the existing MQTT integration with comprehensive real-time
publishing capabilities, integrating seamlessly with TrackingManager for
automatic operation across multiple channels.

Features:
- All existing MQTT functionality preserved
- Real-time WebSocket and SSE integration
- Automatic multi-channel broadcasting
- Performance monitoring across channels
- Backward compatibility with existing system
"""

import asyncio
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
import logging

from ..core.config import MQTTConfig, RoomConfig, get_config
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..models.base.predictor import PredictionResult
from .discovery_publisher import DiscoveryPublisher
from .mqtt_integration_manager import (
    MQTTIntegrationManager,
    MQTTIntegrationStats,
)
from .mqtt_publisher import MQTTPublisher
from .prediction_publisher import PredictionPublisher
from .realtime_publisher import (
    PublishingChannel,
    PublishingMetrics,
    RealtimePredictionEvent,
    RealtimePublishingSystem,
)

logger = logging.getLogger(__name__)


@dataclass
class EnhancedIntegrationStats:
    """Enhanced statistics including real-time publishing metrics."""

    # Base MQTT stats
    mqtt_stats: MQTTIntegrationStats

    # Real-time publishing stats
    realtime_stats: PublishingMetrics

    # Combined metrics
    total_channels_active: int = 0
    total_clients_connected: int = 0
    predictions_per_minute: float = 0.0

    # Performance metrics
    average_publish_latency_ms: float = 0.0
    publish_success_rate: float = 0.0
    last_performance_update: Optional[datetime] = None


class EnhancedMQTTIntegrationManager:
    """
    Enhanced MQTT Integration Manager with comprehensive real-time publishing.

    Extends the base MQTT integration with WebSocket and SSE capabilities,
    providing automatic real-time prediction broadcasting across multiple
    channels while maintaining full backward compatibility.
    """

    def __init__(
        self,
        mqtt_config: Optional[MQTTConfig] = None,
        rooms: Optional[Dict[str, RoomConfig]] = None,
        notification_callbacks: Optional[List[Callable]] = None,
        enabled_realtime_channels: Optional[List[PublishingChannel]] = None,
    ):
        """
        Initialize enhanced MQTT integration manager.

        Args:
            mqtt_config: MQTT configuration (loads from global config if None)
            rooms: Room configurations (loads from global config if None)
            notification_callbacks: Optional notification callbacks
            enabled_realtime_channels: Real-time channels to enable
        """
        # Load configuration
        if mqtt_config is None or rooms is None:
            system_config = get_config()
            mqtt_config = mqtt_config or system_config.mqtt
            rooms = rooms or system_config.rooms

        self.mqtt_config = mqtt_config
        self.rooms = rooms
        self.notification_callbacks = notification_callbacks or []

        # Initialize base MQTT integration
        self.base_mqtt_manager = MQTTIntegrationManager(
            mqtt_config=mqtt_config,
            rooms=rooms,
            notification_callbacks=notification_callbacks,
        )

        # Initialize real-time publishing system
        self.realtime_publisher = RealtimePublishingSystem(
            mqtt_config=mqtt_config,
            rooms=rooms,
            enabled_channels=enabled_realtime_channels
            or [
                PublishingChannel.MQTT,
                PublishingChannel.WEBSOCKET,
                PublishingChannel.SSE,
            ],
        )

        # Enhanced statistics
        self.enhanced_stats = EnhancedIntegrationStats(
            mqtt_stats=MQTTIntegrationStats(),
            realtime_stats=PublishingMetrics(),
        )

        # Performance tracking
        self._publish_times: List[datetime] = []
        self._publish_latencies: List[float] = []
        self._last_stats_update = datetime.utcnow()

        # Integration state
        self._integration_initialized = False
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()

        logger.info(
            "Initialized EnhancedMQTTIntegrationManager with "
            f"{len(self.realtime_publisher.enabled_channels)} channels"
        )

    async def initialize(self) -> None:
        """Initialize both MQTT and real-time publishing systems."""
        try:
            # Initialize base MQTT system
            await self.base_mqtt_manager.initialize()

            # Set prediction publisher for real-time system
            if hasattr(self.base_mqtt_manager, "prediction_publisher"):
                self.realtime_publisher.prediction_publisher = (
                    self.base_mqtt_manager.prediction_publisher
                )

            # Initialize real-time publishing
            await self.realtime_publisher.initialize()

            # Start enhanced monitoring
            await self._start_enhanced_monitoring()

            self._integration_initialized = True
            logger.info("Enhanced MQTT integration initialized successfully")

        except Exception as e:
            logger.error(
                f"Failed to initialize enhanced MQTT integration: {e}"
            )
            raise EnhancedMQTTIntegrationError(
                "Failed to initialize integration", cause=e
            )

    async def shutdown(self) -> None:
        """Shutdown both MQTT and real-time systems."""
        try:
            self._shutdown_event.set()

            # Cancel background tasks
            if self._background_tasks:
                for task in self._background_tasks:
                    task.cancel()
                await asyncio.gather(
                    *self._background_tasks, return_exceptions=True
                )

            # Shutdown real-time publisher
            await self.realtime_publisher.shutdown()

            # Shutdown base MQTT manager
            await self.base_mqtt_manager.shutdown()

            logger.info("Enhanced MQTT integration shutdown complete")

        except Exception as e:
            logger.error(f"Error during enhanced integration shutdown: {e}")

    async def publish_prediction(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publish prediction across all channels (MQTT, WebSocket, SSE).

        This method is called automatically by TrackingManager and provides
        comprehensive real-time broadcasting while maintaining MQTT functionality.

        Args:
            prediction_result: The prediction to publish
            room_id: Room the prediction is for
            current_state: Current occupancy state if known

        Returns:
            Dictionary with publish results for all channels
        """
        publish_start_time = datetime.utcnow()

        try:
            # Use real-time publisher for comprehensive broadcasting
            results = await self.realtime_publisher.publish_prediction(
                prediction_result=prediction_result,
                room_id=room_id,
                current_state=current_state,
            )

            # Track performance metrics
            publish_latency = (
                datetime.utcnow() - publish_start_time
            ).total_seconds() * 1000
            self._record_publish_performance(publish_latency, results)

            # Update base MQTT stats if available
            if hasattr(self.base_mqtt_manager, "stats"):
                self.base_mqtt_manager.stats.predictions_published += 1
                self.base_mqtt_manager.stats.last_prediction_published = (
                    datetime.utcnow()
                )

            logger.debug(
                f"Published prediction for {room_id} across {len(results)} channels "
                f"in {publish_latency:.2f}ms"
            )

            return results

        except Exception as e:
            logger.error(
                f"Error in enhanced prediction publishing for {room_id}: {e}"
            )
            return {"error": str(e), "success": False}

    async def publish_system_status(
        self,
        tracking_stats: Optional[Dict[str, Any]] = None,
        model_stats: Optional[Dict[str, Any]] = None,
        database_connected: bool = True,
        active_alerts: int = 0,
        last_error: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Publish system status across all channels.

        Args:
            tracking_stats: Statistics from tracking manager
            model_stats: Statistics from model ensemble
            database_connected: Database connection status
            active_alerts: Number of active alerts
            last_error: Last error message if any

        Returns:
            Dictionary with publish results for all channels
        """
        try:
            # Prepare comprehensive status data
            status_data = {
                "system_status": self._determine_system_status(
                    database_connected, active_alerts
                ),
                "timestamp": datetime.utcnow().isoformat(),
                "tracking_stats": tracking_stats,
                "model_stats": model_stats,
                "integration_stats": self.get_integration_stats(),
                "database_connected": database_connected,
                "active_alerts": active_alerts,
                "last_error": last_error,
            }

            # Publish via MQTT (base manager)
            mqtt_result = await self.base_mqtt_manager.publish_system_status(
                tracking_stats=tracking_stats,
                model_stats=model_stats,
                database_connected=database_connected,
                active_alerts=active_alerts,
                last_error=last_error,
            )

            # Publish via real-time channels
            realtime_results = (
                await self.realtime_publisher.publish_system_status(
                    status_data=status_data
                )
            )

            # Combine results
            combined_results = {
                "mqtt": {
                    "success": (
                        mqtt_result.success
                        if hasattr(mqtt_result, "success")
                        else True
                    ),
                    "error": getattr(mqtt_result, "error_message", None),
                },
                **realtime_results,
            }

            logger.debug(
                f"Published system status across {len(combined_results)} channels"
            )
            return combined_results

        except Exception as e:
            logger.error(f"Error publishing enhanced system status: {e}")
            return {"error": str(e), "success": False}

    async def handle_websocket_connection(self, websocket, path: str):
        """Handle WebSocket connection for real-time predictions."""
        return await self.realtime_publisher.handle_websocket_connection(
            websocket, path
        )

    async def create_sse_stream(self, room_id: Optional[str] = None):
        """Create Server-Sent Events stream for real-time predictions."""
        return await self.realtime_publisher.create_sse_stream(room_id)

    def add_realtime_callback(self, callback: Callable) -> None:
        """Add callback for real-time broadcast events."""
        self.realtime_publisher.add_broadcast_callback(callback)

    def remove_realtime_callback(self, callback: Callable) -> None:
        """Remove real-time broadcast callback."""
        self.realtime_publisher.remove_broadcast_callback(callback)

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive integration statistics."""
        try:
            # Update enhanced stats
            self._update_enhanced_stats()

            base_stats = {}
            if hasattr(self.base_mqtt_manager, "get_integration_stats"):
                base_stats = self.base_mqtt_manager.get_integration_stats()

            realtime_stats = self.realtime_publisher.get_publishing_stats()

            return {
                "mqtt_integration": base_stats,
                "realtime_publishing": realtime_stats,
                "enhanced_metrics": asdict(self.enhanced_stats),
                "performance": {
                    "predictions_per_minute": self.enhanced_stats.predictions_per_minute,
                    "average_publish_latency_ms": self.enhanced_stats.average_publish_latency_ms,
                    "publish_success_rate": self.enhanced_stats.publish_success_rate,
                },
                "channels": {
                    "total_active": self.enhanced_stats.total_channels_active,
                    "enabled_channels": [
                        channel.value
                        for channel in self.realtime_publisher.enabled_channels
                    ],
                },
                "connections": {
                    "total_clients": self.enhanced_stats.total_clients_connected,
                    "websocket_clients": realtime_stats.get(
                        "websocket_stats", {}
                    ).get("total_active_connections", 0),
                    "sse_clients": realtime_stats.get("sse_stats", {}).get(
                        "total_active_connections", 0
                    ),
                },
            }

        except Exception as e:
            logger.error(f"Error getting integration stats: {e}")
            return {"error": str(e)}

    def get_connection_info(self) -> Dict[str, Any]:
        """Get information about active connections."""
        try:
            realtime_stats = self.realtime_publisher.get_publishing_stats()

            return {
                "websocket_connections": realtime_stats.get(
                    "websocket_stats", {}
                ),
                "sse_connections": realtime_stats.get("sse_stats", {}),
                "mqtt_connection": {
                    "connected": getattr(
                        self.base_mqtt_manager, "mqtt_connected", False
                    )
                },
                "total_active_connections": (
                    realtime_stats.get("websocket_stats", {}).get(
                        "total_active_connections", 0
                    )
                    + realtime_stats.get("sse_stats", {}).get(
                        "total_active_connections", 0
                    )
                ),
            }

        except Exception as e:
            logger.error(f"Error getting connection info: {e}")
            return {"error": str(e)}

    # Delegate methods to base MQTT manager for backward compatibility

    async def start_discovery_publishing(self) -> bool:
        """Start Home Assistant discovery publishing."""
        if hasattr(self.base_mqtt_manager, "start_discovery_publishing"):
            return await self.base_mqtt_manager.start_discovery_publishing()
        return False

    async def stop_discovery_publishing(self) -> None:
        """Stop discovery publishing."""
        if hasattr(self.base_mqtt_manager, "stop_discovery_publishing"):
            await self.base_mqtt_manager.stop_discovery_publishing()

    async def publish_room_batch(
        self,
        predictions: Dict[str, PredictionResult],
        current_states: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """Publish predictions for multiple rooms in batch."""
        results = {}

        for room_id, prediction in predictions.items():
            current_state = (
                current_states.get(room_id) if current_states else None
            )
            result = await self.publish_prediction(
                prediction, room_id, current_state
            )
            results[room_id] = result

        return results

    # Private methods

    async def _start_enhanced_monitoring(self) -> None:
        """Start enhanced monitoring tasks."""
        try:
            # Performance monitoring task
            perf_task = asyncio.create_task(
                self._performance_monitoring_loop()
            )
            self._background_tasks.append(perf_task)

            # Statistics update task
            stats_task = asyncio.create_task(self._stats_update_loop())
            self._background_tasks.append(stats_task)

            logger.info("Enhanced monitoring started")

        except Exception as e:
            logger.error(f"Failed to start enhanced monitoring: {e}")

    def _record_publish_performance(
        self, latency_ms: float, results: Dict[str, Any]
    ) -> None:
        """Record publishing performance metrics."""
        try:
            now = datetime.utcnow()

            # Track publish times and latencies
            self._publish_times.append(now)
            self._publish_latencies.append(latency_ms)

            # Keep only recent data (last hour)
            cutoff_time = now - timedelta(hours=1)
            self._publish_times = [
                t for t in self._publish_times if t > cutoff_time
            ]
            self._publish_latencies = [
                lat
                for lat, t in zip(self._publish_latencies, self._publish_times)
                if t > cutoff_time
            ]

            # Update success rate
            successful_channels = sum(
                1
                for result in results.values()
                if isinstance(result, dict) and result.get("success", False)
            )
            total_channels = len(results)

            if total_channels > 0:
                success_rate = successful_channels / total_channels
                # Exponential moving average
                if self.enhanced_stats.publish_success_rate == 0.0:
                    self.enhanced_stats.publish_success_rate = success_rate
                else:
                    alpha = 0.1
                    self.enhanced_stats.publish_success_rate = (
                        alpha * success_rate
                        + (1 - alpha)
                        * self.enhanced_stats.publish_success_rate
                    )

        except Exception as e:
            logger.error(f"Error recording publish performance: {e}")

    def _update_enhanced_stats(self) -> None:
        """Update enhanced statistics."""
        try:
            now = datetime.utcnow()

            # Update predictions per minute
            recent_publishes = [
                t
                for t in self._publish_times
                if t > now - timedelta(minutes=1)
            ]
            self.enhanced_stats.predictions_per_minute = len(recent_publishes)

            # Update average latency
            if self._publish_latencies:
                self.enhanced_stats.average_publish_latency_ms = sum(
                    self._publish_latencies
                ) / len(self._publish_latencies)

            # Update channel and connection counts
            realtime_stats = self.realtime_publisher.get_publishing_stats()

            self.enhanced_stats.total_channels_active = len(
                self.realtime_publisher.enabled_channels
            )

            ws_connections = realtime_stats.get("websocket_stats", {}).get(
                "total_active_connections", 0
            )
            sse_connections = realtime_stats.get("sse_stats", {}).get(
                "total_active_connections", 0
            )
            self.enhanced_stats.total_clients_connected = (
                ws_connections + sse_connections
            )

            # Update real-time stats from publisher
            if hasattr(self.realtime_publisher, "metrics"):
                self.enhanced_stats.realtime_stats = (
                    self.realtime_publisher.metrics
                )

            # Update MQTT stats from base manager
            if hasattr(self.base_mqtt_manager, "stats"):
                self.enhanced_stats.mqtt_stats = self.base_mqtt_manager.stats

            self.enhanced_stats.last_performance_update = now

        except Exception as e:
            logger.error(f"Error updating enhanced stats: {e}")

    def _determine_system_status(
        self, database_connected: bool, active_alerts: int
    ) -> str:
        """Determine overall system status."""
        try:
            # Check MQTT connection
            mqtt_connected = getattr(
                self.base_mqtt_manager, "mqtt_connected", False
            )

            # Check real-time system
            realtime_active = self.realtime_publisher._publishing_active

            if not database_connected or not mqtt_connected:
                return "offline"
            elif not realtime_active or active_alerts > 5:
                return "degraded"
            else:
                return "online"

        except Exception as e:
            logger.error(f"Error determining system status: {e}")
            return "unknown"

    async def _performance_monitoring_loop(self) -> None:
        """Background loop for performance monitoring."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    self._update_enhanced_stats()

                    # Wait for next monitoring cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=30,  # Update every 30 seconds
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in performance monitoring loop: {e}")
                    await asyncio.sleep(30)

        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Performance monitoring loop failed: {e}")

    async def _stats_update_loop(self) -> None:
        """Background loop for statistics updates."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Clean old performance data
                    cutoff_time = datetime.utcnow() - timedelta(hours=1)

                    self._publish_times = [
                        t for t in self._publish_times if t > cutoff_time
                    ]
                    self._publish_latencies = [
                        lat
                        for lat, t in zip(
                            self._publish_latencies, self._publish_times
                        )
                        if t > cutoff_time
                    ]

                    # Wait for next cleanup cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=300,  # Clean every 5 minutes
                    )

                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error(f"Error in stats update loop: {e}")
                    await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("Stats update loop cancelled")
            raise
        except Exception as e:
            logger.error(f"Stats update loop failed: {e}")


class EnhancedMQTTIntegrationError(OccupancyPredictionError):
    """Raised when enhanced MQTT integration operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="ENHANCED_MQTT_INTEGRATION_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
