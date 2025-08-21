"""
MQTT Integration Manager for automatic Home Assistant integration.

This module provides the high-level orchestration of all MQTT publishing
functionality, integrating seamlessly with the tracking system for
automatic operation without manual intervention.

Features:
- Automatic MQTT publishing when predictions are made
- Integration with TrackingManager for transparent operation
- System status monitoring and publishing
- Home Assistant discovery management
- No manual setup required for users
"""

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Callable, Dict, List, Optional

from ..core.config import MQTTConfig, RoomConfig, get_config
from ..core.exceptions import ErrorSeverity, OccupancyPredictionError
from ..models.base.predictor import PredictionResult
from .discovery_publisher import DiscoveryPublisher
from .mqtt_publisher import MQTTPublisher
from .prediction_publisher import PredictionPublisher

logger = logging.getLogger(__name__)


@dataclass
class MQTTIntegrationStats:
    """Statistics for MQTT integration."""

    initialized: bool = False
    mqtt_connected: bool = False
    discovery_published: bool = False
    predictions_published: int = 0
    status_updates_published: int = 0
    last_prediction_published: Optional[datetime] = None
    last_status_published: Optional[datetime] = None
    total_errors: int = 0
    last_error: Optional[str] = None


class MQTTIntegrationManager:
    """
    High-level MQTT integration manager for automatic Home Assistant integration.

    This manager orchestrates all MQTT publishing functionality and integrates
    seamlessly with the prediction system for fully automatic operation.
    """

    def __init__(
        self,
        mqtt_config: Optional[MQTTConfig] = None,
        rooms: Optional[Dict[str, RoomConfig]] = None,
        notification_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize MQTT integration manager.

        Args:
            mqtt_config: MQTT configuration (loads from global config if None)
            rooms: Room configurations (loads from global config if None)
            notification_callbacks: Optional notification callbacks
        """
        # Load configuration
        if mqtt_config is None or rooms is None:
            system_config = get_config()
            mqtt_config = mqtt_config or system_config.mqtt
            rooms = rooms or system_config.rooms

        self.mqtt_config = mqtt_config
        self.rooms = rooms
        self.notification_callbacks = notification_callbacks or []

        # Core MQTT components
        self.mqtt_publisher: Optional[MQTTPublisher] = None
        self.prediction_publisher: Optional[PredictionPublisher] = None
        self.discovery_publisher: Optional[DiscoveryPublisher] = None

        # Status tracking
        self.stats = MQTTIntegrationStats()
        self.system_start_time = datetime.now(timezone.utc)

        # Background tasks
        self._background_tasks: List[asyncio.Task] = []
        self._shutdown_event = asyncio.Event()
        self._integration_active = False

        # Status publishing state
        self._last_system_stats: Optional[Dict[str, Any]] = None

        logger.info(f"Initialized MQTTIntegrationManager with {len(rooms)} rooms")

    async def initialize(self) -> None:
        """Initialize all MQTT integration components."""
        try:
            if not self.mqtt_config.publishing_enabled:
                logger.info("MQTT publishing is disabled in configuration")
                return

            logger.info("Initializing MQTT integration components")

            # Initialize core MQTT publisher
            self.mqtt_publisher = MQTTPublisher(
                config=self.mqtt_config,
                on_connect_callback=self._on_mqtt_connect,
                on_disconnect_callback=self._on_mqtt_disconnect,
            )
            await self.mqtt_publisher.initialize()

            # Initialize prediction publisher
            self.prediction_publisher = PredictionPublisher(
                mqtt_publisher=self.mqtt_publisher,
                config=self.mqtt_config,
                rooms=self.rooms,
            )

            # Initialize enhanced discovery publisher
            if self.mqtt_config.discovery_enabled:
                self.discovery_publisher = DiscoveryPublisher(
                    mqtt_publisher=self.mqtt_publisher,
                    config=self.mqtt_config,
                    rooms=self.rooms,
                    availability_check_callback=self._check_system_availability,
                    state_change_callback=self._handle_entity_state_change,
                )

                # Publish enhanced discovery messages
                logger.info("Publishing enhanced Home Assistant discovery messages")
                discovery_results = (
                    await self.discovery_publisher.publish_all_discovery()
                )

                successful_discoveries = sum(
                    1 for r in discovery_results.values() if r.success
                )
                total_discoveries = len(discovery_results)

                if successful_discoveries == total_discoveries:
                    self.stats.discovery_published = True
                    logger.info(
                        f"Successfully published all {total_discoveries} discovery messages"
                    )
                else:
                    failed = total_discoveries - successful_discoveries
                    logger.warning(
                        f"Discovery partially successful: {successful_discoveries}/{total_discoveries}, {failed} failed"
                    )
                    self.stats.discovery_published = successful_discoveries > 0

            # Start background tasks
            await self.start_integration()

            self.stats.initialized = True
            self.stats.mqtt_connected = self.mqtt_publisher.connection_status.connected

            logger.info("MQTT integration initialized successfully")

        except Exception as e:
            self.stats.total_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Failed to initialize MQTT integration: {e}")
            raise MQTTIntegrationError("Failed to initialize MQTT integration", cause=e)

    async def start_integration(self) -> None:
        """Start background integration tasks."""
        try:
            if self._integration_active:
                logger.warning("MQTT integration already active")
                return

            if not self.mqtt_config.publishing_enabled:
                logger.info("MQTT publishing is disabled, not starting integration")
                return

            # Start system status publishing task
            if self.mqtt_config.publish_system_status:
                status_task = asyncio.create_task(self._system_status_publishing_loop())
                self._background_tasks.append(status_task)

            self._integration_active = True
            logger.info("Started MQTT integration background tasks")

        except Exception as e:
            logger.error(f"Failed to start MQTT integration: {e}")
            raise MQTTIntegrationError("Failed to start MQTT integration", cause=e)

    async def stop_integration(self) -> None:
        """Stop MQTT integration gracefully."""
        try:
            if not self._integration_active:
                return

            # Signal shutdown
            self._shutdown_event.set()

            # Stop MQTT publisher
            if self.mqtt_publisher:
                await self.mqtt_publisher.stop_publisher()

            # Wait for background tasks to complete
            if self._background_tasks:
                await asyncio.gather(*self._background_tasks, return_exceptions=True)

            self._background_tasks.clear()
            self._integration_active = False

            logger.info("Stopped MQTT integration")

        except Exception as e:
            logger.error(f"Error stopping MQTT integration: {e}")

    async def publish_prediction(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str] = None,
    ) -> bool:
        """
        Publish a prediction result to Home Assistant.

        This method is called automatically by the TrackingManager when
        predictions are recorded.

        Args:
            prediction_result: The prediction to publish
            room_id: Room the prediction is for
            current_state: Current occupancy state if known

        Returns:
            True if published successfully, False otherwise
        """
        try:
            if not self._integration_active or not self.prediction_publisher:
                logger.debug(
                    "MQTT integration not active, skipping prediction publishing"
                )
                return False

            result = await self.prediction_publisher.publish_prediction(
                prediction_result=prediction_result,
                room_id=room_id,
                current_state=current_state,
            )

            if result.success:
                self.stats.predictions_published += 1
                self.stats.last_prediction_published = datetime.now(timezone.utc)
                logger.debug(f"Published prediction for {room_id} to Home Assistant")
                return True
            else:
                self.stats.total_errors += 1
                self.stats.last_error = result.error_message
                logger.error(
                    f"Failed to publish prediction for {room_id}: {result.error_message}"
                )
                return False

        except Exception as e:
            self.stats.total_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error publishing prediction for {room_id}: {e}")
            return False

    async def publish_system_status(
        self,
        tracking_stats: Optional[Dict[str, Any]] = None,
        model_stats: Optional[Dict[str, Any]] = None,
        database_connected: bool = True,
        active_alerts: int = 0,
        last_error: Optional[str] = None,
    ) -> bool:
        """
        Publish system status to Home Assistant.

        Args:
            tracking_stats: Statistics from tracking manager
            model_stats: Statistics from model ensemble
            database_connected: Database connection status
            active_alerts: Number of active alerts
            last_error: Last error message if any

        Returns:
            True if published successfully, False otherwise
        """
        try:
            if not self._integration_active or not self.prediction_publisher:
                return False

            result = await self.prediction_publisher.publish_system_status(
                tracking_stats=tracking_stats,
                model_stats=model_stats,
                database_connected=database_connected,
                active_alerts=active_alerts,
                last_error=last_error,
            )

            if result.success:
                self.stats.status_updates_published += 1
                self.stats.last_status_published = datetime.now(timezone.utc)
                logger.debug("Published system status to Home Assistant")
                return True
            else:
                self.stats.total_errors += 1
                self.stats.last_error = result.error_message
                logger.error(f"Failed to publish system status: {result.error_message}")
                return False

        except Exception as e:
            self.stats.total_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error publishing system status: {e}")
            return False

    async def refresh_discovery(self) -> bool:
        """
        Refresh Home Assistant discovery messages.

        Returns:
            True if refreshed successfully, False otherwise
        """
        try:
            if not self.discovery_publisher:
                logger.warning("Discovery publisher not available")
                return False

            results = await self.discovery_publisher.refresh_discovery()
            successful = sum(1 for r in results.values() if r.success)
            total = len(results)

            if successful == total:
                self.stats.discovery_published = True
                logger.info(f"Refreshed discovery for all {total} entities")
                return True
            else:
                logger.warning(
                    f"Discovery refresh partially failed: {successful}/{total} successful"
                )
                return False

        except Exception as e:
            self.stats.total_errors += 1
            self.stats.last_error = str(e)
            logger.error(f"Error refreshing discovery: {e}")
            return False

    def get_integration_stats(self) -> Dict[str, Any]:
        """Get comprehensive enhanced MQTT integration statistics."""
        stats_dict = {
            "initialized": self.stats.initialized,
            "integration_active": self._integration_active,
            "mqtt_connected": (
                self.mqtt_publisher.connection_status.connected
                if self.mqtt_publisher
                else False
            ),
            "discovery_published": self.stats.discovery_published,
            "predictions_published": self.stats.predictions_published,
            "status_updates_published": self.stats.status_updates_published,
            "last_prediction_published": (
                self.stats.last_prediction_published.isoformat()
                if self.stats.last_prediction_published
                else None
            ),
            "last_status_published": (
                self.stats.last_status_published.isoformat()
                if self.stats.last_status_published
                else None
            ),
            "total_errors": self.stats.total_errors,
            "last_error": self.stats.last_error,
            "system_uptime_seconds": (
                datetime.now(timezone.utc) - self.system_start_time
            ).total_seconds(),
            "background_tasks": len(self._background_tasks),
            "rooms_configured": len(self.rooms),
        }

        # Add MQTT publisher stats if available
        if self.mqtt_publisher:
            stats_dict["mqtt_publisher"] = self.mqtt_publisher.get_publisher_stats()

        # Add prediction publisher stats if available
        if self.prediction_publisher:
            stats_dict["prediction_publisher"] = (
                self.prediction_publisher.get_publisher_stats()
            )

        # Add enhanced discovery publisher stats if available
        if self.discovery_publisher:
            discovery_stats = self.discovery_publisher.get_discovery_stats()
            stats_dict["discovery_publisher"] = discovery_stats

            # Add high-level discovery insights
            stats_dict["discovery_insights"] = {
                "entity_health": (
                    "healthy"
                    if discovery_stats.get("published_entities_count", 0) > 0
                    else "no_entities"
                ),
                "device_available": discovery_stats.get("device_available", False),
                "services_available": discovery_stats.get("available_services_count", 0)
                > 0,
                "metadata_complete": discovery_stats.get("entity_metadata_count", 0)
                == discovery_stats.get("published_entities_count", 0),
                "last_availability_update": discovery_stats.get(
                    "last_availability_publish"
                ),
                "discovery_errors": discovery_stats.get("statistics", {}).get(
                    "discovery_errors", 0
                ),
            }

        # Add system health summary
        stats_dict["system_health"] = {
            "overall_status": (
                "healthy"
                if (
                    stats_dict["mqtt_connected"]
                    and stats_dict["discovery_published"]
                    and stats_dict["total_errors"] < 10
                )
                else "degraded"
            ),
            "component_status": {
                "mqtt": (
                    "connected" if stats_dict["mqtt_connected"] else "disconnected"
                ),
                "discovery": (
                    "published"
                    if stats_dict["discovery_published"]
                    else "not_published"
                ),
                "predictions": (
                    "active" if stats_dict["predictions_published"] > 0 else "inactive"
                ),
                "background_tasks": (
                    "running" if stats_dict["background_tasks"] > 0 else "stopped"
                ),
            },
            "error_rate": stats_dict["total_errors"]
            / max(1, stats_dict["predictions_published"]),
            "uptime_hours": stats_dict["system_uptime_seconds"] / 3600,
        }

        return stats_dict

    def is_connected(self) -> bool:
        """Check if MQTT is connected and ready for publishing."""
        return (
            self._integration_active
            and self.mqtt_publisher is not None
            and self.mqtt_publisher.connection_status.connected
        )

    def add_notification_callback(self, callback: Callable) -> None:
        """Add a notification callback for MQTT events."""
        if callback not in self.notification_callbacks:
            self.notification_callbacks.append(callback)
            logger.info("Added MQTT notification callback")

    def remove_notification_callback(self, callback: Callable) -> None:
        """Remove a notification callback."""
        if callback in self.notification_callbacks:
            self.notification_callbacks.remove(callback)
            logger.info("Removed MQTT notification callback")

    # Private methods

    async def _system_status_publishing_loop(self) -> None:
        """Background loop for publishing system status."""
        try:
            while not self._shutdown_event.is_set():
                try:
                    # Publish system status (would get stats from tracking manager)
                    await self.publish_system_status(
                        tracking_stats=self._last_system_stats,
                        database_connected=True,  # Would be determined from system state
                        active_alerts=0,  # Would be determined from system state
                    )

                    # Wait for next status update cycle
                    await asyncio.wait_for(
                        self._shutdown_event.wait(),
                        timeout=self.mqtt_config.status_update_interval_seconds,
                    )

                except asyncio.TimeoutError:
                    # Expected timeout for status update interval
                    continue
                except Exception as e:
                    logger.error(f"Error in system status publishing loop: {e}")
                    await asyncio.sleep(30)  # Wait before retrying

        except asyncio.CancelledError:
            logger.info("System status publishing loop cancelled")
            raise
        except Exception as e:
            logger.error(f"System status publishing loop failed: {e}")

    async def _on_mqtt_connect(self, client, userdata, flags, reason_code) -> None:
        """Callback for MQTT connection events."""
        try:
            self.stats.mqtt_connected = True
            logger.info("MQTT connected - Home Assistant integration ready")

            # Notify callbacks
            for callback in self.notification_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback("mqtt_connected")
                    else:
                        callback("mqtt_connected")
                except Exception as e:
                    logger.error(f"Error in MQTT connect notification callback: {e}")

        except Exception as e:
            logger.error(f"Error in MQTT connect callback: {e}")

    async def _on_mqtt_disconnect(self, client, userdata, flags, reason_code) -> None:
        """Callback for MQTT disconnection events."""
        try:
            self.stats.mqtt_connected = False
            logger.warning(
                "MQTT disconnected - Home Assistant integration temporarily unavailable"
            )

            # Notify callbacks
            for callback in self.notification_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback("mqtt_disconnected")
                    else:
                        callback("mqtt_disconnected")
                except Exception as e:
                    logger.error(f"Error in MQTT disconnect notification callback: {e}")

        except Exception as e:
            logger.error(f"Error in MQTT disconnect callback: {e}")

    def update_system_stats(self, stats: Dict[str, Any]) -> None:
        """Update cached system stats for status publishing."""
        self._last_system_stats = stats

    # Enhanced Integration Methods

    async def update_device_availability(self, online: bool = True) -> bool:
        """
        Update device availability status in Home Assistant.

        Args:
            online: Whether device is online or offline

        Returns:
            True if updated successfully, False otherwise
        """
        try:
            if not self.discovery_publisher:
                logger.warning(
                    "Discovery publisher not available for availability update"
                )
                return False

            result = await self.discovery_publisher.publish_device_availability(
                online=online
            )

            if result.success:
                logger.info(
                    f"Updated device availability: {'online' if online else 'offline'}"
                )
                return True
            else:
                logger.error(
                    f"Failed to update device availability: {result.error_message}"
                )
                return False

        except Exception as e:
            logger.error(f"Error updating device availability: {e}")
            return False

    async def handle_service_command(
        self, service_name: str, command_data: Dict[str, Any]
    ) -> bool:
        """
        Handle Home Assistant service commands.

        Args:
            service_name: Name of the service being called
            command_data: Command data from Home Assistant

        Returns:
            True if handled successfully, False otherwise
        """
        try:
            logger.info(f"Handling service command: {service_name}")

            if service_name == "manual_retrain":
                # Handle manual retraining request
                room_id = command_data.get("room_id", "all")
                strategy = command_data.get("strategy", "auto")

                logger.info(
                    f"Manual retrain requested for room: {room_id}, strategy: {strategy}"
                )
                # This would integrate with the TrackingManager's retraining system
                return True

            elif service_name == "refresh_discovery":
                # Handle discovery refresh request
                if self.discovery_publisher:
                    results = await self.discovery_publisher.refresh_discovery()
                    successful = sum(1 for r in results.values() if r.success)
                    total = len(results)

                    logger.info(
                        f"Discovery refresh completed: {successful}/{total} successful"
                    )
                    return successful == total
                return False

            elif service_name == "reset_statistics":
                # Handle statistics reset request
                logger.info("Resetting system statistics")
                self.stats = MQTTIntegrationStats()
                return True

            elif service_name == "force_prediction":
                # Handle force prediction request
                room_id = command_data.get("room_id")
                if room_id:
                    logger.info(f"Force prediction requested for room: {room_id}")
                    # This would integrate with the prediction system
                    return True
                else:
                    logger.warning("Force prediction requested without room_id")
                    return False

            else:
                logger.warning(f"Unknown service command: {service_name}")
                return False

        except Exception as e:
            logger.error(f"Error handling service command {service_name}: {e}")
            return False

    async def cleanup_discovery(self, entity_ids: Optional[List[str]] = None) -> bool:
        """
        Clean up Home Assistant discovery entities.

        Args:
            entity_ids: Optional list of specific entities to clean up

        Returns:
            True if cleanup successful, False otherwise
        """
        try:
            if not self.discovery_publisher:
                logger.warning("Discovery publisher not available for cleanup")
                return False

            results = await self.discovery_publisher.cleanup_entities(entity_ids)
            successful = sum(1 for r in results.values() if r.success)
            total = len(results)

            logger.info(f"Discovery cleanup completed: {successful}/{total} successful")
            return successful == total

        except Exception as e:
            logger.error(f"Error cleaning up discovery: {e}")
            return False

    # Enhanced Private Methods

    async def _check_system_availability(self) -> bool:
        """
        Check system availability for discovery publisher.

        Returns:
            True if system is available, False otherwise
        """
        try:
            # Check if MQTT is connected
            if not self.is_connected():
                return False

            # Check if background tasks are running
            if not self._integration_active:
                return False

            # Additional availability checks can be added here
            # (e.g., database connectivity, model readiness, etc.)

            return True

        except Exception as e:
            logger.error(f"Error checking system availability: {e}")
            return False

    async def _handle_entity_state_change(
        self,
        entity_id: str,
        state: Any,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Handle entity state changes from discovery publisher.

        Args:
            entity_id: Entity that changed state
            state: New entity state
            attributes: Optional attributes
        """
        try:
            logger.debug(f"Entity {entity_id} state changed to {state.value}")

            # Notify callbacks if any
            for callback in self.notification_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(f"entity_state_change:{entity_id}:{state.value}")
                    else:
                        callback(f"entity_state_change:{entity_id}:{state.value}")
                except Exception as e:
                    logger.error(f"Error in entity state change callback: {e}")

        except Exception as e:
            logger.error(f"Error handling entity state change: {e}")


class MQTTIntegrationError(OccupancyPredictionError):
    """Raised when MQTT integration operations fail."""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message=message,
            error_code="MQTT_INTEGRATION_ERROR",
            severity=kwargs.get("severity", ErrorSeverity.MEDIUM),
            **kwargs,
        )
