"""
Prediction Publisher for Home Assistant MQTT Integration.

This module provides specialized publishing functionality for occupancy predictions,
formatted and organized for optimal Home Assistant integration and usability.

Features:
- Home Assistant compatible topic structure
- Structured prediction payload formatting
- Automatic prediction state management
- Integration with system status monitoring
- Room-based prediction organization
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import logging
from typing import Any, Dict, List, Optional

from ..core.config import MQTTConfig, RoomConfig
from ..models.base.predictor import PredictionResult
from .mqtt_publisher import MQTTPublisher, MQTTPublishResult

logger = logging.getLogger(__name__)


@dataclass
class PredictionPayload:
    """Home Assistant compatible prediction payload."""

    # Core prediction data
    predicted_time: str  # ISO format timestamp
    transition_type: str  # 'vacant_to_occupied' or 'occupied_to_vacant'
    confidence_score: float  # 0.0 to 1.0
    time_until_seconds: int  # Seconds until predicted transition
    time_until_human: str  # Human readable (e.g., "25 minutes")

    # Prediction metadata
    model_type: str
    model_version: str
    prediction_made_at: str  # ISO timestamp when prediction was made
    room_id: str
    room_name: str

    # Model details
    base_predictions: Dict[str, float]  # Individual model predictions
    model_weights: Dict[str, float]  # Model ensemble weights
    alternatives: List[Dict[str, Any]]  # Alternative predictions

    # Confidence and reliability
    prediction_reliability: str  # 'high', 'medium', 'low'
    features_count: int

    # System context
    system_version: str = "1.0.0"
    last_updated: str = None


@dataclass
class SystemStatusPayload:
    """System status payload for Home Assistant."""

    # System health
    system_status: str  # 'online', 'degraded', 'offline'
    uptime_seconds: int
    last_prediction_time: Optional[str]

    # Performance metrics
    total_predictions_made: int
    predictions_last_hour: int
    average_accuracy_percent: float
    average_confidence: float

    # Model status
    active_models: List[str]
    models_trained: int
    models_failed: int

    # System resources
    database_connected: bool
    mqtt_connected: bool
    tracking_active: bool

    # Error information
    active_alerts: int
    last_error: Optional[str]
    error_count_last_hour: int

    # Timestamps
    last_updated: str
    system_start_time: str


class PredictionPublisher:
    """
    Specialized publisher for occupancy predictions to Home Assistant.

    Handles the formatting, topic management, and publishing of prediction
    data in a structure optimized for Home Assistant consumption.
    """

    def __init__(
        self,
        mqtt_publisher: MQTTPublisher,
        config: MQTTConfig,
        rooms: Dict[str, RoomConfig],
    ):
        """
        Initialize prediction publisher.

        Args:
            mqtt_publisher: Core MQTT publisher instance
            config: MQTT configuration
            rooms: Room configuration mapping
        """
        self.mqtt_publisher = mqtt_publisher
        self.config = config
        self.rooms = rooms

        # Statistics
        self.predictions_published = 0
        self.status_updates_published = 0
        self.last_prediction_time: Optional[datetime] = None
        self.system_start_time = datetime.now(timezone.utc)

        # Cache for room names
        self.room_name_cache = {
            room_id: room_config.name for room_id, room_config in rooms.items()
        }

        logger.info(f"Initialized PredictionPublisher for {len(rooms)} rooms")

    async def publish_prediction(
        self,
        prediction_result: PredictionResult,
        room_id: str,
        current_state: Optional[str] = None,
    ) -> MQTTPublishResult:
        """
        Publish a prediction result to Home Assistant.

        Args:
            prediction_result: The prediction to publish
            room_id: Room the prediction is for
            current_state: Current occupancy state if known

        Returns:
            MQTTPublishResult indicating success/failure
        """
        try:
            # Get room name
            room_name = self.room_name_cache.get(
                room_id, room_id.replace("_", " ").title()
            )

            # Calculate time until transition
            time_until = prediction_result.predicted_time - datetime.now(timezone.utc)
            time_until_seconds = max(0, int(time_until.total_seconds()))

            # Format human readable time
            time_until_human = self._format_time_until(time_until_seconds)

            # Determine prediction reliability
            reliability = self._calculate_reliability(
                prediction_result.confidence_score
            )

            # Extract base model predictions from metadata
            base_predictions = prediction_result.prediction_metadata.get(
                "base_model_predictions", {}
            )
            model_weights = prediction_result.prediction_metadata.get(
                "model_weights", {}
            )

            # Format alternatives
            alternatives = []
            for alt_time, alt_confidence in (prediction_result.alternatives or [])[:3]:
                alternatives.append(
                    {
                        "predicted_time": alt_time.isoformat(),
                        "confidence": float(alt_confidence),
                        "time_until_seconds": max(
                            0,
                            int(
                                (alt_time - datetime.now(timezone.utc)).total_seconds()
                            ),
                        ),
                    }
                )

            # Create prediction payload
            payload = PredictionPayload(
                predicted_time=prediction_result.predicted_time.isoformat(),
                transition_type=prediction_result.transition_type,
                confidence_score=float(prediction_result.confidence_score),
                time_until_seconds=time_until_seconds,
                time_until_human=time_until_human,
                model_type=prediction_result.model_type,
                model_version=prediction_result.model_version or "unknown",
                prediction_made_at=datetime.now(timezone.utc).isoformat(),
                room_id=room_id,
                room_name=room_name,
                base_predictions=base_predictions,
                model_weights=model_weights,
                alternatives=alternatives,
                prediction_reliability=reliability,
                features_count=len(prediction_result.features_used or []),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

            # Determine topic
            topic = f"{self.config.topic_prefix}/{room_id}/prediction"

            # Publish prediction
            result = await self.mqtt_publisher.publish_json(
                topic=topic,
                data=asdict(payload),
                qos=self.config.prediction_qos,
                retain=self.config.retain_predictions,
            )

            if result.success:
                self.predictions_published += 1
                self.last_prediction_time = datetime.now(timezone.utc)

                logger.debug(
                    f"Published prediction for {room_name}: {prediction_result.transition_type} "
                    f"in {time_until_human} (confidence: {prediction_result.confidence_score:.2f})"
                )

                # Also publish to legacy topics for backward compatibility
                await self._publish_legacy_topics(room_id, payload)
            else:
                logger.error(
                    f"Failed to publish prediction for {room_id}: {result.error_message}"
                )

            return result

        except Exception as e:
            logger.error(f"Error publishing prediction for {room_id}: {e}")
            return MQTTPublishResult(
                success=False,
                topic=f"{self.config.topic_prefix}/{room_id}/prediction",
                payload_size=0,
                publish_time=datetime.now(timezone.utc),
                error_message=str(e),
            )

    async def publish_system_status(
        self,
        tracking_stats: Optional[Dict[str, Any]] = None,
        model_stats: Optional[Dict[str, Any]] = None,
        database_connected: bool = True,
        active_alerts: int = 0,
        last_error: Optional[str] = None,
    ) -> MQTTPublishResult:
        """
        Publish system status to Home Assistant.

        Args:
            tracking_stats: Statistics from tracking manager
            model_stats: Statistics from model ensemble
            database_connected: Database connection status
            active_alerts: Number of active alerts
            last_error: Last error message if any

        Returns:
            MQTTPublishResult indicating success/failure
        """
        try:
            # Calculate uptime
            uptime_seconds = int(
                (datetime.now(timezone.utc) - self.system_start_time).total_seconds()
            )

            # Extract tracking statistics
            total_predictions = (
                tracking_stats.get("performance", {}).get(
                    "total_predictions_recorded", 0
                )
                if tracking_stats
                else 0
            )
            predictions_last_hour = 0  # Would need to be calculated from tracking data
            avg_accuracy = 0.0  # Would need to be calculated from tracking data
            avg_confidence = 0.0  # Would need to be calculated from tracking data

            # Extract model statistics
            active_models = []
            models_trained = 0
            models_failed = 0

            if model_stats:
                # Extract model information from stats
                if "active_models" in model_stats:
                    active_models = model_stats["active_models"]
                if "models_trained" in model_stats:
                    models_trained = model_stats["models_trained"]
                if "models_failed" in model_stats:
                    models_failed = model_stats["models_failed"]

            # Determine system status
            system_status = self._determine_system_status(
                mqtt_connected=self.mqtt_publisher.connection_status.connected,
                database_connected=database_connected,
                active_alerts=active_alerts,
                models_failed=models_failed,
            )

            # Create status payload
            payload = SystemStatusPayload(
                system_status=system_status,
                uptime_seconds=uptime_seconds,
                last_prediction_time=(
                    self.last_prediction_time.isoformat()
                    if self.last_prediction_time
                    else None
                ),
                total_predictions_made=total_predictions,
                predictions_last_hour=predictions_last_hour,
                average_accuracy_percent=avg_accuracy,
                average_confidence=avg_confidence,
                active_models=active_models,
                models_trained=models_trained,
                models_failed=models_failed,
                database_connected=database_connected,
                mqtt_connected=self.mqtt_publisher.connection_status.connected,
                tracking_active=(
                    tracking_stats.get("tracking_active", False)
                    if tracking_stats
                    else False
                ),
                active_alerts=active_alerts,
                last_error=last_error,
                error_count_last_hour=0,  # Would need to be tracked
                last_updated=datetime.now(timezone.utc).isoformat(),
                system_start_time=self.system_start_time.isoformat(),
            )

            # Publish status
            topic = f"{self.config.topic_prefix}/system/status"

            result = await self.mqtt_publisher.publish_json(
                topic=topic,
                data=asdict(payload),
                qos=self.config.system_qos,
                retain=self.config.retain_system_status,
            )

            if result.success:
                self.status_updates_published += 1
                logger.debug(f"Published system status: {system_status}")
            else:
                logger.error(f"Failed to publish system status: {result.error_message}")

            return result

        except Exception as e:
            logger.error(f"Error publishing system status: {e}")
            return MQTTPublishResult(
                success=False,
                topic=f"{self.config.topic_prefix}/system/status",
                payload_size=0,
                publish_time=datetime.now(timezone.utc),
                error_message=str(e),
            )

    async def publish_room_batch(
        self,
        predictions: Dict[str, PredictionResult],
        current_states: Optional[Dict[str, str]] = None,
    ) -> Dict[str, MQTTPublishResult]:
        """
        Publish predictions for multiple rooms in batch.

        Args:
            predictions: Dictionary mapping room_id to prediction results
            current_states: Optional current states for each room

        Returns:
            Dictionary mapping room_id to publish results
        """
        results = {}
        current_states = current_states or {}

        for room_id, prediction in predictions.items():
            current_state = current_states.get(room_id)
            result = await self.publish_prediction(prediction, room_id, current_state)
            results[room_id] = result

        successful = sum(1 for r in results.values() if r.success)
        total = len(results)

        logger.info(f"Published batch predictions: {successful}/{total} successful")
        return results

    def get_publisher_stats(self) -> Dict[str, Any]:
        """Get prediction publisher statistics."""
        return {
            "predictions_published": self.predictions_published,
            "status_updates_published": self.status_updates_published,
            "last_prediction_time": (
                self.last_prediction_time.isoformat()
                if self.last_prediction_time
                else None
            ),
            "system_start_time": self.system_start_time.isoformat(),
            "rooms_configured": len(self.rooms),
            "mqtt_publisher_stats": self.mqtt_publisher.get_publisher_stats(),
        }

    # Private methods

    async def _publish_legacy_topics(
        self, room_id: str, payload: PredictionPayload
    ) -> None:
        """Publish to legacy topics for backward compatibility."""
        try:
            # Legacy individual value topics
            base_topic = f"{self.config.topic_prefix}/{room_id}"

            # Next transition time
            await self.mqtt_publisher.publish(
                topic=f"{base_topic}/next_transition_time",
                payload=payload.predicted_time,
                qos=self.config.prediction_qos,
                retain=self.config.retain_predictions,
            )

            # Transition type
            await self.mqtt_publisher.publish(
                topic=f"{base_topic}/transition_type",
                payload=payload.transition_type,
                qos=self.config.prediction_qos,
                retain=self.config.retain_predictions,
            )

            # Confidence
            await self.mqtt_publisher.publish(
                topic=f"{base_topic}/confidence",
                payload=str(payload.confidence_score),
                qos=self.config.prediction_qos,
                retain=self.config.retain_predictions,
            )

            # Time until (human readable)
            await self.mqtt_publisher.publish(
                topic=f"{base_topic}/time_until",
                payload=payload.time_until_human,
                qos=self.config.prediction_qos,
                retain=self.config.retain_predictions,
            )

        except Exception as e:
            logger.warning(f"Failed to publish some legacy topics for {room_id}: {e}")

    def _format_time_until(self, seconds: int) -> str:
        """Format seconds into human readable time."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            minutes = seconds // 60
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        elif seconds < 86400:
            hours = seconds // 3600
            minutes = (seconds % 3600) // 60
            if minutes == 0:
                return f"{hours} hour{'s' if hours != 1 else ''}"
            else:
                return f"{hours}h {minutes}m"
        else:
            days = seconds // 86400
            hours = (seconds % 86400) // 3600
            if hours == 0:
                return f"{days} day{'s' if days != 1 else ''}"
            else:
                return f"{days}d {hours}h"

    def _calculate_reliability(self, confidence: float) -> str:
        """Calculate prediction reliability based on confidence score."""
        if confidence >= 0.8:
            return "high"
        elif confidence >= 0.6:
            return "medium"
        else:
            return "low"

    def _determine_system_status(
        self,
        mqtt_connected: bool,
        database_connected: bool,
        active_alerts: int,
        models_failed: int,
    ) -> str:
        """Determine overall system status."""
        if not mqtt_connected or not database_connected:
            return "offline"
        elif active_alerts > 5 or models_failed > 0:
            return "degraded"
        else:
            return "online"
