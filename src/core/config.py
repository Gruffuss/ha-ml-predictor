"""
Configuration management for Occupancy Prediction System.
Loads and validates configuration from YAML files.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


@dataclass
class HomeAssistantConfig:
    """Home Assistant connection configuration."""

    url: str
    token: str
    websocket_timeout: int = 30
    api_timeout: int = 10


@dataclass
class DatabaseConfig:
    """Database connection configuration."""

    connection_string: str
    pool_size: int = 10
    max_overflow: int = 20


@dataclass
class MQTTConfig:
    """MQTT broker configuration."""

    broker: str
    port: int = 1883
    username: str = ""
    password: str = ""
    topic_prefix: str = "occupancy/predictions"

    # Home Assistant MQTT Discovery
    discovery_enabled: bool = True
    discovery_prefix: str = "homeassistant"
    device_name: str = "Occupancy Predictor"
    device_identifier: str = "ha_ml_predictor"
    device_manufacturer: str = "HA ML Predictor"
    device_model: str = "Smart Room Occupancy Predictor"
    device_sw_version: str = "1.0.0"

    # Publishing configuration
    publishing_enabled: bool = True
    publish_system_status: bool = True
    status_update_interval_seconds: int = 300  # 5 minutes
    prediction_qos: int = 1  # QoS level for prediction messages
    system_qos: int = 0  # QoS level for system status
    retain_predictions: bool = True  # Retain prediction messages
    retain_system_status: bool = True  # Retain system status

    # Connection settings
    keepalive: int = 60
    connection_timeout: int = 30
    reconnect_delay_seconds: int = 5
    max_reconnect_attempts: int = -1  # Infinite retries


@dataclass
class PredictionConfig:
    """Prediction system configuration."""

    interval_seconds: int = 300
    accuracy_threshold_minutes: int = 15
    confidence_threshold: float = 0.7


@dataclass
class FeaturesConfig:
    """Feature engineering configuration."""

    lookback_hours: int = 24
    sequence_length: int = 50
    temporal_features: bool = True
    sequential_features: bool = True
    contextual_features: bool = True


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    format: str = "structured"


@dataclass
class TrackingConfig:
    """Accuracy tracking system configuration."""

    enabled: bool = True
    monitoring_interval_seconds: int = 60
    auto_validation_enabled: bool = True
    validation_window_minutes: int = 30
    max_stored_alerts: int = 1000
    trend_analysis_points: int = 10
    cleanup_interval_hours: int = 24
    alert_thresholds: Dict[str, float] = None

    def __post_init__(self):
        """Set default alert thresholds if not provided."""
        if self.alert_thresholds is None:
            self.alert_thresholds = {
                "accuracy_warning": 70.0,
                "accuracy_critical": 50.0,
                "error_warning": 20.0,
                "error_critical": 30.0,
                "trend_degrading": -5.0,
                "validation_lag_warning": 15.0,
                "validation_lag_critical": 30.0,
            }


@dataclass
class APIConfig:
    """REST API server configuration."""

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Security configuration
    enable_cors: bool = True
    cors_origins: List[str] = None
    api_key_enabled: bool = False
    api_key: Optional[str] = None

    # Rate limiting
    rate_limit_enabled: bool = True
    requests_per_minute: int = 60
    burst_limit: int = 100

    # Request/response configuration
    request_timeout_seconds: int = 30
    max_request_size_mb: int = 10
    include_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"

    # Background task configuration
    background_tasks_enabled: bool = True
    health_check_interval_seconds: int = 60

    # Logging configuration
    access_log: bool = True
    log_requests: bool = True
    log_responses: bool = False  # Can be verbose

    def __post_init__(self):
        """Set default CORS origins if not provided."""
        if self.cors_origins is None:
            self.cors_origins = ["*"]  # Allow all origins by default


@dataclass
class SensorConfig:
    """Individual sensor configuration."""

    entity_id: str
    sensor_type: str
    room_id: str


@dataclass
class RoomConfig:
    """Room configuration with sensors."""

    room_id: str
    name: str
    sensors: Dict[str, Any] = field(default_factory=dict)

    def get_all_entity_ids(self) -> List[str]:
        """Extract all entity IDs from sensors configuration."""
        entity_ids = []

        def extract_ids(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, str) and value.startswith(
                        ("binary_sensor.", "sensor.")
                    ):
                        entity_ids.append(value)
                    else:
                        extract_ids(value)
            elif isinstance(obj, list):
                for item in obj:
                    extract_ids(item)

        extract_ids(self.sensors)
        return entity_ids

    def get_sensors_by_type(self, sensor_type: str) -> Dict[str, str]:
        """Get sensors of a specific type."""
        sensors = {}
        if sensor_type in self.sensors:
            if isinstance(self.sensors[sensor_type], dict):
                sensors.update(self.sensors[sensor_type])
            elif isinstance(self.sensors[sensor_type], str):
                sensors[sensor_type] = self.sensors[sensor_type]
        return sensors


@dataclass
class SystemConfig:
    """Main system configuration."""

    home_assistant: HomeAssistantConfig
    database: DatabaseConfig
    mqtt: MQTTConfig
    prediction: PredictionConfig
    features: FeaturesConfig
    logging: LoggingConfig
    tracking: TrackingConfig
    api: APIConfig
    rooms: Dict[str, RoomConfig] = field(default_factory=dict)

    def get_all_entity_ids(self) -> List[str]:
        """Get all entity IDs from all rooms."""
        entity_ids = []
        for room in self.rooms.values():
            entity_ids.extend(room.get_all_entity_ids())
        return list(set(entity_ids))  # Remove duplicates

    def get_room_by_entity_id(self, entity_id: str) -> Optional[RoomConfig]:
        """Find which room contains a specific entity ID."""
        for room in self.rooms.values():
            if entity_id in room.get_all_entity_ids():
                return room
        return None


class ConfigLoader:
    """Loads and validates configuration from YAML files."""

    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        if not self.config_dir.exists():
            raise FileNotFoundError(f"Configuration directory not found: {config_dir}")

    def load_config(self) -> SystemConfig:
        """Load complete system configuration."""
        # Load main config
        main_config = self._load_yaml("config.yaml")

        # Load rooms config
        rooms_config = self._load_yaml("rooms.yaml")

        # Create configuration objects
        ha_config = HomeAssistantConfig(**main_config["home_assistant"])
        db_config = DatabaseConfig(**main_config["database"])
        mqtt_config = MQTTConfig(**main_config["mqtt"])
        prediction_config = PredictionConfig(**main_config["prediction"])
        features_config = FeaturesConfig(**main_config["features"])
        logging_config = LoggingConfig(**main_config["logging"])
        tracking_config = TrackingConfig(**main_config.get("tracking", {}))
        api_config = APIConfig(**main_config.get("api", {}))

        # Process rooms configuration
        rooms = {}
        for room_id, room_data in rooms_config["rooms"].items():
            # Handle nested room structure (like hallways)
            if any(isinstance(v, dict) and "name" in v for v in room_data.values()):
                # This is a nested structure like hallways
                for sub_room_id, sub_room_data in room_data.items():
                    if isinstance(sub_room_data, dict) and "name" in sub_room_data:
                        full_room_id = f"{room_id}_{sub_room_id}"
                        rooms[full_room_id] = RoomConfig(
                            room_id=full_room_id,
                            name=sub_room_data["name"],
                            sensors=sub_room_data.get("sensors", {}),
                        )
            else:
                # Regular room structure
                rooms[room_id] = RoomConfig(
                    room_id=room_id,
                    name=room_data.get("name", room_id.replace("_", " ").title()),
                    sensors=room_data.get("sensors", {}),
                )

        return SystemConfig(
            home_assistant=ha_config,
            database=db_config,
            mqtt=mqtt_config,
            prediction=prediction_config,
            features=features_config,
            logging=logging_config,
            tracking=tracking_config,
            api=api_config,
            rooms=rooms,
        )

    def _load_yaml(self, filename: str) -> Dict[str, Any]:
        """Load YAML file from config directory."""
        file_path = self.config_dir / filename
        if not file_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {file_path}")

        with open(file_path, "r", encoding="utf-8") as file:
            return yaml.safe_load(file)


# Global configuration instance
_config_instance: Optional[SystemConfig] = None


def get_config() -> SystemConfig:
    """Get global configuration instance."""
    global _config_instance
    if _config_instance is None:
        loader = ConfigLoader()
        _config_instance = loader.load_config()
    return _config_instance


def reload_config() -> SystemConfig:
    """Reload configuration from files."""
    global _config_instance
    loader = ConfigLoader()
    _config_instance = loader.load_config()
    return _config_instance
