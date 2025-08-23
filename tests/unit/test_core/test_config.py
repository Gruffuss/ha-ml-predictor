"""
Unit tests for configuration management.

Tests ConfigLoader, SystemConfig, and all configuration dataclasses.
"""

import gc
from pathlib import Path
import sys
import tempfile
import threading
import time
from unittest.mock import mock_open, patch

import pytest
import yaml

from src.core.config import (
    APIConfig,
    ConfigLoader,
    DatabaseConfig,
    FeaturesConfig,
    HomeAssistantConfig,
    JWTConfig,
    LoggingConfig,
    MQTTConfig,
    PredictionConfig,
    RoomConfig,
    SensorConfig,
    SystemConfig,
    TrackingConfig,
    get_config,
    reload_config,
)
from src.core.exceptions import ConfigFileNotFoundError, ConfigValidationError


class TestHomeAssistantConfig:
    """Test HomeAssistantConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = HomeAssistantConfig(url="http://test:8123", token="test_token")
        assert config.url == "http://test:8123"
        assert config.token == "test_token"
        assert config.websocket_timeout == 30
        assert config.api_timeout == 10

    def test_custom_values(self):
        """Test custom values override defaults."""
        config = HomeAssistantConfig(
            url="https://ha.example.com",
            token="custom_token",
            websocket_timeout=45,
            api_timeout=15,
        )
        assert config.url == "https://ha.example.com"
        assert config.token == "custom_token"
        assert config.websocket_timeout == 45
        assert config.api_timeout == 15


class TestDatabaseConfig:
    """Test DatabaseConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = DatabaseConfig(connection_string="postgresql://test")
        assert config.connection_string == "postgresql://test"
        assert config.pool_size == 10
        assert config.max_overflow == 20

    def test_custom_values(self):
        """Test custom values override defaults."""
        config = DatabaseConfig(
            connection_string="postgresql://postgres:password@localhost/testdb",
            pool_size=5,
            max_overflow=15,
        )
        assert (
            config.connection_string
            == "postgresql://postgres:password@localhost/testdb"
        )
        assert config.pool_size == 5
        assert config.max_overflow == 15


class TestMQTTConfig:
    """Test MQTTConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = MQTTConfig(broker="localhost")
        assert config.broker == "localhost"
        assert config.port == 1883
        assert config.username == ""
        assert config.password == ""
        assert config.topic_prefix == "occupancy/predictions"

    def test_custom_values(self):
        """Test custom values override defaults."""
        config = MQTTConfig(
            broker="mqtt.example.com",
            port=8883,
            username="mqtt_user",
            password="mqtt_pass",
            topic_prefix="custom/topic",
        )
        assert config.broker == "mqtt.example.com"
        assert config.port == 8883
        assert config.username == "mqtt_user"
        assert config.password == "mqtt_pass"
        assert config.topic_prefix == "custom/topic"


class TestPredictionConfig:
    """Test PredictionConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = PredictionConfig()
        assert config.interval_seconds == 300
        assert config.accuracy_threshold_minutes == 15
        assert config.confidence_threshold == 0.7


class TestFeaturesConfig:
    """Test FeaturesConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = FeaturesConfig()
        assert config.lookback_hours == 24
        assert config.sequence_length == 50
        assert config.temporal_features is True
        assert config.sequential_features is True
        assert config.contextual_features is True


class TestLoggingConfig:
    """Test LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default values are set correctly."""
        config = LoggingConfig()
        assert config.level == "INFO"
        assert config.format == "structured"


class TestTrackingConfig:
    """Test TrackingConfig dataclass and methods."""

    def test_default_values(self):
        """Test default tracking configuration values."""
        config = TrackingConfig()

        assert config.enabled is True
        assert config.monitoring_interval_seconds == 60
        assert config.auto_validation_enabled is True
        assert config.validation_window_minutes == 30
        assert config.drift_detection_enabled is True
        assert config.drift_threshold == 0.1
        assert config.auto_retraining_enabled is True
        assert config.adaptive_retraining_enabled is True

    def test_alert_thresholds_post_init(self):
        """Test that alert thresholds are set in __post_init__."""
        config = TrackingConfig()

        assert config.alert_thresholds is not None
        assert "accuracy_warning" in config.alert_thresholds
        assert "accuracy_critical" in config.alert_thresholds
        assert config.alert_thresholds["accuracy_warning"] == 70.0
        assert config.alert_thresholds["accuracy_critical"] == 50.0


class TestJWTConfig:
    """Test JWTConfig dataclass and methods."""

    def test_default_values(self):
        """Test default JWT configuration values."""
        import os

        # Set test environment
        os.environ["ENVIRONMENT"] = "test"
        os.environ["JWT_SECRET_KEY"] = (
            "test_jwt_secret_key_for_testing_purposes_at_least_32_characters_long"
        )

        config = JWTConfig()

        assert config.enabled is True
        assert config.algorithm == "HS256"
        assert config.access_token_expire_minutes == 60
        assert config.refresh_token_expire_days == 30
        assert config.issuer == "ha-ml-predictor"
        assert config.audience == "ha-ml-predictor-api"
        assert len(config.secret_key) >= 32

    def test_jwt_disabled_via_env(self):
        """Test JWT can be disabled via environment variable."""
        import os

        os.environ["JWT_ENABLED"] = "false"

        config = JWTConfig()

        assert config.enabled is False


class TestAPIConfig:
    """Test APIConfig dataclass and methods."""

    def test_default_values(self):
        """Test default API configuration values."""
        import os

        # Clear any existing environment variables that might interfere
        for key in ["API_ENABLED", "API_HOST", "API_PORT", "JWT_SECRET_KEY"]:
            if key in os.environ:
                del os.environ[key]

        # Set test environment
        os.environ["ENVIRONMENT"] = "test"
        os.environ["JWT_SECRET_KEY"] = (
            "test_jwt_secret_key_for_testing_purposes_at_least_32_characters_long"
        )

        config = APIConfig()

        assert config.enabled is True
        assert config.host == "0.0.0.0"
        assert config.port == 8000
        assert config.debug is False
        assert config.enable_cors is True
        assert config.rate_limit_enabled is True
        assert config.requests_per_minute == 60
        assert config.burst_limit == 100
        assert config.background_tasks_enabled is True

    def test_environment_overrides(self):
        """Test that environment variables override defaults."""
        import os

        os.environ["API_ENABLED"] = "false"
        os.environ["API_HOST"] = "127.0.0.1"
        os.environ["API_PORT"] = "9000"
        os.environ["API_DEBUG"] = "true"
        os.environ["JWT_SECRET_KEY"] = (
            "test_jwt_secret_key_for_testing_purposes_at_least_32_characters_long"
        )

        config = APIConfig()

        assert config.enabled is False
        assert config.host == "127.0.0.1"
        assert config.port == 9000
        assert config.debug is True

        # Clean up
        for key in ["API_ENABLED", "API_HOST", "API_PORT", "API_DEBUG"]:
            if key in os.environ:
                del os.environ[key]


class TestRoomConfig:
    """Test RoomConfig dataclass and methods."""

    def test_basic_room_config(self):
        """Test basic room configuration."""
        config = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "presence": {"main": "binary_sensor.test_presence"},
                "door": "binary_sensor.test_door",
            },
        )
        assert config.room_id == "test_room"
        assert config.name == "Test Room"
        assert config.sensors["presence"]["main"] == "binary_sensor.test_presence"
        assert config.sensors["door"] == "binary_sensor.test_door"

    def test_get_all_entity_ids(self):
        """Test extraction of all entity IDs from sensors configuration."""
        config = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "presence": {
                    "main": "binary_sensor.test_presence",
                    "secondary": "binary_sensor.test_motion",
                },
                "door": "binary_sensor.test_door",
                "climate": {
                    "temperature": "sensor.test_temperature",
                    "humidity": "sensor.test_humidity",
                },
                "nested": {"level1": {"level2": "sensor.nested_sensor"}},
            },
        )

        entity_ids = config.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.test_presence",
            "binary_sensor.test_motion",
            "binary_sensor.test_door",
            "sensor.test_temperature",
            "sensor.test_humidity",
            "sensor.nested_sensor",
        ]

        # Sort both lists for comparison
        assert sorted(entity_ids) == sorted(expected_ids)

    def test_get_all_entity_ids_empty_sensors(self):
        """Test get_all_entity_ids with empty sensors."""
        config = RoomConfig(room_id="test_room", name="Test Room", sensors={})
        assert config.get_all_entity_ids() == []

    def test_get_sensors_by_type(self):
        """Test getting sensors by type."""
        config = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "presence": {
                    "main": "binary_sensor.test_presence",
                    "secondary": "binary_sensor.test_motion",
                },
                "door": "binary_sensor.test_door",
                "climate": {"temperature": "sensor.test_temperature"},
            },
        )

        # Test dict type sensors
        presence_sensors = config.get_sensors_by_type("presence")
        expected_presence = {
            "main": "binary_sensor.test_presence",
            "secondary": "binary_sensor.test_motion",
        }
        assert presence_sensors == expected_presence

        # Test string type sensors
        door_sensors = config.get_sensors_by_type("door")
        assert door_sensors == {"door": "binary_sensor.test_door"}

        # Test nested dict
        climate_sensors = config.get_sensors_by_type("climate")
        assert climate_sensors == {"temperature": "sensor.test_temperature"}

        # Test non-existent type
        nonexistent = config.get_sensors_by_type("nonexistent")
        assert nonexistent == {}


class TestSystemConfig:
    """Test SystemConfig dataclass and methods."""

    def test_system_config_creation(self):
        """Test SystemConfig creation with all components."""
        ha_config = HomeAssistantConfig(url="http://test:8123", token="test_token")
        db_config = DatabaseConfig(connection_string="postgresql://test")
        mqtt_config = MQTTConfig(broker="localhost")
        pred_config = PredictionConfig()
        feat_config = FeaturesConfig()
        log_config = LoggingConfig()
        tracking_config = TrackingConfig()
        api_config = APIConfig()

        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={"presence": {"main": "binary_sensor.room1_presence"}},
            ),
            "room2": RoomConfig(
                room_id="room2",
                name="Room 2",
                sensors={"door": "binary_sensor.room2_door"},
            ),
        }

        system_config = SystemConfig(
            home_assistant=ha_config,
            database=db_config,
            mqtt=mqtt_config,
            prediction=pred_config,
            features=feat_config,
            logging=log_config,
            tracking=tracking_config,
            api=api_config,
            rooms=rooms,
        )

        assert system_config.home_assistant == ha_config
        assert system_config.database == db_config
        assert system_config.mqtt == mqtt_config
        assert system_config.prediction == pred_config
        assert system_config.features == feat_config
        assert system_config.logging == log_config
        assert system_config.tracking == tracking_config
        assert system_config.api == api_config
        assert len(system_config.rooms) == 2

    def test_get_all_entity_ids(self):
        """Test getting all entity IDs from all rooms."""
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={
                    "presence": {"main": "binary_sensor.room1_presence"},
                    "door": "binary_sensor.room1_door",
                },
            ),
            "room2": RoomConfig(
                room_id="room2",
                name="Room 2",
                sensors={
                    "presence": {"main": "binary_sensor.room2_presence"},
                    "climate": {"temperature": "sensor.room2_temperature"},
                },
            ),
        }

        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(),
            api=APIConfig(),
            rooms=rooms,
        )

        entity_ids = system_config.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.room1_presence",
            "binary_sensor.room1_door",
            "binary_sensor.room2_presence",
            "sensor.room2_temperature",
        ]

        assert sorted(entity_ids) == sorted(expected_ids)

    def test_get_all_entity_ids_with_duplicates(self):
        """Test that get_all_entity_ids removes duplicates."""
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={"presence": {"main": "binary_sensor.shared_sensor"}},
            ),
            "room2": RoomConfig(
                room_id="room2",
                name="Room 2",
                sensors={
                    "motion": "binary_sensor.shared_sensor"
                },  # Same sensor in different room
            ),
        }

        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(),
            api=APIConfig(),
            rooms=rooms,
        )

        entity_ids = system_config.get_all_entity_ids()
        assert entity_ids == ["binary_sensor.shared_sensor"]  # No duplicates

    def test_get_room_by_entity_id(self):
        """Test finding room by entity ID."""
        room1 = RoomConfig(
            room_id="room1",
            name="Room 1",
            sensors={"presence": {"main": "binary_sensor.room1_presence"}},
        )
        room2 = RoomConfig(
            room_id="room2",
            name="Room 2",
            sensors={"door": "binary_sensor.room2_door"},
        )

        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            tracking=TrackingConfig(),
            api=APIConfig(),
            rooms={"room1": room1, "room2": room2},
        )

        # Test finding existing entities
        found_room1 = system_config.get_room_by_entity_id(
            "binary_sensor.room1_presence"
        )
        assert found_room1 == room1

        found_room2 = system_config.get_room_by_entity_id("binary_sensor.room2_door")
        assert found_room2 == room2

        # Test non-existent entity
        not_found = system_config.get_room_by_entity_id("binary_sensor.nonexistent")
        assert not_found is None


class TestConfigLoader:
    """Test ConfigLoader class."""

    def test_config_loader_init_valid_dir(self, test_config_dir):
        """Test ConfigLoader initialization with valid directory."""
        loader = ConfigLoader(test_config_dir)
        assert loader.config_dir == Path(test_config_dir)

    def test_config_loader_init_invalid_dir(self):
        """Test ConfigLoader initialization with invalid directory."""
        with pytest.raises(FileNotFoundError):
            ConfigLoader("/nonexistent/directory")

    def test_load_config_success(self, test_config_dir):
        """Test successful configuration loading."""
        loader = ConfigLoader(test_config_dir)
        config = loader.load_config()

        # Test basic structure
        assert isinstance(config, SystemConfig)
        assert isinstance(config.home_assistant, HomeAssistantConfig)
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.mqtt, MQTTConfig)
        assert isinstance(config.prediction, PredictionConfig)
        assert isinstance(config.features, FeaturesConfig)
        assert isinstance(config.logging, LoggingConfig)
        assert isinstance(config.tracking, TrackingConfig)
        assert isinstance(config.api, APIConfig)

        # Test values from test config
        assert config.home_assistant.url == "http://test-ha:8123"
        assert config.home_assistant.token == "test_token_12345"
        # Check database config - should use TEST_DB_URL (SQLite for tests)
        assert (
            "sqlite" in config.database.connection_string
            or "postgresql" in config.database.connection_string
        )
        assert config.mqtt.broker == "test-mqtt"

        # Test rooms were loaded
        assert len(config.rooms) > 0
        assert "test_room" in config.rooms
        assert config.rooms["test_room"].name == "Test Room"

    def test_load_config_missing_main_config(self, test_config_dir):
        """Test loading configuration with missing main config file."""
        # Remove main config file
        config_path = Path(test_config_dir) / "config.yaml"
        config_path.unlink()

        loader = ConfigLoader(test_config_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_config()

    def test_load_config_missing_rooms_config(self, test_config_dir):
        """Test loading configuration with missing rooms config file."""
        # Remove rooms config file
        rooms_path = Path(test_config_dir) / "rooms.yaml"
        rooms_path.unlink()

        loader = ConfigLoader(test_config_dir)
        with pytest.raises(FileNotFoundError):
            loader.load_config()

    def test_load_config_nested_rooms(self):
        """Test loading configuration with nested room structure."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create main config
            main_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://localhost/testdb"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Create rooms config with nested structure
            rooms_config = {
                "rooms": {
                    "hallways": {
                        "ground_floor": {
                            "name": "Ground Floor Hallway",
                            "sensors": {"presence": "binary_sensor.ground_hallway"},
                        },
                        "upper": {
                            "name": "Upper Hallway",
                            "sensors": {"presence": "binary_sensor.upper_hallway"},
                        },
                    },
                    "living_room": {
                        "name": "Living Room",
                        "sensors": {"presence": "binary_sensor.living_room"},
                    },
                }
            }

            # Write config files
            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(main_config, f)
            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(rooms_config, f)

            loader = ConfigLoader(str(config_dir))
            config = loader.load_config()

            # Check nested rooms were flattened correctly
            assert "hallways_ground_floor" in config.rooms
            assert "hallways_upper" in config.rooms
            assert "living_room" in config.rooms

            assert config.rooms["hallways_ground_floor"].name == "Ground Floor Hallway"
            assert config.rooms["hallways_upper"].name == "Upper Hallway"
            assert config.rooms["living_room"].name == "Living Room"

    @patch("builtins.open", mock_open(read_data="invalid: yaml: content: ["))
    def test_load_yaml_invalid_format(self, test_config_dir):
        """Test loading invalid YAML file."""
        loader = ConfigLoader(test_config_dir)
        with pytest.raises(yaml.YAMLError):
            loader._load_yaml("config.yaml")


class TestGlobalConfigFunctions:
    """Test global configuration functions."""

    def test_get_config_singleton(self, test_config_dir):
        """Test that get_config returns singleton instance."""
        # Clear any existing instance
        import src.core.config

        src.core.config._config_instance = None

        # Patch the import to force ImportError and use ConfigLoader path
        with patch.dict("sys.modules", {"src.core.environment": None}):
            with patch("src.core.config.ConfigLoader") as mock_loader:
                mock_config = SystemConfig(
                    home_assistant=HomeAssistantConfig(url="test", token="test"),
                    database=DatabaseConfig(connection_string="test"),
                    mqtt=MQTTConfig(broker="test"),
                    prediction=PredictionConfig(),
                    features=FeaturesConfig(),
                    logging=LoggingConfig(),
                    tracking=TrackingConfig(),
                    api=APIConfig(),
                )
                mock_loader.return_value.load_config.return_value = mock_config

                # First call should create instance
                config1 = get_config()

                # Second call should return same instance
                config2 = get_config()

                assert config1 is config2
                mock_loader.assert_called_once()

    def test_reload_config(self, test_config_dir):
        """Test config reloading."""
        # Clear any existing instance
        import src.core.config

        src.core.config._config_instance = None

        # Patch the import to force ImportError and use ConfigLoader path
        with patch.dict("sys.modules", {"src.core.environment": None}):
            with patch("src.core.config.ConfigLoader") as mock_loader:
                mock_config1 = SystemConfig(
                    home_assistant=HomeAssistantConfig(url="test1", token="test1"),
                    database=DatabaseConfig(connection_string="test1"),
                    mqtt=MQTTConfig(broker="test1"),
                    prediction=PredictionConfig(),
                    features=FeaturesConfig(),
                    logging=LoggingConfig(),
                    tracking=TrackingConfig(),
                    api=APIConfig(),
                )
                mock_config2 = SystemConfig(
                    home_assistant=HomeAssistantConfig(url="test2", token="test2"),
                    database=DatabaseConfig(connection_string="test2"),
                    mqtt=MQTTConfig(broker="test2"),
                    prediction=PredictionConfig(),
                    features=FeaturesConfig(),
                    logging=LoggingConfig(),
                    tracking=TrackingConfig(),
                    api=APIConfig(),
                )

                mock_loader.return_value.load_config.side_effect = [
                    mock_config1,
                    mock_config2,
                ]

                # Load initial config
                config1 = get_config()
                assert config1.home_assistant.url == "test1"

                # Reload config
                config2 = reload_config()
                assert config2.home_assistant.url == "test2"

                # Verify new config is returned by get_config
                config3 = get_config()
                assert config3 is config2
                assert config3.home_assistant.url == "test2"


class TestConfigValidation:
    """Test configuration validation and error handling."""

    def test_missing_required_fields(self):
        """Test behavior with missing required configuration fields."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create incomplete main config (missing required fields)
            incomplete_config = {
                "home_assistant": {"url": "http://test:8123"},  # Missing token
                "database": {},  # Missing connection_string
                "mqtt": {},  # Missing broker
                "prediction": {},
                "features": {},
                "logging": {},
            }

            rooms_config = {"rooms": {}}

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(incomplete_config, f)
            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(rooms_config, f)

            loader = ConfigLoader(str(config_dir))

            # Should raise TypeError for missing required fields
            with pytest.raises(TypeError):
                loader.load_config()

    def test_room_config_with_missing_name(self):
        """Test room configuration with missing name field."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            main_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://localhost/testdb"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Room without explicit name should get auto-generated name
            rooms_config = {
                "rooms": {"test_room": {"sensors": {"presence": "binary_sensor.test"}}}
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(main_config, f)
            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(rooms_config, f)

            loader = ConfigLoader(str(config_dir))
            config = loader.load_config()

            # Should auto-generate name from room_id
            assert config.rooms["test_room"].name == "Test Room"


@pytest.mark.unit
class TestConfigIntegration:
    """Integration tests for configuration loading."""

    def test_load_real_config_structure(self, test_config_dir):
        """Test loading configuration with realistic structure."""
        loader = ConfigLoader(test_config_dir)
        config = loader.load_config()

        # Verify complete configuration structure
        assert isinstance(config, SystemConfig)

        # Check Home Assistant config
        assert config.home_assistant.url == "http://test-ha:8123"
        assert config.home_assistant.token == "test_token_12345"
        assert config.home_assistant.websocket_timeout == 30
        assert config.home_assistant.api_timeout == 10

        # Check database config - should use TEST_DB_URL (SQLite for tests)
        assert (
            "sqlite" in config.database.connection_string
            or "postgresql" in config.database.connection_string
        )
        assert config.database.pool_size == 5
        assert config.database.max_overflow == 10

        # Check MQTT config
        assert config.mqtt.broker == "test-mqtt"
        assert config.mqtt.port == 1883
        assert config.mqtt.topic_prefix == "test/occupancy"

        # Check prediction config
        assert config.prediction.interval_seconds == 300
        assert config.prediction.accuracy_threshold_minutes == 15
        assert config.prediction.confidence_threshold == 0.7

        # Check features config
        assert config.features.lookback_hours == 24
        assert config.features.sequence_length == 50
        assert config.features.temporal_features is True

        # Check logging config
        assert config.logging.level == "DEBUG"
        assert config.logging.format == "structured"

        # Check rooms loaded correctly
        assert len(config.rooms) >= 2
        assert "test_room" in config.rooms
        assert "living_room" in config.rooms

        # Check room entity extraction works
        entity_ids = config.get_all_entity_ids()
        assert len(entity_ids) > 0
        assert all(
            entity_id.startswith(("binary_sensor.", "sensor."))
            for entity_id in entity_ids
        )

        # Check room lookup works
        if entity_ids:
            found_room = config.get_room_by_entity_id(entity_ids[0])
            assert found_room is not None
            assert isinstance(found_room, RoomConfig)


class TestConfigurationBoundaryConditions:
    """Test configuration loading with boundary conditions."""

    def test_extremely_large_configuration_file(self):
        """Test loading extremely large configuration files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create large configuration
            large_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Add many repeated sections to make it large
            for i in range(50):  # Reduced for CI performance
                large_config[f"dummy_section_{i}"] = {
                    f"key_{j}": f"value_{j}_" + "x" * 50  # Medium length values
                    for j in range(20)  # 20 keys per section
                }

            # Create large rooms config
            large_rooms = {"rooms": {}}
            for i in range(100):  # 100 rooms
                room_id = f"room_{i:04d}"
                large_rooms["rooms"][room_id] = {
                    "name": f"Room {i:04d} with a long name",
                    "sensors": {
                        "presence": {
                            f"sensor_{j}": f"binary_sensor.room_{i:04d}_presence_{j}"
                            for j in range(3)  # 3 sensors per room
                        },
                    },
                }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(large_config, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(large_rooms, f)

            loader = ConfigLoader(str(config_dir))

            start_time = time.time()
            config = loader.load_config()
            load_time = time.time() - start_time

            # Should load successfully
            assert isinstance(config, SystemConfig)
            assert len(config.rooms) == 100
            assert load_time < 30.0  # Should load within 30 seconds

    def test_configuration_with_extreme_values(self):
        """Test configuration with extreme boundary values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            extreme_config = {
                "home_assistant": {
                    "url": "http://test:8123",
                    "token": "test",
                    "websocket_timeout": 0,  # Minimum value
                    "api_timeout": 86400,  # Very large value
                },
                "database": {
                    "connection_string": "postgresql://test",
                    "pool_size": 1,  # Minimum pool
                    "max_overflow": 1000,  # Very large overflow
                },
                "mqtt": {
                    "broker": "test",
                    "port": 65535,  # Maximum port number
                    "username": "",  # Empty string
                    "password": "",
                },
                "prediction": {
                    "interval_seconds": 1,  # Very frequent
                    "accuracy_threshold_minutes": 10080,  # One week
                },
                "features": {
                    "lookback_hours": 8760,  # One year
                    "sequence_length": 1,  # Minimum sequence
                },
                "logging": {"level": "DEBUG"},
            }

            rooms_config = {"rooms": {}}

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(extreme_config, f)
            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(rooms_config, f)

            loader = ConfigLoader(str(config_dir))
            config = loader.load_config()

            # Should handle extreme values gracefully
            assert config.home_assistant.websocket_timeout == 0
            assert config.home_assistant.api_timeout == 86400
            assert config.database.pool_size == 1
            assert config.database.max_overflow == 1000
            assert config.mqtt.port == 65535
            assert config.prediction.interval_seconds == 1
            assert config.features.lookback_hours == 8760

    def test_configuration_with_unicode_edge_cases(self):
        """Test configuration with various Unicode edge cases."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            unicode_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Rooms with Unicode characters
            unicode_rooms = {
                "rooms": {
                    "café_room": {
                        "name": "Café Room ☕",
                        "sensors": {"presence": "binary_sensor.café"},
                    },
                    "测试房间": {
                        "name": "测试房间 (Test Room)",
                        "sensors": {"presence": "binary_sensor.test_chinese"},
                    },
                    "комната": {
                        "name": "Русская комната",
                        "sensors": {"presence": "binary_sensor.russian"},
                    },
                    "🏠_room": {
                        "name": "Emoji Room 🏠🔥❄️",
                        "sensors": {"presence": "binary_sensor.emoji"},
                    },
                }
            }

            with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
                yaml.dump(unicode_config, f, allow_unicode=True)
            with open(config_dir / "rooms.yaml", "w", encoding="utf-8") as f:
                yaml.dump(unicode_rooms, f, allow_unicode=True)

            loader = ConfigLoader(str(config_dir))
            config = loader.load_config()

            # Should handle Unicode correctly
            assert "café_room" in config.rooms
            assert config.rooms["café_room"].name == "Café Room ☕"
            assert "测试房间" in config.rooms
            assert config.rooms["测试房间"].name == "测试房间 (Test Room)"
            assert "комната" in config.rooms
            assert "🏠_room" in config.rooms


class TestConfigurationMemoryAndResourceConstraints:
    """Test configuration loading under resource constraints."""

    def test_concurrent_configuration_loading_stress(self):
        """Test concurrent configuration loading from multiple threads."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create standard config
            test_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            rooms_config = {
                "rooms": {
                    f"room_{i}": {
                        "name": f"Room {i}",
                        "sensors": {"presence": f"binary_sensor.room_{i}"},
                    }
                    for i in range(50)
                }
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(test_config, f)
            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(rooms_config, f)

            # Test concurrent loading
            results = []
            exceptions = []

            def load_config_thread():
                try:
                    loader = ConfigLoader(str(config_dir))
                    config = loader.load_config()
                    results.append(config)
                except Exception as e:
                    exceptions.append(e)

            # Start multiple threads
            threads = []
            for _ in range(5):  # Reduced for CI stability
                thread = threading.Thread(target=load_config_thread)
                threads.append(thread)
                thread.start()

            # Wait for completion
            for thread in threads:
                thread.join()

            # All should succeed
            assert len(exceptions) == 0
            assert len(results) == 5

            # All results should be equivalent
            for config in results:
                assert isinstance(config, SystemConfig)
                assert len(config.rooms) == 50


class TestConfigurationCorruptionAndRecovery:
    """Test configuration handling with data corruption and recovery."""

    def test_partially_corrupted_yaml_recovery(self):
        """Test recovery from partially corrupted YAML files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create YAML with corruption at end (should still parse beginning)
            partially_corrupted = """
home_assistant:
  url: "http://test:8123"
  token: "test_token"
database:
  connection_string: "postgresql://test"
mqtt:
  broker: "test"
prediction: {}
features: {}
logging: {}
# Corruption starts here: ñÿþ¤ª¨¦§°±²³´µ¶·¸¹º»¼½¾¿ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖ×ØÙÚÛÜÝÞßàáâãäåæçèéêëìíîïðñòóôõö÷øùúûüýþÿ
"""
            with open(config_dir / "config.yaml", "w", encoding="utf-8") as f:
                f.write(partially_corrupted)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Should load successfully despite corruption at end
            config = loader.load_config()
            assert config.home_assistant.url == "http://test:8123"
            assert config.database.connection_string == "postgresql://test"

    def test_yaml_with_mixed_encodings(self):
        """Test YAML files with mixed or invalid encodings."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create file with mixed encoding issues
            config_data = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            config_file = config_dir / "config.yaml"
            # Write as UTF-8
            with open(config_file, "w", encoding="utf-8") as f:
                yaml.dump(config_data, f)

            # Append binary data that's not valid UTF-8
            with open(config_file, "ab") as f:
                f.write(b"\xff\xfe\x00\x01\x02\x03")

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Should handle encoding issues gracefully
            try:
                config = loader.load_config()
                # If it succeeds, verify it got the valid part
                assert config.home_assistant.url == "http://test:8123"
            except UnicodeDecodeError:
                # Acceptable to fail with encoding error
                pass

    def test_yaml_with_circular_references_deep(self):
        """Test YAML with deep circular references."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create YAML with deeply nested circular references
            complex_circular = """
home_assistant: &ha
  url: "http://test:8123"
  token: "test_token"
  reference: &ref1
    back_ref: *ha
    nested: &ref2
      deep_ref: *ref1
      deeper: &ref3
        deepest_ref: *ref2
        back_to_ha: *ha

database:
  connection_string: "postgresql://test"
  ha_config: *ha

mqtt:
  broker: "test"
  config_ref: *ref3

prediction: {}
features: {}
logging: {}
"""
            with open(config_dir / "config.yaml", "w") as f:
                f.write(complex_circular)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # YAML parser should handle references without infinite loops
            config = loader.load_config()
            assert config.home_assistant.url == "http://test:8123"

    def test_configuration_with_extremely_long_values(self):
        """Test configuration with extremely long string values."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config with very long values
            extremely_long_token = "token_" + "x" * 100000  # 100KB token
            extremely_long_url = "http://" + "subdomain." * 1000 + "example.com:8123"

            long_config = {
                "home_assistant": {
                    "url": extremely_long_url,
                    "token": extremely_long_token,
                },
                "database": {
                    "connection_string": "postgresql://user:pass@"
                    + "host." * 100
                    + "com/db"
                },
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(long_config, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Should handle very long values
            config = loader.load_config()
            assert len(config.home_assistant.token) > 100000
            assert config.home_assistant.token.startswith("token_x")

    def test_configuration_with_malformed_data_types(self):
        """Test configuration with malformed data types."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config with wrong data types that might cause issues
            malformed_config = {
                "home_assistant": {
                    "url": ["http://test:8123"],  # List instead of string
                    "token": {"value": "test"},  # Dict instead of string
                    "websocket_timeout": "thirty",  # String instead of int
                    "api_timeout": [10, 20, 30],  # List instead of int
                },
                "database": {
                    "connection_string": 12345,  # Int instead of string
                    "pool_size": {"min": 5, "max": 10},  # Dict instead of int
                    "max_overflow": "unlimited",  # String instead of int
                },
                "mqtt": {
                    "broker": None,  # None instead of string
                    "port": "1883",  # String that can be converted
                    "username": {"user": "test"},  # Dict instead of string
                },
                "prediction": {
                    "interval_seconds": [300],  # List instead of int
                    "confidence_threshold": "70%",  # String instead of float
                },
                "features": {"temporal_features": "yes"},  # String instead of bool
                "logging": {"level": 123},  # Int instead of string
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(malformed_config, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Should fail with type errors when trying to create dataclasses
            with pytest.raises(TypeError):
                loader.load_config()

    def test_configuration_with_yaml_injection_attempts(self):
        """Test configuration security against YAML injection attempts."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # YAML with potential security issues (should be handled safely)
            potentially_malicious_yaml = """
home_assistant:
  url: "http://test:8123"
  token: "test_token"

# Attempt to execute Python code (should not execute)
database: !!python/object/apply:os.system ["echo 'this should not execute'"]

# Alternative malicious attempt
mqtt:
  broker: "test"
  malicious: !!python/module:os

prediction: {}
features: {}
logging: {}
"""
            with open(config_dir / "config.yaml", "w") as f:
                f.write(potentially_malicious_yaml)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # yaml.safe_load should prevent execution of Python code
            # This should either load safely or fail with a construction error
            try:
                config = loader.load_config()
                # If it loads, the malicious parts should be ignored/converted safely
                assert config.home_assistant.url == "http://test:8123"
            except yaml.constructor.ConstructorError:
                # Acceptable - safe_load prevented dangerous construction
                pass
            except Exception as e:
                # Should not execute arbitrary code
                assert "this should not execute" not in str(e)


class TestConfigurationRecoveryMechanisms:
    """Test configuration recovery and fallback mechanisms."""

    def test_configuration_fallback_to_defaults(self):
        """Test fallback to default values when config is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create minimal config that's missing many fields
            minimal_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                # Missing prediction, features, logging sections
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(minimal_config, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Should use defaults for missing sections
            config = loader.load_config()

            # Verify defaults are applied
            assert config.prediction.interval_seconds == 300  # Default
            assert config.features.lookback_hours == 24  # Default
            assert config.logging.level == "INFO"  # Default

    def test_configuration_partial_recovery(self):
        """Test recovery when only part of configuration is corrupted."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config where rooms.yaml is corrupted but config.yaml is fine
            good_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Corrupted rooms file
            corrupted_rooms = "rooms:\n  invalid_yaml: [\n  # Missing closing bracket"

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(good_config, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                f.write(corrupted_rooms)

            loader = ConfigLoader(str(config_dir))

            # Should fail because rooms.yaml is required
            with pytest.raises(yaml.YAMLError):
                loader.load_config()

    def test_configuration_validation_recovery(self):
        """Test recovery from configuration validation failures."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Config with invalid but recoverable values
            invalid_but_recoverable = {
                "home_assistant": {
                    "url": "",  # Empty URL
                    "token": "test_token",
                    "websocket_timeout": -1,  # Invalid timeout
                    "api_timeout": 10,
                },
                "database": {
                    "connection_string": "postgresql://test",
                    "pool_size": 0,  # Invalid pool size
                    "max_overflow": 20,
                },
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(invalid_but_recoverable, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Should fail validation due to empty URL and invalid values
            with pytest.raises(TypeError):
                loader.load_config()


class TestConfigurationStabilityAndResilience:
    """Test configuration system stability and resilience."""

    def test_repeated_configuration_loading_stability(self):
        """Test stability of repeated configuration loading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_data = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(config_data, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump({"rooms": {}}, f)

            loader = ConfigLoader(str(config_dir))

            # Load configuration many times
            configs = []
            for i in range(100):
                config = loader.load_config()
                configs.append(config)

                # Verify consistency
                assert config.home_assistant.url == "http://test:8123"
                assert config.database.connection_string == "postgresql://test"

            # All configs should be equivalent but separate objects
            for i in range(1, len(configs)):
                assert configs[i].home_assistant.url == configs[0].home_assistant.url
                assert configs[i] is not configs[0]  # Different objects

    def test_configuration_loading_with_system_stress(self):
        """Test configuration loading under system stress."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create moderately large config
            stress_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Add moderate complexity
            for i in range(50):
                stress_config[f"section_{i}"] = {
                    f"key_{j}": f"value_{j}" for j in range(20)
                }

            rooms_config = {
                "rooms": {f"room_{i}": {"name": f"Room {i}"} for i in range(200)}
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(stress_config, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(rooms_config, f)

            loader = ConfigLoader(str(config_dir))

            # Load under simulated stress (rapid repeated loads)
            start_time = time.time()
            successful_loads = 0

            while time.time() - start_time < 5.0:  # Run for 5 seconds
                try:
                    config = loader.load_config()
                    successful_loads += 1
                    assert len(config.rooms) == 200
                except Exception as e:
                    pytest.fail(f"Configuration loading failed under stress: {e}")

                # Brief pause to avoid overwhelming system
                time.sleep(0.01)

            # Should have completed many loads successfully
            assert successful_loads > 100  # At least 20 per second

    def test_configuration_memory_stability(self):
        """Test that configuration loading doesn't cause memory leaks."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_data = {
                "home_assistant": {"url": "http://test:8123", "token": "test"},
                "database": {"connection_string": "postgresql://test"},
                "mqtt": {"broker": "test"},
                "prediction": {},
                "features": {},
                "logging": {},
            }

            # Create config with some data to allocate memory
            large_rooms = {
                "rooms": {
                    f"room_{i}": {
                        "name": f"Room {i}",
                        "sensors": {
                            f"sensor_{j}": f"binary_sensor.room_{i}_sensor_{j}"
                            for j in range(10)
                        },
                    }
                    for i in range(100)
                }
            }

            with open(config_dir / "config.yaml", "w") as f:
                yaml.dump(config_data, f)

            with open(config_dir / "rooms.yaml", "w") as f:
                yaml.dump(large_rooms, f)

            loader = ConfigLoader(str(config_dir))

            # Force garbage collection and get initial memory
            gc.collect()
            initial_objects = len(gc.get_objects())

            # Load and release configurations multiple times
            for i in range(50):
                config = loader.load_config()
                # Verify it loaded correctly
                assert len(config.rooms) == 100
                # Release reference
                del config
                # Periodic garbage collection
                if i % 10 == 0:
                    gc.collect()

            # Final garbage collection
            gc.collect()
            final_objects = len(gc.get_objects())

            # Memory usage shouldn't grow significantly
            # (Allow some growth for test framework overhead)
            object_growth = final_objects - initial_objects
            assert object_growth < 1000  # Reasonable threshold
