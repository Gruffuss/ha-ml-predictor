"""
Unit tests for configuration management.

Tests ConfigLoader, SystemConfig, and all configuration dataclasses.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, mock_open

from src.core.config import (
    ConfigLoader, SystemConfig, HomeAssistantConfig, DatabaseConfig,
    MQTTConfig, PredictionConfig, FeaturesConfig, LoggingConfig,
    RoomConfig, SensorConfig, get_config, reload_config
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
            api_timeout=15
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
            connection_string="sqlite:///:memory:",
            pool_size=5,
            max_overflow=15
        )
        assert config.connection_string == "sqlite:///:memory:"
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
            topic_prefix="custom/topic"
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


class TestRoomConfig:
    """Test RoomConfig dataclass and methods."""
    
    def test_basic_room_config(self):
        """Test basic room configuration."""
        config = RoomConfig(
            room_id="test_room",
            name="Test Room",
            sensors={
                "presence": {"main": "binary_sensor.test_presence"},
                "door": "binary_sensor.test_door"
            }
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
                    "secondary": "binary_sensor.test_motion"
                },
                "door": "binary_sensor.test_door",
                "climate": {
                    "temperature": "sensor.test_temperature",
                    "humidity": "sensor.test_humidity"
                },
                "nested": {
                    "level1": {
                        "level2": "sensor.nested_sensor"
                    }
                }
            }
        )
        
        entity_ids = config.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.test_presence",
            "binary_sensor.test_motion",
            "binary_sensor.test_door",
            "sensor.test_temperature",
            "sensor.test_humidity",
            "sensor.nested_sensor"
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
                    "secondary": "binary_sensor.test_motion"
                },
                "door": "binary_sensor.test_door",
                "climate": {
                    "temperature": "sensor.test_temperature"
                }
            }
        )
        
        # Test dict type sensors
        presence_sensors = config.get_sensors_by_type("presence")
        expected_presence = {
            "main": "binary_sensor.test_presence",
            "secondary": "binary_sensor.test_motion"
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
        
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={"presence": {"main": "binary_sensor.room1_presence"}}
            ),
            "room2": RoomConfig(
                room_id="room2", 
                name="Room 2",
                sensors={"door": "binary_sensor.room2_door"}
            )
        }
        
        system_config = SystemConfig(
            home_assistant=ha_config,
            database=db_config,
            mqtt=mqtt_config,
            prediction=pred_config,
            features=feat_config,
            logging=log_config,
            rooms=rooms
        )
        
        assert system_config.home_assistant == ha_config
        assert system_config.database == db_config
        assert system_config.mqtt == mqtt_config
        assert system_config.prediction == pred_config
        assert system_config.features == feat_config
        assert system_config.logging == log_config
        assert len(system_config.rooms) == 2
    
    def test_get_all_entity_ids(self):
        """Test getting all entity IDs from all rooms."""
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={
                    "presence": {"main": "binary_sensor.room1_presence"},
                    "door": "binary_sensor.room1_door"
                }
            ),
            "room2": RoomConfig(
                room_id="room2",
                name="Room 2", 
                sensors={
                    "presence": {"main": "binary_sensor.room2_presence"},
                    "climate": {"temperature": "sensor.room2_temperature"}
                }
            )
        }
        
        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            rooms=rooms
        )
        
        entity_ids = system_config.get_all_entity_ids()
        expected_ids = [
            "binary_sensor.room1_presence",
            "binary_sensor.room1_door",
            "binary_sensor.room2_presence",
            "sensor.room2_temperature"
        ]
        
        assert sorted(entity_ids) == sorted(expected_ids)
    
    def test_get_all_entity_ids_with_duplicates(self):
        """Test that get_all_entity_ids removes duplicates."""
        rooms = {
            "room1": RoomConfig(
                room_id="room1",
                name="Room 1",
                sensors={"presence": {"main": "binary_sensor.shared_sensor"}}
            ),
            "room2": RoomConfig(
                room_id="room2",
                name="Room 2",
                sensors={"motion": "binary_sensor.shared_sensor"}  # Same sensor in different room
            )
        }
        
        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            rooms=rooms
        )
        
        entity_ids = system_config.get_all_entity_ids()
        assert entity_ids == ["binary_sensor.shared_sensor"]  # No duplicates
    
    def test_get_room_by_entity_id(self):
        """Test finding room by entity ID."""
        room1 = RoomConfig(
            room_id="room1",
            name="Room 1",
            sensors={"presence": {"main": "binary_sensor.room1_presence"}}
        )
        room2 = RoomConfig(
            room_id="room2",
            name="Room 2",
            sensors={"door": "binary_sensor.room2_door"}
        )
        
        system_config = SystemConfig(
            home_assistant=HomeAssistantConfig(url="test", token="test"),
            database=DatabaseConfig(connection_string="test"),
            mqtt=MQTTConfig(broker="test"),
            prediction=PredictionConfig(),
            features=FeaturesConfig(),
            logging=LoggingConfig(),
            rooms={"room1": room1, "room2": room2}
        )
        
        # Test finding existing entities
        found_room1 = system_config.get_room_by_entity_id("binary_sensor.room1_presence")
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
        
        # Test values from test config
        assert config.home_assistant.url == "http://test-ha:8123"
        assert config.home_assistant.token == "test_token_12345"
        assert config.database.connection_string == "sqlite+aiosqlite:///:memory:"
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
                'home_assistant': {'url': 'http://test:8123', 'token': 'test'},
                'database': {'connection_string': 'sqlite:///:memory:'},
                'mqtt': {'broker': 'test'},
                'prediction': {},
                'features': {},
                'logging': {}
            }
            
            # Create rooms config with nested structure
            rooms_config = {
                'rooms': {
                    'hallways': {
                        'ground_floor': {
                            'name': 'Ground Floor Hallway',
                            'sensors': {'presence': 'binary_sensor.ground_hallway'}
                        },
                        'upper': {
                            'name': 'Upper Hallway',
                            'sensors': {'presence': 'binary_sensor.upper_hallway'}
                        }
                    },
                    'living_room': {
                        'name': 'Living Room',
                        'sensors': {'presence': 'binary_sensor.living_room'}
                    }
                }
            }
            
            # Write config files
            with open(config_dir / 'config.yaml', 'w') as f:
                yaml.dump(main_config, f)
            with open(config_dir / 'rooms.yaml', 'w') as f:
                yaml.dump(rooms_config, f)
            
            loader = ConfigLoader(str(config_dir))
            config = loader.load_config()
            
            # Check nested rooms were flattened correctly
            assert 'hallways_ground_floor' in config.rooms
            assert 'hallways_upper' in config.rooms
            assert 'living_room' in config.rooms
            
            assert config.rooms['hallways_ground_floor'].name == 'Ground Floor Hallway'
            assert config.rooms['hallways_upper'].name == 'Upper Hallway'
            assert config.rooms['living_room'].name == 'Living Room'
    
    @patch('builtins.open', mock_open(read_data="invalid: yaml: content: ["))
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
        
        with patch('src.core.config.ConfigLoader') as mock_loader:
            mock_config = SystemConfig(
                home_assistant=HomeAssistantConfig(url="test", token="test"),
                database=DatabaseConfig(connection_string="test"),
                mqtt=MQTTConfig(broker="test"),
                prediction=PredictionConfig(),
                features=FeaturesConfig(),
                logging=LoggingConfig()
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
        
        with patch('src.core.config.ConfigLoader') as mock_loader:
            mock_config1 = SystemConfig(
                home_assistant=HomeAssistantConfig(url="test1", token="test1"),
                database=DatabaseConfig(connection_string="test1"),
                mqtt=MQTTConfig(broker="test1"),
                prediction=PredictionConfig(),
                features=FeaturesConfig(),
                logging=LoggingConfig()
            )
            mock_config2 = SystemConfig(
                home_assistant=HomeAssistantConfig(url="test2", token="test2"),
                database=DatabaseConfig(connection_string="test2"),
                mqtt=MQTTConfig(broker="test2"),
                prediction=PredictionConfig(),
                features=FeaturesConfig(),
                logging=LoggingConfig()
            )
            
            mock_loader.return_value.load_config.side_effect = [mock_config1, mock_config2]
            
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
                'home_assistant': {'url': 'http://test:8123'},  # Missing token
                'database': {},  # Missing connection_string
                'mqtt': {},  # Missing broker
                'prediction': {},
                'features': {},
                'logging': {}
            }
            
            rooms_config = {'rooms': {}}
            
            with open(config_dir / 'config.yaml', 'w') as f:
                yaml.dump(incomplete_config, f)
            with open(config_dir / 'rooms.yaml', 'w') as f:
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
                'home_assistant': {'url': 'http://test:8123', 'token': 'test'},
                'database': {'connection_string': 'sqlite:///:memory:'},
                'mqtt': {'broker': 'test'},
                'prediction': {},
                'features': {},
                'logging': {}
            }
            
            # Room without explicit name should get auto-generated name
            rooms_config = {
                'rooms': {
                    'test_room': {
                        'sensors': {'presence': 'binary_sensor.test'}
                    }
                }
            }
            
            with open(config_dir / 'config.yaml', 'w') as f:
                yaml.dump(main_config, f)
            with open(config_dir / 'rooms.yaml', 'w') as f:
                yaml.dump(rooms_config, f)
            
            loader = ConfigLoader(str(config_dir))
            config = loader.load_config()
            
            # Should auto-generate name from room_id
            assert config.rooms['test_room'].name == 'Test Room'


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
        
        # Check database config
        assert "sqlite" in config.database.connection_string
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
        assert all(entity_id.startswith(('binary_sensor.', 'sensor.')) for entity_id in entity_ids)
        
        # Check room lookup works
        if entity_ids:
            found_room = config.get_room_by_entity_id(entity_ids[0])
            assert found_room is not None
            assert isinstance(found_room, RoomConfig)