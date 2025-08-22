"""
Comprehensive unit tests for config_validator.py.
Tests configuration validation, connection testing, and validation framework.
"""

import asyncio
import os
from pathlib import Path
import tempfile
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
import yaml

from src.core.config_validator import (
    ConfigurationValidator,
    DatabaseConfigValidator,
    HomeAssistantConfigValidator,
    MQTTConfigValidator,
    RoomsConfigValidator,
    SystemRequirementsValidator,
    ValidationResult,
)


class TestValidationResult:
    """Test ValidationResult class functionality."""

    def test_validation_result_creation(self):
        """Test creating ValidationResult with default values."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.info == []

    def test_add_error_invalidates_result(self):
        """Test that adding error makes result invalid."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        result.add_error("Test error message")

        assert result.is_valid is False
        assert result.errors == ["Test error message"]

    def test_add_warning_preserves_validity(self):
        """Test that adding warning preserves validity."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        result.add_warning("Test warning message")

        assert result.is_valid is True
        assert result.warnings == ["Test warning message"]

    def test_add_info(self):
        """Test adding info message."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        result.add_info("Test info message")

        assert result.is_valid is True
        assert result.info == ["Test info message"]

    def test_merge_valid_results(self):
        """Test merging two valid results."""
        result1 = ValidationResult(
            is_valid=True, errors=[], warnings=["warn1"], info=["info1"]
        )
        result2 = ValidationResult(
            is_valid=True, errors=[], warnings=["warn2"], info=["info2"]
        )

        result1.merge(result2)

        assert result1.is_valid is True
        assert result1.warnings == ["warn1", "warn2"]
        assert result1.info == ["info1", "info2"]

    def test_merge_invalid_result_invalidates(self):
        """Test merging invalid result invalidates the target."""
        result1 = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        result2 = ValidationResult(
            is_valid=False, errors=["error"], warnings=[], info=[]
        )

        result1.merge(result2)

        assert result1.is_valid is False
        assert result1.errors == ["error"]

    def test_string_representation_valid(self):
        """Test string representation of valid result."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=["All good"]
        )

        string_repr = str(result)

        assert "✅ VALID" in string_repr
        assert "All good" in string_repr

    def test_string_representation_invalid(self):
        """Test string representation of invalid result."""
        result = ValidationResult(
            is_valid=False,
            errors=["Critical error"],
            warnings=["Warning message"],
            info=["Info message"],
        )

        string_repr = str(result)

        assert "❌ INVALID" in string_repr
        assert "Critical error" in string_repr
        assert "Warning message" in string_repr
        assert "Info message" in string_repr


class TestHomeAssistantConfigValidator:
    """Test HomeAssistantConfigValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create HomeAssistantConfigValidator instance."""
        return HomeAssistantConfigValidator()

    def test_validate_valid_config(self, validator):
        """Test validation with valid HA configuration."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "a" * 183,  # Valid token length
                "websocket_timeout": 30,
                "api_timeout": 10,
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert any(
            "Home Assistant URL: http://localhost:8123" in info for info in result.info
        )

    def test_validate_missing_url(self, validator):
        """Test validation with missing URL."""
        config = {"home_assistant": {"token": "valid_token_123"}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Home Assistant URL is required" in result.errors

    def test_validate_invalid_url(self, validator):
        """Test validation with invalid URL format."""
        config = {
            "home_assistant": {
                "url": "not-a-valid-url",
                "token": "valid_token_123",
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any(
            "Invalid Home Assistant URL format" in error for error in result.errors
        )

    def test_validate_missing_token(self, validator):
        """Test validation with missing token."""
        config = {"home_assistant": {"url": "http://localhost:8123"}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Home Assistant token is required" in result.errors

    def test_validate_short_token_warning(self, validator):
        """Test validation with suspiciously short token."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "short_token",  # Too short
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True  # Warning, not error
        assert any(
            "token appears to be too short" in warning for warning in result.warnings
        )

    def test_validate_timeout_warnings(self, validator):
        """Test validation with problematic timeout values."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "a" * 183,
                "websocket_timeout": 5,  # Too low
                "api_timeout": 120,  # Too high
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True  # Warnings, not errors
        assert any(
            "WebSocket timeout is very low" in warning for warning in result.warnings
        )
        assert any("API timeout is very high" in warning for warning in result.warnings)

    @patch("requests.get")
    def test_connection_test_success(self, mock_get, validator):
        """Test successful connection test."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "valid_token",
                "api_timeout": 10,
            }
        }

        # Mock successful response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "2024.1.0"}
        mock_get.return_value = mock_response

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any(
            "✅ Home Assistant API connection successful" in info
            for info in result.info
        )
        assert any("Home Assistant version: 2024.1.0" in info for info in result.info)

        # Verify correct headers were sent
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        assert call_args[1]["headers"]["Authorization"] == "Bearer valid_token"

    @patch("requests.get")
    def test_connection_test_auth_failure(self, mock_get, validator):
        """Test connection test with authentication failure."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "invalid_token",
                "api_timeout": 10,
            }
        }

        # Mock 401 response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Authentication failed: Invalid token" in error for error in result.errors
        )

    @patch("requests.get")
    def test_connection_test_timeout(self, mock_get, validator):
        """Test connection test with timeout."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "valid_token",
                "api_timeout": 5,
            }
        }

        # Mock timeout
        import requests

        mock_get.side_effect = requests.exceptions.Timeout()

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Connection timeout after 5 seconds" in error for error in result.errors
        )

    @patch("requests.get")
    def test_connection_test_connection_error(self, mock_get, validator):
        """Test connection test with connection error."""
        config = {
            "home_assistant": {
                "url": "http://unreachable:8123",
                "token": "valid_token",
            }
        }

        # Mock connection error
        import requests

        mock_get.side_effect = requests.exceptions.ConnectionError()

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("Cannot reach Home Assistant" in error for error in result.errors)

    def test_connection_test_missing_config(self, validator):
        """Test connection test with missing configuration."""
        config = {
            "home_assistant": {
                "url": "http://localhost:8123"
                # Missing token
            }
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Cannot test connection: URL or token missing" in error
            for error in result.errors
        )


class TestDatabaseConfigValidator:
    """Test DatabaseConfigValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create DatabaseConfigValidator instance."""
        return DatabaseConfigValidator()

    def test_validate_valid_config(self, validator):
        """Test validation with valid database configuration."""
        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/occupancy_db",
                "pool_size": 10,
                "max_overflow": 20,
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert any(
            "Database connection string configured" in info for info in result.info
        )

    def test_validate_missing_connection_string(self, validator):
        """Test validation with missing connection string."""
        config = {"database": {"pool_size": 10}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Database connection string is required" in result.errors

    def test_validate_non_postgresql_database(self, validator):
        """Test validation with non-PostgreSQL database."""
        config = {
            "database": {"connection_string": "mysql://user:pass@localhost:3306/db"}
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Only PostgreSQL databases are supported" in result.errors

    def test_validate_missing_timescaledb_warning(self, validator):
        """Test validation warning for missing TimescaleDB."""
        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/regular_db"
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "TimescaleDB extension is recommended" in warning
            for warning in result.warnings
        )

    def test_validate_pool_size_warnings(self, validator):
        """Test validation with problematic pool sizes."""
        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/db",
                "pool_size": 1,  # Too low
                "max_overflow": 1,  # Too low relative to pool size
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Database pool size is very low" in warning for warning in result.warnings
        )
        assert any(
            "Max overflow is low relative to pool size" in warning
            for warning in result.warnings
        )

    @patch("asyncpg.connect")
    @patch("asyncio.run")
    def test_connection_test_success(self, mock_run, mock_connect, validator):
        """Test successful database connection test."""
        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/test_db"
            }
        }

        # Mock successful connection
        mock_conn = AsyncMock()
        mock_conn.fetchval = AsyncMock()
        mock_conn.fetchval.side_effect = [
            "PostgreSQL 14.0 on x86_64-pc-linux-gnu",  # version query
            "3.0.0",  # TimescaleDB version
        ]
        mock_conn.close = AsyncMock()

        async def mock_async_test():
            return mock_conn

        mock_connect.return_value = mock_async_test()

        # Mock asyncio.run to execute our test function
        def run_side_effect(coro):
            return asyncio.get_event_loop().run_until_complete(coro)

        mock_run.side_effect = run_side_effect

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any("✅ Database connection successful" in info for info in result.info)

    @patch("asyncio.run")
    def test_connection_test_auth_failure(self, mock_run, validator):
        """Test database connection test with authentication failure."""
        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:wrong@localhost:5432/test_db"
            }
        }

        # Mock authentication error
        import asyncpg

        mock_run.side_effect = (
            asyncpg.exceptions.InvalidAuthorizationSpecificationError("auth failed")
        )

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Authentication failed: Invalid username/password" in error
            for error in result.errors
        )

    @patch("asyncio.run")
    def test_connection_test_database_not_found(self, mock_run, validator):
        """Test database connection test with non-existent database."""
        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/nonexistent_db"
            }
        }

        # Mock database not found error
        import asyncpg

        mock_run.side_effect = asyncpg.exceptions.InvalidCatalogNameError(
            "database does not exist"
        )

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("Database does not exist" in error for error in result.errors)

    def test_connection_test_missing_config(self, validator):
        """Test connection test with missing configuration."""
        config = {"database": {}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Cannot test connection: connection string missing" in error
            for error in result.errors
        )


class TestMQTTConfigValidator:
    """Test MQTTConfigValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create MQTTConfigValidator instance."""
        return MQTTConfigValidator()

    def test_validate_valid_config(self, validator):
        """Test validation with valid MQTT configuration."""
        config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": "mqtt_user",
                "password": "mqtt_pass",
                "topic_prefix": "occupancy/predictions",
                "prediction_qos": 1,
                "system_qos": 0,
                "discovery_enabled": True,
                "discovery_prefix": "homeassistant",
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert any("MQTT broker: localhost" in info for info in result.info)

    def test_validate_missing_broker(self, validator):
        """Test validation with missing broker."""
        config = {"mqtt": {"port": 1883}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "MQTT broker is required" in result.errors

    def test_validate_invalid_port(self, validator):
        """Test validation with invalid port."""
        config = {"mqtt": {"broker": "localhost", "port": 99999}}  # Invalid port

        result = validator.validate(config)

        assert result.is_valid is False
        assert any("Invalid MQTT port" in error for error in result.errors)

    def test_validate_non_standard_port_warning(self, validator):
        """Test validation with non-standard port."""
        config = {"mqtt": {"broker": "localhost", "port": 9000}}  # Non-standard

        result = validator.validate(config)

        assert result.is_valid is True
        assert any("Non-standard MQTT port" in warning for warning in result.warnings)

    def test_validate_topic_prefix_warnings(self, validator):
        """Test validation with problematic topic prefix."""
        config = {
            "mqtt": {
                "broker": "localhost",
                "topic_prefix": "/starts/with/slash/",  # Bad format
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "should not start or end with '/'" in warning for warning in result.warnings
        )

    def test_validate_invalid_qos_levels(self, validator):
        """Test validation with invalid QoS levels."""
        config = {
            "mqtt": {
                "broker": "localhost",
                "prediction_qos": 5,  # Invalid
                "system_qos": -1,  # Invalid
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any("Invalid prediction QoS level" in error for error in result.errors)
        assert any("Invalid system QoS level" in error for error in result.errors)

    def test_validate_discovery_missing_prefix(self, validator):
        """Test validation with discovery enabled but missing prefix."""
        config = {
            "mqtt": {
                "broker": "localhost",
                "discovery_enabled": True,
                "discovery_prefix": "",  # Empty prefix
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any(
            "Discovery prefix is required when discovery is enabled" in error
            for error in result.errors
        )

    @patch("paho.mqtt.client.Client")
    def test_connection_test_success(self, mock_client_class, validator):
        """Test successful MQTT connection test."""
        config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": "user",
                "password": "pass",
            }
        }

        # Mock MQTT client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful connection callback
        def mock_connect_async(broker, port, keepalive):
            # Simulate calling on_connect callback
            mock_client.on_connect(mock_client, None, None, 0)  # rc=0 means success

        mock_client.connect_async = mock_connect_async

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any(
            "✅ MQTT broker connection successful" in info for info in result.info
        )

        # Verify client was configured correctly
        mock_client.username_pw_set.assert_called_once_with("user", "pass")

    @patch("paho.mqtt.client.Client")
    def test_connection_test_auth_failure(self, mock_client_class, validator):
        """Test MQTT connection test with authentication failure."""
        config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
            }
        }

        # Mock MQTT client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock failed connection callback
        def mock_connect_async(broker, port, keepalive):
            # Simulate calling on_connect callback with auth error
            mock_client.on_connect(mock_client, None, None, 4)  # rc=4 means auth error

        mock_client.connect_async = mock_connect_async

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("bad username or password" in error for error in result.errors)

    @patch("paho.mqtt.client.Client")
    @patch("time.sleep")
    def test_connection_test_timeout(self, mock_sleep, mock_client_class, validator):
        """Test MQTT connection test with timeout."""
        config = {
            "mqtt": {
                "broker": "unreachable",
                "port": 1883,
            }
        }

        # Mock MQTT client that never calls callback
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        mock_client.connect_async = Mock()  # Never calls on_connect

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("MQTT connection timeout" in error for error in result.errors)

    def test_connection_test_missing_broker(self, validator):
        """Test connection test with missing broker."""
        config = {"mqtt": {}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Cannot test connection: broker missing" in error for error in result.errors
        )


class TestRoomsConfigValidator:
    """Test RoomsConfigValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create RoomsConfigValidator instance."""
        return RoomsConfigValidator()

    def test_validate_valid_config(self, validator):
        """Test validation with valid rooms configuration."""
        rooms_config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": {
                            "main": "binary_sensor.living_room_motion",
                            "secondary": "binary_sensor.living_room_motion_2",
                        },
                        "occupancy": "binary_sensor.living_room_occupancy",
                        "door": "binary_sensor.living_room_door",
                    },
                },
                "bedroom": {
                    "name": "Bedroom",
                    "sensors": {
                        "motion": "binary_sensor.bedroom_motion",
                        "occupancy": "binary_sensor.bedroom_occupancy",
                    },
                },
            }
        }

        result = validator.validate(rooms_config)

        assert result.is_valid is True
        assert len(result.errors) == 0
        assert any("Rooms configured: 2" in info for info in result.info)
        assert any("Total sensors:" in info for info in result.info)

    def test_validate_no_rooms(self, validator):
        """Test validation with no rooms configured."""
        rooms_config = {"rooms": {}}

        result = validator.validate(rooms_config)

        assert result.is_valid is False
        assert "No rooms configured" in result.errors

    def test_validate_missing_rooms_key(self, validator):
        """Test validation with missing rooms key."""
        rooms_config = {}

        result = validator.validate(rooms_config)

        assert result.is_valid is False
        assert "No rooms configured" in result.errors

    def test_validate_room_missing_sensors(self, validator):
        """Test validation with room missing sensors."""
        rooms_config = {
            "rooms": {
                "empty_room": {
                    "name": "Empty Room"
                    # No sensors
                }
            }
        }

        result = validator.validate(rooms_config)

        assert result.is_valid is True  # Warning, not error
        assert any(
            "empty_room has no sensors configured" in warning
            for warning in result.warnings
        )

    def test_validate_invalid_entity_ids(self, validator):
        """Test validation with invalid entity IDs."""
        rooms_config = {
            "rooms": {
                "test_room": {
                    "name": "Test Room",
                    "sensors": {
                        "motion": "invalid_entity_id",  # Missing domain.entity format
                        "occupancy": "sensor.with.invalid.domain",  # Invalid domain
                    },
                }
            }
        }

        result = validator.validate(rooms_config)

        assert result.is_valid is False
        assert any(
            "Invalid entity ID format: invalid_entity_id" in error
            for error in result.errors
        )
        assert any(
            "Invalid entity ID format: sensor.with.invalid.domain" in error
            for error in result.errors
        )

    def test_validate_missing_essential_sensors(self, validator):
        """Test validation with missing essential sensor types."""
        rooms_config = {
            "rooms": {
                "minimal_room": {
                    "name": "Minimal Room",
                    "sensors": {
                        "temperature": "sensor.temp"  # Missing motion, occupancy, door
                    },
                }
            }
        }

        result = validator.validate(rooms_config)

        assert result.is_valid is True  # Warning, not error
        assert any(
            "missing essential sensor types" in warning for warning in result.warnings
        )

    def test_validate_complex_sensor_structure(self, validator):
        """Test validation with complex nested sensor structure."""
        rooms_config = {
            "rooms": {
                "complex_room": {
                    "name": "Complex Room",
                    "sensors": {
                        "motion": {
                            "main": "binary_sensor.motion_main",
                            "corner": "binary_sensor.motion_corner",
                            "entrance": "binary_sensor.motion_entrance",
                        },
                        "door": {
                            "main_door": "binary_sensor.main_door",
                            "side_door": "binary_sensor.side_door",
                        },
                        "occupancy": "binary_sensor.room_occupancy",
                    },
                }
            }
        }

        result = validator.validate(rooms_config)

        assert result.is_valid is True
        assert len(result.errors) == 0

    def test_validate_entity_id_formats(self, validator):
        """Test validation of various entity ID formats."""
        rooms_config = {
            "rooms": {
                "test_room": {
                    "sensors": {
                        "valid_binary": "binary_sensor.test_sensor",
                        "valid_sensor": "sensor.temperature",
                        "valid_light": "light.room_light",
                        "valid_switch": "switch.fan",
                        "valid_climate": "climate.hvac",
                        "valid_complex": "binary_sensor.motion_sensor_with_underscores_123",
                    }
                }
            }
        }

        result = validator.validate(rooms_config)

        assert result.is_valid is True
        assert len(result.errors) == 0


class TestSystemRequirementsValidator:
    """Test SystemRequirementsValidator functionality."""

    @pytest.fixture
    def validator(self):
        """Create SystemRequirementsValidator instance."""
        return SystemRequirementsValidator()

    def test_validate_sufficient_system(self, validator):
        """Test validation with sufficient system requirements."""
        config = {}

        with patch("sys.version_info", (3, 11, 0)), patch(
            "shutil.disk_usage"
        ) as mock_disk, patch("psutil.virtual_memory") as mock_memory:

            mock_disk.return_value = (
                100 * 1024**3,
                50 * 1024**3,
                50 * 1024**3,
            )  # 100GB total, 50GB free
            mock_memory.return_value.total = 8 * 1024**3  # 8GB total
            mock_memory.return_value.available = 4 * 1024**3  # 4GB available

            result = validator.validate(config)

            assert result.is_valid is True
            assert any(
                "✅ All required packages are available" in info for info in result.info
            )
            assert any("Available disk space: 50.0 GB" in info for info in result.info)

    def test_validate_old_python_version(self, validator):
        """Test validation with old Python version."""
        config = {}

        with patch("sys.version_info", (3, 8, 0)):  # Too old
            result = validator.validate(config)

            assert result.is_valid is False
            assert any(
                "Python 3.9+ required, found 3.8.0" in error for error in result.errors
            )

    def test_validate_old_python_version_warning(self, validator):
        """Test validation with old but acceptable Python version."""
        config = {}

        with patch("sys.version_info", (3, 9, 0)):  # Acceptable but old
            result = validator.validate(config)

            assert result.is_valid is True
            assert any(
                "Python 3.11+ recommended, found 3.9.0" in warning
                for warning in result.warnings
            )

    def test_validate_insufficient_disk_space(self, validator):
        """Test validation with insufficient disk space."""
        config = {}

        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = (
                10 * 1024**3,
                9.5 * 1024**3,
                0.5 * 1024**3,
            )  # Only 0.5GB free

            result = validator.validate(config)

            assert result.is_valid is False
            assert any("Insufficient disk space" in error for error in result.errors)

    def test_validate_low_disk_space_warning(self, validator):
        """Test validation with low disk space warning."""
        config = {}

        with patch("shutil.disk_usage") as mock_disk:
            mock_disk.return_value = (
                100 * 1024**3,
                97 * 1024**3,
                3 * 1024**3,
            )  # Only 3GB free

            result = validator.validate(config)

            assert result.is_valid is True
            assert any("Low disk space" in warning for warning in result.warnings)

    def test_validate_insufficient_memory(self, validator):
        """Test validation with insufficient memory."""
        config = {}

        with patch("psutil.virtual_memory") as mock_memory:
            mock_memory.return_value.total = 1 * 1024**3  # Only 1GB total
            mock_memory.return_value.available = 0.5 * 1024**3

            result = validator.validate(config)

            assert result.is_valid is False
            assert any("Insufficient memory" in error for error in result.errors)

    def test_validate_missing_packages(self, validator):
        """Test validation with missing required packages."""
        config = {}

        # Mock missing package
        with patch("builtins.__import__") as mock_import:

            def import_side_effect(name, *args, **kwargs):
                if name == "asyncpg":
                    raise ImportError("Module not found")
                return Mock()

            mock_import.side_effect = import_side_effect

            result = validator.validate(config)

            assert result.is_valid is False
            assert any("Missing required packages" in error for error in result.errors)


class TestConfigurationValidator:
    """Test main ConfigurationValidator orchestration."""

    @pytest.fixture
    def validator(self):
        """Create ConfigurationValidator instance."""
        return ConfigurationValidator()

    @pytest.fixture
    def valid_config(self):
        """Valid configuration for testing."""
        return {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "a" * 183,
            },
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/test_db"
            },
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
            },
        }

    @pytest.fixture
    def valid_rooms_config(self):
        """Valid rooms configuration for testing."""
        return {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": "binary_sensor.living_room_motion",
                        "occupancy": "binary_sensor.living_room_occupancy",
                        "door": "binary_sensor.living_room_door",
                    },
                }
            }
        }

    def test_validate_configuration_all_valid(
        self, validator, valid_config, valid_rooms_config
    ):
        """Test validation with all valid configurations."""
        result = validator.validate_configuration(valid_config, valid_rooms_config)

        assert result.is_valid is True
        assert any(
            "Configuration validation completed successfully" in info
            for info in result.info
        )

    def test_validate_configuration_with_errors(self, validator, valid_rooms_config):
        """Test validation with configuration errors."""
        invalid_config = {
            "home_assistant": {
                # Missing URL and token
            },
            "database": {
                # Missing connection string
            },
            "mqtt": {
                # Missing broker
            },
        }

        result = validator.validate_configuration(invalid_config, valid_rooms_config)

        assert result.is_valid is False
        assert any("Configuration validation failed" in info for info in result.info)
        assert len(result.errors) > 0

    @patch.object(HomeAssistantConfigValidator, "test_connection")
    @patch.object(DatabaseConfigValidator, "test_connection")
    @patch.object(MQTTConfigValidator, "test_connection")
    def test_validate_configuration_with_connection_tests(
        self,
        mock_mqtt_test,
        mock_db_test,
        mock_ha_test,
        validator,
        valid_config,
        valid_rooms_config,
    ):
        """Test validation with connection tests enabled."""
        # Mock successful connection tests
        success_result = ValidationResult(
            is_valid=True, errors=[], warnings=[], info=["Connection successful"]
        )
        mock_ha_test.return_value = success_result
        mock_db_test.return_value = success_result
        mock_mqtt_test.return_value = success_result

        result = validator.validate_configuration(
            valid_config, valid_rooms_config, test_connections=True
        )

        assert result.is_valid is True
        assert any("Testing external connections" in info for info in result.info)

        # Verify connection tests were called
        mock_ha_test.assert_called_once()
        mock_db_test.assert_called_once()
        mock_mqtt_test.assert_called_once()

    def test_validate_configuration_strict_mode(
        self, validator, valid_config, valid_rooms_config
    ):
        """Test validation in strict mode where warnings become errors."""
        # Create config that would generate warnings
        config_with_warnings = valid_config.copy()
        config_with_warnings["home_assistant"][
            "token"
        ] = "short_token"  # Too short, generates warning

        result = validator.validate_configuration(
            config_with_warnings, valid_rooms_config, strict_mode=True
        )

        assert result.is_valid is False
        assert any("Strict mode:" in error for error in result.errors)

    def test_validate_config_files_success(self, validator):
        """Test validation of actual config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create valid config files
            config_content = {
                "home_assistant": {
                    "url": "http://localhost:8123",
                    "token": "a" * 183,
                },
                "database": {
                    "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/test_db"
                },
                "mqtt": {"broker": "localhost"},
            }

            rooms_content = {
                "rooms": {
                    "test_room": {
                        "name": "Test Room",
                        "sensors": {"motion": "binary_sensor.test_motion"},
                    }
                }
            }

            (config_dir / "config.yaml").write_text(yaml.dump(config_content))
            (config_dir / "rooms.yaml").write_text(yaml.dump(rooms_content))

            result = validator.validate_config_files(str(config_dir))

            assert result.is_valid is True
            assert any("Loaded configuration from:" in info for info in result.info)
            assert any(
                "Loaded rooms configuration from:" in info for info in result.info
            )

    def test_validate_config_files_missing_config(self, validator):
        """Test validation with missing config file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            result = validator.validate_config_files(tmpdir)

            assert result.is_valid is False
            assert any(
                "Configuration file not found:" in error for error in result.errors
            )

    def test_validate_config_files_environment_specific(self, validator):
        """Test validation with environment-specific config."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create environment-specific config
            config_content = {
                "home_assistant": {
                    "url": "http://staging:8123",
                    "token": "staging_token_" + "a" * 170,
                },
                "database": {
                    "connection_string": "postgresql+asyncpg://user:pass@staging:5432/staging_db"
                },
                "mqtt": {"broker": "staging-mqtt"},
            }

            rooms_content = {
                "rooms": {
                    "test_room": {
                        "name": "Test Room",
                        "sensors": {"motion": "binary_sensor.test_motion"},
                    }
                }
            }

            (config_dir / "config.staging.yaml").write_text(yaml.dump(config_content))
            (config_dir / "rooms.yaml").write_text(yaml.dump(rooms_content))

            result = validator.validate_config_files(
                str(config_dir), environment="staging"
            )

            assert result.is_valid is True
            assert any("config.staging.yaml" in info for info in result.info)

    def test_validate_config_files_invalid_yaml(self, validator):
        """Test validation with invalid YAML syntax."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir)

            # Create invalid YAML
            (config_dir / "config.yaml").write_text("invalid: yaml: content: [unclosed")
            (config_dir / "rooms.yaml").write_text("rooms: {}")

            result = validator.validate_config_files(str(config_dir))

            assert result.is_valid is False
            assert any(
                "Failed to load configuration file:" in error for error in result.errors
            )


if __name__ == "__main__":
    pytest.main([__file__])
