"""
Configuration validation framework.
Validates configuration files, environment settings, and system requirements.
"""

import asyncio
from dataclasses import dataclass
import logging
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import asyncpg
import paho.mqtt.client as mqtt
import psutil
import requests
import yaml

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of a configuration validation."""

    is_valid: bool
    errors: List[str]
    warnings: List[str]
    info: List[str]

    def add_error(self, message: str) -> None:
        """Add an error message."""
        self.errors.append(message)
        self.is_valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def add_info(self, message: str) -> None:
        """Add an info message."""
        self.info.append(message)

    def merge(self, other: "ValidationResult") -> None:
        """Merge another validation result into this one."""
        if not other.is_valid:
            self.is_valid = False
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)

    def __str__(self) -> str:
        """String representation of validation result."""
        status = "✅ VALID" if self.is_valid else "❌ INVALID"
        result = f"Configuration Validation: {status}\n"

        if self.errors:
            result += f"\nErrors ({len(self.errors)}):\n"
            for error in self.errors:
                result += f"  ❌ {error}\n"

        if self.warnings:
            result += f"\nWarnings ({len(self.warnings)}):\n"
            for warning in self.warnings:
                result += f"  ⚠️  {warning}\n"

        if self.info:
            result += f"\nInfo ({len(self.info)}):\n"
            for info in self.info:
                result += f"  ℹ️  {info}\n"

        return result


class HomeAssistantConfigValidator:
    """Validates Home Assistant configuration."""

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate Home Assistant configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        ha_config = config.get("home_assistant", {})

        # Check required fields
        if not ha_config.get("url"):
            result.add_error("Home Assistant URL is required")
        else:
            # Validate URL format
            url = ha_config["url"]
            if not self._is_valid_url(url):
                result.add_error(f"Invalid Home Assistant URL format: {url}")
            else:
                result.add_info(f"Home Assistant URL: {url}")

        if not ha_config.get("token"):
            result.add_error("Home Assistant token is required")
        else:
            token = ha_config["token"]
            if len(token) < 180:  # HA tokens are typically ~183 characters
                result.add_warning("Home Assistant token appears to be too short")
            result.add_info("Home Assistant token is configured")

        # Check timeout values
        websocket_timeout = ha_config.get("websocket_timeout", 30)
        api_timeout = ha_config.get("api_timeout", 10)

        if websocket_timeout < 10:
            result.add_warning(
                "WebSocket timeout is very low, may cause connection issues"
            )
        elif websocket_timeout > 300:
            result.add_warning(
                "WebSocket timeout is very high, may cause slow failure detection"
            )

        if api_timeout < 5:
            result.add_warning("API timeout is very low, may cause request failures")
        elif api_timeout > 60:
            result.add_warning("API timeout is very high, may cause slow responses")

        return result

    def _is_valid_url(self, url: str) -> bool:
        """Check if URL is valid format."""
        try:
            parsed = urlparse(url)
            return bool(parsed.scheme and parsed.netloc)
        except Exception:
            return False

    def test_connection(self, config: Dict[str, Any]) -> ValidationResult:
        """Test actual connection to Home Assistant."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        ha_config = config.get("home_assistant", {})
        url = ha_config.get("url")
        token = ha_config.get("token")
        timeout = ha_config.get("api_timeout", 10)

        if not url or not token:
            result.add_error("Cannot test connection: URL or token missing")
            return result

        try:
            # Test API endpoint
            headers = {
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            }

            response = requests.get(
                f"{url}/api/",
                headers=headers,
                timeout=timeout,
                verify=False,  # Some HA instances use self-signed certificates
            )

            if response.status_code == 200:
                result.add_info("✅ Home Assistant API connection successful")
                api_data = response.json()
                if "version" in api_data:
                    result.add_info(f"Home Assistant version: {api_data['version']}")
            elif response.status_code == 401:
                result.add_error("❌ Authentication failed: Invalid token")
            else:
                result.add_error(
                    f"❌ API connection failed: HTTP {response.status_code}"
                )

        except requests.exceptions.Timeout:
            result.add_error(f"❌ Connection timeout after {timeout} seconds")
        except requests.exceptions.ConnectionError:
            result.add_error("❌ Connection failed: Cannot reach Home Assistant")
        except Exception as e:
            result.add_error(f"❌ Unexpected error testing connection: {e}")

        return result


class DatabaseConfigValidator:
    """Validates database configuration."""

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate database configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        db_config = config.get("database", {})

        # Check connection string
        conn_str = db_config.get("connection_string")
        if not conn_str:
            result.add_error("Database connection string is required")
            return result

        # Parse connection string
        if not conn_str.startswith("postgresql"):
            result.add_error("Only PostgreSQL databases are supported")

        # Validate connection parameters
        if "timescaledb" not in conn_str and "TimescaleDB" not in conn_str:
            result.add_warning(
                "TimescaleDB extension is recommended for time-series data"
            )

        result.add_info("Database connection string configured")

        # Check pool settings
        pool_size = db_config.get("pool_size", 10)
        max_overflow = db_config.get("max_overflow", 20)

        if pool_size < 2:
            result.add_warning("Database pool size is very low")
        elif pool_size > 50:
            result.add_warning(
                "Database pool size is very high, may consume too many connections"
            )

        if max_overflow < pool_size * 0.5:
            result.add_warning("Max overflow is low relative to pool size")

        result.add_info(f"Pool settings: size={pool_size}, max_overflow={max_overflow}")

        return result

    def test_connection(self, config: Dict[str, Any]) -> ValidationResult:
        """Test actual database connection."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        db_config = config.get("database", {})
        conn_str = db_config.get("connection_string")

        if not conn_str:
            result.add_error("Cannot test connection: connection string missing")
            return result

        try:

            async def test_db():
                try:
                    # Convert SQLAlchemy URL to asyncpg format
                    if conn_str.startswith("postgresql+asyncpg://"):
                        asyncpg_url = conn_str.replace(
                            "postgresql+asyncpg://", "postgresql://"
                        )
                    else:
                        asyncpg_url = conn_str

                    conn = await asyncpg.connect(asyncpg_url, timeout=10)

                    # Test basic query
                    version = await conn.fetchval("SELECT version();")
                    result.add_info("✅ Database connection successful")
                    result.add_info(f"Database version: {version.split(',')[0]}")

                    # Check for TimescaleDB
                    try:
                        ts_version = await conn.fetchval(
                            "SELECT extversion FROM pg_extension WHERE extname='timescaledb';"
                        )
                        if ts_version:
                            result.add_info(
                                f"✅ TimescaleDB extension found: {ts_version}"
                            )
                        else:
                            result.add_warning("⚠️  TimescaleDB extension not found")
                    except Exception:
                        result.add_warning("⚠️  Cannot check TimescaleDB extension")

                    await conn.close()

                except asyncpg.exceptions.InvalidAuthorizationSpecificationError:
                    result.add_error(
                        "❌ Authentication failed: Invalid username/password"
                    )
                except asyncpg.exceptions.InvalidCatalogNameError:
                    result.add_error("❌ Database does not exist")
                except asyncio.TimeoutError:
                    result.add_error("❌ Connection timeout")
                except Exception as e:
                    result.add_error(f"❌ Connection failed: {e}")

            asyncio.run(test_db())

        except ImportError:
            result.add_warning(
                "⚠️  Cannot test database connection: asyncpg not available"
            )
        except Exception as e:
            result.add_error(f"❌ Unexpected error testing database: {e}")

        return result


class MQTTConfigValidator:
    """Validates MQTT configuration."""

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate MQTT configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        mqtt_config = config.get("mqtt", {})

        # Check broker
        broker = mqtt_config.get("broker")
        if not broker:
            result.add_error("MQTT broker is required")
        else:
            result.add_info(f"MQTT broker: {broker}")

        # Check port
        port = mqtt_config.get("port", 1883)
        if port < 1 or port > 65535:
            result.add_error(f"Invalid MQTT port: {port}")
        elif port != 1883 and port != 8883:
            result.add_warning(f"Non-standard MQTT port: {port}")

        # Check topic prefix
        topic_prefix = mqtt_config.get("topic_prefix", "")
        if not topic_prefix:
            result.add_warning(
                "Topic prefix is empty, messages will be published to root topic"
            )
        elif topic_prefix.startswith("/") or topic_prefix.endswith("/"):
            result.add_warning("Topic prefix should not start or end with '/'")

        # Check QoS levels
        pred_qos = mqtt_config.get("prediction_qos", 1)
        sys_qos = mqtt_config.get("system_qos", 0)

        if pred_qos not in [0, 1, 2]:
            result.add_error(f"Invalid prediction QoS level: {pred_qos}")
        if sys_qos not in [0, 1, 2]:
            result.add_error(f"Invalid system QoS level: {sys_qos}")

        # Check discovery settings
        if mqtt_config.get("discovery_enabled", True):
            discovery_prefix = mqtt_config.get("discovery_prefix", "homeassistant")
            if not discovery_prefix:
                result.add_error(
                    "Discovery prefix is required when discovery is enabled"
                )
            result.add_info(f"MQTT discovery enabled with prefix: {discovery_prefix}")

        return result

    def test_connection(self, config: Dict[str, Any]) -> ValidationResult:
        """Test MQTT broker connection."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        mqtt_config = config.get("mqtt", {})
        broker = mqtt_config.get("broker")
        port = mqtt_config.get("port", 1883)
        username = mqtt_config.get("username", "")
        password = mqtt_config.get("password", "")

        if not broker:
            result.add_error("Cannot test connection: broker missing")
            return result

        try:
            connection_result: Dict[str, Any] = {"success": False, "error": None}

            def on_connect(client, userdata, flags, rc):
                if rc == 0:
                    connection_result["success"] = True
                    result.add_info("✅ MQTT broker connection successful")
                else:
                    error_messages = {
                        1: "Connection refused - incorrect protocol version",
                        2: "Connection refused - invalid client identifier",
                        3: "Connection refused - server unavailable",
                        4: "Connection refused - bad username or password",
                        5: "Connection refused - not authorized",
                    }
                    error = error_messages.get(rc, f"Connection refused - code {rc}")
                    connection_result["error"] = error

            client = mqtt.Client()
            client.on_connect = on_connect

            if username:
                client.username_pw_set(username, password)

            client.connect_async(broker, port, 60)
            client.loop_start()

            # Wait for connection result
            for _ in range(50):  # Wait up to 5 seconds
                if connection_result["success"] or connection_result["error"]:
                    break
                time.sleep(0.1)

            client.loop_stop()
            client.disconnect()

            if connection_result["error"]:
                result.add_error(
                    f"❌ MQTT connection failed: {connection_result['error']}"
                )
            elif not connection_result["success"]:
                result.add_error("❌ MQTT connection timeout")

        except ImportError:
            result.add_warning(
                "⚠️  Cannot test MQTT connection: paho-mqtt not available"
            )
        except Exception as e:
            result.add_error(f"❌ Unexpected error testing MQTT: {e}")

        return result


class RoomsConfigValidator:
    """Validates rooms and sensors configuration."""

    def validate(self, rooms_config: Dict[str, Any]) -> ValidationResult:
        """Validate rooms configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        rooms = rooms_config.get("rooms", {})
        if not rooms:
            result.add_error("No rooms configured")
            return result

        total_sensors = 0
        room_count = 0

        for room_id, room_data in rooms.items():
            room_result = self._validate_room(room_id, room_data)
            result.merge(room_result)

            if room_result.is_valid:
                room_count += 1
                # Count sensors in this room
                sensors = room_data.get("sensors", {})
                total_sensors += self._count_sensors(sensors)

        result.add_info(f"Rooms configured: {room_count}")
        result.add_info(f"Total sensors: {total_sensors}")

        if total_sensors < 5:
            result.add_warning(
                "Very few sensors configured, predictions may be less accurate"
            )

        return result

    def _validate_room(
        self, room_id: str, room_data: Dict[str, Any]
    ) -> ValidationResult:
        """Validate individual room configuration."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        # Check room name
        if "name" not in room_data:
            result.add_warning(f"Room {room_id} has no name configured")

        # Check sensors
        sensors = room_data.get("sensors", {})
        if not sensors:
            result.add_warning(f"Room {room_id} has no sensors configured")
            return result

        # Validate sensor configuration
        sensor_types = set()
        entity_ids = set()

        for sensor_type, sensor_data in sensors.items():
            sensor_types.add(sensor_type)

            if isinstance(sensor_data, dict):
                for entity_id in sensor_data.values():
                    if isinstance(entity_id, str):
                        if not self._is_valid_entity_id(entity_id):
                            result.add_error(f"Invalid entity ID format: {entity_id}")
                        else:
                            entity_ids.add(entity_id)
            elif isinstance(sensor_data, str):
                if not self._is_valid_entity_id(sensor_data):
                    result.add_error(f"Invalid entity ID format: {sensor_data}")
                else:
                    entity_ids.add(sensor_data)

        # Check for essential sensor types
        essential_types = ["motion", "occupancy", "door"]
        missing_essential = [t for t in essential_types if t not in sensor_types]
        if missing_essential:
            result.add_warning(
                f"Room {room_id} missing essential sensor types: {missing_essential}"
            )

        # Check for duplicate entity IDs (across all rooms - would need global context)
        result.add_info(
            f"Room {room_id}: {len(sensor_types)} sensor types, {len(entity_ids)} entities"
        )

        return result

    def _count_sensors(self, sensors: Dict[str, Any]) -> int:
        """Count total number of sensor entities."""
        count = 0
        for sensor_data in sensors.values():
            if isinstance(sensor_data, dict):
                count += len([v for v in sensor_data.values() if isinstance(v, str)])
            elif isinstance(sensor_data, str):
                count += 1
        return count

    def _is_valid_entity_id(self, entity_id: str) -> bool:
        """Check if entity ID has valid format."""
        pattern = r"^(binary_sensor|sensor|light|switch|climate|fan|cover|lock|alarm_control_panel|vacuum|media_player|camera|device_tracker)\.[a-z0-9_]+$"
        return re.match(pattern, entity_id) is not None


class SystemRequirementsValidator:
    """Validates system requirements and environment."""

    def validate(self, config: Dict[str, Any]) -> ValidationResult:
        """Validate system requirements."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        # Check Python version
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        result.add_info(f"Python version: {python_version}")

        if sys.version_info < (3, 9):
            result.add_error(f"Python 3.9+ required, found {python_version}")
        elif sys.version_info < (3, 11):
            result.add_warning(f"Python 3.11+ recommended, found {python_version}")

        # Check required packages
        required_packages = [
            "asyncio",
            "aiohttp",
            "asyncpg",
            "sqlalchemy",
            "pydantic",
            "fastapi",
            "paho.mqtt",
            "numpy",
            "pandas",
            "scikit-learn",
        ]

        missing_packages = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing_packages.append(package)

        if missing_packages:
            result.add_error(f"Missing required packages: {missing_packages}")
        else:
            result.add_info("✅ All required packages are available")

        # Check disk space
        try:
            total, used, free = shutil.disk_usage(".")
            free_gb = free / (1024**3)

            result.add_info(f"Available disk space: {free_gb:.1f} GB")

            if free_gb < 1:
                result.add_error("Insufficient disk space (< 1 GB available)")
            elif free_gb < 5:
                result.add_warning("Low disk space (< 5 GB available)")
        except Exception:
            result.add_warning("Cannot check disk space")

        # Check memory
        try:
            memory = psutil.virtual_memory()
            memory_gb = memory.total / (1024**3)
            available_gb = memory.available / (1024**3)

            result.add_info(
                f"Total memory: {memory_gb:.1f} GB, Available: {available_gb:.1f} GB"
            )

            if memory_gb < 2:
                result.add_error("Insufficient memory (< 2 GB total)")
            elif memory_gb < 4:
                result.add_warning("Low memory (< 4 GB total)")

            if available_gb < 0.5:
                result.add_warning("Very low available memory")

        except ImportError:
            result.add_warning("Cannot check memory usage (psutil not available)")
        except Exception:
            result.add_warning("Cannot check memory usage")

        return result


class ConfigurationValidator:
    """Main configuration validator that orchestrates all validation checks."""

    def __init__(self):
        self.ha_validator = HomeAssistantConfigValidator()
        self.db_validator = DatabaseConfigValidator()
        self.mqtt_validator = MQTTConfigValidator()
        self.rooms_validator = RoomsConfigValidator()
        self.system_validator = SystemRequirementsValidator()

    def validate_configuration(
        self,
        config: Dict[str, Any],
        rooms_config: Dict[str, Any],
        test_connections: bool = False,
        strict_mode: bool = False,
    ) -> ValidationResult:
        """Perform comprehensive configuration validation."""
        logger.info("Starting configuration validation")

        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        # Validate each configuration section
        sections = [
            ("Home Assistant", lambda: self.ha_validator.validate(config)),
            ("Database", lambda: self.db_validator.validate(config)),
            ("MQTT", lambda: self.mqtt_validator.validate(config)),
            ("Rooms", lambda: self.rooms_validator.validate(rooms_config)),
            ("System Requirements", lambda: self.system_validator.validate(config)),
        ]

        for section_name, validator_func in sections:
            try:
                section_result = validator_func()
                result.merge(section_result)

                if section_result.is_valid:
                    result.add_info(f"✅ {section_name} configuration is valid")
                else:
                    result.add_info(f"❌ {section_name} configuration has errors")

            except Exception as e:
                result.add_error(f"Validation failed for {section_name}: {e}")

        # Test connections if requested
        if test_connections:
            result.add_info("Testing external connections...")

            try:
                ha_conn_result = self.ha_validator.test_connection(config)
                result.merge(ha_conn_result)
            except Exception as e:
                result.add_error(f"Home Assistant connection test failed: {e}")

            try:
                db_conn_result = self.db_validator.test_connection(config)
                result.merge(db_conn_result)
            except Exception as e:
                result.add_error(f"Database connection test failed: {e}")

            try:
                mqtt_conn_result = self.mqtt_validator.test_connection(config)
                result.merge(mqtt_conn_result)
            except Exception as e:
                result.add_error(f"MQTT connection test failed: {e}")

        # Apply strict mode rules
        if strict_mode:
            # In strict mode, warnings become errors
            if result.warnings:
                result.add_error(
                    f"Strict mode: {len(result.warnings)} warnings treated as errors"
                )
                result.is_valid = False

        # Final validation summary
        if result.is_valid:
            result.add_info("🎉 Configuration validation completed successfully")
        else:
            result.add_info(
                f"💥 Configuration validation failed with {len(result.errors)} errors"
            )

        logger.info(
            f"Configuration validation completed: {'PASSED' if result.is_valid else 'FAILED'}"
        )

        return result

    def validate_config_files(
        self,
        config_dir: str = "config",
        environment: Optional[str] = None,
        test_connections: bool = False,
    ) -> ValidationResult:
        """Validate configuration files for a specific environment."""
        config_path = Path(config_dir)

        # Determine config file to use
        if environment:
            config_file = config_path / f"config.{environment}.yaml"
            if not config_file.exists():
                config_file = config_path / "config.yaml"
        else:
            config_file = config_path / "config.yaml"

        rooms_file = config_path / "rooms.yaml"

        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        # Load configuration files
        try:
            if not config_file.exists():
                result.add_error(f"Configuration file not found: {config_file}")
                return result

            with open(config_file, "r") as f:
                config = yaml.safe_load(f)

            result.add_info(f"Loaded configuration from: {config_file}")

        except Exception as e:
            result.add_error(f"Failed to load configuration file: {e}")
            return result

        try:
            if not rooms_file.exists():
                result.add_error(f"Rooms configuration file not found: {rooms_file}")
                return result

            with open(rooms_file, "r") as f:
                rooms_config = yaml.safe_load(f)

            result.add_info(f"Loaded rooms configuration from: {rooms_file}")

        except Exception as e:
            result.add_error(f"Failed to load rooms configuration file: {e}")
            return result

        # Perform validation
        validation_result = self.validate_configuration(
            config, rooms_config, test_connections=test_connections
        )

        result.merge(validation_result)
        return result
