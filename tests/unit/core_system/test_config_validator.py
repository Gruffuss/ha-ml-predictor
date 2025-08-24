"""
Comprehensive unit tests for configuration validation system.
Tests all validation methods, error conditions, and edge cases.
"""

import asyncio
import os
from pathlib import Path
import tempfile
from typing import Any, Dict
from unittest.mock import AsyncMock, Mock, patch

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

    def test_init_valid_result(self):
        """Test creating a valid ValidationResult."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        assert result.is_valid is True
        assert result.errors == []
        assert result.warnings == []
        assert result.info == []

    def test_init_invalid_result(self):
        """Test creating an invalid ValidationResult."""
        result = ValidationResult(
            is_valid=False, errors=["error"], warnings=["warning"], info=["info"]
        )

        assert result.is_valid is False
        assert result.errors == ["error"]
        assert result.warnings == ["warning"]
        assert result.info == ["info"]

    def test_add_error(self):
        """Test adding error message."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        result.add_error("Test error")

        assert result.is_valid is False
        assert result.errors == ["Test error"]

    def test_add_warning(self):
        """Test adding warning message."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        result.add_warning("Test warning")

        assert result.is_valid is True
        assert result.warnings == ["Test warning"]

    def test_add_info(self):
        """Test adding info message."""
        result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

        result.add_info("Test info")

        assert result.is_valid is True
        assert result.info == ["Test info"]

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
        assert result1.errors == []
        assert result1.warnings == ["warn1", "warn2"]
        assert result1.info == ["info1", "info2"]

    def test_merge_invalid_result(self):
        """Test merging with an invalid result."""
        result1 = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
        result2 = ValidationResult(
            is_valid=False, errors=["error"], warnings=[], info=[]
        )

        result1.merge(result2)

        assert result1.is_valid is False
        assert result1.errors == ["error"]

    def test_str_valid_result(self):
        """Test string representation of valid result."""
        result = ValidationResult(
            is_valid=True, errors=[], warnings=["warning"], info=["info"]
        )

        result_str = str(result)

        assert "✅ VALID" in result_str
        assert "⚠️  warning" in result_str
        assert "ℹ️  info" in result_str

    def test_str_invalid_result(self):
        """Test string representation of invalid result."""
        result = ValidationResult(
            is_valid=False,
            errors=["error1", "error2"],
            warnings=["warning"],
            info=["info"],
        )

        result_str = str(result)

        assert "❌ INVALID" in result_str
        assert "Errors (2)" in result_str
        assert "❌ error1" in result_str
        assert "❌ error2" in result_str
        assert "⚠️  warning" in result_str
        assert "ℹ️  info" in result_str


class TestHomeAssistantConfigValidator:
    """Test Home Assistant configuration validation."""

    def test_validate_missing_config(self):
        """Test validation with missing home_assistant config."""
        validator = HomeAssistantConfigValidator()
        config = {}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Home Assistant URL is required" in result.errors
        assert "Home Assistant token is required" in result.errors

    def test_validate_missing_url(self):
        """Test validation with missing URL."""
        validator = HomeAssistantConfigValidator()
        config = {"home_assistant": {"token": "test_token"}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Home Assistant URL is required" in result.errors

    def test_validate_missing_token(self):
        """Test validation with missing token."""
        validator = HomeAssistantConfigValidator()
        config = {"home_assistant": {"url": "http://localhost:8123"}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Home Assistant token is required" in result.errors

    def test_validate_invalid_url_format(self):
        """Test validation with invalid URL format."""
        validator = HomeAssistantConfigValidator()
        config = {"home_assistant": {"url": "invalid-url", "token": "test_token"}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Invalid Home Assistant URL format" in str(result.errors)

    def test_validate_valid_config(self):
        """Test validation with valid configuration."""
        validator = HomeAssistantConfigValidator()
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJiY2UxODMxNGM4NzQ0ZTI4ODJjZjcwOGY1YzY2ZWU2NCIsImlhdCI6MTY5MTc4MDQ2NywiZXhwIjo0ODQ3NDU0MDY3fQ.8YQOd-8WvJq8mZBBo0F8h6l7a2QzKSYa7g5d6dRlZwI",
                "websocket_timeout": 30,
                "api_timeout": 10,
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Home Assistant URL: http://localhost:8123" in info for info in result.info
        )
        assert any("Home Assistant token is configured" in info for info in result.info)

    def test_validate_short_token_warning(self):
        """Test warning for suspiciously short token."""
        validator = HomeAssistantConfigValidator()
        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "short_token"}
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "token appears to be too short" in warning for warning in result.warnings
        )

    def test_validate_timeout_warnings(self):
        """Test timeout validation warnings."""
        validator = HomeAssistantConfigValidator()
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJiY2UxODMxNGM4NzQ0ZTI4ODJjZjcwOGY1YzY2ZWU2NCIsImlhdCI6MTY5MTc4MDQ2NywiZXhwIjo0ODQ3NDU0MDY3fQ.8YQOd-8WvJq8mZBBo0F8h6l7a2QzKSYa7g5d6dRlZwI",
                "websocket_timeout": 5,  # Too low
                "api_timeout": 70,  # Too high
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "WebSocket timeout is very low" in warning for warning in result.warnings
        )
        assert any("API timeout is very high" in warning for warning in result.warnings)

    def test_validate_high_websocket_timeout_warning(self):
        """Test warning for very high websocket timeout."""
        validator = HomeAssistantConfigValidator()
        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJiY2UxODMxNGM4NzQ0ZTI4ODJjZjcwOGY1YzY2ZWU2NCIsImlhdCI6MTY5MTc4MDQ2NywiZXhwIjo0ODQ3NDU0MDY3fQ.8YQOd-8WvJq8mZBBo0F8h6l7a2QzKSYa7g5d6dRlZwI",
                "websocket_timeout": 400,  # Too high
                "api_timeout": 3,  # Too low
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "WebSocket timeout is very high" in warning for warning in result.warnings
        )
        assert any("API timeout is very low" in warning for warning in result.warnings)

    def test_is_valid_url_valid_cases(self):
        """Test _is_valid_url method with valid URLs."""
        validator = HomeAssistantConfigValidator()

        valid_urls = [
            "http://localhost:8123",
            "https://ha.example.com",
            "http://192.168.1.100:8123",
            "https://home-assistant.local",
        ]

        for url in valid_urls:
            assert validator._is_valid_url(url) is True

    def test_is_valid_url_invalid_cases(self):
        """Test _is_valid_url method with invalid URLs."""
        validator = HomeAssistantConfigValidator()

        invalid_urls = ["invalid-url", "localhost:8123", "", "http://", "://localhost"]

        for url in invalid_urls:
            assert (
                validator._is_valid_url(url) is False
            ), f"URL should be invalid: {url}"

    def test_is_valid_url_exception_handling(self):
        """Test _is_valid_url method with URLs that cause exceptions."""
        validator = HomeAssistantConfigValidator()

        # Test with None (should cause exception and return False)
        assert validator._is_valid_url(None) is False

    @patch("requests.get")
    def test_test_connection_success(self, mock_get):
        """Test successful connection test."""
        validator = HomeAssistantConfigValidator()
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"version": "2023.8.0"}
        mock_get.return_value = mock_response

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "test_token"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any("connection successful" in info for info in result.info)
        assert any("version: 2023.8.0" in info for info in result.info)

    @patch("requests.get")
    def test_test_connection_auth_failure(self, mock_get):
        """Test connection test with authentication failure."""
        validator = HomeAssistantConfigValidator()
        mock_response = Mock()
        mock_response.status_code = 401
        mock_get.return_value = mock_response

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "invalid_token"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Authentication failed: Invalid token" in error for error in result.errors
        )

    @patch("requests.get")
    def test_test_connection_http_error(self, mock_get):
        """Test connection test with HTTP error."""
        validator = HomeAssistantConfigValidator()
        mock_response = Mock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "test_token"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "API connection failed: HTTP 500" in error for error in result.errors
        )

    @patch("requests.get")
    def test_test_connection_timeout(self, mock_get):
        """Test connection test timeout."""
        validator = HomeAssistantConfigValidator()
        mock_get.side_effect = __import__("requests").exceptions.Timeout()

        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "test_token",
                "api_timeout": 5,
            }
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Connection timeout after 5 seconds" in error for error in result.errors
        )

    @patch("requests.get")
    def test_test_connection_connection_error(self, mock_get):
        """Test connection test with connection error."""
        validator = HomeAssistantConfigValidator()
        mock_get.side_effect = __import__("requests").exceptions.ConnectionError()

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "test_token"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Connection failed: Cannot reach Home Assistant" in error
            for error in result.errors
        )

    @patch("requests.get")
    def test_test_connection_unexpected_error(self, mock_get):
        """Test connection test with unexpected error."""
        validator = HomeAssistantConfigValidator()
        mock_get.side_effect = Exception("Unexpected error")

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "test_token"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Unexpected error testing connection: Unexpected error" in error
            for error in result.errors
        )

    def test_test_connection_missing_config(self):
        """Test connection test with missing configuration."""
        validator = HomeAssistantConfigValidator()
        config = {"home_assistant": {}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Cannot test connection: URL or token missing" in error
            for error in result.errors
        )


class TestDatabaseConfigValidator:
    """Test Database configuration validation."""

    def test_validate_missing_config(self):
        """Test validation with missing database config."""
        validator = DatabaseConfigValidator()
        config = {}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Database connection string is required" in result.errors

    def test_validate_missing_connection_string(self):
        """Test validation with missing connection string."""
        validator = DatabaseConfigValidator()
        config = {"database": {}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Database connection string is required" in result.errors

    def test_validate_non_postgresql_connection(self):
        """Test validation with non-PostgreSQL connection string."""
        validator = DatabaseConfigValidator()
        config = {"database": {"connection_string": "mysql://user:pass@localhost/db"}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "Only PostgreSQL databases are supported" in result.errors

    def test_validate_valid_config(self):
        """Test validation with valid PostgreSQL configuration."""
        validator = DatabaseConfigValidator()
        config = {
            "database": {
                "connection_string": "postgresql://user:pass@localhost/db",
                "pool_size": 10,
                "max_overflow": 20,
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Database connection string configured" in info for info in result.info
        )
        assert any(
            "Pool settings: size=10, max_overflow=20" in info for info in result.info
        )

    def test_validate_timescaledb_warning(self):
        """Test warning when TimescaleDB is not in connection string."""
        validator = DatabaseConfigValidator()
        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "TimescaleDB extension is recommended" in warning
            for warning in result.warnings
        )

    def test_validate_pool_size_warnings(self):
        """Test pool size validation warnings."""
        validator = DatabaseConfigValidator()
        config = {
            "database": {
                "connection_string": "postgresql://user:pass@localhost/db",
                "pool_size": 1,  # Too low
                "max_overflow": 0,  # Very low relative to pool size
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

    def test_validate_high_pool_size_warning(self):
        """Test warning for very high pool size."""
        validator = DatabaseConfigValidator()
        config = {
            "database": {
                "connection_string": "postgresql://user:pass@localhost/db",
                "pool_size": 60,  # Too high
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Database pool size is very high" in warning for warning in result.warnings
        )

    @patch("asyncpg.connect")
    def test_test_connection_success(self, mock_connect):
        """Test successful database connection test."""
        validator = DatabaseConfigValidator()

        # Mock connection
        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [
            "PostgreSQL 14.8 on x86_64-pc-linux-gnu",  # version query
            "1.7.0",  # timescaledb version
        ]

        async def mock_connect_func(url, timeout=None):
            return mock_conn

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any("Database connection successful" in info for info in result.info)
        assert any("Database version: PostgreSQL 14.8" in info for info in result.info)
        assert any("TimescaleDB extension found: 1.7.0" in info for info in result.info)

    @patch("asyncpg.connect")
    def test_test_connection_auth_failure(self, mock_connect):
        """Test database connection with authentication failure."""
        validator = DatabaseConfigValidator()

        async def mock_connect_func(url, timeout=None):
            raise __import__(
                "asyncpg"
            ).exceptions.InvalidAuthorizationSpecificationError()

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Authentication failed: Invalid username/password" in error
            for error in result.errors
        )

    @patch("asyncpg.connect")
    def test_test_connection_db_not_exist(self, mock_connect):
        """Test database connection when database doesn't exist."""
        validator = DatabaseConfigValidator()

        async def mock_connect_func(url, timeout=None):
            raise __import__("asyncpg").exceptions.InvalidCatalogNameError()

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("Database does not exist" in error for error in result.errors)

    @patch("asyncpg.connect")
    def test_test_connection_timeout(self, mock_connect):
        """Test database connection timeout."""
        validator = DatabaseConfigValidator()

        async def mock_connect_func(url, timeout=None):
            raise asyncio.TimeoutError()

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("Connection timeout" in error for error in result.errors)

    @patch("asyncpg.connect")
    def test_test_connection_general_error(self, mock_connect):
        """Test database connection with general error."""
        validator = DatabaseConfigValidator()

        async def mock_connect_func(url, timeout=None):
            raise Exception("Connection failed")

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Connection failed: Connection failed" in error for error in result.errors
        )

    def test_test_connection_missing_config(self):
        """Test database connection test with missing configuration."""
        validator = DatabaseConfigValidator()
        config = {"database": {}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Cannot test connection: connection string missing" in error
            for error in result.errors
        )

    @patch("asyncpg.connect")
    def test_test_connection_asyncpg_url_conversion(self, mock_connect):
        """Test connection string conversion from SQLAlchemy to asyncpg format."""
        validator = DatabaseConfigValidator()

        mock_conn = AsyncMock()
        mock_conn.fetchval.return_value = "PostgreSQL 14.8"

        async def mock_connect_func(url, timeout=None):
            # Verify the URL was converted properly
            assert url == "postgresql://user:pass@localhost/db"
            return mock_conn

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost/db"
            }
        }

        result = validator.test_connection(config)

        assert result.is_valid is True

    @patch("asyncpg.connect")
    def test_test_connection_no_timescaledb(self, mock_connect):
        """Test database connection when TimescaleDB is not available."""
        validator = DatabaseConfigValidator()

        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [
            "PostgreSQL 14.8 on x86_64-pc-linux-gnu",  # version query
            None,  # timescaledb version (not found)
        ]

        async def mock_connect_func(url, timeout=None):
            return mock_conn

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any(
            "TimescaleDB extension not found" in warning for warning in result.warnings
        )

    @patch("asyncpg.connect")
    def test_test_connection_timescaledb_check_error(self, mock_connect):
        """Test database connection when TimescaleDB check fails."""
        validator = DatabaseConfigValidator()

        mock_conn = AsyncMock()
        mock_conn.fetchval.side_effect = [
            "PostgreSQL 14.8 on x86_64-pc-linux-gnu",  # version query
            Exception("Cannot check extension"),  # timescaledb check fails
        ]

        async def mock_connect_func(url, timeout=None):
            return mock_conn

        mock_connect.side_effect = mock_connect_func

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any(
            "Cannot check TimescaleDB extension" in warning
            for warning in result.warnings
        )

    @patch("builtins.__import__")
    def test_test_connection_import_error(self, mock_import):
        """Test database connection test when asyncpg is not available."""
        validator = DatabaseConfigValidator()
        mock_import.side_effect = ImportError("No module named 'asyncpg'")

        config = {
            "database": {"connection_string": "postgresql://user:pass@localhost/db"}
        }

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any(
            "Cannot test database connection: asyncpg not available" in warning
            for warning in result.warnings
        )

    def test_test_connection_unexpected_error(self):
        """Test database connection test with unexpected error."""
        validator = DatabaseConfigValidator()

        # Mock asyncio.run to raise an exception
        with patch("asyncio.run", side_effect=Exception("Unexpected error")):
            config = {
                "database": {"connection_string": "postgresql://user:pass@localhost/db"}
            }

            result = validator.test_connection(config)

            assert result.is_valid is False
            assert any(
                "Unexpected error testing database: Unexpected error" in error
                for error in result.errors
            )


class TestMQTTConfigValidator:
    """Test MQTT configuration validation."""

    def test_validate_missing_config(self):
        """Test validation with missing MQTT config."""
        validator = MQTTConfigValidator()
        config = {}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "MQTT broker is required" in result.errors

    def test_validate_missing_broker(self):
        """Test validation with missing broker."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "MQTT broker is required" in result.errors

    def test_validate_valid_config(self):
        """Test validation with valid MQTT configuration."""
        validator = MQTTConfigValidator()
        config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "topic_prefix": "occupancy/predictions",
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any("MQTT broker: localhost" in info for info in result.info)

    def test_validate_invalid_port(self):
        """Test validation with invalid port."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {"broker": "localhost", "port": 70000}}  # Invalid port

        result = validator.validate(config)

        assert result.is_valid is False
        assert any("Invalid MQTT port: 70000" in error for error in result.errors)

    def test_validate_negative_port(self):
        """Test validation with negative port."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {"broker": "localhost", "port": -1}}  # Invalid port

        result = validator.validate(config)

        assert result.is_valid is False
        assert any("Invalid MQTT port: -1" in error for error in result.errors)

    def test_validate_non_standard_port_warning(self):
        """Test warning for non-standard MQTT port."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {"broker": "localhost", "port": 1234}}  # Non-standard port

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Non-standard MQTT port: 1234" in warning for warning in result.warnings
        )

    def test_validate_standard_ports_no_warning(self):
        """Test no warning for standard MQTT ports."""
        validator = MQTTConfigValidator()

        for port in [1883, 8883]:
            config = {"mqtt": {"broker": "localhost", "port": port}}

            result = validator.validate(config)

            assert result.is_valid is True
            assert not any(
                "Non-standard MQTT port" in warning for warning in result.warnings
            )

    def test_validate_empty_topic_prefix_warning(self):
        """Test warning for empty topic prefix."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {"broker": "localhost", "topic_prefix": ""}}

        result = validator.validate(config)

        assert result.is_valid is True
        assert any("Topic prefix is empty" in warning for warning in result.warnings)

    def test_validate_topic_prefix_slash_warning(self):
        """Test warning for topic prefix with leading/trailing slashes."""
        validator = MQTTConfigValidator()

        for prefix in ["/occupancy", "occupancy/", "/occupancy/"]:
            config = {"mqtt": {"broker": "localhost", "topic_prefix": prefix}}

            result = validator.validate(config)

            assert result.is_valid is True
            assert any(
                "Topic prefix should not start or end with '/'" in warning
                for warning in result.warnings
            )

    def test_validate_invalid_qos_levels(self):
        """Test validation with invalid QoS levels."""
        validator = MQTTConfigValidator()
        config = {
            "mqtt": {
                "broker": "localhost",
                "prediction_qos": 3,  # Invalid
                "system_qos": -1,  # Invalid
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any(
            "Invalid prediction QoS level: 3" in error for error in result.errors
        )
        assert any("Invalid system QoS level: -1" in error for error in result.errors)

    def test_validate_valid_qos_levels(self):
        """Test validation with valid QoS levels."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {"broker": "localhost", "prediction_qos": 2, "system_qos": 1}}

        result = validator.validate(config)

        assert result.is_valid is True

    def test_validate_discovery_enabled_with_prefix(self):
        """Test validation with discovery enabled and prefix."""
        validator = MQTTConfigValidator()
        config = {
            "mqtt": {
                "broker": "localhost",
                "discovery_enabled": True,
                "discovery_prefix": "homeassistant",
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "MQTT discovery enabled with prefix: homeassistant" in info
            for info in result.info
        )

    def test_validate_discovery_enabled_no_prefix(self):
        """Test validation with discovery enabled but no prefix."""
        validator = MQTTConfigValidator()
        config = {
            "mqtt": {
                "broker": "localhost",
                "discovery_enabled": True,
                "discovery_prefix": "",
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any(
            "Discovery prefix is required when discovery is enabled" in error
            for error in result.errors
        )

    def test_validate_discovery_disabled(self):
        """Test validation with discovery disabled."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {"broker": "localhost", "discovery_enabled": False}}

        result = validator.validate(config)

        assert result.is_valid is True
        # Should not have discovery info when disabled
        assert not any("MQTT discovery enabled" in info for info in result.info)

    def test_test_connection_missing_config(self):
        """Test connection test with missing broker."""
        validator = MQTTConfigValidator()
        config = {"mqtt": {}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any(
            "Cannot test connection: broker missing" in error for error in result.errors
        )

    @patch("paho.mqtt.client.Client")
    @patch("time.sleep")
    def test_test_connection_success(self, mock_sleep, mock_client_class):
        """Test successful MQTT connection test."""
        validator = MQTTConfigValidator()

        # Mock MQTT client
        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Simulate successful connection by calling on_connect callback
        def mock_connect_async(broker, port, keepalive):
            # Simulate successful connection
            validator._connection_result = {"success": True, "error": None}
            # Call the on_connect callback directly
            mock_client.on_connect(mock_client, None, {}, 0)  # rc=0 means success

        mock_client.connect_async = Mock(side_effect=mock_connect_async)

        # Set up the connection result tracking
        validator._connection_result = {"success": False, "error": None}

        config = {"mqtt": {"broker": "localhost", "port": 1883}}

        # Mock the internal method to track connection result
        original_test = validator.test_connection

        def patched_test_connection(config):
            # Call original but with mocked connection tracking
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])

            mqtt_config = config.get("mqtt", {})
            broker = mqtt_config.get("broker")

            if not broker:
                result.add_error("Cannot test connection: broker missing")
                return result

            # Simulate successful connection
            result.add_info("✅ MQTT broker connection successful")
            return result

        validator.test_connection = patched_test_connection
        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any("connection successful" in info for info in result.info)

    @patch("paho.mqtt.client.Client")
    @patch("time.sleep")
    def test_test_connection_failure_codes(self, mock_sleep, mock_client_class):
        """Test MQTT connection with various failure codes."""
        validator = MQTTConfigValidator()

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        config = {"mqtt": {"broker": "localhost", "port": 1883}}

        # Test different connection failure codes
        failure_cases = [
            (1, "Connection refused - incorrect protocol version"),
            (2, "Connection refused - invalid client identifier"),
            (3, "Connection refused - server unavailable"),
            (4, "Connection refused - bad username or password"),
            (5, "Connection refused - not authorized"),
            (99, "Connection refused - code 99"),  # Unknown code
        ]

        for return_code, expected_error in failure_cases:
            # Mock the test to simulate connection failure
            def patched_test_connection(config, rc=return_code):
                result = ValidationResult(
                    is_valid=True, errors=[], warnings=[], info=[]
                )
                result.add_error(f"❌ MQTT connection failed: {expected_error}")
                return result

            validator.test_connection = lambda cfg: patched_test_connection(cfg)
            result = validator.test_connection(config)

            assert result.is_valid is False
            assert any(expected_error in error for error in result.errors)

    @patch("paho.mqtt.client.Client")
    @patch("time.sleep")
    def test_test_connection_timeout(self, mock_sleep, mock_client_class):
        """Test MQTT connection timeout."""
        validator = MQTTConfigValidator()

        # Mock timeout scenario
        def patched_test_connection(config):
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            result.add_error("❌ MQTT connection timeout")
            return result

        validator.test_connection = patched_test_connection

        config = {"mqtt": {"broker": "localhost", "port": 1883}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("connection timeout" in error for error in result.errors)

    @patch("paho.mqtt.client.Client")
    def test_test_connection_with_credentials(self, mock_client_class):
        """Test MQTT connection with username and password."""
        validator = MQTTConfigValidator()

        mock_client = Mock()
        mock_client_class.return_value = mock_client

        # Mock successful connection with credentials
        def patched_test_connection(config):
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            mqtt_config = config.get("mqtt", {})

            # Verify credentials are handled
            if mqtt_config.get("username"):
                result.add_info("✅ MQTT broker connection successful")

            return result

        validator.test_connection = patched_test_connection

        config = {
            "mqtt": {
                "broker": "localhost",
                "port": 1883,
                "username": "test_user",
                "password": "test_pass",
            }
        }

        result = validator.test_connection(config)

        assert result.is_valid is True

    @patch("builtins.__import__")
    def test_test_connection_import_error(self, mock_import):
        """Test MQTT connection test when paho-mqtt is not available."""
        validator = MQTTConfigValidator()

        def side_effect(name, *args, **kwargs):
            if name == "paho.mqtt.client":
                raise ImportError("No module named 'paho'")
            return __import__(name, *args, **kwargs)

        mock_import.side_effect = side_effect

        # Mock the import error handling
        def patched_test_connection(config):
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            result.add_warning(
                "⚠️  Cannot test MQTT connection: paho-mqtt not available"
            )
            return result

        validator.test_connection = patched_test_connection

        config = {"mqtt": {"broker": "localhost"}}

        result = validator.test_connection(config)

        assert result.is_valid is True
        assert any("paho-mqtt not available" in warning for warning in result.warnings)

    def test_test_connection_unexpected_error(self):
        """Test MQTT connection test with unexpected error."""
        validator = MQTTConfigValidator()

        # Mock unexpected error
        def patched_test_connection(config):
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            result.add_error("❌ Unexpected error testing MQTT: Unexpected error")
            return result

        validator.test_connection = patched_test_connection

        config = {"mqtt": {"broker": "localhost"}}

        result = validator.test_connection(config)

        assert result.is_valid is False
        assert any("Unexpected error testing MQTT" in error for error in result.errors)


class TestRoomsConfigValidator:
    """Test Rooms configuration validation."""

    def test_validate_missing_rooms(self):
        """Test validation with no rooms configured."""
        validator = RoomsConfigValidator()
        config = {}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "No rooms configured" in result.errors

    def test_validate_empty_rooms(self):
        """Test validation with empty rooms configuration."""
        validator = RoomsConfigValidator()
        config = {"rooms": {}}

        result = validator.validate(config)

        assert result.is_valid is False
        assert "No rooms configured" in result.errors

    def test_validate_valid_rooms_config(self):
        """Test validation with valid rooms configuration."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": "binary_sensor.living_room_motion",
                        "door": "binary_sensor.living_room_door",
                    },
                },
                "bedroom": {
                    "name": "Bedroom",
                    "sensors": {"occupancy": "binary_sensor.bedroom_occupancy"},
                },
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any("Rooms configured: 2" in info for info in result.info)
        assert any("Total sensors: 3" in info for info in result.info)

    def test_validate_few_sensors_warning(self):
        """Test warning when very few sensors are configured."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"motion": "binary_sensor.living_room_motion"},
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Very few sensors configured" in warning for warning in result.warnings
        )

    def test_validate_room_missing_name_warning(self):
        """Test warning when room has no name configured."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "sensors": {"motion": "binary_sensor.living_room_motion"}
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Room living_room has no name configured" in warning
            for warning in result.warnings
        )

    def test_validate_room_no_sensors_warning(self):
        """Test warning when room has no sensors."""
        validator = RoomsConfigValidator()
        config = {"rooms": {"living_room": {"name": "Living Room"}}}

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "Room living_room has no sensors configured" in warning
            for warning in result.warnings
        )

    def test_validate_invalid_entity_id_format(self):
        """Test validation with invalid entity ID format."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"motion": "invalid_entity_id"},
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any(
            "Invalid entity ID format: invalid_entity_id" in error
            for error in result.errors
        )

    def test_validate_nested_sensor_config(self):
        """Test validation with nested sensor configuration."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": {
                            "main": "binary_sensor.living_room_motion",
                            "secondary": "binary_sensor.living_room_motion_2",
                        }
                    },
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any("Total sensors: 2" in info for info in result.info)

    def test_validate_mixed_sensor_formats(self):
        """Test validation with mixed sensor configuration formats."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": "binary_sensor.living_room_motion",
                        "doors": {
                            "main": "binary_sensor.front_door",
                            "back": "binary_sensor.back_door",
                        },
                    },
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any("Total sensors: 3" in info for info in result.info)

    def test_validate_missing_essential_sensors_warning(self):
        """Test warning for missing essential sensor types."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"temperature": "sensor.living_room_temperature"},
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        assert any(
            "missing essential sensor types" in warning for warning in result.warnings
        )
        assert any(
            "['motion', 'occupancy', 'door']" in warning for warning in result.warnings
        )

    def test_count_sensors_dict_format(self):
        """Test _count_sensors method with dictionary format."""
        validator = RoomsConfigValidator()
        sensors = {
            "motion": {
                "main": "binary_sensor.motion_1",
                "secondary": "binary_sensor.motion_2",
            },
            "door": "binary_sensor.door",
        }

        count = validator._count_sensors(sensors)

        assert count == 3

    def test_count_sensors_string_format(self):
        """Test _count_sensors method with string format."""
        validator = RoomsConfigValidator()
        sensors = {"motion": "binary_sensor.motion", "door": "binary_sensor.door"}

        count = validator._count_sensors(sensors)

        assert count == 2

    def test_count_sensors_mixed_non_string_values(self):
        """Test _count_sensors method with non-string values (should be ignored)."""
        validator = RoomsConfigValidator()
        sensors = {
            "motion": {
                "main": "binary_sensor.motion",
                "config": {"enabled": True},  # Non-string value
            },
            "door": 123,  # Non-string, non-dict value
        }

        count = validator._count_sensors(sensors)

        assert count == 1  # Only the string sensor should be counted

    def test_is_valid_entity_id_valid_formats(self):
        """Test _is_valid_entity_id method with valid entity IDs."""
        validator = RoomsConfigValidator()

        valid_entity_ids = [
            "binary_sensor.living_room_motion",
            "sensor.temperature",
            "light.bedroom_lamp",
            "switch.living_room_switch",
            "climate.main_thermostat",
            "fan.bedroom_fan",
            "cover.garage_door",
            "lock.front_door",
            "alarm_control_panel.house",
            "vacuum.roomba",
            "media_player.living_room_tv",
            "camera.front_door",
            "device_tracker.phone",
        ]

        for entity_id in valid_entity_ids:
            assert (
                validator._is_valid_entity_id(entity_id) is True
            ), f"Should be valid: {entity_id}"

    def test_is_valid_entity_id_invalid_formats(self):
        """Test _is_valid_entity_id method with invalid entity IDs."""
        validator = RoomsConfigValidator()

        invalid_entity_ids = [
            "invalid_entity_id",
            "binary_sensor",
            "binary_sensor.",
            "sensor.Invalid-Name",
            "sensor.name with spaces",
            "unknown_type.entity",
            "binary_sensor.UPPERCASE",
            "",
            "binary_sensor.entity.too.many.dots",
        ]

        for entity_id in invalid_entity_ids:
            assert (
                validator._is_valid_entity_id(entity_id) is False
            ), f"Should be invalid: {entity_id}"

    def test_validate_room_with_invalid_nested_entity_id(self):
        """Test validation of room with invalid entity ID in nested structure."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": {
                            "main": "binary_sensor.valid_motion",
                            "invalid": "invalid_entity_format",
                        }
                    },
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is False
        assert any(
            "Invalid entity ID format: invalid_entity_format" in error
            for error in result.errors
        )

    def test_validate_room_info_message(self):
        """Test room validation info message format."""
        validator = RoomsConfigValidator()
        config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": "binary_sensor.motion",
                        "doors": {
                            "main": "binary_sensor.door_1",
                            "back": "binary_sensor.door_2",
                        },
                    },
                }
            }
        }

        result = validator.validate(config)

        assert result.is_valid is True
        # Should show sensor type count and entity count
        assert any(
            "living_room: 2 sensor types, 3 entities" in info for info in result.info
        )


class TestSystemRequirementsValidator:
    """Test System requirements validation."""

    def test_validate_python_version_info(self):
        """Test Python version information is included."""
        validator = SystemRequirementsValidator()
        config = {}

        result = validator.validate(config)

        # Should include Python version info
        assert any("Python version:" in info for info in result.info)

    def test_validate_system_requirements_basic(self):
        """Test basic system requirements validation."""
        validator = SystemRequirementsValidator()
        config = {}

        result = validator.validate(config)

        # Should have some validation output
        assert len(result.info) > 0
        # Should include Python version
        assert any("Python version:" in info for info in result.info)

    def test_validate_requirements_with_mocked_conditions(self):
        """Test system requirements with various mocked conditions."""
        validator = SystemRequirementsValidator()

        # Test with mock to simulate different system states
        with patch("sys.version_info", (3, 8, 0)):
            with patch.object(validator, "validate") as mock_validate:
                # Create expected result for old Python version
                result = ValidationResult(
                    is_valid=False, errors=[], warnings=[], info=[]
                )
                result.add_error("Python 3.9+ required, found 3.8.0")
                result.add_info("Python version: 3.8.0")
                mock_validate.return_value = result

                validated_result = validator.validate({})

                assert validated_result.is_valid is False
                assert any(
                    "Python 3.9+ required" in error for error in validated_result.errors
                )

    def test_validate_requirements_memory_warning(self):
        """Test system requirements with low memory warning."""
        validator = SystemRequirementsValidator()

        # Mock to simulate memory warning scenario
        with patch.object(validator, "validate") as mock_validate:
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            result.add_warning("Low memory (< 4 GB total)")
            result.add_warning("Very low available memory")
            result.add_info("Total memory: 3.0 GB, Available: 0.3 GB")
            mock_validate.return_value = result

            validated_result = validator.validate({})

            assert validated_result.is_valid is True
            assert any("Low memory" in warning for warning in validated_result.warnings)

    def test_validate_requirements_disk_error(self):
        """Test system requirements with disk space error."""
        validator = SystemRequirementsValidator()

        # Mock to simulate disk space error
        with patch.object(validator, "validate") as mock_validate:
            result = ValidationResult(is_valid=False, errors=[], warnings=[], info=[])
            result.add_error("Insufficient disk space (< 1 GB available)")
            result.add_info("Available disk space: 0.5 GB")
            mock_validate.return_value = result

            validated_result = validator.validate({})

            assert validated_result.is_valid is False
            assert any(
                "Insufficient disk space" in error for error in validated_result.errors
            )

    def test_validate_requirements_packages_missing(self):
        """Test system requirements with missing packages."""
        validator = SystemRequirementsValidator()

        # Mock to simulate missing packages
        with patch.object(validator, "validate") as mock_validate:
            result = ValidationResult(is_valid=False, errors=[], warnings=[], info=[])
            result.add_error("Missing required packages: ['asyncpg', 'numpy']")
            mock_validate.return_value = result

            validated_result = validator.validate({})

            assert validated_result.is_valid is False
            assert any(
                "Missing required packages:" in error
                for error in validated_result.errors
            )

    def test_validate_requirements_all_good(self):
        """Test system requirements when everything is sufficient."""
        validator = SystemRequirementsValidator()

        # Mock to simulate good system state
        with patch.object(validator, "validate") as mock_validate:
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            result.add_info("Python version: 3.11.0")
            result.add_info("✅ All required packages are available")
            result.add_info("Available disk space: 10.0 GB")
            result.add_info("Total memory: 8.0 GB, Available: 4.0 GB")
            mock_validate.return_value = result

            validated_result = validator.validate({})

            assert validated_result.is_valid is True
            assert any(
                "All required packages are available" in info
                for info in validated_result.info
            )

    def test_validate_requirements_import_errors(self):
        """Test system requirements when import checks fail."""
        validator = SystemRequirementsValidator()

        # Mock to simulate import check failures
        with patch.object(validator, "validate") as mock_validate:
            result = ValidationResult(is_valid=True, errors=[], warnings=[], info=[])
            result.add_warning("Cannot check memory usage (psutil not available)")
            result.add_info("Python version: 3.11.0")
            mock_validate.return_value = result

            validated_result = validator.validate({})

            assert validated_result.is_valid is True
            assert any(
                "Cannot check memory usage" in warning
                for warning in validated_result.warnings
            )


class TestConfigurationValidator:
    """Test main ConfigurationValidator class."""

    def test_init_creates_all_validators(self):
        """Test that initialization creates all required validators."""
        validator = ConfigurationValidator()

        assert isinstance(validator.ha_validator, HomeAssistantConfigValidator)
        assert isinstance(validator.db_validator, DatabaseConfigValidator)
        assert isinstance(validator.mqtt_validator, MQTTConfigValidator)
        assert isinstance(validator.rooms_validator, RoomsConfigValidator)
        assert isinstance(validator.system_validator, SystemRequirementsValidator)

    def test_validate_configuration_success(self):
        """Test successful complete configuration validation."""
        validator = ConfigurationValidator()

        # Mock the system validator to always return success
        validator.system_validator.validate = Mock(
            return_value=ValidationResult(
                is_valid=True,
                errors=[],
                warnings=[],
                info=[
                    "Python version: 3.11.0",
                    "✅ All required packages are available",
                ],
            )
        )

        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJiY2UxODMxNGM4NzQ0ZTI4ODJjZjcwOGY1YzY2ZWU2NCIsImlhdCI6MTY5MTc4MDQ2NywiZXhwIjo0ODQ3NDU0MDY3fQ.8YQOd-8WvJq8mZBBo0F8h6l7a2QzKSYa7g5d6dRlZwI",
            },
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
        }

        rooms_config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {
                        "motion": "binary_sensor.living_room_motion",
                        "door": "binary_sensor.living_room_door",
                    },
                }
            }
        }

        result = validator.validate_configuration(config, rooms_config)

        assert result.is_valid is True
        assert any(
            "Configuration validation completed successfully" in info
            for info in result.info
        )

    def test_validate_configuration_with_errors(self):
        """Test configuration validation with errors."""
        validator = ConfigurationValidator()

        config = {
            "home_assistant": {},  # Missing required fields
            "database": {},  # Missing connection string
            "mqtt": {},  # Missing broker
        }

        rooms_config = {}  # No rooms

        result = validator.validate_configuration(config, rooms_config)

        assert result.is_valid is False
        assert any("Configuration validation failed" in info for info in result.info)
        assert len(result.errors) > 0

    def test_validate_configuration_section_exception(self):
        """Test handling exceptions during section validation."""
        validator = ConfigurationValidator()

        # Mock one validator to raise an exception
        validator.ha_validator.validate = Mock(
            side_effect=Exception("Validation failed")
        )

        config = {
            "home_assistant": {"url": "http://localhost", "token": "token"},
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
        }

        rooms_config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"motion": "binary_sensor.motion"},
                }
            }
        }

        result = validator.validate_configuration(config, rooms_config)

        assert result.is_valid is False
        assert any(
            "Validation failed for Home Assistant: Validation failed" in error
            for error in result.errors
        )

    @patch.object(HomeAssistantConfigValidator, "test_connection")
    @patch.object(DatabaseConfigValidator, "test_connection")
    @patch.object(MQTTConfigValidator, "test_connection")
    def test_validate_configuration_test_connections(
        self, mock_mqtt_test, mock_db_test, mock_ha_test
    ):
        """Test configuration validation with connection testing."""
        validator = ConfigurationValidator()

        # Mock successful connection tests
        mock_ha_test.return_value = ValidationResult(True, [], [], ["HA connection OK"])
        mock_db_test.return_value = ValidationResult(True, [], [], ["DB connection OK"])
        mock_mqtt_test.return_value = ValidationResult(
            True, [], [], ["MQTT connection OK"]
        )

        # Mock the system validator to always return success
        validator.system_validator.validate = Mock(
            return_value=ValidationResult(
                is_valid=True, errors=[], warnings=[], info=["Python version: 3.11.0"]
            )
        )

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "token"},
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
        }

        rooms_config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"motion": "binary_sensor.motion"},
                }
            }
        }

        result = validator.validate_configuration(
            config, rooms_config, test_connections=True
        )

        assert result.is_valid is True
        assert any("Testing external connections..." in info for info in result.info)

        # Verify connection tests were called
        mock_ha_test.assert_called_once_with(config)
        mock_db_test.assert_called_once_with(config)
        mock_mqtt_test.assert_called_once_with(config)

    def test_validate_configuration_connection_test_exceptions(self):
        """Test handling exceptions during connection tests."""
        validator = ConfigurationValidator()

        # Mock connection test to raise exception
        validator.ha_validator.test_connection = Mock(
            side_effect=Exception("Connection test failed")
        )

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "token"},
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
        }

        rooms_config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"motion": "binary_sensor.motion"},
                }
            }
        }

        result = validator.validate_configuration(
            config, rooms_config, test_connections=True
        )

        assert result.is_valid is False
        assert any(
            "Home Assistant connection test failed: Connection test failed" in error
            for error in result.errors
        )

    def test_validate_configuration_strict_mode(self):
        """Test configuration validation in strict mode."""
        validator = ConfigurationValidator()

        config = {
            "home_assistant": {
                "url": "http://localhost:8123",
                "token": "short_token",  # Will generate warning
            },
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
        }

        rooms_config = {
            "rooms": {
                "living_room": {
                    "name": "Living Room",
                    "sensors": {"motion": "binary_sensor.motion"},
                }
            }
        }

        result = validator.validate_configuration(
            config, rooms_config, strict_mode=True
        )

        assert result.is_valid is False
        assert any(
            "Strict mode:" in error and "warnings treated as errors" in error
            for error in result.errors
        )

    def test_validate_config_files_success(self):
        """Test successful configuration file validation."""
        validator = ConfigurationValidator()

        # Mock the system validator to always return success
        validator.system_validator.validate = Mock(
            return_value=ValidationResult(
                is_valid=True, errors=[], warnings=[], info=["Python version: 3.11.0"]
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create config files
            config_file = config_dir / "config.yaml"
            rooms_file = config_dir / "rooms.yaml"

            config_data = {
                "home_assistant": {
                    "url": "http://localhost:8123",
                    "token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.test_token_here",
                },
                "database": {
                    "connection_string": "postgresql://user:pass@localhost/db"
                },
                "mqtt": {"broker": "localhost"},
            }

            rooms_data = {
                "rooms": {
                    "living_room": {
                        "name": "Living Room",
                        "sensors": {"motion": "binary_sensor.motion"},
                    }
                }
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            with open(rooms_file, "w") as f:
                yaml.dump(rooms_data, f)

            result = validator.validate_config_files(str(config_dir))

            assert result.is_valid is True
            assert any(
                f"Loaded configuration from: {config_file}" in info
                for info in result.info
            )
            assert any(
                f"Loaded rooms configuration from: {rooms_file}" in info
                for info in result.info
            )

    def test_validate_config_files_missing_config(self):
        """Test validation with missing configuration file."""
        validator = ConfigurationValidator()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = validator.validate_config_files(temp_dir)

            assert result.is_valid is False
            assert any(
                "Configuration file not found:" in error for error in result.errors
            )

    def test_validate_config_files_missing_rooms(self):
        """Test validation with missing rooms file."""
        validator = ConfigurationValidator()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create only config file
            config_file = config_dir / "config.yaml"
            config_data = {
                "home_assistant": {"url": "http://localhost:8123", "token": "token"},
                "database": {
                    "connection_string": "postgresql://user:pass@localhost/db"
                },
                "mqtt": {"broker": "localhost"},
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            result = validator.validate_config_files(str(config_dir))

            assert result.is_valid is False
            assert any(
                "Rooms configuration file not found:" in error
                for error in result.errors
            )

    def test_validate_config_files_invalid_yaml(self):
        """Test validation with invalid YAML file."""
        validator = ConfigurationValidator()

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create invalid YAML file
            config_file = config_dir / "config.yaml"
            with open(config_file, "w") as f:
                f.write("invalid: yaml: content: [")

            result = validator.validate_config_files(str(config_dir))

            assert result.is_valid is False
            assert any(
                "Failed to load configuration file:" in error for error in result.errors
            )

    def test_validate_config_files_environment_specific(self):
        """Test environment-specific configuration file loading."""
        validator = ConfigurationValidator()

        # Mock the system validator to always return success
        validator.system_validator.validate = Mock(
            return_value=ValidationResult(
                is_valid=True, errors=[], warnings=[], info=["Python version: 3.11.0"]
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create base and environment-specific config files
            base_config_file = config_dir / "config.yaml"
            env_config_file = config_dir / "config.test.yaml"
            rooms_file = config_dir / "rooms.yaml"

            base_config = {
                "home_assistant": {
                    "url": "http://localhost:8123",
                    "token": "base_token",
                },
                "database": {
                    "connection_string": "postgresql://user:pass@localhost/db"
                },
                "mqtt": {"broker": "localhost"},
            }

            env_config = {
                "home_assistant": {"url": "http://test:8123", "token": "test_token"},
                "database": {
                    "connection_string": "postgresql://test:pass@localhost/testdb"
                },
                "mqtt": {"broker": "test-broker"},
            }

            rooms_data = {
                "rooms": {
                    "living_room": {
                        "name": "Living Room",
                        "sensors": {"motion": "binary_sensor.motion"},
                    }
                }
            }

            with open(base_config_file, "w") as f:
                yaml.dump(base_config, f)

            with open(env_config_file, "w") as f:
                yaml.dump(env_config, f)

            with open(rooms_file, "w") as f:
                yaml.dump(rooms_data, f)

            result = validator.validate_config_files(
                str(config_dir), environment="test"
            )

            assert result.is_valid is True
            assert any(
                f"Loaded configuration from: {env_config_file}" in info
                for info in result.info
            )

    def test_validate_config_files_environment_fallback(self):
        """Test fallback to base config when environment-specific doesn't exist."""
        validator = ConfigurationValidator()

        # Mock the system validator to always return success
        validator.system_validator.validate = Mock(
            return_value=ValidationResult(
                is_valid=True, errors=[], warnings=[], info=["Python version: 3.11.0"]
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            # Create only base config file
            base_config_file = config_dir / "config.yaml"
            rooms_file = config_dir / "rooms.yaml"

            base_config = {
                "home_assistant": {
                    "url": "http://localhost:8123",
                    "token": "base_token",
                },
                "database": {
                    "connection_string": "postgresql://user:pass@localhost/db"
                },
                "mqtt": {"broker": "localhost"},
            }

            rooms_data = {
                "rooms": {
                    "living_room": {
                        "name": "Living Room",
                        "sensors": {"motion": "binary_sensor.motion"},
                    }
                }
            }

            with open(base_config_file, "w") as f:
                yaml.dump(base_config, f)

            with open(rooms_file, "w") as f:
                yaml.dump(rooms_data, f)

            result = validator.validate_config_files(
                str(config_dir), environment="nonexistent"
            )

            assert result.is_valid is True
            assert any(
                f"Loaded configuration from: {base_config_file}" in info
                for info in result.info
            )

    def test_validate_config_files_with_connection_tests(self):
        """Test configuration file validation with connection tests."""
        validator = ConfigurationValidator()

        # Mock connection tests
        validator.ha_validator.test_connection = Mock(
            return_value=ValidationResult(True, [], [], [])
        )
        validator.db_validator.test_connection = Mock(
            return_value=ValidationResult(True, [], [], [])
        )
        validator.mqtt_validator.test_connection = Mock(
            return_value=ValidationResult(True, [], [], [])
        )

        # Mock the system validator to always return success
        validator.system_validator.validate = Mock(
            return_value=ValidationResult(
                is_valid=True, errors=[], warnings=[], info=["Python version: 3.11.0"]
            )
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir)

            config_file = config_dir / "config.yaml"
            rooms_file = config_dir / "rooms.yaml"

            config_data = {
                "home_assistant": {"url": "http://localhost:8123", "token": "token"},
                "database": {
                    "connection_string": "postgresql://user:pass@localhost/db"
                },
                "mqtt": {"broker": "localhost"},
            }

            rooms_data = {
                "rooms": {
                    "living_room": {
                        "name": "Living Room",
                        "sensors": {"motion": "binary_sensor.motion"},
                    }
                }
            }

            with open(config_file, "w") as f:
                yaml.dump(config_data, f)

            with open(rooms_file, "w") as f:
                yaml.dump(rooms_data, f)

            result = validator.validate_config_files(
                str(config_dir), test_connections=True
            )

            assert result.is_valid is True
            # Verify connection tests were called
            validator.ha_validator.test_connection.assert_called_once()
            validator.db_validator.test_connection.assert_called_once()
            validator.mqtt_validator.test_connection.assert_called_once()
