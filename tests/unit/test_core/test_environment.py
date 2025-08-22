"""
Comprehensive unit tests for environment.py.
Tests environment management, secrets encryption, and configuration validation.
"""

import base64
import json
import os
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch

import pytest
import yaml

from src.core.environment import (
    Environment,
    EnvironmentManager,
    EnvironmentSettings,
    SecretConfig,
    SecretsManager,
    get_environment_manager,
)


class TestEnvironment:
    """Test Environment enum functionality."""

    def test_environment_values(self):
        """Test all environment enum values."""
        assert Environment.DEVELOPMENT.value == "development"
        assert Environment.TESTING.value == "testing"
        assert Environment.STAGING.value == "staging"
        assert Environment.PRODUCTION.value == "production"

    def test_from_string_exact_matches(self):
        """Test from_string with exact matches."""
        assert Environment.from_string("development") == Environment.DEVELOPMENT
        assert Environment.from_string("testing") == Environment.TESTING
        assert Environment.from_string("staging") == Environment.STAGING
        assert Environment.from_string("production") == Environment.PRODUCTION

    def test_from_string_abbreviations(self):
        """Test from_string with common abbreviations."""
        assert Environment.from_string("dev") == Environment.DEVELOPMENT
        assert Environment.from_string("test") == Environment.TESTING
        assert Environment.from_string("stage") == Environment.STAGING
        assert Environment.from_string("prod") == Environment.PRODUCTION

    def test_from_string_case_insensitive(self):
        """Test from_string is case insensitive."""
        assert Environment.from_string("DEVELOPMENT") == Environment.DEVELOPMENT
        assert Environment.from_string("Dev") == Environment.DEVELOPMENT
        assert Environment.from_string("PROD") == Environment.PRODUCTION

    def test_from_string_unknown_defaults_to_development(self):
        """Test from_string defaults to development for unknown values."""
        assert Environment.from_string("unknown") == Environment.DEVELOPMENT
        assert Environment.from_string("") == Environment.DEVELOPMENT
        assert Environment.from_string("invalid") == Environment.DEVELOPMENT


class TestSecretConfig:
    """Test SecretConfig dataclass functionality."""

    def test_secret_config_creation(self):
        """Test creating SecretConfig with all fields."""
        config = SecretConfig(
            key="test_secret",
            required=True,
            default="default_value",
            description="Test secret for testing",
            env_var="TEST_SECRET",
            encrypted=True,
        )

        assert config.key == "test_secret"
        assert config.required is True
        assert config.default == "default_value"
        assert config.description == "Test secret for testing"
        assert config.env_var == "TEST_SECRET"
        assert config.encrypted is True

    def test_secret_config_defaults(self):
        """Test SecretConfig with default values."""
        config = SecretConfig(key="simple_secret")

        assert config.key == "simple_secret"
        assert config.required is True
        assert config.default is None
        assert config.description == ""
        assert config.env_var is None
        assert config.encrypted is False


class TestEnvironmentSettings:
    """Test EnvironmentSettings dataclass functionality."""

    def test_environment_settings_creation(self):
        """Test creating EnvironmentSettings with all fields."""
        settings = EnvironmentSettings(
            name="production",
            debug=False,
            log_level="WARNING",
            database_pool_size=20,
            api_workers=4,
            redis_max_connections=50,
            monitoring_enabled=True,
            backup_enabled=True,
            secrets_encrypted=True,
            config_validation_strict=True,
            performance_monitoring=True,
        )

        assert settings.name == "production"
        assert settings.debug is False
        assert settings.log_level == "WARNING"
        assert settings.database_pool_size == 20
        assert settings.api_workers == 4
        assert settings.redis_max_connections == 50
        assert settings.monitoring_enabled is True
        assert settings.backup_enabled is True
        assert settings.secrets_encrypted is True
        assert settings.config_validation_strict is True
        assert settings.performance_monitoring is True

    def test_environment_settings_defaults(self):
        """Test EnvironmentSettings with default values."""
        settings = EnvironmentSettings(name="test")

        assert settings.name == "test"
        assert settings.debug is False
        assert settings.log_level == "INFO"
        assert settings.database_pool_size == 10
        assert settings.api_workers == 1
        assert settings.redis_max_connections == 10
        assert settings.monitoring_enabled is False
        assert settings.backup_enabled is False
        assert settings.secrets_encrypted is False
        assert settings.config_validation_strict is True
        assert settings.performance_monitoring is False


class TestSecretsManager:
    """Test SecretsManager functionality."""

    @pytest.fixture
    def temp_secrets_dir(self):
        """Create temporary secrets directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def secrets_manager(self, temp_secrets_dir):
        """Create SecretsManager instance."""
        return SecretsManager(temp_secrets_dir)

    def test_init_creates_secrets_directory(self, temp_secrets_dir):
        """Test that initialization creates secrets directory."""
        manager = SecretsManager(temp_secrets_dir)
        assert Path(temp_secrets_dir).exists()
        assert manager.secrets_dir == Path(temp_secrets_dir)

    def test_get_or_create_key_creates_new_key(self, temp_secrets_dir):
        """Test that new encryption key is created if none exists."""
        manager = SecretsManager(temp_secrets_dir)
        key_file = Path(temp_secrets_dir) / "master.key"

        assert key_file.exists()
        key_content = key_file.read_text().strip()

        # Verify it's valid base64
        decoded_key = base64.b64decode(key_content)
        assert len(decoded_key) == 32  # Fernet keys are 32 bytes

    def test_get_or_create_key_loads_existing_key(self, temp_secrets_dir):
        """Test that existing encryption key is loaded."""
        # Create a known key
        from cryptography.fernet import Fernet

        known_key = Fernet.generate_key()
        key_file = Path(temp_secrets_dir) / "master.key"
        key_file.write_text(base64.b64encode(known_key).decode())

        manager = SecretsManager(temp_secrets_dir)

        # Verify the same key is loaded
        assert manager._encryption_key == known_key

    def test_encrypt_decrypt_secret(self, secrets_manager):
        """Test encrypting and decrypting secrets."""
        original_value = "test_secret_value_123"

        encrypted = secrets_manager.encrypt_secret(original_value)
        decrypted = secrets_manager.decrypt_secret(encrypted)

        assert decrypted == original_value
        assert encrypted != original_value  # Should be different when encrypted
        assert len(encrypted) > len(original_value)  # Encrypted should be longer

    def test_store_secret_encrypted(self, secrets_manager, temp_secrets_dir):
        """Test storing encrypted secret."""
        secrets_manager.store_secret(
            Environment.DEVELOPMENT, "test_key", "test_value", encrypt=True
        )

        secrets_file = Path(temp_secrets_dir) / "development.json"
        assert secrets_file.exists()

        with open(secrets_file) as f:
            data = json.load(f)

        assert "test_key" in data
        assert data["test_key"]["encrypted"] is True
        assert data["test_key"]["value"] != "test_value"  # Should be encrypted

    def test_store_secret_unencrypted(self, secrets_manager, temp_secrets_dir):
        """Test storing unencrypted secret."""
        secrets_manager.store_secret(
            Environment.DEVELOPMENT, "test_key", "test_value", encrypt=False
        )

        secrets_file = Path(temp_secrets_dir) / "development.json"
        with open(secrets_file) as f:
            data = json.load(f)

        assert data["test_key"]["encrypted"] is False
        assert data["test_key"]["value"] == "test_value"  # Should be plain

    def test_get_secret_encrypted(self, secrets_manager):
        """Test retrieving encrypted secret."""
        original_value = "encrypted_secret_123"

        secrets_manager.store_secret(
            Environment.PRODUCTION, "encrypted_key", original_value, encrypt=True
        )

        retrieved_value = secrets_manager.get_secret(
            Environment.PRODUCTION, "encrypted_key"
        )

        assert retrieved_value == original_value

    def test_get_secret_unencrypted(self, secrets_manager):
        """Test retrieving unencrypted secret."""
        original_value = "plain_secret_123"

        secrets_manager.store_secret(
            Environment.TESTING, "plain_key", original_value, encrypt=False
        )

        retrieved_value = secrets_manager.get_secret(Environment.TESTING, "plain_key")

        assert retrieved_value == original_value

    def test_get_secret_not_found_returns_default(self, secrets_manager):
        """Test that get_secret returns default for non-existent secret."""
        result = secrets_manager.get_secret(
            Environment.DEVELOPMENT, "nonexistent", default="default_value"
        )

        assert result == "default_value"

    def test_get_secret_no_secrets_file_returns_default(self, secrets_manager):
        """Test that get_secret returns default when no secrets file exists."""
        result = secrets_manager.get_secret(
            Environment.STAGING, "any_key", default="default_value"
        )

        assert result == "default_value"

    def test_list_secrets(self, secrets_manager):
        """Test listing secrets for an environment."""
        secrets_manager.store_secret(Environment.DEVELOPMENT, "secret1", "value1")
        secrets_manager.store_secret(Environment.DEVELOPMENT, "secret2", "value2")
        secrets_manager.store_secret(Environment.PRODUCTION, "secret3", "value3")

        dev_secrets = secrets_manager.list_secrets(Environment.DEVELOPMENT)
        prod_secrets = secrets_manager.list_secrets(Environment.PRODUCTION)

        assert set(dev_secrets) == {"secret1", "secret2"}
        assert prod_secrets == ["secret3"]

    def test_list_secrets_empty_environment(self, secrets_manager):
        """Test listing secrets for environment with no secrets."""
        secrets = secrets_manager.list_secrets(Environment.STAGING)
        assert secrets == []

    def test_rotate_encryption_key(self, secrets_manager):
        """Test encryption key rotation."""
        # Store some secrets
        secrets_manager.store_secret(
            Environment.DEVELOPMENT, "secret1", "value1", encrypt=True
        )
        secrets_manager.store_secret(
            Environment.PRODUCTION, "secret2", "value2", encrypt=True
        )

        # Get values before rotation
        value1_before = secrets_manager.get_secret(Environment.DEVELOPMENT, "secret1")
        value2_before = secrets_manager.get_secret(Environment.PRODUCTION, "secret2")

        # Rotate key
        secrets_manager.rotate_encryption_key()

        # Verify values are still accessible after rotation
        value1_after = secrets_manager.get_secret(Environment.DEVELOPMENT, "secret1")
        value2_after = secrets_manager.get_secret(Environment.PRODUCTION, "secret2")

        assert value1_before == value1_after
        assert value2_before == value2_after

    def test_get_secret_decrypt_failure_returns_default(
        self, secrets_manager, temp_secrets_dir
    ):
        """Test that decryption failure returns default value."""
        # Manually create corrupted encrypted secret
        secrets_file = Path(temp_secrets_dir) / "development.json"
        data = {
            "corrupted_key": {"value": "not_valid_encrypted_data", "encrypted": True}
        }
        with open(secrets_file, "w") as f:
            json.dump(data, f)

        result = secrets_manager.get_secret(
            Environment.DEVELOPMENT, "corrupted_key", default="fallback"
        )

        assert result == "fallback"


class TestEnvironmentManager:
    """Test EnvironmentManager functionality."""

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def temp_secrets_dir(self):
        """Create temporary secrets directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def env_manager(self, temp_config_dir, temp_secrets_dir):
        """Create EnvironmentManager instance."""
        return EnvironmentManager(temp_config_dir, temp_secrets_dir)

    def test_init_creates_managers(self, env_manager):
        """Test that initialization creates secrets manager."""
        assert env_manager.secrets_manager is not None
        assert env_manager.config_dir.exists()

    @patch.dict(os.environ, {"ENVIRONMENT": "production"}, clear=False)
    def test_detect_environment_from_env_var(self, temp_config_dir, temp_secrets_dir):
        """Test environment detection from environment variable."""
        manager = EnvironmentManager(temp_config_dir, temp_secrets_dir)
        assert manager.current_environment == Environment.PRODUCTION

    @patch.dict(os.environ, {"ENV": "staging"}, clear=False)
    def test_detect_environment_from_env_var_alt(
        self, temp_config_dir, temp_secrets_dir
    ):
        """Test environment detection from alternative environment variable."""
        # Clear ENVIRONMENT if it exists
        if "ENVIRONMENT" in os.environ:
            del os.environ["ENVIRONMENT"]

        manager = EnvironmentManager(temp_config_dir, temp_secrets_dir)
        assert manager.current_environment == Environment.STAGING

    def test_detect_environment_from_config_files(
        self, temp_config_dir, temp_secrets_dir
    ):
        """Test environment detection from config files."""
        # Create production config file
        prod_config = Path(temp_config_dir) / "config.prod.yaml"
        prod_config.write_text("production: config")

        with patch.dict(os.environ, {}, clear=True):  # Clear env vars
            manager = EnvironmentManager(temp_config_dir, temp_secrets_dir)
            assert manager.current_environment == Environment.PRODUCTION

    def test_detect_environment_from_docker(self, temp_config_dir, temp_secrets_dir):
        """Test environment detection from Docker container."""
        with patch("os.path.exists") as mock_exists, patch.dict(
            os.environ, {}, clear=True
        ):

            mock_exists.return_value = True  # /.dockerenv exists

            manager = EnvironmentManager(temp_config_dir, temp_secrets_dir)
            assert manager.current_environment == Environment.PRODUCTION

    def test_detect_environment_defaults_to_development(
        self, temp_config_dir, temp_secrets_dir
    ):
        """Test environment detection defaults to development."""
        with patch.dict(os.environ, {}, clear=True), patch(
            "os.path.exists", return_value=False
        ):

            manager = EnvironmentManager(temp_config_dir, temp_secrets_dir)
            assert manager.current_environment == Environment.DEVELOPMENT

    def test_get_environment_settings(self, env_manager):
        """Test getting environment settings."""
        settings = env_manager.get_environment_settings()

        assert isinstance(settings, EnvironmentSettings)
        assert settings.name == env_manager.current_environment.value

    def test_get_config_file_path_environment_specific(self, env_manager):
        """Test getting environment-specific config file path."""
        # Create environment-specific config
        env_name = env_manager.current_environment.value
        env_config = env_manager.config_dir / f"config.{env_name}.yaml"
        env_config.write_text("env_specific: true")

        path = env_manager.get_config_file_path()
        assert path == env_config

    def test_get_config_file_path_fallback(self, env_manager):
        """Test getting config file path falls back to default."""
        # Create only default config
        default_config = env_manager.config_dir / "config.yaml"
        default_config.write_text("default: true")

        path = env_manager.get_config_file_path()
        assert path == default_config

    def test_load_environment_config(self, env_manager):
        """Test loading environment configuration."""
        # Create config file
        config_content = {
            "home_assistant": {"url": "http://localhost:8123"},
            "database": {"connection_string": "postgresql://user@localhost/db"},
        }

        config_file = env_manager.get_config_file_path()
        with open(config_file, "w") as f:
            yaml.dump(config_content, f)

        loaded_config = env_manager.load_environment_config()

        # Should have original config plus environment overrides
        assert "home_assistant" in loaded_config
        assert "database" in loaded_config
        assert "logging" in loaded_config  # Added by environment overrides
        assert "environment" in loaded_config  # Added by environment overrides

    def test_load_environment_config_missing_file(self, env_manager):
        """Test loading config when file doesn't exist."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            env_manager.load_environment_config()

    def test_apply_environment_overrides(self, env_manager):
        """Test application of environment-specific overrides."""
        config = {}
        env_manager._apply_environment_overrides(config)

        settings = env_manager.get_environment_settings()

        assert config["logging"]["level"] == settings.log_level
        assert config["database"]["pool_size"] == settings.database_pool_size
        assert config["api"]["debug"] == settings.debug
        assert config["environment"]["name"] == settings.name

    def test_inject_secrets(self, env_manager):
        """Test injection of secrets into configuration."""
        # Store some secrets
        env_manager.set_secret("ha_token", "test_ha_token")
        env_manager.set_secret("database_password", "test_db_pass")
        env_manager.set_secret("api_secret_key", "test_api_secret")

        config = {
            "home_assistant": {},
            "database": {"connection_string": "postgresql://user@localhost/db"},
            "api": {},
        }

        env_manager._inject_secrets(config)

        assert config["home_assistant"]["token"] == "test_ha_token"
        assert "user:test_db_pass@" in config["database"]["connection_string"]
        assert config["api"]["secret_key"] == "test_api_secret"

    def test_get_secret_from_env_var_first(self, env_manager):
        """Test that get_secret checks environment variables first."""
        # Store secret in secrets manager
        env_manager.set_secret("test_secret", "from_secrets_manager")

        # Mock environment variable
        with patch.dict(os.environ, {"TEST_SECRET": "from_env_var"}):
            result = env_manager.get_secret("test_secret")
            assert result == "from_env_var"

    def test_get_secret_from_secrets_manager(self, env_manager):
        """Test getting secret from secrets manager when no env var."""
        env_manager.set_secret("test_secret", "from_secrets_manager")

        with patch.dict(os.environ, {}, clear=True):
            result = env_manager.get_secret("test_secret")
            assert result == "from_secrets_manager"

    def test_set_secret_encryption_auto_detect(self, env_manager):
        """Test that set_secret auto-detects encryption based on environment."""
        settings = env_manager.get_environment_settings()
        expected_encryption = settings.secrets_encrypted

        env_manager.set_secret("auto_encrypt_test", "test_value")

        # Verify secret was stored with correct encryption setting
        stored_value = env_manager.secrets_manager.get_secret(
            env_manager.current_environment, "auto_encrypt_test"
        )
        assert stored_value == "test_value"

    def test_validate_configuration_structure(self, env_manager):
        """Test basic configuration structure validation."""
        valid_config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "valid_token"},
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
        }

        errors = env_manager.validate_configuration(valid_config)
        assert len(errors) == 0

    def test_validate_configuration_missing_fields(self, env_manager):
        """Test configuration validation with missing required fields."""
        invalid_config = {
            "home_assistant": {},  # Missing URL and token
            "database": {},  # Missing connection string
            "mqtt": {},  # Missing broker
        }

        errors = env_manager.validate_configuration(invalid_config)
        assert len(errors) > 0
        assert any("Home Assistant URL is required" in error for error in errors)
        assert any(
            "Database connection string is required" in error for error in errors
        )

    def test_validate_production_config(self, temp_config_dir, temp_secrets_dir):
        """Test production-specific configuration validation."""
        # Create production environment manager
        with patch.dict(os.environ, {"ENVIRONMENT": "production"}):
            manager = EnvironmentManager(temp_config_dir, temp_secrets_dir)

        config = {
            "home_assistant": {"url": "http://localhost:8123", "token": "token"},
            "database": {"connection_string": "postgresql://user:pass@localhost/db"},
            "mqtt": {"broker": "localhost"},
            "api": {"debug": True},  # Should cause error in production
            "logging": {"level": "DEBUG"},  # Should cause error in production
            "environment": {
                "monitoring_enabled": False,
                "backup_enabled": False,
            },  # Should cause errors
        }

        errors = manager.validate_configuration(config)
        assert len(errors) > 0
        assert any(
            "Debug mode must be disabled in production" in error for error in errors
        )
        assert any("Log level too verbose for production" in error for error in errors)
        assert any(
            "Monitoring must be enabled in production" in error for error in errors
        )
        assert any("Backups must be enabled in production" in error for error in errors)

    def test_export_environment_template(self, env_manager):
        """Test exporting environment configuration template."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            template_file = f.name

        try:
            env_manager.export_environment_template(
                Environment.PRODUCTION, template_file
            )

            with open(template_file) as f:
                template = yaml.safe_load(f)

            assert template["environment"] == "production"
            assert "settings" in template
            assert "required_secrets" in template
            assert len(template["required_secrets"]) > 0

            # Check that production secrets are included
            secret_keys = [s["key"] for s in template["required_secrets"]]
            assert "ha_token" in secret_keys
            assert "database_password" in secret_keys
            assert "api_secret_key" in secret_keys

        finally:
            Path(template_file).unlink()

    @patch("builtins.input")
    @patch("builtins.print")
    def test_setup_environment_secrets_interactive(
        self, mock_print, mock_input, env_manager
    ):
        """Test interactive secrets setup."""
        # Mock user input
        mock_input.side_effect = ["test_token_value", "test_db_password"]

        env_manager.setup_environment_secrets()

        # Verify secrets were stored
        ha_token = env_manager.get_secret("ha_token")
        db_password = env_manager.get_secret("database_password")

        assert ha_token == "test_token_value"
        assert db_password == "test_db_password"

    @patch("builtins.input")
    @patch("builtins.print")
    def test_setup_environment_secrets_skip_existing(
        self, mock_print, mock_input, env_manager
    ):
        """Test skipping existing secrets in interactive setup."""
        # Pre-store a secret
        env_manager.set_secret("ha_token", "existing_token")

        # Mock user choosing not to update
        mock_input.side_effect = ["n", "new_db_password"]

        env_manager.setup_environment_secrets()

        # Verify existing secret wasn't changed
        ha_token = env_manager.get_secret("ha_token")
        assert ha_token == "existing_token"


class TestGlobalEnvironmentManager:
    """Test global environment manager functionality."""

    def test_get_environment_manager_singleton(self):
        """Test that get_environment_manager returns singleton instance."""
        manager1 = get_environment_manager()
        manager2 = get_environment_manager()

        assert manager1 is manager2
        assert isinstance(manager1, EnvironmentManager)

    @patch("src.core.environment._env_manager_instance", None)
    def test_get_environment_manager_creates_new_instance(self):
        """Test that get_environment_manager creates new instance when none exists."""
        # Reset global instance
        import src.core.environment

        src.core.environment._env_manager_instance = None

        manager = get_environment_manager()
        assert isinstance(manager, EnvironmentManager)


if __name__ == "__main__":
    pytest.main([__file__])
