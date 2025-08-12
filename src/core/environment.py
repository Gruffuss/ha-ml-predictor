"""
Environment-specific configuration management.
Handles environment detection, secrets management, and configuration validation.
"""

import base64
from dataclasses import dataclass
from enum import Enum
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from cryptography.fernet import Fernet
import yaml

logger = logging.getLogger(__name__)

# Global environment manager instance
_env_manager_instance: Optional["EnvironmentManager"] = None


class Environment(Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"

    @classmethod
    def from_string(cls, env_str: str) -> "Environment":
        """Convert string to Environment enum."""
        env_map = {
            "dev": cls.DEVELOPMENT,
            "development": cls.DEVELOPMENT,
            "test": cls.TESTING,
            "testing": cls.TESTING,
            "stage": cls.STAGING,
            "staging": cls.STAGING,
            "prod": cls.PRODUCTION,
            "production": cls.PRODUCTION,
        }
        return env_map.get(env_str.lower(), cls.DEVELOPMENT)


@dataclass
class SecretConfig:
    """Configuration for a secret value."""

    key: str
    required: bool = True
    default: Optional[str] = None
    description: str = ""
    env_var: Optional[str] = None
    encrypted: bool = False


@dataclass
class EnvironmentSettings:
    """Environment-specific configuration settings."""

    name: str
    debug: bool = False
    log_level: str = "INFO"
    database_pool_size: int = 10
    api_workers: int = 1
    redis_max_connections: int = 10
    monitoring_enabled: bool = False
    backup_enabled: bool = False
    secrets_encrypted: bool = False
    config_validation_strict: bool = True
    performance_monitoring: bool = False


class SecretsManager:
    """Manages encrypted secrets for different environments."""

    def __init__(self, secrets_dir: str = "secrets"):
        self.secrets_dir = Path(secrets_dir)
        self.secrets_dir.mkdir(exist_ok=True)
        self._encryption_key = self._get_or_create_key()
        self._cipher = Fernet(self._encryption_key)

    def _get_or_create_key(self) -> bytes:
        """Get or create encryption key for secrets."""
        key_file = self.secrets_dir / "master.key"

        if key_file.exists():
            key_data: bytes = base64.b64decode(key_file.read_text().strip())
            return key_data
        else:
            # Generate new key
            key: bytes = Fernet.generate_key()
            key_file.write_text(base64.b64encode(key).decode())
            key_file.chmod(0o600)  # Restrict permissions
            logger.info(f"Generated new encryption key: {key_file}")
            return key

    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value."""
        encrypted = self._cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()

    def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value."""
        encrypted_bytes = base64.b64decode(encrypted_value.encode())
        decrypted_data: str = self._cipher.decrypt(encrypted_bytes).decode()
        return decrypted_data

    def store_secret(
        self, environment: Environment, key: str, value: str, encrypt: bool = True
    ) -> None:
        """Store a secret for a specific environment."""
        env_secrets_file = self.secrets_dir / f"{environment.value}.json"

        # Load existing secrets
        secrets = {}
        if env_secrets_file.exists():
            secrets = json.loads(env_secrets_file.read_text())

        # Store the secret (encrypted or plain)
        secrets[key] = {
            "value": self.encrypt_secret(value) if encrypt else value,
            "encrypted": encrypt,
        }

        # Save back to file
        env_secrets_file.write_text(json.dumps(secrets, indent=2))
        env_secrets_file.chmod(0o600)
        logger.info(f"Stored secret '{key}' for {environment.value}")

    def get_secret(
        self, environment: Environment, key: str, default: Optional[str] = None
    ) -> Optional[str]:
        """Retrieve a secret for a specific environment."""
        env_secrets_file = self.secrets_dir / f"{environment.value}.json"

        if not env_secrets_file.exists():
            return default

        secrets = json.loads(env_secrets_file.read_text())
        secret_data = secrets.get(key)

        if not secret_data:
            return default

        value = secret_data.get("value")
        is_encrypted = secret_data.get("encrypted", False)

        if is_encrypted:
            try:
                return self.decrypt_secret(value)
            except Exception as e:
                logger.error(f"Failed to decrypt secret '{key}': {e}")
                return default
        return str(value) if value is not None else default

    def list_secrets(self, environment: Environment) -> List[str]:
        """List all secret keys for an environment."""
        env_secrets_file = self.secrets_dir / f"{environment.value}.json"

        if not env_secrets_file.exists():
            return []

        secrets = json.loads(env_secrets_file.read_text())
        return list(secrets.keys())

    def rotate_encryption_key(self) -> None:
        """Rotate the master encryption key and re-encrypt all secrets."""
        logger.warning("Starting encryption key rotation...")

        # Store old cipher
        old_cipher = self._cipher

        # Generate new key
        new_key = Fernet.generate_key()
        new_cipher = Fernet(new_key)

        # Re-encrypt all secrets for all environments
        for env in Environment:
            env_secrets_file = self.secrets_dir / f"{env.value}.json"

            if not env_secrets_file.exists():
                continue

            secrets = json.loads(env_secrets_file.read_text())

            for key, secret_data in secrets.items():
                if secret_data.get("encrypted", False):
                    try:
                        # Decrypt with old key
                        old_value = old_cipher.decrypt(
                            base64.b64decode(secret_data["value"])
                        ).decode()

                        # Encrypt with new key
                        new_encrypted = base64.b64encode(
                            new_cipher.encrypt(old_value.encode())
                        ).decode()

                        secrets[key]["value"] = new_encrypted
                    except Exception as e:
                        logger.error(
                            f"Failed to rotate key for secret '{key}' in {env.value}: {e}"
                        )

            # Save updated secrets
            env_secrets_file.write_text(json.dumps(secrets, indent=2))

        # Update master key
        key_file = self.secrets_dir / "master.key"
        key_file.write_text(base64.b64encode(new_key).decode())

        # Update instance
        self._encryption_key = new_key
        self._cipher = new_cipher

        logger.info("Encryption key rotation completed successfully")


class EnvironmentManager:
    """Manages environment-specific configuration and secrets."""

    # Define secrets required for each environment
    REQUIRED_SECRETS = {
        Environment.DEVELOPMENT: [
            SecretConfig(
                "ha_token", required=True, description="Home Assistant API token"
            ),
            SecretConfig(
                "database_password",
                required=False,
                default="dev_pass",
                description="Database password",
            ),
        ],
        Environment.TESTING: [
            SecretConfig(
                "ha_token",
                required=False,
                default="test_token",
                description="Home Assistant API token",
            ),
            SecretConfig(
                "database_password",
                required=False,
                default="test_pass",
                description="Database password",
            ),
        ],
        Environment.STAGING: [
            SecretConfig(
                "ha_token", required=True, description="Home Assistant API token"
            ),
            SecretConfig(
                "database_password", required=True, description="Database password"
            ),
            SecretConfig(
                "redis_password", required=False, description="Redis password"
            ),
        ],
        Environment.PRODUCTION: [
            SecretConfig(
                "ha_token", required=True, description="Home Assistant API token"
            ),
            SecretConfig(
                "database_password", required=True, description="Database password"
            ),
            SecretConfig("redis_password", required=True, description="Redis password"),
            SecretConfig(
                "api_secret_key", required=True, description="API secret key for JWT"
            ),
            SecretConfig(
                "grafana_password", required=False, description="Grafana admin password"
            ),
        ],
    }

    # Environment-specific settings
    ENVIRONMENT_SETTINGS = {
        Environment.DEVELOPMENT: EnvironmentSettings(
            name="development",
            debug=True,
            log_level="DEBUG",
            database_pool_size=5,
            api_workers=1,
            monitoring_enabled=False,
            backup_enabled=False,
            secrets_encrypted=False,
            config_validation_strict=False,
        ),
        Environment.TESTING: EnvironmentSettings(
            name="testing",
            debug=True,
            log_level="DEBUG",
            database_pool_size=2,
            api_workers=1,
            monitoring_enabled=False,
            backup_enabled=False,
            secrets_encrypted=False,
            config_validation_strict=True,
        ),
        Environment.STAGING: EnvironmentSettings(
            name="staging",
            debug=False,
            log_level="INFO",
            database_pool_size=10,
            api_workers=2,
            monitoring_enabled=True,
            backup_enabled=True,
            secrets_encrypted=True,
            config_validation_strict=True,
            performance_monitoring=True,
        ),
        Environment.PRODUCTION: EnvironmentSettings(
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
        ),
    }

    def __init__(self, config_dir: str = "config", secrets_dir: str = "secrets"):
        self.config_dir = Path(config_dir)
        self.secrets_manager = SecretsManager(secrets_dir)
        self.current_environment = self._detect_environment()

        logger.info(
            f"Environment manager initialized for: {self.current_environment.value}"
        )

    def _detect_environment(self) -> Environment:
        """Detect current environment from various sources."""
        # Check environment variables (highest priority)
        env_var = os.getenv("ENVIRONMENT", os.getenv("ENV", ""))
        if env_var:
            return Environment.from_string(env_var)

        # Check for environment-specific config files
        if (self.config_dir / "config.prod.yaml").exists():
            return Environment.PRODUCTION
        elif (self.config_dir / "config.staging.yaml").exists():
            return Environment.STAGING
        elif (self.config_dir / "config.test.yaml").exists():
            return Environment.TESTING

        # Check Docker environment
        if os.path.exists("/.dockerenv"):
            return Environment.PRODUCTION

        # Default to development
        return Environment.DEVELOPMENT

    def get_environment_settings(self) -> EnvironmentSettings:
        """Get settings for current environment."""
        return self.ENVIRONMENT_SETTINGS[self.current_environment]

    def get_config_file_path(self, base_name: str = "config") -> Path:
        """Get environment-specific config file path."""
        env_name = self.current_environment.value
        env_file = self.config_dir / f"{base_name}.{env_name}.yaml"

        if env_file.exists():
            return env_file
        else:
            return self.config_dir / f"{base_name}.yaml"

    def load_environment_config(self) -> Dict[str, Any]:
        """Load environment-specific configuration."""
        config_file = self.get_config_file_path()

        if not config_file.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_file}")

        with open(config_file, "r") as f:
            loaded_config = yaml.safe_load(f)
            config = loaded_config if isinstance(loaded_config, dict) else {}

        # Apply environment-specific overrides
        self._apply_environment_overrides(config)

        # Inject secrets
        self._inject_secrets(config)

        return config

    def _apply_environment_overrides(self, config: Dict[str, Any]) -> None:
        """Apply environment-specific configuration overrides."""
        settings = self.get_environment_settings()

        # Override logging configuration
        if "logging" not in config:
            config["logging"] = {}
        config["logging"]["level"] = settings.log_level

        # Override database configuration
        if "database" not in config:
            config["database"] = {}
        config["database"]["pool_size"] = settings.database_pool_size

        # Override API configuration
        if "api" not in config:
            config["api"] = {}
        config["api"]["debug"] = settings.debug

        # Add environment-specific settings
        config["environment"] = {
            "name": settings.name,
            "monitoring_enabled": settings.monitoring_enabled,
            "backup_enabled": settings.backup_enabled,
            "performance_monitoring": settings.performance_monitoring,
        }

    def _inject_secrets(self, config: Dict[str, Any]) -> None:
        """Inject secrets from secrets manager into configuration."""
        # Home Assistant token
        ha_token = self.get_secret("ha_token")
        if ha_token:
            if "home_assistant" not in config:
                config["home_assistant"] = {}
            config["home_assistant"]["token"] = ha_token

        # Database password
        db_password = self.get_secret("database_password")
        if db_password:
            if "database" not in config:
                config["database"] = {}
            # Update connection string with password
            conn_str = config["database"].get("connection_string", "")
            if "password=" not in conn_str and db_password:
                config["database"]["connection_string"] = conn_str.replace(
                    "occupancy_user@", f"occupancy_user:{db_password}@"
                )

        # Redis password
        redis_password = self.get_secret("redis_password")
        if redis_password:
            config["redis"] = config.get("redis", {})
            config["redis"]["password"] = redis_password

        # API secret key
        api_secret = self.get_secret("api_secret_key")
        if api_secret:
            if "api" not in config:
                config["api"] = {}
            config["api"]["secret_key"] = api_secret

    def get_secret(self, key: str, default: Optional[str] = None) -> Optional[str]:
        """Get a secret value, checking environment variables first."""
        # Check environment variable first
        env_var = os.getenv(key.upper())
        if env_var:
            return env_var

        # Check secrets manager
        return self.secrets_manager.get_secret(self.current_environment, key, default)

    def set_secret(self, key: str, value: str, encrypt: Optional[bool] = None) -> None:
        """Set a secret value for the current environment."""
        if encrypt is None:
            settings = self.get_environment_settings()
            encrypt = settings.secrets_encrypted

        self.secrets_manager.store_secret(self.current_environment, key, value, encrypt)

    def validate_configuration(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration for current environment."""
        errors = []
        required_secrets = self.REQUIRED_SECRETS.get(self.current_environment, [])

        # Check required secrets
        for secret_config in required_secrets:
            if secret_config.required:
                value = self.get_secret(secret_config.key, secret_config.default)
                if not value:
                    errors.append(
                        f"Missing required secret: {secret_config.key} ({secret_config.description})"
                    )

        # Environment-specific validation
        if self.current_environment == Environment.PRODUCTION:
            errors.extend(self._validate_production_config(config))
        elif self.current_environment == Environment.STAGING:
            errors.extend(self._validate_staging_config(config))

        return errors

    def _validate_production_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate production-specific configuration."""
        errors = []

        # Check that debug is disabled
        if config.get("api", {}).get("debug", False):
            errors.append("Debug mode must be disabled in production")

        # Check that monitoring is enabled
        if not config.get("environment", {}).get("monitoring_enabled", False):
            errors.append("Monitoring must be enabled in production")

        # Check that backups are enabled
        if not config.get("environment", {}).get("backup_enabled", False):
            errors.append("Backups must be enabled in production")

        # Check log level
        log_level = config.get("logging", {}).get("level", "INFO")
        if log_level in ["DEBUG", "TRACE"]:
            errors.append("Log level too verbose for production")

        return errors

    def _validate_staging_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate staging-specific configuration."""
        errors = []

        # Check that monitoring is enabled
        if not config.get("environment", {}).get("monitoring_enabled", False):
            errors.append("Monitoring should be enabled in staging")

        return errors

    def setup_environment_secrets(self) -> None:
        """Interactive setup for environment secrets."""
        required_secrets = self.REQUIRED_SECRETS.get(self.current_environment, [])
        settings = self.get_environment_settings()

        print(f"\nSetting up secrets for {self.current_environment.value} environment")
        print("=" * 60)

        for secret_config in required_secrets:
            current_value = self.get_secret(secret_config.key)

            if current_value and not secret_config.required:
                print(f"✓ {secret_config.key}: Already configured")
                continue

            print(f"\n{secret_config.description}")
            print(f"Key: {secret_config.key}")
            print(f"Required: {'Yes' if secret_config.required else 'No'}")

            if current_value:
                print("Current value: ****** (hidden)")
                update = input("Update? (y/n): ").lower().strip() == "y"
                if not update:
                    continue

            if secret_config.default and not secret_config.required:
                value = input(f"Value (default: {secret_config.default}): ").strip()
                if not value:
                    value = secret_config.default
            else:
                value = input("Value: ").strip()

            if value or not secret_config.required:
                self.set_secret(
                    secret_config.key,
                    value or secret_config.default or "",
                    encrypt=settings.secrets_encrypted,
                )
                print(f"✓ Secret '{secret_config.key}' configured")

        print(
            f"\n✓ Environment secrets setup completed for {self.current_environment.value}"
        )

    def export_environment_template(
        self, target_environment: Environment, output_file: str
    ) -> None:
        """Export environment configuration template."""
        template: Dict[str, Any] = {
            "environment": target_environment.value,
            "settings": self.ENVIRONMENT_SETTINGS[target_environment].__dict__.copy(),
            "required_secrets": [],
        }

        # Add required secrets (without values)
        for secret_config in self.REQUIRED_SECRETS.get(target_environment, []):
            required_secrets = template["required_secrets"]
            assert isinstance(required_secrets, list)
            required_secrets.append(
                {
                    "key": secret_config.key,
                    "required": secret_config.required,
                    "description": secret_config.description,
                    "env_var": secret_config.env_var or secret_config.key.upper(),
                }
            )

        output_path = Path(output_file)
        output_path.write_text(yaml.dump(template, default_flow_style=False, indent=2))
        logger.info(f"Environment template exported: {output_path}")


def get_environment_manager() -> EnvironmentManager:
    """Get global environment manager instance."""
    global _env_manager_instance
    if _env_manager_instance is None:
        _env_manager_instance = EnvironmentManager()
    return _env_manager_instance
