#!/usr/bin/env python3
"""
Environment Management Script
Manages environment setup, configuration validation, secrets, and deployment.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from typing import Optional, List

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.environment import EnvironmentManager, Environment
from core.config_validator import ConfigurationValidator
from core.backup_manager import BackupManager

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def setup_environment(env_name: str) -> None:
    """Set up environment-specific configuration and secrets."""
    print(f"\n🚀 Setting up {env_name} environment")
    print("=" * 50)

    try:
        # Initialize environment manager
        env_manager = EnvironmentManager()
        target_env = Environment.from_string(env_name)

        # Override current environment for setup
        env_manager.current_environment = target_env

        # Interactive secrets setup
        env_manager.setup_environment_secrets()

        # Validate configuration
        print(f"\n🔍 Validating {env_name} configuration...")
        validator = ConfigurationValidator()
        result = validator.validate_config_files(
            environment=env_name,
            test_connections=False,  # Don't test connections during setup
        )

        print(result)

        if result.is_valid:
            print(f"✅ {env_name.title()} environment setup completed successfully!")
        else:
            print(f"❌ {env_name.title()} environment setup has configuration errors")
            return False

        return True

    except Exception as e:
        logger.error(f"Environment setup failed: {e}")
        return False


def validate_environment(env_name: str, test_connections: bool = False) -> bool:
    """Validate environment configuration."""
    print(f"\n🔍 Validating {env_name} environment configuration")
    print("=" * 50)

    try:
        validator = ConfigurationValidator()
        result = validator.validate_config_files(
            environment=env_name, test_connections=test_connections
        )

        print(result)
        return result.is_valid

    except Exception as e:
        logger.error(f"Configuration validation failed: {e}")
        return False


def list_environments() -> None:
    """List all available environments and their configurations."""
    print("\n📋 Available Environments")
    print("=" * 30)

    config_dir = Path("config")
    base_config = config_dir / "config.yaml"

    environments = []

    # Check for environment-specific configs
    for env in Environment:
        env_config = config_dir / f"config.{env.value}.yaml"
        if env_config.exists():
            environments.append((env.value, "✅ Configured", str(env_config)))
        else:
            if env == Environment.DEVELOPMENT and base_config.exists():
                environments.append(
                    (env.value, "📄 Uses base config", str(base_config))
                )
            else:
                environments.append((env.value, "❌ Not configured", "N/A"))

    # Print table
    print(f"{'Environment':<12} {'Status':<20} {'Config File'}")
    print("-" * 60)
    for env_name, status, config_file in environments:
        print(f"{env_name:<12} {status:<20} {config_file}")


def manage_secrets(
    env_name: str, action: str, key: Optional[str] = None, value: Optional[str] = None
) -> None:
    """Manage environment secrets."""
    print(f"\n🔐 Managing secrets for {env_name} environment")
    print("=" * 50)

    try:
        env_manager = EnvironmentManager()
        target_env = Environment.from_string(env_name)
        env_manager.current_environment = target_env

        if action == "list":
            secrets = env_manager.secrets_manager.list_secrets(target_env)
            if secrets:
                print("Configured secrets:")
                for secret in secrets:
                    print(f"  🔑 {secret}")
            else:
                print("No secrets configured for this environment")

        elif action == "set":
            if not key or not value:
                print("❌ Key and value are required for 'set' action")
                return

            env_manager.set_secret(key, value)
            print(f"✅ Secret '{key}' set for {env_name} environment")

        elif action == "get":
            if not key:
                print("❌ Key is required for 'get' action")
                return

            secret_value = env_manager.get_secret(key)
            if secret_value:
                print(f"🔑 {key}: {secret_value}")
            else:
                print(f"❌ Secret '{key}' not found")

        elif action == "setup":
            env_manager.setup_environment_secrets()

        else:
            print(f"❌ Unknown action: {action}")
            print("Available actions: list, set, get, setup")

    except Exception as e:
        logger.error(f"Secret management failed: {e}")


def backup_environment(env_name: str, backup_type: str = "all") -> None:
    """Create environment backup."""
    print(f"\n💾 Creating {backup_type} backup for {env_name} environment")
    print("=" * 50)

    try:
        # Load environment configuration
        env_manager = EnvironmentManager()
        target_env = Environment.from_string(env_name)
        env_manager.current_environment = target_env

        config = env_manager.load_environment_config()

        # Initialize backup manager
        backup_manager = BackupManager("backups", config)

        if backup_type == "database" or backup_type == "all":
            print("📊 Creating database backup...")
            db_backup = backup_manager.db_backup_manager.create_backup()
            print(f"✅ Database backup created: {db_backup.backup_id}")

        if backup_type == "models" or backup_type == "all":
            print("🤖 Creating models backup...")
            models_backup = backup_manager.model_backup_manager.create_backup()
            print(f"✅ Models backup created: {models_backup.backup_id}")

        if backup_type == "config" or backup_type == "all":
            print("⚙️  Creating configuration backup...")
            config_backup = backup_manager.config_backup_manager.create_backup()
            print(f"✅ Configuration backup created: {config_backup.backup_id}")

        if backup_type == "disaster-recovery":
            print("🚨 Creating disaster recovery package...")
            package_id = backup_manager.create_disaster_recovery_package()
            print(f"✅ Disaster recovery package created: {package_id}")

    except Exception as e:
        logger.error(f"Backup failed: {e}")


def deploy_environment(env_name: str, start_services: bool = True) -> None:
    """Deploy environment using Docker Compose."""
    print(f"\n🚀 Deploying {env_name} environment")
    print("=" * 50)

    try:
        # Validate configuration first
        print("🔍 Validating configuration before deployment...")
        if not validate_environment(env_name, test_connections=False):
            print("❌ Configuration validation failed. Aborting deployment.")
            return

        # Prepare Docker Compose command
        compose_files = ["docker-compose.yml"]

        if env_name == "production":
            compose_files.append("docker-compose.prod.yml")
        elif env_name == "staging":
            compose_files.append("docker-compose.staging.yml")
        elif env_name == "development":
            compose_files.append("docker-compose.development.yml")

        # Change to docker directory
        docker_dir = Path("docker")
        os.chdir(docker_dir)

        # Build compose command
        compose_cmd = ["docker-compose"]
        for compose_file in compose_files:
            compose_cmd.extend(["-f", compose_file])

        if start_services:
            compose_cmd.extend(["up", "-d", "--build"])
            print(f"🐳 Starting {env_name} services...")
        else:
            compose_cmd.extend(["build"])
            print(f"🔨 Building {env_name} images...")

        # Execute command
        import subprocess

        result = subprocess.run(compose_cmd, capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Deployment completed successfully!")
            if start_services:
                print("\n📊 Service status:")
                status_cmd = compose_cmd[:-3] + ["ps"]  # Remove "up -d --build"
                status_result = subprocess.run(
                    status_cmd, capture_output=True, text=True
                )
                print(status_result.stdout)
        else:
            print("❌ Deployment failed!")
            print(result.stderr)

        # Return to original directory
        os.chdir("..")

    except Exception as e:
        logger.error(f"Deployment failed: {e}")


def export_environment_template(env_name: str, output_file: str) -> None:
    """Export environment configuration template."""
    print(f"\n📤 Exporting {env_name} environment template")
    print("=" * 50)

    try:
        env_manager = EnvironmentManager()
        target_env = Environment.from_string(env_name)

        env_manager.export_environment_template(target_env, output_file)
        print(f"✅ Environment template exported to: {output_file}")

    except Exception as e:
        logger.error(f"Template export failed: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Environment Management for HA ML Predictor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available environments
  %(prog)s list

  # Set up production environment
  %(prog)s setup production

  # Validate staging environment with connection tests
  %(prog)s validate staging --test-connections

  # Manage secrets for production
  %(prog)s secrets production list
  %(prog)s secrets production set ha_token your_token_here

  # Create backup
  %(prog)s backup production --type database

  # Deploy staging environment
  %(prog)s deploy staging

  # Export environment template
  %(prog)s export production production_template.yaml
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # List command
    subparsers.add_parser("list", help="List available environments")

    # Setup command
    setup_parser = subparsers.add_parser("setup", help="Set up environment")
    setup_parser.add_argument(
        "environment", choices=["development", "testing", "staging", "production"]
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate environment configuration"
    )
    validate_parser.add_argument(
        "environment", choices=["development", "testing", "staging", "production"]
    )
    validate_parser.add_argument(
        "--test-connections", action="store_true", help="Test external connections"
    )

    # Secrets command
    secrets_parser = subparsers.add_parser("secrets", help="Manage environment secrets")
    secrets_parser.add_argument(
        "environment", choices=["development", "testing", "staging", "production"]
    )
    secrets_parser.add_argument("action", choices=["list", "set", "get", "setup"])
    secrets_parser.add_argument("--key", help="Secret key")
    secrets_parser.add_argument("--value", help="Secret value")

    # Backup command
    backup_parser = subparsers.add_parser("backup", help="Create environment backup")
    backup_parser.add_argument(
        "environment", choices=["development", "testing", "staging", "production"]
    )
    backup_parser.add_argument(
        "--type",
        choices=["database", "models", "config", "all", "disaster-recovery"],
        default="all",
    )

    # Deploy command
    deploy_parser = subparsers.add_parser("deploy", help="Deploy environment")
    deploy_parser.add_argument(
        "environment", choices=["development", "testing", "staging", "production"]
    )
    deploy_parser.add_argument(
        "--build-only",
        action="store_true",
        help="Build images only, don't start services",
    )

    # Export command
    export_parser = subparsers.add_parser("export", help="Export environment template")
    export_parser.add_argument(
        "environment", choices=["development", "testing", "staging", "production"]
    )
    export_parser.add_argument("output_file", help="Output template file")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    # Execute command
    if args.command == "list":
        list_environments()

    elif args.command == "setup":
        setup_environment(args.environment)

    elif args.command == "validate":
        success = validate_environment(args.environment, args.test_connections)
        sys.exit(0 if success else 1)

    elif args.command == "secrets":
        manage_secrets(args.environment, args.action, args.key, args.value)

    elif args.command == "backup":
        backup_environment(args.environment, args.type)

    elif args.command == "deploy":
        deploy_environment(args.environment, not args.build_only)

    elif args.command == "export":
        export_environment_template(args.environment, args.output_file)


if __name__ == "__main__":
    main()
