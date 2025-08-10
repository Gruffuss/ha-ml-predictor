"""
Backup and disaster recovery management system.
Handles database backups, model backups, and configuration backups.
"""

import os
import sys
import shutil
import gzip
import logging
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any
import subprocess
import json
import yaml
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class BackupMetadata:
    """Metadata for a backup."""

    backup_id: str
    backup_type: str
    timestamp: datetime
    size_bytes: int
    compressed: bool
    checksum: Optional[str] = None
    retention_date: Optional[datetime] = None
    tags: Dict[str, str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "backup_id": self.backup_id,
            "backup_type": self.backup_type,
            "timestamp": self.timestamp.isoformat(),
            "size_bytes": self.size_bytes,
            "compressed": self.compressed,
            "checksum": self.checksum,
            "retention_date": (
                self.retention_date.isoformat() if self.retention_date else None
            ),
            "tags": self.tags or {},
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BackupMetadata":
        """Create from dictionary."""
        return cls(
            backup_id=data["backup_id"],
            backup_type=data["backup_type"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            size_bytes=data["size_bytes"],
            compressed=data["compressed"],
            checksum=data.get("checksum"),
            retention_date=(
                datetime.fromisoformat(data["retention_date"])
                if data.get("retention_date")
                else None
            ),
            tags=data.get("tags", {}),
        )


class DatabaseBackupManager:
    """Manages database backups and restoration."""

    def __init__(self, backup_dir: str, db_config: Dict[str, Any]):
        self.backup_dir = Path(backup_dir) / "database"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.db_config = db_config

    def create_backup(
        self, backup_id: Optional[str] = None, compress: bool = True
    ) -> BackupMetadata:
        """Create a database backup."""
        if not backup_id:
            backup_id = f"db_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_file = self.backup_dir / f"{backup_id}.sql"
        if compress:
            backup_file = backup_file.with_suffix(".sql.gz")

        logger.info(f"Creating database backup: {backup_id}")

        try:
            # Extract database connection details
            connection_string = self.db_config.get("connection_string", "")

            # Parse connection string for pg_dump
            # Format: postgresql+asyncpg://user:pass@host:port/dbname
            if "://" in connection_string:
                _, rest = connection_string.split("://", 1)
                if "@" in rest:
                    user_pass, host_db = rest.split("@", 1)
                    if ":" in user_pass:
                        user, password = user_pass.split(":", 1)
                    else:
                        user = user_pass
                        password = ""
                else:
                    user = "postgres"
                    password = ""
                    host_db = rest

                if "/" in host_db:
                    host_port, dbname = host_db.rsplit("/", 1)
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                    else:
                        host = host_port
                        port = "5432"
                else:
                    host = "localhost"
                    port = "5432"
                    dbname = host_db
            else:
                raise ValueError("Invalid database connection string format")

            # Prepare pg_dump command
            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password

            cmd = [
                "pg_dump",
                "-h",
                host,
                "-p",
                port,
                "-U",
                user,
                "-d",
                dbname,
                "--no-password",
                "--verbose",
                "--clean",
                "--if-exists",
                "--create",
            ]

            # Execute backup
            with open(
                backup_file if not compress else backup_file.with_suffix(".sql"), "w"
            ) as f:
                result = subprocess.run(
                    cmd,
                    stdout=f,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=True,
                    timeout=3600,  # 1 hour timeout
                )

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                raise RuntimeError(f"pg_dump failed: {error_msg}")

            # Compress if requested
            if compress:
                with open(backup_file.with_suffix(".sql"), "rb") as f_in:
                    with gzip.open(backup_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)
                os.remove(backup_file.with_suffix(".sql"))

            # Create metadata
            size_bytes = backup_file.stat().st_size
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type="database",
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                compressed=compress,
                tags={"database": dbname, "host": host},
            )

            self._save_backup_metadata(metadata)
            logger.info(
                f"Database backup created successfully: {backup_file} ({size_bytes / 1024 / 1024:.1f} MB)"
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to create database backup: {e}")
            # Cleanup failed backup file
            if backup_file.exists():
                backup_file.unlink()
            raise

    def restore_backup(self, backup_id: str) -> None:
        """Restore database from backup."""
        metadata = self._load_backup_metadata(backup_id)
        if not metadata or metadata.backup_type != "database":
            raise ValueError(f"Database backup not found: {backup_id}")

        backup_file = self.backup_dir / f"{backup_id}.sql"
        if metadata.compressed:
            backup_file = backup_file.with_suffix(".sql.gz")

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        logger.warning(f"Restoring database from backup: {backup_id}")

        try:
            # Extract database connection details (same as backup)
            connection_string = self.db_config.get("connection_string", "")

            # Parse connection string for psql
            if "://" in connection_string:
                _, rest = connection_string.split("://", 1)
                if "@" in rest:
                    user_pass, host_db = rest.split("@", 1)
                    if ":" in user_pass:
                        user, password = user_pass.split(":", 1)
                    else:
                        user = user_pass
                        password = ""
                else:
                    user = "postgres"
                    password = ""
                    host_db = rest

                if "/" in host_db:
                    host_port, dbname = host_db.rsplit("/", 1)
                    if ":" in host_port:
                        host, port = host_port.split(":", 1)
                    else:
                        host = host_port
                        port = "5432"
                else:
                    host = "localhost"
                    port = "5432"
                    dbname = host_db
            else:
                raise ValueError("Invalid database connection string format")

            # Prepare psql command
            env = os.environ.copy()
            if password:
                env["PGPASSWORD"] = password

            cmd = [
                "psql",
                "-h",
                host,
                "-p",
                port,
                "-U",
                user,
                "-d",
                "postgres",  # Connect to postgres db first
                "--no-password",
                "-v",
                "ON_ERROR_STOP=1",
            ]

            # Prepare input data
            if metadata.compressed:
                # Read compressed file
                with gzip.open(backup_file, "rt") as f:
                    sql_content = f.read()
            else:
                # Read regular file
                with open(backup_file, "r") as f:
                    sql_content = f.read()

            # Execute restoration
            result = subprocess.run(
                cmd,
                input=sql_content,
                stderr=subprocess.PIPE,
                env=env,
                text=True,
                timeout=3600,  # 1 hour timeout
            )

            if result.returncode != 0:
                error_msg = result.stderr if result.stderr else "Unknown error"
                raise RuntimeError(f"Database restoration failed: {error_msg}")

            logger.info(f"Database restored successfully from backup: {backup_id}")

        except Exception as e:
            logger.error(f"Failed to restore database backup: {e}")
            raise

    def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to file."""
        metadata_file = self.backup_dir / f"{metadata.backup_id}.metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata from file."""
        metadata_file = self.backup_dir / f"{backup_id}.metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file, "r") as f:
            data = json.load(f)

        return BackupMetadata.from_dict(data)


class ModelBackupManager:
    """Manages ML model backups and restoration."""

    def __init__(self, backup_dir: str, models_dir: str):
        self.backup_dir = Path(backup_dir) / "models"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir = Path(models_dir)

    def create_backup(
        self, backup_id: Optional[str] = None, compress: bool = True
    ) -> BackupMetadata:
        """Create a model backup."""
        if not backup_id:
            backup_id = f"models_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_file = self.backup_dir / f"{backup_id}.tar"
        if compress:
            backup_file = backup_file.with_suffix(".tar.gz")

        logger.info(f"Creating models backup: {backup_id}")

        try:
            if not self.models_dir.exists():
                logger.warning("Models directory does not exist, creating empty backup")
                self.models_dir.mkdir(parents=True, exist_ok=True)

            # Create tar archive
            cmd = [
                "tar",
                "-cf" if not compress else "-czf",
                str(backup_file),
                "-C",
                str(self.models_dir.parent),
                self.models_dir.name,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=1800
            )  # 30 minutes

            if result.returncode != 0:
                raise RuntimeError(f"tar command failed: {result.stderr}")

            # Create metadata
            size_bytes = backup_file.stat().st_size
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type="models",
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                compressed=compress,
                tags={"models_dir": str(self.models_dir)},
            )

            self._save_backup_metadata(metadata)
            logger.info(
                f"Models backup created successfully: {backup_file} ({size_bytes / 1024 / 1024:.1f} MB)"
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to create models backup: {e}")
            if backup_file.exists():
                backup_file.unlink()
            raise

    def restore_backup(self, backup_id: str) -> None:
        """Restore models from backup."""
        metadata = self._load_backup_metadata(backup_id)
        if not metadata or metadata.backup_type != "models":
            raise ValueError(f"Models backup not found: {backup_id}")

        backup_file = self.backup_dir / f"{backup_id}.tar"
        if metadata.compressed:
            backup_file = backup_file.with_suffix(".tar.gz")

        if not backup_file.exists():
            raise FileNotFoundError(f"Backup file not found: {backup_file}")

        logger.warning(f"Restoring models from backup: {backup_id}")

        try:
            # Backup existing models directory
            if self.models_dir.exists():
                backup_existing = self.models_dir.with_suffix(
                    f".backup.{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                )
                shutil.move(str(self.models_dir), str(backup_existing))
                logger.info(f"Existing models backed up to: {backup_existing}")

            # Extract models
            cmd = [
                "tar",
                "-xf" if not metadata.compressed else "-xzf",
                str(backup_file),
                "-C",
                str(self.models_dir.parent),
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)

            if result.returncode != 0:
                raise RuntimeError(f"tar extraction failed: {result.stderr}")

            logger.info(f"Models restored successfully from backup: {backup_id}")

        except Exception as e:
            logger.error(f"Failed to restore models backup: {e}")
            raise

    def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to file."""
        metadata_file = self.backup_dir / f"{metadata.backup_id}.metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)

    def _load_backup_metadata(self, backup_id: str) -> Optional[BackupMetadata]:
        """Load backup metadata from file."""
        metadata_file = self.backup_dir / f"{backup_id}.metadata.json"
        if not metadata_file.exists():
            return None

        with open(metadata_file, "r") as f:
            data = json.load(f)

        return BackupMetadata.from_dict(data)


class ConfigurationBackupManager:
    """Manages configuration file backups."""

    def __init__(self, backup_dir: str, config_dir: str):
        self.backup_dir = Path(backup_dir) / "config"
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.config_dir = Path(config_dir)

    def create_backup(
        self, backup_id: Optional[str] = None, compress: bool = True
    ) -> BackupMetadata:
        """Create a configuration backup."""
        if not backup_id:
            backup_id = f"config_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        backup_file = self.backup_dir / f"{backup_id}.tar"
        if compress:
            backup_file = backup_file.with_suffix(".tar.gz")

        logger.info(f"Creating configuration backup: {backup_id}")

        try:
            # Create tar archive of config directory
            cmd = [
                "tar",
                "-cf" if not compress else "-czf",
                str(backup_file),
                "-C",
                str(self.config_dir.parent),
                self.config_dir.name,
            ]

            result = subprocess.run(
                cmd, capture_output=True, text=True, timeout=300
            )  # 5 minutes

            if result.returncode != 0:
                raise RuntimeError(f"tar command failed: {result.stderr}")

            # Create metadata
            size_bytes = backup_file.stat().st_size
            metadata = BackupMetadata(
                backup_id=backup_id,
                backup_type="config",
                timestamp=datetime.now(),
                size_bytes=size_bytes,
                compressed=compress,
                tags={"config_dir": str(self.config_dir)},
            )

            self._save_backup_metadata(metadata)
            logger.info(
                f"Configuration backup created successfully: {backup_file} ({size_bytes / 1024:.1f} KB)"
            )

            return metadata

        except Exception as e:
            logger.error(f"Failed to create configuration backup: {e}")
            if backup_file.exists():
                backup_file.unlink()
            raise

    def _save_backup_metadata(self, metadata: BackupMetadata) -> None:
        """Save backup metadata to file."""
        metadata_file = self.backup_dir / f"{metadata.backup_id}.metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata.to_dict(), f, indent=2)


class BackupManager:
    """Centralized backup and disaster recovery manager."""

    def __init__(self, backup_dir: str, config: Dict[str, Any]):
        self.backup_dir = Path(backup_dir)
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        self.config = config

        # Initialize sub-managers
        self.db_backup_manager = DatabaseBackupManager(
            str(self.backup_dir), config.get("database", {})
        )
        self.model_backup_manager = ModelBackupManager(
            str(self.backup_dir), config.get("models_dir", "models")
        )
        self.config_backup_manager = ConfigurationBackupManager(
            str(self.backup_dir), config.get("config_dir", "config")
        )

        self.backup_config = config.get("backup", {})

    async def run_scheduled_backups(self) -> None:
        """Run scheduled backup tasks."""
        if not self.backup_config.get("enabled", False):
            logger.info("Backups are disabled")
            return

        interval_hours = self.backup_config.get("interval_hours", 24)
        retention_days = self.backup_config.get("retention_days", 7)
        compress = self.backup_config.get("compress", True)

        logger.info(f"Starting scheduled backup task (every {interval_hours} hours)")

        while True:
            try:
                # Create backups
                backups_created = []

                # Database backup
                if self.backup_config.get("database_backup", True):
                    db_backup = self.db_backup_manager.create_backup(compress=compress)
                    db_backup.retention_date = datetime.now() + timedelta(
                        days=retention_days
                    )
                    backups_created.append(db_backup)

                # Model backup (less frequent)
                model_interval_hours = self.backup_config.get(
                    "model_backup_interval_hours", 24
                )
                current_hour = datetime.now().hour
                if current_hour % model_interval_hours == 0 and self.backup_config.get(
                    "model_backup", True
                ):
                    model_backup = self.model_backup_manager.create_backup(
                        compress=compress
                    )
                    model_backup.retention_date = datetime.now() + timedelta(
                        days=retention_days
                    )
                    backups_created.append(model_backup)

                # Configuration backup (daily)
                if current_hour == 2:  # 2 AM
                    config_backup = self.config_backup_manager.create_backup(
                        compress=compress
                    )
                    config_backup.retention_date = datetime.now() + timedelta(
                        days=retention_days * 2
                    )  # Keep config backups longer
                    backups_created.append(config_backup)

                logger.info(
                    f"Completed scheduled backup: {len(backups_created)} backups created"
                )

                # Cleanup old backups
                await self.cleanup_expired_backups()

            except Exception as e:
                logger.error(f"Scheduled backup failed: {e}")

            # Wait for next interval
            await asyncio.sleep(interval_hours * 3600)

    async def cleanup_expired_backups(self) -> None:
        """Remove expired backup files."""
        logger.info("Starting backup cleanup")

        cleaned_count = 0
        total_size_freed = 0

        # Find all metadata files
        for metadata_file in self.backup_dir.rglob("*.metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                metadata = BackupMetadata.from_dict(data)

                # Check if backup has expired
                if metadata.retention_date and datetime.now() > metadata.retention_date:
                    # Find and remove backup file
                    backup_extensions = [".sql", ".sql.gz", ".tar", ".tar.gz"]
                    for ext in backup_extensions:
                        backup_file = (
                            metadata_file.parent / f"{metadata.backup_id}{ext}"
                        )
                        if backup_file.exists():
                            size = backup_file.stat().st_size
                            backup_file.unlink()
                            total_size_freed += size
                            break

                    # Remove metadata file
                    metadata_file.unlink()
                    cleaned_count += 1

                    logger.debug(f"Removed expired backup: {metadata.backup_id}")

            except Exception as e:
                logger.error(f"Failed to cleanup backup {metadata_file}: {e}")

        if cleaned_count > 0:
            logger.info(
                f"Backup cleanup completed: {cleaned_count} backups removed, {total_size_freed / 1024 / 1024:.1f} MB freed"
            )

    def list_backups(self, backup_type: Optional[str] = None) -> List[BackupMetadata]:
        """List all available backups."""
        backups = []

        for metadata_file in self.backup_dir.rglob("*.metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)

                metadata = BackupMetadata.from_dict(data)

                if backup_type is None or metadata.backup_type == backup_type:
                    backups.append(metadata)

            except Exception as e:
                logger.error(f"Failed to load backup metadata {metadata_file}: {e}")

        # Sort by timestamp (newest first)
        return sorted(backups, key=lambda x: x.timestamp, reverse=True)

    def get_backup_info(self, backup_id: str) -> Optional[BackupMetadata]:
        """Get information about a specific backup."""
        for metadata_file in self.backup_dir.rglob(f"{backup_id}.metadata.json"):
            try:
                with open(metadata_file, "r") as f:
                    data = json.load(f)
                return BackupMetadata.from_dict(data)
            except Exception as e:
                logger.error(f"Failed to load backup metadata: {e}")

        return None

    def restore_database_backup(self, backup_id: str) -> None:
        """Restore database from backup."""
        self.db_backup_manager.restore_backup(backup_id)

    def restore_models_backup(self, backup_id: str) -> None:
        """Restore models from backup."""
        self.model_backup_manager.restore_backup(backup_id)

    def create_disaster_recovery_package(self) -> str:
        """Create a complete disaster recovery package."""
        package_id = f"disaster_recovery_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        package_dir = self.backup_dir / package_id
        package_dir.mkdir(exist_ok=True)

        logger.info(f"Creating disaster recovery package: {package_id}")

        try:
            # Create all types of backups
            db_backup = self.db_backup_manager.create_backup(f"{package_id}_db")
            model_backup = self.model_backup_manager.create_backup(
                f"{package_id}_models"
            )
            config_backup = self.config_backup_manager.create_backup(
                f"{package_id}_config"
            )

            # Create package manifest
            manifest = {
                "package_id": package_id,
                "created_at": datetime.now().isoformat(),
                "backups": [
                    db_backup.to_dict(),
                    model_backup.to_dict(),
                    config_backup.to_dict(),
                ],
                "system_info": {
                    "platform": os.name,
                    "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
                },
            }

            manifest_file = package_dir / "manifest.json"
            with open(manifest_file, "w") as f:
                json.dump(manifest, f, indent=2)

            logger.info(f"Disaster recovery package created: {package_dir}")
            return package_id

        except Exception as e:
            logger.error(f"Failed to create disaster recovery package: {e}")
            # Cleanup partial package
            if package_dir.exists():
                shutil.rmtree(package_dir)
            raise
