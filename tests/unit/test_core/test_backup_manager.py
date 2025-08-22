"""
Comprehensive unit tests for backup_manager.py.
Tests backup creation, restoration, metadata handling, and disaster recovery capabilities.
"""

import asyncio
from datetime import datetime, timedelta
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import gzip
import pytest

from src.core.backup_manager import (
    BackupManager,
    BackupMetadata,
    ConfigurationBackupManager,
    DatabaseBackupManager,
    ModelBackupManager,
)


class TestBackupMetadata:
    """Test BackupMetadata class functionality."""

    def test_metadata_creation(self):
        """Test creating backup metadata with all fields."""
        timestamp = datetime.now()
        retention_date = timestamp + timedelta(days=7)
        tags = {"database": "test_db", "host": "localhost"}

        metadata = BackupMetadata(
            backup_id="test_backup_001",
            backup_type="database",
            timestamp=timestamp,
            size_bytes=1024000,
            compressed=True,
            checksum="abc123",
            retention_date=retention_date,
            tags=tags,
        )

        assert metadata.backup_id == "test_backup_001"
        assert metadata.backup_type == "database"
        assert metadata.timestamp == timestamp
        assert metadata.size_bytes == 1024000
        assert metadata.compressed is True
        assert metadata.checksum == "abc123"
        assert metadata.retention_date == retention_date
        assert metadata.tags == tags

    def test_metadata_to_dict(self):
        """Test converting metadata to dictionary."""
        timestamp = datetime(2024, 1, 15, 14, 30, 0)
        retention_date = datetime(2024, 1, 22, 14, 30, 0)
        tags = {"env": "test"}

        metadata = BackupMetadata(
            backup_id="test_backup_001",
            backup_type="database",
            timestamp=timestamp,
            size_bytes=1024000,
            compressed=True,
            checksum="abc123",
            retention_date=retention_date,
            tags=tags,
        )

        result = metadata.to_dict()

        expected = {
            "backup_id": "test_backup_001",
            "backup_type": "database",
            "timestamp": "2024-01-15T14:30:00",
            "size_bytes": 1024000,
            "compressed": True,
            "checksum": "abc123",
            "retention_date": "2024-01-22T14:30:00",
            "tags": {"env": "test"},
        }

        assert result == expected

    def test_metadata_from_dict(self):
        """Test creating metadata from dictionary."""
        data = {
            "backup_id": "test_backup_001",
            "backup_type": "database",
            "timestamp": "2024-01-15T14:30:00",
            "size_bytes": 1024000,
            "compressed": True,
            "checksum": "abc123",
            "retention_date": "2024-01-22T14:30:00",
            "tags": {"env": "test"},
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.backup_id == "test_backup_001"
        assert metadata.backup_type == "database"
        assert metadata.timestamp == datetime(2024, 1, 15, 14, 30, 0)
        assert metadata.size_bytes == 1024000
        assert metadata.compressed is True
        assert metadata.checksum == "abc123"
        assert metadata.retention_date == datetime(2024, 1, 22, 14, 30, 0)
        assert metadata.tags == {"env": "test"}

    def test_metadata_from_dict_minimal(self):
        """Test creating metadata from dictionary with minimal fields."""
        data = {
            "backup_id": "test_backup_001",
            "backup_type": "database",
            "timestamp": "2024-01-15T14:30:00",
            "size_bytes": 1024000,
            "compressed": False,
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.backup_id == "test_backup_001"
        assert metadata.backup_type == "database"
        assert metadata.timestamp == datetime(2024, 1, 15, 14, 30, 0)
        assert metadata.size_bytes == 1024000
        assert metadata.compressed is False
        assert metadata.checksum is None
        assert metadata.retention_date is None
        assert metadata.tags == {}


class TestDatabaseBackupManager:
    """Test DatabaseBackupManager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def db_config(self):
        """Sample database configuration."""
        return {
            "connection_string": "postgresql+asyncpg://test_user:test_pass@localhost:5432/test_db"
        }

    @pytest.fixture
    def db_backup_manager(self, temp_backup_dir, db_config):
        """Create DatabaseBackupManager instance."""
        return DatabaseBackupManager(temp_backup_dir, db_config)

    def test_init_creates_backup_directory(self, temp_backup_dir, db_config):
        """Test that initialization creates backup directory."""
        manager = DatabaseBackupManager(temp_backup_dir, db_config)
        assert manager.backup_dir.exists()
        assert manager.backup_dir.name == "database"
        assert manager.db_config == db_config

    @patch("subprocess.run")
    @patch("os.remove")
    def test_create_backup_uncompressed_success(
        self, mock_remove, mock_subprocess, db_backup_manager, temp_backup_dir
    ):
        """Test successful uncompressed database backup creation."""
        # Mock successful pg_dump
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Create a mock SQL file
        backup_file = Path(temp_backup_dir) / "database" / "test_backup.sql"
        backup_file.parent.mkdir(parents=True, exist_ok=True)
        backup_file.write_text("-- Test SQL backup content\nCREATE TABLE test();")

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1024

            metadata = db_backup_manager.create_backup(
                backup_id="test_backup", compress=False
            )

            assert metadata.backup_id == "test_backup"
            assert metadata.backup_type == "database"
            assert metadata.size_bytes == 1024
            assert metadata.compressed is False
            assert isinstance(metadata.timestamp, datetime)

        # Verify pg_dump was called with correct parameters
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert "pg_dump" in call_args[0][0]
        assert "-h" in call_args[0][0]
        assert "localhost" in call_args[0][0]
        assert "-p" in call_args[0][0]
        assert "5432" in call_args[0][0]
        assert "-U" in call_args[0][0]
        assert "test_user" in call_args[0][0]
        assert "-d" in call_args[0][0]
        assert "test_db" in call_args[0][0]

        # Verify environment variable was set
        assert call_args[1]["env"]["PGPASSWORD"] == "test_pass"

        # Verify no compression removal was called
        mock_remove.assert_not_called()

    @patch("subprocess.run")
    @patch("gzip.open")
    @patch("shutil.copyfileobj")
    @patch("os.remove")
    def test_create_backup_compressed_success(
        self,
        mock_remove,
        mock_copyfile,
        mock_gzip_open,
        mock_subprocess,
        db_backup_manager,
        temp_backup_dir,
    ):
        """Test successful compressed database backup creation."""
        # Mock successful pg_dump
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stderr = ""

        # Create a mock SQL file
        backup_file = Path(temp_backup_dir) / "database" / "test_backup.sql.gz"
        backup_file.parent.mkdir(parents=True, exist_ok=True)

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 512  # Compressed size

            metadata = db_backup_manager.create_backup(
                backup_id="test_backup", compress=True
            )

            assert metadata.backup_id == "test_backup"
            assert metadata.backup_type == "database"
            assert metadata.size_bytes == 512
            assert metadata.compressed is True

        # Verify compression was performed
        mock_gzip_open.assert_called_once()
        mock_copyfile.assert_called_once()
        mock_remove.assert_called_once()  # Original SQL file removed

    @patch("subprocess.run")
    def test_create_backup_pg_dump_failure(self, mock_subprocess, db_backup_manager):
        """Test backup creation when pg_dump fails."""
        # Mock failed pg_dump
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Connection failed"

        with pytest.raises(RuntimeError, match="pg_dump failed: Connection failed"):
            db_backup_manager.create_backup(backup_id="test_backup")

    def test_create_backup_auto_generated_id(self, db_backup_manager):
        """Test backup creation with auto-generated backup ID."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stderr = ""

            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 1024

                metadata = db_backup_manager.create_backup()

                assert metadata.backup_id.startswith("db_")
                assert len(metadata.backup_id) > 10  # Should include timestamp

    def test_create_backup_invalid_connection_string(self, temp_backup_dir):
        """Test backup creation with invalid connection string."""
        invalid_config = {"connection_string": "invalid_format"}
        manager = DatabaseBackupManager(temp_backup_dir, invalid_config)

        with pytest.raises(
            ValueError, match="Invalid database connection string format"
        ):
            manager.create_backup()

    @patch("subprocess.run")
    @patch("gzip.open")
    def test_restore_backup_compressed_success(
        self, mock_gzip_open, mock_subprocess, db_backup_manager, temp_backup_dir
    ):
        """Test successful restoration from compressed backup."""
        # Create backup metadata
        backup_dir = Path(temp_backup_dir) / "database"
        metadata_file = backup_dir / "test_backup.metadata.json"
        metadata = {
            "backup_id": "test_backup",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": True,
        }
        metadata_file.write_text(json.dumps(metadata))

        # Create backup file
        backup_file = backup_dir / "test_backup.sql.gz"
        backup_file.touch()

        # Mock successful psql
        mock_subprocess.return_value.returncode = 0

        # Mock gzip read
        mock_gzip_open.return_value.__enter__.return_value.read.return_value = (
            "-- Test SQL content"
        )

        db_backup_manager.restore_backup("test_backup")

        # Verify psql was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert "psql" in call_args[0][0]
        assert call_args[1]["input"] == "-- Test SQL content"

    @patch("subprocess.run")
    def test_restore_backup_uncompressed_success(
        self, mock_subprocess, db_backup_manager, temp_backup_dir
    ):
        """Test successful restoration from uncompressed backup."""
        # Create backup metadata
        backup_dir = Path(temp_backup_dir) / "database"
        metadata_file = backup_dir / "test_backup.metadata.json"
        metadata = {
            "backup_id": "test_backup",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": False,
        }
        metadata_file.write_text(json.dumps(metadata))

        # Create backup file
        backup_file = backup_dir / "test_backup.sql"
        backup_file.write_text("-- Test SQL content")

        # Mock successful psql
        mock_subprocess.return_value.returncode = 0

        db_backup_manager.restore_backup("test_backup")

        # Verify psql was called with correct content
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args
        assert call_args[1]["input"] == "-- Test SQL content"

    def test_restore_backup_not_found(self, db_backup_manager):
        """Test restoration when backup is not found."""
        with pytest.raises(ValueError, match="Database backup not found: nonexistent"):
            db_backup_manager.restore_backup("nonexistent")

    def test_restore_backup_file_not_found(self, db_backup_manager, temp_backup_dir):
        """Test restoration when backup file is missing."""
        # Create metadata but no backup file
        backup_dir = Path(temp_backup_dir) / "database"
        metadata_file = backup_dir / "test_backup.metadata.json"
        metadata = {
            "backup_id": "test_backup",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": False,
        }
        metadata_file.write_text(json.dumps(metadata))

        with pytest.raises(FileNotFoundError, match="Backup file not found"):
            db_backup_manager.restore_backup("test_backup")

    @patch("subprocess.run")
    def test_restore_backup_psql_failure(
        self, mock_subprocess, db_backup_manager, temp_backup_dir
    ):
        """Test restoration when psql fails."""
        # Create backup metadata and file
        backup_dir = Path(temp_backup_dir) / "database"
        metadata_file = backup_dir / "test_backup.metadata.json"
        metadata = {
            "backup_id": "test_backup",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": False,
        }
        metadata_file.write_text(json.dumps(metadata))

        backup_file = backup_dir / "test_backup.sql"
        backup_file.write_text("-- Test SQL content")

        # Mock failed psql
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Database error"

        with pytest.raises(
            RuntimeError, match="Database restoration failed: Database error"
        ):
            db_backup_manager.restore_backup("test_backup")


class TestModelBackupManager:
    """Test ModelBackupManager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def temp_models_dir(self):
        """Create temporary models directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some model files
            models_dir = Path(tmpdir) / "models"
            models_dir.mkdir()
            (models_dir / "model1.pkl").write_text("model1 content")
            (models_dir / "model2.pkl").write_text("model2 content")
            (models_dir / "subdir").mkdir()
            (models_dir / "subdir" / "model3.pkl").write_text("model3 content")
            yield str(models_dir)

    @pytest.fixture
    def model_backup_manager(self, temp_backup_dir, temp_models_dir):
        """Create ModelBackupManager instance."""
        return ModelBackupManager(temp_backup_dir, temp_models_dir)

    def test_init_creates_backup_directory(self, temp_backup_dir, temp_models_dir):
        """Test that initialization creates backup directory."""
        manager = ModelBackupManager(temp_backup_dir, temp_models_dir)
        assert manager.backup_dir.exists()
        assert manager.backup_dir.name == "models"
        assert str(manager.models_dir) == temp_models_dir

    @patch("subprocess.run")
    def test_create_backup_uncompressed_success(
        self, mock_subprocess, model_backup_manager, temp_backup_dir
    ):
        """Test successful uncompressed model backup creation."""
        # Mock successful tar
        mock_subprocess.return_value.returncode = 0

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 2048

            metadata = model_backup_manager.create_backup(
                backup_id="test_models", compress=False
            )

            assert metadata.backup_id == "test_models"
            assert metadata.backup_type == "models"
            assert metadata.size_bytes == 2048
            assert metadata.compressed is False

        # Verify tar was called with correct parameters
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "tar" in call_args
        assert "-cf" in call_args
        assert "test_models.tar" in call_args[-3]

    @patch("subprocess.run")
    def test_create_backup_compressed_success(
        self, mock_subprocess, model_backup_manager
    ):
        """Test successful compressed model backup creation."""
        # Mock successful tar
        mock_subprocess.return_value.returncode = 0

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 1024

            metadata = model_backup_manager.create_backup(
                backup_id="test_models", compress=True
            )

            assert metadata.backup_id == "test_models"
            assert metadata.backup_type == "models"
            assert metadata.size_bytes == 1024
            assert metadata.compressed is True

        # Verify tar was called with compression
        call_args = mock_subprocess.call_args[0][0]
        assert "-czf" in call_args

    @patch("subprocess.run")
    def test_create_backup_tar_failure(self, mock_subprocess, model_backup_manager):
        """Test backup creation when tar fails."""
        # Mock failed tar
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Tar error"

        with pytest.raises(RuntimeError, match="tar command failed: Tar error"):
            model_backup_manager.create_backup()

    def test_create_backup_nonexistent_models_dir(self, temp_backup_dir):
        """Test backup creation when models directory doesn't exist."""
        nonexistent_dir = "/path/that/does/not/exist"
        manager = ModelBackupManager(temp_backup_dir, nonexistent_dir)

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 0

                # Should create empty backup and models directory
                metadata = manager.create_backup()
                assert metadata.backup_type == "models"

        # Verify models directory was created
        assert Path(nonexistent_dir).exists()

    @patch("subprocess.run")
    @patch("shutil.move")
    def test_restore_backup_success(
        self, mock_move, mock_subprocess, model_backup_manager, temp_backup_dir
    ):
        """Test successful model backup restoration."""
        # Create backup metadata
        backup_dir = Path(temp_backup_dir) / "models"
        metadata_file = backup_dir / "test_models.metadata.json"
        metadata = {
            "backup_id": "test_models",
            "backup_type": "models",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": False,
        }
        metadata_file.write_text(json.dumps(metadata))

        # Create backup file
        backup_file = backup_dir / "test_models.tar"
        backup_file.touch()

        # Mock successful tar extraction
        mock_subprocess.return_value.returncode = 0

        model_backup_manager.restore_backup("test_models")

        # Verify existing models were backed up
        mock_move.assert_called_once()

        # Verify tar extraction was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "tar" in call_args
        assert "-xf" in call_args

    @patch("subprocess.run")
    def test_restore_backup_compressed(
        self, mock_subprocess, model_backup_manager, temp_backup_dir
    ):
        """Test restoration of compressed backup."""
        # Create backup metadata for compressed backup
        backup_dir = Path(temp_backup_dir) / "models"
        metadata_file = backup_dir / "test_models.metadata.json"
        metadata = {
            "backup_id": "test_models",
            "backup_type": "models",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": True,
        }
        metadata_file.write_text(json.dumps(metadata))

        # Create backup file
        backup_file = backup_dir / "test_models.tar.gz"
        backup_file.touch()

        # Mock successful tar extraction
        mock_subprocess.return_value.returncode = 0

        model_backup_manager.restore_backup("test_models")

        # Verify compressed extraction was used
        call_args = mock_subprocess.call_args[0][0]
        assert "-xzf" in call_args

    def test_restore_backup_not_found(self, model_backup_manager):
        """Test restoration when backup is not found."""
        with pytest.raises(ValueError, match="Models backup not found: nonexistent"):
            model_backup_manager.restore_backup("nonexistent")

    @patch("subprocess.run")
    def test_restore_backup_tar_failure(
        self, mock_subprocess, model_backup_manager, temp_backup_dir
    ):
        """Test restoration when tar extraction fails."""
        # Create backup metadata and file
        backup_dir = Path(temp_backup_dir) / "models"
        metadata_file = backup_dir / "test_models.metadata.json"
        metadata = {
            "backup_id": "test_models",
            "backup_type": "models",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": False,
        }
        metadata_file.write_text(json.dumps(metadata))

        backup_file = backup_dir / "test_models.tar"
        backup_file.touch()

        # Mock failed tar
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Extraction failed"

        with pytest.raises(
            RuntimeError, match="tar extraction failed: Extraction failed"
        ):
            model_backup_manager.restore_backup("test_models")


class TestConfigurationBackupManager:
    """Test ConfigurationBackupManager functionality."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary config directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            (config_dir / "config.yaml").write_text("setting: value")
            (config_dir / "rooms.yaml").write_text("rooms: []")
            yield str(config_dir)

    @pytest.fixture
    def config_backup_manager(self, temp_backup_dir, temp_config_dir):
        """Create ConfigurationBackupManager instance."""
        return ConfigurationBackupManager(temp_backup_dir, temp_config_dir)

    @patch("subprocess.run")
    def test_create_backup_success(
        self, mock_subprocess, config_backup_manager, temp_backup_dir
    ):
        """Test successful configuration backup creation."""
        # Mock successful tar
        mock_subprocess.return_value.returncode = 0

        with patch("pathlib.Path.stat") as mock_stat:
            mock_stat.return_value.st_size = 512

            metadata = config_backup_manager.create_backup(
                backup_id="test_config", compress=True
            )

            assert metadata.backup_id == "test_config"
            assert metadata.backup_type == "config"
            assert metadata.size_bytes == 512
            assert metadata.compressed is True

        # Verify tar was called
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        assert "tar" in call_args
        assert "-czf" in call_args

    @patch("subprocess.run")
    def test_create_backup_tar_failure(self, mock_subprocess, config_backup_manager):
        """Test backup creation when tar fails."""
        # Mock failed tar
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stderr = "Permission denied"

        with pytest.raises(RuntimeError, match="tar command failed: Permission denied"):
            config_backup_manager.create_backup()

    def test_create_backup_auto_generated_id(self, config_backup_manager):
        """Test backup creation with auto-generated ID."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            with patch("pathlib.Path.stat") as mock_stat:
                mock_stat.return_value.st_size = 512

                metadata = config_backup_manager.create_backup()

                assert metadata.backup_id.startswith("config_")
                assert len(metadata.backup_id) > 15  # Should include timestamp


class TestBackupManager:
    """Test main BackupManager orchestration."""

    @pytest.fixture
    def temp_backup_dir(self):
        """Create temporary backup directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir

    @pytest.fixture
    def sample_config(self):
        """Sample configuration for BackupManager."""
        return {
            "database": {
                "connection_string": "postgresql+asyncpg://user:pass@localhost:5432/test_db"
            },
            "models_dir": "models",
            "config_dir": "config",
            "backup": {
                "enabled": True,
                "interval_hours": 24,
                "retention_days": 7,
                "compress": True,
                "database_backup": True,
                "model_backup": True,
                "model_backup_interval_hours": 24,
            },
        }

    @pytest.fixture
    def backup_manager(self, temp_backup_dir, sample_config):
        """Create BackupManager instance."""
        return BackupManager(temp_backup_dir, sample_config)

    def test_init_creates_sub_managers(self, backup_manager):
        """Test that initialization creates all sub-managers."""
        assert backup_manager.db_backup_manager is not None
        assert backup_manager.model_backup_manager is not None
        assert backup_manager.config_backup_manager is not None

    def test_list_backups_empty(self, backup_manager):
        """Test listing backups when none exist."""
        backups = backup_manager.list_backups()
        assert backups == []

    def test_list_backups_with_data(self, backup_manager, temp_backup_dir):
        """Test listing backups with existing metadata."""
        # Create sample backup metadata
        backup_dir = Path(temp_backup_dir) / "database"
        backup_dir.mkdir()

        metadata1 = {
            "backup_id": "backup_001",
            "backup_type": "database",
            "timestamp": "2024-01-15T10:00:00",
            "size_bytes": 1024,
            "compressed": True,
        }

        metadata2 = {
            "backup_id": "backup_002",
            "backup_type": "models",
            "timestamp": "2024-01-15T12:00:00",
            "size_bytes": 2048,
            "compressed": False,
        }

        (backup_dir / "backup_001.metadata.json").write_text(json.dumps(metadata1))
        (backup_dir / "backup_002.metadata.json").write_text(json.dumps(metadata2))

        backups = backup_manager.list_backups()

        assert len(backups) == 2
        # Should be sorted by timestamp (newest first)
        assert backups[0].backup_id == "backup_002"
        assert backups[1].backup_id == "backup_001"

    def test_list_backups_filtered_by_type(self, backup_manager, temp_backup_dir):
        """Test listing backups filtered by type."""
        # Create mixed backup metadata
        db_dir = Path(temp_backup_dir) / "database"
        db_dir.mkdir()
        models_dir = Path(temp_backup_dir) / "models"
        models_dir.mkdir()

        db_metadata = {
            "backup_id": "db_backup",
            "backup_type": "database",
            "timestamp": "2024-01-15T10:00:00",
            "size_bytes": 1024,
            "compressed": True,
        }

        models_metadata = {
            "backup_id": "models_backup",
            "backup_type": "models",
            "timestamp": "2024-01-15T12:00:00",
            "size_bytes": 2048,
            "compressed": False,
        }

        (db_dir / "db_backup.metadata.json").write_text(json.dumps(db_metadata))
        (models_dir / "models_backup.metadata.json").write_text(
            json.dumps(models_metadata)
        )

        # Test filtering
        db_backups = backup_manager.list_backups(backup_type="database")
        models_backups = backup_manager.list_backups(backup_type="models")

        assert len(db_backups) == 1
        assert db_backups[0].backup_type == "database"

        assert len(models_backups) == 1
        assert models_backups[0].backup_type == "models"

    def test_get_backup_info_found(self, backup_manager, temp_backup_dir):
        """Test getting backup info for existing backup."""
        backup_dir = Path(temp_backup_dir) / "database"
        backup_dir.mkdir()

        metadata = {
            "backup_id": "test_backup",
            "backup_type": "database",
            "timestamp": "2024-01-15T10:00:00",
            "size_bytes": 1024,
            "compressed": True,
        }

        (backup_dir / "test_backup.metadata.json").write_text(json.dumps(metadata))

        info = backup_manager.get_backup_info("test_backup")

        assert info is not None
        assert info.backup_id == "test_backup"
        assert info.backup_type == "database"

    def test_get_backup_info_not_found(self, backup_manager):
        """Test getting backup info for non-existent backup."""
        info = backup_manager.get_backup_info("nonexistent")
        assert info is None

    def test_create_disaster_recovery_package(self, backup_manager):
        """Test creating disaster recovery package."""
        with patch.object(
            backup_manager.db_backup_manager, "create_backup"
        ) as mock_db, patch.object(
            backup_manager.model_backup_manager, "create_backup"
        ) as mock_models, patch.object(
            backup_manager.config_backup_manager, "create_backup"
        ) as mock_config:

            # Mock successful backups
            mock_db.return_value = BackupMetadata(
                "dr_db", "database", datetime.now(), 1024, True
            )
            mock_models.return_value = BackupMetadata(
                "dr_models", "models", datetime.now(), 2048, True
            )
            mock_config.return_value = BackupMetadata(
                "dr_config", "config", datetime.now(), 512, True
            )

            package_id = backup_manager.create_disaster_recovery_package()

            assert package_id.startswith("disaster_recovery_")
            assert len(package_id) > 30  # Should include timestamp

            # Verify all backup types were created
            mock_db.assert_called_once()
            mock_models.assert_called_once()
            mock_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_cleanup_expired_backups(self, backup_manager, temp_backup_dir):
        """Test cleanup of expired backup files."""
        backup_dir = Path(temp_backup_dir) / "database"
        backup_dir.mkdir()

        # Create expired backup
        expired_date = datetime.now() - timedelta(days=10)
        expired_metadata = {
            "backup_id": "expired_backup",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 1024,
            "compressed": False,
            "retention_date": expired_date.isoformat(),
        }

        # Create current backup
        current_date = datetime.now() + timedelta(days=5)
        current_metadata = {
            "backup_id": "current_backup",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 2048,
            "compressed": False,
            "retention_date": current_date.isoformat(),
        }

        # Write metadata files
        (backup_dir / "expired_backup.metadata.json").write_text(
            json.dumps(expired_metadata)
        )
        (backup_dir / "current_backup.metadata.json").write_text(
            json.dumps(current_metadata)
        )

        # Create backup files
        expired_backup_file = backup_dir / "expired_backup.sql"
        current_backup_file = backup_dir / "current_backup.sql"
        expired_backup_file.write_text("expired content")
        current_backup_file.write_text("current content")

        # Run cleanup
        await backup_manager.cleanup_expired_backups()

        # Verify expired backup was removed
        assert not expired_backup_file.exists()
        assert not (backup_dir / "expired_backup.metadata.json").exists()

        # Verify current backup still exists
        assert current_backup_file.exists()
        assert (backup_dir / "current_backup.metadata.json").exists()

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_disabled(self, backup_manager):
        """Test scheduled backups when disabled."""
        backup_manager.backup_config["enabled"] = False

        # Mock to ensure we don't run infinite loop
        with patch("asyncio.sleep") as mock_sleep:
            mock_sleep.side_effect = asyncio.CancelledError()

            try:
                await backup_manager.run_scheduled_backups()
            except asyncio.CancelledError:
                pass

        # Verify no backup managers were called
        with patch.object(backup_manager.db_backup_manager, "create_backup") as mock_db:
            mock_db.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_single_cycle(self, backup_manager):
        """Test single cycle of scheduled backups."""
        # Configure for quick testing
        backup_manager.backup_config.update(
            {
                "enabled": True,
                "interval_hours": 1,  # Short interval for testing
                "retention_days": 1,
                "database_backup": True,
                "model_backup": True,
                "model_backup_interval_hours": 1,
            }
        )

        with patch.object(
            backup_manager.db_backup_manager, "create_backup"
        ) as mock_db, patch.object(
            backup_manager.model_backup_manager, "create_backup"
        ) as mock_models, patch.object(
            backup_manager.config_backup_manager, "create_backup"
        ) as mock_config, patch(
            "asyncio.sleep"
        ) as mock_sleep:

            # Mock successful backups
            mock_db.return_value = BackupMetadata(
                "scheduled_db", "database", datetime.now(), 1024, True
            )
            mock_models.return_value = BackupMetadata(
                "scheduled_models", "models", datetime.now(), 2048, True
            )
            mock_config.return_value = BackupMetadata(
                "scheduled_config", "config", datetime.now(), 512, True
            )

            # Mock time to trigger config backup (hour 2)
            with patch("datetime.datetime") as mock_datetime:
                mock_datetime.now.return_value.hour = 2
                mock_datetime.now.return_value.__sub__ = datetime.now().__sub__

                # Stop after first iteration
                mock_sleep.side_effect = asyncio.CancelledError()

                try:
                    await backup_manager.run_scheduled_backups()
                except asyncio.CancelledError:
                    pass

        # Verify backups were created
        mock_db.assert_called_once()
        mock_models.assert_called_once()  # Model backup triggered due to hour match
        mock_config.assert_called_once()  # Config backup triggered due to hour 2


if __name__ == "__main__":
    pytest.main([__file__])
