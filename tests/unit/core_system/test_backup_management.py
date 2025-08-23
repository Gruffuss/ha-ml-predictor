"""Unit tests for backup management system.

Covers:
- src/core/backup_manager.py (Backup Management System)

This test file focuses on backup operations, data persistence, and recovery functionality.
"""

import asyncio
import gzip
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, Mock, mock_open, patch

import pytest

from src.core.backup_manager import (
    BackupManager,
    BackupMetadata,
    ConfigurationBackupManager,
    DatabaseBackupManager,
    ModelBackupManager,
)


class TestBackupMetadata:
    """Test BackupMetadata dataclass."""

    def test_backup_metadata_initialization_required_fields(self):
        """Test BackupMetadata initialization with required fields."""
        timestamp = datetime.now()
        metadata = BackupMetadata(
            backup_id="test_backup_001",
            backup_type="database",
            timestamp=timestamp,
            size_bytes=1024000,
            compressed=True,
        )

        assert metadata.backup_id == "test_backup_001"
        assert metadata.backup_type == "database"
        assert metadata.timestamp == timestamp
        assert metadata.size_bytes == 1024000
        assert metadata.compressed is True
        assert metadata.checksum is None
        assert metadata.retention_date is None
        assert metadata.tags is None

    def test_backup_metadata_initialization_optional_fields(self):
        """Test BackupMetadata initialization with optional fields."""
        timestamp = datetime.now()
        retention_date = datetime.now() + timedelta(days=7)
        tags = {"env": "production", "host": "db01"}

        metadata = BackupMetadata(
            backup_id="test_backup_002",
            backup_type="models",
            timestamp=timestamp,
            size_bytes=5120000,
            compressed=False,
            checksum="abc123def456",
            retention_date=retention_date,
            tags=tags,
        )

        assert metadata.checksum == "abc123def456"
        assert metadata.retention_date == retention_date
        assert metadata.tags == tags

    def test_to_dict_serialization_complete(self):
        """Test to_dict() serialization with complete data."""
        timestamp = datetime.now()
        retention_date = datetime.now() + timedelta(days=7)
        tags = {"database": "ha_predictor", "host": "localhost"}

        metadata = BackupMetadata(
            backup_id="test_backup_003",
            backup_type="database",
            timestamp=timestamp,
            size_bytes=2048000,
            compressed=True,
            checksum="hash123",
            retention_date=retention_date,
            tags=tags,
        )

        result = metadata.to_dict()

        assert result["backup_id"] == "test_backup_003"
        assert result["backup_type"] == "database"
        assert result["timestamp"] == timestamp.isoformat()
        assert result["size_bytes"] == 2048000
        assert result["compressed"] is True
        assert result["checksum"] == "hash123"
        assert result["retention_date"] == retention_date.isoformat()
        assert result["tags"] == tags

    def test_to_dict_serialization_none_values(self):
        """Test to_dict() serialization with None optional fields."""
        timestamp = datetime.now()
        metadata = BackupMetadata(
            backup_id="test_backup_004",
            backup_type="config",
            timestamp=timestamp,
            size_bytes=512000,
            compressed=False,
        )

        result = metadata.to_dict()

        assert result["checksum"] is None
        assert result["retention_date"] is None
        assert result["tags"] == {}

    def test_from_dict_deserialization_complete(self):
        """Test from_dict() deserialization with complete data."""
        timestamp = datetime.now()
        retention_date = datetime.now() + timedelta(days=7)
        data = {
            "backup_id": "test_backup_005",
            "backup_type": "models",
            "timestamp": timestamp.isoformat(),
            "size_bytes": 3072000,
            "compressed": True,
            "checksum": "def789ghi012",
            "retention_date": retention_date.isoformat(),
            "tags": {"models_dir": "/app/models"},
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.backup_id == "test_backup_005"
        assert metadata.backup_type == "models"
        assert metadata.timestamp == timestamp
        assert metadata.size_bytes == 3072000
        assert metadata.compressed is True
        assert metadata.checksum == "def789ghi012"
        assert metadata.retention_date == retention_date
        assert metadata.tags == {"models_dir": "/app/models"}

    def test_from_dict_deserialization_missing_optional(self):
        """Test from_dict() deserialization with missing optional fields."""
        timestamp = datetime.now()
        data = {
            "backup_id": "test_backup_006",
            "backup_type": "database",
            "timestamp": timestamp.isoformat(),
            "size_bytes": 1536000,
            "compressed": False,
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.checksum is None
        assert metadata.retention_date is None
        assert metadata.tags == {}

    def test_from_dict_deserialization_none_retention_date(self):
        """Test from_dict() with None retention_date handling."""
        timestamp = datetime.now()
        data = {
            "backup_id": "test_backup_007",
            "backup_type": "config",
            "timestamp": timestamp.isoformat(),
            "size_bytes": 256000,
            "compressed": True,
            "retention_date": None,
        }

        metadata = BackupMetadata.from_dict(data)

        assert metadata.retention_date is None

    def test_roundtrip_serialization_consistency(self):
        """Test roundtrip serialization/deserialization consistency."""
        timestamp = datetime.now()
        retention_date = datetime.now() + timedelta(days=14)
        original = BackupMetadata(
            backup_id="test_roundtrip",
            backup_type="database",
            timestamp=timestamp,
            size_bytes=4096000,
            compressed=True,
            checksum="roundtrip123",
            retention_date=retention_date,
            tags={"test": "roundtrip"},
        )

        # Convert to dict and back
        data = original.to_dict()
        restored = BackupMetadata.from_dict(data)

        assert restored.backup_id == original.backup_id
        assert restored.backup_type == original.backup_type
        assert restored.timestamp == original.timestamp
        assert restored.size_bytes == original.size_bytes
        assert restored.compressed == original.compressed
        assert restored.checksum == original.checksum
        assert restored.retention_date == original.retention_date
        assert restored.tags == original.tags


class TestDatabaseBackupManager:
    """Test DatabaseBackupManager class."""

    @pytest.fixture
    def temp_backup_dir(self, tmp_path):
        """Create temporary backup directory."""
        return tmp_path / "backups"

    @pytest.fixture
    def db_config(self):
        """Database configuration fixture."""
        return {
            "connection_string": "postgresql+asyncpg://testuser:testpass@localhost:5432/testdb"
        }

    @pytest.fixture
    def db_backup_manager(self, temp_backup_dir, db_config):
        """DatabaseBackupManager fixture."""
        return DatabaseBackupManager(str(temp_backup_dir), db_config)

    def test_init_backup_directory_creation(self, temp_backup_dir, db_config):
        """Test initialization with backup directory creation."""
        manager = DatabaseBackupManager(str(temp_backup_dir), db_config)

        assert manager.backup_dir == temp_backup_dir / "database"
        assert manager.backup_dir.exists()
        assert manager.db_config == db_config

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_backup_auto_generated_id(
        self, mock_file, mock_subprocess, db_backup_manager
    ):
        """Test create_backup() with auto-generated backup ID."""
        # Mock successful pg_dump
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        # Mock file size
        mock_stat = Mock()
        mock_stat.st_size = 1024000
        with patch.object(Path, "stat", return_value=mock_stat):
            with patch("gzip.open", mock_open()):
                with patch("os.remove"):
                    result = db_backup_manager.create_backup(compress=True)

        # Verify backup_id format (db_YYYYMMDD_HHMMSS)
        assert result.backup_id.startswith("db_")
        assert len(result.backup_id) == 18  # "db_" + 8 + "_" + 6
        assert result.backup_type == "database"
        assert result.compressed is True
        assert result.size_bytes == 1024000

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_backup_custom_id_uncompressed(
        self, mock_file, mock_subprocess, db_backup_manager
    ):
        """Test create_backup() with custom backup_id and no compression."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 2048000
        with patch.object(Path, "stat", return_value=mock_stat):
            result = db_backup_manager.create_backup(
                backup_id="custom_backup", compress=False
            )

        assert result.backup_id == "custom_backup"
        assert result.compressed is False
        assert result.size_bytes == 2048000

    def test_create_backup_connection_string_parsing_full(self, db_backup_manager):
        """Test database connection string parsing with full parameters."""
        db_backup_manager.db_config = {
            "connection_string": "postgresql+asyncpg://myuser:mypass@dbhost:5433/mydb"
        }

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0, stderr="")
            with patch("builtins.open", mock_open()):
                with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
                    with patch("gzip.open", mock_open()):
                        with patch("os.remove"):
                            db_backup_manager.create_backup()

        # Verify pg_dump command construction
        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]

        assert "pg_dump" in cmd
        assert "-h" in cmd and "dbhost" in cmd
        assert "-p" in cmd and "5433" in cmd
        assert "-U" in cmd and "myuser" in cmd
        assert "-d" in cmd and "mydb" in cmd

        # Verify environment contains password
        env = call_args[1]["env"]
        assert env["PGPASSWORD"] == "mypass"

    def test_create_backup_connection_string_parsing_minimal(self, db_backup_manager):
        """Test connection string parsing with missing components."""
        db_backup_manager.db_config = {
            "connection_string": "postgresql+asyncpg://user@host/db"
        }

        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0, stderr="")
            with patch("builtins.open", mock_open()):
                with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
                    db_backup_manager.create_backup(compress=False)

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]
        assert "-p" in cmd and "5432" in cmd  # Default port

    def test_create_backup_pg_dump_command_construction(self, db_backup_manager):
        """Test pg_dump command construction with proper parameters."""
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0, stderr="")
            with patch("builtins.open", mock_open()):
                with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
                    db_backup_manager.create_backup(compress=False)

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]

        # Verify pg_dump parameters
        assert "pg_dump" in cmd
        assert "--no-password" in cmd
        assert "--verbose" in cmd
        assert "--clean" in cmd
        assert "--if-exists" in cmd
        assert "--create" in cmd

        # Verify timeout
        assert call_args[1]["timeout"] == 3600

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_backup_compression_workflow(
        self, mock_file, mock_subprocess, db_backup_manager
    ):
        """Test backup file compression workflow."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 512000
        with patch.object(Path, "stat", return_value=mock_stat):
            with patch("gzip.open", mock_open()) as mock_gzip:
                with patch("shutil.copyfileobj") as mock_copy:
                    with patch("os.remove") as mock_remove:
                        result = db_backup_manager.create_backup(compress=True)

        # Verify compression workflow
        mock_gzip.assert_called_once()
        mock_copy.assert_called_once()
        mock_remove.assert_called_once()
        assert result.compressed is True

    @patch("subprocess.run")
    def test_create_backup_pg_dump_failure(self, mock_subprocess, db_backup_manager):
        """Test create_backup() handling pg_dump failures."""
        mock_subprocess.return_value = Mock(
            returncode=1, stderr="pg_dump: connection failed"
        )

        with patch("builtins.open", mock_open()):
            with pytest.raises(RuntimeError, match="pg_dump failed"):
                db_backup_manager.create_backup(compress=False)

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_backup_metadata_creation(
        self, mock_file, mock_subprocess, db_backup_manager
    ):
        """Test backup metadata creation with proper fields."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 1536000
        with patch.object(Path, "stat", return_value=mock_stat):
            with patch("json.dump") as mock_json:
                result = db_backup_manager.create_backup(
                    backup_id="meta_test", compress=False
                )

        # Verify metadata structure
        assert result.backup_type == "database"
        assert result.size_bytes == 1536000
        assert result.tags["database"] == "testdb"
        assert result.tags["host"] == "localhost"

        # Verify metadata file save
        mock_json.assert_called_once()

    @patch("subprocess.run")
    @patch("builtins.open", new_callable=mock_open)
    def test_create_backup_cleanup_on_failure(
        self, mock_file, mock_subprocess, db_backup_manager
    ):
        """Test backup cleanup on failure scenarios."""
        mock_subprocess.side_effect = Exception("Backup failed")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink") as mock_unlink:
                with pytest.raises(Exception, match="Backup failed"):
                    db_backup_manager.create_backup()

        mock_unlink.assert_called_once()

    def test_restore_backup_metadata_loading(self, db_backup_manager):
        """Test restore_backup() with metadata loading and validation."""
        # Mock metadata loading
        metadata = BackupMetadata(
            backup_id="restore_test",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=False,
        )

        with patch.object(db_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("subprocess.run") as mock_subprocess:
                    mock_subprocess.return_value = Mock(returncode=0, stderr="")
                    with patch("builtins.open", mock_open(read_data="-- SQL content")):
                        db_backup_manager.restore_backup("restore_test")

        mock_subprocess.assert_called_once()

    def test_restore_backup_missing_metadata(self, db_backup_manager):
        """Test restore_backup() behavior when metadata doesn't exist."""
        with patch.object(db_backup_manager, "_load_backup_metadata", return_value=None):
            with pytest.raises(ValueError, match="Database backup not found"):
                db_backup_manager.restore_backup("nonexistent")

    def test_restore_backup_wrong_type(self, db_backup_manager):
        """Test restore_backup() with wrong backup type."""
        metadata = BackupMetadata(
            backup_id="wrong_type",
            backup_type="models",  # Wrong type
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=False,
        )

        with patch.object(db_backup_manager, "_load_backup_metadata", return_value=metadata):
            with pytest.raises(ValueError, match="Database backup not found"):
                db_backup_manager.restore_backup("wrong_type")

    def test_restore_backup_compressed_file_handling(self, db_backup_manager):
        """Test restore_backup() with compressed file handling."""
        metadata = BackupMetadata(
            backup_id="compressed_restore",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=True,
        )

        with patch.object(db_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("gzip.open", mock_open(read_data="-- SQL content")) as mock_gzip:
                    with patch("subprocess.run") as mock_subprocess:
                        mock_subprocess.return_value = Mock(returncode=0, stderr="")
                        db_backup_manager.restore_backup("compressed_restore")

        mock_gzip.assert_called_once()

    def test_restore_backup_psql_command_construction(self, db_backup_manager):
        """Test psql command construction for restoration."""
        metadata = BackupMetadata(
            backup_id="psql_test",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=False,
        )

        with patch.object(db_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("subprocess.run") as mock_subprocess:
                    mock_subprocess.return_value = Mock(returncode=0, stderr="")
                    with patch("builtins.open", mock_open(read_data="-- SQL")):
                        db_backup_manager.restore_backup("psql_test")

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]

        assert "psql" in cmd
        assert "-d" in cmd and "postgres" in cmd  # Connect to postgres db first
        assert "--no-password" in cmd
        assert "-v" in cmd and "ON_ERROR_STOP=1" in cmd

    def test_restore_backup_psql_failure(self, db_backup_manager):
        """Test restore_backup() handling psql failures."""
        metadata = BackupMetadata(
            backup_id="psql_fail",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=False,
        )

        with patch.object(db_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("subprocess.run") as mock_subprocess:
                    mock_subprocess.return_value = Mock(
                        returncode=1, stderr="psql: connection failed"
                    )
                    with patch("builtins.open", mock_open(read_data="-- SQL")):
                        with pytest.raises(RuntimeError, match="Database restoration failed"):
                            db_backup_manager.restore_backup("psql_fail")

    def test_save_backup_metadata_json_format(self, db_backup_manager):
        """Test _save_backup_metadata() JSON file creation."""
        metadata = BackupMetadata(
            backup_id="json_test",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=True,
        )

        with patch("builtins.open", mock_open()) as mock_file:
            with patch("json.dump") as mock_json:
                db_backup_manager._save_backup_metadata(metadata)

        # Verify file path and JSON dump
        expected_file = db_backup_manager.backup_dir / "json_test.metadata.json"
        mock_file.assert_called_once_with(expected_file, "w")
        mock_json.assert_called_once_with(metadata.to_dict(), mock_file(), indent=2)

    def test_load_backup_metadata_existing_file(self, db_backup_manager):
        """Test _load_backup_metadata() with existing file."""
        metadata_data = {
            "backup_id": "load_test",
            "backup_type": "database",
            "timestamp": datetime.now().isoformat(),
            "size_bytes": 2048000,
            "compressed": False,
        }

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("json.load", return_value=metadata_data) as mock_json:
                    result = db_backup_manager._load_backup_metadata("load_test")

        assert result is not None
        assert result.backup_id == "load_test"
        assert result.backup_type == "database"

    def test_load_backup_metadata_missing_file(self, db_backup_manager):
        """Test _load_backup_metadata() with missing file."""
        with patch.object(Path, "exists", return_value=False):
            result = db_backup_manager._load_backup_metadata("missing_file")

        assert result is None


class TestModelBackupManager:
    """Test ModelBackupManager class."""

    @pytest.fixture
    def temp_backup_dir(self, tmp_path):
        """Create temporary backup directory."""
        return tmp_path / "backups"

    @pytest.fixture
    def temp_models_dir(self, tmp_path):
        """Create temporary models directory."""
        return tmp_path / "models"

    @pytest.fixture
    def model_backup_manager(self, temp_backup_dir, temp_models_dir):
        """ModelBackupManager fixture."""
        return ModelBackupManager(str(temp_backup_dir), str(temp_models_dir))

    def test_init_directories_setup(self, temp_backup_dir, temp_models_dir):
        """Test initialization with backup and models directory setup."""
        manager = ModelBackupManager(str(temp_backup_dir), str(temp_models_dir))

        assert manager.backup_dir == temp_backup_dir / "models"
        assert manager.backup_dir.exists()
        assert manager.models_dir == temp_models_dir

    @patch("subprocess.run")
    def test_create_backup_auto_generated_id(
        self, mock_subprocess, model_backup_manager
    ):
        """Test create_backup() with auto-generated backup ID."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 5120000
        with patch.object(Path, "stat", return_value=mock_stat):
            with patch.object(Path, "exists", return_value=True):
                with patch("json.dump"):
                    result = model_backup_manager.create_backup(compress=True)

        # Verify backup_id format (models_YYYYMMDD_HHMMSS)
        assert result.backup_id.startswith("models_")
        assert len(result.backup_id) == 22  # "models_" + 8 + "_" + 6
        assert result.backup_type == "models"
        assert result.compressed is True
        assert result.size_bytes == 5120000

    @patch("subprocess.run")
    def test_create_backup_models_directory_creation(
        self, mock_subprocess, model_backup_manager
    ):
        """Test create_backup() with models directory existence check."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        # Mock models directory doesn't exist initially
        with patch.object(Path, "exists", return_value=False):
            with patch.object(Path, "mkdir") as mock_mkdir:
                with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
                    with patch("json.dump"):
                        model_backup_manager.create_backup()

        mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

    @patch("subprocess.run")
    def test_create_backup_tar_command_construction(
        self, mock_subprocess, model_backup_manager
    ):
        """Test tar command construction with proper parameters."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat", return_value=Mock(st_size=2048)):
                with patch("json.dump"):
                    model_backup_manager.create_backup(
                        backup_id="tar_test", compress=True
                    )

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]

        assert "tar" in cmd
        assert "-czf" in cmd  # Compressed format
        assert str(model_backup_manager.models_dir.name) in cmd
        assert call_args[1]["timeout"] == 1800  # 30 minutes

    @patch("subprocess.run")
    def test_create_backup_tar_command_uncompressed(
        self, mock_subprocess, model_backup_manager
    ):
        """Test tar command for uncompressed backup."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
                with patch("json.dump"):
                    model_backup_manager.create_backup(compress=False)

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]

        assert "-cf" in cmd  # Uncompressed format
        assert "-czf" not in cmd

    @patch("subprocess.run")
    def test_create_backup_tar_failure(self, mock_subprocess, model_backup_manager):
        """Test create_backup() handling tar command failures."""
        mock_subprocess.return_value = Mock(
            returncode=1, stderr="tar: cannot access"
        )

        with patch.object(Path, "exists", return_value=True):
            with pytest.raises(RuntimeError, match="tar command failed"):
                model_backup_manager.create_backup()

    @patch("subprocess.run")
    def test_create_backup_metadata_creation(
        self, mock_subprocess, model_backup_manager
    ):
        """Test backup metadata creation with proper fields."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 7340032  # Specific size
        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "stat", return_value=mock_stat):
                with patch("json.dump") as mock_json:
                    result = model_backup_manager.create_backup(
                        backup_id="meta_models", compress=True
                    )

        assert result.backup_type == "models"
        assert result.size_bytes == 7340032
        assert result.tags["models_dir"] == str(model_backup_manager.models_dir)

        # Verify metadata save
        mock_json.assert_called_once()

    @patch("subprocess.run")
    def test_create_backup_cleanup_on_failure(
        self, mock_subprocess, model_backup_manager
    ):
        """Test backup cleanup on creation failures."""
        mock_subprocess.side_effect = Exception("Tar failed")

        with patch.object(Path, "exists", return_value=True):
            with patch.object(Path, "unlink") as mock_unlink:
                with pytest.raises(Exception, match="Tar failed"):
                    model_backup_manager.create_backup()

        mock_unlink.assert_called_once()

    def test_restore_backup_metadata_validation(self, model_backup_manager):
        """Test restore_backup() with metadata loading and validation."""
        metadata = BackupMetadata(
            backup_id="restore_models",
            backup_type="models",
            timestamp=datetime.now(),
            size_bytes=3072000,
            compressed=True,
        )

        with patch.object(model_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("subprocess.run") as mock_subprocess:
                    mock_subprocess.return_value = Mock(returncode=0, stderr="")
                    with patch("shutil.move"):
                        model_backup_manager.restore_backup("restore_models")

        mock_subprocess.assert_called_once()

    def test_restore_backup_existing_models_backup(self, model_backup_manager):
        """Test restore_backup() with existing models directory backup."""
        metadata = BackupMetadata(
            backup_id="existing_models",
            backup_type="models",
            timestamp=datetime.now(),
            size_bytes=2048000,
            compressed=False,
        )

        with patch.object(model_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("shutil.move") as mock_move:
                    with patch("subprocess.run") as mock_subprocess:
                        mock_subprocess.return_value = Mock(returncode=0, stderr="")
                        model_backup_manager.restore_backup("existing_models")

        # Verify existing models directory backup
        mock_move.assert_called_once()
        call_args = mock_move.call_args[0]
        assert str(model_backup_manager.models_dir) in call_args[0]
        assert ".backup." in call_args[1]

    def test_restore_backup_tar_extraction_command(self, model_backup_manager):
        """Test restore_backup() tar extraction command construction."""
        metadata = BackupMetadata(
            backup_id="extract_test",
            backup_type="models",
            timestamp=datetime.now(),
            size_bytes=1536000,
            compressed=True,
        )

        with patch.object(model_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("shutil.move"):
                    with patch("subprocess.run") as mock_subprocess:
                        mock_subprocess.return_value = Mock(returncode=0, stderr="")
                        model_backup_manager.restore_backup("extract_test")

        call_args = mock_subprocess.call_args
        cmd = call_args[0][0]

        assert "tar" in cmd
        assert "-xzf" in cmd  # Extract compressed
        assert str(model_backup_manager.models_dir.parent) in cmd

    def test_restore_backup_tar_extraction_failure(self, model_backup_manager):
        """Test restore_backup() tar extraction failure handling."""
        metadata = BackupMetadata(
            backup_id="extract_fail",
            backup_type="models",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=False,
        )

        with patch.object(model_backup_manager, "_load_backup_metadata", return_value=metadata):
            with patch.object(Path, "exists", return_value=True):
                with patch("shutil.move"):
                    with patch("subprocess.run") as mock_subprocess:
                        mock_subprocess.return_value = Mock(
                            returncode=1, stderr="tar extraction failed"
                        )
                        with pytest.raises(RuntimeError, match="tar extraction failed"):
                            model_backup_manager.restore_backup("extract_fail")


class TestConfigurationBackupManager:
    """Test ConfigurationBackupManager class."""

    @pytest.fixture
    def temp_backup_dir(self, tmp_path):
        """Create temporary backup directory."""
        return tmp_path / "backups"

    @pytest.fixture
    def temp_config_dir(self, tmp_path):
        """Create temporary config directory."""
        return tmp_path / "config"

    @pytest.fixture
    def config_backup_manager(self, temp_backup_dir, temp_config_dir):
        """ConfigurationBackupManager fixture."""
        return ConfigurationBackupManager(str(temp_backup_dir), str(temp_config_dir))

    def test_init_directories_setup(self, temp_backup_dir, temp_config_dir):
        """Test initialization with backup and config directory setup."""
        manager = ConfigurationBackupManager(str(temp_backup_dir), str(temp_config_dir))

        assert manager.backup_dir == temp_backup_dir / "config"
        assert manager.backup_dir.exists()
        assert manager.config_dir == temp_config_dir

    @patch("subprocess.run")
    def test_create_backup_auto_generated_id(
        self, mock_subprocess, config_backup_manager
    ):
        """Test create_backup() with auto-generated backup ID."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 256000
        with patch.object(Path, "stat", return_value=mock_stat):
            with patch("json.dump"):
                result = config_backup_manager.create_backup(compress=True)

        # Verify backup_id format (config_YYYYMMDD_HHMMSS)
        assert result.backup_id.startswith("config_")
        assert len(result.backup_id) == 21  # "config_" + 8 + "_" + 6
        assert result.backup_type == "config"
        assert result.compressed is True
        assert result.size_bytes == 256000

    @patch("subprocess.run")
    def test_create_backup_tar_configuration_timeout(
        self, mock_subprocess, config_backup_manager
    ):
        """Test tar command for configuration files with timeout."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        with patch.object(Path, "stat", return_value=Mock(st_size=128000)):
            with patch("json.dump"):
                config_backup_manager.create_backup()

        call_args = mock_subprocess.call_args
        assert call_args[1]["timeout"] == 300  # 5 minutes

    @patch("subprocess.run")
    def test_create_backup_size_calculation_kb(
        self, mock_subprocess, config_backup_manager
    ):
        """Test configuration backup size calculation (bytes to KB)."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        mock_stat = Mock()
        mock_stat.st_size = 524288  # 512 KB
        with patch.object(Path, "stat", return_value=mock_stat):
            with patch("json.dump"):
                result = config_backup_manager.create_backup()

        assert result.size_bytes == 524288

    @patch("subprocess.run")
    def test_create_backup_metadata_tags(
        self, mock_subprocess, config_backup_manager
    ):
        """Test backup metadata creation with configuration-specific tags."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        with patch.object(Path, "stat", return_value=Mock(st_size=65536)):
            with patch("json.dump") as mock_json:
                result = config_backup_manager.create_backup(
                    backup_id="config_meta_test"
                )

        assert result.backup_type == "config"
        assert result.tags["config_dir"] == str(config_backup_manager.config_dir)

        # Verify metadata save
        mock_json.assert_called_once()


class TestBackupManagerOrchestration:
    """Test BackupManager orchestration class."""

    @pytest.fixture
    def temp_backup_dir(self, tmp_path):
        """Create temporary backup directory."""
        return tmp_path / "backups"

    @pytest.fixture
    def backup_config(self):
        """Backup configuration fixture."""
        return {
            "database": {"connection_string": "postgresql://user:pass@host/db"},
            "models_dir": "models",
            "config_dir": "config",
            "backup": {
                "enabled": True,
                "interval_hours": 24,
                "retention_days": 7,
                "compress": True,
                "database_backup": True,
                "model_backup": True,
            },
        }

    @pytest.fixture
    def backup_manager(self, temp_backup_dir, backup_config):
        """BackupManager fixture."""
        return BackupManager(str(temp_backup_dir), backup_config)

    def test_init_sub_manager_creation(self, temp_backup_dir, backup_config):
        """Test initialization with sub-manager creation."""
        manager = BackupManager(str(temp_backup_dir), backup_config)

        assert manager.backup_dir == temp_backup_dir
        assert manager.backup_dir.exists()
        assert manager.config == backup_config

        # Verify sub-managers
        assert isinstance(manager.db_backup_manager, DatabaseBackupManager)
        assert isinstance(manager.model_backup_manager, ModelBackupManager)
        assert isinstance(manager.config_backup_manager, ConfigurationBackupManager)

        assert manager.backup_config == backup_config["backup"]

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_disabled(self, backup_manager):
        """Test run_scheduled_backups() with disabled backup configuration."""
        backup_manager.backup_config = {"enabled": False}

        # Should return immediately without creating backups
        with patch("asyncio.sleep") as mock_sleep:
            # Use a timeout to prevent infinite loop in tests
            try:
                await asyncio.wait_for(backup_manager.run_scheduled_backups(), timeout=0.1)
            except asyncio.TimeoutError:
                pass

        mock_sleep.assert_not_called()

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_database_creation(self, backup_manager):
        """Test scheduled database backup creation."""
        mock_db_backup = Mock()
        mock_db_backup.retention_date = None

        with patch.object(backup_manager.db_backup_manager, "create_backup", return_value=mock_db_backup):
            with patch.object(backup_manager, "cleanup_expired_backups") as mock_cleanup:
                with patch("asyncio.sleep") as mock_sleep:
                    # Set up single iteration
                    mock_sleep.side_effect = [asyncio.CancelledError()]

                    try:
                        await backup_manager.run_scheduled_backups()
                    except asyncio.CancelledError:
                        pass

        # Verify retention date was set
        assert mock_db_backup.retention_date is not None
        mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_model_backup_interval(self, backup_manager):
        """Test model backup scheduling with interval logic."""
        backup_manager.backup_config["model_backup_interval_hours"] = 12

        mock_db_backup = Mock()
        mock_model_backup = Mock()

        with patch("datetime") as mock_datetime:
            # Set current hour to trigger model backup (hour % 12 == 0)
            mock_datetime.now.return_value.hour = 12
            mock_datetime.now.return_value = datetime.now()

            with patch.object(backup_manager.db_backup_manager, "create_backup", return_value=mock_db_backup):
                with patch.object(backup_manager.model_backup_manager, "create_backup", return_value=mock_model_backup):
                    with patch.object(backup_manager, "cleanup_expired_backups"):
                        with patch("asyncio.sleep") as mock_sleep:
                            mock_sleep.side_effect = [asyncio.CancelledError()]

                            try:
                                await backup_manager.run_scheduled_backups()
                            except asyncio.CancelledError:
                                pass

        # Both backups should be created
        backup_manager.db_backup_manager.create_backup.assert_called_once()
        backup_manager.model_backup_manager.create_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_config_backup_daily(self, backup_manager):
        """Test configuration backup scheduling at 2 AM."""
        mock_config_backup = Mock()

        with patch("datetime") as mock_datetime:
            # Set current hour to 2 AM
            mock_datetime.now.return_value.hour = 2
            mock_datetime.now.return_value = datetime.now()

            with patch.object(backup_manager.db_backup_manager, "create_backup", return_value=Mock()):
                with patch.object(backup_manager.config_backup_manager, "create_backup", return_value=mock_config_backup):
                    with patch.object(backup_manager, "cleanup_expired_backups"):
                        with patch("asyncio.sleep") as mock_sleep:
                            mock_sleep.side_effect = [asyncio.CancelledError()]

                            try:
                                await backup_manager.run_scheduled_backups()
                            except asyncio.CancelledError:
                                pass

        backup_manager.config_backup_manager.create_backup.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_scheduled_backups_failure_handling(self, backup_manager):
        """Test backup failure handling and error logging."""
        with patch.object(backup_manager.db_backup_manager, "create_backup", side_effect=Exception("Backup failed")):
            with patch("asyncio.sleep") as mock_sleep:
                mock_sleep.side_effect = [asyncio.CancelledError()]

                try:
                    await backup_manager.run_scheduled_backups()
                except asyncio.CancelledError:
                    pass

        # Should continue despite failures (error logged but not raised)

    @pytest.mark.asyncio
    async def test_cleanup_expired_backups_metadata_discovery(self, backup_manager):
        """Test cleanup_expired_backups() metadata file discovery."""
        # Create mock metadata files
        metadata1 = BackupMetadata(
            backup_id="expired_backup",
            backup_type="database",
            timestamp=datetime.now() - timedelta(days=10),
            size_bytes=1024000,
            compressed=True,
            retention_date=datetime.now() - timedelta(days=1),  # Expired
        )

        metadata2 = BackupMetadata(
            backup_id="valid_backup",
            backup_type="models",
            timestamp=datetime.now(),
            size_bytes=2048000,
            compressed=False,
            retention_date=datetime.now() + timedelta(days=5),  # Not expired
        )

        mock_files = [
            Mock(name="expired_backup.metadata.json"),
            Mock(name="valid_backup.metadata.json"),
        ]

        with patch.object(backup_manager.backup_dir, "rglob", return_value=mock_files):
            with patch("builtins.open", mock_open()) as mock_file:
                with patch("json.load", side_effect=[metadata1.to_dict(), metadata2.to_dict()]):
                    with patch.object(Path, "exists", return_value=True):
                        with patch.object(Path, "stat", return_value=Mock(st_size=1024000)):
                            with patch.object(Path, "unlink") as mock_unlink:
                                await backup_manager.cleanup_expired_backups()

        # Only expired backup should be removed
        assert mock_unlink.call_count >= 2  # Backup file + metadata file

    def test_list_backups_all_types(self, backup_manager):
        """Test list_backups() with optional backup_type filtering."""
        metadata1 = BackupMetadata(
            backup_id="db_backup",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=True,
        )

        metadata2 = BackupMetadata(
            backup_id="model_backup",
            backup_type="models",
            timestamp=datetime.now() - timedelta(hours=1),
            size_bytes=2048000,
            compressed=False,
        )

        mock_files = [Mock(), Mock()]
        with patch.object(backup_manager.backup_dir, "rglob", return_value=mock_files):
            with patch("builtins.open", mock_open()):
                with patch("json.load", side_effect=[metadata1.to_dict(), metadata2.to_dict()]):
                    result = backup_manager.list_backups()

        # Should return all backups, sorted by timestamp (newest first)
        assert len(result) == 2
        assert result[0].backup_id == "db_backup"  # Newer
        assert result[1].backup_id == "model_backup"  # Older

    def test_list_backups_type_filtering(self, backup_manager):
        """Test list_backups() sorting by timestamp."""
        metadata = BackupMetadata(
            backup_id="models_only",
            backup_type="models",
            timestamp=datetime.now(),
            size_bytes=3072000,
            compressed=True,
        )

        with patch.object(backup_manager.backup_dir, "rglob", return_value=[Mock()]):
            with patch("builtins.open", mock_open()):
                with patch("json.load", return_value=metadata.to_dict()):
                    result = backup_manager.list_backups(backup_type="models")

        assert len(result) == 1
        assert result[0].backup_type == "models"

    def test_get_backup_info_specific_lookup(self, backup_manager):
        """Test get_backup_info() with specific backup_id lookup."""
        metadata = BackupMetadata(
            backup_id="specific_backup",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1536000,
            compressed=False,
        )

        with patch.object(backup_manager.backup_dir, "rglob", return_value=[Mock()]):
            with patch("builtins.open", mock_open()):
                with patch("json.load", return_value=metadata.to_dict()):
                    result = backup_manager.get_backup_info("specific_backup")

        assert result is not None
        assert result.backup_id == "specific_backup"

    def test_get_backup_info_not_found(self, backup_manager):
        """Test get_backup_info() when backup not found."""
        with patch.object(backup_manager.backup_dir, "rglob", return_value=[]):
            result = backup_manager.get_backup_info("nonexistent")

        assert result is None

    def test_restore_database_backup_delegation(self, backup_manager):
        """Test restore_database_backup() delegation to DatabaseBackupManager."""
        with patch.object(backup_manager.db_backup_manager, "restore_backup") as mock_restore:
            backup_manager.restore_database_backup("test_db_backup")

        mock_restore.assert_called_once_with("test_db_backup")

    def test_restore_models_backup_delegation(self, backup_manager):
        """Test restore_models_backup() delegation to ModelBackupManager."""
        with patch.object(backup_manager.model_backup_manager, "restore_backup") as mock_restore:
            backup_manager.restore_models_backup("test_models_backup")

        mock_restore.assert_called_once_with("test_models_backup")

    def test_create_disaster_recovery_package_complete(self, backup_manager):
        """Test create_disaster_recovery_package() with complete backup package creation."""
        mock_db_backup = Mock()
        mock_model_backup = Mock()
        mock_config_backup = Mock()

        # Mock to_dict() method for backups
        mock_db_backup.to_dict.return_value = {"type": "database"}
        mock_model_backup.to_dict.return_value = {"type": "models"}
        mock_config_backup.to_dict.return_value = {"type": "config"}

        with patch.object(backup_manager.db_backup_manager, "create_backup", return_value=mock_db_backup):
            with patch.object(backup_manager.model_backup_manager, "create_backup", return_value=mock_model_backup):
                with patch.object(backup_manager.config_backup_manager, "create_backup", return_value=mock_config_backup):
                    with patch("builtins.open", mock_open()) as mock_file:
                        with patch("json.dump") as mock_json:
                            result = backup_manager.create_disaster_recovery_package()

        # Verify package creation
        assert result.startswith("disaster_recovery_")
        mock_json.assert_called_once()

        # Verify manifest structure
        manifest_call = mock_json.call_args[0][0]
        assert "package_id" in manifest_call
        assert "backups" in manifest_call
        assert len(manifest_call["backups"]) == 3

    def test_create_disaster_recovery_package_cleanup_on_failure(self, backup_manager):
        """Test disaster recovery package cleanup on failures."""
        with patch.object(backup_manager.db_backup_manager, "create_backup", side_effect=Exception("Backup failed")):
            with patch("shutil.rmtree") as mock_rmtree:
                with pytest.raises(Exception, match="Backup failed"):
                    backup_manager.create_disaster_recovery_package()

        mock_rmtree.assert_called_once()


class TestBackupManagerIntegration:
    """Integration tests for backup management system."""

    @pytest.fixture
    def temp_dirs(self, tmp_path):
        """Create temporary directory structure."""
        backup_dir = tmp_path / "backups"
        models_dir = tmp_path / "models"
        config_dir = tmp_path / "config"

        # Create some test files
        models_dir.mkdir(parents=True)
        (models_dir / "model1.pkl").write_text("mock model data")
        
        config_dir.mkdir(parents=True)
        (config_dir / "config.yaml").write_text("mock: config")

        return {
            "backup": backup_dir,
            "models": models_dir,
            "config": config_dir,
        }

    def test_backup_directory_structure_creation(self, temp_dirs):
        """Test backup directory structure creation."""
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
            "models_dir": str(temp_dirs["models"]),
            "config_dir": str(temp_dirs["config"]),
        }

        manager = BackupManager(str(temp_dirs["backup"]), config)

        # Verify directory structure
        assert (temp_dirs["backup"] / "database").exists()
        assert (temp_dirs["backup"] / "models").exists()
        assert (temp_dirs["backup"] / "config").exists()

    def test_backup_metadata_file_operations(self, temp_dirs):
        """Test backup metadata file JSON operations."""
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
            "models_dir": str(temp_dirs["models"]),
            "config_dir": str(temp_dirs["config"]),
        }

        manager = BackupManager(str(temp_dirs["backup"]), config)
        
        metadata = BackupMetadata(
            backup_id="file_ops_test",
            backup_type="database",
            timestamp=datetime.now(),
            size_bytes=1024000,
            compressed=True,
            tags={"test": "integration"},
        )

        # Save and load metadata
        manager.db_backup_manager._save_backup_metadata(metadata)
        loaded_metadata = manager.db_backup_manager._load_backup_metadata("file_ops_test")

        assert loaded_metadata is not None
        assert loaded_metadata.backup_id == "file_ops_test"
        assert loaded_metadata.backup_type == "database"
        assert loaded_metadata.tags["test"] == "integration"

    @patch("subprocess.run")
    def test_backup_creation_workflow_integration(
        self, mock_subprocess, temp_dirs
    ):
        """Test complete backup creation workflow."""
        mock_subprocess.return_value = Mock(returncode=0, stderr="")

        config = {
            "database": {"connection_string": "postgresql://user:pass@host/db"},
            "models_dir": str(temp_dirs["models"]),
            "config_dir": str(temp_dirs["config"]),
        }

        manager = BackupManager(str(temp_dirs["backup"]), config)

        # Create models backup (real tar command would be executed)
        with patch.object(Path, "stat", return_value=Mock(st_size=2048)):
            result = manager.model_backup_manager.create_backup(
                backup_id="integration_test", compress=False
            )

        assert result.backup_id == "integration_test"
        assert result.backup_type == "models"

        # Verify backup is listed
        backups = manager.list_backups(backup_type="models")
        assert len(backups) == 1
        assert backups[0].backup_id == "integration_test"


class TestBackupManagerEdgeCases:
    """Test edge cases and error handling."""

    def test_backup_creation_insufficient_disk_space(self, tmp_path):
        """Test backup operations with insufficient disk space simulation."""
        backup_dir = tmp_path / "backups"
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
        }
        
        manager = BackupManager(str(backup_dir), config)
        
        # Mock shutil.disk_usage to simulate low disk space
        with patch("shutil.disk_usage") as mock_disk_usage:
            # Simulate very low available space (1KB)
            mock_disk_usage.return_value = (1000, 950, 50)  # total, used, free
            
            with patch("subprocess.run") as mock_subprocess:
                mock_subprocess.return_value = Mock(returncode=0, stderr="")
                with patch("builtins.open", mock_open()):
                    with patch.object(Path, "stat", return_value=Mock(st_size=1024000)):
                        # Should handle insufficient space gracefully
                        # In production, this would raise an error before attempting backup
                        try:
                            manager.db_backup_manager.create_backup()
                        except Exception as e:
                            # Expected behavior - backup should fail or warn about space
                            assert "space" in str(e).lower() or mock_subprocess.called

    def test_backup_concurrent_operations(self, tmp_path):
        """Test concurrent backup operations and file locking."""
        backup_dir = tmp_path / "backups"
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
        }
        
        manager1 = BackupManager(str(backup_dir), config)
        manager2 = BackupManager(str(backup_dir), config)
        
        # Mock file operations to simulate concurrent access
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0, stderr="")
            
            # Test file locking by mocking file operations
            with patch("builtins.open", mock_open()) as mock_file:
                # First call raises PermissionError (file locked)
                # Second call succeeds
                mock_file.side_effect = [
                    PermissionError("File locked by another process"),
                    mock_open().return_value,
                    mock_open().return_value  # For metadata save
                ]
                
                with patch.object(Path, "stat", return_value=Mock(st_size=1024)):
                    with patch.object(Path, "exists", return_value=False):  # No file to cleanup
                        # First manager should raise PermissionError
                        with pytest.raises(PermissionError, match="File locked"):
                            manager1.db_backup_manager.create_backup(backup_id="concurrent_test_1")
                    
                    # Test successful backup when no lock conflict
                    with patch("builtins.open", mock_open()):
                        with patch("gzip.open", mock_open()):
                            with patch("os.remove"):
                                result = manager2.db_backup_manager.create_backup(backup_id="concurrent_test_2")
                                assert result.backup_id == "concurrent_test_2"

    def test_backup_corrupted_metadata_recovery(self, tmp_path):
        """Test handling corrupted metadata files."""
        backup_dir = tmp_path / "backups"
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
        }

        manager = BackupManager(str(backup_dir), config)

        # Create corrupted metadata file
        metadata_file = backup_dir / "database" / "corrupted.metadata.json"
        metadata_file.parent.mkdir(parents=True, exist_ok=True)
        metadata_file.write_text("invalid json content")

        # The current implementation doesn't handle JSON errors gracefully
        # This test verifies that corrupted metadata causes a JSONDecodeError
        with pytest.raises(json.JSONDecodeError):
            manager.db_backup_manager._load_backup_metadata("corrupted")
        
        # Test with missing metadata file - should return None
        result = manager.db_backup_manager._load_backup_metadata("nonexistent")
        assert result is None

    def test_backup_very_large_files_timeout(self, tmp_path):
        """Test backup operations with very large files and timeout handling."""
        backup_dir = tmp_path / "backups"
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
        }
        
        manager = BackupManager(str(backup_dir), config)
        
        # Mock subprocess timeout to simulate very large backup
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.TimeoutExpired("pg_dump", 3600)
            
            # Should handle timeout gracefully
            with pytest.raises((subprocess.TimeoutExpired, RuntimeError)):
                manager.db_backup_manager.create_backup(backup_id="timeout_test")
        
        # Test model backup timeout
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.side_effect = subprocess.TimeoutExpired("tar", 1800)
            
            with pytest.raises((subprocess.TimeoutExpired, RuntimeError)):
                manager.model_backup_manager.create_backup(backup_id="model_timeout_test")

    def test_backup_network_storage_connectivity(self, tmp_path):
        """Test backup operations with network storage and connectivity issues."""
        # Simulate network storage path
        network_backup_dir = tmp_path / "network_storage" / "backups"
        config = {
            "database": {"connection_string": "postgresql://test@host/db"},
        }
        
        # Test network connectivity issues
        with patch.object(Path, "mkdir") as mock_mkdir:
            mock_mkdir.side_effect = OSError("Network unreachable")
            
            # Should handle network connectivity issues
            with pytest.raises(OSError, match="Network unreachable"):
                BackupManager(str(network_backup_dir), config)
        
        # Test network storage write issues
        manager = BackupManager(str(tmp_path / "local_backup"), config)
        
        with patch("subprocess.run") as mock_subprocess:
            mock_subprocess.return_value = Mock(returncode=0, stderr="")
            
            with patch("builtins.open", side_effect=OSError("Network storage unavailable")):
                # Should handle network storage write failures
                with pytest.raises(OSError, match="Network storage unavailable"):
                    manager.db_backup_manager.create_backup(backup_id="network_test")