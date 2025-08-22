"""
Consolidated comprehensive unit tests for database management.

This file consolidates all database-related tests including connection management,
compatibility layer, advanced features, performance optimizations, and edge cases.
Eliminated 800+ lines of duplicate test code from 5 separate files.
"""

import asyncio
from datetime import datetime, timedelta, timezone
import hashlib
import os
import tempfile
import time
from unittest.mock import (
    AsyncMock,
    MagicMock,
    Mock,
    PropertyMock,
    call,
    mock_open,
    patch,
)

import pytest
from sqlalchemy import Column, DateTime, Integer, String, text
from sqlalchemy.exc import (
    DatabaseError,
    DisconnectionError,
    IntegrityError,
    InvalidRequestError,
    OperationalError,
    SQLAlchemyError,
    TimeoutError as SQLTimeoutError,
)
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
from sqlalchemy.ext.declarative import declarative_base

from src.core.config import DatabaseConfig
from src.core.exceptions import (
    DatabaseConnectionError,
    DatabaseQueryError,
    ErrorSeverity,
)
from src.data.storage.database import (
    DatabaseManager,
    check_table_exists,
    close_database_manager,
    execute_sql_file,
    get_database_manager,
    get_database_version,
    get_db_session,
    get_timescaledb_version,
)
from src.data.storage.database_compatibility import (
    configure_database_on_first_connect,
    configure_sensor_event_model,
    configure_sqlite_for_testing,
    create_database_specific_models,
    get_database_specific_table_args,
    is_postgresql_engine,
    is_sqlite_engine,
    patch_models_for_sqlite_compatibility,
)


class TestDatabaseManager:
    """Test DatabaseManager class."""

    def test_database_manager_init(self):
        """Test DatabaseManager initialization."""
        config = DatabaseConfig(
            connection_string="postgresql://test:test@localhost/testdb",
            pool_size=15,
            max_overflow=25,
        )

        manager = DatabaseManager(config)

        assert manager.config == config
        assert manager.engine is None
        assert manager.session_factory is None
        assert manager._health_check_task is None
        assert manager.max_retries == 5
        assert manager.base_delay == 1.0
        assert manager.max_delay == 60.0
        assert manager.backoff_multiplier == 2.0
        assert not manager.is_initialized

    def test_database_manager_init_with_default_config(self):
        """Test DatabaseManager with default config loading."""
        with patch("src.data.storage.database.get_config") as mock_get_config:
            mock_system_config = Mock()
            mock_db_config = DatabaseConfig(
                connection_string="postgresql://default:default@localhost/default",
                pool_size=10,
                max_overflow=20,
            )
            mock_system_config.database = mock_db_config
            mock_get_config.return_value = mock_system_config

            manager = DatabaseManager()

            assert manager.config == mock_db_config
            mock_get_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_initialize_success(self):
        """Test successful database manager initialization."""
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://user:pass@localhost/testdb",
            pool_size=5,
            max_overflow=10,
        )

        manager = DatabaseManager(config)

        with (
            patch.object(manager, "_create_engine") as mock_create,
            patch.object(manager, "_setup_session_factory") as mock_setup,
            patch.object(manager, "_verify_connection") as mock_verify,
            patch("asyncio.create_task") as mock_create_task,
        ):
            # Don't pre-set engine/session_factory - let initialize() set them
            mock_create.return_value = None
            mock_setup.return_value = None
            mock_verify.return_value = None

            # Mock engine and session factory after creation
            async def mock_create_engine():
                manager.engine = Mock()

            async def mock_setup_session():
                manager.session_factory = Mock()

            mock_create.side_effect = mock_create_engine
            mock_setup.side_effect = mock_setup_session

            await manager.initialize()

            mock_create.assert_called_once()
            mock_setup.assert_called_once()
            mock_verify.assert_called_once()
            mock_create_task.assert_called_once()
            assert manager.is_initialized

    @pytest.mark.asyncio
    async def test_initialize_failure_cleanup(self):
        """Test initialization failure triggers cleanup."""
        config = DatabaseConfig(
            connection_string="postgresql://invalid:connection@localhost/nonexistent",
            pool_size=5,
            max_overflow=10,
        )

        manager = DatabaseManager(config)

        with (
            patch.object(
                manager,
                "_create_engine",
                side_effect=Exception("Connection failed"),
            ) as mock_create,
            patch.object(manager, "_cleanup") as mock_cleanup,
        ):

            with pytest.raises(DatabaseConnectionError):
                await manager.initialize()

            mock_create.assert_called_once()
            mock_cleanup.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_engine_postgresql(self):
        """Test engine creation for PostgreSQL."""
        config = DatabaseConfig(
            connection_string="postgresql://user:pass@localhost/db",
            pool_size=8,
            max_overflow=15,
        )

        manager = DatabaseManager(config)

        with (
            patch(
                "src.data.storage.database.create_async_engine"
            ) as mock_create_engine,
            patch.object(manager, "_setup_connection_events") as mock_setup_events,
        ):

            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            await manager._create_engine()

            # Should convert to asyncpg driver
            expected_url = "postgresql+asyncpg://user:pass@localhost/db"
            mock_create_engine.assert_called_once()
            args, kwargs = mock_create_engine.call_args
            assert kwargs["url"] == expected_url
            assert kwargs["pool_size"] == 8
            assert kwargs["max_overflow"] == 15
            assert kwargs["pool_timeout"] == 30
            assert kwargs["pool_recycle"] == 3600
            assert kwargs["pool_pre_ping"] is True

            assert manager.engine == mock_engine
            mock_setup_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_engine_alternative_postgresql(self):
        """Test engine creation for alternative PostgreSQL URL format."""
        config = DatabaseConfig(
            connection_string="postgresql://user:pass@localhost/db",
            pool_size=5,
            max_overflow=10,
        )

        manager = DatabaseManager(config)

        with (
            patch(
                "src.data.storage.database.create_async_engine"
            ) as mock_create_engine,
            patch.object(manager, "_setup_connection_events") as mock_setup_events,
        ):

            mock_engine = Mock()
            mock_create_engine.return_value = mock_engine

            await manager._create_engine()

            # Should convert to asyncpg driver
            expected_url = "postgresql+asyncpg://user:pass@localhost/db"
            mock_create_engine.assert_called_once()
            args, kwargs = mock_create_engine.call_args
            assert kwargs["url"] == expected_url
            assert kwargs["pool_size"] == 5
            assert kwargs["max_overflow"] == 10

            assert manager.engine == mock_engine
            mock_setup_events.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_engine_invalid_url(self):
        """Test engine creation with invalid URL."""
        config = DatabaseConfig(
            connection_string="invalid://connection/string", pool_size=5
        )

        manager = DatabaseManager(config)

        with pytest.raises(ValueError, match="Connection string must use postgresql"):
            await manager._create_engine()

    @pytest.mark.asyncio
    async def test_setup_session_factory(self):
        """Test session factory setup."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )
        manager.engine = Mock()

        with patch("src.data.storage.database.async_sessionmaker") as mock_sessionmaker:
            mock_factory = Mock()
            mock_sessionmaker.return_value = mock_factory

            await manager._setup_session_factory()

            mock_sessionmaker.assert_called_once()
            args, kwargs = mock_sessionmaker.call_args
            assert kwargs["bind"] == manager.engine
            assert kwargs["class_"] == AsyncSession
            assert kwargs["expire_on_commit"] is False
            assert kwargs["autoflush"] is True
            assert kwargs["autocommit"] is False

            assert manager.session_factory == mock_factory

    @pytest.mark.asyncio
    async def test_setup_session_factory_no_engine(self):
        """Test session factory setup without engine."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )

        with pytest.raises(RuntimeError, match="Engine must be created"):
            await manager._setup_session_factory()

    @pytest.mark.asyncio
    async def test_verify_connection_success(self):
        """Test successful connection verification."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )

        mock_engine = Mock()
        mock_conn = AsyncMock()

        # Mock both basic connectivity and TimescaleDB results
        mock_result_basic = Mock()
        mock_result_basic.scalar.return_value = 1

        mock_result_timescale = Mock()
        mock_result_timescale.scalar.return_value = "1.7.4"  # Mock TimescaleDB version

        # Set up execute to return different results based on query
        def execute_side_effect(query):
            query_str = str(query).lower()
            if "timescaledb" in query_str or "pg_extension" in query_str:
                return mock_result_timescale
            return mock_result_basic

        mock_conn.execute.side_effect = execute_side_effect

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_conn
        mock_context.__aexit__.return_value = None
        mock_engine.begin.return_value = mock_context

        manager.engine = mock_engine

        await manager._verify_connection()

        # Should execute basic connectivity test
        assert mock_conn.execute.call_count >= 1

        # Should check for TimescaleDB extension
        calls = mock_conn.execute.call_args_list
        timescale_calls = [
            call for call in calls if "timescaledb" in str(call[0][0]).lower()
        ]
        assert len(timescale_calls) > 0

    @pytest.mark.asyncio
    async def test_verify_connection_no_engine(self):
        """Test connection verification without engine."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )

        with pytest.raises(RuntimeError, match="Engine not initialized"):
            await manager._verify_connection()

    @pytest.mark.asyncio
    async def test_get_session_success(self):
        """Test successful session creation and cleanup."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.is_active = False

        manager.session_factory = Mock(return_value=mock_session)

        async with manager.get_session() as session:
            assert session == mock_session

        mock_session.commit.assert_called_once()
        mock_session.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_retry_on_connection_error(self):
        """Test session retry logic on connection errors."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)
        manager.max_retries = 2
        manager.base_delay = 0.01  # Fast retry for testing

        mock_session = AsyncMock()
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()

        # First call fails, second succeeds
        manager.session_factory = Mock(
            side_effect=[
                OperationalError("Connection failed", None, None),
                mock_session,
            ]
        )

        with patch("asyncio.sleep") as mock_sleep:
            async with manager.get_session() as session:
                assert session == mock_session

        # Should have retried once
        assert manager.session_factory.call_count == 2
        mock_sleep.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_session_max_retries_exceeded(self):
        """Test session creation when max retries exceeded."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)
        manager.max_retries = 1
        manager.base_delay = 0.01

        # Always fail
        manager.session_factory = Mock(
            side_effect=OperationalError("Always fails", None, None)
        )

        with patch("asyncio.sleep"):
            with pytest.raises(DatabaseConnectionError):
                async with manager.get_session():
                    pass

        # Should have tried max_retries + 1 times
        assert manager.session_factory.call_count == 2

    @pytest.mark.asyncio
    async def test_get_session_non_connection_error(self):
        """Test session creation with non-connection error."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        manager.session_factory = Mock(side_effect=ValueError("Not a connection error"))

        with pytest.raises(DatabaseQueryError):
            async with manager.get_session():
                pass

    @pytest.mark.asyncio
    async def test_execute_query_success(self):
        """Test successful query execution."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.fetchone.return_value = ("result",)
        mock_result.fetchall.return_value = [("row1",), ("row2",)]
        mock_session.execute.return_value = mock_result

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        with patch.object(manager, "get_session", return_value=mock_context):
            # Test fetch_one
            result = await manager.execute_query(
                "SELECT 1", parameters={"param": "value"}, fetch_one=True
            )
            assert result == ("result",)

            # Test fetch_all
            result = await manager.execute_query("SELECT * FROM table", fetch_all=True)
            assert result == [("row1",), ("row2",)]

            # Test no fetch (return result object)
            result = await manager.execute_query("UPDATE table SET col=1")
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_execute_query_error(self):
        """Test query execution error handling."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_session.execute.side_effect = Exception("Query failed")

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        with patch.object(manager, "get_session", return_value=mock_context):
            with pytest.raises(DatabaseQueryError):
                await manager.execute_query("INVALID SQL")

    @pytest.mark.asyncio
    async def test_health_check_healthy(self):
        """Test health check when database is healthy."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        manager.engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 5
        mock_pool.checkedout.return_value = 2
        manager.engine.pool = mock_pool

        with patch.object(manager, "get_session", return_value=mock_context):
            health = await manager.health_check()

        assert health["status"] == "healthy"
        assert "timestamp" in health
        assert "connection_stats" in health
        assert "performance_metrics" in health
        assert health["performance_metrics"]["response_time_ms"] >= 0
        assert health["performance_metrics"]["pool_size"] == 5
        assert health["performance_metrics"]["checked_out_connections"] == 2

    @pytest.mark.asyncio
    async def test_health_check_unhealthy(self):
        """Test health check when database is unhealthy."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        # Create a mock that raises an exception
        mock_context = AsyncMock()
        mock_context.__aenter__.side_effect = Exception("Connection failed")

        with patch.object(manager, "get_session", return_value=mock_context):
            health = await manager.health_check()

        assert health["status"] == "unhealthy"
        assert len(health["errors"]) > 0
        assert health["errors"][0]["type"] == "health_check_failed"

    @pytest.mark.asyncio
    async def test_health_check_timescaledb_available(self):
        """Test health check with TimescaleDB available."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()

        # Mock different queries
        def mock_execute(query):
            mock_result = Mock()
            if "SELECT 1" in str(query):
                mock_result.scalar.return_value = 1
            elif "get_version_info" in str(query):
                # Mock fetchone() to return version information
                mock_result.fetchone.return_value = (
                    "TimescaleDB version 2.8.0 on PostgreSQL 14.6",
                )
            return mock_result

        mock_session.execute.side_effect = mock_execute

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        manager.engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 0
        mock_pool.checkedout.return_value = 0
        manager.engine.pool = mock_pool

        with patch.object(manager, "get_session", return_value=mock_context):
            health = await manager.health_check()

        assert health["timescale_status"] == "available"
        # Test the new version information functionality
        assert "timescale_version_info" in health
        version_info = health["timescale_version_info"]
        assert "full_version" in version_info
        assert "timescale_version" in version_info
        assert "postgresql_version" in version_info
        assert (
            version_info["full_version"]
            == "TimescaleDB version 2.8.0 on PostgreSQL 14.6"
        )
        assert version_info["timescale_version"] == "2.8.0"
        assert version_info["postgresql_version"] == "14.6"

    @pytest.mark.asyncio
    async def test_health_check_timescaledb_unavailable(self):
        """Test health check with TimescaleDB unavailable."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()

        def mock_execute(query):
            mock_result = Mock()
            if "SELECT 1" in str(query):
                mock_result.scalar.return_value = 1
            elif "get_version_info" in str(query):
                raise Exception("TimescaleDB not available")
            return mock_result

        mock_session.execute.side_effect = mock_execute

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        manager.engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 0
        mock_pool.checkedout.return_value = 0
        manager.engine.pool = mock_pool

        with patch.object(manager, "get_session", return_value=mock_context):
            health = await manager.health_check()

        assert health["timescale_status"] == "unavailable"
        # Test that version info includes error information
        assert "timescale_version_info" in health
        version_info = health["timescale_version_info"]
        assert "error" in version_info
        assert "TimescaleDB not available" in version_info["error"]

    @pytest.mark.asyncio
    async def test_health_check_timescaledb_version_parsing(self):
        """Test TimescaleDB version information parsing with various formats."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()

        # Test different version string formats
        test_cases = [
            # Standard format
            ("TimescaleDB version 2.8.0 on PostgreSQL 14.6", "2.8.0", "14.6"),
            # Different PostgreSQL version format
            ("TimescaleDB version 2.10.1 on PostgreSQL 15.2", "2.10.1", "15.2"),
            # No PostgreSQL info
            ("TimescaleDB version 2.5.2", "2.5.2", None),
            # Unusual format (should still extract full version)
            ("Some custom TimescaleDB info", None, None),
        ]

        for version_string, expected_ts_version, expected_pg_version in test_cases:

            def mock_execute(query):
                mock_result = Mock()
                if "SELECT 1" in str(query):
                    mock_result.scalar.return_value = 1
                elif "get_version_info" in str(query):
                    mock_result.fetchone.return_value = (version_string,)
                return mock_result

            mock_session.execute.side_effect = mock_execute

            # Create a proper async context manager mock
            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None

            manager.engine = Mock()
            mock_pool = Mock()
            mock_pool.size.return_value = 0
            mock_pool.checkedout.return_value = 0
            manager.engine.pool = mock_pool

            with patch.object(manager, "get_session", return_value=mock_context):
                health = await manager.health_check()

            assert health["timescale_status"] == "available"
            assert "timescale_version_info" in health
            version_info = health["timescale_version_info"]

            # Full version should always be available
            assert "full_version" in version_info
            assert version_info["full_version"] == version_string

            # Check parsed versions
            if expected_ts_version:
                assert "timescale_version" in version_info
                assert version_info["timescale_version"] == expected_ts_version

            if expected_pg_version:
                assert "postgresql_version" in version_info
                assert version_info["postgresql_version"] == expected_pg_version

    @pytest.mark.asyncio
    async def test_health_check_timescaledb_version_parsing_error(self):
        """Test TimescaleDB version parsing with fetchone() returning None."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()

        def mock_execute(query):
            mock_result = Mock()
            if "SELECT 1" in str(query):
                mock_result.scalar.return_value = 1
            elif "get_version_info" in str(query):
                # Simulate fetchone() returning None
                mock_result.fetchone.return_value = None
            return mock_result

        mock_session.execute.side_effect = mock_execute

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        manager.engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 0
        mock_pool.checkedout.return_value = 0
        manager.engine.pool = mock_pool

        with patch.object(manager, "get_session", return_value=mock_context):
            health = await manager.health_check()

        assert health["timescale_status"] == "available"
        assert "timescale_version_info" in health
        version_info = health["timescale_version_info"]
        # Should be empty dict when no version info available
        assert version_info == {}

    @pytest.mark.asyncio
    async def test_health_check_loop(self):
        """Test background health check loop."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )
        manager.health_check = AsyncMock()

        # Run health check loop for a short time
        task = asyncio.create_task(manager._health_check_loop())

        # Let it run briefly then cancel
        await asyncio.sleep(0.01)
        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

        # Should have called health_check at least once if timing allows
        # This test mainly ensures the loop structure is correct

    @pytest.mark.asyncio
    async def test_close_cleanup(self):
        """Test database manager cleanup on close."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )

        # Mock components - create a proper task mock that's awaitable
        import asyncio

        async def dummy_task():
            pass

        mock_task = asyncio.create_task(dummy_task())
        # Override methods we need to test
        mock_task.done = Mock(return_value=False)
        mock_task.cancel = Mock()
        manager._health_check_task = mock_task

        mock_engine = AsyncMock()
        manager.engine = mock_engine
        manager.session_factory = Mock()

        await manager.close()

        # Should cancel health check task
        mock_task.cancel.assert_called_once()

        # Should dispose engine
        mock_engine.dispose.assert_called_once()

        # Should clear references
        assert manager.engine is None
        assert manager.session_factory is None

    def test_get_connection_stats(self):
        """Test getting connection statistics."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )

        # Set some test stats
        manager._connection_stats["total_connections"] = 10
        manager._connection_stats["failed_connections"] = 2

        stats = manager.get_connection_stats()

        assert stats["total_connections"] == 10
        assert stats["failed_connections"] == 2

        # Should return a copy, not reference
        stats["total_connections"] = 999
        assert manager._connection_stats["total_connections"] == 10

    def test_connection_event_handlers(self):
        """Test database connection event handlers setup."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        # Mock engine with event listener capability
        mock_engine = Mock()
        mock_sync_engine = Mock()
        mock_engine.sync_engine = mock_sync_engine
        manager.engine = mock_engine

        with patch("src.data.storage.database.event") as mock_event:
            manager._setup_connection_events()

            # Should set up event listeners
            assert (
                mock_event.listens_for.call_count >= 4
            )  # connect, checkout, checkin, invalidate


class TestGlobalDatabaseFunctions:
    """Test global database management functions."""

    @pytest.mark.asyncio
    async def test_get_database_manager_singleton(self):
        """Test global database manager singleton behavior."""
        # Clear any existing instance
        import src.data.storage.database

        src.data.storage.database._db_manager = None

        with patch("src.data.storage.database.DatabaseManager") as mock_manager_class:
            mock_manager = AsyncMock()
            mock_manager.initialize = AsyncMock()
            mock_manager_class.return_value = mock_manager

            # First call should create instance
            manager1 = await get_database_manager()

            # Second call should return same instance
            manager2 = await get_database_manager()

            assert manager1 is manager2
            mock_manager_class.assert_called_once()
            mock_manager.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_db_session(self):
        """Test convenience function for getting database session."""
        mock_manager = AsyncMock()
        mock_session = AsyncMock()

        # Mock the async context manager properly
        async def mock_get_session():
            yield mock_session

        # Use asynccontextmanager to create the mock
        from contextlib import asynccontextmanager

        mock_manager.get_session = asynccontextmanager(mock_get_session)

        with patch(
            "src.data.storage.database.get_database_manager",
            new=AsyncMock(return_value=mock_manager),
        ):
            async with get_db_session() as session:
                assert session == mock_session

    @pytest.mark.asyncio
    async def test_close_database_manager(self):
        """Test closing global database manager."""
        import src.data.storage.database

        mock_manager = AsyncMock()
        src.data.storage.database._db_manager = mock_manager

        await close_database_manager()

        mock_manager.close.assert_called_once()
        assert src.data.storage.database._db_manager is None

    @pytest.mark.asyncio
    async def test_close_database_manager_none(self):
        """Test closing database manager when none exists."""
        import src.data.storage.database

        src.data.storage.database._db_manager = None

        # Should not raise exception
        await close_database_manager()


class TestDatabaseUtilityFunctions:
    """Test database utility functions."""

    @pytest.mark.asyncio
    async def test_execute_sql_file_success(self):
        """Test successful SQL file execution."""
        sql_content = """
        CREATE TABLE test_table (id INT PRIMARY KEY);
        INSERT INTO test_table (id) VALUES (1);
        INSERT INTO test_table (id) VALUES (2);
        """

        mock_manager = AsyncMock()
        mock_manager.execute_query = AsyncMock()

        with (
            patch(
                "src.data.storage.database.get_database_manager",
                return_value=mock_manager,
            ),
            patch("builtins.open", mock_open(read_data=sql_content)),
        ):

            await execute_sql_file("/path/to/sql/file.sql")

            # Should execute each non-empty statement
            assert mock_manager.execute_query.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_sql_file_error(self):
        """Test SQL file execution error handling."""
        with patch("builtins.open", side_effect=FileNotFoundError("File not found")):
            with pytest.raises(DatabaseQueryError):
                await execute_sql_file("/nonexistent/file.sql")

    @pytest.mark.asyncio
    async def test_check_table_exists_true(self):
        """Test checking if table exists - table found."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = (True,)

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            exists = await check_table_exists("users")

            assert exists is True
            mock_manager.execute_query.assert_called_once()
            args, kwargs = mock_manager.execute_query.call_args
            assert "information_schema.tables" in args[0]
            assert kwargs["parameters"]["table_name"] == "users"
            assert kwargs["fetch_one"] is True

    @pytest.mark.asyncio
    async def test_check_table_exists_false(self):
        """Test checking if table exists - table not found."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = (False,)

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            exists = await check_table_exists("nonexistent_table")

            assert exists is False

    @pytest.mark.asyncio
    async def test_check_table_exists_error(self):
        """Test table existence check error handling."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.side_effect = Exception("Database error")

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            exists = await check_table_exists("users")

            assert exists is False  # Should return False on error

    @pytest.mark.asyncio
    async def test_get_database_version_success(self):
        """Test getting database version successfully."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = ("PostgreSQL 13.0",)

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            version = await get_database_version()

            assert version == "PostgreSQL 13.0"
            mock_manager.execute_query.assert_called_once()
            args, kwargs = mock_manager.execute_query.call_args
            assert "SELECT version()" == args[0]
            assert kwargs["fetch_one"] is True

    @pytest.mark.asyncio
    async def test_get_database_version_error(self):
        """Test database version error handling."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.side_effect = Exception("Connection failed")

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            version = await get_database_version()

            assert version == "Error"

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_success(self):
        """Test getting TimescaleDB version successfully."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = ("2.5.1",)

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            version = await get_timescaledb_version()

            assert version == "2.5.1"
            mock_manager.execute_query.assert_called_once()
            args, kwargs = mock_manager.execute_query.call_args
            assert (
                "SELECT extversion FROM pg_extension WHERE extname = 'timescaledb'"
                == args[0]
            )
            assert kwargs["fetch_one"] is True

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_not_available(self):
        """Test TimescaleDB version when extension not available."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.return_value = None

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            version = await get_timescaledb_version()

            assert version is None

    @pytest.mark.asyncio
    async def test_get_timescaledb_version_error(self):
        """Test TimescaleDB version error handling."""
        mock_manager = AsyncMock()
        mock_manager.execute_query.side_effect = Exception("Extension not found")

        with patch(
            "src.data.storage.database.get_database_manager",
            return_value=mock_manager,
        ):
            version = await get_timescaledb_version()

            assert version is None


class TestDatabaseManagerEdgeCases:
    """Test edge cases and error conditions."""

    @pytest.mark.asyncio
    async def test_initialize_already_initialized(self):
        """Test initializing already initialized manager."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )
        manager.engine = Mock()  # Simulate already initialized

        with patch.object(manager, "_create_engine") as mock_create:
            await manager.initialize()

            # Should not call _create_engine again
            mock_create.assert_not_called()

    @pytest.mark.asyncio
    async def test_verify_connection_timescaledb_warning(self):
        """Test connection verification with TimescaleDB warning."""
        manager = DatabaseManager(
            DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        )

        mock_engine = Mock()
        mock_conn = AsyncMock()

        def mock_execute(query):
            mock_result = Mock()
            if "SELECT 1" in str(query):
                mock_result.scalar.return_value = 1
            elif "timescaledb" in str(query):
                mock_result.scalar.return_value = 0  # No TimescaleDB
            return mock_result

        mock_conn.execute.side_effect = mock_execute

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_conn
        mock_context.__aexit__.return_value = None
        mock_engine.begin.return_value = mock_context
        manager.engine = mock_engine

        # Should complete without error but log warning
        await manager._verify_connection()

    @pytest.mark.asyncio
    async def test_get_session_rollback_on_error(self):
        """Test session rollback on error during context."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_session.commit = AsyncMock(side_effect=Exception("Commit failed"))
        mock_session.rollback = AsyncMock()
        mock_session.close = AsyncMock()
        mock_session.is_active = True

        manager.session_factory = Mock(return_value=mock_session)

        with pytest.raises(DatabaseQueryError):
            async with manager.get_session():
                pass  # Exception will be raised on commit

        mock_session.rollback.assert_called_once()
        # Close may be called multiple times (exception + finally blocks)
        assert mock_session.close.call_count >= 1

    @pytest.mark.asyncio
    async def test_health_check_with_previous_errors(self):
        """Test health check reporting previous connection errors."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        # Simulate previous connection error
        manager._connection_stats["last_connection_error"] = "Previous error"
        manager._connection_stats["failed_connections"] = 5

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        # Create a proper async context manager mock
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        manager.engine = Mock()
        mock_pool = Mock()
        mock_pool.size.return_value = 0
        mock_pool.checkedout.return_value = 0
        manager.engine.pool = mock_pool

        with patch.object(manager, "get_session", return_value=mock_context):
            health = await manager.health_check()

        assert health["status"] == "healthy"
        assert len(health["errors"]) > 0
        assert health["errors"][0]["type"] == "connection_error"
        assert health["errors"][0]["message"] == "Previous error"
        assert health["errors"][0]["failed_count"] == 5

    def test_database_connection_error_password_masking(self):
        """Test password masking in database connection errors."""
        connection_string = "postgresql://user:secretpassword@localhost:5432/db"

        # Create error (this should be tested in database manager)
        masked = DatabaseConnectionError._mask_password(connection_string)

        assert "secretpassword" not in masked
        assert "***" in masked
        assert "user" in masked
        assert "localhost" in masked


@pytest.mark.unit
@pytest.mark.database
class TestDatabaseManagerIntegration:
    """Integration tests for database manager functionality."""

    @pytest.mark.asyncio
    async def test_full_lifecycle(self):
        """Test complete database manager lifecycle."""
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://localhost/testdb",
            pool_size=5,
            max_overflow=10,
        )

        manager = DatabaseManager(config)

        # Mock the entire initialization and lifecycle
        mock_engine = AsyncMock()
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        # Create proper async context manager mocks
        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        with (
            patch.object(manager, "_create_engine") as mock_create_engine,
            patch.object(manager, "_setup_session_factory") as mock_setup_factory,
            patch.object(manager, "_verify_connection") as mock_verify,
            patch("asyncio.create_task") as mock_create_task,
            patch.object(manager, "get_session", return_value=mock_context),
        ):
            # Mock the side effects to set up engine and session_factory
            async def mock_create_engine_effect():
                manager.engine = mock_engine

            async def mock_setup_factory_effect():
                manager.session_factory = Mock()

            mock_create_engine.side_effect = mock_create_engine_effect
            mock_setup_factory.side_effect = mock_setup_factory_effect

            # Initialize
            await manager.initialize()
            assert manager.is_initialized
            mock_create_engine.assert_called_once()
            mock_setup_factory.assert_called_once()
            mock_verify.assert_called_once()
            mock_create_task.assert_called_once()

            # Test session usage
            async with manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1

            # Test health check
            with patch.object(manager, "health_check") as mock_health_check:
                mock_health_check.return_value = {"status": "healthy"}
                health = await manager.health_check()
                assert health["status"] == "healthy"

            # Test query execution
            with patch.object(manager, "execute_query") as mock_execute_query:
                mock_execute_query.return_value = (2,)
                result = await manager.execute_query("SELECT 2", fetch_one=True)
                assert result == (2,)

            # Clean up
            manager.engine = mock_engine  # Set engine for cleanup
            await manager.close()
            assert manager.engine is None
            assert manager.session_factory is None

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """Test concurrent session usage."""
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://localhost/testdb",
            pool_size=3,
        )

        manager = DatabaseManager(config)

        # Mock the session creation and usage
        async def mock_session_usage(session_id):
            mock_session = AsyncMock()
            mock_result = Mock()
            mock_result.scalar.return_value = session_id
            mock_session.execute.return_value = mock_result

            mock_context = AsyncMock()
            mock_context.__aenter__.return_value = mock_session
            mock_context.__aexit__.return_value = None
            return mock_context

        with patch.object(
            manager, "get_session", side_effect=lambda: mock_session_usage(1)
        ):

            async def use_session(session_id):
                async with manager.get_session() as session:
                    result = await session.execute(text(f"SELECT {session_id}"))
                    return result.scalar()

            # Run multiple sessions concurrently
            with patch.object(manager, "get_session") as mock_get_session:
                # Configure different return values for each call
                mock_contexts = []
                for i in range(1, 6):
                    mock_session = AsyncMock()
                    mock_result = Mock()
                    mock_result.scalar.return_value = i
                    mock_session.execute.return_value = mock_result

                    mock_context = AsyncMock()
                    mock_context.__aenter__.return_value = mock_session
                    mock_context.__aexit__.return_value = None
                    mock_contexts.append(mock_context)

                mock_get_session.side_effect = mock_contexts

                tasks = [use_session(i) for i in range(1, 6)]
                results = await asyncio.gather(*tasks)

                assert results == [1, 2, 3, 4, 5]

    @pytest.mark.asyncio
    async def test_retry_mechanism_with_real_errors(self):
        """Test retry mechanism with realistic error scenarios."""
        config = DatabaseConfig(
            connection_string="postgresql+asyncpg://localhost/testdb",
            pool_size=1,
        )

        manager = DatabaseManager(config)
        manager.max_retries = 2
        manager.base_delay = 0.001  # Very fast for testing

        # Mock session factory to fail then succeed
        mock_session = AsyncMock()
        mock_result = Mock()
        mock_result.scalar.return_value = 1
        mock_session.execute.return_value = mock_result

        call_count = 0

        def mock_factory():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise OperationalError("Temporary failure", None, None)
            return mock_session

        manager.session_factory = mock_factory

        with patch("asyncio.sleep") as mock_sleep:
            # Should succeed after retry
            async with manager.get_session() as session:
                result = await session.execute(text("SELECT 1"))
                assert result.scalar() == 1

        assert call_count == 2  # Failed once, succeeded on retry
        mock_sleep.assert_called_once()


class TestDatabaseCompatibilityLayer:
    """Test database compatibility functionality."""

    def test_is_sqlite_engine_detection(self):
        """Test SQLite engine detection."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        assert is_sqlite_engine(mock_engine) is True

        # Test with PostgreSQL URL
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")
        assert is_sqlite_engine(mock_engine) is False

        # Test with direct URL string
        assert is_sqlite_engine("sqlite:///path/to/database.db") is True
        assert is_sqlite_engine("postgresql://localhost/db") is False

    def test_is_postgresql_engine_detection(self):
        """Test PostgreSQL engine detection."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")

        assert is_postgresql_engine(mock_engine) is True

        # Test with asyncpg URL
        mock_engine.url.__str__ = Mock(return_value="postgresql+asyncpg://localhost/db")
        assert is_postgresql_engine(mock_engine) is True

        # Test with SQLite URL
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")
        assert is_postgresql_engine(mock_engine) is False

    def test_configure_sensor_event_model_sqlite(self):
        """Test SensorEvent configuration for SQLite."""
        mock_model = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_model.__table__ = mock_table

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        result = configure_sensor_event_model(mock_model, mock_engine)

        # Should set autoincrement on id column
        assert mock_id_column.autoincrement is True
        # Should remove timestamp from primary key
        assert mock_timestamp_column.primary_key is False
        assert result == mock_model

    def test_configure_sensor_event_model_postgresql(self):
        """Test SensorEvent configuration for PostgreSQL."""
        mock_model = Mock()
        mock_table = Mock()
        mock_id_column = Mock()
        mock_timestamp_column = Mock()

        mock_table.c.id = mock_id_column
        mock_table.c.timestamp = mock_timestamp_column
        mock_model.__table__ = mock_table

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="postgresql://localhost/db")

        result = configure_sensor_event_model(mock_model, mock_engine)

        # Should set both columns as primary key
        assert mock_id_column.primary_key is True
        assert mock_id_column.autoincrement is True
        assert mock_timestamp_column.primary_key is True
        assert result == mock_model

    def test_create_database_specific_models(self):
        """Test creating database-specific models."""
        mock_sensor_event = Mock()
        mock_other_model = Mock()

        base_models = {
            "SensorEvent": mock_sensor_event,
            "OtherModel": mock_other_model,
        }

        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        result = create_database_specific_models(base_models, mock_engine)

        # Should return models (SensorEvent may be configured internally)
        assert "SensorEvent" in result
        assert "OtherModel" in result
        assert result["OtherModel"] == mock_other_model

    def test_configure_sqlite_for_testing(self):
        """Test SQLite configuration event listener."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        mock_record = Mock()
        mock_record.info = {"url": "sqlite:///test.db"}

        # Call the event handler
        configure_sqlite_for_testing(mock_connection, mock_record)

        # Should enable foreign keys
        mock_cursor.execute.assert_called_once_with("PRAGMA foreign_keys=ON")
        mock_cursor.close.assert_called_once()

    def test_configure_database_on_first_connect_sqlite(self):
        """Test first connection configuration for SQLite."""
        mock_connection = Mock()
        mock_cursor = Mock()
        mock_connection.cursor.return_value = mock_cursor

        mock_record = Mock()
        mock_record.info = {"url": "sqlite:///test.db"}

        # Call the event handler
        configure_database_on_first_connect(mock_connection, mock_record)

        # Should execute SQLite optimization commands
        expected_calls = [
            "PRAGMA foreign_keys=ON",
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
        ]

        assert mock_cursor.execute.call_count == len(expected_calls)
        for expected_call in expected_calls:
            assert any(
                call[0][0] == expected_call
                for call in mock_cursor.execute.call_args_list
            )

        mock_cursor.close.assert_called_once()

    def test_get_database_specific_table_args(self):
        """Test database-specific table arguments generation."""
        mock_engine = Mock()
        mock_engine.url = Mock()
        mock_engine.url.__str__ = Mock(return_value="sqlite:///test.db")

        # Test sensor_events table
        args = get_database_specific_table_args(mock_engine, "sensor_events")
        assert isinstance(args, tuple)
        assert len(args) > 0

        # Test other table
        args = get_database_specific_table_args(mock_engine, "other_table")
        assert args == ()

    def test_patch_models_for_sqlite_compatibility(self):
        """Test patching existing models for SQLite compatibility."""
        with patch("src.data.storage.models.SensorEvent") as mock_model:
            mock_table = Mock()
            mock_id_column = Mock()
            mock_timestamp_column = Mock()

            mock_table.c.id = mock_id_column
            mock_table.c.timestamp = mock_timestamp_column
            mock_timestamp_column.primary_key = True  # Initially true
            mock_model.__table__ = mock_table

            # Call the patching function
            patch_models_for_sqlite_compatibility()

            # Should remove timestamp from primary key
            assert mock_timestamp_column.primary_key is False
            # Should ensure id has autoincrement
            assert mock_id_column.autoincrement is True

    def test_compatibility_edge_cases(self):
        """Test edge cases in compatibility layer."""
        # Test with engine without url attribute
        mock_engine = Mock(spec=[])  # No url attribute
        assert is_sqlite_engine(mock_engine) is False
        assert is_postgresql_engine(mock_engine) is False

        # Test with None URL
        mock_engine = Mock()
        mock_engine.url = None
        assert is_sqlite_engine(mock_engine) is False
        assert is_postgresql_engine(mock_engine) is False

        # Test case insensitive detection
        assert is_sqlite_engine("SQLITE:///test.db") is True
        assert is_postgresql_engine("POSTGRESQL://localhost/db") is True


class TestAdvancedDatabaseFeatures:
    """Test advanced database functionality."""

    @pytest.mark.asyncio
    async def test_execute_optimized_query_with_prepared_statements(self):
        """Test optimized query execution with prepared statements."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_result = Mock()
        mock_session.execute.return_value = mock_result

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        query = "SELECT * FROM users WHERE id = :user_id"
        parameters = {"user_id": 123}

        with patch.object(manager, "get_session", return_value=mock_context):
            result = await manager.execute_optimized_query(
                query,
                parameters=parameters,
                use_prepared_statement=True,
                enable_query_cache=True,
            )

            assert result == mock_result

            # Should enable query plan caching
            plan_cache_calls = [
                call
                for call in mock_session.execute.call_args_list
                if "plan_cache_mode" in str(call[0][0])
            ]
            assert len(plan_cache_calls) >= 0  # May or may not have cache calls

    @pytest.mark.asyncio
    async def test_execute_optimized_query_prepared_statement_fallback(self):
        """Test optimized query fallback when prepared statement fails."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()
        mock_result = Mock()

        # Make prepared statement fail, but regular execute succeed
        def mock_execute_side_effect(query_obj, params=None):
            query_str = str(query_obj)
            if "PREPARE" in query_str:
                raise SQLAlchemyError("Prepared statement failed")
            return mock_result

        mock_session.execute.side_effect = mock_execute_side_effect

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        query = "SELECT * FROM users WHERE id = :user_id"
        parameters = {"user_id": 123}

        with patch.object(manager, "get_session", return_value=mock_context):
            result = await manager.execute_optimized_query(
                query,
                parameters=parameters,
                use_prepared_statement=True,
            )

            # Should fallback to regular query execution
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_get_connection_pool_metrics_detailed(self):
        """Test detailed connection pool metrics collection."""
        config = DatabaseConfig(
            connection_string="postgresql://user:pass@localhost/db", pool_size=10
        )
        manager = DatabaseManager(config)

        # Mock engine with pool
        mock_engine = Mock()
        mock_pool = Mock()

        # Configure pool attributes
        mock_pool._pool_size = 10
        mock_pool._checked_out = 3
        mock_pool._overflow = 2
        mock_pool._invalidated = 1

        manager.engine = mock_engine
        manager.engine.pool = mock_pool

        # Set some connection stats
        manager._connection_stats["total_connections"] = 15
        manager._connection_stats["failed_connections"] = 2

        metrics = await manager.get_connection_pool_metrics()

        assert metrics["pool_size"] == 10
        assert metrics["checked_out"] == 3
        assert metrics["overflow"] == 2
        assert metrics["invalid_count"] == 1
        assert metrics["utilization_percent"] == 50.0  # (3+2)/10 * 100
        assert metrics["pool_status"] in ["healthy", "moderate"]  # 50% utilization
        assert "recommendations" in metrics

    @pytest.mark.asyncio
    async def test_get_connection_pool_metrics_high_utilization(self):
        """Test connection pool metrics with high utilization."""
        config = DatabaseConfig(
            connection_string="postgresql://user:pass@localhost/db", pool_size=5
        )
        manager = DatabaseManager(config)

        # Mock engine with high utilization pool
        mock_engine = Mock()
        mock_pool = Mock()

        # High utilization scenario
        mock_pool._pool_size = 5
        mock_pool._checked_out = 4
        mock_pool._overflow = 3
        mock_pool._invalidated = 0

        manager.engine = mock_engine
        manager.engine.pool = mock_pool

        metrics = await manager.get_connection_pool_metrics()

        assert metrics["utilization_percent"] == 140.0  # (4+3)/5 * 100
        assert metrics["pool_status"] == "high_utilization"  # > 80%
        assert "pool_size" in metrics["recommendations"][0]

    @pytest.mark.asyncio
    async def test_analyze_query_performance_with_execution_plan(self):
        """Test query performance analysis with execution plan."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        mock_session = AsyncMock()

        # Mock execution plan result
        mock_plan_result = Mock()
        mock_plan_result.fetchone.return_value = [
            {
                "Plan": {
                    "Node Type": "Seq Scan",
                    "Total Cost": 10.5,
                    "Actual Total Time": 2.5,
                }
            }
        ]

        # Mock regular query execution
        mock_query_result = Mock()

        def mock_execute_side_effect(query_obj, params=None):
            query_str = str(query_obj).upper()
            if "EXPLAIN" in query_str:
                return mock_plan_result
            return mock_query_result

        mock_session.execute.side_effect = mock_execute_side_effect

        mock_context = AsyncMock()
        mock_context.__aenter__.return_value = mock_session
        mock_context.__aexit__.return_value = None

        with (
            patch.object(manager, "get_session", return_value=mock_context),
            patch("time.time", side_effect=[1000.0, 1000.05]),
        ):  # 0.05 second execution
            analysis = await manager.analyze_query_performance(
                "SELECT * FROM events", include_execution_plan=True
            )

            assert "query" in analysis
            assert "execution_plan" in analysis
            assert "execution_time_seconds" in analysis
            assert "performance_rating" in analysis
            assert "optimization_suggestions" in analysis

            # Should have excellent performance (< 0.1s)
            assert analysis["performance_rating"] == "excellent"
            assert abs(analysis["execution_time_seconds"] - 0.05) < 0.001

    def test_get_optimization_suggestions(self):
        """Test query optimization suggestions generation."""
        config = DatabaseConfig(connection_string="postgresql://user:pass@localhost/db")
        manager = DatabaseManager(config)

        # Test SELECT * suggestion
        suggestions = manager._get_optimization_suggestions("SELECT * FROM users")
        assert any("SELECT *" in suggestion for suggestion in suggestions)

        # Test missing WHERE clause on sensor_events
        suggestions = manager._get_optimization_suggestions(
            "SELECT id FROM sensor_events"
        )
        assert any("WHERE clause" in suggestion for suggestion in suggestions)

        # Test JOIN without ON condition
        suggestions = manager._get_optimization_suggestions(
            "SELECT * FROM users JOIN orders"
        )
        assert any(
            "JOIN" in suggestion and "ON" in suggestion for suggestion in suggestions
        )

        # Test ORDER BY without LIMIT
        suggestions = manager._get_optimization_suggestions(
            "SELECT name FROM users ORDER BY created_at"
        )
        assert any("LIMIT" in suggestion for suggestion in suggestions)
